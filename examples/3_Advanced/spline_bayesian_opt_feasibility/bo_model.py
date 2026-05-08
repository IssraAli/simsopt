from typing import Any, Optional, Sequence, Union
from math import sqrt

import torch
import numpy as np
import pickle

from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.models.transforms import Standardize
from botorch.optim import optimize_acqf
from botorch.models import SingleTaskGP, ModelListGP
from botorch.sampling import SobolQMCNormalSampler
from botorch.acquisition.objective import MCAcquisitionObjective
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
from torch.quasirandom import SobolEngine

from gpytorch.constraints.constraints import Interval
from gpytorch.kernels import MaternKernel, RBFKernel
from gpytorch.priors.torch_priors import LogNormalPrior

from bo_utils import from_unit_cube

SQRT3 = sqrt(3)


class _ObjectiveSelector(MCAcquisitionObjective):
    """Selects the first output from a multi-output model for use as the objective."""
    def forward(self, samples, X=None):
        return samples[..., 0]


def get_covar_module_with_dim_scaled_prior_maxlengthscale_constrained(
    ard_num_dims: int,
    batch_shape: Optional[torch.Size] = None,
    use_rbf_kernel: bool = True,
    active_dims: Optional[Sequence[int]] = None,
) -> Union[MaternKernel, RBFKernel]:
    """Returns an RBF or Matern kernel with lengthscale prior and constraint."""
    base_class = RBFKernel if use_rbf_kernel else MaternKernel
    lengthscale_prior = LogNormalPrior(loc=0.5, scale=0.1 * SQRT3)
    base_kernel = base_class(
        ard_num_dims=ard_num_dims,
        batch_shape=batch_shape,
        lengthscale_prior=lengthscale_prior,
        lengthscale_constraint=Interval(
            1e-6, 5.0, transform=None, initial_value=lengthscale_prior.mode
        ),
        eps=1e-8,
        active_dims=active_dims,
    )
    return base_kernel


class GlobalOptimizer:
    def __init__(self, dof_list, lb: Any, ub: Any, target: callable = None):
        self.dof_list = dof_list
        self.lb = lb
        self.ub = ub
        self.target = target


class VanillaBO(GlobalOptimizer):
    def __init__(self, dof_list, lb, ub, X_history, y_history, spline_kwargs, target):
        super().__init__(dof_list, lb, ub, target)
        self.dims = len(lb)
        self.X_history = None   # (N, dims) all candidates in unit cube
        self.y_history = None   # (N, 1)    objective values (only meaningful for feasible)
        self.c_history = None   # (N, 1)    feasibility labels: 1.0=feasible, 0.0=infeasible
        self.device = "cpu"
        self.dtype = torch.double
        self.dof_list = dof_list
        self.spline_kwargs = spline_kwargs

    def _fitting_loop(self, batch_size):
        feasible_mask = self.c_history.squeeze() > 0.5
        n_feasible = int(feasible_mask.sum().item())
        n_total = len(self.c_history)
        print(f"Feasible: {n_feasible}/{n_total}")

        # Edge case: not enough feasible data to fit a GP
        if n_feasible < 2:
            print("Not enough feasible data, exploring randomly.")
            sobol = SobolEngine(dimension=self.dims, scramble=True)
            return None, sobol.draw(batch_size).to(dtype=self.dtype)

        X_feasible = self.X_history[feasible_mask]
        y_feasible = self.y_history[feasible_mask]
        best_f = y_feasible.max()

        print(f"Current best (feasible): {best_f}")
        print(f"at {from_unit_cube(X_feasible[y_feasible.squeeze().argmax()], self.lb, self.ub)}")

        # --- Objective GP (trained on feasible data only) ---
        obj_gp = SingleTaskGP(
            train_X=X_feasible,
            train_Y=y_feasible,
            outcome_transform=Standardize(m=1),
            covar_module=get_covar_module_with_dim_scaled_prior_maxlengthscale_constrained(
                ard_num_dims=self.dims, use_rbf_kernel=True
            ),
        )
        mll_obj = ExactMarginalLogLikelihood(obj_gp.likelihood, obj_gp)
        fit_gpytorch_mll(mll_obj)

        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([2048]))

        n_infeasible = n_total - n_feasible
        if n_infeasible > 0:
            # --- Feasibility GP (trained on ALL data, binary labels) ---
            feas_gp = SingleTaskGP(
                train_X=self.X_history,
                train_Y=self.c_history,
                outcome_transform=Standardize(m=1),
            )
            mll_feas = ExactMarginalLogLikelihood(feas_gp.likelihood, feas_gp)
            fit_gpytorch_mll(mll_feas)

            # Combined model: output 0 = objective, output 1 = feasibility
            model = ModelListGP(obj_gp, feas_gp)
            objective = _ObjectiveSelector()
            # Constraint: feasibility prediction >= 0.5 → constraint value >= 0
            constraints = [lambda Z: Z[..., 1] - 0.5]

            MC_LogEI = qLogExpectedImprovement(
                model=model,
                best_f=best_f,
                sampler=sampler,
                objective=objective,
                constraints=constraints,
                fat=False,
            )
        else:
            # All data is feasible — standard EI, no constraint needed
            MC_LogEI = qLogExpectedImprovement(
                obj_gp, best_f=best_f, sampler=sampler, fat=False
            )

        candidates, _ = optimize_acqf(
            acq_function=MC_LogEI,
            bounds=torch.tensor([[0.0] * self.dims, [1.0] * self.dims]),
            q=batch_size,
            num_restarts=128,
            raw_samples=1024,
        )
        return obj_gp, candidates

    def ask(self, batch_size) -> np.ndarray:
        obj_gp, candidates = self._fitting_loop(batch_size)
        if obj_gp is not None:
            print(f"Obj GP lengthscales: {obj_gp.covar_module.lengthscale.detach()}")
        return candidates

    def tell(self, X_new: np.ndarray, results: torch.Tensor, lb: np.ndarray, ub: np.ndarray):
        """
        results: (batch, 2) tensor — column 0 = objective, column 1 = feasibility
        """
        X_new = torch.tensor(X_new).reshape(-1, self.dims).to(torch.double)
        y_new = results[:, 0:1].to(torch.double)
        c_new = results[:, 1:2].to(torch.double)

        self.X_history = torch.cat((self.X_history, X_new), dim=0)
        self.y_history = torch.cat((self.y_history, y_new))
        self.c_history = torch.cat((self.c_history, c_new))
        self.lb = lb
        self.ub = ub

    def return_best(self):
        feasible_mask = self.c_history.squeeze() > 0.5
        if not feasible_mask.any():
            return None, None
        y_feas = self.y_history[feasible_mask]
        X_feas = self.X_history[feasible_mask]
        return y_feas.max(), X_feas[y_feas.squeeze().argmax()]

    def dump(self, fpath):
        pickle.dump(self, open(fpath, "wb"), pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, fpath):
        with open(fpath, "rb") as f:
            optimizer = pickle.load(f)
        return optimizer
