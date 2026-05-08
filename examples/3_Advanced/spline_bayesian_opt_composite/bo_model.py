"""
Composite Bayesian Optimization over the QA + iota residual vector.

Architecture:
- Multi-output `SingleTaskGP` over the 25-component residual vector defined in
  `test_target.target` (`Standardize(m=25)`, default Matérn-5/2 + dim-scaled
  LogNormal lengthscale prior — botorch default since v0.10).
- `GenericMCObjective` mapping samples → `-0.5 * Σᵢ rᵢ²`, recovering the
  legacy scalar BO objective exactly.
- `qLogNoisyExpectedImprovement` for the acquisition (the composite is a
  non-linear function of GP samples, so `best_f = max(y)` is not well-defined;
  qLogNEI MC-marginalizes the incumbent through the posterior).
- Failed evaluations are dropped from the GP training set (failure rate is
  empirically <10%, so a separate feasibility GP is overkill).
"""

import pickle
from typing import Any, Callable

import numpy as np
import torch

from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.acquisition.objective import GenericMCObjective
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms import Standardize
from botorch.optim import optimize_acqf
from botorch.sampling import SobolQMCNormalSampler
from gpytorch.constraints import GreaterThan
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.quasirandom import SobolEngine

from bo_utils import from_unit_cube


def _composite_objective_fn(samples: torch.Tensor, X=None) -> torch.Tensor:
    """f(x) = -0.5 * Σᵢ rᵢ(x)²  on samples of the multi-output posterior.

    `samples` shape: (..., q, m=25). Returns shape (..., q).
    Maximizing this scalar matches BO's prior `-0.5·Σ residuals²` exactly.
    """
    return -0.5 * (samples ** 2).sum(dim=-1)


COMPOSITE_OBJECTIVE = GenericMCObjective(_composite_objective_fn)


def y_to_scalar(Y: torch.Tensor) -> torch.Tensor:
    """Map the residual-vector history to the legacy scalar (for logging only)."""
    return -0.5 * (Y ** 2).sum(dim=-1, keepdim=True)


class CompositeBO:
    """
    BO over the 25-component residual vector.

    Stores all observed candidates (X) and residual vectors (Y), plus a
    boolean feasibility mask. The GP is fit on the feasible subset only.
    """

    def __init__(self, dof_list, lb, ub, spline_kwargs, target: Callable):
        self.dof_list = dof_list
        self.lb = lb
        self.ub = ub
        self.target = target
        self.spline_kwargs = spline_kwargs
        self.dims = len(lb)
        self.dtype = torch.double
        self.device = torch.device('cpu')

        self.X_history: torch.Tensor | None = None  # (N, dims) in unit cube
        self.Y_history: torch.Tensor | None = None  # (N, 25)   residual vectors
        self.feas_mask: torch.Tensor | None = None  # (N,)      bool

    # ------------------------------------------------------------------ history

    def push_history(self, X_new, Y_new, feas_new):
        """Append a batch to history. Accepts numpy/torch in any reasonable shape."""
        X_new = torch.as_tensor(np.asarray(X_new), dtype=self.dtype).reshape(-1, self.dims)
        Y_new = torch.as_tensor(np.asarray(Y_new), dtype=self.dtype).reshape(-1, Y_new.shape[-1])
        feas_new = torch.as_tensor(np.asarray(feas_new), dtype=torch.bool).reshape(-1)
        assert X_new.shape[0] == Y_new.shape[0] == feas_new.shape[0]

        if self.X_history is None:
            self.X_history = X_new
            self.Y_history = Y_new
            self.feas_mask = feas_new
        else:
            self.X_history = torch.cat([self.X_history, X_new], dim=0)
            self.Y_history = torch.cat([self.Y_history, Y_new], dim=0)
            self.feas_mask = torch.cat([self.feas_mask, feas_new], dim=0)

    # ------------------------------------------------------------------ fit/ask

    def _fit_gp(self):
        """Fit a multi-output GP on feasible data. Returns the fitted model."""
        X_feas = self.X_history[self.feas_mask]
        Y_feas = self.Y_history[self.feas_mask]

        # Per-output noise floor for Cholesky stability with a near-deterministic VMEC.
        likelihood = GaussianLikelihood(
            noise_constraint=GreaterThan(1e-6),
            batch_shape=torch.Size([Y_feas.shape[-1]]),
        )
        gp = SingleTaskGP(
            train_X=X_feas,
            train_Y=Y_feas,
            outcome_transform=Standardize(m=Y_feas.shape[-1]),
            likelihood=likelihood,
            # No covar_module override: use botorch default (Matérn-5/2 +
            # dim-scaled LogNormal lengthscale prior, no upper-bound cap).
        )
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        return gp

    def _fitting_loop(self, batch_size: int):
        n_feasible = int(self.feas_mask.sum().item()) if self.feas_mask is not None else 0
        n_total = len(self.feas_mask) if self.feas_mask is not None else 0
        print(f'Feasible: {n_feasible}/{n_total}')

        # Cold-start guard: Standardize(m=25) needs at least a few samples per
        # output for a non-degenerate std estimate. Until then, just explore.
        if n_feasible < 5:
            print('Not enough feasible data, drawing Sobol exploration batch.')
            sobol = SobolEngine(dimension=self.dims, scramble=True)
            return None, sobol.draw(batch_size).to(dtype=self.dtype)

        gp = self._fit_gp()

        # Diagnostic: log the current best (legacy scalar) and where it lives.
        scalars = y_to_scalar(self.Y_history[self.feas_mask]).squeeze(-1)
        best_idx_in_feas = int(scalars.argmax().item())
        best_scalar = float(scalars.max().item())
        best_X = self.X_history[self.feas_mask][best_idx_in_feas]
        print(f'Current best composite: {best_scalar:.6e}')
        print(f'                    at: {from_unit_cube(best_X.numpy(), self.lb, self.ub)}')

        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([256]))
        acq = qLogNoisyExpectedImprovement(
            model=gp,
            X_baseline=self.X_history[self.feas_mask],
            sampler=sampler,
            objective=COMPOSITE_OBJECTIVE,
            prune_baseline=True,
        )

        bounds = torch.stack([
            torch.zeros(self.dims, dtype=self.dtype),
            torch.ones(self.dims, dtype=self.dtype),
        ])
        candidates, _ = optimize_acqf(
            acq_function=acq,
            bounds=bounds,
            q=batch_size,
            num_restarts=20,
            raw_samples=512,
            sequential=True,
        )
        return gp, candidates

    def ask(self, batch_size: int) -> torch.Tensor:
        gp, candidates = self._fitting_loop(batch_size)
        if gp is not None:
            ls = gp.covar_module.lengthscale.detach()
            print(f'GP lengthscales (per-output): shape={tuple(ls.shape)}, '
                  f'min={ls.min().item():.3e}, median={ls.median().item():.3e}, '
                  f'max={ls.max().item():.3e}')
        return candidates

    def tell(self, X_new, Y_new, feas_new, lb=None, ub=None):
        """Append a batch and (optionally) update bounds."""
        self.push_history(X_new, Y_new, feas_new)
        if lb is not None:
            self.lb = lb
        if ub is not None:
            self.ub = ub

    # -------------------------------------------------------------- best/state

    def return_best(self):
        """Return (best_composite_scalar, best_X) over feasible history."""
        if self.feas_mask is None or not self.feas_mask.any():
            return None, None
        Y_feas = self.Y_history[self.feas_mask]
        X_feas = self.X_history[self.feas_mask]
        scalars = y_to_scalar(Y_feas).squeeze(-1)
        idx = int(scalars.argmax().item())
        return float(scalars[idx].item()), X_feas[idx]

    def dump(self, fpath):
        with open(fpath, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, fpath):
        with open(fpath, 'rb') as f:
            return pickle.load(f)
