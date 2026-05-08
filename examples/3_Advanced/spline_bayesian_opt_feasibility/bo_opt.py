#!/usr/bin/env python

from bo_model import VanillaBO
from simsopt.geo import SurfaceBSpline
from test_target import target, parallel_batch_target
import numpy as np
from bo_utils import write_doflist_maxlist_minlist, from_unit_cube
from mpi4py import MPI
from torch.quasirandom import SobolEngine
import torch

max_iter = 500

if __name__ == "__main__":

    comm = MPI.COMM_WORLD
    nranks = comm.Get_size()
    rank = comm.Get_rank()
    batch_size = nranks

    spline_kwargs = {
        "axis_points": 3,
        "points_per_cs": 5,
        "n_cs": 4,
        "nfp": 2,
        "M": 6,
        "N": 6,
        "p_u": 3,
        "p_v": 3,
        "cs_equispaced": True,
        "rays_equispaced": False,
        "cs_global_angle_free": False,
        "axis_angles_fixed": True,
        "cs_basis": "polar",
        "nurbs": False,
    }

    dof_list, ub, lb = write_doflist_maxlist_minlist(spline_kwargs)

    n_init = 4 * len(lb)

    if rank == 0:
        optimizer = VanillaBO(
            dof_list=dof_list, lb=lb, ub=ub,
            X_history=None, y_history=None,
            spline_kwargs=spline_kwargs, target=parallel_batch_target,
        )

        print(f"lower bounds: {lb}")
        print(f"upper bounds: {ub}")

        # --- Initial Sobol sampling ---
        initial_X = []
        initial_results = []
        count = 0
        X_sobol = SobolEngine(dimension=len(optimizer.lb), scramble=True, seed=0)
        while count < n_init:
            X = X_sobol.draw(batch_size).to(dtype=optimizer.dtype, device=optimizer.device)
            initial_X.append(X)
            stop = [0]
            new_results = parallel_batch_target(X, optimizer.spline_kwargs, optimizer.lb, optimizer.ub, stop)
            initial_results.append(new_results)  # shape (batch, 2)
            count += batch_size

        # Stack and split into objective / feasibility histories
        optimizer.X_history = torch.cat(initial_X, dim=0).to(torch.double)
        all_results = torch.cat(initial_results, dim=0).to(torch.double)  # (n_init, 2)
        optimizer.y_history = all_results[:, 0:1]  # (n_init, 1)
        optimizer.c_history = all_results[:, 1:2]  # (n_init, 1)

        n_feas = int((optimizer.c_history.squeeze() > 0.5).sum().item())
        print(f"Completed {count} initial runs ({n_feas} feasible). Beginning bayesian iterations.")

        # --- Bayesian optimization loop ---
        for i in range(max_iter):
            stop = [0]
            new_x = optimizer.ask(batch_size=nranks)
            new_results = parallel_batch_target(new_x, spline_kwargs, lb, ub, stop)

            print(f"{i}/{max_iter}:")
            for k in range(new_results.shape[0]):
                feas = "✓" if new_results[k, 1] > 0.5 else "✗"
                print(f"  candidate {k}: f={new_results[k, 0]:.4f} [{feas}]")

            optimizer.tell(new_x, new_results, lb, ub)

        # Signal workers to stop
        stop = [1]
        parallel_batch_target(new_x, spline_kwargs, lb, ub, stop)

    else:
        # Worker loop: participate in MPI scatter/gather until stop signal
        stop = [0]
        dummy_surf = SurfaceBSpline(**spline_kwargs)
        while stop[0] == 0:
            parallel_batch_target(dummy_surf.x, spline_kwargs, lb, ub, stop)
