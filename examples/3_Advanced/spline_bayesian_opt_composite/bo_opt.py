#!/usr/bin/env python
"""
Composite Bayesian Optimization driver — multi-output GP over the 25-component
QA + iota residual vector. See `bo_model.CompositeBO` and the plan-file for
details. MPI: rank 0 is the BO controller, ranks 1..N-1 are VMEC evaluators
that loop on `parallel_batch_target` until rank 0 broadcasts `stop=[1]`.
"""

import sys

import numpy as np
import torch
from mpi4py import MPI
from torch.quasirandom import SobolEngine

from bo_model import CompositeBO, y_to_scalar
from bo_utils import (
    from_unit_cube,
    make_warm_start,
    write_doflist_maxlist_minlist,
)
from simsopt.geo import SurfaceBSpline
from test_target import N_RESIDUALS, parallel_batch_target


MAX_ITER = 500


def _print_iter_summary(i, max_iter, new_x, new_Y, new_feas):
    print(f'{i}/{max_iter}:')
    for k in range(new_Y.shape[0]):
        feas = '✓' if bool(new_feas[k].item()) else '✗'
        scalar = float(y_to_scalar(new_Y[k:k+1]).item())
        print(f'  candidate {k}: composite={scalar:.6e} [{feas}]')
    sys.stdout.flush()


def main():
    comm = MPI.COMM_WORLD
    nranks = comm.Get_size()
    rank = comm.Get_rank()
    batch_size = nranks

    spline_kwargs = {
        'axis_points': 3,
        'points_per_cs': 5,
        'n_cs': 4,
        'nfp': 2,
        'M': 6,
        'N': 6,
        'p_u': 3,
        'p_v': 3,
        'cs_equispaced': True,
        'rays_equispaced': False,
        'cs_global_angle_free': False,
        'axis_angles_fixed': True,
        'cs_basis': 'polar',
        'nurbs': False,
    }

    # `write_doflist_maxlist_minlist` applies the same gauge fix the warm-start
    # uses, so the bound vector and the warm-start DOF vector have matching dims.
    dof_list, ub, lb = write_doflist_maxlist_minlist(spline_kwargs)
    dims = len(lb)

    if rank == 0:
        optimizer = CompositeBO(
            dof_list=dof_list, lb=lb, ub=ub,
            spline_kwargs=spline_kwargs, target=parallel_batch_target,
        )

        print(f'dims: {dims}')
        print(f'lower bounds: {lb}')
        print(f'upper bounds: {ub}')

        # ---- 1. Warm-start: a circular cross-section + Gaussian perturbations.
        # The LSQ reference (spline_surface.py) starts here and converges; this
        # gives the GP an anchor inside the feasible basin from iteration 1.
        warm_X = make_warm_start(spline_kwargs, lb, ub, n_perturb=4, sigma=0.02)
        n_warm = warm_X.shape[0]
        print(f'Warm-start: {n_warm} points anchored at default_r=0.4')

        # ---- 2. Random Sobol initialization (smaller now that warm-start exists).
        n_sobol = 4 * dims
        sobol = SobolEngine(dimension=dims, scramble=True, seed=0)
        sobol_X = sobol.draw(n_sobol).to(dtype=torch.double)
        init_X = torch.cat([warm_X, sobol_X], dim=0)
        print(f'Sobol-init: {n_sobol} points; total init = {init_X.shape[0]}')

        # Evaluate init in batches of `batch_size` so all ranks stay busy.
        for start in range(0, init_X.shape[0], batch_size):
            stop = [0]
            X_batch = init_X[start:start + batch_size]
            # Pad to batch_size if the last batch is short — workers are
            # always expecting `batch_size` candidates from comm.scatter.
            if X_batch.shape[0] < batch_size:
                pad = batch_size - X_batch.shape[0]
                X_batch = torch.cat([X_batch, torch.zeros(pad, dims, dtype=torch.double)], dim=0)
                Y_batch, feas_batch = parallel_batch_target(
                    X_batch, spline_kwargs, lb, ub, stop
                )
                # Drop padded rows.
                Y_batch = Y_batch[: -pad]
                feas_batch = feas_batch[: -pad]
                X_batch = X_batch[: -pad]
            else:
                Y_batch, feas_batch = parallel_batch_target(
                    X_batch, spline_kwargs, lb, ub, stop
                )
            optimizer.push_history(X_batch, Y_batch, feas_batch)
            n_feas_total = int(optimizer.feas_mask.sum().item())
            print(f'  init batch {start // batch_size}: '
                  f'{int(feas_batch.sum().item())}/{X_batch.shape[0]} feasible '
                  f'(running total {n_feas_total} feasible)')
            sys.stdout.flush()

        # Sanity: warm-start center should be feasible if VMEC is happy with
        # default_r=0.4 — flag if it isn't.
        if not bool(optimizer.feas_mask[0].item()):
            print('WARNING: warm-start center evaluated infeasible — check bounds/spline_kwargs.')

        n_feasible_init = int(optimizer.feas_mask.sum().item())
        n_total_init = int(optimizer.feas_mask.shape[0])
        print(f'Init complete: {n_feasible_init}/{n_total_init} feasible. Beginning BO.')

        # ---- 3. Bayesian optimization loop.
        for i in range(MAX_ITER):
            stop = [0]
            new_x = optimizer.ask(batch_size=batch_size)
            new_Y, new_feas = parallel_batch_target(new_x, spline_kwargs, lb, ub, stop)

            _print_iter_summary(i, MAX_ITER, new_x, new_Y, new_feas)
            optimizer.tell(new_x, new_Y, new_feas, lb=lb, ub=ub)

            best_scalar, _ = optimizer.return_best()
            if best_scalar is not None:
                print(f'  best-so-far composite: {best_scalar:.6e}')
            sys.stdout.flush()

        # Signal workers to exit and report final result.
        stop = [1]
        parallel_batch_target(new_x, spline_kwargs, lb, ub, stop)

        best_scalar, best_X = optimizer.return_best()
        print('==== FINAL ====')
        if best_scalar is not None:
            print(f'best composite: {best_scalar:.6e}')
            print(f'best X (physical): {from_unit_cube(best_X.numpy(), lb, ub)}')
        else:
            print('no feasible point found')

    else:
        # Worker loop: dummy candidate is irrelevant — comm.scatter on rank 0
        # sends each worker its own unit-cube row.
        stop = [0]
        dummy_surf = SurfaceBSpline(**spline_kwargs)
        dummy_surf.axis.fix('r_axis_0')
        while stop[0] == 0:
            parallel_batch_target(dummy_surf.x, spline_kwargs, lb, ub, stop)


if __name__ == '__main__':
    main()
