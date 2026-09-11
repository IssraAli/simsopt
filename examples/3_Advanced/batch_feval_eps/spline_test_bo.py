#!/usr/bin/env python

import numpy as np
import torch
from bo_model import VanillaBO
from bo_utils import from_unit_cube, write_doflist_maxlist_minlist
from mpi4py import MPI
from simsopt.geo import SurfaceBSpline
from test_target import parallel_batch_target
from torch.quasirandom import SobolEngine

max_iter = 1000

LOG_PATH = "bo_progress.log"


def _to_numpy(x):
    """Convert a torch.Tensor (possibly requiring grad) or array-like to numpy."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def make_logger(log_path):
    """Return a log(msg) function that prints (unbuffered) and appends to a file."""
    log_file = open(log_path, "a", buffering=1)

    def log(msg=""):
        print(msg, flush=True)
        log_file.write(str(msg) + "\n")
        log_file.flush()

    return log


def log_batch(log, tag, X_unit, y, success_mask, best_val, best_x, lb, ub):
    """Log the un-normalized candidates/values for one batch, plus the best ever.

    Failed candidates (success_mask False) are logged but flagged as excluded
    from the GP: their y is a sentinel penalty, not a noisy measurement of the
    real target, so it must not be fit as if it were.
    """
    X_unnorm = from_unit_cube(_to_numpy(X_unit), lb, ub)
    y_arr = _to_numpy(y).reshape(-1)
    success_mask = np.asarray(success_mask, dtype=bool).reshape(-1)
    log(tag)
    for k in range(X_unnorm.shape[0]):
        status = "ok" if success_mask[k] else "FAILED (excluded from GP)"
        log(f"  candidate {k} [{status}]: x={X_unnorm[k].tolist()}, f={y_arr[k]!r}")
    log(f"  best ever: f={best_val!r} at x={np.asarray(best_x).tolist()}")


def _batch_best(y, success_mask):
    """Return (best_val, best_idx) among successful candidates, or (None, None)."""
    y_arr = _to_numpy(y).reshape(-1)
    success_mask = np.asarray(success_mask, dtype=bool).reshape(-1)
    if not success_mask.any():
        return None, None
    valid_idx = np.flatnonzero(success_mask)
    best_idx = valid_idx[np.argmax(y_arr[valid_idx])]
    return float(y_arr[best_idx]), int(best_idx)


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    nranks = comm.Get_size()
    rank = comm.Get_rank()
    batch_size = nranks

    spline_kwargs = {
        "axis_points": 3,
        "points_per_cs": 4,
        "n_cs": 5,
        "nfp": 2,
        "M": 9,
        "N": 4,
        "p_u": 3,
        "p_v": 3,
        "cs_equispaced": True,
        "rays_equispaced": False,
        "cs_global_angle_free": False,
        "axis_angles_fixed": False,
        "cs_basis": "polar",
        "nurbs": False,
        "use_bishop_frame": True,
    }

    dof_list, ub, lb = write_doflist_maxlist_minlist(spline_kwargs)

    n_init = 4 * len(lb)

    if rank == 0:
        log = make_logger(LOG_PATH)

        optimizer = VanillaBO(
            dof_list=dof_list,
            lb=lb,
            ub=ub,
            X_history=None,
            y_history=None,
            spline_kwargs=spline_kwargs,
            target=parallel_batch_target,
        )

        log(f"lower bounds: {lb}")
        log(f"upper bounds: {ub}")

        best_val = -np.inf
        best_x = None

        # initial runs

        initial_X = []
        initial_y = []
        count = 0
        X_sobol = SobolEngine(dimension=len(lb), scramble=True, seed=0)
        while count < n_init:  # len(initial_y) < n_init:
            X = X_sobol.draw(batch_size)
            # print(f'X: {X}')
            stop = [0]
            new_y, success_mask = parallel_batch_target(
                X, spline_kwargs, lb, ub, stop
            )
            success_mask = np.asarray(success_mask, dtype=bool).reshape(-1)
            batch_best_val, batch_best_idx = _batch_best(new_y, success_mask)
            if batch_best_val is not None and batch_best_val > best_val:
                best_val = batch_best_val
                best_x = from_unit_cube(_to_numpy(X[batch_best_idx]), lb, ub)
            log_batch(
                log, f"Random iter {count}:", X, new_y, success_mask,
                best_val, best_x, lb, ub,
            )
            if success_mask.any():
                initial_X.append(_to_numpy(X)[success_mask])
                initial_y.append(_to_numpy(new_y).reshape(-1)[success_mask])
            count += batch_size
        if initial_X:
            optimizer.X_history = torch.Tensor(
                np.concatenate(initial_X, axis=0)
            ).to(torch.double)
            optimizer.y_history = torch.Tensor(
                np.concatenate(initial_y, axis=0).reshape(-1, 1)
            ).to(torch.double)
        else:
            optimizer.X_history = torch.empty((0, optimizer.dims), dtype=torch.double)
            optimizer.y_history = torch.empty((0, 1), dtype=torch.double)
        log(f"Completed {count} initial runs "
            f"({optimizer.X_history.shape[0]} successful). "
            "Beginning bayesian iterations. ")
        # print(initial_y)
        # print(np.std(initial_y))
        # print(np.mean(initial_y))
        i = 0

        # bayesian runs

        while i < max_iter:
            stop = [0]
            new_x = optimizer.ask(batch_size=nranks)
            new_y, success_mask = parallel_batch_target(
                new_x, spline_kwargs, lb, ub, stop
            )
            success_mask = np.asarray(success_mask, dtype=bool).reshape(-1)
            batch_best_val, batch_best_idx = _batch_best(new_y, success_mask)
            if batch_best_val is not None and batch_best_val > best_val:
                best_val = batch_best_val
                best_x = from_unit_cube(
                    _to_numpy(new_x[batch_best_idx]), lb, ub
                )
            log_batch(
                log, f"Bayesian iter {i}/{max_iter}:", new_x, new_y, success_mask,
                best_val, best_x, lb, ub,
            )
            if success_mask.any():
                optimizer.tell(
                    _to_numpy(new_x)[success_mask],
                    _to_numpy(new_y).reshape(-1)[success_mask],
                    lb, ub,
                )
            else:
                log("  all candidates in this batch failed; GP not updated")
            i += 1
            if i % 10 == 0:
                optimizer.dump("checkpoint.pkl")
        stop = [1]
    #     parallel_batch_target(new_x, spline_kwargs, lb, ub, stop)
    else:
        stop = [0]
        dummy_surf = SurfaceBSpline(**spline_kwargs)
        while stop[0] == 0:
            parallel_batch_target(dummy_surf.x, spline_kwargs, lb, ub, stop)
    # # except:
    # #     optimizer.dump(f'opt_{max_iter}_2.pkl')
