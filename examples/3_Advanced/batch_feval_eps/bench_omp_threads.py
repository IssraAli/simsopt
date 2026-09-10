#!/usr/bin/env python
"""
Time a single target() evaluation (spline -> VMEC -> Boozer transform ->
eps_eff) under whatever thread-count env vars the process started with.

Run directly to see one data point:
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python bench_omp_threads.py

Thread-count env vars are read once at interpreter/BLAS-library startup, so
they can't be changed mid-process -- that's why this is a separate script
meant to be re-launched per trial (see bench_omp_threads.sh) rather than a
loop over thread counts in one process.
"""
import os
import time

import numpy as np
from torch.quasirandom import SobolEngine

from bo_utils import write_doflist_maxlist_minlist
from test_target import target

# Same spline_kwargs as spline_test_bo.py, so this exercises the real
# problem size (ns=50 VMEC, booz_mpol=booz_ntor=48, etc. inside target()).
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


def fixed_candidate(lb):
    """Same seed=0 Sobol draw spline_test_bo.py uses for its first candidate,
    so every trial (thread count) evaluates the identical point."""
    sobol = SobolEngine(dimension=len(lb), scramble=True, seed=0)
    return sobol.draw(1)[0].numpy()


def thread_env_report():
    keys = ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]
    return {k: os.environ.get(k, "(unset)") for k in keys}


if __name__ == "__main__":
    dof_list, ub, lb = write_doflist_maxlist_minlist(spline_kwargs)
    X = fixed_candidate(lb)

    print(f"thread env: {thread_env_report()}")

    t0 = time.perf_counter()
    val, var = target(X, spline_kwargs, lb, ub)
    elapsed = time.perf_counter() - t0

    print(f"target() value: {val}, var: {var}")
    print(f"RESULT elapsed_sec={elapsed:.3f}")
