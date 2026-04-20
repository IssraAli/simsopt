#!/usr/bin/env python
r"""Stress / timing harness for :class:`~simsopt.field.psc_bulk.PSCBulkArray` at scale.

Builds a symmetry-validation-style layout (same recipe as
``tests/field/test_passive_bulks._make_symmetry_validation_array``) with a
configurable number of **base** pucks, then reports wall-clock for construction,
median time for ``recompute_currents`` + :meth:`PSCBulkArray.B_at_points`, and
optional RSS (``psutil``).

Usage::

    conda activate stellcoilbench_py312
    python passive_bulks_stress.py --n-base 50

Environment:

    SIMSOPT_JAX_PRIME=1  Optional warm-up trace of hot JAX kernels on first init.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import time
from pathlib import Path

import numpy as np

_TPB = Path(__file__).resolve().parent.parent.parent / "tests" / "field" / "test_passive_bulks.py"
_spec = importlib.util.spec_from_file_location("_tpb_stress", _TPB)
_tpb_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_tpb_mod)
_make_symmetry_validation_array = _tpb_mod._make_symmetry_validation_array


def _rss_mb() -> float | None:
    try:
        import psutil

        return float(psutil.Process(os.getpid()).memory_info().rss) / (1024.0**2)
    except Exception:
        return None


def main() -> None:
    p = argparse.ArgumentParser(description="PSCBulkArray stress / timing.")
    p.add_argument(
        "--n-base",
        type=int,
        default=50,
        help="Number of base pucks (before nfp/stellsym replication).",
    )
    p.add_argument("--nfp", type=int, default=2)
    p.add_argument("--stellsym", action="store_true", help="Enable stellsym (default off).")
    p.add_argument("--warm-iters", type=int, default=25, help="Timed inner-loop iterations.")
    args = p.parse_args()

    rss0 = _rss_mb()
    t0 = time.perf_counter()
    psc = _make_symmetry_validation_array(
        nfp=int(args.nfp),
        stellsym=bool(args.stellsym),
        n_base=int(args.n_base),
    )
    t_build = time.perf_counter() - t0
    rss1 = _rss_mb()

    pts = np.asarray(psc.eval_points[: min(4, psc.eval_points.shape[0])], dtype=float)
    times: list[float] = []
    for _ in range(max(1, int(args.warm_iters))):
        t1 = time.perf_counter()
        psc.recompute_currents()
        _ = psc.B_at_points(pts)
        times.append(time.perf_counter() - t1)
    med_ms = float(np.median(times)) * 1000.0

    print(f"n_base = {psc._n_base_pucks}  nfp = {args.nfp}  stellsym = {args.stellsym}")
    print(f"Build wall-clock = {t_build:.3f} s")
    print(f"Inner loop (recompute + B_at_points) median = {med_ms:.2f} ms  (N={len(times)})")
    if rss0 is not None and rss1 is not None:
        print(f"RSS ≈ {rss0:.1f} MB before, {rss1:.1f} MB after build (+{rss1 - rss0:.1f} MB)")


if __name__ == "__main__":
    main()
