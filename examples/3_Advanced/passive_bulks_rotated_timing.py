# coding: utf-8
# Copyright 2016-2020 HiddenSymmetries, MIT License
r"""Time :class:`PSCBulkArray` rebuilds on 6--12 pucks (or 16+ with ``--large``) with
arbitrary orientations.

Set ``SIMSOPT_PSCBULK_TIMING=1`` to record per-pair-class rows in
``_timing_rows`` (``rebuild_L_assembly_self|near|far|multipole``), plus
``rebuild_L_overhead_*`` kernels (kbasis upload, pair list, R_far calibration,
moments, self-check, symmetric scatter).

Run from the simsopt repo root after install::

    export SIMSOPT_PSCBULK_TIMING=1
    python examples/3_Advanced/passive_bulks_rotated_timing.py
    python examples/3_Advanced/passive_bulks_rotated_timing.py --large
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def _import_fixture():
    tdir = Path(__file__).resolve().parent.parent.parent / "tests" / "field"
    tpb = tdir / "test_passive_bulks.py"
    spec = importlib.util.spec_from_file_location("test_passive_bulks", tpb)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod._make_rotated_puck_array, mod._make_rotated_puck_array_large


def _median(xs: List[float]) -> float:
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return 0.0
    mid = n // 2
    return float(xs[mid]) if n % 2 else 0.5 * (xs[mid - 1] + xs[mid])


def _summarize_rows(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for r in rows:
        ph = str(r.get("phase", ""))
        if ph in (
            "rebuild_kbasis",
            "rebuild_cholesky",
            "rebuild_total",
            "rebuild_L_assembly",
        ):
            out[ph] = out.get(ph, 0.0) + float(r.get("seconds", 0.0))
        if ph.startswith("rebuild_L_assembly_"):
            out[ph] = out.get(ph, 0.0) + float(r.get("seconds", 0.0))
        if ph.startswith("rebuild_L_overhead_"):
            out[ph] = out.get(ph, 0.0) + float(r.get("seconds", 0.0))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--large",
        action="store_true",
        help="Use the large-scale rotating fixture (n_base up to 128); default sizes 16,32,48,64.",
    )
    p.add_argument(
        "--sizes",
        type=str,
        default=None,
        help="Comma-separated n_base. Default: 6,8,10,12 (small) or 16,32,48,64 (--large).",
    )
    p.add_argument(
        "--flags",
        type=str,
        default="",
        help="Space- or semicolon-separated KEY=VAL pairs for the environment.",
    )
    p.add_argument(
        "--out",
        type=str,
        default="",
        help="Write CSV to this path (default: passive_bulks_rotated_timing.csv in CWD).",
    )
    p.add_argument("--warmups", type=int, default=1)
    p.add_argument("--repeats", type=int, default=3)
    args = p.parse_args()
    if args.sizes is None:
        args.sizes = "16,32,48,64" if bool(args.large) else "6,8,10,12"

    os.environ["SIMSOPT_PSCBULK_TIMING"] = "1"
    for token in (args.flags or "").replace(";", " ").split():
        if not token.strip() or "=" not in token:
            continue
        k, v = token.split("=", 1)
        os.environ[k.strip()] = v.strip()

    _make_fn, _make_large = _import_fixture()
    _make = _make_large if bool(args.large) else _make_fn
    sizes = [int(s.strip()) for s in (args.sizes or "").split(",") if s.strip()]

    out_path = args.out or str(Path.cwd() / "passive_bulks_rotated_timing.csv")
    rows_csv: List[Dict[str, Any]] = []

    for nbase in sizes:
        n_min, n_max = (6, 128) if bool(args.large) else (6, 12)
        if nbase < n_min or nbase > n_max:
            print(
                f"skip n_base={nbase} (fixture supports {n_min}--{n_max} in this mode)"
            )
            continue
        psc0 = _make(n_base=nbase, seed=0)
        for _ in range(int(args.warmups)):
            psc0._rebuild()
        medians: List[Dict[str, float]] = []
        for _rep in range(int(args.repeats)):
            psc = _make(n_base=nbase, seed=0)
            t0 = time.perf_counter()
            psc._rebuild()
            tr = time.perf_counter() - t0
            srows = psc._timing_rows or []
            m = _summarize_rows(srows)
            m["_wall_rebuild"] = tr
            medians.append(m)
        # aggregate medians
        keys = set()
        for m in medians:
            keys.update(m.keys())
        comb: Dict[str, float] = {}
        for k in keys:
            comb[k] = _median([float(m.get(k, 0.0)) for m in medians])
        psc2 = _make(n_base=nbase, seed=0)
        psc2.recompute_currents()
        _ep = getattr(psc2, "eval_points", None)
        if _ep is not None and np.asarray(_ep, dtype=float).size > 0:
            psc2.B_at_points(_ep)  # noqa: B018 — end-to-end smoke

        r_near = 2.0
        try:
            r_near = float(os.environ.get("SIMSOPT_PSC_R_NEAR", "2.0"))
        except ValueError:
            pass
        n_pair_self = nbase
        n_pair = nbase * (nbase + 1) // 2
        print(
            f"n_base={nbase}  pairs={n_pair}  (self~{n_pair_self} )  "
            f"r_near={r_near}  key_times={ {k: round(comb[k], 4) for k in sorted(comb) if 'rebuild' in k} }"  # noqa: E501
        )

        row = {
            "n_base": nbase,
            "large": int(bool(args.large)),
            "flags": str(args.flags),
            "n_pairs": n_pair,
        }
        row.update({k: float(v) for k, v in comb.items()})
        rows_csv.append(row)

    if rows_csv:
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_csv[0].keys()))
            w.writeheader()
            w.writerows(rows_csv)
        print("wrote", out_path)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("error", e, file=sys.stderr)
        raise
