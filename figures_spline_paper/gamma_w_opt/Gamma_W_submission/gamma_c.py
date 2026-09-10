#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
gamma_c.py -- Nemov's energetic-particle proxy Gamma_c as a READ-ONLY by-product of
the Gamma_W pipeline (Gamma_W_final.py), from the IDENTICAL bounce integrals, markers
and source measure. No ODE integration, no branch continuation: Gamma_c is a
one-point (launch) statistic by construction -- that is the scientific point of the
comparison against Gamma_W.

Per trapped marker i at its launch (s0, alpha0, B_* = 1/lambda), the Nemov contour
inclination (Nemov et al., Phys. Plasmas 12, 112507 (2005) Eq.61; scalar form PoP 15,
052501 (2008)) is
        gamma_c_i = (2/pi) * atan2( |dJ/dalpha| , |dJ/ds| )   in [0,1]
where dJ/dalpha, dJ/ds are the SAME branch-preserving central differences that
Gamma_W's `_rhs` computes at t=0. We obtain them by calling `_rhs` itself (not a
copy), so the derivatives are bit-identical to the Gamma_W drift by construction:
        sdot = kap/tau * dJ_da   ->  dJ_da = sdot * tau / kap
        adot = -kap/tau * dJ_ds  ->  dJ_ds = -adot * tau / kap
(the common factor tau/kap cancels inside atan2, so gamma_c = (2/pi) atan2(|sdot|,|adot|)).

The isotropic-xi marker measure of make_surface_markers coincides with Nemov's
canonical lambda-kernel (|dxi/dlambda| = B0/(2 sqrt(1-lambda B0))), so the estimator is
the plain weighted mean of gamma_c^2 over trapped markers:
        Gamma_c_trapped = sum_valid w gamma_c^2 / sum_valid w        (per trapped particle)
        Gamma_c_source  = sum_valid w gamma_c^2 / sum_all   w        (folds in trapped frac)
"valid" = trapped-at-birth AND the launch drift resolved. Passing-at-birth and
FD-failed markers are excluded from the numerator always, and from the trapped
denominator (source denominator is always all markers, matching Gamma_W's bt).

CLI:
    python gamma_c.py --boozmn FILE.nc [--N 128] [--s0 0.3] [--seed 7]
    python gamma_c.py --batch 'GLOB' --s0 0.3 --out out.csv [--N 128] [--seed 7]
    python gamma_c.py --selfcheck FILE.nc      # asserts gamma_c uses _rhs's (sdot,adot)
"""
from __future__ import annotations

import argparse
import csv
import glob
import os

import numpy as np

# Reuse the Gamma_W engine unchanged. gamma_c is additive and read-only.
from Gamma_W_final import (FieldBundle, GWParams, make_surface_markers, run_marker,
                           _rhs, _dzeta_grid, Bstar_from_pitch, kappa,
                           V0_ALPHA, TWO_PI, PASSING)

TWO_OVER_PI = 2.0 / np.pi


def gamma_c_for_marker(field, m, P, kap, bcache):
    """One marker's launch-point gamma_c. Mirrors run_marker's setup EXACTLY (same
    B_*, same passing test, same zeta_hint), then one `_rhs` call at t=0."""
    s0, a0, z0, th0, xi = m["s0"], m["alpha0"], m["zeta0"], m["theta0"], m["pitch_xi"]
    B0 = field.modB_at(s0, th0, z0)
    B_star = Bstar_from_pitch(B0, xi)
    if s0 not in bcache:
        bcache[s0] = field.surface_B_range(s0)
    bmin, bmax = bcache[s0]
    row = dict(m)
    if B_star >= bmax * (1 - 1e-6):                       # passing at birth (identical to run_marker)
        row.update(passing_at_birth=True, gamma_c=np.nan, dJ_da=np.nan, dJ_ds=np.nan,
                   fd_fail=False, status=PASSING)
        return row
    dzg = _dzeta_grid(field, P)
    # IDENTICAL to Gamma_W's very first characteristic evaluation: RHS(s0, alpha0 % 2pi,
    # zeta0, tau_med=None). No branch continuation (plain _rhs).
    sdot, adot, base, st = _rhs(field, s0, a0 % TWO_PI, B_star, V0_ALPHA, kap, z0, P,
                                bmin, bmax, dzg, None)
    if base is None or st != "ok":                       # BRANCH_EVENT/TAU_GUARD/FD_MISMATCH
        row.update(passing_at_birth=False, gamma_c=np.nan, dJ_da=np.nan, dJ_ds=np.nan,
                   fd_fail=True, status=st)
        return row
    dJ_da = sdot * base.tau / kap
    dJ_ds = -adot * base.tau / kap
    if abs(dJ_da) < 1e-300 and abs(dJ_ds) < 1e-300:
        gc = 0.0
    else:
        gc = TWO_OVER_PI * np.arctan2(abs(dJ_da), abs(dJ_ds))
    row.update(passing_at_birth=False, gamma_c=float(gc), dJ_da=float(dJ_da),
               dJ_ds=float(dJ_ds), fd_fail=False, status="ok")
    return row


def aggregate_gamma_c(rows):
    w = np.array([float(r["weight"]) for r in rows])
    passing = np.array([bool(r["passing_at_birth"]) for r in rows])
    fdfail = np.array([bool(r.get("fd_fail", False)) for r in rows])
    gc = np.array([float(r["gamma_c"]) for r in rows])       # NaN for passing/fd_fail
    valid = (~passing) & (~fdfail)
    wa = w.sum()
    num = float((w[valid] * gc[valid] ** 2).sum()) if valid.any() else 0.0
    den_tr = float(w[valid].sum())
    return dict(
        Gamma_c_trapped=(num / den_tr) if den_tr > 0 else float("nan"),
        Gamma_c_source=num / wa if wa > 0 else float("nan"),
        trapped_frac=float(w[~passing].sum() / wa),
        n=len(rows),
        n_trapped=int((~passing).sum()),
        n_valid=int(valid.sum()),
        n_fd_fail=int((fdfail & (~passing)).sum()),
        fd_fail_w_frac=float(w[fdfail & (~passing)].sum() / wa),
        gamma_c_median=float(np.nanmedian(gc[valid])) if valid.any() else float("nan"),
    )


def gamma_c_for_boozmn(boozmn_path, N=128, s0=0.3, seed=7, verbose=True):
    """End-to-end Gamma_c for one Boozer equilibrium. Uses the IDENTICAL marker set
    as gamma_w_for_boozmn (same make_surface_markers(field, N, s0, seed) call)."""
    field = FieldBundle(boozmn_path)
    P = GWParams()                                       # defaults: same n_periods, dalpha, ds_fd
    markers = make_surface_markers(field, N, s0, seed=seed)
    kap = kappa(field.Psi_LCFS)
    bcache = {}
    rows = [gamma_c_for_marker(field, m, P, kap, bcache) for m in markers]
    agg = aggregate_gamma_c(rows)
    agg.update(nfp=field.nfp, Psi_LCFS=field.Psi_LCFS, s0=s0, N=N, seed=seed)
    if verbose:
        print(f"  nfp={field.nfp} s0={s0} N={N} seed={seed}")
        print(f"  Gamma_c_trapped = {agg['Gamma_c_trapped']:.6f}   "
              f"Gamma_c_source = {agg['Gamma_c_source']:.6f}")
        print(f"  trapped_frac = {agg['trapped_frac']:.3f}  n_valid = {agg['n_valid']}  "
              f"fd_fail_w_frac = {agg['fd_fail_w_frac']:.3f}"
              + ("  [FLAG >0.05]" if agg['fd_fail_w_frac'] > 0.05 else ""))
    return agg


# ------------------------------------------------------------------ self-check
def selfcheck(boozmn_path, N=32, s0=0.3, seed=7):
    """Assert the launch-point drift used for gamma_c is the SAME (sdot, adot) that
    Gamma_W's `_rhs` returns, on the identical markers, to the last bit; and that a
    re-run with the same seed is byte-identical (determinism)."""
    field = FieldBundle(boozmn_path)
    P = GWParams()
    markers = make_surface_markers(field, N, s0, seed=seed)
    kap = kappa(field.Psi_LCFS)
    bcache = {}
    ok = True
    for m in markers:
        s0m, a0, z0, th0, xi = m["s0"], m["alpha0"], m["zeta0"], m["theta0"], m["pitch_xi"]
        B0 = field.modB_at(s0m, th0, z0)
        B_star = Bstar_from_pitch(B0, xi)
        if s0m not in bcache:
            bcache[s0m] = field.surface_B_range(s0m)
        bmin, bmax = bcache[s0m]
        if B_star >= bmax * (1 - 1e-6):
            continue
        dzg = _dzeta_grid(field, P)
        sd, ad, base, st = _rhs(field, s0m, a0 % TWO_PI, B_star, V0_ALPHA, kap, z0, P, bmin, bmax, dzg, None)
        row = gamma_c_for_marker(field, m, P, kap, bcache)
        if base is not None and st == "ok":
            dJ_da_chk = sd * base.tau / kap
            gc_chk = TWO_OVER_PI * np.arctan2(abs(dJ_da_chk), abs(-ad * base.tau / kap))
            if not (abs(gc_chk - row["gamma_c"]) < 1e-15):
                ok = False
    # determinism
    a1 = gamma_c_for_boozmn(boozmn_path, N=N, s0=s0, seed=seed, verbose=False)
    a2 = gamma_c_for_boozmn(boozmn_path, N=N, s0=s0, seed=seed, verbose=False)
    det = (a1["Gamma_c_trapped"] == a2["Gamma_c_trapped"] and
           a1["Gamma_c_source"] == a2["Gamma_c_source"])
    print(f"  _rhs-agreement: {'OK' if ok else 'FAIL'}   determinism: {'OK' if det else 'FAIL'}")
    return ok and det


# ------------------------------------------------------------------ batch + CLI
def _cid_from_path(p):
    b = os.path.basename(p)
    for pre in ("boozmn_", "boozmn"):
        if b.startswith(pre):
            b = b[len(pre):]
            break
    return b[:-3] if b.endswith(".nc") else b


def run_batch(pattern, out, N=128, s0=0.3, seed=7):
    paths = sorted(glob.glob(pattern))
    if not paths:
        print("no files match", pattern); return
    rows = []
    for i, p in enumerate(paths):
        cid = _cid_from_path(p)
        try:
            a = gamma_c_for_boozmn(p, N=N, s0=s0, seed=seed, verbose=False)
        except Exception as e:
            print(f"[{i+1}/{len(paths)}] {cid} FAIL {str(e)[:60]}", flush=True); continue
        a["cid"] = cid
        rows.append(a)
        print(f"[{i+1}/{len(paths)}] {cid}  Gc_tr={a['Gamma_c_trapped']:.5f} "
              f"Gc_src={a['Gamma_c_source']:.5f} trap={a['trapped_frac']:.3f} "
              f"fdfail={a['fd_fail_w_frac']:.3f}", flush=True)
    if rows:
        keys = ["cid", "Gamma_c_trapped", "Gamma_c_source", "trapped_frac", "n",
                "n_trapped", "n_valid", "n_fd_fail", "fd_fail_w_frac", "gamma_c_median",
                "nfp", "Psi_LCFS", "s0", "N", "seed"]
        with open(out, "w", newline="") as fh:
            wtr = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
            wtr.writeheader()
            for r in rows:
                wtr.writerow(r)
        print(f"wrote {out} ({len(rows)} configs)")


def main():
    ap = argparse.ArgumentParser(description="Nemov Gamma_c inside the Gamma_W pipeline.")
    ap.add_argument("--boozmn", type=str, default=None)
    ap.add_argument("--batch", type=str, default=None, help="glob of boozmn files")
    ap.add_argument("--out", type=str, default="gamma_c_out.csv")
    ap.add_argument("--selfcheck", type=str, default=None, help="boozmn path for the self-check")
    ap.add_argument("--N", type=int, default=128)
    ap.add_argument("--s0", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    if args.selfcheck:
        selfcheck(args.selfcheck, s0=args.s0, seed=args.seed)
    elif args.batch:
        run_batch(args.batch, args.out, N=args.N, s0=args.s0, seed=args.seed)
    elif args.boozmn:
        print(f"About to score {args.boozmn}")
        gamma_c_for_boozmn(args.boozmn, N=args.N, s0=args.s0, seed=args.seed)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
