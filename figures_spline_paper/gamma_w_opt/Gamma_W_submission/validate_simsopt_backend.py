#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validate the firm3d-FREE Gamma_W backend (gamma_w_simsopt.py) against the exact
Fourier field and, if firm3d is installed, against firm3d itself.

    python validate_simsopt_backend.py BOOZMN.nc

Checks that need NO firm3d (always run):
  * grid modB vs exact-Fourier modB agreement (the fast backend vs the exact one)
  * modB throughput of both backends
  * one end-to-end Gamma_W (grid backend), as a smoke test

Checks that run ONLY if firm3d is importable:
  * grid/fourier modB vs firm3d modB
  * DECISIVE end-to-end test: identical markers + identical engine/settings, only
    the field is swapped firm3d <-> grid. This isolates the field backend from
    marker-sampling and run-settings differences.

Reference numbers observed here (Alex QH boozmn, N=128, t*=0.01, dt_max=4e-3, keep):
    0.01_00 : firm3d Gamma_W 0.004090  vs grid 0.004100   (0.2%)
    0.09_17 : firm3d Gamma_W 0.040439  vs grid 0.039958   (1.2%)
    modB grid vs firm3d: median ~3e-7, max ~2e-5 ; grid vs Fourier: median ~3e-8
"""
import os
import sys
import time
import importlib.util
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import booz_xform  # noqa: E402
from Gamma_W_final import GWParams, kappa, run_marker, aggregate  # noqa: E402
from gamma_w_simsopt import (BoozerXformFieldBundle, make_surface_markers,  # noqa: E402
                             gamma_w_from_boozer)

TWO_PI = 2.0 * np.pi


def main(path):
    bx = booz_xform.Booz_xform()
    bx.read_boozmn(path)
    grid = BoozerXformFieldBundle(bx, field_backend="grid")
    four = BoozerXformFieldBundle(bx, field_backend="fourier")
    print(f"file: {path}\n  nfp={grid.nfp}  Psi_LCFS={grid.Psi_LCFS:.6f}")

    # ---- grid vs exact Fourier (no firm3d needed) ----
    rng = np.random.default_rng(0)
    n = 3000
    s = rng.uniform(0.05, 0.95, n)
    th = rng.uniform(0, TWO_PI, n)
    ze = rng.uniform(0, TWO_PI, n)
    bg = grid.modB(s, th, ze)
    b4 = four.modB(s, th, ze)
    rel = np.abs(bg - b4) / np.abs(b4)
    print(f"\n[grid vs exact Fourier]  modB rel.diff median={np.median(rel):.2e}  max={rel.max():.2e}")

    # ---- throughput ----
    pts = np.column_stack([np.full(1280, 0.4), rng.uniform(0, TWO_PI, 1280),
                           rng.uniform(0, TWO_PI, 1280)])
    for name, fld in (("grid", grid.field), ("fourier", four.field)):
        fld.set_points(pts); fld.modB()  # warmup
        t0 = time.time()
        for _ in range(100):
            fld.set_points(pts); fld.modB()
        print(f"[throughput] {name:7s}: {1000 * (time.time() - t0) / 100:.2f} ms per 1280-pt cover")

    # ---- firm3d cross-check (optional) ----
    if importlib.util.find_spec("firm3d") is not None:
        from Gamma_W_final import FieldBundle as FirmFB
        print("\nfirm3d found -> cross-checking (field build ~20s)...")
        firm = FirmFB(path)
        bF = firm.modB(s, th, ze)
        for name, bb in (("grid", bg), ("fourier", b4)):
            r = np.abs(bb - bF) / np.abs(bF)
            print(f"[{name:7s} vs firm3d] modB rel.diff median={np.median(r):.2e}  max={r.max():.2e}")
        print(f"[psi0] firm3d={firm.psi0:.8e}  mine={grid.psi0:.8e}  rel={abs(firm.psi0-grid.psi0)/abs(firm.psi0):.1e}")

        # DECISIVE: same markers + same settings, swap only the field
        print("\n[decisive] identical markers + settings, field swapped:")
        markers = make_surface_markers(firm, 128, 0.3, seed=7)
        P = GWParams(t_star=0.01, dt_max=4e-3, branch_continue=False)
        vals = {}
        for name, fld in (("firm3d", firm), ("grid", grid)):
            kap = kappa(fld.Psi_LCFS)
            bc = {}
            rows = [run_marker(fld, m, P, kap, bc) for m in markers]
            a = aggregate(rows)
            vals[name] = a["Gamma_W"]
            print(f"  {name:7s}: Gamma_W={a['Gamma_W']:.6f}  band[{a['Gamma_W_low']:.4f},{a['Gamma_W_high']:.4f}]")
        print(f"  |firm3d - grid| = {abs(vals['firm3d'] - vals['grid']):.2e}  "
              f"({100*abs(vals['firm3d']-vals['grid'])/max(vals['firm3d'],1e-9):.2f}%)")
    else:
        print("\nfirm3d not installed -> skipping firm3d cross-check (expected on Perlmutter/Docker).")

    # ---- smoke test: full end-to-end via the grid backend ----
    print("\n[smoke] gamma_w_from_boozer grid backend, N=128 t*=0.05:")
    t0 = time.time()
    agg = gamma_w_from_boozer(bx, N=128, t_star=0.05, s0=0.3, seed=7,
                              field_backend="grid", verbose=True)
    print(f"  ({time.time() - t0:.0f}s)")
    return agg


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    main(sys.argv[1])
