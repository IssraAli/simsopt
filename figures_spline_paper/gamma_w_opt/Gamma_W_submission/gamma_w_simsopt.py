#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
================================================================================
gamma_w_simsopt.py  --  firm3d-FREE backend for Gamma_W (simsopt + booz_xform)
================================================================================

Same Gamma_W engine as Gamma_W_final.py, but the Boozer field is supplied by
booz_xform (via simsopt.mhd.Boozer) instead of firm3d. Needs only
simsopt + booz_xform + scipy + numpy -- NO firm3d -- so Gamma_W can run inline in
a simsopt optimization loop (firm3d is hard to build on some clusters and cannot
share a process with simsopt).

TWO FIELD BACKENDS
------------------
field_backend="grid"   (DEFAULT, FAST -- for optimization loops):
    Precompute |B| on a periodic (s, theta, zeta) grid ONCE per equilibrium, then
    evaluate by cubic interpolation (scipy.ndimage.map_coordinates, periodic in
    theta/zeta). This is what firm3d's InterpolatedBoozerField does in C++, so
    each modB call is O(1) instead of O(n_modes). ~20-50x faster than "fourier".
field_backend="fourier" (EXACT -- for validation / reference):
    Direct Fourier sum  |B| = sum_mn bmnc_b_mn(s) cos(xm*theta - xn*zeta), with
    bmnc_b_mn(s) a scipy cubic spline. Exact in the angles (more accurate than
    firm3d's grid spline); matches firm3d's modB to ~1e-8. Slow.

Both share everything else with Gamma_W_final.py: the FieldBundle methods
(profiles/gII/modB/modB_at/surface_B_range/helicity/line_B) are INHERITED
byte-identical -- only __init__ and the raw field object differ. The marker
sampler is copied from firm3d (open source; attribution below).

FIELD CONVENTIONS (identical to firm3d.BoozerSplineField; verified vs source)
    psi0    = -bx.phi[-1] / (2*pi)                       (VMEC sign convention)
    nfp     =  bx.nfp
    iota(s) =  CubicSpline(bx.s_b, bx.iota)
    G(s)    =  CubicSpline(bx.s_b, bx.Boozer_G_all)      (= bvco_b)
    I(s)    =  CubicSpline(bx.s_b, bx.Boozer_I_all)      (= buco_b)
    |B|     =  sum_mn bmnc_b_mn(s) cos(xm_b*theta - xn_b*zeta)  [+ bmns sin if asym]

USAGE
    from simsopt.mhd import Vmec, Boozer
    v = Vmec("input.QA_nfp2"); v.run()
    b = Boozer(v, mpol=16, ntor=16); b.register(v.s_half_grid); b.run()
    agg = gamma_w_from_boozer(b, N=128, t_star=0.2, s0=0.3, seed=7)   # grid backend
    print(agg["Gamma_W"])

or from a saved boozmn:  gamma_w_from_boozer("boozmn_xxx.nc", ...)
or from a Vmec directly: gamma_w_from_boozer(vmec, ..., mpol=16, ntor=16)
================================================================================
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import spline_filter1d, map_coordinates

# ---- pure-numpy engine (importing this does NOT import firm3d; the only firm3d
#      touch in the engine is FieldBundle.__init__, which we override). Prefer the
#      canonical standalone engine; fall back to the perlmutter copy name.
try:
    from Gamma_W_final import (FieldBundle as _FirmFieldBundle, GWParams, kappa,
                               run_marker, aggregate, V0_ALPHA, TWO_PI)
except ImportError:  # colleague's perlmutter tree ships the engine as Gamma_W_final_fix
    from Gamma_W_final_fix import (FieldBundle as _FirmFieldBundle, GWParams, kappa,
                                   run_marker, aggregate, V0_ALPHA, TWO_PI)


# ================================================================ PART A
#   Marker samplers -- COPIED from firm3d (open source, BSD):
#   firm3d/src/firm3d/field/tracing_helpers.py (serial path only).
# ================================================================
def _parallel_loop_bounds(comm, n):
    if comm is None:
        return 0, n
    idxs = [i * n // comm.size for i in range(comm.size + 1)]
    return idxs[comm.rank], idxs[comm.rank + 1]


def initialize_position_uniform_surf(field, nparticles, s, ntheta_max=100,
                                     nzeta_max=100, comm=None, seed=None):
    r"""Positions on surface s, uniform w.r.t. the Boozer volume element (rejection
    against J = (G + iota I)/|B|^2). COPIED from firm3d.field.tracing_helpers."""
    nfp = field.nfp
    np.random.seed(seed)
    theta_grid = np.linspace(0, 2 * np.pi, ntheta_max, endpoint=False)
    zeta_grid = np.linspace(0, 2 * np.pi / nfp, nzeta_max, endpoint=False)
    [zeta_grid, theta_grid] = np.meshgrid(zeta_grid, theta_grid)
    points = np.zeros((len(theta_grid.flatten()), 3))
    points[:, 0] = s
    points[:, 1] = theta_grid.flatten()
    points[:, 2] = zeta_grid.flatten()

    field.set_points(points)
    G = field.G()
    iota = field.iota()
    I = field.I()
    modB = field.modB()
    J = (G + iota * I) / (modB**2)
    J_max = np.max(J)

    theta_init, zeta_init, s_init = [], [], []
    points = np.zeros((1, 3))
    first, last = _parallel_loop_bounds(comm, nparticles)
    for _i in range(first, last):
        while True:
            rand1 = np.random.uniform(0, 1, None)
            theta = np.random.uniform(0, 2 * np.pi, None)
            zeta = np.random.uniform(0, 2 * np.pi / nfp, None)
            points[:, 0] = s
            points[:, 1] = theta
            points[:, 2] = zeta
            field.set_points(points)
            J = (field.G()[0, 0] + field.iota()[0, 0] * field.I()[0, 0]) / (
                field.modB()[0, 0] ** 2
            )
            if rand1 <= J / J_max:
                s_init.append(s)
                theta_init.append(theta)
                zeta_init.append(zeta)
                break

    if comm is not None:
        s_init = [i for o in comm.allgather(s_init) for i in o]
        theta_init = [i for o in comm.allgather(theta_init) for i in o]
        zeta_init = [i for o in comm.allgather(zeta_init) for i in o]

    points = np.zeros((nparticles, 3))
    points[:, 0] = np.asarray(s_init)
    points[:, 1] = np.asarray(theta_init)
    points[:, 2] = np.asarray(zeta_init)
    return points


def initialize_velocity_uniform(vpar0, nParticles, comm=None, seed=None):
    r"""Uniform v_par in [-vpar0, vpar0]. COPIED from firm3d.field.tracing_helpers."""
    verbose = comm.rank == 0 if comm is not None else True
    if seed is not None:
        np.random.seed(seed)
    vpar_init = np.random.uniform(-vpar0, vpar0, (nParticles,)) if verbose else None
    if comm is not None:
        vpar_init = comm.bcast(vpar_init, root=0)
    return vpar_init


# ================================================================ PART B
#   Raw fields (firm3d-InterpolatedBoozerField-compatible: set_points/G/I/iota/modB
#   each returning an (N,1) array). Two implementations share G/I/iota (cheap 1-D
#   splines); they differ only in modB.
# ================================================================
class _RawBoozerBase:
    def __init__(self, bx):
        s_b = np.asarray(bx.s_b, float)
        self.nfp = int(bx.nfp)
        self.xm = np.asarray(bx.xm_b, float)
        self.xn = np.asarray(bx.xn_b, float)
        self.asym = bool(getattr(bx, "asym", False))
        self._bmnc = np.asarray(bx.bmnc_b, float)                 # (mnboz, ns_b)
        self._G_sp = CubicSpline(s_b, np.asarray(bx.Boozer_G_all, float))
        self._I_sp = CubicSpline(s_b, np.asarray(bx.Boozer_I_all, float))
        self._io_sp = CubicSpline(s_b, np.asarray(bx.iota, float))
        self._bmnc_sp = CubicSpline(s_b, self._bmnc.T)            # eval(s) -> (...,mnboz)
        self._bmns = None
        self._bmns_sp = None
        if self.asym:
            self._bmns = np.asarray(bx.bmns_b, float)
            self._bmns_sp = CubicSpline(s_b, self._bmns.T)
        self._pts = None

    def set_points(self, points):
        self._pts = np.asarray(points, float).reshape(-1, 3)
        return self

    def G(self):
        return self._G_sp(self._pts[:, 0]).reshape(-1, 1)

    def I(self):
        return self._I_sp(self._pts[:, 0]).reshape(-1, 1)

    def iota(self):
        return self._io_sp(self._pts[:, 0]).reshape(-1, 1)


class _RawBoozerFourier(_RawBoozerBase):
    """Exact: |B| = sum_mn bmnc(s) cos(xm*theta - xn*zeta). Slow, reference-grade."""
    def modB(self):
        s, th, ze = self._pts[:, 0], self._pts[:, 1], self._pts[:, 2]
        bmnc = np.atleast_2d(self._bmnc_sp(s))                     # (N, mnboz)
        ang = np.outer(th, self.xm) - np.outer(ze, self.xn)       # (N, mnboz)
        B = np.einsum("ij,ij->i", bmnc, np.cos(ang))
        if self.asym:
            B = B + np.einsum("ij,ij->i", np.atleast_2d(self._bmns_sp(s)), np.sin(ang))
        return B.reshape(-1, 1)


class _RawBoozerGrid(_RawBoozerBase):
    """Fast: precompute |B| on a periodic (s,theta,zeta) grid, then cubic
    interpolation (scipy.ndimage.map_coordinates). O(1) per eval. Matches the
    Fourier field to the grid-resolution tolerance (validate_backend checks it)."""
    _PAD = 2   # edge-pad on the (non-periodic) s axis so grid-wrap never wraps s

    def __init__(self, bx, ns=96, nth=None, nze=None):
        super().__init__(bx)
        mboz = int(np.max(np.abs(self.xm))) or 8
        nboz = int(np.max(np.abs(self.xn)) / max(self.nfp, 1)) or 8
        self.NS = int(ns)
        self.NTH = int(nth) if nth else max(96, 6 * mboz)
        self.NZE = int(nze) if nze else max(96, 6 * nboz)

        s_grid = np.linspace(0.0, 1.0, self.NS)                    # [0,1] (spline extrapolates)
        th = np.linspace(0.0, 2 * np.pi, self.NTH, endpoint=False)
        ze = np.linspace(0.0, 2 * np.pi / self.nfp, self.NZE, endpoint=False)
        # cos/sin(xm*theta - xn*zeta) on the angle grid, shared across s: (NTH,NZE,mnboz)
        ang = (np.multiply.outer(th, self.xm)[:, None, :]
               - np.multiply.outer(ze, self.xn)[None, :, :])
        cos_ang = np.cos(ang)
        bmnc_s = self._bmnc_sp(s_grid)                             # (NS, mnboz)
        Bgrid = np.einsum("tzm,sm->stz", cos_ang, bmnc_s)         # (NS,NTH,NZE)
        if self.asym:
            Bgrid += np.einsum("tzm,sm->stz", np.sin(ang), self._bmns_sp(s_grid))
        del ang, cos_ang

        # Prefilter to b-spline coefficients: periodic in theta/zeta, clamped in s.
        coef = spline_filter1d(Bgrid, order=3, axis=1, mode="grid-wrap")
        coef = spline_filter1d(coef, order=3, axis=2, mode="grid-wrap")
        coef = spline_filter1d(coef, order=3, axis=0, mode="nearest")
        # pad s axis with edge values so the grid-wrap gather never wraps s (queries
        # stay >= _PAD cells from either end -> the cubic stencil stays interior).
        self._coef = np.pad(coef, ((self._PAD, self._PAD), (0, 0), (0, 0)), mode="edge")

    def modB(self):
        s = np.clip(self._pts[:, 0], 0.0, 1.0)
        th = self._pts[:, 1] % (2 * np.pi)
        ze = self._pts[:, 2] % (2 * np.pi / self.nfp)
        fs = s * (self.NS - 1) + self._PAD
        fth = th / (2 * np.pi) * self.NTH
        fze = ze / (2 * np.pi / self.nfp) * self.NZE
        B = map_coordinates(self._coef, np.vstack([fs, fth, fze]),
                            order=3, mode="grid-wrap", prefilter=False)
        return B.reshape(-1, 1)


# ================================================================ PART C
#   FieldBundle -- only __init__ differs from the published FieldBundle; every
#   other method (profiles/gII/modB/modB_at/surface_B_range/helicity/line_B) is
#   INHERITED byte-identical.
# ================================================================
class BoozerXformFieldBundle(_FirmFieldBundle):
    def __init__(self, bx, field_backend="grid", grid_res=None):
        if field_backend == "fourier":
            self.field = _RawBoozerFourier(bx)
        elif field_backend == "grid":
            kw = {} if grid_res is None else dict(zip(("ns", "nth", "nze"), grid_res))
            self.field = _RawBoozerGrid(bx, **kw)
        else:
            raise ValueError(f"field_backend must be 'grid' or 'fourier', got {field_backend!r}")
        self.nfp = int(bx.nfp)
        self.psi0 = float(-np.asarray(bx.phi, float)[-1] / TWO_PI)   # VMEC sign conv
        self.Psi_LCFS = abs(self.psi0)
        self._prof = {}
        self._helicity = None


# ================================================================ PART D
#   Markers (mirror of Gamma_W_final.make_surface_markers, firm3d-free)
# ================================================================
def make_surface_markers(field, n, s0, seed=7):
    """Equal-weight markers, volume-uniform on surface s0, xi uniform in [-1,1]."""
    pts = initialize_position_uniform_surf(field.field, n, s0, seed=seed)
    vpar = initialize_velocity_uniform(V0_ALPHA, n, seed=seed + 1)
    _, _, iota = field.profiles(s0)
    return [dict(marker_id=i, s0=float(s0), theta0=float(pts[i, 1]), zeta0=float(pts[i, 2]),
                 alpha0=float(pts[i, 1]) - iota * float(pts[i, 2]),
                 pitch_xi=float(vpar[i] / V0_ALPHA), weight=1.0, base_weight=1.0)
            for i in range(n)]


# ================================================================ PART E
#   Source resolver: simsopt Boozer | booz_xform object | boozmn path | simsopt Vmec
# ================================================================
def _resolve_bx(source, mpol=16, ntor=16, surfaces=None):
    if all(hasattr(source, a) for a in ("bmnc_b", "s_b", "Boozer_G_all", "phi")):
        return source                                             # booz_xform.Booz_xform
    if hasattr(source, "bx") and getattr(source, "bx") is not None:
        return source.bx                                          # simsopt.mhd.Boozer (run)
    if isinstance(source, str):
        import booz_xform
        bx = booz_xform.Booz_xform()
        bx.read_boozmn(source)
        return bx
    from simsopt.mhd import Boozer                                # simsopt Vmec
    vmec = source
    vmec.run()
    b = Boozer(vmec, mpol=mpol, ntor=ntor)
    b.register(vmec.s_half_grid if surfaces is None else surfaces)
    b.run()
    return b.bx


# ================================================================ PART F
#   Entry point -- same semantics as gamma_w_for_boozmn, firm3d-free
# ================================================================
def gamma_w_from_boozer(source, N=128, t_star=0.2, s0=0.3, seed=7, verbose=True,
                        detrap_fate=False, eta_ripple=0.02, branch_continue=True,
                        mpol=16, ntor=16, surfaces=None, dt_max=None,
                        field_backend="grid", grid_res=None):
    """End-to-end Gamma_W for one Boozer equilibrium via simsopt/booz_xform (no
    firm3d). `source`: simsopt.mhd.Boozer (run) | booz_xform object | boozmn path |
    simsopt Vmec. field_backend='grid' (fast, default) or 'fourier' (exact). dt_max
    None -> max(1e-3, t_star/50); pass 4e-3 to match the Alex campaign."""
    bx = _resolve_bx(source, mpol=mpol, ntor=ntor, surfaces=surfaces)
    field = BoozerXformFieldBundle(bx, field_backend=field_backend, grid_res=grid_res)
    P = GWParams(t_star=t_star, dt_max=(max(1e-3, t_star / 50) if dt_max is None else dt_max),
                 detrap_fate=detrap_fate, eta_ripple=eta_ripple,
                 branch_continue=branch_continue)
    markers = make_surface_markers(field, N, s0, seed=seed)
    kap = kappa(field.Psi_LCFS)
    bcache = {}
    rows = [run_marker(field, m, P, kap, bcache) for m in markers]
    agg = aggregate(rows)
    agg.update(nfp=field.nfp, Psi_LCFS=field.Psi_LCFS, N=N, t_star=t_star, s0=s0,
               detrap_fate=detrap_fate, eta_ripple=eta_ripple, branch_continue=branch_continue,
               backend=f"simsopt_booz_xform:{field_backend}")
    if verbose:
        print(f"  [simsopt/booz_xform:{field_backend}] nfp={field.nfp} Psi_LCFS={field.Psi_LCFS:.4f}"
              f"  N={N} t*={t_star}s s0={s0} branch_continue={branch_continue}")
        print(f"  Gamma_W = {agg['Gamma_W']:.5f}   L_W = {agg['L_W']:.4f}   [{agg['regime']}]")
        print(f"  band [low,high] = [{agg['Gamma_W_low']:.5f}, {agg['Gamma_W_high']:.5f}]")
        print(f"  passing_frac = {agg['passing_frac']:.3f}  "
              f"unresolved = {agg['unresolved_w_frac']:.3f}  branch_event = {agg['branch_event_w_frac']:.3f}")
    return agg


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Gamma_W via simsopt/booz_xform (no firm3d).")
    ap.add_argument("boozmn", help="boozmn .nc (or a VMEC wout booz_xform can read)")
    ap.add_argument("--N", type=int, default=128)
    ap.add_argument("--t-star", type=float, default=0.2)
    ap.add_argument("--s0", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--backend", choices=["grid", "fourier"], default="grid")
    ap.add_argument("--keep", action="store_true", help="disable branch_continue")
    a = ap.parse_args()
    gamma_w_from_boozer(a.boozmn, N=a.N, t_star=a.t_star, s0=a.s0, seed=a.seed,
                        branch_continue=not a.keep, field_backend=a.backend)
