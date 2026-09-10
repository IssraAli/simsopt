#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
================================================================================
Gamma_W_final.py  --  STANDALONE, verifiable implementation of the Whitham
                      energetic-particle confinement proxy  Gamma_W.
================================================================================

This single file reproduces the entire Gamma_W engine used to evaluate the proxy
on the Landreman, Paul and Alex stellarator datasets. It is a flattened, heavily
commented version of the `scripts/gamma_w/` package -- the numerics are identical.

--------------------------------------------------------------------------------
WHAT Gamma_W IS
--------------------------------------------------------------------------------
For a trapped fusion alpha launched on flux surface s0 with field-line label
alpha = theta - iota(s) zeta and absolute turning field  B_* = E/mu, define the
PHYSICAL FULL-BOUNCE second adiabatic invariant and bounce time on the field line
(Boozer arc length  dl = |G + iota I| / B  dzeta,  v_par = v0 sqrt(1 - B/B_*)):

    J_a(s,alpha)   = 2 v0 (G+iI) \int_well sqrt(1 - B/B_*) / B           dzeta   [m^2/s]
    tau_b(s,alpha) = (2/v0)(G+iI) \int_well 1 / (B sqrt(1 - B/B_*))      dzeta   [s]

The bounce-center drifts along the occupied-branch action level set; at fixed
(E, mu) the signed, time-parametrized characteristic is

    s_dot     = + kappa (1/tau_b) dJ/dalpha
    alpha_dot = - kappa (1/tau_b) dJ/ds
    kappa     = sign_conv * m_alpha / (Z_alpha e Psi_LCFS),   Psi_LCFS = |psi0| = Phi_edge/2pi

(normalization resolved FROM DEFINITIONS, not fitted; sign_conv = +1 fixed once by
the Landreman gate). Per marker, the normalized radial reach to endpoint t_*:

    R_{W,i}(t_*) = clip[ (max_{0<=t<=t_*} s^W_i(t) - s0,i) / (1 - s0,i), 0, 1 ]

and the single configuration objective (w_i = source quadrature weight):

    Gamma_W(t_*) = sum_i w_i R_{W,i} / sum_i w_i        [0 = stays put, 1 = whole source reaches edge]

L_W = sum_i w_i 1[s_max>=1]/sum_i w_i is a DIAGNOSTIC only (too binary to headline).

--------------------------------------------------------------------------------
VERIFICATION
--------------------------------------------------------------------------------
    python Gamma_W_final.py --selftest
        Pure-numpy/scipy check: the turning-point-aware bounce quadrature vs
        scipy.quad on parabolic and cosine model wells (no firm3d needed).
    python Gamma_W_final.py --boozmn FILE.nc [--N 128] [--t-star 0.2] [--s0 0.3]
        End-to-end Gamma_W for one Boozer equilibrium (needs firm3d + a boozmn
        built with booz_xform flux=True; see build_alex_boozmn.py).

--------------------------------------------------------------------------------
HEADLINE RESULT (converged; see Gamma_W_final.md for the full account)
--------------------------------------------------------------------------------
Gamma_W's leading-order ADIABATIC reach captures the loss MECHANISM (Landreman
gate PASS) and the AMPLITUDE scaling (Alex sigma-bin-median rank 0.985), but is
NOT a quantitatively competitive predictor: on the full Alex 250 it trails the QS
benchmark (Spearman 0.656 vs 0.736, paired bootstrap p<0.05) and on Paul it
under-predicts trapped-channel QA banana loss. The missing quantitative ingredient
is the finite-orbit-width / non-adiabatic drift (traced V_out), which this
field-only proxy omits by construction.

--------------------------------------------------------------------------------
PATCH LOG
--------------------------------------------------------------------------------
2026-07-13  (md5 9de057ad -> bccb469c)  P1/B2 status propagation, per
    HANDOVER_CODE_jul13.md. Previously, a numerical FD-branch mismatch (`_rhs` returns a
    valid `base` with status FD_MISMATCH) fell through the `base is None` test, and `_rk4`
    returned bare None on the resulting substage failure -- which under branch_continue was
    mapped to DETRAP_PASSING and scored CONFINED (a numerical failure disguised as a physical
    outcome, outside the sensitivity band). Now any non-"ok" status stops the marker,
    preserves s_max, and is flagged UNRESOLVED with its originating status; only a genuinely
    empty descendant well (base is None) yields DETRAP_PASSING. `_rk4` returns
    (result, status, base_none) so the caller distinguishes the two. keep-mode Gamma_W / R_W /
    band are numerically UNCHANGED (verified); bc changes only where an FD mismatch fired
    inside a substep (rank-neutral on the paper datasets).
2026-07-13  (same round)  DIAGNOSTIC-ONLY: added GWResult.n_cont_rescues (+ its per-marker
    row field). `cont_flag` is reset on every rhs_cont call and read only at the top of the
    loop, so a continuation that rescues inside an RK4 substage never reached `n_branch += 1`
    -- the mechanism behind historical "trajectory changed yet n_branch == 0". n_cont_rescues
    is a PERSISTENT tally of every rescue (top-of-loop, dt_floor hand-off, AND RK4 substage),
    so n_cont_rescues >= n_branch always. It is never read by control flow -> all scores
    (keep and bc) are byte-identical to the P1-only engine; only a new diagnostic column
    appears. Regenerated comparison tables (R1) should report n_cont_rescues, not n_branch.
2026-08-19  (from md5 575187cf; the resulting md5 is pinned in tests/conftest.py and in the
    package MANIFEST.sha256, so this entry carries no self-referential hash)  DOCSTRING-ONLY.
    The gamma_w_for_boozmn docstring quoted a rippled-tokamak Spearman improvement for
    branch_continue that rests on a single exploratory run recorded only in working notes; no
    frozen artifact carries it (the frozen gamma_w_n{2,3}_fate.csv hold the keep and
    delta_QS-fate columns only, with no branch-continued column). It is replaced by the
    artifact-backed de-trapping-fate numbers. NO EXECUTABLE CODE CHANGED: the diff is confined
    to that docstring and to this log entry, so every score, band, status and diagnostic is
    byte-identical to 575187cf. conftest ENGINE_HASH is updated in the same change.
================================================================================
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field as dcfield

import numpy as np

# ------------------------------------------------------------------ constants
# Physical constants (SI). Imported from firm3d when available so results match
# the campaign exactly; otherwise the standard values below (used only for the
# absolute drift scale -- the self-test and topology are constant-independent).
try:
    from firm3d.util.constants import (ALPHA_PARTICLE_MASS, ALPHA_PARTICLE_CHARGE,
                                        ELEMENTARY_CHARGE, FUSION_ALPHA_PARTICLE_ENERGY)
    M_ALPHA = float(ALPHA_PARTICLE_MASS)
    Q_ALPHA = float(ALPHA_PARTICLE_CHARGE)
    E_CHARGE = float(ELEMENTARY_CHARGE)
    E_ALPHA = float(FUSION_ALPHA_PARTICLE_ENERGY)
except Exception:                                    # standalone fallback (firm3d's
    # exact constants, so the standalone reproduces the campaign bit-for-bit;
    # firm3d's alpha mass is ~0.76% above the CODATA value, kept for reproducibility)
    M_ALPHA = 6.69509884346e-27                       # alpha mass [kg] (firm3d)
    E_CHARGE = 1.602176634e-19                        # [C]
    Q_ALPHA = 3.204353268e-19                         # alpha charge [C] (firm3d, = 2e)
    E_ALPHA = 5.639661751679999e-13                   # 3.52 MeV [J] (firm3d)

Z_ALPHA = Q_ALPHA / E_CHARGE                          # = 2
V0_ALPHA = float(np.sqrt(2.0 * E_ALPHA / M_ALPHA))    # alpha speed [m/s]
SIGN_CONV = +1.0                                      # frozen by the Landreman gate
TWO_PI = 2.0 * np.pi


def kappa(Psi_LCFS, sign_conv=SIGN_CONV):
    """Whitham drift normalization kappa = sign_conv * m_alpha/(Z_alpha e |psi0|)."""
    return sign_conv * M_ALPHA / (Z_ALPHA * E_CHARGE * abs(Psi_LCFS))


def Bstar_from_pitch(B_launch, pitch_xi):
    """Absolute turning field B_* = E/mu = B_launch / (1 - xi^2) in the field's
    own |B| units (xi = v_par/v at launch). Computed from the field's own modB so
    it never depends on stored/Gauss units."""
    return float(B_launch) / max(1.0 - float(pitch_xi) ** 2, 1e-12)


# ================================================================ PART 1
#   Turning-point-aware full-bounce quadrature  (pure numpy; the verifiable core)
# ================================================================
@dataclass
class WellIntegral:
    J_geom: float        # \int_well sqrt(g) w dzeta     (geometric; J = 2 v0 * J_geom)
    T_geom: float        # \int_well (w/sqrt(g)) dzeta    (geometric; tau_b = (2/v0) * T_geom)
    zeta_L: float
    zeta_R: float
    zeta_min: float
    B_min: float
    n_inside: int


def _root_cross(z0, z1, b0, b1, B_star):
    if b1 == b0:
        return 0.5 * (z0 + z1)
    return z0 + (B_star - b0) / (b1 - b0) * (z1 - z0)


def integrate_well(zeta, B, gII, B_star, i0, i1, kmin):
    r"""Bounce integrals for the well occupying samples [i0,i1) with minimum kmin.

    g = 1 - B/B_* vanishes LINEARLY at the turning points, so the J integrand
    ~ sqrt(g) is finite but the tau_b integrand ~ 1/sqrt(g) is an integrable
    singularity. We integrate per-cell assuming g linear and the smooth prefactor
    w = (G+iI)/B cell-constant -- which captures the endpoint sqrt-behavior
    analytically (no large floor). The exact turning points (g=0) are inserted by
    linear root of B = B_*.  Fully vectorized.
    """
    n = len(B)
    a = kmin
    while a - 1 >= i0 and B[a - 1] < B_star:
        a -= 1
    b = kmin
    while b + 1 < i1 and B[b + 1] < B_star:
        b += 1
    zL = _root_cross(zeta[a - 1], zeta[a], B[a - 1], B[a], B_star) if a - 1 >= 0 else zeta[a]
    zR = _root_cross(zeta[b], zeta[b + 1], B[b], B[b + 1], B_star) if b + 1 < n else zeta[b]

    zs = np.concatenate(([zL], zeta[a:b + 1], [zR]))
    Bs = np.concatenate(([B_star], B[a:b + 1], [B_star]))
    g = np.clip(1.0 - Bs / B_star, 0.0, None)
    w = gII / Bs

    dz = np.diff(zs)
    gi, gj = g[:-1], g[1:]
    wmid = 0.5 * (w[:-1] + w[1:])
    sgi, sgj = np.sqrt(gi), np.sqrt(gj)
    valid = dz > 0
    mg = np.where(valid, (gj - gi) / np.where(valid, dz, 1.0), 0.0)
    small = np.abs(mg) < 1e-14
    gmid = np.clip(0.5 * (gi + gj), 0.0, None)
    sgmid = np.sqrt(gmid)
    safe_mg = np.where(small, 1.0, mg)
    J_cells = np.where(small, wmid * sgmid * dz,
                       wmid * (2.0 / 3.0) * (gj * sgj - gi * sgi) / safe_mg)
    with np.errstate(divide="ignore", invalid="ignore"):
        T_lin = np.where(gmid > 0, wmid * dz / np.where(gmid > 0, sgmid, 1.0), 0.0)
    T_cells = np.where(small, T_lin, wmid * 2.0 * (sgj - sgi) / safe_mg)
    J_geom = float(np.sum(np.where(valid, J_cells, 0.0)))
    T_geom = float(np.sum(np.where(valid, T_cells, 0.0)))

    if a < kmin < b:                                   # parabolic well-bottom (diagnostic)
        bm, b0, bp = B[kmin - 1], B[kmin], B[kmin + 1]
        den = bm - 2 * b0 + bp
        if den > 0:
            zmin_i = zeta[kmin] + 0.5 * (bm - bp) / den * (zeta[kmin] - zeta[kmin - 1])
            Bmin_i = b0 - 0.125 * (bm - bp) ** 2 / den
        else:
            zmin_i, Bmin_i = zeta[kmin], b0
    else:
        zmin_i, Bmin_i = zeta[kmin], B[kmin]
    return WellIntegral(J_geom, T_geom, float(zL), float(zR), float(zmin_i),
                        float(Bmin_i), int(b - a + 1))


def physical_JT(wi: WellIntegral, v0):
    """Geometric -> physical full-bounce J [m^2/s], tau_b [s]."""
    return 2.0 * v0 * wi.J_geom, (2.0 / v0) * wi.T_geom


# ================================================================ PART 2
#   Field-line wells and OCCUPIED-branch tracking by continuity
# ================================================================
def find_wells(zeta, B, B_star):
    """Interior trapped wells (contiguous B < B_* with barriers on both sides).
    Returns list of (i0, i1, kmin)."""
    below = (B < B_star).astype(np.int8)
    n = len(B)
    d = np.diff(below)
    starts = list(np.flatnonzero(d == 1) + 1)
    ends = list(np.flatnonzero(d == -1) + 1)
    if below[0]:
        starts = [0] + starts
    if below[-1]:
        ends = ends + [n]
    wells = []
    for i0, i1 in zip(starts, ends):
        if i0 > 0 and i1 < n:
            wells.append((i0, i1, i0 + int(np.argmin(B[i0:i1]))))
    return wells


@dataclass
class OccResult:
    ok: bool
    J: float
    tau: float
    zeta_min: float
    zeta_L: float
    zeta_R: float
    B_min: float
    n_wells: int
    reason: str = ""


def occupied_action(field, s, alpha, B_star, v0, zeta_hint, n_periods=8,
                    pts_per_period=160, iota=None, prev_interval=None):
    """Action/bounce-time of the OCCUPIED well at (s,alpha): the well whose center
    is nearest zeta_hint (continuity), optionally requiring overlap with the
    previous bounce interval. Branch disappearance (no overlapping descendant) is
    flagged ok=False -- an EVENT, never a silent hop."""
    zeta, B, gII = field.line_B(s, alpha, zeta_center=zeta_hint, n_periods=n_periods,
                                pts_per_period=pts_per_period, iota=iota)
    wells = find_wells(zeta, B, B_star)
    if not wells:
        return OccResult(False, np.nan, np.nan, zeta_hint, np.nan, np.nan, np.nan, 0, "no_well")
    cands = []
    for (i0, i1, kmin) in wells:
        wi = integrate_well(zeta, B, gII, B_star, i0, i1, kmin)
        if not (np.isfinite(wi.J_geom) and wi.J_geom > 0):
            continue
        overlap = True
        if prev_interval is not None:
            overlap = (min(wi.zeta_R, prev_interval[1]) - max(wi.zeta_L, prev_interval[0])) > 0
        cands.append((abs(wi.zeta_min - zeta_hint), overlap, wi))
    if not cands:
        return OccResult(False, np.nan, np.nan, zeta_hint, np.nan, np.nan, np.nan, len(wells), "no_valid_well")
    overlapping = [c for c in cands if c[1]]
    if prev_interval is not None and not overlapping:
        return OccResult(False, np.nan, np.nan, zeta_hint, np.nan, np.nan, np.nan, len(wells), "branch_lost_no_overlap")
    dz, _, wi = min(overlapping if (prev_interval is not None and overlapping) else cands, key=lambda c: c[0])
    J, tau = physical_JT(wi, v0)
    return OccResult(True, J, tau, wi.zeta_min, wi.zeta_L, wi.zeta_R, wi.B_min, len(wells), "ok")


# ================================================================ PART 3
#   Signed Whitham characteristic ODE + branch-event guard ladder + R_W
# ================================================================
OK_CONFINED, OK_LOST = "ok_confined", "ok_lost"
BRANCH_EVENT, TAU_GUARD = "branch_event_unresolved", "tau_guard_unresolved"
FD_MISMATCH, STEP_CAP = "fd_branch_mismatch_unresolved", "step_cap_unresolved"
FIELD_FAILED, NO_LAUNCH, PASSING = "field_eval_failed", "no_launch", "passing"
RIPPLE_LOST = "ripple_lost"        # de-trapping fate (b): ripple-trapped -> lost, R_W=1
DETRAP_PASSING = "detrap_passing"  # branch-continue: de-trapped to passing (no descendant well)
_BC_TRACE = None                   # debug: set to [] to record (n_acc, s, sd, ad, cont_step)
UNRESOLVED = {BRANCH_EVENT, TAU_GUARD, FD_MISMATCH, STEP_CAP, FIELD_FAILED, NO_LAUNCH}
EVENT = {BRANCH_EVENT, TAU_GUARD, FD_MISMATCH, STEP_CAP}
LOST = {OK_LOST, RIPPLE_LOST}      # contribute reached_edge / R_W=1


@dataclass
class GWParams:
    n_periods: int = 8
    pts_per_period: int = 160
    dalpha: float = 2.0e-3
    ds_fd: float = 2.0e-3
    t_star: float = 1.0e-2
    dt_max: float = 1.0e-3
    ds_step_max: float = 0.02
    dalpha_step_max: float = 0.03
    dt_floor_frac: float = 1.0e-5
    max_accepted_steps: int = 1200
    s_edge: float = 1.0
    s_floor: float = 1.0e-3
    eta_B: float = 1.0e-4              # well-depth guard:  B_*-B_min < eta_B (B_max-B_min)
    eta_tau: float = 1.0e-3           # tau guard:  tau_b < eta_tau * median(history)
    width_grid_factor: float = 4.0    # trapped width < width_grid_factor * dzeta_grid
    tau_hist_len: int = 12
    desc_steps: int = 5
    turn_frac: float = 0.25
    rise_eps: float = 0.02
    stall_steps: int = 80
    # (b) de-trapping FATE classifier (default OFF -> identical to the published metric).
    # When a trapped orbit de-traps, decide lost vs confined from the LOCAL quasi-symmetry
    # breaking delta_QS = |B| variation along the QS-invariant direction (helicity-aware,
    # see _detrap_ripple): a de-trapped orbit in a region where the relevant symmetry is
    # broken becomes ripple/symmetry-trapped, whose vertical grad-B drift does not
    # bounce-average to zero -> lost (R_W=1). For a tokamak/QA (helicity N=0) delta_QS is
    # exactly the toroidal ripple; for QH it follows the helix, so good QH reads ~0 and is
    # left untouched. Safe to enable on any geometry (default OFF keeps published numbers).
    detrap_fate: bool = False
    eta_ripple: float = 0.02          # delta_QS threshold (Bmax-Bmin)/(Bmax+Bmin) along QS line
    fate_nzeta: int = 64              #   (n=2 sweep: Spearman 0.89-0.93 over eta in [0.01,0.03])
    # SYMMETRY-AGNOSTIC de-trapping fate (branch-continued Whitham). Instead of guessing a
    # de-trapped orbit's fate from a single-helicity fit (delta_QS, which is meaningless on
    # MIXED-symmetry configs -- no dominant helicity to break), FOLLOW the orbit: continue
    # the Whitham characteristic into the descendant (ripple/secondary) well and let its own
    # bounce-averaged radial drift decide lost vs confined. No symmetry assumption -> works
    # on QA, QH, axisymmetric AND mixed uniformly. Overrides detrap_fate when True.
    branch_continue: bool = False
    hop_zeta_window: float = 1.0       # descendant well must be within this many field periods
    fd_shrink: float = 1.0             # FD-step scale for the ripple-well drift (max-overlap tracked)
    max_branch_hops: int = 200         # cap continuation steps (ping-pong backstop)


@dataclass
class GWResult:
    status: str
    s0: float
    s_max: float
    R_W: float
    reached_edge: bool
    n_steps: int
    unresolved: bool
    branch_event: bool
    detrap_ripple: float = np.nan     # local toroidal ripple at a de-trapping event
    n_branch: int = 0                 # accepted continuation (Euler) steps taken; also the hop cap
    checkpoint_times: list = dcfield(default_factory=list)
    R_W_ck: list = dcfield(default_factory=list)
    n_cont_rescues: int = 0           # DIAGNOSTIC: total continuation rescues, INCLUDING those
    #   fired inside an RK4 substage (which n_branch misses). n_cont_rescues >= n_branch always;
    #   n_cont_rescues > 0 with n_branch == 0 is the "trajectory changed yet uncounted" case.


def _dzeta_grid(field, P):
    return (P.n_periods * TWO_PI / field.nfp) / (P.n_periods * P.pts_per_period - 1)


def _quality(field, s, alpha, B_star, v0, zh, P, bmin, bmax, dzg, tau_med):
    """Occupied-well eval + guard ladder. Returns (OccResult|None, status)."""
    base = occupied_action(field, s, alpha, B_star, v0, zh, P.n_periods, P.pts_per_period)
    if not base.ok:
        return None, BRANCH_EVENT
    if (B_star - base.B_min) < P.eta_B * (bmax - bmin):     # well-depth (de-trapping)
        return None, BRANCH_EVENT
    if (base.zeta_R - base.zeta_L) < P.width_grid_factor * dzg:
        return None, BRANCH_EVENT
    if not np.isfinite(base.tau) or base.tau <= 0:
        return None, TAU_GUARD
    if tau_med is not None and base.tau < P.eta_tau * tau_med:
        return None, TAU_GUARD
    return base, "ok"


def _rhs(field, s, alpha, B_star, v0, kap, zh, P, bmin, bmax, dzg, tau_med):
    base, st = _quality(field, s, alpha, B_star, v0, zh, P, bmin, bmax, dzg, tau_med)
    if base is None:
        return 0.0, 0.0, None, st
    # branch-preserving central differences at fixed B_* (= E/mu)
    zc, bint = base.zeta_min, (base.zeta_L, base.zeta_R)
    ev = [occupied_action(field, s, alpha + d, B_star, v0, zc, P.n_periods, P.pts_per_period, prev_interval=bint)
          for d in (P.dalpha, -P.dalpha)]
    ev += [occupied_action(field, s + d, alpha, B_star, v0, zc, P.n_periods, P.pts_per_period, prev_interval=bint)
           for d in (P.ds_fd, -P.ds_fd)]
    if any(not r.ok for r in ev):
        return 0.0, 0.0, base, FD_MISMATCH
    dJ_da = (ev[0].J - ev[1].J) / (2 * P.dalpha)
    dJ_ds = (ev[2].J - ev[3].J) / (2 * P.ds_fd)
    return kap / base.tau * dJ_da, -kap / base.tau * dJ_ds, base, "ok"


def _reach_at(ts, ss, s0, t_ck):
    ts, ss = np.asarray(ts), np.asarray(ss)
    m = ts <= t_ck
    smax = float(ss[m].max()) if m.any() else float(ss.max() if len(ss) else s0)
    return float(np.clip((smax - s0) / (1.0 - s0), 0.0, 1.0))


def _detrap_ripple(field, s, alpha, zeta_hint, P):
    """Local SYMMETRY-BREAKING field strength at a de-trapping point: the variation of
    |B| along the config's quasi-symmetry-invariant direction (dtheta,dzeta) ~ (N, M),
    where chi = M theta - N zeta stays constant, so a perfectly quasi-symmetric field is
    flat and delta_QS = 0. delta_QS = (Bmax - Bmin)/(Bmax + Bmin) along that line.

    For quasi-axisymmetry (N=0: tokamak/QA) this is EXACTLY the toroidal ripple (vary
    zeta at fixed theta) -- so a rippled tokamak is scored on its raw ripple. For
    quasi-helical symmetry the line follows the helix, so a precise QH reads ~0 (its
    de-trapped orbits stay confined) and only genuine symmetry-breaking lifts delta_QS.
    A de-trapped orbit where delta_QS > eta_ripple is ripple/symmetry-trapped -> lost.

    This QS-aware generalization makes the fate classifier safe on stellarators: it
    fires on the symmetry that is actually broken, not on the benign helical |B| swing
    that every QH field has. (Verified: Landreman/Paul/Alex legacy rankings preserved;
    only a genuinely rippled QA config is reclassified.)"""
    M, N = field.helicity(s)
    _, _, iota = field.profiles(s)
    theta0 = alpha + iota * zeta_hint
    t = np.linspace(0.0, TWO_PI, P.fate_nzeta, endpoint=False)
    theta = theta0 + N * t            # move along (dtheta,dzeta) ~ (N, M): chi held fixed
    zeta = zeta_hint + M * t
    B = field.modB(np.full(P.fate_nzeta, s), theta, zeta)
    bmx, bmn = float(B.max()), float(B.min())
    return (bmx - bmn) / (bmx + bmn) if (bmx + bmn) > 0 else 0.0


def _occ_nearest(field, s, alpha, B_star, v0, zh, P, window, exclude=None):
    """The nearest VALID trapped well to zh within +-window in zeta, with the depth/width
    guards RELAXED (shallow, narrow ripple wells are allowed -- they are exactly the
    descendant a de-trapped orbit falls into). If `exclude=(zL,zR)` is given, any well whose
    minimum lies inside that interval is skipped (used to hand OFF a pinching well to its
    neighbour). Returns (wi, J, tau) or None. Symmetry-agnostic: reads the field-line well
    structure directly, no helicity assumed."""
    zeta, B, gII = field.line_B(s, alpha, zeta_center=zh, n_periods=P.n_periods,
                                pts_per_period=P.pts_per_period)
    wells = find_wells(zeta, B, B_star)
    best, bestd = None, window
    for (i0, i1, kmin) in wells:
        wi = integrate_well(zeta, B, gII, B_star, i0, i1, kmin)
        if not (np.isfinite(wi.J_geom) and wi.J_geom > 0):
            continue
        if (B_star - wi.B_min) <= 0:                 # not actually trapped here
            continue
        if exclude is not None and exclude[0] <= wi.zeta_min <= exclude[1]:
            continue                                  # skip the pinching well itself
        d = abs(wi.zeta_min - zh)
        if d < bestd:
            bestd, best = d, wi
    if best is None:
        return None
    J, tau = physical_JT(best, v0)
    if not (np.isfinite(tau) and tau > 0 and np.isfinite(J)):
        return None
    occ = OccResult(True, J, tau, best.zeta_min, best.zeta_L, best.zeta_R,
                    best.B_min, 1, "continued")
    return occ, J, tau


def _continue_branch(field, s, alpha, B_star, v0, kap, zh, P, bmin, bmax, exclude=None):
    """Branch-continued Whitham drift: at a de-trapping event, find the descendant well the
    orbit transitions into and return its bounce-averaged radial drift (sd, ad, base). The
    drift is the SAME Whitham s_dot=kappa/tau dJ/dalpha, but evaluated with a self-anchoring,
    shrunk finite difference so it can track the fragile ripple well (each stencil point
    re-picks its own nearest well within the window). `exclude=(zL,zR)` hands a pinching well
    off to its neighbour. Returns None if there is no descendant well within the window --
    i.e. the orbit de-trapped to PASSING (confined). Symmetry-agnostic (no helicity fit)."""
    window = P.hop_zeta_window * (TWO_PI / field.nfp)
    r0 = _occ_nearest(field, s, alpha, B_star, v0, zh, P, window, exclude=exclude)
    if r0 is None:
        return None
    base, J0, tau0 = r0
    zc, bint = base.zeta_min, (base.zeta_L, base.zeta_R)
    da, ds = P.dalpha * P.fd_shrink, P.ds_fd * P.fd_shrink

    def Jover(sx, ax):
        """J of the well that MOST overlaps the descendant's zeta-interval bint (tight
        continuity -> does NOT jump to an adjacent ripple well). None if the branch is gone."""
        zeta, B, gII = field.line_B(sx, ax, zeta_center=zc, n_periods=P.n_periods,
                                    pts_per_period=P.pts_per_period)
        best_ov, bestJ = 0.0, None
        for (i0, i1, kmin) in find_wells(zeta, B, B_star):
            wi = integrate_well(zeta, B, gII, B_star, i0, i1, kmin)
            if not (np.isfinite(wi.J_geom) and wi.J_geom > 0 and (B_star - wi.B_min) > 0):
                continue
            ov = min(wi.zeta_R, bint[1]) - max(wi.zeta_L, bint[0])
            if ov > best_ov:
                best_ov, bestJ = ov, 2.0 * v0 * wi.J_geom
        return bestJ

    Jap, Jam = Jover(s, alpha + da), Jover(s, alpha - da)
    Jsp, Jsm = Jover(s + ds, alpha), Jover(s - ds, alpha)
    if None in (Jap, Jam, Jsp, Jsm):
        return None
    dJ_da = (Jap - Jam) / (2 * da)
    dJ_ds = (Jsp - Jsm) / (2 * ds)
    return kap / tau0 * dJ_da, -kap / tau0 * dJ_ds, base


def integrate_characteristic(field, s0, alpha0, B_star, v0, kap, zeta0, P, bmin, bmax,
                             checkpoint_times=None):
    """Integrate the reduced characteristic to t_star with predictive single-shot
    stepping (dt set so BOTH |ds|<ds_step_max and |dalpha|<dalpha_step_max -- NO
    adaptive retries in singular regions) and the branch-event guard ladder."""
    dzg = _dzeta_grid(field, P)
    dt_floor = P.dt_floor_frac * P.t_star
    sdot_floor = P.ds_step_max / P.dt_max
    ckts = sorted(checkpoint_times) if checkpoint_times else [P.t_star]

    cont_flag = [False]     # control: did the LAST rhs_cont eval continue? (per-call, drives
    cont_calls = [0]        #   Euler-vs-RK4 + the hop cap). DIAGNOSTIC, separate: a PERSISTENT
                            #   tally of every continuation rescue, including those fired inside
                            #   an RK4 substage (which cont_flag, reset each call, cannot see) --
                            #   so n_cont_rescues > 0 flags continuation activity even when
                            #   n_branch == 0. Never read by control flow (scores unaffected).

    def rhs(s, a, zh, tm):
        return _rhs(field, s, a, B_star, v0, kap, zh, P, bmin, bmax, dzg, tm)

    def rhs_cont(s, a, zh, tm):
        """rhs, but if the primary Whitham branch guard trips, CONTINUE into the descendant
        (ripple/secondary) well instead of failing -- symmetry-agnostic de-trapping fate."""
        cont_flag[0] = False
        sd, ad, base, st = _rhs(field, s, a, B_star, v0, kap, zh, P, bmin, bmax, dzg, tm)
        if base is None and P.branch_continue:
            c = _continue_branch(field, s, a, B_star, v0, kap, zh, P, bmin, bmax)
            if c is not None:
                cont_flag[0] = True
                cont_calls[0] += 1        # persistent: counts substage rescues too (see above)
                return c[0], c[1], c[2], "ok"
        return sd, ad, base, st

    RHS = rhs_cont if P.branch_continue else rhs

    sd, ad, base, st = RHS(s0, alpha0, zeta0, None)
    if base is None or st != "ok":
        # base is None -> never trapped here (NO_LAUNCH); base valid but st != "ok" -> a
        # numerical (FD-branch) failure at launch, which is UNRESOLVED with its own status,
        # never mapped to a physical outcome (B2 status-propagation fix).
        status = NO_LAUNCH if st == BRANCH_EVENT else st
        return GWResult(status, s0, s0, 0.0, False, 0, status in UNRESOLVED,
                        status in EVENT, np.nan, 0, ckts, [0.0] * len(ckts),
                        n_cont_rescues=cont_calls[0])
    s, alpha, zh = float(s0), float(alpha0), base.zeta_min
    t, s_max, n_acc, desc, since, n_branch = 0.0, s, 0, 0, 0, 0
    ts, ss = [0.0], [s]
    tau_hist = [base.tau]
    status = OK_CONFINED

    vmax = P.ds_step_max / dt_floor     # drift ceiling so a clipped step keeps dt >= dt_floor
    amax = P.dalpha_step_max / dt_floor

    while t < P.t_star and n_acc < P.max_accepted_steps:
        tau_med = float(np.median(tau_hist))
        sd, ad, base, st = RHS(s, alpha, zh, tau_med)
        cont_step = cont_flag[0]        # this eval came from a branch continuation
        if base is None:
            # branch_continue: no descendant well -> de-trapped to PASSING (confined reach).
            status = DETRAP_PASSING if P.branch_continue else st
            break
        if st != "ok":
            # base is valid but the drift eval failed numerically (FD branch mismatch). A
            # numerical failure is UNRESOLVED with its originating status and preserves s_max;
            # it must NOT fall through to _rk4 and be mis-scored DETRAP_PASSING under bc (B2).
            status = st
            break
        zh = base.zeta_min
        tau_hist.append(base.tau); tau_hist[:] = tau_hist[-P.tau_hist_len:]
        dt = min(P.ds_step_max / max(abs(sd), sdot_floor),
                 P.dalpha_step_max / max(abs(ad), P.dalpha_step_max / P.dt_max))
        if dt < dt_floor and not cont_step:
            # primary-branch drift blew up = separatrix / de-trapping. Hand the pinching well
            # OFF to a neighbouring (ripple/secondary) well and continue with its own drift.
            if not P.branch_continue:
                status = STEP_CAP; break
            c = _continue_branch(field, s, alpha, B_star, v0, kap, zh, P, bmin, bmax,
                                 exclude=(base.zeta_L, base.zeta_R))
            if c is None:
                status = DETRAP_PASSING; break         # de-trapped to passing (confined)
            sd, ad, base = c
            zh = base.zeta_min
            cont_step = True
            cont_calls[0] += 1        # this dt_floor rescue bypasses rhs_cont; count it so
            #                           n_cont_rescues stays a faithful superset of n_branch
        if cont_step:
            # ripple-tracking: the descendant-well FD drift is fragile (RK4 substeps re-blow
            # up), so clip it to the physical ceiling (sign = drift direction is what matters)
            # and take a robust single Euler step. Losses accrue by net outward reach; noisy
            # sign-flipping orbits stall -> confined (existing desc/stall guards).
            n_branch += 1
            if n_branch > P.max_branch_hops:
                status = STEP_CAP; break
            sd = float(np.clip(sd, -vmax, vmax)); ad = float(np.clip(ad, -amax, amax))
            dt = min(P.ds_step_max / max(abs(sd), sdot_floor),
                     P.dalpha_step_max / max(abs(ad), P.dalpha_step_max / P.dt_max))
            dt = min(dt, P.dt_max, P.t_star - t)
            s1, a1, zh1 = s + dt * sd, alpha + dt * ad, base.zeta_min
        else:
            dt = min(dt, P.dt_max, P.t_star - t)
            trial, fail_st, base_none = _rk4(RHS, s, alpha, zh, dt, (sd, ad, base))
            if trial is None:
                if base_none:            # a substage de-trapped with no descendant well: physical
                    status = DETRAP_PASSING if P.branch_continue else BRANCH_EVENT
                else:                    # a substage failed NUMERICALLY (FD mismatch): UNRESOLVED,
                    status = fail_st if P.branch_continue else BRANCH_EVENT   # never DETRAP_PASSING
                break
            s1, a1, zh1 = trial
        if s1 >= P.s_edge:
            frac = (P.s_edge - s) / (s1 - s) if s1 != s else 1.0
            t += frac * dt; s = P.s_edge; s_max = max(s_max, s)
            ts.append(t); ss.append(s); status = OK_LOST
            break
        if s1 < P.s_floor:
            status = OK_CONFINED
            break
        if _BC_TRACE is not None:                      # debug hook (set _BC_TRACE=[] to record)
            _BC_TRACE.append((n_acc, float(s), float(sd), float(ad), bool(cont_step), float(base.B_min)))
        prev = s
        s, alpha, zh = s1, a1, zh1
        t += dt; n_acc += 1
        if s > s_max:
            s_max = s; since = 0
        else:
            since += 1
        ts.append(t); ss.append(s)
        rise = s_max - s0
        if rise > P.rise_eps and s < s_max - max(0.01, P.turn_frac * rise):
            desc += 1
        elif s >= prev:
            desc = 0
        if desc >= P.desc_steps or since >= P.stall_steps:
            status = OK_CONFINED; break
        if t > 0.5 * P.t_star and rise < 0.5 * P.rise_eps:
            status = OK_CONFINED; break
    if n_acc >= P.max_accepted_steps and status == OK_CONFINED:
        status = STEP_CAP

    # (b) de-trapping fate: record the local ripple at every de-trapping event; with
    # fate on, reclassify de-trapped orbits in a corrugated region as RIPPLE_LOST (R_W=1).
    detrap_ripple = np.nan
    if status in EVENT:
        try:
            detrap_ripple = _detrap_ripple(field, s, alpha, zh, P)
        except Exception:
            detrap_ripple = np.nan
        if P.detrap_fate and detrap_ripple > P.eta_ripple:
            status = RIPPLE_LOST
            s_max = 1.0

    R_W_ck = [_reach_at(ts, ss, s0, tc) for tc in ckts]
    if status == RIPPLE_LOST:
        R_W_ck = [1.0 for _ in ckts]
    R_W = 1.0 if status == RIPPLE_LOST else (
        R_W_ck[-1] if (checkpoint_times and ckts[-1] >= P.t_star - 1e-15)
        else float(np.clip((s_max - s0) / (1.0 - s0), 0.0, 1.0)))
    return GWResult(status, s0, float(s_max), R_W, status in LOST, n_acc,
                    status in UNRESOLVED, status in EVENT, detrap_ripple, n_branch, ckts, R_W_ck,
                    n_cont_rescues=cont_calls[0])


def _rk4(rhs, s, alpha, zh, dt, k1):
    """One RK4 step. Returns (result, status, base_none):
      * success -> ((s1, a1, zh1), "ok", False)
      * failure -> (None, <substage status>, base_none), where base_none distinguishes a
        substage with NO occupied well (base_none=True: a physical de-trapping) from one whose
        drift status is non-ok (base_none=False: a NUMERICAL failure, e.g. FD_MISMATCH). The
        caller must not map the numerical case to a physical outcome (B2)."""
    k1s, k1a, b1 = k1[0], k1[1], k1[2]
    s2 = rhs(s + 0.5 * dt * k1s, alpha + 0.5 * dt * k1a, b1.zeta_min, None)
    if s2[2] is None:
        return None, s2[3], True
    if s2[3] != "ok":
        return None, s2[3], False
    s3 = rhs(s + 0.5 * dt * s2[0], alpha + 0.5 * dt * s2[1], s2[2].zeta_min, None)
    if s3[2] is None:
        return None, s3[3], True
    if s3[3] != "ok":
        return None, s3[3], False
    s4 = rhs(s + dt * s3[0], alpha + dt * s3[1], s3[2].zeta_min, None)
    if s4[2] is None:
        return None, s4[3], True
    if s4[3] != "ok":
        return None, s4[3], False
    s1 = s + dt / 6.0 * (k1s + 2 * s2[0] + 2 * s3[0] + s4[0])
    a1 = alpha + dt / 6.0 * (k1a + 2 * s2[1] + 2 * s3[1] + s4[1])
    return (s1, a1, s4[2].zeta_min), "ok", False


# ================================================================ PART 4
#   Source-averaged aggregation:  Gamma_W = sum_i a_i R_i / sum_i b_i
# ================================================================
def aggregate(rows):
    a = np.array([float(r["weight"]) for r in rows])               # contribution weight
    b = np.array([float(r.get("base_weight", r["weight"])) for r in rows])  # source norm
    R = np.array([float(r["R_W"]) for r in rows])
    edge = np.array([1.0 if r.get("reached_edge") else 0.0 for r in rows])
    passing = np.array([bool(r.get("passing_at_birth")) for r in rows])
    unres = np.array([bool(r.get("unresolved", False)) for r in rows])
    bevent = np.array([bool(r.get("branch_event", False)) for r in rows])
    bt = b.sum()
    # (a) unresolved BAND + REGIME flag ("dictionary"). The fate of a de-trapped orbit
    # (lost vs confined) is the band ambiguity: low scores unresolved R_W=0 (pessimistic),
    # high scores R_W=1 (optimistic; correct when de-trapping -> ripple-trapping -> loss).
    gw_keep = float((a * R).sum() / bt)
    gw_low = float((a * np.where(unres, 0.0, R)).sum() / bt)
    gw_high = float((a * np.where(unres, 1.0, R)).sum() / bt)
    bev_frac = float(a[bevent].sum() / bt)
    pass_frac = float(1.0 - a[~passing].sum() / bt)
    if pass_frac > 0.85:
        regime = "passing-dominated"        # trapped-only Gamma_W is blind
    elif bev_frac > 0.20:
        regime = "de-trapping-dominated"    # Gamma_W under-resolved -> read the band
    else:
        regime = "adiabatic-resolved"       # Gamma_W(keep) is trustworthy
    return dict(
        Gamma_W=gw_keep,
        Gamma_W_low=gw_low,
        Gamma_W_high=gw_high,
        regime=regime,
        gamma_w_valid=(regime == "adiabatic-resolved"),
        L_W=float((a * edge).sum() / bt),
        n=len(rows),
        trapped_frac=float(a[~passing].sum() / bt),
        passing_frac=pass_frac,
        unresolved_w_frac=float(a[unres].sum() / bt),
        branch_event_w_frac=bev_frac,
    )


# ================================================================ PART 5
#   Field interface (firm3d InterpolatedBoozerField) -- the only external dep
# ================================================================
class FieldBundle:
    """Thin wrapper over firm3d's InterpolatedBoozerField. Reads nfp and
    psi0 (= Phi_edge/2pi) from the file; provides modB and field-line covers."""

    def __init__(self, path, ns=48, ntheta=48, nzeta=48, degree=3):
        from firm3d.field.boozermagneticfield import InterpolatedBoozerField
        self.field = InterpolatedBoozerField.from_booz_xform(path, degree=degree,
                                                             ns=ns, ntheta=ntheta, nzeta=nzeta)
        self.nfp = int(self.field.nfp)
        self.psi0 = float(self.field.psi0)
        self.Psi_LCFS = abs(self.psi0)
        self._prof = {}
        self._helicity = None

    def profiles(self, s):
        s = min(max(float(s), 1e-4), 1.0 - 1e-4)     # clamp to domain (firm3d segfaults OOB)
        k = round(float(s), 10)
        if k not in self._prof:
            self.field.set_points(np.array([[float(s), 0.0, 0.0]]))
            self._prof[k] = (float(np.asarray(self.field.G()).ravel()[0]),
                             float(np.asarray(self.field.I()).ravel()[0]),
                             float(np.asarray(self.field.iota()).ravel()[0]))
        return self._prof[k]

    def gII(self, s):
        G, I, io = self.profiles(s)
        return abs(G + io * I)

    def modB(self, s, theta, zeta):
        # clamp s into the interpolation domain: firm3d SEGFAULTS (does not raise) on
        # out-of-domain s, which the branch-continuation FD can request as it drifts near
        # the edge. theta,zeta are periodic (handled by firm3d). No-op for in-domain calls.
        s = np.clip(np.atleast_1d(np.asarray(s, float)), 1e-4, 1.0 - 1e-4)
        theta = np.atleast_1d(np.asarray(theta, float))
        zeta = np.atleast_1d(np.asarray(zeta, float))
        self.field.set_points(np.column_stack([s, theta, zeta]))
        return np.asarray(self.field.modB()).ravel()

    def modB_at(self, s, theta, zeta):
        return float(self.modB(s, theta, zeta)[0])

    def surface_B_range(self, s, nth=256, nze=256):
        th = np.linspace(0, TWO_PI, nth, endpoint=False)
        ze = np.linspace(0, TWO_PI, nze, endpoint=False)
        TH, ZE = np.meshgrid(th, ze, indexing="ij")
        b = self.modB(np.full(TH.size, s), TH.ravel(), ZE.ravel())
        return float(b.min()), float(b.max())

    def helicity(self, s=0.5, nth=48, nze=48):
        """Quasi-symmetry helicity (M, N): the integers for which |B| is most nearly
        a function of chi = M*theta - N*zeta only (B constant along the QS-invariant
        direction (dtheta,dzeta) ~ (N, M)). Found convention-free by minimizing the
        within-chi-bin spread of B over candidate (M,N). N=0 => quasi-axisymmetry
        (the tokamak/QA case, where the QS direction is pure zeta = raw toroidal ripple);
        N=k*nfp => quasi-helical symmetry. Cached per field."""
        if self._helicity is not None:
            return self._helicity
        th = np.linspace(0, TWO_PI, nth, endpoint=False)
        ze = np.linspace(0, TWO_PI, nze, endpoint=False)
        TH, ZE = np.meshgrid(th, ze, indexing="ij")
        B = self.modB(np.full(TH.size, s), TH.ravel(), ZE.ravel()).reshape(nth, nze)
        cand = [(1, 0)] + [(M, k * self.nfp) for M in (1, 2)
                           for k in (1, -1, 2, -2, 3, -3, 4, -4)]
        nb = 24
        best, best_spread = (1, 0), np.inf
        for (M, N) in cand:
            chi = (M * TH - N * ZE) % TWO_PI
            idx = np.clip((chi / TWO_PI * nb).astype(int), 0, nb - 1)
            spread = 0.0
            for k in range(nb):
                vals = B[idx == k]
                if vals.size > 1:
                    spread += float(vals.std())
            if spread < best_spread:
                best_spread, best = spread, (M, N)
        self._helicity = best
        return best

    def line_B(self, s, alpha, zeta_center=0.0, n_periods=8, pts_per_period=160, iota=None):
        if iota is None:
            _, _, iota = self.profiles(s)
        gII = self.gII(s)
        L = n_periods * TWO_PI / self.nfp
        npts = n_periods * pts_per_period
        zeta = np.linspace(zeta_center - 0.5 * L, zeta_center + 0.5 * L, npts)
        B = self.modB(np.full(npts, s), alpha + iota * zeta, zeta)
        return zeta, B, gII


# ================================================================ PART 6
#   Deterministic isotropic source markers (firm3d launch convention)
# ================================================================
def make_surface_markers(field, n, s0, seed=7):
    """Equal-weight markers volume-uniform on surface s0, xi=v_par/v uniform in
    [-1,1] (isotropic) -- the same physical source FIRM3D/SIMPLE launch."""
    from firm3d.field.tracing_helpers import (initialize_position_uniform_surf,
                                              initialize_velocity_uniform)
    pts = initialize_position_uniform_surf(field.field, n, s0, seed=seed)
    vpar = initialize_velocity_uniform(V0_ALPHA, n, seed=seed + 1)
    _, _, iota = field.profiles(s0)
    return [dict(marker_id=i, s0=float(s0), theta0=float(pts[i, 1]), zeta0=float(pts[i, 2]),
                 alpha0=float(pts[i, 1]) - iota * float(pts[i, 2]),
                 pitch_xi=float(vpar[i] / V0_ALPHA), weight=1.0, base_weight=1.0)
            for i in range(n)]


def run_marker(field, m, P, kap, bcache):
    s0, a0, z0, th0, xi = m["s0"], m["alpha0"], m["zeta0"], m["theta0"], m["pitch_xi"]
    B0 = field.modB_at(s0, th0, z0)
    B_star = Bstar_from_pitch(B0, xi)
    if s0 not in bcache:
        bcache[s0] = field.surface_B_range(s0)
    bmin, bmax = bcache[s0]
    row = dict(m)
    if B_star >= bmax * (1 - 1e-6):                     # passing at birth
        row.update(status=PASSING, R_W=0.0, reached_edge=False, passing_at_birth=True,
                   unresolved=False, branch_event=False)
        return row
    res = integrate_characteristic(field, s0, a0 % TWO_PI, B_star, V0_ALPHA, kap, z0, P, bmin, bmax)
    row.update(status=res.status, R_W=res.R_W, reached_edge=res.reached_edge,
               passing_at_birth=False, unresolved=res.unresolved, branch_event=res.branch_event,
               detrap_ripple=res.detrap_ripple, n_branch=res.n_branch,
               n_cont_rescues=res.n_cont_rescues, s_max=res.s_max, n_steps=res.n_steps)
    return row


def gamma_w_for_boozmn(boozmn_path, N=128, t_star=0.2, s0=0.3, seed=7, verbose=True,
                       detrap_fate=False, eta_ripple=0.02, branch_continue=True):
    """End-to-end Gamma_W for one Boozer equilibrium.

    De-trapping treatment (how a de-trapped orbit's fate is decided):
    - branch_continue=True (DEFAULT, recommended): the SYMMETRY-AGNOSTIC branch-continued
      Whitham. At a de-trapping event it continues the characteristic into the descendant
      (ripple/secondary) well and lets its own bounce-averaged radial drift decide lost vs
      confined -- NO helicity fit, so it works uniformly on QA / QH / axisymmetric / MIXED
      symmetry. VALIDATED byte-identical to the published keep-metric on Alex-250 (Spearman
      0.656), Paul, and the Landreman mixed gate (0.701>0.394>0.076 -- loss there is
      adiabatic). On a rippled tokamak a branch-aware treatment is what removes the spurious
      de-trapping dip; the frozen-artifact numbers for that test are the de-trapping FATE
      result (Spearman vs loss 0.05 -> 0.93 at n=2 and 0.58 -> 0.96 at n=3, eta=0.02). A
      branch-continued equivalent of those two numbers has NOT been re-frozen on this engine
      and is deliberately not quoted here. Set branch_continue=False to recover the exact
      published keep-metric bit-for-bit.
    - detrap_fate=True: the older delta_QS fate classifier -- SUPERSEDED by branch_continue
      (delta_QS needs a dominant helicity, which does not exist on mixed symmetry; it also
      over-corrected QA configs, e.g. Paul ariescs). Kept for reference; leave OFF and do not
      combine with branch_continue (branch_continue resolves de-trapping before the fate step).
    The report always carries the band [Gamma_W_low, _high] and the regime flag (a)."""
    field = FieldBundle(boozmn_path)
    P = GWParams(t_star=t_star, dt_max=max(1e-3, t_star / 50),
                 detrap_fate=detrap_fate, eta_ripple=eta_ripple,
                 branch_continue=branch_continue)
    markers = make_surface_markers(field, N, s0, seed=seed)
    kap = kappa(field.Psi_LCFS)
    bcache = {}
    rows = [run_marker(field, m, P, kap, bcache) for m in markers]
    agg = aggregate(rows)
    agg.update(nfp=field.nfp, Psi_LCFS=field.Psi_LCFS, N=N, t_star=t_star, s0=s0,
               detrap_fate=detrap_fate, eta_ripple=eta_ripple, branch_continue=branch_continue)
    if verbose:
        print(f"  nfp={field.nfp} Psi_LCFS={field.Psi_LCFS:.4f}  N={N} t*={t_star}s s0={s0}"
              f"  branch_continue={branch_continue} fate={detrap_fate}")
        print(f"  Gamma_W = {agg['Gamma_W']:.5f}   L_W = {agg['L_W']:.4f}   "
              f"[{agg['regime']}]")
        print(f"  band [low,high] = [{agg['Gamma_W_low']:.5f}, {agg['Gamma_W_high']:.5f}]")
        print(f"  passing_frac = {agg['passing_frac']:.3f}  "
              f"unresolved = {agg['unresolved_w_frac']:.3f}  branch_event = {agg['branch_event_w_frac']:.3f}")
    return agg


# ================================================================ PART 7
#   Verification self-test (pure numpy/scipy; no firm3d)
# ================================================================
def selftest():
    import scipy.integrate as si
    from scipy.optimize import brentq

    def case(name, Bfun, B_star, gII, lo, hi, tolJ=2e-3, tolT=5e-3):
        z = np.linspace(lo, hi, 2001)
        b = Bfun(z)
        kmin = int(np.argmin(b))
        wi = integrate_well(z, b, gII, B_star, 0, len(z), kmin)
        zmin = z[kmin]
        zl = brentq(lambda x: Bfun(x) - B_star, lo, zmin)
        zr = brentq(lambda x: Bfun(x) - B_star, zmin, hi)
        IJ = si.quad(lambda x: np.sqrt(max(0, 1 - Bfun(x) / B_star)) * gII / Bfun(x), zl, zr,
                     points=[zl, zr], limit=200)[0]
        IT = si.quad(lambda x: gII / (Bfun(x) * np.sqrt(max(1e-300, 1 - Bfun(x) / B_star))), zl, zr,
                     points=[zl, zr], limit=200)[0]
        eJ, eT = abs(wi.J_geom - IJ) / abs(IJ), abs(wi.T_geom - IT) / abs(IT)
        ok = eJ < tolJ and eT < tolT
        print(f"  {name:22s} J err={eJ:.2e}  tau err={eT:.2e}  {'OK' if ok else 'FAIL'}")
        return ok

    print("Gamma_W_final self-test -- turning-point-aware bounce quadrature vs scipy.quad:")
    B0, gII = 5.0, 100.0
    allok = True
    allok &= case("parabolic shallow", lambda z: B0 * (1 + 0.5 * z**2), 5.05, gII, -1, 1)
    for q in (0.05, 0.3, 0.8):
        allok &= case(f"cosine q*={q}", lambda z: B0 * (1 + 0.5 * (1 - np.cos(z))),
                      B0 * (1 + q), gII, -np.pi, np.pi)
    print("  ALL PASS" if allok else "  *** SELF-TEST FAILED ***")
    return allok


def main():
    ap = argparse.ArgumentParser(description="Standalone Gamma_W (Whitham EP proxy).")
    ap.add_argument("--selftest", action="store_true", help="run the quadrature self-test (no firm3d)")
    ap.add_argument("--boozmn", type=str, default=None, help="boozmn .nc to score (needs firm3d)")
    ap.add_argument("--N", type=int, default=128)
    ap.add_argument("--t-star", type=float, default=0.2)
    ap.add_argument("--s0", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--keep", action="store_true",
                    help="disable branch_continue -> exact published keep-metric (default: branch_continue ON)")
    args = ap.parse_args()
    ran = False
    if args.selftest:
        selftest(); ran = True
    if args.boozmn:
        gamma_w_for_boozmn(args.boozmn, N=args.N, t_star=args.t_star, s0=args.s0, seed=args.seed,
                           branch_continue=not args.keep)
        ran = True
    if not ran:
        ap.print_help()
        print("\nExamples:\n  python Gamma_W_final.py --selftest"
              "\n  python Gamma_W_final.py --boozmn boozmn_0.05_08.nc --N 128 --t-star 0.2 --s0 0.3")


if __name__ == "__main__":
    main()
