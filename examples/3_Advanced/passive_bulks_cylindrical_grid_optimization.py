#!/usr/bin/env python
r"""
Coupled optimization of TF coils with passive bulk pucks on a **cylindrical
``(r, φ, z)`` lattice** with radial orientation
(:meth:`~simsopt.field.psc_bulk.PSCBulkArray.from_cylindrical_grid`).

This example uses the **reactor-scale** Schuett–Henneberg QA equilibrium
(``wout_schuett_henneberg_nfp2_QA.nc``) and TF initialization matching
``passive_coils_QASH.py`` (35 MA total, ``SchuettHennebergQAnfp2``).

The objective is :math:`J = J_f + w_L \sum_i L_i` where :math:`J_f` is the
:class:`~simsopt.objectives.SquaredFlux` on a target plasma surface and
:math:`L_i` are TF coil length penalties.

The TF coil DOFs (curve shape and currents) are optimized directly, while the
passive bulk currents are *induced* (recomputed from the linear shell solve
at every iteration).  Gradients flow through the VJP of the shell solve,
including **derivatives of the bulk field w.r.t. TF coil DOFs** (same chain
rule as :class:`~simsopt.field.coil.PSCArray`, with surface inductance instead
of filament mutual inductances).

Puck geometry (centers, quaternions, radii, thicknesses) is **fixed**; only TF coil
DOFs are optimized.  Passive bulk currents are induced each iteration.

Surface VTK includes :math:`B_n` and :math:`B_n/|B|` for the TF field alone, the
bulk field alone, and the total field.

VTK snapshots are written **once before** optimization and **once after** (surface,
coils, pucks). No VTK I/O during ``minimize`` so the loop stays near the cost of
``fun()`` alone.

Requirements: JAX, pyevtk (optional, for VTK output).

Companion: ``passive_bulks_winding_surface_optimization.py`` (winding-surface layout).
"""

import os
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from simsopt.field import BiotSavart, coils_to_vtk
from simsopt.field.psc_bulk import PSCBulkArray
from simsopt.field.magneticfield import MagneticFieldSum
from simsopt.field.puck_init import cylindrical_grid_pucks
from simsopt.field.selffield import regularization_rect
from simsopt.geo import CurveLength, SurfaceRZFourier
from simsopt.objectives import SquaredFlux, Weight
from simsopt.util import initialize_coils

OUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "passive_bulks_cylindrical_grid_optimization_out",
)
os.makedirs(OUT_DIR, exist_ok=True)

# Override for quick timing:  PASSIVE_BULKS_MAXITER=8 python passive_bulks_optimization.py
MAXITER = int(os.environ.get("PASSIVE_BULKS_MAXITER", "50"))

# Wall-clock profiling (printed at end)
_WALL = {
    "t_import_to_psc": 0.0,
    "t_sanity_block": 0.0,
    "t_vtk_initial": 0.0,
    "t_taylor_section": 0.0,
    "t_optimize": 0.0,
    "t_vtk_final": 0.0,
    "n_fun_calls": 0,
    "n_opt_fun_calls": 0,
}
_FUN_MS = []  # total ms per fun() [recompute+J+dJ]
_J_MS = []
_DJ_MS = []
# Low resolution for faster iteration (increase for production runs).
nphi = 8
ntheta = 8

_T0 = time.perf_counter()

# ---------------------------------------------------------------------------
# 1. Target plasma surface (reactor-scale QA, half period)
# ---------------------------------------------------------------------------
TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
filename = TEST_DIR / "wout_schuett_henneberg_nfp2_QA.nc"
s = SurfaceRZFourier.from_wout(filename, range="half period", nphi=nphi, ntheta=ntheta)
nfp = s.nfp
stellsym = s.stellsym

qphi = nphi * 4
qtheta = ntheta * 4
quadpoints_phi = np.linspace(0, 1, qphi, endpoint=True)
quadpoints_theta = np.linspace(0, 1, qtheta, endpoint=True)
s_plot = SurfaceRZFourier.from_wout(
    filename,
    range="half period",
    quadpoints_phi=quadpoints_phi,
    quadpoints_theta=quadpoints_theta,
)

# ---------------------------------------------------------------------------
# 2. TF coils (35 MA total, same recipe as passive_coils_QASH)
# ---------------------------------------------------------------------------
base_curves_tf, _curves_tf, coils_tf, base_currents_tf = initialize_coils(
    s,
    "SchuettHennebergQAnfp2",
    regularization_rect(0.2, 0.2),
)

# ---------------------------------------------------------------------------
# 3. Passive bulk pucks on a cylindrical (r, φ, z) lattice (radial axes); all puck
#    DOFs fixed — only TF coils are optimized.
# ---------------------------------------------------------------------------
eval_points = np.ascontiguousarray(s.gamma().reshape(-1, 3))


def _drop_symm_overlapping_base_pucks(
    centers: np.ndarray,
    axes: np.ndarray,
    radii: np.ndarray,
    thicknesses: np.ndarray,
    nfp: int,
    stellsym: bool,
    eps: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Greedy filter: keep a base puck only if no replica of it lies within
    :math:`R_i + R_j + \\varepsilon` of any replica of an already-kept base puck.

    Replicas use the same ``Rz(2*pi*k/nfp)`` rotation and ``diag(1, -1, -1)``
    mirror as :meth:`simsopt.field.coil.PSCBulkArray._replicate_pucks`, so this
    filter reproduces exactly the post-symmetrization geometry that
    :class:`~simsopt.field.coil.PSCBulkArray` will build internally.
    """

    def _replicas(c: np.ndarray) -> np.ndarray:
        out: list[np.ndarray] = []
        for k in range(int(nfp)):
            ang = 2.0 * np.pi * k / float(nfp)
            Rz = np.array(
                [
                    [np.cos(ang), -np.sin(ang), 0.0],
                    [np.sin(ang), np.cos(ang), 0.0],
                    [0.0, 0.0, 1.0],
                ]
            )
            cR = Rz @ c
            out.append(cR)
            if stellsym:
                out.append(np.array([cR[0], -cR[1], -cR[2]]))
        return np.asarray(out)

    n = centers.shape[0]
    kept_idx: list[int] = []
    kept_reps: list[np.ndarray] = []
    for i in range(n):
        ri = float(radii[i]) + 0.5 * float(thicknesses[i])
        reps_i = _replicas(centers[i])
        ok = True
        for j_pos, j in enumerate(kept_idx):
            rj = float(radii[j]) + 0.5 * float(thicknesses[j])
            d = np.linalg.norm(
                reps_i[:, None, :] - kept_reps[j_pos][None, :, :], axis=-1
            )
            if np.any(d <= ri + rj + eps):
                ok = False
                break
        if ok:
            kept_idx.append(i)
            kept_reps.append(reps_i)
    keep = np.asarray(kept_idx, dtype=int)
    return centers[keep], axes[keep], radii[keep], thicknesses[keep]


centers, axes, radii, thicknesses = cylindrical_grid_pucks(
    s,
    dr=0.7,
    dz=1.0,
    n_phi=2,
    z_min=-3.5,
    z_max=3.5,
    d_inner=1.2,
    d_outer=2.6,
    puck_R=0.30,
    puck_t=0.45,
    plasma_clearance=0.1,
    nfp=nfp,
)
centers, axes, radii, thicknesses = _drop_symm_overlapping_base_pucks(
    centers,
    axes,
    radii,
    thicknesses,
    nfp=nfp,
    stellsym=stellsym,
)

# ``adaptive_self_reg=True`` is now the default in :class:`PSCBulkArray`
# (quadrature-cell-scaled self-term regularization).  It is *required*
# for the induced :math:`\mathbf B` on the plasma surface to come out at
# physically correct magnitudes -- the legacy fixed-scalar
# ``regularization_delta`` (pass ``adaptive_self_reg=False`` explicitly)
# overcounts coincident-cell self-integrals by ~:math:`10^{4}-10^{5}` and
# gives :math:`|\mathbf B_{\rm bulk}|` on the order of :math:`10^{-7}` T.
psc_bulk = PSCBulkArray(
    centers,
    axes,
    radii,
    thicknesses,
    coils_tf,
    eval_points=eval_points,
    m_fourier=2,
    l_zernike=4,
    k_chebyshev=2,
    n_rho=6,
    n_phi=8,
    n_z=4,
    nfp=nfp,
    stellsym=stellsym,
    strict_rim_continuity=True,
)

n_base_pucks = psc_bulk._n_base_pucks
print(f"Num base pucks (before symmetrization) = {n_base_pucks}")
print(f"nfp = {nfp}, stellsym = {stellsym}")
n_expected = n_base_pucks * nfp * (2 if stellsym else 1)
print(f"Expected total pucks (after symmetrization) = {n_expected}")

_WALL["t_import_to_psc"] = time.perf_counter() - _T0


def _assert_pucks_nonintersecting(psc):
    """Conservative check: distinct base pucks' hulls (R + t/2) must not overlap.

    Skips pairs that are nfp/stellsym copies of the *same* base puck: those
    centers can be close by construction and are not separate physical bodies.
    """
    pucks = psc._all_pucks
    base_idx = np.asarray(psc._all_puck_base_indices)
    n = len(pucks)
    for i in range(n):
        ci, _, Ri, ti = pucks[i]
        ri = float(Ri) + 0.5 * float(ti)
        for j in range(i + 1, n):
            if int(base_idx[i]) == int(base_idx[j]):
                continue
            cj, _, Rj, tj = pucks[j]
            rj = float(Rj) + 0.5 * float(tj)
            d = float(np.linalg.norm(ci - cj))
            assert d > ri + rj + 1e-9, (
                f"Pucks {i} and {j} may intersect: center distance {d:.6f} <= "
                f"sum of effective radii {ri + rj:.6f}"
            )


_assert_pucks_nonintersecting(psc_bulk)

n_total_pucks = len(psc_bulk._all_pucks)
print(f"Total pucks (after symmetrization) = {n_total_pucks}")
print(f"Null modes removed from L: {psc_bulk.n_null_modes()}")
print(f"Total shell DOFs: {psc_bulk._n_dof_total}, reduced: {psc_bulk._Q.shape[1]}")

# ---------------------------------------------------------------------------
# 4. Combined field and objective
# ---------------------------------------------------------------------------
b_bulk = psc_bulk.biot_savart
b_tf = BiotSavart(coils_tf)
btot = MagneticFieldSum([b_bulk, b_tf])

Jf = SquaredFlux(s, btot)

LENGTH_WEIGHT = Weight(1e-4)
Jls = [CurveLength(c) for c in base_curves_tf]

JF = Jf
for Jl in Jls:
    JF = JF + LENGTH_WEIGHT * Jl


def _print_free_dof_report():
    """Print every free DOF name in ``JF`` (same order as ``JF.x`` / ``JF.dJ()``)."""
    names = np.array(JF.dof_names)
    print(f"  Total optimized DOFs: {len(names)}")
    print("  Free DOF names (object:name):")
    for k, nm in enumerate(names):
        print(f"    [{k:4d}] {nm}")
    puck_prefix = psc_bulk.name + ":"
    n_puck_in_x = int(np.sum([str(nm).startswith(puck_prefix) for nm in names]))
    print(f"  Count with prefix {puck_prefix!r}: {n_puck_in_x}")


def _grad_split_puck_vs_rest(grad: np.ndarray) -> tuple[float, float]:
    """L2 norms of gradient components for ``psc_bulk`` vs all other free DOFs."""
    names = np.array(JF.dof_names)
    puck_prefix = psc_bulk.name + ":"
    mask = np.array([str(nm).startswith(puck_prefix) for nm in names])
    g_p = float(np.linalg.norm(grad[mask])) if np.any(mask) else 0.0
    g_r = float(np.linalg.norm(grad[~mask])) if np.any(~mask) else 0.0
    return g_p, g_r


# ---------------------------------------------------------------------------
# 5. DOF summary
# ---------------------------------------------------------------------------
print()
print("=" * 72)
print("  DOF summary")
print("=" * 72)
_print_free_dof_report()
n_curve_dofs = sum(len(c.x) for c in base_curves_tf)
n_current_dofs = sum(len(c.x) for c in base_currents_tf)
n_puck_dofs_free = sum(psc_bulk.local_dofs_free_status)
print(f"    TF coil curves (base):    {n_curve_dofs} DOFs (shape + position)")
print(f"    TF coil currents (base):  {n_current_dofs} DOFs")
print(
    f"    Puck geometry (free):     {n_puck_dofs_free} DOFs "
    f"(out of {len(psc_bulk.local_full_x)} total, rest fixed)"
)
print("  Induced (not optimized):")
print(
    f"    Passive bulk currents beta: {psc_bulk._Q.shape[1]} reduced coefficients "
    f"(recomputed via L_r^{{-1}} f_r each iteration)"
)
print()

# ---------------------------------------------------------------------------
# 6. Sanity: TF change updates induced beta and bulk B
# ---------------------------------------------------------------------------
_t_s0 = time.perf_counter()
print("=" * 72)
print("  Sanity: perturb TF current -> beta and bulk B change")
print("=" * 72)
# Canonical dirty-flagging pattern for passive bulks:
#   psc_bulk.recompute_currents()
#   psc_bulk.biot_savart.invalidate()  # or btot.clear_cached_properties()
psc_bulk.recompute_currents()
beta0 = np.array(psc_bulk.beta)
btot.set_points(eval_points)
B0_bulk = b_bulk.B().copy()
# Nudge first free current DOF if any
cur0 = base_currents_tf[0].get_value()
base_currents_tf[0].set_dofs(np.array([cur0 * 1.001]))
psc_bulk.recompute_currents()
btot.clear_cached_properties()
beta1 = np.array(psc_bulk.beta)
btot.set_points(eval_points)
B1_bulk = b_bulk.B().copy()
base_currents_tf[0].set_dofs(np.array([cur0]))
psc_bulk.recompute_currents()
btot.clear_cached_properties()
print(
    f"  |beta1 - beta0| / (|beta0|+1e-30) = {np.linalg.norm(beta1 - beta0) / (np.linalg.norm(beta0) + 1e-30):.6e}"
)
print(
    f"  |B_bulk1 - B_bulk0| / (|B_bulk0|+1e-30) = {np.linalg.norm(B1_bulk - B0_bulk) / (np.linalg.norm(B0_bulk) + 1e-30):.6e}"
)
print()
_WALL["t_sanity_block"] = time.perf_counter() - _t_s0

# ---------------------------------------------------------------------------
# 7. VTK snapshot helper
# ---------------------------------------------------------------------------
try:
    from simsopt.field.puck_vtk import pucks_to_vtk

    HAS_VTK = True
except ImportError:
    HAS_VTK = False


def emit_vtk_snapshot(iteration, include_pucks=True):
    """Write surface + coils + pucks VTK files for a single optimization frame.

    Surface point data includes :math:`B_n` and :math:`B_n/|B|` for TF-only,
    bulk-only, and total fields.
    """
    if not HAS_VTK:
        return
    pts = s_plot.gamma().reshape((-1, 3))
    n_hat = s_plot.unitnormal()

    b_tf.set_points(pts)
    B_tf = b_tf.B().reshape((qphi, qtheta, 3))
    b_bulk.set_points(pts)
    B_bulk_arr = b_bulk.B().reshape((qphi, qtheta, 3))
    B_total = B_tf + B_bulk_arr

    def _bn_pair(B: np.ndarray, tag: str) -> dict:
        Bn = np.sum(B * n_hat, axis=-1)
        modB = np.linalg.norm(B, axis=-1)
        return {
            f"B_N_{tag}": Bn[:, :, None],
            f"B_N/|B|_{tag}": (Bn / (modB + 1e-30))[:, :, None],
        }

    pointData: dict = {}
    pointData.update(_bn_pair(B_tf, "tf"))
    pointData.update(_bn_pair(B_bulk_arr, "bulk"))
    pointData.update(_bn_pair(B_total, "total"))

    s_plot.to_vtk(os.path.join(OUT_DIR, f"surf_{iteration:04d}"), extra_data=pointData)
    coils_to_vtk(coils_tf, os.path.join(OUT_DIR, f"coils_{iteration:04d}"))
    if include_pucks:
        pucks_to_vtk(
            psc_bulk,
            os.path.join(OUT_DIR, f"pucks_{iteration:04d}"),
            n_phi=32,
            n_r=12,
        )
    btot.set_points(s.gamma().reshape(-1, 3))


_tvtk0 = time.perf_counter()
emit_vtk_snapshot(0)
_WALL["t_vtk_initial"] = time.perf_counter() - _tvtk0
print(f"  VTK initial snapshot written to {OUT_DIR}")

# ---------------------------------------------------------------------------
# 8. Taylor test (gradient validation)
# ---------------------------------------------------------------------------
print()
print("=" * 72)
print("  Taylor test")
print("=" * 72)

_t_taylor0 = time.perf_counter()


def fun(dofs):
    """Objective + gradient, recomputing passive currents each call."""
    t0 = time.perf_counter()
    JF.x = dofs
    psc_bulk.recompute_currents()
    btot.clear_cached_properties()
    t1 = time.perf_counter()
    J = JF.J()
    t2 = time.perf_counter()
    grad = JF.dJ()
    t3 = time.perf_counter()
    _FUN_MS.append(1000.0 * (t3 - t0))
    _J_MS.append(1000.0 * (t2 - t1))
    _DJ_MS.append(1000.0 * (t3 - t2))
    _WALL["n_fun_calls"] += 1
    return float(J), np.asarray(grad, dtype=float)


dofs = np.copy(JF.x)
J0, dJ0 = fun(dofs)
np.random.seed(1)
h = np.random.randn(len(dofs))
h = h / np.linalg.norm(h)
dJh = np.sum(dJ0 * h)

for eps in [1e-2, 1e-3, 1e-4, 1e-5]:
    Jp, _ = fun(dofs + eps * h)
    Jm, _ = fun(dofs - eps * h)
    fd = (Jp - Jm) / (2 * eps)
    err = abs(fd - dJh) / (abs(dJh) + 1e-30)
    print(f"  eps={eps:.0e}  FD={fd:.8e}  analytic={dJh:.8e}  rel_err={err:.2e}")

# Finite difference on first puck quaternion DOF alone (if present)
idx_fd = None
for k, nm in enumerate(JF.dof_names):
    if str(nm).endswith(":qk_0"):
        idx_fd = k
        break
if idx_fd is not None:
    eps_q = 1e-5
    e = np.zeros(len(dofs))
    e[idx_fd] = 1.0
    Jp, g_dummy = fun(dofs + eps_q * e)
    Jm, _ = fun(dofs - eps_q * e)
    fd_q = (Jp - Jm) / (2 * eps_q)
    print(f"  FD along qk_0 only: {fd_q:.8e},  analytic dJ/dqk_0 = {dJ0[idx_fd]:.8e}")
print()
_WALL["t_taylor_section"] = time.perf_counter() - _t_taylor0

# ---------------------------------------------------------------------------
# 9. Optimization
# ---------------------------------------------------------------------------
print()
print("=" * 72)
print("  Optimization (L-BFGS-B)")
print("=" * 72)

dofs = np.copy(JF.x)
J_init, g_init = fun(dofs)
gp, gr = _grad_split_puck_vs_rest(g_init)
print(f"  Initial |grad|_puck = {gp:.6e}, |grad|_rest = {gr:.6e}")

n_eval = [0]
_n_fun_at_opt_start = [0]


def fun_with_print(dofs):
    J, grad = fun(dofs)
    n_eval[0] += 1
    _WALL["n_opt_fun_calls"] += 1
    gp, gr = _grad_split_puck_vs_rest(grad)
    if n_eval[0] % 5 == 1 or n_eval[0] <= 3:
        btot.set_points(s.gamma().reshape(-1, 3))
        B = btot.B().reshape(s.gamma().shape)
        Bn = np.sum(B * s.unitnormal(), axis=2)
        mean_Bn = np.mean(np.abs(Bn))
        print(
            f"  iter {n_eval[0]:3d}  J={J:.6e}  Jf={Jf.J():.6e}  "
            f"<|B.n|>={mean_Bn:.6e}  |grad|={np.linalg.norm(grad):.2e}  "
            f"|g_puck|={gp:.2e}  |g_rest|={gr:.2e}"
        )
    return J, grad


_n_fun_at_opt_start[0] = _WALL["n_fun_calls"]
_t_opt0 = time.perf_counter()
t1 = time.time()
res = minimize(
    fun_with_print,
    dofs,
    jac=True,
    method="L-BFGS-B",
    options={"maxiter": MAXITER, "maxcor": 200},
    tol=1e-15,
)
t2 = time.time()
_WALL["t_optimize"] = time.perf_counter() - _t_opt0

JF.x = res.x
psc_bulk.recompute_currents()
btot.clear_cached_properties()
J_final = JF.J()

print()
print(f"  Optimization finished in {t2 - t1:.1f}s  ({res.nit} iterations)")
print(f"  J_initial = {J_init:.6e}")
print(f"  J_final   = {J_final:.6e}")

btot.set_points(s.gamma().reshape(-1, 3))
B = btot.B().reshape(s.gamma().shape)
Bn = np.sum(B * s.unitnormal(), axis=2)
print(f"  <|B.n|> final = {np.mean(np.abs(Bn)):.6e}")

gp_f, gr_f = _grad_split_puck_vs_rest(np.asarray(JF.dJ()))
print(f"  Final |grad|_puck = {gp_f:.6e}, |grad|_rest = {gr_f:.6e}")


def _equivalent_currents_per_puck(psc):
    """Same logic as :meth:`PSCBulkArray.get_equivalent_currents` (fallback if old install)."""
    _, K_mag = psc.get_shell_currents()
    n_pucks = len(psc._all_pucks)
    I_eq = np.zeros(n_pucks)
    for p in range(n_pucks):
        r0, r1 = psc._quad_row_ranges[p]
        _, _, R_val, t_val = psc._all_pucks[p]
        Km = K_mag[r0:r1]
        I_side = float(np.max(Km)) * t_val
        I_face = float(np.max(Km)) * 2.0 * R_val
        I_eq[p] = max(I_side, I_face)
    return I_eq


if hasattr(psc_bulk, "get_equivalent_currents"):
    I_eq = psc_bulk.get_equivalent_currents()
else:
    I_eq = _equivalent_currents_per_puck(psc_bulk)
print(f"  I_equivalent per puck (A): min={I_eq.min():.2e}, max={I_eq.max():.2e}")
print(f"  I_equivalent (first 10): {I_eq[:10]}")

# ---------------------------------------------------------------------------
# 10. Final VTK export
# ---------------------------------------------------------------------------
_tvfinal = time.perf_counter()
emit_vtk_snapshot(n_eval[0] + 1, include_pucks=True)
_WALL["t_vtk_final"] = time.perf_counter() - _tvfinal
if HAS_VTK:
    print(f"  VTK files written to {OUT_DIR}")
else:
    print("  (pyevtk not installed; skipping VTK output)")

# ---------------------------------------------------------------------------
# Wall-clock report (explains why total time >> ~110 ms per gradient alone)
# ---------------------------------------------------------------------------
print()
print("=" * 72)
print("  Timing report (wall clock, perf_counter)")
print("=" * 72)
idx0 = _n_fun_at_opt_start[0]
n_opt = _WALL["n_opt_fun_calls"]
ms = _FUN_MS
ms_opt = _FUN_MS[idx0 : idx0 + n_opt] if n_opt else []
import statistics as _statistics


def _med(x):
    return _statistics.median(x) if x else float("nan")


n_taylor_calls = idx0 - 1

print(f"  MAXITER = {MAXITER}  (set PASSIVE_BULKS_MAXITER to change)")
print(f"  Build (import … PSCBulkArray ready):     {_WALL['t_import_to_psc']:.1f} s")
print(
    f"  Sanity block (perturb TF):               {_WALL['t_sanity_block'] * 1000:.1f} ms"
)
print(f"  VTK initial (high-res plot surface):    {_WALL['t_vtk_initial']:.2f} s")
print(
    f"  Taylor + FD section:                     {_WALL['t_taylor_section']:.2f} s  (~{n_taylor_calls} fun() calls)"
)
print(
    f"  minimize() L-BFGS-B:                     {_WALL['t_optimize']:.2f} s  ({n_opt} fun() calls; +1 bootstrap fun() before)"
)
print(f"  VTK final snapshot:                      {_WALL['t_vtk_final']:.2f} s")
print()
print("  fun() = recompute_currents + J + dJ  (per call):")
print(
    f"    All calls:   median {_med(ms):.1f} ms  (mean {sum(ms) / len(ms):.1f} ms)  n={len(ms)}"
)
if ms_opt:
    print(
        f"    Opt only:    median {_med(ms_opt):.1f} ms  (mean {sum(ms_opt) / len(ms_opt):.1f} ms)  n={len(ms_opt)}"
    )
if _J_MS and _DJ_MS:
    print(f"    Split (last call):  J ~ {_J_MS[-1]:.1f} ms,  dJ ~ {_DJ_MS[-1]:.1f} ms")
print()
print("  Why it feels slow: build ~40–50 s once; Taylor uses many fun() calls;")
print(
    "  each begin/end VTK writes high-res B + coils + 24 pucks (heavy vs one fun() step)."
)
print("=" * 72)
print("Done.")
