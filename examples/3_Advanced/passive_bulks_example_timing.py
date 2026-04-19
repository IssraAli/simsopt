#!/usr/bin/env python
r"""
Timing harness for :file:`passive_bulks_optimization.py` (same geometry / basis).

Puck **rotational (quaternion) DOFs stay fixed** so gradients use the fast TF-only
VJP path; omitting this was dominated by multi-minute JAX compiles for the full
puck-geometry path.

Run from repo root::

    PYTHONPATH=src python examples/3_Advanced/passive_bulks_example_timing.py
"""

from __future__ import annotations

import statistics
import time
from pathlib import Path

import numpy as np

from simsopt.field import BiotSavart
from simsopt.field.psc_bulk import PSCBulkArray
from simsopt.field.magneticfield import MagneticFieldSum
from simsopt.field.selffield import regularization_rect
from simsopt.geo import CurveLength, SurfaceRZFourier
from simsopt.objectives import SquaredFlux, Weight
from simsopt.util import initialize_coils


def _median_ms(samples: list[float]) -> float:
    return 1000.0 * statistics.median(samples)


def main() -> None:
    TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
    filename = TEST_DIR / "wout_schuett_henneberg_nfp2_QA.nc"
    nphi = ntheta = 8

    t0 = time.perf_counter()
    s = SurfaceRZFourier.from_wout(
        filename, range="half period", nphi=nphi, ntheta=ntheta
    )
    nfp, stellsym = s.nfp, s.stellsym
    base_curves_tf, _, coils_tf, base_currents_tf = initialize_coils(
        s,
        "SchuettHennebergQAnfp2",
        regularization_rect(0.2, 0.2),
    )
    g_plasma = s.gamma()
    R_outer = float(np.sqrt(g_plasma[:, :, 0] ** 2 + g_plasma[:, :, 1] ** 2).max())
    puck_offset, puck_R, puck_t = 2.5, 0.9, 0.2
    phi_base = np.linspace(np.pi / (8 * nfp), np.pi / nfp - np.pi / (8 * nfp), 3)
    centers_list, axes_list = [], []
    for phi in phi_base:
        for z_off in [1.1, -1.1]:
            r = R_outer + puck_offset
            centers_list.append([r * np.cos(phi), r * np.sin(phi), z_off])
            axes_list.append(-np.array([np.cos(phi), np.sin(phi), 0.0]))
    centers = np.array(centers_list)
    axes = np.array(axes_list)
    radii = np.full(len(centers), puck_R)
    thicknesses = np.full(len(centers), puck_t)
    eval_points = np.ascontiguousarray(s.gamma().reshape(-1, 3))
    t_setup = time.perf_counter() - t0

    t0 = time.perf_counter()
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
    )
    t_psc_init = time.perf_counter() - t0

    # Keep all puck geometry fixed (same as example when quaternion unfix is off):
    # only TF DOFs in JF.x; vjp_setup_B uses _vjp_tf_only (no _ensure_jax_full).

    b_bulk = psc_bulk.biot_savart
    b_tf = BiotSavart(coils_tf)
    btot = MagneticFieldSum([b_bulk, b_tf])
    Jf = SquaredFlux(s, btot)
    LENGTH_WEIGHT = Weight(1e-4)
    Jls = [CurveLength(c) for c in base_curves_tf]
    JF = Jf
    for Jl in Jls:
        JF = JF + LENGTH_WEIGHT * Jl

    dofs = np.copy(JF.x)

    def one_step() -> tuple[float, float]:
        JF.x = dofs
        psc_bulk.recompute_currents()
        btot.clear_cached_properties()
        t_j0 = time.perf_counter()
        float(JF.J())
        t_j = time.perf_counter() - t_j0
        t_g0 = time.perf_counter()
        np.asarray(JF.dJ(), dtype=float)
        t_g = time.perf_counter() - t_g0
        return t_j, t_g

    # Cold first eval (may include JAX compile on dJ)
    t0 = time.perf_counter()
    t_j_cold, t_g_cold = one_step()
    t_full_cold = time.perf_counter() - t0

    n_warm = 8
    t_j_list: list[float] = []
    t_g_list: list[float] = []
    t_full_list: list[float] = []
    for _ in range(n_warm):
        t0 = time.perf_counter()
        tj, tg = one_step()
        t_full_list.append(time.perf_counter() - t0)
        t_j_list.append(tj)
        t_g_list.append(tg)

    # --- Fine-grained warm breakdown (median of n_prof samples per metric) ---
    n_prof = 15

    def _prep_opt_step() -> None:
        """Match one optimization iteration before objective/gradient."""
        JF.x = dofs
        psc_bulk.recompute_currents()
        btot.clear_cached_properties()
        btot.set_points(eval_points)

    # Pre-warm
    for _ in range(3):
        _prep_opt_step()
        _ = Jf.J()
        _ = np.asarray(JF.dJ(), dtype=float)

    t_recompute: list[float] = []
    t_clear: list[float] = []
    t_Jf_J: list[float] = []
    t_len_J: list[float] = []
    t_Jf_dJ: list[float] = []
    t_len_dJ: list[float] = []
    t_JF_dJ: list[float] = []
    t_B_bulk: list[float] = []
    t_B_tf: list[float] = []
    t_solve_beta: list[float] = []

    for _ in range(n_prof):
        JF.x = dofs
        t0 = time.perf_counter()
        psc_bulk.recompute_currents()
        t_recompute.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        btot.clear_cached_properties()
        t_clear.append(time.perf_counter() - t0)

        btot.set_points(eval_points)

        t0 = time.perf_counter()
        _ = float(Jf.J())
        t_Jf_J.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        for Jl in Jls:
            _ = float((LENGTH_WEIGHT * Jl).J())
        t_len_J.append(time.perf_counter() - t0)

        _prep_opt_step()
        t0 = time.perf_counter()
        _ = np.asarray(Jf.dJ(), dtype=float)
        t_Jf_dJ.append(time.perf_counter() - t0)

        _prep_opt_step()
        t0 = time.perf_counter()
        for Jl in Jls:
            _ = np.asarray((LENGTH_WEIGHT * Jl).dJ(), dtype=float)
        t_len_dJ.append(time.perf_counter() - t0)

        _prep_opt_step()
        t0 = time.perf_counter()
        _ = np.asarray(JF.dJ(), dtype=float)
        t_JF_dJ.append(time.perf_counter() - t0)

        _prep_opt_step()
        t0 = time.perf_counter()
        _ = b_bulk.B()
        t_B_bulk.append(time.perf_counter() - t0)

        btot.set_points(eval_points)
        t0 = time.perf_counter()
        _ = b_tf.B()
        t_B_tf.append(time.perf_counter() - t0)

        arr = psc_bulk._tf_arrays()
        t0 = time.perf_counter()
        _ = psc_bulk._solve_beta(arr)
        t_solve_beta.append(time.perf_counter() - t0)

    pts = eval_points
    v_B = np.ones((len(pts), 3), dtype=float) / np.sqrt(len(pts) * 3)
    t0 = time.perf_counter()
    _ = psc_bulk.vjp_setup_B(v_B, pts)
    t_vjp_cold = time.perf_counter() - t0
    t_vjp_warm: list[float] = []
    for _ in range(12):
        t0 = time.perf_counter()
        _ = psc_bulk.vjp_setup_B(v_B, pts)
        t_vjp_warm.append(time.perf_counter() - t0)

    n_p = len(psc_bulk._all_pucks)
    nq = psc_bulk._quad_points.shape[0]
    n_red = psc_bulk._Q.shape[1]
    n_surf = eval_points.shape[0]

    def _pct(label: str, ms: float, total: float) -> str:
        if total <= 0:
            return f"{label}: {ms:.2f} ms"
        return f"{label}: {ms:.2f} ms  ({100.0 * ms / total:.0f}% of dJ-sized step)"

    med_rec = _median_ms(t_recompute)
    med_JfJ = _median_ms(t_Jf_J)
    med_JfdJ = _median_ms(t_Jf_dJ)
    med_JFdJ = _median_ms(t_JF_dJ)
    med_lenJ = _median_ms(t_len_J)
    med_lendJ = _median_ms(t_len_dJ)
    med_Bb = _median_ms(t_B_bulk)
    med_Bt = _median_ms(t_B_tf)
    med_beta = _median_ms(t_solve_beta)
    med_one_step = _median_ms(t_full_list)

    # dJ step dominant pieces (flux dJ ~ Jf_dJ; full ~ JF_dJ)
    dJ_flux_share = 100.0 * med_JfdJ / med_JFdJ if med_JFdJ > 0 else 0.0

    print("passive_bulks example timing (reactor-scale QASH; puck rotations FIXED)")
    print(f"  Surface quad points nphi*ntheta = {n_surf}")
    print()
    print("  --- One-time build ---")
    print(f"  surface+coils setup (no PSC):     {t_setup * 1000:.1f} ms")
    print(
        f"  PSCBulkArray.__init__ (incl. L): {t_psc_init * 1000:.1f} ms  ({n_p} pucks, {nq} quads, {n_red} reduced DOFs)"
    )
    print()
    print("  --- Full objective call (like scipy one_step) ---")
    print(f"  cold  J+dJ (first call, incl. JAX compile): {t_full_cold * 1000:.1f} ms")
    print(
        f"  warm  J+dJ (median of {n_warm}):          {med_one_step:.1f} ms  (J={_median_ms(t_j_list):.2f}, dJ={_median_ms(t_g_list):.2f})"
    )
    print()
    print(
        f"  --- Warm breakdown (median of {n_prof}; each line is timed separately) ---"
    )
    print(_pct("  recompute_currents (beta + hash)", med_rec, med_JFdJ))
    print(_pct("  SquaredFlux Jf.J() [btot.B → bulk+TF]", med_JfJ, med_JFdJ))
    print(_pct("  length penalty J only", med_lenJ, med_JFdJ))
    print(_pct("  SquaredFlux Jf.dJ()  (flux gradient)", med_JfdJ, med_JFdJ))
    print(f"    -> flux dJ is ~{dJ_flux_share:.0f}% of full JF.dJ()")
    print(_pct("  length penalty dJ only", med_lendJ, med_JFdJ))
    print(_pct("  full JF.dJ() (flux + lengths)", med_JFdJ, med_JFdJ))
    print(_pct("  b_bulk.B() alone", med_Bb, med_JfdJ))
    print(_pct("  b_tf.B() alone", med_Bt, med_JfdJ))
    print(
        _pct("  _solve_beta only (JAX L f solve, no B at plasma)", med_beta, med_JfdJ)
    )
    print()
    print("  --- Passive bulk VJP (TF path; inside Jf.dJ) ---")
    print(f"  vjp_setup_B first call:         {t_vjp_cold * 1000:.1f} ms")
    print(f"  vjp_setup_B warm (median of 12): {_median_ms(t_vjp_warm):.1f} ms")
    print()
    print("  Bottleneck summary (warm optimization step):")
    print(
        f"    • dJ ~ {_median_ms(t_g_list):.1f} ms dominates vs J ~ {_median_ms(t_j_list):.1f} ms."
    )
    print(
        f"    • Flux gradient Jf.dJ ~ {med_JfdJ:.1f} ms; length dJ ~ {med_lendJ:.1f} ms."
    )
    print(
        f"    • Passive bulk B at plasma ~ {med_Bb:.1f} ms; TF B ~ {med_Bt:.1f} ms (forward)."
    )
    print(
        f"    • recompute_currents ~ {med_rec:.1f} ms; JAX beta solve alone ~ {med_beta:.1f} ms."
    )
    print(
        f"    • psc_bulk.vjp_setup_B ~ {_median_ms(t_vjp_warm):.1f} ms (matches bulk contribution in dJ)."
    )
    print()
    print(f"  len(JF.x) = {len(dofs)}  (TF only)")


if __name__ == "__main__":
    main()
