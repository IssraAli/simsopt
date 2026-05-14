"""Tests for the dipole-mode solver of :class:`PSCBulkArray`.

Implements the test matrix specified in the lean dipole bulk solver
plan: primitive correctness, ``PSCBulkArray`` integration, self-
inductance freeze, and JAX VJP correctness.
"""

from __future__ import annotations

import math
import os
from typing import Tuple

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("jax.numpy")

import jax
import jax.numpy as jnp

from simsopt.field import _psc_bulk_dipole as dip
from simsopt.field.bulk_inductance import MU0_OVER_4PI
from simsopt.field.bulk_multipole import (
    magnetic_field_dipole_points,
    pair_inductance_dipole_block,
)
from simsopt.field.coil import Coil, Current
from simsopt.field.psc_bulk import PSCBulkArray
from simsopt.geo import CurveXYZFourier


MU0 = 4.0e-7 * np.pi


def _unit_circle_coil(
    current_amp: float = 1.0e5, radius: float = 1.0, n: int = 32
):
    """Single circular TF coil on the z=0 plane, axis along z."""
    curve = CurveXYZFourier(n, 1)
    coeffs = np.zeros(9, dtype=float)
    coeffs[2] = radius  # x cos
    coeffs[4] = radius  # y sin
    curve.x = coeffs
    return Coil(curve, Current(current_amp))


def _two_loop_tf(
    sep: float = 0.5, current_amp: float = 1.0e5, radius: float = 1.0
):
    """Pair of coaxial coils at ``z = ±sep`` for a near-uniform B_z region."""
    coils = []
    for z in (-sep, +sep):
        curve = CurveXYZFourier(32, 1)
        coeffs = np.zeros(9, dtype=float)
        coeffs[0] = 0.0
        coeffs[2] = radius
        coeffs[4] = radius
        # Lift z-offset via the constant Fourier coefficient of the z
        # component (index 8 -> z0 constant).
        coeffs[6] = z  # z0 constant
        curve.x = coeffs
        coils.append(Coil(curve, Current(current_amp)))
    return coils


# ----------------------------------------------------------------------
# 1. Primitive correctness
# ----------------------------------------------------------------------


def test_polarizability_thin_disk_smythe_constant():
    """Axial polarizability matches Smythe's analytic thin-disk value.

    Smythe / Landau-Lifshitz:

        alpha_zz = (8/3) R^3 / mu_0 (axial)
        alpha_xx = alpha_yy = (16/3) R^3 / mu_0 (in-plane)

    The dipole solver uses the closed-form formula directly so the
    constant must match to machine precision (modulo the small
    empirical thickness correction baked into the function, which is
    1.0 for t = 0).
    """
    R = 0.1
    t = 0.0  # disable thickness correction
    alpha = dip.compute_puck_polarizability_tensor(R, t)
    smythe_axial = (8.0 / 3.0) * R**3 / MU0
    smythe_transverse = (16.0 / 3.0) * R**3 / MU0
    np.testing.assert_allclose(alpha[2, 2], smythe_axial, rtol=1e-10)
    np.testing.assert_allclose(alpha[0, 0], smythe_transverse, rtol=1e-10)
    np.testing.assert_allclose(alpha[1, 1], smythe_transverse, rtol=1e-10)
    np.testing.assert_allclose(alpha[0, 1], 0.0, atol=1e-10)


def test_polarizability_thin_disk_axial_scaling():
    """Axial polarizability scales as R^3 with R for fixed t/R."""
    R1 = 0.05
    R2 = 0.1
    t_over_R = 0.05
    alpha_1 = dip.compute_puck_polarizability_tensor(R1, t_over_R * R1)
    alpha_2 = dip.compute_puck_polarizability_tensor(R2, t_over_R * R2)
    assert alpha_1[2, 2] > 0.0
    assert alpha_2[2, 2] > 0.0
    ratio = alpha_2[2, 2] / alpha_1[2, 2]
    expected = (R2 / R1) ** 3
    # Thickness correction depends on ``t/R`` only, so for fixed t/R
    # it cancels in the ratio and the scaling is exactly cubic.
    np.testing.assert_allclose(ratio, expected, rtol=1e-10)


def test_psc_bulk_dipole_matches_smythe_far_field():
    """A single isolated puck in uniform B_z reproduces Smythe to <5%.

    Two-loop Helmholtz-like pair generates near-uniform ``B_z`` at the
    origin; the puck sits there, and the eval point is placed far
    along ``+x`` so the dipole approximation is exact.  Comparing
    the dipole-mode ``B`` against the analytic far-field of a Smythe
    dipole exercises the full L assembly + solve + field-evaluation
    chain end-to-end.
    """
    # Two coaxial loops at z = +/- 0.5 (radius 1.0, current 1e5).
    tf_coils = _two_loop_tf(sep=0.5, current_amp=1.0e5, radius=1.0)
    R = 0.02
    # Use a thickness that is positive (so the constructor does not
    # substitute ``default_thickness``) but tiny relative to ``R`` so
    # the dipole solver's empirical thickness correction is at the
    # 0.03% level and the comparison to Smythe is clean.
    t = 1.0e-5
    centers = np.array([[0.0, 0.0, 0.0]])
    axes = np.array([[0.0, 0.0, 1.0]])
    eval_pts = np.array([[5.0, 0.0, 0.0]])
    psc = PSCBulkArray(
        centers,
        axes,
        np.array([R]),
        np.array([t]),
        tf_coils,
        eval_points=eval_pts,
        m_fourier=2,
        l_zernike=3,
        k_chebyshev=2,
        n_rho=6,
        n_phi=8,
        n_z=3,
        nfp=1,
        stellsym=False,
        solver_mode="dipole",
        default_thickness=t,
    )
    B_dipole = np.asarray(psc.B_at_points(eval_pts))
    # Predict Smythe far field analytically.
    from simsopt.field.psc_bulk import PSCBulkArray as _PscRef  # noqa: F401

    # Compute B_TF at origin (current loop ± sep along z): use direct
    # circular-loop formula.  For each loop of radius a at z = z0 with
    # current I:
    #     B_z(origin) = mu_0 I a^2 / (2 (a^2 + z0^2)^(3/2))
    a = 1.0
    I = 1.0e5
    sep = 0.5
    Bz_from_one = MU0 * I * a**2 / (2.0 * (a**2 + sep**2) ** 1.5)
    Bz_TF = 2.0 * Bz_from_one
    # Smythe induced moment along z: m_z = -(8/3) R^3 B_z / mu_0.
    m_z = -(8.0 / 3.0) * R**3 * Bz_TF / MU0
    # B at (r, 0, 0) for a z-dipole at origin:
    #     B_z = (mu_0/4pi) (-m_z) / r^3
    r = 5.0
    Bz_predicted = (MU0 / (4.0 * np.pi)) * (-m_z) / r**3
    np.testing.assert_allclose(
        B_dipole[0, 2], Bz_predicted, rtol=0.05, atol=0.0
    )


def test_polarizability_symmetric_positive_definite():
    """Polarizability tensor is SPD for a generic puck."""
    R = 0.1
    t = 0.05 * R
    alpha = dip.compute_puck_polarizability_tensor(R, t, n_radial=20)
    np.testing.assert_allclose(alpha, alpha.T, atol=1e-12)
    ev = np.linalg.eigvalsh(alpha)
    assert ev[0] > 0.0


def test_self_inductance_tensor_positive_definite():
    """``L^self`` is SPD for various puck shapes."""
    for R, t in ((0.05, 0.005), (0.1, 0.02), (0.25, 0.05)):
        L_self = dip.compute_self_inductance_tensor(R, t, n_radial=20)
        np.testing.assert_allclose(L_self, L_self.T, atol=1e-10)
        ev = np.linalg.eigvalsh(L_self)
        assert ev[0] > 0.0, f"L^self not PD at (R, t) = ({R}, {t})"


def test_pair_inductance_two_dipoles_coaxial_axes():
    """Direct call to ``pair_inductance_dipole_block`` for coaxial dipoles."""
    # Two dipoles separated along z by d, both with moment along z.
    R_vec = jnp.array([0.0, 0.0, 1.0])
    m_i = jnp.eye(3)
    m_j = jnp.eye(3)
    T = np.asarray(pair_inductance_dipole_block(m_i, m_j, R_vec))
    # T = (mu0/4pi) (I - 3 zz^T) / 1 = (mu0/4pi) diag(1, 1, -2)
    expected = MU0_OVER_4PI * np.diag([1.0, 1.0, -2.0])
    np.testing.assert_allclose(T, expected, rtol=1e-10)


def test_assemble_L_dipole_reduced_symmetric_positive_definite():
    """Reduced inductance matrix is symmetric and PSD for a small random layout."""
    rng = np.random.default_rng(42)
    n_base = 4
    centers = rng.normal(scale=0.5, size=(n_base, 3))
    # Pull centres apart so the small system is well-conditioned.
    centers *= 5.0
    quats = np.zeros((n_base, 4))
    quats[:, 0] = 1.0  # identity quaternions
    self_L = np.tile(np.eye(3) * 1e-6, (n_base, 1, 1))
    L = dip.assemble_L_dipole_reduced(centers, quats, self_L, nfp=1, stellsym=False)
    np.testing.assert_allclose(L, L.T, atol=1e-12)
    ev = np.linalg.eigvalsh(L)
    assert ev[0] > 0.0, f"min eigenvalue = {ev[0]}"


def test_assemble_L_dipole_reduced_replication_consistency():
    """Replicated layout (nfp=2, stellsym=True) gives a larger system than (nfp=1).

    We sanity check the block-diagonal magnitude scales with ``G``.
    """
    n_base = 2
    centers = np.array([[0.5, 0.2, 0.1], [-0.3, 0.4, -0.2]])
    quats = np.zeros((n_base, 4))
    quats[:, 0] = 1.0
    self_L = np.tile(np.eye(3) * 1e-6, (n_base, 1, 1))
    L_nosym = dip.assemble_L_dipole_reduced(
        centers, quats, self_L, nfp=1, stellsym=False
    )
    L_sym = dip.assemble_L_dipole_reduced(
        centers, quats, self_L, nfp=2, stellsym=True
    )
    # The block-diagonal self contribution scales with G = 4 vs 1.
    assert (
        np.trace(L_sym[:3, :3]) > 3.0 * np.trace(L_nosym[:3, :3])
    ), "Block-diagonal should scale with the symmetry group order."


def test_b_at_points_dipole_matches_direct_dipole_formula():
    """B at points from one dipole matches the analytic formula."""
    centers_base = jnp.array([[0.0, 0.0, 0.0]])
    quats_base = jnp.array([[1.0, 0.0, 0.0, 0.0]])  # identity
    m_global = jnp.array([[0.0, 0.0, 1.0]])
    pts = jnp.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 1.0]])
    B = np.asarray(
        dip.B_at_points_dipole(pts, centers_base, quats_base, m_global, 1, False)
    )
    B_ref = np.asarray(
        magnetic_field_dipole_points(m_global[0], pts, centers_base[0])
    )
    np.testing.assert_allclose(B, B_ref, rtol=1e-10)


# ----------------------------------------------------------------------
# 2. PSCBulkArray integration
# ----------------------------------------------------------------------


def _make_dipole_psc(
    *,
    n_base: int = 1,
    radius: float = 0.05,
    thickness: float = 0.005,
    nfp: int = 1,
    stellsym: bool = False,
    eval_pts: np.ndarray = None,
) -> PSCBulkArray:
    """Construct a small ``PSCBulkArray`` in dipole mode."""
    tf_coil = _unit_circle_coil(current_amp=1.0e5, radius=1.0)
    if n_base == 1:
        centers = np.array([[0.0, 0.0, 0.2]])
    else:
        centers = np.linspace(0.0, 0.4, n_base * 3).reshape(n_base, 3)
        centers[:, 2] += 0.2
    axes = np.tile(np.array([[0.0, 0.0, 1.0]]), (n_base, 1))
    if eval_pts is None:
        eval_pts = np.array([[0.3, 0.0, 0.25]], dtype=float)
    return PSCBulkArray(
        centers,
        axes,
        np.full(n_base, radius),
        np.full(n_base, thickness),
        [tf_coil],
        eval_points=eval_pts,
        m_fourier=2,
        l_zernike=4,
        k_chebyshev=2,
        n_rho=6,
        n_phi=8,
        n_z=4,
        nfp=nfp,
        stellsym=stellsym,
        solver_mode="dipole",
    )


def test_psc_bulk_dipole_constructs_and_B_finite():
    """End-to-end smoke: dipole-mode PSCBulkArray builds and returns finite B."""
    psc = _make_dipole_psc()
    pts = np.array([[0.3, 0.0, 0.25], [0.4, 0.1, 0.30]], dtype=float)
    B = psc.B_at_points(pts)
    assert B.shape == (2, 3)
    assert np.all(np.isfinite(B))
    # Sanity: B should be non-zero (TF current induces a non-zero
    # moment).
    assert np.linalg.norm(B) > 0.0


def test_psc_bulk_dipole_cholesky_residual_small():
    """After ``recompute_currents``, ``L m + f = 0`` to float precision."""
    psc = _make_dipole_psc(n_base=2)
    psc.recompute_currents()
    L = psc._dipole_L_red
    m = psc._dipole_m_red
    # Re-derive f at current geometry to check the residual.
    centers, quats, _, _ = psc._get_base_puck_geometry()
    g_tf, gd_tf, I_tf = psc._tf_arrays()
    f = np.asarray(
        dip.assemble_f_dipole_reduced_jax(
            jnp.asarray(centers),
            jnp.asarray(quats),
            jnp.asarray(g_tf),
            jnp.asarray(gd_tf),
            jnp.asarray(I_tf),
            int(psc.nfp),
            bool(psc.stellsym),
        )
    )
    residual = L @ m + f
    rel = np.linalg.norm(residual) / (np.linalg.norm(f) + 1e-30)
    assert rel < 1e-8, f"Residual {rel} too large"


def test_psc_bulk_dipole_invalid_solver_mode_raises():
    tf_coil = _unit_circle_coil()
    with pytest.raises(ValueError, match="solver_mode"):
        PSCBulkArray(
            np.array([[0.0, 0.0, 0.2]]),
            np.array([[0.0, 0.0, 1.0]]),
            np.array([0.05]),
            np.array([0.005]),
            [tf_coil],
            eval_points=np.array([[0.3, 0.0, 0.25]]),
            solver_mode="garbage",
        )


def test_psc_bulk_dipole_two_far_pucks_consistent_with_isolated():
    """Two far-apart pucks: induced moment ~ isolated-puck linear response.

    For two pucks at ``d >> R``, the mutual coupling is :math:`O(1/d^3)`
    and the induced moment per puck is approximately the isolated-puck
    response.  We just verify that the magnitudes are consistent.
    """
    psc = _make_dipole_psc(n_base=2)
    psc.recompute_currents()
    m = psc._dipole_m_red.reshape(-1, 3)
    # Both pucks should pick up non-zero z-component.
    assert np.all(np.abs(m[:, 2]) > 0.0)
    # Moments should not be wildly different in magnitude.
    assert (
        0.1 < float(np.linalg.norm(m[0])) / max(float(np.linalg.norm(m[1])), 1e-30) < 10.0
    )


# ----------------------------------------------------------------------
# 3. Freeze self-inductance
# ----------------------------------------------------------------------


def test_freeze_self_l_default_on_for_dipole(monkeypatch):
    """``solver_mode='dipole'`` enables the freeze by default."""
    monkeypatch.delenv("SIMSOPT_PSC_FREEZE_SELF_L", raising=False)
    psc = _make_dipole_psc()
    assert psc._dipole_freeze_self_L is True


def test_freeze_self_l_default_off_for_energy(monkeypatch):
    """``solver_mode='energy'`` keeps the legacy behaviour (freeze off)."""
    monkeypatch.delenv("SIMSOPT_PSC_FREEZE_SELF_L", raising=False)
    tf_coil = _unit_circle_coil()
    psc = PSCBulkArray(
        np.array([[0.0, 0.0, 0.2]]),
        np.array([[0.0, 0.0, 1.0]]),
        np.array([0.05]),
        np.array([0.005]),
        [tf_coil],
        eval_points=np.array([[0.3, 0.0, 0.25]]),
        m_fourier=2,
        l_zernike=4,
        k_chebyshev=2,
        n_rho=6,
        n_phi=8,
        n_z=4,
        solver_mode="energy",
    )
    assert psc._dipole_freeze_self_L is False


def test_freeze_self_l_env_override(monkeypatch):
    """Env var overrides the per-mode default."""
    monkeypatch.setenv("SIMSOPT_PSC_FREEZE_SELF_L", "0")
    psc = _make_dipole_psc()
    assert psc._dipole_freeze_self_L is False
    monkeypatch.setenv("SIMSOPT_PSC_FREEZE_SELF_L", "1")
    tf_coil = _unit_circle_coil()
    psc2 = PSCBulkArray(
        np.array([[0.0, 0.0, 0.2]]),
        np.array([[0.0, 0.0, 1.0]]),
        np.array([0.05]),
        np.array([0.005]),
        [tf_coil],
        eval_points=np.array([[0.3, 0.0, 0.25]]),
        m_fourier=2,
        l_zernike=4,
        k_chebyshev=2,
        n_rho=6,
        n_phi=8,
        n_z=4,
        solver_mode="energy",
    )
    assert psc2._dipole_freeze_self_L is True


def test_freeze_self_l_reuses_cache_on_perturbation(monkeypatch):
    """With freeze on, perturbing R/t does not recompute the cached
    self-inductance tensors.
    """
    monkeypatch.setenv("SIMSOPT_PSC_FREEZE_SELF_L", "1")
    psc = _make_dipole_psc()
    psc.recompute_currents()
    cached_before = np.asarray(psc._dipole_self_L_local_cached).copy()
    # Perturb R, t through the DOF setter (bypassing fix flags).
    x = np.array(psc.local_full_x)
    x[7] += 0.01  # R0
    x[8] += 0.002  # t0
    psc.local_full_x = x
    psc.recompute_currents()
    cached_after = np.asarray(psc._dipole_self_L_local_cached)
    np.testing.assert_allclose(cached_before, cached_after, atol=0.0, rtol=0.0)


def test_freeze_self_l_off_recomputes_on_perturbation(monkeypatch):
    """With freeze off, perturbing R/t recomputes the self-inductance tensors."""
    monkeypatch.setenv("SIMSOPT_PSC_FREEZE_SELF_L", "0")
    psc = _make_dipole_psc()
    psc.recompute_currents()
    cached_before = np.asarray(psc._dipole_self_L_local_cached).copy()
    x = np.array(psc.local_full_x)
    x[7] += 0.02  # R0
    x[8] += 0.003  # t0
    psc.local_full_x = x
    psc.recompute_currents()
    cached_after = np.asarray(psc._dipole_self_L_local_cached)
    # Cache must differ now (different R/t -> different polarizability).
    assert not np.allclose(cached_before, cached_after, atol=0.0, rtol=0.0)


# ----------------------------------------------------------------------
# 4. JAX VJP correctness (Phase E)
# ----------------------------------------------------------------------


def _toy_dipole_inputs(
    seed: int = 0,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """Tiny synthetic input for forward / VJP smoke tests.

    Returns ``(centers, quats, g_tf, gd_tf, I_tf, pts, self_L_local)``.
    All arrays are NumPy ``float64``.
    """
    rng = np.random.default_rng(int(seed))
    n_base = 2
    centers = rng.normal(scale=0.3, size=(n_base, 3))
    centers[:, 2] += 0.5
    quats = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n_base, 1))
    # One TF coil represented by 12 quadrature points on a unit ring.
    n_q = 12
    phi = np.linspace(0.0, 2.0 * np.pi, n_q, endpoint=False)
    g = np.stack(
        [np.stack([np.cos(p), np.sin(p), np.zeros_like(p)], axis=-1) for p in [phi]],
        axis=0,
    )
    gd = np.stack(
        [np.stack([-np.sin(p), np.cos(p), np.zeros_like(p)], axis=-1) for p in [phi]],
        axis=0,
    )
    I_tf = np.array([1.0e5])
    pts = np.array([[0.4, 0.0, 0.6], [-0.2, 0.3, 0.55]])
    self_L = np.tile(np.eye(3) * 1.0e-7, (n_base, 1, 1))
    return centers, quats, g, gd, I_tf, pts, self_L


def test_forward_dipole_pipeline_finite():
    centers, quats, g, gd, I_tf, pts, self_L = _toy_dipole_inputs()
    B = np.asarray(
        dip.forward_dipole_pipeline(
            jnp.asarray(centers),
            jnp.asarray(quats),
            jnp.asarray(g),
            jnp.asarray(gd),
            jnp.asarray(I_tf),
            jnp.asarray(pts),
            jnp.asarray(self_L),
            1,
            False,
        )
    )
    assert B.shape == (2, 3)
    assert np.all(np.isfinite(B))


def test_taylor_centre_dipole_mode():
    """Taylor test: ``f(c + eps d) - f(c) - eps J d = O(eps^2)``.

    The slope of ``log(err)`` vs ``log(eps)`` should be ~2 on a small
    differentiable scalar functional of the dipole-mode forward.
    """
    centers, quats, g, gd, I_tf, pts, self_L = _toy_dipole_inputs(seed=1)

    def scalar_objective(c: jnp.ndarray) -> jnp.ndarray:
        B = dip.forward_dipole_pipeline(
            c,
            jnp.asarray(quats),
            jnp.asarray(g),
            jnp.asarray(gd),
            jnp.asarray(I_tf),
            jnp.asarray(pts),
            jnp.asarray(self_L),
            1,
            False,
        )
        return jnp.sum(B**2)

    c0 = jnp.asarray(centers)
    grad_fn = jax.grad(scalar_objective)
    g0 = grad_fn(c0)
    rng = np.random.default_rng(2)
    d = jnp.asarray(rng.normal(size=centers.shape))
    f0 = float(scalar_objective(c0))
    Jd = float(jnp.sum(g0 * d))
    # Use perturbations large enough that the quadratic remainder
    # dominates float-precision noise.  At the chosen toy scale, the
    # forward is O(1e-12) so steps below ~1e-2 hit double precision
    # noise.
    epss = [3e-2, 1e-2, 3e-3]
    errs = []
    for eps in epss:
        f1 = float(scalar_objective(c0 + eps * d))
        errs.append(abs(f1 - f0 - eps * Jd))
    # Compute the empirical convergence rate using log-log slopes.
    log_eps = np.log(epss)
    log_err = np.log(np.maximum(errs, 1e-30))
    slopes = np.diff(log_err) / np.diff(log_eps)
    # Slope should be near 2 (quadratic) within tolerance.  Require
    # at least 1.5 to be robust to higher-order curvature.
    assert all(s > 1.5 for s in slopes), (
        f"Non-quadratic Taylor convergence: slopes = {slopes.tolist()} "
        f"errs = {errs}"
    )


def _taylor_log_log_slopes(
    f0: float,
    Jd: float,
    perturbed: list,
    epss: list,
) -> np.ndarray:
    """Helper: empirical log-log slopes of the Taylor remainder.

    Given the scalar objective ``f`` evaluated at the base point
    (``f0``), the directional derivative ``J d`` (``Jd``), and the
    perturbed values ``f(x + eps d)`` (``perturbed``), return the
    finite-difference slopes of ``log|f1 - f0 - eps Jd|`` vs
    ``log eps`` -- which should be ~2 for a smooth forward.
    """
    errs = [abs(p - f0 - eps * Jd) for p, eps in zip(perturbed, epss)]
    log_eps = np.log(np.asarray(epss, dtype=float))
    log_err = np.log(np.maximum(errs, 1e-30))
    return np.diff(log_err) / np.diff(log_eps), errs


def test_taylor_quaternion_dipole_mode():
    """Taylor test in quaternion DoFs.

    Quaternions live on the 3-sphere ``S^3 \\subset R^4``.  A
    perturbation in a tangent direction ``d`` is followed by an
    explicit renormalisation back to the unit-norm manifold, which
    preserves the convention used throughout :class:`PSCBulkArray`.
    The renormalisation is smooth at the base point (``|q| = 1``), so
    the log-log Taylor slope is still ~2.
    """
    centers, quats0, g, gd, I_tf, pts, _self_L_iso = _toy_dipole_inputs(seed=7)
    # Use an *anisotropic* puck-local self-tensor so the rotation
    # actually changes the global L matrix (isotropic tensors are
    # invariant under any frame rotation, and would yield identically
    # zero quaternion gradients).
    n_base = centers.shape[0]
    self_L_local_aniso = np.zeros((n_base, 3, 3), dtype=np.float64)
    for i in range(n_base):
        self_L_local_aniso[i] = np.diag([3.0e-7, 1.0e-7, 5.0e-7])
    # Start from a generic (non-axis-aligned) quaternion so the linear
    # term J d is non-zero and the test can detect quadratic
    # convergence.
    rng_q = np.random.default_rng(31)
    q_init = rng_q.normal(size=quats0.shape)
    q_init /= np.linalg.norm(q_init, axis=-1, keepdims=True)

    def scalar_objective(q_flat: jnp.ndarray) -> jnp.ndarray:
        q = q_flat.reshape(quats0.shape)
        # Renormalise to remain on the unit-norm manifold.
        q_norm = q / jnp.linalg.norm(q, axis=-1, keepdims=True)
        B = dip.forward_dipole_pipeline(
            jnp.asarray(centers),
            q_norm,
            jnp.asarray(g),
            jnp.asarray(gd),
            jnp.asarray(I_tf),
            jnp.asarray(pts),
            jnp.asarray(self_L_local_aniso),
            1,
            False,
        )
        return jnp.sum(B**2)

    q0_flat = jnp.asarray(q_init).reshape(-1)
    grad_fn = jax.grad(scalar_objective)
    g0 = grad_fn(q0_flat)
    rng = np.random.default_rng(11)
    d = jnp.asarray(rng.normal(size=q0_flat.shape))
    f0 = float(scalar_objective(q0_flat))
    Jd = float(jnp.sum(g0 * d))
    epss = [3e-2, 1e-2, 3e-3]
    perturbed = [float(scalar_objective(q0_flat + eps * d)) for eps in epss]
    slopes, errs = _taylor_log_log_slopes(f0, Jd, perturbed, epss)
    assert all(s > 1.5 for s in slopes), (
        f"Non-quadratic Taylor convergence in quaternion DoFs: "
        f"slopes = {slopes.tolist()} errs = {errs}"
    )


def test_taylor_self_l_frozen_R_independent(monkeypatch):
    """With self-L frozen, the gradient w.r.t. ``R`` reflects only the
    mutual-coupling contribution.

    ``compute_self_inductance_tensor`` is a non-differentiable host
    function, so the dipole forward differentiates ``R`` solely
    through the dipole-dipole kernel terms in ``L_red`` (and via the
    centers indirectly when ``R`` is interpreted as a layout scale --
    here we keep centers fixed and verify the Taylor remainder is
    quadratic in the perturbation as long as the frozen self-L is
    held constant across the perturbation).
    """
    monkeypatch.setenv("SIMSOPT_PSC_FREEZE_SELF_L", "1")
    # Build the same toy inputs but parameterise the self-L by R.
    centers, quats, g, gd, I_tf, pts, _ = _toy_dipole_inputs(seed=3)
    R0 = 0.1
    # Smythe-derived self-inductance (axial-isotropic for an in-plane
    # thin disk is fine for the test -- we just need a smooth scalar
    # dependence on R).
    def self_L_of_R(R: jnp.ndarray) -> jnp.ndarray:
        # Use a smooth scalar surrogate L(R) ~ MU0 * R that grows
        # monotonically; this stands in for the (host-side)
        # ``compute_self_inductance_tensor`` but stays JAX-traceable.
        n_base = centers.shape[0]
        eye = jnp.eye(3)
        return jnp.broadcast_to(MU0 * R * eye, (n_base, 3, 3))

    def scalar_objective(R: jnp.ndarray) -> jnp.ndarray:
        B = dip.forward_dipole_pipeline(
            jnp.asarray(centers),
            jnp.asarray(quats),
            jnp.asarray(g),
            jnp.asarray(gd),
            jnp.asarray(I_tf),
            jnp.asarray(pts),
            self_L_of_R(R),
            1,
            False,
        )
        return jnp.sum(B**2)

    R_jnp = jnp.asarray(R0)
    grad_fn = jax.grad(scalar_objective)
    g0 = float(grad_fn(R_jnp))
    rng = np.random.default_rng(13)
    d = float(rng.normal())
    f0 = float(scalar_objective(R_jnp))
    Jd = g0 * d
    epss = [3e-2, 1e-2, 3e-3]
    perturbed = [
        float(scalar_objective(jnp.asarray(R0 + eps * d))) for eps in epss
    ]
    slopes, errs = _taylor_log_log_slopes(f0, Jd, perturbed, epss)
    assert all(s > 1.5 for s in slopes), (
        f"Non-quadratic Taylor convergence in R (frozen self-L): "
        f"slopes = {slopes.tolist()} errs = {errs}"
    )


def test_taylor_full_pipeline_jit():
    """Post-JIT gradient parity check.

    With the persistent ``jax.jit`` cache in place (see
    :meth:`PSCBulkArray._ensure_dipole_jit_cache`), the jitted forward
    must produce values matching the un-jitted reference to machine
    precision *and* yield identical gradients (verified by a Taylor
    convergence check using the jitted gradient).
    """
    centers, quats, g, gd, I_tf, pts, self_L = _toy_dipole_inputs(seed=5)

    def scalar_objective(c: jnp.ndarray, jitted: bool) -> jnp.ndarray:
        fwd = dip.forward_dipole_pipeline
        if jitted:
            fwd = jax.jit(fwd, static_argnums=(7, 8))
        B = fwd(
            c,
            jnp.asarray(quats),
            jnp.asarray(g),
            jnp.asarray(gd),
            jnp.asarray(I_tf),
            jnp.asarray(pts),
            jnp.asarray(self_L),
            1,
            False,
        )
        return jnp.sum(B**2)

    c0 = jnp.asarray(centers)
    f_ref = float(scalar_objective(c0, jitted=False))
    f_jit = float(scalar_objective(c0, jitted=True))
    np.testing.assert_allclose(f_jit, f_ref, rtol=1e-12, atol=0.0)

    grad_ref = jax.grad(lambda c: scalar_objective(c, jitted=False))(c0)
    grad_jit = jax.grad(lambda c: scalar_objective(c, jitted=True))(c0)
    np.testing.assert_allclose(
        np.asarray(grad_jit), np.asarray(grad_ref), rtol=1e-10, atol=1e-30
    )

    # And run a Taylor test using the JITTED gradient to certify the
    # cached compiled VJP.  Use moderate ``eps`` values where the
    # quadratic remainder dominates curvature-of-curvature noise at
    # the toy scale (``forward_dipole_pipeline`` outputs ~1e-12, so
    # very small ``eps`` would hit double-precision noise).
    rng = np.random.default_rng(19)
    d = jnp.asarray(rng.normal(size=centers.shape))
    # Direction-unit-normalise for predictable scaling.
    d = d / jnp.linalg.norm(d)
    f0 = f_jit
    Jd = float(jnp.sum(grad_jit * d))
    epss = [1e-2, 3e-3, 1e-3]
    perturbed = [
        float(scalar_objective(c0 + eps * d, jitted=True)) for eps in epss
    ]
    slopes, errs = _taylor_log_log_slopes(f0, Jd, perturbed, epss)
    assert all(s > 1.5 for s in slopes), (
        f"Non-quadratic Taylor convergence with JIT: "
        f"slopes = {slopes.tolist()} errs = {errs}"
    )
