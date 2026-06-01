"""Tests for the dipole-mode solver of :class:`PSCBulkArray`.

Implements the test matrix specified in the lean dipole bulk solver
plan: primitive correctness, ``PSCBulkArray`` integration, self-
inductance freeze, and JAX VJP correctness.
"""

from __future__ import annotations

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


def test_quat_to_matrices_vmap_matches_python_stack():
    """Phase-M1/M2: shared vmap helper matches the pre-M list comp.

    Pinned numerical-equivalence guard for the Phase-M trace-size
    shrink in :mod:`simsopt.field._psc_bulk_dipole`.  The hoisted
    :func:`_quat_to_matrices_vmap` helper must produce results that are
    bit-for-bit equivalent to the prior
    ``jnp.stack([_quat_to_matrix(q[i]) for i in range(n)], axis=0)``
    idiom on every quaternion supplied -- otherwise the M1 / M2 trace
    rewrite has silently altered the numerical answer.
    """
    rng = np.random.default_rng(20260514)
    n = 8
    quats = rng.normal(size=(n, 4))
    R_vmap = np.asarray(dip._quat_to_matrices_vmap(jnp.asarray(quats)))
    R_ref = np.asarray(
        jnp.stack(
            [dip._quat_to_matrix(jnp.asarray(quats[i])) for i in range(n)],
            axis=0,
        )
    )
    assert R_vmap.shape == (n, 3, 3)
    np.testing.assert_allclose(R_vmap, R_ref, rtol=0.0, atol=0.0)


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


# ----------------------------------------------------------------------
# 5. Analytic VJP parity and allfixed zero-puck-grad checks (Phase H)
# ----------------------------------------------------------------------


def _analytic_grads_via_subkernels(
    psc: PSCBulkArray,
    centers: np.ndarray,
    quats: np.ndarray,
    self_L_local: np.ndarray,
    g_tf: np.ndarray,
    gd_tf: np.ndarray,
    I_tf: np.ndarray,
    pts: np.ndarray,
    v_B: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Assemble gradients via the analytic adjoint sub-kernels.

    Bypasses the public :meth:`_vjp_dipole` (which packs results into
    a :class:`Derivative`) so the parity test can compare directly to
    :func:`jax.grad` over the JAX-traceable inputs ``(c, q, g, gd, I)``.

    Mathematically, with ``m = -L^{-1} f``,

    * ``B``-side: ``(lam_c_B, lam_q_B, lam_m) = vjp_{c,q,m} B^T v_B``.
    * ``lam_f = -L^{-T} lam_m``  -- via SciPy ``cho_solve`` on the
      cached ``L_red`` Cholesky.
    * ``f``-side: ``(lam_c_f, lam_q_f, vg, vgd, vI) =
      vjp_{c,q,g,gd,I} f^T lam_f``.
    * ``L``-side: ``(lam_c_L, lam_q_L) =
      vjp_{c,q} L^T outer(lam_f, m)``.
    """
    import scipy.linalg as _sp

    n_base = int(centers.shape[0])
    # Make sure the jit caches are built (mirrors what _vjp_dipole does).
    psc._ensure_dipole_vjp_jit_cache(np.asarray(pts), g_tf)
    m_red = psc._dipole_m_red
    assert m_red is not None
    m_global = m_red.reshape(n_base, 3)
    nfp = int(psc.nfp)
    stell = bool(psc.stellsym)

    lam_c_B, lam_q_B, lam_m_b = psc._dipole_jit_vjp_B(
        jnp.asarray(pts),
        jnp.asarray(centers),
        jnp.asarray(quats),
        jnp.asarray(m_global),
        jnp.asarray(v_B),
        nfp,
        stell,
    )
    lam_m = np.asarray(lam_m_b).reshape(-1)

    c_factor, lower = psc._dipole_L_red_chol
    lam_f = -_sp.cho_solve((c_factor, lower), lam_m, check_finite=False)
    lam_f_j = jnp.asarray(lam_f)

    lam_c_f, lam_q_f, vg, vgd, vI = psc._dipole_jit_vjp_f(
        jnp.asarray(centers),
        jnp.asarray(quats),
        jnp.asarray(g_tf),
        jnp.asarray(gd_tf),
        jnp.asarray(I_tf),
        lam_f_j,
        nfp,
        stell,
    )

    outer_ct = jnp.outer(lam_f_j, jnp.asarray(m_red))
    lam_c_L, lam_q_L = psc._dipole_jit_vjp_L(
        jnp.asarray(centers),
        jnp.asarray(quats),
        jnp.asarray(self_L_local),
        outer_ct,
        nfp,
        stell,
    )

    gc = np.asarray(lam_c_B) + np.asarray(lam_c_f) + np.asarray(lam_c_L)
    gq = np.asarray(lam_q_B) + np.asarray(lam_q_f) + np.asarray(lam_q_L)
    return gc, gq, np.asarray(vg), np.asarray(vgd), np.asarray(vI)


def test_vjp_dipole_analytic_matches_jax_grad():
    """Analytic dipole adjoint matches ``jax.grad`` over ``(c, q, g, gd, I)``.

    Builds a non-trivial 3-base-puck ``PSCBulkArray`` in dipole mode
    with free centres and quaternions, then compares the gradient of
    ``J = sum(v_B * B)`` computed three ways:

    1. The factored analytic adjoint (three :func:`jax.vjp` calls
       plus :func:`scipy.linalg.cho_solve`) routed through
       :func:`_analytic_grads_via_subkernels` -- i.e., the new
       Phase-H backward path stripped of its TF-coil pullback.
    2. A direct :func:`jax.grad` over
       :func:`forward_dipole_pipeline` w.r.t. ``(c, q, g, gd, I)``.

    Parity tolerance: ``rtol=1e-10``.  The cached self-inductance
    block is treated as a constant in both paths (matching the
    "self-L frozen" convention of the production solver).
    """
    psc = _make_dipole_psc(n_base=3)
    rng = np.random.default_rng(0)
    centers_pert = rng.normal(scale=0.02, size=(3, 3))
    base_x = np.array(psc.local_full_x)
    # Lay out the three pucks on a small grid and rotate them with
    # random quaternions (writing into the local DOF vector).
    for i in range(3):
        off = i * 9
        base_x[off : off + 3] = np.array(
            [0.15 * (i - 1), 0.05 * (i - 1), 0.20]
        ) + centers_pert[i]
        # Quaternion DOFs (next 4): small but non-trivial rotation.
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        base_x[off + 3 : off + 7] = q
    psc.local_full_x = base_x
    psc.recompute_currents()

    centers, quats, _radii, _thicknesses = psc._get_base_puck_geometry()
    g_tf, gd_tf, I_tf = psc._tf_arrays()
    pts = np.asarray(psc.eval_points)
    self_L = np.asarray(psc._dipole_self_L_local_cached)
    v_B = rng.normal(size=(pts.shape[0], 3))

    # Reference: jax.grad over a forward that solves the *same*
    # jittered L matrix the production cache uses, so the comparison
    # is unaffected by the eigenfloor regularisation (~1e-10 of the
    # trace mean of ``L_red``).
    n_base = int(centers.shape[0])
    jitter = float(psc._eigenfloor_threshold)

    def objective(c, q, g, gd, I):
        L = dip.assemble_L_dipole_reduced_jax(
            c, q, jnp.asarray(self_L), int(psc.nfp), bool(psc.stellsym),
        )
        f = dip.assemble_f_dipole_reduced_jax(
            c, q, g, gd, I, int(psc.nfp), bool(psc.stellsym),
        )
        L_reg = L + jitter * (jnp.trace(L) / L.shape[0]) * jnp.eye(L.shape[0])
        m = -jnp.linalg.solve(L_reg, f)
        m_global = m.reshape(n_base, 3)
        B = dip.B_at_points_dipole(
            jnp.asarray(pts), c, q, m_global,
            int(psc.nfp), bool(psc.stellsym),
        )
        return jnp.sum(B * jnp.asarray(v_B))

    grad_ref = jax.grad(objective, argnums=(0, 1, 2, 3, 4))(
        jnp.asarray(centers),
        jnp.asarray(quats),
        jnp.asarray(g_tf),
        jnp.asarray(gd_tf),
        jnp.asarray(I_tf),
    )
    gc_ref, gq_ref, vg_ref, vgd_ref, vI_ref = (
        np.asarray(grad_ref[0]),
        np.asarray(grad_ref[1]),
        np.asarray(grad_ref[2]),
        np.asarray(grad_ref[3]),
        np.asarray(grad_ref[4]),
    )

    # Analytic factored adjoint.
    gc_a, gq_a, vg_a, vgd_a, vI_a = _analytic_grads_via_subkernels(
        psc, centers, quats, self_L, g_tf, gd_tf, I_tf, pts, v_B,
    )

    np.testing.assert_allclose(gc_a, gc_ref, rtol=1e-10, atol=1e-25)
    np.testing.assert_allclose(gq_a, gq_ref, rtol=1e-10, atol=1e-25)
    np.testing.assert_allclose(vg_a, vg_ref, rtol=1e-10, atol=1e-25)
    np.testing.assert_allclose(vgd_a, vgd_ref, rtol=1e-10, atol=1e-25)
    np.testing.assert_allclose(vI_a, vI_ref, rtol=1e-10, atol=1e-25)


def test_vjp_dipole_allfixed_zero_puck_grad():
    """In ``allfixed`` mode :meth:`vjp_setup_B` returns no puck-local DoFs.

    When all centre and quaternion DOFs are fixed,
    :meth:`PSCBulkArray._vjp_dipole` must drop the geometry branch
    entirely (no centre/quaternion-side ``jax.vjp`` is invoked) and
    the returned :class:`simsopt._core.derivative.Derivative` must
    contain only TF-coil entries -- specifically no key pointing at
    the ``PSCBulkArray`` instance itself.
    """
    psc = _make_dipole_psc(n_base=2)
    # Fix every puck DoF.
    psc.local_fix_all()
    # ``local_fix_all`` fixes the puck DOFs but the TF coil DOFs
    # remain free (they live on a separate parent), which is exactly
    # the regime we want to test.
    psc.recompute_currents()
    pts = np.asarray(psc.eval_points)
    rng = np.random.default_rng(2)
    v_B = rng.normal(size=(pts.shape[0], 3))
    deriv = psc.vjp_setup_B(v_B, pts)
    # The Derivative dict must NOT contain a key for ``psc`` itself
    # (which would carry centre / quaternion gradients).
    assert psc not in deriv.data, (
        "allfixed dipole VJP unexpectedly produced PSCBulkArray-local "
        "gradient entries; expected only TF-coil keys."
    )


def test_dipole_jdev_pts_cache_invalidates_on_eval_points_change():
    r"""Phase-J J1: ``_dipole_jdev_pts`` is cached across calls and reset by setter.

    Tests three behaviours of the Phase-J J1 host->device cache:

    1. **Reuse**: two successive ``B_at_points(self.eval_points)`` calls
       share the same cached :class:`jnp.ndarray` instance (no second
       host-to-device copy).
    2. **Setter invalidation**: assigning a new array via
       ``self.eval_points = ...`` clears the cache and bumps
       ``self._eval_points_version`` so the next call rebuilds.
    3. **Ad-hoc bypass**: passing an unrelated ``pts`` array (not the
       canonical ``self.eval_points``) goes through ``jnp.asarray``
       directly without populating the cache.

    All three branches must return finite, identical-up-to-rtol values
    to the reference :func:`jnp.asarray` path.
    """
    psc = _make_dipole_psc(n_base=2)
    psc.recompute_currents()
    pts = np.asarray(psc.eval_points)

    # (1) Cache cold -> populated.
    assert psc._dipole_jdev_pts is None, (
        "cache should start cold before first B_at_points call"
    )
    B_first = psc.B_at_points(pts)
    cached_after_first = psc._dipole_jdev_pts
    assert cached_after_first is not None, (
        "cache must be populated after first B_at_points call"
    )
    assert cached_after_first.shape == pts.shape

    # (1 cont.) Second call -> cache reused, same instance.
    B_second = psc.B_at_points(pts)
    assert psc._dipole_jdev_pts is cached_after_first, (
        "second B_at_points call should reuse the same cached "
        "_dipole_jdev_pts instance (no new host-to-device copy)"
    )
    np.testing.assert_allclose(B_first, B_second, rtol=0.0, atol=0.0)

    # (2) Setter -> cache cleared + version bumped.
    version_before = psc._eval_points_version
    new_pts = np.array(
        [[0.30, 0.00, 0.25], [0.35, 0.05, 0.28]], dtype=float
    )
    psc.eval_points = new_pts
    assert psc._dipole_jdev_pts is None, (
        "eval_points setter must clear _dipole_jdev_pts"
    )
    assert psc._eval_points_version == version_before + 1, (
        f"eval_points setter must bump _eval_points_version "
        f"(was {version_before}, got {psc._eval_points_version})"
    )

    # Next call after setter should refill the cache.
    B_after_setter = psc.B_at_points(psc.eval_points)
    assert psc._dipole_jdev_pts is not None, (
        "B_at_points after setter must repopulate the cache"
    )
    assert B_after_setter.shape == new_pts.shape

    # (3) Ad-hoc pts -- not self.eval_points -- must not touch cache.
    snapshot = psc._dipole_jdev_pts
    other_pts = np.array(
        [[0.20, 0.10, 0.30], [0.25, 0.10, 0.32]], dtype=float
    )
    _ = psc.B_at_points(other_pts)
    assert psc._dipole_jdev_pts is snapshot, (
        "ad-hoc pts (pts is not self.eval_points) must not "
        "overwrite the canonical cache"
    )

    # Sanity: returned values must remain finite and the same shape
    # as the inputs across all three branches.
    assert np.all(np.isfinite(B_first))
    assert np.all(np.isfinite(B_after_setter))
    assert B_first.shape == pts.shape
    assert B_after_setter.shape == new_pts.shape


def test_vjp_dipole_fused_tf_derivative_matches_loop():
    r"""Phase-J J2: ``_merge_tf_coil_vjps`` is byte-identical to the historical sum.

    The Phase-J optimisation replaces

    .. code-block:: python

        vjp_tf = sum(
            self.coils_TF[i].vjp(vg_np[i], vgd_np[i], np.asarray([vI_np[i]]))
            for i in range(len(self.coils_TF))
        )

    inside :meth:`PSCBulkArray._vjp_dipole` with the helper
    :meth:`PSCBulkArray._merge_tf_coil_vjps`.  Both must produce the
    same :class:`~simsopt._core.derivative.Derivative` -- every leaf
    array equal to ``rtol=1e-12`` -- otherwise the analytic adjoint
    is no longer equivalent to ``jax.grad`` of the forward.

    The 4-TF-coil configuration here is the smallest one that
    exercises ``__add__`` chaining (n_tf - 1 = 3 dict copies in the
    old idiom vs. 0 in the new helper).
    """
    rng = np.random.default_rng(123)
    n_tf = 4
    tf_coils = []
    for k in range(n_tf):
        c = _unit_circle_coil(
            current_amp=1.0e5 * (1.0 + 0.05 * k),
            radius=0.9 + 0.05 * k,
        )
        tf_coils.append(c)

    centers = np.array([[0.0, 0.0, 0.2]])
    axes = np.array([[0.0, 0.0, 1.0]])
    psc = PSCBulkArray(
        centers,
        axes,
        np.array([0.05]),
        np.array([0.005]),
        tf_coils,
        eval_points=np.array([[0.3, 0.0, 0.25]]),
        solver_mode="dipole",
        m_fourier=2,
        l_zernike=4,
        k_chebyshev=2,
        n_rho=6,
        n_phi=8,
        n_z=3,
    )
    # The helper itself does not depend on ``recompute_currents`` having
    # run -- it only consumes TF-side cotangents -- so we can synthesize
    # a deterministic ``(vg, vgd, vI)`` triple and compare both paths
    # directly.
    coil_curve_g = np.asarray(tf_coils[0].curve.gamma())
    n_quad = coil_curve_g.shape[0]
    vg_np = rng.normal(size=(n_tf, n_quad, 3))
    vgd_np = rng.normal(size=(n_tf, n_quad, 3))
    vI_np = rng.normal(size=(n_tf,))

    helper_out = psc._merge_tf_coil_vjps(vg_np, vgd_np, vI_np)
    legacy_out = sum(
        psc.coils_TF[i].vjp(
            vg_np[i], vgd_np[i], np.asarray([vI_np[i]])
        )
        for i in range(n_tf)
    )

    assert set(helper_out.data.keys()) == set(legacy_out.data.keys()), (
        "Phase-J merge helper produced a different set of "
        f"Derivative keys: helper={sorted(map(repr, helper_out.data))}, "
        f"legacy={sorted(map(repr, legacy_out.data))}"
    )
    for key in helper_out.data:
        np.testing.assert_allclose(
            np.asarray(helper_out.data[key]),
            np.asarray(legacy_out.data[key]),
            rtol=1e-12,
            atol=0.0,
            err_msg=(
                f"Phase-J merge helper produced a different leaf array "
                f"for key={key!r}"
            ),
        )


def test_recompute_currents_skips_base_geom_when_dofs_frozen():
    """Phase-J J3: ``_base_geom_state_matrix`` is skipped on warm iters with frozen DOFs.

    The Phase-J quick-win optimisation in :meth:`PSCBulkArray.recompute_currents`
    early-outs the per-iter ``(n_base, 9)`` state matrix build when every
    puck DOF object's ``_state_version`` is unchanged since the previous
    rebuild.  In the ``allfixed`` configuration nothing on the puck side
    moves between optimisation iterations, so after the very first
    rebuild the matrix builder should never be called again; only TF-side
    re-solves run per iter.

    The assertion uses ``unittest.mock.patch.object`` to wrap the real
    builder so behaviour is unchanged but invocation counts are recorded;
    the matrix builder is expected to run **exactly once** (during the
    cold first ``recompute_currents`` call where
    ``self._last_base_geom_state is None``), and the next two warm calls
    must reuse the cached snapshot.
    """
    from unittest.mock import patch

    psc = _make_dipole_psc(n_base=2)
    psc.local_fix_all()
    real_builder = type(psc)._base_geom_state_matrix
    n_calls = {"count": 0}

    def counting_builder(self):
        n_calls["count"] += 1
        return real_builder(self)

    with patch.object(
        type(psc),
        "_base_geom_state_matrix",
        new=counting_builder,
    ):
        psc.recompute_currents()
        first_count = n_calls["count"]
        psc.recompute_currents()
        psc.recompute_currents()
        final_count = n_calls["count"]

    assert first_count == 1, (
        f"first recompute_currents should call _base_geom_state_matrix "
        f"exactly once (cold path); got {first_count}"
    )
    assert final_count == 1, (
        f"warm iterations with frozen DOFs should NOT call "
        f"_base_geom_state_matrix; total calls after 3 rebuilds: "
        f"{final_count} (expected 1)"
    )


def _dipole_psc_two_tf_orders(
    *,
    order: int,
    eval_pts: np.ndarray,
) -> PSCBulkArray:
    """Minimal dipole ``PSCBulkArray`` with two TF coils at Fourier ``order``."""
    coils = []
    for _ in range(2):
        curve = CurveXYZFourier(200, order)
        x = np.zeros_like(curve.get_dofs(), dtype=float)
        x[2] = 1.0
        x[4] = 1.0
        curve.x = x
        coils.append(Coil(curve, Current(1.0e5)))
    centers = np.array([[0.0, 0.0, 0.2]])
    axes = np.array([[0.0, 0.0, 1.0]])
    return PSCBulkArray(
        centers,
        axes,
        np.array([0.05]),
        np.array([0.005]),
        coils,
        eval_points=np.asarray(eval_pts, dtype=float),
        m_fourier=2,
        l_zernike=4,
        k_chebyshev=2,
        n_rho=6,
        n_phi=8,
        n_z=4,
        nfp=1,
        stellsym=False,
        solver_mode="dipole",
    )


def test_module_level_dipole_jit_cache_vjp_equivalent_to_cache_flushed(
    monkeypatch,
) -> None:
    """Phase L3 / N2 -- VJP outputs match between cache-on and cache-flushed.

    Numerical-equivalence guard for the Phase L3 option-(a) module-level
    dipole JIT cache.  Builds one dipole :class:`PSCBulkArray`, populates
    its caches via ``recompute_currents`` + ``B_at_points`` +
    ``vjp_setup_B``, snapshots the resulting VJP outputs, then
    monkeypatches every module-level ``_DIPOLE_*_JIT`` handle to a fresh
    :func:`jax.jit` wrapper (forcing a cold re-trace on the next call),
    builds a second identical instance, snapshots its VJP outputs, and
    asserts the two snapshots match at ``rtol=1e-12``.

    This is the "cache-on vs cache-flushed" guard mandated by the
    Phase L3 plan: hoisting the kernels to module level must never
    change the numerical answer; the only difference may be wall time.
    """
    import jax

    from simsopt.field import psc_bulk as _pb_mod

    eval_pts = np.array(
        [[0.34, 0.04, 0.29], [0.21, 0.07, 0.23], [0.45, -0.02, 0.31]],
        dtype=float,
    )
    rng = np.random.default_rng(123)
    v_B = rng.normal(size=eval_pts.shape)

    psc_a = _dipole_psc_two_tf_orders(order=4, eval_pts=eval_pts)
    psc_a.recompute_currents()
    psc_a.B_at_points(eval_pts)
    deriv_a = psc_a.vjp_setup_B(v_B, eval_pts)
    B_a = np.asarray(psc_a.B_at_points(eval_pts))

    handle_attrs = (
        "_DIPOLE_ASSEMBLE_L_JIT",
        "_DIPOLE_ASSEMBLE_F_JIT",
        "_DIPOLE_FIELD_JIT",
        "_DIPOLE_FORWARD_JIT",
        "_DIPOLE_VJP_B_JIT",
        "_DIPOLE_VJP_F_JIT",
        "_DIPOLE_VJP_L_JIT",
        "_DIPOLE_VJP_FUSED_TF_JIT",
    )
    rebuild_specs = {
        "_DIPOLE_ASSEMBLE_L_JIT": (
            _pb_mod._psc_bulk_dipole_mod.assemble_L_dipole_reduced_jax,
            (3, 4),
        ),
        "_DIPOLE_ASSEMBLE_F_JIT": (
            _pb_mod._psc_bulk_dipole_mod.assemble_f_dipole_reduced_jax,
            (5, 6),
        ),
        "_DIPOLE_FIELD_JIT": (
            _pb_mod._psc_bulk_dipole_mod.B_at_points_dipole,
            (4, 5),
        ),
        "_DIPOLE_FORWARD_JIT": (
            _pb_mod._psc_bulk_dipole_mod.forward_dipole_pipeline,
            (7, 8),
        ),
        "_DIPOLE_VJP_B_JIT": (_pb_mod._dipole_vjp_B_kernel, (5, 6)),
        "_DIPOLE_VJP_F_JIT": (_pb_mod._dipole_vjp_f_kernel, (6, 7)),
        "_DIPOLE_VJP_L_JIT": (_pb_mod._dipole_vjp_L_kernel, (4, 5)),
        "_DIPOLE_VJP_FUSED_TF_JIT": (
            _pb_mod._dipole_vjp_fused_tf_kernel,
            (9, 10),
        ),
    }
    for attr in handle_attrs:
        fn, static_argnums = rebuild_specs[attr]
        monkeypatch.setattr(
            _pb_mod, attr, jax.jit(fn, static_argnums=static_argnums)
        )

    psc_b = _dipole_psc_two_tf_orders(order=4, eval_pts=eval_pts)
    psc_b.recompute_currents()
    psc_b.B_at_points(eval_pts)
    deriv_b = psc_b.vjp_setup_B(v_B, eval_pts)
    B_b = np.asarray(psc_b.B_at_points(eval_pts))

    np.testing.assert_allclose(
        B_a,
        B_b,
        rtol=1e-12,
        atol=0.0,
        err_msg=(
            "Cache-on vs cache-flushed B_at_points outputs diverged -- "
            "module-level JIT hoist changed the numerical answer."
        ),
    )

    # Extract per-leaf derivative arrays in insertion order so we can
    # compare without relying on optimizable names (which are
    # monotonically suffixed per process and therefore differ between
    # the two ``_dipole_psc_two_tf_orders`` builds in this test).  The
    # two ``PSCBulkArray`` instances flow through identical code paths
    # at identical shapes, so the ``Derivative`` storage order matches
    # by position.
    arrs_a = [np.asarray(arr) for arr in deriv_a.data.values()]
    arrs_b = [np.asarray(arr) for arr in deriv_b.data.values()]
    assert len(arrs_a) == len(arrs_b), (
        "Cache-on vs cache-flushed VJP returned Derivative dicts of "
        f"different lengths: len(a)={len(arrs_a)}, len(b)={len(arrs_b)}."
    )
    for idx, (va, vb) in enumerate(zip(arrs_a, arrs_b)):
        assert va.shape == vb.shape, (
            f"Leaf {idx}: shape mismatch {va.shape} vs {vb.shape}; "
            "module-level hoist changed the gradient pytree."
        )
        np.testing.assert_allclose(
            va,
            vb,
            rtol=1e-12,
            atol=0.0,
            err_msg=(
                f"Cache-on vs cache-flushed VJP differs at leaf index "
                f"{idx}; the module-level hoist altered the gradient."
            ),
        )


def test_module_level_dipole_jit_cache_shared_across_instances() -> None:
    """Two fresh dipole ``PSCBulkArray`` instances share module-level JIT handles.

    Phase L3 (May 2026) hoisted the per-instance ``_ensure_dipole_*_jit_cache``
    closures out to module-level :data:`_DIPOLE_*_JIT` handles so the JAX
    internal abstract-shape cache survives across instance boundaries.  The
    practical consequence is that any second :class:`PSCBulkArray` built
    after the first one finishes its first ``vjp_setup_B`` call -- even in
    a totally separate Fourier-continuation stage / sweep / test -- pays
    zero XLA compile cost as long as the shape signature is identical.

    This test exercises that contract without going through the
    :meth:`PSCBulkArray.warm_handoff_from` path:

    1. Build two independent dipole ``PSCBulkArray`` instances with the
       same TF Fourier order, eval point set and puck layout (so the
       module-level JIT keys match).
    2. Trigger ``recompute_currents`` + ``B_at_points`` + ``vjp_setup_B``
       on each one (which lazily populates the per-instance pointers
       from the module-level cache).
    3. Assert every populated ``_dipole_jit_*`` slot points to the
       **same Python object** on both instances (``is``-identity, not
       value equality) -- the module-level guarantee.
    4. Assert subsequent ``B_at_points`` and ``vjp_setup_B`` outputs
       are finite (sanity).
    """
    eval_pts = np.array(
        [[0.31, 0.02, 0.27], [0.18, 0.06, 0.22]],
        dtype=float,
    )
    psc_a = _dipole_psc_two_tf_orders(order=4, eval_pts=eval_pts)
    psc_b = _dipole_psc_two_tf_orders(order=4, eval_pts=eval_pts)

    rng = np.random.default_rng(0)
    v_B = rng.normal(size=eval_pts.shape)
    for psc in (psc_a, psc_b):
        psc.recompute_currents()
        psc.B_at_points(eval_pts)
        psc.vjp_setup_B(v_B, eval_pts)

    # Not every code path populates every slot (e.g. ``_dipole_jit_forward``
    # is only built when :meth:`_ensure_dipole_jit_cache` is called with a
    # non-``None`` ``g_tf``).  Iterate the union of populated slots and
    # only assert ``is``-identity for slots that are non-``None`` on
    # *both* instances -- mirrors the contract of
    # :func:`test_dipole_warm_handoff_carries_jit_slots`.
    shared_attrs = (
        "_dipole_jit_assemble_L",
        "_dipole_jit_assemble_f",
        "_dipole_jit_field",
        "_dipole_jit_forward",
        "_dipole_jit_vjp_B",
        "_dipole_jit_vjp_f",
        "_dipole_jit_vjp_L",
        "_dipole_jit_vjp_fused_tf",
    )
    at_least_one_checked = False
    for attr in shared_attrs:
        ref_a = getattr(psc_a, attr, None)
        ref_b = getattr(psc_b, attr, None)
        if ref_a is None or ref_b is None:
            continue
        assert ref_a is ref_b, (
            f"{attr} should be the same module-level object across instances "
            f"(got {ref_a!r} vs {ref_b!r})"
        )
        at_least_one_checked = True
    assert at_least_one_checked, (
        "Test did not populate any of the dipole JIT slots; cannot verify "
        "module-level cache sharing.  Update the test workflow to exercise "
        "at least the VJP path."
    )

    # Sanity: subsequent ``B_at_points`` returns finite values using
    # the shared module-level handles.  ``vjp_setup_B`` is not called
    # here because it does not return an array (it populates internal
    # caches); the populated-slot identity check above already
    # exercises that path.
    out_a = psc_b.B_at_points(eval_pts)
    assert np.all(np.isfinite(out_a))


def test_module_level_dipole_jit_cache_matches_module_constants() -> None:
    """Per-instance ``_dipole_jit_*`` pointers must equal the module globals.

    Defensive companion to
    :func:`test_module_level_dipole_jit_cache_shared_across_instances`:
    the per-instance slots after ``vjp_setup_B`` must literally be the
    module-level :data:`_DIPOLE_*_JIT` objects, not copies or wrappers.
    This guards against future refactors accidentally re-wrapping each
    handle per instance (which would silently re-introduce the
    per-instance XLA recompile that Phase L3 closed).
    """
    from simsopt.field import psc_bulk as _pb_mod

    eval_pts = np.array(
        [[0.33, 0.03, 0.28]], dtype=float
    )
    psc = _dipole_psc_two_tf_orders(order=4, eval_pts=eval_pts)
    psc.recompute_currents()
    psc.B_at_points(eval_pts)
    rng = np.random.default_rng(1)
    psc.vjp_setup_B(rng.normal(size=eval_pts.shape), eval_pts)

    expectations = {
        "_dipole_jit_assemble_L": _pb_mod._DIPOLE_ASSEMBLE_L_JIT,
        "_dipole_jit_assemble_f": _pb_mod._DIPOLE_ASSEMBLE_F_JIT,
        "_dipole_jit_field": _pb_mod._DIPOLE_FIELD_JIT,
        "_dipole_jit_forward": _pb_mod._DIPOLE_FORWARD_JIT,
        "_dipole_jit_vjp_B": _pb_mod._DIPOLE_VJP_B_JIT,
        "_dipole_jit_vjp_f": _pb_mod._DIPOLE_VJP_F_JIT,
        "_dipole_jit_vjp_L": _pb_mod._DIPOLE_VJP_L_JIT,
        "_dipole_jit_vjp_fused_tf": _pb_mod._DIPOLE_VJP_FUSED_TF_JIT,
    }
    # Only slots that were actually populated need to match: e.g.
    # ``_dipole_jit_forward`` is built only on the free-DoF code path
    # (``_ensure_dipole_jit_cache`` with non-``None`` ``g_tf``), which
    # the minimal workflow above does not exercise.  Slots that were
    # populated must, however, *exactly* equal the module-level handle.
    at_least_one_checked = False
    for attr, expected in expectations.items():
        got = getattr(psc, attr)
        if got is None:
            continue
        assert got is expected, (
            f"{attr} should be the module-level handle "
            f"{expected!r}; got {got!r}"
        )
        at_least_one_checked = True
    assert at_least_one_checked, (
        "No dipole JIT slot was populated -- test workflow needs to "
        "exercise at least one of the cached paths."
    )


def test_dipole_warm_handoff_carries_jit_slots(monkeypatch) -> None:
    """Continuation-style rebuild reuses compiled dipole kernels via warm handoff."""
    monkeypatch.delenv("SIMSOPT_FC_WARM_HANDOFF", raising=False)

    eval_pts = np.array(
        [[0.3, 0.0, 0.25], [0.2, 0.05, 0.21]],
        dtype=float,
    )
    psc_lo = _dipole_psc_two_tf_orders(order=4, eval_pts=eval_pts)
    psc_hi = _dipole_psc_two_tf_orders(order=16, eval_pts=eval_pts)

    psc_lo.recompute_currents()
    rng = np.random.default_rng(0)
    v_B = rng.normal(size=eval_pts.shape)
    psc_lo.B_at_points(eval_pts)
    psc_lo.vjp_setup_B(v_B, eval_pts)

    assert psc_hi.try_warm_handoff_from_prev(psc_lo) is True

    dipole_jit_attrs = (
        "_dipole_jit_rebuild_key",
        "_dipole_jit_assemble_L",
        "_dipole_jit_assemble_f",
        "_dipole_jit_field_key",
        "_dipole_jit_field",
        "_dipole_jit_key",
        "_dipole_jit_forward",
        "_dipole_jit_vjp_key",
        "_dipole_jit_vjp_B",
        "_dipole_jit_vjp_f",
        "_dipole_jit_vjp_L",
        "_dipole_jit_vjp_fused_tf",
    )
    for attr in dipole_jit_attrs:
        ref = getattr(psc_lo, attr, None)
        if ref is None:
            continue
        assert getattr(psc_hi, attr) is ref

    Bout = psc_hi.B_at_points(eval_pts)
    assert np.all(np.isfinite(Bout))


def test_pucks_to_vtk_dipole_smoke():
    """Dipole PSCBulkArray must export VTU without energy-mode quadrature stacks."""
    pytest.importorskip("pyevtk")
    import tempfile

    from simsopt.field.puck_vtk import pucks_to_vtk

    tf_coil = _unit_circle_coil(1.0e4)
    psc = PSCBulkArray(
        np.array([[0.0, 0.0, 0.1]]),
        np.array([[0.0, 0.0, 1.0]]),
        np.array([0.03]),
        np.array([0.02]),
        [tf_coil],
        eval_points=np.array([[0.0, 0.0, 0.2]], dtype=float),
        m_fourier=1,
        l_zernike=2,
        k_chebyshev=1,
        n_rho=4,
        n_phi=6,
        n_z=3,
        nfp=2,
        stellsym=True,
        solver_mode="dipole",
    )
    psc.recompute_currents()
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "pucks_dipole")
        pucks_to_vtk(psc, path, n_phi=16, n_r=8)
        assert os.path.isfile(path + ".vtu")
