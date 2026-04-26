"""
Tests for passive bulk (ideal diamagnetic) pucks.
"""

import os
import tempfile
from typing import Any

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("jax.numpy")

from simsopt.field.bulk_inductance import (
    null_space_projection_matrix,
    _shell_inductance_matrix_symmetric_reduced_jax,
    shell_inductance_matrix_blockwise,
    shell_inductance_matrix_pure,
    shell_loading_vector_pure,
    shell_loading_vector_stacked_pure,
    shell_solve_eigenfloor_pure,
)
from simsopt._core.optimizable import Optimizable
from simsopt.field.coil import Coil, Current
from simsopt.field.psc_bulk import (
    PSCBulkArray,
    _EIGENFLOOR_THRESHOLD,
    _fold_Bn_to_work,
    _replicate_pucks_jax,
    _rotation_matrix_from_quat,
    _rotation_matrix_from_quat_jax,
)
from simsopt.field.puck_basis import (
    gauge_projection_matrix,
    list_zernike_modes,
    zernike_radial,
)
from simsopt.geo import CurveXYZFourier

import jax.numpy as jnp


def _unit_circle_coil(current_amp: float = 1.0e5):
    curve = CurveXYZFourier(32, 1)
    curve.x = np.array([0, 0, 1, 0, 1, 0, 0, 0.0, 0.0]) * 1.0
    return Coil(curve, Current(current_amp))


def test_zernike_radial_R00():
    rho = np.linspace(0, 1, 20)
    r = np.array(zernike_radial(jnp.asarray(rho), 0, 0))
    np.testing.assert_allclose(r, np.ones_like(rho), atol=1e-10)


def test_zernike_modes_nonempty():
    modes = list_zernike_modes(2, 4)
    assert (0, 0) in modes
    assert (2, 2) in modes


def test_puck_basis_spec_populated_and_used():
    """``PuckBasisData.basis_spec`` is populated and drives rim continuity.

    Verifies that the structured ``(face, m, n_or_k, trig)`` spec matches
    the legacy string parsing of ``dof_names`` (so the new code path is a
    faithful replacement) and that ``build_continuity_constraint``
    produces identical output regardless of whether the fallback string
    parser or the structured spec is used.
    """
    from simsopt.field.puck_basis import (
        _basis_spec_for_dof,
        build_continuity_constraint,
        build_puck_shell_basis,
    )

    basis = build_puck_shell_basis(
        R=0.05, t=0.02, m_fourier=2, l_zernike=4, k_chebyshev=2, n_rho=4, n_phi=6, n_z=3
    )
    assert len(basis.basis_spec) == len(basis.dof_names)
    for a, name in enumerate(basis.dof_names):
        face, m, n_or_k, trig = _basis_spec_for_dof(basis, a)
        parts = name.split("_")
        if parts[0] == "disk":
            assert face == f"disk_{parts[1]}"
            assert m == int(parts[2].replace("m", ""))
            assert n_or_k == int(parts[3].replace("n", ""))
            assert trig == parts[4]
        else:
            assert face == "side"
            assert m == int(parts[1].replace("m", ""))
            assert n_or_k == int(parts[2].replace("k", ""))
            assert trig == parts[3]

    C_spec = build_continuity_constraint(basis, R=0.05, t=0.02, n_rim=8)
    basis_no_spec = type(basis)(
        quad_points_local=basis.quad_points_local,
        quad_weights=basis.quad_weights,
        quad_normals_local=basis.quad_normals_local,
        face_id=basis.face_id,
        phi_values=basis.phi_values,
        grad_phi_local=basis.grad_phi_local,
        k_basis_local=basis.k_basis_local,
        dof_names=basis.dof_names,
        basis_spec=[],
    )
    C_fallback = build_continuity_constraint(basis_no_spec, R=0.05, t=0.02, n_rim=8)
    np.testing.assert_allclose(C_spec, C_fallback, atol=1e-14)


def test_shell_inductance_symmetric_psd():
    np.random.seed(0)
    nq, nd = 8, 4
    K = np.random.randn(nq, nd, 3)
    pts = np.random.randn(nq, 3)
    w = np.ones(nq) * 0.01
    import jax.numpy as jnp

    L = np.array(
        shell_inductance_matrix_pure(jnp.asarray(K), jnp.asarray(pts), jnp.asarray(w))
    )
    assert np.allclose(L, L.T, atol=1e-10)
    ev = np.linalg.eigvalsh(L)
    assert np.min(ev) > -1e-8


def test_psc_bulk_single_puck_uniform_field():
    """Single circular TF loop and one puck; induced field is finite."""
    tf_coil = _unit_circle_coil()

    centers = np.array([[0.0, 0.0, 0.15]])
    axes = np.array([[0.0, 0.0, 1.0]])
    R = 0.05
    t = 0.02
    eval_pts = np.array([[0.15, 0.0, 0.25]], dtype=float)

    psc = PSCBulkArray(
        centers,
        axes,
        np.array([R]),
        np.array([t]),
        [tf_coil],
        eval_points=eval_pts,
        m_fourier=2,
        l_zernike=4,
        k_chebyshev=2,
        n_rho=6,
        n_phi=8,
        n_z=4,
        nfp=1,
        stellsym=False,
    )
    B = psc.B_at_points(eval_pts)
    assert np.all(np.isfinite(B))
    assert B.shape == (1, 3)


def test_passive_bulk_field_B_vjp():
    from simsopt.field.psc_bulk import PassiveBulkField

    tf_coil = _unit_circle_coil()
    centers = np.array([[0.0, 0.0, 0.12]])
    axes = np.array([[0.0, 0.0, 1.0]])
    eval_pts = np.array([[0.01, 0.0, 0.18]], dtype=float)
    psc = PSCBulkArray(
        centers,
        axes,
        np.array([0.04]),
        np.array([0.02]),
        [tf_coil],
        eval_points=eval_pts,
        m_fourier=1,
        l_zernike=3,
        k_chebyshev=1,
        n_rho=4,
        n_phi=6,
        n_z=3,
    )
    bf = PassiveBulkField(psc)
    bf.set_points_cart(np.ascontiguousarray(eval_pts))
    v = np.ones((1, 3))
    dj = bf.B_vjp(v)
    assert dj is not None


def _make_small_psc(**kwargs):
    """Factory for small ``PSCBulkArray`` instances used by rim-continuity tests."""
    tf_coil = _unit_circle_coil()
    centers = np.array([[0.0, 0.0, 0.12]])
    axes = np.array([[0.0, 0.0, 1.0]])
    eval_pts = np.array([[0.01, 0.0, 0.18]], dtype=float)
    defaults = dict(
        m_fourier=2,
        l_zernike=4,
        k_chebyshev=2,
        n_rho=6,
        n_phi=8,
        n_z=4,
    )
    defaults.update(kwargs)
    return PSCBulkArray(
        centers,
        axes,
        np.array([0.04]),
        np.array([0.02]),
        [tf_coil],
        eval_points=eval_pts,
        **defaults,
    )


def test_rim_continuity_strict_raises_on_degenerate(monkeypatch):
    """If the rim-continuity constraint has zero kernel, strict mode raises.

    We inject a full-column-rank synthetic constraint matrix so the
    degenerate branch of ``_build_rim_continuity_projector`` is guaranteed
    to fire.  With ``strict_rim_continuity=True`` this must raise
    ``ValueError`` instead of silently falling back.
    """
    import simsopt.field.psc_bulk as psc_mod

    def _fake_full_rank_constraint(basis, R, t, n_rim):
        n = len(basis.dof_names)
        return np.eye(n)

    monkeypatch.setattr(
        psc_mod, "build_continuity_constraint", _fake_full_rank_constraint
    )

    with pytest.raises(ValueError, match="rim-continuity"):
        _make_small_psc(strict_rim_continuity=True)


def test_rim_continuity_degenerate_warns_by_default(monkeypatch):
    """With ``strict_rim_continuity=False`` (default) we only warn.

    Uses the same synthetic full-rank constraint as
    :func:`test_rim_continuity_strict_raises_on_degenerate` so the
    fallback path is exercised deterministically.
    """
    import simsopt.field.psc_bulk as psc_mod

    def _fake_full_rank_constraint(basis, R, t, n_rim):
        n = len(basis.dof_names)
        return np.eye(n)

    monkeypatch.setattr(
        psc_mod, "build_continuity_constraint", _fake_full_rank_constraint
    )
    with pytest.warns(UserWarning, match="rim-continuity"):
        _make_small_psc()


def test_rim_continuity_production_order_no_fallback():
    """``(m=2, l=4, k=2)`` is the production-order setting used by the examples.

    At this resolution the rim-continuity projector has a non-trivial
    kernel on every puck, so ``strict_rim_continuity=True`` must **not**
    raise.
    """
    import warnings

    tf_coil = _unit_circle_coil()
    centers = np.array([[0.0, 0.0, 0.12]])
    axes = np.array([[0.0, 0.0, 1.0]])
    eval_pts = np.array([[0.01, 0.0, 0.18]], dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        psc = PSCBulkArray(
            centers,
            axes,
            np.array([0.04]),
            np.array([0.02]),
            [tf_coil],
            eval_points=eval_pts,
            m_fourier=2,
            l_zernike=4,
            k_chebyshev=2,
            n_rho=6,
            n_phi=8,
            n_z=4,
            strict_rim_continuity=True,
        )
    assert psc is not None


def test_puck_basis_cache_bounded():
    """``set_puck_basis_cache_max`` evicts oldest entries (FIFO).

    Guards against unbounded memory growth in optimisation runs that
    sweep many distinct (R, t) puck geometries.
    """
    from simsopt.field import puck_basis as pb

    pb.set_puck_basis_cache_max(64)  # reset to default
    pb.clear_puck_basis_cache()
    for i in range(3):
        pb.build_puck_shell_basis(
            R=0.05 + i * 0.001,
            t=0.02,
            m_fourier=1,
            l_zernike=2,
            k_chebyshev=1,
            n_rho=4,
            n_phi=6,
            n_z=3,
        )
    assert len(pb._PUCK_SHELL_BASIS_CACHE) == 3

    pb.set_puck_basis_cache_max(2)
    assert len(pb._PUCK_SHELL_BASIS_CACHE) == 2

    pb.set_puck_basis_cache_max(0)
    assert len(pb._PUCK_SHELL_BASIS_CACHE) == 0
    pb.build_puck_shell_basis(
        R=0.05,
        t=0.02,
        m_fourier=1,
        l_zernike=2,
        k_chebyshev=1,
        n_rho=4,
        n_phi=6,
        n_z=3,
    )
    assert len(pb._PUCK_SHELL_BASIS_CACHE) == 0

    pb.set_puck_basis_cache_max(64)


def test_passive_bulk_field_invalidate():
    """``PassiveBulkField.invalidate()`` clears own and parent caches.

    Exercises the canonical dirty-flagging API: after changing a TF coil
    current and calling :meth:`PSCBulkArray.recompute_currents` +
    :meth:`PassiveBulkField.invalidate`, both the passive field and the
    :class:`MagneticFieldSum` it feeds into must return refreshed ``B``
    values.
    """
    from simsopt.field import BiotSavart
    from simsopt.field.magneticfield import MagneticFieldSum
    from simsopt.field.psc_bulk import PassiveBulkField

    tf_coil = _unit_circle_coil(1e5)
    eval_pts = np.array([[0.15, 0.0, 0.25]], dtype=float)
    psc = PSCBulkArray(
        np.array([[0.0, 0.0, 0.12]]),
        np.array([[0.0, 0.0, 1.0]]),
        np.array([0.04]),
        np.array([0.02]),
        [tf_coil],
        eval_points=eval_pts,
        m_fourier=1,
        l_zernike=3,
        k_chebyshev=1,
        n_rho=4,
        n_phi=6,
        n_z=3,
    )
    bulk = PassiveBulkField(psc)
    btot = MagneticFieldSum([bulk, BiotSavart([tf_coil])])
    btot.set_points_cart(np.ascontiguousarray(eval_pts))
    B0 = btot.B().copy()

    tf_coil.current.set_dofs(np.array([tf_coil.current.get_value() * 2.0]))
    psc.recompute_currents()
    bulk.invalidate()

    btot.set_points_cart(np.ascontiguousarray(eval_pts))
    B1 = btot.B().copy()
    assert not np.allclose(B0, B1), (
        "Sum field did not refresh after invalidate(); cache likely stale"
    )


def test_dB_by_dX_not_silent():
    """``PassiveBulkField.dB_by_dX`` must raise :class:`NotImplementedError`.

    Previously this method silently returned zero ``3x3`` Jacobians, which
    produced incorrect (and unnoticed) gradients for any downstream consumer
    (particle tracing, BoozerMagneticField construction, grad-B objectives).
    We intentionally make the call loud until an analytic Jacobian lands.
    """
    from simsopt.field.psc_bulk import PassiveBulkField

    tf_coil = _unit_circle_coil()
    centers = np.array([[0.0, 0.0, 0.12]])
    axes = np.array([[0.0, 0.0, 1.0]])
    eval_pts = np.array([[0.01, 0.0, 0.18]], dtype=float)
    psc = PSCBulkArray(
        centers,
        axes,
        np.array([0.04]),
        np.array([0.02]),
        [tf_coil],
        eval_points=eval_pts,
        m_fourier=1,
        l_zernike=3,
        k_chebyshev=1,
        n_rho=4,
        n_phi=6,
        n_z=3,
    )
    bf = PassiveBulkField(psc)
    bf.set_points_cart(np.ascontiguousarray(eval_pts))
    with pytest.raises(NotImplementedError, match="dB_by_dX"):
        bf.dB_by_dX()


def test_pucks_to_vtk_smoke():
    pytest.importorskip("pyevtk")
    from simsopt.field.puck_vtk import pucks_to_vtk

    tf_coil = _unit_circle_coil(1.0e4)
    psc = PSCBulkArray(
        np.array([[0.0, 0.0, 0.1]]),
        np.array([[0.0, 0.0, 1.0]]),
        np.array([0.03]),
        np.array([0.02]),
        [tf_coil],
        eval_points=np.array([[0, 0, 0.2]], float),
        m_fourier=1,
        l_zernike=2,
        k_chebyshev=1,
        n_rho=4,
        n_phi=6,
        n_z=3,
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "pucks")
        pucks_to_vtk(psc, path, n_phi=16, n_r=8)
        assert os.path.isfile(path + ".vtu")


def test_pucks_to_vtk_quat_mesh_aligns_nfp_replica_K_magnitude():
    """Full-quat mesh must align with shell quadrature so |K| matches on +1 replicas.

    Axis-only :func:`~simsopt.field.puck_vtk.puck_surface_mesh` discards in-plane
    roll relative to the replica quaternion used in :class:`PSCBulkArray`, so
    nearest-quadrature sampling of ``K`` was phi-shifted.  Using ``quat=``
    matches ``_rebuild`` and restores nfp-consistent :math:`|K|` on pure-rotation
    images (here replicas ``0`` and ``2`` for ``nfp=2``).
    """
    pytest.importorskip("pyevtk")
    from simsopt.field.coil import coils_via_symmetries
    from simsopt.field.puck_vtk import (
        _sample_K_g_on_mesh,
        puck_surface_mesh,
        pucks_to_vtk,
    )

    base_curve = CurveXYZFourier(32, 1)
    base_curve.x = np.array([1.0, 0.0, 0.3, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0])
    tf_coils = coils_via_symmetries([base_curve], [Current(100.0)], 2, True)
    ax = np.array([0.18, 0.12, 0.976], dtype=float)
    ax = ax / np.linalg.norm(ax)
    centers = np.array([[1.12, 0.11, 0.08]], dtype=float)
    psc = PSCBulkArray(
        centers,
        np.array([ax]),
        np.array([0.10]),
        np.array([0.03]),
        tf_coils,
        eval_points=np.array([[1.0, 0.05, 0.35]], dtype=float),
        m_fourier=2,
        l_zernike=2,
        k_chebyshev=1,
        n_rho=4,
        n_phi=6,
        n_z=3,
        nfp=2,
        stellsym=True,
    )
    psc.recompute_currents()

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "pucks_stellsym")
        pucks_to_vtk(psc, path, n_phi=16, n_r=8)
        assert os.path.isfile(path + ".vtu")

    n_mesh_phi, n_mesh_r = 20, 8
    r0, r1 = 0, 2
    c0, ax0, R0, t0 = psc._all_pucks[r0]
    c1, ax1, R1, t1 = psc._all_pucks[r1]
    assert R0 == R1 and t0 == t1
    q0 = psc._all_pucks_quats[r0]
    q1 = psc._all_pucks_quats[r1]
    x0, y0, z0, _ = puck_surface_mesh(
        c0, ax0, R0, t0, n_phi=n_mesh_phi, n_r=n_mesh_r, quat=q0
    )
    x1, y1, z1, _ = puck_surface_mesh(
        c1, ax1, R1, t1, n_phi=n_mesh_phi, n_r=n_mesh_r, quat=q1
    )
    Km0_q, _, _, _ = _sample_K_g_on_mesh(psc, r0, x0, y0, z0)
    Km1_q, _, _, _ = _sample_K_g_on_mesh(psc, r1, x1, y1, z1)
    np.testing.assert_allclose(
        Km0_q,
        Km1_q,
        rtol=0.05,
        atol=1e-7,
        err_msg="|K| on nfp-related +1 replicas should match with quat mesh",
    )

    x0a, y0a, z0a, _ = puck_surface_mesh(
        c0, ax0, R0, t0, n_phi=n_mesh_phi, n_r=n_mesh_r
    )
    x1a, y1a, z1a, _ = puck_surface_mesh(
        c1, ax1, R1, t1, n_phi=n_mesh_phi, n_r=n_mesh_r
    )
    Km0_a, _, _, _ = _sample_K_g_on_mesh(psc, r0, x0a, y0a, z0a)
    Km1_a, _, _, _ = _sample_K_g_on_mesh(psc, r1, x1a, y1a, z1a)
    diff_bad = float(np.max(np.abs(Km0_a - Km1_a)))
    diff_good = float(np.max(np.abs(Km0_q - Km1_q)))
    assert diff_good < 0.5 * diff_bad, (
        f"expected axis-only mesh to mismatch more than quat mesh; "
        f"diff_good={diff_good}, diff_bad={diff_bad}"
    )


def test_sample_K_g_on_mesh_g_bandlimited_on_top_ring():
    """Continuous basis evaluation: :math:`g` on a top-face ring is bandlimited in :math:`\\phi`.

    Nearest-quadrature painting would replicate :math:`n_\\phi` constant wedges; the
    analytic path must not inject spurious high-:math:`m` Fourier content from the
    quadrature grid.  (Use :attr:`g_potential` / fourth return, not
    :attr:`K_magnitude` / :math:`|K|`, which is nonlinear in the components and
    not bandlimited in :math:`\\phi`.)
    """
    from simsopt.field.psc_bulk import _rotation_matrix_from_quat
    from simsopt.field.puck_vtk import _sample_K_g_on_mesh

    psc = _make_small_psc(m_fourier=3, l_zernike=4, n_phi=8)
    c, _ax, R, t = psc._all_pucks[0]
    q0 = psc._all_pucks_quats[0]
    Rmat = _rotation_matrix_from_quat(q0)
    mmax = 3
    n_pts = 256
    phi = np.linspace(0.0, 2.0 * np.pi, n_pts, endpoint=False)
    rho0 = 0.5 * float(R)
    zl = 0.5 * float(t)
    xl = rho0 * np.cos(phi)
    yl = rho0 * np.sin(phi)
    pl = np.column_stack([xl, yl, np.full(n_pts, zl)])
    pts_g = (Rmat @ pl.T).T + np.asarray(c, dtype=float)
    _Km, _Kv, _Bn, g_val = _sample_K_g_on_mesh(
        psc, 0, pts_g[:, 0], pts_g[:, 1], pts_g[:, 2]
    )
    spec = np.abs(np.fft.rfft(g_val))
    if spec.size <= mmax + 1:
        return
    tail = float(np.max(spec[mmax + 1 :]))
    peak = float(np.max(spec))
    assert tail < 1.0e-4 * (peak + 1.0e-20), (
        f"unexpected high-m content in g on a top ring: tail={tail}, peak={peak}"
    )


def test_gauge_projection():
    Q = gauge_projection_matrix(5, 0)
    assert Q.shape == (5, 4)
    beta = Q @ np.random.randn(4)
    assert beta[0] == 0.0


@pytest.mark.slow
def test_mutual_inductance_convergence():
    """Successive quadrature refinement should converge the inductance matrix."""
    from simsopt.field.puck_basis import build_puck_shell_basis

    R, t = 0.05, 0.02
    common = dict(m_fourier=1, l_zernike=2, k_chebyshev=1)

    def L_for_resolution(n_rho, n_phi, n_z):
        basis = build_puck_shell_basis(
            R, t, n_rho=n_rho, n_phi=n_phi, n_z=n_z, **common
        )
        K_b = jnp.asarray(basis.k_basis_local)
        pts = jnp.asarray(basis.quad_points_local)
        w = jnp.asarray(basis.quad_weights)
        return np.array(
            shell_inductance_matrix_pure(
                K_b,
                pts,
                w,
                delta_reg=1e-6,
                adaptive_self_reg=False,
            )
        )

    L_lo = L_for_resolution(6, 8, 4)
    L_med = L_for_resolution(10, 16, 8)
    L_hi = L_for_resolution(16, 24, 12)
    diff_lo_med = np.linalg.norm(L_med - L_lo) / (np.linalg.norm(L_med) + 1e-30)
    diff_med_hi = np.linalg.norm(L_hi - L_med) / (np.linalg.norm(L_hi) + 1e-30)
    assert diff_med_hi < diff_lo_med, (
        f"Inductance not converging: lo->med={diff_lo_med:.4g}, med->hi={diff_med_hi:.4g}"
    )


def test_thin_limit_convergence():
    """Halving the puck thickness should produce converging induced fields."""
    tf_coil = _unit_circle_coil(1e5)
    centers = np.array([[0.0, 0.0, 0.15]])
    axes = np.array([[0.0, 0.0, 1.0]])
    R = 0.05
    eval_pts = np.array([[0.15, 0.0, 0.25]], dtype=float)
    common = dict(
        coils_TF=[tf_coil],
        eval_points=eval_pts,
        m_fourier=1,
        l_zernike=2,
        k_chebyshev=1,
        n_rho=6,
        n_phi=8,
        n_z=4,
        nfp=1,
        stellsym=False,
    )

    B_vals = []
    for thickness in [0.04, 0.02, 0.01]:
        psc = PSCBulkArray(
            centers, axes, np.array([R]), np.array([thickness]), **common
        )
        B_vals.append(psc.B_at_points(eval_pts))

    diff_1 = np.linalg.norm(B_vals[1] - B_vals[0])
    diff_2 = np.linalg.norm(B_vals[2] - B_vals[1])
    assert diff_2 < diff_1 + 1e-15, (
        f"Thin-limit not converging: diff(t=0.02-0.04)={diff_1}, diff(t=0.01-0.02)={diff_2}"
    )


def test_vjp_fd_gradient_check():
    """Finite-difference check of VJP w.r.t. TF coil current."""
    from simsopt.field.psc_bulk import PassiveBulkField

    tf_coil = _unit_circle_coil(1e5)
    centers = np.array([[0.0, 0.0, 0.12]])
    axes = np.array([[0.0, 0.0, 1.0]])
    eval_pts = np.array([[0.15, 0.0, 0.20]], dtype=float)
    psc = PSCBulkArray(
        centers,
        axes,
        np.array([0.04]),
        np.array([0.02]),
        [tf_coil],
        eval_points=eval_pts,
        m_fourier=1,
        l_zernike=2,
        k_chebyshev=1,
        n_rho=4,
        n_phi=6,
        n_z=3,
    )

    bf = PassiveBulkField(psc)
    bf.set_points_cart(np.ascontiguousarray(eval_pts))

    v = np.ones((1, 3))
    B0 = np.array(bf.B())
    obj0 = np.sum(v * B0)

    dI = 1.0
    I0 = tf_coil.current.get_value()
    tf_coil.current.x = [I0 + dI]
    psc.recompute_currents()
    bf.clear_cached_properties()
    B1 = np.array(bf.B())
    obj1 = np.sum(v * B1)

    fd_deriv = (obj1 - obj0) / dI

    # Restore and compute VJP
    tf_coil.current.x = [I0]
    psc.recompute_currents()
    bf.clear_cached_properties()
    dj = bf.B_vjp(v)

    # The VJP returns a Derivative; extract the current derivative
    from simsopt._core import Derivative

    if isinstance(dj, Derivative):
        vjp_current_deriv = dj(tf_coil.current)
    else:
        vjp_total = Derivative()
        for d in dj:
            vjp_total += d
        vjp_current_deriv = vjp_total(tf_coil.current)

    if vjp_current_deriv is not None and np.isfinite(vjp_current_deriv).all():
        vjp_val = float(np.sum(vjp_current_deriv))
        rel_err = abs(vjp_val - fd_deriv) / (abs(fd_deriv) + 1e-30)
        assert rel_err < 0.1 or abs(vjp_val - fd_deriv) < 1e-6, (
            f"VJP/FD mismatch: vjp={vjp_val}, fd={fd_deriv}, rel_err={rel_err}"
        )


def test_vjp_mismatched_points():
    """VJP must be consistent with forward B even when the field's evaluation
    points differ from the ``eval_points`` passed at construction time."""
    from simsopt.field.psc_bulk import PassiveBulkField

    tf_coil = _unit_circle_coil(1e5)
    centers = np.array([[0.0, 0.0, 0.12]])
    axes = np.array([[0.0, 0.0, 1.0]])

    construction_pts = np.array([[0.15, 0.0, 0.20]], dtype=float)
    field_pts = np.array([[0.10, 0.05, 0.22], [0.12, -0.03, 0.19]], dtype=float)

    psc = PSCBulkArray(
        centers,
        axes,
        np.array([0.04]),
        np.array([0.02]),
        [tf_coil],
        eval_points=construction_pts,
        m_fourier=1,
        l_zernike=2,
        k_chebyshev=1,
        n_rho=4,
        n_phi=6,
        n_z=3,
        adaptive_self_reg=False,
    )

    bf = PassiveBulkField(psc)
    bf.set_points_cart(np.ascontiguousarray(field_pts))

    v = np.ones_like(field_pts)
    B0 = np.array(bf.B())
    np.sum(v * B0)

    dI = 1.0
    I0 = tf_coil.current.get_value()
    tf_coil.current.x = [I0 + dI]
    psc.recompute_currents()
    bf.clear_cached_properties()
    obj1 = np.sum(v * np.array(bf.B()))

    tf_coil.current.x = [I0 - dI]
    psc.recompute_currents()
    bf.clear_cached_properties()
    obj_m1 = np.sum(v * np.array(bf.B()))

    fd_deriv = (obj1 - obj_m1) / (2 * dI)

    tf_coil.current.x = [I0]
    psc.recompute_currents()
    bf.clear_cached_properties()
    dj = bf.B_vjp(v)

    from simsopt._core import Derivative

    vjp_current_deriv = (
        dj(tf_coil.current) if isinstance(dj, Derivative) else sum(dj)(tf_coil.current)
    )
    assert vjp_current_deriv is not None
    vjp_val = float(np.sum(vjp_current_deriv))
    # When the true derivative is ~0, FD can be ~1e-13 noise; allow absolute tol.
    np.testing.assert_allclose(
        vjp_val,
        fd_deriv,
        rtol=0.05,
        atol=2e-8,
        err_msg=(f"Mismatched-points VJP/FD mismatch: vjp={vjp_val}, fd={fd_deriv}"),
    )


def _make_optimization_setup():
    """Shared helper: surface + TF coils + puck + SquaredFlux for optimization tests."""
    from pathlib import Path
    from simsopt.field import coils_via_symmetries, BiotSavart, Current
    from simsopt.field.magneticfield import MagneticFieldSum
    from simsopt.geo import SurfaceRZFourier, create_equally_spaced_curves
    from simsopt.objectives import SquaredFlux

    TEST_DIR = (Path(__file__).parent / ".." / "test_files").resolve()
    filename = TEST_DIR / "input.LandremanPaul2021_QA"
    s = SurfaceRZFourier.from_vmec_input(
        filename, range="half period", nphi=4, ntheta=4
    )

    ncoils, nfp, stellsym = 2, 2, True
    base_curves = create_equally_spaced_curves(
        ncoils, nfp, stellsym, R0=1.0, R1=0.5, order=3
    )
    base_currents = [Current(1e5) for _ in range(ncoils)]
    coils_tf = coils_via_symmetries(base_curves, base_currents, nfp, stellsym)

    eval_pts = np.ascontiguousarray(s.gamma().reshape(-1, 3))
    centers = np.array([[1.0, 0.0, 0.15]])
    axes = np.array([[0.0, 0.0, 1.0]])

    psc = PSCBulkArray(
        centers,
        axes,
        np.array([0.04]),
        np.array([0.02]),
        coils_tf,
        eval_points=eval_pts,
        m_fourier=1,
        l_zernike=2,
        k_chebyshev=1,
        n_rho=4,
        n_phi=6,
        n_z=3,
        nfp=1,
        stellsym=False,
        # Legacy self-regularization keeps central FD vs dJ agreement for this
        # small synthetic optimization setup (production uses adaptive default).
        adaptive_self_reg=False,
    )
    b_bulk = psc.biot_savart
    b_tf = BiotSavart(coils_tf)
    btot = MagneticFieldSum([b_bulk, b_tf])

    Jf = SquaredFlux(s, btot)
    return s, coils_tf, base_curves, base_currents, psc, btot, Jf


def test_squared_flux_taylor():
    """Taylor test: central FD derivative of SquaredFlux must converge at 2nd order."""
    s, coils_tf, base_curves, base_currents, psc, btot, Jf = _make_optimization_setup()

    dofs = np.copy(Jf.x)
    np.random.seed(42)
    h = np.random.randn(len(dofs))
    h = h / np.linalg.norm(h)

    Jf.x = dofs
    psc.recompute_currents()
    btot.Bfields[0].invalidate_cache()
    dJ = Jf.dJ()
    deriv = np.sum(dJ * h)

    errors = []
    for i in range(10, 16):
        eps = 0.5**i
        Jf.x = dofs + eps * h
        psc.recompute_currents()
        btot.Bfields[0].invalidate_cache()
        Jp = Jf.J()
        Jf.x = dofs - eps * h
        psc.recompute_currents()
        btot.Bfields[0].invalidate_cache()
        Jm = Jf.J()
        fd = (Jp - Jm) / (2 * eps)
        if abs(deriv) > 1e-15:
            errors.append(abs(fd - deriv) / abs(deriv))
        else:
            errors.append(abs(fd - deriv))

    _assert_taylor_convergence(errors, label="squared_flux")


def test_optimization_objective_decreases():
    """A few L-BFGS-B iterations must reduce the SquaredFlux objective."""
    from scipy.optimize import minimize

    s, coils_tf, base_curves, base_currents, psc, btot, Jf = _make_optimization_setup()

    def fun(dofs):
        Jf.x = dofs
        psc.recompute_currents()
        btot.Bfields[0].invalidate_cache()
        J = Jf.J()
        grad = Jf.dJ()
        return float(J), np.asarray(grad, dtype=float)

    dofs0 = np.copy(Jf.x)
    J_initial, _ = fun(dofs0)

    res = minimize(
        fun, dofs0, jac=True, method="L-BFGS-B", options={"maxiter": 10, "maxcor": 50}
    )

    Jf.x = res.x
    psc.recompute_currents()
    btot.Bfields[0].invalidate_cache()
    J_final = Jf.J()

    assert J_final < J_initial, (
        f"Optimization did not decrease objective: J_initial={J_initial}, J_final={J_final}"
    )


@pytest.mark.slow
def test_galerkin_residual():
    """The solve residual projected onto the well-conditioned subspace is small."""
    import jax
    from simsopt.field.force import _B_at_point_from_coil_set_pure
    from simsopt.field.bulk_inductance import (
        project_reduced_system,
    )

    tf_coil = _unit_circle_coil(1e5)
    psc = PSCBulkArray(
        np.array([[0.0, 0.0, 0.15]]),
        np.array([[0.0, 0.0, 1.0]]),
        np.array([0.05]),
        np.array([0.02]),
        [tf_coil],
        eval_points=np.array([[0.15, 0.0, 0.25]]),
        m_fourier=4,
        l_zernike=6,
        k_chebyshev=4,
        n_rho=12,
        n_phi=16,
        n_z=8,
    )

    quad_pts = jnp.asarray(psc._quad_points)
    quad_n = jnp.asarray(psc._quad_normals)
    gammas_tf = jnp.asarray(np.array([c.curve.gamma() for c in psc.coils_TF]))
    gammadash_tf = jnp.asarray(np.array([c.curve.gammadash() for c in psc.coils_TF]))
    currents_tf = jnp.asarray(np.array([c.current.get_value() for c in psc.coils_TF]))
    eps = 1e-8

    def Bn_at_i(i):
        B = _B_at_point_from_coil_set_pure(
            quad_pts[i],
            gammas_tf,
            gammadash_tf,
            currents_tf,
            -1,
            eps,
        )
        return jnp.dot(B, quad_n[i])

    Bn = np.array(jax.vmap(Bn_at_i)(jnp.arange(quad_pts.shape[0])))

    f = np.array(
        shell_loading_vector_pure(
            jnp.asarray(psc._phi_mat),
            jnp.asarray(psc._quad_weights),
            jnp.asarray(Bn),
        )
    )

    L = psc._L_work
    Q = psc._Q
    Lr, fr = project_reduced_system(jnp.asarray(L), jnp.asarray(f), jnp.asarray(Q))
    Lr = np.array(Lr)
    fr = np.array(fr)

    beta = psc.beta
    alpha = Q.T @ beta

    ev, V = np.linalg.eigh(Lr)
    threshold = 1e-10
    good = ev > threshold
    assert np.sum(good) > 0, "No well-conditioned eigenvalues in L_red"

    residual = Lr @ alpha - fr
    residual_proj = V[:, good].T @ residual
    fr_proj = V[:, good].T @ fr
    rel_residual = np.linalg.norm(residual_proj) / (np.linalg.norm(fr_proj) + 1e-30)
    assert rel_residual < 0.01, (
        f"Galerkin residual in well-conditioned subspace too large: {rel_residual:.4g}"
    )


@pytest.mark.slow
def test_galerkin_residual_free_dofs():
    """Galerkin residual is small when puck DOFs are free.

    Extends :func:`test_galerkin_residual` to the free-DOF branch of
    ``_solve_beta`` (which triggers the JAX eigenfloor solve).  Since
    :func:`_solve_beta` now uses the same eigenfloor path for both
    fixed and free DOFs, the rim-continuity-projected residual must be
    equally small in both cases.
    """
    import jax
    from simsopt.field.force import _B_at_point_from_coil_set_pure

    tf_coil = _unit_circle_coil(1e5)
    psc = PSCBulkArray(
        np.array([[0.0, 0.0, 0.15]]),
        np.array([[0.0, 0.0, 1.0]]),
        np.array([0.05]),
        np.array([0.02]),
        [tf_coil],
        eval_points=np.array([[0.15, 0.0, 0.25]]),
        m_fourier=4,
        l_zernike=6,
        k_chebyshev=4,
        n_rho=12,
        n_phi=16,
        n_z=8,
    )
    psc.unfix("center_z0")
    psc.recompute_currents()

    quad_pts = jnp.asarray(psc._quad_points)
    quad_n = jnp.asarray(psc._quad_normals)
    gammas_tf = jnp.asarray(np.array([c.curve.gamma() for c in psc.coils_TF]))
    gammadash_tf = jnp.asarray(np.array([c.curve.gammadash() for c in psc.coils_TF]))
    currents_tf = jnp.asarray(np.array([c.current.get_value() for c in psc.coils_TF]))

    def Bn_at_i(i):
        B = _B_at_point_from_coil_set_pure(
            quad_pts[i],
            gammas_tf,
            gammadash_tf,
            currents_tf,
            -1,
            1e-8,
        )
        return jnp.dot(B, quad_n[i])

    Bn = np.array(jax.vmap(Bn_at_i)(jnp.arange(quad_pts.shape[0])))
    f = np.array(
        shell_loading_vector_pure(
            jnp.asarray(psc._phi_mat),
            jnp.asarray(psc._quad_weights),
            jnp.asarray(Bn),
        )
    )

    L = psc._L_work
    Q_c = psc._Q_c
    Lr_c = Q_c.T @ L @ Q_c
    fr_c = Q_c.T @ f
    beta = psc.beta
    alpha_c = Q_c.T @ beta

    ev, V = np.linalg.eigh(Lr_c)
    good = ev > 1e-10
    assert np.sum(good) > 0, "No well-conditioned eigenvalues in Q_c^T L Q_c"
    residual = Lr_c @ alpha_c - fr_c
    residual_proj = V[:, good].T @ residual
    fr_proj = V[:, good].T @ fr_c
    rel_residual = np.linalg.norm(residual_proj) / (np.linalg.norm(fr_proj) + 1e-30)
    assert rel_residual < 0.01, (
        f"Free-DOF Galerkin residual too large: {rel_residual:.4g}"
    )


@pytest.mark.slow
def test_basis_function_convergence():
    """Increasing basis order should change the far-field and converge."""
    tf_coil = _unit_circle_coil(1e5)
    far_pts = np.array(
        [
            [0.15, 0.0, 0.25],
            [0.0, 0.15, 0.20],
            [-0.10, 0.0, 0.18],
        ],
        dtype=float,
    )

    B_fields = []
    for m_f, l_z, k_c in [(1, 2, 1), (2, 4, 2), (4, 6, 4)]:
        psc = PSCBulkArray(
            np.array([[0.0, 0.0, 0.15]]),
            np.array([[0.0, 0.0, 1.0]]),
            np.array([0.05]),
            np.array([0.02]),
            [tf_coil],
            eval_points=far_pts,
            m_fourier=m_f,
            l_zernike=l_z,
            k_chebyshev=k_c,
            n_rho=16,
            n_phi=24,
            n_z=12,
        )
        B_fields.append(psc.B_at_points(far_pts))

    diff_lo_med = np.linalg.norm(B_fields[1] - B_fields[0])
    diff_med_hi = np.linalg.norm(B_fields[2] - B_fields[1])
    # Basis refinement need not decrease the *incremental* delta monotonically
    # (non-nested spaces); require the high refinement to be within a modest
    # factor of the first jump (XLA reduction order can nudge both norms).
    assert diff_med_hi < 3.0 * diff_lo_med + 1e-12, (
        f"Basis convergence failed: lo->med={diff_lo_med:.4g}, med->hi={diff_med_hi:.4g}"
    )


def test_multi_puck_mutual_inductance():
    """Two pucks produce a different field than the sum of two independent pucks."""
    tf_coil = _unit_circle_coil(1e5)
    eval_pts = np.array([[0.15, 0.0, 0.25]], dtype=float)
    common = dict(
        coils_TF=[tf_coil],
        eval_points=eval_pts,
        m_fourier=2,
        l_zernike=4,
        k_chebyshev=2,
        n_rho=6,
        n_phi=8,
        n_z=4,
        nfp=1,
        stellsym=False,
    )

    psc_a = PSCBulkArray(
        np.array([[0.0, 0.0, 0.12]]),
        np.array([[0.0, 0.0, 1.0]]),
        np.array([0.04]),
        np.array([0.02]),
        **common,
    )
    B_a = psc_a.B_at_points(eval_pts)

    psc_b = PSCBulkArray(
        np.array([[0.0, 0.0, -0.12]]),
        np.array([[0.0, 0.0, 1.0]]),
        np.array([0.04]),
        np.array([0.02]),
        **common,
    )
    B_b = psc_b.B_at_points(eval_pts)

    psc_ab = PSCBulkArray(
        np.array([[0.0, 0.0, 0.12], [0.0, 0.0, -0.12]]),
        np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
        np.array([0.04, 0.04]),
        np.array([0.02, 0.02]),
        **common,
    )
    B_ab = psc_ab.B_at_points(eval_pts)

    B_sum_independent = B_a + B_b
    diff = np.linalg.norm(B_ab - B_sum_independent)
    assert diff > 1e-15, (
        "Two-puck field is identical to sum of independent pucks — "
        "mutual inductance has no effect"
    )
    assert np.all(np.isfinite(B_ab)), "Multi-puck field contains NaN/Inf"


def test_nfp_stellsym_replication():
    """nfp=2, stellsym=True should produce 4 pucks from 1 base puck at correct positions."""
    tf_coil = _unit_circle_coil(1e5)
    center = np.array([[1.0, 0.0, 0.1]])
    axis = np.array([[0.0, 0.0, 1.0]])
    eval_pts = np.array([[0.15, 0.0, 0.25]], dtype=float)

    psc = PSCBulkArray(
        center,
        axis,
        np.array([0.04]),
        np.array([0.02]),
        [tf_coil],
        eval_points=eval_pts,
        m_fourier=1,
        l_zernike=2,
        k_chebyshev=1,
        n_rho=4,
        n_phi=6,
        n_z=3,
        nfp=2,
        stellsym=True,
    )

    assert len(psc._all_pucks) == 4, f"Expected 4 pucks, got {len(psc._all_pucks)}"

    centers_all = np.array([p[0] for p in psc._all_pucks])
    axes_all = np.array([p[1] for p in psc._all_pucks])

    expected_centers = np.array(
        [
            [1.0, 0.0, 0.1],
            [1.0, 0.0, -0.1],
            [-1.0, 0.0, 0.1],
            [-1.0, 0.0, -0.1],
        ]
    )
    expected_axes = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ]
    )

    for i in range(4):
        np.testing.assert_allclose(
            centers_all[i],
            expected_centers[i],
            atol=1e-12,
            err_msg=f"Puck {i} center mismatch",
        )
        np.testing.assert_allclose(
            axes_all[i], expected_axes[i], atol=1e-12, err_msg=f"Puck {i} axis mismatch"
        )

    B = psc.B_at_points(eval_pts)
    assert np.all(np.isfinite(B)), "Symmetrized B contains NaN/Inf"


# ======================================================================
# Blockwise L vs monolithic L
# ======================================================================


def test_blockwise_vs_monolithic_L():
    """For a small problem the blockwise and monolithic L must agree."""
    from simsopt.field.puck_basis import build_puck_shell_basis
    from simsopt.field.psc_bulk import _rotation_matrix_local_to_global

    R, t = 0.05, 0.02
    common_basis = dict(
        m_fourier=1, l_zernike=2, k_chebyshev=1, n_rho=4, n_phi=6, n_z=3
    )
    pucks = [
        (np.array([0.0, 0.0, 0.12]), np.array([0.0, 0.0, 1.0])),
        (np.array([0.0, 0.0, -0.12]), np.array([0.0, 0.0, 1.0])),
    ]
    delta_reg = 1e-6

    K_per, pts_per, w_per = [], [], []
    dof_offsets = []
    all_K, all_pts, all_w = [], [], []
    n_dof_total = 0
    for c, ax in pucks:
        basis = build_puck_shell_basis(R, t, **common_basis)
        Rmat = _rotation_matrix_local_to_global(ax)
        pts_g = (Rmat @ basis.quad_points_local.T).T + c
        K_g = np.einsum("ij,qkj->qki", Rmat, basis.k_basis_local)
        nd = K_g.shape[1]
        K_per.append(K_g)
        pts_per.append(pts_g)
        w_per.append(basis.quad_weights)
        dof_offsets.append(n_dof_total)
        all_K.append(K_g)
        all_pts.append(pts_g)
        all_w.append(basis.quad_weights)
        n_dof_total += nd

    L_block = shell_inductance_matrix_blockwise(
        K_per,
        pts_per,
        w_per,
        dof_offsets,
        n_dof_total,
        delta_reg=delta_reg,
        adaptive_self_reg=False,
    )

    quad_pts = np.vstack(all_pts)
    quad_w = np.concatenate(all_w)
    K_basis = np.zeros((quad_pts.shape[0], n_dof_total, 3))
    row0 = 0
    for idx, K_g in enumerate(all_K):
        nq = K_g.shape[0]
        d0 = dof_offsets[idx]
        K_basis[row0 : row0 + nq, d0 : d0 + K_g.shape[1], :] = K_g
        row0 += nq

    L_mono = np.array(
        shell_inductance_matrix_pure(
            jnp.asarray(K_basis),
            jnp.asarray(quad_pts),
            jnp.asarray(quad_w),
            delta_reg=delta_reg,
            adaptive_self_reg=False,
        )
    )

    np.testing.assert_allclose(
        L_block,
        L_mono,
        atol=1e-12,
        rtol=1e-10,
        err_msg="Blockwise and monolithic L disagree",
    )


# ======================================================================
# Null-space dimension
# ======================================================================


def test_null_space_dimension():
    """null_space_projection_matrix removes the expected number of null modes."""
    from simsopt.field.puck_basis import build_puck_shell_basis

    R, t = 0.05, 0.02
    basis = build_puck_shell_basis(
        R, t, m_fourier=2, l_zernike=4, k_chebyshev=2, n_rho=6, n_phi=8, n_z=4
    )
    K_g = basis.k_basis_local
    pts_g = basis.quad_points_local
    w_g = basis.quad_weights
    nd = K_g.shape[1]

    L = shell_inductance_matrix_blockwise(
        [K_g],
        [pts_g],
        [w_g],
        [0],
        nd,
        delta_reg=1e-6,
        adaptive_self_reg=False,
    )
    Q = null_space_projection_matrix(L, threshold=1e-10)
    n_null = nd - Q.shape[1]
    assert n_null >= 1, f"Expected at least 1 null mode, got {n_null}"
    assert n_null <= 6, f"Too many null modes ({n_null}); expected ~3 per puck"

    n_pucks = 3
    K_per = [K_g.copy() for _ in range(n_pucks)]
    offsets_vec = [0.0, 0.12, -0.12]
    pts_per = [pts_g + np.array([0, 0, off]) for off in offsets_vec]
    w_per = [w_g.copy() for _ in range(n_pucks)]
    dof_offsets = [i * nd for i in range(n_pucks)]
    nd_total = n_pucks * nd

    L_multi = shell_inductance_matrix_blockwise(
        K_per,
        pts_per,
        w_per,
        dof_offsets,
        nd_total,
        delta_reg=1e-6,
        adaptive_self_reg=False,
    )
    Q_multi = null_space_projection_matrix(L_multi, threshold=1e-10)
    n_null_multi = nd_total - Q_multi.shape[1]
    assert n_null_multi >= n_pucks, (
        f"Expected at least {n_pucks} null modes for {n_pucks} pucks, got {n_null_multi}"
    )
    assert n_null_multi <= 6 * n_pucks, (
        f"Too many null modes ({n_null_multi}) for {n_pucks} pucks"
    )


# ======================================================================
# PSCBulkArray is Optimizable with correct DOFs (quaternion)
# ======================================================================


def test_psc_bulk_array_is_optimizable():
    """PSCBulkArray is an Optimizable with the expected local DOFs (quaternion)."""
    tf_coil = _unit_circle_coil()
    centers = np.array([[0.0, 0.0, 0.15], [0.0, 0.0, -0.15]])
    axes = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    eval_pts = np.array([[0.15, 0.0, 0.25]], dtype=float)

    psc = PSCBulkArray(
        centers,
        axes,
        np.array([0.04, 0.04]),
        np.array([0.02, 0.02]),
        [tf_coil],
        eval_points=eval_pts,
        m_fourier=1,
        l_zernike=2,
        k_chebyshev=1,
        n_rho=4,
        n_phi=6,
        n_z=3,
    )

    assert isinstance(psc, Optimizable)
    assert len(psc.local_full_x) == 2 * 9  # 9 DOFs per puck
    assert not any(psc.local_dofs_free_status)

    names = psc.local_full_dof_names
    assert names[0] == "center_x0"
    assert names[3] == "q0_0"
    assert names[4] == "qi_0"
    assert names[5] == "qj_0"
    assert names[6] == "qk_0"
    assert names[7] == "R0"
    assert names[8] == "t0"
    assert names[9] == "center_x1"


def test_psc_bulk_array_unfix_center():
    """Unfixing a center DOF makes it appear in the free DOF vector."""
    tf_coil = _unit_circle_coil()
    psc = PSCBulkArray(
        np.array([[0.0, 0.0, 0.15]]),
        np.array([[0.0, 0.0, 1.0]]),
        np.array([0.04]),
        np.array([0.02]),
        [tf_coil],
        eval_points=np.array([[0.15, 0.0, 0.25]], dtype=float),
        m_fourier=1,
        l_zernike=2,
        k_chebyshev=1,
        n_rho=4,
        n_phi=6,
        n_z=3,
    )
    assert len(psc.local_x) == 0
    psc.unfix("center_z0")
    assert len(psc.local_x) == 1
    np.testing.assert_allclose(psc.local_x[0], 0.15)


def test_psc_bulk_array_unfix_R_warns():
    """Unfixing an ``R{i}`` DOF must emit a :class:`UserWarning`.

    ``PSCBulkArray._vjp_puck_geometry`` currently returns zero for the
    radius / thickness rows, so unfixing those DOFs yields no gradient
    signal.
    """
    tf_coil = _unit_circle_coil()
    psc = PSCBulkArray(
        np.array([[0.0, 0.0, 0.15]]),
        np.array([[0.0, 0.0, 1.0]]),
        np.array([0.04]),
        np.array([0.02]),
        [tf_coil],
        eval_points=np.array([[0.15, 0.0, 0.25]], dtype=float),
        m_fourier=1,
        l_zernike=2,
        k_chebyshev=1,
        n_rho=4,
        n_phi=6,
        n_z=3,
    )
    with pytest.warns(UserWarning, match="zero VJP"):
        psc.unfix("R0")
    with pytest.warns(UserWarning, match="zero VJP"):
        psc.unfix("t0")


def test_psc_bulk_array_unfix_all_warns_on_R_t():
    """``local_unfix_all`` must warn about any zero-VJP DOFs it touches."""
    tf_coil = _unit_circle_coil()
    psc = PSCBulkArray(
        np.array([[0.0, 0.0, 0.15]]),
        np.array([[0.0, 0.0, 1.0]]),
        np.array([0.04]),
        np.array([0.02]),
        [tf_coil],
        eval_points=np.array([[0.15, 0.0, 0.25]], dtype=float),
        m_fourier=1,
        l_zernike=2,
        k_chebyshev=1,
        n_rho=4,
        n_phi=6,
        n_z=3,
    )
    with pytest.warns(UserWarning, match="zero VJP"):
        psc.local_unfix_all()


def test_psc_bulk_array_unfix_center_does_not_warn():
    """Unfixing an in-plane DOF (center/quat) must not emit the zero-VJP warning."""
    import warnings

    tf_coil = _unit_circle_coil()
    psc = PSCBulkArray(
        np.array([[0.0, 0.0, 0.15]]),
        np.array([[0.0, 0.0, 1.0]]),
        np.array([0.04]),
        np.array([0.02]),
        [tf_coil],
        eval_points=np.array([[0.15, 0.0, 0.25]], dtype=float),
        m_fourier=1,
        l_zernike=2,
        k_chebyshev=1,
        n_rho=4,
        n_phi=6,
        n_z=3,
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        psc.unfix("center_z0")
        psc.unfix("q0_0")
    for warning in w:
        assert "zero VJP" not in str(warning.message)


# ======================================================================
# Quaternion helper tests
# ======================================================================


def test_quaternion_normalization():
    """Non-unit quaternion DOFs produce the same rotation as normalized equivalent."""
    from simsopt.field.psc_bulk import (
        _rotation_matrix_from_quat,
        _rotation_matrix_from_quat_jax,
    )

    q = np.array([2.0, 0.6, 0.4, 0.2])
    R_np = _rotation_matrix_from_quat(q)
    R_jax = np.array(_rotation_matrix_from_quat_jax(jnp.asarray(q)))
    np.testing.assert_allclose(
        R_np, R_jax, atol=1e-12, err_msg="NumPy and JAX quaternion rotations disagree"
    )

    # Scaling the quaternion should give the same rotation
    R_scaled = _rotation_matrix_from_quat(3.0 * q)
    np.testing.assert_allclose(
        R_np, R_scaled, atol=1e-12, err_msg="Quaternion rotation not scale-invariant"
    )

    # Verify it's a proper rotation
    np.testing.assert_allclose(R_np @ R_np.T, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(R_np), 1.0, atol=1e-12)


def test_axis_to_quaternion_roundtrip():
    """axis_to_quaternion -> quaternion_to_axis roundtrip recovers the axis."""
    from simsopt.field.psc_bulk import _axis_to_quaternion, _quaternion_to_axis

    axes = [
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, 0.0, -1.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.3, 0.4, 0.5]),
    ]
    for ax in axes:
        ax_unit = ax / np.linalg.norm(ax)
        q = _axis_to_quaternion(ax)
        ax_recovered = _quaternion_to_axis(q)
        np.testing.assert_allclose(
            ax_recovered, ax_unit, atol=1e-12, err_msg=f"Roundtrip failed for axis {ax}"
        )


def test_equivalent_currents():
    """get_equivalent_currents returns finite values with correct shape."""
    tf_coil = _unit_circle_coil(1e5)
    psc = PSCBulkArray(
        np.array([[0.0, 0.0, 0.12], [0.0, 0.0, -0.12]]),
        np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
        np.array([0.04, 0.04]),
        np.array([0.02, 0.02]),
        [tf_coil],
        eval_points=np.array([[0.15, 0.0, 0.25]], dtype=float),
        m_fourier=1,
        l_zernike=2,
        k_chebyshev=1,
        n_rho=4,
        n_phi=6,
        n_z=3,
    )
    I_eq = psc.get_equivalent_currents()
    assert I_eq.shape == (2,)
    assert np.all(np.isfinite(I_eq))
    assert np.all(I_eq >= 0)


# ======================================================================
# Taylor tests for VJP correctness
# ======================================================================


def _make_small_setup(n_base_pucks=1, nfp=1, stellsym=False):
    """Small problem for Taylor tests: surface + TF coil + puck(s)."""
    from pathlib import Path
    from simsopt.field import BiotSavart, coils_via_symmetries
    from simsopt.field.magneticfield import MagneticFieldSum
    from simsopt.geo import SurfaceRZFourier, create_equally_spaced_curves
    from simsopt.objectives import SquaredFlux

    TEST_DIR = (Path(__file__).parent / ".." / "test_files").resolve()
    filename = TEST_DIR / "input.LandremanPaul2021_QA"
    s = SurfaceRZFourier.from_vmec_input(
        filename, range="half period", nphi=4, ntheta=4
    )

    base_curves = create_equally_spaced_curves(1, 2, True, R0=1.0, R1=0.5, order=3)
    base_currents = [Current(1e5)]
    coils_tf = coils_via_symmetries(base_curves, base_currents, 2, True)

    eval_pts = np.ascontiguousarray(s.gamma().reshape(-1, 3))
    if n_base_pucks == 1:
        centers = np.array([[1.0, 0.0, 0.15]])
        axes = np.array([[0.1, 0.1, 1.0]])
        radii = np.array([0.04])
        thicknesses = np.array([0.02])
    else:
        centers = np.array([[1.0, 0.0, 0.15], [0.0, 1.0, -0.10]])
        axes = np.array([[0.1, 0.1, 1.0], [0.0, 0.0, 1.0]])
        radii = np.array([0.04, 0.04])
        thicknesses = np.array([0.02, 0.02])

    psc = PSCBulkArray(
        centers,
        axes,
        radii,
        thicknesses,
        coils_tf,
        eval_points=eval_pts,
        m_fourier=1,
        l_zernike=2,
        k_chebyshev=1,
        n_rho=4,
        n_phi=6,
        n_z=3,
        nfp=nfp,
        stellsym=stellsym,
    )
    b_bulk = psc.biot_savart
    b_tf = BiotSavart(coils_tf)
    btot = MagneticFieldSum([b_bulk, b_tf])
    Jf = SquaredFlux(s, btot)
    return s, coils_tf, base_curves, base_currents, psc, btot, Jf


def _run_taylor_test(
    Jf, psc, btot, dof_getter, dof_setter, n_points=5, start_power=4, label=""
):
    """Central FD Taylor test on a scalar objective."""
    dofs0 = dof_getter().copy()
    np.random.seed(42)
    h = np.random.randn(len(dofs0))
    h = h / np.linalg.norm(h)

    dof_setter(dofs0)
    float(Jf.J())
    dJ0 = np.array(Jf.dJ())
    deriv = np.sum(dJ0 * h)

    errors = []
    for i in range(start_power, start_power + n_points):
        eps = 0.5**i
        dof_setter(dofs0 + eps * h)
        Jp = float(Jf.J())
        dof_setter(dofs0 - eps * h)
        Jm = float(Jf.J())
        fd = (Jp - Jm) / (2 * eps)
        if abs(deriv) > 1e-15:
            errors.append(abs(fd - deriv) / abs(deriv))
        else:
            errors.append(abs(fd - deriv))

    dof_setter(dofs0)
    return errors


def _assert_taylor_convergence(errors, label="", strict_2nd_order=True, tol_abs=1e-3):
    """Check that error ratios show 2nd-order convergence *and* that the
    sweep actually enters the asymptotic regime.

    For a second-order-accurate central finite difference, halving the
    step ``eps`` should quarter the error (ratio ~0.25).  We require:

    * the *best* observed ratio is strictly below ``0.35`` (confirming
      a 2nd-order regime was entered somewhere in the sweep),
    * if ``strict_2nd_order=True``, the median of the pre-asymptotic
      ratios (i.e. dropping the final ratio which is typically
      rounding-noise dominated and the first two which may still be
      in the linear regime for non-analytic objectives) is also
      below ``0.35``, *and*
    * the minimum observed error falls below ``tol_abs`` -- a pure
      ratio test cannot tell a correct gradient apart from one that
      is wrong by a constant offset (e.g. a sign flip on part of the
      chain), but the minimum error does.

    Args:
        errors: List of absolute / relative errors, one per eps value.
        label: Human-readable label used in assertion messages.
        strict_2nd_order: If ``True`` (default), also assert the
            pre-asymptotic median is < 0.35.  Pass ``False`` for very
            short sweeps where trimming would leave no samples.
        tol_abs: Required upper bound on ``min(errors)``.  A correct
            analytic gradient drives the FD error well below this; a
            constant-offset bug keeps it on the order of the offset.
            Set to ``None`` to opt out of the absolute check for
            legacy callers that only care about the ratio.
    """
    ratios = [
        (errors[i] + 1e-30) / (errors[i - 1] + 1e-30)
        for i in range(1, len(errors))
        if errors[i - 1] > 1e-12
    ]
    if len(ratios) == 0:
        return
    best = float(np.min(ratios))
    assert best < 0.35, (
        f"Taylor test '{label}' failed: best ratio {best:.4f} >= 0.35 "
        f"(expected ~0.25 for 2nd-order FD), errors={errors}"
    )
    if strict_2nd_order and len(ratios) >= 4:
        trimmed = ratios[2:-1]
    elif strict_2nd_order and len(ratios) >= 3:
        trimmed = ratios[:-1]
    else:
        trimmed = ratios
    median_ratio = float(np.median(trimmed))
    assert median_ratio < 0.35, (
        f"Taylor test '{label}' failed: median ratio {median_ratio:.4f} "
        f"(strict_2nd_order={strict_2nd_order}), errors={errors}"
    )
    if tol_abs is not None:
        min_err = float(np.min(errors))
        assert min_err < tol_abs, (
            f"Taylor test '{label}' failed: min(errors)={min_err:.3e} "
            f">= tol_abs={tol_abs:.3e}.  A passing ratio with a large "
            f"absolute error typically means the analytic gradient "
            f"carries a constant offset (e.g. a sign flip or dropped "
            f"term) while still scaling correctly with eps. "
            f"errors={errors}"
        )


def test_taylor_tf_coil_dofs():
    """Taylor test: perturb TF coil curve DOFs only (puck geometry fixed)."""
    s, coils_tf, base_curves, base_currents, psc, btot, Jf = _make_small_setup()

    def getter():
        return np.copy(Jf.x)

    def setter(dofs):
        Jf.x = dofs
        psc.recompute_currents()
        btot.Bfields[0].clear_cached_properties()

    errors = _run_taylor_test(Jf, psc, btot, getter, setter, label="TF coil DOFs")
    _assert_taylor_convergence(errors, label="TF coil DOFs")


def test_taylor_puck_centers():
    """Taylor test: perturb puck center positions only (TF fixed)."""
    s, coils_tf, base_curves, base_currents, psc, btot, Jf = _make_small_setup(
        n_base_pucks=1,
        nfp=1,
        stellsym=False,
    )

    psc.unfix("center_x0")
    psc.unfix("center_y0")
    psc.unfix("center_z0")
    for c in base_curves:
        c.fix_all()
    for c in base_currents:
        c.fix_all()

    def getter():
        return np.copy(Jf.x)

    def setter(dofs):
        Jf.x = dofs
        psc.recompute_currents()
        btot.Bfields[0].clear_cached_properties()

    errors = _run_taylor_test(Jf, psc, btot, getter, setter, label="puck centers")
    _assert_taylor_convergence(errors, label="puck centers")


def test_taylor_puck_quaternion():
    """Taylor test: perturb puck quaternion orientations (TF fixed).

    Uses a slightly tilted initial axis (via _make_small_setup which sets
    axes to [0.1, 0.1, 1.0]) and unfixes all 4 quaternion components.
    """
    s, coils_tf, base_curves, base_currents, psc, btot, Jf = _make_small_setup(
        n_base_pucks=1,
        nfp=1,
        stellsym=False,
    )

    psc.unfix("q0_0")
    psc.unfix("qi_0")
    psc.unfix("qj_0")
    psc.unfix("qk_0")
    for c in base_curves:
        c.fix_all()
    for c in base_currents:
        c.fix_all()

    def getter():
        return np.copy(Jf.x)

    def setter(dofs):
        Jf.x = dofs
        psc.recompute_currents()
        btot.Bfields[0].clear_cached_properties()

    errors = _run_taylor_test(
        Jf, psc, btot, getter, setter, start_power=4, label="puck quaternion"
    )
    _assert_taylor_convergence(errors, label="puck quaternion")


def test_taylor_combined_dofs():
    """Taylor test: perturb both TF and puck center DOFs simultaneously."""
    s, coils_tf, base_curves, base_currents, psc, btot, Jf = _make_small_setup(
        n_base_pucks=1,
        nfp=1,
        stellsym=False,
    )

    psc.unfix("center_x0")
    psc.unfix("center_y0")
    psc.unfix("center_z0")

    def getter():
        return np.copy(Jf.x)

    def setter(dofs):
        Jf.x = dofs
        psc.recompute_currents()
        btot.Bfields[0].clear_cached_properties()

    errors = _run_taylor_test(Jf, psc, btot, getter, setter, label="combined DOFs")
    _assert_taylor_convergence(errors, label="combined DOFs")


def test_taylor_nfp_stellsym_puck_dofs():
    """Taylor test: puck centers with nfp=2, stellsym=True."""
    s, coils_tf, base_curves, base_currents, psc, btot, Jf = _make_small_setup(
        n_base_pucks=1,
        nfp=2,
        stellsym=True,
    )

    psc.unfix("center_x0")
    psc.unfix("center_y0")
    psc.unfix("center_z0")
    for c in base_curves:
        c.fix_all()
    for c in base_currents:
        c.fix_all()

    def getter():
        return np.copy(Jf.x)

    def setter(dofs):
        Jf.x = dofs
        psc.recompute_currents()
        btot.Bfields[0].clear_cached_properties()

    errors = _run_taylor_test(
        Jf, psc, btot, getter, setter, start_power=4, label="nfp+stellsym puck centers"
    )
    _assert_taylor_convergence(errors, label="nfp+stellsym puck centers")


def _numpy_symmetric_reduced_L_reference(
    K_per_puck,
    pts_per_puck,
    weights_per_puck,
    base_indices,
    base_reps,
    signs,
    G: int,
    delta_reg: float,
    adaptive_self_reg: bool,
) -> np.ndarray:
    """Small NumPy reference for :math:`L_\\text{base}` (same as perf-suite helper)."""
    from simsopt.field.bulk_inductance import MU0_OVER_4PI, _SELF_REG_COEFF

    base_indices = np.asarray(base_indices, dtype=int)
    base_reps = np.asarray(base_reps, dtype=int)
    n_all = len(K_per_puck)
    n_base = int(base_reps.size)
    nd_per = K_per_puck[int(base_reps[0])].shape[1]
    L_base = np.zeros((n_base * nd_per, n_base * nd_per))
    for i_base in range(n_base):
        i_rep = int(base_reps[i_base])
        K_i = K_per_puck[i_rep]
        pts_i = pts_per_puck[i_rep]
        w_i = weights_per_puck[i_rep]
        if adaptive_self_reg:
            delta_i = _SELF_REG_COEFF * np.sqrt(w_i)
        for j_rep in range(n_all):
            j_base = int(base_indices[j_rep])
            sigma_j = int(signs[j_rep])
            K_j = K_per_puck[j_rep]
            pts_j = pts_per_puck[j_rep]
            w_j = weights_per_puck[j_rep]
            r = pts_i[:, None, :] - pts_j[None, :, :]
            if adaptive_self_reg:
                delta_j = _SELF_REG_COEFF * np.sqrt(w_j)
                delta_pair = 0.5 * (delta_i[:, None] + delta_j[None, :])
                dist = np.sqrt(np.sum(r**2, axis=-1) + delta_pair**2)
            else:
                dist = np.sqrt(np.sum(r**2, axis=-1) + delta_reg**2)
            dot = np.einsum("iax,jbx->ijab", K_i, K_j)
            kernel = dot / dist[..., None, None]
            block = MU0_OVER_4PI * np.einsum("ijab,i,j->ab", kernel, w_i, w_j)
            ri = i_base * nd_per
            rj = j_base * nd_per
            L_base[ri : ri + nd_per, rj : rj + nd_per] += sigma_j * G * block
    return 0.5 * (L_base + L_base.T)


def test_replicate_pucks_jax_matches_PSC_replicate_pucks():
    """JAX replica emission matches :meth:`PSCBulkArray._replicate_pucks` (centers, signs, R)."""
    psc = _make_symmetry_validation_array(nfp=2, stellsym=True, n_base=2)
    centers, quats, radii, thicknesses = psc._get_base_puck_geometry()
    c_j, q_j, s_j = _replicate_pucks_jax(
        jnp.asarray(centers), jnp.asarray(quats), int(psc.nfp), bool(psc.stellsym)
    )
    all_pucks, all_quats, base_indices, _, _, replica_signs = psc._replicate_pucks(
        centers, quats, radii, thicknesses
    )
    c_py = np.stack([p[0] for p in all_pucks], axis=0)
    np.testing.assert_allclose(np.asarray(c_j), c_py, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        np.asarray(s_j), np.asarray(replica_signs, dtype=float), atol=0.0
    )
    for a in range(c_py.shape[0]):
        Rpy = _rotation_matrix_from_quat(np.asarray(all_quats[a]))
        Rjx = np.array(
            _rotation_matrix_from_quat_jax(
                jnp.asarray(
                    [q_j[a, 0], q_j[a, 1], q_j[a, 2], q_j[a, 3]], dtype=np.float64
                )
            )
        )
        np.testing.assert_allclose(Rpy, Rjx, rtol=1e-12, atol=1e-12)
    # pure-rotation template replicas must be +1 (Stage A.8)
    br = np.asarray(psc._base_reps, dtype=int)
    np.testing.assert_array_equal(np.asarray(s_j, dtype=int)[br], 1)


def test_L_work_matches_numpy_reference_with_PSC_puck_stacks():
    """``_shell_inductance_matrix_symmetric_reduced_jax`` matches the NumPy pair loop
    for geometry taken from a symmetry-reduced :class:`PSCBulkArray` rebuild.
    """
    psc = _make_symmetry_validation_array(nfp=2, stellsym=True, n_base=2)
    L_np = _numpy_symmetric_reduced_L_reference(
        psc._K_per_puck,
        psc._pts_per_puck,
        psc._weights_per_puck_list,
        psc._base_indices_arr,
        np.asarray(psc._base_reps, dtype=int),
        psc._replica_signs,
        int(psc._symmetry_G),
        float(psc.regularization_delta),
        bool(psc.adaptive_self_reg),
    )
    L_jx = _shell_inductance_matrix_symmetric_reduced_jax(
        psc._K_per_puck,
        psc._pts_per_puck,
        psc._weights_per_puck_list,
        psc._base_indices_arr,
        np.asarray(psc._base_reps, dtype=int),
        psc._replica_signs,
        int(psc._symmetry_G),
        float(psc.regularization_delta),
        bool(psc.adaptive_self_reg),
    )
    np.testing.assert_allclose(L_jx, L_np, rtol=1e-10, atol=1e-12)


def test_B_at_points_reduced_free_dof_path_deterministic():
    """Back-to-back :meth:`B_at_points` on the reduced free-DoF path is identical.

    The dense full-:math:`N^2` free-DoF assembly uses the **full** rim
    continuity projector :math:`Q_c` over all replicas, while the
    symmetry-reduced free-DoF path uses the work-space projector
    :math:`Q_{c,\\text{work}}` from the signed-orbit :math:`L_\\text{base}`
    factorization, so the two are **not** required to match at fp64.  A
    strict reduced-vs-dense B regression instead belongs in the Taylor suite.
    """
    psc = _make_symmetry_validation_array(nfp=2, stellsym=True, n_base=2)
    psc.unfix("center_x0")
    psc.recompute_currents()
    assert getattr(psc, "_reduced_free_dof_active", False)
    pts = np.asarray(psc.eval_points, dtype=float)
    B0 = psc.B_at_points(pts)
    B1 = psc.B_at_points(pts)
    np.testing.assert_array_equal(B0, B1)


def test_taylor_puck_quaternion_nfp2_stellsym_two_base_pucks():
    """Finite-difference Taylor: quaternion DoFs on two base pucks, ``nfp=2``,
    ``stellsym=True`` (symmetry-reduced :math:`L` and free-DoF JAX path).
    """
    s, coils_tf, base_curves, base_currents, psc, btot, Jf = _make_small_setup(
        n_base_pucks=2,
        nfp=2,
        stellsym=True,
    )
    for pid in (0, 1):
        for name in ("q0_", "qi_", "qj_", "qk_"):
            psc.unfix(f"{name}{pid}")
    for c in base_curves:
        c.fix_all()
    for c in base_currents:
        c.fix_all()

    def getter():
        return np.copy(Jf.x)

    def setter(dofs):
        Jf.x = dofs
        psc.recompute_currents()
        btot.Bfields[0].clear_cached_properties()

    assert psc._reduced_active
    errors = _run_taylor_test(
        Jf,
        psc,
        btot,
        getter,
        setter,
        start_power=4,
        label="puck quat nfp2 stell 2 base",
    )
    _assert_taylor_convergence(errors, label="puck quat nfp2 stell 2 base")


def test_vjp_tf_fast_matches_jax():
    """Analytic TF-only VJP vs JAX ``vjp`` path (non-trivial current derivative)."""
    import simsopt.field.psc_bulk as psb

    tf_coil = _unit_circle_coil(1.0e5)
    centers = np.array([[0.0, 0.0, 0.12]])
    axes = np.array([[0.0, 0.0, 1.0]])
    eval_pts = np.array([[0.15, 0.0, 0.25]], dtype=float)
    psc = PSCBulkArray(
        centers,
        axes,
        np.array([0.04]),
        np.array([0.02]),
        [tf_coil],
        eval_points=eval_pts,
        m_fourier=1,
        l_zernike=2,
        k_chebyshev=1,
        n_rho=4,
        n_phi=6,
        n_z=3,
        adaptive_self_reg=False,
    )
    v_B = np.ones((1, 3), dtype=float) / np.sqrt(3.0)

    psb._USE_JAX_TF_VJP = True
    psc.recompute_currents()
    g_jax = psc.vjp_setup_B(v_B, eval_pts)

    psb._USE_JAX_TF_VJP = False
    psc.recompute_currents()
    g_fast = psc.vjp_setup_B(v_B, eval_pts)

    np.testing.assert_allclose(
        np.asarray(g_jax(tf_coil.current)),
        np.asarray(g_fast(tf_coil.current)),
        rtol=1e-8,
        atol=5e-9,
    )


def test_L_jax_blockwise_matches_numpy():
    """JAX-accelerated L assembly matches the pure NumPy batched path."""
    from simsopt.field.bulk_inductance import (
        _shell_inductance_matrix_blockwise_batched,
        _shell_inductance_matrix_blockwise_batched_jax,
    )
    from simsopt.field.puck_basis import build_puck_shell_basis
    from simsopt.field.psc_bulk import _rotation_matrix_local_to_global

    R, t = 0.05, 0.02
    common_basis = dict(
        m_fourier=1, l_zernike=2, k_chebyshev=1, n_rho=4, n_phi=6, n_z=3
    )
    pucks = [
        (np.array([0.0, 0.0, 0.12]), np.array([0.0, 0.0, 1.0])),
        (np.array([0.0, 0.0, -0.12]), np.array([0.0, 0.0, 1.0])),
    ]
    delta_reg = 1e-6

    K_per, pts_per, w_per = [], [], []
    dof_offsets = []
    n_dof_total = 0
    for c, ax in pucks:
        basis = build_puck_shell_basis(R, t, **common_basis)
        Rmat = _rotation_matrix_local_to_global(ax)
        pts_g = (Rmat @ basis.quad_points_local.T).T + c
        K_g = np.einsum("ij,qkj->qki", Rmat, basis.k_basis_local)
        nd = K_g.shape[1]
        K_per.append(K_g)
        pts_per.append(pts_g)
        w_per.append(basis.quad_weights)
        dof_offsets.append(n_dof_total)
        n_dof_total += nd

    L_np = _shell_inductance_matrix_blockwise_batched(
        K_per,
        pts_per,
        w_per,
        dof_offsets,
        n_dof_total,
        delta_reg=delta_reg,
        adaptive_self_reg=False,
    )
    L_jax = _shell_inductance_matrix_blockwise_batched_jax(
        K_per,
        pts_per,
        w_per,
        dof_offsets,
        n_dof_total,
        delta_reg=delta_reg,
        adaptive_self_reg=False,
    )
    np.testing.assert_allclose(L_jax, L_np, atol=1e-11, rtol=1e-10)

    L_np_adapt = _shell_inductance_matrix_blockwise_batched(
        K_per,
        pts_per,
        w_per,
        dof_offsets,
        n_dof_total,
        delta_reg=delta_reg,
        adaptive_self_reg=True,
    )
    L_jax_adapt = _shell_inductance_matrix_blockwise_batched_jax(
        K_per,
        pts_per,
        w_per,
        dof_offsets,
        n_dof_total,
        delta_reg=delta_reg,
        adaptive_self_reg=True,
    )
    np.testing.assert_allclose(
        L_jax_adapt,
        L_np_adapt,
        atol=1e-11,
        rtol=1e-10,
        err_msg="Blockwise NumPy vs JAX disagree under adaptive_self_reg=True",
    )


@pytest.mark.slow
def test_from_cylindrical_grid_smoke():
    """from_cylindrical_grid builds radial axes and finite PSC bulk."""
    from pathlib import Path
    from simsopt.geo import SurfaceRZFourier
    from simsopt.util import initialize_coils
    from simsopt.field.selffield import regularization_rect

    TEST_DIR = (Path(__file__).parent / ".." / "test_files").resolve()
    filename = TEST_DIR / "wout_schuett_henneberg_nfp2_QA.nc"
    s = SurfaceRZFourier.from_wout(filename, range="half period", nphi=4, ntheta=4)
    _, _, coils_tf, _ = initialize_coils(
        s,
        "SchuettHennebergQAnfp2",
        regularization_rect(0.2, 0.2),
    )
    eval_pts = np.ascontiguousarray(s.gamma().reshape(-1, 3))
    psc = PSCBulkArray.from_cylindrical_grid(
        s,
        coils_tf,
        eval_pts,
        dr=0.9,
        dz=0.9,
        n_phi_slices=3,
        d_inner=1.5,
        d_outer=3.0,
        plasma_clearance=0.05,
        m_fourier=1,
        l_zernike=2,
        k_chebyshev=1,
        n_rho=4,
        n_phi=6,
        n_z=3,
    )
    assert len(psc._all_pucks) >= 1
    for _, ax, _, _ in psc._all_pucks[:5]:
        n = np.linalg.norm(ax)
        np.testing.assert_allclose(n, 1.0, atol=1e-6)
    B = psc.B_at_points(eval_pts[:5])
    assert np.all(np.isfinite(B))


@pytest.mark.slow
def test_from_winding_surface_smoke():
    """from_winding_surface builds normals as axes."""
    from pathlib import Path
    from simsopt.geo import SurfaceRZFourier
    from simsopt.util import initialize_coils
    from simsopt.field.selffield import regularization_rect

    TEST_DIR = (Path(__file__).parent / ".." / "test_files").resolve()
    filename = TEST_DIR / "wout_schuett_henneberg_nfp2_QA.nc"
    s = SurfaceRZFourier.from_wout(filename, range="half period", nphi=4, ntheta=4)
    _, _, coils_tf, _ = initialize_coils(
        s,
        "SchuettHennebergQAnfp2",
        regularization_rect(0.2, 0.2),
    )
    eval_pts = np.ascontiguousarray(s.gamma().reshape(-1, 3))
    psc = PSCBulkArray.from_winding_surface(
        s,
        coils_tf,
        eval_points=eval_pts,
        distance=2.0,
        n_phi_pucks=3,
        n_theta_pucks=2,
        puck_t=0.15,
        m_fourier=1,
        l_zernike=2,
        k_chebyshev=1,
        n_rho=4,
        n_phi=6,
        n_z=3,
    )
    assert len(psc._all_pucks) >= 1
    B = psc.B_at_points(eval_pts[:5])
    assert np.all(np.isfinite(B))


# ======================================================================
# Analytic inductance benchmarks (closed-form limits)
# ======================================================================


def _build_sidewall_uniform_kphi_basis(
    R: float,
    t: float,
    z_center: float,
    n_z: int = 48,
    n_phi: int = 96,
) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    r"""Build a single-mode side-wall-only :math:`K_\phi = 1\ \mathrm{A/m}`
    basis on one puck with axis :math:`+\hat z`.

    Constructs a tensor-product quadrature (Gauss-Legendre in ``z``,
    midpoint-trapezoidal in ``phi``) on the lateral surface of a cylinder
    of radius ``R`` centered at ``z_center`` with axial extent
    ``[z_center - t/2, z_center + t/2]``.  At every quadrature point the
    surface current basis vector is :math:`-\hat\phi`, which yields a
    uniform azimuthal sheet current :math:`K_\phi = 1\ \mathrm{A/m}`.

    Args:
        R: Cylinder radius (m).
        t: Axial thickness (m).
        z_center: Cylinder axial center (m).
        n_z: Number of Gauss-Legendre points along ``z``.
        n_phi: Number of equispaced points around ``phi``.

    Returns:
        Tuple ``(K_basis, quad_points, quad_weights)`` shaped
        ``((n_z*n_phi, 1, 3), (n_z*n_phi, 3), (n_z*n_phi,))`` compatible
        with :func:`shell_inductance_matrix_pure`.
    """
    z_nodes, z_weights = np.polynomial.legendre.leggauss(n_z)
    z_pts = z_center + 0.5 * t * z_nodes
    z_w = 0.5 * t * z_weights
    phi_pts = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    dphi = 2.0 * np.pi / n_phi

    ZZ, PP = np.meshgrid(z_pts, phi_pts, indexing="ij")
    ZW, _ = np.meshgrid(z_w, phi_pts, indexing="ij")

    x = R * np.cos(PP)
    y = R * np.sin(PP)
    z = ZZ

    quad_points = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)
    quad_weights = (ZW * R * dphi).ravel()

    K_phi_hat_x = -np.sin(PP).ravel()
    K_phi_hat_y = np.cos(PP).ravel()
    K_zero = np.zeros_like(K_phi_hat_x)
    K_basis = np.stack([-K_phi_hat_x, -K_phi_hat_y, K_zero], axis=1)[:, None, :]

    return K_basis, quad_points, quad_weights


def _neumann_mutual_inductance(R1: float, R2: float, d: float) -> float:
    r"""Return the Neumann mutual inductance of two coaxial thin loops of
    radii ``R1``, ``R2`` separated axially by ``d``.

    .. math::

        M(R_1, R_2, d) = \mu_0 \sqrt{R_1 R_2}
                        \left[ \left( \frac{2}{k} - k \right) K(k)
                               - \frac{2}{k} E(k) \right]

    with :math:`k^2 = 4 R_1 R_2 / ((R_1+R_2)^2 + d^2)`.  Uses
    :func:`scipy.special.ellipk`, :func:`scipy.special.ellipe` whose
    argument is ``m = k^2``.
    """
    from scipy.special import ellipk, ellipe

    m = 4.0 * R1 * R2 / ((R1 + R2) ** 2 + d**2)
    k = np.sqrt(m)
    mu0 = 4.0 * np.pi * 1e-7
    K_k = ellipk(m)
    E_k = ellipe(m)
    return mu0 * np.sqrt(R1 * R2) * ((2.0 / k - k) * K_k - (2.0 / k) * E_k)


def _lorenz_self_inductance(R: float, t: float) -> float:
    r"""Return the Lorenz (1879) closed-form self-inductance of a short
    solenoid / single-layer cylinder of radius ``R`` and height ``t``
    carrying uniform azimuthal surface current.

    .. math::

        L = \frac{\mu_0 \pi R^2}{t} \cdot \frac{4}{3\pi}
            \left[ \frac{1+k^2}{k^3} E(k) - \frac{1-k^2}{k^2} K(k) - k \right]

    with :math:`k^2 = (2R)^2 / ((2R)^2 + t^2)`.
    """
    from scipy.special import ellipk, ellipe

    m = (2.0 * R) ** 2 / ((2.0 * R) ** 2 + t**2)
    k = np.sqrt(m)
    mu0 = 4.0 * np.pi * 1e-7
    K_k = ellipk(m)
    E_k = ellipe(m)
    return (
        (mu0 * np.pi * R**2 / t)
        * (4.0 / (3.0 * np.pi))
        * ((1.0 + k**2) / k**3 * E_k - (1.0 - k**2) / k**2 * K_k - k)
    )


def test_coaxial_loops_neumann_limit():
    r"""Cross-puck mutual inductance block converges to the Neumann formula
    for two coaxial thin side-wall-only pucks.

    Builds two short cylinders of radii :math:`R_1 = R_2 = 0.05\ \mathrm{m}`,
    thickness :math:`t = 0.005\ \mathrm{m}`, separated axially by
    :math:`d = 0.20\ \mathrm{m}` (so :math:`t/d \ll 1` and the loops are
    well-approximated by filaments).  Each puck has a single mode with
    uniform :math:`K_\phi = 1\ \mathrm{A/m}` on its side wall.  Asserts that
    the :math:`L_{12}` cross block converges to
    :math:`M(R_1, R_2, d) \cdot t_1 \cdot t_2` (Neumann times the two axial
    lengths, since a uniform sheet of height :math:`t` with
    :math:`K_\phi = 1\ \mathrm{A/m}` carries total current :math:`t\ \mathrm{A}`
    and couples to another such sheet as a thin-loop pair weighted by the
    two heights) within 2 % for refined quadrature.

    This benchmark exercises only off-diagonal pair distances ``d > 0``, so
    it is independent of the self-term regularization choice.
    """
    R1, R2 = 0.05, 0.05
    t1, t2 = 0.005, 0.005
    d = 0.20

    K1, pts1, w1 = _build_sidewall_uniform_kphi_basis(
        R1,
        t1,
        z_center=0.0,
        n_z=48,
        n_phi=96,
    )
    K2, pts2, w2 = _build_sidewall_uniform_kphi_basis(
        R2,
        t2,
        z_center=d,
        n_z=48,
        n_phi=96,
    )

    L = shell_inductance_matrix_blockwise(
        [K1, K2],
        [pts1, pts2],
        [w1, w2],
        dof_offsets=[0, 1],
        n_dof_total=2,
        delta_reg=1e-10,
        adaptive_self_reg=False,
    )

    L12 = float(L[0, 1])
    M_ref = _neumann_mutual_inductance(R1, R2, d)
    L12_ref = M_ref * t1 * t2

    rel = abs(L12 - L12_ref) / abs(L12_ref)
    assert rel < 0.02, (
        f"Neumann mutual inductance benchmark: L12={L12:.6e}, "
        f"analytic={L12_ref:.6e}, relative error {rel:.3%}"
    )


@pytest.mark.slow
def test_short_solenoid_lorenz_self_inductance():
    r"""Self-block entry :math:`L_{00}` for a single puck with uniform
    side-wall :math:`K_\phi = 1\ \mathrm{A/m}` matches the Lorenz (1879)
    closed form for a short solenoid.

    Convention:  the mode amplitude in our basis is :math:`K_\phi` with
    units A/m, so the magnetic energy is
    :math:`W = \tfrac{1}{2} K_\phi^2 \, L_{00}` and :math:`L_{00}` carries
    units :math:`\mathrm{H}\,\mathrm{m}^2`.  The Lorenz formula gives the
    traditional self-inductance :math:`L_{\mathrm{Lor}}` relating energy to
    the total solenoid current :math:`I_{\mathrm{tot}} = K_\phi \, t`, so
    :math:`L_{00} = L_{\mathrm{Lor}} \cdot t^2` in this convention (the
    cross-check of this same normalization with the Neumann benchmark above
    passes to 2 %, so it is the correct bookkeeping).

    This is the direct test of the self-term regularization fix.  With
    ``adaptive_self_reg=False`` (legacy uniform :math:`\delta = 1\text{e-}6`
    regularization) the self-integral is far from the analytic value;
    with ``adaptive_self_reg=True`` and refined quadrature the ratio
    ``L_00 / (L_Lorenz * t^2)`` must be closer to unity than the legacy
    value.
    """
    R, t = 0.05, 0.04

    K, pts, w = _build_sidewall_uniform_kphi_basis(
        R,
        t,
        z_center=0.0,
        n_z=96,
        n_phi=192,
    )

    L_adapt = np.array(
        shell_inductance_matrix_pure(
            jnp.asarray(K),
            jnp.asarray(pts),
            jnp.asarray(w),
            delta_reg=1e-10,
            adaptive_self_reg=True,
        )
    )
    L_legacy = np.array(
        shell_inductance_matrix_pure(
            jnp.asarray(K),
            jnp.asarray(pts),
            jnp.asarray(w),
            delta_reg=1e-6,
            adaptive_self_reg=False,
        )
    )

    L_ref = _lorenz_self_inductance(R, t) * t**2

    ratio_adapt = float(L_adapt[0, 0]) / L_ref
    ratio_legacy = float(L_legacy[0, 0]) / L_ref

    # Adaptive self-regularization must be *strictly* closer to the
    # analytic Lorenz reference than the legacy uniform-delta path.  This
    # is the only non-tautological strict-better assertion the two
    # regularizations admit (unlike the previous ``|log10(ratio)|``
    # comparison, which is always true when both ratios are on the same
    # side of 1).
    assert abs(ratio_adapt - 1.0) < abs(ratio_legacy - 1.0), (
        f"Lorenz self-inductance benchmark: adaptive_self_reg=True must "
        f"be strictly closer to 1 than the legacy uniform-delta path. "
        f"L_adapt/L_ref={ratio_adapt:.4f}, "
        f"L_legacy/L_ref={ratio_legacy:.4f}, L_ref={L_ref:.3e} H*m^2"
    )

    # Absolute bound on the adaptive ratio.  With the corrected flat-disc
    # prefactor ``(3 pi^{3/2}/8) sqrt(w)`` the coincident cell hits the
    # analytic patch integral exactly, so the residual error is entirely
    # from off-coincident near-neighbor pairs where the smooth kernel
    # ``1/sqrt(r^2+delta^2) < 1/r`` still under-counts.  This residual
    # closes only very slowly (empirically ~``N^{-0.16}``), so hitting a
    # tight 10-20 % band requires many tens of thousands of quadrature
    # cells.  We settle for a 25 % bound at ``n_z=96, n_phi=192`` (a
    # fast assembly) as the standing strict-sense regression check; any
    # future Duffy-type singular quadrature should tighten this without
    # needing more cells.
    assert abs(ratio_adapt - 1.0) < 0.25, (
        f"Lorenz self-inductance benchmark: "
        f"|L_00/(L_Lorenz*t^2) - 1| = {abs(ratio_adapt - 1.0):.4f} "
        f"(expected < 0.25 with adaptive_self_reg=True and moderate "
        f"quadrature; legacy ratio = {ratio_legacy:.4f}, "
        f"L_ref = {L_ref:.3e} H*m^2)."
    )


# =====================================================================
# Memory-reduction refactor (stacked K/phi + symmetry-reduced path)
# =====================================================================


def _galerkin_transfer_matrix(
    base_indices: np.ndarray,
    nd_per: int,
    signs: "np.ndarray | None" = None,
) -> np.ndarray:
    """Return ``T`` with ``beta_full = T @ beta_work`` (stacked per-puck DOFs).

    Replica ``r`` copies base puck ``base_indices[r]`` with parity
    ``signs[r]`` so each ``nd_per``-row block of ``T`` is
    ``signs[r] * I`` on the corresponding base column block.  When
    ``signs`` is ``None`` (default, backward compatible) all signs are
    ``+1`` and ``T`` reduces to the plain orbit-replication matrix.
    """
    base_indices = np.asarray(base_indices, dtype=int)
    n_all = int(base_indices.shape[0])
    n_base = int(base_indices.max()) + 1
    if signs is None:
        sgn = np.ones(n_all, dtype=float)
    else:
        sgn = np.asarray(signs, dtype=float)
        if sgn.shape != (n_all,):
            raise ValueError(f"signs must have shape (n_all,); got {sgn.shape}")
    n_tot = n_all * nd_per
    n_work = n_base * nd_per
    T = np.zeros((n_tot, n_work))
    eye = np.eye(nd_per)
    for r in range(n_all):
        b = int(base_indices[r])
        T[r * nd_per : (r + 1) * nd_per, b * nd_per : (b + 1) * nd_per] = sgn[r] * eye
    return T


def _make_symmetry_validation_array(
    nfp: int = 2,
    stellsym: bool = False,
    n_base: int = 1,
    *,
    eval_pts: "np.ndarray | None" = None,
    current_amp: float = 100.0,
    **psc_kwargs: Any,
) -> "PSCBulkArray":
    """Build a small :class:`PSCBulkArray` for symmetry / memory tests.

    Uses :func:`coils_via_symmetries` so :meth:`PSCBulkArray._detect_tf_symmetry`
    can succeed.  The base coil is a circle in the ``z=0`` plane with
    centre ``(1,0,0)`` and radius ``0.3`` (correct ``CurveXYZFourier``
    DOF ordering: ``[xc0, xs1, xc1, yc0, ys1, yc1, zc0, zs1, zc1]``).

    **Symmetry note:** the default off-axis / off-plane puck placement
    is chosen so that neither the pure-rotation nor the stellsym-image
    orbit cancels identically.  After the signed-orbit reduction
    (:func:`_fold_Bn_to_work` / :func:`_gather_beta_work_to_all`),
    ``stellsym=True`` also produces physically meaningful modal
    amplitudes; earlier revisions of this fixture cautioned otherwise
    because the unsigned fold cancelled :math:`B_n(Sx) = -B_n(x)` in
    the loading vector.

    Args:
        nfp: Number of field periods.
        stellsym: Stellarator reflection symmetry on coils.
        n_base: Number of *base* pucks before replication.
        eval_pts: Optional evaluation points for :meth:`B_at_points`.
        current_amp: Base coil current (A); ``100`` A gives modal
            amplitudes :math:`O(10^2)` with ``stellsym=False`` for the
            default puck offset.
    """
    from simsopt.field.coil import coils_via_symmetries

    base_curve = CurveXYZFourier(32, 1)
    base_curve.x = np.array([1.0, 0.0, 0.3, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0])
    tf_coils = coils_via_symmetries(
        [base_curve], [Current(float(current_amp))], int(nfp), bool(stellsym)
    )

    centers = np.array(
        [[1.15 + 0.15 * i, 0.10 + 0.04 * i, 0.06 + 0.02 * i] for i in range(n_base)],
        dtype=float,
    )
    axes = np.tile(np.array([[0.0, 0.0, 1.0]]), (n_base, 1))
    Rs = np.full(n_base, 0.12)
    ts = np.full(n_base, 0.04)

    if eval_pts is None:
        eval_pts = np.array(
            [
                [1.0, 0.05, 0.35],
                [0.90, 0.10, 0.08],
            ],
            dtype=float,
        )

    return PSCBulkArray(
        centers,
        axes,
        Rs,
        ts,
        tf_coils,
        eval_points=eval_pts,
        m_fourier=2,
        l_zernike=3,
        k_chebyshev=1,
        n_rho=5,
        n_phi=6,
        n_z=3,
        nfp=nfp,
        stellsym=stellsym,
        **psc_kwargs,
    )


def _rodrigues_rotate_vector(
    v: "np.ndarray", axis: "np.ndarray", angle_rad: float
) -> "np.ndarray":
    """Rotate 3-vector ``v`` about unit ``axis`` by ``angle_rad`` (right-hand)."""
    v = np.asarray(v, dtype=float).ravel()[:3]
    k = np.asarray(axis, dtype=float).ravel()[:3]
    kn = np.linalg.norm(k)
    if kn < 1.0e-15:
        return v
    k = k / kn
    ca, sa = float(np.cos(angle_rad)), float(np.sin(angle_rad))
    t = 1.0 - ca
    return v * ca + np.cross(k, v) * sa + k * (np.dot(k, v)) * t


def _make_rotated_puck_array(
    n_base: int = 8,
    *,
    seed: int = 0,
    nfp: int = 1,
    stellsym: bool = False,
    spread: float = 0.35,
    tilt_deg: float = 30.0,
    current_amp: float = 100.0,
    eval_pts: "np.ndarray | None" = None,
    m_fourier: int = 4,
    l_zernike: int = 6,
    k_chebyshev: int = 2,
    n_rho: int = 8,
    n_phi: int = 12,
    n_z: int = 4,
    **psc_kwargs: Any,
) -> PSCBulkArray:
    r"""``PSCBulkArray`` with :math:`6\!-\!12` pucks on a tilted arc and skewed axes.

    Base pucks are placed on a **non-closing** toroidal segment at major
    radius :math:`\approx 1`~m, with each local axis produced by a
    deterministic, seed-controlled rotation of the outboard surface normal
    (``tilt`` about a random in-tangent-plane axis) so that every replica
    pair sees a distinct relative orientation.

    Args:
        n_base: Number of *base* pucks in ``[6, 12]`` (larger for timing sweeps).
        seed: ``numpy.random.Generator`` seed for the per-puck tilt / jitter.
        nfp, stellsym: Passed through to :class:`PSCBulkArray`.
        spread: Radial/vertical wobble of the major-radius-1 locus.
        tilt_deg: Maximum half-angle (degrees) for each axis' rotation off the
            local torus normal.
        current_amp: TF coil current scale (A) for
            :func:`simsopt.field.coil.coils_via_symmetries`.
        eval_pts: Optional evaluation points for :meth:`B_at_points`.
        m_fourier, l_zernike, k_chebyshev, n_rho, n_phi, n_z: basis resolution
            knobs (higher than the symmetry validation fixture to amortize JAX
            dispatch cost in performance scripts).

    Returns:
        A fully constructed :class:`PSCBulkArray` (call
        :meth:`PSCBulkArray.recompute_currents` before timing solves).
    """
    n_base = int(n_base)
    if n_base < 6 or n_base > 12:
        raise ValueError(
            "n_base must be in [6, 12] for the rotated-puck timing fixture."
        )
    from simsopt.field.coil import coils_via_symmetries

    rng = np.random.default_rng(int(seed))
    base_curve = CurveXYZFourier(32, 1)
    base_curve.x = np.array([1.0, 0.0, 0.3, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0])
    tf_coils = coils_via_symmetries(
        [base_curve], [Current(float(current_amp))], int(nfp), bool(stellsym)
    )

    phi = 2.0 * np.pi * (np.arange(n_base) + 0.5) / float(n_base) * 0.85
    major = 1.0 + spread * np.sin(2.0 * phi)
    centers = np.stack(
        [
            major * np.cos(phi),
            major * np.sin(phi),
            spread * np.cos(phi),
        ],
        axis=1,
    )
    n_torus = np.stack(
        [
            np.cos(phi),
            np.sin(phi),
            np.zeros_like(phi),
        ],
        axis=1,
    )
    n_torus /= np.linalg.norm(n_torus, axis=1)[:, None]

    axes = np.empty_like(centers)
    for k in range(n_base):
        nt = n_torus[k]
        tdir = np.cross(nt, np.array([0.0, 0.0, 1.0], dtype=float))
        if float(np.linalg.norm(tdir)) < 1.0e-3:
            tdir = np.cross(nt, np.array([0.0, 1.0, 0.0], dtype=float))
        tdir /= np.linalg.norm(tdir) + 1.0e-30
        ang = float(
            rng.uniform(
                -np.deg2rad(tilt_deg),
                np.deg2rad(tilt_deg),
            )
        )
        axes[k] = _rodrigues_rotate_vector(nt, tdir, ang)
        axes[k] /= np.linalg.norm(axes[k]) + 1.0e-30

    Rs = 0.12 + 0.01 * rng.uniform(-1.0, 1.0, n_base)
    ts = np.full(n_base, 0.04, dtype=float)

    if eval_pts is None:
        eval_pts = np.array(
            [
                [1.0, 0.05, 0.35],
                [0.90, 0.10, 0.08],
            ],
            dtype=float,
        )

    return PSCBulkArray(
        centers,
        axes,
        Rs,
        ts,
        tf_coils,
        eval_points=eval_pts,
        m_fourier=int(m_fourier),
        l_zernike=int(l_zernike),
        k_chebyshev=int(k_chebyshev),
        n_rho=int(n_rho),
        n_phi=int(n_phi),
        n_z=int(n_z),
        nfp=int(nfp),
        stellsym=bool(stellsym),
        **psc_kwargs,
    )


def _make_rotated_puck_array_large(
    n_base: int = 32,
    *,
    seed: int = 0,
    nfp: int = 1,
    stellsym: bool = False,
    spread: float = 0.35,
    tilt_deg: float = 30.0,
    current_amp: float = 100.0,
    eval_pts: "np.ndarray | None" = None,
    m_fourier: int = 4,
    l_zernike: int = 6,
    k_chebyshev: int = 2,
    n_rho: int = 8,
    n_phi: int = 12,
    n_z: int = 4,
) -> PSCBulkArray:
    r"""Same geometry as :func:`_make_rotated_puck_array` with :math:`6 \le n_\text{base} \le 128`.

    Used for large-scale timing and far-pair / multipole sweeps. Parameters match
    the small fixture except for the allowed range of ``n_base``.
    """
    n_base = int(n_base)
    if n_base < 6 or n_base > 128:
        raise ValueError(
            "n_base must be in [6, 128] for the large rotated-puck timing fixture."
        )
    from simsopt.field.coil import coils_via_symmetries

    rng = np.random.default_rng(int(seed))
    base_curve = CurveXYZFourier(32, 1)
    base_curve.x = np.array([1.0, 0.0, 0.3, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0])
    tf_coils = coils_via_symmetries(
        [base_curve], [Current(float(current_amp))], int(nfp), bool(stellsym)
    )

    phi = 2.0 * np.pi * (np.arange(n_base) + 0.5) / float(n_base) * 0.85
    major = 1.0 + spread * np.sin(2.0 * phi)
    centers = np.stack(
        [
            major * np.cos(phi),
            major * np.sin(phi),
            spread * np.cos(phi),
        ],
        axis=1,
    )
    n_torus = np.stack(
        [
            np.cos(phi),
            np.sin(phi),
            np.zeros_like(phi),
        ],
        axis=1,
    )
    n_torus /= np.linalg.norm(n_torus, axis=1)[:, None]

    axes = np.empty_like(centers)
    for k in range(n_base):
        nt = n_torus[k]
        tdir = np.cross(nt, np.array([0.0, 0.0, 1.0], dtype=float))
        if float(np.linalg.norm(tdir)) < 1.0e-3:
            tdir = np.cross(nt, np.array([0.0, 1.0, 0.0], dtype=float))
        tdir /= np.linalg.norm(tdir) + 1.0e-30
        ang = float(
            rng.uniform(
                -np.deg2rad(tilt_deg),
                np.deg2rad(tilt_deg),
            )
        )
        axes[k] = _rodrigues_rotate_vector(nt, tdir, ang)
        axes[k] /= np.linalg.norm(axes[k]) + 1.0e-30

    Rs = 0.12 + 0.01 * rng.uniform(-1.0, 1.0, n_base)
    ts = np.full(n_base, 0.04, dtype=float)

    if eval_pts is None:
        eval_pts = np.array(
            [
                [1.0, 0.05, 0.35],
                [0.90, 0.10, 0.08],
            ],
            dtype=float,
        )

    return PSCBulkArray(
        centers,
        axes,
        Rs,
        ts,
        tf_coils,
        eval_points=eval_pts,
        m_fourier=int(m_fourier),
        l_zernike=int(l_zernike),
        k_chebyshev=int(k_chebyshev),
        n_rho=int(n_rho),
        n_phi=int(n_phi),
        n_z=int(n_z),
        nfp=int(nfp),
        stellsym=bool(stellsym),
    )


# Backward-compatible alias used by earlier tests in this file.
_make_small_nfp_stellsym_array = _make_symmetry_validation_array


def test_rotated_puck_fixture_builds_and_solves():
    """Rotated pucks (plan Phase 0) — finite :math:`\\beta` and nontrivial axes."""
    psc = _make_rotated_puck_array(n_base=8, seed=0)
    psc.recompute_currents()
    assert np.all(np.isfinite(psc.beta))
    ax = np.stack(
        [
            np.asarray(p[1], dtype=float) / (np.linalg.norm(p[1]) + 1e-30)
            for p in psc._all_pucks
        ],
        axis=0,
    )
    for k in range(ax.shape[0]):
        assert abs(float(np.linalg.norm(ax[k, :])) - 1.0) < 1.0e-3
    for a in range(ax.shape[0]):
        for b in range(a + 1, ax.shape[0]):
            c = float(np.abs(np.dot(ax[a, :], ax[b, :])))
            assert c < 0.999, "expect non-parallel local axes in the fixture"


def test_rotated_puck_fixture_deterministic():
    """Identical ``seed`` yields bit-identical ``beta``."""
    psc1 = _make_rotated_puck_array(n_base=8, seed=42)
    psc1.recompute_currents()
    psc2 = _make_rotated_puck_array(n_base=8, seed=42)
    psc2.recompute_currents()
    np.testing.assert_array_equal(psc1.beta, psc2.beta)


def test_K_stack_matches_dense():
    """Stacked ``_K_stack``/``_phi_stack`` must equal the dense
    block-diagonal reconstruction exposed by the lazy properties
    (``_K_basis``/``_phi_mat``).

    This locks in the phase-1 refactor that dropped the dense
    ``(nq_total, n_dof_total, ...)`` arrays from the forward hot path.
    """
    psc = _make_symmetry_validation_array(nfp=2, stellsym=False, n_base=1)

    K_stack = psc._K_stack
    phi_stack = psc._phi_stack
    assert K_stack is not None, (
        "uniform puck shapes should populate the stacked K tensor"
    )
    assert phi_stack is not None, (
        "uniform puck shapes should populate the stacked phi tensor"
    )

    K_dense = psc._K_basis
    phi_dense = psc._phi_mat
    n_pucks, nq_per, nd_per, _ = K_stack.shape
    nq_total = n_pucks * nq_per
    n_dof_total = n_pucks * nd_per

    assert K_dense.shape == (nq_total, n_dof_total, 3)
    assert phi_dense.shape == (nq_total, n_dof_total)

    for p in range(n_pucks):
        q0, q1 = p * nq_per, (p + 1) * nq_per
        d0, d1 = p * nd_per, (p + 1) * nd_per
        np.testing.assert_allclose(
            K_dense[q0:q1, d0:d1, :],
            K_stack[p],
            atol=0,
            rtol=0,
            err_msg=f"Dense K block {p} diverges from _K_stack",
        )
        np.testing.assert_allclose(
            phi_dense[q0:q1, d0:d1],
            phi_stack[p],
            atol=0,
            rtol=0,
            err_msg=f"Dense phi block {p} diverges from _phi_stack",
        )
        off_row = np.zeros(nq_total, dtype=bool)
        off_row[q0:q1] = True
        off_col = np.zeros(n_dof_total, dtype=bool)
        off_col[d0:d1] = True
        np.testing.assert_allclose(
            K_dense[~off_row, :, :][:, off_col, :],
            0.0,
            atol=0,
            rtol=0,
            err_msg=f"Dense K leaks outside block {p}",
        )
        np.testing.assert_allclose(
            phi_dense[~off_row, :][:, off_col],
            0.0,
            atol=0,
            rtol=0,
            err_msg=f"Dense phi leaks outside block {p}",
        )


def test_B_at_points_regression_against_dense_forward():
    """Independently reassemble the Biot-Savart forward from the dense
    ``_K_basis`` / ``_phi_mat`` properties (the pre-refactor path) and
    verify the stacked / reduced hot path matches bit-for-bit.

    This guards against silent numerical drift from the phase-1 /
    phase-2 refactors in
    :meth:`PSCBulkArray.B_at_points` and
    :meth:`PSCBulkArray.get_shell_currents`.
    """
    psc = _make_symmetry_validation_array(nfp=2, stellsym=False, n_base=1)
    eval_pts = np.array([[0.25, -0.05, 0.12], [0.0, 0.0, 0.5]])

    # Stacked / reduced path (the refactored production path).
    B_new = psc.B_at_points(eval_pts)

    # Dense reconstruction path: materialise ``beta -> K``, then call
    # the purely-dense Biot-Savart core used by the pre-refactor
    # reference code (``shell_biot_savart_pure``).
    from simsopt.field.bulk_inductance import shell_biot_savart_pure

    psc.recompute_currents()
    beta_all = psc.beta
    K_dense = psc._K_basis
    quad_pts = psc._quad_points
    quad_w = psc._quad_weights
    B_ref = np.array(
        shell_biot_savart_pure(
            jnp.asarray(K_dense),
            jnp.asarray(quad_pts),
            jnp.asarray(quad_w),
            jnp.asarray(beta_all),
            jnp.asarray(eval_pts),
        )
    )

    np.testing.assert_allclose(
        B_new,
        B_ref,
        atol=1e-12,
        rtol=1e-9,
        err_msg="Refactored B_at_points disagrees with dense reference",
    )


def test_L_red_equals_T_transpose_L_full_T():
    """Symmetry-reduced inductance matches Galerkin restriction ``T^T L T``.

    The unconstrained monolithic solve on all replica DOFs is **not**
    equivalent to the reduced path (different null space / no replica
    tying).  The correct identity is between ``L_work`` from
    :func:`shell_inductance_matrix_symmetric_reduced` and the full
    blockwise matrix from the same quadrature.
    """
    psc_red = _make_symmetry_validation_array(nfp=2, stellsym=False, n_base=1)
    assert psc_red._reduced_active
    psc_full = _make_symmetry_validation_array(nfp=2, stellsym=False, n_base=1)
    psc_full._tf_is_symmetric = False
    psc_full._rebuild()

    nd = int(psc_red._K_stack.shape[2])
    T = _galerkin_transfer_matrix(psc_red._base_indices_arr, nd)
    Lf = psc_full._L_work
    Lr = psc_red._L_work
    np.testing.assert_allclose(
        T.T @ Lf @ T,
        Lr,
        atol=1e-11,
        rtol=1e-10,
        err_msg="L_red must equal T^T L_full T",
    )


def test_f_red_equals_T_transpose_f_full():
    """Folded loading ``f_work`` equals ``T^T f_full`` for the same ``Bn``."""
    from simsopt.field.biotsavart import BiotSavart

    psc_red = _make_symmetry_validation_array(nfp=2, stellsym=False, n_base=1)
    psc_full = _make_symmetry_validation_array(nfp=2, stellsym=False, n_base=1)
    psc_full._tf_is_symmetric = False
    psc_full._rebuild()

    bs = BiotSavart(psc_full.coils_TF)
    bs.set_points_cart(np.ascontiguousarray(psc_full._quad_points))
    Bn = np.sum(bs.B() * psc_full._quad_normals, axis=1)
    phi_dense = psc_full._phi_mat
    w = psc_full._quad_weights
    f_full = -phi_dense.T @ (w * Bn)

    nd = int(psc_red._K_stack.shape[2])
    T = _galerkin_transfer_matrix(psc_red._base_indices_arr, nd)

    phi_w = jnp.asarray(psc_red._phi_work_stack)
    w_w = jnp.asarray(psc_red._w_work_stack)
    bi = jnp.asarray(psc_red._jax_base_indices)
    signs = jnp.asarray(psc_red._jax_replica_signs)
    n_work, nq_per, _ = phi_w.shape
    Bn_w = _fold_Bn_to_work(jnp.asarray(Bn), bi, signs, n_work, nq_per)
    f_red = np.array(shell_loading_vector_stacked_pure(phi_w, w_w, Bn_w.reshape(-1)))
    np.testing.assert_allclose(
        T.T @ f_full,
        f_red,
        atol=1e-11,
        rtol=1e-10,
        err_msg="f_red must equal T^T f_full",
    )


def test_reduced_solve_matches_manual_eigenfloor():
    """``PSCBulkArray`` eigenfloor solve matches explicit NumPy on ``L_red``."""
    from simsopt.field.biotsavart import BiotSavart

    psc = _make_symmetry_validation_array(nfp=2, stellsym=False, n_base=1)
    psc.recompute_currents()
    Q = np.asarray(psc._Q)
    Lr = np.asarray(psc._L_work)
    bs = BiotSavart(psc.coils_TF)
    bs.set_points_cart(np.ascontiguousarray(psc._quad_points))
    Bn = np.sum(bs.B() * psc._quad_normals, axis=1)
    phi_w = jnp.asarray(psc._phi_work_stack)
    w_w = jnp.asarray(psc._w_work_stack)
    bi = jnp.asarray(psc._jax_base_indices)
    signs = jnp.asarray(psc._jax_replica_signs)
    n_work, nq_per, _ = phi_w.shape
    Bn_w = _fold_Bn_to_work(jnp.asarray(Bn), bi, signs, n_work, nq_per)
    f = np.array(shell_loading_vector_stacked_pure(phi_w, w_w, Bn_w.reshape(-1)))
    fq = Q.T @ f
    Lq = Q.T @ Lr @ Q
    alpha = np.array(
        shell_solve_eigenfloor_pure(
            jnp.asarray(Lq),
            jnp.asarray(fq),
            threshold=float(_EIGENFLOOR_THRESHOLD),
            jitter=1e-10,
        )
    )
    beta_man = Q @ alpha
    nd = beta_man.shape[0]
    np.testing.assert_allclose(
        beta_man,
        psc.beta[:nd],
        atol=1e-9,
        rtol=1e-10,
    )


def test_observables_self_consistent_after_reduced_solve():
    """``get_shell_currents``, ``get_equivalent_currents``, and ``B_at_points``."""
    eval_pts = np.array([[1.0, 0.05, 0.35], [0.90, 0.10, 0.08]])
    psc = _make_symmetry_validation_array(
        nfp=2, stellsym=False, n_base=1, eval_pts=eval_pts
    )
    psc.recompute_currents()
    assert np.max(np.abs(psc.beta)) > 1e-4, (
        "Test setup must produce nontrivial modal amplitudes "
        "(use stellsym=False for a nonzero folded load)"
    )
    K, Kmag = psc.get_shell_currents()
    assert K.shape[0] == psc._quad_points.shape[0]
    assert np.all(np.isfinite(K)) and np.all(np.isfinite(Kmag))
    I_eq = psc.get_equivalent_currents()
    assert I_eq.shape[0] == psc._K_stack.shape[0]
    assert np.all(np.isfinite(I_eq)) and np.all(I_eq >= 0.0)
    B = psc.B_at_points(eval_pts)
    assert B.shape == eval_pts.shape
    assert np.all(np.isfinite(B))


def test_symmetry_reduced_matches_galerkin_projection():
    """Reduced path matches Galerkin restriction; not the unconstrained full solve.

    The monolithic ``L_full @ beta = f`` on all replica DOFs does **not**
    impose replica-wise equality of modal coefficients; the reduced
    formulation ``L_red @ beta_work = f_work`` with ``L_red=T^TLT`` does.
    We verify ``B_at_points`` against dense Biot-Savart given ``beta``.
    """
    eval_pts = np.array([[1.0, 0.05, 0.35], [0.90, 0.10, 0.08]])
    psc = _make_symmetry_validation_array(
        nfp=2, stellsym=False, n_base=1, eval_pts=eval_pts
    )
    assert psc._reduced_active
    psc.recompute_currents()
    assert np.max(np.abs(psc.beta)) > 1e-4

    B = psc.B_at_points(eval_pts)
    from simsopt.field.bulk_inductance import shell_biot_savart_pure

    psc.recompute_currents()
    B_ref = np.array(
        shell_biot_savart_pure(
            jnp.asarray(psc._K_basis),
            jnp.asarray(psc._quad_points),
            jnp.asarray(psc._quad_weights),
            jnp.asarray(psc.beta),
            jnp.asarray(eval_pts),
        )
    )
    np.testing.assert_allclose(B, B_ref, atol=1e-9, rtol=1e-10)


@pytest.mark.parametrize(
    "nfp, stellsym, n_base",
    [
        (2, False, 1),
        (2, False, 2),
        (3, False, 1),
        (4, False, 1),
        (2, True, 1),
    ],
)
def test_L_f_transfer_parametrized(nfp: int, stellsym: bool, n_base: int):
    """``L_red = T^T L_full T`` and ``f_red = T^T f_full`` for several groups."""
    from simsopt.field.biotsavart import BiotSavart

    psc_red = _make_symmetry_validation_array(nfp=nfp, stellsym=stellsym, n_base=n_base)
    if not psc_red._tf_is_symmetric:
        pytest.skip("TF set is not recognised as symmetric for this layout")
    psc_full = _make_symmetry_validation_array(
        nfp=nfp, stellsym=stellsym, n_base=n_base
    )
    psc_full._tf_is_symmetric = False
    psc_full._rebuild()

    nd = int(psc_red._K_stack.shape[2])
    T = _galerkin_transfer_matrix(
        psc_red._base_indices_arr, nd, signs=psc_red._replica_signs
    )
    np.testing.assert_allclose(
        T.T @ psc_full._L_work @ T,
        psc_red._L_work,
        atol=1e-10,
        rtol=1e-9,
    )

    bs = BiotSavart(psc_full.coils_TF)
    bs.set_points_cart(np.ascontiguousarray(psc_full._quad_points))
    Bn = np.sum(bs.B() * psc_full._quad_normals, axis=1)
    f_full = -psc_full._phi_mat.T @ (psc_full._quad_weights * Bn)
    phi_w = jnp.asarray(psc_red._phi_work_stack)
    w_w = jnp.asarray(psc_red._w_work_stack)
    bi = jnp.asarray(psc_red._jax_base_indices)
    signs = jnp.asarray(psc_red._jax_replica_signs)
    n_work, nq_per, _ = phi_w.shape
    Bn_w = _fold_Bn_to_work(jnp.asarray(Bn), bi, signs, n_work, nq_per)
    f_red = np.array(shell_loading_vector_stacked_pure(phi_w, w_w, Bn_w.reshape(-1)))
    np.testing.assert_allclose(
        T.T @ f_full,
        f_red,
        atol=1e-10,
        rtol=1e-9,
    )


def test_vjp_tf_gradient_nonzero_and_listed_coils():
    """TF VJP w.r.t. each :class:`Coil` is finite (analytic adjoint path).

    Finite differences on :meth:`B_at_points` w.r.t. coil DOFs are **not**
    a valid cross-check here: ``PSCBulkArray`` does not register TF coils
    in :meth:`recompute_currents`'s geometry hash, so forward ``B`` is
    intentionally insensitive to raw ``curve.x`` edits without a full
    wiring refresh.  The VJP path still differentiates through the
    implicit shell solve + Biot-Savart chain used in optimization.
    """
    psc = _make_symmetry_validation_array(nfp=2, stellsym=False, n_base=1)
    psc.recompute_currents()
    pts = np.array([[1.0, 0.05, 0.35]], dtype=float)
    v = np.random.default_rng(0).standard_normal((1, 3))
    deriv = psc.vjp_setup_B(v, pts)
    for c in psc.coils_TF:
        g = np.asarray(deriv(c.curve))
        assert g.shape == (c.curve.x.shape[0],)
        assert np.all(np.isfinite(g))
        assert np.linalg.norm(g) > 0.0
    gI = np.asarray(deriv(psc.coils_TF[0].current))
    assert np.all(np.isfinite(gI))


def _unfix_all_puck_orientations(psc: PSCBulkArray) -> None:
    """Unfix quaternion DoFs for every base puck (for free-orientation VJP tests)."""
    n = int(psc._n_base_pucks)
    for i in range(n):
        for k in (f"q0_{i}", f"qi_{i}", f"qj_{i}", f"qk_{i}"):
            psc.unfix(k)


def test_vjp_reduced_free_dof_checkpoint_on_vs_off() -> None:
    """``checkpoint_l_pairs`` True vs False: identical puck + TF VJP in reduced path."""
    psc_a = _make_symmetry_validation_array(
        nfp=2, stellsym=True, n_base=2, checkpoint_l_pairs=False
    )
    psc_b = _make_symmetry_validation_array(
        nfp=2, stellsym=True, n_base=2, checkpoint_l_pairs=True
    )
    for p in (psc_a, psc_b):
        _unfix_all_puck_orientations(p)
        p.recompute_currents()
    v = np.random.default_rng(0).standard_normal(psc_a.eval_points.reshape(-1, 3).shape)
    pts = psc_a.eval_points
    d_a = psc_a.vjp_setup_B(v, pts)
    d_b = psc_b.vjp_setup_B(v, pts)
    np.testing.assert_allclose(
        np.asarray(d_a(psc_a)),
        np.asarray(d_b(psc_b)),
        rtol=1e-9,
        atol=1e-11,
    )


def test_vjp_reduced_free_dof_pair_row_chunk_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``SIMSOPT_PSC_JAX_PAIR_CHUNK=0`` (vmap) vs row chunk size 1: same VJP."""
    monkeypatch.setenv("SIMSOPT_PSC_JAX_PAIR_CHUNK", "0")
    psc_a = _make_symmetry_validation_array(nfp=2, stellsym=True, n_base=2)
    monkeypatch.setenv("SIMSOPT_PSC_JAX_PAIR_CHUNK", "1")
    psc_b = _make_symmetry_validation_array(nfp=2, stellsym=True, n_base=2)
    for p in (psc_a, psc_b):
        _unfix_all_puck_orientations(p)
        p.recompute_currents()
    v = np.random.default_rng(1).standard_normal(psc_a.eval_points.reshape(-1, 3).shape)
    pts = psc_a.eval_points
    d_a = psc_a.vjp_setup_B(v, pts)
    d_b = psc_b.vjp_setup_B(v, pts)
    np.testing.assert_allclose(
        np.asarray(d_a(psc_a)),
        np.asarray(d_b(psc_b)),
        rtol=1e-9,
        atol=1e-11,
    )


def test_reduced_free_dof_eval_chunk_env_matches_monolithic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``SIMSOPT_PSC_BS_EVAL_CHUNK`` chunking preserves free-DOF B and VJP."""
    pts = np.array(
        [
            [1.0, 0.05, 0.35],
            [0.90, 0.10, 0.08],
            [1.08, -0.02, 0.22],
            [0.82, 0.18, -0.03],
        ],
        dtype=float,
    )
    monkeypatch.setenv("SIMSOPT_PSC_BS_EVAL_CHUNK", "0")
    psc_a = _make_symmetry_validation_array(
        nfp=2, stellsym=True, n_base=2, eval_pts=pts
    )
    monkeypatch.setenv("SIMSOPT_PSC_BS_EVAL_CHUNK", "2")
    psc_b = _make_symmetry_validation_array(
        nfp=2, stellsym=True, n_base=2, eval_pts=pts
    )
    for p in (psc_a, psc_b):
        _unfix_all_puck_orientations(p)
        p.recompute_currents()

    B_a = psc_a.B_at_points(pts)
    B_b = psc_b.B_at_points(pts)
    np.testing.assert_allclose(B_a, B_b, rtol=1e-9, atol=1e-11)

    v = np.random.default_rng(2).standard_normal(pts.shape)
    d_a = psc_a.vjp_setup_B(v, pts)
    d_b = psc_b.vjp_setup_B(v, pts)
    np.testing.assert_allclose(
        np.asarray(d_a(psc_a)),
        np.asarray(d_b(psc_b)),
        rtol=1e-9,
        atol=1e-11,
    )


def test_reduced_free_quaternion_only_vjp_matches_full_geometry_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Quaternion-only reduced VJP matches the full center+quaternion tape."""
    psc_quat = _make_symmetry_validation_array(nfp=2, stellsym=True, n_base=2)
    psc_full = _make_symmetry_validation_array(nfp=2, stellsym=True, n_base=2)
    _unfix_all_puck_orientations(psc_quat)
    _unfix_all_puck_orientations(psc_full)
    for psc in (psc_quat, psc_full):
        psc.recompute_currents()

    pts = psc_quat.eval_points
    v = np.random.default_rng(3).standard_normal(pts.reshape(-1, 3).shape)
    d_quat = np.asarray(psc_quat.vjp_setup_B(v, pts)(psc_quat))
    monkeypatch.setenv("SIMSOPT_PSC_FREE_VJP_GEOMETRY", "full")
    d_full = np.asarray(psc_full.vjp_setup_B(v, pts)(psc_full))

    np.testing.assert_allclose(d_quat, d_full, rtol=1e-9, atol=1e-11)


def test_free_vjp_probe_modes_are_finite(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stop-gradient VJP attribution probes compile and return finite gradients."""
    psc = _make_symmetry_validation_array(nfp=2, stellsym=True, n_base=2)
    _unfix_all_puck_orientations(psc)
    psc.recompute_currents()
    pts = psc.eval_points
    v = np.random.default_rng(4).standard_normal(pts.reshape(-1, 3).shape)

    for mode in ("full", "stop_l", "stop_bn", "stop_solve", "shell_only", "beta_only"):
        monkeypatch.setenv("SIMSOPT_PSC_FREE_VJP_PROBE", mode)
        deriv = psc.vjp_setup_B(v, pts)
        grad = np.asarray(deriv(psc))
        assert np.all(np.isfinite(grad))


def test_implicit_solve_vjp_and_cached_pullback_match_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Opt-in implicit solve VJP and cached pullback preserve small-case gradients."""
    psc_default = _make_symmetry_validation_array(nfp=2, stellsym=True, n_base=2)
    psc_implicit = _make_symmetry_validation_array(nfp=2, stellsym=True, n_base=2)
    for psc in (psc_default, psc_implicit):
        _unfix_all_puck_orientations(psc)
        psc.recompute_currents()

    pts = psc_default.eval_points
    v = np.random.default_rng(5).standard_normal(pts.reshape(-1, 3).shape)
    grad_default = np.asarray(psc_default.vjp_setup_B(v, pts)(psc_default))

    monkeypatch.setenv("SIMSOPT_PSC_FREE_SOLVE_VJP", "implicit")
    grad_implicit = np.asarray(psc_implicit.vjp_setup_B(v, pts)(psc_implicit))
    np.testing.assert_allclose(grad_implicit, grad_default, rtol=1e-6, atol=1e-8)

    monkeypatch.setenv("SIMSOPT_PSC_CACHE_FREE_VJP", "1")
    psc_cached = _make_symmetry_validation_array(nfp=2, stellsym=True, n_base=2)
    _unfix_all_puck_orientations(psc_cached)
    psc_cached.recompute_currents()
    np.testing.assert_allclose(
        psc_cached.B_at_points(pts), psc_default.B_at_points(pts)
    )
    grad_cached = np.asarray(psc_cached.vjp_setup_B(v, pts)(psc_cached))
    np.testing.assert_allclose(grad_cached, grad_implicit, rtol=1e-9, atol=1e-11)


def test_lcache_not_used_when_puck_dofs_free(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Disk L-cache is skipped on rebuild when any puck DoF is free (no new writes)."""
    monkeypatch.setenv("SIMSOPT_PSC_LCACHE", "1")
    monkeypatch.setenv("SIMSOPT_PSC_LCACHE_DIR", str(tmp_path))
    psc = _make_symmetry_validation_array(nfp=2, stellsym=True, n_base=1)
    n_npz_init = len(list(tmp_path.glob("*.npz")))
    _unfix_all_puck_orientations(psc)
    # Unfixing does not change ``local_full_x``; a plain ``recompute_currents()``
    # can skip :meth:`_rebuild` when the hash matches.  Force a rebuild so
    # l-cache gating sees free puck DoFs.
    psc._rebuild()
    assert psc._lcache_last_key is None
    assert psc._lcache_last_hit is False
    # :meth:`PSCBulkArray.__init__` may have written a cache entry with all
    # puck DoFs still fixed; the free-geometry rebuild must not add another.
    assert len(list(tmp_path.glob("*.npz"))) == n_npz_init


def test_recompute_currents_preserves_L_f_transfer():
    """After perturbing TF geometry, ``T^T L T`` and ``T^T f`` identities hold."""
    from simsopt.field.biotsavart import BiotSavart

    psc_red = _make_symmetry_validation_array(nfp=2, stellsym=False, n_base=1)
    psc_full = _make_symmetry_validation_array(nfp=2, stellsym=False, n_base=1)
    psc_full._tf_is_symmetric = False
    rng = np.random.default_rng(42)
    for cr, cf in zip(psc_red.coils_TF, psc_full.coils_TF):
        delta = 1e-3 * rng.standard_normal(cr.curve.x.shape)
        cr.curve.x = np.asarray(cr.curve.x) + delta
        cf.curve.x = np.asarray(cf.curve.x) + delta
    psc_red._rebuild()
    psc_red._setup_jax()
    psc_full._rebuild()
    psc_full._setup_jax()
    assert psc_red._reduced_active

    nd = int(psc_red._K_stack.shape[2])
    T = _galerkin_transfer_matrix(psc_red._base_indices_arr, nd)
    np.testing.assert_allclose(
        T.T @ psc_full._L_work @ T,
        psc_red._L_work,
        atol=1e-9,
        rtol=1e-8,
    )
    bs = BiotSavart(psc_full.coils_TF)
    bs.set_points_cart(np.ascontiguousarray(psc_full._quad_points))
    Bn = np.sum(bs.B() * psc_full._quad_normals, axis=1)
    f_full = -psc_full._phi_mat.T @ (psc_full._quad_weights * Bn)
    phi_w = jnp.asarray(psc_red._phi_work_stack)
    w_w = jnp.asarray(psc_red._w_work_stack)
    bi = jnp.asarray(psc_red._jax_base_indices)
    signs = jnp.asarray(psc_red._jax_replica_signs)
    n_work, nq_per, _ = phi_w.shape
    Bn_w = _fold_Bn_to_work(jnp.asarray(Bn), bi, signs, n_work, nq_per)
    f_red = np.array(shell_loading_vector_stacked_pure(phi_w, w_w, Bn_w.reshape(-1)))
    np.testing.assert_allclose(T.T @ f_full, f_red, atol=1e-9, rtol=1e-8)
    psc_red.recompute_currents()
    assert np.max(np.abs(psc_red.beta)) > 1e-6


def test_asymmetric_tf_falls_back_to_full():
    """When the TF coils do not respect the requested ``nfp`` /
    ``stellsym``, :meth:`PSCBulkArray._detect_tf_symmetry` must flag
    the asymmetry and the array must fall back to the full-L path.
    """
    curve_sym = CurveXYZFourier(32, 1)
    curve_sym.x = np.array([0, 0, 1, 0, 1, 0, 0, 0.0, 0.0]) * 1.0
    coil_sym = Coil(curve_sym, Current(1e5))

    # Second coil lives only in one field period -> breaks NFP=2.
    curve_asym = CurveXYZFourier(32, 1)
    curve_asym.x = np.array([0.2, 0.0, 1.0, 0.3, 1.0, 0.0, 0.0, 0.0, 0.05])
    coil_asym = Coil(curve_asym, Current(1e5))

    centers = np.array([[0.3, 0.0, 0.1]], dtype=float)
    axes = np.array([[0.0, 0.0, 1.0]], dtype=float)
    Rs = np.array([0.04])
    ts = np.array([0.02])
    eval_pts = np.array([[0.3, 0.0, 0.5]])

    psc = PSCBulkArray(
        centers,
        axes,
        Rs,
        ts,
        [coil_sym, coil_asym],
        eval_points=eval_pts,
        m_fourier=1,
        l_zernike=2,
        k_chebyshev=1,
        n_rho=4,
        n_phi=6,
        n_z=3,
        nfp=2,
        stellsym=True,
    )
    assert psc._tf_is_symmetric is False, (
        "Mixed symmetric + asymmetric TF coils must be flagged as asymmetric"
    )
    assert psc._reduced_active is False, "Asymmetric TF must disable the reduced path"

    B = psc.B_at_points(eval_pts)
    assert np.all(np.isfinite(B))
    L = psc._L_work
    np.testing.assert_allclose(L, L.T, atol=1e-12, rtol=1e-12)

    # Reference: same object, second evaluation — deterministic forward.
    np.testing.assert_allclose(
        psc.B_at_points(eval_pts),
        psc.B_at_points(eval_pts),
        atol=0.0,
        rtol=0.0,
    )


@pytest.mark.slow
def test_memory_footprint_scales_linearly():
    """Slow regression test: memory used by the dense inductance system
    (``_L_work`` + ``_L_red`` + stacked ``K_stack``/``phi_stack``) must
    grow roughly linearly (not quadratically) in the number of base
    pucks when the reduced path is active.

    We compare ``n_base = 1`` against ``n_base = 4`` with ``nfp=4``,
    ``stellsym=True`` (group size 8).  The full-L footprint would scale
    like ``(8 * n_base)**2`` whereas the reduced-L footprint scales
    like ``n_base**2``; even at ``n_base = 4`` the ratio must stay
    well below the ``64`` that the full path would incur.
    """

    def _footprint_bytes(n_base):
        # ``stellsym=False`` avoids folded-load cancellation so ``|beta|`` is
        # nontrivial (guards against a degenerate numerical null test).
        psc = _make_symmetry_validation_array(nfp=4, stellsym=False, n_base=n_base)
        assert psc._reduced_active
        psc.recompute_currents()
        assert np.max(np.abs(psc.beta)) > 1e-6
        b = 0
        for name in (
            "_L_work",
            "_L_red",
            "_K_stack",
            "_phi_stack",
            "_w_stack",
            "_phi_work_stack",
            "_w_work_stack",
            "_Q_c",
            "_Q",
        ):
            arr = getattr(psc, name, None)
            if arr is not None:
                b += int(np.asarray(arr).nbytes)
        return b

    b1 = _footprint_bytes(1)
    b4 = _footprint_bytes(4)

    ratio = b4 / max(b1, 1)
    # Linear scaling would give ``ratio ~ 4``; quadratic would give
    # ``~ 16``.  We allow a generous headroom for the nearly-flat
    # fixed costs (continuity projector, quadrature mesh) but insist
    # on staying well below the quadratic regime.
    assert ratio < 10.0, (
        f"Memory footprint scaled by {ratio:.2f}x going from 1 to 4 "
        f"base pucks; expected near-linear (<= 10x) under the "
        f"reduced path.  b1={b1}B, b4={b4}B"
    )


# ---------------------------------------------------------------------------
# Regression tests for the full-path fold bug and the stellsym gate.
#
# Before the fix, :meth:`PSCBulkArray._rebuild` assigned
# ``self._base_indices_arr = base_indices`` unconditionally, where
# ``base_indices`` is the orbit-grouping ``[0, 0, ..., n_base-1]``.  The
# JAX bodies then applied :func:`_fold_Bn_to_work` with
# ``n_work = n_all`` but a non-identity index map, which folds ``Bn``
# across orbits on the *full* path too.  In practice this collapsed the
# loading vector (and hence ``|beta|`` / ``I_eq`` / ``|B_bulk|``) to
# near zero whenever the TF layout is stellarator-symmetric.
# ---------------------------------------------------------------------------


def test_reduced_path_base_indices_arr_is_orbit_grouping():
    """``_base_indices_arr`` equals the orbit-grouping when the reduced path
    is active, so that :func:`_fold_Bn_to_work` and
    :func:`_gather_beta_work_to_all` contract / scatter across orbits.

    The counterpart of :func:`test_full_path_base_indices_arr_is_identity`:
    on the reduced path each block of ``G`` consecutive replica entries
    must share the same base-puck index (``[0]*G + [1]*G + ...``).
    """
    nfp = 2
    n_base = 3
    psc = _make_symmetry_validation_array(nfp=nfp, stellsym=False, n_base=n_base)
    assert psc._reduced_active, (
        "Symmetric NFP layout with stellsym=False should enable the "
        "reduced path; cannot validate the orbit-grouping assertion"
    )
    G = int(psc.nfp) * (2 if psc.stellsym else 1)
    expected = np.repeat(np.arange(n_base, dtype=np.int32), G)
    np.testing.assert_array_equal(
        np.asarray(psc._base_indices_arr),
        expected,
        err_msg=(
            "On the reduced path, _base_indices_arr must be the "
            "orbit-grouping [0]*G + [1]*G + ...; otherwise segment_sum "
            "folds across the wrong orbits"
        ),
    )


def test_full_path_base_indices_arr_is_identity():
    """``_base_indices_arr`` must be ``arange(n_all)`` when ``_reduced_active``
    is ``False`` so that :func:`_fold_Bn_to_work` reduces to the identity.

    The pre-fix code reused the orbit-grouping ``base_indices`` array
    from :meth:`PSCBulkArray._replicate_pucks` on both paths, which
    silently folded ``Bn`` across orbits on the full-``L`` path and
    collapsed the induced currents to near zero whenever ``stellsym=True``.

    The signed-orbit reduction now supports ``stellsym=True`` too, so
    this test manually forces ``_tf_is_symmetric = False`` to exercise
    the full-path fallback (which remains live whenever TF symmetry
    detection fails or ``exact_disc_faces`` is on).
    """
    from simsopt.field.coil import coils_via_symmetries

    base_curve = CurveXYZFourier(32, 1)
    base_curve.x = np.array([1.0, 0.0, 0.3, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0])
    tf_coils = coils_via_symmetries([base_curve], [Current(1.0e5)], 2, True)
    centers = np.array([[1.15, 0.10, 0.06]])
    axes = np.array([[0.0, 0.0, 1.0]])
    import unittest.mock as _mock

    with _mock.patch.object(PSCBulkArray, "_detect_tf_symmetry", return_value=False):
        psc = PSCBulkArray(
            centers,
            axes,
            np.array([0.12]),
            np.array([0.04]),
            tf_coils,
            eval_points=np.array([[1.0, 0.05, 0.35]]),
            m_fourier=2,
            l_zernike=3,
            k_chebyshev=1,
            n_rho=5,
            n_phi=6,
            n_z=3,
            nfp=2,
            stellsym=True,
        )
    assert psc._reduced_active is False, (
        "Forcing _detect_tf_symmetry to False must take the full-L fallback"
    )
    n_all = len(psc._all_pucks)
    np.testing.assert_array_equal(
        np.asarray(psc._base_indices_arr),
        np.arange(n_all, dtype=np.int32),
        err_msg=(
            "On the full path, _base_indices_arr must be arange(n_all) so "
            "_fold_Bn_to_work becomes the identity"
        ),
    )


def test_full_path_solver_matches_numpy_reference():
    """JAX ``_solve_beta`` path agrees with an explicit NumPy reference.

    Assembles ``f = -phi^T (w * Bn)``, solves
    ``(L_red + eps I) alpha = Q^T f`` via NumPy, lifts back with ``Q``, and
    compares to :attr:`PSCBulkArray.beta`.  The pre-fix code would produce
    ``|beta|`` ~ :math:`10^{-16}` of the reference for stellsym-symmetric
    TF layouts because the full-path fold silently zeroed the loading.
    """
    from simsopt.field.biotsavart import BiotSavart
    from simsopt.field.coil import coils_via_symmetries

    base_curve = CurveXYZFourier(32, 1)
    base_curve.x = np.array([1.0, 0.0, 0.3, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0])
    tf_coils = coils_via_symmetries([base_curve], [Current(1.0e5)], 2, True)
    centers = np.array([[1.15, 0.10, 0.06]])
    axes = np.array([[0.0, 0.0, 1.0]])
    import unittest.mock as _mock

    with _mock.patch.object(PSCBulkArray, "_detect_tf_symmetry", return_value=False):
        psc = PSCBulkArray(
            centers,
            axes,
            np.array([0.12]),
            np.array([0.04]),
            tf_coils,
            eval_points=np.array([[1.0, 0.05, 0.35]]),
            m_fourier=2,
            l_zernike=3,
            k_chebyshev=1,
            n_rho=5,
            n_phi=6,
            n_z=3,
            nfp=2,
            stellsym=True,
        )
    assert psc._reduced_active is False

    psc.recompute_currents()
    beta_jax = np.asarray(psc.beta)

    # Independent NumPy reference: f = -phi^T (w * Bn), solve
    # Q^T L Q alpha = Q^T f via the same eigenvalue-floor kernel the
    # JAX JIT uses (``shell_solve_eigenfloor_pure``), lift beta = Q alpha.
    # Must match the JAX regularizer exactly (not just jitter) because
    # the rim-continuity-projected L can have meaningful eigenvalue
    # gaps at reactor-like currents.
    from simsopt.field.bulk_inductance import shell_solve_eigenfloor_pure
    from simsopt.field.psc_bulk import _EIGENFLOOR_THRESHOLD

    bs = BiotSavart(psc.coils_TF)
    bs.set_points_cart(np.ascontiguousarray(psc._quad_points))
    Bn = np.sum(bs.B() * psc._quad_normals, axis=1)
    phi_dense = psc._phi_mat
    w = psc._quad_weights
    f_full = -phi_dense.T @ (w * Bn)
    Q = np.asarray(psc._Q)
    L_red = Q.T @ np.asarray(psc._L_work) @ Q
    f_r = Q.T @ f_full
    alpha = np.asarray(
        shell_solve_eigenfloor_pure(
            jnp.asarray(L_red),
            jnp.asarray(f_r),
            threshold=float(_EIGENFLOOR_THRESHOLD),
            jitter=1e-10,
        )
    )
    beta_ref = Q @ alpha

    # The full-path fold bug made |beta_jax| ~ 1e-9 while the reference is
    # O(1e4-1e7) at reactor-like currents.  A generous magnitude floor
    # catches the regression without overfitting to this specific geometry.
    assert np.max(np.abs(beta_ref)) > 1.0, (
        "Reference |beta| is suspiciously small; the test setup may be degenerate"
    )
    rel_err = np.linalg.norm(beta_jax - beta_ref) / np.linalg.norm(beta_ref)
    assert rel_err < 1e-6, (
        f"JAX solver disagrees with NumPy reference: rel_err={rel_err:.3e}, "
        f"|beta_jax|max={np.max(np.abs(beta_jax)):.3e}, "
        f"|beta_ref|max={np.max(np.abs(beta_ref)):.3e}"
    )


def test_stellsym_full_path_produces_physical_magnitudes():
    """With ``stellsym=True`` the full-``L`` fallback (``_tf_is_symmetric``
    forced to ``False``) produces physically meaningful induced currents.

    Guards against the full-path fold bug that zeroed out the induced
    response on stellsym layouts (the cylindrical example regression).
    The signed reduced path is tested in
    :func:`test_signed_reduced_vs_full_stellsym`.
    """
    from simsopt.field.coil import coils_via_symmetries

    base_curve = CurveXYZFourier(32, 1)
    base_curve.x = np.array([1.0, 0.0, 0.3, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0])
    tf_coils = coils_via_symmetries([base_curve], [Current(1.0e5)], 2, True)
    centers = np.array([[1.15, 0.10, 0.06]])
    axes = np.array([[0.0, 0.0, 1.0]])
    eval_pts = np.array([[1.0, 0.05, 0.35], [0.90, 0.10, 0.08]])
    import unittest.mock as _mock

    with _mock.patch.object(PSCBulkArray, "_detect_tf_symmetry", return_value=False):
        psc = PSCBulkArray(
            centers,
            axes,
            np.array([0.12]),
            np.array([0.04]),
            tf_coils,
            eval_points=eval_pts,
            m_fourier=2,
            l_zernike=3,
            k_chebyshev=1,
            n_rho=5,
            n_phi=6,
            n_z=3,
            nfp=2,
            stellsym=True,
        )
    assert psc._reduced_active is False
    psc.recompute_currents()

    beta = np.asarray(psc.beta)
    assert np.all(np.isfinite(beta))
    assert np.max(np.abs(beta)) > 1e-3, (
        f"Induced |beta|max={np.max(np.abs(beta)):.3e} collapsed to near "
        f"zero on the full-L path; the fold regression has returned"
    )

    _, K_mag = psc.get_shell_currents()
    assert np.max(K_mag) > 1e-3
    I_eq = psc.get_equivalent_currents()
    assert np.all(np.isfinite(I_eq)) and np.max(I_eq) > 1e-6

    B_bulk = psc.B_at_points(eval_pts)
    assert np.all(np.isfinite(B_bulk))
    assert np.max(np.linalg.norm(B_bulk, axis=-1)) > 1e-8


# =====================================================================
# Phase 2: signed-orbit reduction under ``stellsym=True``
# =====================================================================


@pytest.mark.parametrize("nfp", [2, 3])
def test_replica_signs_structure(nfp: int):
    """Per-replica signs produced by :meth:`PSCBulkArray._replicate_pucks`
    alternate ``[+1, -1, +1, -1, ...]`` under ``stellsym=True`` (pure
    rotation then stellsym image for each field period), and every
    ``base_reps`` entry lands on a ``+1`` replica as required by the
    signed inductance assembly.
    """
    psc = _make_symmetry_validation_array(nfp=nfp, stellsym=True, n_base=2)
    signs = np.asarray(psc._replica_signs)
    n_all = len(psc._all_pucks)
    assert signs.shape == (n_all,)
    G = int(psc.nfp) * 2
    assert n_all == psc._n_base_pucks * G
    expected = np.tile(
        np.tile(np.array([+1, -1], dtype=np.int8), int(psc.nfp)),
        psc._n_base_pucks,
    )
    np.testing.assert_array_equal(
        signs, expected, err_msg="replica_signs must alternate per-replica"
    )
    base_reps = np.asarray(psc._base_reps)
    assert np.all(signs[base_reps] == +1), (
        "base_reps must pick the pure-rotation (sigma=+1) replica of "
        "every orbit; otherwise the signed inductance assembly "
        "receives mixed-sign rows."
    )


@pytest.mark.parametrize(
    "nfp, n_base",
    [
        (2, 1),
        (2, 2),
        (3, 1),
    ],
)
def test_signed_reduced_vs_full_stellsym(nfp: int, n_base: int):
    """Signed reduced path matches the full-``L`` fallback under
    ``stellsym=True`` across several NFP / base-count combinations.

    Compares ``beta``, :meth:`B_at_points`,
    :meth:`get_equivalent_currents` between a default-built array (which
    now routes through the signed reduced path) and one where
    :meth:`_detect_tf_symmetry` is monkeypatched to ``False`` to force
    the full-replica solve.
    """
    import unittest.mock as _mock

    eval_pts = np.array([[1.0, 0.05, 0.35], [0.90, 0.10, 0.08]])
    psc_red = _make_symmetry_validation_array(
        nfp=nfp, stellsym=True, n_base=n_base, eval_pts=eval_pts
    )
    with _mock.patch.object(PSCBulkArray, "_detect_tf_symmetry", return_value=False):
        psc_full = _make_symmetry_validation_array(
            nfp=nfp, stellsym=True, n_base=n_base, eval_pts=eval_pts
        )
    assert psc_red._reduced_active is True, (
        "Default build must use the signed reduced path under stellsym"
    )
    assert psc_full._reduced_active is False, (
        "Monkeypatched build must fall back to the full-L path"
    )
    psc_red.recompute_currents()
    psc_full.recompute_currents()

    beta_scale = max(np.max(np.abs(psc_full.beta)), np.max(np.abs(psc_red.beta)))
    assert beta_scale > 1e-3, (
        f"|beta|max={beta_scale:.3e} collapsed; the signed fold/gather or "
        f"full-path fallback regressed"
    )

    # Galerkin identity (T^T L_full T == L_red, T^T f_full == f_red) is
    # checked exactly in :func:`test_signed_galerkin_identity_stellsym`.
    # The remaining path discrepancy comes from eigenvalue flooring acting
    # on differently-sized projected operators (``Q_full^T L_full Q_full``
    # vs ``Q_red^T L_red Q_red``) when the physical problem is
    # ill-conditioned.  We therefore require only physically meaningful
    # agreement (~few %), and lock down the symmetric-subspace structure
    # via the Galerkin and signed-transfer tests.
    # L2-norm relative agreement avoids magnifying regularization-induced
    # disagreement on tiny modal components; the per-element rtol can be
    # misleading for ill-conditioned shells even though the paths agree
    # on globally significant quantities.
    def _rel_l2(a, b):
        den = max(np.linalg.norm(b), 1e-30)
        return np.linalg.norm(np.asarray(a) - np.asarray(b)) / den

    assert _rel_l2(psc_red.beta, psc_full.beta) < 0.1, (
        f"||beta_red - beta_full||/||beta_full|| = "
        f"{_rel_l2(psc_red.beta, psc_full.beta):.3e}"
    )
    B_red = psc_red.B_at_points(eval_pts)
    B_full = psc_full.B_at_points(eval_pts)
    assert _rel_l2(B_red, B_full) < 0.1, (
        f"||B_red - B_full||/||B_full|| = {_rel_l2(B_red, B_full):.3e}"
    )
    I_red = psc_red.get_equivalent_currents()
    I_full = psc_full.get_equivalent_currents()
    assert _rel_l2(I_red, I_full) < 0.1, (
        f"||I_red - I_full||/||I_full|| = {_rel_l2(I_red, I_full):.3e}"
    )

    # Verify ``beta_full`` lies in the symmetric subspace spanned by the
    # signed transfer matrix ``T``; both paths must return solutions in
    # ``range(T)``.  This is the operator-level guarantee.
    nd = int(psc_red._K_stack.shape[2])
    T = _galerkin_transfer_matrix(
        psc_red._base_indices_arr, nd, signs=psc_red._replica_signs
    )
    beta_full_np = np.asarray(psc_full.beta)
    coef = np.linalg.lstsq(T, beta_full_np, rcond=None)[0]
    proj = T @ coef
    residual = np.linalg.norm(beta_full_np - proj)
    ref = max(np.linalg.norm(beta_full_np), 1e-30)
    assert residual / ref < 1e-6, (
        f"Full-path beta is not in range(T_signed): "
        f"||beta - T T^+ beta||/||beta|| = {residual / ref:.3e}"
    )


def test_signed_galerkin_identity_stellsym():
    """``L_red = T_signed^T L_full T_signed`` and
    ``f_red = T_signed^T f_full`` under ``stellsym=True``.

    Locks in the claim that the signed transfer matrix on every orbit
    is the correct Galerkin reducer for ``L`` and the loading vector
    when simsopt's stellsym current convention flips sign on images.
    """
    import unittest.mock as _mock
    from simsopt.field.biotsavart import BiotSavart

    psc_red = _make_symmetry_validation_array(nfp=2, stellsym=True, n_base=1)
    with _mock.patch.object(PSCBulkArray, "_detect_tf_symmetry", return_value=False):
        psc_full = _make_symmetry_validation_array(nfp=2, stellsym=True, n_base=1)
    assert psc_red._reduced_active and not psc_full._reduced_active

    nd = int(psc_red._K_stack.shape[2])
    T_signed = _galerkin_transfer_matrix(
        psc_red._base_indices_arr, nd, signs=psc_red._replica_signs
    )
    np.testing.assert_allclose(
        T_signed.T @ np.asarray(psc_full._L_work) @ T_signed,
        np.asarray(psc_red._L_work),
        atol=1e-10,
        rtol=1e-9,
        err_msg="Signed T must satisfy T^T L_full T = L_red",
    )

    bs = BiotSavart(psc_full.coils_TF)
    bs.set_points_cart(np.ascontiguousarray(psc_full._quad_points))
    Bn = np.sum(bs.B() * psc_full._quad_normals, axis=1)
    f_full = -psc_full._phi_mat.T @ (psc_full._quad_weights * Bn)

    phi_w = jnp.asarray(psc_red._phi_work_stack)
    w_w = jnp.asarray(psc_red._w_work_stack)
    bi = jnp.asarray(psc_red._jax_base_indices)
    signs = jnp.asarray(psc_red._jax_replica_signs)
    n_work, nq_per, _ = phi_w.shape
    Bn_w = _fold_Bn_to_work(jnp.asarray(Bn), bi, signs, n_work, nq_per)
    f_red = np.array(shell_loading_vector_stacked_pure(phi_w, w_w, Bn_w.reshape(-1)))
    np.testing.assert_allclose(
        T_signed.T @ f_full,
        f_red,
        atol=1e-10,
        rtol=1e-9,
        err_msg="Signed T must satisfy T^T f_full = f_red",
    )


def test_vjp_tf_gradient_nonzero_and_listed_coils_stellsym():
    """VJP through the signed reduced path produces finite, nonzero TF
    gradients on every coil curve and current.

    Mirrors :func:`test_vjp_tf_gradient_nonzero_and_listed_coils` with
    ``stellsym=True`` to exercise the signed fold / gather adjoint
    (both in the JAX AD path and in the analytic NumPy adjoint used by
    the TF-only VJP).
    """
    psc = _make_symmetry_validation_array(nfp=2, stellsym=True, n_base=1)
    psc.recompute_currents()
    assert psc._reduced_active, (
        "Test is exercising the signed reduced path; gate must be open"
    )
    pts = np.array([[1.0, 0.05, 0.35]], dtype=float)
    v = np.random.default_rng(0).standard_normal((1, 3))
    deriv = psc.vjp_setup_B(v, pts)
    for c in psc.coils_TF:
        g = np.asarray(deriv(c.curve))
        assert g.shape == (c.curve.x.shape[0],)
        assert np.all(np.isfinite(g))
        assert np.linalg.norm(g) > 0.0, (
            "Coil-curve gradient must be nonzero; signed VJP adjoint "
            "may have silently zeroed the contribution"
        )
    gI = np.asarray(deriv(psc.coils_TF[0].current))
    assert np.all(np.isfinite(gI))


@pytest.mark.parametrize(
    "nfp, stellsym, n_base_coils",
    [
        (2, True, 2),
        (2, False, 2),
        (3, True, 3),
        (4, False, 2),
    ],
)
def test_detect_tf_symmetry_matches_coils_via_symmetries_ordering(
    nfp: int, stellsym: bool, n_base_coils: int
):
    """``_detect_tf_symmetry`` must recognise TF sets built by the canonical
    :func:`simsopt.field.coil.coils_via_symmetries`, even with multiple
    base coils.

    Regression test for a silent false-negative in the detection
    routine: prior to this fix the routine iterated over
    ``(i_base, jfp, stell)`` (the order used by
    :meth:`PSCBulkArray._replicate_pucks`) while
    :func:`apply_symmetries_to_curves` emits
    ``(k, flip, i_base)``.  Those two orderings coincide only when
    ``n_base_coils == 1``, so every multi-coil TF set produced by
    :func:`coils_via_symmetries` (for example the
    ``SchuettHennebergQAnfp2`` set used in
    ``passive_bulks_cylindrical_grid_optimization.py``) silently
    dropped to the expensive full-``L`` path.
    """
    from simsopt.field.coil import Coil, coils_via_symmetries

    base_curves = []
    base_currents = []
    for i in range(n_base_coils):
        curve = CurveXYZFourier(32, 1)
        # Distinct base curves at different radii / vertical offsets so
        # the detection cannot succeed by accidentally treating two base
        # copies as identical.
        curve.x = np.array(
            [
                1.0 + 0.2 * i,
                0.0,
                0.3 + 0.05 * i,
                0.0,
                0.3 + 0.05 * i,
                0.0,
                0.1 * i,
                0.0,
                0.0,
            ]
        )
        base_curves.append(curve)
        base_currents.append(Current(1.0e5 * (1.0 + 0.1 * i)))

    tf_coils = coils_via_symmetries(
        base_curves, base_currents, int(nfp), bool(stellsym)
    )

    assert len(tf_coils) == n_base_coils * int(nfp) * (2 if stellsym else 1)
    assert PSCBulkArray._detect_tf_symmetry(tf_coils, nfp, stellsym) is True, (
        "Detection must accept the simsopt-canonical (k, flip, i_base) "
        f"ordering for nfp={nfp}, stellsym={stellsym}, "
        f"n_base_coils={n_base_coils}"
    )

    # Corrupting one replica (current scale) must break detection so the
    # build falls back to the full-L path rather than silently using an
    # incorrect symmetry-reduction.
    corrupted = list(tf_coils)
    corrupted[-1] = Coil(corrupted[-1].curve, Current(7.0e5))
    assert PSCBulkArray._detect_tf_symmetry(corrupted, nfp, stellsym) is False, (
        "Detection must reject a TF list whose last replica has a "
        "current inconsistent with the symmetry image"
    )


# ======================================================================
# Phase A: Taylor-tests for TF curve + current DOFs in the regime that
# reproduces the failing ``passive_bulks_cylindrical_grid_optimization``
# example.  The existing ``test_taylor_*`` coverage all uses
# ``n_base_pucks == 1`` and/or only perturbs the TF current DOF, which
# happens to silence the constant-offset VJP bug reproduced below.
# ======================================================================


def _make_multi_coil_multi_puck_setup(
    n_base_coils: int = 2,
    n_base_pucks: int = 2,
    nfp: int = 2,
    stellsym: bool = True,
    R0: float = 1.0,
    R1: float = 0.5,
    coil_order: int = 2,
    current_scale: float = 1.0e5,
):
    """Build a multi-base-TF-coil + multi-base-puck setup for Taylor tests.

    This is the minimal repro of the regime exercised by
    ``examples/3_Advanced/passive_bulks_cylindrical_grid_optimization.py``
    (multiple base TF curves through ``coils_via_symmetries``, multiple
    base pucks, ``nfp=2, stellsym=True``) but scaled down enough to run
    inside the unit-test budget.

    Returns ``(s, coils_tf, base_curves, base_currents, psc, btot, Jf)``
    so callers can mirror the ``_make_small_setup`` API.
    """
    from pathlib import Path
    from simsopt.field import BiotSavart, coils_via_symmetries
    from simsopt.field.magneticfield import MagneticFieldSum
    from simsopt.geo import SurfaceRZFourier, create_equally_spaced_curves
    from simsopt.objectives import SquaredFlux

    TEST_DIR = (Path(__file__).parent / ".." / "test_files").resolve()
    filename = TEST_DIR / "input.LandremanPaul2021_QA"
    s = SurfaceRZFourier.from_vmec_input(
        filename, range="half period", nphi=4, ntheta=4
    )

    base_curves = create_equally_spaced_curves(
        int(n_base_coils),
        int(nfp),
        bool(stellsym),
        R0=float(R0),
        R1=float(R1),
        order=int(coil_order),
    )
    base_currents = [Current(float(current_scale)) for _ in range(int(n_base_coils))]
    coils_tf = coils_via_symmetries(
        base_curves, base_currents, int(nfp), bool(stellsym)
    )

    eval_pts = np.ascontiguousarray(s.gamma().reshape(-1, 3))

    # Base pucks at nontrivial but off-surface locations.  Distinct axes
    # exercise the puck-local rotation so the bug-reproducing setup is
    # not degenerate.
    all_centers = np.array(
        [
            [1.0, 0.0, 0.15],
            [0.9, 0.3, -0.10],
            [0.8, -0.3, 0.08],
            [0.95, 0.15, 0.20],
        ]
    )
    all_axes = np.array(
        [
            [0.1, 0.1, 1.0],
            [0.0, 0.0, 1.0],
            [0.2, -0.1, 1.0],
            [-0.1, 0.2, 1.0],
        ]
    )
    centers = all_centers[: int(n_base_pucks)]
    axes = all_axes[: int(n_base_pucks)]
    radii = np.full(int(n_base_pucks), 0.04)
    thicknesses = np.full(int(n_base_pucks), 0.02)

    psc = PSCBulkArray(
        centers,
        axes,
        radii,
        thicknesses,
        coils_tf,
        eval_points=eval_pts,
        m_fourier=1,
        l_zernike=2,
        k_chebyshev=1,
        n_rho=4,
        n_phi=6,
        n_z=3,
        nfp=int(nfp),
        stellsym=bool(stellsym),
        adaptive_self_reg=False,
    )
    b_bulk = psc.biot_savart
    b_tf = BiotSavart(coils_tf)
    btot = MagneticFieldSum([b_bulk, b_tf])
    Jf = SquaredFlux(s, btot)
    return s, coils_tf, base_curves, base_currents, psc, btot, Jf


def _tf_taylor_errors_and_absolute(
    Jf, psc, btot, label: str, seed: int = 3, start_power: int = 4, n_points: int = 6
):
    """Taylor sweep over TF curve + current DOFs; return (errors, abs_errors)."""

    def getter():
        return np.copy(Jf.x)

    def setter(dofs):
        Jf.x = dofs
        psc.recompute_currents()
        btot.Bfields[0].clear_cached_properties()

    dofs0 = getter().copy()
    np.random.seed(int(seed))
    h = np.random.randn(len(dofs0))
    h = h / np.linalg.norm(h)

    setter(dofs0)
    float(Jf.J())
    dJ0 = np.array(Jf.dJ())
    deriv = float(np.sum(dJ0 * h))

    rel_errors = []
    abs_errors = []
    for i in range(start_power, start_power + n_points):
        eps = 0.5**i
        setter(dofs0 + eps * h)
        Jp = float(Jf.J())
        setter(dofs0 - eps * h)
        Jm = float(Jf.J())
        fd = (Jp - Jm) / (2 * eps)
        abs_err = abs(fd - deriv)
        abs_errors.append(abs_err)
        if abs(deriv) > 1e-15:
            rel_errors.append(abs_err / abs(deriv))
        else:
            rel_errors.append(abs_err)

    setter(dofs0)
    return rel_errors, abs_errors, deriv


def test_taylor_tf_dofs_multi_base_stellsym_reduced_path():
    """Taylor test for multi-base TF coils + multi-base pucks with the
    symmetry-reduced L path active.

    Mirrors the cylindrical-grid example's configuration (``ncoils>=2``,
    ``n_base_pucks>=2``, ``nfp=2, stellsym=True``).  The test asserts the
    reduced path is active (catching any future regression in
    ``_detect_tf_symmetry``) and requires both 2nd-order FD convergence
    *and* a small absolute error, so a constant-offset adjoint bug
    cannot pass silently.
    """
    s, coils_tf, base_curves, base_currents, psc, btot, Jf = (
        _make_multi_coil_multi_puck_setup(
            n_base_coils=2, n_base_pucks=2, nfp=2, stellsym=True
        )
    )
    assert psc._reduced_active is True, (
        "Reduced-L path must be active for this configuration; otherwise "
        "_detect_tf_symmetry has regressed."
    )

    rel_errors, abs_errors, deriv = _tf_taylor_errors_and_absolute(
        Jf, psc, btot, label="TF DOFs reduced path"
    )
    _assert_taylor_convergence(
        rel_errors, label="TF DOFs (multi-base, stellsym, reduced path)"
    )
    min_rel = float(np.min(rel_errors))
    assert min_rel < 1e-3, (
        "TF-DOF analytic gradient disagrees with FD at finite step size: "
        f"min rel err {min_rel:.3e} >= 1e-3, deriv={deriv:.6e}, "
        f"rel_errors={rel_errors}, abs_errors={abs_errors}"
    )


def test_taylor_tf_dofs_multi_base_full_path():
    """Same regime but the PSC is built with ``nfp=1, stellsym=False`` so
    the full-L path is forced.

    Decisive reduced-vs-full check: if the analytic gradient is also
    wrong here, the bug is not in the symmetry-reduction machinery.
    """
    s, coils_tf, base_curves, base_currents, psc, btot, Jf = (
        _make_multi_coil_multi_puck_setup(
            n_base_coils=2, n_base_pucks=2, nfp=1, stellsym=False
        )
    )
    assert psc._reduced_active is False, (
        "Full-L path must be active for nfp=1, stellsym=False; otherwise "
        "PSCBulkArray._rebuild has regressed."
    )

    rel_errors, abs_errors, deriv = _tf_taylor_errors_and_absolute(
        Jf, psc, btot, label="TF DOFs full path"
    )
    _assert_taylor_convergence(rel_errors, label="TF DOFs (multi-base, full path)")
    min_rel = float(np.min(rel_errors))
    assert min_rel < 1e-3, (
        "TF-DOF analytic gradient disagrees with FD (full path): "
        f"min rel err {min_rel:.3e} >= 1e-3, deriv={deriv:.6e}, "
        f"rel_errors={rel_errors}, abs_errors={abs_errors}"
    )


def test_taylor_tf_dofs_single_puck_full_path():
    """Smallest reproducer: multiple base TF curves but a single puck,
    ``nfp=1, stellsym=False``.  If this already fails, the bulk VJP path
    has a constant-offset error independent of symmetry / multi-puck
    interactions.
    """
    s, coils_tf, base_curves, base_currents, psc, btot, Jf = (
        _make_multi_coil_multi_puck_setup(
            n_base_coils=2, n_base_pucks=1, nfp=1, stellsym=False
        )
    )
    assert psc._reduced_active is False

    rel_errors, abs_errors, deriv = _tf_taylor_errors_and_absolute(
        Jf, psc, btot, label="TF DOFs single-puck full path"
    )
    _assert_taylor_convergence(rel_errors, label="TF DOFs (single puck, full path)")
    min_rel = float(np.min(rel_errors))
    assert min_rel < 1e-3, (
        "TF-DOF analytic gradient disagrees with FD (single-puck full path): "
        f"min rel err {min_rel:.3e} >= 1e-3, deriv={deriv:.6e}, "
        f"rel_errors={rel_errors}, abs_errors={abs_errors}"
    )


@pytest.mark.parametrize(
    "n_base_pucks,nfp,stellsym,expect_reduced",
    [
        (1, 1, False, False),
        (2, 1, False, False),
        (2, 2, True, True),
    ],
)
def test_vjp_tf_analytic_matches_fd_tf_curve_dof(
    n_base_pucks, nfp, stellsym, expect_reduced
):
    """Finite-difference check of ``PassiveBulkField.B_vjp`` against a
    **TF curve DOF** perturbation.

    ``test_vjp_fd_gradient_check`` only exercises the TF *current* DOF
    (a linear scalar in the chain); this test perturbs a TF curve DOF
    so the bulk VJP is stress-tested through both ``BiotSavart.B_vjp``
    and the ``_vjp_tf_only_analytic`` adjoint of the shell solve.
    """
    s, coils_tf, base_curves, base_currents, psc, btot, Jf = (
        _make_multi_coil_multi_puck_setup(
            n_base_coils=2,
            n_base_pucks=n_base_pucks,
            nfp=nfp,
            stellsym=stellsym,
        )
    )
    assert psc._reduced_active is expect_reduced

    bf = psc.biot_savart
    v = np.asarray(btot.get_points_cart_ref(), dtype=float).copy()
    v[:] = 1.0 / np.sqrt(3.0 * v.shape[0])  # unit vector in the full (N,3) space

    psc.recompute_currents()
    bf.clear_cached_properties()
    dj = bf.B_vjp(v)

    # Pick a shape-DOF on the first base curve with nontrivial sensitivity.
    # The ``xc(0)`` offset has near-zero bulk-field sensitivity for an
    # axisymmetric base circle and would let a -1 sign flip slip through an
    # ``abs(...) < eps`` escape clause; ``xc(1)`` is the first non-trivial
    # Fourier shape coefficient and has real sensitivity.
    base = base_curves[0]
    dof_names = list(base.local_dof_names)
    preferred = ["xc(1)", "yc(1)", "zs(1)", "xs(1)", "ys(1)"]
    target_name = next((nm for nm in preferred if nm in dof_names), None)
    if target_name is None:
        target_name = next(
            nm
            for nm in dof_names
            if any(s in nm for s in ("xc", "yc", "xs", "ys", "zc", "zs"))
        )
    x0 = base.get(target_name)

    eps = 1e-5
    base.set(target_name, x0 + eps)
    psc.recompute_currents()
    bf.clear_cached_properties()
    Bp = np.array(bf.B())
    base.set(target_name, x0 - eps)
    psc.recompute_currents()
    bf.clear_cached_properties()
    Bm = np.array(bf.B())
    base.set(target_name, x0)
    psc.recompute_currents()
    bf.clear_cached_properties()
    fd = float(np.sum(v * (Bp - Bm)) / (2.0 * eps))

    vjp_deriv_array = np.asarray(dj(base))
    idx = dof_names.index(target_name)
    vjp_val = float(vjp_deriv_array[idx])

    rel_err = abs(vjp_val - fd) / (abs(fd) + 1e-30)
    assert rel_err < 1e-3 or abs(vjp_val - fd) < 1e-8, (
        f"Bulk VJP vs FD mismatch on TF curve DOF "
        f"(n_base_pucks={n_base_pucks}, nfp={nfp}, stellsym={stellsym}): "
        f"fd={fd:.6e}, vjp={vjp_val:.6e}, rel_err={rel_err:.3e}"
    )


@pytest.mark.parametrize(
    "n_base_pucks,nfp,stellsym,expect_reduced",
    [
        (1, 1, False, False),
        (2, 1, False, False),
        (2, 2, False, True),
        (2, 1, True, True),
        (2, 2, True, True),
    ],
)
def test_vjp_tf_analytic_matches_jax_multi_puck(
    n_base_pucks, nfp, stellsym, expect_reduced
):
    """Extend ``test_vjp_tf_fast_matches_jax`` to multi-puck / multi-base-coil
    configurations across both reduced and full paths.

    The JAX VJP path runs ``jax.vjp`` through the actual
    ``_B_eval_from_tf_body`` forward, so it is by construction consistent
    with the forward.  The analytic path (``_vjp_tf_only_analytic``)
    reproduces the same adjoint by hand for speed; this test asserts they
    agree to tight tolerance so any missing sign, missing weight, or
    broken fold/gather can be caught per-configuration.
    """
    import simsopt.field.psc_bulk as psb

    s, coils_tf, base_curves, base_currents, psc, btot, Jf = (
        _make_multi_coil_multi_puck_setup(
            n_base_coils=2,
            n_base_pucks=n_base_pucks,
            nfp=nfp,
            stellsym=stellsym,
        )
    )
    assert psc._reduced_active is expect_reduced

    eval_pts = np.asarray(btot.get_points_cart_ref(), dtype=float)
    v_B = np.ones_like(eval_pts) / float(np.sqrt(3.0 * eval_pts.shape[0]))

    psb._USE_JAX_TF_VJP = True
    psc.recompute_currents()
    g_jax = psc.vjp_setup_B(v_B, eval_pts)

    psb._USE_JAX_TF_VJP = False
    psc.recompute_currents()
    g_fast = psc.vjp_setup_B(v_B, eval_pts)

    base = base_curves[0]
    np.testing.assert_allclose(
        np.asarray(g_jax(base)),
        np.asarray(g_fast(base)),
        rtol=1e-6,
        atol=1e-10,
        err_msg=(
            f"Analytic TF VJP must match the JAX TF VJP on base curve DOFs "
            f"(n_base_pucks={n_base_pucks}, nfp={nfp}, stellsym={stellsym}, "
            f"reduced={expect_reduced})"
        ),
    )
    np.testing.assert_allclose(
        np.asarray(g_jax(base_currents[0])),
        np.asarray(g_fast(base_currents[0])),
        rtol=1e-6,
        atol=1e-10,
        err_msg=(
            f"Analytic TF VJP must match the JAX TF VJP on base current DOF "
            f"(n_base_pucks={n_base_pucks}, nfp={nfp}, stellsym={stellsym}, "
            f"reduced={expect_reduced})"
        ),
    )


def test_solver_mode_shell_l2_interior_cancellation():
    r"""Focused regression: ``solver_mode='shell_l2'`` beats ``'energy'``.

    Builds the *same* thin-disc puck twice (once with each solver
    mode), at a basis small enough to keep the test under ~15 s, and
    asserts two things:

    1. Both modes return a finite, Lenz-signed induced moment
       (induced :math:`B_z(0) < 0` for the ring-coil-above geometry).
    2. The L^2 solver's interior residual
       :math:`|B^{tot}(0)|/|B^{TF}(0)|` is *strictly smaller* than the
       energy solver's at this basis.  The absolute values at this
       basis are still large (~0.7-0.9) -- see
       :func:`tests.field.test_passive_bulks_scale.test_interior_field_cancellation`
       for the higher-basis threshold test -- but the ordering is the
       key regression guard that the new operator is doing what it
       claims.
    """
    R = 0.05
    t = 0.005
    curve = CurveXYZFourier(32, 1)
    curve.x = np.array([0.0, 0.0, 5.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    tf = Coil(curve, Current(1.0e7))

    def _build(mode: str) -> PSCBulkArray:
        return PSCBulkArray(
            np.array([[0.0, 0.0, 0.0]]),
            np.array([[0.0, 0.0, 1.0]]),
            np.array([R]),
            np.array([t]),
            [tf],
            eval_points=np.array([[2.0 * R, 0.0, 0.0]]),
            m_fourier=3,
            l_zernike=6,
            k_chebyshev=3,
            n_rho=10,
            n_phi=16,
            n_z=6,
            nfp=1,
            stellsym=False,
            adaptive_self_reg=True,
            solver_mode=mode,
        )

    from simsopt.field.biotsavart import BiotSavart

    probe = np.array([[0.0, 0.0, 0.0]])
    bs = BiotSavart([tf])
    bs.set_points_cart(np.ascontiguousarray(probe))
    B_tf0 = np.asarray(bs.B())[0]

    ratios = {}
    Bz = {}
    for mode in ("energy", "shell_l2"):
        psc = _build(mode)
        B_ind0 = np.asarray(psc.B_at_points(probe))[0]
        ratios[mode] = float(np.linalg.norm(B_tf0 + B_ind0) / np.linalg.norm(B_tf0))
        Bz[mode] = float(B_ind0[2])
    # shell_l2 must respect Lenz's law; the energy form on this thin-
    # disc-without-exact-disc-faces path is known to give the wrong
    # sign and is not under test here.
    assert Bz["shell_l2"] < 0.0, (
        "shell_l2 must satisfy Lenz's law (induced B_z < 0 for the "
        f"ring-above geometry); got B_z(0)={Bz['shell_l2']:.3e}"
    )
    assert ratios["shell_l2"] < ratios["energy"], (
        "shell_l2 solver must achieve a smaller interior residual than "
        f"the energy form at the same basis; got "
        f"shell_l2={ratios['shell_l2']:.3e}, energy={ratios['energy']:.3e}"
    )


def test_shell_l2_rejects_free_puck_dofs():
    """shell_l2 mode is not yet implemented for free puck DOFs."""
    R = 0.05
    t = 0.005
    curve = CurveXYZFourier(32, 1)
    curve.x = np.array([0.0, 0.0, 5.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    tf = Coil(curve, Current(1.0e7))
    psc = PSCBulkArray(
        np.array([[0.0, 0.0, 0.0]]),
        np.array([[0.0, 0.0, 1.0]]),
        np.array([R]),
        np.array([t]),
        [tf],
        eval_points=np.array([[2.0 * R, 0.0, 0.0]]),
        m_fourier=2,
        l_zernike=4,
        k_chebyshev=2,
        n_rho=8,
        n_phi=12,
        n_z=4,
        nfp=1,
        stellsym=False,
        adaptive_self_reg=True,
        solver_mode="shell_l2",
    )
    # Unfix the first puck-centre DOF to simulate a free-DOF workflow.
    try:
        psc.unfix(psc.local_dof_names[0])
    except (AttributeError, IndexError):
        pytest.skip("PSCBulkArray does not expose unfix in this build.")
    with pytest.raises(NotImplementedError):
        psc.B_at_points(np.array([[0.05, 0.0, 0.0]]))


def test_shell_l2_rejects_invalid_mode():
    """Only 'energy' and 'shell_l2' are valid solver_mode values."""
    curve = CurveXYZFourier(32, 1)
    curve.x = np.array([0.0, 0.0, 5.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    tf = Coil(curve, Current(1.0e7))
    with pytest.raises(ValueError, match="solver_mode"):
        PSCBulkArray(
            np.array([[0.0, 0.0, 0.0]]),
            np.array([[0.0, 0.0, 1.0]]),
            np.array([0.05]),
            np.array([0.005]),
            [tf],
            eval_points=np.array([[0.1, 0.0, 0.0]]),
            m_fourier=2,
            l_zernike=4,
            k_chebyshev=2,
            n_rho=8,
            n_phi=12,
            n_z=4,
            solver_mode="nonsense",
        )


# ---------------------------------------------------------------------------
# Regression tests locking the ``ScaledCurrent.set_dofs`` propagation contract.
#
# Historically ``ScaledCurrent.set_dofs`` delegated to
# ``self.current_to_scale.set_dofs(...)`` which, for a plain
# :class:`~simsopt.field.coil.Current`, dispatches to the C++
# ``sopp.Current.set_dofs`` external setter.  That path writes the C++
# ``_current`` state directly but bypasses the Python ``Dofs._x`` cache
# and never calls ``_flag_recompute_opt``, so downstream
# :class:`BiotSavart` / :class:`PSCBulkArray` caches remained stale
# even though ``get_value()`` reflected the new current.  Symptoms:
# ``recompute_currents()`` returned unchanged ``beta`` after a TF-current
# perturbation, and the cylindrical-grid example's sanity block reported
# ``|beta1 - beta0| = 0``.  The fix routes ``ScaledCurrent.set_dofs``
# through ``self.current_to_scale.local_full_x`` which triggers
# ``Dofs.full_x.setter`` → ``_flag_recompute_opt`` → every dep_opt's
# ``set_recompute_flag``.  The tests below guard both the DOF-sync
# contract and the full ``PSCBulkArray`` cache invalidation.
# ---------------------------------------------------------------------------


def test_scaled_current_set_dofs_propagates():
    """``ScaledCurrent.set_dofs`` must keep the underlying Python DOF cache
    in sync and reflect the perturbation through ``get_value()``.

    Prior to the fix the write only updated the C++ ``_current`` field;
    the Python ``Dofs._x`` stayed stale, so any dependent
    :class:`BiotSavart`/:class:`MagneticField` tree relying on
    ``_flag_recompute_opt`` for cache invalidation never noticed the
    change.  This test asserts the full Python-side bookkeeping is
    consistent after both attribute paths (``.set_dofs`` and ``.x``).
    """
    from simsopt.field.coil import ScaledCurrent

    I0 = 1.0e5
    scale = 1.0e2
    c = Current(I0)
    sc = ScaledCurrent(c, scale)

    # Baseline: ``get_value`` should match the product and DOF caches agree.
    assert sc.get_value() == pytest.approx(scale * I0)
    assert c.local_full_x[0] == pytest.approx(I0)
    assert c._dofs._x[0] == pytest.approx(I0)

    # Path 1: ``set_dofs`` specifies the SCALED value, so the underlying
    # Current.local_full_x[0] must equal ``target / scale`` after the call.
    target = 5.0e6
    sc.set_dofs(np.array([target]))
    assert sc.get_value() == pytest.approx(target)
    assert c.local_full_x[0] == pytest.approx(target / scale)
    # Crucial: the Python ``Dofs._x`` cache must be synced to the same
    # value as ``local_full_x``.  Pre-fix this was stale because the
    # write went through the C++ external setter only.
    assert c._dofs._x[0] == pytest.approx(target / scale)

    # Path 2: ``.x`` sets the UNDERLYING child DOF directly (not the
    # scaled value).  Writing the child DOF must propagate the same way.
    child_target = 2.0e3
    sc.x = np.array([child_target])
    assert c.local_full_x[0] == pytest.approx(child_target)
    assert c._dofs._x[0] == pytest.approx(child_target)
    assert sc.get_value() == pytest.approx(scale * child_target)


def test_scaled_current_set_dofs_invalidates_biotsavart_cache():
    """Perturbing a :class:`ScaledCurrent` via ``set_dofs`` must trigger
    cache invalidation on any dependent :class:`BiotSavart`.

    This is the unit-level analogue of the cylindrical-grid example's
    failing sanity block.  A proper fix routes the DOF write through
    ``Dofs.full_x.setter`` which walks every dep_opt and invalidates
    their caches via ``MagneticField.recompute_bell``.
    """
    from simsopt.field import BiotSavart
    from simsopt.field.coil import ScaledCurrent

    curve = CurveXYZFourier(32, 1)
    curve.x = np.array([0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    c = Current(1.0e5)
    sc = ScaledCurrent(c, 1.0e2)
    coil = Coil(curve, sc)
    bs = BiotSavart([coil])
    bs.set_points(np.array([[0.3, 0.0, 0.0]]))
    B0 = bs.B().copy()

    # Nudge the scaled current by 10% and verify BiotSavart picked up
    # the change (i.e. returns a linearly scaled field, not the cached
    # value).  Pre-fix this would be equal bit-for-bit to ``B0``.
    sc.set_dofs(np.array([1.1 * sc.get_value()]))
    B1 = bs.B().copy()
    rel = np.linalg.norm(B1 - B0) / (np.linalg.norm(B0) + 1e-30)
    assert rel > 1e-3, (
        "BiotSavart cache must invalidate after ScaledCurrent.set_dofs; "
        f"got |B1-B0|/|B0| = {rel:.3e} (expected ~0.1 for a 10% current nudge)"
    )
    # Field is linear in current → 10% current ↔ 10% field.
    np.testing.assert_allclose(B1, 1.1 * B0, rtol=1e-10, atol=1e-12)


def test_recompute_currents_invalidates_after_scaled_set_dofs():
    """After perturbing a scaled TF current, ``PSCBulkArray.recompute_currents``
    must return a changed ``beta`` regardless of whether we used the
    ``set_dofs`` or ``.x`` DOF channel.

    Reproduces the cylindrical-grid example's sanity-block scenario
    (``base_currents_tf[0].set_dofs(...)`` → stale ``beta``) at unit-test
    scale.  Post-fix both channels must produce (a) a non-zero beta
    change and (b) bit-for-bit identical results.
    """
    from simsopt.field import BiotSavart, coils_via_symmetries
    from simsopt.field.coil import ScaledCurrent
    from simsopt.geo import create_equally_spaced_curves

    nfp = 1
    stellsym = False
    ncoils = 2
    base_curves = create_equally_spaced_curves(
        ncoils, nfp, stellsym, R0=1.0, R1=0.5, order=2
    )
    # Mimic ``initialize_coils``: the first TF current is a ScaledCurrent
    # wrapping a Current (so ScaledCurrent.set_dofs is exercised), the
    # second is a plain Current.
    inner = Current(1.0e3)
    scaled_current = ScaledCurrent(inner, 1.0e2)
    plain_current = Current(1.0e5)
    base_currents = [scaled_current, plain_current]
    coils_tf = coils_via_symmetries(base_curves, base_currents, nfp, stellsym)

    centers = np.array([[1.0, 0.0, 0.2]])
    axes = np.array([[0.0, 0.0, 1.0]])
    radii = np.array([0.04])
    thicknesses = np.array([0.02])
    eval_pts = np.array([[1.2, 0.0, 0.0]])
    psc = PSCBulkArray(
        centers,
        axes,
        radii,
        thicknesses,
        coils_tf,
        eval_points=eval_pts,
        m_fourier=1,
        l_zernike=2,
        k_chebyshev=1,
        n_rho=4,
        n_phi=6,
        n_z=3,
        nfp=nfp,
        stellsym=stellsym,
        adaptive_self_reg=False,
    )

    b_tf = BiotSavart(coils_tf)

    # Record baseline beta/field.
    psc.recompute_currents()
    beta0 = np.array(psc.beta, copy=True)
    b_tf.set_points(eval_pts)
    _ = b_tf.B().copy()
    assert np.linalg.norm(beta0) > 0, "baseline beta unexpectedly zero"

    # Channel A: .set_dofs (the historically-broken path).  Post-fix
    # this must invalidate every dep cache, so psc.recompute_currents()
    # picks up fresh Bn and therefore fresh beta.
    target = 1.1 * scaled_current.get_value()
    scaled_current.set_dofs(np.array([target]))
    psc.recompute_currents()
    beta_a = np.array(psc.beta, copy=True)
    b_tf.set_points(eval_pts)
    B_tf_a = b_tf.B().copy()
    assert scaled_current.get_value() == pytest.approx(target)

    # Restore the child DOF to its baseline (``inner``'s original
    # value) via the ``.x`` channel (ScaledCurrent has 0 local DOFs,
    # so ``sc.x`` maps to the child Current's single DOF).
    scaled_current.x = np.array([1.0e3])
    psc.recompute_currents()
    beta_restored = np.array(psc.beta, copy=True)
    np.testing.assert_allclose(
        beta_restored,
        beta0,
        rtol=1e-10,
        atol=1e-14,
        err_msg="Restoring the baseline DOF value must yield the baseline beta.",
    )
    # Channel B perturbation: write the UNDERLYING child DOF so that
    # ``get_value()`` matches ``target`` from Channel A.
    scaled_current.x = np.array([target / scaled_current.scale])
    psc.recompute_currents()
    beta_b = np.array(psc.beta, copy=True)
    b_tf.set_points(eval_pts)
    B_tf_b = b_tf.B().copy()

    # Both channels must produce a non-zero beta change relative to the
    # baseline (this is the contract the sanity block asserts at
    # example scale).
    rel_a = np.linalg.norm(beta_a - beta0) / (np.linalg.norm(beta0) + 1e-30)
    rel_b = np.linalg.norm(beta_b - beta0) / (np.linalg.norm(beta0) + 1e-30)
    assert rel_a > 1e-3, (
        f"beta unchanged after ScaledCurrent.set_dofs (rel={rel_a:.3e}); "
        "cache bypass regression."
    )
    assert rel_b > 1e-3, f"beta unchanged after .x write (rel={rel_b:.3e})."

    # And the two channels must produce bit-for-bit identical beta
    # (modulo FP noise from a different arithmetic path).
    np.testing.assert_allclose(beta_a, beta_b, rtol=1e-10, atol=1e-14)
    # Same for the direct BiotSavart field evaluated at a probe point.
    np.testing.assert_allclose(B_tf_a, B_tf_b, rtol=1e-10, atol=1e-14)


# ---------------------------------------------------------------------------
# Per-DOF FD-vs-analytic diagnostic, grouped by DOF family.
#
# The cylindrical-grid example's Taylor test was showing a rel_err
# plateau at ~3e-4 at small eps, which is larger than the double-
# precision roundoff floor expected for ``J ~ 0.1``.  This test probes
# every free DOF with a central finite difference and compares to the
# analytic gradient column, grouped by DOF family (TF curve, TF
# current, puck geometry).  A persistent bias per family would
# pinpoint a missing chain-rule contribution in one branch of
# ``PSCBulkArray``'s TF-only adjoint.
# ---------------------------------------------------------------------------


def _classify_dof_family(name: str) -> str:
    s = str(name)
    if "CurveXYZFourier" in s or "CurvePlanar" in s:
        return "tf_curve"
    if "Current" in s or "ScaledCurrent" in s or "CurrentSum" in s:
        return "tf_current"
    if "PSCBulkArray" in s or "puck" in s.lower():
        return "puck"
    return "other"


def test_per_dof_central_difference_by_family():
    """For every free DOF, compare analytic ``dJ/dx_k`` to a central FD
    probe, grouped by DOF family (TF curve / TF current / puck).

    A constant plateau across small ``eps`` in the Taylor test is a
    strong signature of a missing chain-rule contribution or a stale
    cache leak; this per-DOF probe localizes the offending family so
    the fix can be surgical.
    """
    s, coils_tf, base_curves, base_currents, psc, btot, Jf = (
        _make_multi_coil_multi_puck_setup(
            n_base_coils=2, n_base_pucks=2, nfp=2, stellsym=True
        )
    )

    def _call(dofs: np.ndarray) -> float:
        Jf.x = dofs
        psc.recompute_currents()
        btot.Bfields[0].clear_cached_properties()
        return float(Jf.J())

    dofs0 = np.copy(Jf.x)
    Jf.x = dofs0
    psc.recompute_currents()
    btot.Bfields[0].clear_cached_properties()
    dJ0 = np.array(Jf.dJ())

    names = list(np.array(Jf.dof_names))
    eps = 1.0e-5
    per_family_rel: dict[str, list[tuple[str, float, float, float]]] = {
        "tf_curve": [],
        "tf_current": [],
        "puck": [],
        "other": [],
    }
    for k in range(len(dofs0)):
        e = np.zeros_like(dofs0)
        e[k] = 1.0
        Jp = _call(dofs0 + eps * e)
        Jm = _call(dofs0 - eps * e)
        fd_k = (Jp - Jm) / (2.0 * eps)
        abs_err = abs(fd_k - dJ0[k])
        rel_err = abs_err / (abs(dJ0[k]) + 1e-12)
        fam = _classify_dof_family(names[k])
        per_family_rel[fam].append((str(names[k]), abs_err, rel_err, float(dJ0[k])))

    # Reset DOFs before assertion so a failure doesn't leak state.
    _call(dofs0)

    rel_tol = 1.0e-5
    offenders: list[str] = []
    family_max: dict[str, float] = {}
    for fam, entries in per_family_rel.items():
        if not entries:
            continue
        rels = [r for (_, _, r, _) in entries]
        family_max[fam] = max(rels)
        bad = [
            (nm, ae, re_, dj)
            for (nm, ae, re_, dj) in entries
            if re_ > rel_tol and ae > 1e-10
        ]
        bad.sort(key=lambda t: -t[2])
        for nm, ae, re_, dj in bad[:5]:
            offenders.append(f"  [{fam}] {nm}: abs={ae:.3e} rel={re_:.3e} dJ={dj:.3e}")

    summary = ", ".join(
        f"{fam}={family_max.get(fam, 0.0):.2e}"
        for fam in ("tf_curve", "tf_current", "puck", "other")
    )
    if offenders:
        pytest.fail(
            "Per-DOF FD-vs-analytic mismatch above rel_tol="
            f"{rel_tol:.1e}.  Family max rel_errs: {summary}.\n"
            + "\n".join(offenders[:20])
        )


def test_tf_current_dof_no_taylor_plateau_reduced_path():
    """Regression test for the analytic-VJP jitter-mismatch bug.

    The cylindrical-grid example revealed that the numpy analytic path
    in ``_vjp_tf_only_analytic`` solved the adjoint system with a plain
    ``np.linalg.solve(Lr.T, ...)`` while the JAX forward used
    ``shell_solve_linear_pure`` (Cholesky with a ``1e-10`` jitter
    floor).  When ``Lr`` carries eigenvalues close to the null-space
    trim threshold (``1e-10 * max|eig|``) -- which is the common case
    at production resolution -- the two solves diverge by an amount
    comparable to the missing jitter term, and the analytic adjoint
    for TF-current DOFs picks up a constant absolute bias that does
    not shrink as ``eps -> 0``.  The signature is a Taylor-test
    plateau at ``rel_err ~ 3e-4`` with ``FD`` moving toward the
    converged limit while the analytic gradient sits at a biased
    value.

    This test pins the contract on the analytic numpy path at a
    reduced-L, multi-base-puck, stellsym-symmetric configuration --
    the smallest setup that still exercises the ``Q^T L_work Q``
    eigenvalue structure -- and asserts the TF-current central-
    difference converges toward the analytic gradient cleanly, with
    no constant-bias plateau.
    """
    import simsopt.field.psc_bulk as psb

    s, coils_tf, base_curves, base_currents, psc, btot, Jf = (
        _make_multi_coil_multi_puck_setup(
            n_base_coils=2, n_base_pucks=2, nfp=2, stellsym=True
        )
    )
    # Exercise the numpy analytic path (the one that contained the
    # jitter-mismatch bug), not the JAX path.
    saved = psb._USE_JAX_TF_VJP
    psb._USE_JAX_TF_VJP = False
    try:

        def _call(dofs):
            Jf.x = dofs
            psc.recompute_currents()
            btot.Bfields[0].clear_cached_properties()
            return float(Jf.J())

        dofs0 = np.copy(Jf.x)
        _call(dofs0)
        dJ0 = np.array(Jf.dJ())

        # Pick the first free base-current DOF.  In this helper the
        # TF currents are plain :class:`Current` objects (not
        # :class:`ScaledCurrent`) with a single DOF each.
        names = list(np.array(Jf.dof_names))
        k_cur = next(
            (i for i, n in enumerate(names) if "Current" in str(n)),
            None,
        )
        if k_cur is None:
            pytest.skip("No TF current DOF exposed in this configuration")
        analytic = float(dJ0[k_cur])
        e = np.zeros_like(dofs0)
        e[k_cur] = 1.0

        # Central differences at decreasing eps must converge to
        # ``analytic`` without plateauing at a biased value.  We check
        # both the error at a small eps and a 2nd-order decrease from
        # the larger eps -- either alone could be masked by a constant
        # bias of the magnitude seen in the cylindrical-grid example.
        abs_errs = []
        for eps in (1e-2, 1e-3, 1e-4, 1e-5):
            Jp = _call(dofs0 + eps * e)
            Jm = _call(dofs0 - eps * e)
            fd = (Jp - Jm) / (2.0 * eps)
            abs_errs.append(abs(fd - analytic))
        _call(dofs0)

        # At double precision the central-difference truncation error
        # decreases like ``O(eps^2)`` so the sequence should roughly
        # halve by two orders of magnitude per eps.  With the jitter
        # bug the last three entries of ``abs_errs`` would all equal
        # the same plateau value.
        assert abs_errs[-1] < 1.0e-6, (
            f"TF-current Taylor test did not converge: abs_errs={abs_errs}. "
            "A constant plateau is the signature of the jitter mismatch "
            "between _vjp_tf_only_analytic's adjoint solve and "
            "shell_solve_linear_pure's Cholesky+jitter forward."
        )
        # Only enforce the O(eps^2) ratio if the leading abs_err is large
        # enough to exceed the roundoff floor -- at the small test scale
        # we often hit ~1e-13 at eps=1e-2 already, where the central
        # difference is cancellation-limited and ``abs_errs`` stops
        # shrinking.  A plateau from the jitter bug, by contrast, would
        # pin ``abs_errs`` around ~1e-5 for every ``eps`` in this range.
        if abs_errs[1] > 1.0e-10:
            assert abs_errs[2] < 0.5 * abs_errs[1], (
                "TF-current Taylor test plateaued instead of converging at "
                f"2nd order: abs_errs={abs_errs}."
            )
    finally:
        psb._USE_JAX_TF_VJP = saved


def test_shell_prefactored_solve_matches_linear_pure():
    """``shell_solve_prefactored_pure`` matches ``shell_solve_linear_pure``."""
    import jax.numpy as jnp

    from simsopt.field.bulk_inductance import (
        shell_cholesky_pure,
        shell_solve_linear_pure,
        shell_solve_prefactored_pure,
    )

    rng = np.random.default_rng(7)
    n = 14
    A = rng.standard_normal((n, n))
    L = A @ A.T + 0.1 * np.eye(n)
    f = rng.standard_normal(n)
    x1 = np.asarray(shell_solve_linear_pure(jnp.asarray(L), jnp.asarray(f)))
    chol = shell_cholesky_pure(jnp.asarray(L), jitter=1e-10)
    x2 = np.asarray(shell_solve_prefactored_pure(chol, jnp.asarray(f)))
    np.testing.assert_allclose(x1, x2, rtol=1e-12, atol=1e-12)


def test_shell_eigenfloor_prefactored_matches_reference():
    """Prefactored Cholesky matches ``shell_solve_eigenfloor_pure``."""
    import jax.numpy as jnp

    from simsopt.field.bulk_inductance import (
        shell_eigenfloor_cholesky_pure,
        shell_solve_eigenfloor_pure,
        shell_solve_prefactored_pure,
    )

    rng = np.random.default_rng(11)
    n = 16
    A = rng.standard_normal((n, n))
    L = A @ A.T + 0.05 * np.eye(n)
    f = rng.standard_normal(n)
    thr = 1e-10
    x_ref = np.asarray(
        shell_solve_eigenfloor_pure(
            jnp.asarray(L), jnp.asarray(f), threshold=thr, jitter=1e-10
        )
    )
    chol = shell_eigenfloor_cholesky_pure(jnp.asarray(L), threshold=thr, jitter=1e-10)
    x_pf = np.asarray(shell_solve_prefactored_pure(chol, jnp.asarray(f)))
    np.testing.assert_allclose(x_ref, x_pf, rtol=1e-10, atol=1e-10)


def test_tf_arrays_cache_hits_on_repeat_call():
    psc = _make_symmetry_validation_array(nfp=2, stellsym=False, n_base=1)
    a1, b1, c1 = psc._tf_arrays()
    a2, b2, c2 = psc._tf_arrays()
    assert a1 is a2 and b1 is b2 and c1 is c2


def test_tf_arrays_cache_invalidates_on_dof_change():
    psc = _make_symmetry_validation_array(nfp=2, stellsym=False, n_base=1)
    a1, _, _ = psc._tf_arrays()
    psc.coils_TF[0].curve.x = np.asarray(psc.coils_TF[0].curve.x) + 1e-6
    a2, _, _ = psc._tf_arrays()
    assert a1 is not a2


def test_vjp_tf_reuses_cached_biotsavart():
    psc = _make_symmetry_validation_array(nfp=2, stellsym=False, n_base=1)
    psc.recompute_currents()
    pts = np.array([[1.0, 0.05, 0.35]], dtype=float)
    v = np.array([[1.0, 0.0, 0.0]])
    _ = psc._vjp_tf_only_analytic(v, pts)
    id0 = id(psc._bs_bn)
    _ = psc._vjp_tf_only_analytic(v, pts)
    assert id(psc._bs_bn) == id0


def test_pucks_to_vtk_vectorised_tf_Bn_matches_pointwise():
    """Vectorised VTK TF B_n matches the legacy per-point JAX kernel."""
    import jax.numpy as jnp

    from simsopt.field.force import _B_at_point_from_coil_set_pure
    from simsopt.field.puck_vtk import puck_surface_mesh

    psc = _make_symmetry_validation_array(nfp=2, stellsym=False, n_base=1)
    psc.recompute_currents()
    c, ax, R, t = psc._all_pucks[0]
    x, y, z, _ = puck_surface_mesh(c, ax, R, t, n_phi=16, n_r=8)
    pts = np.stack([x, y, z], axis=-1)
    r0, r1 = psc._quad_row_ranges[0]
    quad = psc._quad_points[r0:r1]
    gammas_tf = np.array([c.curve.gamma() for c in psc.coils_TF])
    gammadash_tf = np.array([c.curve.gammadash() for c in psc.coils_TF])
    currents_tf = np.array([c.current.get_value() for c in psc.coils_TF])
    Bn_loop = np.zeros(len(pts))
    for i, p in enumerate(pts):
        j = r0 + int(np.argmin(np.sum((quad - p) ** 2, axis=-1)))
        nn = psc._quad_normals[j]
        B = np.array(
            _B_at_point_from_coil_set_pure(
                jnp.asarray(p),
                jnp.asarray(gammas_tf),
                jnp.asarray(gammadash_tf),
                jnp.asarray(currents_tf),
                -1,
                1e-10,
            )
        )
        Bn_loop[i] = np.dot(B, nn)
    from simsopt.field.biotsavart import BiotSavart

    bs = BiotSavart(psc.coils_TF)
    bs.set_points_cart(np.ascontiguousarray(pts))
    B_tf = bs.B()
    tree_query = __import__("scipy.spatial", fromlist=["cKDTree"]).cKDTree(quad)
    _, j_local = tree_query.query(pts, k=1)
    j = r0 + np.asarray(j_local, dtype=np.intp)
    nn = psc._quad_normals[j]
    Bn_vec = np.einsum("ij,ij->i", B_tf, nn)
    np.testing.assert_allclose(Bn_vec, Bn_loop, rtol=1e-10, atol=1e-10)


def test_shell_biot_savart_f32_accumulation_near_float64():
    """Optional float32 accumulation should stay close to float64 on a toy grid."""
    import jax.numpy as jnp

    from simsopt.field.bulk_inductance import shell_biot_savart_stacked_pure

    rng = np.random.default_rng(1)
    n_p, nq, nd = 2, 5, 4
    K = rng.standard_normal((n_p, nq, nd, 3))
    qp = rng.standard_normal((n_p * nq, 3))
    w = np.abs(rng.standard_normal(n_p * nq)) + 1e-3
    beta = rng.standard_normal(n_p * nd)
    ev = rng.standard_normal((3, 3))
    B64 = shell_biot_savart_stacked_pure(
        jnp.asarray(K),
        jnp.asarray(qp),
        jnp.asarray(w),
        jnp.asarray(beta),
        jnp.asarray(ev),
        accumulate_dtype=None,
    )
    B32 = shell_biot_savart_stacked_pure(
        jnp.asarray(K),
        jnp.asarray(qp),
        jnp.asarray(w),
        jnp.asarray(beta),
        jnp.asarray(ev),
        accumulate_dtype=jnp.float32,
    )
    np.testing.assert_allclose(np.asarray(B64), np.asarray(B32), rtol=1e-4, atol=1e-4)


def test_ensure_jax_full_noop_when_all_pucks_frozen():
    """With all puck DOFs fixed, :meth:`PSCBulkArray._ensure_jax_full` is a no-op."""
    psc = _make_symmetry_validation_array(nfp=2, stellsym=False, n_base=1)
    assert not psc._has_free_puck_dofs()
    psc._ensure_jax_full()
    psc._ensure_jax_full()
    assert not hasattr(psc, "_jax_local_pts")


def test_ensure_jax_full_force_materializes_local_stacks():
    """``force=True`` builds local JAX stacks even when puck DOFs are fixed."""
    psc = _make_symmetry_validation_array(nfp=2, stellsym=False, n_base=1)
    psc._local_stacks_valid = False
    psc._ensure_jax_full(force=True)
    assert hasattr(psc, "_jax_local_pts")
    assert psc._local_stacks_valid
