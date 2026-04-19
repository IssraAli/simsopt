"""
Tests for passive bulk (ideal diamagnetic) pucks.
"""

import os
import tempfile

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("jax.numpy")

from simsopt.field.bulk_inductance import (
    null_space_projection_matrix,
    shell_inductance_matrix_blockwise,
    shell_inductance_matrix_pure,
    shell_loading_vector_pure,
)
from simsopt._core.optimizable import Optimizable
from simsopt.field.coil import Coil, Current
from simsopt.field.psc_bulk import PSCBulkArray
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

    L = psc._L_full
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

    L = psc._L_full
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
    assert diff_med_hi < diff_lo_med + 1e-15, (
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


def _assert_taylor_convergence(errors, label="", strict_2nd_order=True):
    """Check that error ratios show 2nd-order convergence.

    For a second-order-accurate central finite difference, halving the
    step ``eps`` should quarter the error (ratio ~0.25).  We require:

    * the *best* observed ratio is strictly below ``0.35`` (confirming
      a 2nd-order regime was entered somewhere in the sweep), and
    * if ``strict_2nd_order=True``, the median of the pre-asymptotic
      ratios (i.e. dropping the final ratio which is typically
      rounding-noise dominated and the first two which may still be
      in the linear regime for non-analytic objectives) is also
      below ``0.35``.

    Args:
        errors: List of absolute / relative errors, one per eps value.
        label: Human-readable label used in assertion messages.
        strict_2nd_order: If ``True`` (default), also assert the
            pre-asymptotic median is < 0.35.  Pass ``False`` for very
            short sweeps where trimming would leave no samples.
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
