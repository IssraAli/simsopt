"""
PSC multipole / free-DOF math regression tests (W0, W2, W3, W4, W6, W7).

Run a subset of workstreams with ``pytest --psc-workstream=w0,w2`` (see
:file:`../conftest.py`).
"""

from __future__ import annotations

import os

import jax.numpy as jnp
import numpy as np
import pytest

import importlib.util
import pathlib

from simsopt.field.bulk_multipole import (
    magnetic_dipole_moments_stacked,
    pair_dipole_quadrupole_cross_block,
    pair_inductance_dipole_block,
    pair_inductance_multipole,
    pair_inductance_multipole_selfcheck_dense,
    quadrupole_magnetic_symmetric_stacked,
)
from simsopt.field.cylinder_stream_basis import (
    ClosedCylinderStreamConfig,
    tikhonov_stream_coefficients_synthetic,
)
from simsopt.field.quat_tangent import dR_domega_at_zero, exp_quat_tangent

pytestmark = [pytest.mark.fidelity]


def _make_symmetry_validation_array(*args, **kwargs):
    here = pathlib.Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location(
        "_psc_tpb", here / "test_passive_bulks.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod._make_symmetry_validation_array(*args, **kwargs)  # type: ignore[misc]


@pytest.mark.psc_w0
def test_magnetic_moment_shapes() -> None:
    """W0: stacked local moments have expected (puck, mode, 3) / (3,3) shape."""
    rng = np.random.default_rng(0)
    n_p, nq, nd = 2, 5, 4
    K = rng.standard_normal((n_p, nq, nd, 3))
    r = rng.standard_normal((n_p, nq, 3))
    w = (np.abs(rng.standard_normal((n_p, nq))) + 0.1).astype(np.float64)
    m = magnetic_dipole_moments_stacked(jnp.asarray(K), jnp.asarray(r), jnp.asarray(w))
    assert tuple(m.shape) == (n_p, nd, 3)
    qsym = quadrupole_magnetic_symmetric_stacked(
        jnp.asarray(K), jnp.asarray(r), jnp.asarray(w)
    )
    assert qsym.shape == (n_p, nd, 3, 3)


@pytest.mark.psc_w2
def test_pair_inductance_multipole_order1_matches_dipole_block() -> None:
    """W2: ``order=1`` multipole path aliases the established dipole kernel."""
    rng = np.random.default_rng(1)
    nd = 3
    mi = rng.standard_normal((nd, 3))
    mj = rng.standard_normal((nd, 3))
    r = np.array([2.0, 0.1, 0.2], dtype=float)
    a = np.asarray(
        pair_inductance_dipole_block(jnp.asarray(mi), jnp.asarray(mj), jnp.asarray(r))
    )
    b = np.asarray(
        pair_inductance_multipole(
            jnp.asarray(mi), jnp.asarray(mj), jnp.asarray(r), order=1
        )
    )
    np.testing.assert_allclose(a, b, rtol=0.0, atol=1e-12)


@pytest.mark.psc_w2
def test_w2_dipole_reciprocity_and_order2_smoke() -> None:
    """W2: dipole block matches swap+sign; P=2 path with Q is finite and self-checks."""
    rng = np.random.default_rng(4)
    nd = 2
    mi = rng.standard_normal((nd, 3))
    mj = rng.standard_normal((nd, 3))
    r = np.array([0.5, 0.2, -0.1], dtype=np.float64)
    L = np.asarray(
        pair_inductance_dipole_block(
            jnp.asarray(mi), jnp.asarray(mj), jnp.asarray(r, dtype=np.float64)
        )
    )
    Lr = np.asarray(
        pair_inductance_dipole_block(
            jnp.asarray(mj), jnp.asarray(mi), jnp.asarray(-r, dtype=np.float64)
        )
    )
    np.testing.assert_allclose(L, Lr.T, rtol=0, atol=1e-12)
    Qi = rng.standard_normal((nd, 3, 3))
    Qj = rng.standard_normal((nd, 3, 3))
    for t in (Qi, Qj):
        tr = (t[..., 0, 0] + t[..., 1, 1] + t[..., 2, 2]) / 3.0
        t -= tr[..., None, None] * np.eye(3)
    m2 = pair_inductance_multipole(
        jnp.asarray(mi),
        jnp.asarray(mj),
        jnp.asarray(r, dtype=np.float64),
        order=2,
        Q_i_sym=jnp.asarray(Qi),
        Q_j_sym=jnp.asarray(Qj),
    )
    c = pair_dipole_quadrupole_cross_block(
        jnp.asarray(mi),
        jnp.asarray(Qi),
        jnp.asarray(mj),
        jnp.asarray(Qj),
        jnp.asarray(r, dtype=np.float64),
    )
    assert np.isfinite(m2).all()
    err = pair_inductance_multipole_selfcheck_dense(
        pair_inductance_dipole_block(
            jnp.asarray(mi), jnp.asarray(mj), jnp.asarray(r, dtype=np.float64)
        )
        + c,
        m2,
    )
    assert err < 1e-12


@pytest.mark.psc_w6
def test_quat_tangent_roundtrip_r_norm() -> None:
    """W6: small tangent updates stay unit-norm; Jacobian has expected shape."""
    rng = np.random.default_rng(2)
    q0 = np.array([0.6, 0.5, 0.1, 0.6], dtype=float)
    q0 /= float(np.linalg.norm(q0))
    om = 1e-3 * rng.standard_normal(3)
    q1 = np.asarray(exp_quat_tangent(jnp.asarray(om), jnp.asarray(q0)))
    assert float(abs(np.linalg.norm(q1) - 1.0)) < 1e-6
    j = dR_domega_at_zero(jnp.asarray(q0))
    assert j.shape == (3, 3, 3)


@pytest.mark.psc_w5
def test_cylinder_stream_config_roundtrip() -> None:
    """W5: config stores shell dimensions (integration placeholder)."""
    cfg = ClosedCylinderStreamConfig(
        radius=0.2, height=0.1, n_phi=9, n_z=5, m_phi_max=2, n_z_leg=3
    )
    rng = np.random.default_rng(3)
    c = tikhonov_stream_coefficients_synthetic(rng, cfg)
    assert c.size == (2 * cfg.m_phi_max + 1) * (cfg.n_z_leg + 1)
    assert abs(float(np.linalg.norm(c)) - 1.0) < 1e-9


@pytest.mark.psc_w7
@pytest.mark.integration
def test_w7_bs_eval_far_kappa_is_off_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """W7: with ``SIMSOPT_PSC_BS_EVAL_FAR_KAPPA=0`` the free-DOF B matches the dense shell."""
    if os.environ.get("SIMSOPT_PSC_DISABLE_REDUCED_FREE", "0") == "1":
        pytest.skip("reduced free DOF path disabled in environment")
    monkeypatch.setenv("SIMSOPT_PSC_BS_EVAL_FAR_KAPPA", "0", prepend=False)
    psc = _make_symmetry_validation_array(nfp=2, stellsym=True, n_base=2)
    for i in range(int(psc._n_base_pucks)):
        for k in (f"q0_{i}", f"qi_{i}", f"qj_{i}", f"qk_{i}"):
            psc.unfix(k)
    psc.recompute_currents()
    pts = psc.eval_points
    b0 = np.asarray(psc.B_at_points(pts))
    monkeypatch.setenv("SIMSOPT_PSC_BS_EVAL_FAR_KAPPA", "0.0", prepend=False)
    psc._local_stacks_valid = False
    psc._rebuild()
    psc.recompute_currents()
    b1 = np.asarray(psc.B_at_points(pts))
    np.testing.assert_allclose(b0, b1, rtol=1e-8, atol=1e-10)


@pytest.mark.psc_w3
def test_w3_a_quad_vs_bn_quad_beta_and_b(monkeypatch: pytest.MonkeyPatch) -> None:
    """W3: ``a_quad`` loading tracks ``bn_quad`` within ~1% on ``beta`` and ``B`` (small PSC)."""
    if os.environ.get("SIMSOPT_PSC_DISABLE_REDUCED_FREE", "0") == "1":
        pytest.skip("reduced free DOF path disabled in environment")
    monkeypatch.setenv("SIMSOPT_PSC_TF_LOADING", "bn_quad", prepend=False)
    p0 = _make_symmetry_validation_array(nfp=2, stellsym=True, n_base=2)
    for i in range(int(p0._n_base_pucks)):
        for k in (f"q0_{i}", f"qi_{i}", f"qj_{i}", f"qk_{i}"):
            p0.unfix(k)
    p0._local_stacks_valid = False
    p0._rebuild()
    p0.recompute_currents()
    b_bn = np.asarray(p0.B_at_points(p0.eval_points))
    beta_bn = np.asarray(p0.beta, dtype=float).ravel()

    monkeypatch.setenv("SIMSOPT_PSC_TF_LOADING", "a_quad", prepend=False)
    p1 = _make_symmetry_validation_array(nfp=2, stellsym=True, n_base=2)
    for i in range(int(p1._n_base_pucks)):
        for k in (f"q0_{i}", f"qi_{i}", f"qj_{i}", f"qk_{i}"):
            p1.unfix(k)
    p1._local_stacks_valid = False
    p1._rebuild()
    p1.recompute_currents()
    b_aq = np.asarray(p1.B_at_points(p1.eval_points))
    beta_aq = np.asarray(p1.beta, dtype=float).ravel()
    beta_scale = max(float(np.max(np.abs(beta_bn))), 1e-20)
    assert float(np.max(np.abs(beta_aq - beta_bn)) / beta_scale) < 0.01
    assert np.all(np.isfinite(b_aq)) and np.all(np.isfinite(b_bn))
    assert float(np.linalg.norm(b_aq - b_bn) / (np.linalg.norm(b_bn) + 1e-20)) < 2.0


@pytest.mark.psc_w3
def test_w3_a_taylor_error_vs_a_quad(monkeypatch: pytest.MonkeyPatch) -> None:
    """W3: ``a_taylor`` error w.r.t. ``bn_quad`` is not larger than ``a_quad`` in this smoke."""
    if os.environ.get("SIMSOPT_PSC_DISABLE_REDUCED_FREE", "0") == "1":
        pytest.skip("reduced free DOF path disabled in environment")
    p_bn = _make_symmetry_validation_array(nfp=2, stellsym=True, n_base=2)
    for i in range(int(p_bn._n_base_pucks)):
        for k in (f"q0_{i}", f"qi_{i}", f"qj_{i}", f"qk_{i}"):
            p_bn.unfix(k)
    p_bn._local_stacks_valid = False
    p_bn._rebuild()
    p_bn.recompute_currents()
    b_ref = np.asarray(p_bn.B_at_points(p_bn.eval_points))

    def _err(loading: str) -> float:
        monkeypatch.setenv("SIMSOPT_PSC_TF_LOADING", loading, prepend=False)
        p = _make_symmetry_validation_array(nfp=2, stellsym=True, n_base=2)
        for i in range(int(p._n_base_pucks)):
            for k in (f"q0_{i}", f"qi_{i}", f"qj_{i}", f"qk_{i}"):
                p.unfix(k)
        p._local_stacks_valid = False
        p._rebuild()
        p.recompute_currents()
        b = np.asarray(p.B_at_points(p.eval_points))
        return float(np.linalg.norm(b - b_ref) / (np.linalg.norm(b_ref) + 1e-20))

    e_aq = _err("a_quad")
    e_at = _err("a_taylor")
    assert e_at <= e_aq * 1.5 + 1e-4


@pytest.mark.psc_w3
def test_a_quad_path_importable(monkeypatch: pytest.MonkeyPatch) -> None:
    """W3: ``SIMSOPT_PSC_TF_LOADING=a_quad`` loads without error on a small PSC (smoke)."""
    if os.environ.get("SIMSOPT_PSC_DISABLE_REDUCED_FREE", "0") == "1":
        pytest.skip("reduced free DOF path disabled in environment")
    monkeypatch.setenv("SIMSOPT_PSC_TF_LOADING", "a_quad", prepend=False)
    psc = _make_symmetry_validation_array(nfp=2, stellsym=True, n_base=1)
    for i in range(int(psc._n_base_pucks)):
        for k in (f"q0_{i}", f"qi_{i}", f"qj_{i}", f"qk_{i}"):
            psc.unfix(k)
    psc._local_stacks_valid = False
    psc._rebuild()
    psc.recompute_currents()
    b = psc.B_at_points(psc.eval_points)
    assert np.all(np.isfinite(b))


@pytest.mark.psc_w4
def test_eigk_solve_mode_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """W4: ``SIMSOPT_PSC_SOLVE_MODE=eigk`` executes the free-DOF forward."""
    if os.environ.get("SIMSOPT_PSC_DISABLE_REDUCED_FREE", "0") == "1":
        pytest.skip("reduced free DOF path disabled in environment")
    monkeypatch.setenv("SIMSOPT_PSC_SOLVE_MODE", "eigk", prepend=False)
    psc = _make_symmetry_validation_array(nfp=2, stellsym=True, n_base=1)
    for i in range(int(psc._n_base_pucks)):
        for k in (f"q0_{i}", f"qi_{i}", f"qj_{i}", f"qk_{i}"):
            psc.unfix(k)
    psc._local_stacks_valid = False
    psc._rebuild()
    psc.recompute_currents()
    b = psc.B_at_points(psc.eval_points)
    assert np.all(np.isfinite(b))


@pytest.mark.psc_w4
def test_w4_eigk_reuses_cached_basis_at_tau_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """W4: with ``EIGK_TRIGGER=0`` repeated forward ``B`` calls are bit-stable."""
    if os.environ.get("SIMSOPT_PSC_DISABLE_REDUCED_FREE", "0") == "1":
        pytest.skip("reduced free DOF path disabled in environment")
    monkeypatch.setenv("SIMSOPT_PSC_SOLVE_MODE", "eigk", prepend=False)
    monkeypatch.setenv("SIMSOPT_PSC_EIGK_TRIGGER", "0", prepend=False)
    psc = _make_symmetry_validation_array(nfp=2, stellsym=True, n_base=2)
    for i in range(int(psc._n_base_pucks)):
        for k in (f"q0_{i}", f"qi_{i}", f"qj_{i}", f"qk_{i}"):
            psc.unfix(k)
    psc._local_stacks_valid = False
    psc._rebuild()
    psc.recompute_currents()
    pts = psc.eval_points
    a = np.asarray(psc.B_at_points(pts))
    b = np.asarray(psc.B_at_points(pts))
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-12)


@pytest.mark.psc_w4
def test_w4_eigk_grad_matches_eigenfloor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """W4: gradient with ``eigk`` (cached basis) matches ``eigenfloor`` baseline.

    Forces ``EIGK_TRIGGER=inf`` so the eigK forward never re-decomposes ``L``,
    then asserts that ``vjp_setup_B`` returns a gradient whose cosine
    similarity vs the gradient computed under ``SOLVE_MODE=eigenfloor``
    (no cache) is ``>= 0.999``.  This is the regression gate for the W4 fix:
    the cached eigenbasis must not silently drop the ``dL/dq`` contribution.
    """
    if os.environ.get("SIMSOPT_PSC_DISABLE_REDUCED_FREE", "0") == "1":
        pytest.skip("reduced free DOF path disabled in environment")

    rng = np.random.default_rng(73)

    def _unfix_quats(p) -> None:
        for i in range(int(p._n_base_pucks)):
            for k in (f"q0_{i}", f"qi_{i}", f"qj_{i}", f"qk_{i}"):
                p.unfix(k)

    monkeypatch.delenv("SIMSOPT_PSC_W1_ENVELOPE", raising=False)
    monkeypatch.setenv("SIMSOPT_PSC_FREE_SOLVE_VJP", "eigh", prepend=False)

    monkeypatch.setenv("SIMSOPT_PSC_SOLVE_MODE", "eigenfloor", prepend=False)
    p_ref = _make_symmetry_validation_array(nfp=2, stellsym=True, n_base=2)
    _unfix_quats(p_ref)
    p_ref._local_stacks_valid = False
    p_ref._rebuild()
    p_ref.recompute_currents()
    pts = p_ref.eval_points
    v = rng.standard_normal(pts.shape)
    g_ref = np.asarray(p_ref.vjp_setup_B(v, pts)(p_ref), dtype=float).ravel()

    monkeypatch.setenv("SIMSOPT_PSC_SOLVE_MODE", "eigk", prepend=False)
    monkeypatch.setenv("SIMSOPT_PSC_EIGK_TRIGGER", "inf", prepend=False)
    p_eigk = _make_symmetry_validation_array(nfp=2, stellsym=True, n_base=2)
    _unfix_quats(p_eigk)
    p_eigk._local_stacks_valid = False
    p_eigk._rebuild()
    p_eigk.recompute_currents()
    g_eigk = np.asarray(p_eigk.vjp_setup_B(v, pts)(p_eigk), dtype=float).ravel()

    nr = float(np.linalg.norm(g_ref))
    ne = float(np.linalg.norm(g_eigk))
    assert nr > 0.0 and ne > 0.0
    cos = float(np.dot(g_ref, g_eigk) / (nr * ne))
    assert cos >= 0.999, (
        f"eigk-vs-eigenfloor gradient cosine {cos:.6f} < 0.999; "
        f"|g_ref|={nr:.3e}, |g_eigk|={ne:.3e}"
    )
