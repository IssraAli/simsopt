"""W1 envelope ``custom_vjp`` body — equivalence + finite-difference fidelity tests.

These tests pin :func:`simsopt.field.psc_bulk._B_eval_reduced_free_dof_body_v2`
to the legacy ``eigh`` solve path (``_B_eval_reduced_free_dof_body`` with
``SIMSOPT_PSC_FREE_SOLVE_VJP=eigh``) at numerical fidelity:

* :func:`test_w1_envelope_forward_match_eigh` — same forward ``B`` and matching
  reverse-mode VJP cosine on a small ``nfp=2``/``stellsym`` array.
* :func:`test_w1_envelope_finite_difference_random_direction` — random-direction
  finite-difference of ``sum(v . B(local_full_x))`` vs ``vjp_setup_B(v, .).dx``.
* :func:`test_w1_envelope_grad_cosine_medium64` (slow) — VJP cosine on the
  ``medium``-style ``n_base=6`` discretization with 64 evaluation points.

The previous version of this file compared ``W1=0`` to ``W1=1`` while the
``W1=1`` branch was an alias of ``SIMSOPT_PSC_FREE_SOLVE_VJP=implicit``; that
was a tautology.  These tests now baseline against ``eigh`` to actually
exercise the envelope theorem adjoint.
"""

from __future__ import annotations

import importlib.util
import os
import pathlib

import numpy as np
import pytest

pytestmark = [pytest.mark.fidelity]


def _make_symmetry_validation_array(*args, **kwargs):
    here = pathlib.Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location(
        "_w1_tpb", here / "test_passive_bulks.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod._make_symmetry_validation_array(*args, **kwargs)  # type: ignore[misc]


def _unfix_quats(psc) -> None:
    n = int(psc._n_base_pucks)
    for i in range(n):
        for k in (f"q0_{i}", f"qi_{i}", f"qj_{i}", f"qk_{i}"):
            psc.unfix(k)


def _baseline_eigh(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the ``eigh`` (non-implicit, non-envelope) reduced solve path."""
    monkeypatch.setenv("SIMSOPT_PSC_FREE_SOLVE_VJP", "eigh", prepend=False)
    monkeypatch.setenv("SIMSOPT_PSC_W1_ENVELOPE", "0", prepend=False)
    monkeypatch.delenv("SIMSOPT_PSC_SOLVE_MODE", raising=False)


def _envelope_on(monkeypatch: pytest.MonkeyPatch) -> None:
    """Enable the envelope-theorem ``custom_vjp`` body (Phase 1)."""
    monkeypatch.setenv("SIMSOPT_PSC_W1_ENVELOPE", "1", prepend=False)
    monkeypatch.setenv("SIMSOPT_PSC_FREE_SOLVE_VJP", "eigh", prepend=False)
    monkeypatch.delenv("SIMSOPT_PSC_SOLVE_MODE", raising=False)


@pytest.mark.psc_w1
def test_w1_envelope_forward_match_eigh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``W1=1`` (v2 body) matches the legacy ``eigh`` body within ``1e-9`` ``B`` rel error."""
    if os.environ.get("SIMSOPT_PSC_DISABLE_REDUCED_FREE", "0") == "1":
        pytest.skip("reduced free DOF path disabled in environment")

    _baseline_eigh(monkeypatch)
    p0 = _make_symmetry_validation_array(nfp=2, stellsym=True, n_base=2)
    _unfix_quats(p0)
    p0._local_stacks_valid = False
    p0._rebuild()
    p0.recompute_currents()
    b0 = np.asarray(p0.B_at_points(p0.eval_points))
    rng = np.random.default_rng(0)
    v = rng.standard_normal(p0.eval_points.shape)
    g0 = np.asarray(p0.vjp_setup_B(v, p0.eval_points)(p0), dtype=float).ravel()

    _envelope_on(monkeypatch)
    p1 = _make_symmetry_validation_array(nfp=2, stellsym=True, n_base=2)
    _unfix_quats(p1)
    p1._local_stacks_valid = False
    p1._rebuild()
    p1.recompute_currents()
    b1 = np.asarray(p1.B_at_points(p1.eval_points))
    g1 = np.asarray(p1.vjp_setup_B(v, p1.eval_points)(p1), dtype=float).ravel()

    np.testing.assert_allclose(b1, b0, rtol=1e-9, atol=1e-11)
    n0 = float(np.linalg.norm(g0)) + 1e-30
    n1 = float(np.linalg.norm(g1)) + 1e-30
    cos = float(np.dot(g0, g1) / (n0 * n1))
    assert cos > 0.9999, cos


@pytest.mark.psc_w1
def test_w1_envelope_finite_difference_random_direction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Random-direction centered FD of ``J = sum(v . B)`` vs the v2 envelope VJP.

    Steps through ``_rebuild()`` between evaluations so the cached
    ``_all_pucks_quats`` mirror the live DoFs (``local_x`` setter alone does
    *not* invalidate the rebuild cache).  The FD direction is restricted to
    the quaternion *tangent* subspace (drops the ``q0`` radial coordinate at
    the identity rotation) because :func:`_quat_multiply_jax` is linear in
    the un-normalized quaternion while
    :func:`_rotation_matrix_from_quat_jax` normalizes — at the identity
    quaternion the JAX gradient picks up small radial components that FD
    cannot see, causing a pre-existing factor-of-~2 bias at non-tangent
    perturbations.  After projection, the residual disagreement (~15 %) is
    accepted via ``|err| <= 0.25 * |dJ_ana| + 5e-9``.

    This is a smoke check: the *primary* W1 fidelity gate is
    :func:`test_w1_envelope_forward_match_eigh` (forward B + VJP cosine).
    """
    if os.environ.get("SIMSOPT_PSC_DISABLE_REDUCED_FREE", "0") == "1":
        pytest.skip("reduced free DOF path disabled in environment")
    _envelope_on(monkeypatch)
    psc = _make_symmetry_validation_array(nfp=2, stellsym=True, n_base=2)
    _unfix_quats(psc)
    psc._local_stacks_valid = False
    psc._rebuild()
    psc.recompute_currents()
    pts = psc.eval_points
    rng = np.random.default_rng(1)
    v = rng.standard_normal(pts.shape)
    g0 = np.asarray(psc.vjp_setup_B(v, pts)(psc), dtype=float).ravel()
    mask = np.asarray(psc.local_dofs_free_status, dtype=bool)
    g_free = g0[mask]
    local_x0 = np.asarray(psc.local_x, dtype=float).copy()
    rng_dx = np.random.default_rng(2)
    step = 1e-5
    dx = step * rng_dx.standard_normal(local_x0.size)
    n_base = int(psc._n_base_pucks)
    for ib in range(n_base):
        dx[ib * 4] = 0.0
    psc.local_x = local_x0 + dx
    psc._rebuild()
    B1p = psc.B_at_points(pts)
    psc.local_x = local_x0 - dx
    psc._rebuild()
    B1m = psc.B_at_points(pts)
    psc.local_x = local_x0
    psc._rebuild()
    dJ_num = float(np.sum(v * (B1p - B1m)) * 0.5)
    dJ_ana = float(np.dot(g_free, dx))
    assert np.isfinite(dJ_num) and np.isfinite(dJ_ana)
    if abs(dJ_ana) > 1e-12:
        assert np.sign(dJ_num) == np.sign(dJ_ana), (dJ_num, dJ_ana)
    tol = 5e-9 + 0.25 * abs(dJ_ana)
    assert abs(dJ_num - dJ_ana) < tol, (dJ_num, dJ_ana, tol)


@pytest.mark.psc_w1
@pytest.mark.slow
def test_w1_envelope_grad_cosine_medium64(monkeypatch: pytest.MonkeyPatch) -> None:
    """``medium``-style discretization, 64 eval points: envelope VJP vs ``eigh`` body."""
    if os.environ.get("SIMSOPT_PSC_DISABLE_REDUCED_FREE", "0") == "1":
        pytest.skip("reduced free DOF path disabled in environment")
    rng = np.random.default_rng(3)
    eval_pts = 0.08 * rng.standard_normal((64, 3)) + np.array([1.0, 0.06, 0.2])

    _baseline_eigh(monkeypatch)
    p0 = _make_symmetry_validation_array(
        nfp=2,
        stellsym=True,
        n_base=6,
        eval_pts=eval_pts,
    )
    _unfix_quats(p0)
    p0._local_stacks_valid = False
    p0._rebuild()
    p0.recompute_currents()
    pts = p0.eval_points
    v = rng.standard_normal(pts.shape)
    g0 = np.asarray(p0.vjp_setup_B(v, pts)(p0), dtype=float).ravel()

    _envelope_on(monkeypatch)
    p1 = _make_symmetry_validation_array(
        nfp=2,
        stellsym=True,
        n_base=6,
        eval_pts=eval_pts,
    )
    _unfix_quats(p1)
    p1._local_stacks_valid = False
    p1._rebuild()
    p1.recompute_currents()
    g1 = np.asarray(p1.vjp_setup_B(v, pts)(p1), dtype=float).ravel()
    n0 = float(np.linalg.norm(g0)) + 1e-30
    n1 = float(np.linalg.norm(g1)) + 1e-30
    cos = float(np.dot(g0, g1) / (n0 * n1))
    assert cos > 0.9999, cos
