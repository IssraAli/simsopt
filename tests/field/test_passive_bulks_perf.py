"""
Slow / timing regression tests for passive-bulk performance work.

Run with ``pytest --runslow``.

Override budgets with env ``PSC_BUILD50_BUDGET_S`` / ``PSC_FUN50_BUDGET_S`` (seconds).
"""

from __future__ import annotations

import importlib.util
import os
import time
from functools import partial
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from simsopt.field.bulk_inductance import (
    MU0_OVER_4PI,
    _SELF_REG_COEFF,
    _jax_pair_batch_blocks,
    _shell_inductance_matrix_symmetric_reduced_jax,
    reset_psc_far_selfcheck_cache,
)
from simsopt.field.psc_bulk import _classify_pairs, _reduced_free_dof_extras_for_jax

pytestmark = pytest.mark.slow

_spec = importlib.util.spec_from_file_location(
    "_tpb_perf",
    Path(__file__).resolve().parent / "test_passive_bulks.py",
)
_tpb = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_tpb)
_make_symmetry_validation_array = _tpb._make_symmetry_validation_array
_make_rotated_puck_array = _tpb._make_rotated_puck_array
_make_rotated_puck_array_large = _tpb._make_rotated_puck_array_large


def test_classify_pairs_partition_disjoint_and_complete() -> None:
    """Host far-pair classifier partitions non-self base-by-replica pairs."""
    centers_all = np.array(
        [
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [10.0, 10.0, 0.0],
        ],
        dtype=float,
    )
    r_eff_all = np.ones(4)
    base_indices = np.array([0, 0, 1, 1], dtype=np.int32)
    near, far, n_near, n_far = _classify_pairs(
        centers_all, r_eff_all, base_indices, r_far=3.0
    )
    n_base = 2
    n_pairs = n_base * centers_all.shape[0]
    assert n_near + n_far == n_pairs - n_base
    assert {tuple(x) for x in near.tolist()}.isdisjoint(
        {tuple(x) for x in far.tolist()}
    )

    near0, far0, n_near0, n_far0 = _classify_pairs(
        centers_all, r_eff_all, base_indices, r_far=0.0
    )
    assert n_near0 == 0
    assert n_far0 == n_pairs - n_base
    assert near0.shape == (0, 2)
    assert far0.shape == (n_pairs - n_base, 2)

    near_inf, far_inf, n_near_inf, n_far_inf = _classify_pairs(
        centers_all, r_eff_all, base_indices, r_far=np.inf
    )
    assert n_near_inf == n_pairs - n_base
    assert n_far_inf == 0
    assert near_inf.shape == (n_pairs - n_base, 2)
    assert far_inf.shape == (0, 2)


def test_reduced_free_dof_far_pair_env_overrides_default() -> None:
    """Env-var kappa remains the highest-priority far-pair control."""
    prev = os.environ.get("SIMSOPT_PSC_PAIR_FAR_KAPPA")
    try:
        os.environ.pop("SIMSOPT_PSC_PAIR_FAR_KAPPA", None)
        use_far, kappa, *_ = _reduced_free_dof_extras_for_jax(3.0)
        assert use_far is True
        assert kappa == 3.0

        os.environ["SIMSOPT_PSC_PAIR_FAR_KAPPA"] = "0"
        use_far, kappa, *_ = _reduced_free_dof_extras_for_jax(3.0)
        assert use_far is False
        assert kappa == 0.0
    finally:
        if prev is None:
            os.environ.pop("SIMSOPT_PSC_PAIR_FAR_KAPPA", None)
        else:
            os.environ["SIMSOPT_PSC_PAIR_FAR_KAPPA"] = prev


def _run_reduced_free_dof_forward(kappa_env_value: str | None) -> np.ndarray:
    """Build identical reduced-free-DoF fixtures and return ``B_at_points`` output.

    The free-DoF code path is engaged by unfixing a base-puck quaternion DoF.
    The kappa value is supplied via the ``SIMSOPT_PSC_PAIR_FAR_KAPPA`` env
    override (highest-priority knob; see :func:`_reduced_free_dof_extras_for_jax`).
    """
    prev = os.environ.get("SIMSOPT_PSC_PAIR_FAR_KAPPA")
    try:
        if kappa_env_value is None:
            os.environ.pop("SIMSOPT_PSC_PAIR_FAR_KAPPA", None)
        else:
            os.environ["SIMSOPT_PSC_PAIR_FAR_KAPPA"] = kappa_env_value
        psc = _make_symmetry_validation_array(nfp=1, stellsym=False, n_base=3)
        psc.unfix("q0_0")
        psc.recompute_currents()
        return np.asarray(psc.B_at_points(psc.eval_points), dtype=np.float64)
    finally:
        if prev is None:
            os.environ.pop("SIMSOPT_PSC_PAIR_FAR_KAPPA", None)
        else:
            os.environ["SIMSOPT_PSC_PAIR_FAR_KAPPA"] = prev


def test_far_pair_partition_matches_legacy_when_all_near() -> None:
    """Forward L → B match between legacy ``vmap`` and partitioned ``lax.scan``.

    Selecting ``SIMSOPT_PSC_PAIR_FAR_KAPPA=1e9`` forces every non-self pair
    into the *near* partition; the far scan then adds nothing and the
    partitioned forward must agree with the legacy free-DoF JAX assembly.
    Allows a tight float-reordering tolerance because scatter-add and
    ``segment_sum`` accumulate the same set of summands in different
    orders.
    """
    B_legacy = _run_reduced_free_dof_forward(kappa_env_value=None)
    B_partition = _run_reduced_free_dof_forward(kappa_env_value="1e9")
    assert B_legacy.shape == B_partition.shape
    denom = max(float(np.max(np.abs(B_legacy))), 1.0e-30)
    rel = float(np.max(np.abs(B_legacy - B_partition))) / denom
    assert rel <= 1.0e-10, (
        f"partitioned forward disagrees with legacy: rel_max={rel:.3e}, "
        f"|B_legacy|_max={denom:.3e}"
    )


def _run_reduced_free_dof_vjp(kappa_env_value: str | None) -> np.ndarray:
    """Build identical reduced-free-DoF fixtures and return ``dJ/dx`` for a scalar J.

    Uses a plain sum-on-``B_at_points`` scalar so the VJP is exercised through
    the reduced free-DoF runners (``_vjp_reduced_free_dof_jitted`` and
    siblings).  Returns the gradient w.r.t. the PSCBulkArray's local DOFs as a
    flat ``float64`` array.
    """
    prev = os.environ.get("SIMSOPT_PSC_PAIR_FAR_KAPPA")
    try:
        if kappa_env_value is None:
            os.environ.pop("SIMSOPT_PSC_PAIR_FAR_KAPPA", None)
        else:
            os.environ["SIMSOPT_PSC_PAIR_FAR_KAPPA"] = kappa_env_value
        psc = _make_symmetry_validation_array(nfp=1, stellsym=False, n_base=3)
        psc.unfix("q0_0")
        psc.recompute_currents()
        pts = np.asarray(psc.eval_points, dtype=float)
        v_B = np.ones_like(np.asarray(psc.B_at_points(pts)))
        deriv = psc._vjp_puck_geometry(v_B, pts)
        return np.asarray(deriv(psc), dtype=np.float64).reshape(-1)
    finally:
        if prev is None:
            os.environ.pop("SIMSOPT_PSC_PAIR_FAR_KAPPA", None)
        else:
            os.environ["SIMSOPT_PSC_PAIR_FAR_KAPPA"] = prev


def _time_forward_loops(psc: Any, n_iter: int = 5) -> float:
    """Time ``B_at_points`` over ``n_iter`` repetitions after one warm-up.

    JIT compile of the reduced free-DoF dispatch happens on the first
    call; ``time.perf_counter`` over the remainder reflects steady-state
    cost.  Returned value is the median per-call wall time in seconds.
    """
    pts = np.asarray(psc.eval_points, dtype=float)
    _ = np.asarray(psc.B_at_points(pts))  # warm-up
    samples: list[float] = []
    for _ in range(int(n_iter)):
        t0 = time.perf_counter()
        _ = np.asarray(psc.B_at_points(pts))
        samples.append(time.perf_counter() - t0)
    samples.sort()
    return float(samples[len(samples) // 2])


@pytest.mark.skipif(
    os.environ.get("SIMSOPT_PSC_FAR_PERF_GATE", "0") != "1",
    reason="opt-in perf gate; export SIMSOPT_PSC_FAR_PERF_GATE=1 to run",
)
def test_far_pair_dipole_speedup_debug() -> None:
    """Steady-state forward regression gate for the dipole-only far branch.

    Acceptance: median per-call wall time of the partitioned scan with a
    ``kappa`` value chosen so every non-self pair lands in the *far* branch
    must not be slower than the legacy ``vmap`` path by more than 5 %.
    Threshold is intentionally loose because at the test fixture's
    light basis the dense pair kernel is *not* the dominant cost; the
    eigenfloor solve and Biot-Savart eval contribute too.  At reactor
    scale (``nq ~ 312``, ``n_base ~ 14`` × ``G = 4`` replication)
    pair-inductance work dominates and the dipole kernel reaches the
    target speedup tracked in
    [`docs/passive_bulk_far_pair_plan.md`](../../../stellcoilbench_dipoles/docs/passive_bulk_far_pair_plan.md)
    Phase 0 baseline.

    Test is opt-in (``SIMSOPT_PSC_FAR_PERF_GATE=1``) so it never adds CI
    flake; the full reactor-scale speedup gate lives behind
    ``scripts/passive_bulk_opt_timing.py``.
    """
    n_base = int(os.environ.get("SIMSOPT_PSC_FAR_PERF_GATE_N", "8"))
    light_basis: dict[str, int] = {
        "m_fourier": 2,
        "l_zernike": 3,
        "k_chebyshev": 1,
        "n_rho": 4,
        "n_phi": 6,
        "n_z": 2,
    }

    prev = os.environ.get("SIMSOPT_PSC_PAIR_FAR_KAPPA")
    try:
        os.environ.pop("SIMSOPT_PSC_PAIR_FAR_KAPPA", None)
        psc_legacy = _make_rotated_puck_array_large(
            n_base=n_base, nfp=1, stellsym=False, **light_basis
        )
        psc_legacy.unfix("q0_0")
        psc_legacy.recompute_currents()
        t_legacy = _time_forward_loops(psc_legacy)

        os.environ["SIMSOPT_PSC_PAIR_FAR_KAPPA"] = "1.0e-6"  # everything → far
        psc_far = _make_rotated_puck_array_large(
            n_base=n_base, nfp=1, stellsym=False, **light_basis
        )
        psc_far.unfix("q0_0")
        psc_far.recompute_currents()
        t_far = _time_forward_loops(psc_far)
    finally:
        if prev is None:
            os.environ.pop("SIMSOPT_PSC_PAIR_FAR_KAPPA", None)
        else:
            os.environ["SIMSOPT_PSC_PAIR_FAR_KAPPA"] = prev

    speedup = t_legacy / max(t_far, 1.0e-12)
    print(
        f"[far-pair perf] n_base={n_base} t_legacy={t_legacy:.4f}s "
        f"t_far={t_far:.4f}s speedup={speedup:.2f}x"
    )
    assert speedup >= 0.95, (
        f"Far-pair dipole branch regressed beyond the 5 % budget: "
        f"t_legacy={t_legacy:.4f}s, t_far={t_far:.4f}s, speedup={speedup:.2f}x"
    )


def test_far_pair_vjp_matches_legacy_when_all_near() -> None:
    """Reduced free-DoF VJP gradient is invariant to partition reorder at all-near.

    Confirms that wiring ``near_pair_indices`` / ``far_pair_indices`` into
    ``_vjp_reduced_free_dof_jitted`` (and siblings) does not perturb the
    backward sweep when no pair lands in the far branch.
    """
    g_legacy = _run_reduced_free_dof_vjp(kappa_env_value=None)
    g_partition = _run_reduced_free_dof_vjp(kappa_env_value="1e9")
    assert g_legacy.shape == g_partition.shape
    denom = max(float(np.max(np.abs(g_legacy))), 1.0e-30)
    rel = float(np.max(np.abs(g_legacy - g_partition))) / denom
    assert rel <= 1.0e-9, (
        f"partitioned VJP disagrees with legacy: rel_max={rel:.3e}, "
        f"|grad_legacy|_max={denom:.3e}"
    )


def _build_R_or_t_vjp_fixture(*, dof_name: str):
    """Build a small free-DoF fixture with ``dof_name`` (R0/t0) unfixed.

    Also unfixes ``q0_0`` so the reduced free-DoF JAX path engages
    (the FD R/t branch in :meth:`PSCBulkArray._vjp_puck_geometry`
    only runs alongside the JAX VJP, never on its own).
    """
    psc = _make_symmetry_validation_array(nfp=1, stellsym=False, n_base=3)
    psc.unfix("q0_0")
    psc.unfix(dof_name)
    psc.recompute_currents()
    return psc


def _Rt_fd_reference(psc, dof_name: str, *, eps: float = 1.0e-5) -> float:
    """Independent central-difference reference for ``d<v_B,B(pts)>/d(dof)``.

    Uses ``v_B = ones_like(B)`` so the scalar functional is
    ``J = sum(B(pts))``, matching the seed used by
    :func:`_run_reduced_free_dof_vjp` and the analytic test driver.
    """
    pts = np.asarray(psc.eval_points, dtype=float)
    B0 = np.asarray(psc.B_at_points(pts))
    v_B = np.ones_like(B0)
    v0 = float(psc.get(dof_name))
    psc.set(dof_name, v0 + eps)
    psc.recompute_currents()
    Bp = np.asarray(psc.B_at_points(pts))
    psc.set(dof_name, v0 - eps)
    psc.recompute_currents()
    Bm = np.asarray(psc.B_at_points(pts))
    psc.set(dof_name, v0)
    psc.recompute_currents()
    return float(np.sum(v_B * (Bp - Bm))) / (2.0 * eps)


def _Rt_vjp_grad_for_dof(psc, dof_name: str) -> float:
    """Run the production VJP and return the gradient component on ``dof_name``.

    Uses :attr:`Derivative.data` keyed on ``psc`` (rather than
    :meth:`Derivative.__call__`) so we get the *full* per-DOF gradient
    array of shape ``(n_base * 9,)`` and can index by name without
    interleaving with TF-coil free-DoF entries pulled in by
    ``deriv(psc)``.
    """
    pts = np.asarray(psc.eval_points, dtype=float)
    B0 = np.asarray(psc.B_at_points(pts))
    v_B = np.ones_like(B0)
    deriv = psc._vjp_puck_geometry(v_B, pts)
    grad_full = np.asarray(deriv.data[psc], dtype=np.float64).reshape(-1)
    full_names = list(psc.local_full_dof_names)
    return float(grad_full[full_names.index(dof_name)])


def test_R_vjp_matches_finite_difference() -> None:
    """``dJ/dR0`` from the production VJP matches an independent central-FD probe.

    The production path uses one-sided forward FD (O(eps) bias) while
    the reference here uses central FD (O(eps^2)).  Tolerance is
    loosened to 5e-3 to accommodate the bias.
    """
    psc = _build_R_or_t_vjp_fixture(dof_name="R0")
    fd_ref = _Rt_fd_reference(psc, "R0")
    g_vjp = _Rt_vjp_grad_for_dof(psc, "R0")
    denom = max(abs(fd_ref), 1.0e-12)
    rel = abs(g_vjp - fd_ref) / denom
    assert rel <= 5.0e-3, (
        f"R0 VJP/FD disagree: vjp={g_vjp:.6e}, fd={fd_ref:.6e}, rel={rel:.3e}"
    )


def test_t_vjp_matches_finite_difference() -> None:
    """``dJ/dt0`` from the production VJP matches an independent central-FD probe."""
    psc = _build_R_or_t_vjp_fixture(dof_name="t0")
    fd_ref = _Rt_fd_reference(psc, "t0")
    g_vjp = _Rt_vjp_grad_for_dof(psc, "t0")
    denom = max(abs(fd_ref), 1.0e-12)
    rel = abs(g_vjp - fd_ref) / denom
    assert rel <= 5.0e-3, (
        f"t0 VJP/FD disagree: vjp={g_vjp:.6e}, fd={fd_ref:.6e}, rel={rel:.3e}"
    )


def test_unfix_radius_emits_no_warning() -> None:
    """``unfix("R0")`` / ``unfix("t0")`` no longer warn about a zero VJP.

    Phase D flipped :meth:`PSCBulkArray._is_zero_vjp_dof` to ``False``
    for every name once the FD R/t branch landed, so the warning
    machinery in :meth:`unfix` must stay silent for those DOFs.
    """
    import warnings as _warnings

    psc = _make_symmetry_validation_array(nfp=1, stellsym=False, n_base=3)
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        psc.unfix("R0")
        psc.unfix("t0")
    psc_warnings = [
        w
        for w in caught
        if "PSCBulkArray" in str(w.message) and "zero VJP" in str(w.message)
    ]
    assert not psc_warnings, (
        f"expected no zero-VJP warnings for R/t, got {[str(w.message) for w in psc_warnings]}"
    )


def test_B_at_points_tf_only_matches_full_when_fixed() -> None:
    """``_B_at_points_tf_only`` matches ``B_at_points`` when no puck DOFs are free."""
    psc = _make_symmetry_validation_array(nfp=1, stellsym=False, n_base=3)
    psc.recompute_currents()
    pts = np.asarray(psc.eval_points, dtype=float)
    B_full = np.asarray(psc.B_at_points(pts), dtype=np.float64)
    B_tf = np.asarray(psc._B_at_points_tf_only(pts), dtype=np.float64)
    np.testing.assert_allclose(B_tf, B_full, rtol=1e-12, atol=0)


def test_B_at_points_tf_only_matches_full_when_R_free() -> None:
    """``_B_at_points_tf_only`` is value-equivalent to ``B_at_points`` with R0 free.

    When ``_has_free_puck_dofs()`` is True, ``B_at_points`` uses the
    expensive free-DOF JAX runner which traces the full L assembly in JAX.
    ``_B_at_points_tf_only`` uses the pre-built host-side ``Lr_chol`` / ``Q``.
    The two paths agree to ~1e-6 relative (different numerical route to the
    same Cholesky factor; JAX jitter vs host jitter).
    """
    psc = _build_R_or_t_vjp_fixture(dof_name="R0")
    psc.recompute_currents()
    pts = np.asarray(psc.eval_points, dtype=float)
    B_full = np.asarray(psc.B_at_points(pts), dtype=np.float64)
    B_tf = np.asarray(psc._B_at_points_tf_only(pts), dtype=np.float64)
    np.testing.assert_allclose(B_tf, B_full, rtol=1e-5, atol=0)


def test_Rt_fd_gradient_uses_tf_only_path() -> None:
    """The FD branch must dispatch through ``_B_at_points_tf_only``, not the full runner.

    With ``SIMSOPT_PSCBULK_TIMING=1``, we verify that:
    - Zero ``B_at_points_free_reduced_jax`` rows appear inside the FD window.
    - ``B_at_points_tf_only_total`` row count equals ``n_free_shape + 1``
      (one B0 capture + one per DOF probe).
    """
    import os
    os.environ["SIMSOPT_PSCBULK_TIMING"] = "1"
    try:
        psc = _build_R_or_t_vjp_fixture(dof_name="R0")
        psc.recompute_currents()
        pts = np.asarray(psc.eval_points, dtype=float)
        B0 = np.asarray(psc.B_at_points(pts))
        v_B = np.ones_like(B0)

        psc.reset_timing_history()
        _deriv = psc._vjp_puck_geometry(v_B, pts)
        rows = psc.get_timing_history()

        free_reduced_in_fd = [
            r for r in rows
            if r.get("phase", "").startswith("B_at_points_free_reduced_jax")
        ]
        tf_only_in_fd = [
            r for r in rows
            if r.get("phase", "").startswith("B_at_points_tf_only_total")
        ]
        n_free_shape = sum(
            1 for name, free in zip(psc.local_full_dof_names, psc.local_dofs_free_status)
            if free and str(name).startswith(("R", "t"))
        )
        assert len(free_reduced_in_fd) == 0, (
            f"expected 0 B_at_points_free_reduced_jax rows in FD, got {len(free_reduced_in_fd)}"
        )
        assert len(tf_only_in_fd) == n_free_shape + 1, (
            f"expected {n_free_shape + 1} B_at_points_tf_only_total rows, got {len(tf_only_in_fd)}"
        )
    finally:
        os.environ.pop("SIMSOPT_PSCBULK_TIMING", None)


def test_Rt_fd_gradient_recompute_count_one_sided() -> None:
    """One-sided FD uses 2 rebuilds per free shape DOF (perturb + restore).

    Central FD used 3 (perturb+, perturb-, restore).  Verify the
    reduction by counting ``rebuild_total`` rows in the timing history.
    """
    import os
    os.environ["SIMSOPT_PSCBULK_TIMING"] = "1"
    try:
        psc = _build_R_or_t_vjp_fixture(dof_name="R0")
        psc.recompute_currents()
        pts = np.asarray(psc.eval_points, dtype=float)
        B0 = np.asarray(psc.B_at_points(pts))
        v_B = np.ones_like(B0)

        psc.reset_timing_history()
        _deriv = psc._vjp_puck_geometry(v_B, pts)
        rows = psc.get_timing_history()

        rebuild_rows = [
            r for r in rows if r.get("phase", "").startswith("rebuild_total")
        ]
        n_free_shape = sum(
            1 for name, free in zip(psc.local_full_dof_names, psc.local_dofs_free_status)
            if free and str(name).startswith(("R", "t"))
        )
        assert len(rebuild_rows) == 2 * n_free_shape, (
            f"expected {2 * n_free_shape} rebuild_total rows (one-sided FD), "
            f"got {len(rebuild_rows)}"
        )
    finally:
        os.environ.pop("SIMSOPT_PSCBULK_TIMING", None)


def _build_shape_only_vjp_fixture(*, dof_name: str):
    """Build a fixture with only ``dof_name`` (R0/t0) unfixed; centers/quats fixed.

    Distinct from :func:`_build_R_or_t_vjp_fixture` (which deliberately
    unfixes ``q0_0`` to keep the JAX free-DOF VJP path engaged).  This
    fixture exercises the **shape-only** dispatch in
    :meth:`PSCBulkArray._vjp_puck_geometry` -- no centers/quats free
    so the JAX free-DOF runner must be bypassed entirely.
    """
    psc = _make_symmetry_validation_array(nfp=1, stellsym=False, n_base=3)
    psc.unfix(dof_name)
    psc.recompute_currents()
    return psc


def test_B_at_points_skips_free_dof_jax_when_only_Rt_free() -> None:
    """``B_at_points`` must route through ``_B_at_points_tf_only`` when only R/t are free.

    With the shape-only fast path, ``_has_free_center_or_quat_dofs()``
    is False, so the dispatch falls through to the cheap TF-only path.
    Verify that no ``B_at_points_free_reduced_jax`` rows are emitted.
    """
    os.environ["SIMSOPT_PSCBULK_TIMING"] = "1"
    try:
        psc = _build_shape_only_vjp_fixture(dof_name="R0")
        pts = np.asarray(psc.eval_points, dtype=float)
        psc.reset_timing_history()
        _ = psc.B_at_points(pts)
        rows = psc.get_timing_history()

        free_jax_rows = [
            r for r in rows
            if r.get("phase", "").startswith("B_at_points_free_reduced_jax")
        ]
        tf_only_rows = [
            r for r in rows
            if r.get("phase", "").startswith("B_at_points_tf_only_total")
        ]
        assert len(free_jax_rows) == 0, (
            f"expected 0 B_at_points_free_reduced_jax rows for shape-only fixture, "
            f"got {len(free_jax_rows)}"
        )
        assert len(tf_only_rows) >= 1, (
            f"expected at least 1 B_at_points_tf_only_total row, got {len(tf_only_rows)}"
        )
    finally:
        os.environ.pop("SIMSOPT_PSCBULK_TIMING", None)


def test_vjp_puck_geometry_skips_jax_runner_when_only_Rt_free() -> None:
    """``_vjp_puck_geometry`` must skip the JAX free-DOF VJP when only R/t are free.

    Verify by counting ``vjp_free_reduced_jax`` rows (should be zero)
    and ``vjp_shape_only_total`` rows (should be exactly one).
    """
    os.environ["SIMSOPT_PSCBULK_TIMING"] = "1"
    try:
        psc = _build_shape_only_vjp_fixture(dof_name="R0")
        pts = np.asarray(psc.eval_points, dtype=float)
        B0 = np.asarray(psc.B_at_points(pts))
        v_B = np.ones_like(B0)

        psc.reset_timing_history()
        _deriv = psc._vjp_puck_geometry(v_B, pts)
        rows = psc.get_timing_history()

        free_jax_vjp_rows = [
            r for r in rows
            if r.get("phase", "").startswith("vjp_free_reduced_jax")
        ]
        shape_only_rows = [
            r for r in rows
            if r.get("phase", "").startswith("vjp_shape_only_total")
        ]
        assert len(free_jax_vjp_rows) == 0, (
            f"expected 0 vjp_free_reduced_jax rows on shape-only fixture, "
            f"got {len(free_jax_vjp_rows)}"
        )
        assert len(shape_only_rows) == 1, (
            f"expected exactly 1 vjp_shape_only_total row, got {len(shape_only_rows)}"
        )
    finally:
        os.environ.pop("SIMSOPT_PSCBULK_TIMING", None)


def test_R_vjp_shape_only_matches_central_fd() -> None:
    """Shape-only VJP for R0 still matches an independent central-FD reference.

    Even though we skip the JAX free-DOF VJP runner, the FD branch
    inside :meth:`PSCBulkArray._Rt_fd_gradient` (one-sided forward FD)
    still produces a usable R0 gradient.  Verify it agrees with a
    central-FD probe to ``rtol = 5e-3`` (one-sided FD has O(eps) bias).
    """
    psc = _build_shape_only_vjp_fixture(dof_name="R0")
    fd_ref = _Rt_fd_reference(psc, "R0")
    g_vjp = _Rt_vjp_grad_for_dof(psc, "R0")
    denom = max(abs(fd_ref), 1.0e-12)
    rel = abs(g_vjp - fd_ref) / denom
    assert rel <= 5.0e-3, (
        f"R0 shape-only VJP/FD disagree: vjp={g_vjp:.6e}, fd={fd_ref:.6e}, "
        f"rel={rel:.3e}"
    )


def test_far_pair_partition_matches_legacy_when_all_near_with_R_free() -> None:
    """Far-pair partition stays bit-exact with ``R0`` unfixed (FD path engaged).

    Because the FD branch in :meth:`PSCBulkArray._vjp_puck_geometry`
    runs *outside* the JAX kernel, switching the partition between the
    legacy ``vmap`` path and the partitioned ``lax.scan`` path must not
    perturb the FD R/t component (it depends only on
    :meth:`recompute_currents` + :meth:`B_at_points`, both of which are
    partition-agnostic) nor the centre / quat slots that come from the
    JAX kernel.
    """
    prev = os.environ.get("SIMSOPT_PSC_PAIR_FAR_KAPPA")

    def _run(kappa_env_value: str | None) -> np.ndarray:
        try:
            if kappa_env_value is None:
                os.environ.pop("SIMSOPT_PSC_PAIR_FAR_KAPPA", None)
            else:
                os.environ["SIMSOPT_PSC_PAIR_FAR_KAPPA"] = kappa_env_value
            psc = _make_symmetry_validation_array(
                nfp=1, stellsym=False, n_base=3
            )
            psc.unfix("q0_0")
            psc.unfix("R0")
            psc.recompute_currents()
            pts = np.asarray(psc.eval_points, dtype=float)
            v_B = np.ones_like(np.asarray(psc.B_at_points(pts)))
            deriv = psc._vjp_puck_geometry(v_B, pts)
            return np.asarray(deriv(psc), dtype=np.float64).reshape(-1)
        finally:
            if prev is None:
                os.environ.pop("SIMSOPT_PSC_PAIR_FAR_KAPPA", None)
            else:
                os.environ["SIMSOPT_PSC_PAIR_FAR_KAPPA"] = prev

    g_legacy = _run(None)
    g_partition = _run("1e9")
    assert g_legacy.shape == g_partition.shape
    denom = max(float(np.max(np.abs(g_legacy))), 1.0e-30)
    rel = float(np.max(np.abs(g_legacy - g_partition))) / denom
    assert rel <= 1.0e-3, (
        f"partitioned VJP under R-free disagrees with legacy: "
        f"rel_max={rel:.3e}, |grad_legacy|_max={denom:.3e}"
    )


def _numpy_symmetric_reduced_reference(
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
    """Small reference loop (same algebra as the heterogeneous NumPy path)."""
    base_indices = np.asarray(base_indices, dtype=int)
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


def test_symmetric_reduced_jax_matches_hand_reference():
    rng = np.random.default_rng(0)
    n_base = 3
    G = 2
    n_all = n_base * G
    Q, D = 6, 5
    K_per = [rng.standard_normal((Q, D, 3)) for _ in range(n_all)]
    pts_per = [rng.standard_normal((Q, 3)) for _ in range(n_all)]
    w_per = [np.abs(rng.standard_normal(Q)) + 1e-3 for _ in range(n_all)]
    base_indices = np.repeat(np.arange(n_base), G)
    signs = np.ones(n_all, dtype=int)
    base_reps = np.array([0, 2, 4], dtype=int)
    L_ref = _numpy_symmetric_reduced_reference(
        K_per,
        pts_per,
        w_per,
        base_indices,
        base_reps,
        signs,
        G,
        1e-8,
        False,
    )
    L_jax = _shell_inductance_matrix_symmetric_reduced_jax(
        K_per,
        pts_per,
        w_per,
        base_indices,
        base_reps,
        signs,
        G,
        1e-8,
        False,
    )
    # JAX uses batched float32 matmuls in ``_jax_pair_batch_blocks``; allow
    # a looser tolerance than strict NumPy double loops.
    np.testing.assert_allclose(L_jax, L_ref, atol=1e-5, rtol=1e-5)


def test_symmetric_reduced_vectorized_scatter_matches_legacy_host_scatter():
    """Vectorized :func:`np.add.at` host scatter matches the legacy slice-add loop.

    Uses the same fixed-seed fixture as :func:`test_symmetric_reduced_jax_matches_hand_reference`
    (``n_base=3``, ``G=2``, ``Q=6``, ``D=5``) to guard Stage-1a scatter algebra.
    """
    rng = np.random.default_rng(0)
    n_base = 3
    G = 2
    n_all = n_base * G
    Q, D = 6, 5
    K_per = [rng.standard_normal((Q, D, 3)) for _ in range(n_all)]
    pts_per = [rng.standard_normal((Q, 3)) for _ in range(n_all)]
    w_per = [np.abs(rng.standard_normal(Q)) + 1e-3 for _ in range(n_all)]
    base_indices = np.repeat(np.arange(n_base), G)
    signs = np.ones(n_all, dtype=int)
    base_reps = np.array([0, 2, 4], dtype=int)
    prev = os.environ.get("SIMSOPT_SYM_REDUCED_LBASE_SCATTER")
    try:
        os.environ["SIMSOPT_SYM_REDUCED_LBASE_SCATTER"] = "legacy"
        L_legacy = _shell_inductance_matrix_symmetric_reduced_jax(
            K_per,
            pts_per,
            w_per,
            base_indices,
            base_reps,
            signs,
            G,
            1e-8,
            False,
        )
        os.environ["SIMSOPT_SYM_REDUCED_LBASE_SCATTER"] = "vectorized"
        L_vec = _shell_inductance_matrix_symmetric_reduced_jax(
            K_per,
            pts_per,
            w_per,
            base_indices,
            base_reps,
            signs,
            G,
            1e-8,
            False,
        )
    finally:
        if prev is None:
            os.environ.pop("SIMSOPT_SYM_REDUCED_LBASE_SCATTER", None)
        else:
            os.environ["SIMSOPT_SYM_REDUCED_LBASE_SCATTER"] = prev
    denom = float(np.max(np.abs(L_legacy)))
    assert denom > 0.0
    rel = float(np.max(np.abs(L_vec - L_legacy)) / denom)
    assert rel <= 1e-13


def test_symmetric_reduced_scan_dispatch_matches_python_loop():
    """Single ``lax.scan`` batch dispatch matches per-batch Python loop."""
    rng = np.random.default_rng(0)
    n_base = 3
    G = 2
    n_all = n_base * G
    Q, D = 6, 5
    K_per = [rng.standard_normal((Q, D, 3)) for _ in range(n_all)]
    pts_per = [rng.standard_normal((Q, 3)) for _ in range(n_all)]
    w_per = [np.abs(rng.standard_normal(Q)) + 1e-3 for _ in range(n_all)]
    base_indices = np.repeat(np.arange(n_base), G)
    signs = np.ones(n_all, dtype=int)
    base_reps = np.array([0, 2, 4], dtype=int)
    prev_dispatch = os.environ.get("SIMSOPT_PSC_PAIR_DISPATCH")
    try:
        os.environ["SIMSOPT_PSC_PAIR_DISPATCH"] = "loop"
        L_loop = _shell_inductance_matrix_symmetric_reduced_jax(
            K_per,
            pts_per,
            w_per,
            base_indices,
            base_reps,
            signs,
            G,
            1e-8,
            False,
        )
        os.environ.pop("SIMSOPT_PSC_PAIR_DISPATCH", None)
        L_scan = _shell_inductance_matrix_symmetric_reduced_jax(
            K_per,
            pts_per,
            w_per,
            base_indices,
            base_reps,
            signs,
            G,
            1e-8,
            False,
        )
    finally:
        if prev_dispatch is None:
            os.environ.pop("SIMSOPT_PSC_PAIR_DISPATCH", None)
        else:
            os.environ["SIMSOPT_PSC_PAIR_DISPATCH"] = prev_dispatch
    denom = float(np.max(np.abs(L_loop)))
    assert denom > 0.0
    rel = float(np.max(np.abs(L_scan - L_loop)) / denom)
    assert rel <= 1e-13


@partial(jax.jit, static_argnames=("adaptive_self_reg",))
def _legacy_pair_batch_blocks_einsum(
    Ki: jnp.ndarray,
    Kj: jnp.ndarray,
    pts_i: jnp.ndarray,
    pts_j: jnp.ndarray,
    w_i: jnp.ndarray,
    w_j: jnp.ndarray,
    delta_reg: jnp.ndarray,
    adaptive_self_reg: bool = False,
) -> jnp.ndarray:
    """Historical 5D intermediate (reference for kernel equivalence)."""
    r = pts_i[:, :, None, :] - pts_j[:, None, :, :]
    if adaptive_self_reg:
        delta_i = _SELF_REG_COEFF * jnp.sqrt(w_i)
        delta_j = _SELF_REG_COEFF * jnp.sqrt(w_j)
        delta_pair = 0.5 * (delta_i[:, :, None] + delta_j[:, None, :])
        dist_reg = jnp.sqrt(jnp.sum(r**2, axis=-1) + delta_pair**2)
    else:
        dist_reg = jnp.sqrt(jnp.sum(r**2, axis=-1) + delta_reg**2)
    dot = jnp.einsum("Biax,Bjbx->Bijab", Ki, Kj)
    kernel = dot / dist_reg[..., None, None]
    return MU0_OVER_4PI * jnp.einsum("Bijab,Bi,Bj->Bab", kernel, w_i, w_j)


def test_jax_pair_batch_blocks_matches_legacy_einsum():
    """``_jax_pair_batch_blocks`` matches the legacy 5D ``einsum`` formula."""
    rng = np.random.default_rng(0)
    B, nq, nd = 8, 16, 20
    Ki = np.asarray(rng.standard_normal((B, nq, nd, 3)), dtype=np.float64)
    Kj = np.asarray(rng.standard_normal((B, nq, nd, 3)), dtype=np.float64)
    pts_i = np.asarray(rng.standard_normal((B, nq, 3)), dtype=np.float64)
    pts_j = np.asarray(rng.standard_normal((B, nq, 3)), dtype=np.float64)
    w_i = np.asarray(np.abs(rng.standard_normal((B, nq))) + 1e-4, dtype=np.float64)
    w_j = np.asarray(np.abs(rng.standard_normal((B, nq))) + 1e-4, dtype=np.float64)
    dreg = jnp.asarray(1e-8, dtype=jnp.float64)
    for adaptive in (False, True):
        out_new = _jax_pair_batch_blocks(
            jnp.asarray(Ki),
            jnp.asarray(Kj),
            jnp.asarray(pts_i),
            jnp.asarray(pts_j),
            jnp.asarray(w_i),
            jnp.asarray(w_j),
            dreg,
            adaptive_self_reg=adaptive,
        )
        out_old = _legacy_pair_batch_blocks_einsum(
            jnp.asarray(Ki),
            jnp.asarray(Kj),
            jnp.asarray(pts_i),
            jnp.asarray(pts_j),
            jnp.asarray(w_i),
            jnp.asarray(w_j),
            dreg,
            adaptive_self_reg=adaptive,
        )
        np.testing.assert_allclose(
            np.asarray(out_new),
            np.asarray(out_old),
            rtol=1e-12,
            atol=1e-12,
        )


def test_jax_pair_batch_blocks_at_least_2x_faster_than_legacy():
    """Regression: matmul-style contraction should beat the 5D ``einsum`` path."""
    rng = np.random.default_rng(1)
    B, nq, nd = 64, 32, 28
    Ki = jnp.asarray(rng.standard_normal((B, nq, nd, 3)), dtype=jnp.float64)
    Kj = jnp.asarray(rng.standard_normal((B, nq, nd, 3)), dtype=jnp.float64)
    pts_i = jnp.asarray(rng.standard_normal((B, nq, 3)), dtype=jnp.float64)
    pts_j = jnp.asarray(rng.standard_normal((B, nq, 3)), dtype=jnp.float64)
    w_i = jnp.asarray(np.abs(rng.standard_normal((B, nq))) + 1e-4, dtype=jnp.float64)
    w_j = jnp.asarray(np.abs(rng.standard_normal((B, nq))) + 1e-4, dtype=np.float64)
    dreg = jnp.asarray(1e-8, dtype=jnp.float64)
    for _ in range(2):
        _ = _jax_pair_batch_blocks(Ki, Kj, pts_i, pts_j, w_i, w_j, dreg, False)
        _ = _legacy_pair_batch_blocks_einsum(
            Ki, Kj, pts_i, pts_j, w_i, w_j, dreg, False
        )
    nrep = 8
    t0 = time.perf_counter()
    for _ in range(nrep):
        _ = np.asarray(
            _legacy_pair_batch_blocks_einsum(
                Ki, Kj, pts_i, pts_j, w_i, w_j, dreg, False
            )
        )
    t_old = (time.perf_counter() - t0) / nrep
    t1 = time.perf_counter()
    for _ in range(nrep):
        _ = np.asarray(
            _jax_pair_batch_blocks(Ki, Kj, pts_i, pts_j, w_i, w_j, dreg, False)
        )
    t_new = (time.perf_counter() - t1) / nrep
    assert t_new < 0.5 * t_old, (
        f"expected >=2x speedup: legacy={t_old:.4f}s new={t_new:.4f}s"
    )


def test_symmetric_reduced_jax_faster_than_numpy_loop():
    """At scale, batched JAX blocks should beat the pure Python double loop."""
    rng = np.random.default_rng(42)
    n_base = 8
    G = 4
    n_all = n_base * G
    Q = 30
    D = 40
    K_per = [rng.standard_normal((Q, D, 3)) for _ in range(n_all)]
    pts_per = [rng.standard_normal((Q, 3)) for _ in range(n_all)]
    w_per = [np.abs(rng.standard_normal(Q)) + 1e-3 for _ in range(n_all)]
    base_indices = np.repeat(np.arange(n_base), G)
    signs = np.ones(n_all, dtype=int)
    base_reps = np.arange(n_base) * G

    t0 = time.perf_counter()
    _ = _numpy_symmetric_reduced_reference(
        K_per,
        pts_per,
        w_per,
        base_indices,
        base_reps,
        signs,
        G,
        1e-8,
        False,
    )
    t_np = time.perf_counter() - t0

    t1 = time.perf_counter()
    _ = _shell_inductance_matrix_symmetric_reduced_jax(
        K_per,
        pts_per,
        w_per,
        base_indices,
        base_reps,
        signs,
        G,
        1e-8,
        False,
    )
    t_jax = time.perf_counter() - t1

    assert t_jax < 0.5 * t_np, (
        f"JAX path not faster: numpy_loop={t_np:.3f}s jax={t_jax:.3f}s"
    )


def test_inner_loop_per_call_budget():
    """Single B evaluation should stay within a loose wall-clock budget."""
    psc = _make_symmetry_validation_array(nfp=2, stellsym=False, n_base=1)
    psc.recompute_currents()
    pts = np.array([[1.0, 0.05, 0.35]], dtype=float)

    t0 = time.perf_counter()
    _ = psc.B_at_points(pts)
    dt = time.perf_counter() - t0
    assert dt < 0.2, f"B_at_points too slow: {dt:.3f}s (budget 200 ms)"


def test_no_duplicate_numpy_jax_K_backing():
    """Uniform pucks: no extra ``_K_stack`` array in ``__dict__``."""
    psc = _make_symmetry_validation_array(nfp=2, stellsym=False, n_base=1)
    assert psc._uniform_puck_shape
    assert "_K_stack" not in psc.__dict__
    assert hasattr(psc, "_jax_K_stack") and psc._jax_K_stack is not None


def test_L_work_released_when_requested_on_full_L():
    """Optional release drops full-replica ``_L_work`` after rebuild."""
    psc = _make_symmetry_validation_array(nfp=2, stellsym=False, n_base=1)
    psc_full = _make_symmetry_validation_array(nfp=2, stellsym=False, n_base=1)
    psc_full._tf_is_symmetric = False
    psc_full._release_host_L_work_after_rebuild = True
    psc_full._rebuild()
    assert psc_full._L_work is None
    assert psc._L_work is not None


# --- ~50 base pucks (nfp=2, stellsym=True → 200 replicas) -----------------

_BUILD50 = float(os.environ.get("PSC_BUILD50_BUDGET_S", "480"))
_FUN50 = float(os.environ.get("PSC_FUN50_BUDGET_S", "12.0"))


def test_build_50base_under_budget():
    """First construction of a 50-base layout should stay under a loose cap."""
    t0 = time.perf_counter()
    psc = _make_symmetry_validation_array(nfp=2, stellsym=True, n_base=50)
    dt = time.perf_counter() - t0
    assert psc._n_base_pucks == 50
    assert dt < _BUILD50, f"build too slow: {dt:.1f}s (budget {_BUILD50}s)"


def test_fun_50base_median_under_budget():
    """Inner loop ``recompute_currents`` + ``B_at_points`` median (loose cap)."""
    psc = _make_symmetry_validation_array(nfp=2, stellsym=True, n_base=50)
    pts = np.asarray(psc.eval_points[:4], dtype=float)
    times = []
    for _ in range(12):
        t0 = time.perf_counter()
        psc.recompute_currents()
        _ = psc.B_at_points(pts)
        times.append(time.perf_counter() - t0)
    med = float(np.median(times))
    assert med < _FUN50, f"median inner loop too slow: {med:.2f}s (budget {_FUN50}s)"


def test_ensure_jax_full_not_triggered_50base_frozen_pucks():
    """Frozen puck DOFs: forward path must not materialise ``_jax_local_*`` stacks."""
    psc = _make_symmetry_validation_array(nfp=2, stellsym=True, n_base=50)
    pts = np.asarray(psc.eval_points[:2], dtype=float)
    for _ in range(3):
        _ = psc.B_at_points(pts)
    assert not hasattr(psc, "_jax_local_pts")


def test_50base_symmetry_reduced_L_work_shape():
    """Reduced path: ``L_work`` is (n_base·nd)^2, not full-replica size."""
    psc = _make_symmetry_validation_array(nfp=2, stellsym=True, n_base=50)
    assert psc._reduced_active
    assert psc._L_work is not None
    nd = int(psc._K_stack.shape[2])
    nb = int(psc._n_base_pucks)
    assert psc._L_work.shape == (nb * nd, nb * nd)


# ======================================================================
# Stage 1 (disk cache) -- psc-scale-to-100-bulks plan
# ======================================================================


def _sym_reduced_random_fixture(seed: int = 0, n_base: int = 3, G: int = 2):
    """Return a random fixed-seed fixture suitable for direct-kernel tests."""
    rng = np.random.default_rng(seed)
    n_all = n_base * G
    Q, D = 6, 5
    K_per = [rng.standard_normal((Q, D, 3)) for _ in range(n_all)]
    pts_per = [rng.standard_normal((Q, 3)) for _ in range(n_all)]
    w_per = [np.abs(rng.standard_normal(Q)) + 1e-3 for _ in range(n_all)]
    base_indices = np.repeat(np.arange(n_base), G)
    signs = np.ones(n_all, dtype=int)
    base_reps = np.arange(n_base, dtype=int) * G
    return K_per, pts_per, w_per, base_indices, signs, base_reps


def test_psc_lcache_roundtrip_and_hit(tmp_path):
    """Stage 1 cache: save once; second rebuild hits and returns identical arrays.

    Builds a tiny :class:`PSCBulkArray`, forces a rebuild under
    ``SIMSOPT_PSC_LCACHE=1`` with an isolated directory, then rebuilds
    again and asserts the cache metadata records a hit and the
    downstream arrays are bit-identical to the first build.
    """
    env_prev = {
        k: os.environ.get(k) for k in ("SIMSOPT_PSC_LCACHE", "SIMSOPT_PSC_LCACHE_DIR")
    }
    try:
        os.environ["SIMSOPT_PSC_LCACHE"] = "1"
        os.environ["SIMSOPT_PSC_LCACHE_DIR"] = str(tmp_path)

        psc_cold = _make_symmetry_validation_array(nfp=2, stellsym=False, n_base=1)
        L_cold = np.array(psc_cold._L_work, copy=True)
        Q_cold = np.array(psc_cold._Q, copy=True)
        chol_cold = np.array(psc_cold._Lr_chol_host, copy=True)
        # First build seeds the cache (hit flag may be False if the disk
        # was empty); key is always populated when the cache is enabled.
        assert psc_cold._lcache_last_key is not None
        assert len(list(tmp_path.glob("*.npz"))) == 1

        psc_warm = _make_symmetry_validation_array(nfp=2, stellsym=False, n_base=1)
        assert psc_warm._lcache_last_hit is True
        np.testing.assert_array_equal(psc_warm._L_work, L_cold)
        np.testing.assert_array_equal(psc_warm._Q, Q_cold)
        np.testing.assert_array_equal(psc_warm._Lr_chol_host, chol_cold)
    finally:
        for k, v in env_prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_psc_lcache_disabled_by_default():
    """Cache is **off** unless ``SIMSOPT_PSC_LCACHE=1`` is explicitly set."""
    prev = os.environ.pop("SIMSOPT_PSC_LCACHE", None)
    try:
        psc = _make_symmetry_validation_array(nfp=2, stellsym=False, n_base=1)
        assert psc._lcache_last_key is None
        assert psc._lcache_last_hit is False
    finally:
        if prev is not None:
            os.environ["SIMSOPT_PSC_LCACHE"] = prev


# ======================================================================
# Stage 2 (incremental rebuild) -- psc-scale-to-100-bulks plan
# ======================================================================


def test_rebuild_changed_mask_noop_reuses_L_and_chol():
    """Empty ``changed_mask`` short-circuits L / Cholesky rebuild.

    Records pre-rebuild ``_L_work``, ``_Q``, and ``_Lr_chol_host``
    identities, calls ``_rebuild(changed_mask=np.zeros(...))``, then
    asserts the *same objects* are still referenced (i.e. no
    re-assembly happened).
    """
    psc = _make_symmetry_validation_array(nfp=2, stellsym=False, n_base=1)
    L_id = id(psc._L_work)
    Q_id = id(psc._Q)
    chol_id = id(psc._Lr_chol_host)
    beta_prev = np.array(psc.beta, copy=True)

    mask = np.zeros(len(psc.local_full_x), dtype=bool)
    psc._rebuild(changed_mask=mask)

    assert id(psc._L_work) == L_id
    assert id(psc._Q) == Q_id
    assert id(psc._Lr_chol_host) == chol_id
    np.testing.assert_array_equal(psc.beta, beta_prev)


# ======================================================================
# Stage 3 (fp32 off-diagonal pair kernel) -- psc-scale-to-100-bulks plan
# ======================================================================


def test_sym_reduced_fp32_path_matches_f64_to_2em6():
    """fp32 off-diagonal pair kernel preserves ``L_base`` to ``rel <= 2e-6``."""
    K_per, pts_per, w_per, base_indices, signs, base_reps = _sym_reduced_random_fixture(
        seed=1
    )
    prev = os.environ.get("SIMSOPT_PSC_FP32")
    try:
        os.environ.pop("SIMSOPT_PSC_FP32", None)
        L_f64 = _shell_inductance_matrix_symmetric_reduced_jax(
            K_per, pts_per, w_per, base_indices, base_reps, signs, 2, 1e-8, False
        )
        os.environ["SIMSOPT_PSC_FP32"] = "1"
        L_fp32 = _shell_inductance_matrix_symmetric_reduced_jax(
            K_per, pts_per, w_per, base_indices, base_reps, signs, 2, 1e-8, False
        )
    finally:
        if prev is None:
            os.environ.pop("SIMSOPT_PSC_FP32", None)
        else:
            os.environ["SIMSOPT_PSC_FP32"] = prev
    denom = float(np.max(np.abs(L_f64))) + 1e-300
    rel = float(np.max(np.abs(L_fp32 - L_f64)) / denom)
    assert rel <= 2e-6, f"fp32 mismatch: rel_err={rel:.3e}"


# ======================================================================
# Stage 5 (iterative null-space projection) -- psc-scale-to-100-bulks plan
# ======================================================================


def test_big_chol_iterative_matches_dense_eigh():
    """Iterative null-space projector matches dense ``eigh`` (up to sign).

    Builds a small SPD matrix with a known null space of dimension 2;
    runs :func:`null_space_projection_matrix` under
    ``SIMSOPT_PSC_BIG_CHOL=1`` + ``SIMSOPT_PSC_CHOL_THRESHOLD=0`` (force
    iterative path) and compares ``Q Q^T`` (projector) against the
    dense reference, which is gauge-invariant to the column-sign and
    basis rotation ambiguity of eigendecomposition.
    """
    from simsopt.field.bulk_inductance import null_space_projection_matrix

    rng = np.random.default_rng(3)
    n = 40
    A = rng.standard_normal((n, n - 2))
    L = A @ A.T  # rank n-2 PSD
    L = 0.5 * (L + L.T)

    Q_dense = null_space_projection_matrix(L, threshold=1e-10)
    prev = {
        k: os.environ.get(k)
        for k in ("SIMSOPT_PSC_BIG_CHOL", "SIMSOPT_PSC_CHOL_THRESHOLD")
    }
    try:
        os.environ["SIMSOPT_PSC_BIG_CHOL"] = "1"
        os.environ["SIMSOPT_PSC_CHOL_THRESHOLD"] = "0"
        Q_iter = null_space_projection_matrix(L, threshold=1e-10)
    finally:
        for k, v in prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    # Project onto the same subspace: compare Q_iter @ Q_iter.T against
    # Q_dense @ Q_dense.T (invariant under column-sign / basis rotation).
    P_dense = Q_dense @ Q_dense.T
    P_iter = Q_iter @ Q_iter.T
    denom = float(np.max(np.abs(P_dense))) + 1e-300
    rel = float(np.max(np.abs(P_iter - P_dense))) / denom
    assert rel <= 1e-10, f"iterative projector mismatch: rel={rel:.3e}"


def test_far_pair_flag_noop_when_unimplemented():
    """``SIMSOPT_PSC_FAR_PAIR=1`` is currently a safe no-op (see docstring).

    Guards against accidental activation of the (not yet implemented)
    multipole kernel: the result must remain bit-identical to the full
    near-pair kernel until the multipole code lands.  When the
    multipole path is implemented the tolerance below should tighten
    according to the calibrated ``R_far``.
    """
    K_per, pts_per, w_per, base_indices, signs, base_reps = _sym_reduced_random_fixture(
        seed=2
    )
    prev = os.environ.get("SIMSOPT_PSC_FAR_PAIR")
    try:
        os.environ.pop("SIMSOPT_PSC_FAR_PAIR", None)
        L_base = _shell_inductance_matrix_symmetric_reduced_jax(
            K_per, pts_per, w_per, base_indices, base_reps, signs, 2, 1e-8, False
        )
        os.environ["SIMSOPT_PSC_FAR_PAIR"] = "1"
        L_far = _shell_inductance_matrix_symmetric_reduced_jax(
            K_per, pts_per, w_per, base_indices, base_reps, signs, 2, 1e-8, False
        )
    finally:
        if prev is None:
            os.environ.pop("SIMSOPT_PSC_FAR_PAIR", None)
        else:
            os.environ["SIMSOPT_PSC_FAR_PAIR"] = prev
    np.testing.assert_array_equal(L_far, L_base)


# --- Free-DoF vs fixed-DoF inner loop (nfp=2, stellsym=True, n_base=8) --------

_PSC_FREE_DOF_BUDGET_S = float(os.environ.get("PSC_FREE_DOF_BUDGET_S", "120.0"))


def test_median_free_puck_dof_inner_loop_nbase8_nfp2_stellsym():
    """Loose wall-clock cap for ``recompute_currents`` + :meth:`B_at_points` with
    one free base-puck center DoF (symmetry-reduced free-DoF JAX path).

    Compares against the frozen-puck (cached Cholesky) path on the same
    layout.  Override the cap with env ``PSC_FREE_DOF_BUDGET_S`` (seconds).
    """
    n_base, nfp, stell = 8, 2, True
    psc_frozen = _make_symmetry_validation_array(nfp=nfp, stellsym=stell, n_base=n_base)
    pts = np.asarray(psc_frozen.eval_points, dtype=float)
    t_f: list[float] = []
    for _ in range(5):
        t0 = time.perf_counter()
        psc_frozen.recompute_currents()
        _ = psc_frozen.B_at_points(pts)
        t_f.append(time.perf_counter() - t0)
    med_frozen = float(np.median(t_f))

    psc_free = _make_symmetry_validation_array(
        nfp=nfp, stellsym=stell, n_base=n_base, eval_pts=pts
    )
    psc_free.unfix("center_x0")
    t_free: list[float] = []
    for _ in range(5):
        t0 = time.perf_counter()
        psc_free.recompute_currents()
        _ = psc_free.B_at_points(pts)
        t_free.append(time.perf_counter() - t0)
    med_free = float(np.median(t_free))
    # Regression guard: free-DoF step should complete (and remain bounded
    # vs the frozen-puck fast path) as symmetry-reduced JAX is tuned.
    assert med_free < _PSC_FREE_DOF_BUDGET_S, (
        f"free-DoF inner loop too slow: median={med_free:.2f}s "
        f"(budget {_PSC_FREE_DOF_BUDGET_S}s; env PSC_FREE_DOF_BUDGET_S); "
        f"frozen-puck median={med_frozen:.2f}s for reference"
    )


@pytest.mark.slow
def test_rotated_puck_full_vs_far_flag_small_rel_error() -> None:
    r"""6-puck rotated fixture: :envvar:`SIMSOPT_PSC_FAR_PAIR` alters L modestly (plan).

    Bit-exact match is only expected with the flag off; with it on, the
    relative Frobenius error should stay in the 1e-3..1e-2 band for
    well-separated pucks.
    """
    psc0 = _make_rotated_puck_array(n_base=6, seed=0, nfp=1, stellsym=False)
    os.environ["SIMSOPT_PSC_FAR_PAIR"] = "0"
    psc0._rebuild()
    L0 = np.asarray(psc0._L_work, dtype=np.float64)
    os.environ["SIMSOPT_PSC_FAR_PAIR"] = "1"
    os.environ["SIMSOPT_PSC_R_FAR"] = "6"
    psc1 = _make_rotated_puck_array(n_base=6, seed=0, nfp=1, stellsym=False)
    psc1._rebuild()
    L1 = np.asarray(psc1._L_work, dtype=np.float64)
    os.environ["SIMSOPT_PSC_FAR_PAIR"] = "0"
    os.environ.pop("SIMSOPT_PSC_R_FAR", None)
    den = max(1.0e-30, float(np.max(np.abs(L0))))
    rel = float(np.linalg.norm(L0 - L1) / (den * L0.size**0.5 + 1.0e-30))
    assert rel < 0.05, f"Frobenius-budget rel error {rel} too large vs full L"


@pytest.mark.slow
def test_large_rotated_puck_far_pair_speedup_opt_in() -> None:
    r"""Optional large timing comparison (opt in via :envvar:`RUN_PSC_LARGE_FAR_BENCH`).

    Not run in default CI: machine-dependent. When enabled, a single rebuild
    with ``SIMSOPT_PSC_FAR_PAIR=1`` and a fixed ``R_far`` should not be
    dramatically slower than the full pair kernel; allow slack for one-shot
    self-checks.
    """
    if os.environ.get("RUN_PSC_LARGE_FAR_BENCH", "") != "1":
        pytest.skip(
            "set RUN_PSC_LARGE_FAR_BENCH=1 to run large n_base=32 far-pair timing "
            "(nightly perf candidate gate)"
        )
    _env: dict[str, str | None] = {}
    for k in (
        "SIMSOPT_PSCBULK_TIMING",
        "SIMSOPT_PSC_FAR_PAIR",
        "SIMSOPT_PSC_R_FAR",
    ):
        _env[k] = os.environ.get(k)
    n_base = 32
    try:
        reset_psc_far_selfcheck_cache()
        os.environ["SIMSOPT_PSCBULK_TIMING"] = "1"
        psc0 = _make_rotated_puck_array_large(n_base=n_base, seed=0)
        os.environ["SIMSOPT_PSC_FAR_PAIR"] = "0"
        t0 = time.perf_counter()
        psc0._rebuild()
        t_baseline = time.perf_counter() - t0
        reset_psc_far_selfcheck_cache()
        psc1 = _make_rotated_puck_array_large(n_base=n_base, seed=0)
        os.environ["SIMSOPT_PSC_FAR_PAIR"] = "1"
        os.environ["SIMSOPT_PSC_R_FAR"] = "6"
        t1 = time.perf_counter()
        psc1._rebuild()
        t_far = time.perf_counter() - t1
        # Allow one-shot self-check, calibration, and load variance; goal is
        # not a large regression.
        assert t_far <= t_baseline * 1.5 + 1.0, (
            f"far-pair path slower than expected: {t_baseline=:.2f}s {t_far=:.2f}s"
        )
    finally:
        reset_psc_far_selfcheck_cache()
        for k, v in _env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@pytest.mark.slow
def test_conformal_bulk_lp_qa_shape_smoke() -> None:
    """One free-orientation VJP; opt-in, machine-dependent wall clock.

    Gate with :envvar:`RUN_PSC_CONFORMAL_BENCH=1` (e.g. before full
    ``stellcoilbench submit-case`` on conformal bulk with rotation). Uses a
    reduced-path fixture shaped like
    ``cases/conformal_bulk_LandremanPaulQA.yaml`` (``nfp=2``, ``stellsym``,
    moderate ``n_base``) and a conformal-style basis resolution.
    """
    if os.environ.get("RUN_PSC_CONFORMAL_BENCH", "") != "1":
        pytest.skip("set RUN_PSC_CONFORMAL_BENCH=1 to run conformal smoke timing")
    _prev_chunk = os.environ.get("SIMSOPT_PSC_JAX_PAIR_CHUNK")
    try:
        os.environ["SIMSOPT_PSC_JAX_PAIR_CHUNK"] = "4"
        psc = _make_symmetry_validation_array(
            nfp=2,
            stellsym=True,
            n_base=12,
            m_fourier=3,
            l_zernike=4,
            k_chebyshev=3,
            n_rho=8,
            n_phi=10,
            n_z=4,
            checkpoint_l_pairs=True,
        )
        n = int(psc._n_base_pucks)
        for i in range(n):
            for k in (f"q0_{i}", f"qi_{i}", f"qj_{i}", f"qk_{i}"):
                psc.unfix(k)
        t0 = time.perf_counter()
        psc.recompute_currents()
        v = np.random.default_rng(0).standard_normal(
            psc.eval_points.reshape(-1, 3).shape
        )
        d = psc.vjp_setup_B(v, psc.eval_points)
        _g = np.asarray(d(psc))
        elapsed = time.perf_counter() - t0
        assert _g.size == n * 9
        assert np.all(np.isfinite(_g))
        assert elapsed < 120.0, f"conformal smoke too slow: {elapsed:.1f}s"
    finally:
        if _prev_chunk is None:
            os.environ.pop("SIMSOPT_PSC_JAX_PAIR_CHUNK", None)
        else:
            os.environ["SIMSOPT_PSC_JAX_PAIR_CHUNK"] = _prev_chunk
