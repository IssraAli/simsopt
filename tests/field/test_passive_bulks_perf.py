"""
Slow / timing regression tests for passive-bulk performance work.

Run with ``pytest --runslow``.

Override budgets with env ``PSC_BUILD50_BUDGET_S`` / ``PSC_FUN50_BUDGET_S`` (seconds).
"""

from __future__ import annotations

import importlib.util
import os
import time
from pathlib import Path

import numpy as np
import pytest

from simsopt.field.bulk_inductance import (
    MU0_OVER_4PI,
    _SELF_REG_COEFF,
    _shell_inductance_matrix_symmetric_reduced_jax,
)

pytestmark = pytest.mark.slow

_spec = importlib.util.spec_from_file_location(
    "_tpb_perf",
    Path(__file__).resolve().parent / "test_passive_bulks.py",
)
_tpb = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_tpb)
_make_symmetry_validation_array = _tpb._make_symmetry_validation_array


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
