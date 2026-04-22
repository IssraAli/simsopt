# Copyright 2016-2020 HiddenSymmetries, MIT License
"""Unit tests for :mod:`simsopt.field.multipole_inductance` (far-pair moment kernel)."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("jax.numpy")

import jax.numpy as jnp  # noqa: E402

from simsopt.field import multipole_inductance  # noqa: E402
from simsopt.field.bulk_inductance import (  # noqa: E402
    _shell_inductance_matrix_blockwise_batched_jax,
)


def test_puck_moments_grid_vs_jax() -> None:
    rng = np.random.default_rng(0)
    nq, nd = 32, 5
    K = rng.standard_normal((nq, nd, 3))
    pts = rng.standard_normal((nq, 3))
    w = np.abs(rng.standard_normal(nq)) + 1.0e-3
    c = (w[:, None] * pts).sum(axis=0) / w.sum()
    M0, D0, Q0 = multipole_inductance.puck_moments_grid_numpy(K, pts, w, c)
    M, D, Q = multipole_inductance.jax_puck_moments(
        jnp.asarray(K[None, ...]),
        jnp.asarray((pts - c)[None, ...]),
        jnp.asarray(w[None, ...]),
        with_quadrupole=True,
    )
    np.testing.assert_allclose(M[0], M0, rtol=1.0e-10)
    np.testing.assert_allclose(D[0], D0, rtol=1.0e-10)
    assert Q is not None
    assert Q0.shape == (nd, 3, 3, 3)
    np.testing.assert_allclose(Q[0], Q0, rtol=1.0e-9)


def test_rotation_invariance_full_kernel_toy() -> None:
    """A common rotation :math:`R` maps :math:`L` to :math:`U L U^T` (orthogonal ``U``)."""
    rng = np.random.default_rng(0)
    nd, nq = 3, 20
    K0 = [
        rng.standard_normal((nq, nd, 3)) * 0.1,
        rng.standard_normal((nq, nd, 3)) * 0.1,
    ]
    c0, c1 = np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.4])
    pts0 = [
        rng.standard_normal((nq, 3)) * 0.01 + c0,
        rng.standard_normal((nq, 3)) * 0.01 + c1,
    ]
    w0 = [
        np.abs(rng.standard_normal(nq)) * 0.01 + 1.0e-3,
        np.abs(rng.standard_normal(nq)) * 0.01 + 1.0e-3,
    ]  # noqa: E501
    d_off = [0, nd]
    old_far = os.environ.get("SIMSOPT_PSC_FAR_PAIR", "0")
    try:
        os.environ["SIMSOPT_PSC_FAR_PAIR"] = "0"
        L = _shell_inductance_matrix_blockwise_batched_jax(
            K0, pts0, w0, d_off, 2 * nd, 1.0e-8, True
        )
    finally:
        if old_far is None:
            os.environ.pop("SIMSOPT_PSC_FAR_PAIR", None)
        else:
            os.environ["SIMSOPT_PSC_FAR_PAIR"] = old_far
    R = np.array(
        [
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1.0],
        ],
        dtype=float,
    )
    K1 = [K0[i] @ R.T for i in range(2)]
    pts1 = [pts0[i] @ R.T for i in range(2)]
    try:
        os.environ["SIMSOPT_PSC_FAR_PAIR"] = "0"
        L1 = _shell_inductance_matrix_blockwise_batched_jax(
            K1, pts1, w0, d_off, 2 * nd, 1.0e-8, True
        )
    finally:
        if old_far is None:
            os.environ.pop("SIMSOPT_PSC_FAR_PAIR", None)
        else:
            os.environ["SIMSOPT_PSC_FAR_PAIR"] = old_far
    U = np.zeros((2 * nd, 2 * nd), dtype=float)
    U[:nd, :nd] = R
    U[nd:, nd:] = R
    np.testing.assert_allclose(L1, U @ L @ U.T, rtol=0.0, atol=1.0e-6)


def test_moments_fourier_matches_grid() -> None:
    r"""``SIMSOPT_PSC_MOMENTS=fourier`` host path matches the grid reference."""
    rng = np.random.default_rng(2)
    nq, nd = 18, 4
    K = rng.standard_normal((nq, nd, 3)) * 0.3
    pts = rng.standard_normal((nq, 3)) * 0.05
    w = np.abs(rng.standard_normal(nq)) + 0.01
    c = (w[:, None] * pts).sum(axis=0) / w.sum()
    a0 = multipole_inductance.puck_moments_grid_numpy(K, pts, w, c)
    old = os.environ.get("SIMSOPT_PSC_MOMENTS")
    try:
        os.environ["SIMSOPT_PSC_MOMENTS"] = "fourier"
        a1 = multipole_inductance.puck_moments_fourier_numpy(K, pts, w, c)
    finally:
        if old is None:
            os.environ.pop("SIMSOPT_PSC_MOMENTS", None)
        else:
            os.environ["SIMSOPT_PSC_MOMENTS"] = old
    for t in range(3):
        np.testing.assert_allclose(a0[t], a1[t], rtol=1.0e-12, atol=1.0e-12)


def test_far_pair_selfcheck_sticky_forces_zero_multipole_on_second_rebuild() -> None:
    r"""If the first multipole self-check fails, the next rebuild scatters 0 *multi* pairs.

    Uses a small :math:`R_\text{far}` and :class:`~simsopt.field.bulk_inductance`
    cache to avoid re-entering a bad far kernel (see
    :func:`simsopt.field.bulk_inductance.reset_psc_far_selfcheck_cache`).
    """
    from simsopt.field import bulk_inductance as _bi
    from simsopt.field.bulk_inductance import reset_psc_far_selfcheck_cache

    spec = importlib.util.spec_from_file_location(
        "tpb_multipole_sticky",
        Path(__file__).resolve().parent / "test_passive_bulks.py",
    )
    tpb = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(tpb)  # type: ignore[union-attr]
    _make = tpb._make_rotated_puck_array  # type: ignore[attr-defined]

    reset_psc_far_selfcheck_cache()
    _env: dict[str, str | None] = {}
    for k in (
        "SIMSOPT_PSCBULK_TIMING",
        "SIMSOPT_PSC_FAR_PAIR",
        "SIMSOPT_PSC_R_FAR",
        "SIMSOPT_PSC_FAR_CHECK",
    ):
        _env[k] = os.environ.get(k)
    try:
        os.environ["SIMSOPT_PSCBULK_TIMING"] = "1"
        os.environ["SIMSOPT_PSC_FAR_PAIR"] = "1"
        os.environ["SIMSOPT_PSC_R_FAR"] = "1.0"
        os.environ["SIMSOPT_PSC_FAR_CHECK"] = "1"
        p1 = _make(n_base=12, seed=0)
        p1._rebuild()
        cache1 = dict(_bi._PSC_FAR_SELFCHECK_OK)
        p2 = _make(n_base=12, seed=0)
        p2._rebuild()
        second = sum(
            int(r.get("n_pairs", 0))
            for r in (p2._timing_rows or [])
            if r.get("phase") == "rebuild_L_assembly_multipole"
        )
        if any(v is False for v in cache1.values()):
            assert second == 0, f"expected sticky disable of multipole, got {second=}"
    finally:
        reset_psc_far_selfcheck_cache()
        for k, v in _env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_calibrate_r_far_returns_float() -> None:
    r, ok = multipole_inductance.calibrate_r_far(
        np.random.default_rng(0).standard_normal((1, 8, 4, 3)) * 0.01,
        np.random.default_rng(1).standard_normal((1, 8, 3)) * 0.1,
        np.random.default_rng(2).random((1, 8)) * 0.01 + 0.01,
        [],
        seed=0,
    )
    assert float(r) >= 2.0
    assert bool(ok) is True
