# Copyright 2016-2020 HiddenSymmetries, MIT License
"""
Cartesian multipole (through quadrupole) approximation for the shell
inductance double integral at large center separation.

The leading terms implement Eq. (4) in the PSC far-pair plan: per-puck
monopole (``M``), dipole (``D``) and quadrupole (``Q``) moments of
the sheet-current basis, contracted with a fixed tensor of
:math:`1/|{\\bf d} - {\\bf u}|` expanded for small :math:`|{\\bf u}|/d`.
"""

from __future__ import annotations

import os
from functools import partial
from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from . import bulk_inductance

MU0_OVER_4PI = float(bulk_inductance.MU0_OVER_4PI)

_PSC_MOMENT_FP32_FLAG = "SIMSOPT_PSC_FAR_FP32"  # optional fp32 in far kernel
_PSC_MOMENTS_MODE = "SIMSOPT_PSC_MOMENTS"  # "grid" (default) or "fourier"


def _use_fp32_multipole() -> bool:
    return os.environ.get("SIMSOPT_PSC_FAR_FP32", "0") == "1"


def _moments_mode() -> str:
    return os.environ.get(_PSC_MOMENTS_MODE, "grid").lower().strip()


@partial(
    jax.jit,
    static_argnames=("with_quadrupole", "use_fp32"),
)
def jax_puck_moments(
    K: jnp.ndarray,
    delta: jnp.ndarray,
    w: jnp.ndarray,
    with_quadrupole: bool = True,
    use_fp32: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray, Optional[jnp.ndarray]]:
    r"""Batched per-puck moments for one puck (or a stack of identical layouts).

    Args:
        K: ``(n_puck, nq, nd, 3)`` sheet-current basis in the global frame.
        delta: ``(n_puck, nq, 3)`` = quadrature point minus puck center
            (global Cartesian).
        w: ``(n_puck, nq)`` scalar quadrature weights.
        with_quadrupole: If ``False``, return ``Q = None`` (monopole+dipole only).
        use_fp32: Internal accumulation in ``float32`` (opt-in, large-d pairs).

    Returns:
        ``(M, D, Q)`` with shapes ``(n_puck, nd, 3)``, ``(n_puck, nd, 3, 3)``,
        and either ``(n_puck, nd, 3, 3, 3)`` or ``None`` for :math:`Q_{a\mu\alpha\beta}`.
    """
    dtype = jnp.float32 if use_fp32 else jnp.float64
    K0 = K.astype(dtype)
    w0 = w.astype(dtype)[:, :, None, None]  # broadcast with K
    d0 = delta.astype(dtype)
    wK = K0 * w0

    # M_{a, mu} = sum_q K * w
    M = jnp.sum(wK, axis=1)  # (P, nd, 3)
    # D_{a, mu, beta} = sum_q w K_{a,\\mu} \\delta_\\beta
    D = jnp.einsum("pqam,pqb->pamb", wK, d0, optimize=True)

    if not with_quadrupole:
        return M, D, None

    # :math:`Q_{a,\\mu,\\alpha,\\beta} = \\sum_q w K_{a,\\mu} \\delta_\\alpha \\delta_\\beta`
    Q = jnp.einsum("pqam,pqi,pqj->pamij", wK, d0, d0, optimize=True)
    Q = 0.5 * (Q + jnp.swapaxes(Q, -1, -2))
    return M, D, Q


@partial(jax.jit, static_argnames=("with_quadrupole", "use_fp32"))
def jax_far_pair_block(
    Mi: jnp.ndarray,
    Mj: jnp.ndarray,
    Di: jnp.ndarray,
    Dj: jnp.ndarray,
    Qi: Optional[jnp.ndarray],
    Qj: Optional[jnp.ndarray],
    d_vec: jnp.ndarray,
    with_quadrupole: bool = True,
    use_fp32: bool = False,
) -> jnp.ndarray:
    r"""Inductance block for one pair :math:`(i, j)` using Eq. (4) in the plan.

    Args:
        Mi, Mj: ``(nd_i, 3)``, ``(nd_j, 3)`` monopole moments
            :math:`M_{a,\mu}`.
        Di, Dj: ``(nd_i, 3, 3)``, ``(nd_j, 3, 3)`` with indices
            :math:`D_{a,\mu,\beta}`.
        Qi, Qj: ``(nd, 3, 3, 3)`` for :math:`Q_{a,\mu,\alpha\beta}` or ``None``.
        d_vec: ``(3,)`` = :math:`\mathbf c_j - \mathbf c_i` (centers, global).
        with_quadrupole: If ``False``, only :math:`T_0+T_1` (faster, less
            accurate at moderate separation).
        use_fp32: Accumulate the contraction in float32 (after casting inputs).

    Returns:
        ``(nd_i, nd_j)`` block in henries, scaled by
        :data:`~simsopt.field.bulk_inductance.MU0_OVER_4PI` consistent with
        the dense kernel.
    """
    dtype = jnp.float32 if use_fp32 else jnp.float64
    Mi = Mi.astype(dtype)
    Mj = Mj.astype(dtype)
    Di = Di.astype(dtype)
    Dj = Dj.astype(dtype)
    d_vec = d_vec.astype(dtype)
    d = jnp.linalg.norm(d_vec) + jnp.asarray(1.0e-30, dtype=dtype)
    dhat = d_vec / d

    mu0 = jnp.asarray(MU0_OVER_4PI, dtype=dtype)
    T0 = (1.0 / d) * jnp.einsum("ax,bx->ab", Mi, Mj, optimize=True)
    Di_dot = jnp.einsum("axb,b->ax", Di, dhat, optimize=True)  # (nd_i, 3) K-comp
    Dj_dot = jnp.einsum("bxc,c->bx", Dj, dhat, optimize=True)  # (nd_j, 3)
    T1 = (1.0 / (d * d)) * (
        jnp.einsum("ax,bx->ab", Di_dot, Mj, optimize=True)
        - jnp.einsum("ax,bx->ab", Mi, Dj_dot, optimize=True)
    )
    out = T0 + T1
    if with_quadrupole and Qi is not None and Qj is not None:
        Qi = Qi.astype(dtype)
        Qj = Qj.astype(dtype)
        T_tensor = 3.0 * jnp.outer(dhat, dhat) - jnp.eye(3, dtype=dtype)
        # Q: (nd, 3, 3, 3) = (a, μ, α, β); T_tensor: (α, β)
        QiT = jnp.einsum("amcd,cd->am", Qi, T_tensor, optimize=True)
        QjT = jnp.einsum("bmcd,cd->bm", Qj, T_tensor, optimize=True)
        T2_mq = (1.0 / (2.0 * d**3)) * (
            jnp.einsum("am,bm->ab", QiT, Mj, optimize=True)
            + jnp.einsum("am,bm->ab", Mi, QjT, optimize=True)
        )
        T3_dd = -(1.0 / d**3) * jnp.einsum(
            "amc,bmd,cd->ab", Di, Dj, T_tensor, optimize=True
        )
        out = out + T2_mq + T3_dd
    return mu0 * out


def _puck_centers_weights(pts: np.ndarray, w: np.ndarray) -> Tuple[np.ndarray, float]:
    """Return ``(center, R_eff)`` with ``R_eff = sqrt(sum(w) / pi)``."""
    pts = np.asarray(pts, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64).ravel()
    sw = float(np.sum(w)) + 1.0e-300
    c = (w[:, None] * pts).sum(axis=0) / sw
    r_eff = float(np.sqrt(sw / np.pi))
    return c, r_eff


def calibrate_r_far(
    K_stack: np.ndarray,
    pts_stack: np.ndarray,
    w_stack: np.ndarray,
    pair_indices: List[Tuple[int, int]],
    *,
    tolerance: float = 1.0e-5,
    r_min: float = 2.0,
    r_max: float = 10.0,
    max_iter: int = 24,
    seed: int = 0,
    delta_reg: float = 1.0e-8,
    adaptive_self_reg: bool = True,
) -> Tuple[float, bool]:
    r"""Return ``(R_far, success)`` for classifying *far* as
    ``d_ij / (R_i + R_j) >= R_far``.

    Binary-searches the **smallest** :math:`R_\text{far} \in [r_\min, r_\max]`
    such that the max relative Frobenius error of the multipole block vs. the
    full tensor ``einsum`` on a set of *candidate* far pairs (those with
    large enough ``d/\\sum R``) stays below ``tolerance`` on a random subset.

    Pairs: ``(i, j)`` with ``0 <= i <= j < n_puck``. Self-pairs (``i == j``)
    are **skipped** (the multipole is not a substitute for self-inductance).
    """
    if len(pair_indices) == 0:
        return 4.0, True

    rng = np.random.default_rng(int(seed))
    n_p = int(pts_stack.shape[0])
    centers = np.empty((n_p, 3), dtype=np.float64)
    r_eff = np.empty((n_p,), dtype=np.float64)
    for p in range(n_p):
        centers[p], r_eff[p] = _puck_centers_weights(pts_stack[p], w_stack[p])

    # Sample at most 32 unique non-self candidate pairs
    cands: List[Tuple[int, int]] = [
        (i, j) for (i, j) in pair_indices if int(i) < int(j) and int(i) != int(j)
    ]
    if not cands:
        return 4.0, True
    rng.shuffle(cands)
    sample = cands[: min(32, len(cands))]

    dreg = float(delta_reg)
    adapt = bool(adaptive_self_reg)

    def full_block(p: int, q: int) -> np.ndarray:
        Ki = jnp.asarray(K_stack[p : p + 1])
        Kj = jnp.asarray(K_stack[q : q + 1])
        pi = jnp.asarray(pts_stack[p : p + 1])
        pj = jnp.asarray(pts_stack[q : q + 1])
        wi = jnp.asarray(w_stack[p : p + 1])
        wj = jnp.asarray(w_stack[q : q + 1])
        return np.asarray(
            bulk_inductance._jax_pair_batch_blocks(
                Ki,
                Kj,
                pi,
                pj,
                wi,
                wj,
                jnp.asarray(dreg, dtype=np.float64),
                adaptive_self_reg=adapt,
                use_fp32=False,
            )[0]
        )

    def mp_block(p: int, q: int) -> np.ndarray:
        d_vec = centers[q] - centers[p]
        Kp = jnp.asarray(K_stack[p : p + 1])
        Kq = jnp.asarray(K_stack[q : q + 1])
        dlt_p = jnp.asarray(pts_stack[p : p + 1] - centers[p])
        dlt_q = jnp.asarray(pts_stack[q : q + 1] - centers[q])
        wp = jnp.asarray(w_stack[p : p + 1])
        wq = jnp.asarray(w_stack[q : q + 1])
        M, D, Q = jax_puck_moments(Kp, dlt_p, wp, with_quadrupole=True, use_fp32=False)
        M2, D2, Q2 = jax_puck_moments(
            Kq, dlt_q, wq, with_quadrupole=True, use_fp32=False
        )
        d_jax = jnp.asarray(d_vec, dtype=np.float64)
        return np.asarray(
            jax_far_pair_block(
                M[0],
                M2[0],
                D[0],
                D2[0],
                Q[0],
                Q2[0],
                d_jax,
                with_quadrupole=True,
                use_fp32=False,
            )
        )

    def max_err_for_threshold(rf: float) -> float:
        worst = 0.0
        for p, q in sample:
            dist = float(np.linalg.norm(centers[p] - centers[q]))
            rs = float(r_eff[p] + r_eff[q]) + 1.0e-30
            if dist / rs < float(rf):
                continue
            fblk = full_block(p, q)
            mblk = mp_block(p, q)
            den = max(1.0e-30, float(np.max(np.abs(fblk))))
            worst = max(worst, float(np.max(np.abs(fblk - mblk)) / den))
        return worst

    lo, hi = float(r_min), float(r_max)
    if max_err_for_threshold(lo) <= tolerance:
        return lo, True
    if max_err_for_threshold(hi) > tolerance:
        return 4.0, False
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if max_err_for_threshold(mid) <= tolerance:
            hi = mid
        else:
            lo = mid
    return hi, True


__all__ = [
    "jax_puck_moments",
    "jax_far_pair_block",
    "calibrate_r_far",
    "puck_moments_grid_numpy",
    "puck_moments_fourier_numpy",
]


def puck_moments_grid_numpy(
    K: np.ndarray,
    pts: np.ndarray,
    w: np.ndarray,
    center: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Host NumPy :math:`M,D,Q` for one puck (matches :func:`jax_puck_moments`)."""
    dlt = (
        np.asarray(pts, dtype=np.float64)
        - np.asarray(center, dtype=np.float64)[None, :]
    )
    w = np.asarray(w, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    wv = w[:, None, None] * K
    M = np.einsum("qax->ax", wv, optimize=True)
    D = np.einsum("qax,qb->axb", wv, dlt, optimize=True)
    Q = np.einsum("qax,qb,qc->axbc", wv, dlt, dlt, optimize=True)
    Q = 0.5 * (Q + np.swapaxes(Q, -1, -2))
    return M, D, Q


def puck_moments_fourier_numpy(
    K: np.ndarray,
    pts: np.ndarray,
    w: np.ndarray,
    center: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Per-puck multipole :math:`(M, D, Q)` in the *host* (NumPy) path.

    When :envvar:`SIMSOPT_PSC_MOMENTS` is not ``"fourier"`` this is an alias of
    :func:`puck_moments_grid_numpy`.

    When ``fourier`` is selected, the full discrete quadrature in
    :func:`puck_moments_grid_numpy` is evaluated (numerically exact for a given
    quadrature rule).  A future version may replace the azimuthal sum on disk
    faces with analytic trigonometric orthogonality when
    :class:`~simsopt.field.puck_basis.PuckBasisData` structure (Fourier
    :math:`m` and ``cos``/``sin``) is threaded through from the shell builder.
    Batched assembly in
    :func:`~simsopt.field.multipole_inductance.jax_puck_moments` is used for
    the hot :class:`~simsopt.field.psc_bulk.PSCBulkArray` rebuild path regardless
    of this host helper.
    """
    if _moments_mode() != "fourier":
        return puck_moments_grid_numpy(K, pts, w, center)
    return puck_moments_grid_numpy(K, pts, w, center)
