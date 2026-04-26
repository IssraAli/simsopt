"""
Multipole moments for passive bulk sheet currents and far-field approximations.

Per-puck magnetic dipole and symmetric quadrupole moments are defined in the
puck **local** frame (origin at the puck centroid) for use in W2 (pair
inductance), W3 (TF loading), and W7 (far eval field).

All integrals use the same convention as :class:`~simsopt.field.psc_bulk.PSCBulkArray`
stacked K basis: ``K_a = n x grad_s Phi_a`` on the shell, ``w`` quadrature
weights, ``r`` position relative to centroid.
"""

from __future__ import annotations

import jax.numpy as jnp

# Match force.py :math:`B` convention (B returned includes mu0/(4pi) * 1e-7 in practice
# for the discrete sum); for moments we use SI-style mu0/(4pi) in far-field B.
from .bulk_inductance import MU0_OVER_4PI


def magnetic_dipole_moments_stacked(
    K_stack: jnp.ndarray,
    quad_r_local: jnp.ndarray,
    w_stack: jnp.ndarray,
) -> jnp.ndarray:
    """Magnetic dipole moment per mode: ``m_a = (1/2) * sum w (r x K_a)`` in local frame.

    Args:
        K_stack: ``(n_puck, nq, nd, 3)`` per-puck basis sheet currents in local frame.
        quad_r_local: ``(n_puck, nq, 3)`` position on shell relative to centroid.
        w_stack: ``(n_puck, nq)`` quadrature weights (area elements).

    Returns:
        ``m`` of shape ``(n_puck, nd, 3)`` in weber-meters (same units as ``MU0`` scaling
        in surrounding inductance formulas).
    """
    K_stack = jnp.asarray(K_stack)
    quad_r_local = jnp.asarray(quad_r_local)
    w_stack = jnp.asarray(w_stack)
    rxK = jnp.cross(quad_r_local[:, :, None, :], K_stack, axis=-1)
    m = 0.5 * jnp.sum(w_stack[:, :, None, None] * rxK, axis=1)
    return m


def quadrupole_magnetic_symmetric_stacked(
    K_stack: jnp.ndarray,
    quad_r_local: jnp.ndarray,
    w_stack: jnp.ndarray,
) -> jnp.ndarray:
    """Symmetric (traceless) rank-2 moment for ``Q : (grad B)`` loading.

    Uses ``S_ab = (1/2) sum w (r x K_a)_a r_b``, symmetrised, then TR removal.
    A compact engineering proxy; exact quadrupole matching is not required for W3.

    """
    K_stack = jnp.asarray(K_stack)
    quad_r_local = jnp.asarray(quad_r_local)
    w_stack = jnp.asarray(w_stack)
    rxK = jnp.cross(quad_r_local[:, :, None, :], K_stack, axis=-1)
    outer = 0.5 * jnp.sum(
        w_stack[:, :, None, None, None]
        * (rxK[:, :, :, :, None] * quad_r_local[:, :, None, :, None]),
        axis=1,
    )
    S = 0.5 * (outer + jnp.swapaxes(outer, -1, -2))
    tr = (S[..., 0, 0] + S[..., 1, 1] + S[..., 2, 2]) / 3.0
    Q = S - tr[..., None, None] * jnp.eye(3, dtype=K_stack.dtype)
    return Q


def current_dipole_l2_stacked(
    K_stack: jnp.ndarray,
    w_stack: jnp.ndarray,
) -> jnp.ndarray:
    """``||sum w K_a||_2`` per (puck, mode) — should be near zero for closed shells.

    Returns:
        Array ``(n_puck, nd)`` of L2 norms.
    """
    K_stack = jnp.asarray(K_stack)
    w_stack = jnp.asarray(w_stack)
    tot = jnp.sum(w_stack[:, :, None] * K_stack, axis=1)
    return jnp.linalg.norm(tot, axis=-1)


def pair_inductance_dipole_dipole(
    m_i: jnp.ndarray,
    m_j: jnp.ndarray,
    R_vec: jnp.ndarray,
) -> jnp.ndarray:
    """Scalar dipole-dipole factor ``m_i^T T_dd m_j`` with ``T_dd = (I-3R̂R̂^T)/R^3``."""
    R = jnp.asarray(R_vec)
    dist = jnp.linalg.norm(R) + 1e-20
    Rhat = R / dist
    num = (jnp.dot(m_i, m_j) - 3.0 * jnp.dot(m_i, Rhat) * jnp.dot(m_j, Rhat)) / (
        dist**3
    )
    return MU0_OVER_4PI * num


def pair_inductance_dipole_block(
    m_i: jnp.ndarray,
    m_j: jnp.ndarray,
    R_vec: jnp.ndarray,
) -> jnp.ndarray:
    """Dipole-dipole pair block: ``(nd_i,3) T (nd_j,3)^T`` with ``T = (I-3r̂r̂^T)/R^3`` (scaled).

    This is the full tensor ``L_ij[ab] = m_i,a^T T m_j,b`` in the small-separation
    expansion (leading term).
    """
    m_i = jnp.asarray(m_i)
    m_j = jnp.asarray(m_j)
    R = jnp.asarray(R_vec, dtype=m_i.dtype)
    dist = jnp.linalg.norm(R) + 1e-20
    rhat = R / dist
    I3 = jnp.eye(3, dtype=m_i.dtype)
    T = (I3 - 3.0 * jnp.outer(rhat, rhat)) / (dist**3)
    T = MU0_OVER_4PI * T
    return m_i @ T @ m_j.T


def pair_dipole_quadrupole_cross_block(
    m_i: jnp.ndarray,
    Q_i_sym: jnp.ndarray,
    m_j: jnp.ndarray,
    Q_j_sym: jnp.ndarray,
    R_vec: jnp.ndarray,
) -> jnp.ndarray:
    r"""Engineering proxy for the dipole--quadrupole cross block (NOT a derived expansion).

    .. warning::
        This function is an **engineering proxy**, not a closed-form
        :math:`P=2` multipole expansion of the mutual inductance.  It models
        the *scaling* of the dipole--quadrupole cross term as
        :math:`\propto 1/R^4` with a roles-symmetrised
        :math:`m_i^a (Q_j^b \hat r) + (Q_i^a \hat r) m_j^b` coupling, but the
        coefficient is **not** derived from the Neumann mutual-inductance
        integral and the full :math:`T_{dq}, T_{qq}` rank-3 / rank-4 kernels
        are not implemented.  In particular, the slope of the residual
        ``||L_dense - L_dipole - L_DQ||_F / ||L_dense||_F`` versus
        :math:`\kappa = R/R_{\max}` is **not** guaranteed to follow the
        :math:`R^{-(P+1)} = R^{-3}` law expected from a true :math:`P=2`
        truncation.  The public :func:`pair_inductance_multipole` therefore
        raises :class:`NotImplementedError` for ``order >= 2`` to prevent
        silent use of this proxy in production.  Direct callers (regression
        tests, diagnostic notebooks) are responsible for understanding the
        approximation.

    Args:
        m_i, m_j: ``(n_d, 3)`` per-mode dipole rows (global frame).
        Q_i_sym, Q_j_sym: ``(n_d, 3, 3)`` traceless symmetric quadrupole tensors.
        R_vec: ``(3,)`` center separation :math:`\mathbf c_j - \mathbf c_i`.

    Returns:
        ``(n_d, n_d)`` increment to the dipole--dipole block (proxy).

    """
    m_i = jnp.asarray(m_i, dtype=float)
    m_j = jnp.asarray(m_j, dtype=float)
    Qi = jnp.asarray(Q_i_sym, dtype=m_i.dtype)
    Qj = jnp.asarray(Q_j_sym, dtype=m_i.dtype)
    r = jnp.asarray(R_vec, dtype=m_i.dtype)
    d = jnp.linalg.norm(r) + 1e-20
    rhat = r / d
    k4 = MU0_OVER_4PI / (d**4)
    qjr = jnp.einsum("jab, b->ja", Qj, rhat)
    qir = jnp.einsum("iab, b->ia", Qi, -rhat)
    inc_ij = k4 * jnp.einsum("ak, bk -> ab", m_i, qjr)
    inc_ji = k4 * jnp.einsum("ak, bk -> ab", qir, m_j)
    return 0.5 * (inc_ij + inc_ji)


def pair_inductance_multipole(
    m_i: jnp.ndarray,
    m_j: jnp.ndarray,
    R_vec: jnp.ndarray,
    order: int = 1,
    Q_i_sym: jnp.ndarray | None = None,
    Q_j_sym: jnp.ndarray | None = None,
) -> jnp.ndarray:
    r"""Far-pair mutual-inductance block at fixed multipole *order* (W2).

    Only ``order=1`` (dipole--dipole) is part of the supported public API;
    the dispatcher delegates directly to :func:`pair_inductance_dipole_block`
    so the result is the genuine :math:`O(1/R^3)` expansion derived from
    Neumann's integral.

    Higher orders are intentionally not implemented in this module: the
    proxy :func:`pair_dipole_quadrupole_cross_block` captures only the
    *scaling* of the dipole--quadrupole cross term and is not a derived
    :math:`P=2` truncation of the mutual-inductance integral.  Calling
    this function with ``order >= 2`` raises :class:`NotImplementedError`
    so the proxy cannot be silently used as a drop-in replacement; tests
    and diagnostics that explicitly want the proxy must call
    :func:`pair_dipole_quadrupole_cross_block` directly.

    Args:
        m_i, m_j: ``(n_d, 3)`` per-mode dipole rows.
        R_vec: ``(3,)`` separation :math:`\mathbf c_j-\mathbf c_i`.
        order: Only ``1`` is supported.
        Q_i_sym, Q_j_sym: Unused at ``order=1``; reserved for future
            higher-order kernels.  Pass ``None`` (default).

    Returns:
        ``(n_d, n_d)`` dipole--dipole block.

    Raises:
        NotImplementedError: If ``order >= 2``.  The dipole--quadrupole
            proxy (:func:`pair_dipole_quadrupole_cross_block`) is a
            scaling-only approximation and is no longer reachable through
            this dispatcher.

    """
    o = int(order)
    if o >= 2:
        raise NotImplementedError(
            "pair_inductance_multipole(order>=2) is not supported. "
            "The order=2 path previously routed to "
            "pair_dipole_quadrupole_cross_block, which is an engineering "
            "proxy that captures only 1/R^4 scaling and not a derived P=2 "
            "multipole expansion.  Call "
            "pair_dipole_quadrupole_cross_block directly if you understand "
            "the approximation."
        )
    return pair_inductance_dipole_block(m_i, m_j, R_vec)


def pair_inductance_multipole_selfcheck_dense(
    dense_block: jnp.ndarray,
    multipole_block: jnp.ndarray,
) -> float:
    """Relative Frobenius error ``||M-D||_F / (||D||_F + eps)`` for W2 regression tests."""
    d = jnp.asarray(dense_block)
    m = jnp.asarray(multipole_block, dtype=d.dtype)
    num = float(jnp.linalg.norm(m - d, ord="fro"))
    den = float(jnp.linalg.norm(d, ord="fro")) + 1e-30
    return num / den


def magnetic_field_dipole_points(
    m_total: jnp.ndarray,
    r_eval: jnp.ndarray,
    r_center: jnp.ndarray,
) -> jnp.ndarray:
    """Standard magnetic dipole field from moment ``m_total`` (A·m²).

    ``B = (μ₀/4π) (3 (m·r̂) r̂ - m) / |r|³``.

    Args:
        m_total: ``(3,)`` total moment.
        r_eval: ``(n, 3)`` evaluation points.
        r_center: ``(3,)`` dipole position.

    Returns:
        ``(n, 3)`` in tesla-consistent scale with ``MU0_OVER_4PI`` in Biot–Savart.
    """
    m_total = jnp.asarray(m_total, dtype=r_eval.dtype)
    r = r_eval - r_center[None, :]
    d = jnp.linalg.norm(r, axis=-1)[:, None] + 1e-20
    rhat = r / d
    mdot = jnp.sum(m_total[None, :] * rhat, axis=-1, keepdims=True)
    b = (3.0 * mdot * rhat - m_total[None, :]) / (d**2 * d)  # (1/r^3) * ...
    return MU0_OVER_4PI * b
