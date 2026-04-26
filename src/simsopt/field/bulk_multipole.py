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
    """Subleading :math:`O(1/R^4)` dipole--quadrupole cross block (proxy for W2 ``P=2``).

    Uses :math:`L^{(DQ)}_{ab} \\propto 1/R^4` with
    :math:`m^a\\cdot (Q^b \\hat r)`-style coupling, symmetrised in the
    (dipole, quadrupole) roles so the output is a modest correction for far pairs.

    Args:
        m_i, m_j: ``(n_d, 3)`` per-mode dipole rows (global frame).
        Q_i_sym, Q_j_sym: ``(n_d, 3, 3)`` traceless symmetric quadrupole tensors.
        R_vec: ``(3,)`` center separation :math:`\\mathbf c_j - \\mathbf c_i`.

    Returns:
        ``(n_d, n_d)`` increment to the dipole--dipole block.

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
    """Far-pair mutual-inductance block to fixed multipole *order* (W2).

    ``order=1``: dipole--dipole via :func:`pair_inductance_dipole_block`.
    ``order>=2`` and both ``Q_*`` provided: add
    :func:`pair_dipole_quadrupole_cross_block` (otherwise same as ``order=1``).

    Args:
        m_i, m_j: ``(n_d, 3)`` per-mode dipole rows.
        R_vec: ``(3,)`` separation :math:`\\mathbf c_j-\\mathbf c_i`.
        order: ``1`` or ``2`` (higher currently treated as ``2`` when Q given).
        Q_i_sym, Q_j_sym: optional ``(n_d, 3, 3)`` traceless quadrupole rows for ``order>=2``.

    Returns:
        ``(n_d, n_d)`` block.

    """
    o = int(max(1, order))
    Ldd = pair_inductance_dipole_block(m_i, m_j, R_vec)
    if o < 2 or Q_i_sym is None or Q_j_sym is None:
        return Ldd
    return Ldd + pair_dipole_quadrupole_cross_block(m_i, Q_i_sym, m_j, Q_j_sym, R_vec)


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
