"""
Tangent-space :math:`\\mathfrak{so}(3)` parameterization of orientation quaternions.

Uses **scalar-first** quaternions ``q = [q0, q1, q2, q3]`` consistent with
:class:`~simsopt.geo.curveplanarfourier.CurvePlanarFourier` and
:class:`~simsopt.field.psc_bulk.PSCBulkArray`.

A small rotation in the right-tangent (body) frame is
``q(\\omega) = \\mathrm{exp}_\\mathrm{quat}([0,\\omega/2]) \\otimes q_0``,
where :math:`\\otimes` is the Hamilton product and
``exp_quat`` maps an imaginary quaternion to a unit quaternion via the
axis-angle map.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from typing import Any, Union

Array = Union[jnp.ndarray, Any]


def _quat_normalize_wxyz(q: jnp.ndarray) -> jnp.ndarray:
    """Return ``q / ||q||`` with safe guard for the zero case."""
    q = jnp.asarray(q, dtype=float)
    n = jnp.sqrt(jnp.maximum(jnp.sum(q * q), 1e-30))
    return q / n


def quat_multiply_wxyz(q1: Array, q2: Array) -> jnp.ndarray:
    """Hamilton product ``q1 \\otimes q2`` in scalar-first (w, x, y, z) storage."""
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
    return jnp.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=jnp.result_type(q1, q2),
    )


def exp_quat_imag(omega: Array) -> jnp.ndarray:
    """Map axis-angle vector ``\\omega`` to a unit quaternion (scalar-first)."""
    omega = jnp.asarray(omega, dtype=float)
    th = jnp.sqrt(jnp.sum(omega * omega) + 1e-30)
    half = 0.5 * th
    c = jnp.cos(half)
    s = jnp.sin(half) / th
    return jnp.array([c, s * omega[0], s * omega[1], s * omega[2]], dtype=omega.dtype)


def exp_quat_tangent(
    omega: Array,
    q0: Array,
) -> jnp.ndarray:
    """Re-anchor: ``q = normalize(exp([0, \\omega/2]) \\otimes q0)`` (right increment).

    Args:
        omega: ``(3,)`` tangent vector in the right-increment convention.
        q0: ``(4,)`` base unit quaternion, scalar-first.

    Returns:
        ``(4,)`` unit quaternion, scalar-first.
    """
    w = jnp.asarray(omega, dtype=float) * 0.5
    dq = exp_quat_imag(w)
    return _quat_normalize_wxyz(quat_multiply_wxyz(dq, jnp.asarray(q0, dtype=dq.dtype)))


def _rotation_matrix_from_quat_wxyz(q: Array) -> jnp.ndarray:
    """``R`` with columns ``(e1,e2,e3)`` mapping body frame to world (matches PSC)."""
    q = _quat_normalize_wxyz(jnp.asarray(q, dtype=float))
    w, x, y, z = q[0], q[1], q[2], q[3]
    return jnp.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=q.dtype,
    )


def dR_domega_at_zero(q0: Array) -> jnp.ndarray:
    """``∂R_ab/∂ω_k|_{ω=0}`` for :func:`exp_quat_tangent` (right-increment, scalar-first quats).

    Returns a tensor of shape ``(3, 3, 3)`` with layout ``(row, col, k)`` for
    ``ω_k``.

    """
    q0 = jnp.asarray(q0, dtype=float)

    def r_mat(om: jnp.ndarray) -> jnp.ndarray:
        return _rotation_matrix_from_quat_wxyz(exp_quat_tangent(om, q0))

    return jax.jacobian(r_mat)(jnp.zeros(3))
