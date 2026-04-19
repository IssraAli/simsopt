"""
Ideal-diamagnetic passive bulk (cylindrical pucks) and :class:`PassiveBulkField`.

Puck geometry DOFs (center, quaternion orientation, radius, thickness per
base puck) are exposed through the :class:`~simsopt._core.optimizable.Optimizable`
framework.  All puck DOFs are **fixed by default**; unfix them with
``psc.unfix('center_x0')`` etc. to include them in the optimization.

Orientation uses the same **scalar-first quaternion** convention as
:class:`~simsopt.geo.curveplanarfourier.CurvePlanarFourier`:
``q = [q0, qi, qj, qk]`` with ``q0 = cos(theta/2)``.  The quaternion is
normalized before computing the rotation matrix, so the DOFs need not lie
on the unit sphere.

Induced currents (beta) are NOT optimizable -- they are recomputed from
``L_r^{-1} f_r`` each time :meth:`PSCBulkArray.recompute_currents` is called.
Gradients propagate through the solve via JAX VJPs.

Environment:

* ``PSC_BULK_USE_JAX_TF_VJP`` — if set to ``1``/``true``, use the legacy JAX
  :func:`_B_at_point_from_coil_set_pure` TF VJP (slow; for debugging). Default
  is the analytic adjoint + C++ :class:`~simsopt.field.biot_savart.BiotSavart`.
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax import vjp

from simsopt._core.derivative import Derivative
from simsopt._core.optimizable import Optimizable

from .bulk_inductance import (
    MU0_OVER_4PI,
    _SELF_REG_COEFF,
    expand_beta_reduced,
    null_space_projection_matrix,
    project_reduced_system,
    shell_biot_savart_pure,
    shell_inductance_matrix_blockwise,
    shell_loading_vector_pure,
    shell_solve_eigenfloor_pure,
    shell_solve_linear_pure,
)
from .biotsavart import BiotSavart
from .force import _B_at_point_from_coil_set_pure
from .magneticfield import MagneticField
from .puck_basis import (
    PuckBasisData,
    build_continuity_constraint,
    build_puck_shell_basis,
)
from .puck_init import cylindrical_grid_pucks, winding_surface_pucks

# Analytic TF-only VJP via C++ BiotSavart (default). Set to ``1`` to use the
# slower JAX ``vjp`` through :func:`_B_at_point_from_coil_set_pure` (for tests).
_USE_JAX_TF_VJP = os.environ.get("PSC_BULK_USE_JAX_TF_VJP", "").lower() in (
    "1",
    "true",
    "yes",
)

# ======================================================================
# Quaternion helpers
# ======================================================================

_DOFS_PER_PUCK = 9  # cx, cy, cz, q0, qi, qj, qk, R, t


def _axis_to_quaternion(axis: np.ndarray) -> np.ndarray:
    """Convert axis vector to quaternion that rotates ``(0,0,1)`` to ``axis``.

    Uses the scalar-first convention ``[q0, qi, qj, qk]``.
    """
    a = np.asarray(axis, dtype=float)
    a = a / (np.linalg.norm(a) + 1e-30)
    z = np.array([0.0, 0.0, 1.0])
    dot = float(np.dot(z, a))
    if dot > 1 - 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    if dot < -1 + 1e-12:
        return np.array([0.0, 1.0, 0.0, 0.0])
    v = np.cross(z, a)
    v = v / np.linalg.norm(v)
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    s = np.sin(theta / 2.0)
    return np.array([np.cos(theta / 2.0), v[0] * s, v[1] * s, v[2] * s])


def _quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product ``q1 * q2``, scalar-first."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def _quat_left_mult_matrix(q_L: np.ndarray) -> np.ndarray:
    """4x4 matrix ``M`` such that ``q_L * q = M @ q`` (Hamilton product)."""
    w, x, y, z = q_L
    return np.array(
        [
            [w, -x, -y, -z],
            [x, w, -z, y],
            [y, z, w, -x],
            [z, -y, x, w],
        ]
    )


def _rotation_matrix_from_quat(q: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix from quaternion ``[w, x, y, z]`` (NumPy)."""
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q)
    if n < 1e-14:
        return np.eye(3)
    q = q / n
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def _rotation_matrix_from_quat_jax(q: jnp.ndarray) -> jnp.ndarray:
    """JAX-differentiable 3x3 rotation from quaternion ``[w, x, y, z]``."""
    norm_q = jnp.linalg.norm(q)
    q_n = jnp.where(norm_q < 1e-8, q / (norm_q + 1e-8), q / norm_q)
    w, x, y, z = q_n[0], q_n[1], q_n[2], q_n[3]
    return jnp.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def _quaternion_to_axis(q: np.ndarray) -> np.ndarray:
    """Puck axis (unit z-vector of the rotated frame) from quaternion."""
    return _rotation_matrix_from_quat(q) @ np.array([0.0, 0.0, 1.0])


def _rotation_matrix_local_to_global(axis_z: np.ndarray) -> np.ndarray:
    """Rotation matrix mapping ``(0,0,1)`` to ``axis_z`` (backward compat).

    Converts to quaternion internally; kept for VTK and test imports.
    """
    q = _axis_to_quaternion(axis_z)
    return _rotation_matrix_from_quat(q)


# ----------------------------------------------------------------------
# Module-level JAX (stable across PSCBulkArray._rebuild; avoids recompile)
# ----------------------------------------------------------------------

_EPS_BS = 1e-8
_EIGENFLOOR_THRESHOLD = 1e-10


def _beta_from_tf_body(
    L_full: jnp.ndarray,
    Qm: jnp.ndarray,
    quad_pts: jnp.ndarray,
    quad_n: jnp.ndarray,
    phi_m: jnp.ndarray,
    w_q: jnp.ndarray,
    g_tf: jnp.ndarray,
    gd_tf: jnp.ndarray,
    I_tf: jnp.ndarray,
) -> jnp.ndarray:
    """TF-only modal coefficients using Q-projected reduced solve (fast path)."""
    gammas_tf = jnp.asarray(g_tf)
    gammadash_tf = jnp.asarray(gd_tf)
    currents_tf = jnp.asarray(I_tf)

    def Bn_at_i(i):
        B = _B_at_point_from_coil_set_pure(
            quad_pts[i],
            gammas_tf,
            gammadash_tf,
            currents_tf,
            -1,
            _EPS_BS,
        )
        return jnp.dot(B, quad_n[i])

    Bn = jax.vmap(Bn_at_i)(jnp.arange(quad_pts.shape[0]))
    f = shell_loading_vector_pure(phi_m, w_q, Bn)
    Lr, fr = project_reduced_system(L_full, f, Qm)
    alpha = shell_solve_linear_pure(Lr, fr)
    return expand_beta_reduced(alpha, Qm)


_beta_from_tf_jitted = jax.jit(_beta_from_tf_body)


def _beta_from_bn_body(
    L_full: jnp.ndarray,
    Qm: jnp.ndarray,
    phi_m: jnp.ndarray,
    w_q: jnp.ndarray,
    Bn: jnp.ndarray,
) -> jnp.ndarray:
    """Reduced solve given precomputed normal field ``Bn`` at quad points (TF path)."""
    f = shell_loading_vector_pure(phi_m, w_q, Bn)
    Lr, fr = project_reduced_system(L_full, f, Qm)
    alpha = shell_solve_linear_pure(Lr, fr)
    return expand_beta_reduced(alpha, Qm)


_beta_from_bn_jitted = jax.jit(_beta_from_bn_body)


def _beta_eigenfloor_from_bn_body(
    L_full: jnp.ndarray,
    Q_c: jnp.ndarray,
    phi_m: jnp.ndarray,
    w_q: jnp.ndarray,
    Bn: jnp.ndarray,
) -> jnp.ndarray:
    """Rim-continuity-projected eigenfloor solve given ``Bn``
    (puck-geometry / free-DOF path)."""
    f = shell_loading_vector_pure(phi_m, w_q, Bn)
    L_r = Q_c.T @ L_full @ Q_c
    f_r = Q_c.T @ f
    alpha = shell_solve_eigenfloor_pure(
        L_r,
        f_r,
        threshold=_EIGENFLOOR_THRESHOLD,
        jitter=1e-10,
    )
    return Q_c @ alpha


_beta_eigenfloor_from_bn_jitted = jax.jit(_beta_eigenfloor_from_bn_body)


def _beta_from_tf_eigenfloor_body(
    L_full: jnp.ndarray,
    quad_pts: jnp.ndarray,
    quad_n: jnp.ndarray,
    phi_m: jnp.ndarray,
    w_q: jnp.ndarray,
    g_tf: jnp.ndarray,
    gd_tf: jnp.ndarray,
    I_tf: jnp.ndarray,
) -> jnp.ndarray:
    """Modal coefficients via full-:math:`L` eigenvalue-floor solve (matches full JAX forward)."""
    gammas_tf = jnp.asarray(g_tf)
    gammadash_tf = jnp.asarray(gd_tf)
    currents_tf = jnp.asarray(I_tf)

    def Bn_at_i(i):
        B = _B_at_point_from_coil_set_pure(
            quad_pts[i],
            gammas_tf,
            gammadash_tf,
            currents_tf,
            -1,
            _EPS_BS,
        )
        return jnp.dot(B, quad_n[i])

    Bn = jax.vmap(Bn_at_i)(jnp.arange(quad_pts.shape[0]))
    f = shell_loading_vector_pure(phi_m, w_q, Bn)
    return shell_solve_eigenfloor_pure(
        L_full,
        f,
        threshold=_EIGENFLOOR_THRESHOLD,
        jitter=1e-10,
    )


_beta_from_tf_eigenfloor_jitted = jax.jit(_beta_from_tf_eigenfloor_body)


def _B_eval_from_tf_body(
    L_full: jnp.ndarray,
    Qm: jnp.ndarray,
    quad_pts: jnp.ndarray,
    quad_n: jnp.ndarray,
    phi_m: jnp.ndarray,
    w_q: jnp.ndarray,
    K_b: jnp.ndarray,
    g_tf: jnp.ndarray,
    gd_tf: jnp.ndarray,
    I_tf: jnp.ndarray,
    pts_eval: jnp.ndarray,
) -> jnp.ndarray:
    """Passive bulk B at eval points; TF-only VJP fast path."""
    b = _beta_from_tf_body(
        L_full,
        Qm,
        quad_pts,
        quad_n,
        phi_m,
        w_q,
        g_tf,
        gd_tf,
        I_tf,
    )
    return shell_biot_savart_pure(
        K_b,
        quad_pts,
        w_q,
        b,
        jnp.asarray(pts_eval),
        eps=_EPS_BS,
    )


_B_eval_jitted = jax.jit(_B_eval_from_tf_body)


def _B_eval_from_bn_body(
    L_full: jnp.ndarray,
    Qm: jnp.ndarray,
    quad_pts: jnp.ndarray,
    phi_m: jnp.ndarray,
    w_q: jnp.ndarray,
    K_b: jnp.ndarray,
    Bn: jnp.ndarray,
    pts_eval: jnp.ndarray,
) -> jnp.ndarray:
    """Passive bulk B at eval points given ``Bn`` (avoids JAX Biot-Savart on TF)."""
    b = _beta_from_bn_body(L_full, Qm, phi_m, w_q, Bn)
    return shell_biot_savart_pure(
        K_b,
        quad_pts,
        w_q,
        b,
        jnp.asarray(pts_eval),
        eps=_EPS_BS,
    )


_B_eval_from_bn_jitted = jax.jit(_B_eval_from_bn_body)


def _B_eval_full_body(
    local_pts_stack: jnp.ndarray,
    local_K_stack: jnp.ndarray,
    local_n_stack: jnp.ndarray,
    local_w_stack: jnp.ndarray,
    local_phi_stack: jnp.ndarray,
    Q_c: jnp.ndarray,
    centers_all: jnp.ndarray,
    quats_all: jnp.ndarray,
    g_tf: jnp.ndarray,
    gd_tf: jnp.ndarray,
    I_tf: jnp.ndarray,
    pts_eval: jnp.ndarray,
    delta_reg: float,
    eigenfloor_threshold: float,
    adaptive_self_reg: bool = False,
) -> jnp.ndarray:
    """Full forward: assemble :math:`L` in JAX, rim-continuity-projected
    eigenfloor solve, and Biot–Savart.

    The projector ``Q_c`` (built once in :meth:`PSCBulkArray._rebuild` via
    :func:`~simsopt.field.puck_basis.build_continuity_constraint`) is a
    dense block-diagonal matrix whose columns span the null space of the
    per-puck rim-continuity constraints (paper eq 55).  It depends only on
    per-puck :math:`(R, t, m_{fourier}, l_{zernike}, k_{chebyshev}, n_\\rho,
    n_\\phi, n_z, n_{phi\\_rim})` and is therefore constant w.r.t. the VJP
    variables (puck centers, quaternions, TF gammas / currents); the reduced
    eigenfloor solve ``Q_c^T L Q_c \\alpha = Q_c^T f`` gives a well-posed
    ideal-diamagnet solution in the continuous basis.

    Args:
        adaptive_self_reg: When ``True``, use the analytic flat-disc
            per-quadrature-cell regularization
            :math:`\\delta_i = (3\\pi^{3/2}/8)\\sqrt{w_i}` (see
            :data:`~simsopt.field.bulk_inductance._SELF_REG_COEFF`) inside
            the inline ``L``-assembly, matching
            :func:`~simsopt.field.bulk_inductance.shell_inductance_matrix_blockwise`.
            This must be ``True`` whenever the precomputed ``self._L_full``
            used to solve for ``beta`` also used the adaptive path;
            otherwise ``self.beta`` and the :math:`\\beta` implicit in this
            re-solve become inconsistent and the induced field reported by
            :meth:`PSCBulkArray.B_at_points` silently disagrees with
            :meth:`PSCBulkArray.get_shell_currents`.  Declared static so
            JAX caches a separate JIT per branch.
    """
    n_pucks = centers_all.shape[0]
    nd = local_K_stack.shape[2]
    n_dof_total = n_pucks * nd
    nq = local_pts_stack.shape[1]

    def transform_one(p):
        Rmat = _rotation_matrix_from_quat_jax(quats_all[p])
        pts_g = (Rmat @ local_pts_stack[p].T).T + centers_all[p]
        K_g = jnp.einsum("ij,qkj->qki", Rmat, local_K_stack[p])
        n_g = (Rmat @ local_n_stack[p].T).T
        return pts_g, K_g, n_g

    puck_data = jax.lax.map(transform_one, jnp.arange(n_pucks))
    pts_stack = puck_data[0]
    K_stack = puck_data[1]
    n_stack = puck_data[2]

    def L_block(pair):
        i, j = pair[0], pair[1]
        r = pts_stack[i, :, None, :] - pts_stack[j, None, :, :]
        if adaptive_self_reg:
            w_i = local_w_stack[i]
            w_j = local_w_stack[j]
            delta_i = _SELF_REG_COEFF * jnp.sqrt(w_i)
            delta_j = _SELF_REG_COEFF * jnp.sqrt(w_j)
            delta_pair = 0.5 * (delta_i[:, None] + delta_j[None, :])
            dist = jnp.sqrt(jnp.sum(r**2, axis=-1) + delta_pair**2)
        else:
            dist = jnp.sqrt(jnp.sum(r**2, axis=-1) + delta_reg**2)
        dot = jnp.einsum("iax,jbx->ijab", K_stack[i], K_stack[j])
        kernel = dot / dist[..., None, None]
        return MU0_OVER_4PI * jnp.einsum(
            "ijab,i,j->ab",
            kernel,
            local_w_stack[i],
            local_w_stack[j],
        )

    pair_i, pair_j = jnp.meshgrid(
        jnp.arange(n_pucks),
        jnp.arange(n_pucks),
        indexing="ij",
    )
    pairs = jnp.stack([pair_i.ravel(), pair_j.ravel()], axis=1)
    L_blocks_flat = jax.lax.map(L_block, pairs)
    L_blocks = L_blocks_flat.reshape(n_pucks, n_pucks, nd, nd)
    L = L_blocks.transpose(0, 2, 1, 3).reshape(n_dof_total, n_dof_total)

    gammas_tf = jnp.asarray(g_tf)
    gammadash_tf = jnp.asarray(gd_tf)
    currents_tf = jnp.asarray(I_tf)
    all_flat_pts = pts_stack.reshape(-1, 3)
    all_flat_n = n_stack.reshape(-1, 3)

    def Bn_at_q(q_idx):
        B = _B_at_point_from_coil_set_pure(
            all_flat_pts[q_idx],
            gammas_tf,
            gammadash_tf,
            currents_tf,
            -1,
            _EPS_BS,
        )
        return jnp.dot(B, all_flat_n[q_idx])

    Bn = jax.vmap(Bn_at_q)(jnp.arange(all_flat_pts.shape[0]))
    Bn_per = Bn.reshape(n_pucks, nq)
    f = jnp.sum(
        local_phi_stack * (local_w_stack * Bn_per)[..., None],
        axis=1,
    ).reshape(-1)

    L_r = Q_c.T @ L @ Q_c
    f_r = Q_c.T @ f
    alpha = shell_solve_eigenfloor_pure(
        L_r,
        f_r,
        threshold=eigenfloor_threshold,
        jitter=1e-10,
    )
    beta = Q_c @ alpha
    beta_per = beta.reshape(n_pucks, nd)
    K_at_quad = jnp.einsum("pqdi,pd->pqi", K_stack, beta_per)
    K_flat = K_at_quad.reshape(-1, 3)
    all_flat_w = local_w_stack.reshape(-1)

    def B_at_x(x):
        r = x[None, :] - all_flat_pts
        rn = jnp.sqrt(jnp.sum(r**2, axis=-1) + _EPS_BS**2)
        integrand = jnp.cross(K_flat, r) / (rn[:, None] ** 3)
        return MU0_OVER_4PI * jnp.sum(integrand * all_flat_w[:, None], axis=0)

    return jax.vmap(B_at_x)(jnp.asarray(pts_eval))


_B_eval_full_jitted = jax.jit(
    _B_eval_full_body,
    static_argnames=("delta_reg", "eigenfloor_threshold", "adaptive_self_reg"),
)


def _vjp_tf_run(
    L_full: jnp.ndarray,
    Qm: jnp.ndarray,
    quad_pts: jnp.ndarray,
    quad_n: jnp.ndarray,
    phi_m: jnp.ndarray,
    w_q: jnp.ndarray,
    K_b: jnp.ndarray,
    g_tf: jnp.ndarray,
    gd_tf: jnp.ndarray,
    I_tf: jnp.ndarray,
    pts_eval: jnp.ndarray,
    v_B: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """TF-only VJP; uses non-JIT forward body so AD is exact inside ``jax.jit``."""

    def fwd(g, gd, I):
        return _B_eval_from_tf_body(
            L_full,
            Qm,
            quad_pts,
            quad_n,
            phi_m,
            w_q,
            K_b,
            g,
            gd,
            I,
            pts_eval,
        )

    _, vjp_fn = vjp(fwd, g_tf, gd_tf, I_tf)
    return vjp_fn(v_B)


_vjp_tf_jitted = jax.jit(_vjp_tf_run)


def _vjp_full_run(
    local_pts_stack: jnp.ndarray,
    local_K_stack: jnp.ndarray,
    local_n_stack: jnp.ndarray,
    local_w_stack: jnp.ndarray,
    local_phi_stack: jnp.ndarray,
    Q_c: jnp.ndarray,
    centers_all: jnp.ndarray,
    quats_all: jnp.ndarray,
    g_tf: jnp.ndarray,
    gd_tf: jnp.ndarray,
    I_tf: jnp.ndarray,
    pts_eval: jnp.ndarray,
    v_B: jnp.ndarray,
    delta_reg: float,
    eigenfloor_threshold: float,
    adaptive_self_reg: bool = False,
) -> Tuple[
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
]:
    """Full VJP w.r.t. puck centers, quaternions, and TF arrays.

    ``Q_c`` (rim-continuity projector) is constant w.r.t. VJP variables.

    Args:
        adaptive_self_reg: Same meaning as in :func:`_B_eval_full_body`;
            must match ``PSCBulkArray.adaptive_self_reg`` so that the
            forward assembly used inside the reverse-mode pass is
            consistent with the one used by ``_rebuild``.
    """

    def fwd(c_all, q_all, g, gd, I):
        return _B_eval_full_body(
            local_pts_stack,
            local_K_stack,
            local_n_stack,
            local_w_stack,
            local_phi_stack,
            Q_c,
            c_all,
            q_all,
            g,
            gd,
            I,
            pts_eval,
            delta_reg,
            eigenfloor_threshold,
            adaptive_self_reg,
        )

    _, vjp_fn = vjp(fwd, centers_all, quats_all, g_tf, gd_tf, I_tf)
    return vjp_fn(v_B)


_vjp_full_jitted = jax.jit(
    _vjp_full_run,
    static_argnames=("delta_reg", "eigenfloor_threshold", "adaptive_self_reg"),
)


# ======================================================================
# PSCBulkArray
# ======================================================================


class PSCBulkArray(Optimizable):
    """
    Passive superconducting bulk cylinders in the ideal-diamagnetic limit.

    Orientation is stored as a **scalar-first quaternion** ``[q0, qi, qj, qk]``
    matching :class:`~simsopt.geo.curveplanarfourier.CurvePlanarFourier`.

    **Optimizable DOFs** (all fixed by default):

    For each base puck *i*, 9 DOFs are registered::

        center_x{i}, center_y{i}, center_z{i},
        q0_{i}, qi_{i}, qj_{i}, qk_{i},
        R{i}, t{i}

    Gradients w.r.t. center and quaternion DOFs are computed analytically
    via JAX VJPs.  Gradients w.r.t. R and t are **not yet** computed
    analytically; those components are zero in the VJP.

    Args:
        puck_centers: ``(N, 3)`` centers in meters.
        puck_axes: ``(N, 3)`` unit normal vectors (bottom to top cap).
            Internally converted to quaternions.
        puck_radii: ``(N,)`` radii.
        puck_thicknesses: ``(N,)`` thicknesses in meters.
        coils_TF: list of TF :class:`~simsopt.field.coil.Coil` objects.
        eval_points: ``(M, 3)`` points where the passive field is evaluated.
        m_fourier, l_zernike, k_chebyshev: basis resolution.
        n_rho, n_phi, n_z: quadrature resolution.
        nfp: number of field periods (replication of pucks).
        stellsym: whether to apply stellarator symmetry to puck positions.
        plasma_flux: optional extra flux per puck (not used in v1).
        regularization_delta: distance regularization for self-terms.
        default_thickness: fallback thickness in meters.
        n_phi_rim: number of azimuthal collocation points used to enforce
            rim continuity (:math:`H^1(\\Sigma_i)/\\mathbb R`, eq 55 of the
            passive-bulk note) between the disk and side-wall basis patches.
            Defaults to ``max(2*m_fourier+1, 8)`` (Nyquist-safe for the
            highest azimuthal mode retained in the disk/side bases).
        null_space_threshold: relative eigenvalue cutoff for the gauge /
            null-space projection of ``L`` in the fixed-DOF path.  Defaults
            to ``1e-10``.
        eigenfloor_threshold: relative eigenvalue floor used by the
            regularized Cholesky solve in the free-DOF JAX path.  Defaults
            to the module-level ``_EIGENFLOOR_THRESHOLD = 1e-10``.
        adaptive_self_reg: If ``True`` (the default), the
            inductance-matrix self-terms use a quadrature-cell-scaled
            regularization :math:`\\delta_i = (3\\pi^{3/2}/8)\\sqrt{w_i}`
            matching the analytic flat-disc self-integral
            :math:`(8/3)(w_i/\\pi)^{3/2}` on the coincident cell.  This
            removes the ~:math:`10^{4}-10^{5}` overcount of coincident
            contributions caused by the legacy fixed scalar
            ``regularization_delta`` in the :math:`1/|r-r'|` double
            integral and is required to recover analytic magnitudes
            (Lorenz short-solenoid, Smythe thin-disc) and physically
            correct induced fields on the plasma surface.  Pass
            ``False`` only as an explicit opt-in to the legacy uniform
            :math:`\\sqrt{r^2+\\delta^2}` regularization for strict
            backward-compatibility regression testing; new applications
            should leave this at the default.
        strict_rim_continuity: If ``True``, require that the per-puck
            rim-continuity constraint
            :math:`H^1(\\Sigma_i)/\\mathbb R` (paper eq 55) has a
            non-trivial kernel on **every** puck and raise
            :class:`ValueError` otherwise.  The default (``False``) only
            emits a :class:`UserWarning` and falls back to the raw basis
            (no rim continuity enforced) - useful for low-order smoke
            tests, but dangerous for production optimizations where the
            lack of continuity silently changes the solution.  Production
            examples should set ``strict_rim_continuity=True``.
    """

    def __init__(
        self,
        puck_centers: np.ndarray,
        puck_axes: np.ndarray,
        puck_radii: np.ndarray,
        puck_thicknesses: np.ndarray,
        coils_TF,
        eval_points: np.ndarray,
        m_fourier: int = 4,
        l_zernike: int = 6,
        k_chebyshev: int = 4,
        n_rho: int = 10,
        n_phi: int = 12,
        n_z: int = 6,
        nfp: int = 1,
        stellsym: bool = False,
        plasma_flux: Optional[np.ndarray] = None,
        regularization_delta: float = 1e-6,
        default_thickness: float = 0.02,
        n_phi_rim: Optional[int] = None,
        null_space_threshold: float = 1e-10,
        eigenfloor_threshold: float = _EIGENFLOOR_THRESHOLD,
        adaptive_self_reg: bool = True,
        strict_rim_continuity: bool = False,
    ):
        self.coils_TF = list(coils_TF)
        self.eval_points = np.asarray(eval_points, dtype=float, order="C")
        self.nfp = int(nfp)
        self.stellsym = bool(stellsym)
        self.regularization_delta = float(regularization_delta)
        self.adaptive_self_reg = bool(adaptive_self_reg)
        self.m_fourier = m_fourier
        self.l_zernike = l_zernike
        self.k_chebyshev = k_chebyshev
        self._n_rho = n_rho
        self._n_phi = n_phi
        self._n_z = n_z
        self._plasma_flux = plasma_flux
        if n_phi_rim is None:
            n_phi_rim = max(2 * int(m_fourier) + 1, 8)
        self._n_phi_rim = int(n_phi_rim)
        self._null_space_threshold = float(null_space_threshold)
        self._eigenfloor_threshold = float(eigenfloor_threshold)
        self._strict_rim_continuity = bool(strict_rim_continuity)

        centers = np.atleast_2d(np.asarray(puck_centers, dtype=float))
        axes = np.atleast_2d(np.asarray(puck_axes, dtype=float))
        radii = np.atleast_1d(np.asarray(puck_radii, dtype=float))
        ths = np.atleast_1d(np.asarray(puck_thicknesses, dtype=float))
        if ths.size == 0:
            ths = np.full(len(centers), default_thickness)
        ths = np.where(ths <= 0, default_thickness, ths)

        self._n_base_pucks = len(centers)

        # Convert axes to quaternions
        quats = np.array([_axis_to_quaternion(axes[i]) for i in range(len(axes))])

        dof_values: List[float] = []
        dof_names: List[str] = []
        for i in range(self._n_base_pucks):
            dof_values.extend(
                [
                    centers[i, 0],
                    centers[i, 1],
                    centers[i, 2],
                    quats[i, 0],
                    quats[i, 1],
                    quats[i, 2],
                    quats[i, 3],
                    radii[i],
                    ths[i],
                ]
            )
            dof_names.extend(
                [
                    f"center_x{i}",
                    f"center_y{i}",
                    f"center_z{i}",
                    f"q0_{i}",
                    f"qi_{i}",
                    f"qj_{i}",
                    f"qk_{i}",
                    f"R{i}",
                    f"t{i}",
                ]
            )

        fixed = [True] * len(dof_values)
        Optimizable.__init__(
            self,
            x0=np.array(dof_values),
            names=dof_names,
            fixed=fixed,
            depends_on=self.coils_TF,
        )

        self._geom_hash: Optional[int] = None
        self._local_stacks_valid: bool = False
        self._structural_key: Optional[tuple] = None
        self._rebuild()
        self._field = PassiveBulkField(self)

    # ------------------------------------------------------------------
    # Optimizable overrides
    # ------------------------------------------------------------------

    @staticmethod
    def _is_zero_vjp_dof(name: str) -> bool:
        """Return ``True`` for DOFs whose VJP is identically zero.

        Currently the VJP w.r.t. puck radius (``R{i}``) and thickness
        (``t{i}``) is not implemented and returns zero; unfixing those
        DOFs produces no gradient signal.
        """
        return name.startswith("R") or name.startswith("t")

    def _warn_zero_vjp(self, names) -> None:
        """Emit a :class:`UserWarning` if any ``names`` has zero VJP."""
        import warnings

        zero = [n for n in names if self._is_zero_vjp_dof(n)]
        if zero:
            warnings.warn(
                "PSCBulkArray: unfixing DOF(s) "
                f"{zero} whose gradient is not implemented (zero VJP). "
                "Optimizers will see no gradient signal for these "
                "variables; fix them unless you are deliberately "
                "exploring with a gradient-free method.",
                stacklevel=3,
            )

    def local_unfix_all(self) -> None:
        """Unfix all local DOFs, warning about zero-VJP R/t DOFs."""
        full_names = list(self.local_full_dof_names)
        free_flags = np.asarray(self._dofs._free, dtype=bool)
        currently_fixed = [n for n, f in zip(full_names, free_flags) if not bool(f)]
        self._warn_zero_vjp(currently_fixed)
        super().local_unfix_all()

    def unfix(self, key) -> None:
        """Unfix a DOF by name or index, warning if its VJP is zero."""
        if isinstance(key, str):
            name = key
        else:
            try:
                name = list(self.local_full_dof_names)[int(key)]
            except Exception:
                name = None
        if name is not None:
            self._warn_zero_vjp([name])
        super().unfix(key)

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def _get_base_puck_geometry(self):
        """Extract ``(centers, quats, radii, thicknesses)`` from current DOFs."""
        x = np.array(self.local_full_x)
        n = self._n_base_pucks
        centers = np.zeros((n, 3))
        quats = np.zeros((n, 4))
        radii = np.zeros(n)
        thicknesses = np.zeros(n)
        for i in range(n):
            off = i * _DOFS_PER_PUCK
            centers[i] = x[off : off + 3]
            quats[i] = x[off + 3 : off + 7]
            radii[i] = x[off + 7]
            thicknesses[i] = x[off + 8]
        return centers, quats, radii, thicknesses

    def _replicate_pucks(self, centers, quats, radii, thicknesses):
        """Apply nfp + stellsym to base pucks.

        Returns:
            all_pucks: list of ``(center, axis, R, t)``
            all_quats: list of quaternions per puck
            base_indices: which base puck each copy came from
            center_jacobians: 3x3 transform from base center to copy center
            quat_jacobians: 4x4 transform from base quaternion to copy quaternion
        """
        all_pucks: List[Tuple[np.ndarray, np.ndarray, float, float]] = []
        all_quats_list: List[np.ndarray] = []
        base_indices: List[int] = []
        center_jacobians: List[np.ndarray] = []
        quat_jacobians: List[np.ndarray] = []

        for i in range(len(centers)):
            c = centers[i]
            q = quats[i]
            R_val, t_val = float(radii[i]), float(thicknesses[i])
            for jfp in range(self.nfp):
                angle = 2.0 * np.pi * jfp / self.nfp
                rot3 = np.array(
                    [
                        [np.cos(angle), -np.sin(angle), 0.0],
                        [np.sin(angle), np.cos(angle), 0.0],
                        [0.0, 0.0, 1.0],
                    ]
                )
                q_rot = np.array([np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)])

                c2 = rot3 @ c
                q2 = _quat_multiply(q_rot, q)
                ax2 = _quaternion_to_axis(q2)
                all_pucks.append((c2, ax2, R_val, t_val))
                all_quats_list.append(q2)
                base_indices.append(i)
                center_jacobians.append(rot3)
                quat_jacobians.append(_quat_left_mult_matrix(q_rot))

                if self.stellsym:
                    S = np.diag([1.0, -1.0, -1.0])
                    q_stell = np.array([0.0, 1.0, 0.0, 0.0])
                    c3 = S @ c2
                    q3 = _quat_multiply(q_stell, q2)
                    ax3 = _quaternion_to_axis(q3)
                    all_pucks.append((c3, ax3, R_val, t_val))
                    all_quats_list.append(q3)
                    base_indices.append(i)
                    center_jacobians.append(S @ rot3)
                    quat_jacobians.append(
                        _quat_left_mult_matrix(q_stell) @ _quat_left_mult_matrix(q_rot)
                    )

        return (
            all_pucks,
            all_quats_list,
            base_indices,
            center_jacobians,
            quat_jacobians,
        )

    # ------------------------------------------------------------------
    # Build / rebuild
    # ------------------------------------------------------------------

    def _build_rim_continuity_projector(self, n_dof_total: int) -> np.ndarray:
        r"""Build the orthonormal basis of the rim-continuity subspace.

        For each puck :math:`i`, :func:`build_continuity_constraint` returns a
        matrix :math:`C_i` of shape ``(2 * n_phi_rim, n_dof_i)`` whose null
        space encodes potentials that are continuous across the top and
        bottom rim curves (:math:`H^1(\\Sigma_i)/\\mathbb R`, paper eq 55
        and the text immediately after eq 57).  These per-puck constraints
        are assembled block-diagonally in the global DOF ordering.

        The returned projector :math:`Q_c` is a tall orthonormal matrix of
        shape ``(n_dof_total, n_free)`` whose columns span ``ker(C)``.  It
        is computed block-by-block (one SVD per puck) which keeps the cost
        linear in ``n_pucks``.

        If any block has zero kernel (degenerate basis at very low order)
        the behavior depends on ``self._strict_rim_continuity``:

        * if ``False`` (default), the whole projector falls back to the
          identity with a one-time :class:`UserWarning`, preserving
          backward compatibility with low-order smoke tests;
        * if ``True``, a :class:`ValueError` is raised so that production
          optimizations cannot silently run without continuity.

        Args:
            n_dof_total: global DOF count (sum of ``n_dof_i``).

        Returns:
            ``Q_c`` of shape ``(n_dof_total, n_free)`` with orthonormal
            columns (``Q_c.T @ Q_c = I``).
        """
        import warnings

        Q_c_blocks: List[np.ndarray] = []
        row_offsets: List[int] = []
        col_offsets: List[int] = []
        n_free_total = 0
        degenerate = False
        for pidx, basis in enumerate(self._basis_per_puck):
            _, _, R_p, t_p = self._all_pucks[pidx]
            C_p = build_continuity_constraint(
                basis,
                float(R_p),
                float(t_p),
                n_rim=self._n_phi_rim,
            )
            nd_p = C_p.shape[1]
            _, sv, Vt = np.linalg.svd(C_p, full_matrices=True)
            rtol = 1e-10 * (float(sv[0]) if sv.size > 0 else 1.0)
            n_nonzero = int(np.sum(sv > rtol))
            n_free_p = nd_p - n_nonzero
            if n_free_p <= 0:
                degenerate = True
                break
            Q_p = Vt[n_nonzero:].T
            Q_c_blocks.append(Q_p)
            row_offsets.append(self._dof_offsets[pidx])
            col_offsets.append(n_free_total)
            n_free_total += n_free_p

        if degenerate or n_free_total == 0:
            msg = (
                "PSCBulkArray: rim-continuity constraint has zero-rank "
                "kernel for at least one puck; the raw basis would be "
                "used (no rim continuity enforced).  Increase "
                "m_fourier / l_zernike / k_chebyshev to use continuity."
            )
            if getattr(self, "_strict_rim_continuity", False):
                raise ValueError(msg + "  strict_rim_continuity=True was requested.")
            warnings.warn(msg, stacklevel=3)
            return np.eye(n_dof_total)

        Q_c = np.zeros((n_dof_total, n_free_total))
        for pidx, Q_p in enumerate(Q_c_blocks):
            r0 = row_offsets[pidx]
            r1 = r0 + Q_p.shape[0]
            c0 = col_offsets[pidx]
            c1 = c0 + Q_p.shape[1]
            Q_c[r0:r1, c0:c1] = Q_p
        return Q_c

    def _rebuild(self) -> None:
        """Recompute all derived quantities from current DOFs.

        When the optional attribute ``self.exact_disc_faces`` is set to
        ``True`` (via ``obj.exact_disc_faces = True`` after construction;
        default ``False`` via :func:`getattr`), the flat top/bottom
        disc-face sub-blocks of the inductance matrix ``L`` are
        overwritten with the semi-analytic exact-up-to-1D-quadrature
        values produced by
        :func:`simsopt.field.disc_self_inductance.assemble_puck_disc_faces_L`
        and
        :func:`simsopt.field.disc_self_inductance.disc_disc_cross_block`.
        Side-wall and non-coaxial mutual entries remain on the existing
        regularized numerical path.  The number of radial nodes is
        controlled by the optional attribute ``self.n_radial_disc``
        (default ``32``).
        """
        centers, quats, radii, thicknesses = self._get_base_puck_geometry()
        structural_key = (
            tuple(np.asarray(radii, dtype=float).ravel()),
            tuple(np.asarray(thicknesses, dtype=float).ravel()),
        )
        structural_changed = structural_key != self._structural_key
        self._structural_key = structural_key
        if structural_changed:
            self._local_stacks_valid = False

        (all_pucks, all_quats_list, base_indices, center_jacs, quat_jacs) = (
            self._replicate_pucks(
                centers,
                quats,
                radii,
                thicknesses,
            )
        )

        self._all_pucks = all_pucks
        self._all_pucks_quats = np.array(all_quats_list)
        self._all_puck_base_indices = base_indices
        self._center_jacobians = center_jacs
        self._quat_jacobians = quat_jacs

        self._basis_per_puck: List[PuckBasisData] = []
        self._dof_offsets: List[int] = []
        K_per_puck_global: List[np.ndarray] = []
        pts_per_puck_global: List[np.ndarray] = []
        weights_per_puck: List[np.ndarray] = []
        normals_per_puck_global: List[np.ndarray] = []
        phi_per_puck: List[np.ndarray] = []

        n_dof_total = 0
        for idx, (c, ax, R_val, t_val) in enumerate(all_pucks):
            basis = build_puck_shell_basis(
                R_val,
                t_val,
                m_fourier=self.m_fourier,
                l_zernike=self.l_zernike,
                k_chebyshev=self.k_chebyshev,
                n_rho=self._n_rho,
                n_phi=self._n_phi,
                n_z=self._n_z,
            )
            self._basis_per_puck.append(basis)
            Rmat = _rotation_matrix_from_quat(all_quats_list[idx])
            pts_g = (Rmat @ basis.quad_points_local.T).T + c[None, :]
            K_g = np.einsum("ij,qkj->qki", Rmat, basis.k_basis_local)
            n_g = (Rmat @ basis.quad_normals_local.T).T

            nd = K_g.shape[1]
            self._dof_offsets.append(n_dof_total)
            n_dof_total += nd

            K_per_puck_global.append(K_g)
            pts_per_puck_global.append(pts_g)
            weights_per_puck.append(basis.quad_weights)
            normals_per_puck_global.append(n_g)
            phi_per_puck.append(basis.phi_values)

        self._n_dof_total = n_dof_total
        self._K_per_puck = K_per_puck_global
        self._pts_per_puck = pts_per_puck_global
        self._weights_per_puck_list = weights_per_puck

        # Monolithic arrays for fast-path Biot-Savart and diagnostics
        quad_points = np.vstack(pts_per_puck_global)
        quad_weights = np.concatenate(weights_per_puck)
        quad_normals = np.vstack(normals_per_puck_global)
        K_basis = np.zeros((quad_points.shape[0], n_dof_total, 3))
        phi_mat = np.zeros((quad_points.shape[0], n_dof_total))

        row0 = 0
        self._quad_row_ranges: List[Tuple[int, int]] = []
        for pidx in range(len(all_pucks)):
            K_g = K_per_puck_global[pidx]
            nq = K_g.shape[0]
            self._quad_row_ranges.append((row0, row0 + nq))
            d0 = self._dof_offsets[pidx]
            d1 = d0 + K_g.shape[1]
            K_basis[row0 : row0 + nq, d0:d1, :] = K_g
            phi_mat[row0 : row0 + nq, d0:d1] = phi_per_puck[pidx]
            row0 += nq

        self._quad_points = quad_points
        self._quad_weights = quad_weights
        self._quad_normals = quad_normals
        self._K_basis = K_basis
        self._phi_mat = phi_mat

        # Blockwise L assembly (NumPy)
        exact_disc_faces = bool(getattr(self, "exact_disc_faces", False))
        if exact_disc_faces:
            # Per-puck (center, axis) in the global frame used by the
            # semi-analytic disc assembler to detect coaxial pairs.
            disc_centers_axes = []
            disc_Rts = []
            for idx, (c, ax, R_val, t_val) in enumerate(all_pucks):
                Rmat = _rotation_matrix_from_quat(all_quats_list[idx])
                axis_global = Rmat @ np.array([0.0, 0.0, 1.0])
                disc_centers_axes.append(
                    (np.asarray(c, dtype=float), np.asarray(axis_global, dtype=float))
                )
                disc_Rts.append((float(R_val), float(t_val)))
            L_np = shell_inductance_matrix_blockwise(
                K_per_puck_global,
                pts_per_puck_global,
                weights_per_puck,
                self._dof_offsets,
                n_dof_total,
                delta_reg=self.regularization_delta,
                adaptive_self_reg=self.adaptive_self_reg,
                exact_disc_faces=True,
                disc_bases=self._basis_per_puck,
                disc_Rts=disc_Rts,
                disc_centers_axes=disc_centers_axes,
                n_radial_disc=int(getattr(self, "n_radial_disc", 32)),
            )
        else:
            L_np = shell_inductance_matrix_blockwise(
                K_per_puck_global,
                pts_per_puck_global,
                weights_per_puck,
                self._dof_offsets,
                n_dof_total,
                delta_reg=self.regularization_delta,
                adaptive_self_reg=self.adaptive_self_reg,
            )
        self._L_full = L_np

        # Rim continuity (paper eqs 55, 56-57): enforce g continuous across
        # both rims of each puck shell before gauge fixing.  Block-diagonal
        # across pucks; each block has ``2 * n_phi_rim`` constraint rows
        # (top rim and bottom rim) by ``n_dof_p`` columns.
        Q_c = self._build_rim_continuity_projector(n_dof_total)
        self._Q_c = Q_c

        # Gauge projection on top of the continuity-restricted subspace:
        # drop the remaining exact null modes (constant per puck) of L.
        L_c = Q_c.T @ L_np @ Q_c
        Q_L = null_space_projection_matrix(
            L_c,
            threshold=self._null_space_threshold,
        )
        self._Q = Q_c @ Q_L
        Lr = self._Q.T @ L_np @ self._Q
        self._L_red = Lr

        # JAX arrays for module-level JIT (no new ``jax.jit`` closures per rebuild)
        self._setup_jax()

        # Solve
        self.beta = self._solve_beta(self._tf_arrays())
        self._geom_hash = hash(tuple(self.local_full_x))

    # ------------------------------------------------------------------
    # JAX fast path (TF-only VJP, pre-computed L and Q)
    # ------------------------------------------------------------------

    def _tf_arrays(self):
        gammas = np.array([c.curve.gamma() for c in self.coils_TF])
        gammadashs = np.array([c.curve.gammadash() for c in self.coils_TF])
        currents = np.array([c.current.get_value() for c in self.coils_TF])
        return gammas, gammadashs, currents

    def _setup_jax(self) -> None:
        """Store JAX views of NumPy geometry; JIT callables are module-level."""
        self._jax_L = jnp.asarray(self._L_full)
        self._jax_Q = jnp.asarray(self._Q)
        self._jax_Q_c = jnp.asarray(self._Q_c)
        self._jax_quad_pts = jnp.asarray(self._quad_points)
        self._jax_quad_n = jnp.asarray(self._quad_normals)
        self._jax_phi_m = jnp.asarray(self._phi_mat)
        self._jax_w_q = jnp.asarray(self._quad_weights)
        self._jax_K_b = jnp.asarray(self._K_basis)

    # ------------------------------------------------------------------
    # JAX full path (local basis stacks for geometry VJP)
    # ------------------------------------------------------------------

    def _ensure_jax_full(self) -> None:
        """Cache per-puck local-frame stacks for :func:`_B_eval_full_jitted`.

        Also exposes the dense rim-continuity projector ``self._Q_c`` as
        ``self._jax_Q_c`` for the free-DOF JAX path so that the constrained
        solve ``Q_c^T L Q_c \\alpha = Q_c^T f`` is used inside the VJP.
        """
        if self._local_stacks_valid:
            return
        self._jax_local_pts = jnp.stack(
            [jnp.asarray(b.quad_points_local) for b in self._basis_per_puck]
        )
        self._jax_local_K = jnp.stack(
            [jnp.asarray(b.k_basis_local) for b in self._basis_per_puck]
        )
        self._jax_local_n = jnp.stack(
            [jnp.asarray(b.quad_normals_local) for b in self._basis_per_puck]
        )
        self._jax_local_w = jnp.stack(
            [jnp.asarray(b.quad_weights) for b in self._basis_per_puck]
        )
        self._jax_local_phi = jnp.stack(
            [jnp.asarray(b.phi_values) for b in self._basis_per_puck]
        )
        self._jax_Q_c = jnp.asarray(self._Q_c)
        self._local_stacks_valid = True

    # ------------------------------------------------------------------
    # Forward computations
    # ------------------------------------------------------------------

    def _compute_bn_at_quads_numpy(self) -> np.ndarray:
        """Normal component of TF :class:`BiotSavart` field at shell quadrature (C++ kernel).

        Caches a single :class:`BiotSavart` instance on ``self._bs_bn`` and
        only calls ``set_points_cart`` when the shell quadrature has been
        regenerated (i.e. after :meth:`_rebuild`), so repeated calls during
        an optimization iteration reuse the point cache managed by the
        C++ kernel.
        """
        bs = getattr(self, "_bs_bn", None)
        pts_id = id(self._quad_points)
        if bs is None or getattr(self, "_bs_bn_pts_id", None) != pts_id:
            bs = BiotSavart(self.coils_TF)
            bs.set_points_cart(np.ascontiguousarray(self._quad_points))
            self._bs_bn = bs
            self._bs_bn_pts_id = pts_id
        B = bs.B()
        return np.sum(B * self._quad_normals, axis=1)

    def _solve_beta(self, tf_arrays) -> np.ndarray:
        """Solve :math:`L \\beta = f` for the modal coefficients ``beta``.

        Uses the **full** null-space-trimmed projector
        :math:`Q = Q_c Q_L` (rim-continuity composed with :math:`L`'s
        null-space trim; assembled in :meth:`_rebuild`) rather than the
        bare rim-continuity projector :math:`Q_c`.  Passing the full
        projector here matches what the TF-only forward JAX path
        (``_B_eval_jitted`` / ``_B_eval_from_bn_jitted``) uses
        internally to compute ``beta`` when evaluating
        :meth:`B_at_points`, so the stored ``self.beta`` is consistent
        with shell-current diagnostics (:meth:`get_shell_currents`,
        :meth:`get_equivalent_currents`) and the induced field.

        Rationale: when the semi-analytic self-block replacement
        ``exact_disc_faces=True`` is enabled, :math:`L` acquires
        genuine near-null modes (the constant-:math:`g` gauge modes)
        whose eigenvalues are at machine zero rather than smeared out
        by the regularized kernel.  The ``shell_solve_eigenfloor_pure``
        jitter floor (:math:`10^{-10}`) then amplifies any
        :math:`Q_c^{T} f` component in those directions by
        :math:`\\sim 10^{10}`, giving a cosmetic ``|beta|`` blow-up
        (those null modes produce :math:`K \\approx 0` analytically,
        so fields are unaffected, but diagnostics are misleading).
        Projecting with :math:`Q = Q_c Q_L` excludes the null space
        before the eigenfloor solve and removes the blow-up.

        Still uses the eigenvalue-floor kernel (not Cholesky) so that
        the solve remains well-defined even when the residual reduced
        matrix :math:`Q^{T} L Q` is not strictly positive definite at
        floating-point precision.
        """
        Bn = self._compute_bn_at_quads_numpy()
        return np.array(
            _beta_eigenfloor_from_bn_jitted(
                self._jax_L,
                self._jax_Q,
                self._jax_phi_m,
                self._jax_w_q,
                jnp.asarray(Bn),
            )
        )

    @property
    def biot_savart(self) -> "PassiveBulkField":
        """Passive bulk contribution as a :class:`PassiveBulkField`."""
        return self._field

    def recompute_currents(self) -> None:
        """Recompute modal coefficients after TF geometry/currents or puck DOFs change."""
        current_hash = hash(tuple(self.local_full_x))
        if current_hash != self._geom_hash:
            self._rebuild()
            # Do not replace ``self._field``: the existing :class:`PassiveBulkField`
            # still delegates to ``self.B_at_points``, which uses updated L, beta,
            # and JIT functions. Replacing the field breaks ``MagneticFieldSum``'s
            # reference to the original object and orphan cache invalidation.
        else:
            self.beta = self._solve_beta(self._tf_arrays())
        self._field.clear_cached_properties()

    def B_at_points(self, points: np.ndarray) -> np.ndarray:
        """Passive bulk B at Cartesian points ``(N, 3)``."""
        g_tf, gd_tf, I_tf = self._tf_arrays()
        pts = np.asarray(points)
        if self._has_free_puck_dofs():
            self._ensure_jax_full()
            centers_all = jnp.asarray(np.stack([p[0] for p in self._all_pucks], axis=0))
            quats_all = jnp.asarray(self._all_pucks_quats)
            return np.array(
                _B_eval_full_jitted(
                    self._jax_local_pts,
                    self._jax_local_K,
                    self._jax_local_n,
                    self._jax_local_w,
                    self._jax_local_phi,
                    self._jax_Q_c,
                    centers_all,
                    quats_all,
                    g_tf,
                    gd_tf,
                    I_tf,
                    pts,
                    float(self.regularization_delta),
                    float(self._eigenfloor_threshold),
                    bool(self.adaptive_self_reg),
                )
            )
        if _USE_JAX_TF_VJP:
            return np.array(
                _B_eval_jitted(
                    self._jax_L,
                    self._jax_Q,
                    self._jax_quad_pts,
                    self._jax_quad_n,
                    self._jax_phi_m,
                    self._jax_w_q,
                    self._jax_K_b,
                    g_tf,
                    gd_tf,
                    I_tf,
                    pts,
                )
            )
        Bn = self._compute_bn_at_quads_numpy()
        return np.array(
            _B_eval_from_bn_jitted(
                self._jax_L,
                self._jax_Q,
                self._jax_quad_pts,
                self._jax_phi_m,
                self._jax_w_q,
                self._jax_K_b,
                jnp.asarray(Bn),
                pts,
            )
        )

    def get_shell_currents(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(K, |K|)``: ``(n_quad, 3)`` sheet current from current ``beta``."""
        b = jnp.asarray(self.beta)
        K = jnp.sum(jnp.asarray(self._K_basis) * b[None, :, None], axis=1)
        return np.array(K), np.array(jnp.linalg.norm(K, axis=-1))

    def get_equivalent_currents(self) -> np.ndarray:
        """Equivalent total current per puck in Amperes.

        For each puck, integrates ``|K|`` across the side-wall height
        (thickness ``t``) and across the face diameter (``2R``), returning
        the larger of the two as the representative equivalent current::

            I_side ~ max(|K_side|) * t
            I_face ~ max(|K_face|) * 2 * R
            I_eq   = max(I_side, I_face)
        """
        K_vec, K_mag = self.get_shell_currents()
        n_pucks = len(self._all_pucks)
        I_eq = np.zeros(n_pucks)
        for p in range(n_pucks):
            r0, r1 = self._quad_row_ranges[p]
            _, _, R_val, t_val = self._all_pucks[p]
            Km = K_mag[r0:r1]
            I_side = float(np.max(Km)) * t_val
            I_face = float(np.max(Km)) * 2.0 * R_val
            I_eq[p] = max(I_side, I_face)
        return I_eq

    # ------------------------------------------------------------------
    # VJP
    # ------------------------------------------------------------------

    def _has_free_puck_dofs(self) -> bool:
        return any(self.local_dofs_free_status)

    def vjp_setup_B(self, v_B, eval_pts=None):
        """VJP of ``sum(v_B * B)`` w.r.t. all DOFs (TF + puck geometry)."""
        v_B = np.asarray(v_B).reshape(-1, 3)
        pts = eval_pts if eval_pts is not None else self.eval_points
        if self._has_free_puck_dofs():
            return self._vjp_puck_geometry(v_B, pts)
        return self._vjp_tf_only(v_B, pts)

    def _vjp_tf_only(self, v_B, pts):
        if _USE_JAX_TF_VJP:
            gammas = np.array([c.curve.gamma() for c in self.coils_TF])
            gammadashs = np.array([c.curve.gammadash() for c in self.coils_TF])
            currents = np.array([c.current.get_value() for c in self.coils_TF])
            vg, vgd, vI = _vjp_tf_jitted(
                self._jax_L,
                self._jax_Q,
                self._jax_quad_pts,
                self._jax_quad_n,
                self._jax_phi_m,
                self._jax_w_q,
                self._jax_K_b,
                jnp.asarray(gammas),
                jnp.asarray(gammadashs),
                jnp.asarray(currents),
                jnp.asarray(pts),
                jnp.asarray(v_B),
            )
            vg, vgd, vI = np.asarray(vg), np.asarray(vgd), np.asarray(vI)
            return sum(
                self.coils_TF[i].vjp(vg[i], vgd[i], np.asarray([vI[i]]))
                for i in range(len(self.coils_TF))
            )
        return self._vjp_tf_only_analytic(v_B, pts)

    def _vjp_tf_only_analytic(self, v_B, pts) -> Derivative:
        """TF VJP: adjoint through reduced shell solve + :class:`BiotSavart` (C++)."""
        v_B = np.asarray(v_B).reshape(-1, 3)
        pts = np.asarray(pts)
        K_b = jnp.asarray(self._K_basis)
        quad_pts = jnp.asarray(self._quad_points)
        w_q = jnp.asarray(self._quad_weights)
        beta = jnp.asarray(self.beta)

        def fwd_bs(b: jnp.ndarray) -> jnp.ndarray:
            return shell_biot_savart_pure(
                K_b,
                quad_pts,
                w_q,
                b,
                jnp.asarray(pts),
                eps=_EPS_BS,
            )

        _, vjp_bs = vjp(fwd_bs, beta)
        lam_beta = np.asarray(vjp_bs(jnp.asarray(v_B))[0])

        Q = self._Q
        Lr = self._L_red
        qt_lam = Q.T @ lam_beta
        lam_r = np.linalg.solve(Lr.T, qt_lam)
        lam_f = Q @ lam_r

        lam_Bn = self._quad_weights * (self._phi_mat @ lam_f)
        v_quad = lam_Bn[:, np.newaxis] * self._quad_normals

        bs = BiotSavart(self.coils_TF)
        bs.set_points_cart(np.ascontiguousarray(self._quad_points))
        return bs.B_vjp(v_quad)

    def _vjp_puck_geometry(self, v_B, pts):
        """VJP w.r.t. puck center + quaternion + TF DOFs via full JAX forward."""
        self._ensure_jax_full()
        centers_all = np.stack([p[0] for p in self._all_pucks], axis=0)
        quats_all = self._all_pucks_quats
        gammas = np.array([c.curve.gamma() for c in self.coils_TF])
        gammadashs = np.array([c.curve.gammadash() for c in self.coils_TF])
        currents = np.array([c.current.get_value() for c in self.coils_TF])
        vc, vq, vg, vgd, vI = _vjp_full_jitted(
            self._jax_local_pts,
            self._jax_local_K,
            self._jax_local_n,
            self._jax_local_w,
            self._jax_local_phi,
            self._jax_Q_c,
            jnp.asarray(centers_all),
            jnp.asarray(quats_all),
            jnp.asarray(gammas),
            jnp.asarray(gammadashs),
            jnp.asarray(currents),
            jnp.asarray(pts),
            jnp.asarray(v_B),
            float(self.regularization_delta),
            float(self._eigenfloor_threshold),
            bool(self.adaptive_self_reg),
        )
        vc = np.asarray(vc)
        vq = np.asarray(vq)
        vg = np.asarray(vg)
        vgd = np.asarray(vgd)
        vI = np.asarray(vI)

        grad_local = np.zeros(self._n_base_pucks * _DOFS_PER_PUCK)
        for j in range(len(self._all_pucks)):
            base_idx = self._all_puck_base_indices[j]
            off = base_idx * _DOFS_PER_PUCK
            grad_local[off : off + 3] += self._center_jacobians[j].T @ vc[j]
            grad_local[off + 3 : off + 7] += self._quat_jacobians[j].T @ vq[j]

        vjp_tf = sum(
            self.coils_TF[i].vjp(vg[i], vgd[i], np.asarray([vI[i]]))
            for i in range(len(self.coils_TF))
        )
        return Derivative({self: grad_local}) + vjp_tf

    # ------------------------------------------------------------------
    # Diagnostic helpers
    # ------------------------------------------------------------------

    def n_null_modes(self) -> int:
        """Number of null modes removed from L."""
        return self._n_dof_total - self._Q.shape[1]

    @classmethod
    def from_cylindrical_grid(
        cls,
        plasma_boundary,
        coils_TF,
        eval_points: np.ndarray,
        *,
        dr: float,
        dz: float,
        n_phi_slices: int,
        r_min: Optional[float] = None,
        r_max: Optional[float] = None,
        z_min: Optional[float] = None,
        z_max: Optional[float] = None,
        d_inner: float = 0.0,
        d_outer: float = 1.0,
        puck_R: Optional[Union[float, np.ndarray]] = None,
        puck_t: Optional[Union[float, np.ndarray]] = None,
        safety: float = 1.05,
        plasma_clearance: float = 0.0,
        nfp: Optional[int] = None,
        stellsym: Optional[bool] = None,
        m_fourier: int = 4,
        l_zernike: int = 6,
        k_chebyshev: int = 4,
        n_rho: int = 10,
        n_phi: int = 12,
        n_z: int = 6,
        regularization_delta: float = 1e-6,
        default_thickness: float = 0.02,
        strict_rim_continuity: bool = False,
        adaptive_self_reg: bool = True,
        exact_disc_faces: bool = False,
        n_radial_disc: int = 32,
    ) -> "PSCBulkArray":
        """Passive bulks on a finite-shape-aware ``(r, φ, z)`` lattice with radial axes.

        See :func:`~simsopt.field.puck_init.cylindrical_grid_pucks` for spacing rules.

        Args:
            adaptive_self_reg: Forwarded to :class:`PSCBulkArray` (default
                ``True``: physically correct coincident-cell regularization).
            exact_disc_faces: If ``True``, after construction enable the
                semi-analytic disc-face self-block via
                ``psc.exact_disc_faces = True`` and rebuild ``L``.  Uses
                Duffy-type 1D quadrature with ``n_radial_disc`` radial
                nodes and is required to reach the Smythe thin-disc
                benchmark.
            n_radial_disc: Radial node count for the exact disc-face
                assembler; ignored when ``exact_disc_faces`` is
                ``False``.

        See :class:`PSCBulkArray` for the meaning of ``strict_rim_continuity``.
        """
        nfp_i = int(nfp if nfp is not None else plasma_boundary.nfp)
        centers, axes, radii, thicknesses = cylindrical_grid_pucks(
            plasma_boundary,
            dr,
            dz,
            n_phi_slices,
            r_min=r_min,
            r_max=r_max,
            z_min=z_min,
            z_max=z_max,
            d_inner=d_inner,
            d_outer=d_outer,
            puck_R=puck_R,
            puck_t=puck_t,
            safety=safety,
            plasma_clearance=plasma_clearance,
            nfp=nfp_i,
        )
        psc = cls(
            centers,
            axes,
            radii,
            thicknesses,
            coils_TF,
            eval_points=np.asarray(eval_points, dtype=float, order="C"),
            m_fourier=m_fourier,
            l_zernike=l_zernike,
            k_chebyshev=k_chebyshev,
            n_rho=n_rho,
            n_phi=n_phi,
            n_z=n_z,
            nfp=nfp_i,
            stellsym=bool(stellsym)
            if stellsym is not None
            else bool(plasma_boundary.stellsym),
            regularization_delta=regularization_delta,
            default_thickness=default_thickness,
            strict_rim_continuity=strict_rim_continuity,
            adaptive_self_reg=bool(adaptive_self_reg),
        )
        if exact_disc_faces:
            psc.exact_disc_faces = True
            psc.n_radial_disc = int(n_radial_disc)
            psc._rebuild()
        return psc

    @classmethod
    def from_winding_surface(
        cls,
        plasma_boundary,
        coils_TF,
        eval_points: np.ndarray,
        *,
        distance: float,
        n_phi_pucks: int,
        n_theta_pucks: int,
        puck_R: Optional[Union[float, np.ndarray]] = None,
        puck_t: Optional[float] = None,
        m_fourier: int = 4,
        l_zernike: int = 6,
        k_chebyshev: int = 4,
        n_rho: int = 10,
        n_phi: int = 12,
        n_z: int = 6,
        nfp: Optional[int] = None,
        stellsym: Optional[bool] = None,
        regularization_delta: float = 1e-6,
        default_thickness: float = 0.02,
        strict_rim_continuity: bool = False,
        adaptive_self_reg: bool = True,
        exact_disc_faces: bool = False,
        n_radial_disc: int = 32,
    ) -> "PSCBulkArray":
        """Passive bulks on a winding surface ``extend_via_normal(distance)`` from the plasma.

        Args:
            adaptive_self_reg: Forwarded to :class:`PSCBulkArray` (default
                ``True``: physically correct coincident-cell regularization).
            exact_disc_faces: If ``True``, after construction enable the
                semi-analytic disc-face self-block via
                ``psc.exact_disc_faces = True`` and rebuild ``L``.  Uses
                Duffy-type 1D quadrature with ``n_radial_disc`` radial
                nodes and is required to reach the Smythe thin-disc
                benchmark.
            n_radial_disc: Radial node count for the exact disc-face
                assembler; ignored when ``exact_disc_faces`` is
                ``False``.

        See :class:`PSCBulkArray` for the meaning of ``strict_rim_continuity``.
        """
        nfp_i = int(nfp if nfp is not None else plasma_boundary.nfp)
        centers, axes, radii, thicknesses = winding_surface_pucks(
            plasma_boundary,
            distance=float(distance),
            n_phi_pucks=int(n_phi_pucks),
            n_theta_pucks=int(n_theta_pucks),
            puck_R=puck_R,
            puck_t=puck_t,
            default_thickness=default_thickness,
        )
        psc = cls(
            centers,
            axes,
            radii,
            thicknesses,
            coils_TF,
            eval_points=np.asarray(eval_points, dtype=float, order="C"),
            m_fourier=m_fourier,
            l_zernike=l_zernike,
            k_chebyshev=k_chebyshev,
            n_rho=n_rho,
            n_phi=n_phi,
            n_z=n_z,
            nfp=nfp_i,
            stellsym=bool(stellsym)
            if stellsym is not None
            else bool(plasma_boundary.stellsym),
            regularization_delta=regularization_delta,
            default_thickness=default_thickness,
            strict_rim_continuity=strict_rim_continuity,
            adaptive_self_reg=bool(adaptive_self_reg),
        )
        if exact_disc_faces:
            psc.exact_disc_faces = True
            psc.n_radial_disc = int(n_radial_disc)
            psc._rebuild()
        return psc


# ======================================================================
# PassiveBulkField
# ======================================================================


class PassiveBulkField(MagneticField):
    """Magnetic field from sheet currents on passive bulk pucks."""

    def __init__(self, psc_bulk: PSCBulkArray):
        self.psc_bulk = psc_bulk
        MagneticField.__init__(self, depends_on=[psc_bulk])

    def _B_impl(self, B):
        pts = self.get_points_cart_ref()
        B[:] = self.psc_bulk.B_at_points(np.asarray(pts))

    def _dB_by_dX_impl(self, dB):
        """Spatial Jacobian :math:`\\partial B_i / \\partial x_j` of the passive
        bulk field.

        Not yet implemented.  Previous behavior silently returned a zero
        Jacobian, which caused any downstream consumer (particle tracing,
        :class:`~simsopt.field.boozermagneticfield.BoozerMagneticField`
        construction, :math:`\\nabla B`-based objectives) to see wrong zeros
        with no warning.  A follow-up will implement the analytic Jacobian
        by differentiating the shell Biot-Savart kernel
        :math:`\\mathbf{K}(\\mathbf{r}') \\times (\\mathbf{x}-\\mathbf{r}') /
        |\\mathbf{x}-\\mathbf{r}'|^3` w.r.t. ``x``.
        """
        raise NotImplementedError(
            "PassiveBulkField.dB_by_dX is not implemented.  Previously this "
            "method silently returned zeros, which is incorrect for any "
            "consumer that consumes the spatial gradient (particle tracing, "
            "BoozerMagneticField construction, grad-B objectives).  File a "
            "feature request or implement the analytic shell Biot-Savart "
            "Jacobian."
        )

    def B_vjp(self, v):
        v = np.asarray(v).reshape(-1, 3)
        pts = np.asarray(self.get_points_cart_ref())
        return self.psc_bulk.vjp_setup_B(v, pts)

    def invalidate(self) -> None:
        """Clear all cached field values and propagate the invalidation.

        After calling :meth:`PSCBulkArray.recompute_currents` (or changing
        TF coil DOFs while the puck DOFs are fixed) the stored modal
        coefficients are fresh but any :class:`MagneticField` /
        :class:`MagneticFieldSum` that previously evaluated ``B`` at a
        set of points still holds stale cached values.  Call this to
        invalidate both this object's cache and any downstream consumer
        (via the :class:`~simsopt._core.optimizable.Optimizable` child
        graph) such as a :class:`MagneticFieldSum` that this field was
        added to.

        This is the canonical dirty-flagging entry point for the passive
        bulk field.  Previously tests used
        ``btot.clear_cached_properties()`` and
        ``btot.Bfields[0].invalidate_cache()`` interchangeably; both are
        still valid, but :meth:`invalidate` makes the intent explicit.
        """
        self.clear_cached_properties()
        for weakref_child in list(self._children):
            child = weakref_child()
            if child is None:
                continue
            clear = getattr(child, "clear_cached_properties", None)
            if callable(clear):
                clear()


def make_bulk_plus_tf_field(psc_bulk: PSCBulkArray, coils_tf):
    """Return ``PassiveBulkField(psc_bulk) + BiotSavart(coils_tf)``."""
    return psc_bulk.biot_savart + BiotSavart(coils_tf)
