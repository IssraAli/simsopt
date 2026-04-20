"""
Ideal-diamagnetic passive bulk (cylindrical pucks) and :class:`PassiveBulkField`.

**Environment (optional)**

- ``SIMSOPT_JAX_CACHE_DIR``: directory for the JAX experimental compilation
  cache (empty string disables).
- ``SIMSOPT_JAX_PRIME``: if ``1``, after the first successful
  :class:`PSCBulkArray` rebuild the process one-shot executes hot ``jax.jit``
  entry points to prime XLA (failures are ignored).

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

The TF-only gradient path has two implementations: a fast analytic adjoint
that calls into the C++ :class:`~simsopt.field.biot_savart.BiotSavart`
(always used in production), and a reference pure-JAX
:func:`_vjp_tf_run`/:func:`_B_eval_from_tf_body` path used by equivalence
tests to cross-check the analytic adjoint.  Tests toggle between the two
by writing the module-level ``_USE_JAX_TF_VJP`` attribute; there is no
user-facing switch.
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax import vjp
from scipy.linalg import solve_triangular

from simsopt._core.derivative import Derivative
from simsopt._core.optimizable import Optimizable

from .bulk_inductance import (
    MU0_OVER_4PI,
    _SELF_REG_COEFF,
    expand_beta_reduced,
    null_space_projection_matrix,
    shell_biot_savart_stacked_pure,
    shell_cholesky_pure,
    shell_eigenfloor_cholesky_pure,
    shell_inductance_matrix_blockwise,
    shell_inductance_matrix_symmetric_reduced,
    shell_loading_vector_stacked_pure,
    shell_normal_field_basis_matrix,
    shell_solve_eigenfloor_pure,
    shell_solve_prefactored_pure,
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

# Route TF-only VJP through the analytic adjoint + C++ BiotSavart (default
# production path).  Equivalence tests flip this module-level flag to
# ``True`` to exercise the slower pure-JAX reference path through
# :func:`_B_at_point_from_coil_set_pure`; see the two ``_USE_JAX_TF_VJP``
# call sites in :mod:`tests.field.test_passive_bulks`.  Deliberately kept
# as a plain module attribute (no env-var ingress) so end users cannot flip
# it and silently pay the ~3-5x runtime overhead.
_USE_JAX_TF_VJP = False

# Persistent XLA compilation cache (speeds cold starts; optional).
_JAX_CACHE_DIR = os.environ.get(
    "SIMSOPT_JAX_CACHE_DIR",
    os.path.join(os.path.expanduser("~"), ".cache", "simsopt", "jax"),
)
if _JAX_CACHE_DIR:
    try:
        from jax.experimental.compilation_cache import compilation_cache as _jax_cc

        os.makedirs(_JAX_CACHE_DIR, exist_ok=True)
        _jax_cc.initialize_cache(_JAX_CACHE_DIR)
    except Exception:
        pass

# Optional one-shot warm-up of module-level ``jax.jit`` kernels (cold XLA).
_PSC_JAX_PRIME_ONCE: bool = False


def _maybe_prime_psc_jax_kernels(psc: "PSCBulkArray") -> None:
    """Trace hot PSCBulk JIT bodies once when ``SIMSOPT_JAX_PRIME=1``.

    Uses the arrays from a real :class:`PSCBulkArray` after :meth:`_rebuild`
    so shapes match the user's problem.  Failures are ignored (read-only FS,
    missing optional deps).
    """
    global _PSC_JAX_PRIME_ONCE
    if os.environ.get("SIMSOPT_JAX_PRIME", "0") != "1":
        return
    if _PSC_JAX_PRIME_ONCE:
        return
    if psc._jax_phi_work_stack is None or psc._jax_K_stack is None:
        return
    try:
        g_tf, gd_tf, I_tf = psc._tf_arrays()
        g_tf = jnp.asarray(g_tf)
        gd_tf = jnp.asarray(gd_tf)
        I_tf = jnp.asarray(I_tf)
        nqtot = int(psc._jax_quad_pts.shape[0])
        Bn_z = jnp.zeros((nqtot,), dtype=jnp.float64)
        ep = jnp.asarray(psc.eval_points[:1])
        _ = _beta_from_tf_jitted(
            psc._jax_Lr_chol,
            psc._jax_Q,
            psc._jax_quad_pts,
            psc._jax_quad_n,
            psc._jax_phi_work_stack,
            psc._jax_w_work_stack,
            psc._jax_base_indices,
            psc._jax_replica_signs,
            g_tf,
            gd_tf,
            I_tf,
        )
        _ = _beta_from_bn_jitted(
            psc._jax_Lr_chol,
            psc._jax_Q,
            psc._jax_phi_work_stack,
            psc._jax_w_work_stack,
            psc._jax_base_indices,
            psc._jax_replica_signs,
            Bn_z,
        )
        _ = _beta_eigenfloor_from_bn_jitted(
            psc._jax_Lr_eigf_chol,
            psc._jax_Q,
            psc._jax_phi_work_stack,
            psc._jax_w_work_stack,
            psc._jax_base_indices,
            psc._jax_replica_signs,
            Bn_z,
        )
        _ = _B_eval_jitted(
            psc._jax_Lr_chol,
            psc._jax_Q,
            psc._jax_quad_pts,
            psc._jax_quad_n,
            psc._jax_phi_work_stack,
            psc._jax_w_work_stack,
            psc._jax_K_stack,
            psc._jax_w_q,
            psc._jax_base_indices,
            psc._jax_replica_signs,
            g_tf,
            gd_tf,
            I_tf,
            ep,
        )
        _ = _B_eval_from_bn_jitted(
            psc._jax_Lr_chol,
            psc._jax_Q,
            psc._jax_quad_pts,
            psc._jax_phi_work_stack,
            psc._jax_w_work_stack,
            psc._jax_K_stack,
            psc._jax_w_q,
            psc._jax_base_indices,
            psc._jax_replica_signs,
            Bn_z,
            ep,
        )
        _PSC_JAX_PRIME_ONCE = True
    except Exception:
        pass


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


# ----------------------------------------------------------------------
# Symmetry-aware body signatures
# ----------------------------------------------------------------------
#
# Every JIT body below operates on **work-sized** arrays
# ``(phi_work_stack, w_work_stack, L_work, Q_work)`` of shape
# ``(n_work, nq_per, ...)``, where ``n_work`` is the number of *independent*
# pucks we need to solve for:
#
# - Full path (no TF symmetry or no puck replication): ``n_work == n_all``
#   and ``base_indices == arange(n_all)``; folding and gathering become
#   no-ops and the code reduces to the old behavior.
# - Reduced path (TF + puck layout share an :math:`n_{fp}`/stellsym
#   group ``G``): ``n_work == n_base`` with ``base_indices[r] = base(r)``
#   mapping replica ``r`` to its base puck.  ``L_work`` is ``L_base`` as
#   built by :func:`shell_inductance_matrix_symmetric_reduced`; the
#   normal field ``Bn`` is computed on every replica and folded into
#   base-space via ``segment_sum``; the solved ``beta_base`` is gathered
#   back to every replica via a ``base_indices`` index lookup before the
#   full-replica Biot-Savart.
#
# This keeps a **single** JIT body per entry point instead of duplicating
# the full/reduced code paths, while still caching separately because the
# traced array shapes differ.


def _fold_Bn_to_work(
    Bn_flat: jnp.ndarray,
    base_indices: jnp.ndarray,
    signs: jnp.ndarray,
    n_work: int,
    nq_per: int,
) -> jnp.ndarray:
    """``Bn_work[i_base, q] = sum_{r in orbit(i_base)} sigma_r Bn_all[r, q]``.

    ``base_indices`` is an ``(n_all,)`` integer map from replica index to
    base index (``0..n_work-1``); ``signs`` is the matching ``(n_all,)``
    array of :math:`\\sigma_r \\in \\{+1, -1\\}`.  For the full path
    (``n_work == n_all``) ``base_indices`` is ``arange(n_all)`` and
    ``signs`` is all ``+1``, so this is the identity.

    The sign convention follows the induced-current parity under the
    symmetry group: pure rotations keep the current direction
    (``sigma = +1``) while simsopt's stellsym image flips the coil
    current (``sigma = -1``), producing ``Bn(Sx) = -Bn(x)`` at
    symmetry-related quadrature points.
    """
    n_all = base_indices.shape[0]
    Bn_stack = Bn_flat.reshape(n_all, nq_per)
    signed = signs.astype(Bn_stack.dtype)[:, None] * Bn_stack
    return jax.ops.segment_sum(signed, base_indices, num_segments=n_work)


def _gather_beta_work_to_all(
    beta_work: jnp.ndarray,
    base_indices: jnp.ndarray,
    signs: jnp.ndarray,
    nd_per: int,
) -> jnp.ndarray:
    """``beta_all[r, :] = sigma_r * beta_work[base_indices[r], :]`` flattened.

    Dual of :func:`_fold_Bn_to_work`: scatters ``beta_work`` (base DOFs)
    back to every replica with the matching per-replica sign so that the
    subsequent Biot-Savart sum over the full-replica ``K_all_stack`` sees
    the correct current pattern.  On the full path ``signs`` is all
    ``+1`` and this reduces to a plain gather.
    """
    n_work_dof = beta_work.shape[0]
    n_work = n_work_dof // nd_per
    beta_work_stack = beta_work.reshape(n_work, nd_per)
    beta_all_stack = beta_work_stack[base_indices]
    return (signs.astype(beta_all_stack.dtype)[:, None] * beta_all_stack).reshape(-1)


def _beta_from_tf_body(
    Lr_chol: jnp.ndarray,
    Qm: jnp.ndarray,
    quad_pts: jnp.ndarray,
    quad_n: jnp.ndarray,
    phi_work_stack: jnp.ndarray,
    w_work_stack: jnp.ndarray,
    base_indices: jnp.ndarray,
    signs: jnp.ndarray,
    g_tf: jnp.ndarray,
    gd_tf: jnp.ndarray,
    I_tf: jnp.ndarray,
) -> jnp.ndarray:
    """TF-only modal coefficients using Q-projected reduced solve (fast path).

    ``Lr_chol`` is the lower Cholesky factor of ``L_red + jitter I`` with
    the same ``jitter`` as :func:`~simsopt.field.bulk_inductance.shell_solve_linear_pure`,
    built once in :meth:`PSCBulkArray._rebuild`.

    Supports both the full (``n_work == n_all``) and the symmetry-reduced
    (``n_work == n_base``) paths via a single unified body; ``signs`` is
    all ``+1`` on the full path and carries the stellsym parity on the
    reduced path (see :func:`_fold_Bn_to_work`).
    """
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

    Bn_all = jax.vmap(Bn_at_i)(jnp.arange(quad_pts.shape[0]))
    n_work, nq_per, _ = phi_work_stack.shape
    Bn_work_stack = _fold_Bn_to_work(Bn_all, base_indices, signs, n_work, nq_per)
    f = shell_loading_vector_stacked_pure(
        phi_work_stack,
        w_work_stack,
        Bn_work_stack.reshape(-1),
    )
    fr = Qm.T @ f
    alpha = shell_solve_prefactored_pure(Lr_chol, fr)
    return expand_beta_reduced(alpha, Qm)


_beta_from_tf_jitted = jax.jit(_beta_from_tf_body)


def _beta_from_bn_body(
    Lr_chol: jnp.ndarray,
    Qm: jnp.ndarray,
    phi_work_stack: jnp.ndarray,
    w_work_stack: jnp.ndarray,
    base_indices: jnp.ndarray,
    signs: jnp.ndarray,
    Bn_all: jnp.ndarray,
) -> jnp.ndarray:
    """Reduced solve given ``Bn`` at every **all-replica** quad point.

    ``Lr_chol`` is the Cholesky factor of ``L_red + jitter I`` (see
    :func:`_beta_from_tf_body`).  When the
    reduced path is active, ``Bn_all`` is folded into base-space with the
    per-replica ``signs`` before forming the loading vector; for the full
    path, ``base_indices`` is the identity and ``signs`` is all ``+1`` so
    the fold is a no-op.
    """
    n_work, nq_per, _ = phi_work_stack.shape
    Bn_work_stack = _fold_Bn_to_work(Bn_all, base_indices, signs, n_work, nq_per)
    f = shell_loading_vector_stacked_pure(
        phi_work_stack,
        w_work_stack,
        Bn_work_stack.reshape(-1),
    )
    fr = Qm.T @ f
    alpha = shell_solve_prefactored_pure(Lr_chol, fr)
    return expand_beta_reduced(alpha, Qm)


_beta_from_bn_jitted = jax.jit(_beta_from_bn_body)


def _beta_eigenfloor_from_bn_body(
    Lr_eigf_chol: jnp.ndarray,
    Q: jnp.ndarray,
    phi_work_stack: jnp.ndarray,
    w_work_stack: jnp.ndarray,
    base_indices: jnp.ndarray,
    signs: jnp.ndarray,
    Bn_all: jnp.ndarray,
) -> jnp.ndarray:
    """Rim-continuity-projected eigenfloor solve given ``Bn``.

    ``Lr_eigf_chol`` is the Cholesky factor of the eigenvalue-floored matrix
    used in :func:`~simsopt.field.bulk_inductance.shell_solve_eigenfloor_pure`.
    Here ``Q = Q_c Q_L`` is the full null-space-trimmed projector already
    composed at rebuild time.  This avoids materialising ``L_work`` on
    device.  Supports both the full and reduced paths via signed
    ``base_indices`` folding; see :func:`_fold_Bn_to_work`.
    """
    n_work, nq_per, _ = phi_work_stack.shape
    Bn_work_stack = _fold_Bn_to_work(Bn_all, base_indices, signs, n_work, nq_per)
    f = shell_loading_vector_stacked_pure(
        phi_work_stack,
        w_work_stack,
        Bn_work_stack.reshape(-1),
    )
    f_r = Q.T @ f
    alpha = shell_solve_prefactored_pure(Lr_eigf_chol, f_r)
    return Q @ alpha


_beta_eigenfloor_from_bn_jitted = jax.jit(_beta_eigenfloor_from_bn_body)


def _B_eval_from_tf_body(
    Lr_chol: jnp.ndarray,
    Qm: jnp.ndarray,
    quad_pts: jnp.ndarray,
    quad_n: jnp.ndarray,
    phi_work_stack: jnp.ndarray,
    w_work_stack: jnp.ndarray,
    K_all_stack: jnp.ndarray,
    w_quad_all: jnp.ndarray,
    base_indices: jnp.ndarray,
    signs: jnp.ndarray,
    g_tf: jnp.ndarray,
    gd_tf: jnp.ndarray,
    I_tf: jnp.ndarray,
    pts_eval: jnp.ndarray,
) -> jnp.ndarray:
    """Passive bulk B at eval points; TF-only VJP fast path.

    Solves in work (base) DOFs using the pre-projected reduced matrix
    ``L_red``, then gathers ``beta_base`` back to every replica (with the
    per-replica sign) for the full-replica Biot-Savart summation using
    ``K_all_stack``.
    """
    b_work = _beta_from_tf_body(
        Lr_chol,
        Qm,
        quad_pts,
        quad_n,
        phi_work_stack,
        w_work_stack,
        base_indices,
        signs,
        g_tf,
        gd_tf,
        I_tf,
    )
    nd_per = K_all_stack.shape[2]
    b_all = _gather_beta_work_to_all(b_work, base_indices, signs, nd_per)
    return shell_biot_savart_stacked_pure(
        K_all_stack,
        quad_pts,
        w_quad_all,
        b_all,
        jnp.asarray(pts_eval),
        eps=_EPS_BS,
    )


_B_eval_jitted = jax.jit(_B_eval_from_tf_body)


def _B_eval_from_bn_body(
    Lr_chol: jnp.ndarray,
    Qm: jnp.ndarray,
    quad_pts: jnp.ndarray,
    phi_work_stack: jnp.ndarray,
    w_work_stack: jnp.ndarray,
    K_all_stack: jnp.ndarray,
    w_quad_all: jnp.ndarray,
    base_indices: jnp.ndarray,
    signs: jnp.ndarray,
    Bn_all: jnp.ndarray,
    pts_eval: jnp.ndarray,
) -> jnp.ndarray:
    """Passive bulk B at eval points given ``Bn`` (avoids JAX Biot-Savart on TF)."""
    b_work = _beta_from_bn_body(
        Lr_chol,
        Qm,
        phi_work_stack,
        w_work_stack,
        base_indices,
        signs,
        Bn_all,
    )
    nd_per = K_all_stack.shape[2]
    b_all = _gather_beta_work_to_all(b_work, base_indices, signs, nd_per)
    return shell_biot_savart_stacked_pure(
        K_all_stack,
        quad_pts,
        w_quad_all,
        b_all,
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
    full_L_band_size: int = 32,
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
            This must be ``True`` whenever the precomputed ``self._L_work``
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

    n_pucks_int = int(local_K_stack.shape[0])
    nb = int(full_L_band_size)
    if nb <= 0 or nb >= n_pucks_int:
        pair_i, pair_j = jnp.meshgrid(
            jnp.arange(n_pucks),
            jnp.arange(n_pucks),
            indexing="ij",
        )
        pairs = jnp.stack([pair_i.ravel(), pair_j.ravel()], axis=1)
        L_blocks_flat = jax.lax.map(L_block, pairs)
        L_blocks = L_blocks_flat.reshape(n_pucks, n_pucks, nd, nd)
        L = L_blocks.transpose(0, 2, 1, 3).reshape(n_dof_total, n_dof_total)
    else:
        L = jnp.zeros((n_dof_total, n_dof_total), dtype=pts_stack.dtype)
        for start in range(0, n_pucks_int, nb):
            end = min(start + nb, n_pucks_int)

            def row_block(i):
                def col_block(j):
                    return L_block(jnp.stack([jnp.asarray(i), jnp.asarray(j)]))

                return jax.lax.map(col_block, jnp.arange(n_pucks))

            band_blocks = jax.lax.map(row_block, jnp.arange(start, end))
            band_flat = band_blocks.transpose(0, 2, 1, 3).reshape(
                (end - start) * nd, n_dof_total
            )
            L = L.at[start * nd : end * nd, :].set(band_flat)

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
    # Sign convention: shell_loading_vector_pure returns
    # ``f_a = -integral Phi_a B_n^TF dS`` (see docstring).  The JIT
    # body used to drop the leading minus (bug: produced the wrong
    # induced-moment sign on free-puck-DOF B_at_points Taylor tests);
    # put it back here to match the fixed-DOF path.
    f = -jnp.sum(
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
    static_argnames=(
        "delta_reg",
        "eigenfloor_threshold",
        "adaptive_self_reg",
        "full_L_band_size",
    ),
)


def _vjp_tf_run(
    Lr_chol: jnp.ndarray,
    Qm: jnp.ndarray,
    quad_pts: jnp.ndarray,
    quad_n: jnp.ndarray,
    phi_work_stack: jnp.ndarray,
    w_work_stack: jnp.ndarray,
    K_all_stack: jnp.ndarray,
    w_quad_all: jnp.ndarray,
    base_indices: jnp.ndarray,
    signs: jnp.ndarray,
    g_tf: jnp.ndarray,
    gd_tf: jnp.ndarray,
    I_tf: jnp.ndarray,
    pts_eval: jnp.ndarray,
    v_B: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """TF-only VJP; uses non-JIT forward body so AD is exact inside ``jax.jit``.

    Takes the Cholesky factor of ``L_red + jitter I`` so the big
    ``L_work`` need not be resident on device.  ``signs`` is the
    per-replica parity passed through the signed fold/gather operators.
    """

    def fwd(g, gd, I):
        return _B_eval_from_tf_body(
            Lr_chol,
            Qm,
            quad_pts,
            quad_n,
            phi_work_stack,
            w_work_stack,
            K_all_stack,
            w_quad_all,
            base_indices,
            signs,
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
    full_L_band_size: int = 32,
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
            full_L_band_size,
        )

    _, vjp_fn = vjp(fwd, centers_all, quats_all, g_tf, gd_tf, I_tf)
    return vjp_fn(v_B)


_vjp_full_jitted = jax.jit(
    _vjp_full_run,
    static_argnames=(
        "delta_reg",
        "eigenfloor_threshold",
        "adaptive_self_reg",
        "full_L_band_size",
    ),
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
        solver_mode: Which discrete weak form to use to determine the
            modal coefficients :math:`\\beta`.

            * ``"energy"`` (default): the current-potential /
              mutual-inductance Gram, ``L beta = f`` with
              ``L_{ab} = (mu_0/4pi) integral K_a . K_b / |r-r'| dS dS'``
              and ``f_a = -integral Phi_a B_n^{TF} dS``.  Minimises
              ``0.5 beta^T L beta + beta^T f`` on the rim-continuity /
              gauge-null subspace.  Recovers the induced dipole
              correctly (Lenz + sphere-scale limits) and is the path
              used for every production optimisation because it supports
              the JIT TF-only VJP and the full-JAX free-DOF VJP.
            * ``"shell_l2"``: REGCOIL-style :math:`L^2`-residual form
              ``M beta = g`` with
              ``M_{ab} = integral B_n^{(a)} B_n^{(b)} dS`` and
              ``g_a = -integral B_n^{(a)} B_n^{TF} dS``, where
              :math:`B_n^{(a)}` is the normal component on the shell of
              the field from basis function ``a``.  Drives
              :math:`\\|B_n^{tot}\\|_{L^2(\\Sigma)}` monotonically to
              zero as the basis is refined and hence drives the ideal-
              diamagnet interior-field cancellation
              :math:`|B^{tot}(\\mathbf x_{\\rm int})|\\to 0`, which the
              ``"energy"`` form only enforces in the Galerkin sense.
              Opt-in because the assembly is dense in puck-pair blocks
              (cost scales like the off-diagonal of ``L`` times ``n_q``)
              and because the JIT TF-VJP and free-DOF-VJP paths are not
              yet implemented on this branch; ``B_at_points`` falls
              back to a pure-NumPy Biot-Savart of the cached
              :math:`\\beta`.  The reduced symmetry path is also
              disabled on ``"shell_l2"`` for the same reason.

        full_L_band_size (constructor): Row-band size for streaming assembly
            of the monolithic ``L`` inside :func:`_B_eval_full_body` when free
            puck DOFs are active; ``<= 0`` or ``>= n_pucks`` uses the dense
            path.  Default ``32``.

        use_f32_bs (constructor): If ``True``, the ``shell_l2`` Biot-Savart
            branch in :meth:`B_at_points` accumulates in ``float32`` inside
            :func:`~simsopt.field.bulk_inductance.shell_biot_savart_stacked_pure`.

        release_host_L_work_after_rebuild (constructor): If ``True``, after
            rebuild drop the host copy of the *full-replica* ``L_work`` when
            ``solver_mode == 'energy'`` and no puck geometry DOF is free,
            to reduce peak RSS on large asymmetric layouts.  Symmetry-reduced
            ``L_work`` (small folded matrix) is always retained.  Default
            ``False`` so :attr:`_L_work` remains available for tests and
            Galerkin identities.
    """

    # Class-level latch so the ``_L_full`` deprecation warning fires at most
    # once per process.  Public attribute name starts with an underscore so
    # it is not considered a DOF by the :class:`Optimizable` discovery.
    _L_full_deprecation_warned: bool = False

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
        solver_mode: str = "energy",
        full_L_band_size: int = 32,
        use_f32_bs: bool = False,
        release_host_L_work_after_rebuild: bool = False,
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
        if solver_mode not in ("energy", "shell_l2"):
            raise ValueError(
                f"solver_mode must be 'energy' or 'shell_l2'; got {solver_mode!r}"
            )
        self.solver_mode = str(solver_mode)
        self._full_L_band_size = int(full_L_band_size)
        self._use_f32_bs = bool(use_f32_bs)
        self._release_host_L_work_after_rebuild = bool(
            release_host_L_work_after_rebuild
        )
        self._H_full = None
        self._Hw = None

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
        # Detect whether the TF coil set is a faithful image of its first
        # ``n_base`` entries under the ``(nfp, stellsym)`` group used to
        # replicate pucks.  When ``True``, ``_rebuild`` routes through the
        # symmetry-reduced inductance + solve path (``|G|`` reduction in
        # assembly work and ``|G|^2`` reduction in peak L storage);
        # otherwise the full ``n_dof_total``-sized system is assembled.
        self._tf_is_symmetric = self._detect_tf_symmetry(
            self.coils_TF,
            self.nfp,
            self.stellsym,
        )
        self._rebuild()
        _maybe_prime_psc_jax_kernels(self)
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

        The per-replica ``sign`` in the returned ``replica_signs`` encodes
        the induced-current parity under the symmetry group: ``+1`` for
        pure :math:`R_z(\\alpha_k)` rotations, ``-1`` for stellsym images
        (which simsopt models as a proper :math:`R_x(\\pi)` rotation
        combined with a coil current sign flip, producing
        ``K_image(x) = -M K_base(M^{-1} x)`` and hence
        ``beta^image = -beta^base`` in the local Fourier-Zernike /
        Fourier-Chebyshev basis; see the module docstring of
        :func:`_fold_Bn_to_work` and Phase 2 of the signed-orbit plan).

        Returns:
            all_pucks: list of ``(center, axis, R, t)``
            all_quats: list of quaternions per puck
            base_indices: which base puck each copy came from
            center_jacobians: 3x3 transform from base center to copy center
            quat_jacobians: 4x4 transform from base quaternion to copy quaternion
            replica_signs: ``int8`` array of length ``n_all`` with
                entries in ``{+1, -1}``; ``+1`` for pure-rotation replicas
                and ``-1`` for stellsym-image replicas.
        """
        all_pucks: List[Tuple[np.ndarray, np.ndarray, float, float]] = []
        all_quats_list: List[np.ndarray] = []
        base_indices: List[int] = []
        center_jacobians: List[np.ndarray] = []
        quat_jacobians: List[np.ndarray] = []
        replica_signs: List[int] = []

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
                replica_signs.append(+1)

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
                    replica_signs.append(-1)

        return (
            all_pucks,
            all_quats_list,
            base_indices,
            center_jacobians,
            quat_jacobians,
            np.asarray(replica_signs, dtype=np.int8),
        )

    @staticmethod
    def _detect_tf_symmetry(
        coils_TF,
        nfp: int,
        stellsym: bool,
        atol: float = 1e-8,
    ) -> bool:
        """Return ``True`` iff ``coils_TF`` is the image of its first
        ``n_total / |G|`` entries under the ``(nfp, stellsym)`` group.

        The comparison uses the canonical simsopt ordering produced by
        :func:`simsopt.field.coil.coils_via_symmetries`, whose helper
        :func:`apply_symmetries_to_curves` iterates
        ``for k in range(nfp): for flip in flip_list: for i in range(n_base): ...``
        (field-period outer, stellsym-flip middle, base-coil inner).  The
        previous implementation iterated in the order used by
        :meth:`_replicate_pucks` (``i`` outer, ``k`` middle, ``flip``
        inner) which only coincides with the canonical simsopt ordering
        when ``n_base == 1``; any multi-coil TF set produced by
        :func:`coils_via_symmetries` (for example the
        ``SchuettHennebergQAnfp2`` layout used in
        ``passive_bulks_cylindrical_grid_optimization.py``) silently
        failed detection and dropped to the expensive full-``L`` path.

        Detection is by direct geometric comparison of
        ``(gamma, gammadash, current)`` between each replica in
        ``coils_TF`` and the deterministically generated image; any
        mismatch (including coil count not divisible by ``|G|``) forces a
        conservative ``False`` so the full-``L`` fallback is used.

        Side-effect free: evaluates ``gamma()`` / ``gammadash()`` /
        ``current.get_value()`` only once per input coil and does not
        store any references beyond this check.
        """
        G = int(nfp) * (2 if stellsym else 1)
        n_total = len(coils_TF)
        if n_total == 0 or G <= 1 or n_total % G != 0:
            return False
        n_base = n_total // G
        # Pull base-coil geometry/currents once; keeping this side-effect
        # free avoids re-entering :meth:`coils_via_symmetries` just to
        # perform a detection pass.
        try:
            base_gammas = [np.asarray(coils_TF[i].curve.gamma()) for i in range(n_base)]
            base_gammadashs = [
                np.asarray(coils_TF[i].curve.gammadash()) for i in range(n_base)
            ]
            base_currents = [
                float(coils_TF[i].current.get_value()) for i in range(n_base)
            ]
        except Exception:
            return False

        # Simsopt-canonical loop ordering (``coils_via_symmetries``):
        #   k outer (field periods), flip middle (stellsym image), i inner
        # (base-coil index).
        idx = 0
        for k in range(int(nfp)):
            angle = 2.0 * np.pi * k / int(nfp)
            rot3 = np.array(
                [
                    [np.cos(angle), -np.sin(angle), 0.0],
                    [np.sin(angle), np.cos(angle), 0.0],
                    [0.0, 0.0, 1.0],
                ]
            )
            flip_list = (False, True) if stellsym else (False,)
            for flip in flip_list:
                if flip:
                    S = np.diag([1.0, -1.0, -1.0])
                    I_sign = -1.0
                else:
                    S = np.eye(3)
                    I_sign = 1.0
                M = S @ rot3
                for i_base in range(n_base):
                    g_pred = (M @ base_gammas[i_base].T).T
                    gd_pred = (M @ base_gammadashs[i_base].T).T
                    I_pred = I_sign * base_currents[i_base]
                    if idx >= n_total:
                        return False
                    try:
                        g_obs = np.asarray(coils_TF[idx].curve.gamma())
                        gd_obs = np.asarray(coils_TF[idx].curve.gammadash())
                        I_obs = float(coils_TF[idx].current.get_value())
                    except Exception:
                        return False
                    # Tight pointwise comparison: the simsopt
                    # ``RotatedCurve`` replicas produced by
                    # :func:`coils_via_symmetries` match without any
                    # cyclic parameter shift.
                    if (
                        g_obs.shape != g_pred.shape
                        or gd_obs.shape != gd_pred.shape
                        or not np.allclose(g_obs, g_pred, atol=atol)
                        or not np.allclose(gd_obs, gd_pred, atol=atol)
                        or not np.isclose(I_obs, I_pred, atol=atol)
                    ):
                        return False
                    idx += 1
        return idx == n_total

    # ------------------------------------------------------------------
    # Build / rebuild
    # ------------------------------------------------------------------

    def _build_rim_continuity_projector(
        self,
        n_dof_total: int,
        puck_subset: "list | None" = None,
    ) -> np.ndarray:
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
            puck_subset: Optional explicit list of replica indices to use
                when assembling blocks.  Defaults to every replica
                (``range(len(self._basis_per_puck))``).  When the
                symmetry-reduced path is active this is set to the base
                replicas only, producing a much smaller ``Q_c_base`` with
                one block per *base* puck (shape
                ``(n_base * nd_per, n_free_base)``).

        Returns:
            ``Q_c`` of shape ``(n_dof_total, n_free)`` with orthonormal
            columns (``Q_c.T @ Q_c = I``).
        """
        import warnings

        if puck_subset is None:
            puck_subset = list(range(len(self._basis_per_puck)))
        subset_dof_offsets: List[int] = []
        off = 0
        for p in puck_subset:
            subset_dof_offsets.append(off)
            off += self._basis_per_puck[p].k_basis_local.shape[1]

        Q_c_blocks: List[np.ndarray] = []
        row_offsets: List[int] = []
        col_offsets: List[int] = []
        n_free_total = 0
        degenerate = False
        for k, pidx in enumerate(puck_subset):
            basis = self._basis_per_puck[pidx]
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
            row_offsets.append(subset_dof_offsets[k])
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
        for k, Q_p in enumerate(Q_c_blocks):
            r0 = row_offsets[k]
            r1 = r0 + Q_p.shape[0]
            c0 = col_offsets[k]
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

        (
            all_pucks,
            all_quats_list,
            base_indices,
            center_jacs,
            quat_jacs,
            replica_signs,
        ) = self._replicate_pucks(
            centers,
            quats,
            radii,
            thicknesses,
        )

        self._all_pucks = all_pucks
        self._all_pucks_quats = np.array(all_quats_list)
        self._all_puck_base_indices = base_indices
        self._center_jacobians = center_jacs
        self._quat_jacobians = quat_jacs
        self._replica_signs = np.asarray(replica_signs, dtype=np.int8)

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

        # Flat/global quadrature geometry (shared across all code paths).
        quad_points = np.vstack(pts_per_puck_global)
        quad_weights = np.concatenate(weights_per_puck)
        quad_normals = np.vstack(normals_per_puck_global)

        # Per-puck row bookkeeping used for diagnostics and reshape paths.
        row0 = 0
        self._quad_row_ranges: List[Tuple[int, int]] = []
        for pidx in range(len(all_pucks)):
            nq = K_per_puck_global[pidx].shape[0]
            self._quad_row_ranges.append((row0, row0 + nq))
            row0 += nq

        self._quad_points = quad_points
        self._quad_weights = quad_weights
        self._quad_normals = quad_normals
        # Pre-shaped for :meth:`_vjp_tf_only_analytic` (avoid per-call ``reshape``).
        n_all_quads = len(all_pucks)
        self._quad_weights_2d = quad_weights.reshape(n_all_quads, -1)

        # Stacked block-diagonal representation of (K_basis, phi_mat).  In the
        # common case (all pucks share the same basis resolution -- which is
        # always true for PSCBulkArray because m_fourier/l_zernike/...
        # are globals on the array) every per-puck block has the same
        # ``(nq_per, nd_per, 3)`` shape and we simply ``np.stack`` them.  The
        # much larger dense ``(nq_total, n_dof_total, ...)`` arrays are never
        # materialised (they are ~O(n_pucks) * bigger because they are
        # block-diagonal), saving ~95-99% memory on the basis tables at scale.
        # Heterogeneous-shape fallback: if any per-puck block has a different
        # shape, fall back to the dense layout for backward compatibility.
        nq_pers = {arr.shape[0] for arr in K_per_puck_global}
        nd_pers = {arr.shape[1] for arr in K_per_puck_global}
        if len(nq_pers) == 1 and len(nd_pers) == 1:
            self._jax_K_stack = jnp.asarray(np.stack(K_per_puck_global, axis=0))
            self._jax_phi_stack = jnp.asarray(np.stack(phi_per_puck, axis=0))
            self._jax_w_stack = jnp.asarray(np.stack(weights_per_puck, axis=0))
            self._uniform_puck_shape = True
        else:
            # Rare path (mixed shapes): keep dense as padded fallback.  This
            # code path is currently exercised only via synthetic tests.
            self._jax_K_stack = None
            self._jax_phi_stack = None
            self._jax_w_stack = None
            self._uniform_puck_shape = False
        # Dense monolithic arrays are available lazily via the ``_K_basis``
        # and ``_phi_mat`` properties for legacy callers (VTK export,
        # regression tests).  They are not cached here to keep the memory
        # footprint down; each access rebuilds them on demand.

        # --------------------------------------------------------------
        # Inductance-matrix assembly.
        #
        # Two paths share the same downstream solver code:
        # * ``_reduced_active`` path: TF coils exhibit the same
        #   ``(nfp, stellsym)`` symmetry as the puck layout, so we only
        #   build the ``(n_base * nd_per, n_base * nd_per)`` folded matrix
        #   ``L_base`` and the corresponding base-puck rim-continuity
        #   projector; the full ``n_dof_total``-sized ``L`` is never
        #   allocated.
        # * Full path: asymmetric TF or no replication.  Build the dense
        #   ``(n_dof_total, n_dof_total)`` matrix and the full-size
        #   projector, as before.
        # --------------------------------------------------------------
        exact_disc_faces = bool(getattr(self, "exact_disc_faces", False))
        n_all = len(all_pucks)
        G = int(self.nfp) * (2 if self.stellsym else 1)
        # Reduced-path is only useful when there are actual replicas.
        #
        # Stellsym is gated off here intentionally.  Under simsopt's
        # ``coils_via_symmetries`` convention for stellarator symmetry,
        # the image coil carries ``I_sign = -1`` combined with a proper
        # rotation ``R_x(pi)`` of the local frame and position reflection
        # ``S = diag(1, -1, -1)``, which makes the normal component of
        # ``B_TF`` antisymmetric at G-related quadrature points:
        # ``Bn(S x) = -Bn(x)``.  The current unsigned duplication
        # operator ``T`` used by :func:`_fold_Bn_to_work` / :func:`_gather_beta_work_to_all`
        # and :func:`~simsopt.field.bulk_inductance.shell_inductance_matrix_symmetric_reduced`
        # then sums orbits to (near) zero and would silently zero out
        # Signed orbit operator (``sigma_r in {+1, -1}``) is now plumbed
        # through the fold/gather/JIT paths and
        # :func:`shell_inductance_matrix_symmetric_reduced`, so the
        # reduced path is safe to enable under ``stellsym=True`` as well.
        reduced_active = bool(
            self._tf_is_symmetric
            and G > 1
            and n_all == self._n_base_pucks * G
            and not exact_disc_faces  # see note below
            and self.solver_mode == "energy"  # shell_l2 assembles H densely across pucks
        )
        # Note: ``exact_disc_faces`` swaps intra-puck self-blocks with a
        # semi-analytic assembly that currently operates puck-by-puck on
        # the full replica list; the reduced path is disabled in that case
        # to keep self-block consistency trivial.  This is a narrow and
        # easy-to-lift restriction if/when the semi-analytic path grows a
        # symmetry-aware entry point.
        self._reduced_active = reduced_active

        # Always build the full-size rim-continuity projector: it is used
        # (as ``_Q_c``) by the puck-DOF full-JAX path, which JIT-assembles
        # L internally on every replica and therefore expects a full-size
        # projector.  The full-size ``Q_c`` is cheap to store (block
        # diagonal across pucks, ~``n_pucks * nd_per^2`` bytes).
        Q_c_full = self._build_rim_continuity_projector(n_dof_total)
        self._Q_c = Q_c_full

        # Base-replica indices (first replica of each base puck).  The
        # signed-orbit reduction assumes ``sigma_{i_rep} = +1`` so that
        # ``L_base[i, j] = G * sum_{s in orbit(j)} sigma_s L[i_rep(i), s]``
        # (see :func:`shell_inductance_matrix_symmetric_reduced`).  By
        # construction :meth:`_replicate_pucks` emits every pure-rotation
        # replica *before* its stellsym image, so the first encountered
        # replica of each base puck always carries ``sign = +1``; assert
        # to guard against future reordering.
        seen = [False] * self._n_base_pucks
        base_reps: List[int] = [0] * self._n_base_pucks
        for replica_idx, bi in enumerate(base_indices):
            if not seen[bi]:
                base_reps[bi] = replica_idx
                seen[bi] = True
        self._base_reps = base_reps
        if n_all > 0 and not np.all(self._replica_signs[base_reps] == +1):
            bad = [int(r) for r in base_reps if self._replica_signs[r] != +1]
            raise RuntimeError(
                "Signed-orbit reduction requires every base_reps entry to "
                f"be a pure-rotation replica (sign = +1); got stellsym-image "
                f"replicas at base_reps indices {bad}.  This indicates "
                "_replicate_pucks reordered the emission sequence."
            )

        if reduced_active:
            L_base = shell_inductance_matrix_symmetric_reduced(
                K_per_puck_global,
                pts_per_puck_global,
                weights_per_puck,
                base_indices,
                delta_reg=self.regularization_delta,
                adaptive_self_reg=self.adaptive_self_reg,
                replica_signs=self._replica_signs,
            )
            self._L_work = L_base
            nd_per_puck = K_per_puck_global[base_reps[0]].shape[1]
            Q_c_work = self._build_rim_continuity_projector(
                self._n_base_pucks * nd_per_puck,
                puck_subset=base_reps,
            )
        else:
            if exact_disc_faces:
                disc_centers_axes = []
                disc_Rts = []
                for idx, (c, ax, R_val, t_val) in enumerate(all_pucks):
                    Rmat = _rotation_matrix_from_quat(all_quats_list[idx])
                    axis_global = Rmat @ np.array([0.0, 0.0, 1.0])
                    disc_centers_axes.append(
                        (
                            np.asarray(c, dtype=float),
                            np.asarray(axis_global, dtype=float),
                        )
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
            self._L_work = L_np
            Q_c_work = Q_c_full  # alias -- full path uses the same projector

            # ---- shell_l2 (REGCOIL-style) operator swap --------------
            # Replace the current-potential Gram ``L`` and the
            # corresponding load vector with the L^2-on-shell normal-
            # field operators ``M = H^T diag(w) H`` and
            # ``g = -H^T diag(w) Bn^TF``.  Dense assembly across every
            # puck-pair; the reduced path is already gated off above in
            # this branch.
            if self.solver_mode == "shell_l2":
                K_basis_dense = self._K_basis  # lazy (nq_total, n_dof, 3)
                H = shell_normal_field_basis_matrix(
                    K_basis_dense,
                    self._quad_points,
                    self._quad_weights,
                    self._quad_normals,
                    delta_reg=self.regularization_delta,
                    adaptive_self_reg=self.adaptive_self_reg,
                )
                Hw = H.T * self._quad_weights  # (n_dof, n_quad)
                M = Hw @ H
                M = 0.5 * (M + M.T)
                self._H_full = H
                self._Hw = Hw
                self._L_work = M

        self._Q_c_work = Q_c_work

        # Gauge projection on top of the continuity-restricted subspace:
        # drop the remaining exact null modes (constant per puck) of L.
        L_c = Q_c_work.T @ self._L_work @ Q_c_work
        Q_L = null_space_projection_matrix(
            L_c,
            threshold=self._null_space_threshold,
        )
        self._Q = Q_c_work @ Q_L
        Lr = self._Q.T @ self._L_work @ self._Q
        self._L_red = Lr
        self._jax_Lr_chol = shell_cholesky_pure(jnp.asarray(Lr), jitter=1e-10)
        self._Lr_chol_host = np.asarray(self._jax_Lr_chol)
        self._jax_Lr_eigf_chol = shell_eigenfloor_cholesky_pure(
            jnp.asarray(Lr),
            threshold=float(self._eigenfloor_threshold),
            jitter=1e-10,
        )

        # Stacked work arrays: on the reduced path these are picked from the
        # base replicas only (same local basis, so they are exactly what the
        # loading vector needs after Bn folding); on the full path they equal
        # ``_phi_stack`` / ``_w_stack``.
        if self._jax_phi_stack is not None:
            if reduced_active:
                self._jax_phi_work_stack = self._jax_phi_stack[
                    np.asarray(self._base_reps)
                ]
                self._jax_w_work_stack = self._jax_w_stack[
                    np.asarray(self._base_reps)
                ]
            else:
                self._jax_phi_work_stack = self._jax_phi_stack
                self._jax_w_work_stack = self._jax_w_stack
        else:
            self._jax_phi_work_stack = None
            self._jax_w_work_stack = None

        # Replica->base index map used by the JIT bodies to fold ``Bn``
        # and gather ``beta_work`` back to every replica.  On the full
        # path this must be the identity (``arange(n_all)``) so that
        # :func:`_fold_Bn_to_work` and :func:`_gather_beta_work_to_all`
        # reduce to no-ops; if we left the orbit-grouping here the
        # JAX segment_sum would (incorrectly) collapse ``Bn`` across
        # orbits even though the full matrix ``L`` is in use.
        if reduced_active:
            self._base_indices_arr = np.asarray(base_indices, dtype=np.int32)
        else:
            self._base_indices_arr = np.arange(n_all, dtype=np.int32)

        # Signs fed to the signed fold / gather and the analytic VJP.
        # They coincide with ``_replica_signs`` only on the reduced
        # path; on the full-``L`` fallback the fold/gather are no-ops,
        # so effective signs must be all ``+1`` to avoid silently
        # sign-flipping stellsym-image replicas' ``Bn`` / ``beta``.
        # ``_replica_signs`` itself retains its raw +1/-1 pattern so
        # structural tests and the signed inductance assembly can read
        # it unchanged.
        if reduced_active:
            self._signs_effective = np.asarray(self._replica_signs, dtype=np.int8)
        else:
            self._signs_effective = np.ones(n_all, dtype=np.int8)

        # JAX arrays for module-level JIT (no new ``jax.jit`` closures per rebuild)
        self._setup_jax()

        # Optional: drop the host copy of the *full-replica* ``L_work`` after
        # factorization (large (BG·D)^2); off by default so diagnostics/tests
        # can still read ``_L_work``.  Symmetry-reduced ``L_work`` is small and
        # is never dropped here.
        if (
            self._release_host_L_work_after_rebuild
            and not self._has_free_puck_dofs()
            and self.solver_mode == "energy"
            and not self._reduced_active
        ):
            self._L_work = None

        # Solve
        self.beta = self._solve_beta(self._tf_arrays())
        self._geom_hash = hash(tuple(self.local_full_x))

    # ------------------------------------------------------------------
    # JAX fast path (TF-only VJP, pre-computed L and Q)
    # ------------------------------------------------------------------

    def _tf_dofs_hash(self) -> int:
        """Hash TF curve and current DOF bytes for :meth:`_tf_arrays` caching."""
        parts = []
        for c in self.coils_TF:
            parts.append(c.curve.x.tobytes())
            parts.append(c.current.x.tobytes())
        return hash(tuple(parts))

    def _tf_arrays(self):
        """Stacked TF geometry/current arrays; cached while DOFs unchanged."""
        key = self._tf_dofs_hash()
        cached = getattr(self, "_tf_arrays_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1], cached[2], cached[3]
        gammas = np.stack([c.curve.gamma() for c in self.coils_TF], axis=0)
        gammadashs = np.stack([c.curve.gammadash() for c in self.coils_TF], axis=0)
        currents = np.array(
            [c.current.get_value() for c in self.coils_TF], dtype=float
        )
        self._tf_arrays_cache = (key, gammas, gammadashs, currents)
        return gammas, gammadashs, currents

    def _gather_beta_work_to_all_numpy(self, beta_work: np.ndarray) -> np.ndarray:
        """Expand work-space ``beta_work`` (n_work * nd_per,) into the full
        ``n_dof_total``-vector ``beta_all`` used everywhere downstream.

        In the full path (``_reduced_active == False``) ``beta_work`` is
        already ``n_dof_total``-sized and this returns a copy; in the
        reduced path ``beta_work`` is ``n_base * nd_per`` and we gather
        via the stored base-index map with the matching per-replica sign
        (``+1`` pure rotation, ``-1`` stellsym image; see
        :func:`_gather_beta_work_to_all`).  ``_signs_effective`` is
        all-``+1`` on the full path, reducing this to a plain copy
        in that case.
        """
        beta_work = np.asarray(beta_work)
        if not self._reduced_active:
            return beta_work
        nd_per = self._K_stack.shape[2]
        n_work = beta_work.size // nd_per
        beta_stack = beta_work.reshape(n_work, nd_per)
        signs = np.asarray(self._signs_effective, dtype=beta_stack.dtype)
        gathered = beta_stack[self._base_indices_arr] * signs[:, None]
        return gathered.reshape(-1)

    def _setup_jax(self) -> None:
        """Store JAX views of NumPy geometry; JIT callables are module-level.

        The dense block-diagonal ``K_basis``/``phi_mat`` DeviceArrays are no
        longer created here.  Instead the JIT bodies consume the stacked
        ``_jax_K_stack``/``_jax_phi_work_stack``/``_jax_w_work_stack`` arrays
        (``n_pucks``-fold smaller) plus ``_jax_base_indices`` for the
        symmetry-reduced path.

        ``_jax_Lr_chol`` / ``_jax_Lr_eigf_chol`` are built in :meth:`_rebuild`
        next to ``_L_red`` (not duplicated here).
        """
        self._jax_Q = jnp.asarray(self._Q)
        self._jax_Q_c = jnp.asarray(self._Q_c)
        self._jax_quad_pts = jnp.asarray(self._quad_points)
        self._jax_quad_n = jnp.asarray(self._quad_normals)
        self._jax_w_q = jnp.asarray(self._quad_weights)
        # Note: full-size ``_jax_phi_stack`` / ``_jax_w_stack`` device
        # copies were intentionally removed -- the JIT bodies consume the
        # (possibly work-sized) ``_jax_phi_work_stack`` /
        # ``_jax_w_work_stack`` variants instead, and the full-size host
        # arrays are kept only for legacy dense ``_phi_mat`` reconstruction.
        if self._jax_phi_work_stack is not None:
            self._jax_phi_work_stack = jnp.asarray(self._jax_phi_work_stack)
        if self._jax_w_work_stack is not None:
            self._jax_w_work_stack = jnp.asarray(self._jax_w_work_stack)
        self._jax_base_indices = jnp.asarray(self._base_indices_arr)
        # Effective per-replica signs (``+1`` pure rotation, ``-1``
        # stellsym image) fed to :func:`_fold_Bn_to_work` /
        # :func:`_gather_beta_work_to_all`.  Matches ``_replica_signs``
        # on the reduced path and is all-``+1`` on the full-``L``
        # fallback; see ``_signs_effective`` in :meth:`_rebuild`.
        # Stored as float32 because JAX's ``segment_sum`` accumulates
        # in the payload's dtype; the sign is exact.
        self._jax_replica_signs = jnp.asarray(
            self._signs_effective, dtype=jnp.float32
        )

    @property
    def _K_stack(self) -> Optional[np.ndarray]:
        """Host view of ``_jax_K_stack`` (no duplicate NumPy storage)."""
        if not self._uniform_puck_shape or self._jax_K_stack is None:
            return None
        return np.asarray(self._jax_K_stack)

    @property
    def _phi_stack(self) -> Optional[np.ndarray]:
        if not self._uniform_puck_shape or self._jax_phi_stack is None:
            return None
        return np.asarray(self._jax_phi_stack)

    @property
    def _w_stack(self) -> Optional[np.ndarray]:
        if not self._uniform_puck_shape or self._jax_w_stack is None:
            return None
        return np.asarray(self._jax_w_stack)

    @property
    def _phi_work_stack(self) -> Optional[np.ndarray]:
        if self._jax_phi_work_stack is None:
            return None
        return np.asarray(self._jax_phi_work_stack)

    @property
    def _w_work_stack(self) -> Optional[np.ndarray]:
        if self._jax_w_work_stack is None:
            return None
        return np.asarray(self._jax_w_work_stack)

    # Lazy dense views for legacy callers (VTK export, regression tests).
    @property
    def _K_basis(self) -> np.ndarray:
        """Dense block-diagonal basis table ``(nq_total, n_dof_total, 3)``.

        Built on demand from the compact stacked representation
        ``_K_stack``.  Retained only for backward compatibility with legacy
        callers (e.g. :mod:`simsopt.field.puck_vtk`); internal hot paths
        use ``_K_stack`` directly.
        """
        if self._K_stack is None:
            raise RuntimeError(
                "Dense _K_basis requested but pucks have heterogeneous shapes; "
                "this code path is not supported.  Use _K_stack explicitly."
            )
        n_pucks, nq_per, nd_per, _ = self._K_stack.shape
        n_dof_total = self._n_dof_total
        nq_total = n_pucks * nq_per
        K = np.zeros((nq_total, n_dof_total, 3), dtype=self._K_stack.dtype)
        for pidx in range(n_pucks):
            r0, r1 = self._quad_row_ranges[pidx]
            d0 = self._dof_offsets[pidx]
            d1 = d0 + nd_per
            K[r0:r1, d0:d1, :] = self._K_stack[pidx]
        return K

    @property
    def _phi_mat(self) -> np.ndarray:
        """Dense block-diagonal ``phi_values`` ``(nq_total, n_dof_total)``.

        Lazily reconstituted from ``_phi_stack`` on access.
        """
        if self._phi_stack is None:
            raise RuntimeError(
                "Dense _phi_mat requested but pucks have heterogeneous shapes; "
                "this code path is not supported.  Use _phi_stack explicitly."
            )
        n_pucks, nq_per, nd_per = self._phi_stack.shape
        n_dof_total = self._n_dof_total
        nq_total = n_pucks * nq_per
        M = np.zeros((nq_total, n_dof_total), dtype=self._phi_stack.dtype)
        for pidx in range(n_pucks):
            r0, r1 = self._quad_row_ranges[pidx]
            d0 = self._dof_offsets[pidx]
            d1 = d0 + nd_per
            M[r0:r1, d0:d1] = self._phi_stack[pidx]
        return M

    @property
    def _L_full(self) -> np.ndarray:
        """Deprecated alias for :attr:`_L_work`.

        Historical name from before the symmetry reduction refactor.  The
        returned matrix is the "work" inductance: on the full-replica path
        (``_reduced_active == False``) this is the full
        ``(n_all * nd_per, n_all * nd_per)`` matrix; on the symmetry-reduced
        path it is the base-puck orbit-folded matrix of shape
        ``(n_base * nd_per, n_base * nd_per)``.  Use :attr:`_L_work`
        directly and disambiguate with :attr:`_reduced_active`.

        Emits a one-shot :class:`DeprecationWarning` per process on first
        access (``_L_full_deprecation_warned`` class attribute) so that
        repeated access inside tight loops does not flood the log.
        """
        if not PSCBulkArray._L_full_deprecation_warned:
            import warnings

            warnings.warn(
                "PSCBulkArray._L_full is a historical alias for _L_work; "
                "prefer _L_work (and use _reduced_active to disambiguate "
                "full vs symmetry-folded shape).",
                DeprecationWarning,
                stacklevel=2,
            )
            PSCBulkArray._L_full_deprecation_warned = True
        if self._L_work is None:
            raise AttributeError(
                "PSCBulkArray._L_full / _L_work was freed after rebuild because "
                "no puck geometry DOF is free and solver_mode=='energy'.  Use "
                "_L_red for the reduced matrix, or unfix a puck DOF and "
                "rebuild to re-materialize the work inductance."
            )
        return self._L_work

    # ------------------------------------------------------------------
    # JAX full path (local basis stacks for geometry VJP)
    # ------------------------------------------------------------------

    def _ensure_jax_full(self, *, force: bool = False) -> None:
        """Cache per-puck local-frame stacks for :func:`_B_eval_full_jitted`.

        Also exposes the dense rim-continuity projector ``self._Q_c`` as
        ``self._jax_Q_c`` for the free-DOF JAX path so that the constrained
        solve ``Q_c^T L Q_c \\alpha = Q_c^T f`` is used inside the VJP.

        When all puck geometry DOFs are fixed, this is a no-op unless
        ``force=True`` (diagnostics / tests that must materialise local stacks).

        Args:
            force: If ``True``, build local stacks even when every puck DOF is
                fixed (invalidates the usual "frozen geometry" fast path).
        """
        if not force and not self._has_free_puck_dofs():
            return
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

        When ``solver_mode == "shell_l2"``, takes a pure-NumPy path:
        assemble ``g = -H^T diag(w) Bn^{TF}`` from the cached ``_Hw``
        and solve ``Q^T M Q alpha = Q^T g`` with the same eigenfloor
        kernel.  The reduced symmetry path is disabled in this mode
        (see :meth:`_rebuild`) so no gather is needed.
        """
        Bn = self._compute_bn_at_quads_numpy()
        if self.solver_mode == "shell_l2":
            g = -self._Hw @ Bn
            g_r = np.asarray(self._Q).T @ g
            alpha = np.asarray(
                shell_solve_prefactored_pure(
                    self._jax_Lr_eigf_chol,
                    jnp.asarray(g_r),
                )
            )
            beta_work = np.asarray(self._Q) @ alpha
            return self._gather_beta_work_to_all_numpy(beta_work)
        beta_work = np.array(
            _beta_eigenfloor_from_bn_jitted(
                self._jax_Lr_eigf_chol,
                self._jax_Q,
                self._jax_phi_work_stack,
                self._jax_w_work_stack,
                self._jax_base_indices,
                self._jax_replica_signs,
                jnp.asarray(Bn),
            )
        )
        return self._gather_beta_work_to_all_numpy(beta_work)

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
        """Passive bulk B at Cartesian points ``(N, 3)``.

        When ``solver_mode == "shell_l2"`` the free-DOF-VJP and TF-only-
        VJP JIT paths are bypassed (not implemented for the L^2 branch
        yet): we fall back to a pure-NumPy Biot-Savart of the cached
        :math:`\\beta`, which is recomputed by :meth:`recompute_currents`
        whenever TF currents, TF geometry or puck DOFs change.
        """
        g_tf, gd_tf, I_tf = self._tf_arrays()
        pts = np.asarray(points)
        if self.solver_mode == "shell_l2":
            if self._has_free_puck_dofs():
                raise NotImplementedError(
                    "solver_mode='shell_l2' does not support free puck DOFs "
                    "yet; fix all puck DOFs or use solver_mode='energy'."
                )
            acc = jnp.float32 if self._use_f32_bs else None
            return np.array(
                shell_biot_savart_stacked_pure(
                    self._jax_K_stack,
                    self._jax_quad_pts,
                    self._jax_w_q,
                    jnp.asarray(self.beta),
                    jnp.asarray(pts),
                    accumulate_dtype=acc,
                )
            )
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
                    int(self._full_L_band_size),
                )
            )
        if _USE_JAX_TF_VJP:
            return np.array(
                _B_eval_jitted(
                    self._jax_Lr_chol,
                    self._jax_Q,
                    self._jax_quad_pts,
                    self._jax_quad_n,
                    self._jax_phi_work_stack,
                    self._jax_w_work_stack,
                    self._jax_K_stack,
                    self._jax_w_q,
                    self._jax_base_indices,
                    self._jax_replica_signs,
                    g_tf,
                    gd_tf,
                    I_tf,
                    pts,
                )
            )
        Bn = self._compute_bn_at_quads_numpy()
        return np.array(
            _B_eval_from_bn_jitted(
                self._jax_Lr_chol,
                self._jax_Q,
                self._jax_quad_pts,
                self._jax_phi_work_stack,
                self._jax_w_work_stack,
                self._jax_K_stack,
                self._jax_w_q,
                self._jax_base_indices,
                self._jax_replica_signs,
                jnp.asarray(Bn),
                pts,
            )
        )

    def get_shell_currents(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(K, |K|)``: ``(n_quad, 3)`` sheet current from current ``beta``.

        Uses the stacked ``_K_stack`` directly (per-puck einsum) so the
        dense ``(nq_total, n_dof_total, 3)`` monolithic basis is never
        allocated -- this keeps memory linear in ``n_pucks`` rather than
        quadratic.
        """
        if self._K_stack is not None:
            n_pucks, nq_per, nd_per, _ = self._K_stack.shape
            b_stack = np.asarray(self.beta).reshape(n_pucks, nd_per)
            K_vec = np.einsum("pqak,pa->pqk", self._K_stack, b_stack).reshape(
                n_pucks * nq_per, 3
            )
            return K_vec, np.linalg.norm(K_vec, axis=-1)
        # Heterogeneous-shape fallback (rare): materialize dense on demand.
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
                self._jax_Lr_chol,
                self._jax_Q,
                self._jax_quad_pts,
                self._jax_quad_n,
                self._jax_phi_work_stack,
                self._jax_w_work_stack,
                self._jax_K_stack,
                self._jax_w_q,
                self._jax_base_indices,
                self._jax_replica_signs,
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
        """TF VJP: adjoint through reduced shell solve + :class:`BiotSavart` (C++).

        Uses the stacked ``_K_stack``/``_phi_stack`` representation so the
        dense ``(nq_total, n_dof_total, ...)`` arrays never materialise.

        When the symmetry-reduced path is active, the adjoint of the
        ``beta_base -> beta_all`` gather is a signed ``segment_sum`` over
        orbits (``lam_beta_work[b, :] = sum_{r in orbit(b)} sigma_r
        lam_beta_all[r, :]``), and the adjoint of the ``Bn_all -> Bn_base``
        signed fold is a sign-weighted broadcast (``lam_Bn_all[r, q] =
        sigma_r lam_Bn_work[base(r), q]``).  Both are applied with plain
        NumPy here (no JAX JIT needed) so the analytic C++
        :class:`BiotSavart` VJP can be used for the outer coil-DOF
        gradient.  On the full path ``signs`` is identically ``+1`` and
        the weighting drops out.

        Sign convention (airtight): the forward loading vector is
        ``f = -Phi^T W Bn`` (see
        :func:`~simsopt.field.bulk_inductance.shell_loading_vector_stacked_pure`),
        so the adjoint of ``f`` w.r.t. ``Bn_work`` carries an explicit
        ``-1`` factor: ``lam_Bn_work = -W * (Phi @ lam_f_work)``.  Missing
        this factor silently negates the entire bulk contribution to the
        TF-DOF gradient and is covered by
        ``test_vjp_tf_analytic_matches_jax_multi_puck``.

        ``BiotSavart`` caveat: ``BiotSavart.dB_by_dcoilcurrents`` returns
        the field-cache entries without triggering ``compute()`` when
        they are missing, so a freshly constructed ``bs`` (no ``B()`` /
        ``compute()`` yet) yields all-zero current-DOF gradients from
        ``bs.B_vjp``.  We prime the cache with ``bs.B()`` before calling
        ``B_vjp`` so both curve- and current-DOF gradients are returned
        (covered by the current-DOF branch of the same test).
        """
        v_B = np.asarray(v_B).reshape(-1, 3)
        pts = np.asarray(pts)
        K_stack = self._jax_K_stack
        quad_pts = self._jax_quad_pts
        w_quad = self._jax_w_q
        beta_all = jnp.asarray(self.beta)

        def fwd_bs(b_all: jnp.ndarray) -> jnp.ndarray:
            return shell_biot_savart_stacked_pure(
                K_stack,
                quad_pts,
                w_quad,
                b_all,
                jnp.asarray(pts),
                eps=_EPS_BS,
            )

        _, vjp_bs = vjp(fwd_bs, beta_all)
        lam_beta_all = np.asarray(vjp_bs(jnp.asarray(v_B))[0])

        # Adjoint of the signed gather
        # ``beta_all[r, :] = sigma_r * beta_work[base(r), :]``: signed
        # segment-sum of ``lam_beta_all`` into ``lam_beta_work``.
        nd_per = self._K_stack.shape[2]
        n_all = self._K_stack.shape[0]
        lam_beta_all_stack = lam_beta_all.reshape(n_all, nd_per)
        signs_arr = np.asarray(
            self._signs_effective, dtype=lam_beta_all_stack.dtype
        )
        n_work = (
            self._n_base_pucks if self._reduced_active else n_all
        )
        lam_beta_work = np.zeros((n_work, nd_per))
        np.add.at(
            lam_beta_work,
            self._base_indices_arr,
            signs_arr[:, None] * lam_beta_all_stack,
        )
        lam_beta_work = lam_beta_work.reshape(-1)

        Q = self._Q
        qt_lam = Q.T @ lam_beta_work
        # Match :func:`~simsopt.field.bulk_inductance.shell_solve_linear_pure`
        # / prefactored ``self._jax_Lr_chol`` (same jitter as forward).
        Lr_chol = self._Lr_chol_host
        _y = solve_triangular(Lr_chol, qt_lam, lower=True, check_finite=True)
        lam_r = solve_triangular(
            Lr_chol.T, _y, lower=False, check_finite=True
        )
        lam_f_work = Q @ lam_r  # (n_work * nd_per,)

        # Adjoint of the loading vector coupled with the signed fold.
        # Forward: f_work[p, a] = -sum_q phi[p, q, a] * w[p, q] * Bn_work[p, q]
        # (the leading minus comes from ``shell_loading_vector_stacked_pure``),
        # followed by Bn_work[b, q] = sum_{r in orbit(b)} sigma_r Bn_all[r, q].
        # The adjoint therefore reads
        #   lam_Bn_all[r, q] = -sigma_r * w_quad[r, q] *
        #                      sum_a phi_work[base(r), q, a] lam_f_work[base(r), a]
        # and the explicit ``-`` below is *essential* (without it the analytic
        # VJP returns ``-grad`` instead of ``grad`` and Taylor tests on large
        # bulk contributions fail with rel_err ~= 2).
        phi_work_stack = self._phi_work_stack
        lam_f_work_stack = lam_f_work.reshape(n_work, nd_per)
        lam_Bn_work_stack = np.einsum(
            "pqa,pa->pq",
            phi_work_stack,
            lam_f_work_stack,
        )
        lam_Bn_all_stack = (
            signs_arr[:, None] * lam_Bn_work_stack[self._base_indices_arr]
        )
        w2d = getattr(self, "_quad_weights_2d", None)
        if w2d is None:
            w2d = self._quad_weights.reshape(n_all, -1)
        lam_Bn = -(w2d * lam_Bn_all_stack).reshape(-1)
        v_quad = lam_Bn[:, np.newaxis] * self._quad_normals

        pts_id = id(self._quad_points)
        bs = getattr(self, "_bs_bn", None)
        if bs is None or getattr(self, "_bs_bn_pts_id", None) != pts_id:
            bs = BiotSavart(self.coils_TF)
            bs.set_points_cart(np.ascontiguousarray(self._quad_points))
            self._bs_bn = bs
            self._bs_bn_pts_id = pts_id
        # Prime the field cache so ``dB_by_dcoilcurrents`` returns the
        # populated per-coil fields rather than fresh zeros; otherwise
        # ``B_vjp`` silently drops the current-DOF contribution.
        bs.B()
        return bs.B_vjp(v_quad)

    def _vjp_puck_geometry(self, v_B, pts):
        """VJP w.r.t. puck center + quaternion + TF DOFs via full JAX forward."""
        if not self._has_free_puck_dofs():
            return self._vjp_tf_only(v_B, pts)
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
            int(self._full_L_band_size),
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
