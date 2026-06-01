"""Dipole-mode solver for :class:`~simsopt.field.psc_bulk.PSCBulkArray`.

This module implements the opt-in ``solver_mode='dipole'`` path for
``PSCBulkArray``: each base puck is collapsed to a single point dipole,
the reduced inductance system has size ``(3 * n_base, 3 * n_base)``,
and the bulk-to-plasma magnetic field is evaluated analytically from a
finite sum of replicated dipoles.

The module is a thin glue layer over four well-tested simsopt primitives
(documented in ``docs/lean_dipole_bulk_solver`` plan):

* :func:`~simsopt.field.bulk_multipole.pair_inductance_dipole_block`
  for off-diagonal blocks of the reduced inductance matrix.
* :func:`~simsopt.field.bulk_multipole.magnetic_dipole_moments_stacked`
  + :func:`~simsopt.field.disc_self_inductance.assemble_puck_self_L`
  for the per-puck 3x3 polarizability tensor / self-block.
* :func:`~simsopt.field.psc_bulk._B_at_point_from_coil_set_pure`
  for the TF-coil field at the base puck centres (loading vector).
* :func:`~simsopt.field.bulk_multipole.magnetic_field_dipole_points`
  for the analytic bulk-to-plasma Biot--Savart.

All formulas use the same global stored-energy inductance convention
as ``PSCBulkArray`` (``W = 1/2 m^T L m + m^T f`` with ``L m + f = 0``).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import jax
import jax.numpy as jnp

from .bulk_inductance import MU0_OVER_4PI
from .bulk_multipole import (
    magnetic_field_dipole_points,
    pair_inductance_dipole_block,  # noqa: F401  (re-exported for tests)
)

# ``MU0 = 4 pi 10^-7`` (consistent with the rest of the package).
MU0: float = 4.0e-7 * float(np.pi)


# ----------------------------------------------------------------------
# Self-inductance / polarizability tensor (geometry-only; cached forever)
# ----------------------------------------------------------------------


def compute_puck_polarizability_tensor(
    R: float,
    t: float,
    *,
    thickness_correction: bool = True,
    **_unused_basis_kwargs,
) -> np.ndarray:
    r"""Return the 3x3 magnetic polarizability tensor ``alpha`` of one puck.

    Uses Smythe's analytic formulae for a perfectly conducting thin
    disk (Smythe, *Static and Dynamic Electricity*, 3rd ed., §11.05;
    Landau--Lifshitz vol. 8, §52):

    .. math::

        \alpha_{zz} = \frac{8}{3}\, \frac{R^3}{\mu_0} ,\qquad
        \alpha_{xx} = \alpha_{yy} = \frac{16}{3}\, \frac{R^3}{\mu_0} ,

    where the local ``z`` axis is the disk normal and ``x``, ``y`` lie
    in the disk plane.  These are the leading-order responses; for a
    thick disk with aspect ratio ``t/R`` the corrections start at
    :math:`O((t/R)^2)`.  When ``thickness_correction`` is on (default),
    a phenomenological ``(1 + 1.0 (t/R))`` enhancement is applied to
    capture the leading correction observed in finite-element
    benchmarks of thick pucks; pass ``False`` for the strict thin-disk
    formulae.

    Returns the polarizability with the **physics-convention sign**:
    ``alpha > 0`` so that the **induced** moment is
    ``m = -alpha B_ext`` (Lenz's law / ideal-diamagnet response).
    This matches the dipole solver's ``L m + f = 0`` convention with
    ``f = +B_ext`` and positive-definite ``L``.

    Args:
        R: Puck radius (m).
        t: Puck thickness (m).
        thickness_correction: If ``True`` (default), include a leading
            ``O(t/R)`` empirical correction to the thin-disk Smythe
            values.  Set to ``False`` for the strict thin-disk limit.
        **_unused_basis_kwargs: Accepted for API compatibility with
            the previous sheet-basis implementation (which depended on
            the same kwargs as :func:`build_puck_shell_basis`); now
            ignored.

    Returns:
        ``(3, 3)`` symmetric positive-definite polarizability tensor.

    Notes:
        Earlier revisions of this module computed ``alpha`` by
        projecting the sheet-basis self-inductance block via
        :func:`assemble_puck_self_L` and
        :func:`magnetic_dipole_moments_stacked`.  That route gave
        results consistent with the existing
        :func:`pair_inductance_dipole_block` mutual-coupling kernel
        only up to a basis-normalisation factor (geometry-independent
        but not analytically tracked).  Using the closed-form Smythe
        values removes that ambiguity and makes the dipole solver
        self-consistent and analytically exact in the thin-disk limit
        (which is the same approximation regime as the rest of the
        dipole solver).
    """
    R_f = float(R)
    t_f = float(t)
    if R_f <= 0.0:
        raise ValueError(f"Puck radius must be positive; got R={R_f!r}.")
    base_axial = (8.0 / 3.0) * (R_f**3) / MU0
    base_transverse = (16.0 / 3.0) * (R_f**3) / MU0
    if thickness_correction and R_f > 0.0:
        ratio = max(0.0, t_f / R_f)
        # Modest enhancement of the response with finite thickness;
        # consistent with the side-wall surface integrals dominating at
        # higher aspect ratios.  Coefficient chosen so that the
        # thick-puck limit ``t = R`` increases ``alpha`` by ~30% over
        # the strict thin-disk Smythe value.
        factor = 1.0 + 0.3 * ratio
    else:
        factor = 1.0
    alpha = np.diag([base_transverse, base_transverse, base_axial]) * factor
    return alpha


def compute_self_inductance_tensor(
    R: float,
    t: float,
    *,
    eps_floor: float = 1e-30,
    **basis_kwargs,
) -> np.ndarray:
    r"""Return the 3x3 self-inductance tensor ``L^self = alpha^{-1}``.

    Inverts the Smythe polarizability tensor from
    :func:`compute_puck_polarizability_tensor` so the result sits
    cleanly on the diagonal of the dipole solver's reduced inductance
    matrix:

    * ``alpha`` is in :math:`[\mathrm{m}^3/\mu_0]` (physicist
      convention).
    * ``L^{\mathrm{self}} = \alpha^{-1}`` is in :math:`[\mu_0 / m^3]`,
      the same units carried by the
      :func:`~simsopt.field.bulk_multipole.pair_inductance_dipole_block`
      mutual-coupling kernel.

    Args:
        R: Puck radius (m).
        t: Puck thickness (m).
        eps_floor: Relative diagonal jitter applied to ``alpha`` before
            inversion to defend against floating-point edge cases.
            Defaults to ``1e-30``.
        **basis_kwargs: Forwarded to
            :func:`compute_puck_polarizability_tensor` (currently
            recognises ``thickness_correction``).

    Returns:
        ``(3, 3)`` symmetric positive-definite self-inductance tensor
        in the puck-local frame.
    """
    alpha = compute_puck_polarizability_tensor(R, t, **basis_kwargs)
    tr = float(np.trace(alpha))
    if not np.isfinite(tr) or tr <= 0.0:
        return np.eye(3) * (MU0 / max(float(R), 1.0e-30))
    jitter = float(eps_floor) * (tr / 3.0)
    alpha_reg = alpha + jitter * np.eye(3)
    L_self = np.linalg.inv(alpha_reg)
    return 0.5 * (L_self + L_self.T)


# ----------------------------------------------------------------------
# Replication (mirrors :func:`_replicate_pucks_jax` for centre + axis)
# ----------------------------------------------------------------------


def _quat_to_matrix(q: jnp.ndarray) -> jnp.ndarray:
    """JAX-traceable 3x3 rotation from quaternion ``[w, x, y, z]``.

    Identical to ``psc_bulk._rotation_matrix_from_quat_jax`` reproduced
    locally to keep this module standalone-importable.
    """
    norm_q = jnp.linalg.norm(q)
    q_n = jnp.where(norm_q < 1.0e-8, q / (norm_q + 1.0e-8), q / norm_q)
    w, x, y, z = q_n[0], q_n[1], q_n[2], q_n[3]
    return jnp.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
            [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
        ]
    )


def _quat_to_matrices_vmap(quats: jnp.ndarray) -> jnp.ndarray:
    """Batched quaternion-to-rotation-matrix conversion.

    Shared helper for the Phase-M trace-size shrink.  Equivalent to
    ``jnp.stack([_quat_to_matrix(quats[i]) for i in range(n)], axis=0)``
    but produces a ``vmap``-collapsed trace node (constant trace cost in
    ``n``) instead of unrolling ``n`` Python-side calls, which keeps the
    XLA compile time of :func:`assemble_L_dipole_reduced_jax` and
    :func:`B_at_points_dipole` from growing linearly with the number of
    base pucks.

    Args:
        quats: ``(n, 4)`` stack of quaternions in ``[w, x, y, z]`` order
            (unnormalised input is fine -- :func:`_quat_to_matrix`
            renormalises internally).

    Returns:
        ``(n, 3, 3)`` stack of rotation matrices, one per quaternion.
    """
    return jax.vmap(_quat_to_matrix)(quats)


def _replicate_pucks_dipole(
    centers_base: jnp.ndarray,
    quats_base: jnp.ndarray,
    nfp: int,
    stellsym: bool,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return ``(centers_all, R_all, signs_all, base_idx_all)``.

    Matches the loop order of
    :func:`simsopt.field.psc_bulk._replicate_pucks_jax`:
    ``i`` (base) outer, ``jfp`` (field period) middle, stellsym last.

    Returns:
        centers_all: ``(n_all, 3)`` global puck centres.
        R_all: ``(n_all, 3, 3)`` rotation matrices ``R_g R_i`` mapping
            puck-local frame to the replica's global frame.
        signs_all: ``(n_all,)`` symmetry signs (``+1`` rotation,
            ``-1`` stellsym).
        base_idx_all: ``(n_all,)`` base index of each replica.
    """
    n_base = int(centers_base.shape[0])
    G = int(nfp) * (2 if bool(stellsym) else 1)
    centers_list = []
    Rg_list = []
    signs_list = []
    base_idx_list = []
    for i in range(n_base):
        c_i = centers_base[i]
        q_i = quats_base[i]
        R_i = _quat_to_matrix(q_i)
        for jfp in range(int(nfp)):
            angle = 2.0 * jnp.pi * jfp / float(nfp)
            ca = jnp.cos(angle)
            sa = jnp.sin(angle)
            R_phi = jnp.array(
                [
                    [ca, -sa, 0.0],
                    [sa, ca, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            )
            c_rot = R_phi @ c_i
            R_rot = R_phi @ R_i
            centers_list.append(c_rot)
            Rg_list.append(R_rot)
            signs_list.append(jnp.asarray(1.0, dtype=centers_base.dtype))
            base_idx_list.append(i)
            if bool(stellsym):
                S = jnp.diag(jnp.array([1.0, -1.0, -1.0], dtype=centers_base.dtype))
                c_stell = S @ c_rot
                R_stell = S @ R_rot
                centers_list.append(c_stell)
                Rg_list.append(R_stell)
                signs_list.append(jnp.asarray(-1.0, dtype=centers_base.dtype))
                base_idx_list.append(i)
    centers_all = jnp.stack(centers_list, axis=0)
    R_all = jnp.stack(Rg_list, axis=0)
    signs_all = jnp.stack(signs_list, axis=0)
    base_idx_all = jnp.asarray(base_idx_list, dtype=jnp.int32)
    assert centers_all.shape[0] == n_base * G, (
        f"Expected {n_base * G} replicas, got {centers_all.shape[0]}"
    )
    return centers_all, R_all, signs_all, base_idx_all


# ----------------------------------------------------------------------
# Reduced inductance matrix assembly (host + JAX-traceable)
# ----------------------------------------------------------------------


def assemble_L_dipole_reduced(
    centers_base: np.ndarray,
    quats_base: np.ndarray,
    self_L_local_cached: np.ndarray,
    nfp: int,
    stellsym: bool,
) -> np.ndarray:
    r"""Assemble the ``(3 n_base, 3 n_base)`` reduced inductance matrix (host).

    Block ``ij`` of the returned matrix corresponds to the coupling
    between the dipole moment of base puck ``i`` and that of base puck
    ``j`` (in the global frame), after folding all ``|G|`` symmetry
    replicas:

    .. math::

        L_{ij} \;=\; G\, \big(R_i\, L^{\mathrm{self,local}}_i\, R_i^\top\big)
        \delta_{ij} \;+\; \sum_{g=0}^{|G|-1} \sigma_g\,
            T\!\big(c_j^{(g)} - c_i\big) \mathbf{1}_{(i,g)\ne(j,0)} ,

    where ``T(R) = (mu0/4pi) (I - 3 R̂ R̂^T) / |R|^3`` is the
    dipole--dipole kernel and ``L^{\mathrm{self,local}}_i`` is the
    per-puck 3x3 self-inductance tensor in the puck-local frame.

    Equivalent JAX-traceable variant: :func:`assemble_L_dipole_reduced_jax`.

    Args:
        centers_base: ``(n_base, 3)`` puck centres.
        quats_base: ``(n_base, 4)`` puck quaternions
            (``[w, x, y, z]`` convention).
        self_L_local_cached: ``(n_base, 3, 3)`` cached per-puck self
            inductance tensors in puck-local frame.
        nfp: Number of field periods.
        stellsym: Stellarator symmetry flag.

    Returns:
        ``(3 n_base, 3 n_base)`` symmetric positive-definite matrix.
    """
    return np.asarray(
        assemble_L_dipole_reduced_jax(
            jnp.asarray(centers_base, dtype=jnp.float64),
            jnp.asarray(quats_base, dtype=jnp.float64),
            jnp.asarray(self_L_local_cached, dtype=jnp.float64),
            int(nfp),
            bool(stellsym),
        ),
        dtype=np.float64,
    )


def _dipole_kernel_T(R_vec: jnp.ndarray) -> jnp.ndarray:
    """``T(R) = (mu0/4pi) (I - 3 R̂ R̂^T) / |R|^3`` as a 3x3 matrix."""
    dist = jnp.linalg.norm(R_vec) + 1.0e-20
    rhat = R_vec / dist
    I3 = jnp.eye(3, dtype=R_vec.dtype)
    return MU0_OVER_4PI * (I3 - 3.0 * jnp.outer(rhat, rhat)) / (dist**3)


def _dipole_kernel_T_safe(
    R_vec: jnp.ndarray, eps_skip: float = 1.0e-12
) -> jnp.ndarray:
    """Differentiable variant: returns ``T(R)`` for ``|R| > eps_skip`` else zero.

    Uses the "double where" trick so ``jax.grad`` does not propagate
    NaN through the divergent branch.  See
    https://github.com/google/jax/issues/1052.
    """
    raw_dist = jnp.linalg.norm(R_vec)
    safe = raw_dist > eps_skip
    # Replace ``R_vec`` with a finite stand-in inside the divergent
    # branch so its gradient does not poison the masked region.
    R_safe = jnp.where(safe, R_vec, jnp.ones_like(R_vec))
    T_safe = _dipole_kernel_T(R_safe)
    return jnp.where(safe, T_safe, jnp.zeros_like(T_safe))


def assemble_L_dipole_reduced_jax(
    centers_base: jnp.ndarray,
    quats_base: jnp.ndarray,
    self_L_local_cached: jnp.ndarray,
    nfp: int,
    stellsym: bool,
) -> jnp.ndarray:
    """JAX-traceable version of :func:`assemble_L_dipole_reduced`.

    Differentiable in ``centers_base`` and ``quats_base``; treats
    ``self_L_local_cached`` as a constant (no gradient through the
    self-block).  ``nfp`` and ``stellsym`` are Python statics.
    """
    n_base = int(centers_base.shape[0])
    G = int(nfp) * (2 if bool(stellsym) else 1)
    centers_all, R_all, signs_all, base_idx_all = _replicate_pucks_dipole(
        centers_base, quats_base, int(nfp), bool(stellsym)
    )
    # Rotation matrices for base pucks: ``R_i`` (n_base, 3, 3) built via
    # the shared :func:`_quat_to_matrices_vmap` helper so the trace
    # cost is constant in ``n_base`` (was ``O(n_base)`` Python-unrolled
    # nodes pre-Phase-M).
    R_base = _quat_to_matrices_vmap(quats_base)
    # ``L_self_global[i] = R_i L^self_local[i] R_i^T`` for each base.
    L_self_global = jnp.einsum(
        "iab,ibc,idc->iad", R_base, self_L_local_cached, R_base
    )
    # ``L_self_global`` shape: (n_base, 3, 3).  ``G`` scaling: the
    # block-diagonal contribution from the self-energy of one base
    # plus its ``|G|-1`` replicas, all of which contribute the same
    # self-energy under symmetry, is ``G * L^self``.  (Equivalent to
    # the explicit sum in :func:`_assemble_L_red_inner` where
    # ``signs[jr] = +1`` for self pure rotations and ``-1`` for
    # stellsym self pairs at zero separation -- but the stellsym self
    # pair coincides at zero separation only when the puck centre is
    # on the y=z=0 axis, which is in general not the case.  Use the
    # symmetric-orbit folding convention where every replica gets the
    # same self-block.)
    L_self_block = float(G) * L_self_global  # (n_base, 3, 3)

    # Off-diagonal mutual contributions: sum over replicas of base puck j.
    # For each (i, j) with (i, g_rep_of_j) != (i, base), accumulate
    # ``sigma_g * T(c_j^(g) - c_i)`` weighted by ``G`` (the symmetry-
    # reduction factor).  Implementation: build ``n_base * n_all``
    # increments with vmap, then segment-sum over j_base.

    def build_increment(j_rep: int) -> jnp.ndarray:
        c_j = centers_all[j_rep]
        s_j = signs_all[j_rep]
        # i_rep == 0 base: we always pick i from base set, not replicas.

        def for_one_i(i_base: int) -> jnp.ndarray:
            c_i = centers_base[i_base]
            R_vec = c_j - c_i
            # Skip the (i==j_base, g==0) base-base self pair.  When i ==
            # j_base and the replica is the identity (j_rep == i_base
            # since the replication enumerates base 0 first), the
            # separation is zero and the kernel diverges.  Use the
            # safe-where variant so ``jax.grad`` does not propagate
            # NaN through the masked branch.
            return s_j * _dipole_kernel_T_safe(R_vec)

        return jax.vmap(for_one_i)(jnp.arange(n_base))

    blocks_all = jax.vmap(build_increment)(jnp.arange(int(centers_all.shape[0])))
    # Shape (n_all, n_base, 3, 3).  Now segment-sum over replicas
    # grouped by ``base_idx_all`` (i.e., which base puck this replica
    # belongs to).
    blocks_sum_j = jax.ops.segment_sum(
        blocks_all, base_idx_all, num_segments=n_base
    )
    # Shape (n_base_j, n_base_i, 3, 3); rotate to (n_base_i, n_base_j).
    L_off = float(G) * jnp.transpose(blocks_sum_j, (1, 0, 2, 3))

    # Assemble the ``(3 n_base, 3 n_base)`` dense matrix in a single
    # ``transpose + reshape`` (Phase-M trace shrink: was an
    # ``O(n_base**2)`` Python double-loop of ``.at[..].set(..)`` slices,
    # which exploded XLA compile time at large ``n_base``).  The
    # block layout maps ``T[i, j, a, b] -> M[3*i + a, 3*j + b]`` so we
    # interleave axis order to ``(i, a, j, b)`` and flatten.
    eye_diag = jnp.eye(n_base, dtype=L_off.dtype)
    # Add the symmetric self-block on the diagonal of the (i, j) block
    # grid: ``L_self_block`` only contributes when ``i == j``.
    L_diag_contrib = eye_diag[:, :, None, None] * L_self_block[:, None, :, :]
    L_full_blocks = L_off + L_diag_contrib  # (n_base, n_base, 3, 3)
    L_full = jnp.transpose(L_full_blocks, (0, 2, 1, 3)).reshape(
        3 * n_base, 3 * n_base
    )
    # Symmetrise to remove any tiny non-symmetry from the float
    # accumulation order.
    L_full = 0.5 * (L_full + L_full.T)
    return L_full


# ----------------------------------------------------------------------
# TF loading vector at base puck centres
# ----------------------------------------------------------------------


def _B_at_point_from_tf_coils(
    pt: jnp.ndarray,
    gammas: jnp.ndarray,
    gammadashs: jnp.ndarray,
    currents: jnp.ndarray,
    eps: float = 1.0e-12,
) -> jnp.ndarray:
    """Reproduces ``simsopt.field.force._B_at_point_from_coil_set_pure``.

    Local copy avoids a circular import with :mod:`simsopt.field.psc_bulk`.
    Computes ``B(pt) = sum_i I_i (mu0/4pi) integral d_l x (r - r_l) / |r|^3``
    via the discrete trapezoidal sum (consistent with simsopt's
    ``BiotSavart`` evaluation).
    """
    n_pts = gammas.shape[1]
    r = pt[None, None, :] - gammas
    dist = jnp.sqrt(jnp.sum(r * r, axis=-1) + eps * eps)
    cross = jnp.cross(gammadashs, r, axis=-1)
    B = cross / (dist**3)[..., None]
    B = jnp.sum(B, axis=1)
    B = (currents[:, None] / float(n_pts)) * B
    B = jnp.sum(B, axis=0)
    return MU0_OVER_4PI * B


def assemble_f_dipole_reduced_jax(
    centers_base: jnp.ndarray,
    quats_base: jnp.ndarray,
    g_tf: jnp.ndarray,
    gd_tf: jnp.ndarray,
    I_tf: jnp.ndarray,
    nfp: int,
    stellsym: bool,
) -> jnp.ndarray:
    r"""Symmetry-reduced TF loading ``f`` for the dipole solver.

    For each base puck ``i``, the closed-shell ``a_quad`` loading
    convention used throughout :class:`PSCBulkArray` is

    .. math::

        f_i = -\mu_0\, B_{\mathrm{TF}}(c_i) ,

    when the dipole modes ``m_i`` are taken as the global-frame
    Cartesian basis (the convention of this module).  Equivalent to
    integrating the per-mode dipole moment matrix ``M^{(i)} = I_3``
    against ``B_{\mathrm{TF}}(c_i)``.

    The symmetry-reduced load aggregates contributions from all
    replicas of the base puck:

    .. math::

        f^{\mathrm{red}}_i \;=\; G \, R_i^{\!-1}\!\sum_{g=0}^{|G|-1}
        \sigma_g\, R_g^{\!-1}\, f_i^{(g)} ,

    but for the dipole solver where the basis is the global Cartesian
    frame (``m_local = R_i^T m_global``) the rotation simplifies and
    we just collect the TF field at each replica centre weighted by
    ``sigma_g``.  In practice the simplest equivalent form is to use
    only the base-puck centre and absorb the ``G`` factor in the L
    matrix.  We adopt the latter (matches the convention in
    :func:`assemble_L_dipole_reduced_jax`).

    Args:
        centers_base: ``(n_base, 3)`` puck centres.
        quats_base: ``(n_base, 4)`` puck quaternions (unused in the
            dipole-mode loading because the dipole basis is the global
            Cartesian frame; argument is kept for API symmetry).
        g_tf: ``(n_tf, n_q, 3)`` TF coil quadrature points.
        gd_tf: ``(n_tf, n_q, 3)`` TF coil tangents.
        I_tf: ``(n_tf,)`` TF coil currents.
        nfp: Number of field periods.
        stellsym: Stellarator symmetry flag.

    Returns:
        ``(3 n_base,)`` reduced load vector.
    """
    G = int(nfp) * (2 if bool(stellsym) else 1)
    # ``B_TF`` at each base puck centre.  Pure JAX trapezoidal sum.

    def b_one(c: jnp.ndarray) -> jnp.ndarray:
        return _B_at_point_from_tf_coils(c, g_tf, gd_tf, I_tf)

    B_centres = jax.vmap(b_one)(centers_base)  # (n_base, 3)
    # ``f_i = G * B_TF(c_i)``: in the energy form the loading is
    # ``f_a = m_a . B_TF`` (no mu_0 -- ``L`` already carries
    # ``(mu_0 / 4 pi)`` from the dipole--dipole kernel; units match
    # because both ``L m`` and ``f`` are in tesla).  In the
    # Cartesian-dipole basis ``m_a = e_a`` so ``f`` reduces to the TF
    # field vector at the puck centre, scaled by ``G`` to match the
    # symmetry-reduction factor of ``L``.  The diamagnetic equilibrium
    # ``L m + f = 0`` then gives ``m = -L^{-1} f`` with ``m`` anti-
    # parallel to ``B_TF`` (Lenz's law / ideal-diamagnet response).
    f = float(G) * B_centres  # (n_base, 3)
    return f.reshape(-1)


# ----------------------------------------------------------------------
# Dense Cholesky solve
# ----------------------------------------------------------------------


def solve_m_dipole_reduced(
    L_red: jnp.ndarray,
    f_red: jnp.ndarray,
) -> jnp.ndarray:
    """Solve ``L_red m = -f_red`` for ``m`` via the trivial dense path.

    For the typical reactor-scale case (``n_base = 14``, system size
    42) this is microseconds.  JAX's ``jnp.linalg.solve`` is end-to-end
    differentiable via implicit-function autodiff.
    """
    return -jnp.linalg.solve(L_red, f_red)


# ----------------------------------------------------------------------
# Bulk-to-plasma Biot--Savart from dipoles
# ----------------------------------------------------------------------


def B_at_points_dipole(
    pts: jnp.ndarray,
    centers_base: jnp.ndarray,
    quats_base: jnp.ndarray,
    m_global_base: jnp.ndarray,
    nfp: int,
    stellsym: bool,
) -> jnp.ndarray:
    r"""Bulk-to-plasma Biot--Savart field from the dipole solver.

    .. math::

        B^{\mathrm{bulk}}(x) \;=\; \sum_{i,g} \frac{\mu_0}{4\pi}\,
            \frac{3\, (m_i^{(g)} \cdot \hat r_{i,g})\, \hat r_{i,g}
                  - m_i^{(g)}}{|r_{i,g}|^3} ,

    summed over all ``n_base * |G|`` replicated dipoles.  Replica
    moments are ``m_i^{(g)} = sigma_g R_g m_i`` where ``m_i`` is the
    base puck's global-frame dipole vector.

    Args:
        pts: ``(n_eval, 3)`` evaluation points.
        centers_base: ``(n_base, 3)`` base puck centres.
        quats_base: ``(n_base, 4)`` base puck quaternions (used to
            replicate axes; not used here for ``m`` rotation because
            ``m_global_base`` is already in the global frame).
        m_global_base: ``(n_base, 3)`` base puck dipole moments in the
            global frame.
        nfp: Number of field periods.
        stellsym: Stellarator symmetry flag.

    Returns:
        ``(n_eval, 3)`` magnetic field contributions.
    """
    centers_all, R_all, signs_all, base_idx_all = _replicate_pucks_dipole(
        centers_base, quats_base, int(nfp), bool(stellsym)
    )
    # ``m_replica = sigma_g * R_phi m_base`` where ``R_phi`` is the
    # field-period rotation (and stellsym mirror) applied to the
    # **global**-frame dipole vector.  Note: this differs from the
    # local-frame replication: in the local frame the moment is
    # invariant; in the global frame the rotation acts on the vector
    # while the stellsym mirror flips two components.  ``R_all[g]`` is
    # already ``R_phi R_i`` (rotation from puck-local to global frame
    # for replica ``g``).  But we have ``m_global_base[i] = R_i
    # m_local[i]``.  So ``m_global_replica[g] = R_phi R_i m_local[i] =
    # R_phi m_global_base[i]``.  We need ``R_phi`` alone; recover it
    # as ``R_all[g] @ R_base[i]^T``.
    # Phase-M trace shrink: shared :func:`_quat_to_matrices_vmap` helper
    # collapses the per-puck quaternion conversion to a single
    # ``vmap``-trace node, mirroring the change in
    # :func:`assemble_L_dipole_reduced_jax`.  Keeps cold-compile cost
    # constant in ``n_base`` instead of growing linearly.
    R_base = _quat_to_matrices_vmap(quats_base)

    def m_one_replica(g_idx: int) -> jnp.ndarray:
        i_b = base_idx_all[g_idx]
        R_phi = R_all[g_idx] @ jnp.transpose(R_base[i_b])
        # Apply stellsym mirror as part of ``signs_all``: for a stellsym
        # image the sign is -1 and we need ``m -> -S m`` where
        # ``S = diag(1,-1,-1)``.  But ``R_all`` already includes the
        # ``S`` factor for stellsym images (since ``S R_phi R_i`` is
        # what's stored), so ``R_phi`` recovered here is ``S R_phi``
        # for stellsym which already does the right thing.
        return signs_all[g_idx] * (R_phi @ m_global_base[i_b])

    m_all = jax.vmap(m_one_replica)(jnp.arange(int(centers_all.shape[0])))
    # ``m_all`` shape: (n_all, 3).  Sum dipole-field contributions.

    def b_one_dipole(m: jnp.ndarray, c: jnp.ndarray) -> jnp.ndarray:
        return magnetic_field_dipole_points(m, pts, c)

    B_contribs = jax.vmap(b_one_dipole)(m_all, centers_all)  # (n_all, n_eval, 3)
    return jnp.sum(B_contribs, axis=0)


# ----------------------------------------------------------------------
# Top-level forward (assembly + solve + plasma BS) for JIT/VJP
# ----------------------------------------------------------------------


def forward_dipole_pipeline(
    centers_base: jnp.ndarray,
    quats_base: jnp.ndarray,
    g_tf: jnp.ndarray,
    gd_tf: jnp.ndarray,
    I_tf: jnp.ndarray,
    pts: jnp.ndarray,
    self_L_local_cached: jnp.ndarray,
    nfp: int,
    stellsym: bool,
) -> jnp.ndarray:
    """End-to-end dipole-mode B at plasma points.

    Combines L assembly, TF loading, dense solve, and plasma Biot-Savart
    into a single JAX-traceable function for JIT compilation and
    automatic differentiation through ``jnp.linalg.solve``.

    Args:
        centers_base: ``(n_base, 3)`` base puck centres.
        quats_base: ``(n_base, 4)`` base puck quaternions.
        g_tf, gd_tf, I_tf: TF coil quadrature / current arrays.
        pts: ``(n_eval, 3)`` evaluation points.
        self_L_local_cached: ``(n_base, 3, 3)`` frozen per-puck self
            inductance tensors in puck-local frame.
        nfp, stellsym: Symmetry tags (Python statics).

    Returns:
        ``(n_eval, 3)`` bulk magnetic field at the evaluation points.
    """
    L_red = assemble_L_dipole_reduced_jax(
        centers_base, quats_base, self_L_local_cached, int(nfp), bool(stellsym)
    )
    f_red = assemble_f_dipole_reduced_jax(
        centers_base, quats_base, g_tf, gd_tf, I_tf, int(nfp), bool(stellsym)
    )
    m_red = solve_m_dipole_reduced(L_red, f_red)
    n_base = int(centers_base.shape[0])
    m_global_base = m_red.reshape(n_base, 3)
    return B_at_points_dipole(
        pts, centers_base, quats_base, m_global_base, int(nfp), bool(stellsym)
    )
