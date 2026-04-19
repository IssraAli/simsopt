"""
Shell inductance matrix and ideal-diamagnetic passive-bulk field (surface currents on :math:`\\partial\\Omega`).
"""

from __future__ import annotations

import warnings
from functools import partial
from typing import List, Tuple

import jax

# Max puck pairs per batch in vectorized block assembly (tune for memory vs speed).
_PAIR_BATCH_SIZE = 64
import jax.numpy as jnp


@partial(jax.jit, static_argnames=("adaptive_self_reg",))
def _jax_pair_batch_blocks(
    Ki: jnp.ndarray,
    Kj: jnp.ndarray,
    pts_i: jnp.ndarray,
    pts_j: jnp.ndarray,
    w_i: jnp.ndarray,
    w_j: jnp.ndarray,
    delta_reg: jnp.ndarray,
    adaptive_self_reg: bool = False,
) -> jnp.ndarray:
    """One batch of upper-triangular puck-pair inductance blocks ``(B, nd, nd)``.

    Args:
        Ki: Per-pair "left" sheet-current basis, shape ``(B, nq_i, nd_i, 3)``.
        Kj: Per-pair "right" sheet-current basis, shape ``(B, nq_j, nd_j, 3)``.
        pts_i: Quadrature points for the left pucks, shape ``(B, nq_i, 3)``.
        pts_j: Quadrature points for the right pucks, shape ``(B, nq_j, 3)``.
        w_i: Quadrature weights for the left pucks, shape ``(B, nq_i)``.
        w_j: Quadrature weights for the right pucks, shape ``(B, nq_j)``.
        delta_reg: Regularization scalar or per-pair array.  If a scalar JAX
            array ``()`` or a Python float, the legacy uniform
            :math:`\\sqrt{r^2 + \\delta^2}` regularization is used
            (ignored when ``adaptive_self_reg=True``).  If an array of shape
            ``(B, nq_i, nq_j)`` it is used verbatim.
        adaptive_self_reg: When ``True``, compute a per-quadrature-cell
            effective patch radius
            :math:`\\delta_k = (3\\pi^{3/2}/8)\\sqrt{w_k}` (analytic
            flat-disc-patch prefactor, see :data:`_SELF_REG_COEFF`) and
            build :math:`\\delta_{ij} = (\\delta_i + \\delta_j)/2`.  This
            makes the coincident :math:`(i=i)` contribution match the
            analytic flat-disc self-integral exactly (to leading order in
            the patch radius) while preserving positive semi-definiteness
            of the regularized kernel.  When ``False``, fall back to the
            legacy uniform ``delta_reg`` path.

    Returns:
        ``(B, nd_i, nd_j)`` inductance blocks in henries.
    """
    r = pts_i[:, :, None, :] - pts_j[:, None, :, :]
    if adaptive_self_reg:
        delta_i = _SELF_REG_COEFF * jnp.sqrt(w_i)
        delta_j = _SELF_REG_COEFF * jnp.sqrt(w_j)
        delta_pair = 0.5 * (delta_i[:, :, None] + delta_j[:, None, :])
        dist = jnp.sqrt(jnp.sum(r**2, axis=-1) + delta_pair**2)
    else:
        dist = jnp.sqrt(jnp.sum(r**2, axis=-1) + delta_reg**2)
    dot = jnp.einsum("Biax,Bjbx->Bijab", Ki, Kj)
    kernel = dot / dist[..., None, None]
    return MU0_OVER_4PI * jnp.einsum("Bijab,Bi,Bj->Bab", kernel, w_i, w_j)


import jax.scipy as jscp
import numpy as np

__all__ = [
    "shell_inductance_matrix_pure",
    "shell_inductance_matrix_blockwise",
    "shell_inductance_matrix_jax_blockwise",
    "shell_loading_vector_pure",
    "shell_solve_linear_pure",
    "shell_solve_eigenfloor_pure",
    "shell_biot_savart_pure",
    "null_space_projection_matrix",
    "mu0_over_4pi",
]

MU0_OVER_4PI = 1e-7

# Analytic flat-disc-patch self-regularization prefactor.
#
# For a uniform sheet current ``K`` on a flat disc of area ``w`` and radius
# ``rho = sqrt(w / pi)``, the double-integral self-term evaluates to
# ``K.K * (8/3) * rho**3 = K.K * (8/3) * (w/pi)**(3/2)``.  A uniform
# Cauchy-like regularization ``1 / sqrt(r**2 + delta**2)`` applied to the
# single coincident quadrature cell yields instead ``K.K * w**2 / delta``.
# Equating the two fixes the coincident prefactor exactly:
#
#     delta_i = (3 * pi**(3/2) / 8) * sqrt(w_i)    (~= 2.0884 * sqrt(w_i))
#
# This replaces the naive ``delta_i = sqrt(w_i / pi)`` choice which is too
# small by a factor ``sqrt(3 * pi**2 / 8) ~= 1.924`` (the coincident cell
# is then overcounted by the square of that ratio, ``3 * pi**2 / 8 ~= 3.70``).
_SELF_REG_COEFF = 3.0 * np.pi**1.5 / 8.0


def mu0_over_4pi() -> float:
    """Return :math:`\\mu_0/(4\\pi)` in SI (H/m)."""
    return MU0_OVER_4PI


def shell_inductance_matrix_pure(
    K_basis: jnp.ndarray,
    quad_points: jnp.ndarray,
    quad_weights: jnp.ndarray,
    delta_reg: float = 1e-8,
    adaptive_self_reg: bool = False,
) -> jnp.ndarray:
    """
    Shell inductance matrix (Eq. 58 in the passive-bulk note):

    .. math::

        L_{ab} = \\frac{\\mu_0}{4\\pi} \\int_\\Sigma \\int_{\\Sigma'}
        \\frac{(\\mathbf{n}\\times\\nabla_s \\Phi_a)\\cdot(\\mathbf{n}'\\times\\nabla_s \\Phi_b')}{|\\mathbf{r}-\\mathbf{r}'|}
        \\, dS' \\, dS

    where here ``K_basis`` stores :math:`\\mathbf{K}_a = \\mathbf{n}\\times\\nabla_s \\Phi_a`.

    Args:
        K_basis: shape ``(n_quad, n_dof, 3)`` sheet-current basis vectors.
        quad_points: shape ``(n_quad, 3)``.
        quad_weights: shape ``(n_quad,)`` scalar quadrature weights (area elements).
        delta_reg: Legacy uniform regularization added under the distance for
            the weak singularity.  Only used when ``adaptive_self_reg`` is
            ``False``.
        adaptive_self_reg: Opt-in flag (default ``False`` for backward
            compatibility).  When ``True``, use a per-quadrature-cell-scaled
            regularization :math:`\\delta_i = (3\\pi^{3/2}/8)\\sqrt{w_i}`
            (see :data:`_SELF_REG_COEFF`) and
            :math:`\\delta_{ij} = (\\delta_i + \\delta_j)/2`.  This matches
            the analytic flat-disc-patch self-integral
            :math:`(8/3)(w_i/\\pi)^{3/2}` exactly on the coincident cell,
            replacing the earlier ``\\sqrt{w_i/\\pi}`` choice which
            overcounted the self term by :math:`3\\pi^2/8 \\approx 3.70`.
            When ``False``, fall back to the legacy uniform
            :math:`\\sqrt{r^2+\\delta^2}` regularization with the scalar
            ``delta_reg``.

    Returns:
        Symmetric matrix ``L`` of shape ``(n_dof, n_dof)`` in henries.
    """
    K_basis = jnp.asarray(K_basis)
    quad_points = jnp.asarray(quad_points)
    quad_weights = jnp.asarray(quad_weights)
    r = quad_points[:, None, :] - quad_points[None, :, :]
    dist = jnp.linalg.norm(r, axis=-1)
    if adaptive_self_reg:
        delta_i = _SELF_REG_COEFF * jnp.sqrt(quad_weights)
        delta_pair = 0.5 * (delta_i[:, None] + delta_i[None, :])
        dist_reg = jnp.sqrt(dist**2 + delta_pair**2)
    else:
        dist_reg = jnp.sqrt(dist**2 + delta_reg**2)
    dot = jnp.einsum("iax,jbx->ijab", K_basis, K_basis)
    kernel = dot / dist_reg[..., None, None]
    wi = quad_weights[:, None, None, None]
    wj = quad_weights[None, :, None, None]
    L = MU0_OVER_4PI * jnp.sum(kernel * wi * wj, axis=(0, 1))
    return L


def shell_inductance_matrix_blockwise(
    K_per_puck: List[np.ndarray],
    pts_per_puck: List[np.ndarray],
    weights_per_puck: List[np.ndarray],
    dof_offsets: List[int],
    n_dof_total: int,
    delta_reg: float = 1e-8,
    adaptive_self_reg: bool = False,
    exact_disc_faces: bool = False,
    disc_bases: "list | None" = None,
    disc_Rts: "list | None" = None,
    disc_centers_axes: "list | None" = None,
    n_radial_disc: int = 32,
) -> np.ndarray:
    """
    Blockwise shell inductance matrix assembly.

    Exploits the block-diagonal sparsity of K_basis: each puck contributes
    to its own quad rows and DOF columns only.  Iterates over all unique
    puck pairs ``(i, j)`` with ``j >= i``, computing the block
    ``L[d_i:d_i+nd_i, d_j:d_j+nd_j]`` from the small per-puck K arrays.

    Peak memory is ``O(nq_per^2 * nd_per^2)`` per block instead of
    ``O(nq_total^2 * nd_total^2)`` for the monolithic version.

    Args:
        K_per_puck: List of ``(nq_p, nd_p, 3)`` arrays (global frame).
        pts_per_puck: List of ``(nq_p, 3)`` arrays (global positions).
        weights_per_puck: List of ``(nq_p,)`` arrays.
        dof_offsets: Starting DOF index for each puck.
        n_dof_total: Total number of DOFs across all pucks.
        delta_reg: Legacy uniform regularization for the weak singularity.
            Only used when ``adaptive_self_reg`` is ``False``.
        adaptive_self_reg: Opt-in flag (default ``False`` for backward
            compatibility).  When ``True``, use the analytic flat-disc
            per-quadrature-cell regularization
            :math:`\\delta_i = (3\\pi^{3/2}/8)\\sqrt{w_i}` (see
            :data:`_SELF_REG_COEFF` and
            :func:`shell_inductance_matrix_pure` for the derivation).
            When ``False``, fall back to the legacy uniform ``delta_reg``.
        exact_disc_faces: Opt-in flag (default ``False``).  When
            ``True``, overwrite every **intra-puck** self-block entry
            (the full ``(nd_i, nd_i)`` puck ``i`` self-block, covering
            all nine face-pair sub-blocks: disc-disc, disc-side-wall
            and side-side-wall) with the semi-analytic, exact-up-to-
            quadrature values computed by
            :func:`simsopt.field.disc_self_inductance.assemble_puck_self_L`.
            Off-diagonal inter-puck blocks (mutual inductance) are
            preserved byte-for-byte from the numerical quadrature,
            even when the pucks happen to be coaxial -- the
            semi-analytic path handles intra-puck self terms only;
            see Phase 3 of ``plans/complete-puck-self-inductance`` for
            an optional extension to coaxial mutuals.  Requires
            ``disc_bases``, ``disc_Rts``, and ``disc_centers_axes``.
            With the full 9-sub-block replacement, the puck self-block
            stays **positive semi-definite** (the property broken by
            the historical disc-only partial replacement), so the
            downstream eigenvalue-floor solver no longer amplifies
            indefinite directions and the Smythe / interior-field
            benchmarks in ``tests/field/test_passive_bulks_scale.py``
            match theory within the documented tolerances.
        disc_bases: Optional ``list`` of
            :class:`~simsopt.field.puck_basis.PuckBasisData` (one per
            puck) giving the Fourier-Zernike/Fourier-Chebyshev basis
            and DOF names.  Only consulted when
            ``exact_disc_faces=True``.
        disc_Rts: Optional ``list`` of ``(R, t)`` tuples (one per puck).
            Only consulted when ``exact_disc_faces=True``.
        disc_centers_axes: Optional ``list`` of
            ``(center: (3,), axis: (3,))`` tuples (one per puck) in the
            global frame.  Retained for forward-compatibility with the
            Phase 3 multi-puck coaxial-mutual path; currently not
            consulted by the default Phase 2 overlay (only diagonal
            self-blocks are replaced) but still validated for length
            consistency.
        n_radial_disc: Number of Gauss-Legendre / Gauss-Laguerre nodes
            used by the semi-analytic puck-self routines (disc-disc,
            disc-side corner, and side-side axial quadratures).  Only
            consulted when ``exact_disc_faces=True``.

    Returns:
        Symmetric ``L`` of shape ``(n_dof_total, n_dof_total)``.
    """
    n_pucks = len(K_per_puck)
    if n_pucks == 0:
        return np.zeros((n_dof_total, n_dof_total))

    shapes = [K.shape for K in K_per_puck]
    same_shape = len(set(shapes)) == 1
    if same_shape:
        # The JAX-batched path is O(n_pucks^2) in device memory because it
        # stacks all pair kernels at once; if the device runs out of
        # memory (``RuntimeError`` from XLA, ``ValueError`` from JAX's
        # shape/dtype checks, or ``ImportError`` if JAX / jaxlib is
        # mis-installed) we fall back to the NumPy blockwise batched
        # assembly, which streams pair kernels puck-by-puck.  A broad
        # ``except Exception:`` here would also swallow real logic errors
        # (e.g. wrong array ranks), so we catch only the narrow set of
        # runtime/environmental errors and surface the fallback via
        # :mod:`warnings` so regressions can't hide silently.
        try:
            L = _shell_inductance_matrix_blockwise_batched_jax(
                K_per_puck,
                pts_per_puck,
                weights_per_puck,
                dof_offsets,
                n_dof_total,
                delta_reg,
                adaptive_self_reg=adaptive_self_reg,
            )
        except (ImportError, RuntimeError, ValueError, MemoryError) as exc:
            warnings.warn(
                "shell_inductance_matrix_blockwise: JAX-batched assembly "
                f"failed with {type(exc).__name__}: {exc!s}.  Falling "
                "back to the NumPy blockwise batched assembly.  "
                "Silent fallbacks can mask performance or correctness "
                "regressions; see repro details above.",
                RuntimeWarning,
                stacklevel=2,
            )
            L = _shell_inductance_matrix_blockwise_batched(
                K_per_puck,
                pts_per_puck,
                weights_per_puck,
                dof_offsets,
                n_dof_total,
                delta_reg,
                adaptive_self_reg=adaptive_self_reg,
            )
    else:
        L = np.zeros((n_dof_total, n_dof_total))
        for i in range(n_pucks):
            K_i = K_per_puck[i]
            pts_i = pts_per_puck[i]
            w_i = weights_per_puck[i]
            nd_i = K_i.shape[1]
            d0_i = dof_offsets[i]
            for j in range(i, n_pucks):
                K_j = K_per_puck[j]
                pts_j = pts_per_puck[j]
                w_j = weights_per_puck[j]
                nd_j = K_j.shape[1]
                d0_j = dof_offsets[j]
                r = pts_i[:, None, :] - pts_j[None, :, :]
                if adaptive_self_reg:
                    delta_i = _SELF_REG_COEFF * np.sqrt(w_i)
                    delta_j = _SELF_REG_COEFF * np.sqrt(w_j)
                    delta_pair = 0.5 * (delta_i[:, None] + delta_j[None, :])
                    dist = np.sqrt(np.sum(r**2, axis=-1) + delta_pair**2)
                else:
                    dist = np.sqrt(np.sum(r**2, axis=-1) + delta_reg**2)
                dot = np.einsum("iax,jbx->ijab", K_i, K_j)
                kernel = dot / dist[..., None, None]
                block = MU0_OVER_4PI * np.einsum("ijab,i,j->ab", kernel, w_i, w_j)
                if i == j:
                    # Diagonal blocks must be symmetric analytically
                    # (the kernel ``1/|r_i - r_j|`` is symmetric in
                    # ``(a, b)`` for ``i == j``).  Quadrature
                    # cancellation can leak O(eps) asymmetry, which
                    # ``np.linalg.eigh`` will then project onto
                    # spurious imaginary eigenvalues.  Explicit
                    # symmetrization is a cheap defensive guard.
                    L[d0_i : d0_i + nd_i, d0_j : d0_j + nd_j] = 0.5 * (block + block.T)
                else:
                    L[d0_i : d0_i + nd_i, d0_j : d0_j + nd_j] = block
                    L[d0_j : d0_j + nd_j, d0_i : d0_i + nd_i] = block.T

    # ------------------------------------------------------------------
    # Opt-in exact intra-puck self-block overlay.  Overwrites the full
    # puck-i self-block (all 9 face-pair sub-blocks) with the appendix-
    # consistent semi-analytic assembly in
    # :func:`assemble_puck_self_L`.  Off-diagonal inter-puck blocks
    # (mutual inductance) are preserved byte-for-byte from the
    # numerical quadrature regardless of coaxiality (Phase 3 follow-up
    # covers coaxial mutuals, see
    # ``plans/complete-puck-self-inductance``).  Default is off so
    # backward compatibility is guaranteed.
    # ------------------------------------------------------------------
    if exact_disc_faces:
        if disc_bases is None or disc_Rts is None or disc_centers_axes is None:
            raise ValueError(
                "exact_disc_faces=True requires disc_bases, disc_Rts, "
                "and disc_centers_axes to be provided (one entry per puck)."
            )
        if not (len(disc_bases) == len(disc_Rts) == len(disc_centers_axes) == n_pucks):
            raise ValueError(
                "exact_disc_faces=True: disc_bases, disc_Rts, "
                "disc_centers_axes must all have length n_pucks = "
                f"{n_pucks}."
            )

        # Local import so the disc module is only required when the
        # opt-in path is exercised; keeps the default import path
        # narrow and avoids a circular-import risk with puck_basis.
        from .disc_self_inductance import assemble_puck_self_L

        for i in range(n_pucks):
            basis_i = disc_bases[i]
            R_i, t_i = disc_Rts[i]
            puck_block = assemble_puck_self_L(
                basis_i,
                R_i,
                t_i,
                n_radial=n_radial_disc,
            )
            d0 = dof_offsets[i]
            nd = puck_block.shape[0]
            # Overwrite every recognized (disk or side) DOF entry of
            # the puck self-block.  NaN entries (unrecognized DOF
            # names, not produced by the standard basis) fall back to
            # the numerical value already in L.
            mask = ~np.isnan(puck_block)
            sub = L[d0 : d0 + nd, d0 : d0 + nd].copy()
            sub[mask] = puck_block[mask]
            # Analytical symmetry guard.
            L[d0 : d0 + nd, d0 : d0 + nd] = 0.5 * (sub + sub.T)
    return L


def _shell_inductance_matrix_blockwise_batched(
    K_per_puck: List[np.ndarray],
    pts_per_puck: List[np.ndarray],
    weights_per_puck: List[np.ndarray],
    dof_offsets: List[int],
    n_dof_total: int,
    delta_reg: float = 1e-8,
    adaptive_self_reg: bool = False,
) -> np.ndarray:
    """
    Vectorized assembly when every puck has identical ``(nq, nd, 3)`` for ``K``.
    Processes upper-triangular puck pairs in batches of ``_PAIR_BATCH_SIZE``.

    Args:
        K_per_puck: List of ``(nq, nd, 3)`` arrays (identical shape required).
        pts_per_puck: List of ``(nq, 3)`` arrays.
        weights_per_puck: List of ``(nq,)`` arrays.
        dof_offsets: Starting DOF index for each puck.
        n_dof_total: Total number of DOFs across all pucks.
        delta_reg: Legacy uniform regularization (only used when
            ``adaptive_self_reg=False``).
        adaptive_self_reg: See :func:`shell_inductance_matrix_blockwise`.

    Returns:
        Symmetric ``L`` of shape ``(n_dof_total, n_dof_total)``.
    """
    n_pucks = len(K_per_puck)
    K = np.stack(K_per_puck, axis=0)
    pts = np.stack(pts_per_puck, axis=0)
    w = np.stack(weights_per_puck, axis=0)
    _nq, nd = K.shape[1], K.shape[2]

    pairs: List[Tuple[int, int]] = [
        (i, j) for i in range(n_pucks) for j in range(i, n_pucks)
    ]
    L = np.zeros((n_dof_total, n_dof_total))
    n_pairs = len(pairs)

    for start in range(0, n_pairs, _PAIR_BATCH_SIZE):
        batch = pairs[start : start + _PAIR_BATCH_SIZE]
        bsz = len(batch)
        i_idx = np.array([p[0] for p in batch], dtype=np.intp)
        j_idx = np.array([p[1] for p in batch], dtype=np.intp)
        Ki = K[i_idx]
        Kj = K[j_idx]
        pts_i = pts[i_idx]
        pts_j = pts[j_idx]
        w_i = w[i_idx]
        w_j = w[j_idx]
        r = pts_i[:, :, None, :] - pts_j[:, None, :, :]
        if adaptive_self_reg:
            delta_i = _SELF_REG_COEFF * np.sqrt(w_i)
            delta_j = _SELF_REG_COEFF * np.sqrt(w_j)
            delta_pair = 0.5 * (delta_i[:, :, None] + delta_j[:, None, :])
            dist = np.sqrt(np.sum(r**2, axis=-1) + delta_pair**2)
        else:
            dist = np.sqrt(np.sum(r**2, axis=-1) + delta_reg**2)
        # Batch index ``B``; ``a,b`` index DOFs (distinct from batch ``B``).
        dot = np.einsum("Biax,Bjbx->Bijab", Ki, Kj)
        kernel = dot / dist[..., None, None]
        blocks = MU0_OVER_4PI * np.einsum("Bijab,Bi,Bj->Bab", kernel, w_i, w_j)
        for bi in range(bsz):
            i, j = int(i_idx[bi]), int(j_idx[bi])
            d0_i = dof_offsets[i]
            d0_j = dof_offsets[j]
            block = blocks[bi]
            if i == j:
                # Symmetrize diagonal blocks -- see the matching guard in
                # :func:`shell_inductance_matrix_blockwise` for rationale.
                L[d0_i : d0_i + nd, d0_j : d0_j + nd] = 0.5 * (block + block.T)
            else:
                L[d0_i : d0_i + nd, d0_j : d0_j + nd] = block
                L[d0_j : d0_j + nd, d0_i : d0_i + nd] = block.T
    return L


def shell_inductance_matrix_jax_blockwise(
    K_per_puck: List[np.ndarray],
    pts_per_puck: List[np.ndarray],
    weights_per_puck: List[np.ndarray],
    dof_offsets: List[int],
    n_dof_total: int,
    delta_reg: float = 1e-8,
    adaptive_self_reg: bool = False,
) -> np.ndarray:
    """Same as :func:`shell_inductance_matrix_blockwise` but always uses the JAX path.

    Raises if JAX is unavailable or shapes are not identical per puck.

    Args:
        adaptive_self_reg: See :func:`shell_inductance_matrix_blockwise`.
    """
    shapes = [K.shape for K in K_per_puck]
    if len(set(shapes)) != 1:
        raise ValueError(
            "shell_inductance_matrix_jax_blockwise requires identical puck shapes"
        )
    return _shell_inductance_matrix_blockwise_batched_jax(
        K_per_puck,
        pts_per_puck,
        weights_per_puck,
        dof_offsets,
        n_dof_total,
        delta_reg,
        adaptive_self_reg=adaptive_self_reg,
    )


def _shell_inductance_matrix_blockwise_batched_jax(
    K_per_puck: List[np.ndarray],
    pts_per_puck: List[np.ndarray],
    weights_per_puck: List[np.ndarray],
    dof_offsets: List[int],
    n_dof_total: int,
    delta_reg: float = 1e-8,
    adaptive_self_reg: bool = False,
) -> np.ndarray:
    """Same as :func:`_shell_inductance_matrix_blockwise_batched` with XLA-accelerated batches.

    Args:
        adaptive_self_reg: See :func:`shell_inductance_matrix_blockwise`.
    """
    n_pucks = len(K_per_puck)
    K = np.stack(K_per_puck, axis=0)
    pts = np.stack(pts_per_puck, axis=0)
    w = np.stack(weights_per_puck, axis=0)
    nd = K.shape[2]

    pairs: List[Tuple[int, int]] = [
        (i, j) for i in range(n_pucks) for j in range(i, n_pucks)
    ]
    L = np.zeros((n_dof_total, n_dof_total))
    n_pairs = len(pairs)

    dreg = jnp.asarray(float(delta_reg))

    for start in range(0, n_pairs, _PAIR_BATCH_SIZE):
        batch = pairs[start : start + _PAIR_BATCH_SIZE]
        bsz = len(batch)
        i_idx = np.array([p[0] for p in batch], dtype=np.intp)
        j_idx = np.array([p[1] for p in batch], dtype=np.intp)
        Ki = jnp.asarray(K[i_idx])
        Kj = jnp.asarray(K[j_idx])
        pts_i = jnp.asarray(pts[i_idx])
        pts_j = jnp.asarray(pts[j_idx])
        w_i = jnp.asarray(w[i_idx])
        w_j = jnp.asarray(w[j_idx])
        blocks = np.asarray(
            _jax_pair_batch_blocks(
                Ki,
                Kj,
                pts_i,
                pts_j,
                w_i,
                w_j,
                dreg,
                adaptive_self_reg=adaptive_self_reg,
            ),
        )
        for bi in range(bsz):
            i, j = int(i_idx[bi]), int(j_idx[bi])
            d0_i = dof_offsets[i]
            d0_j = dof_offsets[j]
            block = blocks[bi]
            L[d0_i : d0_i + nd, d0_j : d0_j + nd] = block
            if i != j:
                L[d0_j : d0_j + nd, d0_i : d0_i + nd] = block.T
    return L


def null_space_projection_matrix(
    L: np.ndarray,
    threshold: float = 1e-10,
) -> np.ndarray:
    r"""
    Projection matrix ``Q`` whose columns span the well-conditioned subspace
    of the symmetric inductance matrix ``L``.

    Eigendecompose ``L = V \Lambda V^T`` and keep eigenvectors whose
    eigenvalue exceeds ``threshold * max(|\lambda|)``.  The reduced system
    ``L_r = Q^T L Q`` is then guaranteed SPD so Cholesky succeeds without
    jitter.

    This replaces the single-pin ``gauge_projection_matrix`` which only
    removed one null mode and left the remaining patch-local constant modes
    (approximately 3 per puck) unhandled.

    Args:
        L: Symmetric matrix of shape ``(n_dof, n_dof)``.
        threshold: Relative eigenvalue cutoff.

    Returns:
        ``Q`` of shape ``(n_dof, n_reduced)`` with ``n_reduced <= n_dof``.
    """
    L = np.asarray(L)
    lam, V = np.linalg.eigh(L)
    max_lam = np.max(np.abs(lam))
    if max_lam < 1e-30:
        return V
    good = lam > threshold * max_lam
    return V[:, good]


def shell_loading_vector_pure(
    phi_values: jnp.ndarray,
    quad_weights: jnp.ndarray,
    B_normal: jnp.ndarray,
) -> jnp.ndarray:
    """
    Load vector :math:`f_a = -\\int_\\Sigma \\Phi_a\\,B_n^{TF}\\,dS` (Eq. 64).

    The sign convention is set by the ideal-diamagnet weak form: requiring
    the *total* normal field on the puck surface to vanish,
    :math:`B_n^{ind}=-B_n^{TF}`, combined with the stationarity of the
    quadratic energy functional for the current potential, yields
    :math:`\\mathbf{L}\\boldsymbol{\\beta} = -\\int \\Phi\\,B_n^{TF}`.
    A diagnostic on a thin disc in :math:`+B_0\\hat{z}` confirms the
    induced dipole points along :math:`-\\hat{z}` (Lenz's law) with this
    sign; the unsigned version of the integral reverses the induced
    moment and yields :math:`|B_{tot}|/|B_{TF}|\\!\\approx\\!1.05` inside
    the puck interior.
    Every downstream caller (``_beta_from_tf_body``,
    ``_beta_from_bn_body``, ``_beta_eigenfloor_from_bn_body``,
    ``_beta_from_tf_eigenfloor_body``, ``_B_eval_full_body``,
    ``_vjp_tf_only_analytic``) picks this sign up automatically.

    Args:
        phi_values: shape ``(n_quad, n_dof)``.
        quad_weights: shape ``(n_quad,)``.
        B_normal: shape ``(n_quad,)``, background (TF) normal field.

    Returns:
        ``f`` of shape ``(n_dof,)``.
    """
    phi_values = jnp.asarray(phi_values)
    w = jnp.asarray(quad_weights)
    Bn = jnp.asarray(B_normal)
    return -jnp.sum(phi_values * (w * Bn)[:, None], axis=0)


def shell_solve_linear_pure(
    L: jnp.ndarray,
    f: jnp.ndarray,
    jitter: float = 1e-10,
) -> jnp.ndarray:
    """Solve ``L @ beta = f`` with Cholesky (symmetric positive definite)."""
    L = jnp.asarray(L)
    f = jnp.asarray(f)
    Ls = L + jitter * jnp.eye(L.shape[0])
    C = jnp.linalg.cholesky(Ls)
    y = jscp.linalg.solve_triangular(C, f, lower=True)
    return jscp.linalg.solve_triangular(C.T, y, lower=False)


def shell_solve_eigenfloor_pure(
    L: jnp.ndarray,
    f: jnp.ndarray,
    threshold: float = 1e-10,
    jitter: float = 1e-10,
) -> jnp.ndarray:
    """Solve ``L @ beta \\approx f`` with eigenvalue-floor regularization.

    Eigendecompose symmetric ``L``, raise eigenvalues below
    ``threshold * max(|\\lambda|)`` to that floor, reconstruct a
    regularized matrix, add ``jitter * I``, and solve with a general
    linear solve.  Output shape is always ``(n_dof,)`` (JIT-stable,
    unlike a variable-size null-space projection matrix).

    Args:
        L: Symmetric inductance matrix, shape ``(n_dof, n_dof)``.
        f: Load vector, shape ``(n_dof,)``.
        threshold: Relative floor for small eigenvalues (same order as
            :func:`null_space_projection_matrix`).
        jitter: Tiny diagonal added before solve for numerical stability.

    Returns:
        ``beta`` of shape ``(n_dof,)``.
    """
    L = jnp.asarray(L)
    f = jnp.asarray(f)
    lam, V = jnp.linalg.eigh(L)
    max_abs = jnp.max(jnp.abs(lam))
    floor = threshold * max_abs
    lam_floor = jnp.maximum(lam, floor)
    L_reg = (V * lam_floor[None, :]) @ V.T
    n = L.shape[0]
    L_reg = L_reg + jitter * jnp.eye(n, dtype=L.dtype)
    return jnp.linalg.solve(L_reg, f)


def shell_biot_savart_pure(
    K_basis: jnp.ndarray,
    quad_points: jnp.ndarray,
    quad_weights: jnp.ndarray,
    beta: jnp.ndarray,
    eval_points: jnp.ndarray,
    eps: float = 1e-8,
) -> jnp.ndarray:
    """
    Magnetic field from sheet currents :math:`\\mathbf{K} = \\sum_a \\beta_a \\mathbf{K}_a`:

    .. math::

        \\mathbf{B}(\\mathbf{x}) = \\frac{\\mu_0}{4\\pi} \\int_\\Sigma
        \\frac{\\mathbf{K}(\\mathbf{x}') \\times (\\mathbf{x}-\\mathbf{x}')}{|\\mathbf{x}-\\mathbf{x}'|^3}
        \\, dS' .

    Args:
        K_basis: ``(n_quad, n_dof, 3)``.
        quad_points: ``(n_quad, 3)``.
        quad_weights: ``(n_quad,)``.
        beta: ``(n_dof,)``.
        eval_points: ``(n_eval, 3)``.

    Returns:
        ``B`` of shape ``(n_eval, 3)`` in Tesla.
    """
    K_basis = jnp.asarray(K_basis)
    quad_points = jnp.asarray(quad_points)
    w = jnp.asarray(quad_weights)
    beta = jnp.asarray(beta)
    eval_points = jnp.asarray(eval_points)
    K = jnp.sum(K_basis * beta[None, :, None], axis=1)  # (n_quad, 3)

    def B_at(x: jnp.ndarray) -> jnp.ndarray:
        r = x[None, :] - quad_points
        rn = jnp.sqrt(jnp.sum(r**2, axis=-1) + eps**2)
        integrand = jnp.cross(K, r) / (rn[:, None] ** 3)
        return MU0_OVER_4PI * jnp.sum(integrand * w[:, None], axis=0)

    return jax.vmap(B_at)(eval_points)


def project_reduced_system(
    L: jnp.ndarray,
    f: jnp.ndarray,
    Q: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Reduced SPD system ``Q^T L Q @ alpha = Q^T f``."""
    Lr = Q.T @ L @ Q
    fr = Q.T @ f
    return Lr, fr


def expand_beta_reduced(alpha: jnp.ndarray, Q: jnp.ndarray) -> jnp.ndarray:
    """``beta = Q @ alpha``."""
    return Q @ alpha
