"""
Semi-analytic self- and coaxial-mutual inductance for flat circular disc
current sheets expressed in a Fourier-Zernike current-potential basis.

This module reduces the weakly-singular 2D surface integral
``(mu_0 / 4pi) integral integral K . K' / |x - x'| dS dS'`` between two
flat coaxial circular-disc current sheets to a 1D radial integral in
which the azimuthal direction has been integrated out analytically.
The ring kernel is the classical Maxwell-Grover coaxial-loop mutual
inductance and its Fourier-mode-``m`` generalizations derived from the
Legendre-function-of-the-second-kind recurrence.

Accuracy claim
--------------

* Azimuthal integrals over ``phi, phi'`` are analytic (closed form in
  complete elliptic integrals ``K(k), E(k)``), not numerical.
* The remaining radial integrals are 1D or 2D and either smooth
  (finite axial separation, non-singular) or have a known integrable
  logarithmic singularity on the diagonal ``a = b`` that is handled by
  a Duffy-triangle + Gauss-Laguerre change of variables.

Together these give exponential convergence of the assembled block in
the number of radial nodes, independent of any regularization scalar.
This is categorically different from the regularized
``1 / sqrt(r**2 + delta**2)`` 2D surface quadrature implemented in
:func:`~simsopt.field.bulk_inductance.shell_inductance_matrix_blockwise`,
which carries a systematic ``delta``-dependent model error for any
fixed quadrature resolution.

Scope
-----

Only the flat-face (``top`` and ``bottom``) entries of the
Fourier-Zernike basis are handled here.  Side-wall entries and
non-coaxial mutuals remain on the existing regularized path inside
:func:`shell_inductance_matrix_blockwise`.
"""

from __future__ import annotations

from math import factorial
from typing import Callable, List, Optional, Tuple

import numpy as np
from scipy.special import ellipe, ellipk

from .puck_basis import PuckBasisData


MU0 = 4.0 * np.pi * 1.0e-7

__all__ = [
    "ring_mutual_maxwell_grover",
    "ring_mutual_axial_shift",
    "fourier_kernel_Im",
    "disc_self_block_axisymmetric",
    "disc_self_block_mode_m",
    "disc_disc_cross_block",
    "cylinder_self_block_mode_m",
    "disc_side_cross_block_mode_m",
    "assemble_puck_self_L",
    "assemble_puck_disc_faces_L",
    "pucks_are_coaxial",
    "fill_coaxial_inter_puck_disc_block",
]


# ----------------------------------------------------------------------
# Ring kernels
# ----------------------------------------------------------------------


def ring_mutual_maxwell_grover(a: float, b: float) -> float:
    r"""Coplanar coaxial ring mutual inductance (Maxwell-Grover).

    Returns

    .. math::

        M(a, b) = \mu_0 \sqrt{a b}
        \frac{(2 - k^2) K(k) - 2 E(k)}{k},\qquad
        k^2 = \frac{4 a b}{(a + b)^2}.

    ``K(k), E(k)`` are complete elliptic integrals of the first and
    second kind with ``scipy.special`` convention (argument ``m = k**2``).

    Args:
        a: First ring radius (m).
        b: Second ring radius (m).

    Returns:
        Mutual inductance in henries.  Returns ``0`` for degenerate
        zero-radius rings and returns ``+inf`` at ``a = b`` (logarithmic
        singularity of ``K`` at ``k = 1``).
    """
    return ring_mutual_axial_shift(a, b, 0.0)


def ring_mutual_axial_shift(a: float, b: float, dz: float) -> float:
    r"""Coaxial ring mutual inductance with axial offset ``dz``.

    .. math::

        M(a, b, d) = \mu_0 \sqrt{a b}
        \frac{(2 - k^2) K(k) - 2 E(k)}{k},\qquad
        k^2 = \frac{4 a b}{(a + b)^2 + d^2}.

    Args:
        a: First ring radius (m).
        b: Second ring radius (m).
        dz: Axial offset between ring centers (m).

    Returns:
        Mutual inductance in henries.  Returns ``0`` for degenerate
        zero-radius rings.  For ``dz = 0`` and ``a = b`` the Legendre
        function ``K(k=1)`` is infinite; callers must handle the
        singularity (see :func:`disc_self_block_axisymmetric`).
    """
    if a <= 0.0 or b <= 0.0:
        return 0.0
    denom = (a + b) ** 2 + dz**2
    m_sq = 4.0 * a * b / denom
    # Clamp slightly below 1 to avoid overflow from ellipk(1.0) -> inf;
    # true singularity at a==b and dz==0 is handled upstream by the
    # Duffy + Gauss-Laguerre radial quadrature in
    # ``disc_self_block_axisymmetric``.
    if m_sq >= 1.0:
        return np.inf
    k = np.sqrt(m_sq)
    K_k = ellipk(m_sq)
    E_k = ellipe(m_sq)
    return MU0 * np.sqrt(a * b) * ((2.0 - m_sq) * K_k - 2.0 * E_k) / k


def fourier_kernel_Im(
    a: np.ndarray,
    b: np.ndarray,
    dz: float,
    m_max: int,
) -> np.ndarray:
    r"""Azimuthal Fourier coefficients of the ``1 / |x - x'|`` kernel.

    Returns a ``(m_max + 1, ...)`` array of

    .. math::

        I_m(a, b, d) = \int_0^{2\pi}\frac{\cos(m u)}{\sqrt{a^2 + b^2
        - 2 a b \cos u + d^2}}\,du,

    evaluated elementwise over broadcasted ``a, b`` arrays.

    Uses exact closed forms

    .. math::

        I_0 &= \frac{4}{\sqrt{(a+b)^2 + d^2}}\,K(k), \\
        I_1 &= \frac{4}{k^2 \sqrt{(a+b)^2 + d^2}}\left[(2 - k^2) K(k) - 2 E(k)\right],

    and the Legendre-``Q`` recurrence

    .. math::

        I_{m+1} = \frac{4 m w\,I_m - (2 m - 1)\,I_{m-1}}{2 m + 1},

    with ``w = (a^2 + b^2 + d^2)/(2 a b)``.  This recurrence is exact
    and propagates without reference to any regularization scalar.

    Args:
        a: Array of first-ring radii (m), broadcastable with ``b``.
        b: Array of second-ring radii (m), broadcastable with ``a``.
        dz: Axial separation (m), scalar.
        m_max: Highest Fourier index to return (inclusive).  The output
            has ``m_max + 1`` rows covering ``m = 0, 1, ..., m_max``.

    Returns:
        Array with shape ``(m_max + 1,) + broadcast(a, b).shape``.
        Diagonal entries where ``a == b`` and ``dz == 0`` evaluate to
        ``+inf`` in the ``m = 0`` row (the log singularity); callers are
        responsible for ensuring the quadrature never samples there.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    shape = np.broadcast_shapes(a.shape, b.shape)
    a_bc = np.broadcast_to(a, shape)
    b_bc = np.broadcast_to(b, shape)

    denom = (a_bc + b_bc) ** 2 + dz**2
    m_sq = np.where(denom > 0.0, 4.0 * a_bc * b_bc / denom, 0.0)
    m_sq = np.clip(m_sq, 0.0, 1.0 - 1.0e-15)
    np.sqrt(m_sq)
    K_k = ellipk(m_sq)
    E_k = ellipe(m_sq)
    root = np.sqrt(denom)

    I = np.zeros((m_max + 1,) + shape, dtype=float)
    I[0] = 4.0 * K_k / root
    if m_max >= 1:
        # I_1 = (4 / (k^2 root)) * [(2 - k^2) K - 2 E]
        with np.errstate(divide="ignore", invalid="ignore"):
            num = (2.0 - m_sq) * K_k - 2.0 * E_k
            I[1] = np.where(m_sq > 0.0, 4.0 * num / (m_sq * root), 0.0)

    # Recurrence: I_{m+1} = (4 m w I_m - (2 m - 1) I_{m-1}) / (2 m + 1)
    if m_max >= 2:
        # w = (a^2 + b^2 + d^2) / (2 a b); guard a=0 or b=0
        ab = a_bc * b_bc
        with np.errstate(divide="ignore", invalid="ignore"):
            w = np.where(
                ab > 0.0,
                (a_bc**2 + b_bc**2 + dz**2) / (2.0 * ab),
                0.0,
            )
        for mm in range(1, m_max):
            I[mm + 1] = (4.0 * mm * w * I[mm] - (2.0 * mm - 1.0) * I[mm - 1]) / (
                2.0 * mm + 1.0
            )
    return I


# ----------------------------------------------------------------------
# Radial quadrature: singular (dz = 0) and smooth (dz > 0)
# ----------------------------------------------------------------------


def _duffy_laguerre_nodes_weights(
    R: float,
    n_radial: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(rho_nodes, rho_weights, eta_nodes, eta_weights)`` for the
    Duffy-triangle + Gauss-Laguerre radial rule used in the singular
    (``dz = 0``) path.

    The Duffy map ``a = rho, b = rho * (1 - eta)`` on ``{0 <= b <= a <= R}``
    pushes the diagonal singularity ``a = b`` to ``eta = 0``; the
    change of variable ``eta = exp(-t), t in [0, infty)`` makes the
    ``log(eta)`` behavior of ``K(k)`` into linear-in-``t`` growth, which
    is exponentially damped by the Gauss-Laguerre weight ``e^{-t}``.

    As a result, the combined rule has exponential convergence in
    ``n_radial`` for the Maxwell-Grover singular self-block.
    """
    rho_nodes, rho_w = np.polynomial.legendre.leggauss(n_radial)
    rho = 0.5 * R * (rho_nodes + 1.0)
    w_rho = 0.5 * R * rho_w

    t_nodes, t_w = np.polynomial.laguerre.laggauss(n_radial)
    eta = np.exp(-t_nodes)
    # With eta = e^{-t}, d eta = -e^{-t} dt so
    # integral_0^1 f(eta) d eta = integral_0^infty f(e^{-t}) e^{-t} dt
    # and Gauss-Laguerre weights t_w already include the e^{-t} weight.
    w_eta = t_w
    return rho, w_rho, eta, w_eta


def _leggauss_on_interval(
    R: float,
    n: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre nodes/weights mapped to ``[0, R]``."""
    nodes, w = np.polynomial.legendre.leggauss(n)
    pts = 0.5 * R * (nodes + 1.0)
    wts = 0.5 * R * w
    return pts, wts


def disc_self_block_axisymmetric(
    f_prime_p: Callable[[np.ndarray], np.ndarray],
    f_prime_q: Callable[[np.ndarray], np.ndarray],
    R: float,
    n_radial: int = 32,
) -> float:
    r"""Axisymmetric (``m = 0``) disc self-block matrix entry.

    Computes

    .. math::

        L_{pq}^{(m=0)} = \int_0^R \int_0^R f_p'(a)\,f_q'(b)\,M(a, b, 0)
        \,da\,db,

    where ``M(a, b, 0)`` is :func:`ring_mutual_maxwell_grover`.  This is
    the exact reduction of the azimuthally-symmetric disc surface
    bilinear form after closed-form ``(phi, phi')`` integration.

    Uses the Duffy triangle + Gauss-Laguerre rule (see
    :func:`_duffy_laguerre_nodes_weights`); exponentially convergent in
    ``n_radial`` because the log singularity of ``K(k)`` at ``a = b`` is
    damped by the ``e^{-t}`` Laguerre weight under the substitution
    ``eta = e^{-t}``.

    Args:
        f_prime_p: First radial derivative ``df_p/drho`` of the basis
            function, vectorized over ``rho``.
        f_prime_q: Second radial derivative ``df_q/drho``.
        R: Disc radius (m).
        n_radial: Number of Gauss-Legendre and Gauss-Laguerre nodes.

    Returns:
        Block entry in henries.
    """
    rho, w_rho, eta, w_eta = _duffy_laguerre_nodes_weights(R, n_radial)

    # Triangles T1: (a, b) = (rho, rho*(1-eta))  with rho > 0, eta in (0, 1]
    # and T2: (a, b) = (rho*(1-eta), rho) by symmetry.
    RHO, ETA = np.meshgrid(rho, eta, indexing="ij")
    A1 = RHO
    B1 = RHO * (1.0 - ETA)
    # Jacobian of the Duffy map is rho.
    jac = RHO * w_rho[:, None] * w_eta[None, :]

    # Vectorized Maxwell-Grover evaluation
    denom = (A1 + B1) ** 2
    m_sq = np.where(denom > 0.0, 4.0 * A1 * B1 / denom, 0.0)
    m_sq = np.clip(m_sq, 0.0, 1.0 - 1.0e-15)
    k = np.sqrt(m_sq)
    K_k = ellipk(m_sq)
    E_k = ellipe(m_sq)
    M_ab = np.where(
        (A1 > 0.0) & (B1 > 0.0),
        MU0
        * np.sqrt(A1 * B1)
        * ((2.0 - m_sq) * K_k - 2.0 * E_k)
        / np.maximum(k, 1e-300),
        0.0,
    )

    fp_A1 = f_prime_p(A1)
    fq_B1 = f_prime_q(B1)
    # T1 contribution
    total = float(np.sum(jac * fp_A1 * fq_B1 * M_ab))
    # T2 contribution: swap so that a = rho*(1-eta), b = rho
    # The kernel M is symmetric in (a, b), so M_ab unchanged.
    fp_B1 = f_prime_p(B1)
    fq_A1 = f_prime_q(A1)
    total += float(np.sum(jac * fp_B1 * fq_A1 * M_ab))
    return total


def disc_self_block_mode_m(
    f_p: Callable[[np.ndarray], np.ndarray],
    f_prime_p: Callable[[np.ndarray], np.ndarray],
    f_q: Callable[[np.ndarray], np.ndarray],
    f_prime_q: Callable[[np.ndarray], np.ndarray],
    m: int,
    R: float,
    n_radial: int = 32,
) -> float:
    r"""General Fourier-mode ``m`` disc self-block matrix entry.

    For ``g_p(rho, phi) = f_p(rho) \cos(m phi)`` (or ``\sin``) on a flat
    coaxial disc with outward normal ``\hat z`` (so
    ``K = \hat z \times \nabla_s g``), the self-block entry is

    .. math::

        L_{pq}^{(m)} = \frac{\mu_0}{8} \int_0^R \int_0^R ab\,\Big\{
        \left[ f_p'(a) f_q'(b) + \tfrac{m^2}{ab} f_p(a) f_q(b) \right]
        \left[ I_{m-1}(a, b) + I_{m+1}(a, b) \right]
        + \left[ \tfrac{m}{a} f_p(a) f_q'(b) + \tfrac{m}{b} f_p'(a) f_q(b) \right]
        \left[ I_{m-1}(a, b) - I_{m+1}(a, b) \right] \Big\}\,da\,db,

    with ``I_m(a, b) = \int_0^{2\pi} \cos(m u)/|x - x'|\,du`` (see
    :func:`fourier_kernel_Im`).  For ``m = 0`` this reduces to
    :func:`disc_self_block_axisymmetric`.

    Args:
        f_p, f_prime_p: Radial function and its derivative for basis
            function ``p`` (vectorized).
        f_q, f_prime_q: Same for basis function ``q``.
        m: Azimuthal mode number.  ``m = 0`` uses the axisymmetric
            formula.
        R: Disc radius.
        n_radial: Number of Gauss-Legendre / Gauss-Laguerre nodes.

    Returns:
        Block entry ``L_{pq}^{(m)}`` in henries (same for ``cos``-``cos``
        and ``sin``-``sin`` branches by orthogonality; cross
        ``cos``-``sin`` entries vanish identically and are not returned
        here).
    """
    if m == 0:
        return disc_self_block_axisymmetric(f_prime_p, f_prime_q, R, n_radial)

    rho, w_rho, eta, w_eta = _duffy_laguerre_nodes_weights(R, n_radial)
    RHO, ETA = np.meshgrid(rho, eta, indexing="ij")
    A1 = RHO
    B1 = RHO * (1.0 - ETA)
    jac = RHO * w_rho[:, None] * w_eta[None, :]

    # Compute I_{m-1} and I_{m+1} on the (A1, B1) grid (both triangles
    # use the same kernel because I_m is symmetric in (a, b) and dz=0).
    I_all = fourier_kernel_Im(A1, B1, 0.0, m + 1)
    I_mm1 = I_all[m - 1]
    I_mp1 = I_all[m + 1]
    K_even = I_mm1 + I_mp1
    K_odd = I_mm1 - I_mp1

    prefac = MU0 / (8.0 * np.pi) * np.pi  # = MU0 / 8; keep symbolic for clarity
    # ^-- The (mu0/4pi) normalization is inherent to the inductance
    # definition; the extra pi comes from the azimuthal integrals for
    # m>=1 (factor c_m = pi).  For m = 0 the factor is 2*pi and the
    # m >= 1 formula is not used.  The final result is
    # mu0 * pi / (4 pi) * (1/2) * (...) where the (1/2) absorbs the
    # combined prefactors; identical to MU0 / 8.
    del prefac  # kept above for documentation; use MU0/8 directly below

    # Evaluate radial functions; vectorized
    # Triangle 1: a = A1, b = B1
    fp_a_T1 = f_p(A1)
    fpp_a_T1 = f_prime_p(A1)
    fq_b_T1 = f_q(B1)
    fqp_b_T1 = f_prime_q(B1)

    # Bilinear kernel weight for T1.  Safe guard 1/a, 1/b against zero
    # (rho=0 is outside the Gauss-Legendre support, but vectorized guard
    # costs nothing).
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_a = np.where(A1 > 0.0, 1.0 / A1, 0.0)
        inv_b = np.where(B1 > 0.0, 1.0 / B1, 0.0)

    # For T1 (a = rho, b = rho*(1-eta))
    ab = A1 * B1
    kernel_even_T1 = (
        fpp_a_T1 * fqp_b_T1 + (m**2) * fp_a_T1 * fq_b_T1 * inv_a * inv_b
    ) * K_even
    kernel_odd_T1 = (
        m * fp_a_T1 * fqp_b_T1 * inv_a + m * fpp_a_T1 * fq_b_T1 * inv_b
    ) * K_odd
    integrand_T1 = ab * (kernel_even_T1 + kernel_odd_T1)

    # For T2 (a = rho*(1-eta), b = rho) — swap argument roles
    fp_a_T2 = f_p(B1)
    fpp_a_T2 = f_prime_p(B1)
    fq_b_T2 = f_q(A1)
    fqp_b_T2 = f_prime_q(A1)
    inv_a_T2 = inv_b
    inv_b_T2 = inv_a
    ab_T2 = ab  # a*b unchanged under swap
    kernel_even_T2 = (
        fpp_a_T2 * fqp_b_T2 + (m**2) * fp_a_T2 * fq_b_T2 * inv_a_T2 * inv_b_T2
    ) * K_even
    kernel_odd_T2 = (
        m * fp_a_T2 * fqp_b_T2 * inv_a_T2 + m * fpp_a_T2 * fq_b_T2 * inv_b_T2
    ) * K_odd
    integrand_T2 = ab_T2 * (kernel_even_T2 + kernel_odd_T2)

    total = (MU0 / 8.0) * float(np.sum(jac * (integrand_T1 + integrand_T2)))
    return total


def disc_disc_cross_block(
    f_p: Callable[[np.ndarray], np.ndarray],
    f_prime_p: Callable[[np.ndarray], np.ndarray],
    f_q: Callable[[np.ndarray], np.ndarray],
    f_prime_q: Callable[[np.ndarray], np.ndarray],
    m: int,
    R_p: float,
    R_q: float,
    dz: float,
    n_radial: int = 32,
) -> float:
    r"""Coaxial disc-disc cross-block matrix entry at axial separation ``dz``.

    Non-singular counterpart of :func:`disc_self_block_mode_m`: kernel
    ``1 / |x - x'|`` never touches the diagonal because ``dz > 0``
    strictly, so plain tensor-product Gauss-Legendre on ``[0, R_p] x
    [0, R_q]`` gives exponential convergence with no subtraction or
    Duffy mapping.

    Used both for the intra-puck top-bottom sub-block (``dz = t`` of the
    puck) and for inter-puck disc-disc sub-blocks when the two pucks
    are coaxial (see :func:`pucks_are_coaxial`).

    Args:
        f_p, f_prime_p: Radial basis and derivative on disc ``p``
            (radius ``R_p``).
        f_q, f_prime_q: Same for disc ``q`` (radius ``R_q``).
        m: Azimuthal mode (shared between both discs; cross entries
            with different ``m`` vanish).
        R_p, R_q: Disc radii.
        dz: Signed axial separation between disc centers (m).  Must be
            strictly nonzero; callers should route ``dz == 0`` through
            :func:`disc_self_block_mode_m`.
        n_radial: Number of Gauss-Legendre nodes per axis.

    Returns:
        Block entry ``L_{pq}^{(m)}`` in henries.
    """
    if abs(dz) < 1.0e-15:
        raise ValueError(
            "disc_disc_cross_block: dz must be strictly nonzero; use "
            "disc_self_block_mode_m for coincident-plane blocks."
        )

    a_nodes, w_a = _leggauss_on_interval(R_p, n_radial)
    b_nodes, w_b = _leggauss_on_interval(R_q, n_radial)
    A, B = np.meshgrid(a_nodes, b_nodes, indexing="ij")
    W = w_a[:, None] * w_b[None, :]

    if m == 0:
        # Axisymmetric: just M(a, b, dz) weighted by f_p'(a) f_q'(b).
        denom = (A + B) ** 2 + dz**2
        m_sq = np.where(denom > 0.0, 4.0 * A * B / denom, 0.0)
        m_sq = np.clip(m_sq, 0.0, 1.0 - 1.0e-15)
        k = np.sqrt(m_sq)
        K_k = ellipk(m_sq)
        E_k = ellipe(m_sq)
        M_ab = np.where(
            (A > 0.0) & (B > 0.0),
            MU0
            * np.sqrt(A * B)
            * ((2.0 - m_sq) * K_k - 2.0 * E_k)
            / np.maximum(k, 1e-300),
            0.0,
        )
        return float(np.sum(W * f_prime_p(A) * f_prime_q(B) * M_ab))

    # m >= 1: full mode-m kernel
    I_all = fourier_kernel_Im(A, B, dz, m + 1)
    I_mm1 = I_all[m - 1]
    I_mp1 = I_all[m + 1]
    K_even = I_mm1 + I_mp1
    K_odd = I_mm1 - I_mp1

    fp_a = f_p(A)
    fpp_a = f_prime_p(A)
    fq_b = f_q(B)
    fqp_b = f_prime_q(B)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_a = np.where(A > 0.0, 1.0 / A, 0.0)
        inv_b = np.where(B > 0.0, 1.0 / B, 0.0)

    kernel_even = (fpp_a * fqp_b + (m**2) * fp_a * fq_b * inv_a * inv_b) * K_even
    kernel_odd = (m * fp_a * fqp_b * inv_a + m * fpp_a * fq_b * inv_b) * K_odd
    integrand = A * B * (kernel_even + kernel_odd)
    return (MU0 / 8.0) * float(np.sum(W * integrand))


# ----------------------------------------------------------------------
# Cylinder (side-wall) self-block
# ----------------------------------------------------------------------


def cylinder_self_block_mode_m(
    f_p: Callable[[np.ndarray], np.ndarray],
    f_prime_p: Callable[[np.ndarray], np.ndarray],
    f_q: Callable[[np.ndarray], np.ndarray],
    f_prime_q: Callable[[np.ndarray], np.ndarray],
    m: int,
    R: float,
    t: float,
    n_z: int = 32,
) -> float:
    r"""Fourier-mode ``m`` self-block entry on a cylindrical side wall.

    For a cylinder of radius ``R`` and axial extent ``z \in [-t/2, t/2]``
    with outward normal ``\hat n = \hat\rho``, a current-potential basis
    function ``g(z, phi) = f_p(z) \cos(m phi)`` (or ``\sin``) yields the
    surface current

    .. math::

        \mathbf K = \hat\rho \times \nabla_s g
        = -(f_p'(z))\cos(m\phi)\,\hat\phi
        + \tfrac{1}{R}\,\partial_\phi g\,\hat z,

    where ``f_p'(z) = dg/dz`` (physical ``z`` coordinate).  After closed-
    form azimuthal integration against ``1/|x-y|`` on the coaxial
    cylinder (``|x-y|^2 = 4 R^2 \sin^2(\Delta\phi/2) + (z-z')^2``) the
    entry reduces to a one-dimensional integral over ``(z, z')``:

    * ``m = 0``:

      .. math::

          L_{pq}^{(m=0)} = \frac{\mu_0 R^2}{2}
          \int\!\!\int dz\,dz'\,f_p'(z)\,f_q'(z')\,
          I_1(R, R, z - z').

      This is the Neumann/Maxwell coaxial-loop form.

    * ``m \ge 1``:

      .. math::

          L_{pq}^{(m)} = \frac{\mu_0}{4}
          \int\!\!\int dz\,dz'
          \Big\{ m^2 f_p(z) f_q(z')\,I_m(R, R, z-z')
          + \tfrac{R^2}{2} f_p'(z) f_q'(z')\,
          \big[ I_{m-1}(R, R, z-z') + I_{m+1}(R, R, z-z') \big]
          \Big\}.

    The kernel ``I_m(R, R, 0)`` has the usual logarithmic singularity at
    ``z = z'``; this is handled exactly as for the disc-disc case, via
    a one-dimensional Duffy map on ``(z + t/2, z' + t/2)`` followed by a
    Gauss-Laguerre rule in ``\eta = \exp(-\tau)``.  The scheme is
    exponentially convergent in ``n_z``.

    ``cos``-``cos`` and ``sin``-``sin`` same-``m`` entries are equal;
    ``cos``-``sin`` cross entries vanish identically (not evaluated
    here).

    Args:
        f_p, f_prime_p: Axial basis function and derivative
            ``f_p(z), df_p/dz`` (physical ``z``).
        f_q, f_prime_q: Same for basis ``q``.
        m: Azimuthal mode (``\ge 0``).
        R: Cylinder radius (m).
        t: Axial extent (m).  The side wall runs over ``z \in [-t/2,
            t/2]``.
        n_z: Number of Gauss-Legendre and Gauss-Laguerre nodes per axis.

    Returns:
        Block entry in henries.
    """
    # One-dimensional Duffy map on w = z + t/2 in [0, t], eta in (0, 1]
    # with the Gauss-Laguerre substitution eta = exp(-tau).  Reuses the
    # same helper as the disc-disc self: interpret 'R' as 't' and 'rho'
    # as 'w'.
    w_nodes, w_w, eta_nodes, w_eta = _duffy_laguerre_nodes_weights(t, n_z)
    W, ETA = np.meshgrid(w_nodes, eta_nodes, indexing="ij")

    # Triangle 1: (w, w') = (W, W*(1-ETA))  with W > W'.
    #   z = W - t/2,  z' = W*(1-ETA) - t/2,  z - z' = W*ETA.
    Z1 = W - 0.5 * t
    Z2 = W * (1.0 - ETA) - 0.5 * t
    DZ = W * ETA  # strictly positive Duffy axial separation
    jac = W * w_w[:, None] * w_eta[None, :]

    # Broadcast scalar radii to the (W, ETA) shape so fourier_kernel_Im
    # uses the correct output shape (it currently infers shape from (a,
    # b) only; passing matching arrays is the robust path).
    A = np.full_like(DZ, R)
    B = np.full_like(DZ, R)

    if m == 0:
        # Neumann/Maxwell form: only the T'_k T'_{k'} term survives.
        I_all = fourier_kernel_Im(A, B, DZ, 1)
        I_1 = I_all[1]
        fpp_z1 = f_prime_p(Z1)
        fqp_z2 = f_prime_q(Z2)
        val_T1 = float(np.sum(jac * fpp_z1 * fqp_z2 * I_1))
        # Triangle 2: swap roles of (z, z') -> (z', z).  Kernel I_1 is
        # symmetric in dz.
        fpp_z2 = f_prime_p(Z2)
        fqp_z1 = f_prime_q(Z1)
        val_T2 = float(np.sum(jac * fpp_z2 * fqp_z1 * I_1))
        return 0.5 * MU0 * R**2 * (val_T1 + val_T2)

    # m >= 1: two-term formula.
    I_all = fourier_kernel_Im(A, B, DZ, m + 1)
    I_mm1 = I_all[m - 1]
    I_mp1 = I_all[m + 1]
    K_even = I_mm1 + I_mp1

    # Triangle 1 integrand.
    fp_z1 = f_p(Z1)
    fpp_z1 = f_prime_p(Z1)
    fq_z2 = f_q(Z2)
    fqp_z2 = f_prime_q(Z2)
    I_m = I_all[m]
    integrand_T1 = (m**2) * fp_z1 * fq_z2 * I_m + 0.5 * (
        R**2
    ) * fpp_z1 * fqp_z2 * K_even

    # Triangle 2 integrand (swap).
    fp_z2 = f_p(Z2)
    fpp_z2 = f_prime_p(Z2)
    fq_z1 = f_q(Z1)
    fqp_z1 = f_prime_q(Z1)
    integrand_T2 = (m**2) * fp_z2 * fq_z1 * I_m + 0.5 * (
        R**2
    ) * fpp_z2 * fqp_z1 * K_even

    total = float(np.sum(jac * (integrand_T1 + integrand_T2)))
    return 0.25 * MU0 * total


# ----------------------------------------------------------------------
# Disc-side cross-block (rim-corner quadrature)
# ----------------------------------------------------------------------


def disc_side_cross_block_mode_m(
    f_disc: Callable[[np.ndarray], np.ndarray],
    f_prime_disc: Callable[[np.ndarray], np.ndarray],
    f_side: Callable[[np.ndarray], np.ndarray],
    f_prime_side: Callable[[np.ndarray], np.ndarray],
    m: int,
    R: float,
    t: float,
    zface: float,
    n_rho: int = 32,
    n_z: int = 32,
) -> float:
    r"""Fourier-mode ``m`` cross-block between a flat disc and the
    coaxial cylindrical side wall of the **same puck**.

    The disc sits at axial coordinate ``z = zface`` with radius ``R``
    and implied outward normal ``+\hat z`` (the caller must apply a
    global sign flip of ``-1`` when the physical disc has ``-\hat z``
    normal -- i.e. the bottom face -- because this routine computes
    with the ``+\hat z`` convention for both ``zface = +t/2`` and
    ``zface = -t/2``).  The side wall sits at ``\rho = R`` with
    ``z' \in [-t/2, t/2]`` and normal ``+\hat\rho``.

    After closed-form azimuthal integration the entry reduces to a
    two-dimensional integral over ``(\rho, z')``:

    * ``m = 0`` (axisymmetric, single trig branch ``\cos``):

      .. math::

          L^{(m=0)} = -\frac{\mu_0 R}{2}
          \int_0^R\!\!\int_{-t/2}^{t/2}
          \rho\,f_\mathrm{disc}'(\rho)\,f_\mathrm{side}'(z')\,
          I_1(\rho, R, z' - z_\mathrm{face})\,d\rho\,dz'.

    * ``m \ge 1``:

      .. math::

          L^{(m)} = -\frac{\mu_0 R}{8}
          \int_0^R\!\!\int_{-t/2}^{t/2}
          f_\mathrm{side}'(z')\,
          \Big\{ m\,f_\mathrm{disc}(\rho)\,
          \big[ I_{m-1} - I_{m+1} \big]
          + \rho\,f_\mathrm{disc}'(\rho)\,
          \big[ I_{m-1} + I_{m+1} \big] \Big\}\,d\rho\,dz',

      with ``I_k = I_k(\rho, R, z' - z_\mathrm{face})``.

    The integrand has a logarithmic (weakly-singular) corner at
    ``(\rho, z') = (R, z_\mathrm{face})`` and is smooth elsewhere.  We
    resolve the corner with a tensor-product Gauss-Laguerre rule under
    the exponential substitution

    .. math::

        \rho = R - R\,e^{-u_\rho},\qquad
        v = |z' - z_\mathrm{face}| = t\,e^{-u_z},

    so that ``u_\rho, u_z \to +\infty`` compresses into the singular
    corner and the Laguerre weight ``e^{-u}`` damps the linear-in-``u``
    growth of the log kernel.  The scheme is exponentially convergent
    in ``(n_\rho, n_z)``.

    ``cos``-``cos`` and ``sin``-``sin`` same-``m`` entries are equal;
    ``cos``-``sin`` cross entries vanish (not evaluated).

    Args:
        f_disc, f_prime_disc: Disc radial basis ``f(\rho)`` and its
            derivative ``df/d\rho`` (physical ``\rho``).
        f_side, f_prime_side: Side-wall axial basis ``f(z')`` and
            ``df/dz'`` (physical ``z'``).
        m: Azimuthal mode (``\ge 0``).
        R: Puck radius (m).  Both the disc and the side-wall share this
            radius.
        t: Puck thickness (m).  The side wall spans ``z' \in [-t/2,
            t/2]``.
        zface: Axial coordinate of the disc plane.  Use ``+t/2`` for
            the top face and ``-t/2`` for the bottom face.  Sign flip
            for the physical ``-\hat z`` normal of the bottom disc is
            the **caller's** responsibility.
        n_rho: Number of Gauss-Laguerre nodes in the ``\rho`` direction.
        n_z: Number of Gauss-Laguerre nodes in the ``z'`` direction.

    Returns:
        Block entry in henries (``+\hat z``-normal convention for the
        disc face).
    """
    if abs(abs(zface) - 0.5 * t) > 1.0e-12 * max(t, 1.0):
        raise ValueError(
            "disc_side_cross_block_mode_m: zface must equal +/- t/2 for "
            f"an intra-puck rim corner; got zface={zface}, t={t}."
        )

    # Side-face orientation: z' = zface - sign_face * v with v in [0, t].
    sign_face = 1.0 if zface > 0.0 else -1.0

    u_rho, w_u_rho = np.polynomial.laguerre.laggauss(n_rho)
    u_z, w_u_z = np.polynomial.laguerre.laggauss(n_z)

    # Exp-substitution nodes.  s = R exp(-u_rho) is the "distance from
    # the rim"; rho = R - s.  v = t exp(-u_z) is |z' - zface|.
    S = R * np.exp(-u_rho)
    V = t * np.exp(-u_z)

    S_grid, V_grid = np.meshgrid(S, V, indexing="ij")
    RHO = R - S_grid
    Z_PRIME = zface - sign_face * V_grid
    DZ = V_grid  # (z' - zface) magnitude; I_m even in dz so sign unused
    # Signed dz for consistency with other callers (I_m uses dz**2 only).
    signed_dz = zface - Z_PRIME  # = sign_face * V_grid, but sign drops out

    # Laguerre weight absorbs exp(-u); integration measure is R*t * w*w.
    meas = (R * t) * (w_u_rho[:, None] * w_u_z[None, :])

    # Evaluate basis.
    fd = f_disc(RHO)
    fd_p = f_prime_disc(RHO)
    fs_p = f_prime_side(Z_PRIME)

    # Kernels.
    A_grid = RHO
    B_grid = np.full_like(RHO, R)

    if m == 0:
        I_all = fourier_kernel_Im(A_grid, B_grid, DZ, 1)
        I_1 = I_all[1]
        integrand = RHO * fd_p * fs_p * I_1
        val = float(np.sum(meas * integrand))
        return -0.5 * MU0 * R * val

    # m >= 1
    I_all = fourier_kernel_Im(A_grid, B_grid, DZ, m + 1)
    I_mm1 = I_all[m - 1]
    I_mp1 = I_all[m + 1]
    K_even = I_mm1 + I_mp1
    K_odd = I_mm1 - I_mp1

    integrand = fs_p * (m * fd * K_odd + RHO * fd_p * K_even)
    val = float(np.sum(meas * integrand))
    # Suppress unused-variable lint: signed_dz is for future generalization
    # to the non-coaxial path.
    del signed_dz
    return -(MU0 * R / 8.0) * val


# ----------------------------------------------------------------------
# Assembler
# ----------------------------------------------------------------------


def _zernike_radial_numpy(rho: np.ndarray, m: int, n: int) -> np.ndarray:
    """Pure-NumPy Zernike radial polynomial on unit disc.

    Duplicates :func:`simsopt.field.puck_basis.zernike_radial` without
    the JAX dependency, so the disc assembler stays NumPy-only.
    """
    rho = np.asarray(rho, dtype=float)
    R = np.zeros_like(rho)
    for k in range((n - m) // 2 + 1):
        num = (-1) ** k * factorial(n - k)
        den = factorial(k) * factorial((n + m) // 2 - k) * factorial((n - m) // 2 - k)
        R += (num / den) * rho ** (n - 2 * k)
    return R


def _zernike_radial_derivative_numpy(
    rho: np.ndarray,
    m: int,
    n: int,
) -> np.ndarray:
    """Pure-NumPy dR_n^m / drho on unit disc."""
    rho = np.asarray(rho, dtype=float)
    dR = np.zeros_like(rho)
    for k in range((n - m) // 2 + 1):
        num = (-1) ** k * factorial(n - k)
        den = factorial(k) * factorial((n + m) // 2 - k) * factorial((n - m) // 2 - k)
        power = n - 2 * k
        if power > 0:
            dR += (num / den) * power * rho ** (power - 1)
    return dR


def _chebyshev_t_numpy(x: np.ndarray, k: int) -> np.ndarray:
    """Pure-NumPy Chebyshev ``T_k(x)`` on ``[-1, 1]`` (clipped)."""
    x = np.clip(np.asarray(x, dtype=float), -1.0, 1.0)
    return np.cos(k * np.arccos(x))


def _chebyshev_t_derivative_numpy(x: np.ndarray, k: int) -> np.ndarray:
    """Pure-NumPy derivative ``dT_k/dx`` on ``[-1, 1]`` (interior)."""
    x = np.asarray(x, dtype=float)
    # Leave finite-difference-style guard away from endpoints.  Gauss-
    # Legendre nodes are interior, so the clip is only a safety belt
    # for callers that evaluate at +/-1.
    x_safe = np.clip(x, -1.0 + 1.0e-15, 1.0 - 1.0e-15)
    if k == 0:
        return np.zeros_like(x_safe)
    return k * np.sin(k * np.arccos(x_safe)) / np.sqrt(1.0 - x_safe**2)


def _make_axial(
    t: float,
    k: int,
) -> Tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]:
    """Return ``(f(z), f'(z))`` for the side-wall basis ``g = T_k(2z/t)``
    in physical ``z`` (not normalized ``zeta``)."""

    def f(z: np.ndarray) -> np.ndarray:
        zeta = 2.0 * np.asarray(z, dtype=float) / t
        return _chebyshev_t_numpy(zeta, k)

    def fp(z: np.ndarray) -> np.ndarray:
        zeta = 2.0 * np.asarray(z, dtype=float) / t
        return (2.0 / t) * _chebyshev_t_derivative_numpy(zeta, k)

    return f, fp


def _parse_side_dof_name(name: str) -> Optional[Tuple[int, int, str]]:
    """Parse ``"side_m{m}_k{k}_{cos|sin}"``.

    Returns ``(m, k, trig)`` or ``None`` if the name is not a side-wall
    DOF.
    """
    parts = name.split("_")
    if len(parts) != 4 or parts[0] != "side":
        return None
    try:
        m = int(parts[1][1:])
        k = int(parts[2][1:])
    except (ValueError, IndexError):
        return None
    trig = parts[3]
    if trig not in ("cos", "sin"):
        return None
    return m, k, trig


def _parse_disk_dof_name(name: str) -> Optional[Tuple[str, int, int, str]]:
    """Parse ``"disk_{top|bot}_m{m}_n{n}_{cos|sin}"``.

    Returns ``(face, m, n, trig)`` or ``None`` if the name is a
    side-wall DOF (or otherwise unrecognized).
    """
    parts = name.split("_")
    if len(parts) != 5 or parts[0] != "disk":
        return None
    face = parts[1]  # "top" or "bot"
    try:
        m = int(parts[2][1:])  # "m0" -> 0
        n = int(parts[3][1:])  # "n2" -> 2
    except (ValueError, IndexError):
        return None
    trig = parts[4]
    if face not in ("top", "bot") or trig not in ("cos", "sin"):
        return None
    return face, m, n, trig


def _make_radial(
    R_disk: float,
    m: int,
    n: int,
) -> Tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]:
    """Return ``(f(rho), f'(rho))`` for the Zernike radial part of
    ``g = R_n^m(rho / R_disk) * trig(m phi)``.
    """

    def f(rho: np.ndarray) -> np.ndarray:
        r = np.clip(np.asarray(rho) / R_disk, 0.0, 1.0)
        return _zernike_radial_numpy(r, m, n)

    def fp(rho: np.ndarray) -> np.ndarray:
        r = np.clip(np.asarray(rho) / R_disk, 0.0, 1.0)
        return (1.0 / R_disk) * _zernike_radial_derivative_numpy(r, m, n)

    return f, fp


def assemble_puck_self_L(
    basis: PuckBasisData,
    R: float,
    t: float,
    n_radial: int = 32,
    n_z_side: Optional[int] = None,
    n_rho_corner: Optional[int] = None,
    n_z_corner: Optional[int] = None,
) -> np.ndarray:
    r"""Full semi-analytic self-block of one puck's inductance matrix.

    Fills **all nine** face-pair sub-blocks of the puck self-block
    ``L^{(i)}`` (top-top, top-bot, top-side, bot-top, bot-bot, bot-side,
    side-top, side-bot, side-side) using the appendix-consistent
    singular/near-singular quadratures, so the resulting full puck
    self-block stays positive semi-definite and can be used directly by
    the downstream eigenvalue-floor solver in
    :func:`~simsopt.field.bulk_inductance.shell_solve_eigenfloor_pure`.

    Block-by-block dispatch:

    * ``disk_top``-``disk_top``, ``disk_bot``-``disk_bot``:
      :func:`disc_self_block_mode_m` (Duffy + Gauss-Laguerre radial
      quadrature, exponentially convergent).
    * ``disk_top``-``disk_bot`` and transpose:
      :func:`disc_disc_cross_block` at ``dz = t`` (smooth 2D Gauss-
      Legendre).  Carries a ``-1`` sign from the opposite outward
      normals.
    * ``side``-``side``:  :func:`cylinder_self_block_mode_m`
      (1D Duffy + Gauss-Laguerre on ``(z, z')``).
    * ``disk_{top,bot}``-``side`` and transposes:
      :func:`disc_side_cross_block_mode_m` (2D Gauss-Laguerre polar-
      exponential rule at the rim corner).  For ``disk_bot`` the sign
      is flipped by ``-1`` because the physical normal is ``-\hat z``
      while the rim-corner routine uses the ``+\hat z`` convention.

    Fourier orthogonality is exploited throughout:

    * Different-``m`` entries vanish identically.
    * ``cos`` vs ``sin`` same-``m`` entries vanish identically.
    * ``cos``-``cos`` and ``sin``-``sin`` same-``m`` entries are equal;
      the ``sin`` branch reuses the ``cos`` computation.

    Entries marked ``NaN`` in the output correspond to DOF pairs whose
    names could not be parsed (neither ``disk_*`` nor ``side_*``); those
    should be treated as "compute from the regularized numerical path"
    by the caller.  In practice the standard basis produced by
    :func:`~simsopt.field.puck_basis.build_puck_shell_basis` has no
    such DOFs, and the returned matrix is fully populated.

    Args:
        basis: :class:`~simsopt.field.puck_basis.PuckBasisData` for one
            puck in its local frame.
        R: Puck radius (m).
        t: Puck thickness (m).
        n_radial: Number of Gauss-Legendre / Gauss-Laguerre nodes used
            in the 1D radial quadratures (disc-disc blocks).
        n_z_side: Number of Gauss-Legendre / Gauss-Laguerre nodes used
            in the side-side 1D axial quadrature.  Defaults to
            ``n_radial``.
        n_rho_corner: Number of Gauss-Laguerre nodes in the radial
            direction of the disc-side rim-corner quadrature.  Defaults
            to ``n_radial``.
        n_z_corner: Number of Gauss-Laguerre nodes in the axial
            direction of the disc-side rim-corner quadrature.  Defaults
            to ``n_radial``.

    Returns:
        ``(n_dof, n_dof)`` NumPy array with the full semi-analytic puck
        self-block.  Entries for unrecognized DOF pairs are ``NaN``
        (ordinarily none with the standard puck basis).
    """
    n_z_side = int(n_z_side if n_z_side is not None else n_radial)
    n_rho_corner = int(n_rho_corner if n_rho_corner is not None else n_radial)
    n_z_corner = int(n_z_corner if n_z_corner is not None else n_radial)

    names = basis.dof_names
    n_dof = len(names)
    L = np.full((n_dof, n_dof), np.nan)

    # Classify all DOFs as either disk or side.
    disk_info: List[Optional[Tuple[str, int, int, str]]] = [
        _parse_disk_dof_name(nm) for nm in names
    ]
    side_info: List[Optional[Tuple[int, int, str]]] = [
        _parse_side_dof_name(nm) for nm in names
    ]

    def _mode_trig(idx: int) -> Optional[Tuple[int, str]]:
        if disk_info[idx] is not None:
            _, m, _, trig = disk_info[idx]  # type: ignore[misc]
            return m, trig
        if side_info[idx] is not None:
            m, _, trig = side_info[idx]  # type: ignore[misc]
            return m, trig
        return None

    # Fill Fourier-orthogonal zeros for all (disk or side) DOF pairs
    # whose (m, trig) mismatch.
    recognized = [
        i for i in range(n_dof) if disk_info[i] is not None or side_info[i] is not None
    ]
    for p in recognized:
        mt_p = _mode_trig(p)
        for q in recognized:
            mt_q = _mode_trig(q)
            if mt_p != mt_q:
                L[p, q] = 0.0

    # Group all recognized DOFs by (m, trig).
    groups: dict = {}
    for idx in recognized:
        key = _mode_trig(idx)
        groups.setdefault(key, []).append(idx)

    # Caches across all groups.
    radial_cache: dict = {}  # (m, n) -> (f, f')  for disk modes
    axial_cache: dict = {}  # k -> (f, f')        for side modes

    def _get_radial(m: int, n: int):
        if (m, n) not in radial_cache:
            radial_cache[(m, n)] = _make_radial(R, m, n)
        return radial_cache[(m, n)]

    def _get_axial(k: int):
        if k not in axial_cache:
            axial_cache[k] = _make_axial(t, k)
        return axial_cache[k]

    T_HALF = 0.5 * t
    for (m, trig), idxs in groups.items():
        # Split into disk-top, disk-bot, side entries for this group.
        tops: List[Tuple[int, int]] = []  # (idx, n)
        bots: List[Tuple[int, int]] = []  # (idx, n)
        sides: List[Tuple[int, int]] = []  # (idx, k)
        for idx in idxs:
            if disk_info[idx] is not None:
                face, mm, nn, _ = disk_info[idx]  # type: ignore[misc]
                if face == "top":
                    tops.append((idx, nn))
                else:
                    bots.append((idx, nn))
            else:
                _, kk, _ = side_info[idx]  # type: ignore[misc]
                sides.append((idx, kk))

        # disk_top x disk_top (singular self)
        for p, n_p in tops:
            f_p, fp_p = _get_radial(m, n_p)
            for q, n_q in tops:
                f_q, fp_q = _get_radial(m, n_q)
                val = disc_self_block_mode_m(
                    f_p,
                    fp_p,
                    f_q,
                    fp_q,
                    m,
                    R,
                    n_radial=n_radial,
                )
                L[p, q] = val  # both faces have +z normal here

        # disk_bot x disk_bot (singular self): same value as top x top
        for p, n_p in bots:
            f_p, fp_p = _get_radial(m, n_p)
            for q, n_q in bots:
                f_q, fp_q = _get_radial(m, n_q)
                val = disc_self_block_mode_m(
                    f_p,
                    fp_p,
                    f_q,
                    fp_q,
                    m,
                    R,
                    n_radial=n_radial,
                )
                L[p, q] = val  # (-z) . (-z) = +1

        # disk_top x disk_bot (smooth, dz = t, opposite normals -> -1)
        for p, n_p in tops:
            f_p, fp_p = _get_radial(m, n_p)
            for q, n_q in bots:
                f_q, fp_q = _get_radial(m, n_q)
                val = disc_disc_cross_block(
                    f_p,
                    fp_p,
                    f_q,
                    fp_q,
                    m,
                    R,
                    R,
                    t,
                    n_radial=n_radial,
                )
                L[p, q] = -val
                L[q, p] = -val

        # side x side (singular self)
        for p, k_p in sides:
            f_p, fp_p = _get_axial(k_p)
            for q, k_q in sides:
                f_q, fp_q = _get_axial(k_q)
                val = cylinder_self_block_mode_m(
                    f_p,
                    fp_p,
                    f_q,
                    fp_q,
                    m,
                    R,
                    t,
                    n_z=n_z_side,
                )
                L[p, q] = val

        # disk_top x side (rim corner at zface = +t/2, +z normal -> +1)
        for p, n_p in tops:
            f_disc, fp_disc = _get_radial(m, n_p)
            for q, k_q in sides:
                f_side, fp_side = _get_axial(k_q)
                val = disc_side_cross_block_mode_m(
                    f_disc,
                    fp_disc,
                    f_side,
                    fp_side,
                    m,
                    R,
                    t,
                    zface=+T_HALF,
                    n_rho=n_rho_corner,
                    n_z=n_z_corner,
                )
                L[p, q] = val
                L[q, p] = val

        # disk_bot x side (rim corner at zface = -t/2, -z normal -> -1)
        for p, n_p in bots:
            f_disc, fp_disc = _get_radial(m, n_p)
            for q, k_q in sides:
                f_side, fp_side = _get_axial(k_q)
                val = disc_side_cross_block_mode_m(
                    f_disc,
                    fp_disc,
                    f_side,
                    fp_side,
                    m,
                    R,
                    t,
                    zface=-T_HALF,
                    n_rho=n_rho_corner,
                    n_z=n_z_corner,
                )
                L[p, q] = -val
                L[q, p] = -val

    return L


def assemble_puck_disc_faces_L(
    basis: PuckBasisData,
    R: float,
    t: float,
    n_radial: int = 32,
) -> np.ndarray:
    r"""Legacy entry point kept for backward compatibility.

    Returns only the disc-face-disc-face sub-blocks of the puck
    self-inductance (top-top, top-bot, bot-bot) computed by
    :func:`assemble_puck_self_L`.  All other entries (involving side-
    wall DOFs) are set to ``NaN``, mirroring the historical behavior
    of this helper.

    New callers should prefer :func:`assemble_puck_self_L`, which
    produces a fully populated PSD puck self-block.

    Args:
        basis: :class:`~simsopt.field.puck_basis.PuckBasisData` for one
            puck in its local frame.
        R: Disc radius (m).
        t: Axial distance between the top and bottom disc planes (m).
        n_radial: Number of Gauss-Legendre / Gauss-Laguerre nodes used
            in the 1D radial quadratures.

    Returns:
        ``(n_dof, n_dof)`` NumPy array; ``NaN`` on every entry
        involving a side-wall DOF.
    """
    L_full = assemble_puck_self_L(basis, R, t, n_radial=n_radial)
    names = basis.dof_names
    side_mask = np.array(
        [_parse_side_dof_name(nm) is not None for nm in names],
        dtype=bool,
    )
    # Blank out all entries that involve a side-wall DOF; keep only
    # disk x disk entries (plus Fourier-orthogonal zeros between disk
    # DOFs).
    n_dof = len(names)
    L = np.full((n_dof, n_dof), np.nan)
    disk_mask = ~side_mask
    # The historical helper also only populated disk-disk entries; keep
    # that contract.
    for p in range(n_dof):
        if not disk_mask[p]:
            continue
        for q in range(n_dof):
            if not disk_mask[q]:
                continue
            L[p, q] = L_full[p, q]
    return L


# ----------------------------------------------------------------------
# Coaxial-mutual helper
# ----------------------------------------------------------------------


def fill_coaxial_inter_puck_disc_block(
    L: np.ndarray,
    d0_i: int,
    d0_j: int,
    basis_i: PuckBasisData,
    basis_j: PuckBasisData,
    R_i: float,
    t_i: float,
    R_j: float,
    t_j: float,
    c_i: np.ndarray,
    axis_i: np.ndarray,
    c_j: np.ndarray,
    axis_j: np.ndarray,
    n_radial: int = 32,
) -> bool:
    r"""If two pucks are coaxial, overwrite disc–disc mutuals with :func:`disc_disc_cross_block`.

    Side-wall and mixed face pairs are left unchanged in ``L``.  Pucks must
    share the same (parallel) global axis; anti-parallel axes are not
    handled.  Gated in the driver by :envvar:`SIMSOPT_PSC_COAXIAL_MUTUAL`.

    Returns:
        ``True`` if a coaxial disc–disc sub-block was written, else ``False``.
    """
    dz0 = pucks_are_coaxial(c_i, axis_i, c_j, axis_j, atol=1.0e-4)  # noqa: F841  — use geometry below
    if dz0 is None:
        return False
    n1 = np.asarray(axis_i, dtype=float).ravel()[:3]
    n2 = np.asarray(axis_j, dtype=float).ravel()[:3]
    n1 = n1 / (np.linalg.norm(n1) + 1.0e-30)
    n2 = n2 / (np.linalg.norm(n2) + 1.0e-30)
    if float(np.dot(n1, n2)) < 0.99:
        return False
    c_i = np.asarray(c_i, dtype=float).reshape(3)
    c_j = np.asarray(c_j, dtype=float).reshape(3)
    dz_cc = float(np.dot(c_j - c_i, n1))
    d_tt = dz_cc + 0.5 * (t_j - t_i)
    d_bb = dz_cc - 0.5 * (t_j - t_i)
    d_tb = dz_cc - 0.5 * (t_i + t_j)
    d_bt = dz_cc + 0.5 * (t_i + t_j)

    names_i = basis_i.dof_names
    names_j = basis_j.dof_names
    disk_i: List[Optional[Tuple[str, int, int, str]]] = [
        _parse_disk_dof_name(nm) for nm in names_i
    ]
    disk_j: List[Optional[Tuple[str, int, int, str]]] = [
        _parse_disk_dof_name(nm) for nm in names_j
    ]
    n_i = len(names_i)
    n_jd = len(names_j)

    def _z_pair(face_i: str, face_j: str) -> float:
        if (face_i, face_j) == ("top", "top"):
            return d_tt
        if (face_i, face_j) == ("bot", "bot"):
            return d_bb
        if (face_i, face_j) == ("top", "bot"):
            return d_tb
        if (face_i, face_j) == ("bot", "top"):
            return d_bt
        raise ValueError(face_i, face_j)

    wrote = False
    for pi in range(n_i):
        if disk_i[pi] is None:
            continue
        face_i, m_i, n_ri, trig_i = disk_i[pi]  # type: ignore[misc]
        for qj in range(n_jd):
            if disk_j[qj] is None:
                continue
            face_j, m_j, n_rj, trig_j = disk_j[qj]  # type: ignore[misc]
            if m_i != m_j or trig_i != trig_j:
                continue
            m = m_i
            zsep = _z_pair(face_i, face_j)
            if abs(zsep) < 1.0e-14:
                continue
            f_p, fp_p = _make_radial(R_i, m, n_ri)
            f_q, fp_q = _make_radial(R_j, m, n_rj)
            val = disc_disc_cross_block(
                f_p,
                fp_p,
                f_q,
                fp_q,
                m,
                R_i,
                R_j,
                zsep,
                n_radial=n_radial,
            )
            sgn = 1.0
            if (face_i == "top" and face_j == "bot") or (
                face_i == "bot" and face_j == "top"
            ):
                sgn = -1.0
            v = sgn * val
            L[d0_i + pi, d0_j + qj] = v
            L[d0_j + qj, d0_i + pi] = v
            wrote = True
    return bool(wrote)


def pucks_are_coaxial(
    c_i: np.ndarray,
    axis_i: np.ndarray,
    c_j: np.ndarray,
    axis_j: np.ndarray,
    atol: float = 1.0e-9,
) -> Optional[float]:
    """Test whether two pucks are coaxial (parallel axes, centers on the
    same axis line) and return the signed axial separation ``dz``.

    Args:
        c_i, c_j: Puck centers (length-3 arrays, m).
        axis_i, axis_j: Unit axis vectors (length-3 arrays).
        atol: Absolute tolerance (m) for the two coaxial checks.

    Returns:
        Signed scalar ``dz = (c_j - c_i) . axis_i`` if pucks are
        coaxial, else ``None``.  A positive ``dz`` means puck ``j`` sits
        on the positive side of puck ``i`` along ``axis_i``.
    """
    c_i = np.asarray(c_i, dtype=float).reshape(3)
    c_j = np.asarray(c_j, dtype=float).reshape(3)
    axis_i = np.asarray(axis_i, dtype=float).reshape(3)
    axis_j = np.asarray(axis_j, dtype=float).reshape(3)

    # Normalize
    n_i = axis_i / (np.linalg.norm(axis_i) + 1e-30)
    n_j = axis_j / (np.linalg.norm(axis_j) + 1e-30)
    # Parallel (up to sign)
    if np.linalg.norm(np.cross(n_i, n_j)) > atol:
        return None
    delta = c_j - c_i
    dz = float(np.dot(delta, n_i))
    # Perpendicular component of delta must be within tolerance
    perp = delta - dz * n_i
    if np.linalg.norm(perp) > atol:
        return None
    return dz
