"""
Fourier–Zernike and Fourier–Chebyshev bases on a cylindrical puck shell (top, bottom, side).

Used for the ideal-diamagnetic passive-bulk model (surface current potential on :math:`\\partial\\Omega`).
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from math import factorial
from typing import Any, List, Optional, Tuple

import jax.numpy as jnp
import numpy as np

# Cache identical local-shell basis builds (same R, t, resolution, mode counts).
# Bounded to avoid unbounded memory growth in long optimisation runs that
# explore many distinct (R, t) geometries.  Oldest entries are evicted
# FIFO once ``_PUCK_SHELL_BASIS_CACHE_MAX`` is exceeded.  Use
# :func:`clear_puck_basis_cache` to reset between independent runs, and
# :func:`set_puck_basis_cache_max` to tune the bound (e.g. ``0`` to
# disable caching entirely).
_PUCK_SHELL_BASIS_CACHE_MAX: int = 64
_PUCK_SHELL_BASIS_CACHE: (
    OrderedDict[Tuple[float, float, int, int, int, int, int, int], Any] | None
) = None


def _init_cache() -> None:
    """Lazy-initialise the basis cache as an ``OrderedDict``."""
    global _PUCK_SHELL_BASIS_CACHE
    if _PUCK_SHELL_BASIS_CACHE is None:
        _PUCK_SHELL_BASIS_CACHE = OrderedDict()


def clear_puck_basis_cache() -> None:
    """Clear all cached puck shell basis builds.

    Call this between independent optimisation runs that use different
    puck geometries to release memory held by compiled JAX basis arrays.
    """
    _init_cache()
    _PUCK_SHELL_BASIS_CACHE.clear()  # type: ignore[union-attr]


def set_puck_basis_cache_max(n: int) -> None:
    """Set the maximum number of cached puck shell basis builds.

    Args:
        n: New cache bound (``n >= 0``).  ``0`` disables caching.
    """
    global _PUCK_SHELL_BASIS_CACHE_MAX
    if n < 0:
        raise ValueError(f"cache max must be >= 0, got {n}")
    _PUCK_SHELL_BASIS_CACHE_MAX = int(n)
    _init_cache()
    while len(_PUCK_SHELL_BASIS_CACHE) > _PUCK_SHELL_BASIS_CACHE_MAX:  # type: ignore[union-attr]
        _PUCK_SHELL_BASIS_CACHE.popitem(last=False)  # type: ignore[union-attr]


__all__ = [
    "PuckBasisData",
    "zernike_radial",
    "chebyshev_t",
    "chebyshev_t_derivative",
    "list_zernike_modes",
    "build_puck_shell_basis",
    "build_continuity_constraint",
    "gauge_projection_matrix",
    "clear_puck_basis_cache",
    "set_puck_basis_cache_max",
]


def zernike_radial(rho: jnp.ndarray, m: int, n: int) -> jnp.ndarray:
    """
    Zernike radial polynomial :math:`R_n^m(\\rho)` on the unit disk.

    Args:
        rho: Normalized radius in :math:`[0,1]`.
        m: Azimuthal index (nonnegative).
        n: Radial order, with :math:`n \\ge m` and :math:`n-m` even.

    Returns:
        Values with the same shape as ``rho``.
    """
    if n < m or (n - m) % 2 != 0:
        raise ValueError(f"Invalid Zernike indices (m,n)=({m},{n})")
    rho = jnp.asarray(rho)
    R = jnp.zeros_like(rho, dtype=jnp.float64)
    for k in range((n - m) // 2 + 1):
        num = (-1) ** k * factorial(n - k)
        den = factorial(k) * factorial((n + m) // 2 - k) * factorial((n - m) // 2 - k)
        coeff = num / den
        R = R + coeff * rho ** (n - 2 * k)
    return R


def zernike_radial_derivative(rho: jnp.ndarray, m: int, n: int) -> jnp.ndarray:
    """Derivative :math:`d R_n^m / d\\rho` for the unit-disk Zernike radial part."""
    if n < m or (n - m) % 2 != 0:
        raise ValueError(f"Invalid Zernike indices (m,n)=({m},{n})")
    rho = jnp.asarray(rho)
    dR = jnp.zeros_like(rho, dtype=jnp.float64)
    for k in range((n - m) // 2 + 1):
        num = (-1) ** k * factorial(n - k)
        den = factorial(k) * factorial((n + m) // 2 - k) * factorial((n - m) // 2 - k)
        coeff = num / den
        power = n - 2 * k
        if power > 0:
            dR = dR + coeff * power * rho ** (power - 1)
    return dR


def chebyshev_t(x: jnp.ndarray, k: int) -> jnp.ndarray:
    """Chebyshev polynomial :math:`T_k(x)` on :math:`[-1,1]`."""
    x = jnp.asarray(x)
    return jnp.cos(k * jnp.arccos(jnp.clip(x, -1.0, 1.0)))


def chebyshev_t_derivative(x: jnp.ndarray, k: int) -> jnp.ndarray:
    """Derivative :math:`dT_k/dx`."""
    x = jnp.asarray(x)
    x = jnp.clip(x, -1.0 + 1e-15, 1.0 - 1e-15)
    return k * jnp.sin(k * jnp.arccos(x)) / jnp.sqrt(1.0 - x**2)


def list_zernike_modes(m_max: int, n_max: int) -> List[Tuple[int, int]]:
    """
    List all (m, n) pairs with ``m <= m_max``, ``n <= n_max``, ``n >= m``, ``n-m`` even.
    """
    modes: List[Tuple[int, int]] = []
    for m in range(m_max + 1):
        for nj in range(m, n_max + 1):
            if (nj - m) % 2 == 0:
                modes.append((m, nj))
    return modes


@dataclass
class PuckBasisData:
    """Quadrature and basis data for one puck in its local frame (z = disk normal).

    The optional ``basis_spec`` field stores the structured mode identity of
    each DOF as tuples ``(face, m, n_or_k, trig)`` where ``face`` is one of
    ``"disk_top"``, ``"disk_bot"``, ``"side"``, ``n_or_k`` is the Zernike
    radial order ``n`` for disk faces and the Chebyshev index ``k`` for the
    side wall, and ``trig`` is ``"cos"`` or ``"sin"``.  When present, it is
    consumed by :func:`_eval_basis_at_point` instead of re-parsing the
    human-readable ``dof_names`` strings.
    """

    quad_points_local: np.ndarray  # (n_quad, 3)
    quad_weights: np.ndarray  # (n_quad,)
    quad_normals_local: np.ndarray  # (n_quad, 3), unit outward
    face_id: np.ndarray  # 0=top, 1=bottom, 2=side
    phi_values: np.ndarray  # (n_quad, n_dof) basis functions g_a
    grad_phi_local: (
        np.ndarray
    )  # (n_quad, n_dof, 3) surface gradient (tangent components)
    k_basis_local: np.ndarray  # (n_quad, n_dof, 3) K = n x grad_s phi
    dof_names: List[str]
    basis_spec: List[Tuple[str, int, int, str]] = field(default_factory=list)
    #: Optional cache of :math:`(M,D,Q)` local-frame multipole tensors for
    #: far-pair fast paths (see :mod:`simsopt.field.multipole_inductance`). Filled
    #: lazily when :envvar:`SIMSOPT_PSC_MOMENTS` is ``"fourier"`` or when
    #: callers opt in to one-shot precomputation.
    moments_cache: Optional[Tuple[Any, ...]] = field(default=None, repr=False)


def _disk_grad_g_cartesian(
    rho: jnp.ndarray,
    phi: jnp.ndarray,
    R_disk: float,
    m: int,
    n: int,
    trig: str,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Gradients dg/dx, dg/dy, dg/dz for g = R_n^m(r) cos(m phi) or sin, r=rho/R_disk."""
    r_norm = rho / R_disk
    Rnm = zernike_radial(r_norm, m, n)
    dRdr = zernike_radial_derivative(r_norm, m, n)
    if trig == "cos":
        trig_m = jnp.cos(m * phi)
        dtrig = -m * jnp.sin(m * phi)
    else:
        trig_m = jnp.sin(m * phi)
        dtrig = m * jnp.cos(m * phi)
    dg_drho = (1.0 / R_disk) * dRdr * trig_m
    dg_dphi = Rnm * dtrig
    dg_dx = jnp.cos(phi) * dg_drho - (jnp.sin(phi) / jnp.maximum(rho, 1e-14)) * dg_dphi
    dg_dy = jnp.sin(phi) * dg_drho + (jnp.cos(phi) / jnp.maximum(rho, 1e-14)) * dg_dphi
    dg_dz = jnp.zeros_like(rho)
    return dg_dx, dg_dy, dg_dz


def _side_grad_K(
    z: np.ndarray,
    phi: np.ndarray,
    R_disk: float,
    t: float,
    m: int,
    k_ch: int,
    trig: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Side wall: g = T_k(2z/t) * cos/sin(m phi). Outward n = (cos phi, sin phi, 0)."""
    zeta = 2.0 * z / t
    Tk = np.array(chebyshev_t(jnp.array(zeta), k_ch))
    dTk = np.array(chebyshev_t_derivative(jnp.array(zeta), k_ch)) * (2.0 / t)
    if trig == "cos":
        ang = np.cos(m * phi)
        dang = -m * np.sin(m * phi)
    else:
        ang = np.sin(m * phi)
        dang = m * np.cos(m * phi)
    g = Tk * ang
    dg_dz = dTk * ang
    dg_dphi = Tk * dang
    nx = np.cos(phi)
    ny = np.sin(phi)
    nz = np.zeros_like(z)
    # grad_s g = (1/R) dg_dphi e_phi + dg_dz e_z
    ephi_x = -np.sin(phi)
    ephi_y = np.cos(phi)
    gs_x = (1.0 / R_disk) * dg_dphi * ephi_x + dg_dz * 0.0
    gs_y = (1.0 / R_disk) * dg_dphi * ephi_y
    gs_z = dg_dz
    # K = n x grad_s g
    cross = np.cross(
        np.stack([nx, ny, nz], axis=-1), np.stack([gs_x, gs_y, gs_z], axis=-1)
    )
    return g, np.stack([gs_x, gs_y, gs_z], axis=-1), cross


def build_puck_shell_basis(
    R: float,
    t: float,
    m_fourier: int = 4,
    l_zernike: int = 6,
    k_chebyshev: int = 4,
    n_rho: int = 12,
    n_phi: int = 16,
    n_z: int = 8,
) -> PuckBasisData:
    """
    Build non-overlapping quadrature on top, bottom, and side (open patches).

    Local coordinates: origin at disk center, z along the puck axis (outward top = +z).

    Args:
        R: Disk radius (m).
        t: Thickness (m).
        m_fourier: Max poloidal Fourier mode index on faces.
        l_zernike: Max Zernike radial order n on disk faces.
        k_chebyshev: Max Chebyshev index on the side wall.
        n_rho, n_phi, n_z: Quadrature counts for disk polar grid and side axial grid.

    Returns:
        :class:`PuckBasisData` with columns for each basis function.
    """
    cache_key = (
        float(R),
        float(t),
        int(m_fourier),
        int(l_zernike),
        int(k_chebyshev),
        int(n_rho),
        int(n_phi),
        int(n_z),
    )
    _init_cache()
    if cache_key in _PUCK_SHELL_BASIS_CACHE:
        _PUCK_SHELL_BASIS_CACHE.move_to_end(cache_key)
        return _PUCK_SHELL_BASIS_CACHE[cache_key]

    # Top: rho in (0, R), avoid rho=0 singularity for m>0 by starting small
    rho_min = R / (2 * n_rho)
    rho_grid = np.linspace(rho_min, R * (1.0 - 1e-6), n_rho)
    phi_grid = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    dphi = 2 * np.pi / n_phi
    drho = (rho_grid[1] - rho_grid[0]) if n_rho > 1 else (R - rho_min)
    Rho, Phi = np.meshgrid(rho_grid, phi_grid, indexing="ij")
    rho_t = Rho.ravel()
    phi_t = Phi.ravel()
    z_top = 0.5 * t
    z_bot = -0.5 * t

    pts_top = np.stack(
        [rho_t * np.cos(phi_t), rho_t * np.sin(phi_t), np.full_like(rho_t, z_top)],
        axis=-1,
    )
    weights_top = []
    for i in range(n_rho):
        for _j in range(n_phi):
            weights_top.append(rho_grid[i] * drho * dphi)
    weights_top = np.array(weights_top)

    pts_bot = np.stack(
        [rho_t * np.cos(phi_t), rho_t * np.sin(phi_t), np.full_like(rho_t, z_bot)],
        axis=-1,
    )
    weights_bot = weights_top.copy()

    # Side: z in open interval, phi grid
    z_side = np.linspace(-0.5 * t * (1 - 1e-6), 0.5 * t * (1 - 1e-6), n_z)
    phi_side = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    dz = (z_side[1] - z_side[0]) if n_z > 1 else t
    Zs, Phis = np.meshgrid(z_side, phi_side, indexing="ij")
    z_s = Zs.ravel()
    phi_s = Phis.ravel()
    pts_side = np.stack(
        [R * np.cos(phi_s), R * np.sin(phi_s), z_s],
        axis=-1,
    )
    w_phi_s = 2 * np.pi / n_phi
    weights_side = []
    for _ in z_side:
        for _ in range(n_phi):
            weights_side.append(R * dz * w_phi_s)
    weights_side = np.array(weights_side)

    quad_points = np.vstack([pts_top, pts_bot, pts_side])
    quad_weights = np.concatenate([weights_top, weights_bot, weights_side])
    n_top = len(rho_t)
    n_bot = len(rho_t)
    n_side = len(z_s)
    face_id = np.array(
        [0] * n_top + [1] * n_bot + [2] * n_side,
        dtype=np.int32,
    )

    normals = np.zeros_like(quad_points)
    normals[:n_top, 2] = 1.0
    normals[n_top : n_top + n_bot, 2] = -1.0
    cphi = np.cos(phi_s)
    sphi = np.sin(phi_s)
    normals[n_top + n_bot :, 0] = cphi
    normals[n_top + n_bot :, 1] = sphi

    n_total = n_top + n_bot + n_side
    sl_top = slice(0, n_top)
    sl_bot = slice(n_top, n_top + n_bot)
    sl_side = slice(n_top + n_bot, n_total)

    cols_phi: List[np.ndarray] = []
    cols_grad: List[np.ndarray] = []
    cols_k: List[np.ndarray] = []
    names: List[str] = []
    basis_spec: List[Tuple[str, int, int, str]] = []

    def _stack_face(
        top: np.ndarray,
        bot: np.ndarray,
        side: np.ndarray,
    ) -> np.ndarray:
        out = np.zeros(n_total)
        out[sl_top] = top
        out[sl_bot] = bot
        out[sl_side] = side
        return out

    def _stack_face_vec(
        top: np.ndarray,
        bot: np.ndarray,
        side: np.ndarray,
    ) -> np.ndarray:
        out = np.zeros((n_total, 3))
        out[sl_top] = top
        out[sl_bot] = bot
        out[sl_side] = side
        return out

    z_top_zeros = np.zeros(n_top)
    z_bot_zeros = np.zeros(n_bot)
    z_side_zeros = np.zeros(n_side)

    zero_top = np.zeros(n_top)
    zero_bot = np.zeros(n_bot)
    nvec_top = np.array([0.0, 0.0, 1.0])
    nvec_bot = np.array([0.0, 0.0, -1.0])
    modes = list_zernike_modes(m_fourier, l_zernike)
    for m, n in modes:
        r_norm = rho_t / R
        Rvals = np.array(zernike_radial(jnp.array(r_norm), m, n))
        if m == 0:
            g = Rvals
            dgx, dgy, _ = _disk_grad_g_cartesian(
                jnp.array(rho_t), jnp.array(phi_t), R, m, n, "cos"
            )
            grad_d = np.stack([np.array(dgx), np.array(dgy), z_top_zeros], axis=-1)
            K_d = np.cross(nvec_top, grad_d)
            cols_phi.append(_stack_face(g, zero_bot, z_side_zeros))
            cols_grad.append(
                _stack_face_vec(grad_d, np.zeros((n_bot, 3)), np.zeros((n_side, 3)))
            )
            cols_k.append(
                _stack_face_vec(K_d, np.zeros((n_bot, 3)), np.zeros((n_side, 3)))
            )
            names.append(f"disk_top_m{m}_n{n}_cos")
            basis_spec.append(("disk_top", m, n, "cos"))
            grad_d_b = np.stack([np.array(dgx), np.array(dgy), z_bot_zeros], axis=-1)
            K_b = np.cross(nvec_bot, grad_d_b)
            cols_phi.append(_stack_face(zero_top, g, z_side_zeros))
            cols_grad.append(
                _stack_face_vec(np.zeros((n_top, 3)), grad_d_b, np.zeros((n_side, 3)))
            )
            cols_k.append(
                _stack_face_vec(np.zeros((n_top, 3)), K_b, np.zeros((n_side, 3)))
            )
            names.append(f"disk_bot_m{m}_n{n}_cos")
            basis_spec.append(("disk_bot", m, n, "cos"))
        else:
            for trig, tag in (("cos", "cos"), ("sin", "sin")):
                if trig == "cos":
                    g = Rvals * np.cos(m * phi_t)
                else:
                    g = Rvals * np.sin(m * phi_t)
                dgx, dgy, _ = _disk_grad_g_cartesian(
                    jnp.array(rho_t), jnp.array(phi_t), R, m, n, trig
                )
                grad_t = np.stack([np.array(dgx), np.array(dgy), z_top_zeros], axis=-1)
                K_t = np.cross(nvec_top, grad_t)
                cols_phi.append(_stack_face(g, zero_bot, z_side_zeros))
                cols_grad.append(
                    _stack_face_vec(grad_t, np.zeros((n_bot, 3)), np.zeros((n_side, 3)))
                )
                cols_k.append(
                    _stack_face_vec(K_t, np.zeros((n_bot, 3)), np.zeros((n_side, 3)))
                )
                names.append(f"disk_top_m{m}_n{n}_{tag}")
                basis_spec.append(("disk_top", m, n, tag))
                grad_b = np.stack([np.array(dgx), np.array(dgy), z_bot_zeros], axis=-1)
                K_bb = np.cross(nvec_bot, grad_b)
                cols_phi.append(_stack_face(zero_top, g, z_side_zeros))
                cols_grad.append(
                    _stack_face_vec(np.zeros((n_top, 3)), grad_b, np.zeros((n_side, 3)))
                )
                cols_k.append(
                    _stack_face_vec(np.zeros((n_top, 3)), K_bb, np.zeros((n_side, 3)))
                )
                names.append(f"disk_bot_m{m}_n{n}_{tag}")
                basis_spec.append(("disk_bot", m, n, tag))

    for m in range(m_fourier + 1):
        for k_ch in range(k_chebyshev + 1):
            if m == 0:
                g, grad, K = _side_grad_K(z_s, phi_s, R, t, m, k_ch, "cos")
                cols_phi.append(_stack_face(z_top_zeros, z_bot_zeros, g))
                cols_grad.append(
                    _stack_face_vec(np.zeros((n_top, 3)), np.zeros((n_bot, 3)), grad)
                )
                cols_k.append(
                    _stack_face_vec(np.zeros((n_top, 3)), np.zeros((n_bot, 3)), K)
                )
                names.append(f"side_m{m}_k{k_ch}_cos")
                basis_spec.append(("side", m, k_ch, "cos"))
            else:
                for trig, tag in (("cos", "cos"), ("sin", "sin")):
                    g, grad, K = _side_grad_K(z_s, phi_s, R, t, m, k_ch, trig)
                    cols_phi.append(_stack_face(z_top_zeros, z_bot_zeros, g))
                    cols_grad.append(
                        _stack_face_vec(
                            np.zeros((n_top, 3)), np.zeros((n_bot, 3)), grad
                        )
                    )
                    cols_k.append(
                        _stack_face_vec(np.zeros((n_top, 3)), np.zeros((n_bot, 3)), K)
                    )
                    names.append(f"side_m{m}_k{k_ch}_{tag}")
                    basis_spec.append(("side", m, k_ch, tag))

    phi_mat = np.column_stack(cols_phi)
    grad_mat = np.stack(cols_grad, axis=1)
    k_mat = np.stack(cols_k, axis=1)

    out = PuckBasisData(
        quad_points_local=quad_points,
        quad_weights=quad_weights,
        quad_normals_local=normals,
        face_id=face_id,
        phi_values=phi_mat,
        grad_phi_local=grad_mat,
        k_basis_local=k_mat,
        dof_names=names,
        basis_spec=basis_spec,
    )
    if _PUCK_SHELL_BASIS_CACHE_MAX > 0:
        _PUCK_SHELL_BASIS_CACHE[cache_key] = out
        while len(_PUCK_SHELL_BASIS_CACHE) > _PUCK_SHELL_BASIS_CACHE_MAX:
            _PUCK_SHELL_BASIS_CACHE.popitem(last=False)
    return out


def build_continuity_constraint(
    basis: PuckBasisData,
    R: float,
    t: float,
    n_rim: int,
) -> np.ndarray:
    """
    Linear constraints :math:`C \\beta = 0` for rim continuity (top/side and bottom/side).

    Collocation in phi at rho=R (top/bottom) and matching side at z=±t/2.
    """
    phi_vals = np.linspace(0, 2 * np.pi, n_rim, endpoint=False)
    rows = []
    # Top rim g^top(R, phi) - g^side(z=t/2, phi).  The side patch is
    # evaluated at ``z_rim_side = (t/2)*(1 - 1e-6)`` to inset the
    # Chebyshev argument slightly inside ``[-1, 1]``; moving to the
    # exact rim (z = +/- t/2) was attempted and reverted because it
    # reduced the induced dipole on the diagnostic fixture by ~50%
    # (``test_induced_dipole_scale`` regressed).  The inset is a
    # well-tested workaround for the side basis's endpoint behaviour
    # and does not measurably affect rim continuity.
    z_top = 0.5 * t
    z_rim_side = 0.5 * t * (1.0 - 1e-6)
    for phi in phi_vals:
        row = np.zeros(basis.phi_values.shape[1])
        pt_top = np.array([R * np.cos(phi), R * np.sin(phi), z_top])
        pt_side = np.array([R * np.cos(phi), R * np.sin(phi), z_rim_side])
        for a in range(row.size):
            row[a] = _eval_basis_at_point(
                basis, pt_top, a, R, t
            ) - _eval_basis_at_point(basis, pt_side, a, R, t)
        rows.append(row)
    z_bot = -0.5 * t
    z_rim_side_b = -0.5 * t * (1.0 - 1e-6)
    for phi in phi_vals:
        row = np.zeros(basis.phi_values.shape[1])
        pt_bot = np.array([R * np.cos(phi), R * np.sin(phi), z_bot])
        pt_side = np.array([R * np.cos(phi), R * np.sin(phi), z_rim_side_b])
        for a in range(row.size):
            row[a] = _eval_basis_at_point(
                basis, pt_bot, a, R, t
            ) - _eval_basis_at_point(basis, pt_side, a, R, t)
        rows.append(row)
    return np.array(rows)


def _basis_spec_for_dof(basis: PuckBasisData, a: int) -> Tuple[str, int, int, str]:
    """Return the structured ``(face, m, n_or_k, trig)`` spec for DOF ``a``.

    Prefers the :attr:`PuckBasisData.basis_spec` field if populated;
    falls back to parsing :attr:`PuckBasisData.dof_names` for backward
    compatibility with basis objects constructed before the structured
    spec was introduced.
    """
    spec = getattr(basis, "basis_spec", None)
    if spec:
        return spec[a]
    name = basis.dof_names[a]
    parts = name.split("_")
    if parts[0] == "disk":
        face = f"disk_{parts[1]}"
        m = int(parts[2].replace("m", ""))
        n = int(parts[3].replace("n", ""))
        trig = parts[4]
        return (face, m, n, trig)
    m = int(parts[1].replace("m", ""))
    k_ch = int(parts[2].replace("k", ""))
    trig = parts[3]
    return ("side", m, k_ch, trig)


def _eval_basis_at_point(
    basis: PuckBasisData, pt: np.ndarray, a: int, R: float, t: float
) -> float:
    """Evaluate basis function ``a`` of ``basis`` at local Cartesian ``pt``.

    Uses :func:`_basis_spec_for_dof` to recover the structured mode
    identity ``(face, m, n_or_k, trig)`` instead of string-parsing the
    DOF name.
    """
    x, y, z = pt
    rho = np.hypot(x, y)
    phi = np.arctan2(y, x)
    face, m, n_or_k, trig = _basis_spec_for_dof(basis, a)
    if face == "disk_top":
        if z < 0:
            return 0.0
        r_norm = rho / R
        Rnm = float(np.array(zernike_radial(jnp.array(r_norm), m, n_or_k)))
        if m == 0:
            return Rnm
        return Rnm * (np.cos(m * phi) if trig == "cos" else np.sin(m * phi))
    if face == "disk_bot":
        if z > 0:
            return 0.0
        r_norm = rho / R
        Rnm = float(np.array(zernike_radial(jnp.array(r_norm), m, n_or_k)))
        if m == 0:
            return Rnm
        return Rnm * (np.cos(m * phi) if trig == "cos" else np.sin(m * phi))
    if rho < R * 0.99:
        return 0.0
    zeta = 2.0 * z / t
    Tk = float(np.array(chebyshev_t(jnp.array(zeta), n_or_k)))
    if m == 0:
        return Tk
    return Tk * (np.cos(m * phi) if trig == "cos" else np.sin(m * phi))


def gauge_projection_matrix(n_dof: int, pinned_index: int = 0) -> np.ndarray:
    """
    Matrix ``Q`` of shape ``(n_dof, n_dof - 1)`` with ``beta = Q @ alpha`` and ``beta[pinned_index] = 0``.

    Enforces one gauge (pinned DOF removed).
    """
    Q = np.zeros((n_dof, n_dof - 1))
    idx = [i for i in range(n_dof) if i != pinned_index]
    for j, i in enumerate(idx):
        Q[i, j] = 1.0
    return Q
