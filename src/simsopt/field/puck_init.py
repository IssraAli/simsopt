"""
Helpers for placing passive-bulk pucks on winding surfaces or cylindrical lattices.

See :class:`~simsopt.field.psc_bulk.PSCBulkArray` factory methods
:meth:`~simsopt.field.psc_bulk.PSCBulkArray.from_winding_surface` and
:meth:`~simsopt.field.psc_bulk.PSCBulkArray.from_cylindrical_grid`.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np

from simsopt.geo.surface import SurfaceClassifier

__all__ = [
    "cylindrical_grid_pucks",
    "drop_overlapping_pucks",
    "winding_surface_pucks",
]


def drop_overlapping_pucks(
    centers: np.ndarray,
    axes: np.ndarray,
    radii: np.ndarray,
    thicknesses: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Greedy subset so sphere ``radius R + t/2`` around each centre is disjoint."""
    n = centers.shape[0]
    kept: list[int] = []
    for i in range(n):
        ci = centers[i]
        Ri = float(radii[i]) + 0.5 * float(thicknesses[i])
        ok = True
        for j in kept:
            cj = centers[j]
            rj = float(radii[j]) + 0.5 * float(thicknesses[j])
            if np.linalg.norm(ci - cj) <= Ri + rj + 1e-9:
                ok = False
                break
        if ok:
            kept.append(i)
    k = np.array(kept, dtype=int)
    return centers[k], axes[k], radii[k], thicknesses[k]


def cylindrical_grid_pucks(
    plasma_boundary,
    dr: float,
    dz: float,
    n_phi: int,
    *,
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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build centers and radial axes on a finite-shape-aware ``(r, φ, z)`` lattice.

    Enforces ``dr ≥ s·t``, ``dz ≥ s·2R``, and ``r_min·dφ ≥ s·2R`` before building.

    Args:
        plasma_boundary: ``SurfaceRZFourier`` (for bounds and optional plasma filter).
        dr: Radial spacing between cell centres (m).
        dz: Vertical spacing (m).
        n_phi: Number of toroidal slices per **half period** ``[0, π/nfp)``.
        r_min, r_max, z_min, z_max: Optional explicit bounds (m). Defaults use
            ``max(R_plasma)+d_inner``, etc., from ``plasma_boundary.gamma()``.
        d_inner: Added to ``max(R_plasma)`` when ``r_min`` is omitted.
        d_outer: Added for outer ``r_max`` / ``|z|_max`` when those are omitted.
        puck_R, puck_t: Scalar radii/thickness (m), or ``None`` for auto-sizing.
        safety: Factor ``s`` in the spacing inequalities.
        plasma_clearance: Extra margin (m) for :class:`SurfaceClassifier` rejection.
        nfp: Field periods; defaults to ``plasma_boundary.nfp``.

    Returns:
        Tuple ``(centers, axes, radii, thicknesses)`` with shapes ``(N, 3)``, ``(N, 3)``,
        ``(N,)``, ``(N,)``.
    """
    nfp = int(nfp if nfp is not None else plasma_boundary.nfp)
    gamma = plasma_boundary.gamma()
    R_plasma = np.sqrt(gamma[..., 0] ** 2 + gamma[..., 1] ** 2)

    if r_min is None:
        r_min = float(R_plasma.max()) + float(d_inner)
    if r_max is None:
        r_max = float(R_plasma.max()) + float(d_outer)
    if z_max is None:
        z_max = float(np.abs(gamma[..., 2]).max()) + float(d_outer)
    if z_min is None:
        z_min = -z_max

    dphi = np.pi / nfp / float(n_phi)
    r_arr = np.arange(r_min + dr / 2.0, r_max, dr, dtype=float)
    if r_arr.size == 0:
        raise ValueError("cylindrical_grid_pucks: empty r_arr; check r_min, r_max, dr")
    r_inner = float(r_arr[0])

    if puck_R is None:
        cap_phi = r_inner * dphi
        R_val = 0.5 / safety * min(dz, cap_phi)
    else:
        R_val = float(puck_R)

    if puck_t is None:
        t_val = dr / safety
    else:
        t_val = float(puck_t)

    if safety * t_val > dr + 1e-14:
        raise ValueError(
            f"Need dr >= safety*t: got dr={dr}, t={t_val}, safety={safety}",
        )
    if safety * 2.0 * R_val > dz + 1e-14:
        raise ValueError(
            f"Need dz >= safety*2R: got dz={dz}, R={R_val}, safety={safety}",
        )
    if safety * 2.0 * R_val > r_inner * dphi + 1e-14:
        raise ValueError(
            f"Need r_min*dphi >= safety*2R: got r_inner*dphi={r_inner * dphi}, "
            f"2R={2 * R_val}, safety={safety}",
        )

    phi_arr = np.linspace(0.0, np.pi / nfp, n_phi, endpoint=False) + dphi / 2.0
    z_arr = np.arange(z_min + dz / 2.0, z_max, dz, dtype=float)
    if z_arr.size == 0:
        raise ValueError("cylindrical_grid_pucks: empty z_arr; check z_min, z_max, dz")

    Rg, Pg, Zg = np.meshgrid(r_arr, phi_arr, z_arr, indexing="ij")
    centers = np.stack(
        [Rg * np.cos(Pg), Rg * np.sin(Pg), Zg],
        axis=-1,
    ).reshape(-1, 3)
    axes = np.stack(
        [np.cos(Pg), np.sin(Pg), np.zeros_like(Zg)],
        axis=-1,
    ).reshape(-1, 3)

    n = centers.shape[0]
    radii = np.full(n, R_val)
    thicknesses = np.full(n, t_val)

    margin = plasma_clearance + np.maximum(radii, thicknesses / 2.0)
    clf = SurfaceClassifier(plasma_boundary, p=2, h=0.05)
    sd = clf.evaluate_xyz(np.ascontiguousarray(centers)).ravel()
    keep = sd < -margin
    centers = centers[keep]
    axes = axes[keep]
    radii = radii[keep]
    thicknesses = thicknesses[keep]

    return centers, axes, radii, thicknesses


def winding_surface_pucks(
    plasma_boundary,
    distance: float,
    n_phi_pucks: int,
    n_theta_pucks: int,
    *,
    puck_R: Optional[Union[float, np.ndarray]] = None,
    puck_t: Optional[float] = None,
    default_thickness: float = 0.02,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Pucks on a surface obtained by ``extend_via_normal(distance)`` from the plasma copy.

    Axes align with the surface unit normal (caps tangent to the winding surface).

    Args:
        plasma_boundary: ``SurfaceRZFourier`` source equilibrium.
        distance: Normal offset (m) for :meth:`~simsopt.geo.surfacerzfourier.SurfaceRZFourier.extend_via_normal`.
        n_phi_pucks: Toroidal quadrature count (half-period range of the copy).
        n_theta_pucks: Poloidal quadrature count.
        puck_R: Optional scalar radius (m); if omitted, use local grid spacing.
        puck_t: Optional scalar thickness (m); default ``default_thickness``.
        default_thickness: Used when ``puck_t`` is ``None``.

    Returns:
        ``(centers, axes, radii, thicknesses)`` for **base** pucks (before ``nfp`` replication).

    Notes:
        The temporary winding surface uses the same toroidal range as the plasma (half period
        if ``stellsym`` else field period) and oversampled quadrature.  Overriding quadpoints to
        span a full field period on a half-period plasma makes the LSQ fit in
        ``extend_via_normal`` ill-conditioned and can yield unphysical coordinates.
    """
    # ``extend_via_normal`` requires enough quadrature points (see SurfaceRZFourier) and
    # sufficient oversampling for the internal Fourier fit (see Notes above).
    ntor = int(getattr(plasma_boundary, "ntor", 1))
    mpol = int(getattr(plasma_boundary, "mpol", 1))
    stellsym = bool(getattr(plasma_boundary, "stellsym", False))
    range_name = "half period" if stellsym else "field period"
    n_phi_hi = max(4 * (2 * ntor + 1), 2 * int(n_phi_pucks), 64)
    n_theta_hi = max(4 * (2 * mpol + 1), 2 * int(n_theta_pucks), 64)
    winding = plasma_boundary.copy(
        nphi=n_phi_hi,
        ntheta=n_theta_hi,
        range=range_name,
    )
    winding.extend_via_normal(float(distance))

    g = winding.gamma()
    n_hat = winding.unitnormal()
    ip = np.linspace(0, n_phi_hi - 1, int(n_phi_pucks)).astype(int)
    jt = np.linspace(0, n_theta_hi - 1, int(n_theta_pucks)).astype(int)
    centers = np.array([g[i, j] for i in ip for j in jt], dtype=float)
    axes = np.array([n_hat[i, j] for i in ip for j in jt], dtype=float)

    g1 = winding.gammadash1()
    g2 = winding.gammadash2()
    ds_phi = float(np.mean(np.linalg.norm(g1, axis=-1))) / float(n_phi_hi)
    ds_theta = float(np.mean(np.linalg.norm(g2, axis=-1))) / float(n_theta_hi)

    if puck_R is None:
        R_val = 0.45 * min(ds_phi, ds_theta)
    else:
        R_val = float(puck_R)

    if puck_t is None:
        t_val = min(R_val, 0.5 * float(distance), default_thickness)
    else:
        t_val = float(puck_t)

    n = centers.shape[0]
    radii = np.full(n, R_val)
    thicknesses = np.full(n, t_val)
    centers, axes, radii, thicknesses = drop_overlapping_pucks(
        centers,
        axes,
        radii,
        thicknesses,
    )
    return centers, axes, radii, thicknesses
