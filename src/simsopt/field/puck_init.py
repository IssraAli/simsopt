"""
Helpers for placing passive-bulk pucks on winding surfaces or toroidal shells.

See :class:`~simsopt.field.psc_bulk.PSCBulkArray` factory methods
:meth:`~simsopt.field.psc_bulk.PSCBulkArray.from_winding_surface`,
:meth:`~simsopt.field.psc_bulk.PSCBulkArray.from_toroidal_shell`, and
:meth:`~simsopt.field.psc_bulk.PSCBulkArray.from_curves`.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import numpy as np

from ..util.dipole_array_helper_functions import rotate_vector

__all__ = [
    "drop_overlapping_pucks",
    "toroidal_shell_pucks",
    "winding_surface_pucks",
    "curves_to_pucks",
]


def _normalize_size_menu(
    size_menu: Sequence[Sequence[float]] | None,
) -> tuple[tuple[float, float], ...] | None:
    """Validate and sort a (radius, thickness) menu in descending radius order.

    Returns ``None`` when ``size_menu`` is ``None``; otherwise a tuple of
    ``(radius, thickness)`` floats sorted by descending radius so the
    per-cell selection in :func:`toroidal_shell_pucks` can pick the
    largest-fitting entry with a simple linear scan.

    Raises:
        ValueError: any entry has fewer than 2 components, or non-positive
            ``radius`` / ``thickness``.
    """
    if size_menu is None:
        return None
    entries: list[tuple[float, float]] = []
    for idx, item in enumerate(size_menu):
        try:
            R_e, t_e = float(item[0]), float(item[1])
        except (TypeError, ValueError, IndexError) as exc:
            raise ValueError(
                f"size_menu[{idx}] must be a (radius, thickness) pair; got {item!r}"
            ) from exc
        if R_e <= 0.0:
            raise ValueError(
                f"size_menu[{idx}] radius must be positive; got {R_e}"
            )
        if t_e <= 0.0:
            raise ValueError(
                f"size_menu[{idx}] thickness must be positive; got {t_e}"
            )
        entries.append((R_e, t_e))
    entries.sort(key=lambda e: e[0], reverse=True)
    return tuple(entries)


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


def toroidal_shell_pucks(
    plasma_boundary,
    *,
    n_theta: int,
    n_phi: int,
    R0: Optional[float] = None,
    minor_radius: Optional[float] = None,
    plasma_clearance: float = 0.0,
    safety: float = 1.05,
    puck_R: Optional[Union[float, np.ndarray]] = None,
    puck_t: Optional[Union[float, np.ndarray]] = None,
    nfp: Optional[int] = None,
    stellsym: Optional[bool] = None,
    size_menu: Optional[Sequence[Sequence[float]]] = None,
    gap_poloidal: float = 0.0,
    gap_toroidal: float = 0.0,
    fill_fraction: float = 1.0,
    min_radius: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Build centers and inward axes on a single axisymmetric toroidal shell enveloping the plasma.

    The shell is parameterised in *simple toroidal coordinates*
    :math:`(\rho, \theta, \phi)` anchored at major radius ``R0`` (the magnetic
    axis):

    .. math::
        R(\theta, \phi) &= R_0 + \rho \cos\theta \\
        Z(\theta, \phi) &= \rho \sin\theta \\
        \text{center} &= (R \cos\phi, R \sin\phi, Z) \\
        \text{axis}   &= -(\cos\theta\cos\phi,\ \cos\theta\sin\phi,\ \sin\theta)

    so the puck axis points **inward** from the shell toward the plasma's
    magnetic axis.  No :class:`SurfaceClassifier` filtering is applied: the
    minor-radius default already includes ``plasma_clearance`` past the
    largest plasma rho.

    Two placement modes are supported:

    * **Scalar / uniform (``size_menu is None``)** -- legacy path.  Every
      puck has the same ``(puck_R, puck_t)`` and the spacing inequalities
      below are checked once against the inboard arc:

      * ``minor_radius * d_theta >= safety * 2R``      (poloidal arc on the shell)
      * ``(R0 - minor_radius) * d_phi >= safety * 2R`` (toroidal arc at the inboard side)
      * ``t <= minor_radius / 2``                       (puck thickness fits inside the shell)

    * **Mixed-size (``size_menu`` provided)** -- finer grid + per-cell
      selection.  Each cell ``(i_theta, i_phi)`` picks the **largest**
      ``(R, t)`` from ``size_menu`` that satisfies

      .. math::
          2 R &\le \texttt{fill\_fraction} \cdot a \cdot d_\theta
                  - 2\,\texttt{gap\_poloidal} \\
          2 R &\le \texttt{fill\_fraction} \cdot
                  R_\text{local} \cdot d_\phi - 2\,\texttt{gap\_toroidal} \\
          t   &\le a / 2,\quad R \ge \texttt{min\_radius}

      where :math:`R_\text{local} = R_0 + a \cos\theta_i` (so outboard
      cells, where :math:`R_\text{local}` is larger, can carry larger
      pucks than inboard cells).  Cells where no menu entry fits are
      dropped; the returned arrays are ragged with
      :math:`N \le n_\theta \cdot n_\phi`.  ``puck_R`` / ``puck_t`` /
      ``safety`` are ignored on this path.

    Args:
        plasma_boundary: ``SurfaceRZFourier`` used for default ``R0`` /
            ``minor_radius`` and the symmetry knobs.
        n_theta: Poloidal samples over the **full** :math:`[0, 2\pi)`
            range, **independent of ``stellsym``**.  The toroidal shell is
            naturally axisymmetric, so a full poloidal ring is the natural
            sample set.  Under stellsym, the downstream
            :class:`PSCBulkArray` replication still applies (mapping a
            base puck at :math:`(R\cos\phi, R\sin\phi, Z)` to a stellsym
            image at :math:`(R\cos\phi, -R\sin\phi, -Z)`, i.e. at
            toroidal angle :math:`-\phi` and poloidal angle
            :math:`2\pi-\theta`); these images do **not** coincide with
            the base set, so the per-toroidal-slice puck distribution
            ends up as a full poloidal ring rather than alternating
            ``Z>0`` / ``Z<0`` rings.
        n_phi: Toroidal samples per **half period** :math:`[0, \pi/\textrm{nfp})`.
        R0: Major radius of the shell (m).  ``None`` defaults to the mean
            ``R = sqrt(x^2 + y^2)`` of ``plasma_boundary.gamma()`` (matches
            the Fourier ``RBC(0,0)`` axis for axisymmetric boundaries).
        minor_radius: Minor radius of the shell (m).  ``None`` defaults to
            :math:`\max_{(\theta,\phi)} \sqrt{(R - R_0)^2 + Z^2} +
            \texttt{plasma\_clearance}` so the shell envelopes the plasma
            with ``plasma_clearance`` of headroom.  Must satisfy
            ``minor_radius < R0``.
        plasma_clearance: Added to the default ``minor_radius`` (m).  Has
            no effect when ``minor_radius`` is given explicitly.
        safety: Factor ``s`` in the **scalar-path** arc-spacing
            inequalities (``>= 1``).  Ignored on the mixed-size path;
            use ``fill_fraction`` / ``gap_*`` instead.
        puck_R: Scalar puck radius (m), or ``None`` for auto-sizing as
            ``0.5 / safety * min(arc_theta, arc_phi_inboard)``.  Ignored
            when ``size_menu`` is provided.
        puck_t: Scalar puck thickness (m), or ``None`` for auto-sizing as
            ``min(0.5 * minor_radius, 0.5 * R)``.  Ignored when
            ``size_menu`` is provided.
        nfp: Field periods; defaults to ``plasma_boundary.nfp``.
        stellsym: Stellarator symmetry; accepted for API compatibility but
            **does not change the base sampling**: theta is always sampled
            over :math:`[0, 2\pi)` because the toroidal shell is naturally
            axisymmetric.  Stellsym replication is still applied
            downstream by :class:`PSCBulkArray` according to its own
            ``stellsym`` flag.
        size_menu: Optional sequence of ``(radius, thickness)`` pairs for
            mixed-size placement.  Entries with non-positive ``radius`` or
            ``thickness`` raise :class:`ValueError`; otherwise the menu is
            sorted descending by radius internally so the per-cell
            selection picks the largest fitting size.  When ``None``, the
            scalar / uniform path is used.
        gap_poloidal: Extra clearance (m) subtracted from the cell
            poloidal arc before the ``2R`` check (mixed-size path only).
        gap_toroidal: Extra clearance (m) subtracted from the cell
            toroidal arc before the ``2R`` check (mixed-size path only).
        fill_fraction: Multiplier in ``(0, 1]`` on the cell's available
            arc (mixed-size path only).  ``1.0`` allows back-to-back
            placement; lower values reserve walking room between pucks.
        min_radius: Lower bound (m) on the per-cell puck radius
            (mixed-size path only).  Menu entries with
            ``radius < min_radius`` are skipped *and* a cell whose
            best-fitting entry is below ``min_radius`` is dropped.

    Returns:
        Tuple ``(centers, axes, radii, thicknesses)`` with shapes
        ``(N, 3)``, ``(N, 3)``, ``(N,)``, ``(N,)``.  These are the
        **base** pucks (before nfp / stellsym replication, which the
        downstream :class:`PSCBulkArray` handles).  ``N = n_theta * n_phi``
        on the scalar path and ``N <= n_theta * n_phi`` on the
        mixed-size path (cells where no menu entry fits are dropped).
    """
    nfp_i = int(nfp if nfp is not None else plasma_boundary.nfp)
    # The ``stellsym`` argument is accepted for API parity with
    # ``winding_surface_pucks`` and the downstream ``PSCBulkArray``
    # constructor, but the toroidal shell is naturally axisymmetric so we
    # always sample theta over the full ``[0, 2*pi)`` range.  Stellsym
    # replication is handled downstream and produces images at
    # ``(-phi, 2*pi - theta)`` rather than at ``(phi, -Z)``.
    _ = stellsym  # explicit no-op so linters don't flag the unused arg
    gamma = plasma_boundary.gamma()
    R_plasma = np.sqrt(gamma[..., 0] ** 2 + gamma[..., 1] ** 2)
    Z_plasma = gamma[..., 2]

    if R0 is None:
        R0 = float(R_plasma.mean())
    R0 = float(R0)
    if minor_radius is None:
        rho_max = float(np.sqrt((R_plasma - R0) ** 2 + Z_plasma ** 2).max())
        minor_radius = rho_max + float(plasma_clearance)
    minor_radius = float(minor_radius)
    if minor_radius <= 0.0:
        raise ValueError(
            f"toroidal_shell_pucks: minor_radius must be positive, got {minor_radius}"
        )
    if minor_radius >= R0:
        raise ValueError(
            f"toroidal_shell_pucks: minor_radius={minor_radius} must be < R0={R0} "
            f"(else the inboard side R = R0 - minor_radius collapses through the axis)."
        )
    if int(n_theta) <= 0 or int(n_phi) <= 0:
        raise ValueError(
            f"toroidal_shell_pucks: n_theta, n_phi must be positive ints; "
            f"got n_theta={n_theta}, n_phi={n_phi}"
        )

    theta_max = 2.0 * np.pi
    d_theta = theta_max / float(n_theta)
    d_phi = (np.pi / nfp_i) / float(n_phi)
    theta_arr = (
        np.linspace(0.0, theta_max, int(n_theta), endpoint=False) + d_theta / 2.0
    )
    phi_arr = (
        np.linspace(0.0, np.pi / nfp_i, int(n_phi), endpoint=False) + d_phi / 2.0
    )
    Tg, Pg = np.meshgrid(theta_arr, phi_arr, indexing="ij")  # (n_theta, n_phi)

    R_grid = R0 + minor_radius * np.cos(Tg)
    Z_grid = minor_radius * np.sin(Tg)
    centers_grid = np.stack(
        [R_grid * np.cos(Pg), R_grid * np.sin(Pg), Z_grid], axis=-1
    )
    axes_grid = -np.stack(
        [np.cos(Tg) * np.cos(Pg), np.cos(Tg) * np.sin(Pg), np.sin(Tg)],
        axis=-1,
    )

    menu = _normalize_size_menu(size_menu)

    if menu is not None:
        if not (0.0 < float(fill_fraction) <= 1.0):
            raise ValueError(
                f"toroidal_shell_pucks: fill_fraction must be in (0, 1]; "
                f"got {fill_fraction}"
            )
        if float(gap_poloidal) < 0.0 or float(gap_toroidal) < 0.0:
            raise ValueError(
                f"toroidal_shell_pucks: gap_poloidal/gap_toroidal must be >= 0; "
                f"got gap_poloidal={gap_poloidal}, gap_toroidal={gap_toroidal}"
            )
        ff = float(fill_fraction)
        gp = float(gap_poloidal)
        gt = float(gap_toroidal)
        rmin = float(min_radius)
        t_cap = 0.5 * minor_radius
        arc_theta_cell = minor_radius * d_theta
        keep_centers: list[np.ndarray] = []
        keep_axes: list[np.ndarray] = []
        keep_radii: list[float] = []
        keep_thicknesses: list[float] = []
        for it in range(int(n_theta)):
            theta_i = float(theta_arr[it])
            R_local = R0 + minor_radius * np.cos(theta_i)
            arc_phi_cell = R_local * d_phi
            cap_theta = ff * arc_theta_cell - 2.0 * gp
            cap_phi = ff * arc_phi_cell - 2.0 * gt
            cap = min(cap_theta, cap_phi)
            chosen: Optional[Tuple[float, float]] = None
            for R_e, t_e in menu:
                if R_e < rmin:
                    continue
                if 2.0 * R_e > cap + 1e-14:
                    continue
                if t_e > t_cap + 1e-14:
                    continue
                chosen = (R_e, t_e)
                break
            if chosen is None:
                continue
            R_e, t_e = chosen
            for ip in range(int(n_phi)):
                keep_centers.append(centers_grid[it, ip])
                keep_axes.append(axes_grid[it, ip])
                keep_radii.append(R_e)
                keep_thicknesses.append(t_e)
        if not keep_centers:
            return (
                np.zeros((0, 3), dtype=float),
                np.zeros((0, 3), dtype=float),
                np.zeros((0,), dtype=float),
                np.zeros((0,), dtype=float),
            )
        return (
            np.asarray(keep_centers, dtype=float),
            np.asarray(keep_axes, dtype=float),
            np.asarray(keep_radii, dtype=float),
            np.asarray(keep_thicknesses, dtype=float),
        )

    centers = centers_grid.reshape(-1, 3)
    axes = axes_grid.reshape(-1, 3)
    n = centers.shape[0]

    arc_theta = float(minor_radius * d_theta)
    arc_phi_min = float((R0 - minor_radius) * d_phi)
    if puck_R is None:
        R_val = 0.5 / float(safety) * min(arc_theta, arc_phi_min)
    else:
        R_val = float(puck_R)
    if puck_t is None:
        t_val = min(0.5 * minor_radius, 0.5 * R_val)
    else:
        t_val = float(puck_t)
    if safety * 2.0 * R_val > arc_theta + 1e-14:
        raise ValueError(
            f"Need minor_radius*d_theta >= safety*2R: "
            f"arc_theta={arc_theta}, safety*2R={safety * 2.0 * R_val}"
        )
    if safety * 2.0 * R_val > arc_phi_min + 1e-14:
        raise ValueError(
            f"Need (R0-minor_radius)*d_phi >= safety*2R: "
            f"arc_phi_inboard={arc_phi_min}, safety*2R={safety * 2.0 * R_val}"
        )
    if t_val > 0.5 * minor_radius + 1e-14:
        raise ValueError(
            f"Need t <= minor_radius/2: t={t_val}, minor_radius={minor_radius}"
        )
    radii = np.full(n, R_val)
    thicknesses = np.full(n, t_val)
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


def curves_to_pucks(
    curves: Sequence,
    *,
    radius: Optional[Union[float, np.ndarray]] = None,
    thickness: Optional[Union[float, np.ndarray]] = None,
    default_thickness: float = 0.02,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert a list of planar coil curves into puck placement arrays.

    Each curve (e.g. a :class:`~simsopt.geo.curveplanarfourier.CurvePlanarFourier`,
    such as those returned by
    :func:`~simsopt.util.dipole_array_helper_functions.generate_windowpane_metric_ring_array`,
    :func:`~simsopt.util.dipole_array_helper_functions.generate_windowpane_ring_array`,
    :func:`~simsopt.util.dipole_array_helper_functions.generate_windowpane_wedge_array`, or
    :func:`~simsopt.util.dipole_array_helper_functions.generate_windowpane_array`) must
    expose ``get('X')``/``get('Y')``/``get('Z')`` (its center) and
    ``get('q0')``/``get('qi')``/``get('qj')``/``get('qk')`` (its orientation
    quaternion). Puck orientation uses the same scalar-first quaternion
    convention as ``CurvePlanarFourier``, so each curve's own quaternion dofs are
    reused directly (by rotating the local +z axis) to get the puck axis.

    A puck is a right-circular cylinder, but the input curves can be elliptical
    or (super)square in cross section (e.g. from a windowpane generator with a
    large ``wp_n``), so the default per-curve radius is only an
    *approximation* of the curve's actual footprint: the area-equivalent
    radius, computed as the RMS distance of the curve's quadrature points
    (``curve.gamma()``) from its own center. Pass ``radius`` explicitly to
    override this (e.g. with each curve's own ``Rpol``/``Rtor``, or a fixed
    value) if the approximation isn't tight enough for your use case.

    Args:
        curves: sequence of planar coil curves, e.g. a list of
                ``CurvePlanarFourier`` objects.
        radius: scalar or per-curve array of puck radii (m). If ``None``
                (default), the area-equivalent radius is computed from each
                curve's own quadrature points.
        thickness: scalar or per-curve array of puck thicknesses (m). If
                   ``None`` (default), every puck gets ``default_thickness``.
        default_thickness: used when ``thickness`` is ``None``.

    Returns:
        Tuple ``(centers, axes, radii, thicknesses)`` with shapes ``(N, 3)``,
        ``(N, 3)``, ``(N,)``, ``(N,)`` -- the same layout
        :func:`toroidal_shell_pucks`/:func:`winding_surface_pucks` return, and
        what :class:`~simsopt.field.psc_bulk.PSCBulkArray` (or
        :meth:`~simsopt.field.psc_bulk.PSCBulkArray.from_curves`) expects.
    """
    n = len(curves)
    centers = np.zeros((n, 3))
    axes = np.zeros((n, 3))
    computed_radii = np.zeros(n)
    for i, curve in enumerate(curves):
        centers[i] = [curve.get("X"), curve.get("Y"), curve.get("Z")]
        q = np.array(
            [curve.get("q0"), curve.get("qi"), curve.get("qj"), curve.get("qk")]
        )
        q = q / np.linalg.norm(q)
        axes[i] = rotate_vector(np.array([0.0, 0.0, 1.0]), q)
        if radius is None:
            rel = curve.gamma() - centers[i]
            computed_radii[i] = np.sqrt(np.mean(np.sum(rel ** 2, axis=-1)))

    if radius is None:
        radii = computed_radii
    elif np.isscalar(radius):
        radii = np.full(n, float(radius))
    else:
        radii = np.asarray(radius, dtype=float)

    if thickness is None:
        thicknesses = np.full(n, float(default_thickness))
    elif np.isscalar(thickness):
        thicknesses = np.full(n, float(thickness))
    else:
        thicknesses = np.asarray(thickness, dtype=float)

    return centers, axes, radii, thicknesses
