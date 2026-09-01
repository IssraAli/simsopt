#!/usr/bin/env python
"""
Sister module to tangent_extension.py: the exact same wedge-
construction algorithm, but with the corner LOCATION for every cross
section derived from ONE shared reference corner instead of each
cross section independently rediscovering its own via
corner_criterion's argmax.

Why: SurfaceBSpline blends control points BY INDEX across the
toroidal (v) direction -- control point row j in cross section i
connects to row j in cross section i + 1. tangent_extension.py's own
corner-finding (_find_corner_us) picks the apex as "whichever free
control point's Greville abscissa is nearest the corner's arc-length
position", decided INDEPENDENTLY per cross section via argmax(Z) or
argmax(curvature) on that cross section's own sampled curve. Since
that's a discrete, greedy, per-cross-section decision, an arbitrarily
small difference in the true corner location between two adjacent
cross sections can flip which INTEGER index gets treated as "the
apex" -- so row j silently means "the apex" in one cross section and
"a straight leg point" in its neighbor. That makes the toroidal
control net inconsistent at that row, which shows up as oscillation
in the full 3D surface even though every individual cross section's
own 2D wedge looks perfectly fine on its own. A cross section with an
unusually broad/flat top makes this worse: argmax(Z) itself can jump
around by more than one sample between cross sections from ordinary
discretization noise, before any control-point snapping even happens.

Fix: find the corner ONCE, as a local polar angle theta (in each
cross section's own local (x, y) plane -- SurfaceBSpline's cs_angle is
0 for every cross section in this surface's construction, so "the
same theta" really does mean "the same direction" everywhere, not
something that twists with a per-cross-section frame rotation -- see
_theta_of_u), on a single REFERENCE cross section (default: the
middle one). Every other cross section then locates "the same corner"
by finding where ITS OWN curve crosses that SAME theta (_theta_to_u,
the polar analogue of tangent_extension's own
_arclength_to_u/_u_to_arclength), instead of re-discovering a
possibly-different corner independently. This ties every cross
section's corner selection together through one shared reference, so
it can only vary as smoothly as the surface's own local theta(u)
functions do across the toroidal direction -- no more independent
per-cross-section rounding/argmax noise.

The reference cross section must be non-z_sym: a z_sym cross section
only ever has ONE (mirror-linked) corner (see tangent_extension.py's
module docstring), so it can't supply both the +Z and -Z reference
corners a non-z_sym cross section needs. Every z_sym cross section in
the surface still only gets reshaped once (using just the reference's
first corner), exactly as in tangent_extension.py.

Everything else -- walking s1/s2 arc length from the corner, the
tangent-line intersection, which control points get freed and how
they're placed on the two straight legs -- is exactly
tangent_extension.py's own algorithm (reused directly via
_corner_geometry_from_u/_apply_tangent_extension); only the corner-
LOCATING step changes.
"""

import matplotlib.pyplot as plt
import numpy as np

import tangent_extension as te


def _theta_of_u(samples):
    r"""
    Local polar angle theta = atan2(y, x) at every sample in
    samples['pos'] (local (x, y) plane), accumulated so it increases
    monotonically over one full loop of u -- valid for any star-shaped
    polar cross section (exactly what cs_basis="polar"'s theta_ctrl
    ordering already assumes throughout tangent_extension.py).
    Mirrors _cross_section_local_curve's own arc-length construction
    (`d = |diff(pos)|; s = cumsum(d)`), just with signed angular steps
    (wrapped into (-pi, pi] before summing) instead of unsigned
    Euclidean distances, and starting from theta[0] instead of 0.

    Returns (theta_cum, theta_total): theta_cum is an (n_samples,)
    array (theta_cum[0] == theta at u=0); theta_total is the value
    theta would reach at u=2*pi (closing the loop back to its own
    start) -- close to theta_cum[0] + 2*pi for a well-behaved simple
    polar curve, computed from the actual samples rather than assumed.
    """
    x, y = samples["pos"][:, 0], samples["pos"][:, 1]
    theta = np.arctan2(y, x)
    dtheta = np.diff(theta, append=theta[:1])
    dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi
    theta_cum = theta[0] + np.concatenate([[0.0], np.cumsum(dtheta)[:-1]])
    theta_total = theta[0] + np.cumsum(dtheta)[-1]
    return theta_cum, theta_total


def _u_to_theta(u, us, theta_cum, theta_total):
    """Interpolate theta_cum(us) at an arbitrary parameter value u --
    the polar analogue of tangent_extension._u_to_arclength."""
    n = len(us)
    u = u % (2 * np.pi)
    j = np.searchsorted(us, u)
    j0 = (j - 1) % n
    j1 = j % n
    u0 = us[j0]
    u1 = 2 * np.pi if j1 == 0 else us[j1]
    t0 = theta_cum[j0]
    t1 = theta_total if j1 == 0 else theta_cum[j1]
    if u1 <= u0:
        return t0
    frac = (u - u0) / (u1 - u0)
    return t0 + frac * (t1 - t0)


def _theta_to_u(theta_target, us, theta_cum, theta_total):
    """Inverse of _u_to_theta: the parameter u (in [0, 2*pi)) where
    this curve's own local polar angle equals theta_target (taken mod
    the curve's own net angular range, so any real value wraps around
    the closed curve) -- the polar analogue of
    tangent_extension._arclength_to_u."""
    n = len(us)
    theta0 = theta_cum[0]
    span = theta_total - theta0
    target = theta0 + ((theta_target - theta0) % span)
    j = np.searchsorted(theta_cum, target)
    j0 = (j - 1) % n
    j1 = j % n
    t0 = theta_cum[j0]
    t1 = theta_total if j1 == 0 else theta_cum[j1]
    u0 = us[j0]
    u1 = 2 * np.pi if j1 == 0 else us[j1]
    if t1 <= t0:
        return u0 % (2 * np.pi)
    frac = (target - t0) / (t1 - t0)
    return (u0 + frac * (u1 - u0)) % (2 * np.pi)


def _shared_corner_us(spline_surf, samples_list, corner_criterion, z_embed_list, reference_cs_index):
    """
    The corner parameter(s) u for EVERY cross section, all derived
    from a single reference corner found (the normal
    tangent_extension.py way) on spline_surf.cs_list[reference_cs_index]
    -- see module docstring.

    Returns a list (one entry per cross section) of lists of u values:
    length 1 for a z_sym cross section, 2 ([+Z, -Z], matching the
    reference's own order) otherwise -- same shape convention as
    tangent_extension._find_corner_us, just computed via a shared
    theta rather than each cross section's own independent argmax.
    """
    ref_cs = spline_surf.cs_list[reference_cs_index]
    if ref_cs.z_sym:
        raise ValueError(
            "reference_cs_index must name a non-z_sym cross section -- "
            "a z_sym cross section only has ONE (mirror-linked) corner, "
            "so it can't supply both the +Z and -Z reference corners a "
            "non-z_sym cross section needs. Pick a different "
            "reference_cs_index."
        )

    ref_samples = samples_list[reference_cs_index]
    ref_us_corner = te._find_corner_us(
        ref_samples, corner_criterion, False, z_embed_list[reference_cs_index]
    )
    ref_theta_cum, ref_theta_total = _theta_of_u(ref_samples)
    theta_refs = [
        _u_to_theta(u, ref_samples["us"], ref_theta_cum, ref_theta_total)
        for u in ref_us_corner
    ]

    corner_us_per_cs = []
    for i, cs in enumerate(spline_surf.cs_list):
        samples = samples_list[i]
        theta_cum, theta_total = _theta_of_u(samples)
        targets = theta_refs[:1] if cs.z_sym else theta_refs
        corner_us_per_cs.append(
            [_theta_to_u(t, samples["us"], theta_cum, theta_total) for t in targets]
        )
    return corner_us_per_cs


def tangent_extension_geometry_shared_corner(
    spline_surf, s1, s2, corner_criterion, reference_cs_index=None, n_samples=4000
):
    """
    tangent_extension.tangent_extension_geometry's read-only
    algorithm, EXCEPT every cross section's corner location(s) come
    from _shared_corner_us instead of each cross section's own
    independent _find_corner_us call.

    reference_cs_index : which cross section's corner to propagate to
        every other cross section (default: the middle one,
        spline_surf.n_cs // 2). Must be non-z_sym -- see
        _shared_corner_us.

    Returns the same shape tangent_extension_geometry does: a list
    (one entry per cross section) of lists of per-corner dicts (length
    1 for z_sym, 2 otherwise).
    """
    if corner_criterion not in te._VALID_CRITERIA:
        raise ValueError(
            f"corner_criterion must be one of {te._VALID_CRITERIA}, got "
            f"{corner_criterion!r}"
        )
    if reference_cs_index is None:
        reference_cs_index = spline_surf.n_cs // 2

    samples_list = [
        te._cross_section_local_curve(spline_surf, cs, n_samples)
        for cs in spline_surf.cs_list
    ]
    z_embed_list = te._build_z_embed_list(spline_surf)

    corner_us_per_cs = _shared_corner_us(
        spline_surf, samples_list, corner_criterion, z_embed_list, reference_cs_index
    )

    return [
        [te._corner_geometry_from_u(samples, u_corner, s1, s2) for u_corner in us_corner]
        for samples, us_corner in zip(samples_list, corner_us_per_cs)
    ]


def tangent_extension_shared_corner(
    spline_surf, s1, s2, corner_criterion, reference_cs_index=None, n_samples=4000
):
    """
    tangent_extension.tangent_extension, but every cross section's
    corner comes from the single shared reference corner
    tangent_extension_geometry_shared_corner computes -- see module
    docstring. Mutates spline_surf in place (via
    tangent_extension._apply_tangent_extension, the exact same
    freeing/repositioning/guards tangent_extension itself uses) and
    returns it.
    """
    geometry = tangent_extension_geometry_shared_corner(
        spline_surf,
        s1,
        s2,
        corner_criterion,
        reference_cs_index=reference_cs_index,
        n_samples=n_samples,
    )
    return te._apply_tangent_extension(spline_surf, geometry)


def wedge_opening_angle_residuals_shared_corner(
    spline_surf,
    s1,
    s2,
    corner_criterion,
    goal_angle,
    reference_cs_index=None,
    n_samples=4000,
):
    """
    tangent_extension.wedge_opening_angle_residuals' shared-corner
    counterpart -- one residual per corner, same cross-section-then-
    corner ordering.
    """
    geometry = tangent_extension_geometry_shared_corner(
        spline_surf,
        s1,
        s2,
        corner_criterion,
        reference_cs_index=reference_cs_index,
        n_samples=n_samples,
    )
    return np.array(
        [g["opening_angle"] - goal_angle for cs_corners in geometry for g in cs_corners]
    )


def print_apex_indices(spline_surf, geometry, label):
    """
    Print, per cross section, which FULL control-point index each
    corner's apex would snap to (tangent_extension._apex_full_index) --
    the diagnostic this whole module exists to fix: watch this
    sequence for tangent_extension_geometry's own (independent-argmax)
    geometry and it'll often jump around erratically between adjacent
    cross sections; the same printout for
    tangent_extension_geometry_shared_corner's geometry should vary
    smoothly instead.
    """
    print(f"apex full index per cross section ({label}):")
    for i, cs_corners in enumerate(geometry):
        idxs = [te._apex_full_index(g) for g in cs_corners]
        print(f"  cross section {i}: {idxs}")


if __name__ == "__main__":
    from simsopt.geo import SurfaceBSpline

    dofs = [
        0.20331546, 0.15746095, 0.43573368, 0.14817898, 0.88822389,
        1.9522663, 0.14087834, 0.24736301, 0.42883166, 0.15160325,
        0.45816498, 0.14868365, 0.59658694, 1.7730913, 2.61799388,
        4.23531002, 4.71238898, 0.07372945, 0.2838024, 0.51265284,
        0.21184257, 0.46871346, 0.12405194, 0.62203555, 1.57720828,
        3.66519143, 4.31988537, 4.71238898, 0.07226604, 0.40665305,
        0.48766582, 0.22361523, 0.53481562, 0.04923251, 1.02054689,
        1.57079633, 3.66519143, 4.47219325, 4.71238898, 0.02003046,
        0.43141331, 0.50441053, 0.22720611, 0.58282917, 0.17844745,
        1.12887769, 1.57079633, 3.66519143, 4.66296652, 4.71238898,
        0.0213662, 0.44783741, 0.51468795, 0.18259442, 1.37237811,
        1.57079633, 0.95657065, 1.10700731, 1.21218964, 0.012124,
    ]
    spline_kwargs = {
        "axis_points": 3, "points_per_cs": 6, "n_cs": 6, "nfp": 2,
        "M": 12, "N": 12, "p_u": 3, "p_v": 3, "cs_equispaced": True,
        "rays_equispaced": False, "cs_global_angle_free": False,
        "axis_angles_fixed": True, "cs_basis": "polar", "nurbs": False,
        "use_bishop_frame": True, "knot_parametrization": "uniform",
    }

    spline_surf = SurfaceBSpline(**spline_kwargs)
    spline_surf.x = dofs
    for _ in range(3):
        spline_surf.refine_poloidal()

    # A separate, un-reshaped copy for the before/after comparison --
    # NOT list(spline_surf.cs_list) (see sharpen.py's plot_before_after
    # docstring for why that aliases the objects tangent_extension is
    # about to mutate in place).
    spline_surf_before = SurfaceBSpline(**spline_kwargs)
    spline_surf_before.x = dofs
    spline_surf_before.refine_poloidal()
    spline_surf_before.refine_poloidal()
    spline_surf_before.refine_poloidal()
    spline_surf_before.fix_all()

    perimeters = [
        te._cross_section_local_curve(spline_surf, cs, n_samples=2000)["L"]
        for cs in spline_surf.cs_list
    ]
    S1 = 0.15 * min(perimeters)
    S2 = 0.10 * min(perimeters)
    CORNER_CRITERION = "max z"
    REFERENCE_CS_INDEX = spline_surf.n_cs // 2

    # Side-by-side diagnostic: independent-per-cross-section
    # corner-finding (tangent_extension.py) vs. this module's shared
    # reference corner, on the SAME (still unmodified) surface.
    independent_geometry = te.tangent_extension_geometry(
        spline_surf, S1, S2, CORNER_CRITERION
    )
    shared_geometry = tangent_extension_geometry_shared_corner(
        spline_surf, S1, S2, CORNER_CRITERION, reference_cs_index=REFERENCE_CS_INDEX
    )
    print_apex_indices(spline_surf, independent_geometry, "independent, tangent_extension.py")
    print_apex_indices(spline_surf, shared_geometry, "shared reference corner")

    te.plot_tangent_extension(
        spline_surf, S1, S2, CORNER_CRITERION, geometry=shared_geometry
    )
    plt.show()

    tangent_extension_shared_corner(
        spline_surf, S1, S2, CORNER_CRITERION, reference_cs_index=REFERENCE_CS_INDEX
    )

    from sharpen import plot_before_after

    plot_before_after(spline_surf, spline_surf_before)

    print("End of figures_spline_paper/corner_sharpening/tangent_extension_shared_corner.py")
