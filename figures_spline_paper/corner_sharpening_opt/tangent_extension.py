#!/usr/bin/env python
"""
"Tangent extension": a purely geometric, deterministic alternative to
corner_angle_residuals (sharpen.py) for shaping a SurfaceBSpline's
cross-section corners into a wedge -- instead of freeing a
neighborhood of control points and letting a least-squares solve pull
them into a straight-legs-plus-apex shape (which needed careful
regularization/bounds to avoid oscillation, see sharpen.py's module
docstring), this constructs the wedge directly:

1. Find "the corner(s)" of a cross section -- its own periodic
   B-spline curve's point of highest Z (corner_criterion="max z") or
   highest in-plane curvature (corner_criterion="curvature"). A z_sym
   cross section only ever gets ONE corner found/reshaped this way:
   its mirror symmetry means the "other" (mirrored) corner is fully
   determined by the same control points already, so reshaping one
   automatically reshapes both (verified numerically -- see
   tangent_extension's docstring). A non-z_sym cross section has no
   such constraint, so its +Z and -Z corners are found and reshaped
   INDEPENDENTLY: the curve is split into a +Z half and -Z half at the
   midpoint between its own max and min Z (same convention as
   sharpen.py's max_curvature_by_z_half_residuals), and the corner
   criterion is applied separately within each half.
2. Walk arc length s1 backward and s2 forward from the corner along
   that same curve, landing on two points P_minus, P_plus (generally
   NOT existing control points).
3. Take the curve's own tangent line at each of P_minus, P_plus, and
   intersect them -- this is where the corner would sit if the curve
   were truly two straight legs meeting at a point, instead of the
   smooth (rounded) corner it currently has.
4. Free (unfix) every control point whose Greville abscissa -- see
   _greville_abscissae -- falls strictly between P_minus and P_plus.
   Exactly one of them (the one nearest the original corner
   parameter) gets moved onto the new tangent-intersection point; the
   rest get sled onto the two straight P_minus-apex / apex-P_plus
   segments, at the same relative arc-length fraction along their leg
   they started at. P_minus and P_plus themselves, and every other
   control point, are untouched.
5. Return the (mutated) SurfaceBSpline.

The assumption (from the user) is that Lane-Riesenfeld doubling
(SurfaceBSpline.refine_poloidal) has already been called enough times
that the control net closely tracks the curve it represents -- dense
enough that "the control points between P_minus and P_plus" is a
well-posed, useful notion, and that exactly one control point sits
close enough to the corner to unambiguously become the new apex
(rather than two or more competing for the role).

Unlike corner_angle_residuals, s1 and s2 (not the individual control
point positions) are the only meaningful degrees of freedom here --
the whole local shape is a deterministic function of them. The
intended use is to run an outer optimization over just s1 and s2 (one
pair per corner, or shared across every corner), with objectives such
as quasisymmetry (recomputed from a fresh VMEC/booundary built from
the reshaped surface) and wedge_opening_angle_residuals below (holding
a specific angle between the two tangents) -- NOT to free the
individual control points touched here for a many-dof least-squares
solve the way sharpen.py does.

tangent_extension_geometry (read-only, no mutation) exposes the
per-corner P_minus/P_plus/tangents/apex/opening angle that
tangent_extension computes internally, both for
wedge_opening_angle_residuals and for inspection/plotting
(plot_tangent_extension). It returns a list (one entry per cross
section) of lists of per-corner dicts -- length 1 for a z_sym cross
section, length 2 ([+Z corner, -Z corner]) otherwise.
"""

import matplotlib.pyplot as plt
import numpy as np
from simsopt.util.spline_helpers import (
    b_p_deriv2,
    chord_length_knots,
    uniform_knots,
)

_VALID_CRITERIA = ("max z", "curvature")


def _periodic_curve_deriv2(
    core, weights, p, knot_parametrization, u, domain=2 * np.pi
):
    r"""
    Position, first derivative, and second derivative (all w.r.t. the
    curve parameter) of a periodic (closed-loop), rational degree-p
    B-spline curve at explicit, arbitrary parameter value(s) `u` --
    same control-point tiling/knot convention as
    simsopt.util.spline_helpers.eval_periodic_curve (which this
    reduces to, for position, at u = an evenly-spaced sample grid --
    verified numerically to match eval_periodic_curve exactly, and its
    derivatives against finite differences), generalized to give
    derivatives at arbitrary u instead of only positions at an
    evenly-spaced grid.

    C(u) = N(u) / D(u) with N(u) = sum_i B_i(u) w_i P_i (vector),
    D(u) = sum_i B_i(u) w_i (scalar, the rational denominator) --
    standard quotient-rule combination for a NURBS curve's derivatives
    from the basis/weight derivatives b_p_deriv2 provides.

    core : (m, dim) array of control points, periodic (point m aliases
        point 0).
    weights : (m,) array of NURBS weights.
    p : degree.
    knot_parametrization : 'uniform' or 'chord'.
    u : scalar or (N,) array of parameter values (any real number --
        wrapped into [0, domain) internally, so periodicity is
        automatic).

    Returns (pos, vel, acc), each (N, dim) (or (dim,) if `u` was a
    scalar collapsed to a length-1 array by the caller -- callers here
    always pass an array).
    """
    m = len(core)
    padded = np.concatenate([core[-p:], core, core[:p]], axis=0)
    padded_w = np.concatenate([weights[-p:], weights, weights[:p]])
    if knot_parametrization == "uniform":
        t = uniform_knots(m - 1, p, domain=domain)
    elif knot_parametrization == "chord":
        t = chord_length_knots(core, p, domain=domain)
    else:
        raise ValueError(
            "knot_parametrization must be 'uniform' or 'chord', got "
            f"{knot_parametrization!r}"
        )
    u = np.mod(np.asarray(u, dtype=float), domain)
    basis, d1, d2 = b_p_deriv2(t, p, u)

    Pw = padded * padded_w[:, None]
    N = basis @ Pw
    D = basis @ padded_w
    Np = d1 @ Pw
    Dp = d1 @ padded_w
    Npp = d2 @ Pw
    Dpp = d2 @ padded_w

    pos = N / D[:, None]
    vel = (Np * D[:, None] - N * Dp[:, None]) / (D**2)[:, None]
    acc = (
        Npp * (D**2)[:, None]
        - N * Dpp[:, None] * D[:, None]
        - 2 * Dp[:, None] * (Np * D[:, None] - N * Dp[:, None])
    ) / (D**3)[:, None]
    return pos, vel, acc


def _full_to_stored_index_map(cs):
    """
    Map each of a cross section's n_ctrl_pts full (mirrored, for
    z_sym) physical control-point indices -- the ordering
    get_r_ctrl_full/get_theta_ctrl_full/cross_section_xy's ctrl_xy all
    use -- back to the stored dof index (the "r_j"/"theta_j" name
    suffix) that actually owns it. Applies the exact same slicing
    get_r_ctrl_full itself uses, to indices instead of values, so it
    reproduces the identical mirroring by construction rather than by
    a separately hand-derived formula.
    """
    idx = np.arange(cs.n_pts)
    if cs.z_sym:
        if cs.n_ctrl_pts % 2 == 1:
            return np.concatenate([idx, idx[:0:-1]])
        else:
            return np.concatenate([idx, idx[-2:0:-1]])
    return idx


def _greville_abscissae(m, p, knot_parametrization, core, domain=2 * np.pi):
    r"""
    Greville abscissa (Piegl & Tiller, "The NURBS Book", eq. 9.7) of
    each of a periodic degree-p B-spline curve's `m` original control
    points -- \xi_j = (t_{j+1} + ... + t_{j+p}) / p for basis function
    j, evaluated at j = the padded-array basis index (p + i) that
    original point i (0..m-1) sits at, using the exact same knot
    vector eval_periodic_curve/_periodic_curve_deriv2 evaluate the
    curve with.

    The Greville abscissa is the standard notion of "where along the
    curve's parameter domain does this control point's influence sit"
    -- for a uniform knot vector it's exactly the point's own
    evenly-spaced parameter position (verified numerically); for
    'chord' it inherits the same uneven parametrization the curve
    itself uses. This is what tangent_extension uses to decide which
    control points fall inside the [P_minus, P_plus] arc-length
    window, and to place the ones that do at the right point along
    their straight leg.

    Returns an (m,) array, values wrapped into [0, domain).
    """
    if knot_parametrization == "uniform":
        t = uniform_knots(m - 1, p, domain=domain)
    elif knot_parametrization == "chord":
        t = chord_length_knots(core, p, domain=domain)
    else:
        raise ValueError(
            "knot_parametrization must be 'uniform' or 'chord', got "
            f"{knot_parametrization!r}"
        )
    greville = np.array(
        [np.mean(t[p + i + 1 : 2 * p + i + 1]) for i in range(m)]
    )
    return np.mod(greville, domain)


def _cross_section_local_curve(spline_surf, cs, n_samples):
    """
    Everything tangent_extension/_geometry need about one cross
    section's own periodic B-spline curve, in its local (x, y) plane
    (same convention as SurfaceBSpline.cross_section_xy -- no cs_angle
    rotation): the full (mirrored, for z_sym) control polygon, degree,
    knot_parametrization, an evenly-spaced parameter sample grid and
    the curve's position/velocity/acceleration there, each control
    point's Greville abscissa, and the sampled curve's cumulative arc
    length + total perimeter.

    Returns a dict with keys: core, w, p, kp, us, pos, vel, acc,
    greville, s, L.
    """
    full_r = cs.get_r_ctrl_full()
    full_theta = cs.get_theta_ctrl_full()
    full_w = cs.get_w_ctrl_full()
    core = np.stack(
        [full_r * np.cos(full_theta), full_r * np.sin(full_theta)], axis=1
    )
    p = spline_surf.p_u
    kp = spline_surf.knot_parametrization
    us = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
    pos, vel, acc = _periodic_curve_deriv2(core, full_w, p, kp, us)
    greville = _greville_abscissae(len(core), p, kp, core)

    d = np.linalg.norm(np.diff(pos, axis=0, append=pos[:1]), axis=1)
    s = np.concatenate([[0.0], np.cumsum(d)[:-1]])
    L = s[-1] + d[-1]

    return {
        "core": core,
        "w": full_w,
        "p": p,
        "kp": kp,
        "us": us,
        "pos": pos,
        "vel": vel,
        "acc": acc,
        "greville": greville,
        "s": s,
        "L": L,
    }


def _u_to_arclength(u, us, s, L):
    """Interpolate the sampled curve's cumulative arc length s(us) at
    an arbitrary parameter value u (wrapped into [0, 2*pi)), assuming
    `us` is the evenly-spaced, ascending sample grid `s` was built
    from (see _cross_section_local_curve)."""
    n = len(us)
    u = u % (2 * np.pi)
    j = np.searchsorted(us, u)
    j0 = (j - 1) % n
    j1 = j % n
    u0 = us[j0]
    u1 = 2 * np.pi if j1 == 0 else us[j1]
    s0 = s[j0]
    s1 = L if j1 == 0 else s[j1]
    if u1 <= u0:
        return s0
    frac = (u - u0) / (u1 - u0)
    return s0 + frac * (s1 - s0)


def _arclength_to_u(s_target, us, s, L):
    """Inverse of _u_to_arclength: the parameter u (in [0, 2*pi)) at
    cumulative arc length s_target (taken mod L, so any real value --
    including negative, or beyond L -- wraps around the closed
    curve)."""
    n = len(us)
    s_target = s_target % L
    j = np.searchsorted(s, s_target)
    j0 = (j - 1) % n
    j1 = j % n
    s0 = s[j0]
    s1 = L if j1 == 0 else s[j1]
    u0 = us[j0]
    u1 = 2 * np.pi if j1 == 0 else us[j1]
    if s1 <= s0:
        return u0 % (2 * np.pi)
    frac = (s_target - s0) / (s1 - s0)
    return (u0 + frac * (u1 - u0)) % (2 * np.pi)


def _curve_curvature(vel, acc):
    """In-plane curvature kappa = |x'y'' - y'x''| / (x'^2+y'^2)^1.5 of
    a 2D curve, given its (N, 2) velocity/acceleration samples --
    invariant under rigid rotation+translation, so it needs no
    lab-frame embedding (unlike "max z")."""
    num = np.abs(vel[:, 0] * acc[:, 1] - vel[:, 1] * acc[:, 0])
    den = (vel[:, 0] ** 2 + vel[:, 1] ** 2) ** 1.5
    return num / den


def _find_corner_us(samples, corner_criterion, z_sym, z_embed):
    """
    The curve parameter(s) u (in [0, 2*pi)) of "the corner(s)" of one
    cross section's curve, per corner_criterion -- see the module
    docstring for why a z_sym cross section only ever gets ONE corner
    (mirror symmetry ties the other one to it automatically) while a
    non-z_sym cross section gets its +Z and -Z corners found
    independently, each restricted to its own half of the loop (split
    at the midpoint between the loop's own max and min Z, same
    convention as sharpen.py's max_curvature_by_z_half_residuals).

    z_embed(pos) -- pos being samples['pos'], (N, 2) local (x, y) --
    must return the (N,) array of real lab-frame Z at those points
    (see tangent_extension_geometry for the embedding, which needs
    this cross section's own cs_angle/axis_local_basis and so can't be
    computed from `samples` alone). Always required (even for
    corner_criterion="curvature") when z_sym is False, to determine
    the +Z/-Z split; unused when z_sym is True and corner_criterion is
    "curvature".

    Returns a list of floats: length 1 (z_sym) or 2 ([+Z corner's u,
    -Z corner's u], not z_sym).
    """
    us, vel, acc = samples["us"], samples["vel"], samples["acc"]

    if corner_criterion == "max z":
        metric = z_embed(samples["pos"])
    elif corner_criterion == "curvature":
        metric = _curve_curvature(vel, acc)
    else:
        raise ValueError(
            f"corner_criterion must be one of {_VALID_CRITERIA}, got "
            f"{corner_criterion!r}"
        )

    if z_sym:
        return [us[int(np.argmax(metric))]]

    z = z_embed(samples["pos"])
    z_mid = 0.5 * (z.max() + z.min())
    plus_mask = z >= z_mid
    minus_mask = ~plus_mask

    if corner_criterion == "max z":
        u_plus = us[plus_mask][int(np.argmax(metric[plus_mask]))]
        u_minus = us[minus_mask][int(np.argmin(metric[minus_mask]))]
    else:
        u_plus = us[plus_mask][int(np.argmax(metric[plus_mask]))]
        u_minus = us[minus_mask][int(np.argmax(metric[minus_mask]))]
    return [u_plus, u_minus]


def _line_intersection(P1, T1, P2, T2):
    """
    Intersection of the two (infinite) 2D lines through P1 (direction
    T1) and P2 (direction T2). Raises ValueError if the lines are
    (numerically) parallel -- the two tangents pointing the same way
    means there's no well-defined corner to extend to (s1/s2 chosen
    too small/large, or the curve is already locally straight).
    """
    A = np.array([T1, -T2]).T
    if abs(np.linalg.det(A)) < 1e-12:
        raise ValueError(
            "tangent lines at P_minus and P_plus are (nearly) parallel -- "
            "no well-defined intersection; try different s1/s2."
        )
    ab = np.linalg.solve(A, P2 - P1)
    return P1 + ab[0] * T1


def _build_z_embed_list(spline_surf):
    """
    One z_embed closure per cross section in spline_surf.cs_list --
    z_embed(xy), xy being local (x, y) points ((N, 2)), returns the
    (N,) array of real lab-frame Z those points sit at, via this cross
    section's own cs_angle/axis_local_basis (SurfaceBSpline's rotation
    from a cross section's local plane into the lab frame). Needed by
    _find_corner_us' "max z" criterion, and by any caller (e.g.
    tangent_extension_shared_corner.py) that wants to locate a corner
    by real Z on more than one cross section without recomputing
    cs_angle/axis_pos/e1/e2 per corner.
    """
    cs_zeta, cs_angle = spline_surf.get_cs_zeta_angle()
    axis_pos, e1, e2 = spline_surf._axis_local_basis(cs_zeta)

    def make_z_embed(i):
        def z_embed(xy, i=i):
            ang = cs_angle[i]
            x, y = xy[:, 0], xy[:, 1]
            xr = x * np.cos(ang) - y * np.sin(ang)
            yr = x * np.sin(ang) + y * np.cos(ang)
            return axis_pos[i, 2] + xr * e1[i, 2] + yr * e2[i, 2]

        return z_embed

    return [make_z_embed(i) for i in range(spline_surf.n_cs)]


def _corner_geometry_from_u(samples, u_corner, s1, s2):
    """
    One corner's full tangent-extension geometry dict -- P_minus,
    P_plus (the arc-length-s1/s2-back points from u_corner along
    `samples`' own curve), their unit tangents, the new apex
    (tangent-line intersection), and the opening angle at the apex --
    given an ALREADY-DETERMINED curve parameter u_corner, regardless
    of how it was found (tangent_extension_geometry's own
    _find_corner_us, or a shared/propagated reference corner from
    another cross section -- see tangent_extension_shared_corner.py).

    Returns a dict with keys P_minus, P_plus, T_minus, T_plus, apex,
    opening_angle, u_corner, u_minus, u_plus, samples (the caller's
    own `samples`, threaded through so downstream code -- e.g.
    tangent_extension's mutation step -- doesn't need it passed
    separately).
    """
    us, s, L = samples["us"], samples["s"], samples["L"]
    s_corner = _u_to_arclength(u_corner, us, s, L)
    u_minus = _arclength_to_u(s_corner - s1, us, s, L)
    u_plus = _arclength_to_u(s_corner + s2, us, s, L)

    pts, vels, _ = _periodic_curve_deriv2(
        samples["core"],
        samples["w"],
        samples["p"],
        samples["kp"],
        np.array([u_minus, u_plus]),
    )
    P_minus, P_plus = pts
    T_minus = vels[0] / np.linalg.norm(vels[0])
    T_plus = vels[1] / np.linalg.norm(vels[1])

    apex = _line_intersection(P_minus, T_minus, P_plus, T_plus)
    d_minus = P_minus - apex
    d_plus = P_plus - apex
    opening_angle = np.arccos(
        np.clip(
            np.dot(d_minus, d_plus)
            / (np.linalg.norm(d_minus) * np.linalg.norm(d_plus)),
            -1.0,
            1.0,
        )
    )

    return {
        "P_minus": P_minus,
        "P_plus": P_plus,
        "T_minus": T_minus,
        "T_plus": T_plus,
        "apex": apex,
        "opening_angle": opening_angle,
        "u_corner": u_corner,
        "u_minus": u_minus,
        "u_plus": u_plus,
        "samples": samples,
    }


def _apex_full_index(g):
    """
    Which FULL control-point index (see _cross_section_local_curve's
    ordering) _apply_tangent_extension would snap onto the apex for
    one corner geometry dict `g` -- read-only, no guards, no mutation
    (unlike _apply_tangent_extension's own version of this
    computation, which also validates the free window against z_sym
    pinning/overlap before using it). Returns None if the corner's
    window happens to contain no free control points at all.

    Useful on its own as a diagnostic: e.g.
    tangent_extension_shared_corner.py prints this per cross section
    to show that the shared-reference corner varies smoothly across
    the toroidal direction, unlike tangent_extension's own
    independent-per-cross-section corner-finding (see that module's
    docstring for why the latter can flip to a different integer
    index between adjacent cross sections for an arbitrarily small
    change in the true corner location).
    """
    samples = g["samples"]
    greville = samples["greville"]
    u_minus, u_plus, u_corner = g["u_minus"], g["u_plus"], g["u_corner"]
    span = (u_plus - u_minus) % (2 * np.pi)
    rel = (greville - u_minus) % (2 * np.pi)
    free_idx = np.nonzero((rel > 0) & (rel < span))[0]
    if len(free_idx) == 0:
        return None
    corner_rel = (u_corner - u_minus) % (2 * np.pi)
    return int(free_idx[np.argmin(np.abs(rel[free_idx] - corner_rel))])


def tangent_extension_geometry(
    spline_surf, s1, s2, corner_criterion, n_samples=4000
):
    """
    Read-only companion to tangent_extension: for every cross section
    in spline_surf.cs_list, compute (without modifying spline_surf)
    the corner-extension geometry corner_criterion/s1/s2 imply --
    P_minus, P_plus (the arc-length-s1/s2-back points), their unit
    tangents, the new apex (tangent-line intersection), and the wedge
    opening angle at the apex (the angle between apex->P_minus and
    apex->P_plus).

    Returns a list (one entry per cross section) of lists of dicts
    (length 1 for a z_sym cross section, 2 -- [+Z corner, -Z corner]
    -- otherwise), each dict with keys P_minus, P_plus, T_minus,
    T_plus, apex, opening_angle, u_corner, u_minus, u_plus, samples
    (the _cross_section_local_curve dict, for callers -- e.g.
    tangent_extension, plot_tangent_extension -- that need the sampled
    curve/Greville abscissae too, without recomputing them).
    """
    if corner_criterion not in _VALID_CRITERIA:
        raise ValueError(
            f"corner_criterion must be one of {_VALID_CRITERIA}, got "
            f"{corner_criterion!r}"
        )

    z_embed_list = _build_z_embed_list(spline_surf)

    geometry = []
    for i, cs in enumerate(spline_surf.cs_list):
        samples = _cross_section_local_curve(spline_surf, cs, n_samples)
        us_corner = _find_corner_us(
            samples, corner_criterion, cs.z_sym, z_embed_list[i]
        )
        cs_corners = [
            _corner_geometry_from_u(samples, u_corner, s1, s2)
            for u_corner in us_corner
        ]
        geometry.append(cs_corners)
    return geometry


def wedge_opening_angle_residuals(
    spline_surf, s1, s2, corner_criterion, goal_angle, n_samples=4000
):
    """
    One residual per corner (one per z_sym cross section, two --
    [+Z, -Z] -- per non-z_sym cross section, in cross-section-then-
    corner order): tangent_extension_geometry's opening_angle there,
    minus `goal_angle` -- a LeastSquaresProblem-ready stand-in for
    corner_angle_residuals' tip-angle term (sharpen.py), but as a
    function of (s1, s2) alone, for an outer optimization that
    reshapes via tangent_extension instead of freeing individual
    control points.

    Does not call tangent_extension / mutate spline_surf -- evaluates
    the same geometry tangent_extension would construct, without
    actually moving any control points.

    Returns a flat array, one entry per corner.
    """
    geometry = tangent_extension_geometry(
        spline_surf, s1, s2, corner_criterion, n_samples=n_samples
    )
    return np.array(
        [
            g["opening_angle"] - goal_angle
            for cs_corners in geometry
            for g in cs_corners
        ]
    )


def _apply_tangent_extension(spline_surf, geometry):
    r"""
    The mutating half of tangent_extension: given an ALREADY-COMPUTED
    `geometry` (tangent_extension_geometry's return value, or
    tangent_extension_shared_corner.py's equivalent -- any list of
    lists of _corner_geometry_from_u-shaped dicts, one list per cross
    section in spline_surf.cs_list), free and reposition exactly the
    control points each corner's construction implies, and return
    spline_surf. Doesn't care how `geometry`'s corners were found, so
    it's shared verbatim by both the independent-per-cross-section
    corner-finding tangent_extension does and
    tangent_extension_shared_corner.py's shared-reference-corner
    variant.

    Guards (raises ValueError) against the ways two corners' freed
    windows could otherwise silently collide: a z_sym cross section's
    single window reaching either of its own mirror-axis-pinned points
    (full index 0, or n_pts - 1 for even n_ctrl_pts -- see
    CrossSectionFixedZeta.__init__) or wrapping far enough around the
    loop to need two different target positions for the same
    (mirror-linked) control point; and, for a non-z_sym cross
    section's independent +Z/-Z corners, their two freed windows
    overlapping. All three mean s1 and/or s2 need to shrink for that
    cross section's own perimeter.
    """
    for cs, cs_corners in zip(spline_surf.cs_list, geometry):
        n_ctrl_pts = cs.n_ctrl_pts
        n_pts = cs.n_pts
        full_to_stored = _full_to_stored_index_map(cs)

        # First pass (no mutation): work out each corner's free_idx/
        # apex_full_idx and validate them, before touching any dofs --
        # so a guard failure on the second corner never leaves the
        # first corner's control points half-moved.
        prepared = []
        seen_free_idx = set()
        for g in cs_corners:
            samples = g["samples"]
            greville = samples["greville"]
            u_minus, u_plus, u_corner = g["u_minus"], g["u_plus"], g["u_corner"]
            span = (u_plus - u_minus) % (2 * np.pi)
            rel = (greville - u_minus) % (2 * np.pi)
            free_idx = np.nonzero((rel > 0) & (rel < span))[0]

            if cs.z_sym:
                pinned = {0}
                if n_ctrl_pts % 2 == 0:
                    pinned.add(n_pts - 1)
                if pinned & set(free_idx.tolist()):
                    raise ValueError(
                        "tangent_extension: freed window reaches a z_sym "
                        "mirror-axis-pinned control point -- shrink s1/s2."
                    )
                stored_free = full_to_stored[free_idx]
                if len(set(stored_free.tolist())) != len(free_idx):
                    raise ValueError(
                        "tangent_extension: freed window wraps far enough "
                        "around a z_sym cross section to cover both a "
                        "control point and its mirror image -- shrink s1/s2."
                    )

            if seen_free_idx & set(free_idx.tolist()):
                raise ValueError(
                    "tangent_extension: the +Z and -Z corners' freed "
                    "windows overlap on this cross section -- shrink "
                    "s1/s2."
                )
            seen_free_idx |= set(free_idx.tolist())

            if len(free_idx) == 0:
                prepared.append((g, free_idx, None))
                continue

            corner_rel = (u_corner - u_minus) % (2 * np.pi)
            apex_full_idx = int(
                free_idx[np.argmin(np.abs(rel[free_idx] - corner_rel))]
            )
            prepared.append((g, free_idx, apex_full_idx))

        for g, free_idx, apex_full_idx in prepared:
            if len(free_idx) == 0:
                continue
            samples = g["samples"]
            greville = samples["greville"]
            us, s, L = samples["us"], samples["s"], samples["L"]
            u_minus, u_corner, u_plus = g["u_minus"], g["u_corner"], g["u_plus"]

            s_minus_abs = _u_to_arclength(u_minus, us, s, L)
            s_corner_abs = s_minus_abs + (
                (_u_to_arclength(u_corner, us, s, L) - s_minus_abs) % L
            )
            s_plus_abs = s_minus_abs + (
                (_u_to_arclength(u_plus, us, s, L) - s_minus_abs) % L
            )

            for idx in free_idx:
                s_idx = s_minus_abs + (
                    (_u_to_arclength(greville[idx], us, s, L) - s_minus_abs) % L
                )
                if idx == apex_full_idx:
                    xy_new = g["apex"]
                elif s_idx <= s_corner_abs:
                    frac = (
                        0.0
                        if s_corner_abs <= s_minus_abs
                        else (s_idx - s_minus_abs)
                        / (s_corner_abs - s_minus_abs)
                    )
                    xy_new = g["P_minus"] + frac * (g["apex"] - g["P_minus"])
                else:
                    frac = (
                        0.0
                        if s_plus_abs <= s_corner_abs
                        else (s_idx - s_corner_abs)
                        / (s_plus_abs - s_corner_abs)
                    )
                    xy_new = g["apex"] + frac * (g["P_plus"] - g["apex"])

                r_new = float(np.hypot(xy_new[0], xy_new[1]))
                theta_new = float(
                    np.arctan2(xy_new[1], xy_new[0]) % (2 * np.pi)
                )

                j = int(full_to_stored[idx])
                if cs.z_sym and idx >= n_pts:
                    theta_new = (2 * np.pi - theta_new) % (2 * np.pi)

                cs.unfix(f"r_{j}")
                cs.set(f"r_{j}", r_new)
                pinned_theta = j == 0 or (
                    cs.z_sym and n_ctrl_pts % 2 == 0 and j == n_pts - 1
                )
                if cs.z_sym and pinned_theta:
                    continue
                cs.unfix(f"theta_{j}")
                cs.set(f"theta_{j}", theta_new)

    return spline_surf


def tangent_extension(spline_surf, s1, s2, corner_criterion, n_samples=4000):
    r"""
    Reshape every cross section's corner (per corner_criterion) into a
    wedge by extending its tangent lines s1/s2 back along the curve --
    see the module docstring for the full algorithm, and
    _apply_tangent_extension's docstring for the mutation/guards.
    Mutates spline_surf in place (freeing and repositioning the
    touched control points) and returns it.

    Assumes refine_poloidal has already been called enough times that
    each cross section's control net densely tracks its own curve
    (see module docstring) -- the reshaping only ever touches existing
    control points, it never inserts new ones.
    """
    geometry = tangent_extension_geometry(
        spline_surf, s1, s2, corner_criterion, n_samples=n_samples
    )
    return _apply_tangent_extension(spline_surf, geometry)


def plot_tangent_extension(
    spline_surf, s1, s2, corner_criterion, n_samples=4000, geometry=None
):
    """
    One 2D subplot per cross section: the CURRENT control net/curve
    (spline_surf.cross_section_xy), the two tangent lines extended
    through P_minus/P_plus, and the resulting new apex --
    tangent_extension_geometry's construction, without calling
    tangent_extension itself (so this can be used to preview a choice
    of s1/s2 before committing to it).

    geometry : optional pre-computed tangent_extension_geometry-shaped
        result (list of lists of _corner_geometry_from_u dicts) --
        e.g. from tangent_extension_shared_corner.py's own geometry
        function, so that script can reuse this exact plot without
        duplicating it. Computed via tangent_extension_geometry if
        omitted.

    Returns (fig, axs).
    """
    if geometry is None:
        geometry = tangent_extension_geometry(
            spline_surf, s1, s2, corner_criterion, n_samples=n_samples
        )
    n_cs = spline_surf.n_cs
    fig, axs = plt.subplots(1, n_cs, squeeze=False, figsize=(4 * n_cs, 4))
    axs = axs[0]
    for ax, cs, cs_corners in zip(axs, spline_surf.cs_list, geometry):
        curve_xy, ctrl_xy = spline_surf.cross_section_xy(cs, n_samples=200)
        curve_closed = np.vstack([curve_xy, curve_xy[:1]])
        ctrl_closed = np.vstack([ctrl_xy, ctrl_xy[:1]])
        ax.plot(curve_closed[:, 0], curve_closed[:, 1], label="curve")
        ax.plot(
            ctrl_closed[:, 0],
            ctrl_closed[:, 1],
            ls="--",
            marker="o",
            alpha=0.4,
            label="control polygon",
        )

        angles = []
        for k, g in enumerate(cs_corners):
            P_minus, P_plus, apex = g["P_minus"], g["P_plus"], g["apex"]
            leg1 = np.stack([P_minus, apex])
            leg2 = np.stack([apex, P_plus])
            label = "tangent extension" if k == 0 else None
            pt_label = "P_minus/P_plus" if k == 0 else None
            apex_label = "new apex" if k == 0 else None
            ax.plot(leg1[:, 0], leg1[:, 1], "r-", lw=1.5, label=label)
            ax.plot(leg2[:, 0], leg2[:, 1], "r-", lw=1.5)
            ax.scatter(*P_minus, c="tab:orange", zorder=3, label=pt_label)
            ax.scatter(*P_plus, c="tab:orange", zorder=3)
            ax.scatter(
                *apex,
                c="tab:red",
                marker="*",
                s=120,
                zorder=3,
                label=apex_label,
            )
            angles.append(f"{np.degrees(g['opening_angle']):.1f}")

        ax.set_aspect("equal")
        ax.set_title("opening angle(s) = " + ", ".join(angles) + " deg")
    axs[0].legend(fontsize=7)
    fig.tight_layout()
    return fig, axs


if __name__ == "__main__":
    from simsopt.geo import SurfaceBSpline

    dofs = [
        0.20331546,
        0.15746095,
        0.43573368,
        0.14817898,
        0.88822389,
        1.9522663,
        0.14087834,
        0.24736301,
        0.42883166,
        0.15160325,
        0.45816498,
        0.14868365,
        0.59658694,
        1.7730913,
        2.61799388,
        4.23531002,
        4.71238898,
        0.07372945,
        0.2838024,
        0.51265284,
        0.21184257,
        0.46871346,
        0.12405194,
        0.62203555,
        1.57720828,
        3.66519143,
        4.31988537,
        4.71238898,
        0.07226604,
        0.40665305,
        0.48766582,
        0.22361523,
        0.53481562,
        0.04923251,
        1.02054689,
        1.57079633,
        3.66519143,
        4.47219325,
        4.71238898,
        0.02003046,
        0.43141331,
        0.50441053,
        0.22720611,
        0.58282917,
        0.17844745,
        1.12887769,
        1.57079633,
        3.66519143,
        4.66296652,
        4.71238898,
        0.0213662,
        0.44783741,
        0.51468795,
        0.18259442,
        1.37237811,
        1.57079633,
        0.95657065,
        1.10700731,
        1.21218964,
        0.012124,
    ]
    spline_kwargs = {
        "axis_points": 3,
        "points_per_cs": 6,
        "n_cs": 6,
        "nfp": 2,
        "M": 12,
        "N": 12,
        "p_u": 3,
        "p_v": 3,
        "cs_equispaced": True,
        "rays_equispaced": False,
        "cs_global_angle_free": False,
        "axis_angles_fixed": True,
        "cs_basis": "polar",
        "nurbs": False,
        "use_bishop_frame": True,
        "knot_parametrization": "uniform",
    }

    spline_surf = SurfaceBSpline(**spline_kwargs)
    spline_surf.x = dofs
    for _ in range(3):
        spline_surf.refine_poloidal()

    cs_zeta, _ = spline_surf.get_cs_zeta_angle()
    perimeters = [
        _cross_section_local_curve(spline_surf, cs, n_samples=2000)["L"]
        for cs in spline_surf.cs_list
    ]
    S1 = 0.15 * min(perimeters)
    S2 = 0.10 * min(perimeters)
    CORNER_CRITERION = "max z"

    plot_tangent_extension(spline_surf, S1, S2, CORNER_CRITERION)
    plt.show()

    tangent_extension(spline_surf, S1, S2, CORNER_CRITERION)

    fig, axs = plt.subplots(
        1, spline_surf.n_cs, squeeze=False, figsize=(4 * spline_surf.n_cs, 4)
    )
    for ax, cs in zip(axs[0], spline_surf.cs_list):
        curve_xy, ctrl_xy = spline_surf.cross_section_xy(cs, n_samples=400)
        curve_closed = np.vstack([curve_xy, curve_xy[:1]])
        ctrl_closed = np.vstack([ctrl_xy, ctrl_xy[:1]])
        ax.plot(curve_closed[:, 0], curve_closed[:, 1])
        ax.plot(
            ctrl_closed[:, 0], ctrl_closed[:, 1], ls="--", marker="o", alpha=0.4
        )
        ax.set_aspect("equal")
    fig.suptitle("after tangent_extension")
    fig.tight_layout()
    plt.show()
