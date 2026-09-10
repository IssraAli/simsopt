#!/usr/bin/env python
"""
"Twin" variant of sharpen_fixed_s.py: after sharpening every cross
section's corners into wedges (tangent_extension.py's algorithm), split
each cross section at its two corners into two arcs -- "outboard"
(larger mean lab-frame major radius) and "inboard" (smaller) -- and
build TWO independent SurfaceBSpline surfaces that CROSS OVER at each
corner:

- surf_outboard keeps the whole outboard arc untouched, and at each
  corner replaces the control points on the inboard side with a
  straight stub that continues the outboard leg's own tangent
  direction through the apex -- it does not bend at the corner, it
  keeps going straight, crossing into where the inboard arc would be.
- surf_inboard is the mirror: keeps the inboard arc, and extends its
  own tangent straight through the apex into outboard territory.

The crossover reach at each corner is `d_fraction * that cross
section's own perimeter` (see sharpen_with_crossover), so it scales
with each cross section's own size.

This is the SurfaceBSpline analogue of the old SurfaceHelicalArc
alpha=0 / alpha=pi surface pair: two overlapping half-surfaces standing
in for an X-point boundary that a single closed spline surface can't
represent cleanly right at the crossing.

A z_sym cross section has two real geometric corners (+Z and -Z,
mirror images) even though only one is independently reshaped (mirror
symmetry ties the other to it) -- _corner_pair_geometry finds both
directly, so every cross section is treated uniformly as having
exactly two corners.

SurfaceBSpline can't be pickled directly (wraps a C++ extension type).
pickle_twin_surface/load_twin_surface pickle/reconstruct a plain-dict
recipe (spline_kwargs, quadpoints, refine_poloidal count, dof vector)
instead.

Evaluating the crossover legs across many toroidal angles (not just
the n_cs discrete cross sections) is done by INTERPOLATING the exact
per-cross-section u_stop values (each one exactly
d_fraction*perimeter arc length past its own apex) across phi -- see
interpolated_leg_u_bounds. Re-detecting corners from scratch at each
phi was tried and is unreliable: a twin's crossover construction makes
the curve pass straight through the apex on the replaced side, so
hunting for a curvature/z extremum right there searches for a feature
that was intentionally erased.
"""

import pickle

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from sharpen import _full_to_stored_index_map
from simsopt.geo import SurfaceBSpline
from simsopt.mhd import Vmec
from simsopt.util import proc0_print
from simsopt.util.mpi import MpiPartition
from tangent_extension import (
    _apex_full_index,
    _apply_tangent_extension,
    _arclength_to_u,
    _build_z_embed_list,
    _corner_geometry_from_u,
    _cross_section_local_curve,
    _curve_curvature,
    _periodic_curve_deriv2,
    _u_to_arclength,
)

mpi = MpiPartition(ngroups=1)
mpi.write()

REFINE_POLOIDAL_COUNT = 3


def _find_both_corner_us(samples, corner_criterion, z_embed):
    """
    Like tangent_extension._find_corner_us, but always returns both the
    +Z and -Z corner's u (splitting the loop at the midpoint of its own
    max/min Z), regardless of z_sym -- a z_sym cross section still has
    two real corners, only one of which is independently reshaped.

    Returns a sorted [u_plus, u_minus].
    """
    us = samples["us"]
    if corner_criterion == "max z":
        metric = z_embed(samples["pos"])
    else:
        metric = _curve_curvature(samples["vel"], samples["acc"])

    z = z_embed(samples["pos"])
    z_mid = 0.5 * (z.max() + z.min())
    plus_mask = z >= z_mid
    minus_mask = ~plus_mask
    u_plus = us[plus_mask][int(np.argmax(metric[plus_mask]))]
    if corner_criterion == "max z":
        # -Z corner is the metric's MINIMUM in the minus half (its
        # maximum there would just be the point closest to the
        # equator). Curvature is always positive, so its corner is a
        # maximum in either half.
        u_minus = us[minus_mask][int(np.argmin(metric[minus_mask]))]
    else:
        u_minus = us[minus_mask][int(np.argmax(metric[minus_mask]))]
    return sorted([u_plus, u_minus])


def _corner_pair_geometry(spline_surf, s1, s2, corner_criterion, n_samples=4000):
    """
    tangent_extension_geometry-shaped result, but always exactly 2
    corners per cross section (see _find_both_corner_us), ordered
    [lower-u corner, higher-u corner].
    """
    z_embed_list = _build_z_embed_list(spline_surf)
    geometry = []
    for i, cs in enumerate(spline_surf.cs_list):
        samples = _cross_section_local_curve(spline_surf, cs, n_samples)
        u0, u1 = _find_both_corner_us(samples, corner_criterion, z_embed_list[i])
        geometry.append(
            [
                _corner_geometry_from_u(samples, u0, s1, s2),
                _corner_geometry_from_u(samples, u1, s1, s2),
            ]
        )
    return geometry


def _crossover_target(g, replace_window, d):
    """
    The point extending corner g's KEPT leg straight through the apex
    by arc length `d`, and the curve parameter u there:

    replace_window='minus' -- this twin replaces g's minus side; the
        kept leg is "plus", so the stub continues that tangent
        BACKWARD through the apex.
    replace_window='plus' -- mirror: kept leg is "minus", stub
        continues FORWARD through the apex.
    """
    us, s, L = g["samples"]["us"], g["samples"]["s"], g["samples"]["L"]
    s_corner = _u_to_arclength(g["u_corner"], us, s, L)
    if replace_window == "minus":
        target = g["apex"] - d * g["T_plus"]
        u_stop = _arclength_to_u(s_corner - d, us, s, L)
    elif replace_window == "plus":
        target = g["apex"] + d * g["T_minus"]
        u_stop = _arclength_to_u(s_corner + d, us, s, L)
    else:
        raise ValueError("replace_window must be 'minus' or 'plus'")
    return target, u_stop


def _apply_crossover_leg(cs, g, replace_window, u_stop, target, apex_idx):
    """
    Reposition every control point strictly between g's apex and
    u_stop onto the straight segment between g['apex'] and `target`, at
    the same relative arc-length fraction each started at (mirrors
    _apply_tangent_extension's own interpolation). `apex_idx` is never
    touched -- it's already exactly at the apex and stays the shared
    meeting point both twins pass through.
    """
    samples = g["samples"]
    greville = samples["greville"]
    us, s, L = samples["us"], samples["s"], samples["L"]
    u_corner = g["u_corner"]
    full_to_stored = _full_to_stored_index_map(cs)
    n_pts = cs.n_pts

    if replace_window == "minus":
        u_lo, u_hi = u_stop, u_corner
        far_lo, far_hi = target, g["apex"]
    else:
        u_lo, u_hi = u_corner, u_stop
        far_lo, far_hi = g["apex"], target

    span = (u_hi - u_lo) % (2 * np.pi)
    rel = (greville - u_lo) % (2 * np.pi)
    free_idx = np.nonzero((rel > 0) & (rel < span))[0]

    s_lo_abs = _u_to_arclength(u_lo, us, s, L)
    s_hi_abs = s_lo_abs + ((_u_to_arclength(u_hi, us, s, L) - s_lo_abs) % L)

    for idx in free_idx:
        if idx == apex_idx:
            continue
        s_idx = s_lo_abs + ((_u_to_arclength(greville[idx], us, s, L) - s_lo_abs) % L)
        frac = 0.0 if s_hi_abs <= s_lo_abs else (s_idx - s_lo_abs) / (s_hi_abs - s_lo_abs)
        xy_new = far_lo + frac * (far_hi - far_lo)

        r_new = float(np.hypot(xy_new[0], xy_new[1]))
        theta_new = float(np.arctan2(xy_new[1], xy_new[0]) % (2 * np.pi))
        j = int(full_to_stored[idx])
        if cs.z_sym and idx >= n_pts:
            theta_new = (2 * np.pi - theta_new) % (2 * np.pi)

        cs.unfix(f"r_{j}")
        cs.set(f"r_{j}", r_new)
        pinned_theta = j == 0 or (cs.z_sym and cs.n_ctrl_pts % 2 == 0 and j == n_pts - 1)
        if cs.z_sym and pinned_theta:
            continue
        cs.unfix(f"theta_{j}")
        cs.set(f"theta_{j}", theta_new)


def _build_r_embed_list(spline_surf):
    """One r_embed closure per cross section -- r_embed(xy) returns the
    lab-frame major radius R = hypot(X, Y) that local (x, y) points sit
    at. Needed because the local (bishop) frame can be twisted relative
    to the lab frame from one cross section to the next, so a
    local-frame proxy for "outboard" isn't reliable on its own."""
    cs_zeta, cs_angle = spline_surf.get_cs_zeta_angle()
    axis_pos, e1, e2 = spline_surf._axis_local_basis(cs_zeta)

    def make_r_embed(i):
        def r_embed(xy, i=i):
            ang = cs_angle[i]
            x, y = xy[:, 0], xy[:, 1]
            xr = x * np.cos(ang) - y * np.sin(ang)
            yr = x * np.sin(ang) + y * np.cos(ang)
            X = axis_pos[i, 0] + xr * e1[i, 0] + yr * e2[i, 0]
            Y = axis_pos[i, 1] + xr * e1[i, 1] + yr * e2[i, 1]
            return np.hypot(X, Y)

        return r_embed

    return [make_r_embed(i) for i in range(spline_surf.n_cs)]


def _arc_mean_lab_radius(samples, u_lo, u_hi, r_embed):
    """Mean lab-frame major radius of the sampled curve strictly
    between u_lo and u_hi (wrapping through 0 if u_hi < u_lo) -- used to
    label the two arcs outboard/inboard."""
    us, pos = samples["us"], samples["pos"]
    span = (u_hi - u_lo) % (2 * np.pi)
    rel = (us - u_lo) % (2 * np.pi)
    mask = rel < span
    return float(np.mean(r_embed(pos[mask])))


def _crossover_stops_per_cs(geometry, d_fraction, keep_between):
    """
    Per-cross-section crossover-stub data for ONE twin (side chosen by
    keep_between): a list of (c0_replace, c0_target, c0_u_stop,
    c1_replace, c1_target, c1_u_stop), one entry per cross section --
    everything _build_twin needs to mutate a twin's control points, and
    everything interpolated_leg_u_bounds needs to build a per-phi
    window from the same exact u_stop values.
    """
    stops = []
    for cs_corners in geometry:
        c0, c1 = cs_corners
        d_natural = d_fraction * c0["samples"]["L"]
        if keep_between:
            c0_replace, c1_replace = "minus", "plus"
        else:
            c0_replace, c1_replace = "plus", "minus"
        c0_target, c0_u_stop = _crossover_target(c0, c0_replace, d_natural)
        c1_target, c1_u_stop = _crossover_target(c1, c1_replace, d_natural)
        stops.append((c0_replace, c0_target, c0_u_stop, c1_replace, c1_target, c1_u_stop))
    return stops


def sharpen_with_crossover(
    spline_surf,
    spline_kwargs,
    s1,
    s2,
    d_fraction,
    corner_criterion,
    n_samples=4000,
    refine_poloidal_count=REFINE_POLOIDAL_COUNT,
):
    """
    Sharpen every cross section's two corners into wedges, then build
    the two crossover-extended twin SurfaceBSplines (see module
    docstring). `d_fraction * that cross section's own perimeter` is
    each corner's crossover reach, in arc length.

    `spline_kwargs` must be the ORIGINAL (pre-refine_poloidal)
    construction dict -- refine_poloidal mutates attributes like
    points_per_cs in place, so reconstructing kwargs from spline_surf
    itself would silently double-refine. `refine_poloidal_count` must
    match how many times spline_surf ITSELF was already refined before
    being passed in (0 if spline_kwargs already describes its final,
    already-refined resolution, as for e.g. real equilibrium dofs
    fit directly at that resolution rather than built up via
    refine_poloidal) -- each twin is a freshly-constructed
    SurfaceBSpline(**spline_kwargs), refined this many times to match
    spline_surf's own dof count before sharpened_full_x is assigned to
    it.

    Mutates spline_surf in place (the sharpening step) and returns
    (surf_outboard, surf_inboard, geometry, outboard_is_between) --
    geometry is _corner_pair_geometry's result (needed to interpolate
    leg bounds later); outboard_is_between says whether surf_outboard
    is the 'between' twin (the arc strictly between the two corners in
    ascending u) or the 'around' twin (wraps through u=0/2*pi) --
    surf_inboard is always the other one.
    """
    geometry = _corner_pair_geometry(spline_surf, s1, s2, corner_criterion, n_samples)
    spline_surf.fix_all()
    _apply_tangent_extension(spline_surf, geometry)
    spline_surf.unfix_all()
    sharpened_full_x = spline_surf.full_x

    r_embed_list = _build_r_embed_list(spline_surf)
    outboard_votes = []
    for i, cs_corners in enumerate(geometry):
        c0, c1 = cs_corners
        samples = c0["samples"]
        r_between = _arc_mean_lab_radius(samples, c0["u_plus"], c1["u_minus"], r_embed_list[i])
        r_around = _arc_mean_lab_radius(samples, c1["u_plus"], c0["u_minus"], r_embed_list[i])
        outboard_votes.append(r_between >= r_around)
    outboard_is_between = sum(outboard_votes) >= len(outboard_votes) / 2
    if len(set(outboard_votes)) > 1:
        proc0_print(
            "sharpen_with_crossover: WARNING -- which arc is 'outboard' "
            "disagrees between cross sections; using the majority vote "
            f"({sum(outboard_votes)}/{len(outboard_votes)} say 'between')."
        )

    def _build_twin(keep_between, ntheta):
        stops = _crossover_stops_per_cs(geometry, d_fraction, keep_between)
        quadpoints_theta = np.linspace(0.0, 1.0, ntheta, endpoint=False)
        twin = SurfaceBSpline(**spline_kwargs, quadpoints_theta=quadpoints_theta)
        for _ in range(refine_poloidal_count):
            twin.refine_poloidal()
        twin.unfix_all()
        # .copy(): DOFs.full_x's setter doesn't copy an already-float64
        # array, so without it every twin would alias the same memory.
        twin.full_x = sharpened_full_x.copy()

        for cs, cs_corners, (
            c0_replace,
            c0_target,
            c0_u_stop,
            c1_replace,
            c1_target,
            c1_u_stop,
        ) in zip(twin.cs_list, geometry, stops):
            c0, c1 = cs_corners
            _apply_crossover_leg(cs, c0, c0_replace, c0_u_stop, c0_target, _apex_full_index(c0))
            _apply_crossover_leg(cs, c1, c1_replace, c1_u_stop, c1_target, _apex_full_index(c1))
        return twin

    ntheta = 60
    surf_between = _build_twin(True, ntheta)
    surf_around = _build_twin(False, ntheta)
    surf_outboard, surf_inboard = (
        (surf_between, surf_around) if outboard_is_between else (surf_around, surf_between)
    )
    return surf_outboard, surf_inboard, geometry, outboard_is_between


def pickle_twin_surface(surf, spline_kwargs, filename, refine_poloidal_count=REFINE_POLOIDAL_COUNT):
    """
    SurfaceBSpline can't be pickled directly (wraps a C++ extension
    type) -- pickle a reconstruction recipe instead. See
    load_twin_surface. `spline_kwargs` must be the ORIGINAL
    (pre-refine_poloidal) construction dict, not introspected from
    `surf` -- see sharpen_with_crossover's docstring. Pass the SAME
    refine_poloidal_count sharpen_with_crossover was called with to
    build `surf`.
    """
    recipe = {
        "spline_kwargs": spline_kwargs,
        "quadpoints_theta": np.asarray(surf.quadpoints_theta),
        "quadpoints_phi": np.asarray(surf.quadpoints_phi),
        "refine_poloidal_count": refine_poloidal_count,
        "full_x": np.asarray(surf.full_x),
    }
    with open(filename, "wb") as f:
        pickle.dump(recipe, f)


def load_twin_surface(filename):
    """Reconstruct a SurfaceBSpline from a pickle_twin_surface recipe."""
    with open(filename, "rb") as f:
        recipe = pickle.load(f)
    surf = SurfaceBSpline(
        **recipe["spline_kwargs"],
        quadpoints_theta=recipe["quadpoints_theta"],
        quadpoints_phi=recipe["quadpoints_phi"],
    )
    for _ in range(recipe["refine_poloidal_count"]):
        surf.refine_poloidal()
    surf.unfix_all()
    surf.full_x = recipe["full_x"]
    return surf


def _cs_curve_xy(surf, i, us=None):
    """The (len(us), 2) local (x, y) curve surf's cross section `i`
    traces at raw curve-parameter values `us` (radians) -- defaults to
    surf's own quadpoints_theta if `us` is omitted."""
    cs = surf.cs_list[i]
    core = np.stack(
        [
            cs.get_r_ctrl_full() * np.cos(cs.get_theta_ctrl_full()),
            cs.get_r_ctrl_full() * np.sin(cs.get_theta_ctrl_full()),
        ],
        axis=1,
    )
    if us is None:
        us = surf.quadpoints_theta * 2 * np.pi
    pos, _, _ = _periodic_curve_deriv2(core, cs.get_w_ctrl_full(), surf.p_u, surf.knot_parametrization, us)
    return pos


# ---------------------------------------------------------------------
# Evaluating the crossover legs at arbitrary phi (via SurfaceBSpline's
# paired-point gamma_lin, not the usual structured quadpoints_phi x
# quadpoints_theta tensor grid): each phi's u window is INTERPOLATED
# from the exact per-cross-section u_stop values _build_twin itself
# used (_crossover_stops_per_cs) -- see module docstring for why
# re-detecting corners from scratch at each phi is unreliable instead.
# ---------------------------------------------------------------------


def _leg_u_row(u_lo, u_hi, keep_between, n):
    """n raw-radian u values spanning one side's leg window: ascending
    directly if keep_between, wrapping through the u=0/2*pi seam
    otherwise."""
    if keep_between:
        return np.linspace(u_lo, u_hi, n, endpoint=True)
    span = (u_hi - u_lo) % (2 * np.pi)
    return np.linspace(u_lo, u_lo + span, n, endpoint=True)


def interpolated_leg_u_bounds(spline_surf, geometry, leg_fraction, keep_between, phis):
    """
    Per-phi (u_lo, u_hi) window bounds for ONE twin (keep_between --
    True for 'between', False for 'around'), built by linearly
    interpolating the exact per-cross-section u_stop values
    _crossover_stops_per_cs computes across phi.

    `leg_fraction` is independent of the d_fraction the twins were
    actually built with (that's baked into their control points) -- it
    only controls how much of the already-straight crossover stub is
    used here. Any leg_fraction <= the original d_fraction stays on
    the guaranteed-straight part of the stub; a larger value would
    reach into the real, untouched curve beyond it.

    `phis` are normalized [0, 1) toroidal fractions within
    [cs_zeta[0], cs_zeta[-1]]/(2*pi) (see phi_fractions) -- no
    extrapolation past the n_cs corner-defining cross sections.

    Returns (u_lo_phi, u_hi_phi), unwrapped (not reduced mod 2*pi), to
    keep the interpolation itself continuous across the 0/2*pi seam.
    """
    cs_zeta, _ = spline_surf.get_cs_zeta_angle()
    cs_phi = cs_zeta / (2 * np.pi)
    stops = _crossover_stops_per_cs(geometry, leg_fraction, keep_between)
    c0_u_stops = np.array([s[2] for s in stops])
    c1_u_stops = np.array([s[5] for s in stops])
    u_los, u_his = (c0_u_stops, c1_u_stops) if keep_between else (c1_u_stops, c0_u_stops)
    u_lo_phi = np.interp(phis, cs_phi, np.unwrap(u_los))
    u_hi_phi = np.interp(phis, cs_phi, np.unwrap(u_his))
    return u_lo_phi, u_hi_phi


def interpolated_corner_u(spline_surf, geometry, phis):
    """
    Per-phi (u0_phi, u1_phi) RAW corner-apex locations (radians,
    unwrapped), linearly interpolated across phi from geometry's own
    per-cross-section u_corner values -- same interpolation technique
    as interpolated_leg_u_bounds, but for the corners (apexes)
    themselves rather than their crossover-stub tips.

    Both twins pass through the exact same apex (see
    _apply_crossover_leg), so these u values, together with any twin's
    gamma_lin, give the apex's lab-frame position at any phi in range
    -- useful for e.g. weighting an objective by distance from the
    nearest corner (see stage_2_scan_twin.py's squared_flux_weights).
    """
    cs_zeta, _ = spline_surf.get_cs_zeta_angle()
    cs_phi = cs_zeta / (2 * np.pi)
    u0 = np.array([cs_corners[0]["u_corner"] for cs_corners in geometry])
    u1 = np.array([cs_corners[1]["u_corner"] for cs_corners in geometry])
    u0_phi = np.interp(phis, cs_phi, np.unwrap(u0))
    u1_phi = np.interp(phis, cs_phi, np.unwrap(u1))
    return u0_phi, u1_phi


def compute_uphi_grid_interpolated(spline_surf, geometry, leg_fraction, keep_between, phis, ntheta=60):
    """
    (phi, theta) grid for ONE twin's own side (keep_between -- must be
    the side that twin actually keeps/owns, i.e. sharpen_with_crossover's
    outboard_is_between for surf_outboard, its negation for surf_inboard
    -- the other side is left as the original untouched sharpened curve,
    so evaluating it here would just duplicate the other twin's arc).

    Returns {'phi': (n_phis, ntheta), 'theta': (n_phis, ntheta)}, ready
    to flatten and pass to surf.gamma_lin(data, phi.reshape(-1),
    theta.reshape(-1)).
    """
    u_lo_phi, u_hi_phi = interpolated_leg_u_bounds(spline_surf, geometry, leg_fraction, keep_between, phis)
    phi_rows, theta_rows = [], []
    for phi, u_lo, u_hi in zip(phis, u_lo_phi, u_hi_phi):
        us = _leg_u_row(u_lo, u_hi, keep_between, ntheta)
        phi_rows.append(np.full(ntheta, phi))
        theta_rows.append(np.mod(us / (2 * np.pi), 1.0))
    return {"phi": np.array(phi_rows), "theta": np.array(theta_rows)}


def phi_fractions(surf, n_phi=30):
    """n_phi normalized [0, 1) toroidal fractions, evenly spaced across
    surf's own n_cs corner-defining cross sections' range (one
    mirror-symmetric half field period)."""
    cs_zeta, _ = surf.get_cs_zeta_angle()
    return np.linspace(cs_zeta[0], cs_zeta[-1], n_phi) / (2 * np.pi)


def reduce_phi_to_fundamental_domain(spline_surf, phi_frac):
    """
    Maps an arbitrary normalized toroidal fraction `phi_frac` (any real
    value) into spline_surf's own fundamental domain
    [0, cs_zeta[-1]/(2*pi)] -- one field period's half, exploiting nfp
    periodicity and stellarator up-down mirror symmetry -- returning
    (phi_reduced, mirrored).

    interpolated_leg_u_bounds/interpolated_corner_u/
    compute_uphi_grid_interpolated all require phi within that
    fundamental domain (see their own docstrings) -- np.interp
    silently CLAMPS (does not wrap/mirror) outside it, so calling them
    directly with an out-of-range phi gives a wrong, silently-clamped
    result. When `mirrored`, evaluate geometry at `phi_reduced` as
    usual, then pass the result through mirror_position_to_phi to get
    the true position at `phi_frac` (verified numerically against
    spline_surf.cross_section: nearest-neighbor shape match ~1e-9).
    """
    nfp = spline_surf.nfp
    phi_period = phi_frac % (1.0 / nfp)
    if phi_period <= 1.0 / (2 * nfp):
        return phi_period, False
    return (1.0 / nfp) - phi_period, True


def mirror_position_to_phi(pos, phi_frac):
    """Given lab-frame (..., 3) positions computed at the MIRRORED
    reduced phi (see reduce_phi_to_fundamental_domain), returns the
    true positions at `phi_frac`: same major radius, flipped Z,
    re-placed at the cylindrical angle 2*pi*phi_frac."""
    R = np.hypot(pos[..., 0], pos[..., 1])
    Z = -pos[..., 2]
    angle = 2 * np.pi * phi_frac
    X = R * np.cos(angle)
    Y = R * np.sin(angle)
    return np.stack([X, Y, Z], axis=-1)


def build_sharpened_twins(
    spline_surf,
    spline_kwargs,
    d_crawl,
    d_ext,
    l_x,
    corner_criterion,
    n_phi=30,
    ntheta=60,
    n_samples=4000,
    refine_poloidal_count=REFINE_POLOIDAL_COUNT,
):
    """
    One-call entry point for the whole pipeline: sharpen spline_surf's
    corners into wedges and split it into the two crossover "twin"
    surfaces (sharpen_with_crossover), then compute the (phi, theta)
    points on each twin needed to show/evaluate exactly
    l_x*perimeter of crossover leg beyond each corner
    (compute_uphi_grid_interpolated).

    d_crawl: shared tangent-extension crawl fraction -- both s1 and s2
        (arc length, as a fraction of each cross section's own
        perimeter) used to measure each corner's tangent directions.
        Not the crossover reach itself.
    d_ext: the crossover-leg extension fraction the twins are actually
        BUILT with (sharpen_with_crossover's own d_fraction) -- how far
        each twin's straight stub reaches past the apex, baked into its
        control points.
    l_x: the X-point leg length fraction used only to pick which
        u-values (the returned grids) to evaluate the ALREADY-built
        twins at -- independent of d_ext, but must be <= d_ext to stay
        on the guaranteed-straight part of the stub (a larger value
        would reach into the real, untouched curve beyond it).
    corner_criterion: "z" (locate each corner by max Z) or "curvature".

    Mutates spline_surf in place (the sharpening step) and returns
    (spline_surf, surf_outboard, surf_inboard, grids) where grids is
    {'outboard': ..., 'inboard': ...} (each a
    compute_uphi_grid_interpolated result -- the points to evaluate via
    surf.gamma_lin to get exactly l_x of crossover leg), plus 'phis',
    'geometry', and 'outboard_is_between' for callers that also want to
    plot_before_after_twin/plot_leg_length_grid.
    """
    if corner_criterion not in ("z", "curvature"):
        raise ValueError(f"corner_criterion must be 'z' or 'curvature', got {corner_criterion!r}")
    internal_criterion = "max z" if corner_criterion == "z" else "curvature"

    perimeters = [
        _cross_section_local_curve(spline_surf, cs, n_samples=n_samples)["L"]
        for cs in spline_surf.cs_list
    ]
    s1 = s2 = d_crawl * min(perimeters)

    surf_outboard, surf_inboard, geometry, outboard_is_between = sharpen_with_crossover(
        spline_surf, spline_kwargs, s1, s2, d_ext, internal_criterion,
        n_samples=n_samples, refine_poloidal_count=refine_poloidal_count,
    )

    phis = phi_fractions(surf_outboard, n_phi=n_phi)
    grid_outboard = compute_uphi_grid_interpolated(
        spline_surf, geometry, l_x, outboard_is_between, phis, ntheta=ntheta
    )
    grid_inboard = compute_uphi_grid_interpolated(
        spline_surf, geometry, l_x, not outboard_is_between, phis, ntheta=ntheta
    )
    grids = {
        "outboard": grid_outboard,
        "inboard": grid_inboard,
        "phis": phis,
        "geometry": geometry,
        "outboard_is_between": outboard_is_between,
    }
    return spline_surf, surf_outboard, surf_inboard, grids


def plot_leg_length_grid(
    spline_surf, geometry, surf_outboard, surf_inboard, outboard_is_between, leg_fraction, phis, ntheta=60
):
    """
    3D surface plot of surf_outboard/surf_inboard on the interpolated
    (phi, theta) grid -- one shaded quad mesh per twin (rows = phis,
    columns = the ntheta points along that phi's own leg window), each
    showing leg_fraction*perimeter arc length of crossover leg beyond
    its own corner. Pass a leg_fraction <= the twins' own d_fraction to
    control how much of the (already-built) legs is shown.

    Returns (fig, ax).
    """
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    all_pos = []
    legend_handles = []
    for surf, keep_between, color, label in (
        (surf_outboard, outboard_is_between, "tab:red", "outboard"),
        (surf_inboard, not outboard_is_between, "tab:blue", "inboard"),
    ):
        grid = compute_uphi_grid_interpolated(spline_surf, geometry, leg_fraction, keep_between, phis, ntheta)
        phi_arr, theta_arr = grid["phi"], grid["theta"]
        pos_flat = np.zeros((phi_arr.size, 3))
        surf.gamma_lin(pos_flat, phi_arr.reshape(-1), theta_arr.reshape(-1))
        pos = pos_flat.reshape(phi_arr.shape + (3,))
        all_pos.append(pos.reshape(-1, 3))
        # plot_surface doesn't support the `label` kwarg -- a proxy
        # Patch stands in for the legend entry instead.
        ax.plot_surface(
            pos[:, :, 0], pos[:, :, 1], pos[:, :, 2],
            color=color, alpha=0.7, linewidth=0, antialiased=True,
        )
        legend_handles.append(Patch(color=color, label=label))
    # 3D axes don't default to equal aspect -- without this, a smooth
    # ribbon can look like it necks down to a point from some angles.
    all_pos = np.concatenate(all_pos, axis=0)
    ax.set_box_aspect(tuple(all_pos.max(axis=0) - all_pos.min(axis=0)))
    ax.set_title(f"Crossover legs (leg_fraction={leg_fraction:.4g})")
    ax.legend(handles=legend_handles, fontsize=8)
    fig.tight_layout()
    return fig, ax


def plot_before_after_twin(
    spline_surf_before, spline_surf, geometry, surf_outboard, surf_inboard,
    outboard_is_between, leg_fraction, ntheta=200,
):
    """One 2D subplot per cross section: the original curve, the
    sharpened (wedge) curve, and each twin's own kept-arc-plus-crossover
    curve (evaluated via the SAME interpolated leg bounds
    plot_leg_length_grid uses, at exactly that cross section's own
    phi), all overlaid in the cross section's local (x, y) plane."""
    n_cs = spline_surf.n_cs
    cs_zeta, _ = spline_surf.get_cs_zeta_angle()
    fig, axs = plt.subplots(1, n_cs, squeeze=False, figsize=(4 * n_cs, 4))
    axs = axs[0]
    for i, ax in enumerate(axs):
        cs_before = spline_surf_before.cs_list[i]
        cs_after = spline_surf.cs_list[i]
        curve_before, _ = spline_surf_before.cross_section_xy(cs_before, n_samples=300)
        curve_after, _ = spline_surf.cross_section_xy(cs_after, n_samples=300)
        curve_before = np.vstack([curve_before, curve_before[:1]])
        curve_after = np.vstack([curve_after, curve_after[:1]])
        ax.plot(curve_before[:, 0], curve_before[:, 1], color="0.75", lw=1, label="before")
        ax.plot(curve_after[:, 0], curve_after[:, 1], "k-", lw=1.5, label="sharpened")

        phi_i = np.array([cs_zeta[i] / (2 * np.pi)])
        for surf, keep_between, color, label in (
            (surf_outboard, outboard_is_between, "tab:red", "outboard twin"),
            (surf_inboard, not outboard_is_between, "tab:blue", "inboard twin"),
        ):
            u_lo, u_hi = interpolated_leg_u_bounds(spline_surf, geometry, leg_fraction, keep_between, phi_i)
            us = _leg_u_row(u_lo[0], u_hi[0], keep_between, ntheta)
            pos = _cs_curve_xy(surf, i, us=us)
            ax.plot(pos[:, 0], pos[:, 1], color=color, lw=2, alpha=0.8, label=label)

        for g in geometry[i]:
            ax.scatter(*g["apex"], c="tab:orange", marker="*", s=90, zorder=3)

        ax.set_aspect("equal")
        ax.set_title(f"cross section {i}")
        if i == 0:
            ax.legend(fontsize=7)
    fig.tight_layout()
    return fig, axs


# Toy demo geometry (not a real equilibrium) -- module-level so other
# scripts can import the same construction without duplicating it.
DEMO_DOFS = [
    0.18588267, 0.15642203, 0.43764541, 0.16784444, 1.26476013, 2.01586034,
    0.12997366, 0.21563905, 0.39986906, 0.16935926, 0.50061529, 0.15108062,
    0.64882775, 1.88036231, 2.61799388, 4.26470135, 4.71238898, 0.09381768,
    0.2645054, 0.44291523, 0.20125536, 0.5234703, 0.15781845, 0.71877959,
    1.66919584, 3.66519143, 4.37137573, 4.71238898, 0.07724166, 0.33769754,
    0.4633352, 0.21862174, 0.58350463, 0.11953112, 0.98968873, 1.57079633,
    3.66519143, 4.52991733, 4.71238898, 0.03416579, 0.39328045, 0.49537881,
    0.22380475, 0.59100414, 0.26095387, 1.16794389, 1.57079633, 3.66519143,
    4.6681426, 4.71238898, 0.02337083, 0.41469386, 0.52910186, 0.18522041,
    1.36128815, 1.57079633, 0.92175992, 1.1061464, 1.21557964, 0.07962817,
]
DEMO_SPLINE_KWARGS = {
    "axis_points": 3,
    "points_per_cs": 6,
    "n_cs": 6,
    "nfp": 2,
    "M": 16,
    "N": 16,
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


def build_demo_surface(fix_all=False):
    """A fresh SurfaceBSpline from DEMO_DOFS/DEMO_SPLINE_KWARGS,
    refined REFINE_POLOIDAL_COUNT times."""
    surf = SurfaceBSpline(**DEMO_SPLINE_KWARGS)
    surf.x = DEMO_DOFS
    for _ in range(REFINE_POLOIDAL_COUNT):
        surf.refine_poloidal()
    if fix_all:
        surf.fix_all()
    return surf


if __name__ == "__main__":
    # d_crawl: tangent-extension crawl fraction (both s1 and s2).
    # d_ext: crossover-leg extension the twins are actually built with.
    # l_x: leg length (<= d_ext) used to pick the evaluation grids.
    D_CRAWL = 0.04
    D_EXT = 0.10
    L_X = 0.04
    CORNER_CRITERION = "curvature"
    N_PHI = 30

    spline_kwargs = DEMO_SPLINE_KWARGS
    spline_surf = build_demo_surface()
    spline_surf_before = build_demo_surface(fix_all=True)

    spline_surf, surf_outboard, surf_inboard, grids = build_sharpened_twins(
        spline_surf, spline_kwargs, D_CRAWL, D_EXT, L_X, CORNER_CRITERION, n_phi=N_PHI
    )
    geometry = grids["geometry"]
    outboard_is_between = grids["outboard_is_between"]

    pickle_twin_surface(surf_outboard, spline_kwargs, "sharpen_fixed_s_twin_outboard.pkl")
    pickle_twin_surface(surf_inboard, spline_kwargs, "sharpen_fixed_s_twin_inboard.pkl")
    proc0_print(
        "Pickled twin surfaces to sharpen_fixed_s_twin_outboard.pkl / "
        "sharpen_fixed_s_twin_inboard.pkl (load with load_twin_surface)."
    )

    # vmec run 
    vmec = Vmec.vmec_from_surf(
        nfp=spline_surf.nfp,
        surf=spline_surf,
        mpi=mpi,
        ns=50,
        M=16,
        N=16,
        ftol=1e-11,
        verbose=True,
        niter=8000,
        ntheta=64,
        nzeta=64,
    )
    vmec.run()

    if mpi.proc0_world:
        # Both created above by build_sharpened_twins -- plotted together here.
        plot_before_after_twin(
            spline_surf_before, spline_surf, geometry, surf_outboard, surf_inboard,
            outboard_is_between, L_X,
        )
        plot_leg_length_grid(
            spline_surf, geometry, surf_outboard, surf_inboard, outboard_is_between,
            L_X, grids["phis"],
        )
        plt.show()

    proc0_print("")
    proc0_print("End of figures_spline_paper/corner_sharpening/sharpen_fixed_s_twin.py")
