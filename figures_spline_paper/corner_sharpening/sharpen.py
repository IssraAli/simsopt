#!/usr/bin/env python
"""
Corner sharpening via a purely geometric optimization: starting from a
SurfaceBSpline already loaded with a reasonable, roughly-W7-X-like
shape (the `dofs` below, e.g. from shape_match.py's final fit), shape
each of its +Z and -Z corners into a WEDGE -- `m` control points on
each side of the corner's tip held STRAIGHT (collinear with their own
immediate neighbors), meeting at the tip at one target opening angle
(corner_angle_residuals) -- instead of maximizing curvature at a
single point.

Why a wedge, not a curvature spike: the goal is an equilibrium design
likely to have X-points near the last closed flux surface (a
generalization of the "lemon" target equilibrium, see
arxiv.org/abs/2510.27624). A real magnetic X-point isn't an
infinitely sharp point -- locally it's two STRAIGHT legs meeting at a
well-defined, FINITE opening angle -- and maximizing curvature (a
second-derivative, unbounded quantity with no natural stopping point)
at one point kept producing oscillation no matter which curvature
formula, how many points were free, or how it was regularized, since
a smooth spline simply can't represent a true single-point cusp
stably. Holding the legs straight and only the tip at a finite target
angle (freeze_except_corner_neighborhoods) is both better-conditioned
(bounded [0, pi], first-derivative only) and a much closer match to
the actual physical target shape than sharing one small angle across
the whole neighborhood would be (that would instead round the whole
neighborhood into a circular-arc-like curve, not a wedge).
`poloidal_curve_curvature` / `max_curvature_by_z_half_residuals` are
kept only as diagnostics, to see how the resulting smooth curve's
curvature responds.

No VMEC, no shape-matching target -- this is a standalone test of how
sharp a corner this spline representation can actually produce, using
the "syntax" (MpiPartition, LeastSquaresProblem.from_tuples,
least_squares_mpi_solve call) of examples/2_Intermediate/
stage_one_splines.py, with the QS/aspect-ratio/iota objective terms
replaced by a single geometric corner-shaping term.
"""

import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI
from simsopt._core import make_optimizable
from simsopt.geo import SurfaceBSpline
from simsopt.mhd import QuasisymmetryRatioResidual, Vmec
from simsopt.objectives import LeastSquaresProblem
from simsopt.solve import least_squares_mpi_solve
from simsopt.util import MpiPartition, proc0_print


def _planar_curve_curvature(phi, g2, g22):
    r"""
    kappa at each of N points, of the (R, Z) PLANAR cross-section
    curve at fixed phi -- the actual curve shown in a cross-section
    plot -- computed analytically from the surface's own
    theta-derivatives (gammadash2, gammadash2dash2), rather than the
    surface's mean curvature H. H mixes together bending in both
    parametric directions (poloidal AND toroidal), so it can be driven
    up by toroidal bending having nothing to do with how sharp the
    poloidal cross-section looks -- not the same quantity "a sharp
    corner in a cross-section plot" means.

    Because phi is held EXACTLY fixed, every theta-derivative of the
    surface stays within the single meridional half-plane spanned by
    R_hat = (cos(2*pi*phi), sin(2*pi*phi), 0) and Z_hat = (0, 0, 1)
    (X = R(theta)*cos(phi), Y = R(theta)*sin(phi), Z = Z(theta) with
    phi constant along the curve): dR/dtheta = gammadash2 . R_hat,
    dZ/dtheta = gammadash2's z component, and likewise for the second
    derivatives -- exactly reproducing the classic 2D curve-curvature
    formula kappa = |R'Z'' - Z'R''| / (R'^2 + Z'^2)^1.5, but from
    analytic derivatives instead of a finite-difference estimate.

    :param phi: scalar or (N,) array of toroidal fractions in [0, 1),
        matching g2/g22's own phi (broadcast if scalar)
    :param g2, g22: gammadash2, gammadash2dash2 at N points, each
        (N, 3), in the pairing convention `_gamma_and_derivs` returns
        (point i's derivatives, not a tensor-product grid)
    :return: kappa, an (N,) array
    """
    phi = np.broadcast_to(np.asarray(phi, dtype=float), (g2.shape[0],))
    angle = 2 * np.pi * phi
    R_hat = np.stack(
        [np.cos(angle), np.sin(angle), np.zeros_like(angle)], axis=1
    )
    dR = np.einsum("ij,ij->i", g2, R_hat)
    dZ = g2[:, 2]
    d2R = np.einsum("ij,ij->i", g22, R_hat)
    d2Z = g22[:, 2]
    return np.abs(dR * d2Z - dZ * d2R) / (dR**2 + dZ**2) ** 1.5


def _poloidal_xyz_and_curve_curvature(spline_surf, phi, ntheta):
    r"""
    (X, Y, Z) and kappa(theta) -- the (R, Z) planar cross-section
    curve's own curvature, not the surface's mean curvature -- together
    at `ntheta` poloidal samples around the fixed-`phi` (a fraction in
    [0, 1), same convention as Surface.cross_section) "zeta = const"
    cut through the surface, from a single `_gamma_and_derivs` call.

    Computed analytically from spline_surf's own recently-implemented
    gammadash2/gammadash2dash2 (via `_gamma_and_derivs`, which --
    unlike `.surface_curvatures()` -- can be evaluated at arbitrary
    explicit (phi, theta) pairs instead of only the surface's own
    fixed, read-only `quadpoints_phi`/`quadpoints_theta` grid) -- see
    _planar_curve_curvature for the actual curvature formula and why
    it (not the surface's mean curvature H) is the right quantity for
    "how sharp does this cross-section look".

    Returns (xyz, kappa): xyz is (ntheta, 3), kappa is (ntheta,).
    """
    theta = np.linspace(0, 1, ntheta, endpoint=False)
    phi_arr = np.full(ntheta, phi)
    xyz, _g1, g2, _g11, _g12, g22 = spline_surf._gamma_and_derivs(
        phi_arr, theta, max_deriv=2
    )
    return xyz, _planar_curve_curvature(phi_arr, g2, g22)


def poloidal_curve_curvature(spline_surf, phi, ntheta=200):
    r"""
    kappa(theta) only -- see _poloidal_xyz_and_curve_curvature for the
    formula and the (X, Y, Z) points this discards.

    Returns kappa, an (ntheta,) array.
    """
    _, kappa = _poloidal_xyz_and_curve_curvature(spline_surf, phi, ntheta)
    return kappa


def max_curvature_by_z_half_residuals(spline_surf, phi_targets, ntheta=200):
    """
    Two residuals per phi in phi_targets -- the maximum (R, Z)
    cross-section curve curvature among poloidal samples on that cross
    section's +Z half and, separately, its -Z half (split at the
    midpoint between the loop's own max and min Z) -- instead of one
    whole-loop max.

    Not used as an optimization objective any more (see the module
    docstring for why maximizing curvature was abandoned in favor of
    corner_angle_residuals' wedge shape) -- kept only as a diagnostic,
    reported before/after the solve to see how the resulting smooth
    curve's curvature actually responds to the wedge-shaping objective.

    Returns a flat (2 * len(phi_targets),) array: [+Z max, -Z max] for
    the first phi, then the next, etc.
    """
    residuals = []
    for phi in phi_targets:
        xyz, kappa = _poloidal_xyz_and_curve_curvature(spline_surf, phi, ntheta)
        Z = xyz[:, 2]
        z_mid = 0.5 * (Z.max() + Z.min())
        above = Z >= z_mid
        residuals.append(kappa[above].max())
        residuals.append(kappa[~above].max())
    return np.array(residuals)


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


def find_corner_tip_indices(spline_surf):
    """
    For each cross section, the FULL (mirrored, get_r_ctrl_full-order)
    control-point index of its highest-Z point and, separately, its
    lowest-Z point -- the "tip" of each corner -- from the surface's
    CURRENT dofs. Same z_full construction
    freeze_except_corner_neighborhoods (and, previously,
    freeze_extremal_z_control_points) uses, just returning the single
    argmax/argmin index per cross section instead of the top/bottom
    `n_extreme`.

    Returns a list of (tip_plus, tip_minus) full-index pairs, one per
    cross section in spline_surf.cs_list.
    """
    cs_zeta, cs_angle = spline_surf.get_cs_zeta_angle()
    axis_pos, e1, e2 = spline_surf._axis_local_basis(cs_zeta)
    tips = []
    for i, cs in enumerate(spline_surf.cs_list):
        r_full = cs.get_r_ctrl_full()
        theta_full = cs.get_theta_ctrl_full()
        offset = np.outer(
            r_full * np.cos(theta_full + cs_angle[i]), e1[i]
        ) + np.outer(r_full * np.sin(theta_full + cs_angle[i]), e2[i])
        z_full = (axis_pos[i] + offset)[:, 2]
        tips.append((int(np.argmax(z_full)), int(np.argmin(z_full))))
    return tips


def freeze_except_corner_neighborhoods(spline_surf, m=3):
    """
    Freeze every dof in spline_surf except, within each cross section,
    the r_i/theta_i pairs belonging to the `m` LEG control points on
    each side of the highest-Z tip, and separately the lowest-Z tip
    (2*m points freed per corner -- the tip itself stays FIXED --
    wrapping around the loop's full index space via
    find_corner_tip_indices).

    The tip is deliberately left fixed: it anchors the corner's
    position, and corner_angle_residuals' tip-angle residual still
    responds to it moving, since that angle is computed from the
    tip's (fixed) position together with its immediate neighbors'
    (free) positions -- so the apex angle still changes as the legs
    swing in/out, without the tip itself needing its own 2 (r, theta)
    dofs to wander. It also means build_no_crossing_linear_constraint/
    build_radius_smoothness_linear_constraint (sharpen_linear_constraint.py)
    automatically treat the tip as an internal fixed anchor splitting
    each corner into two independent, individually-anchored legs, with
    no changes needed there.

    This replaces freeing a single sharpest point (or the top/bottom
    `n_extreme` by Z, an oscillation-prone setup with far more free
    dofs than residuals): see corner_angle_residuals' docstring for
    why an extended neighborhood -- straight legs meeting at one
    finite tip angle -- is the thing we actually want to optimize now.

    A z_sym cross section's mirror-axis-pinned theta_0 (and, for even
    n_ctrl_pts, theta_{n_pts-1} = pi) is left fixed even if it falls
    inside a neighborhood -- see freeze_extremal_z_control_points's
    (removed) docstring reasoning: unfixing it would break the
    up-down symmetry the whole reflect-and-tile construction assumes.

    Returns a list (one entry per cross section) of
    [neighborhood_plus, neighborhood_minus], each a (2*m+1,) array of
    FULL indices (tip included, at the middle index) -- for
    corner_angle_residuals, which still needs the tip's position to
    compute the tip's own angle residual even though it's fixed.
    """
    spline_surf.fix_all()
    tips = find_corner_tip_indices(spline_surf)

    corners = []
    for i, cs in enumerate(spline_surf.cs_list):
        n_ctrl_pts = cs.n_ctrl_pts
        n_pts = cs.n_pts
        full_to_stored = _full_to_stored_index_map(cs)

        pinned_theta = {0}
        if cs.z_sym and cs.n_ctrl_pts % 2 == 0:
            pinned_theta.add(n_pts - 1)

        cs_corners = []
        for tip in tips[i]:
            neighborhood = np.array(
                [(tip + k) % n_ctrl_pts for k in range(-m, m + 1)]
            )
            tip_stored = full_to_stored[tip]
            stored_idx = np.unique(full_to_stored[neighborhood])
            leg_stored_idx = stored_idx[stored_idx != tip_stored]
            for j in leg_stored_idx:
                cs.unfix(f"r_{j}")
                if not (cs.z_sym and j in pinned_theta):
                    cs.unfix(f"theta_{j}")
            cs_corners.append(neighborhood)
        corners.append(cs_corners)

        proc0_print(
            f"cross section {i}: +Z corner neighborhood full indices "
            f"{cs_corners[0].tolist()}, -Z corner neighborhood full "
            f"indices {cs_corners[1].tolist()}"
        )
    return corners


def _local_polar_xy(r_full, theta_full):
    """
    (r*cos(theta), r*sin(theta)) in a cross section's own local 2D
    plane. Rotation by the cross section's own cs_angle is omitted --
    it's the same constant offset for every point in one cross
    section, so it cancels out in any angle-between-vectors
    computation (corner_angle_residuals) anyway.
    """
    return np.stack(
        [r_full * np.cos(theta_full), r_full * np.sin(theta_full)], axis=1
    )


def corner_angle_residuals(spline_surf, corners, goal_angle):
    r"""
    One residual per control point in every corner neighborhood (from
    freeze_except_corner_neighborhoods): that point's own local
    interior (turning) angle, against its immediate polygon neighbors
    in its cross section's local 2D plane, minus a target -- `goal_angle`
    for the neighborhood's own tip (the actual apex/X-point crossing
    angle: the angle between the segment from the nearest leg point to
    the tip, and the segment from the tip to the nearest leg point on
    the opposite side), and `pi` (collinear -- no bend at all) for
    every other point in the neighborhood (the `m` leg points on each
    side).

    interior angle at point P, with polygon neighbors P_minus, P_plus:
        n1 = normalize(P_minus - P), n2 = normalize(P_plus - P)
        angle = arccos(n1 . n2)
    ranges over [0, pi]: pi means P sits exactly on the straight line
    through its neighbors (no bend at all); 0 means P has folded all
    the way back onto its own neighbors (an infinitely sharp spike).

    Why a whole neighborhood (straight legs + one apex), rather than
    one maximally-sharp point: maximizing curvature at a single point
    (a second-derivative, unbounded quantity -- there's no natural
    ceiling, so the optimizer just keeps demanding more forever) is
    what produced the earlier oscillation, no matter which curvature
    formula or how many points fed it. A real magnetic X-point isn't
    an infinitely sharp point either -- locally it's two STRAIGHT legs
    meeting at a well-defined, FINITE opening angle -- so the legs'
    own points should stay straight (target pi), not share the tip's
    small angle: sharing one small target across every point in the
    neighborhood would instead round the whole neighborhood into a
    circular-arc-like curve (constant turning angle at every vertex is
    exactly what a regular polygon/circle looks like), not a wedge.
    This -- tip at goal_angle, legs held straight -- is what actually
    reproduces the local X-point geometry with a smooth spline, and
    each residual is still bounded/first-derivative-only, unlike
    curvature.

    `goal_angle` is the wedge's opening angle. There's no
    universally-correct value -- pi/2 is a common divertor X-point
    crossing angle, but the right value for a specific target
    equilibrium (e.g. a "lemon"-style multi-X-point design) depends on
    the physics design intent -- treat it as a knob to scan rather
    than a fixed constant.

    The tip's own (r, theta) are left FIXED by
    freeze_except_corner_neighborhoods (only the `m` leg points on
    each side are free) -- its angle residual still moves, though,
    since it's computed from the tip's (fixed) position together with
    its immediate neighbors' (free) positions, so the apex angle still
    responds as the legs swing in/out.

    corners : output of freeze_except_corner_neighborhoods -- for each
        cross section, a list of two arrays of FULL (mirrored,
        get_r_ctrl_full-order) indices, one per corner, tip at the
        middle index of each array (index m, since the array runs
        tip-m, ..., tip, ..., tip+m).
    goal_angle : target interior angle in radians for the tip only.

    Returns a flat array, one residual per point across every
    neighborhood (in cross-section, then corner, then point order).
    """
    residuals = []
    for cs, cs_corners in zip(spline_surf.cs_list, corners):
        n_ctrl_pts = cs.n_ctrl_pts
        xy = _local_polar_xy(cs.get_r_ctrl_full(), cs.get_theta_ctrl_full())
        for neighborhood in cs_corners:
            tip = neighborhood[len(neighborhood) // 2]
            for idx in neighborhood:
                n1 = xy[(idx - 1) % n_ctrl_pts] - xy[idx]
                n2 = xy[(idx + 1) % n_ctrl_pts] - xy[idx]
                cos_angle = np.dot(n1, n2) / (
                    np.linalg.norm(n1) * np.linalg.norm(n2)
                )
                angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
                target = goal_angle if idx == tip else np.pi
                residuals.append(angle - target)
    return np.array(residuals)


def dof_deviation_residuals(spline_surf, x0):
    """
    One residual per free dof: its current value minus `x0` (a
    snapshot of spline_surf.x taken right after
    freeze_except_corner_neighborhoods, before the solve starts).

    Paired below with goal=0 and a small weight, this is a light
    Tikhonov-style regularizer that pulls every free dof back toward
    its own starting value. corner_angle_residuals now supplies one
    residual per free control point (nearly 1:1 against the r/theta
    dofs, unlike the old single-point-per-Z-half objective that left
    most of freeze_extremal_z_control_points' freed dofs in an
    unconstrained null space), so this matters much less than it used
    to -- but each point still has 2 dofs (r, theta) against its 1
    angle residual, so a small per-point null space remains. Keep this
    term's weight small relative to the corner term's: it should only
    damp that leftover null-space direction, not resist the one
    direction that actually shapes the wedge.

    `x0` must be built from the SAME free/fixed dof selection this is
    evaluated against (i.e. captured after
    freeze_except_corner_neighborhoods, not before) -- spline_surf.x
    only contains free dofs, in dof_names order, so a mismatched
    selection would compare unrelated dofs.
    """
    return np.asarray(spline_surf.x) - x0


def blended_cross_section_local(spline_surf, cs_zeta_value, ntheta=200):
    """
    The ACTUAL (fully tensor-product-blended) surface's cross section
    near one control row, evaluated at u=theta samples and v fixed at
    that row's own physical zeta, then projected into that row's own
    local 2D frame -- the SAME (axis_pos, e1, e2) frame
    cross_section_xy's raw (r*cos(theta), r*sin(theta)) control curve
    is already expressed in (SurfaceBSpline._axis_local_basis: Bishop
    frame (-N, -B) when use_bishop_frame, else the fixed
    (-R_hat, Z_hat) phi-plane frame).

    v = cs_zeta_value (rather than the surface's own v-knot for this
    row, which would need a Greville-abscissa-style lookup into
    knots_v) is exact when cs_equispaced=True and
    knot_parametrization="uniform" (both the cs_zeta spacing and the
    internal uniform v-knot spacing are then the same evenly-spaced
    sequence over the same domain, by construction) -- true for this
    script's spline_kwargs. For a chord-length or non-equispaced
    surface this is only an approximation of "this row's own v".

    Unlike cross_section_xy's per-row-only curve (built purely from
    that row's own control points, degree p_u in theta only), this
    DOES include the real v-direction blending with neighboring cross
    sections' control points -- so it's the thing to compare the
    control net's own isolated curve against, to see how much the
    neighbors are actually pulling the true surface away from this
    row's own control polygon.

    Returns an (ntheta, 2) array of local (e1, e2) coordinates.
    """
    theta = np.linspace(0, 2 * np.pi, ntheta, endpoint=False)
    v = np.full(ntheta, cs_zeta_value)
    x, y, z = spline_surf.surf_callable(theta, v)
    axis_pos, e1, e2 = spline_surf._axis_local_basis(np.array([cs_zeta_value]))
    rel = np.stack([x, y, z], axis=1) - axis_pos[0]
    return np.stack([rel @ e1[0], rel @ e2[0]], axis=1)


def widen_violated_bounds(spline_surf, margin=0.1):
    """
    Widen any dof's box bounds that the surface's CURRENT x already
    violates, so it's a feasible starting point for
    least_squares_mpi_solve.

    Needed whenever an externally-optimized dofs array (e.g. saved from
    a *different* script's run, as here) is loaded into a freshly-built
    SurfaceBSpline: theta bounds are set as "midpoint to neighbor" at
    whatever positions the theta values happen to be in at construction
    (or refine_poloidal) time -- the default evenly-spaced circle, or
    wherever refine_poloidal() finds them -- tailored to THAT
    configuration, not to the real, differently-shaped values being
    loaded afterward. Loading a real optimized shape into a fresh
    structure this way will generically violate whatever bounds that
    structure happens to have, regardless of whether the fresh
    structure was built via refine_poloidal() or by directly doubling
    points_per_cs -- both compute bounds before the real values are
    loaded in.

    margin : extra slack added on the loaded-value side of the
        violated bound (so it isn't left sitting exactly on the new
        bound either).

    set_lower_bound/set_upper_bound only accept a dof's *local*,
    unqualified name (e.g. "theta_1") and must be called on the child
    object that actually owns it (a CrossSectionFixedZeta or the axis)
    -- calling them on spline_surf itself with the fully-qualified name
    from spline_surf.dof_names (e.g. "CrossSectionFixedZeta6:theta_1")
    raises ValueError, since spline_surf's own local dofs are just its
    cs_zeta/cs_angle values, not its children's -- so this iterates the
    actual owning objects directly instead.
    """
    for owner in [spline_surf.axis] + spline_surf.cs_list:
        lb, ub = owner.bounds
        x = np.asarray(owner.x)
        local_names = [n.split(":", 1)[1] for n in owner.dof_names]
        for name, xi, lbi, ubi in zip(local_names, x, lb, ub):
            if xi < lbi:
                owner.set_lower_bound(name, xi - margin)
            elif xi > ubi:
                owner.set_upper_bound(name, xi + margin)


def print_active_bounds(prob, rtol=1e-6, atol=1e-10):
    """
    Print every free dof currently sitting at (within
    atol + rtol*|bound|) its own lower or upper bound -- i.e. box
    constraints the solver is actually being limited by, as opposed to
    ones just sitting unused far from the current value. Silent (one
    "none active" line) if none are active.

    prob : anything exposing .x, .dof_names, .lower_bounds,
        .upper_bounds in matching order -- e.g. spline_surf itself, or
        (more correctly, if other Optimizables in its dependency graph
        such as vmec/qs also contribute free dofs) the
        LeastSquaresProblem actually passed to least_squares_mpi_solve.

    Infinite bounds (e.g. from disabling box bounds via
    prob.upper_bounds = np.inf * ...) never count as active.
    """
    x = np.asarray(prob.x)
    lb = np.asarray(prob.lower_bounds)
    ub = np.asarray(prob.upper_bounds)
    names = prob.dof_names

    at_lower = np.isfinite(lb) & (np.abs(x - lb) <= atol + rtol * np.abs(lb))
    at_upper = np.isfinite(ub) & (np.abs(x - ub) <= atol + rtol * np.abs(ub))

    active = np.nonzero(at_lower | at_upper)[0]
    if len(active) == 0:
        proc0_print("No dof is currently at a bound.")
        return

    proc0_print(f"{len(active)} dof(s) currently at a bound:")
    for i in active:
        which = "lower" if at_lower[i] else "upper"
        bound_val = lb[i] if at_lower[i] else ub[i]
        proc0_print(f"  {names[i]}: x={x[i]!r} at {which} bound {bound_val!r}")


def plot_control_points_free_status(spline_surf, ntheta=200):
    """
    One 2D subplot per cross section, in the same local (control-net)
    frame plot_cross_sections_aligned uses, showing that cross
    section's control polygon with each physical control point (the
    full, mirrored-for-z_sym set cross_section_xy's ctrl_xy returns)
    colored green if it's currently free to move and gray if it's
    fixed -- e.g. to visually confirm freeze_except_corner_neighborhoods
    picked the points you'd expect.

    A control point counts as free if either its r or theta dof is
    free: freeze_except_corner_neighborhoods only ever unfixes both
    together, except for a z_sym cross section's mirror-axis-pinned
    theta_0 (and theta_{n_pts-1} = pi for even n_ctrl_pts), where only
    r gets unfrozen.

    Returns (fig, axs).
    """
    n_cs = spline_surf.n_cs
    fig, axs = plt.subplots(1, n_cs, squeeze=False, figsize=(4 * n_cs, 4))
    axs = axs[0]
    for i, (ax, cs) in enumerate(zip(axs, spline_surf.cs_list)):
        curve_xy, ctrl_xy = spline_surf.cross_section_xy(cs, n_samples=ntheta)
        curve_closed = np.vstack([curve_xy, curve_xy[:1]])
        ax.plot(curve_closed[:, 0], curve_closed[:, 1], alpha=0.4, zorder=1)

        full_to_stored = _full_to_stored_index_map(cs)
        is_free = np.array(
            [
                cs.is_free(f"r_{j}") or cs.is_free(f"theta_{j}")
                for j in full_to_stored
            ]
        )
        colors = np.where(is_free, "tab:green", "tab:gray")
        ax.scatter(ctrl_xy[:, 0], ctrl_xy[:, 1], c=colors, zorder=2)

        ax.set_aspect("equal")
        ax.set_title(f"cross section {i}")

    axs[0].legend(
        handles=[
            plt.Line2D(
                [], [], marker="o", ls="", color="tab:green", label="free"
            ),
            plt.Line2D(
                [], [], marker="o", ls="", color="tab:gray", label="fixed"
            ),
        ]
    )
    fig.tight_layout()
    return fig, axs


def plot_cross_sections_aligned(spline_surf, ntheta=200):
    """
    Like SurfaceBSpline.plot_cross_sections, but for the surface's
    current dofs only, with one extra curve per subplot:
    blended_cross_section_local's true-blended-surface cut, overlaid on
    the existing control-polygon + row-only-curve view, so the two are
    directly comparable in the same local frame the surface actually
    uses (Bishop frame here, since use_bishop_frame=True) -- answers
    "is the actual surface's corner as sharp as the control net alone
    suggests."

    Returns (fig, axs).
    """
    cs_zeta, _ = spline_surf.get_cs_zeta_angle()
    n_cs = spline_surf.n_cs
    fig, axs = plt.subplots(1, n_cs, squeeze=False, figsize=(4 * n_cs, 4))
    axs = axs[0]
    for i, (ax, cs, zeta_i) in enumerate(
        zip(axs, spline_surf.cs_list, cs_zeta)
    ):
        _, ctrl_xy = spline_surf.cross_section_xy(cs, n_samples=ntheta)
        ctrl_closed = np.vstack([ctrl_xy, ctrl_xy[:1]])
        ax.plot(
            ctrl_closed[:, 0],
            ctrl_closed[:, 1],
            ls="--",
            marker="o",
            alpha=0.5,
            label="control polygon",
        )
        # ax.plot(
        #     curve_closed[:, 0],
        #     curve_closed[:, 1],
        #     label="control net curve (row-only)",
        # )

        blended_xy = blended_cross_section_local(
            spline_surf, zeta_i, ntheta=ntheta
        )
        blended_closed = np.vstack([blended_xy, blended_xy[:1]])
        ax.plot(
            blended_closed[:, 0],
            blended_closed[:, 1],
            ls=":",
            label="actual surface (blended)",
        )

        ax.set_aspect("equal")
        ax.set_title(f"cross section {i}")
    axs[0].legend()
    fig.tight_layout()
    return fig, axs


def plot_before_after(spline_surf, spline_surf_before, ntheta=200):
    """
    Every before/after comparison plot used once the corner-sharpening
    solve has finished, consolidated into one call -- ALWAYS comparing
    against a genuinely INDEPENDENT SurfaceBSpline snapshot
    (spline_surf_before), never a bare list(spline_surf.cs_list):
    CrossSectionFixedZeta objects are mutated IN PLACE (every
    cs.set(...)/spline_surf.x = ... call mutates spline_surf's own
    child objects, not a copy of them), so a snapshot taken as
    list(spline_surf.cs_list) and a later read of spline_surf.cs_list
    point at the exact same (by-then-already-mutated) objects --
    "before" and "after" would silently show the identical, already-
    sharpened shape twice. Build spline_surf_before the same way the
    callers here already build it for the free/fixed comparison below
    (same spline_kwargs/dofs/refine_poloidal call count as
    spline_surf, and never touched afterward) and this problem can't
    happen, since it's a wholly separate object graph.

    - SurfaceBSpline.plot_cross_sections' before/after cross-section
      overlay: one 2D subplot per cross section, both curves drawn on
      the SAME axes (spline_surf_before.cs_list vs.
      spline_surf.cs_list).
    - plot_cross_sections_aligned, for the CURRENT (post-solve)
      surface only -- shows the actual blended surface, not a
      per-cross-section snapshot, so there's no meaningful "before"
      version of it.
    - One 3D plot with BOTH surfaces drawn on the SAME Axes3D
      (spline_surf_before translucent/gray underneath, spline_surf in
      its normal color on top) -- so the overall 3D shape change is
      visible directly, not just per cross section.
    - plot_control_points_free_status for spline_surf_before and
      spline_surf, side by side, so the free/fixed control-point
      selection can be visually compared before vs. after.

    spline_surf : the SurfaceBSpline holding the CURRENT (post-solve)
        dofs.
    spline_surf_before : a separate SurfaceBSpline holding the
        PRE-solve dofs (same construction/refine_poloidal calls as
        spline_surf, just never mutated afterward).
    ntheta : samples per cross section, passed through to
        plot_cross_sections_aligned/plot_control_points_free_status.

    Draws nothing and returns immediately on any rank other than 0.
    """
    if MPI.COMM_WORLD.rank != 0:
        return

    spline_surf.plot_cross_sections(
        [spline_surf_before.cs_list, spline_surf.cs_list],
        labels=["before", "after"],
    )
    plot_cross_sections_aligned(spline_surf, ntheta=ntheta)

    # fig = plt.figure()
    # ax = fig.add_subplot(projection="3d")
    # common_kwargs = dict(
    #     _ctrl_points=False,
    #     _ctrl_points_full=False,
    #     _pseudo_axis=False,
    #     _pseudo_axis_ctrl_pts=False,
    #     _centroid_axis=False,
    #     _rtz_vectors=False,
    # )
    # spline_surf_before.plot(
    #     ax=ax,
    #     _surf_kwargs={"alpha": 0.25, "color": "gray", "rcount": 64, "ccount": 64},
    #     **common_kwargs,
    # )
    # spline_surf.plot(
    #     ax=ax,
    #     _surf_kwargs={"alpha": 0.5, "color": "tab:red", "rcount": 64, "ccount": 64},
    #     **common_kwargs,
    # )
    # ax.set_title("before (gray) vs. after (red), same axes")
    # plt.show()

    plot_control_points_free_status(spline_surf_before, ntheta=ntheta)
    plot_control_points_free_status(spline_surf, ntheta=ntheta)
    plt.show()


if __name__ == "__main__":
    mpi = MpiPartition()
    mpi.write()

    proc0_print("Running figures_spline_paper/corner_sharpening/sharpen.py")
    proc0_print("==================================================")

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
    spline_surf_init = SurfaceBSpline(**spline_kwargs)
    proc0_print(f"spline_surf.dof_names: {spline_surf.dof_names}")
    spline_surf.x = dofs
    spline_surf_init.x = dofs
    spline_surf.refine_poloidal()
    spline_surf.refine_poloidal()
    spline_surf.refine_poloidal()
    spline_surf.refine_poloidal()

    # spline_surf_init stays untouched from here on -- it's the
    # "before" snapshot plot_before_after compares spline_surf
    # against, so it needs the SAME refine_poloidal call count (an
    # exact doubling, so this doesn't change the curve it represents,
    # only its resolution) to match spline_surf's structure.
    spline_surf_init.refine_poloidal()
    spline_surf_init.refine_poloidal()
    spline_surf_init.refine_poloidal()
    spline_surf_init.refine_poloidal()

    # spline_surf.refine_poloidal()

    # spline_surf.to_RZFourier(plot=True)
    # plot_cross_sections_aligned(spline_surf)
    # spline_surf.plot()
    # plt.show()
    # Number of control points held STRAIGHT (collinear) on EACH leg
    # of a corner, on each side of the tip (2*M_CORNER + 1 freed per
    # corner, tip included) -- only the tip itself targets GOAL_ANGLE
    # below; see freeze_except_corner_neighborhoods'/
    # corner_angle_residuals' docstrings for why a wedge (straight
    # legs + one finite tip angle), not a single point, is targeted.
    M_CORNER = 6

    proc0_print(f"Before freezing: ndofs={len(spline_surf.x)}")
    corners = freeze_except_corner_neighborhoods(spline_surf, m=M_CORNER)
    proc0_print(f"After freezing: ndofs={len(spline_surf.x)}")

    if MPI.COMM_WORLD.rank == 0:
        plot_control_points_free_status(spline_surf)
        plt.show()

    # Snapshot of the free dofs' own starting values, for
    # dof_deviation_residuals -- must be taken with the SAME free/fixed
    # selection it'll be compared against later (i.e. after freezing).
    x0 = np.copy(spline_surf.x)

    # Target cross sections: the surface's own n_cs cross sections, at
    # their actual physical zeta (get_cs_zeta_angle returns radians in
    # [0, pi/nfp]; cross_section wants a fraction in [0, 1)) -- used
    # only by the max_curvature_by_z_half_residuals diagnostic below.
    cs_zeta, _ = spline_surf.get_cs_zeta_angle()
    phi_targets = cs_zeta / (2 * np.pi)
    NTHETA = 200
    # The wedge's opening (interior) angle -- see corner_angle_residuals'
    # docstring: no universally-correct value, this is a knob to scan
    # for the actual target equilibrium's X-point crossing angle, not
    # a fixed constant. pi/2 here is just a starting point.
    GOAL_ANGLE = np.pi / 2 # np.pi / 2
    # Small relative to the corner/QS weights below -- see
    # dof_deviation_residuals' docstring. Just needs to be big enough
    # to damp the remaining per-point (r, theta) null-space direction
    # without measurably resisting the wedge shape itself.
    REG_WEIGHT = 1

    corner_obj = make_optimizable(
        corner_angle_residuals, spline_surf, corners, GOAL_ANGLE
    )
    regularization_obj = make_optimizable(
        dof_deviation_residuals, spline_surf, x0
    )

    vmec = Vmec.vmec_from_surf(
        nfp=spline_surf.nfp,
        surf=spline_surf,
        mpi=mpi,
        ns=25,
        M=12,
        N=12,
        ftol=1e-8,
        niter=5000,
        # verbose=True,
    )

    qs = QuasisymmetryRatioResidual(
        vmec,
        np.arange(0, 1.01, 0.1),  # Radii to target
        helicity_m=1,
        helicity_n=0,  # -1
    )  # (M, N) you want in |B|
    # nonlinear constraints
    # tuples_nlc = [(vmec.aspect, -np.inf, 8), (vmec.mean_iota, -1.05, -1.0)]

    # define problem
    prob = LeastSquaresProblem.from_tuples(
        [
            (qs.residuals, 0, 1000),
            (corner_obj.J, 0, 1),
            (regularization_obj.J, 0, REG_WEIGHT),
        ]
    )

    # prob.upper_bounds = np.inf * np.ones_like(prob.upper_bounds)
    # prob.lower_bounds = -np.inf * np.ones_like(prob.upper_bounds)

    proc0_print(
        f"Initial corner interior angles [rad] (goal = {GOAL_ANGLE:.4f}):",
        corner_angle_residuals(spline_surf, corners, GOAL_ANGLE) + GOAL_ANGLE,
    )
    proc0_print(
        "Initial [+Z max, -Z max] poloidal curve curvature per cross"
        " section (diagnostic only -- not part of the objective; just"
        " shows how the resulting smooth curve's curvature responds):",
        max_curvature_by_z_half_residuals(
            spline_surf, phi_targets, ntheta=NTHETA
        ).reshape(-1, 2),
    )

    proc0_print("Beginning optimization")
    proc0_print(f"ndofs: {len(prob.x)}")
    proc0_print(f"dofs names: {prob.dof_names}")

    least_squares_mpi_solve(
        prob,
        mpi,
        grad=True,
        rel_step=1e-12,
        abs_step=1e-6,
        x_scale="jac",
        max_nfev=100,
    )
    xopt = prob.x

    proc0_print("")
    print_active_bounds(prob)

    # evaluate the solution
    spline_surf.x = xopt

    plot_before_after(spline_surf, spline_surf_init, ntheta=NTHETA)
    proc0_print("")

    proc0_print(
        f"Final corner interior angles [rad] (goal = {GOAL_ANGLE:.4f}):",
        corner_angle_residuals(spline_surf, corners, GOAL_ANGLE) + GOAL_ANGLE,
    )
    proc0_print(
        "Final [+Z max, -Z max] poloidal curve curvature per cross"
        " section (diagnostic only -- not part of the objective):",
        max_curvature_by_z_half_residuals(
            spline_surf, phi_targets, ntheta=NTHETA
        ).reshape(-1, 2),
    )
    proc0_print("spline_surf.x:", repr(xopt))

    proc0_print("")
    proc0_print("End of figures_spline_paper/corner_sharpening/sharpen.py")
    proc0_print("=================================================")
