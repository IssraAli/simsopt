#!/usr/bin/env python
"""
Sister script to sharpen.py: same wedge-shaping corner objective
(corner_angle_residuals over freeze_except_corner_neighborhoods,
imported directly from sharpen.py -- importing it only pulls in its
function/class definitions, not its __main__ block), but replacing
sharpen.py's per-dof box bounds with two genuine LINEAR INEQUALITY
CONSTRAINTS: adjacent theta control points can't cross/reorder
(build_no_crossing_linear_constraint), and adjacent radius control
points can't jump by more than a data-driven amount
(build_radius_smoothness_linear_constraint) -- instead of a fixed box
bound per point.

Why: CrossSectionFixedZeta's box bounds are set once, at
construction/refine time, as "midpoint to neighbor" (theta) or a wide
fixed range (r) -- either more restrictive than necessary (if the
neighbor never actually needs to move) or silently stale (if the
neighbor DOES move -- which is exactly what happens once a corner
neighborhood's several points are all free at once -- the bound was
computed against the neighbor's OLD position). There's no single
non-arbitrary width to widen either to. A linear constraint directly
on each PAIR enforces the actual thing that matters and stays correct
for however far both points move, with nothing to pre-derive or widen
by hand.

Ordering theta alone is not enough on its own, though: a perfectly
monotonic theta sequence with a wildly oscillating radius between
consecutive points still produces a jagged, self-crossing-looking
polygon (theta marches forward correctly; r zigzags) -- exactly the
kind of shape the no-crossing constraint alone does not prevent. The
radius-smoothness constraint closes that gap.

scipy.optimize.least_squares (the solver behind least_squares_mpi_solve,
used in sharpen.py) only supports box bounds, not general linear
constraints. So this uses simsopt's ConstrainedProblem +
constrained_mpi_solve instead (SLSQP/trust-constr/COBYLA via
scipy.optimize.minimize), which means the objective has to be a single
SCALAR (a hand-combined weighted sum of squared residuals), not a
least-squares residual vector -- see total_scalar_objective.
"""

import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI
from simsopt._core import make_optimizable
from simsopt.geo import SurfaceBSpline
from simsopt.mhd import QuasisymmetryRatioResidual, Vmec
from simsopt.objectives import ConstrainedProblem
from simsopt.solve import constrained_mpi_solve
from simsopt.util import MpiPartition, proc0_print

from sharpen import (
    _full_to_stored_index_map,
    corner_angle_residuals,
    dof_deviation_residuals,
    freeze_except_corner_neighborhoods,
    max_curvature_by_z_half_residuals,
    plot_before_after,
    plot_control_points_free_status,
)


def widen_free_theta_bounds(spline_surf):
    """
    Set every currently-free theta_j dof's box bounds to
    (-inf, +inf), on every cross section in spline_surf.cs_list.

    Only theta -- r's existing [0, 1] box bounds are left in place as
    a coarse global sanity check (no negative or absurd radius); the
    LOCAL smoothness job a tighter r bound might otherwise have
    attempted is instead handled properly by
    build_radius_smoothness_linear_constraint, and theta's ordering by
    build_no_crossing_linear_constraint.
    """
    for cs in spline_surf.cs_list:
        for j in range(cs.n_pts):
            name = f"theta_{j}"
            if cs.is_free(name):
                cs.set_lower_bound(name, -np.inf)
                cs.set_upper_bound(name, np.inf)


def _build_adjacent_diff_linear_constraint(
    spline_surf, dof_names, dof_prefix, lower_diff, upper_diff
):
    r"""
    Shared core of build_no_crossing_linear_constraint (dof_prefix
    "theta") and build_radius_smoothness_linear_constraint (dof_prefix
    "r"): build (A, l, u) for a scipy LinearConstraint enforcing
    lower_diff <= x_{j+1} - x_j <= upper_diff for every pair of
    adjacent STORED "{dof_prefix}_j"/"{dof_prefix}_{j+1}" dofs
    (j = 0 .. n_pts-2) on every cross section in spline_surf.cs_list,
    regardless of which ones happen to be free right now.

    Three cases per adjacent pair, since a corner neighborhood's
    OUTERMOST free point still has a FIXED anchor just beyond it the
    constraint must account for too:
      - both free: a genuine 2-dof linear row,
        A row = [..., -1 at x_j, +1 at x_{j+1}, ...],
        l = lower_diff, u = upper_diff.
      - only x_j free (x_{j+1} fixed at constant c): row has a single
        -1 at x_j; from lower_diff <= c - x_j <= upper_diff,
        l = lower_diff - c, u = upper_diff - c (note the row itself is
        -x_j, so bounds get negated-and-swapped relative to that
        derivation -- see the code).
      - only x_{j+1} free (x_j fixed at constant c): row has a single
        +1 at x_{j+1}; l = lower_diff + c, u = upper_diff + c.
      - both fixed: no constraint needed (nothing to optimize here).
    A fixed neighbor's value is read via cs.get(...), which works
    regardless of free/fixed status, and is a genuine constant for the
    whole solve (fixed dofs don't change), so this case never goes
    stale the way a pre-computed box bound could.

    dof_names : the FULL free-dof-name ordering (e.g. the
        ConstrainedProblem/its objective's own `.dof_names`) the
        returned A's columns must match.

    Returns (A, l, u): A is (n_constraints, len(dof_names)).
    """
    name_to_col = {name: i for i, name in enumerate(dof_names)}
    n = len(dof_names)

    rows = []
    lowers = []
    uppers = []
    for cs in spline_surf.cs_list:
        for j in range(cs.n_pts - 1):
            name_j = f"{cs.name}:{dof_prefix}_{j}"
            name_j1 = f"{cs.name}:{dof_prefix}_{j + 1}"
            free_j = name_j in name_to_col
            free_j1 = name_j1 in name_to_col
            if not free_j and not free_j1:
                continue

            row = np.zeros(n)
            if free_j and free_j1:
                row[name_to_col[name_j]] = -1.0
                row[name_to_col[name_j1]] = 1.0
                lower, upper = lower_diff, upper_diff
            elif free_j:
                c = cs.get(f"{dof_prefix}_{j + 1}")
                row[name_to_col[name_j]] = -1.0
                # lower_diff <= c - x_j <= upper_diff
                # => c - upper_diff <= x_j <= c - lower_diff
                # => -(c - lower_diff) <= -x_j <= -(c - upper_diff)
                lower, upper = lower_diff - c, upper_diff - c
            else:
                c = cs.get(f"{dof_prefix}_{j}")
                row[name_to_col[name_j1]] = 1.0
                lower, upper = lower_diff + c, upper_diff + c
            rows.append(row)
            lowers.append(lower)
            uppers.append(upper)

    A = np.array(rows) if rows else np.zeros((0, n))
    l = np.array(lowers) if lowers else np.zeros(0)
    u = np.array(uppers) if uppers else np.zeros(0)
    return A, l, u


def build_no_crossing_linear_constraint(spline_surf, dof_names, margin=1e-3):
    r"""
    theta_{j+1} - theta_j >= margin for every pair of adjacent STORED
    theta dofs -- "don't let neighboring control points cross/reorder"
    -- see _build_adjacent_diff_linear_constraint for the general
    construction.

    margin : minimum allowed theta gap, in radians. Just needs to
        stay comfortably above 0 to avoid exactly-coincident points;
        not otherwise a meaningful physical scale, so keep it small
        relative to the surface's actual poloidal point spacing.

    Returns (A, l, u); u is +inf everywhere (only a floor on the
    spacing is enforced, never a ceiling).
    """
    A, l, _ = _build_adjacent_diff_linear_constraint(
        spline_surf, dof_names, "theta", margin, np.inf
    )
    u = np.full(len(l), np.inf)
    return A, l, u


def default_max_radius_jump(spline_surf, factor=3.0):
    """
    `factor` times the largest |r_{j+1} - r_j}| already present
    between adjacent STORED control points, across every cross
    section, in spline_surf's CURRENT (pre-solve) shape -- a
    data-driven scale for build_radius_smoothness_linear_constraint's
    `max_jump`, instead of an arbitrary hardcoded number: it allows
    at least as much local radius variation as the starting shape
    already has (times some slack, `factor`), but not an order of
    magnitude more.
    """
    max_jump = 0.0
    for cs in spline_surf.cs_list:
        r = cs.r_ctrl
        if len(r) > 1:
            max_jump = max(max_jump, np.max(np.abs(np.diff(r))))
    return factor * max_jump


def build_radius_smoothness_linear_constraint(spline_surf, dof_names, max_jump):
    r"""
    -max_jump <= r_{j+1} - r_j <= max_jump for every pair of adjacent
    STORED radius dofs -- see _build_adjacent_diff_linear_constraint
    for the general construction.

    Why this is needed alongside build_no_crossing_linear_constraint:
    keeping theta strictly increasing prevents control points from
    reordering, but says nothing about how much r can swing between
    consecutive points. A wildly oscillating radius with perfectly
    monotonic theta still produces a jagged, self-crossing-looking
    polygon (theta marches forward correctly; r zigzags) -- exactly
    the failure mode a no-crossing constraint on theta ALONE does not
    prevent.

    max_jump : maximum allowed |r_{j+1} - r_j}|. See
        default_max_radius_jump for a data-driven default instead of
        picking this by hand.

    Returns (A, l, u).
    """
    return _build_adjacent_diff_linear_constraint(
        spline_surf, dof_names, "r", -max_jump, max_jump
    )


def print_corner_boundary_anchors(spline_surf, corners):
    """
    For every corner neighborhood (from freeze_except_corner_neighborhoods),
    print the stored index and fixed status of the point immediately
    OUTSIDE each end -- the "adjacent frozen point" the first/last free
    point in that cluster must be bounded against.

    This is a direct, printed check that
    _build_adjacent_diff_linear_constraint actually has an anchor to
    build a boundary row from at both ends of every cluster (it always
    will, as long as the neighborhood doesn't run all the way to a
    z_sym-pinned seam point or wrap past the array's own ends -- both
    called out explicitly below if seen, rather than silently assumed
    fine).
    """
    for i, (cs, cs_corners) in enumerate(zip(spline_surf.cs_list, corners)):
        full_to_stored = _full_to_stored_index_map(cs)
        n_pts = cs.n_pts
        for label, neighborhood in zip(("+Z", "-Z"), cs_corners):
            stored = np.unique(full_to_stored[neighborhood])
            lo, hi = int(stored.min()), int(stored.max())
            left, right = lo - 1, hi + 1
            left_ok = 0 <= left < n_pts
            right_ok = 0 <= right < n_pts
            left_status = (
                f"theta_{left}={cs.get(f'theta_{left}'):.4f}"
                f" (fixed={cs.is_fixed(f'theta_{left}')})"
                if left_ok
                else "NONE -- neighborhood runs to the array's own edge"
            )
            right_status = (
                f"theta_{right}={cs.get(f'theta_{right}'):.4f}"
                f" (fixed={cs.is_fixed(f'theta_{right}')})"
                if right_ok
                else "NONE -- neighborhood runs to the array's own edge"
            )
            proc0_print(
                f"cross section {i} {label} corner: free stored indices"
                f" [{lo}, {hi}] -- left anchor: {left_status};"
                f" right anchor: {right_status}"
            )


def total_scalar_objective(
    spline_surf, corners, goal_angle, qs, qs_weight, x0, reg_weight
):
    """
    Single scalar objective for ConstrainedProblem: the same three
    terms sharpen.py balances via LeastSquaresProblem's per-term
    weights (qs.residuals, corner_angle_residuals,
    dof_deviation_residuals), combined by hand into one
    weight * sum-of-squares total, since ConstrainedProblem needs a
    scalar f(x), not a residual vector.
    """
    corner_res = corner_angle_residuals(spline_surf, corners, goal_angle)
    reg_res = dof_deviation_residuals(spline_surf, x0)
    qs_res = qs.residuals()
    return (
        qs_weight * np.sum(qs_res ** 2)
        + np.sum(corner_res ** 2)
        + reg_weight * np.sum(reg_res ** 2)
    )


if __name__ == "__main__":
    mpi = MpiPartition()
    mpi.write()

    proc0_print(
        "Running figures_spline_paper/corner_sharpening/"
        "sharpen_linear_constraint.py"
    )
    proc0_print("==================================================")

    # Same starting shape as sharpen.py.
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
    spline_surf.x = dofs
    spline_surf_init.x = dofs
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

    # Number of control points held STRAIGHT (collinear) on EACH leg
    # of a corner, on each side of the tip (2*M_CORNER + 1 freed per
    # corner, tip included) -- only the tip itself targets GOAL_ANGLE
    # below; see freeze_except_corner_neighborhoods'/
    # corner_angle_residuals' docstrings (in sharpen.py) for why a
    # wedge (straight legs + one finite tip angle), not a single
    # point, is targeted.
    M_CORNER = 2

    proc0_print(f"Before freezing: ndofs={len(spline_surf.x)}")
    corners = freeze_except_corner_neighborhoods(spline_surf, m=M_CORNER)
    proc0_print(f"After freezing: ndofs={len(spline_surf.x)}")
    print_corner_boundary_anchors(spline_surf, corners)

    # Replace the (restrictive/stale) per-point theta box bounds with
    # an unbounded box + the no-crossing linear constraint built below.
    widen_free_theta_bounds(spline_surf)

    if MPI.COMM_WORLD.rank == 0:
        plot_control_points_free_status(spline_surf)
        plt.show()

    # Snapshot of the free dofs' own starting values, for
    # dof_deviation_residuals.
    x0 = np.copy(spline_surf.x)

    # Used only by the max_curvature_by_z_half_residuals diagnostic.
    cs_zeta, _ = spline_surf.get_cs_zeta_angle()
    phi_targets = cs_zeta / (2 * np.pi)
    NTHETA = 200

    # The wedge's opening (interior) angle -- see corner_angle_residuals'
    # docstring (sharpen.py): no universally-correct value, this is a
    # knob to scan for the actual target equilibrium's X-point
    # crossing angle, not a fixed constant.
    GOAL_ANGLE = np.pi / 6
    QS_WEIGHT = 1000
    # Small relative to QS -- see dof_deviation_residuals' docstring.
    REG_WEIGHT = 1
    # Minimum allowed theta gap between adjacent control points, in
    # radians -- see build_no_crossing_linear_constraint's docstring.
    MIN_THETA_GAP = 1e-3
    # Maximum allowed |r_{j+1} - r_j}| between adjacent control
    # points -- see build_radius_smoothness_linear_constraint's
    # docstring. Computed from the starting shape's own largest
    # existing adjacent-r gap (times a slack factor) rather than
    # picked by hand -- see default_max_radius_jump.
    MAX_RADIUS_JUMP = default_max_radius_jump(spline_surf, factor=3.0)

    vmec = Vmec.vmec_from_surf(
        nfp=spline_surf.nfp,
        surf=spline_surf,
        mpi=mpi,
        ns=25,
        M=12,
        N=12,
        ftol=1e-8,
        niter=5000,
    )

    qs = QuasisymmetryRatioResidual(
        vmec,
        np.arange(0, 1.01, 0.1),  # Radii to target
        helicity_m=1,
        helicity_n=0,
    )

    total_obj = make_optimizable(
        total_scalar_objective,
        spline_surf,
        corners,
        GOAL_ANGLE,
        qs,
        QS_WEIGHT,
        x0,
        REG_WEIGHT,
    )

    A_theta, l_theta, u_theta = build_no_crossing_linear_constraint(
        spline_surf, total_obj.dof_names, margin=MIN_THETA_GAP
    )
    A_r, l_r, u_r = build_radius_smoothness_linear_constraint(
        spline_surf, total_obj.dof_names, max_jump=MAX_RADIUS_JUMP
    )
    A_lc = np.vstack([A_theta, A_r])
    l_lc = np.concatenate([l_theta, l_r])
    u_lc = np.concatenate([u_theta, u_r])

    prob = ConstrainedProblem(total_obj.J, tuple_lc=(A_lc, l_lc, u_lc))

    proc0_print(f"ndofs: {len(prob.x)}")
    proc0_print(f"dof names: {prob.dof_names}")
    proc0_print(
        f"{A_theta.shape[0]} no-crossing (theta) + {A_r.shape[0]}"
        f" radius-smoothness (max jump {MAX_RADIUS_JUMP:.4g}) linear"
        " constraints"
    )

    # Sanity check: the starting point should already be feasible (the
    # initial shape doesn't already cross itself or have a bigger
    # adjacent-r jump than MAX_RADIUS_JUMP allows) -- a violation here
    # means MIN_THETA_GAP/MAX_RADIUS_JUMP is too tight for the
    # starting shape, not that the sharpening solve did anything wrong
    # yet. u_lc is +inf for the theta rows, so the upper-bound slack
    # there is always satisfied trivially.
    Ax0 = A_lc @ prob.x
    lower_slack = Ax0 - l_lc
    upper_slack = u_lc - Ax0
    proc0_print(
        f"Initial linear-constraint slack: min(Ax-l)={lower_slack.min():.3e},"
        f" min(u-Ax)={upper_slack.min():.3e} (both should be >= 0)"
    )

    proc0_print(
        f"Initial corner interior angles [rad] (goal = {GOAL_ANGLE:.4f}):",
        corner_angle_residuals(spline_surf, corners, GOAL_ANGLE) + GOAL_ANGLE,
    )
    proc0_print(
        "Initial [+Z max, -Z max] poloidal curve curvature per cross"
        " section (diagnostic only -- not part of the objective):",
        max_curvature_by_z_half_residuals(
            spline_surf, phi_targets, ntheta=NTHETA
        ).reshape(-1, 2),
    )
    proc0_print(f"Initial scalar objective: {prob.objective():.6e}")

    proc0_print("Beginning optimization")
    constrained_mpi_solve(
        prob,
        mpi,
        grad=True,
        abs_step=1e-6,
        opt_method="SLSQP",
        options={"maxiter": 100},
    )
    xopt = prob.x

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
    proc0_print(f"Final scalar objective: {prob.objective():.6e}")
    proc0_print("spline_surf.x:", repr(xopt))

    proc0_print("")
    proc0_print(
        "End of figures_spline_paper/corner_sharpening/"
        "sharpen_linear_constraint.py"
    )
    proc0_print("=================================================")
