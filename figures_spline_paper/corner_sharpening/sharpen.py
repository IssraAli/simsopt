#!/usr/bin/env python
"""
Corner sharpening via a purely geometric optimization: starting from a
SurfaceBSpline already loaded with a reasonable, roughly-W7-X-like
shape (the `dofs` below, e.g. from shape_match.py's final fit), push
the maximum poloidal curvature of each of its fixed-zeta cross sections
as high as it will go, subject only to the dofs' own box bounds.

No VMEC, no shape-matching target -- this is a standalone test of how
sharp a corner this spline representation can actually produce, using
the "syntax" (MpiPartition, LeastSquaresProblem.from_tuples,
least_squares_mpi_solve call) of examples/2_Intermediate/
stage_one_splines.py, with the QS/aspect-ratio/iota objective terms
replaced by a single geometric curvature-maximization term.
"""

import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI
from simsopt._core import make_optimizable
from simsopt.geo import SurfaceBSpline
from simsopt.objectives import LeastSquaresProblem
from simsopt.solve import least_squares_mpi_solve
from simsopt.util import MpiPartition, proc0_print


def poloidal_curvature(spline_surf, phi, ntheta=200):
    r"""
    Curvature kappa(theta) of the planar (R, Z) cross-section curve at
    fixed toroidal angle `phi` (a fraction in [0, 1), same convention
    as Surface.cross_section) -- the "zeta = const" cut through the
    surface plotted by plot_cross_section_comparison (shape_match.py)
    and plot_cross_sections (SurfaceBSpline itself).

    Uses periodic (wraparound) central finite differences in theta
    rather than np.gradient's default one-sided edges: a cross section
    is a closed loop, and the corner we're trying to sharpen could
    easily sit right at the theta=0 seam (e.g. a z_sym cross section's
    pinned theta=0 point) -- a one-sided edge stencil there would be
    both less accurate and asymmetric between the two neighbors of the
    seam. Curvature is invariant to how the parameter is scaled (an
    affine reparametrization theta -> c*theta scales d/dtheta by 1/c
    and d^2/dtheta^2 by 1/c^2, which cancel exactly in the ratio
    below), so it doesn't matter that dtheta here is a fraction-of-1
    rather than radians.

    Returns kappa, an (ntheta,) array.
    """
    pts = spline_surf.cross_section(phi, thetas=ntheta)
    R = np.hypot(pts[:, 0], pts[:, 1])
    Z = pts[:, 2]
    dtheta = 1.0 / ntheta
    dR = (np.roll(R, -1) - np.roll(R, 1)) / (2 * dtheta)
    dZ = (np.roll(Z, -1) - np.roll(Z, 1)) / (2 * dtheta)
    d2R = (np.roll(R, -1) - 2 * R + np.roll(R, 1)) / dtheta**2
    d2Z = (np.roll(Z, -1) - 2 * Z + np.roll(Z, 1)) / dtheta**2
    return np.abs(dR * d2Z - dZ * d2R) / (dR**2 + dZ**2) ** 1.5


def max_poloidal_curvature_residuals(spline_surf, phi_targets, ntheta=200):
    """
    One residual per phi in phi_targets: that cross section's own
    maximum poloidal curvature over theta. Paired below with a large,
    deliberately unreachable `goals` value (GOAL_CURVATURE) rather than
    0, this pushes each cross section's sharpest point to get sharper
    still, instead of trying to match some specific target curvature --
    there's no "corner sharpness" value known a priori, just a
    direction (sharper). Used directly for reporting (as here) and
    wrapped via make_optimizable below to become the
    LeastSquaresProblem's funcs_in.
    """
    return np.array(
        [
            poloidal_curvature(spline_surf, phi, ntheta=ntheta).max()
            for phi in phi_targets
        ]
    )


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


if __name__ == "__main__":
    mpi = MpiPartition()
    mpi.write()

    proc0_print("Running figures_spline_paper/corner_sharpening/sharpen.py")
    proc0_print("==================================================")

    dofs = [ 0.01746813,  0.33482434,  0.5469486 ,  0.20786794,  1.36767126,
        1.57079633,  0.02320235,  0.16363566,  0.57129204,  0.21985043,
        0.51934407,  0.39059644,  1.57079633,  1.64396992,  2.84748048,
        4.71238898,  5.07690209,  0.04660777,  0.06421164,  0.5410862 ,
        0.21416606,  0.50382936,  0.37801152,  1.57079633,  1.78925086,
        2.77310279,  4.71238898,  5.22479099,  0.07276264,  0.11254715,
        0.47182411,  0.20008525,  0.50596486,  0.32390405,  1.57079633,
        1.90369998,  2.64908177,  4.71238898,  5.49409733,  0.12960449,
        0.1431623 ,  0.44325262,  0.12011375,  0.46156554,  0.30050331,
        1.38834054,  1.94265805,  3.33298488,  4.70641149,  5.75958653,
        0.24616773,  0.19472327,  0.40038165,  0.09635272,  0.40566145,
        0.26765611,  1.38600546,  1.87665892,  3.66519143,  4.5835633 ,
        5.75958653,  0.28131815,  0.2153004 ,  0.41224895,  0.08976461,
        0.39274749,  0.1836771 ,  0.68095918,  1.77270439,  3.00613072,
        4.43734244,  5.26234657,  0.16048712,  0.28373422,  0.43769427,
        0.1299005 ,  0.43048199,  0.13408321,  0.52359878,  1.66956142,
        2.61799388,  4.30064794,  4.91861814,  0.05725048,  0.28790487,
        0.49691683,  0.16748246,  0.47878066,  0.09779584,  0.67706499,
        1.58017474,  3.05782456,  4.2857545 ,  4.71238898,  0.03822514,
        0.35578142,  0.51065609,  0.2115491 ,  0.53784385,  0.02825369,
        1.03176195,  1.57079633,  3.34630174,  4.40731892,  4.71238898,
        0.02452516,  0.40111207,  0.51805152,  0.22355059,  0.57785929,
        0.11499188,  1.16330892,  1.57079633,  3.47881853,  4.62413217,
        4.71238898,  0.03029292,  0.39322811,  0.52066474,  0.18724438,
        1.36384173,  1.57079633,  1.4566114 ,  1.44506858,  1.70929255,
       -0.06270618]

    spline_kwargs = {
        "axis_points": 3,
        "points_per_cs": 6,
        "n_cs": 12,
        "nfp": 1,
        "M": 24,
        "N": 24,
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
    proc0_print(f"spline_surf.dof_names: {spline_surf.dof_names}")
    spline_surf.x = dofs
    spline_surf.refine_poloidal()
    plot_cross_sections_aligned(spline_surf)
    spline_surf.plot()
    plt.show()

    # Snapshot before optimization, purely for the before/after
    # plot_cross_sections comparison at the end -- unaffected by
    # spline_surf.x changing later (each CrossSectionFixedZeta here is
    # the actual pre-optimization object, not a live view of it).
    old_cs_list = list(spline_surf.cs_list)

    # Target cross sections: the surface's own n_cs cross sections, at
    # their actual physical zeta (get_cs_zeta_angle returns radians in
    # [0, pi/nfp]; cross_section wants a fraction in [0, 1)).
    cs_zeta, _ = spline_surf.get_cs_zeta_angle()
    phi_targets = cs_zeta / (2 * np.pi)
    NTHETA = 200
    # Deliberately unreachable -- see max_poloidal_curvature_residuals'
    # docstring. Sharpening a corner is an open-ended "as sharp as
    # possible", not a match to a known target curvature.
    GOAL_CURVATURE = 1.0e3

    curvature_obj = make_optimizable(
        max_poloidal_curvature_residuals, spline_surf, phi_targets
    )

    # define problem
    prob = LeastSquaresProblem.from_tuples(
        [(curvature_obj.J, GOAL_CURVATURE, 1)]
    )

    proc0_print(
        "Initial max poloidal curvature per cross section:",
        max_poloidal_curvature_residuals(
            spline_surf, phi_targets, ntheta=NTHETA
        ),
    )

    proc0_print("Beginning optimization")
    proc0_print(f"ndofs: {len(prob.x)}")
    proc0_print(f"dofs names: {prob.dof_names}")

    least_squares_mpi_solve(
        prob,
        mpi,
        grad=True,
        rel_step=1e-12,
        abs_step=5e-6,
        x_scale="jac",
    )
    xopt = prob.x

    # evaluate the solution
    spline_surf.x = xopt

    if MPI.COMM_WORLD.rank == 0:
        spline_surf.plot_cross_sections(
            [old_cs_list, spline_surf.cs_list],
            labels=["before", "after"],
        )
        plot_cross_sections_aligned(spline_surf)
        spline_surf.plot()
        plt.show()
    proc0_print("")

    proc0_print(
        "Final max poloidal curvature per cross section:",
        max_poloidal_curvature_residuals(
            spline_surf, phi_targets, ntheta=NTHETA
        ),
    )
    proc0_print("spline_surf.x:", repr(xopt))

    proc0_print("")
    proc0_print("End of figures_spline_paper/corner_sharpening/sharpen.py")
    proc0_print("=================================================")
