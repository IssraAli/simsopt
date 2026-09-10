#!/usr/bin/env python
"""
Stage-2 coil optimization against the "twin" crossover surfaces from
corner_sharpening/sharpen_fixed_s_twin.py (surf_outboard/surf_inboard),
standing in for the old SurfaceHelicalArc alpha=0/alpha=pi pair (see
x_point/ml_originals/20240305-01-stage_2_scan.py, which this is
adapted from).

The squared-flux objective is evaluated only on the interpolated
(u, phi) crossover-leg grid (compute_uphi_grid_interpolated), not each
twin's own full structured quadrature grid: a standard Surface's
quadpoints_phi x quadpoints_theta tensor grid can't represent this grid
(each phi row has its own, differently-sized theta window), so
SquaredFlux itself can't be used directly against it.
SquaredFluxOnGrid below reimplements SquaredFlux's "local" definition
on an explicit (N, 3) point list instead.

The flux objective is additionally weighted by distance from the
nearest corner (X-line), per Elder et al. 2024, "Stellarator divertor
design by optimizing coils for surfaces with sharp corners"
(arXiv:2510.27624v1), eq. 4-5:

    f_w = int w(x) B_n(x)^2 dA,   w(x) = (1 - |x - x0| / d_max)^p

where x0 is the nearest corner to x. Unlike that paper's closed-form
"rotating lemon" (whose corner locations are given analytically), our
spline twins' corners have to be located numerically -- see
squared_flux_weights, which uses sharpen_fixed_s_twin.py's
interpolated_corner_u to get each grid point's own row's corner
position exactly (not just the nearest of the n_cs discrete corners).
d_max is taken as the largest such distance actually present in our
grid, since that grid is already restricted to each corner's own
crossover-leg neighborhood (unlike the paper's whole-surface integral).

Runs a random hyperparameter sweep (N_JOBS runs) over coil order, R1,
and the length/curvature/coil-coil-distance penalty weights, one
subdirectory of results per run -- same pattern as ml_originals'
script, minus the surface-shape parameters (theta0, radius_d/a), which
don't apply to this fixed twin geometry.
"""

import json
import os
import sys

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "..", "corner_sharpening")
)

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from sharpen_fixed_s_twin import (
    DEMO_SPLINE_KWARGS,
    build_demo_surface,
    build_sharpened_twins,
    interpolated_corner_u,
    plot_leg_length_grid,
)
from simsopt._core.derivative import derivative_dec
from simsopt._core.optimizable import Optimizable
from simsopt.field import BiotSavart, Current, coils_via_symmetries
from simsopt.geo import (
    CurveCurveDistance,
    CurveLength,
    LinkingNumber,
    LpCurveCurvature,
    MeanSquaredCurvature,
    create_equally_spaced_curves,
    curves_to_vtk,
)
from simsopt.objectives import QuadraticPenalty

nfp = DEMO_SPLINE_KWARGS["nfp"]

# Coils
ncoils = 4
R0 = 1.0
R1 = 0.6
order = 5

# Crossover-leg sharpening (see build_sharpened_twins)
D_CRAWL = 0.05
D_EXT = 0.18
CORNER_CRITERION = "z"

# Squared-flux evaluation grid: N_PHI toroidal angles, NTHETA points per
# leg per angle. LEG_FRACTION is l_x (<= D_EXT).
N_PHI = 64
NTHETA = 64
LEG_FRACTION = 0.03

# Distance-based flux weighting (see module docstring, Elder et al.
# 2024 eq. 5) -- p=0 recovers unweighted squared flux; larger p
# concentrates the objective closer to the corner.
P_WEIGHT = 2.0

MAXITER = 500
SWEEP_DIR = "stage_2_scan_twin_out/"
N_JOBS = 5


class SquaredFluxOnGrid(Optimizable):
    """
    SquaredFlux's "local" definition (see
    simsopt.objectives.SquaredFlux), evaluated on an explicit (N, 3)
    positions/normals point list instead of a Surface's own structured
    quadrature grid, with an optional per-point multiplicative weight
    (see squared_flux_weights / module docstring):

        J = 0.5 * mean_i[ w_i * (B_i . n_hat_i)^2 / |B_i|^2 * |n_i| ]

    `normals` are the non-unit cross(gammadash1, gammadash2) vectors,
    matching Surface.normal()'s own convention (verified numerically
    against _gamma_and_derivs). `weights` defaults to all ones (plain
    squared flux). `field` is set to `positions` once, at
    construction. Depends only on `field`'s dofs, not any surface's --
    matching how SquaredFlux is used here, against a fixed target
    surface (`weights`, like `normals`, is fixed geometry, not a
    function of the field, so it just multiplies straight through the
    gradient below).
    """

    def __init__(self, positions, normals, field, weights=None):
        self.normals = normals
        self.weights = np.ones(len(positions)) if weights is None else weights
        self.field = field
        self.field.set_points(positions)
        super().__init__(x0=np.asarray([]), depends_on=[field])

    def J(self):
        absn = np.linalg.norm(self.normals, axis=1)
        unitn = self.normals / absn[:, None]
        B = self.field.B()
        B_n = np.sum(B * unitn, axis=1)
        modB2 = np.sum(B * B, axis=1)
        return 0.5 * np.mean(self.weights * B_n**2 / modB2 * absn)

    @derivative_dec
    def dJ(self):
        absn = np.linalg.norm(self.normals, axis=1)
        unitn = self.normals / absn[:, None]
        B = self.field.B()
        B_n = np.sum(B * unitn, axis=1)
        modB = np.linalg.norm(B, axis=1)
        dJdB = (
            (B_n / modB)[:, None]
            * (unitn / modB[:, None] - (B_n / modB**3)[:, None] * B)
            * (self.weights * absn)[:, None]
        ) / absn.shape[0]
        return self.field.B_vjp(dJdB)

    def mean_abs_bdotn(self):
        """mean(|B.n_hat|) over the grid -- the ⟨B·n⟩ diagnostic
        ml_originals' script prints/saves alongside Jf. Unweighted
        (this is a plain diagnostic, not the optimization objective)."""
        absn = np.linalg.norm(self.normals, axis=1)
        unitn = self.normals / absn[:, None]
        B_n = np.sum(self.field.B() * unitn, axis=1)
        return float(np.mean(np.abs(B_n)))


def positions_and_normals_on_grid(surf, grid):
    """(N, 3) positions and (N, 3) normals at the flattened (phi,
    theta) pairs of a compute_uphi_grid_interpolated grid dict --
    normals = cross(gammadash1, gammadash2), matching
    Surface.normal()'s own convention."""
    phi_flat = grid["phi"].reshape(-1)
    theta_flat = grid["theta"].reshape(-1)
    pos, g1, g2 = surf._gamma_and_derivs(phi_flat, theta_flat, max_deriv=1)
    return pos, np.cross(g1, g2, axis=1)


def leg_grid_to_vtk(filename, positions, grid, extra_data=None):
    """
    Export a crossover-leg grid's (N, 3) positions to VTK, mirroring
    Surface.to_vtk's own gridToVTK pattern -- needed because
    Surface.to_vtk always writes the surface's OWN quadpoints_phi/theta
    (the full closed loop), not this restricted (n_phis, ntheta)
    leg-only grid.

    `extra_data` (e.g. from bn_pointdata): a dict of (n_phis, ntheta)
    or (1, n_phis, ntheta) scalar arrays to attach as point data
    (colorable in Paraview) -- the normalized-flux analogue of
    ml_originals' script's own per-point "B_N" field-error convention.
    """
    from pyevtk.hl import gridToVTK

    n_phis, ntheta = grid["phi"].shape
    pos = positions.reshape(n_phis, ntheta, 3)
    x = pos[:, :, 0].reshape((1, n_phis, ntheta)).copy()
    y = pos[:, :, 1].reshape((1, n_phis, ntheta)).copy()
    z = pos[:, :, 2].reshape((1, n_phis, ntheta)).copy()
    point_data = {}
    if extra_data is not None:
        for key, val in extra_data.items():
            point_data[key] = np.ascontiguousarray(
                np.asarray(val).reshape((1, n_phis, ntheta))
            )
    gridToVTK(str(filename), x, y, z, pointData=point_data)


def bn_pointdata(field, normals):
    """{'Bn/B': (N,)} -- the NORMALIZED normal-field error
    (B.n_hat)/|B| at each grid point (the same normalization
    SquaredFluxOnGrid's own "local" definition uses, unlike
    ml_originals' script's raw B_N = B.n_hat). `field` must already
    have had set_points(...) called on it with this grid's positions
    (SquaredFluxOnGrid does this at construction)."""
    B = field.B()
    unitn = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    B_n = np.sum(B * unitn, axis=1)
    modB = np.linalg.norm(B, axis=1)
    return {"Bn/B": B_n / modB}


def squared_flux_weights(spline_surf, geometry, surf, grid, positions, p):
    """
    (N,) distance-based flux weights, per Elder et al. 2024 eq. 4-5
    (see module docstring): w = clip(1 - dist/d_max, 0, None)**p.

    `dist` is each grid point's Euclidean distance to the NEAREST of
    its own phi row's two corner apexes, each interpolated to that
    row's EXACT phi (interpolated_corner_u) rather than snapped to the
    nearest of the n_cs discrete corners -- since the grid includes
    phis strictly between those discrete cross sections, using the
    discrete corners directly would introduce spurious jumps in the
    weight as phi sweeps between them.

    `d_max` is the largest such distance actually present in `grid`
    (not a whole-surface normalization, unlike the paper's -- this
    grid is already restricted to each corner's own crossover-leg
    neighborhood by construction, so its own reach IS the relevant
    scale).

    `positions` must be the SAME (N, 3) array (reshape-compatible with
    grid['phi']) that positions_and_normals_on_grid already computed
    for this surf/grid -- passed in rather than recomputed here.
    """
    n_phis, ntheta = grid["phi"].shape
    phi_rows = grid["phi"][:, 0]
    u0_phi, u1_phi = interpolated_corner_u(spline_surf, geometry, phi_rows)
    corner0_pos = np.zeros((n_phis, 3))
    corner1_pos = np.zeros((n_phis, 3))
    surf.gamma_lin(corner0_pos, phi_rows, np.mod(u0_phi / (2 * np.pi), 1.0))
    surf.gamma_lin(corner1_pos, phi_rows, np.mod(u1_phi / (2 * np.pi), 1.0))

    pos = positions.reshape(n_phis, ntheta, 3)
    d0 = np.linalg.norm(pos - corner0_pos[:, None, :], axis=2)
    d1 = np.linalg.norm(pos - corner1_pos[:, None, :], axis=2)
    dist = np.minimum(d0, d1).reshape(-1)
    d_max = dist.max()
    return np.clip(1.0 - dist / d_max, 0.0, None) ** p


def build_flux_grids():
    """The (fixed, coil-independent) twin surfaces and their flux-grid
    positions/normals -- built once and reused across every sweep run,
    since none of it depends on the coil hyperparameters being scanned."""
    spline_surf = build_demo_surface()
    spline_surf, surf_outboard, surf_inboard, grids = build_sharpened_twins(
        spline_surf, DEMO_SPLINE_KWARGS, D_CRAWL, D_EXT, LEG_FRACTION, CORNER_CRITERION,
        n_phi=N_PHI, ntheta=NTHETA,
    )
    grid_outboard, grid_inboard = grids["outboard"], grids["inboard"]

    pos_outboard, normal_outboard = positions_and_normals_on_grid(
        surf_outboard, grid_outboard
    )
    pos_inboard, normal_inboard = positions_and_normals_on_grid(
        surf_inboard, grid_inboard
    )
    weights_outboard = squared_flux_weights(
        spline_surf, grids["geometry"], surf_outboard, grid_outboard, pos_outboard, P_WEIGHT
    )
    weights_inboard = squared_flux_weights(
        spline_surf, grids["geometry"], surf_inboard, grid_inboard, pos_inboard, P_WEIGHT
    )

    # Show the two twins on the exact same custom (u, phi) grids the
    # flux objective below is evaluated on, before any coil
    # optimization runs.
    fig, _ax = plot_leg_length_grid(
        spline_surf,
        grids["geometry"],
        surf_outboard,
        surf_inboard,
        grids["outboard_is_between"],
        LEG_FRACTION,
        grids["phis"],
    )
    fig.savefig(os.path.join(SWEEP_DIR, "flux_grids.png"))
    plt.show()

    return (
        surf_outboard,
        surf_inboard,
        grid_outboard,
        grid_inboard,
        pos_outboard,
        normal_outboard,
        weights_outboard,
        pos_inboard,
        normal_inboard,
        weights_inboard,
    )


def run_optimization(
    grid_outboard,
    grid_inboard,
    pos_outboard,
    normal_outboard,
    weights_outboard,
    pos_inboard,
    normal_inboard,
    weights_inboard,
    R1,
    order,
    length_target,
    length_weight,
    max_curvature_threshold,
    max_curvature_weight,
    msc_threshold,
    msc_weight,
    cc_threshold,
    cc_weight,
    index,
    n_jobs,
):
    directory = (
        f"order_{order}_R1_{R1:.2}_length_target_{length_target:.2}_weight_{length_weight:.2}"
        + f"_max_curvature_{max_curvature_threshold:.2}_weight_{max_curvature_weight:.2}"
        + f"_msc_{msc_threshold:.2}_weight_{msc_weight:.2}"
        + f"_cc_{cc_threshold:.2}_weight_{cc_weight:.2}"
    )

    print()
    print("***********************************************")
    print(f"Job {index} of {n_jobs}")
    print("Parameters:", directory)
    print("***********************************************")
    print()

    out_dir = os.path.join(SWEEP_DIR, directory) + "/"
    os.makedirs(out_dir, exist_ok=True)

    base_curves = create_equally_spaced_curves(
        ncoils,
        nfp,
        stellsym=True,
        R0=R0,
        R1=R1,
        order=order,
        numquadpoints=order * 16,
    )
    base_currents = [Current(1.0) for _ in range(ncoils)]
    base_currents[0].fix_all()  # avoid the all-zero-current solution
    coils = coils_via_symmetries(base_curves, base_currents, nfp, True)
    curves = [c.curve for c in coils]

    bs_outboard = BiotSavart(coils)
    bs_inboard = BiotSavart(coils)
    Jf_outboard = SquaredFluxOnGrid(
        pos_outboard, normal_outboard, bs_outboard, weights_outboard
    )
    Jf_inboard = SquaredFluxOnGrid(
        pos_inboard, normal_inboard, bs_inboard, weights_inboard
    )

    Jls = [CurveLength(c) for c in base_curves]
    Jccdist = CurveCurveDistance(curves, cc_threshold, num_basecurves=ncoils)
    Jcs = [LpCurveCurvature(c, 2, max_curvature_threshold) for c in base_curves]
    Jmscs = [MeanSquaredCurvature(c) for c in base_curves]

    JF = (
        Jf_outboard
        + Jf_inboard
        + length_weight * QuadraticPenalty(sum(Jls), length_target * ncoils)
        + cc_weight * Jccdist
        + max_curvature_weight * sum(Jcs)
        + msc_weight
        * sum(QuadraticPenalty(J, msc_threshold, "max") for J in Jmscs)
        + LinkingNumber(curves, 2)
    )

    iteration = 0

    def fun(dofs):
        nonlocal iteration
        JF.x = dofs
        J = JF.J()
        grad = JF.dJ()
        jf = Jf_outboard.J() + Jf_inboard.J()
        BdotN = (Jf_outboard.mean_abs_bdotn() + Jf_inboard.mean_abs_bdotn()) / 2
        outstr = f"{iteration:4}  J={J:.1e}, Jf={jf:.1e}, ⟨B·n⟩={BdotN:.1e}"
        cl_string = ", ".join(f"{J.J():.1f}" for J in Jls)
        kap_string = ", ".join(f"{np.max(c.kappa()):.1f}" for c in base_curves)
        msc_string = ", ".join(f"{J.J():.1f}" for J in Jmscs)
        outstr += (
            f", Len=sum([{cl_string}])={sum(J.J() for J in Jls):.1f}, "
            f"ϰ=[{kap_string}], ∫ϰ²/L=[{msc_string}]"
        )
        outstr += f", C-C-Sep={Jccdist.shortest_distance():.2f}"
        outstr += f", ║∇J║={np.linalg.norm(grad):.1e}"
        print(outstr)
        iteration += 1
        return J, grad

    # Pointwise field-error VTK output, same convention as
    # ml_originals' script's "B_N" point data -- but on our OWN
    # restricted crossover-leg grid (leg_grid_to_vtk), not the twins'
    # full closed loops, since that grid is what the optimization
    # actually sees.
    leg_grid_to_vtk(
        out_dir + "surf_outboard_init",
        pos_outboard,
        grid_outboard,
        extra_data=bn_pointdata(bs_outboard, normal_outboard),
    )
    leg_grid_to_vtk(
        out_dir + "surf_inboard_init",
        pos_inboard,
        grid_inboard,
        extra_data=bn_pointdata(bs_inboard, normal_inboard),
    )
    curves_to_vtk(curves, out_dir + "curves_init", close=True)

    res = minimize(
        fun,
        JF.x,
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": MAXITER, "maxcor": 300},
        tol=1e-15,
    )
    JF.x = res.x
    print(res.message)

    leg_grid_to_vtk(
        out_dir + "surf_outboard_opt",
        pos_outboard,
        grid_outboard,
        extra_data=bn_pointdata(bs_outboard, normal_outboard),
    )
    leg_grid_to_vtk(
        out_dir + "surf_inboard_opt",
        pos_inboard,
        grid_inboard,
        extra_data=bn_pointdata(bs_inboard, normal_inboard),
    )
    curves_to_vtk(curves, out_dir + "curves_opt", close=True)
    bs_outboard.save(out_dir + "biot_savart.json")

    BdotN = (Jf_outboard.mean_abs_bdotn() + Jf_inboard.mean_abs_bdotn()) / 2

    results = {
        "nfp": nfp,
        "ncoils": ncoils,
        "R0": R0,
        "R1": R1,
        "d_crawl": D_CRAWL,
        "d_ext": D_EXT,
        "leg_fraction": LEG_FRACTION,
        "n_phi": N_PHI,
        "ntheta": NTHETA,
        "p_weight": P_WEIGHT,
        "order": order,
        "length_target": length_target,
        "length_weight": length_weight,
        "max_curvature_threshold": max_curvature_threshold,
        "max_curvature_weight": max_curvature_weight,
        "msc_threshold": msc_threshold,
        "msc_weight": msc_weight,
        "JF": float(JF.J()),
        "Jf": float(Jf_outboard.J() + Jf_inboard.J()),
        "BdotN": BdotN,
        "lengths": [float(J.J()) for J in Jls],
        "length": float(sum(J.J() for J in Jls)),
        "max_curvatures": [float(np.max(c.kappa())) for c in base_curves],
        "max_max_curvature": float(max(np.max(c.kappa()) for c in base_curves)),
        "coil_coil_distance": Jccdist.shortest_distance(),
        "gradient_norm": float(np.linalg.norm(JF.dJ())),
        "linking_number": LinkingNumber(curves).J(),
        "directory": directory,
        "mean_squared_curvatures": [float(J.J()) for J in Jmscs],
        "max_mean_squared_curvature": float(max(J.J() for J in Jmscs)),
        "message": res.message,
        "success": bool(res.success),
        "iterations": int(res.nit),
        "function_evaluations": int(res.nfev),
        "coil_currents": [c.get_value() for c in base_currents],
    }
    with open(out_dir + "results.json", "w") as outfile:
        json.dump(results, outfile, indent=2)


def rand(lo, hi):
    return np.random.rand() * (hi - lo) + lo


if __name__ == "__main__":
    os.makedirs(SWEEP_DIR, exist_ok=True)
    (
        surf_outboard,
        surf_inboard,
        grid_outboard,
        grid_inboard,
        pos_outboard,
        normal_outboard,
        weights_outboard,
        pos_inboard,
        normal_inboard,
        weights_inboard,
    ) = build_flux_grids()
    surf_outboard.to_vtk(SWEEP_DIR + "surf_outboard")
    surf_inboard.to_vtk(SWEEP_DIR + "surf_inboard")
    flux_grids = (
        grid_outboard,
        grid_inboard,
        pos_outboard,
        normal_outboard,
        weights_outboard,
        pos_inboard,
        normal_inboard,
        weights_inboard,
    )

    for index in range(N_JOBS):
        run_optimization(
            *flux_grids,
            R1=rand(0.4, 0.9),
            order=int(round(rand(4, 16))),
            length_target=rand(3.2, 5),
            length_weight=10.0 ** rand(-1, 1),
            max_curvature_threshold=rand(5, 15),
            max_curvature_weight=10.0 ** rand(-7, -3),
            msc_threshold=rand(5, 15),
            msc_weight=10.0 ** rand(-7, -3),
            cc_threshold=rand(0.025, 0.12),
            cc_weight=10.0 ** rand(-1, 4),
            index=index,
            n_jobs=N_JOBS,
        )
