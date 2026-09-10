#!/usr/bin/env python
"""
Manifold-optimization refinement of the coils optimized by
stage_2_single_run.py, following Elder et al. 2024, "Stellarator
divertor design by optimizing coils for surfaces with sharp corners"
(arXiv:2510.27624v1), Sec. II.3 eq. 6:

    J_maniopt = sum_i min_S |x_i - S|

x_i is the image of a seed point after tracing forward through exactly
one field period; S ranges over the seed point's own twin surface
(surf_outboard/surf_inboard) at that same toroidal angle. Large
J_maniopt means the divertor-leg region isn't locally invariant (the
field is stochastic there); driving it down keeps the traced legs
locked to the twins' own shape, refining -- not re-deriving -- the
flux-optimized coils, matching the paper's own use of this objective
as a refinement of an already-optimized coil set (their Sec. II.3
notes that manifold optimization "does not naively work with cold-start
optimization" for exactly this reason).

Unlike stage_2_scan_twin.py's flux/length/curvature/msc/cc-distance
terms (which have analytic gradients via simsopt's Derivative/adjoint
machinery), field-line tracing has no analytic gradient available in
this codebase -- see TargetSurfaceDeviation in field_topology_
optimizables.py, which reuses SimpleIntegrator (a plain scipy
solve_ivp tracer) rather than pyoculus's own map/Newton machinery,
since eq. 6 needs only a single forward integration per seed point, not
a fixed point. So the WHOLE combined problem below (including the
terms that could otherwise use analytic gradients) is solved via
least_squares_mpi_solve's finite-difference Jacobian (grad=True) --
MPIFiniteDifference distributes the per-dof columns across MPI ranks,
so run this with `mpirun -n <nprocs>`.

Seed points are a union of outboard+inboard crossover-leg grid points
-- the same KIND of grid plot_leg_length_grid/build_flux_grids uses for
the flux objective (not the twins' full closed surfaces), but built at
its own N_PHI_MANIFOLD/NTHETA_MANIFOLD resolution (see below) via
build_manifold_seed_grid, independent of stage_2_scan_twin.py's own
N_PHI/NTHETA for the flux objective -- change N_PHI_MANIFOLD/
NTHETA_MANIFOLD to trade seed-grid density for finite-difference cost
without touching the flux objective's own grid at all.

Cost warning: at N_PHI_MANIFOLD=NTHETA_MANIFOLD=64 (~4096 points per
twin, ~8192 total) with ALL coil dofs free (~250, shape + current),
each finite-difference Jacobian costs ~8192*(ndofs+1) field-line
integrations. Lower N_PHI_MANIFOLD/NTHETA_MANIFOLD for a cheaper first
pass before committing to the full-resolution run.
"""

import os

import numpy as np
import simsopt
from field_topology_optimizables import TargetSurfaceDeviation

# stage_2_scan_twin's own import (above) already put corner_sharpening on
# sys.path -- see its own module docstring/imports.
from sharpen_fixed_s_twin import (
    DEMO_SPLINE_KWARGS,
    build_demo_surface,
    build_sharpened_twins,
)
from simsopt._core import make_optimizable
from simsopt.field import BiotSavart
from simsopt.geo import (
    CurveCurveDistance,
    CurveLength,
    LinkingNumber,
    LpCurveCurvature,
    MeanSquaredCurvature,
    curves_to_vtk,
)
from simsopt.objectives import LeastSquaresProblem, QuadraticPenalty
from simsopt.solve import least_squares_mpi_solve
from simsopt.util import MpiPartition, proc0_print
from stage_2_scan_twin import (
    CORNER_CRITERION,
    D_CRAWL,
    D_EXT,
    LEG_FRACTION,
    SWEEP_DIR,
    SquaredFluxOnGrid,
    bn_pointdata,
    build_flux_grids,
    leg_grid_to_vtk,
    ncoils,
    nfp,
    positions_and_normals_on_grid,
)

# The coils stage_2_single_run.py optimized -- see the previous
# conversation turn: order=10, R1=0.42, etc. reconstructed from the old
# poincare script's coils_filename directory name.
RUN_DIR = os.path.join(
    SWEEP_DIR,
    "order_10_R1_0.42_length_target_4.8_weight_0.21_max_curvature_7.1_weight_0.00022"
    "_msc_1.4e+01_weight_0.00044_cc_0.11_weight_9.7e+01",
)
COILS_FILENAME = os.path.join(RUN_DIR, "biot_savart.json")
OUT_DIR = os.path.join(RUN_DIR, "manifold_opt/")

# Same targets/thresholds stage_2_single_run.py used for this run (see
# results.json in RUN_DIR) -- reproduced here (rather than re-derived
# from base_curves/base_currents alone) so the length/curvature/msc/cc
# terms below stay anchored to the same goals during this refinement.
LENGTH_TARGET = 4.8
LENGTH_WEIGHT = 0.21
MAX_CURVATURE_THRESHOLD = 7.1
MAX_CURVATURE_WEIGHT = 0.00022
MSC_THRESHOLD = 14.0
MSC_WEIGHT = 0.00044
CC_THRESHOLD = 0.11
CC_WEIGHT = 97.0

# Manifold-objective seed grid -- independent of the flux objective's
# own N_PHI/NTHETA (stage_2_scan_twin.py's globals): change these to
# make the manifold seed grid coarser/finer without touching the flux
# objective's own grid at all. Defaults match the flux grid's own
# resolution (see cost warning in the module docstring).
N_PHI_MANIFOLD = 12
NTHETA_MANIFOLD = 12
NTHETA_TARGET = 400  # poloidal resolution of each seed point's own
# target cross section (nearest-point search)
MANIFOLD_WEIGHT = 10.0  # TODO: tune against the flux term's own scale
# (Jf ~ 5.8e-5 at the stage-2 optimum) before a
# full run -- this is a first guess, not a
# validated value.

MAXNFEV = 100


def load_base_curves_and_currents(coils_filename):
    """The `ncoils` unique (non-symmetrized) curves/currents from a
    biot_savart.json saved by stage_2_scan_twin.py's run_optimization
    -- coils_via_symmetries always places the originals first (indices
    0..ncoils-1) and their RotatedCurve/symmetric-current copies after,
    so this matches run_optimization's own base_curves/base_currents
    exactly (verified against the saved coil dof_size/fixed pattern)."""
    bs = simsopt.load(coils_filename)
    base_curves = [c.curve for c in bs.coils[:ncoils]]
    base_currents = [c.current for c in bs.coils[:ncoils]]
    curves = [c.curve for c in bs.coils]
    return bs, base_curves, base_currents, curves


def build_manifold_seed_grid(n_phi, ntheta):
    """A crossover-leg (phi, theta) grid at (n_phi, ntheta) resolution,
    independent of stage_2_scan_twin.py's own flux-objective grid
    (N_PHI/NTHETA) -- built the same way build_flux_grids() builds its
    own (same DEMO_SPLINE_KWARGS/D_CRAWL/D_EXT/LEG_FRACTION/
    CORNER_CRITERION), so the resulting twin-surface geometry is
    identical, just sampled at a different seed-point density.

    Returns (surf_outboard, surf_inboard, seed_xyz, target_surfs), where
    seed_xyz is (N, 3) (outboard points followed by inboard points) and
    target_surfs[i] is seed_xyz[i]'s own twin -- ready to pass straight
    to TargetSurfaceDeviation.
    """
    spline_surf = build_demo_surface()
    _, surf_outboard, surf_inboard, grids = build_sharpened_twins(
        spline_surf,
        DEMO_SPLINE_KWARGS,
        D_CRAWL,
        D_EXT,
        LEG_FRACTION,
        CORNER_CRITERION,
        n_phi=n_phi,
        ntheta=ntheta,
    )
    pos_outboard, _ = positions_and_normals_on_grid(
        surf_outboard, grids["outboard"]
    )
    pos_inboard, _ = positions_and_normals_on_grid(
        surf_inboard, grids["inboard"]
    )
    seed_xyz = np.concatenate([pos_outboard, pos_inboard], axis=0)
    target_surfs = [surf_outboard] * len(pos_outboard) + [surf_inboard] * len(
        pos_inboard
    )
    return surf_outboard, surf_inboard, seed_xyz, target_surfs


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    mpi = MpiPartition()

    proc0_print(f"Loading coils from {COILS_FILENAME}")
    _bs_loaded, base_curves, _, curves = load_base_curves_and_currents(
        COILS_FILENAME
    )

    proc0_print("Rebuilding twin surfaces and flux grids")
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

    # Fresh BiotSavart over the SAME curve/current objects loaded above,
    # so JF's gradient graph (for the flux/length/curvature/msc/cc terms)
    # is rooted at those live dofs, matching stage_2_scan_twin.py's own
    # run_optimization pattern.
    bs_outboard = BiotSavart(_bs_loaded.coils)
    bs_inboard = BiotSavart(_bs_loaded.coils)

    Jf_outboard = SquaredFluxOnGrid(
        pos_outboard, normal_outboard, bs_outboard, weights_outboard
    )
    Jf_inboard = SquaredFluxOnGrid(
        pos_inboard, normal_inboard, bs_inboard, weights_inboard
    )

    Jls = [CurveLength(c) for c in base_curves]
    Jccdist = CurveCurveDistance(curves, CC_THRESHOLD, num_basecurves=ncoils)
    Jcs = [LpCurveCurvature(c, 2, MAX_CURVATURE_THRESHOLD) for c in base_curves]
    Jmscs = [MeanSquaredCurvature(c) for c in base_curves]

    JF_stage2 = (
        Jf_outboard
        + Jf_inboard
        + LENGTH_WEIGHT * QuadraticPenalty(sum(Jls), LENGTH_TARGET * ncoils)
        + CC_WEIGHT * Jccdist
        + MAX_CURVATURE_WEIGHT * sum(Jcs)
        + MSC_WEIGHT
        * sum(QuadraticPenalty(J, MSC_THRESHOLD, "max") for J in Jmscs)
        + LinkingNumber(curves, 2)
    )

    def _stage2_residual_fn(JF):
        """sqrt(JF.J()) as a single least-squares residual -- JF is a
        sum of manifestly nonnegative penalty terms (SquaredFlux,
        QuadraticPenalty, CurveCurveDistance, LpCurveCurvature,
        LinkingNumber are all >= 0), so squaring this one residual
        reproduces the exact same scalar objective stage_2_single_run.py
        minimized via L-BFGS-B."""
        return np.sqrt(max(JF.J(), 0.0))

    # make_optimizable is needed here (unlike manifold_term.residuals
    # below, already a bound method of the Optimizable TargetSurfaceDeviation)
    # because LeastSquaresProblem.from_tuples requires each funcs_in
    # entry to be a bound method of an Optimizable (it reads fn.__self__
    # to register the dof-graph dependency) -- _stage2_residual_fn is a
    # plain closure, not a method, so it needs wrapping to participate.
    stage2_obj = make_optimizable(_stage2_residual_fn, JF_stage2)

    # Manifold seed points: built at N_PHI_MANIFOLD/NTHETA_MANIFOLD (see
    # module docstring/constants above), independent of the flux
    # objective's own grid built above.
    proc0_print(
        f"Building manifold seed grid (N_PHI={N_PHI_MANIFOLD}, NTHETA={NTHETA_MANIFOLD})"
    )
    _, _, seed_xyz, target_surfs = build_manifold_seed_grid(
        N_PHI_MANIFOLD, NTHETA_MANIFOLD
    )

    proc0_print(f"Manifold objective: {len(seed_xyz)} seed points")

    manifold_term = TargetSurfaceDeviation(
        _bs_loaded, seed_xyz, target_surfs, nfp, ntheta_target=NTHETA_TARGET
    )

    prob = LeastSquaresProblem.from_tuples(
        [
            (stage2_obj.J, 0, 1.0),
            (manifold_term.residuals, 0, MANIFOLD_WEIGHT),
        ]
    )

    proc0_print(f"Initial stage-2 objective JF={JF_stage2.J():.6e}")
    proc0_print(f"Initial manifold objective J_maniopt={manifold_term.J():.6e}")

    curves_to_vtk(curves, OUT_DIR + "curves_init", close=True)
    leg_grid_to_vtk(
        OUT_DIR + "surf_outboard_init",
        pos_outboard,
        grid_outboard,
        extra_data=bn_pointdata(bs_outboard, normal_outboard),
    )
    leg_grid_to_vtk(
        OUT_DIR + "surf_inboard_init",
        pos_inboard,
        grid_inboard,
        extra_data=bn_pointdata(bs_inboard, normal_inboard),
    )

    proc0_print("Beginning manifold-optimization refinement")
    least_squares_mpi_solve(
        prob,
        mpi,
        grad=True,
        abs_step=1e-6,
        max_nfev=MAXNFEV,
    )

    proc0_print(f"Final stage-2 objective JF={JF_stage2.J():.6e}")
    proc0_print(f"Final manifold objective J_maniopt={manifold_term.J():.6e}")

    if mpi.proc0_world:
        curves_to_vtk(curves, OUT_DIR + "curves_opt", close=True)
        leg_grid_to_vtk(
            OUT_DIR + "surf_outboard_opt",
            pos_outboard,
            grid_outboard,
            extra_data=bn_pointdata(bs_outboard, normal_outboard),
        )
        leg_grid_to_vtk(
            OUT_DIR + "surf_inboard_opt",
            pos_inboard,
            grid_inboard,
            extra_data=bn_pointdata(bs_inboard, normal_inboard),
        )
        _bs_loaded.save(OUT_DIR + "biot_savart.json")
