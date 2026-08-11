#!/usr/bin/env python
# Duplicate of stage_one_splines.py that runs the existing optimization,
# refines the spline's knots (growing its control-point resolution via
# SurfaceBSpline.refine -- see surfacespline.py), and then optimizes
# again with the extra resolution. This is the spline analog of the
# continuation strategy stage_one_fourier.py gets for free via
# max_mode -- growing resolution progressively without discarding a
# converged optimum, rather than freeing every dof from the start.


import numpy as np
from mpi4py import MPI
from simsopt.geo import SurfaceBSpline
from simsopt.mhd import QuasisymmetryRatioResidual, Vmec
from simsopt.objectives import LeastSquaresProblem
from simsopt.solve import least_squares_mpi_solve
from simsopt.util import MpiPartition, proc0_print
import matplotlib.pyplot as plt 

mpi = MpiPartition()
mpi.write()

proc0_print("Running 2_Intermediate/stage_one_splines_refine.py")
proc0_print("==================================================")

spline_kwargs = {
    "axis_points": 3,
    "points_per_cs": 4,
    "n_cs": 6,
    "nfp": 2,
    "M": 12,
    "N": 8,
    "p_u": 3,
    "p_v": 3,
    "cs_equispaced": True,
    "rays_equispaced": False,
    "cs_global_angle_free": False,
    "axis_angles_fixed": False,
    "cs_basis": "polar",
    "nurbs": False,
    "use_bishop_frame": True,
    "knot_parametrization": "uniform",

}

spline_surf = SurfaceBSpline(**spline_kwargs, default_r=0.2)
spline_surf.axis.fix("r_axis_0")

proc0_print(f"spline_surf.dof_names: {spline_surf.dof_names}")

vmec = Vmec.vmec_from_surf(
    nfp=spline_surf.nfp, surf=spline_surf, mpi=mpi, ns=13, M=12, N=12, ftol=1e-8
)

# Configure quasisymmetry objective:
qs = QuasisymmetryRatioResidual(
    vmec,
    np.arange(0, 1.01, 0.1),  # Radii to target
    helicity_m=1,
    helicity_n=0,  # -1
)  # (M, N) you want in |B|

# define problem -- built once; the Optimizable graph underneath (rooted
# at vmec, depending on spline_surf) dynamically reflects spline_surf's
# dof count, so this same prob object stays valid after refine() grows
# it below -- no need to reconstruct it for the second optimization.
prob = LeastSquaresProblem.from_tuples(
    [(qs.residuals, 0, 1), (vmec.aspect, 6, 10), (vmec.mean_iota, 0.42, 10)]
)

vmec.run()
proc0_print("Initial Quasisymmetry:", qs.total())
proc0_print("Initial aspect ratio:", vmec.aspect())
proc0_print("Initial rotational transform:", vmec.mean_iota())

proc0_print("")
proc0_print("Beginning stage 1 optimization")
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

# Preserve the output file from the last iteration, so it is not
# deleted when vmec runs again:
vmec.files_to_delete = []

# evaluate the stage-1 solution
spline_surf.x = xopt
vmec.run()
proc0_print("")
proc0_print(f"Stage 1 final vmec iteration = {vmec.iter}")
proc0_print("Stage 1 Quasisymmetry:", qs.total())
proc0_print("Stage 1 aspect ratio:", vmec.aspect())
proc0_print("Stage 1 rotational transform:", vmec.mean_iota())
if MPI.COMM_WORLD.rank == 0:
    spline_surf.plot()
    plt.show()
# Refine: grow the poloidal (per-cross-section) resolution via exact
# Lane-Riesenfeld/Boehm knot insertion (toroidal, i.e. cross-section-count,
# refinement isn't implemented -- see SurfaceBSpline.refine_poloidal's
# docstring), then re-evaluate at the new resolution before continuing
# to optimize.
proc0_print("")
proc0_print("Refining spline resolution (poloidal)")
spline_surf.refine_poloidal()
proc0_print(f"spline_surf.dof_names after refine: {spline_surf.dof_names}")

vmec.run()
proc0_print("")
proc0_print("Post-refine (pre-stage-2) Quasisymmetry:", qs.total())
proc0_print("Post-refine (pre-stage-2) aspect ratio:", vmec.aspect())
proc0_print("Post-refine (pre-stage-2) rotational transform:", vmec.mean_iota())

proc0_print("")
proc0_print("Beginning stage 2 optimization (post-refine)")
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

vmec.files_to_delete = []

# evaluate the final solution
spline_surf.x = xopt
vmec.run()
if MPI.COMM_WORLD.rank == 0:
    spline_surf.plot()
    plt.show()
proc0_print("")

proc0_print(f"Final vmec iteration = {vmec.iter}")
proc0_print("Quasisymmetry:", qs.total())
proc0_print("aspect ratio:", vmec.aspect())
proc0_print("rotational transform:", vmec.mean_iota())


proc0_print("")
proc0_print("End of 2_Intermediate/stage_one_splines_refine.py")
proc0_print("=================================================")
