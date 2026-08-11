#!/usr/bin/env python
# Same stage-1/refine/stage-2 QS optimization as stage_one_splines_refine.py,
# but starting from an already-good warm-start point (the dofs pasted into
# spline_surface_plot.py) instead of a plain circular default -- this is
# meant to demonstrate that growing resolution via refine_poloidal() (exact
# Lane-Riesenfeld knot insertion, see SurfaceBSpline.refine_poloidal's
# docstring) doesn't perturb an already-converged shape, so optimization can
# continue smoothly at the higher resolution instead of restarting from
# scratch.

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


def report_bound_violations(prob, label):
    """Print any prob.x entries outside their prob.bounds, by dof name."""
    lb, ub = prob.bounds
    violations = [
        (name, val, l, u)
        for name, val, l, u in zip(prob.dof_names, prob.x, lb, ub)
        if not (l <= val <= u)
    ]
    if violations:
        proc0_print(f"{label}: {len(violations)} bound violation(s):")
        for name, val, l, u in violations:
            proc0_print(f"  {name} = {val!r} not in [{l!r}, {u!r}]")
    else:
        proc0_print(f"{label}: no bound violations")

proc0_print("Running 2_Intermediate/splines_warmstart_refined.py")
proc0_print("====================================================")

spline_kwargs = {
    "axis_points": 3,
    "points_per_cs": 4,
    "n_cs": 5,
    "nfp": 2,
    "M": 8,
    "N": 4,
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

spline_surf = SurfaceBSpline(**spline_kwargs, default_r=0.3)
spline_surf.axis.fix("r_axis_0")

# Warm-start point -- same dofs as pasted into spline_surface_plot.py.
warmstart_x = np.array(
    [
        7.5201273052623300e-06,
        4.2501631406549062e-01,
        1.3789545468136427e-01,
        1.3105015477699220e00,
        2.1246033886416580e-02,
        3.6328529493187850e-01,
        1.2818811782170014e-01,
        3.8943918345599199e-01,
        1.0694883206553649e00,
        3.2405748878384344e00,
        4.6539785945862961e00,
        1.1001511421235184e-01,
        2.5965822664705596e-01,
        1.2801936062875557e-01,
        3.1951585968736973e-01,
        7.9623244052944575e-01,
        3.5832690095641251e00,
        4.4201185993943746e00,
        1.6583920792183407e-01,
        1.6349040489652469e-01,
        2.0944161298573122e-01,
        2.0661100786519029e-01,
        7.8544529603976498e-01,
        3.4575321265156798e00,
        4.2632018765042359e00,
        1.8738734035077514e-01,
        1.1221864515467743e-01,
        2.4601657859914894e-01,
        1.5985464601264134e00,
        7.2302277317421226e-01,
        5.2938320909160241e-01,
        2.3054354773013225e-01,
        8.0640415675395638e-01,
    ]
)
spline_surf.x = warmstart_x

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
# dof count, so this same prob object stays valid after refine_poloidal()
# grows it below -- no need to reconstruct it for the second optimization.
prob = LeastSquaresProblem.from_tuples(
    [(qs.residuals, 0, 1), (vmec.aspect, 6, 10), (vmec.mean_iota, 0.42, 10)]
)

vmec.run()
proc0_print("Warm-start Quasisymmetry:", qs.total())
proc0_print("Warm-start aspect ratio:", vmec.aspect())
proc0_print("Warm-start rotational transform:", vmec.mean_iota())
if MPI.COMM_WORLD.rank == 0:
    spline_surf.plot()
    plt.show()

proc0_print("")
proc0_print("Beginning stage 1 optimization (at warm-start resolution)")
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
report_bound_violations(prob, "Stage 1")

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
    rel_step=1e-7,
    abs_step=1e-6,
    x_scale="jac",
)
xopt = prob.x
report_bound_violations(prob, "Stage 2")

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
proc0_print("End of 2_Intermediate/splines_warmstart_refined.py")
proc0_print("===================================================")
