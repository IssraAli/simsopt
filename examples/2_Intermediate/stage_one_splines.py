#!/usr/bin/env python


import booz_xform
import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI
from simsopt.geo import SurfaceBSpline
from simsopt.mhd import Boozer, QuasisymmetryRatioResidual, Vmec
from simsopt.objectives import LeastSquaresProblem
from simsopt.solve import least_squares_mpi_solve
from simsopt.util import MpiPartition, proc0_print

mpi = MpiPartition()
mpi.write()

proc0_print("Running 2_Intermediate/stage_one_splines.py")
proc0_print("==================================================")

spline_kwargs = {
    "axis_points": 3,
    "points_per_cs": 6,
    "n_cs": 6,
    "nfp": 2,
    "M": 12,
    "N": 12,
    "p_u": 3,
    "p_v": 3,
    "toroidal_equispaced": False,
    "poloidal_equispaced": False,
    "cs_global_angle_free": True,
    "axis_angles_fixed": True,
    "nurbs": False,
    "use_bishop_frame": True,
}

spline_surf = SurfaceBSpline(**spline_kwargs, default_r=0.2)
spline_surf.axis.fix("r_axis_0")

proc0_print(f"spline_surf.dof_names: {spline_surf.dof_names}")

vmec = Vmec.vmec_from_surf(
    nfp=spline_surf.nfp,
    surf=spline_surf,
    mpi=mpi,
    ns=13,
    M=12,
    N=12,
    ftol=1e-10,
)

# Configure quasisymmetry objective:
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
    [(qs.residuals, 0, 1), (vmec.aspect, 6, 1), (vmec.mean_iota, 0.42, 1)]
    # [(qs.residuals, 0, 1), (vmec.aspect, 8, 10), (vmec.mean_iota, -1.05, 10)]
)

vmec.run()
proc0_print("Initial Quasisymmetry:", qs.total())
proc0_print("Initial aspect ratio:", vmec.aspect())
proc0_print("Initial rotational transform:", vmec.mean_iota())

proc0_print("Beginning optimization")
proc0_print(f"ndofs: {len(prob.x)}")
proc0_print(f"dofs names: {prob.dof_names}")

# solver options
# options = {"disp": True, "ftol": 1e-7, "maxiter": 300}
# solve the problem
least_squares_mpi_solve(
    prob,
    mpi,
    grad=True,
    rel_step=1e-9,
    abs_step=5e-6,  # **options
    x_scale="jac",
)
xopt = prob.x

# Preserve the output file from the last iteration, so it is not
# deleted when vmec runs again:
vmec.files_to_delete = []


# evaluate the solution
spline_surf.x = xopt
vmec.run()

# Boozer transform on the surfaces to plot; each s snaps to the nearest
# VMEC half-grid surface (s = 1 gives the outermost half-grid surface)
boozer_s = [0.1, 0.25, 0.5, 1.0]
boozer = Boozer(vmec)
boozer.register(boozer_s)
boozer.run()

if MPI.COMM_WORLD.rank == 0:
    spline_surf.plot(show=False)

    # |B| in Boozer coordinates on each surface
    plt.figure(figsize=(11, 8))
    for i, s in enumerate(boozer_s):
        plt.subplot(2, 2, i + 1)
        booz_xform.surfplot(boozer.bx, js=boozer.s_to_index[s], fill=False)
        plt.title(f"|B| on s = {boozer.s_used[s]:.3g}")
    plt.tight_layout()

    # rotational transform profile
    plt.figure()
    plt.plot(vmec.s_half_grid, vmec.wout.iotas[1:], ".-")
    plt.xlabel("Normalized toroidal flux s")
    plt.ylabel("Rotational transform iota")
    plt.title("iota profile")
    plt.tight_layout()

    plt.show()
proc0_print("")

proc0_print(f"Final vmec iteration = {vmec.iter}")
proc0_print("Quasisymmetry:", qs.total())
proc0_print("aspect ratio:", vmec.aspect())
proc0_print("rotational transform:", vmec.mean_iota())
proc0_print("spline_surf.x:", repr(xopt))

proc0_print("")
proc0_print("End of 2_Intermediate/stage_one_splines.py")
proc0_print("=================================================")
