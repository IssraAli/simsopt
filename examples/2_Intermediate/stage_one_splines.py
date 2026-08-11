#!/usr/bin/env python


import numpy as np
from mpi4py import MPI
from simsopt.geo import SurfaceBSpline
from simsopt.mhd import QuasisymmetryRatioResidual, Vmec
from simsopt.objectives import LeastSquaresProblem
from simsopt.solve import least_squares_mpi_solve
from simsopt.util import MpiPartition, proc0_print

mpi = MpiPartition()
mpi.write()

proc0_print("Running 2_Intermediate/stage_one_splines.py")
proc0_print("==================================================")

spline_kwargs = {
    "axis_points": 3,
    "points_per_cs": 4,
    "n_cs": 5,
    "nfp": 2,
    "M": 5,
    "N": 4,
    "p_u": 3,
    "p_v": 3,
    "cs_equispaced": False,
    "rays_equispaced": False,
    "cs_global_angle_free": False,
    "axis_angles_fixed": False,
    "cs_basis": "polar",
    "nurbs": False,
    "use_bishop_frame": True,
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
    rel_step=1e-12,
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
if MPI.COMM_WORLD.rank == 0:
    spline_surf.plot()
proc0_print("")

proc0_print(f"Final vmec iteration = {vmec.iter}")
proc0_print("Quasisymmetry:", qs.total())
proc0_print("aspect ratio:", vmec.aspect())
proc0_print("rotational transform:", vmec.mean_iota())


proc0_print("")
proc0_print("End of 2_Intermediate/stage_one_splines.py")
proc0_print("=================================================")


# spline_kwargs = {
#     "axis_points": 3,
#     "points_per_cs": 5,
#     "n_cs": 5,
#     "nfp": 2,
#     "M": 8,
#     "N": 4,
#     "p_u": 3,
#     "p_v": 3,
#     "cs_equispaced": False,
#     "rays_equispaced": False,
#     "cs_global_angle_free": False,
#     "axis_angles_fixed": False,
#     "cs_basis": "polar",
#     "nurbs": False,
#     "use_bishop_frame": True,
# }


# Running 2_Intermediate/stage_one_splines.py
# ==================================================
# spline_surf.dof_names: ['CrossSectionFixedZeta1:r_0', 'CrossSectionFixedZeta1:r_1', 'CrossSectionFixedZeta1:r_2', 'CrossSectionFixedZeta1:theta_1', 'CrossSectionFixedZeta1:theta_2', 'CrossSectionFixedZeta2:r_0', 'CrossSectionFixedZeta2:r_1', 'CrossSectionFixedZeta2:r_2', 'CrossSectionFixedZeta2:r_3', 'CrossSectionFixedZeta2:r_4', 'CrossSectionFixedZeta2:theta_1', 'CrossSectionFixedZeta2:theta_2', 'CrossSectionFixedZeta2:theta_3', 'CrossSectionFixedZeta2:theta_4', 'CrossSectionFixedZeta3:r_0', 'CrossSectionFixedZeta3:r_1', 'CrossSectionFixedZeta3:r_2', 'CrossSectionFixedZeta3:r_3', 'CrossSectionFixedZeta3:r_4', 'CrossSectionFixedZeta3:theta_1', 'CrossSectionFixedZeta3:theta_2', 'CrossSectionFixedZeta3:theta_3', 'CrossSectionFixedZeta3:theta_4', 'CrossSectionFixedZeta4:r_0', 'CrossSectionFixedZeta4:r_1', 'CrossSectionFixedZeta4:r_2', 'CrossSectionFixedZeta4:r_3', 'CrossSectionFixedZeta4:r_4', 'CrossSectionFixedZeta4:theta_1', 'CrossSectionFixedZeta4:theta_2', 'CrossSectionFixedZeta4:theta_3', 'CrossSectionFixedZeta4:theta_4', 'CrossSectionFixedZeta5:r_0', 'CrossSectionFixedZeta5:r_1', 'CrossSectionFixedZeta5:r_2', 'CrossSectionFixedZeta5:theta_1', 'CrossSectionFixedZeta5:theta_2', 'PseudoAxis1:r_axis_0', 'PseudoAxis1:r_axis_1', 'PseudoAxis1:r_axis_2', 'PseudoAxis1:z_axis_1', 'PseudoAxis1:zeta_axis_1', 'SurfaceBSpline1:cs_zeta1', 'SurfaceBSpline1:cs_zeta2', 'SurfaceBSpline1:cs_zeta3']
# Initial Quasisymmetry: 0.0001940297678687007
# Initial aspect ratio: 5.812632015268553
# Initial rotational transform: 5.08677125506282e-19
# Beginning optimization
# ndofs: 45
# dofs names: ['CrossSectionFixedZeta1:r_0', 'CrossSectionFixedZeta1:r_1', 'CrossSectionFixedZeta1:r_2', 'CrossSectionFixedZeta1:theta_1', 'CrossSectionFixedZeta1:theta_2', 'CrossSectionFixedZeta2:r_0', 'CrossSectionFixedZeta2:r_1', 'CrossSectionFixedZeta2:r_2', 'CrossSectionFixedZeta2:r_3', 'CrossSectionFixedZeta2:r_4', 'CrossSectionFixedZeta2:theta_1', 'CrossSectionFixedZeta2:theta_2', 'CrossSectionFixedZeta2:theta_3', 'CrossSectionFixedZeta2:theta_4', 'CrossSectionFixedZeta3:r_0', 'CrossSectionFixedZeta3:r_1', 'CrossSectionFixedZeta3:r_2', 'CrossSectionFixedZeta3:r_3', 'CrossSectionFixedZeta3:r_4', 'CrossSectionFixedZeta3:theta_1', 'CrossSectionFixedZeta3:theta_2', 'CrossSectionFixedZeta3:theta_3', 'CrossSectionFixedZeta3:theta_4', 'CrossSectionFixedZeta4:r_0', 'CrossSectionFixedZeta4:r_1', 'CrossSectionFixedZeta4:r_2', 'CrossSectionFixedZeta4:r_3', 'CrossSectionFixedZeta4:r_4', 'CrossSectionFixedZeta4:theta_1', 'CrossSectionFixedZeta4:theta_2', 'CrossSectionFixedZeta4:theta_3', 'CrossSectionFixedZeta4:theta_4', 'CrossSectionFixedZeta5:r_0', 'CrossSectionFixedZeta5:r_1', 'CrossSectionFixedZeta5:r_2', 'CrossSectionFixedZeta5:theta_1', 'CrossSectionFixedZeta5:theta_2', 'PseudoAxis1:r_axis_0', 'PseudoAxis1:r_axis_1', 'PseudoAxis1:r_axis_2', 'PseudoAxis1:z_axis_1', 'PseudoAxis1:zeta_axis_1', 'SurfaceBSpline1:cs_zeta1', 'SurfaceBSpline1:cs_zeta2', 'SurfaceBSpline1:cs_zeta3']
#    Iteration     Total nfev        Cost      Cost reduction    Step norm     Optimality
#        0              1         1.0576e+00                                    8.17e+00
#        1              2         8.8208e-01      1.76e-01       4.70e-02       4.49e-02
#        2              4         8.8200e-01      8.91e-05       2.43e-02       2.06e-02
#        3              5         8.8096e-01      1.03e-03       4.78e-02       8.31e-02
#        4              6         8.6783e-01      1.31e-02       1.01e-01       4.37e-01
#        5              7         8.0075e-01      6.71e-02       1.46e-01       1.16e+00
#        6              8         5.7907e-01      2.22e-01       2.96e-01       2.02e+00
#        7              9         2.3059e-01      3.48e-01       4.54e-01       4.38e+00
#        8             10         6.9036e-02      1.62e-01       5.57e-01       1.80e+00
# WARNING:simsopt.objectives.least_squares:Function evaluation failed for <bound method QuasisymmetryRatioResidual.residuals of <simsopt.mhd.vmec_diagnostics.QuasisymmetryRatioResidual object at 0x30dc37610>>
#        9             12         3.0215e-02      3.88e-02       6.97e-02       1.02e-01
#       10             13         2.0217e-02      1.00e-02       1.06e-01       4.50e+00
#       11             14         1.6085e-02      4.13e-03       1.37e-01       1.85e-01
#       12             15         1.1487e-02      4.60e-03       1.19e-01       2.81e-01
#       13             16         7.4926e-03      3.99e-03       1.12e-01       9.36e-02
#       14             17         3.3126e-03      4.18e-03       2.22e-01       1.50e-01
#       15             18         1.7633e-03      1.55e-03       2.46e-01       3.49e-01
#       16             19         7.9525e-04      9.68e-04       8.54e-02       9.99e-02
#       17             21         5.4778e-04      2.47e-04       1.02e-01       4.87e-02
#       18             22         4.9357e-04      5.42e-05       1.71e-02       4.22e-02
#       19             23         2.9694e-04      1.97e-04       1.26e-01       3.19e-02
#       20             24         2.8690e-04      1.00e-05       6.30e-03       2.26e-02
#       21             25         2.0860e-04      7.83e-05       7.39e-02       2.16e-02
#       22             26         2.0074e-04      7.86e-06       8.05e-03       1.57e-02
#       23             27         1.7713e-04      2.36e-05       4.39e-02       1.39e-02
#       24             28         1.4838e-04      2.88e-05       9.57e-02       3.07e-02
#       25             29         1.4307e-04      5.31e-06       8.71e-04       6.97e-03
#       26             30         1.4265e-04      4.20e-07       1.47e-04       1.03e-02
#       27             31         1.4094e-04      1.71e-06       4.20e-03       1.17e-02
#       28             32         1.1896e-04      2.20e-05       6.82e-02       9.67e-03
#       29             33         1.1849e-04      4.77e-07       1.18e-04       6.27e-03
#       30             34         1.1819e-04      3.01e-07       1.28e-04       3.60e-03
#       31             35         1.1808e-04      1.04e-07       5.99e-05       4.30e-03
#       32             36         1.1793e-04      1.53e-07       5.95e-04       5.65e-03
#       33             38         1.1023e-04      7.70e-06       4.05e-02       2.35e-02
#       34             39         1.0992e-04      3.10e-07       2.61e-05       1.70e-02
#       35             40         1.0882e-04      1.10e-06       1.86e-04       3.71e-03
#       36             41         1.0815e-04      6.65e-07       6.79e-02       4.74e-02
#       37             42         9.7280e-05      1.09e-05       6.33e-04       1.68e-02
#       38             43         9.6110e-05      1.17e-06       1.67e-04       7.98e-03
#       39             44         9.2849e-05      3.26e-06       1.54e-02       1.66e-03
#       40             45         9.2818e-05      3.12e-08       8.98e-05       6.08e-04
#       41             46         8.9103e-05      3.72e-06       4.27e-02       1.59e-02
#       42             47         8.7966e-05      1.14e-06       2.09e-04       2.44e-03
#       43             48         8.3941e-05      4.03e-06       2.52e-02       1.91e-03
#       44             49         8.3923e-05      1.81e-08       5.74e-05       7.80e-04
#       45             50         8.3906e-05      1.65e-08       9.32e-05       1.29e-03
#       46             51         8.0989e-05      2.92e-06       2.28e-02       1.10e-03
#       47             52         8.0965e-05      2.38e-08       1.37e-04       6.20e-04
#       48             53         7.9147e-05      1.82e-06       5.05e-02       6.31e-03
#       49             54         7.8769e-05      3.78e-07       3.60e-04       7.15e-03
#       50             55         7.5055e-05      3.71e-06       2.78e-02       2.73e-03
#       51             56         7.4970e-05      8.47e-08       6.55e-05       4.01e-03
#       52             58         7.4438e-05      5.32e-07       1.19e-02       7.53e-04
#       53             59         7.4416e-05      2.22e-08       8.89e-05       2.16e-03
#       54             60         7.3772e-05      6.43e-07       1.76e-02       2.26e-03
#       55             61         7.3738e-05      3.44e-08       4.12e-05       5.35e-04
#       56             63         7.3633e-05      1.05e-07       1.86e-02       3.30e-03
#       57             64         7.3575e-05      5.74e-08       4.91e-05       4.96e-04
#       58             65         7.3467e-05      1.09e-07       3.13e-03       6.08e-04
#       59             66         7.3322e-05      1.45e-07       3.02e-03       7.27e-05
#       60             67         7.3236e-05      8.53e-08       7.84e-03       2.95e-04
#       61             68         7.3058e-05      1.79e-07       7.98e-03       3.74e-04
#       62             69         7.2912e-05      1.46e-07       7.19e-03       2.83e-04
#       63             71         7.2845e-05      6.64e-08       1.27e-03       3.15e-05
#       64             73         7.2815e-05      3.01e-08       5.71e-04       3.63e-05
#       65             75         7.2801e-05      1.40e-08       2.50e-04       3.50e-05
#       66             76         7.2774e-05      2.73e-08       4.09e-04       3.44e-05
#       67             79         7.2771e-05      3.39e-09       5.04e-05       3.46e-05
#       68             80         7.2764e-05      6.75e-09       1.01e-04       3.52e-05
#       69             83         7.2763e-05      8.42e-10       1.26e-05       1.16e-01
#       70             84         7.2761e-05      1.65e-09       3.25e-05       7.08e-05
#       71             85         7.2761e-05      7.83e-10       3.03e-05       3.45e-05
#       72             87         7.2760e-05      2.65e-10       7.30e-06       1.18e-01
#       73             88         7.2760e-05      3.75e-11       2.68e-05       5.46e-05
#       74             89         7.2760e-05      2.24e-10       1.02e-05       1.88e-02
#       75             92         7.2760e-05      0.00e+00       0.00e+00       1.88e-02
# `xtol` termination condition is satisfied.
# Function evaluations 92, initial cost 1.0576e+00, final cost 7.2760e-05, first-order optimality 1.88e-02.
# /Users/issraali/envs/simsopt_e/lib/python3.10/site-packages/mpl_toolkits/mplot3d/art3d.py:1403: RuntimeWarning: divide by zero encountered in matmul
#   shade = ((normals / np.linalg.norm(normals, axis=1, keepdims=True))
# /Users/issraali/envs/simsopt_e/lib/python3.10/site-packages/mpl_toolkits/mplot3d/art3d.py:1403: RuntimeWarning: overflow encountered in matmul
#   shade = ((normals / np.linalg.norm(normals, axis=1, keepdims=True))

# Final vmec iteration = 394
# Quasisymmetry: 0.0001454742473170254
# aspect ratio: 6.000005685115758
# rotational transform: 0.41993244084565823

# End of 2_Intermediate/stage_one_splines.py
# =================================================


# Running 2_Intermediate/stage_one_splines.py
# ==================================================
# spline_surf.dof_names: ['CrossSectionFixedZeta1:r_0', 'CrossSectionFixedZeta1:r_1', 'CrossSectionFixedZeta1:r_2', 'CrossSectionFixedZeta1:theta_1', 'CrossSectionFixedZeta2:r_0', 'CrossSectionFixedZeta2:r_1', 'CrossSectionFixedZeta2:r_2', 'CrossSectionFixedZeta2:r_3', 'CrossSectionFixedZeta2:theta_1', 'CrossSectionFixedZeta2:theta_2', 'CrossSectionFixedZeta2:theta_3', 'CrossSectionFixedZeta3:r_0', 'CrossSectionFixedZeta3:r_1', 'CrossSectionFixedZeta3:r_2', 'CrossSectionFixedZeta3:r_3', 'CrossSectionFixedZeta3:theta_1', 'CrossSectionFixedZeta3:theta_2', 'CrossSectionFixedZeta3:theta_3', 'CrossSectionFixedZeta4:r_0', 'CrossSectionFixedZeta4:r_1', 'CrossSectionFixedZeta4:r_2', 'CrossSectionFixedZeta4:r_3', 'CrossSectionFixedZeta4:theta_1', 'CrossSectionFixedZeta4:theta_2', 'CrossSectionFixedZeta4:theta_3', 'CrossSectionFixedZeta5:r_0', 'CrossSectionFixedZeta5:r_1', 'CrossSectionFixedZeta5:r_2', 'CrossSectionFixedZeta5:theta_1', 'PseudoAxis1:r_axis_1', 'PseudoAxis1:r_axis_2', 'PseudoAxis1:z_axis_1', 'PseudoAxis1:zeta_axis_1', 'SurfaceBSpline1:cs_zeta1', 'SurfaceBSpline1:cs_zeta2', 'SurfaceBSpline1:cs_zeta3']
# Initial Quasisymmetry: 0.00017188462825497205
# Initial aspect ratio: 6.776912874936196
# Initial rotational transform: 1.4133509666956468e-19
# Beginning optimization
# ndofs: 36
# dofs names: ['CrossSectionFixedZeta1:r_0', 'CrossSectionFixedZeta1:r_1', 'CrossSectionFixedZeta1:r_2', 'CrossSectionFixedZeta1:theta_1', 'CrossSectionFixedZeta2:r_0', 'CrossSectionFixedZeta2:r_1', 'CrossSectionFixedZeta2:r_2', 'CrossSectionFixedZeta2:r_3', 'CrossSectionFixedZeta2:theta_1', 'CrossSectionFixedZeta2:theta_2', 'CrossSectionFixedZeta2:theta_3', 'CrossSectionFixedZeta3:r_0', 'CrossSectionFixedZeta3:r_1', 'CrossSectionFixedZeta3:r_2', 'CrossSectionFixedZeta3:r_3', 'CrossSectionFixedZeta3:theta_1', 'CrossSectionFixedZeta3:theta_2', 'CrossSectionFixedZeta3:theta_3', 'CrossSectionFixedZeta4:r_0', 'CrossSectionFixedZeta4:r_1', 'CrossSectionFixedZeta4:r_2', 'CrossSectionFixedZeta4:r_3', 'CrossSectionFixedZeta4:theta_1', 'CrossSectionFixedZeta4:theta_2', 'CrossSectionFixedZeta4:theta_3', 'CrossSectionFixedZeta5:r_0', 'CrossSectionFixedZeta5:r_1', 'CrossSectionFixedZeta5:r_2', 'CrossSectionFixedZeta5:theta_1', 'PseudoAxis1:r_axis_1', 'PseudoAxis1:r_axis_2', 'PseudoAxis1:z_axis_1', 'PseudoAxis1:zeta_axis_1', 'SurfaceBSpline1:cs_zeta1', 'SurfaceBSpline1:cs_zeta2', 'SurfaceBSpline1:cs_zeta3']
#    Iteration     Total nfev        Cost      Cost reduction    Step norm     Optimality
#        0              1         3.9001e+00                                    1.84e+01
#        1              2         9.1821e-01      2.98e+00       1.18e-01       1.78e+00
#        2              3         8.8214e-01      3.61e-02       5.44e-02       6.65e-02
#        3              6         8.7791e-01      4.23e-03       8.94e-02       1.18e-01
#        4              7         8.1261e-01      6.53e-02       3.26e-01       3.42e-01
#        5              8         5.7322e-01      2.39e-01       3.67e-01       7.24e-01
#        6              9         1.7782e-01      3.95e-01       5.48e-01       1.23e+00
#        7             11         8.2505e-02      9.53e-02       1.19e-01       3.84e-01
#        8             12         3.4581e-02      4.79e-02       1.93e-01       5.89e-01
#        9             14         2.1001e-02      1.36e-02       1.27e-01       4.46e-01
#       10             16         1.5965e-02      5.04e-03       7.69e-02       1.01e-01
#       11             17         9.7372e-03      6.23e-03       1.63e-01       3.74e-01
#       12             19         6.4129e-03      3.32e-03       7.96e-02       9.76e-02
#       13             20         3.6861e-03      2.73e-03       1.78e-01       4.11e-01
#       14             21         3.5748e-03      1.11e-04       3.01e-01       8.84e-01
#       15             22         1.1437e-03      2.43e-03       5.72e-02       3.05e-02
#       16             23         9.2619e-04      2.18e-04       1.23e-01       1.20e-01
#       17             24         7.5617e-04      1.70e-04       9.89e-02       7.18e-02
#       18             25         7.4366e-04      1.25e-05       2.07e-04       1.38e-02
#       19             26         7.3914e-04      4.53e-06       5.08e-03       1.50e-02
#       20             28         6.9205e-04      4.71e-05       4.89e-02       1.66e-02
#       21             29         6.9104e-04      1.00e-06       1.42e-03       1.32e-02
#       22             30         6.3307e-04      5.80e-05       8.79e-02       8.02e-02
#       23             31         6.1892e-04      1.42e-05       2.68e-04       3.14e-02
#       24             33         5.8577e-04      3.31e-05       4.29e-02       2.61e-02
#       25             34         5.8439e-04      1.38e-06       7.36e-05       4.80e-03
#       26             35         5.4518e-04      3.92e-05       7.33e-02       7.03e-02
#       27             36         5.3502e-04      1.02e-05       2.02e-04       2.09e-01
#       28             37         5.2624e-04      8.77e-06       1.20e-02       1.17e-02
#       29             39         4.9923e-04      2.70e-05       3.48e-02       1.52e-02
#       30             40         4.9700e-04      2.23e-06       5.37e-03       7.27e-03
#       31             41         4.4787e-04      4.91e-05       9.14e-02       6.99e-02
#       32             42         4.3531e-04      1.26e-05       2.48e-04       8.43e-03
#       33             43         4.3285e-04      2.46e-06       3.86e-03       1.03e-02
#       34             45         4.0627e-04      2.66e-05       3.56e-02       1.65e-02
#       35             46         4.0409e-04      2.17e-06       4.54e-03       1.12e-02
#       36             47         3.7842e-04      2.57e-05       4.85e-02       2.46e-02
#       37             48         3.7678e-04      1.63e-06       9.66e-05       4.97e-03
#       38             49         3.6233e-04      1.45e-05       6.90e-02       7.38e-02
#       39             50         3.4941e-04      1.29e-05       2.51e-04       1.52e-02
#       40             51         3.2905e-04      2.04e-05       6.23e-02       6.62e-02
#       41             52         3.1931e-04      9.75e-06       1.99e-04       7.84e-03
#       42             54         3.0495e-04      1.44e-05       3.14e-02       1.98e-02
#       43             55         3.0409e-04      8.59e-07       5.84e-05       2.68e-03
#       44             56         2.9103e-04      1.31e-05       5.88e-02       7.28e-02
#       45             57         2.7792e-04      1.31e-05       2.35e-04       8.72e-03
#       46             58         2.6899e-04      8.93e-06       5.87e-02       7.95e-02
#       47             59         2.5557e-04      1.34e-05       2.37e-04       1.23e-02
#       48             61         2.4965e-04      5.93e-06       1.67e-02       6.86e-03
#       49             62         2.3733e-04      1.23e-05       3.49e-02       2.15e-02
#       50             63         2.3528e-04      2.05e-06       9.87e-05       1.72e-03
#       51             64         2.3508e-04      2.02e-07       5.01e-04       2.48e-03
#       52             65         2.1899e-04      1.61e-05       4.22e-02       2.55e-02
#       53             66         2.1717e-04      1.82e-06       8.33e-05       3.77e-03
#       54             67         2.1699e-04      1.75e-07       7.67e-05       3.48e-03
#       55             69         2.0731e-04      9.68e-06       2.22e-02       8.73e-03
#       56             70         2.0716e-04      1.51e-07       3.85e-04       9.16e-03
#       57             71         1.9555e-04      1.16e-05       5.86e-02       4.20e-02
#       58             72         1.8825e-04      7.30e-06       1.89e-04       2.77e-03
#       59             73         1.8792e-04      3.38e-07       1.25e-04       6.07e-03
#       60             74         1.7832e-04      9.59e-06       4.33e-02       3.78e-02
#       61             75         1.7636e-04      1.96e-06       9.55e-05       8.95e-03
#       62             77         1.6967e-04      6.69e-06       3.10e-02       2.16e-02
#       63             78         1.6929e-04      3.82e-07       4.59e-05       4.68e-03
#       64             79         1.6668e-04      2.61e-06       5.91e-02       8.59e-02
#       65             80         1.5997e-04      6.72e-06       1.97e-04       1.88e-02
#       66             81         1.5296e-04      7.01e-06       4.29e-02       5.59e-02
#       67             82         1.4993e-04      3.02e-06       1.36e-04       1.24e-02
#       68             84         1.4235e-04      7.58e-06       2.37e-02       1.53e-02
#       69             85         1.4206e-04      2.92e-07       3.94e-05       3.64e-03
#       70             86         1.3629e-04      5.77e-06       4.66e-02       7.22e-02
#       71             87         1.2927e-04      7.02e-06       1.98e-04       1.74e-02
#       72             88         1.2302e-04      6.25e-06       1.67e-02       1.84e-02
#       73             89         1.2263e-04      3.90e-07       6.52e-05       4.70e-03
#       74             90         1.1577e-04      6.87e-06       3.57e-02       4.21e-02
#       75             91         1.1212e-04      3.65e-06       1.39e-04       1.07e-02
#       76             92         1.1180e-04      3.19e-07       7.25e-04       8.33e-03
#       77             93         1.1098e-04      8.20e-07       3.41e-02       4.65e-02
#       78             94         1.0295e-04      8.03e-06       1.95e-04       1.39e-02
#       79             95         1.0248e-04      4.68e-07       1.36e-04       7.67e-03
#       80             96         1.0227e-04      2.04e-07       4.81e-04       8.87e-03
#       81             97         9.9574e-05      2.70e-06       9.67e-03       3.07e-03
#       82             98         9.9531e-05      4.30e-08       1.97e-04       3.05e-03
#       83             99         9.9493e-05      3.76e-08       1.41e-05       7.18e-04
#       84             100        9.9483e-05      9.92e-09       7.21e-05       8.79e-04
#       85             101        9.6743e-05      2.74e-06       4.46e-02       1.22e-02
#       86             102        9.6273e-05      4.70e-07       4.61e-05       2.50e-03
#       87             103        9.6240e-05      3.28e-08       2.56e-05       2.05e-03
#       88             104        9.6223e-05      1.66e-08       1.72e-05       1.12e-03
#       89             105        9.6210e-05      1.31e-08       6.66e-05       1.02e-03
#       90             106        9.5962e-05      2.48e-07       3.10e-02       2.11e-02
#       91             107        9.4789e-05      1.17e-06       7.01e-05       3.77e-03
#       92             108        9.4699e-05      8.98e-08       8.48e-05       4.01e-03
#       93             109        9.4674e-05      2.46e-08       1.36e-05       1.70e-03
#       94             110        9.4653e-05      2.15e-08       9.05e-05       6.23e-04
#       95             111        9.3834e-05      8.19e-07       9.49e-03       3.50e-04
#       96             112        9.3828e-05      6.01e-09       1.13e-04       3.87e-04
#       97             113        9.0969e-05      2.86e-06       1.60e-02       1.42e-03
#       98             114        9.0958e-05      1.11e-08       8.16e-05       1.30e-03
#       99             116        9.0442e-05      5.16e-07       1.06e-02       2.72e-03
#       100            117        9.0424e-05      1.77e-08       8.20e-06       3.12e-04
#       101            118        9.0420e-05      4.14e-09       3.46e-05       2.01e-04
#       102            120        9.0278e-05      1.42e-07       2.39e-03       7.46e-05
#       103            123        9.0261e-05      1.65e-08       4.98e-04       8.53e-05
#       104            127        9.0261e-05      4.90e-10       1.49e-05       8.57e-05
#       105            130        9.0261e-05      6.07e-11       1.83e-06       1.82e+00
#       106            132        9.0261e-05      0.00e+00       0.00e+00       1.82e+00
# `xtol` termination condition is satisfied.
# Function evaluations 132, initial cost 3.9001e+00, final cost 9.0261e-05, first-order optimality 1.82e+00.
# /Users/issraali/envs/simsopt_e/lib/python3.10/site-packages/mpl_toolkits/mplot3d/art3d.py:1403: RuntimeWarning: divide by zero encountered in matmul
#   shade = ((normals / np.linalg.norm(normals, axis=1, keepdims=True))
# /Users/issraali/envs/simsopt_e/lib/python3.10/site-packages/mpl_toolkits/mplot3d/art3d.py:1403: RuntimeWarning: overflow encountered in matmul
#   shade = ((normals / np.linalg.norm(normals, axis=1, keepdims=True))

# Final vmec iteration = 558
# Quasisymmetry: 0.00018048450748623903
# aspect ratio: 6.000005235178616
# rotational transform: 0.4199392268024837

# End of 2_Intermediate/stage_one_splines.py
# =================================================
