"""
auglag_alan.py
===============

This script performs coil optimization for stellarator devices using the Augmented Lagrangian Method (ALM). The optimization aims to design coil shapes that generate a target magnetic surface, subject to engineering and physics constraints. The script leverages the Simsopt library for geometry, field, and optimization routines.

Main Features:
--------------
- Reads a VMEC equilibrium file to define the target magnetic surface.
- Initializes a set of non-planar coils with configurable symmetry and Fourier order.
- Defines an objective function based on the squared normal magnetic field (squared flux) on the target surface.
- Adds constraints and penalties for engineering requirements such as coil length, coil-to-coil distance, coil-to-surface distance, and curvature.
- Implements the Augmented Lagrangian optimization loop, updating Lagrange multipliers and penalty parameters.
- Outputs VTK files for visualization of the surface and coil shapes at various stages.

Usage:
------
- Configure the optimization parameters and constraints in the script.
- Run the script directly to perform optimization using the Augmented Lagrangian or traditional method.
- Output files are saved in the './output/' directory for post-processing and visualization.

Dependencies:
-------------
- simsopt
- numpy
- scipy
- matplotlib

"""

import numpy as np
import os
from scipy.optimize import minimize
from simsopt.objectives import SquaredFlux
from simsopt.objectives import Weight
from simsopt.objectives import QuadraticPenalty

from simsopt.geo import SurfaceRZFourier
from simsopt.geo import curves_to_vtk, create_equally_spaced_curves
from simsopt.geo import LinkingNumber, LpCurveCurvature
from simsopt.geo import TotalCurveLengths
from simsopt.solve import augmented_lagrangian_method
from simsopt.field import BiotSavart
from simsopt.field import Current, coils_via_symmetries
from simsopt.solve import construct_equality_constraints
from simsopt.geo import CurveSurfaceMinimumDistance, CurveCurveMinimumDistance
from simsopt.geo import CurveLength
from pathlib import Path
import time

# Define the output directory   
OUT_DIR = "./auglag_inequality_constraints/"
os.makedirs(OUT_DIR, exist_ok=True)

# Define the test directory
TEST_DIR = Path(__file__).parent / '../' / '../' / '../' / 'tests/test_files'

# Define the filename
filename = TEST_DIR / 'input.LandremanPaul2021_QA_lowres'

# Define the number of phi and theta points
nphi = 32
ntheta = 32

# Define the surface
s = SurfaceRZFourier.from_vmec_input(
    filename,
    range="half period",
    nphi=nphi,
    ntheta=ntheta)

qphi = 4 * nphi
qtheta = 4 * ntheta
quadpoints_phi = np.linspace(0, 1, qphi)
quadpoints_theta = np.linspace(0, 1, qtheta)
s_plot = SurfaceRZFourier.from_vmec_input(
    filename,
    range="full torus",
    quadpoints_phi=quadpoints_phi,
    quadpoints_theta=quadpoints_theta)

ncoils = 4
# Define the target length, coil-to-coil distance, coil-to-surface distance, and curvature
LENGTH_TARGET = 17.4
CC_THRESHOLD = 0.1
CS_THRESHOLD = 0.3
CURVATURE_THRESHOLD = 5.0
MSC_THRESHOLD = 0
FLUX_THRESHOLD = 1e-7

# Define the number of coils, rotation order, and non-planar base curves
R0 = s.x[0]
R1 = 0.6 * s.x[0]
order = 5
curves = create_equally_spaced_curves(
    ncoils, s.nfp, stellsym=s.stellsym, R0=R0, R1=R1, order=order, numquadpoints=128)
base_currents = [Current(1.0) * 1e5 for i in range(ncoils)]
# Since the target field is zero, one possible solution is just to set all
# currents to 0. To avoid the minimizer finding that solution, we fix one
# of the currents:
base_currents[0].fix_all()
base_curves = curves[:ncoils]
coils = coils_via_symmetries(base_curves, base_currents, s.nfp, s.stellsym)
curves = [c.curve for c in coils]
currents = [c.current for c in coils]
print("Number of coils:", len(coils))

# Save the biot-savart field data
bs = BiotSavart(coils)
curves = [c.curve for c in coils]
curves_to_vtk(curves, OUT_DIR + "curves_init")
bs.set_points(s_plot.gamma().reshape((-1, 3))) 
pointData = {
             "B_N": np.sum(bs.B().reshape((qphi, qtheta, 3)) *
                               s_plot.unitnormal(), axis=2)[:, :, None],
             "B_N/|B|": np.sum(bs.B().reshape((qphi, qtheta, 3)) *
                               s_plot.unitnormal(), axis=2)[:, :, None] / bs.AbsB().reshape((qphi, qtheta, 1)),
             "modB": bs.AbsB().reshape((qphi, qtheta, 1))}
s_plot.to_vtk(OUT_DIR + "surf_init", extra_data=pointData)

# Define the individual terms objective function:
bs.set_points(s.gamma().reshape((-1, 3)))
coil_flux_constraint = SquaredFlux(s, bs, definition='normalized')
coil_surface_minimum_distance_constraint = CurveSurfaceMinimumDistance(base_curves, s, downsample=1)
coil_coil_minimum_distance_constraint = CurveCurveMinimumDistance(curves, downsample=1,
                                                                  minimum_distance=CC_THRESHOLD)
coil_total_length_constraint = TotalCurveLengths(base_curves)
Jcs = [LpCurveCurvature(c, p=10, threshold=CURVATURE_THRESHOLD) for c in base_curves]
coil_curvature_constraint = sum(Jcs) #MaxCurvature(base_curves)
coil_linking_number_constraint = LinkingNumber(curves, downsample=2)

# Setup the slack variables to convert the inequality constraints
# into equality constraints.
objs = [
        coil_flux_constraint,  # code assumes that the first constraint is a flux constraint
        coil_surface_minimum_distance_constraint, 
        coil_coil_minimum_distance_constraint, 
        coil_total_length_constraint, 
        coil_curvature_constraint, 
        coil_linking_number_constraint
]
types = ['upper', 'lower', 'lower', 'upper', 'upper', 'upper'] 

# Curvature and CC-sep do not need slack variable thresholds because 
# they are already handled by the LpCurveCurvature and CurveCurveDistance
# objectives. By default, the slack variables always have bounds s_i >= 0 
# and inequality constraints become e.g. g_i(x) + s_i = threshold, s_i >= 0,
# which is equivalent to g_i(x) <= threshold.
thresholds = [FLUX_THRESHOLD, CS_THRESHOLD, 0.0, LENGTH_TARGET, 0.0, 0.0]
inequality_constraints = construct_equality_constraints(objs, types, thresholds)

MAXITER = 1000
MAXITER_lag = 100

start_time = time.time()
x, fnc, lag_mul = augmented_lagrangian_method(
    inequality_constraints=inequality_constraints,
    # verbose=True,
    MAXITER=MAXITER, MAXITER_lag=MAXITER_lag,
    )
end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")
print('Final normalized flux:', coil_flux_constraint.J())
print('Final CS-Sep constraint:', coil_surface_minimum_distance_constraint.J())
print('Final CC-Sep constraint:', coil_coil_minimum_distance_constraint.J() + CC_THRESHOLD)
print('Final Len constraint:', coil_total_length_constraint.J())
print('Final Curv constraint:', coil_curvature_constraint.J())
print('Final Link constraint:', coil_linking_number_constraint.J())
print('Final Max Curvatures:', [np.max(c.kappa()) for c in base_curves])
print('Final Lengths:', [CurveLength(c).J() for c in base_curves])

curves_to_vtk(curves, OUT_DIR + "optimized_coils_auglag")
bs.set_points(s_plot.gamma().reshape((-1, 3)))
pointData = {"B_N": np.sum(bs.B().reshape((qphi, qtheta, 3)) *
                        s_plot.unitnormal(), axis=2)[:, :, None],
        "B_N/|B|": np.sum(bs.B().reshape((qphi, qtheta, 3)) *
                        s_plot.unitnormal(), axis=2)[:, :, None] /
        bs.AbsB().reshape((qphi, qtheta, 1)),
        "modB": bs.AbsB().reshape((qphi, qtheta, 1))}
s_plot.to_vtk(OUT_DIR + "surf_optimized_auglag", extra_data=pointData)
bs.set_points(s.gamma().reshape((-1, 3)))
print("--------------------------------------------------------------------------------------------------------------------------------------------")
print("FINAL LAGRANGE MULTIPLIERS:", lag_mul)
print("--------------------------------------------------------------------------------------------------------------------------------------------")
print("Final NORMALIZED SQUARED FLUX:", coil_flux_constraint.J())
print('FINISHED OPTIMIZATION')