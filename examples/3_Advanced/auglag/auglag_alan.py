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
from simsopt.geo import LinkingNumber
from simsopt.geo import CurveLength, CurveCurveDistance, \
    MeanSquaredCurvature, LpCurveCurvature, CurveSurfaceDistance
from simsopt.solve import augmented_lagrangian_method
from simsopt.field import BiotSavart
from simsopt.field.force import LpCurveForce, regularization_circ
from simsopt.field import Current, coils_via_symmetries
from pathlib import Path
import time

# Define the output directory   
OUT_DIR = "./output/"
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

# Define the target length, coil-to-coil distance, coil-to-surface distance, and curvature
LENGTH_TARGET = 17.4
CC_THRESHOLD = 0.1
CS_THRESHOLD = 0.3
CURVATURE_THRESHOLD = 5e4
MSC_THRESHOLD = 0

# Define the number of coils, rotation order, and non-planar base curves
R0 = s.x[0]
R1 = 0.6 * s.x[0]
order = 5
ncoils = 4
curves = create_equally_spaced_curves(
    ncoils, s.nfp, stellsym=s.stellsym, R0=R0, R1=R1, order=order, numquadpoints=128)
base_currents = [Current(1e5) for i in range(ncoils)]
# Since the target field is zero, one possible solution is just to set all
# currents to 0. To avoid the minimizer finding that solution, we fix one
# of the currents:
base_currents[0].fix_all()
base_curves = curves[:ncoils]
coils = coils_via_symmetries(base_curves, base_currents, s.nfp, s.stellsym)
base_coils = coils[:ncoils]
curves = [c.curve for c in coils]
currents = [c.current for c in coils]
print("Number of coils:", len(coils))

# Save the biot-savart field data
bs = BiotSavart(coils)
curves = [c.curve for c in coils]
curves_to_vtk(curves, OUT_DIR + "curves_init")
bs.set_points(s_plot.gamma().reshape((-1, 3))) 
pointData = {"B_N/|B|": np.sum(bs.B().reshape((qphi, qtheta, 3)) *
                               s_plot.unitnormal(), axis=2)[:, :, None] / bs.AbsB().reshape((qphi, qtheta, 1)),
             "modB": bs.AbsB().reshape((qphi, qtheta, 1))}
s_plot.to_vtk(OUT_DIR + "surf_init", extra_data=pointData)

# Define the individual terms objective function:
bs.set_points(s.gamma().reshape((-1, 3)))
Jf = SquaredFlux(s, bs, definition="normalized")
Jls = [CurveLength(c) for c in base_curves]
Jl = sum(QuadraticPenalty(jj, LENGTH_TARGET, "max") for jj in Jls)
Jccdist = CurveCurveDistance(curves, CC_THRESHOLD, num_basecurves=ncoils)
Jcsdist = CurveSurfaceDistance(curves, s, CS_THRESHOLD)
Jcs = [LpCurveCurvature(c, 2, CURVATURE_THRESHOLD) for c in base_curves]
Jmscs = [MeanSquaredCurvature(c) for c in base_curves]
Jlink = LinkingNumber(curves, downsample=2)
Jforce = [LpCurveForce(c, coils, regularization_circ(0.05), p=2.0, threshold=1e3) for c in base_coils]
outstr_dict = {'Jls': Jls, 'Jl': Jl, 'Jccdist': Jccdist, 
               'Jcsdist': Jcsdist, 'Jcs': Jcs, 'Jmscs': Jmscs, 'Jlink': Jlink,
               'Jf': Jf,
               'bs': bs,
               's': s,
               }

# Main optimization function
f = Jf  # Weight(0.0) * Jf

# Constraint list
# c_list = [Jl, Jcsdist, QuadraticPenalty(sum(Jls), LENGTH_TARGET, "max"), sum(Jcs)]
c_list = [Jcsdist, QuadraticPenalty(sum(Jls), LENGTH_TARGET, "max"), sum(Jcs), sum(Jforce)]

MINIMIZE_METHODS_NEW_CB = [
    'nelder-mead',
    'powell',
    'cg',
    'bfgs',
    'newton-cg',
    'l-bfgs-b',
    'trust-constr',
    'dogleg',
    'trust-ncg',
    'trust-exact',
    'trust-krylov']

start_time = time.time()
x, fnc, lag_mul = augmented_lagrangian_method(f, equality_constraints=c_list, mu_init=10, grad_tol=1e-10, c_tol=1e-10,
                                                MAXITER=200, argmin_tol=1e-16, minimize_method=MINIMIZE_METHODS_NEW_CB[5], 
                                                MAXITER_lag=10,
                                                lagrangian_form=None)

end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")
print('Final normalized flux:', Jf.J())
print('Final CS-Sep constraint:', Jcsdist.J())
print('Final CS-sep minimum distance:', Jcsdist.shortest_distance())
print('Final CC-Sep constraint:', Jccdist.J())
print('Final CC-sep minimum distance:', Jccdist.shortest_distance())
print('Final Len constraint:', Jl.J())
print('Final Curv constraint:', sum(Jcs).J())
print('Final Link constraint:', Jlink.J())
print('Final Max Curvatures:', [np.max(c.kappa()) for c in base_curves])
print('Final Lengths:', [CurveLength(c).J() for c in base_curves], sum(Jls))

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
print("Final NORMALIZED SQUARED FLUX:", Jf.J())
print('FINISHED OPTIMIZATION')