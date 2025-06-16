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
from simsopt.objectives import SquaredFlux
from simsopt.objectives import QuadraticPenalty
from simsopt.geo import SurfaceRZFourier
from simsopt.geo import create_equally_spaced_curves
from simsopt.geo import LinkingNumber
from simsopt.geo import CurveLength, CurveCurveDistance, \
    LpCurveCurvature, CurveSurfaceDistance, MeanSquaredCurvature
from simsopt.solve import augmented_lagrangian_method
from simsopt.field import BiotSavart, coils_to_vtk
from simsopt.field.force import LpCurveForce, regularization_circ, coil_force
from simsopt.field import Current, coils_via_symmetries
from pathlib import Path
from simsopt.util import calculate_modB_on_major_radius
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

# Define the upper and lower bounds for the constraints
LENGTH_TARGET = 25 # comically large length upper bound
# LENGTH_TARGET = 5
FLUX_THRESHOLD = 1e-15
CC_THRESHOLD = 0.083
CS_THRESHOLD = 0.15
MSC_THRESHOLD = 6 
CURVATURE_THRESHOLD = 12
FORCE_THRESHOLD = 0.0105  # units of MN/m

# Define the number of coils, rotation order, and non-planar base curves
R0 = s.x[0]
R1 = 0.7 * s.x[0]
order = 16
ncoils = 5
curves = create_equally_spaced_curves(
    ncoils, s.nfp, stellsym=s.stellsym, R0=R0, R1=R1, order=order, numquadpoints=128)

total_current = 3e5
# Since we know the total sum of currents, we only optimize for ncoils-1
# currents, and then pick the last one so that they all add up to the correct
# value.
base_currents = [Current(total_current / ncoils * 1e-5) * 1e5 for _ in range(ncoils-1)]
# Above, the factors of 1e-5 and 1e5 are included so the current
# degrees of freedom are O(1) rather than ~ MA.  The optimization
# algorithm may not perform well if the dofs are scaled badly.
total_current = Current(total_current)
total_current.fix_all()
base_currents += [total_current - sum(base_currents)]


base_curves = curves[:ncoils]
coils = coils_via_symmetries(base_curves, base_currents, s.nfp, s.stellsym)
base_coils = coils[:ncoils]
curves = [c.curve for c in coils]
currents = [c.current for c in coils]
print("Number of coils:", len(coils))

# Save the biot-savart field data
bs = BiotSavart(coils)
curves = [c.curve for c in coils]
coils_to_vtk(coils, OUT_DIR + "curves_init")
bs.set_points(s_plot.gamma().reshape((-1, 3))) 
calculate_modB_on_major_radius(bs, s_plot)
bs.set_points(s_plot.gamma().reshape((-1, 3))) 

pointData = {"B_N/|B|": np.sum(bs.B().reshape((qphi, qtheta, 3)) *
                               s_plot.unitnormal(), axis=2)[:, :, None] / bs.AbsB().reshape((qphi, qtheta, 1)),
             "modB": bs.AbsB().reshape((qphi, qtheta, 1))}
s_plot.to_vtk(OUT_DIR + "surf_init", extra_data=pointData)


# Define the individual terms objective function:
bs.set_points(s.gamma().reshape((-1, 3)))
Jf = SquaredFlux(s, bs, definition="normalized", threshold=FLUX_THRESHOLD) #definition="normalized"
Jls = [CurveLength(c) for c in base_curves]
Jl = sum(QuadraticPenalty(jj, LENGTH_TARGET, "max") for jj in Jls)

Jccdist = CurveCurveDistance(curves, CC_THRESHOLD, num_basecurves=ncoils)
Jcsdist = CurveSurfaceDistance(curves, s, CS_THRESHOLD)
Jcs = [LpCurveCurvature(c, 2, CURVATURE_THRESHOLD) for c in base_curves]
Jlink = LinkingNumber(curves, downsample=2)
Jforce = LpCurveForce(base_coils, coils, p=2.0, threshold=FORCE_THRESHOLD)
Jmscs = [MeanSquaredCurvature(c) for c in base_curves]

force = [np.max(np.linalg.norm(coil_force(c, coils), axis=1)) for c in base_coils]
print("Forces:")
print(",".join(f"{f:.2e}" for f in force)) 
print('Initial normalized flux:', Jf.J())
print('Initial CS-Sep constraint:', Jcsdist.J())
print('Initial CS-sep minimum distance:', Jcsdist.shortest_distance())
print('Initial CC-Sep constraint:', Jccdist.J())
print('Initial CC-sep minimum distance:', Jccdist.shortest_distance())
print('Initial Len constraint:', Jl.J())
print('Initial Curv constraint:', sum(Jcs).J())
print('Initial Link constraint:', Jlink.J())
print('Initial Max Curvatures:', [np.max(c.kappa()) for c in base_curves])
print('Initial Max MSC Curvatures:', [float(J.J()) for J in Jmscs])
print('Initial Lengths:', [CurveLength(c).J() for c in base_curves], sum(Jls).J())
print('Initial Force constraint:', Jforce.J())
# Main optimization function
f = None

# Constraint list
c_list = [ Jf,
        Jccdist, 
        Jcsdist, 
        #    Jl,
        sum(QuadraticPenalty(J, MSC_THRESHOLD, "max") for J in Jmscs),
        QuadraticPenalty(sum(Jls), LENGTH_TARGET, "max"), 
        sum(Jcs), 
        Jlink,
        Jforce
]

start_time = time.time()
x, fnc, lag_mul = augmented_lagrangian_method(f=f,
    equality_constraints=c_list,
    tau=2,
    MAXITER=1500,
    MAXITER_lag=50,
    grad_tol=1e-8,
    c_tol=1e-8,
)

bs.save(OUT_DIR + "biot_savart_same_setup.json")
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
print('Final Max MSC Curvatures:', [float(J.J()) for J in Jmscs])
print('Final Lengths:', [CurveLength(c).J() for c in base_curves], sum(Jls).J())
print('Final Force constraint:', Jforce.J())
force = [np.max(np.linalg.norm(coil_force(c, coils), axis=1)) for c in base_coils]
print("Forces:")
print(",".join(f"{f:.2e}" for f in force))
coils_to_vtk(coils, OUT_DIR + "optimized_coils_auglag")
bs.set_points(s_plot.gamma().reshape((-1, 3)))
calculate_modB_on_major_radius(bs, s_plot)
bs.set_points(s_plot.gamma().reshape((-1, 3))) 

pointData = {"B_N": np.sum(bs.B().reshape((qphi, qtheta, 3)) *
                        s_plot.unitnormal(), axis=2)[:, :, None],
        "B_N/|B|": np.sum(bs.B().reshape((qphi, qtheta, 3)) *
                        s_plot.unitnormal(), axis=2)[:, :, None] /
        bs.AbsB().reshape((qphi, qtheta, 1)),
        "modB": bs.AbsB().reshape((qphi, qtheta, 1))}
s_plot.to_vtk(OUT_DIR + "surf_optimized_auglag", extra_data=pointData)
max_BdotN_overB = np.max(np.sum(bs.B().reshape((qphi, qtheta, 3)) *
                        s_plot.unitnormal(), axis=2)[:, :, None] /
        bs.AbsB().reshape((qphi, qtheta, 1)))
bs.set_points(s.gamma().reshape((-1, 3)))
BdotN = np.mean(np.abs(np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)))
avg_BdotN_over_B = BdotN / bs.AbsB().mean()
print("--------------------------------------------------------------------------------------------------------------------------------------------")
print(f"<B_N>/<|B|> = {avg_BdotN_over_B:.2e}, Max BdotN/|B| = {max_BdotN_overB:.2e}")
print("FINAL LAGRANGE MULTIPLIERS:", lag_mul)
print("--------------------------------------------------------------------------------------------------------------------------------------------")
print("Final NORMALIZED SQUARED FLUX:", Jf.J())
print('FINISHED OPTIMIZATION')