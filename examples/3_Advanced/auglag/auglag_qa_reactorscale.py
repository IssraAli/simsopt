#!/usr/bin/env python
"""
auglag_qa_reactorscale.py
===============
This script performs coil optimization for the reactor-scale Landreman-Paul 2021 precise QA configuration 
using the Augmented Lagrangian Method (ALM).

In order to reproduce the coilsets of the paper the following thresholds/parameters should be set:
1) 3 coils solution
    ncoils : 3
    LENGTH_TARGET = 182.2
    FLUX_THRESHOLD = 1e-15
    CC_THRESHOLD = 1.0
    CS_THRESHOLD = 1.5
    CURVATURE_THRESHOLD = 0.5
    MSC_THRESHOLD = 0.05
    FORCE_THRESHOLD = 1.1e2

2) 4 coils solution
    ncoils : 4
    LENGTH_TARGET = 182.2
    FLUX_THRESHOLD = 1e-15
    CC_THRESHOLD = 1.0
    CS_THRESHOLD = 1.5
    CURVATURE_THRESHOLD = 0.5
    MSC_THRESHOLD = 0.05
    FORCE_THRESHOLD = 1.1e2
"""
import numpy as np
import os
from simsopt.objectives import SquaredFlux
from simsopt.objectives import QuadraticPenalty
from simsopt.geo import SurfaceRZFourier
from simsopt.geo import LinkingNumber
from simsopt.geo import CurveLength, CurveCurveDistance, \
    LpCurveCurvature, CurveSurfaceDistance, MeanSquaredCurvature
from simsopt.solve import augmented_lagrangian_method
from simsopt.field import BiotSavart, coils_to_vtk
from simsopt.field.force import LpCurveForce, coil_force
from simsopt.field import regularization_circ
from pathlib import Path
from simsopt.util import calculate_modB_on_major_radius, initialize_coils_simple
from simsopt.util import in_github_actions
import time

# Define the test directory
TEST_DIR = Path(__file__).parent / '../' / '../' / '../' / 'tests/test_files'

# Define the filename
filename = TEST_DIR / 'input.LandremanPaul2021_QA_reactorscale_lowres'

# Set some parameters -- warning this is super low resolution!
if in_github_actions:
    nphi = 4
    ntheta = 4
    MAXITER = 10
    MAXITER_lag = 5
else:
    # Define the number of phi and theta points
    nphi = 32
    ntheta = 32
    MAXITER = 1000  # 1000 for high-resolution
    MAXITER_lag = 50  # 50 for high-resolution

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
LENGTH_TARGET = 182.2  # 182.2 m is smallest of the Wechsung et al. 2021 coils
FLUX_THRESHOLD = 1e-15
CC_THRESHOLD = 1.0
CS_THRESHOLD = 1.5
MSC_THRESHOLD = 0.05
CURVATURE_THRESHOLD = 0.5
FORCE_THRESHOLD = 1.1e2  # Stay within ~ 0.57 MN/m assuming 200 turns of coil

# Define the number of coils, rotation order, and non-planar base curves
ncoils = 3
a = 0.15  # radius of the coil
coils = initialize_coils_simple(s, ncoils=ncoils, regularization=regularization_circ(a))
base_coils = coils[:ncoils]
curves = [c.curve for c in coils]
base_curves = curves[:ncoils]
currents = [c.current for c in coils]
print("Number of coils:", len(coils))

# Define the output directory   
OUT_DIR = f"./output_paper/output_ncoils{ncoils}_lengthtarget{LENGTH_TARGET}_fluxthreshold{FLUX_THRESHOLD}_ccthreshold{CC_THRESHOLD}_csthreshold{CS_THRESHOLD}_mscthreshold{MSC_THRESHOLD}_curvaturethreshold{CURVATURE_THRESHOLD}_forcethreshold{FORCE_THRESHOLD}/"
os.makedirs(OUT_DIR, exist_ok=True)

# Save the biot-savart field data
bs = BiotSavart(coils)
curves = [c.curve for c in coils]
coils_to_vtk(coils, OUT_DIR + "coils_init_qa_reactorscale")
bs.set_points(s_plot.gamma().reshape((-1, 3))) 
calculate_modB_on_major_radius(bs, s_plot)
bs.set_points(s_plot.gamma().reshape((-1, 3))) 

pointData = {"B_N/|B|": np.sum(bs.B().reshape((qphi, qtheta, 3)) *
                               s_plot.unitnormal(), axis=2)[:, :, None] / bs.AbsB().reshape((qphi, qtheta, 1)),
             "modB": bs.AbsB().reshape((qphi, qtheta, 1))}
s_plot.to_vtk(OUT_DIR + "surf_init_qa_reactorscale", extra_data=pointData)


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

# For the 25m Long coils Pareto front, the parameters for optimization were:
#       tau: 2    
#       MAXITER: 1500
#       MAXITER_lag: 50
#       grad_tol: 1e-8
#       c_tol: 1e-8
# For the 30m Long coils Pareto front, the parameters for optimization were:
#       tau: 4, 5 and 6 (4 for 14 kN/m, 5 for 12, 11 and 9 kN/m, 6 for 9.5 kN/m)
#       MAXITER: 1500 and 2000 (2000 for 9.5 kN/m)
#       grad_tol: 1e-8
#       c_tol: 1e-8
## The pareto front for 30m long coils seemed to be much more sensitive to the tau parameter than the 25. 
# between 1500 and 2000 MAXITER doesn't actually affect the optimization. 

x, fnc, lag_mul = augmented_lagrangian_method(f=f,
    equality_constraints=c_list,
    tau=10, #4 for 14, 5 for 12, 5 for 11, 5 for 10, 6 for 9.5, 5 for 9
    MAXITER=MAXITER, #1500 for all results except for 9.5 and 9
    MAXITER_lag=MAXITER_lag,
    grad_tol=1e-8,
    c_tol=1e-8,
)

bs.save(OUT_DIR + "biot_savart_optimized_auglag_qa_reactorscale.json")
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
coils_to_vtk(coils, OUT_DIR + "coils_optimized_auglag_qa_reactorscale")
bs.set_points(s_plot.gamma().reshape((-1, 3)))
calculate_modB_on_major_radius(bs, s_plot)
bs.set_points(s_plot.gamma().reshape((-1, 3))) 

pointData = {"B_N": np.sum(bs.B().reshape((qphi, qtheta, 3)) *
                        s_plot.unitnormal(), axis=2)[:, :, None],
        "B_N/|B|": np.sum(bs.B().reshape((qphi, qtheta, 3)) *
                        s_plot.unitnormal(), axis=2)[:, :, None] /
        bs.AbsB().reshape((qphi, qtheta, 1)),
        "modB": bs.AbsB().reshape((qphi, qtheta, 1))}
s_plot.to_vtk(OUT_DIR + "surf_optimized_auglag_qa_reactorscale", extra_data=pointData)
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
print("Output directory:", OUT_DIR)
