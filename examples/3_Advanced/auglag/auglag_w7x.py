#!/usr/bin/env python
r"""
auglag_w7x.py
===============
This script performs coil optimization for the W7-X coil design 
using the Augmented Lagrangian Method (ALM).

In order to reproduce the coilsets of the paper the following thresholds/parameters should be set:
1) 4 coils solution (#1)
    ncoils_choice : 4
    LENGTH_TARGET = 38
    FLUX_THRESHOLD = 1e-15
    CC_THRESHOLD = 0.28
    CS_THRESHOLD = 0.3 
    CURVATURE_THRESHOLD = 2.5
    MSC_THRESHOLD = 1.5
    FORCE_THRESHOLD = 3.8 # units of MN/m

2) 5 coils solution (#2)
    ncoils_choice : 5
    LENGTH_TARGET = 45
    FLUX_THRESHOLD = 1e-15
    CC_THRESHOLD = 0.28
    CS_THRESHOLD = 0.3 
    CURVATURE_THRESHOLD = 2
    MSC_THRESHOLD = 1.5
    FORCE_THRESHOLD = 3.0 # units of MN/m

3) 5 coils solution (#3)
    ncoils_choice : 5
    LENGTH_TARGET = 43
    FLUX_THRESHOLD = 1e-15
    CC_THRESHOLD = 0.28
    CS_THRESHOLD = 0.3
    CURVATURE_THRESHOLD = 2
    MSC_THRESHOLD = 1.5
    FORCE_THRESHOLD = 3.5 # units of MN/m
"""

import os
from pathlib import Path
import time
import numpy as np
from simsopt.field import BiotSavart
from simsopt.field import coils_to_vtk
from simsopt.field import coils_via_symmetries, regularization_circ
from simsopt.solve import augmented_lagrangian_method
from simsopt.field.force import LpCurveForce, B2Energy
from simsopt.util import calculate_modB_on_major_radius
from simsopt.geo import (
    CurveLength, CurveCurveDistance, 
    LpCurveCurvature, CurveSurfaceDistance, LinkingNumber,
    SurfaceRZFourier, MeanSquaredCurvature
)
from simsopt.objectives import SquaredFlux, QuadraticPenalty
from simsopt.util import in_github_actions

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
    MAXITER = 50  # 1000 for high-resolution
    MAXITER_lag = 10  # 40 for high-resolution

# from simsopt.mhd import VirtualCasing

CC_THRESHOLD = 0.28 #initially with 1.0 
CS_THRESHOLD = 0.3
LENGTH_TARGET = 38 # initially with 138
FORCE_THRESHOLD = 3.8  # 3 for the 5 coil solution, 3.8 for the other solutions
FLUX_THRESHOLD = 1e-15 # initially with 1e-6
ncoils_choice = 4
CURVATURE_THRESHOLD = 2.5 #2 for the 5 coil solutions
MSC_THRESHOLD = 1.5 # for the 5 coil solutions

t1 = time.time()

# Directory for output
OUT_DIR = (f"./output_paper/w7x_ncoils{ncoils_choice}_curvature{CURVATURE_THRESHOLD}_" + \
           f"force{FORCE_THRESHOLD}_flux{FLUX_THRESHOLD}_length{LENGTH_TARGET}_" + \
           f"cc{CC_THRESHOLD}_cs{CS_THRESHOLD}/")
os.makedirs(OUT_DIR, exist_ok=True)

# File for the desired boundary magnetic surface:
TEST_DIR = Path(__file__).parent / '../' / '../' / '../' / 'tests/test_files'
input_name = 'input.W7-X_without_coil_ripple_beta0p05_d23p4_tm'
filename = TEST_DIR / input_name


# Initialize the boundary magnetic surface:
range_param = "half period"
s = SurfaceRZFourier.from_vmec_input(filename, range=range_param, nphi=nphi, ntheta=ntheta)

qphi = nphi * 2
qtheta = ntheta * 2
quadpoints_phi = np.linspace(0, 1, qphi, endpoint=True)
quadpoints_theta = np.linspace(0, 1, qtheta, endpoint=True)

# Make high resolution, full torus version of the plasma boundary for plotting
s_plot = SurfaceRZFourier.from_vmec_input(
    filename,
    quadpoints_phi=quadpoints_phi,
    quadpoints_theta=quadpoints_theta
)


from simsopt.configs.zoo import get_w7x_data
curves_orig, currents_orig, _ = get_w7x_data(Nt_coils = 30, ppp=40)
ncoils = 5
a = 0.15
regularizations = [regularization_circ(a) for _ in range(ncoils)]
coils_orig = coils_via_symmetries(curves_orig, currents_orig, nfp=5, stellsym=True, regularizations=regularizations)

print([c.current.get_value() for c in coils_orig])
curves_TF = [c.curve for c in coils_orig]
base_curves_TF = curves_TF[:ncoils]
base_coils_TF = coils_orig[:ncoils]
bs = BiotSavart(coils_orig)
bs.set_points(s.gamma().reshape((-1, 3)))
Bn = np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)
absB = np.linalg.norm(bs.B().reshape(nphi, ntheta, 3), axis=-1)
pointData = {"B_N": Bn[:, :, None],
             "B_N / B": (Bn / absB)[:, :, None],
            }
s.to_vtk(OUT_DIR + "s_original", extra_data=pointData)
Jf = SquaredFlux(s, bs, definition="normalized")
Jls = [CurveLength(c) for c in base_curves_TF]
Jl = sum(QuadraticPenalty(jj, LENGTH_TARGET, "max") for jj in Jls)
Jccdist = CurveCurveDistance(curves_TF, CC_THRESHOLD, num_basecurves=ncoils)
Jcsdist = CurveSurfaceDistance(curves_TF, s, CS_THRESHOLD)
Jcs = [LpCurveCurvature(c, 2, CURVATURE_THRESHOLD) for c in base_curves_TF]
Jlink = LinkingNumber(curves_TF, downsample=2)
Jmscs = [MeanSquaredCurvature(c) for c in base_curves_TF]
Jforce = LpCurveForce(base_coils_TF, coils_orig, p=2.0)
print('Initial normalized flux:', Jf.J())
print('Initial CS-sep minimum distance:', Jcsdist.shortest_distance())
print('Initial CC-sep minimum distance:', Jccdist.shortest_distance())
print('Initial Link constraint:', Jlink.J())
print('Initial Max Curvatures:', [np.max(c.kappa()) for c in base_curves_TF])
print('Initial Mean Squared Curvature', [MeanSquaredCurvature(c).J() for c in base_curves_TF])
print('Initial Lengths:', [CurveLength(c).J() for c in base_curves_TF], sum(Jls).J())
# print('Initial Force:', Jforce.J())

coils_to_vtk(coils_orig, OUT_DIR + "coils_original")
calculate_modB_on_major_radius(bs, s)
bs.set_points(s_plot.gamma().reshape((-1, 3)))
pointData = {"B_N": np.sum(bs.B().reshape((qphi, qtheta, 3)) * s_plot.unitnormal(), axis=2)[:, :, None],
             "B_N / B": (np.sum(bs.B().reshape((qphi, qtheta, 3)) * s_plot.unitnormal(), axis=2
                                ) / np.linalg.norm(bs.B().reshape(qphi, qtheta, 3), axis=-1))[:, :, None]}
s_plot.to_vtk(OUT_DIR + "surf_original", extra_data=pointData)


# initialize the TF coils
def w7x_coils(s, ncoils=3, order=8):
    from simsopt.geo import create_equally_spaced_curves
    from simsopt.field import Current

    # parameters for the TF coils, increase order for a better solution
    # Total current scaled to give B ~ 5.7 T on axis (actually averaged over the major radius)
    R0 = s.get_rc(0, 0) * 1
    R1 = s.get_rc(1, 0) * 3
    amperes = 15000.0
    # Non-planar coils have 108 turns. Planar coils have 36 turns.
    turns = 108
    total_current = amperes * turns * 5

    print('Total current = ', total_current)

    # Create the initial coils
    base_curves = create_equally_spaced_curves(
        ncoils, s.nfp, stellsym=True,
        R0=R0, R1=R1, order=order, numquadpoints=256,
    )
    base_currents = [(Current(total_current / ncoils * 1e-7) * 1e7) for _ in range(ncoils - 1)]
    total_current = Current(total_current)
    total_current.fix_all()
    base_currents += [total_current - sum(base_currents)]
    regularizations = [regularization_circ(a) for _ in range(ncoils)]
    coils = coils_via_symmetries(base_curves, base_currents, s.nfp, True, regularizations=regularizations)
    curves = [c.curve for c in coils]
    return base_curves, curves, coils, base_currents


ncoils = ncoils_choice
base_curves_TF, curves_TF, coils_TF, currents_TF = w7x_coils(s, ncoils=ncoils, order=16)
ncoils = len(base_curves_TF)
base_curves_TF = curves_TF[:ncoils]
base_coils_TF = coils_TF[:ncoils]

# # Calculate average, approximate on-axis B field strength
bs = BiotSavart(coils_TF)
btot = bs
calculate_modB_on_major_radius(btot, s)
btot.set_points(s.gamma().reshape((-1, 3)))

btot.set_points(s_plot.gamma().reshape((-1, 3)))
pointData = {"B_N": np.sum(btot.B().reshape((qphi, qtheta, 3)) * s_plot.unitnormal(), axis=2)[:, :, None],
             "B_N / B": (np.sum(btot.B().reshape((qphi, qtheta, 3)) * s_plot.unitnormal(), axis=2
                                ) / np.linalg.norm(btot.B().reshape(qphi, qtheta, 3), axis=-1))[:, :, None]}
s_plot.to_vtk(OUT_DIR + "surf_initial", extra_data=pointData)
btot.set_points(s.gamma().reshape((-1, 3)))

# Currently, all force terms involve all the coils
all_coils = coils_TF
all_base_coils = base_coils_TF

# Define the individual terms objective function:
bs.set_points(s.gamma().reshape((-1, 3)))
Jf = SquaredFlux(s, bs, definition="normalized", threshold=FLUX_THRESHOLD)
Jls = [CurveLength(c) for c in base_curves_TF]
Jl = sum(QuadraticPenalty(jj, LENGTH_TARGET, "max") for jj in Jls)
Jccdist = CurveCurveDistance(curves_TF, CC_THRESHOLD, num_basecurves=ncoils)
Jcsdist = CurveSurfaceDistance(curves_TF, s, CS_THRESHOLD)
Jcs = [LpCurveCurvature(c, 2, CURVATURE_THRESHOLD) for c in base_curves_TF]
Jlink = LinkingNumber(curves_TF, downsample=2)
Jforce = LpCurveForce(base_coils_TF, coils_TF, p=2.0, threshold=FORCE_THRESHOLD, downsample=2)
B2Energy_obj = B2Energy(coils_TF)
Jmscs = [MeanSquaredCurvature(c) for c in base_curves_TF]

# Main optimization function
# f = Weight(0.0) * Jf

# Constraint list
c_list = [Jf, 
          Jccdist, 
          Jcsdist, 
          QuadraticPenalty(sum(Jls), LENGTH_TARGET, "max"),
          sum(QuadraticPenalty(J, MSC_THRESHOLD, "max") for J in Jmscs),
          sum(Jcs), 
          Jlink,
          Jforce
]

start_time = time.time()
x, fnc, lag_mul = augmented_lagrangian_method(
    equality_constraints=c_list,
    tau=5,
    MAXITER=MAXITER,
    MAXITER_lag=MAXITER_lag
)
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
print('Final Max Curvatures:', [np.max(c.kappa()) for c in base_curves_TF])
print('Final Lengths:', [CurveLength(c).J() for c in base_curves_TF], sum(Jls).J())
print('Final Force constraint:', Jforce.J())
coils_to_vtk(coils_TF, OUT_DIR + "coils_optimized")

btot.set_points(s_plot.gamma().reshape((-1, 3)))
pointData = {"B_N": np.sum(btot.B().reshape((qphi, qtheta, 3)) * s_plot.unitnormal(), axis=2)[:, :, None],
             "B_N / B": (np.sum(btot.B().reshape((qphi, qtheta, 3)) * s_plot.unitnormal(), axis=2
                                ) / np.linalg.norm(btot.B().reshape(qphi, qtheta, 3), axis=-1))[:, :, None]}
s_plot.to_vtk(OUT_DIR + "surf_optimized", extra_data=pointData)

btot.set_points(s.gamma().reshape((-1, 3)))
calculate_modB_on_major_radius(btot, s)
btot.set_points(s_plot.gamma().reshape((-1, 3)))

t2 = time.time()
max_BdotN_overB = np.max((np.sum(btot.B().reshape((qphi, qtheta, 3)) * s_plot.unitnormal(), axis=2
                                ) / np.linalg.norm(btot.B().reshape(qphi, qtheta, 3), axis=-1))[:, :, None])
bs.set_points(s.gamma().reshape((-1, 3)))
BdotN = np.mean(np.abs(np.sum(btot.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)))
avg_BdotN_over_B = BdotN / btot.AbsB().mean()
print("--------------------------------------------------------------------------------------------------------------------------------------------")
print(f"<B_N>/<|B|> = {avg_BdotN_over_B:.2e}, Max BdotN/|B| = {max_BdotN_overB:.2e}")
print('Total time = ', t2 - t1)
btot.save(OUT_DIR + "biot_savart_optimized" + ".json")
print(OUT_DIR)
