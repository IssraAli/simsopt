r"""
config_verification.py
===============

This is a very basic script that gives the information on the optimized coilsets. 
For the QH and QA configurations it required some arbitrary rescaling to make bring the coilsets to 
ARIES-CS parameters, which made the script slightly trickier. 

Usage:
------
On the command line run:
python config_verification _your_configuration_

_your_configuration can be one of the following:
- QA
- QA_scaled
- QH
- stellaris
- w7x
- hsx

Dependencies:
-------------
- simsopt
- numpy


"""

import numpy as np
import os
from simsopt.field import load_coils_from_makegrid_file, coils_to_vtk
from simsopt.field import regularization_rect
import sys
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
from simsopt import load

def read_wechsung_coils(filename,s,ncoils, outdir):
    R0 = s.x[0]
    R1 = 0.7 * s.x[0]
    order = 16
    curves_wechsung = create_equally_spaced_curves(
        ncoils, s.nfp, stellsym=s.stellsym, R0=R0, R1=R1, order=order, numquadpoints=160)

    # Since we know the total sum of currents, we only optimize for ncoils-1
    # currents, and then pick the last one so that they all add up to the correct
    # value.
    base_currents_wechsung = [Current(1.) * 1e5 for _ in range(ncoils)]
    # Above, the factors of 1e-5 and 1e5 are included so the current
    # degrees of freedom are O(1) rather than ~ MA.  The optimization
    # algorithm may not perform well if the dofs are scaled badly.
    # base_currents_wechsung[0].fix_all()

    new_dofs = np.loadtxt(outdir + filename)
    
    base_curves_wechsung = curves_wechsung[:ncoils]
    coils_wechsung = coils_via_symmetries(base_curves_wechsung, base_currents_wechsung, s.nfp, s.stellsym)
    bs = BiotSavart(coils_wechsung)
    bs.x = np.concatenate((np.array([1]),new_dofs))
    coils_wechsung = bs.coils
    # coils_to_vtk(coils_wechsung, outdir+'curves_wechsung')
    return(coils_wechsung)

def scale_config(coils, current_factor, geometric_factor):
    from simsopt.field import Coil, apply_symmetries_to_curves, apply_symmetries_to_currents
    base_currents, base_curves, new_coils = [], [], []
    iter = 0
    for c in coils[:ncoils]:
        print('iter', iter)
        print('before', c.curve.x[0])
        c.curve.x = c.curve.x*geometric_factor
        print('after', c.curve.x[0])

        current = c.current.get_value()*current_factor
        base_currents.append(Current(current))
        iter+=1
    base_curves = [c.curve for c in coils[:ncoils]]
    currents = apply_symmetries_to_currents( base_currents, nfp=s.nfp, stellsym=True )
    curves = apply_symmetries_to_curves( base_curves, nfp=s.nfp, stellsym=True )
    new_coils = [ Coil( c, curr ) for ( c, curr ) in zip( curves, currents ) ]
    return(new_coils)

# Define the output directory   
OUT_DIR = "./output/"

os.makedirs(OUT_DIR, exist_ok=True)
# Define the test directory
TEST_DIR = Path(__file__).parent / '../' / '../' / '../' / 'tests/test_files'

if str(sys.argv[1]) == 'QH':
    filename = TEST_DIR / 'input.LandremanPaul2021_QH_reactorScale_lowres'
if str(sys.argv[1]) == 'w7x':
    filename = TEST_DIR /'input.W7-X_without_coil_ripple_beta0p05_d23p4_tm'
if str(sys.argv[1]) == 'stellaris':
    filename = '/home/gipe/auglag/simsopt/examples/3_Advanced/auglag/stellaris/input.stellaris'
if str(sys.argv[1]) == 'hsx':
    filename = TEST_DIR / 'input.QHS_mn1824_ns101'
if str(sys.argv[1]) == 'QA':
    filename = TEST_DIR / 'input.LandremanPaul2021_QA_lowres'
if str(sys.argv[1]) == 'QA_scaled':
    filename = TEST_DIR / 'input.LandremanPaul2021_QA_reactorScale_lowres'
    
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

print('MINOR RADIUS:', s.minor_radius())

LENGTH_TARGET = 5
FLUX_THRESHOLD = 1e-15
CC_THRESHOLD = 0.05
CS_THRESHOLD = 0.15
CURVATURE_THRESHOLD = 12.0
FORCE_THRESHOLD = 0.009  # units of MN/m

# Define the number of coils, rotation order, and non-planar base curves
R0 = s.x[0]
R1 = 0.6 * s.x[0]
if str(sys.argv[1]) == 'w7x':
    ncoils = 4
    from simsopt.configs.zoo import get_w7x_data
    from simsopt.field import BiotSavart
    curves, currents, _ = get_w7x_data(Nt_coils = 30, ppp=40)
    base_curves = curves[:ncoils]
    base_currents = currents[:ncoils]
    coils = coils_via_symmetries(base_curves, base_currents, nfp = 5, stellsym = True)
    bs = BiotSavart(coils)
    bs = load('/home/gipe/auglag/simsopt/examples/3_Advanced/auglag/output_paper/w7x_ncoils4_curvature2.5_force3.8_flux1e-15_length38_cc0.28_cs0.3/biot_savart_optimized.json')
    # bs = load('/home/gipe/auglag/simsopt/examples/3_Advanced/auglag/output_paper/w7x_ncoils5_curvature2_force3_flux1e-15_length45_cc0.28_cs0.3/biot_savart_optimized.json')
    # bs = load('/home/gipe/auglag/simsopt/examples/3_Advanced/auglag/output_paper/w7x_ncoils5_curvature2_force3.8_flux1e-15_length43_cc0.28_cs0.3/biot_savart_optimized.json') 
    coils = bs.coils
if str(sys.argv[1]) == 'hsx':
    ncoils = 4
    # To get the original HSX Coilset:

    # from simsopt.configs.zoo import get_hsx_data
    # from simsopt.field import BiotSavart
    # curves, currents, _ = get_hsx_data(Nt_coils = 16, ppp=40)
    # coils = coils_via_symmetries(curves, currents, nfp = 4, stellsym = True)
    # bs = BiotSavart(coils)

    # To get the optimized coilsets:

    # bs = load('/home/gipe/auglag/simsopt/examples/3_Advanced/auglag/output_paper/hsx_ncoils6_curvature8_force10_flux1e-15_length13.4_cc0.1_cs0.1/biot_savart_hsx.json')
    # bs = load('/home/gipe/auglag/simsopt/examples/3_Advanced/auglag/output_paper/hsx_ncoils5_curvature12_force0.12_flux1e-15_length12.083333333333332_cc0.1_cs0.14/biot_savart_hsx.json')
    bs = load('/home/gipe/auglag/simsopt/examples/3_Advanced/auglag/output_paper/hsx_ncoils4_curvature10_force10_flux1e-15_length10.0_cc0.1_cs0.14/biot_savart_hsx.json')
    coils = bs.coils
if str(sys.argv[1]) == 'stellaris':
    
    a = 0.32
    b = 0.32
    nturns_TF = 256
    FORCE_THRESHOLD *= nturns_TF
    coils_orig = load_coils_from_makegrid_file('./stellaris/coils.stellaris', order=30, ppp=40)
    print(len(coils_orig))
    print(coils_orig[0].curve)
    for c in coils_orig:
        c.regularization = regularization_rect(a, b)
    ncoils = 6
    print([c.current.get_value() for c in coils_orig])
    curves_TF = [c.curve for c in coils_orig]
    base_curves_TF = curves_TF[:ncoils]
    base_coils_TF = coils_orig[:ncoils]
    bs = BiotSavart(coils_orig)
    coils = bs.coils

if str(sys.argv[1]) == 'QA':
    bs = load('/home/gipe/auglag/simsopt/examples/3_Advanced/auglag/output_paper/LP_QA_total_length_30_force_threshold14kNm/biot_savart_same_setup.json')
    coils = bs.coils

if str(sys.argv[1]) == 'QH':
    wiedman = False
    # To get the original Wiedman Coilset:

    # from simsopt.field import load_coils_from_makegrid_file
    # from simsopt.field import BiotSavart
    # coilname = '/home/gipe/auglag/simsopt/examples/3_Advanced/auglag/output_paper/coils_wiedman/coils.curves_22_7_21'
    # coils = load_coils_from_makegrid_file(coilname, order=30, ppp=40)
    # from simsopt.geo import curves_to_vtk
    # curves_to_vtk([c.curve for c in coils], './output/wiedman_curves')
    # bs = BiotSavart(coils)
    # wiedman = True
    # OUT_DIR = '/home/gipe/auglag/simsopt/examples/3_Advanced/auglag/output_paper/coils_wiedman/'

    # To get the optimized coilsets:

    ncoils = 5
    # bs = load('/home/gipe/auglag/simsopt/examples/3_Advanced/auglag/output_paper/LP_QH_total_length_4coils_setup/biot_savart_qh.json')
    bs = load('/home/gipe/auglag/simsopt/examples/3_Advanced/auglag/output_paper/LP_QH_total_length_5coils_setup/biot_savart_qh.json')
    coils = bs.coils
    geometric_factor = 1
    if wiedman:
        current_factor = 0.858
    else:
        current_factor = 0.789 #0.571 for 4 coils and 0.789 for 5 coils
    coils = scale_config(coils, current_factor, geometric_factor)
    a = 0.3
    for c in coils:
        c.regularization = regularization_circ(a)
    bs = BiotSavart(coils)
    bs.save(OUT_DIR + "biot_savart_qh_scaled_aries.json")

if str(sys.argv[1]) == 'QA_scaled':
    wechsung = True
    if wechsung:
        OUT_DIR = './wechsung_coils/L24/'
        filename_wechsung = 'wechsung_L24.txt'
        ncoils_wechsung = 4
        ncoils = ncoils_wechsung
        coils_wechsung = read_wechsung_coils(filename_wechsung, s,ncoils_wechsung, OUT_DIR)
        bs = BiotSavart(coils_wechsung)
        # print(bs.dof_names)

        coils = coils_wechsung
        
    else:
        bs = load('/home/gipe/auglag/simsopt/examples/3_Advanced/auglag/output_paper/LP_QA_total_length_30_force_threshold14kNm/biot_savart_same_setup.json')
        OUT_DIR = '/home/gipe/auglag/simsopt/examples/3_Advanced/auglag/output_paper/LP_QA_total_length_force_threshold12kNm/'
        bs = load('/home/gipe/auglag/simsopt/examples/3_Advanced/auglag/output_paper/LP_QA_total_length_force_threshold12kNm/biot_savart_total_length.json')
        coils = bs.coils
        ncoils = 5
    geometric_factor = 10.12658382028
    if wechsung:
        current_factor = 164.7 #   158.8
    else:
        current_factor = 220 
    coils = scale_config(coils, current_factor, geometric_factor)
    a = 0.3
    for c in coils:
        c.regularization = regularization_circ(a)
    bs = BiotSavart(coils)
    bs.save(OUT_DIR + "biot_savart_qa_scaled_aries.json")

# Save the biot-savart field data
curves = [c.curve for c in coils]
from simsopt.geo import curves_to_vtk
curves_to_vtk(curves,OUT_DIR + 'lpf_scan_curves' )
base_curves = curves[:ncoils]
base_coils = coils[:ncoils]

# Save the biot-savart field data
bs.set_points(s_plot.gamma().reshape((-1, 3))) 
calculate_modB_on_major_radius(bs, s_plot, print_flag = True)
bs.set_points(s_plot.gamma().reshape((-1, 3))) 

pointData = {"B_N/|B|": np.sum(bs.B().reshape((qphi, qtheta, 3)) *
                               s_plot.unitnormal(), axis=2)[:, :, None] / bs.AbsB().reshape((qphi, qtheta, 1)),
             "modB": bs.AbsB().reshape((qphi, qtheta, 1))}
s_plot.to_vtk(OUT_DIR + "surf_init", extra_data=pointData)


# Define the individual terms objective function:
bs.set_points(s.gamma().reshape((-1, 3)))
Jf = SquaredFlux(s, bs, definition="normalized", threshold=FLUX_THRESHOLD)
Jls = [CurveLength(c) for c in base_curves]
Jl = sum(QuadraticPenalty(jj, LENGTH_TARGET, "max") for jj in Jls)

Jccdist = CurveCurveDistance(curves, CC_THRESHOLD, num_basecurves=ncoils)
Jcsdist = CurveSurfaceDistance(curves, s, CS_THRESHOLD)
Jcs = [LpCurveCurvature(c, 2, CURVATURE_THRESHOLD) for c in base_curves]
Jlink = LinkingNumber(curves, downsample=2)
Jforce = LpCurveForce(base_coils, coils, p=2.0, threshold=FORCE_THRESHOLD)
Jmscs = [MeanSquaredCurvature(c) for c in base_curves]

force = [np.max(np.linalg.norm(coil_force(c, coils), axis=1)) for c in base_coils]

print('Final normalized flux:', Jf.J())
print('Final CS-sep minimum distance:', Jcsdist.shortest_distance())
print('Final CC-sep minimum distance:', Jccdist.shortest_distance())
print('Final Max Curvatures:', [np.max(c.kappa()) for c in base_curves])
print('Final Max MSC Curvatures:', [float(J.J()) for J in Jmscs])
print('Final Lengths:', [CurveLength(c).J() for c in base_curves], sum(Jls).J())
# print('Final Force constraint:', Jforce.J())
force = [np.max(np.linalg.norm(coil_force(c, coils), axis=1)) for c in base_coils]
print("Forces:")
print(",".join(f"{f:.2e}" for f in force))
bs.set_points(s_plot.gamma().reshape((-1, 3)))
calculate_modB_on_major_radius(bs, s_plot)
bs.set_points(s_plot.gamma().reshape((-1, 3))) 

pointData = {"B_N": np.sum(bs.B().reshape((qphi, qtheta, 3)) *
                        s_plot.unitnormal(), axis=2)[:, :, None],
        "B_N/|B|": np.sum(bs.B().reshape((qphi, qtheta, 3)) *
                        s_plot.unitnormal(), axis=2)[:, :, None] /
        bs.AbsB().reshape((qphi, qtheta, 1)),
        "modB": bs.AbsB().reshape((qphi, qtheta, 1))}
max_BdotN_overB = np.max(np.sum(bs.B().reshape((qphi, qtheta, 3)) *
                        s_plot.unitnormal(), axis=2)[:, :, None] /
        bs.AbsB().reshape((qphi, qtheta, 1)))
s_plot.to_vtk(OUT_DIR + "surf_"+str(sys.argv[1]), extra_data=pointData)
bs.set_points(s.gamma().reshape((-1, 3)))
BdotN = np.mean(np.abs(np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)))
avg_BdotN_over_B = BdotN / bs.AbsB().mean()
coils_to_vtk(coils, OUT_DIR + 'coils_'+str(sys.argv[1]))
print("--------------------------------------------------------------------------------------------------------------------------------------------")
print(f"<B_N>/<|B|> = {avg_BdotN_over_B:.2e}, Max BdotN/|B| = {max_BdotN_overB:.2e}")
print("--------------------------------------------------------------------------------------------------------------------------------------------")
print("Final NORMALIZED SQUARED FLUX:", Jf.J())
