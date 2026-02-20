#!/usr/bin/env python
"""
standard_qa_dynamic_runs.py
===============

This script performs the 20 dynamic resolution standard coil optimizations for the Pareto-front Figure 1 of the paper. 

The optimization aims to design coil shapes that generate a target magnetic surface, subject to engineering and physics constraints. The script leverages the Simsopt library for geometry, field, and optimization routines.

Main Features:
--------------
- Reads a VMEC equilibrium file to define the target magnetic surface.
- Initializes a set of non-planar coils with configurable symmetry and Fourier order.
- Defines an objective function based on the squared normal magnetic field (squared flux) on the target surface.
- Adds constraints and penalties for engineering requirements such as coil length, coil-to-coil distance, coil-to-surface distance, and curvature.
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

All thresholds except the Force threshold are kept fixed for all optimizations as they try to reproduce the work in 
A. A. Kaptanoglu, A. Wiedman, J. Halpern, S. Hurwitz, E. J. Paul, and M. Landreman, Reactor-scale stellarators
with force and torque minimized dipole coils, Nuclear Fusion 65, 046029 (2025).
    
The pareto front for 30m long coils seemed to be much more sensitive to the tau parameter than the 25. 

"""
from scipy.optimize import minimize
import numpy as np
import os
from simsopt.objectives import SquaredFlux
from simsopt.objectives import QuadraticPenalty
from simsopt.geo import SurfaceRZFourier
from simsopt.geo import LinkingNumber, create_equally_spaced_curves
from simsopt.geo import CurveLength, CurveCurveDistance, \
    LpCurveCurvature, CurveSurfaceDistance, MeanSquaredCurvature
from simsopt.field import BiotSavart, coils_to_vtk
from simsopt.field.force import LpCurveForce
from simsopt.field import regularization_circ, coils_via_symmetries
from simsopt.field import Current
from pathlib import Path
from simsopt.util import calculate_modB_on_major_radius
from simsopt.util import in_github_actions
import time

def fixed_range(curve, cmin=0, cmax=6, smin=1, smax=6, fix = True):
    fn = curve.fix if fix else curve.unfix
    check = curve.is_fixed if not fix else curve.is_free
    for kc in range(cmin, cmax + 1):
            if check(f'xc({kc})'):
                fn(f'xc({kc})')
            if check(f'yc({kc})'):
                fn(f'yc({kc})')
            if check(f'zc({kc})'):
                fn(f'zc({kc})')
    for ls in range(smin, smax + 1):
            if check(f'xs({ls})'):
                fn(f'xs({ls})')
            if check(f'ys({ls})'):
                fn(f'ys({ls})')
            if check(f'zs({ls})'):  
                fn(f'zs({ls})')

# Define the test directory
TEST_DIR = Path(__file__).parent / '../' / '../' / '../' / 'tests/test_files'

# Define the filename
filename = TEST_DIR / 'input.LandremanPaul2021_QA_lowres'

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
    MAXITER = 1500 #for high-resolution
    MAXITER_lag = 20 #for high-resolution

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

def rand(min, max):
    """Generate a random float between min and max."""
    return np.random.rand() * (max - min) + min

# Number of optimization runs
num_runs = 20

for i in range(num_runs):
    print("\n################################################################################")
    print(f"### Starting Optimization Run {i + 1}/{num_runs} ###")
    print("################################################################################\n")

    # Threshold and weight for the coil-to-coil distance penalty in the objective function:
    CC_THRESHOLD = 0.1
    CC_WEIGHT = 10.0 ** rand(-9, -1) #1000

    # Define the upper and lower bounds for the constraints
    LENGTH_TARGET = 30 # 25 for pareto front plots
    LENGTH_WEIGHT = 10.0 ** rand(-9, -1)

    FLUX_THRESHOLD = 1e-16

    CS_THRESHOLD = 0.15
    CS_WEIGHT = 10.0 ** rand(-9, -1)

    MSC_THRESHOLD = 5
    MSC_WEIGHT = 10.0 ** rand(-9, -1)

    CURVATURE_THRESHOLD = 5
    CURVATURE_WEIGHT = 10.0 ** rand(-9, -1)

    FORCE_THRESHOLD = 0.01
    FORCE_WEIGHT = 10.0 ** rand(-9, -1)

    LINKING_WEIGHT = 10.0 ** rand(-9, -1)

    ncoils = 5
    print('########### WEIGHTS ############')
    print(f"Run {i+1} weights:")
    print(f"CC_WEIGHT: {CC_WEIGHT:.2e}, CS_WEIGHT: {CS_WEIGHT:.2e}, LENGTH_WEIGHT: {LENGTH_WEIGHT:.2e}, "
          f"CURVATURE_WEIGHT: {CURVATURE_WEIGHT:.2e}, MSC_WEIGHT: {MSC_WEIGHT:.2e}, "
          f"LINKING_WEIGHT: {LINKING_WEIGHT:.2e}, FORCE_WEIGHT: {FORCE_WEIGHT:.2e}")
    print('##################################')
    
    # Define the output directory for the current run  
    OUT_DIR = (f"./output_paper/run_{i+1}_no_continuation/qa_standard_ncoils{ncoils}_curvature{CURVATURE_THRESHOLD}_msc{MSC_THRESHOLD}_" + \
               f"force{FORCE_THRESHOLD}_flux{FLUX_THRESHOLD}_length{LENGTH_TARGET}_" + \
               f"cc{CC_THRESHOLD}_cs{CS_THRESHOLD}/")
    os.makedirs(OUT_DIR, exist_ok=True)

    a = 0.05
    R0 = s.x[0]
    R1 = 0.7 * s.x[0]
    order = 19
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
    regularizations = [regularization_circ(a) for _ in range(ncoils)]
    coils = coils_via_symmetries(base_curves, base_currents, s.nfp, s.stellsym, regularizations=regularizations)
    base_coils = coils[:ncoils]
    curves = [c.curve for c in coils]
    base_curves = curves[:ncoils]
    currents = [c.current for c in coils]
    print("Number of coils:", len(coils))

    # Save the biot-savart field data
    bs = BiotSavart(coils)
    curves = [c.curve for c in coils]
    coils_to_vtk(coils, OUT_DIR + "coils_init_qa")
    bs.set_points(s_plot.gamma().reshape((-1, 3))) 
    calculate_modB_on_major_radius(bs, s_plot)
    bs.set_points(s_plot.gamma().reshape((-1, 3))) 

    pointData = {"B_N/|B|": np.sum(bs.B().reshape((qphi, qtheta, 3)) *
                                   s_plot.unitnormal(), axis=2)[:, :, None] / bs.AbsB().reshape((qphi, qtheta, 1)),
                 "modB": bs.AbsB().reshape((qphi, qtheta, 1))}
    s_plot.to_vtk(OUT_DIR + "surf_init_qa", extra_data=pointData)


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

    force = [np.max(np.linalg.norm(c.force(coils), axis=1)) for c in base_coils]
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

    JF = Jf \
        + LENGTH_WEIGHT * QuadraticPenalty(sum(Jls), LENGTH_TARGET, "max") \
        + CC_WEIGHT * Jccdist \
        + CS_WEIGHT * Jcsdist \
        + CURVATURE_WEIGHT * sum(Jcs) \
        + MSC_WEIGHT * sum(QuadraticPenalty(J, MSC_THRESHOLD, "max") for J in Jmscs) \
        + LINKING_WEIGHT * Jlink \
        + FORCE_WEIGHT * Jforce

    def fun(dofs):
        JF.x = dofs
        J = JF.J()
        grad = JF.dJ()
        jf = Jf.J()
        BdotN = np.mean(np.abs(np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)))
        outstr = f"Mode {min_coil_mode}/{order}"
        outstr += f", J={J:.1e}, Jf={jf:.1e}, ⟨B·n⟩={BdotN:.1e}"
        cl_string = ", ".join([f"{J.J():.1f}" for J in Jls])
        kap_string = ", ".join(f"{np.max(c.kappa()):.1f}" for c in base_curves)
        msc_string = ", ".join(f"{J.J():.1f}" for J in Jmscs)
        linking_string = ", ".join(f"{Jlink.J()}")
        outstr += f", Len=sum([{cl_string}])={sum(J.J() for J in Jls):.1f}, ϰ=[{kap_string}], ∫ϰ²/L=[{msc_string}]"
        outstr += f", C-C-Sep={Jccdist.shortest_distance():.2f}, C-S-Sep={Jcsdist.shortest_distance():.2f}"
        outstr += f", Link={linking_string}"
        outstr += f", Jforce={Jforce.J():.1e}"
        outstr += f", ║∇J║={np.linalg.norm(grad):.1e}"
        print(outstr)
        return J, grad

    # Commenting out the Taylor test as it's not needed for repeated runs
    # print("""
    # ################################################################################
    # ### Perform a Taylor test ######################################################
    # ################################################################################
    # """)
    # min_coil_mode = 3
    # f = fun
    # dofs = JF.x
    # np.random.seed(1)
    # h = np.random.uniform(size=dofs.shape)
    # J0, dJ0 = f(dofs)
    # dJh = sum(dJ0 * h)
    # for eps in [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]:
    #     J1, _ = f(dofs + eps*h)
    #     J2, _ = f(dofs - eps*h)
    #     print("err", (J1-J2)/(2*eps) - dJh)

    print("""
    ################################################################################
    ### Run the optimisation #######################################################
    ################################################################################
    """)

    avg_BdotN_over_Bs = []
    max_forces = []
    cs_distances = []
    cc_distances = []
    curvatures = []
    msc_curvatures = []
    lengths = []
    links = []
    start_time = time.time()
    # The original loop `range(order,order+1)` only runs once for the final mode.
    # This behavior is preserved.
    for min_coil_mode in range(19,order+1):
        for c in base_curves:      
            c.unfix_all()                                                                                                 
            fixed_range(c, cmin=min_coil_mode, smin=min_coil_mode, cmax=order, smax=order, fix=True)
        print(f'############################################ Max Coil Mode {min_coil_mode} / {order} ############################################')

        f = fun
        dofs = JF.x
        res = minimize(fun, dofs, jac=True, method='L-BFGS-B', options={'maxiter': MAXITER, 'maxcor': 300}, tol=1e-15)
        dofs = res.x
        BdotN = np.mean(np.abs(np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)))
        avg_BdotN_over_B = BdotN / bs.AbsB().mean()
        avg_BdotN_over_Bs.append(avg_BdotN_over_B)
        force = np.array([np.max(np.linalg.norm(c.force(coils), axis=1)) for c in base_coils])
        max_forces.append(np.max(force))
        cs_distances.append(Jcsdist.shortest_distance())
        cc_distances.append(Jccdist.shortest_distance())
        curvatures.append(np.max(np.array([np.max(c.kappa()) for c in base_curves])))
        msc_curvatures.append(np.max(np.array([np.max(c.kappa()) for c in base_curves])))
        lengths.append(sum(Jls).J())
        links.append(Jlink.J())

    links = np.array(links)
    lengths = np.array(lengths)
    cs_distances = np.array(cs_distances)
    cc_distances = np.array(cc_distances)
    curvatures = np.array(curvatures)
    msc_curvatures = np.array(msc_curvatures)    
    avg_BdotN_over_Bs = np.array(avg_BdotN_over_Bs)
    max_forces = np.array(max_forces)

    # Save optimization data for the current run
    np.savez(OUT_DIR + 'opt_data_{i}.npz', BdotN = avg_BdotN_over_Bs, forces = max_forces, lengths = lengths, 
             cs_distances = cs_distances, cc_distances = cc_distances, curvatures = curvatures, msc_curvatures = msc_curvatures,
             links = links, CC_WEIGHT=CC_WEIGHT, CS_WEIGHT=CS_WEIGHT, LENGTH_WEIGHT=LENGTH_WEIGHT, CURVATURE_WEIGHT=CURVATURE_WEIGHT, 
             MSC_WEIGHT=MSC_WEIGHT, LINKING_WEIGHT=LINKING_WEIGHT, FORCE_WEIGHT=FORCE_WEIGHT)

    bs.save(OUT_DIR + "biot_savart_optimized_auglag_qa.json")
    end_time = time.time()
    print(f"Time taken for run {i+1}: {end_time - start_time:.2f} seconds")
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
    force = [np.max(np.linalg.norm(c.force(coils), axis=1)) for c in base_coils]
    print("Forces:")
    print(",".join(f"{f:.2e}" for f in force))
    coils_to_vtk(coils, OUT_DIR + "coils_optimized_auglag_qa")
    bs.set_points(s_plot.gamma().reshape((-1, 3)))
    calculate_modB_on_major_radius(bs, s_plot)
    bs.set_points(s_plot.gamma().reshape((-1, 3))) 

    pointData = {"B_N": np.sum(bs.B().reshape((qphi, qtheta, 3)) *
                            s_plot.unitnormal(), axis=2)[:, :, None],
            "B_N/|B|": np.sum(bs.B().reshape((qphi, qtheta, 3)) *
                            s_plot.unitnormal(), axis=2)[:, :, None] /
            bs.AbsB().reshape((qphi, qtheta, 1)),
            "modB": bs.AbsB().reshape((qphi, qtheta, 1))}
    s_plot.to_vtk(OUT_DIR + "surf_optimized_auglag_qa", extra_data=pointData)
    max_BdotN_overB = np.max(np.sum(bs.B().reshape((qphi, qtheta, 3)) *
                            s_plot.unitnormal(), axis=2)[:, :, None] /
            bs.AbsB().reshape((qphi, qtheta, 1)))
    bs.set_points(s.gamma().reshape((-1, 3)))
    BdotN = np.mean(np.abs(np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)))
    avg_BdotN_over_B = BdotN / bs.AbsB().mean()
    print("--------------------------------------------------------------------------------------------------------------------------------------------")
    print(f"<B_N>/<|B|> = {avg_BdotN_over_B:.2e}, Max BdotN/|B| = {max_BdotN_overB:.2e}")
    print("--------------------------------------------------------------------------------------------------------------------------------------------")
    print("Final NORMALIZED SQUARED FLUX:", Jf.J())
    print(f"FINISHED OPTIMIZATION RUN {i+1}/{num_runs}")
    print("Output directory:", OUT_DIR)
    print('########### WEIGHTS ############')
    print(f"Run {i+1} weights:")
    print(f"CC_WEIGHT: {CC_WEIGHT:.2e}, CS_WEIGHT: {CS_WEIGHT:.2e}, LENGTH_WEIGHT: {LENGTH_WEIGHT:.2e}, "
          f"CURVATURE_WEIGHT: {CURVATURE_WEIGHT:.2e}, MSC_WEIGHT: {MSC_WEIGHT:.2e}, "
          f"LINKING_WEIGHT: {LINKING_WEIGHT:.2e}, FORCE_WEIGHT: {FORCE_WEIGHT:.2e}")
    print('##################################')