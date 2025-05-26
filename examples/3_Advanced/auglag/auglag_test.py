
''' THIS IS THE FILE FOR COIL OPTIMIZATION USING THE AUGMENTED LAGRANGIAN METHOD'''

import numpy as np
from simsopt.util import MpiPartition
import os
from scipy.optimize import minimize
from simsopt.objectives import SquaredFlux
from simsopt.objectives import QuadraticPenalty

from simsopt.geo import SurfaceRZFourier
from simsopt.geo import curves_to_vtk, create_equally_spaced_curves
from simsopt.geo import (FrameRotation, FramedCurveCentroid, CurveFilament,
                         LinkingNumber)
from simsopt.geo import CurveLength, CurveCurveDistance, \
    MeanSquaredCurvature, LpCurveCurvature, CurveSurfaceDistance , \
        LPTorsionalStrainPenalty, LPBinormalCurvatureStrainPenalty
from simsopt.field import BiotSavart
from simsopt.field import Current, coils_via_symmetries
from simsopt.mhd import Vmec
import matplotlib.pyplot as plt

mpi = MpiPartition()
mpi.write()
OUT_DIR = "./output/"
os.makedirs(OUT_DIR, exist_ok=True)

# filename = '/home/gipe/NeutronCoils/simsopt/pedro_WIP/Configurations/wout_20240303_v1.nc'
# filename = '/home/gipe/epos-simsopt/pedro_WIP/SingleStage/Scans/input_files/input.final_scan_81_R1_iota0.19_A3.5_well0.1'

filename = '/home/gipe/epos-simsopt/pedro_WIP/SingleStage/input.LandremanPaul2021_QA_lowres'
# filename = '/home/gipe/epos-simsopt/pedro_WIP/SingleStage/input.LandremanPaul2021_QH'
# filename = '/home/gipe/NeutronCoils/test_folder/pedro_WIP/output/input.v20250408_v1_lgradb3_scaled_1m'
vmec = Vmec(filename, mpi = mpi, verbose=False)

surf = vmec.boundary
nfp = surf.nfp
surf.to_vtk(OUT_DIR+'QA_v1_surf')
# axis = load_axis(vmec)


plt.rcParams.update({'font.size': 22})

ncoils = 4
rot_order = 3
stellsym = True
tape_width = 3e-3
second_opt = False

R0 = vmec.x[0]

# Minor radius for the initial circular coils:
R1 = 0.6*vmec.x[0]

# Number of Fourier modes describing each Cartesian component of each coil:
order = 5
planar_order = 3

# Squared Flux Weight
FLUX_WEIGHT = 1.5e5

# Weight on the curve lengths in the objective function.
LENGTH_WEIGHT = 10000
LONG_WEIGHT = 2500
LENGTH_TARGET = 2.8

# Threshold and weight for the coil-to-coil distance penalty in the objective function:
CC_THRESHOLD = 0.4
CC_WEIGHT = 1e4 #1e-4

# Threshold and weight for the coil-to-surface distance penalty in the objective function:
CS_THRESHOLD = 0.5
CS_WEIGHT = 1e4#50

# Interlock weight
LK_WEIGHT = 1e4
# Threshold and weight for the curvature penalty in the objective function:
CURVATURE_THRESHOLD = 5
CURVATURE_WEIGHT = 0 #1e-6

# Threshold and weight for the mean squared curvature penalty in the objective function:
MSC_THRESHOLD = 0
MSC_WEIGHT = 0 #1

# Threshold and weight for the torsional strain penalty in the objective function:
BIN_WEIGHT = 1e1
TOR_WEIGHT = 1e1
strain_threshold = 2e-9
    
# Initialize the boundary magnetic surface:
ntheta = 128
nphi = 128
range_torus = "full torus"  
# "field period"
# "half period"
# 
# s = SurfaceRZFourier.from_wout(filename, range=range_torus, nphi=nphi, ntheta=ntheta)
s = SurfaceRZFourier.from_vmec_input(filename, range=range_torus, nphi=nphi, ntheta=ntheta)

non_planar_base_curves = create_equally_spaced_curves(ncoils, s.nfp, stellsym=stellsym, R0=R0, R1=R1, order=order, numquadpoints=128)

rotations = [ FrameRotation( c.quadpoints, rot_order ) for c in non_planar_base_curves ]

framed_curves = [ FramedCurveCentroid( c, rot ) for ( c,rot ) in zip( non_planar_base_curves, rotations ) ]
non_planar_curves = [ CurveFilament( fc, 0, 0 ) for fc in framed_curves ]

base_currents = [Current(7e4) for i in range(ncoils)]
#base_currents[0].fix_all()

base_curves = non_planar_base_curves

curves = non_planar_curves

non_planar_coils = coils_via_symmetries(non_planar_base_curves, base_currents, s.nfp, stellsym)
    
coils = non_planar_coils #+ coils_set1

print("Number of coils:", len(non_planar_coils))

bs = BiotSavart(coils)
bs.set_points(s.gamma().reshape((-1, 3)))
curves = [c.curve for c in coils]
curves_to_vtk(curves, OUT_DIR + "curves_init")
pointData = {"B_N/|B|": np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)[:, :, None]/bs.AbsB().reshape((s.quadpoints_phi.size,s.quadpoints_theta.size,1))}
s.to_vtk(OUT_DIR + "surf_init", extra_data=pointData)
pointData = {"modB": bs.AbsB().reshape((s.quadpoints_phi.size,s.quadpoints_theta.size,1))}
s.to_vtk(OUT_DIR + "surf_modB", extra_data=pointData)
# Define the individual terms objective function:

Jf = SquaredFlux(s, bs, definition="local")
Jls = [CurveLength(c) for c in base_curves]
Jl = sum(QuadraticPenalty(Jl, LENGTH_TARGET, "max") for Jl in Jls) #[CurveLength(c) for c in base_curves] ## Change this to make it with a target
Jccdist = CurveCurveDistance(curves, CC_THRESHOLD, num_basecurves=ncoils)
Jcsdist = CurveSurfaceDistance(curves, s, CS_THRESHOLD)
Jcs = [LpCurveCurvature(c, 2, CURVATURE_THRESHOLD) for c in base_curves]
Jmscs = [MeanSquaredCurvature(c) for c in base_curves]
Jtor = [LPTorsionalStrainPenalty(c, width=tape_width, p=2, threshold=strain_threshold) for c in framed_curves ]
Jbin = [LPBinormalCurvatureStrainPenalty(c, width=tape_width, p=2, threshold=strain_threshold) for c in framed_curves]
Jlink = LinkingNumber(curves, downsample=2)

def jac_constraint(constraint_list, dofs):  
    """Compute the Jacobian of the constraints.
    Args:
        constraint_list: List of constraint functions.
        dofs: Degrees of freedom of the coils.
    Returns:
        J: Jacobian of the constraints."""
    
    J = np.zeros((len(constraint_list), len(dofs)), dtype=float)
    for i, c_i in enumerate(constraint_list):
        dof_dif = len(dofs) - len(c_i.x)
        if not dof_dif:
            J[i, :] = c_i.dJ()
        else:
            J[i,dof_dif:] = c_i.dJ()
    return J


def augmented_lagrangian(f, dofs, c_list, lag_mul, mu, option = None):
    """Compute augmented Lagrangian value.
    Args:
        x: Current point.
        f: Objective function, for coil optimization usually the SquaredFlux.
        c_list: List of coil constraint functions.
        lag_mul: Dictionary of Lagrange multipliers for coil constraints.
        mu: Penalty parameter for the augmented Lagrangian method.
    Returns:
        L: Augmented Lagrangian value."""
        
    f.x = dofs
    c_vals = np.array([J.J() for J in c_list])
    # Equality constraints
    if option == 'least-squares':
        L = 1/2*np.linalg.norm(f.J())**2
        L += 1/2*np.linalg.norm(-lag_mul/np.sqrt(mu) + np.sqrt(mu) * c_vals) ** 2
    else:
        L = f.J()
        L += -np.dot(lag_mul, c_vals) + mu/2*np.linalg.norm(c_vals)**2
    return L 

def grad_augmented_lagrangian(dofs, f, c_list, lag_mul, mu, option = None):
    """Compute augmented Lagrangian value.
    Args:
        x: Current point.
        f: Objective function, for coil optimization usually the SquaredFlux.
        c_list: List of coil constraint functions.
        lag_mul: Dictionary of Lagrange multipliers for coil constraints.
        mu: Penalty parameter for the augmented Lagrangian method.
    Returns:
        L: Augmented Lagrangian value."""
    
    # Calculate the Jacobian Matrix of the constraints vector g
    c_jac = jac_constraint(c_list, f.x)
    
    c_vals = np.array([J.J() for J in c_list])
    
    if option == 'least-squares':
        dL = np.asarray(f.dJ(), dtype=float)*f.J() # Gradient of the objective function
        dL += np.sqrt(mu) * np.dot(c_vals.T, c_jac)
    else:
    #Equality constraints
        dL = np.asarray(f.dJ(), dtype=float)
        for i, c_i in enumerate(c_list):
            dof_dif = len(dofs) - len(c_i.x)
            if not dof_dif: # Make sure that all gradients are the same size
                c_dJ = c_i.dJ()
            else:
                buffer = np.zeros(dof_dif, dtype=float)
                c_dJ = np.concatenate((buffer, c_i.dJ()))
            dL += -lag_mul[i] * c_dJ
        dL += mu * np.dot(c_jac.T, c_vals) # adding the derivative of the L2 norm of the constraints
    return dL 

def progress_arrow(val0, val1):
    if val1 > val0:
        var = "↑"
    elif val1 < val0:
        var = "↓"
    elif val1 == val0:
        var = "="
    return var

def augmented_lagrangian_method(f, c_list=[],mu_init=1.0, grad_tol=1e-6, c_tol=1e-6, MAXITER = 50, argmin_tol=1e-6, 
                                minimize_method = 'L-BFGS-B', MAXITER_lag=1000, lagrangian_form = None):
    """ Wrapper function to run the augmented lagrangian method following the scheme describe in R. Conlin's PhD Thesis Section
    6.3 p96 
        Args:
            f: main optimization function to be included in the augmented lagrangian, typically the squared flux.
            c_list: list containing the Optimizable objects acting as constraints to the optimization problem.
            mu_init: initial value for the penalty weight.
            grad_tol: optimization tolerance of the gradient of the augmented lagrangian.
            c_tol: optimization tolerance for the norm on the constraints.
            MAXITER: maximum iterations for the arg min step.
            argmin_tol: tolerance on the optimization function for the arg min step.
            minimize_method: method used for the arg min step.
            MAXITER_lag: maximum number of iterations for the augmented lagrangian method.
            lagrangian_form: if 'least_squares' will use the least-squares form for the lagrangian as described in the reference 
                            given above. If None, defaults to the traditional augmented lagrangian form.
                            
        Returns: 
            x: array containing the optimized degrees of freedom of the coils.
            res.fun: augmented lagrangian function after the optimization.
            lag_mul: array containing the optimized lagrangian multipliers. """
    
    
    
    if mu_init <= 0 or grad_tol <= 0 or c_tol <= 0:
        raise ValueError("eta_init, mu_init and omega_init  must be strictly positive")
    
    print('----------------------------------------------------------------')
    print(f'METHOD {minimize_method} IS SELECTED FOR THE OPTIMIZATION')
    print(f'INITIAL SQUARED FLUX: {f.J():0.6f}')
    print('----------------------------------------------------------------')
    mu_k = mu_init
    omega_k = 1/mu_init
    eta_k = 1/mu_init**0.1
    k = 1
    m = len(c_list)
    x = f.x
    # Initialize multipliers
    lag_mul = np.zeros(m, dtype=float)
    
    # Evaluate initial lagrangian
    aug_lag = augmented_lagrangian(f, x, c_list, lag_mul, mu_k)
    grad_aug_lag_norm = np.linalg.norm(grad_augmented_lagrangian(x , f, c_list, lag_mul, mu_k, option = lagrangian_form))
    c_vals = np.array([J.J() for J in c_list]) if m > 0 else 0
    c_norm = np.linalg.norm(c_vals)
    
    grad_aug_lag_norm_km1 = grad_aug_lag_norm
    c_norm_km1 = c_norm
    def fun(dofs):
        aug_lag = augmented_lagrangian(f, dofs, c_list, lag_mul, mu_k, option = lagrangian_form)
        grad_aug_lag = grad_augmented_lagrangian(dofs, f, c_list, lag_mul, mu_k, option = lagrangian_form)
        return aug_lag, grad_aug_lag
    
    print("--------------------------------------------------------------------------------------------------------------------------------------------")
    print(f"Iteration {0}, \u03BC_k={mu_k:.2e}, \u03C9_k={omega_k:.2e}, \u03B7_k={eta_k:.2e}, ║∇L_A║ = {grad_aug_lag_norm:.2e}, ║g║ = {c_norm:.2e}")
    
    cl_string = ", ".join([f"{J.J():.1f}" for J in Jls])
    kap_string = ", ".join(f"{np.max(c.kappa()):.1f}" for c in base_curves)
    msc_string = ", ".join(f"{J.J():.1f}" for J in Jmscs)
    bdotn_km1 = np.max(np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)[:, :, None]/bs.AbsB().reshape((s.quadpoints_phi.size,s.quadpoints_theta.size,1)))
    outstr = f"B_N/|B| = {bdotn_km1:.2e}"
    outstr += f", SF = {f.J():0.5f}"
    outstr += f", Len=sum([{cl_string}])={sum(J.J() for J in Jls):.1f}, ϰ=[{kap_string}], ∫ϰ²/L=[{msc_string}]"
    outstr += f", C-C-Sep={Jccdist.shortest_distance():.2f}, C-S-Sep={Jcsdist.shortest_distance():.2f}"
    outstr += f", Link={Jlink.J():.2f}"
    print(outstr)
    print(f'L_A value: {aug_lag:0.5f}')
    print("--------------------------------------------------------------------------------------------------------------------------------------------")

    while (grad_aug_lag_norm > grad_tol or c_norm > c_tol) and k < MAXITER_lag:
        # Solve arg min of the augmented lagrangian
        dofs_before = x
        res = minimize(fun, x, method=minimize_method, options={'disp': False, 'maxiter': MAXITER, 'gtol': omega_k}, jac=True, tol = argmin_tol) #'gtol': omega_k
        dofs_after = res.x
        print('||Δx||:', np.linalg.norm(dofs_after - dofs_before))
        x = res.x   
        
        curves_to_vtk(curves, OUT_DIR +"last_optimized_coils_auglag"+str(k))
        # Evaluate gradient of the augmented Lagrangian
        grad_aug_lag_norm = np.linalg.norm(grad_augmented_lagrangian(x , f, c_list, lag_mul, mu_k, option = lagrangian_form))
        # Evaluate constraints
        c_vals = np.array([J.J() for J in c_list])

        # Check convergence
        c_norm = np.linalg.norm(c_vals, ord=np.inf) if m > 0 else 0
        
        ############### START PRINTING STATEMENTS ##################
        var_grad = progress_arrow(grad_aug_lag_norm_km1, grad_aug_lag_norm)
        grad_aug_lag_norm_km1 = grad_aug_lag_norm
        var = progress_arrow(c_norm_km1, c_norm)
        c_norm_km1 = c_norm
        print(f"Iteration {k}, \u03BC_k={mu_k:.2e}, \u03C9_k={omega_k:.2e}, \u03B7_k={eta_k:.2e}, ║∇L_A║ = {grad_aug_lag_norm:.2e} ({var_grad}), ║g║ = {c_norm:.2e} ({var})")
        bdotn = np.max(np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)[:, :, None]/bs.AbsB().reshape((s.quadpoints_phi.size,s.quadpoints_theta.size,1)))
        var_b = progress_arrow(bdotn_km1, bdotn)
        bdotn_km1 = bdotn    
        cl_string = ", ".join([f"{J.J():.1f}" for J in Jls])
        kap_string = ", ".join(f"{np.max(c.kappa()):.1f}" for c in base_curves)
        msc_string = ", ".join(f"{J.J():.1f}" for J in Jmscs)
        bdotn_km1 = np.max(np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)[:, :, None]/bs.AbsB().reshape((s.quadpoints_phi.size,s.quadpoints_theta.size,1)))
        
        outstr = f"B_N/|B| = {bdotn_km1:.2e} ({var_b})"
        outstr += f", SF = {f.J():0.5f}"
        outstr += f", Len=sum([{cl_string}])={sum(J.J() for J in Jls):.1f}, ϰ=[{kap_string}], ∫ϰ²/L=[{msc_string}]"
        outstr += f", C-C-Sep={Jccdist.shortest_distance():.2f}, C-S-Sep={Jcsdist.shortest_distance():.2f}"
        outstr += f", Link={Jlink.J():.2f}"
        print(outstr)
        ############################ END PRINTING STATEMENTS ########################3
        
        # Constraints are making good progress, update Lagrange multipliers
        if c_norm < eta_k:
            print("*Constraints are satisfied*")
            lag_mul += -mu_k * c_vals
            omega_k = omega_k /mu_k
            eta_k = eta_k / mu_k
        
        # Need to improve constraint satisfaction, increase penalty term
        else:
            print("*Constraints are not satisfied*")
            mu_k = 10*mu_k
            omega_k = 1/mu_k
            eta_k = 1/mu_k
        print("LAGRANGE MULTIPLIERS:", lag_mul)
        print("--------------------------------------------------------------------------------------------------------------------------------------------")

        k += 1

    return x, res.fun, lag_mul

if __name__ == "__main__":
    
    """ opt_method: 'aug' optimizes coils using the augmented lagrangian scheme.
                    'trad' optimizes coils using the traditional local optimizer. 
                    """
                    
    opt_method = 'aug' # 'trad'
    if opt_method == 'aug':
        
        # Main optimization function
        f = Jf
        
        # Constraint list
        c_list = [Jl, Jcsdist, Jccdist]

        MINIMIZE_METHODS_NEW_CB = ['nelder-mead', 'powell', 'cg', 'bfgs', 'newton-cg',
                            'l-bfgs-b', 'trust-constr', 'dogleg', 'trust-ncg',
                            'trust-exact', 'trust-krylov']
        
        x, fnc, lag_mul = augmented_lagrangian_method(f, c_list=c_list, mu_init=10, grad_tol=2e-4, c_tol=1e-4, 
                                                    MAXITER = 100, argmin_tol=1e-16, minimize_method=MINIMIZE_METHODS_NEW_CB[5], MAXITER_lag=500, 
                                                    lagrangian_form=None)
        
        curves_to_vtk(curves, OUT_DIR +"optimized_coils_auglag")
        s.to_vtk(OUT_DIR + "surf_optimized_auglag", extra_data=pointData)
        print("--------------------------------------------------------------------------------------------------------------------------------------------")
        print("INITIAL LAGRANGE MULTIPLIERS:", np.zeros(len(c_list), dtype=float))
        print("FINAL LAGRANGE MULTIPLIERS:", lag_mul)
        print("--------------------------------------------------------------------------------------------------------------------------------------------")
        print("Final SQUARED FLUX:", Jf.J())
        print('FINISHED OPTIMIZATION')

    else:
        print('----------------------------------------------------------------')
        print(f'METHOD {opt_method} IS SELECTED FOR THE OPTIMIZATION')
        print(f'INITIAL SQUARED FLUX: {Jf.J():0.6f}')
        print('----------------------------------------------------------------')
        MAXITER = 200
        JF = Jf

        def fun(dofs):
            JF.x = dofs
            J = JF.J()
            grad = JF.dJ()
            jf = Jf.J()
            BdotN = np.mean(np.abs(np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)))
            outstr = f"J={J:.1e}, Jf={jf:.1e}, ⟨B·n⟩={BdotN:.1e}"
            cl_string = ", ".join([f"{J.J():.1f}" for J in Jls])
            kap_string = ", ".join(f"{np.max(c.kappa()):.1f}" for c in base_curves)
            msc_string = ", ".join(f"{J.J():.1f}" for J in Jmscs)
            outstr += f", Len=sum([{cl_string}])={sum(J.J() for J in Jls):.1f}, ϰ=[{kap_string}], ∫ϰ²/L=[{msc_string}]"
            outstr += f", C-C-Sep={Jccdist.shortest_distance():.2f}, C-S-Sep={Jcsdist.shortest_distance():.2f}"
            outstr += f", ║∇J║={np.linalg.norm(grad):.1e}" 
            print(outstr)


            return J, grad    
    
        f = fun
        dofs = JF.x
        res = minimize(fun, dofs, jac=True, method='L-BFGS-B', options={'maxiter': MAXITER, 'maxcor': 300, 'iprint': 0, 'disp': 1}, tol=1e-16)
        dofs = res.x
        
        curves_to_vtk(curves, OUT_DIR + 'curves_opt_trad')
        bdotn = np.max(np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)[:, :, None]/bs.AbsB().reshape((s.quadpoints_phi.size,s.quadpoints_theta.size,1)))
        outstr = f"B_N/|B| = {bdotn:.2e}"
        print('--------------------------------------')
        print(outstr)
        print("FINAL SQUARED FLUX:", Jf.J())
        print('--------------------------------------')
        print('FINISHED OPTIMIZATION')
