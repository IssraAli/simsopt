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
from simsopt.objectives import QuadraticPenalty

from simsopt.geo import SurfaceRZFourier
from simsopt.geo import curves_to_vtk, create_equally_spaced_curves
from simsopt.geo import (FrameRotation, FramedCurveCentroid, CurveFilament,
                         LinkingNumber)
from simsopt.geo import CurveLength, CurveCurveDistance, \
    MeanSquaredCurvature, LpCurveCurvature, CurveSurfaceDistance
from simsopt.field import BiotSavart
from simsopt.field import Current, coils_via_symmetries
from pathlib import Path

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
CURVATURE_THRESHOLD = 5
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
Jf = SquaredFlux(s, bs, definition="local")
Jls = [CurveLength(c) for c in base_curves]
Jl = sum(QuadraticPenalty(jj, LENGTH_TARGET, "max") for jj in Jls)
Jccdist = CurveCurveDistance(curves, CC_THRESHOLD, num_basecurves=ncoils)
Jcsdist = CurveSurfaceDistance(curves, s, CS_THRESHOLD)
Jcs = [LpCurveCurvature(c, 2, CURVATURE_THRESHOLD) for c in base_curves]
Jmscs = [MeanSquaredCurvature(c) for c in base_curves]
Jlink = LinkingNumber(curves, downsample=2)

def jac_constraint(constraint_list, dofs):
    """
    Compute the Jacobian matrix of a list of constraint functions with respect to the coil degrees of freedom.

    Args:
        constraint_list (list): List of constraint objects, each with .x and .dJ() methods.
        dofs (np.ndarray): The full vector of coil degrees of freedom.

    Returns:
        np.ndarray: Jacobian matrix of shape (len(constraint_list), len(dofs)), where each row is the gradient of a constraint.
    """
    J = np.zeros((len(constraint_list), len(dofs)), dtype=float)
    for i, c_i in enumerate(constraint_list):
        dof_dif = len(dofs) - len(c_i.x)
        if not dof_dif:
            J[i, :] = c_i.dJ()
        else:
            J[i, dof_dif:] = c_i.dJ()
    return J


def augmented_lagrangian(f, dofs, c_list, lag_mul, mu, option=None):
    """
    Compute the value of the augmented Lagrangian for the current optimization variables.

    The standard augmented Lagrangian is:

        .. math::

            L_A(x, \lambda, \mu) = f(x) - \lambda^T g(x) + \frac{\mu}{2} \|g(x)\|^2

    where:
        - :math:`f(x)`: objective function (e.g., squared flux)
        - :math:`g(x)`: vector of constraint functions
        - :math:`\lambda`: vector of Lagrange multipliers
        - :math:`\mu`: penalty parameter

    If ``option == 'least-squares'``, uses the least-squares form:

        .. math::

            L_{LS}(x, \lambda, \mu) = \frac{1}{2} \|f(x)\|^2 + \frac{1}{2} \left\| -\frac{\lambda}{\sqrt{\mu}} + \sqrt{\mu} g(x) \right\|^2

    Args:
        f: Objective function object (with .J() and .x attributes).
        dofs (np.ndarray): Current degrees of freedom.
        c_list (list): List of constraint objects (with .J() and .dJ() methods).
        lag_mul (np.ndarray): Lagrange multipliers.
        mu (float): Penalty parameter.
        option (str, optional): If 'least-squares', use least-squares form.

    Returns:
        float: Value of the augmented Lagrangian.
    """
    f.x = dofs
    c_vals = np.array([J.J() for J in c_list])
    # Equality constraints
    if option == 'least-squares':
        L = 1 / 2 * np.linalg.norm(f.J())**2
        L += 1 / 2 * np.linalg.norm(-lag_mul /
                                    np.sqrt(mu) + np.sqrt(mu) * c_vals) ** 2
    else:
        L = f.J()
        L += -np.dot(lag_mul, c_vals) + mu / 2 * np.linalg.norm(c_vals)**2
    return L


def grad_augmented_lagrangian(dofs, f, c_list, lag_mul, mu, option=None):
    """
    Compute the gradient of the augmented Lagrangian with respect to the optimization variables.

    For the standard form:

        .. math::

            \nabla_x L_A(x, \lambda, \mu) = \nabla f(x) - \sum_i \lambda_i \nabla g_i(x) + \mu J_g^T g(x)

    where:
        - :math:`J_g`: Jacobian matrix of constraints (rows: constraints, columns: dofs)
        - :math:`g(x)`: vector of constraint values

    For the least-squares form (if ``option == 'least-squares'``):

        .. math::

            \nabla_x L_{LS}(x, \lambda, \mu) = f(x) \nabla f(x) + \sqrt{\mu} J_g^T g(x)

    Args:
        dofs (np.ndarray): Current degrees of freedom.
        f: Objective function object (with .J() and .dJ() methods).
        c_list (list): List of constraint objects.
        lag_mul (np.ndarray): Lagrange multipliers.
        mu (float): Penalty parameter.
        option (str, optional): If 'least-squares', use least-squares form.

    Returns:
        np.ndarray: Gradient of the augmented Lagrangian.
    """
    # Calculate the Jacobian Matrix of the constraints vector g
    c_jac = jac_constraint(c_list, f.x)

    # Ensure all constraint values are floats to avoid dtype=object arrays
    c_vals = np.array([float(J.J()) for J in c_list], dtype=np.float64)

    if option == 'least-squares':
        # Gradient of the objective function
        dL = np.asarray(f.dJ(), dtype=np.float64) * f.J()
        dL = np.asarray(dL + np.sqrt(mu) * np.dot(c_vals.T, c_jac), dtype=np.float64)
    else:
        # Equality constraints
        dL = np.asarray(f.dJ(), dtype=np.float64)
        for i, c_i in enumerate(c_list):
            dof_dif = len(dofs) - len(c_i.x)
            if not dof_dif:  # Make sure that all gradients are the same size
                c_dJ = np.asarray(c_i.dJ(), dtype=np.float64)
            else:
                buffer = np.zeros(dof_dif, dtype=np.float64)
                c_dJ = np.concatenate((buffer, np.asarray(c_i.dJ(), dtype=np.float64)))
            dL = np.asarray(dL - lag_mul[i] * c_dJ, dtype=np.float64)
        # Ensure dL is float64 before adding np.dot result
        dL = np.asarray(dL, dtype=np.float64)
        dot_result = np.asarray(np.dot(c_jac.T, c_vals), dtype=np.float64)
        # adding the derivative of the L2 norm of the constraints
        dL = np.asarray(dL + mu * dot_result, dtype=np.float64)
    return dL


def progress_arrow(val0, val1):
    """
    Utility function to return an arrow indicating progress direction.

    Args:
        val0 (float): Previous value.
        val1 (float): Current value.

    Returns:
        str: '↑' if val1 > val0, '↓' if val1 < val0, '=' if equal.
    """
    if val1 > val0:
        var = "↑"
    elif val1 < val0:
        var = "↓"
    elif val1 == val0:
        var = "="
    return var


def augmented_lagrangian_method(
        f,
        c_list=[],
        mu_init=1.0,
        grad_tol=1e-6,
        c_tol=1e-6,
        MAXITER=50,
        argmin_tol=1e-6,
        minimize_method='L-BFGS-B',
        MAXITER_lag=1000,
        lagrangian_form=None):
    """
    Run the Augmented Lagrangian Method (ALM) for constrained optimization.

    This method solves:

        .. math::

            \min_x f(x) \quad \text{subject to} \quad g_i(x) = 0

    by iteratively minimizing the augmented Lagrangian and updating multipliers and penalty parameters.

    The main loop alternates between minimizing the augmented Lagrangian and updating the multipliers/penalty:
        - Minimize :math:`L_A(x, \lambda, \mu)` with respect to :math:`x`
        - Update :math:`\lambda` and :math:`\mu` based on constraint satisfaction

    Args:
        f: Main objective function (with .J(), .dJ(), .x attributes).
        c_list (list): List of constraint objects.
        mu_init (float): Initial penalty parameter (must be > 0).
        grad_tol (float): Tolerance for gradient norm of Lagrangian.
        c_tol (float): Tolerance for constraint norm.
        MAXITER (int): Max iterations for inner minimization.
        argmin_tol (float): Tolerance for inner minimization.
        minimize_method (str): Optimization method for inner loop.
        MAXITER_lag (int): Max outer ALM iterations.
        lagrangian_form (str, optional): If 'least-squares', use least-squares form.

    Returns:
        tuple: (x, final_Lagrangian_value, lagrange_multipliers)
            - x (np.ndarray): Optimized degrees of freedom
            - final_Lagrangian_value (float): Final value of the augmented Lagrangian
            - lagrange_multipliers (np.ndarray): Final Lagrange multipliers
    """
    if mu_init <= 0 or grad_tol <= 0 or c_tol <= 0:
        raise ValueError(
            "eta_init, mu_init and omega_init  must be strictly positive")

    print('----------------------------------------------------------------')
    print(f'METHOD {minimize_method} IS SELECTED FOR THE OPTIMIZATION')
    print(f'INITIAL SQUARED FLUX: {f.J():0.6f}')
    print('----------------------------------------------------------------')
    mu_k = mu_init
    omega_k = 1.0 / mu_init
    eta_k = 1.0 / mu_init**0.1
    k = 1
    m = len(c_list)
    x = f.x
    # Initialize multipliers
    lag_mul = np.zeros(m, dtype=float)

    # Evaluate initial lagrangian
    aug_lag = augmented_lagrangian(f, x, c_list, lag_mul, mu_k)
    grad_aug_lag_norm = np.linalg.norm(grad_augmented_lagrangian(
        x, f, c_list, lag_mul, mu_k, option=lagrangian_form))
    c_vals = np.array([J.J() for J in c_list]) if m > 0 else 0
    c_norm = np.linalg.norm(c_vals)

    grad_aug_lag_norm_km1 = grad_aug_lag_norm
    c_norm_km1 = c_norm

    def fun(dofs):
        aug_lag = augmented_lagrangian(
            f, dofs, c_list, lag_mul, mu_k, option=lagrangian_form)
        grad_aug_lag = grad_augmented_lagrangian(
            dofs, f, c_list, lag_mul, mu_k, option=lagrangian_form)
        return aug_lag, grad_aug_lag

    print("--------------------------------------------------------------------------------------------------------------------------------------------")
    print(f"Iteration {0}, \u03BC_k={mu_k:.2e}, \u03C9_k={omega_k:.2e}, \u03B7_k={eta_k:.2e}, \u221A║∇L_A║ = {grad_aug_lag_norm:.2e}, \u221A║g║ = {c_norm:.2e}")

    cl_string = ", ".join([f"{J.J():.1f}" for J in Jls])
    kap_string = ", ".join(f"{np.max(c.kappa()):.1f}" for c in base_curves)
    msc_string = ", ".join(f"{J.J():.1f}" for J in Jmscs)
    BdotN = np.mean(np.abs(np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)))
    BdotN_over_B = BdotN / bs.AbsB().mean()
    outstr = f"<B_N> = {BdotN:.2e}"
    outstr += f"<B_N>/<|B|> = {BdotN_over_B:.2e}"
    outstr += f", SF = {f.J():0.5f}"
    outstr += f", Len=sum([{cl_string}])={sum(J.J() for J in Jls):.1f}, \u221A║g║=[{kap_string}], \u222B\u221A²/L=[{msc_string}]"
    outstr += f", C-C-Sep={Jccdist.shortest_distance():.2f}, C-S-Sep={Jcsdist.shortest_distance():.2f}"
    outstr += f", Link={Jlink.J():.2f}"
    print(outstr)
    print(f'L_A value: {aug_lag:0.5f}')
    print("--------------------------------------------------------------------------------------------------------------------------------------------")

    while (grad_aug_lag_norm > grad_tol or c_norm > c_tol) and k < MAXITER_lag:
        # Solve arg min of the augmented lagrangian
        dofs_before = x
        res = minimize(
            fun,
            x,
            method=minimize_method,
            options={
                'disp': False,
                'maxiter': MAXITER,
                'gtol': omega_k},
            jac=True,
            tol=argmin_tol)  # 'gtol': omega_k
        dofs_after = res.x
        print('||Δx||:', np.linalg.norm(dofs_after - dofs_before))
        x = res.x

        curves_to_vtk(curves, OUT_DIR + "last_optimized_coils_auglag" + str(k))
        # Evaluate gradient of the augmented Lagrangian
        grad_aug_lag_norm = np.linalg.norm(grad_augmented_lagrangian(
            x, f, c_list, lag_mul, mu_k, option=lagrangian_form))
        # Evaluate constraints
        c_vals = np.array([J.J() for J in c_list])

        # Check convergence
        c_norm = np.linalg.norm(c_vals, ord=np.inf) if m > 0 else 0

        ############### START PRINTING STATEMENTS ##################
        var_grad = progress_arrow(grad_aug_lag_norm_km1, grad_aug_lag_norm)
        grad_aug_lag_norm_km1 = grad_aug_lag_norm
        var = progress_arrow(c_norm_km1, c_norm)
        c_norm_km1 = c_norm
        print(f"Iteration {k}, \u03BC_k={mu_k:.2e}, \u03C9_k={omega_k:.2e}, \u03B7_k={eta_k:.2e}, \u221A║∇L_A║ = {grad_aug_lag_norm:.2e} ({var_grad}), \u221A║g║ = {c_norm:.2e} ({var})")
        BdotN = np.mean(np.abs(np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)))
        BdotN_over_B = BdotN / bs.AbsB().mean()
        var_b = progress_arrow(BdotN_over_B, BdotN)
        cl_string = ", ".join([f"{J.J():.1f}" for J in Jls])
        kap_string = ", ".join(f"{np.max(c.kappa()):.1f}" for c in base_curves)
        msc_string = ", ".join(f"{J.J():.1f}" for J in Jmscs)
        outstr = f"<B_N> = {BdotN:.2e}"
        outstr += f"<B_N>/<|B|> = {BdotN_over_B:.2e} ({var_b})"
        outstr += f", SF = {f.J():0.5f}"
        outstr += f", Len=sum([{cl_string}])={sum(J.J() for J in Jls):.1f}, \u221A║g║=[{kap_string}], \u222B\u221A²/L=[{msc_string}]"
        outstr += f", C-C-Sep={Jccdist.shortest_distance():.2f}, C-S-Sep={Jcsdist.shortest_distance():.2f}"
        outstr += f", Link={Jlink.J():.2f}"
        print(outstr)
        # END PRINTING STATEMENTS ########################3

        # Constraints are making good progress, update Lagrange multipliers
        if c_norm < eta_k:
            print("*Constraints are satisfied*")
            lag_mul += -mu_k * c_vals
            omega_k = omega_k / mu_k
            eta_k = eta_k / mu_k

        # Need to improve constraint satisfaction, increase penalty term
        else:
            print("*Constraints are not satisfied*")
            mu_k = 10.0 * mu_k
            omega_k = 1.0 / mu_k
            eta_k = 1.0 / mu_k
        print("LAGRANGE MULTIPLIERS:", lag_mul)
        print("--------------------------------------------------------------------------------------------------------------------------------------------")

        k += 1

    return x, res.fun, lag_mul


if __name__ == "__main__":

    """ opt_method: 'aug' optimizes coils using the augmented lagrangian scheme.
                    'trad' optimizes coils using the traditional local optimizer.
                    """

    opt_method = 'aug'  # 'trad'
    if opt_method == 'aug':

        # Main optimization function
        f = Jf

        # Constraint list
        # c_list = [Jl, Jcsdist, QuadraticPenalty(sum(Jls), LENGTH_TARGET, "max"), sum(Jcs)]
        c_list = [Jf, Jcsdist, QuadraticPenalty(sum(Jls), LENGTH_TARGET, "max"), sum(Jcs)]

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

        x, fnc, lag_mul = augmented_lagrangian_method(f, c_list=c_list, mu_init=10, grad_tol=1e-10, c_tol=1e-10,
                                                      MAXITER=200, argmin_tol=1e-16, minimize_method=MINIMIZE_METHODS_NEW_CB[5], MAXITER_lag=50,
                                                      lagrangian_form=None)

        curves_to_vtk(curves, OUT_DIR + "optimized_coils_auglag")
        bs.set_points(s_plot.gamma().reshape((-1, 3)))
        pointData = {"B_N/|B|": np.sum(bs.B().reshape((qphi, qtheta, 3)) *
                               s_plot.unitnormal(), axis=2)[:, :, None] /
             bs.AbsB().reshape((qphi, qtheta, 1)),
             "modB": bs.AbsB().reshape((qphi, qtheta, 1))}
        s_plot.to_vtk(OUT_DIR + "surf_optimized_auglag", extra_data=pointData)
        bs.set_points(s.gamma().reshape((-1, 3)))
        print("--------------------------------------------------------------------------------------------------------------------------------------------")
        print(
            "INITIAL LAGRANGE MULTIPLIERS:",
            np.zeros(
                len(c_list),
                dtype=float))
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
            """Objective and gradient for traditional optimization."""
            JF.x = dofs
            J = JF.J()
            grad = JF.dJ()
            jf = Jf.J()
            BdotN = np.mean(
                np.abs(
                    np.sum(
                        bs.B().reshape(
                            (nphi, ntheta, 3)) * s.unitnormal(), axis=2)))
            outstr = f"J={J:.1e}, Jf={jf:.1e}, <B·n>={BdotN:.1e}"
            cl_string = ", ".join([f"{J.J():.1f}" for J in Jls])
            kap_string = ", ".join(
                f"{np.max(c.kappa()):.1f}" for c in base_curves)
            msc_string = ", ".join(f"{J.J():.1f}" for J in Jmscs)
            outstr += f", Len=sum([{cl_string}])={sum(J.J() for J in Jls):.1f}, kappa=[{kap_string}], int_kappa2/L=[{msc_string}]"
            outstr += f", C-C-Sep={Jccdist.shortest_distance():.2f}, C-S-Sep={Jcsdist.shortest_distance():.2f}"
            outstr += f", ||∇J||={np.linalg.norm(grad):.1e}"
            print(outstr)

            return J, grad

        f = fun
        dofs = JF.x
        res = minimize(
            fun,
            dofs,
            jac=True,
            method='L-BFGS-B',
            options={
                'maxiter': MAXITER,
                'maxcor': 300,
                'iprint': 0,
                'disp': 1},
            tol=1e-16)
        dofs = res.x

        curves_to_vtk(curves, OUT_DIR + 'curves_opt_trad')
        bdotn = np.max(
            np.sum(
                bs.B().reshape(
                    (nphi,
                     ntheta,
                     3)) *
                s.unitnormal(),
                axis=2)[
                :,
                :,
                None] /
            bs.AbsB().reshape(
                (s.quadpoints_phi.size,
                 s.quadpoints_theta.size,
                 1)))
        outstr = f"B_N/|B| = {bdotn:.2e}"
        print('--------------------------------------')
        print(outstr)
        print("FINAL SQUARED FLUX:", Jf.J())
        print('--------------------------------------')
        print('FINISHED OPTIMIZATION')
