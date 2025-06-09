import numpy as np
from scipy.optimize import minimize
from simsopt.geo import curves_to_vtk
from simsopt.objectives import SquaredFlux
import threading # Import the threading module
from threadpoolctl import threadpool_limits

__all__ = ['augmented_lagrangian_objective', 
           'grad_augmented_lagrangian', 'augmented_lagrangian_method',
]

class dummyObjective:
    def __init__(self, x):
        self.x = x
    def J(self):
        return 0.0
    def dJ(self):
        return np.zeros_like(self.x)

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
        # ignore any Current class dofs, which are assumed to be at the beginning of the dofs array
        dof_dif = len(dofs) - len(c_i.x) 
        J[i, dof_dif:] = c_i.dJ()
    return J


def augmented_lagrangian_objective(dofs, f, equality_constraints, lag_mul, mu, option=None):
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
        equality_constraints (list): List of constraint objects (with .J() and .dJ() methods).
        lag_mul (np.ndarray): Lagrange multipliers.
        mu (float): Penalty parameter.
        option (str, optional): If 'least-squares', use least-squares form.

    Returns:
        float: Value of the augmented Lagrangian.
    """
    f.x = dofs
    for c in equality_constraints:
        dof_dif = len(dofs) - len(c.x)
        c.x = dofs[dof_dif:]  # Ensure constraint is evaluated at current dofs
    c_vals = np.array([J.J() for J in equality_constraints])
    # Equality constraints
    if option == 'least-squares':
        L = 1 / 2 * np.linalg.norm(f.J())**2
        L += 1 / 2 * np.linalg.norm(-lag_mul /
                                    np.sqrt(mu) + np.sqrt(mu) * c_vals) ** 2
    else:
        # print(f.J(), lag_mul @ c_vals, mu / 2.0 * np.linalg.norm(c_vals)**2)
        L = f.J() - lag_mul @ c_vals + mu / 2.0 * np.linalg.norm(c_vals)**2
    return L


def grad_augmented_lagrangian(dofs, f, equality_constraints, lag_mul, mu, option=None):
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
        equality_constraints (list): List of constraint objects.
        lag_mul (np.ndarray): Lagrange multipliers.
        mu (float): Penalty parameter.
        option (str, optional): If 'least-squares', use least-squares form.

    Returns:
        np.ndarray: Gradient of the augmented Lagrangian.
    """
    f.x = dofs
    for c in equality_constraints:
        dof_dif = len(dofs) - len(c.x)
        c.x = dofs[dof_dif:]  # Ensure constraint is evaluated at current dofs
    # Calculate the Jacobian Matrix of the constraints vector g
    # t0 = time.time()
    c_jac = jac_constraint(equality_constraints, f.x)
    # t1 = time.time()
    # print(f"[profile] jac_constraint: {t1-t0:.3f}s")

    # Ensure all constraint values are floats to avoid dtype=object arrays
    c_vals = np.array([float(J.J()) for J in equality_constraints], dtype=np.float64)

    if option == 'least-squares':
        # Gradient of the objective function
        dL = np.asarray(f.dJ(), dtype=np.float64) * f.J()
        dL = np.asarray(dL + mu * np.dot(c_vals.T - lag_mul / mu, c_jac), dtype=np.float64)
    else:
        # Equality constraints
        dL = f.dJ() - lag_mul @ c_jac + mu * np.dot(c_jac.T, c_vals)
    return dL

def augmented_lagrangian_method(
        f=None,
        equality_constraints=[],
        mu_init=10.0,
        grad_tol=1e-15,
        c_tol=1e-15,
        MAXITER=50,
        argmin_tol=1e-15,
        minimize_method='L-BFGS-B',
        MAXITER_lag=10,
        lagrangian_form=None,
        OUT_DIR='',
        verbose=False,
        ):
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
        f (Optimizable, optional): Main objective function (with .J(), .dJ(), .x attributes).
            If not provided, only constraints are used in the optimization.
        equality_constraints (list of Optimizable): List of constraint objects corresponding to equality constraints.
            These are equivalent to inequality constraints if the constraint is set up so that if the value is below
            some threshold, the value is set to zero.
        mu_init (float, optional, default=10.0): Initial penalty parameter (must be > 0).
        grad_tol (float, optional, default=1e-15): Tolerance for gradient norm of Lagrangian.
        c_tol (float, optional, default=1e-15): Tolerance for constraint norm.
        MAXITER (int, optional, default=50): Max iterations for inner minimization.
        argmin_tol (float, optional, default=1e-15): Tolerance for inner minimization.
        minimize_method (str, optional, default='L-BFGS-B'): Optimization method for inner loop.
        MAXITER_lag (int, optional, default=10): Max outer ALM iterations.
        lagrangian_form (str, optional, default=None): If 'least-squares', use least-squares form.
        OUT_DIR (str, optional, default=''): Directory for output files.
        verbose (bool, optional, default=False): Whether to print verbose output.

    Returns:
        tuple: (x, final_Lagrangian_value, lagrange_multipliers)
            - x (np.ndarray): Optimized degrees of freedom
            - final_Lagrangian_value (float): Final value of the augmented Lagrangian
            - lagrange_multipliers (np.ndarray): Final Lagrange multipliers
    """
    np.random.seed(1)
    if mu_init <= 0 or grad_tol <= 0 or c_tol <= 0:
        raise ValueError(
            "eta_init, mu_init and omega_init  must be strictly positive")

    k = 1

    # Picks the most dofs from the objective function or the first equality constraint
    try:
        x = f.x 
        if len(x) < len(equality_constraints[0].Jobj.x):
            x = equality_constraints[0].x
    except:
        x = equality_constraints[0].x
    
    m_eq = len(equality_constraints)

    if verbose:
        print('----------------------------------------------------------------')
        print(f'METHOD {minimize_method} IS SELECTED FOR THE OPTIMIZATION')
        try:
            print(f'INITIAL SQUARED FLUX: {f.J():0.6f}')
        except:
            print(f'INITIAL SQUARED FLUX: {equality_constraints[0].J():0.6f}')
        print('----------------------------------------------------------------')

    if f is None:
        f = dummyObjective(x)

    # Initialize multipliers randomly -- DO NOT INITIALIZE TO ZEROS since then the
    # first iteration will not be able to improve the constraints and DO NOT
    # INITIALIZE TO -mu_init * c_vals since then some of the lagrange multipliers
    # will be set to exactly zero, turning off the corresponding constraint for 
    # all remaining iterations. However, need to make sure lag_mul matches c_vals sign
    c_vals = np.array([J.J() for J in equality_constraints])
    lag_mul = -np.random.rand(m_eq) * np.sign(c_vals)

    # Evaluate initial lagrangian
    c_norm = np.linalg.norm(c_vals)
    mu_k = mu_init
    omega_k = 1.0 / mu_init
    eta_k = 1.0 / mu_init ** 0.1
    aug_lag = augmented_lagrangian_objective(x, f, equality_constraints, lag_mul, mu_k, option=lagrangian_form)
    grad_aug_lag_norm = np.linalg.norm(grad_augmented_lagrangian(
        x, f, equality_constraints, lag_mul, mu_k, option=lagrangian_form))

    if verbose:
        print("--------------------------------------------------------------------------------------------------------------------------------------------")
        print(f"Iteration {0}, \u03BC_k={mu_k:.2e}, \u03C9_k={omega_k:.2e}, \u03B7_k={eta_k:.2e}, \u221A║∇L_A║ = {grad_aug_lag_norm:.2e}, \u221A║g║ = {c_norm:.2e}")
        print(f'L_A value: {aug_lag:0.5f}')
        print("--------------------------------------------------------------------------------------------------------------------------------------------")

    options = {
        'disp': False,
        'maxiter': MAXITER,
    }
    if minimize_method == 'L-BFGS-B':
        options['maxcor'] = 100

    # If the penalty parameter is too large, stop the optimization
    while (grad_aug_lag_norm > grad_tol or c_norm > c_tol) and k < MAXITER_lag:
        # Solve arg min of the augmented lagrangian
        dofs_before = x.copy()
        def fun(dofs):
            aug_lag = augmented_lagrangian_objective(
                dofs, f, equality_constraints, lag_mul, mu_k, option=lagrangian_form)
            grad_aug_lag = grad_augmented_lagrangian(
                dofs, f, equality_constraints, lag_mul, mu_k, option=lagrangian_form)
            return aug_lag, grad_aug_lag
        options['gtol'] = omega_k
        # options['ftol'] = omega_k
        # t_min_start = time.time()
        # import cProfile, pstats, io
        # from pstats import SortKey
        # pr = cProfile.Profile()
        # pr.enable()
        if k == 1:
            print("------------------------------------------------------------------------------------------------")
            print("Taylor test:")
            h = np.random.uniform(size=x.shape)
            J0, dJ0 = fun(x)
            dJh = sum(dJ0 * h)
            err = 1e100
            for eps in [1e-2, 1e-3, 1e-4]:
                J1, _ = fun(x + eps*h)
                J2, _ = fun(x - eps*h)
                err_new = np.abs((J1-J2)/(2*eps) - dJh)
                print("err", err, "err_new", err_new)
                if not (err_new < 0.3 * err):
                    print("Taylor test failed, err_new = {:.2e}, err = {:.2e}".format(err_new, err))
                    raise ValueError("Taylor test failed, check your objective and constraint functions")
                err = err_new
            print("Taylor test passed")
            print("------------------------------------------------------------------------------------------------")
        x = dofs_before.copy()
        res = minimize(fun, x, method=minimize_method, options=options, 
                        jac=True, tol=argmin_tol)
        # pr.disable()
        # ss = io.StringIO()
        # sortby = SortKey.TIME
        # ps = pstats.Stats(pr, stream=ss).sort_stats(sortby)
        # ps.print_stats(10)
        # print(ss.getvalue())
        # t_min_end = time.time()
        # print(f"[profile] minimize: {t_min_end-t_min_start:.3f}s")
        dofs_after = res.x
        if verbose:
            print('||Δx||:', np.linalg.norm(dofs_after - dofs_before))
        x = res.x

        grad_vec = grad_augmented_lagrangian(x, f, equality_constraints, lag_mul, mu_k, option=lagrangian_form)
        
        if isinstance(f, SquaredFlux):
            curves = [c.curve for c in f.field.coils]  # f is assumed to be a SquaredFlux object 
            curves_to_vtk(curves, OUT_DIR + "last_optimized_coils_auglag" + str(k))
        else:
            try:
                curves = [c.curve for c in equality_constraints[0].Jobj.field.coils]  # equality_constraints[0] is assumed to be a SquaredFlux object
                curves_to_vtk(curves, OUT_DIR + "last_optimized_coils_auglag" + str(k))
            except:
                # print("WARNING: equality_constraints[0] is not a SquaredFlux object, so no coils will be visualized")
                pass

        # Evaluate gradient of the augmented Lagrangian
        grad_aug_lag_norm = np.linalg.norm(grad_vec)
        # Evaluate constraints
        c_vals = np.array([J.J() for J in equality_constraints])

        # Check convergence
        c_norm = np.linalg.norm(c_vals, ord=np.inf) if m_eq > 0 else 0

        # Print detailed progress
        if verbose:
            print(f"Iteration {k}")
            # print(f"  x = {x}")
            print(f"  Objective f(x) = {f.J()}")
            print(f"  Constraints g(x) = {c_vals}")
            print(f"  Constraint norm = {c_norm}")
            print(f"  Gradient norm = {grad_aug_lag_norm}")
            print(f"  Lagrange multipliers = {lag_mul}")
            print(f"  Penalty parameter mu_k = {mu_k}")
            print(f"  Penalty parameter omega_k = {omega_k}")
            print(f"  Penalty parameter eta_k = {eta_k}")
            print(f"  Change in x = {np.linalg.norm(x - dofs_before)}")
            print("--------------------------------------------------")

        if m_eq > 0:
            # try:
            #     print(f"Iteration {k}")
            #     print('Deviation from target: NSF = {:.2e}, CS-Sep = {:.2e}, CC-Sep = {:.2e}, Len = {:.2e}, Curv = {:.2e}, Link = {:.2e}'.format(
            #         abs(c_vals[0]), abs(c_vals[1]), abs(c_vals[2]), abs(c_vals[3]), abs(c_vals[4]), abs(c_vals[5])))
            #     if verbose:
            #         print('Contributions to the objective:', lag_mul * c_vals, mu_k / 2 * np.linalg.norm(c_vals)**2)
            # except:
            c_str = f"Iter {k}: " + 'Jf = {:.2e}'.format(f.J()) + ', '
            for i, c in enumerate(c_vals):
                c_str += 'c{:d} = {:.2e}, '.format(i, abs(c))
            print(c_str)
        if verbose:
            try:
                print('Max curvatures:', [np.max(c.kappa()) for c in equality_constraints[1].Jobj.curves])
            except:
                pass
        # Constraints are making good progress, update Lagrange multipliers
        if c_norm < eta_k:
            if verbose:
                print("*Constraints are satisfied*")
            lag_mul += -mu_k * c_vals
            omega_k = max(omega_k / mu_k, grad_tol)
            eta_k = max(eta_k / mu_k, c_tol)

        # Need to improve constraint satisfaction, increase penalty term
        else:
            if verbose:
                print("*Constraints are not satisfied*")
            mu_k = 10.0 * mu_k
            omega_k = max(1.0 / mu_k, grad_tol)
            eta_k = max(1.0 / mu_k, c_tol)
        if verbose:
            print("LAGRANGE MULTIPLIERS:", lag_mul)
            print("--------------------------------------------------------------------------------------------------------------------------------------------")

        k += 1

    if verbose:
        print('While loop finished because something became false: \n',
            'grad_tol_check = ', grad_aug_lag_norm > grad_tol,
            ', c_tol_check = ', c_norm > c_tol,
            ', maxiter_check = ', k < MAXITER_lag,
        )
    try:
        return x, res.fun, lag_mul
    except:  # while loop did not run a single time
        return x, None, lag_mul