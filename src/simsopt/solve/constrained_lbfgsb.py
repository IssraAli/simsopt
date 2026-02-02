"""
Constrained L-BFGS-B optimizer with hard constraint support during line search.

This module provides a custom L-BFGS-B implementation that supports hard constraints
that must be satisfied at every iteration. Unlike penalty-based approaches, hard
constraints are NOT included in the objective function - they only determine whether
a proposed step is acceptable.

**Key feature**: Uses the DCSRCH line search routine (the exact same Fortran line
search used by scipy's L-BFGS-B), ensuring identical results to scipy when no hard
constraints are active. When hard constraints are provided, infeasible trial points
are rejected during line search by returning a large objective value.

This is particularly useful for topological constraints like LinkingNumber, where:
1. The constraint value is discrete (0, ±1, ±2, ...)
2. The gradient is zero (no useful direction information)
3. Violations should be avoided entirely, not penalized

Example usage:
    >>> optimizer = ConstrainedLBFGSB(
    ...     fun=objective_and_grad,
    ...     x0=initial_dofs,
    ...     hard_constraints=[linking_number_obj],
    ...     feasibility_check=lambda hcs: all(abs(hc.J()) < 0.5 for hc in hcs)
    ... )
    >>> result = optimizer.minimize()
"""

import numpy as np
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Any, Union
from scipy.optimize._linesearch import DCSRCH

__all__ = ['ConstrainedLBFGSB', 'ConstrainedLBFGSBResult', 'minimize_with_hard_constraints']


@dataclass
class ConstrainedLBFGSBResult:
    """Result object from ConstrainedLBFGSB optimization.
    
    Attributes:
        x: Final optimized parameters
        fun: Final objective value
        jac: Final gradient
        nit: Number of iterations
        nfev: Number of function evaluations
        njev: Number of Jacobian evaluations
        success: Whether optimization converged successfully
        message: Description of termination reason
        n_constraint_rejections: Number of steps rejected due to hard constraint violations
        constraint_rejection_history: List of (iteration, alpha) tuples where rejections occurred
    """
    x: np.ndarray
    fun: float
    jac: np.ndarray
    nit: int
    nfev: int
    njev: int
    success: bool
    message: str
    n_constraint_rejections: int
    constraint_rejection_history: List[Tuple[int, float]]


class ConstrainedLBFGSB:
    """
    L-BFGS-B optimizer with support for hard constraints during line search.
    
    Uses the DCSRCH line search routine (the same Fortran line search used by
    scipy's L-BFGS-B), ensuring identical results to scipy when no hard constraints
    are provided. Hard constraints are checked at each trial point during line search;
    infeasible steps are rejected by returning a large objective value, causing
    the line search to backtrack.
    
    Key differences from penalty-based constraint handling:
    1. Hard constraints are NOT in the objective - no gradient contribution
    2. L-BFGS Hessian approximation is only updated with feasible points
    3. The optimizer "sees" a well-conditioned problem
    4. Identical to scipy L-BFGS-B when no hard constraints are active
    
    Parameters
    ----------
    fun : callable
        Objective function. If jac=True, should return (f, g) tuple.
        Signature: fun(x) -> float or fun(x) -> (float, ndarray)
    x0 : ndarray
        Initial guess for parameters.
    jac : bool or callable, optional
        If True, fun returns (f, g). If callable, jac(x) returns gradient.
        Default is True.
    bounds : sequence of (min, max) pairs, optional
        Bounds for each parameter. None means unbounded.
    hard_constraints : list, optional
        List of constraint objects with .J() method. A point is feasible
        if all constraints evaluate to acceptable values.
    feasibility_check : callable, optional
        Function that takes the list of hard_constraints and returns True
        if the current point is feasible. Default checks if all constraints
        have J() values with absolute value < 0.5 (for integer constraints like LinkingNumber).
    maxiter : int, optional
        Maximum number of iterations. Default is 100.
        Note: scipy's L-BFGS-B default is 15000. We use a smaller default since
        this optimizer is typically used within outer loops (e.g., augmented Lagrangian).
    maxcor : int, optional
        Maximum number of variable metric corrections (L-BFGS memory).
        Default is 10. [Matches scipy L-BFGS-B default]
    ftol : float, optional
        Function tolerance for convergence. Iteration stops when
        ``(f^k - f^{k+1})/max{|f^k|,|f^{k+1}|,1} <= ftol``.
        Default is 2.220446049250313e-09. [Matches scipy L-BFGS-B default]
    gtol : float, optional
        Gradient tolerance for convergence. Iteration stops when
        ``max{|proj g_i|} <= gtol`` where proj g_i is the i-th component
        of the projected gradient.
        Default is 1e-5. [Matches scipy L-BFGS-B default]
    maxls : int, optional
        Maximum number of line search steps per iteration.
        Default is 20. [Matches scipy L-BFGS-B default]
    c1 : float, optional
        Armijo (sufficient decrease) condition parameter for line search.
        Default is 1e-4. [Standard value for quasi-Newton methods]
    c2 : float, optional
        Curvature condition parameter for line search.
        Default is 0.9. [Standard value for quasi-Newton methods]
    alpha_init : float, optional
        Initial step size for line search. Default is 1.0.
    alpha_min : float, optional
        Minimum step size before line search gives up. Default is 1e-12.
    verbose : int, optional
        Verbosity level (0=silent, 1=summary, 2=detailed). Default is 0.
    callback : callable, optional
        Called after each iteration: callback(xk).
    
    Notes
    -----
    The following parameters match scipy.optimize.minimize(method='L-BFGS-B') defaults:
    maxcor=10, ftol=2.220446049250313e-09, gtol=1e-5, maxls=20.
    
    The maxiter default (100) differs from scipy's default (15000) because this
    optimizer is designed for use within outer iteration loops where fewer
    inner iterations are typically needed.
    
    See Also
    --------
    scipy.optimize.minimize : General minimization interface
    minimize_with_hard_constraints : Convenience function with scipy-like interface
    """
    
    def __init__(
        self,
        fun: Callable,
        x0: np.ndarray,
        jac: Union[bool, Callable] = True,
        bounds: Optional[List[Tuple[float, float]]] = None,
        hard_constraints: Optional[List[Any]] = None,
        feasibility_check: Optional[Callable] = None,
        maxiter: int = 100,
        maxcor: int = 10,
        ftol: float = 2.220446049250313e-09,  # Matches scipy L-BFGS-B default
        gtol: float = 1e-5,
        maxls: int = 20,
        c1: float = 1e-4,
        c2: float = 0.9,
        alpha_init: float = 1.0,
        alpha_min: float = 1e-12,
        verbose: int = 0,
        callback: Optional[Callable] = None,
    ):
        self.fun = fun
        self.x0 = np.asarray(x0, dtype=np.float64).copy()
        self.n = len(self.x0)
        self.jac = jac
        self.bounds = bounds
        self.hard_constraints = hard_constraints or []
        self.maxiter = maxiter
        self.maxcor = maxcor
        self.ftol = ftol
        self.gtol = gtol
        self.maxls = maxls
        self.c1 = c1
        self.c2 = c2
        self.alpha_init = alpha_init
        self.alpha_min = alpha_min
        self.verbose = verbose
        self.callback = callback
        
        # Default feasibility check: all constraint values must be < 0.5 in absolute value
        # This works for integer constraints like LinkingNumber where 0 = feasible, ±1 = infeasible
        if feasibility_check is None:
            self.feasibility_check = lambda hcs: all(abs(hc.J()) < 0.5 for hc in hcs)
        else:
            self.feasibility_check = feasibility_check
        
        # Parse bounds
        if bounds is not None:
            self.lower = np.array([b[0] if b[0] is not None else -np.inf for b in bounds])
            self.upper = np.array([b[1] if b[1] is not None else np.inf for b in bounds])
        else:
            self.lower = np.full(self.n, -np.inf)
            self.upper = np.full(self.n, np.inf)
        
        # Counters and history
        self.nfev = 0
        self.njev = 0
        self.n_constraint_rejections = 0
        self.constraint_rejection_history = []
        
        # L-BFGS memory storage
        self.S = []  # s_k = x_{k+1} - x_k
        self.Y = []  # y_k = g_{k+1} - g_k
        self.rho = []  # 1 / (y_k^T s_k)
    
    def _evaluate(self, x: np.ndarray) -> Tuple[float, np.ndarray]:
        """Evaluate objective and gradient at x."""
        if self.jac is True:
            result = self.fun(x)
            f, g = result[0], np.asarray(result[1], dtype=np.float64)
            self.nfev += 1
            self.njev += 1
        elif callable(self.jac):
            f = self.fun(x)
            g = np.asarray(self.jac(x), dtype=np.float64)
            self.nfev += 1
            self.njev += 1
        else:
            f = self.fun(x)
            self.nfev += 1
            g = self._numerical_gradient(x)
        return float(f), g
    
    def _numerical_gradient(self, x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        """Compute gradient via finite differences."""
        g = np.zeros(self.n)
        f0 = self.fun(x)
        for i in range(self.n):
            x_plus = x.copy()
            x_plus[i] += eps
            f_plus = self.fun(x_plus)
            g[i] = (f_plus - f0) / eps
            self.nfev += 1
        return g
    
    def _project(self, x: np.ndarray) -> np.ndarray:
        """Project x onto the box constraints."""
        return np.clip(x, self.lower, self.upper)
    
    def _is_feasible(self, x: np.ndarray) -> bool:
        """Check if x satisfies all hard constraints."""
        if not self.hard_constraints:
            return True
        
        # Update DOFs on constraint objects
        for hc in self.hard_constraints:
            try:
                dof_dif = len(x) - len(hc.x)
                hc.x = x[dof_dif:]
            except AttributeError:
                pass  # Constraint doesn't have settable x
        
        return self.feasibility_check(self.hard_constraints)
    
    def _lbfgs_direction(self, g: np.ndarray) -> np.ndarray:
        """
        Compute L-BFGS search direction using two-loop recursion.
        
        Returns -H_k * g where H_k is the inverse Hessian approximation.
        """
        q = g.copy()
        m = len(self.S)
        
        if m == 0:
            # Initial iteration: use steepest descent
            return -g
        
        alpha = np.zeros(m)
        
        # First loop (backward)
        for i in range(m - 1, -1, -1):
            alpha[i] = self.rho[i] * np.dot(self.S[i], q)
            q = q - alpha[i] * self.Y[i]
        
        # Compute H_0 * q (initial Hessian approximation)
        # Use scaled identity: H_0 = (s^T y / y^T y) * I
        s_last = self.S[-1]
        y_last = self.Y[-1]
        gamma = np.dot(s_last, y_last) / (np.dot(y_last, y_last) + 1e-12)
        r = gamma * q
        
        # Second loop (forward)
        for i in range(m):
            beta = self.rho[i] * np.dot(self.Y[i], r)
            r = r + (alpha[i] - beta) * self.S[i]
        
        return -r
    
    def _update_lbfgs_memory(self, s: np.ndarray, y: np.ndarray):
        """Update L-BFGS memory with new s, y pair."""
        ys = np.dot(y, s)
        
        # Skip update if curvature condition not satisfied
        if ys <= 1e-12 * np.dot(y, y):
            if self.verbose >= 2:
                print(f"  Skipping L-BFGS update: y^T s = {ys:.2e}")
            return
        
        if len(self.S) >= self.maxcor:
            self.S.pop(0)
            self.Y.pop(0)
            self.rho.pop(0)
        
        self.S.append(s.copy())
        self.Y.append(y.copy())
        self.rho.append(1.0 / ys)
    
    def _line_search_with_feasibility(
        self, 
        x: np.ndarray, 
        f: float, 
        g: np.ndarray, 
        d: np.ndarray,
        iteration: int,
        old_old_fval: Optional[float] = None
    ) -> Tuple[float, np.ndarray, float, np.ndarray, bool, int, float]:
        """
        Line search using DCSRCH (same Fortran routine as scipy's L-BFGS-B).
        
        Uses scipy.optimize._linesearch.DCSRCH which is the exact same line search
        used internally by scipy's L-BFGS-B, ensuring identical behavior when no
        hard constraints are active.
        
        When hard constraints are provided, feasibility is checked at each trial
        point and infeasible steps are rejected.
        
        Returns:
            alpha: Accepted step size
            x_new: New iterate
            f_new: Function value at x_new
            g_new: Gradient at x_new
            success: Whether a valid step was found
            n_rejected: Number of steps rejected due to constraint violation
            old_fval: Function value for use in next iteration
        """
        descent = np.dot(g, d)
        
        if descent >= 0:
            if self.verbose >= 2:
                print(f"  Warning: Not a descent direction (g^T d = {descent:.2e}), using -g")
            d = -g.copy()
            descent = -np.dot(g, g)
        
        n_rejected = [0]  # Use list to allow modification in closure
        
        # Cache for function evaluations
        cache = {}
        
        def evaluate_at(x_trial):
            """Evaluate objective and gradient at x_trial, with caching."""
            key = tuple(x_trial)
            if key not in cache:
                if self.jac is True:
                    result = self.fun(x_trial)
                    cache[key] = (float(result[0]), np.asarray(result[1], dtype=np.float64))
                elif callable(self.jac):
                    f_val = float(self.fun(x_trial))
                    g_val = np.asarray(self.jac(x_trial), dtype=np.float64)
                    cache[key] = (f_val, g_val)
                else:
                    f_val = float(self.fun(x_trial))
                    g_val = self._numerical_gradient(x_trial)
                    cache[key] = (f_val, g_val)
            return cache[key]
        
        # phi(alpha) = f(x + alpha*d)
        # derphi(alpha) = grad f(x + alpha*d) . d
        def phi(alpha):
            x_trial = self._project(x + alpha * d)
            
            # Check feasibility for hard constraints
            if self.hard_constraints and not self._is_feasible(x_trial):
                n_rejected[0] += 1
                self.constraint_rejection_history.append((iteration, alpha))
                if self.verbose >= 2:
                    print(f"  Line search: alpha={alpha:.2e} REJECTED (constraint violation)")
                # Return large value to force line search to reject this step
                return 1e20
            
            f_val, _ = evaluate_at(x_trial)
            return f_val
        
        def derphi(alpha):
            x_trial = self._project(x + alpha * d)
            
            # If infeasible, return a large positive derivative (uphill)
            if self.hard_constraints and not self._is_feasible(x_trial):
                return 1e20
            
            _, g_val = evaluate_at(x_trial)
            return np.dot(g_val, d)
        
        # L-BFGS-B initial step: 1/||d|| on first iteration, 1.0 thereafter
        d_norm = np.linalg.norm(d)
        if iteration == 1:
            stp0 = 1.0 / d_norm if d_norm > 0 else 1.0
        else:
            stp0 = 1.0
        
        # L-BFGS-B line search parameters (from Fortran source)
        # ftol = 1e-3, gtol = 0.9, xtol = 0.1
        ls = DCSRCH(phi, derphi, ftol=1e-3, gtol=0.9, xtol=0.1, 
                    stpmin=0.0, stpmax=1e10)
        
        try:
            alpha, f_new, _, task = ls(stp0, phi0=f, derphi0=descent, maxiter=self.maxls)
        except Exception as e:
            if self.verbose >= 1:
                print(f"  Line search exception at iteration {iteration}: {e}")
            return 0.0, x, f, g, False, n_rejected[0], f
        
        # Check if line search converged
        if alpha is None or alpha <= 0 or f_new >= 1e19:
            if self.verbose >= 1:
                print(f"  Line search failed at iteration {iteration}")
            return 0.0, x, f, g, False, n_rejected[0], f
        
        x_new = self._project(x + alpha * d)
        
        # Get gradient at new point
        _, g_new = evaluate_at(x_new)
        
        # Update evaluation counters
        if self.jac is True:
            self.nfev += len(cache)
            self.njev += len(cache)
        else:
            self.nfev += len(cache)
            self.njev += len(cache)
        
        if self.verbose >= 2:
            print(f"  Line search: alpha={alpha:.2e} ACCEPTED, f_new={f_new:.6e}")
        
        return alpha, x_new, f_new, g_new, True, n_rejected[0], f
    
    def minimize(self) -> ConstrainedLBFGSBResult:
        """
        Run the constrained L-BFGS-B optimization.
        
        Returns:
            ConstrainedLBFGSBResult with optimization results.
        """
        x = self._project(self.x0.copy())
        
        # Check initial feasibility
        if not self._is_feasible(x):
            return ConstrainedLBFGSBResult(
                x=x,
                fun=np.inf,
                jac=np.zeros(self.n),
                nit=0,
                nfev=self.nfev,
                njev=self.njev,
                success=False,
                message="Initial point is infeasible",
                n_constraint_rejections=0,
                constraint_rejection_history=[]
            )
        
        f, g = self._evaluate(x)
        g_norm = np.linalg.norm(g, ord=np.inf)
        
        if self.verbose >= 1:
            print(f"Initial: f={f:.6e}, ||g||_inf={g_norm:.2e}")
        
        # Check initial convergence
        if g_norm <= self.gtol:
            return ConstrainedLBFGSBResult(
                x=x,
                fun=f,
                jac=g,
                nit=0,
                nfev=self.nfev,
                njev=self.njev,
                success=True,
                message="Initial point satisfies gradient tolerance",
                n_constraint_rejections=0,
                constraint_rejection_history=[]
            )
        
        f_prev = f
        old_old_fval = None  # For scipy line search
        
        for iteration in range(1, self.maxiter + 1):
            # Compute search direction
            d = self._lbfgs_direction(g)
            
            # Line search with feasibility checking (using scipy's line search)
            alpha, x_new, f_new, g_new, ls_success, n_rejected, old_fval = \
                self._line_search_with_feasibility(x, f, g, d, iteration, old_old_fval)
            
            self.n_constraint_rejections += n_rejected
            
            if not ls_success:
                # Try steepest descent as fallback
                d = -g.copy()
                alpha, x_new, f_new, g_new, ls_success, n_rejected, old_fval = \
                    self._line_search_with_feasibility(x, f, g, d, iteration, old_old_fval)
                self.n_constraint_rejections += n_rejected
                
                if not ls_success:
                    return ConstrainedLBFGSBResult(
                        x=x,
                        fun=f,
                        jac=g,
                        nit=iteration,
                        nfev=self.nfev,
                        njev=self.njev,
                        success=False,
                        message=f"Line search failed at iteration {iteration}",
                        n_constraint_rejections=self.n_constraint_rejections,
                        constraint_rejection_history=self.constraint_rejection_history
                    )
            
            # Update L-BFGS memory (only with feasible points)
            s = x_new - x
            y = g_new - g
            self._update_lbfgs_memory(s, y)
            
            # Update iterate
            x = x_new
            g = g_new
            old_old_fval = f_prev  # Store for next iteration's line search
            f_prev = f
            f = f_new
            
            g_norm = np.linalg.norm(g, ord=np.inf)
            
            if self.verbose >= 1:
                print(f"Iter {iteration}: f={f:.6e}, ||g||_inf={g_norm:.2e}, "
                      f"alpha={alpha:.2e}, rejections={n_rejected}")
            
            if self.callback is not None:
                self.callback(x)
            
            # Check convergence
            if g_norm <= self.gtol:
                return ConstrainedLBFGSBResult(
                    x=x,
                    fun=f,
                    jac=g,
                    nit=iteration,
                    nfev=self.nfev,
                    njev=self.njev,
                    success=True,
                    message="Gradient tolerance reached",
                    n_constraint_rejections=self.n_constraint_rejections,
                    constraint_rejection_history=self.constraint_rejection_history
                )
            
            if abs(f_prev - f) <= self.ftol * max(abs(f), abs(f_prev), 1.0):
                return ConstrainedLBFGSBResult(
                    x=x,
                    fun=f,
                    jac=g,
                    nit=iteration,
                    nfev=self.nfev,
                    njev=self.njev,
                    success=True,
                    message="Function tolerance reached",
                    n_constraint_rejections=self.n_constraint_rejections,
                    constraint_rejection_history=self.constraint_rejection_history
                )
        
        return ConstrainedLBFGSBResult(
            x=x,
            fun=f,
            jac=g,
            nit=self.maxiter,
            nfev=self.nfev,
            njev=self.njev,
            success=False,
            message="Maximum iterations reached",
            n_constraint_rejections=self.n_constraint_rejections,
            constraint_rejection_history=self.constraint_rejection_history
        )


def minimize_with_hard_constraints(
    fun: Callable,
    x0: np.ndarray,
    hard_constraints: Optional[List[Any]] = None,
    feasibility_check: Optional[Callable] = None,
    jac: Union[bool, Callable] = True,
    bounds: Optional[List[Tuple[float, float]]] = None,
    options: Optional[dict] = None,
) -> ConstrainedLBFGSBResult:
    """
    Minimize a function subject to hard constraints using constrained L-BFGS-B.
    
    This is a convenience function that creates a ConstrainedLBFGSB optimizer
    and runs it. It provides an interface similar to scipy.optimize.minimize.
    
    Parameters
    ----------
    fun : callable
        Objective function. If jac=True, should return (f, g) tuple.
    x0 : ndarray
        Initial guess for parameters.
    hard_constraints : list, optional
        List of constraint objects with .J() method.
    feasibility_check : callable, optional
        Function that takes the list of hard_constraints and returns True if feasible.
    jac : bool or callable, optional
        If True, fun returns (f, g). If callable, jac(x) returns gradient. Default True.
    bounds : sequence of (min, max) pairs, optional
        Bounds for each parameter.
    options : dict, optional
        Additional options passed to ConstrainedLBFGSB. Supported keys:
        
        - maxiter (int): Maximum iterations. Default 100.
        - maxcor (int): L-BFGS memory size. Default 10. [scipy default]
        - ftol (float): Function tolerance. Default 2.220446049250313e-09. [scipy default]
        - gtol (float): Gradient tolerance. Default 1e-5. [scipy default]
        - maxls (int): Max line search steps. Default 20. [scipy default]
        - c1 (float): Armijo parameter. Default 1e-4.
        - c2 (float): Curvature parameter. Default 0.9.
        - alpha_init (float): Initial step size. Default 1.0.
        - alpha_min (float): Minimum step size. Default 1e-12.
        - verbose (int): Verbosity level. Default 0.
        - callback (callable): Called after each iteration.
    
    Returns
    -------
    ConstrainedLBFGSBResult
        Optimization result.
    
    Notes
    -----
    Default parameters (maxcor, ftol, gtol, maxls) match scipy.optimize.minimize
    with method='L-BFGS-B', except maxiter which defaults to 100 instead of 15000.
    
    Example
    -------
    >>> def rosenbrock_with_grad(x):
    ...     f = (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2
    ...     g = np.array([
    ...         -2*(1 - x[0]) - 400*x[0]*(x[1] - x[0]**2),
    ...         200*(x[1] - x[0]**2)
    ...     ])
    ...     return f, g
    >>> result = minimize_with_hard_constraints(rosenbrock_with_grad, np.array([0.0, 0.0]))
    >>> print(result.x)  # Should be close to [1, 1]
    """
    opts = options or {}
    
    optimizer = ConstrainedLBFGSB(
        fun=fun,
        x0=x0,
        jac=jac,
        bounds=bounds,
        hard_constraints=hard_constraints,
        feasibility_check=feasibility_check,
        maxiter=opts.get('maxiter', 100),
        maxcor=opts.get('maxcor', 10),
        ftol=opts.get('ftol', 2.220446049250313e-09),  # scipy default
        gtol=opts.get('gtol', 1e-5),
        maxls=opts.get('maxls', 20),
        c1=opts.get('c1', 1e-4),
        c2=opts.get('c2', 0.9),
        alpha_init=opts.get('alpha_init', 1.0),
        alpha_min=opts.get('alpha_min', 1e-12),
        verbose=opts.get('verbose', 0),
        callback=opts.get('callback', None),
    )
    
    return optimizer.minimize()
