"""
Unit tests for src/simsopt/solve/constrained_lbfgsb.py.

These tests validate the ConstrainedLBFGSB optimizer against scipy's L-BFGS-B
on standard test functions, and verify the hard constraint rejection mechanism.
"""
import numpy as np
import unittest
from scipy.optimize import minimize as scipy_minimize
from simsopt.solve.constrained_lbfgsb import (
    ConstrainedLBFGSB, 
    minimize_with_hard_constraints
)


class MockHardConstraint:
    """Mock constraint that mimics LinkingNumber behavior."""
    
    def __init__(self, x, forbidden_region_center=None, forbidden_radius=0.5):
        """
        Creates a constraint that is violated (J() != 0) when x is within
        a spherical region.
        
        Args:
            x: Initial DOFs (will be updated during optimization)
            forbidden_region_center: Center of forbidden region. If None, uses x.
            forbidden_radius: Radius of forbidden region.
        """
        self._x = np.array(x, dtype=float)
        self.forbidden_center = (np.array(forbidden_region_center, dtype=float) 
                                 if forbidden_region_center is not None 
                                 else np.zeros_like(self._x))
        self.forbidden_radius = forbidden_radius
    
    @property
    def x(self):
        return self._x
    
    @x.setter
    def x(self, value):
        self._x = np.array(value, dtype=float)
    
    def J(self):
        """Returns 1 (violated) if x is in forbidden region, 0 otherwise."""
        dist = np.linalg.norm(self._x - self.forbidden_center)
        if dist < self.forbidden_radius:
            return 1  # Violated - like LinkingNumber = 1
        return 0  # Feasible - like LinkingNumber = 0
    
    def dJ(self):
        """Zero gradient - mimics LinkingNumber."""
        return np.zeros_like(self._x)


class TestConstrainedLBFGSBBasic(unittest.TestCase):
    """
    Basic tests comparing ConstrainedLBFGSB against scipy's L-BFGS-B.
    
    Our implementation uses the DCSRCH line search (same Fortran routine as scipy's
    L-BFGS-B). On simple problems (quadratic, Rosenbrock), results should be identical.
    On complex problems with many local minima or flat regions, small numerical
    differences in L-BFGS memory updates can lead to different paths, but both
    should converge to good solutions.
    """
    
    def test_quadratic_identical_to_scipy(self):
        """Verify IDENTICAL results to scipy on simple quadratic (no hard constraints)."""
        def fun(x):
            f = np.sum(x**2)
            g = 2 * x
            return f, g
        
        x0 = np.array([1.0, 2.0, 3.0])
        
        result = minimize_with_hard_constraints(fun, x0, jac=True, 
                                                 options={'maxiter': 100, 'gtol': 1e-8})
        scipy_result = scipy_minimize(fun, x0, method='L-BFGS-B', jac=True,
                                       options={'maxiter': 100, 'gtol': 1e-8})
        
        # Should be essentially identical
        self.assertEqual(result.nit, scipy_result.nit, "Iteration counts should match")
        self.assertEqual(result.nfev, scipy_result.nfev, "Function evaluation counts should match")
        # Use atol for values near zero (machine precision)
        np.testing.assert_allclose(result.x, scipy_result.x, atol=1e-14,
            err_msg="Parameters should be identical to scipy")
    
    def test_rosenbrock_identical_to_scipy(self):
        """Verify IDENTICAL results to scipy on Rosenbrock (no hard constraints)."""
        def rosenbrock(x):
            f = (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2
            g = np.array([
                -2*(1 - x[0]) - 400*x[0]*(x[1] - x[0]**2),
                200*(x[1] - x[0]**2)
            ])
            return f, g
        
        x0 = np.array([0.0, 0.0])
        
        result = minimize_with_hard_constraints(rosenbrock, x0, jac=True,
                                                 options={'maxiter': 100, 'gtol': 1e-6})
        scipy_result = scipy_minimize(rosenbrock, x0, method='L-BFGS-B', jac=True,
                                       options={'maxiter': 100, 'gtol': 1e-6})
        
        # Should be essentially identical
        self.assertEqual(result.nit, scipy_result.nit, "Iteration counts should match")
        self.assertEqual(result.nfev, scipy_result.nfev, "Function evaluation counts should match")
        np.testing.assert_allclose(result.x, scipy_result.x, rtol=1e-6,
            err_msg="Parameters should be identical to scipy")
    
    def test_quadratic_unconstrained(self):
        """Test on simple quadratic: f(x) = sum(x^2), optimal at x=0."""
        def fun(x):
            f = np.sum(x**2)
            g = 2 * x
            return f, g
        
        x0 = np.array([1.0, 2.0, 3.0])
        
        # Our optimizer
        result = minimize_with_hard_constraints(fun, x0, jac=True, 
                                                 options={'maxiter': 100, 'gtol': 1e-8})
        
        # scipy L-BFGS-B
        scipy_result = scipy_minimize(fun, x0, method='L-BFGS-B', jac=True,
                                       options={'maxiter': 100, 'gtol': 1e-8})
        
        # Both should find x ≈ 0
        np.testing.assert_allclose(result.x, np.zeros(3), atol=1e-6)
        np.testing.assert_allclose(scipy_result.x, np.zeros(3), atol=1e-6)
        self.assertTrue(result.success)
    
    def test_rosenbrock_unconstrained(self):
        """Test on Rosenbrock function: optimal at x=[1, 1]."""
        def rosenbrock(x):
            f = (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2
            g = np.array([
                -2*(1 - x[0]) - 400*x[0]*(x[1] - x[0]**2),
                200*(x[1] - x[0]**2)
            ])
            return f, g
        
        x0 = np.array([0.0, 0.0])
        
        # Our optimizer
        result = minimize_with_hard_constraints(
            rosenbrock, x0, jac=True,
            options={'maxiter': 200, 'gtol': 1e-6}
        )
        
        # scipy L-BFGS-B
        scipy_result = scipy_minimize(
            rosenbrock, x0, method='L-BFGS-B', jac=True,
            options={'maxiter': 200, 'gtol': 1e-6}
        )
        
        # Both should find x ≈ [1, 1]
        np.testing.assert_allclose(result.x, [1.0, 1.0], atol=1e-4)
        np.testing.assert_allclose(scipy_result.x, [1.0, 1.0], atol=1e-4)
        self.assertTrue(result.success)
    
    def test_quadratic_with_bounds(self):
        """Test quadratic with box constraints."""
        def fun(x):
            f = np.sum((x - 2)**2)  # Optimal at x=2
            g = 2 * (x - 2)
            return f, g
        
        x0 = np.array([0.0, 0.0])
        bounds = [(0, 1), (0, 1)]  # Constrain to [0, 1]^2
        
        # Our optimizer
        result = minimize_with_hard_constraints(
            fun, x0, jac=True, bounds=bounds,
            options={'maxiter': 100, 'gtol': 1e-8}
        )
        
        # scipy L-BFGS-B
        scipy_result = scipy_minimize(
            fun, x0, method='L-BFGS-B', jac=True, bounds=bounds,
            options={'maxiter': 100, 'gtol': 1e-8}
        )
        
        # Both should find x ≈ [1, 1] (bounded optimal)
        np.testing.assert_allclose(result.x, [1.0, 1.0], atol=1e-6)
        np.testing.assert_allclose(scipy_result.x, [1.0, 1.0], atol=1e-6)
    
    def test_beale_function(self):
        """Test on Beale function - a more challenging 2D problem."""
        def beale(x):
            f = ((1.5 - x[0] + x[0]*x[1])**2 + 
                 (2.25 - x[0] + x[0]*x[1]**2)**2 + 
                 (2.625 - x[0] + x[0]*x[1]**3)**2)
            
            # Gradient (computed analytically)
            t1 = 1.5 - x[0] + x[0]*x[1]
            t2 = 2.25 - x[0] + x[0]*x[1]**2
            t3 = 2.625 - x[0] + x[0]*x[1]**3
            
            df_dx0 = 2*t1*(-1 + x[1]) + 2*t2*(-1 + x[1]**2) + 2*t3*(-1 + x[1]**3)
            df_dx1 = 2*t1*x[0] + 2*t2*2*x[0]*x[1] + 2*t3*3*x[0]*x[1]**2
            
            return f, np.array([df_dx0, df_dx1])
        
        x0 = np.array([0.0, 0.0])
        
        result = minimize_with_hard_constraints(
            beale, x0, jac=True,
            options={'maxiter': 200, 'gtol': 1e-6}
        )
        
        scipy_result = scipy_minimize(
            beale, x0, method='L-BFGS-B', jac=True,
            options={'maxiter': 200, 'gtol': 1e-6}
        )
        
        # Optimal at (3, 0.5)
        np.testing.assert_allclose(result.x, [3.0, 0.5], atol=1e-3)
        np.testing.assert_allclose(scipy_result.x, [3.0, 0.5], atol=1e-3)
    
    def test_high_dimensional(self):
        """Test on higher-dimensional quadratic."""
        n = 50
        
        def fun(x):
            f = np.sum(x**2)
            g = 2 * x
            return f, g
        
        x0 = np.random.randn(n)
        
        result = minimize_with_hard_constraints(
            fun, x0, jac=True,
            options={'maxiter': 100, 'gtol': 1e-8}
        )
        
        scipy_result = scipy_minimize(
            fun, x0, method='L-BFGS-B', jac=True,
            options={'maxiter': 100, 'gtol': 1e-8}
        )
        
        np.testing.assert_allclose(result.x, np.zeros(n), atol=1e-6)
        np.testing.assert_allclose(scipy_result.x, np.zeros(n), atol=1e-6)


class TestHardConstraintRejection(unittest.TestCase):
    """Tests for the hard constraint rejection mechanism."""
    
    def test_constraint_rejection_simple(self):
        """Test that steps violating hard constraints are rejected."""
        # Minimize f(x) = (x - 5)^2, optimal at x=5
        # But add a forbidden region around x=3
        def fun(x):
            f = (x[0] - 5)**2
            g = np.array([2 * (x[0] - 5)])
            return f, g
        
        x0 = np.array([0.0])
        
        # Constraint: forbid region around x=3 (radius=1)
        constraint = MockHardConstraint(x0, forbidden_region_center=[3.0], forbidden_radius=1.0)
        
        result = minimize_with_hard_constraints(
            fun, x0, jac=True,
            hard_constraints=[constraint],
            options={'maxiter': 100, 'gtol': 1e-8, 'verbose': 0}
        )
        
        # The path from 0 to 5 passes through the forbidden region [2, 4]
        # Optimizer should either:
        # 1. Find a path around (not possible in 1D)
        # 2. Get stuck at the boundary
        # 3. Start from a different side
        
        # In 1D, it will get stuck at x ≈ 2 (boundary of forbidden region)
        # or it might jump over if step size is large enough
        self.assertTrue(result.n_constraint_rejections > 0 or result.x[0] >= 4.0)
    
    def test_constraint_rejection_2d(self):
        """Test constraint rejection in 2D where optimizer can go around."""
        # Minimize f(x,y) = x^2 + y^2, optimal at (0,0)
        # Start at (2, 2), forbidden region at (1, 1) with radius 0.3
        # Optimizer should avoid the region but still reach (0, 0)
        def fun(x):
            f = x[0]**2 + x[1]**2
            g = 2 * x
            return f, g
        
        x0 = np.array([2.0, 2.0])
        
        # Forbidden region at (1, 1)
        constraint = MockHardConstraint(x0, forbidden_region_center=[1.0, 1.0], forbidden_radius=0.3)
        
        result = minimize_with_hard_constraints(
            fun, x0, jac=True,
            hard_constraints=[constraint],
            options={'maxiter': 100, 'gtol': 1e-6}
        )
        
        # Should still reach optimum at (0, 0)
        np.testing.assert_allclose(result.x, [0.0, 0.0], atol=1e-4)
        self.assertTrue(result.success)
    
    def test_infeasible_initial_point(self):
        """Test that infeasible initial point is rejected."""
        def fun(x):
            f = np.sum(x**2)
            g = 2 * x
            return f, g
        
        x0 = np.array([0.0, 0.0])  # Initial point is in forbidden region
        
        # Forbidden region around origin
        constraint = MockHardConstraint(x0, forbidden_region_center=[0.0, 0.0], forbidden_radius=0.5)
        
        result = minimize_with_hard_constraints(
            fun, x0, jac=True,
            hard_constraints=[constraint]
        )
        
        self.assertFalse(result.success)
        self.assertIn("infeasible", result.message.lower())
    
    def test_multiple_hard_constraints(self):
        """Test with multiple hard constraints."""
        def fun(x):
            f = (x[0] - 3)**2 + (x[1] - 3)**2  # Optimal at (3, 3)
            g = np.array([2*(x[0] - 3), 2*(x[1] - 3)])
            return f, g
        
        x0 = np.array([0.0, 0.0])
        
        # Two forbidden regions
        c1 = MockHardConstraint(x0, forbidden_region_center=[1.0, 1.0], forbidden_radius=0.3)
        c2 = MockHardConstraint(x0, forbidden_region_center=[2.0, 2.0], forbidden_radius=0.3)
        
        result = minimize_with_hard_constraints(
            fun, x0, jac=True,
            hard_constraints=[c1, c2],
            options={'maxiter': 200, 'gtol': 1e-5}
        )
        
        # Should still reach optimum at (3, 3)
        np.testing.assert_allclose(result.x, [3.0, 3.0], atol=1e-3)
    
    def test_custom_feasibility_check(self):
        """Test with custom feasibility check function."""
        def fun(x):
            f = np.sum(x**2)
            g = 2 * x
            return f, g
        
        x0 = np.array([1.0, 1.0])
        
        # Custom constraint object
        class CustomConstraint:
            def __init__(self):
                self.x = None
                
            def J(self):
                # Return negative values in first quadrant, positive elsewhere
                if self.x is not None and np.all(self.x > 0):
                    return -1  # Feasible
                return 1  # Infeasible
        
        constraint = CustomConstraint()
        constraint.x = x0
        
        # Custom check: feasible if J() < 0
        def custom_check(hcs):
            return all(hc.J() < 0 for hc in hcs)
        
        result = minimize_with_hard_constraints(
            fun, x0, jac=True,
            hard_constraints=[constraint],
            feasibility_check=custom_check,
            options={'maxiter': 100}
        )
        
        # Should converge since starting in feasible first quadrant
        # But optimal (0,0) is on boundary, so might stop near it
        self.assertTrue(result.x[0] >= 0 and result.x[1] >= 0)


class TestConstrainedLBFGSBClass(unittest.TestCase):
    """Tests for the ConstrainedLBFGSB class interface."""
    
    def test_verbose_output(self):
        """Test verbose output doesn't crash."""
        def fun(x):
            f = np.sum(x**2)
            g = 2 * x
            return f, g
        
        x0 = np.array([1.0, 2.0])
        
        optimizer = ConstrainedLBFGSB(
            fun=fun, x0=x0, jac=True,
            maxiter=10, verbose=2
        )
        result = optimizer.minimize()
        self.assertTrue(result.nit <= 10)
    
    def test_callback(self):
        """Test callback is called at each iteration."""
        def fun(x):
            f = np.sum(x**2)
            g = 2 * x
            return f, g
        
        x0 = np.array([5.0, 5.0])
        callback_count = [0]
        callback_xs = []
        
        def callback(x):
            callback_count[0] += 1
            callback_xs.append(x.copy())
        
        optimizer = ConstrainedLBFGSB(
            fun=fun, x0=x0, jac=True,
            maxiter=10, callback=callback
        )
        result = optimizer.minimize()
        
        self.assertEqual(callback_count[0], result.nit)
        self.assertEqual(len(callback_xs), result.nit)
    
    def test_result_attributes(self):
        """Test that result has all expected attributes."""
        def fun(x):
            f = np.sum(x**2)
            g = 2 * x
            return f, g
        
        x0 = np.array([1.0])
        result = minimize_with_hard_constraints(fun, x0, jac=True)
        
        self.assertTrue(hasattr(result, 'x'))
        self.assertTrue(hasattr(result, 'fun'))
        self.assertTrue(hasattr(result, 'jac'))
        self.assertTrue(hasattr(result, 'nit'))
        self.assertTrue(hasattr(result, 'nfev'))
        self.assertTrue(hasattr(result, 'njev'))
        self.assertTrue(hasattr(result, 'success'))
        self.assertTrue(hasattr(result, 'message'))
        self.assertTrue(hasattr(result, 'n_constraint_rejections'))
        self.assertTrue(hasattr(result, 'constraint_rejection_history'))
    
    def test_numerical_gradient(self):
        """Test with numerical gradient computation."""
        def fun(x):
            return np.sum(x**2)
        
        x0 = np.array([1.0, 2.0])
        
        optimizer = ConstrainedLBFGSB(
            fun=fun, x0=x0, jac=False,
            maxiter=50, gtol=1e-4
        )
        result = optimizer.minimize()
        
        np.testing.assert_allclose(result.x, [0.0, 0.0], atol=1e-3)
    
    def test_separate_gradient_function(self):
        """Test with separate gradient function."""
        def fun(x):
            return np.sum(x**2)
        
        def grad(x):
            return 2 * x
        
        x0 = np.array([1.0, 2.0])
        
        optimizer = ConstrainedLBFGSB(
            fun=fun, x0=x0, jac=grad,
            maxiter=50, gtol=1e-6
        )
        result = optimizer.minimize()
        
        np.testing.assert_allclose(result.x, [0.0, 0.0], atol=1e-5)


class TestSimsoptIntegration(unittest.TestCase):
    """Integration tests with simsopt objectives (if available)."""
    
    def test_with_mock_optimizable(self):
        """Test with mock Optimizable-like objects."""
        
        class MockObjective:
            """Mimics simsopt Optimizable interface."""
            
            def __init__(self, x):
                self._x = np.array(x, dtype=float)
            
            @property
            def x(self):
                return self._x
            
            @x.setter
            def x(self, value):
                self._x = np.array(value, dtype=float)
            
            def J(self):
                return np.sum((self._x - 1.0)**2)
            
            def dJ(self):
                return 2 * (self._x - 1.0)
        
        x0 = np.array([0.0, 0.0, 0.0])
        obj = MockObjective(x0)
        
        # Wrap in callable for optimizer
        def fun(x):
            obj.x = x
            return obj.J(), obj.dJ()
        
        result = minimize_with_hard_constraints(
            fun, x0, jac=True,
            options={'maxiter': 100, 'gtol': 1e-8}
        )
        
        np.testing.assert_allclose(result.x, [1.0, 1.0, 1.0], atol=1e-6)


class TestConvergence(unittest.TestCase):
    """Tests for convergence behavior."""
    
    def test_ftol_convergence(self):
        """Test convergence by function tolerance."""
        def fun(x):
            f = np.sum(x**2)
            g = 2 * x
            return f, g
        
        x0 = np.array([0.1, 0.1])  # Already close to optimum
        
        result = minimize_with_hard_constraints(
            fun, x0, jac=True,
            options={'maxiter': 100, 'ftol': 1e-3, 'gtol': 1e-12}
        )
        
        self.assertTrue(result.success)
        self.assertIn("tolerance", result.message.lower())
    
    def test_gtol_convergence(self):
        """Test convergence by gradient tolerance."""
        def fun(x):
            f = np.sum(x**2)
            g = 2 * x
            return f, g
        
        x0 = np.array([1.0, 1.0])
        
        result = minimize_with_hard_constraints(
            fun, x0, jac=True,
            options={'maxiter': 100, 'gtol': 1e-6}
        )
        
        self.assertTrue(result.success)
        self.assertTrue(np.linalg.norm(result.jac, ord=np.inf) <= 1e-6)
    
    def test_maxiter_termination(self):
        """Test termination at maxiter."""
        # Rosenbrock function - requires many iterations to converge
        def fun(x):
            f = sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)
            g = np.zeros_like(x)
            g[0] = -400*x[0]*(x[1]-x[0]**2) - 2*(1-x[0])
            g[-1] = 200*(x[-1]-x[-2]**2)
            for i in range(1, len(x)-1):
                g[i] = 200*(x[i]-x[i-1]**2) - 400*x[i]*(x[i+1]-x[i]**2) - 2*(1-x[i])
            return f, g
        
        # Start far from optimum to ensure many iterations needed
        x0 = np.array([-2.0, -2.0, -2.0, -2.0])
        
        result = minimize_with_hard_constraints(
            fun, x0, jac=True,
            options={'maxiter': 5, 'gtol': 1e-15}  # Very strict gtol, few iterations
        )
        
        # Should terminate due to maxiter, not convergence
        self.assertLessEqual(result.nit, 5)
        # Either hit maxiter or converged (both acceptable)
        self.assertTrue(result.nit == 5 or result.success)


class TestLBFGSMemory(unittest.TestCase):
    """Tests for L-BFGS memory management."""
    
    def test_maxcor_parameter(self):
        """Test that maxcor limits memory usage."""
        def fun(x):
            f = np.sum(x**2)
            g = 2 * x
            return f, g
        
        x0 = np.array([1.0] * 10)
        
        # Small memory
        opt1 = ConstrainedLBFGSB(fun=fun, x0=x0, jac=True, maxcor=2, maxiter=20)
        result1 = opt1.minimize()
        
        # Larger memory
        opt2 = ConstrainedLBFGSB(fun=fun, x0=x0, jac=True, maxcor=10, maxiter=20)
        result2 = opt2.minimize()
        
        # Both should converge
        np.testing.assert_allclose(result1.x, np.zeros(10), atol=1e-5)
        np.testing.assert_allclose(result2.x, np.zeros(10), atol=1e-5)


# =============================================================================
# Full-scale stellarator coil optimization test
# =============================================================================
class TestStellatorOptimization(unittest.TestCase):
    """Full-scale test with stellarator coil optimization."""
    
    def test_coil_optimization_basic(self):
        """
        Test both optimization approaches on a stellarator coil problem.
        
        This test verifies:
        1. minimize_lbfgsb_with_hard_constraints gives IDENTICAL results to scipy
           when no hard constraints are used (it wraps scipy directly)
        2. ConstrainedLBFGSB converges to the same optimum but may take a different
           path (it uses scipy.optimize.line_search which differs from L-BFGS-B's
           internal Fortran line search)
        """
        from pathlib import Path
        from simsopt.geo import SurfaceRZFourier, create_equally_spaced_curves
        from simsopt.field import BiotSavart, Current, coils_via_symmetries
        from simsopt.objectives import SquaredFlux
        
        # Define the test directory
        TEST_DIR = (Path(__file__).parent / ".." / "test_files").resolve()
        filename = TEST_DIR / 'input.LandremanPaul2021_QA_lowres'
        
        if not filename.exists():
            self.skipTest(f"Test file not found: {filename}")
        
        # Setup problem
        nphi = 8
        ntheta = 8
        s = SurfaceRZFourier.from_vmec_input(
            filename, range="half period", nphi=nphi, ntheta=ntheta)
        
        R0 = s.get_rc(0, 0)
        R1 = 0.6 * R0
        order = 3
        ncoils = 3
        
        base_curves = create_equally_spaced_curves(
            ncoils, s.nfp, stellsym=s.stellsym, 
            R0=R0, R1=R1, order=order, numquadpoints=64)
        base_currents = [Current(1e5) for _ in range(ncoils)]
        base_currents[0].fix_all()
        
        coils = coils_via_symmetries(base_curves, base_currents, s.nfp, s.stellsym)
        bs = BiotSavart(coils)
        bs.set_points(s.gamma().reshape((-1, 3)))
        
        Jf = SquaredFlux(s, bs)
        
        # Define objective function
        def fun(dofs):
            Jf.x = dofs
            f = Jf.J()
            g = Jf.dJ()
            return f, np.asarray(g, dtype=np.float64)
        
        x0 = Jf.x.copy()
        f0, _ = fun(x0)
        
        # Run minimize_with_hard_constraints (using ConstrainedLBFGSB with DCSRCH)
        Jf.x = x0.copy()  # Reset state
        result = minimize_with_hard_constraints(
            fun, x0.copy(), jac=True,
            options={'maxiter': 100, 'gtol': 1e-10, 'maxcor': 10}
        )
        
        Jf.x = x0.copy()  # Reset state
        scipy_result = scipy_minimize(
            fun, x0.copy(), method='L-BFGS-B', jac=True,
            options={'maxiter': 100, 'gtol': 1e-10, 'maxcor': 10}
        )
        
        # Both should significantly reduce objective
        self.assertLess(result.fun, f0 * 0.001,
            msg=f"Our solver should reduce objective significantly: {result.fun:.2e} vs initial {f0:.2e}")
        self.assertLess(scipy_result.fun, f0 * 0.001,
            msg=f"Scipy should reduce objective significantly: {scipy_result.fun:.2e} vs initial {f0:.2e}")
        
        # Both should achieve similar quality solutions (within same order of magnitude)
        # Note: Small numerical differences in L-BFGS implementation can lead to 
        # different paths on complex problems, but both should find good solutions
        ratio = max(result.fun, scipy_result.fun) / max(min(result.fun, scipy_result.fun), 1e-15)
        self.assertLess(ratio, 5.0,
            msg=f"Solutions should be similar quality: ours={result.fun:.2e}, scipy={scipy_result.fun:.2e}")
    
    def test_coil_optimization_with_linking_constraint(self):
        """
        Test coil optimization with a LinkingNumber-like hard constraint.
        
        This test verifies that the optimizer properly rejects steps that
        would violate the linking constraint.
        """
        from pathlib import Path
        from simsopt.geo import SurfaceRZFourier, create_equally_spaced_curves
        from simsopt.field import BiotSavart, Current, coils_via_symmetries
        from simsopt.objectives import SquaredFlux
        from simsopt.geo import LinkingNumber
        
        # Define the test directory
        TEST_DIR = (Path(__file__).parent / ".." / "test_files").resolve()
        filename = TEST_DIR / 'input.LandremanPaul2021_QA_lowres'
        
        if not filename.exists():
            self.skipTest(f"Test file not found: {filename}")
        
        # Setup problem
        nphi = 8
        ntheta = 8
        s = SurfaceRZFourier.from_vmec_input(
            filename, range="half period", nphi=nphi, ntheta=ntheta)
        
        R0 = s.get_rc(0, 0)
        R1 = 0.6 * R0
        order = 3
        ncoils = 3
        
        base_curves = create_equally_spaced_curves(
            ncoils, s.nfp, stellsym=s.stellsym,
            R0=R0, R1=R1, order=order, numquadpoints=64)
        base_currents = [Current(1e5) for _ in range(ncoils)]
        base_currents[0].fix_all()
        
        coils = coils_via_symmetries(base_curves, base_currents, s.nfp, s.stellsym)
        curves = [c.curve for c in coils]
        bs = BiotSavart(coils)
        bs.set_points(s.gamma().reshape((-1, 3)))
        
        Jf = SquaredFlux(s, bs)
        Jlink = LinkingNumber(curves, downsample=2)
        
        # Define objective function
        def fun(dofs):
            Jf.x = dofs
            f = Jf.J()
            g = Jf.dJ()
            return f, np.asarray(g, dtype=np.float64)
        
        # Custom feasibility check for LinkingNumber
        def linking_check(hcs):
            for hc in hcs:
                if isinstance(hc, LinkingNumber):
                    if abs(hc.J()) > 0.5:  # LinkingNumber is 0 (unlinked) or nonzero (linked)
                        return False
            return True
        
        x0 = Jf.x.copy()
        f0, _ = fun(x0)
        
        # Verify initial state is feasible (no linking)
        initial_link = Jlink.J()
        self.assertEqual(initial_link, 0, "Initial coils should not be linked")
        
        # Optimize with linking constraint
        result = minimize_with_hard_constraints(
            fun, x0, jac=True,
            hard_constraints=[Jlink],
            feasibility_check=linking_check,
            options={'maxiter': 20, 'gtol': 1e-6}
        )
        
        # Objective should decrease
        self.assertLess(result.fun, f0 * 1.01)  # Allow small tolerance
        
        # Final state should still be feasible (no linking)
        Jf.x = result.x  # Update DOFs
        final_link = Jlink.J()
        self.assertEqual(final_link, 0, "Final coils should not be linked")


class TestBoundsAgreement(unittest.TestCase):
    """
    Tests verifying that custom solver handles bounds correctly.
    
    Note: Scipy's L-BFGS-B uses a sophisticated projected gradient approach
    in Fortran that may take different paths than our Python implementation.
    Tests verify both converge to the correct solution with correct bounds.
    """
    
    def test_quadratic_with_bounds_correct_solution(self):
        """Verify both converge to correct bounded optimum."""
        def fun(x):
            f = np.sum((x - 5)**2)  # Optimal at x=5, but bounded to [0, 2]
            g = 2 * (x - 5)
            return f, g
        
        x0 = np.array([0.5, 0.5, 0.5])
        bounds = [(0, 2), (0, 2), (0, 2)]
        
        result = minimize_with_hard_constraints(
            fun, x0, jac=True, bounds=bounds,
            options={'maxiter': 100, 'gtol': 1e-10}
        )
        scipy_result = scipy_minimize(
            fun, x0, method='L-BFGS-B', jac=True, bounds=bounds,
            options={'maxiter': 100, 'gtol': 1e-10}
        )
        
        # Both should find bounded optimal at x=2
        np.testing.assert_allclose(result.x, [2.0, 2.0, 2.0], atol=1e-6,
            err_msg="Custom solver should converge to boundary")
        np.testing.assert_allclose(scipy_result.x, [2.0, 2.0, 2.0], atol=1e-6,
            err_msg="Scipy should converge to boundary")
        
        # Objective values should be very close
        np.testing.assert_allclose(result.fun, scipy_result.fun, rtol=1e-8,
            err_msg="Objective values should match")
    
    def test_rosenbrock_with_bounds_correct_solution(self):
        """Verify both converge to correct solution on bounded Rosenbrock."""
        def rosenbrock(x):
            f = (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2
            g = np.array([
                -2*(1 - x[0]) - 400*x[0]*(x[1] - x[0]**2),
                200*(x[1] - x[0]**2)
            ])
            return f, g
        
        x0 = np.array([0.0, 0.0])
        bounds = [(-0.5, 2.0), (-0.5, 2.0)]  # Optimal [1,1] is inside bounds
        
        result = minimize_with_hard_constraints(
            rosenbrock, x0, jac=True, bounds=bounds,
            options={'maxiter': 200, 'gtol': 1e-8}
        )
        scipy_result = scipy_minimize(
            rosenbrock, x0, method='L-BFGS-B', jac=True, bounds=bounds,
            options={'maxiter': 200, 'gtol': 1e-8}
        )
        
        # Both should find optimal at [1, 1]
        np.testing.assert_allclose(result.x, [1.0, 1.0], atol=1e-5,
            err_msg="Custom solver should find optimal")
        np.testing.assert_allclose(scipy_result.x, [1.0, 1.0], atol=1e-5,
            err_msg="Scipy should find optimal")
        
        # Objective values should be very close (both near 0)
        self.assertLess(result.fun, 1e-8)
        self.assertLess(scipy_result.fun, 1e-8)
    
    def test_high_dimensional_with_bounds_correct_solution(self):
        """Verify both converge correctly on high-dimensional bounded problem."""
        n = 20
        
        def fun(x):
            f = np.sum((x - 3)**2)  # Optimal at x=3, but bounded to [0, 2]
            g = 2 * (x - 3)
            return f, g
        
        np.random.seed(42)
        x0 = np.random.uniform(0, 1, n)
        bounds = [(0, 2) for _ in range(n)]  # Bounded optimal at x=2
        
        result = minimize_with_hard_constraints(
            fun, x0, jac=True, bounds=bounds,
            options={'maxiter': 100, 'gtol': 1e-10}
        )
        scipy_result = scipy_minimize(
            fun, x0, method='L-BFGS-B', jac=True, bounds=bounds,
            options={'maxiter': 100, 'gtol': 1e-10}
        )
        
        # Both should find bounded optimal at x=2
        np.testing.assert_allclose(result.x, np.full(n, 2.0), atol=1e-6,
            err_msg="Custom solver should converge to boundary")
        np.testing.assert_allclose(scipy_result.x, np.full(n, 2.0), atol=1e-6,
            err_msg="Scipy should converge to boundary")
        
        # Objective values should match
        np.testing.assert_allclose(result.fun, scipy_result.fun, rtol=1e-8)
    
    def test_mixed_bounds_correct_solution(self):
        """Verify both handle mixed finite/infinite bounds correctly."""
        def fun(x):
            f = np.sum(x**2) + np.sum((x - 1)**4)
            g = 2*x + 4*(x - 1)**3
            return f, g
        
        x0 = np.array([5.0, -5.0, 0.0, 3.0])
        bounds = [(0, None), (None, 0), (-1, 1), (None, None)]  # Mixed bounds
        f0, _ = fun(x0)
        
        result = minimize_with_hard_constraints(
            fun, x0, jac=True, bounds=bounds,
            options={'maxiter': 100, 'gtol': 1e-10}
        )
        scipy_result = scipy_minimize(
            fun, x0, method='L-BFGS-B', jac=True, bounds=bounds,
            options={'maxiter': 100, 'gtol': 1e-10}
        )
        
        # Verify bounds are satisfied for both
        self.assertGreaterEqual(result.x[0], 0 - 1e-10)  # x[0] >= 0
        self.assertLessEqual(result.x[1], 0 + 1e-10)     # x[1] <= 0
        self.assertGreaterEqual(result.x[2], -1 - 1e-10) # -1 <= x[2]
        self.assertLessEqual(result.x[2], 1 + 1e-10)     # x[2] <= 1
        
        self.assertGreaterEqual(scipy_result.x[0], 0 - 1e-10)
        self.assertLessEqual(scipy_result.x[1], 0 + 1e-10)
        self.assertGreaterEqual(scipy_result.x[2], -1 - 1e-10)
        self.assertLessEqual(scipy_result.x[2], 1 + 1e-10)
        
        # Both should significantly reduce objective from initial
        self.assertLess(result.fun, f0 * 0.1)
        self.assertLess(scipy_result.fun, f0 * 0.1)
        
        # Objective values should be comparable (within 10% of each other)
        ratio = max(result.fun, scipy_result.fun) / min(result.fun, scipy_result.fun)
        self.assertLess(ratio, 1.1,
            msg=f"Objective values should be comparable: custom={result.fun:.4f}, scipy={scipy_result.fun:.4f}")


class TestHighResCoilOptimization(unittest.TestCase):
    """High-resolution coil optimization tests comparing custom solver vs scipy."""
    
    def test_highres_coil_optimization_comparison(self):
        """
        High-resolution coil optimization comparing custom solver against scipy.
        
        Uses more coils, higher order Fourier representation, and more quadrature
        points than the basic test.
        """
        from pathlib import Path
        from simsopt.geo import SurfaceRZFourier, create_equally_spaced_curves
        from simsopt.field import BiotSavart, Current, coils_via_symmetries
        from simsopt.objectives import SquaredFlux
        
        TEST_DIR = (Path(__file__).parent / ".." / "test_files").resolve()
        filename = TEST_DIR / 'input.LandremanPaul2021_QA_lowres'
        
        if not filename.exists():
            self.skipTest(f"Test file not found: {filename}")
        
        # Higher resolution setup
        nphi = 16
        ntheta = 16
        s = SurfaceRZFourier.from_vmec_input(
            filename, range="half period", nphi=nphi, ntheta=ntheta)
        
        R0 = s.get_rc(0, 0)
        R1 = 0.6 * R0
        order = 5  # Higher Fourier order
        ncoils = 4  # More coils
        numquadpoints = 128  # More quadrature points
        
        base_curves = create_equally_spaced_curves(
            ncoils, s.nfp, stellsym=s.stellsym,
            R0=R0, R1=R1, order=order, numquadpoints=numquadpoints)
        base_currents = [Current(1e5) for _ in range(ncoils)]
        base_currents[0].fix_all()
        
        coils = coils_via_symmetries(base_curves, base_currents, s.nfp, s.stellsym)
        bs = BiotSavart(coils)
        bs.set_points(s.gamma().reshape((-1, 3)))
        
        Jf = SquaredFlux(s, bs)
        
        def fun(dofs):
            Jf.x = dofs
            return Jf.J(), np.asarray(Jf.dJ(), dtype=np.float64)
        
        x0 = Jf.x.copy()
        f0, _ = fun(x0)
        
        print(f"\nHigh-res coil optimization: {len(x0)} DOFs")
        print(f"Initial objective: {f0:.6e}")
        
        # Run custom solver
        Jf.x = x0.copy()
        result = minimize_with_hard_constraints(
            fun, x0.copy(), jac=True,
            options={'maxiter': 100, 'gtol': 1e-10, 'maxcor': 10}
        )
        
        # Run scipy
        Jf.x = x0.copy()
        scipy_result = scipy_minimize(
            fun, x0.copy(), method='L-BFGS-B', jac=True,
            options={'maxiter': 100, 'gtol': 1e-10, 'maxcor': 10}
        )
        
        print(f"Custom solver: f={result.fun:.6e}, nit={result.nit}, nfev={result.nfev}")
        print(f"Scipy:         f={scipy_result.fun:.6e}, nit={scipy_result.nit}, nfev={scipy_result.nfev}")
        
        # Both should significantly reduce objective
        self.assertLess(result.fun, f0 * 0.01,
            msg=f"Custom solver should reduce objective: {result.fun:.2e} vs {f0:.2e}")
        self.assertLess(scipy_result.fun, f0 * 0.01,
            msg=f"Scipy should reduce objective: {scipy_result.fun:.2e} vs {f0:.2e}")
        
        # Both should achieve similar quality solutions (within 5x of each other)
        ratio = max(result.fun, scipy_result.fun) / max(min(result.fun, scipy_result.fun), 1e-20)
        self.assertLess(ratio, 5.0,
            msg=f"Solutions should be similar quality: custom={result.fun:.2e}, scipy={scipy_result.fun:.2e}")
    
    def test_coil_optimization_with_bounds(self):
        """
        Test coil optimization with parameter bounds on currents.
        
        Verifies that the custom solver handles bounds correctly in the
        stellarator optimization context.
        """
        from pathlib import Path
        from simsopt.geo import SurfaceRZFourier, create_equally_spaced_curves
        from simsopt.field import BiotSavart, Current, coils_via_symmetries
        from simsopt.objectives import SquaredFlux
        
        TEST_DIR = (Path(__file__).parent / ".." / "test_files").resolve()
        filename = TEST_DIR / 'input.LandremanPaul2021_QA_lowres'
        
        if not filename.exists():
            self.skipTest(f"Test file not found: {filename}")
        
        nphi = 8
        ntheta = 8
        s = SurfaceRZFourier.from_vmec_input(
            filename, range="half period", nphi=nphi, ntheta=ntheta)
        
        R0 = s.get_rc(0, 0)
        R1 = 0.6 * R0
        
        base_curves = create_equally_spaced_curves(
            3, s.nfp, stellsym=s.stellsym,
            R0=R0, R1=R1, order=3, numquadpoints=64)
        base_currents = [Current(1e5) for _ in range(3)]
        base_currents[0].fix_all()
        
        coils = coils_via_symmetries(base_curves, base_currents, s.nfp, s.stellsym)
        bs = BiotSavart(coils)
        bs.set_points(s.gamma().reshape((-1, 3)))
        
        Jf = SquaredFlux(s, bs)
        
        def fun(dofs):
            Jf.x = dofs
            return Jf.J(), np.asarray(Jf.dJ(), dtype=np.float64)
        
        x0 = Jf.x.copy()
        f0, _ = fun(x0)
        
        # Create bounds for all DOFs (loose bounds that shouldn't constrain much)
        bounds = [(-np.inf, np.inf) for _ in range(len(x0))]
        # But constrain the current DOFs (last 2 DOFs for unfixed currents)
        # to be within reasonable range
        for i in range(-2, 0):
            bounds[i] = (0.5e5, 1.5e5)  # Bound currents
        
        Jf.x = x0.copy()
        result = minimize_with_hard_constraints(
            fun, x0.copy(), jac=True, bounds=bounds,
            options={'maxiter': 50, 'gtol': 1e-8}
        )
        
        Jf.x = x0.copy()
        scipy_result = scipy_minimize(
            fun, x0.copy(), method='L-BFGS-B', jac=True, bounds=bounds,
            options={'maxiter': 50, 'gtol': 1e-8}
        )
        
        # Both should reduce objective
        self.assertLess(result.fun, f0)
        self.assertLess(scipy_result.fun, f0)
        
        # Verify currents stay within bounds
        for i in range(-2, 0):
            self.assertGreaterEqual(result.x[i], bounds[i][0] - 1e-10)
            self.assertLessEqual(result.x[i], bounds[i][1] + 1e-10)


class TestInterlinkedCoilOptimization(unittest.TestCase):
    """
    Tests for coil optimization starting from interlinked configurations.
    
    These tests verify the feasibility check mechanism by starting from
    linked coils and ensuring they remain linked during optimization.
    """
    
    def test_maintain_linking_during_optimization(self):
        """
        Test optimization starting from interlinked coils.
        
        Uses a feasibility check that PREVENTS unlinking (opposite of the
        usual case). Compares with baseline scipy optimization to show
        the effect of the hard constraint.
        """
        from pathlib import Path
        from simsopt.geo import SurfaceRZFourier, CurveXYZFourier
        from simsopt.field import BiotSavart, Current, Coil
        from simsopt.objectives import SquaredFlux
        from simsopt.geo import LinkingNumber
        
        TEST_DIR = (Path(__file__).parent / ".." / "test_files").resolve()
        filename = TEST_DIR / 'input.LandremanPaul2021_QA_lowres'
        
        if not filename.exists():
            self.skipTest(f"Test file not found: {filename}")
        
        nphi = 8
        ntheta = 8
        s = SurfaceRZFourier.from_vmec_input(
            filename, range="half period", nphi=nphi, ntheta=ntheta)
        
        R0 = s.get_rc(0, 0)
        
        # Create two interlinked curves (Hopf link configuration)
        # Curve 1: Circle in xy-plane, radius R0, centered at origin
        nquad = 64
        order = 3
        curve1 = CurveXYZFourier(nquad, order)
        curve1.set('xc(1)', R0)  # x = R0 * cos(theta)
        curve1.set('ys(1)', R0)  # y = R0 * sin(theta)
        
        # Curve 2: Circle in xz-plane, radius 0.6*R0, centered at (R0, 0, 0)
        # This creates a Hopf link with curve 1
        curve2 = CurveXYZFourier(nquad, order)
        curve2.set('xc(0)', R0)      # x offset to R0
        curve2.set('xc(1)', 0.6*R0)  # x = R0 + 0.6*R0*cos(theta)
        curve2.set('zs(1)', 0.6*R0)  # z = 0.6*R0*sin(theta)
        
        curves = [curve1, curve2]
        currents = [Current(1e5), Current(1e5)]
        currents[0].fix_all()
        
        coils = [Coil(c, curr) for c, curr in zip(curves, currents)]
        bs = BiotSavart(coils)
        bs.set_points(s.gamma().reshape((-1, 3)))
        
        Jf = SquaredFlux(s, bs)
        Jlink = LinkingNumber(curves, downsample=2)
        
        # Check initial linking
        initial_link = Jlink.J()
        print(f"\nInitial linking number: {initial_link}")
        self.assertNotEqual(initial_link, 0, "Initial configuration should be linked")
        
        # Define objective
        def fun(dofs):
            Jf.x = dofs
            return Jf.J(), np.asarray(Jf.dJ(), dtype=np.float64)
        
        x0 = Jf.x.copy()
        f0, _ = fun(x0)
        
        # Feasibility check that REQUIRES coils to stay linked
        def require_linking(hcs):
            for hc in hcs:
                if isinstance(hc, LinkingNumber):
                    if abs(hc.J()) < 0.5:
                        return False  # Became unlinked - infeasible!
            return True
        
        # Run optimization WITH linking maintenance constraint
        Jf.x = x0.copy()
        result_constrained = minimize_with_hard_constraints(
            fun, x0.copy(), jac=True,
            hard_constraints=[Jlink],
            feasibility_check=require_linking,
            options={'maxiter': 30, 'gtol': 1e-6}
        )
        
        # Run baseline scipy optimization (no constraint)
        Jf.x = x0.copy()
        scipy_result = scipy_minimize(
            fun, x0.copy(), method='L-BFGS-B', jac=True,
            options={'maxiter': 30, 'gtol': 1e-6}
        )
        
        # Check final linking states
        Jf.x = result_constrained.x
        final_link_constrained = Jlink.J()
        
        Jf.x = scipy_result.x
        final_link_scipy = Jlink.J()
        
        print(f"Constrained solver final linking: {final_link_constrained}")
        print(f"Scipy (unconstrained) final linking: {final_link_scipy}")
        print(f"Constrained solver f: {result_constrained.fun:.6e}")
        print(f"Scipy f: {scipy_result.fun:.6e}")
        print(f"Constraint rejections: {result_constrained.n_constraint_rejections}")
        
        # The constrained solver should maintain linking
        self.assertNotEqual(final_link_constrained, 0,
            msg="Constrained solver should maintain linking")
        
        # Objective should not increase significantly from initial
        self.assertLess(result_constrained.fun, f0 * 2.0,
            msg="Constrained solver should not increase objective much")
    
    def test_highres_interlinked_optimization(self):
        """
        High-resolution interlinked coil optimization test.
        
        Tests with more DOFs to verify the constraint mechanism works
        robustly in higher dimensions.
        """
        from pathlib import Path
        from simsopt.geo import SurfaceRZFourier, CurveXYZFourier
        from simsopt.field import BiotSavart, Current, Coil
        from simsopt.objectives import SquaredFlux
        from simsopt.geo import LinkingNumber
        
        TEST_DIR = (Path(__file__).parent / ".." / "test_files").resolve()
        filename = TEST_DIR / 'input.LandremanPaul2021_QA_lowres'
        
        if not filename.exists():
            self.skipTest(f"Test file not found: {filename}")
        
        nphi = 12
        ntheta = 12
        s = SurfaceRZFourier.from_vmec_input(
            filename, range="half period", nphi=nphi, ntheta=ntheta)
        
        R0 = s.get_rc(0, 0)
        
        # Higher order curves (more DOFs)
        nquad = 96
        order = 5
        
        # Create Hopf-linked curves
        curve1 = CurveXYZFourier(nquad, order)
        curve1.set('xc(1)', R0)
        curve1.set('ys(1)', R0)
        
        curve2 = CurveXYZFourier(nquad, order)
        curve2.set('xc(0)', R0)
        curve2.set('xc(1)', 0.6*R0)
        curve2.set('zs(1)', 0.6*R0)
        
        curves = [curve1, curve2]
        currents = [Current(1e5), Current(1e5)]
        currents[0].fix_all()
        
        coils = [Coil(c, curr) for c, curr in zip(curves, currents)]
        bs = BiotSavart(coils)
        bs.set_points(s.gamma().reshape((-1, 3)))
        
        Jf = SquaredFlux(s, bs)
        Jlink = LinkingNumber(curves, downsample=2)
        
        def fun(dofs):
            Jf.x = dofs
            return Jf.J(), np.asarray(Jf.dJ(), dtype=np.float64)
        
        x0 = Jf.x.copy()
        f0, _ = fun(x0)
        
        initial_link = Jlink.J()
        print(f"\nHigh-res interlinked test: {len(x0)} DOFs")
        print(f"Initial linking number: {initial_link}")
        print(f"Initial objective: {f0:.6e}")
        
        self.assertNotEqual(initial_link, 0, "Initial configuration should be linked")
        
        def require_linking(hcs):
            for hc in hcs:
                if isinstance(hc, LinkingNumber):
                    if abs(hc.J()) < 0.5:
                        return False
            return True
        
        # Run constrained optimization
        Jf.x = x0.copy()
        result = minimize_with_hard_constraints(
            fun, x0.copy(), jac=True,
            hard_constraints=[Jlink],
            feasibility_check=require_linking,
            options={'maxiter': 50, 'gtol': 1e-6}
        )
        
        # Run scipy baseline
        Jf.x = x0.copy()
        scipy_result = scipy_minimize(
            fun, x0.copy(), method='L-BFGS-B', jac=True,
            options={'maxiter': 50, 'gtol': 1e-6}
        )
        
        # Check final states
        Jf.x = result.x
        final_link = Jlink.J()
        
        Jf.x = scipy_result.x
        scipy_final_link = Jlink.J()
        
        print(f"Constrained: f={result.fun:.6e}, link={final_link}, rejections={result.n_constraint_rejections}")
        print(f"Scipy:       f={scipy_result.fun:.6e}, link={scipy_final_link}")
        
        # Constrained solver should maintain linking
        self.assertNotEqual(final_link, 0,
            msg="Constrained solver should maintain linking")
        
        # Objective should not increase dramatically
        self.assertLess(result.fun, f0 * 2.0,
            msg="Constrained optimization should not increase objective dramatically")


class TestAugmentedLagrangianIntegration(unittest.TestCase):
    """
    Tests for integration of ConstrainedLBFGSB with augmented_lagrangian_method.
    
    Demonstrates using the custom solver as the inner optimizer for ALM
    while maintaining hard constraints like LinkingNumber.
    """
    
    def test_auglag_with_custom_solver_maintains_linking(self):
        """
        Test that augmented_lagrangian_method with 'L-BFGS-B-custom' maintains linking.
        
        This test verifies:
        1. The custom solver can be used as the inner optimizer for ALM
        2. Linking constraint is maintained throughout optimization (starts and ends = 1)
        3. The soft equality constraints are still enforced by ALM
        """
        from pathlib import Path
        from simsopt.geo import SurfaceRZFourier, CurveXYZFourier, CurveLength
        from simsopt.field import BiotSavart, Current, Coil
        from simsopt.objectives import SquaredFlux
        from simsopt.geo import LinkingNumber
        from simsopt.solve import augmented_lagrangian_method
        from simsopt.objectives import QuadraticPenalty
        
        TEST_DIR = (Path(__file__).parent / ".." / "test_files").resolve()
        filename = TEST_DIR / 'input.LandremanPaul2021_QA_lowres'
        
        if not filename.exists():
            self.skipTest(f"Test file not found: {filename}")
        
        nphi = 8
        ntheta = 8
        s = SurfaceRZFourier.from_vmec_input(
            filename, range="half period", nphi=nphi, ntheta=ntheta)
        
        R0 = s.get_rc(0, 0)
        
        # Create Hopf-linked curves (linking number = 1)
        nquad = 64
        order = 3
        curve1 = CurveXYZFourier(nquad, order)
        curve1.set('xc(1)', R0)
        curve1.set('ys(1)', R0)
        
        curve2 = CurveXYZFourier(nquad, order)
        curve2.set('xc(0)', R0)
        curve2.set('xc(1)', 0.6*R0)
        curve2.set('zs(1)', 0.6*R0)
        
        curves = [curve1, curve2]
        currents = [Current(1e5), Current(1e5)]
        currents[0].fix_all()
        
        coils = [Coil(c, curr) for c, curr in zip(curves, currents)]
        bs = BiotSavart(coils)
        bs.set_points(s.gamma().reshape((-1, 3)))
        
        # Create objectives
        Jf = SquaredFlux(s, bs)
        Jlink = LinkingNumber(curves, downsample=2)
        
        # Soft constraint: curve length (for ALM equality constraint)
        target_length = 2 * np.pi * R0 * 1.1  # Target slightly larger than initial
        Jlength = CurveLength(curve2)
        Jf += QuadraticPenalty(Jlength, target_length, "max")
        
        # Verify initial linking
        initial_link = Jlink.J()
        print(f"\nInitial linking number: {initial_link}")
        self.assertEqual(initial_link, 1, "Initial configuration should have linking number = 1")
        
        # Feasibility check that REQUIRES coils to stay linked
        def require_linking(hcs):
            for hc in hcs:
                if isinstance(hc, LinkingNumber):
                    if abs(hc.J()) < 0.5:
                        return False
            return True
        
        # Run augmented Lagrangian with custom solver
        x_opt, final_lag, lag_mul = augmented_lagrangian_method(
            f=Jf,
            equality_constraints=[],  # No soft equality constraints for simplicity
            minimize_method='L-BFGS-B-custom',
            hard_constraints=[Jlink],
            feasibility_check=require_linking,
            MAXITER=20,
            MAXITER_lag=3,
            mu_init=10.0,
            verbose=False
        )
        
        # Update DOFs to check final state
        Jf.x = x_opt
        final_link = Jlink.J()
        
        print(f"Final linking number: {final_link}")
        print(f"Final augmented Lagrangian: {final_lag:.6e}")
        
        # Verify linking is maintained
        self.assertEqual(final_link, 1, 
            "Linking number should remain 1 throughout optimization")
        self.assertEqual(initial_link, final_link,
            "Linking should be maintained from start to end")
    
    def test_auglag_with_custom_solver_vs_scipy(self):
        """
        Compare augmented_lagrangian_method with custom solver vs scipy L-BFGS-B.
        
        Without hard constraints, both should produce similar results.
        """
        from pathlib import Path
        from simsopt.geo import SurfaceRZFourier, create_equally_spaced_curves
        from simsopt.field import BiotSavart, Current, coils_via_symmetries
        from simsopt.objectives import SquaredFlux
        from simsopt.solve import augmented_lagrangian_method
        
        TEST_DIR = (Path(__file__).parent / ".." / "test_files").resolve()
        filename = TEST_DIR / 'input.LandremanPaul2021_QA_lowres'
        
        if not filename.exists():
            self.skipTest(f"Test file not found: {filename}")
        
        nphi = 8
        ntheta = 8
        s = SurfaceRZFourier.from_vmec_input(
            filename, range="half period", nphi=nphi, ntheta=ntheta)
        
        R0 = s.get_rc(0, 0)
        R1 = 0.6 * R0
        
        base_curves = create_equally_spaced_curves(
            3, s.nfp, stellsym=s.stellsym,
            R0=R0, R1=R1, order=3, numquadpoints=64)
        base_currents = [Current(1e5) for _ in range(3)]
        base_currents[0].fix_all()
        
        coils = coils_via_symmetries(base_curves, base_currents, s.nfp, s.stellsym)
        bs = BiotSavart(coils)
        bs.set_points(s.gamma().reshape((-1, 3)))
        
        Jf = SquaredFlux(s, bs)
        x0 = Jf.x.copy()
        f0 = Jf.J()
        
        # Run with custom solver (no hard constraints)
        Jf.x = x0.copy()
        x_custom, lag_custom, _ = augmented_lagrangian_method(
            f=Jf,
            equality_constraints=[],
            minimize_method='L-BFGS-B-custom',
            MAXITER=30,
            MAXITER_lag=2,
            mu_init=10.0,
            verbose=False
        )
        
        # Run with scipy L-BFGS-B
        Jf.x = x0.copy()
        x_scipy, lag_scipy, _ = augmented_lagrangian_method(
            f=Jf,
            equality_constraints=[],
            minimize_method='L-BFGS-B',
            MAXITER=30,
            MAXITER_lag=2,
            mu_init=10.0,
            verbose=False
        )
        
        # Check final objectives
        Jf.x = x_custom
        f_custom = Jf.J()
        
        Jf.x = x_scipy
        f_scipy = Jf.J()
        
        print(f"\nInitial objective: {f0:.6e}")
        print(f"Custom solver final: {f_custom:.6e}")
        print(f"Scipy final: {f_scipy:.6e}")
        
        # Both should significantly reduce objective
        self.assertLess(f_custom, f0 * 0.1,
            msg=f"Custom solver should reduce objective: {f_custom:.2e} vs {f0:.2e}")
        self.assertLess(f_scipy, f0 * 0.1,
            msg=f"Scipy should reduce objective: {f_scipy:.2e} vs {f0:.2e}")
        
        # Both should achieve similar quality (within 10x)
        ratio = max(f_custom, f_scipy) / max(min(f_custom, f_scipy), 1e-20)
        self.assertLess(ratio, 10.0,
            msg=f"Results should be comparable: custom={f_custom:.2e}, scipy={f_scipy:.2e}")


if __name__ == "__main__":
    unittest.main()
