"""
Unit tests for src/simsopt/solve/augmented_lagrangian.py.

These tests cover the core functionality of the Augmented Lagrangian implementation, including Jacobian computation, objective and gradient evaluation, progress indication, and the main optimization loop. Mock objects are used for objectives and constraints to ensure fast, isolated tests.
"""
import numpy as np
import unittest
from simsopt.solve import augmented_lagrangian as al
from monty.tempfile import ScratchDir

class MockObjective:
    def __init__(self, x):
        self.x = np.array(x, dtype=float)
    def J(self):
        return np.sum(self.x ** 2)
    def dJ(self):
        return 2 * self.x

class MockConstraint:
    def __init__(self, x, offset=0.0):
        self.x = np.array(x, dtype=float)
        self.offset = offset
    def J(self):
        return np.sum(self.x) + self.offset
    def dJ(self):
        return np.ones_like(self.x)

class ALTests(unittest.TestCase):

    def test_jac_constraint(self):
        """
        Test that jac_constraint returns the correct Jacobian matrix for a list of constraints.
        Each constraint returns a gradient of ones, so the Jacobian should be all ones.
        """
        dofs = np.array([1.0, 2.0, 3.0])
        c1 = MockConstraint([1.0, 2.0, 3.0])
        c2 = MockConstraint([4.0, 5.0, 6.0])
        J = al.jac_constraint([c1, c2], dofs)
        assert J.shape == (2, 3)
        np.testing.assert_array_equal(J[0], np.ones(3), err_msg="J[0] should be all ones")
        np.testing.assert_array_equal(J[1], np.ones(3), err_msg="J[1] should be all ones")


    def test_augmented_lagrangian_objective_standard(self):
        """
        Test the standard form of the augmented Lagrangian objective.
        Checks that the value matches the formula:
            L = f(x) - lag_mul * g(x) + mu/2 * ||g(x)||^2
        for a quadratic objective and linear constraint.
        """
        dofs = np.array([1.0, 2.0])
        f = MockObjective(dofs)
        c1 = MockConstraint(dofs, offset=1.0)
        lag_mul = np.array([0.5])
        mu = 2.0
        val = al.augmented_lagrangian_objective(dofs, f, [c1], lag_mul, mu)
        # L = f(x) - lag_mul * g(x) + mu/2 * ||g(x)||^2
        fx = np.sum(dofs ** 2)
        gx = np.sum(dofs) + 1.0
        expected = fx - 0.5 * gx + 1.0 * gx ** 2
        assert np.isclose(val, expected)


    def test_augmented_lagrangian_objective_least_squares(self):
        """
        Test the least-squares form of the augmented Lagrangian objective.
        Checks that the value matches the formula:
            L = 0.5 * ||f(x)||^2 + 0.5 * ||-lag_mul/sqrt(mu) + sqrt(mu)*g(x)||^2
        for a quadratic objective and linear constraint.
        """
        dofs = np.array([1.0, 2.0])
        f = MockObjective(dofs)
        c1 = MockConstraint(dofs, offset=1.0)
        lag_mul = np.array([0.5])
        mu = 2.0
        val = al.augmented_lagrangian_objective(dofs, f, [c1], lag_mul, mu, option='least-squares')
        # L = 0.5 * ||f(x)||^2 + 0.5 * ||-lag_mul/sqrt(mu) + sqrt(mu)*g(x)||^2
        fx = np.sum(dofs ** 2)
        gx = np.sum(dofs) + 1.0
        expected = 0.5 * fx ** 2 + 0.5 * ((-0.5/np.sqrt(2.0) + np.sqrt(2.0)*gx) ** 2)
        assert np.isclose(val, expected)


    def test_grad_augmented_lagrangian_standard(self):
        """
        Test the gradient of the standard augmented Lagrangian.
        Checks that the gradient matches the formula:
            grad = grad_f - lag_mul * grad_g + mu * J_g^T * g
        for a quadratic objective and linear constraint.
        """
        dofs = np.array([1.0, 2.0])
        f = MockObjective(dofs)
        c1 = MockConstraint(dofs, offset=1.0)
        lag_mul = np.array([0.5])
        mu = 2.0
        grad = al.grad_augmented_lagrangian(dofs, f, [c1], lag_mul, mu)
        # grad = grad_f - lag_mul * grad_g + mu * J_g^T * g
        grad_f = 2 * dofs
        grad_g = np.ones_like(dofs)
        gx = np.sum(dofs) + 1.0
        expected = grad_f - 0.5 * grad_g + 2.0 * grad_g * gx
        np.testing.assert_allclose(grad, expected)


    def test_grad_augmented_lagrangian_least_squares(self):
        """
        Test the gradient of the least-squares augmented Lagrangian.
        Checks that the gradient matches the formula:
            grad = grad_f * f(x) + sqrt(mu) * grad_g * g(x)
        for a quadratic objective and linear constraint.
        """
        dofs = np.array([1.0, 2.0])
        f = MockObjective(dofs)
        c1 = MockConstraint(dofs, offset=1.0)
        lag_mul = np.array([0.5])
        mu = 2.0
        grad = al.grad_augmented_lagrangian(dofs, f, [c1], lag_mul, mu, option='least-squares')
        grad_f = 2 * dofs
        grad_g = np.ones_like(dofs)
        fx = np.sum(dofs ** 2)
        gx = np.sum(dofs) + 1.0
        expected = grad_f * fx + np.sqrt(mu) * grad_g * gx
        np.testing.assert_allclose(grad, expected)

    def test_augmented_lagrangian_method_equality(self):
        """
        Test the augmented Lagrangian optimization loop on a quadratic problem with an equality constraint:
            Minimize f(x) = (x-1)^2 subject to x = 2.
        Checks that the optimizer finds x ≈ 2 and the constraint is satisfied to high precision.
        This test is run for a grid of ALM parameters.
        """
        class SimpleObjective:
            def __init__(self, x):
                self.x = np.array(x, dtype=float)
            def J(self):
                return np.sum((self.x - 1.0) ** 2)
            def dJ(self):
                return 2 * (self.x - 1.0)
        class EqualityConstraint:
            def __init__(self, x):
                self.x = np.array(x, dtype=float)
            def J(self):
                return np.sum(self.x - 2.0)
            def dJ(self):
                return np.ones_like(self.x)
        x0 = np.array([0.0])
        f = SimpleObjective(x0)
        c = EqualityConstraint(x0)
        mu_inits = [1.0, 10.0, 100.0]
        grad_tols = [1e-2, 1e-6]
        c_tols = [1e-2, 1e-6]
        MAXITERs = [50, 200]
        argmin_tols = [1e-3, 1e-8]
        MAXITER_lags = [10, 20]  # 5 is too few at low res
        lagrangian_forms = [None, 'least-squares']
        for mu_init in mu_inits:
            for grad_tol in grad_tols:
                for c_tol in c_tols:
                    for MAXITER in MAXITERs:
                        for argmin_tol in argmin_tols:
                            for MAXITER_lag in MAXITER_lags:
                                for lagrangian_form in lagrangian_forms:
                                    print(f"Testing: mu_init={mu_init}, grad_tol={grad_tol}, c_tol={c_tol}, MAXITER={MAXITER}, argmin_tol={argmin_tol}, MAXITER_lag={MAXITER_lag}, lagrangian_form={lagrangian_form}")
                                    f = SimpleObjective(x0)
                                    c = EqualityConstraint(x0)
                                    x_opt, final_L, lag_mul = al.augmented_lagrangian_method(
                                        f, [c], mu_init=mu_init, grad_tol=grad_tol, c_tol=c_tol, MAXITER=MAXITER, 
                                        argmin_tol=argmin_tol, MAXITER_lag=MAXITER_lag, lagrangian_form=lagrangian_form)
                                    constraint_val = x_opt - 2.0
                                    print('Equality:', x_opt)
                                    assert np.allclose(x_opt, 2, atol=1e-2)
                                    assert abs(constraint_val) < 1e-2
                                    assert lag_mul.shape == (1,)

    def test_augmented_lagrangian_method_slack(self):
        """
        Test the augmented Lagrangian optimization loop on a quadratic problem with a slack variable:
            Minimize f(x) = (x-1)^2 subject to x + s = 0, s <= -2 (i.e., x >= 2).
        Checks that the optimizer finds x ≈ 2 and s ≈ -2.
        """
        class SlackObjective:
            def __init__(self, xs):
                self.x = np.array(xs, dtype=float)
            def J(self):
                x = self.x[0]
                return (x - 1.0) ** 2
            def dJ(self):
                x = self.x[0]
                return np.array([2 * (x - 1.0), 0.0])
        class SlackConstraint:
            def __init__(self, xs):
                self.x = np.array(xs, dtype=float)
            def J(self):
                x, s = self.x[0], self.x[1]
                return x + s
            def dJ(self):
                return np.array([1.0, 1.0])
        xs0 = np.array([0.0, 1.0])
        bounds = [(None, None), (None, -2)]  # x unbounded, s >= 0
        mu_inits = [1.0, 10.0, 100.0]
        grad_tols = [1e-2, 1e-6]
        c_tols = [1e-2, 1e-6]
        MAXITERs = [50, 200]
        argmin_tols = [1e-3, 1e-8]
        MAXITER_lags = [10, 20]
        lagrangian_forms = [None, 'least-squares']
        for mu_init in mu_inits:
            for grad_tol in grad_tols:
                for c_tol in c_tols:
                    for MAXITER in MAXITERs:
                        for argmin_tol in argmin_tols:
                            for MAXITER_lag in MAXITER_lags:
                                for lagrangian_form in lagrangian_forms:
                                    print(f"Testing: mu_init={mu_init}, grad_tol={grad_tol}, c_tol={c_tol}, MAXITER={MAXITER}, argmin_tol={argmin_tol}, MAXITER_lag={MAXITER_lag}, lagrangian_form={lagrangian_form}")
                                    f_slack = SlackObjective(xs0)
                                    c_slack = SlackConstraint(xs0)
                                    x_opt, final_L, lag_mul = al.augmented_lagrangian_method(
                                        f_slack, [c_slack], mu_init=mu_init, grad_tol=grad_tol, c_tol=c_tol, MAXITER=MAXITER, 
                                        argmin_tol=argmin_tol, MAXITER_lag=MAXITER_lag, lagrangian_form=lagrangian_form)
                                    x_val, s_val = x_opt[0], x_opt[1]
                                    print('Slack:', x_val, s_val)
                                    assert np.allclose(x_val, 2, atol=1e-2)
                                    assert s_val <= -2.0
                                    assert lag_mul.shape == (1,)

    def test_augmented_lagrangian_method_coils_slack(self):
        """
        Test the augmented Lagrangian optimization loop on a coils problem 
        with slack variables for a grid of ALM parameters and surface resolutions. Inequality
        constraints are handled in augmented Lagrangian by introducing slack variables.
        
        The slack variables are used to enforce INEQUALITY constraints such as:
        -- minimum coil-coil distance
        -- minimum coil-surface distance
        -- maximum coil curvature
        -- maximum coil-surface curvature
        -- minimum Bnormal error on the plasma surface
        """
        from pathlib import Path
        from simsopt.geo import SurfaceRZFourier, create_equally_spaced_curves, curves_to_vtk
        from simsopt.field import BiotSavart, ScaledCurrent, coils_via_symmetries
        from simsopt.solve import augmented_lagrangian_method
        from simsopt.objectives import SquaredFlux
        from simsopt.geo import CurveCurveDistance, CurveSurfaceMinimumDistance, LinkingNumber
        from simsopt.geo import CurveLength, CurveCurveMinimumDistance
        from simsopt.geo import LpCurveCurvature
        from simsopt.solve import construct_equality_constraints
        import os

        # Define the test directory
        TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()

        # Define the filename
        filename = TEST_DIR / 'input.LandremanPaul2021_QA_lowres'

        nphis = [8, 16]  # surface resolution
        mu_inits = [100.0, 10.0]
        grad_tols = [1e-6, 1e-12]
        c_tols = [1e-6, 1e-12]
        MAXITERs = [800]  # Need 800 here to get the slack variables to zero
        argmin_tols = [1e-12]  # Needs to be fairly stringent
        MAXITER_lags = [10] 
        lagrangian_forms = [None]  #, 'least-squares']
        LENGTH_TARGET = 17.4
        CC_THRESHOLD = 0.1
        CS_THRESHOLD = 0.3
        CURVATURE_THRESHOLD = 5
        FLUX_THRESHOLD = 1e-3

        with ScratchDir(".") as tmpdir:
            OUT_DIR = tmpdir + "/output"
            os.makedirs(OUT_DIR, exist_ok=True)
            for nphi in nphis:
                ntheta = nphi
                s = SurfaceRZFourier.from_vmec_input(
                    filename,
                    range="half period",
                    nphi=nphi,
                    ntheta=ntheta)                          
                R0 = s.x[0]
                R1 = 0.6 * s.x[0]
                order = 5
                ncoils = 4
                curves = create_equally_spaced_curves(
                    ncoils, s.nfp, stellsym=s.stellsym, R0=R0, R1=R1, order=order, numquadpoints=128)
                base_currents = [ScaledCurrent(1) * 1e5 for i in range(ncoils)]
                base_currents[0].fix_all()
                base_curves = curves[:ncoils]
                coils = coils_via_symmetries(base_curves, base_currents, s.nfp, s.stellsym)
                curves = [c.curve for c in coils]
                bs = BiotSavart(coils)
                curves_to_vtk(curves, OUT_DIR + "curves_init")
                bs.set_points(s.gamma().reshape((-1, 3)))
                pointData = {"B_N/|B|": np.sum(bs.B().reshape((nphi, ntheta, 3)) *
                                            s.unitnormal(), axis=2)[:, :, None] / bs.AbsB().reshape((nphi, ntheta, 1)),
                                "modB": bs.AbsB().reshape((nphi, ntheta, 1))}
                s.to_vtk(OUT_DIR + "surf_init", extra_data=pointData)
                Jccdist = CurveCurveDistance(curves, CC_THRESHOLD, num_basecurves=ncoils)
                Jcs = [LpCurveCurvature(c, p=10, threshold=CURVATURE_THRESHOLD) for c in base_curves]
                bs.set_points(s.gamma().reshape((-1, 3)))
                coil_flux_constraint = SquaredFlux(s, bs, definition='normalized')
                coil_surface_minimum_distance_constraint = CurveSurfaceMinimumDistance(base_curves, s)
                coil_coil_minimum_distance_constraint = CurveCurveMinimumDistance(curves, minimum_distance=CC_THRESHOLD, downsample=2)
                coil_total_length_constraint = sum(CurveLength(c) for c in base_curves)
                coil_curvature_constraint = sum(Jcs)
                coil_linking_number_constraint = LinkingNumber(curves, downsample=2)

                # Setup the slack variables to convert the inequality constraints
                # into equality constraints.
                objs = [
                        coil_flux_constraint, # Code expects first constraint to be the flux constraint
                        coil_surface_minimum_distance_constraint, 
                        coil_coil_minimum_distance_constraint, 
                        coil_total_length_constraint, 
                        coil_curvature_constraint, 
                        coil_linking_number_constraint]
                types = ['upper', 'lower', 'lower', 'upper', 'upper', 'upper']

                # Curvature and CC-sep do not need slack variable thresholds because 
                # they are already handled by the LpCurveCurvature and CurveCurveDistance
                # objectives.
                thresholds = [FLUX_THRESHOLD, CS_THRESHOLD, 0.0, LENGTH_TARGET, 0.0, 0.0]
                inequality_constraints = construct_equality_constraints(objs, types, thresholds)
                dofs_orig = inequality_constraints[0].x.copy()
                for mu_init in mu_inits:
                    for grad_tol in grad_tols:
                        for c_tol in c_tols:
                            for MAXITER in MAXITERs:
                                for argmin_tol in argmin_tols:
                                    for MAXITER_lag in MAXITER_lags:
                                        for lagrangian_form in lagrangian_forms:
                                            print(f"Testing: nphi={nphi}, mu_init={mu_init}, grad_tol={grad_tol}, c_tol={c_tol}, MAXITER={MAXITER}, argmin_tol={argmin_tol}, MAXITER_lag={MAXITER_lag}, lagrangian_form={lagrangian_form}")
                                            # Just check that the optimization runs without error for each parameter set
                                            inequality_constraints[0].x = dofs_orig.copy()
                                            x, fnc, lag_mul = augmented_lagrangian_method(
                                                inequality_constraints=inequality_constraints, 
                                                # verbose=True,
                                                mu_init=mu_init, grad_tol=grad_tol, c_tol=c_tol,
                                                MAXITER=MAXITER, argmin_tol=argmin_tol, MAXITER_lag=MAXITER_lag,
                                                lagrangian_form=lagrangian_form)
                                            assert x is not None
                                            print(x[-len(inequality_constraints):])
                                            for i in range(len(inequality_constraints)):
                                                print('Jobj: ', i, inequality_constraints[i].Jobj.J())
                                            print('coil_surface_minimum_distance_constraint.J():', coil_surface_minimum_distance_constraint.J())
                                            print('coil_coil_minimum_distance_constraint.J():', coil_coil_minimum_distance_constraint.J())
                                            print('Jccdist.shortest_distance():', Jccdist.shortest_distance())
                                            print('coil_total_length_constraint.J():', coil_total_length_constraint.J())
                                            print('coil_curvature_constraint.J():', coil_curvature_constraint.J())
                                            print('coil_flux_constraint.J():', coil_flux_constraint.J())
                                            print('coil_linking_number_constraint.J():', coil_linking_number_constraint.J())
                                            assert coil_surface_minimum_distance_constraint.J() > CS_THRESHOLD
                                            assert Jccdist.shortest_distance() >= CC_THRESHOLD - 1e-2
                                            assert coil_total_length_constraint.J() < LENGTH_TARGET
                                            assert coil_curvature_constraint.J() < CURVATURE_THRESHOLD
                                            assert coil_flux_constraint.J() < FLUX_THRESHOLD
                                            assert np.isclose(coil_linking_number_constraint.J(), 0.0)

    def test_augmented_lagrangian_method_coils(self):
        """
        Test the augmented Lagrangian optimization loop on a coils problem for a grid of ALM parameters and surface resolutions.
        """
        from pathlib import Path
        from simsopt.geo import SurfaceRZFourier, create_equally_spaced_curves, curves_to_vtk
        from simsopt.field import BiotSavart, Current, coils_via_symmetries
        from simsopt.solve import augmented_lagrangian_method
        from simsopt.objectives import SquaredFlux, QuadraticPenalty
        from simsopt.geo import CurveSurfaceDistance, LpCurveCurvature
        from simsopt.geo import CurveLength
        import os

        # Define the test directory
        TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()

        # Define the filename
        filename = TEST_DIR / 'input.LandremanPaul2021_QA_lowres'

        nphis = [8, 16, 32]  # surface resolution
        mu_inits = [10.0]
        grad_tols = [1e-2, 1e-6]
        c_tols = [1e-2, 1e-6]
        MAXITERs = [10, 20]
        argmin_tols = [1e-2, 1e-8]
        MAXITER_lags = [5]
        lagrangian_forms = [None, 'least-squares']
        LENGTH_TARGET = 17.4
        CC_THRESHOLD = 0.1
        CS_THRESHOLD = 0.3
        CURVATURE_THRESHOLD = 5

        with ScratchDir(".") as tmpdir:
            OUT_DIR = tmpdir + "/output"
            os.makedirs(OUT_DIR, exist_ok=True)
            for nphi in nphis:
                ntheta = nphi
                s = SurfaceRZFourier.from_vmec_input(
                    filename,
                    range="half period",
                    nphi=nphi,
                    ntheta=ntheta)                          
                R0 = s.x[0]
                R1 = 0.6 * s.x[0]
                order = 5
                ncoils = 4
                curves = create_equally_spaced_curves(
                    ncoils, s.nfp, stellsym=s.stellsym, R0=R0, R1=R1, order=order, numquadpoints=128)
                base_currents = [Current(1e5) for i in range(ncoils)]
                base_currents[0].fix_all()
                base_curves = curves[:ncoils]
                coils = coils_via_symmetries(base_curves, base_currents, s.nfp, s.stellsym)
                curves = [c.curve for c in coils]
                bs = BiotSavart(coils)
                curves_to_vtk(curves, OUT_DIR + "curves_init")
                bs.set_points(s.gamma().reshape((-1, 3)))
                pointData = {"B_N/|B|": np.sum(bs.B().reshape((nphi, ntheta, 3)) *
                                            s.unitnormal(), axis=2)[:, :, None] / bs.AbsB().reshape((nphi, ntheta, 1)),
                                "modB": bs.AbsB().reshape((nphi, ntheta, 1))}
                s.to_vtk(OUT_DIR + "surf_init", extra_data=pointData)
                bs.set_points(s.gamma().reshape((-1, 3)))
                Jf = SquaredFlux(s, bs)
                Jls = [CurveLength(c) for c in base_curves]
                Jcsdist = CurveSurfaceDistance(curves, s, CS_THRESHOLD)
                Jcs = [LpCurveCurvature(c, 2, CURVATURE_THRESHOLD) for c in base_curves]
                equality_constraints = [Jf, Jcsdist, QuadraticPenalty(sum(Jls), LENGTH_TARGET, "max"), sum(Jcs)]
                for mu_init in mu_inits:
                    for grad_tol in grad_tols:
                        for c_tol in c_tols:
                            for MAXITER in MAXITERs:
                                for argmin_tol in argmin_tols:
                                    for MAXITER_lag in MAXITER_lags:
                                        for lagrangian_form in lagrangian_forms:
                                            print(f"Testing: nphi={nphi}, mu_init={mu_init}, grad_tol={grad_tol}, c_tol={c_tol}, MAXITER={MAXITER}, argmin_tol={argmin_tol}, MAXITER_lag={MAXITER_lag}, lagrangian_form={lagrangian_form}")
                                            # Just check that the optimization runs without error for each parameter set
                                            x, fnc, lag_mul = augmented_lagrangian_method(
                                                equality_constraints=equality_constraints, mu_init=mu_init, grad_tol=grad_tol, c_tol=c_tol,
                                                MAXITER=MAXITER, argmin_tol=argmin_tol, MAXITER_lag=MAXITER_lag,
                                                lagrangian_form=lagrangian_form)
                                            assert x is not None
                                            assert Jf.J() < 1e-2


    def test_augmented_lagrangian_method_coils_larger_bounds(self):
        """
        Test the augmented Lagrangian optimization loop on a coils problem with larger upper bounds,
        to see if the optimization converges well with a variety of parameters.
        """
        from pathlib import Path
        from simsopt.geo import SurfaceRZFourier, create_equally_spaced_curves, curves_to_vtk
        from simsopt.field import BiotSavart, Current, coils_via_symmetries
        from simsopt.solve import augmented_lagrangian_method
        from simsopt.objectives import SquaredFlux
        from simsopt.geo import CurveCurveDistance, CurveSurfaceMinimumDistance, LinkingNumber
        from simsopt.geo import CurveLength, CurveCurveMinimumDistance
        from simsopt.geo import LpCurveCurvature
        from simsopt.solve import construct_equality_constraints
        import os
        np.random.seed(1)

        # Define the test directory
        TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()

        # Define the filename
        filename = TEST_DIR / 'input.LandremanPaul2021_QA_lowres'

        nphi = 16  # surface resolution
        MAXITER = 800
        MAXITER_lags = [10]
        LENGTH_TARGET = [15.0, 20.0, 30.0, 40.0]
        CC_THRESHOLD = [0.1, 0.2, 0.3]
        CS_THRESHOLD = [0.1, 0.2, 0.3]
        CURVATURE_THRESHOLDS = [5, 15]
        FLUX_THRESHOLDS = [1e-2, 1e-3]

        with ScratchDir(".") as tmpdir:
            OUT_DIR = tmpdir + "/output"
            os.makedirs(OUT_DIR, exist_ok=True)
            ntheta = nphi
            s = SurfaceRZFourier.from_vmec_input(
                filename,
                range="half period",
                nphi=nphi,
                ntheta=ntheta)                          
            R0 = s.x[0]
            R1 = 0.6 * s.x[0]
            order = 5
            ncoils = 4
            curves = create_equally_spaced_curves(
                ncoils, s.nfp, stellsym=s.stellsym, R0=R0, R1=R1, order=order, numquadpoints=128)
            base_currents = [Current(1.0) * 1e5 for i in range(ncoils)]
            base_currents[0].fix_all()
            base_curves = curves[:ncoils]
            coils = coils_via_symmetries(base_curves, base_currents, s.nfp, s.stellsym)
            curves = [c.curve for c in coils]
            bs = BiotSavart(coils)
            curves_to_vtk(curves, OUT_DIR + "curves_init")
            bs.set_points(s.gamma().reshape((-1, 3)))
            pointData = {"B_N/|B|": np.sum(bs.B().reshape((nphi, ntheta, 3)) *
                                        s.unitnormal(), axis=2)[:, :, None] / bs.AbsB().reshape((nphi, ntheta, 1)),
                            "modB": bs.AbsB().reshape((nphi, ntheta, 1))}
            s.to_vtk(OUT_DIR + "surf_init", extra_data=pointData)
            bs.set_points(s.gamma().reshape((-1, 3)))
            coil_flux_constraint = SquaredFlux(s, bs, definition='normalized')
            dofs_before = np.concatenate([coil_flux_constraint.x.copy(), np.zeros(6)])

            for LENGTH_TARGET in LENGTH_TARGET:
                for CC_THRESHOLD in CC_THRESHOLD:
                    for CS_THRESHOLD in CS_THRESHOLD:
                        for CURVATURE_THRESHOLD in CURVATURE_THRESHOLDS:
                            for FLUX_THRESHOLD in FLUX_THRESHOLDS:
                                for MAXITER_lag in MAXITER_lags:
                                    print(f"Testing: nphi={nphi}, LENGTH_TARGET={LENGTH_TARGET}, CC_THRESHOLD={CC_THRESHOLD}, CS_THRESHOLD={CS_THRESHOLD}, CURVATURE_THRESHOLD={CURVATURE_THRESHOLD}, FLUX_THRESHOLD={FLUX_THRESHOLD}, MAXITER_lag={MAXITER_lag}")
                                    # Just check that the optimization runs without error for each parameter set
                                    Jccdist = CurveCurveDistance(curves, CC_THRESHOLD, num_basecurves=ncoils)
                                    Jcs = [LpCurveCurvature(c, p=10, threshold=CURVATURE_THRESHOLD) for c in base_curves]
                                    coil_surface_minimum_distance_constraint = CurveSurfaceMinimumDistance(base_curves, s)
                                    coil_coil_minimum_distance_constraint = CurveCurveMinimumDistance(curves, minimum_distance=CC_THRESHOLD, downsample=2)
                                    coil_total_length_constraint = sum(CurveLength(c) for c in base_curves)
                                    coil_curvature_constraint = sum(Jcs)
                                    coil_linking_number_constraint = LinkingNumber(curves, downsample=2)

                                    # Setup the slack variables to convert the inequality constraints
                                    # into equality constraints.
                                    objs = [
                                            coil_flux_constraint, # Code expects first constraint to be the flux constraint
                                            coil_surface_minimum_distance_constraint, 
                                            coil_coil_minimum_distance_constraint, 
                                            coil_total_length_constraint, 
                                            coil_curvature_constraint, 
                                            coil_linking_number_constraint]
                                    types = ['upper', 'lower', 'lower', 'upper', 'upper', 'upper']

                                    # Curvature and CC-sep do not need slack variable thresholds because 
                                    # they are already handled by the LpCurveCurvature and CurveCurveDistance
                                    # objectives.
                                    thresholds = [FLUX_THRESHOLD, CS_THRESHOLD, 0.0, LENGTH_TARGET, 0.0, 0.0]
                                    inequality_constraints = construct_equality_constraints(objs, types, thresholds)
                                    for i in range(len(inequality_constraints)):
                                        dof_diff = len(inequality_constraints[0].x) - len(inequality_constraints[i].x)
                                        inequality_constraints[i].x = dofs_before[dof_diff:].copy()
                                    x, fnc, lag_mul = augmented_lagrangian_method(
                                        inequality_constraints=inequality_constraints, 
                                        verbose=True, 
                                        MAXITER=MAXITER, MAXITER_lag=MAXITER_lag,
                                    )
                                    assert x is not None
                                    print(x[-len(inequality_constraints):])
                                    for i in range(len(inequality_constraints)):
                                        print('Jobj: ', i, inequality_constraints[i].Jobj.J())
                                    print('coil_surface_minimum_distance_constraint.J():', coil_surface_minimum_distance_constraint.J())
                                    print('coil_coil_minimum_distance_constraint.J():', coil_coil_minimum_distance_constraint.J())
                                    print('Jccdist.shortest_distance():', Jccdist.shortest_distance())
                                    print('coil_total_length_constraint.J():', coil_total_length_constraint.J())
                                    print('coil_curvature_constraint.J():', coil_curvature_constraint.J())
                                    print('coil_flux_constraint.J():', coil_flux_constraint.J())
                                    print('coil_linking_number_constraint.J():', coil_linking_number_constraint.J())
                                    assert coil_surface_minimum_distance_constraint.J() > CS_THRESHOLD
                                    assert Jccdist.shortest_distance() >= CC_THRESHOLD - 1e-2
                                    assert coil_total_length_constraint.J() < LENGTH_TARGET
                                    print(coil_curvature_constraint.J(), CURVATURE_THRESHOLD)
                                    assert coil_curvature_constraint.J() < CURVATURE_THRESHOLD
                                    assert coil_flux_constraint.J() < FLUX_THRESHOLD
                                    assert np.isclose(coil_linking_number_constraint.J(), 0.0)


if __name__ == "__main__":
    unittest.main()