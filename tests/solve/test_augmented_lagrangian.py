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

    def test_augmented_lagrangian_method_coils(self):
        """
        Test the augmented Lagrangian optimization loop on a coils problem for a grid of ALM parameters and surface resolutions.
        """
        from pathlib import Path
        from simsopt.geo import SurfaceRZFourier, create_equally_spaced_curves, curves_to_vtk
        from simsopt.field import BiotSavart, Current, coils_via_symmetries
        from simsopt.solve import augmented_lagrangian_method
        from simsopt.objectives import SquaredFlux, QuadraticPenalty
        from simsopt.geo import CurveSurfaceDistance, LpCurveCurvature, CurveCurveDistance
        from simsopt.geo import CurveLength, LinkingNumber
        import os

        # Define the test directory
        TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()

        # Define the filename
        filename = TEST_DIR / 'input.LandremanPaul2021_QA_lowres'

        nphis = [8, 16, 32]  # surface resolution
        mu_inits = [10.0]
        grad_tols = [1e-4, 1e-6]
        c_tols = [1e-4, 1e-6]
        MAXITERs = [10, 20]
        argmin_tols = [1e-4, 1e-8]
        MAXITER_lags = [5]
        lagrangian_forms = [None, 'least-squares']
        FLUX_THRESHOLD = 1e-3
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
                Jf = SquaredFlux(s, bs, threshold=FLUX_THRESHOLD)
                dofs_orig = Jf.x.copy()
                Jls = [CurveLength(c) for c in base_curves]
                Jcsdist = CurveSurfaceDistance(curves, s, CS_THRESHOLD)
                Jccdist = CurveCurveDistance(curves, CC_THRESHOLD, num_basecurves=ncoils)
                Jcs = [LpCurveCurvature(c, 10, CURVATURE_THRESHOLD) for c in base_curves]
                Jlink = LinkingNumber(curves, downsample=2)
                equality_constraints = [Jf, Jccdist,Jcsdist, QuadraticPenalty(sum(Jls), LENGTH_TARGET, "max"), sum(Jcs), Jlink]
                for mu_init in mu_inits:
                    for grad_tol in grad_tols:
                        for c_tol in c_tols:
                            for MAXITER in MAXITERs:
                                for argmin_tol in argmin_tols:
                                    for MAXITER_lag in MAXITER_lags:
                                        for lagrangian_form in lagrangian_forms:
                                            Jf.x = dofs_orig.copy()
                                            print(f"Testing: nphi={nphi}, mu_init={mu_init}, grad_tol={grad_tol}, c_tol={c_tol}, MAXITER={MAXITER}, argmin_tol={argmin_tol}, MAXITER_lag={MAXITER_lag}, lagrangian_form={lagrangian_form}")
                                            # Just check that the optimization runs without error for each parameter set
                                            x, fnc, lag_mul = augmented_lagrangian_method(
                                                equality_constraints=equality_constraints, mu_init=mu_init, grad_tol=grad_tol, c_tol=c_tol,
                                                MAXITER=MAXITER, argmin_tol=argmin_tol, MAXITER_lag=MAXITER_lag,
                                                lagrangian_form=lagrangian_form)
                                            assert x is not None
                                            assert Jf.J() < FLUX_THRESHOLD
                                            assert Jccdist.J() < CC_THRESHOLD
                                            assert Jcsdist.J() < CS_THRESHOLD
                                            assert sum(Jcs).J() < CURVATURE_THRESHOLD
                                            assert sum(Jls).J() < LENGTH_TARGET
                                            assert Jlink.J() == 0


    def test_augmented_lagrangian_method_coils_scan(self):
        """
        Test the augmented Lagrangian optimization loop on a coils problem for a grid of ALM parameters and surface resolutions.
        """
        from pathlib import Path
        from simsopt.geo import SurfaceRZFourier, create_equally_spaced_curves, curves_to_vtk
        from simsopt.field import BiotSavart, Current, coils_via_symmetries
        from simsopt.solve import augmented_lagrangian_method
        from simsopt.objectives import SquaredFlux, QuadraticPenalty
        from simsopt.geo import CurveSurfaceDistance, LpCurveCurvature, CurveCurveDistance
        from simsopt.geo import CurveLength, LinkingNumber
        import os

        # Define the test directory
        TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()

        # Define the filename
        filename = TEST_DIR / 'input.LandremanPaul2021_QA_lowres'

        nphi = 8  # surface resolution
        MAXITER = 100
        FLUX_THRESHOLDS = [3e-4]
        LENGTH_TARGETS = [17.4, 40.0, 100.0]
        CC_THRESHOLDS = [0.1, 0.2]
        CS_THRESHOLDS = [0.3, 0.4]
        CURVATURE_THRESHOLDS = [5]

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
            eps = 1e-2
            dofs_orig = Jf.x.copy()
            for FLUX_THRESHOLD in FLUX_THRESHOLDS:
                for LENGTH_TARGET in LENGTH_TARGETS:
                    for CC_THRESHOLD in CC_THRESHOLDS:
                        for CS_THRESHOLD in CS_THRESHOLDS:
                            for CURVATURE_THRESHOLD in CURVATURE_THRESHOLDS:
                                Jf = SquaredFlux(s, bs, threshold=FLUX_THRESHOLD)
                                Jls = [CurveLength(c) for c in base_curves]
                                Jcsdist = CurveSurfaceDistance(curves, s, CS_THRESHOLD)
                                Jccdist = CurveCurveDistance(curves, CC_THRESHOLD, num_basecurves=ncoils)
                                Jcs = [LpCurveCurvature(c, 10, CURVATURE_THRESHOLD) for c in base_curves]
                                Jlink = LinkingNumber(curves, downsample=2)
                                equality_constraints = [Jf, Jccdist,Jcsdist, 
                                                        QuadraticPenalty(sum(Jls), LENGTH_TARGET, "max"), 
                                                        sum(Jcs),
                                                        Jlink]
    
                                print(f"Testing: nphi={nphi}, FLUX_THRESHOLD={FLUX_THRESHOLD}, LENGTH_TARGET={LENGTH_TARGET}, CC_THRESHOLD={CC_THRESHOLD}, CS_THRESHOLD={CS_THRESHOLD}, CURVATURE_THRESHOLD={CURVATURE_THRESHOLD}")
                                # Just check that the optimization runs without error for each parameter set
                                Jf.x = dofs_orig.copy()
                                x, fnc, lag_mul = augmented_lagrangian_method(
                                    equality_constraints=equality_constraints, 
                                    MAXITER_lag=10,
                                    MAXITER=MAXITER)
                                assert x is not None
                                assert Jf.J() < FLUX_THRESHOLD + eps
                                assert Jccdist.J() < CC_THRESHOLD + eps
                                assert Jcsdist.J() < CS_THRESHOLD + eps
                                assert sum(Jcs).J() < CURVATURE_THRESHOLD + eps
                                assert sum(Jls).J() < LENGTH_TARGET + eps
                                assert Jlink.J() == 0

if __name__ == "__main__":
    unittest.main()