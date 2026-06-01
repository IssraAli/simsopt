"""Unit tests for ``NonQuasiSymmetricRatioHelical``.

Two checks:

1. ``(M, N) = (1, 0)`` reduction: ``J``, ``dJ_by_dB``,
   ``dJ_by_dsurfacecoefficients`` and the full reverse-mode ``dJ`` must agree
   with ``NonQuasiSymmetricRatio`` to numerical precision on the NCSX coil
   set (mirrors the QA/QH metric used by ``boozerQA.py``).

2. 4-point finite-difference checks of ``dJ_by_dB . dB/dI`` (the new gradient
   path) for ``(M, N) in {(1, 0), (1, nfp), (0, 1)}``, evaluated at a frozen
   surface so the test is independent of BoozerSurface Newton tolerances.
"""

import numpy as np
import unittest

from simsopt.configs import get_data
from simsopt.field import BiotSavart
from simsopt.geo import (
    SurfaceXYZTensorFourier,
    BoozerSurface,
    Volume,
    NonQuasiSymmetricRatio,
    NonQuasiSymmetricRatioHelical,
)


class NonQuasiSymmetricRatioHelicalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_curves, cls.base_currents, cls.ma, cls.nfp, cls.bs = get_data(
            "ncsx"
        )
        mpol, ntor = 6, 6
        phis = np.linspace(0, 1 / cls.nfp, 2 * ntor + 1, endpoint=False)
        thetas = np.linspace(0, 1.0, 2 * mpol + 1, endpoint=False)
        s = SurfaceXYZTensorFourier(
            mpol=mpol,
            ntor=ntor,
            stellsym=True,
            nfp=cls.nfp,
            quadpoints_phi=phis,
            quadpoints_theta=thetas,
        )
        s.fit_to_curve(cls.ma, 0.1, flip_theta=True)
        vol = Volume(s)
        cls.boozer_surface = BoozerSurface(cls.bs, s, vol, vol.J())
        # We do not require a tight Newton: the partial-derivative checks are
        # against a frozen (self-consistent) state of (surface, B field).
        cls.boozer_surface.solve_residual_equation_exactly_newton(
            tol=1e-13, maxiter=40, iota=-0.406, G=None
        )
        cls.boozer_surface.need_to_run_code = False
        cls.bs_eval = BiotSavart(cls.bs.coils)

    def _freeze(self):
        self.boozer_surface.need_to_run_code = False

    def test_reduction_to_QA(self):
        """For (M, N) = (1, 0) the helical metric must match QA exactly."""
        J_qa = NonQuasiSymmetricRatio(
            self.boozer_surface, self.bs_eval, sDIM=16
        )
        J_hel = NonQuasiSymmetricRatioHelical(
            self.boozer_surface, self.bs_eval, helicity_M=1, helicity_N=0,
            sDIM=16,
        )
        self.assertAlmostEqual(
            J_qa.J(), J_hel.J(), places=14,
            msg="J() reduction (M, N) = (1, 0) -> QA failed",
        )
        np.testing.assert_allclose(
            J_qa.dJ_by_dB(), J_hel.dJ_by_dB(),
            rtol=1e-12, atol=1e-15,
            err_msg="dJ_by_dB reduction failed",
        )
        np.testing.assert_allclose(
            J_qa.dJ_by_dsurfacecoefficients(),
            J_hel.dJ_by_dsurfacecoefficients(),
            rtol=1e-10, atol=1e-13,
            err_msg="dJ_by_dsurfacecoefficients reduction failed",
        )
        np.testing.assert_allclose(
            J_qa.dJ(), J_hel.dJ(),
            rtol=1e-10, atol=1e-13,
            err_msg="full dJ() reduction failed",
        )

    def _ana_dJ_dI(self, obj, current_obj):
        """dJ/dI = (dJ/dB) . (dB/dI) at the auxiliary surface."""
        aux_pts = obj.surface.gamma().reshape((-1, 3))
        self.bs_eval.set_points(aux_pts)
        self._freeze()
        obj.recompute_bell()
        obj.J()
        dJ_dB = obj.dJ_by_dB().reshape((-1, 3))

        val0 = current_obj.get_value()
        eps = max(abs(val0) * 1e-4, 1.0)
        current_obj.x = np.array([val0 + eps])
        self.bs_eval.set_points(aux_pts)
        B_plus = self.bs_eval.B().copy()
        current_obj.x = np.array([val0 - eps])
        self.bs_eval.set_points(aux_pts)
        B_minus = self.bs_eval.B().copy()
        current_obj.x = np.array([val0])
        self._freeze()
        dB_dI = (B_plus - B_minus) / (2 * eps)
        return float(np.sum(dJ_dB * dB_dI))

    def _fd_dJ_dI(self, obj, current_obj, eps):
        val0 = current_obj.get_value()
        aux_pts = obj.surface.gamma().reshape((-1, 3))

        def f(x):
            current_obj.x = np.array([x])
            self._freeze()
            self.bs_eval.set_points(aux_pts)
            obj.recompute_bell()
            return float(obj.J())

        J1 = f(val0 + 2 * eps)
        J2 = f(val0 + eps)
        J3 = f(val0 - eps)
        J4 = f(val0 - 2 * eps)
        current_obj.x = np.array([val0])
        self._freeze()
        return (-J1 + 8 * J2 - 8 * J3 + J4) / (12 * eps)

    def test_taylor_dJ_dB_against_coil_currents(self):
        """4-point FD: dJ/dI from analytical (dJ/dB).(dB/dI) vs FD of J."""
        for (M, N) in [(1, 0), (1, self.nfp), (0, 1)]:
            Jh = NonQuasiSymmetricRatioHelical(
                self.boozer_surface, self.bs_eval,
                helicity_M=M, helicity_N=N, sDIM=16,
            )
            for i, c in enumerate(self.base_currents):
                ana = self._ana_dJ_dI(Jh, c)
                eps = max(abs(c.get_value()) * 1e-4, 1.0)
                fd = self._fd_dJ_dI(Jh, c, eps)
                denom = max(abs(ana), 1e-30)
                rel = abs(fd - ana) / denom
                self.assertLess(
                    rel, 1e-6,
                    f"FD vs analytical dJ/dI mismatch for "
                    f"(M, N) = ({M}, {N}), coil {i}: "
                    f"ana={ana:.6e}, fd={fd:.6e}, rel={rel:.3e}",
                )


if __name__ == "__main__":
    unittest.main()
