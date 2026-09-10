"""Unit tests for the pure-numpy core of Gamma_W (no firm3d field dependency).

Style: firm3d/simsopt convention -- unittest.TestCase classes, run under pytest,
with DESC-style markers. These are the FAST tests (mark: unit): the turning-point
bounce quadrature, well detection, the B_*/kappa normalizations, the occupied-well
action wiring, the grid/reach helpers, the aggregation band/regime logic, the
GWParams defaults, and the Nemov gamma_c angle. None imports firm3d field code, so
they run anywhere in <1 s.

Run:  pytest tests/test_core.py            (or)  python -m pytest -m unit
"""
import unittest

import numpy as np
import pytest
import scipy.integrate as si
from scipy.optimize import brentq

# Canonical engine (put on sys.path by conftest.py; becomes `from gammaw import ...`
# once packaging lands).
from Gamma_W_final import (integrate_well, physical_JT, find_wells, Bstar_from_pitch,
                           kappa, aggregate, occupied_action, selftest,
                           _dzeta_grid, _reach_at, GWParams, WellIntegral, OccResult,
                           M_ALPHA, Z_ALPHA, E_CHARGE, SIGN_CONV, TWO_PI, V0_ALPHA)


# ============================================================ helpers
class _CosineLine:
    """Minimal analytic stand-in for FieldBundle: a single cosine trapped well on a
    field line, enough to exercise find_wells -> integrate_well -> occupied_action
    without firm3d. B(zeta) = B0[1 + eps(1 - cos(nfp*zeta))]; gII constant."""

    def __init__(self, B0=5.0, eps=0.5, gII=100.0, nfp=1):
        self.B0, self.eps, self._gII, self.nfp = B0, eps, gII, nfp

    def line_B(self, s, alpha, zeta_center=0.0, n_periods=8, pts_per_period=160, iota=None):
        L = n_periods * TWO_PI / self.nfp
        npts = n_periods * pts_per_period
        zeta = np.linspace(zeta_center - 0.5 * L, zeta_center + 0.5 * L, npts)
        B = self.B0 * (1.0 + self.eps * (1.0 - np.cos(self.nfp * zeta)))
        return zeta, B, self._gII


# ============================================================ tests
@pytest.mark.unit
class TestBounceQuadrature(unittest.TestCase):
    """The verifiable heart: turning-point-aware J and tau_b vs scipy.quad.
    Mirrors Gamma_W_final --selftest, promoted to parametrized assertions."""

    def _reference(self, Bfun, B_star, gII, lo, hi):
        z = np.linspace(lo, hi, 2001)
        b = Bfun(z)
        kmin = int(np.argmin(b))
        wi = integrate_well(z, b, gII, B_star, 0, len(z), kmin)
        zmin = z[kmin]
        zl = brentq(lambda x: Bfun(x) - B_star, lo, zmin)
        zr = brentq(lambda x: Bfun(x) - B_star, zmin, hi)
        IJ = si.quad(lambda x: np.sqrt(max(0, 1 - Bfun(x) / B_star)) * gII / Bfun(x),
                     zl, zr, points=[zl, zr], limit=200)[0]
        IT = si.quad(lambda x: gII / (Bfun(x) * np.sqrt(max(1e-300, 1 - Bfun(x) / B_star))),
                     zl, zr, points=[zl, zr], limit=200)[0]
        return wi, IJ, IT

    def test_parabolic_shallow_well(self):
        B0, gII = 5.0, 100.0
        wi, IJ, IT = self._reference(lambda z: B0 * (1 + 0.5 * z ** 2), 5.05, gII, -1, 1)
        np.testing.assert_allclose(wi.J_geom, IJ, rtol=2e-3)
        np.testing.assert_allclose(wi.T_geom, IT, rtol=5e-3)

    def test_cosine_wells_across_depth(self):
        """The integrable 1/sqrt(g) turning-point singularity must be handled at
        shallow (q*=0.05) through deep (q*=0.8) wells."""
        B0, gII = 5.0, 100.0
        for q in (0.05, 0.3, 0.8):
            with self.subTest(q_star=q):
                Bfun = lambda z: B0 * (1 + 0.5 * (1 - np.cos(z)))
                wi, IJ, IT = self._reference(Bfun, B0 * (1 + q), gII, -np.pi, np.pi)
                self.assertLess(abs(wi.J_geom - IJ) / abs(IJ), 2e-3)
                self.assertLess(abs(wi.T_geom - IT) / abs(IT), 5e-3)

    def test_physical_scaling(self):
        """J = 2 v0 * J_geom, tau_b = (2/v0) * T_geom -- the physical prefactors."""
        B0, gII, v0 = 5.0, 100.0, 3e6
        z = np.linspace(-1, 1, 2001)
        b = B0 * (1 + 0.5 * z ** 2)
        wi = integrate_well(z, b, gII, 5.05, 0, len(z), int(np.argmin(b)))
        J, tau = physical_JT(wi, v0)
        np.testing.assert_allclose(J, 2 * v0 * wi.J_geom)
        np.testing.assert_allclose(tau, (2.0 / v0) * wi.T_geom)

    def test_deeper_well_has_larger_action(self):
        """Monotonicity sanity: a deeper turning field (larger B_*) admits more of
        the well, so both J_geom and tau_b grow with B_*."""
        B0, gII = 5.0, 100.0
        z = np.linspace(-np.pi, np.pi, 4001)
        b = B0 * (1 + 0.5 * (1 - np.cos(z)))
        kmin = int(np.argmin(b))
        Js, Ts = [], []
        for q in (0.1, 0.4, 0.8):
            wi = integrate_well(z, b, gII, B0 * (1 + q), 0, len(z), kmin)
            Js.append(wi.J_geom); Ts.append(wi.T_geom)
        self.assertTrue(np.all(np.diff(Js) > 0), f"J not increasing: {Js}")
        self.assertTrue(np.all(np.diff(Ts) > 0), f"tau not increasing: {Ts}")


@pytest.mark.unit
class TestWellDetection(unittest.TestCase):
    def test_single_well(self):
        z = np.linspace(-np.pi, np.pi, 401)
        B = 5.0 * (1 + 0.5 * (1 - np.cos(z)))
        wells = find_wells(z, B, B_star=5.0 * 1.3)
        self.assertEqual(len(wells), 1)
        i0, i1, kmin = wells[0]
        self.assertTrue(i0 < kmin < i1)                 # minimum strictly interior
        np.testing.assert_allclose(z[kmin], 0.0, atol=z[1] - z[0])

    def test_two_wells(self):
        # domain aligned so BOTH minima (z=0, pi) are interior and the ends + the
        # middle (z=+-pi/2) are barriers above B_star -> two bounded interior wells.
        z = np.linspace(-np.pi / 2, 3 * np.pi / 2, 801)
        B = 5.0 * (1 + 0.4 * (1 - np.cos(2 * z)))
        wells = find_wells(z, B, B_star=5.0 * 1.3)
        self.assertEqual(len(wells), 2)

    def test_no_well_when_passing(self):
        z = np.linspace(-np.pi, np.pi, 401)
        B = 5.0 * (1 + 0.5 * (1 - np.cos(z)))
        # B_star below the global min -> particle never turns -> no bounded well
        self.assertEqual(len(find_wells(z, B, B_star=4.0)), 0)

    def test_boundary_touching_well_is_excluded(self):
        """A trough that runs off the domain edge (no barrier on one side) is NOT a
        bounded interior well and must be dropped (i0==0 or i1==n filtered out)."""
        z = np.linspace(0.0, np.pi, 401)                # only the rising half of a cos well
        B = 5.0 * (1 + 0.5 * (1 - np.cos(z)))           # minimum sits at the z=0 boundary
        self.assertEqual(len(find_wells(z, B, B_star=5.0 * 1.3)), 0)


@pytest.mark.unit
class TestNormalizations(unittest.TestCase):
    def test_bstar_from_pitch(self):
        # B_* = B/(1-xi^2); xi=0 -> B_*=B (deeply trapped), xi->1 -> B_*->inf (passing)
        np.testing.assert_allclose(Bstar_from_pitch(3.0, 0.0), 3.0)
        np.testing.assert_allclose(Bstar_from_pitch(3.0, 0.5), 3.0 / 0.75)
        self.assertGreater(Bstar_from_pitch(3.0, 0.999999), 1e6)

    def test_bstar_monotone_in_pitch(self):
        b = [Bstar_from_pitch(3.0, xi) for xi in (0.0, 0.3, 0.6, 0.9)]
        self.assertTrue(np.all(np.diff(b) > 0))

    def test_kappa_sign_and_magnitude(self):
        # kappa = sign_conv * m_alpha/(Z_alpha e |psi|), sign_conv=+1 -> positive
        k = kappa(6.66)
        self.assertGreater(k, 0.0)
        np.testing.assert_allclose(k, M_ALPHA / (Z_ALPHA * E_CHARGE * 6.66), rtol=1e-12)
        # kappa uses |psi| -> sign of psi does not flip it
        np.testing.assert_allclose(kappa(-6.66), k)

    def test_kappa_inverse_linear_and_sign_conv(self):
        # kappa ~ 1/|psi| ; and the sign convention flips the drift sign
        np.testing.assert_allclose(kappa(2.0) / kappa(4.0), 2.0, rtol=1e-12)
        np.testing.assert_allclose(kappa(6.66, sign_conv=-1.0), -kappa(6.66))
        self.assertEqual(SIGN_CONV, 1.0)                # frozen by the Landreman gate


@pytest.mark.unit
class TestOccupiedActionCore(unittest.TestCase):
    """occupied_action wiring on an analytic cosine line (no firm3d): the nearest
    well to the hint is selected, its physical J/tau are positive and finite, and
    they match a direct integrate_well on the same samples."""

    def test_occupied_well_matches_direct_integral(self):
        line = _CosineLine(B0=5.0, eps=0.5, gII=100.0, nfp=1)
        B_star = 5.0 * 1.4
        occ = occupied_action(line, s=0.3, alpha=0.0, B_star=B_star, v0=V0_ALPHA,
                              zeta_hint=0.0, n_periods=8, pts_per_period=160, iota=0.0)
        self.assertTrue(occ.ok)
        self.assertGreater(occ.J, 0.0)
        self.assertGreater(occ.tau, 0.0)
        self.assertTrue(np.isfinite(occ.J) and np.isfinite(occ.tau))
        np.testing.assert_allclose(occ.zeta_min, 0.0, atol=1e-2)   # occupied well centered at hint
        # cross-check against integrate_well on the very samples occupied_action saw
        zeta, B, gII = line.line_B(0.3, 0.0, 0.0, 8, 160, iota=0.0)
        wells = find_wells(zeta, B, B_star)
        kmin = min(wells, key=lambda w: abs(zeta[w[2]]))
        wi = integrate_well(zeta, B, gII, B_star, *kmin)
        J_direct, tau_direct = physical_JT(wi, V0_ALPHA)
        np.testing.assert_allclose(occ.J, J_direct, rtol=1e-9)
        np.testing.assert_allclose(occ.tau, tau_direct, rtol=1e-9)

    def test_no_well_flags_not_ok(self):
        line = _CosineLine(B0=5.0, eps=0.5, gII=100.0, nfp=1)
        # B_star below the global field min -> no trapped well anywhere -> ok=False
        occ = occupied_action(line, s=0.3, alpha=0.0, B_star=4.0, v0=V0_ALPHA,
                              zeta_hint=0.0, iota=0.0)
        self.assertFalse(occ.ok)
        self.assertEqual(occ.reason, "no_well")


@pytest.mark.unit
class TestGridAndReach(unittest.TestCase):
    """The pure geometry/bookkeeping helpers _dzeta_grid and _reach_at."""

    class _F:
        nfp = 3

    def test_dzeta_grid_formula(self):
        P = GWParams(n_periods=8, pts_per_period=160)
        expect = (P.n_periods * TWO_PI / 3) / (P.n_periods * P.pts_per_period - 1)
        np.testing.assert_allclose(_dzeta_grid(self._F(), P), expect, rtol=1e-12)

    def test_reach_is_running_max_normalized(self):
        # R_W(t) = clip((max_{t'<=t} s - s0)/(1 - s0), 0, 1); uses the running max,
        # not the endpoint value, so a peak-then-fall still scores the peak.
        ts = [0.0, 1.0, 2.0, 3.0]
        ss = [0.3, 0.65, 0.5, 0.4]                       # peak 0.65 at t=1, then falls
        s0 = 0.3
        np.testing.assert_allclose(_reach_at(ts, ss, s0, 3.0), (0.65 - 0.3) / (1 - 0.3))
        np.testing.assert_allclose(_reach_at(ts, ss, s0, 0.0), 0.0)   # only launch seen
        np.testing.assert_allclose(_reach_at(ts, ss, s0, 1.0), (0.65 - 0.3) / (1 - 0.3))

    def test_reach_clips_to_unit_interval(self):
        np.testing.assert_allclose(_reach_at([0.0, 1.0], [0.3, 1.4], 0.3, 1.0), 1.0)  # >edge -> 1
        np.testing.assert_allclose(_reach_at([0.0, 1.0], [0.3, 0.2], 0.3, 1.0), 0.0)  # inward -> 0


@pytest.mark.unit
class TestGWParamsDefaults(unittest.TestCase):
    """The documented defaults must not silently change (they set the published
    metric). branch_continue defaults OFF at the dataclass level; the end-to-end
    entry point flips it ON -- that split is asserted in test_regression."""

    def test_documented_defaults(self):
        P = GWParams()
        self.assertEqual(P.n_periods, 8)
        self.assertEqual(P.pts_per_period, 160)
        np.testing.assert_allclose(P.dalpha, 2.0e-3)
        np.testing.assert_allclose(P.ds_fd, 2.0e-3)
        np.testing.assert_allclose(P.s_edge, 1.0)
        np.testing.assert_allclose(P.s_floor, 1.0e-3)
        self.assertEqual(P.max_accepted_steps, 1200)
        self.assertFalse(P.detrap_fate)                 # published metric untouched by default
        self.assertFalse(P.branch_continue)             # dataclass default OFF


@pytest.mark.unit
class TestAggregateBandAndRegime(unittest.TestCase):
    """The keep/low/high band arithmetic and the regime flag thresholds
    (pure dict-in/dict-out; no field)."""

    def _row(self, R_W, passing=False, unresolved=False, bevent=False, reached=False, w=1.0, bw=None):
        return dict(weight=w, base_weight=(w if bw is None else bw), R_W=R_W,
                    passing_at_birth=passing, unresolved=unresolved,
                    branch_event=bevent, reached_edge=reached)

    def test_keep_is_source_weighted_mean(self):
        rows = [self._row(0.2), self._row(0.4), self._row(0.0, passing=True)]
        a = aggregate(rows)
        np.testing.assert_allclose(a["Gamma_W"], (0.2 + 0.4 + 0.0) / 3)
        np.testing.assert_allclose(a["passing_frac"], 1 / 3)
        np.testing.assert_allclose(a["trapped_frac"], 2 / 3)

    def test_band_brackets_only_unresolved(self):
        rows = [self._row(0.5), self._row(0.3, unresolved=True)]
        a = aggregate(rows)
        # low: unresolved -> 0 ; high: unresolved -> 1 ; keep: unresolved -> its R_W
        np.testing.assert_allclose(a["Gamma_W_low"], (0.5 + 0.0) / 2)
        np.testing.assert_allclose(a["Gamma_W_high"], (0.5 + 1.0) / 2)
        np.testing.assert_allclose(a["Gamma_W"], (0.5 + 0.3) / 2)
        np.testing.assert_allclose(a["unresolved_w_frac"], 0.5)
        # band always brackets the keep estimate
        self.assertLessEqual(a["Gamma_W_low"], a["Gamma_W"])
        self.assertLessEqual(a["Gamma_W"], a["Gamma_W_high"])

    def test_base_weight_is_the_source_normalizer(self):
        # contribution weight `weight` in the numerator, `base_weight` sets the denom.
        rows = [self._row(1.0, w=2.0, bw=1.0), self._row(0.0, w=2.0, bw=1.0)]
        a = aggregate(rows)
        np.testing.assert_allclose(a["Gamma_W"], (2.0 * 1.0 + 2.0 * 0.0) / (1.0 + 1.0))

    def test_all_passing(self):
        a = aggregate([self._row(0.0, passing=True)] * 8)
        np.testing.assert_allclose(a["passing_frac"], 1.0)
        np.testing.assert_allclose(a["trapped_frac"], 0.0)
        np.testing.assert_allclose(a["Gamma_W"], 0.0)
        self.assertEqual(a["regime"], "passing-dominated")
        self.assertFalse(a["gamma_w_valid"])

    def test_all_lost_reaches_edge(self):
        a = aggregate([self._row(1.0, reached=True)] * 6)
        np.testing.assert_allclose(a["Gamma_W"], 1.0)
        np.testing.assert_allclose(a["L_W"], 1.0)
        np.testing.assert_allclose(a["Gamma_W_low"], 1.0)
        np.testing.assert_allclose(a["Gamma_W_high"], 1.0)

    def test_regime_flags(self):
        # passing-dominated: >85% passing
        r = aggregate([self._row(0.0, passing=True)] * 9 + [self._row(0.3)])
        self.assertEqual(r["regime"], "passing-dominated")
        # de-trapping-dominated: >20% branch-event weight (and not passing-dominated)
        r = aggregate([self._row(0.3, bevent=True)] * 3 + [self._row(0.3)] * 7)
        self.assertEqual(r["regime"], "de-trapping-dominated")
        np.testing.assert_allclose(r["branch_event_w_frac"], 0.3)
        # adiabatic-resolved otherwise
        r = aggregate([self._row(0.3)] * 10)
        self.assertEqual(r["regime"], "adiabatic-resolved")
        self.assertTrue(r["gamma_w_valid"])


@pytest.mark.unit
class TestGammaCAngle(unittest.TestCase):
    """Nemov contour inclination gamma_c = (2/pi) atan2(|dJ_da|, |dJ_ds|) in [0,1].
    (The pure-math kernel; the identity with the engine's _rhs is in test_gamma_c.)"""

    @staticmethod
    def gamma_c(dJ_da, dJ_ds):
        if abs(dJ_da) < 1e-300 and abs(dJ_ds) < 1e-300:
            return 0.0
        return (2.0 / np.pi) * np.arctan2(abs(dJ_da), abs(dJ_ds))

    def test_range_and_limits(self):
        self.assertEqual(self.gamma_c(0.0, 0.0), 0.0)                 # both underflow
        np.testing.assert_allclose(self.gamma_c(1.0, 0.0), 1.0)       # dJ_ds->0 superbanana
        np.testing.assert_allclose(self.gamma_c(0.0, 1.0), 0.0)       # dJ_da->0 omnigeneous
        np.testing.assert_allclose(self.gamma_c(1.0, 1.0), 0.5)       # equal -> pi/4 -> 1/2
        for da, ds in [(0.3, 1.7), (2.1, 0.4), (-1.0, 2.0)]:
            g = self.gamma_c(da, ds)
            self.assertGreaterEqual(g, 0.0)
            self.assertLessEqual(g, 1.0)

    def test_metric_prefactor_cancels_in_ratio(self):
        # a positive scalar factor on both derivatives must not change gamma_c
        for c in (0.1, 3.0, 100.0):
            np.testing.assert_allclose(self.gamma_c(0.7 * c, 1.3 * c), self.gamma_c(0.7, 1.3))


@pytest.mark.unit
class TestSelfTestAndProvenance(unittest.TestCase):
    def test_selftest_returns_true(self):
        """The engine's own quadrature self-test (pure numpy/scipy) must pass."""
        self.assertTrue(selftest())

    def test_engine_provenance_hash(self):
        """Guard against silently testing a drifted engine: the imported
        Gamma_W_final must be the exact file the frozen regression values pin to.
        (conftest.ENGINE_HASH / ENGINE_FILE.)"""
        import conftest
        self.assertEqual(conftest._engine_md5(), conftest.ENGINE_HASH)


if __name__ == "__main__":
    unittest.main(verbosity=2)
