"""Unit tests for the helpers in the QUASR stage-1 example.

The example script
[examples/3_Advanced/dipole_array_stage1_quasr.py](../../examples/3_Advanced/dipole_array_stage1_quasr.py)
lives outside the installed simsopt package, so we load it manually with
``importlib.util`` (registering it in ``sys.modules`` so ``@dataclass`` can
resolve ``cls.__module__`` for the ``Case`` dataclass).

These tests cover the pure-Python helpers and the I/O glue.  They never run
the optimization, the BoozerLS solver, or any Poincare tracing; those paths
are exercised by the acceptance smoke set documented in the plan.
"""

import csv
import importlib.util
import json
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from simsopt.field import Coil, Current
from simsopt.geo import CurveXYZFourier


EX_PATH = (
    Path(__file__).parent / ".." / ".." / "examples" / "3_Advanced"
    / "dipole_array_stage1_quasr.py"
).resolve()


def _load_example():
    spec = importlib.util.spec_from_file_location("quasr_stage1_example", EX_PATH)
    mod = importlib.util.module_from_spec(spec)
    # Required for @dataclass type resolution.
    sys.modules["quasr_stage1_example"] = mod
    spec.loader.exec_module(mod)
    return mod


quasr = _load_example()


def _make_case(**overrides):
    base = dict(
        name="testcase",
        target="QA",
        plasma_input="input.LandremanPaul2021_QA",
        iota_target=0.19,
    )
    base.update(overrides)
    return quasr.Case(**base)


def _simple_curve(seed: int = 0):
    """A trivial closed curve (DOFs default to zero except radius)."""
    np.random.seed(seed)
    c = CurveXYZFourier(quadpoints=32, order=1)
    dofs = np.zeros(c.dof_size)
    names = c.local_full_dof_names
    name_to_idx = {n: i for i, n in enumerate(names)}
    dofs[name_to_idx["xc(1)"]] = 1.0
    dofs[name_to_idx["ys(1)"]] = 1.0
    c.x = dofs
    return c


class CaseDefaultsTests(unittest.TestCase):
    def test_case_defaults(self):
        c = _make_case()
        self.assertEqual(c.boozer_type, "ls")
        self.assertEqual(c.weights["CONSTRAINT_WEIGHT"], 1.0e4)
        self.assertEqual(c.weights["W_BR"], 1.0e4)
        self.assertEqual(c.weights["W_MR"], 10.0)
        self.assertEqual(c.weights["W_VOL"], 1.0)
        self.assertEqual(c.iota_guess, 0.0)
        self.assertTrue(c.do_poincare)
        self.assertEqual(c.stages, quasr.DEFAULT_STAGES_QA)
        self.assertEqual(c.interp_degree, 4)
        self.assertEqual(c.interp_grid_n, 20)

    def test_case_layout_defaults(self):
        c = _make_case()
        self.assertIsNone(c.n_dipole_phi)
        self.assertIsNone(c.n_dipole_theta)
        self.assertIsNone(c.n_dipole_radial)
        self.assertEqual(c.df_planar_order, 2)
        self.assertTrue(c.prune_inboard)
        self.assertTrue(c.prune_interlinking)

    def test_case_aspect_target_default_is_none(self):
        c = _make_case()
        self.assertIsNone(c.aspect_target)
        self.assertEqual(c.weights["W_AR"], 0.0)
        self.assertEqual(c.grid_layout, "cartesian_bbox")
        self.assertIsNone(c.stellsym_override)
        self.assertEqual(c.df_loop_radius_factor, 0.4)

    def test_case_for_ci_collapses(self):
        c = _make_case()
        ci = quasr.case_for_ci(c)
        self.assertEqual(ci.stages, quasr.CI_STAGES)
        self.assertFalse(ci.do_poincare)
        self.assertFalse(ci.do_taylor_test)
        # Original case must not be mutated (dataclasses.replace returns new).
        self.assertEqual(c.stages, quasr.DEFAULT_STAGES_QA)

    def test_poincare_smoke_shrink(self):
        c = _make_case()
        s = quasr._poincare_smoke_shrink(c)
        self.assertEqual(len(s.stages), 1)
        self.assertLessEqual(s.poincare_nlines, 4)
        self.assertLessEqual(s.poincare_tmax, 400.0)
        self.assertEqual(s.interp_degree, 2)
        self.assertLessEqual(s.interp_grid_n, 12)


class ReactorScaleCasesTests(unittest.TestCase):
    """All default ``CASES`` must point at reactor-scale plasma inputs.

    The TF coil currents from ``initialize_coils`` (``LandremanPaulQA`` /
    ``LandremanPaulQH``) are calibrated for ~5.7 T on axis at major radius
    of order 10 m.  Pairing them with a small-scale plasma input gives a
    non-physical field magnitude on the seed Boozer surface, which is the
    "scale issue" the stage-1 example used to hit on QA.
    """

    _REACTOR_INPUTS = {
        "input.LandremanPaul2021_QA_reactorScale_lowres",
        "input.LandremanPaul2021_QH_reactorScale_lowres",
    }

    def test_all_cases_use_reactor_scale_plasma_inputs(self):
        for c in quasr.CASES:
            self.assertIn(
                c.plasma_input,
                self._REACTOR_INPUTS,
                msg=f"Case {c.name!r} uses non-reactor-scale plasma {c.plasma_input!r}",
            )

    def test_case_for_ci_preserves_plasma_input(self):
        c = _make_case(plasma_input="input.LandremanPaul2021_QA_reactorScale_lowres")
        ci = quasr.case_for_ci(c)
        self.assertEqual(ci.plasma_input, c.plasma_input)

    def test_reactor_cases_present(self):
        names = {c.name for c in quasr.CASES}
        self.assertIn("QA_LP_reactor_DF_24x12", names)
        self.assertIn("QH_LP_reactor_DF_24x12", names)
        qa = next(c for c in quasr.CASES if c.name == "QA_LP_reactor_DF_24x12")
        qh = next(c for c in quasr.CASES if c.name == "QH_LP_reactor_DF_24x12")
        for c in (qa, qh):
            self.assertEqual(c.n_dipole_phi, 24)
            self.assertEqual(c.n_dipole_theta, 12)
            self.assertEqual(c.n_dipole_radial, 1)
            self.assertFalse(c.prune_inboard)
            self.assertFalse(c.prune_interlinking)
            self.assertEqual(c.df_planar_order, 2)
        self.assertEqual(
            qa.plasma_input, "input.LandremanPaul2021_QA_reactorScale_lowres"
        )
        self.assertEqual(
            qh.plasma_input, "input.LandremanPaul2021_QH_reactorScale_lowres"
        )


class DenseGridLayoutTests(unittest.TestCase):
    """Grid layout via ``build_initial_coils`` (no optimization)."""

    def test_dense_grid_case_resolves_n_phi_theta_radial(self):
        c = _make_case(
            name="dense_grid_smoke",
            plasma_input="input.LandremanPaul2021_QA_reactorScale_lowres",
            n_dipole_phi=24,
            n_dipole_theta=12,
            n_dipole_radial=1,
            prune_inboard=False,
            prune_interlinking=False,
            plasma_poff=0.4,
            plasma_coff=0.4,
        )
        setup_dense = quasr.build_initial_coils(c)
        c_sparse = _make_case(
            name="sparse_compare",
            plasma_input="input.LandremanPaul2021_QA_reactorScale_lowres",
            plasma_poff=0.4,
            plasma_coff=0.8,
        )
        setup_sparse = quasr.build_initial_coils(c_sparse)
        n_dense = len(setup_dense["base_wp_curves"])
        n_sparse = len(setup_sparse["base_wp_curves"])
        self.assertGreater(
            n_dense,
            n_sparse,
            msg=f"dense {n_dense} should exceed sparse {n_sparse}",
        )

    def test_legacy_n_dipole_per_axis_still_works(self):
        c = _make_case(
            name="legacy_grid",
            plasma_input="input.LandremanPaul2021_QA",
            n_dipole_per_axis=3,
            prune_inboard=False,
            prune_interlinking=False,
            plasma_poff=0.4,
            plasma_coff=0.8,
        )
        setup = quasr.build_initial_coils(c)
        n = len(setup["base_wp_curves"])
        self.assertGreater(n, 0)
        self.assertLessEqual(n, 3 * 3 * 3)

    def test_prune_toggles_are_respected(self):
        c_pruned = _make_case(
            name="pruned",
            plasma_input="input.LandremanPaul2021_QA",
            n_dipole_per_axis=4,
            prune_inboard=True,
            prune_interlinking=True,
            plasma_poff=0.4,
            plasma_coff=0.8,
        )
        c_full = _make_case(
            name="full",
            plasma_input="input.LandremanPaul2021_QA",
            n_dipole_per_axis=4,
            prune_inboard=False,
            prune_interlinking=False,
            plasma_poff=0.4,
            plasma_coff=0.8,
        )
        n_pruned = len(quasr.build_initial_coils(c_pruned)["base_wp_curves"])
        n_full = len(quasr.build_initial_coils(c_full)["base_wp_curves"])
        self.assertGreater(n_full, n_pruned)

    def test_shell_conformal_grid_yields_paper_density(self):
        c = _make_case(
            name="shell_24x12",
            plasma_input="input.LandremanPaul2021_QA_reactorScale_lowres",
            grid_layout="shell_conformal",
            n_dipole_phi=24,
            n_dipole_theta=12,
            n_dipole_radial=1,
            prune_inboard=False,
            prune_interlinking=False,
            plasma_poff=0.4,
            plasma_coff=0.4,
        )
        setup = quasr.build_initial_coils(c)
        self.assertEqual(len(setup["base_wp_curves"]), 288)


class AuditFixesTests(unittest.TestCase):
    """Stage-1 audit fixes: aspect target, symmetry overrides, shell grid."""

    _LEGACY_NAMES = (
        "QA_LP_reactor_iota0p19",
        "QA_LP_reactor_iota0p42",
        "QH_LP_reactor_iota0p92",
        "QH_LP_reactor_iota1p10",
    )

    def test_legacy_cases_unchanged(self):
        for name in self._LEGACY_NAMES:
            c = next(x for x in quasr.CASES if x.name == name)
            self.assertEqual(c.grid_layout, "cartesian_bbox")
            self.assertIsNone(c.aspect_target)
            self.assertEqual(c.weights["W_AR"], 0.0)
            self.assertEqual(c.dipole_current_init_amp, 0.0)
            self.assertEqual(len(c.stages), 3)
            self.assertEqual(c.bfgs_tol, 1e-10)
            self.assertEqual(c.bfgs_maxiter, 1500)

    def test_reactor_24x12_cases_use_shell_conformal_and_aspect_target(self):
        qa = next(c for c in quasr.CASES if c.name == "QA_LP_reactor_DF_24x12")
        qh = next(c for c in quasr.CASES if c.name == "QH_LP_reactor_DF_24x12")
        for c, ar_tgt in ((qa, 6.0), (qh, 10.0)):
            self.assertEqual(c.grid_layout, "shell_conformal")
            self.assertEqual(c.aspect_target, ar_tgt)
            self.assertEqual(c.weights["W_AR"], 1.0)

    def test_normal_to_quaternion_z_hat_is_identity(self):
        q = quasr._normal_to_quaternion(np.array([0.0, 0.0, 1.0]))
        self.assertAlmostEqual(q[0], 1.0, places=10)
        self.assertAlmostEqual(q[1], 0.0, places=10)
        self.assertAlmostEqual(q[2], 0.0, places=10)
        self.assertAlmostEqual(q[3], 0.0, places=10)

    def test_aspect_bounds_for_case(self):
        c = _make_case(aspect_target=6.0)
        lo, hi = quasr.aspect_bounds_for_case(c)
        self.assertAlmostEqual(lo, 1.8)
        self.assertAlmostEqual(hi, 18.0)
        self.assertEqual(quasr.aspect_bounds_for_case(_make_case()), (1.5, 500.0))

    def test_nfp_override_propagates_to_coils_via_symmetries(self):
        base = _make_case(
            name="nfp_base",
            plasma_input="input.LandremanPaul2021_QA_reactorScale_lowres",
            n_dipole_per_axis=3,
            prune_inboard=False,
            prune_interlinking=False,
            plasma_poff=0.4,
            plasma_coff=0.8,
        )
        over = _make_case(
            name="nfp_over",
            plasma_input="input.LandremanPaul2021_QA_reactorScale_lowres",
            n_dipole_per_axis=3,
            nfp_override=3,
            prune_inboard=False,
            prune_interlinking=False,
            plasma_poff=0.4,
            plasma_coff=0.8,
        )
        n_base = len(quasr.build_initial_coils(base)["wp_coils"])
        n_over = len(quasr.build_initial_coils(over)["wp_coils"])
        self.assertAlmostEqual(n_over / n_base, 3.0 / 2.0, places=6)

    def test_stellsym_override_false_halves_expansion(self):
        base = _make_case(
            name="ss_base",
            plasma_input="input.LandremanPaul2021_QA_reactorScale_lowres",
            n_dipole_per_axis=3,
            prune_inboard=False,
            prune_interlinking=False,
            plasma_poff=0.4,
            plasma_coff=0.8,
        )
        no_ss = _make_case(
            name="ss_off",
            plasma_input="input.LandremanPaul2021_QA_reactorScale_lowres",
            n_dipole_per_axis=3,
            stellsym_override=False,
            prune_inboard=False,
            prune_interlinking=False,
            plasma_poff=0.4,
            plasma_coff=0.8,
        )
        n_base = len(quasr.build_initial_coils(base)["wp_coils"])
        n_off = len(quasr.build_initial_coils(no_ss)["wp_coils"])
        self.assertAlmostEqual(n_off / n_base, 0.5, places=6)
        self.assertFalse(quasr.build_initial_coils(no_ss)["stellsym"])

    def test_stellsym_override_true_on_nonstellsym_plasma_raises(self):
        c = _make_case(
            name="ss_bad",
            plasma_input="input.rotating_ellipse",
            stellsym_override=True,
            n_dipole_per_axis=2,
            prune_inboard=False,
            prune_interlinking=False,
        )
        real_mps = quasr.make_plasma_surface

        def _nonstellsym_plasma(case):
            s, path = real_mps(case)
            s.stellsym = False
            return s, path

        with patch.object(quasr, "make_plasma_surface", side_effect=_nonstellsym_plasma):
            with self.assertRaises(ValueError):
                quasr.build_initial_coils(c)

    def _ncsx_boozer_setup(self):
        from simsopt.configs import get_data
        from simsopt.field import BiotSavart
        from simsopt.geo import BoozerSurface, SurfaceXYZTensorFourier, Volume

        base_curves, base_currents, ma, nfp, bs = get_data("ncsx")
        g0 = (
            2.0
            * np.pi
            * nfp
            * sum(abs(c.get_value()) for c in base_currents)
            * (4 * np.pi * 1e-7 / (2 * np.pi))
        )
        mpol, ntor = 4, 4
        phis = np.linspace(0, 1 / nfp, 2 * ntor + 1, endpoint=False)
        thetas = np.linspace(0, 1.0, 2 * mpol + 1, endpoint=False)
        s = SurfaceXYZTensorFourier(
            mpol=mpol,
            ntor=ntor,
            stellsym=True,
            nfp=nfp,
            quadpoints_phi=phis,
            quadpoints_theta=thetas,
        )
        s.fit_to_curve(ma, 0.1, flip_theta=True)
        vol = Volume(s)
        btot = BiotSavart(bs.coils)
        bsurf = BoozerSurface(btot, s, vol, vol.J(), constraint_weight=1.0e4)
        bsurf.res = {"iota": -0.4, "G": g0, "success": True}
        quasr._populate_PLU_at_current_state(
            bsurf, -0.4, g0, 1.0e4, True
        )
        return bsurf, btot, nfp, g0

    def test_build_objective_skips_J_ar_when_target_is_none(self):
        bsurf, btot, nfp, _g0 = self._ncsx_boozer_setup()
        case = _make_case()
        cur = Current(0.0) * 1e6
        jf, terms = quasr.build_objective(
            bsurf,
            btot,
            [cur],
            16,
            case,
            nfp,
            0.19,
            1.0,
        )
        self.assertNotIn("J_ar", terms)
        _ = jf.J()

    def test_build_objective_includes_J_ar_when_target_set(self):
        bsurf, btot, nfp, _g0 = self._ncsx_boozer_setup()
        case = _make_case(
            aspect_target=6.0,
            weights=dict(
                W_SYMM=1.0,
                W_IOTA=1.0,
                W_MR=10.0,
                W_VOL=1.0,
                W_BR=1.0e4,
                W_AR=1.0,
                W_IMAX_FRAC=1.0e-3,
                CONSTRAINT_WEIGHT=1.0e4,
            ),
        )
        cur = Current(0.0) * 1e6
        jf, terms = quasr.build_objective(
            bsurf,
            btot,
            [cur],
            16,
            case,
            nfp,
            0.19,
            1.0,
        )
        self.assertIn("J_ar", terms)
        j_ar = float(terms["J_ar"].J())
        self.assertGreater(j_ar, 0.0)
        j_total = float(jf.J())
        self.assertGreater(j_total, j_ar)

    def test_w_vol_zeroed_when_aspect_target_set(self):
        bsurf, btot, nfp, _g0 = self._ncsx_boozer_setup()
        cur = Current(0.0) * 1e6
        wts = dict(
            W_SYMM=1.0,
            W_IOTA=1.0,
            W_MR=10.0,
            W_VOL=1.0,
            W_BR=1.0e4,
            W_AR=1.0,
            W_IMAX_FRAC=1.0e-3,
            CONSTRAINT_WEIGHT=1.0e4,
        )
        jf_ar, terms_ar = quasr.build_objective(
            bsurf, btot, [cur], 16,
            _make_case(aspect_target=6.0, weights=wts),
            nfp, 0.19, 1.0,
        )
        jf_no_ar, terms_no_ar = quasr.build_objective(
            bsurf, btot, [cur], 16,
            _make_case(weights=dict(wts, W_AR=0.0)),
            nfp, 0.19, 1.0,
        )
        w_vol = 1.0
        j_vol = float(terms_ar["J_vol"].J())
        j_rest_ar = (
            float(jf_ar.J())
            - float(terms_ar["J_ar"].J())
            - w_vol * j_vol
        )
        j_rest_no_ar = float(jf_no_ar.J()) - w_vol * j_vol
        self.assertAlmostEqual(j_rest_ar, j_rest_no_ar, places=6)

    def test_w_vol_unchanged_when_aspect_target_none(self):
        bsurf, btot, nfp, _g0 = self._ncsx_boozer_setup()
        cur = Current(0.0) * 1e6
        jf, terms = quasr.build_objective(
            bsurf, btot, [cur], 16, _make_case(), nfp, 0.19, 1.0,
        )
        j_vol_w = float(terms["J_vol"].J())
        j_no_vol = (
            float(jf.J())
            - float(_make_case().weights.get("W_VOL", 1.0)) * j_vol_w
        )
        jf2, _ = quasr.build_objective(
            bsurf,
            btot,
            [cur],
            16,
            _make_case(weights=dict(_make_case().weights, W_VOL=0.0)),
            nfp,
            0.19,
            1.0,
        )
        self.assertAlmostEqual(j_no_vol, float(jf2.J()), places=6)

    def test_seed_a_edge_uses_aspect_target(self):
        c_dense = next(
            x for x in quasr.CASES if x.name == "QA_LP_reactor_DF_24x12"
        )
        s, _ = quasr.make_plasma_surface(c_dense)
        r0_target = float(c_dense.R0_factor * s.get_rc(0, 0))
        plasma_minor = float(s.minor_radius())
        a_edge_final = min(
            r0_target / float(c_dense.aspect_target), 1.05 * plasma_minor
        )
        self.assertAlmostEqual(a_edge_final, r0_target / 6.0, places=3)
        a_legacy = 0.25 * plasma_minor
        self.assertGreater(a_edge_final, a_legacy)  # aspect-target seed is larger

        c_legacy = next(
            x for x in quasr.CASES if x.name == "QA_LP_reactor_iota0p19"
        )
        s2, _ = quasr.make_plasma_surface(c_legacy)
        a_legacy = c_legacy.a_edge_frac * float(s2.minor_radius())
        self.assertIsNone(c_legacy.aspect_target)
        self.assertGreater(a_legacy, 0.0)

    def test_dipole_current_init_amp_zero_keeps_zero_currents(self):
        c = _make_case(dipole_current_init_amp=0.0)
        setup = quasr.build_initial_coils(c)
        vals = [abs(c.get_value()) for c in setup["base_wp_currents"]]
        self.assertTrue(all(v == 0.0 for v in vals))

    def test_dipole_current_init_amp_creates_bounded_nonzero_currents(self):
        c = _make_case(
            name="amp_test",
            plasma_input="input.LandremanPaul2021_QA_reactorScale_lowres",
            dipole_current_init_amp=1.0e-3,
            n_dipole_per_axis=4,
            prune_inboard=False,
            prune_interlinking=False,
        )
        setup = quasr.build_initial_coils(c)
        vals = np.array([abs(c.get_value()) for c in setup["base_wp_currents"]])
        self.assertTrue(np.all(vals > 0))
        self.assertTrue(np.all(vals <= 1.0e3))
        self.assertGreater(np.std(vals), 0.0)

    def test_dense_24x12_cases_have_current_init_amp(self):
        for name in ("QA_LP_reactor_DF_24x12", "QH_LP_reactor_DF_24x12"):
            c = next(x for x in quasr.CASES if x.name == name)
            self.assertEqual(c.dipole_current_init_amp, 1.0e-3)

    def test_dense_24x12_cases_iota_target_0p5(self):
        for name in ("QA_LP_reactor_DF_24x12", "QH_LP_reactor_DF_24x12"):
            c = next(x for x in quasr.CASES if x.name == name)
            self.assertEqual(c.iota_target, 0.5)
            self.assertEqual(c.iota_guess, 0.25)

    def test_legacy_cases_keep_amp_zero(self):
        for name in self._LEGACY_NAMES:
            c = next(x for x in quasr.CASES if x.name == name)
            self.assertEqual(c.dipole_current_init_amp, 0.0)

    def test_default_stages_are_two_step(self):
        self.assertEqual(len(quasr.DEFAULT_STAGES_QA), 2)
        self.assertEqual(len(quasr.DEFAULT_STAGES_QH), 2)
        self.assertEqual(quasr.DEFAULT_STAGES_QA[0][0], 4)
        self.assertEqual(quasr.DEFAULT_STAGES_QA[-1][0], 8)
        self.assertEqual(quasr.DEFAULT_STAGES_QH[-1][0], 8)

    def test_legacy_cases_keep_three_stage_tuples(self):
        for name in self._LEGACY_NAMES:
            c = next(x for x in quasr.CASES if x.name == name)
            self.assertEqual(len(c.stages), 3)

    def test_make_boozer_surface_sets_limited_memory(self):
        from simsopt.configs import get_data
        from simsopt.field import BiotSavart

        _curves, _currents, ma, nfp, bs = get_data("ncsx")
        btot = BiotSavart(bs.coils)
        case = _make_case()
        bsurf = quasr.make_boozer_surface(
            btot, ma, 0.1, nfp, 4, 4, case, stellsym=True
        )
        self.assertTrue(bsurf.options.get("limited_memory"))
        self.assertEqual(bsurf.options.get("bfgs_tol"), case.bfgs_tol)
        self.assertEqual(bsurf.options.get("bfgs_maxiter"), case.bfgs_maxiter)
        self.assertEqual(case.bfgs_tol, 1e-8)
        self.assertEqual(case.bfgs_maxiter, 500)


class FlatDictAndJsonTests(unittest.TestCase):
    def test_case_to_flat_dict(self):
        c = _make_case()
        d = quasr.case_to_flat_dict(c)
        self.assertIsInstance(d["stages_repr"], str)
        self.assertNotIn("weights", d)
        self.assertNotIn("stages", d)
        self.assertEqual(d["weight_W_SYMM"], 1.0)
        self.assertEqual(d["weight_CONSTRAINT_WEIGHT"], 1.0e4)

    def test_metrics_for_json_serialises(self):
        m = {
            "a_float64": np.float64(1.5),
            "an_int64": np.int64(7),
            "an_array": np.array([1.0, 2.0, 3.0]),
            "a_bool": True,
            "a_str": "ok",
        }
        out = quasr.metrics_for_json(m)
        s = json.dumps(out)
        loaded = json.loads(s)
        self.assertEqual(loaded["a_float64"], 1.5)
        self.assertEqual(loaded["an_int64"], 7)
        self.assertEqual(loaded["an_array"], [1.0, 2.0, 3.0])
        self.assertTrue(loaded["a_bool"])
        self.assertEqual(loaded["a_str"], "ok")


class ScanSummaryTests(unittest.TestCase):
    def test_write_scan_summary_union_columns(self):
        import tempfile

        rows = [
            {"name": "A", "x": 1.0, "shared": 0.1},
            {"name": "B", "y": 2.0, "shared": 0.2},
        ]
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "scan.csv"
            quasr.write_scan_summary(rows, path)
            self.assertTrue(path.exists())
            with open(path, newline="") as fp:
                reader = csv.DictReader(fp)
                header = reader.fieldnames
                read_rows = list(reader)
        self.assertEqual(header, sorted({"name", "x", "y", "shared"}))
        self.assertEqual(read_rows[0]["name"], "A")
        self.assertEqual(read_rows[0]["y"], "")
        self.assertEqual(read_rows[1]["name"], "B")
        self.assertEqual(read_rows[1]["x"], "")
        self.assertEqual(read_rows[1]["shared"], "0.2")


class PickCasesTests(unittest.TestCase):
    def _args(self, *, case_idx=None, mpi=False):
        return types.SimpleNamespace(case_idx=case_idx, mpi=mpi)

    def test_serial_returns_all(self):
        cases = quasr.pick_cases(self._args(), rank=0, size=1)
        self.assertEqual([c.name for c in cases], [c.name for c in quasr.CASES])

    def test_case_idx_single(self):
        cases = quasr.pick_cases(self._args(case_idx=1), rank=0, size=1)
        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0].name, quasr.CASES[1].name)

    def test_mpi_round_robin_partition(self):
        if not quasr.HAS_MPI:
            self.skipTest("mpi4py not installed")
        a0 = quasr.pick_cases(self._args(mpi=True), rank=0, size=2)
        a1 = quasr.pick_cases(self._args(mpi=True), rank=1, size=2)
        names0 = {c.name for c in a0}
        names1 = {c.name for c in a1}
        self.assertEqual(names0 & names1, set())
        self.assertEqual(
            names0 | names1, {c.name for c in quasr.CASES}
        )


class GFromTfTests(unittest.TestCase):
    def test_matches_mu0_sum_abs_I(self):
        I = 1.23e6
        c1 = Coil(_simple_curve(0), Current(I))
        c2 = Coil(_simple_curve(1), Current(-I))
        g = quasr.G_from_TF([c1, c2])
        self.assertAlmostEqual(g, quasr.MU0 * 2.0 * I, places=6)


class CurrentMagnitudeSquaredTests(unittest.TestCase):
    def test_J_matches_I_squared_for_plain_current(self):
        cur = Current(2.5e5)
        cms = quasr.CurrentMagnitudeSquared(cur)
        self.assertAlmostEqual(cms.J(), (2.5e5) ** 2, places=0)

    def test_J_matches_I_squared_for_scaled_current(self):
        cur = Current(3.0) * 1e6
        cms = quasr.CurrentMagnitudeSquared(cur)
        self.assertAlmostEqual(cms.J(), (3.0e6) ** 2, places=0)

    def test_dJ_taylor_4pt(self):
        cur = Current(1.5)
        scaled = cur * 1e6
        cms = quasr.CurrentMagnitudeSquared(scaled)

        # ``@derivative_dec`` makes ``dJ()`` return the full reverse-mode
        # gradient w.r.t. the free DOFs of ``cms``.
        x0 = cur.x.copy()
        g = np.asarray(cms.dJ())
        h = np.array([1.0])
        analytical = float(np.dot(g, h))

        results = []
        for eps in [1e-3, 1e-4, 1e-5]:
            cur.x = x0 + 2 * eps * h
            J1 = cms.J()
            cur.x = x0 + eps * h
            J2 = cms.J()
            cur.x = x0 - eps * h
            J3 = cms.J()
            cur.x = x0 - 2 * eps * h
            J4 = cms.J()
            fd = (-J1 + 8 * J2 - 8 * J3 + J4) / (12 * eps)
            results.append((eps, fd))
        cur.x = x0

        for eps, fd in results:
            rel = abs(fd - analytical) / max(abs(analytical), 1e-30)
            self.assertLess(rel, 5e-6, msg=f"eps={eps} fd={fd} analytical={analytical}")


class SaveCoilsVtkTests(unittest.TestCase):
    def test_writes_vtu_file(self):
        import tempfile

        coil = Coil(_simple_curve(0), Current(1.0e5))
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            quasr.save_coils_vtk([coil], out, "smoke")
            vtu = out / "coils_smoke.vtu"
            self.assertTrue(vtu.exists(), f"missing {vtu}")
            self.assertGreater(vtu.stat().st_size, 0)


class _FakeSurface:
    """Tiny stub that satisfies the attributes ``validate_boozer_state`` reads."""

    def __init__(self, vol=1.0, nfp=2, ar=5.0, raise_si=False, si=False):
        self._vol = vol
        self.nfp = nfp
        self._ar = ar
        self._raise_si = raise_si
        self._si = si

    def volume(self):
        return self._vol

    def aspect_ratio(self):
        return self._ar

    def is_self_intersecting(self, angle=0.0):
        if self._raise_si:
            raise RuntimeError("bentley_ottmann missing")
        return self._si


class _FakeBsurf:
    def __init__(self, res, surface):
        self.res = res
        self.surface = surface


class ValidateBoozerStateTests(unittest.TestCase):
    def test_accepts_good_state(self):
        res = {"success": True, "iota": -0.4, "G": 5.0}
        bs = _FakeBsurf(res, _FakeSurface(vol=2.5, nfp=2, raise_si=True, ar=5.0))
        ok, why = quasr.validate_boozer_state(bs, G0=5.0)
        self.assertTrue(ok, why)
        self.assertEqual(why, "ok")

    def test_rejects_solver_exception(self):
        bs = _FakeBsurf(
            {"success": False, "exception": "blew up"},
            _FakeSurface(),
        )
        ok, why = quasr.validate_boozer_state(bs, G0=5.0)
        self.assertFalse(ok)
        self.assertIn("exception", why)

    def test_accepts_success_false_but_physical_state(self):
        # scipy's L-BFGS-B routinely returns success=False on a perfectly
        # usable Boozer surface at our tight bfgs_tol.  Validation must
        # accept based on the physical state alone.
        bs = _FakeBsurf(
            {"success": False, "iota": -0.4, "G": 5.0},
            _FakeSurface(vol=2.5, nfp=2, raise_si=True, ar=5.0),
        )
        ok, why = quasr.validate_boozer_state(bs, G0=5.0)
        self.assertTrue(ok, why)

    def test_rejects_huge_iota(self):
        bs = _FakeBsurf({"success": True, "iota": 1.0e8, "G": 5.0}, _FakeSurface())
        ok, why = quasr.validate_boozer_state(bs, G0=5.0)
        self.assertFalse(ok)
        self.assertIn("iota", why)

    def test_rejects_bad_G_band(self):
        # G is 100x larger than G0 -> outside [0.3 G0, 3 G0]
        bs = _FakeBsurf({"success": True, "iota": 0.1, "G": 500.0}, _FakeSurface())
        ok, why = quasr.validate_boozer_state(bs, G0=5.0)
        self.assertFalse(ok)
        self.assertIn("G", why)

    def test_rejects_self_intersecting(self):
        bs = _FakeBsurf(
            {"success": True, "iota": 0.1, "G": 5.0},
            _FakeSurface(si=True, raise_si=False),
        )
        ok, why = quasr.validate_boozer_state(bs, G0=5.0)
        self.assertFalse(ok)
        self.assertIn("self", why)


class FindMagneticAxisTests(unittest.TestCase):
    def test_picard_converges_to_circle_for_toroidal_field(self):
        # Pure toroidal field B = B0 * R0 / R e_phi -> magnetic axis is any
        # circle of constant R (and Z=0).  Starting from a slightly-off guess,
        # Picard should converge to the same (R, Z).
        from simsopt.field import ToroidalField

        btot = ToroidalField(R0=10.0, B0=5.0)
        axis = quasr.find_magnetic_axis(
            btot, R0_guess=10.0, nfp=2, Z0_guess=0.0,
            order=4, npts=64, n_iter=10, tol=1e-9,
        )
        g = axis.gamma()
        R = np.sqrt(g[:, 0] ** 2 + g[:, 1] ** 2)
        Z = g[:, 2]
        self.assertTrue(np.allclose(R, 10.0, atol=1e-6),
                        msg=f"R range [{R.min():.6f}, {R.max():.6f}]")
        self.assertTrue(np.allclose(Z, 0.0, atol=1e-6),
                        msg=f"max|Z|={np.max(np.abs(Z)):.2e}")

    def test_picard_nonconvergence_message_not_integration_failure(self):
        """Picard exhaustion prints a distinct message from integration failure."""
        import io
        from contextlib import redirect_stdout
        from unittest.mock import MagicMock, patch

        from simsopt.field import ToroidalField

        btot = ToroidalField(R0=10.0, B0=5.0)

        def fake_solve_ivp(rhs, t_span, y0, **kwargs):
            sol = MagicMock()
            sol.success = True
            te = kwargs.get("t_eval")
            sol.t = np.asarray(te) if te is not None else np.linspace(
                t_span[0], t_span[1], 8, endpoint=False
            )
            n = len(sol.t)
            sol.y = np.zeros((2, n))
            sol.y[0, :] = y0[0] + 0.05
            sol.y[1, :] = y0[1] + 0.05
            return sol

        buf = io.StringIO()
        with patch.object(quasr, "solve_ivp", side_effect=fake_solve_ivp):
            with redirect_stdout(buf):
                axis = quasr.find_magnetic_axis(
                    btot,
                    R0_guess=10.0,
                    nfp=2,
                    n_iter=2,
                    tol=1e-12,
                    npts=8,
                    order=4,
                )
        out = buf.getvalue()
        self.assertIn("Picard did not converge", out)
        self.assertIn("last residual=", out)
        self.assertNotIn("integration failed", out)
        g = axis.gamma()
        R = np.sqrt(g[0, 0] ** 2 + g[0, 1] ** 2)
        self.assertAlmostEqual(R, 10.0, places=5)

    def test_helical_orbit_return_map_not_mean_map(self):
        """On a helical (R,Z)(phi) axis, return-map residual is zero but mean-map is not."""
        R0, a, b = 2.0, 0.15, 0.08
        phis = np.linspace(0.0, 2.0 * np.pi, 512, endpoint=False)
        Rs = R0 + a * np.cos(2.0 * phis)
        Zs = b * np.sin(2.0 * phis)
        R_start, Z_start = float(Rs[0]), float(Zs[0])
        R_end = float(R0 + a * np.cos(4.0 * np.pi))
        Z_end = float(b * np.sin(4.0 * np.pi))
        ret_res = abs(R_end - R_start) + abs(Z_end - Z_start)
        mean_res = abs(float(np.mean(Rs)) - R_start) + abs(float(np.mean(Zs)) - Z_start)
        self.assertLess(ret_res, 1e-10)
        self.assertGreater(mean_res, 1e-3)

    def test_picard_converges_for_reactor_qa_tf_field(self):
        """Return-map Picard on LP QA TF coils should converge (not fall back)."""
        import io
        from contextlib import redirect_stdout

        from simsopt.field import BiotSavart

        case = next(c for c in quasr.CASES if c.name == "QA_LP_reactor_DF_24x12")
        setup = quasr.build_initial_coils(case)
        btot_tf = BiotSavart(setup["coils_TF"])
        buf = io.StringIO()
        with redirect_stdout(buf):
            axis = quasr.find_magnetic_axis(
                btot_tf,
                setup["plasma"].get_rc(0, 0),
                setup["nfp"],
                tol=1e-5,
                n_iter=50,
            )
        out = buf.getvalue()
        self.assertNotIn("Picard did not converge", out)
        self.assertNotIn("integration failed", out)
        g = axis.gamma()
        R = np.sqrt(g[:, 0] ** 2 + g[:, 1] ** 2)
        self.assertGreater(float(np.mean(R)), 9.0)
        self.assertLess(float(np.mean(R)), 11.0)


class SolveLsNoNewtonTests(unittest.TestCase):
    def test_run_code_does_not_invoke_newton(self):
        """``_install_ls_no_newton`` must not touch the Newton polish.

        We monkey-patch ``BoozerSurface.minimize_boozer_penalty_constraints_newton``
        to raise; the absence of that ``AssertionError`` after a patched
        ``run_code`` proves Newton was never invoked.
        """
        from simsopt.configs import get_data
        from simsopt.geo import BoozerSurface as _BS
        from simsopt.geo import SurfaceXYZTensorFourier, Volume

        original_newton = _BS.minimize_boozer_penalty_constraints_newton

        def boom(self, *a, **kw):
            raise AssertionError("Newton polish must not be invoked")

        _BS.minimize_boozer_penalty_constraints_newton = boom
        try:
            base_curves, base_currents, ma, nfp, bs = get_data("ncsx")
            G0 = (
                2.0
                * np.pi
                * nfp
                * sum(abs(c.get_value()) for c in base_currents)
                * (4 * np.pi * 1e-7 / (2 * np.pi))
            )
            mpol, ntor = 4, 4
            phis = np.linspace(0, 1 / nfp, 2 * ntor + 1, endpoint=False)
            thetas = np.linspace(0, 1.0, 2 * mpol + 1, endpoint=False)
            s = SurfaceXYZTensorFourier(
                mpol=mpol, ntor=ntor, stellsym=True, nfp=nfp,
                quadpoints_phi=phis, quadpoints_theta=thetas,
            )
            s.fit_to_curve(ma, 0.1, flip_theta=True)
            vol = Volume(s)
            bsurf = _BS(bs, s, vol, vol.J(), constraint_weight=1.0e4)
            bsurf.options["verbose"] = False
            bsurf.options["bfgs_tol"] = 1e-10
            bsurf.options["bfgs_maxiter"] = 200
            quasr._install_ls_no_newton(bsurf, G0_ref=G0)
            bsurf.need_to_run_code = True
            res = bsurf.run_code(-0.4, G=G0)
            self.assertTrue(res is not None)
            self.assertIn("iota", res)
        finally:
            _BS.minimize_boozer_penalty_constraints_newton = original_newton


class MakeFunRevertsTests(unittest.TestCase):
    def test_make_fun_reverts_on_invalid_state(self):
        """When ``validate_boozer_state`` fails, ``make_fun`` returns
        ``J = max(10 prev_J, 1e6)``, ``grad = -prev_grad`` and restores the
        previous surface DOFs/iota/G.

        We use a tiny ``_FakeBsurf`` plus a hand-rolled ``_FakeJF`` and toggle
        the bsurf's ``success`` flag mid-call to simulate degeneracy.
        """
        good_res = {"success": True, "iota": -0.4, "G": 5.0}
        bs = _FakeBsurf(dict(good_res), _FakeSurface(vol=2.5, nfp=2,
                                                     raise_si=True, ar=5.0))
        bs.surface.x = np.array([1.0, 2.0, 3.0])

        class _FakeJF:
            def __init__(self):
                self.x = np.array([0.0, 0.0, 0.0])
                self._call = 0
            def J(self):
                return 1.0
            def dJ(self):
                return np.array([0.1, 0.2, 0.3])

        JF = _FakeJF()

        # The make_fun closure reads bsurf.surface.x.copy() during init; provide a
        # surface object with .x attribute.
        class _SurfWithX:
            def __init__(self):
                self.x = np.array([10.0, 20.0, 30.0])
                self.nfp = 2
            def volume(self): return 2.5
            def aspect_ratio(self): return 5.0
            def is_self_intersecting(self, angle=0.0): raise RuntimeError()

        bs.surface = _SurfWithX()
        state = {"sdofs": bs.surface.x.copy(),
                 "iota": bs.res["iota"],
                 "G": bs.res["G"]}
        fun = quasr.make_fun(JF, bs, state, G0_ref=5.0, verbose=False)
        # First call: good state -> accept
        j, g = fun(np.array([0.0, 0.0, 0.0]))
        self.assertEqual(j, 1.0)
        # Now corrupt the bsurf state so validate_boozer_state rejects:
        # bump iota outside the allowed band.
        bs.res["iota"] = 1.0e9
        j2, g2 = fun(np.array([0.1, 0.0, 0.0]))
        self.assertGreaterEqual(j2, 10.0 * 1.0)
        # Surface DOFs reverted
        np.testing.assert_array_equal(bs.surface.x, np.array([10.0, 20.0, 30.0]))
        # iota reverted to last accepted value
        self.assertEqual(bs.res["iota"], -0.4)
        self.assertEqual(bs.res["G"], 5.0)


if __name__ == "__main__":
    unittest.main()
