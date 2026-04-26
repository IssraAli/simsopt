"""
Targeted regression tests for ``MagneticField.recompute_bell`` and
``Optimizable._any_free_in_lineage`` cache.

The cache invalidation hot path was previously dominated by a per-call
``np.concatenate`` over the lineage of every magnetic field node — see the
"Dipole DOF graph speedup" notes. These tests pin down that:

1. ``recompute_bell`` is allocation-free / near-O(1) on a small ``BiotSavart``
   graph in steady state (loose perf guard, not a strict timing assertion).
2. ``_any_free_in_lineage`` flips correctly across ``fix_all`` and
   ``unfix_all`` on each ``Optimizable`` node in the lineage and on the
   field itself.
3. The cached ``_has_any_free_in_lineage`` answer is invalidated by
   structural / fix-state changes (so a stale ``False`` cannot survive an
   ``unfix_all``).
"""

from __future__ import annotations

import time
import unittest

import numpy as np

from simsopt.field import BiotSavart, Current, coils_via_symmetries
from simsopt.geo import CurveXYZFourier


def _build_small_biotsavart(n_curves: int = 2) -> tuple[BiotSavart, list, list]:
    """Build a small BiotSavart with a handful of coils.

    Parameters
    ----------
    n_curves : int
        Number of independent base curves to symmetrise.

    Returns
    -------
    bs : BiotSavart
        The assembled field.
    curves : list[CurveXYZFourier]
        The base curves (unique DOF-bearing optimizables).
    currents : list[Current]
        The base currents (unique DOF-bearing optimizables).
    """
    rng = np.random.default_rng(0)
    curves: list[CurveXYZFourier] = []
    currents: list[Current] = []
    for _ in range(n_curves):
        c = CurveXYZFourier(quadpoints=20, order=2)
        c.set_dofs(rng.standard_normal(c.dof_size) * 0.1)
        curves.append(c)
        currents.append(Current(1.0e6))
    coils = coils_via_symmetries(curves, currents, 2, True)
    bs = BiotSavart(coils)
    bs.set_points(np.array([[1.0, 0.0, 0.0]]))
    return bs, curves, currents


class RecomputeBellPerfTests(unittest.TestCase):
    """Steady-state ``recompute_bell`` should be near-O(1) per call."""

    def test_recompute_bell_is_fast(self) -> None:
        """10k recompute_bell calls finish in <0.5s on dev hardware.

        This is intentionally a loose guard (not a tight perf test) so it
        does not flake on busy CI runners. The pre-fix implementation took
        ~10s+ on the same workload because it called
        ``np.concatenate`` on every invocation.
        """
        bs, _curves, _currents = _build_small_biotsavart(n_curves=2)
        n_calls = 10_000
        t0 = time.perf_counter()
        for _ in range(n_calls):
            bs.recompute_bell()
        elapsed = time.perf_counter() - t0
        # Loose guard. On dev hw the post-fix path is ~0.01s; we use 0.5s
        # to absorb noise and slow CI machines.
        self.assertLess(
            elapsed,
            0.5,
            msg=(
                f"recompute_bell took {elapsed:.3f}s for {n_calls} calls; "
                "this likely indicates the per-call np.concatenate over "
                "self.dofs_free_status has been re-introduced."
            ),
        )


class AnyFreeInLineageTests(unittest.TestCase):
    """``_any_free_in_lineage`` cache reflects fix_all/unfix_all correctly."""

    def test_flips_with_fix_all_and_unfix_all(self) -> None:
        bs, curves, currents = _build_small_biotsavart(n_curves=1)
        # Initially everything is free.
        self.assertTrue(bs._any_free_in_lineage)
        self.assertTrue(curves[0]._any_free_in_lineage)
        self.assertTrue(currents[0]._any_free_in_lineage)

        # Fix every DOF in the lineage.
        curves[0].fix_all()
        currents[0].fix_all()
        self.assertFalse(bs._any_free_in_lineage)
        self.assertFalse(curves[0]._any_free_in_lineage)
        self.assertFalse(currents[0]._any_free_in_lineage)

        # Unfix one parent — the cache must invalidate and now report True.
        curves[0].unfix_all()
        self.assertTrue(bs._any_free_in_lineage)
        self.assertTrue(curves[0]._any_free_in_lineage)
        # The current is still fully fixed locally, but its own
        # ``_any_free_in_lineage`` only considers its own (empty) lineage,
        # so it stays False.
        self.assertFalse(currents[0]._any_free_in_lineage)

        # Re-fix and confirm the cache flips back to False again.
        curves[0].fix_all()
        self.assertFalse(bs._any_free_in_lineage)

    def test_matches_dofs_free_status(self) -> None:
        """The cached answer agrees with ``np.any(dofs_free_status)``."""
        bs, curves, currents = _build_small_biotsavart(n_curves=2)
        # Before any modification.
        self.assertEqual(
            bs._any_free_in_lineage,
            bool(np.any(bs.dofs_free_status)),
        )
        # After fixing one curve.
        curves[0].fix_all()
        self.assertEqual(
            bs._any_free_in_lineage,
            bool(np.any(bs.dofs_free_status)),
        )
        # After fixing everything.
        for c in curves:
            c.fix_all()
        for cur in currents:
            cur.fix_all()
        self.assertEqual(
            bs._any_free_in_lineage,
            bool(np.any(bs.dofs_free_status)),
        )
        self.assertFalse(bs._any_free_in_lineage)
        # After unfixing again.
        curves[1].unfix_all()
        self.assertEqual(
            bs._any_free_in_lineage,
            bool(np.any(bs.dofs_free_status)),
        )
        self.assertTrue(bs._any_free_in_lineage)

    def test_recompute_bell_clears_cache_only_when_free(self) -> None:
        """``recompute_bell`` should only flush when the lineage has free DOFs.

        We can't directly observe the C++ cache, but we can observe the
        Python-side branch: after ``fix_all`` everywhere, we confirm
        ``_any_free_in_lineage`` is False (so ``recompute_bell`` returns
        without flushing). After ``unfix_all`` it is True again.
        """
        bs, curves, currents = _build_small_biotsavart(n_curves=1)
        # Fix everything. recompute_bell should be the no-op branch.
        curves[0].fix_all()
        currents[0].fix_all()
        self.assertFalse(bs._any_free_in_lineage)
        # Calling does not raise and returns nothing meaningful.
        self.assertIsNone(bs.recompute_bell())

        # Unfix and confirm True.
        curves[0].unfix_all()
        self.assertTrue(bs._any_free_in_lineage)
        self.assertIsNone(bs.recompute_bell())


if __name__ == "__main__":
    unittest.main()
