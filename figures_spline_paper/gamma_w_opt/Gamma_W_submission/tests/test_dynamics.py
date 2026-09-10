r"""Unit tests for the Whitham characteristic control logic (integrate_characteristic).

These lock the ENGINE'S DECISION LADDER -- the launch guards, the confined/lost
termination, the step cap, and (the scientifically important part) the keep vs
branch_continue dispatch at a de-trapping event -- as regression tests, WITHOUT
firm3d. The trick (from the review's B-series mocks): monkeypatch the two field-
touching functions `_rhs` and `_continue_branch` so we script exactly the sequence
of drifts / events the integrator sees, then assert on the resulting GWResult and
on how one row lands in aggregate's [low, high] band.

`integrate_characteristic` looks up `_rhs`/`_continue_branch` as module globals at
call time, so monkeypatching `Gamma_W_final._rhs` reaches the closures inside it.
A tiny FakeField supplies only what the integrator itself reads: `nfp` (for the
dzeta grid) and the benign helicity/profiles/modB used by the end-of-run
`_detrap_ripple` bookkeeping (which returns 0 here -> never reclassifies).
"""
import numpy as np
import pytest

import Gamma_W_final as gw
from Gamma_W_final import (integrate_characteristic, GWParams, OccResult, aggregate,
                           V0_ALPHA, BRANCH_EVENT, TAU_GUARD, FD_MISMATCH, STEP_CAP,
                           DETRAP_PASSING, OK_CONFINED, OK_LOST, NO_LAUNCH,
                           UNRESOLVED, EVENT, LOST)

pytestmark = pytest.mark.unit


# ----------------------------------------------------------- test scaffolding
class FakeField:
    """Only what integrate_characteristic reads directly. All field-line physics is
    supplied through the monkeypatched _rhs/_continue_branch, so line_B is never
    called. helicity/profiles/modB feed the harmless end-of-run ripple bookkeeping."""
    nfp = 1

    def helicity(self, s=0.5):
        return (1, 0)

    def profiles(self, s):
        return (1.0, 0.0, 0.3)

    def modB(self, s, theta, zeta):
        return np.ones(np.atleast_1d(np.asarray(theta, float)).shape)


def _base(zeta_min=0.0, tau=1.0e-6, B_min=0.9):
    """A valid occupied-well result carrying just the fields the integrator uses."""
    return OccResult(True, 1.0, tau, zeta_min, zeta_min - 1.0, zeta_min + 1.0,
                     B_min, 1, "ok")


# A single field-period of leeway on both drifts; t_star/dt chosen so the mocks take
# ~tens of accepted steps before the scripted event.
def _run(rhs, P, cont=None, s0=0.3, alpha0=0.0, monkeypatch=None):
    monkeypatch.setattr(gw, "_rhs", rhs)
    if cont is not None:
        monkeypatch.setattr(gw, "_continue_branch", cont)
    return integrate_characteristic(FakeField(), s0, alpha0, B_star=1.0, v0=V0_ALPHA,
                                    kap=1.0, zeta0=0.0, P=P, bmin=0.5, bmax=1.5)


def _rhs_const(sd):
    """A _rhs that always returns a constant radial drift sd (no events)."""
    def rhs(field, s, alpha, B_star, v0, kap, zh, P, bmin, bmax, dzg, tau_med):
        return sd, 0.0, _base(), "ok"
    return rhs


def _rhs_until(threshold, kind):
    """Outward drift while s < threshold; then the scripted event above it.
    kind='branch' -> de-trapping (base None); kind='fd' -> FD stencil fail (base kept)."""
    def rhs(field, s, alpha, B_star, v0, kap, zh, P, bmin, bmax, dzg, tau_med):
        if s < threshold:
            return 5.0, 0.0, _base(), "ok"
        if kind == "branch":
            return 0.0, 0.0, None, BRANCH_EVENT
        return 0.0, 0.0, _base(), FD_MISMATCH
    return rhs


_LONG = dict(t_star=1.0, dt_max=1.0e-2, max_accepted_steps=1200)


# ----------------------------------------------------------- launch guards
def test_launch_no_launch(monkeypatch):
    """A de-trapping event on the very first evaluation is NO_LAUNCH (never trapped),
    n_steps=0, R_W=0, flagged unresolved but not a branch_event."""
    def rhs(field, s, a, B_star, v0, kap, zh, P, bmin, bmax, dzg, tm):
        return 0.0, 0.0, None, BRANCH_EVENT
    res = _run(rhs, GWParams(**_LONG), monkeypatch=monkeypatch)
    assert res.status == NO_LAUNCH
    assert res.n_steps == 0
    assert res.R_W == 0.0
    assert res.unresolved is True
    assert res.branch_event is False
    assert res.n_branch == 0


def test_launch_tau_guard_passes_through(monkeypatch):
    """A non-branch launch failure keeps its own status (not remapped to NO_LAUNCH)."""
    def rhs(field, s, a, B_star, v0, kap, zh, P, bmin, bmax, dzg, tm):
        return 0.0, 0.0, None, TAU_GUARD
    res = _run(rhs, GWParams(**_LONG), monkeypatch=monkeypatch)
    assert res.status == TAU_GUARD
    assert res.unresolved is True


# ----------------------------------------------------------- confined / lost / cap
def test_inward_drift_is_confined(monkeypatch):
    """A steady inward drift falls below s_floor -> OK_CONFINED, never reaches edge,
    R_W=0 (s never exceeds s0)."""
    res = _run(_rhs_const(-5.0), GWParams(**_LONG), monkeypatch=monkeypatch)
    assert res.status == OK_CONFINED
    assert res.reached_edge is False
    assert res.unresolved is False
    assert res.R_W == 0.0
    assert res.n_branch == 0


def test_outward_drift_reaches_edge(monkeypatch):
    """A steady strong outward drift crosses s_edge -> OK_LOST, reached_edge, R_W=1."""
    res = _run(_rhs_const(+5.0), GWParams(**_LONG), monkeypatch=monkeypatch)
    assert res.status == OK_LOST
    assert res.reached_edge is True
    assert res.status in LOST
    assert res.R_W == 1.0


def test_step_cap_when_neither_lost_nor_confined(monkeypatch):
    """A slow monotone outward drift that neither reaches the edge nor turns around
    exhausts max_accepted_steps -> STEP_CAP (unresolved)."""
    P = GWParams(t_star=1.0, dt_max=1.0e-2, max_accepted_steps=5)
    res = _run(_rhs_const(+1.0), P, monkeypatch=monkeypatch)
    assert res.status == STEP_CAP
    assert res.unresolved is True
    assert res.reached_edge is False


def test_bounded_excursion_early_stops_at_first_peak(monkeypatch):
    r"""B3 (early-stop): a librating bounce-center (s_dot = A cos(alpha), alpha_dot=omega)
    rises to a first peak s0+A/omega, then descends; the descent/stall guard stops the
    integration and s_max records that FIRST excursion -- it does not run on to catch a
    later, possibly larger, excursion. This is the documented adiabatic-reach
    approximation. We assert the bounded, confined outcome, not an exact peak value."""
    A, OMEGA = 0.3, 1.0
    def rhs(field, s, alpha, B_star, v0, kap, zh, P, bmin, bmax, dzg, tm):
        return A * np.cos(alpha), OMEGA, _base(), "ok"
    P = GWParams(t_star=3.0, dt_max=1.0e-2, max_accepted_steps=1200)
    res = _run(rhs, P, monkeypatch=monkeypatch)
    assert res.status == OK_CONFINED
    assert res.reached_edge is False
    assert 0.0 < res.R_W < 1.0
    # first-excursion amplitude ~ s0 + A/omega = 0.6 (bounded; not the whole domain)
    assert 0.5 < res.s_max < 0.65
    assert res.n_branch == 0


# ----------------------------------------------------------- keep vs branch_continue
def test_detrapping_keep_is_unresolved_branch_event(monkeypatch):
    """keep metric (branch_continue OFF): a de-trapping event is an UNRESOLVED
    branch_event -- s_max preserved above launch, no continuation, and the reach is
    the honest running-max normalization."""
    P = GWParams(branch_continue=False, **_LONG)
    res = _run(_rhs_until(0.5, "branch"), P, monkeypatch=monkeypatch)
    assert res.status == BRANCH_EVENT
    assert res.unresolved is True
    assert res.branch_event is True
    assert res.n_branch == 0
    assert 0.3 < res.s_max < 1.0                        # preserved, above launch, below edge
    np.testing.assert_allclose(res.R_W, np.clip((res.s_max - 0.3) / (1 - 0.3), 0, 1))


def test_detrapping_bc_continues_into_descendant(monkeypatch):
    """branch_continue ON with a descendant well available: the characteristic
    continues (n_branch increments -- ONLY bc does this) instead of stopping, and here
    the descendant drift carries it to the edge."""
    P = GWParams(branch_continue=True, **_LONG)
    def cont(field, s, alpha, B_star, v0, kap, zh, P_, bmin, bmax, exclude=None):
        return 5.0, 0.0, _base(zeta_min=0.1)            # outward descendant drift
    res = _run(_rhs_until(0.5, "branch"), P, cont=cont, monkeypatch=monkeypatch)
    assert res.n_branch >= 1                            # continuation happened (bc only)
    assert res.n_cont_rescues >= res.n_branch           # rescue tally is a superset of n_branch
    assert res.status != BRANCH_EVENT                   # resolved, not left as an event
    assert res.status in LOST and res.reached_edge is True


def test_substage_rescue_counted_by_n_cont_rescues_not_n_branch(monkeypatch):
    """n_branch-undercount fix (Fable's July-13 finding): a continuation that rescues INSIDE an
    RK4 substage changes the accepted step but is invisible to n_branch -- `cont_flag` is reset
    on every rhs_cont call and read only at the top of the loop, so only top-of-loop rescues
    reach `n_branch += 1`. The persistent `n_cont_rescues` tally catches substage rescues too.
    Here the de-trapping is scripted to fire on the first RK4 s2 substage (the 3rd _rhs call:
    launch, iter-1 top, iter-1 s2), so n_branch stays 0 while n_cont_rescues >= 1 on a trajectory
    the rescue in fact altered. Diagnostic only -- scores are unaffected (see test_regression)."""
    calls = [0]
    def rhs(field, s, a, B_star, v0, kap, zh, P, bmin, bmax, dzg, tm):
        calls[0] += 1
        if calls[0] == 3:                               # iter-1 RK4 s2 substage -> de-trap
            return 0.0, 0.0, None, BRANCH_EVENT
        return 5.0, 0.0, _base(), "ok"
    res = _run(rhs, GWParams(branch_continue=True, **_LONG),
               cont=lambda *a, **k: (5.0, 0.0, _base(0.1)), monkeypatch=monkeypatch)
    assert res.n_branch == 0                            # substage rescue invisible to n_branch
    assert res.n_cont_rescues >= 1                      # ... but caught by the persistent tally
    assert res.n_cont_rescues >= res.n_branch           # invariant holds


def test_n_cont_rescues_zero_without_continuation(monkeypatch):
    """No continuation ever fires (keep mode / clean orbit) -> n_cont_rescues stays 0."""
    res = _run(_rhs_const(-5.0), GWParams(branch_continue=False, **_LONG), monkeypatch=monkeypatch)
    assert res.n_cont_rescues == 0
    assert res.n_branch == 0


def test_detrapping_bc_no_descendant_is_detrap_passing(monkeypatch):
    """branch_continue ON but NO descendant well -> the orbit de-trapped to PASSING:
    DETRAP_PASSING (confined, resolved -- not unresolved, not a branch_event, no hop)."""
    P = GWParams(branch_continue=True, **_LONG)
    def cont(field, s, alpha, B_star, v0, kap, zh, P_, bmin, bmax, exclude=None):
        return None                                     # no descendant -> passing
    res = _run(_rhs_until(0.5, "branch"), P, cont=cont, monkeypatch=monkeypatch)
    assert res.status == DETRAP_PASSING
    assert res.unresolved is False
    assert res.branch_event is False
    assert res.reached_edge is False
    assert res.n_branch == 0


def test_keep_only_increments_n_branch_under_bc(monkeypatch):
    """Same de-trapping scenario, keep vs bc: only bc takes continuations, so n_branch
    is 0 under keep and positive under bc. Locks the dispatch invariant directly."""
    def cont(field, s, alpha, B_star, v0, kap, zh, P_, bmin, bmax, exclude=None):
        return 5.0, 0.0, _base(zeta_min=0.1)
    keep = _run(_rhs_until(0.5, "branch"), GWParams(branch_continue=False, **_LONG),
                monkeypatch=monkeypatch)
    bc = _run(_rhs_until(0.5, "branch"), GWParams(branch_continue=True, **_LONG),
              cont=cont, monkeypatch=monkeypatch)
    assert keep.n_branch == 0
    assert bc.n_branch >= 1


# ----------------------------------------------------------- FD mismatch -> band
def test_fd_mismatch_is_unresolved_and_lands_in_band(monkeypatch):
    """B1: an FD-branch mismatch is treated as UNRESOLVED (it enters aggregate's
    [low, high] band with low<-0, high<-1), NOT silently resolved as passing/confined.
    s_max is preserved and it is never DETRAP_PASSING under keep."""
    P = GWParams(branch_continue=False, **_LONG)
    res = _run(_rhs_until(0.45, "fd"), P, monkeypatch=monkeypatch)
    assert res.unresolved is True
    assert res.status in UNRESOLVED
    assert res.status != DETRAP_PASSING
    assert 0.3 < res.s_max < 1.0

    row = dict(weight=1.0, base_weight=1.0, R_W=res.R_W, passing_at_birth=False,
               unresolved=res.unresolved, branch_event=res.branch_event,
               reached_edge=res.reached_edge)
    a = aggregate([row])
    assert a["Gamma_W_low"] == 0.0                      # unresolved scored 0 in the low estimate
    assert a["Gamma_W_high"] == 1.0                     # ... and 1 in the high estimate
    assert a["Gamma_W_low"] <= a["Gamma_W"] <= a["Gamma_W_high"]
    np.testing.assert_allclose(a["unresolved_w_frac"], 1.0)


def test_fd_mismatch_under_bc_is_unresolved_not_detrap_passing(monkeypatch):
    """B2/P1 regression (the July-13 status-propagation fix): under branch_continue, an
    FD-branch mismatch (a NUMERICAL failure -- `_rhs` returns a valid base with a non-ok
    status) must be flagged UNRESOLVED with its originating status and preserve s_max. It
    must NEVER be mapped to the physical DETRAP_PASSING (scored confined, outside the band)
    -- which is exactly what the pre-patch engine did via `_rk4` returning bare None. Keep
    mode was always benign (-> BRANCH_EVENT, unresolved); this locks the bc mode too."""
    P = GWParams(branch_continue=True, **_LONG)
    res = _run(_rhs_until(0.5, "fd"), P, monkeypatch=monkeypatch)
    assert res.status == FD_MISMATCH                    # originating status propagated
    assert res.status != DETRAP_PASSING                 # the bug: numerical failure -> confined
    assert res.unresolved is True
    assert res.status in UNRESOLVED
    assert res.reached_edge is False
    assert 0.3 < res.s_max < 1.0                        # s_max preserved above launch
