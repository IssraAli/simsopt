r"""End-to-end regression tests against firm3d, on the shipped small QA boozmn
(tests/test_files/boozmn_LandremanPaul2021_QA_lowres.nc, 604 kB).

These are the SLOW tests: each gamma_w_for_boozmn call builds an
InterpolatedBoozerField (~20 s), so they run only under `pytest -m regression`
(and are skipped entirely if firm3d is not importable). Reference values are
PINNED to the engine hash asserted by test_core.test_engine_provenance_hash;
change the engine and that guard fails first.

Frozen point (N=32, t_star=0.05, s0=0.3, seed=7) -- captured & re-verified in this
env:

    keep (branch_continue=False): Gamma_W=0  band [0, 0.25]  de-trapping-dominated
    bc   (branch_continue=True):  Gamma_W=0  band [0, 0.00]  adiabatic-resolved

i.e. this good QA stays put (Gamma_W=0 either way), and ALL the trapped weight that
`keep` leaves unresolved as a de-trapping event is RESOLVED by branch_continue --
the band collapses and the regime flag flips. That keep<->bc equivalence-on-the-
estimate but resolution-of-the-band is the headline finding this suite locks.
"""
import importlib.util

import numpy as np
import pytest

FIRM3D = importlib.util.find_spec("firm3d") is not None
requires_firm3d = pytest.mark.skipif(not FIRM3D, reason="firm3d not importable")

# frozen-point parameters
N, TSTAR, S0, SEED = 32, 0.05, 0.3, 7


# ----------------------------------------------------------- FieldBundle interface
@requires_firm3d
@pytest.mark.regression
def test_fieldbundle_interface(qa_field):
    """FieldBundle reads the QA equilibrium correctly and its accessors behave."""
    fb = qa_field
    assert fb.nfp == 2
    np.testing.assert_allclose(fb.psi0, -0.01335, atol=1e-4)
    assert fb.Psi_LCFS == abs(fb.psi0)
    assert fb.helicity() == (1, 0)                       # quasi-axisymmetric (QA)
    G, I, iota = fb.profiles(S0)
    np.testing.assert_allclose(iota, 0.42074, atol=1e-3)
    assert np.isfinite(G) and np.isfinite(I)
    bmin, bmax = fb.surface_B_range(S0)
    assert 0.0 < bmin < bmax                              # a real |B| range
    assert bmin <= fb.modB_at(S0, 0.0, 0.0) <= bmax

@requires_firm3d
@pytest.mark.regression
def test_fieldbundle_s_clamp_guard(qa_field):
    """firm3d SEGFAULTS on out-of-domain s; FieldBundle.modB must clamp so the
    branch-continuation FD can safely probe near/over the edge (returns finite)."""
    val = qa_field.modB(2.0, 0.0, 0.0)                   # s=2 -> clamped to <1
    assert np.all(np.isfinite(val))


# ----------------------------------------------------------- frozen Gamma_W
@requires_firm3d
@pytest.mark.slow
def test_frozen_gamma_w_keep(boozmn_qa):
    """The published `keep` metric on the QA fixture: Gamma_W=0 with an unresolved
    de-trapping band [0, ~0.25] and the de-trapping-dominated regime flag."""
    from Gamma_W_final import gamma_w_for_boozmn
    a = gamma_w_for_boozmn(boozmn_qa, N=N, t_star=TSTAR, s0=S0, seed=SEED,
                           verbose=False, branch_continue=False)
    assert a["nfp"] == 2
    assert a["Gamma_W"] == 0.0
    assert a["Gamma_W_low"] == 0.0
    assert a["L_W"] == 0.0
    assert a["Gamma_W_high"] == pytest.approx(0.25, abs=0.06)
    assert a["Gamma_W_high"] > 0.0
    assert a["regime"] == "de-trapping-dominated"
    assert a["passing_frac"] == pytest.approx(0.75, abs=0.06)
    # band always brackets the estimate
    assert a["Gamma_W_low"] <= a["Gamma_W"] <= a["Gamma_W_high"]


@requires_firm3d
@pytest.mark.slow
def test_keep_vs_branch_continue_resolves_the_band(boozmn_qa):
    """Headline keep<->bc behavior: identical Gamma_W estimate (both 0 -> rank-
    neutral), identical source classification, but branch_continue RESOLVES the
    de-trapping band that keep leaves open (high 0.25 -> 0, regime flips)."""
    from Gamma_W_final import gamma_w_for_boozmn
    keep = gamma_w_for_boozmn(boozmn_qa, N=N, t_star=TSTAR, s0=S0, seed=SEED,
                              verbose=False, branch_continue=False)
    bc = gamma_w_for_boozmn(boozmn_qa, N=N, t_star=TSTAR, s0=S0, seed=SEED,
                            verbose=False, branch_continue=True)
    # rank-neutral on the estimate
    assert keep["Gamma_W"] == bc["Gamma_W"] == 0.0
    # same source measure (passing/trapped split is a launch property, dispatch-free)
    assert keep["passing_frac"] == bc["passing_frac"]
    # bc resolves the band keep leaves open
    assert keep["Gamma_W_high"] > bc["Gamma_W_high"]
    assert bc["Gamma_W_high"] == 0.0
    assert bc["unresolved_w_frac"] == 0.0
    assert keep["regime"] == "de-trapping-dominated"
    assert bc["regime"] == "adiabatic-resolved"
    assert bc["gamma_w_valid"] is True


@requires_firm3d
@pytest.mark.slow
def test_gamma_w_determinism(boozmn_qa):
    """Same seed -> identical Gamma_W and band, bit-for-bit."""
    from Gamma_W_final import gamma_w_for_boozmn
    kw = dict(N=N, t_star=TSTAR, s0=S0, seed=SEED, verbose=False, branch_continue=False)
    a1 = gamma_w_for_boozmn(boozmn_qa, **kw)
    a2 = gamma_w_for_boozmn(boozmn_qa, **kw)
    for k in ("Gamma_W", "Gamma_W_low", "Gamma_W_high", "passing_frac",
              "unresolved_w_frac", "L_W"):
        assert a1[k] == a2[k], f"{k} not deterministic: {a1[k]} != {a2[k]}"
