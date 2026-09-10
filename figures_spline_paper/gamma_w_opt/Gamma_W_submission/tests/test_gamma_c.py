r"""Tests for gamma_c.py -- the Nemov contour-inclination proxy computed as a
read-only by-product of Gamma_W's bounce integrals.

Two layers:
- unit (no firm3d): the algebraic identity that lets gamma_c REUSE Gamma_W's `_rhs`
  drift -- the common tau/kappa factor cancels inside atan2, so
  gamma_c = (2/pi) atan2(|sdot|, |adot|) = (2/pi) atan2(|dJ_da|, |dJ_ds|).
- regression (firm3d): on the shipped QA boozmn, gamma_c per marker equals the
  value re-derived from `_rhs` (bit-identical by construction), stays in [0,1], is
  deterministic, and `_rhs` still returns its 4-tuple (non-invasive integration).
"""
import importlib.util

import numpy as np
import pytest

FIRM3D = importlib.util.find_spec("firm3d") is not None
requires_firm3d = pytest.mark.skipif(not FIRM3D, reason="firm3d not importable")

S0, SEED, N = 0.25, 7, 64


# ----------------------------------------------------------- unit: the identity
@pytest.mark.unit
def test_tau_over_kappa_cancels_in_angle():
    """gamma_c is built from sdot=kap/tau dJ_da and adot=-kap/tau dJ_ds. The shared
    kap/tau prefactor must cancel, so the angle depends only on the drift ratio."""
    two_over_pi = 2.0 / np.pi
    for dJ_da, dJ_ds, tau, kap in [(0.7, 1.3, 3.1e-6, 2.0e-8),
                                   (2.0, 0.5, 1.0e-5, 5.0e-8),
                                   (-1.0, 2.0, 7.7e-6, 9.0e-9)]:
        sdot = kap / tau * dJ_da
        adot = -kap / tau * dJ_ds
        gc_from_drift = two_over_pi * np.arctan2(abs(sdot), abs(adot))
        gc_from_J = two_over_pi * np.arctan2(abs(dJ_da), abs(dJ_ds))
        np.testing.assert_allclose(gc_from_drift, gc_from_J, rtol=1e-13)
        assert 0.0 <= gc_from_J <= 1.0


# ----------------------------------------------------------- regression: firm3d
@requires_firm3d
@pytest.mark.regression
def test_gamma_c_equals_rhs_derivation_and_in_range(qa_field):
    """Per trapped marker, gamma_c == (2/pi) atan2(|dJ_da|, |dJ_ds|) with the
    derivatives taken FROM `_rhs` itself, and gamma_c in [0,1]."""
    import gamma_c as gc
    from Gamma_W_final import (GWParams, make_surface_markers, _rhs, _dzeta_grid,
                               Bstar_from_pitch, kappa, V0_ALPHA, TWO_PI)
    field, P = qa_field, GWParams()
    markers = make_surface_markers(field, N, S0, seed=SEED)
    kap = kappa(field.Psi_LCFS)
    dzg = _dzeta_grid(field, P)
    bcache, n_checked = {}, 0
    for m in markers:
        row = gc.gamma_c_for_marker(field, m, P, kap, bcache)
        if row["passing_at_birth"] or row["fd_fail"]:
            continue
        s0, a0, z0, th0, xi = m["s0"], m["alpha0"], m["zeta0"], m["theta0"], m["pitch_xi"]
        B_star = Bstar_from_pitch(field.modB_at(s0, th0, z0), xi)
        bmin, bmax = bcache[s0]
        sd, ad, base, st = _rhs(field, s0, a0 % TWO_PI, B_star, V0_ALPHA, kap, z0, P,
                                bmin, bmax, dzg, None)
        assert st == "ok" and base is not None
        dJ_da, dJ_ds = sd * base.tau / kap, -ad * base.tau / kap
        expect = (2.0 / np.pi) * np.arctan2(abs(dJ_da), abs(dJ_ds))
        np.testing.assert_allclose(row["gamma_c"], expect, atol=1e-14)
        assert 0.0 <= row["gamma_c"] <= 1.0
        n_checked += 1
    assert n_checked >= 3, f"too few trapped/resolved markers to test ({n_checked})"


@requires_firm3d
@pytest.mark.regression
def test_rhs_signature_intact(qa_field):
    """gamma_c took the non-invasive path: Gamma_W's `_rhs` still returns a 4-tuple
    (sdot, adot, base, status). Guards against a future edit breaking the contract."""
    from Gamma_W_final import GWParams, _rhs, _dzeta_grid, kappa, V0_ALPHA
    field, P = qa_field, GWParams()
    bmin, bmax = field.surface_B_range(S0)
    out = _rhs(field, S0, 0.0, bmin * 1.001, V0_ALPHA, kappa(field.Psi_LCFS), 0.0, P,
               bmin, bmax, _dzeta_grid(field, P), None)
    assert isinstance(out, tuple) and len(out) == 4


@requires_firm3d
@pytest.mark.slow
def test_gamma_c_determinism(boozmn_qa):
    """Same seed -> identical Gamma_c aggregates, and both finite."""
    import gamma_c as gc
    a1 = gc.gamma_c_for_boozmn(boozmn_qa, N=N, s0=S0, seed=SEED, verbose=False)
    a2 = gc.gamma_c_for_boozmn(boozmn_qa, N=N, s0=S0, seed=SEED, verbose=False)
    assert a1["Gamma_c_trapped"] == a2["Gamma_c_trapped"]
    assert a1["Gamma_c_source"] == a2["Gamma_c_source"]
    assert np.isfinite(a1["Gamma_c_trapped"]) and np.isfinite(a1["Gamma_c_source"])
