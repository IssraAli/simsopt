"""
Fidelity: PSC named timing profiles (``--profiles`` in passive_bulks_bottleneck_timing) vs baseline.

``baseline`` means cleared :data:`_PSC_PROFILE_KNOB_KEYS` in the example harness; here we
match that by ``monkeypatch.delenv`` for the same keys before rebuild.
"""

from __future__ import annotations

import importlib.util
import os
import pathlib
from typing import Any, Callable, Dict, Tuple

import numpy as np
import pytest

_PSC_KNOB_KEYS: Tuple[str, ...] = (
    "SIMSOPT_PSC_FREE_SOLVE_VJP",
    "SIMSOPT_PSC_PAIR_FAR_KAPPA",
    "SIMSOPT_PSC_TF_LOADING",
    "SIMSOPT_PSC_SOLVE_MODE",
    "SIMSOPT_PSC_BS_EVAL_FAR_KAPPA",
    "SIMSOPT_PSC_W1_ENVELOPE",
)

_PROFILE_BUNDLES: Dict[str, str] = {
    "baseline": "",
    "solve_implicit": "SIMSOPT_PSC_FREE_SOLVE_VJP=implicit",
    "far_pair_3": "SIMSOPT_PSC_PAIR_FAR_KAPPA=3",
    "tf_aquad": "SIMSOPT_PSC_TF_LOADING=a_quad",
    "tf_ataylor": "SIMSOPT_PSC_TF_LOADING=a_taylor",
    "eigk": "SIMSOPT_PSC_SOLVE_MODE=eigk",
    "bs_eval_far_3": "SIMSOPT_PSC_BS_EVAL_FAR_KAPPA=3",
    "combo_fast": (
        "SIMSOPT_PSC_FREE_SOLVE_VJP=implicit;"
        "SIMSOPT_PSC_PAIR_FAR_KAPPA=3;"
        "SIMSOPT_PSC_TF_LOADING=a_quad;"
        "SIMSOPT_PSC_BS_EVAL_FAR_KAPPA=3;"
        "SIMSOPT_PSC_SOLVE_MODE=eigk"
    ),
}


def _load_make_symmetry() -> Callable[..., Any]:
    here = pathlib.Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location(
        "_ws_tpb", here / "test_passive_bulks.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod._make_symmetry_validation_array


def _unfix_quats(psc: Any) -> None:
    n = int(psc._n_base_pucks)
    for i in range(n):
        for k in (f"q0_{i}", f"qi_{i}", f"qj_{i}", f"qk_{i}"):
            psc.unfix(k)


def _apply_bundle(monkeypatch: pytest.MonkeyPatch, bundle: str) -> None:
    for k in _PSC_KNOB_KEYS:
        monkeypatch.delenv(k, raising=False)
    for token in (bundle or "").replace(";", " ").split():
        if not token or "=" not in token:
            continue
        key, value = token.split("=", 1)
        monkeypatch.setenv(key.strip(), value.strip(), prepend=False)


def _clear_psc_tuning(monkeypatch: pytest.MonkeyPatch) -> None:
    for k in _PSC_KNOB_KEYS:
        monkeypatch.delenv(k, raising=False)


def _b_and_grad(psc: Any, pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    v = np.random.default_rng(42).standard_normal(pts.shape)
    B = np.asarray(psc.B_at_points(pts))
    d = psc.vjp_setup_B(v, pts)
    g = np.asarray(d(psc), dtype=float).ravel()
    return B, g


def _run_profile_tight_fidelity(
    n_base: int,
    profile: str,
    monkeypatch: pytest.MonkeyPatch,
    *,
    b_rtol: float,
    cos_min: float,
) -> None:
    if os.environ.get("SIMSOPT_PSC_DISABLE_REDUCED_FREE", "0") == "1":
        pytest.skip("reduced free DOF path disabled")
    make = _load_make_symmetry()
    psc = make(nfp=2, stellsym=True, n_base=n_base)
    _unfix_quats(psc)
    _clear_psc_tuning(monkeypatch)
    psc._rebuild()
    psc.recompute_currents()
    pts = psc.eval_points
    B0, g0 = _b_and_grad(psc, pts)
    n0 = float(np.linalg.norm(B0.ravel())) + 1e-30
    g0n = float(np.linalg.norm(g0)) + 1e-30

    _apply_bundle(monkeypatch, _PROFILE_BUNDLES[profile])
    psc._local_stacks_valid = False
    psc._rebuild()
    psc.recompute_currents()
    B1, g1 = _b_and_grad(psc, pts)
    assert np.isfinite(B1).all() and np.isfinite(g1).all(), (profile, n_base)

    err_b = float(np.linalg.norm((B1 - B0).ravel()) / n0)
    assert err_b < b_rtol, (profile, n_base, err_b)
    g1n = float(np.linalg.norm(g1)) + 1e-30
    cos = float(np.dot(g0, g1) / (g0n * g1n))
    assert cos > cos_min, (profile, n_base, cos)


def _run_profile_smoke(
    n_base: int, profile: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Named bundles that change physics vs dense ``bn_quad`` — only require finite B/VJP."""
    if os.environ.get("SIMSOPT_PSC_DISABLE_REDUCED_FREE", "0") == "1":
        pytest.skip("reduced free DOF path disabled")
    make = _load_make_symmetry()
    psc = make(nfp=2, stellsym=True, n_base=n_base)
    _unfix_quats(psc)
    _apply_bundle(monkeypatch, _PROFILE_BUNDLES[profile])
    psc._local_stacks_valid = False
    psc._rebuild()
    psc.recompute_currents()
    pts = psc.eval_points
    B1, g1 = _b_and_grad(psc, pts)
    assert np.isfinite(B1).all() and np.isfinite(g1).all(), (profile, n_base)
    assert float(np.linalg.norm(B1.ravel())) > 1e-20
    assert float(np.linalg.norm(g1)) > 1e-20


_STRICT: Tuple[str, ...] = ("solve_implicit", "eigk")
_ENGINEERING: Tuple[str, ...] = (
    "far_pair_3",
    "tf_aquad",
    "tf_ataylor",
    "bs_eval_far_3",
    "combo_fast",
)


@pytest.mark.fidelity
@pytest.mark.parametrize("n_base", [2, 6])
@pytest.mark.parametrize("profile", _STRICT, ids=lambda p: p)
def test_profile_tight_fidelity_against_baseline(
    n_base: int, profile: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Solves / AD modes that should track ``bn_quad``+baseline within ~1% (B) and 0.999 (grad)."""
    _run_profile_tight_fidelity(
        n_base, profile, monkeypatch, b_rtol=0.01, cos_min=0.999
    )


@pytest.mark.fidelity
@pytest.mark.parametrize("n_base", [2, 6])
@pytest.mark.parametrize("profile", _ENGINEERING, ids=lambda p: p)
def test_profile_engineering_runs(
    n_base: int, profile: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Multipole / cheap-TF / far-B bundles are not expected to match dense ``bn_quad``; smoke-only."""
    _run_profile_smoke(n_base, profile, monkeypatch)
