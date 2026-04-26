"""
Field-module pytest helpers and PSC / passive-bulk markers.

Cumulative workstream selection is enabled from the *root* ``tests/conftest.py``
(``--psc-workstream``) so a single :func:`pytest_configure` hook is not
duplicated incompatibly.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest


def _rebuild_with_env(
    psc: Any,
    env: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Set ``os.environ`` keys, rebuild PSC bulk state, and restore on exit."""
    for k, v in env.items():
        monkeypatch.setenv(k, v, prepend=False)
    psc._local_stacks_valid = False
    psc._rebuild()


def _assert_grad_cosine(g_num: np.ndarray, g_ana: np.ndarray) -> None:
    """``cos(g_num, g_ana)`` should be indistinguishable from unity at unit tolerance."""
    a = np.asarray(g_num, dtype=float).ravel()
    b = np.asarray(g_ana, dtype=float).ravel()
    na = float(np.linalg.norm(a)) + 1e-30
    nb = float(np.linalg.norm(b)) + 1e-30
    c = float(np.dot(a, b) / (na * nb))
    assert c > 0.999, c


def _random_dof_direction(n: int, seed: int = 0) -> np.ndarray:
    """Unit vector in ``R^n`` (deterministic RNG)."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(int(n))
    v /= float(np.linalg.norm(v)) + 1e-30
    return v


def _assert_loglog_slope(
    xs: np.ndarray, ys: np.ndarray, expect: float, *, atol: float = 0.3
) -> None:
    """Assert least-squares log-log slope is ``expect`` within ``atol``."""
    lx = np.log(np.asarray(xs, dtype=float) + 1e-30)
    ly = np.log(np.abs(np.asarray(ys, dtype=float)) + 1e-30)
    coeff = np.polyfit(lx, ly, 1)
    slope = float(coeff[0])
    assert abs(slope - expect) < atol, (slope, expect)


def pytest_configure(config) -> None:
    """Field-local markers (compatible with the root :file:`../conftest.py`)."""
    for line in (
        "fidelity: PSC / field numerical fidelity checks (can be slower)",
        "integration: end-to-end PSC / field smoke tests",
        "psc_w0: W0 workstream (multipole cache)",
        "psc_w1: W1 (implicit / envelope solve adjoint)",
        "psc_w2: W2 (multipole far pairs)",
        "psc_w3: W3 (TF A / Taylor loading)",
        "psc_w4: W4 (eigK ROM)",
        "psc_w5: W5 (stream basis)",
        "psc_w6: W6 (quaternion tangents)",
        "psc_w7: W7 (far Biot/dipole eval)",
    ):
        config.addinivalue_line("markers", line)
