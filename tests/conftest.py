"""Shared pytest configuration for the ``simsopt`` test suite.

Registers the ``slow`` marker and adds a ``--runslow`` CLI option so that
tests decorated with ``@pytest.mark.slow`` are skipped by default.  This
keeps iteration on non-slow tests fast without losing the ability to
run the full suite.

Run only fast tests (default)::

    pytest tests/field/test_passive_bulks.py

Run everything including slow tests::

    pytest tests/field/test_passive_bulks.py --runslow

Run only slow tests::

    pytest tests/field/test_passive_bulks.py -m slow --runslow
"""

from __future__ import annotations

import pytest


def pytest_addoption(parser) -> None:
    """Register ``--runslow`` to opt into slow tests."""
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="Run tests marked with @pytest.mark.slow (skipped by default).",
    )
    parser.addoption(
        "--psc-workstream",
        action="store",
        default="all",
        help=(
            "Restrict PSC multipole workstream tests (markers psc_w0..psc_w7). "
            "Use 'all' (default) or a comma-separated list such as 'w0,w7'."
        ),
    )


def pytest_configure(config) -> None:
    """Register the ``slow`` marker so ``--strict-markers`` is happy."""
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (skip unless --runslow is given)",
    )


def pytest_collection_modifyitems(config, items) -> None:
    """Skip ``slow``-marked tests unless ``--runslow`` was passed."""
    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="Need --runslow to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)

    # Optional PSC workstream filter (``tests/field/test_psc_multipole.py``).
    # Example: ``pytest --psc-workstream=w0,w7`` keeps only @pytest.mark.psc_w0
    # and @pytest.mark.psc_w7 tests.
    try:
        ws_raw = config.getoption("--psc-workstream")
    except (ValueError, KeyError, AttributeError):
        ws_raw = "all"
    wss = str(ws_raw).lower().strip()
    if wss in ("", "all", "none"):
        return
    allowed: set[str] = set()
    for p in wss.split(","):
        t = p.strip()
        if not t:
            continue
        if t.startswith("psc_w"):
            allowed.add(t)
        elif t.startswith("w") and t[1:].isdigit():
            allowed.add("psc_" + t)
        elif t.isdigit():
            allowed.add("psc_w" + t)
        else:
            allowed.add("psc_w" + t)
    if not allowed:
        return
    skip_ws = pytest.mark.skip(
        reason=f"Filtered out by --psc-workstream={ws_raw!r} (need one of {sorted(allowed)!r})"
    )
    for item in items:
        psc_keys = [str(k) for k in item.keywords if str(k).startswith("psc_w")]
        if not psc_keys or any(k in allowed for k in psc_keys):
            continue
        item.add_marker(skip_ws)
