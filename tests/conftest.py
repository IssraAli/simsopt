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
