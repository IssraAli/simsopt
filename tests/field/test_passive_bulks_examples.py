"""End-to-end smoke tests for the two new passive-bulk example scripts.

Each example is run in a subprocess with ``PASSIVE_BULKS_MAXITER=2`` and a hard
timeout.  ``PYTHONPATH`` is stripped so the interpreter uses the same import
resolution as a normal user (editable install or site-packages).  This catches
stale site-packages copies, missing modules such as ``puck_init``, and broken
example parameters.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
EXAMPLES_DIR: Path = REPO_ROOT / "examples" / "3_Advanced"

PASSIVE_BULK_EXAMPLES: tuple[Path, ...] = (
    EXAMPLES_DIR / "passive_bulks_winding_surface_optimization.py",
    EXAMPLES_DIR / "passive_bulks_toroidal_shell_optimization.py",
)


@pytest.mark.slow
@pytest.mark.parametrize("script", PASSIVE_BULK_EXAMPLES, ids=lambda p: p.name)
def test_passive_bulk_example_runs(script: Path, tmp_path: Path) -> None:
    """Run a passive-bulk optimization example to completion (short iteration cap)."""
    assert script.is_file(), f"Missing example script: {script}"
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PASSIVE_BULKS_MAXITER"] = "2"
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, (
        f"Example {script.name} failed.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "Done." in result.stdout, (
        f"Example {script.name} did not print completion marker.\n"
        f"stdout:\n{result.stdout}"
    )
