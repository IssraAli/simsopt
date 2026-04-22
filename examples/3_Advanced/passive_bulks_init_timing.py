#!/usr/bin/env python3
"""Thin launcher for passive-bulk init timing (implementation in stellcoilbench_dipoles).

The real script lives next to the case YAMLs::

    stellcoilbench_dipoles/scripts/passive_bulk_init_timing.py

This file only forwards ``sys.argv`` so you can discover the harness from the
simsopt checkout when both repos sit under ``stellcoilbench_repos/``.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    simsopt_root = Path(__file__).resolve().parents[2]
    script = (
        simsopt_root.parent
        / "stellcoilbench_dipoles"
        / "scripts"
        / "passive_bulk_init_timing.py"
    )
    if not script.is_file():
        raise FileNotFoundError(
            f"Expected timing script at {script}. "
            "Clone stellcoilbench_dipoles alongside simsopt_fork."
        )
    simsopt_src = simsopt_root / "src"
    env = os.environ.copy()
    sep = os.pathsep
    env["PYTHONPATH"] = f"{simsopt_src}{sep}{env.get('PYTHONPATH', '')}"
    raise SystemExit(
        subprocess.call([sys.executable, str(script), *sys.argv[1:]], env=env)
    )


if __name__ == "__main__":
    main()
