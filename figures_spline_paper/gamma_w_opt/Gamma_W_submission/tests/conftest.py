"""Shared test configuration for the standalone Gamma_W package.

Layout-robust: it finds the canonical engine by walking UP from this file until it
sees `Gamma_W_final.py` (rather than a hardcoded parent depth), and puts that dir
plus `scripts/` (for `create_scan.py`) on sys.path. So the whole suite runs from
anywhere inside the package:  cd tests && pytest -m unit

Responsibilities:
1. Make `Gamma_W_final`, `gamma_c`, and `create_scan` importable.
2. Pin the engine md5 (frozen regression values track a specific engine).
3. Provide the firm3d flag + fixtures (`boozmn_qa`, `qa_field`) for the
   regression/slow tiers; the `unit` tier never touches firm3d and runs anywhere.

GOTCHA: never import `simsopt` in a test process (importing it then firm3d
segfaults). firm3d is detected with importlib.find_spec (no import), so collection
is always safe; only regression tests import firm3d field code.
"""
import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest


def _find_engine_dir(start: Path) -> Path:
    for d in [start] + list(start.parents):
        if (d / "Gamma_W_final.py").is_file():
            return d
    raise RuntimeError("Gamma_W_final.py not found above " + str(start))


ENGINE_DIR = _find_engine_dir(Path(__file__).resolve().parent)   # package root
for extra in (ENGINE_DIR, ENGINE_DIR / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

# The frozen regression values (test_regression.py) are only valid for this exact
# engine; pin it so a swapped/edited engine fails loudly instead of drifting.
#   lineage: 9de057ad -> 5952f45f (P1/B2 status propagation)
#                      -> 575187cf (diagnostic-only n_cont_rescues counter)
#                      -> b58272e4 (2026-08-19 docstring-only; bytecode identical)
ENGINE_HASH = "b58272e47b7352264209735bdef2aaa9"
ENGINE_FILE = ENGINE_DIR / "Gamma_W_final.py"

FIRM3D_AVAILABLE = importlib.util.find_spec("firm3d") is not None

_LOCAL_FIXTURE = Path(__file__).parent / "test_files" / "boozmn_LandremanPaul2021_QA_lowres.nc"


def _engine_md5():
    return hashlib.md5(ENGINE_FILE.read_bytes()).hexdigest()


@pytest.fixture(scope="session")
def engine_hash():
    return _engine_md5()


@pytest.fixture(scope="session")
def boozmn_qa():
    """Path to the small low-res QA boozmn shipped with the tests."""
    if _LOCAL_FIXTURE.is_file():
        return str(_LOCAL_FIXTURE)
    pytest.skip("QA boozmn fixture not found (tests/test_files/)")


@pytest.fixture(scope="session")
def qa_field(boozmn_qa):
    """One shared FieldBundle over the QA fixture (the ~20 s field build is the cost
    of the regression tier). Skips cleanly if firm3d is not importable."""
    if not FIRM3D_AVAILABLE:
        pytest.skip("firm3d not importable")
    from Gamma_W_final import FieldBundle
    return FieldBundle(boozmn_qa)
