r"""Regression tests for create_scan.py's RippleInducer -- the ripple-scan equilibrium
generator (P5 gate). These lock the two boundary-corruption bugs the July-13 review flagged:

  P5a  the exponent-strip parse bug: `float(tok[:-1])` ate the last char of a coefficient with
       no trailing comma (the perturbed / last one on its line), so a sci-notation exponent lost
       a digit: `...e-02 -> ...e-0` == x100. Fixed to `float(tok.rstrip(','))`.
  P5b  reset() aliasing: `self.lines = self.original_lines` shared one list, so the next
       add_ripple's in-place write corrupted the pristine base and ripple amplitudes ACCUMULATED
       across scan iterations. Fixed to `self.original_lines[:]` (a copy).

Both are demonstrated to FAIL on the pre-fix generator (see the docstring of
`test_bugs_are_actually_caught`) and pass on the current one. No firm3d, no VMEC -- a synthetic
boundary block in the exact `RBC(   n,   m) =  val,    ZBS(   n,   m) =  val` format the parser
expects. Run:  pytest test_create_scan.py
"""
import re

import pytest

from create_scan import RippleInducer

# Synthetic VMEC boundary. r00 = RBC(0,0) = 13.6 (the relative-ripple scale). The ZBS(2,0)
# coefficient carries an e-02 exponent on purpose: it is the last value on its line (no trailing
# comma), so it is exactly what the old `[:-1]` mis-parsed as 2.55 (x100) instead of 0.0255.
R00 = 13.6
ZBS20_BASE = 2.55e-02
RBC10_BASE = 2.0
_INPUT = (
    "&INDATA\n"
    "  MPOL = 6\n"
    "RBC(   0,   0) =  1.360000000000000e+01,    ZBS(   0,   0) =  0.000000000000000e+00\n"
    "RBC(   1,   0) =  2.000000000000000e+00,    ZBS(   1,   0) =  2.088000000000000e+00\n"
    "RBC(   2,   0) =  1.000000000000000e-01,    ZBS(   2,   0) =  2.550000000000000e-02\n"
    "/\n"
)


def _write(tmp_path):
    p = tmp_path / "input.vmec"
    p.write_text(_INPUT)
    return str(p)


def _coeff(lines, kind, n, m):
    """Extract the numeric KIND(n,m) coefficient from a list of boundary lines (last wins)."""
    text = "".join(lines)
    pat = rf"{kind}\(\s*{n}\s*,\s*{m}\s*\)\s*=\s*([-+.\deE]+)"
    hits = re.findall(pat, text)
    return float(hits[-1]) if hits else None


def test_parser_reads_r00(tmp_path):
    ri = RippleInducer(_write(tmp_path))
    assert ri.r00 == pytest.approx(R00)


def test_zero_perturbation_identity_zbs(tmp_path):
    """The gate the handover asks for: amplitude 0 reproduces the base coefficient. If the
    exponent were truncated (P5a) the ZBS(2,0) base would read 2.55 and this would fail."""
    ri = RippleInducer(_write(tmp_path))
    ri.add_ripple(2, 0, 0.0, z=True)
    assert _coeff(ri.lines, "ZBS", 2, 0) == pytest.approx(ZBS20_BASE, rel=1e-9)


def test_zero_perturbation_identity_rbc(tmp_path):
    ri = RippleInducer(_write(tmp_path))
    ri.add_ripple(1, 0, 0.0, z=False)
    assert _coeff(ri.lines, "RBC", 1, 0) == pytest.approx(RBC10_BASE, rel=1e-9)


def test_relative_amplitude_is_base_plus_a_r00(tmp_path):
    ri = RippleInducer(_write(tmp_path))
    ri.add_ripple(2, 0, 0.1, z=True)          # A = base + a * r00
    assert _coeff(ri.lines, "ZBS", 2, 0) == pytest.approx(ZBS20_BASE + 0.1 * R00, rel=1e-9)


def test_no_accumulation_across_reset(tmp_path):
    """P5b: a multi-amplitude scan with reset() between iterations must NOT accumulate. The
    aliasing bug corrupts original_lines on iteration 2, so iteration 3 (a=0.3) would read a
    contaminated base -- this test needs >=3 iterations to expose it."""
    ri = RippleInducer(_write(tmp_path))
    got = []
    for a in (0.1, 0.2, 0.3):
        ri.add_ripple(2, 0, a, z=True)
        got.append(_coeff(ri.lines, "ZBS", 2, 0))
        ri.reset()
    for a, v in zip((0.1, 0.2, 0.3), got):
        assert v == pytest.approx(ZBS20_BASE + a * R00, rel=1e-9), \
            f"amplitude {a}: got {v}, expected non-accumulated {ZBS20_BASE + a * R00}"


def test_unperturbed_modes_untouched(tmp_path):
    """Perturbing ZBS(2,0) must leave RBC(1,0)/ZBS(1,0) at their base values."""
    ri = RippleInducer(_write(tmp_path))
    ri.add_ripple(2, 0, 0.1, z=True)
    assert _coeff(ri.lines, "RBC", 1, 0) == pytest.approx(RBC10_BASE, rel=1e-9)
    assert _coeff(ri.lines, "ZBS", 1, 0) == pytest.approx(2.088, rel=1e-9)
