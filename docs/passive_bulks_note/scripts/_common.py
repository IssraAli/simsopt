"""Shared helpers for the passive-bulks technical-note figure scripts.

Each ``fig_*.py`` script imports this module to build the same ring coils
used by the unit tests, to save a figure to
``docs/passive_bulks_note/figures/<name>.pdf`` in a consistent serif-math
style, and to pretty-print a small audit line so that the numbers shown in
the figure caption can be cross-checked against what the script itself
prints to stdout.

These helpers do not touch any state outside this subdirectory.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from simsopt.field.coil import Coil, Current
from simsopt.geo import CurveXYZFourier

FIGURES_DIR: Path = (Path(__file__).resolve().parent.parent / "figures").resolve()


# Permeability of free space in SI units (H/m).
MU0: float = 4.0 * np.pi * 1.0e-7


def setup_mpl() -> None:
    """Install a serif, math-friendly matplotlib style for the report.

    Kept deliberately minimal so the figures match what a LaTeX article
    produced by ``latexmk`` expects, while remaining pure matplotlib (no
    external usetex dependency).
    """
    mpl.rcParams.update({
        "figure.figsize": (6.2, 4.2),
        "figure.dpi": 120,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.family": "serif",
        "font.size": 10,
        "mathtext.fontset": "cm",
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "legend.frameon": False,
        "axes.grid": True,
        "grid.alpha": 0.35,
        "grid.linestyle": ":",
        "lines.linewidth": 1.6,
        "lines.markersize": 5,
    })


def save_pdf(fig: plt.Figure, name: str) -> Path:
    """Save ``fig`` to ``figures/<name>.pdf`` and return the path."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / f"{name}.pdf"
    fig.savefig(out)
    print(f"[save] {out}")
    return out


def ring_coil(
    radius: float = 1.0,
    z: float = 0.0,
    current: float = 1.0e5,
    order: int = 1,
    quadpoints: int = 64,
) -> Coil:
    """Return a thin circular filament in the ``z = const`` plane.

    Uses the same nine-coefficient :class:`CurveXYZFourier` recipe as the
    ``_ring_coil`` helpers in :mod:`tests.field.test_passive_bulks_scale`
    and :mod:`tests.field.test_passive_bulks_mode_convergence`, so the
    figures cross-check against the unit tests without drift.

    Args:
        radius: Ring radius (m).
        z: Axial plane (m).
        current: Ring current (A).
        order: Fourier order for :class:`CurveXYZFourier`.
        quadpoints: Curve quadrature point count.

    Returns:
        :class:`simsopt.field.coil.Coil`.
    """
    curve = CurveXYZFourier(int(quadpoints), int(order))
    curve.x = np.array([0.0, 0.0, radius, 0.0, radius, 0.0, 0.0, 0.0, z])
    return Coil(curve, Current(float(current)))


def approx_B0_from_ring(radius: float, current: float) -> float:
    r"""Uniform axial field at the center of a circular loop,
    :math:`B_0 = \mu_0 I / (2 R_{\text{coil}})`.

    This is the reference field used by the thin-disc benchmark.
    """
    return MU0 * float(current) / (2.0 * float(radius))


def print_audit(name: str, fields: Iterable[tuple[str, float]]) -> None:
    """Pretty-print a single-line audit record keyed by ``name``."""
    parts = [f"{key}={val:.6e}" for key, val in fields]
    print(f"[audit] {name} :: " + "  ".join(parts))


def silence_tf_warnings() -> None:
    """Silence noisy XLA init prints that clutter the figure scripts."""
    os.environ.setdefault("JAX_LOG_COMPILES", "0")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
