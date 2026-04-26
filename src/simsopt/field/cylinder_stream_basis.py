"""
Global stream-function basis on a **closed** cylindrical shell (W5 planning).

A single-valued :math:`\\psi` on the side wall plus top/bottom caps, periodic in
azimuth, enforces (weakly) :math:`\\nabla_s \\times (\\mathbf n \\times \\nabla_s \\psi) = 0`
for the sheet-current representation :math:`\\mathbf K = \\mathbf n \\times \\nabla_s\\psi`
when paired with a harmonic extension / gauge choice.

W5 in the free-DOF plan replaces tensor-product :math:`Zernike\\times Fourier\\times
Chebyshev` patches and their rim projection ``Q_c`` with a **single global**
:math:`H^1`-conforming gluing; until that integration lands in
:mod:`simsopt.field.puck_basis`, this module documents the **API surface** only
(no PSC wiring here).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class ClosedCylinderStreamConfig:
    """Resolution knobs for a separable :math:`\\psi(u, z)` on a radius-``R`` shell."""

    radius: float
    height: float
    n_phi: int
    n_z: int
    m_phi_max: int
    n_z_leg: int

    def __post_init__(self) -> None:
        if self.radius <= 0.0 or self.height <= 0.0:
            raise ValueError("radius and height must be positive")
        if self.n_phi < 3 or self.n_z < 2:
            raise ValueError("need enough grid points for a periodic+interval basis")


def tikhonov_stream_coefficients_synthetic(
    rng: np.random.Generator, cfg: ClosedCylinderStreamConfig
) -> np.ndarray:
    """Return random normalized coefficients (placeholder for a future real solve)."""
    n_modes = (2 * cfg.m_phi_max + 1) * (cfg.n_z_leg + 1)
    x = rng.standard_normal(n_modes)
    n = float(np.linalg.norm(x)) + 1e-20
    return (x / n).astype(float)


def stream_basis_grid_shape(cfg: ClosedCylinderStreamConfig) -> Tuple[int, int]:
    """Return ``(n_phi, n_z)`` sample counts on the side surface."""
    return (int(cfg.n_phi), int(cfg.n_z))
