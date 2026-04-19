"""Convergence of :class:`~simsopt.field.coil.PSCBulkArray` induced field w.r.t. shell basis.

Fixes puck geometry, coil geometry, and evaluation points; increases the shell basis
resolution ``(m_fourier, l_zernike, k_chebyshev)`` from coarse to fine and checks that
the induced bulk :math:`\\mathbf{B}` at those points converges toward a higher-resolution
reference solve (monotone relative error decrease and final error below 1%).
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")

from simsopt.field.coil import Coil, Current
from simsopt.field.psc_bulk import PSCBulkArray
from simsopt.geo import CurveXYZFourier


def _ring_coil(radius: float = 1.0, z: float = 0.0, current: float = 1.0e5) -> Coil:
    """Build a single circular filament in the :math:`x\\text{–}y` plane at height ``z``."""
    curve = CurveXYZFourier(32, 1)
    curve.x = np.array([0, 0, radius, 0, radius, 0, 0, 0.0, z])
    return Coil(curve, Current(current))


@pytest.mark.slow
def test_mode_convergence_bulk_field() -> None:
    """Sweep shell mode counts; induced bulk field at fixed points should converge."""
    tf_coils = [
        _ring_coil(radius=1.2, z=+0.4, current=2.0e5),
        _ring_coil(radius=1.2, z=-0.4, current=2.0e5),
    ]
    centers = np.array(
        [
            [0.85, 0.00, 0.25],
            [-0.85, 0.00, 0.25],
            [0.85, 0.00, -0.25],
            [-0.85, 0.00, -0.25],
        ]
    )
    axes = centers / np.linalg.norm(centers, axis=1, keepdims=True)
    radii = np.full(4, 0.06)
    thicknesses = np.full(4, 0.03)
    eval_pts = np.array([[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5]])

    def build(m: int, l: int, k: int) -> PSCBulkArray:
        return PSCBulkArray(
            centers,
            axes,
            radii,
            thicknesses,
            tf_coils,
            eval_points=eval_pts,
            m_fourier=m,
            l_zernike=l,
            k_chebyshev=k,
            n_rho=8,
            n_phi=12,
            n_z=6,
        )

    psc_ref = build(4, 8, 4)
    psc_ref.recompute_currents()
    b_ref = np.asarray(psc_ref.B_at_points(eval_pts))

    sweep = [(1, 2, 1), (2, 4, 2), (3, 6, 3)]
    errs: list[float] = []
    for m, l, k in sweep:
        psc = build(m, l, k)
        psc.recompute_currents()
        b_arr = np.asarray(psc.B_at_points(eval_pts))
        errs.append(
            float(np.linalg.norm(b_arr - b_ref) / (np.linalg.norm(b_ref) + 1e-30))
        )

    for i in range(len(errs) - 1):
        assert errs[i + 1] <= errs[i] * 0.8, (
            f"Mode-convergence not monotone: errs = {errs}"
        )
    assert errs[-1] < 1e-2, f"Finest sweep error too large: {errs[-1]}"
