"""Figure: convergence of :func:`cylinder_self_block_mode_m` in ``n_z``.

Sweeps ``n_z`` for three representative Fourier-mode / Chebyshev-order
combinations and plots the relative distance from a high-resolution
reference at ``n_z = 72`` on a log-y axis.  The 1D Duffy + Gauss-Laguerre
axial rule inherits the exponential convergence of its disc counterpart
and the plot linearizes on a log scale, confirming that observation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import print_audit, save_pdf, setup_mpl  # noqa: E402

from simsopt.field.disc_self_inductance import (  # noqa: E402
    _chebyshev_t_derivative_numpy,
    _chebyshev_t_numpy,
    cylinder_self_block_mode_m,
)


def _make_axial(t: float, k: int):
    """Return ``(f, f')`` for the side-wall basis ``g = T_k(2 z / t)``.

    Mirrors :func:`simsopt.field.disc_self_inductance._make_axial`.
    """

    def f(z: np.ndarray) -> np.ndarray:
        zeta = 2.0 * np.asarray(z, dtype=float) / t
        return _chebyshev_t_numpy(zeta, k)

    def fp(z: np.ndarray) -> np.ndarray:
        zeta = 2.0 * np.asarray(z, dtype=float) / t
        return (2.0 / t) * _chebyshev_t_derivative_numpy(zeta, k)

    return f, fp


def main() -> None:
    """Generate and save the cylinder-side-block convergence figure."""
    setup_mpl()
    R = 0.30
    t = 0.30
    modes = [(0, 1), (1, 1), (2, 2)]  # (m, k_ch)
    ns = np.array([4, 6, 8, 12, 16, 24, 32, 48], dtype=int)
    n_ref = 72

    errs: dict[tuple[int, int], list[float]] = {mode: [] for mode in modes}
    refs: dict[tuple[int, int], float] = {}
    for (m, k) in modes:
        f, fp = _make_axial(t, k)
        ref = cylinder_self_block_mode_m(f, fp, f, fp, m, R, t, n_z=n_ref)
        refs[(m, k)] = ref
        for nq in ns:
            val = cylinder_self_block_mode_m(f, fp, f, fp, m, R, t, n_z=int(nq))
            errs[(m, k)].append(abs(val - ref) / (abs(ref) + 1e-30))

    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    markers = ["o", "s", "^"]
    for (mode, marker) in zip(modes, markers):
        m, k = mode
        ax.semilogy(ns, errs[mode], f"-{marker}", label=rf"$(m, k) = ({m}, {k})$")
    ax.set_xlabel(r"$n_z$")
    ax.set_ylabel(r"rel.\ error vs $n_z = %d$" % n_ref)
    ax.set_title(r"Cylinder-side self-block quadrature (R=%.2f m, t=%.2f m)" % (R, t))
    ax.set_ylim(1.0e-16, 1.0)
    ax.legend(loc="upper right")
    ax.grid(True, which="both", alpha=0.35)
    fig.tight_layout()

    for mode in modes:
        m, k = mode
        print_audit(
            f"cyl_quad_conv_m{m}_k{k}",
            [("L_self_ref", refs[mode]), ("err_nz48", errs[mode][-1])],
        )
    save_pdf(fig, "cyl_quad_convergence")


if __name__ == "__main__":
    main()
