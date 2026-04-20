"""Figure: exponential convergence of the semi-analytic disc-disc quadratures.

Sweeps ``n_radial`` for two routines from
:mod:`simsopt.field.disc_self_inductance`:

* :func:`disc_self_block_axisymmetric` (axisymmetric :math:`m = 0` singular
  self-block; Duffy + Gauss-Laguerre rule) and
* :func:`disc_disc_cross_block` (smooth cross-block between two coaxial
  discs at axial separation :math:`\\Delta z = t`; tensor-product
  Gauss-Legendre rule).

Both should converge exponentially in ``n_radial``; the plot shows the
relative distance from a high-resolution reference solve at ``n_radial
= 72`` on a log-y axis.  The linear-in-``n_radial`` envelope on a log
scale is the expected hallmark of spectral convergence and rules out the
regularization-limited error floors seen on the legacy
``1/sqrt(r^2 + delta^2)`` quadrature.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import print_audit, save_pdf, setup_mpl  # noqa: E402

from simsopt.field.disc_self_inductance import (  # noqa: E402
    disc_disc_cross_block,
    disc_self_block_axisymmetric,
    disc_self_block_mode_m,
)


def _make_radial_basis(R: float, m: int, n: int):
    """Return ``(f, f')`` for the Zernike radial part ``R_n^m(rho / R)``."""
    from simsopt.field.disc_self_inductance import (  # local import
        _zernike_radial_derivative_numpy as _dR,
        _zernike_radial_numpy as _R_nm,
    )

    def f(rho: np.ndarray) -> np.ndarray:
        r = np.clip(np.asarray(rho) / R, 0.0, 1.0)
        return _R_nm(r, m, n)

    def fp(rho: np.ndarray) -> np.ndarray:
        r = np.clip(np.asarray(rho) / R, 0.0, 1.0)
        return (1.0 / R) * _dR(r, m, n)

    return f, fp


def main() -> None:
    """Generate and save the disc-quadrature convergence figure."""
    setup_mpl()
    R = 0.30  # disc radius (m)
    t = 0.30  # axial separation for the cross-block (m)
    modes = [(0, 2), (1, 3), (2, 4)]  # (m, n)
    ns = np.array([4, 6, 8, 12, 16, 24, 32, 48], dtype=int)
    n_ref = 72

    # -------- self block (singular) ------------------------------------
    self_ref = {}
    self_err = {mode: [] for mode in modes}
    for m, n in modes:
        f, fp = _make_radial_basis(R, m, n)
        if m == 0:
            ref = disc_self_block_axisymmetric(fp, fp, R, n_radial=n_ref)
        else:
            ref = disc_self_block_mode_m(f, fp, f, fp, m, R, n_radial=n_ref)
        self_ref[(m, n)] = ref
        for nq in ns:
            if m == 0:
                val = disc_self_block_axisymmetric(fp, fp, R, n_radial=int(nq))
            else:
                val = disc_self_block_mode_m(f, fp, f, fp, m, R, n_radial=int(nq))
            self_err[(m, n)].append(abs(val - ref) / (abs(ref) + 1e-30))

    # -------- cross block (smooth) --------------------------------------
    cross_ref = {}
    cross_err = {mode: [] for mode in modes}
    for m, n in modes:
        f, fp = _make_radial_basis(R, m, n)
        ref = disc_disc_cross_block(f, fp, f, fp, m, R, R, t, n_radial=n_ref)
        cross_ref[(m, n)] = ref
        for nq in ns:
            val = disc_disc_cross_block(f, fp, f, fp, m, R, R, t, n_radial=int(nq))
            cross_err[(m, n)].append(abs(val - ref) / (abs(ref) + 1e-30))

    # -------- plot ------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), sharey=True)
    markers = ["o", "s", "^"]
    for ax, data, title in (
        (axes[0], self_err, r"disc self (singular, $m \geq 0$)"),
        (axes[1], cross_err, r"disc$\times$disc cross at $\Delta z = t$ (smooth)"),
    ):
        for (mode, marker) in zip(modes, markers):
            m, n = mode
            ax.semilogy(ns, data[mode], f"-{marker}", label=rf"$(m, n) = ({m}, {n})$")
        ax.set_xlabel(r"$n_{\text{radial}}$")
        ax.set_title(title)
        ax.set_ylim(1.0e-16, 1.0)
        ax.grid(True, which="both", alpha=0.35)
    axes[0].set_ylabel(r"rel.\ error vs $n_{\text{radial}} = %d$" % n_ref)
    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle(r"Spectral convergence of disc self/cross-block quadratures ($R = %.2f$ m, $t = %.2f$ m)"
                 % (R, t), y=1.02)
    fig.tight_layout()

    for (m, n) in modes:
        print_audit(
            f"disc_quad_conv_m{m}_n{n}",
            [
                ("L_self_ref", self_ref[(m, n)]),
                ("L_cross_ref", cross_ref[(m, n)]),
                ("self_err_nq48", self_err[(m, n)][-1]),
                ("cross_err_nq48", cross_err[(m, n)][-1]),
            ],
        )
    save_pdf(fig, "disc_quad_convergence")


if __name__ == "__main__":
    main()
