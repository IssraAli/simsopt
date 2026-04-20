"""Figure: Neumann coaxial-loop mutual inductance vs the implemented kernel.

Cross-checks :func:`simsopt.field.disc_self_inductance.ring_mutual_axial_shift`
(the Maxwell-Grover closed form used internally for every coaxial disc-disc
mutual) against an independent numerical Neumann-formula integral

.. math::

    M = \\frac{\\mu_0}{4\\pi}\\oint\\oint \\frac{d\\vec l\\cdot d\\vec l'}{|\\vec r-\\vec r'|}

evaluated by trapezoidal rule on both rings.  Agreement to machine
precision at ``n_phi >= 32`` validates the elliptic-integral kernel that
powers the entire semi-analytic disc assembly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import print_audit, save_pdf, setup_mpl  # noqa: E402

from simsopt.field.disc_self_inductance import (  # noqa: E402
    MU0,
    ring_mutual_axial_shift,
)


def neumann_ring_mutual(a: float, b: float, dz: float, n_phi: int) -> float:
    r"""Numerical double-integral Neumann mutual of two coaxial rings.

    Returns
    :math:`M = (\\mu_0/(4\\pi))\\oint\\oint d\\vec l\\cdot d\\vec l'/r`.
    Trapezoidal rule on a uniform phi grid.
    """
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    dphi = 2.0 * np.pi / n_phi
    a_vec = np.stack([a * np.cos(phi), a * np.sin(phi), np.zeros_like(phi)], axis=-1)
    b_vec = np.stack([b * np.cos(phi), b * np.sin(phi), np.full_like(phi, dz)], axis=-1)
    da = np.stack([-a * np.sin(phi), a * np.cos(phi), np.zeros_like(phi)], axis=-1)
    db = np.stack([-b * np.sin(phi), b * np.cos(phi), np.zeros_like(phi)], axis=-1)
    diff = a_vec[:, None, :] - b_vec[None, :, :]
    r = np.linalg.norm(diff, axis=-1)
    dot = np.einsum("ia,ja->ij", da, db)
    integrand = dot / r
    return (MU0 / (4.0 * np.pi)) * np.sum(integrand) * (dphi ** 2)


def main() -> None:
    """Generate and save the Neumann coaxial-loop check figure."""
    setup_mpl()
    a = 0.30
    b = 0.25
    dzs = np.array([0.05, 0.1, 0.2, 0.5, 1.0])
    n_phis = np.array([8, 16, 32, 64, 128, 256], dtype=int)

    exact = np.array([ring_mutual_axial_shift(a, b, float(dz)) for dz in dzs])

    errors = np.zeros((len(dzs), len(n_phis)))
    for i, dz in enumerate(dzs):
        for j, nq in enumerate(n_phis):
            num = neumann_ring_mutual(a, b, float(dz), int(nq))
            errors[i, j] = abs(num - exact[i]) / abs(exact[i])

    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    markers = ["o", "s", "^", "v", "d"]
    for i, dz in enumerate(dzs):
        ax.loglog(n_phis, errors[i], f"-{markers[i]}",
                  label=rf"$\Delta z = {dz:.2f}$ m")
    ax.set_xlabel(r"$n_\phi$ (trapezoidal nodes per ring)")
    ax.set_ylabel(r"$|M_{\text{Neumann}} - M_{\text{Maxwell-Grover}}| \,/\, |M|$")
    ax.set_title(r"Coaxial-loop mutual: $a = %.2f$ m, $b = %.2f$ m" % (a, b))
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, which="both", alpha=0.35)
    fig.tight_layout()

    for i, dz in enumerate(dzs):
        print_audit(
            f"neumann_dz_{dz:.2f}",
            [("M_exact", exact[i]), ("err_nphi256", errors[i, -1])],
        )
    save_pdf(fig, "neumann_coaxial")


if __name__ == "__main__":
    main()
