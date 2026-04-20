"""Figure 0: geometry schematic of a single cylindrical puck.

Renders the three surface patches (top/bottom caps and side wall) of a
:class:`~simsopt.field.psc_bulk.PSCBulkArray` puck in its local frame,
together with the outward-normal arrow on each patch and the two rim
circles where :func:`~simsopt.field.puck_basis.build_continuity_constraint`
enforces :math:`g` continuity.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Allow running ``python scripts/fig_puck_schematic.py`` from the repo root
# or from inside ``docs/passive_bulks_note``.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import print_audit, save_pdf, setup_mpl  # noqa: E402


def main() -> None:
    """Render and save the puck-geometry schematic."""
    setup_mpl()
    R = 0.3
    t = 0.25
    n_phi = 80

    fig = plt.figure(figsize=(5.0, 4.6))
    ax = fig.add_subplot(111, projection="3d")

    phi = np.linspace(0.0, 2.0 * np.pi, n_phi)
    x_rim = R * np.cos(phi)
    y_rim = R * np.sin(phi)

    # Top/bottom disk patches (mesh of radial rings).
    rhos = np.linspace(0.0, R, 12)
    for rho in rhos:
        ax.plot(rho * np.cos(phi), rho * np.sin(phi), np.full_like(phi, +0.5 * t),
                color="tab:blue", alpha=0.3, lw=0.6)
        ax.plot(rho * np.cos(phi), rho * np.sin(phi), np.full_like(phi, -0.5 * t),
                color="tab:orange", alpha=0.3, lw=0.6)

    # Side wall (grid of meridians).
    zs = np.linspace(-0.5 * t, 0.5 * t, 10)
    for phi_m in np.linspace(0.0, 2.0 * np.pi, 18, endpoint=False):
        ax.plot([R * np.cos(phi_m)] * len(zs),
                [R * np.sin(phi_m)] * len(zs),
                zs, color="tab:green", alpha=0.35, lw=0.6)

    # Rim circles (top/bot).
    ax.plot(x_rim, y_rim, np.full_like(phi, +0.5 * t),
            color="k", lw=1.8, label=r"top rim ($C\beta = 0$)")
    ax.plot(x_rim, y_rim, np.full_like(phi, -0.5 * t),
            color="k", lw=1.8, ls="--", label=r"bottom rim")

    # Outward normals on each patch (short arrows).
    arrow_len = 0.18 * R
    ax.quiver(0, 0, +0.5 * t, 0, 0, +arrow_len, color="tab:blue",
              label=r"top $\hat n = +\hat z$", arrow_length_ratio=0.25)
    ax.quiver(0, 0, -0.5 * t, 0, 0, -arrow_len, color="tab:orange",
              label=r"bottom $\hat n = -\hat z$", arrow_length_ratio=0.25)
    ax.quiver(R, 0, 0, arrow_len, 0, 0, color="tab:green",
              label=r"side $\hat n = \hat\rho$", arrow_length_ratio=0.25)

    # Axes / frame.
    ax.set_xlabel(r"$x$ (m)")
    ax.set_ylabel(r"$y$ (m)")
    ax.set_zlabel(r"$z$ (m)")
    ax.set_box_aspect((1.0, 1.0, 1.4 * t / R))
    ax.set_title(r"Puck geometry: three surface patches $\Sigma = \Sigma_t \cup \Sigma_b \cup \Sigma_s$")
    ax.view_init(elev=22, azim=-60)
    ax.legend(loc="upper left", fontsize=8)

    print_audit("puck_schematic", [("R", R), ("t", t), ("n_phi_mesh", float(n_phi))])
    save_pdf(fig, "puck_schematic")


if __name__ == "__main__":
    main()
