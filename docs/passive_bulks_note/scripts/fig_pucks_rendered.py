r"""Figure: rendered puck geometry with surface currents.

Builds a small array of passive pucks around a pair of ring TF coils,
solves the shell system, and visualises in 3D:

* each puck as a scatter of its quadrature points, coloured by
  :math:`|\mathbf K|` (A/m) from :meth:`PSCBulkArray.get_shell_currents`;
* the TF ring coils as thick black curves;
* the evaluation plane as a faint rectangular patch.

This is the inline-figure counterpart of the VTK exports produced by
the ``examples/3_Advanced/passive_bulks_*`` scripts.  Only matplotlib
is used, so the figure renders headless (CI friendly) and the user can
re-run it without a VTK viewer.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d projection)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import print_audit, ring_coil, save_pdf, setup_mpl  # noqa: E402

from simsopt.field.psc_bulk import PSCBulkArray  # noqa: E402


def main() -> None:
    """Generate and save the puck-rendered figure."""
    setup_mpl()
    tf_coils = [
        ring_coil(radius=1.2, z=+0.4, current=2.0e5, order=1, quadpoints=64),
        ring_coil(radius=1.2, z=-0.4, current=2.0e5, order=1, quadpoints=64),
    ]
    centers = np.array([
        [0.85, 0.00, 0.25],
        [0.00, 0.85, 0.25],
        [-0.85, 0.00, 0.25],
        [0.00, -0.85, 0.25],
        [0.85, 0.00, -0.25],
        [0.00, 0.85, -0.25],
        [-0.85, 0.00, -0.25],
        [0.00, -0.85, -0.25],
    ])
    axes = centers / np.linalg.norm(centers, axis=1, keepdims=True)
    radii = np.full(centers.shape[0], 0.09)
    thicknesses = np.full(centers.shape[0], 0.04)
    eval_pts = np.array([[0.5, 0.0, 0.0]])
    psc = PSCBulkArray(
        centers, axes, radii, thicknesses, tf_coils,
        eval_points=eval_pts,
        m_fourier=2, l_zernike=3, k_chebyshev=2,
        n_rho=6, n_phi=10, n_z=4,
    )
    psc.recompute_currents()
    pts = np.asarray(psc._quad_points)
    _, K_mag = psc.get_shell_currents()

    fig = plt.figure(figsize=(7.2, 6.0))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                    c=K_mag, cmap="viridis", s=4, depthshade=False)
    for coil in tf_coils:
        g = np.asarray(coil.curve.gamma())
        g = np.vstack([g, g[:1]])
        ax.plot(g[:, 0], g[:, 1], g[:, 2], color="k", lw=1.8)

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title(r"Induced sheet current $|K|$ on puck shells (A/m)")
    cbar = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label(r"$|K|$ (A/m)")
    lim = 1.4
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-0.6, 0.6)
    ax.view_init(elev=22, azim=35)
    fig.tight_layout()

    print_audit(
        "pucks_rendered",
        [("n_pucks", float(centers.shape[0])),
         ("max_K", float(K_mag.max())),
         ("mean_K", float(K_mag.mean()))],
    )
    save_pdf(fig, "pucks_rendered")


if __name__ == "__main__":
    main()
