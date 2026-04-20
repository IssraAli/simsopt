r"""Figure: rim-continuity residual for solved-in modes.

The scalar current potential ``g`` must be continuous across the two
puck rims (top-edge-to-side and bottom-edge-to-side).  This is enforced
by a linear constraint :math:`C\,\boldsymbol\beta = 0` assembled by
:func:`~simsopt.field.puck_basis.build_continuity_constraint`, and the
solver restricts :math:`\boldsymbol\beta` to :math:`\operatorname{null}(C)`
by SVD.  In this figure we:

1. solve for :math:`\boldsymbol\beta` at a representative basis size;
2. evaluate :math:`\|C\boldsymbol\beta\|_\infty` for every puck in the
   array;
3. plot it together with the rim-collocation count (``n_phi_rim``).

The residual should sit at float64 round-off (:math:`\sim 10^{-12}`)
independent of the basis resolution, which is the figure's message.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import print_audit, ring_coil, save_pdf, setup_mpl  # noqa: E402

from simsopt.field.psc_bulk import PSCBulkArray  # noqa: E402
from simsopt.field.puck_basis import build_continuity_constraint  # noqa: E402


def main() -> None:
    """Generate and save the rim continuity figure."""
    setup_mpl()
    tf = ring_coil(radius=5.0, z=0.0, current=1.0e7, order=1, quadpoints=32)
    R = 0.30
    t = 0.30
    centers = np.array([
        [0.9, 0.0, 0.0],
        [-0.9, 0.0, 0.0],
    ])
    quats = np.tile(np.array([0.0, 0.0, 1.0]), (centers.shape[0], 1))
    configs = [(1, 2, 1, 8), (2, 4, 2, 12), (3, 6, 3, 16)]
    residuals_per_config = []
    xlabels = []
    for (m, lz, k, n_rim) in configs:
        psc = PSCBulkArray(
            centers, quats,
            np.full(centers.shape[0], R), np.full(centers.shape[0], t),
            [tf],
            eval_points=np.array([[2.0, 0.0, 0.0]]),
            m_fourier=m, l_zernike=lz, k_chebyshev=k,
            n_rho=max(2 * lz, 8), n_phi=max(2 * m + 4, 8), n_z=max(2 * k, 4),
            n_phi_rim=n_rim,
            nfp=1, stellsym=False, adaptive_self_reg=True,
        )
        psc._rebuild()
        beta = np.asarray(psc.beta)
        per_puck = []
        for i in range(centers.shape[0]):
            d0 = psc._dof_offsets[i]
            d1 = d0 + psc._basis_per_puck[i].phi_values.shape[1]
            beta_i = beta[d0:d1]
            C = build_continuity_constraint(
                psc._basis_per_puck[i], R=R, t=t, n_rim=n_rim,
            )
            res = float(np.max(np.abs(C @ beta_i))) if C.size else 0.0
            per_puck.append(res)
        residuals_per_config.append(per_puck)
        xlabels.append(f"({m},{lz},{k})\nn_rim={n_rim}")
        print(f"[sweep] cfg=({m},{lz},{k},{n_rim})  max_rim_residual={max(per_puck):.3e}")

    residuals_per_config = np.array(residuals_per_config)

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    x = np.arange(residuals_per_config.shape[0])
    for j in range(residuals_per_config.shape[1]):
        ax.semilogy(x, residuals_per_config[:, j], "-o", label=f"puck {j}")
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)
    ax.set_ylabel(r"$\|C\beta\|_\infty$")
    ax.set_title(r"Rim continuity residual vs. basis resolution")
    ax.grid(True, which="both", alpha=0.35)
    ax.legend(fontsize=8)
    fig.tight_layout()

    print_audit(
        "rim_continuity",
        [("max_residual_finest", float(np.max(residuals_per_config[-1])))],
    )
    save_pdf(fig, "rim_continuity")


if __name__ == "__main__":
    main()
