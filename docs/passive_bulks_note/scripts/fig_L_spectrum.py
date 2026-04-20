r"""Figure: spectrum of the assembled shell-inductance matrix :math:`L`.

The reduced system solved for the modal coefficients is

.. math::

    L\,\boldsymbol\beta = \mathbf f , \qquad
    L_{ab} = \frac{\mu_0}{4\pi}\iint_\Sigma\iint_{\Sigma'}
    \frac{(\hat n\times\nabla_s\Phi_a)\cdot(\hat n'\times\nabla_s\Phi_b')}
         {|\mathbf r - \mathbf r'|}\,dS\,dS' .

:math:`L` is real-symmetric and, after projection onto the rim-continuity
null space, positive-semidefinite.  Its lowest modes correspond to
pure-gauge gradients of the scalar potential on each face, which are
flooded at a small relative threshold in
:func:`~simsopt.field.bulk_inductance.eigenfloor_projected_solve`.
This figure plots the sorted eigenvalues of :math:`Q_c^\top L Q_c`
(where :math:`Q_c` is the rim-null-space projector) and highlights the
condition number and the near-null plateau, demonstrating that the
projected operator is numerically PSD and well behaved.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import print_audit, ring_coil, save_pdf, setup_mpl  # noqa: E402

from simsopt.field.psc_bulk import PSCBulkArray  # noqa: E402


def main() -> None:
    """Generate and save the L-spectrum figure."""
    setup_mpl()
    tf = ring_coil(radius=5.0, z=0.0, current=1.0e7, order=1, quadpoints=32)
    R = 0.30
    t = 0.30
    centers = np.array([[0.9, 0.0, 0.0], [-0.9, 0.0, 0.0]])
    quats = np.tile(np.array([0.0, 0.0, 1.0]), (centers.shape[0], 1))

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.6))
    for use_exact, marker, color, label in [
        (False, "-o", "tab:blue", "default regularization"),
        (True, "-s", "tab:orange", r"\texttt{exact\_disc\_faces=True}"),
    ]:
        psc = PSCBulkArray(
            centers, quats,
            np.full(centers.shape[0], R), np.full(centers.shape[0], t),
            [tf],
            eval_points=np.array([[2.0, 0.0, 0.0]]),
            m_fourier=2, l_zernike=4, k_chebyshev=2,
            n_rho=8, n_phi=10, n_z=6,
            nfp=1, stellsym=False, adaptive_self_reg=True,
        )
        psc.exact_disc_faces = bool(use_exact)
        psc._rebuild()
        L = np.asarray(psc._L_work)
        Q = np.asarray(psc._Q_c)
        Lred = Q.T @ L @ Q
        Lred = 0.5 * (Lred + Lred.T)
        w = np.linalg.eigvalsh(Lred)
        w_sorted = np.sort(w)
        positive = w_sorted[w_sorted > 0]
        kappa = float(positive[-1] / positive[0]) if positive.size else float("nan")
        print(f"[sweep] exact_disc_faces={use_exact}  n_modes={w.size}  "
              f"lambda_min={w_sorted[0]:.3e}  lambda_max={w_sorted[-1]:.3e}  kappa~{kappa:.2e}")

        axes[0].semilogy(np.arange(w_sorted.size), np.clip(w_sorted, 1e-30, None),
                         marker, markersize=3, linewidth=1.0, color=color, label=label)
        axes[1].semilogy(np.arange(positive.size),
                         np.sort(positive / positive.max()),
                         marker, markersize=3, linewidth=1.0, color=color, label=label)

    axes[0].set_xlabel("index (sorted)")
    axes[0].set_ylabel(r"$\lambda\ (\mathrm{H})$")
    axes[0].set_title(r"Spectrum of $Q_c^\top L Q_c$")
    axes[0].legend(fontsize=8)

    axes[1].set_xlabel("index (sorted)")
    axes[1].set_ylabel(r"$\lambda / \lambda_{\max}$")
    axes[1].set_title("Normalized spectrum")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    print_audit("L_spectrum", [("note", 0.0)])
    save_pdf(fig, "L_spectrum")


if __name__ == "__main__":
    main()
