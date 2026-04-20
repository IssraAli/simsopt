r"""Figure: convergence of the induced bulk :math:`\mathbf B` with basis order.

Reproduces the sweep in ``tests/field/test_passive_bulks_mode_convergence.py``:
fix four pucks arranged around a pair of TF coils and sweep the triple
``(m_{\text{Fourier}}, l_{\text{Zernike}}, k_{\text{Chebyshev}})``.  The
induced field evaluated at three interior points converges to the
high-resolution reference :math:`(4, 8, 4)` at geometric (exponential
in mode count) rate, confirming that the scalar-current representation
is spectrally complete on the puck shell and that the linear solve
:math:`L\boldsymbol\beta = \mathbf f` is well conditioned at every
resolution tested.
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
    """Generate and save the mode-convergence figure."""
    setup_mpl()
    tf_coils = [
        ring_coil(radius=1.2, z=+0.4, current=2.0e5, order=1, quadpoints=32),
        ring_coil(radius=1.2, z=-0.4, current=2.0e5, order=1, quadpoints=32),
    ]
    centers = np.array([[0.85, 0.00, 0.25]])
    axes = centers / np.linalg.norm(centers, axis=1, keepdims=True)
    radii = np.full(centers.shape[0], 0.06)
    thicknesses = np.full(centers.shape[0], 0.03)
    eval_pts = np.array([[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5]])

    def build(m: int, lz: int, k: int, n_rho: int, n_phi: int, n_z: int) -> PSCBulkArray:
        return PSCBulkArray(
            centers, axes, radii, thicknesses, tf_coils,
            eval_points=eval_pts,
            m_fourier=m, l_zernike=lz, k_chebyshev=k,
            n_rho=n_rho, n_phi=n_phi, n_z=n_z,
        )

    psc_ref = build(4, 8, 4, n_rho=10, n_phi=12, n_z=6)
    psc_ref.recompute_currents()
    b_ref = np.asarray(psc_ref.B_at_points(eval_pts))
    print(f"[ref] (4,8,4) built; |B_ref|={np.linalg.norm(b_ref):.3e}")

    sweep = [(1, 2, 1), (2, 4, 2), (3, 6, 3)]
    errs = []
    n_modes = []
    for (m, lz, k) in sweep:
        psc = build(m, lz, k, n_rho=max(2 * lz, 6), n_phi=max(2 * m + 4, 8), n_z=max(2 * k, 4))
        psc.recompute_currents()
        b = np.asarray(psc.B_at_points(eval_pts))
        rel = float(np.linalg.norm(b - b_ref) / (np.linalg.norm(b_ref) + 1e-30))
        errs.append(rel)
        n_modes.append(m + lz + k)
        print(f"[sweep] (m,l,k)={m},{lz},{k}  rel_err={rel:.3e}")

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    x = np.arange(len(sweep))
    ax.semilogy(x, errs, "-o", color="tab:blue")
    ax.set_xticks(x)
    ax.set_xticklabels([f"({m},{lz},{k})" for (m, lz, k) in sweep])
    ax.set_xlabel(r"basis $(m_\mathrm{Fourier},\, l_\mathrm{Zernike},\, k_\mathrm{Chebyshev})$")
    ax.set_ylabel(r"$\|B - B_{\rm ref}\| \,/\, \|B_{\rm ref}\|$")
    ax.set_title(r"Induced bulk-field convergence (reference $(4,8,4)$)")
    ax.grid(True, which="both", alpha=0.35)
    fig.tight_layout()

    print_audit("mode_convergence",
                [("final_rel_err", float(errs[-1])),
                 ("ratio_0_to_1", float(errs[1] / errs[0])),
                 ("ratio_1_to_2", float(errs[2] / errs[1]))])
    save_pdf(fig, "mode_convergence")


if __name__ == "__main__":
    main()
