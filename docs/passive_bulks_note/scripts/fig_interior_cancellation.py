r"""Figure: interior-field cancellation inside an ideal-diamagnetic puck.

For a superconducting body in an externally applied field, the induced
surface currents must drive the total field to zero in the interior.
This figure compares the two solver formulations exposed by
:class:`~simsopt.field.psc_bulk.PSCBulkArray`:

* ``solver_mode="energy"``: the original magnetostatic-energy Gram-matrix
  formulation, which enforces the Meissner condition only in a Galerkin
  sense (:math:`\int \Phi_a B_n^{\rm tot}\,dS = 0`); and
* ``solver_mode="shell_l2"``: the new REGCOIL-style
  :math:`L^2`-residual formulation, which minimises
  :math:`\|B_n^{\rm TF} + B_n^{\rm ind}\|_{L^2(\partial V)}^2` directly.

The two are *not* variationally equivalent: the weak-form "energy" mode
reproduces the correct far-field (dipole moment) but leaves an
:math:`O(1)` residual on the shell, and therefore in the interior.
The :math:`L^2`-residual solver attacks the shell residual directly and
drives the interior field toward zero.

A thin, small puck
(:math:`R/R_{\rm coil}\sim 10^{-2}`, :math:`t/R = 0.1`) is placed at the
centre of a large ring TF coil so the TF field is locally uniform to
better than :math:`10^{-4}` on the puck scale (same geometry as
``tests/field/test_passive_bulks_scale.py::test_single_puck_induced_dipole_direction``).
We sweep the basis resolution
:math:`(m_{\rm Fourier}, l_{\rm Zernike}, k_{\rm Chebyshev})` and plot

.. math::

    \rho = \frac{|B_{\rm total}(0)|}{|B_{\rm TF}(0)|}

for both solver modes.  Only odd :math:`m, k` (with :math:`l = 2m`) are
swept: for the thin-disc geometry the symmetric Chebyshev basis has a
parity structure such that purely axial (``k`` even) basis functions
carry no contribution to :math:`B_n` on the disc faces, so the shell
:math:`L^2` residual is insensitive to those DOF and convergence is
cleanest when monitored on the odd-index subsequence.  The horizontal reference at
:math:`(R/R_{\rm coil})^2` shows the geometric ceiling set by TF
non-uniformity (quadratic correction to the uniform-field idealisation);
any residual above this line is numerical, not physical.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import approx_B0_from_ring, print_audit, ring_coil, save_pdf, setup_mpl  # noqa: E402

from simsopt.field.biotsavart import BiotSavart  # noqa: E402
from simsopt.field.psc_bulk import PSCBulkArray  # noqa: E402


def _ratio(psc: PSCBulkArray, tf_coil) -> float:
    """Return :math:`|B_{\\text{total}}(0)|/|B_{\\text{TF}}(0)|`."""
    probe = np.array([[0.0, 0.0, 0.0]])
    bs = BiotSavart([tf_coil])
    bs.set_points_cart(np.ascontiguousarray(probe))
    B_tf = np.asarray(bs.B())[0]
    B_ind = np.asarray(psc.B_at_points(probe))[0]
    return float(np.linalg.norm(B_tf + B_ind) / (np.linalg.norm(B_tf) + 1e-30))


def _build_and_ratio(R, t, tf, m, lz, k, *, solver_mode: str):
    """Build a single-puck ``PSCBulkArray`` at the given basis and return the ratio.

    Quadrature is chosen generously enough that the *integration* error is
    well below the basis-truncation error, so the sweep reflects basis
    convergence rather than quadrature noise.
    """
    centers = np.array([[0.0, 0.0, 0.0]])
    axes = np.array([[0.0, 0.0, 1.0]])
    eval_pts = np.array([[2.0 * R, 0.0, 0.0]])
    n_rho = max(2 * lz, 8)
    n_phi = max(2 * m + 4, 10)
    n_z = max(2 * k, 6)
    psc = PSCBulkArray(
        centers, axes, np.array([R]), np.array([t]), [tf],
        eval_points=eval_pts,
        m_fourier=m, l_zernike=lz, k_chebyshev=k,
        n_rho=n_rho, n_phi=n_phi, n_z=n_z,
        nfp=1, stellsym=False, adaptive_self_reg=True,
        solver_mode=solver_mode,
    )
    return _ratio(psc, tf)


def main() -> None:
    """Generate and save the interior-cancellation figure."""
    setup_mpl()
    R_coil = 5.0
    I_coil = 1.0e7
    R = 0.05
    t = 0.005
    tf = ring_coil(radius=R_coil, z=0.0, current=I_coil, order=1, quadpoints=64)
    B_0 = approx_B0_from_ring(R_coil, I_coil)
    geom_floor = (R / R_coil) ** 2

    resolutions = [(1, 2, 1), (3, 6, 3), (5, 10, 5)]
    r_energy: list[float] = []
    r_l2: list[float] = []
    for (m, lz, k) in resolutions:
        re = _build_and_ratio(R, t, tf, m, lz, k, solver_mode="energy")
        rl = _build_and_ratio(R, t, tf, m, lz, k, solver_mode="shell_l2")
        r_energy.append(re)
        r_l2.append(rl)
        print(
            f"[sweep] (m,l,k)=({m},{lz},{k})  "
            f"ratio_energy={re:.3e}  ratio_shell_l2={rl:.3e}"
        )

    labels = [rf"({m},{lz},{k})" for (m, lz, k) in resolutions]
    x = np.arange(len(resolutions))

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.semilogy(x, r_energy, "-o", label=r"energy Gram-matrix (weak form)")
    ax.semilogy(x, r_l2, "-s", label=r"$L^2$-residual (REGCOIL-style)")
    ax.axhline(geom_floor, color="k", ls=":", lw=1.2,
               label=rf"$(R/R_{{\rm coil}})^2 = {geom_floor:.1e}$")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel(r"basis resolution $(m_{\rm Fourier},\, l_{\rm Zernike},\, k_{\rm Chebyshev})$")
    ax.set_ylabel(r"$|B_{\rm total}(0)| \,/\, |B_{\rm TF}(0)|$")
    ax.set_title(
        r"Thin-disc interior cancellation: energy vs $L^2$-residual "
        r"($R=%.2f$ m, $t/R=%.2f$, $R_{\rm coil}=%.1f$ m, $B_0\approx%.2e$ T)"
        % (R, t / R, R_coil, B_0)
    )
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, which="both", alpha=0.35)
    fig.tight_layout()

    print_audit(
        "interior_cancellation",
        [("B_0", B_0), ("geom_floor", geom_floor),
         ("ratio_energy_fine", r_energy[-1]),
         ("ratio_shell_l2_fine", r_l2[-1])],
    )
    save_pdf(fig, "interior_cancellation")


if __name__ == "__main__":
    main()
