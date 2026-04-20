"""Figure: Smythe thin-disc limit for an ideal-diamagnetic puck.

For a zero-thickness perfectly-conducting disc of radius :math:`a` in a
uniform axial background field :math:`B_0\\hat z`, Smythe (1950) gives
the induced magnetic dipole moment

.. math::

    |m_{\\text{Smythe}}| = \\frac{4 a^3}{3 \\mu_0} B_0 .

This script sweeps puck aspect ratio :math:`t/R` down toward the thin-
disc limit for a single puck centered in a nearly-uniform field generated
by a large ring coil.  The induced moment is computed directly from the
reconstructed sheet current :math:`\\mathbf K` by
:math:`\\mathbf m = \\tfrac{1}{2}\\int \\mathbf r\\times\\mathbf K\\,dS`
and normalized by the Smythe value.  Two curves compare the default
(``exact_disc_faces=False``) path to the semi-analytic intra-puck self-
block path (``exact_disc_faces=True``): only the latter is expected to
approach the thin-disc limit.  The plot mirrors the sweep exercised in
``tests/field/test_passive_bulks_scale.py::test_thin_disc_smythe_limit``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import MU0, approx_B0_from_ring, print_audit, ring_coil, save_pdf, setup_mpl  # noqa: E402

from simsopt.field.psc_bulk import PSCBulkArray  # noqa: E402


def _dipole_moment(psc: PSCBulkArray) -> np.ndarray:
    """Compute :math:`\\tfrac{1}{2}\\int \\mathbf r\\times\\mathbf K\\,dS`."""
    K, _ = psc.get_shell_currents()
    pts = psc._quad_points
    w = psc._quad_weights
    return 0.5 * np.sum(np.cross(pts, K) * w[:, None], axis=0)


def _build_single_puck(
    R: float,
    t: float,
    coil,
    *,
    exact_disc_faces: bool,
) -> PSCBulkArray:
    centers = np.array([[0.0, 0.0, 0.0]])
    axes = np.array([[0.0, 0.0, 1.0]])
    eval_pts = np.array([[2.0 * R, 0.0, 0.0]])
    psc = PSCBulkArray(
        centers, axes, np.array([R]), np.array([t]), [coil],
        eval_points=eval_pts,
        m_fourier=3, l_zernike=6, k_chebyshev=3,
        n_rho=10, n_phi=16, n_z=6,
        nfp=1, stellsym=False, adaptive_self_reg=True,
    )
    if exact_disc_faces:
        psc.exact_disc_faces = True
        psc._rebuild()
    return psc


def main() -> None:
    """Generate and save the Smythe thin-disc figure."""
    setup_mpl()
    R_coil = 5.0
    I_coil = 1.0e7
    R = 0.30
    t_over_R = np.array([0.6, 0.4, 0.3, 0.2, 0.15, 0.1])
    tf = ring_coil(radius=R_coil, z=0.0, current=I_coil, order=1, quadpoints=32)
    B_0 = approx_B0_from_ring(R_coil, I_coil)
    m_ref = 4.0 * R ** 3 * B_0 / (3.0 * MU0)

    ratios_default = []
    ratios_exact = []
    for tR in t_over_R:
        t = R * float(tR)
        psc_d = _build_single_puck(R, t, tf, exact_disc_faces=False)
        psc_e = _build_single_puck(R, t, tf, exact_disc_faces=True)
        m_d = float(np.linalg.norm(_dipole_moment(psc_d)))
        m_e = float(np.linalg.norm(_dipole_moment(psc_e)))
        ratios_default.append(m_d / m_ref)
        ratios_exact.append(m_e / m_ref)
        print(f"[sweep] t/R={tR:.3f} |m|/|m_Smythe| default={m_d/m_ref:.3f}  exact_disc={m_e/m_ref:.3f}")

    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    ax.plot(t_over_R, ratios_default, "-o", label=r"regularized quadrature (default)")
    ax.plot(t_over_R, ratios_exact, "-s", label="semi-analytic intra-puck self (exact_disc_faces=True)")
    ax.axhline(1.0, color="k", ls=":", lw=1.2, label=r"Smythe thin-disc limit")
    ax.set_xlabel(r"$t / R$")
    ax.set_ylabel(r"$|\mathbf{m}_{\text{computed}}| \,/\, |\mathbf{m}_{\text{Smythe}}|$")
    ax.invert_xaxis()
    ax.set_title(r"Thin-disc limit: $R = %.2f$ m, $B_0 \approx %.2e$ T" % (R, B_0))
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    print_audit(
        "smythe_thin_disc",
        [
            ("B_0", B_0),
            ("m_ref_Smythe", m_ref),
            ("ratio_default_tR_0.1", float(ratios_default[-1])),
            ("ratio_exact_tR_0.1", float(ratios_exact[-1])),
        ],
    )
    save_pdf(fig, "smythe_thin_disc")


if __name__ == "__main__":
    main()
