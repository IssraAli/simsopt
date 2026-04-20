r"""Figure: Meissner-sphere dipole limit for a stubby cylindrical puck.

For a perfectly-conducting (Meissner) sphere of radius :math:`a` placed
in a uniform external field :math:`B_0 \hat z`, the induced magnetic
moment is

.. math::

    \mathbf m_{\text{sphere}} = -2\pi a^3\,B_0 / \mu_0\,\hat z .

A stubby right-circular cylinder (:math:`R = t`) is the closest single-
puck approximation to a sphere; its ideal-diamagnet moment should be
aligned with :math:`-\hat z` (Lenz's law) and of magnitude within an
``O(1)`` geometric factor of the sphere value.

To expose *basis-only* convergence we fix

* ``exact_disc_faces=True`` (semi-analytic intra-puck self-block) for
  every sweep point, so intra-puck singularities are handled by the
  same routine regardless of basis size,
* the null-space and eigenfloor thresholds explicitly, so the gauge /
  rim-null-space projector :math:`Q_L` retains the same modes as the
  basis is refined,
* quadrature orders at a fixed (large) value so the observed variation
  is attributable to basis-refinement, not quadrature noise.

Two normalisations are shown:

* left: :math:`|m_z|/|m_{\rm sphere}|` versus the equivalent-volume
  sphere, highlighting the intrinsic geometric mismatch cylinder vs.
  sphere,
* right: the Lenz-sign check :math:`m_z<0`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import MU0, approx_B0_from_ring, print_audit, ring_coil, save_pdf, setup_mpl  # noqa: E402

from simsopt.field.psc_bulk import PSCBulkArray  # noqa: E402


_NULL_SPACE_THRESHOLD = 1e-8
_EIGENFLOOR_THRESHOLD = 1e-10
_QUAD_RHO = 8
_QUAD_PHI = 16
_QUAD_Z = 6


def _moment(psc: PSCBulkArray) -> np.ndarray:
    """Return the total induced magnetic dipole moment :math:`\\mathbf m`."""
    K, _ = psc.get_shell_currents()
    pts = psc._quad_points
    w = psc._quad_weights
    return 0.5 * np.sum(np.cross(pts, K) * w[:, None], axis=0)


def _build_psc(R: float, t: float, tf, m: int, lz: int, k: int) -> PSCBulkArray:
    """Build a single-puck PSCBulkArray with fixed thresholds/quadrature."""
    psc = PSCBulkArray(
        np.array([[0.0, 0.0, 0.0]]),
        np.array([[0.0, 0.0, 1.0]]),
        np.array([R]), np.array([t]), [tf],
        eval_points=np.array([[2.0 * R, 0.0, 0.0]]),
        m_fourier=m, l_zernike=lz, k_chebyshev=k,
        n_rho=_QUAD_RHO, n_phi=_QUAD_PHI, n_z=_QUAD_Z,
        nfp=1, stellsym=False, adaptive_self_reg=True,
        null_space_threshold=_NULL_SPACE_THRESHOLD,
        eigenfloor_threshold=_EIGENFLOOR_THRESHOLD,
    )
    psc.exact_disc_faces = True
    psc._rebuild()
    return psc


def main() -> None:
    """Generate and save the Meissner-sphere figure."""
    setup_mpl()
    R_coil = 5.0
    I_coil = 1.0e7
    R = 0.30
    t = 0.30
    tf = ring_coil(radius=R_coil, z=0.0, current=I_coil, order=1, quadpoints=32)
    B_0 = approx_B0_from_ring(R_coil, I_coil)
    volume = np.pi * R ** 2 * t
    a_eq = (3.0 * volume / (4.0 * np.pi)) ** (1.0 / 3.0)
    m_sphere = 2.0 * np.pi * a_eq ** 3 * B_0 / MU0

    resolutions = [(1, 2, 1), (2, 4, 2), (3, 6, 3), (4, 8, 4)]
    labels = [rf"({m},{lz},{k})" for (m, lz, k) in resolutions]
    ratios_sphere = []
    mz_signs = []
    for (m, lz, k) in resolutions:
        psc = _build_psc(R, t, tf, m, lz, k)
        mvec = _moment(psc)
        ratios_sphere.append(abs(mvec[2]) / m_sphere)
        mz_signs.append(float(mvec[2]))
        print(f"[sweep] ({m},{lz},{k}) |m_z|/|m_sphere|={ratios_sphere[-1]:.3f}  m_z={mvec[2]:.3e}")

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4))
    x = np.arange(len(resolutions))

    axes[0].plot(x, ratios_sphere, "-s", color="tab:blue", label=r"$|m_z|/|m_{\rm sphere}|$")
    axes[0].axhline(1.0, color="k", ls=":", lw=1.0, label="sphere reference")
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels)
    axes[0].set_xlabel(r"basis $(m_{\rm F},\, l_{\rm Z},\, k_{\rm C})$")
    axes[0].set_ylabel(r"dimensionless induced dipole")
    axes[0].set_title(r"Cylinder vs.\ equivalent-volume sphere ($a_{\rm eq}=%.3f$ m)" % a_eq)
    axes[0].set_ylim(0.0, 1.1)
    axes[0].grid(True, which="both", alpha=0.35)
    axes[0].legend(loc="best", fontsize=8)

    axes[1].plot(x, mz_signs, "-o", color="tab:red")
    axes[1].axhline(0.0, color="k", ls="-", lw=0.6)
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels)
    axes[1].set_xlabel(r"basis $(m_{\rm F},\, l_{\rm Z},\, k_{\rm C})$")
    axes[1].set_ylabel(r"$m_z$ (A m$^2$)")
    axes[1].set_title(r"Lenz-sign check ($m_z<0$ expected)")
    axes[1].grid(True, which="both", alpha=0.35)

    fig.tight_layout()
    print_audit(
        "sphere_meissner",
        [("B_0", B_0), ("a_eq", a_eq), ("m_sphere", m_sphere),
         ("ratio_sphere_fine", ratios_sphere[-1]),
         ("mz_fine", mz_signs[-1])],
    )
    save_pdf(fig, "sphere_meissner")


if __name__ == "__main__":
    main()
