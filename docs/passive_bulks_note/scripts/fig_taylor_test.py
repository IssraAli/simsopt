r"""Figure: Taylor tests for the passive-bulk adjoint gradients.

For a scalar objective :math:`J(\mathbf x)`, the central finite
difference

.. math::

    \widetilde{\mathbf d^\top\nabla J}(\mathbf x; \epsilon)
    = \frac{J(\mathbf x + \epsilon\mathbf d) - J(\mathbf x - \epsilon\mathbf d)}
           {2\epsilon}

must approach :math:`\mathbf d^\top\nabla J(\mathbf x)` with relative
error :math:`O(\epsilon^2)` for smooth :math:`J`.  This figure
exercises both families of gradient paths inside
:class:`~simsopt.field.psc_bulk.PSCBulkArray`:

* **TF-only**: all puck geometry DOFs are fixed, only TF coil DOFs are
  unfrozen.  The adjoint follows the analytic/JAX path selected by
  ``_USE_JAX_TF_VJP``.
* **Puck geometry**: puck centers and quaternion axes are unfrozen in
  addition to the TF DOFs, which triggers the full JAX VJP path.

The two corresponding relative-error curves should each track the
dashed :math:`\epsilon^2` reference slope from roughly
``eps ~ 1e-2`` down until round-off kicks in near ``1e-7``.  This
mirrors the ``test_taylor_*`` unit tests in
``tests/field/test_passive_bulks.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import print_audit, save_pdf, setup_mpl  # noqa: E402

from simsopt.field import BiotSavart, Current, coils_via_symmetries  # noqa: E402
from simsopt.field.magneticfield import MagneticFieldSum  # noqa: E402
from simsopt.field.psc_bulk import PSCBulkArray  # noqa: E402
from simsopt.geo import SurfaceRZFourier, create_equally_spaced_curves  # noqa: E402
from simsopt.objectives import SquaredFlux  # noqa: E402


def _build() -> tuple[PSCBulkArray, MagneticFieldSum, SquaredFlux, list]:
    TEST_DIR = (Path(__file__).resolve().parent.parent.parent.parent /
                "tests" / "test_files").resolve()
    filename = TEST_DIR / "input.LandremanPaul2021_QA"
    s = SurfaceRZFourier.from_vmec_input(filename, range="half period", nphi=4, ntheta=4)
    base_curves = create_equally_spaced_curves(1, 2, True, R0=1.0, R1=0.5, order=3)
    base_currents = [Current(1e5)]
    coils_tf = coils_via_symmetries(base_curves, base_currents, 2, True)
    eval_pts = np.ascontiguousarray(s.gamma().reshape(-1, 3))
    psc = PSCBulkArray(
        np.array([[1.0, 0.0, 0.15]]),
        np.array([[0.1, 0.1, 1.0]]),
        np.array([0.04]),
        np.array([0.02]),
        coils_tf,
        eval_points=eval_pts,
        m_fourier=1, l_zernike=2, k_chebyshev=1,
        n_rho=4, n_phi=6, n_z=3,
    )
    b_tf = BiotSavart(coils_tf)
    btot = MagneticFieldSum([psc.biot_savart, b_tf])
    Jf = SquaredFlux(s, btot)
    return psc, btot, Jf, coils_tf


def _taylor_sweep(Jf: SquaredFlux, psc: PSCBulkArray, btot: MagneticFieldSum,
                  start_power: int = 3, n: int = 10):
    dofs0 = np.copy(Jf.x)
    rng = np.random.default_rng(42)
    h = rng.standard_normal(dofs0.size)
    h /= np.linalg.norm(h)

    def _set(x):
        Jf.x = x
        psc.recompute_currents()
        btot.Bfields[0].clear_cached_properties()

    _set(dofs0)
    dJ0 = np.array(Jf.dJ())
    deriv = float(np.sum(dJ0 * h))
    eps_list = np.array([0.5 ** i for i in range(start_power, start_power + n)])
    errs = []
    for eps in eps_list:
        _set(dofs0 + eps * h)
        Jp = float(Jf.J())
        _set(dofs0 - eps * h)
        Jm = float(Jf.J())
        fd = 0.5 * (Jp - Jm) / eps
        rel = abs(fd - deriv) / max(abs(deriv), 1e-30)
        errs.append(rel)
    _set(dofs0)
    return eps_list, np.array(errs), deriv


def main() -> None:
    """Generate and save the Taylor-test figure."""
    setup_mpl()
    fig, ax = plt.subplots(figsize=(6.4, 4.2))

    # TF-only: puck DOFs fixed by default; ensure TF curves remain free.
    psc, btot, Jf, coils_tf = _build()
    eps_tf, err_tf, d_tf = _taylor_sweep(Jf, psc, btot, start_power=1, n=12)
    ax.loglog(eps_tf, err_tf, "-o", label="TF coil DOFs only")
    print(f"[sweep] TF-only   directional={d_tf:.3e}  min_err={err_tf.min():.3e}")

    # Geometry: also unfix puck centers.
    psc2, btot2, Jf2, coils_tf2 = _build()
    psc2.unfix("center_x0")
    psc2.unfix("center_y0")
    psc2.unfix("center_z0")
    eps_geom, err_geom, d_geom = _taylor_sweep(Jf2, psc2, btot2, start_power=1, n=12)
    ax.loglog(eps_geom, err_geom, "-s", label="TF + puck centers")
    print(f"[sweep] TF+geom   directional={d_geom:.3e}  min_err={err_geom.min():.3e}")

    ref = (eps_tf / eps_tf[0]) ** 2 * err_tf[0]
    ax.loglog(eps_tf, ref, "--", color="gray", alpha=0.6, label=r"$\propto \epsilon^2$")
    ax.set_xlabel(r"$\epsilon$")
    ax.set_ylabel("relative error of central-FD directional derivative")
    ax.set_title("Taylor test for adjoint gradients")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, which="both", alpha=0.35)
    fig.tight_layout()

    print_audit(
        "taylor_test",
        [
            ("min_err_tf", float(err_tf.min())),
            ("min_err_geom", float(err_geom.min())),
        ],
    )
    save_pdf(fig, "taylor_test")


if __name__ == "__main__":
    main()
