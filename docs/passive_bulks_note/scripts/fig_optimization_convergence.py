r"""Figure: optimization convergence of SquaredFlux with passive-bulk field.

This reproduces the convergence trace of ``test_optimization_objective_decreases``
(:mod:`tests.field.test_passive_bulks`): a small QA surface, a pair of TF
coils, and a single passive puck, with the objective
:math:`J = \tfrac12 \iint (\mathbf B_{\text{total}}\cdot\hat{\mathbf n})^2\,dS`.
Only the TF coil DOFs are optimized; the puck currents are induced at
every iteration through the adjoint VJP in
:meth:`~simsopt.field.psc_bulk.PSCBulkArray.vjp_setup_B`.

We record :math:`J` at every function evaluation inside ``scipy`` and
plot :math:`J / J_0` vs. iteration.  The curve dropping monotonically
below ``1`` confirms that gradients flow correctly through both the
Biot–Savart kernel of the induced sheet currents and the implicit
linear solve :math:`L\boldsymbol\beta = \mathbf f`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import print_audit, save_pdf, setup_mpl  # noqa: E402

from simsopt.field import BiotSavart, Current, coils_via_symmetries  # noqa: E402
from simsopt.field.magneticfield import MagneticFieldSum  # noqa: E402
from simsopt.field.psc_bulk import PSCBulkArray  # noqa: E402
from simsopt.geo import SurfaceRZFourier, create_equally_spaced_curves  # noqa: E402
from simsopt.objectives import SquaredFlux  # noqa: E402


def main() -> None:
    """Generate and save the optimization convergence figure."""
    setup_mpl()
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
    btot = MagneticFieldSum([psc.biot_savart, BiotSavart(coils_tf)])
    Jf = SquaredFlux(s, btot)

    dofs0 = np.copy(Jf.x)
    trajectory: list[float] = []

    def fun(x: np.ndarray) -> tuple[float, np.ndarray]:
        Jf.x = x
        psc.recompute_currents()
        btot.Bfields[0].clear_cached_properties()
        J = float(Jf.J())
        dJ = np.array(Jf.dJ())
        trajectory.append(J)
        return J, dJ

    J0 = fun(dofs0)[0]
    print(f"[init] J0 = {J0:.6e}")
    res = minimize(fun, dofs0, jac=True, method="L-BFGS-B",
                   options={"maxiter": 25, "disp": False, "gtol": 1e-12, "ftol": 1e-14})
    trajectory.append(float(res.fun))
    J_final = float(res.fun)
    print(f"[final] J = {J_final:.6e}  reduction = {J_final / J0:.3e}")

    it = np.arange(len(trajectory))
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.semilogy(it, np.array(trajectory) / J0, "-o", color="tab:blue")
    ax.axhline(1.0, color="k", ls="-", lw=0.6, alpha=0.4)
    ax.set_xlabel("function evaluation")
    ax.set_ylabel(r"$J(\mathbf{x})\,/\,J(\mathbf{x}_0)$")
    ax.set_title("Coupled TF + passive-bulk optimization (L-BFGS-B)")
    ax.grid(True, which="both", alpha=0.35)
    fig.tight_layout()

    print_audit("optimization_convergence",
                [("J0", J0), ("J_final", J_final), ("reduction", J_final / J0)])
    save_pdf(fig, "optimization_convergence")


if __name__ == "__main__":
    main()
