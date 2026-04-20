r"""Figure: short-solenoid self-inductance vs Lorenz-formula reference.

For the side-wall basis function with azimuthal mode ``m = 0`` and
Chebyshev index ``k = 1`` (``g = T_1(2z/t) = 2z/t``), the induced sheet
current is uniform azimuthal:

.. math::

    \mathbf K = \hat\rho\times\nabla_s g = -\tfrac{2}{t}\,\hat\phi .

The self-inductance block entry ``L_{k=1,k=1}`` therefore equals the
Lorenz-type self-inductance of a thin-walled, uniformly-current cylinder
of radius :math:`R` and length :math:`t` carrying the normalized current
pattern above.  We cross-check the implementation's value
(:func:`~simsopt.field.disc_self_inductance.cylinder_self_block_mode_m`)
against an independent reference computed by stacking coaxial rings and
summing :func:`ring_mutual_axial_shift` (the Maxwell-Grover closed form)
on a finely-refined trapezoidal grid, which is the standard numerical
implementation of the Lorenz (short-solenoid) mutual-loop formula
(Grover, 1946).  The agreement curve isolates the accuracy of the side-
wall quadrature from everything else in the puck solve.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import print_audit, save_pdf, setup_mpl  # noqa: E402

from simsopt.field.disc_self_inductance import (  # noqa: E402
    _chebyshev_t_derivative_numpy,
    _chebyshev_t_numpy,
    cylinder_self_block_mode_m,
    ring_mutual_axial_shift,
)


def _make_axial(t: float, k: int):
    def f(z: np.ndarray) -> np.ndarray:
        zeta = 2.0 * np.asarray(z, dtype=float) / t
        return _chebyshev_t_numpy(zeta, k)

    def fp(z: np.ndarray) -> np.ndarray:
        zeta = 2.0 * np.asarray(z, dtype=float) / t
        return (2.0 / t) * _chebyshev_t_derivative_numpy(zeta, k)

    return f, fp


def lorenz_reference(R: float, t: float, n_trap: int = 2048) -> float:
    r"""Reference value for ``L_{k=1,k=1}^{\text{side}, m=0}`` from stacked rings.

    Uses the identity that for the ``g = T_1(2z/t)`` basis, the sheet
    current is uniform :math:`K_\phi = -2/t` and the inductance block
    reduces to

    .. math::

        L = \frac{4}{t^2}\int_{-t/2}^{t/2}\!\!\int_{-t/2}^{t/2}
        M(R, R, z - z')\,dz\,dz'

    with :math:`M` = :func:`ring_mutual_axial_shift`.  A very fine
    trapezoidal rule gives the reference value (the integrand is smooth
    except at :math:`z = z'`, which is a weak log singularity of
    measure zero under the 2D integral).
    """
    z = np.linspace(-0.5 * t, 0.5 * t, n_trap)
    dz = z[1] - z[0]
    # Build M(R, R, |z - z'|) on the grid.
    DZ = np.abs(z[:, None] - z[None, :])
    M = np.empty_like(DZ)
    for i in range(n_trap):
        for j in range(n_trap):
            if DZ[i, j] < 1e-12:
                # Skip the exact-diagonal sample (log-singular measure zero).
                M[i, j] = 0.0
            else:
                M[i, j] = ring_mutual_axial_shift(R, R, DZ[i, j])
    # Trapezoidal weights (dz on interior, dz/2 on endpoints).
    w = np.full(n_trap, dz)
    w[0] = w[-1] = 0.5 * dz
    integral = float(np.sum(M * (w[:, None] * w[None, :])))
    return (4.0 / t ** 2) * integral


def main() -> None:
    """Generate and save the Lorenz short-solenoid figure."""
    setup_mpl()
    R = 0.30
    t_over_R = np.array([3.0, 2.0, 1.5, 1.0, 0.75, 0.5, 0.3, 0.2])
    L_impl = []
    L_ref = []
    for tR in t_over_R:
        t = R * float(tR)
        f, fp = _make_axial(t, 1)
        val = cylinder_self_block_mode_m(f, fp, f, fp, 0, R, t, n_z=48)
        ref = lorenz_reference(R, t, n_trap=512)
        L_impl.append(val)
        L_ref.append(ref)
        print(f"[sweep] t/R={tR:5.2f}  impl={val:.6e}  ref={ref:.6e}  rel_err={abs(val-ref)/abs(ref):.2e}")

    L_impl = np.array(L_impl)
    L_ref = np.array(L_ref)
    rel_err = np.abs(L_impl - L_ref) / np.abs(L_ref)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))

    axes[0].loglog(t_over_R, L_impl, "-o", label="implementation")
    axes[0].loglog(t_over_R, L_ref, "--s", label="stacked-ring reference (Lorenz)")
    axes[0].set_xlabel(r"$t / R$")
    axes[0].set_ylabel(r"$L_{kk}^{\text{side},\, m=0,\,k=1}$ (H)")
    axes[0].set_title(r"Short-solenoid self-inductance")
    axes[0].legend(loc="best", fontsize=8)

    axes[1].loglog(t_over_R, rel_err, "-d", color="tab:red")
    axes[1].set_xlabel(r"$t / R$")
    axes[1].set_ylabel(r"relative error")
    axes[1].set_title(r"$|L_{\text{impl}} - L_{\text{ref}}| / |L_{\text{ref}}|$")
    axes[1].grid(True, which="both", alpha=0.35)

    fig.tight_layout()

    print_audit(
        "lorenz_short_solenoid",
        [
            ("R", R),
            ("rel_err_tR_1.0", float(rel_err[np.argmin(np.abs(t_over_R - 1.0))])),
            ("rel_err_tR_0.2", float(rel_err[-1])),
        ],
    )
    save_pdf(fig, "lorenz_short_solenoid")


if __name__ == "__main__":
    main()
