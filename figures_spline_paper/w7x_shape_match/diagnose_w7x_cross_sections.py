"""
Diagnostic: compare the rescaled W7-X target boundary, the optimized W7-X
SurfaceBSpline's own cross sections, and the spline's Fourier representation
(to_RZFourier), both without spectral condensation (spec_cond=None, what the
shape-fit objective sees) and with the default variational condensation, at
a few toroidal cuts across one field period. Interactive matplotlib window.
"""

import matplotlib

matplotlib.use("qtagg")
import matplotlib.pyplot as plt
import numpy as np
from simsopt.geo import SurfaceBSpline
from simsopt.mhd import Vmec
from w7x_spline_dofs import dofs, spline_kwargs

TARGET_FILE = (
    "/Users/issraali/codes/simsopt/tests/test_files/"
    "input.W7-X_without_coil_ripple_beta0p05_d23p4_tm"
)
N_CUTS = 6
NTHETA = 300

# Target rescaled to R0 = 1, as in spline_fit_w7x_unconstrained.py
target_surf = Vmec(TARGET_FILE, verbose=False).boundary
scale = 1.0 / abs(target_surf.get_rc(0, 0))
target_surf.rc[:, :] *= scale
target_surf.zs[:, :] *= scale

surf = SurfaceBSpline(**spline_kwargs)
surf.x = dofs

fourier_surfs = {
    "Fourier, spec_cond=None": surf.to_RZFourier(spec_cond=None),
    "Fourier, variational": surf.to_RZFourier(),
}
styles = {
    "Fourier, spec_cond=None": dict(color="tab:blue", ls="--"),
    "Fourier, variational": dict(color="tab:orange", ls=":"),
}


def closed_rz(xyz):
    r = np.hypot(xyz[:, 0], xyz[:, 1])
    return np.append(r, r[0]), np.append(xyz[:, 2], xyz[0, 2])


phis = np.linspace(0, 1.0 / surf.nfp, N_CUTS, endpoint=False)
fig, axes = plt.subplots(2, N_CUTS // 2, figsize=(12, 7), sharex=True, sharey=True)
for ax, phi in zip(axes.flat, phis):
    ax.plot(*closed_rz(target_surf.cross_section(phi, thetas=NTHETA)),
            color="0.6", lw=4, alpha=0.6, label="target (W7-X)")
    ax.plot(*closed_rz(surf.cross_section(phi, thetas=NTHETA)), "k-", lw=2,
            label="spline")
    for label, rz_surf in fourier_surfs.items():
        ax.plot(*closed_rz(rz_surf.cross_section(phi, thetas=NTHETA)), lw=1.5,
                label=label, **styles[label])
    ax.set_title(rf"$\phi$ = {np.degrees(phi * 2 * np.pi):.1f}$^\circ$")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
for ax in axes[-1]:
    ax.set_xlabel("R")
for ax in axes[:, 0]:
    ax.set_ylabel("Z")
axes.flat[0].legend(fontsize=8)
fig.suptitle("W7-X target vs. spline vs. Fourier cross sections")
fig.tight_layout()
plt.show()
