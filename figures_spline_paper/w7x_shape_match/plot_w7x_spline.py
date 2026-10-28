"""
Plot the optimized W7-X SurfaceBSpline (from spline_fit_w7x_unconstrained.py)
to a pdf, in the pink/seafoam style of SurfaceBSpline.plot on ia/SplineSurface:
pink semi-transparent surface, black pseudo-axis, seafoam control net that is opaque
over one half field period and faded elsewhere.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import Bbox
from simsopt.geo import Surface, SurfaceBSpline
from w7x_spline_dofs import dofs, spline_kwargs

PINK = "#ffa5c6"  # #ff8fb8 (ia/SplineSurface) lightened 20% toward white
SEAFOAM = "#5fcfa8"
SEAFOAM_DARK = "#3a9e7c"
FADED = 0.1  # alpha of the control net outside one half field period

# Finer quadpoints than the default 61x62 over the full torus, which looks
# faceted; only used for drawing the surface here.
surf = SurfaceBSpline(
    quadpoints_phi=np.linspace(0, 1, 500, endpoint=False),
    quadpoints_theta=np.linspace(0, 1, 150, endpoint=False),
    **spline_kwargs,
)
surf.x = dofs

fig = plt.figure(figsize=(6, 3.4))
ax = fig.add_subplot(projection="3d")
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
# Camera at the toroidal angle midway through the first half field period,
# so the opaque part of the control net faces the viewer.
ax.view_init(elev=35, azim=np.degrees(np.pi / (2 * surf.nfp)))

# Surface.plot directly, bypassing SurfaceBSpline's own (debug-style) plot
Surface.plot(
    surf, engine="matplotlib", ax=ax, show=False, close=True, axis_equal=False,
    color=PINK, alpha=0.65, rcount=500, ccount=150, rasterized=True,
)

phi = np.linspace(0, 2 * np.pi, 400)
r_axis, z_axis = surf._axis_rz(phi)
ax.plot(r_axis * np.cos(phi), r_axis * np.sin(phi), z_axis, color="k", lw=1)

# (n_cross_sections, points_per_cs, 3); the first n_cs rows span one half
# field period (drawn opaque), the rest are faded.
net = np.array(surf._get_control_points_xyz())
half = surf.n_cs
points = {1.0: net[:half].reshape(-1, 3), FADED: net[half:].reshape(-1, 3)}
net = np.concatenate([net, net[:, :1]], axis=1)
net = np.concatenate([net, net[:1]], axis=0)
columns = net.transpose(1, 0, 2)
lines = {
    1.0: list(net[:half]) + [col[:half] for col in columns],
    FADED: list(net[half:-1]) + [col[half - 1:] for col in columns],
}
for alpha in (1.0, FADED):
    ax.scatter(*points[alpha].T, color=SEAFOAM_DARK, s=0.8, depthshade=False,
               alpha=alpha)
    for line in lines[alpha]:
        ax.plot(*line.T, color=SEAFOAM, lw=0.4, alpha=alpha)

# Equal scale on all axes, with the box shrunk to the data extents (rather
# than a cube) and zoomed in so the torus fills the figure.
ranges = [np.ptp(lim) for lim in (ax.get_xlim(), ax.get_ylim(), ax.get_zlim())]
ax.set_box_aspect(ranges, zoom=1.5)
ax.set_axis_off()

# bbox_inches="tight" keeps the whole (zoomed, oversized) 3D axes box, so
# crop instead to the pixels actually drawn on, plus a small pad.
fig.canvas.draw()
drawn = np.argwhere(np.asarray(fig.canvas.buffer_rgba())[..., :3].min(axis=2) < 250)
(row0, col0), (row1, col1) = drawn.min(axis=0), drawn.max(axis=0)
h = fig.canvas.get_width_height()[1]
pad = 0.05 * fig.dpi
crop = Bbox([[col0 - pad, h - row1 - pad], [col1 + pad, h - row0 + pad]])
fig.savefig("w7x_spline.pdf", dpi=300, bbox_inches=crop.transformed(fig.dpi_scale_trans.inverted()))
print("Saved w7x_spline.pdf")
