"""Plot the optimized W7-X SurfaceBSpline (w7x_spline_dofs.py) with PyVista.

Same style and camera as DESC's
examples/stage_one_splines_output/plot_stage_one_splines_pyvista.py, rendered
with VTK, so the control net is properly occluded by the surface (depth peeling
handles the translucent surface). Writes a cropped PNG and a PDF holding that
image.

With --show, opens an interactive window instead: drag to rotate, scroll to zoom,
and press "c" to print the ELEV and AZIM_OFFSET of the current view, to paste
in below for the saved figure.
"""

import os
import sys

import numpy as np
import pyvista as pv
from PIL import Image
from simsopt.geo import SurfaceBSpline

PINK = "#ffa5c6"
SEAFOAM = "#5fcfa8"
SEAFOAM_DARK = "#3a9e7c"
FADED = 0.1  # opacity of the control net outside one half field period
SPOKES = 0.45  # opacity of the dotted lines from the axis to the control points
HIDDEN = 0.35  # extra opacity of the half-period net where the surface hides it
ELEV = 30  # camera elevation (degrees)
AZIM_OFFSET = -15  # camera rotation (degrees) from the middle of the half period;
# negative turns the surface counterclockwise seen from above
POINT_RADIUS = 0.012  # control point radius (m)
SHOW = "--show" in sys.argv  # interactive window instead of saving the figure
SCALE = 1 if SHOW else 3  # resolution multiplier; line widths are scaled to match
WINDOW = (1500, 900) if SHOW else (3000 * SCALE, 1800 * SCALE)  # in pixels
DPI = 300 * SCALE  # for the page size, so the physical size doesn't change with SCALE
JPEG_QUALITY = 92  # the PDF holds a JPEG, much smaller than the PNG

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
from w7x_spline_dofs import dofs, spline_kwargs  # noqa: E402

n_theta, n_zeta = 150, 500
surf = SurfaceBSpline(
    quadpoints_phi=np.linspace(0, 1, n_zeta, endpoint=False),
    quadpoints_theta=np.linspace(0, 1, n_theta, endpoint=False),
    **spline_kwargs,
)
surf.x = dofs

# close the seams in both directions
xyz = surf.gamma()
xyz = np.concatenate([xyz, xyz[:, :1]], axis=1)
xyz = np.concatenate([xyz, xyz[:1]], axis=0)

phi = np.linspace(0, 2 * np.pi, 400)
r_axis, z_axis = surf._axis_rz(phi)
axis_xyz = np.stack([r_axis * np.cos(phi), r_axis * np.sin(phi), z_axis], axis=1)

# (n cross sections, points per cs, 3); the first n_cs rows span one half
# field period (drawn opaque), the rest are faded
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

pl = pv.Plotter(off_screen=not SHOW, window_size=WINDOW)
pl.set_background("white")
pl.enable_depth_peeling(number_of_peels=20)
pl.add_mesh(
    pv.StructuredGrid(*xyz.transpose(2, 0, 1)),
    color=PINK,
    opacity=0.65,
    smooth_shading=False,
    specular=0,  # matte
)
pl.add_mesh(pv.lines_from_points(axis_xyz), color="black", line_width=4 * SCALE)
for alpha in (1.0, FADED):
    for line in lines[alpha]:
        pl.add_mesh(pv.lines_from_points(line), color=SEAFOAM, line_width=3 * SCALE,
                    opacity=alpha)
    # sphere meshes rather than render_points_as_spheres, whose shader is glossy
    spheres = pv.PolyData(points[alpha]).glyph(
        geom=pv.Sphere(radius=POINT_RADIUS, theta_resolution=24, phi_resolution=24),
        scale=False, orient=False,
    )
    pl.add_mesh(spheres, color=SEAFOAM_DARK, opacity=alpha, smooth_shading=False,
                specular=0)
    if alpha == 1.0:
        net_meshes = [pv.lines_from_points(line) for line in lines[alpha]]
        net_spheres = spheres

# dotted lines from each cross section's centre on the interior axis to its control
# points, over the opaque half field period; VTK has no line stipple, so the dots
# are short segments
cs_zeta, _ = surf.get_cs_zeta_angle()
centres = np.asarray(surf._axis_local_basis(cs_zeta)[0])
dashes = []
for centre, ring in zip(centres, net[:half, :-1]):
    for point in ring:
        t = np.linspace(0, 1, 2 * max(int(np.linalg.norm(point - centre) / 0.02), 1))
        ends = centre + t[:, None] * (point - centre)
        dashes += [pv.Line(a, b) for a, b in zip(ends[::2], ends[1::2])]
spokes = pv.merge(dashes)
spokes.clear_data()  # pv.Line attaches a "Distance" array the mapper would color by

# the opaque net again, translucent, on an overlay layer drawn without depth testing
# against the surface, so the parts it hides show through more
overlay = pv.Renderer(pl)
overlay.SetLayer(1)
overlay.SetActiveCamera(pl.renderer.GetActiveCamera())
pl.ren_win.SetNumberOfLayers(2)
pl.ren_win.AddRenderer(overlay)
# (the spokes are only drawn here, since they run mostly inside the surface)
layers = [(m, SEAFOAM, HIDDEN, 3) for m in net_meshes]
layers += [(net_spheres, SEAFOAM_DARK, HIDDEN, 3), (spokes, SEAFOAM_DARK, SPOKES, 2)]
for mesh, color, opacity, width in layers:
    actor = pv.Actor(mapper=pv.DataSetMapper(mesh))
    actor.prop.color = color
    actor.prop.opacity = opacity
    actor.prop.line_width = width * SCALE
    actor.prop.interpolation = "flat"
    overlay.AddActor(actor)

# camera at the toroidal angle near the middle of the first half field period,
# rotated by AZIM_OFFSET, looking at the centre of the data
pts = np.concatenate([xyz.reshape(-1, 3), net.reshape(-1, 3)])
center = (pts.max(axis=0) + pts.min(axis=0)) / 2
azim = np.pi / (2 * surf.nfp) + np.radians(AZIM_OFFSET)
elev = np.radians(ELEV)
direction = np.array(
    [np.cos(elev) * np.cos(azim), np.cos(elev) * np.sin(azim), np.sin(elev)]
)
pl.camera.focal_point = center
pl.camera.position = center + 10 * np.ptp(pts, axis=0).max() * direction
pl.camera.up = (0, 0, 1)
pl.enable_parallel_projection()
pl.reset_camera()

if SHOW:

    def print_view():
        d = np.asarray(pl.camera.position) - np.asarray(pl.camera.focal_point)
        elev = np.degrees(np.arcsin(d[2] / np.linalg.norm(d)))
        azim = np.degrees(np.arctan2(d[1], d[0])) - 90 / surf.nfp
        azim = (azim + 180) % 360 - 180
        print(f"ELEV = {elev:.0f}\nAZIM_OFFSET = {azim:.0f}")

    pl.add_key_event("c", print_view)
    pl.show()
    sys.exit()

# crop to the pixels actually drawn on, plus a small pad
img = pl.screenshot(return_img=True)
drawn = np.argwhere(img[..., :3].min(axis=2) < 250)
(row0, col0), (row1, col1) = drawn.min(axis=0), drawn.max(axis=0)
pad = int(0.05 * DPI)
img = img[max(row0 - pad, 0):row1 + pad, max(col0 - pad, 0):col1 + pad]

out = os.path.join(here, "w7x_spline_pyvista")
Image.fromarray(img).save(out + ".png", dpi=(DPI, DPI), optimize=True)
Image.fromarray(img).convert("RGB").save(
    out + ".pdf", resolution=DPI, quality=JPEG_QUALITY
)
print("Saved", out + ".png", "and", out + ".pdf")
