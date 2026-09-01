#!/usr/bin/env python
"""
Standalone demo of tangent_extension.py's corner sharpening, with
s1/s2 FIXED constants -- no optimization, no VMEC/QS. This just shows
the deterministic wedge construction itself, before/after, on a real
spline surface: tangent_extension(spline_surf, s1, s2,
corner_criterion) computes where each corner's tangent lines would
meet if extended by a fixed arc length s1/s2 back along the curve, and
moves exactly the control points between them onto that wedge.

s1 and s2 are fixed here because they're tangent_extension's actual
degrees of freedom (not the individual control points it moves --
those are fully determined once s1/s2 are chosen) -- see
tangent_extension.py's module docstring. Treating s1/s2 as
optimization variables (e.g. against quasisymmetry) is a natural next
step, but is NOT what this script does; it only demonstrates the
geometric construction for one fixed, hand-picked choice of s1/s2.
"""

import matplotlib.pyplot as plt
import numpy as np
from sharpen import plot_before_after
from simsopt import save
from simsopt.geo import SurfaceBSpline
from simsopt.mhd import Vmec
from simsopt.util import MpiPartition, proc0_print
from tangent_extension import (
    _cross_section_local_curve,
    plot_tangent_extension,
    tangent_extension,
    tangent_extension_geometry,
)

mpi = MpiPartition()
mpi.write()

if __name__ == "__main__":
    dofs = [
        0.18588267,
        0.15642203,
        0.43764541,
        0.16784444,
        1.26476013,
        2.01586034,
        0.12997366,
        0.21563905,
        0.39986906,
        0.16935926,
        0.50061529,
        0.15108062,
        0.64882775,
        1.88036231,
        2.61799388,
        4.26470135,
        4.71238898,
        0.09381768,
        0.2645054,
        0.44291523,
        0.20125536,
        0.5234703,
        0.15781845,
        0.71877959,
        1.66919584,
        3.66519143,
        4.37137573,
        4.71238898,
        0.07724166,
        0.33769754,
        0.4633352,
        0.21862174,
        0.58350463,
        0.11953112,
        0.98968873,
        1.57079633,
        3.66519143,
        4.52991733,
        4.71238898,
        0.03416579,
        0.39328045,
        0.49537881,
        0.22380475,
        0.59100414,
        0.26095387,
        1.16794389,
        1.57079633,
        3.66519143,
        4.6681426,
        4.71238898,
        0.02337083,
        0.41469386,
        0.52910186,
        0.18522041,
        1.36128815,
        1.57079633,
        0.92175992,
        1.1061464,
        1.21557964,
        0.07962817,
    ]

    spline_kwargs = {
        "axis_points": 3,
        "points_per_cs": 6,
        "n_cs": 6,
        "nfp": 2,
        "M": 12,
        "N": 12,
        "p_u": 3,
        "p_v": 3,
        "cs_equispaced": True,
        "rays_equispaced": False,
        "cs_global_angle_free": False,
        "axis_angles_fixed": True,
        "cs_basis": "polar",
        "nurbs": False,
        "use_bishop_frame": True,
        "knot_parametrization": "uniform",
    }

    spline_surf = SurfaceBSpline(**spline_kwargs)
    spline_surf.x = dofs

    # tangent_extension assumes the control net already densely tracks
    # its own curve (see its module docstring) -- Lane-Riesenfeld
    # doubling gets us there.
    spline_surf.refine_poloidal()
    spline_surf.refine_poloidal()
    spline_surf.refine_poloidal()

    spline_surf.p_u = 3
    # A separate, un-reshaped copy (same dofs/refine_poloidal calls,
    # never touched afterward) for plot_before_after to compare
    # against -- NOT list(spline_surf.cs_list), which would just be
    # references to the same CrossSectionFixedZeta objects
    # tangent_extension is about to mutate in place, making "before"
    # silently show the already-sharpened shape too. Also gives
    # plot_before_after's free/fixed control-point comparison a
    # meaningful "before" (all gray, via fix_all() below) to contrast
    # with "after" (the wedge region green).
    spline_surf_before = SurfaceBSpline(**spline_kwargs)
    spline_surf_before.x = dofs
    spline_surf_before.refine_poloidal()
    spline_surf_before.refine_poloidal()
    spline_surf_before.refine_poloidal()
    spline_surf_before.fix_all()

    # Fixed (not solver variables) arc-length reach for every corner's
    # tangent extension, as a fraction of the SMALLEST cross section's
    # own perimeter -- so the same fraction means a comparable "how
    # far back along the curve" everywhere, and no cross section's
    # window can accidentally exceed its own size.
    S1_FRACTION = 0.06
    S2_FRACTION = 0.06
    CORNER_CRITERION = "curvature"

    perimeters = [
        _cross_section_local_curve(spline_surf, cs, n_samples=2000)["L"]
        for cs in spline_surf.cs_list
    ]
    S1 = S1_FRACTION * min(perimeters)
    S2 = S2_FRACTION * min(perimeters)
    proc0_print(
        f"Fixed s1={S1:.4f}, s2={S2:.4f} "
        f"(smallest cross-section perimeter is {min(perimeters):.4f})"
    )

    # Preview: the CURRENT curve/control net, with the tangent lines
    # and new apex tangent_extension is about to build, before
    # actually moving anything.
    plot_tangent_extension(spline_surf, S1, S2, CORNER_CRITERION)
    plt.show()

    initial_geometry = tangent_extension_geometry(
        spline_surf, S1, S2, CORNER_CRITERION
    )
    initial_angles = np.array(
        [
            g["opening_angle"]
            for cs_corners in initial_geometry
            for g in cs_corners
        ]
    )
    proc0_print(
        "Opening angle(s) per corner, from the CURRENT (smooth) curve's "
        "own tangent lines [deg]:",
        np.degrees(initial_angles),
    )

    # tangent_extension only ever unfixes the control points it
    # touches -- fix_all() first makes those (and nothing else) free
    # afterward.
    spline_surf.fix_all()
    tangent_extension(spline_surf, S1, S2, CORNER_CRITERION)
    spline_surf.unfix_all()
    print(repr(spline_surf.x))
    # save(spline_surf, filename="sharpened.json")
    # proc0_print(f"After tangent_extension: {len(spline_surf.x)} dofs free")

    # # Re-running the same corner-finding/tangent-extension geometry on
    # # the NOW-reshaped curve: not exactly identical to initial_angles
    # # above (the B-spline curve through the moved control points only
    # # approximates the exact straight-line wedge, and corner-finding
    # # runs fresh on the new curve), but should be very close if the
    # # control net is fine enough -- a useful self-consistency check.
    # final_geometry = tangent_extension_geometry(
    #     spline_surf, S1, S2, CORNER_CRITERION
    # )
    # final_angles = np.array(
    #     [
    #         g["opening_angle"]
    #         for cs_corners in final_geometry
    #         for g in cs_corners
    #     ]
    # )

    # # Sanity check before handing the reshaped surface to VMEC:
    # # VMEC only ever sees this spline through the RZFourier
    # # coefficients to_RZFourier() would build for it, so this is the
    # # same ft() call to_RZFourier() makes internally (default nu/nv,
    # # collocation="exact"), with plot_ft=True to overlay the resulting
    # # "FT" reconstruction against the spline's own "Spline (ground
    # # truth)" cross sections (plot_ft_vs_ground_truth) -- if the wedge
    # # is too sharp for M/N to represent, it'll show up here as a
    # # visibly rounded/ringing "FT" curve, before VMEC ever runs on it.
    # spline_surf.ft(plot_ft=True, spec_cond=True)
    # plt.show()

    # vmec = Vmec.vmec_from_surf(
    #     nfp=spline_surf.nfp,
    #     surf=spline_surf,
    #     mpi=mpi,
    #     ns=50,
    #     M=12,
    #     N=12,
    #     ftol=1e-10,
    #     niter=8000,
    #     verbose=True,
    # )
    # # vmec.indata.write_indata_namelist('input.sharpened')

    # vmec.run()

    # proc0_print(
    #     "Opening angle(s) per corner after tangent_extension [deg]:",
    #     np.degrees(final_angles),
    # )

    # plot_before_after(spline_surf, spline_surf_before)

    # proc0_print("")
    # proc0_print(
    #     "End of figures_spline_paper/corner_sharpening/sharpen_fixed_s.py"
    # )
