#!/usr/bin/env python

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from simsopt.geo.surfacespline import SurfaceBSpline
from simsopt.mhd import Vmec
from simsopt.util.mpi import MpiPartition
from simsopt.util.spline_helpers import print_dofs_nicely

matplotlib.use("qtagg")
mpi = MpiPartition()
mpi.write()


def boundary_poincare_plot(
    ax,
    rbc,
    zbs,
    phi,
    N,
    M,
    nfp,
    plotting_kwargs,
    scatter=False,
    ntheta=200,
):
    xn = np.arange(-N, N + 1, 1)
    xm = np.arange(0, M + 1, 1)
    theta = np.linspace(0, 2 * np.pi, num=ntheta)

    R = np.zeros((ntheta, 1))
    Z = np.zeros((ntheta, 1))

    for i in range(rbc.shape[0]):
        for j in range(rbc.shape[1]):
            if rbc[i, j] != 0 or zbs[i, j] != 0:
                angle = xm[j] * theta - xn[i] * phi * nfp
                R = R + rbc[i, j] * np.cos(angle)  # /(np.abs(i) + np.abs(j))
                Z = Z + zbs[i, j] * np.sin(angle)  # /(np.abs(i) + np.abs(j))
    if not scatter:
        ax.plot(R.flatten(), Z.flatten(), **plotting_kwargs)
    else:
        ax.scatter(R.flatten(), Z.flatten(), **plotting_kwargs)


def write_doflist_maxlist_minlist(spline_kwargs):

    template_surf = SurfaceBSpline(**spline_kwargs)
    template_surf.axis.fix("r_axis_0")

    doflist = template_surf.dof_names

    lb = np.copy(template_surf.lower_bounds)
    ub = np.copy(template_surf.upper_bounds)

    return doflist, ub, lb


if __name__ == "__main__":
    # target_surf.plot()

    # spline_kwargs = {
    #     "axis_points": 3,
    #     "points_per_cs": 6,
    #     "n_cs": 4,
    #     "nfp": 2,
    #     "M": 12,
    #     "N": 12,
    #     "p_u": 3,
    #     "p_v": 3,
    #     "cs_equispaced": True,
    #     "rays_equispaced": False,
    #     "cs_global_angle_free": False,
    #     "axis_angles_fixed": True,
    #     "cs_basis": "polar",
    #     "nurbs": False,
    #     "knot_parametrization": "chord",
    # }

    spline_kwargs = {
        "axis_points": 3,
        "points_per_cs": 4,
        "n_cs": 5,
        "nfp": 2,
        "M": 8,
        "N": 4,
        "p_u": 3,
        "p_v": 3,
        "cs_equispaced": False,
        "rays_equispaced": False,
        "cs_global_angle_free": False,
        "axis_angles_fixed": False,
        "cs_basis": "polar",
        "nurbs": False,
        "use_bishop_frame": True,
        "knot_parametrization": "uniform",
    }

    dof_list, ub, lb = write_doflist_maxlist_minlist(spline_kwargs)

    spline_surf = SurfaceBSpline(
        default_r=0.3,
        **spline_kwargs,
    )
    spline_surf.axis.fix("r_axis_0")
    # spline_surf.axis.fix("r_axis_0")
    # new_x = np.array(
    #     [
    #         0.3016216,
    #         0.20890161,
    #         0.44766483,
    #         0.49065546,
    #         0.31258963,
    #         2.72104182,
    #         0.13272592,
    #         0.01054766,
    #         0.55550307,
    #         0.34290225,
    #         0.24441989,
    #         0.54088171,
    #         1.42225604,
    #         2.49345601,
    #         3.26726428,
    #         3.90173828,
    #         5.75958653,
    #         0.03943073,
    #         0.17513655,
    #         0.61373303,
    #         0.17959677,
    #         0.26507675,
    #         0.72124269,
    #         1.57072331,
    #         2.02542396,
    #         3.16716533,
    #         4.71228105,
    #         5.49176038,
    #         0.04404639,
    #         0.59041262,
    #         0.48036493,
    #         0.16049688,
    #         1.39121464,
    #         1.62414016,
    #         1.68508211,
    #         2.00728408,
    #         -0.28335058,
    #     ]
    # )
    new_x = np.array(
        [
            7.5201273052623300e-06,
            4.2501631406549062e-01,
            1.3789545468136427e-01,
            1.3105015477699220e00,
            2.1246033886416580e-02,
            3.6328529493187850e-01,
            1.2818811782170014e-01,
            3.8943918345599199e-01,
            1.0694883206553649e00,
            3.2405748878384344e00,
            4.6539785945862961e00,
            1.1001511421235184e-01,
            2.5965822664705596e-01,
            1.2801936062875557e-01,
            3.1951585968736973e-01,
            7.9623244052944575e-01,
            3.5832690095641251e00,
            4.4201185993943746e00,
            1.6583920792183407e-01,
            1.6349040489652469e-01,
            2.0944161298573122e-01,
            2.0661100786519029e-01,
            7.8544529603976498e-01,
            3.4575321265156798e00,
            4.2632018765042359e00,
            1.8738734035077514e-01,
            1.1221864515467743e-01,
            2.4601657859914894e-01,
            1.5985464601264134e00,
            7.2302277317421226e-01,
            5.2938320909160241e-01,
            2.3054354773013225e-01,
            8.0640415675395638e-01,
        ]
    )
    print(len(new_x))
    # spline_surf.set_dofs_from_vec(new_x)
    spline_surf.x = new_x
    print_dofs_nicely(spline_surf)

    spline_surf.plot()
    plt.show()

    data_init = np.zeros((64, 64, 3))
    u = np.linspace(0, 2 * np.pi, 64, endpoint=True)
    v = np.linspace(0, 2 * np.pi, 64, endpoint=True)

    initial = spline_surf.gamma_impl(
        data_init, v / (2 * np.pi), u / (2 * np.pi)
    )

    rz_surf = spline_surf.to_RZFourier(
        # nu=64,
        # nv=64,
        # nv_interp=128,
        # nu_interp=128,
        # collocation="arclength",
        # plot=True,
        # spec_cond=None,
        # spec_cond_options={
        #     "plot": False,
        #     "ftol": 1e-4,
        #     "Mtol": 1.1,
        #     "shapetol": None,
        #     "niters": 2000,
        #     "verbose": True,
        #     "cutoff": 1e-5,
        # },
    )

    # condensed, data = rz_surf.condense_spectrum(method='trf', Fourier_continuation=True)
    # plot_spectral_condensation(rz_surf, condensed, data)

    plt.show()

    # refine_poloidal() (Lane-Riesenfeld/Boehm knot insertion) is exact --
    # the max abs deviation below should be ~1e-14. A plain mean of the
    # signed before/after difference would hide a real error: it lets
    # positive and negative deviations across the grid cancel, so it can
    # look reassuringly close to zero even when the max deviation is not
    # small at all -- max abs is the meaningful check. (Toroidal, i.e.
    # cross-section-count, refinement isn't implemented -- see
    # SurfaceBSpline.refine_poloidal's docstring.)
    cs_list_before = list(spline_surf.cs_list)
    spline_surf.refine_poloidal()
    data_fin = np.zeros((64, 64, 3))
    spline_surf.gamma_impl(data_fin, v / (2 * np.pi), u / (2 * np.pi))
    print(
        "max |gamma diff| after refine_poloidal() (expect ~machine precision):",
        np.max(np.abs(data_init - data_fin)),
    )
    spline_surf.plot_cross_sections(
        [cs_list_before, spline_surf.cs_list],
        labels=["before refine_poloidal", "after refine_poloidal"],
    )
    plt.show()

    spline_surf.plot()
    plt.show()

    vmec = Vmec.vmec_from_surf(
        nfp=rz_surf.nfp, surf=rz_surf, mpi=mpi, ns=13, M=12, N=12, ftol=1e-7
    )
    vmec.run()
    print(f"Volume: {vmec.volume()}")
