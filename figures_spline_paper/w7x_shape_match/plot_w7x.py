import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import rcParams
from simsopt.geo import SurfaceBSpline

mpl.use("QTAgg")

TARGET_FILE = "/Users/issraali/codes/simsopt/tests/test_files/input.NCSX_c09r00_halfTeslaTF"


def print_dofs_nicely(surf):
    dofs_lb_ub = list(zip(surf.x, surf.lower_bounds, surf.upper_bounds))
    dofs_dict = dict(zip(surf.dof_names, dofs_lb_ub))
    print(
        "{:<30} {:<20} {:<20} {:<20}".format(
            "dof", "value", "lower bound", "upper bound"
        )
    )
    for k, v in dofs_dict.items():
        val, lb, ub = v
        print("{:<30} {:<20} {:<20} {:<20}".format(k, val, lb, ub))


if __name__ == "__main__":
    spline_kwargs = {
        "axis_points": 3,
        "points_per_cs": 6,
        "n_cs": 5,
        "nfp": 5,
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
    }

    surf = SurfaceBSpline(**spline_kwargs)
    surf.x = np.array(
        [
            4.05247181e-62,
            2.18674898e-01,
            1.61673161e-01,
            9.27558884e-02,
            1.18847652e00,
            1.57079633e00,
            5.55176298e-41,
            2.15958282e-01,
            1.73734624e-01,
            8.15742795e-02,
            2.17619686e-01,
            1.37203386e-01,
            6.41839361e-01,
            1.57079633e00,
            3.66519143e00,
            4.71238898e00,
            4.85176018e00,
            1.40532392e-01,
            2.16056728e-01,
            1.24474530e-01,
            8.96945769e-02,
            2.11725895e-01,
            3.97578557e-03,
            6.59295508e-01,
            1.57079633e00,
            3.66519143e00,
            4.57136006e00,
            4.71238898e00,
            1.64293469e-01,
            1.64267848e-01,
            7.96049660e-02,
            1.16422038e-01,
            1.79508228e-01,
            9.93539362e-02,
            1.17513485e00,
            2.41146910e00,
            3.66519143e00,
            3.66519143e00,
            5.68966332e00,
            1.05535800e-01,
            1.59739705e-01,
            1.47522042e-01,
            1.48479889e-01,
            9.84174928e-01,
            3.14159265e00,
            1.09388638e00,
            1.08348931e00,
            8.99041841e-01,
            6.09278924e-02,
        ]
    )



    ########################
    # plotting

    # style
    rcParams.update(
        {
            "font.size": 8,
            "text.usetex": True,
            "font.weight": "book",
            "font.family": "Futura PT",
            "lines.linewidth": 1,
        }
    )
    paper_width = (174) / 25.4  # width of paper in inches
    fig_height = 3.5  # figure height in inches
    cmap = mpl.colormaps["plasma"]
    color = cmap(np.linspace(0.8, 0.2, 3))
    style_dict_uniform = {"label": "Polar angle", "color": color[0], "ls": "-"}
    style_dict_arclength = {"label": "Arclength", "color": color[1], "ls": "-"}
    style_dict_cond = {"label": "Condensed", "color": color[2], "ls": "-"}

    fig = plt.figure()  # (figsize=(paper_width, fig_height), dpi=300)
    fig.canvas.draw()
    gs = gridspec.GridSpec(1, 1, wspace=0.4, top=1, left=0, bottom=0, right=1)
    ax1 = fig.add_subplot(gs[0, 0], projection="3d")
    ax1.view_init(elev=35, azim=-25.5, roll=0)
    fig.canvas.draw()

    # # plotting only convex hull of control points
    # xyz_list = surf.get_xyz_full_device()

    # pts = np.vstack(xyz_list)
    # # print(pts.shape)

    # x_proj, y_proj, _ = proj3d.proj_transform(
    #     pts[:, 0], pts[:, 1], pts[:, 2], ax1.get_proj()
    # )

    # xy_disp = ax1.transData.transform(np.vstack([x_proj, y_proj]).T)
    # # print(xy_disp)
    # # fig, ax2 = plt.subplots()
    # # ax2.plot(xy_disp[:, 0], xy_disp[:, 1])
    # xmin = np.min(xy_disp[:, 0])
    # xmax = np.max(xy_disp[:, 0])
    # ymin = np.min(xy_disp[:, 1])
    # ymax = np.max(xy_disp[:, 1])

    # # #print(xmin, xmax, ymin, ymax)
    # # bbox_display = Bbox.from_extents(xmin, ymin, xmax, ymax)
    # # print(bbox_display)

    # # fig_bbox_disp = bbox_display.transformed(fig.transFigure.inverted())
    # # bbox_inches = bbox_display.transformed(fig.dpi_scale_trans.inverted())

    # # xmin, ymin = xy_disp.min(axis=0)
    # # xmax, ymax = xy_disp.max(axis=0)

    # # fb = fig.get_window_extent()  # display px bounds of the figure canvas
    # # xmin = max(fb.x0, xmin); ymin = max(fb.y0, ymin)
    # # xmax = min(fb.x1, xmax); ymax = min(fb.y1, ymax)

    # # bbox_disp = Bbox.from_extents(xmin, ymin, xmax, ymax)
    # # print(bbox_disp)
    # # x0 = (bbox_disp.x0 - fb.x0) / fig.dpi
    # # y0 = (bbox_disp.y0 - fb.y0) / fig.dpi
    # # x1 = (bbox_disp.x1 - fb.x0) / fig.dpi
    # # y1 = (bbox_disp.y1 - fb.y0) / fig.dpi

    # # bbox_inches = Bbox.from_extents(x0, y0, x1, y1)

    # target_surf = SurfaceRZFourier.from_focus(TARGET_FILE)
    # R0_orig = abs(target_surf.get_rc(0, 0))
    # scale = 1.0 / R0_orig
    # target_surf.rc[:, :] *= scale
    # target_surf.zs[:, :] *= scale
    # target_surf.plot(ax=ax1, rcount=100, ccount=100, show=False)

    surf.plot(
        _surf=True,
        _surf_points=False,
        _ctrl_points=False,
        _ctrl_points_full=False,
        _ctrl_net=True,
        _pseudo_axis=True,
        _centroid_axis=False,
        _rtz_vectors=True,
        _RZ_vectors=False,
        _surf_kwargs={
            "color": "#64CA99",
            "alpha": 0.4,
            "rcount": 100,
            "ccount": 100,
        },
        _ctrl_net_kwargs={
            "color": "#1CC8A0",
            "ls": "--",
            "linewidth": 0.5,
        },
        _pseudo_axis_kwargs={"color": "#005818", "ls": "-", "lw": 1},
        _centroid_axis_kwargs={"color": "#22A7FA", "ls": "--", "lw": 1},
        _pseudo_axis_ctrl_pts_kwargs={
            "color": "#000000",
            "marker": ".",
            "lw": 0.5,
            "markersize": 2,
        },
        _rtz_vectors_kwargs={"color": "#005818", "lw": 1, "alpha": 0.1},
        ax=ax1,
    )

    # plt.show()

    # print_dofs_nicely(surf)
    # plt.savefig(dpi=fig.dpi, fname="ncsx_splines.pdf", pad_inches=0.05)

    plt.show()
