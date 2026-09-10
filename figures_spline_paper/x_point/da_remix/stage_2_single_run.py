#!/usr/bin/env python
"""
Single stage-2 coil optimization run, reproducing the run whose output
directory is referenced (but no longer present) as the `coils_filename`
in 20240305-02-poincare_plot_for_stage_2_with_x_point_theta0_0.88_SOL.py:

    order_10_R1_0.42_length_target_4.8_weight_0.21_max_curvature_7.1_weight_0.00022
    _msc_1.4e+01_weight_0.00044_cc_0.11_weight_9.7e+01

The hyperparameters below are exactly what run_optimization's own
directory-naming f-string (stage_2_scan_twin.py) would round to (via its
":.2" -- 2-significant-figure -- format spec) for that directory name, so
this reproduces the original run's output path. msc_threshold=14.0 and
cc_weight=97.0 aren't uniquely determined by that rounding (multiple
values round to "1.4e+01"/"9.7e+01"), so the exact center of each
rounding bucket is used.

Reuses build_flux_grids/run_optimization from stage_2_scan_twin.py
rather than duplicating that logic, running once (N_JOBS=1) instead of
stage_2_scan_twin.py's random sweep.
"""

import os

from stage_2_scan_twin import SWEEP_DIR, build_flux_grids, run_optimization

if __name__ == "__main__":
    os.makedirs(SWEEP_DIR, exist_ok=True)
    (
        surf_outboard,
        surf_inboard,
        grid_outboard,
        grid_inboard,
        pos_outboard,
        normal_outboard,
        weights_outboard,
        pos_inboard,
        normal_inboard,
        weights_inboard,
    ) = build_flux_grids()
    surf_outboard.to_vtk(SWEEP_DIR + "surf_outboard")
    surf_inboard.to_vtk(SWEEP_DIR + "surf_inboard")

    run_optimization(
        grid_outboard,
        grid_inboard,
        pos_outboard,
        normal_outboard,
        weights_outboard,
        pos_inboard,
        normal_inboard,
        weights_inboard,
        R1=0.42,
        order=10,
        length_target=4.8,
        length_weight=0.21,
        max_curvature_threshold=7.1,
        max_curvature_weight=0.00022,
        msc_threshold=14.0,
        msc_weight=0.00044,
        cc_threshold=0.11,
        cc_weight=97.0,
        index=0,
        n_jobs=1,
    )
