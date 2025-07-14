#!/usr/bin/env python
"""
auglag_qa_scan_call.py
===============

This script performs coil optimization for stellarator devices using the Augmented
Lagrangian Method (ALM). This version has been modified to accept command-line
arguments for key parameters to facilitate automated scanning.

Usage (for scanning):
---------------------
python auglag_qa_scan.py --cs_threshold [VALUE] --length_target [VALUE] --out_dir [DIRECTORY]

"""
import json
import numpy as np
import os
import time
import argparse
from pathlib import Path

# Simsopt imports
from simsopt.objectives import SquaredFlux, QuadraticPenalty
from simsopt.geo import (
    SurfaceRZFourier, create_equally_spaced_curves, LinkingNumber,
    CurveLength, CurveCurveDistance, LpCurveCurvature, CurveSurfaceDistance,
    MeanSquaredCurvature
)
from simsopt.solve import augmented_lagrangian_method
from simsopt.util import in_github_actions
from simsopt.field import BiotSavart, Current, coils_to_vtk, coils_via_symmetries, regularization_circ

def parse_arguments():
    """
    Parses command-line arguments for the optimization script.
    """
    parser = argparse.ArgumentParser(description="Run a single coil optimization scenario.")
    parser.add_argument(
        "--cs_threshold",
        type=float,
        default=0.15,
        help="Coil-to-surface distance threshold."
    )
    parser.add_argument(
        "--length_target",
        type=float,
        default=30.0,
        help="Target for the maximum coil length."
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./output/",
        help="Directory to save output files."
    )
    return parser.parse_args()

def main():
    """
    Main function to set up and run the coil optimization.
    """
    args = parse_arguments()

    # Use arguments to define constants
    OUT_DIR = args.out_dir
    CS_THRESHOLD = args.cs_threshold
    LENGTH_TARGET = args.length_target
    
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Running optimization with:")
    print(f"  Coil-Surface Threshold: {CS_THRESHOLD}")
    print(f"  Coil Length Target: {LENGTH_TARGET}")
    print(f"  Output Directory: {OUT_DIR}")


    # --- Boilerplate from original script ---

    # Define the test directory
    TEST_DIR = Path(__file__).parent / '../../..' / 'tests/test_files'

    # Define the filename
    filename = TEST_DIR / 'input.LandremanPaul2021_QA_lowres'

    # Set some parameters -- warning this is super low resolution!
    if in_github_actions:
        nphi = 4
        ntheta = 4
        MAXITER = 10
        MAXITER_lag = 5
    else:
        # Define the number of phi and theta points
        nphi = 32
        ntheta = 32
        MAXITER = 50  # 1500 for high-resolution
        MAXITER_lag = 10  # 50 for high-resolution

    # Define the surface
    s = SurfaceRZFourier.from_vmec_input(
        filename,
        range="half period",
        nphi=nphi,
        ntheta=ntheta)

    qphi = 4 * nphi
    qtheta = 4 * ntheta
    quadpoints_phi = np.linspace(0, 1, qphi)
    quadpoints_theta = np.linspace(0, 1, qtheta)
    s_plot = SurfaceRZFourier.from_vmec_input(
        filename,
        range="full torus",
        quadpoints_phi=quadpoints_phi,
        quadpoints_theta=quadpoints_theta)
    
    # Define the fixed bounds for the other constraints
    FLUX_THRESHOLD = 1e-15
    CC_THRESHOLD = 0.083
    MSC_THRESHOLD = 6
    CURVATURE_THRESHOLD = 12
    # FORCE_THRESHOLD = 0.012  # units of MN/m

    # Define the number of coils, rotation order, and non-planar base curves
    R0 = s.x[0]
    R1 = 0.7 * s.x[0]
    order = 7
    ncoils = 5
    curves = create_equally_spaced_curves(
        ncoils, s.nfp, stellsym=s.stellsym, R0=R0, R1=R1, order=order, numquadpoints=128)

    total_current_val = 3e5
    base_currents = [Current(total_current_val / ncoils * 1e-5) * 1e5 for _ in range(ncoils-1)]
    total_current = Current(total_current_val)
    total_current.fix_all()
    base_currents += [total_current - sum(base_currents)]


    base_curves = curves[:ncoils]
    a = 0.05
    regularizations = [regularization_circ(a) for _ in range(ncoils)]
    coils = coils_via_symmetries(base_curves, base_currents, s.nfp, s.stellsym, regularizations=regularizations)
    curves = [c.curve for c in coils]

    print("Number of coils:", len(coils))

    # Save the biot-savart field dataW
    bs = BiotSavart(coils)
    coils_to_vtk(coils, os.path.join(OUT_DIR, "coils_init_qa_scan"))
    
    # Plot initial surface
    bs.set_points(s_plot.gamma().reshape((-1, 3)))
    pointData = {"B_N/|B|": np.sum(bs.B().reshape((qphi, qtheta, 3)) *
                                   s_plot.unitnormal(), axis=2)[:, :, None] / bs.AbsB().reshape((qphi, qtheta, 1))}
    s_plot.to_vtk(os.path.join(OUT_DIR, "surf_init_qa_scan"), extra_data=pointData)

    # Define the individual terms objective function:
    bs.set_points(s.gamma().reshape((-1, 3)))
    Jf = SquaredFlux(s, bs, definition="normalized", threshold=FLUX_THRESHOLD)
    Jls = [CurveLength(c) for c in base_curves]
    # Jl = sum(QuadraticPenalty(jj, LENGTH_TARGET, "max") for jj in Jls)

    Jccdist = CurveCurveDistance(curves, CC_THRESHOLD, num_basecurves=ncoils)
    Jcsdist = CurveSurfaceDistance(curves, s, CS_THRESHOLD)
    Jcs = [LpCurveCurvature(c, 2, CURVATURE_THRESHOLD) for c in base_curves]
    Jlink = LinkingNumber(curves, downsample=2)
    # Jforce = LpCurveForce(coils[:ncoils], coils, p=2.0, threshold=FORCE_THRESHOLD)
    Jmscs = [MeanSquaredCurvature(c) for c in base_curves]
    
    # Print initial values
    print('Initial normalized flux:', Jf.J())
    print('Initial CS-sep minimum distance:', Jcsdist.shortest_distance())
    print('Initial CC-sep minimum distance:', Jccdist.shortest_distance())

    # Main optimization function
    f = None

    # Constraint list
    c_list = [ Jf,
            Jccdist,
            Jcsdist,
            sum(QuadraticPenalty(J, MSC_THRESHOLD, "max") for J in Jmscs),
            QuadraticPenalty(sum(Jls), LENGTH_TARGET, "max"),
            sum(Jcs),
            Jlink,
            # Jforce
    ]

    start_time = time.time()
    # Perform the optimization
    _ = augmented_lagrangian_method(f=f,
        equality_constraints=c_list,
        tau=4,
        MAXITER=MAXITER,
        MAXITER_lag=MAXITER_lag,
        grad_tol=1e-10,
        c_tol=1e-10,
    )
    end_time = time.time()
    
    print(f"Time taken: {end_time - start_time:.2f} seconds")

    # Save final results
    coils_to_vtk(coils, os.path.join(OUT_DIR, "coils_optimized_auglag_qa_scan"))
    bs.set_points(s_plot.gamma().reshape((-1, 3)))
    
    pointData = {"B_N/|B|": np.sum(bs.B().reshape((qphi, qtheta, 3)) *
                                   s_plot.unitnormal(), axis=2)[:, :, None] /
                   bs.AbsB().reshape((qphi, qtheta, 1))}
    max_BdotN_overB = np.max(np.sum(bs.B().reshape((qphi, qtheta, 3)) *
                            s_plot.unitnormal(), axis=2)[:, :, None] /
            bs.AbsB().reshape((qphi, qtheta, 1)))
    bs.set_points(s.gamma().reshape((-1, 3)))
    BdotN = np.mean(np.abs(np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)))
    avg_BdotN_over_B = BdotN / bs.AbsB().mean()

    results = {
        "nfp": s.nfp,
        "ncoils": int(ncoils),
        "order": int(order),
        "nphi": nphi,
        "ntheta": ntheta,
        "R0": R0,  # if initialized from circles, else None
        "R1": R1,  # if initialized from circles, else None
        "length_target": LENGTH_TARGET,
        "max_κ_threshold": CURVATURE_THRESHOLD,
        "msc_threshold": MSC_THRESHOLD,
        "cc_threshold": CC_THRESHOLD,
        "cs_threshold": CS_THRESHOLD,
        # "force_threshold": FORCE_THRESHOLD,
        "Jf": float(Jf.J()),
        "lengths": [float(J.J()) for J in Jls],
        "max_length": max(float(J.J()) for J in Jls),
        "max_κ": [np.max(c.kappa()) for c in base_curves],
        "max_max_κ": max(np.max(c.kappa()) for c in base_curves),
        "MSCs": [float(J.J()) for J in Jmscs],
        "max_MSC": max(float(J.J()) for J in Jmscs),
        "BdotN": BdotN,
        "normalized_BdotN": avg_BdotN_over_B,
        "max_BdotN_overB": max_BdotN_overB,
        "coil_coil_distance": Jccdist.shortest_distance(),
        "coil_surface_distance": Jcsdist.shortest_distance(),
        "coil_currents": [c.get_value() for c in base_currents],
        "eval_time": time.perf_counter() - start_time,
    }

    with open(os.path.join(OUT_DIR, "results.json"), "w") as outfile:
        json.dump(results, outfile, indent=2)
    bs.save(os.path.join(OUT_DIR,"biot_savart_optimized_auglag_qa_scan.json"))  # save the optimized coil shapes and currents

    s_plot.to_vtk(os.path.join(OUT_DIR, "surf_optimized_auglag_qa_scan"), extra_data=pointData)
    print("----------------------------------------------------")
    print("FINAL NORMALIZED SQUARED FLUX:", avg_BdotN_over_B)
    print('Final CS-sep minimum distance:', Jcsdist.shortest_distance())
    print('Final CC-sep minimum distance:', Jccdist.shortest_distance())
    print('Final Lengths:', [CurveLength(c).J() for c in base_curves])
    print('FINISHED OPTIMIZATION')

if __name__ == "__main__":
    main()
