#!/usr/bin/env python
r"""
Passive bulk (ideal diamagnetic) cylindrical pucks: linear shell solve and ParaView export.

This example uses a single planar TF coil and one thick cylindrical puck (default 20 mm thickness).
The passive field is ``PassiveBulkField`` + ``BiotSavart`` on the TF coil.

Requirements: JAX, pyevtk (for VTK output).
"""

import os

import numpy as np

from simsopt.field import Coil, Current, PSCBulkArray
from simsopt.field.biotsavart import BiotSavart
from simsopt.field.magneticfield import MagneticFieldSum
from simsopt.field.puck_vtk import pucks_to_vtk
from simsopt.geo import CurveXYZFourier

OUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "passive_bulks_out",
)
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    # Single circular TF coil in the z=0 plane, major radius 1 m
    curve = CurveXYZFourier(48, 1)
    curve.x = np.array([0, 0, 1, 0, 1, 0, 0, 0.0, 0.0]) * 1.0
    tf_coil = Coil(curve, Current(5.0e4))
    coils_tf = [tf_coil]

    # One puck: center above the coil, axis along +z, 5 cm radius, 2 cm thickness
    centers = np.array([[0.0, 0.0, 0.2]])
    axes = np.array([[0.0, 0.0, 1.0]])
    R = 0.05
    t = 0.02

    eval_pts = np.array([[0.02, 0.0, 0.25]], dtype=float)

    psc = PSCBulkArray(
        centers,
        axes,
        np.array([R]),
        np.array([t]),
        coils_tf,
        eval_points=eval_pts,
        m_fourier=2,
        l_zernike=4,
        k_chebyshev=2,
        n_rho=6,
        n_phi=8,
        n_z=4,
        nfp=1,
        stellsym=False,
    )

    b_bulk = psc.biot_savart
    b_tf = BiotSavart(coils_tf)
    b_tot = MagneticFieldSum([b_bulk, b_tf])

    b_tot.set_points_cart(np.ascontiguousarray(eval_pts))
    B = b_tot.B()
    print("B total at eval point (T):", B)

    K, Kmag = psc.get_shell_currents()
    print("max |K| on shell quad (A/m):", np.max(Kmag))

    vtk_path = os.path.join(OUT_DIR, "passive_bulk_pucks")
    pucks_to_vtk(psc, vtk_path, n_phi=32, n_r=12)
    print("Wrote", vtk_path + ".vtu")

    print("Done.")


if __name__ == "__main__":
    main()
