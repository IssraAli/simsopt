"""Export passive-bulk puck primitives and sheet currents for COMSOL import.

COMSOL Multiphysics does not import ``.vtu`` VTK as geometry (VTK is
export-only in recent COMSOL versions); for steady magnetostatics use a
CAD primitive or analytic cylinder plus an ``Interpolation`` function
built from spreadsheet data.  This module writes:

- A single JSON manifest listing each replica puck's ``(R, t, center,
  axis, quaternion)`` so a COMSOL *Method* can instantiate a cylinder
  per entry.
- One CSV per puck with columns ``x,y,z,nx,ny,nz,w,Kx,Ky,Kz`` at every
  shell quadrature point (aligned with :meth:`PSCBulkArray.get_shell_currents`).

Each CSV can feed ``Definitions > Functions > Interpolation`` (spreadsheet
format; use spatial coordinates ``x,y,z`` as the three arguments) and
then a ``Magnetic Fields > Surface Current Density`` boundary condition.

See :func:`save_pucks_comsol_bundle` for the on-disk layout.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

import numpy as np

if TYPE_CHECKING:
    from .psc_bulk import PSCBulkArray

_COMSOL_NOTES = (
    "Per-puck CSV: create Definitions > Functions > Interpolation from spreadsheet; "
    "set argument columns to x,y,z and function columns to Kx,Ky,Kz (vector). "
    "Apply Surface Current Density on the corresponding puck boundary using those functions. "
    "Geometry: build each puck as a Cylinder from the JSON center, axis, R, and thickness t "
    "(replicate orientation via the quaternion or rotation from axis)."
)


def _axis_quat_for(
    psc: "PSCBulkArray", puck_index: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Return unit axis and quaternion for replica ``puck_index``."""
    center, axis, _R, _t = psc._all_pucks[puck_index]
    ax = np.asarray(axis, dtype=np.float64).reshape(3)
    nrm = float(np.linalg.norm(ax))
    if nrm <= 0.0:
        raise ValueError(f"Puck {puck_index}: axis has zero norm.")
    ax_u = ax / nrm
    q = np.asarray(psc._all_pucks_quats[puck_index], dtype=np.float64).reshape(4)
    return ax_u, q


def save_pucks_comsol_bundle(
    psc_bulk: "PSCBulkArray",
    dirpath: str | Path,
    *,
    prefix: str = "puck",
    csv_delimiter: str = ",",
) -> Dict[str, Any]:
    """Write a COMSOL-oriented bundle: JSON geometry manifest + per-puck K CSVs.

    Parameters
    ----------
    psc_bulk
        Rebuilt :class:`~simsopt.field.psc_bulk.PSCBulkArray` with valid
        ``_quad_*`` geometry and ``beta`` (call :meth:`recompute_currents`
        before export if TF coils moved).
    dirpath
        Output directory (created if missing).
    prefix
        Base name for files: ``{prefix}_geometry.json`` and
        ``{prefix}_{id:03d}_K.csv``.
    csv_delimiter
        Single-character delimiter for CSV rows (default comma).

    Returns
    -------
    dict
        Summary with keys ``manifest_path``, ``n_pucks``,
        ``csv_paths`` (list of :class:`pathlib.Path` objects).

    Raises
    ------
    ValueError
        If quadrature bookkeeping is inconsistent with ``get_shell_currents``.
    """
    if len(csv_delimiter) != 1:
        raise ValueError("csv_delimiter must be a single character.")

    out = Path(dirpath)
    out.mkdir(parents=True, exist_ok=True)

    n_pucks = len(psc_bulk._all_pucks)
    if len(psc_bulk._quad_row_ranges) != n_pucks:
        raise ValueError(
            "_quad_row_ranges length does not match _all_pucks; rebuild PSCBulkArray."
        )

    K_vec, _K_mag = psc_bulk.get_shell_currents()
    K_vec = np.asarray(K_vec, dtype=np.float64).reshape(-1, 3)
    nq_total = int(psc_bulk._quad_points.shape[0])
    if K_vec.shape[0] != nq_total:
        raise ValueError(
            "get_shell_currents length does not match _quad_points; inconsistent state."
        )

    pts = np.asarray(psc_bulk._quad_points, dtype=np.float64).reshape(nq_total, 3)
    normals = np.asarray(psc_bulk._quad_normals, dtype=np.float64).reshape(nq_total, 3)
    weights = np.asarray(psc_bulk._quad_weights, dtype=np.float64).reshape(nq_total)

    base_indices = np.asarray(
        getattr(psc_bulk, "_all_puck_base_indices", np.arange(n_pucks)),
        dtype=np.int64,
    ).reshape(n_pucks)

    pucks_meta: List[Dict[str, Any]] = []
    csv_paths: List[Path] = []

    for p in range(n_pucks):
        center, _axis0, R_val, t_val = psc_bulk._all_pucks[p]
        ax_u, quat = _axis_quat_for(psc_bulk, p)
        r0, r1 = psc_bulk._quad_row_ranges[p]
        n_local = int(r1 - r0)
        if n_local <= 0:
            raise ValueError(f"Puck {p}: empty quadrature range.")

        fname = f"{prefix}_{p:03d}_K.csv"
        csv_path = out / fname

        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            wtr = csv.writer(fh, delimiter=csv_delimiter)
            wtr.writerow(["x", "y", "z", "nx", "ny", "nz", "w", "Kx", "Ky", "Kz"])
            fh.write(
                f"% puck_id={p} base_id={int(base_indices[p])} "
                f"R={float(R_val):.16g} t={float(t_val):.16g} n_quad={n_local}\n"
            )
            block = np.hstack(
                [
                    pts[r0:r1],
                    normals[r0:r1],
                    weights[r0:r1].reshape(-1, 1),
                    K_vec[r0:r1],
                ]
            )
            for row in block:
                wtr.writerow([f"{float(v):.16g}" for v in row])

        csv_paths.append(csv_path)

        pucks_meta.append(
            {
                "id": int(p),
                "base_id": int(base_indices[p]),
                "R": float(R_val),
                "t": float(t_val),
                "center": [float(x) for x in np.asarray(center, dtype=np.float64).ravel()[:3]],
                "axis": [float(x) for x in ax_u],
                "quat": [float(x) for x in quat.ravel()[:4]],
                "n_quad": n_local,
                "csv": fname,
            }
        )

    manifest: Dict[str, Any] = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_base_pucks": int(getattr(psc_bulk, "_n_base_pucks", 0)),
        "n_pucks": int(n_pucks),
        "nfp": int(psc_bulk.nfp),
        "stellsym": bool(psc_bulk.stellsym),
        "units": {"length": "m", "K": "A/m"},
        "comsol_notes": _COMSOL_NOTES,
        "pucks": pucks_meta,
    }

    man_path = out / f"{prefix}_geometry.json"
    with man_path.open("w", encoding="utf-8") as jf:
        json.dump(manifest, jf, indent=2)

    return {
        "manifest_path": man_path,
        "n_pucks": n_pucks,
        "csv_paths": csv_paths,
    }


__all__ = ["save_pucks_comsol_bundle"]
