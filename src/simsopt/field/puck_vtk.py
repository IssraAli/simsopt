"""
VTK export for cylindrical passive-bulk pucks (finite-build visualization in ParaView).
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from scipy.spatial import cKDTree

from simsopt.field.biotsavart import BiotSavart

contig = np.ascontiguousarray

try:
    from pyevtk.hl import unstructuredGridToVTK
    from pyevtk.vtk import VtkTriangle
except ImportError:
    unstructuredGridToVTK = None
    VtkTriangle = None

__all__ = ["puck_surface_mesh", "pucks_to_vtk", "pucks_field_to_vtk"]


def puck_surface_mesh(
    center: np.ndarray,
    axis_z: np.ndarray,
    R: float,
    t: float,
    n_phi: int = 64,
    n_r: int = 16,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Triangulated closed surface of a right circular cylinder (caps + side).

    Args:
        center: (3,) center of the puck.
        axis_z: (3,) unit vector along the cylinder axis (bottom to top).
        R: radius (m).
        t: thickness / height (m).
        n_phi: azimuthal segments.
        n_r: radial segments on each disk (including rim).

    Returns:
        ``x, y, z`` (n_pts,) and ``triangles`` (n_tri, 3) vertex indices.
    """
    from .psc_bulk import _rotation_matrix_local_to_global

    c = np.asarray(center, dtype=float).ravel()
    ax = np.asarray(axis_z, dtype=float)
    ax = ax / np.linalg.norm(ax)
    Rmat = _rotation_matrix_local_to_global(ax)

    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    np.linspace(0, R, n_r)
    z_top = 0.5 * t
    z_bot = -0.5 * t

    pts: List[np.ndarray] = []
    # Top cap (fan from center for simplicity - duplicate center vertex)
    center_top = Rmat @ np.array([0.0, 0.0, z_top]) + c
    pts.append(center_top)
    idx0 = 0
    rim_top = []
    for j in range(n_phi):
        rho = R
        p = Rmat @ np.array([rho * np.cos(phi[j]), rho * np.sin(phi[j]), z_top]) + c
        pts.append(p)
        rim_top.append(1 + j)
    n_top = len(pts)
    # Bottom cap
    center_bot = Rmat @ np.array([0.0, 0.0, z_bot]) + c
    pts.append(center_bot)
    idx_bot_center = n_top
    rim_bot = []
    for j in range(n_phi):
        rho = R
        p = Rmat @ np.array([rho * np.cos(phi[j]), rho * np.sin(phi[j]), z_bot]) + c
        pts.append(p)
        rim_bot.append(idx_bot_center + 1 + j)
    len(pts)
    # Side wall: two rings at rho=R
    side_top = []
    side_bot = []
    for j in range(n_phi):
        p_top = Rmat @ np.array([R * np.cos(phi[j]), R * np.sin(phi[j]), z_top]) + c
        p_bot = Rmat @ np.array([R * np.cos(phi[j]), R * np.sin(phi[j]), z_bot]) + c
        side_top.append(len(pts))
        pts.append(p_top)
        side_bot.append(len(pts))
        pts.append(p_bot)

    xyz = np.array(pts)
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

    triangles: List[Tuple[int, int, int]] = []
    # Top triangles (fan)
    for j in range(n_phi):
        jn = (j + 1) % n_phi
        triangles.append((idx0, 1 + j, 1 + jn))
    # Bottom triangles (fan)
    for j in range(n_phi):
        jn = (j + 1) % n_phi
        triangles.append(
            (idx_bot_center, idx_bot_center + 1 + jn, idx_bot_center + 1 + j)
        )
    # Side quads -> two triangles
    for j in range(n_phi):
        jn = (j + 1) % n_phi
        a, b = side_top[j], side_top[jn]
        c1, d = side_bot[j], side_bot[jn]
        triangles.append((a, b, d))
        triangles.append((a, d, c1))

    tri = np.array(triangles, dtype=np.int64)
    return x, y, z, tri


def _sample_K_g_on_mesh(
    psc_bulk,
    puck_index: int,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate shell K and g at mesh vertices using nearest quad on this puck's shell patch."""
    pts = np.stack([x, y, z], axis=-1)
    r0, r1 = psc_bulk._quad_row_ranges[puck_index]
    quad = psc_bulk._quad_points[r0:r1]
    beta = np.asarray(psc_bulk.beta)
    # Use the stacked basis tables so we never allocate the dense
    # (nq_total, n_dof_total, 3) / (nq_total, n_dof_total) arrays just to
    # look up this puck's quadrature values.
    K_stack = psc_bulk._K_stack
    phi_stack = psc_bulk._phi_stack
    if K_stack is None or phi_stack is None:
        # The dense fallback that used to live here (``_K_basis`` /
        # ``_phi_mat`` lazy reconstructions) is unreachable in practice:
        # both lazy accessors raise ``RuntimeError`` whenever the stacks
        # are ``None`` (i.e. under heterogeneous puck shapes), and every
        # current factory (``cylindrical_grid_pucks``,
        # ``winding_surface_pucks``) builds homogeneous pucks with
        # populated stacks.  Surface the precondition loudly instead of
        # silently attempting a dead code path.
        raise RuntimeError(
            "puck VTK export requires homogeneous-shape pucks with "
            "populated _K_stack / _phi_stack; this PSCBulkArray was built "
            "with heterogeneous pucks which are not supported."
        )
    n_pucks, nq_per, nd_per, _ = K_stack.shape
    beta_s = beta.reshape(n_pucks, nd_per)
    K_vec_stack = np.einsum("pqak,pa->pqk", K_stack, beta_s)
    phi_vals_stack = np.einsum("pqa,pa->pq", phi_stack, beta_s)
    K_all = K_vec_stack.reshape(n_pucks * nq_per, 3)
    g_all = phi_vals_stack.reshape(n_pucks * nq_per)
    K_out = np.zeros_like(pts)
    g_out = np.zeros(len(pts))
    tree = cKDTree(quad)
    _, j_local = tree.query(pts, k=1)
    j_local = np.asarray(j_local, dtype=np.intp)
    j = r0 + j_local
    K_out = K_all[j]
    g_out = g_all[j]
    nn = psc_bulk._quad_normals[j]
    bs_tf = getattr(psc_bulk, "_vtk_bs_tf", None)
    if bs_tf is None:
        bs_tf = BiotSavart(psc_bulk.coils_TF)
        psc_bulk._vtk_bs_tf = bs_tf
    bs_tf.set_points_cart(contig(pts))
    B_tf = bs_tf.B()
    Bn = np.einsum("ij,ij->i", B_tf, nn)
    return np.linalg.norm(K_out, axis=-1), K_out, Bn, g_out


def pucks_to_vtk(
    psc_bulk,
    filename: str,
    n_phi: int = 64,
    n_r: int = 16,
) -> None:
    """
    Write all pucks in a :class:`~simsopt.field.psc_bulk.PSCBulkArray` to a VTK unstructured grid.

    Point data: ``K_magnitude``, ``K_vector`` (3), ``B_n_bg``, ``g_potential``.
    """
    if unstructuredGridToVTK is None:
        raise ImportError("pucks_to_vtk requires pyevtk (pip install pyevtk)")
    I_eq = psc_bulk.get_equivalent_currents()

    all_x: List[np.ndarray] = []
    all_y: List[np.ndarray] = []
    all_z: List[np.ndarray] = []
    all_Km: List[np.ndarray] = []
    all_Kv: List[np.ndarray] = []
    all_Bn: List[np.ndarray] = []
    all_g: List[np.ndarray] = []
    all_Ieq: List[np.ndarray] = []
    offset = 0
    all_tri: List[np.ndarray] = []

    for idx, (c, ax, R, t) in enumerate(psc_bulk._all_pucks):
        x, y, z, tri = puck_surface_mesh(c, ax, R, t, n_phi=n_phi, n_r=n_r)
        Km, Kv, Bn, g = _sample_K_g_on_mesh(psc_bulk, idx, x, y, z)
        all_x.append(x)
        all_y.append(y)
        all_z.append(z)
        all_Km.append(Km)
        all_Kv.append(Kv)
        all_Bn.append(Bn)
        all_g.append(g)
        all_Ieq.append(np.full_like(x, I_eq[idx]))
        all_tri.append(tri + offset)
        offset += len(x)

    x = contig(np.concatenate(all_x))
    y = contig(np.concatenate(all_y))
    z = contig(np.concatenate(all_z))
    tri = contig(np.vstack(all_tri))
    connectivity = contig(tri.reshape(-1))
    offsets = contig(3 * np.arange(tri.shape[0]) + 3)
    cell_types = contig(np.full(offsets.shape, VtkTriangle.tid))

    K_mag = contig(np.concatenate(all_Km))
    K_vec = (
        contig(np.concatenate(all_Kv)[:, 0]),
        contig(np.concatenate(all_Kv)[:, 1]),
        contig(np.concatenate(all_Kv)[:, 2]),
    )
    Bn_bg = contig(np.concatenate(all_Bn))
    g_pot = contig(np.concatenate(all_g))
    I_equivalent = contig(np.concatenate(all_Ieq))

    unstructuredGridToVTK(
        str(filename),
        x,
        y,
        z,
        connectivity,
        offsets,
        cell_types,
        pointData={
            "K_magnitude": K_mag,
            "K_vector": K_vec,
            "B_n_bg": Bn_bg,
            "g_potential": g_pot,
            "I_equivalent": I_equivalent,
        },
    )  # type: ignore[call-arg]


def pucks_field_to_vtk(
    psc_bulk,
    filename: str,
    grid_points_xyz: np.ndarray,
) -> None:
    """
    Write :math:`\\mathbf{B}` from passive bulk on a user grid (N, 3) as VTK structured grid.

    Requires ``pyevtk`` and ``gridToVTK``.
    """
    from pyevtk.hl import gridToVTK

    pts = np.asarray(grid_points_xyz, dtype=float, order="C")
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("grid_points_xyz must have shape (N, 3)")
    B = psc_bulk.B_at_points(pts)
    # Unstructured point cloud as 1x1xN structured hack
    n = pts.shape[0]
    x = contig(pts[:, 0].reshape((1, 1, n)))
    y = contig(pts[:, 1].reshape((1, 1, n)))
    z = contig(pts[:, 2].reshape((1, 1, n)))
    gridToVTK(
        str(filename),
        x,
        y,
        z,
        pointData={
            "B": (
                contig(B[:, 0].reshape((1, 1, n))),
                contig(B[:, 1].reshape((1, 1, n))),
                contig(B[:, 2].reshape((1, 1, n))),
            )
        },
    )
