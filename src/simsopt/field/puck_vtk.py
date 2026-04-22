"""
VTK export for cylindrical passive-bulk pucks (finite-build visualization in ParaView).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

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
    *,
    quat: Optional[np.ndarray] = None,
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
        quat: Optional scalar-first quaternion ``[w, x, y, z]`` mapping the
            local frame to global.  When provided, this rotation is used instead
            of the shortest-arc map from ``(0,0,1)`` to ``axis_z`` (which drops
            in-plane roll).  For :class:`~simsopt.field.psc_bulk.PSCBulkArray` VTK
            export, pass the replica's ``_all_pucks_quats[i]`` so mesh vertices
            align with the quadrature used to build ``K`` and ``beta``.

    Returns:
        ``x, y, z`` (n_pts,) and ``triangles`` (n_tri, 3) vertex indices.
    """
    from .psc_bulk import _rotation_matrix_from_quat, _rotation_matrix_local_to_global

    c = np.asarray(center, dtype=float).ravel()
    ax = np.asarray(axis_z, dtype=float)
    ax = ax / np.linalg.norm(ax)
    if quat is not None:
        q = np.asarray(quat, dtype=float).ravel()
        Rmat = _rotation_matrix_from_quat(q)
    else:
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
    """Evaluates sheet current and scalar potential on mesh vertices in the
    **continuous** Fourier--Zernike / Fourier--Chebyshev basis.

    **Not** a nearest-quadrature lookup: the polar-grid quadrature only samples
    the basis for the Galerkin system; the continuous :math:`\\mathbf{K}(\\mathbf
    x) = \\sum_a \\beta_a \\mathbf{K}_a(\\mathbf x)` and :math:`g` can therefore
    be evaluated at an arbitrary :math:`\\mathbf x` on the closed puck shell
    (including the cap rim) without the ``n_phi``-keyed aliasing that shows up
    in ParaView if one splats quadrature values by nearest cell.

    The outward unit normal for :math:`B_n^{\\rm TF} = \\hat{\\mathbf n}\\cdot
    \\mathbf{B}^{\\rm TF}` still uses the **nearest** shell quadrature point so
    coincident top-rim and side-mesh nodes (duplicate 3D coordinates) get the
    correct face normal for the triangle the vertex belongs to.
    """
    import jax.numpy as jnp

    from .puck_basis import (
        _basis_spec_for_dof,
        _disk_grad_g_cartesian,
        _side_grad_K,
        zernike_radial,
    )
    from .psc_bulk import _rotation_matrix_from_quat

    pts = np.stack([x, y, z], axis=-1)
    r0, r1 = psc_bulk._quad_row_ranges[puck_index]
    quad = psc_bulk._quad_points[r0:r1]
    K_stack = psc_bulk._K_stack
    phi_stack = psc_bulk._phi_stack
    if K_stack is None or phi_stack is None:
        raise RuntimeError(
            "puck VTK export requires homogeneous-shape pucks with "
            "populated _K_stack / _phi_stack; this PSCBulkArray was built "
            "with heterogeneous pucks which are not supported."
        )

    c, _axis, R, t = psc_bulk._all_pucks[puck_index]
    q = np.asarray(psc_bulk._all_pucks_quats[puck_index], dtype=float).ravel()
    Rmat = _rotation_matrix_from_quat(q)
    c = np.asarray(c, dtype=float).ravel()

    basis = psc_bulk._basis_per_puck[puck_index]
    d0 = psc_bulk._dof_offsets[puck_index]
    nd = len(basis.dof_names)
    beta = np.asarray(psc_bulk.beta, dtype=float)
    beta_p = beta[d0 : d0 + nd]

    p_loc = (pts - c[None, :]) @ Rmat
    xl, yl, zl = p_loc[:, 0], p_loc[:, 1], p_loc[:, 2]
    rho_l = np.hypot(xl, yl)
    phi_l = np.arctan2(yl, xl)

    n_pts = int(pts.shape[0])
    g_out = np.zeros(n_pts, dtype=float)
    K_local = np.zeros((n_pts, 3), dtype=float)

    tol_z = 1e-9 * max(float(R), float(t), 1e-15)
    tol_r = 1e-6 * max(float(R), 1e-15)

    spec_list = (
        list(basis.basis_spec) if len(getattr(basis, "basis_spec", [])) == nd else None
    )

    for a in range(nd):
        ba = float(beta_p[a])
        if not np.isfinite(ba) or abs(ba) < 1e-300:
            continue
        if spec_list is not None:
            face, m, n_or_k, trig = spec_list[a]
        else:
            face, m, n_or_k, trig = _basis_spec_for_dof(basis, a)
        if face == "disk_top":
            mask = (np.abs(zl - 0.5 * t) < tol_z) & (rho_l <= R + tol_r)
            if not np.any(mask):
                continue
            sel = mask
            rho_s = rho_l[sel]
            phi_s = phi_l[sel]
            r_norm = jnp.array(rho_s / R)
            Rvals = np.array(zernike_radial(r_norm, m, n_or_k))
            ang = (
                1.0
                if m == 0
                else (np.cos(m * phi_s) if trig == "cos" else np.sin(m * phi_s))
            )
            g_a = Rvals * ang
            dgx, dgy, _ = _disk_grad_g_cartesian(
                jnp.array(rho_s),
                jnp.array(phi_s),
                R,
                m,
                n_or_k,
                trig if m > 0 else "cos",
            )
            dgx = np.asarray(dgx, dtype=float)
            dgy = np.asarray(dgy, dtype=float)
            g_out[sel] += ba * g_a
            K_local[sel, 0] += ba * (-dgy)
            K_local[sel, 1] += ba * (dgx)
        elif face == "disk_bot":
            mask = (np.abs(zl + 0.5 * t) < tol_z) & (rho_l <= R + tol_r)
            if not np.any(mask):
                continue
            sel = mask
            rho_s = rho_l[sel]
            phi_s = phi_l[sel]
            r_norm = jnp.array(rho_s / R)
            Rvals = np.array(zernike_radial(r_norm, m, n_or_k))
            ang = (
                1.0
                if m == 0
                else (np.cos(m * phi_s) if trig == "cos" else np.sin(m * phi_s))
            )
            g_a = Rvals * ang
            dgx, dgy, _ = _disk_grad_g_cartesian(
                jnp.array(rho_s),
                jnp.array(phi_s),
                R,
                m,
                n_or_k,
                trig if m > 0 else "cos",
            )
            dgx = np.asarray(dgx, dtype=float)
            dgy = np.asarray(dgy, dtype=float)
            g_out[sel] += ba * g_a
            K_local[sel, 0] += ba * dgy
            K_local[sel, 1] += ba * (-dgx)
        elif face == "side":
            mask = (rho_l >= R * (1.0 - 1e-5)) & (np.abs(zl) <= 0.5 * t * (1.0 + 1e-5))
            if not np.any(mask):
                continue
            sel = mask
            g_a, _grad, K_a = _side_grad_K(
                zl[sel],
                phi_l[sel],
                R,
                t,
                m,
                n_or_k,
                trig if m > 0 else "cos",
            )
            g_out[sel] += ba * g_a
            K_local[sel] += ba * K_a
        else:  # pragma: no cover
            continue

    K_out = K_local @ Rmat.T
    tree = cKDTree(quad)
    _, j_local = tree.query(pts, k=1)
    j_local = np.asarray(j_local, dtype=np.intp)
    j = r0 + j_local
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
        q_replica = np.asarray(psc_bulk._all_pucks_quats[idx], dtype=float)
        x, y, z, tri = puck_surface_mesh(
            c, ax, R, t, n_phi=n_phi, n_r=n_r, quat=q_replica
        )
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
