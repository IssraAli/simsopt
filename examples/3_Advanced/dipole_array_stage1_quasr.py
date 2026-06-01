#!/usr/bin/env python
r"""
Stage-1 QUASR-style optimization for a fixed-geometry TF + dipole coil set.

Vacuum Biot-Savart field only; boundary is a ``BoozerSurface`` (psi-free QS
metrics). Default Boozer solver is **BoozerLS** (L-BFGS-B + Newton polish), which
is more robust than BoozerExact and still supplies ``res['PLU']`` / ``res['vjp']``
for gradient-aware objectives.

Scan modes
----------
- Serial: run all entries in ``CASES``, or restrict with ``--case-idx N``.
- MPI: ``mpirun -n K python dipole_array_stage1_quasr.py --mpi`` assigns cases
  round-robin to ranks; rank 0 writes ``scan_summary.csv``.
- ``--ci`` collapses Fourier stages and disables Poincaré; serial ``scan_summary.csv``
  is refreshed after **each** completed case.

For a **fast Poincaré path check** (still not ``--ci`` / not ``--no-poincare``), set
``QUASR_POINCARE_SMOKE=1`` to shorten each selected case to a single cheap stage
with smaller fieldline tracing limits.

Outputs per case (under ``--out-dir / <case.name> /``)
-----------------------------------------------------
- ``coils_initial.vtu``, ``coils_optimized.vtu`` — signed current ``I`` via
  :func:`simsopt.field.coils_to_vtk`.
- ``plasma_err_<tag>.vts`` — target plasma boundary with ``B_N``, ``B_N_over_B``, ``modB``.
- ``surf_<stage>.vts``, ``surf_optimized.vts`` — Boozer surface with ``modB``.
- ``poincare_<tag>.png``, ``fieldlines_<tag>.vtu`` — after optimization (unless ``--no-poincare``).
- ``metrics.json``, ``stage*.json``, ``stage1_final.json``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from functools import partial
from scipy.integrate import solve_ivp
from scipy.linalg import lu as _scipy_lu
from scipy.optimize import minimize

from simsopt.geo.surfaceobjectives import boozer_surface_dlsqgrad_dcoils_vjp

from simsopt import save as simsopt_save
from simsopt._core import Optimizable
from simsopt._core.derivative import derivative_dec
from simsopt.field import (
    BiotSavart,
    Current,
    InterpolatedField,
    SurfaceClassifier,
    coils_to_vtk,
    coils_via_symmetries,
    compute_fieldlines,
    LevelsetStoppingCriterion,
    plot_poincare_data,
    particles_to_vtk,
    regularization_rect,
)
from simsopt.geo import (
    AspectRatio,
    BoozerResidual,
    BoozerSurface,
    CurveXYZFourier,
    Iotas,
    MajorRadius,
    NonQuasiSymmetricRatio,
    NonQuasiSymmetricRatioHelical,
    SurfaceRZFourier,
    SurfaceXYZTensorFourier,
    Volume,
    create_planar_curves_between_two_toroidal_surfaces,
)
from simsopt.objectives import QuadraticPenalty
from simsopt.util import (
    align_dipoles_with_plasma,
    initialize_coils,
    remove_inboard_dipoles,
    remove_interlinking_dipoles_and_TFs,
)

try:
    from mpi4py import MPI as _MPI

    HAS_MPI = True
except ImportError:
    _MPI = None
    HAS_MPI = False


TEST_FILES_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()

CI_STAGES: Tuple[Tuple[int, int, int, int, float], ...] = (
    (4, 4, 16, 5, 1e-11),
)

DEFAULT_STAGES_QA: Tuple[Tuple[int, int, int, int, float], ...] = (
    (4, 4, 16, 100, 1e-11),
    (8, 8, 24, 500, 1e-13),
)

DEFAULT_STAGES_QH: Tuple[Tuple[int, int, int, int, float], ...] = (
    (4, 4, 16, 200, 1e-11),
    (8, 8, 24, 1500, 1e-13),
)

# Frozen 3-stage schedules for legacy CASES 0-3 (byte-identical to pre-speedup runs).
_LEGACY_STAGES_QA: Tuple[Tuple[int, int, int, int, float], ...] = (
    (4, 4, 16, 100, 1e-11),
    (6, 6, 20, 200, 1e-12),
    (8, 8, 24, 500, 1e-13),
)

_LEGACY_STAGES_QH: Tuple[Tuple[int, int, int, int, float], ...] = (
    (4, 4, 16, 200, 1e-11),
    (6, 6, 20, 500, 1e-12),
    (8, 8, 24, 1500, 1e-13),
)


@dataclass
class Case:
    """One physics configuration / scan point."""

    name: str
    target: str  # 'QA' | 'QH'
    plasma_input: str  # basename under tests/test_files
    iota_target: float
    aspect_target: Optional[float] = None
    R0_factor: float = 1.0
    nfp_override: Optional[int] = None
    stellsym_override: Optional[bool] = None
    helicity_M: int = 1
    helicity_N: Optional[int] = None  # default: plasma nfp
    boozer_type: str = "ls"
    weights: Dict[str, float] = field(
        default_factory=lambda: dict(
            W_SYMM=1.0,
            W_IOTA=1.0,
            W_MR=10.0,
            W_VOL=1.0,
            W_BR=1.0e4,
            W_AR=0.0,
            W_IMAX_FRAC=1.0e-3,
            CONSTRAINT_WEIGHT=1.0e4,
        )
    )
    grid_layout: str = "cartesian_bbox"
    df_loop_radius_factor: float = 0.4
    stages: Tuple[Tuple[int, int, int, int, float], ...] = field(
        default_factory=lambda: DEFAULT_STAGES_QA
    )
    a_edge_frac: float = 0.25
    n_dipole_per_axis: int = 4  # legacy; used when the triple below is None
    n_dipole_phi: Optional[int] = None
    n_dipole_theta: Optional[int] = None
    n_dipole_radial: Optional[int] = None
    df_planar_order: int = 2
    prune_inboard: bool = True
    prune_interlinking: bool = True
    plasma_poff: float = 1.5
    plasma_coff: float = 3.0
    dipole_current_init: float = 0.0
    dipole_current_init_amp: float = 0.0  # MA; uniform in [-amp, amp] when > 0
    iota_guess: float = 0.0
    tf_reg_a: float = 0.20
    tf_reg_b: float = 0.20
    I_threshold: float = 5.0e6
    # NOTE: ``newton_maxiter`` is retained for backward-compat but no longer
    # used: this example replaces the Newton polish in ``BoozerSurface.run_code``
    # with an LBFGS-only resolver (see ``_install_ls_no_newton``).
    newton_maxiter: int = 200
    bfgs_tol: float = 1e-8
    bfgs_maxiter: int = 500
    do_taylor_test: bool = False
    do_poincare: bool = True
    poincare_nlines: int = 8
    poincare_tmax: float = 8000.0
    poincare_tol: float = 1.0e-9
    interp_grid_n: int = 20
    interp_degree: int = 4
    plot_resolution_phi: int = 64
    plot_resolution_theta: int = 64


CASES: List[Case] = [
    Case(
        name="QA_LP_reactor_iota0p19",
        target="QA",
        plasma_input="input.LandremanPaul2021_QA_reactorScale_lowres",
        iota_target=0.19,
        stages=_LEGACY_STAGES_QA,
        bfgs_tol=1e-10,
        bfgs_maxiter=1500,
    ),
    Case(
        name="QA_LP_reactor_iota0p42",
        target="QA",
        plasma_input="input.LandremanPaul2021_QA_reactorScale_lowres",
        iota_target=0.42,
        stages=_LEGACY_STAGES_QA,
        bfgs_tol=1e-10,
        bfgs_maxiter=1500,
    ),
    Case(
        name="QH_LP_reactor_iota0p92",
        target="QH",
        plasma_input="input.LandremanPaul2021_QH_reactorScale_lowres",
        iota_target=0.92,
        stages=_LEGACY_STAGES_QH,
        bfgs_tol=1e-10,
        bfgs_maxiter=1500,
    ),
    Case(
        name="QH_LP_reactor_iota1p10",
        target="QH",
        plasma_input="input.LandremanPaul2021_QH_reactorScale_lowres",
        iota_target=1.10,
        stages=_LEGACY_STAGES_QH,
        bfgs_tol=1e-10,
        bfgs_maxiter=1500,
    ),
    # Yu/Zhu et al. 2026 (arxiv:2605.03599) style: 24x12 toroidal x poloidal DF grid
    # on a thin conformal shell, no pruning (288 base coils per half-period).
    Case(
        name="QA_LP_reactor_DF_24x12",
        target="QA",
        plasma_input="input.LandremanPaul2021_QA_reactorScale_lowres",
        iota_target=0.5,
        iota_guess=0.25,
        aspect_target=6.0,
        weights=dict(
            W_SYMM=1.0,
            W_IOTA=1.0,
            W_MR=10.0,
            W_VOL=1.0,
            W_BR=1.0e4,
            W_AR=1.0,
            W_IMAX_FRAC=1.0e-3,
            CONSTRAINT_WEIGHT=1.0e4,
        ),
        n_dipole_phi=24,
        n_dipole_theta=12,
        n_dipole_radial=1,
        grid_layout="shell_conformal",
        prune_inboard=False,
        prune_interlinking=False,
        plasma_poff=0.4,
        plasma_coff=0.4,
        df_planar_order=2,
        dipole_current_init_amp=1.0e-3,
        stages=DEFAULT_STAGES_QA,
    ),
    Case(
        name="QH_LP_reactor_DF_24x12",
        target="QH",
        plasma_input="input.LandremanPaul2021_QH_reactorScale_lowres",
        iota_target=0.5,
        iota_guess=0.25,
        aspect_target=10.0,
        weights=dict(
            W_SYMM=1.0,
            W_IOTA=1.0,
            W_MR=10.0,
            W_VOL=1.0,
            W_BR=1.0e4,
            W_AR=1.0,
            W_IMAX_FRAC=1.0e-3,
            CONSTRAINT_WEIGHT=1.0e4,
        ),
        n_dipole_phi=24,
        n_dipole_theta=12,
        n_dipole_radial=1,
        grid_layout="shell_conformal",
        prune_inboard=False,
        prune_interlinking=False,
        plasma_poff=0.4,
        plasma_coff=0.4,
        df_planar_order=2,
        dipole_current_init_amp=1.0e-3,
        stages=DEFAULT_STAGES_QH,
    ),
]


def _dipole_current_seed(case: Case, ndipoles: int) -> np.ndarray:
    """Deterministic per-case base-coil current seeds in MA."""
    if case.dipole_current_init_amp > 0:
        seed = int.from_bytes(
            hashlib.blake2b(case.name.encode(), digest_size=4).digest(), "big"
        )
        rng = np.random.default_rng(seed)
        return rng.uniform(
            -float(case.dipole_current_init_amp),
            float(case.dipole_current_init_amp),
            ndipoles,
        )
    return np.full(ndipoles, float(case.dipole_current_init))


# ---------------------------------------------------------------------------
MU0 = 4.0 * np.pi * 1.0e-7


def case_for_ci(c: Case) -> Case:
    return replace(c, stages=CI_STAGES, do_poincare=False, do_taylor_test=False)


def aspect_bounds_for_case(case: Case) -> Tuple[float, float]:
    """Aspect-ratio band for :func:`validate_boozer_state`."""
    if case.aspect_target is not None and float(case.aspect_target) > 0:
        t = float(case.aspect_target)
        return (0.3 * t, 3.0 * t)
    return (1.5, 500.0)


def _normal_to_quaternion(n: np.ndarray) -> Tuple[float, float, float, float]:
    """Map a unit normal to ``CurvePlanarFourier`` quaternion DOFs (3-2-1 Euler)."""
    n = np.asarray(n, dtype=float).reshape(3)
    norm = float(np.linalg.norm(n))
    if norm < 1e-30:
        return 1.0, 0.0, 0.0, 0.0
    n = n / norm
    alpha = float(np.arcsin(np.clip(-n[1], -1.0, 1.0)))
    delta = float(np.arctan2(n[0], n[2]))
    a2, d2 = alpha / 2.0, delta / 2.0
    ca, sa, cd, sd = np.cos(a2), np.sin(a2), np.cos(d2), np.sin(d2)
    return float(ca * cd), float(sa * cd), float(ca * sd), float(-sa * sd)


def _build_shell_conformal_grid(
    input_path: str, case: Case, n_phi: int, n_theta: int
) -> List[Any]:
    """Place planar DF loops on a normal-offset shell at uniform (phi, theta)."""
    from simsopt.geo import CurvePlanarFourier

    s_shell = SurfaceRZFourier.from_vmec_input(
        input_path,
        range="half period",
        nphi=n_phi,
        ntheta=n_theta,
    )
    offset = float(case.plasma_poff) + 0.5 * float(case.plasma_coff)
    s_shell.extend_via_normal(offset)
    pts = s_shell.gamma().reshape(n_phi, n_theta, 3)
    nrm = s_shell.unitnormal().reshape(n_phi, n_theta, 3)

    dphi = (
        float(np.linalg.norm(pts[1, 0] - pts[0, 0]))
        if n_phi > 1
        else 1.0
    )
    dtheta = (
        float(np.linalg.norm(pts[0, 1] - pts[0, 0]))
        if n_theta > 1
        else 1.0
    )
    r_loop = float(case.df_loop_radius_factor) * min(dphi, dtheta)

    order = int(case.df_planar_order)
    nquad = max(40, (order + 1) * 40)
    curves = []
    for ip in range(n_phi):
        for it in range(n_theta):
            p = pts[ip, it]
            q0, qi, qj, qk = _normal_to_quaternion(nrm[ip, it])
            c = CurvePlanarFourier(nquad, order)
            dofs = np.zeros(2 * order + 8)
            dofs[0] = r_loop
            dofs[2 * order + 1] = q0
            dofs[2 * order + 2] = qi
            dofs[2 * order + 3] = qj
            dofs[2 * order + 4] = qk
            dofs[2 * order + 5 : 2 * order + 8] = p
            c.set_dofs(dofs)
            c.x = c.x
            curves.append(c)
    return curves


def make_plasma_surface(case: Case, nphi: int = 32, ntheta: int = 32):
    path = TEST_FILES_DIR / case.plasma_input
    if not path.is_file():
        raise FileNotFoundError(f"Plasma VMEC input not found: {path}")
    s = SurfaceRZFourier.from_vmec_input(
        str(path), range="half period", nphi=nphi, ntheta=ntheta
    )
    return s, str(path)


def make_circular_axis(R0: float, order: int = 10, npts: int = 128):
    axis = CurveXYZFourier(quadpoints=npts, order=order)
    dofs = np.zeros(axis.dof_size)
    names = axis.local_full_dof_names
    name_to_idx = {n: i for i, n in enumerate(names)}
    dofs[name_to_idx["xc(1)"]] = R0
    dofs[name_to_idx["ys(1)"]] = R0
    axis.x = dofs
    return axis


def _set_curve_xyz_fourier_from_rz(curve: CurveXYZFourier, phis, Rs, Zs):
    """Fit ``CurveXYZFourier`` DOFs from sampled ``(R, Z)`` over phi in [0, 2π).

    The samples are first transformed to ``(x, y, z) = (R cos phi, R sin phi, Z)``
    and then real Fourier coefficients are recovered by ordinary least squares
    against the basis ``{1, cos(2π k t), sin(2π k t)}_k`` with ``t = phi/(2π)``.
    """
    phis = np.asarray(phis, dtype=float)
    Rs = np.asarray(Rs, dtype=float)
    Zs = np.asarray(Zs, dtype=float)
    xs = Rs * np.cos(phis)
    ys = Rs * np.sin(phis)
    zs = Zs

    order = curve.order
    t = phis / (2.0 * np.pi)
    cols = [np.ones_like(t)]
    for k in range(1, order + 1):
        cols.append(np.cos(2.0 * np.pi * k * t))
        cols.append(np.sin(2.0 * np.pi * k * t))
    Bmat = np.stack(cols, axis=1)
    cx, _, _, _ = np.linalg.lstsq(Bmat, xs, rcond=None)
    cy, _, _, _ = np.linalg.lstsq(Bmat, ys, rcond=None)
    cz, _, _, _ = np.linalg.lstsq(Bmat, zs, rcond=None)

    names = curve.local_full_dof_names
    name_to_idx = {n: i for i, n in enumerate(names)}
    dofs = np.zeros(curve.dof_size)
    dofs[name_to_idx["xc(0)"]] = cx[0]
    dofs[name_to_idx["yc(0)"]] = cy[0]
    dofs[name_to_idx["zc(0)"]] = cz[0]
    for k in range(1, order + 1):
        dofs[name_to_idx[f"xc({k})"]] = cx[2 * k - 1]
        dofs[name_to_idx[f"xs({k})"]] = cx[2 * k]
        dofs[name_to_idx[f"yc({k})"]] = cy[2 * k - 1]
        dofs[name_to_idx[f"ys({k})"]] = cy[2 * k]
        dofs[name_to_idx[f"zc({k})"]] = cz[2 * k - 1]
        dofs[name_to_idx[f"zs({k})"]] = cz[2 * k]
    curve.x = dofs


def find_magnetic_axis(
    btot: BiotSavart,
    R0_guess: float,
    nfp: int,
    Z0_guess: float = 0.0,
    order: int = 10,
    npts: int = 128,
    n_iter: int = 50,
    tol: float = 1e-5,
    relax: float = 1.0,
    verbose: bool = False,
) -> CurveXYZFourier:
    """Picard fixed-point search for the magnetic axis of ``btot``.

    Integrates the phi-parameterised field-line ODE

    .. math::

        \\frac{dR}{d\\varphi} = \\frac{R\\,B_R}{B_\\varphi},
        \\qquad
        \\frac{dZ}{d\\varphi} = \\frac{R\\,B_Z}{B_\\varphi}

    over one toroidal period ``[0, 2π]``, then applies a return-map Picard
    update: the next guess is the endpoint ``(R,Z)(2π)`` (optionally
    under-relaxed with ``relax`` in ``(0, 1]``).  Convergence is when the
    return-map residual ``|R(2π)-R(0)|+|Z(2π)-Z(0)|`` is below ``tol``.
    Once converged, the final one-period trace is Fourier-fit into a
    ``CurveXYZFourier`` with the given ``order``.  If Picard fails to
    converge in ``n_iter`` steps the function falls back to
    ``make_circular_axis(R0_guess)`` and prints a warning.

    ``nfp`` is reserved for future field-period-aware tracing; integration
    is always over a full torus ``[0, 2π]``.
    """
    del nfp  # unused; full-torus tracing only

    def field_rhs(phi, y):
        R, Z = float(y[0]), float(y[1])
        pt = np.array([[R * np.cos(phi), R * np.sin(phi), Z]])
        btot.set_points(pt)
        B = np.asarray(btot.B()).flatten()
        Bx, By, Bz = B[0], B[1], B[2]
        Bphi = -np.sin(phi) * Bx + np.cos(phi) * By
        BR = np.cos(phi) * Bx + np.sin(phi) * By
        if abs(Bphi) < 1e-300:
            raise RuntimeError("Bphi vanished while integrating axis")
        return [R * BR / Bphi, R * Bz / Bphi]

    scale = max(abs(R0_guess), 1.0)
    R, Z = float(R0_guess), float(Z0_guess)
    ts = np.linspace(0.0, 2.0 * np.pi, 4 * npts, endpoint=False)
    converged = False
    last_residual = None
    for k in range(n_iter):
        try:
            sol = solve_ivp(
                field_rhs,
                [0.0, 2.0 * np.pi],
                [R, Z],
                rtol=1e-9,
                atol=1e-12,
                t_eval=ts,
                method="RK45",
            )
        except Exception as exc:
            print(f"   find_magnetic_axis: integration failed ({exc}); fallback")
            return make_circular_axis(R0_guess, order=order, npts=npts)
        if not sol.success:
            print(
                "   find_magnetic_axis: solver did not converge; using "
                "fallback circular axis"
            )
            return make_circular_axis(R0_guess, order=order, npts=npts)
        R_end = float(sol.y[0, -1])
        Z_end = float(sol.y[1, -1])
        residual = abs(R_end - R) + abs(Z_end - Z)
        last_residual = residual
        if verbose:
            print(
                f"   find_magnetic_axis: iter {k + 1}/{n_iter} "
                f"R={R:.6f} Z={Z:.6f} residual={residual:.3e}"
            )
        if residual < tol * scale:
            R = (1.0 - relax) * R + relax * R_end
            Z = (1.0 - relax) * Z + relax * Z_end
            converged = True
            break
        R = (1.0 - relax) * R + relax * R_end
        Z = (1.0 - relax) * Z + relax * Z_end
    if not converged:
        res_msg = (
            f" (last residual={last_residual:.3e})"
            if last_residual is not None
            else ""
        )
        print(
            f"   find_magnetic_axis: Picard did not converge in {n_iter} "
            f"steps{res_msg}; falling back to circular axis"
        )
        return make_circular_axis(R0_guess, order=order, npts=npts)

    final_ts = np.linspace(0.0, 2.0 * np.pi, npts, endpoint=False)
    try:
        sol = solve_ivp(
            field_rhs,
            [0.0, 2.0 * np.pi],
            [R, Z],
            rtol=1e-10,
            atol=1e-13,
            t_eval=final_ts,
            method="RK45",
        )
    except Exception as exc:
        print(f"   find_magnetic_axis: final trace failed ({exc}); fallback")
        return make_circular_axis(R0_guess, order=order, npts=npts)
    if not sol.success:
        return make_circular_axis(R0_guess, order=order, npts=npts)

    axis = CurveXYZFourier(quadpoints=npts, order=order)
    _set_curve_xyz_fourier_from_rz(axis, sol.t, sol.y[0], sol.y[1])
    return axis


def build_initial_coils(case: Case) -> Dict[str, Any]:
    s, input_path = make_plasma_surface(case)
    nfp_eff = case.nfp_override if case.nfp_override is not None else s.nfp
    stellsym_eff = (
        case.stellsym_override
        if case.stellsym_override is not None
        else s.stellsym
    )
    if case.stellsym_override is True and not s.stellsym:
        raise ValueError("Cannot impose stellsym on a non-stellsym plasma input")

    config = "LandremanPaulQH" if case.target == "QH" else "LandremanPaulQA"
    reg_tf = regularization_rect(case.tf_reg_a, case.tf_reg_b)
    base_curves_TF, curves_TF, coils_TF, base_currents_TF = initialize_coils(
        s, config, reg_tf
    )

    s_inner = SurfaceRZFourier.from_vmec_input(
        input_path,
        range="half period",
        nphi=len(s.quadpoints_phi) * 4,
        ntheta=len(s.quadpoints_theta) * 4,
    )
    s_outer = SurfaceRZFourier.from_vmec_input(
        input_path,
        range="half period",
        nphi=len(s.quadpoints_phi) * 4,
        ntheta=len(s.quadpoints_theta) * 4,
    )
    n_phi = case.n_dipole_phi if case.n_dipole_phi is not None else case.n_dipole_per_axis
    n_theta = (
        case.n_dipole_theta if case.n_dipole_theta is not None else case.n_dipole_per_axis
    )
    n_radial = (
        case.n_dipole_radial if case.n_dipole_radial is not None else case.n_dipole_per_axis
    )
    if case.grid_layout == "shell_conformal":
        base_wp_curves = _build_shell_conformal_grid(
            input_path, case, n_phi, n_theta
        )
    else:
        s_inner.extend_via_normal(case.plasma_poff)
        s_outer.extend_via_normal(case.plasma_poff + case.plasma_coff)
        base_wp_curves, _ = create_planar_curves_between_two_toroidal_surfaces(
            s,
            s_inner,
            s_outer,
            n_phi,
            n_theta,
            n_radial,
            order=case.df_planar_order,
        )
        if case.prune_inboard:
            base_wp_curves = remove_inboard_dipoles(s, base_wp_curves)
        if case.prune_interlinking:
            base_wp_curves = remove_interlinking_dipoles_and_TFs(
                base_wp_curves, base_curves_TF
            )
        alphas, deltas = align_dipoles_with_plasma(s, base_wp_curves)
        for i, c in enumerate(base_wp_curves):
            a2, d2 = alphas[i] / 2.0, deltas[i] / 2.0
            ca, sa, cd, sd = np.cos(a2), np.sin(a2), np.cos(d2), np.sin(d2)
            c.set("q0", ca * cd)
            c.set("qi", sa * cd)
            c.set("qj", ca * sd)
            c.set("qk", -sa * sd)

    ndipoles = len(base_wp_curves)
    print(f"   --> {ndipoles} dipole base coils after pruning")

    base_wp_currents = [
        Current(float(s)) * 1e6 for s in _dipole_current_seed(case, ndipoles)
    ]
    reg_df = regularization_rect(0.1, 0.1)
    wp_coils = coils_via_symmetries(
        base_wp_curves,
        base_wp_currents,
        nfp_eff,
        stellsym_eff,
        regularizations=[reg_df] * ndipoles,
    )
    all_coils = wp_coils + coils_TF
    axis = make_circular_axis(s.get_rc(0, 0))

    return {
        "plasma": s,
        "input_path": input_path,
        "axis": axis,
        "nfp": nfp_eff,
        "stellsym": stellsym_eff,
        "base_curves_TF": base_curves_TF,
        "coils_TF": coils_TF,
        "base_currents_TF": base_currents_TF,
        "base_wp_curves": base_wp_curves,
        "base_wp_currents": base_wp_currents,
        "wp_coils": wp_coils,
        "all_coils": all_coils,
    }


def freeze_geometry_free_currents(setup: Dict[str, Any]) -> None:
    for c in setup["base_curves_TF"]:
        c.fix_all()
    for c in setup["base_wp_curves"]:
        c.fix_all()
    for c in setup["base_currents_TF"][:-1]:
        c.unfix_all()
    setup["base_currents_TF"][-1].fix_all()
    for c in setup["base_wp_currents"]:
        c.unfix_all()


def G_from_TF(coils_TF) -> float:
    return MU0 * sum(abs(c.current.get_value()) for c in coils_TF)


def validate_boozer_state(
    bsurf: "BoozerSurface",
    G0: float,
    aspect_bounds: Tuple[float, float] = (1.5, 500.0),
) -> Tuple[bool, str]:
    """Multi-criterion gate for a Boozer LS result.

    The BoozerLS residual is scale-invariant in ``|B|`` when
    ``weight_inv_modB=True``: a "successful" optimizer can still leave the
    surface in the degenerate ``|B|→∞, G→0, iota→∞`` fixed point.  Note we
    deliberately do **not** require scipy's ``res['success']`` flag -- with
    the tight ``bfgs_tol`` we use, L-BFGS-B routinely terminates with
    ``success=False`` (hit maxiter or unable to drive gradient below
    ``gtol``) on a perfectly usable Boozer surface.  Instead the gate
    checks the physical state directly:

    1. ``iota`` is finite and ``|iota| <= 5``.
    2. ``G`` is finite and (if ``G0 > 0``) within ``[0.3 G0, 3.0 G0]``.
    3. ``Volume`` is finite and strictly positive.
    4. The surface is not self-intersecting at ``phi=0`` and
       ``phi=π/nfp`` (falls back to an aspect-ratio heuristic if the
       ``bentley_ottmann`` optional dependency is missing).

    If the result dict contains an explicit ``"exception"`` key (set by
    ``_install_ls_no_newton`` when an LBFGS attempt raised), validation
    fails immediately.
    """
    res = getattr(bsurf, "res", None) or {}
    s = bsurf.surface

    if "exception" in res:
        return False, f"solver exception: {res['exception']}"
    iota = res.get("iota")
    G = res.get("G")
    if iota is None or not np.isfinite(iota) or abs(iota) > 5.0:
        return False, f"iota={iota!r}"
    if G is None or not np.isfinite(G):
        return False, f"G={G!r}"
    if G0 is not None and G0 > 0 and not (0.3 * G0 <= abs(G) <= 3.0 * G0):
        return False, f"|G|/G0={abs(G) / G0:.2e}"

    try:
        vol = float(s.volume())
    except Exception as exc:
        return False, f"volume eval failed: {exc!s}"
    # SurfaceXYZTensorFourier built via ``fit_to_curve(..., flip_theta=True)``
    # has a negative signed volume by construction; the constraint pins it
    # to a (negative) ``vol_target``.  Reject only non-finite / zero volume,
    # not negative volume.
    if not np.isfinite(vol) or abs(vol) < 1e-30:
        return False, f"vol={vol:.3e}"

    try:
        nfp = max(1, int(s.nfp))
        si0 = bool(s.is_self_intersecting(angle=0.0))
        si1 = bool(s.is_self_intersecting(angle=float(np.pi / nfp)))
        if si0 or si1:
            return False, "self-intersecting"
    except Exception:
        # ``is_self_intersecting`` requires the optional ``bentley_ottmann``
        # package; fall back to a cheap aspect-ratio sanity check.  We use a
        # generous upper bound because thin continuation-seed surfaces can
        # legitimately have aspect ~100+ before the inflate ramp completes.
        try:
            ar = float(s.aspect_ratio())
        except Exception as exc:
            return False, f"aspect eval failed: {exc!s}"
        lo, hi = aspect_bounds
        if not np.isfinite(ar) or ar < lo or ar > hi:
            return False, f"aspect={ar:.2f}"
    return True, "ok"


def _populate_PLU_at_current_state(
    bsurf: "BoozerSurface", iota: float, G: Optional[float],
    constraint_weight: float, weight_inv_modB: bool,
) -> None:
    """Populate ``bsurf.res['PLU']`` and ``bsurf.res['vjp']`` after LBFGS.

    The simsopt adjoint gradient pipeline (``BoozerResidual.dJ``,
    ``MajorRadius.dJ``, ``Iotas.dJ``, ``NonQuasiSymmetricRatio.dJ``) reads
    ``boozer_surface.res['PLU']`` (LU factorisation of the Hessian) and
    ``res['vjp']``.  The Newton solver populates these by construction;
    the LBFGS path does not.  This helper computes them in **one** call to
    ``boozer_penalty_constraints_vectorized(..., derivatives=2)`` -- no
    Newton iterations -- so the LBFGS-only state still drives the outer
    adjoint without the Newton polish that corrupts the LS minimum.
    """
    s = bsurf.surface
    if G is None:
        x = np.concatenate((s.get_dofs(), [iota]))
        optimize_G = False
    else:
        x = np.concatenate((s.get_dofs(), [iota, G]))
        optimize_G = True
    try:
        _val, dval, d2val = bsurf.boozer_penalty_constraints_vectorized(
            x,
            derivatives=2,
            constraint_weight=constraint_weight,
            optimize_G=optimize_G,
            weight_inv_modB=weight_inv_modB,
        )
    except Exception as exc:
        bsurf.res["PLU_error"] = str(exc)
        return
    try:
        P, L, U = _scipy_lu(d2val)
    except Exception as exc:
        bsurf.res["PLU_error"] = f"LU failed: {exc!s}"
        return
    bsurf.res["PLU"] = (P, L, U)
    bsurf.res["vjp"] = partial(
        boozer_surface_dlsqgrad_dcoils_vjp, weight_inv_modB=weight_inv_modB
    )
    bsurf.res["jacobian"] = dval
    bsurf.res["residual"] = dval
    bsurf.res["hessian"] = d2val
    # Downstream consumers (NonQuasiSymmetricRatio.compute,
    # BoozerResidual.compute) require these keys.
    bsurf.res.setdefault("weight_inv_modB", weight_inv_modB)
    bsurf.res["weight_inv_modB"] = weight_inv_modB
    bsurf.res.setdefault("type", "ls")


def _install_ls_no_newton(
    bsurf: "BoozerSurface",
    G0_ref: float,
    aspect_bounds: Tuple[float, float] = (1.5, 500.0),
) -> "BoozerSurface":
    """Replace ``bsurf.run_code`` with an LBFGS-only LS resolver.

    The default ``BoozerSurface.run_code`` (LS branch) chains
    ``minimize_boozer_penalty_constraints_LBFGS`` with a Newton polish that
    frequently slides the surface into the degenerate ``|B|→∞`` fixed
    point.  This patched ``run_code``:

    * calls only ``minimize_boozer_penalty_constraints_LBFGS``;
    * ramps ``(weight_inv_modB, constraint_weight)`` over
      ``((True, base), (True, 1e5), (True, 1e6), (False, base), (False, 1e5), (False, 1e6))``;
    * after each successful LBFGS step, populates ``res['PLU']`` and
      ``res['vjp']`` via a single Hessian evaluation (no Newton
      iterations) so the adjoint gradient through ``BoozerResidual``,
      ``MajorRadius``, ``Iotas`` and ``NonQuasiSymmetricRatio`` works;
    * falls back to ``minimize_boozer_penalty_constraints_ls(method="lm")``
      (Levenberg-Marquardt, still Newton-free) once the ramp is exhausted;
    * validates each attempt via :func:`validate_boozer_state` and returns
      the first attempt that passes.

    The ``G0_ref`` argument is captured in a closure and used for the
    ``|G|/G0`` band check; it should be the TF-coil reference ``G0``.

    Also tracks ``bsurf._n_resolves`` (incremented on every call) for
    diagnostic accounting.
    """

    base_cw = float(bsurf.constraint_weight) if bsurf.constraint_weight else 1.0e4
    schedule = (
        (True, base_cw),
        (True, 1.0e5),
        (True, 1.0e6),
        (False, base_cw),
        (False, 1.0e5),
        (False, 1.0e6),
    )

    bsurf._target_G0 = float(G0_ref)
    bsurf._aspect_bounds = aspect_bounds
    bsurf._n_resolves = 0

    def _patched_run_code(iota, G=None):
        if not bsurf.need_to_run_code:
            return bsurf.res
        bsurf._n_resolves += 1
        G_val = G if G is not None else (bsurf.res or {}).get("G")
        if G_val is None or not np.isfinite(G_val):
            G_val = bsurf._target_G0
        last_res = None
        for inv_modB, cw in schedule:
            bsurf.options["weight_inv_modB"] = inv_modB
            bsurf.constraint_weight = cw
            bsurf.need_to_run_code = True
            try:
                res = bsurf.minimize_boozer_penalty_constraints_LBFGS(
                    tol=bsurf.options.get("bfgs_tol", 1e-10),
                    maxiter=bsurf.options.get("bfgs_maxiter", 1500),
                    constraint_weight=cw,
                    iota=iota,
                    G=G_val,
                    limited_memory=bool(bsurf.options.get("limited_memory", False)),
                    weight_inv_modB=inv_modB,
                    verbose=bool(bsurf.options.get("verbose", False)),
                )
            except Exception as exc:
                last_res = {"success": False, "exception": str(exc)}
                continue
            last_res = res
            _populate_PLU_at_current_state(
                bsurf,
                iota=float(res.get("iota", iota)),
                G=res.get("G", G_val),
                constraint_weight=cw,
                weight_inv_modB=inv_modB,
            )
            ok, _ = validate_boozer_state(
                bsurf,
                bsurf._target_G0,
                aspect_bounds=getattr(bsurf, "_aspect_bounds", (1.5, 500.0)),
            )
            if ok:
                return res
        # Final fallback: scipy.optimize.least_squares "lm" (no Newton).
        try:
            bsurf.options["weight_inv_modB"] = True
            bsurf.constraint_weight = base_cw
            bsurf.need_to_run_code = True
            res = bsurf.minimize_boozer_penalty_constraints_ls(
                tol=1e-12,
                maxiter=200,
                constraint_weight=base_cw,
                iota=iota,
                G=G_val,
                method="lm",
                weight_inv_modB=True,
            )
            _populate_PLU_at_current_state(
                bsurf,
                iota=float(res.get("iota", iota)) if res else iota,
                G=res.get("G", G_val) if res else G_val,
                constraint_weight=base_cw,
                weight_inv_modB=True,
            )
            return res
        except Exception:
            return last_res if last_res is not None else (bsurf.res or {})

    bsurf.run_code = _patched_run_code  # type: ignore[method-assign]
    return bsurf


def _solve_ls_no_newton(
    bsurf: "BoozerSurface", iota: float, G0: float
) -> Tuple[Dict[str, Any], float]:
    """Resolve ``bsurf`` via the LBFGS-only path and return ``(res, cw_used)``.

    Assumes ``_install_ls_no_newton(bsurf, G0)`` was already called (so
    ``bsurf.run_code`` is the patched LBFGS-only version).  Forces
    ``need_to_run_code=True`` so the patched ``run_code`` does a real solve.
    """
    bsurf.need_to_run_code = True
    res = bsurf.run_code(iota, G=G0) or {}
    cw_used = float(bsurf.constraint_weight) if bsurf.constraint_weight else float("nan")
    return res, cw_used


def make_boozer_surface(
    btot,
    ma,
    edge_minor_radius: float,
    nfp: int,
    mpol: int,
    ntor: int,
    case: Case,
    stellsym: bool = True,
) -> BoozerSurface:
    phis = np.linspace(0, 1 / nfp, 2 * ntor + 1, endpoint=False)
    thetas = np.linspace(0, 1.0, 2 * mpol + 1, endpoint=False)
    s = SurfaceXYZTensorFourier(
        mpol=mpol,
        ntor=ntor,
        stellsym=stellsym,
        nfp=nfp,
        quadpoints_phi=phis,
        quadpoints_theta=thetas,
    )
    s.fit_to_curve(ma, edge_minor_radius, flip_theta=True)
    vol_label = Volume(s)
    vol_target = vol_label.J()
    cw = (
        None
        if case.boozer_type == "exact"
        else float(case.weights.get("CONSTRAINT_WEIGHT", 100.0))
    )
    bsurf = BoozerSurface(btot, s, vol_label, vol_target, constraint_weight=cw)
    # Newton polish has been removed (see ``_install_ls_no_newton``); only
    # the LBFGS-related options are still meaningful.
    bsurf.options.update(
        {
            "verbose": False,
            "bfgs_tol": case.bfgs_tol,
            "bfgs_maxiter": case.bfgs_maxiter,
            "weight_inv_modB": True,
            "limited_memory": True,
        }
    )
    return bsurf


def _seed_boozer_continuation(
    btot: BiotSavart,
    axis: CurveXYZFourier,
    plasma_minor: float,
    nfp: int,
    mpol0: int,
    ntor0: int,
    case: Case,
    G0: float,
    stellsym: bool = True,
    aspect_bounds: Tuple[float, float] = (1.5, 500.0),
    a_edge_final: Optional[float] = None,
) -> Tuple[BoozerSurface, Dict[str, Any]]:
    """Build the initial Boozer surface by inflating from a thin seed.

    Solves the BoozerLS problem at progressively larger edge minor radii
    ``a = {0.05, 0.10, 0.20, 1.0} * a_edge_final`` (or, when
    ``a_edge_final`` is None, ``frac * plasma_minor`` with
    ``frac in {0.05, 0.10, 0.20, case.a_edge_frac}``), refitting the
    surface to the axis at each step and warm-starting ``iota``/``G`` from
    the previous solve.  Returns the bsurf populated with the deepest
    successfully validated state.
    """
    if a_edge_final is not None and a_edge_final > 0:
        a_max = float(a_edge_final)
        fracs = sorted({0.05, 0.10, 0.20, 1.0})
        a_edges = [f * a_max for f in fracs]
    else:
        fracs = sorted({0.05, 0.10, 0.20, float(case.a_edge_frac)})
        fracs = [f for f in fracs if 0 < f <= float(case.a_edge_frac)]
        if not fracs:
            fracs = [float(case.a_edge_frac)]
        a_edges = [f * plasma_minor for f in fracs]

    bsurf: Optional[BoozerSurface] = None
    last_res: Dict[str, Any] = {}
    iota_warm = float(case.iota_guess)
    G_warm = float(G0)
    last_good_dofs = None
    last_good_iota = iota_warm
    last_good_G = G_warm

    for i, (frac, a_edge) in enumerate(zip(fracs, a_edges)):
        if bsurf is None:
            bsurf = make_boozer_surface(
                btot, axis, a_edge, nfp, mpol0, ntor0, case, stellsym=stellsym
            )
            _install_ls_no_newton(bsurf, G0, aspect_bounds=aspect_bounds)
        else:
            # Refit existing tensor-Fourier surface to the axis at the new edge
            bsurf.surface.fit_to_curve(axis, a_edge, flip_theta=True)
            bsurf.targetlabel = float(bsurf.label.J())
        bsurf.need_to_run_code = True
        res, cw = _solve_ls_no_newton(bsurf, iota_warm, G_warm)
        ok, why = validate_boozer_state(bsurf, G0, aspect_bounds=aspect_bounds)
        print(
            f"   Seed step {i} a_edge={a_edge:.4f} (frac={frac:.2f}): "
            f"success={res.get('success', False)} iter={res.get('iter', -1)} "
            f"iota={res.get('iota', float('nan')):.4e} "
            f"G={res.get('G', float('nan')):.4e} cw={cw:.2e} validate={why}"
        )
        last_res = res or {}
        if ok:
            iota_warm = float(res["iota"])
            G_warm = float(res["G"])
            last_good_dofs = bsurf.surface.x.copy()
            last_good_iota = iota_warm
            last_good_G = G_warm
        else:
            if last_good_dofs is not None:
                # Roll back to the last validated step; do not inflate further.
                bsurf.surface.x = last_good_dofs
                bsurf.res = {
                    **(bsurf.res or {}),
                    "iota": last_good_iota,
                    "G": last_good_G,
                    "success": True,
                }
                print(
                    f"   Seed step {i} failed validation; rolling back to "
                    f"last good (frac={fracs[i - 1]:.2f}, "
                    f"iota={last_good_iota:.4e})"
                )
            break

    assert bsurf is not None
    return bsurf, last_res


def upgrade_boozer_surface(boozer_surface_old: BoozerSurface, mpol_new: int, ntor_new: int):
    s_old = boozer_surface_old.surface
    nfp = s_old.nfp
    stellsym = s_old.stellsym

    phis_new = np.linspace(0, 1 / nfp, 2 * ntor_new + 1, endpoint=False)
    thetas_new = np.linspace(0, 1.0, 2 * mpol_new + 1, endpoint=False)
    s_new = SurfaceXYZTensorFourier(
        mpol=mpol_new,
        ntor=ntor_new,
        stellsym=stellsym,
        nfp=nfp,
        quadpoints_phi=phis_new,
        quadpoints_theta=thetas_new,
    )

    target = np.zeros((len(phis_new), len(thetas_new), 3))
    for i, phi in enumerate(phis_new):
        target[i, :, :] = s_old.cross_section(phi, thetas=thetas_new)
    s_new.least_squares_fit(target)

    label_new = Volume(s_new)
    bsurf_new = BoozerSurface(
        boozer_surface_old.biotsavart,
        s_new,
        label_new,
        label_new.J(),
        constraint_weight=boozer_surface_old.constraint_weight,
        options=dict(boozer_surface_old.options),
    )
    return bsurf_new


class CurrentMagnitudeSquared(Optimizable):
    """Returns ``I**2`` for use inside ``QuadraticPenalty(..., 'max')``."""

    def __init__(self, current):
        super().__init__(depends_on=[current])
        self.current = current

    def J(self):
        return float(self.current.get_value()) ** 2

    @derivative_dec
    def dJ(self):
        I = float(self.current.get_value())
        return self.current.vjp(np.array([2.0 * I]))


def build_qs_term(boozer_surface, bs_eval, sdim: int, case: Case, plasma_nfp: int):
    if case.target == "QA":
        return NonQuasiSymmetricRatio(boozer_surface, bs_eval, sDIM=sdim)
    if case.target == "QH":
        hn = case.helicity_N if case.helicity_N is not None else plasma_nfp
        return NonQuasiSymmetricRatioHelical(
            boozer_surface,
            bs_eval,
            helicity_M=case.helicity_M,
            helicity_N=hn,
            sDIM=sdim,
        )
    raise ValueError(f"Unknown target {case.target!r}")


def build_objective(
    boozer_surface,
    bs_eval,
    base_wp_currents,
    sdim: int,
    case: Case,
    plasma_nfp: int,
    iota_target: float,
    r0_target: float,
    v_target: Optional[float] = None,
):
    J_nonqs = build_qs_term(boozer_surface, bs_eval, sdim, case, plasma_nfp)
    J_iota = QuadraticPenalty(Iotas(boozer_surface), iota_target, "identity")
    J_mr = QuadraticPenalty(MajorRadius(boozer_surface), r0_target, "identity")
    J_br = BoozerResidual(boozer_surface, bs_eval)
    if v_target is None:
        v_target = float(boozer_surface.targetlabel)
    J_vol = QuadraticPenalty(boozer_surface.label, v_target, "identity")
    w = case.weights
    w_imax_frac = float(w.get("W_IMAX_FRAC", 1e-3))
    w_imax = w_imax_frac / case.I_threshold**2
    J_imax = sum(
        QuadraticPenalty(
            CurrentMagnitudeSquared(c), case.I_threshold**2, "max"
        )
        for c in base_wp_currents
    )
    terms: Dict[str, Any] = dict(
        J_nonQS=J_nonqs,
        J_iota=J_iota,
        J_mr=J_mr,
        J_vol=J_vol,
        J_br=J_br,
        J_Imax=J_imax,
    )
    w_vol_eff = (
        0.0
        if case.aspect_target is not None
        else float(w.get("W_VOL", 1.0))
    )
    jf = (
        float(w.get("W_SYMM", 1.0)) * J_nonqs
        + float(w.get("W_IOTA", 1.0)) * J_iota
        + float(w.get("W_MR", 10.0)) * J_mr
        + w_vol_eff * J_vol
        + float(w.get("W_BR", 1.0e4)) * J_br
        + w_imax * J_imax
    )
    if case.aspect_target is not None and float(w.get("W_AR", 0.0)) > 0.0:
        J_ar = QuadraticPenalty(
            AspectRatio(boozer_surface.surface),
            float(case.aspect_target),
            "identity",
        )
        jf = jf + float(w.get("W_AR", 0.0)) * J_ar
        terms["J_ar"] = J_ar
    return jf, terms


def make_fun(
    JF,
    boozer_surface: BoozerSurface,
    state: Dict[str, Any],
    G0_ref: float,
    verbose: bool = False,
    aspect_bounds: Tuple[float, float] = (1.5, 500.0),
):
    """Build the outer L-BFGS-B objective callable with a robust revert gate.

    Mirrors the ``prevs``-revert pattern used in
    ``examples/2_Intermediate/boozerQA_ls_mpi.py``: on any failure of
    :func:`validate_boozer_state` (non-converged solve, ``|iota|>5``,
    ``|G|/G0`` outside band, non-positive volume, self-intersecting
    surface, non-finite ``J`` or ``grad``), revert the boozer surface
    DOFs / ``iota`` / ``G`` to the last accepted state and return
    ``J = max(10 prevs['J'], 1e6)`` with ``grad = -prevs['dJ']`` so the
    L-BFGS-B line search backs off.
    """
    prevs: Dict[str, Any] = {
        "sdofs": boozer_surface.surface.x.copy(),
        "iota": boozer_surface.res["iota"],
        "G": boozer_surface.res["G"],
    }

    def _commit_initial():
        prevs["J"] = float(JF.J())
        prevs["dJ"] = np.asarray(JF.dJ()).copy()

    _commit_initial()

    def fun(dofs):
        sdofs_prev = prevs["sdofs"].copy()
        iota_prev = prevs["iota"]
        g_prev = prevs["G"]

        JF.x = dofs
        try:
            j = float(JF.J())
            grad = np.asarray(JF.dJ()).copy()
        except Exception as exc:
            print(f"   make_fun: JF.J/dJ raised ({exc}); reverting")
            j = max(10.0 * prevs.get("J", 1.0), 1.0e6)
            grad = -prevs["dJ"]
            boozer_surface.surface.x = sdofs_prev
            boozer_surface.res["iota"] = iota_prev
            boozer_surface.res["G"] = g_prev
            return j, grad

        ok, why = validate_boozer_state(
            boozer_surface, G0_ref, aspect_bounds=aspect_bounds
        )
        finite = np.isfinite(j) and np.all(np.isfinite(grad))
        if not (ok and finite):
            if verbose:
                why_full = why if ok else f"not-finite J/grad (reason={why})"
                print(f"   make_fun: rejecting step ({why_full}); reverting")
            j = max(10.0 * prevs.get("J", 1.0), 1.0e6)
            grad = -prevs["dJ"]
            boozer_surface.surface.x = sdofs_prev
            boozer_surface.res["iota"] = iota_prev
            boozer_surface.res["G"] = g_prev
            return j, grad

        prevs["sdofs"] = boozer_surface.surface.x.copy()
        prevs["iota"] = boozer_surface.res["iota"]
        prevs["G"] = boozer_surface.res["G"]
        prevs["J"] = j
        prevs["dJ"] = grad.copy()
        state["sdofs"] = prevs["sdofs"]
        state["iota"] = prevs["iota"]
        state["G"] = prevs["G"]

        if verbose:
            print(
                f"   J={j:.3e}  iota={boozer_surface.res['iota']:.3e}  "
                f"||dJ||={np.linalg.norm(grad):.2e}"
            )
        return j, grad

    fun.prevs = prevs  # exposed for tests / diagnostics
    return fun


def taylor_test(fun_eval, JF, h_seed: int = 0, verbose: bool = True):
    np.random.seed(h_seed)
    dofs0 = JF.x.copy()
    hvec = np.random.randn(dofs0.size)
    hvec /= np.linalg.norm(hvec)
    _, dj0 = fun_eval(dofs0)
    djh = float(np.dot(dj0, hvec))
    if verbose:
        print(f"   Taylor (seed={h_seed}): dJ.h={djh:.3e}")
    for eps in [1e-3, 1e-4, 1e-5, 1e-6]:
        j1, _ = fun_eval(dofs0 + 2 * eps * hvec)
        j2, _ = fun_eval(dofs0 + eps * hvec)
        j3, _ = fun_eval(dofs0 - eps * hvec)
        j4, _ = fun_eval(dofs0 - 2 * eps * hvec)
        fd = (-j1 + 8 * j2 - 8 * j3 + j4) / (12 * eps)
        rel = (fd - djh) / max(abs(djh), 1e-30)
        if verbose:
            print(f"     eps={eps:.0e}  fd={fd:.3e}  rel-err={rel:+.2e}")
    JF.x = dofs0


def save_coils_vtk(all_coils, out_dir: Path, tag: str) -> None:
    coils_to_vtk(all_coils, str(out_dir / f"coils_{tag}"), close=True)


def save_plasma_surface_error_vtk(
    plasma_input_path: str,
    btot: BiotSavart,
    out_dir: Path,
    tag: str,
    qphi: int,
    qtheta: int,
) -> None:
    s_plot = SurfaceRZFourier.from_vmec_input(
        plasma_input_path,
        range="full torus",
        nphi=qphi,
        ntheta=qtheta,
    )
    g = s_plot.gamma().reshape((-1, 3))
    btot.set_points(g)
    B = btot.B().reshape((qphi, qtheta, 3))
    un = s_plot.unitnormal()
    bn = np.sum(B * un, axis=2)
    modb = np.linalg.norm(B, axis=2)
    ratio = bn / np.maximum(modb, 1e-30)
    extra = {
        "B_N": bn[:, :, None],
        "B_N_over_B": ratio[:, :, None],
        "modB": modb[:, :, None],
    }
    s_plot.to_vtk(str(out_dir / f"plasma_err_{tag}"), extra_data=extra)


def save_boozer_surface_vtk(boozer_surface: BoozerSurface, btot: BiotSavart, out_dir: Path, tag: str):
    surf = boozer_surface.surface
    g = surf.gamma()
    btot.set_points(g.reshape((-1, 3)))
    B = btot.B().reshape(g.shape[0], g.shape[1], 3)
    modb = np.linalg.norm(B, axis=-1)
    surf.to_vtk(str(out_dir / f"surf_{tag}"), extra_data={"modB": modb[:, :, None]})


def load_classifier_surface(plasma_input_path: str, nphi: int = 180, ntheta: int = 40):
    return SurfaceRZFourier.from_vmec_input(
        plasma_input_path, range="full torus", nphi=nphi, ntheta=ntheta
    )


def save_poincare(
    btot: BiotSavart,
    boozer_surface: BoozerSurface,
    plasma_input_path: str,
    out_dir: Path,
    tag: str,
    case: Case,
):
    surf_cls = load_classifier_surface(plasma_input_path)
    surf_plot = SurfaceRZFourier.from_vmec_input(
        plasma_input_path, range="half period", nphi=96, ntheta=48
    )
    nfp = surf_cls.nfp
    sc = SurfaceClassifier(surf_cls, h=0.03, p=2)

    # Wrap ``btot`` in an InterpolatedField so each fieldline step costs O(1)
    # instead of O(coils) — speedup is 1-2 orders of magnitude in practice
    # (mirrors ``examples/1_Simple/tracing_fieldlines_QA.py``).
    rs_arr = np.linalg.norm(surf_cls.gamma()[:, :, 0:2], axis=2)
    zs_arr = surf_cls.gamma()[:, :, 2]
    n = case.interp_grid_n
    rrange = (float(rs_arr.min()), float(rs_arr.max()), n)
    phirange = (0.0, 2 * np.pi / nfp, 2 * n)
    zrange = (0.0, float(zs_arr.max()), max(n // 2, 4))

    def skip(rs_, phis_, zs_):
        rphiz = np.asarray([rs_, phis_, zs_]).T.copy()
        return list((sc.evaluate_rphiz(rphiz) < -0.05).flatten())

    bsh = InterpolatedField(
        btot,
        case.interp_degree,
        rrange,
        phirange,
        zrange,
        True,
        nfp=nfp,
        stellsym=True,
        skip=skip,
    )

    xyz = boozer_surface.surface.gamma()[0, :, :]
    R = np.sqrt(xyz[:, 0] ** 2 + xyz[:, 1] ** 2)
    Z = xyz[:, 2]
    nt = len(R)
    idx = np.linspace(0, nt - 1, case.poincare_nlines, dtype=int)
    R0 = R[idx].tolist()
    Z0 = Z[idx].tolist()

    phis = [(i / 4.0) * (2 * np.pi / nfp) for i in range(4)]
    res_tys, res_phi_hits = compute_fieldlines(
        bsh,
        R0,
        Z0,
        tmax=case.poincare_tmax,
        tol=case.poincare_tol,
        phis=phis,
        stopping_criteria=[LevelsetStoppingCriterion(sc.dist)],
    )
    particles_to_vtk(res_tys, str(out_dir / f"fieldlines_{tag}"))
    import matplotlib

    matplotlib.use("Agg", force=True)
    png_path = str(out_dir / f"poincare_{tag}.png")
    try:
        plot_poincare_data(
            res_phi_hits,
            phis,
            png_path,
            dpi=200,
            surf=surf_plot,
        )
    except Exception as exc:
        print(
            f"   WARNING: Poincaré plot with plasma outline failed ({exc}); "
            "replotting scatter only."
        )
        plot_poincare_data(
            res_phi_hits,
            phis,
            png_path,
            dpi=200,
            surf=None,
        )


def plasma_bn_rms(plasma_half: SurfaceRZFourier, btot: BiotSavart) -> float:
    g = plasma_half.gamma().reshape((-1, 3))
    btot.set_points(g)
    B = btot.B().reshape(plasma_half.gamma().shape)
    un = plasma_half.unitnormal()
    bn = np.sum(B * un, axis=2)
    modb = np.linalg.norm(B, axis=2)
    ratio = bn / np.maximum(modb, 1e-30)
    return float(np.sqrt(np.mean(ratio**2)))


def compute_final_metrics(
    boozer_surface: BoozerSurface,
    J_nonqs: NonQuasiSymmetricRatio | NonQuasiSymmetricRatioHelical,
    setup: Dict[str, Any],
    case: Case,
    btot: BiotSavart,
    plasma_half: SurfaceRZFourier,
    last_opt_result,
    nit_total: int,
    G0_ref: float = 0.0,
    aspect_bounds: Tuple[float, float] = (1.5, 500.0),
    aspect_bounds_nominal: Optional[Tuple[float, float]] = None,
) -> Dict[str, Any]:
    f_symm = float(J_nonqs.J())
    bsurf = boozer_surface.surface
    mr = MajorRadius(boozer_surface).J()

    ids = [abs(c.get_value()) for c in setup["base_wp_currents"]]

    try:
        si0 = bool(bsurf.is_self_intersecting(angle=0.0))
        nfp_eff = max(1, int(bsurf.nfp))
        si1 = bool(bsurf.is_self_intersecting(angle=float(np.pi / nfp_eff)))
        self_intersecting_final = bool(si0 or si1)
    except Exception:
        self_intersecting_final = False

    try:
        boozer_residual_final = float(BoozerResidual(boozer_surface, btot).J())
    except Exception:
        boozer_residual_final = float("nan")

    ok_final, why_final = validate_boozer_state(
        boozer_surface, G0_ref, aspect_bounds=aspect_bounds
    )

    ar_final = float(bsurf.aspect_ratio())
    metrics = {
        "f_symm": f_symm,
        "quasisymmetry_residual": float(np.sqrt(max(f_symm, 0.0))),
        # Boozer-surface rotational transform (same scalar as Iotas.J()); not VMEC edge ι.
        "iota_boozer": boozer_surface.res.get("iota"),
        "iota_edge": boozer_surface.res.get("iota"),
        "iota_target": float(case.iota_target),
        "iota_residual": (
            float(boozer_surface.res.get("iota", 0.0) - case.iota_target)
            if boozer_surface.res.get("iota") is not None
            else None
        ),
        "G": boozer_surface.res.get("G"),
        "R0": float(mr),
        "minor_a": float(bsurf.minor_radius()),
        "aspect_ratio": ar_final,
        "aspect_target": (
            float(case.aspect_target) if case.aspect_target is not None else None
        ),
        "aspect_residual": (
            float(ar_final - case.aspect_target)
            if case.aspect_target is not None
            else None
        ),
        "nfp_effective": int(setup["nfp"]),
        "stellsym_effective": bool(setup.get("stellsym", True)),
        "n_base_DF_coils": int(len(setup["base_wp_curves"])),
        "n_symmetrized_DF_coils": int(len(setup["wp_coils"])),
        "grid_layout": case.grid_layout,
        "aspect_bounds_lo": (
            float(aspect_bounds_nominal[0])
            if aspect_bounds_nominal is not None
            else None
        ),
        "aspect_bounds_hi": (
            float(aspect_bounds_nominal[1])
            if aspect_bounds_nominal is not None
            else None
        ),
        "volume": float(bsurf.volume()),
        "max_abs_I_DF": float(max(ids)) if ids else 0.0,
        "mean_abs_I_DF": float(np.mean(ids)) if ids else 0.0,
        "sum_abs_I_DF": float(sum(ids)) if ids else 0.0,
        "rms_I_DF": float(np.sqrt(np.mean(np.square(ids)))) if ids else 0.0,
        "sum_abs_I_TF": float(
            sum(abs(c.current.get_value()) for c in setup["coils_TF"])
        ),
        "Bn_RMS_plasma": plasma_bn_rms(plasma_half, btot),
        "boozer_success_final": bool(boozer_surface.res.get("success", False)),
        "boozer_iter_final": int(boozer_surface.res.get("iter", -1)),
        "boozer_residual_final": boozer_residual_final,
        "boozer_validate_final": str(why_final),
        "boozer_state_ok_final": bool(ok_final),
        "self_intersecting_final": self_intersecting_final,
        "n_boozer_resolves_total": int(getattr(boozer_surface, "_n_resolves", -1)),
        "optimizer_success": bool(last_opt_result.success),
        "J_final": float(last_opt_result.fun),
        "nit_total": int(nit_total),
    }
    return metrics


def save_stage_outputs(
    out_dir: Path,
    stage_idx: int,
    boozer_surface: BoozerSurface,
    all_coils,
    btot: BiotSavart,
    res,
    per_stage_json: bool = False,
):
    """Per-stage outputs.

    Coil geometry is frozen so ``coils_<tag>.vtu`` is **not** rewritten here
    (use ``coils_initial.vtu`` / ``coils_optimized.vtu`` instead). The Boozer
    surface VTK (``surf_<tag>.vts``) and a tiny optimizer-result text file are
    cheap and always written. The big ``stage<k>.json`` snapshot is opt-in.
    """
    tag = f"stage{stage_idx}"
    save_boozer_surface_vtk(boozer_surface, btot, out_dir, tag)
    if per_stage_json:
        simsopt_save([all_coils, boozer_surface], str(out_dir / f"{tag}.json"))
    if res is not None:
        with open(out_dir / f"{tag}_optresult.txt", "w") as f:
            f.write(f"success = {bool(res.success)}\n")
            f.write(f"fun     = {float(res.fun):.6e}\n")
            f.write(f"nit     = {int(res.nit)}\n")
            f.write(f"nfev    = {int(res.nfev)}\n")
            f.write(f"message = {res.message}\n")


def case_to_flat_dict(case: Case) -> Dict[str, Any]:
    d = asdict(case)
    stages = d.pop("stages")
    d["stages_repr"] = repr(stages)
    w = d.pop("weights")
    for key, val in w.items():
        d[f"weight_{key}"] = val
    return d


def metrics_for_json(metrics: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in metrics.items():
        if isinstance(v, (np.floating, np.integer)):
            out[k] = float(v) if isinstance(v, np.floating) else int(v)
        elif isinstance(v, np.ndarray):
            out[k] = v.tolist()
        else:
            out[k] = v
    return out


def write_scan_summary(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


def run_case(
    case: Case,
    out_dir: Path,
    no_poincare: bool = False,
    per_stage_json: bool = False,
):
    t0 = time.time()
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Case {case.name}  target={case.target}  boozer={case.boozer_type} ===")
    print(f"    out_dir={out_dir}")

    setup = build_initial_coils(case)
    freeze_geometry_free_currents(setup)
    all_coils = setup["all_coils"]
    plasma_half = setup["plasma"]
    plasma_nfp = plasma_half.nfp
    # Nominal band for metrics; validation keeps generous limits so thin
    # continuation seeds (AR can be O(10-100)) are not rejected before the
    # outer ``W_AR`` penalty can drive the surface toward ``aspect_target``.
    aspect_bounds_nominal = aspect_bounds_for_case(case)
    validate_bounds = (1.5, 500.0)

    btot = BiotSavart(all_coils)
    G0 = G_from_TF(setup["coils_TF"])
    r0_target = float(case.R0_factor * plasma_half.get_rc(0, 0))

    print(
        f"   G0={G0:.5e}  iota_guess={case.iota_guess:.4f}  "
        f"iota_target={case.iota_target:.4f}  R0_target={r0_target:.5f}"
    )

    nfp = setup["nfp"]
    plasma_minor = float(plasma_half.minor_radius())
    if case.aspect_target is not None and float(case.aspect_target) > 0:
        a_edge_final = float(r0_target / case.aspect_target)
        a_edge_final = min(a_edge_final, 1.05 * plasma_minor)
    else:
        a_edge_final = float(case.a_edge_frac * plasma_minor)
    print(
        f"   plasma minor a={plasma_minor:.5f}  "
        f"seed a_edge_final={a_edge_final:.5f}"
    )

    # Replace the circular axis with a Picard-traced magnetic axis of the
    # TF-only field; this dramatically improves the QA seed.
    print("   Locating magnetic axis via Picard fieldline iteration...")
    btot_tf = BiotSavart(setup["coils_TF"])
    axis = find_magnetic_axis(
        btot_tf, plasma_half.get_rc(0, 0), plasma_nfp
    )
    setup["axis"] = axis

    save_coils_vtk(all_coils, out_dir, "initial")
    save_plasma_surface_error_vtk(
        setup["input_path"],
        btot,
        out_dir,
        "initial",
        case.plot_resolution_phi,
        case.plot_resolution_theta,
    )

    stages = case.stages
    mp0, nt0, _, _, _ = stages[0]

    # Boozer continuation: solve at a_edge_frac 0.05 -> 0.10 -> 0.20 -> case.a_edge_frac
    # with warm starts, and validate at each step.
    stellsym_eff = bool(setup.get("stellsym", True))
    boozer_surface, res = _seed_boozer_continuation(
        btot,
        axis,
        plasma_minor,
        nfp,
        mp0,
        nt0,
        case,
        G0,
        stellsym=stellsym_eff,
        aspect_bounds=validate_bounds,
        a_edge_final=a_edge_final,
    )
    ok_seed, why_seed = validate_boozer_state(
        boozer_surface, G0, aspect_bounds=validate_bounds
    )
    print(
        f"   Seed final: success={res.get('success', False)} "
        f"iter={res.get('iter', -1)} iota={res.get('iota', float('nan')):.5e} "
        f"G={res.get('G', float('nan')):.5e} validate={why_seed}"
    )
    if not ok_seed:
        print(f"   WARNING: seed surface failed validation ({why_seed})")

    prev_iota = res.get("iota", case.iota_guess) or case.iota_guess
    prev_x = None
    nit_total = 0
    last_result = None
    final_terms = None
    v_target_anchor = float(boozer_surface.targetlabel)

    for stage_idx, (mp, nt, sdim_k, maxiter_k, _ntol_k) in enumerate(stages):
        print(
            f"\n--- Stage {stage_idx}: (mpol, ntor)=({mp},{nt}) "
            f"sDIM={sdim_k} maxiter={maxiter_k} ---"
        )

        if stage_idx > 0:
            prev_dofs = boozer_surface.surface.x.copy()
            prev_iota_lift = boozer_surface.res["iota"]
            prev_G_lift = boozer_surface.res["G"]
            new_bsurf = upgrade_boozer_surface(boozer_surface, mp, nt)
            new_bsurf.options["verbose"] = False
            _install_ls_no_newton(new_bsurf, G0, aspect_bounds=validate_bounds)
            res = new_bsurf.run_code(prev_iota_lift, G=prev_G_lift)
            ok_lift, why_lift = validate_boozer_state(
                new_bsurf, G0, aspect_bounds=validate_bounds
            )
            print(
                f"   Lifted surface: success={(res or {}).get('success', False)} "
                f"iota={(res or {}).get('iota', float('nan')):.5e} "
                f"validate={why_lift}"
            )
            if ok_lift:
                boozer_surface = new_bsurf
            else:
                print(
                    "   WARNING: lifted surface failed validation; "
                    "holding previous (mpol,ntor) for this stage"
                )
                # Restore previous low-res surface state.
                boozer_surface.surface.x = prev_dofs
                boozer_surface.res["iota"] = prev_iota_lift
                boozer_surface.res["G"] = prev_G_lift

        bs_eval = BiotSavart(all_coils)
        JF, terms = build_objective(
            boozer_surface,
            bs_eval,
            setup["base_wp_currents"],
            sdim_k,
            case,
            plasma_nfp,
            case.iota_target,
            r0_target,
            v_target=v_target_anchor,
        )
        if stage_idx == 0:
            _probe_ls_adjoint(JF)
        if prev_x is not None and prev_x.size == JF.x.size:
            JF.x = prev_x

        state = {
            "sdofs": boozer_surface.surface.x.copy(),
            "iota": boozer_surface.res["iota"],
            "G": boozer_surface.res["G"],
        }
        fun = make_fun(
            JF,
            boozer_surface,
            state,
            G0,
            verbose=False,
            aspect_bounds=validate_bounds,
        )

        if case.do_taylor_test:
            print("   Taylor test:")
            taylor_test(fun, JF, h_seed=stage_idx)

        print(f"   L-BFGS-B maxiter={maxiter_k}")
        last_result = minimize(
            fun,
            JF.x,
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": maxiter_k, "maxcor": 500},
            tol=1e-12,
        )
        # Restore JF DOFs to the last accepted state (prevs), since the L-BFGS-B
        # final iterate may have been a rejected step that landed on the revert
        # branch.
        prevs = getattr(fun, "prevs", None)
        if prevs is not None:
            boozer_surface.surface.x = prevs["sdofs"]
            boozer_surface.res["iota"] = prevs["iota"]
            boozer_surface.res["G"] = prevs["G"]
        nit_total += int(last_result.nit)
        prev_iota = boozer_surface.res["iota"]
        prev_x = JF.x.copy()
        final_terms = terms

        print(
            f"   Stage {stage_idx} done: J={last_result.fun:.3e} "
            f"f_symm={float(terms['J_nonQS'].J()):.3e} "
            f"iota={prev_iota:.4e} "
            f"R0={float(MajorRadius(boozer_surface).J()):.4e} "
            f"nit={int(last_result.nit)} nfev={int(last_result.nfev)} "
            f"msg={str(last_result.message)!r}"
        )

        save_stage_outputs(
            out_dir,
            stage_idx,
            boozer_surface,
            all_coils,
            btot,
            last_result,
            per_stage_json=per_stage_json,
        )

    assert final_terms is not None and last_result is not None

    metrics = compute_final_metrics(
        boozer_surface,
        final_terms["J_nonQS"],
        setup,
        case,
        btot,
        plasma_half,
        last_result,
        nit_total,
        G0_ref=G0,
        aspect_bounds=validate_bounds,
        aspect_bounds_nominal=aspect_bounds_nominal,
    )

    save_coils_vtk(all_coils, out_dir, "optimized")
    save_plasma_surface_error_vtk(
        setup["input_path"],
        btot,
        out_dir,
        "optimized",
        case.plot_resolution_phi,
        case.plot_resolution_theta,
    )
    save_boozer_surface_vtk(boozer_surface, btot, out_dir, "optimized")

    do_pc = case.do_poincare and not no_poincare
    if do_pc:
        try:
            save_poincare(btot, boozer_surface, setup["input_path"], out_dir, "optimized", case)
        except Exception as exc:
            print(f"   WARNING: Poincaré tracing failed ({exc})")

    simsopt_save([all_coils, boozer_surface], str(out_dir / "stage1_final.json"))

    row = {
        **case_to_flat_dict(case),
        "nfp": plasma_nfp,
        **metrics_for_json(metrics),
    }
    row["wall_clock_s"] = time.time() - t0

    print("\n--- Final metrics ---")
    for key in sorted(metrics.keys()):
        print(f"   {key}: {metrics[key]}")
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics_for_json(metrics), f, indent=2)

    print(f"\nTotal time = {time.time() - t0:.1f} s")
    return row


def _probe_ls_adjoint(JF) -> None:
    """Single ``JF.dJ()`` probe used per case after the first ``build_objective``.

    Exercises the LS adjoint path (the existing checker only inspected
    ``bsurf.res`` keys, which are populated regardless of whether the gradient
    is actually usable).  Prints a warning if the gradient is non-finite or
    zero-norm.
    """
    g = np.asarray(JF.dJ())
    finite = bool(np.all(np.isfinite(g)))
    norm = float(np.linalg.norm(g))
    if not finite or norm == 0.0:
        print(
            f"   WARNING: LS adjoint gradient looks degenerate: "
            f"finite={finite} ||dJ||={norm:.2e}"
        )


def _poincare_smoke_shrink(case: Case) -> Case:
    """Single short stage + light tracing for ``QUASR_POINCARE_SMOKE=1``."""
    return replace(
        case,
        stages=((4, 4, 24, 15, 1e-10),),
        poincare_nlines=3,
        poincare_tmax=300.0,
        poincare_tol=1.0e-7,
        interp_grid_n=12,
        interp_degree=2,
    )


def pick_cases(args, rank: int, size: int) -> List[Case]:
    if args.case_idx is not None:
        c = CASES[args.case_idx]
        if args.mpi and HAS_MPI:
            if rank != (args.case_idx % size):
                return []
        return [c]
    if args.mpi and HAS_MPI:
        return [CASES[i] for i in range(rank, len(CASES), size)]
    return list(CASES)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-idx", type=int, default=None)
    parser.add_argument(
        "--out-dir",
        type=str,
        default="./stage1_quasr_scan/",
        help="Base directory for scan outputs.",
    )
    parser.add_argument("--no-poincare", action="store_true")
    parser.add_argument("--ci", action="store_true")
    parser.add_argument(
        "--per-stage-json",
        action="store_true",
        help=(
            "Write the heavy ``stage<k>.json`` snapshot (coils + Boozer surface) "
            "at every Fourier stage. Off by default."
        ),
    )
    parser.add_argument(
        "--mpi",
        action="store_true",
        help="Round-robin CASES across MPI ranks (needs mpi4py).",
    )
    args = parser.parse_args()

    use_mpi = args.mpi and HAS_MPI
    if args.mpi and not HAS_MPI:
        print("Warning: --mpi requested but mpi4py is not installed; running serial.")

    comm = _MPI.COMM_WORLD if use_mpi else None
    rank = comm.Get_rank() if comm else 0
    size = comm.Get_size() if comm else 1

    base_out = Path(args.out_dir).resolve()
    base_out.mkdir(parents=True, exist_ok=True)
    cases_run = pick_cases(args, rank, size)
    if (
        os.environ.get("QUASR_POINCARE_SMOKE") == "1"
        and not args.ci
        and cases_run
    ):
        cases_run = [_poincare_smoke_shrink(c) for c in cases_run]
    elif os.environ.get("QUASR_POINCARE_SMOKE") == "1" and args.ci:
        print("Note: QUASR_POINCARE_SMOKE=1 has no effect together with --ci.")

    summary_path = base_out / "scan_summary.csv"

    local_rows: List[Dict[str, Any]] = []
    for case in cases_run:
        resolved = case_for_ci(case) if args.ci else case
        case_out = base_out / resolved.name
        row = run_case(
            resolved,
            case_out,
            no_poincare=args.no_poincare,
            per_stage_json=args.per_stage_json,
        )
        local_rows.append(row)
        if not use_mpi:
            write_scan_summary(local_rows, summary_path)

    if use_mpi:
        gathered = comm.gather(local_rows, root=0)
        if rank == 0:
            flat = [r for chunk in gathered for r in chunk]
            write_scan_summary(flat, summary_path)
            print(f"Wrote {summary_path} ({len(flat)} rows).")
    elif local_rows:
        print(f"Wrote {summary_path} ({len(local_rows)} rows).")


if __name__ == "__main__":
    main()
