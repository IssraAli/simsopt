"""
Inner local optimization: LSQ-TRF wrapping VMEC, parameterized by an initial
DOF vector in the unit cube. Lifted from `examples/2_Intermediate/spline_surface.py`
but exposed as a function so it can be driven by an outer global sampler.

The instrumented `_PASOpt.J_qa` appends per-call diagnostics to a `trace` list,
giving an SSR-vs-evaluation curve we can plot per trial. This includes both
LSQ-TRF outer iterations and the FD-Jacobian inner evaluations.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any

import fnmatch
import numpy as np
from scipy.interpolate import interp1d

from simsopt._core import Optimizable
from simsopt._core.types import RealArray
from simsopt.geo.surfacespline import SurfaceBSpline
from simsopt.mhd import Vmec
from simsopt.objectives.least_squares import LeastSquaresProblem
from simsopt.solve.mpi import bounded_least_squares_mpi_solve
from simsopt.util.mpi import MpiPartition

# Number of components in the QA + iota residual vector — same decomposition as
# the LSQ reference (spline_surface.py): ar + ι_edge + ι_axis + 20·63·64 + 2·63·64.
# We keep the *flat* residual vector here (matching LSQ semantics) rather than
# the per-surface-norm aggregation used by the composite-BO variant, so the LSQ
# Jacobian sees full Jacobian structure.

LB_PHYS_DEFAULTS = {
    'CrossSectionFixedZeta*r*': (0.01, 0.8),
    'PseudoAxis*r_axis*': (0.7, 2.2),
    'PseudoAxis*z_axis*': (-0.5, 0.5),
}


def set_lsq_bounds(surf: SurfaceBSpline) -> SurfaceBSpline:
    """Apply the LSQ-style physical bounds to `surf`. Mutates and returns it."""
    lb = np.copy(surf.lower_bounds)
    ub = np.copy(surf.upper_bounds)
    for pattern, (lo, hi) in LB_PHYS_DEFAULTS.items():
        mask = [fnmatch.fnmatch(d, pattern) for d in surf.dof_names]
        lb[mask] = lo
        ub[mask] = hi
    surf.lower_bounds = lb
    surf.upper_bounds = ub
    return surf


# --------------------------------------------------------------------- physics


def _alan_qs_residuals(vmec: Vmec, surfaces, helicity_m=1, helicity_n=0,
                       weights=None, ntheta=63, nphi=64):
    """QS residuals as a flat 1D array (matches `alan_QuasisymmetryRatioResidual`
    in spline_surface.py and the BO test_target.py — bit-identical numerics)."""
    try:
        surfaces = list(surfaces)
    except TypeError:
        surfaces = [surfaces]
    if weights is None:
        weights = np.ones(len(surfaces))

    vmec.run()
    if vmec.wout.lasym:
        raise RuntimeError('QS class cannot yet handle non-stellarator-symmetric configs')

    ns = len(surfaces)
    nfp = vmec.wout.nfp
    d_psi_d_s = -vmec.wout.phi[-1] / (2 * np.pi)
    iota = interp1d(vmec.s_half_grid, vmec.wout.iotas[1:], fill_value='extrapolate')(surfaces)
    G = interp1d(vmec.s_half_grid, vmec.wout.bvco[1:], fill_value='extrapolate')(surfaces)
    I = interp1d(vmec.s_half_grid, vmec.wout.buco[1:], fill_value='extrapolate')(surfaces)
    gmnc = interp1d(vmec.s_half_grid, vmec.wout.gmnc[:, 1:], fill_value='extrapolate')(surfaces)
    bmnc = interp1d(vmec.s_half_grid, vmec.wout.bmnc[:, 1:], fill_value='extrapolate')(surfaces)
    bsubumnc = interp1d(vmec.s_half_grid, vmec.wout.bsubumnc[:, 1:], fill_value='extrapolate')(surfaces)
    bsubvmnc = interp1d(vmec.s_half_grid, vmec.wout.bsubvmnc[:, 1:], fill_value='extrapolate')(surfaces)
    bsupumnc = interp1d(vmec.s_half_grid, vmec.wout.bsupumnc[:, 1:], fill_value='extrapolate')(surfaces)
    bsupvmnc = interp1d(vmec.s_half_grid, vmec.wout.bsupvmnc[:, 1:], fill_value='extrapolate')(surfaces)

    theta1d = np.linspace(0, 2 * np.pi, ntheta, endpoint=False)
    phi1d = np.linspace(0, 2 * np.pi / nfp, nphi, endpoint=False)
    phi2d, theta2d = np.meshgrid(phi1d, theta1d)
    phi3d = phi2d.reshape((1, ntheta, nphi))
    theta3d = theta2d.reshape((1, ntheta, nphi))

    shp = (ns, ntheta, nphi)
    modB = np.zeros(shp); d_B_d_theta = np.zeros(shp); d_B_d_phi = np.zeros(shp)
    sqrtg = np.zeros(shp); bsubu = np.zeros(shp); bsubv = np.zeros(shp)
    bsupu = np.zeros(shp); bsupv = np.zeros(shp); residuals3d = np.zeros(shp)
    for jmn in range(len(vmec.wout.xm_nyq)):
        m = vmec.wout.xm_nyq[jmn]; n = vmec.wout.xn_nyq[jmn]
        angle = m * theta3d - n * phi3d
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        modB += np.kron(bmnc[jmn].reshape((ns, 1, 1)), cos_a)
        d_B_d_theta += np.kron(bmnc[jmn].reshape((ns, 1, 1)), -m * sin_a)
        d_B_d_phi += np.kron(bmnc[jmn].reshape((ns, 1, 1)), n * sin_a)
        sqrtg += np.kron(gmnc[jmn].reshape((ns, 1, 1)), cos_a)
        bsubu += np.kron(bsubumnc[jmn].reshape((ns, 1, 1)), cos_a)
        bsubv += np.kron(bsubvmnc[jmn].reshape((ns, 1, 1)), cos_a)
        bsupu += np.kron(bsupumnc[jmn].reshape((ns, 1, 1)), cos_a)
        bsupv += np.kron(bsupvmnc[jmn].reshape((ns, 1, 1)), cos_a)

    B_dot_grad_B = bsupu * d_B_d_theta + bsupv * d_B_d_phi
    B_cross_grad_B_dot_grad_psi = d_psi_d_s * (bsubu * d_B_d_phi - bsubv * d_B_d_theta) / sqrtg

    dtheta = theta1d[1] - theta1d[0]
    dphi = phi1d[1] - phi1d[0]
    V_prime = nfp * dtheta * dphi * np.sum(sqrtg, axis=(1, 2))
    meanB = np.abs(np.sum(modB * sqrtg, axis=(1, 2)) / V_prime * nfp * dtheta * dphi)
    nn = helicity_n * nfp
    for js in range(ns):
        residuals3d[js] = np.sqrt(weights[js] * nfp * dtheta * dphi / V_prime[js] * sqrtg[js]) * (
            B_cross_grad_B_dot_grad_psi[js] * (nn - iota[js] * helicity_m)
            - B_dot_grad_B[js] * (helicity_m * G[js] + nn * I[js])
        ) / (modB[js] ** 2 * meanB[js])

    return residuals3d.reshape((ns * ntheta * nphi,))


# --------------------------------------------------------------------- LSQ wrapper

class _PASOpt(Optimizable):
    """Same residual function as PASOpt in spline_surface.py, with a `trace` list
    that records every J_qa call's SSR for post-hoc plotting."""

    def __init__(self, surf, spline_kwargs, mpi):
        self.surf = surf
        self.spline_kwargs = spline_kwargs
        self.mpi = mpi
        self.trace: list[dict[str, Any]] = []  # populated by J_qa
        Optimizable.__init__(self, depends_on=[surf])

    def J_qa(self):
        try:
            new_surf = SurfaceBSpline(**self.spline_kwargs)
            new_surf.axis.fix('r_axis_0')
            new_surf.set_dofs_from_vec(self.x)

            rz_surf = new_surf.to_RZFourier(
                nu=64, nv=64, nv_interp=128, nu_interp=128,
                collocation='arclength', plot=False,
                spec_cond_options={
                    'plot': False, 'ftol': 1e-4, 'Mtol': 1.1, 'shapetol': None,
                    'niters': 2000, 'verbose': False, 'cutoff': 1e-5,
                },
            )
            vmec = Vmec.vmec_from_surf(
                nfp=self.surf.nfp, surf=rz_surf, mpi=self.mpi,
                ns=13, M=12, N=12, ftol=1e-7,
            )
            vmec.run()

            ar = vmec.aspect() - 6.0
            iota_edge = np.sqrt(10) * (vmec.iota_edge() - 0.42)
            iota_axis = np.sqrt(10) * (vmec.iota_axis() - 0.42)
            qs_inner = _alan_qs_residuals(vmec, np.linspace(0.02, 1, 20), 1, 0)
            qs_outer = np.sqrt(5) * _alan_qs_residuals(vmec, np.arange(1.0, 1.1, 0.1), 1, 0)

            residuals = np.concatenate(([ar, iota_edge, iota_axis], qs_inner, qs_outer))
            ssr = float(np.sum(residuals ** 2))
            # Cheap summary of *which surfaces dominate* — useful for diagnostics
            # without storing the full 88k-vector. Per-surface QS L2 norms:
            # (per-surface aggregation, same as spline_bayesian_opt_composite).
            ntheta, nphi = 63, 64
            qs_inner_norms = np.linalg.norm(qs_inner.reshape(20, ntheta * nphi), axis=1)
            qs_outer_norms = np.sqrt(5) * np.linalg.norm(
                (qs_outer / np.sqrt(5)).reshape(2, ntheta * nphi), axis=1
            )
            self.trace.append({
                'eval_idx': len(self.trace),
                'ssr': ssr,
                'ar': float(ar),
                'iota_edge': float(iota_edge),
                'iota_axis': float(iota_axis),
                'qs_inner_norms': qs_inner_norms.tolist(),
                'qs_outer_norms': qs_outer_norms.tolist(),
            })
            return residuals
        except Exception as e:
            # On VMEC blowup, return a large finite residual so the LSQ doesn't
            # crash the whole trial — same convention as `_f_proc0` in solve/mpi.py
            # but with a known sentinel size we can recover from.
            self.trace.append({
                'eval_idx': len(self.trace),
                'ssr': float('inf'),
                'failed': True,
                'error': repr(e)[:200],
            })
            raise  # let solve/mpi.py's _f_proc0 catch it; it will substitute 1e12


@contextmanager
def _cwd(path):
    """chdir to `path` for the duration of the block."""
    if path is None:
        yield
        return
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def from_unit_cube(x, lb, ub):
    return x * (ub - lb) + lb


def to_unit_cube(x, lb, ub):
    return (x - lb) / (ub - lb)


def build_template_surf_and_bounds(spline_kwargs: dict, default_r: float = 0.4):
    """Build the gauge-fixed template surface and read its (physical) DOF bounds.

    Returned bounds are in *physical* units (matching the LSQ formulation). Use
    `to_unit_cube`/`from_unit_cube` to convert between unit cube and physical.
    """
    surf = SurfaceBSpline(**spline_kwargs, default_r=default_r)
    surf.axis.fix('r_axis_0')
    set_lsq_bounds(surf)
    lb = np.array(surf.lower_bounds, dtype=np.float64)
    ub = np.array(surf.upper_bounds, dtype=np.float64)
    return surf, lb, ub


def run_lsq_from_dofs(
    unit_cube_x: np.ndarray,
    spline_kwargs: dict,
    lb: np.ndarray,
    ub: np.ndarray,
    mpi: MpiPartition,
    *,
    max_nfev: int = 500,
    abs_step: float = 1e-4,
    rel_step: float = 1e-8,
    diff_method: str = 'forward',
    work_dir: str | None = None,
) -> dict[str, Any]:
    """
    Run bounded LSQ-TRF inside `work_dir` starting from `unit_cube_x` (in [0,1]^d).

    Returns a dict with:
      - 'init_x_unit', 'init_x_phys'
      - 'final_x_unit', 'final_x_phys'
      - 'init_ssr', 'final_ssr', 'init_nfev', 'nfev'
      - 'trace'   : list of per-eval dicts (eval_idx, ssr, ar, iota_*, qs_*_norms)
      - 'status'  : scipy least_squares result.status (None on master-only paths)
      - 'message' : scipy least_squares result.message
    """
    surf = SurfaceBSpline(**spline_kwargs)
    surf.axis.fix('r_axis_0')
    set_lsq_bounds(surf)
    assert surf.x.shape == unit_cube_x.shape, (
        f'surf has {surf.x.shape} dofs but unit_cube_x is {unit_cube_x.shape}; '
        'check that gauge fix + spline_kwargs match between sampler and inner LSQ'
    )

    init_phys = from_unit_cube(unit_cube_x, lb, ub)
    surf.set_dofs_from_vec(init_phys)

    myopt = _PASOpt(surf, spline_kwargs, mpi)
    prob = LeastSquaresProblem.from_tuples([(myopt.J_qa, 0, 1)])

    with _cwd(work_dir):
        bounded_least_squares_mpi_solve(
            prob, mpi=mpi, grad=True,
            abs_step=abs_step, rel_step=rel_step, diff_method=diff_method,
            x0=init_phys,
            max_nfev=max_nfev,
        )
        # `bounded_least_squares_mpi_solve` writes `result.npy` to cwd.
        result_obj: Any = None
        try:
            with open('result.npy', 'rb') as f:
                result_obj = np.load(f, allow_pickle=True).item()
        except (FileNotFoundError, ValueError):
            pass

    final_phys = np.array(myopt.x, dtype=np.float64)
    final_unit = to_unit_cube(final_phys, lb, ub)

    if not myopt.trace:
        # Should never happen unless VMEC failed on the very first call.
        init_ssr = float('inf')
        final_ssr = float('inf')
    else:
        init_ssr = myopt.trace[0]['ssr']
        # Final SSR = SSR at result.x. Recompute via running J_qa once at the
        # solution — this also appends to the trace, so the trace tail reflects
        # the true convergence point.
        try:
            myopt.J_qa()
            final_ssr = myopt.trace[-1]['ssr']
        except Exception:
            final_ssr = myopt.trace[-1]['ssr']

    status = getattr(result_obj, 'status', None) if result_obj is not None else None
    message = getattr(result_obj, 'message', None) if result_obj is not None else None
    nfev = getattr(result_obj, 'nfev', len(myopt.trace)) if result_obj is not None else len(myopt.trace)

    return {
        'init_x_unit': np.array(unit_cube_x, dtype=np.float64).tolist(),
        'init_x_phys': init_phys.tolist(),
        'final_x_unit': final_unit.tolist(),
        'final_x_phys': final_phys.tolist(),
        'init_ssr': float(init_ssr),
        'final_ssr': float(final_ssr),
        'nfev': int(nfev),
        'trace': myopt.trace,
        'status': int(status) if status is not None else None,
        'message': str(message) if message is not None else None,
    }
