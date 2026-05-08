import numpy as np
import torch
from simsopt.geo import SurfaceBSpline
from simsopt.mhd import Vmec
from simsopt.util.mpi import MpiPartition
from simsopt._core.types import RealArray
from bo_utils import from_unit_cube
from mpi4py import MPI
from scipy.interpolate import interp1d
from typing import Union

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Length of the residual vector returned on success: ar + ι_edge + ι_axis + 20 inner QS surfaces + 2 boundary QS surfaces.
N_RESIDUALS = 3 + 20 + 2  # = 25

_mpi = None


def get_mpi():
    global _mpi
    if _mpi is None:
        _mpi = MpiPartition()
    return _mpi


def parallel_batch_target(candidates, spline_kwargs, lb, ub, stopp):
    """
    Rank-0 returns (Y, feas_mask):
        Y         : torch.Tensor, shape (batch, N_RESIDUALS), dtype double
        feas_mask : torch.Tensor, shape (batch,), dtype bool
    Workers return None.
    Failed evaluations are reported as a zero row + feas=False; the driver
    drops those rows from the GP training set.
    """
    stopp[0] = comm.bcast(stopp[0], root=0)
    if stopp[0] == 0:
        x = comm.scatter(candidates, root=0)
        val, feas = target(x.flatten(), spline_kwargs, lb, ub)
        gathered_val = comm.gather(val)
        gathered_feas = comm.gather(feas)
        if rank == 0:
            print("batch complete.")
            Y = torch.tensor(np.asarray(gathered_val), dtype=torch.double).reshape(-1, N_RESIDUALS)
            feas_mask = torch.tensor(np.asarray(gathered_feas), dtype=torch.bool).reshape(-1)
            return Y, feas_mask


def target(X, spline_kwargs, lb, ub):
    """
    Evaluate the QA + iota target at a unit-cube point X.

    Returns (vec, feas):
        vec  : np.ndarray, shape (N_RESIDUALS,) — the residual vector whose
               sum-of-squares equals the legacy scalar BO objective `Σ r²`
        feas : bool — True iff VMEC ran and the residuals are finite
    On failure the vec is `np.zeros(N_RESIDUALS)`; downstream drops those.
    """
    dofs = from_unit_cube(X, lb, ub)

    surf = SurfaceBSpline(**spline_kwargs)
    # Match the gauge fix used in the LSQ reference (spline_surface.py).
    surf.axis.fix('r_axis_0')
    assert len(surf.x) == len(dofs), f'len(surf.x): {len(surf.x)}, len(dofs): {len(dofs)}'
    surf.set_dofs_from_vec(np.array(dofs))

    try:
        rz_surf = surf.to_RZFourier(
            nu=64, nv=64, nv_interp=128, nu_interp=128,
            collocation='arclength', plot=False, spec_cond=True,
            spec_cond_options={
                'plot': False, 'ftol': 1e-4, 'Mtol': 1.1, 'shapetol': None,
                'niters': 5000, 'verbose': False, 'cutoff': 1e-6,
            },
        )
        vmec = Vmec.vmec_from_surf(
            nfp=rz_surf.nfp, surf=rz_surf, mpi=get_mpi(),
            ns=13, M=12, N=12, ftol=1e-7,
        )
        vmec.run()

        ar_penalty = ar_target(vmec, 6)
        iota_edge_penalty = np.sqrt(10) * (vmec.iota_edge() - 0.42)
        iota_axis_penalty = np.sqrt(10) * (vmec.iota_axis() - 0.42)

        # Per-surface QS L2 norms — preserves Σ r² exactly (see plan-file derivation).
        # `_qs_residuals3d` returns shape (ns, ntheta, nphi); norm collapses (theta, phi).
        qs_inner_3d = _qs_residuals3d(vmec, np.linspace(0.02, 1, 20), 1, 0)   # (20, 63, 64)
        qs_outer_3d = _qs_residuals3d(vmec, np.arange(1.0, 1.1, 0.1), 1, 0)   # (2,  63, 64)

        qs_inner_norms = np.linalg.norm(qs_inner_3d.reshape(qs_inner_3d.shape[0], -1), axis=1)  # (20,)
        qs_outer_norms = np.sqrt(5) * np.linalg.norm(
            qs_outer_3d.reshape(qs_outer_3d.shape[0], -1), axis=1
        )                                                                                       # (2,)

        vec = np.concatenate((
            np.array([ar_penalty, iota_edge_penalty, iota_axis_penalty]),
            qs_inner_norms,
            qs_outer_norms,
        ))
        assert vec.shape == (N_RESIDUALS,), f'unexpected residual length: {vec.shape}'

        # Sanity check: SSR of the 25-vector must equal the legacy flat-vector SSR.
        # This validates the per-surface aggregation math empirically. Cost is one
        # extra dot product; failure would indicate an indexing/weighting bug.
        legacy_residuals = np.concatenate((
            np.array([ar_penalty, iota_edge_penalty, iota_axis_penalty]),
            qs_inner_3d.reshape(-1),
            np.sqrt(5) * qs_outer_3d.reshape(-1),
        ))
        ssr_25 = float(np.sum(vec ** 2))
        ssr_full = float(np.sum(legacy_residuals ** 2))
        rel_err = abs(ssr_25 - ssr_full) / max(abs(ssr_full), 1e-30)
        if rel_err > 1e-10:
            print(f'WARNING: SSR aggregation mismatch — rel_err={rel_err:.3e} '
                  f'(ssr_25={ssr_25:.6e}, ssr_full={ssr_full:.6e})')

        if not np.all(np.isfinite(vec)):
            print('Non-finite residual produced; marking infeasible')
            return np.zeros(N_RESIDUALS), False
        return vec, True

    except Exception as e:
        print(f'Failed with exception {e}; marking infeasible')
        return np.zeros(N_RESIDUALS), False


def composite_scalar(vec):
    """The scalar the BO acquisition maximizes: f = -0.5 * Σᵢ rᵢ²."""
    return -0.5 * float(np.sum(np.asarray(vec) ** 2))


def ar_target(vmec, target):
    return vmec.aspect() - target


def iota_target(vmec, target):
    return vmec.mean_iota() - target


def _qs_residuals3d(vmec: Vmec,
                    surfaces: Union[float, RealArray],
                    helicity_m: int = 1,
                    helicity_n: int = 0,
                    weights: RealArray = None,
                    ntheta: int = 63,
                    nphi: int = 64):
    """
    QS-violation residuals on `surfaces`, returned as a (ns, ntheta, nphi) tensor
    *before* flattening. The per-(θ, φ) values are bit-identical to the legacy
    `alan_QuasisymmetryRatioResidual` — only the final reshape is omitted, so
    the sum-of-squares over all entries matches the legacy implementation.
    """
    try:
        surfaces = list(surfaces)
    except TypeError:
        surfaces = [surfaces]

    if weights is None:
        weights = np.ones(len(surfaces))
    assert len(weights) == len(surfaces)

    vmec.run()
    if vmec.wout.lasym:
        raise RuntimeError('Quasisymmetry class cannot yet handle non-stellarator-symmetric configs')

    ns = len(surfaces)
    nfp = vmec.wout.nfp
    d_psi_d_s = -vmec.wout.phi[-1] / (2 * np.pi)

    interp = interp1d(vmec.s_half_grid, vmec.wout.iotas[1:], fill_value="extrapolate")
    iota = interp(surfaces)
    interp = interp1d(vmec.s_half_grid, vmec.wout.bvco[1:], fill_value="extrapolate")
    G = interp(surfaces)
    interp = interp1d(vmec.s_half_grid, vmec.wout.buco[1:], fill_value="extrapolate")
    I = interp(surfaces)
    interp = interp1d(vmec.s_half_grid, vmec.wout.gmnc[:, 1:], fill_value="extrapolate")
    gmnc = interp(surfaces)
    interp = interp1d(vmec.s_half_grid, vmec.wout.bmnc[:, 1:], fill_value="extrapolate")
    bmnc = interp(surfaces)
    interp = interp1d(vmec.s_half_grid, vmec.wout.bsubumnc[:, 1:], fill_value="extrapolate")
    bsubumnc = interp(surfaces)
    interp = interp1d(vmec.s_half_grid, vmec.wout.bsubvmnc[:, 1:], fill_value="extrapolate")
    bsubvmnc = interp(surfaces)
    interp = interp1d(vmec.s_half_grid, vmec.wout.bsupumnc[:, 1:], fill_value="extrapolate")
    bsupumnc = interp(surfaces)
    interp = interp1d(vmec.s_half_grid, vmec.wout.bsupvmnc[:, 1:], fill_value="extrapolate")
    bsupvmnc = interp(surfaces)

    theta1d = np.linspace(0, 2 * np.pi, ntheta, endpoint=False)
    phi1d = np.linspace(0, 2 * np.pi / nfp, nphi, endpoint=False)
    phi2d, theta2d = np.meshgrid(phi1d, theta1d)
    phi3d = phi2d.reshape((1, ntheta, nphi))
    theta3d = theta2d.reshape((1, ntheta, nphi))

    myshape = (ns, ntheta, nphi)
    modB = np.zeros(myshape)
    d_B_d_theta = np.zeros(myshape)
    d_B_d_phi = np.zeros(myshape)
    sqrtg = np.zeros(myshape)
    bsubu = np.zeros(myshape)
    bsubv = np.zeros(myshape)
    bsupu = np.zeros(myshape)
    bsupv = np.zeros(myshape)
    residuals3d = np.zeros(myshape)
    for jmn in range(len(vmec.wout.xm_nyq)):
        m = vmec.wout.xm_nyq[jmn]
        n = vmec.wout.xn_nyq[jmn]
        angle = m * theta3d - n * phi3d
        cosangle = np.cos(angle)
        sinangle = np.sin(angle)
        modB += np.kron(bmnc[jmn, :].reshape((ns, 1, 1)), cosangle)
        d_B_d_theta += np.kron(bmnc[jmn, :].reshape((ns, 1, 1)), -m * sinangle)
        d_B_d_phi += np.kron(bmnc[jmn, :].reshape((ns, 1, 1)), n * sinangle)
        sqrtg += np.kron(gmnc[jmn, :].reshape((ns, 1, 1)), cosangle)
        bsubu += np.kron(bsubumnc[jmn, :].reshape((ns, 1, 1)), cosangle)
        bsubv += np.kron(bsubvmnc[jmn, :].reshape((ns, 1, 1)), cosangle)
        bsupu += np.kron(bsupumnc[jmn, :].reshape((ns, 1, 1)), cosangle)
        bsupv += np.kron(bsupvmnc[jmn, :].reshape((ns, 1, 1)), cosangle)

    B_dot_grad_B = bsupu * d_B_d_theta + bsupv * d_B_d_phi
    B_cross_grad_B_dot_grad_psi = d_psi_d_s * (bsubu * d_B_d_phi - bsubv * d_B_d_theta) / sqrtg

    dtheta = theta1d[1] - theta1d[0]
    dphi = phi1d[1] - phi1d[0]
    V_prime = nfp * dtheta * dphi * np.sum(sqrtg, axis=(1, 2))
    assert np.sum(np.abs(np.sqrt((1 / V_prime) * nfp * dtheta * dphi * np.sum(sqrtg, axis=(1, 2))) - 1)) < 1e-12

    meanB = np.abs((np.sum(modB * sqrtg, axis=(1, 2)) / V_prime * nfp * dtheta * dphi))
    nn = helicity_n * nfp
    for js in range(ns):
        residuals3d[js, :, :] = np.sqrt(weights[js] * nfp * dtheta * dphi / V_prime[js] * sqrtg[js, :, :]) \
            * (B_cross_grad_B_dot_grad_psi[js, :, :] * (nn - iota[js] * helicity_m)
               - B_dot_grad_B[js, :, :] * (helicity_m * G[js] + nn * I[js])) \
            / (modB[js, :, :] ** 2 * meanB[js])

    return residuals3d
