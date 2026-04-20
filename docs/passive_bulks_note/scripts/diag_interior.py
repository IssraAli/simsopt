r"""Throwaway diagnostic: where is the interior field coming from?

For the stubby diagnostic-fixture puck :math:`R=t=0.30` in a nearly
uniform axial field :math:`B_0\hat z` (large ring coil), we measure:

* **Algebraic residual**:
  :math:`\|L\beta + f\|_\infty / \|f\|_\infty` where
  :math:`\beta = \mathrm{Q}_c Q_L \alpha` and :math:`\alpha` is the
  eigenfloor-solved reduced solution.  This should be at floating-point
  precision if the solver itself is converged -- it is a pure sanity
  check on the linear algebra path.
* **Off-shell interior normal residual**:
  :math:`\rho_S^{\rm in}(\epsilon) = \|B_n^{TF}+B_n^{ind}\|_{L^2(\Sigma_\epsilon)}
  / \|B_n^{TF}\|_{L^2(\Sigma_\epsilon)}` where :math:`\Sigma_\epsilon`
  is the shell pushed inward along the outward normal by
  :math:`\epsilon`.  Avoids the :math:`1/r^2` singularity that appears
  when Biot-Savart is evaluated exactly on the source quadrature.
* **Interior multi-point ratio**:
  :math:`|B^{tot}(\mathbf x_k)|/|B^{TF}(\mathbf x_k)|` averaged over a
  small interior cloud.  Answers "does the ideal-diamagnet body field
  cancel in the bulk interior, not just at the centre?".
* **Retained mode count** ``Q.shape[1]`` for three null-space
  thresholds :math:`\{10^{-6}, 10^{-10}, 10^{-14}\}`.

Decision tree:

* Algebraic residual machine-precision, off-shell
  :math:`\rho_S^{\rm in}\to 0` but interior multi-point plateau
  :math:`\Rightarrow` genuine weak-form mismatch: Phase 2 Path A.
* Algebraic residual machine-precision, but :math:`\rho_S^{\rm in}`
  ALSO plateaus :math:`\Rightarrow` the energy form is converged (all
  Galerkin dofs satisfy :math:`\int\Phi_a B_n^{tot}=0`) yet the
  pointwise :math:`B_n` is large in the basis-orthogonal complement.
  Still Path A, more severely.
* Algebraic residual itself large or sensitive to null-space threshold
  :math:`\Rightarrow` Path B (threshold / null-space issue dominates).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ring_coil  # noqa: E402

from simsopt.field.biotsavart import BiotSavart  # noqa: E402
from simsopt.field.bulk_inductance import (  # noqa: E402
    null_space_projection_matrix,
    shell_loading_vector_pure,
)
from simsopt.field.psc_bulk import PSCBulkArray  # noqa: E402


def _build(R, t, tf, m, lz, k, thr_ns=1e-10):
    """Build the fixture puck with an explicit null-space threshold."""
    psc = PSCBulkArray(
        np.array([[0.0, 0.0, 0.0]]),
        np.array([[0.0, 0.0, 1.0]]),
        np.array([R]), np.array([t]), [tf],
        eval_points=np.array([[2.0 * R, 0.0, 0.0]]),
        m_fourier=m, l_zernike=lz, k_chebyshev=k,
        n_rho=10, n_phi=16, n_z=6,
        nfp=1, stellsym=False, adaptive_self_reg=True,
        null_space_threshold=thr_ns, eigenfloor_threshold=1e-10,
    )
    psc.exact_disc_faces = True
    psc._rebuild()
    return psc


def _algebraic_residual(psc: PSCBulkArray) -> float:
    """Return ``||L beta - (-f)||_inf / ||f||_inf`` -- solver sanity."""
    L = psc._L_work
    Bn = psc._compute_bn_at_quads_numpy()
    phi = psc._phi_mat
    w = psc._quad_weights
    f = np.asarray(shell_loading_vector_pure(phi, w, Bn))
    beta = np.asarray(psc.beta)
    lhs = L @ beta
    return float(np.max(np.abs(lhs + f)) / (np.max(np.abs(f)) + 1e-30))


def _inset_shell_points(psc: PSCBulkArray, eps_frac: float) -> np.ndarray:
    """Push quadrature points inward along the outward normal."""
    R = float(psc._all_pucks[0][2])
    eps = eps_frac * R
    return psc._quad_points - eps * psc._quad_normals


def _offshell_residual(psc: PSCBulkArray, tf, eps_frac: float) -> float:
    """L2 residual of ``B_n`` on a shell inset by ``eps_frac * R``."""
    pts = _inset_shell_points(psc, eps_frac)
    n = psc._quad_normals
    w = psc._quad_weights
    bs = BiotSavart([tf]); bs.set_points_cart(np.ascontiguousarray(pts))
    Bn_tf = np.sum(np.asarray(bs.B()) * n, axis=1)
    B_ind = np.asarray(psc.B_at_points(pts))
    Bn_ind = np.sum(B_ind * n, axis=1)
    num = np.sqrt(float(np.sum(w * (Bn_tf + Bn_ind) ** 2)))
    den = np.sqrt(float(np.sum(w * Bn_tf ** 2))) + 1e-30
    return num / den


def _interior_points(R: float, t: float, n: int = 16) -> np.ndarray:
    """Return a small cloud of interior probe points."""
    rng = np.random.default_rng(0)
    r = 0.3 * R * rng.random(n) ** (1.0 / 3.0)
    th = 2.0 * np.pi * rng.random(n)
    z = 0.3 * t * (2.0 * rng.random(n) - 1.0)
    return np.stack([r * np.cos(th), r * np.sin(th), z], axis=1)


def _interior_multipoint(psc: PSCBulkArray, tf, R: float, t: float) -> float:
    """Mean ``|B^tot|/|B^TF|`` over a small interior cloud."""
    pts = _interior_points(R, t)
    bs = BiotSavart([tf]); bs.set_points_cart(np.ascontiguousarray(pts))
    B_tf = np.asarray(bs.B())
    B_ind = np.asarray(psc.B_at_points(pts))
    num = np.linalg.norm(B_tf + B_ind, axis=1)
    den = np.linalg.norm(B_tf, axis=1) + 1e-30
    return float(np.mean(num / den))


def _interior_ratio_center(psc: PSCBulkArray, tf) -> float:
    """``|B^tot(0)|/|B^TF(0)|`` at the puck centre."""
    probe = np.array([[0.0, 0.0, 0.0]])
    bs = BiotSavart([tf]); bs.set_points_cart(probe)
    B_tf = np.asarray(bs.B())[0]
    B_ind = np.asarray(psc.B_at_points(probe))[0]
    return float(np.linalg.norm(B_tf + B_ind) / (np.linalg.norm(B_tf) + 1e-30))


def _retained_rank(psc: PSCBulkArray, thresholds):
    """``Q_L.shape[1]`` for a set of null-space thresholds."""
    L = psc._L_work
    Q_c = psc._Q_c
    L_c = Q_c.T @ L @ Q_c
    out = {}
    for thr in thresholds:
        Q_L = null_space_projection_matrix(L_c, threshold=thr)
        out[thr] = int(Q_L.shape[1])
    return out


def main() -> None:
    """Run the diagnostic sweep."""
    R_coil = 5.0; I_coil = 1.0e7
    R = 0.30; t = 0.30
    tf = ring_coil(R_coil, 0.0, I_coil, order=1, quadpoints=32)

    basis_sweep = [(2, 4, 2), (3, 6, 3), (4, 8, 4)]
    thresholds = [1e-6, 1e-10, 1e-14]
    eps_frac = 0.05

    hdr = (f"{'(m,l,k)':>8s}  {'nDoFs':>6s}  {'rank_6/10/14':>14s}  "
           f"{'|Lb+f|':>10s}  {'rho_S@5%':>10s}  {'rho_V_ctr':>10s}  {'rho_V_cloud':>11s}")
    print(hdr)
    print("-" * len(hdr))
    for (m, lz, k) in basis_sweep:
        psc = _build(R, t, tf, m, lz, k)
        n_dofs = psc._L_work.shape[0]
        ra = _algebraic_residual(psc)
        rS = _offshell_residual(psc, tf, eps_frac)
        rVc = _interior_ratio_center(psc, tf)
        rVm = _interior_multipoint(psc, tf, R, t)
        ranks = _retained_rank(psc, thresholds)
        rank_s = f"{ranks[1e-6]}/{ranks[1e-10]}/{ranks[1e-14]}"
        print(f"({m},{lz},{k})".rjust(8) + f"  {n_dofs:>6d}  {rank_s:>14s}  "
              f"{ra:>10.2e}  {rS:>10.3e}  {rVc:>10.3e}  {rVm:>11.3e}", flush=True)


if __name__ == "__main__":
    main()
