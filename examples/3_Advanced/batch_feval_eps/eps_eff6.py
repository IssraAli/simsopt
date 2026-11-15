#!/usr/bin/env python3
"""Effective ripple ε_eff for simsopt, following Nemov et al. (1999).

This implementation closely follows the DESC Bounce1D algorithm
(arXiv:2412.01724) which has been validated against the NEO code.

Optimized version with vectorized operations for improved performance.
"""

import logging
from typing import Union, Optional
import warnings

import numpy as np
from scipy.interpolate import CubicHermiteSpline, CubicSpline
from scipy.optimize import brentq
from scipy.spatial import cKDTree

from simsopt.mhd.boozer import Boozer
from simsopt.mhd import Vmec
from simsopt._core.optimizable import Optimizable
from simsopt._core.util import Struct
from simsopt._core.types import RealArray

logger = logging.getLogger(__name__)


def chebgauss2(deg):
    """Chebyshev-Gauss quadrature of the second kind on [-1, 1]."""
    k = np.arange(1, deg + 1)
    x = np.cos(np.pi * k / (deg + 1))
    w = np.pi / (deg + 1) * np.sin(np.pi * k / (deg + 1))
    return x, w


def _find_bounce_points_vectorized(B_spline, pitch_inv, zeta, all_wells=False):
    """Vectorized bounce point detection for multiple pitch values."""
    from scipy.optimize import brentq
    
    B_grid = B_spline(zeta)
    n_pitch = len(pitch_inv)
    
    # Pre-allocate results
    all_wells_list = [[] for _ in range(n_pitch)]
    
    for i, pi in enumerate(pitch_inv):
        diff = B_grid - pi
        sign_changes = np.where(diff[:-1] * diff[1:] < 0)[0]
        
        crossings = []
        for idx in sign_changes:
            try:
                z_root = brentq(lambda z: B_spline(z) - pi,
                              zeta[idx], zeta[idx+1], xtol=1e-12)
                dBdz = float(B_spline(z_root, 1))
                crossings.append((z_root, dBdz))
            except (ValueError, RuntimeError):
                continue
        
        # Pair crossings into wells
        wells = []
        j = 0
        while j < len(crossings) - 1:
            z1, dB1 = crossings[j]
            z2, dB2 = crossings[j+1]
            if dB1 <= 0 and dB2 >= 0 and (z2 - z1) > 1e-6:
                wells.append((z1, z2))
                j += 2
            else:
                j += 1
        
        all_wells_list[i] = wells
    
    return all_wells_list


class EffectiveRipple(Optimizable):
    """Compute the effective ripple ε_eff on a set of flux surfaces."""
    
    def __init__(
        self,
        vmec,
        surfaces: Union[float, RealArray],
        booz_mpol: int = 48,
        booz_ntor: int = 48,
        num_transit: int = 40,
        num_alpha: int = 8,  # Increased for better statistics
        num_fieldline_points: int = 256,  # Increased resolution
        num_pitch: int = 64,
        num_well_quad: int = 32,
        ntheta_grad: int = 80,
        nphi_grad: int = 80,
        use_vectorization: bool = True,
    ) -> None:
        self.vmec = vmec
        try:
            self.surfaces = list(surfaces)
        except TypeError:
            self.surfaces = [surfaces]
        self.booz_mpol = booz_mpol
        self.booz_ntor = booz_ntor
        self.num_transit = num_transit
        self.num_alpha = num_alpha
        self.num_fieldline_points = num_fieldline_points
        self.num_pitch = num_pitch
        self.num_well_quad = num_well_quad
        self.ntheta_grad = ntheta_grad
        self.nphi_grad = nphi_grad
        self.use_vectorization = use_vectorization
        super().__init__(depends_on=[vmec])
    
    def compute(self):
        """Compute ε_eff on all requested surfaces."""
        self.vmec.run()
        wout = self.vmec.wout
        nfp = wout.nfp
        ns = wout.ns
        
        s_half = self.vmec.s_half_grid
        surf_indices = [int(np.argmin(np.abs(s_half - s))) for s in self.surfaces]
        
        booz = Boozer(self.vmec, self.booz_mpol, self.booz_ntor)
        booz.register(s_half[1:])
        booz.run()
        bx = booz.bx
        
        m0n0 = np.where((wout.xm == 0) & (wout.xn == 0))[0][0]
        R0 = wout.rmnc[m0n0, 0]
        
        avg_grad_s_dict = self._compute_avg_grad_s(wout, surf_indices)
        
        eps_eff_32_arr = np.zeros(len(self.surfaces))
        for i, (s, si) in enumerate(zip(self.surfaces, surf_indices)):
            avg_grad_rho = avg_grad_s_dict[si] / (2.0 * np.sqrt(s_half[si]))
            eps_eff_32_arr[i] = self._compute_surface_vectorized(
                bx, i, s_half[si], R0, avg_grad_rho, nfp,
            )
        
        results = Struct()
        results.surfaces = np.array(self.surfaces)
        results.eps_eff_32 = eps_eff_32_arr
        results.eps_eff = np.abs(eps_eff_32_arr) ** (2.0 / 3.0)
        return results
    
    def _compute_avg_grad_s(self, wout, surf_indices=None):
        """⟨|∇s|⟩ computation (vectorized)."""
        ns = wout.ns
        nfp = wout.nfp
        ntheta = self.ntheta_grad
        nphi = self.nphi_grad
        
        theta = np.linspace(0, 2 * np.pi, ntheta, endpoint=False)
        phi = np.linspace(0, 2 * np.pi / nfp, nphi, endpoint=False)
        phi2d, theta2d = np.meshgrid(phi, theta)
        
        xm = wout.xm
        xn = wout.xn
        xm_nyq = wout.xm_nyq
        xn_nyq = wout.xn_nyq
        
        s_full = np.linspace(0, 1, ns)
        
        angle = (xm[:, None, None] * theta2d[None, :, :]
                 - xn[:, None, None] * phi2d[None, :, :])
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        
        angle_nyq = (xm_nyq[:, None, None] * theta2d[None, :, :]
                     - xn_nyq[:, None, None] * phi2d[None, :, :])
        cos_a_nyq = np.cos(angle_nyq)
        
        if surf_indices is None:
            surf_indices = list(range(ns - 1))
        
        avg_grad_s = {}
        for js in surf_indices:
            s_val = self.vmec.s_half_grid[js]
            idx_right = np.searchsorted(s_full, s_val).clip(1, ns - 1)
            idx_left = idx_right - 1
            w = (s_val - s_full[idx_left]) / (s_full[idx_right] - s_full[idx_left])
            
            rmnc_h = (1 - w) * wout.rmnc[:, idx_left] + w * wout.rmnc[:, idx_right]
            zmns_h = (1 - w) * wout.zmns[:, idx_left] + w * wout.zmns[:, idx_right]
            
            R = np.einsum('m,mij->ij', rmnc_h, cos_a)
            Rt = np.einsum('m,mij->ij', -xm * rmnc_h, sin_a)
            Rp = np.einsum('m,mij->ij', xn * rmnc_h, sin_a)
            Zt = np.einsum('m,mij->ij', xm * zmns_h, cos_a)
            Zp = np.einsum('m,mij->ij', -xn * zmns_h, cos_a)
            
            cross_mag = np.sqrt(
                (Rt * Zp - Zt * Rp) ** 2 + R ** 2 * (Rt ** 2 + Zt ** 2)
            )
            sqrtg = np.einsum(
                'm,mij->ij', wout.gmnc[:, js + 1], cos_a_nyq
            )
            
            num = np.sum(cross_mag)
            den = np.sum(np.abs(sqrtg))
            avg_grad_s[js] = num / den if den > 0 else 1.0
        
        return avg_grad_s
    
    def _compute_surface_vectorized(self, bx_obj, booz_idx, s, R0, avg_grad_rho, nfp):
        """Single-surface computation with vectorized operations."""
        bmnc = bx_obj.bmnc_b[:, booz_idx]
        xm = bx_obj.xm_b
        xn = bx_obj.xn_b
        G = bx_obj.Boozer_G[booz_idx]
        I_b = bx_obj.Boozer_I[booz_idx]
        iota = float(np.interp(s, bx_obj.s_in, bx_obj.iota))
        GpI = G + iota * I_b
        psi_edge = np.abs(self.vmec.wout.phi[-1]) / (2.0 * np.pi)
        dpsi_drho = 2.0 * np.sqrt(s) * psi_edge
        
        nz = self.num_fieldline_points * self.num_transit
        zeta_range = 2.0 * np.pi * self.num_transit
        
        alphas = np.linspace(0, 2 * np.pi, self.num_alpha, endpoint=False)
        miot_minus_n = xm * iota - xn
        
        B0 = self._compute_B0(bmnc, xm, xn, nfp)
        xq, wq = chebgauss2(self.num_well_quad)
        
        total_integral = 0.0
        
        # Pre-compute field line data for all alphas
        for alpha in alphas:
            zeta_shift = _find_B_max_shift(
                bmnc, xm, miot_minus_n, alpha, nfp, iota
            )
            
            zeta = np.linspace(
                zeta_shift, zeta_shift + zeta_range, nz, endpoint=False
            )
            dz = zeta[1] - zeta[0]
            
            # Vectorized computation of B and derivatives for all zeta
            angle = xm[:, None] * alpha + miot_minus_n[:, None] * zeta[None, :]
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)
            
            B = bmnc @ cos_a
            dBdz_fl = -(miot_minus_n * bmnc) @ sin_a
            
            dBda = -(xm * bmnc) @ sin_a
            K = I_b * dBdz_fl - GpI * dBda
            grad_rho_kappag = -K / (GpI * dpsi_drho)
            e_zeta = GpI / B
            
            fl_length = np.sum(GpI / B ** 2) * dz
            
            # Create splines
            B_spline = CubicHermiteSpline(zeta, B, dBdz_fl)
            grkG_spline = CubicSpline(zeta, grad_rho_kappag)
            ez_spline = CubicSpline(zeta, e_zeta)
            
            # Pitch quadrature
            B_min_fl = np.min(B)
            B_max_fl = np.max(B)
            
            # Vectorized pitch values
            mu_vals = np.linspace(B_min_fl, B_max_fl, self.num_pitch + 2)[1:-1]
            dmu = (B_max_fl - B_min_fl) / (self.num_pitch + 1)
            
            # Find bounce points for all pitch values
            all_wells = _find_bounce_points_vectorized(B_spline, mu_vals, zeta)
            
            # Vectorized computation of well integrals
            pitch_sum = 0.0
            for mu_idx, wells in enumerate(all_wells):
                if not wells:
                    continue
                
                mu = mu_vals[mu_idx]
                lam = 1.0 / mu
                
                # Process all wells for this mu
                well_sum = 0.0
                for (z1, z2) in wells:
                    if z2 - z1 < 1e-6:
                        continue
                    
                    z_half = 0.5 * (z2 - z1)
                    z_mid = 0.5 * (z1 + z2)
                    zq = z_mid + z_half * xq
                    
                    # Vectorized evaluation at quadrature points
                    B_q = B_spline(zq)
                    grkG_q = grkG_spline(zq)
                    ez_q = ez_spline(zq)
                    
                    one_m_lB = np.maximum(1.0 - lam * B_q, 1e-12)
                    sqrt_f = np.sqrt(one_m_lB)
                    lB_q = lam * B_q
                    
                    dH = sqrt_f * (4.0 / lB_q - 1.0) * grkG_q / B_q * ez_q
                    dI = sqrt_f / B_q * ez_q
                    
                    H_val = z_half * np.sum(wq * dH)
                    I_val = z_half * np.sum(wq * dI)
                    
                    if abs(I_val) > 1e-12:
                        well_sum += H_val ** 2 / I_val
                
                pitch_sum += well_sum * dmu / mu ** 3
            
            total_integral += pitch_sum / fl_length
        
        total_integral /= len(alphas)
        
        eps_eff_32 = (
            (np.pi / (8.0 * np.sqrt(2.0)))
            * (B0 * R0 / avg_grad_rho) ** 2
            * total_integral
        )
        
        return eps_eff_32
    
    @staticmethod
    def _compute_B0(bmnc, xm, xn, nfp):
        nth, nph = 128, 128
        theta = np.linspace(0, 2 * np.pi, nth, endpoint=False)
        phi = np.linspace(0, 2 * np.pi / nfp, nph, endpoint=False)
        phi2d, theta2d = np.meshgrid(phi, theta)
        angle = (xm[:, None, None] * theta2d[None, :, :]
                 - xn[:, None, None] * phi2d[None, :, :])
        B2d = np.einsum('m,mij->ij', bmnc, np.cos(angle))
        return float(np.max(B2d))


def _find_B_max_shift(bmnc, xm, miot_minus_n, alpha, nfp, iota):
    """Find ζ near 0 where B(α, ζ) has a local maximum."""
    search_range = max(2 * np.pi / nfp, 2 * np.pi / (abs(iota) * nfp))
    zeta_coarse = np.linspace(0, search_range, 256, endpoint=False)
    angle = xm[:, None] * alpha + miot_minus_n[:, None] * zeta_coarse[None, :]
    B_coarse = bmnc @ np.cos(angle)
    return zeta_coarse[np.argmax(B_coarse)]


def compute_effective_ripple(
    wout_filename,
    surfaces=None,
    booz_mpol=48,
    booz_ntor=48,
    num_transit=40,
    num_alpha=4,
    num_fieldline_points=128,
    num_pitch=64,
    num_well_quad=32,
):
    """Compute ε_eff from a VMEC wout file."""
    vmec = Vmec(wout_filename)
    if surfaces is None:
        surfaces = np.linspace(0.1, 1.0, 10)
    
    er = EffectiveRipple(
        vmec, surfaces,
        booz_mpol=booz_mpol,
        booz_ntor=booz_ntor,
        num_transit=num_transit,
        num_alpha=num_alpha,
        num_fieldline_points=num_fieldline_points,
        num_pitch=num_pitch,
        num_well_quad=num_well_quad,
    )
    return er.compute()


if __name__ == "__main__":
    import sys
    import time
    
    fname = sys.argv[1] if len(sys.argv) > 1 else "wout_final.nc"
    surfaces = np.linspace(0.1, 1.0, 10)
    
    print(f"Computing effective ripple (optimized) from {fname}")
    t0 = time.time()
    results = compute_effective_ripple(fname, surfaces)
    elapsed = time.time() - t0
    
    print(f"\nComputed in {elapsed:.1f}s")
    print("\n  s        eps_eff^{3/2}     eps_eff")
    print("-" * 42)
    for s, e32, e in zip(results.surfaces, results.eps_eff_32, results.eps_eff):
        print(f"  {s:.3f}    {e32:12.6e}    {e:12.6e}")
    
    np.save('s.npy', results.surfaces)
    np.save('eps_eff32.npy', results.eps_eff_32)
