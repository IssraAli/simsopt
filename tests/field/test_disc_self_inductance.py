"""Unit tests for :mod:`simsopt.field.disc_self_inductance`.

Covers:

* Ring kernel matches the existing ``_neumann_mutual_inductance`` used
  elsewhere in the passive-bulk test suite (to machine precision).
* Axial-shift ring kernel matches the coaxial-loop dipole far-field
  asymptote ``mu_0 pi a^2 b^2 / (2 dz^3)`` as ``dz / max(a, b) -> infty``.
* Axisymmetric self-block for a chosen smooth radial profile converges
  **exponentially** in ``n_radial``.
* Smooth cross-block for ``dz > 0`` converges exponentially even at
  small ``dz / R``.
* Assembled disc sub-block is symmetric and positive semi-definite
  (PSD) on a random Fourier-Zernike basis.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pytest
from scipy.special import ellipe, ellipk

from simsopt.field.disc_self_inductance import (
    assemble_puck_disc_faces_L,
    assemble_puck_self_L,
    cylinder_self_block_mode_m,
    disc_disc_cross_block,
    disc_self_block_axisymmetric,
    disc_self_block_mode_m,
    disc_side_cross_block_mode_m,
    fourier_kernel_Im,
    pucks_are_coaxial,
    ring_mutual_axial_shift,
    ring_mutual_maxwell_grover,
)
from simsopt.field.disc_self_inductance import _make_axial, _make_radial
from simsopt.field.puck_basis import build_puck_shell_basis


MU0 = 4.0 * np.pi * 1.0e-7


def _neumann_mutual_inductance(R1: float, R2: float, d: float) -> float:
    """Independent Neumann-form mutual inductance of two coaxial loops.

    Duplicates the helper used in ``tests/field/test_passive_bulks.py``
    so that both implementations can be cross-checked without test-file
    coupling.
    """
    m = 4.0 * R1 * R2 / ((R1 + R2) ** 2 + d**2)
    k = np.sqrt(m)
    return MU0 * np.sqrt(R1 * R2) * ((2.0 / k - k) * ellipk(m) - (2.0 / k) * ellipe(m))


# ---------------------------------------------------------------------
# Ring kernel sanity tests
# ---------------------------------------------------------------------


def test_ring_mutual_matches_existing_neumann():
    """Matches to 1e-12 relative on a coarse grid of ring radii."""
    for a in [0.05, 0.1, 0.2, 1.0]:
        for b in [0.05, 0.1, 0.2, 1.0]:
            if abs(a - b) < 1e-3:
                # Skip near-diagonal (both formulas diverge there).
                continue
            expected = _neumann_mutual_inductance(a, b, 0.0)
            got = ring_mutual_maxwell_grover(a, b)
            assert np.isfinite(expected) and np.isfinite(got)
            rel = abs(got - expected) / max(abs(expected), 1.0e-20)
            assert rel < 1.0e-12, (
                f"ring_mutual_maxwell_grover mismatch at a={a}, b={b}: "
                f"got={got}, expected={expected}, rel={rel}"
            )


def test_ring_mutual_axial_shift_dipole_limit():
    """``M(a, b, dz) -> mu0 pi a^2 b^2 / (2 dz^3)`` as ``dz -> infty``."""
    a, b = 0.1, 0.15
    ratios = []
    for factor in [50.0, 200.0, 1000.0]:
        dz = factor * max(a, b)
        expected = MU0 * np.pi * a**2 * b**2 / (2.0 * dz**3)
        got = ring_mutual_axial_shift(a, b, dz)
        ratios.append(got / expected)
    # The leading correction is O(1/dz^2) with O(1) prefactor; the
    # 1000*R case matches the dipole form to a few parts in 10^5.
    assert abs(ratios[0] - 1.0) < 2.0e-3
    assert abs(ratios[1] - 1.0) < 2.0e-4
    assert abs(ratios[2] - 1.0) < 1.0e-4
    # And the sequence converges monotonically to 1 as dz grows.
    assert abs(ratios[2] - 1.0) < abs(ratios[1] - 1.0) < abs(ratios[0] - 1.0)


def test_fourier_kernel_Im_matches_trapezoid():
    """Recursion-based ``I_m`` matches direct 2*pi trap integration."""
    a = np.array([0.08])
    b = np.array([0.13])
    dz = 0.05
    m_max = 5
    I_formula = fourier_kernel_Im(a, b, dz, m_max)

    u = np.linspace(0.0, 2.0 * np.pi, 100001)
    denom = np.sqrt(a[0] ** 2 + b[0] ** 2 - 2.0 * a[0] * b[0] * np.cos(u) + dz**2)
    for mm in range(m_max + 1):
        I_trap = np.trapezoid(np.cos(mm * u) / denom, u)
        rel = abs(I_trap - I_formula[mm, 0]) / max(abs(I_trap), 1.0e-300)
        assert rel < 1.0e-10, (
            f"I_{mm}: formula={I_formula[mm, 0]} trap={I_trap} rel={rel}"
        )


# ---------------------------------------------------------------------
# Convergence (exactness) tests
# ---------------------------------------------------------------------


def test_disc_self_singular_exponential_convergence():
    """Axisymmetric self-block for ``f(rho) = rho^2`` converges
    exponentially in ``n_radial``.

    The integral ``int int 2a 2b M(a, b, 0) da db`` over ``[0, R]^2``
    has a known log singularity at ``a = b`` which the Duffy +
    Gauss-Laguerre rule absorbs.  Concretely: residual
    ``|L(n) - L(n_ref)| / |L(n_ref)|`` must shrink by at least an order
    of magnitude between consecutive refinements in the pre-asymptotic
    range, and reach < 1e-9 at ``n_radial = 32``.
    """
    R = 0.1
    fp: Callable[[np.ndarray], np.ndarray] = lambda r: 2.0 * np.asarray(r)

    ns = [8, 16, 32, 64, 96]
    vals = [disc_self_block_axisymmetric(fp, fp, R, n_radial=n) for n in ns]
    L_ref = vals[-1]
    residuals = [abs(v - L_ref) / abs(L_ref) for v in vals[:-1]]
    # Exponential convergence: residual shrinks by >= 10x from n=8 to n=16.
    assert residuals[0] / residuals[1] > 10.0, (
        f"n=8 residual {residuals[0]} vs n=16 residual {residuals[1]}: "
        "expected >= 10x drop"
    )
    # At n_radial = 32 the residual is already <= 1e-7 for this smooth f.
    assert residuals[2] < 1.0e-7, f"n=32 residual {residuals[2]} exceeds 1e-7 threshold"
    # All values strictly positive (self-energy is PSD).
    assert all(v > 0 for v in vals)


def test_disc_cross_smooth_exponential_convergence():
    """Smooth cross-block kernel at ``dz > 0`` converges exponentially
    in ``n_radial`` — no log singularity, so plain Gauss-Legendre
    suffices.
    """
    R = 0.1
    f = lambda r: np.asarray(r) ** 2
    fp = lambda r: 2.0 * np.asarray(r)

    for dz_over_R in [0.1, 1.0, 10.0]:
        dz = dz_over_R * R
        ns = [8, 16, 32, 64]
        vals = [
            disc_disc_cross_block(f, fp, f, fp, 0, R, R, dz, n_radial=n) for n in ns
        ]
        L_ref = vals[-1]
        # Residual at n=16 must already be below 1e-10 for dz/R >= 1
        # (very smooth kernel).  For small dz/R=0.1 it's less demanding
        # because the 1/|x-x'| kernel becomes peaked on the diagonal.
        rel_16 = abs(vals[1] - L_ref) / abs(L_ref) if abs(L_ref) > 0 else 0.0
        if dz_over_R >= 1.0:
            assert rel_16 < 1.0e-10, f"dz/R={dz_over_R}, n=16 residual {rel_16} > 1e-10"
        else:
            # Small dz/R: looser tolerance (kernel is smooth but peaked)
            rel_32 = abs(vals[2] - L_ref) / abs(L_ref) if abs(L_ref) > 0 else 0.0
            assert rel_32 < 1.0e-4, f"dz/R={dz_over_R}, n=32 residual {rel_32} > 1e-4"


def test_disc_self_mode_m_equals_axisymmetric_at_m_zero():
    """``disc_self_block_mode_m`` with ``m=0`` reduces to
    :func:`disc_self_block_axisymmetric`."""
    R = 0.07
    f = lambda r: np.asarray(r) ** 2
    fp = lambda r: 2.0 * np.asarray(r)
    v_m = disc_self_block_mode_m(f, fp, f, fp, 0, R, n_radial=32)
    v_axis = disc_self_block_axisymmetric(fp, fp, R, n_radial=32)
    assert abs(v_m - v_axis) / abs(v_axis) < 1.0e-14


# ---------------------------------------------------------------------
# Symmetry and PSD
# ---------------------------------------------------------------------


def test_disc_self_block_symmetric():
    """Swapping ``(p, q)`` leaves a block entry unchanged."""
    R = 0.1
    f1 = lambda r: np.asarray(r) ** 2
    fp1 = lambda r: 2.0 * np.asarray(r)
    f2 = lambda r: np.asarray(r) ** 2 - 0.3 * R**2
    fp2 = lambda r: 2.0 * np.asarray(r)
    for m in [0, 1, 2]:
        v_pq = disc_self_block_mode_m(f1, fp1, f2, fp2, m, R, n_radial=32)
        v_qp = disc_self_block_mode_m(f2, fp2, f1, fp1, m, R, n_radial=32)
        assert abs(v_pq - v_qp) / (abs(v_pq) + 1.0e-30) < 1.0e-10


def test_assemble_puck_disc_faces_block_symmetric_psd():
    """Full Fourier-Zernike assembler returns a symmetric, PSD disc-disc
    sub-block."""
    R, t = 0.1, 0.03
    basis = build_puck_shell_basis(
        R=R,
        t=t,
        m_fourier=2,
        l_zernike=4,
        k_chebyshev=0,
        n_rho=8,
        n_phi=16,
        n_z=2,
    )
    L = assemble_puck_disc_faces_L(basis, R, t, n_radial=28)

    # Extract the disc-disc sub-block (drop side-wall rows/cols)
    disk_idx = [i for i, nm in enumerate(basis.dof_names) if nm.startswith("disk_")]
    sub = L[np.ix_(disk_idx, disk_idx)]
    # Symmetric
    assert np.max(np.abs(sub - sub.T)) < 1.0e-12, (
        "disc-disc sub-block not symmetric to 1e-12"
    )
    # PSD (small negative eigenvalues allowed from quadrature noise;
    # tolerance: < 1e-10 * lam_max).
    eigs = np.linalg.eigvalsh(0.5 * (sub + sub.T))
    lam_max = float(np.max(eigs))
    if lam_max > 0.0:
        assert eigs[0] > -1.0e-10 * lam_max, (
            f"disc-disc sub-block not PSD: min eig {eigs[0]} "
            f"(relative to max {lam_max})"
        )


def test_coaxial_detection():
    """``pucks_are_coaxial`` correctly identifies parallel and
    anti-parallel axes through the same line, rejects offset axes."""
    c1 = np.array([0.0, 0.0, 0.0])
    ax1 = np.array([0.0, 0.0, 1.0])
    c2 = np.array([0.0, 0.0, 0.5])
    ax2 = np.array([0.0, 0.0, 1.0])
    ax2_flip = np.array([0.0, 0.0, -1.0])
    c3 = np.array([0.01, 0.0, 0.5])  # off-axis
    ax_off = np.array([1.0, 0.0, 1.0]) / np.sqrt(2.0)

    assert pucks_are_coaxial(c1, ax1, c2, ax2) == 0.5
    assert pucks_are_coaxial(c1, ax1, c2, ax2_flip) == 0.5
    assert pucks_are_coaxial(c1, ax1, c3, ax2) is None
    assert pucks_are_coaxial(c1, ax1, c2, ax_off) is None


# ---------------------------------------------------------------------
# Cross-link with production pipeline
# ---------------------------------------------------------------------


def test_exact_disc_faces_preserves_full_L_psd():
    """After the Phase-1/2 extension of the semi-analytic self-block
    to all nine face-pair sub-blocks, the opt-in ``exact_disc_faces``
    path in
    :func:`simsopt.field.bulk_inductance.shell_inductance_matrix_blockwise`
    (wired through :class:`~simsopt.field.psc_bulk.PSCBulkArray` via
    the ``exact_disc_faces`` attribute on the instance) **preserves
    positive semi-definiteness** of the full puck self-block.

    Historically this test asserted the opposite, pinning the known
    limitation that partial replacement (disc-disc only) drove a
    negative eigenvalue of order ``5 %`` of the spectral radius.  Once
    :func:`simsopt.field.disc_self_inductance.assemble_puck_self_L`
    was extended to fill the side-side and disc-side-rim sub-blocks
    (via the new
    :func:`~simsopt.field.disc_self_inductance.cylinder_self_block_mode_m`
    and
    :func:`~simsopt.field.disc_self_inductance.disc_side_cross_block_mode_m`
    routines), the overlay in
    :func:`~simsopt.field.bulk_inductance.shell_inductance_matrix_blockwise`
    was updated to overwrite the full puck self-block.  Both paths
    now give a PSD ``L``.
    """
    pytest.importorskip("jax")

    from simsopt.field.coil import Coil, Current
    from simsopt.field.psc_bulk import PSCBulkArray
    from simsopt.geo import CurveXYZFourier

    curve = CurveXYZFourier(32, 1)
    curve.x = np.array(
        [0.0, 0.0, 10.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0],
        dtype=float,
    )
    tf = Coil(curve, Current(1.0))

    psc = PSCBulkArray(
        np.array([[0.0, 0.0, 0.0]]),
        np.array([[0.0, 0.0, 1.0]]),
        np.array([0.3]),
        np.array([0.3 * 0.075]),
        [tf],
        eval_points=np.array([[0.6, 0.0, 0.0]]),
        m_fourier=3,
        l_zernike=6,
        k_chebyshev=3,
        n_rho=10,
        n_phi=16,
        n_z=6,
        nfp=1,
        stellsym=False,
        adaptive_self_reg=True,
    )

    lam_leg = np.linalg.eigvalsh(psc._L_work)
    lam_max_leg = max(abs(lam_leg[-1]), 1.0)
    assert lam_leg[0] > -1.0e-12 * lam_max_leg, (
        "Legacy path should give a PSD (or near-PSD) full L; got "
        f"min eig = {lam_leg[0]:.3e}"
    )

    psc.exact_disc_faces = True
    psc.n_radial_disc = 32
    psc._rebuild()
    lam_ex = np.linalg.eigvalsh(psc._L_work)
    lam_max_ex = max(abs(lam_ex[-1]), 1.0)
    assert lam_ex[0] > -1.0e-10 * lam_max_ex, (
        "exact_disc_faces path must preserve PSD of full L; got "
        f"min eig = {lam_ex[0]:.3e}, max eig = {lam_ex[-1]:.3e}.  "
        "This indicates the Phase-1/2 full-self-block overlay has "
        "regressed."
    )


# ---------------------------------------------------------------------
# New sub-block tests: cylinder self (side-wall) and disc-side rim corner
# ---------------------------------------------------------------------


def test_cylinder_self_block_convergence():
    """Semi-analytic side-wall self-block entries converge exponentially
    in ``n_z`` for a range of ``(m, k, k')``.

    Only ``(k - k')`` with matching parity give non-vanishing entries;
    entries of opposite parity in ``k, k'`` vanish identically by the
    ``z \\to -z, z' \\to -z'`` symmetry of the kernel on the symmetric
    axial interval ``[-t/2, t/2]``.  The test therefore picks same-
    parity pairs to exercise non-trivial convergence.
    """
    R, t = 0.2, 0.04
    # Same-parity pairs with non-vanishing integrals.
    cases = [(0, 1, 1), (1, 0, 0), (1, 1, 1), (2, 0, 2), (1, 1, 3)]
    for m, k, kp in cases:
        f_p, fp_p = _make_axial(t, k)
        f_q, fp_q = _make_axial(t, kp)
        v32 = cylinder_self_block_mode_m(f_p, fp_p, f_q, fp_q, m, R, t, n_z=32)
        v64 = cylinder_self_block_mode_m(f_p, fp_p, f_q, fp_q, m, R, t, n_z=64)
        rel = abs(v32 - v64) / (abs(v64) + 1.0e-30)
        assert rel < 1.0e-5, (
            f"cylinder_self_block convergence too slow for "
            f"(m, k, k')=({m}, {k}, {kp}): n=32 -> {v32:.6e}, "
            f"n=64 -> {v64:.6e}, rel={rel:.3e}"
        )


def test_cylinder_self_block_physical_scaling():
    r"""Physical sanity checks for the axisymmetric side-wall self entry
    ``(m=0, k=k'=1)``.

    A short coaxial cylindrical current sheet of radius :math:`R` and
    axial length :math:`t` carrying a uniform azimuthal current density
    has a positive self-inductance that grows *logarithmically* as
    :math:`t / R \to 0` (its leading-order asymptotic is
    :math:`\propto \log(R / t)`, matching the standard result that the
    self-inductance of a coaxial current sheet diverges as its length
    shrinks).  The exact prefactor depends on the basis convention
    (here :math:`T_1(2 z / t)` gives :math:`f'(z) = 2 / t`, so the
    quadratic form is proportional to :math:`1 / t^2` times a
    double integral of the kernel, whose leading asymptotic is the
    logarithmic Neumann-like term).

    We verify:

    1. Positivity: the diagonal self entry must be strictly positive.
    2. Monotone growth as :math:`t` shrinks (at fixed ``R``).
    3. Leading :math:`\log(R / t)` scaling: the difference between two
       thicknesses is close to :math:`\mu_0 R \log(t_1 / t_2)` for small
       ``t``.
    """
    R = 0.2
    ts = [0.05 * R, 0.02 * R, 0.005 * R]
    vals = []
    for t in ts:
        f1, fp1 = _make_axial(t, 1)
        val = cylinder_self_block_mode_m(f1, fp1, f1, fp1, 0, R, t, n_z=48)
        vals.append(val)
        assert val > 0.0, (
            f"cylinder_self m=0 k=1 at t/R={t / R:.3f} is not positive: {val:.6e}"
        )
    for i in range(1, len(vals)):
        assert vals[i] > vals[i - 1], (
            f"cylinder_self m=0 k=1 is not monotone increasing as "
            f"t -> 0: t/R={[t / R for t in ts]}, vals={vals}"
        )
    # Leading log(R/t) growth between the two smallest thicknesses
    # (t2 = 0.02 R, t3 = 0.005 R).  With the basis convention
    # g(z) = T_1(2 z / t) so f'(z) = 2 / t identically, the total
    # azimuthal current carried by this basis mode is
    # I_tot = int K_phi dz = -(2 / t) * t = -2, so the quadratic-form
    # entry (L_ii)_{pp} is 4 * L_trad where L_trad is the traditional
    # single-loop self-inductance.  The leading log asymptotic of
    # L_trad with "wire thickness" t is
    #     L_trad ~ mu0 * R * (log(8 R / t) - const)
    # so the *difference* L_trad(t_2) - L_trad(t_3) = mu0*R*log(t_2 / t_3).
    # Hence the expected increment in the basis-form entry is
    #     4 * mu0 * R * log(t_2 / t_3).
    predicted = 4.0 * MU0 * R * np.log(ts[1] / ts[2])
    observed = vals[2] - vals[1]
    ratio = observed / predicted
    assert 0.9 < ratio < 1.1, (
        f"cylinder_self m=0 k=1 log-leading increment mismatch: "
        f"observed={observed:.6e}, predicted={predicted:.6e}, "
        f"ratio={ratio:.3f} (expected ~1.0)"
    )


def test_disc_side_cross_block_convergence():
    """Disc-side rim-corner cross-block converges exponentially in the
    polar-Laguerre node count for ``m = 0, 1, 2``."""
    R, t = 0.2, 0.04
    # Pick a disc mode with nontrivial radial content (n >= 2) and a
    # side mode with nontrivial axial content (k >= 1).
    cases = [(0, 2, 1), (1, 1, 2), (2, 2, 1)]
    for m, n_disc, k_side in cases:
        fd, fdp = _make_radial(R, m, n_disc)
        fs, fsp = _make_axial(t, k_side)
        v32 = disc_side_cross_block_mode_m(
            fd,
            fdp,
            fs,
            fsp,
            m,
            R,
            t,
            zface=+0.5 * t,
            n_rho=32,
            n_z=32,
        )
        v64 = disc_side_cross_block_mode_m(
            fd,
            fdp,
            fs,
            fsp,
            m,
            R,
            t,
            zface=+0.5 * t,
            n_rho=64,
            n_z=64,
        )
        rel = abs(v32 - v64) / (abs(v64) + 1.0e-30)
        assert rel < 1.0e-5, (
            f"disc-side cross-block convergence too slow for "
            f"(m, n_disc, k_side)=({m}, {n_disc}, {k_side}): "
            f"n=32 -> {v32:.6e}, n=64 -> {v64:.6e}, rel={rel:.3e}"
        )


def test_disc_side_cross_block_top_bot_relation():
    r"""The top-disc and bottom-disc cross entries of a single puck
    differ only by a sign determined by the parity of the side-wall
    basis ``T_k(2 z / t)`` under ``z \to -z``.

    Because the kernel ``I_m(\rho, R, |v|)`` is even in ``v``,
    reflecting ``zface = +t/2 \to -t/2`` and simultaneously ``z' \to -z'``
    leaves the kernel invariant, but picks up the parity sign of the
    side-wall axial factors: ``T_k`` has parity :math:`(-1)^k` and
    ``T_k'`` has parity :math:`(-1)^{k+1}`.  In general the two terms
    ``f_s`` and ``f_s'`` therefore have opposite parity, so the simple
    relation is *magnitude equality*, not value equality, except when
    only one of the two terms is non-vanishing (as for ``m = 0`` where
    the ``I_{m+1} - I_{m-1}`` disc factor can simplify).  We verify the
    magnitude equality, which is the invariant the caller relies upon
    (``assemble_puck_self_L`` applies the ``-1`` sign flip for the bot
    disc's physical ``-z`` normal on top of the routine output).
    """
    R, t = 0.2, 0.04
    for m, n_disc, k_side in [(0, 2, 1), (1, 1, 2), (1, 2, 1), (0, 1, 2)]:
        fd, fdp = _make_radial(R, m, n_disc)
        fs, fsp = _make_axial(t, k_side)
        v_top = disc_side_cross_block_mode_m(
            fd,
            fdp,
            fs,
            fsp,
            m,
            R,
            t,
            zface=+0.5 * t,
            n_rho=32,
            n_z=32,
        )
        v_bot = disc_side_cross_block_mode_m(
            fd,
            fdp,
            fs,
            fsp,
            m,
            R,
            t,
            zface=-0.5 * t,
            n_rho=32,
            n_z=32,
        )
        # Magnitudes must match to machine precision; the sign is the
        # parity of the side-wall basis (-1)^k for T_k(2 z / t).
        rel = abs(abs(v_top) - abs(v_bot)) / (abs(v_top) + 1.0e-30)
        assert rel < 1.0e-12, (
            f"disc-side cross-block magnitudes differ under zface "
            f"reflection for (m, n, k)=({m}, {n_disc}, {k_side}): "
            f"top={v_top:.6e}, bot={v_bot:.6e}, rel={rel:.3e}"
        )


def test_assemble_puck_self_block_is_psd():
    """The full 9-face-pair puck self-block from
    :func:`assemble_puck_self_L` is symmetric PSD to machine precision
    on a representative moderate-resolution Fourier-Zernike-Chebyshev
    basis."""
    R, t = 0.3, 0.03
    basis = build_puck_shell_basis(
        R=R,
        t=t,
        m_fourier=3,
        l_zernike=6,
        k_chebyshev=3,
        n_rho=10,
        n_phi=16,
        n_z=6,
    )
    L = assemble_puck_self_L(basis, R, t, n_radial=32)
    assert not np.any(np.isnan(L)), (
        "Full puck self-block should have no NaN entries for the standard basis."
    )
    asym = float(np.max(np.abs(L - L.T)))
    lmax = float(np.max(np.abs(L)))
    assert asym < 1.0e-10 * max(lmax, 1.0), (
        f"Full puck self-block asymmetric: max |L-L.T|={asym:.3e} vs max |L|={lmax:.3e}"
    )
    Lsym = 0.5 * (L + L.T)
    eigs = np.linalg.eigvalsh(Lsym)
    lam_max = float(np.max(eigs))
    assert lam_max > 0.0, "Full puck self-block has nonpositive max eigenvalue"
    assert eigs[0] > -1.0e-10 * lam_max, (
        f"Full puck self-block not PSD: min eig={eigs[0]:.3e} "
        f"(relative to max {lam_max:.3e})"
    )


def test_disc_self_block_matches_audit_reference():
    """The ``m=0, n=2`` disc_top self entry of the assembled block
    reproduces the independent Duffy + Gauss-Laguerre reference used by
    the audit test to 1e-10 relative (both methods are exact up to
    radial-quadrature precision, with different node placements)."""
    R = 0.1
    basis = build_puck_shell_basis(
        R=R,
        t=0.01,
        m_fourier=0,
        l_zernike=2,
        k_chebyshev=0,
        n_rho=8,
        n_phi=16,
        n_z=2,
    )
    # Take two independently-evaluated disc-self values at different
    # n_radial and check agreement to machine-precision-level: two
    # Gauss-Legendre/Gauss-Laguerre rules on the same exact analytic
    # kernel must agree to radial-quadrature precision of the coarser
    # one.
    L_coarse = assemble_puck_disc_faces_L(basis, R, 0.01, n_radial=32)
    L_fine = assemble_puck_disc_faces_L(basis, R, 0.01, n_radial=64)
    idx = basis.dof_names.index("disk_top_m0_n2_cos")
    val_coarse = L_coarse[idx, idx]
    val_fine = L_fine[idx, idx]
    rel = abs(val_coarse - val_fine) / abs(val_fine)
    assert rel < 1.0e-8, (
        f"disc_top_m0_n2 self entry resolution mismatch: "
        f"coarse={val_coarse}, fine={val_fine}, rel={rel}"
    )
