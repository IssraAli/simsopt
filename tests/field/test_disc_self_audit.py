"""Read-only audit of the current top/bottom disc self-inductance path.

Diagnostic only: prints a table comparing the existing
:func:`~simsopt.field.bulk_inductance.shell_inductance_matrix_blockwise`
self-block entries against an independent semi-analytic Maxwell-Grover
ring-kernel reference, swept over quadrature resolutions and Zernike
modes.  No assertions; this test is meant to be run with ``-s`` and to
lock in the quantitative baseline of the regularized quadrature before
the semi-analytic replacement is introduced.
"""

from __future__ import annotations

from typing import Callable, List, Tuple

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("jax.numpy")

import jax.numpy as jnp  # noqa: E402
from scipy.special import ellipe, ellipk  # noqa: E402

from simsopt.field.bulk_inductance import (  # noqa: E402
    shell_inductance_matrix_blockwise,
)
from simsopt.field.puck_basis import (  # noqa: E402
    build_puck_shell_basis,
    zernike_radial_derivative,
)


MU0 = 4.0 * np.pi * 1.0e-7


def _maxwell_grover(a: float, b: float, dz: float = 0.0) -> float:
    r"""Maxwell-Grover mutual inductance of two coaxial coplanar-ish
    circular filaments of radii ``a, b`` at axial separation ``dz``.

    .. math::

        M(a, b, d) = \mu_0 \sqrt{a b}\,
        \frac{(2 - k^2) K(k) - 2 E(k)}{k},\qquad
        k^2 = \frac{4 a b}{(a + b)^2 + d^2}.

    Returns 0.0 for degenerate cases (``a = 0`` or ``b = 0``).
    """
    if a <= 0.0 or b <= 0.0:
        return 0.0
    m = 4.0 * a * b / ((a + b) ** 2 + dz**2)
    if m >= 1.0:
        m = 1.0 - 1.0e-16
    k = np.sqrt(m)
    K_k = ellipk(m)
    E_k = ellipe(m)
    return MU0 * np.sqrt(a * b) * ((2.0 - m) * K_k - 2.0 * E_k) / k


def _disc_self_block_reference(
    f_prime: Callable[[float], float], R: float, n_radial: int = 64
) -> float:
    r"""Reference semi-analytic value of

    .. math::

        \int_0^R \int_0^R f'(a)\,f'(b)\,M(a, b, 0)\,da\,db

    using the Duffy-triangle parameterization ``a = rho``,
    ``b = rho (1 - eta)`` with ``rho in [0, R]``, ``eta in (0, 1]``,
    plus Gauss-Legendre on ``rho`` and Gauss-Laguerre on
    ``t = -ln(eta)`` (so the kernel's log singularity at ``eta = 0``
    sits at ``t -> infty`` where the Laguerre weight ``e^{-t}``
    exponentially damps the linear-in-``t`` divergence of ``K(k)``).

    Only called inside the audit test as an *independent* sanity
    reference; not a production routine.
    """
    rho_nodes, rho_w = np.polynomial.legendre.leggauss(n_radial)
    rho = 0.5 * R * (rho_nodes + 1.0)
    w_rho = 0.5 * R * rho_w

    t_nodes, t_w = np.polynomial.laguerre.laggauss(n_radial)
    eta = np.exp(-t_nodes)

    total = 0.0
    for i in range(n_radial):
        for j in range(n_radial):
            a1 = rho[i]
            b1 = rho[i] * (1.0 - eta[j])
            # eta=1 corresponds to b=0; M should be 0 there but guard anyway
            M_ab = _maxwell_grover(a1, b1, 0.0)
            factor = w_rho[i] * t_w[j] * rho[i]
            total += factor * f_prime(a1) * f_prime(b1) * M_ab
            # Other triangle (a < b): swap f_prime arguments (same function here)
            total += factor * f_prime(b1) * f_prime(a1) * M_ab
    return float(total)


def _disk_only_basis_data(
    R: float,
    t: float,
    m_fourier: int = 0,
    l_zernike: int = 0,
    n_rho: int = 12,
    n_phi: int = 32,
    n_z: int = 2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int], List[str]]:
    """Return ``(K_basis, quad_points, quad_weights, disk_top_indices,
    dof_names)`` with side-wall basis columns included (they are part of
    the production basis) but disk-top DOF indices flagged for easy
    selection.

    The top-face ``m=0, n=0`` constant-potential DOF has zero tangential
    ``K`` and so gives ``L[0,0] = 0``.  We actually want the first
    non-constant top-face Zernike mode ``m=0, n=2`` (the lowest radial
    mode with non-trivial ``K_phi``), which we flag here.
    """
    basis = build_puck_shell_basis(
        R=R,
        t=t,
        m_fourier=m_fourier,
        l_zernike=l_zernike,
        k_chebyshev=0,
        n_rho=n_rho,
        n_phi=n_phi,
        n_z=n_z,
    )
    disk_top_idx = [
        a for a, name in enumerate(basis.dof_names) if name.startswith("disk_top_")
    ]
    return (
        basis.k_basis_local,
        basis.quad_points_local,
        basis.quad_weights,
        disk_top_idx,
        list(basis.dof_names),
    )


def test_disc_self_inductance_audit(capsys):
    r"""Quantitative audit of
    :func:`shell_inductance_matrix_blockwise` for the top-face disc
    self-block.

    Extracts the ``disk_top_m0_n2_cos`` diagonal entry (the lowest
    non-trivial axisymmetric radial mode; ``m=0, n=0`` is a constant
    potential and has ``K = 0``, which would make the audit trivial)
    from the current regularized numerical pipeline with both
    ``adaptive_self_reg in {False, True}``, and compares against an
    independent Duffy + Gauss-Laguerre Maxwell-Grover reference.

    Also sweeps ``(m, n) in {(0, 2), (1, 1)}`` to confirm the error is
    mode-resolution-independent.
    """
    R = 0.10
    t = 0.01  # dummy thickness; disc top block does not see ``t``

    # Modes under audit: (m, n). Skip (0, 0) because R_0^0 is the constant
    # potential (dR/dr = 0), so ``K = 0`` there and the audit is degenerate.
    audit_modes: List[Tuple[int, int]] = [(0, 2), (1, 1)]

    lines: List[str] = []
    lines.append(
        "=== Disc top-face self-block audit (shell_inductance_matrix_blockwise) ==="
    )
    lines.append(f"R = {R:.3e} m, t = {t:.3e} m")
    lines.append("")

    for m, n in audit_modes:
        lines.append(f"--- Mode (m={m}, n={n}) ---")

        # Analytic ``f'(rho)`` for the chosen Zernike radial mode:
        # f(rho) = R_n^m(rho / R); f'(rho) = (1/R) dR_n^m/dr|_{r=rho/R}.
        def f_prime_factory(mm: int, nn: int, RR: float):
            def f_prime(rho: float) -> float:
                r = np.clip(rho / RR, 0.0, 1.0 - 1.0e-14)
                return float(
                    (1.0 / RR)
                    * np.asarray(zernike_radial_derivative(jnp.asarray(r), mm, nn))
                )

            return f_prime

        f_prime = f_prime_factory(m, n, R)

        # Independent reference (high-resolution)
        L_ref = _disc_self_block_reference(f_prime, R, n_radial=64)
        # Factor for the mode-m kernel.  For m=0 the full bilinear form
        # is ``L_ref`` as defined.  For m>=1 the azimuthal coupling of
        # the ``e_phi`` and ``e_rho`` tangential components adds further
        # ``I_{m-1} pm I_{m+1}`` kernel contributions; see the Phase-2
        # disc_self_inductance module for the full expression.  For the
        # audit we only compare to the axisymmetric radial form to
        # establish the baseline error of the current path on the
        # ``K_phi`` part.
        lines.append(f"  axisymmetric radial reference L_ref = {L_ref:.6e}")

        for n_rho, n_phi in [(8, 16), (12, 32), (20, 64)]:
            basis = build_puck_shell_basis(
                R=R,
                t=t,
                m_fourier=max(m, 1),
                l_zernike=max(n, 2),
                k_chebyshev=0,
                n_rho=n_rho,
                n_phi=n_phi,
                n_z=2,
            )
            # Locate the DOF named "disk_top_m{m}_n{n}_cos"
            target = f"disk_top_m{m}_n{n}_cos"
            try:
                idx = basis.dof_names.index(target)
            except ValueError:
                lines.append(
                    f"  [skip] DOF '{target}' not in basis "
                    f"(n_rho={n_rho}, n_phi={n_phi})"
                )
                continue

            n_dof = basis.k_basis_local.shape[1]

            for adaptive in (False, True):
                L_np = shell_inductance_matrix_blockwise(
                    [basis.k_basis_local],
                    [basis.quad_points_local],
                    [basis.quad_weights],
                    dof_offsets=[0],
                    n_dof_total=n_dof,
                    delta_reg=1.0e-10,
                    adaptive_self_reg=adaptive,
                )
                L_numerical = float(L_np[idx, idx])
                ratio = L_numerical / L_ref if L_ref != 0.0 else float("nan")
                tag = "adaptive" if adaptive else "legacy  "
                lines.append(
                    f"  (n_rho={n_rho:3d}, n_phi={n_phi:3d}) {tag}: "
                    f"L_numerical={L_numerical:.3e}  ratio={ratio:+.3e}"
                )

        lines.append("")

    # Print to stdout so ``pytest -s`` shows it; also attach via capsys.
    for line in lines:
        print(line)

    with capsys.disabled():
        # (Diagnostic only — no assertions.)
        pass
