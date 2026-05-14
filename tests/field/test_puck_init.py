"""Unit tests for :mod:`simsopt.field.puck_init`.

Locks the geometric contracts of :func:`toroidal_shell_pucks` -- in particular
the new mixed-size / per-cell selection path added on top of the legacy
scalar-``puck_R`` mode.  The plasma boundary is stubbed so the tests do not
depend on a particular NetCDF / VMEC asset; only ``plasma_boundary.gamma()``
and ``plasma_boundary.nfp`` are consumed by the function under test.
"""

from __future__ import annotations

import numpy as np
import pytest

from simsopt.field.puck_init import toroidal_shell_pucks


class _StubBoundary:
    """Minimal stand-in for ``SurfaceRZFourier`` (only ``.gamma()`` + ``.nfp``).

    Returns a fixed-shape ``gamma`` array consistent with a major-radius-``R0``,
    minor-radius-``a`` axisymmetric torus so that ``R_plasma.mean() == R0`` and
    ``max(sqrt((R - R0)**2 + Z**2)) == a``.  This lets tests override ``R0`` /
    ``minor_radius`` explicitly without dragging in a real boundary.
    """

    def __init__(self, R0: float = 10.0, a: float = 4.0, nfp: int = 2) -> None:
        n = 16
        theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        phi = np.linspace(0.0, 2.0 * np.pi / nfp, n, endpoint=False)
        Tg, Pg = np.meshgrid(theta, phi, indexing="ij")
        R = R0 + a * np.cos(Tg)
        Z = a * np.sin(Tg)
        self._gamma = np.stack(
            [R * np.cos(Pg), R * np.sin(Pg), Z], axis=-1
        )
        self.nfp = int(nfp)
        self.stellsym = True

    def gamma(self) -> np.ndarray:
        return self._gamma


def _on_shell(centers: np.ndarray, R0: float, a: float) -> np.ndarray:
    """Return per-puck ``sqrt((R - R0)**2 + Z**2)`` for centre-on-shell checks."""
    R = np.sqrt(centers[:, 0] ** 2 + centers[:, 1] ** 2)
    Z = centers[:, 2]
    return np.sqrt((R - R0) ** 2 + Z ** 2)


# ----------------------------------------------------------------------
# Scalar / uniform path -- preserved
# ----------------------------------------------------------------------


def test_toroidal_shell_scalar_path_unchanged() -> None:
    """``size_menu=None`` reproduces the legacy uniform-size behaviour."""
    bd = _StubBoundary(R0=10.0, a=4.0, nfp=2)
    R0, a, n_t, n_p, puck_R, puck_t = 10.0, 4.0, 6, 4, 0.4, 0.3
    centers, axes, radii, ths = toroidal_shell_pucks(
        bd,
        n_theta=n_t,
        n_phi=n_p,
        R0=R0,
        minor_radius=a,
        puck_R=puck_R,
        puck_t=puck_t,
        nfp=2,
    )
    assert centers.shape == (n_t * n_p, 3)
    assert axes.shape == (n_t * n_p, 3)
    assert radii.shape == (n_t * n_p,)
    assert ths.shape == (n_t * n_p,)
    np.testing.assert_allclose(radii, puck_R)
    np.testing.assert_allclose(ths, puck_t)
    # Axes unit-norm and inward-pointing.
    np.testing.assert_allclose(np.linalg.norm(axes, axis=1), 1.0, atol=1e-12)
    # Centres on the prescribed shell.
    np.testing.assert_allclose(_on_shell(centers, R0, a), a, atol=1e-10)


# ----------------------------------------------------------------------
# Mixed-size / per-cell selection path
# ----------------------------------------------------------------------


def test_toroidal_shell_size_menu_picks_largest_per_cell() -> None:
    """Each cell picks the largest menu entry whose 2R fits both arcs.

    Geometry:

    * ``R0 = 10``, ``a = 4``, ``nfp = 2``, ``n_theta = 4``, ``n_phi = 3``.
    * ``d_theta = 2*pi/4 = pi/2``; ``d_phi = (pi/2)/3 = pi/6``.
    * Outboard centre (``theta = pi/4``, i.e. ``cos(theta) > 0``): local
      ``R_local = R0 + a*cos(pi/4) ~= 12.83``; arc_phi ~ 6.72 m.
    * Inboard centre (``theta = 5pi/4``, ``cos(theta) < 0``): local
      ``R_local ~= 7.17``; arc_phi ~ 3.75 m.
    * Both rings see ``arc_theta = 4 * pi/2 = 6.28`` m.

    The menu ``[(2.0, 1.0), (1.5, 0.5), (0.5, 0.2)]`` therefore selects:

    * Outboard rows: ``2R = 4.0`` fits ``arc_theta = 6.28``, ``arc_phi >= 5``
      cells -> the ``R = 2.0`` entry.
    * Inboard rows: ``2R = 4.0 > arc_phi(inboard) ~ 3.75``, ``2R = 3.0 <
      arc_phi(inboard) = 3.75``-? exactly 3.0 < 3.75 so the ``R = 1.5``
      entry fits.  The ``R = 0.5`` fallback covers the worst-case cells.
    """
    bd = _StubBoundary(R0=10.0, a=4.0, nfp=2)
    R0, a, n_t, n_p = 10.0, 4.0, 4, 3
    menu = [(2.0, 1.0), (1.5, 0.5), (0.5, 0.2)]
    centers, axes, radii, ths = toroidal_shell_pucks(
        bd,
        n_theta=n_t,
        n_phi=n_p,
        R0=R0,
        minor_radius=a,
        nfp=2,
        size_menu=menu,
        fill_fraction=1.0,
    )
    assert centers.shape[0] == n_t * n_p, (
        "fill_fraction=1.0 with no gaps should keep all cells."
    )
    unique_R = np.unique(np.round(radii, 9))
    assert unique_R.size >= 2, (
        f"Expected at least two distinct radii after per-cell selection, "
        f"got {unique_R!r}."
    )
    # Each theta-row is uniform under axisymmetry (same R_local), so
    # collect the per-row radius.
    radii_grid = radii.reshape(n_t, n_p)
    row_radii = radii_grid[:, 0]
    # Per-row radius must monotonically decrease as the row moves from
    # outboard (cos > 0) toward inboard (cos < 0) because cells get
    # smaller.  Verify via the analytic per-cell cap.
    d_theta = 2.0 * np.pi / n_t
    d_phi = (np.pi / 2) / n_p
    theta_arr = np.linspace(0.0, 2.0 * np.pi, n_t, endpoint=False) + d_theta / 2.0
    arc_theta_cell = a * d_theta
    menu_sorted = sorted(menu, key=lambda e: e[0], reverse=True)
    for i, th_i in enumerate(theta_arr):
        R_local = R0 + a * np.cos(th_i)
        arc_phi_cell = R_local * d_phi
        cap = min(arc_theta_cell, arc_phi_cell)
        # Expected radius is the largest menu entry with 2R <= cap.
        expected_R = next(
            (R_e for R_e, _ in menu_sorted if 2.0 * R_e <= cap + 1e-14),
            None,
        )
        assert expected_R is not None, (
            f"Cell at theta={th_i:.3f} cap={cap:.3f} fits no menu entry."
        )
        assert np.isclose(row_radii[i], expected_R), (
            f"Row {i} (theta={th_i:.3f}): expected R={expected_R}, "
            f"got {row_radii[i]}."
        )
    # Centres still on the prescribed shell, axes unit norm.
    np.testing.assert_allclose(_on_shell(centers, R0, a), a, atol=1e-10)
    np.testing.assert_allclose(np.linalg.norm(axes, axis=1), 1.0, atol=1e-12)


def test_toroidal_shell_size_menu_drops_cells_below_min_radius() -> None:
    """Cells whose best-fitting entry falls below ``min_radius`` are dropped."""
    bd = _StubBoundary(R0=10.0, a=4.0, nfp=2)
    R0, a, n_t, n_p = 10.0, 4.0, 4, 3
    menu = [(2.0, 1.0), (0.4, 0.2)]
    centers_full, _, radii_full, _ = toroidal_shell_pucks(
        bd,
        n_theta=n_t,
        n_phi=n_p,
        R0=R0,
        minor_radius=a,
        nfp=2,
        size_menu=menu,
        fill_fraction=1.0,
        min_radius=0.0,
    )
    centers_min, _, radii_min, _ = toroidal_shell_pucks(
        bd,
        n_theta=n_t,
        n_phi=n_p,
        R0=R0,
        minor_radius=a,
        nfp=2,
        size_menu=menu,
        fill_fraction=1.0,
        min_radius=0.5,
    )
    assert centers_min.shape[0] < centers_full.shape[0], (
        "Setting min_radius above the smallest menu entry must drop some cells."
    )
    assert np.all(radii_min >= 0.5 - 1e-14)
    assert (radii_full < 0.5).any(), (
        "Sanity: the no-floor run should have used the small menu entry "
        "in at least one cell."
    )


def test_toroidal_shell_size_menu_respects_gaps() -> None:
    """``gap_poloidal`` / ``gap_toroidal`` tighten the per-cell cap."""
    bd = _StubBoundary(R0=10.0, a=4.0, nfp=2)
    R0, a, n_t, n_p = 10.0, 4.0, 4, 3
    menu = [(2.0, 1.0), (1.0, 0.5)]
    _, _, radii_no_gap, _ = toroidal_shell_pucks(
        bd,
        n_theta=n_t,
        n_phi=n_p,
        R0=R0,
        minor_radius=a,
        nfp=2,
        size_menu=menu,
        fill_fraction=1.0,
        gap_poloidal=0.0,
        gap_toroidal=0.0,
    )
    _, _, radii_with_gap, _ = toroidal_shell_pucks(
        bd,
        n_theta=n_t,
        n_phi=n_p,
        R0=R0,
        minor_radius=a,
        nfp=2,
        size_menu=menu,
        fill_fraction=1.0,
        gap_poloidal=1.5,
        gap_toroidal=1.5,
    )
    # With a sizeable gap, every cell must fall back to the smaller menu
    # entry because the 2R = 4.0 footprint no longer fits any cell.
    assert (radii_no_gap == 2.0).any()
    assert np.all(radii_with_gap <= 1.0 + 1e-14)


def test_toroidal_shell_size_menu_validation() -> None:
    """Bad menus / fill_fractions raise :class:`ValueError`."""
    bd = _StubBoundary(R0=10.0, a=4.0, nfp=2)
    with pytest.raises(ValueError, match="radius must be positive"):
        toroidal_shell_pucks(
            bd,
            n_theta=4,
            n_phi=3,
            R0=10.0,
            minor_radius=4.0,
            nfp=2,
            size_menu=[(0.0, 0.5)],
        )
    with pytest.raises(ValueError, match="thickness must be positive"):
        toroidal_shell_pucks(
            bd,
            n_theta=4,
            n_phi=3,
            R0=10.0,
            minor_radius=4.0,
            nfp=2,
            size_menu=[(0.5, -0.1)],
        )
    with pytest.raises(ValueError, match="fill_fraction"):
        toroidal_shell_pucks(
            bd,
            n_theta=4,
            n_phi=3,
            R0=10.0,
            minor_radius=4.0,
            nfp=2,
            size_menu=[(0.5, 0.2)],
            fill_fraction=0.0,
        )
    with pytest.raises(ValueError, match="gap_poloidal"):
        toroidal_shell_pucks(
            bd,
            n_theta=4,
            n_phi=3,
            R0=10.0,
            minor_radius=4.0,
            nfp=2,
            size_menu=[(0.5, 0.2)],
            gap_poloidal=-0.1,
        )


def test_toroidal_shell_size_menu_no_entry_fits_returns_empty() -> None:
    """If no menu entry fits any cell, the function returns 0-length arrays."""
    bd = _StubBoundary(R0=10.0, a=4.0, nfp=2)
    centers, axes, radii, ths = toroidal_shell_pucks(
        bd,
        n_theta=4,
        n_phi=3,
        R0=10.0,
        minor_radius=4.0,
        nfp=2,
        size_menu=[(100.0, 1.0)],
        fill_fraction=1.0,
    )
    assert centers.shape == (0, 3)
    assert axes.shape == (0, 3)
    assert radii.shape == (0,)
    assert ths.shape == (0,)
