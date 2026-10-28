#!/usr/bin/env python
"""
Corner sharpening IN THE LOOP: optimize a low-resolution SurfaceBSpline
for quasisymmetry, aspect ratio matching the original QUASR equilibrium,
and corner opening angles of 90 degrees, where every objective evaluation
sees the SHARPENED surface.

Optimization variables: the low-resolution spline's own dofs (the
65-dof, 6-points-per-cross-section surface -- i.e. BEFORE any
Lane-Riesenfeld doubling). Sharpening itself has no dofs; it is a
deterministic function of those low dofs:

    low dofs -> refine_poloidal x REFINE_POLOIDAL_COUNT (Lane-Riesenfeld)
             -> corner (seed point) per cross section, located by max-Z
                or max-curvature
             -> walk back d_crawl*perimeter, take the two tangents
             -> intersect, push the control points between onto the two
                tangent legs (tangent_extension._apply_tangent_extension)
             -> sharpened surface -> Fourier boundary -> VMEC -> QS/iota/AR

SharpenedBoundary below is the Optimizable that sits between the low
surface and VMEC (VMEC only needs `.to_RZFourier()` from its boundary).

Objective (least squares):
  - QS residuals (helicity (1, 0), same as sharpen.py)
  - |mean iota| >= IOTA_MIN (free above), as a one-sided penalty. Without
    it the optimizer buys QS by driving iota -> 0 (the (1, 0) residual has
    a term proportional to iota).
  - aspect ratio - the original QUASR equilibrium's aspect ratio (read from
    TARGET_FILE's boundary, VMEC definition)
  - opening angle at every corner of every cross section - TARGET_ANGLE_DEG
    (90 deg), currently with a tiny weight (W_ANGLE = 0.01) so it barely
    competes with QS / aspect ratio.
  (The apex-Z residuals are still defined below but are not in the
  objective.)

The crossover point (tangent-line intersection, i.e. the apex) of every
corner has a target lab-frame Z (apex_z_residuals); it is exposed by
SharpenedBoundary.apex_RZ() as (R, Z). Targets for both the opening angle
and the apex Z are set by TARGET_ANGLE_DEG / TARGET_APEX_Z (None = keep each
corner's original value).

The per-corner opening angle is tangent_extension's own `opening_angle`
(angle at the apex between apex->P_minus and apex->P_plus; the angle
between the two tangent DIRECTIONS is pi minus this, so holding one holds
the other). It is exposed per corner by SharpenedBoundary.opening_angles().

Two things specific to optimizing through this pipeline with finite
differences:
  1. sharpen_fixed_s_twin's corner finder returns an argmax over an
     n_samples grid, so the corner location is quantized (~2*pi/4000
     rad) -- a 1e-6 dof perturbation would then move it by exactly 0 or
     by a full grid step. _corner_pair_geometry_polished polishes each
     corner with a bounded 1D search on the exact curve, making the
     corner (and so the opening angle) a continuous function of the dofs.
  2. What remains discontinuous is inherent to the algorithm: which
     control points fall inside the tangent window (and which one snaps
     to the apex) changes discretely as the window slides. Infinitesimal
     finite-difference steps almost never straddle that, but a full
     optimizer step can, so expect a rougher landscape than plain QS.
     Failures of the construction (parallel tangents, overlapping
     windows) are turned into ObjectiveFailure so the solver just backs
     off.

Run (from this directory):
    mpirun -n 12 python corner_sharpen_opt.py
    python corner_sharpen_opt.py --evaluate-only     # one evaluation, no optimization
"""

import argparse
import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar

# numpy 2.x + macOS Accelerate emits spurious divide-by-zero/overflow/invalid
# warnings from matmul on perfectly finite inputs (results verified identical
# to einsum); they'd otherwise flood every B-spline evaluation.
warnings.filterwarnings("ignore", message=".*encountered in matmul")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "corner_sharpening"))

import sharpen_fixed_s_twin as sfst  # noqa: E402  (needs the sys.path entry)
from sharpen import widen_violated_bounds  # noqa: E402
from sharpen_fixed_s_twin import (  # noqa: E402
    _find_both_corner_us,
    build_sharpened_twins,
    pickle_twin_surface,
    plot_before_after_twin,
    plot_leg_length_grid,
)
from simsopt._core import Optimizable, make_optimizable  # noqa: E402
from simsopt._core.util import ObjectiveFailure  # noqa: E402
from simsopt.geo import SurfaceBSpline, SurfaceRZFourier  # noqa: E402
from simsopt.mhd import QuasisymmetryRatioResidual, Vmec  # noqa: E402
from simsopt.objectives import LeastSquaresProblem  # noqa: E402
from simsopt.solve import least_squares_mpi_solve  # noqa: E402
from simsopt.util import MpiPartition, proc0_print  # noqa: E402
from tangent_extension import (  # noqa: E402
    _apply_tangent_extension,
    _build_z_embed_list,
    _corner_geometry_from_u,
    _cross_section_local_curve,
    _curve_curvature,
    _periodic_curve_deriv2,
)

# ---------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------

# The QUASR equilibrium the spline was fit to (see shape_match.py), used
# for the aspect-ratio target.
TARGET_FILE = os.path.join(HERE, "..", "corner_sharpening", "input.0203395")

# Low-dof spline fit to that equilibrium: stage 1 (no poloidal refine) of
# ../corner_sharpening/shape_match.py, rerun with SPLINE_KWARGS below
# (65 dofs; max/mean exact shape error 3.1e-3 / 9.7e-4 at R0 = 1).
INITIAL_DOFS = np.array(
    [
        0.18018380,
        0.16852591,
        0.43668447,
        0.17774171,
        1.44093066,
        2.03468302,
        0.12529887,
        0.20606790,
        0.40662734,
        0.18032182,
        0.49580785,
        0.14472342,
        0.67225077,
        1.90897319,
        2.61799388,
        4.24893800,
        4.71238898,
        0.09340778,
        0.25936872,
        0.43163652,
        0.16634469,
        0.52514823,
        0.14305778,
        0.71764233,
        1.69295334,
        3.30310028,
        4.32899377,
        4.71238898,
        0.08304195,
        0.31762877,
        0.46650957,
        0.21455094,
        0.55675441,
        0.14260905,
        0.89934280,
        1.57079633,
        3.66519143,
        4.48166409,
        4.71238898,
        0.04948990,
        0.38153149,
        0.48589366,
        0.22758722,
        0.60529620,
        0.12038121,
        1.12816124,
        1.57079633,
        3.66519143,
        4.61873460,
        4.71238898,
        0.01610439,
        0.43932354,
        0.49271770,
        0.22449632,
        0.55532019,
        0.36753594,
        1.29195991,
        1.63056199,
        3.66519143,
        4.70701418,
        4.77931857,
        0.90245209,
        1.11239778,
        1.19157150,
        0.07070622,
    ]
)
SPLINE_KWARGS = (
    sfst.DEMO_SPLINE_KWARGS
)  # identical to sharpen.py's spline_kwargs

REFINE_POLOIDAL_COUNT = 3
CORNER_CRITERION = "z"  # "z" or "curvature"
D_CRAWL = 0.04  # INITIAL tangent-seed walk-back (a dof of the optimization), fraction of min cross-section perimeter
D_CRAWL_BOUNDS = (1e-2, 0.1)
N_SAMPLES = 2000  # curve samples per cross section (corners are polished after)

# Twin-surface construction at the end (see sharpen_fixed_s_twin.__main__)
D_EXT = 0.10
L_X = 0.04

# VMEC inside the loop (same as sharpen.py) and the final verification run
VMEC_LOOP = dict(ns=25, M=12, N=12, ftol=1e-8, niter=5000)
VMEC_FINAL = dict(
    ns=50, M=16, N=16, ftol=1e-10, niter=20000, ntheta=64, nzeta=64
)

# Objective weights. Residual scales: QS residuals ~1e-3 each (x1000, as in
# sharpen.py); iota/AR are O(0.01-0.1) errors; angles are radians.
# These are knobs, not derived.
W_QS = 1000
W_IOTA = 100
W_AR = 10

# One-sided (inequality) constraint, a quadratic penalty that is exactly zero
# inside the allowed region:
#   |mean iota| >= IOTA_MIN   (free to grow; magnitude because the QUASR
#                              equilibrium's iota is negative in VMEC's sign
#                              convention -- the sign is left free)
IOTA_MIN = 0.08
W_ANGLE = 0.01
BOUND_MARGIN = 0.1  # slack added when widening bounds the fitted dofs violate
W_APEX_Z = (
    10  # crossover-point Z residuals, in the same length units as the surface
)

# Targets for the corners. None -> hold each corner at its ORIGINAL value
# (measured on the initial low-dof surface); a number -> drive EVERY corner to it.
TARGET_ANGLE_DEG = 90.0  # opening angle at the apex [deg]
TARGET_APEX_Z = (
    0.6  # |Z| of the crossover point [same units as the surface], e.g. 0.35;
)
#                       the +Z corners target +value, the -Z corners -value


# ---------------------------------------------------------------------
# Corner geometry with continuous (polished) corner locations
# ---------------------------------------------------------------------


def _polish_corner_u(samples, u0, criterion, z_embed, maximize):
    """Refine a grid-quantized corner parameter u0 to the exact local
    extremum of the corner metric (z for "max z", curvature otherwise)
    within one sample spacing of u0."""
    h = 2 * np.pi / len(samples["us"])

    def metric(u):
        pos, vel, acc = _periodic_curve_deriv2(
            samples["core"],
            samples["w"],
            samples["p"],
            samples["kp"],
            np.array([u]),
        )
        if criterion == "max z":
            return float(z_embed(pos)[0])
        return float(_curve_curvature(vel, acc)[0])

    sign = 1.0 if maximize else -1.0
    res = minimize_scalar(
        lambda u: -sign * metric(u),
        bounds=(u0 - h, u0 + h),
        method="bounded",
        options={"xatol": 1e-12},
    )
    return float(res.x % (2 * np.pi))


def _corner_pair_geometry_polished(
    spline_surf, s1, s2, corner_criterion, n_samples=4000
):
    """Drop-in for sharpen_fixed_s_twin._corner_pair_geometry (same return
    shape: per cross section, [lower-u corner, higher-u corner]) with each
    corner's u polished to a continuous function of the dofs."""
    z_embed_list = _build_z_embed_list(spline_surf)
    geometry = []
    for i, cs in enumerate(spline_surf.cs_list):
        samples = _cross_section_local_curve(spline_surf, cs, n_samples)
        z_embed = z_embed_list[i]
        u_coarse = _find_both_corner_us(samples, corner_criterion, z_embed)
        z_at = [
            float(
                z_embed(
                    _periodic_curve_deriv2(
                        samples["core"],
                        samples["w"],
                        samples["p"],
                        samples["kp"],
                        np.array([u]),
                    )[0]
                )[0]
            )
            for u in u_coarse
        ]
        plus_idx = int(np.argmax(z_at))
        us = []
        for k, u0 in enumerate(u_coarse):
            # "max z": +Z corner maximizes z, -Z corner minimizes it;
            # curvature is maximized at both.
            maximize = True if corner_criterion != "max z" else (k == plus_idx)
            us.append(
                _polish_corner_u(
                    samples, u0, corner_criterion, z_embed, maximize
                )
            )
        us = sorted(us)
        geometry.append(
            [
                _corner_geometry_from_u(samples, us[0], s1, s2),
                _corner_geometry_from_u(samples, us[1], s1, s2),
            ]
        )
    return geometry


def corner_opening_angles(geometry):
    """Flat array of opening angles [rad], cross-section-major, two corners
    ([lower-u, higher-u]) per cross section."""
    return np.array(
        [g["opening_angle"] for cs_corners in geometry for g in cs_corners]
    )


def corner_apex_RZ(spline_surf, geometry):
    """Lab-frame (R, Z) of every corner's crossover point (the tangent-line
    intersection, g['apex']) -- (2*n_cs, 2) array, same ordering as
    corner_opening_angles. Uses the same local->lab embedding as
    _build_z_embed_list / sharpen_fixed_s_twin._build_r_embed_list, so it
    is insensitive to how the local bishop frame twists between sections."""
    cs_zeta, cs_angle = spline_surf.get_cs_zeta_angle()
    axis_pos, e1, e2 = spline_surf._axis_local_basis(cs_zeta)
    out = []
    for i, cs_corners in enumerate(geometry):
        ang = cs_angle[i]
        for g in cs_corners:
            x, y = g["apex"]
            xr = x * np.cos(ang) - y * np.sin(ang)
            yr = x * np.sin(ang) + y * np.cos(ang)
            XYZ = axis_pos[i] + xr * e1[i] + yr * e2[i]
            out.append([np.hypot(XYZ[0], XYZ[1]), XYZ[2]])
    return np.array(out)


# ---------------------------------------------------------------------
# The sharpened boundary Optimizable
# ---------------------------------------------------------------------


def build_refined_surface(spline_kwargs, full_x, refine_count):
    """A fresh SurfaceBSpline holding `full_x` (the LOW-resolution surface's
    full dof vector), doubled `refine_count` times."""
    surf = SurfaceBSpline(**spline_kwargs)
    surf.full_x = np.array(full_x, dtype=float)
    for _ in range(refine_count):
        surf.refine_poloidal()
    return surf


def sharpen_surface(
    spline_kwargs, full_x, d_crawl, corner_criterion, refine_count, n_samples
):
    """low dofs -> (sharpened high-res surface, geometry). Raises ValueError
    if the tangent construction is ill-posed for these dofs."""
    surf = build_refined_surface(spline_kwargs, full_x, refine_count)
    perimeters = [
        _cross_section_local_curve(surf, cs, n_samples=n_samples)["L"]
        for cs in surf.cs_list
    ]
    s = d_crawl * min(perimeters)
    internal = "max z" if corner_criterion == "z" else "curvature"
    geometry = _corner_pair_geometry_polished(surf, s, s, internal, n_samples)
    surf.fix_all()
    _apply_tangent_extension(surf, geometry)
    surf.unfix_all()
    return surf, geometry


class SharpenedBoundary(Optimizable):
    """
    VMEC boundary = the sharpened version of `low_surf`. Its one own dof is
    the walk-back fraction `d_crawl` (bounded to d_crawl_bounds); it also
    depends on low_surf, so the optimization variables are low_surf's dofs plus
    d_crawl, and every change to either invalidates this (and VMEC).
    """

    def __init__(
        self,
        low_surf,
        spline_kwargs,
        d_crawl,
        corner_criterion,
        refine_count=REFINE_POLOIDAL_COUNT,
        n_samples=N_SAMPLES,
        d_crawl_bounds=D_CRAWL_BOUNDS,
    ):
        self.low_surf = low_surf
        self.spline_kwargs = spline_kwargs
        self.corner_criterion = corner_criterion
        self.refine_count = refine_count
        self.n_samples = n_samples
        self._key = None
        self._cache = None
        Optimizable.__init__(
            self,
            x0=np.array([d_crawl], dtype=float),
            names=["d_crawl"],
            lower_bounds=np.array([d_crawl_bounds[0]]),
            upper_bounds=np.array([d_crawl_bounds[1]]),
            depends_on=[low_surf],
        )

    @property
    def d_crawl(self):
        return float(self.local_full_x[0])

    def _evaluate(self):
        key = (
            np.asarray(self.low_surf.full_x).tobytes()
            + np.float64(self.d_crawl).tobytes()
        )
        if key != self._key:
            try:
                surf, geometry = sharpen_surface(
                    self.spline_kwargs,
                    self.low_surf.full_x,
                    self.d_crawl,
                    self.corner_criterion,
                    self.refine_count,
                    self.n_samples,
                )
            except ValueError as e:
                self._key = None
                raise ObjectiveFailure(f"sharpening failed: {e}") from e
            self._key = key
            self._cache = (surf, geometry)
        return self._cache

    @property
    def sharpened_surface(self):
        return self._evaluate()[0]

    def opening_angles(self):
        return corner_opening_angles(self._evaluate()[1])

    def apex_RZ(self):
        """(2*n_cs, 2) lab-frame (R, Z) of each corner's crossover point."""
        surf, geometry = self._evaluate()
        return corner_apex_RZ(surf, geometry)

    def to_RZFourier(self, **kwargs):
        return self.sharpened_surface.to_RZFourier(**kwargs)


def iota_lower_bound_residual(vmec, iota_min):
    """max(0, iota_min - |mean iota|): zero once |iota| >= iota_min."""
    return np.array([max(0.0, iota_min - abs(vmec.mean_iota()))])


def original_aspect_ratio(target_file):
    """Aspect ratio of the original equilibrium's boundary (VMEC definition,
    see SurfaceRZFourier.aspect_ratio), without running VMEC on it."""
    return SurfaceRZFourier.from_vmec_input(target_file).aspect_ratio()


def opening_angle_residuals(boundary, goal_angles):
    """Per-corner (opening angle - original opening angle)."""
    return boundary.opening_angles() - goal_angles


def apex_z_residuals(boundary, goal_apex_z):
    """Per-corner lab-frame Z of the crossover point minus its target."""
    return boundary.apex_RZ()[:, 1] - goal_apex_z


def build_targets(
    original_angles, original_apex_RZ, target_angle_deg, target_apex_z
):
    """Per-corner target opening angles [rad] and apex Z. A None target keeps
    each corner's original value; a number applies to every corner (the apex-Z
    magnitude keeps each corner's own sign, +Z vs -Z)."""
    if target_angle_deg is None:
        angles = original_angles.copy()
    else:
        angles = np.full_like(original_angles, np.radians(target_angle_deg))
    z0 = original_apex_RZ[:, 1]
    if target_apex_z is None:
        apex_z = z0.copy()
    else:
        apex_z = np.sign(z0) * abs(target_apex_z)
    return angles, apex_z


# ---------------------------------------------------------------------
# Reporting / plots
# ---------------------------------------------------------------------


def report(label, vmec, qs, boundary, goal_angles, goal_apex_z, goal_aspect):
    ang = boundary.opening_angles()
    proc0_print(f"--- {label} ---")
    proc0_print(f"  QS total (sum of squared residuals): {qs.total():.4e}")
    proc0_print(f"  d_crawl: {boundary.d_crawl:.5f}")
    proc0_print(f"  mean iota: {vmec.mean_iota():.5f}   (require |iota| >= {IOTA_MIN})")
    proc0_print(f"  aspect:    {vmec.aspect():.5f}   (target {goal_aspect:.5f})")
    proc0_print(
        "  opening angle per corner [deg] (rows = cross sections, cols = [lower-u, higher-u]):"
    )
    proc0_print(np.degrees(ang).reshape(-1, 2))
    proc0_print(
        f"  max |angle - target| = {np.degrees(np.max(np.abs(ang - goal_angles))):.3f} deg"
    )
    apex = boundary.apex_RZ()
    proc0_print(
        "  crossover point (R, Z) per corner (rows = cross sections; R0 Z0 R1 Z1):"
    )
    proc0_print(apex.reshape(-1, 4))
    proc0_print(
        f"  max |apex Z - target| = {np.max(np.abs(apex[:, 1] - goal_apex_z)):.4e}"
    )


def plot_angles(goal, initial, final, path):
    fig, ax = plt.subplots(figsize=(6, 4))
    idx = np.arange(len(goal))
    ax.plot(idx, np.degrees(goal), "k_", ms=18, mew=2, label="target")
    ax.plot(
        idx, np.degrees(initial), "o", mfc="none", label="start of optimization"
    )
    ax.plot(idx, np.degrees(final), "x", label="final")
    ax.set_xlabel("corner (2 per cross section)")
    ax.set_ylabel("opening angle [deg]")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--max-nfev", type=int, default=50)
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    mpi = MpiPartition()
    mpi.write()
    proc0_print(
        "Running figures_spline_paper/corner_sharpening_opt/corner_sharpen_opt.py"
    )
    proc0_print(
        "========================================================================="
    )

    # The low-dof surface the optimizer actually moves.
    low_surf = SurfaceBSpline(**SPLINE_KWARGS)
    low_surf.x = INITIAL_DOFS
    # The fitted dofs violate the fresh surface's default (construction-time)
    # box bounds, which the solver rejects as an infeasible start -- widen
    # only the violated ones, as sharpen.py does.
    lb, ub = low_surf.bounds
    n_viol = int(np.sum((low_surf.x < lb) | (low_surf.x > ub)))
    widen_violated_bounds(low_surf, margin=BOUND_MARGIN)
    proc0_print(
        f"Widened box bounds on {n_viol} of {len(low_surf.x)} dofs (margin {BOUND_MARGIN})."
    )
    low_surf_initial_full_x = np.array(low_surf.full_x)
    proc0_print(f"ndofs (low-resolution spline): {len(low_surf.x)}")

    boundary = SharpenedBoundary(
        low_surf, SPLINE_KWARGS, D_CRAWL, CORNER_CRITERION
    )
    initial_angles = boundary.opening_angles().copy()
    initial_apex = boundary.apex_RZ().copy()
    proc0_print("Original opening angles [deg]:")
    proc0_print(np.degrees(initial_angles).reshape(-1, 2))
    proc0_print(
        "Original crossover points (R, Z) [R0 Z0 R1 Z1 per cross section]:"
    )
    proc0_print(initial_apex.reshape(-1, 4))
    goal_angles, goal_apex_z = build_targets(
        initial_angles, initial_apex, TARGET_ANGLE_DEG, TARGET_APEX_Z
    )
    proc0_print("Target opening angles [deg]:")
    proc0_print(np.degrees(goal_angles).reshape(-1, 2))
    proc0_print("Target crossover-point Z [Z0 Z1 per cross section]:")
    proc0_print(goal_apex_z.reshape(-1, 2))

    goal_aspect = original_aspect_ratio(TARGET_FILE)
    proc0_print(f"Target aspect ratio (original equilibrium): {goal_aspect:.5f}")

    vmec = Vmec.vmec_from_surf(
        nfp=low_surf.nfp, surf=boundary, mpi=mpi, **VMEC_LOOP
    )
    qs = QuasisymmetryRatioResidual(
        vmec, np.arange(0, 1.01, 0.1), helicity_m=1, helicity_n=0
    )
    iota_obj = make_optimizable(iota_lower_bound_residual, vmec, IOTA_MIN)
    angle_obj = make_optimizable(opening_angle_residuals, boundary, goal_angles)
    # apex_obj = make_optimizable(apex_z_residuals, boundary, goal_apex_z)

    prob = LeastSquaresProblem.from_tuples(
        [
            (qs.residuals, 0, W_QS),
            (iota_obj.J, 0, W_IOTA),
            (vmec.aspect, goal_aspect, W_AR),
            (angle_obj.J, 0, W_ANGLE),
            # (apex_obj.J, 0, W_APEX_Z),
        ]
    )
    proc0_print(
        f"ndofs in problem: {len(prob.x)} (incl. d_crawl, bounds {D_CRAWL_BOUNDS})"
    )

    report(
        "start",
        vmec,
        qs,
        boundary,
        goal_angles,
        goal_apex_z,
        goal_aspect,
    )
    if args.evaluate_only:
        proc0_print(f"initial objective = {prob.objective():.6e}")
        return

    proc0_print("Beginning optimization")
    least_squares_mpi_solve(
        prob,
        mpi,
        grad=True,
        rel_step=1e-12,
        abs_step=1e-6,
        x_scale="jac",
        max_nfev=args.max_nfev,
    )

    report(
        "final",
        vmec,
        qs,
        boundary,
        goal_angles,
        goal_apex_z,
        goal_aspect,
    )
    final_angles = boundary.opening_angles().copy()
    np.save(
        os.path.join(HERE, "corner_sharpen_opt_low_x.npy"), np.array(low_surf.x)
    )
    np.save(
        os.path.join(HERE, "corner_sharpen_opt_low_full_x.npy"),
        np.array(low_surf.full_x),
    )
    proc0_print("low_surf.x:", repr(np.array(low_surf.x)))
    proc0_print(f"final d_crawl = {boundary.d_crawl:.6f}")
    np.save(
        os.path.join(HERE, "corner_sharpen_opt_d_crawl.npy"),
        np.array([boundary.d_crawl]),
    )

    # Final outputs: twin half-surfaces for stage-two, built from the
    # optimized low dofs with the SAME polished corner finder the loop used.
    sfst._corner_pair_geometry = _corner_pair_geometry_polished
    final_surf = build_refined_surface(
        SPLINE_KWARGS, low_surf.full_x, REFINE_POLOIDAL_COUNT
    )
    before_surf = build_refined_surface(
        SPLINE_KWARGS, low_surf_initial_full_x, REFINE_POLOIDAL_COUNT
    )
    before_surf.fix_all()
    _, surf_outboard, surf_inboard, grids = build_sharpened_twins(
        final_surf,
        SPLINE_KWARGS,
        boundary.d_crawl,
        D_EXT,
        L_X,
        CORNER_CRITERION,
        ntheta=60,
        n_samples=N_SAMPLES,
    )
    if mpi.proc0_world:
        pickle_twin_surface(
            surf_outboard,
            SPLINE_KWARGS,
            os.path.join(HERE, "corner_sharpen_opt_twin_outboard.pkl"),
        )
        pickle_twin_surface(
            surf_inboard,
            SPLINE_KWARGS,
            os.path.join(HERE, "corner_sharpen_opt_twin_inboard.pkl"),
        )
        proc0_print(
            "Pickled optimized twins (load with sharpen_fixed_s_twin.load_twin_surface)."
        )

    # Higher-resolution VMEC run on the optimized sharpened surface. A
    # failure here only loses the check, not the (already saved) results
    # or the plots below.
    vmec_final = Vmec.vmec_from_surf(
        nfp=low_surf.nfp, surf=final_surf, mpi=mpi, **VMEC_FINAL
    )
    try:
        vmec_final.run()
        qs_final = QuasisymmetryRatioResidual(
            vmec_final, np.arange(0, 1.01, 0.1), helicity_m=1, helicity_n=0
        )
        proc0_print(
            f"High-res check: QS total = {qs_final.total():.4e}, "
            f"iota = {vmec_final.mean_iota():.5f}, aspect = {vmec_final.aspect():.5f}"
        )
    except ObjectiveFailure as e:
        proc0_print(f"High-res check skipped: {e}")

    if mpi.proc0_world and not args.no_plots:
        plot_angles(
            goal_angles,
            initial_angles,
            final_angles,
            os.path.join(HERE, "corner_sharpen_opt_angles.png"),
        )
        fig, _ = plot_before_after_twin(
            before_surf,
            final_surf,
            grids["geometry"],
            surf_outboard,
            surf_inboard,
            grids["outboard_is_between"],
            L_X,
        )
        fig.savefig(
            os.path.join(HERE, "corner_sharpen_opt_cross_sections.png"), dpi=150
        )
        fig, _ = plot_leg_length_grid(
            final_surf,
            grids["geometry"],
            surf_outboard,
            surf_inboard,
            grids["outboard_is_between"],
            L_X,
            grids["phis"],
        )
        fig.savefig(os.path.join(HERE, "corner_sharpen_opt_legs.png"), dpi=150)
        plt.show()

    proc0_print("")
    proc0_print(
        "End of figures_spline_paper/corner_sharpening_opt/corner_sharpen_opt.py"
    )


if __name__ == "__main__":
    main()
