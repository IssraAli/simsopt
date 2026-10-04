#!/usr/bin/env python
"""
Draw SurfaceBSpline dof vectors at random from the feasible region used
in the constrained spline examples, and count how many of the resulting
boundaries VMEC converges on. The feasible region is:
  - the class's default box bounds on the cross-section thetas, the
    cross-section rotation angles (cs_angle, free unless
    --fixed-cs-angles) and the axis angles (zeta_axis) only, and
  - SurfaceBSpline.write_inequality_constraints() on everything else
    (a * r_cs <= mean(r_axis) with a = --cs-axis-ratio (default 2.6; 'none'
    gives r_cs <= every axis r instead), axis r <= first axis r or the
    fixed-scale cap (r_axis_0 is fixed at 1 unless --free-r-axis-0);
    |z_axis| <= --z-axis-max (0.5); r >= 0).
Every other dof has its box bound dropped, as in those examples. The
zeta_axis dofs get no linear constraints, only their box bounds.

Defaults aimed at aspect ratio ~3.5-8 (1.3 * a / F with F, the per-draw
mean cross-section fraction, log-uniform in --cs-size-range 0.46 0.92): on
256 draws 96% land in [3.5, 8]. VMEC runs with ns=51, M=N=8, a 64x64 grid,
ftol=1e-11, niter=5000; the spline's own Fourier transform uses M=N=8
(--spline-M).

Cross-section shapes (--cs-shape): 'random' draws independent radii; 'ellipse'
(sobol only) makes every cross section of a draw an ellipse of one
elongation (log-uniform up to --ellipse-max-elong) whose orientation turns by
N*pi/2 over the half period (N in -W..W, --ellipse-max-winding), the
near-axis way to produce iota; the cs_angle dofs are fixed then because the
end cross sections are pinned at angle 0 by stellarator symmetry, so a
winding has to live in the shape.

Two samplers (--sampler):
  sobol (default): a scrambled Sobol design in normalized coordinates that
    map a unit box onto the region, so scale and shape ratios are covered
    evenly (volume-uniform draws instead sit almost entirely at the
    large-radius corner, since the radius constraints are homogeneous):
      r_axis_0 = s * cap                  s in [min_scale, 1]  (if free)
      r_axis_i = r_axis_0 * u_i           u_i in [min_axis_ratio, 1]
                                          (cap * u_i if r_axis_0 is fixed at 1)
      r_cs_j   = min_i(r_axis_i) * f_j    f_j in [min_cs_frac, 1]
                 (or mean(r_axis)/a * f_j with --cs-axis-ratio a; with
                 --cs-size-range the f_j are rescaled per draw to mean F)
      z_axis_i = zmax * (2 w_i - 1)       w_i in [0, 1]
      theta_k, zeta_axis_i                uniform within their box bounds
    where cap = --axis-r-max (the fixed-scale cap if r_axis_0 is fixed)
    and zmax = --z-axis-max (--z-mode absolute) or
    min(--z-axis-max, r_axis_0) (relative).
    The lower limits on s, u and f keep the surface from degenerating
    (zero-radius axis or cross-section points); they are extra choices on
    top of the constraints, not part of them.
  hit-and-run: a random walk from the Chebyshev centre of the region,
    asymptotically uniform over its volume; successive draws are
    correlated, so they are thinned (--thin) after a burn-in (--burn-in).

Each draw is classified as
  - "converged":   vmec.run() succeeded,
  - "vmec_failed": vmec.run() raised ObjectiveFailure (did not converge /
                   ier_flag != 0),
  - "error":       any other exception (e.g. the boundary could not be
                   built from the draw); the message is recorded.
Whether the axis is toroidally monotonic (the assumption _solve_v relies
on) is recorded for every draw so failures can be broken down by it.

Draws are split across MPI worker groups (one group per process by
default). Results, including every draw, are saved to an .npz file.

Usage:
    mpirun -n 4 python random_draw_vmec_convergence.py --n-samples 200
"""

import argparse
import os
import time

import numpy as np
from scipy.optimize import linprog
from scipy.stats import qmc
from simsopt._core.util import ObjectiveFailure
from simsopt.geo import SurfaceBSpline
from simsopt.mhd import Vmec
from simsopt.util import MpiPartition, proc0_print

parser = argparse.ArgumentParser()
parser.add_argument("--n-samples", type=int, default=100)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--ns", type=int, default=51, help="VMEC radial surfaces")
parser.add_argument("--vmec-M", type=int, default=8)
parser.add_argument("--vmec-N", type=int, default=8)
parser.add_argument("--ntheta", type=int, default=64, help="VMEC poloidal grid")
parser.add_argument("--nzeta", type=int, default=64, help="VMEC toroidal grid")
parser.add_argument("--ftol", type=float, default=1e-11)
parser.add_argument(
    "--niter", type=int, default=5000, help="VMEC max iterations"
)
parser.add_argument("--ngroups", type=int, default=None)
parser.add_argument("--axis-r-max", type=float, default=1.0)
parser.add_argument("--z-axis-max", type=float, default=0.5)
def float_or_none(text):
    return None if text.lower() == "none" else float(text)


parser.add_argument(
    "--cs-axis-ratio",
    type=float_or_none,
    default=2.6,
    help="a in a*r_cs <= mean(r_axis) (default 2.6, which with the default "
    "--cs-size-range keeps the aspect ratio mostly in [3.5, 8]); 'none' "
    "uses r_cs <= every axis radius instead",
)
parser.add_argument("--spline-M", type=int, default=8, help="spline M = N")
parser.add_argument(
    "--cs-angle-step-max",
    type=float_or_none,
    default=-1.0,
    help="max |cs_angle_{k+1} - cs_angle_k| between neighboring cross "
    "sections; default (-1) = 2*pi/points_per_cs, 'none' = no limit",
)
parser.add_argument(
    "--cs-shape",
    choices=["random", "ellipse"],
    default="random",
    help="'random': independent cross-section radii. 'ellipse': each draw "
    "uses elliptical cross sections of one elongation whose orientation "
    "winds along the half period (sobol only; the cs_angle dofs are then "
    "fixed, since the orientation lives in the shape)",
)
parser.add_argument("--ellipse-max-elong", type=float, default=2.0)
parser.add_argument(
    "--ellipse-max-winding",
    type=int,
    default=1,
    help="ellipse orientation turns by N*pi/2 over the half period, with N "
    "uniform in -W..W (the end cross sections stay up-down symmetric)",
)
parser.add_argument("--ellipse-noise", type=float, default=0.15)
parser.add_argument(
    "--theta-spread",
    type=float,
    default=0.5,
    help="ellipse design: each theta stays within this fraction of its box "
    "width, centered on the box middle",
)
parser.add_argument(
    "--fixed-cs-angles",
    action="store_true",
    help="pin the cross-section rotation angles at 0 (default: free, "
    "bounded by the class's own +-pi/2 box bounds)",
)
parser.add_argument(
    "--cs-size-range",
    type=float,
    nargs=2,
    default=[0.46, 0.92],
    metavar=("LO", "HI"),
    help="per-draw mean cross-section fraction F, log-uniform in [LO, HI]; "
    "each draw's cross-section fractions are rescaled to mean F (sobol only; "
    "default 0.46 0.92). Aspect ratio ~ 1.3 * a / F.",
)
parser.add_argument(
    "--no-cs-size-range",
    action="store_true",
    help="skip the per-draw size factor (plain Sobol cross-section fractions)",
)
parser.add_argument(
    "--free-r-axis-0",
    action="store_true",
    help="leave r_axis_0 free (default: fixed at 1, which sets the scale)",
)
parser.add_argument(
    "--fixed-axis-angles",
    action="store_true",
    help="fix the axis angles (zeta_axis) instead of sampling them",
)
parser.add_argument(
    "--sampler", choices=["sobol", "hit-and-run"], default="sobol"
)
parser.add_argument("--min-scale", type=float, default=0.3)
parser.add_argument("--min-axis-ratio", type=float, default=0.3)
parser.add_argument("--min-cs-frac", type=float, default=0.05)
parser.add_argument(
    "--z-mode",
    choices=["absolute", "relative"],
    default="absolute",
    help="z_axis spans +-1 (absolute) or +-min(1, r_axis_0) (relative)",
)
parser.add_argument("--burn-in", type=int, default=2000)
parser.add_argument("--thin", type=int, default=100)
parser.add_argument(
    "--outdir",
    default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "vmec_scratch"
    ),
    help="working directory for VMEC's input/wout files and the results",
)
args = parser.parse_args()
args.outdir = os.path.abspath(args.outdir)  # the script chdirs into it
if args.no_cs_size_range:
    args.cs_size_range = None

mpi = MpiPartition(ngroups=args.ngroups)
mpi.write()

os.makedirs(args.outdir, exist_ok=True)
os.chdir(args.outdir)

spline_kwargs = {
    "axis_points": 3,
    "points_per_cs": 4,
    "n_cs": 5,
    "nfp": 2,
    "M": args.spline_M,
    "N": args.spline_M,
    "p_u": 3,
    "p_v": 3,
    "cs_equispaced": True,
    "rays_equispaced": False,
    "cs_global_angle_free": not (
        args.fixed_cs_angles or args.cs_shape == "ellipse"
    ),
    "axis_angles_fixed": False,
    "cs_basis": "polar",
    "nurbs": False,
    "use_bishop_frame": True,
}

if args.cs_angle_step_max == -1.0:
    args.cs_angle_step_max = 2 * np.pi / spline_kwargs["points_per_cs"]

spline_surf = SurfaceBSpline(**spline_kwargs)
if not args.free_r_axis_0:
    spline_surf.axis.set("r_axis_0", 1.0)
    spline_surf.axis.fix("r_axis_0")
# Built first: with --fixed-axis-angles this fixes the zeta_axis dofs,
# which changes the dof vector everything below is sized from.
A_lc, lb_lc, ub_lc, lc_titles = spline_surf.write_inequality_constraints(
    axis_r_max=args.axis_r_max,
    fix_axis_angles=args.fixed_axis_angles,
    z_axis_max=args.z_axis_max,
    cs_axis_ratio=args.cs_axis_ratio,
    cs_angle_step_max=args.cs_angle_step_max,
)
n_dofs = len(spline_surf.x)
names = spline_surf.dof_names

# Box bounds are kept only for thetas and axis angles; everything else is
# governed by the linear constraints (same split as the constrained
# examples).
keep_box = np.array(
    [(":theta_" in n) or (":zeta_axis_" in n) or (":cs_angle" in n) for n in names]
)
box_lb = np.where(keep_box, spline_surf.lower_bounds, -np.inf)
box_ub = np.where(keep_box, spline_surf.upper_bounds, np.inf)
zeta_cols = [i for i, n in enumerate(names) if ":zeta_axis_" in n]
assert not np.any(A_lc[:, zeta_cols]), "axis angles must be box-bounded only"

# Stack everything as G x <= h (finite sides only).
G_rows, h_rows = [], []
for i in np.where(keep_box)[0]:
    e = np.zeros(n_dofs)
    e[i] = 1.0
    G_rows += [e, -e]
    h_rows += [box_ub[i], -box_lb[i]]
for row, lo, hi in zip(A_lc, lb_lc, ub_lc):
    if np.isfinite(hi):
        G_rows.append(row)
        h_rows.append(hi)
    if np.isfinite(lo):
        G_rows.append(-row)
        h_rows.append(-lo)
G = np.array(G_rows)
h = np.array(h_rows)


def chebyshev_center(G, h):
    """Point maximizing the distance to the nearest constraint plane."""
    norms = np.linalg.norm(G, axis=1)
    c = np.zeros(G.shape[1] + 1)
    c[-1] = -1.0
    res = linprog(
        c,
        A_ub=np.hstack([G, norms[:, None]]),
        b_ub=h,
        bounds=[(None, None)] * G.shape[1] + [(0, None)],
    )
    if not res.success or res.x[-1] <= 0:
        raise RuntimeError("Feasible region is empty or has no interior.")
    return res.x[:-1]


def hit_and_run(G, h, x0, n_samples, burn_in, thin, rng):
    x = x0.copy()
    out = np.empty((n_samples, len(x0)))
    for step in range(burn_in + n_samples * thin):
        d = rng.standard_normal(len(x))
        d /= np.linalg.norm(d)
        Gd = G @ d
        slack = h - G @ x
        pos, neg = Gd > 1e-14, Gd < -1e-14
        t_hi = np.min(slack[pos] / Gd[pos]) if pos.any() else np.inf
        t_lo = np.max(slack[neg] / Gd[neg]) if neg.any() else -np.inf
        if not (np.isfinite(t_hi) and np.isfinite(t_lo)):
            raise RuntimeError("Feasible region is unbounded.")
        x = x + rng.uniform(t_lo, t_hi) * d
        if step >= burn_in and (step - burn_in) % thin == thin - 1:
            out[(step - burn_in) // thin] = x
    return out


vmec = Vmec.vmec_from_surf(
    nfp=spline_surf.nfp,
    surf=spline_surf,
    mpi=mpi,
    ns=args.ns,
    M=args.vmec_M,
    N=args.vmec_N,
    ftol=args.ftol,
    niter=args.niter,
    ntheta=args.ntheta,
    nzeta=args.nzeta,
)


def sobol_design(n_samples, seed):
    """Scrambled Sobol design in the normalized coordinates described in
    the module docstring, returned as (n_samples, n_dofs) in dof order."""
    col = {name: i for i, name in enumerate(names)}
    axis_r = [n for n in names if ":r_axis_" in n]
    r0_name = f"{spline_surf.axis.name}:r_axis_0"
    r0_free = r0_name in col
    cs_r = [n for n in names if ":r_" in n and ":r_axis_" not in n]
    z_axis = [n for n in names if ":z_axis_" in n]
    box = [n for n, k in zip(names, keep_box) if k]
    known = set(axis_r) | set(cs_r) | set(z_axis) | set(box)
    unknown = [n for n in names if n not in known]
    assert not unknown, f"Sobol design has no mapping for dofs {unknown}"

    # one unit-cube coordinate per free dof, in this order
    cube_names = (
        ([r0_name] if r0_free else [])
        + [n for n in axis_r if n != r0_name]
        + cs_r
        + z_axis
        + box
        + (["__size__"] if args.cs_size_range else [])
        + (["__elong__", "__wind__"] if args.cs_shape == "ellipse" else [])
    )
    m = int(np.ceil(np.log2(max(n_samples, 2))))
    cube = qmc.Sobol(d=len(cube_names), scramble=True, seed=seed).random_base2(
        m
    )[:n_samples]
    c = {n: cube[:, i] for i, n in enumerate(cube_names)}

    cap = args.axis_r_max if r0_free else 1.0  # axis_r_fixed_max default
    out = np.zeros((n_samples, n_dofs))
    if r0_free:
        a0 = cap * (args.min_scale + (1 - args.min_scale) * c[r0_name])
        out[:, col[r0_name]] = a0
    else:
        a0 = np.full(n_samples, spline_surf.axis.get("r_axis_0"))
    other_axis = []  # axis radii i >= 1
    for n in axis_r:
        if n == r0_name:
            continue
        u = args.min_axis_ratio + (1 - args.min_axis_ratio) * c[n]
        v = (a0 if r0_free else cap) * u
        out[:, col[n]] = v
        other_axis.append(v)
    if args.cs_axis_ratio is None:
        # r_cs <= every axis radius; r_axis_0 is redundant unless alone
        cs_cap = np.min(other_axis, axis=0) if other_axis else a0
    else:
        # a * r_cs <= mean over all axis radii, r_axis_0 included
        cs_cap = np.mean([a0] + other_axis, axis=0) / args.cs_axis_ratio
    fr = {n: args.min_cs_frac + (1 - args.min_cs_frac) * c[n] for n in cs_r}
    if args.cs_size_range:
        lo, hi = args.cs_size_range
        F = np.exp(np.log(lo) + (np.log(hi) - np.log(lo)) * c["__size__"])
        scale = F / np.mean([fr[n] for n in cs_r], axis=0)
        fr = {
            n: np.clip(fr[n] * scale, args.min_cs_frac, 1.0) for n in cs_r
        }
    for n in cs_r:
        out[:, col[n]] = cs_cap * fr[n]
    zmax = (
        args.z_axis_max
        if args.z_mode == "absolute"
        else np.minimum(args.z_axis_max, a0)
    )
    for n in z_axis:
        out[:, col[n]] = zmax * (2 * c[n] - 1)
    # cs angles: map sequentially into [max(box lo, prev - step), min(box hi,
    # prev + step)], also keeping within step of the next fixed neighbor,
    # so the neighbor-step rows hold by construction
    step = args.cs_angle_step_max
    ang = sorted(
        [n for n in box if ":cs_angle" in n],
        key=lambda n: int(n.split("cs_angle")[-1]),
    )
    prev = np.zeros(n_samples)  # cs_angle0 is fixed at 0 (stellarator symmetry)
    n_cs = spline_kwargs["n_cs"]
    for n in ang:
        i = col[n]
        k = int(n.split("cs_angle")[-1])
        lo, hi = box_lb[i] * np.ones(n_samples), box_ub[i] * np.ones(n_samples)
        if step is not None:
            lo, hi = np.maximum(lo, prev - step), np.minimum(hi, prev + step)
            # distance to the fixed last cross section (angle 0) after
            # n_cs-1-k further steps
            reach = step * (n_cs - 1 - k)
            lo, hi = np.maximum(lo, -reach), np.minimum(hi, reach)
        out[:, i] = lo + (hi - lo) * c[n]
        prev = out[:, i]
    for n in box:
        if n in ang:
            continue
        i = col[n]
        if args.cs_shape == "ellipse" and ":theta_" in n:
            mid = 0.5 * (box_lb[i] + box_ub[i])
            out[:, i] = mid + args.theta_spread * (c[n] - 0.5) * (
                box_ub[i] - box_lb[i]
            )
        else:
            out[:, i] = box_lb[i] + (box_ub[i] - box_lb[i]) * c[n]
    if args.cs_shape == "ellipse":
        assert args.cs_size_range, "--cs-shape ellipse needs --cs-size-range"
        lo, hi = args.cs_size_range
        F = np.exp(np.log(lo) + (np.log(hi) - np.log(lo)) * c["__size__"])
        elong = np.exp(c["__elong__"] * np.log(args.ellipse_max_elong))
        a_ax, b_ax = np.sqrt(elong), 1 / np.sqrt(elong)  # a * b = 1
        W = args.ellipse_max_winding
        wind = np.minimum((c["__wind__"] * (2 * W + 1)).astype(int), 2 * W) - W
        for k, cs in enumerate(spline_surf.cs_list):
            psi = wind * (np.pi / 2) * k / (n_cs - 1)
            rho, rnames = [], []
            for j in range(cs.n_pts):
                rn = f"{cs.name}:r_{j}"
                if rn not in col:
                    continue
                tn = f"{cs.name}:theta_{j}"
                th = out[:, col[tn]] if tn in col else cs.get(f"theta_{j}")
                phi = th - psi
                rho.append(
                    a_ax * b_ax
                    / np.sqrt(
                        (b_ax * np.cos(phi)) ** 2 + (a_ax * np.sin(phi)) ** 2
                    )
                )
                rnames.append(rn)
            rho = np.array(rho)
            rho_hat = rho / rho.mean(axis=0)
            for rn, rh in zip(rnames, rho_hat):
                f = F * rh * (1 + 2 * args.ellipse_noise * (c[rn] - 0.5))
                out[:, col[rn]] = cs_cap * np.clip(f, args.min_cs_frac, 1.0)
    return out


# Same draws on every process (seeded), so each group can just take its
# own share of the indices.
if args.sampler == "sobol":
    draws = sobol_design(args.n_samples, args.seed)
else:
    rng = np.random.default_rng(args.seed)
    draws = hit_and_run(
        G,
        h,
        chebyshev_center(G, h),
        args.n_samples,
        args.burn_in,
        args.thin,
        rng,
    )
max_violation = np.max(G @ draws.T - h[:, None])
assert max_violation < 1e-9, f"draws violate constraints by {max_violation}"

proc0_print(
    f"ndofs: {n_dofs}, {G.shape[0]} constraint rows, n_samples: {args.n_samples}, seed: {args.seed}"
)
proc0_print(f"sampler: {args.sampler}")
proc0_print(f"dof names: {names}")

results = []
for i in range(mpi.group, args.n_samples, mpi.ngroups):
    rec = {
        "index": i,
        "status": "",
        "message": "",
        "monotonic": False,
        "aspect": np.nan,
        "mean_iota": np.nan,
        "seconds": 0.0,
    }
    t0 = time.time()
    try:
        spline_surf.x = draws[i]
        rec["monotonic"] = bool(spline_surf.axis.is_toroidally_monotonic())
        vmec.run()
        rec["status"] = "converged"
        rec["aspect"] = float(vmec.aspect())
        rec["mean_iota"] = float(vmec.mean_iota())
    except ObjectiveFailure as e:
        rec["status"] = "vmec_failed"
        rec["message"] = str(e)
    except Exception as e:  # noqa: BLE001 -- recorded, not hidden
        rec["status"] = "error"
        rec["message"] = f"{type(e).__name__}: {e}"
    rec["seconds"] = time.time() - t0
    results.append(rec)
    print(
        f"[group {mpi.group}] draw {i}: {rec['status']} "
        f"(monotonic={rec['monotonic']}, {rec['seconds']:.1f}s) {rec['message']}",
        flush=True,
    )

# Only one process per group (the leader) holds that group's results.
if mpi.proc0_groups:
    gathered = mpi.comm_leaders.gather(results, root=0)
else:
    gathered = None

if mpi.proc0_world:
    all_results = sorted(
        (r for part in gathered for r in part), key=lambda r: r["index"]
    )
    status = np.array([r["status"] for r in all_results])
    monotonic = np.array([r["monotonic"] for r in all_results])
    n = len(all_results)
    n_conv = int(np.sum(status == "converged"))
    p = n_conv / n
    print("\n==================== Summary ====================")
    print(f"draws:        {n}")
    print(
        f"converged:    {n_conv} ({100 * p:.1f}%, "
        f"+/- {100 * np.sqrt(p * (1 - p) / n):.1f}% standard error)"
    )
    print(f"vmec_failed:  {int(np.sum(status == 'vmec_failed'))}")
    print(f"error:        {int(np.sum(status == 'error'))}")
    for flag in (True, False):
        m = monotonic == flag
        if m.any():
            print(
                f"axis monotonic={flag}: {int(m.sum())} draws, "
                f"{int(np.sum(status[m] == 'converged'))} converged"
            )
    out = os.path.join(args.outdir, f"random_draw_results_seed{args.seed}.npz")
    np.savez(
        out,
        draws=draws[[r["index"] for r in all_results]],
        index=np.array([r["index"] for r in all_results]),
        status=status,
        monotonic=monotonic,
        aspect=np.array([r["aspect"] for r in all_results]),
        mean_iota=np.array([r["mean_iota"] for r in all_results]),
        seconds=np.array([r["seconds"] for r in all_results]),
        message=np.array([r["message"] for r in all_results]),
        dof_names=np.array(spline_surf.dof_names),
    )
    print(f"saved {out}")
