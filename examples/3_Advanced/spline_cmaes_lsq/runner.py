#!/usr/bin/env python
"""
CMA-ES + LSQ-TRF Baldwinian wrapper for the QA + iota stellarator problem.

Outer: Optuna `CmaEsSampler` samples the full DOF vector in [0, 1]^d.
Inner: `bounded_least_squares_mpi_solve` (TRF + finite-diff Jacobian) starting
       at that sampled point, capped at `--max-nfev` evaluations.
Baldwinian: only the post-LSQ scalar `-SSR` is reported back to CMA-ES; the
            CMA-ES sample is *not* replaced with the locally-optimized point.
            (Same convention as the well-placement reference script.)

MPI: single group; one CMA-ES trial at a time. All ranks run inside the inner
LSQ together (FD Jacobian parallelizes across ranks via MPIFiniteDifference).
The outer Optuna loop runs only on rank 0; other ranks enter the LSQ helper
and follow the leader.

Usage:
    conda activate simsopt-env
    mpiexec -n 4 python runner.py --n-trials 60 --popsize 6 --max-nfev 500
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import optuna
from mpi4py import MPI
from optuna.distributions import FloatDistribution
from optuna.samplers import CmaEsSampler

from simsopt.util.mpi import MpiPartition

# Local modules
from logging_utils import (
    plot_convergence_diagnostics,
    plot_inner_lsq_traces,
    plot_optuna_history,
    save_trial_snapshot,
    write_results_csv,
    write_run_manifest,
)
from lsq_inner import (
    build_template_surf_and_bounds,
    run_lsq_from_dofs,
)


DEFAULT_SPLINE_KWARGS = {
    'axis_points': 3,
    'points_per_cs': 6,
    'n_cs': 4,
    'nfp': 2,
    'M': 6,
    'N': 6,
    'p_u': 3,
    'p_v': 3,
    'cs_equispaced': True,
    'rays_equispaced': False,
    'cs_global_angle_free': False,
    'axis_angles_fixed': True,
    'cs_basis': 'polar',
    'nurbs': False,
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='CMA-ES + LSQ Baldwinian stellarator BO')
    p.add_argument('--n-trials', type=int, default=60,
                   help='Total CMA-ES trials (one inner LSQ each).')
    p.add_argument('--popsize', type=int, default=6,
                   help='CMA-ES population size per generation.')
    p.add_argument('--max-nfev', type=int, default=500,
                   help='Inner LSQ max function evaluations per trial.')
    p.add_argument('--abs-step', type=float, default=1e-4)
    p.add_argument('--rel-step', type=float, default=1e-8)
    p.add_argument('--diff-method', type=str, default='forward',
                   choices=['forward', 'centered'])
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--default-r', type=float, default=0.4,
                   help='Initial circular minor radius (used to construct bounds + warm reference).')
    p.add_argument('--output-root', type=str, default='cmaes_lsq_outputs',
                   help='Root for snapshots / plots / manifest.')
    return p.parse_args()


def _make_run_dirs(output_root: Path) -> dict[str, Path]:
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = output_root / f'run_{timestamp}'
    paths = {
        'run': run_dir,
        'snapshots': run_dir / 'trial_snapshots',
        'plots': run_dir / 'summary_plots',
        'lsq_workdirs': run_dir / 'lsq_workdirs',
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths | {'timestamp': timestamp}


def _build_distributions(dims: int) -> dict[str, FloatDistribution]:
    """Pre-declare all parameters as a single 'fixed search space' so
    CmaEsSampler can sample them jointly from trial 0 (avoids the
    `RandomSampler fallback because of dynamic search space` warning)."""
    return {f'd{i}': FloatDistribution(0.0, 1.0) for i in range(dims)}


def _trial_unit_x(trial: optuna.Trial, dims: int) -> np.ndarray:
    """Read the dims-vector of unit-cube params from a Trial whose params were
    pre-populated via `study.ask(fixed_distributions=...)`."""
    return np.array([trial.params[f'd{i}'] for i in range(dims)])


def _broadcast_payload(comm, payload):
    """Rank 0 broadcasts a per-trial payload to all ranks.
    Payload is either a dict {'unit_x': ndarray, 'workdir': str} or `None`
    (the latter signals workers to exit the loop)."""
    return comm.bcast(payload, root=0)


def main() -> None:
    args = _parse_args()
    comm = MPI.COMM_WORLD
    world_rank = comm.Get_rank()

    # One worker group per rank — each rank runs an independent VMEC during the
    # forward-diff Jacobian, matching `spline_surface.py`'s default. With
    # ngroups=1 the FD perturbations would serialize through one VMEC at a time
    # (~Nx slower for N ranks); ngroups=None lets simsopt pick `nprocs_world`,
    # i.e. one group per rank, so 31 perturbations per LSQ iter fan out over
    # the whole pool.
    mpi = MpiPartition()

    # Build template surface, gauge fix, and physical bounds. All ranks need
    # `lb`/`ub`/`dims` to participate in the inner LSQ.
    _template_surf, lb, ub = build_template_surf_and_bounds(
        DEFAULT_SPLINE_KWARGS, default_r=args.default_r,
    )
    dims = int(lb.shape[0])

    # Output dirs (rank 0 only writes; other ranks just need to know nothing).
    paths = None
    if world_rank == 0:
        paths = _make_run_dirs(Path(args.output_root))
        print(f'Run output dir: {paths["run"]}')
        print(f'dims (after gauge fix): {dims}')
        print(f'lb: {lb}')
        print(f'ub: {ub}')
        sys.stdout.flush()

    # ----------------------------------------------------------- main loop
    distributions = _build_distributions(dims)

    if world_rank == 0:
        # `warn_independent_sampling=False`: CmaEsSampler always falls back to
        # RandomSampler for trial 0 (it needs ≥1 completed trial before
        # building its covariance), which would otherwise emit one warning
        # per parameter. `n_startup_trials=0` means trials 1+ use CMA-ES.
        sampler = CmaEsSampler(seed=args.seed, popsize=args.popsize,
                               n_startup_trials=0,
                               warn_independent_sampling=False)
        study = optuna.create_study(direction='maximize', sampler=sampler)

        trial_records: list[dict] = []
        run_start = time.time()

        for trial_index in range(args.n_trials):
            # `fixed_distributions` declares the full 31-dim search space up
            # front so CmaEsSampler doesn't fall back to RandomSampler with
            # the dynamic-search-space warning.
            trial = study.ask(fixed_distributions=distributions)
            unit_x = _trial_unit_x(trial, dims)
            generation = trial_index // args.popsize

            trial_workdir = paths['lsq_workdirs'] / f'trial_{trial_index:04d}'
            trial_workdir.mkdir(parents=True, exist_ok=True)

            # Broadcast both unit_x AND workdir — all ranks must chdir to the
            # same dir so VMEC's writer (rank 0) and readers (all others)
            # agree on `wout_*.nc`'s location.
            _broadcast_payload(comm, {
                'unit_x': unit_x,
                'workdir': str(trial_workdir),
            })

            t0 = time.time()
            result = run_lsq_from_dofs(
                unit_cube_x=unit_x,
                spline_kwargs=DEFAULT_SPLINE_KWARGS,
                lb=lb, ub=ub, mpi=mpi,
                max_nfev=args.max_nfev,
                abs_step=args.abs_step, rel_step=args.rel_step,
                diff_method=args.diff_method,
                work_dir=str(trial_workdir),
            )
            wall_s = time.time() - t0

            final_ssr = result['final_ssr']
            # Optuna maximizes; we want to minimize SSR. Use -SSR. Sentinel
            # huge SSR (failed VMEC, NaN) → very negative score so CMA-ES
            # learns to avoid that region.
            if not np.isfinite(final_ssr):
                study.tell(trial, -1e12)
            else:
                study.tell(trial, -final_ssr)

            # Persist
            save_trial_snapshot(paths['snapshots'], trial_index, generation, result)
            rec = {
                'trial_id': trial_index,
                'generation': generation,
                'init_ssr': result['init_ssr'],
                'final_ssr': result['final_ssr'],
                'nfev': result['nfev'],
                'status': result.get('status'),
                'message': result.get('message'),
                'wall_s': wall_s,
                'trace': result['trace'],
            }
            trial_records.append(rec)

            print(
                f'[trial {trial_index:4d}/gen {generation:3d}] '
                f'init_ssr={result["init_ssr"]:.4e}  '
                f'final_ssr={result["final_ssr"]:.4e}  '
                f'nfev={result["nfev"]:4d}  '
                f'wall={wall_s:6.1f}s  '
                f'status={result.get("status")}'
            )

            # Per-generation summary at population boundaries
            if (trial_index + 1) % args.popsize == 0 or trial_index + 1 == args.n_trials:
                gen_records = [r for r in trial_records if r['generation'] == generation]
                gen_finals = np.array([r['final_ssr'] for r in gen_records])
                gen_finals_finite = gen_finals[np.isfinite(gen_finals)]
                if gen_finals_finite.size:
                    print(
                        f'  -- gen {generation} done: '
                        f'best={gen_finals_finite.min():.4e}, '
                        f'median={np.median(gen_finals_finite):.4e}, '
                        f'worst={gen_finals_finite.max():.4e}, '
                        f'feasible={gen_finals_finite.size}/{len(gen_records)}'
                    )
                else:
                    print(f'  -- gen {generation}: ALL TRIALS FAILED VMEC')
            sys.stdout.flush()

        # Tell workers to exit (None payload is the sentinel)
        _broadcast_payload(comm, None)

        run_wall_s = time.time() - run_start

        # ---------------------------------------------------- output artifacts
        plot_optuna_history(study, paths['plots'] / 'optuna_history.png')
        plot_inner_lsq_traces(trial_records, paths['plots'] / 'inner_lsq_traces.png')
        plot_convergence_diagnostics(
            trial_records, paths['plots'] / 'convergence_diagnostics.png'
        )
        write_results_csv(paths['run'] / 'results.csv', trial_records)
        write_run_manifest(paths['run'] / 'manifest.json', {
            'created_at': datetime.now().isoformat(timespec='seconds'),
            'n_trials': args.n_trials,
            'popsize': args.popsize,
            'max_nfev': args.max_nfev,
            'abs_step': args.abs_step,
            'rel_step': args.rel_step,
            'diff_method': args.diff_method,
            'seed': args.seed,
            'default_r': args.default_r,
            'spline_kwargs': DEFAULT_SPLINE_KWARGS,
            'dims': dims,
            'lb': lb.tolist(),
            'ub': ub.tolist(),
            'wall_seconds': run_wall_s,
            'best_trial': {
                'value': float(study.best_trial.value),
                'final_ssr': float(-study.best_trial.value),
                'params': dict(study.best_trial.params),
            } if study.best_trial else None,
            'trial_summary': [
                {k: r[k] for k in ('trial_id', 'generation', 'init_ssr',
                                   'final_ssr', 'nfev', 'status', 'wall_s')}
                for r in trial_records
            ],
        })

        print(f'\nRun complete in {run_wall_s:.1f}s '
              f'({run_wall_s/max(args.n_trials,1):.1f}s/trial avg).')
        if study.best_trial:
            print(f'Best final SSR: {-study.best_trial.value:.6e}')
        print(f'Outputs in: {paths["run"]}')

    else:
        # Worker ranks: loop forever, taking each broadcast payload and joining
        # the LSQ. `None` payload = exit signal.
        while True:
            payload = _broadcast_payload(comm, None)
            if payload is None:
                break
            unit_x = payload['unit_x']
            workdir = payload['workdir']
            # Workers don't write snapshots and don't care about the return
            # value; they exist to participate in the FD Jacobian under
            # `bounded_least_squares_mpi_solve`. They MUST chdir to the same
            # directory rank 0 uses so VMEC's `wout_*.nc` write/read agrees.
            try:
                run_lsq_from_dofs(
                    unit_cube_x=unit_x,
                    spline_kwargs=DEFAULT_SPLINE_KWARGS,
                    lb=lb, ub=ub, mpi=mpi,
                    max_nfev=args.max_nfev,
                    abs_step=args.abs_step, rel_step=args.rel_step,
                    diff_method=args.diff_method,
                    work_dir=workdir,
                )
            except Exception as e:
                # A worker exception here would deadlock; log and continue.
                # `bounded_least_squares_mpi_solve` already swallows per-eval
                # exceptions, but defensive logging in case something at the
                # outer wrapper level breaks.
                print(f'[worker rank {world_rank}] exception during LSQ: {e!r}',
                      file=sys.stderr)


if __name__ == '__main__':
    main()
