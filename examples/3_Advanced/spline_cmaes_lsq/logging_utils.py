"""
Per-trial snapshots + summary plots for the CMA-ES + LSQ Baldwinian wrapper.

Three plots are produced after the run:
  1. `optuna_history.png`     — Optuna's `plot_optimization_history` (trial-best
                                 final SSR vs. trial; CMA-ES outer convergence).
  2. `inner_lsq_traces.png`   — Per-trial inner-LSQ SSR trace overlaid, colored
                                 by generation, plus a step-down best-so-far
                                 envelope. The user-requested complement to the
                                 Optuna history: shows what the *inside* of each
                                 CMA-ES trial looked like, not just its endpoint.
  3. `convergence_diagnostics.png` — Per-trial nfev, status code, and the
                                 init→final SSR drop, useful for spotting trials
                                 where the LSQ stalled or VMEC blew up.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


# --------------------------------------------------------------------- snapshots

def save_trial_snapshot(snapshot_dir: Path, trial_id: int, generation: int,
                        result: dict[str, Any]) -> Path:
    """Persist a single LSQ trial's full result + trace to JSON."""
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        'trial_id': int(trial_id),
        'generation': int(generation),
        'init_x_unit': result['init_x_unit'],
        'init_x_phys': result['init_x_phys'],
        'final_x_unit': result['final_x_unit'],
        'final_x_phys': result['final_x_phys'],
        'init_ssr': result['init_ssr'],
        'final_ssr': result['final_ssr'],
        'nfev': result['nfev'],
        'status': result.get('status'),
        'message': result.get('message'),
        'trace': result['trace'],
    }
    out = snapshot_dir / f'trial_{trial_id:04d}.json'
    with open(out, 'w') as f:
        json.dump(payload, f, indent=2)
    return out


def write_results_csv(csv_path: Path, trial_records: list[dict[str, Any]]) -> None:
    """One row per trial: trial_id, generation, init_ssr, final_ssr, nfev, status."""
    import csv
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['trial_id', 'generation', 'init_ssr', 'final_ssr',
                    'ssr_drop', 'nfev', 'status', 'message'])
        for rec in trial_records:
            init_ssr = rec['init_ssr']
            final_ssr = rec['final_ssr']
            drop = (init_ssr - final_ssr) if np.isfinite(init_ssr) and np.isfinite(final_ssr) else float('nan')
            w.writerow([
                rec['trial_id'], rec['generation'],
                init_ssr, final_ssr, drop,
                rec['nfev'], rec.get('status'), rec.get('message'),
            ])


# --------------------------------------------------------------------- plots

def _generation_palette(n_gens: int):
    """Return n_gens distinguishable colors from a perceptually-uniform colormap."""
    cmap = plt.get_cmap('viridis')
    if n_gens <= 1:
        return [cmap(0.5)]
    return [cmap(i / max(n_gens - 1, 1)) for i in range(n_gens)]


def plot_optuna_history(study, out_path: Path) -> None:
    """Optuna's built-in convergence-history plot (best-so-far envelope)."""
    from optuna.visualization.matplotlib import plot_optimization_history
    ax = plot_optimization_history(study)
    fig = ax.figure
    fig.set_size_inches(11, 6)
    ax.set_title('CMA-ES outer-loop convergence (best-so-far)')
    ax.set_ylabel('-SSR (Optuna maximizes)')
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def plot_inner_lsq_traces(
    trial_records: list[dict[str, Any]],
    out_path: Path,
    log_y: bool = True,
) -> None:
    """
    Overlay every trial's inner LSQ SSR trace on a single axis, colored by CMA-ES
    generation. Adds a step-down 'best-so-far' envelope across all trials in the
    order the trials completed.

    X-axis: a single global eval index (each trial's trace is concatenated end
    to end), so the plot reads left-to-right as wall-clock VMEC effort. The
    user can read "did this generation make progress" from the slope of its
    color band, and "how good is the best" from the white envelope.
    """
    if not trial_records:
        return

    fig, (ax_traces, ax_endpoints) = plt.subplots(
        2, 1, figsize=(13, 9), height_ratios=[2.2, 1.0], sharex=False,
    )

    # Per-trial inner LSQ traces, concatenated.
    n_gens = max(rec['generation'] for rec in trial_records) + 1
    palette = _generation_palette(n_gens)

    global_eval_offset = 0
    best_so_far_x: list[int] = []
    best_so_far_y: list[float] = [float('inf')]
    running_best = float('inf')

    for rec in trial_records:
        trace = rec.get('trace', [])
        if not trace:
            continue
        ssrs = np.array([t['ssr'] for t in trace], dtype=np.float64)
        # Replace inf/nan with a sentinel so log-scale plotting still works
        finite_mask = np.isfinite(ssrs)
        if not finite_mask.any():
            global_eval_offset += len(trace)
            continue

        xs = np.arange(len(ssrs)) + global_eval_offset
        color = palette[rec['generation']]
        ax_traces.plot(
            xs[finite_mask], ssrs[finite_mask],
            color=color, alpha=0.55, linewidth=1.0,
        )

        # Mark trial start/end with markers
        ax_traces.scatter(
            xs[finite_mask][0], ssrs[finite_mask][0],
            color=color, s=18, marker='o', edgecolors='k', linewidths=0.4, zorder=4,
        )
        ax_traces.scatter(
            xs[finite_mask][-1], ssrs[finite_mask][-1],
            color=color, s=28, marker='s', edgecolors='k', linewidths=0.4, zorder=5,
        )

        # Best-so-far envelope: track the running min over all observed SSRs.
        for x, y in zip(xs[finite_mask], ssrs[finite_mask]):
            if y < running_best:
                running_best = float(y)
            best_so_far_x.append(int(x))
            best_so_far_y.append(running_best)

        global_eval_offset += len(trace)

    # Drop the leading inf used to seed the envelope
    best_so_far_y = best_so_far_y[1:]
    if best_so_far_x:
        ax_traces.plot(
            best_so_far_x, best_so_far_y,
            color='k', linewidth=2.0, label='Best-so-far SSR',
        )

    ax_traces.set_xlabel('Global VMEC evaluation index (concatenated across trials)')
    ax_traces.set_ylabel('SSR (Σ residuals²)')
    ax_traces.set_title(
        'Inner-LSQ SSR traces by generation, with running best-so-far envelope'
    )
    if log_y:
        ax_traces.set_yscale('log')
    ax_traces.grid(True, which='both', alpha=0.3)

    # Color-bar-like generation legend (compressed to a few entries).
    n_legend = min(n_gens, 8)
    if n_gens > 1:
        for i in np.linspace(0, n_gens - 1, n_legend, dtype=int):
            ax_traces.plot([], [], color=palette[i], linewidth=2.5,
                           label=f'gen {i}')
    ax_traces.legend(loc='upper right', fontsize=9, ncols=2, framealpha=0.9)

    # Per-trial endpoint scatter: a cleaner version of Optuna's history but
    # colored by generation, so you can see the population's spread per gen.
    trial_idx = np.arange(len(trial_records))
    final_ssrs = np.array([
        r['final_ssr'] if np.isfinite(r['final_ssr']) else np.nan
        for r in trial_records
    ])
    init_ssrs = np.array([
        r['init_ssr'] if np.isfinite(r['init_ssr']) else np.nan
        for r in trial_records
    ])
    colors = [palette[r['generation']] for r in trial_records]
    ax_endpoints.scatter(trial_idx, final_ssrs, c=colors, s=40, marker='s',
                         edgecolors='k', linewidths=0.4, label='Final SSR',
                         zorder=4)
    ax_endpoints.scatter(trial_idx, init_ssrs, c=colors, s=22, marker='o',
                         edgecolors='k', linewidths=0.4, alpha=0.6,
                         label='Initial SSR', zorder=3)
    # init→final segments
    for i, rec in enumerate(trial_records):
        if np.isfinite(rec['init_ssr']) and np.isfinite(rec['final_ssr']):
            ax_endpoints.plot(
                [i, i], [rec['init_ssr'], rec['final_ssr']],
                color=palette[rec['generation']], alpha=0.4, linewidth=1.0,
                zorder=2,
            )

    # Best-so-far over trials only (i.e., over endpoints) — overlay
    finite = np.isfinite(final_ssrs)
    if finite.any():
        running = np.minimum.accumulate(np.where(finite, final_ssrs, np.inf))
        ax_endpoints.plot(trial_idx, running, color='k', linewidth=1.8,
                          label='Best-so-far (endpoints)', zorder=5)

    ax_endpoints.set_xlabel('Trial index')
    ax_endpoints.set_ylabel('SSR')
    if log_y:
        ax_endpoints.set_yscale('log')
    ax_endpoints.set_title('Per-trial init → final SSR')
    ax_endpoints.grid(True, which='both', alpha=0.3)
    ax_endpoints.legend(loc='upper right', fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def plot_convergence_diagnostics(
    trial_records: list[dict[str, Any]],
    out_path: Path,
) -> None:
    """nfev and LSQ status by trial — useful to spot stalls or VMEC failures."""
    if not trial_records:
        return
    fig, (ax_nfev, ax_status) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)

    trial_idx = np.arange(len(trial_records))
    nfev_vals = np.array([r['nfev'] for r in trial_records])
    n_gens = max(r['generation'] for r in trial_records) + 1
    palette = _generation_palette(n_gens)
    colors = [palette[r['generation']] for r in trial_records]

    ax_nfev.bar(trial_idx, nfev_vals, color=colors, edgecolor='k', linewidth=0.4)
    ax_nfev.set_ylabel('nfev (LSQ inner)')
    ax_nfev.grid(True, axis='y', alpha=0.3)
    ax_nfev.set_title('Inner LSQ effort + termination status')

    # Map status to integer (None → -1, scipy returns 0 or positive)
    status_vals = [r.get('status') if r.get('status') is not None else -1
                   for r in trial_records]
    ax_status.scatter(trial_idx, status_vals, c=colors, s=40,
                      edgecolors='k', linewidths=0.4)
    ax_status.set_ylabel('LSQ result.status\n(-1=no result; 0=max nfev;\n1=gtol; 2=ftol; 3=xtol; 4=ftol+xtol)')
    ax_status.set_xlabel('Trial index')
    ax_status.grid(True, alpha=0.3)
    ax_status.set_yticks([-1, 0, 1, 2, 3, 4])

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


# --------------------------------------------------------------------- manifest

def write_run_manifest(manifest_path: Path, run_metadata: dict[str, Any]) -> None:
    with open(manifest_path, 'w') as f:
        json.dump(run_metadata, f, indent=2, default=str)
