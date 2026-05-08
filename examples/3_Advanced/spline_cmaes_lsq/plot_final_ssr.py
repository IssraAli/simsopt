"""Plot final SSR vs trial id (colored by generation, log y, cropped <0.1)
with a running-best line, for a given cmaes_lsq run directory.

Usage:
    python plot_final_ssr.py [RUN_DIR]

If RUN_DIR is omitted, the most recent run under cmaes_lsq_outputs/ is used.
"""
import argparse
import csv
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np

OUTPUTS_DIR = pathlib.Path(__file__).parent / "cmaes_lsq_outputs"


def latest_run() -> pathlib.Path:
    runs = sorted(p for p in OUTPUTS_DIR.iterdir() if p.is_dir() and p.name.startswith("run_"))
    if not runs:
        sys.exit(f"no run_* directories under {OUTPUTS_DIR}")
    return runs[-1]


def load_results(csv_path: pathlib.Path):
    trial_id, generation, final_ssr = [], [], []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            trial_id.append(int(row["trial_id"]))
            generation.append(int(row["generation"]))
            final_ssr.append(float(row["final_ssr"]))
    arr = (np.array(trial_id), np.array(generation), np.array(final_ssr))
    order = np.argsort(arr[0])
    return tuple(a[order] for a in arr)


def plot(run_dir: pathlib.Path, ssr_cap: float = 0.1) -> pathlib.Path:
    trial_id, generation, final_ssr = load_results(run_dir / "results.csv")
    running_best = np.minimum.accumulate(final_ssr)
    mask = final_ssr < ssr_cap

    fig, ax = plt.subplots(figsize=(8, 5))
    sc = ax.scatter(
        trial_id[mask], final_ssr[mask],
        c=generation[mask], cmap="viridis",
        s=24, alpha=0.85, edgecolor="none",
    )
    ax.plot(trial_id, running_best, color="crimson", lw=1.5, label="running best")

    # Log-linear best-fit over the cropped points (average trend in log space).
    if mask.sum() >= 2:
        x_fit = trial_id[mask].astype(float)
        y_fit = np.log10(final_ssr[mask])
        slope, intercept = np.polyfit(x_fit, y_fit, 1)
        x_line = np.array([x_fit.min(), x_fit.max()])
        y_line = 10 ** (slope * x_line + intercept)
        ax.plot(
            x_line, y_line, color="black", lw=1.5, ls="--",
            label=f"log-linear fit (slope={slope:.2e}/trial)",
        )
    ax.set_yscale("log")
    ax.set_xlabel("trial id")
    ax.set_ylabel("final SSR")
    ax.set_ylim(top=ssr_cap)
    ax.set_title(f"Final SSR vs trial (cropped <{ssr_cap}) — {run_dir.name}")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("generation")
    ax.legend(loc="upper right")
    fig.tight_layout()

    out_dir = run_dir / "summary_plots"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / "final_ssr_vs_trial.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", nargs="?", type=pathlib.Path, default=None,
                    help="path to a run_YYYYMMDD_HHMMSS directory (default: latest)")
    ap.add_argument("--cap", type=float, default=0.1, help="upper bound for final_ssr crop")
    args = ap.parse_args()

    run_dir = args.run_dir or latest_run()
    out = plot(run_dir, ssr_cap=args.cap)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
