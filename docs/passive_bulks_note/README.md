# Passive bulks technical note

This directory contains the LaTeX source, figure-generation scripts, and
compiled PDFs for the technical note that documents the passive-bulk
(`PSCBulkArray`, `PassiveBulkField`) implementation added in
`src/simsopt/field/psc_bulk.py`,
`src/simsopt/field/bulk_inductance.py`,
`src/simsopt/field/disc_self_inductance.py`, and
`src/simsopt/field/puck_basis.py`.

## Layout

```
docs/passive_bulks_note/
├── passive_bulks_note.tex      main LaTeX source
├── refs.bib                    bibliography
├── Makefile                    build driver (make figures / make pdf / make all)
├── README.md                   this file
├── figures/                    compiled PDF figures (committed)
└── scripts/                    standalone figure-generation scripts
    ├── _common.py              shared matplotlib style + ring-coil helper
    └── fig_*.py                one script per figure
```

## Building

All commands assume the `stellcoilbench_py312` conda environment is active
and the working directory is `docs/passive_bulks_note/`.

```sh
conda activate stellcoilbench_py312

# 1. Regenerate all figures (each script writes figures/<name>.pdf).
make figures

# 2. Compile the PDF.
make pdf

# Or, in one go:
make all
```

Each `scripts/fig_*.py` can also be invoked directly, e.g.

```sh
python scripts/fig_mode_convergence.py
```

to regenerate the single figure it owns.

## Runtime

Most figure scripts finish in a few seconds. The two slowest are:

- `fig_mode_convergence.py` - builds a reference high-resolution solve and
  three coarser solves; ~20 s on a laptop.
- `fig_optimization_convergence.py` - runs a short `scipy.optimize.minimize`
  loop on a small puck array; ~30 s on a laptop.

All scripts write a single line starting with `[audit]` to stdout that
numerically summarizes the figure, so the reader can quickly cross-check
the plotted values against the corresponding analytic formula.
