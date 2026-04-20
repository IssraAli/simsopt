# Build system for the passive-bulks technical note.
#
# Figures are regenerated from standalone Python scripts and the LaTeX
# document is compiled with latexmk.  Intended to be run inside the
# ``stellcoilbench_py312`` conda environment.
#
# Usage::
#
#     make figures    # regenerate all figures/*.pdf from scripts/fig_*.py
#     make pdf        # compile passive_bulks_note.pdf (assumes figures exist)
#     make all        # figures + pdf
#     make clean      # remove LaTeX build artifacts
#     make distclean  # also remove figures/*.pdf
#
# Most individual figure scripts finish in a few seconds; the mode-resolution
# convergence (``fig_mode_convergence.py``) and optimization-convergence
# (``fig_optimization_convergence.py``) scripts are the slowest
# (~tens of seconds each) because they solve the full puck system multiple
# times.

PYTHON ?= python

SCRIPTS := scripts/fig_puck_schematic.py \
           scripts/fig_mode_convergence.py \
           scripts/fig_disc_quad_convergence.py \
           scripts/fig_cyl_quad_convergence.py \
           scripts/fig_smythe_thin_disc.py \
           scripts/fig_lorenz_short_solenoid.py \
           scripts/fig_neumann_coaxial.py \
           scripts/fig_sphere_meissner.py \
           scripts/fig_interior_cancellation.py \
           scripts/fig_rim_continuity.py \
           scripts/fig_L_spectrum.py \
           scripts/fig_taylor_test.py \
           scripts/fig_optimization_convergence.py \
           scripts/fig_pucks_rendered.py

FIGURES := $(patsubst scripts/fig_%.py,figures/%.pdf,$(SCRIPTS))

.PHONY: all figures pdf clean distclean

all: figures pdf

figures: $(FIGURES)

figures/%.pdf: scripts/fig_%.py scripts/_common.py
	$(PYTHON) $<

pdf: passive_bulks_note.pdf

passive_bulks_note.pdf: passive_bulks_note.tex refs.bib $(FIGURES)
	latexmk -pdf -interaction=nonstopmode passive_bulks_note.tex

clean:
	latexmk -C passive_bulks_note.tex 2>/dev/null || true
	rm -f passive_bulks_note.aux passive_bulks_note.log passive_bulks_note.out \
	      passive_bulks_note.bbl passive_bulks_note.blg passive_bulks_note.fls \
	      passive_bulks_note.fdb_latexmk passive_bulks_note.toc

distclean: clean
	rm -f figures/*.pdf figures/*.png
