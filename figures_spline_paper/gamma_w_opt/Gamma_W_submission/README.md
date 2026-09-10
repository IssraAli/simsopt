# Γ_W — submission package (2026-08-19)

> **STATUS: READY FOR SUBMISSION AND FOR SHARING WITH CO-WORKERS.**
> Every quantitative claim in this package is backed by a frozen artifact that was
> re-verified on 2026-08-19. Claims that rested only on working notes have been removed
> or replaced with their artifact-backed equivalents. See **`SUBMISSION_NOTE.md`** for
> the verification record, the one change made relative to the 2026-07-25 merged
> release, and the scope statement to quote when citing Γ_W.

The complete, up-to-date standalone release of **Γ_W** (`Gamma_W`), the Whitham
energetic-particle (fusion-alpha) confinement proxy. This package merges the two
July-2026 releases under **one canonical engine**, so you get both the documented,
tested reference *and* a way to run Γ_W with **no firm3d**, without choosing between
packages:

- the **documentation + full test suite + figures** (from `Gamma_W_standalone`), and
- the **firm3d-free `simsopt` field backend** (`gamma_w_simsopt.py`, from `Gamma_W_simsopt`).

**Canonical engine:** `Gamma_W_final.py`, md5 `b58272e47b7352264209735bdef2aaa9`
(lineage `9de057ad → 5952f45f → 575187cf → b58272e4`: P1 status-propagation patch,
`n_cont_rescues` diagnostic, then the 2026-08-19 docstring correction below — that last step
changed no executable code, and the bytecode is identical to `575187cf`).

## Read first

- **`SUBMISSION_NOTE.md`** — what was verified, what changed, and how to cite the scope.
- **`Gamma_W_doc.pdf`** — the 11-page physicist reference (physics, math, numerics,
  diagnostics, validation history, scope). Start here for the science.
- **`GAMMA_W_SIMSOPT.md`** — the firm3d-free backend: why it exists and how to use it.

## Two field backends, one engine

The engine in `Gamma_W_final.py` (bounce quadrature, Whitham drift ODE, branch
continuation, aggregation) is **identical** for both paths. Only the **Boozer-field
source** and the marker sampler differ.

### 1. firm3d engine — default (needs firm3d)

```bash
source /opt/miniconda3/etc/profile.d/conda.sh && conda activate simsopt
python Gamma_W_final.py --selftest                         # numerical core -- NO firm3d needed
python Gamma_W_final.py --boozmn FILE.nc --N 128 --t-star 0.2 --s0 0.3
```

`--keep` recovers the published pre-`branch_continue` "keep" metric exactly. The field
here comes from firm3d's `InterpolatedBoozerField`.

### 2. firm3d-free `simsopt` backend (needs only simsopt + booz_xform)

```python
from gamma_w_simsopt import gamma_w_from_boozer
# source: a boozmn path, a simsopt.mhd.Boozer, a booz_xform object, or a simsopt Vmec
agg = gamma_w_from_boozer("FILE.nc", N=128, t_star=0.2, s0=0.3, seed=7)   # fast grid backend
print(agg["Gamma_W"], agg["Gamma_W_low"], agg["Gamma_W_high"], agg["regime"])
```

Because there is no firm3d here, Γ_W can be called **inline in a simsopt optimization
objective** (no process isolation, no `malloc` clash). This is the backend used for the
Landreman / B3 metric studies. `field_backend="grid"` (default, ~22x faster modB) or
`"fourier"` (exact, for validation).

> **WARNING: firm3d and simsopt cannot share one Python process** (a pybind11/C++ heap
> clash -> `malloc(): invalid size`). Use **one** backend per process. That is also why
> the two validation lanes below are run separately.

## Nemov Γ_c sibling

`gamma_c.py` computes the Nemov `Γ_c = (2/π)·arctan(|∂_αJ|/|∂_sJ|)` as a read-only
by-product of the same Γ_W drift (τ/κ cancels), so it is bit-consistent with Γ_W.

## Tests & validation (two separate lanes -- do NOT mix backends in one process)

```bash
# (a) the canonical engine + firm3d path -- pytest
pytest -m unit          # engine core, ~1 s, ~43 tests, NO firm3d
pytest                  # + regression tier (~25 s, uses firm3d); add -m slow for end-to-end (~2.5 min)

# (b) the firm3d-free simsopt backend -- its own script (grid-vs-Fourier; + firm3d cross-check if present)
python validate_simsopt_backend.py tests/test_files/boozmn_LandremanPaul2021_QA_lowres.nc
```

The simsopt-backend validator is a standalone script (not a pytest test) precisely so it
never shares a process with the firm3d-using regression tests.

## Contents

```
Gamma_W_final.py            canonical engine (b58272e4) -- CLI + importable
gamma_w_simsopt.py          firm3d-FREE Boozer-field backend (simsopt + booz_xform)
gamma_c.py                  Nemov Gamma_c sibling
validate_simsopt_backend.py grid-vs-Fourier validator (+ firm3d cross-check if importable)
Gamma_W_doc.{md,pdf,tex}    11-page reference
GAMMA_W_SIMSOPT.md          firm3d-free backend notes
pyproject.toml              pytest markers: unit / regression / slow
tests/                      pytest suite + tests/test_files/ QA boozmn fixture
figures/                    engine-drawn schematics (anatomy_of_a_well, drift_and_detrapping, qh_ripple_diagnostic)
scripts/                    make_doc_figures.py, create_scan.py (ripple-scan generator, boundary-parse fixes applied)
```

## Provenance

Merged 2026-07-25 from `Gamma_W_standalone.zip` (2026-07-14) and `Gamma_W_simsopt.zip`
(2026-07-14); both shipped engine md5 `575187cf`. Engine unchanged by the merge.

Re-cut 2026-08-19 as the **submission package**. Two corrections, both replacing a claim that
rested only on working notes with its artifact-backed equivalent: one in `Gamma_W_doc`, and one
in the `gamma_w_for_boozmn` docstring of the engine itself. The engine edit re-mints the md5
(`575187cf → b58272e4`) but touches **no executable code** — the compiled bytecode of all 63
code objects is identical, and `tests/conftest.py` pins the new hash in the same change.
`gamma_w_simsopt.py`, `gamma_c.py`, `validate_simsopt_backend.py`, the test bodies and the
fixture are byte-identical to the 2026-07-25 merged release. See `SUBMISSION_NOTE.md`.
