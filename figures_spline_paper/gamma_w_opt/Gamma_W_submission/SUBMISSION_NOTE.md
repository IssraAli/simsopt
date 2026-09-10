# Γ_W submission package — status note

**Date:** 2026-08-19
**Status:** **READY FOR SUBMISSION AND FOR SHARING WITH CO-WORKERS.**
**Canonical engine:** `Gamma_W_final.py`, md5 `b58272e47b7352264209735bdef2aaa9`
(lineage `9de057ad → 5952f45f → 575187cf → b58272e4`)

This package is the complete, self-contained release of **Γ_W**, the Whitham radial-reach
diagnostic for energetic particles. It supersedes `Gamma_W_merged.zip` (2026-07-25), which in
turn superseded the separate `Gamma_W_standalone` and `Gamma_W_simsopt` releases. Choosing
between earlier packages is obsolete — use this one.

---

## 1. What was verified before this package was cut

Every item below was re-opened and re-checked on 2026-08-19, not carried over from notes.

| check | result |
|---|---|
| Engine identity | md5 `b58272e4` — the **repaired** canonical engine, plus the 2026-08-19 docstring correction in §2 |
| Numerical core | `--selftest` **ALL PASS**: bounce quadrature vs `scipy.quad`, rel. err ≤ 4.2e-5 on 4 model wells |
| Test suite | **43 unit tests pass** (~0.8 s) from a clean extract, no firm3d needed |
| Package integrity | all `MANIFEST.sha256` entries verify from a fresh unzip |
| Landreman gate | 0.7012 / 0.3935 / 0.0757 — reproduced from `role_resolved_gammaW.json` |
| Alex-250 headline | Γ_W +0.656 vs QS +0.736 — reproduced from the results tables |
| Γ_c same-pipeline | +0.446 / +0.451, collinearity +0.398 — reproduced |
| keep vs `branch_continue` | rank agreement 0.9999–1.0000 on all three legacy datasets |
| Rippled tokamak | Spearman **recomputed from the frozen CSVs**: n=2 keep +0.048 → fate +0.934; n=3 +0.577 → +0.955 |

**Why the engine is the *repaired* one matters.** The older `9de057ad` engine — still present in
some July zips — could score a numerical finite-difference failure as `DETRAP_PASSING`, i.e. as a
physically *confined* marker, placing a numerical failure outside the sensitivity band. The engine
here has the P1 status-propagation fix: any non-`ok` status stops the marker, preserves its
verified `s_max`, and is flagged **unresolved** with its originating status. Only a genuinely empty
descendant well yields `DETRAP_PASSING`.

## 2. What changed relative to `Gamma_W_merged.zip` (2026-07-25)

**Two corrections, both of the same kind:** a claim that rested only on working notes, replaced by
its artifact-backed equivalent. One was in the documentation, one was in the engine's own docstring.

**The claim.** Both places stated that branch continuation removes the rippled-tokamak de-trapping
dip with *"Spearman-vs-loss 0.69 → 0.905 at n=2"*. Those two numbers come from a single exploratory
run recorded in working notes; **no frozen artifact carries them** — the frozen
`gamma_w_n{2,3}_fate.csv` files hold the `keep` and δ_QS-fate columns only, with no branch-continued
column. They are removed, and replaced by the claims that *are* artifact-backed:

- the **de-trapping fate** result on the rippled tokamak, recomputed from the frozen CSVs:
  Spearman-vs-loss **0.05 → 0.93** at n=2 and **0.58 → 0.96** at n=3 (η=0.02, robust over
  η ∈ [0.01, 0.03]); the base axisymmetric case stays exactly 0, as it must by symmetry;
- the **keep vs `branch_continue`** comparison, which *is* frozen: active in proportion to
  de-trapping (2645 continuation hops on the mixed-symmetry Landreman cfg4) yet **rank-preserving
  everywhere**, 0.9999–1.0000 on Alex-250, Paul and Landreman.

The qualitative statement — that a branch-aware treatment is what removes the spurious dip — is
retained and is supported by Figure 3. Only the unfrozen *numbers* are gone.

**The engine edit re-mints the md5, and that is deliberate.** `575187cf → b58272e4`. This follows the
convention already used twice in this file's own PATCH LOG (`9de057ad → 5952f45f → 575187cf`):
patch in place, record it in the log, and move the `conftest.py` pin in the same change.

**It changes no executable code, and this was verified, not asserted:**

| check | result |
|---|---|
| textual diff | two hunks, both inside string literals (the module PATCH LOG and one function docstring) |
| AST with docstrings stripped | **identical** |
| compiled bytecode, all 63 code objects | **identical** (`co_code`, `co_names`, `co_varnames`, `co_argcount`) |
| `--selftest` | ALL PASS |
| `pytest -m unit` | 43 passed, with `ENGINE_HASH` re-pinned to the new md5 |

So every score, band, status and diagnostic this engine produces is bit-for-bit what `575187cf`
produced. `575187cf` remains the engine of record for every published number; this package ships
its documentation-corrected successor.

**Everything else is byte-identical** to the 2026-07-25 merged release: `gamma_w_simsopt.py`,
`gamma_c.py`, `validate_simsopt_backend.py`, `pyproject.toml`, all seven test bodies, the QA boozmn
fixture, both scripts and all three figures.

## 3. Scope statement — please quote this, not a stronger one

> Γ_W's leading-order adiabatic reach captures the loss **mechanism** (Landreman gate) and the
> **amplitude scaling** (Alex σ-bin-median 0.985), but it is **not** a drop-in quantitative
> replacement for the quasisymmetry error: on the full Alex-250 it trails QS (0.656 vs 0.736,
> paired bootstrap Δ = −0.080, 95% CI [−0.142, −0.018]), and on Paul it under-predicts
> trapped-channel QA banana loss. The missing ingredient is the finite-orbit-width / non-adiabatic
> drift that a field-only proxy omits by construction. **Γ_W is best used as a mechanism /
> amplitude / ripple diagnostic, not a predictive surrogate for QS.**

Three further limits worth stating explicitly to anyone picking this up:

1. **Γ_W is not differentiable** (hard positive part, hard well mask, a depth threshold, a
   finite-time maximum, stopping events). It can be called inside a *derivative-free* optimizer —
   that is what the firm3d-free backend is for — but it is not a gradient objective.
2. **Split/merge is handled deterministically**, by a maximum-overlap descendant rule plus a
   `[low, high]` bracket. It is *not* a separatrix-crossing capture probability. This is inert on
   every configuration quoted here, and becomes first-order only when de-trapping dominates.
3. **Read the regime flag.** `adiabatic-resolved` means the point estimate is trustworthy;
   `de-trapping-dominated` means report the band; `passing-dominated` means a trapped-only reach
   metric is blind to most of the source.

## 4. Running it

```bash
source /opt/miniconda3/etc/profile.d/conda.sh && conda activate simsopt
python Gamma_W_final.py --selftest                      # no firm3d needed
pytest -m unit                                          # 43 tests, ~1 s
python Gamma_W_final.py --boozmn FILE.nc --N 128 --t-star 0.2 --s0 0.3
```

Three things a new user will otherwise hit:

- **Never import firm3d and simsopt in the same process** — a pybind11/C++ heap clash that surfaces
  as `malloc(): invalid size`. One backend per process; that is why the two validation lanes are
  run separately.
- The boozmn file must be built with **`flux=True`**.
- `branch_continue=True` is the default at the entry point; **`--keep`** recovers the published
  pre-continuation metric bit-for-bit.

## 5. Reproducing the package hashes

```bash
shasum -a 256 -c Gamma_W_submission.zip.sha256   # the release
unzip -q Gamma_W_submission.zip && cd Gamma_W_submission
shasum -c MANIFEST.sha256                        # every file
```

The manifest is generated after removing `__pycache__` / `.pytest_cache` / `.DS_Store`, so a fresh
extract verifies clean. Run the tests *after* verifying, since pytest recreates cache directories.
