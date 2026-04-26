# Optional CSV / JSON from `passive_bulks_bottleneck_timing.py`

Write harness output with e.g. `--out timing_outputs/profiles_$(hostname).csv`.
This directory is a stable place to drop reports (not committed with real numbers by default).

---

## Dipole examples: post-fix verification and next bottlenecks

**Environment:** `stellcoilbench_py312`, `CI` **unset** (so `in_github_actions` is false and `MAXITER=100` in both scripts).  
**Fix in tree:** `Optimizable._any_free_in_lineage` + `MagneticField.recompute_bell` (no `dofs_free_status` in the cache-bell hot path).

### End-to-end runs (correctness / wall time)

| Script | Wall time (real) | Note |
|--------|------------------|------|
| [`../dipole_array_tutorial.py`](../dipole_array_tutorial.py) | **~36 s** | `time -p` on full run; L-BFGS uses `maxiter=100` |
| [`../dipole_array_tutorial_advanced.py`](../dipole_array_tutorial_advanced.py) (`passive_coil_array=False`) | **~106–118 s** | `time -p` and in-script `Total time` agree within ~10 s (JAX/IO) |

**Tutorial (full):** completes without NaNs; final objective lines in [`dipole/tutorial_full_FIXED.log`](dipole/tutorial_full_FIXED.log) show `J≈1.1`, `⟨B·n⟩/⟨B⟩≈7e-3`, `║∇J║` down to O(1) at the end of the run. VTK / coil save paths execute successfully.

**Advanced (full):** toroidally averaged `Bmag` at `R=4.894…`, `Z=0` ≈ **5.786 T** (see log [`dipole/advanced_full_FIXED.log`](dipole/advanced_full_FIXED.log), matches profiled run to printed precision).

**Logs:** `dipole/tutorial_full_FIXED.log`, `dipole/advanced_full_FIXED.log`.

### cProfile outputs (post-fix, `MAXITER=100`)

- [`dipole/tutorial_full_FIXED.prof`](dipole/tutorial_full_FIXED.prof) — in-process cProfile of the tutorial (~40 s CPU in profiler harness).
- [`dipole/advanced_full_FIXED.prof`](dipole/advanced_full_FIXED.prof) — in-process cProfile of the advanced example (~107 s CPU).

Load with: `python -m pstats dipole/<name>.prof` or snakeviz.

**Tutorial — dominant residual costs (illustrative, one machine / one run):**

- `MagneticField.clear_cached_properties` / C++ `invalidate_cache` — very high call count, largest **self** time.
- `Optimizable.set_recompute_flag` — visits each `id(Optimizable)` at most once per `Optimizable.x` / `full_x` assignment via the W1 `ContextVar`-shared visited set. Self time is now dominated by the single recursive walk and the `recompute_bell`-driven version compare in `MagneticField` (W2).
- `Optimizable.__hash__` — millions of calls (set / dict use during graph walks).
- `recompute_bell` — O(1) boolean branch; time is mostly the **downstream** `clear_cached_properties` when any DOF in lineage is free.
- `dofs_free_status` — no longer a top hotspot (orders of magnitude smaller than pre-fix); still used elsewhere (e.g. derivatives).

**Advanced — additional dominant costs:**

- `numpy.linalg.lstsq` — ~13 s for **5 calls** (setup / `polyfit`-style fits).
- `simsoptpp.get_pointclouds_closer_than_threshold_within_collection` — high **self** time, **O(100+)** calls over the run (geometry / proximity).
- JAX: `apply_primitive`, compilation (`backend_compile_and_load`), and plumbing — dominate cumulative time after the C++ / NumPy setup costs.

`set_recompute_flag` is a smaller fraction of the advanced run than the tutorial, consistent with a different objective wiring and fewer magnetic-field graph invalidation passes per optimizer step.

### Prioritized “dramatic speedup” follow-ups (A–E)

**A. Visited set in `set_recompute_flag` (implemented)**

- **Where:** [`src/simsopt/_core/optimizable.py`](../../../src/simsopt/_core/optimizable.py) around `set_recompute_flag` (e.g. line ~1137).
- **Idea:** Optional `_visited: set[Optimizable]` (or `id`-based set) on the outer call so DAG-shared children are not processed multiple times per `JF.x = …` update.
- **Why:** Cuts duplicate `recompute_bell` / `clear_cached_properties` waves on `MagneticFieldSum` / `MagneticFieldMultiply` graphs.
- **Risk:** Low if keyed by object identity; must not break intentional multi-parent semantics.

**B. Versioned DOF state — skip redundant `clear_cached_properties` (implemented)**

- **Idea:** Monotonic `dof_state_version` on `Dofs`, bumped when global `x` actually changes; each `MagneticField` records last seen version; `recompute_bell` no-ops the C++ invalidate if version unchanged.
- **Why:** Line searches and adjoint/forward re-evaluation can revisit the same `x` many times; avoids repeated native cache flushes.

**C. Batch `lstsq` in `SurfaceRZFourier.extend_via_normal` (non-stellarator-symmetric surfaces)**

- **Where:** [`src/simsopt/geo/surfacerzfourier.py`](../../../src/simsopt/geo/surfacerzfourier.py) — when R and Z share the same Fourier basis, a single `lstsq(B, column_stack(R,Z))` replaces two independent solves. Stellarator-symmetric fits still use two bases (cos/sin split) and two solves, with a shared `_lstsq_robust` helper for the CI retry pattern.

**D. Persistent JAX compilation cache**

- **Implemented:** [`src/simsopt/geo/__init__.py`](../../../src/simsopt/geo/__init__.py) — `jax_compilation_cache_dir` defaults to `~/.cache/simsopt/jax`; override with env **`SIMSOPT_JAX_CACHE`**. Amortizes **across** process restarts.

**E. Curve proximity and `get_pointclouds_*`**

- **Implemented (partial):** [`src/simsopt/geo/curveobjectives.py`](../../../src/simsopt/geo/curveobjectives.py) — `CurveCurveDistance` / `CurveSurfaceDistance` only clear `candidates` when a relevant `Dofs._state_version` changes. The all-pairs `cdist` **fallback** in `CurveCurveDistance.shortest_distance` uses `scipy.spatial.cKDTree` min-distance queries instead of dense `cdist`.

### Follow-up speedups (implemented in code, 2026-04-26)

- **W1** — [`set_recompute_flag`](../../../src/simsopt/_core/optimizable.py): `id(self)` visited-set dedup; plus [`ContextVar`](../../../src/simsopt/_core/optimizable.py) batch so one shared visited set is used for all `local_x` writes in a single `Optimizable.x` / `full_x` assignment.
- **W2** — [`Dofs._state_version`](../../../src/simsopt/_core/optimizable.py) (bump only on real value change); [`MagneticField.recompute_bell`](../../../src/simsopt/field/magneticfield.py) skips `invalidate_cache` when the per-`Dofs` version map is unchanged.
- **W4** — [`Derivative.__iadd__`](../../../src/simsopt/_core/derivative.py) (pre-existing) + [`OptimizableSum.dJ`](../../../src/simsopt/_core/optimizable.py) / [`MPIObjective.dJ`](../../../src/simsopt/objectives/utilities.py) accumulate in-place.
- **W5b** — Persistent JAX cache (see D above).
- **W6** — [`B2Energy`](../../../src/simsopt/field/force.py) / [`SquaredMeanForce._J_args`](../../../src/simsopt/field/force.py): `np.stack` + single `jnp.asarray` for coil geometry blocks (`_stack_then_device`).
- **W7** — [`LinkingNumber.J`](../../../src/simsopt/geo/curveobjectives.py) cached while `(id(Dofs), _state_version)` per curve is unchanged.

### DOF-graph speedup plan, PR1+PR2 (2026-04-26)

PR1 (always-on, low-risk hygiene): closes a W2 correctness gap by clearing `MagneticField._mf_last_seen_dof_versions` whenever `_has_any_free_in_lineage` is invalidated (free-status change or lineage reshape can shift the unique-`Dofs` walk used as the version-map key); documents the `Dofs._state_version` invariant; and refreshes the README. The bell short-circuit originally proposed in the plan was implemented and reverted: the existing W1 `ContextVar`-shared visited set already guarantees `recompute_bell` is called at most once per `Optimizable` per batch, and the `new_x` flag is sticky across batches — so a second-batch short-circuit would skip a needed `clear_cached_properties` (caught by `tests/field/test_passive_bulks.py::test_scaled_current_set_dofs_invalidates_biotsavart_cache`).

PR2 (opt-in via `SIMSOPT_DEFER_RECOMPUTE=1`) collects every `Dofs._flag_recompute_opt` notification during a joint `Optimizable.x` / `full_x` write into a `ContextVar` queue, then runs one `set_recompute_flag` walk per unique dependent root with a shared visited set (collapses `N` walks into 1). `local_dof_setter` still fires immediately so any C++-mirrored DOF state stays in lockstep.

Quick A/B on `dipole_array_tutorial.py` (`CI=1`, MAXITER=10), cProfile dumps committed at [`tutorial_pr2_off_ci.prof`](dipole/tutorial_pr2_off_ci.prof) / [`tutorial_pr2_on_ci.prof`](dipole/tutorial_pr2_on_ci.prof):

| Metric | `SIMSOPT_DEFER_RECOMPUTE` OFF | ON | Δ |
| --- | --- | --- | --- |
| Total wall time | 11.98 s | 9.20 s | -23% |
| `set_recompute_flag` ncalls | 229 139 | 205 673 | -10% |
| `set_recompute_flag` tottime | 0.330 s | 0.282 s | -15% |
| `MagneticField.recompute_bell` tottime | 0.128 s | 0.101 s | -21% |
| Final `J` / `Jf` | 9.88e+02 / 9.87e+02 | 9.88e+02 / 9.87e+02 | identical |

The new equivalence test [`tests/core/test_optimizable.py::TestSimsoptDeferRecompute`](../../../tests/core/test_optimizable.py) asserts identical `f`, `dJ`, and per-`Dofs` `_state_version` sequences between the two branches across multiple joint `x` writes and confirms `local_dof_setter` runs immediately even with the queue open.

### DOF-graph speedup plan, PR3 (2026-04-26)

[`PSCBulkArray.recompute_currents`](../../../src/simsopt/field/psc_bulk.py) used to copy `self.local_full_x` and call `hash(tuple(...))` on every invocation to decide whether anything changed. That probe is `O(local_full_dof_size)` and includes one Python tuple build + numpy-to-Python conversion per DOF, even on the no-op path. PR3 replaces that probe with an `O(N_unique_dof_opts)` comparison of `(id(opt._dofs), opt._dofs._state_version)` tuples — `_state_version` is bumped only on real value changes, so this is exact for in-process state. The persistent on-disk cache keys (`_geom_hash`, `_psc_lcache_digest`) are kept because `_state_version` resets every process and cannot key a cross-process cache.

Quick microbench (4 TF coils, 30-puck array, 270 local DOFs, 13 unique `Optimizable`s):

| Probe | µs/call | Speedup |
| --- | --- | --- |
| OLD `hash(tuple(self.local_full_x))` | 6.79 | 1.00x |
| NEW `(id(_dofs), _state_version)` tuple | 1.79 | **3.79x** |

Verified with `pytest tests/field/test_passive_bulks.py -k "not test_implicit_solve_vjp_and_cached_pullback_match_default" tests/field/test_psc_multipole.py` (110 + 18 passed).

---

*Generated from full runs and cProfile collection on 2026-04-26; absolute seconds vary by hardware and load.*
