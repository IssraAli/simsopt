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

PR2 (on by default since DOF-graph round 4; set `SIMSOPT_DEFER_RECOMPUTE=0` to opt out) collects every `Dofs._flag_recompute_opt` notification during a joint `Optimizable.x` / `full_x` write into a `ContextVar` queue, then runs one `set_recompute_flag` walk per unique dependent root with a shared visited set (collapses `N` walks into 1). `local_dof_setter` still fires immediately so any C++-mirrored DOF state stays in lockstep.

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

### Before/after on the four non-bulk dipole / passive-coils examples (CI=1)

Measured with `CI=1`, `MAXITER=10`, and `/usr/bin/time -p conda run -n stellcoilbench_py312 python ...`. The "before" files are the `536c5310` DOF-graph/PSC files, with the nested-`ScaledCurrent.set_dofs` passive-coils correctness fix applied on both sides so `passive_coils_QASH.py` and `passive_coils_CSX.py` can run. The "after" files are the current PR1+PR2+PR3 tree plus that same fix.

- `dipole_array_tutorial.py` — before 13.38 s; after 12.66 s; **1.06x faster**. J trajectory matched exactly over 62 printed objective lines; final toroidally averaged `Bmag` matched exactly.
- `dipole_array_tutorial_advanced.py` — before 46.37 s; after 45.72 s; **1.01x faster**. The first three printed objective lines matched, then the optimizer trajectory diverged and the after run landed at a lower final printed objective (`2.05e+02` vs `2.57e+02`, final relative delta `2.02e-01`); final toroidally averaged `Bmag` differed by `1.67e-02`. Treat this as a behavioral change to investigate, not a bit-equivalence pass.
- `passive_coils_QASH.py` — before 118.37 s; after 106.68 s; **1.11x faster**. J trajectory matched exactly over 72 printed objective lines; final toroidally averaged `Bmag` differed by `1.13e-10`.
- `passive_coils_CSX.py` — before 14.92 s; after 20.20 s; **0.74x** (after slower). J trajectory matched exactly over 48 printed objective lines; final toroidally averaged `Bmag` matched exactly.

### Phase C follow-up: hash-traffic reduction in lineage maintenance

Fresh `cProfile` runs on the post-PR1+PR2+PR3 tree pointed to hash traffic as the next DOF-graph lever:

- `dipole_array_tutorial.py` — `Optimizable.__hash__` was 0.406 s / 3.71M calls; the largest callers were `_update_full_dof_size_indices`, `update_free_dof_size_indices`, and `_get_ancestors`'s `dict.fromkeys`.
- `passive_coils_QASH.py` — `Optimizable.__hash__` was 0.292 s / 0.98M calls; the same lineage-maintenance functions dominated the DOF-graph subset.

The implemented change replaces hash-based lineage de-duplication in [`Optimizable._get_ancestors`](../../../src/simsopt/_core/optimizable.py), [`update_free_dof_size_indices`](../../../src/simsopt/_core/optimizable.py), and [`_update_full_dof_size_indices`](../../../src/simsopt/_core/optimizable.py) with `id(...)` sets. This is bit-equivalent for graph identity and avoids calling `Optimizable.__hash__` / `Dofs.__hash__` in those hot loops. A regression in [`tests/core/test_optimizable.py`](../../../tests/core/test_optimizable.py) checks a diamond dependency graph keeps a shared ancestor exactly once.

While measuring the passive-coils examples, `PSCArray` exposed a separate correctness bug in nested `ScaledCurrent.set_dofs`: fixed fake PSC currents can be represented as `ScaledCurrent(ScaledCurrent(Current(...)))`, so writing through `current_to_scale.local_full_x` can target a composite current with zero local DOFs. [`ScaledCurrent.set_dofs`](../../../src/simsopt/field/coil.py) now recurses through composite scaled currents until it reaches the scalar `Current` owner, preserving cache invalidation. A regression in [`tests/field/test_coil.py`](../../../tests/field/test_coil.py) covers that nested path.

Phase C CI-mode re-run against the post-PR1+PR2+PR3 after logs:

- `dipole_array_tutorial.py` — after 12.66 s; Phase C 9.92 s; **1.28x faster vs after**; J trajectory and final `Bmag` unchanged.
- `dipole_array_tutorial_advanced.py` — after 45.72 s; Phase C 40.09 s; **1.14x faster vs after**; optimizer trajectory remains run-sensitive (max printed-J relative delta `9.95e-01`, final `Bmag` delta `1.20e-02`) even comparing two current-tree runs.
- `passive_coils_QASH.py` — after 106.68 s; Phase C 111.23 s; **0.96x vs after**; J trajectory unchanged; final `Bmag` delta `1.69e-10`.
- `passive_coils_CSX.py` — after 20.20 s; Phase C 13.13 s; **1.54x faster vs after**; J trajectory and final `Bmag` unchanged.

Verification: `pytest tests/core/test_optimizable.py tests/core/test_dofs.py tests/field/test_recompute_bell_perf.py tests/field/test_coil.py -q` passed (120 passed, 3 skipped, 12 subtests), and `pytest tests/field/test_passive_bulks.py -k "not test_implicit_solve_vjp_and_cached_pullback_match_default" tests/field/test_psc_multipole.py -q` passed (133 passed, 9 skipped, 1 deselected).

### Round 4 (2026-04-26)

- **`SIMSOPT_DEFER_RECOMPUTE` default is now `1`** (set `=0` for the legacy eager walk).  Equivalence is still covered by `TestSimsoptDeferRecompute`.
- **Parent de-duplication** in the `funcs_in` construction path no longer uses `list(dict.fromkeys(depends_on))` (avoids `Optimizable.__hash__` on every parent insert).  `tests/core/test_optimizable.py::test_funcs_in_dedupes_parent_object_identity` locks the `[opt1, opt2]` behavior when `opt2` is registered twice.

---

*Generated from full runs and cProfile collection on 2026-04-26; absolute seconds vary by hardware and load.*
