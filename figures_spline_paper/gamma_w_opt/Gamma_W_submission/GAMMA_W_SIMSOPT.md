# Γ_W without firm3d — the `simsopt.mhd.Boozer` backend

`gamma_w_simsopt.py` computes the Whitham energetic-particle proxy **Γ_W** using
only **simsopt + booz_xform + scipy + numpy** — no firm3d. It is a drop-in field
backend for the standalone engine in `Gamma_W_final.py`; all the physics (bounce
quadrature, Whitham drift ODE, branch continuation, aggregation) is imported from
that engine **unchanged**. Only the Boozer-field source and the marker sampler are
reimplemented firm3d-free.

## Why

firm3d is hard to build on some clusters (Perlmutter / Docker) and **cannot share a
Python process with simsopt** (a pybind11/C++ heap clash → `malloc(): invalid size`).
An optimizer driver necessarily imports simsopt, so a firm3d-based Γ_W had to run
out-of-process. This backend removes firm3d entirely, so **Γ_W can be called inline
in a simsopt optimization objective.**

## Install / requirements

```
simsopt (with simsopt.mhd.Boozer, which wraps booz_xform), booz_xform, scipy, numpy
```
No firm3d. Quick check:
```bash
python -c "import booz_xform; from simsopt.mhd import Boozer; print('ok')"
```
Keep `gamma_w_simsopt.py` next to `Gamma_W_final.py` (it imports the engine from
there; it also falls back to `Gamma_W_final_fix` if that is the engine name present).

## Usage

```python
from simsopt.mhd import Vmec, Boozer
from gamma_w_simsopt import gamma_w_from_boozer

v = Vmec("input.QA_nfp2"); v.run()
b = Boozer(v, mpol=16, ntor=16); b.register(v.s_half_grid); b.run()

agg = gamma_w_from_boozer(b, N=128, t_star=0.2, s0=0.3, seed=7)   # fast grid backend
print(agg["Gamma_W"], agg["Gamma_W_low"], agg["Gamma_W_high"], agg["regime"])
```

`source` (first argument) may be any of:
- a **`simsopt.mhd.Boozer`** that has been `.run()` (uses its `.bx`),
- a **`booz_xform.Booz_xform`** object,
- a **boozmn `.nc` path**,
- a **`simsopt.mhd.Vmec`** (a Boozer is built and run for you; pass `mpol`, `ntor`,
  `surfaces`).

### As an optimization objective (inline — no subprocess needed)

```python
from gamma_w_simsopt import gamma_w_from_boozer

def GammaW(vmec):
    vmec.run()
    return gamma_w_from_boozer(vmec, N=128, t_star=0.2, s0=0.3, seed=7,
                               mpol=16, ntor=16)["Gamma_W"]
```
(There is no firm3d here, so the process-isolation / `malloc` problem that broke the
earlier firm3d version does not arise.)

## Two field backends

| `field_backend=` | how modB is evaluated | use for |
|---|---|---|
| `"grid"` (default) | precompute \|B\| on a periodic (s,θ,ζ) grid once, then cubic `scipy.ndimage.map_coordinates` (firm3d's approach, in numpy) | production / optimization |
| `"fourier"` | direct sum `Σ bmnc_b(s) cos(mθ−nζ)`, exact in the angles | reference / validation |

Both share the profiles (G, I, ι via cubic splines) and psi0. The grid backend is
**~22× faster per modB call** than the exact Fourier one and reproduces it to ~1e-8
(median). Grid resolution is chosen from the mode content; override with
`grid_res=(ns, nθ, nζ)`.

## Field conventions (verified against firm3d's `BoozerSplineField`)

```
psi0    = -bx.phi[-1] / (2π)               (VMEC sign convention)
iota(s) =  CubicSpline(bx.s_b, bx.iota)
G(s)    =  CubicSpline(bx.s_b, bx.Boozer_G_all)   (= bvco_b)
I(s)    =  CubicSpline(bx.s_b, bx.Boozer_I_all)   (= buco_b)
|B|     =  Σ bmnc_b_mn(s) cos(xm_b·θ − xn_b·ζ)    [+ bmns sin if asym]
```

## Validation (see `validate_simsopt_backend.py`)

Against firm3d on Alex QH boozmn files:

- **Field**: `psi0` **exact**, G/ι to ~1e-9, **modB** median ~3e-7 / max ~2e-5
  (i.e. as close to firm3d as firm3d's own grid spline is to the exact Fourier sum).
  Every quantity that carries a 2, a 2π, or a sign matches — the factors of 2 are
  correct.
- **End-to-end, decisive test** (identical markers + identical engine/settings, only
  the field swapped firm3d ↔ grid):

  | config | firm3d Γ_W | grid Γ_W | agreement |
  |---|---|---|---|
  | `0.01_00` | 0.004090 | 0.004100 | 0.2% |
  | `0.09_17` | 0.040439 | 0.039958 | 1.2% |

  The residual is the 2e-5 modB difference propagating through the drift — well below
  marker-sampling noise. Run:
  ```bash
  python validate_simsopt_backend.py outputs/gamma_w/alex_boozmn/boozmn_0.09_17.nc
  ```

## Performance & the one caveat

- Grid modB build: ~0.3 s. modB throughput: **0.53 ms/cover (grid) vs 11.8 ms (Fourier)**.
- End-to-end Γ_W is currently **~75–110 s per config** (N=128). This is *not* the modB
  cost and *not* a step-cap crawl (verified: lowering `max_accepted_steps` 1200→300 gave
  no speedup) — it's the inherent number of drift steps × bounce covers. firm3d's C++
  is ~1.5× faster here for the same reason (it's the step count, not the field).
- Levers if you need it faster (not yet applied — they trade metric fidelity or need an
  engine rewrite): coarser bounce covers / drift stepping (`n_periods`, `pts_per_period`,
  `ds_step_max`) ≈ 3–5×; or batch modB across markers (metric-identical, larger change)
  ≈ 5–15×. `N=64` halves the time but adds ~30% sampling noise — not recommended for
  gradients.
- In an MPI optimizer the (1+ndofs) evaluations run in parallel, so Γ_W adds ~one
  eval-time (~100 s) to the per-iteration wall-clock on top of VMEC + booz_xform.

## Files

- `gamma_w_simsopt.py` — the firm3d-free backend (this).
- `Gamma_W_final.py` — the shared physics engine (imported unchanged).
- `validate_simsopt_backend.py` — reproducible field + end-to-end validation.
