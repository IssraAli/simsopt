# PSCBulkArray: R / t VJP audit

This note documents which kernels in `simsopt/field/psc_bulk.py` consume
the per-puck local geometry arrays (`local_pts_base`, `local_K_base`,
`local_n_base`, `local_w_base`, `local_phi_base`) and would therefore
need `(R_base, t_base)` plumbed through them if we were to make the
puck-radius (`R{i}`) and thickness (`t{i}`) DOFs differentiable through
the *primary* JAX VJP path (Phase D.2 / D.3 of the
`free_dof_yamls_and_bulk_distance_penalties` plan).

## Scope

The VJP we want to expose is

\[
  \nabla_{R, t}\,\langle v_B,\, B_{\text{induced}}(p)\rangle
\]

at fixed TF coil and base-puck centre / quaternion DOFs.  Today
`PSCBulkArray._is_zero_vjp_dof` returns `True` for any DOF name starting
with `R` or `t`, and `local_unfix_all` emits a `UserWarning` when those
DOFs are unfixed.

## Kernels that read `local_*_base`

The following JAX-jitted kernels consume one or more of the
`local_pts_base / local_K_base / local_n_base / local_w_base /
local_phi_base` arrays as part of their differentiable input tuple
(treating them as static / non-traced):

| File | Function (line approx.) | Reads |
|------|-------------------------|-------|
| `psc_bulk.py` | `_assemble_L_red_inner` (~1670) | `local_pts_base`, `local_K_base`, `local_n_base`, `local_w_base` |
| `psc_bulk.py` | `_assemble_f_red_inner` (~1809) | `local_pts_base`, `local_K_base`, `local_n_base`, `local_w_base` |
| `psc_bulk.py` | `_B_eval_reduced_free_dof_body[_v2]` (~2200, ~2515) | `local_pts_base`, `local_K_base`, `local_n_base`, `local_w_base` |
| `psc_bulk.py` | `_B_eval_reduced_free_dof_dispatch` | dispatches into the body kernels |
| `psc_bulk.py` | `_vjp_reduced_free_dof_run` (3-DOF flavours) | `local_pts_base`, `local_K_base`, `local_n_base`, `local_w_base`, `local_phi_base`, `Q_c_base` |
| `psc_bulk.py` | `_vjp_full_jitted` (legacy non-reduced path) | full per-replica `_jax_local_*` arrays |

`local_phi_base` carries the basis-evaluation table used by the
W1-envelope load vector and far-pair multipole moments; it depends on
`R` (Zernike radial scaling, `1/R` in the gradient) and on `t` only via
the side-wall Chebyshev table.

## How `local_*_base` is built today

`local_*_base` are filled in `PSCBulkArray._rebuild` (around line 4474)
from `build_puck_shell_basis(R, t, ...)` (`puck_basis.py:238`).  The
basis construction is **NumPy / Python** (with a few `jax.numpy` helpers
for analytic gradients of the Zernike basis itself); it is *not*
JAX-traceable as a function of `(R, t)`.

Specifically `build_puck_shell_basis` does:

1. Constructs disk top / bottom / side quadrature grids whose
   *coordinates* and *weights* depend linearly on `R` and `t`.
2. Builds a disk Zernike-Fourier table whose values depend on `rho/R`
   and a side-wall Chebyshev table whose values depend on `z / (t/2)`.
3. Computes `K = n × grad_s phi` from the analytic gradients
   (`_disk_grad_g_cartesian`, `_side_grad_K`); these scale as `1/R`,
   `1/t` for the radial/axial parts.
4. Caches the result in a per-process `OrderedDict` keyed by
   `(R, t, m_fourier, l_zernike, k_chebyshev, n_rho, n_phi, n_z)`.

For analytic `(R, t)` VJPs through the JAX path we would need to
either:

* Re-implement the entire `build_puck_shell_basis` in `jax.numpy` and
  trace through Zernike radial polynomial evaluation, derivative
  tables, etc., **or**
* Exploit the explicit scaling structure (see *Scaling* below) and only
  trace the scale-and-renormalise step in JAX, leaving the unit-puck
  basis precomputed.

Either path is a multi-day refactor and risks recompiling every JIT
cache key in the existing far-pair / non-reduced paths.

## Scaling structure

For a thick disk of radius `R` and thickness `t` in its local frame:

* `quad_points_local[:, :2] ∝ R`, `quad_points_local[:, 2] ∝ t`
* `quad_normals_local` is independent of `R, t` (always axis-aligned).
* `quad_weights_top, weights_bot ∝ R^2`, `weights_side ∝ R * t`.
* `phi_values` (basis at quad points) depends on `(rho/R, phi)` for the
  disk faces (so independent of `R` once normalised) and on
  `(z/(t/2))` for the side wall (independent of `t`).
* `grad_phi_local` carries explicit `1/R` factors on disk faces and
  `2/t` on the side wall.
* `k_basis_local = n × grad_s phi` inherits the same `1/R, 2/t`
  scaling.

So the *fully JAX-traceable* construction is feasible if we pre-build
the basis at a "unit puck" `(R = 1, t = 1)` and carry `(R, t)` as
trace-time scaling factors.  This is the approach Phase D.2 in the
plan describes (`_build_local_geometry_jax(R, t, basis_consts)`).

## Choice taken in this commit

For the present implementation we take the **finite-difference**
approach instead of the full JAX trace refactor:

* `_is_zero_vjp_dof` is flipped to `False` for `R / t`; the
  unfix-warning is dropped.
* `PSCBulkArray._vjp_puck_geometry` is extended with a
  `_Rt_fd_gradient(v_B, pts)` helper that evaluates
  `<v_B, B_induced(pts)>` at `R{i} ± eps` and `t{i} ± eps` for every
  *base* puck whose R/t DOF is currently free, using the existing
  `recompute_currents` machinery (which already supports a
  `changed_mask=changed_pucks` partial-rebuild fast path).
* Cost: `2 * N_free_R_or_t` extra `recompute_currents` invocations per
  VJP call, each of which goes through the partial-rebuild path so
  only the perturbed puck's basis / inductance row is rebuilt.  At
  reactor scale (`n_base = 14`) and with both `R` and `t` free this is
  a `~56x` slowdown of the VJP relative to the centre-only / quat-only
  paths — acceptable for an exploratory free-radius run, with a
  scoped follow-up to migrate to the JAX-traceable refactor if it
  becomes the dominant cost.

This choice keeps the JAX cache topology unchanged (no new traced
arguments), avoids invalidating the far-pair partition tests, and
limits the surface area of the change.  It is a **temporary**
implementation — Phase D.2/D.3 (full JAX trace through
`(R_base, t_base)`) remain follow-up work.
