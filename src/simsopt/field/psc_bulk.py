"""
Ideal-diamagnetic passive bulk (cylindrical pucks) and :class:`PassiveBulkField`.

**Environment (optional)**

- ``SIMSOPT_PSCBULK_TIMING``: if ``1``, :meth:`PSCBulkArray._rebuild` records
  per-phase wall times in :attr:`PSCBulkArray._timing_rows`.  Rows accumulated
  during a rebuild and the subsequent JAX forward / VJP calls are also
  appended to :attr:`PSCBulkArray._timing_history` (with a ``rebuild_idx``
  tag) before the next rebuild clears ``_timing_rows``, so the full
  per-phase history of an optimizer run can be retrieved via
  :meth:`PSCBulkArray.get_timing_history`.
- ``SIMSOPT_JAX_CACHE``: if ``0``, disables the default on-disk JAX
  compilation cache directory.
- ``SIMSOPT_JAX_CACHE_DIR``: directory for the JAX experimental compilation
  cache (empty string disables).
- ``SIMSOPT_JAX_PRIME``: if ``1``, after the first successful
  :class:`PSCBulkArray` rebuild the process one-shot executes hot ``jax.jit``
  entry points to prime XLA (failures are ignored).

- ``SIMSOPT_PSC_JAX_CHECKPOINT``: optional. When unset, the free-puck-DoF
  JAX VJP auto-enables :func:`jax.checkpoint` on per-pair ``L`` blocks
  when any puck DoF is unfixed, unless the constructor
  :class:`PSCBulkArray` ``checkpoint_l_pairs`` is set explicitly. See
  the constructor docstring.
- ``SIMSOPT_PSC_JAX_PAIR_CHUNK``: stream assembly of the symmetry-reduced
  free-DoF ``L`` in row batches (``0``/unset = default auto sizing or
  full ``vmap``; positive int = block width). See the constructor
  docstring for ``jax_pair_row_chunk``.
- ``SIMSOPT_PSC_PAIR_REPLICA_CHUNK``: in-row chunking of the symmetry
  replica axis ``n_all`` for the free-DoF pair-inductance kernel.  Default
  unset/``0`` keeps the existing ``vmap`` over all replicas (peak
  ``n_all * nq^2 * nd^2 * 8`` bytes per row).  Setting to a positive
  integer ``k`` runs the replica fan-in via :func:`jax.lax.scan` in
  groups of ``k`` so peak memory drops to ``k * nq^2 * nd^2 * 8`` bytes,
  enabling the production conformal-bulk free-orientation case to fit on
  smaller hosts.

- ``SIMSOPT_PSC_PAIR_FAR_KAPPA`` (``0`` = off): center-distance ratio
  above which per-pair blocks use the W2 dipole mutual-inductance kernel.
- ``SIMSOPT_PSC_TF_LOADING`` (``bn_quad`` default): ``bn_quad`` (surface
  B_n quadrature), ``a_quad`` (dipole / ``m·B`` at centroid), or
  ``a_taylor`` (``m·B + Q:∇B``).
- ``SIMSOPT_PSC_SOLVE_MODE`` (``eigenfloor`` or ``eigk``): reduced solve
  via Cholesky-eigenfloor or floored-eigenbasis + triangular solve.
- ``SIMSOPT_PSC_EIGK_TRIGGER``: relative Frobenius drift of ``L_red`` vs the
  cached anchor to refresh the host eigendecomposition in
  :meth:`PSCBulkArray.recompute_currents` (``0`` disables the drift check).
- ``SIMSOPT_PSC_W1_ENVELOPE`` (default ``1``): when truthy, the reduced
  free-DOF forward uses :func:`_B_eval_reduced_free_dof_body_v2`, which wraps
  the ``alpha`` solve in a ``jax.custom_vjp`` whose backward differentiates
  the scalar envelope adjoint
  ``E_adj(theta) = lam.T @ f_r(theta) - lam.T @ L_r(theta) @ alpha``
  with ``lam = L_r^{-1} ct_alpha``.  This avoids JAX retracing through the
  linear solve in the backward pass and gives the correct gradient through
  ``L_r`` even when the eigK forward branch reuses a host-cached
  :math:`(U,\\Lambda)` snapshot.  Set ``=0`` to fall back to the legacy v1
  body for diagnostics.

**Recommended fast profile** (Stage 3 of plan
``psc_bulk_speedups_and_mode_reduction``; see
``examples/3_Advanced/passive_bulks_bottleneck_timing.py``
``--profiles combo_fast_v2``): ``SIMSOPT_PSC_PAIR_FAR_KAPPA=3``,
``SIMSOPT_PSC_TF_LOADING=a_quad``, ``SIMSOPT_PSC_BS_EVAL_FAR_KAPPA=3``,
``SIMSOPT_PSC_FREE_SOLVE_VJP=implicit``.  ``SIMSOPT_PSC_W1_ENVELOPE`` is on
by default and supplies the implicit-function backward; the
``FREE_SOLVE_VJP=implicit`` knob is documented as "redundant under the
W1 envelope" but in practice is harmless and is left in ``combo_fast_v2``
for parity with the legacy ``combo_fast`` bundle.

The legacy ``combo_fast`` bundle additionally set
``SIMSOPT_PSC_SOLVE_MODE=eigk``; the leave-one-out sweep on ``medium/6``
(see ``examples/3_Advanced/timing_outputs/profiles_medium6_combo_fast_LOO.csv``)
showed that ``eigk`` adds a per-rebuild eigendecomposition that does *not*
amortise at small ``n_reduced_dof`` and is responsible for most of the
``combo_fast`` regression at this scale.  ``combo_fast_v2`` therefore
drops it.  See ``examples/3_Advanced/timing_outputs/PSC_FLAG_INVENTORY.md``
for the full per-flag sweep with accuracy deltas.

**Measured speedup** (Apple Silicon CPU, ``stellcoilbench_py312`` env, see
``examples/3_Advanced/timing_outputs/RESULTS.md`` for raw CSVs): on the
``basis=medium, n_base=6, n_eval=8, nfp=2, stellsym=True,
free=quaternions`` fixture, ``SIMSOPT_PSC_W1_ENVELOPE=1`` reduces the
``vjp_setup_B`` median from ``69.66s`` (``=0``) to ``59.51s`` (``=1``),
a ``1.17x`` speedup on `combo_fast`.  At the smaller
``basis=small, n_base=6, n_eval=2`` fixture the envelope adds ``~0.18s``
fixed tracing overhead (``1.59s`` -> ``1.77s``); the envelope's
mechanical benefit (one ``solve`` instead of ``eigh + solve`` in the
backward) scales with ``n_reduced_dof``, so the gain widens at larger
``n_base``.  The plan-mandated ``medium/64`` (``n_base=64``) target was
not validated within the available compute budget.
- ``SIMSOPT_PSC_BS_EVAL_FAR_KAPPA`` (``0`` = off): distance / local
  shell-radius ratio for the W7 far-eval dipole / shell Biot–Savart blend
  in the free-DoF JAX forward.

Puck geometry DOFs (center, quaternion orientation, radius, thickness per
base puck) are exposed through the :class:`~simsopt._core.optimizable.Optimizable`
framework.  All puck DOFs are **fixed by default**; unfix them with
``psc.unfix('center_x0')`` etc. to include them in the optimization.

Orientation uses the same **scalar-first quaternion** convention as
:class:`~simsopt.geo.curveplanarfourier.CurvePlanarFourier`:
``q = [q0, qi, qj, qk]`` with ``q0 = cos(theta/2)``.  The quaternion is
normalized before computing the rotation matrix, so the DOFs need not lie
on the unit sphere.

Induced currents (beta) are NOT optimizable -- they are recomputed from
``L_r^{-1} f_r`` each time :meth:`PSCBulkArray.recompute_currents` is called.
Gradients propagate through the solve via JAX VJPs.

The TF-only gradient path has two implementations: a fast analytic adjoint
that calls into the C++ :class:`~simsopt.field.biot_savart.BiotSavart`
(always used in production), and a reference pure-JAX
:func:`_vjp_tf_run`/:func:`_B_eval_from_tf_body` path used by equivalence
tests to cross-check the analytic adjoint.  Tests toggle between the two
by writing the module-level ``_USE_JAX_TF_VJP`` attribute; there is no
user-facing switch.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
import time
from contextlib import contextmanager
from functools import partial
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import jax
import jax.numpy as jnp
import numpy as np
from jax import vjp
import scipy.linalg as sp_linalg
from scipy.linalg import solve_triangular

from simsopt._core.derivative import Derivative
from simsopt._core.optimizable import Optimizable

from .bulk_inductance import (
    MU0_OVER_4PI,
    _SELF_REG_COEFF,
    expand_beta_reduced,
    null_space_projection_matrix,
    shell_biot_savart_stacked_pure,
    shell_cholesky_pure,
    shell_eigenfloor_cholesky_from_eig,
    shell_eigendecomposition_floored,
    shell_inductance_matrix_blockwise,
    shell_inductance_matrix_symmetric_reduced,
    shell_loading_vector_stacked_pure,
    shell_normal_field_basis_matrix,
    shell_solve_eigenfloor_pure,
    shell_solve_eigK_pure,
    shell_solve_prefactored_pure,
)
from .bulk_multipole import (
    magnetic_dipole_moments_stacked,
    magnetic_field_dipole_points,
    pair_inductance_multipole,
    quadrupole_magnetic_symmetric_stacked,
)
from .biotsavart import BiotSavart
from .force import (
    _B_at_point_from_coil_set_pure,
    _grad_B_at_point_from_coil_set_pure,
)
from .magneticfield import MagneticField
from .puck_basis import (
    PuckBasisData,
    apply_mode_truncate,
    build_continuity_constraint,
    build_puck_shell_basis,
    normalize_mode_truncate,
)
from .puck_init import curves_to_pucks, toroidal_shell_pucks, winding_surface_pucks
from . import _psc_bulk_dipole as _psc_bulk_dipole_mod

# Route TF-only VJP through the analytic adjoint + C++ BiotSavart (default
# production path).  Equivalence tests flip this module-level flag to
# ``True`` to exercise the slower pure-JAX reference path through
# :func:`_B_at_point_from_coil_set_pure`; see the two ``_USE_JAX_TF_VJP``
# call sites in :mod:`tests.field.test_passive_bulks`.  Deliberately kept
# as a plain module attribute (no env-var ingress) so end users cannot flip
# it and silently pay the ~3-5x runtime overhead.
_USE_JAX_TF_VJP = False


def _psc_bulk_timing_enabled() -> bool:
    """Return True when fine-grained :meth:`PSCBulkArray._rebuild` timing is on."""
    return os.environ.get("SIMSOPT_PSCBULK_TIMING", "0") == "1"


def _timing_block_until_ready(value: Any) -> None:
    """Synchronize JAX values before stopping a timing span."""
    if value is None:
        return
    block = getattr(value, "block_until_ready", None)
    if callable(block):
        block()
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            _timing_block_until_ready(item)
    elif isinstance(value, dict):
        for item in value.values():
            _timing_block_until_ready(item)


# Persistent XLA compilation cache (speeds cold starts; optional).
# Opt out with ``SIMSOPT_JAX_CACHE=0`` (also disables default cache dir).
_JAX_CACHE_DISABLED = os.environ.get("SIMSOPT_JAX_CACHE", "1") == "0"
_JAX_CACHE_DIR = os.environ.get(
    "SIMSOPT_JAX_CACHE_DIR",
    os.path.join(os.path.expanduser("~"), ".cache", "simsopt", "jax"),
)
if _JAX_CACHE_DISABLED:
    _JAX_CACHE_DIR = ""
elif _JAX_CACHE_DIR:
    try:
        from jax.experimental.compilation_cache import compilation_cache as _jax_cc

        os.makedirs(_JAX_CACHE_DIR, exist_ok=True)
        _jax_cc.initialize_cache(_JAX_CACHE_DIR)
    except Exception:
        pass

# ======================================================================
# On-disk cache for ``L_base`` + Cholesky factors (Stage 1 of the
# psc-scale-to-100-bulks plan).
#
# Key is a SHA-256 over the structural inputs that determine ``L_base``
# and its Cholesky factorization (quadrature geometry, K-basis stack,
# replica map, regularization parameters, solver mode, simsopt version).
# On cache hit ``_rebuild`` bypasses the expensive assembly + Cholesky
# phases entirely and proceeds to ``_setup_jax`` / ``_solve_beta``.
#
# Environment knobs:
# - ``SIMSOPT_PSC_LCACHE``   (default "0"): set to "1" to enable.
# - ``SIMSOPT_PSC_LCACHE_DIR`` (default "~/.cache/simsopt/psc_lbase"):
#   cache directory.  Ignored when ``SIMSOPT_PSC_LCACHE`` is off.
# - ``SIMSOPT_PSC_LCACHE_MAX_GIB`` (default "2.0"): LRU soft cap; when
#   total cache size exceeds this value oldest entries are dropped on
#   the next write.
# When any puck geometry DOF is *unfixed*, this cache is bypassed: the
# digest would change every step (orientation / position), so entries
# would never hit in typical optimization.
# ======================================================================


def _psc_jax_checkpoint_env() -> Optional[bool]:
    """Tri-state for :envvar:`SIMSOPT_PSC_JAX_CHECKPOINT` (JAX VJP pair tape).

    ``"1"``/``"true"``/``"yes"``/``"on"``  ->  force ``True`` (always checkpoint).
    ``"0"``/``"false"``/``"no"``/``"off"``  ->  force ``False`` (no checkpoint).
    Unset/empty  ->  ``None`` (defer to :class:`PSCBulkArray` heuristics).
    """
    raw = os.environ.get("SIMSOPT_PSC_JAX_CHECKPOINT")
    if raw is None or str(raw).strip() == "":
        return None
    s = str(raw).strip().lower()
    if s in ("1", "true", "yes", "on"):
        return True
    if s in ("0", "false", "no", "off"):
        return False
    return None


def _psc_jax_pair_chunk_from_env() -> Optional[int]:
    """Parse :envvar:`SIMSOPT_PSC_JAX_PAIR_CHUNK` (free-DOF ``L`` row batching).

    * Unset: ``None``  ->  auto (see :meth:`PSCBulkArray._resolved_jax_pair_row_chunk`).
    * ``"0"``: force the single-``vmap`` (monolithic) row assembly (no row chunking).
    * Positive: force that many base rows at a time inside ``lax.scan``.
    """
    raw = os.environ.get("SIMSOPT_PSC_JAX_PAIR_CHUNK")
    if raw is None or str(raw).strip() == "":
        return None
    try:
        v = int(str(raw).strip(), 10)
    except ValueError:
        return None
    return max(0, v)


def _psc_pair_replica_chunk_from_env() -> Optional[int]:
    """Parse :envvar:`SIMSOPT_PSC_PAIR_REPLICA_CHUNK` (n_all-axis pair-kernel chunking).

    Controls in-row chunking of the symmetry-replica axis ``n_all`` inside the
    free-DOF JAX pair-inductance assembly.  The default ``vmap`` over all
    replicas materializes a single ``(n_all, nq, nq, nd, nd)`` kernel batch,
    which can exceed available memory at production basis / replica sizes.
    Setting this env var to a positive integer ``k`` runs the replica fan-in
    via :func:`jax.lax.scan` in groups of ``k`` so peak memory inside
    :func:`_pair_inductance_block` scales as ``k * nq * nq * nd * nd * 8``.

    * Unset: ``None``  ->  off (current single-``vmap`` behaviour, no change).
    * ``"0"``: explicit off (same as unset).
    * Positive integer: replica chunk width.
    """
    raw = os.environ.get("SIMSOPT_PSC_PAIR_REPLICA_CHUNK")
    if raw is None or str(raw).strip() == "":
        return None
    try:
        v = int(str(raw).strip(), 10)
    except ValueError:
        return None
    return max(0, v)


def _psc_bs_eval_chunk_from_env() -> Optional[int]:
    """Parse :envvar:`SIMSOPT_PSC_BS_EVAL_CHUNK` (free-DOF shell-B batching).

    * Unset: ``None`` -> defer to :class:`PSCBulkArray` heuristics.
    * ``"0"``: force monolithic ``vmap`` over all evaluation points.
    * Positive: evaluate padded fixed-size chunks of this many points.
    """
    raw = os.environ.get("SIMSOPT_PSC_BS_EVAL_CHUNK")
    if raw is None or str(raw).strip() == "":
        return None
    try:
        v = int(str(raw).strip(), 10)
    except ValueError:
        return None
    return max(0, v)


def _psc_free_vjp_probe_env() -> str:
    """Return the free-DOF VJP attribution probe mode."""
    mode = os.environ.get("SIMSOPT_PSC_FREE_VJP_PROBE", "full").strip().lower()
    allowed = {
        "full",
        "stop_l",
        "stop_bn",
        "stop_solve",
        "shell_only",
        "beta_only",
    }
    return mode if mode in allowed else "full"


def _psc_free_solve_vjp_env() -> str:
    """Return the free-DOF solve VJP mode.

    ``envelope`` currently uses the same kernel as ``implicit`` (adjoint
    through the eigenfloor regularized system without an ``eig``-based VJP
    of the Cholesky path); a fully distinct envelope implementation can be
    layered here later.
    """
    mode = os.environ.get("SIMSOPT_PSC_FREE_SOLVE_VJP", "eigh").strip().lower()
    allowed = {"eigh", "implicit", "envelope"}
    m = mode if mode in allowed else "eigh"
    if m == "envelope":
        return "implicit"
    return m


def _psc_tf_loading_env() -> str:
    """TF loading in free-DOF JAX path: ``Bn_quad`` (default), ``A_quad``, ``A_taylor``."""
    mode = os.environ.get("SIMSOPT_PSC_TF_LOADING", "Bn_quad").strip().lower()
    allowed = {"bn_quad", "a_quad", "a_taylor"}
    m = mode.replace("-", "_")
    return m if m in allowed else "bn_quad"


def _psc_pair_far_kappa_env_override() -> Optional[float]:
    """Return explicit env override for far-pair kappa, or ``None`` if unset."""
    if "SIMSOPT_PSC_PAIR_FAR_KAPPA" not in os.environ:
        return None
    raw = os.environ.get("SIMSOPT_PSC_PAIR_FAR_KAPPA", "0").strip()
    try:
        v = float(raw)
    except ValueError:
        return 0.0
    return max(0.0, v)


def _psc_pair_far_kappa_env() -> float:
    """Distance / (sum of effective radii) above which dipole pair inductance is used."""
    override = _psc_pair_far_kappa_env_override()
    return 0.0 if override is None else float(override)


def _psc_partial_l_reuse_env() -> bool:
    """Enable changed-mask incremental rebuild path (default ``1``).

    When on, :meth:`PSCBulkArray._rebuild` may reuse the previous
    ``L_base`` and update only the rows/columns of base pucks whose
    geometry actually changed.  Set ``SIMSOPT_PSC_PARTIAL_L_REUSE=0``
    to force a full reassembly every step (legacy behavior).
    """
    return os.environ.get("SIMSOPT_PSC_PARTIAL_L_REUSE", "1").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _psc_partial_l_reuse_threshold_env() -> float:
    """Return changed-puck fraction threshold for partial-L reuse."""
    raw = os.environ.get("SIMSOPT_PSC_PARTIAL_L_REUSE_THRESHOLD", "0.3").strip()
    try:
        val = float(raw)
    except ValueError:
        return 0.3
    return float(min(1.0, max(0.0, val)))


def _psc_tf_solve_eigfloor_env() -> bool:
    """Use the eigenfloor-stabilised Cholesky for cheap TF-only forward + adjoint.

    When ``SIMSOPT_PSC_TF_SOLVE_EIGFLOOR`` is ``"1"`` / ``"true"`` / ``"yes"`` /
    ``"on"`` (default ``"1"``), :meth:`PSCBulkArray._B_at_points_tf_only` and
    :meth:`PSCBulkArray._vjp_tf_only_analytic` consume ``_jax_Lr_eigf_chol`` /
    ``_Lr_eigf_chol_host`` instead of the plain minimal-jitter
    ``_jax_Lr_chol`` / ``_Lr_chol_host``. This mirrors the factor already used by
    :meth:`PSCBulkArray._solve_beta` and removes asymmetry that caused shape-only
    NaNs on poorly conditioned geometries (Phase 4, May 2026).

    Set to ``"0"`` for A/B against the legacy plain-Cholesky cheap path.
    """
    raw = os.environ.get("SIMSOPT_PSC_TF_SOLVE_EIGFLOOR", "1").strip().lower()
    return raw in ("1", "true", "yes", "on")


def _psc_pair_replica_chunk_auto_env() -> bool:
    """Tune replica-axis chunk size when unset (Phase-5 audit L3).

    Opt out with ``SIMSOPT_PSC_PAIR_REPLICA_CHUNK_AUTO=0``.
    """

    raw = os.environ.get("SIMSOPT_PSC_PAIR_REPLICA_CHUNK_AUTO", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _psc_disc_face_sym_reduced_env() -> bool:
    """Allow symmetry-reduced path with ``exact_disc_faces`` overlays (audit L6)."""

    raw = os.environ.get("SIMSOPT_PSC_DISC_FACE_SYM", "0").strip().lower()
    return raw in ("1", "true", "yes", "on")


def _psc_movingsub_B_env() -> bool:
    """Moving-puck reduced-free B optimisation (audit L4; API opt-in hook)."""

    raw = os.environ.get("SIMSOPT_PSC_MOVING_PUCK_SUBSET", "0").strip().lower()
    return raw in ("1", "true", "yes", "on")


def _psc_partial_l_jax_env() -> bool:
    """Batched JAX Stage-C partial rows (audit L7). Disabled with ``SIMSOPT_PSC_PARTIAL_L_JAX=0``."""

    raw = os.environ.get("SIMSOPT_PSC_PARTIAL_L_JAX", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _psc_rt_analytic_combo_env() -> bool:
    """Expose pooled ``R{t}`` directional derivatives via :meth:`PSCBulkArray._Rt_analytic_directional_combo` (L8)."""

    raw = os.environ.get("SIMSOPT_PSC_RT_ANALYTIC", "0").strip().lower()
    return raw in ("1", "true", "yes", "on")


def _fc_bulk_warm_handoff_env(solver_mode: Optional[str] = None) -> bool:
    """Optional PSC warm handoff across FC orders (audit L11).

    ``SIMSOPT_FC_WARM_HANDOFF`` explicitly enables (``1``/``on``) or
    disables (``0``/``off``) warm handoff. When the variable is **unset**,
    dipole mode defaults to **on** so continuation stages reuse compiled
    dipole JAX kernels; energy / shell_l2 modes keep the historical
    default (**off**) unless the environment is set.

    Parameters
    ----------
    solver_mode:
        ``PSCBulkArray.solver_mode`` of the receiver when inferring the
        unset-env default.
    """

    raw = os.environ.get("SIMSOPT_FC_WARM_HANDOFF", "").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return False
    if raw in ("1", "true", "yes", "on"):
        return True
    return str(solver_mode or "") == "dipole"


def _psc_rt_fd_eps_env() -> float:
    """Return step size for one-sided FD R/t gradient (default ``1e-5``).

    Controlled by ``SIMSOPT_PSC_RT_FD_EPS``.  One-sided forward FD has
    O(eps) bias; a slightly larger eps (e.g. ``5e-5``) can improve
    signal-to-noise at the cost of marginally larger bias.
    """
    raw = os.environ.get("SIMSOPT_PSC_RT_FD_EPS", "1e-5").strip()
    try:
        return float(raw)
    except ValueError:
        return 1e-5


def _psc_geom_fd_eps_env() -> float:
    """Return step size for one-sided FD geometry (centre/quat) gradient.

    Controlled by ``SIMSOPT_PSC_GEOM_FD_EPS``; default ``1e-5`` to match
    the ``SIMSOPT_PSC_RT_FD_EPS`` convention.  Used by
    :meth:`PSCBulkArray._geometry_fd_gradient` when a caller (typically
    ``stellcoilbench_dipoles``'s
    :class:`~stellcoilbench.coil_optimization
    ._bulk_center_parameterization.ConstrainedJFWrapper`) wants a
    cheap host-side FD gradient w.r.t. caller-supplied per-puck
    ``(dC, dQ)`` perturbations instead of the JAX free-DoF VJP.
    """
    raw = os.environ.get("SIMSOPT_PSC_GEOM_FD_EPS", "1e-5").strip()
    try:
        return float(raw)
    except ValueError:
        return 1e-5


def _psc_continuity_cache_env() -> bool:
    """Enable continuity-projector cache across rebuilds (default ``1``).

    The continuity projector ``Q_c`` only depends on the per-puck topology
    (``n_dof_total``, ``puck_subset``, ``n_phi_rim`` and rim-continuity
    strictness) and never on TF or puck-orientation DOFs.  Caching it
    across rebuilds is the only Stage 3 knob that produced a measurable
    win in the May 2026 audit (~9.7x reduction on continuity cProfile
    cumulative time), so it is now on by default; set
    ``SIMSOPT_PSC_CONTINUITY_CACHE=0`` to disable for diagnostics.
    """
    return os.environ.get("SIMSOPT_PSC_CONTINUITY_CACHE", "1").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _psc_bs_eval_far_kappa_env() -> float:
    """Eval distance / puck scale for dipole far-field B (``0`` = off)."""
    raw = os.environ.get("SIMSOPT_PSC_BS_EVAL_FAR_KAPPA", "0").strip()
    try:
        v = float(raw)
    except ValueError:
        return 0.0
    return max(0.0, v)


def _psc_solve_mode_env() -> str:
    """Reduced solve mode for the free-DOF reduced system.

    Allowed values are ``eigenfloor`` (default, Cholesky on a spectrally
    floored operator) and ``eigk`` (floored-eigenbasis + triangular
    solve).  The ``cg`` and ``lowrank`` modes that briefly shipped in the
    May 2026 audit produced no measurable wall-clock win on the debug
    fixture and were removed to keep the kernel surface small.
    """
    mode = os.environ.get("SIMSOPT_PSC_SOLVE_MODE", "eigenfloor").strip().lower()
    allowed = {"eigenfloor", "eigk"}
    return mode if mode in allowed else "eigenfloor"


def _psc_eigk_trigger_env() -> float:
    """Relative Frobenius drift to refresh a host-side eigK reference (future use)."""
    raw = os.environ.get("SIMSOPT_PSC_EIGK_TRIGGER", "1e-3").strip()
    try:
        return float(raw)
    except ValueError:
        return 1e-3


def _psc_w1_envelope_env() -> bool:
    """Select the W1 envelope-theorem ``custom_vjp`` body for the reduced free-DOF path.

    When ``True`` (the default), :func:`B_at_points` and the three reduced VJP
    runners dispatch through :func:`_B_eval_reduced_free_dof_body_v2`, which
    wraps the ``alpha = L_r^{-1} f_r`` solve in a ``jax.custom_vjp`` whose
    backward differentiates the *scalar* envelope adjoint instead of retracing
    through the linear solve.  Set ``SIMSOPT_PSC_W1_ENVELOPE=0`` to fall back
    to the legacy v1 body (``_B_eval_reduced_free_dof_body``).
    """
    raw = os.environ.get("SIMSOPT_PSC_W1_ENVELOPE", "1").strip().lower()
    if raw in ("0", "false", "no", "off", ""):
        return False
    return raw in ("1", "true", "yes", "on")


def _reduced_free_dof_extras_for_jax(
    pair_far_kappa_default: float = 0.0,
) -> Tuple[bool, float, str, str, float]:
    """Bundle env-driven reduced forward/VJP options for symmetry-reduced free DOFs.

    Returns:
        ``(use_far_dipole_pair, pair_far_kappa, tf_loading, solve_mode_reduced,
        bs_eval_far_kappa)`` with normalized strings for JIT static-arg hashing.

    The ``SIMSOPT_PSC_PAIR_FAR_KAPPA`` environment variable is an explicit
    override.  When unset, ``pair_far_kappa_default`` comes from the
    :class:`PSCBulkArray` instance, e.g. the stellcoilbench YAML knob.
    """
    env_pk = _psc_pair_far_kappa_env_override()
    pk = float(pair_far_kappa_default if env_pk is None else env_pk)
    tfm = str(_psc_tf_loading_env()).lower().replace("-", "_")
    sm = str(_psc_solve_mode_env()).lower()
    return (
        bool(pk > 0.0),
        pk,
        tfm,
        sm,
        float(_psc_bs_eval_far_kappa_env()),
    )


def _classify_pairs(
    centers_all: np.ndarray,
    r_eff_all: np.ndarray,
    base_indices: np.ndarray,
    r_far: float,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Classify symmetry-reduced base-by-replica pairs into near and far sets.

    Parameters
    ----------
    centers_all : np.ndarray
        Replica puck centers with shape ``(n_all, 3)``.
    r_eff_all : np.ndarray
        Effective radius for each replica puck with shape ``(n_all,)``.
    base_indices : np.ndarray
        Replica-to-base map with shape ``(n_all,)``.
    r_far : float
        Distance ratio threshold.  Non-self pair ``(base, replica)`` is far
        when ``distance / (r_eff_i + r_eff_j) >= r_far``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, int, int]
        ``(near_idx, far_idx, n_near, n_far)``.  Index arrays have shape
        ``(n, 2)`` and store ``(base_row, replica_col)`` pairs.  Self-pairs are
        excluded from both arrays; callers add them through the dense kernel.
    """
    centers = np.asarray(centers_all, dtype=float)
    r_eff = np.asarray(r_eff_all, dtype=float).reshape(-1)
    bidx = np.asarray(base_indices, dtype=np.int32).reshape(-1)
    if centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError("centers_all must have shape (n_all, 3)")
    if centers.shape[0] != bidx.size or r_eff.size != bidx.size:
        raise ValueError("centers_all, r_eff_all, and base_indices lengths differ")
    n_all = int(bidx.size)
    n_base = int(np.max(bidx)) + 1 if n_all else 0
    if n_all == 0 or n_base == 0:
        empty = np.empty((0, 2), dtype=np.int32)
        return empty, empty, 0, 0
    # First-occurrence base-rep map; vectorised replacement for the prior
    # ``for jr, bi in enumerate(bidx)`` loop.  ``np.unique`` returns the index
    # of the *first* occurrence of each unique value, which is exactly what
    # the legacy assignment ``base_reps[bi] = jr`` (only-when-unset) produced.
    _uniq, first_idx = np.unique(bidx, return_index=True)
    base_reps = np.full(n_base, -1, dtype=np.int32)
    base_reps[_uniq.astype(np.intp)] = first_idx.astype(np.int32)
    valid_base = base_reps >= 0
    if not np.any(valid_base):
        empty = np.empty((0, 2), dtype=np.int32)
        return empty, empty, 0, 0

    # Build the (n_base, n_all) pair grid, then drop self-pairs.
    base_rows = np.where(valid_base)[0].astype(np.int32)
    i_reps = base_reps[base_rows].astype(np.intp)
    jr_grid = np.broadcast_to(
        np.arange(n_all, dtype=np.int32)[None, :],
        (base_rows.size, n_all),
    )
    ib_grid = np.broadcast_to(base_rows[:, None], (base_rows.size, n_all))
    self_mask = jr_grid == i_reps[:, None].astype(np.int32)
    keep_mask = ~self_mask
    pairs = np.stack([ib_grid[keep_mask], jr_grid[keep_mask]], axis=1).astype(np.int32)

    kappa = float(r_far)
    if kappa <= 0.0:
        empty = np.empty((0, 2), dtype=np.int32)
        return empty, pairs, 0, int(pairs.shape[0])
    if not np.isfinite(kappa):
        empty = np.empty((0, 2), dtype=np.int32)
        return pairs, empty, int(pairs.shape[0]), 0

    # Vectorised distance / sum-radius ratio for all kept pairs.
    ib_keep = pairs[:, 0].astype(np.intp)
    jr_keep = pairs[:, 1].astype(np.intp)
    i_rep_keep = base_reps[ib_keep].astype(np.intp)
    delta = centers[jr_keep] - centers[i_rep_keep]
    dist = np.sqrt(np.sum(delta * delta, axis=1))
    denom = r_eff[i_rep_keep] + r_eff[jr_keep] + 1.0e-12
    is_far = (dist / denom) >= kappa
    far_arr = pairs[is_far]
    near_arr = pairs[~is_far]
    return near_arr, far_arr, int(near_arr.shape[0]), int(far_arr.shape[0])


def _psc_cache_free_vjp_env() -> bool:
    """Return whether to cache a reduced free-DOF VJP pullback after ``B``."""
    return os.environ.get("SIMSOPT_PSC_CACHE_FREE_VJP", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _psc_free_vjp_geometry_env() -> str:
    """Return the free-DOF VJP geometry argument selection mode."""
    mode = os.environ.get("SIMSOPT_PSC_FREE_VJP_GEOMETRY", "auto").strip().lower()
    allowed = {"auto", "full"}
    return mode if mode in allowed else "auto"


def _psc_lcache_enabled() -> bool:
    """Return ``True`` iff the on-disk ``L_base`` cache is enabled.

    Toggled at runtime via ``SIMSOPT_PSC_LCACHE=1``; default off to
    preserve bit-identical behaviour for callers unaware of the cache.
    """
    return os.environ.get("SIMSOPT_PSC_LCACHE", "0") == "1"


def _psc_dipole_freeze_self_l_env(*, default_on: bool) -> bool:
    """Return ``True`` iff the dipole-mode self-inductance freeze is on.

    Controlled by ``SIMSOPT_PSC_FREEZE_SELF_L``: ``"1"`` / ``"true"`` /
    ``"yes"`` enable, ``"0"`` / ``"false"`` / ``"no"`` disable.  When the
    variable is unset (most common case) the default depends on the
    caller:

    * ``solver_mode='dipole'``: default ``True`` (skip self-block
      recomputation across optimisation iterations; matches the lean
      dipole bulk solver Phase D recommendation).
    * Other solver modes: default ``False`` to preserve existing
      semantics bit-for-bit.

    Args:
        default_on: Default value when the env var is unset.

    Returns:
        Resolved freeze setting.
    """
    raw = os.environ.get("SIMSOPT_PSC_FREEZE_SELF_L", None)
    if raw is None:
        return bool(default_on)
    val = raw.strip().lower()
    if val in ("1", "true", "yes", "on"):
        return True
    if val in ("0", "false", "no", "off"):
        return False
    return bool(default_on)


def _psc_dipole_n_radial_env() -> int:
    """Number of radial nodes for the one-time dipole self-tensor compute.

    Controlled by ``SIMSOPT_PSC_DIPOLE_N_RADIAL`` (default ``32``).
    Only affects the one-time per-``(R, t)`` polarizability tensor
    computation at setup; runtime cost is negligible thereafter.
    """
    try:
        return max(4, int(os.environ.get("SIMSOPT_PSC_DIPOLE_N_RADIAL", "32")))
    except ValueError:
        return 32


def _psc_dipole_warn_r_close_env() -> float:
    """Close-packing warn threshold for dipole-mode pucks.

    Controlled by ``SIMSOPT_PSC_DIPOLE_WARN_R_CLOSE`` (default ``2.0``).
    On the first dipole-mode rebuild a :class:`UserWarning` is emitted
    when any pair of base / replica puck centres satisfies
    ``|c_i - c_j| / (R_i + R_j) < threshold`` because the dipole
    truncation error grows rapidly at close packing.
    """
    try:
        return max(0.0, float(os.environ.get("SIMSOPT_PSC_DIPOLE_WARN_R_CLOSE", "2.0")))
    except ValueError:
        return 2.0


def _psc_lcache_dir() -> str:
    """Return the resolved cache directory path (creates it on demand)."""
    default = os.path.join(os.path.expanduser("~"), ".cache", "simsopt", "psc_lbase")
    d = os.environ.get("SIMSOPT_PSC_LCACHE_DIR", default)
    try:
        os.makedirs(d, exist_ok=True)
    except OSError:
        pass
    return d


def _psc_lcache_max_bytes() -> int:
    """Return the LRU soft cap in bytes (default 2 GiB)."""
    try:
        gib = float(os.environ.get("SIMSOPT_PSC_LCACHE_MAX_GIB", "2.0"))
    except ValueError:
        gib = 2.0
    return max(0, int(gib * (1024**3)))


def _psc_lcache_path(cache_key: str) -> str:
    """Resolve the full ``.npz`` path for a given cache key."""
    return os.path.join(_psc_lcache_dir(), f"{cache_key}.npz")


def _psc_lcache_digest(
    quad_points: np.ndarray,
    quad_weights: np.ndarray,
    quad_normals: np.ndarray,
    K_stack: Optional[np.ndarray],
    base_indices: np.ndarray,
    replica_signs: np.ndarray,
    base_reps: np.ndarray,
    *,
    delta_reg: float,
    adaptive_self_reg: bool,
    solver_mode: str,
    G: int,
    n_base_pucks: int,
    nd_per_puck: int,
    null_space_threshold: float,
    eigenfloor_threshold: float,
    reduced_active: bool,
    simsopt_version: str,
    extra: Tuple[Any, ...] = (),
) -> str:
    """Compute a deterministic SHA-256 hash for the current structural state.

    The digest covers every input that feeds into the reduced ``L_base``
    and its downstream Cholesky factors.  Changing any of the inputs
    (e.g. moving a puck, changing a basis parameter, toggling the
    adaptive self-regularization) invalidates the cache entry.

    Parameters mirror the relevant fields of :class:`PSCBulkArray` after
    K-basis assembly; ``K_stack`` may be ``None`` on the heterogeneous
    fallback path, in which case we fall back to hashing the flattened
    quadrature arrays (which together still uniquely determine
    ``L_base``).
    """
    h = hashlib.sha256()
    h.update(simsopt_version.encode("utf-8"))
    h.update(b"|")
    h.update(np.ascontiguousarray(quad_points, dtype=np.float64).tobytes())
    h.update(np.ascontiguousarray(quad_weights, dtype=np.float64).tobytes())
    h.update(np.ascontiguousarray(quad_normals, dtype=np.float64).tobytes())
    if K_stack is not None:
        h.update(np.ascontiguousarray(K_stack, dtype=np.float64).tobytes())
    h.update(np.ascontiguousarray(base_indices, dtype=np.int64).tobytes())
    h.update(np.ascontiguousarray(replica_signs, dtype=np.int8).tobytes())
    h.update(np.ascontiguousarray(base_reps, dtype=np.int64).tobytes())
    meta = (
        f"delta={delta_reg!r};adapt={int(adaptive_self_reg)};"
        f"mode={solver_mode};G={G};nb={n_base_pucks};nd={nd_per_puck};"
        f"nstol={null_space_threshold!r};efth={eigenfloor_threshold!r};"
        f"reduced={int(reduced_active)};"
    )
    h.update(meta.encode("utf-8"))
    for e in extra:
        h.update(repr(e).encode("utf-8"))
    return h.hexdigest()


def _psc_lcache_load(
    cache_key: str,
) -> Optional[Dict[str, np.ndarray]]:
    """Return cached arrays for ``cache_key`` or ``None`` on miss/corruption."""
    path = _psc_lcache_path(cache_key)
    if not os.path.isfile(path):
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            out: Dict[str, np.ndarray] = {k: np.asarray(data[k]) for k in data.files}
        try:
            os.utime(path, None)
        except OSError:
            pass
        return out
    except (OSError, ValueError, EOFError):
        try:
            os.remove(path)
        except OSError:
            pass
        return None


def _psc_lcache_store(
    cache_key: str,
    payload: Dict[str, np.ndarray],
) -> None:
    """Atomically persist a cache entry (write-to-tmp then ``os.replace``)."""
    path = _psc_lcache_path(cache_key)
    d = os.path.dirname(path) or "."
    try:
        os.makedirs(d, exist_ok=True)
    except OSError:
        pass
    fd, tmp = tempfile.mkstemp(prefix=".psc_lcache_", suffix=".npz", dir=d)
    try:
        os.close(fd)
        np.savez_compressed(tmp, **payload)
        os.replace(tmp, path)
    except OSError:
        try:
            os.remove(tmp)
        except OSError:
            pass
        return
    _psc_lcache_trim(_psc_lcache_max_bytes())


def _psc_lcache_trim(max_bytes: int) -> None:
    """Drop oldest ``.npz`` entries until total size falls below ``max_bytes``.

    ``max_bytes <= 0`` disables trimming.  Silent on ``OSError`` so that a
    read-only or missing cache directory cannot break the rebuild.
    """
    if max_bytes <= 0:
        return
    d = _psc_lcache_dir()
    try:
        entries = [
            (os.path.join(d, name), os.path.getmtime(os.path.join(d, name)))
            for name in os.listdir(d)
            if name.endswith(".npz")
        ]
    except OSError:
        return
    try:
        total = sum(os.path.getsize(p) for p, _ in entries)
    except OSError:
        return
    if total <= max_bytes:
        return
    entries.sort(key=lambda t: t[1])  # oldest first
    for path, _ in entries:
        if total <= max_bytes:
            break
        try:
            sz = os.path.getsize(path)
            os.remove(path)
            total -= sz
        except OSError:
            continue


# Optional one-shot warm-up of module-level ``jax.jit`` kernels (cold XLA).
_PSC_JAX_PRIME_ONCE: bool = False


def _maybe_prime_psc_jax_kernels(psc: "PSCBulkArray") -> None:
    """Trace hot PSCBulk JIT bodies once when ``SIMSOPT_JAX_PRIME=1``.

    Uses the arrays from a real :class:`PSCBulkArray` after :meth:`_rebuild`
    so shapes match the user's problem.  Failures are ignored (read-only FS,
    missing optional deps).
    """
    global _PSC_JAX_PRIME_ONCE
    if os.environ.get("SIMSOPT_JAX_PRIME", "0") != "1":
        return
    if _PSC_JAX_PRIME_ONCE:
        return
    # Dipole-mode :meth:`_rebuild` returns before the sheet-basis path
    # allocates ``_jax_phi_work_stack`` / ``_jax_K_stack``; treat missing
    # attributes like ``None`` so JAX priming is skipped instead of raising.
    if getattr(psc, "_jax_phi_work_stack", None) is None or getattr(
        psc, "_jax_K_stack", None
    ) is None:
        return
    try:
        g_tf, gd_tf, I_tf = psc._tf_arrays()
        g_tf = jnp.asarray(g_tf)
        gd_tf = jnp.asarray(gd_tf)
        I_tf = jnp.asarray(I_tf)
        nqtot = int(psc._jax_quad_pts.shape[0])
        Bn_z = jnp.zeros((nqtot,), dtype=jnp.float64)
        ep = jnp.asarray(psc.eval_points[:1])
        _ = _beta_from_tf_jitted(
            psc._jax_Lr_chol,
            psc._jax_Q,
            psc._jax_quad_pts,
            psc._jax_quad_n,
            psc._jax_phi_work_stack,
            psc._jax_w_work_stack,
            psc._jax_base_indices,
            psc._jax_replica_signs,
            g_tf,
            gd_tf,
            I_tf,
        )
        _ = _beta_from_bn_jitted(
            psc._jax_Lr_chol,
            psc._jax_Q,
            psc._jax_phi_work_stack,
            psc._jax_w_work_stack,
            psc._jax_base_indices,
            psc._jax_replica_signs,
            Bn_z,
        )
        _ = _beta_eigenfloor_from_bn_jitted(
            psc._jax_Lr_eigf_chol,
            psc._jax_Q,
            psc._jax_phi_work_stack,
            psc._jax_w_work_stack,
            psc._jax_base_indices,
            psc._jax_replica_signs,
            Bn_z,
        )
        _ = _B_eval_jitted(
            psc._jax_Lr_chol,
            psc._jax_Q,
            psc._jax_quad_pts,
            psc._jax_quad_n,
            psc._jax_phi_work_stack,
            psc._jax_w_work_stack,
            psc._jax_K_stack,
            psc._jax_w_q,
            psc._jax_base_indices,
            psc._jax_replica_signs,
            g_tf,
            gd_tf,
            I_tf,
            ep,
        )
        _ = _B_eval_from_bn_jitted(
            psc._jax_Lr_chol,
            psc._jax_Q,
            psc._jax_quad_pts,
            psc._jax_phi_work_stack,
            psc._jax_w_work_stack,
            psc._jax_K_stack,
            psc._jax_w_q,
            psc._jax_base_indices,
            psc._jax_replica_signs,
            Bn_z,
            ep,
        )
        if (
            os.environ.get("SIMSOPT_JAX_PRIME_REDUCED", "0") == "1"
            and bool(getattr(psc, "_reduced_free_dof_active", False))
            and getattr(psc, "_jax_local_pts", None) is not None
            and getattr(psc, "_jax_Q_c_base", None) is not None
            and getattr(psc, "_jax_L_red_eig_U", None) is not None
        ):
            br = np.asarray(psc._base_reps, dtype=np.int32)
            c_base = jnp.asarray(
                np.stack([psc._all_pucks[i][0] for i in br], axis=0)
            )
            q_base = jnp.asarray(psc._all_pucks_quats[br])
            solve_vjp_mode = _psc_free_solve_vjp_env()
            use_far, pair_k, tf_load, sol_mode, bs_far = (
                _reduced_free_dof_extras_for_jax(
                    psc._resolved_bulk_far_pair_kappa()
                )
            )
            if use_far and pair_k > 0.0:
                c_np, q_np, _, _ = psc._get_base_puck_geometry()
                near_idx, far_idx, _, _ = psc._reduced_far_pair_indices(
                    c_np, q_np, float(pair_k)
                )
            else:
                near_idx = jnp.zeros((0, 2), dtype=jnp.int32)
                far_idx = jnp.zeros((0, 2), dtype=jnp.int32)
            static_tail = (
                psc._jax_base_indices,
                psc._jax_base_reps,
                jnp.asarray(float(psc._symmetry_G)),
                int(psc.nfp),
                bool(psc.stellsym),
                float(psc.regularization_delta),
                float(psc._eigenfloor_threshold),
                bool(psc.adaptive_self_reg),
                bool(psc._resolved_checkpoint_l_pairs()),
                int(psc._resolved_jax_pair_row_chunk()),
                int(psc._resolved_bs_eval_chunk(ep.shape[0])),
                "full",
                solve_vjp_mode,
                use_far,
                float(pair_k),
                psc._jax_m_local[br],
                psc._jax_Q_sym_local[br],
                str(tf_load),
                str(sol_mode),
                float(bs_far),
                bool(_psc_w1_envelope_env()),
                int(psc._resolved_pair_replica_chunk()),
                near_idx,
                far_idx,
            )
            _ = _B_eval_reduced_free_dof_jitted(
                psc._jax_local_pts[br],
                psc._jax_local_K[br],
                psc._jax_local_n[br],
                psc._jax_local_w[br],
                psc._jax_local_phi[br],
                psc._jax_Q_c_base,
                c_base,
                q_base,
                g_tf,
                gd_tf,
                I_tf,
                ep,
                psc._jax_L_red_eig_U,
                psc._jax_L_red_eig_lam,
                *static_tail,
            )
        _PSC_JAX_PRIME_ONCE = True
    except Exception:
        pass


# ======================================================================
# Quaternion helpers
# ======================================================================

_DOFS_PER_PUCK = 9  # cx, cy, cz, q0, qi, qj, qk, R, t


def _axis_to_quaternion(axis: np.ndarray) -> np.ndarray:
    """Convert axis vector to quaternion that rotates ``(0,0,1)`` to ``axis``.

    Uses the scalar-first convention ``[q0, qi, qj, qk]``.
    """
    a = np.asarray(axis, dtype=float)
    a = a / (np.linalg.norm(a) + 1e-30)
    z = np.array([0.0, 0.0, 1.0])
    dot = float(np.dot(z, a))
    if dot > 1 - 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    if dot < -1 + 1e-12:
        return np.array([0.0, 1.0, 0.0, 0.0])
    v = np.cross(z, a)
    v = v / np.linalg.norm(v)
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    s = np.sin(theta / 2.0)
    return np.array([np.cos(theta / 2.0), v[0] * s, v[1] * s, v[2] * s])


def _quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product ``q1 * q2``, scalar-first."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def _quat_left_mult_matrix(q_L: np.ndarray) -> np.ndarray:
    """4x4 matrix ``M`` such that ``q_L * q = M @ q`` (Hamilton product)."""
    w, x, y, z = q_L
    return np.array(
        [
            [w, -x, -y, -z],
            [x, w, -z, y],
            [y, z, w, -x],
            [z, -y, x, w],
        ]
    )


def _rotation_matrix_from_quat(q: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix from quaternion ``[w, x, y, z]`` (NumPy)."""
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q)
    if n < 1e-14:
        return np.eye(3)
    q = q / n
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def _rotation_matrix_from_quat_jax(q: jnp.ndarray) -> jnp.ndarray:
    """JAX-differentiable 3x3 rotation from quaternion ``[w, x, y, z]``."""
    norm_q = jnp.linalg.norm(q)
    q_n = jnp.where(norm_q < 1e-8, q / (norm_q + 1e-8), q / norm_q)
    w, x, y, z = q_n[0], q_n[1], q_n[2], q_n[3]
    return jnp.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def _quaternion_to_axis(q: np.ndarray) -> np.ndarray:
    """Puck axis (unit z-vector of the rotated frame) from quaternion."""
    return _rotation_matrix_from_quat(q) @ np.array([0.0, 0.0, 1.0])


def _rotation_matrix_local_to_global(axis_z: np.ndarray) -> np.ndarray:
    """Rotation matrix mapping ``(0,0,1)`` to ``axis_z`` (backward compat).

    Converts to quaternion internally; kept for VTK and test imports.
    """
    q = _axis_to_quaternion(axis_z)
    return _rotation_matrix_from_quat(q)


# ----------------------------------------------------------------------
# Module-level JAX (stable across PSCBulkArray._rebuild; avoids recompile)
# ----------------------------------------------------------------------

_EPS_BS = 1e-8
_EIGENFLOOR_THRESHOLD = 1e-10


# ----------------------------------------------------------------------
# Symmetry-aware body signatures
# ----------------------------------------------------------------------
#
# Every JIT body below operates on **work-sized** arrays
# ``(phi_work_stack, w_work_stack, L_work, Q_work)`` of shape
# ``(n_work, nq_per, ...)``, where ``n_work`` is the number of *independent*
# pucks we need to solve for:
#
# - Full path (no TF symmetry or no puck replication): ``n_work == n_all``
#   and ``base_indices == arange(n_all)``; folding and gathering become
#   no-ops and the code reduces to the old behavior.
# - Reduced path (TF + puck layout share an :math:`n_{fp}`/stellsym
#   group ``G``): ``n_work == n_base`` with ``base_indices[r] = base(r)``
#   mapping replica ``r`` to its base puck.  ``L_work`` is ``L_base`` as
#   built by :func:`shell_inductance_matrix_symmetric_reduced`; the
#   normal field ``Bn`` is computed on every replica and folded into
#   base-space via ``segment_sum``; the solved ``beta_base`` is gathered
#   back to every replica via a ``base_indices`` index lookup before the
#   full-replica Biot-Savart.
#
# This keeps a **single** JIT body per entry point instead of duplicating
# the full/reduced code paths, while still caching separately because the
# traced array shapes differ.


def _psc_disable_reduced_free() -> bool:
    """If ``1``, disable symmetry-reduced free-puck-DoF :func:`B_at_points` path.

    Compare against the full-N² assembly for debugging.  Environment variable:
    ``SIMSOPT_PSC_DISABLE_REDUCED_FREE``.
    """
    return os.environ.get("SIMSOPT_PSC_DISABLE_REDUCED_FREE", "0") == "1"


def _quat_multiply_jax(q1: jnp.ndarray, q2: jnp.ndarray) -> jnp.ndarray:
    """Hamilton product ``q1 * q2`` (scalar-first), matching :func:`_quat_multiply`."""
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
    return jnp.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def _replicate_pucks_jax(
    centers_base: jnp.ndarray,
    quats_base: jnp.ndarray,
    nfp: int,
    stellsym: bool,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Differentiable replica of :meth:`PSCBulkArray._replicate_pucks` (JAX).

    Returns:
        ``centers_all``, ``quats_all`` (``n_all``, 3) / (``n_all``, 4) and
        ``signs`` (``n_all``,) with :math:`\\pm 1` (pure rotation = ``+1``,
        stellsym image = ``-1``).  Loop order matches :meth:`_replicate_pucks`
        (``i`` outer, ``jfp`` middle, stellsym last).
    """
    n_base = int(centers_base.shape[0])
    blocks_c: List[jnp.ndarray] = []
    blocks_q: List[jnp.ndarray] = []
    blocks_s: List[jnp.ndarray] = []
    for i in range(n_base):
        c = centers_base[i]
        q = quats_base[i]
        for jfp in range(nfp):
            angle = 2.0 * jnp.pi * jfp / float(nfp)
            ca = jnp.cos(angle)
            sa = jnp.sin(angle)
            rot3 = jnp.array(
                [
                    [ca, -sa, 0.0],
                    [sa, ca, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            )
            q_rot = jnp.array([jnp.cos(0.5 * angle), 0.0, 0.0, jnp.sin(0.5 * angle)])
            c2 = rot3 @ c
            q2 = _quat_multiply_jax(q_rot, q)
            blocks_c.append(c2)
            blocks_q.append(q2)
            blocks_s.append(jnp.array(1.0, dtype=centers_base.dtype))
            if stellsym:
                S = jnp.diag(jnp.array([1.0, -1.0, -1.0], dtype=centers_base.dtype))
                q_stell = jnp.array([0.0, 1.0, 0.0, 0.0], dtype=centers_base.dtype)
                c3 = S @ c2
                q3 = _quat_multiply_jax(q_stell, q2)
                blocks_c.append(c3)
                blocks_q.append(q3)
                blocks_s.append(jnp.array(-1.0, dtype=centers_base.dtype))
    centers_all = jnp.stack(blocks_c, axis=0)
    quats_all = jnp.stack(blocks_q, axis=0)
    signs = jnp.stack(blocks_s, axis=0)
    return centers_all, quats_all, signs


def _transform_local_to_all_replicas(
    centers_all: jnp.ndarray,
    quats_all: jnp.ndarray,
    local_pts_base: jnp.ndarray,
    local_K_base: jnp.ndarray,
    local_n_base: jnp.ndarray,
    local_w_base: jnp.ndarray,
    base_idx_per_rep: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Rotate base-local basis stacks to each replica (broadcast gather by base)."""

    def one_rep(
        r: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        r = r.astype(jnp.int32)
        b = base_idx_per_rep[r].astype(jnp.int32)
        Rm = _rotation_matrix_from_quat_jax(quats_all[r])
        c = centers_all[r]
        pts = (Rm @ local_pts_base[b].T).T + c
        Kg = jnp.einsum("ij,qkj->qki", Rm, local_K_base[b])
        ng = (Rm @ local_n_base[b].T).T
        w = local_w_base[b]
        return pts, Kg, ng, w

    n_all = int(centers_all.shape[0])
    out = jax.lax.map(one_rep, jnp.arange(n_all, dtype=jnp.int32))
    return out[0], out[1], out[2], out[3]


def _pair_inductance_block(
    pts_stack: jnp.ndarray,
    K_stack: jnp.ndarray,
    local_w_stack: jnp.ndarray,
    i: jnp.ndarray,
    j: jnp.ndarray,
    delta_reg: float,
    adaptive_self_reg: bool,
) -> jnp.ndarray:
    r = pts_stack[i, :, None, :] - pts_stack[j, None, :, :]
    if adaptive_self_reg:
        w_i = local_w_stack[i]
        w_j = local_w_stack[j]
        delta_i = _SELF_REG_COEFF * jnp.sqrt(w_i)
        delta_j = _SELF_REG_COEFF * jnp.sqrt(w_j)
        delta_pair = 0.5 * (delta_i[:, None] + delta_j[None, :])
        dist = jnp.sqrt(jnp.sum(r**2, axis=-1) + delta_pair**2)
    else:
        dist = jnp.sqrt(jnp.sum(r**2, axis=-1) + delta_reg**2)
    dot = jnp.einsum("iax,jbx->ijab", K_stack[i], K_stack[j])
    kernel = dot / dist[..., None, None]
    return MU0_OVER_4PI * jnp.einsum(
        "ijab,i,j->ab", kernel, local_w_stack[i], local_w_stack[j]
    )


def _row_segment_sum_replica_chunked(
    scaled_block: Any,
    base_indices: jnp.ndarray,
    n_all: int,
    n_base: int,
    nd: int,
    replica_chunk: int,
    dtype: Any,
) -> jnp.ndarray:
    """Fold ``segment_sum(vmap(scaled_block)(arange(n_all)), base_indices, n_base)``.

    When ``replica_chunk <= 0`` or ``replica_chunk >= n_all`` this is the
    monolithic ``vmap`` + ``segment_sum`` (current behaviour).  When
    ``0 < replica_chunk < n_all`` the replica axis is consumed in
    :func:`jax.lax.scan` chunks of ``replica_chunk`` and the segment sum is
    accumulated incrementally into an ``(n_base, nd, nd)`` carry, so peak
    autodiff memory inside ``scaled_block`` (which holds the
    ``(replica_chunk, nq, nq, nd, nd)`` pair kernel) scales as
    ``replica_chunk * nq * nq * nd * nd * 8`` instead of the full
    ``n_all * nq * nq * nd * nd * 8`` bytes.  Output is mathematically
    identical to the unchunked path for any ``replica_chunk``.

    Args:
        scaled_block: callable mapping a single ``j`` index (int32 scalar)
            to an ``(nd, nd)`` block already weighted by ``G * sigma_j``.
            Must broadcast under :func:`jax.vmap` along the leading axis.
        base_indices: ``(n_all,)`` int map from replica index to base index
            in ``[0, n_base)``.
        n_all: total number of replicas (Python int, JIT-static).
        n_base: number of base pucks (Python int, JIT-static).
        nd: per-puck local DOF count (Python int, JIT-static; needed to
            allocate the carry buffer).
        replica_chunk: replica-axis chunk width (Python int, JIT-static).
            ``<= 0`` or ``>= n_all`` keeps the legacy ``vmap`` path.
        dtype: dtype of the carry buffer / output array.

    Returns:
        ``(n_base, nd, nd)`` segment-summed inductance row.
    """
    rc = int(replica_chunk)
    bi = base_indices.astype(jnp.int32)
    if rc <= 0 or rc >= n_all:
        j_full = jnp.arange(n_all, dtype=jnp.int32)
        blocks = jax.vmap(scaled_block)(j_full)
        return jax.ops.segment_sum(blocks, bi, num_segments=n_base)

    n_c = (n_all + rc - 1) // rc

    def _scan_one(carry: jnp.ndarray, ic: jnp.ndarray) -> Tuple[jnp.ndarray, None]:
        start = ic * rc
        idx = start + jnp.arange(rc, dtype=jnp.int32)
        valid = idx < n_all
        idx_safe = jnp.where(valid, idx, 0)
        chunk = jax.vmap(scaled_block)(idx_safe)
        chunk = chunk * valid[:, None, None].astype(chunk.dtype)
        seg_idx = bi[idx_safe]
        chunk_seg = jax.ops.segment_sum(chunk, seg_idx, num_segments=n_base)
        return carry + chunk_seg, None

    carry0 = jnp.zeros((n_base, nd, nd), dtype=dtype)
    out, _ = jax.lax.scan(_scan_one, carry0, jnp.arange(n_c, dtype=jnp.int32))
    return out


def _fold_Bn_to_work(
    Bn_flat: jnp.ndarray,
    base_indices: jnp.ndarray,
    signs: jnp.ndarray,
    n_work: int,
    nq_per: int,
) -> jnp.ndarray:
    """``Bn_work[i_base, q] = sum_{r in orbit(i_base)} sigma_r Bn_all[r, q]``.

    ``base_indices`` is an ``(n_all,)`` integer map from replica index to
    base index (``0..n_work-1``); ``signs`` is the matching ``(n_all,)``
    array of :math:`\\sigma_r \\in \\{+1, -1\\}`.  For the full path
    (``n_work == n_all``) ``base_indices`` is ``arange(n_all)`` and
    ``signs`` is all ``+1``, so this is the identity.

    The sign convention follows the induced-current parity under the
    symmetry group: pure rotations keep the current direction
    (``sigma = +1``) while simsopt's stellsym image flips the coil
    current (``sigma = -1``), producing ``Bn(Sx) = -Bn(x)`` at
    symmetry-related quadrature points.
    """
    n_all = base_indices.shape[0]
    Bn_stack = Bn_flat.reshape(n_all, nq_per)
    signed = signs.astype(Bn_stack.dtype)[:, None] * Bn_stack
    return jax.ops.segment_sum(signed, base_indices, num_segments=n_work)


def _gather_beta_work_to_all(
    beta_work: jnp.ndarray,
    base_indices: jnp.ndarray,
    signs: jnp.ndarray,
    nd_per: int,
) -> jnp.ndarray:
    """``beta_all[r, :] = sigma_r * beta_work[base_indices[r], :]`` flattened.

    Dual of :func:`_fold_Bn_to_work`: scatters ``beta_work`` (base DOFs)
    back to every replica with the matching per-replica sign so that the
    subsequent Biot-Savart sum over the full-replica ``K_all_stack`` sees
    the correct current pattern.  On the full path ``signs`` is all
    ``+1`` and this reduces to a plain gather.
    """
    n_work_dof = beta_work.shape[0]
    n_work = n_work_dof // nd_per
    beta_work_stack = beta_work.reshape(n_work, nd_per)
    beta_all_stack = beta_work_stack[base_indices]
    return (signs.astype(beta_all_stack.dtype)[:, None] * beta_all_stack).reshape(-1)


def _beta_from_tf_body(
    Lr_chol: jnp.ndarray,
    Qm: jnp.ndarray,
    quad_pts: jnp.ndarray,
    quad_n: jnp.ndarray,
    phi_work_stack: jnp.ndarray,
    w_work_stack: jnp.ndarray,
    base_indices: jnp.ndarray,
    signs: jnp.ndarray,
    g_tf: jnp.ndarray,
    gd_tf: jnp.ndarray,
    I_tf: jnp.ndarray,
) -> jnp.ndarray:
    """TF-only modal coefficients using Q-projected reduced solve (fast path).

    ``Lr_chol`` is the lower Cholesky factor of ``L_red + jitter I`` with
    the same ``jitter`` as :func:`~simsopt.field.bulk_inductance.shell_solve_linear_pure`,
    built once in :meth:`PSCBulkArray._rebuild`.

    Supports both the full (``n_work == n_all``) and the symmetry-reduced
    (``n_work == n_base``) paths via a single unified body; ``signs`` is
    all ``+1`` on the full path and carries the stellsym parity on the
    reduced path (see :func:`_fold_Bn_to_work`).
    """
    gammas_tf = jnp.asarray(g_tf)
    gammadash_tf = jnp.asarray(gd_tf)
    currents_tf = jnp.asarray(I_tf)

    def Bn_at_i(i):
        B = _B_at_point_from_coil_set_pure(
            quad_pts[i],
            gammas_tf,
            gammadash_tf,
            currents_tf,
            -1,
            _EPS_BS,
        )
        return jnp.dot(B, quad_n[i])

    Bn_all = jax.vmap(Bn_at_i)(jnp.arange(quad_pts.shape[0]))
    n_work, nq_per, _ = phi_work_stack.shape
    Bn_work_stack = _fold_Bn_to_work(Bn_all, base_indices, signs, n_work, nq_per)
    f = shell_loading_vector_stacked_pure(
        phi_work_stack,
        w_work_stack,
        Bn_work_stack.reshape(-1),
    )
    fr = Qm.T @ f
    alpha = shell_solve_prefactored_pure(Lr_chol, fr)
    return expand_beta_reduced(alpha, Qm)


_beta_from_tf_jitted = jax.jit(_beta_from_tf_body)


def _beta_from_bn_body(
    Lr_chol: jnp.ndarray,
    Qm: jnp.ndarray,
    phi_work_stack: jnp.ndarray,
    w_work_stack: jnp.ndarray,
    base_indices: jnp.ndarray,
    signs: jnp.ndarray,
    Bn_all: jnp.ndarray,
) -> jnp.ndarray:
    """Reduced solve given ``Bn`` at every **all-replica** quad point.

    ``Lr_chol`` is the Cholesky factor of ``L_red + jitter I`` (see
    :func:`_beta_from_tf_body`).  When the
    reduced path is active, ``Bn_all`` is folded into base-space with the
    per-replica ``signs`` before forming the loading vector; for the full
    path, ``base_indices`` is the identity and ``signs`` is all ``+1`` so
    the fold is a no-op.
    """
    n_work, nq_per, _ = phi_work_stack.shape
    Bn_work_stack = _fold_Bn_to_work(Bn_all, base_indices, signs, n_work, nq_per)
    f = shell_loading_vector_stacked_pure(
        phi_work_stack,
        w_work_stack,
        Bn_work_stack.reshape(-1),
    )
    fr = Qm.T @ f
    alpha = shell_solve_prefactored_pure(Lr_chol, fr)
    return expand_beta_reduced(alpha, Qm)


_beta_from_bn_jitted = jax.jit(_beta_from_bn_body)


def _beta_eigenfloor_from_bn_body(
    Lr_eigf_chol: jnp.ndarray,
    Q: jnp.ndarray,
    phi_work_stack: jnp.ndarray,
    w_work_stack: jnp.ndarray,
    base_indices: jnp.ndarray,
    signs: jnp.ndarray,
    Bn_all: jnp.ndarray,
) -> jnp.ndarray:
    """Rim-continuity-projected eigenfloor solve given ``Bn``.

    ``Lr_eigf_chol`` is the Cholesky factor of the eigenvalue-floored matrix
    used in :func:`~simsopt.field.bulk_inductance.shell_solve_eigenfloor_pure`.
    Here ``Q = Q_c Q_L`` is the full null-space-trimmed projector already
    composed at rebuild time.  This avoids materialising ``L_work`` on
    device.  Supports both the full and reduced paths via signed
    ``base_indices`` folding; see :func:`_fold_Bn_to_work`.
    """
    n_work, nq_per, _ = phi_work_stack.shape
    Bn_work_stack = _fold_Bn_to_work(Bn_all, base_indices, signs, n_work, nq_per)
    f = shell_loading_vector_stacked_pure(
        phi_work_stack,
        w_work_stack,
        Bn_work_stack.reshape(-1),
    )
    f_r = Q.T @ f
    alpha = shell_solve_prefactored_pure(Lr_eigf_chol, f_r)
    return Q @ alpha


_beta_eigenfloor_from_bn_jitted = jax.jit(_beta_eigenfloor_from_bn_body)


def _B_eval_from_tf_body(
    Lr_chol: jnp.ndarray,
    Qm: jnp.ndarray,
    quad_pts: jnp.ndarray,
    quad_n: jnp.ndarray,
    phi_work_stack: jnp.ndarray,
    w_work_stack: jnp.ndarray,
    K_all_stack: jnp.ndarray,
    w_quad_all: jnp.ndarray,
    base_indices: jnp.ndarray,
    signs: jnp.ndarray,
    g_tf: jnp.ndarray,
    gd_tf: jnp.ndarray,
    I_tf: jnp.ndarray,
    pts_eval: jnp.ndarray,
) -> jnp.ndarray:
    """Passive bulk B at eval points; TF-only VJP fast path.

    Solves in work (base) DOFs using the pre-projected reduced matrix
    ``L_red``, then gathers ``beta_base`` back to every replica (with the
    per-replica sign) for the full-replica Biot-Savart summation using
    ``K_all_stack``.
    """
    b_work = _beta_from_tf_body(
        Lr_chol,
        Qm,
        quad_pts,
        quad_n,
        phi_work_stack,
        w_work_stack,
        base_indices,
        signs,
        g_tf,
        gd_tf,
        I_tf,
    )
    nd_per = K_all_stack.shape[2]
    b_all = _gather_beta_work_to_all(b_work, base_indices, signs, nd_per)
    return shell_biot_savart_stacked_pure(
        K_all_stack,
        quad_pts,
        w_quad_all,
        b_all,
        jnp.asarray(pts_eval),
        eps=_EPS_BS,
    )


_B_eval_jitted = jax.jit(_B_eval_from_tf_body)


def _B_eval_from_bn_body(
    Lr_chol: jnp.ndarray,
    Qm: jnp.ndarray,
    quad_pts: jnp.ndarray,
    phi_work_stack: jnp.ndarray,
    w_work_stack: jnp.ndarray,
    K_all_stack: jnp.ndarray,
    w_quad_all: jnp.ndarray,
    base_indices: jnp.ndarray,
    signs: jnp.ndarray,
    Bn_all: jnp.ndarray,
    pts_eval: jnp.ndarray,
) -> jnp.ndarray:
    """Passive bulk B at eval points given ``Bn`` (avoids JAX Biot-Savart on TF)."""
    b_work = _beta_from_bn_body(
        Lr_chol,
        Qm,
        phi_work_stack,
        w_work_stack,
        base_indices,
        signs,
        Bn_all,
    )
    nd_per = K_all_stack.shape[2]
    b_all = _gather_beta_work_to_all(b_work, base_indices, signs, nd_per)
    return shell_biot_savart_stacked_pure(
        K_all_stack,
        quad_pts,
        w_quad_all,
        b_all,
        jnp.asarray(pts_eval),
        eps=_EPS_BS,
    )


_B_eval_from_bn_jitted = jax.jit(_B_eval_from_bn_body)


def _shell_biot_savart_flat_chunked(
    k_flat: jnp.ndarray,
    source_pts: jnp.ndarray,
    source_w: jnp.ndarray,
    pts_eval: jnp.ndarray,
    eval_chunk: int = 0,
) -> jnp.ndarray:
    """Evaluate shell Biot-Savart from flattened sources with optional chunking."""

    def b_at(x: jnp.ndarray) -> jnp.ndarray:
        rvec = x[None, :] - source_pts
        rn = jnp.sqrt(jnp.sum(rvec**2, axis=-1) + _EPS_BS**2)
        integr = jnp.cross(k_flat, rvec) / (rn[:, None] ** 3)
        return MU0_OVER_4PI * jnp.sum(integr * source_w[:, None], axis=0)

    pts = jnp.asarray(pts_eval)
    n_eval = int(pts.shape[0])
    chunk = int(eval_chunk)
    if chunk <= 0 or chunk >= n_eval:
        return jax.vmap(b_at)(pts)

    n_chunks = (n_eval + chunk - 1) // chunk
    pad = n_chunks * chunk - n_eval
    pts_pad = jnp.pad(pts, ((0, pad), (0, 0)))
    pts_blocks = pts_pad.reshape(n_chunks, chunk, 3)

    def one_block(block: jnp.ndarray) -> jnp.ndarray:
        return jax.vmap(b_at)(block)

    return jax.lax.map(one_block, pts_blocks).reshape(n_chunks * chunk, 3)[:n_eval]


def _eigenfloor_regularized_matrix(
    L: jnp.ndarray,
    threshold: float = 1e-10,
    jitter: float = 1e-10,
) -> jnp.ndarray:
    """Regularize ``L`` with the same eigenfloor used by the free-DOF solve."""
    lam, V = jnp.linalg.eigh(L)
    max_abs = jnp.max(jnp.abs(lam))
    floor = threshold * max_abs
    lam_floor = jnp.maximum(lam, floor)
    L_reg = (V * lam_floor[None, :]) @ V.T
    return L_reg + jitter * jnp.eye(L.shape[0], dtype=L.dtype)


@jax.custom_vjp
def _solve_eigenfloor_implicit_vjp(
    L: jnp.ndarray,
    f: jnp.ndarray,
    threshold: float = 1e-10,
) -> jnp.ndarray:
    """Solve with eigenfloor primal and implicit linear-solve VJP."""
    L_reg = _eigenfloor_regularized_matrix(L, threshold=threshold, jitter=1e-10)
    return jnp.linalg.solve(L_reg, f)


def _solve_eigenfloor_implicit_vjp_fwd(
    L: jnp.ndarray,
    f: jnp.ndarray,
    threshold: float = 1e-10,
) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
    L_reg = _eigenfloor_regularized_matrix(L, threshold=threshold, jitter=1e-10)
    alpha = jnp.linalg.solve(L_reg, f)
    return alpha, (L_reg, alpha)


def _solve_eigenfloor_implicit_vjp_bwd(
    res: Tuple[jnp.ndarray, jnp.ndarray],
    ct_alpha: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, None]:
    L_reg, alpha = res
    lam_f = jnp.linalg.solve(L_reg.T, ct_alpha)
    lam_L = -0.5 * (jnp.outer(lam_f, alpha) + jnp.outer(alpha, lam_f))
    return lam_L, lam_f, None


_solve_eigenfloor_implicit_vjp.defvjp(
    _solve_eigenfloor_implicit_vjp_fwd,
    _solve_eigenfloor_implicit_vjp_bwd,
)


@jax.custom_vjp
def _eigK_solve_with_correct_vjp(
    L: jnp.ndarray,
    U_cached: jnp.ndarray,
    lambda_cached: jnp.ndarray,
    f: jnp.ndarray,
    threshold: float = 1e-10,
) -> jnp.ndarray:
    """Forward uses cached eigenbasis; backward applies the implicit-function VJP.

    The forward solve ``alpha = U (U^T f / lambda)`` is the cheap eigK shortcut
    using a pre-eigendecomposition cached on the host.  The backward
    differentiates through ``L`` and ``f`` exactly the same way as
    :func:`_solve_eigenfloor_implicit_vjp` (regularize ``L`` with the same
    eigenfloor and solve ``L_reg lam = ct_alpha``), so gradients flow correctly
    through the *current* ``L`` even when ``(U_cached, lambda_cached)`` are
    NumPy-host snapshots from an earlier ``L``.

    Args:
        L: Current symmetric reduced inductance matrix ``(n, n)``.
        U_cached: ``(n, k)`` host-derived eigenvectors (treated as constants by autodiff).
        lambda_cached: ``(k,)`` host-derived floored eigenvalues (treated as constants).
        f: Reduced load vector ``(n,)``.
        threshold: Same eigenfloor used for the implicit backward.

    Returns:
        ``alpha`` of shape ``(n,)``.

    """
    return shell_solve_eigK_pure(U_cached, lambda_cached, f)


def _eigK_solve_with_correct_vjp_fwd(
    L: jnp.ndarray,
    U_cached: jnp.ndarray,
    lambda_cached: jnp.ndarray,
    f: jnp.ndarray,
    threshold: float = 1e-10,
) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
    alpha = shell_solve_eigK_pure(U_cached, lambda_cached, f)
    L_reg = _eigenfloor_regularized_matrix(L, threshold=threshold, jitter=1e-10)
    return alpha, (L_reg, alpha)


def _eigK_solve_with_correct_vjp_bwd(
    res: Tuple[jnp.ndarray, jnp.ndarray],
    ct_alpha: jnp.ndarray,
) -> Tuple[jnp.ndarray, None, None, jnp.ndarray, None]:
    L_reg, alpha = res
    lam_f = jnp.linalg.solve(L_reg.T, ct_alpha)
    lam_L = -0.5 * (jnp.outer(lam_f, alpha) + jnp.outer(alpha, lam_f))
    return lam_L, None, None, lam_f, None


_eigK_solve_with_correct_vjp.defvjp(
    _eigK_solve_with_correct_vjp_fwd,
    _eigK_solve_with_correct_vjp_bwd,
)


def _assemble_L_red_inner(
    centers_base: jnp.ndarray,
    quats_base: jnp.ndarray,
    local_pts_base: jnp.ndarray,
    local_K_base: jnp.ndarray,
    local_n_base: jnp.ndarray,
    local_w_base: jnp.ndarray,
    Q_c_base: jnp.ndarray,
    base_indices: jnp.ndarray,
    base_reps: jnp.ndarray,
    G_float: jnp.ndarray,
    nfp: int,
    stellsym: bool,
    delta_reg: float,
    adaptive_self_reg: bool,
    checkpoint_L_pairs: bool = False,
    pair_row_chunk: int = 0,
    use_far_dipole_pair: bool = False,
    pair_far_kappa: float = 0.0,
    m_local_base: Optional[jnp.ndarray] = None,
    pair_replica_chunk: int = 0,
    near_pair_indices: Optional[jnp.ndarray] = None,
    far_pair_indices: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Pure assembly of the symmetry-reduced inductance matrix ``L_red = Q_c^T L_base Q_c``.

    Used both inside the W1 envelope ``custom_vjp`` (for primal and adjoint
    retraces) and in regression tests.  Mirrors the L-assembly block of
    :func:`_B_eval_reduced_free_dof_body` exactly.

    Args:
        centers_base: ``(n_base, 3)`` base puck centers.
        quats_base: ``(n_base, 4)`` base puck quaternions.
        local_pts_base, local_K_base, local_n_base, local_w_base: per-base local stacks.
        Q_c_base: ``(n_base*nd, n_red)`` continuity constraint range basis.
        base_indices: ``(n_all,)`` map from replica index to base index.
        base_reps: ``(n_base,)`` representative replica index for each base.
        G_float: scalar ``(|G|,)`` symmetry order, used as a fold weight.
        nfp, stellsym, delta_reg, adaptive_self_reg: assembly statics.
        checkpoint_L_pairs: ``jax.checkpoint`` per pair block when ``True``.
        pair_row_chunk: positive integer enables ``lax.scan`` over base rows.
        use_far_dipole_pair, pair_far_kappa, m_local_base: W2 multipole shortcut.
        pair_replica_chunk: positive integer enables ``lax.scan`` over the
            ``n_all`` replica axis inside each base row, capping the per-row
            pair-kernel batch at ``pair_replica_chunk * nq * nq * nd * nd``
            instead of the full ``n_all * nq * nq * nd * nd``.  ``0`` or
            ``>= n_all`` falls back to the monolithic ``vmap``.

    Returns:
        ``(n_red, n_red)`` reduced inductance ``L_red``.

    """
    n_base = int(local_pts_base.shape[0])
    nd = int(local_K_base.shape[2])
    n_all = int(base_indices.shape[0])
    centers_all, quats_all, signs = _replicate_pucks_jax(
        centers_base, quats_base, nfp, stellsym
    )
    pts_stack, K_stack, _n_stack, w_stack = _transform_local_to_all_replicas(
        centers_all,
        quats_all,
        local_pts_base,
        local_K_base,
        local_n_base,
        local_w_base,
        base_indices,
    )
    if m_local_base is None:
        m_local_base = jnp.zeros((n_base, nd, 3), dtype=K_stack.dtype)

    def l_block_pair(i_rep: jnp.ndarray, j_rep: jnp.ndarray) -> jnp.ndarray:
        return _pair_inductance_block(
            pts_stack,
            K_stack,
            w_stack,
            i_rep,
            j_rep,
            float(delta_reg),
            bool(adaptive_self_reg),
        )

    l_fn: Any = l_block_pair
    if checkpoint_L_pairs and not use_far_dipole_pair:
        l_fn = jax.checkpoint(l_block_pair)

    rc = int(pair_replica_chunk)
    block_dtype = K_stack.dtype

    if (
        use_far_dipole_pair
        and pair_far_kappa > 0.0
        and near_pair_indices is not None
        and far_pair_indices is not None
    ):
        l_rows0 = jnp.zeros((n_base, n_base, nd, nd), dtype=block_dtype)

        def _add_dense_pair(carry: jnp.ndarray, pair: jnp.ndarray) -> Tuple[jnp.ndarray, None]:
            ib = pair[0].astype(jnp.int32)
            jr = pair[1].astype(jnp.int32)
            i_rep = base_reps[ib]
            bj = base_indices[jr]
            blk = G_float * signs[jr] * l_fn(i_rep, jr)
            return carry.at[ib, bj, :, :].add(blk), None

        self_pairs = jnp.stack(
            [jnp.arange(n_base, dtype=jnp.int32), base_reps.astype(jnp.int32)], axis=1
        )
        l_rows, _ = jax.lax.scan(_add_dense_pair, l_rows0, self_pairs)
        l_rows, _ = jax.lax.scan(
            _add_dense_pair, l_rows, near_pair_indices.astype(jnp.int32)
        )

        far_idx = far_pair_indices.astype(jnp.int32)

        def _far_pair_block(pair: jnp.ndarray) -> jnp.ndarray:
            ib = pair[0].astype(jnp.int32)
            jr = pair[1].astype(jnp.int32)
            i_rep = base_reps[ib]
            bj = base_indices[jr]
            Rij = centers_all[jr] - centers_all[i_rep]
            Ri = _rotation_matrix_from_quat_jax(quats_all[i_rep])
            Rj = _rotation_matrix_from_quat_jax(quats_all[jr])
            m_i = jnp.einsum("ij,nj->ni", Ri, m_local_base[ib, :, :])
            m_j = jnp.einsum("ij,nj->ni", Rj, m_local_base[bj, :, :])
            return G_float * signs[jr] * pair_inductance_multipole(m_i, m_j, Rij, order=1)

        if far_idx.shape[0] > 0:
            far_blocks = jax.vmap(_far_pair_block)(far_idx)
            l_rows = l_rows.at[far_idx[:, 0], base_indices[far_idx[:, 1]], :, :].add(
                far_blocks
            )
    else:

        def one_row_f(ib: jnp.ndarray) -> jnp.ndarray:
            i_rep = base_reps[ib.astype(jnp.int32)]

            def scaled_block(jr: jnp.ndarray) -> jnp.ndarray:
                blk = l_fn(i_rep, jr)
                return G_float * signs[jr] * blk

            return _row_segment_sum_replica_chunked(
                scaled_block, base_indices, n_all, n_base, nd, rc, block_dtype
            )

        n_chunk = int(pair_row_chunk)
        if n_chunk > 0 and n_chunk < n_base:

            def _scan_one_chunk(_: None, ic: jnp.ndarray) -> Tuple[None, jnp.ndarray]:
                start = ic * n_chunk
                idx = start + jnp.arange(n_chunk, dtype=jnp.int32)
                valid = idx < n_base
                idx_safe = jnp.where(valid, idx, 0)
                rows_b = jax.vmap(one_row_f)(idx_safe)
                rows_b = rows_b * valid[:, None, None, None].astype(rows_b.dtype)
                return None, rows_b

            n_c = (n_base + n_chunk - 1) // n_chunk
            _, row_blocks = jax.lax.scan(
                _scan_one_chunk, None, jnp.arange(n_c, dtype=jnp.int32)
            )
            l_rows = row_blocks.reshape(n_c * n_chunk, n_base, nd, nd)[:n_base, ...]
        else:
            l_rows = jax.vmap(one_row_f)(jnp.arange(n_base, dtype=jnp.int32))
    l_base = l_rows.transpose(0, 2, 1, 3).reshape(n_base * nd, n_base * nd)
    l_base = 0.5 * (l_base + l_base.T)
    return Q_c_base.T @ l_base @ Q_c_base


def _assemble_f_red_inner(
    centers_base: jnp.ndarray,
    quats_base: jnp.ndarray,
    g_tf: jnp.ndarray,
    gd_tf: jnp.ndarray,
    I_tf: jnp.ndarray,
    local_pts_base: jnp.ndarray,
    local_K_base: jnp.ndarray,
    local_n_base: jnp.ndarray,
    local_w_base: jnp.ndarray,
    local_phi_base: jnp.ndarray,
    Q_c_base: jnp.ndarray,
    base_indices: jnp.ndarray,
    nfp: int,
    stellsym: bool,
    tf_loading: str,
    m_local_base: Optional[jnp.ndarray] = None,
    Q_sym_local: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Pure assembly of the symmetry-reduced load vector ``f_red = Q_c^T f_base``.

    Selects between ``bn_quad`` (default surface-quadrature dot product),
    ``a_quad`` (dipole closed-shell ``f = m·B``), and ``a_taylor``
    (``f = m·B + Q : ∇B``).  Used in tandem with
    :func:`_assemble_L_red_inner` inside the W1 envelope ``custom_vjp``.

    Args:
        centers_base, quats_base: base DoFs (differentiable).
        g_tf, gd_tf, I_tf: TF coil set arrays (differentiable).
        local_pts_base, local_K_base, local_n_base, local_w_base, local_phi_base:
            base-puck local geometry stacks.
        Q_c_base: continuity-constraint reduction matrix.
        base_indices: ``(n_all,)`` replica-to-base map.
        nfp, stellsym: symmetry tags.
        tf_loading: ``bn_quad``/``a_quad``/``a_taylor``.
        m_local_base: ``(n_base, nd, 3)`` local-frame magnetic moments (W3).
        Q_sym_local: ``(n_base, nd, 3, 3)`` local-frame quadrupoles (W3 a_taylor).

    Returns:
        ``(n_red,)`` reduced load vector ``f_red``.

    """
    n_base = int(local_pts_base.shape[0])
    nq = int(local_pts_base.shape[1])
    nd = int(local_K_base.shape[2])
    centers_all, quats_all, signs = _replicate_pucks_jax(
        centers_base, quats_base, nfp, stellsym
    )
    pts_stack, _K_stack, n_stack, _w_stack = _transform_local_to_all_replicas(
        centers_all,
        quats_all,
        local_pts_base,
        local_K_base,
        local_n_base,
        local_w_base,
        base_indices,
    )
    gammas_tf = jnp.asarray(g_tf)
    gammadash_tf = jnp.asarray(gd_tf)
    currents_tf = jnp.asarray(I_tf)
    all_flat_pts = pts_stack.reshape(-1, 3)
    all_flat_n = n_stack.reshape(-1, 3)
    if m_local_base is None:
        m_local_base = jnp.zeros((n_base, nd, 3), dtype=local_K_base.dtype)

    tfm = (tf_loading or "bn_quad").lower().replace("-", "_")
    if tfm == "a_taylor":
        Qsl = (
            Q_sym_local
            if Q_sym_local is not None
            else jnp.zeros((n_base, nd, 3, 3), dtype=local_K_base.dtype)
        )

        def f_row_taylor(ib: jnp.ndarray) -> jnp.ndarray:
            c = centers_base[ib]
            Rm = quats_base[ib]
            Ri = _rotation_matrix_from_quat_jax(Rm)
            B0 = _B_at_point_from_coil_set_pure(
                c, gammas_tf, gammadash_tf, currents_tf, -1, _EPS_BS
            )
            gB = _grad_B_at_point_from_coil_set_pure(
                c, gammas_tf, gammadash_tf, currents_tf, -1, _EPS_BS
            )
            Gsym = 0.5 * (gB + gB.T)
            m_g = jnp.einsum("ij,nj->ni", Ri, m_local_base[ib, :, :])
            Qb = Qsl[ib, :, :, :]
            Qg = jnp.einsum("ik,nkl,jl->nij", Ri, Qb, Ri)
            return jnp.einsum("na, a->n", m_g, B0) + jnp.einsum("nab, ab->n", Qg, Gsym)

        f = jax.vmap(f_row_taylor)(jnp.arange(n_base, dtype=jnp.int32)).reshape(-1)
    elif tfm == "a_quad":

        def f_row_aquad(ib: jnp.ndarray) -> jnp.ndarray:
            c = centers_base[ib]
            Rm = quats_base[ib]
            Ri = _rotation_matrix_from_quat_jax(Rm)
            b0 = _B_at_point_from_coil_set_pure(
                c, gammas_tf, gammadash_tf, currents_tf, -1, _EPS_BS
            )
            m_g = jnp.einsum("ij, nj->ni", Ri, m_local_base[ib, :, :])
            return jnp.einsum("na, a->n", m_g, b0)

        f = jax.vmap(f_row_aquad)(jnp.arange(n_base, dtype=jnp.int32)).reshape(-1)
    else:

        def Bn_at_q(q_idx: jnp.ndarray) -> jnp.ndarray:
            B = _B_at_point_from_coil_set_pure(
                all_flat_pts[q_idx],
                gammas_tf,
                gammadash_tf,
                currents_tf,
                -1,
                _EPS_BS,
            )
            return jnp.dot(B, all_flat_n[q_idx])

        Bn = jax.vmap(Bn_at_q)(jnp.arange(all_flat_pts.shape[0]))
        Bn_w = _fold_Bn_to_work(
            Bn,
            base_indices.astype(jnp.int32),
            signs.astype(Bn.dtype),
            n_base,
            nq,
        )
        f = -jnp.sum(local_phi_base * (local_w_base * Bn_w)[..., None], axis=1).reshape(
            -1
        )

    return Q_c_base.T @ f


def _post_solve_to_B(
    alpha: jnp.ndarray,
    Q_c_base: jnp.ndarray,
    base_indices: jnp.ndarray,
    signs: jnp.ndarray,
    K_stack: jnp.ndarray,
    pts_stack: jnp.ndarray,
    w_stack: jnp.ndarray,
    centers_all: jnp.ndarray,
    quats_all: jnp.ndarray,
    R_est_rep: jnp.ndarray,
    m_local_base: jnp.ndarray,
    pts_eval: jnp.ndarray,
    eval_chunk: int,
    bs_eval_far_kappa: float,
    n_all: int,
    nd: int,
) -> jnp.ndarray:
    """Map reduced ``alpha`` to ``B`` at evaluation points (Biot-Savart shell + W7 dipole tail).

    Wraps :math:`\\beta = Q_c \\alpha`, gather to replicas, sheet-current
    Biot-Savart, and the optional far-eval dipole blend (W7) used by
    :func:`_B_eval_reduced_free_dof_body_v2`.

    Args:
        alpha: ``(n_red,)`` reduced solve output.
        Q_c_base: continuity reduction matrix.
        base_indices, signs: replica fold maps.
        K_stack, pts_stack, w_stack: per-replica stacks (rotated to globals).
        centers_all, quats_all: all-replica centers / quats.
        R_est_rep: ``(n_all,)`` replica radii estimates for the W7 blend.
        m_local_base: ``(n_base, nd, 3)`` local-frame moments (W7).
        pts_eval: ``(n_eval, 3)`` evaluation points.
        eval_chunk: chunk width for the shell BS sum.
        bs_eval_far_kappa: ``0`` disables the W7 far-eval dipole blend.
        n_all, nd: shapes (Python ints).

    Returns:
        ``B`` of shape ``(n_eval, 3)``.

    """
    beta = Q_c_base @ alpha
    beta_all = _gather_beta_work_to_all(beta, base_indices, signs, nd)
    b_stack = beta_all.reshape(n_all, nd)
    k_at_quad = jnp.einsum("pqdi,pd->pqi", K_stack, b_stack)
    k_flat = k_at_quad.reshape(-1, 3)
    w_flat = w_stack.reshape(-1)
    all_flat_pts = pts_stack.reshape(-1, 3)
    b_shell = _shell_biot_savart_flat_chunked(
        k_flat, all_flat_pts, w_flat, pts_eval, int(eval_chunk)
    )
    if float(bs_eval_far_kappa) <= 0.0:
        return b_shell

    m_loc_on_rep = m_local_base[base_indices.astype(jnp.int32)]

    def _m_total_rep(r: jnp.ndarray) -> jnp.ndarray:
        rm = _rotation_matrix_from_quat_jax(quats_all[r])
        m_g = jnp.einsum("ij,dj->di", rm, m_loc_on_rep[r])
        return jnp.einsum("d, di->i", b_stack[r], m_g)

    m_rep = jax.vmap(_m_total_rep)(jnp.arange(n_all, dtype=jnp.int32))
    dmat = (
        jnp.linalg.norm(pts_eval[:, None, :] - centers_all[None, :, :], axis=-1) + 1e-20
    )
    cap = float(bs_eval_far_kappa) * (R_est_rep[None, :] + 1e-12)
    ratio = dmat / cap
    s_pt = jnp.min(jax.nn.sigmoid(10.0 * (ratio - 1.0)), axis=1)[:, None]

    def _dip(p: jnp.ndarray) -> jnp.ndarray:
        return magnetic_field_dipole_points(m_rep[p], pts_eval, centers_all[p])

    b_dip = jnp.sum(jax.vmap(_dip)(jnp.arange(n_all, dtype=jnp.int32)), axis=0)
    return s_pt * b_dip + (1.0 - s_pt) * b_shell


def _B_eval_full_body(
    local_pts_stack: jnp.ndarray,
    local_K_stack: jnp.ndarray,
    local_n_stack: jnp.ndarray,
    local_w_stack: jnp.ndarray,
    local_phi_stack: jnp.ndarray,
    Q_c: jnp.ndarray,
    centers_all: jnp.ndarray,
    quats_all: jnp.ndarray,
    g_tf: jnp.ndarray,
    gd_tf: jnp.ndarray,
    I_tf: jnp.ndarray,
    pts_eval: jnp.ndarray,
    delta_reg: float,
    eigenfloor_threshold: float,
    adaptive_self_reg: bool = False,
    full_L_band_size: int = 32,
    checkpoint_L_pairs: bool = False,
    eval_chunk: int = 0,
) -> jnp.ndarray:
    """Full forward: assemble :math:`L` in JAX, rim-continuity-projected
    eigenfloor solve, and Biot–Savart.

    The projector ``Q_c`` (built once in :meth:`PSCBulkArray._rebuild` via
    :func:`~simsopt.field.puck_basis.build_continuity_constraint`) is a
    dense block-diagonal matrix whose columns span the null space of the
    per-puck rim-continuity constraints (paper eq 55).  It depends only on
    per-puck :math:`(R, t, m_{fourier}, l_{zernike}, k_{chebyshev}, n_\\rho,
    n_\\phi, n_z, n_{phi\\_rim})` and is therefore constant w.r.t. the VJP
    variables (puck centers, quaternions, TF gammas / currents); the reduced
    eigenfloor solve ``Q_c^T L Q_c \\alpha = Q_c^T f`` gives a well-posed
    ideal-diamagnet solution in the continuous basis.

    Args:
        adaptive_self_reg: When ``True``, use the analytic flat-disc
            per-quadrature-cell regularization
            :math:`\\delta_i = (3\\pi^{3/2}/8)\\sqrt{w_i}` (see
            :data:`~simsopt.field.bulk_inductance._SELF_REG_COEFF`) inside
            the inline ``L``-assembly, matching
            :func:`~simsopt.field.bulk_inductance.shell_inductance_matrix_blockwise`.
            This must be ``True`` whenever the precomputed ``self._L_work``
            used to solve for ``beta`` also used the adaptive path;
            otherwise ``self.beta`` and the :math:`\\beta` implicit in this
            re-solve become inconsistent and the induced field reported by
            :meth:`PSCBulkArray.B_at_points` silently disagrees with
            :meth:`PSCBulkArray.get_shell_currents`.  Declared static so
            JAX caches a separate JIT per branch.
    """
    n_pucks = centers_all.shape[0]
    nd = local_K_stack.shape[2]
    n_dof_total = n_pucks * nd
    nq = local_pts_stack.shape[1]

    def transform_one(p):
        Rmat = _rotation_matrix_from_quat_jax(quats_all[p])
        pts_g = (Rmat @ local_pts_stack[p].T).T + centers_all[p]
        K_g = jnp.einsum("ij,qkj->qki", Rmat, local_K_stack[p])
        n_g = (Rmat @ local_n_stack[p].T).T
        return pts_g, K_g, n_g

    puck_data = jax.lax.map(transform_one, jnp.arange(n_pucks))
    pts_stack = puck_data[0]
    K_stack = puck_data[1]
    n_stack = puck_data[2]

    def L_block(pair):
        i, j = pair[0], pair[1]
        r = pts_stack[i, :, None, :] - pts_stack[j, None, :, :]
        if adaptive_self_reg:
            w_i = local_w_stack[i]
            w_j = local_w_stack[j]
            delta_i = _SELF_REG_COEFF * jnp.sqrt(w_i)
            delta_j = _SELF_REG_COEFF * jnp.sqrt(w_j)
            delta_pair = 0.5 * (delta_i[:, None] + delta_j[None, :])
            dist = jnp.sqrt(jnp.sum(r**2, axis=-1) + delta_pair**2)
        else:
            dist = jnp.sqrt(jnp.sum(r**2, axis=-1) + delta_reg**2)
        dot = jnp.einsum("iax,jbx->ijab", K_stack[i], K_stack[j])
        kernel = dot / dist[..., None, None]
        return MU0_OVER_4PI * jnp.einsum(
            "ijab,i,j->ab",
            kernel,
            local_w_stack[i],
            local_w_stack[j],
        )

    L_kern = jax.checkpoint(L_block) if checkpoint_L_pairs else L_block

    n_pucks_int = int(local_K_stack.shape[0])
    nb = int(full_L_band_size)
    if nb <= 0 or nb >= n_pucks_int:
        pair_i, pair_j = jnp.meshgrid(
            jnp.arange(n_pucks),
            jnp.arange(n_pucks),
            indexing="ij",
        )
        pairs = jnp.stack([pair_i.ravel(), pair_j.ravel()], axis=1)
        L_blocks_flat = jax.lax.map(L_kern, pairs)
        L_blocks = L_blocks_flat.reshape(n_pucks, n_pucks, nd, nd)
        L = L_blocks.transpose(0, 2, 1, 3).reshape(n_dof_total, n_dof_total)
    else:
        L = jnp.zeros((n_dof_total, n_dof_total), dtype=pts_stack.dtype)
        for start in range(0, n_pucks_int, nb):
            end = min(start + nb, n_pucks_int)

            def row_block(i):
                def col_block(j):
                    return L_kern(jnp.stack([jnp.asarray(i), jnp.asarray(j)]))

                return jax.lax.map(col_block, jnp.arange(n_pucks))

            band_blocks = jax.lax.map(row_block, jnp.arange(start, end))
            band_flat = band_blocks.transpose(0, 2, 1, 3).reshape(
                (end - start) * nd, n_dof_total
            )
            L = L.at[start * nd : end * nd, :].set(band_flat)

    gammas_tf = jnp.asarray(g_tf)
    gammadash_tf = jnp.asarray(gd_tf)
    currents_tf = jnp.asarray(I_tf)
    all_flat_pts = pts_stack.reshape(-1, 3)
    all_flat_n = n_stack.reshape(-1, 3)

    def Bn_at_q(q_idx):
        B = _B_at_point_from_coil_set_pure(
            all_flat_pts[q_idx],
            gammas_tf,
            gammadash_tf,
            currents_tf,
            -1,
            _EPS_BS,
        )
        return jnp.dot(B, all_flat_n[q_idx])

    Bn = jax.vmap(Bn_at_q)(jnp.arange(all_flat_pts.shape[0]))
    Bn_per = Bn.reshape(n_pucks, nq)
    # Sign convention: shell_loading_vector_pure returns
    # ``f_a = -integral Phi_a B_n^TF dS`` (see docstring).  The JIT
    # body used to drop the leading minus (bug: produced the wrong
    # induced-moment sign on free-puck-DOF B_at_points Taylor tests);
    # put it back here to match the fixed-DOF path.
    f = -jnp.sum(
        local_phi_stack * (local_w_stack * Bn_per)[..., None],
        axis=1,
    ).reshape(-1)

    L_r = Q_c.T @ L @ Q_c
    f_r = Q_c.T @ f
    alpha = shell_solve_eigenfloor_pure(
        L_r,
        f_r,
        threshold=eigenfloor_threshold,
        jitter=1e-10,
    )
    beta = Q_c @ alpha
    beta_per = beta.reshape(n_pucks, nd)
    K_at_quad = jnp.einsum("pqdi,pd->pqi", K_stack, beta_per)
    K_flat = K_at_quad.reshape(-1, 3)
    all_flat_w = local_w_stack.reshape(-1)

    return _shell_biot_savart_flat_chunked(
        K_flat, all_flat_pts, all_flat_w, pts_eval, int(eval_chunk)
    )


def _B_eval_reduced_free_dof_body(
    local_pts_base: jnp.ndarray,
    local_K_base: jnp.ndarray,
    local_n_base: jnp.ndarray,
    local_w_base: jnp.ndarray,
    local_phi_base: jnp.ndarray,
    Q_c_base: jnp.ndarray,
    centers_base: jnp.ndarray,
    quats_base: jnp.ndarray,
    g_tf: jnp.ndarray,
    gd_tf: jnp.ndarray,
    I_tf: jnp.ndarray,
    pts_eval: jnp.ndarray,
    L_red_eig_U: jnp.ndarray,
    L_red_eig_lam: jnp.ndarray,
    base_indices: jnp.ndarray,
    base_reps: jnp.ndarray,
    G_float: jnp.ndarray,
    nfp: int,
    stellsym: bool,
    delta_reg: float,
    eigenfloor_threshold: float,
    adaptive_self_reg: bool,
    checkpoint_L_pairs: bool = False,
    pair_row_chunk: int = 0,
    eval_chunk: int = 0,
    vjp_probe_mode: str = "full",
    solve_vjp_mode: str = "eigh",
    use_far_dipole_pair: bool = False,
    pair_far_kappa: float = 0.0,
    m_local_base: Optional[jnp.ndarray] = None,
    Q_sym_local: Optional[jnp.ndarray] = None,
    tf_loading: str = "bn_quad",
    solve_mode_reduced: str = "eigenfloor",
    bs_eval_far_kappa: float = 0.0,
    use_w1_envelope: bool = False,
    pair_replica_chunk: int = 0,
    near_pair_indices: Optional[jnp.ndarray] = None,
    far_pair_indices: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Symmetry-reduced free-puck-DoF forward: fold ``L`` to base replicas only.

    Pair count is ``n_base * n_all`` (factor ``1/G`` vs the full
    :func:`_B_eval_full_body`) matching
    :func:`~simsopt.field.bulk_inductance.shell_inductance_matrix_symmetric_reduced`.

    Args:
        pair_row_chunk: If positive and less than ``n_base``, build ``L``-rows in
            ``lax.scan`` blocks of this width to lower peak autodiff memory. ``0`` or
            ``>= n_base`` uses a single :func:`jax.vmap` over base rows.
        pair_replica_chunk: If positive and less than ``n_all``, fold the
            replica axis inside each base row via :func:`jax.lax.scan` in
            chunks of this width.  Caps the per-row pair-kernel batch at
            ``pair_replica_chunk * nq * nq * nd * nd`` instead of the
            full ``n_all * nq * nq * nd * nd``.  ``0`` or ``>= n_all``
            keeps the legacy monolithic ``vmap``.
    """
    n_base = int(local_pts_base.shape[0])
    nq = int(local_pts_base.shape[1])
    nd = int(local_K_base.shape[2])
    n_all = int(base_indices.shape[0])

    centers_all, quats_all, signs = _replicate_pucks_jax(
        centers_base, quats_base, nfp, stellsym
    )
    pts_stack, K_stack, n_stack, w_stack = _transform_local_to_all_replicas(
        centers_all,
        quats_all,
        local_pts_base,
        local_K_base,
        local_n_base,
        local_w_base,
        base_indices,
    )

    r_rel = pts_stack - centers_all[:, None, :]
    R_est_rep = jnp.max(jnp.linalg.norm(r_rel, axis=-1), axis=1)
    if m_local_base is None:
        m_local_base = jnp.zeros((n_base, nd, 3), dtype=K_stack.dtype)

    def l_block_pair(i_rep: jnp.ndarray, j_rep: jnp.ndarray) -> jnp.ndarray:
        return _pair_inductance_block(
            pts_stack,
            K_stack,
            w_stack,
            i_rep,
            j_rep,
            float(delta_reg),
            bool(adaptive_self_reg),
        )

    l_fn: Any = l_block_pair
    if checkpoint_L_pairs and not use_far_dipole_pair:
        l_fn = jax.checkpoint(l_block_pair)

    rc = int(pair_replica_chunk)
    block_dtype = K_stack.dtype

    if (
        use_far_dipole_pair
        and pair_far_kappa > 0.0
        and near_pair_indices is not None
        and far_pair_indices is not None
    ):
        l_rows0 = jnp.zeros((n_base, n_base, nd, nd), dtype=block_dtype)

        def _add_dense_pair(carry: jnp.ndarray, pair: jnp.ndarray) -> Tuple[jnp.ndarray, None]:
            ib = pair[0].astype(jnp.int32)
            jr = pair[1].astype(jnp.int32)
            i_rep = base_reps[ib]
            bj = base_indices[jr]
            blk = G_float * signs[jr] * l_fn(i_rep, jr)
            return carry.at[ib, bj, :, :].add(blk), None

        self_pairs = jnp.stack(
            [jnp.arange(n_base, dtype=jnp.int32), base_reps.astype(jnp.int32)], axis=1
        )
        l_rows, _ = jax.lax.scan(_add_dense_pair, l_rows0, self_pairs)
        l_rows, _ = jax.lax.scan(
            _add_dense_pair, l_rows, near_pair_indices.astype(jnp.int32)
        )

        far_idx = far_pair_indices.astype(jnp.int32)

        def _far_pair_block(pair: jnp.ndarray) -> jnp.ndarray:
            ib = pair[0].astype(jnp.int32)
            jr = pair[1].astype(jnp.int32)
            i_rep = base_reps[ib]
            bj = base_indices[jr]
            Rij = centers_all[jr] - centers_all[i_rep]
            Ri = _rotation_matrix_from_quat_jax(quats_all[i_rep])
            Rj = _rotation_matrix_from_quat_jax(quats_all[jr])
            m_i = jnp.einsum("ij,nj->ni", Ri, m_local_base[ib, :, :])
            m_j = jnp.einsum("ij,nj->ni", Rj, m_local_base[bj, :, :])
            return G_float * signs[jr] * pair_inductance_multipole(m_i, m_j, Rij, order=1)

        if far_idx.shape[0] > 0:
            far_blocks = jax.vmap(_far_pair_block)(far_idx)
            l_rows = l_rows.at[far_idx[:, 0], base_indices[far_idx[:, 1]], :, :].add(
                far_blocks
            )
    else:

        def one_row_f(ib: jnp.ndarray) -> jnp.ndarray:
            i_rep = base_reps[ib.astype(jnp.int32)]

            def scaled_block(jr: jnp.ndarray) -> jnp.ndarray:
                blk = l_fn(i_rep, jr)
                return G_float * signs[jr] * blk

            return _row_segment_sum_replica_chunked(
                scaled_block, base_indices, n_all, n_base, nd, rc, block_dtype
            )

        n_chunk = int(pair_row_chunk)
        if n_chunk > 0 and n_chunk < n_base:

            def _scan_one_chunk(_: None, ic: jnp.ndarray) -> Tuple[None, jnp.ndarray]:
                start = ic * n_chunk
                idx = start + jnp.arange(n_chunk, dtype=jnp.int32)
                valid = idx < n_base
                idx_safe = jnp.where(valid, idx, 0)
                rows_b = jax.vmap(one_row_f)(idx_safe)
                rows_b = rows_b * valid[:, None, None, None].astype(rows_b.dtype)
                return None, rows_b

            n_c = (n_base + n_chunk - 1) // n_chunk
            _, row_blocks = jax.lax.scan(
                _scan_one_chunk, None, jnp.arange(n_c, dtype=jnp.int32)
            )
            l_rows = row_blocks.reshape(n_c * n_chunk, n_base, nd, nd)[:n_base, ...]
        else:
            l_rows = jax.vmap(one_row_f)(jnp.arange(n_base, dtype=jnp.int32))
    l_base = l_rows.transpose(0, 2, 1, 3).reshape(n_base * nd, n_base * nd)
    l_base = 0.5 * (l_base + l_base.T)
    if vjp_probe_mode == "stop_l":
        l_base = jax.lax.stop_gradient(l_base)

    gammas_tf = jnp.asarray(g_tf)
    gammadash_tf = jnp.asarray(gd_tf)
    currents_tf = jnp.asarray(I_tf)
    all_flat_pts = pts_stack.reshape(-1, 3)
    all_flat_n = n_stack.reshape(-1, 3)

    tfm = (tf_loading or "bn_quad").lower().replace("-", "_")
    if tfm == "a_taylor":
        Qsl = (
            Q_sym_local
            if Q_sym_local is not None
            else jnp.zeros((n_base, nd, 3, 3), dtype=K_stack.dtype)
        )

        def f_row_taylor(ib: jnp.ndarray) -> jnp.ndarray:
            c = centers_base[ib]
            Rm = quats_base[ib]
            Ri = _rotation_matrix_from_quat_jax(Rm)
            B0 = _B_at_point_from_coil_set_pure(
                c, gammas_tf, gammadash_tf, currents_tf, -1, _EPS_BS
            )
            gB = _grad_B_at_point_from_coil_set_pure(
                c, gammas_tf, gammadash_tf, currents_tf, -1, _EPS_BS
            )
            Gsym = 0.5 * (gB + gB.T)
            m_g = jnp.einsum("ij,nj->ni", Ri, m_local_base[ib, :, :])
            Qb = Qsl[ib, :, :, :]
            Qg = jnp.einsum("ik,nkl,jl->nij", Ri, Qb, Ri)
            return jnp.einsum("na, a->n", m_g, B0) + jnp.einsum("nab, ab->n", Qg, Gsym)

        f = jax.vmap(f_row_taylor)(jnp.arange(n_base, dtype=jnp.int32)).reshape(-1)
    elif tfm == "a_quad":
        # Leading closed-shell / Galerkin identity: f_a ≈ m_a · B_TF(c) (dipole only).
        def f_row_aquad(ib: jnp.ndarray) -> jnp.ndarray:
            c = centers_base[ib]
            Rm = quats_base[ib]
            Ri = _rotation_matrix_from_quat_jax(Rm)
            b0 = _B_at_point_from_coil_set_pure(
                c, gammas_tf, gammadash_tf, currents_tf, -1, _EPS_BS
            )
            m_g = jnp.einsum("ij, nj->ni", Ri, m_local_base[ib, :, :])
            return jnp.einsum("na, a->n", m_g, b0)

        f = jax.vmap(f_row_aquad)(jnp.arange(n_base, dtype=jnp.int32)).reshape(-1)
    else:

        def Bn_at_q(q_idx: jnp.ndarray) -> jnp.ndarray:
            B = _B_at_point_from_coil_set_pure(
                all_flat_pts[q_idx],
                gammas_tf,
                gammadash_tf,
                currents_tf,
                -1,
                _EPS_BS,
            )
            return jnp.dot(B, all_flat_n[q_idx])

        Bn = jax.vmap(Bn_at_q)(jnp.arange(all_flat_pts.shape[0]))
        if vjp_probe_mode == "stop_bn":
            Bn = jax.lax.stop_gradient(Bn)
        Bn_w = _fold_Bn_to_work(
            Bn,
            base_indices.astype(jnp.int32),
            signs.astype(Bn.dtype),
            n_base,
            nq,
        )
        f = -jnp.sum(local_phi_base * (local_w_base * Bn_w)[..., None], axis=1).reshape(
            -1
        )

    l_r = Q_c_base.T @ l_base @ Q_c_base
    f_r = Q_c_base.T @ f
    solve_mode_norm = str(solve_mode_reduced).lower()
    eigs = solve_mode_norm == "eigk"
    if eigs and L_red_eig_U.shape[0] == f_r.shape[0] and L_red_eig_U.shape[0] > 0:
        alpha = _eigK_solve_with_correct_vjp(
            l_r, L_red_eig_U, L_red_eig_lam, f_r, eigenfloor_threshold
        )
    elif eigs:
        Lreg, _, _ = shell_eigendecomposition_floored(
            l_r, threshold=eigenfloor_threshold, jitter=1e-10
        )
        alpha = jnp.linalg.solve(Lreg, f_r)
    elif solve_vjp_mode == "implicit" or use_w1_envelope:
        alpha = _solve_eigenfloor_implicit_vjp(
            l_r,
            f_r,
            eigenfloor_threshold,
        )
    else:
        alpha = shell_solve_eigenfloor_pure(
            l_r,
            f_r,
            threshold=eigenfloor_threshold,
            jitter=1e-10,
        )
    beta = Q_c_base @ alpha
    if vjp_probe_mode in ("stop_solve", "shell_only"):
        beta = jax.lax.stop_gradient(beta)
    beta_all = _gather_beta_work_to_all(beta, base_indices, signs, nd)
    b_stack = beta_all.reshape(n_all, nd)
    K_for_shell = (
        jax.lax.stop_gradient(K_stack) if vjp_probe_mode == "beta_only" else K_stack
    )
    k_at_quad = jnp.einsum("pqdi,pd->pqi", K_for_shell, b_stack)
    k_flat = k_at_quad.reshape(-1, 3)
    source_pts = (
        jax.lax.stop_gradient(all_flat_pts)
        if vjp_probe_mode == "beta_only"
        else all_flat_pts
    )
    w_source = (
        jax.lax.stop_gradient(w_stack) if vjp_probe_mode == "beta_only" else w_stack
    )
    w_flat = w_source.reshape(-1)

    b_shell = _shell_biot_savart_flat_chunked(
        k_flat, source_pts, w_flat, pts_eval, int(eval_chunk)
    )
    if float(bs_eval_far_kappa) <= 0.0:
        return b_shell
    m_loc_on_rep = m_local_base[base_indices.astype(jnp.int32)]

    def _m_total_rep(r: jnp.ndarray) -> jnp.ndarray:
        rm = _rotation_matrix_from_quat_jax(quats_all[r])
        m_g = jnp.einsum("ij,dj->di", rm, m_loc_on_rep[r])
        return jnp.einsum("d, di->i", b_stack[r], m_g)

    m_rep = jax.vmap(_m_total_rep)(jnp.arange(n_all, dtype=jnp.int32))
    dmat = (
        jnp.linalg.norm(pts_eval[:, None, :] - centers_all[None, :, :], axis=-1) + 1e-20
    )
    cap = float(bs_eval_far_kappa) * (R_est_rep[None, :] + 1e-12)
    ratio = dmat / cap
    s_pt = jnp.min(jax.nn.sigmoid(10.0 * (ratio - 1.0)), axis=1)[:, None]

    def _dip(p: jnp.ndarray) -> jnp.ndarray:
        return magnetic_field_dipole_points(m_rep[p], pts_eval, centers_all[p])

    b_dip = jnp.sum(jax.vmap(_dip)(jnp.arange(n_all, dtype=jnp.int32)), axis=0)
    return s_pt * b_dip + (1.0 - s_pt) * b_shell


def _B_eval_reduced_free_dof_body_v2(
    local_pts_base: jnp.ndarray,
    local_K_base: jnp.ndarray,
    local_n_base: jnp.ndarray,
    local_w_base: jnp.ndarray,
    local_phi_base: jnp.ndarray,
    Q_c_base: jnp.ndarray,
    centers_base: jnp.ndarray,
    quats_base: jnp.ndarray,
    g_tf: jnp.ndarray,
    gd_tf: jnp.ndarray,
    I_tf: jnp.ndarray,
    pts_eval: jnp.ndarray,
    L_red_eig_U: jnp.ndarray,
    L_red_eig_lam: jnp.ndarray,
    base_indices: jnp.ndarray,
    base_reps: jnp.ndarray,
    G_float: jnp.ndarray,
    nfp: int,
    stellsym: bool,
    delta_reg: float,
    eigenfloor_threshold: float,
    adaptive_self_reg: bool,
    checkpoint_L_pairs: bool = False,
    pair_row_chunk: int = 0,
    eval_chunk: int = 0,
    use_far_dipole_pair: bool = False,
    pair_far_kappa: float = 0.0,
    m_local_base: Optional[jnp.ndarray] = None,
    Q_sym_local: Optional[jnp.ndarray] = None,
    tf_loading: str = "bn_quad",
    solve_mode_reduced: str = "eigenfloor",
    bs_eval_far_kappa: float = 0.0,
    pair_replica_chunk: int = 0,
    near_pair_indices: Optional[jnp.ndarray] = None,
    far_pair_indices: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """W1 envelope variant: ``alpha`` solve via ``jax.custom_vjp`` envelope theorem.

    Same forward semantics as :func:`_B_eval_reduced_free_dof_body` (when
    ``vjp_probe_mode='full'``), but the ``alpha = L_r^{-1} f_r`` step is wrapped
    in a custom ``custom_vjp`` whose backward differentiates the *scalar*
    ``E_adj(theta) = lam.T @ f_r(theta) - lam.T @ L_r(theta) @ alpha`` with
    ``lam = L_r^{-1} ct_alpha``.  This avoids JAX retracing through the linear
    solve in the backward pass and enables a cleaner adjoint when the eigK
    cache is used (forward path can pick the cached basis without poisoning the
    backward).  Drops the diagnostic ``vjp_probe_mode`` / ``solve_vjp_mode``
    knobs from v1; gate selection v1 vs v2 via :func:`_psc_w1_envelope_env`.

    ``pair_replica_chunk`` is forwarded to :func:`_assemble_L_red_inner` for
    in-row replica-axis chunking of the pair-inductance kernel; see
    :func:`_B_eval_reduced_free_dof_body` for semantics.
    """
    n_base = int(local_pts_base.shape[0])
    nd = int(local_K_base.shape[2])
    n_all = int(base_indices.shape[0])

    centers_all, quats_all, signs = _replicate_pucks_jax(
        centers_base, quats_base, nfp, stellsym
    )
    pts_stack, K_stack, _n_stack, w_stack = _transform_local_to_all_replicas(
        centers_all,
        quats_all,
        local_pts_base,
        local_K_base,
        local_n_base,
        local_w_base,
        base_indices,
    )
    r_rel = pts_stack - centers_all[:, None, :]
    R_est_rep = jnp.max(jnp.linalg.norm(r_rel, axis=-1), axis=1)
    if m_local_base is None:
        m_local_base = jnp.zeros((n_base, nd, 3), dtype=K_stack.dtype)

    def L_assemble(c_b: jnp.ndarray, q_b: jnp.ndarray) -> jnp.ndarray:
        return _assemble_L_red_inner(
            c_b,
            q_b,
            local_pts_base,
            local_K_base,
            local_n_base,
            local_w_base,
            Q_c_base,
            base_indices,
            base_reps,
            G_float,
            nfp,
            stellsym,
            delta_reg,
            adaptive_self_reg,
            checkpoint_L_pairs=checkpoint_L_pairs,
            pair_row_chunk=pair_row_chunk,
            use_far_dipole_pair=use_far_dipole_pair,
            pair_far_kappa=pair_far_kappa,
            m_local_base=m_local_base,
            pair_replica_chunk=pair_replica_chunk,
            near_pair_indices=near_pair_indices,
            far_pair_indices=far_pair_indices,
        )

    def f_assemble(
        c_b: jnp.ndarray,
        q_b: jnp.ndarray,
        g: jnp.ndarray,
        gd: jnp.ndarray,
        I: jnp.ndarray,
    ) -> jnp.ndarray:
        return _assemble_f_red_inner(
            c_b,
            q_b,
            g,
            gd,
            I,
            local_pts_base,
            local_K_base,
            local_n_base,
            local_w_base,
            local_phi_base,
            Q_c_base,
            base_indices,
            nfp,
            stellsym,
            tf_loading,
            m_local_base=m_local_base,
            Q_sym_local=Q_sym_local,
        )

    eigk_mode = str(solve_mode_reduced).lower() == "eigk"
    cache_n = int(L_red_eig_U.shape[0])

    @jax.custom_vjp
    def _alpha_from_q(
        c_b: jnp.ndarray,
        q_b: jnp.ndarray,
        g: jnp.ndarray,
        gd: jnp.ndarray,
        I: jnp.ndarray,
    ) -> jnp.ndarray:
        L_r = L_assemble(c_b, q_b)
        f_r = f_assemble(c_b, q_b, g, gd, I)
        if eigk_mode and cache_n > 0 and cache_n == f_r.shape[0]:
            return shell_solve_eigK_pure(L_red_eig_U, L_red_eig_lam, f_r)
        return _solve_eigenfloor_implicit_vjp(L_r, f_r, eigenfloor_threshold)

    def _alpha_from_q_fwd(
        c_b: jnp.ndarray,
        q_b: jnp.ndarray,
        g: jnp.ndarray,
        gd: jnp.ndarray,
        I: jnp.ndarray,
    ) -> Tuple[
        jnp.ndarray,
        Tuple[
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
        ],
    ]:
        L_r = L_assemble(c_b, q_b)
        f_r = f_assemble(c_b, q_b, g, gd, I)
        if eigk_mode and cache_n > 0 and cache_n == f_r.shape[0]:
            alpha = shell_solve_eigK_pure(L_red_eig_U, L_red_eig_lam, f_r)
        else:
            alpha = _solve_eigenfloor_implicit_vjp(L_r, f_r, eigenfloor_threshold)
        return alpha, (c_b, q_b, g, gd, I, L_r, alpha)

    def _alpha_from_q_bwd(
        residuals: Tuple[
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
        ],
        ct_alpha: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        c_b, q_b, g, gd, I, L_r, alpha = residuals
        L_reg = _eigenfloor_regularized_matrix(
            L_r, threshold=eigenfloor_threshold, jitter=1e-10
        )
        lam = jnp.linalg.solve(L_reg, ct_alpha)

        def E_adj(
            c: jnp.ndarray,
            q: jnp.ndarray,
            gg: jnp.ndarray,
            ggd: jnp.ndarray,
            II: jnp.ndarray,
        ) -> jnp.ndarray:
            Lc = L_assemble(c, q)
            fc = f_assemble(c, q, gg, ggd, II)
            return jnp.dot(lam, fc) - jnp.dot(lam, Lc @ alpha)

        return jax.grad(E_adj, argnums=(0, 1, 2, 3, 4))(c_b, q_b, g, gd, I)

    _alpha_from_q.defvjp(_alpha_from_q_fwd, _alpha_from_q_bwd)

    alpha = _alpha_from_q(centers_base, quats_base, g_tf, gd_tf, I_tf)

    return _post_solve_to_B(
        alpha,
        Q_c_base,
        base_indices,
        signs,
        K_stack,
        pts_stack,
        w_stack,
        centers_all,
        quats_all,
        R_est_rep,
        m_local_base,
        pts_eval,
        eval_chunk,
        bs_eval_far_kappa,
        n_all,
        nd,
    )


def _B_eval_reduced_free_dof_dispatch(
    local_pts_base: jnp.ndarray,
    local_K_base: jnp.ndarray,
    local_n_base: jnp.ndarray,
    local_w_base: jnp.ndarray,
    local_phi_base: jnp.ndarray,
    Q_c_base: jnp.ndarray,
    centers_base: jnp.ndarray,
    quats_base: jnp.ndarray,
    g_tf: jnp.ndarray,
    gd_tf: jnp.ndarray,
    I_tf: jnp.ndarray,
    pts_eval: jnp.ndarray,
    L_red_eig_U: jnp.ndarray,
    L_red_eig_lam: jnp.ndarray,
    base_indices: jnp.ndarray,
    base_reps: jnp.ndarray,
    G_float: jnp.ndarray,
    nfp: int,
    stellsym: bool,
    delta_reg: float,
    eigenfloor_threshold: float,
    adaptive_self_reg: bool,
    checkpoint_L_pairs: bool = False,
    pair_row_chunk: int = 0,
    eval_chunk: int = 0,
    vjp_probe_mode: str = "full",
    solve_vjp_mode: str = "eigh",
    use_far_dipole_pair: bool = False,
    pair_far_kappa: float = 0.0,
    m_local_base: Optional[jnp.ndarray] = None,
    Q_sym_local: Optional[jnp.ndarray] = None,
    tf_loading: str = "bn_quad",
    solve_mode_reduced: str = "eigenfloor",
    bs_eval_far_kappa: float = 0.0,
    use_w1_envelope: bool = False,
    pair_replica_chunk: int = 0,
    near_pair_indices: Optional[jnp.ndarray] = None,
    far_pair_indices: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Static dispatch between v1 and v2 reduced free-DOF bodies.

    ``use_w1_envelope=True`` selects :func:`_B_eval_reduced_free_dof_body_v2`
    (envelope-theorem ``custom_vjp`` for ``alpha``).  ``False`` keeps the
    legacy ``_B_eval_reduced_free_dof_body`` path with ``vjp_probe_mode`` /
    ``solve_vjp_mode`` diagnostics.  Branch is on a JIT-static argument so each
    branch compiles separately.

    ``pair_replica_chunk`` is forwarded to both v1 and v2 bodies for
    in-row replica-axis chunking of the pair-inductance kernel.
    """
    if use_w1_envelope:
        return _B_eval_reduced_free_dof_body_v2(
            local_pts_base,
            local_K_base,
            local_n_base,
            local_w_base,
            local_phi_base,
            Q_c_base,
            centers_base,
            quats_base,
            g_tf,
            gd_tf,
            I_tf,
            pts_eval,
            L_red_eig_U,
            L_red_eig_lam,
            base_indices,
            base_reps,
            G_float,
            nfp,
            stellsym,
            delta_reg,
            eigenfloor_threshold,
            adaptive_self_reg,
            checkpoint_L_pairs=checkpoint_L_pairs,
            pair_row_chunk=pair_row_chunk,
            eval_chunk=eval_chunk,
            use_far_dipole_pair=use_far_dipole_pair,
            pair_far_kappa=pair_far_kappa,
            m_local_base=m_local_base,
            Q_sym_local=Q_sym_local,
            tf_loading=tf_loading,
            solve_mode_reduced=solve_mode_reduced,
            bs_eval_far_kappa=bs_eval_far_kappa,
            pair_replica_chunk=pair_replica_chunk,
            near_pair_indices=near_pair_indices,
            far_pair_indices=far_pair_indices,
        )
    return _B_eval_reduced_free_dof_body(
        local_pts_base,
        local_K_base,
        local_n_base,
        local_w_base,
        local_phi_base,
        Q_c_base,
        centers_base,
        quats_base,
        g_tf,
        gd_tf,
        I_tf,
        pts_eval,
        L_red_eig_U,
        L_red_eig_lam,
        base_indices,
        base_reps,
        G_float,
        nfp,
        stellsym,
        delta_reg,
        eigenfloor_threshold,
        adaptive_self_reg,
        checkpoint_L_pairs=checkpoint_L_pairs,
        pair_row_chunk=pair_row_chunk,
        eval_chunk=eval_chunk,
        vjp_probe_mode=vjp_probe_mode,
        solve_vjp_mode=solve_vjp_mode,
        use_far_dipole_pair=use_far_dipole_pair,
        pair_far_kappa=pair_far_kappa,
        m_local_base=m_local_base,
        Q_sym_local=Q_sym_local,
        tf_loading=tf_loading,
        solve_mode_reduced=solve_mode_reduced,
        bs_eval_far_kappa=bs_eval_far_kappa,
        use_w1_envelope=False,
        pair_replica_chunk=pair_replica_chunk,
        near_pair_indices=near_pair_indices,
        far_pair_indices=far_pair_indices,
    )


_B_eval_full_jitted = jax.jit(
    _B_eval_full_body,
    static_argnames=(
        "delta_reg",
        "eigenfloor_threshold",
        "adaptive_self_reg",
        "full_L_band_size",
        "checkpoint_L_pairs",
        "eval_chunk",
    ),
)
_B_eval_reduced_free_dof_jitted = jax.jit(
    _B_eval_reduced_free_dof_dispatch,
    static_argnames=(
        "nfp",
        "stellsym",
        "delta_reg",
        "eigenfloor_threshold",
        "adaptive_self_reg",
        "checkpoint_L_pairs",
        "pair_row_chunk",
        "eval_chunk",
        "vjp_probe_mode",
        "solve_vjp_mode",
        "use_far_dipole_pair",
        "pair_far_kappa",
        "tf_loading",
        "solve_mode_reduced",
        "bs_eval_far_kappa",
        "use_w1_envelope",
        "pair_replica_chunk",
    ),
)


def _vjp_tf_run(
    Lr_chol: jnp.ndarray,
    Qm: jnp.ndarray,
    quad_pts: jnp.ndarray,
    quad_n: jnp.ndarray,
    phi_work_stack: jnp.ndarray,
    w_work_stack: jnp.ndarray,
    K_all_stack: jnp.ndarray,
    w_quad_all: jnp.ndarray,
    base_indices: jnp.ndarray,
    signs: jnp.ndarray,
    g_tf: jnp.ndarray,
    gd_tf: jnp.ndarray,
    I_tf: jnp.ndarray,
    pts_eval: jnp.ndarray,
    v_B: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """TF-only VJP; uses non-JIT forward body so AD is exact inside ``jax.jit``.

    Takes the Cholesky factor of ``L_red + jitter I`` so the big
    ``L_work`` need not be resident on device.  ``signs`` is the
    per-replica parity passed through the signed fold/gather operators.
    """

    def fwd(g, gd, I):
        return _B_eval_from_tf_body(
            Lr_chol,
            Qm,
            quad_pts,
            quad_n,
            phi_work_stack,
            w_work_stack,
            K_all_stack,
            w_quad_all,
            base_indices,
            signs,
            g,
            gd,
            I,
            pts_eval,
        )

    _, vjp_fn = vjp(fwd, g_tf, gd_tf, I_tf)
    return vjp_fn(v_B)


_vjp_tf_jitted = jax.jit(_vjp_tf_run)


def _vjp_full_run(
    local_pts_stack: jnp.ndarray,
    local_K_stack: jnp.ndarray,
    local_n_stack: jnp.ndarray,
    local_w_stack: jnp.ndarray,
    local_phi_stack: jnp.ndarray,
    Q_c: jnp.ndarray,
    centers_all: jnp.ndarray,
    quats_all: jnp.ndarray,
    g_tf: jnp.ndarray,
    gd_tf: jnp.ndarray,
    I_tf: jnp.ndarray,
    pts_eval: jnp.ndarray,
    v_B: jnp.ndarray,
    delta_reg: float,
    eigenfloor_threshold: float,
    adaptive_self_reg: bool = False,
    full_L_band_size: int = 32,
    checkpoint_L_pairs: bool = False,
    eval_chunk: int = 0,
) -> Tuple[
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
]:
    """Full VJP w.r.t. puck centers, quaternions, and TF arrays.

    ``Q_c`` (rim-continuity projector) is constant w.r.t. VJP variables.

    Args:
        adaptive_self_reg: Same meaning as in :func:`_B_eval_full_body`;
            must match ``PSCBulkArray.adaptive_self_reg`` so that the
            forward assembly used inside the reverse-mode pass is
            consistent with the one used by ``_rebuild``.
    """

    def fwd(c_all, q_all, g, gd, I):
        return _B_eval_full_body(
            local_pts_stack,
            local_K_stack,
            local_n_stack,
            local_w_stack,
            local_phi_stack,
            Q_c,
            c_all,
            q_all,
            g,
            gd,
            I,
            pts_eval,
            delta_reg,
            eigenfloor_threshold,
            adaptive_self_reg,
            full_L_band_size,
            checkpoint_L_pairs,
            eval_chunk,
        )

    fwd_run = jax.checkpoint(fwd, prevent_cse=False) if checkpoint_L_pairs else fwd
    _, vjp_fn = vjp(fwd_run, centers_all, quats_all, g_tf, gd_tf, I_tf)
    return vjp_fn(v_B)


_vjp_full_jitted = jax.jit(
    _vjp_full_run,
    static_argnames=(
        "delta_reg",
        "eigenfloor_threshold",
        "adaptive_self_reg",
        "full_L_band_size",
        "checkpoint_L_pairs",
        "eval_chunk",
    ),
)


def _vjp_reduced_free_dof_run(
    local_pts_base: jnp.ndarray,
    local_K_base: jnp.ndarray,
    local_n_base: jnp.ndarray,
    local_w_base: jnp.ndarray,
    local_phi_base: jnp.ndarray,
    Q_c_base: jnp.ndarray,
    centers_base: jnp.ndarray,
    quats_base: jnp.ndarray,
    g_tf: jnp.ndarray,
    gd_tf: jnp.ndarray,
    I_tf: jnp.ndarray,
    pts_eval: jnp.ndarray,
    L_red_eig_U: jnp.ndarray,
    L_red_eig_lam: jnp.ndarray,
    v_B: jnp.ndarray,
    base_indices: jnp.ndarray,
    base_reps: jnp.ndarray,
    G_float: jnp.ndarray,
    nfp: int,
    stellsym: bool,
    delta_reg: float,
    eigenfloor_threshold: float,
    adaptive_self_reg: bool,
    checkpoint_L_pairs: bool,
    pair_row_chunk: int,
    eval_chunk: int,
    vjp_probe_mode: str,
    solve_vjp_mode: str,
    m_local_base: jnp.ndarray,
    Q_sym_local: jnp.ndarray,
    use_far_dipole_pair: bool,
    pair_far_kappa: float,
    tf_loading: str,
    solve_mode_reduced: str,
    bs_eval_far_kappa: float,
    use_w1_envelope: bool = False,
    pair_replica_chunk: int = 0,
    near_pair_indices: Optional[jnp.ndarray] = None,
    far_pair_indices: Optional[jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """VJP for :func:`_B_eval_reduced_free_dof_dispatch` w.r.t. base centers, quats, TF."""

    def fwd(c_b, q_b, g, gd, I):
        return _B_eval_reduced_free_dof_dispatch(
            local_pts_base,
            local_K_base,
            local_n_base,
            local_w_base,
            local_phi_base,
            Q_c_base,
            c_b,
            q_b,
            g,
            gd,
            I,
            pts_eval,
            L_red_eig_U,
            L_red_eig_lam,
            base_indices,
            base_reps,
            G_float,
            nfp,
            stellsym,
            delta_reg,
            eigenfloor_threshold,
            adaptive_self_reg,
            checkpoint_L_pairs,
            pair_row_chunk,
            eval_chunk,
            vjp_probe_mode,
            solve_vjp_mode,
            use_far_dipole_pair,
            pair_far_kappa,
            m_local_base,
            Q_sym_local,
            tf_loading,
            solve_mode_reduced,
            bs_eval_far_kappa,
            use_w1_envelope,
            pair_replica_chunk,
            near_pair_indices,
            far_pair_indices,
        )

    fwd_run = jax.checkpoint(fwd, prevent_cse=False) if checkpoint_L_pairs else fwd
    _, vjp_fn = vjp(fwd_run, centers_base, quats_base, g_tf, gd_tf, I_tf)
    return vjp_fn(v_B)


_vjp_reduced_free_dof_jitted = jax.jit(
    _vjp_reduced_free_dof_run,
    static_argnames=(
        "nfp",
        "stellsym",
        "delta_reg",
        "eigenfloor_threshold",
        "adaptive_self_reg",
        "checkpoint_L_pairs",
        "pair_row_chunk",
        "eval_chunk",
        "vjp_probe_mode",
        "solve_vjp_mode",
        "use_far_dipole_pair",
        "pair_far_kappa",
        "tf_loading",
        "solve_mode_reduced",
        "bs_eval_far_kappa",
        "use_w1_envelope",
        "pair_replica_chunk",
    ),
)


def _vjp_reduced_free_quat_only_run(
    local_pts_base: jnp.ndarray,
    local_K_base: jnp.ndarray,
    local_n_base: jnp.ndarray,
    local_w_base: jnp.ndarray,
    local_phi_base: jnp.ndarray,
    Q_c_base: jnp.ndarray,
    centers_base: jnp.ndarray,
    quats_base: jnp.ndarray,
    g_tf: jnp.ndarray,
    gd_tf: jnp.ndarray,
    I_tf: jnp.ndarray,
    pts_eval: jnp.ndarray,
    L_red_eig_U: jnp.ndarray,
    L_red_eig_lam: jnp.ndarray,
    v_B: jnp.ndarray,
    base_indices: jnp.ndarray,
    base_reps: jnp.ndarray,
    G_float: jnp.ndarray,
    nfp: int,
    stellsym: bool,
    delta_reg: float,
    eigenfloor_threshold: float,
    adaptive_self_reg: bool,
    checkpoint_L_pairs: bool,
    pair_row_chunk: int,
    eval_chunk: int,
    vjp_probe_mode: str,
    solve_vjp_mode: str,
    m_local_base: jnp.ndarray,
    Q_sym_local: jnp.ndarray,
    use_far_dipole_pair: bool,
    pair_far_kappa: float,
    tf_loading: str,
    solve_mode_reduced: str,
    bs_eval_far_kappa: float,
    use_w1_envelope: bool = False,
    pair_replica_chunk: int = 0,
    near_pair_indices: Optional[jnp.ndarray] = None,
    far_pair_indices: Optional[jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Reduced VJP when only puck quaternion DOFs are active."""

    def fwd(q_b, g, gd, I):
        return _B_eval_reduced_free_dof_dispatch(
            local_pts_base,
            local_K_base,
            local_n_base,
            local_w_base,
            local_phi_base,
            Q_c_base,
            centers_base,
            q_b,
            g,
            gd,
            I,
            pts_eval,
            L_red_eig_U,
            L_red_eig_lam,
            base_indices,
            base_reps,
            G_float,
            nfp,
            stellsym,
            delta_reg,
            eigenfloor_threshold,
            adaptive_self_reg,
            checkpoint_L_pairs,
            pair_row_chunk,
            eval_chunk,
            vjp_probe_mode,
            solve_vjp_mode,
            use_far_dipole_pair,
            pair_far_kappa,
            m_local_base,
            Q_sym_local,
            tf_loading,
            solve_mode_reduced,
            bs_eval_far_kappa,
            use_w1_envelope,
            pair_replica_chunk,
            near_pair_indices,
            far_pair_indices,
        )

    fwd_run = jax.checkpoint(fwd, prevent_cse=False) if checkpoint_L_pairs else fwd
    _, vjp_fn = vjp(fwd_run, quats_base, g_tf, gd_tf, I_tf)
    return vjp_fn(v_B)


_vjp_reduced_free_quat_only_jitted = jax.jit(
    _vjp_reduced_free_quat_only_run,
    static_argnames=(
        "nfp",
        "stellsym",
        "delta_reg",
        "eigenfloor_threshold",
        "adaptive_self_reg",
        "checkpoint_L_pairs",
        "pair_row_chunk",
        "eval_chunk",
        "vjp_probe_mode",
        "solve_vjp_mode",
        "use_far_dipole_pair",
        "pair_far_kappa",
        "tf_loading",
        "solve_mode_reduced",
        "bs_eval_far_kappa",
        "use_w1_envelope",
        "pair_replica_chunk",
    ),
)


def _vjp_reduced_free_center_only_run(
    local_pts_base: jnp.ndarray,
    local_K_base: jnp.ndarray,
    local_n_base: jnp.ndarray,
    local_w_base: jnp.ndarray,
    local_phi_base: jnp.ndarray,
    Q_c_base: jnp.ndarray,
    centers_base: jnp.ndarray,
    quats_base: jnp.ndarray,
    g_tf: jnp.ndarray,
    gd_tf: jnp.ndarray,
    I_tf: jnp.ndarray,
    pts_eval: jnp.ndarray,
    L_red_eig_U: jnp.ndarray,
    L_red_eig_lam: jnp.ndarray,
    v_B: jnp.ndarray,
    base_indices: jnp.ndarray,
    base_reps: jnp.ndarray,
    G_float: jnp.ndarray,
    nfp: int,
    stellsym: bool,
    delta_reg: float,
    eigenfloor_threshold: float,
    adaptive_self_reg: bool,
    checkpoint_L_pairs: bool,
    pair_row_chunk: int,
    eval_chunk: int,
    vjp_probe_mode: str,
    solve_vjp_mode: str,
    m_local_base: jnp.ndarray,
    Q_sym_local: jnp.ndarray,
    use_far_dipole_pair: bool,
    pair_far_kappa: float,
    tf_loading: str,
    solve_mode_reduced: str,
    bs_eval_far_kappa: float,
    use_w1_envelope: bool = False,
    pair_replica_chunk: int = 0,
    near_pair_indices: Optional[jnp.ndarray] = None,
    far_pair_indices: Optional[jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Reduced VJP when only puck center DOFs are active."""

    def fwd(c_b, g, gd, I):
        return _B_eval_reduced_free_dof_dispatch(
            local_pts_base,
            local_K_base,
            local_n_base,
            local_w_base,
            local_phi_base,
            Q_c_base,
            c_b,
            quats_base,
            g,
            gd,
            I,
            pts_eval,
            L_red_eig_U,
            L_red_eig_lam,
            base_indices,
            base_reps,
            G_float,
            nfp,
            stellsym,
            delta_reg,
            eigenfloor_threshold,
            adaptive_self_reg,
            checkpoint_L_pairs,
            pair_row_chunk,
            eval_chunk,
            vjp_probe_mode,
            solve_vjp_mode,
            use_far_dipole_pair,
            pair_far_kappa,
            m_local_base,
            Q_sym_local,
            tf_loading,
            solve_mode_reduced,
            bs_eval_far_kappa,
            use_w1_envelope,
            pair_replica_chunk,
            near_pair_indices,
            far_pair_indices,
        )

    fwd_run = jax.checkpoint(fwd, prevent_cse=False) if checkpoint_L_pairs else fwd
    _, vjp_fn = vjp(fwd_run, centers_base, g_tf, gd_tf, I_tf)
    return vjp_fn(v_B)


_vjp_reduced_free_center_only_jitted = jax.jit(
    _vjp_reduced_free_center_only_run,
    static_argnames=(
        "nfp",
        "stellsym",
        "delta_reg",
        "eigenfloor_threshold",
        "adaptive_self_reg",
        "checkpoint_L_pairs",
        "pair_row_chunk",
        "eval_chunk",
        "vjp_probe_mode",
        "solve_vjp_mode",
        "use_far_dipole_pair",
        "pair_far_kappa",
        "tf_loading",
        "solve_mode_reduced",
        "bs_eval_far_kappa",
        "use_w1_envelope",
        "pair_replica_chunk",
    ),
)


# ======================================================================
# Module-level dipole JIT handles (Phase L3, May 2026)
# ======================================================================
#
# Hoisted out of the per-instance ``_ensure_dipole_*_jit_cache`` methods
# below so that the JAX internal abstract-shape cache survives across
# :class:`PSCBulkArray` instances.  Before this change every new
# instance built during Fourier continuation (or any sweep / test
# re-instantiation) created a fresh ``jax.jit(...)`` wrapper whose
# internal cache started empty, forcing a full XLA re-trace of the
# dipole VJP graph at every stage (tens to hundreds of seconds per
# stage at reactor scale, see
# ``bench_results/dipole_solver/fc_bulk_stage_profile_before.md``).
#
# The module-level handles are shared by ALL instances; JAX keys its
# internal cache on ``(input shapes, static argument values)`` so
# matching geometries hit the existing XLA artifact regardless of which
# :class:`PSCBulkArray` first compiled it.  Phase-K
# :meth:`PSCBulkArray.warm_handoff_from` still copies the per-instance
# ``_dipole_jit_*`` slots; since those slots point at the same
# module-level objects, the copy is a no-op but the contract continues
# to hold.

from jax.scipy.linalg import cho_solve as _DIPOLE_JAX_CHO_SOLVE


_DIPOLE_ASSEMBLE_L_JIT = jax.jit(
    _psc_bulk_dipole_mod.assemble_L_dipole_reduced_jax,
    static_argnums=(3, 4),
)
"""Module-level JIT handle for :func:`assemble_L_dipole_reduced_jax`."""


_DIPOLE_ASSEMBLE_F_JIT = jax.jit(
    _psc_bulk_dipole_mod.assemble_f_dipole_reduced_jax,
    static_argnums=(5, 6),
)
"""Module-level JIT handle for :func:`assemble_f_dipole_reduced_jax`."""


_DIPOLE_FIELD_JIT = jax.jit(
    _psc_bulk_dipole_mod.B_at_points_dipole,
    static_argnums=(4, 5),
)
"""Module-level JIT handle for :func:`B_at_points_dipole`."""


_DIPOLE_FORWARD_JIT = jax.jit(
    _psc_bulk_dipole_mod.forward_dipole_pipeline,
    static_argnums=(7, 8),
)
"""Module-level JIT handle for :func:`forward_dipole_pipeline`."""


def _dipole_vjp_B_kernel(
    pts_arg: "jnp.ndarray",
    centers_base: "jnp.ndarray",
    quats_base: "jnp.ndarray",
    m_global_base: "jnp.ndarray",
    v_B_arg: "jnp.ndarray",
    nfp_arg: int,
    stellsym_arg: bool,
) -> "tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]":
    """Backward through :func:`B_at_points_dipole` (centres / quats / m).

    Discards the ``pts`` cotangent because evaluation points are not
    optimisation DoFs.  Static ``(nfp, stellsym)`` are concretised inside
    JAX so independent ``(nfp, stellsym)`` pairs share this trace
    template without recompilation of the kernel object itself.
    """
    _, vjp_fn = jax.vjp(
        _psc_bulk_dipole_mod.B_at_points_dipole,
        pts_arg,
        centers_base,
        quats_base,
        m_global_base,
        int(nfp_arg),
        bool(stellsym_arg),
    )
    _, lam_c, lam_q, lam_m, _, _ = vjp_fn(v_B_arg)
    return lam_c, lam_q, lam_m


_DIPOLE_VJP_B_JIT = jax.jit(_dipole_vjp_B_kernel, static_argnums=(5, 6))
"""Module-level JIT handle for the dipole VJP through ``B_at_points_dipole``."""


def _dipole_vjp_f_kernel(
    centers_base: "jnp.ndarray",
    quats_base: "jnp.ndarray",
    g_tf_arg: "jnp.ndarray",
    gd_tf_arg: "jnp.ndarray",
    I_tf_arg: "jnp.ndarray",
    lam_f_arg: "jnp.ndarray",
    nfp_arg: int,
    stellsym_arg: bool,
) -> "tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]":
    """Backward through :func:`assemble_f_dipole_reduced_jax`.

    Returns ``(lam_c, lam_q, vg, vgd, vI)`` from the cotangent of
    ``f_red`` (which is computed on the host via :func:`cho_solve`).
    """
    _, vjp_fn = jax.vjp(
        _psc_bulk_dipole_mod.assemble_f_dipole_reduced_jax,
        centers_base,
        quats_base,
        g_tf_arg,
        gd_tf_arg,
        I_tf_arg,
        int(nfp_arg),
        bool(stellsym_arg),
    )
    lam_c, lam_q, vg, vgd, vI, _, _ = vjp_fn(lam_f_arg)
    return lam_c, lam_q, vg, vgd, vI


_DIPOLE_VJP_F_JIT = jax.jit(_dipole_vjp_f_kernel, static_argnums=(6, 7))
"""Module-level JIT handle for the dipole VJP through ``assemble_f``."""


def _dipole_vjp_L_kernel(
    centers_base: "jnp.ndarray",
    quats_base: "jnp.ndarray",
    self_L_local_cached: "jnp.ndarray",
    L_cotangent: "jnp.ndarray",
    nfp_arg: int,
    stellsym_arg: bool,
) -> "tuple[jnp.ndarray, jnp.ndarray]":
    """Backward through :func:`assemble_L_dipole_reduced_jax` -- centres / quats only.

    Only invoked when centre or quaternion DoFs are free.
    """
    _, vjp_fn = jax.vjp(
        _psc_bulk_dipole_mod.assemble_L_dipole_reduced_jax,
        centers_base,
        quats_base,
        self_L_local_cached,
        int(nfp_arg),
        bool(stellsym_arg),
    )
    lam_c, lam_q, _, _, _ = vjp_fn(L_cotangent)
    return lam_c, lam_q


_DIPOLE_VJP_L_JIT = jax.jit(_dipole_vjp_L_kernel, static_argnums=(4, 5))
"""Module-level JIT handle for the dipole VJP through ``assemble_L``."""


def _dipole_vjp_fused_tf_kernel(
    pts_arg: "jnp.ndarray",
    centers_base: "jnp.ndarray",
    quats_base: "jnp.ndarray",
    m_global_base: "jnp.ndarray",
    g_tf_arg: "jnp.ndarray",
    gd_tf_arg: "jnp.ndarray",
    I_tf_arg: "jnp.ndarray",
    L_chol: "jnp.ndarray",
    v_B_arg: "jnp.ndarray",
    nfp_arg: int,
    stellsym_arg: bool,
) -> "tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]":
    """Fused TF-only dipole VJP: B-backward + cho_solve + f-backward.

    Combines the three small backward kernels into a single JAX trace
    to minimise per-call dispatch overhead in the hot loop where the
    L-side adjoint is not needed (i.e. centres + quats fixed).
    """
    _, vjp_B = jax.vjp(
        _psc_bulk_dipole_mod.B_at_points_dipole,
        pts_arg,
        centers_base,
        quats_base,
        m_global_base,
        int(nfp_arg),
        bool(stellsym_arg),
    )
    _, lam_c_B, lam_q_B, lam_m_b, _, _ = vjp_B(v_B_arg)
    lam_m = lam_m_b.reshape(-1)
    lam_f = -_DIPOLE_JAX_CHO_SOLVE((L_chol, True), lam_m)
    _, vjp_f = jax.vjp(
        _psc_bulk_dipole_mod.assemble_f_dipole_reduced_jax,
        centers_base,
        quats_base,
        g_tf_arg,
        gd_tf_arg,
        I_tf_arg,
        int(nfp_arg),
        bool(stellsym_arg),
    )
    lam_c_f, lam_q_f, vg, vgd, vI, _, _ = vjp_f(lam_f)
    vc = lam_c_B + lam_c_f
    vq = lam_q_B + lam_q_f
    return vc, vq, vg, vgd, vI


_DIPOLE_VJP_FUSED_TF_JIT = jax.jit(
    _dipole_vjp_fused_tf_kernel, static_argnums=(9, 10)
)
"""Module-level JIT handle for the fused dipole VJP (TF-only hot loop)."""


# ======================================================================
# PSCBulkArray
# ======================================================================


class PSCBulkArray(Optimizable):
    """
    Passive superconducting bulk cylinders in the ideal-diamagnetic limit.

    Orientation is stored as a **scalar-first quaternion** ``[q0, qi, qj, qk]``
    matching :class:`~simsopt.geo.curveplanarfourier.CurvePlanarFourier`.

    **Optimizable DOFs** (all fixed by default):

    For each base puck *i*, 9 DOFs are registered::

        center_x{i}, center_y{i}, center_z{i},
        q0_{i}, qi_{i}, qj_{i}, qk_{i},
        R{i}, t{i}

    Gradients w.r.t. center and quaternion DOFs are computed analytically
    via JAX VJPs.  Gradients w.r.t. R and t are **not yet** computed
    analytically; those components are zero in the VJP.

    Args:
        puck_centers: ``(N, 3)`` centers in meters.
        puck_axes: ``(N, 3)`` unit normal vectors (bottom to top cap).
            Internally converted to quaternions.
        puck_radii: ``(N,)`` radii.
        puck_thicknesses: ``(N,)`` thicknesses in meters.
        coils_TF: list of TF :class:`~simsopt.field.coil.Coil` objects.
        eval_points: ``(M, 3)`` points where the passive field is evaluated.
        m_fourier, l_zernike, k_chebyshev: basis resolution.
        n_rho, n_phi, n_z: quadrature resolution.
        nfp: number of field periods (replication of pucks).
        stellsym: whether to apply stellarator symmetry to puck positions.
        plasma_flux: optional extra flux per puck (not used in v1).
        regularization_delta: distance regularization for self-terms.
        default_thickness: fallback thickness in meters.
        n_phi_rim: number of azimuthal collocation points used to enforce
            rim continuity (:math:`H^1(\\Sigma_i)/\\mathbb R`, eq 55 of the
            passive-bulk note) between the disk and side-wall basis patches.
            Defaults to ``max(2*m_fourier+1, 8)`` (Nyquist-safe for the
            highest azimuthal mode retained in the disk/side bases).
        null_space_threshold: relative eigenvalue cutoff for the gauge /
            null-space projection of ``L`` in the fixed-DOF path.  Defaults
            to ``1e-10``.
        eigenfloor_threshold: relative eigenvalue floor used by the
            regularized Cholesky solve in the free-DOF JAX path.  Defaults
            to the module-level ``_EIGENFLOOR_THRESHOLD = 1e-10``.
        adaptive_self_reg: If ``True`` (the default), the
            inductance-matrix self-terms use a quadrature-cell-scaled
            regularization :math:`\\delta_i = (3\\pi^{3/2}/8)\\sqrt{w_i}`
            matching the analytic flat-disc self-integral
            :math:`(8/3)(w_i/\\pi)^{3/2}` on the coincident cell.  This
            removes the ~:math:`10^{4}-10^{5}` overcount of coincident
            contributions caused by the legacy fixed scalar
            ``regularization_delta`` in the :math:`1/|r-r'|` double
            integral and is required to recover analytic magnitudes
            (Lorenz short-solenoid, Smythe thin-disc) and physically
            correct induced fields on the plasma surface.  Pass
            ``False`` only as an explicit opt-in to the legacy uniform
            :math:`\\sqrt{r^2+\\delta^2}` regularization for strict
            backward-compatibility regression testing; new applications
            should leave this at the default.
        strict_rim_continuity: If ``True``, require that the per-puck
            rim-continuity constraint
            :math:`H^1(\\Sigma_i)/\\mathbb R` (paper eq 55) has a
            non-trivial kernel on **every** puck and raise
            :class:`ValueError` otherwise.  The default (``False``) only
            emits a :class:`UserWarning` and falls back to the raw basis
            (no rim continuity enforced) - useful for low-order smoke
            tests, but dangerous for production optimizations where the
            lack of continuity silently changes the solution.  Production
            examples should set ``strict_rim_continuity=True``.
        solver_mode: Which discrete weak form to use to determine the
            modal coefficients :math:`\\beta`.

            * ``"energy"`` (default): the current-potential /
              mutual-inductance Gram, ``L beta = f`` with
              ``L_{ab} = (mu_0/4pi) integral K_a . K_b / |r-r'| dS dS'``
              and ``f_a = -integral Phi_a B_n^{TF} dS``.  Minimises
              ``0.5 beta^T L beta + beta^T f`` on the rim-continuity /
              gauge-null subspace.  Recovers the induced dipole
              correctly (Lenz + sphere-scale limits) and is the path
              used for every production optimisation because it supports
              the JIT TF-only VJP and the full-JAX free-DOF VJP.
            * ``"shell_l2"``: REGCOIL-style :math:`L^2`-residual form
              ``M beta = g`` with
              ``M_{ab} = integral B_n^{(a)} B_n^{(b)} dS`` and
              ``g_a = -integral B_n^{(a)} B_n^{TF} dS``, where
              :math:`B_n^{(a)}` is the normal component on the shell of
              the field from basis function ``a``.  Drives
              :math:`\\|B_n^{tot}\\|_{L^2(\\Sigma)}` monotonically to
              zero as the basis is refined and hence drives the ideal-
              diamagnet interior-field cancellation
              :math:`|B^{tot}(\\mathbf x_{\\rm int})|\\to 0`, which the
              ``"energy"`` form only enforces in the Galerkin sense.
              Opt-in because the assembly is dense in puck-pair blocks
              (cost scales like the off-diagonal of ``L`` times ``n_q``)
              and because the JIT TF-VJP and free-DOF-VJP paths are not
              yet implemented on this branch; ``B_at_points`` falls
              back to a pure-NumPy Biot-Savart of the cached
              :math:`\\beta`.  The reduced symmetry path is also
              disabled on ``"shell_l2"`` for the same reason.

        full_L_band_size (constructor): Row-band size for streaming assembly
            of the monolithic ``L`` inside :func:`_B_eval_full_body` when free
            puck DOFs are active; ``<= 0`` or ``>= n_pucks`` uses the dense
            path.  Default ``32``.

        use_symmetry_reduced_free_dof (constructor): If ``True`` (default),
            and the signed-orbit symmetry-reduced host ``L_work`` path is
            active, use the factor-:math:`|G|` cheaper
            :func:`_B_eval_reduced_free_dof_body` / VJP when any puck DOF is
            free.  Set ``False`` to force the dense :math:`N^2` pair assembly
            for debugging.  Can also be disabled at runtime with
            ``SIMSOPT_PSC_DISABLE_REDUCED_FREE=1``.

        checkpoint_l_pairs (constructor): If ``True``, force
            :func:`jax.checkpoint` on per-pair ``L`` blocks in the free-DOF JAX
            path.  If ``False``, disable checkpointing.  If ``None`` (default),
            enable checkpointing when any puck DoF is unfixed, unless
            :envvar:`SIMSOPT_PSC_JAX_CHECKPOINT` is set to override.

        jax_pair_row_chunk (constructor): Host-side row-batch width for
            :func:`_B_eval_reduced_free_dof_body` when the symmetry-reduced
            free-DoF path is active.  ``0`` (default) means use
            :envvar:`SIMSOPT_PSC_JAX_PAIR_CHUNK` or an automatic ~1~GiB target;
            a positive int caps the batch width.

        pair_replica_chunk (constructor): In-row chunk width along the
            symmetry replica axis ``n_all`` for the free-DoF pair-inductance
            kernel.  ``0`` (default) keeps the legacy single-``vmap`` over all
            replicas (peak ``n_all * nq^2 * nd^2 * 8`` bytes per row).  A
            positive int folds the replica axis via :func:`jax.lax.scan` in
            groups of that size, capping the per-row pair-kernel batch at
            ``pair_replica_chunk * nq^2 * nd^2 * 8`` bytes.  Overridden by
            :envvar:`SIMSOPT_PSC_PAIR_REPLICA_CHUNK`.

        use_f32_bs (constructor): If ``True``, the ``shell_l2`` Biot-Savart
            branch in :meth:`B_at_points` accumulates in ``float32`` inside
            :func:`~simsopt.field.bulk_inductance.shell_biot_savart_stacked_pure`.

        release_host_L_work_after_rebuild (constructor): If ``True``, after
            rebuild (``solver_mode == 'energy'``) drop the host :attr:`_L_work`
            when it is not needed: the *full-replica* dense matrix in the
            non-reduced path, or the folded base matrix in the
            symmetry-reduced path (the free-DOF JAX VJP re-assembles ``L`` on
            device and does not read :attr:`_L_work`).  Default ``False`` so
            :attr:`_L_work` remains available for tests and Galerkin identities.

        mode_truncate (constructor): Optional ``Mapping[str, Any]`` requesting
            per-puck basis-column pruning *above* the cached
            :func:`~simsopt.field.puck_basis.build_puck_shell_basis` output.
            Recognised keys are ``"max_m_disk"``, ``"max_n_disk"``,
            ``"max_m_side"``, ``"max_k_side"`` (all ``int``), and
            ``"drop_disk_top"`` / ``"drop_disk_bot"`` / ``"drop_side"``
            (all ``bool``).  See
            :func:`simsopt.field.puck_basis.apply_mode_truncate` for the
            full semantics.  ``None`` (default) leaves the basis unchanged.
            Aggressive truncation can drive the rim-continuity matrix
            rank-deficient; pair with ``strict_rim_continuity=True`` to
            fail fast in that case, or accept the existing
            :class:`UserWarning` fallback to the raw basis.  The
            normalised form of this mapping is stored in
            :attr:`_mode_truncate_norm` and is intentionally constant
            after construction (changing the truncation requires a new
            instance).

        normal_offset: Optional ``Mapping`` requesting the
            normal-offset center parameterization for a subset of pucks
            (typically built by
            :meth:`from_winding_surface`'s ``center_parameterization=
            "normal_offset"`` option rather than passed directly). Keys:
            ``"mask"`` (``(N,) bool``, which base pucks use this
            parameterization), ``"anchors"`` (``(N, 3)``, meaningful
            where ``mask`` is ``True``), ``"normals"`` (``(N, 3)``,
            likewise). For each puck ``i`` where ``mask[i]``, a new
            scalar DOF ``d{i}`` is added with ``center_i = anchors[i] +
            d{i} * normals[i]``; ``center_x{i}``/``center_y{i}``/
            ``center_z{i}`` remain present but become permanently
            unfixable (shadowed) for that puck. ``None`` (default)
            leaves every puck on the ordinary independent
            ``center_x/y/z`` parameterization.
    """

    def __init__(
        self,
        puck_centers: np.ndarray,
        puck_axes: np.ndarray,
        puck_radii: np.ndarray,
        puck_thicknesses: np.ndarray,
        coils_TF,
        eval_points: np.ndarray,
        m_fourier: int = 4,
        l_zernike: int = 6,
        k_chebyshev: int = 4,
        n_rho: int = 10,
        n_phi: int = 12,
        n_z: int = 6,
        nfp: int = 1,
        stellsym: bool = False,
        plasma_flux: Optional[np.ndarray] = None,
        regularization_delta: float = 1e-6,
        default_thickness: float = 0.02,
        n_phi_rim: Optional[int] = None,
        null_space_threshold: float = 1e-10,
        eigenfloor_threshold: float = _EIGENFLOOR_THRESHOLD,
        adaptive_self_reg: bool = True,
        strict_rim_continuity: bool = False,
        solver_mode: str = "energy",
        full_L_band_size: int = 32,
        use_f32_bs: bool = False,
        release_host_L_work_after_rebuild: bool = False,
        use_symmetry_reduced_free_dof: bool = True,
        checkpoint_l_pairs: Optional[bool] = None,
        jax_pair_row_chunk: int = 0,
        pair_replica_chunk: int = 0,
        mode_truncate: Optional[Mapping[str, Any]] = None,
        mode_truncation_tol: float = 0.0,
        bulk_far_pair_kappa: float = 0.0,
        bulk_far_pair_tol: float = 1.0e-2,
        fixed_field_matrix_dtype: str = "float64",
        normal_offset: Optional[Mapping[str, np.ndarray]] = None,
    ):
        self.coils_TF = list(coils_TF)
        # Phase-J J1: ``eval_points`` is now a property whose setter
        # bumps :attr:`_eval_points_version` and clears the cached
        # device-side ``_dipole_jdev_pts``.  Initialise the dipole
        # cache slot to ``None`` first so the setter's invalidation
        # branch sees a defined attribute on every assignment path.
        self._dipole_jdev_pts: Optional[Any] = None
        self._eval_points_version: int = 0
        self.eval_points = np.asarray(eval_points, dtype=float, order="C")
        self.nfp = int(nfp)
        self.stellsym = bool(stellsym)
        self.regularization_delta = float(regularization_delta)
        self.adaptive_self_reg = bool(adaptive_self_reg)
        self.m_fourier = m_fourier
        self.l_zernike = l_zernike
        self.k_chebyshev = k_chebyshev
        self._n_rho = n_rho
        self._n_phi = n_phi
        self._n_z = n_z
        self._plasma_flux = plasma_flux
        if n_phi_rim is None:
            n_phi_rim = max(2 * int(m_fourier) + 1, 8)
        self._n_phi_rim = int(n_phi_rim)
        self._null_space_threshold = float(null_space_threshold)
        self._eigenfloor_threshold = float(eigenfloor_threshold)
        self._strict_rim_continuity = bool(strict_rim_continuity)
        if solver_mode not in ("energy", "shell_l2", "dipole"):
            raise ValueError(
                "solver_mode must be 'energy', 'shell_l2', or 'dipole'; "
                f"got {solver_mode!r}"
            )
        self.solver_mode = str(solver_mode)
        # Dipole-mode caches (Phase A/B/D of the lean dipole bulk solver
        # plan).  Set lazily on the first ``_rebuild`` call when
        # ``solver_mode == 'dipole'``; left ``None`` otherwise so the
        # legacy paths see no overhead.
        self._dipole_self_L_local_cached: Optional[np.ndarray] = None
        self._dipole_self_L_keys: Optional[Tuple[Tuple[float, float], ...]] = None
        self._dipole_L_red: Optional[np.ndarray] = None
        self._dipole_L_red_chol: Optional[Tuple[np.ndarray, bool]] = None
        self._dipole_m_red: Optional[np.ndarray] = None
        # Persistent jax.jit handles for the dipole forward
        # (free-DoF + value path) and the cached plasma-field
        # evaluation.  Keyed on
        # ``(n_base, n_eval, nfp, stellsym, n_tf, n_tf_quad)`` so that
        # repeated calls with the same shapes reuse the compiled
        # kernel instead of re-tracing through ``vmap`` every time.
        self._dipole_jit_forward: Optional[Any] = None
        self._dipole_jit_field: Optional[Any] = None
        self._dipole_jit_key: Optional[tuple] = None
        # Persistent jits for the rebuild path -- ``assemble_L`` and
        # ``assemble_f`` are called from :meth:`_rebuild_dipole` on
        # every outer optimisation iter so a cached compiled kernel
        # eliminates the dominant per-iter cost.
        self._dipole_jit_assemble_L: Optional[Any] = None
        self._dipole_jit_assemble_f: Optional[Any] = None
        self._dipole_jit_rebuild_key: Optional[tuple] = None
        # Separate JIT-key for the cached-B path that drops TF shape.
        # The closed-form dipole sum does not depend on TF arrays so
        # this key is just ``(n_base, n_eval, nfp, stellsym)``.
        self._dipole_jit_field_key: Optional[tuple] = None
        # Phase-H analytic VJP: three small jitted kernels replace the
        # old monolithic ``jax.grad(forward_dipole_pipeline)`` so the
        # backward never reassembles ``L_red`` or refactors Cholesky.
        # ``_dipole_jit_vjp_B``        -- jax.vjp of ``B_at_points_dipole`` (no TF)
        # ``_dipole_jit_vjp_f``        -- jax.vjp of ``assemble_f_dipole_reduced_jax``
        # ``_dipole_jit_vjp_L``        -- jax.vjp of ``assemble_L_dipole_reduced_jax``
        # All three are keyed on the shape tuple stored in
        # ``_dipole_jit_vjp_key``.  ``_dipole_jit_vjp`` is the legacy
        # monolithic slot kept for the parity test.
        self._dipole_jit_vjp: Optional[Any] = None
        self._dipole_jit_vjp_B: Optional[Any] = None
        self._dipole_jit_vjp_f: Optional[Any] = None
        self._dipole_jit_vjp_L: Optional[Any] = None
        # Fused TF-only analytic backward: collapses
        # vjp_B + cho_solve + vjp_f into a single JIT trace so the
        # allfixed iter pays a single device dispatch.  The L-side
        # vjp branch is kept separate because it is only used when
        # centre / quaternion DoFs are free.
        self._dipole_jit_vjp_fused_tf: Optional[Any] = None
        self._dipole_jit_vjp_key: Optional[tuple] = None
        # Device-side ``cho_solve`` factor of ``L_red`` for the
        # fused TF-only backward.  Populated alongside the host-side
        # ``_dipole_L_red_chol`` in :meth:`_rebuild_dipole`.
        self._dipole_jdev_L_red_chol: Optional[Any] = None
        # Device-resident snapshots of static puck-geometry arrays.
        # Populated at the end of each ``_rebuild_dipole`` and reused
        # by the cached forward + analytic VJP so we don't pay an
        # ``np.asarray`` -> ``jnp.asarray`` host-to-device copy per
        # call.  Cleared (set to ``None``) when the rebuild changes
        # ``L_red`` or the moments.
        self._dipole_jdev_centers: Optional[Any] = None
        self._dipole_jdev_quats: Optional[Any] = None
        self._dipole_jdev_self_L_local: Optional[Any] = None
        self._dipole_jdev_m_global: Optional[Any] = None
        # When ``True``, the cached ``_dipole_self_L_local_cached`` is
        # reused across rebuilds even if ``R_i`` / ``t_i`` change.
        # Default behaviour comes from
        # :func:`_psc_dipole_freeze_self_l_env`.
        self._dipole_freeze_self_L: bool = _psc_dipole_freeze_self_l_env(
            default_on=(self.solver_mode == "dipole")
        )
        self._full_L_band_size = int(full_L_band_size)
        self._use_f32_bs = bool(use_f32_bs)
        self._release_host_L_work_after_rebuild = bool(
            release_host_L_work_after_rebuild
        )
        self.use_symmetry_reduced_free_dof = bool(use_symmetry_reduced_free_dof)
        # None = auto: enable per-pair checkpoint in the free-puck-DoF JAX
        # path when any puck DoF is unfixed (see :meth:`_resolved_checkpoint_l_pairs`).
        if checkpoint_l_pairs is None:
            self._psc_checkpoint_l_pairs_opt: Optional[bool] = None
        else:
            self._psc_checkpoint_l_pairs_opt = bool(checkpoint_l_pairs)
        self._jax_pair_row_chunk_ctor: int = int(jax_pair_row_chunk)
        self._pair_replica_chunk_ctor: int = int(pair_replica_chunk)
        self._bulk_far_pair_kappa: float = max(0.0, float(bulk_far_pair_kappa))
        self._bulk_far_pair_tol: float = max(0.0, float(bulk_far_pair_tol))
        # Storage dtype for the fixed-puck BS-to-plasma matrix ``_M_field``.
        # ``"float64"`` (default) keeps full precision; ``"float32"`` halves
        # the matrix footprint and bandwidth at the cost of ~1e-4 relative
        # error in the bulk field.  The result of the gemv is always
        # upcast to ``float64`` before reshape so downstream callers see
        # the same dtype regardless of this knob.
        if fixed_field_matrix_dtype not in ("float64", "float32"):
            raise ValueError(
                "fixed_field_matrix_dtype must be 'float64' or 'float32'; "
                f"got {fixed_field_matrix_dtype!r}"
            )
        self._fixed_field_matrix_dtype: str = str(fixed_field_matrix_dtype)
        # Per-puck basis truncation (Stage 2 of the
        # ``psc_bulk_speedups_and_mode_reduction`` plan).  ``mode_truncate``
        # is a constant attribute of the instance: it is captured once at
        # construction, normalised into a hashable tuple, and applied in
        # :meth:`_rebuild` immediately after :func:`build_puck_shell_basis`.
        # Storing both the original mapping (for repr/debug) and its
        # normalised form (for cache keys / equality checks) makes the
        # downstream code self-documenting.
        self._mode_truncate_input: Optional[Mapping[str, Any]] = (
            dict(mode_truncate) if mode_truncate is not None else None
        )
        self._mode_truncate_norm: Optional[Tuple[Tuple[str, Any], ...]] = (
            normalize_mode_truncate(mode_truncate)
        )
        # ``mode_truncation_tol`` is a *spectral* truncation applied to the
        # gauge-null-projected operator ``L_c = Q_c^T L Q_c`` after its
        # eigendecomposition: any eigenmode whose response amplitude
        # ``1 / lambda_k`` is below ``mode_truncation_tol`` times the
        # maximum (i.e. ``lambda_k > lambda_min_kept / mode_truncation_tol``)
        # is dropped from ``Q``.  This is the appendix's speedup (vii)
        # ("mode truncation") realised in the eigenbasis of ``L_c``;
        # ``mode_truncate`` (the structural Mapping above) is unrelated -- it
        # filters Fourier-Zernike modes *before* assembly, while this knob
        # filters spectral modes *after* assembly.  Default ``0.0`` is a
        # no-op.  Production-safe values typically lie in ``[1e-4, 1e-2]``;
        # see :meth:`_rebuild` for the exact discard rule.
        self._mode_truncation_tol = float(mode_truncation_tol)
        self._H_full = None
        self._Hw = None

        centers = np.atleast_2d(np.asarray(puck_centers, dtype=float))
        axes = np.atleast_2d(np.asarray(puck_axes, dtype=float))
        radii = np.atleast_1d(np.asarray(puck_radii, dtype=float))
        ths = np.atleast_1d(np.asarray(puck_thicknesses, dtype=float))
        if ths.size == 0:
            ths = np.full(len(centers), default_thickness)
        ths = np.where(ths <= 0, default_thickness, ths)

        self._n_base_pucks = len(centers)

        # Convert axes to quaternions
        quats = np.array([_axis_to_quaternion(axes[i]) for i in range(len(axes))])

        dof_values: List[float] = []
        dof_names: List[str] = []
        for i in range(self._n_base_pucks):
            dof_values.extend(
                [
                    centers[i, 0],
                    centers[i, 1],
                    centers[i, 2],
                    quats[i, 0],
                    quats[i, 1],
                    quats[i, 2],
                    quats[i, 3],
                    radii[i],
                    ths[i],
                ]
            )
            dof_names.extend(
                [
                    f"center_x{i}",
                    f"center_y{i}",
                    f"center_z{i}",
                    f"q0_{i}",
                    f"qi_{i}",
                    f"qj_{i}",
                    f"qk_{i}",
                    f"R{i}",
                    f"t{i}",
                ]
            )

        # Normal-offset center parameterization (opt-in): for pucks with
        # ``mask[i]`` set, replace the independent ``center_x/y/z{i}``
        # DOFs with a single derived scalar ``d{i}`` s.t. ``center_i =
        # anchor_i + d_i * normal_i``.  ``center_x/y/z{i}`` stay present
        # (for layout uniformity with ordinary pucks) but become
        # permanently unfixable for these pucks; see ``unfix``/
        # ``local_unfix_all`` and ``_get_base_puck_geometry``.
        self._normal_offset_anchor: Dict[int, np.ndarray] = {}
        self._normal_offset_normal: Dict[int, np.ndarray] = {}
        self._normal_offset_puck_indices: List[int] = []
        self._normal_offset_dof_index: Dict[int, int] = {}
        if normal_offset is not None:
            mask = np.asarray(normal_offset["mask"], dtype=bool)
            anchors_in = np.asarray(normal_offset["anchors"], dtype=float)
            normals_in = np.asarray(normal_offset["normals"], dtype=float)
            if mask.shape != (self._n_base_pucks,):
                raise ValueError(
                    "normal_offset['mask'] must have shape "
                    f"({self._n_base_pucks},); got {mask.shape}"
                )
            if anchors_in.shape != (self._n_base_pucks, 3) or normals_in.shape != (
                self._n_base_pucks,
                3,
            ):
                raise ValueError(
                    "normal_offset['anchors'] and ['normals'] must have "
                    f"shape ({self._n_base_pucks}, 3); got "
                    f"{anchors_in.shape} and {normals_in.shape}"
                )
            for i in np.flatnonzero(mask):
                i = int(i)
                normal_norm = float(np.linalg.norm(normals_in[i]))
                if normal_norm < 1e-12:
                    raise ValueError(
                        f"normal_offset: puck {i}'s normal is degenerate "
                        f"(norm={normal_norm:.3e})."
                    )
                normal_i = normals_in[i] / normal_norm
                anchor_i = anchors_in[i]
                d0_i = float(np.dot(centers[i] - anchor_i, normal_i))
                residual = centers[i] - (anchor_i + d0_i * normal_i)
                residual_norm = float(np.linalg.norm(residual))
                if residual_norm > 1e-6:
                    raise ValueError(
                        f"normal_offset: puck {i}'s center does not lie "
                        f"on its (anchor, normal) line -- off-axis "
                        f"residual {residual_norm:.3e} m. centers[{i}] "
                        "must equal anchors[i] + d*normals[i] for some "
                        "d, else the projection would silently discard "
                        "the off-axis component."
                    )
                self._normal_offset_anchor[i] = anchor_i
                self._normal_offset_normal[i] = normal_i
                self._normal_offset_puck_indices.append(i)
                self._normal_offset_dof_index[i] = len(dof_values)
                dof_values.append(d0_i)
                dof_names.append(f"d{i}")
        self._normal_offset_dof_names: frozenset = frozenset(
            f"d{i}" for i in self._normal_offset_puck_indices
        )
        self._n_geom_dofs: int = (
            self._n_base_pucks * _DOFS_PER_PUCK + len(self._normal_offset_puck_indices)
        )

        fixed = [True] * len(dof_values)
        Optimizable.__init__(
            self,
            x0=np.array(dof_values),
            names=dof_names,
            fixed=fixed,
            depends_on=self.coils_TF,
        )

        self._geom_hash: Optional[int] = None
        # In-process snapshot of ``(id(opt._dofs), opt._dofs._state_version)``
        # for ``opt in self._unique_dof_opts`` at the last successful
        # ``recompute_currents``.  Used by ``recompute_currents`` for an
        # ``O(N_unique_dof_opts)`` "have any DOFs changed?" check; this
        # replaces the previous ``hash(tuple(self.local_full_x))`` per call.
        # ``_geom_hash`` and the disk-cache ``_psc_lcache_digest`` are kept
        # because ``_state_version`` resets every process and cannot serve as
        # a cross-process key.
        self._geom_versions: Optional[tuple] = None
        self._local_stacks_valid: bool = False
        self._timing_rows: Optional[List[Dict[str, Any]]] = None
        # Accumulator that survives across :meth:`_rebuild` calls.  Each
        # row in ``_timing_rows`` is appended here (with a ``rebuild_idx``
        # tag) just before ``_timing_rows`` is reset at the start of the
        # next ``_rebuild``.  This lets callers (e.g. the optimizer
        # timing harness) pull the full per-phase history of a run via
        # :meth:`get_timing_history` after scipy returns; without this
        # accumulator, only the rows since the most recent rebuild would
        # remain in memory.  Only populated when timing is enabled
        # (:func:`_psc_bulk_timing_enabled`); otherwise stays empty so
        # there is no measurable overhead.
        self._timing_history: List[Dict[str, Any]] = []
        self._timing_rebuild_counter: int = 0
        self._structural_key: Optional[tuple] = None
        self._free_vjp_cache: Optional[Dict[str, Any]] = None
        self._continuity_cache: Dict[tuple, np.ndarray] = {}
        self._last_base_geom_state: Optional[np.ndarray] = None
        self._partial_reuse_active: bool = False
        # When ``True``, :meth:`B_at_points` skips the JAX free-DOF runner
        # even if ``_has_free_center_or_quat_dofs()`` reports True, and
        # routes through :meth:`_B_at_points_tf_only` instead.  Toggled by
        # the :meth:`force_tf_only_forward` context manager so callers
        # (e.g. ``stellcoilbench_dipoles.coil_optimization
        # ._bulk_center_parameterization.ConstrainedJFWrapper`` running its
        # reduced-DOF FD loop) can cheaply re-evaluate ``B(pts)`` after a
        # one-puck DOF perturbation without paying the ~22 s/call wall of
        # the free-DOF JAX forward.  Default ``False`` so the existing
        # JAX free-DOF path remains the default for free-centre /
        # free-quat optimisations.
        self._force_tf_only_forward: bool = False
        # Stage C partial-L cache: the last full L_base together with the
        # K / pts / weights / signs that produced it.  When
        # ``_partial_reuse_active`` is true on a subsequent rebuild *and*
        # the cached topology (n_base, nd_per, n_all, base_indices,
        # signs, G) matches the current geometry, only the rows / cols
        # corresponding to ``changed_pucks`` are recomputed; the
        # remaining blocks are copied from ``_L_base_prev``.  See
        # :meth:`_maybe_partial_update_L_base_reduced` and
        # :meth:`_maybe_partial_update_L_base_full` for the exact logic.
        self._L_base_prev: Optional[np.ndarray] = None
        self._partial_L_cache: Optional[Dict[str, Any]] = None
        # Bound the drift introduced by repeated incremental updates by
        # forcing a full rebuild every ``_full_rebuild_period`` calls.
        # Default ``50`` matches the appendix's suggested cadence.
        self._full_rebuild_period: int = int(
            os.environ.get("SIMSOPT_PSC_FULL_REBUILD_PERIOD", "50")
        )
        self._partial_L_call_count: int = 0
        # Fixed-puck linear map cache.  When all puck geometry DOFs are fixed,
        # the BS-to-plasma step of :meth:`_B_at_points_tf_only` is a
        # time-invariant linear operator ``M_field`` mapping ``beta_all``
        # (the n_dof_all-vector of modal coefficients) to ``B`` at the
        # current evaluation points.  The cheap half of the chain
        # (``Bn -> beta_work -> beta_all``) stays in the existing JAX
        # ``_beta_from_bn_jitted`` kernel, so we only need to precompute
        # and cache the ``(3 * N_eval, n_dof_all)`` matrix here.  That is
        # ~3x smaller than the previous full ``T`` (``Bn -> B``) and lets
        # the per-call gemv be bandwidth-friendly.
        self._fixed_bulk_operator_enabled: bool = False
        self._M_field: Optional[np.ndarray] = None
        self._M_field_key: Optional[tuple] = None
        # Monotone counter bumped by :meth:`PassiveBulkField.set_points_cart`.
        # The owning :class:`PassiveBulkField` reseats the C++ points
        # buffer when callers (or the parent ``MagneticFieldSum``) push
        # new eval points; bumping this version guarantees the
        # ``_M_field`` cache is invalidated on the very next forward,
        # even when the underlying buffer happens to reuse the same
        # address.
        self._eval_pts_version: int = 0
        # Reduced-free moving-subset cache key (Phase-5 audit L4 hook); cleared on rebuild.
        self._reduced_movingsub_pts_uid: Optional[int] = None
        # Detect whether the TF coil set is a faithful image of its first
        # ``n_base`` entries under the ``(nfp, stellsym)`` group used to
        # replicate pucks.  When ``True``, ``_rebuild`` routes through the
        # symmetry-reduced inductance + solve path (``|G|`` reduction in
        # assembly work and ``|G|^2`` reduction in peak L storage);
        # otherwise the full ``n_dof_total``-sized system is assembled.
        self._tf_is_symmetric = self._detect_tf_symmetry(
            self.coils_TF,
            self.nfp,
            self.stellsym,
        )
        self._rebuild()
        self._last_base_geom_state = self._base_geom_state_matrix()
        _maybe_prime_psc_jax_kernels(self)
        self._field = PassiveBulkField(self)

    # ------------------------------------------------------------------
    # Optimizable overrides
    # ------------------------------------------------------------------

    @staticmethod
    def _is_zero_vjp_dof(name: str) -> bool:
        """Return ``True`` for DOFs whose VJP is identically zero.

        ``R{i}`` and ``t{i}`` are now exposed via a finite-difference
        VJP (see :meth:`_Rt_fd_gradient` and the FD branch of
        :meth:`_vjp_puck_geometry`).  No PSCBulkArray DOF is currently
        zero-VJP; the helper is retained as the policy hook for any
        future zero-gradient DOFs that need to suppress
        :meth:`local_unfix_all` warnings.
        """
        return False

    def _warn_zero_vjp(self, names) -> None:
        """Emit a :class:`UserWarning` if any ``names`` has zero VJP."""
        import warnings

        zero = [n for n in names if self._is_zero_vjp_dof(n)]
        if zero:
            warnings.warn(
                "PSCBulkArray: unfixing DOF(s) "
                f"{zero} whose gradient is not implemented (zero VJP). "
                "Optimizers will see no gradient signal for these "
                "variables; fix them unless you are deliberately "
                "exploring with a gradient-free method.",
                stacklevel=3,
            )

    def _normal_offset_shadowed_names(self) -> frozenset:
        """``center_x/y/z{i}`` names permanently shadowed by a ``d{i}`` DOF.

        Unlike a zero-VJP DOF (see :meth:`_is_zero_vjp_dof`), these are not
        merely gradient-less: :meth:`_get_base_puck_geometry` unconditionally
        overwrites ``centers[i]`` from ``anchor_i + d_i*normal_i`` every
        rebuild, so any value an optimizer wrote into these three DOFs
        would be silently discarded. Unfixing them is therefore a hard
        error rather than a warning.
        """
        names: set = set()
        for i in self._normal_offset_puck_indices:
            names.update((f"center_x{i}", f"center_y{i}", f"center_z{i}"))
        return frozenset(names)

    def local_unfix_all(self) -> None:
        """Unfix all local DOFs, warning about zero-VJP R/t DOFs.

        Raises ``ValueError`` if any puck uses the normal-offset
        parameterization: there is no valid "unfix literally everything"
        state once ``center_x/y/z{i}`` DOFs are permanently shadowed by a
        ``d{i}`` DOF (see :meth:`_normal_offset_shadowed_names`); unfix the
        desired DOFs individually via :meth:`unfix` instead.
        """
        if self._normal_offset_puck_indices:
            raise ValueError(
                "PSCBulkArray.local_unfix_all(): this array has "
                f"{len(self._normal_offset_puck_indices)} normal-offset "
                "puck(s) whose center_x/y/z DOFs are permanently shadowed "
                "by a d{i} DOF and cannot be unfixed; call unfix() on the "
                "individual DOFs you actually want to free instead."
            )
        full_names = list(self.local_full_dof_names)
        free_flags = np.asarray(self._dofs._free, dtype=bool)
        currently_fixed = [n for n, f in zip(full_names, free_flags) if not bool(f)]
        self._warn_zero_vjp(currently_fixed)
        super().local_unfix_all()

    def unfix(self, key) -> None:
        """Unfix a DOF by name or index, warning if its VJP is zero.

        Raises ``ValueError`` if ``key`` names a ``center_x/y/z{i}`` DOF
        shadowed by a normal-offset ``d{i}`` DOF (see
        :meth:`_normal_offset_shadowed_names`).
        """
        if isinstance(key, str):
            name = key
        else:
            try:
                name = list(self.local_full_dof_names)[int(key)]
            except Exception:
                name = None
        if name is not None and name in self._normal_offset_shadowed_names():
            raise ValueError(
                f"PSCBulkArray.unfix({name!r}): this DOF is permanently "
                "shadowed by a normal-offset d{i} DOF on the same puck "
                "(center_i = anchor_i + d_i*normal_i is recomputed every "
                "rebuild, discarding any value written here); unfix the "
                "corresponding d{i} DOF instead."
            )
        if name is not None:
            self._warn_zero_vjp([name])
        super().unfix(key)

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def _get_base_puck_geometry(self):
        """Extract ``(centers, quats, radii, thicknesses)`` from current DOFs."""
        x = np.array(self.local_full_x)
        n = self._n_base_pucks
        centers = np.zeros((n, 3))
        quats = np.zeros((n, 4))
        radii = np.zeros(n)
        thicknesses = np.zeros(n)
        for i in range(n):
            off = i * _DOFS_PER_PUCK
            centers[i] = x[off : off + 3]
            quats[i] = x[off + 3 : off + 7]
            radii[i] = x[off + 7]
            thicknesses[i] = x[off + 8]
        # Normal-offset pucks: override the raw (unused/shadowed)
        # center_x/y/z storage with ``anchor + d * normal``, ``d`` being
        # the puck's ``d{i}`` DOF value.  This is the single choke point
        # every downstream consumer of puck geometry goes through, so
        # patching it here propagates correctness everywhere.
        for i in self._normal_offset_puck_indices:
            d_val = x[self._normal_offset_dof_index[i]]
            centers[i] = self._normal_offset_anchor[i] + d_val * self._normal_offset_normal[i]
        return centers, quats, radii, thicknesses

    def _base_geom_state_matrix(self) -> np.ndarray:
        """Return base-puck geometry DOFs as a contiguous ``(n_base, 9)`` matrix.

        Returns a fresh copy so callers (e.g.
        :meth:`recompute_currents`) can safely store the snapshot for
        change-detection without aliasing the live DOF storage that
        backs :attr:`local_full_x`.  Without the explicit copy, storing
        the returned array as ``_last_base_geom_state`` would silently
        compare a live view against itself on the next call and report
        zero changed pucks, which silently disables the partial-L
        rebuild path.

        Normal-offset pucks (see :meth:`_get_base_puck_geometry`) have
        their ``center_x/y/z{i}`` columns overridden in-place from
        ``anchor + d{i}*normal`` so that a ``d{i}``-only change is still
        visible to row-wise change detection -- the raw stored
        ``center_x/y/z{i}`` values never change for these pucks, only
        the trailing ``d{i}`` DOF does.  Applied as a small local patch
        (not via :meth:`_get_base_puck_geometry`) to keep this function's
        existing O(1) reshape+copy cost for every array without
        normal-offset pucks; this is called on essentially every
        :meth:`recompute_currents`, including pure-TF-motion steps.
        """
        mat = np.ascontiguousarray(
            np.asarray(self.local_full_x, dtype=float)[
                : self._n_base_pucks * _DOFS_PER_PUCK
            ]
        ).reshape(self._n_base_pucks, 9).copy()
        if self._normal_offset_puck_indices:
            x = self.local_full_x
            for i in self._normal_offset_puck_indices:
                d_val = x[self._normal_offset_dof_index[i]]
                mat[i, 0:3] = (
                    self._normal_offset_anchor[i]
                    + d_val * self._normal_offset_normal[i]
                )
        return mat

    def _replicate_pucks(self, centers, quats, radii, thicknesses):
        """Apply nfp + stellsym to base pucks.

        The per-replica ``sign`` in the returned ``replica_signs`` encodes
        the induced-current parity under the symmetry group: ``+1`` for
        pure :math:`R_z(\\alpha_k)` rotations, ``-1`` for stellsym images
        (which simsopt models as a proper :math:`R_x(\\pi)` rotation
        combined with a coil current sign flip, producing
        ``K_image(x) = -M K_base(M^{-1} x)`` and hence
        ``beta^image = -beta^base`` in the local Fourier-Zernike /
        Fourier-Chebyshev basis; see the module docstring of
        :func:`_fold_Bn_to_work` and Phase 2 of the signed-orbit plan).

        Returns:
            all_pucks: list of ``(center, axis, R, t)``
            all_quats: list of quaternions per puck
            base_indices: which base puck each copy came from
            center_jacobians: 3x3 transform from base center to copy center
            quat_jacobians: 4x4 transform from base quaternion to copy quaternion
            replica_signs: ``int8`` array of length ``n_all`` with
                entries in ``{+1, -1}``; ``+1`` for pure-rotation replicas
                and ``-1`` for stellsym-image replicas.
        """
        all_pucks: List[Tuple[np.ndarray, np.ndarray, float, float]] = []
        all_quats_list: List[np.ndarray] = []
        base_indices: List[int] = []
        center_jacobians: List[np.ndarray] = []
        quat_jacobians: List[np.ndarray] = []
        replica_signs: List[int] = []

        for i in range(len(centers)):
            c = centers[i]
            q = quats[i]
            R_val, t_val = float(radii[i]), float(thicknesses[i])
            for jfp in range(self.nfp):
                angle = 2.0 * np.pi * jfp / self.nfp
                rot3 = np.array(
                    [
                        [np.cos(angle), -np.sin(angle), 0.0],
                        [np.sin(angle), np.cos(angle), 0.0],
                        [0.0, 0.0, 1.0],
                    ]
                )
                q_rot = np.array([np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)])

                c2 = rot3 @ c
                q2 = _quat_multiply(q_rot, q)
                ax2 = _quaternion_to_axis(q2)
                all_pucks.append((c2, ax2, R_val, t_val))
                all_quats_list.append(q2)
                base_indices.append(i)
                center_jacobians.append(rot3)
                quat_jacobians.append(_quat_left_mult_matrix(q_rot))
                replica_signs.append(+1)

                if self.stellsym:
                    S = np.diag([1.0, -1.0, -1.0])
                    q_stell = np.array([0.0, 1.0, 0.0, 0.0])
                    c3 = S @ c2
                    q3 = _quat_multiply(q_stell, q2)
                    ax3 = _quaternion_to_axis(q3)
                    all_pucks.append((c3, ax3, R_val, t_val))
                    all_quats_list.append(q3)
                    base_indices.append(i)
                    center_jacobians.append(S @ rot3)
                    quat_jacobians.append(
                        _quat_left_mult_matrix(q_stell) @ _quat_left_mult_matrix(q_rot)
                    )
                    replica_signs.append(-1)

        return (
            all_pucks,
            all_quats_list,
            base_indices,
            center_jacobians,
            quat_jacobians,
            np.asarray(replica_signs, dtype=np.int8),
        )

    @staticmethod
    def _detect_tf_symmetry(
        coils_TF,
        nfp: int,
        stellsym: bool,
        atol: float = 1e-8,
    ) -> bool:
        """Return ``True`` iff ``coils_TF`` is the image of its first
        ``n_total / |G|`` entries under the ``(nfp, stellsym)`` group.

        The comparison uses the canonical simsopt ordering produced by
        :func:`simsopt.field.coil.coils_via_symmetries`, whose helper
        :func:`apply_symmetries_to_curves` iterates
        ``for k in range(nfp): for flip in flip_list: for i in range(n_base): ...``
        (field-period outer, stellsym-flip middle, base-coil inner).  The
        previous implementation iterated in the order used by
        :meth:`_replicate_pucks` (``i`` outer, ``k`` middle, ``flip``
        inner) which only coincides with the canonical simsopt ordering
        when ``n_base == 1``; any multi-coil TF set produced by
        :func:`coils_via_symmetries` (for example the
        ``SchuettHennebergQAnfp2`` layout used in
        ``passive_bulks_toroidal_shell_optimization.py``) silently
        failed detection and dropped to the expensive full-``L`` path.

        Detection is by direct geometric comparison of
        ``(gamma, gammadash, current)`` between each replica in
        ``coils_TF`` and the deterministically generated image; any
        mismatch (including coil count not divisible by ``|G|``) forces a
        conservative ``False`` so the full-``L`` fallback is used.

        Side-effect free: evaluates ``gamma()`` / ``gammadash()`` /
        ``current.get_value()`` only once per input coil and does not
        store any references beyond this check.
        """
        G = int(nfp) * (2 if stellsym else 1)
        n_total = len(coils_TF)
        if n_total == 0 or G <= 1 or n_total % G != 0:
            return False
        n_base = n_total // G
        # Pull base-coil geometry/currents once; keeping this side-effect
        # free avoids re-entering :meth:`coils_via_symmetries` just to
        # perform a detection pass.
        try:
            base_gammas = [np.asarray(coils_TF[i].curve.gamma()) for i in range(n_base)]
            base_gammadashs = [
                np.asarray(coils_TF[i].curve.gammadash()) for i in range(n_base)
            ]
            base_currents = [
                float(coils_TF[i].current.get_value()) for i in range(n_base)
            ]
        except Exception:
            return False

        # Simsopt-canonical loop ordering (``coils_via_symmetries``):
        #   k outer (field periods), flip middle (stellsym image), i inner
        # (base-coil index).
        idx = 0
        for k in range(int(nfp)):
            angle = 2.0 * np.pi * k / int(nfp)
            rot3 = np.array(
                [
                    [np.cos(angle), -np.sin(angle), 0.0],
                    [np.sin(angle), np.cos(angle), 0.0],
                    [0.0, 0.0, 1.0],
                ]
            )
            flip_list = (False, True) if stellsym else (False,)
            for flip in flip_list:
                if flip:
                    S = np.diag([1.0, -1.0, -1.0])
                    I_sign = -1.0
                else:
                    S = np.eye(3)
                    I_sign = 1.0
                M = S @ rot3
                for i_base in range(n_base):
                    g_pred = (M @ base_gammas[i_base].T).T
                    gd_pred = (M @ base_gammadashs[i_base].T).T
                    I_pred = I_sign * base_currents[i_base]
                    if idx >= n_total:
                        return False
                    try:
                        g_obs = np.asarray(coils_TF[idx].curve.gamma())
                        gd_obs = np.asarray(coils_TF[idx].curve.gammadash())
                        I_obs = float(coils_TF[idx].current.get_value())
                    except Exception:
                        return False
                    # Tight pointwise comparison: the simsopt
                    # ``RotatedCurve`` replicas produced by
                    # :func:`coils_via_symmetries` match without any
                    # cyclic parameter shift.
                    if (
                        g_obs.shape != g_pred.shape
                        or gd_obs.shape != gd_pred.shape
                        or not np.allclose(g_obs, g_pred, atol=atol)
                        or not np.allclose(gd_obs, gd_pred, atol=atol)
                        or not np.isclose(I_obs, I_pred, atol=atol)
                    ):
                        return False
                    idx += 1
        return idx == n_total

    # ------------------------------------------------------------------
    # Build / rebuild
    # ------------------------------------------------------------------

    def _build_rim_continuity_projector(
        self,
        n_dof_total: int,
        puck_subset: "list | None" = None,
    ) -> np.ndarray:
        r"""Build the orthonormal basis of the rim-continuity subspace.

        For each puck :math:`i`, :func:`build_continuity_constraint` returns a
        matrix :math:`C_i` of shape ``(2 * n_phi_rim, n_dof_i)`` whose null
        space encodes potentials that are continuous across the top and
        bottom rim curves (:math:`H^1(\\Sigma_i)/\\mathbb R`, paper eq 55
        and the text immediately after eq 57).  These per-puck constraints
        are assembled block-diagonally in the global DOF ordering.

        The returned projector :math:`Q_c` is a tall orthonormal matrix of
        shape ``(n_dof_total, n_free)`` whose columns span ``ker(C)``.  It
        is computed block-by-block (one SVD per puck) which keeps the cost
        linear in ``n_pucks``.

        If any block has zero kernel (degenerate basis at very low order)
        the behavior depends on ``self._strict_rim_continuity``:

        * if ``False`` (default), the whole projector falls back to the
          identity with a one-time :class:`UserWarning`, preserving
          backward compatibility with low-order smoke tests;
        * if ``True``, a :class:`ValueError` is raised so that production
          optimizations cannot silently run without continuity.

        Args:
            n_dof_total: global DOF count (sum of ``n_dof_i``).
            puck_subset: Optional explicit list of replica indices to use
                when assembling blocks.  Defaults to every replica
                (``range(len(self._basis_per_puck))``).  When the
                symmetry-reduced path is active this is set to the base
                replicas only, producing a much smaller ``Q_c_base`` with
                one block per *base* puck (shape
                ``(n_base * nd_per, n_free_base)``).

        Returns:
            ``Q_c`` of shape ``(n_dof_total, n_free)`` with orthonormal
            columns (``Q_c.T @ Q_c = I``).
        """
        import warnings

        if puck_subset is None:
            puck_subset = list(range(len(self._basis_per_puck)))
        use_cont_cache = _psc_continuity_cache_env() or bool(
            getattr(self, "_partial_reuse_active", False)
        )
        if use_cont_cache:
            cache_key = (
                int(n_dof_total),
                tuple(int(p) for p in puck_subset),
                int(self._n_phi_rim),
                bool(getattr(self, "_strict_rim_continuity", False)),
            )
            cached_qc = self._continuity_cache.get(cache_key)
            if cached_qc is not None:
                return np.asarray(cached_qc, dtype=float, order="C").copy()
        subset_dof_offsets: List[int] = []
        off = 0
        for p in puck_subset:
            subset_dof_offsets.append(off)
            off += self._basis_per_puck[p].k_basis_local.shape[1]

        Q_c_blocks: List[np.ndarray] = []
        row_offsets: List[int] = []
        col_offsets: List[int] = []
        n_free_total = 0
        degenerate = False
        for k, pidx in enumerate(puck_subset):
            basis = self._basis_per_puck[pidx]
            _, _, R_p, t_p = self._all_pucks[pidx]
            C_p = build_continuity_constraint(
                basis,
                float(R_p),
                float(t_p),
                n_rim=self._n_phi_rim,
            )
            nd_p = C_p.shape[1]
            _, sv, Vt = np.linalg.svd(C_p, full_matrices=True)
            rtol = 1e-10 * (float(sv[0]) if sv.size > 0 else 1.0)
            n_nonzero = int(np.sum(sv > rtol))
            n_free_p = nd_p - n_nonzero
            if n_free_p <= 0:
                degenerate = True
                break
            Q_p = Vt[n_nonzero:].T
            Q_c_blocks.append(Q_p)
            row_offsets.append(subset_dof_offsets[k])
            col_offsets.append(n_free_total)
            n_free_total += n_free_p

        if degenerate or n_free_total == 0:
            msg = (
                "PSCBulkArray: rim-continuity constraint has zero-rank "
                "kernel for at least one puck; the raw basis would be "
                "used (no rim continuity enforced).  Increase "
                "m_fourier / l_zernike / k_chebyshev to use continuity."
            )
            if getattr(self, "_strict_rim_continuity", False):
                raise ValueError(msg + "  strict_rim_continuity=True was requested.")
            warnings.warn(msg, stacklevel=3)
            return np.eye(n_dof_total)

        Q_c = np.zeros((n_dof_total, n_free_total))
        for k, Q_p in enumerate(Q_c_blocks):
            r0 = row_offsets[k]
            r1 = r0 + Q_p.shape[0]
            c0 = col_offsets[k]
            c1 = c0 + Q_p.shape[1]
            Q_c[r0:r1, c0:c1] = Q_p
        if use_cont_cache:
            self._continuity_cache[cache_key] = np.asarray(Q_c, dtype=float, order="C")
        return Q_c

    def _mark_phase_start(self) -> Optional[float]:
        """Start wall-clock slice for :meth:`_rebuild` when timing is enabled."""
        return time.perf_counter() if _psc_bulk_timing_enabled() else None

    def _mark_phase_end(
        self, name: str, t0: Optional[float], value: Any = None
    ) -> None:
        """Append ``{phase, seconds}`` to :attr:`_timing_rows`."""
        if t0 is None:
            return
        _timing_block_until_ready(value)
        if self._timing_rows is None:
            self._timing_rows = []
        self._timing_rows.append(
            {"phase": name, "seconds": float(time.perf_counter() - t0)}
        )

    def get_timing_history(self) -> List[Dict[str, Any]]:
        """Return a deep copy of the cross-rebuild timing history.

        When ``SIMSOPT_PSCBULK_TIMING=1`` is set, every row appended to
        :attr:`_timing_rows` (during :meth:`_rebuild` and any subsequent
        JAX forward / VJP call before the next rebuild) is migrated into
        :attr:`_timing_history` at the start of the next rebuild, tagged
        with a monotonically increasing ``rebuild_idx``.  The rows still
        sitting in :attr:`_timing_rows` at the moment this method is
        called (i.e. those from the most recent rebuild and any JAX
        phases since) are included in the returned list as well, so
        callers always get a complete picture without having to force an
        extra rebuild.

        Returns
        -------
        list of dict
            Independent copies of the recorded phase rows.  Each row has
            at least ``"phase"`` and either ``"seconds"`` or
            ``"n_pairs"`` / metadata keys, plus ``"rebuild_idx"`` when
            populated automatically.  When timing is disabled the list
            is empty.
        """
        history: List[Dict[str, Any]] = [dict(row) for row in self._timing_history]
        if self._timing_rows:
            current_idx = int(self._timing_rebuild_counter)
            for row in self._timing_rows:
                tagged = dict(row)
                tagged.setdefault("rebuild_idx", current_idx)
                history.append(tagged)
        return history

    def reset_timing_history(self) -> None:
        """Clear :attr:`_timing_history` and the live :attr:`_timing_rows` buffer.

        Useful for tests that want to assert on a fresh, well-defined
        slice of phase rows without carrying state from earlier
        rebuilds.  The ``rebuild_idx`` counter is also reset to ``0`` so
        the next rebuild starts a fresh series.
        """
        self._timing_history = []
        self._timing_rows = None
        self._timing_rebuild_counter = 0

    def _clear_reduced_free_b_aux_cache(self) -> None:
        """Clear reduced-free forward aux state on any non-TF-only rebuild."""
        self._reduced_movingsub_pts_uid = None

    def compatible_fc_warm_handoff(self, prev: Optional["PSCBulkArray"]) -> bool:
        """Return ``True`` if ``prev`` is safe to steal factorisation caches from (L11)."""
        if prev is None:
            return False
        if (
            int(getattr(prev, "_n_base_pucks", -1))
            != int(getattr(self, "_n_base_pucks", -2))
            or int(getattr(prev, "_symmetry_G", -1))
            != int(getattr(self, "_symmetry_G", -2))
        ):
            return False
        if float(prev.regularization_delta) != float(self.regularization_delta):
            return False
        if float(getattr(prev, "_eigenfloor_threshold", 0.0)) != float(
            getattr(self, "_eigenfloor_threshold", 0.0)
        ):
            return False
        ext_prev = getattr(prev, "_get_base_puck_geometry", None)
        ext_cur = getattr(self, "_get_base_puck_geometry", None)
        if not callable(ext_prev) or not callable(ext_cur):
            return False
        c_pr, q_pr, r_pr, t_pr = ext_prev()
        c_cu, q_cu, r_cu, t_cu = ext_cur()
        if not np.allclose(np.asarray(c_pr), np.asarray(c_cu), rtol=0.0, atol=1e-11):
            return False
        if not np.allclose(np.asarray(q_pr), np.asarray(q_cu), rtol=0.0, atol=1e-11):
            return False
        if not np.allclose(np.asarray(r_pr), np.asarray(r_cu), rtol=0.0, atol=1e-11):
            return False
        if not np.allclose(np.asarray(t_pr), np.asarray(t_cu), rtol=0.0, atol=1e-11):
            return False
        if getattr(prev, "_L_red", None) is None or getattr(prev, "_jax_Lr_chol", None) is None:
            return False
        return True

    def compatible_dipole_warm_handoff(self, prev: Optional["PSCBulkArray"]) -> bool:
        """Return ``True`` if ``prev`` dipole JIT slots may be reused on ``self``.

        Requires identical reduced problem shape (base puck count, eval grid,
        symmetry, TF coil count) and identical TF quadrature layout on the
        first TF coil (``gamma()`` shape matches the dipole field/JIT keys).
        """
        if prev is None:
            return False
        if str(getattr(prev, "solver_mode", "")) != "dipole":
            return False
        if str(getattr(self, "solver_mode", "")) != "dipole":
            return False
        if int(getattr(prev, "_n_base_pucks", -1)) != int(
            getattr(self, "_n_base_pucks", -2)
        ):
            return False
        ep_prev = np.asarray(getattr(prev, "eval_points", []))
        ep_self = np.asarray(getattr(self, "eval_points", []))
        if ep_prev.shape != ep_self.shape:
            return False
        if int(getattr(prev, "nfp", -1)) != int(getattr(self, "nfp", -2)):
            return False
        if bool(getattr(prev, "stellsym", False)) != bool(
            getattr(self, "stellsym", False)
        ):
            return False
        tfp = getattr(prev, "coils_TF", None) or []
        tfc = getattr(self, "coils_TF", None) or []
        if len(tfp) != len(tfc):
            return False
        if len(tfc) == 0:
            return True
        g0 = np.asarray(tfp[0].curve.gamma())
        g1 = np.asarray(tfc[0].curve.gamma())
        return g0.shape == g1.shape

    def warm_handoff_from(self, prev: "PSCBulkArray") -> None:
        """Copy PSC factorisation / inductance workspaces from compatible ``prev`` (L11)."""
        jax_attrs = (
            "_jax_Lr_chol",
            "_jax_Lr_eigf_chol",
            "_jax_Q",
            "_jax_L_red_eig_U",
            "_jax_L_red_eig_lam",
        )
        host_np_attrs = (
            "_L_red",
            "_L_work",
            "_Q",
            "_Lr_chol_host",
            "_Lr_eigf_chol_host",
            "_L_red_eig_U_np",
            "_L_red_eig_lam_np",
            "_L_red_eig_anchor_np",
        )
        misc_attrs = (
            "_L_base_prev",
            "_geom_versions",
            "_geom_hash",
            "_puck_dofs_hash_at_rebuild",
            "_partial_L_cache",
        )
        for attr in jax_attrs + host_np_attrs + misc_attrs:
            if hasattr(prev, attr):
                setattr(self, attr, getattr(prev, attr))

        dipole_jit_attrs = (
            "_dipole_jit_rebuild_key",
            "_dipole_jit_assemble_L",
            "_dipole_jit_assemble_f",
            "_dipole_jit_field_key",
            "_dipole_jit_field",
            "_dipole_jit_key",
            "_dipole_jit_forward",
            "_dipole_jit_vjp_key",
            "_dipole_jit_vjp_B",
            "_dipole_jit_vjp_f",
            "_dipole_jit_vjp_L",
            "_dipole_jit_vjp_fused_tf",
        )
        if (
            str(getattr(prev, "solver_mode", "")) == "dipole"
            and str(getattr(self, "solver_mode", "")) == "dipole"
        ):
            for attr in dipole_jit_attrs:
                if hasattr(prev, attr):
                    setattr(self, attr, getattr(prev, attr))

    def try_warm_handoff_from_prev(self, prev: Optional["PSCBulkArray"]) -> bool:
        """Best-effort JAX / host-factor reuse across FC orders when geometries match."""
        if prev is None or not _fc_bulk_warm_handoff_env(self.solver_mode):
            return False
        if self.compatible_dipole_warm_handoff(prev):
            self.warm_handoff_from(prev)
            return True
        if not self.compatible_fc_warm_handoff(prev):
            return False
        Lp = getattr(prev, "_L_red", None)
        Lc = getattr(self, "_L_red", None)
        if Lp is None or Lc is None or np.asarray(Lp).shape != np.asarray(Lc).shape:
            return False
        n0 = float(np.linalg.norm(np.asarray(Lp), ord="fro")) + 1e-300
        rel = float(np.linalg.norm(np.asarray(Lc) - np.asarray(Lp), ord="fro") / n0)
        if rel > 1e-11:
            return False
        self.warm_handoff_from(prev)
        return True

    def _apply_exact_disc_faces_overlay_reduced(
        self,
        L_base: np.ndarray,
        base_reps: Sequence[int],
        nd_per: int,
    ) -> None:
        """Overlay semi-analytic disc self-blocks on symmetry-reduced ``L_base``.

        Mirrors the ``exact_disc_faces`` branch of :func:`shell_inductance_matrix_blockwise`
        restricted to intra-puck diagonals in the folded base ordering.
        """
        from simsopt.field.disc_self_inductance import assemble_puck_self_L

        n_radial = int(getattr(self, "n_radial_disc", 32))
        all_list = getattr(self, "_all_pucks", []) or []
        basis_list = getattr(self, "_basis_per_puck", []) or []
        n_base_loc = min(int(self._n_base_pucks), len(base_reps))
        for a_idx in range(n_base_loc):
            i_rep = int(base_reps[a_idx])
            if (
                i_rep >= len(all_list)
                or i_rep >= len(basis_list)
                or i_rep < 0
            ):
                continue
            _, _, R_val, t_val = all_list[i_rep]
            basis_i = basis_list[i_rep]
            puck_block = assemble_puck_self_L(
                basis_i,
                float(R_val),
                float(t_val),
                n_radial=n_radial,
            )
            d0 = int(a_idx) * int(nd_per)
            nd_pb = int(puck_block.shape[0])
            mask = ~np.isnan(puck_block)
            sub = L_base[d0 : d0 + nd_pb, d0 : d0 + nd_pb].copy()
            sub[mask] = puck_block[mask]
            L_base[d0 : d0 + nd_pb, d0 : d0 + nd_pb] = 0.5 * (sub + sub.T)

    def _refresh_L_red_eig_host_cache(self, Lr: np.ndarray) -> None:
        """Recompute host NumPy eigensystem of :math:`L_\\text{red}` for eigK cache (W4)."""
        Lr = np.asarray(Lr, dtype=np.float64)
        if Lr.size == 0:
            self._L_red_eig_U_np = None
            self._L_red_eig_lam_np = None
            self._L_red_eig_anchor_np = None
            return
        w, v = np.linalg.eigh(0.5 * (Lr + Lr.T))
        mx = float(np.max(np.abs(w)))
        th = float(self._eigenfloor_threshold) * max(mx, 1e-30)
        self._L_red_eig_lam_np = np.maximum(w, th)
        self._L_red_eig_U_np = v
        self._L_red_eig_anchor_np = Lr.copy()

    @staticmethod
    def _pair_block_inductance(
        K_i: np.ndarray,
        pts_i: np.ndarray,
        w_i: np.ndarray,
        K_j: np.ndarray,
        pts_j: np.ndarray,
        w_j: np.ndarray,
        delta_reg: float,
        adaptive_self_reg: bool,
    ) -> np.ndarray:
        """Single ``(nd_i, nd_j)`` mutual-inductance block.

        Mirrors :func:`shell_inductance_matrix_blockwise`'s inner loop so
        Stage C partial-L updates remain bit-exact with the
        :class:`numpy` fall-back assembly.  Self-symmetrization is the
        caller's responsibility.
        """
        r = pts_i[:, None, :] - pts_j[None, :, :]
        if adaptive_self_reg:
            delta_i = _SELF_REG_COEFF * np.sqrt(w_i)
            delta_j = _SELF_REG_COEFF * np.sqrt(w_j)
            delta_pair = 0.5 * (delta_i[:, None] + delta_j[None, :])
            dist = np.sqrt(np.sum(r * r, axis=-1) + delta_pair * delta_pair)
        else:
            dist = np.sqrt(np.sum(r * r, axis=-1) + float(delta_reg) ** 2)
        dot = np.einsum("iax,jbx->ijab", K_i, K_j)
        kernel = dot / dist[..., None, None]
        return MU0_OVER_4PI * np.einsum("ijab,i,j->ab", kernel, w_i, w_j)

    @staticmethod
    @partial(jax.jit, static_argnames=("delta_reg", "adaptive_self_reg"))
    def _partial_update_mutual_blocks_vmap_row(
        K_i: jnp.ndarray,
        pts_i: jnp.ndarray,
        w_i: jnp.ndarray,
        K_stack: jnp.ndarray,
        pts_stack: jnp.ndarray,
        w_stack: jnp.ndarray,
        delta_reg: float,
        adaptive_self_reg: bool,
    ) -> jnp.ndarray:
        """Mutual-block row ``(n_all, nd, nd)`` for fixed left puck (audit L7)."""

        def one_block(K_j: jnp.ndarray, pts_j: jnp.ndarray, w_j: jnp.ndarray) -> jnp.ndarray:
            r = pts_i[:, None, :] - pts_j[None, :, :]
            if adaptive_self_reg:
                delta_i = _SELF_REG_COEFF * jnp.sqrt(w_i)
                delta_j = _SELF_REG_COEFF * jnp.sqrt(w_j)
                delta_pair = 0.5 * (delta_i[:, None] + delta_j[None, :])
                dist = jnp.sqrt(jnp.sum(r * r, axis=-1) + delta_pair * delta_pair)
            else:
                dist = jnp.sqrt(jnp.sum(r * r, axis=-1) + float(delta_reg) ** 2)
            dot = jnp.einsum("iax,jbx->ijab", K_i, K_j)
            kernel = dot / dist[..., None, None]
            return MU0_OVER_4PI * jnp.einsum("ijab,i,j->ab", kernel, w_i, w_j)

        return jax.vmap(one_block)(K_stack, pts_stack, w_stack)

    def _maybe_partial_update_L_base_reduced(
        self,
        K_per_puck: List[np.ndarray],
        pts_per_puck: List[np.ndarray],
        weights_per_puck: List[np.ndarray],
        base_indices: np.ndarray,
        base_reps: List[int],
        signs: np.ndarray,
        G: int,
        changed_pucks: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Symmetric-reduced partial L_base update; ``None`` if not applicable.

        Returns the updated ``L_base`` (shape ``(n_base * nd_per,) * 2``)
        when every cache invariant is satisfied, otherwise ``None``
        which signals the caller to fall back to the full assembly.
        """
        cache = self._partial_L_cache
        if cache is None or self._L_base_prev is None:
            return None
        n_all = len(K_per_puck)
        if cache.get("kind") != "reduced":
            return None
        if cache.get("n_base") != int(self._n_base_pucks):
            return None
        if cache.get("n_all") != int(n_all):
            return None
        if cache.get("G") != int(G):
            return None
        nd_per = int(K_per_puck[base_reps[0]].shape[1])
        if cache.get("nd_per") != nd_per:
            return None
        if not np.array_equal(
            np.asarray(cache.get("base_indices")),
            np.asarray(base_indices, dtype=int),
        ):
            return None
        if not np.array_equal(
            np.asarray(cache.get("signs")),
            np.asarray(signs, dtype=int),
        ):
            return None
        if not np.array_equal(
            np.asarray(cache.get("base_reps")),
            np.asarray(base_reps, dtype=int),
        ):
            return None
        changed_idx = np.flatnonzero(np.asarray(changed_pucks, dtype=bool))
        if changed_idx.size == 0:
            return self._L_base_prev.copy()
        if changed_idx.size >= self._n_base_pucks:
            return None  # full rebuild is cheaper

        L_base = np.array(self._L_base_prev, dtype=np.float64, copy=True)
        changed_set = set(int(x) for x in changed_idx)
        for i_base in changed_idx:
            ri = int(i_base) * nd_per
            L_base[ri : ri + nd_per, :] = 0.0
            L_base[:, ri : ri + nd_per] = 0.0

        delta_reg = float(self.regularization_delta)
        adaptive = bool(self.adaptive_self_reg)
        use_partial_jax = _psc_partial_l_jax_env()
        k_stack_np: Optional[np.ndarray] = None
        pts_stack_np: Optional[np.ndarray] = None
        w_stack_np: Optional[np.ndarray] = None
        if use_partial_jax:
            k_stack_np = np.stack(
                [np.asarray(K_per_puck[j], dtype=np.float64) for j in range(n_all)],
                axis=0,
            )
            pts_stack_np = np.stack(
                [np.asarray(pts_per_puck[j], dtype=np.float64) for j in range(n_all)],
                axis=0,
            )
            w_stack_np = np.stack(
                [np.asarray(weights_per_puck[j], dtype=np.float64) for j in range(n_all)],
                axis=0,
            )

        sigma_np = np.asarray(signs, dtype=np.int64)
        j_chunk = max(128, min(4096, n_all)) if n_all > 500 else n_all
        for i_base in changed_idx:
            i_rep = int(base_reps[int(i_base)])
            K_i = K_per_puck[i_rep]
            pts_i = pts_per_puck[i_rep]
            w_i = weights_per_puck[i_rep]
            ri = int(i_base) * nd_per

            if use_partial_jax and k_stack_np is not None:
                assert pts_stack_np is not None and w_stack_np is not None
                for j_lo in range(0, n_all, j_chunk):
                    j_hi = min(n_all, j_lo + j_chunk)
                    K_sub = jnp.asarray(k_stack_np[j_lo:j_hi])
                    pts_sub = jnp.asarray(pts_stack_np[j_lo:j_hi])
                    w_sub = jnp.asarray(w_stack_np[j_lo:j_hi])
                    blocks_j = PSCBulkArray._partial_update_mutual_blocks_vmap_row(
                        jnp.asarray(K_i),
                        jnp.asarray(pts_i),
                        jnp.asarray(w_i),
                        K_sub,
                        pts_sub,
                        w_sub,
                        delta_reg,
                        adaptive,
                    )
                    weighted = np.asarray(
                        blocks_j * (float(G) * sigma_np[j_lo:j_hi, None, None]),
                        dtype=np.float64,
                    )
                    for j_loc, j_rep in enumerate(range(j_lo, j_hi)):
                        j_base = int(base_indices[j_rep])
                        rj = j_base * nd_per
                        L_base[ri : ri + nd_per, rj : rj + nd_per] += weighted[j_loc]
                continue

            for j_rep in range(n_all):
                j_base = int(base_indices[j_rep])
                sigma_j = int(sigma_np[j_rep])
                K_j = K_per_puck[j_rep]
                pts_j = pts_per_puck[j_rep]
                w_j = weights_per_puck[j_rep]
                block = self._pair_block_inductance(
                    K_i, pts_i, w_i, K_j, pts_j, w_j, delta_reg, adaptive
                )
                rj = j_base * nd_per
                L_base[ri : ri + nd_per, rj : rj + nd_per] += sigma_j * G * block
        # Mirror the changed rows to the corresponding columns of
        # *unchanged* base pucks (those columns were zeroed above and
        # not refilled by the row-only loop).  Changed-vs-changed
        # cross-blocks are already symmetric because both rows were
        # filled by the loop.
        for i_base in changed_idx:
            ri = int(i_base) * nd_per
            for k_base in range(int(self._n_base_pucks)):
                if int(k_base) in changed_set:
                    continue
                rk = int(k_base) * nd_per
                L_base[rk : rk + nd_per, ri : ri + nd_per] = (
                    L_base[ri : ri + nd_per, rk : rk + nd_per].T
                )
        L_base = 0.5 * (L_base + L_base.T)
        return L_base

    def _store_partial_L_cache(
        self,
        *,
        kind: str,
        L_base: np.ndarray,
        K_per_puck: List[np.ndarray],
        pts_per_puck: List[np.ndarray],
        weights_per_puck: List[np.ndarray],
        base_indices: np.ndarray,
        base_reps: List[int],
        signs: np.ndarray,
        G: int,
        nd_per: int,
        n_all: int,
    ) -> None:
        """Snapshot the inputs that produced ``L_base`` for partial-L reuse."""
        self._L_base_prev = np.asarray(L_base, dtype=np.float64).copy()
        self._partial_L_cache = {
            "kind": str(kind),
            "n_base": int(self._n_base_pucks),
            "n_all": int(n_all),
            "G": int(G),
            "nd_per": int(nd_per),
            "base_indices": np.asarray(base_indices, dtype=int).copy(),
            "base_reps": np.asarray(base_reps, dtype=int).copy(),
            "signs": np.asarray(signs, dtype=int).copy(),
        }
        self._partial_L_call_count = 0

    def _rebuild_dipole(
        self,
        changed_mask: Optional[np.ndarray] = None,
    ) -> None:
        """Dipole-mode rebuild (``solver_mode == 'dipole'``).

        Replaces the sheet-basis pipeline entirely with a tiny
        ``(3 n_base, 3 n_base)`` dense Cholesky solve.  See
        :mod:`simsopt.field._psc_bulk_dipole` for the underlying
        primitives and the lean dipole bulk solver plan for the
        mathematical derivation.

        Args:
            changed_mask: Reserved; matches the signature of
                :meth:`_rebuild`.  In dipole mode the per-puck delta is
                not exploited (the full assembly costs microseconds at
                production scale) so the mask is unused.
        """
        _t_total: Optional[float] = None
        if _psc_bulk_timing_enabled():
            _t_total = time.perf_counter()
        centers, quats, radii, thicknesses = self._get_base_puck_geometry()
        n_base = int(self._n_base_pucks)
        # Phase-D + lean dipole speedup: TF-only short-circuit.  When the
        # caller (typically :meth:`recompute_currents`) reports that no
        # puck DoFs changed since the last rebuild, the only thing that
        # can have moved is a TF current / coil geometry.  ``L_red`` and
        # its Cholesky are puck-geometry-only, so we can reuse them and
        # only re-solve ``m_red`` against a fresh ``f_red``.  This
        # collapses the per-iter cost in free-DoF optimisations
        # (free-centres / free-rot / free-radius) to a ``(3 n_base)``
        # triangular solve.
        if (
            changed_mask is not None
            and self._dipole_L_red is not None
            and self._dipole_L_red_chol is not None
            and self._dipole_self_L_local_cached is not None
        ):
            try:
                cm_arr = np.asarray(changed_mask, dtype=bool)
                if cm_arr.size > 0 and not bool(cm_arr.any()):
                    g_tf_sc, gd_tf_sc, I_tf_sc = self._tf_arrays()
                    self._ensure_dipole_rebuild_jit_cache(g_tf_sc)
                    assert self._dipole_jit_assemble_f is not None
                    # Reuse the device-side puck geometry snapshot
                    # from the previous rebuild (puck DoFs haven't
                    # moved, by construction of this branch).  Saves
                    # a fresh H->D transfer of ``centers`` and
                    # ``quats`` per iter.
                    centers_j = (
                        self._dipole_jdev_centers
                        if self._dipole_jdev_centers is not None
                        else jnp.asarray(centers)
                    )
                    quats_j = (
                        self._dipole_jdev_quats
                        if self._dipole_jdev_quats is not None
                        else jnp.asarray(quats)
                    )
                    f_red_sc = np.asarray(
                        self._dipole_jit_assemble_f(
                            centers_j,
                            quats_j,
                            jnp.asarray(g_tf_sc),
                            jnp.asarray(gd_tf_sc),
                            jnp.asarray(I_tf_sc),
                            int(self.nfp),
                            bool(self.stellsym),
                        ),
                        dtype=np.float64,
                    )
                    c_factor_sc, low_sc = self._dipole_L_red_chol
                    m_red_sc = sp_linalg.cho_solve(
                        (c_factor_sc, low_sc),
                        -f_red_sc,
                        check_finite=False,
                    )
                    self._dipole_m_red = np.asarray(m_red_sc, dtype=np.float64)
                    self._geom_hash = hash(tuple(self.local_full_x))
                    self._puck_dofs_hash_at_rebuild = (
                        self._hash_puck_local_dofs()
                    )
                    # Refresh only the moments snapshot: centres and
                    # quaternions are unchanged in this branch, so
                    # their device-side copies stay valid.
                    self._refresh_dipole_device_cache(
                        centers, quats, moments_only=True
                    )
                    if _t_total is not None and self._timing_rows is not None:
                        self._timing_rows.append(
                            {
                                "phase": "rebuild_dipole_short_circuit",
                                "elapsed_s": float(
                                    time.perf_counter() - _t_total
                                ),
                            }
                        )
                    return
            except (TypeError, ValueError):
                # Fall through to the full rebuild path below if the
                # ``changed_mask`` cannot be interpreted as a bool array.
                pass
        keys = tuple(
            (float(radii[i]), float(thicknesses[i])) for i in range(n_base)
        )
        need_recompute_self = self._dipole_self_L_local_cached is None
        if not self._dipole_freeze_self_L:
            need_recompute_self = (
                need_recompute_self
                or self._dipole_self_L_keys != keys
            )
        if need_recompute_self:
            self_L_list = []
            for i in range(n_base):
                self_L_list.append(
                    _psc_bulk_dipole_mod.compute_self_inductance_tensor(
                        float(radii[i]),
                        float(thicknesses[i]),
                    )
                )
            self._dipole_self_L_local_cached = np.stack(self_L_list, axis=0)
            self._dipole_self_L_keys = keys
        assert self._dipole_self_L_local_cached is not None  # for type-checkers
        # Assemble the reduced (3 n_base, 3 n_base) inductance matrix.
        # Use the persistent ``jax.jit`` for ``assemble_L`` so repeated
        # rebuilds at fixed ``n_base`` / symmetry skip the tracing
        # overhead.  ``g_tf`` shape is not part of ``assemble_L``'s key
        # but the helper uses it only for ``assemble_f`` -- pre-fetch
        # the TF arrays here so the key is consistent across both
        # jitted kernels.
        g_tf_for_key, gd_tf_for_key, I_tf_for_key = self._tf_arrays()
        self._ensure_dipole_rebuild_jit_cache(g_tf_for_key)
        assert self._dipole_jit_assemble_L is not None
        L_red = np.asarray(
            self._dipole_jit_assemble_L(
                jnp.asarray(centers),
                jnp.asarray(quats),
                jnp.asarray(self._dipole_self_L_local_cached),
                int(self.nfp),
                bool(self.stellsym),
            ),
            dtype=np.float64,
        )
        self._dipole_L_red = L_red
        # Cholesky factor with a tiny relative jitter for stability.  L
        # is SPD up to float noise but stellsym replication can
        # introduce near-zero modes at coincident centres; the floor is
        # at the level of the existing ``_eigenfloor_threshold`` of
        # ``PSCBulkArray``.
        jitter = float(self._eigenfloor_threshold) * (
            float(np.trace(L_red)) / max(1, int(L_red.shape[0]))
        )
        L_reg = L_red + jitter * np.eye(L_red.shape[0])
        try:
            c, lower = sp_linalg.cho_factor(L_reg, lower=True, check_finite=False)
            self._dipole_L_red_chol = (np.asarray(c), bool(lower))
            # Mirror the lower-triangular Cholesky factor onto the
            # device so the fused TF-only backward kernel can perform
            # ``jax.scipy.linalg.cho_solve`` without a fresh H->D
            # transfer per VJP call.  We deliberately materialise the
            # strictly lower-triangular part so ``cho_solve(..., lower=True)``
            # sees a clean factor (``cho_factor`` overwrites only the
            # used triangle).
            self._dipole_jdev_L_red_chol = jnp.asarray(np.tril(c))
        except sp_linalg.LinAlgError as exc:
            raise RuntimeError(
                "Dipole-mode reduced inductance matrix is not "
                "positive-definite (after eigenfloor jitter)."
            ) from exc
        # Solve for moments using current TF arrays.  Reuse the TF
        # arrays already fetched above for the JIT-key calculation;
        # use the cached jitted ``assemble_f`` for amortised tracing.
        g_tf, gd_tf, I_tf = g_tf_for_key, gd_tf_for_key, I_tf_for_key
        assert self._dipole_jit_assemble_f is not None
        f_red = np.asarray(
            self._dipole_jit_assemble_f(
                jnp.asarray(centers),
                jnp.asarray(quats),
                jnp.asarray(g_tf),
                jnp.asarray(gd_tf),
                jnp.asarray(I_tf),
                int(self.nfp),
                bool(self.stellsym),
            ),
            dtype=np.float64,
        )
        c_factor, low = self._dipole_L_red_chol
        m_red = sp_linalg.cho_solve(
            (c_factor, low), -f_red, check_finite=False
        )
        self._dipole_m_red = np.asarray(m_red, dtype=np.float64)
        # Update geometry-state caches so ``recompute_currents`` can
        # detect "no DoFs moved" and short-circuit.
        self._geom_hash = hash(tuple(self.local_full_x))
        self._puck_dofs_hash_at_rebuild = self._hash_puck_local_dofs()
        # Refresh device-side puck-geometry snapshots used by the
        # cached B path and the analytic VJP.
        self._refresh_dipole_device_cache(centers, quats)
        if _t_total is not None:
            elapsed = time.perf_counter() - _t_total
            if self._timing_rows is not None:
                self._timing_rows.append(
                    {"phase": "rebuild_dipole_total", "elapsed_s": float(elapsed)}
                )
        # Close-packing warning on the very first rebuild.
        if not getattr(self, "_dipole_close_pack_warned", False):
            self._dipole_close_pack_warned = True
            threshold = _psc_dipole_warn_r_close_env()
            if threshold > 0.0:
                centers_all, _, _, base_idx_all = (
                    _psc_bulk_dipole_mod._replicate_pucks_dipole(
                        jnp.asarray(centers),
                        jnp.asarray(quats),
                        int(self.nfp),
                        bool(self.stellsym),
                    )
                )
                c_all_np = np.asarray(centers_all)
                base_idx_np = np.asarray(base_idx_all)
                worst_ratio = np.inf
                G_size = int(self.nfp) * (2 if self.stellsym else 1)
                for ii in range(n_base):
                    R_i = float(radii[ii])
                    # ``ii * G_size`` is the identity-replica of base
                    # puck ``ii`` (loop order in
                    # ``_replicate_pucks_dipole``).
                    ii_self_rep = ii * G_size
                    for jj in range(int(c_all_np.shape[0])):
                        if jj == ii_self_rep:
                            continue
                        j_base = int(base_idx_np[jj])
                        R_j = float(radii[j_base])
                        sep = float(np.linalg.norm(c_all_np[jj] - centers[ii]))
                        denom = R_i + R_j
                        if denom > 0.0:
                            worst_ratio = min(worst_ratio, sep / denom)
                if np.isfinite(worst_ratio) and worst_ratio < threshold:
                    import warnings

                    warnings.warn(
                        "PSCBulkArray(solver_mode='dipole'): "
                        f"closest puck pair has separation/(R_i + R_j) "
                        f"= {worst_ratio:.3f} < threshold "
                        f"{threshold:.2f}; dipole truncation error may "
                        f"be larger than acceptable.  Set "
                        "SIMSOPT_PSC_DIPOLE_WARN_R_CLOSE=0 to silence.",
                        UserWarning,
                        stacklevel=2,
                    )

    def _rebuild(
        self,
        changed_mask: Optional[np.ndarray] = None,
    ) -> None:
        """Recompute all derived quantities from current DOFs.

        When the optional attribute ``self.exact_disc_faces`` is set to
        ``True`` (via ``obj.exact_disc_faces = True`` after construction;
        default ``False`` via :func:`getattr`), the flat top/bottom
        disc-face sub-blocks of the inductance matrix ``L`` are
        overwritten with the semi-analytic exact-up-to-1D-quadrature
        values produced by
        :func:`simsopt.field.disc_self_inductance.assemble_puck_disc_faces_L`
        and
        :func:`simsopt.field.disc_self_inductance.disc_disc_cross_block`.
        Side-wall and non-coaxial mutual entries remain on the existing
        regularized numerical path.  The number of radial nodes is
        controlled by the optional attribute ``self.n_radial_disc``
        (default ``32``).

        **Stage 2 incremental-rebuild hook (plan ``psc-scale-to-100-bulks``):**

        When ``changed_mask`` is a boolean array of length
        ``n_base_pucks * 9`` (one entry per local DoF) with no ``True``
        entries, this method skips the full K-basis / ``L_base`` / Cholesky
        rebuild and only re-solves ``beta`` from the current TF arrays.
        This is the "TF-only" fast path: typical optimizer steps that
        only move TF DoFs (coil geometry / currents) leave ``L_base``
        unchanged because the shell inductance kernel is TF-independent.

        A ``changed_mask`` that is ``None`` (default) preserves the
        previous unconditional full-rebuild semantics so existing
        callers behave identically.  When ``SIMSOPT_PSC_PARTIAL_L_REUSE=1``
        and the changed-puck fraction is below
        ``SIMSOPT_PSC_PARTIAL_L_REUSE_THRESHOLD`` (default ``0.3``), a
        conservative incremental path is activated: continuity projectors
        and free-DOF pullback caches may be reused while the reduced solve
        and beta assembly are refreshed from current geometry/TF state.
        """
        if self.solver_mode == "dipole":
            # Dispatch to the dipole-mode rebuild and return early; this
            # path bypasses the entire sheet basis / K-stack / continuity
            # pipeline (Phase A of the lean dipole bulk solver plan).
            self._rebuild_dipole(changed_mask=changed_mask)
            return
        self._partial_reuse_active = False
        changed_pucks_mask: Optional[np.ndarray] = None
        if (
            changed_mask is not None
            and self._L_work is not None
            and self._Q is not None
            and self._jax_Lr_chol is not None
            and self._jax_Lr_eigf_chol is not None
        ):
            try:
                cm_arr = np.asarray(changed_mask, dtype=bool)
                if cm_arr.size == self._n_base_pucks:
                    changed_pucks_mask = cm_arr.copy()
                elif self._n_base_pucks > 0 and cm_arr.size % self._n_base_pucks == 0:
                    changed_pucks_mask = cm_arr.reshape(
                        self._n_base_pucks, -1
                    ).any(axis=1)
                else:
                    changed_pucks_mask = np.array(
                        [bool(cm_arr.any())] * self._n_base_pucks, dtype=bool
                    )
                if cm_arr.size > 0 and not bool(cm_arr.any()):
                    # No DoFs changed: reuse L_base / Q / Cholesky from
                    # the previous rebuild and just resolve beta from the
                    # (possibly updated) TF DoFs.
                    _t_sc = self._mark_phase_start()
                    self.beta = self._solve_beta(self._tf_arrays())
                    self._geom_hash = hash(tuple(self.local_full_x))
                    self._puck_dofs_hash_at_rebuild = self._hash_puck_local_dofs()
                    self._mark_phase_end("rebuild_short_circuit", _t_sc)
                    return
                if (
                    _psc_partial_l_reuse_env()
                    and changed_pucks_mask is not None
                    and changed_pucks_mask.size > 0
                    and bool(changed_pucks_mask.any())
                ):
                    # Periodic full rebuild: bound drift from many
                    # consecutive incremental updates.
                    if (
                        self._partial_L_call_count
                        >= int(self._full_rebuild_period) > 0
                    ):
                        self._partial_reuse_active = False
                    else:
                        changed_fraction = float(np.mean(changed_pucks_mask))
                        if changed_fraction <= _psc_partial_l_reuse_threshold_env():
                            self._partial_reuse_active = True
            except (ValueError, TypeError):
                pass

        self._clear_reduced_free_b_aux_cache()
        _t_rebuild_total0: Optional[float] = None
        if _psc_bulk_timing_enabled():
            # Snapshot any rows accumulated since the previous rebuild
            # (the previous rebuild's phases plus all subsequent JAX
            # forward / VJP phase rows pushed by ``_mark_phase_end``)
            # into ``_timing_history`` with a ``rebuild_idx`` tag, so
            # they survive the reset below.  Without this, only the
            # rows from the final rebuild + post-rebuild JAX calls
            # would be readable after ``optimize_coils`` returns.
            #
            # Counter semantics: ``_timing_rebuild_counter`` is the
            # zero-based index of the rebuild whose rows are currently
            # live in ``_timing_rows`` (or, between rebuilds, of the
            # most recently completed rebuild).  It is *only*
            # incremented when there is something to flush -- i.e. on
            # the second and subsequent rebuilds in a series.  This
            # keeps ``rebuild_idx=0`` for the first rebuild after a
            # fresh start (or after :meth:`reset_timing_history`).
            if self._timing_rows is not None:
                if self._timing_rows:
                    _idx = int(self._timing_rebuild_counter)
                    for _row in self._timing_rows:
                        _tagged = dict(_row)
                        _tagged.setdefault("rebuild_idx", _idx)
                        self._timing_history.append(_tagged)
                self._timing_rebuild_counter += 1
            self._timing_rows = []
            _t_rebuild_total0 = time.perf_counter()

        centers, quats, radii, thicknesses = self._get_base_puck_geometry()
        structural_key = (
            tuple(np.asarray(radii, dtype=float).ravel()),
            tuple(np.asarray(thicknesses, dtype=float).ravel()),
        )
        structural_changed = structural_key != self._structural_key
        self._structural_key = structural_key
        if structural_changed:
            self._local_stacks_valid = False

        (
            all_pucks,
            all_quats_list,
            base_indices,
            center_jacs,
            quat_jacs,
            replica_signs,
        ) = self._replicate_pucks(
            centers,
            quats,
            radii,
            thicknesses,
        )

        self._all_pucks = all_pucks
        self._all_pucks_quats = np.array(all_quats_list)
        self._all_puck_base_indices = base_indices
        self._center_jacobians = center_jacs
        self._quat_jacobians = quat_jacs
        self._replica_signs = np.asarray(replica_signs, dtype=np.int8)

        _t_kbasis = self._mark_phase_start()
        self._basis_per_puck: List[PuckBasisData] = []
        self._dof_offsets: List[int] = []
        K_per_puck_global: List[np.ndarray] = []
        pts_per_puck_global: List[np.ndarray] = []
        weights_per_puck: List[np.ndarray] = []
        normals_per_puck_global: List[np.ndarray] = []
        phi_per_puck: List[np.ndarray] = []

        n_dof_total = 0
        for idx, (c, ax, R_val, t_val) in enumerate(all_pucks):
            basis = build_puck_shell_basis(
                R_val,
                t_val,
                m_fourier=self.m_fourier,
                l_zernike=self.l_zernike,
                k_chebyshev=self.k_chebyshev,
                n_rho=self._n_rho,
                n_phi=self._n_phi,
                n_z=self._n_z,
            )
            if self._mode_truncate_norm is not None:
                basis = apply_mode_truncate(basis, self._mode_truncate_input)
            self._basis_per_puck.append(basis)
            Rmat = _rotation_matrix_from_quat(all_quats_list[idx])
            pts_g = (Rmat @ basis.quad_points_local.T).T + c[None, :]
            K_g = np.einsum("ij,qkj->qki", Rmat, basis.k_basis_local)
            n_g = (Rmat @ basis.quad_normals_local.T).T

            nd = K_g.shape[1]
            self._dof_offsets.append(n_dof_total)
            n_dof_total += nd

            K_per_puck_global.append(K_g)
            pts_per_puck_global.append(pts_g)
            weights_per_puck.append(basis.quad_weights)
            normals_per_puck_global.append(n_g)
            phi_per_puck.append(basis.phi_values)

        self._n_dof_total = n_dof_total
        self._K_per_puck = K_per_puck_global
        self._pts_per_puck = pts_per_puck_global
        self._weights_per_puck_list = weights_per_puck

        # Flat/global quadrature geometry (shared across all code paths).
        quad_points = np.vstack(pts_per_puck_global)
        quad_weights = np.concatenate(weights_per_puck)
        quad_normals = np.vstack(normals_per_puck_global)

        # Per-puck row bookkeeping used for diagnostics and reshape paths.
        row0 = 0
        self._quad_row_ranges: List[Tuple[int, int]] = []
        for pidx in range(len(all_pucks)):
            nq = K_per_puck_global[pidx].shape[0]
            self._quad_row_ranges.append((row0, row0 + nq))
            row0 += nq

        self._quad_points = quad_points
        self._quad_weights = quad_weights
        self._quad_normals = quad_normals
        # Pre-shaped for :meth:`_vjp_tf_only_analytic` (avoid per-call ``reshape``).
        n_all_quads = len(all_pucks)
        self._quad_weights_2d = quad_weights.reshape(n_all_quads, -1)

        # Stacked block-diagonal representation of (K_basis, phi_mat).  In the
        # common case (all pucks share the same basis resolution -- which is
        # always true for PSCBulkArray because m_fourier/l_zernike/...
        # are globals on the array) every per-puck block has the same
        # ``(nq_per, nd_per, 3)`` shape and we simply ``np.stack`` them.  The
        # much larger dense ``(nq_total, n_dof_total, ...)`` arrays are never
        # materialised (they are ~O(n_pucks) * bigger because they are
        # block-diagonal), saving ~95-99% memory on the basis tables at scale.
        # Heterogeneous-shape fallback: if any per-puck block has a different
        # shape, fall back to the dense layout for backward compatibility.
        nq_pers = {arr.shape[0] for arr in K_per_puck_global}
        nd_pers = {arr.shape[1] for arr in K_per_puck_global}
        if len(nq_pers) == 1 and len(nd_pers) == 1:
            self._jax_K_stack = jnp.asarray(np.stack(K_per_puck_global, axis=0))
            self._jax_phi_stack = jnp.asarray(np.stack(phi_per_puck, axis=0))
            self._jax_w_stack = jnp.asarray(np.stack(weights_per_puck, axis=0))
            self._uniform_puck_shape = True
        else:
            # Rare path (mixed shapes): keep dense as padded fallback.  This
            # code path is currently exercised only via synthetic tests.
            self._jax_K_stack = None
            self._jax_phi_stack = None
            self._jax_w_stack = None
            self._uniform_puck_shape = False
        self._mark_phase_end("rebuild_kbasis", _t_kbasis)
        # Dense monolithic arrays are available lazily via the ``_K_basis``
        # and ``_phi_mat`` properties for legacy callers (VTK export,
        # regression tests).  They are not cached here to keep the memory
        # footprint down; each access rebuilds them on demand.

        _t_L = self._mark_phase_start()
        _p_timer: Optional[Callable[[str, int, float], None]] = None
        if _psc_bulk_timing_enabled():
            if self._timing_rows is None:
                self._timing_rows = []

            def _p_timer_cb(phase: str, n_pairs: int, secs: float) -> None:  # noqa: E301
                assert self._timing_rows is not None
                self._timing_rows.append(
                    {
                        "phase": str(phase),
                        "n_pairs": int(n_pairs),
                        "seconds": float(secs),
                    }
                )

            _p_timer = _p_timer_cb
        # --------------------------------------------------------------
        # Inductance-matrix assembly.
        #
        # Two paths share the same downstream solver code:
        # * ``_reduced_active`` path: TF coils exhibit the same
        #   ``(nfp, stellsym)`` symmetry as the puck layout, so we only
        #   build the ``(n_base * nd_per, n_base * nd_per)`` folded matrix
        #   ``L_base`` and the corresponding base-puck rim-continuity
        #   projector; the full ``n_dof_total``-sized ``L`` is never
        #   allocated.
        # * Full path: asymmetric TF or no replication.  Build the dense
        #   ``(n_dof_total, n_dof_total)`` matrix and the full-size
        #   projector, as before.
        # --------------------------------------------------------------
        exact_disc_faces = bool(getattr(self, "exact_disc_faces", False))
        n_all = len(all_pucks)
        G = int(self.nfp) * (2 if self.stellsym else 1)
        # Reduced-path is only useful when there are actual replicas.
        #
        # Stellsym is gated off here intentionally.  Under simsopt's
        # ``coils_via_symmetries`` convention for stellarator symmetry,
        # the image coil carries ``I_sign = -1`` combined with a proper
        # rotation ``R_x(pi)`` of the local frame and position reflection
        # ``S = diag(1, -1, -1)``, which makes the normal component of
        # ``B_TF`` antisymmetric at G-related quadrature points:
        # ``Bn(S x) = -Bn(x)``.  The current unsigned duplication
        # operator ``T`` used by :func:`_fold_Bn_to_work` / :func:`_gather_beta_work_to_all`
        # and :func:`~simsopt.field.bulk_inductance.shell_inductance_matrix_symmetric_reduced`
        # then sums orbits to (near) zero and would silently zero out
        # Signed orbit operator (``sigma_r in {+1, -1}``) is now plumbed
        # through the fold/gather/JIT paths and
        # :func:`shell_inductance_matrix_symmetric_reduced`, so the
        # reduced path is safe to enable under ``stellsym=True`` as well.
        disc_sym_reduced_ok = (
            not exact_disc_faces or _psc_disc_face_sym_reduced_env()
        )
        reduced_active = bool(
            self._tf_is_symmetric
            and G > 1
            and n_all == self._n_base_pucks * G
            and disc_sym_reduced_ok
            and self.solver_mode
            == "energy"  # shell_l2 assembles H densely across pucks
        )
        self._symmetry_G = G
        self._reduced_free_dof_active = bool(
            reduced_active
            and self.solver_mode == "energy"
            and disc_sym_reduced_ok
            and self.use_symmetry_reduced_free_dof
            and not _psc_disable_reduced_free()
        )
        # When ``exact_disc_faces`` is enabled, the default path uses the dense
        # block assembler.  Setting ``SIMSOPT_PSC_DISC_FACE_SYM=1`` keeps the
        # symmetry-reduced ``L_base`` assembly and overlays each base puck's
        # self-block with :func:`~simsopt.field.disc_self_inductance.assemble_puck_self_L`
        # (audit L6, May 2026).
        self._reduced_active = reduced_active

        # Always build the full-size rim-continuity projector: it is used
        # (as ``_Q_c``) by the puck-DOF full-JAX path, which JIT-assembles
        # L internally on every replica and therefore expects a full-size
        # projector.  The full-size ``Q_c`` is cheap to store (block
        # diagonal across pucks, ~``n_pucks * nd_per^2`` bytes).
        Q_c_full = self._build_rim_continuity_projector(n_dof_total)
        self._Q_c = Q_c_full

        # Base-replica indices (first replica of each base puck).  The
        # signed-orbit reduction assumes ``sigma_{i_rep} = +1`` so that
        # ``L_base[i, j] = G * sum_{s in orbit(j)} sigma_s L[i_rep(i), s]``
        # (see :func:`shell_inductance_matrix_symmetric_reduced`).  By
        # construction :meth:`_replicate_pucks` emits every pure-rotation
        # replica *before* its stellsym image, so the first encountered
        # replica of each base puck always carries ``sign = +1``; assert
        # to guard against future reordering.
        seen = [False] * self._n_base_pucks
        base_reps: List[int] = [0] * self._n_base_pucks
        for replica_idx, bi in enumerate(base_indices):
            if not seen[bi]:
                base_reps[bi] = replica_idx
                seen[bi] = True
        self._base_reps = base_reps
        if n_all > 0 and not np.all(self._replica_signs[base_reps] == +1):
            bad = [int(r) for r in base_reps if self._replica_signs[r] != +1]
            raise RuntimeError(
                "Signed-orbit reduction requires every base_reps entry to "
                f"be a pure-rotation replica (sign = +1); got stellsym-image "
                f"replicas at base_reps indices {bad}.  This indicates "
                "_replicate_pucks reordered the emission sequence."
            )

        # Stage 1 disk cache: skip assembly + Cholesky entirely on hit.
        cache_hit_payload: Optional[Dict[str, np.ndarray]] = None
        cache_key: Optional[str] = None
        lcache_enabled = (
            _psc_lcache_enabled()
            and reduced_active
            and not exact_disc_faces
            and not self._has_free_puck_dofs()
        )
        if lcache_enabled:
            try:
                from simsopt import __version__ as _simsopt_ver
            except Exception:  # noqa: BLE001
                _simsopt_ver = "unknown"
            try:
                K_stack_np = (
                    np.stack(K_per_puck_global, axis=0)
                    if self._uniform_puck_shape
                    else None
                )
                cache_key = _psc_lcache_digest(
                    quad_points=quad_points,
                    quad_weights=quad_weights,
                    quad_normals=quad_normals,
                    K_stack=K_stack_np,
                    base_indices=np.asarray(base_indices, dtype=np.int64),
                    replica_signs=self._replica_signs,
                    base_reps=np.asarray(base_reps, dtype=np.int64),
                    delta_reg=self.regularization_delta,
                    adaptive_self_reg=self.adaptive_self_reg,
                    solver_mode=self.solver_mode,
                    G=G,
                    n_base_pucks=int(self._n_base_pucks),
                    nd_per_puck=int(K_per_puck_global[base_reps[0]].shape[1]),
                    null_space_threshold=float(self._null_space_threshold),
                    eigenfloor_threshold=float(self._eigenfloor_threshold),
                    reduced_active=reduced_active,
                    simsopt_version=str(_simsopt_ver),
                    extra=(
                        "SIMSOPT_PSC_FAR_PAIR="
                        + str(os.environ.get("SIMSOPT_PSC_FAR_PAIR", "0"))
                        + "",
                        "SIMSOPT_PSC_R_FAR="
                        + str(os.environ.get("SIMSOPT_PSC_R_FAR", "")),
                        "SIMSOPT_PSC_R_NEAR="
                        + str(os.environ.get("SIMSOPT_PSC_R_NEAR", "2.0")),
                    ),
                )
                cache_hit_payload = _psc_lcache_load(cache_key)
            except Exception:  # noqa: BLE001
                cache_hit_payload = None
        self._lcache_last_key = cache_key
        self._lcache_last_hit = cache_hit_payload is not None

        if cache_hit_payload is not None:
            L_base = np.asarray(cache_hit_payload["L_base"], dtype=np.float64)
            self._L_work = L_base
            nd_per_puck = K_per_puck_global[base_reps[0]].shape[1]
            Q_c_work = np.asarray(cache_hit_payload["Q_c_work"], dtype=np.float64)
            # Treat the disk-cache hit as a fresh full rebuild for the
            # purposes of Stage C partial-L reuse so subsequent steps that
            # only move a puck or two can incrementally update from this
            # baseline instead of paying the full assembly again.
            self._store_partial_L_cache(
                kind="reduced",
                L_base=L_base,
                K_per_puck=K_per_puck_global,
                pts_per_puck=pts_per_puck_global,
                weights_per_puck=weights_per_puck,
                base_indices=np.asarray(base_indices, dtype=int),
                base_reps=base_reps,
                signs=np.asarray(self._replica_signs, dtype=int),
                G=G,
                nd_per=nd_per_puck,
                n_all=n_all,
            )
            self._partial_reuse_active = False
        elif reduced_active:
            partial_L_used = False
            L_base = None
            if (
                self._partial_reuse_active
                and changed_pucks_mask is not None
                and self._L_base_prev is not None
                and self._partial_L_cache is not None
            ):
                L_base_partial = self._maybe_partial_update_L_base_reduced(
                    K_per_puck=K_per_puck_global,
                    pts_per_puck=pts_per_puck_global,
                    weights_per_puck=weights_per_puck,
                    base_indices=np.asarray(base_indices, dtype=int),
                    base_reps=base_reps,
                    signs=np.asarray(self._replica_signs, dtype=int),
                    G=G,
                    changed_pucks=changed_pucks_mask,
                )
                if L_base_partial is not None:
                    L_base = L_base_partial
                    partial_L_used = True
                    self._partial_L_call_count = (
                        int(self._partial_L_call_count) + 1
                    )
            if L_base is None:
                L_base = shell_inductance_matrix_symmetric_reduced(
                    K_per_puck_global,
                    pts_per_puck_global,
                    weights_per_puck,
                    base_indices,
                    delta_reg=self.regularization_delta,
                    adaptive_self_reg=self.adaptive_self_reg,
                    replica_signs=self._replica_signs,
                    pair_class_timer=_p_timer,
                )
                if (
                    exact_disc_faces
                    and _psc_disc_face_sym_reduced_env()
                    and not partial_L_used
                ):
                    self._apply_exact_disc_faces_overlay_reduced(
                        L_base,
                        base_reps,
                        int(K_per_puck_global[base_reps[0]].shape[1]),
                    )
            self._L_work = L_base
            nd_per_puck = K_per_puck_global[base_reps[0]].shape[1]
            Q_c_work = self._build_rim_continuity_projector(
                self._n_base_pucks * nd_per_puck,
                puck_subset=base_reps,
            )
            if not partial_L_used:
                self._store_partial_L_cache(
                    kind="reduced",
                    L_base=L_base,
                    K_per_puck=K_per_puck_global,
                    pts_per_puck=pts_per_puck_global,
                    weights_per_puck=weights_per_puck,
                    base_indices=np.asarray(base_indices, dtype=int),
                    base_reps=base_reps,
                    signs=np.asarray(self._replica_signs, dtype=int),
                    G=G,
                    nd_per=nd_per_puck,
                    n_all=n_all,
                )
            self._partial_reuse_active = bool(partial_L_used)
        else:
            if exact_disc_faces:
                disc_centers_axes = []
                disc_Rts = []
                for idx, (c, ax, R_val, t_val) in enumerate(all_pucks):
                    Rmat = _rotation_matrix_from_quat(all_quats_list[idx])
                    axis_global = Rmat @ np.array([0.0, 0.0, 1.0])
                    disc_centers_axes.append(
                        (
                            np.asarray(c, dtype=float),
                            np.asarray(axis_global, dtype=float),
                        )
                    )
                    disc_Rts.append((float(R_val), float(t_val)))
                L_np = shell_inductance_matrix_blockwise(
                    K_per_puck_global,
                    pts_per_puck_global,
                    weights_per_puck,
                    self._dof_offsets,
                    n_dof_total,
                    delta_reg=self.regularization_delta,
                    adaptive_self_reg=self.adaptive_self_reg,
                    exact_disc_faces=True,
                    disc_bases=self._basis_per_puck,
                    disc_Rts=disc_Rts,
                    disc_centers_axes=disc_centers_axes,
                    n_radial_disc=int(getattr(self, "n_radial_disc", 32)),
                )
            else:
                L_np = shell_inductance_matrix_blockwise(
                    K_per_puck_global,
                    pts_per_puck_global,
                    weights_per_puck,
                    self._dof_offsets,
                    n_dof_total,
                    delta_reg=self.regularization_delta,
                    adaptive_self_reg=self.adaptive_self_reg,
                    pair_class_timer=_p_timer,
                )
            if (
                not exact_disc_faces
                and os.environ.get("SIMSOPT_PSC_COAXIAL_MUTUAL", "0") == "1"
            ):
                from simsopt.field.disc_self_inductance import (
                    fill_coaxial_inter_puck_disc_block,
                )

                nrd = int(getattr(self, "n_radial_disc", 32))
                for i in range(n_all):
                    for j in range(i + 1, n_all):
                        ci, ai, Ri, ti = self._all_pucks[i]
                        cj, aj, Rj, tj = self._all_pucks[j]
                        fill_coaxial_inter_puck_disc_block(
                            L_np,
                            int(self._dof_offsets[i]),
                            int(self._dof_offsets[j]),
                            self._basis_per_puck[i],
                            self._basis_per_puck[j],
                            float(Ri),
                            float(ti),
                            float(Rj),
                            float(tj),
                            np.asarray(ci, dtype=float).ravel()[:3],
                            np.asarray(ai, dtype=float).ravel()[:3],
                            np.asarray(cj, dtype=float).ravel()[:3],
                            np.asarray(aj, dtype=float).ravel()[:3],
                            n_radial=nrd,
                        )
            self._L_work = L_np
            Q_c_work = Q_c_full  # alias -- full path uses the same projector

            # ---- shell_l2 (REGCOIL-style) operator swap --------------
            # Replace the current-potential Gram ``L`` and the
            # corresponding load vector with the L^2-on-shell normal-
            # field operators ``M = H^T diag(w) H`` and
            # ``g = -H^T diag(w) Bn^TF``.  Dense assembly across every
            # puck-pair; the reduced path is already gated off above in
            # this branch.
            if self.solver_mode == "shell_l2":
                K_basis_dense = self._K_basis  # lazy (nq_total, n_dof, 3)
                H = shell_normal_field_basis_matrix(
                    K_basis_dense,
                    self._quad_points,
                    self._quad_weights,
                    self._quad_normals,
                    delta_reg=self.regularization_delta,
                    adaptive_self_reg=self.adaptive_self_reg,
                )
                Hw = H.T * self._quad_weights  # (n_dof, n_quad)
                M = Hw @ H
                M = 0.5 * (M + M.T)
                self._H_full = H
                self._Hw = Hw
                self._L_work = M

        self._Q_c_work = Q_c_work
        self._mark_phase_end("rebuild_L_assembly", _t_L)

        _t_chol = self._mark_phase_start()
        if cache_hit_payload is not None:
            # Restore the Cholesky-factor triple from the cached payload.
            # We do not re-run ``null_space_projection_matrix`` or
            # ``shell_cholesky_pure``; the cached arrays are byte-identical
            # to what the full path would have produced for this structural
            # signature.
            self._Q = np.asarray(cache_hit_payload["Q"], dtype=np.float64)
            Lr = np.asarray(cache_hit_payload["Lr"], dtype=np.float64)
            self._L_red = Lr
            Lr_chol = np.asarray(cache_hit_payload["Lr_chol"], dtype=np.float64)
            self._Lr_chol_host = Lr_chol
            self._jax_Lr_chol = jnp.asarray(Lr_chol)
            Lr_eigf_chol = np.asarray(
                cache_hit_payload["Lr_eigf_chol"], dtype=np.float64
            )
            self._jax_Lr_eigf_chol = jnp.asarray(Lr_eigf_chol)
            self._Lr_eigf_chol_host = np.asarray(Lr_eigf_chol, dtype=np.float64)
            self._refresh_L_red_eig_host_cache(Lr)
        else:
            # Gauge projection on top of the continuity-restricted subspace:
            # drop the remaining exact null modes (constant per puck) of L.
            L_c = Q_c_work.T @ self._L_work @ Q_c_work
            if self._mode_truncation_tol > 0.0:
                # Inline the eigendecomposition so we can apply both the
                # null-space cut *and* the spectral mode-truncation cut in
                # one pass.  After the null-space cut, sort the surviving
                # eigenmodes by eigenvalue (smallest first, since the
                # modal beta amplitude is ``|U^T f| / lambda``) and keep
                # the smallest set of modes whose cumulative
                # ``sum 1 / lambda`` covers ``1 - mode_truncation_tol``
                # of the total surviving response.  This realises the
                # appendix (vii) speedup ("drops basis columns whose
                # contribution to Bsc at the plasma surface is below
                # tolerance") without anchoring to a single noise-floor
                # eigenvalue, so it stays well-behaved even for wide
                # spectra typical of n_base >= 48 fixtures.  The
                # discarded modes shrink the downstream Cholesky and
                # eigenfloor work, with up to cubic savings in
                # ``n_kept`` per the appendix's complexity argument.
                lam_c, V_c = np.linalg.eigh(L_c)
                max_lam_c = (
                    float(np.max(np.abs(lam_c))) if lam_c.size > 0 else 0.0
                )
                if max_lam_c < 1e-30:
                    keep = np.ones(lam_c.size, dtype=bool)
                else:
                    keep = lam_c > self._null_space_threshold * max_lam_c
                    if bool(np.any(keep)):
                        kept_idx = np.flatnonzero(keep)
                        lam_kept = lam_c[kept_idx]
                        order = np.argsort(lam_kept)
                        inv_sorted = 1.0 / lam_kept[order]
                        total = float(np.sum(inv_sorted))
                        if total > 0.0:
                            cum_frac = np.cumsum(inv_sorted) / total
                            target = 1.0 - float(self._mode_truncation_tol)
                            keep_count = int(
                                np.searchsorted(cum_frac, target, side="left")
                            ) + 1
                            keep_count = max(1, min(keep_count, lam_kept.size))
                            keep_local = np.zeros(lam_kept.size, dtype=bool)
                            keep_local[order[:keep_count]] = True
                            keep = np.zeros_like(keep)
                            keep[kept_idx[keep_local]] = True
                if not bool(np.any(keep)):
                    keep = np.zeros(lam_c.size, dtype=bool)
                    if lam_c.size > 0:
                        keep[int(np.argmin(np.abs(lam_c)))] = True
                Q_L = V_c[:, keep]
            else:
                Q_L = null_space_projection_matrix(
                    L_c,
                    threshold=self._null_space_threshold,
                )
            self._Q = Q_c_work @ Q_L
            Lr = self._Q.T @ self._L_work @ self._Q
            self._L_red = Lr
            Lr_arr = np.asarray(Lr, dtype=np.float64)
            Lr_sym = 0.5 * (Lr_arr + Lr_arr.T)
            w_host, V_host = np.linalg.eigh(Lr_sym)
            max_w = float(np.max(np.abs(w_host)))
            th_eig = float(self._eigenfloor_threshold) * max(max_w, 1e-30)
            self._L_red_eig_lam_np = np.maximum(w_host, th_eig)
            self._L_red_eig_U_np = V_host
            self._L_red_eig_anchor_np = Lr_sym.copy()
            self._jax_Lr_chol = shell_cholesky_pure(jnp.asarray(Lr_arr), jitter=1e-10)
            self._Lr_chol_host = np.asarray(self._jax_Lr_chol)
            self._jax_Lr_eigf_chol = shell_eigenfloor_cholesky_from_eig(
                jnp.asarray(w_host),
                jnp.asarray(V_host),
                threshold=float(self._eigenfloor_threshold),
                jitter=1e-10,
            )
            self._Lr_eigf_chol_host = np.asarray(self._jax_Lr_eigf_chol)
            if lcache_enabled and cache_key is not None:
                try:
                    _psc_lcache_store(
                        cache_key,
                        {
                            "L_base": np.asarray(self._L_work, dtype=np.float64),
                            "Q_c_work": np.asarray(Q_c_work, dtype=np.float64),
                            "Q": np.asarray(self._Q, dtype=np.float64),
                            "Lr": np.asarray(Lr, dtype=np.float64),
                            "Lr_chol": np.asarray(self._Lr_chol_host, dtype=np.float64),
                            "Lr_eigf_chol": np.asarray(
                                self._jax_Lr_eigf_chol, dtype=np.float64
                            ),
                        },
                    )
                except Exception:  # noqa: BLE001
                    pass
        self._mark_phase_end("rebuild_cholesky", _t_chol)

        # Stacked work arrays: on the reduced path these are picked from the
        # base replicas only (same local basis, so they are exactly what the
        # loading vector needs after Bn folding); on the full path they equal
        # ``_phi_stack`` / ``_w_stack``.
        if self._jax_phi_stack is not None:
            if reduced_active:
                self._jax_phi_work_stack = self._jax_phi_stack[
                    np.asarray(self._base_reps)
                ]
                self._jax_w_work_stack = self._jax_w_stack[np.asarray(self._base_reps)]
            else:
                self._jax_phi_work_stack = self._jax_phi_stack
                self._jax_w_work_stack = self._jax_w_stack
        else:
            self._jax_phi_work_stack = None
            self._jax_w_work_stack = None

        # Replica->base index map used by the JIT bodies to fold ``Bn``
        # and gather ``beta_work`` back to every replica.  On the full
        # path this must be the identity (``arange(n_all)``) so that
        # :func:`_fold_Bn_to_work` and :func:`_gather_beta_work_to_all`
        # reduce to no-ops; if we left the orbit-grouping here the
        # JAX segment_sum would (incorrectly) collapse ``Bn`` across
        # orbits even though the full matrix ``L`` is in use.
        if reduced_active:
            self._base_indices_arr = np.asarray(base_indices, dtype=np.int32)
        else:
            self._base_indices_arr = np.arange(n_all, dtype=np.int32)

        # Signs fed to the signed fold / gather and the analytic VJP.
        # They coincide with ``_replica_signs`` only on the reduced
        # path; on the full-``L`` fallback the fold/gather are no-ops,
        # so effective signs must be all ``+1`` to avoid silently
        # sign-flipping stellsym-image replicas' ``Bn`` / ``beta``.
        # ``_replica_signs`` itself retains its raw +1/-1 pattern so
        # structural tests and the signed inductance assembly can read
        # it unchanged.
        if reduced_active:
            self._signs_effective = np.asarray(self._replica_signs, dtype=np.int8)
        else:
            self._signs_effective = np.ones(n_all, dtype=np.int8)

        # JAX arrays for module-level JIT (no new ``jax.jit`` closures per rebuild)
        _t_vjp = self._mark_phase_start()
        self._setup_jax()
        self._mark_phase_end("rebuild_vjp_setup", _t_vjp)

        self._fixed_bulk_operator_enabled = bool(
            self.solver_mode == "energy"
            and not self._has_free_puck_dofs()
            and self._K_stack is not None
            and self._phi_work_stack is not None
            and self._w_work_stack is not None
        )
        # Defer ``_M_field`` invalidation to the cache-key check.  Bumping
        # a version counter here would unconditionally invalidate even on
        # TF-only rebuilds (which fire whenever ``release_host_L_work_after_rebuild``
        # forces the short-circuit in :meth:`_rebuild` to fail).  The cache
        # key includes :meth:`_hash_puck_local_dofs`, which changes only
        # when puck DOFs actually move, so a TF-only rebuild leaves the
        # key intact and the next ``_B_at_points_tf_only`` reuses the
        # existing matrix at zero cost.

        # Optional: drop the host copy of ``L_work`` after factorization
        # when the downstream JAX / NumPy stack does not need a dense
        # host mirror (full or reduced path).
        if self._release_host_L_work_after_rebuild and self.solver_mode == "energy":
            if self._reduced_active:
                self._L_work = None
            elif not self._has_free_puck_dofs():
                self._L_work = None

        # Solve
        self.beta = self._solve_beta(self._tf_arrays())
        self._geom_hash = hash(tuple(self.local_full_x))
        self._puck_dofs_hash_at_rebuild = self._hash_puck_local_dofs()

        if _psc_bulk_timing_enabled() and _t_rebuild_total0 is not None:
            assert self._timing_rows is not None
            G_meta = int(self.nfp) * (2 if self.stellsym else 1)
            nq0 = int(K_per_puck_global[0].shape[0]) if K_per_puck_global else 0
            nd0 = int(K_per_puck_global[0].shape[1]) if K_per_puck_global else 0
            self._timing_rows.append(
                {
                    "phase": "rebuild_meta",
                    "n_base": int(self._n_base_pucks),
                    "G": G_meta,
                    "nq_per": nq0,
                    "nd_per": nd0,
                }
            )
            self._timing_rows.append(
                {
                    "phase": "rebuild_total",
                    "seconds": float(time.perf_counter() - _t_rebuild_total0),
                }
            )

    # ------------------------------------------------------------------
    # JAX fast path (TF-only VJP, pre-computed L and Q)
    # ------------------------------------------------------------------

    def _tf_dofs_hash(self) -> int:
        """Hash TF curve and current DOF bytes for :meth:`_tf_arrays` caching."""
        parts = []
        for c in self.coils_TF:
            parts.append(c.curve.x.tobytes())
            parts.append(c.current.x.tobytes())
        return hash(tuple(parts))

    def _free_vjp_cache_key(self, pts: np.ndarray, solve_vjp_mode: str) -> tuple:
        """Hash the reduced free-DOF linearization cache state."""
        pts_arr = np.ascontiguousarray(np.asarray(pts).reshape(-1, 3), dtype=float)
        use_far, pair_k, tf_load, sol_mode, bs_far = _reduced_free_dof_extras_for_jax(
            self._resolved_bulk_far_pair_kappa()
        )
        if use_far and pair_k > 0.0:
            centers, quats, _, _ = self._get_base_puck_geometry()
            near_idx, far_idx, n_near, n_far = self._reduced_far_pair_indices(
                centers, quats, float(pair_k)
            )
            pair_partition_key = (
                tuple(np.asarray(near_idx).reshape(-1).tolist()),
                tuple(np.asarray(far_idx).reshape(-1).tolist()),
            )
        else:
            near_idx = jnp.zeros((0, 2), dtype=jnp.int32)
            far_idx = jnp.zeros((0, 2), dtype=jnp.int32)
            n_near = 0
            n_far = 0
            pair_partition_key = ((), ())
        return (
            self._hash_puck_local_dofs(),
            self._tf_dofs_hash(),
            pts_arr.shape,
            hash(pts_arr.tobytes()),
            bool(getattr(self, "_reduced_free_dof_active", False)),
            int(self.nfp),
            bool(self.stellsym),
            float(self.regularization_delta),
            float(self._eigenfloor_threshold),
            bool(self.adaptive_self_reg),
            bool(self._resolved_checkpoint_l_pairs()),
            int(self._resolved_jax_pair_row_chunk()),
            int(self._resolved_bs_eval_chunk(pts_arr.shape[0])),
            str(solve_vjp_mode),
            bool(use_far),
            float(pair_k),
            str(tf_load),
            str(sol_mode),
            float(bs_far),
            bool(_psc_w1_envelope_env()),
            int(self._resolved_pair_replica_chunk()),
            int(n_near),
            int(n_far),
            pair_partition_key,
        )

    def _tf_arrays(self):
        """Stacked TF geometry/current arrays; cached while DOFs unchanged."""
        key = self._tf_dofs_hash()
        cached = getattr(self, "_tf_arrays_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1], cached[2], cached[3]
        gammas = np.stack([c.curve.gamma() for c in self.coils_TF], axis=0)
        gammadashs = np.stack([c.curve.gammadash() for c in self.coils_TF], axis=0)
        currents = np.array([c.current.get_value() for c in self.coils_TF], dtype=float)
        self._tf_arrays_cache = (key, gammas, gammadashs, currents)
        return gammas, gammadashs, currents

    def _gather_beta_work_to_all_numpy(self, beta_work: np.ndarray) -> np.ndarray:
        """Expand work-space ``beta_work`` (n_work * nd_per,) into the full
        ``n_dof_total``-vector ``beta_all`` used everywhere downstream.

        In the full path (``_reduced_active == False``) ``beta_work`` is
        already ``n_dof_total``-sized and this returns a copy; in the
        reduced path ``beta_work`` is ``n_base * nd_per`` and we gather
        via the stored base-index map with the matching per-replica sign
        (``+1`` pure rotation, ``-1`` stellsym image; see
        :func:`_gather_beta_work_to_all`).  ``_signs_effective`` is
        all-``+1`` on the full path, reducing this to a plain copy
        in that case.
        """
        beta_work = np.asarray(beta_work)
        if not self._reduced_active:
            return beta_work
        nd_per = self._K_stack.shape[2]
        n_work = beta_work.size // nd_per
        beta_stack = beta_work.reshape(n_work, nd_per)
        signs = np.asarray(self._signs_effective, dtype=beta_stack.dtype)
        gathered = beta_stack[self._base_indices_arr] * signs[:, None]
        return gathered.reshape(-1)

    def _invalidate_fixed_field_matrix(self) -> None:
        """Drop the cached ``_M_field`` matrix and its key.

        Called from the full-path :meth:`_rebuild` tail (puck DOFs or
        geometry changed; the ``M_field`` rows must be rebuilt).  Use
        :meth:`_bump_eval_pts_version` from the points-changed hook
        instead -- it both invalidates and increments the eval-pts
        version, so subsequent cache-key builds reflect the new points.
        No-op when nothing is cached so it is safe to call eagerly.
        """
        self._M_field = None
        self._M_field_key = None

    def _bump_eval_pts_version(self) -> None:
        """Invalidate ``_M_field`` and bump :attr:`_eval_pts_version`.

        Hooked from :meth:`PassiveBulkField.set_points_cart` so that any
        reseating of the C++ eval-points buffer -- in-place, or via a
        fresh ndarray allocation -- forces a rebuild on the very next
        :meth:`B_at_points` call.  Cheap enough (``O(1)``) to call
        unconditionally.
        """
        self._eval_pts_version += 1
        self._invalidate_fixed_field_matrix()

    def _fixed_field_matrix_key(
        self,
        pts: np.ndarray,
        *,
        use_eigf: bool,
    ) -> tuple:
        """Return a fast cache key for ``_M_field``.

        The key combines:

        - the eval-points buffer address (``pts.ctypes.data``) -- stable
          across calls that reuse the same underlying C++ point buffer
          (the common case inside ``MagneticFieldSum.B()``), so we never
          rebuild on a buffer-stable repeat call;
        - ``_eval_pts_version`` -- explicitly bumped by
          :meth:`PassiveBulkField.set_points_cart`, so any rewrite of
          the eval-points buffer (in-place or otherwise) invalidates;
        - ``_hash_puck_local_dofs()`` -- a ``hash(bytes)`` of only the
          puck local DOFs (``9 * n_base_pucks`` floats, ~tens of bytes
          in practice).  This is stable across TF-only rebuilds (which
          may fire every iteration when ``release_host_L_work_after_rebuild``
          is set), so a fixed-puck optimization sees a stable key for the
          entire run.  Note: this *is* a tobytes hash, but the slice is
          tiny -- microseconds, not the milliseconds the previous
          ``pts.tobytes()`` hash incurred over the full eval-points
          buffer.
        - ``use_eigf`` and the storage dtype, for completeness.
        """
        pts_arr = np.asarray(pts)
        try:
            buf_addr = int(pts_arr.ctypes.data)
        except (AttributeError, TypeError):
            buf_addr = id(pts_arr)
        return (
            buf_addr,
            tuple(pts_arr.shape),
            int(self._eval_pts_version),
            bool(use_eigf),
            self._hash_puck_local_dofs(),
            str(self._fixed_field_matrix_dtype),
        )

    def _build_fixed_field_matrix(
        self,
        pts: np.ndarray,
        *,
        use_eigf: bool,
    ) -> np.ndarray:
        """Materialize the fixed-geometry map ``beta_all -> B`` at ``pts``.

        With every puck DOF fixed, only the BS-to-plasma half of
        :meth:`_B_at_points_tf_only` is geometry-invariant.  This routine
        precomputes that half as a dense matrix ``M_field`` of shape
        ``(3 * n_eval, n_dof_all)`` so the hot path collapses to:

        - JAX ``_beta_from_bn_jitted`` (~10 ms on CPU)
        - cheap NumPy gather (``beta_work -> beta_all``)
        - one BLAS gemv ``out = M_field @ beta_all``

        The result has row order compatible with ``reshape(-1, 3)``.
        ``use_eigf`` is accepted only to keep the call signature
        symmetric with the upstream solve toggle; it does not affect
        ``M_field`` itself (the eigenfloor lives in the Cholesky factor
        consumed by ``_beta_from_bn_jitted``, not here).
        """
        if self._K_stack is None:
            raise RuntimeError(
                "Fixed field matrix requires uniform per-puck K_stack geometry."
            )

        pts_eval = np.ascontiguousarray(np.asarray(pts, dtype=np.float64).reshape(-1, 3))
        K_stack = np.asarray(self._K_stack, dtype=np.float64)
        quad_points = np.asarray(self._quad_points, dtype=np.float64)
        quad_weights = np.asarray(self._quad_weights, dtype=np.float64)
        n_all, nq_per, nd_per, _ = K_stack.shape
        n_dof_all = int(n_all * nd_per)

        store_dtype = (
            np.float32 if self._fixed_field_matrix_dtype == "float32" else np.float64
        )
        n_eval = int(pts_eval.shape[0])
        out = np.empty((n_eval * 3, n_dof_all), dtype=store_dtype)
        eval_chunk = max(1, min(128, n_eval or 1))
        eps = float(_EPS_BS)
        for i0 in range(0, n_eval, eval_chunk):
            i1 = min(i0 + eval_chunk, n_eval)
            m_chunk = np.zeros(
                ((i1 - i0) * 3, n_dof_all),
                dtype=np.float64,
            )
            pts_chunk = pts_eval[i0:i1]
            for pidx in range(n_all):
                q0 = pidx * nq_per
                d0 = pidx * nd_per
                q1 = q0 + nq_per
                r = pts_chunk[:, None, :] - quad_points[q0:q1][None, :, :]
                rn = np.sqrt(np.sum(r * r, axis=-1) + eps * eps)
                scaled_r = r * (quad_weights[q0:q1][None, :] / (rn**3))[:, :, None]
                Kp = K_stack[pidx]
                bx = (
                    np.einsum("qa,iq->ia", Kp[:, :, 1], scaled_r[:, :, 2])
                    - np.einsum("qa,iq->ia", Kp[:, :, 2], scaled_r[:, :, 1])
                )
                by = (
                    np.einsum("qa,iq->ia", Kp[:, :, 2], scaled_r[:, :, 0])
                    - np.einsum("qa,iq->ia", Kp[:, :, 0], scaled_r[:, :, 2])
                )
                bz = (
                    np.einsum("qa,iq->ia", Kp[:, :, 0], scaled_r[:, :, 1])
                    - np.einsum("qa,iq->ia", Kp[:, :, 1], scaled_r[:, :, 0])
                )
                m_chunk[0::3, d0 : d0 + nd_per] = MU0_OVER_4PI * bx
                m_chunk[1::3, d0 : d0 + nd_per] = MU0_OVER_4PI * by
                m_chunk[2::3, d0 : d0 + nd_per] = MU0_OVER_4PI * bz
            if store_dtype is np.float64:
                out[3 * i0 : 3 * i1, :] = m_chunk
            else:
                out[3 * i0 : 3 * i1, :] = m_chunk.astype(store_dtype, copy=False)
        return out

    def _fixed_field_matrix_for(
        self,
        pts: np.ndarray,
        *,
        use_eigf: bool,
    ) -> Optional[np.ndarray]:
        """Return a cached fixed-puck ``M_field``, or ``None`` when inapplicable.

        Caches the materialized matrix on ``self._M_field`` keyed on a
        cheap version-based tuple (see
        :meth:`_fixed_field_matrix_key`).  Rebuilds are timed under the
        ``B_at_points_fixed_build`` phase so a regression in build cost
        is visible in :meth:`get_timing_history`.
        """
        if (
            not self._fixed_bulk_operator_enabled
            or self._has_free_puck_dofs()
            or self.solver_mode != "energy"
        ):
            return None
        key = self._fixed_field_matrix_key(pts, use_eigf=use_eigf)
        if self._M_field is not None and self._M_field_key == key:
            return self._M_field
        _t_build = self._mark_phase_start()
        op = self._build_fixed_field_matrix(pts, use_eigf=use_eigf)
        self._M_field = op
        self._M_field_key = key
        self._mark_phase_end("B_at_points_fixed_build", _t_build)
        return op

    def _setup_jax(self) -> None:
        """Store JAX views of NumPy geometry; JIT callables are module-level.

        The dense block-diagonal ``K_basis``/``phi_mat`` DeviceArrays are no
        longer created here.  Instead the JIT bodies consume the stacked
        ``_jax_K_stack``/``_jax_phi_work_stack``/``_jax_w_work_stack`` arrays
        (``n_pucks``-fold smaller) plus ``_jax_base_indices`` for the
        symmetry-reduced path.

        ``_jax_Lr_chol`` / ``_jax_Lr_eigf_chol`` are built in :meth:`_rebuild`
        next to ``_L_red`` (not duplicated here).
        """
        self._jax_Q = jnp.asarray(self._Q)
        self._jax_Q_c = jnp.asarray(self._Q_c)
        self._jax_quad_pts = jnp.asarray(self._quad_points)
        self._jax_quad_n = jnp.asarray(self._quad_normals)
        self._jax_w_q = jnp.asarray(self._quad_weights)
        # Note: full-size ``_jax_phi_stack`` / ``_jax_w_stack`` device
        # copies were intentionally removed -- the JIT bodies consume the
        # (possibly work-sized) ``_jax_phi_work_stack`` /
        # ``_jax_w_work_stack`` variants instead, and the full-size host
        # arrays are kept only for legacy dense ``_phi_mat`` reconstruction.
        if self._jax_phi_work_stack is not None:
            self._jax_phi_work_stack = jnp.asarray(self._jax_phi_work_stack)
        if self._jax_w_work_stack is not None:
            self._jax_w_work_stack = jnp.asarray(self._jax_w_work_stack)
        self._jax_base_indices = jnp.asarray(self._base_indices_arr)
        # Effective per-replica signs (``+1`` pure rotation, ``-1``
        # stellsym image) fed to :func:`_fold_Bn_to_work` /
        # :func:`_gather_beta_work_to_all`.  Matches ``_replica_signs``
        # on the reduced path and is all-``+1`` on the full-``L``
        # fallback; see ``_signs_effective`` in :meth:`_rebuild`.
        # Stored as float32 because JAX's ``segment_sum`` accumulates
        # in the payload's dtype; the sign is exact.
        self._jax_replica_signs = jnp.asarray(self._signs_effective, dtype=jnp.float32)
        self._jax_base_reps = jnp.asarray(
            np.asarray(self._base_reps, dtype=np.int32), dtype=jnp.int32
        )
        if self._reduced_active and getattr(self, "_Q_c_work", None) is not None:
            self._jax_Q_c_base = jnp.asarray(self._Q_c_work)
        else:
            self._jax_Q_c_base = None
        u_host = getattr(self, "_L_red_eig_U_np", None)
        if u_host is not None and getattr(self, "_L_red_eig_lam_np", None) is not None:
            self._jax_L_red_eig_U = jnp.asarray(self._L_red_eig_U_np)
            self._jax_L_red_eig_lam = jnp.asarray(self._L_red_eig_lam_np)
        else:
            self._jax_L_red_eig_U = jnp.zeros((0, 0), dtype=jnp.float64)
            self._jax_L_red_eig_lam = jnp.zeros((0,), dtype=jnp.float64)

    @property
    def _K_stack(self) -> Optional[np.ndarray]:
        """Host view of ``_jax_K_stack`` (no duplicate NumPy storage)."""
        if not getattr(self, "_uniform_puck_shape", False):
            return None
        if getattr(self, "_jax_K_stack", None) is None:
            return None
        return np.asarray(self._jax_K_stack)

    @property
    def _phi_stack(self) -> Optional[np.ndarray]:
        if not getattr(self, "_uniform_puck_shape", False):
            return None
        if getattr(self, "_jax_phi_stack", None) is None:
            return None
        return np.asarray(self._jax_phi_stack)

    @property
    def _w_stack(self) -> Optional[np.ndarray]:
        if not getattr(self, "_uniform_puck_shape", False):
            return None
        if getattr(self, "_jax_w_stack", None) is None:
            return None
        return np.asarray(self._jax_w_stack)

    @property
    def _phi_work_stack(self) -> Optional[np.ndarray]:
        if self._jax_phi_work_stack is None:
            return None
        return np.asarray(self._jax_phi_work_stack)

    @property
    def _w_work_stack(self) -> Optional[np.ndarray]:
        if self._jax_w_work_stack is None:
            return None
        return np.asarray(self._jax_w_work_stack)

    # Lazy dense views for legacy callers (VTK export, regression tests).
    @property
    def _K_basis(self) -> np.ndarray:
        """Dense block-diagonal basis table ``(nq_total, n_dof_total, 3)``.

        Built on demand from the compact stacked representation
        ``_K_stack``.  Retained only for backward compatibility with legacy
        callers (e.g. :mod:`simsopt.field.puck_vtk`); internal hot paths
        use ``_K_stack`` directly.
        """
        if self._K_stack is None:
            raise RuntimeError(
                "Dense _K_basis requested but pucks have heterogeneous shapes; "
                "this code path is not supported.  Use _K_stack explicitly."
            )
        n_pucks, nq_per, nd_per, _ = self._K_stack.shape
        n_dof_total = self._n_dof_total
        nq_total = n_pucks * nq_per
        K = np.zeros((nq_total, n_dof_total, 3), dtype=self._K_stack.dtype)
        for pidx in range(n_pucks):
            r0, r1 = self._quad_row_ranges[pidx]
            d0 = self._dof_offsets[pidx]
            d1 = d0 + nd_per
            K[r0:r1, d0:d1, :] = self._K_stack[pidx]
        return K

    @property
    def _phi_mat(self) -> np.ndarray:
        """Dense block-diagonal ``phi_values`` ``(nq_total, n_dof_total)``.

        Lazily reconstituted from ``_phi_stack`` on access.
        """
        if self._phi_stack is None:
            raise RuntimeError(
                "Dense _phi_mat requested but pucks have heterogeneous shapes; "
                "this code path is not supported.  Use _phi_stack explicitly."
            )
        n_pucks, nq_per, nd_per = self._phi_stack.shape
        n_dof_total = self._n_dof_total
        nq_total = n_pucks * nq_per
        M = np.zeros((nq_total, n_dof_total), dtype=self._phi_stack.dtype)
        for pidx in range(n_pucks):
            r0, r1 = self._quad_row_ranges[pidx]
            d0 = self._dof_offsets[pidx]
            d1 = d0 + nd_per
            M[r0:r1, d0:d1] = self._phi_stack[pidx]
        return M

    # ------------------------------------------------------------------
    # JAX full path (local basis stacks for geometry VJP)
    # ------------------------------------------------------------------

    def _ensure_jax_full(self, *, force: bool = False) -> None:
        """Cache per-puck local-frame stacks for :func:`_B_eval_full_jitted`.

        Also exposes the dense rim-continuity projector ``self._Q_c`` as
        ``self._jax_Q_c`` for the free-DOF JAX path so that the constrained
        solve ``Q_c^T L Q_c \\alpha = Q_c^T f`` is used inside the VJP.

        When all puck geometry DOFs are fixed, this is a no-op unless
        ``force=True`` (diagnostics / tests that must materialise local stacks).

        Args:
            force: If ``True``, build local stacks even when every puck DOF is
            fixed (invalidates the usual "frozen geometry" fast path).
        """
        if not force and not self._has_free_puck_dofs():
            return
        if self._local_stacks_valid:
            return
        for _attr in (
            "_jax_local_pts",
            "_jax_local_K",
            "_jax_local_n",
            "_jax_local_w",
            "_jax_local_phi",
            "_jax_Q_c",
        ):
            _prev = getattr(self, _attr, None)
            if _prev is not None:
                try:
                    _prev.delete()
                except (AttributeError, TypeError, ValueError):
                    pass
                setattr(self, _attr, None)
        self._jax_local_pts = jnp.stack(
            [jnp.asarray(b.quad_points_local) for b in self._basis_per_puck]
        )
        self._jax_local_K = jnp.stack(
            [jnp.asarray(b.k_basis_local) for b in self._basis_per_puck]
        )
        self._jax_local_n = jnp.stack(
            [jnp.asarray(b.quad_normals_local) for b in self._basis_per_puck]
        )
        self._jax_local_w = jnp.stack(
            [jnp.asarray(b.quad_weights) for b in self._basis_per_puck]
        )
        self._jax_local_phi = jnp.stack(
            [jnp.asarray(b.phi_values) for b in self._basis_per_puck]
        )
        self._jax_Q_c = jnp.asarray(self._Q_c)
        self._jax_m_local = magnetic_dipole_moments_stacked(
            self._jax_local_K, self._jax_local_pts, self._jax_local_w
        )
        self._jax_Q_sym_local = quadrupole_magnetic_symmetric_stacked(
            self._jax_local_K, self._jax_local_pts, self._jax_local_w
        )
        self._local_stacks_valid = True

    # ------------------------------------------------------------------
    # Forward computations
    # ------------------------------------------------------------------

    def _compute_bn_at_quads_numpy(self) -> np.ndarray:
        """Normal component of TF :class:`BiotSavart` field at shell quadrature (C++ kernel).

        Caches a single :class:`BiotSavart` instance on ``self._bs_bn`` and
        only calls ``set_points_cart`` when the shell quadrature has been
        regenerated (i.e. after :meth:`_rebuild`), so repeated calls during
        an optimization iteration reuse the point cache managed by the
        C++ kernel.
        """
        _t_total = self._mark_phase_start()
        bs = getattr(self, "_bs_bn", None)
        pts_id = id(self._quad_points)
        if bs is None or getattr(self, "_bs_bn_pts_id", None) != pts_id:
            _t_set = self._mark_phase_start()
            bs = BiotSavart(self.coils_TF)
            bs.set_points_cart(np.ascontiguousarray(self._quad_points))
            self._bs_bn = bs
            self._bs_bn_pts_id = pts_id
            self._mark_phase_end("solve_beta_tf_normal_set_points", _t_set)
        _t_b = self._mark_phase_start()
        B = bs.B()
        self._mark_phase_end("solve_beta_tf_normal_B", _t_b)
        _t_dot = self._mark_phase_start()
        Bn = np.sum(B * self._quad_normals, axis=1)
        self._mark_phase_end("solve_beta_tf_normal_dot", _t_dot)
        self._mark_phase_end("solve_beta_tf_normal_total", _t_total)
        return Bn

    def _solve_beta(self, tf_arrays) -> np.ndarray:
        """Solve :math:`L \\beta = f` for the modal coefficients ``beta``.

        Uses the **full** null-space-trimmed projector
        :math:`Q = Q_c Q_L` (rim-continuity composed with :math:`L`'s
        null-space trim; assembled in :meth:`_rebuild`) rather than the
        bare rim-continuity projector :math:`Q_c`.  Passing the full
        projector here matches what the TF-only forward JAX path
        (``_B_eval_jitted`` / ``_B_eval_from_bn_jitted``) uses
        internally to compute ``beta`` when evaluating
        :meth:`B_at_points`, so the stored ``self.beta`` is consistent
        with shell-current diagnostics (:meth:`get_shell_currents`,
        :meth:`get_equivalent_currents`) and the induced field.

        Rationale: when the semi-analytic self-block replacement
        ``exact_disc_faces=True`` is enabled, :math:`L` acquires
        genuine near-null modes (the constant-:math:`g` gauge modes)
        whose eigenvalues are at machine zero rather than smeared out
        by the regularized kernel.  The ``shell_solve_eigenfloor_pure``
        jitter floor (:math:`10^{-10}`) then amplifies any
        :math:`Q_c^{T} f` component in those directions by
        :math:`\\sim 10^{10}`, giving a cosmetic ``|beta|`` blow-up
        (those null modes produce :math:`K \\approx 0` analytically,
        so fields are unaffected, but diagnostics are misleading).
        Projecting with :math:`Q = Q_c Q_L` excludes the null space
        before the eigenfloor solve and removes the blow-up.

        Still uses the eigenvalue-floor kernel (not Cholesky) so that
        the solve remains well-defined even when the residual reduced
        matrix :math:`Q^{T} L Q` is not strictly positive definite at
        floating-point precision.

        When ``solver_mode == "shell_l2"``, takes a pure-NumPy path:
        assemble ``g = -H^T diag(w) Bn^{TF}`` from the cached ``_Hw``
        and solve ``Q^T M Q alpha = Q^T g`` with the same eigenfloor
        kernel.  The reduced symmetry path is disabled in this mode
        (see :meth:`_rebuild`) so no gather is needed.
        """
        _t_total = self._mark_phase_start()
        _t_bn = self._mark_phase_start()
        Bn = self._compute_bn_at_quads_numpy()
        self._mark_phase_end("solve_beta_bn_total", _t_bn)
        if self.solver_mode == "shell_l2":
            _t_load = self._mark_phase_start()
            g = -self._Hw @ Bn
            g_r = np.asarray(self._Q).T @ g
            self._mark_phase_end("solve_beta_l2_load", _t_load)
            _t_solve = self._mark_phase_start()
            alpha = np.asarray(
                shell_solve_prefactored_pure(
                    self._jax_Lr_eigf_chol,
                    jnp.asarray(g_r),
                )
            )
            self._mark_phase_end("solve_beta_l2_solve", _t_solve)
            _t_gather = self._mark_phase_start()
            beta_work = np.asarray(self._Q) @ alpha
            beta = self._gather_beta_work_to_all_numpy(beta_work)
            self._mark_phase_end("solve_beta_gather", _t_gather)
            self._mark_phase_end("solve_beta_total", _t_total)
            return beta
        _t_solve = self._mark_phase_start()
        beta_work = np.array(
            _beta_eigenfloor_from_bn_jitted(
                self._jax_Lr_eigf_chol,
                self._jax_Q,
                self._jax_phi_work_stack,
                self._jax_w_work_stack,
                self._jax_base_indices,
                self._jax_replica_signs,
                jnp.asarray(Bn),
            )
        )
        self._mark_phase_end("solve_beta_jax_loading_solve", _t_solve)
        _t_gather = self._mark_phase_start()
        beta = self._gather_beta_work_to_all_numpy(beta_work)
        self._mark_phase_end("solve_beta_gather", _t_gather)
        self._mark_phase_end("solve_beta_total", _t_total)
        return beta

    @property
    def biot_savart(self) -> "PassiveBulkField":
        """Passive bulk contribution as a :class:`PassiveBulkField`."""
        return self._field

    @property
    def eval_points(self) -> np.ndarray:
        """``(n_eval, 3)`` Cartesian evaluation grid for :meth:`B_at_points`.

        The attribute is exposed as a property (Phase-J J1) so that
        reassignments invalidate the device-side cache used by the
        dipole-mode forward and backward kernels.  Reads behave
        identically to the historical plain attribute -- callers can
        continue to do ``psc.eval_points[i]`` etc.

        Returns
        -------
        np.ndarray
            ``(n_eval, 3)`` contiguous ``float64`` array.  The
            returned array is the canonical backing store; callers
            should not mutate it in place (doing so will silently leave
            the device-side cache stale, since the setter is the only
            invalidation hook).
        """
        return self._eval_points

    @eval_points.setter
    def eval_points(self, value: np.ndarray) -> None:
        r"""Replace the evaluation grid and invalidate the dipole device cache.

        Bumps :attr:`_eval_points_version` so any helper that snapshots
        a previous version can detect the change in ``O(1)``, and
        clears :attr:`_dipole_jdev_pts` so the next call to
        :meth:`_get_dipole_jdev_pts` rebuilds the device-side copy.

        The input is normalised exactly like the original
        :meth:`__init__` assignment -- ``float64`` dtype and C-order
        contiguous layout -- so downstream JAX kernels see the same
        memory layout regardless of how the caller built ``value``.

        Parameters
        ----------
        value : np.ndarray
            ``(n_eval, 3)`` array of Cartesian evaluation points.

        Notes
        -----
        Defensive ``getattr`` calls are used because :meth:`__init__`
        assigns ``self.eval_points`` before the dipole cache slots are
        guaranteed to exist on every code path; production reassignments
        always hit both branches.
        """
        self._eval_points = np.asarray(value, dtype=float, order="C")
        # ``getattr`` is defensive against being called from a subclass
        # ``__init__`` that runs before our own slot initialisation.
        self._eval_points_version = (
            getattr(self, "_eval_points_version", 0) + 1
        )
        self._dipole_jdev_pts = None

    def _get_dipole_jdev_pts(self, pts: np.ndarray) -> "jnp.ndarray":
        r"""Return a device-resident ``jnp.ndarray`` for ``pts``, cached when reusable.

        Phase-J J1 host->device cache.  Each call to
        :meth:`_B_at_points_dipole` and :meth:`_vjp_dipole` historically
        re-wrapped ``pts`` with :func:`jnp.asarray`, paying a
        ``(n_eval, 3) float64`` host-to-device copy every iteration.
        For a single optimisation, ``pts is self.eval_points`` and
        only changes when the plasma quadrature changes.

        This helper returns the cached :attr:`_dipole_jdev_pts` when
        ``pts`` is the canonical backing array, and falls back to a
        fresh :func:`jnp.asarray` conversion otherwise (e.g., when an
        ad-hoc one-off eval grid is passed via ``B_at_points(other_pts)``).
        The cache is invalidated automatically by the
        :meth:`eval_points` setter.

        Parameters
        ----------
        pts : np.ndarray
            ``(n_eval, 3)`` host array of evaluation points.

        Returns
        -------
        jnp.ndarray
            ``(n_eval, 3)`` device array equivalent to
            ``jnp.asarray(pts)``.  When ``pts is self.eval_points``
            the same array reference is returned on every subsequent
            call until :meth:`eval_points` is reassigned.
        """
        if pts is self._eval_points:
            cached = self._dipole_jdev_pts
            if cached is not None and cached.shape == pts.shape:
                return cached
            jdev = jnp.asarray(pts)
            self._dipole_jdev_pts = jdev
            return jdev
        return jnp.asarray(pts)

    def recompute_currents(self) -> None:
        """Recompute modal coefficients after TF geometry/currents or puck DOFs change.

        When puck DoFs are unchanged since the previous rebuild this
        short-circuits to a TF-only re-solve (cheap); otherwise the full
        ``_rebuild`` is invoked.  The Stage 2 incremental hook
        (``_rebuild(changed_mask=...)``) is available for callers that
        know which puck DoFs moved and want to exercise the fast path
        explicitly; the default path here does not yet emit per-DoF
        masks.

        The "have any DOFs changed?" branch is decided by an
        ``O(N_unique_dof_opts)`` comparison of
        ``(id(opt._dofs), opt._dofs._state_version)`` tuples instead of
        the previous ``hash(tuple(self.local_full_x))`` (which copied the
        full DOF array on every call).  ``Dofs._state_version`` is bumped
        only when DOF *values* actually change (see
        :class:`~simsopt._core.optimizable.DOFs`), so this is exact for
        in-process state.  The persistent ``_geom_hash`` /
        ``_psc_lcache_digest`` paths are kept because ``_state_version``
        resets every process and cannot key a cross-process cache.
        """
        _t_total = self._mark_phase_start()
        current_versions = tuple(
            (id(opt._dofs), opt._dofs._state_version)
            for opt in self._unique_dof_opts
        )
        puck_dofs_unchanged = (
            self._last_base_geom_state is not None
            and current_versions == self._geom_versions
        )
        if puck_dofs_unchanged:
            # Phase-J J3: when every puck DOF object's ``_state_version``
            # is unchanged since the previous rebuild, no row of the
            # ``(n_base, 9)`` base-geometry state matrix can have moved.
            # Reuse the previous snapshot directly and skip the
            # ``ascontiguousarray + reshape + copy + np.any`` overhead.
            # Mostly matters in the ``allfixed`` configuration where
            # only TF currents move and the puck DOF graph never
            # changes; saves ~10-30us per ``recompute_currents`` call.
            base_geom_now = self._last_base_geom_state
            changed_pucks = np.zeros(self._n_base_pucks, dtype=bool)
        else:
            base_geom_now = self._base_geom_state_matrix()
            changed_pucks = None
            if self._last_base_geom_state is not None:
                changed_pucks = np.any(
                    base_geom_now != self._last_base_geom_state, axis=1
                )
        if self.solver_mode == "dipole":
            # Dipole mode: ``_rebuild_dipole`` itself dispatches between
            # the full reduced-L assembly (puck DoFs moved) and the
            # TF-only short-circuit (only TF currents / coil geometry
            # moved) using the supplied ``changed_mask``.  This mirrors
            # the energy-mode ``rebuild_short_circuit`` and is the
            # fast-path for free-DoF optimisations where most outer
            # iterations only touch the TF currents.
            self._rebuild_dipole(changed_mask=changed_pucks)
            self._geom_versions = current_versions
            self._last_base_geom_state = base_geom_now
            self._free_vjp_cache = None
            self._field.clear_cached_properties()
            self._mark_phase_end(
                "recompute_currents_total_dipole", _t_total
            )
            return
        if current_versions != self._geom_versions:
            # Always forward ``changed_mask`` when available so the Stage-2
            # all-false TF-only short-circuit in :meth:`_rebuild` can fire even if
            # ``SIMSOPT_PSC_PARTIAL_L_REUSE`` is unset (partial-L incremental
            # logic inside ``_rebuild`` remains gated on that env separately).
            self._rebuild(changed_mask=changed_pucks)
            # Do not replace ``self._field``: the existing :class:`PassiveBulkField`
            # still delegates to ``self.B_at_points``, which uses updated L, beta,
            # and JIT functions. Replacing the field breaks ``MagneticFieldSum``'s
            # reference to the original object and orphan cache invalidation.
            _path = "rebuild"
        else:
            Lr = np.asarray(getattr(self, "_L_red", []), dtype=np.float64)
            anc = getattr(self, "_L_red_eig_anchor_np", None)
            tau = _psc_eigk_trigger_env()
            if Lr.size and anc is not None and anc.shape == Lr.shape and tau > 0.0:
                n0 = float(np.linalg.norm(anc, ord="fro")) + 1e-30
                rel = float(np.linalg.norm(Lr - anc, ord="fro") / n0)
                if rel > tau:
                    self._refresh_L_red_eig_host_cache(Lr)
                    self._setup_jax()
            self.beta = self._solve_beta(self._tf_arrays())
            _path = "tf_only"
        self._geom_versions = current_versions
        self._last_base_geom_state = base_geom_now
        self._free_vjp_cache = None
        self._field.clear_cached_properties()
        self._mark_phase_end(f"recompute_currents_total_{_path}", _t_total)

    @contextmanager
    def force_tf_only_forward(self) -> Iterator[None]:
        """Context manager: route :meth:`B_at_points` through the cheap path.

        While this context is active, :meth:`B_at_points` short-circuits
        through :meth:`_B_at_points_tf_only` regardless of whether
        :meth:`_has_free_center_or_quat_dofs` returns ``True``.  This lets
        callers that perform their own host-side gradient (for example,
        ``stellcoilbench_dipoles``'s
        :class:`~stellcoilbench.coil_optimization
        ._bulk_center_parameterization.ConstrainedJFWrapper` running its
        reduced-DOF FD loop) avoid the ~22 s warm cost of the JAX free-DOF
        forward when only the *value* of ``B(pts)`` is needed.

        The caller is responsible for ensuring :meth:`recompute_currents`
        has been invoked since the last DOF write, so the host-side
        ``_jax_Lr_chol`` / ``_jax_Q`` stacks reflect the perturbed state.

        Example
        -------
        ::

            with psc_bulk.force_tf_only_forward():
                # B(pts) returned from the cheap TF-only path that uses
                # the most recently rebuilt host-side Cholesky.
                B = psc_bulk.B_at_points(pts)

        Notes
        -----
        Nesting is supported: the previous value of
        ``self._force_tf_only_forward`` is restored on exit, so an outer
        block that already enabled the flag continues to see the cheap
        path after an inner block exits.  Exceptions inside the block do
        not leak the flag; the ``finally`` clause always restores.
        """
        if not hasattr(self, "_tf_only_ctx_entries"):
            self._tf_only_ctx_entries = 0
        self._tf_only_ctx_entries += 1
        prev = bool(self._force_tf_only_forward)
        self._force_tf_only_forward = True
        try:
            yield
        finally:
            self._force_tf_only_forward = prev

    @contextmanager
    def force_quat_only_forward(self) -> Iterator[None]:
        """Context manager: route :meth:`B_at_points` through the cheap path.

        Semantically identical to :meth:`force_tf_only_forward` -- both
        short-circuit through :meth:`_B_at_points_tf_only`.  The
        separate name exists to document intent: this path is used when
        **quaternion** DoFs are the free variables and the caller's FD
        loop calls :meth:`recompute_currents` between probes so that
        host-side ``_jax_Lr_chol`` / ``_jax_Q`` stacks reflect the
        geometry change induced by the quaternion perturbation.

        The caller (``ConstrainedJFWrapper`` in ``mode='rotation'``)
        must invoke ``recompute_psc_bulk_currents`` after every
        ``_sync_to_JF`` -- this is wired automatically by the
        ``recompute_callback`` supplied at wrapper construction time.

        Phase 2 (May-2026 speedup campaign, Phase B2).
        """
        if not hasattr(self, "_quat_only_ctx_entries"):
            self._quat_only_ctx_entries = 0
        self._quat_only_ctx_entries += 1
        prev = bool(self._force_tf_only_forward)
        self._force_tf_only_forward = True
        try:
            yield
        finally:
            self._force_tf_only_forward = prev

    def B_at_points(
        self, points: np.ndarray, *, moving_puck_mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Passive bulk B at Cartesian points ``(N, 3)``.

        When ``solver_mode == "shell_l2"`` the free-DOF-VJP and TF-only-
        VJP JIT paths are bypassed (not implemented for the L^2 branch
        yet): we fall back to a pure-NumPy Biot-Savart of the cached
        :math:`\\beta`, which is recomputed by :meth:`recompute_currents`
        whenever TF currents, TF geometry or puck DOFs change.

        When :attr:`_force_tf_only_forward` is ``True`` (set via the
        :meth:`force_tf_only_forward` context manager), the gate that
        selects between the JAX free-DOF runner and the cheap TF-only
        path is forced into the cheap branch even if
        :meth:`_has_free_center_or_quat_dofs` reports ``True``.

        Parameters
        ----------
        points
            Observation points shaped ``(N, 3)``.
        moving_puck_mask
            Optional boolean mask of shape ``(n_base_pucks,)`` reserved for
            the symmetry-reduced moving-puck fast path (:envvar:
            ``SIMSOPT_PSC_MOVING_PUCK_SUBSET``). Full per-base decomposition
            of :func:`_B_eval_reduced_free_dof_dispatch` plus incremental
            Biot-Savart updates is gated until parity benchmarks land; until
            then the argument is validated (when supplied) but does not alter
            the numerical path.
        """
        _t_total = self._mark_phase_start()
        pts = np.asarray(points)
        if moving_puck_mask is not None:
            mk = np.asarray(moving_puck_mask, dtype=bool).reshape(-1)
            if mk.shape != (int(self._n_base_pucks),):
                raise ValueError(
                    "moving_puck_mask must have shape "
                    f"({self._n_base_pucks},), got {mk.shape}"
                )
        if self.solver_mode == "dipole":
            # Defer the ``_tf_arrays()`` host-side stack until we know
            # the dispatch needs it.  The cached B branch (allfixed +
            # also free-DoF *unless* recompute_currents has not yet
            # run since the last DoF change) ignores TF arrays
            # entirely.
            out = self._B_at_points_dipole(pts)
            self._mark_phase_end("B_at_points_dipole_total", _t_total)
            return out
        g_tf, gd_tf, I_tf = self._tf_arrays()
        if self.solver_mode == "shell_l2":
            if self._has_free_puck_dofs():
                raise NotImplementedError(
                    "solver_mode='shell_l2' does not support free puck DOFs "
                    "yet; fix all puck DOFs or use solver_mode='energy'."
                )
            acc = jnp.float32 if self._use_f32_bs else None
            out = np.array(
                shell_biot_savart_stacked_pure(
                    self._jax_K_stack,
                    self._jax_quad_pts,
                    self._jax_w_q,
                    jnp.asarray(self.beta),
                    jnp.asarray(pts),
                    accumulate_dtype=acc,
                )
            )
            self._mark_phase_end("B_at_points_shell_l2_total", _t_total)
            return out
        if self._has_free_center_or_quat_dofs() and not self._force_tf_only_forward:
            self._ensure_jax_full()
            br = np.asarray(self._base_reps, dtype=np.int32)
            reduced_ok = bool(
                getattr(self, "_reduced_free_dof_active", False)
                and self._jax_Q_c_base is not None
                and not _psc_disable_reduced_free()
            )
            if reduced_ok:
                c_base = jnp.asarray(
                    np.stack([self._all_pucks[i][0] for i in br], axis=0)
                )
                q_base = jnp.asarray(self._all_pucks_quats[br])
                g_float = jnp.asarray(float(self._symmetry_G), dtype=jnp.float64)
                _t_forward = self._mark_phase_start()
                solve_vjp_mode = _psc_free_solve_vjp_env()
                use_far, pair_k, tf_load, sol_mode, bs_far = (
                    _reduced_free_dof_extras_for_jax(
                        self._resolved_bulk_far_pair_kappa()
                    )
                )
                if use_far and pair_k > 0.0:
                    c_np, q_np, _, _ = self._get_base_puck_geometry()
                    near_idx, far_idx, _, _ = self._reduced_far_pair_indices(
                        c_np, q_np, float(pair_k)
                    )
                else:
                    near_idx = jnp.zeros((0, 2), dtype=jnp.int32)
                    far_idx = jnp.zeros((0, 2), dtype=jnp.int32)
                m_loc_b = self._jax_m_local[br]
                q_loc_b = self._jax_Q_sym_local[br]
                static_tail = (
                    self._jax_base_indices,
                    self._jax_base_reps,
                    g_float,
                    int(self.nfp),
                    bool(self.stellsym),
                    float(self.regularization_delta),
                    float(self._eigenfloor_threshold),
                    bool(self.adaptive_self_reg),
                    bool(self._resolved_checkpoint_l_pairs()),
                    int(self._resolved_jax_pair_row_chunk()),
                    int(self._resolved_bs_eval_chunk(pts.shape[0])),
                    "full",
                    solve_vjp_mode,
                    use_far,
                    float(pair_k),
                    m_loc_b,
                    q_loc_b,
                    str(tf_load),
                    str(sol_mode),
                    float(bs_far),
                    bool(_psc_w1_envelope_env()),
                    int(self._resolved_pair_replica_chunk()),
                    near_idx,
                    far_idx,
                )
                if _psc_cache_free_vjp_env():

                    def fwd_cached(c_b, q_b, g, gd, I):
                        return _B_eval_reduced_free_dof_jitted(
                            self._jax_local_pts[br],
                            self._jax_local_K[br],
                            self._jax_local_n[br],
                            self._jax_local_w[br],
                            self._jax_local_phi[br],
                            self._jax_Q_c_base,
                            c_b,
                            q_b,
                            g,
                            gd,
                            I,
                            pts,
                            self._jax_L_red_eig_U,
                            self._jax_L_red_eig_lam,
                            *static_tail,
                        )

                    y, pullback = vjp(
                        fwd_cached,
                        c_base,
                        q_base,
                        jnp.asarray(g_tf),
                        jnp.asarray(gd_tf),
                        jnp.asarray(I_tf),
                    )
                    out = np.asarray(y)
                    self._free_vjp_cache = {
                        "key": self._free_vjp_cache_key(pts, solve_vjp_mode),
                        "pullback": pullback,
                        "centers_shape": tuple(c_base.shape),
                        "quats_shape": tuple(q_base.shape),
                    }
                else:
                    out = np.array(
                        _B_eval_reduced_free_dof_jitted(
                            self._jax_local_pts[br],
                            self._jax_local_K[br],
                            self._jax_local_n[br],
                            self._jax_local_w[br],
                            self._jax_local_phi[br],
                            self._jax_Q_c_base,
                            c_base,
                            q_base,
                            g_tf,
                            gd_tf,
                            I_tf,
                            pts,
                            self._jax_L_red_eig_U,
                            self._jax_L_red_eig_lam,
                            *static_tail,
                        )
                    )
                    self._free_vjp_cache = None
                self._mark_phase_end("B_at_points_free_reduced_jax", _t_forward)
                self._mark_phase_end("B_at_points_free_reduced_total", _t_total)
                return out
            centers_all = jnp.asarray(np.stack([p[0] for p in self._all_pucks], axis=0))
            quats_all = jnp.asarray(self._all_pucks_quats)
            _t_forward = self._mark_phase_start()
            out = np.array(
                _B_eval_full_jitted(
                    self._jax_local_pts,
                    self._jax_local_K,
                    self._jax_local_n,
                    self._jax_local_w,
                    self._jax_local_phi,
                    self._jax_Q_c,
                    centers_all,
                    quats_all,
                    g_tf,
                    gd_tf,
                    I_tf,
                    pts,
                    float(self.regularization_delta),
                    float(self._eigenfloor_threshold),
                    bool(self.adaptive_self_reg),
                    int(self._full_L_band_size),
                    bool(self._resolved_checkpoint_l_pairs()),
                    int(self._resolved_bs_eval_chunk(pts.shape[0])),
                )
            )
            self._mark_phase_end("B_at_points_free_full_jax", _t_forward)
            self._mark_phase_end("B_at_points_free_full_total", _t_total)
            return out
        out = self._B_at_points_tf_only(pts, g_tf=g_tf, gd_tf=gd_tf, I_tf=I_tf)
        self._mark_phase_end("B_at_points_tf_total", _t_total)
        return out

    def _refresh_dipole_device_cache(
        self,
        centers: np.ndarray,
        quats: np.ndarray,
        *,
        moments_only: bool = False,
    ) -> None:
        """Store device-side snapshots of puck geometry for reuse.

        Phase-H Step 3.  Called at the tail of every successful
        :meth:`_rebuild_dipole` return (full rebuild and TF-only
        short-circuit).  The cached ``jax.Array`` objects are reused
        by :meth:`_B_at_points_dipole` (cached branch) and
        :meth:`_vjp_dipole` so the per-iter forward + backward only
        re-wrap ``pts`` and the cotangent.  The buffers are
        invalidated by setting any one of them to ``None`` when the
        underlying geometry / moments change again -- which we don't
        need to do explicitly because every change goes through
        ``_rebuild_dipole`` and lands back in this refresh.

        Args:
            centers: ``(n_base, 3)`` base puck centres (host).
            quats: ``(n_base, 4)`` base puck quaternions (host).
            moments_only: When ``True`` (TF-only short-circuit), only
                ``m_global`` is re-wrapped to device; the static
                centre / quaternion / self-L snapshots from the
                previous full rebuild are left untouched.  This saves
                one H->D transfer per iter in the dominant
                ``allfixed`` regime.
        """
        n_base = int(self._n_base_pucks)
        if not moments_only:
            self._dipole_jdev_centers = jnp.asarray(centers)
            self._dipole_jdev_quats = jnp.asarray(quats)
            if self._dipole_self_L_local_cached is not None:
                self._dipole_jdev_self_L_local = jnp.asarray(
                    self._dipole_self_L_local_cached
                )
            else:
                self._dipole_jdev_self_L_local = None
        if self._dipole_m_red is not None:
            self._dipole_jdev_m_global = jnp.asarray(
                self._dipole_m_red.reshape(n_base, 3)
            )
        else:
            self._dipole_jdev_m_global = None

    def _B_at_points_dipole(
        self,
        pts: np.ndarray,
    ) -> np.ndarray:
        """Evaluate ``B(pts)`` for ``solver_mode == 'dipole'``.

        Path:

        1. If any puck DoFs changed since the last rebuild, redo the
           dipole-mode forward end-to-end via
           :func:`_psc_bulk_dipole_mod.forward_dipole_pipeline` (small,
           JIT-friendly).  This route is taken in free-DoF
           optimisation and ensures gradients propagate through every
           differentiable input.
        2. Otherwise, use the cached ``_dipole_m_red`` and only
           evaluate the closed-form dipole field
           :func:`_psc_bulk_dipole_mod.B_at_points_dipole`, which is
           ``O(n_eval * n_all)`` and very cheap.

        The cached path **never** fetches TF coil arrays (the closed-
        form dipole sum does not depend on them); ``_tf_arrays()`` is
        only invoked when the free-DoF branch is taken.

        Args:
            pts: ``(n_eval, 3)`` evaluation points.

        Returns:
            ``(n_eval, 3)`` bulk magnetic field.
        """
        take_free_dof_branch = (
            self._has_free_center_or_quat_dofs()
            and not getattr(self, "_force_tf_only_forward", False)
        )
        if take_free_dof_branch:
            # Free-DoF branch: end-to-end JAX pipeline; needs TF.
            centers, quats, _radii, _thicknesses = (
                self._get_base_puck_geometry()
            )
            g_tf, gd_tf, I_tf = self._tf_arrays()
            self._ensure_dipole_jit_cache(pts, g_tf)
            self_L_cached = self._dipole_self_L_local_cached
            assert self_L_cached is not None, (
                "PSCBulkArray dipole mode: self-inductance cache missing; "
                "call recompute_currents first."
            )
            assert self._dipole_jit_forward is not None
            out = np.asarray(
                self._dipole_jit_forward(
                    jnp.asarray(centers),
                    jnp.asarray(quats),
                    jnp.asarray(g_tf),
                    jnp.asarray(gd_tf),
                    jnp.asarray(I_tf),
                    self._get_dipole_jdev_pts(pts),
                    jnp.asarray(self_L_cached),
                    int(self.nfp),
                    bool(self.stellsym),
                ),
                dtype=np.float64,
            )
            return out
        # Cached fast path: just the closed-form dipole sum.  No
        # TF arrays are touched.
        m_red = self._dipole_m_red
        assert m_red is not None, (
            "PSCBulkArray dipole mode: moments not solved; call "
            "recompute_currents first."
        )
        n_base = int(self._n_base_pucks)
        # Build the field jit kernel only (drops TF from the key).
        self._ensure_dipole_jit_cache(pts, None)
        assert self._dipole_jit_field is not None
        # Reuse device-side puck geometry snapshot when available.
        centers_j = self._dipole_jdev_centers
        quats_j = self._dipole_jdev_quats
        m_global_j = self._dipole_jdev_m_global
        if centers_j is None or quats_j is None or m_global_j is None:
            centers, quats, _radii, _thicknesses = (
                self._get_base_puck_geometry()
            )
            centers_j = jnp.asarray(centers)
            quats_j = jnp.asarray(quats)
            m_global_j = jnp.asarray(m_red.reshape(n_base, 3))
        out = np.asarray(
            self._dipole_jit_field(
                self._get_dipole_jdev_pts(pts),
                centers_j,
                quats_j,
                m_global_j,
                int(self.nfp),
                bool(self.stellsym),
            ),
            dtype=np.float64,
        )
        return out

    def _ensure_dipole_rebuild_jit_cache(self, g_tf: np.ndarray) -> None:
        """Bind the per-instance assemble-L / assemble-f handles.

        As of Phase L3 the actual ``jax.jit`` decorators live at module
        level (:data:`_DIPOLE_ASSEMBLE_L_JIT`,
        :data:`_DIPOLE_ASSEMBLE_F_JIT`), so every instance shares JAX's
        internal abstract-shape cache.  This method now only updates the
        per-instance pointers + cache key; the first call from a fresh
        :class:`PSCBulkArray` at a previously-seen shape signature pays
        zero XLA compile cost.

        Parameters
        ----------
        g_tf : np.ndarray
            TF coil gamma array, shape ``(n_tf, n_tf_q, 3)``.  Only its
            shape is consulted (for the staleness key).
        """
        n_base = int(self._n_base_pucks)
        n_tf = int(np.asarray(g_tf).shape[0])
        n_tf_q = int(np.asarray(g_tf).shape[1])
        key = (n_base, int(self.nfp), bool(self.stellsym), n_tf, n_tf_q)
        if (
            self._dipole_jit_rebuild_key == key
            and self._dipole_jit_assemble_L is _DIPOLE_ASSEMBLE_L_JIT
            and self._dipole_jit_assemble_f is _DIPOLE_ASSEMBLE_F_JIT
        ):
            return
        self._dipole_jit_assemble_L = _DIPOLE_ASSEMBLE_L_JIT
        self._dipole_jit_assemble_f = _DIPOLE_ASSEMBLE_F_JIT
        self._dipole_jit_rebuild_key = key

    def _ensure_dipole_jit_cache(
        self, pts: np.ndarray, g_tf: Optional[np.ndarray]
    ) -> None:
        """Bind the per-instance forward and field jit handles.

        As of Phase L3 the actual ``jax.jit`` decorators live at module
        level (:data:`_DIPOLE_FIELD_JIT`, :data:`_DIPOLE_FORWARD_JIT`),
        so the JAX internal abstract-shape cache survives across
        :class:`PSCBulkArray` instances.  This method only updates the
        per-instance pointers + staleness keys.

        Two separate caches are managed here so that the cached
        :func:`B_at_points_dipole` path (which only depends on the
        dipole sum) is keyed independently of the TF-aware
        :func:`forward_dipole_pipeline` path:

        * ``_dipole_jit_forward`` -- end-to-end pipeline used in
          free-DoF mode; keyed on
          ``(n_base, n_eval, nfp, stellsym, n_tf, n_tf_q)``.  When
          ``g_tf`` is ``None`` (cached path only) the forward handle
          is **not** bound here.
        * ``_dipole_jit_field`` -- :func:`B_at_points_dipole` only;
          keyed on ``(n_base, n_eval, nfp, stellsym)``.

        Parameters
        ----------
        pts : np.ndarray
            Evaluation points, shape ``(n_eval, 3)``.  Only the shape
            is consulted.
        g_tf : np.ndarray | None
            TF coil gamma array.  When ``None``, only the field handle
            is bound.
        """
        n_base = int(self._n_base_pucks)
        n_eval = int(np.asarray(pts).shape[0])
        field_key = (n_base, n_eval, int(self.nfp), bool(self.stellsym))
        if (
            self._dipole_jit_field_key != field_key
            or self._dipole_jit_field is not _DIPOLE_FIELD_JIT
        ):
            self._dipole_jit_field = _DIPOLE_FIELD_JIT
            self._dipole_jit_field_key = field_key
        if g_tf is None:
            return
        n_tf = int(np.asarray(g_tf).shape[0])
        n_tf_q = int(np.asarray(g_tf).shape[1])
        fwd_key = (
            n_base,
            n_eval,
            int(self.nfp),
            bool(self.stellsym),
            n_tf,
            n_tf_q,
        )
        if (
            self._dipole_jit_key == fwd_key
            and self._dipole_jit_forward is _DIPOLE_FORWARD_JIT
        ):
            return
        self._dipole_jit_forward = _DIPOLE_FORWARD_JIT
        self._dipole_jit_key = fwd_key

    def _ensure_dipole_vjp_jit_cache(
        self, pts: np.ndarray, g_tf: np.ndarray
    ) -> None:
        """Bind the per-instance analytic VJP handles to the module cache.

        As of Phase L3 the actual ``jax.jit`` decorators for the four
        analytic-adjoint sub-kernels live at module level
        (:data:`_DIPOLE_VJP_B_JIT`, :data:`_DIPOLE_VJP_F_JIT`,
        :data:`_DIPOLE_VJP_L_JIT`, :data:`_DIPOLE_VJP_FUSED_TF_JIT`).
        That makes the JAX internal abstract-shape cache survive across
        :class:`PSCBulkArray` instances; a freshly constructed instance
        whose shape signature matches a previously-compiled one pays
        zero XLA compile cost on first call.

        The four sub-kernels implement the Phase-H analytic adjoint --
        i.e. the dipole backward pass decomposed so the host-side
        chain rule reuses the cached Cholesky factor and never
        reassembles ``L_red`` (see :data:`_dipole_vjp_B_kernel`,
        :data:`_dipole_vjp_f_kernel`, :data:`_dipole_vjp_L_kernel`,
        :data:`_dipole_vjp_fused_tf_kernel` for kernel-level docstrings).

        Parameters
        ----------
        pts : np.ndarray
            Evaluation points, shape ``(n_eval, 3)``.
        g_tf : np.ndarray
            TF coil gamma array, shape ``(n_tf, n_tf_q, 3)``.
        """
        n_base = int(self._n_base_pucks)
        n_eval = int(np.asarray(pts).shape[0])
        n_tf = int(np.asarray(g_tf).shape[0])
        n_tf_q = int(np.asarray(g_tf).shape[1])
        key = (n_base, n_eval, int(self.nfp), bool(self.stellsym), n_tf, n_tf_q)
        if (
            self._dipole_jit_vjp_key == key
            and self._dipole_jit_vjp_B is _DIPOLE_VJP_B_JIT
            and self._dipole_jit_vjp_f is _DIPOLE_VJP_F_JIT
            and self._dipole_jit_vjp_L is _DIPOLE_VJP_L_JIT
            and self._dipole_jit_vjp_fused_tf is _DIPOLE_VJP_FUSED_TF_JIT
        ):
            return

        self._dipole_jit_vjp_B = _DIPOLE_VJP_B_JIT
        self._dipole_jit_vjp_f = _DIPOLE_VJP_F_JIT
        self._dipole_jit_vjp_L = _DIPOLE_VJP_L_JIT
        self._dipole_jit_vjp_fused_tf = _DIPOLE_VJP_FUSED_TF_JIT
        self._dipole_jit_vjp_key = key

    def _B_at_points_tf_only(
        self,
        pts: np.ndarray,
        *,
        g_tf: Optional[np.ndarray] = None,
        gd_tf: Optional[np.ndarray] = None,
        I_tf: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Evaluate B(pts) using a pre-built shell Cholesky factor and ``_jax_Q``.

        By default (:envvar:`SIMSOPT_PSC_TF_SOLVE_EIGFLOOR` ``1``, see
        :func:`_psc_tf_solve_eigfloor_env`) this uses ``_jax_Lr_eigf_chol``, the
        same eigenfloor-stabilised factor as :meth:`_solve_beta`. Otherwise it
        uses the plain minimal-jitter ``_jax_Lr_chol`` (legacy Phase-3 baseline).

        This is the cheap value-only path that bypasses the free-DOF JAX
        runner.  It uses the same dispatch as the TF-only branch of
        :meth:`B_at_points` but is callable even when ``_has_free_puck_dofs()``
        is True.  The caller must ensure that :meth:`recompute_currents` has
        been called so that factors, ``_jax_Q``, and related arrays are
        up-to-date.

        Used by :meth:`_Rt_fd_gradient` to avoid the ~120x overhead of the
        JAX free-DOF runner inside the finite-difference loop.
        """
        _t_total = self._mark_phase_start()
        assert self._jax_Lr_chol is not None, (
            "_B_at_points_tf_only called before Cholesky factors are initialised"
        )
        assert self._jax_Q is not None, (
            "_B_at_points_tf_only called before _jax_Q is initialised"
        )
        use_eigf = _psc_tf_solve_eigfloor_env()
        if use_eigf:
            assert self._jax_Lr_eigf_chol is not None, (
                "_B_at_points_tf_only: eigenfloor Cholesky missing"
            )
        Lr_chol_jax = self._jax_Lr_eigf_chol if use_eigf else self._jax_Lr_chol
        pts = np.asarray(pts)
        if g_tf is None or gd_tf is None or I_tf is None:
            g_tf, gd_tf, I_tf = self._tf_arrays()
        if _USE_JAX_TF_VJP:
            _t_forward = self._mark_phase_start()
            out = np.array(
                _B_eval_jitted(
                    Lr_chol_jax,
                    self._jax_Q,
                    self._jax_quad_pts,
                    self._jax_quad_n,
                    self._jax_phi_work_stack,
                    self._jax_w_work_stack,
                    self._jax_K_stack,
                    self._jax_w_q,
                    self._jax_base_indices,
                    self._jax_replica_signs,
                    g_tf,
                    gd_tf,
                    I_tf,
                    pts,
                )
            )
            self._mark_phase_end("B_at_points_tf_only_jax", _t_forward)
            self._mark_phase_end("B_at_points_tf_only_total", _t_total)
            return out
        _t_bn = self._mark_phase_start()
        Bn = self._compute_bn_at_quads_numpy()
        self._mark_phase_end("B_at_points_fixed_bn", _t_bn)
        M_field = self._fixed_field_matrix_for(pts, use_eigf=bool(use_eigf))
        if M_field is not None:
            _t_solve = self._mark_phase_start()
            beta_work = np.asarray(
                _beta_from_bn_jitted(
                    Lr_chol_jax,
                    self._jax_Q,
                    self._jax_phi_work_stack,
                    self._jax_w_work_stack,
                    self._jax_base_indices,
                    self._jax_replica_signs,
                    jnp.asarray(Bn),
                ),
                dtype=np.float64,
            )
            self._mark_phase_end("B_at_points_fixed_solve", _t_solve, beta_work)
            _t_gather = self._mark_phase_start()
            beta_all = self._gather_beta_work_to_all_numpy(beta_work)
            self._mark_phase_end("B_at_points_fixed_gather", _t_gather)
            _t_gemv = self._mark_phase_start()
            if M_field.dtype == np.float32:
                out = np.asarray(
                    M_field @ beta_all.astype(np.float32, copy=False),
                    dtype=np.float64,
                ).reshape(-1, 3)
            else:
                out = np.asarray(M_field @ beta_all).reshape(-1, 3)
            self._mark_phase_end("B_at_points_fixed_gemv", _t_gemv)
            self._mark_phase_end("B_at_points_tf_only_total", _t_total)
            return out
        _t_forward = self._mark_phase_start()
        out = np.array(
            _B_eval_from_bn_jitted(
                Lr_chol_jax,
                self._jax_Q,
                self._jax_quad_pts,
                self._jax_phi_work_stack,
                self._jax_w_work_stack,
                self._jax_K_stack,
                self._jax_w_q,
                self._jax_base_indices,
                self._jax_replica_signs,
                jnp.asarray(Bn),
                pts,
            )
        )
        self._mark_phase_end("B_at_points_tf_only_from_bn_jax", _t_forward)
        self._mark_phase_end("B_at_points_tf_only_total", _t_total)
        return out

    def get_shell_currents(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(K, |K|)``: ``(n_quad, 3)`` sheet current from current ``beta``.

        Uses the stacked ``_K_stack`` directly (per-puck einsum) so the
        dense ``(nq_total, n_dof_total, 3)`` monolithic basis is never
        allocated -- this keeps memory linear in ``n_pucks`` rather than
        quadratic.
        """
        if self._K_stack is not None:
            n_pucks, nq_per, nd_per, _ = self._K_stack.shape
            b_stack = np.asarray(self.beta).reshape(n_pucks, nd_per)
            K_vec = np.einsum("pqak,pa->pqk", self._K_stack, b_stack).reshape(
                n_pucks * nq_per, 3
            )
            return K_vec, np.linalg.norm(K_vec, axis=-1)
        # Heterogeneous-shape fallback (rare): materialize dense on demand.
        b = jnp.asarray(self.beta)
        K = jnp.sum(jnp.asarray(self._K_basis) * b[None, :, None], axis=1)
        return np.array(K), np.array(jnp.linalg.norm(K, axis=-1))

    def get_equivalent_currents(self) -> np.ndarray:
        """Equivalent total current per puck in Amperes.

        For each puck, integrates ``|K|`` across the side-wall height
        (thickness ``t``) and across the face diameter (``2R``), returning
        the larger of the two as the representative equivalent current::

            I_side ~ max(|K_side|) * t
            I_face ~ max(|K_face|) * 2 * R
            I_eq   = max(I_side, I_face)
        """
        K_vec, K_mag = self.get_shell_currents()
        n_pucks = len(self._all_pucks)
        I_eq = np.zeros(n_pucks)
        for p in range(n_pucks):
            r0, r1 = self._quad_row_ranges[p]
            _, _, R_val, t_val = self._all_pucks[p]
            Km = K_mag[r0:r1]
            I_side = float(np.max(Km)) * t_val
            I_face = float(np.max(Km)) * 2.0 * R_val
            I_eq[p] = max(I_side, I_face)
        return I_eq

    # ------------------------------------------------------------------
    # VJP
    # ------------------------------------------------------------------

    def _has_free_puck_dofs(self) -> bool:
        return any(self.local_dofs_free_status)

    def _free_puck_dof_kinds(self) -> set[str]:
        """Classify currently free local puck DOFs by geometry family."""
        kinds: set[str] = set()
        for name, is_free in zip(
            self.local_full_dof_names, self.local_dofs_free_status
        ):
            if not is_free:
                continue
            if name in self._normal_offset_dof_names:
                kinds.add("normal_offset")
            elif str(name).startswith("center_"):
                kinds.add("center")
            elif str(name).startswith("q"):
                kinds.add("quaternion")
            elif str(name).startswith(("R", "t")):
                kinds.add("shape")
            else:
                kinds.add("other")
        return kinds

    def _has_free_center_or_quat_dofs(self) -> bool:
        """True iff any ``center_*`` or ``q*_`` puck DOF is currently free.

        Used to gate the JAX free-DOF runner: when only R/t are free,
        the runner traces gradients w.r.t. fixed centers/quats that are
        discarded downstream.  Routing the forward through
        :meth:`_B_at_points_tf_only` and the VJP through
        :meth:`_vjp_tf_only` + :meth:`_Rt_fd_gradient` saves ~3000 s on
        a 5-iter freeradius optimisation (see
        ``stellcoilbench_dipoles/bench_results/free_dof_variants/README.md``).
        """
        kinds = self._free_puck_dof_kinds()
        return bool(kinds & {"center", "quaternion"})

    def _hash_puck_local_dofs(self) -> int:
        """Hash bytes of the ``9 * n_base`` puck-geometry DOFs plus any
        trailing normal-offset ``d{i}`` DOFs (see :attr:`_n_geom_dofs`)."""
        x = np.asarray(self.local_full_x, dtype=float)
        return hash(x[: self._n_geom_dofs].tobytes())

    def _puck_geometry_unchanged_since_rebuild(self) -> bool:
        """True iff base-puck center/quaternion/R/t values match last :meth:`_rebuild`."""
        h0 = getattr(self, "_puck_dofs_hash_at_rebuild", None)
        if h0 is None:
            return False
        return self._hash_puck_local_dofs() == h0

    def _resolved_checkpoint_l_pairs(self) -> bool:
        """Effective ``checkpoint_L_pairs`` for the free-DOF JAX path (host)."""
        e = _psc_jax_checkpoint_env()
        if e is not None:
            return e
        if self._psc_checkpoint_l_pairs_opt is not None:
            return bool(self._psc_checkpoint_l_pairs_opt)
        return self._has_free_puck_dofs()

    def _resolved_jax_pair_row_chunk(self) -> int:
        """Row-batch size for :func:`_B_eval_reduced_free_dof_body` (host, static in JIT)."""
        env = _psc_jax_pair_chunk_from_env()
        n_b = int(self._n_base_pucks)
        if n_b < 1:
            return 0
        n_all = int(len(self._all_pucks)) if getattr(self, "_all_pucks", None) else 1
        basis = getattr(self, "_basis_per_puck", None)
        b0 = basis[0] if basis else None
        if b0 is None:
            return 0
        nq = int(b0.k_basis_local.shape[0])
        nd = int(b0.k_basis_local.shape[1])
        per_row = n_all * nq * nq * nd * nd * 8
        auto = 0
        if per_row > 0:
            c = max(1, int(1_000_000_000 // per_row))
            auto = int(min(n_b, c))
        ctor = int(self._jax_pair_row_chunk_ctor)
        if env is not None:
            if env == 0:
                return 0
            return int(min(int(env), n_b))
        if ctor > 0:
            return int(min(ctor, n_b))
        if auto >= n_b:
            return 0
        return auto

    def _resolved_pair_replica_chunk(self) -> int:
        """Replica-axis chunk for the free-DoF pair-inductance kernel (host, static in JIT).

        Resolution order mirrors :meth:`_resolved_jax_pair_row_chunk`:
        explicit :envvar:`SIMSOPT_PSC_PAIR_REPLICA_CHUNK`
        overrides the constructor knob, which overrides the AUTO heuristic
        (see :envvar:`SIMSOPT_PSC_PAIR_REPLICA_CHUNK_AUTO`).

        * ``SIMSOPT_PSC_PAIR_REPLICA_CHUNK``: unset/absent/zero -> skip to
          constructor/auto; positive -> ``min(env, n_all)``.
        * Constructor ``pair_replica_chunk``: positive ->
          ``min(ctor, n_all)``.
        * AUTO heuristic: ``k = min(n_all, max(1, 1e9 // (nq**2 * nd**2 *
          8)))``; ``0`` if ``k >= n_all`` (same as legacy monolithic
          ``vmap``).
        """
        n_all = int(len(self._all_pucks)) if getattr(self, "_all_pucks", None) else 1
        if n_all < 1:
            return 0
        env = _psc_pair_replica_chunk_from_env()
        if env is not None:
            if env <= 0:
                return 0
            return int(min(int(env), n_all))
        ctor = int(self._pair_replica_chunk_ctor)
        if ctor > 0:
            return int(min(ctor, n_all))
        if not _psc_pair_replica_chunk_auto_env():
            return 0
        basis = getattr(self, "_basis_per_puck", None)
        b0 = basis[0] if basis else None
        if b0 is None:
            return 0
        nq = int(b0.k_basis_local.shape[0])
        nd = int(b0.k_basis_local.shape[1])
        per_replica = int(nq) * int(nq) * int(nd) * int(nd) * 8
        if per_replica <= 0:
            return 0
        chunk = max(1, int(1_000_000_000 // per_replica))
        auto = int(min(n_all, chunk))
        return 0 if auto >= n_all else auto

    def _resolved_bulk_far_pair_kappa(self) -> float:
        """Return effective far-pair kappa, with env var overriding constructor/YAML."""
        env = _psc_pair_far_kappa_env_override()
        if env is not None:
            return float(env)
        return float(getattr(self, "_bulk_far_pair_kappa", 0.0))

    def _reduced_far_pair_indices(
        self,
        centers_base: np.ndarray,
        quats_base: np.ndarray,
        pair_far_kappa: float,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, int, int]:
        """Return host-classified non-self near/far pairs for reduced free-DOF JAX.

        The classification happens outside the JAX trace so far-pair scans never
        contain the dense pair kernel.  Self-pairs are intentionally excluded and
        are always added through the dense kernel inside the traced function.
        """
        centers = np.asarray(centers_base, dtype=float)
        quats = np.asarray(quats_base, dtype=float)
        _, _, radii, thicknesses = self._get_base_puck_geometry()
        all_pucks, _, base_indices, _, _, _ = self._replicate_pucks(
            centers, quats, radii, thicknesses
        )
        centers_all = np.stack([p[0] for p in all_pucks], axis=0)
        r_eff = np.asarray(
            [max(abs(float(p[2])), 0.5 * abs(float(p[3]))) for p in all_pucks],
            dtype=float,
        )
        near, far, n_near, n_far = _classify_pairs(
            centers_all, r_eff, np.asarray(base_indices, dtype=np.int32), pair_far_kappa
        )
        return jnp.asarray(near), jnp.asarray(far), n_near, n_far

    def _resolved_bs_eval_chunk(self, n_eval: int) -> int:
        """Eval-point chunk size for free-DOF shell Biot-Savart JAX bodies."""
        env = _psc_bs_eval_chunk_from_env()
        n = max(1, int(n_eval))
        if env is not None:
            if env == 0:
                return 0
            return int(min(max(1, env), n))
        # Keep tiny objectives monolithic.  Surface-sized objectives benefit from
        # smaller reverse-mode graphs and lower compile pressure.
        if n <= 8:
            return 0
        return int(min(16, n))

    def vjp_setup_B(self, v_B, eval_pts=None):
        """VJP of ``sum(v_B * B)`` w.r.t. all DOFs (TF + puck geometry)."""
        _t_total = self._mark_phase_start()
        self._check_normal_offset_vjp_support()
        v_B = np.asarray(v_B).reshape(-1, 3)
        pts = eval_pts if eval_pts is not None else self.eval_points
        if self.solver_mode == "dipole":
            out = self._vjp_dipole(v_B, pts)
            self._mark_phase_end("vjp_setup_B_dipole_total", _t_total)
            return out
        if self._has_free_puck_dofs():
            out = self._vjp_puck_geometry(v_B, pts)
            self._mark_phase_end("vjp_setup_B_free_total", _t_total)
            return out
        out = self._vjp_tf_only(v_B, pts)
        self._mark_phase_end("vjp_setup_B_tf_total", _t_total)
        return out

    def _merge_tf_coil_vjps(
        self,
        vg_np: np.ndarray,
        vgd_np: np.ndarray,
        vI_np: np.ndarray,
    ) -> Derivative:
        r"""Sum :meth:`Coil.vjp` outputs across all TF coils into one ``Derivative``.

        Phase-J J2 micro-optimisation that replaces the historical
        idiom

        .. code-block:: python

            vjp_tf = sum(
                self.coils_TF[i].vjp(vg_np[i], vgd_np[i], np.asarray([vI_np[i]]))
                for i in range(len(self.coils_TF))
            )

        used at every dipole-mode backward call.  ``sum`` walks
        :meth:`Derivative.__add__`, which calls :func:`copy_numpy_dict`
        on the running accumulator every iteration -- ``n_tf - 1``
        full-dict copies per backward pass.  For ``n_tf in {2..4}``
        the Python-side cost is ~10-20 us per copy, and it scales
        linearly with the number of TF coils.

        This helper accumulates into the first coil's freshly-allocated
        ``Derivative`` via :meth:`Derivative.__iadd__` so the running
        dict is mutated in place; the per-call cost becomes
        ``len(coils_TF)`` ``Coil.vjp`` calls + ``len(coils_TF) - 1``
        in-place dict merges (no LHS-side copies).  Behaviour is
        otherwise byte-identical to the historical ``sum`` -- each
        leaf array is the same elementwise.

        Parameters
        ----------
        vg_np, vgd_np : np.ndarray
            ``(n_tf, n_quad, 3)`` arrays of TF-side ``v_gamma`` and
            ``v_gammadash`` cotangents (from the fused dipole VJP).
        vI_np : np.ndarray
            ``(n_tf,)`` array of TF-side current cotangents.

        Returns
        -------
        Derivative
            Sum of ``self.coils_TF[i].vjp(vg_np[i], vgd_np[i], [vI_np[i]])``
            across all TF coils.  When ``self.coils_TF`` is empty the
            returned :class:`Derivative` has an empty data dict.
        """
        n_tf = len(self.coils_TF)
        if n_tf == 0:
            return Derivative({})
        out = self.coils_TF[0].vjp(
            vg_np[0], vgd_np[0], np.asarray([vI_np[0]])
        )
        for i in range(1, n_tf):
            out += self.coils_TF[i].vjp(
                vg_np[i], vgd_np[i], np.asarray([vI_np[i]])
            )
        return out

    def _vjp_dipole(self, v_B: np.ndarray, pts: np.ndarray) -> Derivative:
        """Dipole-mode analytic VJP.

        Phase-H factored adjoint.  With the forward
        ``m = -L^{-1} f`` and ``B = B(c, q, m, pts)``, the backward
        for ``J = <v_B, B>`` factors as

        1. ``(lam_c_B, lam_q_B, lam_m) = (dB/dc, dB/dq, dB/dm)^T v_B``
           (no L involved -- one ``jax.vjp`` through
           :func:`B_at_points_dipole`).
        2. ``lam_f = -L^{-T} lam_m`` -- one
           :func:`scipy.linalg.cho_solve` against the cached
           ``_dipole_L_red_chol`` (NO Cholesky refactor, NO L
           reassembly).
        3. ``(lam_c_f, lam_q_f, vg, vgd, vI) = (df/d{c,q,g,gd,I})^T lam_f``
           -- one ``jax.vjp`` through
           :func:`assemble_f_dipole_reduced_jax` (no L involved).
        4. For free centre / quaternion DoFs only, add
           ``(lam_c_L, lam_q_L) = (dL/d{c,q})^T outer(lam_f, m)`` --
           one ``jax.vjp`` through
           :func:`assemble_L_dipole_reduced_jax` (does **not** invert
           or refactor L; the cotangent is built from the already-
           solved ``m`` and ``lam_f``).

        The TF-side ``(vg, vgd, vI)`` are then mapped back through
        each ``coils_TF[i].vjp(...)`` (unchanged from the old kernel).
        """
        n_base = int(self._n_base_pucks)
        centers, quats, _radii, _thicknesses = self._get_base_puck_geometry()
        self_L_cached = self._dipole_self_L_local_cached
        assert self_L_cached is not None, (
            "PSCBulkArray dipole mode: self-inductance cache missing; "
            "call recompute_currents first."
        )
        chol = self._dipole_L_red_chol
        assert chol is not None, (
            "PSCBulkArray dipole mode: Cholesky factor missing; "
            "call recompute_currents first."
        )
        m_red = self._dipole_m_red
        assert m_red is not None, (
            "PSCBulkArray dipole mode: moments missing; call "
            "recompute_currents first."
        )
        g_tf, gd_tf, I_tf = self._tf_arrays()
        self._ensure_dipole_vjp_jit_cache(np.asarray(pts), g_tf)
        assert self._dipole_jit_vjp_B is not None
        assert self._dipole_jit_vjp_f is not None
        assert self._dipole_jit_vjp_fused_tf is not None

        # Free normal-offset (``d{i}``) DOFs also need the analytic per-puck
        # center gradient below (``dJ/dd_i = vc_np[i] . normal_i``), so they
        # take the split-kernel path too, not just raw free center/quat DOFs.
        free_geom = (
            self._has_free_center_or_quat_dofs()
            or self._has_free_normal_offset_dofs()
        )

        # Reuse device-resident puck geometry snapshots when valid;
        # otherwise wrap fresh NumPy arrays.
        centers_j = (
            self._dipole_jdev_centers
            if self._dipole_jdev_centers is not None
            else jnp.asarray(centers)
        )
        quats_j = (
            self._dipole_jdev_quats
            if self._dipole_jdev_quats is not None
            else jnp.asarray(quats)
        )
        m_global = m_red.reshape(n_base, 3)
        m_global_j = (
            self._dipole_jdev_m_global
            if self._dipole_jdev_m_global is not None
            else jnp.asarray(m_global)
        )

        if not free_geom:
            # ``allfixed`` fast path: a single fused JAX kernel does
            # vjp_B + on-device cho_solve + vjp_f.  Cuts the per-iter
            # backward to one device dispatch + one D->H copy.
            L_chol_j = self._dipole_jdev_L_red_chol
            assert L_chol_j is not None, (
                "dipole device-side Cholesky factor missing; call "
                "recompute_currents first."
            )
            _t_jax = self._mark_phase_start()
            vc_j, vq_j, vg, vgd, vI = self._dipole_jit_vjp_fused_tf(
                self._get_dipole_jdev_pts(pts),
                centers_j,
                quats_j,
                m_global_j,
                jnp.asarray(g_tf),
                jnp.asarray(gd_tf),
                jnp.asarray(I_tf),
                L_chol_j,
                jnp.asarray(v_B),
                int(self.nfp),
                bool(self.stellsym),
            )
            self._mark_phase_end("vjp_dipole_fused_tf", _t_jax)
            # ``vc`` / ``vq`` are not used downstream when puck DoFs
            # are fixed, but materialising them is essentially free
            # next to a single JAX dispatch.
            vg_np = np.asarray(vg)
            vgd_np = np.asarray(vgd)
            vI_np = np.asarray(vI)
            _t_tf_pullback = self._mark_phase_start()
            vjp_tf = self._merge_tf_coil_vjps(vg_np, vgd_np, vI_np)
            self._mark_phase_end("vjp_dipole_tf_pullback", _t_tf_pullback)
            return vjp_tf

        # Free centre / quaternion DoFs: keep the split kernels +
        # host-side cho_solve and add the L-side adjoint.
        _t_jax = self._mark_phase_start()
        lam_c_B, lam_q_B, lam_m_b = self._dipole_jit_vjp_B(
            self._get_dipole_jdev_pts(pts),
            centers_j,
            quats_j,
            m_global_j,
            jnp.asarray(v_B),
            int(self.nfp),
            bool(self.stellsym),
        )
        self._mark_phase_end("vjp_dipole_vjp_B", _t_jax)
        lam_m_np = np.asarray(lam_m_b).reshape(-1)

        _t_solve = self._mark_phase_start()
        c_factor, low = chol
        lam_f = -sp_linalg.cho_solve(
            (c_factor, low), lam_m_np, check_finite=False
        )
        self._mark_phase_end("vjp_dipole_chol_solve", _t_solve)
        lam_f_j = jnp.asarray(lam_f)

        _t_jax_f = self._mark_phase_start()
        lam_c_f, lam_q_f, vg, vgd, vI = self._dipole_jit_vjp_f(
            centers_j,
            quats_j,
            jnp.asarray(g_tf),
            jnp.asarray(gd_tf),
            jnp.asarray(I_tf),
            lam_f_j,
            int(self.nfp),
            bool(self.stellsym),
        )
        self._mark_phase_end("vjp_dipole_vjp_f", _t_jax_f)

        # L-side adjoint: ``(dL/d{c,q})^T outer(lam_f, m)``.
        assert self._dipole_jit_vjp_L is not None
        self_L_local_j = (
            self._dipole_jdev_self_L_local
            if self._dipole_jdev_self_L_local is not None
            else jnp.asarray(self_L_cached)
        )
        _t_jax_L = self._mark_phase_start()
        outer_ct = jnp.outer(lam_f_j, jnp.asarray(m_red))
        lam_c_L, lam_q_L = self._dipole_jit_vjp_L(
            centers_j,
            quats_j,
            self_L_local_j,
            outer_ct,
            int(self.nfp),
            bool(self.stellsym),
        )
        self._mark_phase_end("vjp_dipole_vjp_L", _t_jax_L)
        vc_np = (
            np.asarray(lam_c_B) + np.asarray(lam_c_f) + np.asarray(lam_c_L)
        )
        vq_np = (
            np.asarray(lam_q_B) + np.asarray(lam_q_f) + np.asarray(lam_q_L)
        )

        vg_np = np.asarray(vg)
        vgd_np = np.asarray(vgd)
        vI_np = np.asarray(vI)

        # Sized to ``_n_geom_dofs`` (not just ``n_base*_DOFS_PER_PUCK``) so
        # this doesn't crash via a ``Derivative`` length mismatch on an
        # array with (currently-fixed) normal-offset DOFs configured.
        grad_local = np.zeros(self._n_geom_dofs)
        for i in range(n_base):
            off = i * _DOFS_PER_PUCK
            grad_local[off : off + 3] = vc_np[i]
            grad_local[off + 3 : off + 7] = vq_np[i]

        # Normal-offset DOF chain rule: center_i = anchor_i + d_i * normal_i
        # is linear in d_i, so dJ/dd_i = vc_np[i] . normal_i -- no separate
        # FD term needed since vc_np is already the exact analytic
        # dJ/dcenter_i computed above.
        for i in self._normal_offset_puck_indices:
            grad_local[self._normal_offset_dof_index[i]] += float(
                np.dot(vc_np[i], self._normal_offset_normal[i])
            )

        _t_tf_pullback = self._mark_phase_start()
        vjp_tf = self._merge_tf_coil_vjps(vg_np, vgd_np, vI_np)
        self._mark_phase_end("vjp_dipole_tf_pullback", _t_tf_pullback)
        return Derivative({self: grad_local}) + vjp_tf

    def _vjp_tf_only(self, v_B, pts):
        if _USE_JAX_TF_VJP:
            _t_jax = self._mark_phase_start()
            gammas = np.array([c.curve.gamma() for c in self.coils_TF])
            gammadashs = np.array([c.curve.gammadash() for c in self.coils_TF])
            currents = np.array([c.current.get_value() for c in self.coils_TF])
            vg, vgd, vI = _vjp_tf_jitted(
                self._jax_Lr_chol,
                self._jax_Q,
                self._jax_quad_pts,
                self._jax_quad_n,
                self._jax_phi_work_stack,
                self._jax_w_work_stack,
                self._jax_K_stack,
                self._jax_w_q,
                self._jax_base_indices,
                self._jax_replica_signs,
                jnp.asarray(gammas),
                jnp.asarray(gammadashs),
                jnp.asarray(currents),
                jnp.asarray(pts),
                jnp.asarray(v_B),
            )
            vg, vgd, vI = np.asarray(vg), np.asarray(vgd), np.asarray(vI)
            self._mark_phase_end("vjp_tf_reference_jax", _t_jax)
            _t_pullback = self._mark_phase_start()
            out = sum(
                self.coils_TF[i].vjp(vg[i], vgd[i], np.asarray([vI[i]]))
                for i in range(len(self.coils_TF))
            )
            self._mark_phase_end("vjp_tf_reference_pullback", _t_pullback)
            return out
        return self._vjp_tf_only_analytic(v_B, pts)

    def _vjp_tf_only_analytic(self, v_B, pts) -> Derivative:
        """TF VJP: adjoint through reduced shell solve + :class:`BiotSavart` (C++).

        Uses the stacked ``_K_stack``/``_phi_stack`` representation so the
        dense ``(nq_total, n_dof_total, ...)`` arrays never materialise.

        When the symmetry-reduced path is active, the adjoint of the
        ``beta_base -> beta_all`` gather is a signed ``segment_sum`` over
        orbits (``lam_beta_work[b, :] = sum_{r in orbit(b)} sigma_r
        lam_beta_all[r, :]``), and the adjoint of the ``Bn_all -> Bn_base``
        signed fold is a sign-weighted broadcast (``lam_Bn_all[r, q] =
        sigma_r lam_Bn_work[base(r), q]``).  Both are applied with plain
        NumPy here (no JAX JIT needed) so the analytic C++
        :class:`BiotSavart` VJP can be used for the outer coil-DOF
        gradient.  On the full path ``signs`` is identically ``+1`` and
        the weighting drops out.

        Sign convention (airtight): the forward loading vector is
        ``f = -Phi^T W Bn`` (see
        :func:`~simsopt.field.bulk_inductance.shell_loading_vector_stacked_pure`),
        so the adjoint of ``f`` w.r.t. ``Bn_work`` carries an explicit
        ``-1`` factor: ``lam_Bn_work = -W * (Phi @ lam_f_work)``.  Missing
        this factor silently negates the entire bulk contribution to the
        TF-DOF gradient and is covered by
        ``test_vjp_tf_analytic_matches_jax_multi_puck``.

        ``BiotSavart`` caveat: ``BiotSavart.dB_by_dcoilcurrents`` returns
        the field-cache entries without triggering ``compute()`` when
        they are missing, so a freshly constructed ``bs`` (no ``B()`` /
        ``compute()`` yet) yields all-zero current-DOF gradients from
        ``bs.B_vjp``.  We prime the cache with ``bs.B()`` before calling
        ``B_vjp`` so both curve- and current-DOF gradients are returned
        (covered by the current-DOF branch of the same test).
        """
        v_B = np.asarray(v_B).reshape(-1, 3)
        pts = np.asarray(pts)
        K_stack = self._jax_K_stack
        quad_pts = self._jax_quad_pts
        w_quad = self._jax_w_q
        beta_all = jnp.asarray(self.beta)

        def fwd_bs(b_all: jnp.ndarray) -> jnp.ndarray:
            return shell_biot_savart_stacked_pure(
                K_stack,
                quad_pts,
                w_quad,
                b_all,
                jnp.asarray(pts),
                eps=_EPS_BS,
            )

        _t_shell = self._mark_phase_start()
        _, vjp_bs = vjp(fwd_bs, beta_all)
        lam_beta_all = np.asarray(vjp_bs(jnp.asarray(v_B))[0])
        self._mark_phase_end("vjp_tf_shell_biot_savart_adjoint", _t_shell)

        # Adjoint of the signed gather
        # ``beta_all[r, :] = sigma_r * beta_work[base(r), :]``: signed
        # segment-sum of ``lam_beta_all`` into ``lam_beta_work``.
        _t_solve = self._mark_phase_start()
        nd_per = self._K_stack.shape[2]
        n_all = self._K_stack.shape[0]
        lam_beta_all_stack = lam_beta_all.reshape(n_all, nd_per)
        signs_arr = np.asarray(self._signs_effective, dtype=lam_beta_all_stack.dtype)
        n_work = self._n_base_pucks if self._reduced_active else n_all
        lam_beta_work = np.zeros((n_work, nd_per))
        np.add.at(
            lam_beta_work,
            self._base_indices_arr,
            signs_arr[:, None] * lam_beta_all_stack,
        )
        lam_beta_work = lam_beta_work.reshape(-1)

        Q = self._Q
        qt_lam = Q.T @ lam_beta_work
        # Prefer eigenfloor stabilisation when env default is on -- matches
        # :meth:`_solve_beta` / cheap forward :meth:`_B_at_points_tf_only`.
        Lr_chol = (
            self._Lr_eigf_chol_host
            if _psc_tf_solve_eigfloor_env()
            else self._Lr_chol_host
        )
        _y = solve_triangular(Lr_chol, qt_lam, lower=True, check_finite=True)
        lam_r = solve_triangular(Lr_chol.T, _y, lower=False, check_finite=True)
        lam_f_work = Q @ lam_r  # (n_work * nd_per,)
        self._mark_phase_end("vjp_tf_solve_adjoint", _t_solve)

        # Adjoint of the loading vector coupled with the signed fold.
        # Forward: f_work[p, a] = -sum_q phi[p, q, a] * w[p, q] * Bn_work[p, q]
        # (the leading minus comes from ``shell_loading_vector_stacked_pure``),
        # followed by Bn_work[b, q] = sum_{r in orbit(b)} sigma_r Bn_all[r, q].
        # The adjoint therefore reads
        #   lam_Bn_all[r, q] = -sigma_r * w_quad[r, q] *
        #                      sum_a phi_work[base(r), q, a] lam_f_work[base(r), a]
        # and the explicit ``-`` below is *essential* (without it the analytic
        # VJP returns ``-grad`` instead of ``grad`` and Taylor tests on large
        # bulk contributions fail with rel_err ~= 2).
        phi_work_stack = self._phi_work_stack
        lam_f_work_stack = lam_f_work.reshape(n_work, nd_per)
        lam_Bn_work_stack = np.einsum(
            "pqa,pa->pq",
            phi_work_stack,
            lam_f_work_stack,
        )
        lam_Bn_all_stack = (
            signs_arr[:, None] * lam_Bn_work_stack[self._base_indices_arr]
        )
        w2d = getattr(self, "_quad_weights_2d", None)
        if w2d is None:
            w2d = self._quad_weights.reshape(n_all, -1)
        lam_Bn = -(w2d * lam_Bn_all_stack).reshape(-1)
        v_quad = lam_Bn[:, np.newaxis] * self._quad_normals

        _t_tf_pullback = self._mark_phase_start()
        pts_id = id(self._quad_points)
        bs = getattr(self, "_bs_bn", None)
        if bs is None or getattr(self, "_bs_bn_pts_id", None) != pts_id:
            bs = BiotSavart(self.coils_TF)
            bs.set_points_cart(np.ascontiguousarray(self._quad_points))
            self._bs_bn = bs
            self._bs_bn_pts_id = pts_id
        # Prime the field cache so ``dB_by_dcoilcurrents`` returns the
        # populated per-coil fields rather than fresh zeros; otherwise
        # ``B_vjp`` silently drops the current-DOF contribution.
        bs.B()
        out = bs.B_vjp(v_quad)
        self._mark_phase_end("vjp_tf_biotsavart_pullback", _t_tf_pullback)
        return out

    def _has_free_Rt_dofs(self) -> bool:
        """Return ``True`` iff any ``R{i}`` / ``t{i}`` DOF is currently free.

        Used by :meth:`_vjp_puck_geometry` to gate the finite-difference
        R / t gradient extension (see
        ``simsopt_fork/docs/notes/psc_bulk_Rt_vjp_audit.md``).
        """
        for name, is_free in zip(
            self.local_full_dof_names, self.local_dofs_free_status
        ):
            if not is_free:
                continue
            if str(name).startswith(("R", "t")):
                return True
        return False

    def _has_free_normal_offset_dofs(self) -> bool:
        """Return ``True`` iff any normal-offset ``d{i}`` DOF is currently free."""
        for name, is_free in zip(
            self.local_full_dof_names, self.local_dofs_free_status
        ):
            if is_free and name in self._normal_offset_dof_names:
                return True
        return False

    def _check_normal_offset_vjp_support(self) -> None:
        """Raise if free normal-offset DOFs are combined with something the
        FD gradient path (:meth:`_normal_offset_fd_gradient`) doesn't cover.

        One remaining gap: mixing a free ``d{i}`` with a free raw
        ``center_x/y/z``/``q*`` DOF on some *other* puck routes through the
        full JAX free-DOF runner (:meth:`_vjp_puck_geometry`'s
        ``center``/``quaternion`` branch), which has no ``d{i}`` handling at
        all. This only applies to ``solver_mode in ("energy", "shell_l2")``:
        :meth:`_vjp_dipole` computes an analytic per-puck center gradient
        densely over every base puck once *any* center/quat/normal-offset DOF
        is free, so mixing parameterizations across pucks is fine there (see
        the ``dJ/dd_i = vc_np[i] . normal_i`` chain rule in
        :meth:`_vjp_dipole`). Called once from :meth:`vjp_setup_B` so it
        covers every solver mode.
        """
        if not self._has_free_normal_offset_dofs():
            return
        if self.solver_mode == "dipole":
            return
        if self._has_free_center_or_quat_dofs():
            raise NotImplementedError(
                "PSCBulkArray: free normal-offset d{i} DOF(s) cannot "
                "currently be combined with a free center_x/y/z or "
                "quaternion DOF on another puck in the same array -- the "
                "full JAX free-center/quat VJP path (solver_mode='energy'/"
                "'shell_l2') has no handling for d{i} gradients. Fix all "
                "non-normal-offset center/quaternion DOFs, or avoid mixing "
                "parameterizations."
            )

    def _Rt_analytic_directional_combo(
        self, v_B: np.ndarray, pts: np.ndarray
    ) -> np.ndarray:
        """Directional derivative of ``<v_B, B(pts)>`` w.r.t. free ``R{i}`` / ``t{i}``.

        Phase L8 (May-2026 audit): closed-form ``BulkBasis`` Jacobians paired
        with ``jax.jvp`` are not wired yet.  When :func:`_psc_rt_analytic_combo_env`
        enables the analytic hook, callers still observe the FD reference
        from :meth:`_Rt_fd_gradient` until the Jacobian path lands.
        """
        return self._Rt_fd_gradient(
            np.asarray(v_B, dtype=np.float64), np.asarray(pts, dtype=np.float64)
        )

    def _Rt_fd_gradient(
        self,
        v_B: np.ndarray,
        pts: np.ndarray,
    ) -> np.ndarray:
        """One-sided FD gradient of ``<v_B, B(pts)>`` w.r.t. ``R{i}``/``t{i}``.

        Returns a ``(n_base, 2)`` float64 array where column 0 is the
        derivative w.r.t. ``R{i}`` and column 1 is w.r.t. ``t{i}``.
        Entries are zero for *fixed* R/t DOFs.

        Algorithm (one-sided forward FD with TF-only B-eval):

        1. Evaluate ``B0 = _B_at_points_tf_only(pts)`` once at the
           current (unperturbed) state.
        2. For each free shape DOF:
           a. Record ``v0``, set the DOF to ``v0 + eps``.
           b. Call :meth:`recompute_currents` (partial-rebuild fast
              path on the perturbed puck).
           c. Evaluate ``B_p = _B_at_points_tf_only(pts)``.
           d. Restore ``v0``, call :meth:`recompute_currents`.
           e. Emit ``(v_B . (B_p - B0)) / eps``.

        Cost per VJP: ``n_free_shape * (T_rebuild + T_B_tf_only) +
        n_free_shape * T_rebuild_restore + T_B0``.  At reactor scale
        (``n_base = 14``, 28 free shape DOFs) this is ~8 s vs the
        previous central-FD + free-DOF-jax path at ~231 s.

        The step size ``eps`` is read from ``SIMSOPT_PSC_RT_FD_EPS``
        (default ``1e-5``).  One-sided FD has O(eps) bias which is
        acceptable for L-BFGS-B line searches.

        See ``docs/passive_bulk_freeradius_diagnosis.md`` for the full
        performance diagnosis and Phase D.2 replacement plan.

        Non-finite ``B0``, ``B_p``, or emitted FD entries propagate as an
        all-NaN ``(n_base, 2)`` return value (Passive-bulk Phase 3, May 2026)
        so callers can distinguish poisoned probes from legitimate zeros.
        Audit: ``docs/notes/psc_bulk_Rt_vjp_audit.md``.

        Cache-consistency note (May 2026 audit follow-up): the partial-L-reuse
        incremental rebuild path (``SIMSOPT_PSC_PARTIAL_L_REUSE=1``, the
        default) caches per-puck ``L_base`` updates keyed by
        ``changed_pucks_mask`` and reuses portions of the previous Cholesky
        factor.  When the FD loop perturbs ``R{i}``/``t{i}`` and immediately
        restores the value, the resulting cache state depends on the prior
        invocation history (i.e., which path through ``_vjp_puck_geometry``
        most recently populated ``_jax_Lr_chol``/``_jax_Q``).  This produces
        path-dependent ``B0``/``B_p`` snapshots and constant-offset gradient
        errors observed in repeated TF-fixed evaluations.  To make this
        routine deterministic w.r.t. the surrounding call sequence, the FD
        loop temporarily disables partial-L-reuse via
        :envvar:`SIMSOPT_PSC_PARTIAL_L_REUSE` and restores the previous
        environment-variable value on exit.  See the regression test
        ``test_Rt_fd_gradient_deterministic_across_paths`` in
        ``tests/field/test_passive_bulks.py``.
        """
        import warnings as _warnings

        eps = _psc_rt_fd_eps_env()
        v_B_arr = np.ascontiguousarray(v_B, dtype=np.float64)
        pts_arr = np.ascontiguousarray(pts, dtype=np.float64)
        n_base = int(self._n_base_pucks)
        out = np.zeros((n_base, 2), dtype=np.float64)
        free_flags = np.asarray(self._dofs._free, dtype=bool)
        names = list(self.local_full_dof_names)
        name_to_idx = {n: i for i, n in enumerate(names)}

        _saved_partial_reuse = os.environ.get("SIMSOPT_PSC_PARTIAL_L_REUSE")
        os.environ["SIMSOPT_PSC_PARTIAL_L_REUSE"] = "0"
        try:
            _t_b0 = self._mark_phase_start()
            B0 = np.asarray(self._B_at_points_tf_only(pts_arr), dtype=np.float64)
            self._mark_phase_end("vjp_free_Rt_fd_b0_capture", _t_b0)
            if not np.all(np.isfinite(B0)):
                return np.full((n_base, 2), np.nan, dtype=np.float64)

            for b in range(n_base):
                for axis_idx, prefix in enumerate(("R", "t")):
                    dof_name = f"{prefix}{b}"
                    idx = name_to_idx.get(dof_name)
                    if idx is None or not bool(free_flags[idx]):
                        continue
                    v0 = float(self.get(dof_name))
                    try:
                        self.set(dof_name, v0 + eps)
                        self.recompute_currents()
                        B_p = np.asarray(
                            self._B_at_points_tf_only(pts_arr), dtype=np.float64
                        )
                    except (np.linalg.LinAlgError, RuntimeError, ValueError) as exc:
                        _warnings.warn(
                            f"PSCBulkArray._Rt_fd_gradient: skipping {dof_name!r} "
                            f"because the perturbed solve raised {exc.__class__.__name__}: {exc}",
                            category=RuntimeWarning,
                            stacklevel=3,
                        )
                        out[b, axis_idx] = 0.0
                        self.set(dof_name, v0)
                        try:
                            self.recompute_currents()
                        except Exception:
                            pass
                        continue
                    self.set(dof_name, v0)
                    self.recompute_currents()
                    if not np.all(np.isfinite(B_p)):
                        return np.full((n_base, 2), np.nan, dtype=np.float64)
                    fd_entry = float(np.sum(v_B_arr * (B_p - B0))) / eps
                    if not np.isfinite(fd_entry):
                        return np.full((n_base, 2), np.nan, dtype=np.float64)
                    out[b, axis_idx] = fd_entry
            return out
        finally:
            if _saved_partial_reuse is None:
                os.environ.pop("SIMSOPT_PSC_PARTIAL_L_REUSE", None)
            else:
                os.environ["SIMSOPT_PSC_PARTIAL_L_REUSE"] = _saved_partial_reuse

    def _normal_offset_fd_gradient(
        self,
        v_B: np.ndarray,
        pts: np.ndarray,
    ) -> np.ndarray:
        """One-sided FD gradient of ``<v_B, B(pts)>`` w.r.t. free ``d{i}`` DOFs.

        Mirrors :meth:`_Rt_fd_gradient` exactly (perturb the named DOF
        directly, ``recompute_currents``, evaluate, restore) rather than
        :meth:`_geometry_fd_gradient` (perturb ``center_x/y/z`` by a
        caller-supplied direction): for a normal-offset puck,
        ``center_x/y/z{i}`` are shadowed and unconditionally overwritten
        from ``d{i}`` by :meth:`_get_base_puck_geometry` on every
        rebuild, so perturbing them directly (as
        :meth:`_geometry_fd_gradient` does) has zero effect on the
        actual geometry -- only perturbing ``d{i}`` itself moves the puck.

        Returns a ``(n_base,)`` float64 array, zero for pucks that are
        not normal-offset or whose ``d{i}`` is currently fixed.
        """
        import warnings as _warnings

        eps = _psc_rt_fd_eps_env()
        v_B_arr = np.ascontiguousarray(v_B, dtype=np.float64)
        pts_arr = np.ascontiguousarray(pts, dtype=np.float64)
        n_base = int(self._n_base_pucks)
        out = np.zeros(n_base, dtype=np.float64)
        free_flags = np.asarray(self._dofs._free, dtype=bool)

        _saved_partial_reuse = os.environ.get("SIMSOPT_PSC_PARTIAL_L_REUSE")
        os.environ["SIMSOPT_PSC_PARTIAL_L_REUSE"] = "0"
        try:
            _t_b0 = self._mark_phase_start()
            B0 = np.asarray(self._B_at_points_tf_only(pts_arr), dtype=np.float64)
            self._mark_phase_end("vjp_free_normal_offset_fd_b0_capture", _t_b0)
            if not np.all(np.isfinite(B0)):
                return np.full(n_base, np.nan, dtype=np.float64)

            for i in self._normal_offset_puck_indices:
                dof_name = f"d{i}"
                idx = self._normal_offset_dof_index[i]
                if not bool(free_flags[idx]):
                    continue
                v0 = float(self.get(dof_name))
                try:
                    self.set(dof_name, v0 + eps)
                    self.recompute_currents()
                    B_p = np.asarray(
                        self._B_at_points_tf_only(pts_arr), dtype=np.float64
                    )
                except (np.linalg.LinAlgError, RuntimeError, ValueError) as exc:
                    _warnings.warn(
                        "PSCBulkArray._normal_offset_fd_gradient: skipping "
                        f"{dof_name!r} because the perturbed solve raised "
                        f"{exc.__class__.__name__}: {exc}",
                        category=RuntimeWarning,
                        stacklevel=3,
                    )
                    out[i] = 0.0
                    self.set(dof_name, v0)
                    try:
                        self.recompute_currents()
                    except Exception:
                        pass
                    continue
                self.set(dof_name, v0)
                self.recompute_currents()
                if not np.all(np.isfinite(B_p)):
                    return np.full(n_base, np.nan, dtype=np.float64)
                fd_entry = float(np.sum(v_B_arr * (B_p - B0))) / eps
                if not np.isfinite(fd_entry):
                    return np.full(n_base, np.nan, dtype=np.float64)
                out[i] = fd_entry
            return out
        finally:
            if _saved_partial_reuse is None:
                os.environ.pop("SIMSOPT_PSC_PARTIAL_L_REUSE", None)
            else:
                os.environ["SIMSOPT_PSC_PARTIAL_L_REUSE"] = _saved_partial_reuse

    def _geometry_fd_gradient(
        self,
        v_B: np.ndarray,
        pts: np.ndarray,
        perturbations: Sequence[Tuple[int, np.ndarray, np.ndarray]],
        eps: Optional[float] = None,
    ) -> np.ndarray:
        """One-sided FD gradient of ``<v_B, B(pts)>`` w.r.t. caller-supplied
        per-puck ``(centre, quaternion)`` perturbations.

        Mirrors :meth:`_Rt_fd_gradient` for callers that have a closed-form
        map from a small reduced DoF vector (e.g. ``delta_r`` for the
        toroidal-shell radial parameterisation, or ``(delta_theta,
        delta_phi)`` for the shell parameterisation) to per-puck
        ``(centre, quaternion)`` perturbations and want a cheap host-side
        gradient that bypasses the JAX free-DoF VJP runner.

        Parameters
        ----------
        v_B
            Cotangent on B at eval points; shape ``(n_eval, 3)``.
        pts
            Eval points; shape ``(n_eval, 3)``.
        perturbations
            Sequence of ``(puck_idx, dC, dQ)`` triples where:

            * ``puck_idx`` is a base puck index in ``[0, n_base_pucks)``.
            * ``dC`` is a length-3 Cartesian centre displacement direction
              in *per-unit-reduced-DoF* units (i.e.
              ``d(center_x{i}, center_y{i}, center_z{i}) /
              d(reduced_dof_k)`` evaluated at the current state).
            * ``dQ`` is a length-4 quaternion displacement direction with
              the same unit semantics; pass ``np.zeros(4)`` for reduced
              DoFs that do not affect orientation (e.g. radial mode).
        eps
            Step size; when ``None``, read from
            ``SIMSOPT_PSC_GEOM_FD_EPS`` (default ``1e-5``).  One-sided
            forward FD has O(eps) bias, acceptable for L-BFGS-B line
            searches.

        Returns
        -------
        np.ndarray
            Float64 array of length ``len(perturbations)`` with each
            entry equal to ``<v_B, B(c0 + eps*dC, q0 + eps*dQ) - B0> /
            eps``, where ``B`` is evaluated through the cheap TF-only
            host path.

        Notes
        -----
        Algorithm (one-sided forward FD with TF-only B-eval, mirroring
        :meth:`_Rt_fd_gradient`):

        1. Capture ``B0 = _B_at_points_tf_only(pts)`` once at the
           current (unperturbed) state (phase tag
           ``vjp_geometry_fd_b0_capture``).
        2. For each perturbation entry ``(puck_idx, dC, dQ)``:

           a. Record the current centre ``c0`` and quaternion ``q0`` of
              ``puck_idx``.
           b. Set centre to ``c0 + eps*dC``, quaternion to
              ``q0 + eps*dQ``.
           c. Call :meth:`recompute_currents` (partial-rebuild fast
              path on the perturbed puck, since only one puck moved).
           d. Evaluate ``B_p = _B_at_points_tf_only(pts)``.
           e. Restore ``c0``, ``q0``, call :meth:`recompute_currents`
              again so the array is back in its original state.
           f. Emit ``np.sum(v_B * (B_p - B0)) / eps``.

        Per-probe cost: ``T_recompute_partial + T_B_tf_only`` — at
        reactor scale (``n_base = 18``) this is ~0.15 s.  For an
        18-DoF radial-mode wrapper (1 perturbation per puck) the total
        VJP wall is roughly ``T_B0 + 18 * 0.15 ≈ 2.9 s``, vs ~25 s
        for the JAX free-DoF VJP on the same configuration.

        On a perturbed solve raising
        :class:`numpy.linalg.LinAlgError` /
        :class:`RuntimeError` / :class:`ValueError` (e.g. an L-BFGS-B
        line search probed an ill-conditioned puck configuration), the
        corresponding gradient entry is set to zero and a
        :class:`RuntimeWarning` is emitted; the underlying DOFs are
        restored before continuing.  Mirrors the silently-NaN-tainted
        behaviour of the pre-fix JAX free-DoF path so the optimiser can
        backtrack rather than crash.

        Phase tag: ``vjp_geometry_fd``.
        """
        import warnings as _warnings

        if eps is None:
            eps = _psc_geom_fd_eps_env()
        eps = float(eps)
        v_B_arr = np.ascontiguousarray(v_B, dtype=np.float64)
        pts_arr = np.ascontiguousarray(pts, dtype=np.float64)
        n_pert = len(perturbations)
        out = np.zeros(n_pert, dtype=np.float64)
        if n_pert == 0:
            return out

        _t_total = self._mark_phase_start()
        _t_b0 = self._mark_phase_start()
        B0 = np.asarray(self._B_at_points_tf_only(pts_arr), dtype=np.float64)
        self._mark_phase_end("vjp_geometry_fd_b0_capture", _t_b0)

        n_base = int(self._n_base_pucks)
        for k, entry in enumerate(perturbations):
            if len(entry) != 3:
                raise ValueError(
                    f"_geometry_fd_gradient: perturbations[{k}] must be "
                    f"(puck_idx, dC(3,), dQ(4,)), got {entry!r}"
                )
            puck_idx, dC, dQ = entry
            i = int(puck_idx)
            if not (0 <= i < n_base):
                raise IndexError(
                    f"_geometry_fd_gradient: perturbations[{k}] puck_idx={i} "
                    f"out of range [0, {n_base})."
                )
            dC_arr = np.asarray(dC, dtype=np.float64).reshape(-1)
            dQ_arr = np.asarray(dQ, dtype=np.float64).reshape(-1)
            if dC_arr.size != 3:
                raise ValueError(
                    f"_geometry_fd_gradient: perturbations[{k}] dC must be "
                    f"length 3, got size {dC_arr.size}."
                )
            if dQ_arr.size != 4:
                raise ValueError(
                    f"_geometry_fd_gradient: perturbations[{k}] dQ must be "
                    f"length 4, got size {dQ_arr.size}."
                )

            center_names = (f"center_x{i}", f"center_y{i}", f"center_z{i}")
            quat_names = (f"q0_{i}", f"qi_{i}", f"qj_{i}", f"qk_{i}")
            c0 = np.array(
                [float(self.get(n)) for n in center_names], dtype=np.float64
            )
            q0 = np.array(
                [float(self.get(n)) for n in quat_names], dtype=np.float64
            )
            c_p = c0 + eps * dC_arr
            q_p = q0 + eps * dQ_arr

            try:
                for nm, val in zip(center_names, c_p):
                    self.set(nm, float(val))
                for nm, val in zip(quat_names, q_p):
                    self.set(nm, float(val))
                self.recompute_currents()
                B_p = np.asarray(
                    self._B_at_points_tf_only(pts_arr), dtype=np.float64
                )
            except (np.linalg.LinAlgError, RuntimeError, ValueError) as exc:
                _warnings.warn(
                    "PSCBulkArray._geometry_fd_gradient: skipping "
                    f"perturbation index {k} (puck {i}) because the "
                    f"perturbed solve raised {exc.__class__.__name__}: {exc}",
                    category=RuntimeWarning,
                    stacklevel=3,
                )
                out[k] = 0.0
                for nm, val in zip(center_names, c0):
                    self.set(nm, float(val))
                for nm, val in zip(quat_names, q0):
                    self.set(nm, float(val))
                try:
                    self.recompute_currents()
                except Exception:
                    pass
                continue

            for nm, val in zip(center_names, c0):
                self.set(nm, float(val))
            for nm, val in zip(quat_names, q0):
                self.set(nm, float(val))
            self.recompute_currents()
            out[k] = float(np.sum(v_B_arr * (B_p - B0))) / eps

        self._mark_phase_end("vjp_geometry_fd", _t_total)
        return out

    def _vjp_puck_geometry(self, v_B, pts):
        """VJP w.r.t. puck center + quaternion + TF DOFs via full JAX forward."""
        if not self._has_free_puck_dofs():
            return self._vjp_tf_only(v_B, pts)
        if not self._has_free_center_or_quat_dofs():
            import warnings as _warnings

            _t_shape = self._mark_phase_start()
            _tf_failed = False
            try:
                tf_deriv = self._vjp_tf_only(v_B, pts)
            except (np.linalg.LinAlgError, RuntimeError, ValueError) as exc:
                # Phase 3 (May-2026): ``_Lr_chol_host`` can be ill-conditioned on
                # FC-perturbed initial states / bad line searches. Returning a
                # *zero* TF gradient lets L-BFGS-B treat ``g==0`` as valid and blow
                # coils to nonsense (see passive-bulk Phase 3 campaign notes /
                # ``toroidal_dof_scan_phase2_post`` freeradius logs). Propagate
                # NaN locally so scipy aborts the line search.
                _warnings.warn(
                    "PSCBulkArray._vjp_puck_geometry: shape-only TF VJP "
                    f"raised {exc.__class__.__name__}: {exc}; returning "
                    "NaN-filled gradient so scipy aborts the line search "
                    "(Phase 3 May-2026: replaces zero-gradient sentinel).",
                    category=RuntimeWarning,
                    stacklevel=3,
                )
                tf_deriv = Derivative({})
                _tf_failed = True
            if _tf_failed:
                grad_local = np.full(
                    self._n_geom_dofs,
                    np.nan,
                    dtype=float,
                )
                self._mark_phase_end("vjp_shape_only_total", _t_shape)
                return Derivative({self: grad_local}) + tf_deriv
            grad_local = np.zeros(self._n_geom_dofs)
            if self._has_free_Rt_dofs():
                _t_Rt_fd = self._mark_phase_start()
                rt_fn = (
                    self._Rt_analytic_directional_combo
                    if _psc_rt_analytic_combo_env()
                    else self._Rt_fd_gradient
                )
                g_Rt = rt_fn(np.asarray(v_B), np.asarray(pts))
                for i in range(self._n_base_pucks):
                    off = i * _DOFS_PER_PUCK
                    grad_local[off + 7] += g_Rt[i, 0]
                    grad_local[off + 8] += g_Rt[i, 1]
                self._mark_phase_end("vjp_free_Rt_fd", _t_Rt_fd)
            if self._has_free_normal_offset_dofs():
                _t_d_fd = self._mark_phase_start()
                g_d = self._normal_offset_fd_gradient(np.asarray(v_B), np.asarray(pts))
                for i in self._normal_offset_puck_indices:
                    grad_local[self._normal_offset_dof_index[i]] += g_d[i]
                self._mark_phase_end("vjp_free_normal_offset_fd", _t_d_fd)
            self._mark_phase_end("vjp_shape_only_total", _t_shape)
            return Derivative({self: grad_local}) + tf_deriv
        _t_stacks = self._mark_phase_start()
        self._ensure_jax_full()
        self._mark_phase_end("vjp_free_ensure_jax_full", _t_stacks)
        _t_tf_arrays = self._mark_phase_start()
        gammas = np.array([c.curve.gamma() for c in self.coils_TF])
        gammadashs = np.array([c.curve.gammadash() for c in self.coils_TF])
        currents = np.array([c.current.get_value() for c in self.coils_TF])
        self._mark_phase_end("vjp_free_tf_arrays", _t_tf_arrays)
        br = np.asarray(self._base_reps, dtype=np.int32)
        reduced_ok = bool(
            getattr(self, "_reduced_free_dof_active", False)
            and self._jax_Q_c_base is not None
            and not _psc_disable_reduced_free()
        )
        if reduced_ok:
            c_base = jnp.asarray(np.stack([self._all_pucks[i][0] for i in br], axis=0))
            q_base = jnp.asarray(self._all_pucks_quats[br])
            g_float = jnp.asarray(float(self._symmetry_G), dtype=jnp.float64)
            free_kinds = self._free_puck_dof_kinds()
            vjp_probe_mode = _psc_free_vjp_probe_env()
            solve_vjp_mode = _psc_free_solve_vjp_env()
            use_far, pair_k, tf_load, sol_mode, bs_far = (
                _reduced_free_dof_extras_for_jax(self._resolved_bulk_far_pair_kappa())
            )
            if use_far and pair_k > 0.0:
                c_np, q_np, _, _ = self._get_base_puck_geometry()
                near_idx, far_idx, _, _ = self._reduced_far_pair_indices(
                    c_np, q_np, float(pair_k)
                )
            else:
                near_idx = jnp.zeros((0, 2), dtype=jnp.int32)
                far_idx = jnp.zeros((0, 2), dtype=jnp.int32)
            m_loc_b = self._jax_m_local[br]
            q_loc_b = self._jax_Q_sym_local[br]
            _t_jax = self._mark_phase_start()
            cache = getattr(self, "_free_vjp_cache", None)
            cache_key = self._free_vjp_cache_key(np.asarray(pts), solve_vjp_mode)
            if (
                _psc_cache_free_vjp_env()
                and vjp_probe_mode == "full"
                and cache is not None
                and cache.get("key") == cache_key
            ):
                vc, vq, vg, vgd, vI = cache["pullback"](jnp.asarray(v_B))
                self._free_vjp_cache = None
                self._mark_phase_end(
                    "vjp_free_reduced_cached_pullback", _t_jax, (vc, vq, vg, vgd, vI)
                )
            else:
                common_args = (
                    self._jax_local_pts[br],
                    self._jax_local_K[br],
                    self._jax_local_n[br],
                    self._jax_local_w[br],
                    self._jax_local_phi[br],
                    self._jax_Q_c_base,
                    c_base,
                    q_base,
                    jnp.asarray(gammas),
                    jnp.asarray(gammadashs),
                    jnp.asarray(currents),
                    jnp.asarray(pts),
                    self._jax_L_red_eig_U,
                    self._jax_L_red_eig_lam,
                    jnp.asarray(v_B),
                    self._jax_base_indices,
                    self._jax_base_reps,
                    g_float,
                    int(self.nfp),
                    bool(self.stellsym),
                    float(self.regularization_delta),
                    float(self._eigenfloor_threshold),
                    bool(self.adaptive_self_reg),
                    bool(self._resolved_checkpoint_l_pairs()),
                    int(self._resolved_jax_pair_row_chunk()),
                    int(
                        self._resolved_bs_eval_chunk(
                            np.asarray(pts).reshape(-1, 3).shape[0]
                        )
                    ),
                    vjp_probe_mode,
                    solve_vjp_mode,
                    m_loc_b,
                    q_loc_b,
                    use_far,
                    float(pair_k),
                    str(tf_load),
                    str(sol_mode),
                    float(bs_far),
                    bool(_psc_w1_envelope_env()),
                    int(self._resolved_pair_replica_chunk()),
                    near_idx,
                    far_idx,
                )
                if _psc_free_vjp_geometry_env() == "full":
                    vc, vq, vg, vgd, vI = _vjp_reduced_free_dof_jitted(*common_args)
                elif free_kinds == {"quaternion"}:
                    vq, vg, vgd, vI = _vjp_reduced_free_quat_only_jitted(*common_args)
                    vc = jnp.zeros_like(c_base)
                elif free_kinds == {"center"}:
                    vc, vg, vgd, vI = _vjp_reduced_free_center_only_jitted(*common_args)
                    vq = jnp.zeros_like(q_base)
                else:
                    vc, vq, vg, vgd, vI = _vjp_reduced_free_dof_jitted(*common_args)
                self._mark_phase_end(
                    "vjp_free_reduced_jax", _t_jax, (vc, vq, vg, vgd, vI)
                )
        else:
            centers_all = np.stack([p[0] for p in self._all_pucks], axis=0)
            quats_all = self._all_pucks_quats
            _t_jax = self._mark_phase_start()
            vc, vq, vg, vgd, vI = _vjp_full_jitted(
                self._jax_local_pts,
                self._jax_local_K,
                self._jax_local_n,
                self._jax_local_w,
                self._jax_local_phi,
                self._jax_Q_c,
                jnp.asarray(centers_all),
                jnp.asarray(quats_all),
                jnp.asarray(gammas),
                jnp.asarray(gammadashs),
                jnp.asarray(currents),
                jnp.asarray(pts),
                jnp.asarray(v_B),
                float(self.regularization_delta),
                float(self._eigenfloor_threshold),
                bool(self.adaptive_self_reg),
                int(self._full_L_band_size),
                bool(self._resolved_checkpoint_l_pairs()),
                int(
                    self._resolved_bs_eval_chunk(
                        np.asarray(pts).reshape(-1, 3).shape[0]
                    )
                ),
            )
            self._mark_phase_end("vjp_free_full_jax", _t_jax, (vc, vq, vg, vgd, vI))
        vc = np.asarray(vc)
        vq = np.asarray(vq)
        vg = np.asarray(vg)
        vgd = np.asarray(vgd)
        vI = np.asarray(vI)

        _t_puck_pullback = self._mark_phase_start()
        # Sized to ``_n_geom_dofs``: this branch only runs when a real
        # center/quat DOF is free, a combination ``_check_normal_offset_
        # vjp_support`` forbids alongside any free ``d{i}``, so no FD term
        # is needed here -- but the array must still be long enough to
        # cover any (fixed) normal-offset DOFs configured on this array.
        grad_local = np.zeros(self._n_geom_dofs)
        if reduced_ok:
            for i in range(self._n_base_pucks):
                off = i * _DOFS_PER_PUCK
                grad_local[off : off + 3] = vc[i]
                grad_local[off + 3 : off + 7] = vq[i]
        else:
            for j in range(len(self._all_pucks)):
                base_idx = self._all_puck_base_indices[j]
                off = base_idx * _DOFS_PER_PUCK
                grad_local[off : off + 3] += self._center_jacobians[j].T @ vc[j]
                grad_local[off + 3 : off + 7] += self._quat_jacobians[j].T @ vq[j]
        self._mark_phase_end("vjp_free_puck_pullback", _t_puck_pullback)

        if self._has_free_Rt_dofs():
            _t_Rt_fd = self._mark_phase_start()
            rt_fn = (
                self._Rt_analytic_directional_combo
                if _psc_rt_analytic_combo_env()
                else self._Rt_fd_gradient
            )
            g_Rt = rt_fn(np.asarray(v_B), np.asarray(pts))
            for i in range(self._n_base_pucks):
                off = i * _DOFS_PER_PUCK
                grad_local[off + 7] += g_Rt[i, 0]
                grad_local[off + 8] += g_Rt[i, 1]
            self._mark_phase_end("vjp_free_Rt_fd", _t_Rt_fd)

        _t_tf_pullback = self._mark_phase_start()
        vjp_tf = sum(
            self.coils_TF[i].vjp(vg[i], vgd[i], np.asarray([vI[i]]))
            for i in range(len(self.coils_TF))
        )
        self._mark_phase_end("vjp_free_tf_pullback", _t_tf_pullback)
        return Derivative({self: grad_local}) + vjp_tf

    # ------------------------------------------------------------------
    # Diagnostic helpers
    # ------------------------------------------------------------------

    def n_null_modes(self) -> int:
        """Number of null modes removed from L."""
        return self._n_dof_total - self._Q.shape[1]

    @classmethod
    def from_toroidal_shell(
        cls,
        plasma_boundary,
        coils_TF,
        eval_points: np.ndarray,
        *,
        n_theta: int,
        n_phi_slices: int,
        R0: Optional[float] = None,
        minor_radius: Optional[float] = None,
        plasma_clearance: float = 0.0,
        puck_R: Optional[Union[float, np.ndarray]] = None,
        puck_t: Optional[Union[float, np.ndarray]] = None,
        safety: float = 1.05,
        nfp: Optional[int] = None,
        stellsym: Optional[bool] = None,
        m_fourier: int = 4,
        l_zernike: int = 6,
        k_chebyshev: int = 4,
        n_rho: int = 10,
        n_phi: int = 12,
        n_z: int = 6,
        regularization_delta: float = 1e-6,
        default_thickness: float = 0.02,
        strict_rim_continuity: bool = False,
        adaptive_self_reg: bool = True,
        exact_disc_faces: bool = False,
        n_radial_disc: int = 32,
    ) -> "PSCBulkArray":
        """Passive bulks on an axisymmetric toroidal shell enveloping the plasma.

        See :func:`~simsopt.field.puck_init.toroidal_shell_pucks` for the
        ``(R0, minor_radius, n_theta, n_phi_slices)`` parameterisation and
        spacing-inequality details; puck axes point **inward** from the
        shell toward the magnetic axis.

        Args:
            n_theta: Poloidal samples per :math:`\\theta` range.
            n_phi_slices: Toroidal samples per **half period**
                :math:`[0, \\pi/\\textrm{nfp})`.
            R0: Major radius of the shell (m); defaults to the mean of
                ``sqrt(x^2 + y^2)`` over ``plasma_boundary.gamma()``.
            minor_radius: Minor radius of the shell (m); defaults to
                ``rho_max(plasma) + plasma_clearance``.
            plasma_clearance: Headroom (m) added to the default
                ``minor_radius``.
            adaptive_self_reg: Forwarded to :class:`PSCBulkArray` (default
                ``True``: physically correct coincident-cell regularization).
            exact_disc_faces: If ``True``, after construction enable the
                semi-analytic disc-face self-block via
                ``psc.exact_disc_faces = True`` and rebuild ``L``.  Uses
                Duffy-type 1D quadrature with ``n_radial_disc`` radial
                nodes and is required to reach the Smythe thin-disc
                benchmark.
            n_radial_disc: Radial node count for the exact disc-face
                assembler; ignored when ``exact_disc_faces`` is
                ``False``.

        See :class:`PSCBulkArray` for the meaning of ``strict_rim_continuity``.
        """
        nfp_i = int(nfp if nfp is not None else plasma_boundary.nfp)
        stellsym_i = bool(
            stellsym if stellsym is not None
            else getattr(plasma_boundary, "stellsym", False)
        )
        centers, axes, radii, thicknesses = toroidal_shell_pucks(
            plasma_boundary,
            n_theta=int(n_theta),
            n_phi=int(n_phi_slices),
            R0=R0,
            minor_radius=minor_radius,
            plasma_clearance=plasma_clearance,
            safety=safety,
            puck_R=puck_R,
            puck_t=puck_t,
            nfp=nfp_i,
            stellsym=stellsym_i,
        )
        psc = cls(
            centers,
            axes,
            radii,
            thicknesses,
            coils_TF,
            eval_points=np.asarray(eval_points, dtype=float, order="C"),
            m_fourier=m_fourier,
            l_zernike=l_zernike,
            k_chebyshev=k_chebyshev,
            n_rho=n_rho,
            n_phi=n_phi,
            n_z=n_z,
            nfp=nfp_i,
            stellsym=bool(stellsym)
            if stellsym is not None
            else bool(plasma_boundary.stellsym),
            regularization_delta=regularization_delta,
            default_thickness=default_thickness,
            strict_rim_continuity=strict_rim_continuity,
            adaptive_self_reg=bool(adaptive_self_reg),
        )
        if exact_disc_faces:
            psc.exact_disc_faces = True
            psc.n_radial_disc = int(n_radial_disc)
            psc._rebuild()
        return psc

    @classmethod
    def from_winding_surface(
        cls,
        plasma_boundary,
        coils_TF,
        eval_points: np.ndarray,
        *,
        distance: float,
        n_phi_pucks: int,
        n_theta_pucks: int,
        puck_R: Optional[Union[float, np.ndarray]] = None,
        puck_t: Optional[float] = None,
        m_fourier: int = 4,
        l_zernike: int = 6,
        k_chebyshev: int = 4,
        n_rho: int = 10,
        n_phi: int = 12,
        n_z: int = 6,
        nfp: Optional[int] = None,
        stellsym: Optional[bool] = None,
        regularization_delta: float = 1e-6,
        default_thickness: float = 0.02,
        strict_rim_continuity: bool = False,
        adaptive_self_reg: bool = True,
        exact_disc_faces: bool = False,
        n_radial_disc: int = 32,
        center_parameterization: str = "xyz",
    ) -> "PSCBulkArray":
        """Passive bulks on a winding surface ``extend_via_normal(distance)`` from the plasma.

        Args:
            adaptive_self_reg: Forwarded to :class:`PSCBulkArray` (default
                ``True``: physically correct coincident-cell regularization).
            exact_disc_faces: If ``True``, after construction enable the
                semi-analytic disc-face self-block via
                ``psc.exact_disc_faces = True`` and rebuild ``L``.  Uses
                Duffy-type 1D quadrature with ``n_radial_disc`` radial
                nodes and is required to reach the Smythe thin-disc
                benchmark.
            n_radial_disc: Radial node count for the exact disc-face
                assembler; ignored when ``exact_disc_faces`` is
                ``False``.
            center_parameterization: ``"xyz"`` (default) gives every puck
                the usual independent ``center_x/y/z{i}`` DOFs. ``
                "normal_offset"`` instead gives every puck a single scalar
                DOF ``d{i}`` s.t. ``center_i = anchor_i + d_i * normal_i``,
                where ``anchor_i``/``normal_i`` are that puck's point/unit
                normal on the *raw* ``plasma_boundary`` (not the extended
                winding surface); ``d_i`` starts at exactly ``distance``.
                See :meth:`PSCBulkArray.unfix` and the ``normal_offset``
                constructor argument for details, and
                :func:`~simsopt.field.puck_init.winding_surface_pucks`
                for the anchor/normal sampling.

        See :class:`PSCBulkArray` for the meaning of ``strict_rim_continuity``.
        """
        nfp_i = int(nfp if nfp is not None else plasma_boundary.nfp)
        normal_offset = None
        if center_parameterization == "normal_offset":
            centers, axes, radii, thicknesses, anchors, plasma_normals = (
                winding_surface_pucks(
                    plasma_boundary,
                    distance=float(distance),
                    n_phi_pucks=int(n_phi_pucks),
                    n_theta_pucks=int(n_theta_pucks),
                    puck_R=puck_R,
                    puck_t=puck_t,
                    default_thickness=default_thickness,
                    center_parameterization="normal_offset",
                )
            )
            normal_offset = {
                "mask": np.ones(len(centers), dtype=bool),
                "anchors": anchors,
                "normals": plasma_normals,
            }
        elif center_parameterization == "xyz":
            centers, axes, radii, thicknesses = winding_surface_pucks(
                plasma_boundary,
                distance=float(distance),
                n_phi_pucks=int(n_phi_pucks),
                n_theta_pucks=int(n_theta_pucks),
                puck_R=puck_R,
                puck_t=puck_t,
                default_thickness=default_thickness,
            )
        else:
            raise ValueError(
                "center_parameterization must be 'xyz' or 'normal_offset'; "
                f"got {center_parameterization!r}"
            )
        psc = cls(
            centers,
            axes,
            radii,
            thicknesses,
            coils_TF,
            eval_points=np.asarray(eval_points, dtype=float, order="C"),
            m_fourier=m_fourier,
            l_zernike=l_zernike,
            k_chebyshev=k_chebyshev,
            n_rho=n_rho,
            n_phi=n_phi,
            n_z=n_z,
            nfp=nfp_i,
            stellsym=bool(stellsym)
            if stellsym is not None
            else bool(plasma_boundary.stellsym),
            regularization_delta=regularization_delta,
            default_thickness=default_thickness,
            strict_rim_continuity=strict_rim_continuity,
            adaptive_self_reg=bool(adaptive_self_reg),
            normal_offset=normal_offset,
        )
        if exact_disc_faces:
            psc.exact_disc_faces = True
            psc.n_radial_disc = int(n_radial_disc)
            psc._rebuild()
        return psc

    @classmethod
    def from_curves(
        cls,
        curves,
        coils_TF,
        eval_points: np.ndarray,
        *,
        radius: Optional[Union[float, np.ndarray]] = None,
        thickness: Optional[Union[float, np.ndarray]] = None,
        m_fourier: int = 4,
        l_zernike: int = 6,
        k_chebyshev: int = 4,
        n_rho: int = 10,
        n_phi: int = 12,
        n_z: int = 6,
        nfp: int = 1,
        stellsym: bool = False,
        regularization_delta: float = 1e-6,
        default_thickness: float = 0.02,
        strict_rim_continuity: bool = False,
        adaptive_self_reg: bool = True,
        center_parameterization: str = "xyz",
        solver_mode: str = "energy",
        **kwargs: Any,
    ) -> "PSCBulkArray":
        """Passive bulk pucks placed directly from a list of planar coil curves.

        Reuses each curve's own center and orientation, so it composes
        naturally with e.g.
        :func:`~simsopt.util.dipole_array_helper_functions.generate_windowpane_metric_ring_array`
        (or the other ``generate_windowpane_*`` functions) -- pass their
        returned curves straight in.  See :func:`~simsopt.field.puck_init.curves_to_pucks`
        for the center/axis/radius conversion and its approximation caveat
        (a puck is a circular disc; the input curves need not be).

        Args:
            curves: sequence of planar coil curves (e.g. ``CurvePlanarFourier``)
                with ``get('X'/'Y'/'Z'/'q0'/'qi'/'qj'/'qk')`` dofs.
            radius: forwarded to :func:`~simsopt.field.puck_init.curves_to_pucks`;
                ``None`` (default) uses each curve's own area-equivalent radius.
            thickness: forwarded to :func:`~simsopt.field.puck_init.curves_to_pucks`;
                ``None`` (default) uses ``default_thickness`` for every puck.
            center_parameterization: ``"xyz"`` (default) gives every puck
                the usual independent ``center_x/y/z{i}`` DOFs. ``
                "normal_offset"`` instead gives every puck a single scalar
                DOF ``d{i}`` s.t. ``center_i = anchor_i + d_i * normal_i``,
                where ``anchor_i`` is that curve's own center and
                ``normal_i`` is that curve's own axis (both from
                :func:`~simsopt.field.puck_init.curves_to_pucks`, i.e. the
                *initial* center/orientation baked in at construction
                time -- moving ``d_i`` away from ``0`` slides the puck along
                its own face-normal, off the curve's original plane).
                ``d_i`` starts at exactly ``0``. See
                :meth:`PSCBulkArray.unfix` for details.
            solver_mode: ``"energy"`` (default), ``"shell_l2"``, or
                ``"dipole"`` -- forwarded verbatim to the
                :class:`PSCBulkArray` constructor. ``"dipole"`` collapses
                every base puck to a single point magnetic dipole moment
                (reduced system size ``(3*n_base, 3*n_base)``); off-diagonal
                puck-pair blocks use the closed-form dipole-dipole mutual
                inductance (:func:`~simsopt.field.bulk_multipole.pair_inductance_dipole_block`)
                instead of a quadrature-quadrature double integral, so it
                avoids the large intermediate tensors the default
                ``"energy"`` solver can hit at high puck counts. It's a
                leading-order far-field truncation: a :class:`UserWarning`
                is raised on the first rebuild if any puck pair is closer
                than ``2 * (R_i + R_j)`` (configurable via
                :envvar:`SIMSOPT_PSC_DIPOLE_WARN_R_CLOSE`), since truncation
                error grows quickly at close packing. See :class:`PSCBulkArray`'s
                ``solver_mode`` docs for ``"shell_l2"``.
            **kwargs: Forwarded verbatim to the :class:`PSCBulkArray`
                constructor -- e.g. ``bulk_far_pair_kappa``,
                ``bulk_far_pair_tol``, ``mode_truncate``,
                ``checkpoint_l_pairs``, ``full_L_band_size``, etc. Anything
                not explicitly listed above as a named parameter of this
                classmethod can still be reached this way.

        See :class:`PSCBulkArray` for the meaning of the remaining arguments.
        """
        centers, axes, radii, thicknesses = curves_to_pucks(
            curves,
            radius=radius,
            thickness=thickness,
            default_thickness=default_thickness,
        )
        normal_offset = None
        if center_parameterization == "normal_offset":
            normal_offset = {
                "mask": np.ones(len(centers), dtype=bool),
                "anchors": centers.copy(),
                "normals": axes.copy(),
            }
        elif center_parameterization != "xyz":
            raise ValueError(
                "center_parameterization must be 'xyz' or 'normal_offset'; "
                f"got {center_parameterization!r}"
            )
        return cls(
            centers,
            axes,
            radii,
            thicknesses,
            coils_TF,
            eval_points=np.asarray(eval_points, dtype=float, order="C"),
            m_fourier=m_fourier,
            l_zernike=l_zernike,
            k_chebyshev=k_chebyshev,
            n_rho=n_rho,
            n_phi=n_phi,
            n_z=n_z,
            nfp=nfp,
            stellsym=stellsym,
            regularization_delta=regularization_delta,
            default_thickness=default_thickness,
            strict_rim_continuity=strict_rim_continuity,
            adaptive_self_reg=bool(adaptive_self_reg),
            normal_offset=normal_offset,
            solver_mode=solver_mode,
            **kwargs,
        )


# ======================================================================
# PassiveBulkField
# ======================================================================


class PassiveBulkField(MagneticField):
    """Magnetic field from sheet currents on passive bulk pucks."""

    def __init__(self, psc_bulk: PSCBulkArray):
        self.psc_bulk = psc_bulk
        MagneticField.__init__(self, depends_on=[psc_bulk])

    def set_points_cart(self, xyz):
        """Set eval points and invalidate the fixed-puck ``M_field`` cache.

        The cached ``_M_field`` matrix on the owning :class:`PSCBulkArray`
        is keyed in part on
        :attr:`PSCBulkArray._eval_pts_version`; bumping that version via
        :meth:`PSCBulkArray._bump_eval_pts_version` here guarantees the
        cache is rebuilt on the next forward, regardless of whether the
        C++ side reuses the prior points buffer in-place or allocates a
        fresh one.  Called both by direct user code and by the parent
        :class:`MagneticFieldSum`'s ``_set_points_cb`` dispatch.
        """
        self.psc_bulk._bump_eval_pts_version()
        return super().set_points_cart(xyz)

    def _B_impl(self, B):
        pts = self.get_points_cart_ref()
        B[:] = self.psc_bulk.B_at_points(np.asarray(pts))

    def _dB_by_dX_impl(self, dB):
        """Spatial Jacobian :math:`\\partial B_i / \\partial x_j` of the passive
        bulk field.

        Not yet implemented.  Previous behavior silently returned a zero
        Jacobian, which caused any downstream consumer (particle tracing,
        :class:`~simsopt.field.boozermagneticfield.BoozerMagneticField`
        construction, :math:`\\nabla B`-based objectives) to see wrong zeros
        with no warning.  A follow-up will implement the analytic Jacobian
        by differentiating the shell Biot-Savart kernel
        :math:`\\mathbf{K}(\\mathbf{r}') \\times (\\mathbf{x}-\\mathbf{r}') /
        |\\mathbf{x}-\\mathbf{r}'|^3` w.r.t. ``x``.
        """
        raise NotImplementedError(
            "PassiveBulkField.dB_by_dX is not implemented.  Previously this "
            "method silently returned zeros, which is incorrect for any "
            "consumer that consumes the spatial gradient (particle tracing, "
            "BoozerMagneticField construction, grad-B objectives).  File a "
            "feature request or implement the analytic shell Biot-Savart "
            "Jacobian."
        )

    def B_vjp(self, v):
        v = np.asarray(v).reshape(-1, 3)
        pts = np.asarray(self.get_points_cart_ref())
        return self.psc_bulk.vjp_setup_B(v, pts)

    def invalidate(self) -> None:
        """Clear all cached field values and propagate the invalidation.

        After calling :meth:`PSCBulkArray.recompute_currents` (or changing
        TF coil DOFs while the puck DOFs are fixed) the stored modal
        coefficients are fresh but any :class:`MagneticField` /
        :class:`MagneticFieldSum` that previously evaluated ``B`` at a
        set of points still holds stale cached values.  Call this to
        invalidate both this object's cache and any downstream consumer
        (via the :class:`~simsopt._core.optimizable.Optimizable` child
        graph) such as a :class:`MagneticFieldSum` that this field was
        added to.

        This is the canonical dirty-flagging entry point for the passive
        bulk field.  Previously tests used
        ``btot.clear_cached_properties()`` and
        ``btot.Bfields[0].invalidate_cache()`` interchangeably; both are
        still valid, but :meth:`invalidate` makes the intent explicit.
        """
        self.clear_cached_properties()
        for weakref_child in list(self._children):
            child = weakref_child()
            if child is None:
                continue
            clear = getattr(child, "clear_cached_properties", None)
            if callable(clear):
                clear()


def make_bulk_plus_tf_field(psc_bulk: PSCBulkArray, coils_tf):
    """Return ``PassiveBulkField(psc_bulk) + BiotSavart(coils_tf)``."""
    return psc_bulk.biot_savart + BiotSavart(coils_tf)
