"""Tests for the opt-in fp32 mixed-precision path of :class:`SquaredFlux.dJ`.

These tests exercise the behaviour added by the
``SIMSOPT_SQUARED_FLUX_FP32`` opt-in: when enabled, :meth:`SquaredFlux.dJ`
must build its ``dJdB`` cotangent in ``float32`` and pass it to the field
VJP, then up-cast the resulting :class:`Derivative` back to ``float64`` so
the optimizer's parameter vector stays fp64.

The tests use lightweight stubs (no surfaces / no compiled simsopt extension
code) so the assertions isolate the SquaredFlux gradient pathway itself
rather than the dtype handling of the underlying field implementation.
"""

from __future__ import annotations

import importlib
import os
import unittest
from typing import Optional

import numpy as np

from simsopt._core.derivative import Derivative
from simsopt._core.optimizable import Optimizable
from simsopt.objectives import fluxobjective as fluxobjective_module
from simsopt.objectives.fluxobjective import (
    _SQUARED_FLUX_FP32_ENV_VAR,
    SquaredFlux,
)


class _StubSurface:
    """Minimal surface stub exposing :meth:`gamma` and :meth:`normal`.

    The arrays are random fp64 tensors; the actual geometry is irrelevant
    for verifying gradient-dtype propagation, only their shapes and dtypes
    matter.
    """

    def __init__(self, nphi: int = 4, ntheta: int = 5, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self._gamma = rng.standard_normal((nphi, ntheta, 3))
        # Keep the normal away from zero so unitn is well-defined.
        self._normal = rng.standard_normal((nphi, ntheta, 3)) + 0.5

    def gamma(self) -> np.ndarray:
        return self._gamma

    def normal(self) -> np.ndarray:
        return self._normal


class _StubField(Optimizable):
    """Minimal field stub: ``B(p) = (M @ x).reshape(npts, 3)``.

    The VJP applies ``M.T`` to the (flattened) cotangent ``v``. The dtype
    of ``v`` is preserved into the returned :class:`Derivative` so the test
    can verify that the SquaredFlux fp32 path
        (a) actually hands an fp32 cotangent to ``B_vjp``, and
        (b) up-casts the dof-derivative back to fp64 before returning.
    """

    def __init__(self, n_pts: int, ndofs: int = 6, seed: int = 1) -> None:
        rng = np.random.default_rng(seed)
        self._M = rng.standard_normal((3 * n_pts, ndofs))
        self._n_pts = n_pts
        self._pts: Optional[np.ndarray] = None
        self.last_v_dtype: Optional[np.dtype] = None
        Optimizable.__init__(self, x0=rng.standard_normal(ndofs))

    def set_points(self, pts: np.ndarray) -> "_StubField":
        self._pts = np.asarray(pts)
        return self

    def B(self) -> np.ndarray:
        x = np.asarray(self.x, dtype=np.float64)
        return (self._M @ x).reshape(self._n_pts, 3)

    def B_vjp(self, v: np.ndarray) -> Derivative:
        v_arr = np.asarray(v)
        self.last_v_dtype = v_arr.dtype
        v_flat = v_arr.reshape(-1)
        if v_arr.dtype == np.float32:
            # Mimic a JAX kernel that runs in fp32 when the cotangent is fp32:
            # cast the operator down so the resulting dof-derivative is fp32
            # and the SquaredFlux up-cast is exercised.
            grad = (self._M.astype(np.float32).T @ v_flat).astype(np.float32)
        else:
            grad = self._M.T @ v_flat.astype(np.float64)
        return Derivative({self: grad})


def _make_squared_flux(definition: str = "quadratic flux") -> SquaredFlux:
    """Build a tiny :class:`SquaredFlux` wrapped around the stub surface/field."""
    surf = _StubSurface(nphi=4, ntheta=5, seed=0)
    n_pts = surf.normal().shape[0] * surf.normal().shape[1]
    field = _StubField(n_pts=n_pts, ndofs=6, seed=1)
    target = np.asarray(0.1 * np.arange(n_pts).reshape(surf.normal().shape[:2]))
    return SquaredFlux(surf, field, target=target, definition=definition)


class _Fp32EnvGuard:
    """Context manager that sets/clears the fp32 env var and resets the latch."""

    def __init__(self, value: Optional[str]) -> None:
        self._value = value
        self._prev: Optional[str] = None
        self._prev_latch: bool = False

    def __enter__(self) -> "_Fp32EnvGuard":
        self._prev = os.environ.get(_SQUARED_FLUX_FP32_ENV_VAR)
        if self._value is None:
            os.environ.pop(_SQUARED_FLUX_FP32_ENV_VAR, None)
        else:
            os.environ[_SQUARED_FLUX_FP32_ENV_VAR] = self._value
        # Reset the one-shot logging latch so each test exercises the banner
        # path independently.
        self._prev_latch = fluxobjective_module._squared_flux_fp32_logged
        fluxobjective_module._squared_flux_fp32_logged = False
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._prev is None:
            os.environ.pop(_SQUARED_FLUX_FP32_ENV_VAR, None)
        else:
            os.environ[_SQUARED_FLUX_FP32_ENV_VAR] = self._prev
        fluxobjective_module._squared_flux_fp32_logged = self._prev_latch


class SquaredFluxFp32Tests(unittest.TestCase):
    """Verify the opt-in fp32 SquaredFlux gradient path."""

    def test_env_helper_recognises_truthy_values(self) -> None:
        """``_squared_flux_fp32_enabled`` parses the env var as documented."""
        for raw, expected in [
            (None, False),
            ("", False),
            ("0", False),
            ("false", False),
            ("no", False),
            ("1", True),
            ("True", True),
            ("yes", True),
            ("ON", True),
            ("  1  ", True),
        ]:
            with _Fp32EnvGuard(raw):
                self.assertEqual(
                    fluxobjective_module._squared_flux_fp32_enabled(),
                    expected,
                    msg=f"raw={raw!r}",
                )

    def test_default_path_unchanged_dtype(self) -> None:
        """With the env var unset the cotangent reaching the field VJP is fp64."""
        with _Fp32EnvGuard(None):
            jf = _make_squared_flux()
            grad = jf.dJ()
            self.assertEqual(grad.dtype, np.float64)
            self.assertEqual(jf.field.last_v_dtype, np.float64)

    def test_fp32_path_downcasts_v_and_upcasts_grad(self) -> None:
        """With the env var set, B_vjp sees fp32 ``v`` but the gradient stays fp64."""
        for definition in ["quadratic flux", "normalized", "local"]:
            with self.subTest(definition=definition), _Fp32EnvGuard("1"):
                jf = _make_squared_flux(definition=definition)
                deriv = jf.dJ(partials=True)
                grad = deriv(jf)

                # The field VJP must have received a float32 cotangent.
                self.assertEqual(
                    jf.field.last_v_dtype,
                    np.float32,
                    msg=f"definition={definition}",
                )

                # The Derivative's stored dof-derivative arrays must be
                # up-cast to float64 before returning to the caller.
                for v in deriv.data.values():
                    self.assertEqual(np.asarray(v).dtype, np.float64)

                # The flattened gradient handed to the optimizer is fp64.
                self.assertEqual(grad.dtype, np.float64)

    def test_fp32_gradient_matches_fp64_within_tolerance(self) -> None:
        """The fp32 mixed-precision gradient is close to the fp64 reference."""
        for definition in ["quadratic flux", "normalized", "local"]:
            with self.subTest(definition=definition):
                with _Fp32EnvGuard(None):
                    jf64 = _make_squared_flux(definition=definition)
                    grad64 = jf64.dJ()
                with _Fp32EnvGuard("1"):
                    jf32 = _make_squared_flux(definition=definition)
                    grad32 = jf32.dJ()

                self.assertEqual(grad64.shape, grad32.shape)
                # Generous tolerance; the operator has condition number
                # comparable to a small dense Gaussian and fp32 retains ~7
                # decimals which dominates the relative-error budget here.
                np.testing.assert_allclose(
                    grad32,
                    grad64,
                    rtol=1e-4,
                    atol=1e-6 * (np.abs(grad64).max() + 1.0),
                    err_msg=f"definition={definition}",
                )

    def test_fp32_banner_emitted_once(self) -> None:
        """The fp32 banner is printed only on the first dJ call per process."""
        # Importing in a clean state mirrors fresh-process behaviour for the
        # latch; we manually reset it below.
        importlib.reload(fluxobjective_module)
        # The reload above re-binds the SquaredFlux symbol used by the rest of
        # the tests in this module; make sure subsequent tests still use the
        # current module-level reference.
        from simsopt.objectives.fluxobjective import SquaredFlux as ReloadedSF

        with _Fp32EnvGuard("1"):
            surf = _StubSurface(nphi=3, ntheta=3, seed=2)
            n_pts = surf.normal().shape[0] * surf.normal().shape[1]
            field = _StubField(n_pts=n_pts, ndofs=4, seed=3)
            jf = ReloadedSF(surf, field, definition="quadratic flux")
            from io import StringIO
            from contextlib import redirect_stdout

            buf = StringIO()
            with redirect_stdout(buf):
                jf.dJ()
                jf.dJ()
            out = buf.getvalue()
            self.assertEqual(
                out.count(_SQUARED_FLUX_FP32_ENV_VAR),
                1,
                msg=f"expected exactly one banner line, got: {out!r}",
            )


if __name__ == "__main__":
    unittest.main()
