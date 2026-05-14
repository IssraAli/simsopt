import os

import numpy as np

import simsoptpp as sopp
from .._core.optimizable import Optimizable
from .._core.derivative import Derivative, derivative_dec


__all__ = ["SquaredFlux"]


_SQUARED_FLUX_FP32_ENV_VAR = "SIMSOPT_SQUARED_FLUX_FP32"
_SQUARED_FLUX_FP32_TRUTHY = frozenset({"1", "true", "yes", "on"})

# Module-level latch so the opt-in fp32 path only emits its banner once per
# process. Reset only by re-importing the module (cheap; intended for run.log
# visibility, not for fine-grained diagnostics).
_squared_flux_fp32_logged = False


def _squared_flux_fp32_enabled() -> bool:
    """Return whether the SquaredFlux fp32 mixed-precision gradient is opted in.

    The opt-in is controlled by the ``SIMSOPT_SQUARED_FLUX_FP32`` environment
    variable. It is interpreted as a truthy value if (case-insensitively, after
    stripping surrounding whitespace) it equals one of ``{"1", "true", "yes",
    "on"}``. Any other value (including unset, empty string, or ``"0"``) leaves
    the default fp64 gradient behaviour in place.

    Returns:
        ``True`` when the environment variable is set to a truthy value, in
        which case :meth:`SquaredFlux.dJ` should down-cast its intermediate
        cotangent arrays to ``float32`` before invoking the field VJP and
        up-cast the resulting derivative back to ``float64``. ``False``
        otherwise.
    """
    raw = os.environ.get(_SQUARED_FLUX_FP32_ENV_VAR)
    if raw is None:
        return False
    return raw.strip().lower() in _SQUARED_FLUX_FP32_TRUTHY


def _maybe_log_squared_flux_fp32_active() -> None:
    """Emit a one-shot info banner the first time the fp32 dJ path executes.

    Side-effects only: prints to stdout (which `run.log` captures for the
    benchmark harness) and flips a module-level latch so subsequent
    :meth:`SquaredFlux.dJ` calls in the same process stay quiet.
    """
    global _squared_flux_fp32_logged
    if _squared_flux_fp32_logged:
        return
    _squared_flux_fp32_logged = True
    print(
        f"[SquaredFlux] {_SQUARED_FLUX_FP32_ENV_VAR}=1 active: "
        "computing dJdB in float32 and up-casting field VJP result back to float64.",
        flush=True,
    )


def _upcast_derivative_to_fp64(deriv: Derivative) -> Derivative:
    """Up-cast every entry of a :class:`Derivative` to ``float64`` in-place.

    Used by the SquaredFlux fp32 gradient path so that the gradient handed
    back to the optimizer's parameter vector stays in fp64 even when the
    field VJP was driven with an fp32 cotangent (and therefore returned
    fp32-typed dof-derivative arrays).

    Args:
        deriv: The :class:`Derivative` returned by ``self.field.B_vjp(...)``
            under the fp32 path.

    Returns:
        The same :class:`Derivative` instance, with each value array replaced
        by an fp64 copy when its dtype was not already ``np.float64``.
    """
    data = deriv.data
    for k, v in list(data.items()):
        arr = np.asarray(v)
        if arr.dtype != np.float64:
            data[k] = arr.astype(np.float64, copy=False)
    return deriv


class SquaredFlux(Optimizable):
    r"""
    Objective representing quadratic-flux-like quantities, useful for stage-2
    coil optimization. Several variations are available, which can be selected
    using the ``definition`` argument. For ``definition="quadratic flux"``
    (the default), the objective is defined as

    .. math::
        J = \frac12 \int_{S} (\mathbf{B}\cdot \mathbf{n} - B_T)^2 ds,

    where :math:`\mathbf{n}` is the surface unit normal vector and
    :math:`B_T` is an optional (zero by default) target value for the
    magnetic field. Also :math:`\int_{S} ds` indicates a surface integral.
    For ``definition="normalized"``, the objective is defined as

    .. math::
        J = \frac12 \frac{\int_{S} (\mathbf{B}\cdot \mathbf{n} - B_T)^2 ds}
                         {\int_{S} |\mathbf{B}|^2 ds}.

    For ``definition="local"``, the objective is defined as

    .. math::
        J = \frac12 \int_{S} \frac{(\mathbf{B}\cdot \mathbf{n} - B_T)^2}{|\mathbf{B}|^2} ds.

    The definition ``"quadratic flux"`` has the advantage of simplicity, and it
    is used in other contexts such as REGCOIL. However for stage-2 optimization,
    the optimizer can "cheat", lowering this objective by reducing the magnitude
    of the field. The definitions ``"normalized"`` and ``"local"`` close this loophole.

    Args:
        surface: A :obj:`simsopt.geo.surface.Surface` object on which to compute the flux
        field: A :obj:`simsopt.field.magneticfield.MagneticField` for which to compute the flux.
            May include :class:`~simsopt.field.psc_bulk.PassiveBulkField` in a :class:`~simsopt.field.magneticfield.MagneticFieldSum`.
        target: A ``nphi x ntheta`` numpy array containing target values for the flux. Here
          ``nphi`` and ``ntheta`` correspond to the number of quadrature points on `surface`
          in ``phi`` and ``theta`` direction.
        definition: A string to select among the definitions above. The
          available options are ``"quadratic flux"``, ``"normalized"``, and ``"local"``.
    """

    def __init__(
        self, surface, field, target=None, definition="quadratic flux", threshold=0.0
    ):
        self.surface = surface
        if target is not None:
            self.target = np.ascontiguousarray(target)
        else:
            self.target = np.zeros(self.surface.normal().shape[:2])
        self.field = field
        xyz = self.surface.gamma()
        self.field.set_points(xyz.reshape((-1, 3)))
        if definition not in ["quadratic flux", "normalized", "local"]:
            raise ValueError("Unrecognized option for 'definition'.")
        self.definition = definition
        self.threshold = threshold
        Optimizable.__init__(self, x0=np.asarray([]), depends_on=[field])

    def J(self):
        n = self.surface.normal()
        Bcoil = self.field.B().reshape(n.shape)
        sq_flux = sopp.integral_BdotN(Bcoil, self.target, n, self.definition)
        if sq_flux < self.threshold:
            return 0.0
        else:
            return sq_flux

    @derivative_dec
    def dJ(self):
        fp32 = _squared_flux_fp32_enabled()
        if fp32:
            _maybe_log_squared_flux_fp32_active()

            def _cast(a):
                """Down-cast ``a`` to ``np.float32`` for the fp32 dJdB path."""
                return np.asarray(a, dtype=np.float32)
        else:

            def _cast(a):
                """Identity passthrough for the default fp64 dJdB path."""
                return a

        n = _cast(self.surface.normal())
        absn = np.linalg.norm(n, axis=2)
        unitn = n * (1.0 / absn)[:, :, None]
        Bcoil = _cast(self.field.B().reshape(n.shape))
        Bcoil_n = np.sum(Bcoil * unitn, axis=2)
        if self.target is not None:
            B_n = Bcoil_n - _cast(self.target)
        else:
            B_n = Bcoil_n

        if self.definition == "quadratic flux":
            dJdB = (B_n[..., None] * unitn * absn[..., None]) / absn.size
            dJdB = dJdB.reshape((-1, 3))

        elif self.definition == "local":
            mod_Bcoil = np.linalg.norm(Bcoil, axis=2)
            dJdB = (
                (
                    (B_n / mod_Bcoil)[..., None]
                    * (
                        unitn / mod_Bcoil[..., None]
                        - (B_n / mod_Bcoil**3)[..., None] * Bcoil
                    )
                )
                * absn[..., None]
            ) / absn.size

        elif self.definition == "normalized":
            mod_Bcoil = np.linalg.norm(Bcoil, axis=2)
            num = np.mean(B_n**2 * absn)
            denom = np.mean(mod_Bcoil**2 * absn)

            dnum = 2 * (B_n[..., None] * unitn * absn[..., None]) / absn.size
            ddenom = 2 * (Bcoil * absn[..., None]) / absn.size
            dJdB = 0.5 * (dnum / denom - num * ddenom / denom**2)

        else:
            raise ValueError("Should never get here")

        dJdB = dJdB.reshape((-1, 3))
        if fp32:
            dJdB = np.ascontiguousarray(dJdB, dtype=np.float32)
        if np.isclose(self.J(), 0.0, atol=1e-10, rtol=1e-10):
            deriv = self.field.B_vjp(np.zeros_like(dJdB))
        else:
            deriv = self.field.B_vjp(dJdB)
        if fp32:
            _upcast_derivative_to_fp64(deriv)
        return deriv
