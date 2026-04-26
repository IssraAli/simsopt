import os

import jax

jax.config.update("jax_enable_x64", True)
# Persistent on-disk compilation cache (speeds repeat runs of scripts that
# would otherwise pay ``backend_compile_and_load`` every process). Override
# with env ``SIMSOPT_JAX_CACHE`` (expanded with os.path.expanduser).
_jax_cache_dir = os.path.expanduser(
    os.environ.get(
        "SIMSOPT_JAX_CACHE",
        os.path.join(os.path.expanduser("~"), ".cache", "simsopt", "jax"),
    )
)
try:
    os.makedirs(_jax_cache_dir, mode=0o755, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", _jax_cache_dir)
    # Compile even small functions so the cache is useful in unit tests.
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
except OSError:
    # Read-only environments: skip cache dir setup.
    pass
from .config import *

from .curve import *
from .curvehelical import *
from .curverzfourier import *
from .curvexyzfourier import *
from .curvexyzfouriersymmetries import *
from .curveperturbed import *
from .curveobjectives import *
from .curveplanarfourier import *
from .curveplanarellipticalcylindrical import *
from .framedcurve import *
from .finitebuild import *
from .plotting import *

from .boozersurface import *
from .qfmsurface import *
from .surface import *
from .surfacegarabedian import *
from .surfacehenneberg import *
from .surfaceobjectives import *
from .surfacerzfourier import *
from .surfacexyzfourier import *
from .surfacexyztensorfourier import *
from .strain_optimization import *
from .wireframe_toroidal import *
from .ports import *

from .permanent_magnet_grid import *

__all__ = (
    curve.__all__
    + curvehelical.__all__
    + curverzfourier.__all__
    + curvexyzfourier.__all__
    + curvexyzfouriersymmetries.__all__
    + curveperturbed.__all__
    + curveobjectives.__all__
    + curveplanarfourier.__all__
    + curveplanarellipticalcylindrical.__all__
    + finitebuild.__all__
    + plotting.__all__
    + boozersurface.__all__
    + qfmsurface.__all__
    + surface.__all__
    + surfacegarabedian.__all__
    + surfacehenneberg.__all__
    + surfacerzfourier.__all__
    + surfacexyzfourier.__all__
    + surfacexyztensorfourier.__all__
    + surfaceobjectives.__all__
    + strain_optimization.__all__
    + framedcurve.__all__
    + wireframe_toroidal.__all__
    + ports.__all__
    + permanent_magnet_grid.__all__
)
