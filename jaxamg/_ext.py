"""Loads the compiled AmgX native extension with an actionable error message.

The native extension (``jaxamg._amgx``) links against AmgX (``libamgxsh.so``)
and the CUDA runtime. When those shared libraries are not discoverable at import
time, the raw ``ImportError`` from the C extension is hard to act on, so we wrap
it with installation guidance. Import the extension from here
(``from ._ext import _amgx``) rather than directly.
"""

try:
    from . import _amgx
except ImportError as exc:  # pragma: no cover - environment-specific
    raise ImportError(
        "jaxamg's native extension (jaxamg._amgx) could not be loaded. This "
        "usually means AmgX or the CUDA runtime is not on your library path. "
        "Ensure libamgxsh.so and the CUDA libraries are discoverable, e.g.:\n\n"
        "    export LD_LIBRARY_PATH=$AMGX_BUILD:$CUDA_HOME/lib64:$LD_LIBRARY_PATH\n\n"
        "and that jaxamg was built against your AmgX/CUDA installation. See "
        "https://jx-wang-s-group.github.io/JAX-AMG/install/ for details."
    ) from exc

__all__ = ["_amgx"]
