from ._version import __version__
from .cache import (
    cache_mpi_metadata,
    with_cache,
)
from .jaxamg import (
    AMGXStatus,
    clear_solver_cache,
    finalize,
    get_solver_cache_info,
    solve,
)
from .preconditioners import make_lineax_preconditioner, make_preconditioner
from .sparsity import cache_coloring

__all__ = [
    "__version__",
    "solve",
    "with_cache",
    "cache_coloring",
    "cache_mpi_metadata",
    "AMGXStatus",
    "make_preconditioner",
    "make_lineax_preconditioner",
    "clear_solver_cache",
    "get_solver_cache_info",
    "finalize",
]
