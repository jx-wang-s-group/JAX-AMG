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
from .sharding import make_sharded_solver, make_sharded_vector, solve_sharded
from .sparsity import cache_coloring

__all__ = [
    "__version__",
    "solve",
    "with_cache",
    "cache_coloring",
    "cache_mpi_metadata",
    "make_sharded_solver",
    "make_sharded_vector",
    "solve_sharded",
    "AMGXStatus",
    "make_preconditioner",
    "make_lineax_preconditioner",
    "clear_solver_cache",
    "get_solver_cache_info",
    "finalize",
]
