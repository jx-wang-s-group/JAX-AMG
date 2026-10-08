from ._version import __version__
from .cache import (
    cache_mpi_metadata,
    with_cache,
)
from .global_operator import GlobalOperator, global_operator
from .halo import HaloOperator, halo_operator
from .jaxamg import (
    AMGXStatus,
    clear_solver_cache,
    finalize,
    get_solver_cache_info,
    solve,
)
from .nullspace import NullSpaceWarning
from .patterns import Pattern, pattern
from .preconditioners import make_lineax_preconditioner, make_preconditioner
from .sharding import (
    ShardedMatrix,
    ShardedSolve,
    make_sharded_matrix,
    make_sharded_solver,
    make_sharded_vector,
)
from .sparsity import cache_coloring

__all__ = [
    "__version__",
    "solve",
    "with_cache",
    "cache_coloring",
    "Pattern",
    "pattern",
    "cache_mpi_metadata",
    "halo_operator",
    "global_operator",
    "GlobalOperator",
    "HaloOperator",
    "ShardedMatrix",
    "ShardedSolve",
    "make_sharded_matrix",
    "make_sharded_solver",
    "make_sharded_vector",
    "AMGXStatus",
    "NullSpaceWarning",
    "make_preconditioner",
    "make_lineax_preconditioner",
    "clear_solver_cache",
    "get_solver_cache_info",
    "finalize",
]
