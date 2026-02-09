from .cache import (
    cache_coloring,
    cache_mpi_metadata,
    with_cache,
)
from .jaxamg import (
    AMGXStatus,
    amg_solve,
)

# User-friendly alias
solve = amg_solve

__all__ = [
    "amg_solve",
    "solve",
    "with_cache",
    "cache_coloring",
    "cache_mpi_metadata",
    "AMGXStatus",
]
