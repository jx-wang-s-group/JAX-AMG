"""
Caching utilities.

This module provides functions to cache metadata, enabling efficient usage with JAX JIT compilation.
"""

import jax
import jax.numpy as jnp
import numpy as np

from . import config as amgx_config
from .utils import *

from typing import Any, Callable, TYPE_CHECKING
from jax.typing import ArrayLike


if TYPE_CHECKING:
    from mpi4py.MPI import Comm


def with_cache(
    A: MatrixOrOperator,
    *,
    coloring: (
        tuple[np.ndarray, np.ndarray, np.ndarray, int, tuple[int, int]] | None
    ) = None,
    mpi: dict[str, Any] | None = None,
    is_symmetric: bool = False,
) -> MatrixOrOperator:
    """
    Attach cached metadata (coloring, MPI info, or symmetry) to a matrix or operator.

    This cache allows using matrices/operators inside JIT-compiled functions
    without recomputing metadata or passing it as separate arguments.

    Args:
        A: A matrix or operator.
        coloring: Cached coloring information from `cache_coloring()`.
        mpi: Cached MPI metadata from `cache_mpi_metadata()`.
        is_symmetric: If True, indicates the matrix is symmetric, allowing
                      optimizations like skipping transpose in backward pass.

    Returns:
        The same matrix/operator with requested cache attached.
    """
    if coloring is not None:
        try:
            object.__setattr__(A, "_coloring_info", coloring)
        except Exception as e:
            raise TypeError(
                f"Cannot attach coloring cache to object of type {type(A).__name__}. "
                f"Error: {e}"
            )

    if mpi is not None:
        try:
            object.__setattr__(A, "_mpi_cache", mpi)
        except Exception as e:
            raise TypeError(
                f"Cannot attach MPI cache to object of type {type(A).__name__}. "
                f"Error: {e}"
            )

    if is_symmetric:
        try:
            object.__setattr__(A, "_is_symmetric", True)
        except Exception as e:
            raise TypeError(
                f"Cannot attach symmetry info to object of type {type(A).__name__}. "
                f"Error: {e}"
            )

    return A


def cache_mpi_metadata(
    config: dict,
    comm: "Comm",
    nglobal: int,
    partition_info: tuple[int, int],
) -> dict[str, Any]:
    """
    Pre-compute and cache MPI metadata for JIT-compatible solver usage.

    This function performs all non-traceable MPI operations outside the JIT boundary:
    - Computes static MPI communication metadata (recvcounts, displs)
    - Prepares MPI communicator pointer and local rank
    - Prepares config string

    The cached metadata can be reused across multiple JIT-compiled function calls
    with different matrices or operators.

    Args:
        config: AmgX configuration dict or string
        comm: MPI communicator (from mpi4py.MPI.COMM_WORLD)
        nglobal: Global matrix size (total rows across all ranks)
        partition_info: tuple (row_start, row_end) indicating which rows this rank owns

    Returns:
        Dict containing MPI metadata:
        - 'recvcounts_tuple': Tuple of row counts per rank
        - 'displs_tuple': Tuple of displacement offsets
        - 'comm_ptr': MPI communicator pointer
        - 'lrank': Local GPU rank
        - 'nglobal': Global matrix size
        - 'config_str': Prepared configuration string
    """
    from mpi4py import MPI

    rank = comm.Get_rank()
    row_start, row_end = partition_info
    n_local = row_end - row_start

    # Compute MPI communication metadata
    all_sizes = comm.allgather(n_local)
    recvcounts = jnp.array(all_sizes, dtype=jnp.int32)
    displs = jnp.cumsum(jnp.concatenate([jnp.array([0]), recvcounts[:-1]])).astype(
        jnp.int32
    )

    # Get MPI communicator pointer and local rank
    comm_ptr = MPI._addressof(comm)
    gpu_count = jax.device_count()
    lrank = rank % gpu_count

    # Prepare config string
    config_str = amgx_config.prepare_config(config)

    return {
        "recvcounts_tuple": tuple(recvcounts.tolist()),
        "displs_tuple": tuple(displs.tolist()),
        "comm_ptr": comm_ptr,
        "lrank": lrank,
        "nglobal": nglobal,
        "config_str": config_str,
    }


def cache_coloring(
    operator: Any, shape: tuple[int, int] | int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, tuple[int, int]]:
    """
    Compute and cache coloring information for an operator.

    This function computes the sparsity pattern and graph coloring for a callable
    operator, enabling efficient use inside JIT-compiled functions.

    Args:
        operator: A callable operator A(x) that returns A @ x.
        shape: Shape of the operator (n, m) or int size (for n×n matrix).

    Returns:
        Cached coloring information that can be reattached with `with_cache(..., coloring=...)`.

    Example:
        >>> A = tridiagonal_operator(diagonal_value=2.0)
        >>> cache = cache_coloring(A, shape=100)
        >>>
        >>> @jax.jit
        >>> def solve(diag, b):
        >>>     A = with_cache(tridiagonal_operator(diag), coloring=cache)
        >>>     return amg_solve(A, b)
    """
    if isinstance(shape, int):
        shape = (shape, shape)

    # Check if already cached
    existing_cache = getattr(operator, "_coloring_info", None)
    if existing_cache is not None:
        # Verify size matches
        cached_shape = existing_cache[4]
        if cached_shape == shape:
            return existing_cache
        else:
            raise ValueError(
                f"Operator already has cached coloring for shape {cached_shape}, "
                f"but requested shape {shape}. Create a new operator instance."
            )

    # Compute sparsity pattern and coloring
    rows, cols = get_sparsity_pattern(operator, shape)
    column_colors, n_colors = get_column_coloring(rows, cols, shape)

    cache = (rows, cols, column_colors, n_colors, shape)

    # Try to attach to operator for convenience
    try:
        setattr(operator, "_coloring_info", cache)
    except Exception:
        pass  # Ignore if caching fails

    return cache
