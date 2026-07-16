"""
Caching utilities.

This module provides functions to cache metadata, enabling efficient usage with JAX JIT compilation.
"""

from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np

from . import config as amgx_config
from .utils import *

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
    without recomputing metadata or passing it as separate arguments. See [Caching Guide](caching.md) for more details.

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
    A: MatrixOrOperator,
    is_symmetric: bool = False,
    save_stats: bool = False,
) -> dict[str, Any]:
    """
    Pre-compute and cache MPI metadata for JIT-compatible solver usage.

    The cached metadata can be reused across multiple JIT-compiled function calls
    with different matrices or operators (same structure).

    Note:
        This function performs all non-traceable MPI operations outside the JIT boundary:

        - Computes static MPI communication metadata (recvcounts, displs)
        - Prepares MPI communicator pointer and local rank
        - Prepares config string
        - Computes max nnz across all ranks


    Args:
        config: AmgX configuration dict or string
        comm: MPI communicator (from mpi4py.MPI.COMM_WORLD)
        nglobal: Global matrix size (total rows across all ranks)
        partition_info: tuple (row_start, row_end) indicating which rows this rank owns
        A: Matrix or operator to compute max nnz for buffer sizing
        is_symmetric: If True, the backward pass never transposes, so the
            transpose output size (`nnz_out`) is left unset (``None``). Should
            match the `is_symmetric` passed to `with_cache`; the default (False)
            computes it, which is always safe.
        save_stats: If True, prepare the config with solver statistics output
            enabled, so a later `solve(..., save_stats_file=...)` on the cached
            matrix produces a complete stats file.

    Returns:
        A dictionary containing MPI metadata.

    Note:
        The returned dictionary includes the following keys:

        - `recvcounts_tuple`: Tuple of row counts per rank
        - `displs_tuple`: Tuple of displacement offsets
        - `comm_ptr`: MPI communicator pointer
        - `lrank`: Local GPU rank
        - `nglobal`: Global matrix size
        - `config_str`: Prepared configuration string
        - `max_nnz`: Maximum nnz across all ranks
        - `nnz_out`: This rank's local nnz(A^T) for the transpose output, or
          `None` when `is_symmetric` is True
        - `halo_plan`: Backward-pass halo-exchange plan for the gradient w.r.t.
          A (fetches only referenced remote solution entries)
    """
    rank = comm.Get_rank()
    row_start, row_end = partition_info
    n_local = row_end - row_start

    # Compute MPI communication metadata
    all_sizes = comm.allgather(n_local)
    recvcounts = jnp.array(all_sizes, dtype=jnp.int32)
    displs = jnp.cumsum(jnp.concatenate([jnp.array([0]), recvcounts[:-1]])).astype(
        jnp.int32
    )

    from .mpi_utils import build_halo_plan, local_transpose_nnz, register_comm

    # Register the communicator so the cached solver's backward pass can recover
    # it for its collectives; comm_ptr is its address.
    comm_ptr = register_comm(comm)
    gpu_count = jax.device_count()
    lrank = rank % gpu_count

    # Prepare config string
    config_str = amgx_config.prepare_config(config, save_stats=save_stats, mpi=True)

    # Compute max_nnz across all ranks, and capture this rank's global column
    # indices (needed for nnz_out, the transpose output sizing).
    # For sparse matrices (BCSR), get nnz/indices from the arrays directly.
    if hasattr(A, "data"):
        local_nnz = len(A.data)
        local_col_indices = np.asarray(A.indices)
    elif callable(A):
        # For distributed operators, we need to use global size for proper materialization
        # The operator shape is (n_local, n_global): takes global vector, returns local portion
        from .sparsity import cache_coloring, materialize_sparse_matrix

        # Check if operator already has cached coloring
        cached_info = getattr(A, "_coloring_info", None)

        if cached_info is None:
            # Detect + colour via cache_coloring (tracing, then one-hot probing as
            # fallback) using the distributed (n_local, n_global) shape.
            cached_info = cache_coloring(A, (n_local, nglobal))

        rows, cols, column_colors, n_colors, shape = cached_info
        A_materialized = materialize_sparse_matrix(
            A, shape, rows, cols, column_colors, n_colors
        )

        local_nnz = len(A_materialized.data)
        local_col_indices = np.asarray(A_materialized.indices)
    else:
        raise TypeError(
            f"Matrix A must be BCSR, BCOO, SciPy sparse, dense array, or callable. "
            f"Got {type(A).__name__}."
        )

    all_nnz = comm.allgather(local_nnz)
    max_nnz = max(all_nnz)

    recvcounts_tuple = tuple(int(s) for s in all_sizes)

    # This rank's local nnz(A^T) for the transpose output buffers (backward pass
    # of non-symmetric solves). Symmetric solves never transpose, so skip it.
    if is_symmetric:
        nnz_out = None
    else:
        nnz_out = local_transpose_nnz(local_col_indices, recvcounts_tuple, comm)

    # Backward-pass halo plan: fetches only the remote solution entries this
    # rank's rows reference for the gradient w.r.t. A, instead of gathering the
    # full global solution. Needed for both symmetric and non-symmetric matrices.
    halo_plan = build_halo_plan(
        local_col_indices, recvcounts_tuple, partition_info, comm
    )

    cache_dict = {
        "recvcounts_tuple": tuple(recvcounts.tolist()),
        "displs_tuple": tuple(displs.tolist()),
        "comm_ptr": comm_ptr,
        "lrank": lrank,
        "nglobal": nglobal,
        "config_str": config_str,
        "max_nnz": max_nnz,
        "nnz_out": nnz_out,
        "halo_plan": halo_plan,
    }

    return cache_dict
