"""
Caching utilities.

This module provides functions to cache metadata, enabling efficient usage with JAX JIT compilation.
"""

from typing import TYPE_CHECKING, Any

import jax
import numpy as np

from . import config as amgx_config
from .utils import *

if TYPE_CHECKING:
    from mpi4py.MPI import Comm

    from .mpi_utils import HaloPlan, TransposePlan


def _build_mpi_cache(
    config: dict,
    comm: "Comm",
    nglobal: int,
    recvcounts_tuple: tuple[int, ...],
    max_nnz: int,
    nnz_out: int | None,
    halo_plan: "HaloPlan",
    *,
    transpose_plan: "TransposePlan | None" = None,
    row_indices: np.ndarray | jax.Array | None = None,
    save_stats: bool = False,
    block_dim: int = 1,
    lrank: int | None = None,
) -> dict[str, Any]:
    """Assemble MPI metadata after structure-dependent collectives are done."""
    from .mpi_utils import register_comm

    rank = comm.Get_rank()
    comm_ptr = register_comm(comm)
    if lrank is None:
        lrank = rank % jax.device_count()

    # These plans are solve operands, not setup metadata. Keep one device copy
    # in the cache so repeated eager solves do not re-transfer every routing
    # array from the host. The original NumPy buffers can then be released.
    local_devices = jax.local_devices()
    device = local_devices[lrank % len(local_devices)]
    halo_plan = halo_plan._replace(
        col_to_combined=jax.device_put(halo_plan.col_to_combined, device),
        send_ids_2d=jax.device_put(halo_plan.send_ids_2d, device),
        recv_ghost_slot_2d=jax.device_put(halo_plan.recv_ghost_slot_2d, device),
    )
    if transpose_plan is not None:
        with temp_enable_x64():
            transpose_plan = transpose_plan._replace(
                indices=jax.device_put(transpose_plan.indices, device),
                indptr=jax.device_put(transpose_plan.indptr, device),
                local_source_ids=jax.device_put(
                    transpose_plan.local_source_ids, device
                ),
                local_target_ids=jax.device_put(
                    transpose_plan.local_target_ids, device
                ),
                send_ids_2d=jax.device_put(transpose_plan.send_ids_2d, device),
                recv_target_ids_2d=jax.device_put(
                    transpose_plan.recv_target_ids_2d, device
                ),
            )
    if row_indices is not None:
        row_indices = jax.device_put(row_indices, device)

    config_str = amgx_config.prepare_config(
        config, save_stats=save_stats, mpi=True, block_dim=block_dim
    )
    return {
        "recvcounts_tuple": recvcounts_tuple,
        "comm_ptr": comm_ptr,
        "lrank": lrank,
        "nglobal": nglobal,
        "config_str": config_str,
        "max_nnz": max_nnz,
        "nnz_out": nnz_out,
        "halo_plan": halo_plan,
        "transpose_plan": transpose_plan,
        "row_indices": row_indices,
        "block_dim": block_dim,
    }


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
    block_dim: int = 1,
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
        block_dim: BSR block size for AmgX (see `jaxamg.solve`). Each rank's
            local partition must be divisible by it.

    Returns:
        A dictionary containing MPI metadata.

    Note:
        The returned dictionary includes the following keys:

        - `recvcounts_tuple`: Tuple of row counts per rank
        - `comm_ptr`: MPI communicator pointer
        - `lrank`: Local GPU rank
        - `nglobal`: Global matrix size
        - `config_str`: Prepared configuration string
        - `max_nnz`: Maximum nnz across all ranks
        - `nnz_out`: This rank's local nnz(A^T) for the transpose output, or
          `None` when `is_symmetric` is True
        - `halo_plan`: Backward-pass halo-exchange plan for the gradient w.r.t.
          A (fetches only referenced remote solution entries)
        - `transpose_plan`: Fixed transpose structure and value routing for a
          nonsymmetric matrix, or `None` when `is_symmetric` is True
        - `row_indices`: Local CSR row index for every matrix nonzero
    """
    row_start, row_end = partition_info
    n_local = row_end - row_start

    block_dim = int(block_dim)
    if block_dim < 1:
        raise ValueError("block_dim must be a positive integer")
    if block_dim > 1 and (n_local % block_dim != 0 or nglobal % block_dim != 0):
        raise ValueError(
            f"local partition ({n_local} rows) and nglobal ({nglobal}) must "
            f"be divisible by block_dim {block_dim}"
        )

    # Compute MPI communication metadata
    all_sizes = comm.allgather(n_local)

    from .mpi_utils import build_halo_plan, build_transpose_plan

    # Compute max_nnz across all ranks, and capture this rank's global column
    # indices (needed for nnz_out, the transpose output sizing).
    # For sparse matrices (BCSR), get nnz/indices from the arrays directly.
    if all(hasattr(A, field) for field in ("data", "indices", "indptr")):
        local_nnz = len(A.data)
        local_col_indices = np.asarray(A.indices)
        local_indptr = np.asarray(A.indptr)
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
        local_indptr = np.asarray(A_materialized.indptr)
    else:
        A_materialized = to_bcsr_matrix(
            A,
            b=jnp.empty(n_local, dtype=get_preferred_dtype(A, None)),
            use_int64_indices=True,
        )
        local_nnz = len(A_materialized.data)
        local_col_indices = np.asarray(A_materialized.indices)
        local_indptr = np.asarray(A_materialized.indptr)

    all_nnz = comm.allgather(local_nnz)
    max_nnz = max(all_nnz)

    recvcounts_tuple = tuple(int(s) for s in all_sizes)

    # This rank's local nnz(A^T) for the transpose output buffers (backward pass
    # of non-symmetric solves). Symmetric solves never transpose, so skip it.
    if is_symmetric:
        nnz_out = None
        transpose_plan = None
    else:
        transpose_plan = build_transpose_plan(
            local_col_indices,
            local_indptr,
            recvcounts_tuple,
            partition_info,
            comm,
        )
        nnz_out = transpose_plan.nnz

    # Backward-pass halo plan: fetches only the remote solution entries this
    # rank's rows reference for the gradient w.r.t. A, instead of gathering the
    # full global solution. Needed for both symmetric and non-symmetric matrices.
    halo_plan = build_halo_plan(
        local_col_indices, recvcounts_tuple, partition_info, comm
    )
    row_indices = np.repeat(
        np.arange(n_local, dtype=np.int32), np.diff(local_indptr)
    ).astype(np.int32)

    return _build_mpi_cache(
        config,
        comm,
        nglobal,
        recvcounts_tuple,
        max_nnz,
        nnz_out,
        halo_plan,
        transpose_plan=transpose_plan,
        row_indices=row_indices,
        save_stats=save_stats,
        block_dim=block_dim,
    )
