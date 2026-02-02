"""MPI utilities for distributed AMGX solving."""

import jax.numpy as jnp
import jax.experimental.sparse as jsp
import numpy as np
from typing import Tuple
from dataclasses import dataclass


@dataclass
class DistributedCSR:
    """Distributed CSR matrix with global column indexing."""

    data: jnp.ndarray
    indices: jnp.ndarray
    indptr: jnp.ndarray
    shape: Tuple[int, int]
    row_start: int
    row_end: int
    nglobal: int


def partition_csr_matrix(A_global, rank, nranks):
    """Partition global CSR matrix across MPI ranks (row-based).

    Args:
        A_global: Global CSR matrix (SciPy sparse or JAX BCSR)
        rank: MPI rank (0-indexed)
        nranks: Total number of MPI ranks

    Returns:
        A_local: Local BCSR matrix partition (JAX)
        row_start: Starting row index (global)
        row_end: Ending row index (global, exclusive)

    Note:
        Preserves input dtype (float32/float64). Avoids unnecessary conversions
        by using matrix attributes directly.
    """
    is_scipy = hasattr(A_global, "todense")

    if hasattr(A_global, "indptr"):
        indptr, indices, data = A_global.indptr, A_global.indices, A_global.data
        n = A_global.shape[0]
    else:
        raise ValueError(f"Unsupported matrix type: {type(A_global)}")

    # Row-based partitioning
    rows_per_rank = n // nranks
    row_start = rank * rows_per_rank
    row_end = (rank + 1) * rows_per_rank if rank < nranks - 1 else n
    n_local = row_end - row_start

    # Extract local partition
    nnz_start = indptr[row_start]
    nnz_end = indptr[row_end]
    local_indptr = indptr[row_start : row_end + 1] - nnz_start
    local_indices = indices[nnz_start:nnz_end]
    local_data = data[nnz_start:nnz_end]

    # Create BCSR: convert to JAX if SciPy, or ensure int32 indices if already JAX
    if is_scipy:
        local_data = jnp.array(local_data)
        local_indices = jnp.array(local_indices)
        local_indptr = jnp.array(local_indptr)
    else:
        local_indices = local_indices.astype(jnp.int32)
        local_indptr = local_indptr.astype(jnp.int32)

    A_local = jsp.BCSR((local_data, local_indices, local_indptr), shape=(n_local, n))
    return A_local, row_start, row_end


def validate_partition(A_local, nglobal, row_start, row_end):
    """Validate partitioned matrix structure and print diagnostics."""
    n_local = row_end - row_start

    assert (
        A_local.shape[0] == n_local
    ), f"Row count mismatch: {A_local.shape[0]} != {n_local}"
    assert (
        A_local.shape[1] == nglobal
    ), f"Column count mismatch: {A_local.shape[1]} != {nglobal}"
    assert (
        A_local.indptr[0] == 0
    ), f"First row pointer should be 0, got {A_local.indptr[0]}"
    assert A_local.indptr[-1] == len(
        A_local.data
    ), f"Last row pointer mismatch: {A_local.indptr[-1]} != {len(A_local.data)}"

    if len(A_local.indices) > 0:
        max_col = jnp.max(A_local.indices)
        min_col = jnp.min(A_local.indices)
        assert (
            max_col < nglobal
        ), f"Column index {max_col} exceeds global size {nglobal}"
        assert min_col >= 0, f"Column index {min_col} is negative"
        print(f"✓ Partition validated: {n_local} rows, cols [{min_col}, {max_col}]")
    else:
        print(f"✓ Partition validated: {n_local} rows, no non-zeros")


def partition_vector(b_global, rank, nranks):
    """Partition global vector across MPI ranks (row-based).

    Args:
        b_global: Global vector
        rank: MPI rank (0-indexed)
        nranks: Total number of MPI ranks

    Returns:
        b_local: Local vector partition
        row_start: Starting row index (global)
        row_end: Ending row index (global, exclusive)
    """
    n = len(b_global)
    rows_per_rank = n // nranks
    row_start = rank * rows_per_rank
    row_end = (rank + 1) * rows_per_rank if rank < nranks - 1 else n
    return b_global[row_start:row_end], row_start, row_end


def compute_halo_size(A_local, row_start, row_end):
    """Compute number of unique halo elements (off-rank column indices).

    Args:
        A_local: Local matrix partition
        row_start: Starting row index (global)
        row_end: Ending row index (global, exclusive)

    Returns:
        Number of unique column indices outside [row_start, row_end)
    """
    col_indices = np.array(A_local.indices)
    halo_cols = col_indices[(col_indices < row_start) | (col_indices >= row_end)]
    return len(np.unique(halo_cols))


def gather_solution(x_local, comm, root=0):
    """Gather distributed solution to root rank using MPI Gatherv.

    Args:
        x_local: Local solution vector
        comm: MPI communicator
        root: Root rank to gather to (default: 0)

    Returns:
        JAX array of global solution (root rank only), None otherwise
    """
    from mpi4py import MPI

    rank = comm.Get_rank()
    x_local_np = np.array(x_local, dtype=np.float64)
    n_local = len(x_local_np)
    all_sizes = comm.gather(n_local, root=root)

    if rank == root:
        n_global = sum(all_sizes)
        x_global = np.zeros(n_global, dtype=x_local_np.dtype)
        displacements = [0] + list(np.cumsum(all_sizes[:-1]))
        mpi_type = MPI.DOUBLE if x_local_np.dtype == np.float64 else MPI.FLOAT
        comm.Gatherv(
            x_local_np, [x_global, all_sizes, displacements, mpi_type], root=root
        )
        return jnp.array(x_global)
    else:
        comm.Gatherv(x_local_np, None, root=root)
        return None
