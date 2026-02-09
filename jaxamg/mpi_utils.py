"""MPI utilities for distributed AmgX solving."""

from typing import TYPE_CHECKING, cast

import jax.experimental.sparse as jsp
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
from jax.typing import ArrayLike

if TYPE_CHECKING:
    from mpi4py.MPI import Comm


def partition_csr_matrix(
    A_global: jsp.BCSR | sp.csr_matrix, rank: int, nranks: int
) -> tuple[jsp.BCSR, int, int]:
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
    row_start, row_end, n_local = get_partition_info(n, rank, nranks)

    # Extract local partition
    nnz_start = indptr[row_start]
    nnz_end = indptr[row_end]

    # Create BCSR: convert to JAX if SciPy, or ensure int32 indices if already JAX
    local_indptr = jnp.asarray(indptr[row_start : row_end + 1] - nnz_start)
    local_indices = jnp.asarray(indices[nnz_start:nnz_end])
    local_data = jnp.asarray(data[nnz_start:nnz_end])

    if not is_scipy:
        local_indices = local_indices.astype(jnp.int32)
        local_indptr = local_indptr.astype(jnp.int32)

    A_local = jsp.BCSR((local_data, local_indices, local_indptr), shape=(n_local, n))
    return A_local, row_start, row_end


def validate_partition(
    A_local: jsp.BCSR, nglobal: int, row_start: int, row_end: int
) -> None:
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


def partition_vector(
    b_global: ArrayLike, rank: int, nranks: int
) -> tuple[ArrayLike, int, int]:
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
    b_global = jnp.asarray(b_global)
    n = len(b_global)
    row_start, row_end, _ = get_partition_info(n, rank, nranks)
    return b_global[row_start:row_end], row_start, row_end


def gather_solution(
    x_local: ArrayLike, comm: "Comm", root: int = 0
) -> ArrayLike | None:
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
    all_sizes = cast(list, all_sizes)

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


def get_partition_info(n_global: int, rank: int, nranks: int) -> tuple[int, int, int]:
    """Compute partition information for distributed problem."""
    local_size = n_global // nranks
    remainder = n_global % nranks

    if rank < remainder:
        n_local = local_size + 1
        row_start = rank * n_local
    else:
        n_local = local_size
        row_start = rank * local_size + remainder

    row_end = row_start + n_local

    return row_start, row_end, n_local
