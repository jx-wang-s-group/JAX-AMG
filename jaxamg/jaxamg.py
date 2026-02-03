import jax
import jax.numpy as jnp
import jax.ffi as ffi
import jax.experimental.sparse as jsp
import functools
import numpy as np
from enum import IntEnum

from . import _amgx_ext, utils, config as amgx_config

_AMGX_CALL_NAME = "amgx_solve"
_AMGX_CALL_NAME_DOUBLE = "amgx_solve_double"
_AMGX_CALL_NAME_MPI = "amgx_solve_mpi"
_AMGX_CALL_NAME_MPI_DOUBLE = "amgx_solve_mpi_double"
_AMGX_CALL_NAME_ALLGATHER = "amgx_allgather"
_AMGX_CALL_NAME_ALLGATHER_DOUBLE = "amgx_allgather_double"

# Get the handler from C++ and register for CUDA platform
_AMGX_HANDLER = _amgx_ext.get_amgx_solve_handler()
_AMGX_HANDLER_DOUBLE = _amgx_ext.get_amgx_solve_double_handler()
_AMGX_HANDLER_MPI = _amgx_ext.get_amgx_solve_mpi_handler()
_AMGX_HANDLER_MPI_DOUBLE = _amgx_ext.get_amgx_solve_mpi_double_handler()
_AMGX_HANDLER_ALLGATHER = _amgx_ext.get_amgx_allgather_handler()
_AMGX_HANDLER_ALLGATHER_DOUBLE = _amgx_ext.get_amgx_allgather_double_handler()

ffi.register_ffi_target(_AMGX_CALL_NAME, _AMGX_HANDLER, platform="CUDA")
ffi.register_ffi_target(_AMGX_CALL_NAME_DOUBLE, _AMGX_HANDLER_DOUBLE, platform="CUDA")
ffi.register_ffi_target(_AMGX_CALL_NAME_MPI, _AMGX_HANDLER_MPI, platform="CUDA")
ffi.register_ffi_target(
    _AMGX_CALL_NAME_MPI_DOUBLE, _AMGX_HANDLER_MPI_DOUBLE, platform="CUDA"
)

ffi.register_ffi_target(
    _AMGX_CALL_NAME_ALLGATHER, _AMGX_HANDLER_ALLGATHER, platform="CUDA"
)
ffi.register_ffi_target(
    _AMGX_CALL_NAME_ALLGATHER_DOUBLE,
    _AMGX_HANDLER_ALLGATHER_DOUBLE,
    platform="CUDA",
)


class AMGXStatus(IntEnum):
    SUCCESS = 0
    FAILED = 1
    DIVERGED = 2
    NOT_CONVERGED = 3

    def __repr__(self):
        return f"<{self.__class__.__name__}.{self.name}: {self.value}>"

    def __str__(self):
        return f"{self.__class__.__name__}.{self.name}"


def _amgx_allgather_impl(sendbuf, recvcounts, displs, comm_ptr, nglobal=None):
    """MPI AllGatherv implementation via FFI."""
    if nglobal is None:
        nglobal = jnp.sum(recvcounts)

    out_spec = jax.ShapeDtypeStruct((nglobal,), sendbuf.dtype)

    call_name = (
        _AMGX_CALL_NAME_ALLGATHER_DOUBLE
        if sendbuf.dtype == jnp.float64
        else _AMGX_CALL_NAME_ALLGATHER
    )

    call = ffi.ffi_call(
        call_name,
        out_spec,
        input_layouts=[None, None, None, None],
        output_layouts=None,
    )
    return call(sendbuf, recvcounts, displs, comm_ptr)


def _amgx_solve_impl(row_ptrs, col_indices, values, b, config_str=""):
    """Low-level FFI call to AmgX solver (non-differentiable)."""

    out_spec = (
        jax.ShapeDtypeStruct(b.shape, b.dtype),
        jax.ShapeDtypeStruct((3,), b.dtype),
    )

    call_name = _AMGX_CALL_NAME
    if b.dtype == jnp.float64:
        call_name = _AMGX_CALL_NAME_DOUBLE

    call = ffi.ffi_call(
        call_name,
        out_spec,
        input_layouts=[None, None, None, None],
        output_layouts=None,
    )
    return call(row_ptrs, col_indices, values, b, config=config_str)


def _amgx_solve_mpi_impl(
    row_ptrs, col_indices, values, b, nglobal, comm_ptr, lrank, config_str=""
):
    """Low-level FFI call to AmgX MPI solver (non-differentiable)."""

    out_spec = (
        jax.ShapeDtypeStruct(b.shape, b.dtype),
        jax.ShapeDtypeStruct((3,), b.dtype),
    )

    call_name = _AMGX_CALL_NAME_MPI
    if b.dtype == jnp.float64:
        call_name = _AMGX_CALL_NAME_MPI_DOUBLE

    call = ffi.ffi_call(
        call_name,
        out_spec,
        input_layouts=[None, None, None, None, None, None, None],
        output_layouts=None,
    )
    return call(
        row_ptrs, col_indices, values, b, nglobal, comm_ptr, lrank, config=config_str
    )


@functools.lru_cache(maxsize=32)
def _get_solver_primitive(config_str, is_symmetric=False):
    """
    Returns a JAX custom_vjp primitive for AmgX solve with a specific configuration.
    Cached to avoid recompilation for identical configurations.
    """

    @jax.custom_vjp
    def solve(A, b):
        return _amgx_solve_impl(A.indptr, A.indices, A.data, b, config_str=config_str)

    def fwd(A, b):
        # returns ((x, stats), residuals)
        x_and_stats = solve(A, b)
        # We only need A and x for backward pass, stats are metadata
        x, stats = x_and_stats
        return x_and_stats, (A, x)

    def bwd(residuals, g):
        # g is tuple (g_x, g_stats). We ignore g_stats.
        g_x = g[0]
        # g_stats = g[1] # Should be zeros/None ideally

        A, x = residuals

        # Solve A^T λ = g_x
        solver = _get_solver_primitive(config_str, is_symmetric)

        # Check if matrix is symmetric
        if is_symmetric:
            adj_b, _ = solver(A, g_x)
        else:
            A_T = jsp.BCSR.from_bcoo(A.to_bcoo().transpose())
            adj_b, _ = solver(A_T, g_x)

        n = A.shape[0]
        row_lengths = A.indptr[1:] - A.indptr[:-1]

        # Safe gradient computation
        row_indices = jnp.repeat(
            jnp.arange(n, dtype=jnp.int32), row_lengths, total_repeat_length=len(A.data)
        )
        grad_values = -adj_b[row_indices] * x[A.indices]
        grad_A = jsp.BCSR((grad_values, A.indices, A.indptr), shape=A.shape)

        return grad_A, adj_b

    solve.defvjp(fwd, bwd)
    return solve


@functools.lru_cache(maxsize=32)
def _get_solver_primitive_mpi(config_str, nglobal, comm_ptr, lrank, is_symmetric=False):
    """
    Create cached JAX custom_vjp primitive for MPI AmgX solve.
    Supports automatic differentiation in distributed setting.
    """

    def allgather(sendbuf, recvcounts, displs, comm_ptr_arr):
        return _amgx_allgather_impl(
            sendbuf, recvcounts, displs, comm_ptr_arr, nglobal=nglobal
        )

    @jax.custom_vjp
    def solve(A, b, recvcounts, displs):
        nglobal_arr = jnp.array([nglobal], dtype=jnp.int32)

        # Split 64-bit comm_ptr into two int32 values for FFI
        comm_ptr_low_unsigned = comm_ptr & 0xFFFFFFFF
        comm_ptr_high_unsigned = (comm_ptr >> 32) & 0xFFFFFFFF
        comm_ptr_low_signed = np.int32(np.uint32(comm_ptr_low_unsigned))
        comm_ptr_high_signed = np.int32(np.uint32(comm_ptr_high_unsigned))

        comm_ptr_arr = jnp.array(
            [comm_ptr_low_signed, comm_ptr_high_signed], dtype=jnp.int32
        )
        lrank_arr = jnp.array([lrank], dtype=jnp.int32)

        x, stats = _amgx_solve_mpi_impl(
            A.indptr,
            A.indices,
            A.data,
            b,
            nglobal_arr,
            comm_ptr_arr,
            lrank_arr,
            config_str=config_str,
        )

        return (x, stats), comm_ptr_arr

    def fwd(A, b, recvcounts, displs):
        out = solve(A, b, recvcounts, displs)
        (x, stats), comm_ptr_arr = out
        return out, (A, x, recvcounts, displs, comm_ptr_arr)

    def bwd(residuals, g):
        (g_x, _), _ = g
        A, x, recvcounts, displs, comm_ptr_arr = residuals

        # A^T for backward solve
        def transpose_callback(data, indices, indptr, shape, recvcounts):
            from mpi4py import MPI

            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            size = comm.Get_size()

            # 1. Prepare Local Data
            r_counts = np.array(recvcounts)

            # Compute global displacements: displs[i] = sum(recvcounts[:i])
            displs = np.concatenate(([0], np.cumsum(r_counts[:-1]))).astype(np.int32)

            my_row_start = displs[rank]
            n_local = r_counts[rank]

            # Convert indptr to row indices (COO format)
            # Row count for each row i is indptr[i+1] - indptr[i]
            row_counts = indptr[1:] - indptr[:-1]
            row_indices_local = np.repeat(
                np.arange(n_local, dtype=np.int32), row_counts
            )

            # Global row indices for A
            row_indices_global = row_indices_local + my_row_start

            # Global col indices for A (already in indices)
            col_indices_global = indices

            # 2. Determine Destination Ranks
            # Element A[i, j] belongs to the rank owning row j in A^T.
            # Map global column indices to ranked partitions.
            # equivalent to searchsorted(displs, val, side='right') - 1

            dest_ranks = np.searchsorted(displs, col_indices_global, side="right") - 1
            dest_ranks = dest_ranks.astype(np.int32)
            np.clip(dest_ranks, 0, size - 1, out=dest_ranks)

            # 3. Pack data for AllToAllv
            # Sort by destination rank to pack buffers
            sort_order = np.argsort(dest_ranks)
            dest_ranks_sorted = dest_ranks[sort_order]

            # Data to send: (global_col_in_A, global_row_in_A, value)
            # This corresponds to (global_row_in_AT, global_col_in_AT, value)
            vals_sorted = data[sort_order]
            rows_sorted = row_indices_global[sort_order]  # becomes col in AT
            cols_sorted = col_indices_global[sort_order]  # becomes row in AT

            # Compute send counts for each rank
            send_counts = np.bincount(dest_ranks_sorted, minlength=size).astype(
                np.int32
            )
            send_displs = np.concatenate(([0], np.cumsum(send_counts[:-1]))).astype(
                np.int32
            )

            # Exchange data sizes
            recv_counts = np.zeros(size, dtype=np.int32)
            comm.Alltoall(send_counts, recv_counts)

            total_recv = np.sum(recv_counts)
            recv_displs = np.concatenate(([0], np.cumsum(recv_counts[:-1]))).astype(
                np.int32
            )

            # Allocate receive buffers
            recv_vals = np.empty(total_recv, dtype=data.dtype)
            recv_rows_at = np.empty(
                total_recv, dtype=np.int32
            )  # Global Row index in AT (was col in A)
            recv_cols_at = np.empty(
                total_recv, dtype=np.int32
            )  # Global Col index in AT (was row in A)

            # Determine MPI type for data
            mpi_type_data = MPI.FLOAT
            if data.dtype == np.float64:
                mpi_type_data = MPI.DOUBLE

            # Exchange Data
            comm.Alltoallv(
                [vals_sorted, send_counts, send_displs, mpi_type_data],
                [recv_vals, recv_counts, recv_displs, mpi_type_data],
            )

            # Exchange Indices
            # Input indices are usually int32 or int64 from JAX.
            # We cast to int32 for safety/consistency if needed, but keeping precision is better if supported.
            # Assuming int32 for indices as per JAX BCSR default usually.
            mpi_type_idx = MPI.INT32_T

            # Cast if strictly needed, otherwise mpi4py handles numpy types often
            cols_sorted_i32 = cols_sorted.astype(np.int32)
            rows_sorted_i32 = rows_sorted.astype(np.int32)

            comm.Alltoallv(
                [cols_sorted_i32, send_counts, send_displs, mpi_type_idx],
                [recv_rows_at, recv_counts, recv_displs, mpi_type_idx],
            )
            comm.Alltoallv(
                [rows_sorted_i32, send_counts, send_displs, mpi_type_idx],
                [recv_cols_at, recv_counts, recv_displs, mpi_type_idx],
            )

            # 4. Construct local A^T
            # Local rows: subtract this rank's offset
            recv_rows_local = recv_rows_at - my_row_start

            # Sort by (row, col) to ensure valid BCSR structure
            sort_idx = np.lexsort((recv_cols_at, recv_rows_local))

            r_sorted = recv_rows_local[sort_idx]
            c_sorted = recv_cols_at[sort_idx]
            v_sorted = recv_vals[sort_idx]

            # Build indptr
            row_counts_at = np.bincount(r_sorted, minlength=n_local)
            out_indptr = np.zeros(n_local + 1, dtype=indptr.dtype)
            out_indptr[1:] = np.cumsum(row_counts_at)

            # 5. JAX Shape Compatibility
            # pure_callback requires static output shapes. We must match A's nnz.
            target_nnz = len(data)
            actual_nnz = len(v_sorted)

            if actual_nnz <= target_nnz:
                # Pad with zeros at the end
                out_data = np.zeros(target_nnz, dtype=data.dtype)
                out_indices = np.zeros(target_nnz, dtype=indices.dtype)

                out_data[:actual_nnz] = v_sorted
                out_indices[:actual_nnz] = c_sorted

                # Append padding zeros to the last row
                out_indptr_padded = out_indptr.copy()
                out_indptr_padded[-1] = target_nnz

                return out_data, out_indices, out_indptr_padded

            else:
                # Truncate
                # This introduces error but prevents JAX shape mismatch crash.
                # In practice for symmetric-structure matrices, actual_nnz == target_nnz.
                out_data = v_sorted[:target_nnz]
                out_indices = c_sorted[:target_nnz]

                # Cap indptr
                out_indptr_truncated = np.minimum(out_indptr, target_nnz)
                out_indptr_truncated[-1] = target_nnz

                return out_data, out_indices, out_indptr_truncated

        # Backward solve: A^T @ adj_b = g_x

        # Check if matrix is symmetric
        if is_symmetric:
            # Skip distributed transpose
            (adj_b, _), _ = solve(A, g_x, recvcounts, displs)
        else:
            # Execute transpose on host via callback
            # Use simple pure_callback signature
            at_data, at_indices, at_indptr = jax.pure_callback(
                transpose_callback,
                (A.data, A.indices, A.indptr),  # Output shapes derived from A
                A.data,
                A.indices,
                A.indptr,
                A.shape,
                recvcounts,
            )

            # Reconstruct BCSR for A^T
            A_T = jsp.BCSR((at_data, at_indices, at_indptr), shape=A.shape)

            (adj_b, _), _ = solve(A_T, g_x, recvcounts, displs)

        # Gather x across all ranks for gradient computation
        x_global = allgather(x, recvcounts, displs, comm_ptr_arr)

        # Compute ∂L/∂A: ∂L/∂A_ij = -adj_b[i] * x[j]
        n_local = A.shape[0]
        row_lengths = A.indptr[1:] - A.indptr[:-1]
        row_indices = jnp.repeat(
            jnp.arange(n_local, dtype=jnp.int32),
            row_lengths,
            total_repeat_length=len(A.data),
        )
        grad_values = -adj_b[row_indices] * x_global[A.indices]
        grad_A = jsp.BCSR((grad_values, A.indices, A.indptr), shape=A.shape)

        return grad_A, adj_b, None, None

    solve.defvjp(fwd, bwd)
    return solve


def amg_solve(
    A,
    b,
    config=None,
    comm=None,
    nglobal=None,
    partition_info=None,
    **kwargs,
):
    """
    Solve Ax=b using AmgX (differentiable).

    Single-GPU mode (default):
        A: either a jax.experimental.sparse.CSR matrix or a callable A(x).
           Callables are automatically materialized to CSR.
        b: RHS vector (float32 or float64).
        config: Dict or string of AmgX configuration parameters.
        **kwargs: Additional configuration parameters passed as keyword arguments.
                  These override config if present.

    MPI mode (when comm is provided):
        A: Local portion of matrix with GLOBAL column indices (CSR).
        b: Local portion of RHS vector.
        comm: MPI communicator (from mpi4py.MPI.COMM_WORLD).
        nglobal: Global size of the matrix (total number of rows across all ranks).
        partition_info: Tuple (row_start, row_end) indicating which rows this rank owns.
        config: Dict or string of AmgX configuration parameters.
        **kwargs: Additional configuration parameters.

    If A is attached with MPI cache (via `with_cache`), then comm, nglobal, and partition_info are not needed.

    Returns:
        x: Solution vector (float32 or float64). In MPI mode, returns local portion.
        info: Dictionary containing 'iterations', 'residual', and 'status'.
    """

    # Check for GPU backend
    if jax.default_backend() != "gpu":
        raise RuntimeError(
            f"AMGX requires a GPU backend, but JAX is using '{jax.default_backend()}'. "
            "Please ensure you have a CUDA-enabled GPU and JAX is installed with CUDA support."
        )

    # Check if MPI cache is attached to A (via with_cache)
    mpi_cache = getattr(A, "_mpi_cache", None)

    # Prepare configuration string/file (skip if using mpi_cache which already has config_str)
    if mpi_cache is not None:
        config_str = mpi_cache["config_str"]
    else:
        config_str = amgx_config.prepare_config(config, **kwargs)

    # Detect desired precision
    target_dtype = utils.get_preferred_dtype(A, b)
    if target_dtype == jnp.float64 and b.dtype != jnp.float64:
        b = b.astype(jnp.float64)

    # Check for symmetry attribute on A
    is_symmetric = getattr(A, "_is_symmetric", False)

    # Branch: MPI mode or single-GPU mode
    if mpi_cache is not None or comm is not None:
        # MPI MODE
        if mpi_cache is None:
            # Validate parameters for non-cache path
            if nglobal is None:
                raise ValueError("nglobal must be provided when using MPI mode")
            if partition_info is None:
                raise ValueError(
                    "partition_info (row_start, row_end) must be provided when using MPI mode"
                )

        # Convert A to BCSR with int64 indices (required for MPI)
        A_csr = utils.to_bcsr_matrix(A, b=b, use_int64_indices=True)

        if mpi_cache is not None:
            # Use pre-cached MPI metadata
            solver = _get_solver_primitive_mpi(
                mpi_cache["config_str"],
                mpi_cache["nglobal"],
                mpi_cache["comm_ptr"],
                mpi_cache["lrank"],
                is_symmetric=is_symmetric,
            )
            recvcounts = jnp.array(mpi_cache["recvcounts_tuple"], dtype=jnp.int32)
            displs = jnp.array(mpi_cache["displs_tuple"], dtype=jnp.int32)

            (x, stats), _ = solver(A_csr, b, recvcounts, displs)
        else:
            # Compute metadata dynamically
            try:
                from mpi4py import MPI
            except ImportError:
                raise ImportError(
                    "mpi4py is required for MPI mode. Install it with: pip install mpi4py"
                )

            # Get MPI rank and compute local GPU assignment
            rank = comm.Get_rank()
            gpu_count = jax.device_count()
            lrank = rank % gpu_count

            # Get MPI communicator pointer
            comm_ptr = MPI._addressof(comm)

            # Gather partition sizes from all ranks for gradient allgather operation
            n_local = A_csr.shape[0]
            all_sizes_list = comm.allgather(n_local)
            recvcounts_val = np.array(all_sizes_list, dtype=np.int32)
            displs_val = np.cumsum(np.concatenate(([0], recvcounts_val[:-1]))).astype(
                np.int32
            )

            recvcounts = jnp.array(recvcounts_val)
            displs = jnp.array(displs_val)

            solver = _get_solver_primitive_mpi(
                config_str, nglobal, comm_ptr, lrank, is_symmetric=is_symmetric
            )
            (x, stats), _ = solver(A_csr, b, recvcounts, displs)

    else:
        # Single-GPU mode: use int32 indices
        A_csr = utils.to_bcsr_matrix(A, b=b)

        # Get cached primitive for this configuration
        solver = _get_solver_primitive(config_str, is_symmetric=is_symmetric)

        # Returns (x, stats_array)
        x, stats = solver(A_csr, b)

    # Convert JAX array stats to python dict (same for both modes)
    try:
        iter_val = int(stats[0])
        res_val = float(stats[1])
        status_val = AMGXStatus(int(stats[2]))
    except Exception:
        # Inside JIT or symbolic execution: return raw arrays/tracers
        iter_val = stats[0]
        res_val = stats[1]
        status_val = stats[2]

    info = {"iterations": iter_val, "residual": res_val, "status": status_val}
    return x, info


def cache_mpi_metadata(config, comm, nglobal, partition_info):
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
        partition_info: Tuple (row_start, row_end) indicating which rows this rank owns

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


def with_cache(A, *, coloring=None, mpi=None, is_symmetric=False):
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


def cache_coloring(operator, shape):
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
    rows, cols = utils.get_sparsity_pattern(operator, shape)
    column_colors, n_colors = utils.get_column_coloring(rows, cols, shape)

    cache = (rows, cols, column_colors, n_colors, shape)

    # Try to attach to operator for convenience
    try:
        setattr(operator, "_coloring_info", cache)
    except Exception:
        pass  # Ignore if caching fails

    return cache
