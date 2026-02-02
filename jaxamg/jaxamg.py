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

# Get the handler from C++ and register for CUDA platform
_AMGX_HANDLER = _amgx_ext.get_amgx_solve_handler()
_AMGX_HANDLER_DOUBLE = _amgx_ext.get_amgx_solve_double_handler()
_AMGX_HANDLER_MPI = _amgx_ext.get_amgx_solve_mpi_handler()
_AMGX_HANDLER_MPI_DOUBLE = _amgx_ext.get_amgx_solve_mpi_double_handler()

ffi.register_ffi_target(_AMGX_CALL_NAME, _AMGX_HANDLER, platform="CUDA")
ffi.register_ffi_target(_AMGX_CALL_NAME_DOUBLE, _AMGX_HANDLER_DOUBLE, platform="CUDA")
ffi.register_ffi_target(_AMGX_CALL_NAME_MPI, _AMGX_HANDLER_MPI, platform="CUDA")
ffi.register_ffi_target(
    _AMGX_CALL_NAME_MPI_DOUBLE, _AMGX_HANDLER_MPI_DOUBLE, platform="CUDA"
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
def _get_solver_primitive(config_str):
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
        solver = _get_solver_primitive(config_str)
        # Solver returns (adj_b, adj_stats). We only care about adj_b.
        adj_b, _ = solver(A, g_x)

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
def _get_solver_primitive_mpi(config_str, nglobal, comm_ptr, lrank):
    """
    Returns a JAX custom_vjp primitive for MPI AmgX solve with a specific configuration.
    Cached to avoid recompilation for identical configurations.

    Note: Gradient support for MPI mode is not yet implemented.
    """

    @jax.custom_vjp
    def solve(A, b):
        # Convert parameters to JAX arrays on device
        nglobal_arr = jnp.array([nglobal], dtype=jnp.int32)

        # Handle comm_ptr: split into two int32 values
        # Use numpy to handle unsigned properly, then convert to signed int32 for JAX
        comm_ptr_low_unsigned = comm_ptr & 0xFFFFFFFF
        comm_ptr_high_unsigned = (comm_ptr >> 32) & 0xFFFFFFFF

        # Convert unsigned to signed int32 (reinterpret bits)
        comm_ptr_low_signed = np.int32(np.uint32(comm_ptr_low_unsigned))
        comm_ptr_high_signed = np.int32(np.uint32(comm_ptr_high_unsigned))

        comm_ptr_arr = jnp.array(
            [comm_ptr_low_signed, comm_ptr_high_signed], dtype=jnp.int32
        )
        lrank_arr = jnp.array([lrank], dtype=jnp.int32)

        return _amgx_solve_mpi_impl(
            A.indptr,
            A.indices,
            A.data,
            b,
            nglobal_arr,
            comm_ptr_arr,
            lrank_arr,
            config_str=config_str,
        )

    def fwd(A, b):
        x_and_stats = solve(A, b)
        x, stats = x_and_stats
        return x_and_stats, (A, x)

    def bwd(residuals, g):
        # Gradient computation for MPI mode not yet implemented
        raise NotImplementedError(
            "Automatic differentiation is not yet supported for MPI mode. "
            "Please use forward mode only or implement custom gradients."
        )

    solve.defvjp(fwd, bwd)
    return solve


def amg_solve(
    A, b, config=None, comm=None, nglobal=None, partition_info=None, **kwargs
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

    # Prepare configuration string/file
    config_str = amgx_config.prepare_config(config, **kwargs)

    # Detect desired precision
    target_dtype = utils.get_preferred_dtype(A, b)
    if target_dtype == jnp.float64 and b.dtype != jnp.float64:
        b = b.astype(jnp.float64)

    # Branch: MPI mode or single-GPU mode
    if comm is not None:
        # MPI MODE
        if nglobal is None:
            raise ValueError("nglobal must be provided when using MPI mode")
        if partition_info is None:
            raise ValueError(
                "partition_info (row_start, row_end) must be provided when using MPI mode"
            )

        # Import mpi4py for communicator handling
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

        # A must already be CSR format with global column indices in MPI mode
        # Use int64 indices for MPI (AMGX global API requires int64)
        A_csr = utils.to_bcsr_matrix(A, b=b, use_int64_indices=True)

        # Get cached MPI primitive for this configuration
        solver = _get_solver_primitive_mpi(config_str, nglobal, comm_ptr, lrank)

        # Solve
        x, stats = solver(A_csr, b)

    else:
        # SINGLE-GPU MODE (original behavior)
        # Use int32 indices for single-GPU (default behavior)
        A_csr = utils.to_bcsr_matrix(A, b=b)

        # Get cached primitive for this configuration
        solver = _get_solver_primitive(config_str)

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


def cache_coloring(operator, size):
    """
    Compute and cache coloring information for an operator.

    This function computes the sparsity pattern and graph coloring for a callable
    operator, enabling efficient use inside JIT-compiled functions.

    Args:
        operator: A callable operator A(x) that returns A @ x.
        size: Size of the operator (n for an n×n matrix).

    Returns:
        Cached coloring information that can be reattached with `with_coloring()`.

    Example:
        >>> A = tridiagonal_operator(diagonal_value=2.0)
        >>> cache = cache_coloring(A, size=100)
        >>>
        >>> @jax.jit
        >>> def solve(diag, b):
        >>>     A = with_coloring(tridiagonal_operator(diag), cache)
        >>>     return amg_solve(A, b)
    """
    # Check if already cached
    existing_cache = getattr(operator, "_amgx_coloring_info", None)
    if existing_cache is not None:
        # Verify size matches
        cached_shape = existing_cache[4]
        if cached_shape == (size, size):
            return existing_cache
        else:
            raise ValueError(
                f"Operator already has cached coloring for size {cached_shape[0]}, "
                f"but requested size {size}. Create a new operator instance."
            )

    # Compute sparsity pattern and coloring
    shape = (size, size)
    rows, cols = utils.get_sparsity_pattern(operator, shape)
    column_colors, n_colors = utils.get_column_coloring(rows, cols, shape)

    cache = (rows, cols, column_colors, n_colors, shape)

    # Try to attach to operator for convenience
    try:
        setattr(operator, "_amgx_coloring_info", cache)
    except Exception:
        pass  # Ignore if caching fails

    return cache


def with_coloring(operator, cache):
    """
    Attach cached coloring information to an operator.

    This allows using parameterized operators inside JIT-compiled functions
    without recomputing the sparsity pattern and coloring.

    Args:
        operator: A callable operator.
        cache: Cached coloring information from `cache_coloring()`.

    Returns:
        The same operator with coloring cache attached.
    """
    try:
        object.__setattr__(operator, "_amgx_coloring_info", cache)
    except Exception as e:
        raise TypeError(
            f"Cannot attach coloring cache to operator of type {type(operator).__name__}. "
            f"Error: {e}"
        )
    return operator
