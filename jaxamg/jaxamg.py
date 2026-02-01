import jax
import jax.numpy as jnp
import jax.ffi as ffi
import jax.experimental.sparse as jsp
from jax import core
import functools
from enum import IntEnum
import scipy.sparse as sp
import numpy as np
import os

from . import _amgx_ext, utils, config as amgx_config


_AMGX_CALL_NAME = "amgx_solve"

# Get the handler from C++ and register for CUDA platform
_AMGX_HANDLER = _amgx_ext.get_amgx_solve_handler()
ffi.register_ffi_target(_AMGX_CALL_NAME, _AMGX_HANDLER, platform="CUDA")


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
        jax.ShapeDtypeStruct((3,), jnp.float32),
    )
    call = ffi.ffi_call(
        _AMGX_CALL_NAME,
        out_spec,
        input_layouts=[None, None, None, None],
        output_layouts=None,
    )
    return call(row_ptrs, col_indices, values, b, config=config_str)


def _normalize_linear_operator(A, *, b=None):
    """
    Normalize input 'A' to a BCSR matrix.

    Supports multiple input formats, including BCSR, BCOO, SciPy sparse (CSR, CSC, COO), dense arrays (NumPy, JAX), and callable operators.

    If 'A' is callable:
    1. Outside JIT: Computes sparsity pattern and graph coloring (O(N)), caches it on 'A', and materializes.
    2. Inside JIT: Uses cached coloring to materialize efficiently (O(Colors)).
    """
    # BCSR: validate and return
    if isinstance(A, jsp.BCSR):
        if A.data.dtype != jnp.float32:
            raise ValueError(f"Matrix values must be float32, got {A.data.dtype}.")
        if A.indices.dtype != jnp.int32:
            raise ValueError(
                f"Matrix column indices must be int32, got {A.indices.dtype}."
            )
        if A.indptr.dtype != jnp.int32:
            raise ValueError(
                f"Matrix row pointers must be int32, got {A.indptr.dtype}."
            )
        return A

    # BCOO: convert to BCSR
    if isinstance(A, jsp.BCOO):
        A_bcsr = jsp.BCSR.from_bcoo(A)
        if A_bcsr.data.dtype != jnp.float32:
            A_bcsr = jsp.BCSR(
                (A_bcsr.data.astype(jnp.float32), A_bcsr.indices, A_bcsr.indptr),
                shape=A_bcsr.shape,
            )
        if A_bcsr.indices.dtype != jnp.int32:
            A_bcsr = jsp.BCSR(
                (
                    A_bcsr.data,
                    A_bcsr.indices.astype(jnp.int32),
                    A_bcsr.indptr.astype(jnp.int32),
                ),
                shape=A_bcsr.shape,
            )
        return A_bcsr

    # SciPy sparse: convert to BCSR
    if sp.issparse(A):
        A_bcsr = jsp.BCSR.from_scipy_sparse(A.astype(np.float32))

        if A_bcsr.indices.dtype != jnp.int32:
            A_bcsr = jsp.BCSR(
                (
                    A_bcsr.data,
                    A_bcsr.indices.astype(jnp.int32),
                    A_bcsr.indptr.astype(jnp.int32),
                ),
                shape=A_bcsr.shape,
            )
        return A_bcsr

    # Dense arrays: convert to BCSR
    if isinstance(A, (np.ndarray, jnp.ndarray)):
        if A.ndim != 2:
            raise ValueError(f"Dense matrix must be 2D, got shape {A.shape}")
        if isinstance(A, np.ndarray):
            A = jnp.array(A, dtype=jnp.float32)
        elif A.dtype != jnp.float32:
            A = A.astype(jnp.float32)

        A_bcsr = jsp.BCSR.fromdense(A)

        if A_bcsr.indices.dtype != jnp.int32:
            A_bcsr = jsp.BCSR(
                (
                    A_bcsr.data,
                    A_bcsr.indices.astype(jnp.int32),
                    A_bcsr.indptr.astype(jnp.int32),
                ),
                shape=A_bcsr.shape,
            )
        return A_bcsr

    # Callable: materialize using graph coloring
    if callable(A):
        # Check for cached coloring info attached to the callable
        cached_info = getattr(A, "_amgx_coloring_info", None)

        if b is None:
            raise TypeError("Callable A requires RHS b to infer size.")
        if b.ndim != 1:
            raise ValueError(f"RHS b must be 1D, got shape {b.shape}.")
        shape = (int(b.shape[0]), int(b.shape[0]))

        is_jit = isinstance(b, core.Tracer)

        if cached_info is None:
            if is_jit:
                # Inside JIT without cache: Impossible to determine sparsity dynamically.
                raise ValueError(
                    "Callable operators must be pre-scanned before JIT compilation to determine sparsity.\n"
                    "Call amg_solve(A, b) once outside of JIT to compute and cache the sparsity pattern."
                )

            # Outside JIT: Compute sparsity and coloring (expensive O(N))
            rows, cols = utils.get_sparsity_pattern(A, shape)
            column_colors, n_colors = utils.get_column_coloring(rows, cols, shape)

            cached_info = (rows, cols, column_colors, n_colors, shape)
            try:
                setattr(A, "_amgx_coloring_info", cached_info)
            except Exception:
                pass  # Ignore if caching fails (e.g. on partials or immutable objects)

        rows, cols, column_colors, n_colors, cached_shape = cached_info

        if shape != cached_shape:
            raise ValueError(
                f"Operator shape changed from {cached_shape} to {shape}. Create a new callable."
            )

        # Materialize using graph coloring (works efficienty inside JIT)
        A_bcsr = utils.materialize_sparse_matrix(
            A, shape, rows, cols, column_colors, n_colors
        )

        if A_bcsr.data.dtype != jnp.float32:
            raise ValueError(
                f"Callable A must return dtype float32. Got {A_bcsr.data.dtype}."
            )

        return A_bcsr

    raise TypeError(
        f"Matrix A must be one of: BCSR, BCOO, SciPy sparse, dense array (NumPy/JAX), or callable. "
        f"Got {type(A).__name__}."
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


def amg_solve(A, b, config=None, **kwargs):
    """
    Solve Ax=b using AmgX (differentiable).

    Args:
        A: either a jax.experimental.sparse.CSR matrix or a callable A(x).
           Callables are automatically materialized to CSR.
        b: RHS vector (float32).
        config: Dict or string of AmgX configuration parameters.
        **kwargs: Additional configuration parameters passed as keyword arguments.
                  These override config if present.

    Returns:
        x: Solution vector (float32).
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

    A_csr = _normalize_linear_operator(A, b=b)

    # Get cached primitive for this configuration
    solver = _get_solver_primitive(config_str)

    # Returns (x, stats_array)
    x, stats = solver(A_csr, b)

    # Convert JAX array stats to python dict
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
