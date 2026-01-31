import jax
import jax.numpy as jnp
import jax.ffi as ffi
import jax.experimental.sparse as jsp
from jax import core

from . import _amgx_ext, utils


_AMGX_CALL_NAME = "amgx_solve"

# Get the handler from C++ and register for CUDA platform
_AMGX_HANDLER = _amgx_ext.get_amgx_solve_handler()
ffi.register_ffi_target(_AMGX_CALL_NAME, _AMGX_HANDLER, platform="CUDA")


def _amgx_solve_impl(row_ptrs, col_indices, values, b):
    """Low-level FFI call to AmgX solver (non-differentiable)."""
    out_spec = jax.ShapeDtypeStruct(b.shape, b.dtype)
    call = ffi.ffi_call(
        _AMGX_CALL_NAME,
        out_spec,
        input_layouts=[None, None, None, None],
        output_layouts=None,
    )
    return call(row_ptrs, col_indices, values, b)


def _normalize_linear_operator(A, *, b=None):
    """
    Normalize input 'A' to a CSR matrix.

    If 'A' is callable:
    1. Outside JIT: Computes sparsity pattern and graph coloring (O(N)), caches it on 'A', and materializes.
    2. Inside JIT: Uses cached coloring to materialize efficiently (O(Colors)).
    """
    if isinstance(A, jsp.CSR):
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
        A_csr = utils.materialize_sparse_matrix(
            A, shape, rows, cols, column_colors, n_colors
        )

        if A_csr.data.dtype != jnp.float32:
            raise ValueError(
                f"Callable A must return dtype float32. Got {A_csr.data.dtype}."
            )

        return A_csr

    raise TypeError(
        f"Matrix A must be either a jax.experimental.sparse.CSR matrix or a callable. "
        f"Got {type(A).__name__}."
    )


@jax.custom_vjp
def _amgx_solve_csr(A, b):
    """
    Solve Ax=b using AmgX with custom VJP for gradients.
    Input A must be a jsp.CSR matrix.
    """
    return _amgx_solve_impl(A.indptr, A.indices, A.data, b)


def _amgx_fwd(A, b):
    x = _amgx_solve_impl(A.indptr, A.indices, A.data, b)
    return x, (A, x)


def _amgx_bwd(residuals, g):
    """
    Backward pass using Adjoint State Method.
    Solves A^T λ = g to find adjoint λ, then computes gradients.
    """
    A, x = residuals

    # 1. Solve for Adjoint λ = A^{-T} g
    # Since A is assumed symmetric for AmgX usually (or we just use transpose solve),
    # we solve A^T λ = g.
    adj_b = _amgx_solve_impl(A.indptr, A.indices, A.data, g)

    # 2. Compute gradients w.r.t. matrix values
    # ∂L/∂A_ij = -λ_i * x_j
    # Efficiently gather -λ[row[k]] * x[col[k]] for all non-zero entries k.

    n = A.shape[0]
    row_lengths = A.indptr[1:] - A.indptr[:-1]
    row_indices = jnp.repeat(
        jnp.arange(n, dtype=jnp.int32), row_lengths, total_repeat_length=len(A.data)
    )

    grad_values = -adj_b[row_indices] * x[A.indices]

    grad_A = jsp.CSR((grad_values, A.indices, A.indptr), shape=A.shape)

    return grad_A, adj_b


_amgx_solve_csr.defvjp(_amgx_fwd, _amgx_bwd)


def amg_solve(A, b):
    """
    Solve Ax=b using AmgX (differentiable).

    Args:
        A: either a jax.experimental.sparse.CSR matrix or a callable A(x).
           Callables are automatically materialized to CSR.
        b: RHS vector (float32).

    Returns:
        x: Solution vector (float32).

    """
    A_csr = _normalize_linear_operator(A, b=b)
    return _amgx_solve_csr(A_csr, b)


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
