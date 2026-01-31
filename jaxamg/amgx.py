import jax
import jax.numpy as jnp
import jax.ffi as ffi
import jax.experimental.sparse as jsp

from . import _amgx_ext


_AMGX_CALL_NAME = "amgx_solve"

# Get the handler from C++
_AMGX_HANDLER = _amgx_ext.get_amgx_solve_handler()

# Register FFI target for CUDA platform
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


@jax.custom_vjp
def amgx_solve(A, b):
    """Solve Ax=b using AmgX (differentiable).

    Args:
        A: CSR matrix (jax.experimental.sparse.CSR)
        b: Right-hand side vector (float32)

    Returns:
        x: Solution vector (float32)

    Raises:
        TypeError: If A is not a jax.experimental.sparse.CSR matrix
        ValueError: If A has incorrect dtypes (indices must be int32, values must be float32)
    """
    # Validate input type
    if not isinstance(A, jsp.CSR):
        raise TypeError(
            f"Matrix A must be a jax.experimental.sparse.CSR object, got {type(A).__name__}. "
        )

    # Validate dtypes
    if A.data.dtype != jnp.float32:
        raise ValueError(
            f"Matrix values must be float32, got {A.data.dtype}. "
        )
    if A.indices.dtype != jnp.int32:
        raise ValueError(
            f"Matrix column indices must be int32, got {A.indices.dtype}."
        )
    if A.indptr.dtype != jnp.int32:
        raise ValueError(
            f"Matrix row pointers must be int32, got {A.indptr.dtype}."
        )

    # Extract CSR components
    row_ptrs = A.indptr
    col_indices = A.indices
    values = A.data

    return _amgx_solve_impl(row_ptrs, col_indices, values, b)


def _amgx_fwd(A, b):
    # Extract CSR components
    row_ptrs = A.indptr
    col_indices = A.indices
    values = A.data

    x = _amgx_solve_impl(row_ptrs, col_indices, values, b)
    # Save the CSR matrix for backward pass
    return x, A


def _amgx_bwd(A, g):
    """Backward pass: solve A^T λ = g for gradient.

    For linear solve x = A^{-1} b, we have:
        ∂L/∂b = A^{-T} (∂L/∂x)

    Since A is symmetric for our problems, A^T = A, so we solve:
        A λ = g  where g = ∂L/∂x
    """
    # Extract CSR components
    row_ptrs = A.indptr
    col_indices = A.indices
    values = A.data

    # Solve A^T λ = g (for symmetric A, this is A λ = g)
    adj_b = _amgx_solve_impl(row_ptrs, col_indices, values, g)

    # Gradients w.r.t. matrix structure are zero (not differentiating w.r.t. A)
    zeros_A = jax.tree.map(jnp.zeros_like, A)

    return zeros_A, adj_b


amgx_solve.defvjp(_amgx_fwd, _amgx_bwd)
