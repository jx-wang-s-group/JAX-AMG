import jax
import jax.numpy as jnp
import jax.ffi as ffi

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
def amgx_solve(row_ptrs, col_indices, values, b):
    """Solve Ax=b using AmgX (differentiable).

    Args:
        row_ptrs: CSR row pointers (int32)
        col_indices: CSR column indices (int32)
        values: CSR values (float32)
        b: Right-hand side vector (float32)

    Returns:
        x: Solution vector (float32)
    """
    return _amgx_solve_impl(row_ptrs, col_indices, values, b)


def _amgx_fwd(row_ptrs, col_indices, values, b):
    x = _amgx_solve_impl(row_ptrs, col_indices, values, b)
    return x, (row_ptrs, col_indices, values)


def _amgx_bwd(res, g):
    """Backward pass: solve A^T λ = g for gradient.

    For linear solve x = A^{-1} b, we have:
        ∂L/∂b = A^{-T} (∂L/∂x)

    Since A is symmetric for our problems, A^T = A, so we solve:
        A λ = g  where g = ∂L/∂x
    """
    row_ptrs, col_indices, values = res
    # Solve A^T λ = g (for symmetric A, this is A λ = g)
    adj_b = _amgx_solve_impl(row_ptrs, col_indices, values, g)
    # Gradients w.r.t. matrix structure are zero (not differentiating w.r.t. A)
    zeros_r = jnp.zeros_like(row_ptrs)
    zeros_c = jnp.zeros_like(col_indices)
    zeros_v = jnp.zeros_like(values)
    return zeros_r, zeros_c, zeros_v, adj_b


amgx_solve.defvjp(_amgx_fwd, _amgx_bwd)
