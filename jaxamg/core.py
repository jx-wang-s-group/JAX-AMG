"""The implicit derivative of a native solve, for every derivative order.

``x = A⁻¹ b`` on a fixed pattern has tangent ``ẋ = L (ḃ − Ȧ x)``, where ``L`` is
the native solve from a zero start. It is applied through
``jax.lax.custom_linear_solve`` with the pattern SpMV as its operator and the
native transposed solve as its transpose, so JAX derives reverse mode and
higher orders; the native solve itself is never differentiated. The warm start
and ``info`` have zero derivatives.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.experimental.sparse as jsp
import jax.numpy as jnp
import numpy as np
from jax.custom_derivatives import SymbolicZero


def row_ids(indptr: jax.Array, nnz: int) -> jax.Array:
    """The row of every stored entry of a CSR matrix (traceable)."""
    n_rows = indptr.shape[0] - 1
    return jnp.repeat(
        jnp.arange(n_rows, dtype=jnp.int32),
        indptr[1:] - indptr[:-1],
        total_repeat_length=nnz,
    )


def spmv(
    values: jax.Array, indices: jax.Array, rows: jax.Array, n_rows: int, x: jax.Array
):
    """``A x`` for the CSR values on a fixed pattern; linear in ``values`` and ``x``."""
    return jax.ops.segment_sum(
        values * x[indices], rows, num_segments=n_rows, indices_are_sorted=True
    )


def _zero_tangent(value):
    """A zero tangent for one output leaf (``float0`` for integer leaves)."""
    if jnp.issubdtype(value.dtype, jnp.inexact):
        return jnp.zeros_like(value)
    return np.zeros(np.shape(value), dtype=jax.dtypes.float0)


def _instantiate(tangent, primal):
    """A symbolic-zero tangent as an array of zeros (else unchanged)."""
    if isinstance(tangent, SymbolicZero):
        return jnp.zeros_like(primal)
    return tangent


def csr_spmv(values, indices, indptr, aux, x):
    """Single-GPU pattern SpMV (``aux`` unused)."""
    del aux
    return spmv(
        values, indices, row_ids(indptr, values.shape[0]), indptr.shape[0] - 1, x
    )


def implicit_solver(
    native: Callable[..., tuple[jax.Array, jax.Array]],
    native_zero_start: Callable[..., jax.Array],
    native_transpose: Callable[..., jax.Array],
    is_symmetric: bool,
    layout_spmv: Callable[..., jax.Array] = csr_spmv,
    collective: bool = False,
) -> Callable[..., tuple[jax.Array, jax.Array]]:
    """Build the implicit-policy solve over a layout's native services.

    Args:
        native: ``(indptr, indices, values, b, x0, aux) -> (x, info)``, the
            primal solve with the caller's configuration (warm start, stats).
        native_zero_start: ``(indptr, indices, values, rhs, aux) -> x``, the
            same configuration from a zero start, for tangent solves.
        native_transpose: ``(indptr, indices, values, rhs, aux) -> λ`` solving
            ``Aᵀ λ = rhs`` from a zero start; with ``is_symmetric`` it is the
            forward solve.
        is_symmetric: The caller's declaration; selects the transpose service.
        layout_spmv: ``(values, indices, indptr, aux, x) -> A x`` on this
            layout's local rows (with its halo exchange under MPI), linear in
            ``values`` and ``x`` with transposable parts.
        collective: The services communicate (distributed layouts). A
            symbolic-zero tangent is then this rank's knowledge only, so the
            rule never skips the SpMV or the tangent solve on it: every rank
            whose solve is differentiated takes part in the same calls.

    ``aux`` is a pytree of the layout's non-differentiable plan operands
    (empty on a single GPU).
    """

    def core(values, indices, indptr, b, x0, aux):
        return native(indptr, indices, values, b, x0, aux)

    core = jax.custom_jvp(core)

    def core_jvp(primals, tangents):
        values, indices, indptr, b, x0, aux = primals
        dvalues, _, _, db, _, _ = tangents  # x0 and the plan have no derivative
        x, info = core(values, indices, indptr, b, x0, aux)
        zero_info = jax.tree_util.tree_map(_zero_tangent, info)
        if collective:
            dvalues = _instantiate(dvalues, values)
            db = _instantiate(db, b)
        dvalues_zero = isinstance(dvalues, SymbolicZero)
        db_zero = isinstance(db, SymbolicZero)
        if dvalues_zero and db_zero:
            return (x, info), (jnp.zeros_like(x), zero_info)
        if dvalues_zero:
            rhs = db
        else:
            rhs = -layout_spmv(dvalues, indices, indptr, aux, x)
            if not db_zero:
                rhs = db + rhs
        dx = jax.lax.custom_linear_solve(
            lambda v: layout_spmv(values, indices, indptr, aux, v),
            rhs,
            solve=lambda _, r: native_zero_start(indptr, indices, values, r, aux),
            transpose_solve=lambda _, r: native_transpose(
                indptr, indices, values, r, aux
            ),
            symmetric=is_symmetric,
        )
        return (x, info), (dx, zero_info)

    core.defjvp(core_jvp, symbolic_zeros=True)

    def solve(
        A: jsp.BCSR, b: jax.Array, x0: jax.Array, aux=()
    ) -> tuple[jax.Array, jax.Array]:
        return core(A.data, A.indices, A.indptr, b, x0, aux)

    return solve
