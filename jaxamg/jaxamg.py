import functools
import json
import os
import warnings
from collections.abc import Callable
from enum import IntEnum
from typing import TYPE_CHECKING, Any, cast

import jax
import jax.experimental.sparse as jsp
import jax.ffi as ffi
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from . import config as amgx_config
from .mpi_utils import (
    TransposePlan,
    build_halo_plan,
    build_transpose_plan,
    register_comm,
    resolve_comm,
    transpose_values,
)
from .nullspace import (
    _DENSE_LU_MSG,
    _MISSING_NULLSPACE_MSG,
    _MISSING_TRANSPOSE_MSG,
    LabelSpec,
    NullSpaceSpec,
    NullSpaceWarning,
    as_labels,
    as_nullspace_basis,
    make_mpi_reduce_sum,
    project_out,
    relative_norm,
    row_index,
    validate_basis,
    verify_nullspace,
    warn_if_singular,
)
from .transport import halo_gather, mpi_ordered_effect
from .utils import *

if TYPE_CHECKING:
    from mpi4py.MPI import Comm

_AMGX_CALL_NAME = "amgx_solve"
_AMGX_CALL_NAME_DOUBLE = "amgx_solve_double"
_AMGX_CALL_NAME_MPI = "amgx_solve_mpi"
_AMGX_CALL_NAME_MPI_DOUBLE = "amgx_solve_mpi_double"

# The native extension is loaded lazily on first use, so importing jaxamg
# (for sparsity tracing, matrix helpers, config handling, or on a CPU-only
# machine such as a CI runner) does not require the AmgX/CUDA libraries.
_amgx: Any = None

# Whether the native extension was compiled with MPI support (JAXAMG_WITH_MPI).
# A non-MPI build omits the MPI FFI handlers entirely. Set by _ensure_backend().
HAS_MPI = False


def _ensure_backend() -> Any:
    """Load the native AmgX extension and register its FFI targets (once)."""
    global _amgx, HAS_MPI
    if _amgx is not None:
        return _amgx
    from ._ext import _amgx as ext

    HAS_MPI = bool(getattr(ext, "mpi_enabled", False))
    ffi.register_ffi_target(
        _AMGX_CALL_NAME, ext.get_amgx_solve_handler(), platform="CUDA"
    )
    ffi.register_ffi_target(
        _AMGX_CALL_NAME_DOUBLE, ext.get_amgx_solve_double_handler(), platform="CUDA"
    )
    if HAS_MPI:
        ffi.register_ffi_target(
            _AMGX_CALL_NAME_MPI, ext.get_amgx_solve_mpi_handler(), platform="CUDA"
        )
        ffi.register_ffi_target(
            _AMGX_CALL_NAME_MPI_DOUBLE,
            ext.get_amgx_solve_mpi_double_handler(),
            platform="CUDA",
        )
    _amgx = ext
    return ext


class AMGXStatus(IntEnum):
    """High-level AmgX solve status codes returned in `info["status"]` after calling `jaxamg.solve`.

    These values are mapped from the native backend status for quick checks in
    Python code and in docs.

    Members:
        - `SUCCESS`: Solve converged successfully.
        - `FAILED`: Solver failed due to an internal/runtime error.
        - `DIVERGED`: Iterations diverged.
        - `NOT_CONVERGED`: Reached stopping criteria without convergence.
    """

    SUCCESS = 0
    FAILED = 1
    DIVERGED = 2
    NOT_CONVERGED = 3

    def __repr__(self):
        return f"<{self.__class__.__name__}.{self.name}: {self.value}>"

    def __str__(self):
        return f"{self.__class__.__name__}.{self.name}"


def _amgx_solve_impl(
    row_ptrs: ArrayLike,
    col_indices: ArrayLike,
    values: ArrayLike,
    b: ArrayLike,
    x0: ArrayLike | None = None,
    config_str: str = "",
    transpose_solve: bool = False,
    return_stats: bool = False,
    reuse_setup: bool = False,
    res_history_len: int = 0,
    use_x0: bool = False,
    block_dim: int = 1,
) -> tuple[jax.Array, jax.Array]:
    """Low-level FFI call to AmgX solver (non-differentiable)."""

    _ensure_backend()
    b = jnp.asarray(b)

    out_spec = (
        jax.ShapeDtypeStruct(b.shape, b.dtype),
        jax.ShapeDtypeStruct((3 + res_history_len,), b.dtype),
    )

    call_name = _AMGX_CALL_NAME
    if b.dtype == jnp.float64:
        call_name = _AMGX_CALL_NAME_DOUBLE

    # AmgX mutates cached solver resources, so this call is not functionally pure.
    call = ffi.ffi_call(
        call_name,
        out_spec,
        has_side_effect=True,
        input_layouts=[None, None, None, None, None],
        output_layouts=None,
        vmap_method="sequential",
    )
    # The x0 slot is a required input; pass b as a same-shape dummy when
    # unused (ignored by the C++ side when use_x0 is 0).
    results = call(
        row_ptrs,
        col_indices,
        values,
        b,
        x0 if x0 is not None else b,
        config=config_str,
        transpose_solve=np.int32(transpose_solve),
        return_stats=np.int32(return_stats),
        reuse_setup=np.int32(reuse_setup),
        use_x0=np.int32(use_x0),
        block_dim=np.int32(block_dim),
    )

    return cast(tuple, results)


def _amgx_mpi_abstract_eval(*args, res_history_len, ordered, **params):
    b = args[3]
    outs = (
        b.update(weak_type=False),
        b.update(shape=(3 + res_history_len,), weak_type=False),
    )
    return outs, ({mpi_ordered_effect()} if ordered else set())


def _amgx_mpi_lowering(ctx, *operands, res_history_len, ordered, config, **params):
    # The private builder (as mpi4jax uses): a typed-FFI call with a token.
    from jax._src.interpreters.mlir import custom_call as _custom_call
    from jax._src.lib.mlir.dialects import hlo
    from jax.interpreters import mlir

    b_aval = ctx.avals_in[3]
    target = (
        _AMGX_CALL_NAME_MPI_DOUBLE
        if b_aval.dtype == jnp.float64
        else _AMGX_CALL_NAME_MPI
    )
    effect = mpi_ordered_effect()
    token = ctx.tokens_in.get(effect) if ordered else hlo.create_token()
    attrs = {
        "config": mlir.ir_attribute(config),
        "device_mpi": mlir.ir_attribute(
            np.int32(
                json.loads(config or "{}").get("communicator", "MPI") == "MPI_DIRECT"
            )
        ),
    }
    attrs.update({k: mlir.ir_attribute(np.int32(v)) for k, v in params.items()})
    call = _custom_call(
        target,
        result_types=[mlir.aval_to_ir_type(a) for a in ctx.avals_out]
        + [hlo.TokenType.get()],
        operands=[*operands, token],
        backend_config=attrs,
        api_version=4,
        has_side_effect=True,
        operand_layouts=[tuple(range(len(a.shape) - 1, -1, -1)) for a in ctx.avals_in]
        + [()],
        result_layouts=[tuple(range(len(a.shape) - 1, -1, -1)) for a in ctx.avals_out]
        + [()],
    )
    *outs, token_out = call.results
    if ordered:
        ctx.set_tokens_out(mlir.TokenSet({effect: token_out}))
    return outs


def _amgx_mpi_batch(args, dims, **params):
    """One system at a time (communication has no batched form)."""
    from jax.interpreters import batching

    moved = [
        a if d is batching.not_mapped else jnp.moveaxis(a, d, 0)
        for a, d in zip(args, dims)
    ]
    batched = [d is not batching.not_mapped for d in dims]

    def one(sliced):
        it = iter(sliced)
        full = [next(it) if bt else a for a, bt in zip(moved, batched)]
        return tuple(_amgx_mpi_p.bind(*full, **params))

    outs = jax.lax.map(one, tuple(a for a, bt in zip(moved, batched) if bt))
    return list(outs), [0, 0]


def _make_amgx_mpi_primitive():
    from jax._src import dispatch
    from jax.extend import core
    from jax.interpreters import batching, mlir

    prim = core.Primitive("jaxamg_amgx_solve_mpi")
    prim.multiple_results = True
    prim.def_impl(lambda *a, **p: dispatch.apply_primitive(prim, *a, **p))
    prim.def_effectful_abstract_eval(_amgx_mpi_abstract_eval)
    mlir.register_lowering(prim, _amgx_mpi_lowering, platform="cuda")
    batching.primitive_batchers[prim] = _amgx_mpi_batch
    return prim


_amgx_mpi_p = _make_amgx_mpi_primitive()


def _amgx_solve_mpi_impl(
    row_ptrs: ArrayLike,
    col_indices: ArrayLike,
    values: ArrayLike,
    b: ArrayLike,
    x0: ArrayLike | None,
    nglobal: ArrayLike,
    comm_ptr: ArrayLike,
    lrank: ArrayLike,
    config_str: str = "",
    transpose_solve: bool = False,
    return_stats: bool = False,
    reuse_setup: bool = False,
    res_history_len: int = 0,
    use_x0: bool = False,
    block_dim: int = 1,
    ordered: bool = True,
    local_sizes: ArrayLike | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Low-level FFI call to AmgX MPI solver (non-differentiable).

    ``ordered`` (per-rank programs) joins mpi4jax's ordered effect, so every
    rank issues its AmgX solves, neighbour exchanges and mpi4jax collectives
    in one program order under every transformation: independent solves in one
    program can otherwise be scheduled differently on ranks whose programs
    differ (local sizes), matching one solve's collectives with another's or
    deadlocking. SPMD programs (``shard_map``) pass ``ordered=False``.

    ``local_sizes`` (int32 ``(n_local, nnz)``, a runtime operand) marks padded
    buffers: only their prefixes are solved and the solution's tail is zero,
    so ranks with different local sizes can share one program. The default
    ``(-1, -1)`` uses the buffers' sizes."""

    _ensure_backend()
    b = jnp.asarray(b)
    # The x0 slot is a required input; pass b as a same-shape dummy when
    # unused (ignored by the C++ side when use_x0 is 0).
    x, stats = _amgx_mpi_p.bind(
        jnp.asarray(row_ptrs),
        jnp.asarray(col_indices),
        jnp.asarray(values),
        b,
        b if x0 is None else jnp.asarray(x0),
        jnp.asarray(nglobal),
        jnp.asarray(comm_ptr),
        jnp.asarray(lrank),
        (
            jnp.array([-1, -1], dtype=jnp.int32)
            if local_sizes is None
            else jnp.asarray(local_sizes, dtype=jnp.int32)
        ),
        config=config_str,
        transpose_solve=int(transpose_solve),
        return_stats=int(return_stats),
        reuse_setup=int(reuse_setup),
        res_history_len=int(res_history_len),
        use_x0=int(use_x0),
        block_dim=int(block_dim),
        ordered=bool(ordered),
    )
    return x, stats


@functools.lru_cache(maxsize=32)
def _get_adjoint_primitive(
    config_str: str,
    is_symmetric: bool = False,
    return_stats: bool = False,
    reuse_setup: bool = False,
    res_history_len: int = 0,
    use_x0: bool = False,
    block_dim: int = 1,
) -> Callable:
    """
    The adjoint-rule solve (a ``custom_vjp``, reverse mode only) for one native
    configuration, cached per configuration.

    reuse_setup: Skip warm AMGX resetup and keep the cached hierarchy.
    res_history_len: Residual-history slots appended to the stats output.
    use_x0: Honor the x0 operand as the initial guess (otherwise it is a
        same-shape dummy and the solve starts from zero).
    block_dim: BSR block size for AmgX (the JAX-side matrix stays scalar CSR).
    """

    @jax.custom_vjp
    def solve(A: jsp.BCSR, b: jax.Array, x0: jax.Array) -> tuple[jax.Array, jax.Array]:
        x, info = _amgx_solve_impl(
            A.indptr,
            A.indices,
            A.data,
            b,
            x0,
            config_str=config_str,
            return_stats=return_stats,
            reuse_setup=reuse_setup,
            res_history_len=res_history_len,
            use_x0=use_x0,
            block_dim=block_dim,
        )
        return x, info

    def fwd(
        A: jsp.BCSR, b: jax.Array, x0: jax.Array
    ) -> tuple[tuple[jax.Array, jax.Array], tuple[jsp.BCSR, jax.Array]]:
        x, info = solve(A, b, x0)
        # Returns ((x, info), residuals)
        return (x, info), (A, x)

    def bwd(
        residuals: tuple[jsp.BCSR, jax.Array], g: tuple[jax.Array, jax.Array]
    ) -> tuple[jsp.BCSR, jax.Array, jax.Array]:
        g_x = g[0]
        A, x = residuals

        # Solve A^T λ = g_x (always from a zero start: x0 shifts only the
        # forward iteration, never the solution, so the adjoint ignores it).
        solver = _get_adjoint_primitive(
            config_str,
            is_symmetric,
            return_stats=False,
            reuse_setup=reuse_setup,
            block_dim=block_dim,
        )

        # Check if matrix is symmetric
        if is_symmetric:
            adj_b, _ = solver(A, g_x, g_x)
        else:
            # The native transposed solve. Where it cannot be used (for
            # example under a higher reverse derivative of this rule: the
            # opaque native call has no JVP), solve an explicit Aᵀ with this
            # custom-VJP rule instead.
            try:
                adj_b, _ = _amgx_solve_impl(
                    A.indptr,
                    A.indices,
                    A.data,
                    g_x,
                    config_str=config_str,
                    transpose_solve=True,
                    reuse_setup=reuse_setup,
                    block_dim=block_dim,
                )
            except Exception:
                A_T = jsp.BCSR.from_bcoo(A.to_bcoo().transpose())
                adj_b, _ = solver(A_T, g_x, g_x)

        n = A.shape[0]
        row_lengths = A.indptr[1:] - A.indptr[:-1]

        # Safe gradient computation
        row_indices = jnp.repeat(
            jnp.arange(n, dtype=jnp.int32), row_lengths, total_repeat_length=len(A.data)
        )
        grad_values = -adj_b[row_indices] * x[A.indices]
        grad_A = jsp.BCSR((grad_values, A.indices, A.indptr), shape=A.shape)

        # The exact solution does not depend on the initial guess.
        return grad_A, adj_b, jnp.zeros_like(adj_b)

    solve.defvjp(fwd, bwd)
    return solve


def _single_gpu_services(
    config_str: str,
    is_symmetric: bool = False,
    return_stats: bool = False,
    reuse_setup: bool = False,
    res_history_len: int = 0,
    use_x0: bool = False,
    block_dim: int = 1,
) -> tuple[Callable, Callable, Callable]:
    """The single-GPU native services of the implicit core: the primal solve,
    the zero-start solve and the declared transposed solve. Tangent and adjoint
    solves use the adjoint rule's native calls (same configuration, zero
    start, no statistics), so first reverse results match the adjoint rule's."""

    def native(indptr, indices, values, b, x0, aux):
        return _amgx_solve_impl(
            indptr,
            indices,
            values,
            b,
            x0,
            config_str=config_str,
            return_stats=return_stats,
            reuse_setup=reuse_setup,
            res_history_len=res_history_len,
            use_x0=use_x0,
            block_dim=block_dim,
        )

    def native_zero_start(indptr, indices, values, rhs, aux=()):
        return _amgx_solve_impl(
            indptr,
            indices,
            values,
            rhs,
            None,
            config_str=config_str,
            reuse_setup=reuse_setup,
            block_dim=block_dim,
        )[0]

    def native_transpose(indptr, indices, values, rhs, aux=()):
        if is_symmetric:
            return native_zero_start(indptr, indices, values, rhs)
        try:
            return _amgx_solve_impl(
                indptr,
                indices,
                values,
                rhs,
                None,
                config_str=config_str,
                transpose_solve=True,
                reuse_setup=reuse_setup,
                block_dim=block_dim,
            )[0]
        except Exception:
            # An explicit Aᵀ where the native transposed solve cannot be used,
            # as in the adjoint rule.
            A = jsp.BCSR((values, indices, indptr), shape=(rhs.shape[0], rhs.shape[0]))
            A_T = jsp.BCSR.from_bcoo(A.to_bcoo().transpose())
            return native_zero_start(A_T.indptr, A_T.indices, A_T.data, rhs)

    return native, native_zero_start, native_transpose


@functools.lru_cache(maxsize=32)
def _get_implicit_primitive(
    config_str: str,
    is_symmetric: bool = False,
    return_stats: bool = False,
    reuse_setup: bool = False,
    res_history_len: int = 0,
    use_x0: bool = False,
    block_dim: int = 1,
) -> Callable:
    """The implicit-policy single-GPU solve (``jaxamg.core``): ``solve(A, b, x0)
    -> (x, info)`` with forward, reverse and higher-order derivatives."""
    from .core import implicit_solver

    native, zero_start, transpose = _single_gpu_services(
        config_str,
        is_symmetric,
        return_stats,
        reuse_setup,
        res_history_len,
        use_x0,
        block_dim,
    )
    return implicit_solver(native, zero_start, transpose, is_symmetric)


_DERIVATIVE_POLICIES = ("implicit", "adjoint")


def _transpose_plan_operands(
    A: jsp.BCSR, plan: TransposePlan | None
) -> tuple[jax.Array, ...]:
    """Transpose metadata as runtime operands: ``Aᵀ``'s structure, then the
    routing arrays ``(local_source, local_target, send, recv_target)``."""
    if plan is None:
        empty = jnp.empty(0, dtype=jnp.int32)
        return A.indices, A.indptr, empty, empty, empty, empty

    with temp_enable_x64():
        indices = jnp.asarray(plan.indices, dtype=jnp.int64)
    return (
        indices,
        jnp.asarray(plan.indptr),
        jnp.asarray(plan.local_source_ids),
        jnp.asarray(plan.local_target_ids),
        jnp.asarray(plan.send_ids),
        jnp.asarray(plan.recv_target_ids),
    )


@functools.lru_cache(maxsize=32)
def _get_adjoint_primitive_mpi(
    config_str: str,
    nglobal: int,
    comm_ptr: int,
    lrank: int,
    is_symmetric: bool = False,
    transpose_nnz: int | None = None,
    halo_exchange: Any = None,
    transpose_exchange: Any = None,
    transpose_reverse: bool = False,
    return_stats: bool = False,
    reuse_setup: bool = False,
    res_history_len: int = 0,
    use_x0: bool = False,
    block_dim: int = 1,
    ordered: bool = True,
) -> Callable:
    """
    The adjoint-rule pure-MPI solve (a ``custom_vjp``, reverse mode only),
    cached per configuration.

    The transpose values and the backward halo travel by neighbour exchanges
    (``jaxamg.transport``); ``halo_exchange``/``transpose_exchange`` are their
    plans.

    """

    # The backward pass's gradient w.r.t. A needs the solution at the columns
    # this rank's rows reference. The halo arrays (col_to_combined, send_ids)
    # are pattern-specific, so they flow as custom_vjp operands; only the
    # static exchange plans are captured by this memoized factory.
    @jax.custom_vjp
    def solve(
        A: jsp.BCSR,
        b: jax.Array,
        x0: jax.Array,
        halo: tuple[jax.Array, ...],
        transpose: tuple[jax.Array, ...],
    ) -> tuple[jax.Array, jax.Array]:
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

        x, info = _amgx_solve_mpi_impl(
            A.indptr,
            A.indices,
            A.data,
            b,
            x0,
            nglobal_arr,
            comm_ptr_arr,
            lrank_arr,
            config_str=config_str,
            return_stats=return_stats,
            reuse_setup=reuse_setup,
            res_history_len=res_history_len,
            use_x0=use_x0,
            block_dim=block_dim,
            ordered=ordered,
        )

        return x, info

    def fwd(A, b, x0, halo, transpose):
        out = solve(A, b, x0, halo, transpose)
        x, info = out
        return out, (A, x, halo, transpose)

    def bwd(residuals, g):
        g_x, _ = g
        A, x, halo, transpose = residuals
        col_to_combined, send_ids, row_indices = halo
        transpose_indices, transpose_indptr, *routing = transpose

        # Backward solves always start from zero: x0 shifts only the forward
        # iteration, never the solution, so the adjoint ignores it (and skips
        # the residual-history readback).
        adj_solver = _get_adjoint_primitive_mpi(
            config_str,
            nglobal,
            comm_ptr,
            lrank,
            is_symmetric=is_symmetric,
            transpose_nnz=len(A.data),
            halo_exchange=halo_exchange,
            transpose_exchange=transpose_exchange,
            transpose_reverse=not transpose_reverse,
            return_stats=return_stats,
            reuse_setup=reuse_setup,
            block_dim=block_dim,
            ordered=ordered,
        )

        # Backward solve: A^T @ adj_b = g_x
        if is_symmetric:
            # Symmetric: skip the distributed transpose.
            adj_b, _ = adj_solver(A, g_x, g_x, halo, transpose)
        else:
            # Distributed transpose by neighbour exchange (JIT-compatible,
            # GPU-direct when MPI4JAX_USE_CUDA_MPI=1).
            if transpose_nnz is None:
                raise ValueError(
                    "a transpose plan is required for nonsymmetric MPI gradients"
                )
            # Order the transpose after the forward solve (it otherwise reads
            # only A); without this XLA may interleave their MPI collectives in
            # a rank-inconsistent order and deadlock.
            a_data, _ = jax.lax.optimization_barrier((A.data, x))
            at_data = transpose_values(
                a_data,
                tuple(routing),
                transpose_exchange,
                transpose_nnz,
                reverse=transpose_reverse,
                ordered=ordered,
            )

            # Reconstruct BCSR for A^T
            A_T = jsp.BCSR(
                (at_data, transpose_indices, transpose_indptr), shape=A.shape
            )

            # The nested Aᵀ solve's own backward (reverse over reverse)
            # routes Aᵀ's values back to A's: the same plan, reversed.
            adj_b, _ = adj_solver(A_T, g_x, g_x, halo, (A.indices, A.indptr, *routing))

        # Gradient w.r.t. A: ∂L/∂A_ij = -adj_b[i] * x[j]. Fetch only the solution
        # entries this rank's rows reference via the halo exchange, ordered after
        # the backward solve (same as the transpose above) so all ranks issue MPI
        # collectives in a consistent order.
        x_bar, _ = jax.lax.optimization_barrier((x, adj_b))
        x_combined = halo_gather(x_bar, send_ids, halo_exchange, ordered=ordered)

        grad_values = -adj_b[row_indices] * x_combined[col_to_combined]
        grad_A = jsp.BCSR((grad_values, A.indices, A.indptr), shape=A.shape)

        # The exact solution does not depend on the initial guess.
        return (
            grad_A,
            adj_b,
            jnp.zeros_like(adj_b),
            (None, None, None),
            (None, None, None, None, None, None),
        )

    solve.defvjp(fwd, bwd)
    return solve


@functools.lru_cache(maxsize=32)
def _get_implicit_primitive_mpi(
    config_str: str,
    nglobal: int,
    comm_ptr: int,
    lrank: int,
    is_symmetric: bool = False,
    transpose_nnz: int | None = None,
    halo_exchange: Any = None,
    transpose_exchange: Any = None,
    return_stats: bool = False,
    reuse_setup: bool = False,
    res_history_len: int = 0,
    use_x0: bool = False,
    block_dim: int = 1,
    ordered: bool = True,
) -> Callable:
    """The implicit-policy pure-MPI solve (``jaxamg.core``), with the adjoint
    rule's signature ``solve(A, b, x0, halo, transpose)``.

    Services: the native MPI solves (the adjoint rule's configuration for the
    zero-start and transposed solves), the explicit ``Aᵀ`` from the transpose
    plan, and a local SpMV whose halo gather is a linear neighbour exchange
    with a declared transpose (``jaxamg.transport``).
    """
    from .core import implicit_solver

    nglobal_arr = np.array([nglobal], dtype=np.int32)
    low = np.int32(np.uint32(comm_ptr & 0xFFFFFFFF))
    high = np.int32(np.uint32((comm_ptr >> 32) & 0xFFFFFFFF))
    comm_ptr_arr = np.array([low, high], dtype=np.int32)
    lrank_arr = np.array([lrank], dtype=np.int32)

    def call(indptr, indices, values, b, x0, *, stats, history, warm):
        return _amgx_solve_mpi_impl(
            indptr,
            indices,
            values,
            b,
            x0,
            jnp.asarray(nglobal_arr),
            jnp.asarray(comm_ptr_arr),
            jnp.asarray(lrank_arr),
            config_str=config_str,
            return_stats=stats,
            reuse_setup=reuse_setup,
            res_history_len=history,
            use_x0=warm,
            block_dim=block_dim,
            ordered=ordered,
        )

    def native(indptr, indices, values, b, x0, aux):
        return call(
            indptr,
            indices,
            values,
            b,
            x0,
            stats=return_stats,
            history=res_history_len,
            warm=use_x0,
        )

    def native_zero_start(indptr, indices, values, rhs, aux):
        # The adjoint rule's configuration: zero start, no residual history.
        return call(
            indptr, indices, values, rhs, rhs, stats=return_stats, history=0, warm=False
        )[0]

    def native_transpose(indptr, indices, values, rhs, aux):
        if is_symmetric:
            return native_zero_start(indptr, indices, values, rhs, aux)
        if transpose_nnz is None:
            raise ValueError(
                "a transpose plan is required for nonsymmetric MPI gradients"
            )
        _, transpose = aux
        transpose_indices, transpose_indptr, *routing = transpose
        # Order the value exchange after the cotangent it answers (and so after
        # the forward solve), keeping every rank's collectives in one order.
        values, _ = jax.lax.optimization_barrier((values, rhs))
        at_data = transpose_values(
            values, tuple(routing), transpose_exchange, transpose_nnz, ordered=ordered
        )
        return native_zero_start(transpose_indptr, transpose_indices, at_data, rhs, aux)

    def layout_spmv(values, indices, indptr, aux, x):
        (col_to_combined, send_ids, row_indices), _ = aux
        combined = halo_gather(x, send_ids, halo_exchange, ordered=ordered)
        return jax.ops.segment_sum(
            values * combined[col_to_combined],
            row_indices,
            num_segments=x.shape[0],
            indices_are_sorted=True,
        )

    core = implicit_solver(
        native,
        native_zero_start,
        native_transpose,
        is_symmetric,
        layout_spmv,
        collective=True,
    )

    def solve(A, b, x0, halo, transpose):
        return core(A, b, x0, (halo, transpose))

    return solve


def _format_and_save_stats(
    stats_str: str,
    save_stats_file: str | os.PathLike,
    comm: "Comm | None" = None,
    mpi_cache: dict | None = None,
) -> None:
    """Resolve MPI rank and save formatted AmgX statistics to a file."""
    rank: int | None = None
    if comm is not None:
        rank = comm.Get_rank()
    elif mpi_cache is not None and "lrank" in mpi_cache:
        rank = mpi_cache["lrank"]
    format_amgx_stats(stats_str, save_stats_file, rank=rank)
    if rank is None or rank == 0:
        print(f"Stats saved to {save_stats_file}")


def _capture_and_save_stats(
    save_stats_file: str | os.PathLike,
    comm: "Comm | None" = None,
    mpi_cache: dict | None = None,
) -> None:
    """Read the captured AmgX statistics from the extension and save them."""
    stats_str = _ensure_backend().get_stats_string()
    _format_and_save_stats(stats_str, save_stats_file, comm=comm, mpi_cache=mpi_cache)


def solve(
    A: MatrixOrOperator,
    b: ArrayLike,
    x0: ArrayLike | None = None,
    config: dict | None = None,
    block_dim: int = 1,
    comm: "Comm | None" = None,
    nglobal: int | None = None,
    partition_info: tuple[int, int] | None = None,
    save_stats_file: str | os.PathLike | None = None,
    reuse_setup: bool = False,
    nullspace: NullSpaceSpec = None,
    transpose_nullspace: NullSpaceSpec = None,
    labels: LabelSpec = None,
    label_sum: Callable[[jax.Array], jax.Array] | None = None,
    derivative: str = "implicit",
    **kwargs: Any,
) -> tuple[jax.Array, dict]:
    """Solve `Ax=b` using the AmgX backend. See [Examples](examples.md) for usage.

    Args:
        A: Matrix or callable operator A(x). All matrices/operators are converted to `jax.experimental.sparse.bcsr` sparse matrices internally. In MPI mode this is the local partition.
        b: Right-hand-side vector. In MPI mode this is the local RHS partition.
        x0: Optional initial guess (same shape as `b`; local partition in MPI mode). Defaults to zero. A good warm start (e.g. the previous solution in a time-stepping or optimization loop) cuts iterations; it does not change the converged solution or its gradients. Note that with the default `RELATIVE_INI` convergence the tolerance is relative to the *initial* residual, so a very good `x0` tightens the target; consider `convergence="ABSOLUTE"` for warm-started loops.
        config: AmgX configuration dictionary (see [Solver Configuration](config.md) for details). If `None`, JAX-AMG defaults are used.
        block_dim: Treat the matrix as a block matrix with square `block_dim x block_dim` blocks (e.g. coupled multi-component PDE systems with node-major interleaved unknowns: row `i*block_dim + c` is component `c` of node `i`). `A` and `b` keep their ordinary scalar CSR/vector form; the conversion to AmgX's BSR format happens internally. Rows must be divisible by `block_dim` (each rank's local partition in MPI mode). Since AmgX's classical AMG does not support blocks, the AMG defaults switch to aggregation (`SIZE_2` + `BLOCK_JACOBI`); explicitly configured CLASSICAL AMG is rejected. In MPI mode the aggregation defaults use block-Jacobi coarse sweeps instead of `DENSE_LU_SOLVER` (which is broken in AmgX for distributed block matrices on 3+ ranks and rejected if configured explicitly); see [Solver Configuration](config.md#block-matrices).
        comm: MPI communicator (typically `mpi4py.MPI.COMM_WORLD`). If provided, the solve runs in MPI mode. If not provided, MPI mode can still be used if MPI metadata has already been attached via `with_cache(..., mpi=...)`.
        nglobal: Global matrix row count for MPI mode. Required when `comm` is provided and MPI metadata is not pre-attached to `A`.
        partition_info: `(row_start, row_end)` owned by this rank in MPI mode.  Required when `comm` is provided and MPI metadata is not pre-attached to `A`.
        save_stats_file: Optional file path to save detailed AmgX solver statistics.  If None, no file is created.
        reuse_setup: For repeated solves with the same sparsity pattern, skip warm `AMGX_solver_resetup` and keep the cached hierarchy. This is cheaper per solve but may require more iterations if matrix coefficients change significantly.
        nullspace: Basis of `null(A)` for singular systems: `"constant"`, a length-`n` vector, or an `(n, k)` array (local rows in MPI mode). The solution is pinned orthogonal to it, and the transpose of that pin projects the adjoint right-hand side onto `range(Aᵀ)`, which makes the backward solve converge. Gradients w.r.t. `A` assume perturbations that preserve the declared null spaces (`dA·N = 0`, `Mᵀ·dA = 0`), as coefficient changes of a conservative discretization do. Defaults to the basis attached with `with_cache`.
        transpose_nullspace: Basis of `null(Aᵀ)` (same formats). `b` is projected onto `range(A)` (removed fraction in `info["rhs_inconsistency"]`) and, by transposition, the adjoint solution is pinned: the forward returns `A⁺b`, `jax.grad` returns `(Aᵀ)⁺g`. Equals `nullspace` for symmetric `A` (filled in when `A` is marked symmetric); for nonsymmetric `A` it differs, e.g. `A = D⁻¹L` has `nullspace="constant"` but `transpose_nullspace=V` (cell volumes). Defaults to the basis attached with `with_cache`.
        labels: The pair `(count, labels)`, as `scipy.sparse.csgraph.connected_components` returns it: a static label count and one integer label per row (local rows in MPI mode, possibly traced), -1 for rows in no component, for a disconnected domain: every declared `nullspace`/`transpose_nullspace` column then applies separately on each label's rows (the null space is block diagonal over the labels), e.g. `nullspace="constant"` with labels declares one constant vector per component. A label whose columns vanish on its rows (no rows, or all of them -1 in this solve) is skipped exactly; columns dependent within a label are rejected (concrete values). Without labels the null space is the columns themselves. Defaults to the labels attached with `with_cache`.
        label_sum: In MPI mode with labels, the reduction of each rank's per-label partial sums (an array of `count` rows) to the labels' totals, a linear function (a `jaxamg.linear_map.LinearMap` where JAX's own transpose would be wrong, as for an MPI all-reduce). By default, the all-reduce over global label numbers (one `count`-row array on every rank); a caller whose labels span ranks in its own numbering supplies the reduction. Defaults to the one attached with `with_cache`.
        derivative: `"implicit"` (default) differentiates `x = A⁻¹b` implicitly on the fixed pattern, `ẋ = A⁻¹(ḃ − Ȧx)` with each `A⁻¹` a native solve: forward and reverse mode, and higher orders where the operator supports them. `"adjoint"` is the reverse-only rule. Both use the same first reverse rule. Neither differentiates the AmgX iterations: derivatives are those of the exact solution, accurate to the solve tolerance.
        **kwargs: Additional AmgX config parameters. These override values in `config` when both are provided.

    Returns:
        x: Solution vector (float32 or float64). In MPI mode, returns local portion.
        info: Dictionary containing `iterations`, `residual`, `status`, and `residual_history` (residual norm per outer iteration, entry 0 being the initial residual; inside `jit` it has fixed length `max_iters + 1` with NaN padding past entry `iterations`). With `transpose_nullspace`, also `rhs_inconsistency` (`‖b − b'‖/‖b‖`).

    Warns:
        NullSpaceWarning: `A·1 = 0` without a declared `nullspace`; `nullspace` without `transpose_nullspace` (or vice versa) for a matrix not marked symmetric; a basis failing `A·N ≈ 0` / `Aᵀ·M ≈ 0`; or a `DENSE_LU_SOLVER` coarse solve. These checks run only on concrete matrix values.
    """

    if derivative not in _DERIVATIVE_POLICIES:
        raise ValueError(
            f"derivative must be one of {_DERIVATIVE_POLICIES}, got {derivative!r}"
        )
    b = jnp.asarray(b)

    # Check for GPU backend
    if jax.default_backend() != "gpu":
        raise RuntimeError(
            f"AMGX requires a GPU backend, but JAX is using '{jax.default_backend()}'. "
            "Please ensure you have a CUDA-enabled GPU and JAX is installed with CUDA support."
        )

    # Load the native extension and register FFI targets (sets HAS_MPI).
    _ensure_backend()

    block_dim = int(block_dim)
    if block_dim < 1:
        raise ValueError("block_dim must be a positive integer")
    if block_dim > 1 and b.shape[0] % block_dim != 0:
        raise ValueError(
            f"b length {b.shape[0]} is not divisible by block_dim {block_dim}"
            " (in MPI mode each rank's local partition must be block-aligned)"
        )

    # MPI cache may be pre-attached to A via `with_cache`
    mpi_cache = getattr(A, "_mpi_cache", None)

    # Null-space bases: explicit arguments override ones attached via with_cache.
    if nullspace is None:
        nullspace = getattr(A, "_nullspace", None)
    if transpose_nullspace is None:
        transpose_nullspace = getattr(A, "_transpose_nullspace", None)
    if labels is None:
        labels = getattr(A, "_labels", None)
    if label_sum is None:
        label_sum = getattr(A, "_label_sum", None)
    singular = nullspace is not None or transpose_nullspace is not None
    if labels is not None and not singular:
        raise ValueError(
            "labels apply to the declared nullspace/transpose_nullspace columns; "
            "declare them (e.g. nullspace='constant')"
        )

    # Prepare configuration string/file (skip if using mpi_cache which already has config_str)
    if mpi_cache is not None:
        if config is not None or kwargs:
            warnings.warn(
                "A carries cached MPI metadata (with_cache(..., mpi=...)); the "
                "config/kwargs passed to solve() are ignored in favor of the "
                "cached config. Pass the config to cache_mpi_metadata() instead.",
                stacklevel=2,
            )
        config_str = mpi_cache["config_str"]
        cached_block_dim = mpi_cache.get("block_dim", 1)
        if block_dim not in (1, cached_block_dim):
            warnings.warn(
                "A carries cached MPI metadata; the block_dim passed to "
                "solve() is ignored in favor of the cached value. Pass "
                "block_dim to cache_mpi_metadata() instead.",
                stacklevel=2,
            )
        block_dim = cached_block_dim
        if save_stats_file is not None and '"print_solve_stats"' not in config_str:
            warnings.warn(
                "save_stats_file was passed, but the cached MPI config was "
                "prepared without stats output; the stats file will be missing "
                "solver statistics. Pass save_stats=True to cache_mpi_metadata().",
                stacklevel=2,
            )
    else:
        config_str = amgx_config.prepare_config(
            config,
            save_stats=(save_stats_file is not None),
            mpi=(comm is not None),
            block_dim=block_dim,
            singular=singular,
            **kwargs,
        )
    if singular and amgx_config.uses_dense_lu_coarse_solver(config_str):
        warnings.warn(_DENSE_LU_MSG, NullSpaceWarning, stacklevel=2)

    # Residual-history slots appended to the stats output (one per outer
    # iteration, plus the initial residual).
    res_history_len = amgx_config.outer_max_iters(config_str) + 1

    # Detect desired precision (non-float RHS dtypes are promoted to float32)
    target_dtype = get_preferred_dtype(A, b)
    if b.dtype != target_dtype:
        b = b.astype(target_dtype)

    use_x0 = x0 is not None
    if use_x0:
        x0 = jnp.asarray(x0)
        if x0.shape != b.shape:
            raise ValueError(
                f"x0 must have the same shape as b; got {x0.shape} vs {b.shape}"
            )
        if x0.dtype != target_dtype:
            x0 = x0.astype(target_dtype)
    # The primitive always takes an x0 operand; b doubles as a same-shape
    # dummy that the backend ignores when use_x0 is off.
    x0_arg = x0 if use_x0 else b

    # Check for symmetry attribute on A
    is_symmetric = getattr(A, "_is_symmetric", False)
    mpi_primitive = (
        _get_implicit_primitive_mpi
        if derivative == "implicit"
        else _get_adjoint_primitive_mpi
    )

    # Branch: MPI mode or single-GPU mode
    if mpi_cache is not None or comm is not None:
        # MPI MODE
        if not HAS_MPI:
            raise RuntimeError(
                "jaxamg was built without MPI support, but an MPI solve was "
                "requested (comm was passed or MPI metadata is attached to A). "
                "Rebuild with MPI enabled (JAXAMG_ENABLE_MPI=1, with mpicxx on "
                "PATH) to use distributed solves."
            )
        if mpi_cache is None:
            # Validate parameters for non-cache path
            if nglobal is None:
                raise ValueError("nglobal must be provided when using MPI mode")
            if partition_info is None:
                raise ValueError(
                    "partition_info (row_start, row_end) must be provided when using MPI mode"
                )

        # Convert A to BCSR with int64 indices (required for MPI)
        from .halo import HaloOperator

        if isinstance(A, HaloOperator):
            # Rank-local materialization over [x_local | x_ghost].
            if mpi_cache is not None:
                cached_comm = resolve_comm(mpi_cache["comm_ptr"])
                counts = mpi_cache["recvcounts_tuple"]
                first_row = int(sum(counts[: cached_comm.Get_rank()]))
                n_all = mpi_cache["nglobal"]
            else:
                assert partition_info is not None
                first_row, n_all = int(partition_info[0]), int(nglobal)
            A_csr = A._local_matrix(
                first_row,
                n_all,
                get_preferred_dtype(None, b),
                traced=isinstance(b, jax.core.Tracer),
            )
        else:
            A_csr = to_bcsr_matrix(A, b=b, use_int64_indices=True)
        # Per-rank programs order their communication on mpi4jax's effect.
        ordered = True

        if mpi_cache is not None:
            # Use pre-cached MPI metadata
            halo_plan = mpi_cache["halo_plan"]
            row_indices = mpi_cache.get("row_indices")
            if row_indices is None:
                # Caches built without row indices (e.g. the sharding-internal
                # solver caches) fall back to the traceable computation.
                row_indices = jnp.repeat(
                    jnp.arange(A_csr.shape[0], dtype=jnp.int32),
                    A_csr.indptr[1:] - A_csr.indptr[:-1],
                    total_repeat_length=len(A_csr.data),
                )
            transpose_plan = mpi_cache.get("transpose_plan")
            transpose_operands = _transpose_plan_operands(A_csr, transpose_plan)
            solver = mpi_primitive(
                mpi_cache["config_str"],
                mpi_cache["nglobal"],
                mpi_cache["comm_ptr"],
                mpi_cache["lrank"],
                is_symmetric=is_symmetric,
                transpose_nnz=None if transpose_plan is None else transpose_plan.nnz,
                halo_exchange=halo_plan.exchange,
                transpose_exchange=(
                    None if transpose_plan is None else transpose_plan.exchange
                ),
                return_stats=1 if save_stats_file else 0,
                reuse_setup=reuse_setup,
                res_history_len=res_history_len,
                use_x0=use_x0,
                block_dim=block_dim,
                ordered=ordered,
            )

        elif comm is not None:
            # Compute metadata dynamically
            import importlib.util

            if importlib.util.find_spec("mpi4py") is None:
                raise ImportError(
                    "mpi4py is required for MPI mode. Install it with: pip install mpi4py"
                )

            # Get MPI rank and compute local GPU assignment
            rank = comm.Get_rank()
            lrank = rank % jax.device_count()
            # Register the communicator and get its address (so the backward pass
            # can recover it for its collectives).
            comm_ptr = register_comm(comm)

            # Gather partition sizes from all ranks (row partition + displacements)
            n_local = A_csr.shape[0]
            all_sizes_list = comm.allgather(n_local)
            recvcounts_val = np.array(all_sizes_list, dtype=np.int32)
            displs_val = np.cumsum(np.concatenate(([0], recvcounts_val[:-1]))).astype(
                np.int32
            )
            recvcounts_tuple = tuple(recvcounts_val.tolist())

            # Validate partition_info against the partition actually implied by
            # the local matrix shapes (which is what AmgX uses). Reduce first so
            # every rank raises together -- a rank-divergent raise would leave
            # the other ranks deadlocked in the collectives below.
            from mpi4py import MPI

            row_start = int(displs_val[rank])
            derived_partition = (row_start, row_start + n_local)
            mismatch = tuple(partition_info) != derived_partition
            if comm.allreduce(mismatch, op=MPI.LOR):
                detail = (
                    f"rank {rank}: partition_info {tuple(partition_info)} != "
                    f"derived {derived_partition}"
                    if mismatch
                    else f"rank {rank} is consistent, but another rank's is not"
                )
                raise ValueError(
                    "partition_info does not match the row partition derived "
                    f"from the local matrix shapes ({detail}). Each rank must "
                    "pass its own (row_start, row_end) matching its local "
                    "partition."
                )

            # Halo-exchange plan for the backward pass (fetches only the remote
            # solution entries this rank references, instead of all-gathering).
            halo_plan = build_halo_plan(
                A_csr.indices,
                recvcounts_tuple,
                (row_start, row_start + n_local),
                comm,
            )
            transpose_plan = (
                None
                if is_symmetric
                else build_transpose_plan(
                    A_csr.indices,
                    A_csr.indptr,
                    recvcounts_tuple,
                    derived_partition,
                    comm,
                )
            )
            transpose_operands = _transpose_plan_operands(A_csr, transpose_plan)
            row_indices = np.repeat(
                np.arange(n_local, dtype=np.int32),
                np.diff(np.asarray(A_csr.indptr)),
            ).astype(np.int32)

            solver = mpi_primitive(
                config_str,
                nglobal,
                comm_ptr,
                lrank,
                is_symmetric=is_symmetric,
                transpose_nnz=None if transpose_plan is None else transpose_plan.nnz,
                halo_exchange=halo_plan.exchange,
                transpose_exchange=(
                    None if transpose_plan is None else transpose_plan.exchange
                ),
                return_stats=1 if save_stats_file else 0,
                reuse_setup=reuse_setup,
                res_history_len=res_history_len,
                use_x0=use_x0,
                block_dim=block_dim,
                ordered=ordered,
            )

        halo_args = (
            jnp.asarray(halo_plan.col_to_combined),
            jnp.asarray(halo_plan.send_ids),
        )
        if mpi_cache is not None:
            # AmgX runs on the cached communicator; so must everything else.
            comm_obj = resolve_comm(mpi_cache["comm_ptr"])
            if comm is not None and register_comm(comm) != mpi_cache["comm_ptr"]:
                warnings.warn(
                    "comm differs from the communicator cached on A; using the "
                    "cached one.",
                    stacklevel=2,
                )
        else:
            assert comm is not None
            comm_obj = comm
        # Global reductions for the null-space projections.
        reduce_sum = make_mpi_reduce_sum(comm_obj)

        def run(b_: jax.Array, x0_: jax.Array) -> tuple[jax.Array, jax.Array]:
            return solver(
                A_csr,
                b_,
                x0_,
                (*halo_args, jnp.asarray(row_indices)),
                transpose_operands,
            )

        def local_spmv(v: jax.Array, data: jax.Array) -> jax.Array:
            # This rank's rows of A v, through the halo exchange.
            combined = halo_gather(v, halo_args[1], halo_plan.exchange, ordered=ordered)
            return jax.ops.segment_sum(
                data * combined[halo_args[0]],
                row_index(A_csr),
                num_segments=A_csr.shape[0],
            )

        def matvec(basis: jax.Array) -> jax.Array:
            return jnp.stack(
                [local_spmv(basis[:, j], A_csr.data) for j in range(basis.shape[1])],
                axis=1,
            )

        column_labels: Callable[[jax.Array], jax.Array] | None

        def column_labels(rows: jax.Array) -> jax.Array:
            # Each stored entry's column label through the halo exchange; a
            # caller's own numbering compares only on this rank's columns.
            combined = halo_gather(
                rows, halo_args[1], halo_plan.exchange, ordered=ordered
            )
            columns = combined[halo_args[0]]
            if label_sum is None:
                return columns
            return jnp.where(halo_args[0] < A_csr.shape[0], columns, -2)

        def matvec_T(basis: jax.Array) -> jax.Array:
            # Aᵀ as the transpose of the halo SpMV (the reverse exchange).
            transpose = jax.linear_transpose(
                lambda v: local_spmv(v, A_csr.data),
                jnp.zeros(A_csr.shape[0], basis.dtype),
            )
            return jnp.stack(
                [transpose(basis[:, j])[0] for j in range(basis.shape[1])], axis=1
            )

    else:
        # Single-GPU mode: use int32 indices
        A_csr = to_bcsr_matrix(A, b)
        # Get cached primitive for this configuration and derivative policy
        primitive = (
            _get_implicit_primitive
            if derivative == "implicit"
            else _get_adjoint_primitive
        )
        solver = primitive(
            config_str,
            is_symmetric=is_symmetric,
            return_stats=1 if save_stats_file else 0,
            reuse_setup=reuse_setup,
            res_history_len=res_history_len,
            use_x0=use_x0,
            block_dim=block_dim,
        )
        comm_obj = None
        reduce_sum = None

        def run(b_: jax.Array, x0_: jax.Array) -> tuple[jax.Array, jax.Array]:
            return solver(A_csr, b_, x0_)

        def matvec(basis: jax.Array) -> jax.Array:
            return A_csr @ basis

        def matvec_T(basis: jax.Array) -> jax.Array:
            return A_csr.to_bcoo().T @ basis

        column_labels = None

    # Null-space projections as JAX ops around the primitive; their
    # transposes are the adjoint projections (see nullspace.py).
    n_local = A_csr.shape[0]
    if comm_obj is None:
        n_columns = None
    else:
        n_columns = mpi_cache["nglobal"] if mpi_cache is not None else nglobal
    N = as_nullspace_basis(nullspace, n_local, target_dtype, "nullspace", n_columns)
    M = as_nullspace_basis(
        transpose_nullspace, n_local, target_dtype, "transpose_nullspace", n_columns
    )
    labels = as_labels(labels, n_local, comm=comm_obj)
    if N is not None:
        validate_basis(N, "nullspace", comm_obj, labels, label_sum)
    if M is not None:
        validate_basis(M, "transpose_nullspace", comm_obj, labels, label_sum)
    if M is None and N is not None:
        if is_symmetric:
            M = N
        else:
            warnings.warn(_MISSING_TRANSPOSE_MSG, NullSpaceWarning, stacklevel=2)
    elif N is None and M is not None:
        if is_symmetric:
            N = M
        else:
            warnings.warn(_MISSING_NULLSPACE_MSG, NullSpaceWarning, stacklevel=2)
    # Unlabelled bases share one column scale across ranks. Labelled bases
    # are scaled per component by validation and projection, before products
    # are formed; a whole-column scale could erase smaller components.
    if labels is None and (N is not None or M is not None):
        from .nullspace import unit_bases

        basis_max: Callable[[jax.Array], jax.Array] | None = None
        if comm_obj is not None:
            import mpi4jax
            from mpi4py import MPI

            def basis_max(value):
                return mpi4jax.allreduce(value, op=MPI.MAX, comm=comm_obj)

        N, M = unit_bases((N, M), basis_max)
    if N is None and M is None:
        warn_if_singular(A_csr, comm=comm_obj, stacklevel=3)
    else:
        checks: dict[str, Any] = dict(comm=comm_obj, stacklevel=3, labels=labels)
        if labels is not None:
            checks["column_labels"] = column_labels
        if N is not None:
            verify_nullspace(A_csr, N, matvec, "nullspace", **checks)
        if M is not None:
            verify_nullspace(A_csr, M, matvec_T, "transpose_nullspace", **checks)

    # The labels' sums across ranks: the caller's, or the all-reduce.
    project_sum = reduce_sum if labels is None or label_sum is None else label_sum
    rhs_inconsistency = None
    if M is not None:
        b_proj = project_out(b, M, project_sum, labels)
        rhs_inconsistency = relative_norm(b - b_proj, b, reduce_sum)
        b = b_proj
    if M is not None and not use_x0:
        x0_arg = b

    x, info = run(b, x0_arg)

    if N is not None:
        x = project_out(x, N, project_sum, labels)

    if any(isinstance(v, jax.core.Tracer) for v in (x, info, rhs_inconsistency)):
        # Inside JIT or another trace (e.g. a gradient, where info can come back
        # concrete while x and the projection are traced): return as-is.
        # The history keeps its fixed trace-time length (outer max_iters + 1),
        # NaN-padded past entry `iterations`.
        traced_info = {
            "iterations": info[0],
            "residual": info[1],
            "status": info[2],
            "residual_history": info[3:],
        }
        if rhs_inconsistency is not None:
            traced_info["rhs_inconsistency"] = rhs_inconsistency
        return x, traced_info

    info_dict = {
        "iterations": int(info[0]),
        "residual": float(info[1]),
        "status": AMGXStatus(int(info[2])),
        # Entry i is the residual norm after outer iteration i (entry 0 is the
        # initial residual); trim the NaN padding outside a trace.
        "residual_history": info[3 : 4 + int(info[0])],
    }
    if rhs_inconsistency is not None:
        info_dict["rhs_inconsistency"] = float(rhs_inconsistency)
    # os.devnull enables stats capture in the FFI without writing a file; the
    # sharding interface uses it and reads the captured stats afterwards.
    if save_stats_file is not None and str(save_stats_file) != os.devnull:
        _capture_and_save_stats(save_stats_file, comm=comm, mpi_cache=mpi_cache)
    return x, info_dict


def clear_solver_cache() -> None:
    """
    Clear the internal C++ AmgX solver cache.
    This releases all cached AmgX resources (matrices, solvers, vectors).
    In MPI mode call it on every rank at the same point: the ranks' caches
    must agree, and a distributed solve refuses (on every rank) when they
    do not.
    """
    _ensure_backend().clear_solver_cache()


def get_solver_cache_info() -> dict[str, Any]:
    """
    Inspect the internal C++ AmgX solver caches.

    Returns:
        A dictionary with cache size/capacity and entry summaries
        for single-GPU and MPI caches, plus whether isolated mode
        (`JAXAMG_CACHE_SIZE=0`) is active.
    """
    solver_info = _ensure_backend().get_solver_cache_info()

    # Convert config strings to JSON
    solver_info["single_gpu"]["entries"] = [
        {**entry, "config": json.loads(entry["config"])}
        for entry in solver_info["single_gpu"]["entries"]
    ]
    solver_info["mpi"]["entries"] = [
        {**entry, "config": json.loads(entry["config"])}
        for entry in solver_info["mpi"]["entries"]
    ]

    return solver_info


def finalize() -> None:
    """
    Manually finalize AmgX resources.
    This clears the cache and calls AMGX_finalize.
    Normally only needed to be called manually in MPI mode to avoid shutdown-time resource warnings.
    """
    clear_solver_cache()
    _ensure_backend().finalize()
