"""Sparsity detection and assembly for matrix-free operators.

Turns a callable operator ``A(x)`` into its sparse matrix: (1) detect the sparsity
pattern, (2) colour the columns, (3) materialise the values with one operator
evaluation per colour. ``cache_coloring`` orchestrates it, tries the two detection
methods in order, and verifies the result -- so it is correct for ANY operator:

- **Tracing** (``trace_sparsity_pattern``): interpret the operator's jaxpr and
  propagate a connectivity (index-set) structure through each primitive to recover
  the EXACT *global* sparsity pattern in a single trace. Returns ``None`` for
  operators that can't be traced structurally (opaque calls, data-dependent
  indexing). This is the JAX-native analogue of the operator-overloading sparsity
  detection of SparseConnectivityTracer.jl [1, 2]: JAX is trace-based rather than
  dispatch-based, so instead of overloading a custom number type we trace the
  function to a jaxpr and interpret it.
- **Probing** (``probe_sparsity_pattern``): exhaustive one-hot basis-vector
  probing; the always-correct fallback when tracing is unavailable.

References:
    [1] A. Hill and G. Dalle, "Sparser, Better, Faster, Stronger: Sparsity
        Detection for Efficient Automatic Differentiation," arXiv:2501.17737
        (2025).
    [2] SparseConnectivityTracer.jl,
        https://github.com/adrhill/SparseConnectivityTracer.jl
"""

from __future__ import annotations

import contextlib
import warnings
from collections.abc import Callable, Iterator, Sequence
from typing import Any

import jax
import jax.experimental.sparse as jsp
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
from jax import lax
from jax.extend.core import Jaxpr, JaxprEqn, Literal, Var
from jax.typing import ArrayLike

from .utils import temp_enable_x64

# A connectivity matrix ``C`` has ``C[i, j] = 1`` iff output element ``i`` depends
# on global input ``j``; propagated through the jaxpr as a scipy CSR matrix. Typed
# as Any since scipy.sparse ships no usable stubs -- the alias documents intent at
# call sites without mypy trying (and failing) to check CSR internals.
Conn = Any

# --- Tracing-based detection: interpret the operator's jaxpr ---
# Element-wise primitives: output element depends on the union of its operands'
# elements at the same (broadcast) position. Unary ops preserve connectivity
# (value-dependence), incl. nominally derivative-zero ones (sign/floor/...) since
# we detect VALUE dependence to match probing-based materialization.
_ELEMENTWISE = {
    # n-ary
    "add",
    "sub",
    "mul",
    "div",
    "pow",
    "atan2",
    "max",
    "min",
    "rem",
    "nextafter",
    "and",
    "or",
    "xor",
    "add_any",
    "select_n",
    "select",
    "eq",
    "ne",
    "lt",
    "gt",
    "le",
    "ge",
    # unary
    "neg",
    "abs",
    "exp",
    "exp2",
    "expm1",
    "log",
    "log1p",
    "sin",
    "cos",
    "tan",
    "asin",
    "acos",
    "atan",
    "sinh",
    "cosh",
    "tanh",
    "asinh",
    "acosh",
    "atanh",
    "sqrt",
    "rsqrt",
    "cbrt",
    "square",
    "integer_pow",
    "logistic",
    "erf",
    "erfc",
    "erf_inv",
    "sign",
    "floor",
    "ceil",
    "round",
    "real",
    "imag",
    "conj",
    "is_finite",
    "not",
    "convert_element_type",
    "copy",
    "reduce_precision",
    "stop_gradient",
    "clamp",
}

# Higher-order primitives whose sub-jaxpr(s) we interpret recursively.
_CALL_PRIMS = {
    "pjit",
    "jit",
    "closed_call",
    "core_call",
    "xla_call",
    "custom_jvp_call",
    "custom_vjp_call",
    "remat2",
    "remat_call",
    "remat",
}

# Opaque -> cannot trace structure -> signal fallback to probing.
_OPAQUE = {
    "custom_call",
    "pure_callback",
    "io_callback",
    "ffi_call",
    "while",  # data-dependent iteration count -> not traced
}

_CPU = jax.devices("cpu")[0]

_UNKNOWN = object()  # sentinel: a value that depends on the operator input


@contextlib.contextmanager
def _host_exact() -> Iterator[None]:
    """Run host-side index/position binds on CPU with x64 enabled, so position
    arithmetic stays exact regardless of the global precision config."""
    with temp_enable_x64(), jax.default_device(_CPU):
        yield


class _BailOut(Exception):
    """Raised when the operator cannot be traced structurally; caller falls back."""


def _empty(n_rows: int, n_global: int) -> Conn:
    return sp.csr_matrix((n_rows, n_global), dtype=np.int8)


def _is_empty(C: Conn) -> bool:
    return C.nnz == 0


def _const_zero_mask(
    val: Any, in_shape: tuple[int, ...], out_shape: tuple[int, ...]
) -> np.ndarray:
    """Flat bool mask over output positions where ``val`` (broadcast from
    ``in_shape`` to ``out_shape``) is exactly zero."""
    is_zero = np.asarray(val) == 0
    if tuple(in_shape) == tuple(out_shape):
        return is_zero.reshape(-1)
    return np.broadcast_to(is_zero, out_shape).reshape(-1)


def _mul_zero_positions(
    in_shapes: list[tuple[int, ...]],
    in_vals: list[Any],
    out_shape: tuple[int, ...],
) -> np.ndarray | None:
    """Flat bool mask of ``mul`` outputs forced to zero by a structurally-zero
    constant operand (``0*x``), or ``None`` if neither operand is such a constant.
    Only ``mul``: div/integer_pow zero-skips fire only on nonlinear operators."""
    mask = None
    for shape, v in zip(in_shapes, in_vals):
        if v is _UNKNOWN:
            continue
        zm = _const_zero_mask(v, shape, out_shape)
        mask = zm if mask is None else mask | zm
    return mask


def _index_operand_positions(prim: str, n_operands: int) -> set[int]:
    """Operand positions that are indices/offsets rather than data, for a movement
    primitive. Their values steer where data lands but never flow into the output,
    so they are bound as concrete values (not connectivity id blocks)."""
    if prim == "dynamic_slice":
        return set(range(1, n_operands))  # operand, then one start index per dim
    if prim == "gather":
        return {1}  # operand, indices
    return set()  # pad: both operand and padding value are data


def _movement_rowmap(
    prim: str,
    params: dict[str, Any],
    in_shapes: list[tuple[int, ...]],
    in_vals: list[Any],
) -> tuple[np.ndarray, int]:
    """Return (rowmap, n_combined): for movement primitives, rowmap[o] is the
    source row (in the vstack of all operands) feeding output element o.

    Each operand element is labelled by its row in the vstacked connectivity, so
    binding the primitive on those labels recovers, per output element, which
    source element it came from. Index operands (slice offsets, gather indices)
    are bound as their concrete values instead -- and we bail if such an index is
    itself input-dependent (data-dependent indexing is not structurally traceable).
    """
    sizes = [int(np.prod(s)) for s in in_shapes]
    offsets = np.concatenate([[0], np.cumsum(sizes)])
    blocks = [
        np.arange(offsets[i], offsets[i + 1]).reshape(in_shapes[i])
        for i in range(len(in_shapes))
    ]

    if prim == "reshape":
        rowmap = blocks[0].reshape(-1)
    elif prim in ("squeeze", "expand_dims"):
        rowmap = blocks[0].reshape(-1)
    elif prim == "transpose":
        rowmap = np.transpose(blocks[0], params["permutation"]).reshape(-1)
    elif prim == "rev":
        rowmap = np.flip(blocks[0], params["dimensions"]).reshape(-1)
    elif prim == "slice":
        sl = tuple(
            slice(s, l, (params["strides"][k] if params["strides"] else 1))
            for k, (s, l) in enumerate(
                zip(params["start_indices"], params["limit_indices"])
            )
        )
        rowmap = blocks[0][sl].reshape(-1)
    elif prim == "broadcast_in_dim":
        out_shape = tuple(params["shape"])
        bdims = tuple(params["broadcast_dimensions"])
        in_shape = in_shapes[0]
        full = [1] * len(out_shape)
        for k, d in enumerate(bdims):
            full[d] = in_shape[k]
        rowmap = np.broadcast_to(blocks[0].reshape(full), out_shape).reshape(-1)
    elif prim == "concatenate":
        rowmap = np.concatenate(blocks, axis=params["dimension"]).reshape(-1)
    else:
        # pad / dynamic_slice / gather: let JAX compute the element map. Data
        # operands are bound as id blocks; index operands as concrete values.
        prim_obj = _PRIM_BY_NAME.get(prim)
        if prim_obj is None:
            raise _BailOut(f"no movement rule for {prim}")
        index_positions = _index_operand_positions(prim, len(blocks))
        args = []
        for i, block in enumerate(blocks):
            if i in index_positions:
                if in_vals[i] is _UNKNOWN:
                    raise _BailOut(f"{prim} with data-dependent indices")
                args.append(jnp.asarray(in_vals[i]))
            else:
                args.append(jnp.asarray(block))
        with _host_exact():
            out = prim_obj.bind(*args, **params)
        rowmap = np.asarray(out).reshape(-1)
    return rowmap.astype(np.int64), int(offsets[-1])


# Lazily-built name->primitive table for the rare bind path.
_PRIM_BY_NAME: dict = {}


def _build_prim_table() -> None:
    for name in ("pad", "dynamic_slice", "gather"):
        obj = getattr(lax, f"{name}_p", None)
        if obj is not None:
            _PRIM_BY_NAME[str(obj)] = obj


_build_prim_table()


def _reduce_map(
    params: dict[str, Any], in_shape: tuple[int, ...], out_shape: tuple[int, ...]
) -> tuple[np.ndarray, int]:
    """Map each operand element to its reduced output element (reduce_* ops)."""
    axes = set(params["axes"])
    keep = [d for d in range(len(in_shape)) if d not in axes]
    op_size = int(np.prod(in_shape))
    mesh = np.unravel_index(np.arange(op_size), in_shape)
    out_multi = tuple(mesh[d] for d in keep)
    out_shape = tuple(out_shape)
    if out_multi:
        out_flat = np.ravel_multi_index(out_multi, out_shape)
    else:
        out_flat = np.zeros(op_size, dtype=np.int64)
    return out_flat.astype(np.int64), int(np.prod(out_shape) if out_shape else 1)


def _interp(
    jaxpr: Jaxpr,
    consts: Sequence[Any],
    n_global: int,
    arg_conns: list[Conn],
    arg_vals: list[Any] | None = None,
) -> list[Conn]:
    """Interpret one (sub-)jaxpr, returning the connectivity of each outvar.

    arg_vals carries known concrete values of the inputs (or _UNKNOWN), so that
    constant-folding (e.g. of convolution kernels or scatter indices) works across
    call boundaries.
    """
    env: dict[Var, Conn] = {}
    vals: dict[Var, Any] = {}  # concrete values of input-INDEPENDENT vars
    if arg_vals is None:
        arg_vals = [_UNKNOWN] * len(arg_conns)

    def read(v: Var | Literal) -> Conn:
        if isinstance(v, Literal):
            return _empty(int(np.prod(np.shape(v.val))) or 1, n_global)
        return env[v]

    def val_of(v: Var | Literal) -> Any:
        if isinstance(v, Literal):
            return np.asarray(v.val)
        return vals.get(v, _UNKNOWN)

    def shape_of(v: Var | Literal) -> tuple[int, ...]:
        return tuple(v.aval.shape)

    for v, C, val in zip(jaxpr.invars, arg_conns, arg_vals):
        env[v] = C
        if val is not _UNKNOWN:
            vals[v] = val
    for cv, c in zip(jaxpr.constvars, consts):
        env[cv] = _empty(int(np.prod(np.shape(c))) or 1, n_global)
        vals[cv] = np.asarray(c)

    for eqn in jaxpr.eqns:
        prim = str(eqn.primitive)
        in_Cs = [read(v) for v in eqn.invars]
        out_shapes = [shape_of(v) for v in eqn.outvars]
        out_sizes = [int(np.prod(s)) or 1 for s in out_shapes]

        # Constant-fold input-independent vars (e.g. iota-built scatter indices,
        # precomputed grid metrics, convolution kernels) so they can be resolved
        # later. Pure call primitives (jit) fold too when all inputs are known.
        # Nullary generators like ``iota`` have no inputs (``all([])`` is True), so
        # they fold too -- this is what lets index chains built from ``jnp.arange``
        # resolve to concrete values for the gather/scatter/dynamic_slice rules.
        if prim not in _OPAQUE:
            in_vals = [val_of(v) for v in eqn.invars]
            if all(x is not _UNKNOWN for x in in_vals):
                try:
                    # Cast each input to the dtype the jaxpr declares for it:
                    # primitives with a fixed signature (e.g. pjit sub-jaxprs)
                    # require an exact dtype match, and a host int literal would
                    # otherwise arrive as int64 and mismatch an int32 operand.
                    with _host_exact():
                        outs = eqn.primitive.bind(
                            *[
                                jnp.asarray(x, dtype=v.aval.dtype)
                                for v, x in zip(eqn.invars, in_vals)
                            ],
                            **eqn.params,
                        )
                    outs = outs if eqn.primitive.multiple_results else [outs]
                    for ov, o in zip(eqn.outvars, outs):
                        vals[ov] = np.asarray(o)
                except Exception:
                    pass

        if prim in _OPAQUE:
            raise _BailOut(f"opaque primitive: {prim}")

        # Shortcut: if no operand carries any input dependence, outputs are empty.
        if all(_is_empty(C) for C in in_Cs):
            for v, sz in zip(eqn.outvars, out_sizes):
                env[v] = _empty(sz, n_global)
            continue

        if prim in _CALL_PRIMS:
            sub = eqn.params.get("jaxpr") or eqn.params.get("call_jaxpr")
            if sub is None:
                raise _BailOut(f"call primitive without jaxpr: {prim}")
            sub_jaxpr = getattr(sub, "jaxpr", sub)
            sub_consts = getattr(sub, "consts", [])
            in_vals = [val_of(v) for v in eqn.invars]
            outs = _interp(sub_jaxpr, sub_consts, n_global, in_Cs, in_vals)
            for v, C in zip(eqn.outvars, outs):
                env[v] = C
            continue

        if prim == "cond":
            # invars: [index, *operands]; branches: tuple of ClosedJaxpr. The
            # output is a union over branches (we don't know which is taken).
            pred_C, branch_conns = in_Cs[0], in_Cs[1:]
            branch_vals = [val_of(v) for v in eqn.invars[1:]]
            acc = None
            for br in eqn.params["branches"]:
                bj = getattr(br, "jaxpr", br)
                bc = getattr(br, "consts", [])
                outs = _interp(bj, bc, n_global, branch_conns, branch_vals)
                acc = outs if acc is None else [a.maximum(o) for a, o in zip(acc, outs)]
            # The branch taken depends on the predicate, so if the predicate is
            # itself input-dependent every output element inherits that dependence.
            pred_dep = None if _is_empty(pred_C) else pred_C.max(axis=0).tocsr()
            for v, C in zip(eqn.outvars, acc):
                if pred_dep is not None:
                    ones = sp.csr_matrix(np.ones((C.shape[0], 1), dtype=np.int8))
                    C = C + ones @ pred_dep
                env[v] = C
            continue

        if prim in _ELEMENTWISE:
            out_shape = out_shapes[0]
            out_size = out_sizes[0]
            acc = _empty(out_size, n_global)
            for iv, C in zip(eqn.invars, in_Cs):
                if _is_empty(C):
                    continue
                s = shape_of(iv)
                if tuple(s) == tuple(out_shape):
                    contrib = C
                else:
                    bpos = np.broadcast_to(
                        np.arange(int(np.prod(s)) or 1).reshape(s), out_shape
                    ).reshape(-1)
                    contrib = C[bpos]
                acc = acc + contrib
            # Zero-skip: rows multiplied by a structurally-zero constant carry no
            # dependence; drop them so they don't inflate the colouring.
            if prim == "mul":
                zmask = _mul_zero_positions(
                    [shape_of(v) for v in eqn.invars],
                    [val_of(v) for v in eqn.invars],
                    out_shape,
                )
                if zmask is not None and zmask.any():
                    keep = sp.diags(
                        (~zmask).astype(np.int8), format="csr", dtype=np.int8
                    )
                    acc = (keep @ acc).tocsr()
                    acc.eliminate_zeros()
            env[eqn.outvars[0]] = acc
            continue

        if prim.startswith("reduce_") and "axes" in eqn.params:
            (C,) = in_Cs
            in_shape = shape_of(eqn.invars[0])
            out_flat, out_size = _reduce_map(eqn.params, in_shape, out_shapes[0])
            op_size = C.shape[0]
            M = sp.csr_matrix(
                (np.ones(op_size, dtype=np.int8), (out_flat, np.arange(op_size))),
                shape=(out_size, op_size),
            )
            env[eqn.outvars[0]] = M @ C
            continue

        if prim in ("scatter-add", "scatter", "add_jaxvals"):
            idx_val = val_of(eqn.invars[1])
            if idx_val is _UNKNOWN:
                raise _BailOut("scatter with data-dependent indices")
            env[eqn.outvars[0]] = _scatter(
                eqn, in_Cs, np.asarray(idx_val), combine=prim != "scatter"
            )
            continue

        if prim == "dynamic_update_slice":
            env[eqn.outvars[0]] = _dynamic_update_slice(
                eqn, in_Cs, [val_of(v) for v in eqn.invars]
            )
            continue

        if prim == "conv_general_dilated":
            env[eqn.outvars[0]] = _conv(eqn, in_Cs, val_of, out_sizes[0], n_global)
            continue

        if prim == "dot_general":
            env[eqn.outvars[0]] = _dot_general(eqn, in_Cs, val_of, n_global)
            continue

        if prim == "scan":
            outs = _scan(eqn, in_Cs, [val_of(v) for v in eqn.invars], n_global)
            for v, C in zip(eqn.outvars, outs):
                env[v] = C
            continue

        # Movement primitives (single output).
        try:
            in_shapes = [shape_of(v) for v in eqn.invars]
            in_vals = [val_of(v) for v in eqn.invars]
            rowmap, n_comb = _movement_rowmap(prim, eqn.params, in_shapes, in_vals)
            combined = sp.vstack(in_Cs, format="csr") if len(in_Cs) > 1 else in_Cs[0]
            if combined.shape[0] != n_comb:
                raise _BailOut(f"shape mismatch in {prim}")
            env[eqn.outvars[0]] = combined[rowmap]
            continue
        except _BailOut:
            raise
        except Exception as e:
            raise _BailOut(f"unhandled primitive {prim}: {e}")

    return [read(v) for v in jaxpr.outvars]


def _dynamic_update_slice(eqn: JaxprEqn, in_Cs: list[Conn], in_vals: list[Any]) -> Conn:
    """Connectivity of dynamic_update_slice(operand, update, *starts): the update
    block replaces a window of the operand, so window positions take the update's
    connectivity and the rest keep the operand's. Static starts only."""
    operand_C, update_C = in_Cs[0], in_Cs[1]
    operand_shape = tuple(eqn.invars[0].aval.shape)
    update_shape = tuple(eqn.invars[1].aval.shape)
    if any(s is _UNKNOWN for s in in_vals[2:]):
        raise _BailOut("dynamic_update_slice with data-dependent start")
    # lax clamps each start so the window fits inside the operand.
    starts = [
        max(0, min(int(s), d - u))
        for s, d, u in zip(in_vals[2:], operand_shape, update_shape)
    ]
    op_size = int(np.prod(operand_shape)) or 1
    upd_size = int(np.prod(update_shape)) or 1
    window = tuple(slice(s, s + u) for s, u in zip(starts, update_shape))
    window_flat = np.arange(op_size).reshape(operand_shape)[window].reshape(-1)
    # Update block -> its window positions; operand kept outside the window.
    M = sp.csr_matrix(
        (np.ones(upd_size, dtype=np.int8), (window_flat, np.arange(upd_size))),
        shape=(op_size, upd_size),
    )
    keep = np.ones(op_size, dtype=np.int8)
    keep[window_flat] = 0
    D = sp.diags(keep, format="csr", dtype=np.int8)
    return (D @ operand_C + M @ update_C).tocsr()


def _scatter_targets(
    idx_val: np.ndarray,
    eqn: JaxprEqn,
    operand_shape: tuple[int, ...],
    updates_shape: tuple[int, ...],
) -> np.ndarray:
    """Forward map ``update_flat -> operand_flat`` for a scatter with static indices,
    recovering EVERY (operand, update) pair so overlapping updates are not lost.
    Out-of-bounds updates map to -1 (dropped). Bails on batched dimension numbers
    or CLIP mode (both uncommon)."""
    dnums = eqn.params["dimension_numbers"]
    if getattr(dnums, "operand_batching_dims", ()) or getattr(
        dnums, "scatter_indices_batching_dims", ()
    ):
        raise _BailOut("batched scatter not traced")
    mode = eqn.params.get("mode")
    if mode is not None and "CLIP" in str(mode):
        raise _BailOut("scatter CLIP mode not traced")

    uwd = tuple(dnums.update_window_dims)
    iwd = tuple(dnums.inserted_window_dims)
    sdod = tuple(dnums.scatter_dims_to_operand_dims)
    op_ndim = len(operand_shape)
    upd_ndim = len(updates_shape)
    index_vectors = idx_val.reshape(-1, idx_val.shape[-1])  # (n_scatter, index_depth)

    # Scalar scatter (no window dims): one update element per scatter index, in
    # row-major order -> fully vectorised.
    if not uwd:
        operand_multi = np.zeros((index_vectors.shape[0], op_ndim), dtype=np.int64)
        for k, d in enumerate(sdod):
            operand_multi[:, d] = index_vectors[:, k]
        oob = np.zeros(index_vectors.shape[0], dtype=bool)
        for d in range(op_ndim):
            oob |= (operand_multi[:, d] < 0) | (operand_multi[:, d] >= operand_shape[d])
        target = np.full(index_vectors.shape[0], -1, dtype=np.int64)
        target[~oob] = np.ravel_multi_index(
            [operand_multi[~oob, d] for d in range(op_ndim)], operand_shape
        )
        return target

    # Window scatter: walk each (scatter index, window offset) pair.
    batch_dims = [d for d in range(upd_ndim) if d not in uwd]
    batch_shape = tuple(updates_shape[d] for d in batch_dims)
    window_operand_dims = [d for d in range(op_ndim) if d not in iwd]
    window_shape = tuple(updates_shape[d] for d in uwd)
    target = np.full(int(np.prod(updates_shape)) or 1, -1, dtype=np.int64)
    for b, index_vector in enumerate(index_vectors):
        batch_coords = np.unravel_index(b, batch_shape) if batch_shape else ()
        start = [0] * op_ndim
        for k, d in enumerate(sdod):
            start[d] = int(index_vector[k])
        for win_idx in np.ndindex(*window_shape) if window_shape else [()]:
            operand_idx = list(start)
            for j, d in enumerate(window_operand_dims):
                operand_idx[d] += win_idx[j]
            if any(
                operand_idx[d] < 0 or operand_idx[d] >= operand_shape[d]
                for d in range(op_ndim)
            ):
                continue
            upd_multi = [0] * upd_ndim
            for i, d in enumerate(batch_dims):
                upd_multi[d] = int(batch_coords[i])
            for j, d in enumerate(uwd):
                upd_multi[d] = win_idx[j]
            target[int(np.ravel_multi_index(upd_multi, updates_shape))] = int(
                np.ravel_multi_index(operand_idx, operand_shape)
            )
    return target


def _scatter(
    eqn: JaxprEqn, in_Cs: list[Conn], idx_val: np.ndarray, combine: bool
) -> Conn:
    """Connectivity of a scatter with static indices, from the full update->operand
    forward map so overlapping updates are all captured. ``combine`` (scatter-add/...):
    written positions depend on the operand AND every update landing there. Plain
    scatter (replace): written positions depend only on the updates.
    """
    operand_C, _idx_C, updates_C = in_Cs
    operand_shape = tuple(eqn.invars[0].aval.shape)
    updates_shape = tuple(eqn.invars[2].aval.shape)
    op_size = int(np.prod(operand_shape)) or 1
    upd_size = int(np.prod(updates_shape)) or 1
    target = _scatter_targets(idx_val, eqn, operand_shape, updates_shape)
    has = target >= 0
    # update u -> operand target[u]; overlap = several updates in one operand row.
    M = sp.csr_matrix(
        (np.ones(int(has.sum()), dtype=np.int8), (target[has], np.nonzero(has)[0])),
        shape=(op_size, upd_size),
    )
    if combine:
        return (operand_C + M @ updates_C).tocsr()
    # Replace: written positions lose their operand dependence.
    keep = np.ones(op_size, dtype=np.int8)
    keep[target[has]] = 0
    D = sp.diags(keep, format="csr", dtype=np.int8)
    return (D @ operand_C + M @ updates_C).tocsr()


def _conv(
    eqn: JaxprEqn,
    in_Cs: list[Conn],
    val_of: Callable[[Var | Literal], Any],
    out_size: int,
    n_global: int,
) -> Conn:
    """Connectivity of a convolution with a constant kernel: each output element
    depends on the union of input elements in its receptive field. Computed by
    probing each nonzero kernel tap with a 1-hot kernel to get the shift map
    (1-based positions; 0 marks padding), so it is exact for any stride/pad/
    dilation. Bails if both operands are input-dependent or the kernel is unknown.
    """
    lhs_C, rhs_C = in_Cs
    lhs_v, rhs_v = eqn.invars
    lhs_empty, rhs_empty = _is_empty(lhs_C), _is_empty(rhs_C)
    if lhs_empty == rhs_empty:
        raise _BailOut("conv needs exactly one constant operand")
    if rhs_empty:  # standard case: lhs=data, rhs=kernel
        data_C, data_shape, data_is_lhs = lhs_C, tuple(lhs_v.aval.shape), True
        kernel = val_of(rhs_v)
        kernel_shape = tuple(rhs_v.aval.shape)
    else:
        data_C, data_shape, data_is_lhs = rhs_C, tuple(rhs_v.aval.shape), False
        kernel = val_of(lhs_v)
        kernel_shape = tuple(lhs_v.aval.shape)
    if kernel is _UNKNOWN:
        raise _BailOut("conv with non-constant kernel")
    kernel = np.asarray(kernel)
    data_size = int(np.prod(data_shape)) or 1
    # 1-based float positions (exact in float64 for realistic sizes); 0 = padding.
    pos = (np.arange(data_size, dtype=np.float64) + 1.0).reshape(data_shape)
    taps = np.argwhere(np.abs(kernel) > 0)
    acc = _empty(out_size, n_global)
    with _host_exact():
        pos_j = jnp.asarray(pos)
        for tap in taps:
            onehot = np.zeros(kernel_shape, dtype=np.float64)
            onehot[tuple(tap)] = 1.0
            oh = jnp.asarray(onehot)
            args = (pos_j, oh) if data_is_lhs else (oh, pos_j)
            m = np.asarray(lax.conv_general_dilated_p.bind(*args, **eqn.params))
            m = np.rint(m.reshape(-1)).astype(np.int64)
            valid = m > 0
            if not valid.any():
                continue
            rows_out = np.nonzero(valid)[0]
            cols_in = m[valid] - 1
            M = sp.csr_matrix(
                (np.ones(rows_out.size, dtype=np.int8), (rows_out, cols_in)),
                shape=(out_size, data_size),
            )
            acc = acc + M @ data_C
    return acc


def _scan(
    eqn: JaxprEqn, in_Cs: list[Conn], in_vals: list[Any], n_global: int
) -> list[Conn]:
    """Connectivity through a scan, by unrolling: thread the carry connectivity
    step by step and stack the per-step outputs. Bails on very long scans.
    """
    p = eqn.params
    body = p["jaxpr"]
    body_jaxpr, body_consts = getattr(body, "jaxpr", body), getattr(body, "consts", [])
    nconsts, ncarry, length = p["num_consts"], p["num_carry"], p["length"]
    reverse = p.get("reverse", False)
    if length > 4096:
        raise _BailOut("scan too long to unroll")

    consts_C = list(in_Cs[:nconsts])
    carry_C = list(in_Cs[nconsts : nconsts + ncarry])
    xs_C = list(in_Cs[nconsts + ncarry :])
    consts_V = list(in_vals[:nconsts])
    xs_V = list(in_vals[nconsts + ncarry :])
    xs_vars = eqn.invars[nconsts + ncarry :]
    x_rest = [int(np.prod(v.aval.shape[1:])) or 1 for v in xs_vars]

    nys = len(body_jaxpr.outvars) - ncarry
    y_steps = [[None] * length for _ in range(nys)]
    order = range(length - 1, -1, -1) if reverse else range(length)
    for t in order:
        x_slice_C = [
            xs_C[i][t * x_rest[i] : (t + 1) * x_rest[i]] for i in range(len(xs_C))
        ]
        x_slice_V = [(_UNKNOWN if v is _UNKNOWN else np.asarray(v)[t]) for v in xs_V]
        args_C = consts_C + carry_C + x_slice_C
        args_V = consts_V + [_UNKNOWN] * ncarry + x_slice_V
        outs = _interp(body_jaxpr, body_consts, n_global, args_C, args_V)
        carry_C = list(outs[:ncarry])
        for j in range(nys):
            y_steps[j][t] = outs[ncarry + j]

    result = list(carry_C)
    for j in range(nys):
        result.append(
            sp.vstack(y_steps[j], format="csr") if length else _empty(0, n_global)
        )
    return result


def _dot_general(
    eqn: JaxprEqn,
    in_Cs: list[Conn],
    val_of: Callable[[Var | Literal], Any],
    n_global: int,
) -> Conn:
    """Connectivity of a contraction (dot_general / matmul / einsum) with one
    constant operand: each output element depends on the union of the data
    operand's elements over the contracted fibre where the constant is nonzero.
    Built directly from the constant's nonzero structure. Bails if both operands
    are input-dependent, the constant is unknown, or the dense edge count is huge.
    """
    lhs_C, rhs_C = in_Cs
    lhs_v, rhs_v = eqn.invars
    (lc, rc), (lb, rb) = eqn.params["dimension_numbers"]
    lc, rc, lb, rb = tuple(lc), tuple(rc), tuple(lb), tuple(rb)
    le, re_ = _is_empty(lhs_C), _is_empty(rhs_C)
    if le == re_:
        raise _BailOut("dot_general needs exactly one constant operand")
    if re_:  # lhs is data, rhs is constant
        data_C, ds = lhs_C, tuple(lhs_v.aval.shape)
        d_contract, d_batch = lc, lb
        const, cs, c_contract, c_batch = val_of(rhs_v), tuple(rhs_v.aval.shape), rc, rb
        data_is_lhs = True
    else:
        data_C, ds = rhs_C, tuple(rhs_v.aval.shape)
        d_contract, d_batch = rc, rb
        const, cs, c_contract, c_batch = val_of(lhs_v), tuple(lhs_v.aval.shape), lc, lb
        data_is_lhs = False
    if const is _UNKNOWN:
        raise _BailOut("dot_general with non-constant operand")
    const = np.asarray(const)

    d_free = [d for d in range(len(ds)) if d not in d_contract and d not in d_batch]
    c_free = [d for d in range(len(cs)) if d not in c_contract and d not in c_batch]
    out_shape = tuple(eqn.outvars[0].aval.shape)
    out_size = int(np.prod(out_shape)) or 1
    data_size = int(np.prod(ds)) or 1
    dfree_size = int(np.prod([ds[d] for d in d_free])) or 1

    nz = np.argwhere(np.abs(const) > 0)  # (cnnz, len(cs))
    cnnz = nz.shape[0]
    if cnnz == 0:
        return _empty(out_size, n_global)
    if cnnz * dfree_size > 50_000_000:
        raise _BailOut("dot_general too large (dense)")

    def _cols(arr: np.ndarray, dims: tuple[int, ...]) -> np.ndarray:
        return arr[:, list(dims)] if dims else np.zeros((arr.shape[0], 0), np.int64)

    cb_idx, cc_idx, cf_idx = (
        _cols(nz, c_batch),
        _cols(nz, c_contract),
        _cols(nz, c_free),
    )
    dfree_shape = [ds[d] for d in d_free]
    dfree_grid = (
        np.array(list(np.ndindex(*dfree_shape)), dtype=np.int64).reshape(
            dfree_size, len(d_free)
        )
        if d_free
        else np.zeros((1, 0), np.int64)
    )

    R, S = cnnz, dfree_size
    rr = np.repeat(np.arange(R), S)
    ss = np.tile(np.arange(S), R)

    data_multi = np.zeros((R * S, len(ds)), dtype=np.int64)
    for j, d in enumerate(d_batch):
        data_multi[:, d] = cb_idx[rr, j]
    for j, d in enumerate(d_contract):
        data_multi[:, d] = cc_idx[rr, j]
    for j, d in enumerate(d_free):
        data_multi[:, d] = dfree_grid[ss, j]
    data_flat = (
        np.ravel_multi_index([data_multi[:, d] for d in range(len(ds))], ds)
        if ds
        else np.zeros(R * S, np.int64)
    )

    lhs_free = dfree_grid[ss] if data_is_lhs else cf_idx[rr]
    rhs_free = cf_idx[rr] if data_is_lhs else dfree_grid[ss]
    out_multi = np.concatenate([cb_idx[rr], lhs_free, rhs_free], axis=1)
    out_flat = (
        np.ravel_multi_index(
            [out_multi[:, k] for k in range(out_multi.shape[1])], out_shape
        )
        if out_multi.shape[1]
        else np.zeros(R * S, np.int64)
    )

    M = sp.csr_matrix(
        (np.ones(out_flat.size, dtype=np.int8), (out_flat, data_flat)),
        shape=(out_size, data_size),
    )
    return M @ data_C


def trace_sparsity_pattern(
    operator: Callable, shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray] | None:
    """Recover (rows, cols) of a JAX operator's sparsity, or None to fall back.

    Traces the operator to a jaxpr and propagates a connectivity (index-set)
    structure through each primitive, recovering the exact *global* sparsity
    pattern in one trace -- the JAX-native analogue of SparseConnectivityTracer.jl
    (Hill & Dalle, arXiv:2501.17737, 2025); see the module docstring.

    Args:
        operator: callable ``A(x)`` mapping a length-``n_global`` vector to a
            length-``n_local`` vector (the local row block for distributed use).
        shape: ``(n_local, n_global)``.

    Returns:
        ``(rows, cols)`` int32 arrays of the local block's nonzero pattern, or
        ``None`` if the operator cannot be traced structurally (opaque or
        unsupported primitive, data-dependent indexing, etc.).
    """
    n_local, n_global = shape
    try:
        closed = jax.make_jaxpr(operator)(jnp.ones(n_global))
    except Exception:
        return None
    jaxpr = closed.jaxpr
    if len(jaxpr.invars) != 1:
        return None
    seed = sp.identity(n_global, dtype=np.int8, format="csr")
    try:
        outs = _interp(jaxpr, closed.consts, n_global, [seed])
    except _BailOut:
        return None
    except Exception:
        return None
    if len(outs) != 1:
        return None
    out_C = outs[0].tocoo()
    if out_C.shape != (n_local, n_global):
        return None
    order = np.lexsort((out_C.col, out_C.row))
    return out_C.row[order].astype(np.int32), out_C.col[order].astype(np.int32)


# --- Probing-based detection: exhaustive one-hot basis-vector probing ---
def _probe_columns(
    A_callable: Callable, shape: tuple[int, int], tol: float
) -> tuple[np.ndarray, np.ndarray]:
    """Exhaustive probing with one-hot basis vectors (correct for any operator).

    Probes columns in batches and extracts the non-zeros per block (the full
    (m, n) matrix is never assembled), halving the batch on OOM. O(m) probes.
    """
    n, m = shape

    # Batch the probes with vmap when the operator supports it; fall back to a
    # sequential lax.map for operators that have no vmap rule (e.g. pure_callback
    # / FFI), matching materialize_sparse_matrix. Decide once via a cheap
    # eval_shape, which trips the missing-vmap-rule error without executing.
    try:
        jax.eval_shape(jax.vmap(A_callable), jax.ShapeDtypeStruct((1, m), jnp.float32))
        batched_A = jax.vmap(A_callable)
    except Exception:

        def batched_A(basis):
            return jax.lax.map(A_callable, basis)

    def _eval_batch(start: int, size: int) -> tuple[np.ndarray, np.ndarray]:
        indices = jnp.arange(start, start + size)
        basis = jax.nn.one_hot(indices, m, dtype=jnp.float32)  # (size, m)
        out = batched_A(basis)  # (size, n)
        if out.shape != (size, n):
            raise ValueError(
                f"Operator returned shape {out.shape}, expected ({size}, {n})."
            )
        # out[c, i] = A(e_{start+c})[i] = A[i, start + c].
        col_local, row = np.where(np.abs(np.array(out)) > tol)
        return row.astype(np.int32), (start + col_local).astype(np.int32)

    def _run(batch_size: int) -> tuple[np.ndarray, np.ndarray]:
        blocks = [
            _eval_batch(start, min(batch_size, m - start))
            for start in range(0, m, batch_size)
        ]
        rows = np.concatenate([b[0] for b in blocks])
        cols = np.concatenate([b[1] for b in blocks])
        return rows, cols

    def _is_oom(e: Exception) -> bool:
        s = str(e).lower()
        return "resource exhausted" in s or "out of memory" in s or "oom" in s

    batch_size = m
    result: tuple[np.ndarray, np.ndarray] | None = None
    while result is None and batch_size >= 1:
        try:
            result = _run(batch_size)
        except Exception as e:
            if _is_oom(e):
                batch_size //= 2
                if batch_size >= 1:
                    warnings.warn(
                        f"OOM in probe_sparsity_pattern; retrying with batch_size={batch_size}.",
                        stacklevel=2,
                    )
            else:
                raise

    if result is None:
        raise RuntimeError("OOM even with batch_size=1; operator may be too large.")
    return result


def probe_sparsity_pattern(
    A_callable: Callable,
    shape: tuple[int, int],
    tol: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray]:
    """Determine the sparsity pattern of a linear operator by one-hot probing.

    Probes the operator with batches of one-hot basis vectors and extracts the
    nonzeros per block (the full (m, n) matrix is never assembled), halving the
    batch on OOM. Correct for any operator; this is the fallback used when
    jaxpr tracing is unavailable (opaque or data-dependent operators).

    Must be run outside of JIT compilation. Returns (rows, cols).
    """
    n, m = shape
    if n == 0 or m == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32)
    return _probe_columns(A_callable, shape, tol)


# --- Column coloring and value materialization ---
def get_column_coloring(
    rows: np.ndarray, cols: np.ndarray, shape: tuple[int, int]
) -> tuple[np.ndarray, int]:
    """
    Compute a coloring of the columns such that no two columns with the same color
    share a non-zero row, enabling simultaneous evaluation.

    Builds the column conflict graph via a sparse A^T A product (replaces the O(nnz²)
    Python loop), then runs the Jones-Plassman parallel greedy coloring: each round
    selects a maximal independent set (MIS) using a vectorized JAX scatter-max over
    random weights, assigns the current color to the whole MIS, and repeats.

    Returns:
        colors: array of shape (m,) where colors[j] is the color ID of column j.
        n_colors: total number of colors used.
    """
    n, m = shape

    if len(rows) == 0:
        return np.full(m, -1, dtype=np.int32), 0

    # Build column conflict graph: ATA[c1, c2] > 0 iff columns c1 and c2 share a row.
    ones = np.ones(len(rows), dtype=np.float32)
    A_bool = sp.csr_matrix((ones, (rows, cols)), shape=(n, m))
    ATA = (A_bool.T @ A_bool).tocsr()
    ATA.setdiag(0)
    ATA.eliminate_zeros()

    # Flat edge list: edge k goes from src_nodes[k] to dst_nodes[k].
    # Precomputed once; used every round for the scatter-max.
    nnz_per_col = np.diff(ATA.indptr)
    src_nodes = jnp.array(np.repeat(np.arange(m), nnz_per_col), dtype=jnp.int32)
    dst_nodes = jnp.array(ATA.indices, dtype=jnp.int32)
    has_edges = len(src_nodes) > 0

    # Jones-Plassman coloring
    # Each round: assign random weights, find MIS (nodes whose weight beats every
    # uncolored neighbor), color MIS with the current color, mark them done.
    colors = np.full(m, -1, dtype=np.int32)
    in_pattern = np.zeros(m, dtype=bool)
    in_pattern[np.unique(cols)] = True
    uncolored = in_pattern.copy()  # numpy mask; updated each round

    key = jax.random.PRNGKey(0)
    color_id = 0

    while uncolored.any():
        key, subkey = jax.random.split(key)
        # Weights in (0, 1] for uncolored nodes; 0 for already-colored nodes so
        # they can never dominate an uncolored neighbor in the max comparison.
        w = jax.random.uniform(subkey, (m,), minval=1e-7, maxval=1.0)
        w = w * jnp.array(uncolored, dtype=jnp.float32)

        # neighbor_max[c] = max weight among all neighbors of c.
        # Scatter-max over the flat edge list: O(nnz), no Python loops.
        if has_edges:
            neighbor_max = jnp.full(m, -jnp.inf).at[src_nodes].max(w[dst_nodes])
        else:
            neighbor_max = jnp.full(m, -jnp.inf)

        # MIS: uncolored nodes that beat every neighbor → valid independent set.
        mis_np = np.array(jnp.array(uncolored) & (w > neighbor_max))

        colors[mis_np] = color_id
        uncolored[mis_np] = False
        color_id += 1

    return colors, color_id


def materialize_sparse_matrix(
    A_callable: Callable,
    shape: tuple[int, int],
    rows: ArrayLike,
    cols: ArrayLike,
    column_colors: ArrayLike,
    n_colors: int,
) -> jsp.BCSR:
    """
    Materialize the values of a sparse matrix inside JIT using graph coloring.

    This reduces the number of operator evaluations from N (columns) to C (colors).

    Args:
        A_callable: The function A(x) -> y. Can be differentiated through.
        shape: (n, m)
        rows, cols: Fixed sparsity pattern indices (JAX or Numpy arrays).
        column_colors: Array mapping column index to color ID.
        n_colors: Number of colors.

    Returns:
        A_bcsr: jax.experimental.sparse.BCSR matrix containing the values from A_callable.
    """
    n, m = shape

    # The sparsity pattern (rows/cols/colours) is static -- known at cache time.
    # Compute the CSR ordering and row pointers on the host with NumPy so XLA
    # receives them as ready constants, instead of constant-folding a large
    # lexsort inside JIT (which dominates compile time at scale). Only the values
    # (the operator evaluations) stay traced. Falls back to the JAX path if the
    # indices arrive as tracers (not the normal case).
    try:
        rows_np = np.asarray(rows).astype(np.int32)
        cols_np = np.asarray(cols).astype(np.int32)
        colors_np = np.asarray(column_colors).astype(np.int32)
        static = True
    except Exception:
        static = False

    column_colors = jnp.array(column_colors, dtype=jnp.int32)

    def evaluate_color(color_id: ArrayLike) -> jax.Array:
        # Create probe vector v_c such that v_c[j] = 1 if color[j] == c, else 0
        mask = column_colors == color_id
        v = mask.astype(jnp.float32)
        w = A_callable(v)
        return w

    # Map over all colors: (n_colors, n)
    # Use lax.map instead of vmap to support primitives without batching rules (e.g. CSR matvec)
    w_matrix = jax.lax.map(evaluate_color, jnp.arange(n_colors))

    if static:
        # Host-side static CSR construction; only `values_sorted` is traced.
        order = np.lexsort((cols_np, rows_np))
        rows_sorted = rows_np[order]
        cols_sorted_np = cols_np[order]
        colors_for_cols_sorted = colors_np[cols_sorted_np]
        indptr_np = np.zeros(int(n) + 1, dtype=np.int32)
        indptr_np[1:] = np.cumsum(np.bincount(rows_sorted, minlength=int(n)))
        values_sorted = w_matrix[
            jnp.asarray(colors_for_cols_sorted), jnp.asarray(rows_sorted)
        ]
        return jsp.BCSR(
            (values_sorted, jnp.asarray(cols_sorted_np), jnp.asarray(indptr_np)),
            shape=shape,
        )

    # Fallback: indices are traced -> do the sort in JAX.
    rows = jnp.array(rows, dtype=jnp.int32)
    cols = jnp.array(cols, dtype=jnp.int32)
    colors_for_cols = column_colors[cols]
    values = w_matrix[colors_for_cols, rows]
    sort_idx = jnp.lexsort((cols, rows))
    cols_sorted = cols[sort_idx]
    values_sorted = values[sort_idx]
    indptr = jnp.zeros(int(n) + 1, dtype=jnp.int32)
    row_counts = jnp.bincount(rows[sort_idx], length=n)
    indptr = indptr.at[1:].set(jnp.cumsum(row_counts).astype(jnp.int32))
    return jsp.BCSR((values_sorted, cols_sorted, indptr), shape=shape)


# --- Verification and orchestration (cache_coloring) ---
def _drop_zeros(
    A_bcsr: jsp.BCSR, tol: float = 1e-9
) -> tuple[jsp.BCSR, np.ndarray, np.ndarray]:
    """Return (BCSR, rows, cols) with near-zero entries removed."""
    data = np.asarray(A_bcsr.data)
    indices = np.asarray(A_bcsr.indices)
    indptr = np.asarray(A_bcsr.indptr)
    n_rows = A_bcsr.shape[0]
    keep = np.abs(data) > tol
    row_of = np.repeat(np.arange(n_rows, dtype=np.int32), np.diff(indptr))
    new_indptr = np.zeros(n_rows + 1, dtype=np.int32)
    new_indptr[1:] = np.cumsum(np.bincount(row_of[keep], minlength=n_rows))
    A = jsp.BCSR(
        (
            jnp.asarray(data[keep]),
            jnp.asarray(indices[keep], dtype=jnp.int32),
            jnp.asarray(new_indptr),
        ),
        shape=A_bcsr.shape,
    )
    return A, row_of[keep], indices[keep].astype(np.int32)


def _verify_recovery(
    operator: Callable, A_bcsr: jsp.BCSR, n_global: int, n_check: int = 5
) -> bool:
    """Check the recovered matrix reproduces the operator on random vectors.

    If A_bcsr != A (entries missing because the operator was not really
    translation-invariant, or boundary couplings were not captured) then
    (A_bcsr - A) v != 0 for almost every v, so a few random probes catch it.

    Probes follow the configured precision -- float64 when x64 is enabled (as in
    the distributed solvers), float32 otherwise -- with the tolerance loosened to
    match, so it neither warns about an unavailable dtype nor false-rejects a
    correct float32 recovery.
    """
    x64 = jax.config.jax_enable_x64
    dtype = jnp.float64 if x64 else jnp.float32
    tol = 1e-6 if x64 else 1e-4
    key = jax.random.PRNGKey(0)
    for _ in range(n_check):
        key, sub = jax.random.split(key)
        v = jax.random.normal(sub, (n_global,), dtype=dtype)
        y_op = np.asarray(operator(v))
        y_rec = np.asarray(A_bcsr @ v)
        if np.linalg.norm(y_rec - y_op) > tol * (np.linalg.norm(y_op) + 1e-30):
            return False
    return True


def _try_trace_coloring(
    operator: Callable, n_local: int, n_global: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, tuple[int, int]] | None:
    """Exact sparsity from the operator's jaxpr (no probing), VERIFIED against the
    operator. Works for any JAX-expressed operator; returns None for operators
    that can't be traced structurally (opaque calls, data-dependent indexing),
    so the caller falls back to the probing detector.
    """
    try:
        pattern = trace_sparsity_pattern(operator, (n_local, n_global))
        if pattern is None:
            return None
        rows, cols = pattern
        if rows.size == 0:
            return None
        column_colors, n_colors = get_column_coloring(rows, cols, (n_local, n_global))
        A = materialize_sparse_matrix(
            operator, (n_local, n_global), rows, cols, column_colors, n_colors
        )
        # The pattern is exact; drop_zeros only removes structurally-present but
        # numerically-zero entries (e.g. a vanishing variable coefficient), and
        # verify is the safety net in case a transfer rule is wrong.
        A, final_rows, final_cols = _drop_zeros(A)
        if not _verify_recovery(operator, A, n_global):
            return None
        return (final_rows, final_cols, column_colors, n_colors, (n_local, n_global))
    except Exception:
        return None  # any failure -> probing fallback (correctness preserved)


def cache_coloring(
    operator: Any,
    shape: tuple[int, int] | int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, tuple[int, int]]:
    """
    Compute and cache coloring information for a callable operator.

    Detection uses two methods, so the result is correct for ANY operator:

    1. Tracing: interpret the operator's jaxpr to recover the EXACT sparsity in
       a single trace (no probing), then colour and materialise it. Works for any
       JAX-expressed operator; skipped for operators that can't be traced
       structurally (opaque calls, data-dependent indexing).
    2. Probing (``probe_sparsity_pattern`` + ``get_column_coloring``): exhaustive
       one-hot basis-vector probing, correct for any operator -- the fallback when
       tracing is unavailable.

    Args:
        operator: A callable operator A(x) that returns ``A @ x``.
        shape: Shape of the operator (n, m) or int size (for an n×n matrix). For a
            distributed operator this is the local block ``(n_local, n_global)``.

    Returns:
        Cached coloring information for reattachment with ``with_cache(..., coloring=...)``.
    """
    if isinstance(shape, int):
        shape = (shape, shape)

    existing_cache = getattr(operator, "_coloring_info", None)
    if existing_cache is not None:
        cached_shape = existing_cache[4]
        if cached_shape == shape:
            return existing_cache
        raise ValueError(
            f"Operator already has cached coloring for shape {cached_shape}, "
            f"but requested shape {shape}. Create a new operator instance."
        )

    n_local, n_global = shape

    # 1. Tracing (exact, any JAX operator). 2. Probing (any operator). Tracing
    # verifies before being accepted; probing is exact by construction.
    cache = _try_trace_coloring(operator, n_local, n_global)
    if cache is None:
        rows, cols = probe_sparsity_pattern(operator, shape)
        column_colors, n_colors = get_column_coloring(rows, cols, shape)
        cache = (rows, cols, column_colors, n_colors, shape)

    try:
        setattr(operator, "_coloring_info", cache)
    except Exception:
        pass

    return cache
