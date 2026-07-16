"""Tracing-based sparsity detection: interpret an operator's jaxpr and propagate a
connectivity (index-set) structure through each primitive to recover the EXACT
global sparsity pattern in a single trace.

This is the JAX-native analogue of the operator-overloading sparsity detection of
SparseConnectivityTracer.jl [1, 2]: JAX is trace-based rather than dispatch-based,
so instead of overloading a custom number type we trace the function to a jaxpr and
interpret it. ``trace_sparsity_pattern`` returns ``None`` for operators that cannot
be traced structurally (opaque calls, data-dependent indexing); ``sparsity.py`` then
falls back to exhaustive probing.

References:
    [1] A. Hill and G. Dalle, "Sparser, Better, Faster, Stronger: Sparsity
        Detection for Efficient Automatic Differentiation," arXiv:2501.17737 (2025).
    [2] SparseConnectivityTracer.jl,
        https://github.com/adrhill/SparseConnectivityTracer.jl
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator, Sequence
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
from jax import lax
from jax.extend.core import Jaxpr, JaxprEqn, Literal, Var

from .utils import temp_enable_x64

# Private DCE API; tracing degrades gracefully (less pruning) if it ever moves.
_dce_jaxpr: Callable[..., Any] | None
try:
    from jax._src.interpreters.partial_eval import dce_jaxpr as _dce_jaxpr
except Exception:  # pragma: no cover - exercised only if the private API moves
    _dce_jaxpr = None


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

# Single-output movement primitives handled by _movement_rowmap. A primitive
# outside this set (and the dispatch table) is unknown -> conservative fallback.
_MOVEMENT_PRIMS = {
    "reshape",
    "squeeze",
    "expand_dims",
    "transpose",
    "rev",
    "slice",
    "broadcast_in_dim",
    "concatenate",
    "pad",
    "dynamic_slice",
    "gather",
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


def _select_pick(
    case_Cs: list[Conn], which_val: Any, out_shape: tuple[int, ...], out_size: int
) -> Conn | None:
    """Per-position connectivity for a constant-predicate ``select_n``: output ``i``
    takes branch ``which[i]``, so the unselected branches are dropped instead of
    unioned. Returns ``None`` (keep the conservative union) if a case is not
    output-shaped."""
    if any(C.shape[0] != out_size for C in case_Cs):
        return None
    which_flat = np.broadcast_to(np.asarray(which_val), out_shape).reshape(-1)
    result = _empty(out_size, case_Cs[0].shape[1])
    for c, case_C in enumerate(case_Cs):
        mask = which_flat == c
        if mask.any():
            Dc = sp.diags(mask.astype(np.int8), format="csr", dtype=np.int8)
            result = result + Dc @ case_C
    return result.tocsr()


def _conservative_outputs(
    in_Cs: list[Conn], out_sizes: list[int], n_global: int, prim: str
) -> list[Conn]:
    """Superset connectivity for an un-traceable primitive: every output element
    depends on the union of the primitive's input columns. Bails to probing when
    that union is not local (spans more than half the inputs), so a globally-opaque
    operator is still detected exactly by probing rather than over-approximated to
    a near-dense pattern."""
    nonempty = [C for C in in_Cs if C.nnz]
    if not nonempty:
        return [_empty(sz, n_global) for sz in out_sizes]
    stacked = sp.vstack(nonempty, format="csr") if len(nonempty) > 1 else nonempty[0]
    cols = np.unique(stacked.indices)
    if cols.size > n_global // 2:
        raise _BailOut(f"{prim} too non-local to over-approximate")
    outs = []
    for sz in out_sizes:
        rows = np.repeat(np.arange(sz, dtype=np.int64), cols.size)
        data = np.ones(rows.size, dtype=np.int8)
        outs.append(
            sp.csr_matrix((data, (rows, np.tile(cols, sz))), shape=(sz, n_global))
        )
    return outs


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


class _Ctx(NamedTuple):
    """Per-call interpreter context shared with the primitive handlers."""

    n_global: int
    read: Callable[[Any], Conn]
    val_of: Callable[[Any], Any]
    shape_of: Callable[[Any], tuple[int, ...]]


# --- Per-primitive connectivity handlers: (eqn, in_Cs, ctx) -> outvar conns ---


def _h_call(eqn: JaxprEqn, in_Cs: list[Conn], ctx: _Ctx) -> list[Conn]:
    """jit/pjit/custom_*_call: interpret the captured sub-jaxpr recursively."""
    sub = eqn.params.get("jaxpr") or eqn.params.get("call_jaxpr")
    if sub is None:
        raise _BailOut(f"call primitive without jaxpr: {eqn.primitive}")
    sub_jaxpr = getattr(sub, "jaxpr", sub)
    sub_consts = getattr(sub, "consts", [])
    in_vals = [ctx.val_of(v) for v in eqn.invars]
    return _interp(sub_jaxpr, sub_consts, ctx.n_global, in_Cs, in_vals)


def _h_cond(eqn: JaxprEqn, in_Cs: list[Conn], ctx: _Ctx) -> list[Conn]:
    """cond: union over branches (we don't know which is taken); a data-dependent
    predicate makes every output inherit the predicate's dependence."""
    pred_C, branch_conns = in_Cs[0], in_Cs[1:]
    branch_vals = [ctx.val_of(v) for v in eqn.invars[1:]]
    acc = None
    for br in eqn.params["branches"]:
        bj = getattr(br, "jaxpr", br)
        bc = getattr(br, "consts", [])
        outs = _interp(bj, bc, ctx.n_global, branch_conns, branch_vals)
        acc = outs if acc is None else [a.maximum(o) for a, o in zip(acc, outs)]
    pred_dep = None if _is_empty(pred_C) else pred_C.max(axis=0).tocsr()
    result = []
    for C in acc or []:
        if pred_dep is not None:
            ones = sp.csr_matrix(np.ones((C.shape[0], 1), dtype=np.int8))
            C = C + ones @ pred_dep
        result.append(C)
    return result


def _h_elementwise(eqn: JaxprEqn, in_Cs: list[Conn], ctx: _Ctx) -> list[Conn]:
    """Element-wise op: union operands at each (broadcast) position. ``mul`` drops
    rows killed by a structurally-zero constant; a constant ``select_n`` predicate
    picks the selected branch per position instead of unioning."""
    prim = str(eqn.primitive)
    out_shape = ctx.shape_of(eqn.outvars[0])
    out_size = int(np.prod(out_shape)) or 1
    acc = _empty(out_size, ctx.n_global)
    for iv, C in zip(eqn.invars, in_Cs):
        if _is_empty(C):
            continue
        s = ctx.shape_of(iv)
        if tuple(s) == tuple(out_shape):
            contrib = C
        else:
            bpos = np.broadcast_to(
                np.arange(int(np.prod(s)) or 1).reshape(s), out_shape
            ).reshape(-1)
            contrib = C[bpos]
        acc = acc + contrib
    if prim == "mul":
        zmask = _mul_zero_positions(
            [ctx.shape_of(v) for v in eqn.invars],
            [ctx.val_of(v) for v in eqn.invars],
            out_shape,
        )
        if zmask is not None and zmask.any():
            keep = sp.diags((~zmask).astype(np.int8), format="csr", dtype=np.int8)
            acc = (keep @ acc).tocsr()
            acc.eliminate_zeros()
    elif prim in ("select_n", "select"):
        which_val = ctx.val_of(eqn.invars[0])
        if which_val is not _UNKNOWN:
            picked = _select_pick(in_Cs[1:], which_val, out_shape, out_size)
            if picked is not None:
                acc = picked
    return [acc]


def _h_reduce(eqn: JaxprEqn, in_Cs: list[Conn], ctx: _Ctx) -> list[Conn]:
    """reduce_*: each output depends on the union of its reduced input fibre."""
    (C,) = in_Cs
    in_shape = ctx.shape_of(eqn.invars[0])
    out_flat, out_size = _reduce_map(eqn.params, in_shape, ctx.shape_of(eqn.outvars[0]))
    op_size = C.shape[0]
    M = sp.csr_matrix(
        (np.ones(op_size, dtype=np.int8), (out_flat, np.arange(op_size))),
        shape=(out_size, op_size),
    )
    return [M @ C]


def _h_scatter(eqn: JaxprEqn, in_Cs: list[Conn], ctx: _Ctx) -> list[Conn]:
    """scatter / scatter-add: static indices only (data-dependent -> bail)."""
    idx_val = ctx.val_of(eqn.invars[1])
    if idx_val is _UNKNOWN:
        raise _BailOut("scatter with data-dependent indices")
    combine = str(eqn.primitive) != "scatter"
    return [_scatter(eqn, in_Cs, np.asarray(idx_val), combine=combine)]


def _h_dynamic_update_slice(eqn: JaxprEqn, in_Cs: list[Conn], ctx: _Ctx) -> list[Conn]:
    return [_dynamic_update_slice(eqn, in_Cs, [ctx.val_of(v) for v in eqn.invars])]


def _h_conv(eqn: JaxprEqn, in_Cs: list[Conn], ctx: _Ctx) -> list[Conn]:
    out_size = int(np.prod(ctx.shape_of(eqn.outvars[0]))) or 1
    return [_conv(eqn, in_Cs, ctx.val_of, out_size, ctx.n_global)]


def _h_dot_general(eqn: JaxprEqn, in_Cs: list[Conn], ctx: _Ctx) -> list[Conn]:
    return [_dot_general(eqn, in_Cs, ctx.val_of, ctx.n_global)]


def _h_scan(eqn: JaxprEqn, in_Cs: list[Conn], ctx: _Ctx) -> list[Conn]:
    return _scan(eqn, in_Cs, [ctx.val_of(v) for v in eqn.invars], ctx.n_global)


def _h_movement(eqn: JaxprEqn, in_Cs: list[Conn], ctx: _Ctx) -> list[Conn]:
    """Movement primitive: gather each output element from its single source row."""
    prim = str(eqn.primitive)
    try:
        in_shapes = [ctx.shape_of(v) for v in eqn.invars]
        in_vals = [ctx.val_of(v) for v in eqn.invars]
        rowmap, n_comb = _movement_rowmap(prim, eqn.params, in_shapes, in_vals)
        combined = sp.vstack(in_Cs, format="csr") if len(in_Cs) > 1 else in_Cs[0]
        if combined.shape[0] != n_comb:
            raise _BailOut(f"shape mismatch in {prim}")
        return [combined[rowmap]]
    except _BailOut:
        raise
    except Exception as e:
        raise _BailOut(f"movement rule failed for {prim}: {e}")


def _h_conservative(eqn: JaxprEqn, in_Cs: list[Conn], ctx: _Ctx) -> list[Conn]:
    """Opaque or unknown primitive: conservative superset (or bail if non-local)."""
    out_sizes = [int(np.prod(ctx.shape_of(v))) or 1 for v in eqn.outvars]
    return _conservative_outputs(in_Cs, out_sizes, ctx.n_global, str(eqn.primitive))


# prim name -> handler. Reductions match by ``reduce_`` prefix; anything not found
# is an unknown primitive handled conservatively.
_Handler = Callable[[JaxprEqn, list[Conn], _Ctx], list[Conn]]
_DISPATCH: dict[str, _Handler] = {}
for _p in _ELEMENTWISE:
    _DISPATCH[_p] = _h_elementwise
for _p in _CALL_PRIMS:
    _DISPATCH[_p] = _h_call
for _p in _MOVEMENT_PRIMS:
    _DISPATCH[_p] = _h_movement
for _p in _OPAQUE:
    _DISPATCH[_p] = _h_conservative
for _p in ("scatter-add", "scatter", "add_jaxvals"):
    _DISPATCH[_p] = _h_scatter
_DISPATCH["cond"] = _h_cond
_DISPATCH["dynamic_update_slice"] = _h_dynamic_update_slice
_DISPATCH["conv_general_dilated"] = _h_conv
_DISPATCH["dot_general"] = _h_dot_general
_DISPATCH["scan"] = _h_scan


def _dispatch(prim: str, eqn: JaxprEqn) -> _Handler:
    handler = _DISPATCH.get(prim)
    if handler is not None:
        return handler
    if prim.startswith("reduce_") and "axes" in eqn.params:
        return _h_reduce
    return _h_conservative


def _interp(
    jaxpr: Jaxpr,
    consts: Sequence[Any],
    n_global: int,
    arg_conns: list[Conn],
    arg_vals: list[Any] | None = None,
) -> list[Conn]:
    """Interpret one (sub-)jaxpr, returning the connectivity of each outvar.

    A thin driver: it seeds the inputs, constant-folds input-independent vars, then
    dispatches each equation to its per-primitive handler. ``arg_vals`` carries known
    concrete values of the inputs (or ``_UNKNOWN``) so constant-folding (e.g. of
    convolution kernels or scatter indices) works across call boundaries.
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

    ctx = _Ctx(n_global, read, val_of, shape_of)
    for eqn in jaxpr.eqns:
        prim = str(eqn.primitive)
        in_Cs = [read(v) for v in eqn.invars]

        # Constant-fold input-independent vars (e.g. iota-built scatter indices,
        # precomputed grid metrics, convolution kernels) so handlers can resolve
        # them later. Nullary generators like ``iota`` fold too (``all([])`` is
        # True), which is what lets index chains built from ``jnp.arange`` resolve
        # to concrete values for the gather/scatter/dynamic_slice rules.
        if prim not in _OPAQUE:
            in_vals = [val_of(v) for v in eqn.invars]
            if all(x is not _UNKNOWN for x in in_vals):
                try:
                    # Cast each input to the dtype the jaxpr declares for it: some
                    # primitives require an exact dtype match, and a host int literal
                    # would otherwise arrive as int64 and mismatch an int32 operand.
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

        # Shortcut: if no operand carries any input dependence, outputs are empty.
        if all(_is_empty(C) for C in in_Cs):
            for v in eqn.outvars:
                env[v] = _empty(int(np.prod(shape_of(v))) or 1, n_global)
            continue

        for v, C in zip(eqn.outvars, _dispatch(prim, eqn)(eqn, in_Cs, ctx)):
            env[v] = C

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
    if "num_consts" in p:
        nconsts, ncarry = p["num_consts"], p["num_carry"]
    else:
        # jax >= 0.11: the counts moved into the ft_in FlatTree, whose unpacked
        # (consts, carry, xs) groups hold one placeholder per eqn invar. Bail on
        # any forwarding shape where invars are not their plain concatenation.
        groups = p["ft_in"].unpack()
        if any(entry is not None for group in groups for entry in group):
            raise _BailOut("scan with forwarded ft_in entries")
        nconsts, ncarry = len(groups[0]), len(groups[1])
        if nconsts + ncarry + len(groups[2]) != len(eqn.invars):
            raise _BailOut("scan ft_in inconsistent with invars")
    if len(body_jaxpr.outvars) != len(eqn.outvars):
        raise _BailOut("scan body outputs inconsistent with outvars")
    length = p["length"]
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


def _dce(closed: Any) -> tuple[Jaxpr, Sequence[Any]]:
    """Drop equations that don't feed the outputs, so a dead unsupported primitive
    (computed but unused) doesn't force a needless bail. Falls back to the original
    jaxpr if DCE is unavailable or would misalign the captured consts."""
    if _dce_jaxpr is not None:
        try:
            used = [True] * len(closed.jaxpr.outvars)
            new_jaxpr, _ = _dce_jaxpr(closed.jaxpr, used, instantiate=True)
            if len(new_jaxpr.constvars) == len(closed.consts):
                return new_jaxpr, closed.consts
        except Exception:
            pass
    return closed.jaxpr, closed.consts


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
    if len(closed.jaxpr.invars) != 1:
        return None
    # DCE first so a dead unsupported primitive (computed but never used) doesn't
    # force a needless bail to probing.
    jaxpr, consts = _dce(closed)
    seed = sp.identity(n_global, dtype=np.int8, format="csr")
    try:
        outs = _interp(jaxpr, consts, n_global, [seed])
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
