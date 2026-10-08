"""Neighbour-only exchange between ranks, as a linear JAX primitive.

``exchange(buf, plan)`` sends consecutive segments of ``buf`` to the plan's
send peers and returns the received segments in receive-peer order, in one
native call. Its transpose is the reverse exchange, so every derivative order
composes. In per-rank MPI programs it joins mpi4jax's ordered effect; inside
``shard_map`` (``ordered=False``) the common SPMD program fixes the order.
"""

from __future__ import annotations

import dataclasses
import os
import weakref
from collections.abc import Mapping
from operator import index
from typing import TYPE_CHECKING, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax.extend import core
from jax.interpreters import ad, batching, mlir

if TYPE_CHECKING:
    from mpi4py.MPI import Comm

_TARGET = "jaxamg_neighbour_exchange"
_registered = False
# The next plan id free on this process. Plans on subcommunicators advance
# only their members' counters, so each plan takes the largest next id over its
# own communicator: equal on its ranks, unused on each of them.
_next_plan_id = 1
# One duplicated communicator per caller communicator, so the transport's
# messages never match AmgX's or the caller's.
_comm_keyval: int | None = None
# MPI counts are C ints: one message carries at most this many bytes.
_MAX_MESSAGE_BYTES = 2**31 - 1


def _ensure_target() -> None:
    global _registered
    if _registered:
        return
    from ._ext import _amgx

    if not getattr(_amgx, "mpi_enabled", False):
        raise RuntimeError("jaxamg was built without MPI: no neighbour exchange")
    jax.ffi.register_ffi_target(
        _TARGET, _amgx.get_neighbour_exchange_handler(), platform="CUDA"
    )
    _registered = True


def _device_buffers() -> bool:
    """Pass device buffers to MPI (CUDA-aware MPI), else stage them on the
    host: mpi4jax's switch."""
    value = os.environ.get("MPI4JAX_USE_CUDA_MPI", "0")
    return value.strip().lower() in ("1", "true", "yes", "on")


def _free_duplicate(comm: Comm, _keyval: int, duplicate: Comm) -> None:
    # Runs on every rank inside the collective free of ``comm``.
    duplicate.Free()


def transport_comm(comm: Comm) -> Comm:
    """The transport's duplicate of ``comm`` (collective on first use), cached
    as an MPI attribute of ``comm`` and freed with it."""
    global _comm_keyval
    from mpi4py import MPI

    if _comm_keyval is None:
        _comm_keyval = MPI.Comm.Create_keyval(delete_fn=_free_duplicate)
    duplicate = comm.Get_attr(_comm_keyval)
    if duplicate is None:
        duplicate = comm.Dup()
        comm.Set_attr(_comm_keyval, duplicate)
    return cast("Comm", duplicate)


@dataclasses.dataclass(frozen=True)
class NeighbourPlan:
    """This rank's side of one exchange; ``plan_id`` agrees on every rank."""

    plan_id: int
    send_peers: tuple[int, ...]
    send_counts: tuple[int, ...]
    recv_peers: tuple[int, ...]
    recv_counts: tuple[int, ...]
    max_degree: int = 0  # the largest degree over ranks (0: nothing travels)
    max_count: int = 0  # the largest message (elements) over ranks

    @property
    def send_total(self) -> int:
        return int(sum(self.send_counts))

    @property
    def recv_total(self) -> int:
        return int(sum(self.recv_counts))

    def reversed(self) -> NeighbourPlan:
        """The same registered plan read in the reverse direction (roles
        swapped), for size bookkeeping."""
        return dataclasses.replace(
            self,
            send_peers=self.recv_peers,
            send_counts=self.recv_counts,
            recv_peers=self.send_peers,
            recv_counts=self.send_counts,
        )


def make_plan(
    comm: Comm,
    sends: Mapping[int, int],
    recvs: Mapping[int, int],
) -> NeighbourPlan:
    """Register an exchange plan (collective: every rank calls it in the same
    order). ``sends``/``recvs`` map peer rank -> element count; zero counts and
    the rank itself are dropped (local data never travels)."""
    global _next_plan_id
    from mpi4py import MPI

    _ensure_target()
    rank = comm.Get_rank()

    def side(counts: Mapping[int, int]) -> tuple[tuple[int, ...], tuple[int, ...]]:
        entries = [(index(p), index(c)) for p, c in counts.items()]
        if any(p < 0 or p >= comm.Get_size() or c < 0 for p, c in entries):
            raise ValueError
        if sum(c for _, c in entries) > np.iinfo(np.int64).max:
            raise ValueError
        kept = sorted((p, c) for p, c in entries if c and p != rank)
        return tuple(p for p, _ in kept), tuple(c for _, c in kept)

    invalid = False
    try:
        send_peers, send_counts = side(sends)
        recv_peers, recv_counts = side(recvs)
    except (TypeError, ValueError, OverflowError):
        invalid = True
        send_peers = send_counts = recv_peers = recv_counts = ()
    tcomm = transport_comm(comm)
    degree = len(set(send_peers) | set(recv_peers))
    largest = max((*send_counts, *recv_counts), default=0)
    # One O(1) control reduction: the plan id, the largest degree and message.
    bounds = np.array([_next_plan_id, degree, largest, invalid], dtype=np.int64)
    tcomm.Allreduce(MPI.IN_PLACE, bounds, op=MPI.MAX)
    if bounds[3]:
        raise ValueError("exchange peers or counts are invalid on at least one rank")
    plan_id = int(bounds[0])
    _next_plan_id = plan_id + 1
    from ._ext import _amgx

    _amgx.register_exchange_plan(
        plan_id,
        int(MPI._handleof(tcomm)),
        list(send_peers),
        list(send_counts),
        list(recv_peers),
        list(recv_counts),
    )
    plan = NeighbourPlan(
        plan_id,
        send_peers,
        send_counts,
        recv_peers,
        recv_counts,
        int(bounds[1]),
        int(bounds[2]),
    )
    # The native plan lives as long as this object, which the exchanges'
    # traced programs and executables hold (see _lowering).
    from .deferred_release import release_after_device_work

    finalizer = weakref.finalize(
        plan, release_after_device_work, _amgx.release_exchange_plan, plan_id
    )
    # typeshed models this writable property as a field on an empty-slots class.
    finalizer.atexit = False  # type: ignore[misc]
    return plan


def sparse_exchange(
    comm: Comm, outgoing: Mapping[int, np.ndarray], dtype=np.int64
) -> dict[int, np.ndarray]:
    """Send ``outgoing[p]`` to each peer ``p`` and return what every rank sent
    here, keyed by source, discovering the sources without any size-P data:
    synchronous sends, probing, and a nonblocking barrier entered once this
    rank's sends are matched (Hoefler, Siebert and Lumsdaine's NBX)."""
    from mpi4py import MPI

    tcomm = transport_comm(comm)
    tag = 7202
    dtype = np.dtype(dtype)
    mpi_type = MPI._typedict[dtype.char]
    buffers = {
        int(p): np.ascontiguousarray(v, dtype=dtype)
        for p, v in outgoing.items()
        if len(v)
    }
    requests = [
        tcomm.Issend([buf, mpi_type], dest=p, tag=tag) for p, buf in buffers.items()
    ]
    received: dict[int, np.ndarray] = {}
    barrier = None
    status = MPI.Status()
    while True:
        if tcomm.Iprobe(source=MPI.ANY_SOURCE, tag=tag, status=status):
            source, count = status.Get_source(), status.Get_count(mpi_type)
            buf = np.empty(count, dtype=dtype)
            tcomm.Recv([buf, mpi_type], source=source, tag=tag)
            received[source] = buf
        if barrier is None:
            if MPI.Request.Testall(requests):
                barrier = tcomm.Ibarrier()
        elif barrier.Test():
            return received


def owners_of(ids: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """The owning rank of each global row id (``offsets``: first rows, length
    P + 1, the O(P) control metadata)."""
    return (np.searchsorted(offsets, ids, side="right") - 1).astype(np.int64)


exchange_p = core.Primitive("jaxamg_neighbour_exchange")


def mpi_ordered_effect():
    """mpi4jax's ordered effect. Communicating primitives in per-rank programs
    carry it, so their calls and mpi4jax's collectives run in one program
    order on every rank."""
    from mpi4jax._src.utils import ordered_effect

    return ordered_effect


def _impl(x, **params):
    from jax._src import dispatch

    return dispatch.apply_primitive(exchange_p, x, **params)


def _abstract_eval(x, *, plan, reverse, out_size, ordered, device_buffers):
    # ``update`` keeps the input's other type fields (shard_map's varying axes).
    out = x.update(shape=(out_size,), weak_type=False)
    return out, ({mpi_ordered_effect()} if ordered else set())


def _lowering(ctx, x, *, plan, reverse, out_size, ordered, device_buffers):
    from jax._src.interpreters.mlir import custom_call as _custom_call
    from jax._src.lib.mlir.dialects import hlo

    # The plan is a parameter, so traced programs own it; the executable too.
    ctx.module_context.add_keepalive(plan)

    token = ctx.tokens_in.get(mpi_ordered_effect()) if ordered else hlo.create_token()
    call = _custom_call(
        _TARGET,
        result_types=[mlir.aval_to_ir_type(ctx.avals_out[0]), hlo.TokenType.get()],
        operands=[x, token],
        backend_config={
            "plan_id": mlir.ir_attribute(np.int64(plan.plan_id)),
            "reverse": mlir.ir_attribute(np.int32(reverse)),
            "device_buffers": mlir.ir_attribute(np.int32(device_buffers)),
        },
        api_version=4,
        has_side_effect=True,
        operand_layouts=[(0,), ()],
        result_layouts=[(0,), ()],
    )
    out, token_out = call.results
    if ordered:
        ctx.set_tokens_out(mlir.TokenSet({mpi_ordered_effect(): token_out}))
    return [out]


# Symbolic zeros are this rank's knowledge only: another rank may still send
# here, so the rules instantiate them and always take part in the exchange.
def _jvp(primals, tangents, **params):
    (x,), (dx,) = primals, tangents
    out = exchange_p.bind(x, **params)
    return out, exchange_p.bind(ad.instantiate_zeros(dx), **params)


def _transpose(ct, x, *, plan, reverse, out_size, ordered, device_buffers):
    return [
        exchange_p.bind(
            ad.instantiate_zeros(ct),
            plan=plan,
            reverse=not reverse,
            out_size=x.aval.shape[0],
            ordered=ordered,
            device_buffers=device_buffers,
        )
    ]


def _batch(args, dims, **params):
    (x,), (dx,) = args, dims
    x = jnp.moveaxis(x, dx, 0)
    return jax.lax.map(lambda v: exchange_p.bind(v, **params), x), 0


exchange_p.def_impl(_impl)
exchange_p.def_effectful_abstract_eval(_abstract_eval)
mlir.register_lowering(exchange_p, _lowering, platform="cuda")
ad.primitive_jvps[exchange_p] = _jvp
ad.primitive_transposes[exchange_p] = _transpose
batching.primitive_batchers[exchange_p] = _batch


def exchange(
    buf: jax.Array,
    plan: NeighbourPlan,
    *,
    reverse: bool = False,
    out_size: int | None = None,
    ordered: bool = True,
) -> jax.Array:
    """Exchange the packed ``buf`` along ``plan`` (or its reverse).

    ``buf`` holds at least the plan's send total, and at least one entry; the
    result has ``out_size`` entries (default: the receive total, at least one),
    zero past the received ones. ``ordered=False`` is for SPMD programs
    (inside ``shard_map``).

    Every rank must bind the exchange at the same program points, even with
    nothing to send (a constant buffer could be evaluated elsewhere and
    mismatch the messages), so buffers have at least one entry; only a plan
    empty on every rank is skipped."""
    side = plan.reversed() if reverse else plan
    if plan.max_degree == 0:  # nothing travels on any rank
        size = 1 if out_size is None else int(out_size)
        return jnp.zeros(size, buf.dtype)
    if plan.max_count * jnp.dtype(buf.dtype).itemsize > _MAX_MESSAGE_BYTES:
        # Decided from the plan's global maximum: every rank refuses together.
        raise ValueError(
            f"a message of {plan.max_count} {buf.dtype} values exceeds the "
            f"exchange's {_MAX_MESSAGE_BYTES}-byte message limit"
        )
    if buf.ndim != 1 or buf.shape[0] < max(side.send_total, 1):
        raise ValueError(
            f"exchange buffer of shape {buf.shape} is shorter than the plan's "
            f"{side.send_total} values (or empty)"
        )
    size = max(side.recv_total, 1) if out_size is None else int(out_size)
    if size < side.recv_total:
        raise ValueError("out_size is shorter than the plan's received values")
    _ensure_target()
    return exchange_p.bind(
        buf,
        plan=plan,
        reverse=bool(reverse),
        out_size=size,
        ordered=bool(ordered),
        device_buffers=_device_buffers(),
    )


def halo_gather(
    x: jax.Array,
    send_ids: jax.Array,
    plan: NeighbourPlan,
    *,
    n_ghost: int | None = None,
    ordered: bool = True,
) -> jax.Array:
    """``[x_local | x_ghost]``. Ghosts arrive in the plan's receive order
    (ascending owner, then ascending global column), which is the ghost slot
    order, padded to ``n_ghost`` (sharded layout) or to at least one slot."""
    return jnp.concatenate(
        [x, exchange(x[send_ids], plan, out_size=n_ghost, ordered=ordered)]
    )
