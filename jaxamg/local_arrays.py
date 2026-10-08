"""Rank-local arrays as runtime operands of rank-identical programs.

`register_local` places each rank's array on its device under an id agreed by
all ranks; `load_local` reads it inside a program by id. Every rank thus
compiles the same program and receives its own data at run time. Programs that
load an array keep it alive, and it is released only after the device work
queued before its release (`jaxamg.deferred_release`).
"""

from __future__ import annotations

import dataclasses
import weakref
import zlib
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from jax.extend.core import Primitive
from jax.interpreters import mlir

if TYPE_CHECKING:
    from mpi4py.MPI import Comm

_TARGET = "jaxamg_local_array"
_registered = False
_next_id = 1


def _ensure_target() -> None:
    global _registered
    if not _registered:
        from ._ext import _amgx

        jax.ffi.register_ffi_target(
            _TARGET, _amgx.get_local_array_handler(), platform="CUDA"
        )
        _registered = True


@dataclasses.dataclass(frozen=True, eq=False)
class LocalArray:
    """A registered rank-local array: the same id, shape and dtype on every
    rank; different contents (``value``, this rank's device array)."""

    array_id: int
    shape: tuple[int, ...]
    dtype: np.dtype
    value: jax.Array


def register_local(
    comm: Comm, array: np.ndarray, device: jax.Device | None = None
) -> LocalArray:
    """Register this rank's ``array`` (collective, in the same order on every
    rank; one O(1) reduction agrees the id and checks shape and dtype). It is
    uploaded to ``device`` (default: this process's first local device), the
    device the programs reading it run on. The native copy is released once
    neither the returned handle nor a program loading it is alive."""
    global _next_id
    from mpi4py import MPI

    from ._ext import _amgx
    from .deferred_release import release_after_device_work
    from .utils import temp_enable_x64

    array = np.ascontiguousarray(array)
    # A fixed-length signature, so ranks that disagree still reduce alike.
    shape_code = zlib.crc32(np.asarray(array.shape, dtype=np.int64).tobytes())
    signature = [array.ndim, array.size, array.dtype.num, shape_code]
    bounds = np.array([_next_id, *signature, *(-v for v in signature)], np.int64)
    comm.Allreduce(MPI.IN_PLACE, bounds, op=MPI.MAX)
    k = len(signature)
    if np.any(bounds[1 : 1 + k] != -bounds[1 + k :]):
        raise ValueError(
            "rank-local arrays must have one shape and dtype on every rank; "
            f"this rank has {array.shape} {array.dtype}"
        )
    array_id = int(bounds[0])
    _next_id = array_id + 1
    if device is None:
        device = jax.local_devices()[0]
    # Concrete even while tracing, in the registered dtype (64-bit indices
    # too), complete before registration, and never written afterwards.
    with jax.ensure_compile_time_eval(), temp_enable_x64():
        value = jax.device_put(array, device).block_until_ready()
    _amgx.register_local_array(array_id, value.unsafe_buffer_pointer(), value.nbytes)
    handle = LocalArray(array_id, tuple(array.shape), array.dtype, value)
    finalizer = weakref.finalize(
        handle,
        release_after_device_work,
        _amgx.release_local_array,
        array_id,
        keep=(value,),
    )
    # typeshed models this writable property as a field on an empty-slots class.
    finalizer.atexit = False  # type: ignore[misc]
    return handle


# The handle is a parameter of the load, so a traced program owns it.
load_p = Primitive("jaxamg_load_local")
load_p.def_abstract_eval(
    lambda *, handle: jax.core.ShapedArray(handle.shape, jnp.dtype(handle.dtype))
)
load_p.def_impl(lambda *, handle: handle.value)


def _lowering(ctx, *, handle):
    ctx.module_context.add_keepalive(handle)  # the executable owns it too
    return jax.ffi.ffi_lowering(_TARGET)(ctx, array_id=np.int64(handle.array_id))


mlir.register_lowering(load_p, _lowering, platform="cuda")


def load_local(handle: LocalArray, *, varying_axis=None) -> jax.Array:
    """This rank's array in a program. Inside ``shard_map`` pass the mesh axis
    (or axes) the value varies over, since every rank holds different data."""
    _ensure_target()
    out = load_p.bind(handle=handle)
    if varying_axis is not None:
        out = jax.lax.pcast(out, varying_axis, to="varying")
    return out
