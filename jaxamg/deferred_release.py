"""Release native resources only after the device work that may use them.

Programs load registered arrays and exchange plans by id and run
asynchronously, so a resource may still be read after its last Python owner
goes. `release_after_device_work` hands the release to a worker thread, which
waits for this process's devices (a barrier computation per device, then a
device synchronize) before releasing. (Garbage collection can run inside a
host callback of the very execution it would wait for, hence the thread.)
This relies on executions on a device completing in dispatch order (JAX 0.9.1).

If anything fails, that release and all later ones are kept, never made
unsafely, and `wait_for_releases` raises. At exit the worker stops before
JAX's backends are torn down; pending resources are reclaimed by process exit.
"""

from __future__ import annotations

import atexit
import queue
import threading
from collections.abc import Callable
from typing import Any

_pending: queue.SimpleQueue = queue.SimpleQueue()
_worker: threading.Thread | None = None
_state = threading.Condition()  # guards every field below
_outstanding = 0
_failure: BaseException | None = None
_retained: list = []  # batches kept after a failure or at exit (never released)
_stopping = False
_no_worker = False  # the worker thread could not be started (terminal)
_STOP = object()


def _drain(devices, ordinals, barrier) -> None:
    """Wait for the work dispatched so far on this process's devices."""
    import jax
    import numpy as np

    from ._ext import _amgx

    for device in devices:
        jax.block_until_ready(barrier(jax.device_put(np.zeros((), np.float32), device)))
    _amgx.synchronize_devices(ordinals)


def _finish(count: int) -> None:
    global _outstanding
    with _state:
        _outstanding -= count
        _state.notify_all()


def _retain(batch: list, error: BaseException | None = None) -> None:
    """Keep ``batch`` (never released); record the first failure and wake
    the waiters, who report it."""
    global _failure
    with _state:
        if batch:
            _retained.append(batch)
        if error is not None and _failure is None:
            _failure = error
            _state.notify_all()


def _process(batch: list, devices, ordinals, barrier) -> None:
    # A frame of its own: nothing of the batch outlives the call unless it is
    # retained deliberately.
    try:
        if _failure is not None:
            raise RuntimeError("an earlier release failed") from _failure
        _drain(devices, ordinals, barrier)
        if _stopping:  # exit began during the drain: start nothing more
            _retain(batch)
            return
        for release, args, _ in batch:
            release(*args)
    except BaseException as error:  # retain rather than free unsafely
        _retain(batch, error)
    finally:
        _finish(len(batch))


def _next_batch() -> tuple[list, bool]:
    batch = [_pending.get()]
    while True:
        try:
            batch.append(_pending.get_nowait())
        except queue.Empty:
            break
    stop = any(item is _STOP for item in batch)
    return [item for item in batch if item is not _STOP], stop


def _run() -> None:
    devices = []
    ordinals: list[int] = []
    barrier = None
    try:
        import jax

        devices = [d for d in jax.local_devices() if d.platform == "gpu"]
        ordinals = sorted({int(getattr(d, "local_hardware_id", d.id)) for d in devices})
        barrier = jax.jit(lambda x: x + 1)
    except BaseException as error:  # the manager cannot work: retain everything
        _retain([], error)
    while True:
        batch, stop = _next_batch()
        if stop or _stopping or barrier is None:
            # At exit, or without a working manager: keep them (process exit
            # reclaims them at exit); the failure, if any, is reported.
            _retain(batch)
            _finish(len(batch))
            if stop or _stopping:
                return
            continue
        _process(batch, devices, ordinals, barrier)
        del batch  # else it stays alive while the next batch is awaited


def _retain_queued() -> None:
    """Retain everything still queued (no worker will process it)."""
    batch = []
    while True:
        try:
            item = _pending.get_nowait()
        except queue.Empty:
            break
        if item is not _STOP:
            batch.append(item)
    if batch:
        _retain(batch)
        _finish(len(batch))


def release_after_device_work(
    release: Callable[..., Any], *args: Any, keep: tuple = ()
) -> None:
    """Call ``release(*args)`` once the device work dispatched so far has
    finished; ``keep`` (for example the device array a registry entry points
    into) stays alive until then."""
    global _worker, _outstanding, _no_worker
    item = (release, args, keep)
    if _stopping or _no_worker:
        # At exit, or with no worker to run it: keep it, never release it.
        _retain([item])
        return
    with _state:
        _outstanding += 1
        _pending.put(item)
        if _worker is not None:
            return
        _worker = threading.Thread(target=_run, name="jaxamg-release", daemon=True)
        try:
            _worker.start()
        except BaseException as error:  # terminal: no consumer, ever
            _no_worker = True
            _retain([], error)
            _retain_queued()


def wait_for_releases(timeout: float | None = None) -> bool:
    """Wait until every release handed over so far has been processed; False
    on timeout. Raises as soon as a release could not be made safely (the
    resources are then retained)."""
    with _state:
        done = _state.wait_for(
            lambda: _outstanding == 0 or _failure is not None, timeout
        )
        if _failure is not None:
            raise RuntimeError(
                "a deferred release failed; its resources and every later "
                "release are retained"
            ) from _failure
        return done


def _stop_at_exit() -> None:
    # Registered after JAX is imported, so it runs before JAX's own exit
    # handlers: the worker has stopped before the backends are torn down. A
    # drain in progress is waited for (it only waits on work already
    # dispatched); what it would have released is retained instead.
    global _stopping
    _stopping = True
    if _worker is not None and _worker.is_alive():
        _pending.put(_STOP)
        _worker.join()


atexit.register(_stop_at_exit)
