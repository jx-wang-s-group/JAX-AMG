"""The deferred-release worker (``jaxamg.deferred_release``) with the native
device synchronization replaced. Each case runs in its own process, since a
failure is terminal for the worker:

- ``ok``: releases run and drop their payloads;
- ``sync``: a failed synchronization retains everything and is reported;
- ``thread``: a worker thread that cannot start; every request is retained
  and reported, and a waiter without a timeout returns;
- ``startup``: a worker whose setup fails, likewise;
- ``exit``: the exit handler returns only once the worker has stopped, and a
  drain that ends after exit began releases nothing.
"""

import json
import os
import pathlib
import subprocess
import sys
import textwrap

import pytest

SCRIPT = textwrap.dedent("""
    import gc, json, sys, threading, time, types, weakref
    import jax
    mode = sys.argv[1]
    def synchronize(devices):
        if mode == "sync":
            raise RuntimeError("injected synchronization failure")
    sys.modules["jaxamg._ext"] = types.SimpleNamespace(
        _amgx=types.SimpleNamespace(synchronize_devices=synchronize)
    )
    from jaxamg import deferred_release as d
    released, reports, out = [], [], {}

    def submit_and_wait(n, **keep):
        d.release_after_device_work(released.append, n, **keep)
        try:
            d.wait_for_releases()  # the default: no timeout
        except RuntimeError:
            reports.append(n)

    if mode in ("ok", "sync"):
        class Payload:
            pass
        payload = Payload()
        alive = weakref.ref(payload)
        submit_and_wait(1, keep=(payload,))
        del payload
        submit_and_wait(2)
        gc.collect()
        out.update(payload_alive=alive() is not None)
    elif mode == "thread":
        def refuse(self):
            raise RuntimeError("injected thread start failure")
        threading.Thread.start = refuse
        submit_and_wait(1)
        submit_and_wait(2)
        out.update(queued=not d._pending.empty())
    elif mode == "startup":
        def fail():
            raise RuntimeError("injected device discovery failure")
        jax.local_devices = fail
        d.release_after_device_work(released.append, 1)
        submit_and_wait(2)
        # The report comes as soon as the failure is recorded; the worker then
        # retains the rest.
        deadline = time.monotonic() + 30
        while d._outstanding and time.monotonic() < deadline:
            time.sleep(0.01)
    else:  # exit while a drain is in progress
        started, finish = threading.Event(), threading.Event()
        def drain(*args):
            started.set()
            finish.wait(30)
        d._drain = drain
        d.release_after_device_work(released.append, 1)
        started.wait(30)
        threading.Timer(1.0, finish.set).start()
        d._stop_at_exit()
    out.update(
        released=released,
        reports=reports,
        outstanding=d._outstanding,
        retained=sum(len(batch) for batch in d._retained),
        worker_alive=d._worker is not None and d._worker.is_alive(),
    )
    print(json.dumps(out))
    """)

EXPECTED = {
    "ok": dict(payload_alive=False, released=[1, 2], reports=[], retained=0),
    "sync": dict(payload_alive=True, released=[], reports=[1, 2], retained=2),
    "thread": dict(queued=False, released=[], reports=[1, 2], retained=2),
    "startup": dict(released=[], reports=[2], retained=2),
    "exit": dict(released=[], reports=[], retained=1),
}
WORKER_ALIVE = {"ok": True, "sync": True, "thread": False, "startup": True}


@pytest.mark.parametrize("mode", list(EXPECTED))
def test_deferred_release(mode, tmp_path):
    script = tmp_path / "worker.py"
    script.write_text(SCRIPT)
    result = subprocess.run(
        [sys.executable, str(script), mode],
        capture_output=True,
        text=True,
        timeout=120,
        env={
            **os.environ,
            "JAX_PLATFORMS": "cpu",
            "PYTHONPATH": str(pathlib.Path(__file__).resolve().parents[1]),
        },
    )
    assert result.returncode == 0, result.stderr
    out = json.loads(result.stdout.strip().splitlines()[-1])
    assert out.pop("outstanding") == 0
    assert out.pop("worker_alive") == WORKER_ALIVE.get(mode, False)
    assert out == EXPECTED[mode]
