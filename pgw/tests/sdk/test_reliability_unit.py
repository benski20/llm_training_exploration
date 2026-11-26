"""Reliability of the client library without a provider: cancellation and close, process supervision, transport
pools per event loop, error messages (``docs/sdk/reliability.md``). The end-to-end failure tests are in
``tests/sdk/test_reliability.py``."""

from __future__ import annotations

import asyncio
import contextlib
import errno
import gc
import http.server
import json
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from pgw.sdk import errors as E


# ---- a stand-in engine: the client's stream classes and Engine.submit/close on a real loop ----------------------
class _StubEngine:
    """``chat_stream`` yields a chunk every 10 ms until cancelled; records whether it was cancelled."""

    def __init__(self) -> None:
        from pgw.sdk.engine import Engine

        self._engine = Engine.__new__(Engine)  # only the loop machinery, never the gateway runtime
        e = self._engine
        e.pid, e._mu, e._closed, e._loop, e._thread, e._tasks = os.getpid(), threading.Lock(), False, None, None, set()
        self.cancelled = threading.Event()
        self.started = threading.Event()
        self.produced = 0

    def submit(self, coro: Any) -> Any:
        return self._engine.submit(coro)

    def closed_error(self) -> Any:
        return self._engine.closed_error()

    async def chat_stream(self, req: Any, *, conversation: Any = None, timeout_s: Any = None) -> AsyncIterator[int]:
        self.started.set()
        try:
            while True:
                await asyncio.sleep(0.01)
                self.produced += 1
                yield self.produced
        except asyncio.CancelledError:
            self.cancelled.set()
            raise

    def stop_loop(self) -> None:
        loop, t = self._engine._loop, self._engine._thread
        if loop is not None and t is not None:
            loop.call_soon_threadsafe(loop.stop)
            t.join(5)


def test_a_dropped_sync_stream_stops_generating() -> None:
    """An application that stops iterating and drops the stream (e.g. its own HTTP client disconnected) must not
    keep generating until max_tokens: nothing references the stream from the engine loop, so its finalizer cancels
    the request (its unused masks are burned)."""
    from pgw.sdk.client import _SyncStream

    eng = _StubEngine()
    try:
        s = _SyncStream(eng, None, None, timeout_s=5.0)
        assert next(s) == 1
        del s
        gc.collect()
        assert eng.cancelled.wait(5), "the request kept running after its stream was dropped"
    finally:
        eng.stop_loop()


def test_closing_a_sync_stream_and_ctrl_c_stop_generating() -> None:
    from pgw.sdk.client import _SyncStream

    eng = _StubEngine()
    try:
        with _SyncStream(eng, None, None, timeout_s=5.0) as s:
            next(s)
        assert eng.cancelled.wait(5)
        eng.cancelled.clear()
        s2 = _SyncStream(eng, None, None, timeout_s=5.0)
        next(s2)
        s2._q.get = lambda timeout=None: (_ for _ in ()).throw(KeyboardInterrupt())  # type: ignore[method-assign]
        with pytest.raises(KeyboardInterrupt):
            next(s2)
        assert eng.cancelled.wait(5)
    finally:
        eng.stop_loop()


def test_cancelling_the_consumer_of_an_async_stream_stops_generating() -> None:
    """A web framework cancels the handler task when its client disconnects mid-SSE: the generation stops."""
    from pgw.sdk.client import _AsyncStream

    eng = _StubEngine()

    async def main() -> None:
        s = _AsyncStream(eng, None, None, timeout_s=5.0)

        async def consume() -> None:
            async for _ in s:
                pass

        t = asyncio.create_task(consume())
        await asyncio.sleep(0.1)
        t.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await t
        assert await asyncio.to_thread(eng.cancelled.wait, 5)
        eng.cancelled.clear()
        s2 = _AsyncStream(eng, None, None, timeout_s=5.0)
        async for _ in s2:
            break
        del s2
        gc.collect()
        assert await asyncio.to_thread(eng.cancelled.wait, 5), "an abandoned async stream kept generating"

    try:
        asyncio.run(main())
    finally:
        eng.stop_loop()


def test_close_stops_requests_in_flight_with_a_typed_error(tmp_path: Path) -> None:
    """``close()`` with a request still running (another thread, no timeout) must not leave that caller waiting
    forever: the request is cancelled and raises APIConnectionError; close itself is bounded."""
    from pgw.sdk.engine import Engine
    from pgw.sdk.keys import SdkKey
    from pgw.sdk.options import SDKOptions

    eng = Engine(SdkKey.new("test", "https://127.0.0.1:1", b"\x01" * 32), SDKOptions(usage_reports=False, cache_dir=tmp_path / "c",
                                                                                state_dir=tmp_path / "s"))

    async def forever() -> None:
        await asyncio.Event().wait()

    out: dict[str, Any] = {}

    def caller() -> None:
        try:
            eng.call(forever())
        except Exception as exc:
            out["exc"] = exc

    t = threading.Thread(target=caller)
    t.start()
    time.sleep(0.2)
    t0 = time.monotonic()
    eng.close(timeout_s=5.0)
    assert time.monotonic() - t0 < 10
    t.join(10)
    assert not t.is_alive(), "the caller still waits after close()"
    assert isinstance(out.get("exc"), E.APIConnectionError) and "closed" in out["exc"].message
    with pytest.raises(E.PGWAPIError, match="closed"):
        eng.call(forever())


def test_async_caller_of_a_closed_engine_gets_an_error_not_cancelled(tmp_path: Path) -> None:
    from pgw.sdk.client import _await
    from pgw.sdk.engine import Engine
    from pgw.sdk.keys import SdkKey
    from pgw.sdk.options import SDKOptions

    eng = Engine(SdkKey.new("test", "https://127.0.0.1:1", b"\x01" * 32), SDKOptions(usage_reports=False, cache_dir=tmp_path / "c",
                                                                                state_dir=tmp_path / "s"))

    async def forever() -> None:
        await asyncio.Event().wait()

    async def main() -> Exception | None:
        fut = eng.submit(forever())
        threading.Timer(0.2, eng.close, kwargs={"timeout_s": 5.0}).start()
        try:
            await _await(eng, fut)
        except E.APIConnectionError as exc:
            return exc
        return None

    exc = asyncio.run(main())
    assert isinstance(exc, E.APIConnectionError)


# ---- transport pools per event loop ------------------------------------------------------------------------------
class _Hello(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:
        self.send_response(200)
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, *a: Any) -> None:
        pass


def test_per_loop_transport_serves_two_event_loops() -> None:
    """The compute client calls the provider from the engine's loop and from its private sync loop (the trip
    drivers' HTTP fallback). One shared connection pool fails on the second loop ("bound to a different event
    loop": a revoked key then showed up as a retryable 503); each loop gets its own pool."""
    from pgw.gateway.egress import PerLoopTransport

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Hello)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}/"
    made: list[Any] = []

    def factory() -> httpx.AsyncBaseTransport:
        t = httpx.AsyncHTTPTransport()
        made.append(t)
        return t

    shared = PerLoopTransport(factory)
    other = asyncio.new_event_loop()
    th = threading.Thread(target=other.run_forever, daemon=True)
    th.start()

    async def get() -> int:
        async with httpx.AsyncClient(transport=shared) as c:  # per-loop clients over one transport, as ComputeClient
            r = await c.get(url)
            return r.status_code

    async def keep_alive_get() -> int:
        c = httpx.AsyncClient(transport=shared)
        return (await c.get(url)).status_code  # left open: its pool keeps a connection on this loop

    try:
        loop_a = asyncio.new_event_loop()
        assert loop_a.run_until_complete(keep_alive_get()) == 200
        assert asyncio.run_coroutine_threadsafe(get(), other).result(10) == 200
        assert loop_a.run_until_complete(keep_alive_get()) == 200
        assert len(made) == 2 and shared.pools >= 1
        loop_a.run_until_complete(shared.aclose())  # closes loop A's pool only
        assert asyncio.run_coroutine_threadsafe(get(), other).result(10) == 200
        loop_a.close()
    finally:
        other.call_soon_threadsafe(other.stop)
        th.join(5)
        srv.shutdown()


# ---- processes -------------------------------------------------------------------------------------------------
_ORPHAN = """
import os, signal, sys, time
sys.path[:0] = [sys.argv[1]]
from pgw.core.proc_tuning import exit_with_parent
signal.signal(signal.SIGTERM, signal.SIG_IGN)   # a stop that never finishes
exit_with_parent(grace_s=1.0, poll_s=0.1)
print(os.getpid(), flush=True)
time.sleep(600)
"""

_PARENT = """
import subprocess, sys, time
p = subprocess.Popen([sys.executable, "-c", sys.argv[1], sys.argv[2]], stdout=sys.stdout)
time.sleep(600)
"""


def test_a_helper_process_never_outlives_its_parent(tmp_path: Path) -> None:
    """A factory child or fleet worker whose parent is SIGKILLed (out of memory) ends: SIGTERM at once, then a hard
    exit when its graceful stop hangs. Before: the fleet workers of a killed factory kept running."""
    src = str(Path(__file__).resolve().parents[2] / "src")
    parent = subprocess.Popen([sys.executable, "-c", _PARENT, _ORPHAN, src], stdout=subprocess.PIPE, text=True)  # noqa: S603
    try:
        assert parent.stdout is not None
        child = int(parent.stdout.readline())
        parent.kill()
        parent.wait(10)
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline and _alive(child):
            time.sleep(0.1)
        assert not _alive(child), "the helper outlived its parent"
    finally:
        with contextlib.suppress(ProcessLookupError):
            parent.kill()


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    try:  # a zombie is gone for every practical purpose
        with open(f"/proc/{pid}/stat") as fh:
            return fh.read().split()[2] != "Z"
    except OSError:
        return False


def _plan(tmp_path: Path) -> Any:
    from pgw.sdk.factory import FactoryPlan

    return FactoryPlan(model="m", pack_path=tmp_path / "pack", ring="K32A12C27", state_dir=tmp_path / "f",
                       store_root=tmp_path / "masks", kek_file=tmp_path / "kek", counter_file=tmp_path / "ctr",
                       scheme="plain", check_field="p31", masked_head=False, workers=1, threads=1, nice=10,
                       target_slots=1000, low_water_slots=500, segment_slots=256, first_segment_slots=64)


def _supervise(fp: Any, until: Any, timeout_s: float = 20.0) -> None:
    fp.start()
    deadline = time.monotonic() + timeout_s
    try:
        while time.monotonic() < deadline and not until():
            time.sleep(0.05)
    finally:
        fp.stop(timeout_s=5)


def test_a_hung_factory_is_killed_and_restarted(tmp_path: Path) -> None:
    """A child that stops responding (SIGSTOP, a deadlock holding the interpreter lock: no heartbeat) is killed and
    restarted, and the reason is reported. Before: a hung child was never noticed (``running`` stayed true)."""
    from pgw.sdk.factory import FactoryProcess

    fp = FactoryProcess(_plan(tmp_path), poll_s=0.1, hang_s=1.0)
    fp.command = (sys.executable, "-c", "import time; time.sleep(600)")  # alive, never beats
    _supervise(fp, lambda: fp.starts >= 2)
    assert fp.starts >= 2 and fp.failures >= 1
    assert fp.last_error is not None and "stopped responding" in fp.last_error


def test_a_beating_factory_is_left_alone(tmp_path: Path) -> None:
    from pgw.sdk.factory import FactoryProcess

    fp = FactoryProcess(_plan(tmp_path), poll_s=0.1, hang_s=1.5)
    beat = fp.plan.state_dir / "heartbeat"
    fp.command = (sys.executable, "-c", f"import os, time\nwhile True:\n    open({str(beat)!r}, 'a').close(); "
                  f"os.utime({str(beat)!r}); time.sleep(0.2)")
    _supervise(fp, lambda: False, timeout_s=4.0)
    assert fp.starts == 1 and fp.failures == 0


def test_an_oom_killed_factory_is_reported_as_such(tmp_path: Path) -> None:
    from pgw.sdk.factory import FactoryProcess

    fp = FactoryProcess(_plan(tmp_path), poll_s=0.1)
    fp.log_file.parent.mkdir(parents=True, exist_ok=True)
    fp.log_file.write_text("error [config]: an old error of an earlier child\n")
    fp.command = (sys.executable, "-c", "import os, signal; os.kill(os.getpid(), signal.SIGKILL)")
    _supervise(fp, lambda: fp.failures >= 1)
    assert fp.last_error is not None and fp.last_error.startswith("killed by SIGKILL"), fp.last_error
    assert "out-of-memory" in fp.last_error  # not the earlier child's error line


# ---- errors and messages ---------------------------------------------------------------------------------------
def test_disk_full_is_not_reported_as_a_connection_error() -> None:
    exc = OSError(errno.ENOSPC, "No space left on device", "/cache/pgw/sessions/x/secrets/service_token")
    err = E.from_pgw_error(exc)
    assert isinstance(err, E.InternalServerError) and "disk" in err.message and "Free disk space" in err.message
    assert not err.retryable
    assert isinstance(E.from_pgw_error(ConnectionResetError(errno.ECONNRESET, "reset")), E.APIConnectionError)
    from pgw.core.errors import PGWError

    msg = E.from_pgw_error(PGWError("state commit failed: database or disk is full")).message
    assert "free disk space" in msg


def test_demand_file_is_best_effort(tmp_path: Path) -> None:
    from pgw.sdk.factory import write_demand

    write_demand(tmp_path / "d.json", 123, ttl_s=5)
    d = json.loads((tmp_path / "d.json").read_text())
    assert d["slots"] == 123 and d["until"] > time.time()
    write_demand(tmp_path / "missing" / "dir" / "d.json", 1)  # never raises (a full or read-only disk)
    (tmp_path / "ro").mkdir()
    (tmp_path / "ro").chmod(0o500)
    try:
        write_demand(tmp_path / "ro" / "x" / "d.json", 1)
    finally:
        (tmp_path / "ro").chmod(0o700)


def test_child_env_passes_only_the_disk_reserves(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from pgw.sdk.factory import _child_env

    monkeypatch.setenv("PGW_FACTORY_MIN_FREE_BYTES", "1048576")
    monkeypatch.setenv("PGW_MASKSTORE_MIN_FREE_BYTES", "2097152")
    monkeypatch.setenv("PGW_API_KEY", "pgw_live_never_for_the_child")
    env = _child_env(_plan(tmp_path))
    assert env["PGW_FACTORY_MIN_FREE_BYTES"] == "1048576" and env["PGW_MASKSTORE_MIN_FREE_BYTES"] == "2097152"
    assert "PGW_API_KEY" not in env


_CHILD = """
import sys, time
sys.path[:0] = [sys.argv[1]]
from pgw.core.proc_tuning import exit_with_parent
exit_with_parent(grace_s=2.0, poll_s=0.1)
print("ready", flush=True)
time.sleep(600)
"""

_PARENT_PID = """
import subprocess, sys, time
p = subprocess.Popen([sys.executable, "-c", sys.argv[1], sys.argv[2]], stdout=sys.stdout)
print(p.pid, flush=True)
time.sleep(600)
"""


def test_sigterm_without_a_handler_still_ends_the_helpers() -> None:
    """SIGTERM to an application that installs no handler ends it without atexit (Python's default): its helper
    processes still exit with it (``PR_SET_PDEATHSIG`` / the parent watch)."""
    if not sys.platform.startswith("linux"):
        pytest.skip("Linux process semantics")
    src = str(Path(__file__).resolve().parents[2] / "src")
    parent = subprocess.Popen([sys.executable, "-c", _PARENT_PID, _CHILD, src],  # noqa: S603
                              stdout=subprocess.PIPE, text=True)
    try:
        assert parent.stdout is not None
        child = int(parent.stdout.readline())
        assert parent.stdout.readline().strip() == "ready"
        parent.send_signal(signal.SIGTERM)
        parent.wait(10)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and _alive(child):
            time.sleep(0.1)
        assert not _alive(child)
    finally:
        with contextlib.suppress(ProcessLookupError):
            parent.kill()
