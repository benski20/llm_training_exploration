#!/usr/bin/env python3
"""Per-trip latency of a decode token's round-trip schedule, by transport (chat-speed plan WP-A0).

A 7B decode token is ``4 x layers`` sequential round trips (qkv, o, gate_up, down per layer); with
the WebSocket stream channel a trip costs about 1.46 ms on Kaggle (run 27), 163 ms per token. The
native trip lane (WP-A1/A3) targets 0.29-0.37 ms. This bench measures the same trip schedule, with
the real byte sizes (``rows x in x K/8`` up, ``rows x out x K/8`` down, K = 32), over:

* ``tcp`` / ``uds``: a bare length-prefixed request/reply over TCP loopback (``TCP_NODELAY``) or a
  Unix domain socket, Python on both ends (the server is a thread of this process). ``--wait spin``
  busy-polls a non-blocking socket instead of blocking in ``recv`` (the client half of the hybrid
  wait), so the gap between the two shows the wake-up cost;
* ``stream``: today's path. :class:`pgw.wire.client.ComputeClient` (``compute_fast``, the lean
  stream lane) against the real service app in a subprocess with zero-cost stub kernels at 7B
  shapes (``wire_overhead_bench.py --serve``): everything but the GPU;
* ``--transport MODULE:FUNC`` (any number): a plug-in transport. ``FUNC(shapes, ring_k)`` returns an
  object with ``trip(linear: str, rows: int, body: bytes) -> bytes`` (and optionally ``close()``);
  the native lane client registers itself this way, so this file never has to change for it.

Prints JSON: per transport and rows, the per-trip p50/p90/p99 in microseconds and the projected
``ms_per_token`` = trips per token x p50. Numbers from this shared 4-vCPU box are only relative;
Kaggle/Modal runs are what count.

Usage::

    python benchmarks/lane_bench.py [--transports tcp,uds,stream] [--rows 1,4,8] [--tokens 20]
        [--wait block|spin|both] [--layers 28] [--transport my.module:factory] [--json out.json]
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import importlib
import json
import os
import platform
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

#: 7B linear shapes (out, in), the order of a layer's trips.
SHAPES_7B: dict[str, tuple[int, int]] = {"qkv": (4608, 3584), "o": (3584, 3584), "gate_up": (37888, 3584),
                                        "down": (3584, 18944)}
RING_K = 32
_HDR = 8  # request: u32 body length, u32 reply length (both little-endian)


# ---- bare socket floor ----------------------------------------------------------------------------
def _recv_exact(sock: socket.socket, view: memoryview, spin: bool = False) -> bool:
    got, n = 0, len(view)
    while got < n:
        try:
            k = sock.recv_into(view[got:], n - got)
        except BlockingIOError:
            if spin:
                continue
            raise
        if k == 0:
            return False
        got += k
    return True


def _serve_conn(conn: socket.socket, max_reply: int) -> None:
    if conn.family != socket.AF_UNIX:
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    hdr = bytearray(_HDR)
    buf = bytearray(1 << 22)
    reply = bytes(max_reply)
    try:
        while True:
            if not _recv_exact(conn, memoryview(hdr)):
                return
            n_up, n_down = int.from_bytes(hdr[:4], "little"), int.from_bytes(hdr[4:], "little")
            if n_up > len(buf):
                buf = bytearray(n_up)
            if not _recv_exact(conn, memoryview(buf)[:n_up]):
                return
            conn.sendall(memoryview(reply)[:n_down])
    except OSError:
        return
    finally:
        conn.close()


class FloorServer:
    """Request/reply server in a thread: reads ``[u32 up][u32 down][up bytes]``, replies ``down`` bytes."""

    def __init__(self, family: str, max_reply: int):
        self._tmp: tempfile.TemporaryDirectory[str] | None = None
        if family == "uds":
            self._tmp = tempfile.TemporaryDirectory(prefix="pgw-lane-bench-")
            self.address: Any = os.path.join(self._tmp.name, "lane.sock")
            self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        else:
            self.address = ("127.0.0.1", 0)
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind(self.address)
        if family != "uds":
            self.address = self.sock.getsockname()
        self.sock.listen(4)
        self.family = family
        self._max_reply = max_reply
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self) -> None:
        while True:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                return
            threading.Thread(target=_serve_conn, args=(conn, self._max_reply), daemon=True).start()

    def close(self) -> None:
        self.sock.close()
        if self._tmp is not None:
            self._tmp.cleanup()


class FloorTransport:
    def __init__(self, server: FloorServer, spin: bool):
        fam = socket.AF_UNIX if server.family == "uds" else socket.AF_INET
        self.sock = socket.socket(fam, socket.SOCK_STREAM)
        self.sock.connect(server.address)
        if fam == socket.AF_INET:
            self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.spin = spin
        if spin:
            self.sock.setblocking(False)
        self._reply = bytearray(1 << 22)

    def trip(self, linear: str, rows: int, body: bytes, n_down: int) -> memoryview:
        msg = len(body).to_bytes(4, "little") + n_down.to_bytes(4, "little")
        if self.spin:
            self.sock.setblocking(True)
            self.sock.sendall(msg + body)
            self.sock.setblocking(False)
        else:
            self.sock.sendall(msg + body)
        if n_down > len(self._reply):
            self._reply = bytearray(n_down)
        view = memoryview(self._reply)[:n_down]
        if not _recv_exact(self.sock, view, spin=self.spin):
            raise ConnectionError("floor server closed")
        return view

    def close(self) -> None:
        self.sock.close()


# ---- today's stream channel against a stub service ------------------------------------------------
class StreamTransport:
    """ComputeClient.compute_fast against ``wire_overhead_bench.py --serve`` (stub kernels, 7B shapes)."""

    FP = "ab" * 32
    RING = "K32A12C27"

    def __init__(self) -> None:
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            self.port = int(s.getsockname()[1])
        script = Path(__file__).resolve().parent / "wire_overhead_bench.py"
        self.proc = subprocess.Popen([sys.executable, str(script), "--serve", str(self.port)],  # noqa: S603
                                     stderr=subprocess.DEVNULL)
        self.loop = asyncio.new_event_loop()
        self.client = self.loop.run_until_complete(self._connect())

    async def _connect(self) -> Any:
        from pgw.wire.client import ClientConfig, ComputeClient

        client = ComputeClient(ClientConfig(base_url=f"http://127.0.0.1:{self.port}", http2=False, stream=True))
        for _ in range(600):
            if self.proc.poll() is not None:
                raise RuntimeError("stub service exited")
            if await client.ready():
                break
            await asyncio.sleep(0.1)
        await client.connect(self.FP, self.RING)
        return client

    def trip(self, linear: str, rows: int, body: bytes, n_down: int) -> Any:
        from pgw.wire.messages import ComputeItem, ComputeRequest

        req = ComputeRequest(uuid.uuid4().hex, self.FP, self.RING, (ComputeItem(f"0.{linear}", rows, body),))
        return self.loop.run_until_complete(self.client.compute_fast(req))

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self.loop.run_until_complete(self.client.aclose())
        self.loop.close()
        self.proc.send_signal(2)
        try:
            self.proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self.proc.kill()


class PluginTransport:
    def __init__(self, spec: str):
        mod, _, fn = spec.partition(":")
        factory: Callable[..., Any] = getattr(importlib.import_module(mod), fn)
        self.impl = factory(SHAPES_7B, RING_K)

    def trip(self, linear: str, rows: int, body: bytes, n_down: int) -> Any:
        return self.impl.trip(linear, rows, body)

    def close(self) -> None:
        close = getattr(self.impl, "close", None)
        if close is not None:
            close()


# ---- driver -------------------------------------------------------------------------------------
def measure(transport: Any, rows: int, tokens: int, warmup_tokens: int = 2) -> dict[str, Any]:
    """Time ``tokens`` x one layer's 4 trips (each trip timed alone); per-token = 4 x layers x p50."""
    import numpy as np

    rng = np.random.default_rng(0)
    nb = RING_K // 8
    bodies = {k: rng.integers(0, 256, size=rows * n * nb, dtype=np.uint8).tobytes() for k, (_m, n) in SHAPES_7B.items()}
    lat: list[float] = []
    for tok in range(warmup_tokens + tokens):
        for linear, (m, _n) in SHAPES_7B.items():
            t = time.perf_counter()
            transport.trip(linear, rows, bodies[linear], rows * m * nb)
            if tok >= warmup_tokens:
                lat.append(time.perf_counter() - t)
    lat.sort()

    def pct(q: float) -> float:
        return round(lat[min(len(lat) - 1, int(q * len(lat)))] * 1e6, 1)

    return {"rows": rows, "trips": len(lat), "p50_us": pct(0.5), "p90_us": pct(0.9), "p99_us": pct(0.99),
            "mean_us": round(statistics.fmean(lat) * 1e6, 1)}


def run(args: argparse.Namespace) -> dict[str, Any]:
    rows_list = [int(r) for r in str(args.rows).split(",") if r]
    waits = ["block", "spin"] if args.wait == "both" else [args.wait]
    max_reply = max(rows_list) * max(m for m, _ in SHAPES_7B.values()) * RING_K // 8
    out: dict[str, Any] = {"layers": args.layers, "trips_per_token": 4 * args.layers, "ring_k": RING_K,
                           "cpu_count": os.cpu_count(), "machine": platform.machine(), "results": []}
    names = [t for t in args.transports.split(",") if t] + list(args.transport or [])
    for name in names:
        variants: list[tuple[str, Callable[[], Any], Callable[[], None]]] = []
        if name in ("tcp", "uds"):
            server = FloorServer(name, max_reply)
            for w in waits:
                variants.append((f"{name}-{w}", lambda s=server, w=w: FloorTransport(s, w == "spin"), lambda: None))
            variants[-1] = (variants[-1][0], variants[-1][1], server.close)
        elif name == "stream":
            variants.append(("stream", StreamTransport, lambda: None))
        else:
            variants.append((name, lambda n=name: PluginTransport(n), lambda: None))
        for label, make, cleanup in variants:
            try:
                tr = make()
            except Exception as exc:
                out["results"].append({"transport": label, "error": f"{type(exc).__name__}: {exc}"[:500]})
                cleanup()
                continue
            try:
                for rows in rows_list:
                    res = {"transport": label, **measure(tr, rows, args.tokens)}
                    res["ms_per_token"] = round(res["p50_us"] * 4 * args.layers / 1e3, 2)
                    out["results"].append(res)
                    print(json.dumps(res), flush=True)
            finally:
                tr.close()
                cleanup()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--transports", default="tcp,uds,stream", help="comma list of tcp, uds, stream")
    ap.add_argument("--transport", action="append", help="plug-in transport MODULE:FUNC (repeatable)")
    ap.add_argument("--rows", default="1,4,8", help="rows per trip, comma list")
    ap.add_argument("--tokens", type=int, default=20, help="measured tokens (x4 trips each) per rows value")
    ap.add_argument("--wait", default="both", choices=["block", "spin", "both"], help="client wait (tcp/uds)")
    ap.add_argument("--layers", type=int, default=28, help="layers per token for the projection (7B: 28)")
    ap.add_argument("--json", default=None, help="also write the result here")
    args = ap.parse_args()
    res = run(args)
    text = json.dumps(res, indent=1)
    print(text)
    if args.json:
        Path(args.json).write_text(text + "\n")


if __name__ == "__main__":
    main()
