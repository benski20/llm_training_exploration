#!/usr/bin/env python3
"""Network floor: plain TCP ping-pong round-trip time, the lower bound for one masked round trip.

A decode step needs 112 sequential round trips at 7B (28 layers x 4 linears), so the per-trip
latency of the bare transport sets a floor under s/token that no gateway or service change can
remove. This bench measures that floor with the payload sizes a masked decode call really sends:

* ``64 B``: a header-sized message (the smallest frame either side ever sends);
* ``80 KB``: about one 1-row decode reply at 7B (gate_up: 37888 outputs x 4 bytes = 148 KB down,
  qkv 4608 x 4 = 18 KB; 80 KB is a mid-size call).

Each round trip sends ``size`` bytes, the echo server reads them completely and sends them back, and
the client reads the full echo: request plus response, like a compute call. ``TCP_NODELAY`` is set
on both ends (as uvicorn and httpx do). Over loopback this is the co-located floor; pass ``--host``
and run ``--server`` on another machine for a two-host floor.

Usage::

    python benchmarks/net_floor_bench.py                        # loopback, server in a thread
    python benchmarks/net_floor_bench.py --server --port 9555   # on host A
    python benchmarks/net_floor_bench.py --host A --port 9555   # on host B
"""

from __future__ import annotations

import argparse
import json
import platform
import socket
import statistics
import threading
import time
from pathlib import Path
from typing import Any

SIZES = (64, 80 * 1024)


def _recv_exact(sock: socket.socket, buf: memoryview) -> bool:
    got = 0
    n = len(buf)
    while got < n:
        k = sock.recv_into(buf[got:], n - got)
        if k == 0:
            return False
        got += k
    return True


def _serve_conn(conn: socket.socket) -> None:
    """Echo protocol: 4-byte big-endian length, then that many bytes; reply with the same frame."""
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    hdr = bytearray(4)
    buf = bytearray(1 << 20)
    try:
        while True:
            if not _recv_exact(conn, memoryview(hdr)):
                return
            n = int.from_bytes(hdr, "big")
            if n > len(buf):
                return  # bounded: the bench never sends more than 1 MiB
            view = memoryview(buf)[:n]
            if not _recv_exact(conn, view):
                return
            conn.sendall(hdr)
            conn.sendall(view)
    finally:
        conn.close()


def serve(host: str, port: int, ready: threading.Event | None = None) -> socket.socket:
    """Start the echo server (one thread per connection); returns the listening socket."""
    ls = socket.create_server((host, port), reuse_port=False)
    ls.listen(16)
    if ready is not None:
        ready.set()

    def accept_loop() -> None:
        while True:
            try:
                conn, _ = ls.accept()
            except OSError:
                return
            threading.Thread(target=_serve_conn, args=(conn,), daemon=True).start()

    threading.Thread(target=accept_loop, daemon=True, name="net-floor-server").start()
    return ls


def _pct(xs: list[float], q: float) -> float:
    s = sorted(xs)
    return s[min(len(s) - 1, max(0, round(q * (len(s) - 1))))]


def ping_pong(host: str, port: int, size: int, iters: int, warmup: int) -> dict[str, Any]:
    """Round trips of ``size`` bytes each way; returns latency stats in microseconds."""
    sock = socket.create_connection((host, port))
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    payload = bytes(size)
    frame = size.to_bytes(4, "big") + payload
    hdr = bytearray(4)
    back = bytearray(size)
    times: list[float] = []
    try:
        for i in range(warmup + iters):
            t0 = time.perf_counter()
            sock.sendall(frame)
            if not _recv_exact(sock, memoryview(hdr)) or not _recv_exact(sock, memoryview(back)):
                raise ConnectionError("echo server closed the connection")
            dt = time.perf_counter() - t0
            if i >= warmup:
                times.append(dt * 1e6)
    finally:
        sock.close()
    return {
        "bytes_each_way": size,
        "iters": iters,
        "p50_us": round(statistics.median(times), 1),
        "p90_us": round(_pct(times, 0.90), 1),
        "p99_us": round(_pct(times, 0.99), 1),
        "min_us": round(min(times), 1),
        "mean_us": round(statistics.fmean(times), 1),
        "floor_s_per_token_112_trips": round(statistics.median(times) * 112 / 1e6, 4),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=0, help="0: pick a free port (loopback mode only)")
    ap.add_argument("--server", action="store_true", help="only run the echo server (two-host mode)")
    ap.add_argument("--sizes", type=int, nargs="+", default=list(SIZES), help="bytes each way per round trip")
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--json", default=None, help="also write the result here")
    args = ap.parse_args()
    if any(s <= 0 or s > (1 << 20) for s in args.sizes):
        ap.error("--sizes must be between 1 and 1048576 bytes")

    if args.server:
        ls = serve(args.host, args.port)
        print(f"echo server on {ls.getsockname()}", flush=True)
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            ls.close()
        return 0

    ls = None
    host, port = args.host, args.port
    if port == 0:
        ready = threading.Event()
        ls = serve(host, 0, ready)
        ready.wait()
        port = int(ls.getsockname()[1])
    out: dict[str, Any] = {
        "host": host if ls is None else "loopback",
        "python": platform.python_version(),
        "machine": platform.machine(),
        "results": [ping_pong(host, port, s, args.iters, args.warmup) for s in args.sizes],
    }
    if ls is not None:
        ls.close()
    print(json.dumps(out, indent=1))
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
