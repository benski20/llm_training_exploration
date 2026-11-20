"""Per-call wire/service overhead of the compute channel, without the GPU.

Starts the real compute service app (uvicorn, the default server) in a subprocess with a registry
of *stub* kernels at 7B shapes (Qwen2.5-7B: qkv 4608x3584, o 3584x3584, gate_up 37888x3584,
down 3584x18944) that return a preallocated int32 result (the GPU's compact form) at zero cost, then
times sequential 1-row calls from a :class:`pgw.wire.ComputeClient` (the stream channel by default,
``--http`` for the HTTP fallback). Everything a call costs here is Python, msgpack, WebSocket/HTTP
and validation on both ends: exactly what the GPU does not see.

Usage::

    python benchmarks/wire_overhead_bench.py [--calls 2000] [--rows 1] [--http] [--token]
        [--server-loop uvloop] [--client-loop uvloop] [--profile-client out.prof] [--profile-server out.prof]

Prints a JSON line: latency percentiles in microseconds, and CPU (user + system) per call of the
client and of the service process. Run from ``pgw/`` with ``PYTHONPATH=src`` to measure this tree.
"""

from __future__ import annotations

import argparse
import asyncio
import cProfile
import json
import os
import socket
import statistics
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

FP = "ab" * 32
RING = "K32A12C27"
SHAPES = {"qkv": (4608, 3584), "o": (3584, 3584), "gate_up": (37888, 3584), "down": (3584, 18944)}
LAYERS = 2


class StubKernel:
    """Duck-typed :class:`pgw.service.ComputeKernel`: zero-cost result of the right shape."""

    backend = "stub"

    def __init__(self, spec: Any):
        import torch

        self.spec = spec
        self.device = torch.device("cpu")
        self._out = torch.zeros((64, spec.out_features), dtype=torch.int32)

    @property
    def linear(self) -> str:
        return str(self.spec.id.key)

    @property
    def kind(self) -> str:
        return str(self.spec.id.kind.value)

    @property
    def out_features(self) -> int:
        return int(self.spec.out_features)

    @property
    def in_features(self) -> int:
        return int(self.spec.in_features)

    device_bytes = 0

    def run_compact(self, x: Any, ring: Any) -> Any:
        return self._out[: x.shape[0]]


def _server(port: int, token_file: str | None, profile: str | None) -> None:
    from pgw.core.types import LinearId, LinearKind, LinearSpec
    from pgw.service.app import create_app
    from pgw.service.config import ServiceSettings
    from pgw.service.registry import ModelRegistry, RegisteredModel
    from pgw.service.server import serve_async

    specs = []
    for layer in range(LAYERS):
        for kind, (m, n) in SHAPES.items():
            specs.append(LinearSpec(LinearId(layer, LinearKind(kind)), m, n))
    reg = ModelRegistry("cpu", rings=(RING,))
    reg.register(RegisteredModel("stub7b", FP, tuple(specs), {s.id.key: StubKernel(s) for s in specs},  # type: ignore[misc]
                                 rings=(RING,)))
    kw: dict[str, Any] = {"allow_unauthenticated": token_file is None}
    if token_file:
        kw["tokens_file"] = Path(token_file)
    settings = ServiceSettings(host="127.0.0.1", port=port, server="uvicorn", batch_window_ms=0.0, **kw)
    app = create_app(settings, registry=reg)
    prof = cProfile.Profile() if profile else None

    async def main() -> None:
        if prof is not None:
            prof.enable()
        try:
            await serve_async(settings, app=app)
        finally:
            if prof is not None:
                prof.disable()
                prof.dump_stats(profile)

    import logging

    logging.basicConfig(level=logging.WARNING)
    _run(main(), os.environ.get("PGW_BENCH_SERVER_LOOP", "asyncio"))


def _run(coro: Any, loop: str) -> Any:
    if loop == "uvloop":
        import uvloop

        return uvloop.run(coro)
    return asyncio.run(coro)


def _proc_stat(pid: int) -> tuple[float, float]:
    """(user CPU seconds, system CPU seconds) of a process."""
    with open(f"/proc/{pid}/stat") as f:
        parts = f.read().rsplit(")", 1)[1].split()
    tck = os.sysconf("SC_CLK_TCK")
    return int(parts[11]) / tck, int(parts[12]) / tck


async def _client(port: int, args: argparse.Namespace, token: str | None, server_pid: int) -> dict[str, Any]:
    import numpy as np
    from pydantic import SecretStr

    from pgw.wire.client import ClientConfig, ComputeClient
    from pgw.wire.messages import ComputeItem, ComputeRequest

    cfg = ClientConfig(base_url=f"http://127.0.0.1:{port}", http2=False, stream=not args.http,
                       token=SecretStr(token) if token else None)
    client = ComputeClient(cfg)
    for _ in range(300):
        if await client.ready():
            break
        await asyncio.sleep(0.1)
    await client.connect(FP, RING)
    rng = np.random.default_rng(0)
    payloads = {k: rng.integers(0, 1 << 32, size=(args.rows, n), dtype=np.uint64).astype("<u4").tobytes()
                for k, (_m, n) in SHAPES.items()}
    order = [f"{layer}.{k}" for layer in range(LAYERS) for k in SHAPES]

    async def run(n: int, lat: list[float] | None) -> None:
        for i in range(n):
            lid = order[i % len(order)]
            item = ComputeItem(lid, args.rows, payloads[lid.split(".")[1]])
            req = ComputeRequest(uuid.uuid4().hex, FP, RING, (item,))
            t0 = time.perf_counter()
            await client.compute(req)
            if lat is not None:
                lat.append((time.perf_counter() - t0) * 1e6)

    await run(200, None)  # warm-up
    lat: list[float] = []
    prof = cProfile.Profile() if args.profile_client else None
    if prof is not None:
        prof.enable()
    (c0, cf0), (s0, sf0) = _proc_stat(os.getpid()), _proc_stat(server_pid)
    t0 = time.perf_counter()
    await run(args.calls, lat)
    wall = time.perf_counter() - t0
    (c1, cf1), (s1, sf1) = _proc_stat(os.getpid()), _proc_stat(server_pid)
    if prof is not None:
        prof.disable()
        prof.dump_stats(args.profile_client)
    await client.aclose()
    lat.sort()

    def pct(p: float) -> float:
        return round(lat[min(len(lat) - 1, int(p * len(lat)))], 1)

    return {"transport": "http" if args.http else "stream", "server_loop": args.server_loop,
            "client_loop": args.client_loop, "rows": args.rows, "calls": len(lat),
            "p50_us": pct(0.5), "p90_us": pct(0.9), "p99_us": pct(0.99),
            "mean_us": round(statistics.fmean(lat), 1), "wall_us_per_call": round(wall / len(lat) * 1e6, 1),
            "client_cpu_us_per_call": round((c1 - c0 + cf1 - cf0) / len(lat) * 1e6, 1),
            "server_cpu_us_per_call": round((s1 - s0 + sf1 - sf0) / len(lat) * 1e6, 1),
            "client_sys_us_per_call": round((cf1 - cf0) / len(lat) * 1e6, 1),
            "server_sys_us_per_call": round((sf1 - sf0) / len(lat) * 1e6, 1)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--calls", type=int, default=2000)
    ap.add_argument("--rows", type=int, default=1)
    ap.add_argument("--http", action="store_true", help="HTTP fallback instead of the stream channel")
    ap.add_argument("--token", action="store_true", help="bearer-token authentication (default: none)")
    ap.add_argument("--server-loop", default="asyncio", choices=["asyncio", "uvloop"],
                    help="event loop of the service process (benchmark harness only)")
    ap.add_argument("--client-loop", default="asyncio", choices=["asyncio", "uvloop"])
    ap.add_argument("--profile-client")
    ap.add_argument("--profile-server")
    ap.add_argument("--serve", type=int, help=argparse.SUPPRESS)
    ap.add_argument("--token-file", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.serve:
        _server(args.serve, args.token_file, args.profile_server)
        return
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = int(s.getsockname()[1])
    token = None
    with tempfile.TemporaryDirectory() as td:
        cmd = [sys.executable, __file__, "--serve", str(port)]
        if args.token:
            token = "bench-token-" + uuid.uuid4().hex
            tf = Path(td) / "tokens.txt"
            import hashlib

            tf.write_text(f"bench:{hashlib.sha256(token.encode()).hexdigest()}\n")
            cmd += ["--token-file", str(tf)]
        if args.profile_server:
            cmd += ["--profile-server", args.profile_server]
        proc = subprocess.Popen(cmd, env={**os.environ, "PGW_BENCH_SERVER_LOOP": args.server_loop})  # noqa: S603
        try:
            res = _run(_client(port, args, token, proc.pid), args.client_loop)
        finally:
            proc.send_signal(2)
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
    print(json.dumps(res))


if __name__ == "__main__":
    main()
