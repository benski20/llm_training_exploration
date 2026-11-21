"""Per-call cost of ``POST /pgw/v1/maskgen`` *outside* the GPU: client pack/unpack, msgpack, HTTP, TLS.

The encrypted mask factory calls the provider once per linear per batch of 256-slot groups. On Modal
(one T4 container, Qwen2.5-0.5B, 512 slots per pass) the factory's ``provider`` time was 9.40 s of a
15.15 s wall, while the provider's device work for the same items is well under a second
(``benchmarks/maskgen_provider_bench.py``). This bench measures what the rest costs.

The real service app runs in a subprocess (uvicorn, the default server; ``--tls`` for TLS as on Modal)
with stub kernels at real layer shapes whose maskgen item returns a preallocated reply of the right size
at zero cost (``--server-compute real`` runs the real CPU kernel instead, small shapes only). The client
is the factory's :class:`pgw.hemask.client.RemoteMulmod64` on a :class:`pgw.wire.ComputeClient`, called
from a worker thread exactly as :class:`pgw.hemask.backend.HEMaskBackend` does.

Reported per (format, linear kind): seconds per call, MB up/down, and the client-side split
(``pack``: dither + bit packing + msgpack of the request; ``wire``: from send to the reply bytes;
``unpack``: decoding the reply to ``uint64``), plus slots/s of a whole model pass (one call per linear,
``--layers`` layers, ``--cols`` slots) when everything but the GPU is accounted for.

    python benchmarks/maskgen_wire_bench.py --preset qwen2.5-0.5b --cols 512 --tls --formats v1 v2 [--json out]
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

FP = "cd" * 32
PRESETS = {  # (layers, {kind: (m, n)})
    "qwen2.5-0.5b": (24, {"qkv": (1152, 896), "o": (896, 896), "gate_up": (9728, 896), "down": (896, 4864)}),
    "qwen2.5-7b": (28, {"qkv": (4608, 3584), "o": (3584, 3584), "gate_up": (37888, 3584), "down": (3584, 18944)}),
    "tiny": (2, {"qkv": (96, 64), "o": (64, 64), "gate_up": (256, 64), "down": (64, 128)}),
}


def _server(port: int, preset: str, compute: str) -> None:
    import asyncio

    import torch

    from pgw.core.types import LinearId, LinearKind, LinearSpec
    from pgw.service import handler as handler_mod
    from pgw.service.app import create_app
    from pgw.service.config import ServiceSettings
    from pgw.service.registry import ModelRegistry, RegisteredModel
    from pgw.service.server import serve_async

    _layers, shapes = PRESETS[preset]

    class StubKernel:
        backend = "stub"
        device_bytes = 0

        def __init__(self, spec: Any):
            self.spec = spec
            self.device = torch.device("cpu")
            self._pw = None

        linear = property(lambda self: str(self.spec.id.key))
        kind = property(lambda self: str(self.spec.id.kind.value))
        out_features = property(lambda self: int(self.spec.out_features))
        in_features = property(lambda self: int(self.spec.in_features))

    specs = [LinearSpec(LinearId(0, LinearKind(k)), m, n) for k, (m, n) in shapes.items()]
    if compute == "real":  # real kernels (random int8 weights, seed 0) on PGW_BENCH_DEVICE (default: cuda if any)
        from pgw.service.kernels import ComputeKernel

        dev = os.environ.get("PGW_BENCH_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
        reg = ModelRegistry(dev, rings=("K32A12C27",))
        kerns: dict[str, Any] = {}
        for sp in specs:
            w = _weights(sp.out_features, sp.in_features, str(sp.id.kind.value))
            kerns[sp.id.key] = ComputeKernel(sp, torch.from_numpy(w), device=dev, cuda_graphs=False, triton_gemv=False)
    else:
        reg = ModelRegistry("cpu", rings=("K32A12C27",))
        kerns = {sp.id.key: StubKernel(sp) for sp in specs}
    reg.register(RegisteredModel("bench", FP, tuple(specs), kerns, rings=("K32A12C27",)))  # type: ignore[arg-type]
    replies: dict[tuple[str, int], bytes] = {}

    def stub_item(kern: Any, it: Any, decode_busy: Any = None) -> bytes:
        nbytes = it.y_nbytes(kern.out_features)
        key = (kern.linear, nbytes)
        r = replies.get(key)
        if r is None:
            r = replies[key] = os.urandom(nbytes)
        return r

    if compute == "stub":
        handler_mod._maskgen_item = stub_item  # type: ignore[assignment]
    kw: dict[str, Any] = {"allow_unauthenticated": True}
    tls_dir = os.environ.get("PGW_BENCH_TLS_DIR")
    if tls_dir:
        from pgw.core.settings import TLSConfig

        kw["tls"] = TLSConfig(cert_file=Path(tls_dir) / "server.pem", key_file=Path(tls_dir) / "server.key")
    bulk_port = os.environ.get("PGW_BENCH_BULK_PORT")
    settings = ServiceSettings(host="127.0.0.1", port=port, server=os.environ.get("PGW_BENCH_SERVER", "uvicorn"),
                               batch_window_ms=0.0, maskgen_bulk_port=int(bulk_port) if bulk_port else None, **kw)
    app = create_app(settings, registry=reg)

    prof_path = os.environ.get("PGW_BENCH_PROFILE_SERVER")

    async def main() -> None:
        import cProfile

        print(json.dumps({"server_ready": True}), file=sys.stderr, flush=True)
        prof = cProfile.Profile() if prof_path else None
        if prof is not None:
            prof.enable()
        try:
            await serve_async(settings, app=app)
        finally:
            if prof is not None:
                prof.disable()
                prof.dump_stats(prof_path)

    import logging

    from pgw.core.proc_tuning import new_event_loop

    logging.basicConfig(level=logging.WARNING)
    asyncio.run(main(), loop_factory=new_event_loop)


def _weights(m: int, n: int, kind: str) -> Any:
    import numpy as np

    seed = sum(map(ord, kind))
    return np.random.default_rng(seed).integers(-128, 128, size=(m, n), dtype=np.int8)


def _check_exact(remote: Any, shapes: dict[str, tuple[int, int]], q_bits: int) -> dict[str, Any]:
    """v1 replies (deterministic) over this transport vs the CPU reference ``W8 b mod 2^q`` on a column slice."""
    import numpy as np

    from pgw.hemask.lwe import matmul_mod64

    out = {}
    rng = np.random.default_rng(5)
    for kind, (m, n) in shapes.items():
        b = rng.integers(0, 1 << q_bits, size=(n, 512), dtype=np.uint64)
        y = remote._call_one(f"0.{kind}", b) if remote.bulk is None else remote(f"0.{kind}", b)
        w = _weights(m, n, kind)
        ref = matmul_mod64(w[:64], b[:, :32], fast=False) & np.uint64((1 << q_bits) - 1)
        got = y[:64, :32]
        # v1 with drop 16: the reply is round(y / 2^16) << 16
        refr = ((ref + np.uint64(1 << 15)) >> np.uint64(16) << np.uint64(16)) & np.uint64((1 << q_bits) - 1)
        out[kind] = bool(np.array_equal(got, refr))
    return out


def _free_port() -> int:
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _cpu(pid: int) -> float:
    with open(f"/proc/{pid}/stat") as f:
        parts = f.read().rsplit(")", 1)[1].split()
    return (int(parts[11]) + int(parts[12])) / os.sysconf("SC_CLK_TCK")


def _client(port: int, args: argparse.Namespace, ca: Path | None, server_pid: int, bulk_port: int | None
            ) -> list[dict[str, Any]]:
    import numpy as np

    from pgw.hemask.bulk import BulkMaskgenClient
    from pgw.hemask.client import RemoteMulmod64
    from pgw.hemask.lwe import HEParams
    from pgw.wire.client import ClientConfig, ComputeClient, build_ssl_context

    layers, shapes = PRESETS[args.preset]
    scheme = "https" if ca is not None else "http"
    cfg = ClientConfig(base_url=f"{scheme}://127.0.0.1:{port}", ca_file=ca, request_timeout_s=600,
                       max_request_bytes=1 << 30)
    client = ComputeClient(cfg)
    params = HEParams(k=32, rank=16)
    out_features = {f"0.{k}": m for k, (m, _n) in shapes.items()}
    rng = np.random.default_rng(0)
    rows = []
    for transport in args.transports:
        for fmt in args.formats:
            bulk = None
            if transport.startswith("bulk"):
                assert bulk_port is not None
                ctx = build_ssl_context(cfg, register=False) if ca is not None else None
                bulk = BulkMaskgenClient("127.0.0.1", bulk_port, ssl_context=ctx,
                                         server_hostname="127.0.0.1" if ctx else None)
            split = int(transport.split(":")[1]) if ":" in transport else 1
            norms = {lid: 600.0 for lid in out_features} if fmt == "v2" else None
            remote = RemoteMulmod64(client, FP, out_features, q_bits=params.q, params=params if norms else None,
                                    row_norms=norms, bulk=bulk, bulk_split=split)
            if args.server_compute == "real" and fmt == "v1":
                exact = _check_exact(remote, shapes, params.q)
                print(json.dumps({"transport": transport, "exact_vs_cpu_reference": exact}), flush=True)
                rows.append({"transport": transport, "exact_vs_cpu_reference": exact})
            # the first calls learn the server's versions (v1); then v2 is used when planned
            warm_b = rng.integers(0, 1 << params.q, size=(shapes["o"][1], 256), dtype=np.uint64)
            for _ in range(2):
                remote("0.o", warm_b)
            total = 0.0
            for kind, (m, n) in shapes.items():
                if args.kinds and kind not in args.kinds:
                    continue
                lid = f"0.{kind}"
                b = rng.integers(0, 1 << params.q, size=(n, args.cols), dtype=np.uint64)
                up0, down0 = remote.bytes_up, remote.bytes_down
                t_call = []
                c0, s0 = _cpu(os.getpid()), _cpu(server_pid)
                for _ in range(args.repeat):
                    t0 = time.perf_counter()
                    remote(lid, b)
                    t_call.append(time.perf_counter() - t0)
                c1, s1 = _cpu(os.getpid()), _cpu(server_pid)
                med = statistics.median(t_call)
                up = (remote.bytes_up - up0) / args.repeat
                down = (remote.bytes_down - down0) / args.repeat
                fmt_obj = remote._use(lid, n)
                t0 = time.perf_counter()
                remote._request(lid, b, fmt_obj)
                pack_ms = (time.perf_counter() - t0) * 1e3
                row = {"transport": transport, "fmt": fmt, "kind": kind, "m": m, "n": n, "cols": args.cols,
                       "up_mb": round(up / 1e6, 2), "down_mb": round(down / 1e6, 2), "call_ms": round(med * 1e3, 1),
                       "pack_ms_1piece": round(pack_ms, 1), "mb_per_s": round((up + down) / 1e6 / med, 1),
                       "client_cpu_ms": round((c1 - c0) / args.repeat * 1e3, 1),
                       "server_cpu_ms": round((s1 - s0) / args.repeat * 1e3, 1)}
                total += med
                print(json.dumps(row), flush=True)
                rows.append(row)
            per_pass = total * layers
            summ = {"transport": transport, "fmt": fmt, "summary": True, "layers": layers, "cols": args.cols,
                    "provider_s_per_pass": round(per_pass, 3), "slots_per_s_bound": round(args.cols / per_pass, 1),
                    "bulk_calls": remote.calls_bulk}
            print(json.dumps(summ), flush=True)
            rows.append(summ)
            if bulk is not None:
                bulk.close()
            if remote._split_pool is not None:
                remote._split_pool.shutdown()
    client.close()
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="qwen2.5-0.5b", choices=sorted(PRESETS))
    ap.add_argument("--cols", type=int, default=512)
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--formats", nargs="+", default=["v1", "v2"], choices=["v1", "v2"])
    ap.add_argument("--transports", nargs="+", default=["http", "bulk", "bulk:2"],
                    help="http, bulk (one connection per request) or bulk:N (requests split into N concurrent pieces)")
    ap.add_argument("--tls", action="store_true")
    ap.add_argument("--server-compute", default="stub", choices=["stub", "real"])
    ap.add_argument("--json")
    ap.add_argument("--kinds", nargs="*")
    ap.add_argument("--profile-server")
    ap.add_argument("--profile-client")
    ap.add_argument("--_server", type=int, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args._server:
        _server(args._server, args.preset, args.server_compute)
        return
    port = _free_port()
    bulk_port = _free_port()
    env = dict(os.environ)
    env["PGW_BENCH_BULK_PORT"] = str(bulk_port)
    ca = None
    with tempfile.TemporaryDirectory() as td:
        if args.profile_server:
            env["PGW_BENCH_PROFILE_SERVER"] = args.profile_server
        if args.tls:
            from wire_overhead_bench import _make_tls

            ca, _cert, _key = _make_tls(Path(td))
            env["PGW_BENCH_TLS_DIR"] = td
        cmd = [sys.executable, __file__, "--preset", args.preset, "--server-compute", args.server_compute,
               "--_server", str(port)]
        proc = subprocess.Popen(cmd, env=env, stderr=subprocess.PIPE, text=True)  # noqa: S603 - this script
        try:
            assert proc.stderr is not None
            for line in proc.stderr:
                if "server_ready" in line:
                    break
            time.sleep(1.0)
            prof = None
            if args.profile_client:
                import cProfile

                prof = cProfile.Profile()
                prof.enable()
            rows = _client(port, args, ca, proc.pid, bulk_port)
            if prof is not None:
                prof.disable()
                prof.dump_stats(args.profile_client)
        finally:
            import signal

            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(20)
            except subprocess.TimeoutExpired:
                proc.kill()
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
