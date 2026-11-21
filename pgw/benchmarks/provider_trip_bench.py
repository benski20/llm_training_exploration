#!/usr/bin/env python3
"""Provider (compute service) time per round trip at the Modal demo's shapes, on the GPU.

The Modal demo (Qwen2.5-0.5B, 24 layers, qkv and o computed on the gateway) makes two sequential
1-row trips per layer (``gate_up`` 9728x896, ``down`` 896x4864). This measures what the provider
adds to each of them, in two parts:

1. **kernel** (in process): :meth:`pgw.service.ComputeKernel.run_compact` (host int64 rows in, host
   int32 result out: staging copies, kernel, stream sync; what the service's request path calls)
   for the fused Triton ring GEMV, the eager digit-GEMM path and the digit-GEMM path with CUDA
   graphs, at 1-16 rows; the device-only time of the fused kernel (CUDA events); and the fused
   kernel's launch configuration and variant on this GPU (self-check verdict, rows served, time).
   Every timed result is compared bit for bit between the paths.
2. **service** (``--service``): the real service app (uvicorn, stream channel, optionally TLS) in a
   subprocess, with real CUDA kernels on random int8 weights at the 0.5B shapes for every layer,
   and a :class:`pgw.wire.ComputeClient` running the demo's trip sequence (per token: for each
   layer, ``gate_up`` then ``down``, 1 row). Reports p50/p90 per trip, the service's CPU time per
   trip and its peak RSS, for each ``--cores`` setting of the service process (CPU affinity: the
   provider container has 2 cores with a GPU). Loopback, so the network time is ~0.

Usage (GPU host, from ``pgw/``)::

    PYTHONPATH=src python benchmarks/provider_trip_bench.py --json out.json
    PYTHONPATH=src python benchmarks/provider_trip_bench.py --service --tls --cores 0 0,1 all --tokens 30

The weights are random: only shapes matter for time, and the comparisons are exact either way.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

FP = "cd" * 32
RING = "K32A12C27"
#: Qwen2.5-0.5B linear shapes (m, n); the demo sends only gate_up and down (qkv and o stay local)
SHAPES = {"qkv": (1152, 896), "o": (896, 896), "gate_up": (9728, 896), "down": (896, 4864)}
TRIP_KINDS = ("gate_up", "down")
LAYERS = 24


def _host_ms(fn: Any, xs: list[Any], iters: int) -> tuple[float, float]:
    for x in xs[:3]:
        fn(x)
    samples = []
    for i in range(iters):
        t0 = time.perf_counter()
        fn(xs[i % len(xs)])
        samples.append(time.perf_counter() - t0)
    samples.sort()
    return statistics.median(samples) * 1e3, samples[int(0.9 * (len(samples) - 1))] * 1e3


def _dev_ms(fn: Any, iters: int) -> float:
    import torch

    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end)) / iters


def kernel_part(args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np
    import torch

    from pgw.core import LinearId, LinearKind, LinearSpec, RingParams
    from pgw.core import triton_gemv as tg
    from pgw.service.gpu_exec import device_executor, from_compact
    from pgw.service.kernels import ComputeKernel

    dev = torch.device(args.device)
    ring = RingParams.from_name(RING)
    k = ring.k
    t0 = time.perf_counter()
    ok = tg.self_check(dev)
    check_s = time.perf_counter() - t0
    props = torch.cuda.get_device_properties(dev)
    key = tg._key(dev)
    info: dict[str, Any] = {
        "gpu": props.name, "capability": list(torch.cuda.get_device_capability(dev)),
        "sms": props.multi_processor_count,
        "torch": torch.__version__, "triton": _triton_version(), "self_check": ok, "self_check_s": round(check_s, 2),
        "tensor_core_variant": key not in tg._NO_DOT, "dot_unsupported": tg._dot_unsupported(dev),
        "max_rows": tg.max_rows(dev), "stream_rows": tg.multirow_rows(dev),
        "table_rules": [str(r.config) + " " + r.note for r in tg._RULES.get(key, ())],
    }
    # does the int8 tensor-core variant compile here at all? (direct launch, bypassing the self-check's choice)
    try:
        xp = torch.zeros((16, 64), dtype=torch.int32, device=dev)
        tg.ring_gemv(xp, torch.zeros((32, 64), dtype=torch.int8, device=dev), 32,
                     config=tg.GemvConfig(16, 16, 32, 1, dot=True))
        torch.cuda.synchronize(dev)
        info["dot_probe"] = "compiled and ran"
    except Exception as exc:
        lines = str(exc).strip().splitlines()
        info["dot_probe"] = f"{type(exc).__name__}: {lines[-1][:300] if lines else ''}"
    print(json.dumps(info), flush=True)
    gen = torch.Generator().manual_seed(0)
    rng = np.random.default_rng(0)
    rows_list = [int(r) for r in args.rows.split(",")]
    out: list[dict[str, Any]] = []
    for name, (m, n) in SHAPES.items():
        w8 = torch.randint(-128, 128, (m, n), dtype=torch.int8, generator=gen)
        spec = LinearSpec(LinearId(0, LinearKind(name)), m, n)
        kern = {"triton": ComputeKernel(spec, w8, device=dev, cuda_graphs=False, triton_gemv=True),
                "eager": ComputeKernel(spec, w8, device=dev, cuda_graphs=False, triton_gemv=False),
                "graphs": ComputeKernel(spec, w8, device=dev, cuda_graphs=True, triton_gemv=False)}
        wpad = kern["eager"]._pw._w_dev  # type: ignore[union-attr]
        for rows in rows_list:
            xs = [torch.from_numpy(rng.integers(0, ring.modulus, size=(rows, n), dtype=np.int64)) for _ in range(4)]
            ref = [from_compact(kern["eager"].run_compact(x, ring), k) for x in xs]
            rec: dict[str, Any] = {"linear": name, "m": m, "n": n, "rows": rows}
            for label, kk in kern.items():
                for x, want in zip(xs, ref, strict=True):
                    if not torch.equal(from_compact(kk.run_compact(x, ring), k), want):
                        raise SystemExit(f"MISMATCH {name} rows={rows} {label}")
                p50, p90 = _host_ms(lambda x, kk=kk: kk.run_compact(x, ring), xs, args.iters)
                rec[f"{label}_ms"] = round(p50, 4)
                rec[f"{label}_p90_ms"] = round(p90, 4)
            rec["graphs_captured"] = kern["graphs"].cuda_graphs
            if rows <= tg.max_rows(dev):
                x32 = tg.low32(xs[0]).to(dev)
                cfg = tg.gemv_config(rows, m, int(wpad.shape[1]), dev)
                rec["triton_dev_ms"] = round(_dev_ms(lambda x32=x32, cfg=cfg, w=wpad, n=n, m=m: tg.ring_gemv(
                    x32, w, k, n=n, m=m, config=cfg), args.iters), 4)
                rec["config"] = str(cfg)
                rec["weight_gbs"] = round(m * n / (rec["triton_dev_ms"] * 1e-3) / 1e9, 1)
            rec["served_by"] = "triton" if kern["triton"].triton_gemv and rows <= tg.max_rows(dev) else "digit"
            print(json.dumps(rec), flush=True)
            out.append(rec)
        del kern
        torch.cuda.empty_cache()
    trip = {r["rows"]: 0.0 for r in out}
    for r in out:
        if r["linear"] in TRIP_KINDS:
            trip[r["rows"]] += r["triton_ms"] if r["served_by"] == "triton" else min(r["eager_ms"], r["graphs_ms"])
    summary = {"service_call_ms_per_layer_trips": {str(k_): round(v, 4) for k_, v in trip.items()},
               "service_call_ms_per_token_1row": round(trip.get(1, 0.0) * LAYERS, 3),
               "executor_stats": str(device_executor(dev).stats)}
    print(json.dumps(summary), flush=True)
    return {"info": info, "calls": out, "summary": summary}


def _triton_version() -> str | None:
    try:
        import triton

        return str(triton.__version__)
    except Exception:
        return None


# ---- service part --------------------------------------------------------------------------------
def _pki(d: Path) -> tuple[Path, Path, Path]:
    """A throwaway CA and a server certificate for 127.0.0.1 (cert, key, ca)."""
    import datetime as dt
    import ipaddress

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    now = dt.datetime.now(dt.UTC)
    ca_key = ec.generate_private_key(ec.SECP256R1())
    ca_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "bench CA")])
    ca = (x509.CertificateBuilder().subject_name(ca_name).issuer_name(ca_name).public_key(ca_key.public_key())
          .serial_number(x509.random_serial_number()).not_valid_before(now - dt.timedelta(minutes=5))
          .not_valid_after(now + dt.timedelta(days=1))
          .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
          .add_extension(x509.KeyUsage(digital_signature=True, content_commitment=False, key_encipherment=False,
                                       data_encipherment=False, key_agreement=False, key_cert_sign=True,
                                       crl_sign=True, encipher_only=False, decipher_only=False), critical=True)
          .add_extension(x509.SubjectKeyIdentifier.from_public_key(ca_key.public_key()), critical=False)
          .sign(ca_key, hashes.SHA256()))
    key = ec.generate_private_key(ec.SECP256R1())
    cert = (x509.CertificateBuilder().subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "127.0.0.1")]))
            .issuer_name(ca_name).public_key(key.public_key()).serial_number(x509.random_serial_number())
            .not_valid_before(now - dt.timedelta(minutes=5)).not_valid_after(now + dt.timedelta(days=1))
            .add_extension(x509.SubjectAlternativeName([x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]),
                           critical=False)
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()), critical=False)
            .sign(ca_key, hashes.SHA256()))
    paths = (d / "server.pem", d / "server.key", d / "ca.pem")
    paths[0].write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    paths[1].write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                           serialization.NoEncryption()))
    paths[2].write_bytes(ca.public_bytes(serialization.Encoding.PEM))
    return paths


def _server(port: int, device: str, tls_dir: str | None, ready_file: str) -> None:
    import logging

    import torch

    from pgw.core.types import LinearId, LinearKind, LinearSpec
    from pgw.service.app import create_app
    from pgw.service.config import ServiceSettings
    from pgw.service.kernels import ComputeKernel
    from pgw.service.registry import ModelRegistry, RegisteredModel
    from pgw.service.server import serve_async

    logging.basicConfig(level=logging.INFO)
    gen = torch.Generator().manual_seed(1)
    specs, kernels = [], {}
    for layer in range(LAYERS):
        for kind in TRIP_KINDS:
            m, n = SHAPES[kind]
            spec = LinearSpec(LinearId(layer, LinearKind(kind)), m, n)
            specs.append(spec)
            kernels[spec.id.key] = ComputeKernel(spec, torch.randint(-128, 128, (m, n), dtype=torch.int8,
                                                                     generator=gen), device=device)
    reg = ModelRegistry(device, rings=(RING,))
    reg.register(RegisteredModel("bench05b", FP, tuple(specs), kernels, rings=(RING,)))
    kw: dict[str, Any] = {}
    if tls_dir:
        kw["tls"] = {"cert_file": str(Path(tls_dir) / "server.pem"), "key_file": str(Path(tls_dir) / "server.key")}
    # the provider's settings: defaults (no idle batch window, inline decode batches, stream channel)
    settings = ServiceSettings(host="127.0.0.1", port=port, allow_unauthenticated=True, **kw)
    app = create_app(settings, registry=reg)
    Path(ready_file).write_text(json.dumps({"pid": os.getpid(), "affinity": sorted(os.sched_getaffinity(0)),
                                            "torch_threads": torch.get_num_threads()}))
    asyncio.run(serve_async(settings, app=app))


def _proc_cpu(pid: int) -> float:
    with open(f"/proc/{pid}/stat") as f:
        parts = f.read().rsplit(")", 1)[1].split()
    return (int(parts[11]) + int(parts[12])) / os.sysconf("SC_CLK_TCK")


def _rss_mib(pid: int) -> float | None:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return round(int(line.split()[1]) / 1024, 1)
    except OSError:
        pass
    return None


async def _client(port: int, tls_ca: Path | None, server_pid: int, tokens: int) -> dict[str, Any]:
    import numpy as np

    from pgw.wire.client import ClientConfig, ComputeClient
    from pgw.wire.messages import ComputeItem, ComputeRequest

    scheme = "https" if tls_ca else "http"
    cfg = ClientConfig(base_url=f"{scheme}://127.0.0.1:{port}", http2=False, stream=True, ca_file=tls_ca)
    client = ComputeClient(cfg)
    for _ in range(1200):
        if await client.ready():
            break
        await asyncio.sleep(0.25)
    await client.connect(FP, RING)
    rng = np.random.default_rng(0)
    payload = {kind: rng.integers(0, 1 << 32, size=(1, SHAPES[kind][1]), dtype=np.uint64).astype("<u4").tobytes()
               for kind in TRIP_KINDS}
    order = [(f"{layer}.{kind}", kind) for layer in range(LAYERS) for kind in TRIP_KINDS]

    async def run(n_tokens: int, lat: list[float] | None) -> None:
        for _ in range(n_tokens):
            for lid, kind in order:
                req = ComputeRequest(uuid.uuid4().hex, FP, RING, (ComputeItem(lid, 1, payload[kind]),))
                t0 = time.perf_counter()
                await client.compute(req)
                if lat is not None:
                    lat.append((time.perf_counter() - t0) * 1e3)

    await run(3, None)  # warm-up: first-call cross-checks, compiles
    lat: list[float] = []
    c0 = _proc_cpu(server_pid)
    t0 = time.perf_counter()
    await run(tokens, lat)
    wall = time.perf_counter() - t0
    c1 = _proc_cpu(server_pid)
    await client.aclose()
    lat.sort()
    return {"trips": len(lat), "p50_ms": round(lat[len(lat) // 2], 4), "p90_ms": round(lat[int(0.9 * len(lat))], 4),
            "p99_ms": round(lat[int(0.99 * len(lat))], 4), "mean_ms": round(statistics.fmean(lat), 4),
            "ms_per_token": round(wall / tokens * 1e3, 2),
            "service_cpu_ms_per_trip": round((c1 - c0) / len(lat) * 1e3, 4),
            "service_peak_rss_mib": _rss_mib(server_pid)}


def service_part(args: argparse.Namespace) -> list[dict[str, Any]]:
    import socket

    results = []
    with tempfile.TemporaryDirectory() as td:
        tls = _pki(Path(td)) if args.tls else None
        for cores in args.cores:
            with socket.socket() as s:
                s.bind(("127.0.0.1", 0))
                port = int(s.getsockname()[1])
            ready = Path(td) / f"ready-{port}"
            cmd = [sys.executable, __file__, "--serve", str(port), "--device", args.device, "--ready-file", str(ready)]
            if tls:
                cmd += ["--tls-dir", td]
            if cores != "all":
                cmd = ["taskset", "-c", cores, *cmd]
            env = dict(os.environ)
            if args.omp_threads:
                env["OMP_NUM_THREADS"] = str(args.omp_threads)
            log = open(Path(td) / f"service-{port}.log", "w")
            proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)  # noqa: S603
            try:
                t0 = time.time()
                while not ready.exists():
                    if proc.poll() is not None:
                        raise SystemExit(f"service exited: {(Path(td) / f'service-{port}.log').read_text()[-3000:]}")
                    if time.time() - t0 > 900:
                        raise SystemExit("service not ready after 900 s")
                    time.sleep(0.5)
                meta = json.loads(ready.read_text())
                res = asyncio.run(_client(port, tls[2] if tls else None, proc.pid, args.tokens))
                res.update(cores=cores, tls=bool(tls), service=meta, host_cpus=os.cpu_count())
                print(json.dumps(res), flush=True)
                results.append(res)
            finally:
                proc.terminate()
                try:
                    proc.wait(30)
                except subprocess.TimeoutExpired:
                    proc.kill()
                log.close()
                tail = (Path(td) / f"service-{port}.log").read_text()[-2500:]
                print("service log tail:", tail, flush=True)
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--rows", default="1,2,4,8,16")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--no-kernel", action="store_true", help="skip the in-process kernel part")
    ap.add_argument("--service", action="store_true", help="also run the service part")
    ap.add_argument("--tls", action="store_true", help="service part over TLS (as on Modal)")
    ap.add_argument("--cores", nargs="*", default=["0", "0,1", "all"], help="service CPU affinity per run")
    ap.add_argument("--omp-threads", type=int, default=0, help="OMP_NUM_THREADS for the service (0: unset)")
    ap.add_argument("--tokens", type=int, default=20)
    ap.add_argument("--json", default=None)
    ap.add_argument("--serve", type=int, help=argparse.SUPPRESS)
    ap.add_argument("--tls-dir", help=argparse.SUPPRESS)
    ap.add_argument("--ready-file", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.serve:
        _server(args.serve, args.device, args.tls_dir, args.ready_file)
        return 0
    out: dict[str, Any] = {}
    if not args.no_kernel:
        out["kernel"] = kernel_part(args)
    if args.service:
        out["service"] = service_part(args)
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
