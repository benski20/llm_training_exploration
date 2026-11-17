"""Load generator for the compute service.

Simulates ``--sessions`` concurrent gateway sessions. Each session runs the real call pattern of
the engine: per token, for every layer, the four linear groups in order (``qkv, o, gate_up, down``),
one round trip each, ``--rows`` rows per call (1 = decode, larger = prefill chunks). Payloads are
uniform random ring elements, exactly what a gateway sends. Reports throughput, latency
percentiles and the server-side batch sizes it observed (cross-session batching at work).

Examples::

    # against a running service
    python -m pgw.service.loadtest --url https://compute:8443 --token-file tok.txt --sessions 32 --tokens 8

    # self-contained: start an in-process service for a pack on a free port, then load it
    python -m pgw.service.loadtest --pack packs/tiny --sessions 16 --tokens 4 --verify

``--verify`` recomputes every result locally from the pack (slow; small models only) and fails on
any mismatch. Exit code: 0 ok, 1 errors or mismatches.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import socket
import statistics
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pgw.core.errors import PGWError
from pgw.core.packing import pack_ring, unpack_ring
from pgw.core.ring import RingParams

__all__ = ["LoadResult", "main", "run_load"]

log = logging.getLogger(__name__)


@dataclass
class LoadResult:
    """Summary of one load run."""

    sessions: int
    tokens_per_session: int
    rows_per_call: int
    calls: int = 0
    errors: int = 0
    mismatches: int = 0
    seconds: float = 0.0
    calls_per_s: float = 0.0
    rows_per_s: float = 0.0
    tokens_per_s: float = 0.0
    latency_ms_p50: float = 0.0
    latency_ms_p95: float = 0.0
    latency_ms_p99: float = 0.0
    server_queue_ms_mean: float = 0.0
    server_compute_ms_mean: float = 0.0
    server_batch_rows_mean: float = 0.0
    server_batch_rows_max: int = 0
    bytes_up: int = 0
    bytes_down: int = 0
    error_codes: dict[str, int] = field(default_factory=dict)

    def to_json(self) -> str:
        """Pretty JSON."""
        return json.dumps(asdict(self), indent=2, sort_keys=True)


def _pct(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    i = min(len(s) - 1, max(0, round(q * (len(s) - 1))))
    return s[i]


async def run_load(
    client: Any,
    *,
    fingerprint: str | None = None,
    ring: str = "K40A12",
    sessions: int = 8,
    tokens: int = 4,
    rows: int = 1,
    layers: int | None = None,
    reference: dict[str, torch.Tensor] | None = None,
    seed: int = 0,
) -> LoadResult:
    """Run the load pattern with an already-configured :class:`pgw.wire.ComputeClient`.

    Args:
        client: Compute client (``connect`` is called here).
        fingerprint: Model to load; default: the only served model.
        ring: Ring preset name.
        sessions: Concurrent simulated sessions.
        tokens: Tokens (layer passes) per session.
        rows: Rows per call.
        layers: Restrict to the first N layers (default: all).
        reference: ``{linear: int8 W8}`` to verify every result exactly.
        seed: RNG seed for payloads.
    """
    params = RingParams.from_name(ring)
    models = await client.connect(fingerprint, ring)
    info = models.find(fingerprint) if fingerprint else (models.models[0] if models.models else None)
    if info is None:
        raise PGWError("no model to load-test")
    dims = info.linear_dims()
    order = [li.id for li in info.linears]
    if layers is not None:
        order = [lid for lid in order if int(lid.split(".", 1)[0]) < layers]
    res = LoadResult(sessions=sessions, tokens_per_session=tokens, rows_per_call=rows)
    lat: list[float] = []
    q_ms: list[float] = []
    c_ms: list[float] = []
    b_rows: list[int] = []
    from pgw.wire.messages import ComputeItem, ComputeRequest

    async def session(si: int) -> None:
        rng = np.random.default_rng(seed + si)
        for _t in range(tokens):
            for lid in order:
                _m, n = dims[lid]
                x = torch.from_numpy(rng.integers(0, params.modulus, size=(rows, n), dtype=np.int64))
                payload = pack_ring(x, params)
                item = ComputeItem(lid, rows, payload)
                req = ComputeRequest(uuid.uuid4().hex, info.fingerprint, params.name, (item,))
                t0 = time.perf_counter()
                try:
                    resp = await client.compute(req)
                except PGWError as exc:
                    res.errors += 1
                    res.error_codes[str(exc.code)] = res.error_codes.get(str(exc.code), 0) + 1
                    continue
                lat.append((time.perf_counter() - t0) * 1e3)
                res.calls += 1
                res.bytes_up += len(payload)
                res.bytes_down += resp.payload_bytes
                q_ms.append(resp.timing.queue_ms)
                c_ms.append(resp.timing.compute_ms)
                b_rows.append(int(resp.timing.batch_rows))
                if reference is not None:
                    m = dims[lid][0]
                    y = unpack_ring(resp.items[0].y_hat, (rows, m), params)
                    from pgw.core.ringmm import ringmm

                    want = ringmm(x, reference[lid], params, backend="cpu_f64")
                    if not torch.equal(y, want):
                        res.mismatches += 1

    t0 = time.perf_counter()
    await asyncio.gather(*(session(i) for i in range(sessions)))
    res.seconds = time.perf_counter() - t0
    if res.seconds > 0:
        res.calls_per_s = res.calls / res.seconds
        res.rows_per_s = res.calls * rows / res.seconds
        res.tokens_per_s = (res.calls / max(1, len(order))) * rows / res.seconds
    res.latency_ms_p50 = _pct(lat, 0.5)
    res.latency_ms_p95 = _pct(lat, 0.95)
    res.latency_ms_p99 = _pct(lat, 0.99)
    res.server_queue_ms_mean = statistics.fmean(q_ms) if q_ms else 0.0
    res.server_compute_ms_mean = statistics.fmean(c_ms) if c_ms else 0.0
    res.server_batch_rows_mean = statistics.fmean(b_rows) if b_rows else 0.0
    res.server_batch_rows_max = max(b_rows, default=0)
    return res


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


async def _amain(args: argparse.Namespace) -> LoadResult:
    from pydantic import SecretStr

    from pgw.wire.client import ClientConfig, ComputeClient

    server_task: asyncio.Task[None] | None = None
    stop = asyncio.Event()
    url = args.url
    token = None
    if args.token_file:
        token = Path(args.token_file).read_text().strip()
    elif os.environ.get("PGW_LOADTEST_TOKEN"):
        token = os.environ["PGW_LOADTEST_TOKEN"]
    if args.pack and not url:
        from pgw.service.config import ServiceSettings
        from pgw.service.server import serve_async

        port = _free_port()
        settings = ServiceSettings(
            host="127.0.0.1", port=port, packs=[Path(args.pack)], allow_unauthenticated=True,
            batch_window_ms=args.window_ms, verify_packs="fingerprint", device=args.device,
            max_inflight_requests=max(256, 4 * args.sessions),
        )
        server_task = asyncio.create_task(serve_async(settings, shutdown_event=stop))
        url = f"http://127.0.0.1:{port}"
        cfg0 = ClientConfig(base_url=url, http2=False)
        probe = ComputeClient(cfg0)
        for _ in range(600):
            if server_task.done():
                server_task.result()
            if await probe.ready():
                break
            await asyncio.sleep(0.1)
        await probe.aclose()
    if not url:
        raise SystemExit("either --url or --pack is required")
    cfg = ClientConfig(
        base_url=url, token=SecretStr(token) if token else None, http2=not url.startswith("http://"),
        ca_file=Path(args.ca_file) if args.ca_file else None,
        client_cert_file=Path(args.cert) if args.cert else None,
        client_key_file=Path(args.key) if args.key else None,
        max_connections=max(16, args.sessions), request_timeout_s=args.timeout,
    )
    reference: dict[str, torch.Tensor] | None = None
    if args.verify:
        if not args.pack:
            raise SystemExit("--verify needs --pack")
        from pgw.modelpack import ModelPack

        pack = ModelPack.open(args.pack, components=("linears",))
        reference = {lw.id.key: lw.w8.clone() for lw in pack.iter_linears()}
        pack.close()
    client = ComputeClient(cfg)
    try:
        return await run_load(
            client, fingerprint=args.fingerprint, ring=args.ring, sessions=args.sessions, tokens=args.tokens,
            rows=args.rows, layers=args.layers, reference=reference, seed=args.seed,
        )
    finally:
        await client.aclose()
        if server_task is not None:
            stop.set()
            await server_task


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; prints a JSON summary."""
    ap = argparse.ArgumentParser(prog="python -m pgw.service.loadtest", description=__doc__.split("\n\n")[0])
    ap.add_argument("--url", help="service base URL (https://host:8443)")
    ap.add_argument("--pack", help="model pack: start an in-process service (unless --url) and/or --verify")
    ap.add_argument("--fingerprint", help="model fingerprint (default: the only served model)")
    ap.add_argument("--ring", default="K40A12")
    ap.add_argument("--sessions", type=int, default=8)
    ap.add_argument("--tokens", type=int, default=4)
    ap.add_argument("--rows", type=int, default=1, help="rows per call (1 = decode)")
    ap.add_argument("--layers", type=int, help="only the first N layers")
    ap.add_argument("--window-ms", type=float, default=2.0, help="batch window of the in-process service")
    ap.add_argument("--device", default="auto", help="device of the in-process service")
    ap.add_argument("--token-file", help="file with the bearer token (or env PGW_LOADTEST_TOKEN)")
    ap.add_argument("--ca-file")
    ap.add_argument("--cert", help="mTLS client certificate")
    ap.add_argument("--key", help="mTLS client key")
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verify", action="store_true", help="check every result against the pack (slow)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    try:
        res = asyncio.run(_amain(args))
    except PGWError as exc:
        print(f"loadtest failed: {exc.code}: {exc.message}", file=sys.stderr)
        return 1
    print(res.to_json())
    return 1 if (res.errors or res.mismatches) else 0


if __name__ == "__main__":
    sys.exit(main())
