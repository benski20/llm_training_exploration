"""Per-token breakdown of a masked decode over a real service with an injected network delay.

Starts a real compute service (``pgw service serve``, CPU) and a TCP delay proxy in their own
processes, then decodes greedily in this process through a real ``ComputeClient`` (WebSocket stream
channel) and ``MaskedBackend``, configured like the Modal demo gateway (``local_linears``, one float
thread, ring K32A12C27). The proxy delays every chunk by ``delay/2`` in each direction, so every round
trip pays ``delay`` on top of loopback (a stand-in for the Modal tunnel).

Per decode token it reports:

=============  ======================================================================================
``net``        injected delay x remote round trips (the network the gateway cannot change)
``service``    service request time (``pgw_service_request_seconds`` sum from its ``/metrics``)
``transport``  round-trip time seen by the backend minus ``net`` and ``service`` (client and server
               framing, event-loop wake-ups, loopback, proxy forwarding)
``mask``       gateway prepare + finish of remote trips (quantize, mask, pack, verify, unmask)
``local_lin``  exact local linears on the gateway (``local_linears``)
``head``       final norm + exact LM head
``float``      the rest of the plaintext step (attention, norms, rope, glue), sampling, hops, loop
=============  ======================================================================================

Usage::

    python benchmarks/trip_profile.py --pack PACK --delay-ms 2 17 --tokens 24 --workdir DIR [--sync]

``--pack`` defaults to a synthetic Qwen2.5-0.5B-shaped pack (24 layers, built once in ``--workdir``).
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import re
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


# ---------------------------------------------------------------------------------------- delay proxy
def _proxy_main(listen: int, target: int, delay_s: float) -> None:
    """Forward TCP 127.0.0.1:listen -> 127.0.0.1:target; each chunk is written ``delay_s`` after it was
    read, in order (blocking threads and ``time.sleep``: asyncio timers overshoot by ~20% of the delay)."""
    import queue
    import threading

    def pipe(src: socket.socket, dst: socket.socket) -> None:
        q: queue.SimpleQueue[tuple[float, bytes]] = queue.SimpleQueue()

        def writer() -> None:
            while True:
                due, data = q.get()
                if not data:
                    break
                wait = due - time.perf_counter()
                if wait > 0:
                    time.sleep(wait)
                try:
                    dst.sendall(data)
                except OSError:
                    break
            with contextlib.suppress(OSError):
                dst.shutdown(socket.SHUT_WR)

        threading.Thread(target=writer, daemon=True).start()
        while True:
            try:
                data = src.recv(1 << 20)
            except OSError:
                data = b""
            q.put((time.perf_counter() + delay_s, data))
            if not data:
                return

    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", listen))
    srv.listen()
    while True:
        c, _ = srv.accept()
        s = socket.create_connection(("127.0.0.1", target))
        for x in (c, s):
            x.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        threading.Thread(target=pipe, args=(c, s), daemon=True).start()
        threading.Thread(target=pipe, args=(s, c), daemon=True).start()


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _wait_http(url: str, proc: subprocess.Popen[bytes], timeout: float = 300.0) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"service exited with {proc.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=2) as r:  # noqa: S310 - loopback
                if r.status == 200:
                    return
        except Exception:  # noqa: S110 - not up yet
            pass
        time.sleep(0.5)
    raise TimeoutError(url)


def _service_seconds(port: int) -> float:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=5) as r:
        text = r.read().decode()
    return sum(float(m.group(1)) for m in re.finditer(r"^pgw_service_request_seconds_sum\{[^}]*\} (\S+)$", text,
                                                       re.M))


def _synthetic_pack(work: Path) -> Path:
    from transformers import Qwen2Config, Qwen2ForCausalLM

    from pgw.modelpack import pack_checkpoint

    ck, pk = work / "ckpt-q05", work / "pack-q05"
    if (pk / "manifest.json").exists():
        return pk
    import torch

    sys.path.insert(0, str(ROOT))
    from tests.conftest import _save_byte_tokenizer  # type: ignore[import-not-found]

    cfg = Qwen2Config(vocab_size=151936, hidden_size=896, intermediate_size=4864, num_hidden_layers=24,
                      num_attention_heads=14, num_key_value_heads=2, max_position_embeddings=4096,
                      tie_word_embeddings=True, torch_dtype="bfloat16")
    torch.manual_seed(0)
    Qwen2ForCausalLM(cfg).to(torch.bfloat16).eval().save_pretrained(ck, safe_serialization=True)
    _save_byte_tokenizer(ck)
    pack_checkpoint(ck, pk, name="synthetic-q05", overwrite=True)
    return pk


# ---------------------------------------------------------------------------------------- gateway side
class _Timers:
    def __init__(self) -> None:
        self.t: dict[str, float] = {}
        self.n: dict[str, int] = {}

    def wrap(self, owner: Any, attr: str, name: str) -> None:
        fn = getattr(owner, attr)
        if asyncio.iscoroutinefunction(fn):
            async def atimed(*a: Any, **k: Any) -> Any:
                t0 = time.perf_counter()
                try:
                    return await fn(*a, **k)
                finally:
                    self.t[name] = self.t.get(name, 0.0) + time.perf_counter() - t0
                    self.n[name] = self.n.get(name, 0) + 1
            setattr(owner, attr, atimed)
            return

        def timed(*a: Any, **k: Any) -> Any:
            t0 = time.perf_counter()
            try:
                return fn(*a, **k)
            finally:
                self.t[name] = self.t.get(name, 0.0) + time.perf_counter() - t0
                self.n[name] = self.n.get(name, 0) + 1
        setattr(owner, attr, timed)

    def reset(self) -> None:
        self.t.clear()
        self.n.clear()


async def _gateway(args: argparse.Namespace, url: str, token: str, svc_port: int, delay_s: float,
                   store: Any, pack: Any) -> dict[str, Any]:

    from pgw.core import RingParams
    from pgw.engine import Generator, MaskedBackend, SamplingParams, TransformerModel
    from pgw.gateway.masks import WindowedReservation
    from pgw.wire.client import ClientConfig, ComputeClient

    ring = RingParams.from_name(args.ring)
    client = ComputeClient(ClientConfig(base_url=url, token=token))
    await client.connect(pack.fingerprint, ring.name)
    if args.lane == "stream":
        from pgw.wire.sync_stream import SyncStreamClient

        client = SyncStreamClient(client)
    backend = MaskedBackend(pack, None, client, ring, pack.fingerprint, min_soundness_bits=20,
                            max_rows_per_item=int(getattr(client.model, "max_rows_per_item", 8192) or 8192))
    await asyncio.to_thread(backend.warm_weight_norms)
    await asyncio.to_thread(backend.configure_local_linears, args.local_linears, rtt_s=None)
    model = TransformerModel(pack, backend)
    model.enable_float_threads(1)
    if args.sync or args.lane == "stream":
        model.sync_decode = True
    tim = _Timers()
    tim.wrap(backend, "_prepare", "mask")
    tim.wrap(backend, "_finish", "mask")
    tim.wrap(backend, "forward_local", "local_lin")
    tim.wrap(backend, "send_prepared", "rtt")
    tim.wrap(client, "compute_sync", "rtt")
    tim.wrap(model, "_logits", "head")
    fp = pack.fingerprint
    per = 16 + args.tokens + 8

    def factory(n: int) -> Any:
        res = store.reserve(fp, ring, min(n, per))
        return res if n <= per else WindowedReservation(res, lambda s: store.reserve(fp, ring, s), cap=n, window=16)

    gen = Generator(model, None, max_batch=4, session_factory=factory)
    prompt = [(7 * i + 3) % (pack.arch.vocab_size - 1) + 1 for i in range(8)]
    params = SamplingParams(max_tokens=args.tokens, temperature=0.0, ignore_eos=True, logprobs=0)
    out = await gen.agenerate(prompt, params)  # warm-up (channel, caches, threads)
    rows = []
    for _ in range(args.repeat):
        tim.reset()
        st0 = backend.stats.snapshot()
        s0 = _service_seconds(svc_port)
        stamps: list[float] = []
        t0 = time.perf_counter()
        st = gen._new_state(prompt, params, None, False)
        async for _ev in gen._astream_state(st):
            stamps.append(time.perf_counter())
        res = gen._result(st)
        st1 = backend.stats.snapshot()
        svc = _service_seconds(svc_port) - s0
        n = len(res.token_ids)
        trips = st1["round_trips"] - st0["round_trips"]
        dec = max(1, n - 1)
        decode_wall = stamps[-1] - stamps[0] if len(stamps) > 1 else time.perf_counter() - t0
        tot_wall = time.perf_counter() - t0
        frac = decode_wall / tot_wall  # attribute components to decode by wall share (prefill is 1 pass)
        per_tok = trips / (n)  # round trips per pass (prefill counts like a token: same trip count)
        row = {
            "token_ids": res.token_ids,
            "decode_ms_per_token": 1e3 * decode_wall / dec,
            "trips_per_token": per_tok,
        }
        comp = {k: 1e3 * v / n for k, v in tim.t.items()}
        comp["service"] = 1e3 * svc / n
        comp["net"] = per_tok * delay_s * 1e3
        comp["transport"] = comp.get("rtt", 0.0) - comp["net"] - comp["service"]
        comp["float_other"] = 1e3 * tot_wall / n - sum(comp.get(k, 0.0) for k in ("rtt", "mask", "local_lin", "head"))
        row["per_pass_ms"] = {k: round(v, 3) for k, v in comp.items()}
        row["decode_share"] = round(frac, 3)
        rows.append(row)
    best = min(rows, key=lambda r: r["decode_ms_per_token"])
    lane_stats = getattr(client, "sync_stream_stats", None)
    if lane_stats is not None:
        best["lane"] = vars(lane_stats)
    best["trip_engine"] = bool(getattr(backend, "trip_engine", False))
    best["prefetched_rows"] = int(getattr(backend, "prefetched_rows", 0))
    await gen.aclose()
    model.close()
    await client.aclose()
    best["first_run_ids"] = out.token_ids
    return best


def _make_store(args: argparse.Namespace, pack: Any, work: Path, slots: int) -> Any:
    from pgw.core import RingParams
    from pgw.factory import MaskGenerator
    from pgw.maskstore import FileKeyProvider, MaskStore
    from pgw.maskstore.counters import FileMonotonicCounter

    ring = RingParams.from_name(args.ring)
    kp = FileKeyProvider(work / "kek", create=True)
    tag = time.time_ns()
    store = MaskStore.open(work / f"store-{tag}", kp, fsync=False,
                           counter=FileMonotonicCounter(work / f"counter-{tag}", fsync=False))
    t0 = time.perf_counter()
    MaskGenerator(pack, ring).generate_into_store(store, slots, chunk_slots=16, factory_id="trip-profile")
    print(f"factory: {slots} slots in {time.perf_counter() - t0:.1f}s", file=sys.stderr)
    return store


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--proxy", nargs=3, type=float, metavar=("LISTEN", "TARGET", "DELAY_S"), help=argparse.SUPPRESS)
    ap.add_argument("--pack", default=None)
    ap.add_argument("--workdir", required=False)
    ap.add_argument("--delay-ms", type=float, nargs="+", default=[2.0, 17.0])
    ap.add_argument("--tokens", type=int, default=16)
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--ring", default="K32A12C27")
    ap.add_argument("--local-linears", default="attention")
    ap.add_argument("--sync", action="store_true", help="TransformerModel.sync_decode (one plaintext-thread call "
                    "per decode pass)")
    ap.add_argument("--lane", default="off", choices=("off", "stream"), help="'stream': the blocking stream lane "
                    "(gateway lane = 'stream')")
    ap.add_argument("--env", nargs="*", default=[], help="KEY=VALUE set in this process before the run")
    args = ap.parse_args()
    if args.proxy:
        _proxy_main(int(args.proxy[0]), int(args.proxy[1]), float(args.proxy[2]))
        return
    for kv in args.env:
        k, v = kv.split("=", 1)
        os.environ[k] = v
    work = Path(args.workdir)
    work.mkdir(parents=True, exist_ok=True)
    from pgw.modelpack import ModelPack

    pack_dir = Path(args.pack) if args.pack else _synthetic_pack(work)
    pack = ModelPack.open(pack_dir)
    env = {k: v for k, v in os.environ.items() if not k.startswith("PGW_") or k in ("PGW_OFFLINE",)}
    env["PYTHONPATH"] = str(ROOT / "src")
    secrets = work / "secrets"
    if not (secrets / "service_token").exists():
        secrets.mkdir(parents=True, exist_ok=True, mode=0o700)
        subprocess.run([sys.executable, "-m", "pgw.cli", "service", "token", "create", "--tenant", "bench",  # noqa: S603
                        "--tokens-file", str(secrets / "tokens"), "--out", str(secrets / "service_token")],
                       env=env, check=True, capture_output=True)
    token = (secrets / "service_token").read_text().strip()
    svc_port = _free_port()
    svc_env = {**env, "PGW_SERVICE_HOST": "127.0.0.1", "PGW_SERVICE_PORT": str(svc_port),
               "PGW_SERVICE_PACKS": json.dumps([str(pack_dir)]), "PGW_SERVICE_RINGS": json.dumps([args.ring]),
               "PGW_SERVICE_TOKENS_FILE": str(secrets / "tokens"), "PGW_SERVICE_DEVICE": "cpu",
               "PGW_SERVICE_VERIFY_PACKS": "fingerprint", "PGW_SERVICE_METRICS_ON_MAIN_PORT": "true",
               "PGW_LOG_LEVEL": "WARNING"}
    procs: list[subprocess.Popen[bytes]] = []
    log = open(work / "service.log", "ab")
    try:
        procs.append(subprocess.Popen([sys.executable, "-m", "pgw.cli", "service", "serve"], env=svc_env,
                                      stdout=log, stderr=subprocess.STDOUT))
        _wait_http(f"http://127.0.0.1:{svc_port}/readyz", procs[0])
        reps = len(args.delay_ms) * (args.repeat + 1)
        store = _make_store(args, pack, work, reps * (16 + args.tokens + 8) + 16)
        results = {}
        for d in args.delay_ms:
            pport = _free_port()
            p = subprocess.Popen([sys.executable, __file__, "--proxy", str(pport), str(svc_port), str(d / 2e3)],  # noqa: S603
                                 env=env)
            procs.append(p)
            time.sleep(1.0)
            res = asyncio.run(_gateway(args, f"http://127.0.0.1:{pport}", token, svc_port, d / 1e3, store, pack))
            p.terminate()
            results[f"{d:g}ms"] = res
            print(json.dumps({"delay_ms": d, **{k: v for k, v in res.items() if "ids" not in k}}), flush=True)
        ids = {json.dumps(r["token_ids"]) for r in results.values()} | {json.dumps(r["first_run_ids"])
                                                                        for r in results.values()}
        print(json.dumps({"identical_tokens_across_runs": len(ids) == 1, "token_ids": results[next(iter(results))]
                          ["token_ids"]}))
        store.close()
    finally:
        for p in procs:
            p.terminate()
        for p in procs:
            with contextlib.suppress(Exception):
                p.wait(10)
        log.close()


if __name__ == "__main__":
    main()
