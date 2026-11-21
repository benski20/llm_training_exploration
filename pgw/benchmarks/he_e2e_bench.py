#!/usr/bin/env python3
"""End-to-end encrypted mask generation at real model shapes: provider service + gateway HE, two processes.

Package X3 of the scale plan (``research/plans/scale-north-star.md``). Two processes on one host:

* **provider**: the real compute service app (uvicorn, ``POST /pgw/v1/maskgen``) serving a synthetic
  pack of ``--preset`` shapes (random int8 weights, one weight tensor per shape shared by all layers;
  any weights give the same cost) on ``--provider-gpu`` (``CUDA_VISIBLE_DEVICES``), or the CPU.
* **gateway**, one process per ``--configs`` entry: the real :class:`pgw.factory.MaskGenerator` with
  :class:`pgw.hemask.HEMaskBackend` and :class:`pgw.hemask.client.RemoteMulmod64` over loopback HTTP,
  on ``--gateway-gpu``. The factory's every-row Freivalds self-check runs on every generated row, so
  every measured slot is checked exact against the gateway's own weights.

Gateway configs (``name[:layers]``; layers default to ``--layers``, results are per full-model slot):

* ``dense_dev``: dense byte-plane encryption and decryption as fused device passes on the gateway GPU
  (today's default on a gateway with a GPU),
* ``ntt_cpu``: ``he_encrypt=ntt_pre`` before this package (A in NTT form on the CPU, decryption on the
  CPU engines: ``device=None``),
* ``ntt_dev``: ``ntt_pre`` encryption on the CPU and decryption as device passes on the gateway GPU,
* ``*_stream``: the same with the device hint cache off (every hint pass streamed over PCIe), the
  regime of a 7B+ model whose hint (34 GB at 7B) cannot stay on a 16 GB gateway GPU.

Per config: full-model slots/s of the real generator (``wall``) and its stage seconds (encrypt, provider
round trips, decrypt from :attr:`HEMaskBackend.stats`; rng, check from the generator), the provider's own
compute seconds (service CPU time is not GPU time; the service reports no per-item timing, so the
isolated stage rates below are the attribution), and **isolated stage rates** on one layer at
``--stage-rows`` rows: encrypt only, provider round trip only (``--inflight`` requests in flight), decrypt
only, each as full-model slots/s, so the bottleneck is the smallest of them. The isolated decrypt also
checks its output bit for bit against an exact integer product on a few rows.

    PYTHONPATH=src python benchmarks/he_e2e_bench.py --preset qwen2.5-7b --layers 28 --slots 1536 \
        --configs dense_dev ntt_dev ntt_cpu:2 --provider-gpu 0 --gateway-gpu 1 --json out.json

Local smoke run (CPU only): ``--preset qwen2.5-0.5b --layers 1 --slots 256 --configs ntt_cpu --cpu``.
Also measures the hint store's O_DIRECT read rate (``--disk-read``): at 7B the hint does not fit in RAM,
so in production every decryption pass reads it from disk.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

#: Gateway configs: encryption mode, decryption device (``"auto"``: the gateway GPU when it passes the
#: self-test; ``None``: CPU engines), extra environment, and optional chunk / pass / slots overrides.
CONFIGS: dict[str, dict[str, Any]] = {
    "dense_dev": {"encrypt": "dense", "device": "auto"},
    "dense_dev_c4096": {"encrypt": "dense", "device": "auto", "chunk": 4096, "pass_mb": 2048, "slots": 4096},
    "ntt_cpu": {"encrypt": "ntt_pre", "device": None, "slots": 512},
    "ntt_dev": {"encrypt": "ntt_pre", "device": "auto"},
    "ntt_dev_lru": {"encrypt": "ntt_pre", "device": "auto", "env": {"PGW_HEMASK_GPU_HINT_STALE_S": "0"}},
    "ntt_dev_v2": {"encrypt": "ntt_pre", "device": "auto", "wire_v2": True},
    "ntt_dev_stream": {"encrypt": "ntt_pre", "device": "auto", "env": {"PGW_HEMASK_GPU_HINT_MB": "0"}},
    "ntt_dev_stream_noprefetch": {"encrypt": "ntt_pre", "device": "auto",
                                  "env": {"PGW_HEMASK_GPU_HINT_MB": "0", "PGW_HEMASK_HINT_PREFETCH": "0"}},
    "ntt_dev_c4096": {"encrypt": "ntt_pre", "device": "auto", "chunk": 4096, "pass_mb": 2048, "slots": 4096},
    "ntt_dev_c4096_stream": {"encrypt": "ntt_pre", "device": "auto", "chunk": 4096, "pass_mb": 2048, "slots": 4096,
                             "env": {"PGW_HEMASK_GPU_HINT_MB": "0"}},
}


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


# ---- provider process ----------------------------------------------------------------------------
def serve(args: argparse.Namespace) -> None:
    import asyncio
    import logging

    import torch
    from gateway_gpu_bench import _Pack

    from pgw.service.app import create_app
    from pgw.service.config import ServiceSettings
    from pgw.service.kernels import ComputeKernel
    from pgw.service.registry import ModelRegistry, RegisteredModel
    from pgw.service.server import serve_async

    logging.basicConfig(level=logging.INFO, format="%(levelname)s provider %(name)s: %(message)s")
    dev = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    pack = _Pack(args.preset, args.layers)
    shared: dict[tuple[int, int], Any] = {}
    kernels = {}
    for s in pack.specs:
        key = (s.out_features, s.in_features)
        if key not in shared:  # one device copy per shape: the GEMM cost does not depend on the values
            shared[key] = ComputeKernel(s, pack.w[key], device=dev)
        kernels[s.id.key] = shared[key]
    rings = (args.ring,)
    reg = ModelRegistry(dev, rings=rings)
    reg.register(RegisteredModel(pack.name, pack.fingerprint, tuple(pack.specs), kernels, rings=rings))
    settings = ServiceSettings(host="127.0.0.1", port=args.serve, allow_unauthenticated=True, rings=list(rings))
    app = create_app(settings, registry=reg)
    Path(args.ready_file).write_text(json.dumps({"pid": os.getpid(), "device": dev,
                                                 "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else None}))
    asyncio.run(serve_async(settings, app=app))


# ---- gateway process -----------------------------------------------------------------------------
class _TimedRemote:
    """Wraps the remote: sums each request's round-trip seconds and the union of busy time."""

    def __init__(self, remote: Any):
        self.remote = remote
        self.lock = threading.Lock()
        self.call_s = 0.0
        self.calls = 0
        self._active = 0
        self._t_busy0 = 0.0
        self.busy_s = 0.0  # wall seconds with at least one request in flight

    def __call__(self, linear: str, b: Any) -> Any:
        t0 = time.perf_counter()
        with self.lock:
            if self._active == 0:
                self._t_busy0 = t0
            self._active += 1
        try:
            return self.remote(linear, b)
        finally:
            t1 = time.perf_counter()
            with self.lock:
                self.call_s += t1 - t0
                self.calls += 1
                self._active -= 1
                if self._active == 0:
                    self.busy_s += t1 - self._t_busy0


def _hint_store(root: Path, pack: Any, rank: int, q_bits: int) -> tuple[Any, float]:
    from gateway_gpu_bench import _hints

    from pgw.hemask.backend import HintStore

    hs = HintStore.open(root, pack.fingerprint, rank, q_bits)
    if hs is not None:
        return hs, 0.0
    t0 = time.perf_counter()
    hs = _hints(root, pack, rank, q_bits)
    return hs, time.perf_counter() - t0


def _exact_rows(w: Any, r: Any, k: int) -> Any:
    """``(r @ w^T) mod 2^k`` for a few rows, exactly (int64: ``|w| < 128``, ``r < 2^k``, ``n < 2^24`` for k <= 32)."""
    import numpy as np

    if k > 32:
        acc = r.astype(object) @ w.T.astype(object)
        return np.array(acc % (1 << k), dtype=np.int64)
    return (r.astype(np.int64) @ w.T.astype(np.int64)) & ((1 << k) - 1)


def gateway(args: argparse.Namespace) -> dict[str, Any]:
    import logging

    import numpy as np
    import torch
    from gateway_gpu_bench import _Pack

    from pgw.core.prng import new_seed
    from pgw.core.ring import RingParams
    from pgw.factory import MaskGenerator
    from pgw.factory.bench import MODEL_PRESETS
    from pgw.hemask import DEGREE, fastmm
    from pgw.hemask.backend import HEMaskBackend
    from pgw.hemask.client import RemoteMulmod64
    from pgw.hemask.lwe import HEParams
    from pgw.wire import ClientConfig, ComputeClient

    logging.basicConfig(level=logging.INFO, format="%(levelname)s gateway %(name)s: %(message)s")
    name, _, lay = args.gateway.partition(":")
    layers = int(lay) if lay else args.layers
    cfg = CONFIGS[name]
    encrypt, device = cfg["encrypt"], cfg["device"]
    slots = int(cfg.get("slots", args.slots))
    chunk_req = cfg.get("chunk", args.chunk_slots)
    ring = RingParams.from_name(args.ring)
    p = HEParams(k=ring.k, rank=args.rank)
    pack = _Pack(args.preset, layers)
    hints, t_hint = _hint_store(Path(args.work) / "hints", pack, args.rank, p.q)
    client = ComputeClient(ClientConfig(base_url=f"http://127.0.0.1:{args.port}", request_timeout_s=900.0,
                                        max_request_bytes=1 << 30))
    out_f = {s.id.key: s.out_features for s in pack.specs}
    kw: dict[str, Any] = {}
    if args.wire_v2 or cfg.get("wire_v2"):
        kw = {"params": p, "row_norms": hints.meta["row_norms"]}
    remote = _TimedRemote(RemoteMulmod64(client, pack.fingerprint, out_f, timeout_s=900.0, q_bits=p.q, **kw))
    t0 = time.perf_counter()
    be = HEMaskBackend(pack, ring, hints, remote, groups_per_call=args.groups_per_call, max_inflight=args.inflight,
                       encrypt=encrypt, device=device, pass_bytes=int(cfg.get("pass_mb", 1024)) << 20)
    t_init = time.perf_counter() - t0
    gdev = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    gen = MaskGenerator(pack, ring, device=gdev, remote=be)
    chunk = gen.chunk_slots_for(chunk_req)
    scale = MODEL_PRESETS[args.preset].layers / layers
    res: dict[str, Any] = {"config": args.gateway, "encrypt": encrypt, "device_arg": device, "engine": be.engine,
                           "fastmm_engine": fastmm.engine(), "layers": layers, "scale_to_full_model": scale,
                           "chunk_slots": chunk, "pass_groups_widest": be._pass_groups(max(s.out_features for s in pack.specs)),
                           "hint_build_s": round(t_hint, 1), "init_s": round(t_init, 2),
                           "hint_gb_per_layer": round(sum(hints.hint(s.id.key).nbytes for s in pack.specs)
                                                      / layers / 1e9, 3),
                           "env": {k: v for k, v in os.environ.items() if k.startswith(("PGW_HEMASK", "PGW_NTT"))}}
    t0 = time.perf_counter()
    for _ in gen.generate(new_seed(), min(args.warmup_slots, slots), checks=2):  # uploads, compiles, v2 switch
        pass
    res["warmup_s"] = round(time.perf_counter() - t0, 2)
    he0 = dict(be.stats)
    r0 = (remote.call_s, remote.busy_s, remote.calls)
    w0 = (remote.remote.calls_v2, remote.remote.bytes_up, remote.remote.bytes_down)
    dev0 = dict(be._dev.stats) if be._dev is not None else {}
    cpu0 = os.times()
    t0 = time.perf_counter()
    for _ in gen.generate(new_seed(), slots, checks=2, chunk_slots=chunk):
        pass
    wall = time.perf_counter() - t0
    cpu1 = os.times()
    res["gateway_cpu_cores_busy"] = round((cpu1.user + cpu1.system - cpu0.user - cpu0.system) / wall, 2)
    res["window"] = [round(time.time() - wall, 2), round(time.time(), 2)]
    st = gen.last_stats
    he = {k: round(be.stats[k] - he0[k], 3) for k in he0 if isinstance(he0[k], int | float)}
    per_slot = lambda s: s * scale / slots  # noqa: E731
    res["e2e"] = {
        "slots": slots, "wall_s": round(wall, 3),
        "slots_per_s_full_model": round(slots / wall / scale, 2),
        "self_checked_rows": st.self_checked_rows,
        "ms_per_slot_full_model": {
            "wall": round(1e3 * per_slot(wall), 3),
            "encrypt": round(1e3 * per_slot(he["encrypt_s"]), 3),
            "decrypt": round(1e3 * per_slot(he["decrypt_s"]), 3),
            "provider_busy": round(1e3 * per_slot(remote.busy_s - r0[1]), 3),
            "provider_call_sum": round(1e3 * per_slot(remote.call_s - r0[0]), 3),
            "rng": round(1e3 * per_slot(st.rng_seconds), 3),
            "check": round(1e3 * per_slot(st.check_seconds), 3),
            "backend_wall": round(1e3 * per_slot(he["wall_s"]), 3),
        },
        "he": he, "calls": remote.calls - r0[2],
        "wire_v2_calls": remote.remote.calls_v2 - w0[0],
        "bytes_up_per_slot_full_model": round((remote.remote.bytes_up - w0[1]) * scale / slots),
        "bytes_down_per_slot_full_model": round((remote.remote.bytes_down - w0[2]) * scale / slots),
        "device_stats": ({k: v - dev0.get(k, 0) for k, v in be._dev.stats.items()} if be._dev is not None else None),
        "hint_cache_gib": round(be._dev.cache.used / 2**30, 2) if be._dev is not None else None,
    }
    print("E2E " + json.dumps(res["e2e"]), flush=True)
    if args.stage_rows:
        res["stages"] = _stages(args, be, remote, pack, ring, p, scale_layers=MODEL_PRESETS[args.preset].layers)
    return res


def _stages(args: argparse.Namespace, be: Any, remote: Any, pack: Any, ring: Any, p: Any,
            scale_layers: int) -> dict[str, Any]:
    """Isolated encrypt / provider / decrypt rates on layer 0 (full-model slots/s each)."""
    import numpy as np

    from pgw.hemask import DEGREE

    rows = max(DEGREE, args.stage_rows // DEGREE * DEGREE)
    rng = np.random.default_rng(5)
    enc_s = prov_s = dec_s = 0.0
    exact = True
    per_linear: dict[str, Any] = {}
    for s in pack.specs[:4]:
        key, n, m = s.id.key, s.in_features, s.out_features
        r = rng.integers(0, 1 << ring.k, size=(rows, n), dtype=np.int64)
        t0 = time.perf_counter()
        cts = be._encrypt(r, n)
        te = time.perf_counter() - t0
        try:
            step = len(cts) if args.groups_per_call <= 0 else args.groups_per_call
            batches = [cts[i:i + step] for i in range(0, len(cts), step)]
            bs = [np.concatenate([ct.b for ct in bt], axis=1) for bt in batches]
            t0 = time.perf_counter()
            with ThreadPoolExecutor(max(1, args.inflight)) as ex:
                ys = list(ex.map(lambda b: remote(key, b), bs))
            tp = time.perf_counter() - t0
            out = np.empty((rows, m), dtype=np.int64)
            t0 = time.perf_counter()
            items, s0 = [], 0
            for y, bt in zip(ys, batches, strict=True):
                items.append((y, bt, s0))
                s0 += sum(ct.rows for ct in bt)
            for i0 in range(0, len(items), max(1, be._pass_groups(m) // max(1, step))):
                be._decrypt_into(key, items[i0:i0 + max(1, be._pass_groups(m) // max(1, step))], out, m)
            td = time.perf_counter() - t0
        finally:
            for ct in cts:
                ct.wipe()
        w = pack.w[(m, n)].numpy()
        pick = rng.choice(rows, size=3, replace=False)
        ok = bool(np.array_equal(out[pick], _exact_rows(w, r[pick], ring.k)))
        exact &= ok
        per_linear[s.id.kind.value] = {"encrypt_s": round(te, 3), "provider_s": round(tp, 3), "decrypt_s": round(td, 3),
                                       "exact_rows_ok": ok}
        enc_s += te
        prov_s += tp
        dec_s += td
    rate = lambda t: round(rows / (t * scale_layers), 2)  # noqa: E731
    res = {"rows": rows, "per_linear_layer0": per_linear, "exact": exact,
           "slots_per_s_full_model": {"encrypt": rate(enc_s), "provider": rate(prov_s), "decrypt": rate(dec_s)}}
    print("STAGES " + json.dumps(res), flush=True)
    return res


# ---- driver --------------------------------------------------------------------------------------
def _proc_cpu_s(pid: int) -> float:
    with open(f"/proc/{pid}/stat") as f:
        parts = f.read().rsplit(")", 1)[1].split()
    return (int(parts[11]) + int(parts[12])) / os.sysconf("SC_CLK_TCK")


class _Sampler:
    """Background samples (epoch time) of the provider's CPU seconds and every GPU's utilisation/memory."""

    def __init__(self, provider_pid: int, gpus: bool, period_s: float = 0.5):
        self.pid = provider_pid
        self.cpu: list[tuple[float, float]] = []
        self.gpu: list[tuple[float, int, float, float]] = []  # (t, index, util %, memory MiB)
        self._stop = threading.Event()
        self._smi = None
        if gpus:
            try:
                self._smi = subprocess.Popen(  # noqa: S603
                    ["nvidia-smi", "--query-gpu=timestamp,index,utilization.gpu,memory.used", "--format=csv,noheader,nounits",
                     f"-lms={int(period_s * 1000)}"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
                threading.Thread(target=self._read_smi, daemon=True).start()
            except OSError:
                self._smi = None
        threading.Thread(target=self._run, args=(period_s,), daemon=True).start()

    def _read_smi(self) -> None:
        assert self._smi is not None and self._smi.stdout is not None
        for line in self._smi.stdout:
            try:
                _ts, idx, util, mem = (x.strip() for x in line.split(","))
                self.gpu.append((time.time(), int(idx), float(util), float(mem)))
            except ValueError:
                continue

    def _run(self, period_s: float) -> None:
        while not self._stop.wait(period_s):
            try:
                self.cpu.append((time.time(), _proc_cpu_s(self.pid)))
            except OSError:
                return

    def summary(self, t0: float, t1: float) -> dict[str, Any]:
        cpu = [c for c in self.cpu if t0 <= c[0] <= t1]
        out: dict[str, Any] = {}
        if len(cpu) >= 2:
            out["provider_cpu_cores_busy"] = round((cpu[-1][1] - cpu[0][1]) / max(1e-9, cpu[-1][0] - cpu[0][0]), 2)
        by_gpu: dict[int, list[tuple[float, float]]] = {}
        for t, idx, util, mem in self.gpu:
            if t0 <= t <= t1:
                by_gpu.setdefault(idx, []).append((util, mem))
        for idx, v in sorted(by_gpu.items()):
            out[f"gpu{idx}_util_pct_mean"] = round(sum(u for u, _ in v) / len(v), 1)
            out[f"gpu{idx}_mem_mib_max"] = max(m for _, m in v)
        return out

    def stop(self) -> None:
        self._stop.set()
        if self._smi is not None:
            self._smi.terminate()


def _disk_read(path: Path, mb: int = 2048) -> dict[str, Any]:
    """Sequential O_DIRECT read rate of a hint file (bypasses the page cache)."""
    files = sorted(path.rglob("*.npy"), key=lambda f: -f.stat().st_size)
    if not files:
        return {}
    f = files[0]
    cmd = ["dd", f"if={f}", "of=/dev/null", "bs=16M", f"count={max(1, mb // 16)}", "iflag=direct"]
    t0 = time.perf_counter()
    r = subprocess.run(cmd, capture_output=True, text=True, check=False)  # noqa: S603
    dt = time.perf_counter() - t0
    size = min(f.stat().st_size, mb << 20)
    return {"file_gb": round(f.stat().st_size / 1e9, 2), "read_gb": round(size / 1e9, 2),
            "gb_per_s": round(size / 1e9 / dt, 2), "rc": r.returncode, "dd": r.stderr.strip().splitlines()[-1:]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--preset", default="qwen2.5-7b")
    ap.add_argument("--ring", default="K32A12C27")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--layers", type=int, default=28)
    ap.add_argument("--slots", type=int, default=1536)
    ap.add_argument("--warmup-slots", type=int, default=256)
    ap.add_argument("--chunk-slots", type=int, default=None)
    ap.add_argument("--groups-per-call", type=int, default=4)
    ap.add_argument("--inflight", type=int, default=2)
    ap.add_argument("--stage-rows", type=int, default=1024, help="isolated stage rows on layer 0 (0: skip)")
    ap.add_argument("--configs", nargs="*", default=["dense_dev", "ntt_dev"])
    ap.add_argument("--wire-v2", action="store_true", help="offer maskgen wire v2 (the factory runtime does not)")
    ap.add_argument("--provider-gpu", default="0")
    ap.add_argument("--gateway-gpu", default="1")
    ap.add_argument("--cpu", action="store_true", help="no GPU anywhere (local smoke run)")
    ap.add_argument("--work", default=None)
    ap.add_argument("--disk-read", action="store_true")
    ap.add_argument("--json", default=None)
    ap.add_argument("--budget-s", type=float, default=0.0, help="skip configs once this much time has passed (0: none)")
    ap.add_argument("--config-timeout-s", type=float, default=1800.0)
    ap.add_argument("--serve", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--ready-file", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--gateway", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--port", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--result-file", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.serve is not None:
        serve(args)
        return
    if args.gateway is not None:
        res = gateway(args)
        Path(args.result_file).write_text(json.dumps(res))
        return
    work = Path(args.work or tempfile.mkdtemp(prefix="pgw_he_e2e_"))
    work.mkdir(parents=True, exist_ok=True)
    port = _free_port()
    ready = work / f"ready-{port}"
    base_env = dict(os.environ)
    env = dict(base_env, CUDA_VISIBLE_DEVICES="" if args.cpu else args.provider_gpu)
    common = ["--preset", args.preset, "--ring", args.ring, "--layers", str(args.layers)] + (["--cpu"] if args.cpu else [])
    log_path = work / "provider.log"
    with open(log_path, "w") as log:
        prov = subprocess.Popen([sys.executable, __file__, *common, "--serve", str(port), "--ready-file", str(ready)],  # noqa: S603
                                env=env, stdout=log, stderr=subprocess.STDOUT)
    out: dict[str, Any] = {"preset": args.preset, "ring": args.ring, "rank": args.rank, "slots": args.slots,
                           "groups_per_call": args.groups_per_call, "inflight": args.inflight,
                           "wire_v2": args.wire_v2, "host_cpus": os.cpu_count(), "configs": []}
    try:
        t0 = time.time()
        while not ready.exists():
            if prov.poll() is not None:
                raise SystemExit("provider exited:\n" + log_path.read_text()[-4000:])
            if time.time() - t0 > 1200:
                raise SystemExit("provider not ready after 1200 s")
            time.sleep(0.5)
        out["provider"] = json.loads(ready.read_text())
        out["provider"]["ready_s"] = round(time.time() - t0, 1)
        sampler = _Sampler(prov.pid, gpus=not args.cpu)
        t_start = time.time()
        for cfg in args.configs:
            if args.budget_s and time.time() - t_start > args.budget_s:
                out["configs"].append({"config": cfg, "error": "skipped: time budget"})
                continue
            name = cfg.partition(":")[0]
            if name not in CONFIGS:
                raise SystemExit(f"unknown config {name}; choose from {sorted(CONFIGS)}")
            genv = dict(base_env, CUDA_VISIBLE_DEVICES="" if args.cpu else args.gateway_gpu,
                        **CONFIGS[name].get("env", {}))
            rfile = work / f"result-{name}.json"
            cmd = [sys.executable, __file__, *common, "--gateway", cfg, "--port", str(port), "--work", str(work),
                   "--rank", str(args.rank), "--slots", str(args.slots), "--warmup-slots", str(args.warmup_slots),
                   "--groups-per-call", str(args.groups_per_call), "--inflight", str(args.inflight),
                   "--stage-rows", str(args.stage_rows), "--result-file", str(rfile)]
            if args.chunk_slots:
                cmd += ["--chunk-slots", str(args.chunk_slots)]
            if args.wire_v2:
                cmd += ["--wire-v2"]
            print(f"=== gateway config {cfg}", flush=True)
            t1 = time.time()
            try:
                rc = subprocess.run(cmd, env=genv, check=False, timeout=args.config_timeout_s).returncode  # noqa: S603
            except subprocess.TimeoutExpired:
                rc = -999
            if rc != 0 or not rfile.exists():
                out["configs"].append({"config": cfg, "error": f"exit {rc}"})
                continue
            res = json.loads(rfile.read_text())
            res["process_s"] = round(time.time() - t1, 1)
            res["during_measurement"] = sampler.summary(*res.get("window", (0, 0)))
            out["configs"].append(res)
            print("RESULT " + json.dumps(res), flush=True)
            if args.json:
                Path(args.json).write_text(json.dumps(out, indent=1))
        if args.disk_read:
            out["hint_disk_read"] = _disk_read(work / "hints")
    finally:
        if "sampler" in locals():
            sampler.stop()
        prov.terminate()
        try:
            prov.wait(30)
        except subprocess.TimeoutExpired:
            prov.kill()
        print("provider log tail:\n" + log_path.read_text()[-2500:], flush=True)
    print("SUMMARY " + json.dumps(out), flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
