#!/usr/bin/env python3
"""Measurement plan for the RunPod H100 session: provider memory, per-trip device time, lane fixed cost, throughput.

Runs inside the provider pod (the provider image, ``cd /opt/pgw`` or the repository's ``pgw/``), phase by phase,
each phase's output in ``--out`` (JSON + log), with a time budget per phase and for the session. Nothing here talks
to RunPod's API (``launch.py`` creates and stops pods; its ``--max-usd`` and TTL guards apply). ``--dry-run``
prints the plan, the commands and the minutes.

Phases (``--phases``, default all, in this order):

``env``          GPU, driver, torch, Triton, ``pgw_fast`` features, the fused GEMV's self-check verdict.
``memory``       Provider memory at full model size, measured against :mod:`pgw.service.memplan`'s prediction:
                 random int8 weights of every linear (+ masked head) loaded through ``ComputeKernel`` (the service
                 path: one padded upload per linear), then warm-up of every bucket, lane graph capture (rows 1-8),
                 a ``--eager-rows`` eager call on the largest linear and on the head, a pks-v3 hint build
                 (``mulmod64``, 4096 columns) and a 4-group device key switch on the largest linear. Per stage:
                 ``memory_allocated`` / ``max_memory_allocated`` deltas, wall seconds (start-up time). 7B, 32B, and
                 72B when the plan says it fits one GPU (``--models``).
``device_time``  ``benchmarks/provider_scale_bench.py``: per-trip service time (kernel path, graph replay, persistent
                 worker) and the GEMV device floor at 7B / 32B / 72B widths, 1-8 rows, TP 1 (and 2 with 2 GPUs).
``gemv_table``   ``benchmarks/gpu_kernel_bench.py --gemv --gemv-sweep``: the fused GEMV's launch table for this GPU
                 (emitted as JSON for ``PGW_SERVICE_TRITON_GEMV_TABLE``; every rule self-checked at load).
``lane_local``   ``benchmarks/lane_gpu_trip_bench.py``: lane trips over UDS / shm next to an idle, a Python-path
                 and a maskgen-GEMM-loaded GPU (lane overhead without the NIC).
``maskgen``      ``benchmarks/maskgen_provider_bench.py``: encrypted maskgen throughput (wire v2, fused) at 7B and
                 70B widths, 1/4/16 groups, 1/2/4 concurrent items, and decode latency next to it. Then pks-v3
                 items in process (:func:`phase_pks_items`) on the largest linears of ``--models``: the hint cache
                 path (miss: new seed, hit: same seed) against the per-item device hint
                 (``PGW_SERVICE_PKS_DEVICE_HINT``) at 512/1024/2048 MB work budgets with the key switch's NTTs
                 as ``PGW_SERVICE_PKS_GPU_NTT=auto`` picks them (the fused Triton kernels after their self-test),
                 and at 512 MB with the eager torch NTTs: seconds, peak memory, the NTT in use, byte-identical
                 replies, next to ``memplan.pks_item_seconds``'s estimates (torch and fused NTTs).
``lane_nic``     ``benchmarks/lane_nic_bench.py serve`` for ``--nic-minutes`` (7B, then 72B widths): run the printed
                 ``client`` command in the client pod (``launch.py --client-pod``, same data center) meanwhile; it
                 reports the lane's fixed cost per trip over the real NIC and throughput at 1-64 connections.

Order and budget: ``env`` 1 min, ``memory`` ~20 min (72B: ~67 GB of random weights), ``device_time`` ~12 min,
``gemv_table`` ~6 min, ``lane_local`` ~10 min, ``maskgen`` ~30 min, ``lane_nic`` 2 x ``--nic-minutes``. About 95 min
of one H100 (~$3 at ~$2.5/h Secure Cloud) plus the client CPU pod. The decisions each number feeds are in
``STATUS.md`` (2026-11-26 provider performance audit).

CPU smoke test (phases ``env,memory`` with a small model; numbers meaningless)::

    PYTHONPATH=src python deploy/runpod/provider_perf_session.py --cpu --phases env,memory \\
        --models qwen2.5-0.5b --eager-rows 64 --out /tmp/perf
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PGW = Path(__file__).resolve().parents[2]
PHASES = ("env", "memory", "device_time", "gemv_table", "lane_local", "maskgen", "lane_nic")
MINUTES = {"env": 1, "memory": 20, "device_time": 12, "gemv_table": 6, "lane_local": 10, "maskgen": 25, "lane_nic": 0}


def _log(out: Path, phase: str, msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {phase}: {msg}"
    print(line, flush=True)
    with (out / "session.log").open("a") as f:
        f.write(line + "\n")


def _run(out: Path, phase: str, cmd: list[str], minutes: float, dry: bool) -> dict[str, Any]:
    _log(out, phase, "$ " + " ".join(shlex.quote(c) for c in cmd))
    if dry:
        return {"cmd": cmd, "dry_run": True}
    env = dict(os.environ, PYTHONPATH=str(PGW / "src"))
    t0 = time.time()
    try:
        p = subprocess.run(cmd, cwd=PGW, env=env, capture_output=True, text=True,  # noqa: S603 - our own benches
                           timeout=max(60.0, minutes * 60 * 1.5))
        rc, so, se = p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired as exc:
        rc, so, se = -9, str(exc.stdout or ""), "timeout"
    (out / f"{phase}.stdout.txt").write_text(so)
    (out / f"{phase}.stderr.txt").write_text(se[-200_000:])
    _log(out, phase, f"exit {rc} in {time.time() - t0:.0f} s")
    return {"cmd": cmd, "rc": rc, "s": round(time.time() - t0, 1)}


# ---- env ---------------------------------------------------------------------------------------------------
def phase_env(cpu: bool) -> dict[str, Any]:
    import torch

    info: dict[str, Any] = {"python": sys.version.split()[0], "torch": torch.__version__, "host": platform.node(),
                            "cpus": os.cpu_count()}
    try:
        import triton

        info["triton"] = triton.__version__
    except Exception as exc:
        info["triton"] = f"unavailable: {exc}"
    try:
        from pgw.core.native import fast

        info["pgw_fast"] = fast is not None and hasattr(fast, "LaneService")
    except Exception as exc:
        info["pgw_fast"] = f"unavailable: {exc}"
    if torch.cuda.is_available() and not cpu:
        info["gpus"] = []
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            info["gpus"].append({"name": p.name, "cc": f"{p.major}.{p.minor}",
                                 "total_gib": round(p.total_memory / 2**30, 2), "sms": p.multi_processor_count})
        try:
            info["nvidia_smi"] = subprocess.run(["nvidia-smi",  # noqa: S607 "--query-gpu=name,driver_version,clocks.max.sm,"
                                                 "clocks.max.mem,power.limit", "--format=csv"], capture_output=True,
                                                text=True, timeout=30).stdout
        except Exception as exc:
            info["nvidia_smi"] = str(exc)
        from pgw.core.triton_gemv import max_rows, self_check

        dev = torch.device("cuda:0")
        info["gemv_self_check"] = bool(self_check(dev))
        info["gemv_max_rows"] = int(max_rows(dev))
    return info


# ---- memory ------------------------------------------------------------------------------------------------
class _Mem:
    """``memory_allocated`` / peak deltas on CUDA (zeros on CPU)."""

    def __init__(self, dev: Any):
        import torch

        self.dev = dev
        self.cuda = dev.type == "cuda"
        self.torch = torch

    def now(self) -> int:
        return int(self.torch.cuda.memory_allocated(self.dev)) if self.cuda else 0

    def reset(self) -> int:
        if self.cuda:
            self.torch.cuda.synchronize(self.dev)
            self.torch.cuda.reset_peak_memory_stats(self.dev)
        return self.now()

    def peak(self) -> int:
        if self.cuda:
            self.torch.cuda.synchronize(self.dev)
            return int(self.torch.cuda.max_memory_allocated(self.dev))
        return 0


def phase_memory(model: str, cpu: bool, eager_rows: int, masked_head: bool, force: bool) -> dict[str, Any]:
    import numpy as np
    import torch

    from pgw.core.ring import RingParams
    from pgw.core.types import LinearId, LinearKind, LinearSpec
    from pgw.service import memplan
    from pgw.service.kernels import ComputeKernel

    dev = torch.device("cpu" if cpu or not torch.cuda.is_available() else "cuda:0")
    shape = memplan.MODELS[model]
    lins = shape.linears(masked_head)
    gpu = memplan.gpu_spec_of(dev) if dev.type == "cuda" else memplan.GPUS["h100-80"]
    cfg = memplan.MemConfig(masked_head=masked_head)
    pl = memplan.plan(model, lins, gpu, cfg)
    pred = pl.devices[0]
    res: dict[str, Any] = {"model": model, "device": str(dev), "gpu": gpu.name, "predicted": pl.summary()}
    if not pl.fits and not force:
        res["skipped"] = "the memory plan says it does not fit one GPU (--force to try anyway)"
        return res
    G = 1 << 30
    mem = _Mem(dev)
    ring = RingParams.from_name("K32A12C27")
    gen = torch.Generator().manual_seed(1)
    stage: dict[str, Any] = {}
    # load: every linear through the service's kernel constructor (host weight -> one padded upload)
    base = mem.reset()
    t0 = time.perf_counter()
    kernels: list[Any] = []
    for key, m, n in lins:
        layer, kind = key.split(".")
        lid = LinearId(int(layer), LinearKind.LM_HEAD if kind == "head" else LinearKind(kind))
        w = torch.randint(-128, 128, (m, n), dtype=torch.int8, generator=gen)
        kernels.append(ComputeKernel(LinearSpec(lid, m, n), w, device=dev))
        del w
    stage["load"] = {"s": round(time.perf_counter() - t0, 1), "resident_gib": round((mem.now() - base) / G, 3),
                     "peak_gib": round((mem.peak() - base) / G, 3)}
    # warm-up of every bucket (graphs) for the first ring, as the registry does
    b0 = mem.reset()
    t0 = time.perf_counter()
    for k in kernels:
        k.warmup(ring, all_buckets=True)
    stage["warmup"] = {"s": round(time.perf_counter() - t0, 1), "resident_delta_gib": round((mem.now() - b0) / G, 3),
                       "peak_delta_gib": round((mem.peak() - b0) / G, 3)}
    # lane graphs (rows 1..8) for every linear, as the service captures them
    if dev.type == "cuda":
        try:
            stage["lane_capture"] = _lane_capture(model, lins, kernels, mem)
        except Exception as exc:
            stage["lane_capture"] = {"error": f"{type(exc).__name__}: {str(exc)[:300]}"}
    # one prefill-sized eager call on the largest layer linear and on the head
    big = max((k for k in kernels if k.kind != "lm_head"), key=lambda k: k.out_features * k.in_features)
    for name, k in (("eager_largest_linear", big), ("eager_head", kernels[-1] if masked_head else None)):
        if k is None:
            continue
        x = torch.randint(0, 1 << 32, (eager_rows, k.in_features), dtype=torch.int64, generator=gen)
        b1 = mem.reset()
        t0 = time.perf_counter()
        k.run_compact(x, ring)
        stage[name] = {"linear": k.linear, "rows": eager_rows, "s": round(time.perf_counter() - t0, 3),
                       "peak_delta_gib": round((mem.peak() - b1) / G, 3)}
        del x
    # pks-v3: hint build (4096 columns) and a 4-group key switch on the largest linear
    rng = np.random.default_rng(2)
    m, n = big.out_features, big.in_features
    if cpu:  # the CPU smoke test keeps these small
        m = min(m, 256)
    a_cols = 4096
    b2 = mem.reset()
    t0 = time.perf_counter()
    a = rng.integers(0, 1 << 56, (n, a_cols), dtype=np.uint64)
    h = big.mulmod64(a, 7) if not cpu else rng.integers(0, 1 << 56, (m, a_cols), dtype=np.uint64)
    stage["pks_hint_build"] = {"linear": big.linear, "cols": a_cols, "s": round(time.perf_counter() - t0, 2),
                               "peak_delta_gib": round((mem.peak() - b2) / G, 3)}
    del a
    # the provider's host-side modulus switch of that hint (PKSProvider.switched_hint; numpy, one thread)
    from pgw.hemask import pks as _pks

    rows = min(len(h), 8192)
    t0 = time.perf_counter()
    _pks.switch_hint(np.asarray(h[:rows], dtype=np.uint64) & np.uint64((1 << 56) - 1), _pks.PKSParams())
    dt = time.perf_counter() - t0
    stage["pks_switch_hint_host"] = {"rows_timed": rows, "s": round(dt, 2),
                                     "s_whole_linear": round(dt * len(h) / rows, 1),
                                     "hint_gb_per_seed_model": round(sum(mm for _, mm, _ in lins) * 4096 * 8 / 1e9, 1)}
    del h
    try:
        from pgw.hemask import pks
        from pgw.hemask.pks import NR, NU, RANK, D, PKSParams
        from pgw.service import pks_gpu

        groups, par = 4, PKSParams()
        mm = (m // 32) * 32 if not cpu else 64
        hs = rng.integers(0, par.q1, (mm, NU), dtype=np.uint64)
        ops = np.stack([np.stack([rng.integers(0, p, (RANK, NR), dtype=np.uint64) for p in pks.QP])
                        for _ in range(groups)]).astype(np.uint32)
        y = rng.integers(0, 1 << 56, (mm, D * groups), dtype=np.uint64)
        b3 = mem.reset()
        t0 = time.perf_counter()
        if not cpu:
            pks_gpu.keyswitch(hs, ops, y, dev)
        stage["pks_keyswitch"] = {"rows": mm, "groups": groups, "s": round(time.perf_counter() - t0, 2),
                                  "peak_delta_gib": round((mem.peak() - b3) / G, 3)}
    except Exception as exc:
        stage["pks_keyswitch"] = {"error": f"{type(exc).__name__}: {str(exc)[:300]}"}
    res["measured"] = stage
    if dev.type == "cuda":
        res["compare_gib"] = {
            "weights+rowsums (pred) vs load resident": [round((pred.steady["weights"] + pred.steady["rowsums"]) / G, 3),
                                                        stage["load"]["resident_gib"]],
            "graph pool + arenas (pred) vs warm-up resident": [
                round((pred.steady.get("graph_pool", 0) + pred.steady.get("graph_arenas", 0)) / G, 3),
                stage["warmup"]["resident_delta_gib"]],
            "eager call (pred, any linear) vs measured": [
                round(pred.serving["eager_call"] / G, 3),
                max(stage.get("eager_largest_linear", {}).get("peak_delta_gib", 0),
                    stage.get("eager_head", {}).get("peak_delta_gib", 0))],
        }
    return res


def _lane_capture(model: str, lins: list[tuple[str, int, int]], kernels: list[Any], mem: _Mem) -> dict[str, Any]:
    import tempfile
    from types import SimpleNamespace

    from pgw.core.native import fast
    from pgw.core.ring import RingParams
    from pgw.service.gpu_exec import device_executor
    from pgw.service.lane import _Slot, capture_lane_graphs
    from pgw.service.registry import RegisteredModel

    fp = "ef" * 32
    ring = "K32A12C27"
    specs = tuple(k.spec for k in kernels)
    reg_model = RegisteredModel(name=model, fingerprint=fp, specs=specs, kernels={k.linear: k for k in kernels},
                                rings=(ring,))
    td = tempfile.mkdtemp(prefix="pgwmem")
    svc = fast.LaneService(f"{td}/lane.sock", max_rows=8, use_graphs=True)
    mgr = SimpleNamespace(svc=svc, state=SimpleNamespace(registry=SimpleNamespace(get=lambda _fp: reg_model)),
                          settings=SimpleNamespace(lane_max_rows=8), _slots={fp: 0},
                          _tables={0: _Slot(fp, [s.id.key for s in specs], [RingParams.from_name(ring)])},
                          _graphs=[], _f2slots={})
    svc.set_model(0, [(s.in_features, s.out_features, s.id.kind.value) for s in specs], [(ring, 4)])
    device_executor(kernels[0].device).share_lock(svc.device_lock(int(kernels[0].device.index or 0)))
    b = mem.reset()
    t0 = time.perf_counter()
    rep = capture_lane_graphs(mgr)  # type: ignore[arg-type]
    out = {"s": round(time.perf_counter() - t0, 1), "captured": rep.captured, "checked": rep.checked,
           "disabled": rep.disabled, "resident_delta_gib": round((mem.now() - b) / 2**30, 3),
           "peak_delta_gib": round((mem.peak() - b) / 2**30, 3)}
    svc.clear_graphs()
    svc.stop(1.0)
    mgr._graphs.clear()
    return out


# ---- maskgen: pks-v3 items (hint cached on the host vs recomputed per item on the device) -------------------
#: The largest linears of each model: (key, m, n). 7B's masked head is the largest item of all.
PKS_LINEARS = {
    "qwen2.5-0.5b": [("0.o", 896, 896)],
    "qwen2.5-7b": [("0.gate_up", 37888, 3584), ("0.down", 3584, 18944), ("28.head", 152064, 3584)],
    "qwen2.5-32b": [("0.gate_up", 55296, 5120), ("0.down", 5120, 27648)],
    "qwen2.5-72b": [("0.gate_up", 59136, 8192), ("0.down", 8192, 29568)],
}


#: Settings of the device item's fused kernels (``pgw.service.pks_gpu``) the maskgen phase compares: ``auto`` (every
#: kernel after its self-test), ``prev`` (the fused NTTs only) and ``torch`` (every kernel plain).
PKS_ENV = ("PGW_SERVICE_PKS_GPU_NTT", "PGW_SERVICE_PKS_GPU_FWD4", "PGW_SERVICE_PKS_GPU_SWITCH",
           "PGW_SERVICE_PKS_GPU_RING", "PGW_SERVICE_PKS_PINNED", "PGW_SERVICE_PKS_HOST_THREADS",
           "PGW_SERVICE_PKS_GPU_WIRE")
_PKS_PLAIN = {"PGW_SERVICE_PKS_GPU_FWD4": "torch", "PGW_SERVICE_PKS_GPU_SWITCH": "torch",
              "PGW_SERVICE_PKS_GPU_RING": "torch", "PGW_SERVICE_PKS_GPU_WIRE": "torch",
              "PGW_SERVICE_PKS_PINNED": "0", "PGW_SERVICE_PKS_HOST_THREADS": "1"}
PKS_CONFIGS = {
    "auto": {},
    "fwd4": {"PGW_SERVICE_PKS_GPU_FWD4": "auto"},  # the 4-prime forward NTT (off by default: slower on the T4)
    "prev": _PKS_PLAIN,
    "torch": {**_PKS_PLAIN, "PGW_SERVICE_PKS_GPU_NTT": "torch"},
}


def phase_pks_items(models: list[str], cpu: bool, cols: int = 1024, work_mbs: tuple[int, ...] = (512, 1024, 2048)
                    ) -> dict[str, Any]:
    """Seconds and peak device memory of one pks-v3 item per path on the largest linears (random int8 weights through
    the service's ``ComputeKernel``): ``miss`` (the cache path with a new customer seed: hint GEMM, host switch,
    key switch), ``hit`` (same seed again), ``device`` (``pks_gpu.item``) at several ``PGW_SERVICE_PKS_GPU_WORK_MB``.
    Replies of every path must be byte-identical. Feeds ``memplan.PksRates`` (per-row key switch time, GEMM rate) and
    the hint policy (does a cache hit beat recomputing?)."""
    import numpy as np
    import torch

    from pgw.core.types import LinearId, LinearKind, LinearSpec
    from pgw.hemask import pks
    from pgw.hemask.wire import pack_bits
    from pgw.service import memplan, pks_gpu
    from pgw.service.kernels import ComputeKernel
    from pgw.service.pks_provider import PKSProvider, pack_ksk

    dev = torch.device("cpu" if cpu or not torch.cuda.is_available() else "cuda:0")
    mem = _Mem(dev)
    rng = np.random.default_rng(7)
    groups = cols // pks.D
    out: dict[str, Any] = {"device": str(dev), "cols": cols}
    for model in models:
        lins = PKS_LINEARS.get(model, [])
        for key, m, n in lins:
            if cpu:  # the CPU smoke test keeps these small
                m, n = min(m, 64), min(n, 64)
            layer, kind = key.split(".")
            lid = LinearId(int(layer), LinearKind.LM_HEAD if kind == "head" else LinearKind(kind))
            w = torch.randint(-128, 128, (m, n), dtype=torch.int8, generator=torch.Generator().manual_seed(3))
            kern = ComputeKernel(LinearSpec(lid, m, n), w, device=dev)
            del w
            b = rng.integers(0, 1 << 56, (n, cols), dtype=np.uint64)
            ops = np.stack([np.stack([rng.integers(0, p, (pks.RANK, pks.NR), dtype=np.uint64) for p in pks.QP])
                            for _ in range(groups)]).astype(np.uint32)
            from types import SimpleNamespace

            item = SimpleNamespace(linear=key, cols=cols, b=pack_bits(b.reshape(-1), 56), b_bits=56,
                                   ksk=pack_ksk(ops))
            res: dict[str, Any] = {"m": m, "n": n}
            replies: dict[str, bytes] = {}
            cache = PKSProvider(cache_bytes=10**13, mode="cache")
            cache._device_path(kern)  # the device key switch's self-test outside the timing
            seed = os.urandom(32)
            for path in ("miss", "hit"):
                b0 = mem.reset()
                t0 = time.perf_counter()
                replies[path] = bytes(cache.run_item("fp", kern, item, seed))
                res[path] = {"s": round(time.perf_counter() - t0, 3),
                             "peak_delta_gib": round((mem.peak() - b0) / 2**30, 3)}
            del cache
            # the device path with every fused kernel as chosen by ``auto`` (the fused NTTs, 4-prime forward, hint
            # switch + recombination, device ring rows and pinned overlapped copies on CUDA after their self-tests)
            # at every work budget; at the first budget also ``fwd4`` (with the 4-prime forward NTT), ``prev`` (the
            # fused NTTs only: the previous state) and ``torch`` (every kernel plain)
            runs = [(wmb, "auto") for wmb in (work_mbs if not cpu else (512,))] + [
                (work_mbs[0], c) for c in ("fwd4", "prev", "torch")]
            for wmb, cfg in runs:
                for k in PKS_ENV:
                    os.environ.pop(k, None)
                os.environ.update(PKS_CONFIGS[cfg])
                os.environ["PGW_SERVICE_PKS_GPU_WORK_MB"] = str(wmb)
                pv = PKSProvider(mode="device")
                pv.use_device("fp", kern, key)  # self-test outside the timing
                bytes(pv.run_item("fp", kern, item, seed))  # warm-up: Triton compilation, allocator
                b0 = mem.reset()
                t0 = time.perf_counter()
                y = bytes(pv.run_item("fp", kern, item, seed))
                dt = time.perf_counter() - t0
                tag = f"device_{wmb}mb" + ("" if cfg == "auto" else f"_{cfg}")
                res[tag] = {"s": round(dt, 3), "peak_delta_gib": round((mem.peak() - b0) / 2**30, 3),
                            "us_per_row": round(dt / m * 1e6, 2), "device_items": pv.device_items,
                            "rows_per_block": pks_gpu.item_rows(m, kern.in_features, cols, groups),
                            "kernels": pks_gpu.fused_status(dev)}
                replies[tag] = y
            for k in (*PKS_ENV, "PGW_SERVICE_PKS_GPU_WORK_MB"):
                os.environ.pop(k, None)
            res["identical"] = len(set(replies.values())) == 1
            res["model_estimate_s"] = {p: round(sum(memplan.pks_item_seconds(m, n, cols, p).values()), 3)
                                       for p in ("miss", "hit", "device")}
            res["model_estimate_s"]["device_fused"] = round(
                sum(memplan.pks_item_seconds(m, n, cols, "device", ntt="fused").values()), 3)
            res["model_estimate_s"]["device_prev"] = round(
                sum(memplan.pks_item_seconds(m, n, cols, "device", ntt="fused", kernels="ntt").values()), 3)
            out[f"{model}:{key}"] = res
            del kern
            if dev.type == "cuda":
                torch.cuda.empty_cache()
    return out


# ---- driver ------------------------------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default="/workspace/pgw/perf-" + time.strftime("%Y%m%d-%H%M"))
    ap.add_argument("--phases", default=",".join(PHASES))
    ap.add_argument("--models", default="qwen2.5-7b,qwen2.5-32b,qwen2.5-72b")
    ap.add_argument("--eager-rows", type=int, default=4096)
    ap.add_argument("--no-masked-head", action="store_true")
    ap.add_argument("--force", action="store_true", help="memory: load a model even if the plan says it does not fit")
    ap.add_argument("--host", default=os.environ.get("RUNPOD_PUBLIC_IP", ""), help="lane_nic: the pod's public address")
    ap.add_argument("--nic-minutes", type=float, default=10.0)
    ap.add_argument("--nic-port", type=int, default=8444, help="lane_nic: GPU lane port (a RunPod-mapped TCP port)")
    ap.add_argument("--nic-stub-port", type=int, default=8443,
                    help="lane_nic: transport-only stub lane port (RunPod maps 8443 and 8444 for the provider: stop "
                         "the provider service first, e.g. pkill -f 'pgw service serve')")
    ap.add_argument("--budget-min", type=float, default=120.0, help="stop starting new phases after this")
    ap.add_argument("--cpu", action="store_true", help="CPU smoke test")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    phases = [p for p in a.phases.split(",") if p]
    for p in phases:
        if p not in PHASES:
            ap.error(f"unknown phase {p!r}; choose from {PHASES}")
    py = sys.executable
    ngpu = 0
    if not a.cpu:
        try:
            import torch

            ngpu = torch.cuda.device_count()
        except Exception:
            ngpu = 0
    tp = "1,2" if ngpu >= 2 else "1"
    t_start = time.time()
    summary: dict[str, Any] = {"phases": {}}
    est = sum(MINUTES[p] for p in phases) + (2 * a.nic_minutes if "lane_nic" in phases else 0)
    _log(out, "plan", f"phases {phases}; ~{est:.0f} min")
    for p in phases:
        if (time.time() - t_start) / 60 > a.budget_min:
            _log(out, p, "skipped: session budget reached")
            break
        if p == "env":
            r: Any = {"dry_run": True} if a.dry_run else phase_env(a.cpu)
        elif p == "memory":
            r = {}
            for model in a.models.split(","):
                _log(out, p, f"{model}")
                if a.dry_run:
                    r[model] = {"dry_run": True}
                    continue
                try:
                    r[model] = phase_memory(model, a.cpu, a.eager_rows, not a.no_masked_head, a.force)
                except Exception as exc:  # e.g. an OOM: that is a finding
                    r[model] = {"error": f"{type(exc).__name__}: {str(exc)[:500]}"}
                (out / "memory.json").write_text(json.dumps(r, indent=1))
                try:
                    import gc

                    import torch

                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception as exc:  # best effort between models
                    _log(out, p, f"cache release failed: {exc}")
        elif p == "device_time":
            r = _run(out, p, [py, "benchmarks/provider_scale_bench.py", "--presets", "7b,32b,72b", "--tp", tp,
                              "--rows", "1,2,4,8", "--iters", "400", "--json", str(out / "device_time.json")],
                     MINUTES[p], a.dry_run)
        elif p == "gemv_table":
            r = _run(out, p, [py, "benchmarks/gpu_kernel_bench.py", "--gemv", "--gemv-sweep", "--shapes", "7b,0.5b",
                              "--rows", "1,2,3,4,5,6,7,8", "--emit-table", str(out / "gemv_table.json"),
                              "--json", str(out / "gemv.json")], MINUTES[p], a.dry_run)
        elif p == "lane_local":
            r = {pre: _run(out, f"{p}_{pre}", [py, "benchmarks/lane_gpu_trip_bench.py", "--preset", pre, "--iters",
                                               "300", "--repeat", "3", "--load", "none,python,gemm", "--json",
                                               str(out / f"lane_local_{pre}.json")], MINUTES[p] / 2, a.dry_run)
                 for pre in ("7b", "72b")}
        elif p == "maskgen":
            r = {"v2": _run(out, p, [py, "benchmarks/maskgen_provider_bench.py", "--presets", "qwen2.5-7b",
                                     "llama3-70b", "--formats", "v2:53:35", "--groups", "1", "4", "16", "--conc", "1",
                                     "2", "4", "--modes", "fused", "--decode", "--json", str(out / "maskgen.json")],
                            MINUTES[p] / 2, a.dry_run)}
            _log(out, p, f"pks-v3 items (cache miss / hit / device hint) on {a.models}")
            if a.dry_run:
                r["pks_v3"] = {"dry_run": True}
            else:
                try:
                    r["pks_v3"] = phase_pks_items(a.models.split(","), a.cpu)
                except Exception as exc:  # e.g. an OOM: that is a finding
                    r["pks_v3"] = {"error": f"{type(exc).__name__}: {str(exc)[:500]}"}
                (out / "maskgen_pks_v3.json").write_text(json.dumps(r["pks_v3"], indent=1))
        else:  # lane_nic
            if not a.host:
                _log(out, p, "skipped: --host (the pod's public IP) is needed for the certificate")
                continue
            r = {}
            for pre in ("7b", "72b"):
                info = out / f"lane_nic_{pre}.json"
                cmd = [py, "benchmarks/lane_nic_bench.py", "serve", "--preset", pre, "--host", a.host,
                       "--port", str(a.nic_port), "--stub-port", str(a.nic_stub_port), "--out", str(info),
                       "--seconds", str(a.nic_minutes * 60)]
                _log(out, p, f"{pre}: copy {info} to the client pod, then run there: "
                     f"python benchmarks/lane_nic_bench.py client --info lane_nic_{pre}.json "
                     f"--port <RunPod public port of {a.nic_port}> --stub-port <of {a.nic_stub_port}> "
                     f"--conc 1,4,16,64 --iters 400 --json lane_nic_{pre}_client.json")
                r[pre] = _run(out, f"{p}_{pre}", cmd, a.nic_minutes, a.dry_run)
        summary["phases"][p] = r
        (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    _log(out, "done", f"{(time.time() - t_start) / 60:.1f} min; results in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
