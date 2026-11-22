#!/usr/bin/env python3
"""Provider service time per 1-row trip at 7B / 32B / 72B shapes, on 1 and N GPUs (scaling plan targets 3-4).

For each preset (one decoder layer's four linears, random int8 weights: only shapes matter for time) and each
tensor-parallel degree, three serving paths are timed per trip, cycling ``qkv, o, gate_up, down`` like decode:

* ``kernel``: :meth:`ComputeKernel.run_compact` / :meth:`ShardedKernel.run_compact` (the Python kernel path the
  stream channel uses: staging, fused Triton GEMV, sync; shards on helper threads);
* ``graph``: the native lane's CUDA-graph replay (``LaneService.run_graph``: device lock, body copy to pinned
  memory, ``cuGraphLaunch`` per shard, spin wait, gather) -- the per-trip path measured at 0.066 ms on the H100;
* ``persistent``: the native lane's persistent-worker path (``LaneService.run_persistent``: body copy to mapped
  memory, doorbell, spin on ``done``; resident NVRTC kernel per GPU, no launch).

and the device floor of the GEMV alone (CUDA events around back-to-back fused-kernel launches on one shard).
``overhead = service - floor`` is the number the plan targets (<= 15-25 us per trip; TP gather <= 5 us).
Every path's bytes are compared with the kernel path on every timed trip's first round (exact).

Usage (GPU host, from ``pgw/``)::

    PYTHONPATH=src python benchmarks/provider_scale_bench.py --presets 7b,32b,72b --tp 1,4 --rows 1,2 \\
        --iters 400 --json /tmp/provider_scale.json

On Modal (do not run without the lead's go-ahead; 4 x H100 at about $16/h plus CPU: a 10-15 min run is about $3-4)::

    modal run deploy/modal/gpu_bench.py --target h100x4_c32 --timeout-min 30 --out provider_scale.json \\
        --cmd "python benchmarks/provider_scale_bench.py --presets 7b,32b,72b --tp 1,2,4 --rows 1,2 \\
               --iters 400 --json /tmp/provider_scale.json"

CPU smoke test (no GPU: the persistent workers are emulated, graphs are skipped; timings meaningless)::

    PYTHONPATH=src python benchmarks/provider_scale_bench.py --presets tiny --tp 1,2 --cpu --iters 20
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

FP = "ab" * 32
RING = "K32A8"
#: Qwen2.5 shapes (out m, in n) per linear: hidden, intermediate, heads, kv heads (head dim 128)
PRESETS: dict[str, dict[str, tuple[int, int]]] = {
    "tiny": {"qkv": (96, 64), "o": (64, 64), "gate_up": (256, 64), "down": (64, 128)},
    "7b": {"qkv": ((28 + 2 * 4) * 128, 3584), "o": (3584, 3584), "gate_up": (2 * 18944, 3584), "down": (3584, 18944)},
    "32b": {"qkv": ((40 + 2 * 8) * 128, 5120), "o": (5120, 5120), "gate_up": (2 * 27648, 5120), "down": (5120, 27648)},
    "72b": {"qkv": ((64 + 2 * 8) * 128, 8192), "o": (8192, 8192), "gate_up": (2 * 29568, 8192), "down": (8192, 29568)},
}
KINDS = ("qkv", "o", "gate_up", "down")


def _stats(samples: list[float]) -> dict[str, float]:
    s = sorted(samples)
    return {"mean_us": round(statistics.fmean(s) * 1e6, 2), "p50_us": round(s[len(s) // 2] * 1e6, 2),
            "p90_us": round(s[int(len(s) * 0.9)] * 1e6, 2), "n": len(s)}


def _build(preset: str, tp: int, cpu: bool) -> tuple[Any, list[Any]]:
    from pgw.core.types import LinearId, LinearKind, LinearSpec
    from pgw.service import ComputeKernel, ShardedKernel
    from pgw.service.registry import RegisteredModel

    kinds = {"qkv": LinearKind.QKV, "o": LinearKind.O, "gate_up": LinearKind.GATE_UP, "down": LinearKind.DOWN}
    devs = ["cpu"] * tp if cpu else [f"cuda:{i}" for i in range(tp)]
    gen_dev = "cpu" if cpu else "cuda:0"
    specs, kernels = [], {}
    for i, kind in enumerate(KINDS):
        m, n = PRESETS[preset][kind]
        spec = LinearSpec(LinearId(0, kinds[kind]), m, n)
        g = torch.Generator(device=gen_dev).manual_seed(i)
        w = torch.randint(-128, 128, (m, n), dtype=torch.int8, device=gen_dev, generator=g)
        k = ShardedKernel(spec, w, devs) if tp > 1 else ComputeKernel(spec, w, device=devs[0])
        del w
        specs.append(spec)
        kernels[spec.id.key] = k
    model = RegisteredModel(name=f"bench-{preset}", fingerprint=FP, specs=tuple(specs), kernels=kernels, rings=(RING,))
    return model, [kernels[s.id.key] for s in specs]


class _Mgr:
    """The parts of :class:`pgw.service.lane.LaneManager` that graph capture and the persistent start read."""

    def __init__(self, svc: Any, model: Any, max_rows: int):
        from pgw.core.ring import RingParams
        from pgw.service.lane import _Slot

        self.svc = svc
        reg = SimpleNamespace(get=lambda fp: model, models=lambda: [model])
        self.state = SimpleNamespace(registry=reg)
        self.settings = SimpleNamespace(lane_max_rows=max_rows)
        self._slots = {FP: 0}
        self._tables = {0: _Slot(FP, [s.id.key for s in model.specs], [RingParams.from_name(RING)])}
        self._graphs: list[Any] = []
        self._persistent: list[Any] = []
        svc.set_model(0, [(s.in_features, s.out_features, s.id.kind.value) for s in model.specs], [(RING, 4)])


def _floor_us(kern: Any, rows: int, iters: int) -> float | None:
    """Device-only time of the fused GEMV for shard 0 (CUDA events, back-to-back launches)."""
    from pgw.core.triton_gemv import ring_gemv

    k0 = getattr(kern, "shards", [kern])[0]
    if k0.device.type != "cuda" or not k0.triton_gemv:
        return None
    w = k0._pw._w_dev
    n, m = int(k0.in_features), int(k0.out_features)
    x = torch.randint(0, 1 << 31, (rows, n), dtype=torch.int32, device=k0.device)
    y = torch.empty((rows, m), dtype=torch.int32, device=k0.device)
    with torch.cuda.device(k0.device):
        for _ in range(5):
            ring_gemv(x, w, 32, n=n, m=m, out=y)
        a = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
        b = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
        a.record()
        for _ in range(iters):
            ring_gemv(x, w, 32, n=n, m=m, out=y)
        b.record()
        b.synchronize()
        return float(a.elapsed_time(b)) * 1e3 / iters


def bench(preset: str, tp: int, rows_list: list[int], iters: int, cpu: bool, modes: list[str]) -> dict[str, Any]:
    from pgw.core.native import fast
    from pgw.core.packing import pack_ring
    from pgw.core.ring import RingParams
    from pgw.service.gpu_exec import device_executor
    from pgw.service.lane import capture_lane_graphs
    from pgw.service.persistent import start_lane_persistent, stop_lane_persistent

    ring = RingParams.from_name(RING)
    t0 = time.perf_counter()
    model, kerns = _build(preset, tp, cpu)
    out: dict[str, Any] = {"preset": preset, "tp": tp, "build_s": round(time.perf_counter() - t0, 1),
                           "shapes": {k: PRESETS[preset][k] for k in KINDS}, "rows": {}}
    tmp = Path(tempfile.mkdtemp(prefix="pgwb"))
    svc = fast.LaneService(str(tmp / "l.sock"), max_rows=8, use_graphs=not cpu, persistent=True)
    mgr = _Mgr(svc, model, max(rows_list))
    if not cpu:
        for d in {str(dv) for k in kerns for dv in getattr(k, "devices", [k.device])}:
            dev = torch.device(d)
            device_executor(dev).share_lock(svc.device_lock(int(dev.index or 0)))
    rng = np.random.default_rng(1)
    bodies = {r: [[rng.integers(0, 1 << 32, size=(r, k.in_features), dtype=np.int64) for _ in range(4)] for k in kerns]
              for r in rows_list}
    want = {r: [[pack_ring(k.run_compact(torch.from_numpy(x), ring), ring) for x in bodies[r][i]]
                 for i, k in enumerate(kerns)] for r in rows_list}
    packed = {r: [[pack_ring(torch.from_numpy(x), ring) for x in bodies[r][i]] for i in range(4)] for r in rows_list}
    for r in rows_list:
        out["rows"][r] = {"floor_us": {KINDS[i]: (None if cpu else _floor_us(k, r, 50)) for i, k in enumerate(kerns)}}

    def timed(fn: Any, r: int) -> tuple[dict[str, Any], int]:
        per: dict[str, list[float]] = {k: [] for k in KINDS}
        bad = 0
        for it in range(iters + 8):
            for i in range(4):
                j = it % 4
                t = time.perf_counter()
                got = fn(i, r, j)
                dt = time.perf_counter() - t
                if it < 4 and got != want[r][i][j]:
                    bad += 1
                if it >= 8:
                    per[KINDS[i]].append(dt)
        res: dict[str, Any] = {k: _stats(v) for k, v in per.items()}
        res["trip_mean_us"] = round(statistics.fmean([res[k]["mean_us"] for k in KINDS]), 2)
        return res, bad

    try:
        if "kernel" in modes:
            for r in rows_list:
                def run_kernel(i: int, r: int, j: int) -> bytes:
                    return pack_ring(kerns[i].run_compact(torch.from_numpy(bodies[r][i][j]), ring), ring)

                res, bad = timed(run_kernel, r)
                out["rows"][r]["kernel"] = {**res, "mismatches": bad}
        if "graph" in modes and not cpu:
            grep = capture_lane_graphs(mgr, rows_list=rows_list)  # type: ignore[arg-type]
            out["graph_report"] = {"captured": grep.captured, "checked": grep.checked, "disabled": grep.disabled,
                                   "skipped": grep.skipped, "s": round(grep.seconds, 1)}
            if not grep.disabled:
                for r in rows_list:
                    res, bad = timed(lambda i, r, j: bytes(svc.run_graph(0, i, 0, r, packed[r][i][j])), r)
                    out["rows"][r]["graph"] = {**res, "mismatches": bad}
        if "persistent" in modes:
            prow = [r for r in rows_list if r <= 4]
            prep = start_lane_persistent(mgr, emulate=cpu, rows_list=prow)
            out["persistent_report"] = {"groups": prep.groups, "checked": prep.checked, "disabled": prep.disabled,
                                        "s": round(prep.seconds, 1),
                                        "grid": [getattr(w, "grid", None) for g in mgr._persistent for w in g.workers]}
            if not prep.disabled:
                for r in prow:
                    res, bad = timed(lambda i, r, j: bytes(svc.run_persistent(0, i, 0, r, packed[r][i][j])), r)
                    out["rows"][r]["persistent"] = {**res, "mismatches": bad}
            stop_lane_persistent(mgr)
    finally:
        svc.clear_graphs()
        svc.stop(1.0)
        for k in kerns:
            g = getattr(k, "group", None)
            if g is not None:
                g.close()
    for r in rows_list:
        row = out["rows"][r]
        fl = row["floor_us"]
        if all(v is not None for v in fl.values()):
            fmean = statistics.fmean(fl.values())
            row["floor_trip_mean_us"] = round(fmean, 2)
            for mode in ("kernel", "graph", "persistent"):
                if mode in row:
                    row[mode]["overhead_trip_mean_us"] = round(row[mode]["trip_mean_us"] - fmean, 2)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--presets", default="7b,32b,72b")
    ap.add_argument("--tp", default="1,4", help="tensor-parallel degrees (skipped above the GPU count)")
    ap.add_argument("--rows", default="1,2")
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--modes", default="kernel,graph,persistent")
    ap.add_argument("--cpu", action="store_true", help="CPU smoke test (emulated persistent workers)")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    ngpu = 0 if a.cpu else torch.cuda.device_count()
    if not a.cpu and ngpu == 0:
        print("no GPU (use --cpu for a smoke test)", file=sys.stderr)
        return 2
    results: dict[str, Any] = {"gpus": [torch.cuda.get_device_name(i) for i in range(ngpu)], "torch": torch.__version__,
                               "runs": []}
    for preset in a.presets.split(","):
        for tp in (int(t) for t in a.tp.split(",")):
            if not a.cpu and tp > ngpu:
                continue
            r = bench(preset, tp, [int(x) for x in a.rows.split(",")], a.iters, a.cpu, a.modes.split(","))
            results["runs"].append(r)
            line = {rows: {m: v.get("trip_mean_us") for m, v in d.items()
                           if isinstance(v, dict) and "trip_mean_us" in v} for rows, d in r["rows"].items()}
            floors = [d.get("floor_trip_mean_us") for d in r["rows"].values()]
            print(f"{preset} tp={tp}: trip mean us {line} floors {floors}", flush=True)
            if not a.cpu:
                torch.cuda.empty_cache()
    text = json.dumps(results, indent=1, default=str)
    if a.json:
        Path(a.json).write_text(text)
    print(text[-6000:])
    return 0


if __name__ == "__main__":
    sys.exit(main())
