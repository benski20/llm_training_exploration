#!/usr/bin/env python3
"""End-to-end 1-row lane trips on a GPU: client channel -> native listener -> CUDA-graph replay -> reply.

Where ``provider_scale_bench.py`` times ``LaneService.run_graph`` (no socket), this times what a co-located
gateway sees per trip at 7B (or 32B / 72B) shapes: the native ``TripChannel`` over the Unix domain socket (or
the opt-in shared-memory transport) to a real ``LaneService`` with captured, self-checked graphs, cycling
``qkv, o, gate_up, down``. Variants (interleaved, ``--repeat`` rounds, medians):

* ``uds``: today's defaults (fixed 50 us client spin, blocking server reads, reply copied out of ``h_out``);
* ``uds_tuned``: ``PGW_LANE_SPIN_AUTO_US`` (adaptive client spin), server read spin, zero-copy replies;
* ``shm``: as ``uds_tuned`` over the shared-memory transport (``PGW_SERVICE_LANE_SHM`` / ``PGW_LANE_SHM``).

``--load python``: a background thread keeps the Python kernel path busy on the same GPU (8-row stream-channel
style calls through ``ComputeKernel.run_compact``, which take ``DeviceExecutor``'s lock); each variant then runs
with the lane sharing that lock (today) and with its own lock (``PGW_SERVICE_LANE_OWN_LOCK``). ``--load gemm``:
a background CUDA stream runs back-to-back int8 GEMMs (maskgen-like device occupancy, no lock).

Every reply's bytes are compared with the kernel path on the first rounds (exact). Per variant and linear:
trip p50 / mean, the service's ``compute_us`` and ``queue_us`` (device-lock wait), and ``overhead`` = trip minus
``compute_us`` (transport, wake-ups, body copies: the number this work targets, <= ~150 us all-in).

Usage (GPU host, from ``pgw/``)::

    PYTHONPATH=src python benchmarks/lane_gpu_trip_bench.py --preset 7b --iters 300 --repeat 3 \\
        --load none,python,gemm --json /tmp/lane_gpu.json

CPU smoke test (no graphs: the Python executor serves; timings meaningless)::

    PYTHONPATH=src python benchmarks/lane_gpu_trip_bench.py --preset tiny --cpu --iters 20 --repeat 1
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import statistics
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from provider_scale_bench import KINDS, RING, _build, _Mgr

VARIANTS = {
    "uds": {"svc": {}, "spin_auto_us": 0, "shm": False},
    "uds_tuned": {"svc": {"zero_copy": True, "read_spin_us": 2000}, "spin_auto_us": 400, "shm": False},
    "shm": {"svc": {"shm": True, "read_spin_us": 2000}, "spin_auto_us": 400, "shm": True},
}


def _start(model: Any, kerns: list[Any], variant: str, own_lock: bool, cpu: bool, td: Path) -> tuple[Any, Any, bytes]:
    from pgw.core.native import fast
    from pgw.core.packing import pack_ring
    from pgw.core.ring import RingParams
    from pgw.service.gpu_exec import device_executor
    from pgw.service.lane import capture_lane_graphs

    ring = RingParams.from_name(RING)

    def py_execute(_m: int, li: int, _r: int, rows: int, body: bytes, _t: str, _left: float) -> bytes:
        x = torch.from_numpy(np.frombuffer(body, dtype="<u4").astype(np.int64).reshape(rows, -1))
        return pack_ring(kerns[li].run_compact(x, ring), ring)

    path = td / f"{variant}-{int(own_lock)}.sock"
    svc = fast.LaneService(str(path), max_rows=8, use_graphs=not cpu, executor=py_execute, **VARIANTS[variant]["svc"])
    mgr = _Mgr(svc, model, 1)
    if not cpu:
        for d in {str(dv) for k in kerns for dv in getattr(k, "devices", [k.device])}:
            dev = torch.device(d)
            ex = device_executor(dev)
            if own_lock:
                import threading as _t

                ex.share_lock(_t.Lock())  # the Python path's own lock again
            else:
                ex.share_lock(svc.device_lock(int(dev.index or 0)))
        os.environ["PGW_SERVICE_LANE_OWN_LOCK"] = "1" if own_lock else "0"
        rep = capture_lane_graphs(mgr, rows_list=[1])  # type: ignore[arg-type]
        if rep.disabled:
            raise SystemExit(f"lane graphs disabled: {rep.disabled}")
    t = secrets.token_bytes(32)
    svc.add_ticket(t, "bench", 0, 600.0)
    return svc, mgr, t


def _client(svc_path: str, ticket: bytes, variant: str) -> Any:
    from pgw.core.native import fast

    v = VARIANTS[variant]
    ch = fast.TripChannel.connect_service_uds(svc_path, ticket, 5.0, 8, 50)
    if v["spin_auto_us"]:
        ch.set_spin_auto(int(v["spin_auto_us"]))
    if v["shm"] and not ch.attach_shm():
        raise SystemExit("shm refused")
    return ch


def _load_thread(kind: str, kerns: list[Any], stop: threading.Event, cpu: bool) -> threading.Thread | None:
    from pgw.core.ring import RingParams

    ring = RingParams.from_name(RING)
    if kind == "none":
        return None
    if kind == "python":
        xs = [torch.randint(0, 1 << 31, (8, k.in_features), dtype=torch.int64) for k in kerns]

        def run() -> None:
            i = 0
            while not stop.is_set():
                kerns[i % 4].run_compact(xs[i % 4], ring)
                i += 1
    elif kind == "gemm" and not cpu:
        dev = torch.device("cuda:0")
        a = torch.randint(-128, 128, (8192, 4096), dtype=torch.int8, device=dev)
        b = torch.randint(-128, 128, (4096, 4096), dtype=torch.int8, device=dev)
        st = torch.cuda.Stream(dev)  # type: ignore[no-untyped-call]

        def run() -> None:
            with torch.cuda.stream(st):
                while not stop.is_set():
                    for _ in range(4):
                        torch._int_mm(a, b)
                    st.synchronize()
    else:
        return None
    th = threading.Thread(target=run, daemon=True)
    th.start()
    return th


def one(model: Any, kerns: list[Any], want: list[bytes], bodies: list[bytes], variant: str, own_lock: bool,
        load: str, iters: int, cpu: bool) -> dict[str, Any]:
    td = Path(tempfile.mkdtemp(prefix="pgwg"))
    svc, mgr, ticket = _start(model, kerns, variant, own_lock, cpu, td)
    stop = threading.Event()
    th = None
    try:
        ch = _client(str(td / f"{variant}-{int(own_lock)}.sock"), ticket, variant)
        outs = [bytearray(len(w)) for w in want]
        per: dict[str, dict[str, list[float]]] = {k: {"trip": [], "compute": [], "ovh": []} for k in KINDS}
        bad = 0
        th = _load_thread(load, kerns, stop, cpu)
        for it in range(iters + 20):
            for i in range(4):
                t0 = time.perf_counter()
                st, cu = ch.trip(0, i, 1, secrets.token_bytes(16), 10_000, bodies[i], outs[i], 5.0)
                dt = (time.perf_counter() - t0) * 1e6
                if st != 0:
                    raise SystemExit(f"lane refused (status {st})")
                if it < 3 and bytes(outs[i]) != want[i]:
                    bad += 1
                if it >= 20:
                    per[KINDS[i]]["trip"].append(dt)
                    per[KINDS[i]]["compute"].append(cu)
                    per[KINDS[i]]["ovh"].append(dt - cu)
        cs = ch.stats()
        ch.close()
    finally:
        stop.set()
        if th is not None:
            th.join(5)
        svc.clear_graphs()
        svc.stop(1.0)
        mgr._graphs.clear()
    res: dict[str, Any] = {"variant": variant, "own_lock": own_lock, "load": load, "mismatches": bad}
    for k in KINDS:
        p = per[k]
        res[k] = {"trip_p50": round(statistics.median(p["trip"]), 1),
                  "trip_mean": round(statistics.fmean(p["trip"]), 1),
                  "compute_p50": round(statistics.median(p["compute"]), 1),
                  "overhead_p50": round(statistics.median(p["ovh"]), 1),
                  "overhead_mean": round(statistics.fmean(p["ovh"]), 1)}
    n = max(1, cs["trips"])
    res["trip_mean"] = round(statistics.fmean(res[k]["trip_mean"] for k in KINDS), 1)
    res["overhead_mean"] = round(statistics.fmean(res[k]["overhead_mean"] for k in KINDS), 1)
    res["overhead_p50"] = round(statistics.fmean(res[k]["overhead_p50"] for k in KINDS), 1)
    res["channel"] = {"blocked_frac": round(cs["blocked"] / n, 3), "svc_queue_us": round(cs["queue_us"] / n, 1),
                      "svc_compute_us": round(cs["compute_us"] / n, 1), "wait_us": round(cs["wait_us"] / n, 1)}
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--preset", default="7b")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--variants", default="uds,uds_tuned,shm")
    ap.add_argument("--load", default="none", help="comma list of none, python, gemm")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    from pgw.core.packing import pack_ring
    from pgw.core.ring import RingParams

    ring = RingParams.from_name(RING)
    model, kerns = _build(args.preset, 1, args.cpu)
    rng = np.random.default_rng(7)
    xs = [rng.integers(0, 1 << 32, size=(1, k.in_features), dtype=np.int64) for k in kerns]
    bodies = [pack_ring(torch.from_numpy(x), ring) for x in xs]
    want = [pack_ring(k.run_compact(torch.from_numpy(x), ring), ring) for k, x in zip(kerns, xs, strict=True)]
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    loads = [v.strip() for v in args.load.split(",") if v.strip()]
    runs: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    for _ in range(args.repeat):
        for load in loads:
            for v in variants:
                for own in ([False, True] if load == "python" else [False]):
                    r = one(model, kerns, want, bodies, v, own, load, args.iters, args.cpu)
                    runs.append(r)
                    print(json.dumps({k: r[k] for k in ("variant", "own_lock", "load", "mismatches", "trip_mean",
                                                        "overhead_p50", "overhead_mean", "channel")}), flush=True)
    summary: dict[str, Any] = {}
    for r in runs:
        key = f"{r['load']}/{r['variant']}{'/own_lock' if r['own_lock'] else ''}"
        summary.setdefault(key, []).append(r)
    table = {}
    for key, rs in summary.items():
        table[key] = {m: round(statistics.median(x[m] for x in rs), 1) for m in ("trip_mean", "overhead_p50",
                                                                               "overhead_mean")}
        table[key]["queue_us"] = round(statistics.median(x["channel"]["svc_queue_us"] for x in rs), 1)
        table[key]["compute_us"] = round(statistics.median(x["channel"]["svc_compute_us"] for x in rs), 1)
        table[key]["per_linear_overhead_p50"] = {k: round(statistics.median(x[k]["overhead_p50"] for x in rs), 1)
                                                 for k in KINDS}
        table[key]["mismatches"] = sum(x["mismatches"] for x in rs)
    out = {"preset": args.preset, "iters": args.iters, "repeat": args.repeat,
           "seconds": round(time.perf_counter() - t0, 1), "summary": table, "runs": runs}
    print(json.dumps({"summary": table}, indent=1))
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
