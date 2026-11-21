"""Gateway CPU of one decode round trip (1 row, one session) on the TripEngine path, per linear kind.

Runs ``MaskedBackend.forward_many`` for the remote linears of a pack built by ``decode_cpu_bench.py``
(``--workdir``, same preset and layer count), with a real encrypted mask store, the outbound-noise
monitor at the gateway defaults and a stub service that answers instantly with zeros. Reports the median
calling-thread CPU (``time.thread_time``) and wall per call: the gateway's own per-trip work (prepare +
finish + Python), which on a contended machine is far more stable than totals.

Usage:
    PYTHONPATH=src python benchmarks/trip_cpu_bench.py --workdir /tmp/pgw-dec [--preset 0.5b --layers 4]
        [--calls 300] [--kinds gate_up,down]
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--preset", default="0.5b")
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--calls", type=int, default=300)
    ap.add_argument("--kinds", default="gate_up,down")
    ap.add_argument("--ring", default="K32A12C27")
    ap.add_argument("--profile", action="store_true", help="cProfile the measured calls (tottime table)")
    args = ap.parse_args()
    from pgw.core import RingParams
    from pgw.core.types import LinearId
    from pgw.engine import LinearCall, MaskedBackend
    from pgw.factory import MaskGenerator
    from pgw.maskstore import FileKeyProvider, MaskStore
    from pgw.maskstore.counters import FileMonotonicCounter
    from pgw.modelpack import ModelPack
    from pgw.observability.noise import OutboundNoiseMonitor
    from pgw.wire.messages import ComputeResponse, ComputeResult

    work = Path(args.workdir)
    pack = ModelPack.open(work / f"pack-{args.preset}-{args.layers}")
    ring = RingParams.from_name(args.ring)
    kinds = args.kinds.split(",")
    slots = args.calls + 40
    store_dir = work / f"trip-store-{time.time_ns()}"
    store = MaskStore.open(store_dir, FileKeyProvider(work / "kek", create=True), fsync=False,
                           counter=FileMonotonicCounter(work / f"trip-counter-{time.time_ns()}", fsync=False))
    MaskGenerator(pack, ring).generate_into_store(store, slots, chunk_slots=16, factory_id="trip")
    outm = {sp.id.key: sp.out_features for sp in pack.linear_specs()}
    nb = ring.k // 8

    class Stub:
        sync_stream = True

        def compute_sync(self, req: Any) -> Any:
            return ComputeResponse(req.request_id, tuple(
                ComputeResult(i.linear, i.rows, bytes(i.rows * outm[i.linear] * nb)) for i in req.items))

    be = MaskedBackend(pack, None, Stub(), ring, pack.fingerprint, min_soundness_bits=20,
                       noise_monitor=OutboundNoiseMonitor(ring, sample_rate=0.05))
    be.warm_trip_engine()
    res = store.reserve(pack.fingerprint, ring, slots)
    specs = {sp.id.kind.value: sp for sp in pack.linear_specs() if sp.id.layer == 0}
    g = torch.Generator().manual_seed(0)
    out: dict[str, Any] = {"preset": args.preset, "trip_engine": be.trip_engine,
                           "prefetch_decode": getattr(be, "prefetch_decode", None)}
    cpu: dict[str, list[float]] = {k: [] for k in kinds}
    wall: dict[str, list[float]] = {k: [] for k in kinds}
    import cProfile
    import io
    import pstats

    prof = cProfile.Profile() if args.profile else None
    for i in range(args.calls):
        if prof is not None and i == 20:
            prof.enable()
        for k in kinds:
            sp = specs[k]
            x = torch.randn(1, sp.in_features, generator=g)
            c, t = time.thread_time(), time.perf_counter()
            be.forward_many([LinearCall(LinearId(0, sp.id.kind), x, res)])
            if i >= 20:
                cpu[k].append(time.thread_time() - c)
                wall[k].append(time.perf_counter() - t)
    if prof is not None:
        prof.disable()
        sio = io.StringIO()
        pstats.Stats(prof, stream=sio).sort_stats("tottime").print_stats(25)
        print(sio.getvalue())
    for k in kinds:
        out[k] = {"thread_cpu_us": round(1e6 * statistics.median(cpu[k]), 1),
                  "wall_us": round(1e6 * statistics.median(wall[k]), 1)}
    print(json.dumps(out))
    res.close()
    be.close()
    store.close()
    import shutil

    shutil.rmtree(store_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
