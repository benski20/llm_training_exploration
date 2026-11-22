"""Gateway CPU per masked trip on the decode path: ``PGW_TRIP_FAST`` vs ``PGW_TRIP_NATIVE`` (one native call
per trip on each side of the lane call), per linear kind, at real shapes.

A synthetic pack (``decode_cpu_bench.build_pack``: zero linear weights, so the stub service's all-zero answer
is exact), a disk mask store, a windowed reservation as the gateway makes, the outbound-noise monitor at the
gateway defaults and a stub lane that answers instantly. Each trip is ``forward_many([call])`` on 1 row, in
model order (one "token" = one trip of every linear), so the store, lease refills and window extensions are
amortized as in decode. Reports microseconds per trip (wall, single thread) and ms per token of trips.

Usage:
    PYTHONPATH=src python benchmarks/trip_native_bench.py --preset 7b --layers 2 --tokens 48 \
        --workdir /dev/shm/pgw-tn [--masked-head] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from decode_cpu_bench import PRESETS, build_pack, set_wnorm


def run(args: argparse.Namespace, native_on: bool) -> dict[str, Any]:
    os.environ["PGW_TRIP_FAST"] = "1"
    os.environ["PGW_TRIP_NATIVE"] = "1" if native_on else "0"
    from pgw.core import RingParams
    from pgw.engine import MaskedBackend
    from pgw.engine.backends import LinearCall
    from pgw.factory import MaskGenerator
    from pgw.gateway.masks import WindowedReservation
    from pgw.maskstore import FileKeyProvider, MaskStore
    from pgw.maskstore.counters import FileMonotonicCounter
    from pgw.modelpack import ModelPack
    from pgw.observability.noise import OutboundNoiseMonitor
    from pgw.wire.messages import ComputeResponse, ComputeResult

    work = Path(args.workdir)
    work.mkdir(parents=True, exist_ok=True)
    pack: Any = ModelPack.open(build_pack(work, args.preset, args.layers))
    if args.masked_head:
        from pgw.engine.masked_head import MaskedHeadPack
        from pgw.modelpack.loader import LinearWeights

        class ZeroHeadPack(MaskedHeadPack):
            def _head(self) -> Any:
                if self._w is None:
                    s = self._spec
                    self._w = LinearWeights(self._lid, torch.zeros((s.out_features, s.in_features), dtype=torch.int8),
                                            torch.ones(s.out_features), None)
                return self._w

        pack = ZeroHeadPack(pack)
    ring = RingParams.from_name(args.ring)
    specs = list(pack.linear_specs())
    slots = args.tokens + args.warm + 8
    tag = f"{time.time_ns()}"
    store = MaskStore.open(work / f"tn-store-{tag}", FileKeyProvider(work / "kek", create=True), fsync=False,
                           counter=FileMonotonicCounter(work / f"tn-ctr-{tag}", fsync=False))
    MaskGenerator(pack, ring).generate_into_store(store, slots, chunk_slots=16, factory_id="tn")
    outm = {sp.id.key: sp.out_features for sp in specs}
    nb = ring.k // 8
    zeros = {k: bytes(m * nb) for k, m in outm.items()}

    class Stub:
        sync_stream = True

        def compute_sync(self, req: Any) -> Any:
            return ComputeResponse(req.request_id, tuple(
                ComputeResult(i.linear, i.rows, zeros[i.linear] if i.rows == 1 else bytes(i.rows * outm[i.linear] * nb))
                for i in req.items))

    noise = OutboundNoiseMonitor(ring, sample_rate=0.05, fail_closed=True)
    be = MaskedBackend(pack, None, Stub(), ring, pack.fingerprint, min_soundness_bits=20, noise_monitor=noise)
    set_wnorm(be, pack, args.wnorm_frac)
    be.warm_trip_engine()
    fp = pack.fingerprint
    res = WindowedReservation(store.reserve(fp, ring, args.window), lambda s: store.reserve(fp, ring, s),
                              cap=slots - 4, window=args.window)
    g = torch.Generator().manual_seed(0)
    xs = {sp.id.key: torch.randn(1, sp.in_features, generator=g) for sp in specs}
    per: dict[str, list[float]] = {}
    outs: list[bytes] = []
    try:
        for tok in range(args.warm + args.tokens):
            for sp in specs:
                c = LinearCall(sp.id, xs[sp.id.key], res)
                t0 = time.perf_counter()
                y = be.forward_many([c])[0]
                dt = time.perf_counter() - t0
                if tok >= args.warm:
                    per.setdefault(sp.id.kind.value, []).append(dt)
                if tok == args.warm:
                    outs.append(y.numpy().tobytes())
            be.join_checks()
    finally:
        res.close()
        store.close()
    kinds = {k: {"us_median": round(1e6 * sorted(v)[len(v) // 2], 1), "us_mean": round(1e6 * sum(v) / len(v), 1),
                 "calls": len(v)} for k, v in per.items()}
    tot = sum(sum(v) for v in per.values()) / args.tokens
    full, lay = PRESETS[args.preset]["full_layers"], args.layers
    head = sum(per.get("lm_head", [])) / args.tokens
    return {"native": native_on, "per_kind": kinds, "ms_per_token_trips": round(1e3 * tot, 3),
            "ms_per_token_trips_full_model": round(1e3 * (head + (tot - head) * full / lay), 2),
            "trips_per_token": len(specs), "outputs": outs}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="0.5b", choices=sorted(PRESETS))
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--tokens", type=int, default=48)
    ap.add_argument("--warm", type=int, default=8)
    ap.add_argument("--window", type=int, default=16)
    ap.add_argument("--ring", default="K32A12C27")
    ap.add_argument("--wnorm-frac", type=float, default=0.3)
    ap.add_argument("--masked-head", action="store_true")
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    a = run(args, False)
    b = run(args, True)
    res = {"preset": args.preset, "layers": args.layers, "masked_head": args.masked_head,
           "outputs_identical": a.pop("outputs") == b.pop("outputs"), "fast": a, "native": b}
    print(json.dumps(res, indent=2))
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
