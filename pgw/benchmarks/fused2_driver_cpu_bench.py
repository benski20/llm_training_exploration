"""Gateway CPU per trip of the native fused2 decode driver (``pgw_fast.Fused2Driver``) at real widths.

The fused2 counterpart of ``decode_cpu_bench.py --lane-proc``: a synthetic pack with the preset's hidden,
intermediate and head shapes but ``--layers`` layers (zero linear weights, ``PGW_BENCH_VOCAB`` shrinks the
vocabulary), the public fused2 tables derived from it (migration multipliers), an in-process mask source
whose rows are computed ahead (``prefill``: the factory's work is not timed), and a native ``LaneService`` on a
Unix socket (``--shm``: the shared-memory transport) whose stub executor answers every item honestly for zero
weights (a pair's ``H_hat`` is its ``R_hat``, ``Z_hat`` is zero; a single's ``Y_hat`` is zero), so every range
and p61 Freivalds check of both stages runs and passes. The gateway runs the real ``Fused2MaskedBackend`` and
the native driver on ``float_ops = "native"``; only the driver's local time (quantize, mask, pack, unpack,
unmask, both checks, float work) is reported per trip kind, from the driver's own counters.

Usage:
    PGW_BENCH_VOCAB=16384 PYTHONPATH=src:. python benchmarks/fused2_driver_cpu_bench.py --preset 32b --layers 2 \
        --tokens 24 --workdir /tmp/pgw-f2 [--shm] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from decode_cpu_bench import PRESETS, build_pack


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="32b", choices=sorted(PRESETS))
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--tokens", type=int, default=24)
    ap.add_argument("--prompt-len", type=int, default=4)
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--shm", action="store_true")
    ap.add_argument("--spin-us", type=int, default=400)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    from pgw.core import RingParams
    from pgw.core.native import fast
    from pgw.engine import SequenceInput, TransformerModel
    from pgw.engine.fused2 import (
        Fused2MaskedBackend,
        Fused2MaskSource,
        Fused2Weights,
        enable_fused2,
        migration_multipliers,
    )
    from pgw.engine.fused2_lane import LaneFused2Provider, items_of
    from pgw.engine.fused2_native import enable_fused2_native
    from pgw.modelpack import ModelPack
    from pgw.service.fused2_lane import fused2_items

    ring = RingParams(56, 12, 53)
    nb = ring.k // 8
    work = Path(args.workdir or tempfile.mkdtemp(prefix="pgw-f2-"))
    work.mkdir(parents=True, exist_ok=True)
    pack = ModelPack.open(build_pack(work, args.preset, args.layers))
    t0 = time.perf_counter()
    cached = work / f"f2w-{args.preset}-{args.layers}-v{pack.arch.vocab_size}.pt"  # the tables, reused across runs
    if cached.exists():
        weights = Fused2Weights.load(cached)
    else:
        weights = Fused2Weights(pack, None, multipliers=migration_multipliers(pack, 8))
        weights.save(cached)
    print(f"fused2 tables: {time.perf_counter() - t0:.1f}s", file=sys.stderr)
    items = fused2_items(weights)
    widths = [(it.n1, it.m1, it.m2, it.k2 is not None) for it in items]

    def execute(_model: int, linear: int, _ring: int, rows: int, body: bytes, _tenant: str, _left: float) -> Any:
        n1, m1, m2, pair = widths[linear]
        if not pair:  # Y_hat = W x_hat = 0
            return bytes(rows * m1 * nb)
        # H_hat = R_hat + hmul * W1 x_hat = R_hat; Z_hat = W2' H_hat = 0
        return bytes(body[rows * n1 * nb:rows * (n1 + m1) * nb]) + bytes(rows * m2 * nb)

    sock_dir = tempfile.mkdtemp(prefix="pgwf2")
    sock = os.path.join(sock_dir, "lane.sock")
    svc = fast.LaneService(sock, max_rows=8, executor=execute, shm=bool(args.shm))
    svc.set_model(0, [(it.n, it.m, it.kind) for it in items], [(ring.name, nb)])
    ticket = secrets.token_bytes(32)
    svc.add_ticket(ticket, "t", 0, 3600.0)
    ch = fast.TripChannel.connect_service_uds(sock, ticket, 30.0, 8, int(args.spin_us))
    if args.shm:
        assert ch.attach_shm() is True and ch.transport == "shm"
    prov = LaneFused2Provider(ch, [it.name for it in items], items_of(weights), ring.k, max_rows=8)
    src = Fused2MaskSource(weights, ring)
    passes = args.tokens + 3
    t0 = time.perf_counter()
    src.prefill(args.prompt_len + passes + 2)
    print(f"masks prefilled: {time.perf_counter() - t0:.1f}s", file=sys.stderr)
    be = Fused2MaskedBackend(ring, prov, src)
    model = TransformerModel(pack, be)
    model.set_float_ops("native")
    enable_fused2(model, weights, ring)
    nd = enable_fused2_native(model)

    g = torch.Generator().manual_seed(7)
    v = int(model.vocab_size)
    cache = model.new_cache(args.prompt_len + passes + 8)
    out = model.forward_batch([SequenceInput(torch.randint(0, v, (args.prompt_len,), generator=g), cache,
                                             all_logits=True)])
    walls = []
    for step in range(passes):
        if step == 3:
            fast.f2_phase_ns(True)
            tot0 = list(nd.totals)
        tok = torch.tensor([int(out[-1].reshape(-1, v)[-1].argmax())])
        t = time.perf_counter()
        out = model.forward_batch([SequenceInput(tok, cache)])
        walls.append(time.perf_counter() - t)
    ph = fast.f2_phase_ns(True)
    tot = [a - b for a, b in zip(nd.totals, tot0, strict=True)]
    kinds = ("single (first qkv)", "pair A (o + gate_up)", "pair B (down + next qkv)")
    per_kind = {}
    for i, name in enumerate(kinds):
        n = ph[6 + i]
        if n:
            per_kind[name] = {"trips": n, "before_lane_us": round(ph[i] / 1e3 / n, 1),
                              "after_lane_us": round(ph[3 + i] / 1e3 / n, 1),
                              "local_us": round((ph[i] + ph[3 + i]) / 1e3 / n, 1),
                              **({"unmask_us": round(ph[9 + i] / 1e3 / n, 1),
                                  "checks_us": round(ph[12 + i] / 1e3 / n, 1),
                                  "float_us": round(ph[15 + i] / 1e3 / n, 1)} if len(ph) >= 18 else {})}
    L = args.layers
    full = PRESETS[args.preset]["full_layers"]
    a = per_kind.get(kinds[1], {}).get("local_us", 0.0)
    b = per_kind.get(kinds[2], {}).get("local_us", 0.0)
    res = {
        "preset": args.preset, "layers": L, "ring": ring.name, "transport": "shm" if args.shm else "uds",
        "decode_passes": passes - 3, "trips_per_pass": 2 * L + 1,
        "local_us_per_trip": round(tot[3] / 1e3 / max(1, tot[2]), 1),
        "lane_us_per_trip": round(tot[4] / 1e3 / max(1, tot[2]), 1),
        "per_kind": per_kind,
        "mean_pair_trip_local_us": round((a + b) / 2, 1),
        "extrapolated_full_model_local_ms_per_token": round(full * (a + b) / 1e3, 2),
        "wall_ms_per_pass_median": round(1e3 * statistics.median(walls[3:]), 3),
        "bytes_per_trip_up_down": [prov.bytes_up // max(1, prov.calls), prov.bytes_down // max(1, prov.calls)],
    }
    print(json.dumps(res, indent=1))
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(res, fh, indent=1)
    prov.close()
    svc.stop(2.0)
    np.zeros(1)  # keep numpy imported for the extension's arrays


if __name__ == "__main__":
    main()
