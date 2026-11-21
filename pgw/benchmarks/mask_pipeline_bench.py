#!/usr/bin/env python3
"""Gateway mask pipeline throughput: factory writer -> disk store -> batched decode consumption -> wipe.

    PGW_OFFLINE=1 PYTHONPATH=src python benchmarks/mask_pipeline_bench.py --model 7b --layers 4 \
        --slots 256 --sessions 32 --tokens 8 [--root DIR] [--json OUT]

Phases (all on the real :class:`pgw.maskstore.MaskStore`, ring K32A12C27):

* **write**: ``SegmentWriter.write_linear`` with compact int32 ``W r`` rows (the factory's GPU form),
  ``commit`` (fsync + rename + SQLite registration). Reports GB/s of ``W r`` plaintext and slots/s.
* **reserve**: ``sessions`` reservations of ``tokens`` slots each (durable cursor txn).
* **decode**: batched decode, one row per (session, linear) per step: ``take_checked(lid, 1, True)``
  for every session and linear, ``tokens`` steps. Reports slot-tokens/s and GB/s of ``r`` + ``W r``
  handed to the masking kernels. ``--claim-threads N`` claims through
  :func:`pgw.maskstore.take_checked_many` (all sessions of a linear per call, N native threads).
* **close/gc**: closing reservations (zeroization) and the secure wipe of consumed segments.

Slot numbers are per *full model* slot: with ``--layers`` below the model's layer count the measured
work is scaled by ``layers_total / layers`` (per-layer work is identical).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from pgw.core.ring import RingParams
from pgw.factory.bench import MODEL_PRESETS
from pgw.maskstore import MaskStore, StaticKeyProvider

FP = "ab" * 32
ALIASES = {"0.5b": "qwen2.5-0.5b", "7b": "qwen2.5-7b", "70b": "llama3-70b", "8b": "llama3-8b"}


def _specs(model: str, layers: int):
    shape = MODEL_PRESETS[ALIASES.get(model, model)]
    specs = []
    for layer in range(layers):
        specs.extend(shape.layer_specs(layer))
    return shape, specs


def run(args: argparse.Namespace) -> dict:
    shape, specs = _specs(args.model, args.layers)
    ring = RingParams.from_name(args.ring)
    scale = shape.layers / args.layers
    wr_bytes_slot = sum(s.out_features for s in specs) * ring.bytes_per_element
    r_bytes_slot = sum(s.in_features for s in specs) * 8  # int64 ring tensors handed out
    if args.root:
        Path(args.root).mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="pgw-maskbench-", dir=args.root))  # fresh store every run
    res: dict = {
        "model": shape.name, "layers": args.layers, "layers_total": shape.layers, "ring": ring.name,
        "slots": args.slots, "segments": args.segments, "sessions": args.sessions, "tokens": args.tokens,
        "wr_mb_per_full_slot": wr_bytes_slot * scale / 1e6, "fsync": not args.no_fsync,
        "claim_threads": args.claim_threads,
    }
    store = MaskStore.open(root / "store", StaticKeyProvider(os.urandom(32)), low_water_slots=0,
                           allow_no_external_counter=True, fsync=not args.no_fsync)
    try:
        g = np.random.default_rng(0)
        rows = args.write_rows
        bufs = {s.out_features: torch.from_numpy(g.integers(-2**31, 2**31, (rows, s.out_features), dtype=np.int32))
                for s in specs}
        uws = {s.in_features: torch.from_numpy(g.integers(0, 2**32, (2, s.in_features), dtype=np.int64))
               for s in specs}
        t0 = time.perf_counter()
        for _ in range(args.segments):
            w = store.writer(FP, ring, specs, slots=args.slots, checks=2, chunk_slots=args.chunk_slots)
            for s in specs:
                done = 0
                while done < args.slots:
                    n = min(rows, args.slots - done)
                    w.write_linear(s.id, bufs[s.out_features][:n], uws[s.in_features])
                    done += n
            w.commit()
        tw = time.perf_counter() - t0
        total_slots = args.slots * args.segments
        res["write_s"] = tw
        res["write_gbps"] = total_slots * wr_bytes_slot / tw / 1e9
        res["write_full_slots_per_s"] = total_slots / tw / scale

        t0 = time.perf_counter()
        resv = [store.reserve(FP, ring, args.tokens, session_id=f"s{i}") for i in range(args.sessions)]
        res["reserve_ms_each"] = (time.perf_counter() - t0) / args.sessions * 1e3

        lids = [s.id for s in specs]
        if args.prefetch:
            for rv in resv:
                for lid in lids:
                    rv.prefetch(lid, args.tokens)
        t0 = time.perf_counter()
        for _ in range(args.tokens):
            if args.claim_threads is not None:
                from pgw.maskstore import take_checked_many  # not in older trees (A/B runs)

                for lid in lids:  # one engine step per linear: every session's row of it
                    take_checked_many([(rv, lid, 1, True) for rv in resv], threads=args.claim_threads)
            else:
                for lid in lids:
                    for rv in resv:
                        rv.take_checked(lid, 1, True)
        td = time.perf_counter() - t0
        slot_tokens = args.tokens * args.sessions
        res["decode_s"] = td
        res["decode_full_slots_per_s"] = slot_tokens / td / scale
        res["decode_gbps_wr"] = slot_tokens * wr_bytes_slot / td / 1e9
        res["decode_gbps_r_wr"] = slot_tokens * (wr_bytes_slot + r_bytes_slot) / td / 1e9
        res["decode_us_per_take"] = td / (slot_tokens * len(lids)) * 1e6

        t0 = time.perf_counter()
        for rv in resv:
            rv.close()
        res["close_ms_each"] = (time.perf_counter() - t0) / args.sessions * 1e3
        # Consume the rest and wipe consumed segments.
        left = store.available_slots(FP, ring)
        if left:
            store.reserve(FP, ring, left).close()
        t0 = time.perf_counter()
        n = store.gc()
        store.wait_for_wipes(600)
        tg = time.perf_counter() - t0
        res["gc_segments"] = n
        res["gc_s"] = tg
        res["gc_gbps"] = total_slots * wr_bytes_slot / tg / 1e9 if tg > 0 else None
    finally:
        store.close()
        shutil.rmtree(root, ignore_errors=True)
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="0.5b")
    ap.add_argument("--layers", type=int, default=0, help="layers to materialise (0: all)")
    ap.add_argument("--ring", default="K32A12C27")
    ap.add_argument("--slots", type=int, default=256, help="slots per segment")
    ap.add_argument("--segments", type=int, default=2)
    ap.add_argument("--chunk-slots", type=int, default=None)
    ap.add_argument("--write-rows", type=int, default=64, help="rows per write_linear call")
    ap.add_argument("--sessions", type=int, default=32)
    ap.add_argument("--tokens", type=int, default=8, help="decode steps (slots per session)")
    ap.add_argument("--prefetch", action="store_true", help="stage every session's rows before decode")
    ap.add_argument("--claim-threads", type=int, default=None,
                    help="claim through pgw.maskstore.take_checked_many (native threads; 0 = auto)")
    ap.add_argument("--no-fsync", action="store_true")
    ap.add_argument("--root", default=None, help="parent directory of the (temporary) store")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    if not args.layers:
        args.layers = MODEL_PRESETS[ALIASES.get(args.model, args.model)].layers
    res = run(args)
    for k, v in res.items():
        print(f"{k:>26}: {v:.4g}" if isinstance(v, float) else f"{k:>26}: {v}")
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
