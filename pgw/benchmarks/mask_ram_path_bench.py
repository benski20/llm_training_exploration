#!/usr/bin/env python3
"""Sustained mask data path: factory writer -> in-memory ring (or disk store) -> gateway takes, concurrently.

    PGW_OFFLINE=1 PYTHONPATH=src python benchmarks/mask_ram_path_bench.py --model 7b --layers 2 \
        --tier ring --seconds 6 --consumers 2 --sessions 16 --tokens 16 [--json OUT]

Unlike ``mask_pipeline_bench.py`` (phases one after the other), a producer thread keeps writing
segments (compact int32 ``W r`` rows, the factory's GPU form) while ``--consumers`` threads each run
``--sessions`` reservations of ``--tokens`` slots through a batched decode: one ``take_checked`` row
per (session, linear) per step, then ``close`` (wipe/burn). Reports the delivered rate (slots and
GB/s of ``W r``) at steady state, the producer and consumer stall shares, and CPU seconds per slot.
Also measures the per-slot floors on this host: a single-thread ``memcpy`` of the ``W r`` bytes and the
ChaCha20 keystream the gateway expands for ``r`` (8 keystream bytes per element, SPEC 7.2).

``--tier store`` runs the same loop on the encrypted disk store (AES-GCM, SQLite cursor; fsync off
unless ``--fsync``): the outage reserve's rate. Slot numbers are per full-model slot (measured work
x ``layers_total / layers``).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import torch

from pgw.core.errors import MaskExhaustedError, MaskStoreError
from pgw.core.prng import SeedExpander
from pgw.core.ring import RingParams
from pgw.factory.bench import MODEL_PRESETS
from pgw.maskstore import MaskStore, StaticKeyProvider
from pgw.maskstore.format import auto_chunk_slots

FP = "cd" * 32
ALIASES = {"0.5b": "qwen2.5-0.5b", "7b": "qwen2.5-7b", "70b": "llama3-70b", "8b": "llama3-8b"}


def _specs(model: str, layers: int):
    shape = MODEL_PRESETS[ALIASES.get(model, model)]
    layers = layers or shape.layers
    specs = []
    for layer in range(layers):
        specs.extend(shape.layer_specs(layer))
    return shape, layers, specs


def floors(specs, ring: RingParams, scale: float, reps: int = 5) -> dict:
    """Per-full-slot single-thread floors: memcpy of the W r bytes (a full slot's worth, so the copy is
    not cache resident), ChaCha20 keystream for every r row (measured layers, scaled)."""
    wr = int(sum(s.out_features for s in specs) * ring.bytes_per_element * scale)
    src = np.random.default_rng(1).integers(0, 255, wr, dtype=np.uint8)
    dst = np.empty_like(src)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        np.copyto(dst, src)
        ts.append(time.perf_counter() - t0)
    key = bytearray(SeedExpander(os.urandom(32)).stream_key("r", "x"))
    tr = []
    for _ in range(reps):
        t0 = time.perf_counter()
        for s in specs:
            SeedExpander.ring_uniform_rows_with_key(key, s.in_features, ring.k, 7, 1)
        tr.append(time.perf_counter() - t0)
    return {"memcpy_wr_ms": min(ts) * 1e3, "r_keystream_ms": min(tr) * 1e3 * scale}


def run(args: argparse.Namespace) -> dict:
    shape, layers, specs = _specs(args.model, args.layers)
    ring = RingParams.from_name(args.ring)
    scale = shape.layers / layers
    wr_bytes = sum(s.out_features for s in specs) * ring.bytes_per_element
    root = Path(tempfile.mkdtemp(prefix="pgw-rampath-", dir=args.root))
    cs = auto_chunk_slots([s.out_features for s in specs], ring)
    mring = None
    store = None
    if args.tier == "ring":
        from pgw.maskstore.ringbuf import MaskRing, SeedRegistry, ring_geometry

        geo = ring_geometry(specs, ring, 1 << 40, segment_slots=args.slots)
        mring = MaskRing(FP, ring, specs, capacity_bytes=args.slabs * geo.slab_bytes, segment_slots=args.slots,
                         allow_unlocked=True, min_soundness_bits=0.0, seeds=SeedRegistry())
    else:
        store = MaskStore.open(root / "store", StaticKeyProvider(os.urandom(32)), low_water_slots=0,
                               allow_no_external_counter=True, fsync=args.fsync)
    g = np.random.default_rng(0)
    rows = args.write_rows
    bufs = {s.out_features: torch.from_numpy(g.integers(-2**31, 2**31, (rows, s.out_features), dtype=np.int32))
            for s in specs}
    uws = {s.in_features: torch.from_numpy(g.integers(0, 2**32, (2, s.in_features), dtype=np.int64)) for s in specs}
    lids = [s.id for s in specs]
    stop = threading.Event()
    stats = {"produced": 0, "prod_wait_s": 0.0, "consumed": 0, "cons_wait_s": 0.0, "errors": []}
    lock = threading.Lock()

    def available() -> int:
        return int(mring.available) if mring is not None else store.available_slots(FP, ring)

    def producer() -> None:
        while not stop.is_set():
            if store is not None and available() >= args.store_cap_slots:
                time.sleep(0.002)  # the disk tier is capped like a reserve (else the producer runs away)
                continue
            try:
                if mring is not None:
                    w = mring.writer(FP, ring, specs, slots=args.slots, seed=os.urandom(32), checks=2, chunk_slots=cs)
                else:
                    w = store.writer(FP, ring, specs, slots=args.slots, checks=2)
            except MaskStoreError:  # RingFull: wait for a wipe
                t0 = time.perf_counter()
                time.sleep(0.001)
                with lock:
                    stats["prod_wait_s"] += time.perf_counter() - t0
                continue
            try:
                for s in specs:
                    done = 0
                    while done < args.slots:
                        n = min(rows, args.slots - done)
                        w.write_linear(s.id, bufs[s.out_features][:n], uws[s.in_features])
                        done += n
                w.commit()
            except MaskStoreError:
                if stop.is_set():
                    return  # the ring closed under the last fill
                raise
            with lock:
                stats["produced"] += args.slots

    def consumer() -> None:
        while not stop.is_set():
            t0 = time.perf_counter()
            try:
                if mring is not None:
                    resv = [mring.reserve(args.tokens, session_id="b", wait_s=0.0) for _ in range(args.sessions)]
                else:
                    resv = [store.reserve(FP, ring, args.tokens, session_id="b") for _ in range(args.sessions)]
            except (MaskExhaustedError, MaskStoreError):
                time.sleep(0.001)
                with lock:
                    stats["cons_wait_s"] += time.perf_counter() - t0
                continue
            except Exception as exc:  # pragma: no cover - reported
                stats["errors"].append(repr(exc))
                return
            try:
                for _ in range(args.tokens):
                    for lid in lids:
                        for rv in resv:
                            _, r, wr, _ = rv.take_checked(lid, 1, True)
                            r.numpy().fill(0)  # what the engine does with consumed pads
                            wr.numpy().fill(0)
            finally:
                for rv in resv:
                    rv.close()
            with lock:
                stats["consumed"] += args.tokens * args.sessions

    threads = [threading.Thread(target=producer, daemon=True)]
    threads += [threading.Thread(target=consumer, daemon=True) for _ in range(args.consumers)]
    try:
        # prime: let the producer fill before measuring (steady state, not a cold start)
        threads[0].start()
        t_end = time.monotonic() + 30
        while available() < args.sessions * args.tokens * args.consumers and time.monotonic() < t_end:
            time.sleep(0.01)
        for t in threads[1:]:
            t.start()
        time.sleep(args.warmup)
        with lock:
            p0, c0, pw0, cw0 = stats["produced"], stats["consumed"], stats["prod_wait_s"], stats["cons_wait_s"]
        cpu0, t0 = time.process_time(), time.perf_counter()
        time.sleep(args.seconds)
        with lock:
            p1, c1, pw1, cw1 = stats["produced"], stats["consumed"], stats["prod_wait_s"], stats["cons_wait_s"]
        cpu1, t1 = time.process_time(), time.perf_counter()
        stop.set()
        for t in threads:
            t.join(60)
    finally:
        if mring is not None:
            mring.close()
        if store is not None:
            store.close()
        shutil.rmtree(root, ignore_errors=True)
    dt = t1 - t0
    full_c = (c1 - c0) / scale
    res = {
        "model": shape.name, "layers": layers, "layers_total": shape.layers, "ring": ring.name, "tier": args.tier,
        "consumers": args.consumers, "sessions": args.sessions, "tokens": args.tokens, "seconds": dt,
        "wr_mb_per_full_slot": wr_bytes * scale / 1e6,
        "produced_full_slots_per_s": (p1 - p0) / scale / dt,
        "consumed_full_slots_per_s": full_c / dt,
        "consumed_gbps_wr": (c1 - c0) * wr_bytes / dt / 1e9,
        "producer_wait_share": (pw1 - pw0) / dt,
        "consumer_wait_share": (cw1 - cw0) / dt / max(1, args.consumers),
        "cpu_ms_per_full_slot": (cpu1 - cpu0) / max(full_c, 1e-9) * 1e3,
        "errors": stats["errors"],
    }
    fl = floors(specs, ring, scale)
    res["floor_memcpy_wr_ms_per_full_slot"] = fl["memcpy_wr_ms"]
    res["floor_r_keystream_ms_per_full_slot"] = fl["r_keystream_ms"]
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="0.5b")
    ap.add_argument("--layers", type=int, default=0, help="layers to materialise (0: all)")
    ap.add_argument("--ring", default="K32A12C27")
    ap.add_argument("--tier", choices=("ring", "store"), default="ring")
    ap.add_argument("--slots", type=int, default=256, help="slots per segment (slab)")
    ap.add_argument("--slabs", type=int, default=8)
    ap.add_argument("--write-rows", type=int, default=64)
    ap.add_argument("--consumers", type=int, default=2)
    ap.add_argument("--sessions", type=int, default=16)
    ap.add_argument("--tokens", type=int, default=16)
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--warmup", type=float, default=1.0)
    ap.add_argument("--store-cap-slots", type=int, default=4096)
    ap.add_argument("--fsync", action="store_true")
    ap.add_argument("--root", default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    res = run(args)
    for k, v in res.items():
        print(f"{k:>36}: {v:.4g}" if isinstance(v, float) else f"{k:>36}: {v}")
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
