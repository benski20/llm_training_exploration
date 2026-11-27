#!/usr/bin/env python3
"""Mask cost at consumer scale (``research/reviews/mask-cost-consumer.md``): two device measurements.

* ``client``: the customer's pks-v3 kernels on the device (:mod:`pgw.hemask.pks_torch`, exact eager torch int64)
  per distinct linear shape of Qwen2.5-7B / 32B (masked head included): seconds per 256-slot group of the upload
  bodies (pad NTTs + CRT + layout; with the public ``a``'s transforms per group and cached), the reply key and the
  decryption, and a noise-sampling proxy (CBD(21) popcounts of device random bits: timing only, not the stream);
  summed to **ms per slot** over the model. Exactness is the pytest step (``PGW_TEST_CUDA=1``).
* ``coalesce``: provider items of one group each, ``S`` of them run one by one (``run_item``) against the same ``S``
  coalesced into one device item (``run_items``: one hint computation for all groups); replies must be
  byte-identical.

Usage: ``python benchmarks/pks_client_gpu_bench.py --json out.json [--sections client,coalesce] [--budget-s 1200]``.
"""
# ruff: noqa: B023 (every lambda is timed inside its own loop iteration)

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from pgw.hemask import pks
from pgw.hemask import pks_torch as pt
from pgw.hemask.pks import NR, NU, PACK, QP, RANK, D

MODEL_KEYS = {"7B": "qwen2.5-7b", "32B": "qwen2.5-32b"}


def _sync(dev: torch.device) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def _time(fn: Any, dev: torch.device, reps: int) -> float:
    fn()
    _sync(dev)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        _sync(dev)
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def shapes(model: str) -> Counter[tuple[int, int]]:
    from pgw.service.memplan import MODELS

    return Counter((m, n) for _, m, n in MODELS[MODEL_KEYS[model]].linears(True))


def _cbd_proxy(n: int, dev: torch.device) -> torch.Tensor:
    raw = torch.randint(-(1 << 62), 1 << 62, (n, D), dtype=torch.int64, device=dev)
    mask = (1 << 21) - 1

    def pop(x: torch.Tensor) -> torch.Tensor:  # SWAR popcount of 21-bit fields
        x = x - ((x >> 1) & 0x55555555)
        x = (x & 0x33333333) + ((x >> 2) & 0x33333333)
        x = (x + (x >> 4)) & 0x0F0F0F0F
        return (x * 0x01010101 & 0xFFFFFFFF) >> 24

    return pop(raw & mask) - pop((raw >> 21) & mask)


def client(dev: torch.device, models: list[str], reps: int, deadline: float) -> dict[str, Any]:
    rng = np.random.default_rng(5)
    out: dict[str, Any] = {"device": str(dev), "per_shape": {}, "ms_per_slot": {}, "ms_per_slot_cached_a": {},
                           "parts_ms_per_slot": {}}
    k_hat = torch.from_numpy(pks.ks_a(b"k" * 32).view(np.int64)).to(dev)
    tr = torch.from_numpy(rng.integers(-1, 2, NR)).to(dev)
    ek = torch.from_numpy(rng.integers(-21, 22, (RANK, NR))).to(dev)
    tc = torch.from_numpy(rng.integers(-1, 2, (RANK, D))).to(dev)
    t = torch.from_numpy(rng.integers(-1, 2, NU)).to(dev)
    cache: dict[tuple[int, int], dict[str, float]] = {}
    for model in models:
        tot = Counter[str]()
        for (m, n), cnt in sorted(shapes(model).items()):
            if time.time() > deadline:
                out["per_shape"][f"{model}:{m}x{n}"] = "skipped (time budget)"
                continue
            if (m, n) not in cache:
                a = torch.from_numpy(rng.integers(0, 1 << 56, (n // RANK, NU), dtype=np.uint64).view(np.int64)).to(dev)
                r = torch.from_numpy(rng.integers(0, 1 << 32, (D, n))).to(dev)
                e = torch.from_numpy(rng.integers(-21, 22, (n, D))).to(dev)
                ahat = pt.pad_ahat(a, dev)
                c0r = torch.from_numpy(rng.integers(0, 1 << 36, (m // PACK, NR))).to(dev)
                hr = [torch.from_numpy(rng.integers(0, p, (m // PACK, NR))).to(dev) for p in pks.Q1]
                fused: dict[str, Any] = {}
                if dev.type == "cuda":
                    try:
                        same = bool(torch.equal(pt.decrypt_fused(c0r, hr, tr, D, dev), pt.decrypt(c0r, hr, tr, D, dev)))
                        fused = {"decrypt_fused_s": _time(lambda: pt.decrypt_fused(c0r, hr, tr, D, dev), dev, reps),
                                 "decrypt_fused_identical": float(same)}
                    except Exception as exc:  # pragma: no cover - GPU only
                        out["decrypt_fused_error"] = repr(exc)
                res = {
                    **fused,
                    "encrypt_s": _time(lambda: pt.encrypt_body(r, a, t, e, dev), dev, reps),
                    "encrypt_cached_a_s": _time(lambda: pt.encrypt_body(r, a, t, e, dev, ahat), dev, reps),
                    "noise_proxy_s": _time(lambda: _cbd_proxy(n, dev), dev, reps),
                    "reply_key_s": _time(lambda: pt.reply_key(tc, k_hat, tr, ek, dev), dev, reps),
                    "decrypt_s": _time(lambda: pt.decrypt(c0r, hr, tr, D, dev), dev, reps),
                }
                cache[(m, n)] = res
                del a, r, e, ahat, c0r, hr
                if dev.type == "cuda":
                    torch.cuda.empty_cache()
            res = cache[(m, n)]
            out["per_shape"][f"{model}:{m}x{n}"] = {"count": cnt, **{k: round(v, 5) for k, v in res.items()}}
            for k, v in res.items():
                tot[k] += cnt * v
            print(model, m, n, cnt, res, flush=True)
        per = {k: 1e3 * v / D for k, v in tot.items() if not k.endswith("_identical")}
        ident = [cache[s]["decrypt_fused_identical"] for s in shapes(model) if "decrypt_fused_identical" in
                 cache.get(s, {})]
        out.setdefault("decrypt_fused_identical", {})[model] = bool(ident) and all(x == 1.0 for x in ident)
        out["parts_ms_per_slot"][model] = {k: round(v, 3) for k, v in per.items()}
        base = per.get("noise_proxy_s", 0) + per.get("reply_key_s", 0) + per.get("decrypt_s", 0)
        out["ms_per_slot"][model] = round(base + per.get("encrypt_s", 0), 3)
        out["ms_per_slot_cached_a"][model] = round(base + per.get("encrypt_cached_a_s", 0), 3)
    return out


def _joined(replies: list[bytes], m: int) -> bytes:
    """The wire bytes of one item whose groups' replies are ``replies`` (each ``pack_bits(., 36)`` of one group)."""
    from pgw.hemask.wire import pack_bits, unpack_bits

    vals = np.concatenate([unpack_bits(r, (m // PACK) * NR, 36) for r in replies])
    return bytes(pack_bits(vals, 36))


def _unpack(data: bytes, n: int) -> np.ndarray:
    from pgw.hemask.wire import unpack_bits

    return unpack_bits(data, n * D, 56).reshape(n, D)


def coalesce(dev: torch.device, s_items: int, deadline: float) -> dict[str, Any]:
    from pgw.core.types import LinearId, LinearKind, LinearSpec
    from pgw.hemask.wire import pack_bits
    from pgw.service.kernels import ComputeKernel
    from pgw.service.pks_provider import PKSProvider, pack_ksk

    rng = np.random.default_rng(9)
    cases = [("7B", "0.gate_up", 37888, 3584), ("7B", "0.down", 3584, 18944), ("32B", "0.gate_up", 55296, 5120),
             ("32B", "0.down", 5120, 27648), ("32B", "0.qkv", 7168, 5120)]
    if dev.type != "cuda":
        cases = [("cpu", "0.o", 64, 48)]
    out: dict[str, Any] = {"items": s_items}
    seed = pks.crs_a_seed("bench")
    for model, key, m, n in cases:
        if time.time() > deadline:
            out[f"{model}:{key}"] = "skipped (time budget)"
            continue
        layer, kind = key.split(".")
        w = torch.randint(-128, 128, (m, n), dtype=torch.int8, generator=torch.Generator().manual_seed(3))
        kern = ComputeKernel(LinearSpec(LinearId(int(layer), LinearKind(kind)), m, n), w, device=dev,
                             **({} if dev.type == "cuda" else {"backend": "cpu_int8"}))
        del w
        its = []
        for _ in range(s_items):
            b = rng.integers(0, 1 << 56, (n, D), dtype=np.uint64)
            ops = np.stack([rng.integers(0, p, (RANK, NR), dtype=np.uint64) for p in QP])[None].astype(np.uint32)
            its.append(SimpleNamespace(linear=key, cols=D, b=pack_bits(b.reshape(-1), 56), b_bits=56,
                                       ksk=pack_ksk(ops)))
        pv = PKSProvider(mode="device")
        if dev.type != "cuda":
            pv._item_ok["cpu"] = True
        assert pv.use_device("fp", kern, key)
        bytes(pv.run_item("fp", kern, its[0], seed))  # warm-up
        _sync(dev)
        t0 = time.perf_counter()
        alone = [bytes(pv.run_item("fp", kern, it, seed)) for it in its]
        _sync(dev)
        t_alone = time.perf_counter() - t0
        # baseline: ONE item of the same groups (one customer's upload of S groups), the upper bound of coalescing
        bcat = np.concatenate([np.asarray(_unpack(it.b, n)) for it in its], axis=1)
        one = SimpleNamespace(linear=key, cols=D * s_items, b=pack_bits(bcat.reshape(-1), 56), b_bits=56,
                              ksk=b"".join(bytes(it.ksk) for it in its))
        bytes(pv.run_item("fp", kern, one, seed))
        _sync(dev)
        t0 = time.perf_counter()
        y_one = bytes(pv.run_item("fp", kern, one, seed))
        _sync(dev)
        t_one = time.perf_counter() - t0
        pv.run_items("fp", kern, its[:2], seed)  # warm-up of the wider item
        _sync(dev)
        t0 = time.perf_counter()
        co = [bytes(y) for y in pv.run_items("fp", kern, its, seed)]
        _sync(dev)
        t_co = time.perf_counter() - t0
        res = {"m": m, "n": n, "alone_s": round(t_alone, 4), "coalesced_s": round(t_co, 4),
               "one_item_s": round(t_one, 4), "speedup": round(t_alone / max(t_co, 1e-9), 3),
               "speedup_one_item": round(t_alone / max(t_one, 1e-9), 3), "identical": alone == co,
               "one_item_identical": y_one == _joined(co, m),
               "coalesced_runs": pv.coalesced_runs}
        try:
            from pgw.service import memplan

            if dev.type == "cuda":
                t1 = sum(memplan.pks_item_seconds(m, n, D, "device", memplan.T4_RATES, ntt="fused",
                                                  kernels="all").values())
                ts = sum(memplan.pks_item_seconds(m, n, D * s_items, "device", memplan.T4_RATES, ntt="fused",
                                                  kernels="all").values())
                res["model_t4_speedup"] = round(s_items * t1 / ts, 3)
        except Exception as exc:  # pragma: no cover - model only
            res["model_error"] = str(exc)
        out[f"{model}:{key}"] = res
        print(model, key, res, flush=True)
        del kern, its
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    ap.add_argument("--sections", default="client,coalesce")
    ap.add_argument("--models", default="7B,32B")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--items", type=int, default=8)
    ap.add_argument("--budget-s", type=float, default=1200.0)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    deadline = time.time() + a.budget_s
    res: dict[str, Any] = {"device": str(dev), "torch": torch.__version__}
    if dev.type == "cuda":
        res["gpu"] = torch.cuda.get_device_name(dev)

    def save() -> None:
        if a.json:
            with open(a.json, "w") as f:
                json.dump(res, f, indent=1)

    secs = a.sections.split(",")
    if "coalesce" in secs:
        res["coalesce"] = coalesce(dev, a.items, deadline)
        save()
    if "client" in secs:
        models = a.models.split(",") if dev.type == "cuda" else []
        res["client"] = client(dev, models, a.reps, deadline)
        if "client" in res and res["client"]["ms_per_slot"]:
            res["ms_per_slot"] = res["client"]["ms_per_slot"]
        save()
    print(json.dumps(res, indent=1))
    _ = os
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
