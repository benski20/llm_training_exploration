#!/usr/bin/env python3
"""Fused Triton NTTs of the pks-v3 provider key switch (``pgw.service.pks_ntt``) against the torch path, on one GPU.

Sections (one JSON object, ``--json``):

* ``exact``: the fused transforms equal :class:`pgw.hemask.pks.Ntt` on every prime (random and edge-case
  polynomials), the fused key switch equals the torch key switch and the CPU key switch
  (:func:`pgw.hemask.pks_native.keyswitch`), and a larger random key switch equals torch on the device;
* ``ntt_us``: microseconds per 8192-point transform (fused forward incl. the operand gather, fused MAC + inverse,
  fused tail per reply ciphertext; torch forward / inverse) at several ``num_warps``;
* ``keyswitch``: microseconds per output row of the whole key switch (torch vs fused) for 1 and 4 groups;
* ``items``: one pks-v3 item (1024 columns) of the largest linears of 7B / 32B / 72B through ``ComputeKernel`` +
  ``PKSProvider`` (device hint path) with ``PGW_SERVICE_PKS_GPU_NTT=torch`` and ``fused``: seconds and byte-identical
  replies;
* ``cycle``: key-switch seconds of one 1024-slot decode cycle per model (every linear), from the per-row times.

Runs on CPU too (``--cpu``, Triton interpreter, tiny shapes): a smoke test of the script only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

if "--cpu" in sys.argv:  # Triton reads it at import
    os.environ.setdefault("TRITON_INTERPRET", "1")

import numpy as np
import torch

from pgw.hemask import pks
from pgw.hemask import pks_native as pn
from pgw.hemask.pks import NR, NU, PACK, RANK, D, PKSParams
from pgw.service import memplan, pks_gpu, pks_ntt

PAR = PKSParams()
LINEARS = {
    "qwen2.5-7b": [("0.gate_up", 37888, 3584), ("0.down", 3584, 18944), ("28.head", 152064, 3584)],
    "qwen2.5-32b": [("0.gate_up", 55296, 5120), ("0.down", 5120, 27648)],
    "qwen2.5-72b": [("0.gate_up", 59136, 8192), ("0.down", 8192, 29568)],
}


def _sync(dev: torch.device) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def _time(fn: Any, dev: torch.device, reps: int) -> float:
    """Median seconds of ``fn()`` over ``reps`` runs after one warm-up."""
    fn()
    _sync(dev)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        _sync(dev)
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def _keys(rng: np.random.Generator, groups: int) -> np.ndarray:
    return np.stack([np.stack([rng.integers(0, p, (RANK, NR), dtype=np.int64) for p in pks.QP])
                     for _ in range(groups)])


def exact(dev: torch.device, big_rows: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    rng = np.random.default_rng(1)
    hs = pks_gpu._edge_hint_rows(rng, 3 * PACK)
    hs[PACK:2 * PACK] = (PAR.q1 + 1) // 2
    hs[-PACK:] = PAR.q1 - 1
    hs_t = torch.from_numpy(hs.view(np.int64)).to(dev)
    nb = hs.shape[0] // PACK
    at = hs.astype(np.int64).reshape(nb, PACK, RANK, D).transpose(0, 2, 3, 1).reshape(nb, RANK, NR)
    at = np.where(at >= (PAR.q1 + 1) // 2, at - PAR.q1, at)
    ok_f, ok_i = [], []
    for p in pks.QP:
        nt = pks.Ntt.of(p)
        ah = pks_ntt.forward(hs_t, p)
        ref = nt.fwd(np.mod(at, p).astype(np.uint64))
        ok_f.append(bool(np.array_equal(ah.cpu().numpy().astype(np.uint64), ref)))
        ops = rng.integers(0, p, (2, RANK, NR), dtype=np.uint64)
        ops[0] = p - 1
        ops[1, :, :4] = [0, 1, p - 1, p - 2]
        got = pks_ntt.mac_inverse(ah, torch.from_numpy(ops.astype(np.uint32)).to(dev), p).cpu().numpy()
        acc = np.zeros((2, nb, NR), dtype=np.uint64)
        for li in range(RANK):
            acc = (acc + ref[None, :, li] * ops[:, None, li] % np.uint64(p)) % np.uint64(p)
        ok_i.append(bool(np.array_equal(got.astype(np.uint64), nt.inv(acc))))
    out["fwd_equals_pks_ntt"] = ok_f
    out["mac_inv_equals_pks_ntt"] = ok_i
    groups = 2
    y = rng.integers(0, 1 << 56, (hs.shape[0], D * groups), dtype=np.uint64)
    y[0, :3] = [0, (1 << 56) - 1, 1 << 55]
    ops = _keys(rng, groups)
    keys = pks_gpu._Keys(torch.from_numpy(ops).to(dev))
    y_t = torch.from_numpy(y.view(np.int64)).to(dev)
    fused = pks_ntt.keyswitch(hs_t, keys.u32(), y_t).cpu()
    out["keyswitch_fused_equals_torch"] = bool(torch.equal(fused, pks_gpu._keyswitch_torch(hs_t, keys.t, y_t).cpu()))
    out["keyswitch_fused_equals_cpu"] = bool(np.array_equal(fused.numpy().astype(np.uint64),
                                                            pn.keyswitch(hs, ops.astype(np.uint32), y)))
    try:
        pks_gpu.fused_self_test(dev)
        out["fused_self_test"] = True
    except Exception as e:
        out["fused_self_test"] = repr(e)
    # a larger random key switch (4 groups), fused vs torch on the device
    hs_b = torch.from_numpy(rng.integers(0, PAR.q1, (big_rows, NU), dtype=np.int64)).to(dev)
    y_b = torch.from_numpy(rng.integers(0, 1 << 56, (big_rows, 4 * D), dtype=np.int64)).to(dev)
    kb = pks_gpu._Keys(torch.from_numpy(_keys(rng, 4)).to(dev))
    out[f"keyswitch_{big_rows}rows_g4_fused_equals_torch"] = bool(torch.equal(
        pks_ntt.keyswitch(hs_b, kb.u32(), y_b), pks_gpu._keyswitch_torch(hs_b, kb.t, y_b)))
    out["all"] = all(v is True or (isinstance(v, list) and all(v)) for v in out.values())
    return out


def ntt_us(dev: torch.device, polys: int, reps: int, warps: list[int]) -> dict[str, Any]:
    """Microseconds per transform."""
    rng = np.random.default_rng(2)
    nb = max(1, polys // RANK)
    hs = torch.from_numpy(rng.integers(0, PAR.q1, (nb * PACK, NU), dtype=np.int64)).to(dev)
    out: dict[str, Any] = {"polys": nb * RANK}
    p = pks.QP[2]
    groups = 4
    ops_u = pks_gpu._Keys(torch.from_numpy(_keys(rng, groups)).to(dev)).u32()
    y = torch.from_numpy(rng.integers(0, 1 << 56, (nb * PACK, D * groups), dtype=np.int64)).to(dev)
    for w in warps:
        os.environ["PGW_SERVICE_PKS_NTT_WARPS"] = str(w)
        t0 = time.perf_counter()
        ah = pks_ntt.forward(hs, p)
        pks_ntt.mac_inverse(ah, ops_u[2], p)
        _sync(dev)
        comp = time.perf_counter() - t0
        tf = _time(lambda: pks_ntt.forward(hs, p), dev, reps)
        ti = _time(lambda a=ah: pks_ntt.mac_inverse(a, ops_u[2], p), dev, reps)
        tk = _time(lambda: pks_ntt.keyswitch(hs, ops_u, y), dev, reps)
        out[f"warps_{w}"] = {"first_call_s": round(comp, 2), "fwd_us": round(tf / (nb * RANK) * 1e6, 3),
                             "mac_inv_us": round(ti / (nb * groups) * 1e6, 3),
                             "keyswitch_g4_us_per_row": round(tk / (nb * PACK) * 1e6, 3)}
    os.environ.pop("PGW_SERVICE_PKS_NTT_WARPS", None)
    z, zi, ninv = pks_gpu._dev_tables(p, dev)
    x = (hs.view(nb, PACK, RANK, D).permute(0, 2, 3, 1).reshape(nb, RANK, NR) % p)
    tf = _time(lambda: pks_gpu._fwd(x, p, z), dev, reps)
    acc = pks_gpu._fwd(x, p, z)[:, :groups].contiguous()
    ti = _time(lambda: pks_gpu._inv(acc, p, zi, ninv), dev, reps)
    out["torch"] = {"fwd_us": round(tf / (nb * RANK) * 1e6, 3), "inv_us": round(ti / (nb * groups) * 1e6, 3)}
    return out


def keyswitch_rows(dev: torch.device, rows: int, reps: int) -> dict[str, Any]:
    rng = np.random.default_rng(3)
    hs = torch.from_numpy(rng.integers(0, PAR.q1, (rows, NU), dtype=np.int64)).to(dev)
    out: dict[str, Any] = {"rows": rows}
    for groups in (1, 4):
        y = torch.from_numpy(rng.integers(0, 1 << 56, (rows, D * groups), dtype=np.int64)).to(dev)
        keys = pks_gpu._Keys(torch.from_numpy(_keys(rng, groups)).to(dev))
        ops_u = keys.u32()
        tt = _time(lambda k=keys, y=y: pks_gpu._keyswitch_torch(hs, k.t, y), dev, reps)
        tf = _time(lambda o=ops_u, y=y: pks_ntt.keyswitch(hs, o, y), dev, reps)
        same = bool(torch.equal(pks_gpu._keyswitch_torch(hs, keys.t, y), pks_ntt.keyswitch(hs, ops_u, y)))
        out[f"g{groups}"] = {"torch_us_per_row": round(tt / rows * 1e6, 3),
                             "fused_us_per_row": round(tf / rows * 1e6, 3), "speedup": round(tt / tf, 2),
                             "identical": same}
    return out


def items(dev: torch.device, models: list[str], cols: int, cpu: bool, deadline: float) -> dict[str, Any]:
    from types import SimpleNamespace

    from pgw.core.types import LinearId, LinearKind, LinearSpec
    from pgw.hemask.wire import pack_bits
    from pgw.service.kernels import ComputeKernel
    from pgw.service.pks_provider import PKSProvider, pack_ksk

    rng = np.random.default_rng(7)
    groups = cols // D
    out: dict[str, Any] = {"cols": cols}
    for model in models:
        for key, m, n in LINEARS[model]:
            if time.time() > deadline:
                out[f"{model}:{key}"] = "skipped (time budget)"
                continue
            if cpu:
                m, n = min(m, 64), min(n, 64)
            layer, kind = key.split(".")
            lid = LinearId(int(layer), LinearKind.LM_HEAD if kind == "head" else LinearKind(kind))
            w = torch.randint(-128, 128, (m, n), dtype=torch.int8, generator=torch.Generator().manual_seed(3))
            kern = ComputeKernel(LinearSpec(lid, m, n), w, device=dev)
            del w
            b = rng.integers(0, 1 << 56, (n, cols), dtype=np.uint64)
            ops = np.stack([np.stack([rng.integers(0, p, (RANK, NR), dtype=np.uint64) for p in pks.QP])
                            for _ in range(groups)]).astype(np.uint32)
            it = SimpleNamespace(linear=key, cols=cols, b=pack_bits(b.reshape(-1), 56), b_bits=56, ksk=pack_ksk(ops))
            seed = os.urandom(32)
            res: dict[str, Any] = {"m": m, "n": n}
            replies = {}
            for mode in ("torch", "fused"):
                os.environ["PGW_SERVICE_PKS_GPU_NTT"] = mode
                pv = PKSProvider(mode="device")
                pv.use_device("fp", kern, key)  # self-tests outside the timing
                bytes(pv.run_item("fp", kern, it, seed))  # warm-up (kernel compilation, allocator)
                _sync(dev)
                t0 = time.perf_counter()
                replies[mode] = bytes(pv.run_item("fp", kern, it, seed))
                res[f"{mode}_s"] = round(time.perf_counter() - t0, 3)
            os.environ.pop("PGW_SERVICE_PKS_GPU_NTT", None)
            res["identical"] = replies["torch"] == replies["fused"]
            res["speedup"] = round(res["torch_s"] / max(res["fused_s"], 1e-9), 2)
            res["model_estimate_device_s"] = round(sum(memplan.pks_item_seconds(m, n, cols, "device").values()), 3)
            out[f"{model}:{key}"] = res
            print(model, key, res, flush=True)
            del kern
            if dev.type == "cuda":
                torch.cuda.empty_cache()
    return out


def cycle(ks: dict[str, Any]) -> dict[str, Any]:
    """Key-switch seconds per 1024-slot cycle (every linear, 4 groups) from the per-row times."""
    g4 = ks["g4"]
    out = {}
    for model in ("qwen2.5-7b", "qwen2.5-32b", "qwen2.5-72b"):
        rows = sum(m for _, m, _ in memplan.MODELS[model].linears(True))
        out[model] = {"rows": rows, "torch_s": round(rows * g4["torch_us_per_row"] / 1e6, 1),
                      "fused_s": round(rows * g4["fused_us_per_row"] / 1e6, 1)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--json", default="")
    ap.add_argument("--cpu", action="store_true", help="smoke test on CPU (Triton interpreter, tiny shapes)")
    ap.add_argument("--models", default="qwen2.5-7b,qwen2.5-32b,qwen2.5-72b")
    ap.add_argument("--warps", default="4,8,16")
    ap.add_argument("--budget-s", type=float, default=2400)
    a = ap.parse_args()
    t_start = time.time()
    if a.cpu:
        os.environ.setdefault("TRITON_INTERPRET", "1")
    dev = torch.device("cpu" if a.cpu or not torch.cuda.is_available() else "cuda:0")
    import triton

    res: dict[str, Any] = {"device": torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu",
                           "torch": torch.__version__, "triton": triton.__version__}
    path = a.json

    def save() -> None:
        if path:
            with open(path, "w") as f:
                json.dump(res, f, indent=1)

    def section(name: str, fn: Any) -> None:
        t0 = time.time()
        try:
            res[name] = fn()
        except Exception as e:
            import traceback

            res[name] = {"error": f"{type(e).__name__}: {str(e)[:800]}", "trace": traceback.format_exc()[-3000:]}
        print(name, round(time.time() - t0, 1), "s:", json.dumps(res[name])[:2000], flush=True)
        save()

    small = a.cpu
    section("exact", lambda: exact(dev, PACK * (2 if small else 64)))
    section("ntt_us", lambda: ntt_us(dev, 16 if small else 2048, 1 if small else 5,
                                     [8] if small else [int(w) for w in a.warps.split(",")]))
    best = None
    if isinstance(res.get("ntt_us"), dict):
        cand = {k: v["keyswitch_g4_us_per_row"] for k, v in res["ntt_us"].items() if k.startswith("warps_")}
        if cand:
            best = min(cand, key=cand.get).split("_")[1]
            os.environ["PGW_SERVICE_PKS_NTT_WARPS"] = best
    res["warps_used"] = best
    section("keyswitch", lambda: keyswitch_rows(dev, PACK if small else 2048, 1 if small else 5))
    if isinstance(res.get("keyswitch"), dict) and "g4" in res["keyswitch"]:
        section("cycle", lambda: cycle(res["keyswitch"]))
    section("items", lambda: items(dev, a.models.split(","), 1024 if not small else 256, small,
                                   t_start + a.budget_s))
    res["total_s"] = round(time.time() - t_start, 1)
    save()
    ok = isinstance(res.get("exact"), dict) and res["exact"].get("all") is True
    ok = ok and all(v.get("identical", True) for v in res.get("items", {}).values() if isinstance(v, dict))
    print("OK" if ok else "FAILED", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
