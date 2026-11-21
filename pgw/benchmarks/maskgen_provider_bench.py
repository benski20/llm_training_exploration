"""Provider-side encrypted maskgen throughput (``POST /pgw/v1/maskgen`` compute) at real layer shapes.

One layer of each linear kind is loaded on the GPU(s) with random int8 weights and random ciphertext
payloads (any bytes are valid MLWE coefficients for the timing). Items go through the service's own
item function (:func:`pgw.service.handler._maskgen_item`) from a thread pool of ``--conc`` workers per
device, the way the service runs them, many rounds back to back (steady state). Two paths:

* ``legacy``: ``PGW_SERVICE_MASKGEN_FUSED=0`` (the previous CUDA path: v2 unpack/pack on the host,
  8 planes for v2, unfused combine),
* ``fused``: :mod:`pgw.service.maskgen_gpu` (device-side unpack/pack, ``ceil(bb/8)`` planes,
  Triton combine).

Reported per preset/format/groups/conc: seconds per layer round, model slots/s (one slot = one token
row through every linear: ``256 * groups / (layers * s_per_round)``), effective int8 TOPS
(``2 * planes * cols * sum(m n)`` per second) and the 7-plane equivalent. Outputs of both paths are
compared byte for byte (and with the CPU reference on a row slice). With two GPUs it also measures
layers sharded over both devices. ``--decode`` measures the latency of a small decode GEMM alone and
next to saturating maskgen (with and without ``PGW_SERVICE_MASKGEN_YIELD_MS``).

    python benchmarks/maskgen_provider_bench.py --presets qwen2.5-0.5b qwen2.5-7b llama3-70b --json out.json
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import torch

from pgw.core.ring import RingParams
from pgw.core.types import LinearId, LinearSpec
from pgw.hemask import DEGREE
from pgw.hemask.lwe import matmul_mod64
from pgw.hemask.wire import MaskgenItem, pack_y, pack_y_v2, unpack_b, unpack_bits
from pgw.service import handler as handler_mod
from pgw.service import maskgen_gpu as mg
from pgw.service.kernels import ComputeKernel

PRESETS = {  # (layers, {kind: (m, n)})
    "qwen2.5-0.5b": (24, {"qkv": (1152, 896), "o": (896, 896), "gate_up": (9728, 896), "down": (896, 4864)}),
    "qwen2.5-7b": (28, {"qkv": (4608, 3584), "o": (3584, 3584), "gate_up": (37888, 3584), "down": (3584, 18944)}),
    "llama3-70b": (80, {"qkv": (10240, 8192), "o": (8192, 8192), "gate_up": (57344, 8192), "down": (8192, 28672)}),
}


def _fmt(s: str) -> tuple[str, int, int]:
    v, a, b = s.split(":")
    return v, int(a), int(b)


def _item(lid: str, n: int, cols: int, fmt: str, rng: np.random.Generator) -> MaskgenItem:
    v, a, b = _fmt(fmt)
    if v == "v1":
        return MaskgenItem(lid, cols, rng.bytes(n * cols * a // 8), b, a)
    return MaskgenItem(lid, cols, bytearray(rng.bytes(n * cols * a // 8)), b_bits=a, y_bits=b)


def _planes(fmt: str, mode: str) -> int:
    v, a, _ = _fmt(fmt)
    return 8 if (v == "v2" and mode == "legacy") else -(-a // 8)


def _set_mode(mode: str) -> None:
    os.environ[mg.ENV_FUSED] = "0" if mode == "legacy" else "1"


def _run_rounds(kerns: list[dict[str, ComputeKernel]], items: dict[str, MaskgenItem], rounds: int, conc: int,
                decode_busy: Any = None) -> float:
    """Wall seconds for ``rounds`` rounds of every kind on every device set (pools of ``conc`` per device)."""
    pools = [ThreadPoolExecutor(conc) for _ in kerns]
    try:
        t0 = time.perf_counter()
        futs = []
        for _ in range(rounds):
            for pool, ks in zip(pools, kerns, strict=True):
                for kind, kern in ks.items():
                    args = (kern, items[kind]) if decode_busy is None else (kern, items[kind], decode_busy)
                    futs.append(pool.submit(handler_mod._maskgen_item, *args))
        for f in futs:
            f.result()
        return time.perf_counter() - t0
    finally:
        for p in pools:
            p.shutdown()


def _reference_rows(kern_w: np.ndarray, it: MaskgenItem, n: int, rows: int) -> bytes:
    v, a, b = (("v2", it.b_bits, it.y_bits) if it.v2 else ("v1", it.q_bits, it.drop_bits))
    cols = it.cols
    if v == "v1":
        bm = unpack_b(it.b, (n, cols), a)
        return pack_y(matmul_mod64(kern_w[:rows], bm, fast=False), b, a)
    bm = unpack_bits(it.b, n * cols, a).reshape(n, cols)
    return bytes(pack_y_v2(matmul_mod64(kern_w[:rows], bm, fast=False), a, b))


def bench_preset(name: str, args: argparse.Namespace, ndev: int) -> dict[str, Any]:
    layers, shapes = PRESETS[name]
    rng = np.random.default_rng(0)
    devs = [f"cuda:{i}" for i in range(ndev)]
    weights: dict[str, np.ndarray] = {}
    kerns: list[dict[str, ComputeKernel]] = []
    for d in devs:
        ks = {}
        for kind, (m, n) in shapes.items():
            if kind not in weights:
                weights[kind] = rng.integers(-128, 128, size=(m, n), dtype=np.int8)
            ks[kind] = ComputeKernel(LinearSpec(LinearId(0, kind), m, n), torch.from_numpy(weights[kind]), device=d,
                                     cuda_graphs=False, triton_gemv=False)
        kerns.append(ks)
    macs_per_col = sum(m * n for m, n in shapes.values())
    res: dict[str, Any] = {"layers": layers, "shapes": shapes, "macs_per_slot_per_plane": layers * macs_per_col,
                           "runs": []}
    groups = [g for g in args.groups if name != "llama3-70b" or g <= args.max_groups_70b]
    for fmt in args.formats:
        for g in groups:
            cols = DEGREE * g
            items = {kind: _item(f"0.{kind}", n, cols, fmt, rng) for kind, (m, n) in shapes.items()}
            outs: dict[str, dict[str, bytes]] = {}
            for mode in args.modes:
                _set_mode(mode)
                # warm-up + outputs for the exactness check
                outs[mode] = {kind: bytes(handler_mod._maskgen_item(kerns[0][kind], items[kind])) for kind in shapes}
                torch.cuda.synchronize()
                for conc in args.conc:
                    for multi in ([False, True] if ndev > 1 and args.multi_gpu else [False]):
                        if mode == "legacy" and (g > args.max_groups_legacy or multi):
                            continue
                        ks = kerns if multi else kerns[:1]
                        rounds = max(1, args.rounds)
                        wall = _run_rounds(ks, items, rounds, conc)
                        nd = len(ks)
                        s_round = wall / (rounds * nd)  # per layer-round on one device (aggregate)
                        planes = _planes(fmt, mode)
                        rec = {"fmt": fmt, "groups": g, "mode": mode, "conc": conc, "devices": nd,
                               "s_per_layer_round": round(s_round, 5),
                               "slots_per_s": round(cols / (layers * s_round), 2),
                               "planes": planes,
                               "gemm_tops": round(2 * planes * cols * macs_per_col / s_round / 1e12, 2),
                               "tops_7plane": round(2 * 7 * cols * macs_per_col / s_round / 1e12, 2)}
                        res["runs"].append(rec)
                        print(name, json.dumps(rec), flush=True)
            if len(outs) > 1:
                ref_mode = args.modes[0]
                same = {kind: all(outs[mo][kind] == outs[ref_mode][kind] for mo in outs) for kind in shapes}
                res.setdefault("identical", {})[f"{fmt}/g{g}"] = same
                print(name, fmt, g, "identical across paths:", same, flush=True)
            # CPU reference on a row slice (rows are contiguous at the start of every packed layout only for
            # v1; for v2 compare by repacking the whole reply's first rows is not a prefix, so do it for v1
            # and for v2 via a small separate item)
            if g == groups[0] and "fused" in outs:
                kind = "o"
                n = shapes[kind][1]
                rows = 16
                it = items[kind]
                if not it.v2:
                    nbo = (it.q_bits - it.drop_bits) // 8
                    ok = outs["fused"][kind][:rows * cols * nbo] == _reference_rows(weights[kind], it, n, rows)
                else:
                    sub = rng.integers(-128, 128, size=(rows, n), dtype=np.int8)
                    sk = ComputeKernel(LinearSpec(LinearId(0, kind), rows, n), torch.from_numpy(sub), device=devs[0],
                                       cuda_graphs=False, triton_gemv=False)
                    _set_mode("fused")
                    ok = bytes(handler_mod._maskgen_item(sk, it)) == _reference_rows(sub, it, n, rows)
                res.setdefault("cpu_reference_ok", {})[f"{fmt}/g{g}"] = bool(ok)
                print(name, fmt, g, "CPU reference:", ok, flush=True)
    if args.profile:
        _set_mode("fused")
        res["profile"] = {}
        for kind in ("gate_up", "down"):
            n = shapes[kind][1]
            try:
                res["profile"][kind] = profile_item(kerns[0][kind], _item(f"0.{kind}", n, DEGREE * args.profile_groups,
                                                                          "v2:53:35", rng))
            except Exception as exc:  # pragma: no cover - diagnostics only
                res["profile"][kind] = {"error": f"{type(exc).__name__}: {exc}"}
            print(name, "profile", kind, json.dumps(res["profile"][kind]), flush=True)
    if args.decode:
        res["decode"] = bench_decode(kerns[0], shapes, args, rng)
    del kerns
    torch.cuda.empty_cache()
    return res


def _sync_time(fn: Any, reps: int = 3) -> float:
    fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / reps


def profile_item(kern: ComputeKernel, it: MaskgenItem) -> dict[str, Any]:
    """Stage times (ms, synchronised) of one fused v2 item, and the bare int8 GEMM ceiling for its shape."""
    dev = kern.device
    wpad = kern._pw._w_dev
    m, n, cols = kern.out_features, kern.in_features, it.cols
    mp, np_ = wpad.shape
    nb = -(-it.b_bits // 8)
    const = kern._maskgen_const(nb, 0)
    out: dict[str, Any] = {"m": m, "n": n, "cols": cols}
    with torch.cuda.device(dev):
        host = torch.empty(len(it.b), dtype=torch.uint8, pin_memory=True)
        src = torch.frombuffer(it.b, dtype=torch.uint8)
        out["host_copy_ms"] = 1e3 * _sync_time(lambda: host.copy_(src))
        dd = host.to(dev)
        out["h2d_ms"] = 1e3 * _sync_time(lambda: host.to(dev, non_blocking=True))
        step = mg._block_cols(mp, nb, cols, mg.PART_BYTES)
        cb = min(step, cols)
        cbp = -(-cb // 32) * 32
        srcs = mg.v2_planes_source(dd, n, cols, it.b_bits)
        planes = torch.empty((nb, cbp, np_), dtype=torch.uint8, device=dev)

        def mk() -> None:
            for d, (sv, j) in enumerate(mg.v2_planes_source(dd, n, cols, it.b_bits)):
                torch.bitwise_xor(sv[:, :cb, j].t(), 0x80, out=planes[d, :cb, :n])

        nblk = -(-cols // step)
        out["unpack_planes_ms"] = 1e3 * nblk * _sync_time(mk)
        del srcs
        rhs = planes.view(torch.int8).view(nb * cbp, np_).t()
        t_gemm = _sync_time(lambda: torch._int_mm(wpad, rhs))
        out["gemm_ms"] = 1e3 * nblk * t_gemm
        out["gemm_tops_block"] = round(2 * mp * np_ * nb * cbp / t_gemm / 1e12, 2)
        part = torch._int_mm(wpad, rhs).view(mp, nb, cbp)
        out["combine_fused_ms"] = 1e3 * nblk * _sync_time(lambda: mg.combine(part, m, cb, const, fused=True))
        out["combine_torch_ms"] = 1e3 * nblk * _sync_time(lambda: mg.combine(part, m, cb, const, fused=False))
        y = mg.combine(part, m, cb, const)
        pk = torch.empty(m * cols * it.y_bits // 8, dtype=torch.uint8, device=dev)
        out["pack_ms"] = 1e3 * nblk * _sync_time(lambda: mg.pack_v2_block(y, it.b_bits, it.y_bits, pk, cols, 0))
        res = torch.empty(pk.numel(), dtype=torch.uint8, pin_memory=True)
        out["d2h_ms"] = 1e3 * _sync_time(lambda: res.copy_(pk))
        out["total_item_ms"] = 1e3 * _sync_time(lambda: handler_mod._maskgen_item(kern, it))
        # the ceiling: one large square-ish int8 GEMM on this device
        a = torch.randint(-128, 127, (8192, 8192), dtype=torch.int8, device=dev)
        bt = torch.randint(-128, 127, (8192, 8192), dtype=torch.int8, device=dev).t()
        tc = _sync_time(lambda: torch._int_mm(a, bt))
        out["gemm_tops_8k_ceiling"] = round(2 * 8192**3 / tc / 1e12, 2)
    return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out.items()}


def bench_decode(ks: dict[str, ComputeKernel], shapes: dict[str, tuple[int, int]], args: argparse.Namespace,
                 rng: np.random.Generator) -> dict[str, Any]:
    """Latency of a decode-sized GEMM (qkv, 8 rows) alone, next to saturating maskgen, and with the yield."""
    ring = RingParams.from_name("K32A12C27")
    kern = ks["qkv"]
    x = torch.from_numpy(rng.integers(0, 1 << 32, size=(8, kern.in_features), dtype=np.int64))
    g = args.decode_groups
    items = {kind: _item(f"0.{kind}", n, DEGREE * g, "v2:53:35", rng) for kind, (m, n) in shapes.items()}
    out: dict[str, Any] = {}
    for label, yield_ms in (("alone", None), ("maskgen", 0.0), ("maskgen_yield", args.yield_ms)):
        _set_mode("fused")
        busy = threading.Event()
        stop = threading.Event()
        os.environ[mg.ENV_YIELD_MS] = str(yield_ms or 0)

        def hammer(stop: threading.Event = stop, busy: threading.Event = busy) -> None:
            while not stop.is_set():
                _run_rounds([ks], items, 1, 2, decode_busy=busy.is_set)

        th = threading.Thread(target=hammer, daemon=True) if yield_ms is not None else None
        for _ in range(5):
            kern.run(x, ring)
        if th:
            th.start()
            time.sleep(1.0)
        lat = []
        t_end = time.perf_counter() + args.decode_seconds
        while time.perf_counter() < t_end:
            busy.set()
            t = time.perf_counter()
            kern.run(x, ring)
            lat.append(time.perf_counter() - t)
            busy.clear()
            time.sleep(0.002)  # a decode step's gateway work between GEMMs
        stop.set()
        if th:
            th.join()
        a = np.array(lat) * 1e3
        out[label] = {"n": len(lat), "p50_ms": round(float(np.percentile(a, 50)), 3),
                      "p90_ms": round(float(np.percentile(a, 90)), 3), "p99_ms": round(float(np.percentile(a, 99)), 3)}
        print("decode", label, json.dumps(out[label]), flush=True)
    os.environ[mg.ENV_YIELD_MS] = "0"
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--presets", nargs="+", default=["qwen2.5-0.5b", "qwen2.5-7b"], choices=sorted(PRESETS))
    ap.add_argument("--formats", nargs="+", default=["v2:53:35", "v1:56:16"],
                    help="v2:bb:yb (bit-packed, default upload 53/35 for K32) or v1:q:drop")
    ap.add_argument("--groups", nargs="+", type=int, default=[1, 4, 16])
    ap.add_argument("--conc", nargs="+", type=int, default=[1, 2, 4])
    ap.add_argument("--modes", nargs="+", default=["legacy", "fused"], choices=["legacy", "fused"])
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--max-groups-legacy", type=int, default=4)
    ap.add_argument("--max-groups-70b", type=int, default=4)
    ap.add_argument("--multi-gpu", action="store_true", default=True)
    ap.add_argument("--no-multi-gpu", dest="multi_gpu", action="store_false")
    ap.add_argument("--decode", action="store_true")
    ap.add_argument("--profile", action="store_true", help="stage breakdown of one gate_up / down item")
    ap.add_argument("--profile-groups", type=int, default=4)
    ap.add_argument("--decode-groups", type=int, default=4)
    ap.add_argument("--decode-seconds", type=float, default=6.0)
    ap.add_argument("--yield-ms", type=float, default=5.0)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    ndev = torch.cuda.device_count()
    if ndev == 0:
        raise SystemExit("needs CUDA")
    res: dict[str, Any] = {"gpus": [torch.cuda.get_device_name(i) for i in range(ndev)], "torch": torch.__version__,
                           "triton_combine": mg.triton_combine_ok(torch.device("cuda", 0)), "cpus": os.cpu_count(),
                           "presets": {}}
    print(json.dumps({k: v for k, v in res.items() if k != "presets"}), flush=True)
    for name in args.presets:
        try:
            res["presets"][name] = bench_preset(name, args, ndev)
        except Exception as exc:  # keep the other presets
            import traceback

            traceback.print_exc()
            res["presets"][name] = {"error": f"{type(exc).__name__}: {exc}"}
        if args.json:
            with open(args.json, "w") as fh:
                json.dump(res, fh, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
