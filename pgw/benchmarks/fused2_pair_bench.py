"""Micro-benchmark of the fused2 provider pair op (:class:`pgw.core.ring64.PairOp`) at Qwen2.5-7B shapes.

One GPU, random public int8 weights (the op's cost does not depend on their values), K56. For each 7B trip item
(``o+gate_up/f2``, ``down+qkv/f2``, the single ``qkv/f2``) and row count it:

1. checks the op byte for byte against the prototype provider (eager digit GEMMs, :class:`pgw.engine.fused2_remote.
   Fused2Service`'s arithmetic) on random masked inputs (the small-shape kernel self-check against the Python-int
   reference runs first);
2. times the CUDA graph the lane replays (``[H2D pinned, unpack, gemv, epilogue, gemv, epilogue, D2H pinned]``,
   launch + synchronize per call, median of ``--reps``) for the heuristic launch configuration and a few others;
3. times today's graphed single-linear K32 trip (``ring_gemv`` graph with its copies) for the same linears, so a
   pair trip can be compared with the two single trips it replaces, and the prototype's eager pair (0.97 ms
   measured on H100 in the 7B e2e run).

Usage (H100, <= ~2 min of GPU time; see deploy/modal/gpu_bench.py)::

    modal run deploy/modal/gpu_bench.py --target h100_c4 --timeout-min 4 \\
        --cmd "python benchmarks/fused2_pair_bench.py --json /tmp/f2pair.json"

Local CPU (Triton interpreter, tiny shapes, correctness of the script only)::

    TRITON_INTERPRET=1 python benchmarks/fused2_pair_bench.py --tiny
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from typing import Any

import numpy as np
import torch

SHAPES_7B = {"o": (3584, 3584), "gate_up": (37888, 3584), "down": (3584, 18944), "qkv": (4608, 3584)}
#: Other Qwen2.5 shapes (``--model``); ``head`` = the LM head (152064 x hidden) for the folded-head pair item.
SHAPES = {
    "7b": {**SHAPES_7B, "head": (152064, 3584)},
    "32b": {"o": (5120, 5120), "gate_up": (55296, 5120), "down": (5120, 27648), "qkv": (7168, 5120),
            "head": (152064, 5120)},
    "72b": {"o": (8192, 8192), "gate_up": (59136, 8192), "down": (8192, 29568), "qkv": (10240, 8192),
            "head": (152064, 8192)},
}
LAYERS = {"7b": 28, "32b": 64, "72b": 80}
SHAPES_TINY = {"o": (64, 64), "gate_up": (160, 64), "down": (64, 96), "qkv": (96, 64)}


def _w(gen: torch.Generator, m: int, n: int, dev: torch.device) -> torch.Tensor:
    return torch.randint(-128, 128, (m, n), generator=gen, dtype=torch.int8).to(dev)


def _prototype_pair(k: int, w1: Any, w2: Any, hm: torch.Tensor | None, x: torch.Tensor, r: torch.Tensor | None
                    ) -> bytes:
    """The prototype provider's arithmetic (eager digit GEMMs), packed as the lane reply."""
    from pgw.core.packing import pack_ring

    mask = (1 << k) - 1
    y = w1.matmul(x, k)
    if w2 is None:
        return pack_ring(y.cpu(), k)
    if hm is not None:
        y = (y * hm) & mask
    h = (r + y) & mask
    z = w2.matmul(h, k)
    return pack_ring(h.cpu(), k) + pack_ring(z.cpu(), k)


def _time(fn: Any, reps: int, sync: Any) -> float:
    fn()
    sync()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        sync()
        ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--json", default="")
    ap.add_argument("--rows", default="1,2,4,8")
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--k", type=int, default=56)
    ap.add_argument("--tiny", action="store_true", help="tiny shapes (CPU / interpreter smoke run)")
    ap.add_argument("--no-sweep", action="store_true", help="heuristic configurations only")
    ap.add_argument("--variants", default="planes,sum",
                    help="variants to time (the first is the headline; 'sum' is the 2026-11-23 baseline)")
    ap.add_argument("--budget-s", type=float, default=0.0,
                    help="stop sweeping non-heuristic configurations after this many seconds (0: no limit)")
    ap.add_argument("--model", default="7b", choices=sorted(SHAPES), help="Qwen2.5 shapes (GPU only)")
    ap.add_argument("--extra-items", action="store_true",
                    help="also the folded-head pair (last down + LM head) and the parallel-block pair "
                         "([o | down] then [qkv ; gate_up]): research/reviews/consumer-round-trips.md")
    ap.add_argument("--prototype-timing", action="store_true", help="also time the prototype's eager pair")
    args = ap.parse_args(argv)

    from pgw.core.packing import pack_ring
    from pgw.core.ring64 import PairConfig, PairOp, _per_split, _rand_ring, pair_config, self_check
    from pgw.core.ringmm import PreparedWeight

    cuda = torch.cuda.is_available()
    dev = torch.device("cuda:0" if cuda else "cpu")
    shapes = dict(SHAPES_TINY if args.tiny or not cuda else SHAPES[args.model])
    if args.tiny or not cuda:
        shapes["head"] = (256, 64)
    nl = LAYERS.get(args.model, 28)
    rows_list = [int(r) for r in args.rows.split(",") if r]
    if not cuda:
        rows_list = [r for r in rows_list if r <= 2]
        args.reps = min(args.reps, 2)
    k = args.k
    out: dict[str, Any] = {"device": torch.cuda.get_device_name(0) if cuda else "cpu", "k": k, "shapes": shapes,
                           "torch": torch.__version__}
    t0 = time.perf_counter()
    variants = [v for v in args.variants.split(",") if v]
    fails = self_check(dev, k=k, rows=(1, 2, 8) if cuda else (1, 2), variants=tuple(variants))
    out["self_check"] = {"failures": fails, "s": round(time.perf_counter() - t0, 1)}
    print(f"self-check: {'ok' if not fails else fails[:3]} ({out['self_check']['s']} s)", flush=True)
    if fails:
        print(json.dumps(out, indent=1))
        return 1
    gen = torch.Generator().manual_seed(1)
    W = {name: _w(gen, m, n, dev) for name, (m, n) in shapes.items()}
    hm_o = torch.randint(1, 256, (shapes["o"][0],), generator=gen, dtype=torch.int64).to(dev)
    hm_d = torch.randint(1, 256, (shapes["down"][0],), generator=gen, dtype=torch.int64).to(dev)
    items = {"o+gate_up": (W["o"], W["gate_up"], hm_o), "down+qkv": (W["down"], W["qkv"], hm_d),
             "qkv(single)": (W["qkv"], None, None)}
    if args.extra_items:
        items["down+head"] = (W["down"], W["head"], hm_d)
        par1 = torch.cat([W["o"], W["down"]], dim=1).contiguous()  # [W_o | W_down]: (d, d + I)
        par2 = torch.cat([W["qkv"], W["gate_up"]], dim=0).contiguous()  # [W_qkv ; W_gate_up]: (qkv + 2I, d)
        items["parallel o|down+qkv;gate_up"] = (par1, par2, hm_o)
    W = {k_: v_ for k_, v_ in W.items() if k_ != "head"}  # the K32 single baselines: the four layer linears
    rmax = max(rows_list)
    sync = torch.cuda.synchronize if cuda else (lambda: None)
    rng = np.random.default_rng(3)
    res: dict[str, Any] = {}
    stream = torch.cuda.Stream(dev) if cuda else None
    pool = torch.cuda.graph_pool_handle() if cuda else None
    for name, (w1, w2, hm) in items.items():
        op = PairOp(k, w1, w2, hm, device=dev, max_rows=rmax)
        pw1 = PreparedWeight(w1, device=dev, cache_float=False)
        pw2 = None if w2 is None else PreparedWeight(w2, device=dev, cache_float=False)
        h_in = torch.zeros(op.in_bytes(rmax), dtype=torch.uint8, pin_memory=cuda)
        h_out = torch.zeros(op.out_bytes(rmax), dtype=torch.uint8, pin_memory=cuda)
        res[name] = {}
        for rows in rows_list:
            x = _rand_ring(rng, (rows, op.n1), k)
            r = _rand_ring(rng, (rows, op.m1), k) if op.pair else None
            body = pack_ring(x, k) + (pack_ring(r, k) if r is not None else b"")
            want = _prototype_pair(k, pw1, pw2, hm, x.to(dev), None if r is None else r.to(dev))
            ent: dict[str, Any] = {"exact": op.run(rows, body) == want}
            if args.prototype_timing:
                ent["prototype_eager_ms"] = round(_time(lambda: _prototype_pair(k, pw1, pw2, hm, x.to(dev),
                                                                                None if r is None else r.to(dev)),
                                                        max(3, args.reps // 5), sync), 4)
            # heuristic of every requested variant first, then (budget permitting) a small sweep per variant
            cands: list[tuple[PairConfig, PairConfig | None]] = []
            for v in variants:
                c1 = pair_config(rows, op.m1, op.n1, dev, v)
                c2 = pair_config(rows, op.m2, op.m1, dev, v) if op.pair else None
                cands.append((c1, c2))
            n_heur = len(cands)
            if cuda and not args.no_sweep:
                for v in variants:
                    if v == "planes":
                        tiles = [(64, 256, 4, 2.0), (64, 128, 4, 1.0), (128, 256, 8, 1.0), (64, 512, 4, 1.0),
                                 (64, 256, 4, 0.5), (128, 128, 4, 2.0), (64, 256, 8, 1.0)]
                    else:
                        tiles = [(16, 256, 4, 1.0), (8, 512, 4, 1.0), (32, 128, 4, 1.0)]
                    for bm, bn, warps, f in tiles:
                        def mk(m_: int, n_: int, bm: int = bm, bn: int = bn, warps: int = warps, f: float = f,
                               v: str = v) -> PairConfig:
                            base = pair_config(rows, m_, n_, dev, v)
                            return PairConfig(bm, bn, max(1, round(base.split * f)), warps, v)
                        c = (mk(op.m1, op.n1), mk(op.m2, op.m1) if op.pair else None)
                        if c not in cands:
                            cands.append(c)
            timings = []
            for ci, (c1, c2) in enumerate(cands):
                if ci >= n_heur and args.budget_s and time.perf_counter() - t0 > args.budget_s:
                    break
                try:
                    op.set_config(rows, c1, c2)
                    if op.run(rows, body) != want:
                        timings.append({"cfg": [str(c1), str(c2)], "exact": False})
                        continue
                    if cuda:
                        h_in[: len(body)].copy_(torch.frombuffer(bytearray(body), dtype=torch.uint8))
                        g = op.capture(rows, h_in, h_out, stream=stream, pool=pool)
                        ms = _time(g.replay, args.reps, sync)
                        ok = bytes(h_out[: op.out_bytes(rows)].numpy().tobytes()) == want
                    else:
                        ms = _time(lambda b=body: op.run(rows, b), args.reps, sync)
                        ok = True
                    splits = [_per_split(op.n1, c1)[1], None if c2 is None else _per_split(op.m1, c2)[1]]
                    timings.append({"cfg": [str(c1), str(c2)], "splits": splits, "ms": round(ms, 4), "exact": ok})
                except Exception as exc:  # a configuration that does not compile on this device
                    timings.append({"cfg": [str(c1), str(c2)], "error": f"{type(exc).__name__}: {str(exc)[:160]}"})
            good = [t for t in timings if t.get("exact") and "ms" in t]
            ent["graph_ms_best"] = min((t["ms"] for t in good), default=None)
            ent["graph_ms_heuristic"] = {v: timings[i].get("ms") for i, v in enumerate(variants) if i < len(timings)}
            for v in variants:
                vt = [t["ms"] for t in good if f"variant='{v}'" in t["cfg"][0]]
                ent[f"graph_ms_best_{v}"] = min(vt) if vt else None
            ent["best_cfg"] = min(good, key=lambda t: t["ms"])["cfg"] if good else None
            ent["configs"] = timings
            ent["exact"] = ent["exact"] and all(t.get("exact", True) for t in timings)
            res[name][str(rows)] = ent
            print(f"{name} rows={rows}: exact={ent['exact']} graph best {ent['graph_ms_best']} ms "
                  f"(heuristic {ent['graph_ms_heuristic']}, {len(timings)} configs)", flush=True)
        del op, pw1, pw2
    out["pair"] = res
    if cuda:  # today's graphed single-linear K32 trip, same copies (the lane's [H2D, GEMV, D2H] graph)
        from pgw.core.triton_gemv import ring_gemv
        from pgw.core.triton_gemv import self_check as gemv_check

        base: dict[str, Any] = {}
        if gemv_check(dev):
            for name, w in W.items():
                m, n = w.shape
                base[name] = {}
                for rows in sorted({1, max(rows_list)}):
                    hin = torch.zeros((rows, n), dtype=torch.int32, pin_memory=True)
                    hout = torch.zeros((rows, m), dtype=torch.int32, pin_memory=True)
                    xin = torch.zeros((rows, n), dtype=torch.int32, device=dev)
                    yout = torch.zeros((rows, m), dtype=torch.int32, device=dev)

                    def body(xin: Any = xin, hin: Any = hin, yout: Any = yout, hout: Any = hout, w: Any = w) -> None:
                        xin.copy_(hin, non_blocking=True)
                        ring_gemv(xin, w, 32, out=yout)
                        hout.copy_(yout, non_blocking=True)

                    stream.wait_stream(torch.cuda.current_stream(dev))
                    with torch.cuda.stream(stream):
                        body()
                    stream.synchronize()
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g, pool=pool, stream=stream):
                        body()
                    base[name][str(rows)] = round(_time(g.replay, args.reps, sync), 4)
        out["k32_single_graph_ms"] = base
        summ: dict[str, Any] = {}
        for rk in sorted({1, max(rows_list)}):
            if not base or str(rk) not in res["o+gate_up"]:
                continue
            one = {n: v[str(rk)] for n, v in base.items()}
            for v in [*variants, "best"]:
                key = "graph_ms_best" if v == "best" else f"graph_ms_best_{v}"
                a, b = res["o+gate_up"][str(rk)].get(key), res["down+qkv"][str(rk)].get(key)
                if a is None or b is None:
                    continue
                summ[f"{rk} rows, {v}"] = {
                    "o+gate_up pair / (o + gate_up singles)": round(a / (one["o"] + one["gate_up"]), 3),
                    "down+qkv pair / (down + qkv singles)": round(b / (one["down"] + one["qkv"]), 3),
                    f"per-token provider compute fused2 ({nl} layers, ms)": round(nl * (a + b), 3),
                    f"per-token provider compute standard K32 ({nl} layers, ms)": round(nl * sum(one.values()), 3),
                }
        if args.extra_items:
            for rk in sorted({1, max(rows_list)}):
                par = res.get("parallel o|down+qkv;gate_up", {}).get(str(rk), {}).get("graph_ms_best")
                hd = res.get("down+head", {}).get(str(rk), {}).get("graph_ms_best")
                if par is not None:
                    summ[f"{rk} rows, parallel"] = {
                        f"per-token provider compute parallel fused ({nl} layers, ms)": round(nl * par, 3),
                        "down+head pair (ms)": hd}
        out["summary"] = summ
    out["wall_s"] = round(time.perf_counter() - t0, 1)
    print(json.dumps({k2: v for k2, v in out.items() if k2 != "pair"}, indent=1))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1)
    return 0 if all(e["exact"] for r in res.values() for e in r.values()) else 1


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
    sys.exit(main())
