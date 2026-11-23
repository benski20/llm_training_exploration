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
    args = ap.parse_args(argv)

    from pgw.core.packing import pack_ring
    from pgw.core.ring64 import PairConfig, PairOp, _rand_ring, pair_config, self_check
    from pgw.core.ringmm import PreparedWeight

    cuda = torch.cuda.is_available()
    dev = torch.device("cuda:0" if cuda else "cpu")
    shapes = SHAPES_TINY if args.tiny or not cuda else SHAPES_7B
    rows_list = [int(r) for r in args.rows.split(",") if r]
    if not cuda:
        rows_list = [r for r in rows_list if r <= 2]
        args.reps = min(args.reps, 2)
    k = args.k
    out: dict[str, Any] = {"device": torch.cuda.get_device_name(0) if cuda else "cpu", "k": k, "shapes": shapes,
                           "torch": torch.__version__}
    t0 = time.perf_counter()
    fails = self_check(dev, k=k, rows=(1, 2, 8) if cuda else (1, 2))
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
            ent["prototype_eager_ms"] = round(_time(lambda: _prototype_pair(k, pw1, pw2, hm, x.to(dev),
                                                                            None if r is None else r.to(dev)),
                                                    max(3, args.reps // 5), sync), 4)
            cands: list[tuple[PairConfig, PairConfig | None]] = [op.config(rows)]
            if cuda and not args.no_sweep:
                variants = ["sum", "dot"] if rows <= 2 else ["dot", "sum"]
                for v in variants:
                    for bm, bn in ((16, 256), (32, 128), (64, 128)) if v == "dot" else ((16, 256), (8, 512), (32, 128)):
                        c1 = pair_config(rows, op.m1, op.n1, dev, v)
                        c1 = PairConfig(bm, bn, c1.split, c1.num_warps, v)
                        c2 = None
                        if op.pair:
                            c2 = pair_config(rows, op.m2, op.m1, dev, v)
                            c2 = PairConfig(bm, bn, c2.split, c2.num_warps, v)
                        if (c1, c2) not in cands:
                            cands.append((c1, c2))
            timings = []
            for c1, c2 in cands:
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
                    timings.append({"cfg": [str(c1), str(c2)], "ms": round(ms, 4), "exact": ok})
                except Exception as exc:  # a configuration that does not compile on this device
                    timings.append({"cfg": [str(c1), str(c2)], "error": f"{type(exc).__name__}: {str(exc)[:160]}"})
            good = [t for t in timings if t.get("exact") and "ms" in t]
            ent["graph_ms_best"] = min((t["ms"] for t in good), default=None)
            ent["graph_ms_heuristic"] = timings[0].get("ms")
            ent["best_cfg"] = min(good, key=lambda t: t["ms"])["cfg"] if good else None
            ent["configs"] = timings
            ent["exact"] = ent["exact"] and all(t.get("exact", True) for t in timings)
            res[name][str(rows)] = ent
            print(f"{name} rows={rows}: exact={ent['exact']} graph best {ent['graph_ms_best']} ms "
                  f"(heuristic {ent['graph_ms_heuristic']}), prototype eager {ent['prototype_eager_ms']} ms", flush=True)
        del op, pw1, pw2
    out["pair"] = res
    if cuda:  # today's graphed single-linear K32 trip, same copies (the lane's [H2D, GEMV, D2H] graph)
        from pgw.core.triton_gemv import ring_gemv, self_check as gemv_check

        base: dict[str, Any] = {}
        if gemv_check(dev):
            for name, w in W.items():
                m, n = w.shape
                base[name] = {}
                for rows in (1, 8) if 8 in rows_list else (1,):
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
        if base:
            one = {n: v["1"] for n, v in base.items()}
            out["summary_1row"] = {
                "o+gate_up pair / (o + gate_up singles)": (res["o+gate_up"]["1"]["graph_ms_best"] or 0)
                / (one["o"] + one["gate_up"]),
                "down+qkv pair / (down + qkv singles)": (res["down+qkv"]["1"]["graph_ms_best"] or 0)
                / (one["down"] + one["qkv"]),
                "per-token provider compute fused2 (28 layers, ms)": 28 * ((res["o+gate_up"]["1"]["graph_ms_best"] or 0)
                                                                         + (res["down+qkv"]["1"]["graph_ms_best"] or 0)),
                "per-token provider compute standard K32 (28 layers, ms)": 28 * sum(one.values()),
            }
    out["wall_s"] = round(time.perf_counter() - t0, 1)
    print(json.dumps({k2: v for k2, v in out.items() if k2 != "pair"}, indent=1))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1)
    return 0 if all(e["exact"] for r in res.values() for e in r.values()) else 1


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
    sys.exit(main())
