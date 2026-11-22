"""Which torch thread counts give the gateway's decode attention the same bits, and how fast each is.

The gateway's float ops must match the plaintext reference bit for bit, and torch's CPU SDPA can depend on
the thread count. For the decode shape of a model (one query row, ``--heads`` query heads over ``--kv-heads``
KV heads, ``--head-dim``) and cache lengths up to ``--max-len``, this compares SDPA (the gateway's
``_attend``) at every thread count in ``--threads`` with the result at the default count, and times each.

    PYTHONPATH=src python benchmarks/sdpa_threads_bench.py --heads 14 --kv-heads 2 --head-dim 64
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", type=int, default=14)
    ap.add_argument("--kv-heads", type=int, default=2)
    ap.add_argument("--head-dim", type=int, default=64)
    ap.add_argument("--lengths", default="1,17,41,64,100,130,257,513,700")
    ap.add_argument("--threads", default="")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--reps", type=int, default=200)
    args = ap.parse_args()
    from pgw.engine.model import _attend

    default = torch.get_num_threads()
    counts = [int(v) for v in args.threads.split(",") if v] or sorted({1, 2, 4, 7, 8, default // 2, default, 16})
    dtype = getattr(torch, args.dtype)
    gen = torch.Generator().manual_seed(7)
    groups = args.heads // args.kv_heads
    scale = args.head_dim ** -0.5
    cases = []
    for length in (int(v) for v in args.lengths.split(",")):
        q = torch.randn(1, args.heads, args.head_dim, generator=gen).to(dtype)
        k = torch.randn(length, args.kv_heads, args.head_dim, generator=gen).to(dtype)
        v = torch.randn(length, args.kv_heads, args.head_dim, generator=gen).to(dtype)
        cases.append((length, q, k, v))
    res: dict = {"default_threads": default, "dtype": args.dtype, "per_count": {}}
    for gqa in (False, True):
        torch.set_num_threads(default)
        ref = [_attend(q, k, v, length - 1, scale, groups, gqa) for length, q, k, v in cases]
        for n in counts:
            torch.set_num_threads(n)
            same = all(torch.equal(_attend(q, k, v, length - 1, scale, groups, gqa), r)
                       for (length, q, k, v), r in zip(cases, ref, strict=True))
            length, q, k, v = cases[min(4, len(cases) - 1)]
            ts = []
            for _ in range(args.reps):
                t0 = time.perf_counter()
                _attend(q, k, v, length - 1, scale, groups, gqa)
                ts.append(time.perf_counter() - t0)
            res["per_count"][f"gqa{int(gqa)}_t{n}"] = {"identical_to_default": same,
                                                        "us_per_call": round(1e6 * statistics.median(ts), 1)}
    torch.set_num_threads(default)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
