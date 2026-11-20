"""Cost of the float32 LM head with N concurrent sequences: per sequence (default) vs batched (opt-in).

The engine computes ``rms_norm(h) @ lm_head^T`` once per sequence and step, so with N concurrent
decoding sequences the ``vocab x hidden`` float32 head (2.2 GB for Qwen2.5-7B) is read N times per
step. ``TransformerModel.batch_lm_head`` (gateway model setting ``batch_lm_head``) does one
``(N, hidden) @ (hidden, vocab)`` product instead; its float32 summation order differs, so this
script also reports how far the logits move (last bits only) and whether the argmax changes.

Usage:
    python benchmarks/lm_head_bench.py --hidden 3584 --vocab 152064 --n 1 2 4 8
"""

from __future__ import annotations

import argparse
import json
import time

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=3584)
    ap.add_argument("--vocab", type=int, default=152064)
    ap.add_argument("--n", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()
    torch.manual_seed(0)
    w = torch.randn(args.vocab, args.hidden, dtype=torch.float32) * 0.02
    rows = []
    for n in args.n:
        h = torch.randn(n, args.hidden, dtype=torch.float32)

        def per_seq(h: torch.Tensor = h) -> list[torch.Tensor]:
            return [(h[i : i + 1] @ w.t())[0] for i in range(h.shape[0])]

        def batched(h: torch.Tensor = h) -> torch.Tensor:
            return h @ w.t()

        def best(f: object) -> float:
            f()  # type: ignore[operator]
            ts = []
            for _ in range(args.reps):
                t = time.perf_counter()
                f()  # type: ignore[operator]
                ts.append(time.perf_counter() - t)
            return min(ts)

        a, b = per_seq(), batched()
        diff = max(float((a[i] - b[i]).abs().max()) for i in range(n))
        same_bits = all(torch.equal(a[i], b[i]) for i in range(n))
        argmax_same = all(int(a[i].argmax()) == int(b[i].argmax()) for i in range(n))
        rows.append({
            "n": n,
            "per_sequence_s": best(per_seq),
            "batched_s": best(batched),
            "bit_identical": same_bits,
            "max_abs_diff": diff,
            "argmax_identical": argmax_same,
        })
        print(json.dumps(rows[-1]))


if __name__ == "__main__":
    main()
