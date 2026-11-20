"""Cost of the float32 LM head with N rows (concurrent sequences / speculative verify rows).

The engine computes ``rms_norm(h) @ lm_head^T`` (``vocab x hidden`` float32: 2.2 GB for
Qwen2.5-7B, memory bound). Compares, for N rows:

* ``torch_per_row_s``: torch, one ``(1, hidden) @ W^T`` per row (the engine before the deterministic
  kernel);
* ``torch_batched_s``: torch, one ``(N, hidden) @ W^T`` (fast but its bits depend on N);
* ``det_per_row_s`` / ``det_batched_s``: ``pgw.engine.lm_head.lm_head_logits`` (native
  ``pgw_fast.lm_head`` when built, else the torch fallback), per row and all N rows in one pass.
  Its rows are bit-identical either way (checked here too).

Usage:
    python benchmarks/lm_head_bench.py --hidden 3584 --vocab 152064 --n 1 4 8
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable

import torch

from pgw.engine.lm_head import lm_head_logits, native_lm_head


def best(f: Callable[[], object], reps: int) -> float:
    f()
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        f()
        ts.append(time.perf_counter() - t)
    return min(ts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=3584)
    ap.add_argument("--vocab", type=int, default=152064)
    ap.add_argument("--n", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()
    torch.manual_seed(0)
    w = torch.randn(args.vocab, args.hidden, dtype=torch.float32) * 0.02
    print(json.dumps({"native": native_lm_head(), "torch_threads": torch.get_num_threads(),
                      "weight_gb": w.numel() * 4 / 1e9}))
    for n in args.n:
        h = torch.randn(n, args.hidden, dtype=torch.float32)
        rows = [h[i : i + 1] for i in range(n)]
        a = [lm_head_logits(r, w)[0] for r in rows]
        b = lm_head_logits(h, w)
        t = h @ w.t()
        print(json.dumps({
            "n": n,
            "torch_per_row_s": best(lambda rows=rows: [r @ w.t() for r in rows], args.reps),
            "torch_batched_s": best(lambda h=h: h @ w.t(), args.reps),
            "det_per_row_s": best(lambda rows=rows: [lm_head_logits(r, w) for r in rows], args.reps),
            "det_batched_s": best(lambda h=h: lm_head_logits(h, w), args.reps),
            "det_rows_bit_identical": all(torch.equal(a[i], b[i]) for i in range(n)),
            "det_vs_torch_max_abs_diff": float((b - t).abs().max()),
            "argmax_identical": all(int(b[i].argmax()) == int(t[i].argmax()) for i in range(n)),
        }))


if __name__ == "__main__":
    main()
