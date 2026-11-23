#!/usr/bin/env python3
"""Per-step and per-pass CPU time of the target-feature draft head (``FeatureHeadDrafter``) on the gateway.

Loads a trained head file (``benchmarks/feature_head_train.py``) or makes a random one of the given shape
(``--hidden/--d/--vocab``; Qwen2.5-7B: H = 3584, vocab 152,064), with a random target embedding table of the
target's shape (timing only: the embedding is read one row per draft step). Times ``propose`` for one greedy
sequence with a chain of ``R - 1`` drafts (``R - 1`` head steps: the work of one strict-speculation pass) for
each kernel (float32, native int8), single-threaded and on the native pool, and reports the median.

    PYTHONPATH=src python benchmarks/feature_head_cpu_bench.py --head head_d384_v8192.pt --json out.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pgw.core import DEFAULT_RING
from pgw.engine.generate import DraftJob, FeatureHeadDrafter


def random_head(path: Path, hidden: int, d: int, vocab: int, target_vocab: int) -> None:
    from pgw.core.ring import quantize_weight_rows

    g = torch.Generator().manual_seed(0)
    w = {k: torch.randn(*s, generator=g) * 0.02 for k, s in
         (("w_in", (d, 2 * hidden)), ("w1", (2 * d, d)), ("w2", (d, 2 * d)), ("w_out", (hidden, d)), ("u", (vocab, d)))}
    torch.save({"format": FeatureHeadDrafter.FILE_FORMAT, "target": "random", "hidden": hidden, "d": d,
                "vocab": torch.randperm(target_vocab, generator=g)[:vocab].tolist(),
                "int8": {k: dict(zip(("w8", "scale"), quantize_weight_rows(v), strict=True)) for k, v in w.items()},
                "bias": {"b_in": torch.zeros(d), "b1": torch.zeros(2 * d), "b2": torch.zeros(d)}}, path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", help="trained head file (default: a random head of --hidden/--d/--vocab)")
    ap.add_argument("--hidden", type=int, default=3584)
    ap.add_argument("--d", type=int, default=384)
    ap.add_argument("--vocab", type=int, default=8192)
    ap.add_argument("--target-vocab", type=int, default=152064)
    ap.add_argument("--rs", default="2,3,4")
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--json")
    args = ap.parse_args()
    tmp = tempfile.TemporaryDirectory()
    path = Path(args.head) if args.head else Path(tmp.name) / "random_head.pt"
    if not args.head:
        random_head(path, args.hidden, args.d, args.vocab, args.target_vocab)
    blob = torch.load(path, weights_only=True)
    hidden = int(blob["hidden"])
    emb = torch.randn(args.target_vocab, hidden) * 0.02
    res: dict = {"head": str(path), "hidden": hidden, "d": int(blob["d"]), "vocab": len(blob["vocab"]),
                 "cpu": os.cpu_count(), "torch_threads": torch.get_num_threads(), "runs": {}}
    g = torch.Generator().manual_seed(1)
    feat = torch.randn(hidden, generator=g) * 3
    for kernel, threads in (("float", None), ("native_int8", 1), ("native_int8", 0)):
        try:
            head = FeatureHeadDrafter.from_file(path, emb, target_vocab=args.target_vocab, kernel=kernel,
                                                ring=DEFAULT_RING, threads=1 if threads is None else threads)
        except Exception as exc:
            print(f"{kernel}: unavailable ({exc})")
            continue
        label = f"{kernel}" + ("" if threads is None else f" threads={'pool' if threads == 0 else threads}")
        res["bytes_per_step_int8"] = head.nbytes_per_step()
        for r in [int(x) for x in args.rs.split(",")]:
            ts = []
            for i in range(args.reps + 10):
                job = DraftJob(state=None, context=[11, 22, 33 + i % 7], k=r - 1, feature=feat)
                t0 = time.perf_counter()
                head.propose([job])
                if i >= 10:
                    ts.append(time.perf_counter() - t0)
            med = statistics.median(ts) * 1e6
            res["runs"][f"{label} R={r}"] = {"us_per_pass": med, "us_per_step": med / (r - 1)}
            print(f"{label:28s} R={r}: {med:8.0f} us per pass ({med / (r - 1):6.0f} us per head step)", flush=True)
    print(f"weights read per head step: {res['bytes_per_step_int8'] / 1e6:.2f} MB (int8)")
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
