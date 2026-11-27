"""Trips per token of the fused schedules, end to end on CPU (in-process provider and plain factory), exact.

For each schedule: greedy decoding of ``--tokens`` tokens through the masked backend (mask, provider product on
masked ring elements only, unmask, range + F_(2^61-1) Freivalds on every returned value), then the plain reference
replays the same calls and must give the same logits bit for bit. Reports provider trips per decoded token, checks
per token, bytes up / down per token (K56: 7 bytes per element) and this box's ms per token (CPU provider: not a
speed measurement). Schedules:

* ``standard``: today's client-library reference (4 linears per layer + the masked int8 head): trips counted from the
  engine's trip plan (``4L + 1``), not run here (the masked standard path has its own e2e suites);
* ``fused2-folded``: fused2 + pure migration (8 bits) + the LM head folded into the last trip + layer 0's qkv local
  (``2L`` trips);
* ``parallel``: the pack read as a parallel-block model (a DIFFERENT model: only trips, exactness and checks are
  meaningful), one trip per layer + layer 0's (``L + 1``), head folded.

Usage:
    python benchmarks/fused_trips_e2e.py --hf-dir <Qwen2.5-0.5B-Instruct snapshot> --tokens 8 --json out.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch


def run(model: Any, plain: Any, be: Any, ids: torch.Tensor, tokens: int) -> dict[str, Any]:
    cache = model.new_cache(len(ids) + tokens + 1)
    calls = [ids]
    t0 = time.perf_counter()
    logits = [model.forward(ids, cache)]
    ttft = time.perf_counter() - t0
    f2 = be.f2
    t0_trips, t0_checks, t0_up, t0_down = f2.trips, f2.checks, f2.bytes_up, f2.bytes_down
    steps = []
    for _ in range(tokens):
        x = torch.tensor([int(logits[-1].argmax())])
        calls.append(x)
        t1 = time.perf_counter()
        logits.append(model.forward(x, cache))
        steps.append(time.perf_counter() - t1)
    n = max(1, tokens)
    pc = plain.new_cache(len(ids) + tokens + 1)
    ref = [plain.forward(c, pc) for c in calls]
    exact = all(torch.equal(a, b) for a, b in zip(ref, logits, strict=True))
    return {"trips_per_token": (f2.trips - t0_trips) / n, "checks_per_token": (f2.checks - t0_checks) / n,
            "bytes_up_per_token": (f2.bytes_up - t0_up) / n, "bytes_down_per_token": (f2.bytes_down - t0_down) / n,
            "ms_per_token_cpu": 1e3 * sum(steps) / n, "ttft_s_cpu": ttft, "exact": exact,
            "positions": len(calls), "trip_plan": model._trip_plan()[0]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", required=True)
    ap.add_argument("--tokens", type=int, default=8)
    ap.add_argument("--prompt", default="<|im_start|>user\nWhy is the sky blue?<|im_end|>\n<|im_start|>assistant\n")
    ap.add_argument("--schedules", default="fused2-folded,parallel")
    ap.add_argument("--json", default=None)
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    from transformers import AutoTokenizer

    from pgw.core.ring import RingParams
    from pgw.engine.backends import PlainBackend
    from pgw.engine.fused2 import (
        Fused2MaskedBackend,
        Fused2MaskSource,
        Fused2PlainBackend,
        Fused2Provider,
        Fused2Weights,
        enable_fused2,
        migration_multipliers,
    )
    from pgw.engine.fused_parallel import ParallelWeights, enable_fused_parallel
    from pgw.engine.masked_head import MaskedHeadPack, enable_masked_head
    from pgw.engine.model import TransformerModel
    from pgw.engine.pack import InMemoryPack

    ring = RingParams(56, 12, 53)
    pack = InMemoryPack.from_hf(Path(args.hf_dir))
    tok = AutoTokenizer.from_pretrained(args.hf_dir)
    ids = tok(args.prompt, return_tensors="pt").input_ids[0]
    L = pack.arch.num_layers
    hp = MaskedHeadPack(pack)
    std = TransformerModel(hp, PlainBackend(hp, RingParams(32, 12, 27)))
    enable_masked_head(std)
    out: dict[str, Any] = {"model": args.hf_dir, "layers": L, "prompt_tokens": int(ids.numel()),
                           "standard": {"trip_plan": std._trip_plan()[0], "expected": 4 * L + 1}}
    del std
    with torch.no_grad():
        for name in args.schedules.split(","):
            t0 = time.time()
            if name == "fused2-folded":
                w: Any = Fused2Weights(pack, multipliers=migration_multipliers(pack, 8, head=True), head=True)

                def enable(m: Any, w: Any = w) -> Any:
                    return enable_fused2(m, w, ring, head_fold=True, local_qkv0=True)
                expected = 2 * L
            elif name == "parallel":
                w = ParallelWeights(pack, bits=8, head=True)

                def enable(m: Any, w: Any = w) -> Any:
                    return enable_fused_parallel(m, w, ring, head_fold=True)
                expected = L + 1
            else:
                raise SystemExit(f"unknown schedule {name}")
            be = Fused2MaskedBackend(ring, Fused2Provider(w, ring.k), Fused2MaskSource(w, ring))
            model = TransformerModel(pack, be)
            enable(model)
            plain = TransformerModel(pack, Fused2PlainBackend(ring))
            enable(plain)
            r = run(model, plain, be, ids, args.tokens)
            r["expected_trips"] = expected
            r["build_s"] = time.time() - t0
            out[name] = r
            print(name, json.dumps(r), flush=True)
            if not r["exact"]:
                raise SystemExit(f"{name}: masked != plain")
            del model, plain, be, w
    print(json.dumps(out, indent=1))
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
