#!/usr/bin/env python3
"""Strict speculation on a CPU-only gateway (every linear remote, native float ops), real 0.5B weights.

The real :class:`pgw.engine.Generator` in padded strict mode (``spec_mask_rows``) over :class:`MaskedBackend`
with ``local_linears = off`` and an in-process provider (``LocalComputeClient``: the provider's ring GEMMs on
this same CPU, so absolute speeds are far below a GPU provider's). Draft: a Qwen2.5 pack on
``NativeInt8DraftBackend`` (default: the 0.5B target itself, the only smaller-or-equal Qwen2.5 on this box,
so its acceptance is an upper bound; what carries over are the mechanics: pass cost at R rows, the draft
cost per pass, the pipelined drafter's hit rate and how much of the draft leaves the critical path).

Configurations (``--configs``): ``plain``; ``seq R/C`` (drafting between passes, :class:`BuiltinDrafter`);
``pipe R/C/m`` (:class:`PipelinedDrafter` with m foreground steps). Every run's tokens are compared with
plain decoding. Masks: the bench-only cycling reservation of ``spec_no_compromise_bench`` (mask reuse;
benchmark only).

Usage::

    PYTHONPATH=src python benchmarks/spec_cpu_gateway_bench.py --pack P [--draft-pack D] [--max-tokens 32]
        [--configs plain,seq4/4,pipe4/4/0,pipe4/4/1]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pack", required=True)
    ap.add_argument("--draft-pack", default=None)
    ap.add_argument("--ring", default="K32A12C27")
    ap.add_argument("--prompts", default="4,5")
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--configs", default="plain,seq4/4,pipe4/4/0,pipe4/4/1")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    import torch
    from spec_accept_bench import PROMPTS
    from spec_no_compromise_bench import _CycleReservation, _run

    from pgw.core.ring import RingParams
    from pgw.engine import ChatTokenizer, Generator, LocalComputeClient, MaskedBackend, SamplingParams, TransformerModel
    from pgw.engine.backends import NativeInt8DraftBackend, native_int8_lm_head
    from pgw.engine.generate import BuiltinDrafter, PipelinedDrafter
    from pgw.modelpack import ModelPack

    ring = RingParams.from_name(args.ring)
    pack = ModelPack.open(args.pack, verify="none")
    tok = ChatTokenizer.from_pack(pack)
    client = LocalComputeClient(pack)
    backend = MaskedBackend(pack, None, client, ring, pack.fingerprint, local_linears="off")
    model = TransformerModel(pack, backend)
    model.set_float_ops("native")
    lids = [lid for layer in model._lids for lid in layer]
    res = _CycleReservation(pack, ring, lids)
    dpack = ModelPack.open(args.draft_pack or args.pack, verify="none")
    dbe = NativeInt8DraftBackend(dpack, ring)
    dbe.preload()
    draft = TransformerModel(dpack, dbe)
    draft.lm_head_fn = native_int8_lm_head(draft.lm_head, ring)
    draft.set_float_ops("native")
    params = SamplingParams(max_tokens=args.max_tokens, temperature=0.0, ignore_eos=True)
    tv = int(pack.arch.vocab_size)
    ref: dict[int, list[int]] = {}
    report: dict[str, Any] = {"pack": args.pack, "draft": args.draft_pack or args.pack, "rows": [],
                              "torch_threads": torch.get_num_threads()}
    for cfg in args.configs.split(","):
        kw: dict[str, Any] = {}
        drafter: Any = None
        if cfg != "plain":
            parts = cfg[4:].split("/")
            r, c = int(parts[0]), float(parts[1])
            kw = {"spec_rows": r, "spec_strict": True, "spec_mask_rows": c}
            if cfg.startswith("pipe"):
                drafter = PipelinedDrafter(draft, post_steps=int(parts[2]), target_vocab=tv)
            else:
                drafter = BuiltinDrafter(draft, target_vocab=tv)
            kw["drafter"] = drafter
        agg = {"tokens": 0, "passes": 0.0, "decode_s": 0.0, "drain_s": 0.0, "identical": True, "draft_ms": 0.0,
               "draft_calls": 0.0}
        for pi in (int(x) for x in args.prompts.split(",")):
            ids = list(tok.apply_chat_template(PROMPTS[pi], add_generation_prompt=True))
            gen = Generator(model, tok, max_batch=1, **kw)
            out = asyncio.run(_run(gen, ids, params, res))
            asyncio.run(gen.aclose())
            if cfg == "plain":
                ref[pi] = out["tokens"]
            agg["identical"] &= out["tokens"] == ref.get(pi, out["tokens"])
            n = len(out["tokens"])
            agg["tokens"] += n
            st = gen.spec_stats
            agg["passes"] += (st.get("spec_passes", 0) + 1) if kw else n
            agg["decode_s"] += out["stamps"][-1] - out["stamps"][0]
            agg["drain_s"] += out["drain_s"]
            agg["draft_ms"] += st.get("draft_ms", 0.0)
            agg["draft_calls"] += st.get("draft_calls", 0.0)
        toks = max(1, agg["tokens"])
        nprom = len(args.prompts.split(","))
        all_passes = toks  # strict: L provider passes per answer (real + dummy)
        row: dict[str, Any] = {
            "config": cfg, "identical_to_plain": agg["identical"], "tokens": agg["tokens"],
            "tokens_per_real_pass": round(toks / max(1, agg["passes"]), 3),
            "answer_tok_s": round((toks - nprom) / max(1e-9, agg["decode_s"]), 3),
            "answer_ms_per_real_pass": round(1e3 * agg["decode_s"] / max(1, agg["passes"] - nprom), 1),
            "drain_s": round(agg["drain_s"], 2),
            "foreground_draft_ms_per_call": round(agg["draft_ms"] / max(1, agg["draft_calls"]), 1),
            "provider_passes": all_passes,
        }
        if drafter is not None:
            row["drafter_stats"] = {k: round(v, 1) for k, v in drafter.stats.items()}
            if isinstance(drafter, PipelinedDrafter):
                calls = max(1.0, drafter.stats["calls"])
                row["hit_rate"] = round(drafter.stats["hits"] / calls, 3)
                row["bg_ms_per_call"] = round(drafter.stats["bg_ms"] / calls, 1)
                row["wait_ms_per_call"] = round(drafter.stats["wait_ms"] / calls, 1)
        report["rows"].append(row)
        print(json.dumps(row), flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
