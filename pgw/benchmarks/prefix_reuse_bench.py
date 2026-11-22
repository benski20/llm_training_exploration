"""Conversation KV-prefix reuse (``prefix_cache = "conversation"``, SPEC 9.4b): mask slots and TTFT per turn of a
multi-turn chat, with and without reuse, on a real checkpoint (default Qwen2.5-0.5B-Instruct from the local HF cache).

Both runs use the native float path and the exact integer reference backend (``PlainBackend``: the gateway's
own per-row work of the masked path, without the network and the service), greedy decoding, the model's chat
template, and the same conversation (the assistant turns are the model's own answers). Every turn's answer and
logprobs are checked bit-identical between the two runs. Per turn it reports the prompt length, the reused
prefix, the prompt-pass rows (what the provider would see), the mask slots the request reserves
(``Generator.mask_rows_needed``) and consumes, TTFT (wall) and the prompt pass's process CPU time.

    python benchmarks/prefix_reuse_bench.py --turns 5 --max-tokens 32 --json /tmp/prefix_reuse.json
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path
from typing import Any

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from pgw.core import DEFAULT_RING  # noqa: E402
from pgw.engine import (  # noqa: E402
    ChatTokenizer,
    Generator,
    InMemoryPack,
    PlainBackend,
    SamplingParams,
    TransformerModel,
)
from pgw.engine.prefix_cache import ConversationPrefixCache  # noqa: E402

SYSTEM = ("You are a careful assistant for an engineering team. Answer briefly and precisely, state assumptions, "
          "prefer concrete numbers, and say when you are unsure. Keep answers under five sentences.")
USER = [
    "We run a gateway that masks activations before sending them to a GPU provider. What limits its latency?",
    "Suppose each token needs 96 round trips at 0.3 ms each. What is the floor in tokens per second?",
    "How would speculative decoding with two drafted tokens change that estimate?",
    "And if the gateway reuses the KV cache of earlier turns, which part of a chat gets cheaper?",
    "Summarize the three numbers we discussed in one line.",
    "One more: what should we measure first next week?",
]


def default_model_dir() -> str:
    hits = glob.glob(str(Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/*/"))
    if not hits:
        raise SystemExit("Qwen2.5-0.5B-Instruct is not in the local HF cache; pass --model-dir")
    return hits[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--turns", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    d = args.model_dir or default_model_dir()
    t0 = time.perf_counter()
    pack = InMemoryPack.from_hf(d)
    tok = ChatTokenizer(d)
    model = TransformerModel(pack, PlainBackend(pack, DEFAULT_RING))
    model.set_float_ops("native")
    # warm-up: every linear's weights are prepared on first use (not part of any turn's TTFT)
    Generator(model, tok).generate(tok.encode("warm up", add_special_tokens=False), SamplingParams(max_tokens=2))
    print(f"loaded and warmed {Path(d).parts[-3]} in {time.perf_counter() - t0:.1f} s", flush=True)
    reuse_gen = Generator(model, tok)
    plain_gen = Generator(model, tok)
    ok, why = reuse_gen.prefix_reuse_supported()
    assert ok, why
    store = ConversationPrefixCache(max_bytes=1 << 30, ttl_s=600.0)
    key = "key:bench\x00user:\x00conv:"
    params = SamplingParams(max_tokens=args.max_tokens, temperature=0.0, logprobs=1)
    msgs: list[dict[str, str]] = [{"role": "system", "content": SYSTEM}]
    rows: list[dict[str, Any]] = []
    for t in range(args.turns):
        msgs.append({"role": "user", "content": USER[t % len(USER)]})
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True)
        out: dict[str, Any] = {"turn": t + 1, "prompt_tokens": len(ids)}
        results = {}
        for name in ("recompute", "reuse"):  # alternate so both see the same machine load
            gen = plain_gen if name == "recompute" else reuse_gen
            kw: dict[str, Any] = {}
            reused = 0
            if name == "reuse":
                hit = store.take(key, ids)
                reused = hit.length if hit is not None else 0
                kw = {"prefix": hit, "prefix_sink": store.sink(key)}
            c0 = time.process_time()
            res = gen.generate(ids, params, **kw)
            cpu = time.process_time() - c0
            results[name] = res
            out[name] = {
                "reused": reused, "prompt_rows": len(ids) - reused,
                "mask_slots_reserved": gen.mask_rows_needed(len(ids), params.max_tokens, reused),
                "mask_slots_consumed": res.masks_consumed, "completion_tokens": res.completion_tokens,
                "ttft_ms": round(res.timings_ms["ttft_ms"], 1), "total_ms": round(res.timings_ms["total_ms"], 1),
                "cpu_s": round(cpu, 2),
            }
        a, b = results["reuse"], results["recompute"]
        out["identical"] = a.token_ids == b.token_ids and a.logprobs == b.logprobs and a.text == b.text
        rows.append(out)
        print(json.dumps(out), flush=True)
        msgs.append({"role": "assistant", "content": a.text})
    tot = {n: {k: sum(r[n][k] for r in rows) for k in ("prompt_rows", "mask_slots_reserved", "mask_slots_consumed",
                                                       "ttft_ms", "cpu_s")} for n in ("recompute", "reuse")}
    later = rows[1:]
    summary = {
        "turns": args.turns, "max_tokens": args.max_tokens, "all_identical": all(r["identical"] for r in rows),
        "totals": tot,
        "mask_slots_reduction": round(
            1 - tot["reuse"]["mask_slots_reserved"] / tot["recompute"]["mask_slots_reserved"], 3),
        "prompt_rows_reduction": round(1 - tot["reuse"]["prompt_rows"] / tot["recompute"]["prompt_rows"], 3),
        "ttft_ms_turns_2plus": {n: [r[n]["ttft_ms"] for r in later] for n in ("recompute", "reuse")},
        "prefix_cache": store.status(),
    }
    print(json.dumps(summary, indent=1), flush=True)
    store.close()
    if args.json:
        Path(args.json).write_text(json.dumps({"rows": rows, "summary": summary}, indent=1))


if __name__ == "__main__":
    main()
