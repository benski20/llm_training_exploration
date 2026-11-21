#!/usr/bin/env python3
"""Draft acceptance of exact strict speculation on real chat prompts (tokens per masked pass).

Strict fixed-row speculation (``spec_rows`` = R with ``spec_strict_passes``) verifies the pending token
and R - 1 drafts in one masked pass, at the same number of sequential round trips as a plain decode
pass, so the answer streams ``tokens_per_pass`` times faster when the trip time dominates. Whether that
pays off depends only on how often the target agrees with the draft, which this script measures on the
exact quantized target (``PlainBackend``: bit-identical to the masked gateway, so the acceptance is the
masked acceptance; no service and no masks are involved).

Two measurements:

* **Simulation over every R at once (greedy).** The target's exact greedy answer is generated once per
  prompt; each drafter is then asked, at every position, for its proposal given the target's own
  context. With greedy acceptance the drafts of a pass are accepted while they equal the target's
  tokens, so the number of tokens a pass at position ``i`` emits is ``min(m(i), R - 1) + 1``, where
  ``m(i)`` is the length of the drafter's correct prefix there. A draft model is run teacher-forced
  over the answer (one forward pass): while its proposals are correct, its context equals the target's,
  so ``m(i)`` is the run of positions from ``i`` where its argmax equals the target token.
* **Generator check (``--check R``).** The real :class:`pgw.engine.Generator` in strict fixed-row mode
  (``spec_rows=R``, ``spec_strict=True``) on the same prompts, greedy and sampled (``--temperature``):
  tokens must equal the plain generation (exactness), and its own counters give the measured tokens per
  pass, the draft time per pass and the dummy passes.

Drafters: ``lookup`` (prompt lookup over the conversation, ``n`` = 4..3, the gateway's
:func:`pgw.engine.generate._lookup_drafts_fixed`), ``lookup2`` (n down to 2), ``model`` (the draft pack,
``--draft-pack``), ``lookup+model`` (lookup first, the model completes the proposal; the gateway's
:class:`BuiltinDrafter`).

Usage::

    PYTHONPATH=src python benchmarks/spec_accept_bench.py --target-pack P [--draft-pack D] [--device cuda]
        [--draft-device cuda] [--max-tokens 128] [--prompts 24] [--rs 2,3,4,5,6,8] [--check 4]
        [--temperature 0.7] [--json out.json]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

#: Chat prompts (single and multi-turn) of the kinds a chat demo sees: explanations, how-tos, code,
#: rewriting and summarizing given text (which prompt lookup can copy from), lists, short math.
PROMPTS: list[list[dict[str, str]]] = [
    [{"role": "user", "content": "Explain in two sentences why one-time pads are secure."}],
    [{"role": "user", "content": "What is the difference between a process and a thread? Keep it short."}],
    [{"role": "user", "content": "Write a Python function that checks whether a string is a palindrome, "
                                 "ignoring case and punctuation."}],
    [{"role": "user", "content": "Give me five tips for sleeping better."}],
    [{"role": "user", "content": "Summarize this paragraph in one sentence: The mitochondria is the organelle "
                                 "that produces most of the chemical energy needed to power the cell's "
                                 "biochemical reactions. Chemical energy produced by the mitochondria is stored "
                                 "in a small molecule called adenosine triphosphate (ATP). Mitochondria contain "
                                 "their own small chromosomes."}],
    [{"role": "user", "content": "Rewrite this email to sound more polite: Hi Tom, the report you sent is late "
                                 "again and has errors in the second table. Fix it by Friday. Thanks, Ana"}],
    [{"role": "user", "content": "How do I make a simple tomato pasta sauce?"}],
    [{"role": "user", "content": "What is 17 times 23? Show your steps."}],
    [{"role": "user", "content": "Translate to French: The weather is nice today, so we are going to the park."}],
    [{"role": "user", "content": "Write a haiku about autumn leaves."}],
    [{"role": "user", "content": "List the planets of the solar system in order from the sun."}],
    [{"role": "user", "content": "What does HTTP status code 404 mean, and when should a server return 410 "
                                 "instead?"}],
    [{"role": "user", "content": "Explain recursion to a ten-year-old."}],
    [{"role": "user", "content": "Write a SQL query that returns the ten customers with the highest total order "
                                 "amount from tables customers(id, name) and orders(id, customer_id, amount)."}],
    [{"role": "user", "content": "Convert this list to JSON with keys name and age: Alice 31, Bob 27, "
                                 "Carol 45."}],
    [{"role": "user", "content": "What are the pros and cons of remote work?"}],
    [{"role": "user", "content": "Fix the bug in this code: def add_all(xs):\n    total = 0\n    for x in xs:\n"
                                 "        total = x\n    return total"}],
    [{"role": "user", "content": "Draft a short thank-you note to a colleague who helped me move apartments."}],
    [{"role": "user", "content": "Why is the sky blue?"}],
    [{"role": "user", "content": "Give a one-paragraph overview of the French Revolution."}],
    [{"role": "user", "content": "What is a hash table?"},
     {"role": "assistant", "content": "A hash table is a data structure that maps keys to values using a hash "
                                      "function to compute an index into an array of buckets."},
     {"role": "user", "content": "How are collisions handled? Give two methods."}],
    [{"role": "user", "content": "Write a bash one-liner that counts the lines of all .py files in a folder."},
     {"role": "assistant", "content": "find . -name '*.py' | xargs wc -l"},
     {"role": "user", "content": "Now make it skip the venv folder and print only the total."}],
    [{"role": "user", "content": "My name is Priya and I am planning a trip to Japan in April."},
     {"role": "assistant", "content": "That sounds wonderful, Priya! April is cherry blossom season in Japan. "
                                      "How can I help you plan your trip?"},
     {"role": "user", "content": "Suggest a 5-day itinerary for Tokyo and Kyoto."}],
    [{"role": "system", "content": "You are a concise assistant. Answer in at most three sentences."},
     {"role": "user", "content": "What causes inflation?"}],
]

DEFAULT_RS = (2, 3, 4, 5, 6, 8)
KMAX = 7  # longest draft chain simulated (R = 8)


def _run_lengths(match: list[bool]) -> list[int]:
    """``out[i]`` = number of consecutive True from ``i``."""
    out = [0] * (len(match) + 1)
    for i in range(len(match) - 1, -1, -1):
        out[i] = out[i + 1] + 1 if match[i] else 0
    return out[: len(match)]


def simulate_passes(m: list[int], n_gen: int, r: int) -> int:
    """Decode passes (after the prefill pass, which emits token 0) to emit ``n_gen`` tokens with R rows,
    when a pass at position ``i`` has ``m[i]`` correct drafts (greedy chain acceptance)."""
    i, passes = 1, 0
    while i < n_gen:
        a = min(m[i], r - 1)
        i += a + 1
        passes += 1
    return passes


def lookup_m(full: list[int], p: int, gen: list[int], max_n: int, min_n: int) -> list[int]:
    from pgw.engine.generate import _lookup_drafts_fixed

    m = [0] * len(gen)
    for i in range(1, len(gen)):
        prop = _lookup_drafts_fixed(full[: p + i], KMAX, max_n, min_n)
        k = 0
        while k < len(prop) and i + k < len(gen) and prop[k] == gen[i + k]:
            k += 1
        m[i] = k
    return m


def model_match(draft: Any, full: list[int], p: int, gen: list[int], vocab: int | None) -> list[bool]:
    """Teacher-forced draft argmax at every answer position: ``match[i]`` = the draft, given the target's
    context ``full[:p+i]``, proposes ``gen[i]``."""
    import torch

    from pgw.engine.model import SequenceInput

    cache = draft.new_cache(len(full) + 1)
    ids = [t if vocab is None or t < vocab else vocab - 1 for t in full]
    with torch.no_grad():
        logits = draft.forward_batch([SequenceInput(ids, cache, None, all_logits=True)])[0]
    pred = logits.float().argmax(-1).tolist()  # row j predicts full[j + 1]
    return [pred[p + i - 1] == gen[i] for i in range(len(gen))]


def combo_m(look: list[int], mm: list[int]) -> list[int]:
    """lookup+model: the lookup proposal first; a correct lookup proposal shorter than the chain is
    completed by the draft model (whose context then equals the target's)."""
    return [max(a, b) for a, b in zip(look, mm, strict=True)]


def summarize(ms: dict[str, list[list[int]]], gens: list[list[int]], rs: tuple[int, ...]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, per_prompt in ms.items():
        row: dict[str, Any] = {}
        for r in rs:
            passes = sum(simulate_passes(m, len(g), r) for m, g in zip(per_prompt, gens, strict=True))
            toks = sum(max(0, len(g) - 1) for g in gens)
            row[str(r)] = round(toks / passes, 3) if passes else None
        first = [m[i] > 0 for m, g in zip(per_prompt, gens, strict=True) for i in range(1, len(g))]
        row["p_first_draft_ok"] = round(sum(first) / max(1, len(first)), 3)
        out[name] = row
    return out


async def _generate(model: Any, tok: Any, ids: list[int], params: Any, **gen_kw: Any) -> tuple[list[int], dict]:
    from pgw.engine import Generator

    gen = Generator(model, tok, max_batch=1, **gen_kw)
    t0 = time.perf_counter()
    res = await gen.agenerate(ids, params, session=None, conversation="bench")
    wall = time.perf_counter() - t0
    await gen.wait_drained()
    stats = dict(gen.spec_stats)
    await gen.aclose()
    stats["wall_s"] = wall
    return list(res.token_ids), stats


def build_target(pack_path: str, ring: str, device: str) -> tuple[Any, Any, Any]:
    from pgw.core.ring import RingParams
    from pgw.engine import ChatTokenizer, PlainBackend, TransformerModel
    from pgw.modelpack import ModelPack

    pack = ModelPack.open(pack_path, verify="fingerprint")
    tok = ChatTokenizer.from_pack(pack)
    model = TransformerModel(pack, PlainBackend(pack, RingParams.from_name(ring), device=device), device=device)
    return pack, tok, model


def build_draft(pack_path: str, ring: str, device: str) -> tuple[Any, str]:
    """The draft as the gateway loads it (``GatewayModel._load_draft``): float16 on a GPU, weight-only int8
    (when faster) or float32 on a CPU."""
    import torch

    from pgw.core.ring import RingParams
    from pgw.engine import TransformerModel
    from pgw.engine.backends import FloatBackend, Int8DraftBackend, int8_lm_head
    from pgw.modelpack import ModelPack

    rp = RingParams.from_name(ring)
    dpack = ModelPack.open(pack_path, verify="fingerprint")
    if torch.device(device).type == "cuda":
        draft = TransformerModel(dpack, FloatBackend(dpack, rp, device=device, dtype=torch.float16, batch_rows=True),
                                 device=device, dtype=torch.float16)
        w16 = draft.lm_head.to(torch.float16)
        draft.lm_head = w16
        draft.lm_head_fn = lambda h: (h.to(torch.float16) @ w16.t()).to(torch.float32)
        return draft, "float16"
    if Int8DraftBackend.available() and Int8DraftBackend.faster_than_float(dpack):
        draft = TransformerModel(dpack, Int8DraftBackend(dpack, rp), device=device)
        draft.lm_head_fn = int8_lm_head(draft.lm_head)
        return draft, "int8"
    return TransformerModel(dpack, FloatBackend(dpack, rp, device=device, batch_rows=True), device=device), "float32"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--target-pack", required=True)
    ap.add_argument("--draft-pack", default=None)
    ap.add_argument("--ring", default="K32A12C27")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--draft-device", default=None, help="default: --device")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--prompts", type=int, default=len(PROMPTS))
    ap.add_argument("--rs", default=",".join(map(str, DEFAULT_RS)))
    ap.add_argument("--check", default="", help="comma-separated R values for the Generator check ('' = none)")
    ap.add_argument("--check-prompts", type=int, default=8)
    ap.add_argument("--check-drafter", default="auto", choices=["auto", "lookup", "model", "lookup+model"])
    ap.add_argument("--temperature", type=float, default=0.7, help="sampled Generator check (0 = greedy only)")
    ap.add_argument("--label", default="")
    ap.add_argument("--json", default=None)
    ap.add_argument("--dump-tokens", default=None, help="write prompts' ids and greedy answers here (JSON)")
    args = ap.parse_args(argv)

    import torch

    from pgw.engine import SamplingParams

    rs = tuple(int(x) for x in args.rs.split(",") if x)
    pack, tok, model = build_target(args.target_pack, args.ring, args.device)
    draft = None
    draft_kind = None
    ddev = args.draft_device or args.device
    if args.draft_pack:
        draft, draft_kind = build_draft(args.draft_pack, args.ring, ddev)
    prompts = PROMPTS[: args.prompts]
    report: dict[str, Any] = {"label": args.label, "target": getattr(pack, "name", args.target_pack),
                              "draft": args.draft_pack, "draft_kind": draft_kind, "device": args.device,
                              "draft_device": ddev, "max_tokens": args.max_tokens, "prompts": len(prompts),
                              "torch": torch.__version__,
                              "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    greedy = SamplingParams(max_tokens=args.max_tokens, temperature=0.0)
    ids_all: list[list[int]] = []
    gens: list[list[int]] = []
    t0 = time.perf_counter()
    for msgs in prompts:
        ids = list(tok.apply_chat_template(msgs, add_generation_prompt=True))
        out, _ = asyncio.run(_generate(model, tok, ids, greedy))
        ids_all.append(ids)
        gens.append(out)
    report["greedy_s"] = round(time.perf_counter() - t0, 2)
    report["answer_tokens"] = [len(g) for g in gens]
    print(f"[spec_accept] greedy answers: {sum(map(len, gens))} tokens in {report['greedy_s']} s", flush=True)

    ms: dict[str, list[list[int]]] = {"lookup": [], "lookup2": []}
    if draft is not None:
        ms["model"] = []
        ms["lookup+model"] = []
    dvocab = int(draft.arch.vocab_size) if draft is not None else None
    t0 = time.perf_counter()
    for ids, gen in zip(ids_all, gens, strict=True):
        full = ids + gen
        p = len(ids)
        lk = lookup_m(full, p, gen, 4, 3)
        ms["lookup"].append(lk)
        ms["lookup2"].append(lookup_m(full, p, gen, 4, 2))
        if draft is not None:
            mm = _run_lengths(model_match(draft, full, p, gen, dvocab))
            ms["model"].append(mm)
            ms["lookup+model"].append(combo_m(lk, mm))
    report["sim_s"] = round(time.perf_counter() - t0, 2)
    report["tokens_per_pass_sim"] = summarize(ms, gens, rs)
    # per prompt at R = 4 for the strongest drafter (spread across chat kinds)
    best = "lookup+model" if draft is not None else "lookup"
    report["per_prompt_r4"] = {best: [round((len(g) - 1) / max(1, simulate_passes(m, len(g), 4)), 2)
                                      for m, g in zip(ms[best], gens, strict=True)]}
    print(json.dumps(report["tokens_per_pass_sim"], indent=1), flush=True)

    checks: list[dict[str, Any]] = []
    drafter = args.check_drafter
    if drafter == "auto":
        drafter = "lookup+model" if draft is not None else "lookup"
    for r in (int(x) for x in args.check.split(",") if x):
        temps = [0.0] + ([args.temperature] if args.temperature > 0 else [])
        for temp in temps:
            agg: dict[str, float] = {"tokens": 0, "passes": 0, "accepted": 0, "draft_ms": 0.0, "draft_calls": 0,
                                     "dummy_passes": 0, "spec_wall_s": 0.0, "plain_wall_s": 0.0}
            identical = True
            for j, ids in enumerate(ids_all[: args.check_prompts]):
                params = SamplingParams(max_tokens=args.max_tokens, temperature=temp, top_p=0.8 if temp else 1.0,
                                        seed=1234 + j)
                plain, pst = asyncio.run(_generate(model, tok, ids, params))
                kw: dict[str, Any] = {"spec_rows": r, "spec_strict": True}
                if drafter in ("lookup", "lookup+model"):
                    kw["lookup"] = 4
                if drafter in ("model", "lookup+model"):
                    kw["draft"] = draft
                spec, sst = asyncio.run(_generate(model, tok, ids, params, **kw))
                identical &= spec == plain
                agg["tokens"] += len(spec)
                agg["passes"] += sst["spec_passes"]
                agg["accepted"] += sst["spec_accepted"]
                agg["draft_ms"] += sst["draft_ms"]
                agg["draft_calls"] += sst["draft_calls"]
                agg["dummy_passes"] += sst.get("dummy_passes", 0)
                agg["spec_wall_s"] += sst["wall_s"]
                agg["plain_wall_s"] += pst["wall_s"]
            row = {"R": r, "temperature": temp, "drafter": drafter, "identical_to_plain": identical,
                   "tokens_per_pass": round((agg["passes"] + agg["accepted"]) / max(1, agg["passes"]), 3),
                   "draft_ms_per_pass": round(agg["draft_ms"] / max(1, agg["draft_calls"]), 2),
                   **{k: round(v, 2) for k, v in agg.items()}}
            print("[spec_accept] check", json.dumps(row), flush=True)
            checks.append(row)
    report["generator_checks"] = checks
    if args.dump_tokens:
        Path(args.dump_tokens).write_text(json.dumps({"prompts": ids_all, "answers": gens}))
    txt = json.dumps(report, indent=1)
    if args.json:
        Path(args.json).write_text(txt)
    print(txt)
    return 0 if all(c["identical_to_plain"] for c in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
