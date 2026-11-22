"""Quality of the masked LM head's int8 head against the float head (``head_mode = "provider_int8"``).

The masked head is a new pinned reference: its logits are those of the pack's head quantized per row to int8
(``pgw.engine.masked_head``), not of the float head. This compares both on a fixed prompt set, with everything
else identical (the same int8 linears on ``PlainBackend``, the same ring, float path and hidden states):

* teacher-forced: for every position of prompt + the float head's greedy continuation, top-1 agreement of the two
  heads, ``|delta logprob|`` of the float head's top-1 token (max, mean, p99) and the max over the whole
  vocabulary of ``|delta logprob|`` (all log-softmax values);
* free-running greedy: tokens until the two heads' greedy answers first differ, and whole-answer equality.

Usage::

    python benchmarks/masked_head_quality.py --pack /path/to/qwen2.5-0.5b-instruct-pack [--tokens 64] [--json out]
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

import torch

PROMPTS = [
    "Explain in two sentences why the sky is blue.",
    "Write a haiku about a lighthouse in winter.",
    "What is the capital of Australia, and why is it not Sydney?",
    "Give three tips for writing readable Python code.",
    "Translate into French: The meeting has been moved to Thursday afternoon.",
    "Summarize the plot of Romeo and Juliet in one paragraph.",
    "A train leaves at 9:40 and arrives at 13:15. How long is the trip?",
    "List the planets of the solar system in order from the Sun.",
    "Write a short SQL query that counts orders per customer.",
    "What are the main differences between TCP and UDP?",
    "Describe the water cycle to a ten-year-old.",
    "Suggest a name for a coffee shop run by cats, and explain it.",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pack", required=True)
    ap.add_argument("--ring", default="K32A12C27")
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--float-ops", default=None, choices=[None, "torch", "native"])
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    from pgw.core.ring import RingParams
    from pgw.engine import ChatTokenizer, Generator, PlainBackend, SamplingParams, TransformerModel
    from pgw.engine.masked_head import MaskedHeadPack, enable_masked_head
    from pgw.engine.model import SequenceInput
    from pgw.modelpack import ModelPack

    ring = RingParams.from_name(args.ring)
    pack = ModelPack.open(args.pack, verify="fingerprint")
    hpack = MaskedHeadPack(pack)
    tok = ChatTokenizer.from_pack(pack)
    flt = TransformerModel(pack, PlainBackend(pack, ring))
    i8 = TransformerModel(hpack, PlainBackend(hpack, ring))
    enable_masked_head(i8, drop_float_head=False)
    if args.float_ops:
        flt.set_float_ops(args.float_ops)
        i8.set_float_ops(args.float_ops)
    params = SamplingParams(max_tokens=args.tokens, temperature=0.0, ignore_eos=True)
    gf, gi = Generator(flt, tok, max_batch=1), Generator(i8, tok, max_batch=1)
    rows = agree = 0
    d_top: list[float] = []
    d_all_max = 0.0
    per_prompt: list[dict[str, Any]] = []
    t0 = time.time()
    for p in PROMPTS:
        ids = tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True)
        a = gf.generate(ids, params)
        b = gi.generate(ids, params)
        first = next((i for i, (x, y) in enumerate(zip(a.token_ids, b.token_ids, strict=False)) if x != y), None)
        seq = list(ids) + list(a.token_ids)
        la = flt.forward_batch([SequenceInput(seq, flt.new_cache(len(seq) + 1), None, all_logits=True)])[0]
        lb = i8.forward_batch([SequenceInput(seq, i8.new_cache(len(seq) + 1), None, all_logits=True)])[0]
        pa, pb = torch.log_softmax(la.double(), -1), torch.log_softmax(lb.double(), -1)
        top_a, top_b = la.argmax(-1), lb.argmax(-1)
        n = int(la.shape[0])
        rows += n
        ag = int((top_a == top_b).sum())
        agree += ag
        dt = (pa.gather(1, top_a[:, None]) - pb.gather(1, top_a[:, None])).abs().flatten()
        d_top.extend(float(x) for x in dt)
        dall = float((pa - pb).abs().max())
        d_all_max = max(d_all_max, dall)
        per_prompt.append({"prompt": p[:40], "positions": n, "top1_agree": round(ag / n, 4),
                           "max_dlogprob_top1": round(float(dt.max()), 5), "max_dlogprob_all": round(dall, 5),
                           "greedy_equal": a.token_ids == b.token_ids, "first_divergence": first})
    d = torch.tensor(d_top)
    out = {
        "pack": pack.name, "fingerprint": pack.fingerprint[:12], "head_fingerprint": hpack.fingerprint[:12],
        "ring": ring.name, "tied_head": bool(pack.arch.tie_word_embeddings), "prompts": len(PROMPTS),
        "tokens_per_prompt": args.tokens, "positions": rows,
        "top1_agreement": round(agree / rows, 5),
        "max_abs_dlogprob_top1": round(float(d.max()), 5),
        "mean_abs_dlogprob_top1": round(float(d.mean()), 6),
        "p99_abs_dlogprob_top1": round(float(d.quantile(0.99)), 5),
        "max_abs_dlogprob_all_vocab": round(d_all_max, 5),
        "greedy_answers_equal": sum(1 for x in per_prompt if x["greedy_equal"]),
        "median_first_divergence": sorted([x["first_divergence"] if x["first_divergence"] is not None
                                           else args.tokens for x in per_prompt])[len(per_prompt) // 2],
        "seconds": round(time.time() - t0, 1),
        "per_prompt": per_prompt,
    }
    print(json.dumps(out, indent=1))
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
