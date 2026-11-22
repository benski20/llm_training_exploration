"""Quality of the masked LM head's int8 head against the float head (``head_mode = "provider_int8"``).

The masked head is a new pinned reference: its logits are those of the pack's head quantized per row to int8
(``pgw.engine.masked_head``), not of the float head. This compares both with everything else identical (the same
int8 linears on ``PlainBackend``, the same ring and float path, the same final hidden rows):

* teacher-forced: one prompt pass per text (chat-templated prompts, plus optional wikitext passages) with the final
  hidden rows captured; every position's float-head logits (the model's own head) and int8-head logits (the head
  linear on the plaintext backend, as the masked head computes it). Reported: top-1 agreement, ``|delta logprob|``
  of the float head's top-1 token (max, mean, p99) and the max over the whole vocabulary of ``|delta logprob|``;
* free-running greedy (``--gen-prompts`` x ``--gen-tokens``): tokens until the two heads' answers first differ.

Usage::

    python benchmarks/masked_head_quality.py --pack /path/to/qwen2.5-0.5b-instruct-pack [--wikitext test.parquet]
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


def _texts(tok: Any, wikitext: str | None, passages: int, passage_tokens: int) -> list[tuple[str, list[int]]]:
    out = [(p[:40], list(tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True)))
           for p in PROMPTS]
    if wikitext:
        import pyarrow.parquet as pq

        lines = [x for x in pq.read_table(wikitext).column("text").to_pylist() if len(x.strip()) > 400]
        for i, line in enumerate(lines[:passages]):
            ids = list(tok.encode(line))[:passage_tokens]
            out.append((f"wikitext[{i}]", ids))
    return out


def _int8_head(x: torch.Tensor, hw: tuple[torch.Tensor, torch.Tensor, float], ring: Any) -> torch.Tensor:
    """:class:`pgw.engine.PlainBackend`'s head product (quantize rows, exact integer product, rescale), in
    blocks of output rows to bound memory (every output element depends only on its row and column)."""
    from pgw.core.ring import dequantize_outputs, quantize_activations
    from pgw.engine.backends import exact_int_matmul

    w8, scale, wnorm = hw
    xq, sx = quantize_activations(x, ring, wnorm if ring.cap else None)
    parts = []
    for a in range(0, int(w8.shape[0]), 8192):
        w8t = w8[a : a + 8192].t().contiguous()
        yq = exact_int_matmul(xq, w8t, ring.a)
        parts.append(dequantize_outputs(yq, sx, scale[a : a + 8192], None, torch.float32))
    return torch.cat(parts, dim=1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pack", required=True)
    ap.add_argument("--ring", default="K32A12C27")
    ap.add_argument("--wikitext", default=None, help="a wikitext parquet file (passages of more than 400 chars)")
    ap.add_argument("--passages", type=int, default=8)
    ap.add_argument("--passage-tokens", type=int, default=192)
    ap.add_argument("--layers", choices=("float", "bf16", "plain"), default="float",
                    help="backend of the decoder layers: 'float' (dequantized int8 weights, fast; both heads see the "
                         "same hidden rows either way), 'bf16' (the same in bfloat16: half the memory) or 'plain' "
                         "(the exact int8 reference, slow on a CPU)")
    ap.add_argument("--gen-prompts", type=int, default=0)
    ap.add_argument("--gen-tokens", type=int, default=24)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    from pgw.core.ring import RingParams
    from pgw.engine import ChatTokenizer, Generator, PlainBackend, SamplingParams, TransformerModel
    from pgw.engine.masked_head import MaskedHeadPack, enable_masked_head
    from pgw.engine.model import SequenceInput, rms_norm
    from pgw.modelpack import ModelPack

    ring = RingParams.from_name(args.ring)
    pack = ModelPack.open(args.pack, verify="fingerprint")
    hpack = MaskedHeadPack(pack)
    tok = ChatTokenizer.from_pack(pack)
    from pgw.engine import FloatBackend

    layers = {"float": lambda pk: FloatBackend(pk, ring),
              "bf16": lambda pk: FloatBackend(pk, ring, dtype=torch.bfloat16),
              "plain": lambda pk: PlainBackend(pk, ring)}[args.layers]
    flt = TransformerModel(pack, layers(pack))
    lid = hpack.head_id
    lw = hpack.load_linear(lid)  # the int8 head the provider serves (per-row int8 of the pack's plain head)
    from pgw.modelpack.loader import row_norm_sq_max

    hw = (lw.w8, lw.scale, __import__("math").sqrt(float(row_norm_sq_max(lw.w8))))
    last = int(pack.arch.num_layers) - 1
    rows = agree = 0
    d_top: list[torch.Tensor] = []
    d_all_max = 0.0
    per_text: list[dict[str, Any]] = []
    t0 = time.time()
    for name, ids in _texts(tok, args.wikitext, args.passages, args.passage_tokens):
        s = SequenceInput(ids, flt.new_cache(len(ids) + 1), None, all_logits=True, capture_hidden=(last,))
        la = flt.forward_batch([s])[0]
        h = s.hidden[last]  # type: ignore[index]
        x = rms_norm(h, flt.final_norm, flt.eps).to(torch.float32)  # as the model norms all_logits rows
        lb = _int8_head(x, hw, ring)
        pa, pb = torch.log_softmax(la, -1), torch.log_softmax(lb, -1)
        top_a, top_b = la.argmax(-1), lb.argmax(-1)
        n = int(la.shape[0])
        ag = int((top_a == top_b).sum())
        rows += n
        agree += ag
        dt = (pa.gather(1, top_a[:, None]) - pb.gather(1, top_a[:, None])).abs().flatten()
        d_top.append(dt)
        dall = float((pa - pb).abs().max())
        d_all_max = max(d_all_max, dall)
        per_text.append({"text": name, "positions": n, "top1_agree": round(ag / n, 4),
                         "max_dlogprob_top1": round(float(dt.max()), 5), "max_dlogprob_all": round(dall, 5)})
    d = torch.cat(d_top)
    greedy: list[dict[str, Any]] = []
    if args.gen_prompts > 0:
        i8 = TransformerModel(hpack, PlainBackend(hpack, ring))  # the exact pinned reference (slow on a CPU)
        flt = TransformerModel(pack, PlainBackend(pack, ring))
        enable_masked_head(i8, drop_float_head=False)
        params = SamplingParams(max_tokens=args.gen_tokens, temperature=0.0, ignore_eos=True)
        gf, gi = Generator(flt, tok, max_batch=1), Generator(i8, tok, max_batch=1)
        for p in PROMPTS[: args.gen_prompts]:
            ids = tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True)
            a, b = gf.generate(ids, params), gi.generate(ids, params)
            first = next((i for i, (u, v) in enumerate(zip(a.token_ids, b.token_ids, strict=False)) if u != v), None)
            greedy.append({"prompt": p[:40], "equal": a.token_ids == b.token_ids, "first_divergence": first})
    out = {
        "pack": pack.name, "fingerprint": pack.fingerprint[:12], "head_fingerprint": hpack.fingerprint[:12],
        "ring": ring.name, "layers": args.layers, "tied_head": bool(pack.arch.tie_word_embeddings),
        "texts": len(per_text),
        "positions": rows, "top1_agreement": round(agree / rows, 5),
        "max_abs_dlogprob_top1": round(float(d.max()), 5),
        "mean_abs_dlogprob_top1": round(float(d.mean()), 6),
        "p99_abs_dlogprob_top1": round(float(d.quantile(0.99)), 5),
        "max_abs_dlogprob_all_vocab": round(d_all_max, 5),
        "greedy_equal": f"{sum(1 for g in greedy if g['equal'])}/{len(greedy)} x {args.gen_tokens} tokens",
        "seconds": round(time.time() - t0, 1),
        "per_text": per_text, "greedy": greedy,
    }
    print(json.dumps(out, indent=1))
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
