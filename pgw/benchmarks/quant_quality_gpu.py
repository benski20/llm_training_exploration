"""Ring-format quality at 7B+ scale on one GPU (wikitext-2 perplexity and top-1 agreement).

The CPU harness (``quant_quality.py``) holds float64 copies and cannot load 7B. This version keeps each
linear as int8 rows plus float32 scales on the GPU (about 7 GB for 7B), quantizes activations per token
on the GPU and multiplies in float32. All formats use the same arithmetic, so the comparison isolates
the quantization:

* ``w8a12``  - int8 weights, 12-bit activations: today's K40A12 (worst-case ring, no cap).
* ``c27``    - K32A12C27: per token, the most bits <= 12 with ``max_j ||W8_j|| * ||xq|| <= 2^27``.
* ``cNN``    - the same with another cap (e.g. ``c23`` to see where reductions start).

Top-1 agreement is measured against the first config (put ``w8a12`` first).

    python benchmarks/quant_quality_gpu.py --model Qwen/Qwen2.5-7B-Instruct --configs w8a12 c27 c23
"""

from __future__ import annotations

import argparse
import json
import math
import time
from typing import Any

import torch
from torch import nn

GROUPS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


class QLinearGPU(nn.Module):
    """int8 weight rows + scales; per-token activation quantization (optionally capped)."""

    def __init__(self, lin: nn.Linear):
        super().__init__()
        w = lin.weight.detach().float()
        sc = w.abs().amax(dim=1) / 127.0
        sc = torch.where(sc == 0, torch.ones_like(sc), sc)
        self.register_buffer("w8", torch.clamp(torch.round(w / sc[:, None]), -127, 127).to(torch.int8))
        self.register_buffer("sc", sc)
        self.register_buffer("b", None if lin.bias is None else lin.bias.detach().float())
        self.wnorm = float(self.w8.float().pow(2).sum(dim=1).sqrt().max())
        self.abits = 12
        self.cap: int | None = None
        self.stats = {"tokens": 0, "reduced": 0, "min_bits": 99}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        xf = x.reshape(-1, shape[-1]).float()
        amax = xf.abs().amax(dim=1)
        rows = xf.shape[0]
        xq = torch.empty_like(xf)
        s = torch.ones(rows, device=xf.device)
        chosen = torch.full((rows,), -1, device=xf.device, dtype=torch.int64)
        for bits in range(self.abits, 1, -1):
            qmax = (1 << (bits - 1)) - 1
            sb = torch.where(amax > 0, amax / qmax, torch.ones_like(amax))
            q = torch.clamp(torch.round(xf / sb[:, None]), -qmax, qmax)
            if self.cap is None:
                fits = torch.ones(rows, dtype=torch.bool, device=xf.device)
            else:
                fits = q.norm(dim=1) * self.wnorm * (1 + 1e-6) <= float(1 << self.cap)
            take = fits & (chosen < 0)
            xq[take] = q[take]
            s[take] = sb[take]
            chosen[take] = bits
            if bool((chosen >= 0).all()):
                break
        st = self.stats
        st["tokens"] += rows
        st["reduced"] += int((chosen < self.abits).sum())
        st["min_bits"] = min(st["min_bits"], int(chosen.min()))
        y = (xq @ self.w8.float().t()) * s[:, None] * self.sc[None, :]
        if self.b is not None:
            y = y + self.b
        return y.to(x.dtype).reshape(*shape[:-1], -1)


def load_tokens(tok: Any, n: int) -> torch.Tensor:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download("Salesforce/wikitext", "wikitext-2-raw-v1/test-00000-of-00001.parquet",
                           repo_type="dataset")
    text = "\n\n".join(pq.read_table(path).column("text").to_pylist())
    return tok(text, return_tensors="pt").input_ids[0][:n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--configs", nargs="+", default=["w8a12", "c27", "c23"])
    ap.add_argument("--eval-tokens", type=int, default=16384)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_grad_enabled(False)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if dev.type == "cuda" else torch.float32
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, low_cpu_mem_usage=True).eval()
    qlins: list[QLinearGPU] = []
    for layer in model.model.layers:
        for parent in (layer.self_attn, layer.mlp):
            for name in GROUPS:
                lin = getattr(parent, name, None)
                if isinstance(lin, nn.Linear):
                    q = QLinearGPU(lin)
                    setattr(parent, name, q)
                    qlins.append(q)
        layer.to(dev)  # int8 linears + fp16 norms: about 0.25 GB per 7B layer
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    model.model.embed_tokens.to(dev)
    model.model.norm.to(dev)
    model.model.rotary_emb.to(dev)
    model.lm_head.to(dev)
    gpu = torch.cuda.get_device_name(0) if dev.type == "cuda" else None
    mem = round(torch.cuda.memory_allocated() / 1e9, 2) if dev.type == "cuda" else None
    print(json.dumps({"load_s": round(time.time() - t0, 1), "gpu_mem_GB": mem, "linears": len(qlins), "gpu": gpu}),
          flush=True)
    ids = load_tokens(tok, args.eval_tokens)

    results = []
    ref_top: list[torch.Tensor] | None = None
    for name in args.configs:
        cap = None if name == "w8a12" else int(name[1:])
        for q in qlins:
            q.cap = cap
            q.stats = {"tokens": 0, "reduced": 0, "min_bits": 99}
        t1 = time.time()
        nll, count, agree, tops = 0.0, 0, 0, []
        for k, i in enumerate(range(0, len(ids) - args.seq + 1, args.seq)):
            x = ids[i:i + args.seq][None].to(dev)
            logits = model(x, use_cache=False).logits[0].float()
            lp = torch.log_softmax(logits[:-1], -1)
            nll -= float(lp.gather(1, x[0, 1:, None]).sum())
            count += args.seq - 1
            top = logits.argmax(-1).cpu()
            tops.append(top)
            if ref_top is not None:
                agree += int((top == ref_top[k]).sum())
        if ref_top is None:
            ref_top = tops
        tot = sum(q.stats["tokens"] for q in qlins)
        red = sum(q.stats["reduced"] for q in qlins)
        row = {"config": name, "cap": cap, "ppl": round(math.exp(nll / count), 4),
               "top1_agreement_vs_first": None if not agree else round(agree / sum(len(t) for t in tops), 5),
               "token_linear_calls": tot, "fraction_with_fewer_bits": round(red / max(1, tot), 6),
               "min_activation_bits": min(q.stats["min_bits"] for q in qlins), "seconds": round(time.time() - t1, 1),
               "max_wnorm_log2": round(math.log2(max(q.wnorm for q in qlins)), 2)}
        results.append(row)
        print(json.dumps(row), flush=True)
    out = {"model": args.model, "eval_tokens": args.eval_tokens, "seq": args.seq, "results": results}
    print(json.dumps(out, indent=1))
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
