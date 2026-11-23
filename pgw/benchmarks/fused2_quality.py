"""Quality of the fused2 fixed-point reference (2 trips per layer) vs the float model and the int8 reference.

Runs the real engine (``TransformerModel``) with exact integer backends, so every number is the pinned
reference itself (no fake quantization):

* ``float``: the HF checkpoint in float32 (transformers), the quality ceiling;
* ``std:<ring>``: today's pinned int8 reference (standard 4-trip schedule, ``PlainBackend``);
* ``f2:<ring>``: fused2 with norm gains folded into the *float* weights before int8 rounding (pack time);
* ``f2i8:<ring>``: fused2 folded from the pack's int8 weights (no float checkpoint at pack time);
* ``f2m<b>+g<k>:<ring>``: the migrated split fold (:func:`pgw.engine.fused2.calibrate_fused2`, ``b``
  migration bits, ``k`` top columns, calibrated on wikitext-2 test tokens far from the evaluated ones);
* ``f2mig<b>:<ring>``: pure pack-time migration (:func:`pgw.engine.fused2.migration_multipliers`, ``b`` bits;
  no calibration data, no gateway columns; needs K56C53 at ``b = 8``);
* ``f2+g<k>:<ring>``: fused2 with the split fold (``k`` gateway columns per folded linear, int16 on the
  gateway; ``Fused2Weights(gateway_columns=k)``).

Metrics: wikitext-2 test perplexity (non-overlapping windows), top-1 agreement with float, and on fixed
prompts the max ``|delta logprob|`` of the actual next token and over the whole vocabulary (L-inf). For
fused2 also the activation bits actually used per trip row (adaptive bits below ``A`` = the cap bound).

Usage:
    python benchmarks/fused2_quality.py --configs std:K32A12C27 f2:K48A12C45 f2:K40A12C37 --tokens 4096
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import torch

PROMPTS = [
    "The capital of France is Paris. The capital of Germany is",
    "def fibonacci(n):\n    if n < 2:\n        return n\n    return",
    "Photosynthesis is the process by which green plants convert sunlight into",
    "<|im_start|>user\nWrite a haiku about the sea.<|im_end|>\n<|im_start|>assistant\n",
    "In 1969, Neil Armstrong became the first person to walk on the",
    "To be, or not to be, that is the question: Whether 'tis nobler in the mind to suffer",
]


def hf_dir(name: str = "Qwen2.5-0.5B-Instruct") -> Path:
    snaps = sorted((Path.home() / f".cache/huggingface/hub/models--Qwen--{name}/snapshots").glob("*"))
    if not snaps:
        raise SystemExit(f"{name} is not in the local HF cache")
    return snaps[-1]


def load_tokens(tok: Any, n: int, offset: int = 0) -> torch.Tensor:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download("Salesforce/wikitext", "wikitext-2-raw-v1/test-00000-of-00001.parquet",
                           repo_type="dataset", local_files_only=True)
    text = "\n\n".join(pq.read_table(path).column("text").to_pylist())
    return tok(text, return_tensors="pt").input_ids[0][offset : offset + n]


def windows(ids: torch.Tensor, seq: int) -> list[torch.Tensor]:
    return [ids[i : i + seq] for i in range(0, len(ids) - 1, seq) if len(ids[i : i + seq]) > 1]


@torch.no_grad()
def float_logits(model: Any, ids: torch.Tensor) -> torch.Tensor:
    return model(ids[None]).logits[0].float()


#: Bits of the gateway columns' integer weights (``--gateway-bits``).
GATEWAY_BITS = 12


def engine_runner(pack: Any, cfg: str, state: Any, calib: list[torch.Tensor] | None = None) -> tuple[Any, Any]:
    from pgw.core.ring import RingParams
    from pgw.engine.backends import PlainBackend
    from pgw.engine.fused2 import (
        Fused2PlainBackend,
        Fused2Weights,
        calibrate_fused2,
        enable_fused2,
        hf_float_linears,
        migration_multipliers,
    )
    from pgw.engine.model import TransformerModel

    kind, ring_name = cfg.split(":")
    ring = RingParams.from_name(ring_name)
    if kind == "std":
        return TransformerModel(pack, PlainBackend(pack, ring)), None
    kind, _, gcols = kind.partition("+g")  # "f2+g8": split fold, 8 gateway columns per folded linear
    if kind.startswith("f2mig"):  # "f2mig8": pure pack-time migration, 8 bits (no calibration, no gateway columns)
        fl = hf_float_linears(state, pack.arch)
        w = Fused2Weights(pack, fl, multipliers=migration_multipliers(pack, int(kind[5:] or 8)))
    elif kind.startswith("f2m"):  # "f2m6+g4": migrated split fold, 6 migration bits, 4 top columns
        assert calib is not None
        fl = hf_float_linears(state, pack.arch)
        cal = calibrate_fused2(pack, fl, calib, ring, bits=int(kind[3:] or 6), columns=int(gcols or 4),
                               gateway_bits=GATEWAY_BITS)
        print("calibration", cal["stats"], flush=True)
        w = Fused2Weights(pack, fl, gateway_columns=cal["gateway_columns"], multipliers=cal["multipliers"],
                          gateway_bits=cal["gateway_bits"])
    else:
        fl = hf_float_linears(state, pack.arch) if kind == "f2" else None
        w = Fused2Weights(pack, fl, gateway_columns=int(gcols or 0))
    del fl
    model = TransformerModel(pack, Fused2PlainBackend(ring))
    sched = enable_fused2(model, w, ring)
    return model, sched


@torch.no_grad()
def engine_logits(model: Any, ids: torch.Tensor) -> torch.Tensor:
    cache = model.new_cache(len(ids) + 1)
    return model.forward(ids, cache, all_logits=True).float()


def stats(logits: torch.Tensor, ids: torch.Tensor) -> tuple[float, int, torch.Tensor]:
    lp = torch.log_softmax(logits[:-1].float(), dim=-1)
    nll = -lp.gather(1, ids[1:, None]).sum().item()
    return nll, len(ids) - 1, logits[:-1].argmax(-1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=["std:K32A12C27", "f2:K48A12C45"])
    ap.add_argument("--tokens", type=int, default=4096)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--json", default=None)
    ap.add_argument("--calib-offset", type=int, default=150000,
                    help="calibration text for f2m: wikitext-2 test tokens from here (far from the eval tokens)")
    ap.add_argument("--calib-tokens", type=int, default=1024)
    ap.add_argument("--gateway-bits", type=int, default=12, help="f2m: bits of the gateway columns' weights")
    ap.add_argument("--model", default="Qwen2.5-0.5B-Instruct", help="a Qwen checkpoint in the local HF cache")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--save-prompts", default=None, help="path prefix: keep each config's prompt logprobs")
    ap.add_argument("--pack", default=None, help="an on-disk pgw pack of the same checkpoint (less memory)")
    ap.add_argument("--ref", default=None, help="cache of the float reference (and the int8 prompt logprobs)")
    args = ap.parse_args()
    global GATEWAY_BITS
    GATEWAY_BITS = args.gateway_bits
    if args.threads:
        torch.set_num_threads(args.threads)
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from pgw.engine.pack import InMemoryPack, arch_config_from_hf

    d = hf_dir(args.model)
    tok = AutoTokenizer.from_pretrained(d)
    ids = load_tokens(tok, args.tokens)
    wins = windows(ids, args.seq)
    prompts = [tok(p, return_tensors="pt").input_ids[0] for p in PROMPTS]
    ref_path = Path(args.ref) if args.ref else None
    if ref_path is not None and ref_path.is_file():
        ref = torch.load(ref_path)
    else:
        t0 = time.time()
        fm = AutoModelForCausalLM.from_pretrained(d, dtype=torch.float32).eval()
        ref_top: list[torch.Tensor] = []
        nll = cnt = 0.0
        for w in wins:
            a, b, top = stats(float_logits(fm, w), w)
            nll += a
            cnt += b
            ref_top.append(top)
        ref_prompt = [torch.log_softmax(float_logits(fm, p).double(), -1) for p in prompts]
        ref = {"ppl": math.exp(nll / cnt), "tokens": int(cnt), "top": ref_top, "prompt": ref_prompt, "int8": None}
        print(f"float ppl {ref['ppl']:.4f} over {int(cnt)} tokens ({time.time() - t0:.0f}s)", flush=True)
        del fm
        if ref_path is not None:
            torch.save(ref, ref_path)
    ref_top, ref_prompt = ref["top"], ref["prompt"]
    results: dict[str, Any] = {}
    if args.json and Path(args.json).is_file():
        results = json.loads(Path(args.json).read_text())
    results["float"] = {"ppl": ref["ppl"], "tokens": ref["tokens"]}
    state = d / "model.safetensors"  # read lazily, one linear at a time (low memory)
    if args.pack:
        from pgw.modelpack import ModelPack

        pack = ModelPack.open(args.pack)
    else:
        pack = InMemoryPack.from_state_dict(load_file(str(state)),
                                            arch_config_from_hf(json.loads((d / "config.json").read_text())))
    base_prompt: list[torch.Tensor] | None = ref.get("int8")
    for cfg in args.configs:
        t0 = time.time()
        calib = None
        if cfg.startswith("f2m") and not cfg.startswith("f2mig"):
            cids = load_tokens(tok, args.calib_tokens, args.calib_offset)
            calib = [cids[i : i + args.seq] for i in range(0, len(cids), args.seq)]
        model, sched = engine_runner(pack, cfg, state, calib)
        nll = cnt = agree = 0.0
        for w, rt in zip(wins, ref_top, strict=True):
            a, b, top = stats(engine_logits(model, w), w)
            nll += a
            cnt += b
            agree += float((top == rt).sum())
        lps = [torch.log_softmax(engine_logits(model, p).double(), -1) for p in prompts]
        if args.save_prompts:
            torch.save({"lps": [lp.float() for lp in lps]}, f"{args.save_prompts}.{cfg.replace(':', '_')}.pt")
        d_next = max(float((lp - rp)[torch.arange(len(p) - 1), p[1:]].abs().max())
                     for lp, rp, p in zip(lps, ref_prompt, prompts, strict=True))
        d_inf = max(float((lp - rp).abs().max()) for lp, rp in zip(lps, ref_prompt, strict=True))
        top_p = sum(int((lp.argmax(-1) == rp.argmax(-1)).sum()) for lp, rp in zip(lps, ref_prompt, strict=True))
        npos = sum(len(p) for p in prompts)
        r: dict[str, Any] = {
            "ppl": math.exp(nll / cnt), "top1_vs_float": agree / cnt,
            "prompt_max_dlogprob_next_vs_float": d_next, "prompt_max_dlogprob_vocab_vs_float": d_inf,
            "prompt_top1_vs_float": top_p / npos, "seconds": time.time() - t0,
        }
        if base_prompt is None and cfg.startswith("std:"):
            base_prompt = lps
            if ref_path is not None:
                ref["int8"] = lps
                torch.save(ref, ref_path)
        elif base_prompt is not None:
            r["prompt_max_dlogprob_next_vs_int8"] = max(
                float((lp - bp)[torch.arange(len(p) - 1), p[1:]].abs().max())
                for lp, bp, p in zip(lps, base_prompt, prompts, strict=True))
            r["prompt_top1_vs_int8"] = sum(int((lp.argmax(-1) == bp.argmax(-1)).sum())
                                           for lp, bp in zip(lps, base_prompt, strict=True)) / npos
        if sched is not None:
            tot = sum(sched.bits_hist.values())
            r["act_bits_hist"] = {str(k): v for k, v in sorted(sched.bits_hist.items())}
            r["rows_below_A"] = 1 - sched.bits_hist.get(sched.ring.a, 0) / max(tot, 1)
            r["max_out_bits"] = getattr(model.backend, "max_bits", None)
            r["gateway_cost"] = sched.w.gateway_cost()
        results[cfg] = r
        print(cfg, json.dumps(r), flush=True)
        del model, sched
        if args.json:
            Path(args.json).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
