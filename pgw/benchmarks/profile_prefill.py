#!/usr/bin/env python3
"""Time to first token: where a masked prefill of T prompt tokens goes (chat-speed plan WP-A0).

Run 27 (7B, T4): TTFT 2.35 s for a 40-token prompt; prefill is throughput-bound (8.9 MB of wire and
5.56 MB of masks per prompt token at K=32). Per prompt length T this reports:

* ``volume``: per prompt token and per prompt, the bytes up (``x_hat``: sum of ``in x K/8`` over the
  linears), down (``y_hat``: sum of ``out x K/8``) and of masks consumed (``W r`` rows, the same size
  as ``y_hat``), and the round trips per block (4 x layers);
* ``float``: the gateway plaintext work of one whole-block prefill over a stub linear backend (no
  masking, no service): the full pass, the pass with the LM head replaced (``layers_ms``, scaled to
  the full layer count), and the instrumented split (attention, norms, RoPE, KV append, glue);
* ``blocks``: the same prompt in ``B`` sequential chunks (``--blocks``, e.g. 1,2,4): float time, and
  whether the last row's logits stay bit-identical to the whole-block pass (chunked attention is
  not guaranteed to be; WP-A5's B-way prefill blocks must prove it);
* ``masking`` (``--masking``): gateway mask/unmask ms per prompt token from
  ``profile_gateway.py --rows T --null-service`` (builds a synthetic checkpoint; slower).

Usage::

    PYTHONPATH=src python benchmarks/profile_prefill.py [--preset 7b] [--layers 2] [--tokens 40,200]
        [--blocks 1,2,4] [--steps 3] [--masking] [--json out.json]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent


def load_bench(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(f"pgw_bench_{name}", HERE / f"{name}.py")
    if spec is None or spec.loader is None:  # pragma: no cover
        raise ImportError(name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _median_ms(fn: Any, reps: int) -> float:
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return 1000 * statistics.median(ts)


def volume(arch: Any, ring_k: int, tokens: int) -> dict[str, Any]:
    """Wire and mask bytes of a masked prefill (all linears remote, one block)."""
    from pgw.core.types import LinearKind
    from pgw.engine.pack import linear_split_sizes

    ins = {LinearKind.QKV: arch.hidden_size, LinearKind.O: arch.num_heads * arch.head_dim,
           LinearKind.GATE_UP: arch.hidden_size, LinearKind.DOWN: arch.intermediate_size}
    nb = ring_k // 8
    up = sum(n for n in ins.values()) * nb * arch.num_layers
    down = sum(sum(linear_split_sizes(arch, k)) for k in ins) * nb * arch.num_layers
    return {"up_bytes_per_token": up, "down_bytes_per_token": down, "mask_bytes_per_token": down,
            "wire_MB_per_prompt": round((up + down) * tokens / 1e6, 2),
            "mask_MB_per_prompt": round(down * tokens / 1e6, 2),
            "round_trips_per_block": 4 * arch.num_layers}


def float_profile(args: argparse.Namespace, tokens_list: list[int],
                  blocks_list: list[int]) -> tuple[dict[int, Any], Any]:
    pt = load_bench("profile_token")
    import torch

    from pgw.engine import model as model_mod
    from pgw.engine.kv_cache import KVCache
    from pgw.engine.model import SequenceInput, TransformerModel
    from pgw.engine.rope import RotaryEmbedding

    if args.threads:
        torch.set_num_threads(args.threads)
    preset = dict(pt.PRESETS[args.preset])
    full_layers = preset["num_layers"]
    if args.vocab:
        preset["vocab_size"] = args.vocab
    pack = pt.SynthPack(preset, args.layers or None)
    a = pack.arch
    scale = full_layers / a.num_layers
    model = TransformerModel(pack, pt.make_backends(a, 1)[0])
    out: dict[int, Any] = {}
    for tokens in tokens_list:
        prompt = [(17 * i + 3) % a.vocab_size for i in range(tokens)]
        cache = model.new_cache(tokens + 8)

        def whole(p: list[int] = prompt, c: KVCache = cache) -> torch.Tensor:
            c.truncate(0)
            return model.forward_batch([SequenceInput(p, c)])[0]

        def chunked(b: int, p: list[int] = prompt, c: KVCache = cache) -> torch.Tensor:
            c.truncate(0)
            step = -(-len(p) // b)
            lg = None
            for i in range(0, len(p), step):
                lg = model.forward_batch([SequenceInput(p[i : i + step], c)])[0]
            assert lg is not None
            return lg

        whole()
        total = _median_ms(whole, args.steps)
        ref = whole().clone()
        orig = model_mod.lm_head_logits
        model_mod.lm_head_logits = lambda h, w: torch.zeros((h.shape[0], a.vocab_size))
        try:
            layers = _median_ms(whole, args.steps)
        finally:
            model_mod.lm_head_logits = orig
        timers = pt.Timers()
        timers.wrap(model_mod, "_attend", "attention")
        timers.wrap(model_mod, "rms_norm", "norm")
        timers.wrap(model_mod, "apply_rope", "rope")
        timers.wrap(RotaryEmbedding, "cos_sin", "rope_tables")
        timers.wrap(KVCache, "append", "kv_append")
        try:
            t = time.perf_counter()
            whole()
            wall = (time.perf_counter() - t) * 1e3
            split = {k: round(v * 1e3, 3) for k, v in timers.total.items()}
        finally:
            timers.restore()
        split["glue_and_head"] = round(wall - sum(split.values()), 3)
        res: dict[str, Any] = {
            "pass_ms_measured_layers": round(total, 3),
            "layers_ms": round(layers * scale, 3),
            "head_ms": round(total - layers, 3),
            "layers_ms_per_token": round(layers * scale / tokens, 4),
            "split_ms_measured_layers": split,
            "blocks": {},
        }
        for b in blocks_list:
            if b <= 1 or b > tokens:
                continue
            lg = chunked(b)
            res["blocks"][str(b)] = {"layers_and_head_ms": round(_median_ms(lambda b=b: chunked(b), args.steps), 3),
                                     "last_logits_bit_identical": bool(torch.equal(lg, ref))}
        out[tokens] = res
        print(json.dumps({"tokens": tokens, **{k: v for k, v in res.items() if k != "split_ms_measured_layers"}}),
              flush=True)
    model.close()
    return out, a


def masking(args: argparse.Namespace, tokens: int) -> float:
    """Gateway masking ms for one whole-block prefill of ``tokens`` rows (full model)."""
    js = Path(args.workdir) / f"mask_t{tokens}.json"
    cmd = [sys.executable, str(HERE / "profile_gateway.py"), "--preset", args.preset, "--layers", "1", "--steps", "3",
           "--rows", str(tokens), "--sessions", "1", "--null-service", "--ring", args.ring,
           "--workdir", str(Path(args.workdir) / "pg"), "--json", str(js)]
    subprocess.run(cmd, check=True, capture_output=True, timeout=7200)  # noqa: S603
    return float(json.loads(js.read_text())["gateway_s_per_step_full_model"]) * 1e3


def run(args: argparse.Namespace) -> dict[str, Any]:
    from pgw.core.ring import RingParams

    tokens_list = [int(t) for t in str(args.tokens).split(",") if t]
    blocks_list = [int(b) for b in str(args.blocks).split(",") if b]
    ring = RingParams.from_name(args.ring)
    Path(args.workdir).mkdir(parents=True, exist_ok=True)
    flt, arch = float_profile(args, tokens_list, blocks_list)
    pt = load_bench("profile_token")
    full = dict(pt.PRESETS[args.preset])
    from pgw.modelpack.manifest import ArchConfig

    full_arch = ArchConfig(
        arch="qwen2", hidden_size=full["hidden_size"], intermediate_size=full["intermediate_size"],
        num_layers=full["num_layers"], num_heads=full["num_heads"], num_kv_heads=full["num_kv_heads"],
        head_dim=full["hidden_size"] // full["num_heads"], vocab_size=full["vocab_size"],
        rms_norm_eps=full["rms_norm_eps"], rope_theta=full["rope_theta"], max_position_embeddings=32768,
        tie_word_embeddings=False)
    per: dict[str, Any] = {}
    for t in tokens_list:
        row: dict[str, Any] = {"volume": volume(full_arch, ring.k, t), "float": flt[t]}
        if args.masking:
            m = masking(args, t)
            row["masking_ms"] = round(m, 3)
            row["masking_ms_per_token"] = round(m / t, 4)
        per[str(t)] = row
    return {"preset": args.preset, "layers_measured": arch.num_layers, "ring": ring.name,
            "omp_wait_policy": os.environ.get("OMP_WAIT_POLICY"), "prompts": per}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--preset", default="7b", choices=["7b", "0.5b", "8b"])
    ap.add_argument("--layers", type=int, default=2, help="layers measured (scaled to the preset's; 0 = all)")
    ap.add_argument("--vocab", type=int, default=0, help="override the vocabulary (0 = the preset's; smoke tests)")
    ap.add_argument("--tokens", default="40,200", help="prompt lengths, comma list")
    ap.add_argument("--blocks", default="1,2,4", help="sequential prefill chunks to compare, comma list")
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--ring", default="K32A12C27")
    ap.add_argument("--masking", action="store_true", help="add gateway masking (profile_gateway subprocess)")
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    if args.workdir is None:
        args.workdir = tempfile.mkdtemp(prefix="pgw-prefill-")
    res = run(args)
    text = json.dumps(res, indent=1)
    print(text)
    if args.json:
        Path(args.json).write_text(text + "\n")


if __name__ == "__main__":
    main()
