#!/usr/bin/env python3
"""Numbers of ``consumer-crypto-alternatives.md``: FHE, HE-linear, FSS and hybrid cost models for consumer chat.

Run from ``pgw/``: ``python research/reviews/consumer_crypto_numbers.py [--json out.json]``.

Standard library only. Every number is either (a) a published anchor (constant named ``A_*``, source in the
note's citations) or (b) an estimate derived from the anchors by the scaling rule written next to it. The pgw
reference numbers (trips, bytes per row) are copied from ``consumer-round-trips.md`` / ``consumer-bandwidth.md``.
Nothing here is measured; error bars are the ranges the functions return.
"""

from __future__ import annotations

import argparse
import json
import math
from typing import Any

# ---------------------------------------------------------------------------------------------------------------
# Shapes: (layers, d_model, intermediate, q_heads*head_dim, kv_heads*head_dim, vocab, params)
SHAPES = {
    "llama3-8b": (32, 4096, 14336, 4096, 1024, 128256, 8.03e9),
    "qwen2.5-7b": (28, 3584, 18944, 3584, 512, 152064, 7.62e9),
    "qwen2.5-32b": (64, 5120, 27648, 5120, 1024, 152064, 32.8e9),
}

# ---------------------------------------------------------------------------------------------------------------
# Published anchors (see the note's section 10 for sources).
#: Park et al. 2026 (arXiv 2601.18511v2), Llama-3-8B, CKKS/HEaaN2, 128 private tokens, unmodified weights:
A_DECODE_S_4B200 = 12.0  # s per output token on 4x B200
A_DECODE_S_8RP6000 = 18.0  # s per output token on 8x RTX PRO 6000
A_PREFILL_S_4B200 = 26.0  # s for the 128-token encrypted prefill on 4x B200
#: Table VI of the same paper, decode, 1 GPU, ms per layer: linear projections / ciphertext attention / rest.
A_LAYER_MS = {"linear": 135 + 80 + 133 + 71, "attn_cc": 154 + 49 + 155, "rest": 9 + 2 + 20 + 4}
A_LAYER_TOTAL_MS_1GPU = 980.0  # stated total per layer (the rows sum to ~812: bootstrapping not itemised)
#: Cerium (SOSP'26, arXiv 2512.11269): CKKS bootstrapping 7.5 ms on a datacentre GPU, N = 2^16, logPQ < 1782,
#: refresh to 18 levels; Llama3-8B 128-token prefill = 5890 bootstraps, 43 s TTFT (multi-GPU).
A_BOOT_MS = 7.5
A_LOGPQ = 1782
A_N = 1 << 16
#: SIGMA (PoPETs'24), Llama2-7B, one 128-token forward pass (= one next token without KV cache), GPU, LAN:
A_SIGMA = {"online_s": 23.09, "online_gb": 12.07, "keys_gb": 255.41, "wan_min": 5.72, "wan_mbit": 305, "wan_ms": 60}
#: TFHE-rs GPU (vendor figure, 2026): ~47,900 PBS/s on one H100 (small-precision LUTs).
A_TFHE_PBS_S_H100 = 47_900

# ---------------------------------------------------------------------------------------------------------------
# Assumptions (stated as such in the note).
GPU_HOUR_USD = {"B200": (3.0, 6.0), "H100": (2.0, 3.5)}  # cloud rental range, late 2026 (assumption)
#: plaintext serving: aggregate decode tok/s per GPU at large batch (assumption, order of magnitude; vLLM-class)
PLAIN_TOKS_PER_GPU = {"qwen2.5-7b": (5_000, 20_000), "qwen2.5-32b": (1_500, 6_000)}

# pgw references (consumer-round-trips.md section 0 / 9, consumer-bandwidth.md section 0)
PGW = {
    "qwen2.5-7b": {"trips_today": 113, "trips_fused": 56, "trips_parallel": 29, "row_mb_today": 23.1,
                   "row_mb_best": 6.5, "online_mb_today": 9.51},
    "qwen2.5-32b": {"trips_today": 257, "trips_fused": 128, "trips_parallel": 65, "row_mb_today": 72.9,
                    "row_mb_best": 21.8, "online_mb_today": 30.25},
}


def _layer_units(name: str, ctx: int) -> dict[str, float]:
    """Relative FHE work per layer: linear part ~ weight count, ciphertext attention ~ q width x context,
    rest (norms, activations, bootstrapping of activations) ~ activation count (d + I)."""
    L, d, inter, qw, kvw, vocab, _ = SHAPES[name]
    return {"linear": d * qw + 2 * d * kvw + qw * d + 3 * d * inter, "attn_cc": qw * ctx, "rest": d + inter}


def fhe_decode(name: str, ctx: int = 128) -> dict[str, Any]:
    """Seconds per output token (single stream) and GPU-seconds per token, scaled from the Llama-3-8B anchor.

    The anchor's per-layer split (Table VI) is scaled component by component; the bootstrap/overhead remainder
    (980 - 812 ms) scales with the 'rest' units. Range: x0.7 (better packing at wider shapes) .. x1.5 (worse
    multi-GPU scaling, memory spill) around the central estimate. Context > 128 scales the ciphertext attention
    linearly (an upper-ish estimate; slot packing improves with more cached tokens)."""
    ref = _layer_units("llama3-8b", 128)
    cur = _layer_units(name, ctx)
    L = SHAPES[name][0]
    remainder = A_LAYER_TOTAL_MS_1GPU - sum(A_LAYER_MS.values())
    ms = (A_LAYER_MS["linear"] * cur["linear"] / ref["linear"]
          + A_LAYER_MS["attn_cc"] * cur["attn_cc"] / ref["attn_cc"]
          + (A_LAYER_MS["rest"] + remainder) * cur["rest"] / ref["rest"])
    ratio = ms * L / (A_LAYER_TOTAL_MS_1GPU * 32)  # vs the Llama-3-8B anchor at the same context (128)
    s_4b200 = A_DECODE_S_4B200 * ratio
    gpu_s = 4 * s_4b200  # B200-seconds per token (4 GPUs busy for the token's latency)
    return {"ratio_vs_llama8b": ratio, "s_per_token_4xB200": (0.7 * s_4b200, s_4b200, 1.5 * s_4b200),
            "B200_s_per_token": (0.7 * gpu_s, gpu_s, 1.5 * gpu_s)}


def fhe_latency_floor(name: str, seq_boots_per_layer: tuple[int, ...] = (1, 2, 3, 4),
                      boot_ms: tuple[float, ...] = (A_BOOT_MS, 2.0, 0.5)) -> dict[str, Any]:
    """Lower bound on single-stream latency from the chain of sequential bootstraps alone (unlimited GPUs).

    1 bootstrap per layer = an FHE-co-designed model (low-degree activations, power-softmax, no rsqrt);
    3-4 = an unmodified transformer (softmax ~8-15 levels, rsqrt, SiLU deg 31, ~6 matmul levels, 18 usable
    levels per bootstrap). boot_ms: 7.5 ms = best published GPU (Cerium); 2.0 / 0.5 ms = hypothetical future
    GPU / ASIC-class latency (not published for N = 2^16 at full precision as far as found)."""
    L = SHAPES[name][0]
    out = {}
    for b in seq_boots_per_layer:
        for t in boot_ms:
            s = L * b * t / 1000.0
            out[f"{b}boot/layer@{t}ms"] = {"s_per_token": s, "max_tok_s": 1.0 / s}
    return out


def fhe_cost(name: str) -> dict[str, Any]:
    gs = fhe_decode(name)["B200_s_per_token"]
    lo, hi = GPU_HOUR_USD["B200"]
    usd_per_1k = (gs[0] * lo / 3.6, gs[1] * (lo + hi) / 2 / 3.6, gs[2] * hi / 3.6)  # $ per 1000 tokens
    plain_lo, plain_hi = PLAIN_TOKS_PER_GPU[name]
    return {"usd_per_1k_tokens": usd_per_1k,
            "B200s_for_20_tok_s": tuple(20 * g for g in gs),
            "users_per_B200_at_0.1_tok_s": tuple(1.0 / (0.1 * g) for g in gs),
            "overhead_vs_plaintext_serving": (gs[0] * plain_lo, gs[2] * plain_hi)}


def fhe_bytes(name: str, prompt: int = 500) -> dict[str, Any]:
    """Client <-> provider bytes for non-interactive FHE (estimates; no paper reports them for LLM decode).

    Transport ciphertexts are mod-switched to log q = 50 (enough for ~12-bit results plus noise) and
    coefficient-packed in the smallest power-of-two ring holding the vector; uploads are seeded (b only)."""
    L, d, inter, qw, kvw, vocab, _ = SHAPES[name]
    logq = 50

    def ct_bytes(values: int, seeded: bool) -> float:
        n = 1 << max(11, math.ceil(math.log2(values)))
        return (1 if seeded else 2) * n * logq / 8

    up_tok = ct_bytes(d, True)  # the next token's embedding (device does the lookup)
    down_hidden = ct_bytes(d, False)  # final hidden state; LM head on the device
    down_logits = ct_bytes(vocab, False)  # or the logits
    prompt_up = math.ceil(prompt * d / A_N) * A_N * logq / 8  # packed prompt embeddings, seeded
    # evaluation keys: one switching key = dnum * 2 * N * logPQ bits (dnum ~ 3), seeded halves it; 50-200 keys
    one_key = 3 * 2 * A_N * A_LOGPQ / 8
    keys = (50 * one_key / 2, 200 * one_key)
    kv_ct_per_token = L * 2 * kvw * 2 * 2 * 100 / 8  # K,V per token at a 2-limb (~100-bit) storage level, 2 polys
    kv_plain_bf16 = L * 2 * kvw * 2
    return {"up_per_token_B": up_tok, "down_hidden_B": down_hidden, "down_logits_B": down_logits,
            "prompt_up_B": prompt_up, "eval_keys_B": keys,
            "server_kv_per_token_B": kv_ct_per_token, "kv_expansion_vs_bf16": kv_ct_per_token / kv_plain_bf16}


def client_aided_refresh(name: str, layers_per_refresh: tuple[int, ...] = (1, 2),
                         logq_top: tuple[int, ...] = (1000, 1500)) -> dict[str, Any]:
    """Client-aided bootstrapping: every k layers the client decrypts and re-encrypts (or uploads fresh
    Enc_top(s) masks offline). A fresh top-level ciphertext is uniform mod Q: >= N log Q bits (seeded), and one
    decode token's residual (d values) uses d / (N/2) of its slots. Amortising one fresh ciphertext over up to
    (N/2)/d refreshes (slot rotation) is the optimistic case."""
    L, d, *_ = SHAPES[name]
    out = {}
    for k in layers_per_refresh:
        for lq in logq_top:
            per_ct = A_N * lq / 8
            amort = max(1, (A_N // 2) // d)
            refreshes = math.ceil(L / k)
            out[f"k={k},logQ={lq}"] = {"trips": refreshes, "MB_per_token_no_amort": refreshes * per_ct / 1e6,
                                       "MB_per_token_amortised": refreshes * per_ct / amort / 1e6}
    return out


def he_linear_bytes(name: str) -> dict[str, Any]:
    """Cheetah/BOLT-style HE for the linears with the client evaluating non-linears in the clear (public weights).

    Per input value: seeded coefficient ciphertext at log q ~ 109 (N = 4096) -> ~13.6 B; per output value after a
    mod switch to ~2^40 and output packing ~10 B (estimate). pgw today: 4 B online + pks-v3 7 B up / 4.5 B down per
    value; pgw floor with device masks ~3.5 B per value each way (consumer-bandwidth.md)."""
    L, d, inter, qw, kvw, vocab, _ = SHAPES[name]
    n_in = L * (3 * d + inter) + d
    n_out = L * (qw + 2 * kvw + 2 * d + 2 * inter) + vocab
    he = n_in * 13.6 + n_out * 10.0
    pgw_online = 4 * (n_in + n_out)
    pgw_with_pks = pgw_online + 7 * n_in + 4.5 * n_out
    return {"values_in": n_in, "values_out": n_out, "he_linear_MB_per_row": he / 1e6,
            "pgw_online_MB": pgw_online / 1e6, "pgw_online_plus_pksv3_MB": pgw_with_pks / 1e6}


def sigma_per_token(name: str) -> dict[str, Any]:
    """SIGMA scaled to decode: the published run is one 128-token forward pass; with a KV cache a decode token
    costs ~1/128 of the sequence-linear parts (optimistic: treats everything as sequence-linear), scaled by params."""
    p = SHAPES[name][6] / 6.74e9
    keys_tok = A_SIGMA["keys_gb"] / 128 * p
    online_tok = A_SIGMA["online_gb"] / 128 * p
    return {"fss_keys_GB_per_token": keys_tok, "online_GB_per_token": online_tok,
            "prompt500_keys_GB": keys_tok * 500, "note": "needs a trusted non-colluding dealer, or the client as dealer uploading the keys"}


def tfhe_gates(name: str) -> dict[str, Any]:
    """TFHE lower bound from the activations alone: one PBS per SiLU input, no softmax / norms / linears."""
    L, d, inter, *_ = SHAPES[name]
    pbs = L * inter
    return {"pbs_per_token_silu_only": pbs, "H100_s_per_token_silu_only": pbs / A_TFHE_PBS_S_H100}


def local_inference(name: str) -> dict[str, Any]:
    """Memory-bound decode on the user's own hardware: tok/s ~ 0.7 x bandwidth / weight bytes (estimate)."""
    params = SHAPES[name][6]
    devices = {"phone (77 GB/s)": 77, "thin laptop (120 GB/s)": 120, "M4 Pro (273 GB/s)": 273,
               "M4 Max (546 GB/s)": 546, "RTX 4090 (1008 GB/s)": 1008}
    out = {}
    for dev, bw in devices.items():
        out[dev] = {f"{bits}-bit": 0.7 * bw * 1e9 / (params * bits / 8) for bits in (16, 8, 4)}
    return out


def pgw_reference(name: str) -> dict[str, Any]:
    r = PGW[name]
    tok = {}
    for rtt in (5, 20, 50):
        tok[f"rtt{rtt}"] = {k: 1000.0 / (r[k] * rtt) for k in ("trips_today", "trips_fused", "trips_parallel")}
    gbit_20 = {k: r[k] * 8 * 20 * 1.3 / 1000 for k in ("row_mb_today", "row_mb_best")}  # 1.3 rows/token
    return {"tok_s_trip_bound": tok, "gbit_s_at_20_tok_s": gbit_20,
            "prompt500_GB_best": r["row_mb_best"] * 500 / 1000}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    args = ap.parse_args()
    res: dict[str, Any] = {}
    for m in ("qwen2.5-7b", "qwen2.5-32b"):
        res[m] = {
            "fhe_decode_ctx128": fhe_decode(m, 128),
            "fhe_decode_ctx1k": fhe_decode(m, 1024),
            "fhe_decode_ctx4k": fhe_decode(m, 4096),
            "fhe_latency_floor": fhe_latency_floor(m),
            "fhe_cost": fhe_cost(m),
            "fhe_bytes": fhe_bytes(m),
            "client_aided_refresh": client_aided_refresh(m),
            "he_linear": he_linear_bytes(m),
            "sigma": sigma_per_token(m),
            "tfhe": tfhe_gates(m),
            "local": local_inference(m),
            "pgw": pgw_reference(m),
        }
    if args.json:
        with open(args.json, "w") as f:
            json.dump(res, f, indent=1)

    def fmt(x: Any) -> str:
        if isinstance(x, float):
            return f"{x:.3g}"
        if isinstance(x, tuple):
            return "(" + ", ".join(fmt(v) for v in x) + ")"
        return str(x)

    def walk(prefix: str, obj: Any) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                walk(f"{prefix}.{k}" if prefix else k, v)
        else:
            print(f"{prefix} = {fmt(obj)}")

    walk("", res)


if __name__ == "__main__":
    main()
