"""Client sizing for the library (``pgw.OpenAI``) at N concurrent streams: cores, RAM and disk per model.

Arithmetic on the model shapes (exact) and on per-unit costs measured on this repository's CPU boxes (STATUS.md,
2026-11-26 sections, and ``benchmarks/sdk_concurrency_bench.py``):

* mask slot: one decode token or one uncached prompt token of one stream; its bytes are ``4 x sum(out_features)``
  (K32 ring, every linear incl. the masked LM head);
* client CPU per mask slot (pks-v3 factory, 1 thread): 7.2 ms at 0.5B (measured), 24.9 / 80.0 / 120.8 ms at
  7B / 32B / 72B (measured per part at 0.5B, extrapolated at real shapes);
* gateway CPU per decode step: ``4L + 1`` trips of 0.10 / 0.115 / 0.12 / 0.128 ms of native work with one row at
  0.5B / 7B / 32B / 72B (STATUS), split into a part shared by every stream batched into the step (0.069 ms per
  trip) and a per-row part (the rest; the measured marginal / fixed ratio of a batched row, about 0.45 at 0.5B),
  plus 1.3 ms of sampling and hand-off per row; one plaintext thread per process does it (the GIL): ~0.85 of a
  core is the useful ceiling of one gateway process;
* KV cache: float32 keys and values of every layer, grown geometrically (x1.5 on average) up to the context.

    python benchmarks/sdk_scale_sizing.py                     # the docs table (N = 1, 4, 16, 64)
    python benchmarks/sdk_scale_sizing.py --model 7b --streams 32 --tok-s 15 --prompt 2000 --json

Every output is an estimate for planning; measure with ``sdk_concurrency_bench.py`` on the target host.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass

#: Qwen2.5 shapes: layers, hidden, intermediate, kv heads, head dim, vocab
SHAPES: dict[str, tuple[int, int, int, int, int, int]] = {
    "0.5b": (24, 896, 4864, 2, 64, 151936),
    "7b": (28, 3584, 18944, 4, 128, 152064),
    "32b": (64, 5120, 27648, 8, 128, 152064),
    "72b": (80, 8192, 29568, 8, 128, 152064),
}
#: client CPU seconds per mask slot, pks-v3, one thread (STATUS 2026-11-26)
MASK_CPU_S = {"0.5b": 0.0072, "7b": 0.0249, "32b": 0.0800, "72b": 0.1208}
#: decode tok/s of one stream at RTT 0.2 ms on an H100 provider (round_trip_model.py --sdk; 0.5B: same model)
STREAM_TOK_S = {"0.5b": 30.0, "7b": 17.0, "32b": 6.7, "72b": 5.1}
#: gateway native work per trip (1 row), seconds; sampling + hand-off per token (STATUS 2026-11-26)
TRIP_FIXED_S = 0.000069
SAMPLE_S = 0.0013
#: gateway native work per trip with one row (STATUS 2026-11-26; 32B interpolated)
TRIP_ONE_ROW_S = {"0.5b": 0.000100, "7b": 0.000115, "32b": 0.000122, "72b": 0.000128}
#: usable fraction of the gateway's plaintext thread (the rest: event loop, lane I/O on the same GIL)
GW_CORE = 0.85


@dataclass
class Sizing:
    model: str
    streams: int
    tok_s_per_stream: float
    slot_mb: float
    kv_kb_per_token: float
    mask_slots_per_s: float
    factory_cores: float
    gateway_cpu_per_step_ms: float
    gateway_processes: int
    gateway_cores: float
    cores_total: float
    ram_gb: float
    stock_disk_gb: float
    stock_overcommit_gb: float
    cache_disk_gb: float
    mask_store_gb_setting: float


def shape_numbers(model: str) -> tuple[int, float, int]:
    """(sum of out_features incl. the head, KV bytes per token, layers)."""
    layers, h, inter, kvh, hd, vocab = SHAPES[model]
    per_layer = (h + 2 * kvh * hd) + h + 2 * inter + h
    return per_layer * layers + vocab, 2.0 * layers * kvh * hd * 4, layers


def size(model: str, streams: int, *, tok_s: float | None = None, prompt: int = 500, max_tokens: int = 512,
         gen: int = 256, cached_frac: float = 0.0, burst_s: float = 30.0) -> Sizing:
    """Cores, RAM, disk for ``streams`` concurrent streams of ``tok_s`` tokens/s each.

    ``prompt``: prompt tokens per request; ``cached_frac``: share of them reused from the conversation prefix cache
    (no mask, no prompt pass); ``gen``: generated tokens per request (average); ``max_tokens``: the requests'
    limit (the admission entitlement); ``burst_s``: seconds of full-rate demand the stock absorbs with the factory
    stopped (or behind).
    """
    out, kv_tok, layers = shape_numbers(model)
    slot = 4.0 * out
    tps = STREAM_TOK_S[model] if tok_s is None else tok_s
    # masks: one slot per generated token and per uncached prompt token
    slots_per_tok = 1.0 + prompt * (1.0 - cached_frac) / gen
    slots_s = streams * tps * slots_per_tok
    factory = slots_s * MASK_CPU_S[model]
    # gateway: a step serves every stream once; fixed part + per row part (scaled by width from 0.5B)
    trips = 4 * layers + 1
    fixed = trips * TRIP_FIXED_S
    per_row = trips * (TRIP_ONE_ROW_S[model] - TRIP_FIXED_S) + SAMPLE_S
    step = fixed + streams * per_row
    # processes: the fewest whose plaintext thread (one per process, the GIL) stays under GW_CORE at the target rate;
    # each process pays the fixed part of every step for its own share of the streams
    procs = 1
    while procs < streams and tps * (fixed + math.ceil(streams / procs) * per_row) > GW_CORE:
        procs += 1
    gw_load = procs * tps * (fixed + streams / procs * per_row)
    gw_cores = gw_load + 0.3 * procs  # event loop, lane I/O, decrypt-ahead per process
    # RAM: per process the float32 LM head and the runtime, tokenizer, embeddings (~1 GB); per stream
    # the KV cache at prompt + max_tokens, decrypted masks (about one slot) and buffers
    _, h, _, _, _, vocab = SHAPES[model]
    head_gb = vocab * h * 4 / 1e9  # the float32 LM head (STATUS: 2.2 / 3.1 / 5 GB at 7B / 32B / 72B)
    ram = procs * (head_gb + 1.0) + streams * ((prompt + max_tokens) * kv_tok * 1.5 + 2 * slot) / 1e9
    # disk: the admission entitlement of every live stream plus burst_s of demand; plus the cache (STATUS)
    entitled = streams * (prompt * (1.0 - cached_frac) + max_tokens + 24)
    stock = (entitled + burst_s * slots_s) * slot / 1e9
    # mask_admission="overcommit": each live stream is promised its prompt pass and one chunk only
    stock_oc = (streams * (prompt * (1.0 - cached_frac) + 16) + burst_s * slots_s) * slot / 1e9
    cache = {"0.5b": 2.2, "7b": 13.0, "32b": 45.0, "72b": 91.0}[model]
    return Sizing(model, streams, tps, slot / 1e6, kv_tok / 1e3, slots_s, factory, step * 1e3, procs, gw_cores,
                  factory + gw_cores, ram, stock, stock_oc, cache, math.ceil(stock))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", choices=list(SHAPES), action="append")
    ap.add_argument("--streams", type=int, action="append")
    ap.add_argument("--tok-s", type=float, default=None)
    ap.add_argument("--prompt", type=int, default=500)
    ap.add_argument("--gen", type=int, default=256)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--cached-frac", type=float, default=0.0)
    ap.add_argument("--burst-s", type=float, default=30.0)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    rows = [size(m, n, tok_s=a.tok_s, prompt=a.prompt, max_tokens=a.max_tokens, gen=a.gen,
                 cached_frac=a.cached_frac, burst_s=a.burst_s)
            for m in (a.model or ["0.5b", "7b", "32b", "72b"]) for n in (a.streams or [1, 4, 16, 64])]
    if a.json:
        print(json.dumps([r.__dict__ for r in rows], indent=2))
        return
    print("| model | N | tok/s each | mask slots/s | factory cores | gateway procs | gateway cores | cores total "
          "| RAM GB | mask stock GB (entitled / overcommit) | cache GB |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r.model} | {r.streams} | {r.tok_s_per_stream:g} | {r.mask_slots_per_s:,.0f} | "
              f"{r.factory_cores:,.1f} | {r.gateway_processes} | {r.gateway_cores:.1f} | {r.cores_total:,.1f} | "
              f"{r.ram_gb:,.1f} | {r.stock_disk_gb:,.0f} / {r.stock_overcommit_gb:,.0f} | {r.cache_disk_gb:g} |")


if __name__ == "__main__":
    main()
