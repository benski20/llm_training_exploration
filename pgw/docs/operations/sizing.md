# Sizing guide

How much bandwidth, mask storage, mask-factory capacity and compute a PGW deployment needs. Every
number is labelled:

* **measured**: observed in a benchmark run (source in [measurements.json](../measurements.json));
* **analytic**: exact arithmetic on the model shape and ring;
* **derived**: arithmetic on a measured number;
* **estimate**: rests on an assumption. Validate it in your environment before committing capacity.

Calculator: `python docs/tools/pgw_sizing.py --model llama-3-8b --ring K48A20 --tokens-per-day 1000000`.
It needs only the Python standard library, and `--json` gives machine-readable output. The default ring
is the product default, K32A12C27. `--encrypted` adds the encrypted mask-supply figures (traffic, hint,
link-bound rate; see [encrypted-masks.md](encrypted-masks.md)) and `--spec-row-budget` the worst-case
mask use with speculative decoding:
`python docs/tools/pgw_sizing.py --model qwen2.5-7b --ring K32A12C27 --encrypted --spec-row-budget --tokens-per-day 1000000`.

## 1. Per-token formulas (analytic)

For a model with `L` layers and the four masked groups per layer (`qkv`, `o`, `gate_up`, `down`) with
output rows `m_g` and input features `n_g`, and ring `K` (bytes per element `K/8`):

| quantity | formula | Llama-3-8B (`Σm` = 43008, `Σn` = 26624, `L` = 32) |
|---|---|---|
| round trips per token | `4·L` | 128 round trips |
| upload (`x̂`) per token | `L·Σn·K/8` | 5.1 MB at K48A20, 4.26 MB at K40A12 |
| download (`ŷ`) per token | `L·Σm·K/8` | 8.3 MB at K48A20, 6.88 MB at K40A12 |
| wire total per token | sum of both | 13.4 MB at K48A20, 11.1 MB at K40A12 |
| mask storage per slot (token) | `L·Σm·K/8` (only `W r` is stored; `r` is regenerated from the seed) | 8.3 MB at K48A20, 6.9 MB at K40A12 |
| per-segment overhead | `32 + c·L·Σn·K/8` bytes (seed, `uᵀW`) plus 16 bytes of tag per chunk | under 0.1% at 4096 slots/segment |
| gateway online work | regenerate `L·Σn` mask elements, subtract `L·Σm` | about 2.2 M ring elements per token |

Prompt tokens cost the same bytes and masks as generated tokens (one slot each), but share round
trips: a whole prompt goes through each group in one round trip.

Wire framing (msgpack, HTTP/2, TLS) adds well under 1% because payloads are a few large binary
blobs. Masked data is uniform, so **compression gains nothing**.

### Reference table

| model | ring | up MB/token | down MB/token | total MB/token | masks MB/slot | slots per TB | rounds |
|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | K32A12C27 | 0.73 | 1.22 | 1.94 | 1.22 | 822k | 96 |
| Qwen2.5-7B | K32A12C27 | 3.33 | 5.56 | 8.89 | 5.56 MB | 180k | 112 |
| Llama-3-8B | K32A12C27 | 3.41 | 5.51 | 8.91 | 5.51 | 182k | 128 |
| Qwen2.5-0.5B | K40A12 | 0.91 | 1.52 | 2.43 | 1.52 | 657k | 96 |
| Qwen2.5-7B | K40A12 | 4.16 | 6.95 | 11.1 | 6.95 | 144k | 112 |
| Llama-3-8B | K32A8 | 3.41 | 5.51 | 8.91 | 5.51 | 182k | 128 |
| Llama-3-8B | K40A12 | 4.26 | 6.88 | 11.1 | 6.9 MB | 145k | 128 |
| Llama-3-8B | K48A20 | 5.11 | 8.26 | 13.4 MB | 8.3 MB | 121k | 128 |
| Llama-3-70B | K40A12 | 21.3 | 33.6 | 54.9 | 33.6 | 29.8k | 320 |
| Llama-3-70B | K48A20 | 25.6 | 40.3 | 65.9 | 40.3 MB | 24.8k | 320 round trips |

These are analytic values, confirmed for 8B and 70B by the prototype run (5.1 MB up and 8.3 MB down
per 8B token at K=48).

**K32A12C27 is the default ring** (4 bytes per element; the output cap lets it decode any width). The
K40 and K48 rows describe the older presets. Sizing a K32A12C27 deployment from a K40 row
**over-sizes** the store and the links by 25% (5 instead of 4 bytes per element); it never
under-sizes them. Qwen2.5-7B at the default ring needs 5.56 MB of masks per slot.

## 2. Bandwidth

```
mean bandwidth [bit/s] = tokens_per_second × wire_total_bytes × 8
```

Example: 8B at K48A20 moves about 13.4 MB per token. 100 tokens/s (prompt and generated tokens
together) is about 1.34 GB/s, or **10.7 Gbit/s** both directions together (*analytic*).
Plan link capacity for peak, not mean. Prefill bursts send whole prompts at once.

Traffic is asymmetric: the download (`ŷ`) is larger than the upload whenever `Σm > Σn`, which is
true for all supported architectures (the `gate_up` group dominates).

## 3. Latency

```
latency per generated token ≈ 4·L × (RTT + service time per round) + gateway time
```

Round-trip cost by placement tier (*estimates*; the round trips of one token run one after another, so
each adds `rounds × RTT`; 112 round trips for Qwen2.5-7B, 128 for Llama-3-8B):

| tier | RTT (estimate) | added per token, Qwen2.5-7B | basis |
|---|---|---|---|
| same host or same AZ, co-located | 0.1-0.3 ms | +0.01 to +0.035 s | estimate |
| same region, other AZ (cross-AZ) | 0.5-1 ms | +0.05 to +0.11 s | estimate |
| same region over PrivateLink / PSC | ~1 ms | about +0.11 s | estimate |
| internet, same continent | 10-30 ms | +1.1 to +3.4 s | estimate |

Keep the gateway in the zone of the service endpoint it calls (zonal endpoint DNS; see
`deploy/marketplace/aws/README.md`). The service's own time per round trip comes on top; on a T4 the
measured single-row call is about 1.4 ms (`STATUS.md`, run 25).

| component | value | basis |
|---|---|---|
| network floor, same region over private link (RTT ~1 ms) | 0.13 s/token for 8B | estimate (128 round trips × RTT) |
| network floor over the internet (RTT 10-30 ms) | 1.3-4 s/token for 8B | estimate |
| service compute, 8B, T4, batch 1 | 0.45 s/token | measured |
| service compute, 8B, T4, batch 32 | 0.014 s/token amortized | measured |
| plain fp16 8B on the same T4, batch 1 (reference) | 0.059 s/token | measured |
| service compute, 70B, T4 | 3.1 s/token batch 1, 0.093 s/token batch 32 | measured |
| service compute, 8B, A100/H100 batch 32 | 2 ms/token amortized | estimate (int8 TOPS ratio) |
| gateway online crypto, Qwen2.5-0.5B | 0.18 s for 23 tokens | measured (unoptimised prototype) |

Batching is the main lever. Concurrent sessions on one gateway share each round trip
(`max_concurrent_sessions`, default 32), and the service coalesces rows across gateways
while a batch waits for its device worker (no fixed wait: `batch_window_ms` defaults to 0, so an idle
service answers at once). Per-token latency then approaches the RTT floor while throughput scales with
the batch.

## 4. Mask storage

```
store_bytes = slots_on_hand × masks_per_slot
slots_on_hand ≥ tokens_per_day × buffer_days       (buffer ≥ refill outage you must survive)
```

* Every prompt **and** every generated token consumes one slot. The gateway reserves
  `prompt_tokens + max_tokens` slots per request and burns the unused ones
  (`pgw_masks_burned_slots_total`). Keep `max_tokens` realistic, because over-reservation burns
  masks. Budget 10-30% burn on top of used tokens (*estimate*; measure it with the metrics).
* Example (*analytic*): 1 million tokens/day of an 8B model at K40A12 is 6.9 TB/day of masks. A
  2-day buffer is about 14 TB of fast local disk.
* 8B at K48A20: 121k slots per TB (*analytic*).
* Read throughput: the gateway reads `masks_per_slot` per token (8.3 MB for 8B at K48A20). At 100
  tokens/s that is 830 MB/s of sustained random-ish reads. Use NVMe.
* Consumed segments are shredded (deleted, DEK destroyed), so capacity is recycled.
* **Speculative decoding** (off by default; a draft model or prompt lookup on the gateway) reserves
  up to `spec_row_budget` mask slots per requested token (default 3) and burns the unused ones when the
  request ends. Size the store and the factory for up to `spec_row_budget ×` the token volume while it
  is on (`--spec-row-budget` in the calculator); the burn metric shows what is actually used.
* With **encrypted mask generation** the store holds minutes to hours instead of days, but every slot
  also crosses the network once more; see the costs in
  [encrypted-masks.md](encrypted-masks.md#costs-per-token-measured).

Low water and target: set `PGW_MASKSTORE_LOW_WATER_SLOTS` to at least
`time to deliver one segment × peak_tokens_per_second` (`pgw doctor --role factory --pack ...` times
one chunk and prints the segment time), and `PGW_MASKSTORE_TARGET_SLOTS` to your buffer. The Helm
alert `PGWMaskRunwayLow` fires when the masks left last less than `alerts.runwayMinSeconds`
(default 7200 s) at the current consumption rate; set it above the segment delivery time.

A new or drained store is refilled with a ramp when `PGW_FACTORY_FIRST_SEGMENT_SLOTS` is set (the
setup wizard sets 256): while below low water, one segment of `min(segment_slots,
max(first_segment_slots, next_pow2(2 × available)))` slots is made at a time (256, 512, 2048, 4096
with the defaults), so the gateway can serve after the first small segment.

## 5. Mask-factory throughput

| hardware | model, ring | rate | basis |
|---|---|---|---|
| NVIDIA T4 | 8B, K48A20 | 0.0066 s/slot, 550k slots/h | measured |
| NVIDIA T4 | 8B, K32A8 | 0.0043 s/slot | measured |
| NVIDIA T4 | 70B, K48A20 (weights streamed from host RAM) | 0.047 s/slot, 77k slots/h | measured |
| NVIDIA T4 | 70B, K32A8 (streamed) | 0.032 s/slot | measured |
| 4 vCPU Xeon 2.0 GHz (fbgemm int8) | 8B, K48A20 | 0.28 s/slot (GEMM time only) | derived from measured 0.30 TOPS |
| 4 vCPU Xeon 2.0 GHz (fbgemm int8) | 70B, K48A20 | 2.8 s/slot | derived |
| 4-core container, `cpu_int8` backend (`torch._int_mm`) | int8 GEMM | 1.0 TOPS | measured |
| Laptop NPU, 40 TOPS | 8B, K48A20 | 3-7 ms/slot | estimate, untested |

Cost of masks on cloud T4s (derived from the measured rate and list prices): 8B at K48A20 costs
$0.36 per million tokens on spot and $0.96 per million on-demand. 70B at K48A20 costs $2.60 per million on spot.

GPUs needed (24/7) = `tokens_per_day × s_per_slot / 86400`, plus 30-50% headroom for refill
after outages. Example (*derived*): 10 million 8B tokens/day at K48A20 needs 10M × 0.0066 / 86400
≈ 0.76 T4. One T4 is enough, at three quarters of its time.

Measure your own hardware with `pgw factory benchmark --pack <pack> --ring <ring>`.

### One model, one ring: Qwen2.5-7B at K32A12C27

The rates above mix models and rings. For the default deployment, in slots (= tokens) per hour:

| mask source | rate | basis |
|---|---|---|
| local T4 factory | about 895k slots/h (0.0040 s/slot) | estimate (8B K32 T4 measurement scaled by weights) |
| local 4 vCPU factory (fbgemm int8) | about 21k slots/h (0.174 s/slot) | derived (GEMM time only) |
| encrypted, gateway encrypt + decrypt (4 vCPU with AMX) | about 31k slots/h (0.116 s/slot) | derived from measured |
| encrypted, provider T4 | about 190k slots/h (0.019 s/slot) | derived from measured |
| encrypted, 1 Gbit/s link (12.8 MB per slot) | about 35k slots/h | estimate |
| encrypted over the Modal tunnel (1.5 MB/s) | about 420 slots/h | derived |

The slowest stage sets the rate. One decode stream at 0.365 s/token consumes about 9.9k slots/h, so
an encrypted supply on one 4-vCPU gateway feeds about three streams, and the Modal tunnel not even one.

## 6. Gateway host

| resource | sizing | basis |
|---|---|---|
| CPU | 4 vCPU per gateway for 8B at moderate load. Mask regeneration (ChaCha20) and unmasking are about 2.2 M ring elements per token; the SPEC target is 10 ms/token on CPU | estimate (SPEC 14 target) |
| RAM | plain weights (embeddings, LM head, norms; about 4.2 GB in fp32 for Llama-3-8B with its untied LM head, half that with `--plain-dtype bfloat16`, which is lossless for bfloat16 checkpoints and lets the gateway read a 16-bit LM head with float32-identical logits; `float16` also halves it but rounds the weights) + KV cache (`2·L·kv_heads·head_dim·4 bytes` per token per session; 256 KB per token for 8B in fp32) + mask buffers | analytic |
| Disk | mask store (section 4) on NVMe | analytic |
| GPU | not needed | — |

## 7. Compute service (our side, for capacity discussions)

| item | value | basis |
|---|---|---|
| 8B, T4, batch 1 | 0.45 s/token | measured |
| 8B, T4, batch 32 | 0.014 s/token amortized | measured |
| ratio to plain fp16 serving at batch 1 | about 8x (0.45 vs 0.059 s/token) | measured |
| scaling | horizontal; stateless replicas behind an L7 load balancer | design |

## 8. Worked example

A hospital runs Llama-3-8B at K40A12 for 2 million tokens/day (prompts plus completions), peak
3x the mean, in its own AWS account over PrivateLink (*analytic* unless noted):

| item | value |
|---|---|
| wire per day | 2M × 11.1 MB = 22.3 TB/day; mean 2.1 Gbit/s, peak about 6.2 Gbit/s |
| masks per day | 2M × 6.88 MB = 13.8 TB/day; 2-day buffer about 28 TB NVMe |
| factory | 2M × 0.0066 s / 86400 ≈ 0.15 T4 (derived from the K48 measurement; K40 is faster); one T4 with ample headroom |
| latency floor | 128 × ~1 ms ≈ 0.13 s/token (estimate) |

Run `python docs/tools/pgw_sizing.py --model llama-3-8b --ring K40A12 --tokens-per-day 2000000 --rtt-ms 1`
to reproduce this table.
