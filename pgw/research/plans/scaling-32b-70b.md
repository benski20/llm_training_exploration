# Scaling to Qwen2.5-32B and 70B-class targets at 25-30 tokens/s (CPU-only gateway, no customer GPU)

Date: 2026-11-22. Design and quantitative model only; no Modal, no Kaggle, nothing implemented beyond the model.
Model: `benchmarks/scaling_model.py` (standard library, runnable; arithmetic unit-tested in
`tests/benchmarks/test_scaling_model.py`). Every number below is tagged **measured** (in this repository, source
named), **derived** (arithmetic on a measured number) or **estimate** (an assumption or a target not yet built).
Re-run with `python benchmarks/scaling_model.py [--alpha A] [--ctx N] [--json out.json]`.

Requirement: 25-30 tokens/s streamed for Qwen2.5-32B and Qwen2.5-72B / Llama-3-70B, the provider seeing only
masked data, every result integrity-checked, outputs exact w.r.t. a pinned reference, no GPU and no large storage
on the customer's premises (thin client). The customer's gateway is a CPU-only VM the customer controls,
co-located with the provider; the mask factory may run in a customer-controlled cloud environment.

## 0. Answer

| model (ctx 2048, alpha 0.70) | A today | B engineered | C B + floors + fast draft | D C + fused 2 trips/layer | E D + bf16 KV + B200 | D at measured alpha 0.49 |
|---|---|---|---|---|---|---|
| Qwen2.5-7B | 6.1 | 61.9 | 102.7 | 140.2 | 145.9 | 120.7 |
| Qwen2.5-14B | 3.0 | 41.8 | 55.0 | 71.5 | 77.6 | 60.0 |
| **Qwen2.5-32B** | 2.1 | 29.2 | **40.2** | **49.0** | 53.9 | 42.3 |
| **Qwen2.5-72B** | 1.2 | 20.1 | 27.0 | **32.8** | 35.6 | 28.7 |
| **Llama-3-70B** | 1.2 | 20.1 | 24.9 | **30.9** | 33.7 | 26.6 |
| bound (32B / 72B) | gateway attention (1 core) | gateway attention | gateway attention | gateway attention + provider GPU | gateway attention | gateway attention |

Tokens/s, model estimates (the 7B "today" column is at ctx 2048 with a single-core attention estimate; the
measured 7B H100 run at short context gave 11.0 tok/s, section 2). Scenarios are cumulative sets of the
levers of section 4; "today" is the measured paths (native trip path, p31, chain speculation available).

1. **32B at 25-30 tokens/s is reachable without any protocol change** (scenario C: 40 at alpha 0.70, 33 at 0.49):
   it needs engineering only, on the provider's service path, the network and above all the gateway's CPU work
   (multi-core trips and attention, native glue, a fast draft).
2. **70B-class at 25 tokens/s needs the same plus either fused linear chains (2 trips per layer) or a
   whole-host gateway; 30 tokens/s needs fused chains** (D: 31-33 at alpha 0.70, 27-29 at 0.49), and at 0.49
   also bf16 KV or B200s. Without fused chains the best 72B figure is 27.0 (C).
3. **The binding component at 32B-72B is the gateway's attention over the private KV cache**, not the provider
   GPU and not the round trips once those are engineered: 1.34 GB of float32 KV per pass at 72B and 2k context
   (655 KB per context token) and 5.4 GFLOP per query row. A CPU gateway cannot make this go away; it can only
   spread it over cores and memory channels (100-300 GB/s), halve it with a bfloat16-KV pinned reference, or
   keep the context short. At 8k context no lever set reaches 25 tokens/s at 72B (14-17), at 32k 5-6.
4. **Speculation gives less at this scale than at 7B.** The measured 7B H100 run gave tau = 1.19 at R = 2
   (alpha about 0.49 for the pipelined drafter), every extra row costs gateway attention FLOPs and trip work,
   and a 0.5B draft step on a bandwidth-bound host costs 60 ms (measured). Best modes are R = 2-3; tree
   speculation never wins in the model (each depth level is one more sequential draft step).
5. **Masks**: a 72B slot is 28 MB (6.2 MB at 7B); decode alone at 30 tok/s needs 33-40 slots/s (0.9-1.1 GB/s),
   with prompts and burned reservations (measured 1.7 extra slots per generated token at 7B) about 95 slots/s,
   2.6 GB/s. With today's MLWE factory that is 44 CPU cores and a 172 GB hint (345 GB in NTT form) at the
   customer. **Recommended: a plain factory on a customer-cloud GPU** (0.2 of an L4, no hint, 72 GB of public
   weights), or a hint-free RLWE scheme if no customer GPU at all is acceptable (section 6).

## 1. Shapes and per-token quantities (analytic, K32)

| | Qwen2.5-7B | 14B | 32B | 72B | Llama-3-70B |
|---|---|---|---|---|---|
| layers / hidden / intermediate | 28 / 3584 / 18944 | 48 / 5120 / 13824 | 64 / 5120 / 27648 | 80 / 8192 / 29568 | 80 / 8192 / 28672 |
| heads / kv heads / head dim / vocab | 28/4/128/152064 | 40/8/128/152064 | 40/8/128/152064 | 64/8/128/152064 | 64/8/128/128256 |
| params (model; tested against the published size) | 7.62 B | 14.77 B | 32.76 B | 72.71 B | 70.55 B |
| trips per pass, 4/layer + masked head / fused 2/layer | 113 / 57 | 193 / 97 | 257 / 129 | 321 / 161 | 321 / 161 |
| sum n (input elements per row-token, incl. head) | 0.84 M | 1.41 M | 2.76 M | 4.34 M | 4.27 M |
| sum m (output elements = mask words) | 1.54 M | 2.31 M | 4.81 M | 7.01 M | 6.85 M |
| wire per row-token, up / down | 3.3 / 6.2 MB | 5.6 / 9.3 MB | 11.0 / 19.2 MB | 17.4 / 28.1 MB | 17.1 / 27.4 MB |
| int8 weights at the provider (incl. head) | 7.1 GB | 14.0 GB | 32.0 GB | 71.5 GB | 69.5 GB |
| mask slot (W r, 4 B per output) | 6.17 MB | 9.3 MB | 19.2 MB | 28.1 MB | 27.4 MB |
| MLWE hint (24.6 KB per output row) | 38 GB | 57 GB | 118 GB | 172 GB | 168 GB |
| KV, float32, per context token | 115 KB | 393 KB | 524 KB | 655 KB | 655 KB |
| attention FLOPs per row at ctx 2048 | 0.82 G | 2.0 G | 2.7 G | 5.4 G | 5.4 G |

Wire per trip is `R x (n + m) x 4 B`; the largest is 72B gate_up, 270 KB per row (22 us at 100 Gbit/s).
With fused chains the elements per pass are unchanged by assumption (the fused-chain design is another
package's research; if it removes the gate_up output from the wire, wire and masks fall too).

## 2. Calibration and what was measured

| input | value | tag / source |
|---|---|---|
| gateway kernels per element (p31, 7B shapes, dev-box core) | prepare + input half 7.36 ns per input; W r + unmask + check 3.89 ns per output; r keystream 4.55 ns per input | derived from `native_trip_floor_bench.py` (STATUS p31 table: 6.15 + 2.2 + 2.55 + 1.25 + 3.8 ms over 113 trips); a test reproduces the per-token floor |
| orchestration per trip around the kernels | 0.15 ms | derived: trips-only process CPU 29-37 ms minus the 15.9 ms floor, over 113 trips |
| 1-row trip on Modal, H100 provider, gVisor | 0.26 ms mean (service 0.066, wire 0.19), p50 0.21 | measured (lead, 2026-11-22) |
| service overhead (T4 runs) | 0.10-0.16 ms | measured (deploy/modal/README) |
| 0.5B int8 draft step | 75 ms (1 dev-box core), 46-54 ms (4 vCPU), 60 ms (32-vCPU gVisor host) | measured; 8.4 GB/s effective, memory-bound |
| tokens per pass, 7B target / 0.5B draft, R = 2 pipelined | 1.19, i.e. alpha = 0.49 | measured tau; alpha derived by inverting the two-state model (exact match) |
| HE customer work, 7B slot | 24.2 ms wall on 4 vCPU = 96.8 core-ms (NTT, A and hint in NTT form) | measured (scale-north-star 6) |
| GPU NTT on a T4, end to end | 5.0 us encrypt + 2.1 us decrypt per row and 256-slot group | measured (X5) |
| factory at 7B, H100 maskgen + 4 CPU workers | 20-25 slots/s produced vs 16-17 consumed | measured (lead) |
| slots per generated token, 7B, prompts included | 2.7 plain, 4.2 with R = 2 | measured (lead) |
| H100 / H200 / B200 HBM, int8 peak | 3.35 / 4.8 / 8.0 TB/s; 1979 / 1979 / 4500 TOPS; 75% of HBM and 50% of int8 peak achieved | estimate (datasheet x efficiency) |
| attention on the gateway | 10 GB/s and 10 GFLOP/s on one core today; 100 GB/s, 600 GFLOP/s split over 16-32 cores; 300 GB/s, 1.5 TFLOP/s on a whole 2-socket host | estimate (fixed-order non-FMA float32) |
| acceptance | 0.70 central, 0.49 measured at 7B; second-choice capture gamma 0.35 for trees | estimate except the 0.49 |

**Validation against the measured 7B run** (`validate_7b()`; local bf16 head, short context, gVisor lane):
the model's lane time per token is 28.7 ms against 29.1 measured; its gateway time is 100.7 ms against
61.9 ms measured (91 - 112 x 0.26). The model's local-head read (1.09 GB at 20 GB/s = 54.5 ms) plus dev-box
per-trip coefficients over-predict that host's gateway CPU by up to 1.6x (the split between head and trips was
not measured). The gateway-CPU columns below are therefore conservative where they use today's coefficients;
the engineered scenarios use targets (section 4), not these coefficients.

The lead's key lesson is built in: **anything the gateway reads per pass must be small**. The model therefore
uses the masked head (no 1.09-2.5 GB head read per token; `masked_head = False` and `host_gb_s` model the local
head), and charges attention by bytes and FLOPs.

## 3. Per-pass model

`pass = sum over trips (RTT + service + GPU(R) + wire(R) + gateway(R)) + float ops + attention(R) + sampling + draft`,
`tokens/s = tau / pass`.

* **GPU(R)** = kernel floor + max(weights / TP / (HBM x 0.75), 2 x weights x 4 limbs x R / TP / (int8 x 0.5))
  + gather sync (TP > 1). Memory-bound up to about 70 rows on an H100 (4 limbs per row); every PGW decode trip
  is far below that, so a trip costs one weight read regardless of R or of other tenants' rows in it.
* **TP without all-reduce.** Each trip is one linear whose output goes back to the gateway (the nonlinear
  steps are the gateway's), so the provider splits a linear by output rows: each GPU reads 1/TP of the weights,
  the input (at most 236 KB at 72B, R = 2) is broadcast, and the output slices are gathered (or written straight
  to the reply buffer by each GPU). No reduction mod 2^K, no NCCL all-reduce on the critical path; the sync is a
  one-shot gather: 20 us NCCL-class today (estimate), 5 us with a P2P/NVLink flag-based gather in the persistent
  kernel (target).
* **gateway(R)** = orchestration + R x (n x (ns_in [+ ns_r]) + m x ns_out) / core speed / cores per trip + split
  sync. The r keystream is activation-independent and moves to helper cores (`r_offcore`).
* **attention(R)** = max(ctx x KV bytes per token / bandwidth, R x 4 L H d ctx / FLOP rate). At 2 rows and
  above it is FLOP-bound on 600 GFLOP/s.
* **speculation**: `chain_seq` (R - 1 draft steps between passes), `chain_pipe` (the reviewed PipelinedDrafter,
  m = 0; the background block hidden under the pass), `chain_pipe_hyp` (new, not built: the background block
  drafts one chain per acceptance point, guessing the draft's second choice after a rejection; same foreground
  work whatever the outcome, so no acceptance signal), `tree_seq` (greedy optimal tree of R nodes, one draft
  step per depth level). Rows: 1..8 (16 for trees); the model picks the best per scenario.

Per-pass breakdown, ms (ctx 2048, alpha 0.70):

| scenario | model | RTT | service | GPU | wire | gateway trips | float | attention | draft | pass | tau | tok/s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A today | 32B | 25.7 | 36.0 | 14.8 | 9.7 | 90.1 | 37.8 | 268.4 | 0 | 482.7 | 1.00 | 2.1 |
| A today | 72B | 32.1 | 44.9 | 23.2 | 14.5 | 127.1 | 53.1 | 536.9 | 0 | 832.0 | 1.00 | 1.2 |
| B engineered | 32B | 10.3 | 6.4 | 5.2 | 7.3 | 7.8 | 4.6 | 13.4 | 0 | 55.3 | 1.61 (pipe R3) | 29.2 |
| B engineered | 72B | 12.8 | 8.0 | 9.7 | 7.3 | 9.3 | 5.2 | 17.9 | 14.2 | 84.6 | 1.70 (seq R2) | 20.1 |
| C + floors + fast draft | 32B | 7.7 | 3.9 | 5.0 | 3.6 | 7.8 | 4.6 | 13.4 | 8.1 | 54.4 | 2.19 (seq R3) | 40.2 |
| C + floors + fast draft | 72B | 9.6 | 4.8 | 9.4 | 5.4 | 10.1 | 6.5 | 26.8 | 8.1 | 81.2 | 2.19 (seq R3) | 27.0 |
| D + fused 2/layer | 32B | 3.9 | 1.9 | 4.1 | 3.6 | 4.7 | 4.6 | 13.4 | 8.1 | 44.7 | 2.19 (seq R3) | 49.0 |
| D + fused 2/layer | 72B | 4.8 | 2.4 | 8.2 | 3.6 | 5.5 | 5.2 | 17.9 | 4.1 | 51.9 | 1.70 (seq R2) | 32.8 |

Sensitivity (tokens/s, best mode):

| scenario, model | alpha 0.49 | 0.60 | 0.70 | 0.80 | ctx 512 | 2048 | 8192 | 32768 |
|---|---|---|---|---|---|---|---|---|
| C, 32B | 33.2 | 36.0 | 40.2 | 44.9 | 49.4 | 40.2 | 23.1 | 8.6 |
| C, 72B | 23.3 | 25.0 | 27.0 | 30.1 | 35.9 | 27.0 | 14.4 | 5.1 |
| D, 32B | 42.3 | 45.4 | 49.0 | 55.1 | 65.9 | 49.0 | 25.8 | 8.9 |
| D, 72B | 28.7 | 30.8 | 32.8 | 35.4 | 44.8 | 32.8 | 16.1 | 5.3 |
| D, Llama-3-70B | 26.6 | 28.6 | 30.9 | 33.7 | 43.3 | 30.9 | 15.5 | 5.2 |
| F = E + whole-host gateway (300 GB/s), 72B | | | 45.9 | | | | | |

Llama-3-70B trails Qwen2.5-72B only through its draft (Llama-3.2-1B, 1.24 GB of int8 per step vs 0.49 GB).

## 4. The minimum improvements, with numeric targets

Levers (each a set of scenario values in `LEVERS`; "today" in brackets):

| lever | target | today | why it matters at 72B (D) |
|---|---|---|---|
| `net` / `net_floor` | lane RTT 40 / **30 us**, NIC 100 / 200 Gbit/s | 100 us (range 30-100), 25 Gbit/s; 190 us lane wire measured under gVisor | 161-321 trips x RTT; 45 MB per row-token |
| `persistent` / `service_floor` | service **25 / 15 us** per trip, GPU kernel floor 3 / 2 us | 66 us measured (H100), 100-160 us (T4) | 161-321 x service |
| `tp_fast` | TP 4 (72B on H100), gather **<= 5 us** | TP by fit, 20 us NCCL-class | GPU 72 GB per pass: 7 ms at TP4 |
| `gw_native` | orchestration **<= 20 us/trip**, glue <= 30 us/layer, r on helper cores | 150 us/trip, 250 us/layer | 321 x 150 us = 48 ms alone |
| `gw_parallel` | each trip's prepare/finish split over **16 cores**, 1.5x dev-box core, 48 vCPU VM | 1 core | 72B row-token = 59 core-ms of kernels |
| `gw_kernels` | 2.0 ns per input, 1.5 ns per output element | 7.4 / 3.9 (dev box) | marginal once split over 16 cores |
| `attn_parallel` | KV attention at **100 GB/s and 600 GFLOP/s** (KV heads x context blocks over 16-32 cores, same fixed summation order) | ~10 GB/s, ~10 GFLOP/s (1 core, estimate) | 1.34 GB and 5.4 GFLOP per row per pass |
| `kv_bf16` | bfloat16 KV as a **new pinned reference** | float32 | halves the KV bytes |
| `attn_host` (stretch) | a whole 2-socket host per stream: 300 GB/s, 1.5 TFLOP/s | | the only lever that makes 8k context work |
| `draft_bw` / `draft_fast` | draft step **<= 4-5 ms**: int4 0.5B on its own cores at 80 GB/s | 60 ms measured | otherwise speculation loses |
| `pipe_hyp` | hypothesis tree in the pipelined drafter's block | not built | +5-10% tau at m = 0 |
| `fused2` | **2 trips per layer** | 4 | halves RTT + service + sync |
| `tree` | tree speculation up to 16 rows | chains | never chosen by the model |
| `b200` | B200 provider | H100 | GPU 8 -> 4 ms per pass at 72B |

Smallest lever sets (exhaustive search over the non-stretch levers, `minimal_lever_sets`; tokens/s reached):

| goal | alpha 0.70 | measured alpha 0.49 |
|---|---|---|
| 32B >= 25 and >= 30 | gw_native + gw_parallel + attn_parallel + draft_fast + net_floor + service_floor (33.2) | 25: same (26.9); 30: tp_fast + gw_native + gw_parallel + attn_parallel + fused2 + net_floor + service_floor (33.9) |
| 72B >= 25 | gw_native + gw_parallel + attn_parallel + fused2 + draft_fast + net_floor + service_floor (26.3); without fused2: + tp_fast (25.4) | tp_fast + gw_native + gw_parallel + attn_parallel + kv_bf16 + fused2 + net_floor + service_floor (27.4) |
| 72B >= 30 | tp_fast + gw_native + gw_parallel + attn_parallel + fused2 + draft_fast + net_floor + service_floor (30.8) | not reachable without B200 or a whole-host gateway; with `b200` + `kv_bf16`: 30.9 (E) |
| Llama-3-70B >= 25 | tp_fast + gw_native + gw_parallel + attn_parallel + fused2 + draft_fast + net_floor + service_floor (28.6) | ... + kv_bf16 instead of draft_fast (27.7) |
| Llama-3-70B >= 30 | the 72B set + gw_kernels (30.2) | B200 + bf16 KV + whole set (31.1) |

So the **minimum common set** for 32B at 30 and 70B-class at 25 is, as numeric targets:

1. gateway, per trip: **<= 20 us of orchestration** (one native call per side, no Python), prepare/finish **split
   over 16 cores** (>= 1.5x the dev box per core), r keystream on helper cores; per-trip gateway time at 72B
   gate_up <= 40 us at R = 2;
2. gateway attention: **>= 100 GB/s and >= 600 GFLOP/s** over the KV cache for one stream, context <= 2-4k at
   72B;
3. lane: **RTT <= 30 us**, NIC >= 100 Gbit/s (placement group or same host, busy-polled kernel-bypass or UDS
   across a VM boundary);
4. provider: **<= 15-25 us of service overhead per trip** (persistent kernel), TP 4 with a <= 5 us gather;
5. draft: **<= 4-5 ms per draft step** and alpha >= 0.6 for the chosen R (2-3);
6. for 70B-class: **fused chains (2 trips per layer)**; for 30 tokens/s at alpha ~0.5 also bfloat16 KV and
   B200-class HBM (or a whole-host gateway).

What does *not* bind: provider GPU compute (memory-bound, 5-16% busy per stream), mask arithmetic on the gateway
once split, the wire at 100 Gbit/s (3.6 ms per pass at 72B, R = 2).

## 5. Physics limits (what no engineering removes)

* **Sequential trips**: per pass, trips x (RTT + service). 72B: 321 x 45 us = 14.4 ms at the stretch floors
  (4 per layer), 7.2 ms fused. At 100 us + 66 us (measured-class) 53 ms: 18.8 tok/s ceiling with zero other work.
* **Provider HBM**: one weight read per trip set: 71.5 GB per pass. H100 x1 at 75%: 28 ms (and does not fit);
  H200 x1: 20 ms; H100 x4: 7 ms; B200 x4: 3 ms. Speculation rows and other tenants' rows ride free.
* **Gateway attention**: 655 KB (fp32) per context token per pass at 72B, 524 KB at 32B; 4 x L x H x d FLOPs per
  row and context token (5.4 GFLOP per row at 72B, 2k). On a CPU this is the term that grows with context.
* **Wire**: 45.5 MB per row-token at 72B; 30 tok/s x 1.2-1.7 rows per token = 13-19 Gbit/s sustained.
* **Masks**: 28 MB per slot at 72B is irreducible information (the gateway must unmask every output); `W r mod
  2^29` saves 9.4%.
* **Exactness**: every row's float path is row-independent (native float ops), so speculative and padding rows
  are exact by construction; tree attention needs the same key order as single-row decoding (prefix, then
  ancestors by depth). A bf16-KV reference or an int4 draft are allowed (the first is a new pin; the second only
  proposes). K32A12C27 at 72B: the cap is per call (Cauchy-Schwarz), so widths 29568 decode, but the activation
  bits chosen may fall at 72B; perplexity vs K40A12 must be re-measured before pinning (K40 costs +25% wire and
  masks).

## 6. Masks and the mask factory

At the rate of scenario D (decode only / with prompts and burned reservations at the measured 1.7 extra slots
per generated token):

| | 32B | 72B | Llama-3-70B |
|---|---|---|---|
| slots per generated token | 1.37 / 3.07 | 1.18 / 2.88 | 1.36 / 3.06 |
| slots/s | 67 / 150 | 39 / 94 | 42 / 95 |
| mask stream into the gateway | 1.3 / 2.9 GB/s | 1.1 / 2.6 GB/s | 1.2 / 2.6 GB/s |
| RAM ring, 10 s | 13 / 29 GB | 11 / 26 GB | 12 / 26 GB |

Factory schemes at the with-prompts rate (estimates scaled from the 7B measurements by elements):

| scheme | 32B | 72B | needs |
|---|---|---|---|
| MLWE + hint (today), CPU NTT | 46 cores; hint 118 GB (236 GB NTT form); 47 Gbit/s factory <-> provider; 7% of an H100 | 44 cores; hint 172 GB (345 GB); 45 Gbit/s; 10% of an H100 | big CPU pool + large storage at the customer |
| MLWE + hint, GPU NTT on customer-cloud T4s (end to end) | 14 T4s | 13 T4s | GPUs anyway; host-bound today |
| **plain factory on a customer-cloud GPU** (computes W8 r itself; public weights) | 0.16 of an L4 / 0.04 of an H100 | 0.22 of an L4 / 0.05 of an H100 | 32-72 GB of public weights, one cloud GPU, no hint, no provider work |
| hint-free RLWE (placeholder until the RLWE package reports) | 26 cores, 90 Gbit/s, 31% of an H100 | 24 cores, 84 Gbit/s, 44% of an H100 | no hint; wire is the problem |

* Prompt tokens dominate mask demand in chat: every prompt row consumes a slot. **Keep the conversation's KV
  cache at the gateway across turns** (only new tokens need masks), reserve per pass instead of per
  `max_tokens` (the measured 24-31% burn), and size the prompt pass separately: a 2k-token prompt at 72B is
  56 GB of masks.
* Recommendation: the factory in the customer's cloud account, same region as the gateway, on one small GPU (L4
  class), computing `W8 r` directly. It holds only public weights and seeds, needs no provider involvement (so no
  selective-failure surface, no hint), and its output goes to the gateway's RAM ring at 1-3 GB/s. If "no
  customer GPU anywhere" is a hard rule, today's MLWE factory needs ~45 cores and ~172-345 GB of hint per 72B
  stream; the RLWE line must bring the customer CPU below ~10 cores and the wire below ~25 Gbit/s to be
  preferable.
* The plain cloud-GPU factory should batch >= 1024 slots per weight pass (compute-bound) and stream outputs per
  linear so its host RAM stays below ~2 segments.

## 7. Storage

| where | 32B | 72B | note |
|---|---|---|---|
| gateway: embeddings (bf16, `PGW_EMBED_COMPACT=1`) | 1.6 GB | 2.5 GB | memory-mapped; one row per token |
| gateway: scales, norms, biases | 22 MB | 33 MB | |
| gateway: p31 check vectors (u and uW per segment) | ~0.06 GB | ~0.09 GB | per active segment |
| gateway: KV per sequence at 2k (fp32 / bf16) | 1.07 / 0.54 GB | 1.34 / 0.67 GB | per concurrent chat |
| gateway: draft | 0.49 GB (0.25 int4) | 0.49 GB | Llama: 1.24 GB |
| gateway: mask RAM ring, 10 s | 13-29 GB | 11-26 GB | RAM, not disk; 2-4 s suffice with a co-located factory |
| factory (plain GPU) | 32 GB weights | 71.5 GB weights | public data |
| factory (MLWE) | 118-236 GB hint | 172-345 GB hint | the "large storage" the requirement excludes on premises |
| customer premises | 0 | 0 | thin client: API key and TLS only |

## 8. Provider-side design

* **Persistent-kernel trip serving.** One resident kernel per GPU (or per GPU partition) polls a pinned ring
  of trip descriptors (linear id, rows, input offset) written by the lane (or by the NIC with GPUDirect RDMA),
  runs the K/8-limb GEMV over its weight slice, writes the output slice and a completion flag the lane spins on.
  No launch, no graph replay, no host thread per trip. Target: <= 15-25 us of service overhead per trip
  (today 66 us on the H100 run, 100-160 us on T4s). The CUDA-graph replay path stays as the fallback (and for
  prompt passes).
* **Tensor parallel for 70B**: output-row split (section 3): 4 x H100 (80 GB) or 2 x H200 / 1 x H200 (72B int8
  fits 141 GB) or 4 x B200. No all-reduce; the gather is per trip: each GPU's persistent kernel writes its slice
  into GPU 0's reply buffer over NVLink with a flag (P2P stores), <= 5 us. The masked head (1.25 GB) is split the
  same way. Factory maskgen (MLWE) uses the same weights at low priority.
* **Multi-tenant batching without hurting latency.** A trip of one stream occupies 5-17% of the GPU set's time
  (D). Two mechanisms: (a) time multiplexing: at rho 0.6 three 72B streams per 4 x H100 set add ~3.7 ms per
  pass (M/D/1 mean wait over 161 trips); (b) opportunistic coalescing: the persistent kernel takes every queued
  descriptor of the same linear when it starts a weight read (never waits for more), so coinciding trips of
  different customers share one weight read. Rows are masked uniformly; outputs are routed by descriptor; each
  customer's strict-speculation invariants hold w.r.t. its own data. Cross-tenant timing interference is
  metadata (SPEC 3.4) and does not depend on any customer's plaintext beyond what its own pass count/rows show.
* **$/hour** (estimates: H100 $3.0/GPU-h, H200 $3.8, B200 $6.0, $0.045 per vCPU-h): scenario D, 72B: 4 x H100 =
  $12/h shared by 3 streams = $4/h + 48-vCPU gateway $2.2/h = **$6.2/h per stream, ~$52 per million tokens** at
  32.8 tok/s; 32B: $4.2/h, ~$24 per million; factory (plain, L4 share) +$0.2/h; MLWE factory instead +~$2/h of
  CPU plus hint storage.

## 9. Generalising the code (design only; nothing implemented here)

1. **One trip plan shared by engine, factory, model and service.** `pgw.engine.tripplan.TripPlan.from_pack(pack,
   trips_per_layer=4|2, masked_head=bool)` -> ordered groups `(name, linears, n, m, weight_bytes)`; the engine's
   per-layer loop, the factory's slot layout, the reservation's per-linear cursors and `scaling_model.py`'s
   `trip_groups` all read it. Fused chains (other package) plug in as `trips_per_layer = 2`.
2. **Gateway config knobs** (`GatewayModelConfig`, all default off / today's value):
   `masked_head: bool` (wire the existing library opt-in into service registry, factory CLI and runtime; required
   for >= 14B), `trip_cores: int` (split prepare/finish of one trip over the native pool; `PGW_TRIP_SPLIT_MIN_ELEMS`
   threshold), `attn_threads: int` and `kv_dtype: "float32" | "bfloat16"` (bf16 = a new pinned reference,
   `plaintext_reference(kv_dtype=)`), `draft_quant: "int8" | "int4"`, `draft_cores: int` (dedicated draft pool,
   follow-up of the strict-speculation plan), `spec_pipeline_hypotheses: bool` (the `chain_pipe_hyp` drafter),
   `spec_tree: int` (tree rows; native attention with an ancestor mask in single-row key order).
   Mask ring sized in seconds of demand (`mask_ring_s`), not slots, since a slot is 6-28 MB.
3. **Native split kernels** (`rust/pgw-fast`): `trip_prepare_into` / `trip_finish_into` / `trip_prime_output`
   take an element range and return partial Freivalds sums (mod p) to combine; the lease hands out one row to a
   group of cores; the per-trip fork/join is a spinning pool (<= 4 us).
4. **Attention** (`attn.rs`): parallel over KV heads and context blocks with the fixed summation order kept
   (partial softmax maxima/sums combined in a fixed tree order; the reference uses the same order, so bits do
   not depend on the thread count, as for the existing native float path).
5. **Provider** (`pgw.service`): `PGW_SERVICE_TP=G` (output-row shards of each linear at load; the pack's
   fingerprint unchanged, shard offsets derived), `PGW_SERVICE_EXEC=graph|persistent`, `PGW_SERVICE_COALESCE=1`
   (opportunistic same-linear coalescing, never waits), `lane_max_rows` raised to cover R x tenants. Wire
   protocol unchanged: sharding is internal.
6. **Factory** (`pgw.factory`): `factory_scheme = "mlwe_hint" | "plain_gpu" | "rlwe"` behind one
   `MaskSource` interface (`produce(segment_spec) -> W r chunks + check vectors`), so the store, ring and gateway
   do not care which. `plain_gpu` is the existing trusted-factory path run on a customer-cloud GPU with weights
   streamed per batch.
7. **Model**: `benchmarks/scaling_model.py` takes its scenario values from these knobs' names, so a measured run
   can be dropped in (`validate_7b` is the pattern: one dict of measured values per run).

## 10. Next measurements (in order of the uncertainty they remove)

1. Gateway attention at 32B/72B shapes on a server VM (bytes/s and FLOP/s vs threads; the term that binds).
2. The split-trip native path on 16 cores: per-trip gateway time at 72B gate_up (target <= 40 us at R = 2).
3. A persistent-kernel provider on one H100: service overhead per trip and the TP gather on 4 GPUs.
4. Acceptance of the 0.5B (int4) draft against 32B and 72B targets, chain and hypothesis-tree pipelined.
5. Lane RTT between two VMs in one placement group without gVisor (target 30 us).
6. Perplexity of K32A12C27 at 72B vs K40A12 (the ring choice changes wire and masks by 25%).
