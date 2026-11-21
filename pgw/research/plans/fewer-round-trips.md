# R1: fewer sequential provider round trips per token

Date: 2026-11-21. Scope: reduce the sequential provider round trips per decoded token below 2 per layer, or
overlap them, **without** weakening privacy (the provider learns nothing about data; nothing
data-dependent leaks through pass counts, rows or timing) and **without** losing exactness (bit-identical
to the plaintext int8 reference). New modes default off. Light CPU only; no Modal runs.

Prototype in this branch: `local_linears = attention_down | all | auto_kinds` plus `local_max_rows`
(engine: `pgw.engine.backends`, `pgw.engine.model`; gateway config and runtime wiring), exactness tests in
`tests/engine/test_local_linears_r1.py`. Latency model: `benchmarks/round_trip_model.py`.

## 0. Summary

1. **Within the 2-party, stateless-provider, exact model, 2 trips per layer is the floor whenever both MLP
   linears run on the provider.** The chain is `gate_up -> SiLU*up + requantize -> down -> residual, norm,
   attention, ... -> next gate_up`. Each arrow with a nonlinear step needs the gateway, because only the
   gateway can evaluate a nonlinear function of a masked value. Section 2 shows that FSS, Beaver-triple MPC,
   garbled circuits and FHE either **do not save rounds** (FSS and MPC need both parties online at every
   nonlinear layer) or cost **10^3 to 10^5 times** the current per-token budget (GC, FHE). They also cannot be
   bit-exact to a float SiLU reference without full IEEE-754 circuits.
2. **The exact lever that works is to move a linear to the gateway.** A local linear removes its round trip.
   The provider then sees strictly less, the linear takes no masks, and the bits are the same (same
   quantization, exact integer product). Today `auto` places only qkv and o. Extending placement to `down` (1
   trip per layer) or to everything in decode (0 trips) is a large win on slow links for small models:
   **0.5B at 17 ms goes from 0.90 to 0.12 s/token (7.6x, measured local costs)**. For 7B the win needs a real
   gateway: 1.22 to 0.85 s/token (1.4x) with `attention_down`, or to 0.67 s/token (1.8x) all-local, on a
   dedicated 4-8 core gateway (estimate). Prefill must stay remote (`local_max_rows`): local prefill is
   compute-bound on the CPU.
3. **Throughput levers (batching, ping-pong cohorts) do not cut per-token latency.** Speculation multiplies
   whatever placement gives by tau (tokens per pass, about 2 to 3). Under the strict-pass rule, though, it
   adds dummy passes that delay a follow-up turn by up to (L - L/tau) x pass time. At 17 ms that is tens of
   seconds, unless decode is all-local.

## 1. Where the time goes (latency model)

`s/token = remote_trips_per_layer * L * (RTT + o) + L * local_linear_ms + F`, then divided by tau with
speculation.

| input | 0.5B (Qwen2.5-0.5B) | 7B (Qwen2.5-7B) | source |
|---|---|---|---|
| layers L | 24 | 28 | config |
| local exact GEMV, 1 row, ms per layer: qkv / o / gate_up / down | 0.25 / 0.22 / 2.52 / 1.24 | 4.19 / 3.23 / 51.5 / 23.5 | **measured here**: `exact_int_matmul`, 1 torch thread, shared 4-vCPU box, about 3 GB/s of int8 weights (2 digit GEMMs at A=12) |
| same, 4 rows | 0.30 / 0.25 / 2.64 / 1.53 | 4.55 / 3.64 / 58.5 / 24.0 | measured: memory-bound, so rows are nearly free |
| local exact GEMM, prefill | 64 G MAC/s per thread | same | measured (7B qkv shape, 64 to 256 rows) |
| o = per-trip cost beyond RTT | 1.2 ms (est.) | 2.2 ms | Kaggle run 27: 1.47 ms remote at loopback + 0.71 ms gateway masking per trip |
| F = gateway fixed per token | 17 ms (est.) | 94 ms | run 27 "other" (LM head, float ops) |

"Dedicated" below means the local linear cost divided by 4. That is an **estimate** for a 4-8 core gateway
with its own memory bandwidth, or for a 1-pass int16 x int8 kernel instead of 2 digit GEMMs. Under the 4
threads available here the box was contended (load 4+) and slower, so this was not measured.

**0.5B**, s/token (tok/s); measured local cost / dedicated estimate:

| placement | trips/token | RTT 1 ms | RTT 5 ms | RTT 17 ms |
|---|---|---|---|---|
| all remote | 96 | 0.228 (4.4) | 0.612 (1.6) | 1.764 (0.6) |
| attention local (today's `auto`) | 48 | 0.134 / 0.125 | 0.326 / 0.317 | 0.902 / 0.893 (1.1) |
| `attention_down` (new) | 24 | 0.111 / 0.080 | 0.207 / 0.176 | 0.495 / 0.464 (2.0-2.2) |
| `all` local in decode (new) | 0 | 0.119 / 0.042 | 0.119 / 0.042 | **0.119 / 0.042 (8.4-24)** |

**7B**:

| placement | trips/token | RTT 1 ms | RTT 5 ms | RTT 17 ms |
|---|---|---|---|---|
| all remote | 112 | **0.452 (2.2)** | 0.900 (1.1) | 2.244 (0.4) |
| attention local (today) | 56 | 0.481 / **0.325** | 0.705 / 0.549 | 1.377 / 1.221 (0.7-0.8) |
| `attention_down` | 28 | 1.049 / 0.400 | 1.161 / **0.512** | 1.497 / 0.848 (1.2) |
| `all` local in decode | 0 | 2.402 / 0.671 | 2.402 / 0.671 | 2.402 / **0.671 (1.5)** |

Reading: at RTT 1 ms the provider should do everything except, on a dedicated gateway, qkv and o. At 5 ms
`attention_down` wins for both sizes. At 17 ms (Modal) 0.5B should decode entirely on the gateway. 7B gains
1.4-1.8x only with a real gateway (and 6.5 GB of int8 weights in its RAM for `all`). The Modal measurement
(26 ms median trip including the CPU provider, 49.5 trips per token, 47 s for 32 tokens) matches the model's
0.5B "attention local" row once o is about 9 ms. `auto_kinds` picks this placement from a start-up
calibration (each kind local iff its local time is below one round trip). Every remote linear is its own
trip, so the kinds are independent.

**Prefill must stay remote.** Local prefill runs at about 64 G MAC/s per thread here. A 1000-token 7B prompt
takes about 13 s for qkv+o alone (this already applies to today's `auto`/`attention`) and about 100 s for
`all`, against well under 1 s on the provider's GPU. `local_max_rows` (new) keeps local linears for steps of
at most N rows in total and sends every linear for larger steps. The cut depends only on the row count, which
the provider already sees for any remote step (SPEC 3.4).

## 2. Cryptographic options for one trip per MLP (b)

What would be needed: the provider computes `down(Q(SiLU(g) * u))` from the masked `[g, u]` with no
gateway message in between. Here `Q` is the per-row absmax requantization (a data-dependent max over 4864 or
18944 elements, a division and a rounding), and SiLU is torch float32 `x * sigmoid(x)` of
`yq * sx[t] * sw[j]` (the secret per-token scale `sx` is involved).

| technique | rounds for the MLP | keeps 2-party + stateless provider? | exact (bit-identical)? | cost per token, 7B (530k gate elements = 18944 x 28) | verdict |
|---|---|---|---|---|---|
| FSS gates with a dealer (BGI'19/'21; SIGMA, PETS'24, uses it for GPT/Llama) | **>= 1 per nonlinear layer**: both parties evaluate keys on the masked value, then reconstruct. Our factory could be the dealer. | yes (dealer = factory) | only with a fixed-point reference; float SiLU is approximated (splines/LUT) | SIGMA: about 27 s/token Llama2-7B, 44 s 13B (GPU, LAN) | **no round saving**: the gateway must be online exactly where it is today; it only replaces the gateway's plaintext SiLU with an expensive joint one |
| Beaver triples / CrypTen / SecretFlow-SPU (PUMA) | O(log) rounds per comparison, 1 per multiplication: **more** rounds | yes (triples from the factory) | no (fixed-point approximations) | PUMA: LLaMA-7B about 5 min/token (3-party) | worse on every axis |
| Garbled circuits (gateway garbles offline = "precompute-ahead", provider evaluates; arithmetic labels make the int GEMMs "free" on labels, BMR'16) | **1 per layer**: the provider evaluates GEMM -> SiLU*up -> requant -> GEMM on labels. With all linears it could even be 1 per pass except attention. | yes | **yes in principle**: a circuit can emulate IEEE fp32 exactly | fp32 SiLU + mul + absmax + div/round >= ~5e4 AND gates per element -> 2.6e10 AND/token x 32 B (half-gates) = **~850 GB/token** of garbled tables streamed gateway -> provider. With an integer 12-bit LUT reference (which would change the model): ~2^12 x 16 B = 64 KB/element = **~34 GB/token** | exact but 10^4-10^5 x the bandwidth budget; the evaluator's input-label OT is also an extra round unless arithmetic labels flow straight from the GEMM |
| FHE (TFHE programmable bootstrapping for the gate, provider alone) | 0 interaction | yes | only with an integer LUT reference of <= ~8-bit inputs (27-bit gate values need costly WoP-PBS) | >= 530k PBS/token; at about 10^4 PBS/s on a GPU that is **~50 s/token** for the gates, plus LWE-ciphertext GEMMs (~n_lwe = 700-1000 x the int8 work) | infeasible, and inexact against our reference |
| HSS (DDH/LWE homomorphic secret sharing) | 0-1 | yes | fixed-point only | slower than FHE for this depth | no |

Conclusion for (b): no exact cryptographic route reaches one trip per layer at a cost within 100x of today.
FSS/MPC, the only ones near practical, keep the same number of gateway interactions. The project's design is
already the round-optimal 2PC with preprocessing: a one-time-pad masked linear layer plus a gateway-evaluated
nonlinear layer is exactly an FSS protocol where the "gate" runs in the clear at the party that owns the data.
**Not pursued.** One caveat: if a future pack variant *defines* an integer-only MLP (an I-BERT-style SiLU
LUT and a fixed requant scale per layer instead of the per-row absmax), GC cost drops by about 25x. It is
still about 10^3 x the budget.

## 3. Pipelining and overlap (a), and precompute-ahead (d)

- **Multi-request batching** (implemented): S sessions share each trip, so aggregate throughput is about S x
  single-stream until the gateway CPU saturates. Per-token latency is unchanged. At 17 ms the gateway is idle
  about 90% of each pass (0.5B: 48 x 18 ms of link versus about 30 ms of compute), so a 0.5B Modal gateway
  could carry about 8-16 batched streams at about 1 tok/s each with today's placement.
- **Two-cohort ping-pong** (implemented, `engine_pingpong`): it overlaps cohort A's gateway compute with
  cohort B's trip. The gain is at most `compute / (compute + RTT + o)`: about 5% at 17 ms, about 30-40% at
  1 ms (run 27 measured +1% at loopback with 4 streams). It is a low-RTT tool, not a fix for slow links.
- **Interleaving two conversations' trips**: the same as ping-pong (throughput only). Both conversations see
  the full per-token latency.
- **Chunked/wavefront prefill**: chunk c+1 at layer l only needs chunk c's KV at layer l, so wavefronts
  overlap transfer and compute. TTFT still pays at least `2L x RTT` once (0.82 s for 0.5B at 17 ms). It helps
  only long prompts where bandwidth dominates. This is not an RTT lever.
- **Precompute-ahead**: masks are already prefetched. The useful budget is the gateway's idle time while a
  trip is in flight. Spending it on local linears (section 1) or on a bigger draft model (section 4) is how it
  turns into latency. Nothing in the next trip's input can be computed before the current trip returns (each
  input is a nonlinear function of the previous output and the secret scale).

## 4. Speculation variants (c) under the fixed-rows / fixed-pass rule

Strict speculation (`spec_rows` + `spec_strict`, another agent's work) gives fixed rows per pass and exactly
L passes for L tokens, with the extra passes run as dummies after the answer streams. The user sees
`pass_time / tau`.

| variant | tau (tokens/pass), literature | rows/pass | fit |
|---|---|---|---|
| draft model on gateway (strict, existing) | 2-3 (2.6x fewer trips measured on 7B with a 0.5B draft) | k+1 = 3-5 | in progress elsewhere |
| Lookahead/Jacobi (Fu et al., ICML'24), no draft model | 1.5-2.3x reported (A100, 7B; W=15, N=5 needs ~100+ rows/pass) | W(N-1)+G(N-1): 20-130 | poor: each row costs masks and gateway mask/unmask; with remote MLP linears, 20+ rows/pass is 20x the masks for <2x |
| Medusa heads on the gateway (Cai et al. '24) | 2.2-2.8x (Medusa-2) | tree of 16-64 nodes; fixed-size tree OK under fixed rows | heads are LM-head-sized per head (7B: 4 x 545M fp32) = heavy CPU; EAGLE-style 1-layer drafter is cheaper (EAGLE-3 reports ~3-6x; `eagle_capture_layers` exists) |
| PEARL-style parallel draft/verify (draft the next block while this pass's trips are in flight) | hides draft time behind RTT | same | good fit for slow links: the gateway is idle ~90% of the pass at 17 ms |

Two findings:
- **Strict dummies are expensive on slow links.** After an answer of L tokens, about `L (1 - 1/tau)` dummy
  passes run before a follow-up turn of the same conversation may start. For 0.5B, 17 ms, attention local
  (0.9 s/pass), L = 100 and tau = 2.5, that is **60 passes = about 54 s** of blocking. The fix is placement:
  with decode all-local the provider sees no decode pass, so dummies (local CPU only, 0.12 s each) cost about
  7 s. A follow-up could also allow non-strict speculation when **no** decode step can go remote (cap never
  exceeded). That needs a privacy sign-off and is not implemented.
- Rows are almost free for local linears (GEMV is memory-bound: 4 rows cost 1.05-1.2x one row, measured), so
  speculation composes with local placement at about the full tau.

## 5. Ranked ideas (speedup x feasibility)

| # | idea | speedup (single stream) | feasibility | status |
|---|---|---|---|---|
| 1 | **Generalized exact local placement** (`attention_down`, `all`, `auto_kinds`) + `local_max_rows` so prefill stays remote | 0.5B @17 ms: 0.90 -> 0.12 s/token (**7.6x**, measured local cost); @5 ms 0.33 -> 0.12 (2.7x). 7B @17 ms: 1.22 -> 0.85 (1.4x, `attention_down`) or 0.67 (1.8x, `all`) on a dedicated gateway (est.); 7B @1 ms: none (stay with today's) | high: small change, no protocol change, provider sees strictly less, fewer masks used (down is 64% of the 7B input-mask elements) | **prototyped** (default off), tests pass |
| 2 | **Speculation on top of placement** (strict draft-verify elsewhere; PEARL-style drafting during trips; relax strict dummies only when decode is fully local) | x tau = 2-3 on any row above; e.g. 0.5B @17 ms all-local 0.12 -> ~0.05 s/token | medium: drafter exists elsewhere; the "no remote decode" relaxation needs a privacy review | memo only |
| 3 | **Batching + ping-pong for throughput** (and `local_max_rows` to protect TTFT of today's `auto`) | aggregate about S x at 17 ms; per-token latency 1x; ping-pong <= 5% at 17 ms | high (exists) | recommend config only |
| - | FSS / MPC / GC / FHE for a one-trip MLP | <= 1x (FSS/MPC) or 10^-3-10^-5 x (GC/FHE) | none | rejected (section 2) |

## 6. Prototype

- `MaskedBackend(local_linears=...)` accepts `attention_down` (qkv, o, down local), `all` (every linear
  local) and `auto_kinds` (per-kind calibration at start-up: local iff one row's local time < one RTT, from
  `local_rtt_s`, measured decode trips, or the gateway's start-up dummy-trip measurement). Existing `off`,
  `attention` and `auto` are unchanged.
- `local_max_rows=N`: a step whose calls hold more than N rows in total sends every linear (the masked path
  and the local path give identical bits, so mixing them per step is exact). `None` = today's behaviour.
- Gateway: `GatewayModelConfig.local_linears` accepts the new values and `local_max_rows` is new (also a
  `SETUP_MODEL_OPTION_KEYS` key). The runtime applies the cap and measures RTT for `auto_kinds`.
- Tests (`tests/engine/test_local_linears_r1.py`, 10 cases, all pass): logits from the shared scenario
  (prefill, chunked prefill, decode, lockstep, verify pass with serial tail, mixed batch; sync and async
  split-hop) are bit-identical to `PlainBackend` for `attention_down` and `all`, with and without a row cap.
  The provider receives only gate_up (or nothing) without the cap. With cap 1, a 3-row prefill sends all 4L
  linears and takes 3 mask rows each, and a 1-row decode sends nothing and takes no mask. Policy and
  `auto_kinds` calibration are covered both ways, and the default is unchanged. Existing
  `test_local_linears.py`, `test_model_hooks.py`, `test_speed_wiring.py` and `test_speed_config_defaults.py`
  still pass.
- Not done: a gateway-level end-to-end test of the new config values, an `/admin/status` display of
  `max_rows`, an int16 x int8 1-pass local kernel (about 2x local cost), and a Modal measurement.

Privacy argument: placement is fixed at start-up from public calibration. The row cap is a function of row
counts the provider sees anyway. A local linear sends nothing and takes no mask. With `all` and a cap, the
provider sees no decode passes at all, which hides the generation length (strictly less metadata than today).
Exactness: the local path is the same quantization plus the same exact integer product plus the same range
check as the masked path (already proven for qkv/o; now tested for gate_up and down).

## Sources

- SIGMA: Secure GPT Inference with Function Secret Sharing, PoPETs 2024 — https://eprint.iacr.org/2023/1269 , https://petsymposium.org/popets/2024/popets-2024-0107.php
- Boyle, Gilboa, Ishai et al., Function Secret Sharing for Mixed-Mode and Fixed-Point Secure Computation (Eurocrypt'21) — https://eprint.iacr.org/2020/1392
- Ball, Malkin, Rosulek, Garbling Gadgets for Boolean and Arithmetic Circuits (CCS'16) — https://eprint.iacr.org/2016/969
- Zahur, Rosulek, Evans, Two Halves Make a Whole (half-gates) — https://eprint.iacr.org/2014/756
- PUMA: Secure Inference of LLaMA-7B in Five Minutes — https://arxiv.org/abs/2307.12533
- Fu et al., Break the Sequential Dependency of LLM Inference Using Lookahead Decoding (ICML'24) — https://arxiv.org/abs/2402.02057
- Cai et al., Medusa — https://arxiv.org/abs/2401.10774 ; Li et al., EAGLE-3 — https://arxiv.org/abs/2503.01840
- PEARL: Parallel Speculative Decoding with Adaptive Draft Length — https://arxiv.org/abs/2408.11850
- Internal: STATUS.md (run 27, Modal e2e), research/plans/speed-readiness-plan-2026-11-20.md
