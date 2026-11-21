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
   The provider then sees strictly less, the linear *takes* no mask rows, and the bits are the same (same
   quantization, exact integer product). (Critique: mask *generation* is not reduced today. A slot holds
   masks for every linear, so the local kinds' rows are generated, stored and burned; see section 7.) Today `auto` places only qkv and o. Extending placement to `down` (1
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
| same, after the critique's digit stacking (both digits as rows of one int8 GEMM, one weight pass) | 0.5B gate_up 1.27 / down 0.64; 7B-qkv shape 2.17 ms (1.87-1.97x faster) | | measured by the critique, 1 thread, load 11; bit-identical (tested) |
| local exact GEMM, prefill | 64 G MAC/s per thread | same | measured (7B qkv shape, 64 to 256 rows) |
| o = per-trip cost beyond RTT | 1.2 ms (est.; **the Modal e2e implies ~9 ms**, see below) | 2.2 ms | Kaggle run 27: 1.47 ms remote at loopback + 0.71 ms gateway masking per trip |
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
0.5B "attention local" row once o is about 9 ms. **Critique:** the tables above use o = 1.2 ms, which is
optimistic for Modal. With the Modal-calibrated o = 9 ms (`round_trip_model.py --modal-o`), 0.5B at 17 ms
is 1.28 s/token today, 0.68 `attention_down`, 0.12 `all` (0.07 with digit stacking): **10.7x (18x)**, not
7.6x. The 7B "dedicated" column remains an unmeasured estimate; with only the measured digit stacking
(factor 0.53) 7B at 17 ms is 1.28 today, 1.09 `attention_down` (1.17x) and 1.32 `all` (slower). `auto_kinds` picks this placement from a start-up
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
  single-stream until the gateway CPU saturates. (Critique: local placement moves work onto that CPU. It
  is cheap per row only when sessions run in lockstep in one step; unbatched sessions each pay the full
  weight read, e.g. 7B `all` ~1.2 s of a core per token, so a busy gateway can lose aggregate throughput.) Per-token latency is unchanged. At 17 ms the gateway is idle
  about 97% of each pass (0.5B: 48 x 18 ms of link versus about 30 ms of compute), so a 0.5B Modal gateway
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
| PEARL-style parallel draft/verify (draft the next block while this pass's trips are in flight) | hides draft time behind RTT | same | good fit for slow links: the gateway is idle ~97% of the pass at 17 ms |

Two findings:
- **Strict dummies are expensive on slow links.** After an answer of L tokens, about `L (1 - 1/tau)` dummy
  passes run before a follow-up turn of the same conversation may start. For 0.5B, 17 ms, attention local
  (0.9 s/pass), L = 100 and tau = 2.5, that is **60 passes = about 54 s** of blocking. The fix is placement:
  with decode all-local the provider sees no decode pass, so dummies (local CPU only, 0.12 s each) cost about
  7 s. (Critique: the provider still sees the gap between one turn's prefill trips and the next turn's,
  which is at least the decode time, about L x 0.12 s. All-local decode removes the exact pass count, not
  a timing bound on L; that bound is why the relaxation needs the review.) A follow-up could also allow non-strict speculation when **no** decode step can go remote (cap never
  exceeded). That needs a privacy sign-off and is not implemented.
- Rows are almost free for local linears (GEMV is memory-bound: 4 rows cost 1.05-1.2x one row, measured), so
  speculation composes with local placement at about the full tau.

## 5. Ranked ideas (speedup x feasibility)

| # | idea | speedup (single stream) | feasibility | status |
|---|---|---|---|---|
| 1 | **Generalized exact local placement** (`attention_down`, `all`, `auto_kinds`) + `local_max_rows` so prefill stays remote, + digit stacking in `exact_int_matmul` (critique, 1.9x local cost) | 0.5B @17 ms: 0.90 -> 0.12 s/token (**7.6x** at o = 1.2 ms; 10.7x at the Modal o = 9 ms; 0.07 s with stacking); @5 ms 0.33 -> 0.12 (2.7x). 7B @17 ms: 1.22 -> 0.85 (1.4x, `attention_down`) or 0.67 (1.8x, `all`) on a dedicated gateway (est.); 7B @1 ms: none (stay with today's) | high: small change, no protocol change, provider sees strictly less; mask *takes* drop, but mask *generation* only drops after a factory change (section 7) | **prototyped** (default off), tests pass |
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
provider sees no decode passes at all, so it no longer sees the exact pass count; a timing bound on the
generation length remains (section 4). Everything the provider sees is a deterministic function of what it
sees today with the same placement off, so no new information leaks.
Exactness: the local path is the same quantization plus the same exact integer product plus the same range
check as the masked path (already proven for qkv/o; now tested for gate_up and down).

## 7. Adversarial review (2026-11-21)

Checked: every number in sections 1 and 5 (recomputed with `round_trip_model.py`), the code diff, the tests
(10 + 6 new, plus `test_local_linears.py`, `test_backends.py` and `test_model_hooks.py`: 87 passed, 12 CUDA
skips), SPEC 3.2/3.4/7/9.4a. Exactness holds: the local path is `quantize_activations` + exact int8-digit GEMM
(int32 accumulation, max |partial| 128 x 128 x 18944 < 2^31) + the same range check. The privacy argument is
correct: the cap is a function of the step's total rows, which a remote step reveals anyway, and placement is
fixed at start-up. Corrections and additions:

1. **Masks.** SPEC 7: a slot holds `r` and `W r` for every linear of the pack, and the gateway reserves
   `prompt + max_tokens` slots. Rows of local kinds are reserved, never taken, and burned at the end. Today's
   `attention` already does this. So "down is 64% of the input-mask elements, so local down saves masks"
   holds only for takes, not for factory or encrypted-maskgen work, storage or download. **Follow-up (large,
   cheap):** let the factory emit segments for a public subset of linears (the header already lists
   `linears [{id, m, n}]`), chosen per deployment from the placement. With `attention_down`, 7B generation per
   slot shrinks from Σn = 831,488 to 28 x 3584 = 100,352 (8.3x fewer `r` elements; `W r` from Σm 1,390,592
   to 28 x 37,888 = 1,060,864). That also cuts the work of the encrypted (MLWE) mask path.
2. **Per-trip overhead.** The model's o = 1.2 ms for 0.5B contradicts its own Modal calibration (o ~ 9 ms).
   With 9 ms the 0.5B gain at 17 ms is 10.7x, not 7.6x (section 1).
3. **Local kernel 2x now, not "not done".** `exact_int_matmul` ran one int8 GEMM per activation digit,
   reading the weight twice. Stacking the digits as rows of one GEMM gives 1.87-1.97x at 1 row (measured,
   1 thread), bit-identical. Implemented in this branch (CPU path only; CUDA unchanged), tested against a
   float64 reference at A = 12 and 20 with extreme values. It also speeds up today's `attention` placement and
   `PlainBackend`.
4. **7B on a dedicated gateway is unmeasured.** The 4x factor was an estimate. Stacking supplies a measured
   ~1.9x, so the column now assumes another ~2x from cores and bandwidth. A 7B `all` decode reads 6.5 GB of
   int8 per token, so at a realistic 10-20 GB/s the floor is 0.33-0.65 s plus F. Do not promise 7B gains until
   a dedicated box is measured.
5. **"Hides the generation length" was overstated** (sections 4 and 6 corrected): turn-boundary timing still
   bounds L. Short prompts of at most `local_max_rows` rows also become invisible. That is still a function of
   today's view, so it is fine.
6. **SPEC 9.4a contract.** Settings in `SETUP_MODEL_OPTION_KEYS` must be "tested gateway-level against the
   plaintext reference". The branch adds `local_max_rows` and the new `local_linears` values without that
   test or a SPEC 9.4a entry. Both are required before anyone presets them (still default off, so this does
   not block the prototype).
7. **`auto_kinds` calibration** runs on an idle CPU at start-up, so under load it underestimates local cost.
   It also ignores gateway CPU contention between unbatched sessions. Prefer an explicit placement in
   production; keep `auto_kinds` for demos.
8. **Garbled circuits, an extra reason.** With arithmetic labels, multiplying by a public weight is free for
   the evaluator only. The garbler must compute `W x A_0` on the zero labels (a lambda-bit GEMM on the
   gateway, at least as expensive as the local exact GEMM). So GC does not offload the linears at all.
9. SIGMA's 27 s for Llama2-7B is its next-token-generation figure (RWC 2024 slides; the paper abstract gives
   44 s for 13B). OK as cited.

Verdicts: (1) placement plus `local_max_rows` plus stacking: **pursue** (0.5B and Modal now; 7B only after a
measured dedicated gateway), with the gateway-level test, the SPEC entry and the factory subset follow-up.
(2) Speculation on placement: **pursue** the composition. **Park** the non-strict relaxation until a privacy
review addresses the timing bound. (3) Batching and ping-pong: **park** (config only, no latency gain;
`local_max_rows` should be the default companion of today's `auto` for long prompts, which is a separate
small fix).

## Sources

- SIGMA: Secure GPT Inference with Function Secret Sharing, PoPETs 2024 — https://eprint.iacr.org/2023/1269 , https://petsymposium.org/popets/2024/popets-2024-0107.php
- Boyle, Gilboa, Ishai et al., Function Secret Sharing for Mixed-Mode and Fixed-Point Secure Computation (Eurocrypt'21) — https://eprint.iacr.org/2020/1392
- Ball, Malkin, Rosulek, Garbling Gadgets for Boolean and Arithmetic Circuits (CCS'16) — https://eprint.iacr.org/2016/969
- Zahur, Rosulek, Evans, Two Halves Make a Whole (half-gates) — https://eprint.iacr.org/2014/756
- PUMA: Secure Inference of LLaMA-7B in Five Minutes — https://arxiv.org/abs/2307.12533
- Fu et al., Break the Sequential Dependency of LLM Inference Using Lookahead Decoding (ICML'24) — https://arxiv.org/abs/2402.02057
- Cai et al., Medusa — https://arxiv.org/abs/2401.10774 ; Li et al., EAGLE-3 — https://arxiv.org/abs/2503.01840
- PEARL: Parallel Speculative Decoding with Adaptive Draft Length — https://arxiv.org/abs/2408.11850
- SIGMA RWC 2024 slides (next-token table) — https://iacr.org/submit/files/slides/2024/rwc/rwc2024/26/slides.pdf
- Internal: STATUS.md (run 27, Modal e2e), research/plans/speed-readiness-plan-2026-11-20.md
