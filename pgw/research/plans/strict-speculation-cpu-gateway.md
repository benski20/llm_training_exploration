# Strict speculation on a CPU-only, weight-light gateway (7B target, 0.5B draft)

Date: 2026-11-22. Scope: make the reviewed strict speculation mode (L3: `spec_rows` R + `spec_strict_passes` +
`spec_mask_rows` C / `spec_mask_credit` B, `research/plans/speculation-no-compromise.md`) work with the
CPU-only gateway of `research/plans/weight-free-cpu-gateway-7b.md` (every linear remote, `float_ops = native`,
no gateway GPU, no target weights on the gateway), with a Qwen2.5-0.5B draft on the gateway's CPU cores.
Local engineering only (4 shared vCPUs, load 5-15 from other agents during this work; no Modal, no Kaggle).

Rules kept: outputs bit-identical to the plaintext reference; the provider's view (rows per pass, items,
pass count, mask demand, gateway turnaround) independent of the acceptance; new settings default off.

## 0. Summary

| item | result |
|---|---|
| exactness, CPU-only config | **proven by tests**: with every linear remote and native float ops, each verify row of an R-row pass (R = 2, 4, 8; with padding rows) equals single-row decoding bit for bit; padded strict speculation (R/C = 4/4, 4/2, 2/1.5; greedy and seeded; five draft sources incl. the pipelined drafter) equals plain decoding bit for bit; the provider sees R rows per decode pass and exactly L passes. Real 0.5B shapes: R = 2, 4, 8 verify rows identical to single-row decoding with both float paths (`benchmarks/spec_rows_exactness_check.py`). |
| the "GPU target not bit-identical" refusal | applies to the **target device**, not the float path: `float_ops = native` only runs on a CPU float32 model (`set_float_ops` refuses otherwise), and on the CPU both float paths verify exactly (measured above). Nothing to relax: the CPU-only configuration (device `cpu`, `local_linears = off`, `float_ops = native`, `spec_mask_rows`, a CPU draft) was already accepted and is now tested end to end through the gateway. |
| draft on CPU | new `NativeInt8DraftBackend` + `native_int8_lm_head` (`pgw_fast.local_linear_into`, int8 weights resident, columns over the native pool), gateway `draft_backend = "native_int8"`. Qwen2.5-0.5B: **75 ms per draft token on 1 core, 46-54 ms on the 4 shared vCPUs** vs 350-390 / 106-112 ms with today's torch weight-only int8 (3.5-5x faster); same bits for any thread count (tested). Memory-bound (0.49 GB of int8 per token): about 10-12 ms per token expected on 16 server cores (~50 GB/s), 20 ms if half that. |
| pipelining | new `PipelinedDrafter` (gateway `spec_pipeline_steps = m`): drafts the next block in a background thread while the pass is in flight, assuming full acceptance and the draft's own bonus guess; exactly m draft steps between passes whether the guess held or not, so the gap the provider sees does not depend on the acceptance (tested: same step counts and rows fed, hit or miss). m = 0 takes the draft off the critical path at a lower tokens-per-pass. |
| masks | the public charge schedule is unchanged by pipelining (it depends only on the pass index): **R = 4, C = 4, B = 160: 2.50 slots per generated token** (2.84 tokens per pass, 1.5B/0.5B simulation of L3); C = 3, B = 96: 1.91 slots per token at 2.22 tokens per pass. |
| 25 tokens/s target (section 7) | smallest config with margin: **R = 2, C = 2, pipelined m = 0 at 0.3 ms trips: 29.3-31.7 tokens/s (alpha 0.70-0.78), <= 2 slots per token**, robust to a 20 ms draft step; 0.4 ms trips need R = 3 (26-30); 0.5 ms trips reach 25 only at alpha ~0.78. |
| 7B tokens/s (model, section 5) | at 0.3 ms trips on a 16-core gateway with a 12 ms draft step: plain 21.9; sequential drafting R = 4: 28.6-32.3; **pipelined R = 4, m = 0: 33.0-38.6** (alpha 0.70-0.78). At 0.5 ms trips: 22.8-28.4. On this 4-vCPU box the draft (50 ms per step) makes speculation slower than plain. |

## 1. Exactness with every linear remote and native float ops

* What has to hold: verify row j of a pass (rows `p, d1..dk`, `serial_tail = k`, padding rows after the real
  ones) must produce exactly the logits single-row decoding produces for the same prefix. Linears: every
  linear is a masked trip whose result is the exact integer product with the row's own activation scale, so
  rows are independent by construction (`MaskedBackend`, row-wise prepare/finish; padding rows only append).
  Float ops: the native path computes every row from its own inputs and the cache prefix in a fixed order
  (`layer.rs`, `attn.rs`), and the serial tail attends row j only to rows before it. LM head: the native
  head kernel is row-wise (`RowLogits`).
* Tests (`tests/engine/test_spec_cpu_gateway.py`):
  `test_native_verify_rows_equal_single_row_decode` (R = 2, 4, 8, 0 or 1 padding rows; against masked
  single-row decoding and against the plain native reference), `test_cpu_gateway_strict_speculation_is_exact`
  (30 cases: R/C 4/4, 4/2, 2/1.5 x greedy/seeded x draft sources: native int8 self-draft, native int8 other
  model, pipelined m = 0 self-draft, pipelined m = 1 other model, prompt lookup; tokens, text, finish reason
  and logprobs equal plain decoding; `0.qkv` and `1.down` items have R rows on every decode pass and L passes
  in all), gateway level `test_cpu_only_gateway_strict_speculation_native` (sequential, m = 0, m = 1; same
  bits as the same gateway without speculation; R rows per pass, same pass count).
* Real shapes (Qwen2.5-0.5B pack, exact integer linears = the masked path's integers): verify rows of R = 2,
  4, 8 equal single-row decoding bit for bit, with `float_ops = torch` (4 torch threads) and `native`
  (`benchmarks/spec_rows_exactness_check.py`). Exactness of the CPU target is therefore not what limits the
  CPU-only gateway; the native path additionally does not depend on the ISA or torch's thread count.
* Config: the refusal `speculation ... needs device='cpu'` is about float kernels of a CUDA target (Kaggle T4
  greedy answers diverged); `float_ops = native` cannot be combined with a CUDA target at all, so there is no
  "native on GPU" case to relax. `test_cpu_only_strict_speculation_config_is_accepted` pins the accepted
  combination.

## 2. The draft model on the gateway's CPU

`NativeInt8DraftBackend(pack, ring)`: per linear the pack's int8 weight (transposed once, 1 byte per weight
resident; no float copy) and `pgw_fast.local_linear_into` (the target ring's constant-time quantizer, an
exact int16 x int8 product with int64 flushes, columns or input-feature blocks over the native pool,
rescale). `native_int8_lm_head` quantizes the draft's (tied) head per row to int8 and runs it on the same
kernel. Draft float ops on the native path. Approximate (int8 activations), which is fine: the draft only
proposes, the masked target decides every token.

Measured (`benchmarks/draft_cpu_bench.py`, Qwen2.5-0.5B, 96-token prompt, 1-row steps, median):

| backend | cores | ms per draft token | 4-row step |
|---|---|---|---|
| torch weight-only int8 (today's `Int8DraftBackend` + `int8_lm_head`) | 1 | 390 | 1056 |
| torch weight-only int8 | 4 threads | 112 | 294 |
| **native int8 (this branch)** | 1 (`PGW_POOL_WORKERS=0`) | **75** | 106 |
| **native int8** | 4 shared vCPUs (`PGW_POOL_WORKERS=3`) | **46-54** | 91 |

Where the time goes (py-spy, 1 core): 88% in the native kernel. Warm-cache microbenchmarks give 18 ms of
layer linears + 8.7 ms of head per token, so the full step (75 ms) is DRAM-bound: 0.49 GB of int8 weights
per token (0.36 GB of layers, 0.14 GB of head) at about 6.6 GB/s on one core and 9-10 GB/s with all four
shared vCPUs of this VM (the extra workers barely help: the VM's memory bandwidth, not compute, is the
limit). **Extrapolation to 16 cores** (honest range): a 16-vCPU server streams 40-80 GB/s with 8+ cores,
so 6-12 ms of weights plus ~2 ms of float ops and dispatch: **~10-12 ms per draft token, 20 ms if the
gateway's bandwidth is half that or shared with the trip work**. Multi-row steps are compute-bound in this
kernel (4 rows cost 3.3x one row warm), which matters only for the catch-up step (1 row in this mode) and for
a hypothetical tree of hypotheses (section 3).

Storage/RAM: the 0.5B draft pack is 0.87 GB on disk (int8 linears 0.36 GB + bf16-exact embedding table);
resident: 0.36 GB int8 layers, 0.14 GB int8 head, the float32 embedding/head table 0.54 GB (0.27 GB with
`PGW_EMBED_COMPACT=1`-style bf16 storage, not wired for the draft). No target weights on the gateway.

Determinism and timing: the kernel's sums are exact, so the draft's outputs do not depend on the thread
count or the order of the column split (`test_native_int8_draft_same_bits_any_thread_count`); every loop
runs over public sizes (rows, n, m), so a draft step's time depends only on the rows fed and the context
length, not on token values. With the fixed step schedule of the drafters (k steps per pass, each feeding one
row after the first), the draft's contribution to the inter-pass gap is a constant up to the documented
residuals (catch-up rows; attention over a growing context).

## 3. Pipelined drafting without an acceptance signal

The obvious pipeline ("draft the next block while the pass is in flight, assuming full acceptance; if that
held, skip drafting; else draft after the pass") **leaks**: the gap between two passes is then ~0 after a hit
and k draft steps after a miss, a direct timing signal of the acceptance. Rejected.

`PipelinedDrafter(model, post_steps=m)` keeps the gap acceptance-independent:

* **background block** (started when `propose` returns, runs while the generator sends the pass's trips):
  exactly `k + 1` draft steps per sequence on the hypothesis "every draft accepted and the target's bonus
  token equals the draft's next token (the guess)": step 0 feeds the unseen context and this pass's drafts
  and yields the guess; steps 1..k feed the guess and the new drafts. Always the same number of steps.
* **foreground** (next `propose`, between passes): exactly `m` draft steps per sequence. Hit (the new context
  is the hypothesis): the proposal is the block's k drafts, and the m steps are dummy steps (filler token,
  cache rolled back). Miss: the m steps draft afresh from the real context (1 row fed: the bonus token), and
  the pass carries m drafts (the rest are padding rows, indistinguishable on the wire).
* The generator then waits for nothing else: `propose` joins the previous block first (same work every
  pass; if the block outlasts the pass the extra wait is the same every pass).

Privacy argument: the provider sees R rows and one item per linear per pass, L passes, and the public mask
charge as before (unchanged code paths). The gateway's work between passes is m one-row draft steps plus the
join, identical for hits and misses (`test_pipelined_drafter_work_does_not_depend_on_hits`: foreground steps
= m per call, background steps = k + 1 per call, every step one row, for a self-draft that hits and a foreign
draft that misses). The background block runs concurrently with the trips on other cores; its CPU and memory
traffic is the same every pass (k + 1 one-row steps), so it adds a constant load, not an acceptance signal.
Residuals (same class as `BuiltinDrafter`): attention cost grows with the context during the answer and stays
flat during dummy passes (pre-existing, L3 section 2.7); the native pool is shared with the trip kernels
(`Pool::run` falls back to the calling thread when busy), which couples the two loads' timing but not to the
acceptance; a dedicated draft pool would make the overlap cleaner (follow-up).

Hit rate: a hit needs all drafts accepted and the bonus guessed, about `alpha^(k+1)` after a full pass and
`alpha^(m+1)` after an m-draft pass. With alpha = 0.78, R = 4: tokens per pass 2.03 (m = 0), 2.31 (m = 1),
2.86 sequential (two-state chain in `spec_7b_cpu_gateway_model.py`). m = 0 trades 30% of the tokens per pass
for a critical path without any draft step; it wins when a draft step costs more than about 10% of a pass.
Sampled requests: the drafter guesses greedily (exact outputs; lower acceptance than the coupled sampler).

Not done (follow-ups): a hypothesis tree in the background block (one chain per acceptance point, guessing
the draft's second choice after a rejection; the draft is memory-bound, so R hypotheses cost about one chain
on a server, but 3.3x here); a dedicated draft thread pool; prompt lookup inside the pipelined drafter.

## 4. Masks per generated token

The pipelined drafter changes only which tokens fill the rows. The charge schedule
`prompt + j + min(B, floor((C - 1) j))` depends on the pass index j only, so slots per token are those of
L3 (`benchmarks/spec_charge_sim.py` on the 1.5B/0.5B exact answers, 24 chat prompts, up to 128 tokens):

| config | tokens / pass | slots / token |
|---|---|---|
| no speculation | 1.00 | 1.00 |
| R = 4 masked fillers (pre-L3) | 2.84 | 3.96 |
| **R = 4, C = 4, B = 160** | 2.84 | **2.50** |
| R = 4, C = 3, B = 96 | 2.22 | 1.91 |
| R = 4, C = 4, B = 128 | 2.62 | 2.24 |

With m = 0 pipelining a miss pass carries no drafts, so fewer real rows are spent per pass and the credit B
lasts more passes (the charge is the same; only the real rows used fall). For long answers (L = 512) the
same B gives 1 + 160/512 = 1.31 slots per token but speeds up only the first ~53 passes. For a 7B (6.17 MB of
`W r` per slot) 2.5 slots per token at 30 tokens per second is 460 MB/s of mask supply: the mask factory, not
the gateway, is the next bottleneck of this configuration.

## 5. Per-token model, 7B, CPU-only gateway (`benchmarks/spec_7b_cpu_gateway_model.py`)

Inputs: 113 sequential trips per pass (28 x 4 + the masked LM head; a CPU bfloat16 head would read 1.09 GB
per pass); gateway CPU per pass measured here with `native_trip_rows_bench.py --rows R` at 7B shapes: main
core 15.4 ms at R = 1 (r keystream 3.8, W r 1.2, prepare 6.0, finish 4.4) and 59.1 ms at R = 4 (r 26.9, W r
4.2, prepare 16.5, finish 11.5); with the r keystream prefetched on a spare core (it does not depend on the
activations) 11.6 and 32.3 ms; plus 7.2 ms of orchestration for the native per-trip path being built (~0.2 ms
per 1-row trip). 16-core server: core 1.5x faster, extra rows mostly overlapped on spare cores, 4 ms of
orchestration. Draft step 50 ms (this box), 12 ms (16-core server), 20 ms (slow server). Acceptance: i.i.d.
alpha = 0.78 reproduces the measured 2.84 tokens per pass of the 1.5B/0.5B pair at R = 4 (upper reference);
0.70 (2.53 tokens per pass at R = 4) is the central assumption for 7B/0.5B: no 7B acceptance was measured in
this repository, and none is quoted here as published.

| trip | gateway | draft step | alpha | plain | sequential, best R | sequential R = 4 | pipelined, best (R, m) | pipelined R = 4, m = 0 |
|---|---|---|---|---|---|---|---|---|
| 0.3 ms | this box | 50 ms | 0.70 / 0.78 | 19.0 | 15.5 / 16.2 (R = 2) | 11.3 / 12.8 | 14.0 / 15.2 (2, 0) | 8.7 / 10.2 |
| 0.3 ms | 16 cores | 12 ms | 0.70 | 21.9 | 29.5 (R = 3) | 28.6 | **33.0 (4, 0)** | 33.0 |
| 0.3 ms | 16 cores | 12 ms | 0.78 | 21.9 | 32.3 (R = 4) | 32.3 | **38.6 (4, 0)** | 38.6 |
| 0.3 ms | 16 cores | 20 ms | 0.70 / 0.78 | 21.9 | 25.0 / 26.5 | 22.5 / 25.4 | 29.3 / 31.7 (2, 0) | 21.7 / 25.4 |
| 0.5 ms | 16 cores | 12 ms | 0.70 / 0.78 | 14.7 | 22.8 / 25.8 | 22.8 / 25.7 | 23.4 / 28.4 | 23.1 / 27.0 |
| 0.5 ms | 16 cores | 20 ms | 0.70 / 0.78 | 14.7 | 19.4 / 21.2 | 18.7 / 21.2 | 22.1 / 25.4 | 21.7 / 25.4 |

Reading: (1) speculation pays only with a draft step well under a pass (here: under ~20 ms against a
~50 ms pass); on a 4-vCPU gateway it loses to plain decoding. (2) With a 16-core gateway, 0.3 ms trips and
a ~12 ms draft step, **30+ tokens/s is reachable only with the pipelined drafter (33-39)**; sequential
drafting gives 29-32. (3) At 0.5 ms trips the ceiling is ~23-28 tokens/s with this draft. (4) The plain
CPU-only gateway (no speculation) is capped at ~22 tokens/s at 0.3 ms trips even with the native trip path.

## 6. Local end-to-end measurement (0.5B target, real weights, this box)

`benchmarks/spec_cpu_gateway_bench.py`: the real `Generator`, padded strict mode, `MaskedBackend` with
`local_linears = off`, native float ops, in-process provider (its GEMMs on the same 4 vCPUs), the 0.5B as its
own native-int8 draft (the only Qwen2.5 checkpoint at or below the target here, so its acceptance is an
upper bound and its cost equals the target's: this run shows mechanics, not a speed-up). Two chat prompts,
24 greedy tokens each, load 15-18 from other agents:

| config | identical to plain | tokens / real pass | answer tok/s | ms / real pass | draft ms between passes | pipelined: hit rate, block ms, wait ms | drain |
|---|---|---|---|---|---|---|---|
| plain | yes | 1.00 | 3.03 | 330 | 0 | | 0 |
| sequential R = 2, C = 2 | yes | 1.85 | 3.63 | 529 | 77 | | 11.2 s |
| pipelined R = 2, C = 2, m = 0 | yes | 1.85 | 3.43 | 559 | 3.9 (bookkeeping) | 0.48, 229, 0.0 | 11.5 s |
| sequential R = 4, C = 4 | yes | 3.20 | 4.54 | 779 | 173 | | 22.2 s |
| pipelined R = 4, C = 4, m = 0 | yes | 2.67 | 4.68 | 615 | 2.3 | 0.26, 369, 0.0 | 18.8 s |

Every run is token-identical to plain decoding and the provider saw 48 passes in each (L, strict). The
pipelined drafter removed the draft from the critical path (77 -> 4 ms and 173 -> 2 ms between passes) and
its block always finished under the pass (wait 0), but here the block competes for the same 4 vCPUs as the
gateway and the in-process provider, so the pass itself slowed by about as much as was saved (R = 2) or
somewhat less (R = 4: 615 vs 779 ms per pass at 2.67 vs 3.20 tokens per pass). The hit rates (0.48 at R = 2
over answer and dummy passes; dummy passes on random filler always miss) and the drop in tokens per pass at
R = 4 match the two-state model. The gain needs spare cores and memory bandwidth for the draft, i.e. the
16-core gateway of section 5; on a 4-vCPU box sharing its CPU with the provider it is a wash.

## 7. Recommended configuration (7B, CPU-only gateway, target 25 tokens/s streaming)

The target is 25 tokens/s (~40 ms per generated token). Plain decoding cannot reach it on a CPU-only gateway
(21.9 at 0.3 ms trips even with the native trip path), so strict speculation is needed; the smallest R
(fewest rows per trip and fewest mask slots) that clears it with ~10% margin (27.5) is chosen
(`spec_7b_cpu_gateway_model.py`, second table):

| trip | 16-core gateway, draft step 12 ms (20 ms) | alpha 0.70 | alpha 0.78 |
|---|---|---|---|
| 0.3 ms | sequential R = 2 | 28.4 (25.0) | 29.7 (26.2) |
| 0.3 ms | **pipelined R = 2, m = 0** | **29.3 (29.3)** | **31.7 (31.7)** |
| 0.4 ms | pipelined R = 3, m = 0 | 26.2 (26.2) | 29.7 (29.7) |
| 0.5 ms | pipelined R = 3 / R = 5, m = 0 | below 25 | 25.1 / 28.0 |

R = 2 at 0.3 ms trips is the recommendation: one draft row per pass, the pipelined block is only 2 draft
steps (24-40 ms) and fits under a ~46 ms pass even with a slow draft, so the margin does not depend on the
draft's speed. Mask cost: C = 2 (every pass may carry its draft; L3 simulation 1.78 tokens per pass at 1.98
slots per token for R = 2 masked fillers, the same charge with padding), no credit cap needed: **<= 2 slots
per generated token** (12.3 MB of `W r` per token at 7B, 310 MB/s at 25 tokens/s).

```json
{"device": "cpu", "local_linears": "off", "float_ops": "native",
 "draft_pack_path": "<Qwen2.5-0.5B-Instruct pack>", "draft_backend": "native_int8",
 "spec_rows": 2, "spec_strict_passes": true, "spec_mask_rows": 2,
 "spec_pipeline_steps": 0, "freivalds_rate": 1.0}
```

plus the masked LM head (library opt-in, `pgw.engine.masked_head`), the native per-trip path when it lands,
and a gateway of at least 16 cores (the draft needs ~40-80 GB/s of memory bandwidth next to the trip work).
If trips are 0.4 ms: `spec_rows = 3, spec_mask_rows = 3, spec_mask_credit = 96` (pipelined m = 0; about
1.9 slots per token by the L3 simulation of C = 3, B = 96). At 0.5 ms trips 25 tokens/s needs alpha near
0.78 and R = 5 (no margin at alpha 0.70); R = 4, C = 4, B = 160 (2.50 slots per token) is the strongest
setting there. Everything stays opt-in; defaults unchanged.

## 8. Open items

* Measure on a 16-core host with a GPU provider at 0.3 ms trips (Modal): draft ms per step under the trip
  load, hit rate, tokens per pass of 7B with the 0.5B draft (the alpha assumption).
* The dummy-vs-real timing residual of L3 (per-row take vs padding row) must still be re-measured with the
  real mask store; pipelining does not change it.
* Dedicated draft thread pool; r keystream prefetch for padding rows and real rows on spare cores; a
  hypothesis tree in the background block; bf16 storage of the draft's embedding table.
