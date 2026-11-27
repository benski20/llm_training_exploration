# Consumer devices over the internet: fusing linear layers across RMSNorm (and the trips x RTT model)

Date: 2026-11-27. Scope (from the coordinator): consumer devices on the public internet talking to our hosted GPUs,
with the core principle intact (plaintext only on the device; the provider sees only masked data; every result
integrity-checked). This review owns **layer fusion across RMSNorm** (o -> gate_up, down -> next qkv, the head and
the first qkv), the **parallel attention + MLP block** variant of it (one trip per layer), the provider-side cost of
the fused items, and the shared **trips x RTT model** (`benchmarks/round_trip_model.py --consumer / --twenty`, with
tokens per pass and rows per pass as parameters so the speculation, architecture and WAN studies plug in). Other
agents own speculation, model architecture changes (converting Qwen to parallel blocks) and WAN transport.

Everything here is CPU-only (4-vCPU shared box) except the H100 numbers, which are the 2026-11-23 Modal measurements
already in the repo (`benchmarks/results/fused2/pair_op_planes_h100_2026-11-23.json`). No GPU was used in this work.

## 0. Summary

| | trips per pass | 0.5B / 7B / 32B / 72B | status |
|---|---|---|---|
| today, client library (4 linears per layer + masked int8 head) | 4L + 1 | 97 / 113 / 257 / 321 | default |
| fused2 (o + gate_up, down + next qkv per trip), head as its own K32 trip | 2L + 2 | 50 / 58 / 130 / 162 | exists (2026-11-22/23), gateway service only |
| **fused2 + head folded into the last trip + layer 0's qkv local** (this work) | **2L** | **48 / 56 / 128 / 160** | **prototype, default off, exact, tested** |
| **parallel attn + MLP blocks, fused** (this work; needs a parallel-block model) | **L + 1** | **25 / 29 / 65 / 81** | **prototype, default off, exact, tested** (tiny packs and 0.5B shapes read as parallel blocks) |
| below 2L for a sequential (Qwen/Llama) layer | - | - | **impossible** while the client evaluates softmax and SiLU and the provider sees only one-time-padded values (section 5) |

1. **The composite matrices of the request are not needed, and they would cost double.** The provider can apply two
   linear maps in sequence within one trip (`H = R + W1 x`, then `Z = W2' H`), so `qkv(norm(h + down(a)))` needs no
   product matrix `W_qkv diag(g) W_down`. Composites would add +90-99 % of the model's int8 weights to the provider
   (6.3 GB at 7B, 31 GB at 32B, 63 GB at 72B), double the bytes read per trip, double the mask factory products and
   the pks-v3 hint, and add an extra rounding; the chained form reads each weight once. This chained form is what the
   repo's fused2 (WP-B4, `src/pgw/engine/fused2.py`) already implements, with a mask reparametrization that removes
   the product chain from the masks (section 2).
2. **What was missing for a weight-free consumer client, now prototyped:** the LM head folded into the last trip as
   its second stage (`down(L-1) + head`: the final RMSNorm folds like every other norm) and layer 0's qkv computed
   locally (a public 1 / 17 / 37 / 84 MB int8 matrix at 0.5B / 7B / 32B / 72B; nothing can fuse with it: its input
   is the embedding of the token just sampled). Together: exactly **2L trips per pass, 2.02x fewer than today**.
   The parallel-block variant halves that again to L + 1, but only for models built with parallel blocks.
3. **Quality (Qwen2.5-0.5B-Instruct, 4088 wikitext-2 tokens, the engine's exact integer pipelines):** the fused
   reference with the folded head has ppl 19.89 vs 19.85 for the current pipeline (standard + masked int8 head),
   top-1 = float 96.3 % vs 96.7 %, and picks the same top-1 token as the current pipeline at **97.2 %** of positions
   (the current pipeline vs today's float-head reference: 98.6 %). Every trip row kept all 12 activation bits.
   It is a new pinned reference (like the masked head was), not bit-identical to today's.
4. **Privacy and integrity are unchanged:** the provider's view per trip is uniform ring elements of public shapes
   (`(R + v - hmul W1 r_x, x + r_x)`, `v` and `r_x` fresh and independent), each mask row used once; every returned
   value (`H`, `Z`, the head's logits, the rows without logits) passes the range check and a Freivalds check over
   `F_(2^61-1)` (60.8 bits per vector) on the unmasked integers before use. p31 cannot be used: the fused outputs
   need 41-49 bits (ring K56C53), so the check field must exceed `2^54`. Tests prove each property (section 8).
5. **The costs are not free:** K56 instead of K32 (1.75x bytes per element on the wire and in the mask store); the
   pks-v3 encrypted factory needs digit decomposition at K56 (7 K32 products per mask: about **6x client factory CPU
   and 7x provider maskgen GPU per slot**, which bounds users per GPU); the H100 pair op costs 1.22x the K32 singles
   it replaces at 1 row (measured, 7B) and today's kernels scale badly with rows (5.4x at 8 rows).
6. **The 20 tok/s target for 32B over the consumer internet is out of reach for any per-layer protocol.** 20 tok/s
   is 50 ms per token; with T tokens per pass and 2L = 128 trips at 32B every trip (RTT + device + client work) must
   fit in `50 T / 128` ms: 0.39 ms at T = 1, 1.6 ms at T = 4, 3.1 ms at T = 8. After device and client work (target
   kernels, 2 H100s) the largest RTT that meets 20 tok/s is **0.1 / 0.4 / 1.1 / 2.3 ms at T = 1 / 2 / 4 / 8** (section
   9); parallel blocks raise it to 0.3 / 1.0 / 2.3 / 4.8 ms. At RTT 20 ms, 32B gives about 0.4 tok/s (T = 1) to
   1.5 tok/s (T = 4) fused, 0.75-2.9 with parallel blocks. 20 tok/s needs a provider in the same metro (RTT <= ~1-2
   ms) and T >= 4-8, or a model that is not 64 layers deep.

**Recommendation.** Keep the fused schedules opt-in; make **fused2 with the folded head + local qkv0** the high-RTT
schedule of the client library (2x fewer trips; at RTT >= 5 ms it is 1.8-2x faster per token at every size,
section 9), after the production pieces of section 10: the K56 pks-v3 digit factory in the SDK, the head-fold
item on the lane, the native driver's head fold, and the GPU validation of section 11. Price the K56 mask cost into
the product (users per GPU) or fund a direct K56 HE scheme (`q >= 2^80`, about 2x instead of 7x; new cryptography,
needs review). Treat **parallel blocks** as the architecture lever (L + 1 trips) for the model team; the protocol
and engine side is ready (prototype). Do not build composite matrices.

## 1. Baseline: what a pass costs today

The client library's decode pass: per layer qkv, o, gate_up, down (4 trips), then the masked int8 LM head (1 trip):
`4L + 1` sequential trips. Between them the client does RMSNorm (+ requantization), RoPE + softmax attention, the
residual adds and SiLU * up. Every trip waits for the previous one: the input of trip `t + 1` is a nonlinear function
of the output of trip `t` and the client's secret per-token scales.

## 2. The maths: fusing across RMSNorm

### 2.1 The residual stream, exactly
Sequential layer `l` (Qwen2 / Llama), residual `h` entering the layer:

    a  = Attn(RoPE(qkv(n1(h))))            n(x) = g * x / rms(x),  rms(x) = sqrt(mean(x^2) + eps)
    h1 = h + W_o a
    [gate | up] = W_gu n2(h1) = (1 / rms(h1)) * W_gu diag(g2) h1                       (*)
    m  = silu(gate) * up
    h2 = h1 + W_down m
    qkv_next = W_qkv' n1'(h2) = (1 / rms(h2)) * W_qkv' diag(g1') h2 (+ bias)

(*) is the whole trick: RMSNorm is linear up to the per-row scalar `1 / rms`, and the client can compute `rms(h1)`
itself once it has `h1` in the clear.

### 2.2 The request's composite form vs the chained form
The request writes (*) as `W_gu diag(g2) h + (W_gu diag(g2) W_o) a`, i.e. a **composite** matrix `C_A = W_gu
diag(g2) W_o` (and `C_B = W_qkv diag(g1) W_down` for the down -> qkv fusion). The provider would compute, from masked
`a` and masked `h`: `W_o a` (needed anyway: the client needs `h1` for `rms(h1)` and for the next residual), `C_A a`
and `W_gu' h`: **three** products. But the provider can simply compute `H = R + W_o x` and then `Z = W_gu' H` in the
same trip (a chain of two linear maps on masked data is still linear): **two** products, each weight read once. The
two forms are the same function of the data; the composite one only adds:

| per layer, int8 bytes | 0.5B | 3B | 7B | 32B | 72B |
|---|---|---|---|---|---|
| model weights per layer (qkv + o + gate_up + down) | 15 MB | 77 MB | 233 MB | 488 MB | 878 MB |
| `C_A = W_gu diag(g) W_o` (2I x d) | 9 MB | 45 MB | 136 MB | 283 MB | 484 MB |
| `C_B = W_qkv diag(g) W_down` (qkv x I) | 6 MB | 28 MB | 87 MB | 198 MB | 303 MB |
| composites, whole model (extra provider memory) | 0.34 GB | 2.6 GB | 6.3 GB | 30.8 GB | 63.0 GB |
| ... relative to the model | +96 % | +95 % | +96 % | +99 % | +90 % |
| chained (fused2): extra matrices | 0 | 0 | 0 | 0 | 0 |

Cost of composites beyond memory: each trip reads `W_o`, `C_A` and `W_gu'` (about 2x the bytes of the chained form,
so about 2x the H100 time of a memory-bound trip); the masks need `C_A r` and `C_B r` as well (2x the factory MACs per
slot); the pks-v3 hint covers every matrix a masked input meets (2x hint storage and hint-build work); and `int8(C_A)`
is one more rounding of a product matrix. There is no upside: the ring-size problem (2.3) is identical. **Composites
are rejected; the chained form is what the repo's fused2 implements and what this work extends.**

### 2.3 Fixed point: why K32 does not hold and K56C53 does
On masked data the provider can only do ring arithmetic, so the residual add must be exact integer arithmetic. fused2
puts the residual on the output grid of the trip's first linear: `R[j] = rint(h[j] / ((sw1[j] / hmul[j]) s))` (`s` the
token's activation scale, `sw1` the int8 row scales, `hmul` public integer multipliers), so `H = R + hmul * (W1_8 aq)`
is exact and `h1 = H * (sw1 / hmul) * s`. The folded second linear is `W2' = int8(W2 diag(g * sw1 / hmul))`; the
client computes `z = (Z * sw2') * (s / rms(h1)) + bias` in float64. The residual stream (with Qwen's massive
activation channels) on the grid of a linear's output is large: measured at 0.5B with 8-bit migration (the quality
fix of 2026-11-23, which moves the norm gain's spread into `hmul` so the folded int8 rows are as fine as today's):
`|H|` up to 41.5 bits, `|Z|` up to 48.6 bits (the folded head included: same maxima in this run), a-priori bound 52-53
bits. K32 (cap 27-29) would force most rows to fewer activation bits (measured 2026-11-22: unusable); **K56C53** keeps
every row at 12 bits (measured here: 201,312 of 201,312 trip rows at 12 bits over 4088 tokens). Freivalds: the outputs
are up to `2^53`, so the p31 field (`2^31 - 1`) cannot hold `2B`; fused2 checks over `F_(2^61-1)` (needs
`2^(C+1) < p`).

### 2.4 Masks: no product chain
Masking the chain naively needs `W2' W1 r` (a depth-2 product for an encrypted factory). fused2's reparametrization:
draw `r_x`, `v` uniform and independent; send `x_hat = aq + r_x` and `R_hat = R + v - hmul * W1 r_x`. The provider
computes `H_hat = R_hat + hmul * W1 x_hat = H + v` and `Z_hat = W2' H_hat = Z + W2' v`; the client removes `v` and `W2'
v`. The factory computes `W1 r_x` and `W2' v`: two independent single-stage products, the same element counts as a
standard slot of the four linears (per layer `3d + I` mask inputs, `qkv + 2d + 2I` corrections). `(R_hat, x_hat)` is
uniform for any data because `v` is uniform and independent of `r_x`.

### 2.5 Biases, rotary, GQA, Qwen's qkv bias
* **qkv bias** (Qwen): added by the client after the scale, `z = Z * sw * s / rms + b` (exact float64, then float32):
  not a blocker. o / down have no bias in Qwen2 / Llama (a bias there would join `R` on the client side).
* **RoPE** acts on q and k after the trip, on the client, exactly as today: not a blocker.
* **GQA layout**: only the row order of the qkv output (the fold is per input column): not a blocker.
* **Tied embeddings** (0.5B, 3B): the folded head uses the embedding matrix as the head's float source.
* **LayerNorm** (for parallel-block models such as Pythia / Falcon): `LN(x) = g * C x / std(x) + b` with the centering
  matrix `C` (linear): fold `W diag(g) C` (a rank-1 correction of `W diag(g)`), `W b` joins the bias, `std` comes from
  the verified `H`. Not a blocker; not implemented (no LayerNorm architecture in the engine yet).

## 3. The head and the first qkv

* **Head fused with the last down (implemented, `head_fold`).** The final RMSNorm folds into the head exactly like a
  layer norm: the last trip becomes the pair `(down(L-1), int8(W_head diag(g_final * grid)))` for the rows that need
  logits (the last row; all rows with `all_logits`; the last `last_logits` rows of a verify pass); the other rows of
  a prefill take a single `down` trip in the same round trip (one lane multi call). The last down now takes migration
  multipliers too (gain = the final norm's). Logits `= (Z * sw_head) * (s / rms(h_L))`. This removes the separate
  masked head trip: 2L + 2 -> 2L + 1. The provider learns how many rows get logits, exactly what today's masked head
  trip reveals (its row count).
* **The first qkv (implemented, `local_qkv0`).** Its input is the embedding of the token just sampled, and sampling is
  nonlinear (and secret), so it cannot be fused with the previous pass. It is a public matrix (the folded `W_qkv0
  diag(g)`, int8: 1.0 / 5.2 / 16.5 / 36.7 / 83.9 MB at 0.5B / 3B / 7B / 32B / 72B), so the client computes the same
  exact integers itself: one trip fewer (2L + 1 -> 2L), the provider sees strictly less, no mask. Client cost: one
  memory-bound int8 GEMV, about 0.1 / 1.7 / 3.7 / 8.4 ms on one core at 0.5B / 7B / 32B / 72B (about one trip's worth
  at RTT 2-8 ms): a win only on high-RTT links, hence its own switch. Alternative with zero compute: the output
  depends only on the token id, so a per-token float32 table (vocab x qkv: 0.7 / 2.8 / 4.4 / 6.2 GB) gives the same
  bits.

## 4. Parallel attention + MLP blocks: one trip per layer (implemented, `pgw.engine.fused_parallel`)

A parallel block reads one residual for both branches: `h' = h + W_o Attn(n1(h)) + W_down m(n2(h))`. Fused:

* stage 1: `H = R + hmul * (W1_8 x)` with **one** int8 matrix `W1 = [W_o | W_down]` (d x (d + I), per-row scales over
  both parts) and `x = quant([a | m])` (one per-row activation scale); integer multipliers cannot align two
  independently scaled inputs on one residual grid, so concatenation is the exact form;
* stage 2: `Z = W2'_8 H` with `W2' = [W_qkv diag(g1 grid) ; W_gu diag(g2 grid)]` (both next norms folded, rows stacked);
* the client: `rms(h')` from the verified `H`, q/k/v and gate/up from `Z`, RoPE + softmax attention and SiLU (both
  nonlinearities in the same client step), the next trip.

Trips per pass: layer 0's `[qkv | gate_up]` (a single trip; local would need the 10-568 MB stacked matrix), then one pair
per layer, the head folded into the last: **L + 1** (25 / 29 / 65 / 81 at 0.5B / 7B / 32B / 72B shapes). Same masks,
reparametrization, checks and privacy argument as fused2 (the same backend runs it). Mask elements per slot are
slightly *below* a standard slot (per layer `2d + I` inputs and `d + qkv + 2I` outputs vs `3d + I` and `2d + qkv + 2I`),
the provider reads the same weights per pass as fused2 (`W1` and `W2'` are the four linears regrouped: 81 + 152 MB per
7B layer), so device time per pass is unchanged and the trips halve. Composite matrices are not needed here either.

Quality cannot be measured on Qwen (a sequential model read as parallel is a different, untrained model); the tests
show closeness of the fixed-point schedule to a float64 parallel-block forward of the same weights. A real
parallel-block checkpoint with a commercial licence exists (Pythia 70M-12B, GPT-NeoX-20B: Apache-2.0; Falcon-7B/40B:
Apache-2.0), but they use LayerNorm, GELU MLPs (non-gated), partial rotary and biases: their architecture must be added
to the engine first (section 2.5 says how each folds). The conversion of Qwen to parallel blocks by fine-tuning is
another agent's study.

## 5. Can fusion go below 2 trips per sequential layer? No.

Claim: with the client evaluating softmax-attention and SiLU-gating, and the provider seeing only one-time-padded ring
values (no trusted hardware, no MPC/FHE nonlinear gates), a sequential layer needs at least two client steps, so a pass
needs at least 2L round trips (2L + 1 if the first qkv or the head stays a separate trip).

Argument. The provider's computation on a masked input `x + r` is useful only if the client can remove the mask from
the result with data it can compute without the plaintext output, i.e. `F(x + r) - c(r) = F(x)` for a mask-only
correction `c`: this holds for all `x` exactly when `F` is affine over the ring (`F(x + r) - F(x)` independent of `x`).
Softmax(q K^T) V (q from the trip, a ratio of exponentials) and silu(gate) * up are not affine in their inputs, so the
provider cannot carry a value through either; each needs the plaintext, i.e. a client step, i.e. the end of a round
trip. In a sequential layer the two nonlinearities are in series (attention feeds o -> residual -> norm -> gate_up;
SiLU feeds down -> residual -> norm -> next qkv), so the round trips per layer are at least 2; fused2 meets this bound
with one affine chain per segment (o + gate_up, down + next qkv). The "combine the attention output trip with
something" question has the answer fused2 already gives: o is fused forward with gate_up; nothing before o can join it
because attention is in between. Ways around the bound, all outside this protocol: (1) put the two nonlinearities in
parallel (parallel blocks, section 4: L + 1); (2) hold some linears on the client (`attention_down` placement of
`research/plans/fewer-round-trips.md`: 1 trip per layer, the client reads 42 % of the weights per token); (3) MPC/FHE
gates on the provider, quantified in `fewer-round-trips.md` section 2 as 10^3-10^5x the per-token budget or no round
saving (FSS needs both parties per nonlinear layer); (4) speculation (T tokens per pass), which divides, not removes.

## 6. Numerics and quality vs the current pipeline

`benchmarks/fused2_quality.py --configs stdmh:K32A12C27 std:K32A12C27 f2mighead8:K56A12C53 f2mig8:K56A12C53 --tokens
4096` (Qwen2.5-0.5B-Instruct, wikitext-2 test, 8 windows of 512 = 4088 predictions, plus 6 fixed prompts = 98
positions; the engine itself with exact integer backends: every number is the pinned reference, no fake quantization;
`benchmarks/results/fused2/quality_qwen05b_headfold.json`):

| reference | ring | trips (0.5B) | ppl | top-1 = float | top-1 = current pipeline | prompts top-1 = float | max dlogprob next token vs float (prompts) |
|---|---|---|---|---|---|---|---|
| float (HF fp32) | - | - | 19.931 | - | - | - | - |
| **current pipeline** (standard + masked int8 head, the SDK default) | K32A12C27 | 97 | 19.847 | 96.67 % | 100 % | 97.96 % | 2.52 |
| standard + float head (gateway service default) | K32A12C27 | 96 + float head | 19.834 | 97.09 % | 98.56 % | 96.94 % | 2.45 |
| **fused2 + migration 8 + folded head (+ local qkv0)** | K56A12C53 | **48** | **19.887** | **96.26 %** | **97.19 %** | 95.92 % | 2.18 |
| fused2 + migration 8, float head on the gateway (2026-11-23 run, same setup) | K56A12C53 | 49 + float head | 19.855 | 96.77 % | not measured | 96.94 % | 2.21 |

(This run's fourth config, the float-head fused2, was killed by the box's OOM killer under other jobs' load; its
row is the identical configuration from `quality_qwen05b_4k.json`, 2026-11-23.)

Reading. The folded-head fused reference is within 0.4 points of the current pipeline's top-1 agreement with float
(17 tokens of 4088; the paired difference is near the sampling noise of this sample) and +0.04 in perplexity; the
current pipeline itself moved 1.4 % of top-1 tokens when the masked int8 head replaced the float head. The int8 head
costs about the same in both pipelines (float head -> int8 head: standard 97.09 -> 96.67 %, fused 96.77 -> 96.26 %),
so the gap to the current pipeline is fused2's own (-0.3 points at the float head, as on 2026-11-23). Every trip row
kept 12 activation bits; the largest `|H|` and `|Z|` were 41.5 and 48.6 bits (cap 53). `local_qkv0` changes no bit
(same integers). Not measured: 7B / 32B quality of the folded head (the 1.5B simulator of 2026-11-23 had fused2 +
migration at +0.3 points vs the int8 reference; the bit budget was the same at 0.5B and 1.5B, so K56C53 should hold;
the 7B check is in the GPU plan, section 11).

## 7. Privacy argument (unchanged, and why)

* **Per trip, the provider receives** `x_hat = xq + r_x` and (pair trips) `R_hat = R + v - hmul * W1 r_x`, with `r_x`
  and `v` fresh uniform ring vectors drawn from the mask stream at a new slot: `(x_hat, R_hat)` is uniform and
  independent of the data. It computes `H_hat`, `Z_hat` itself, so seeing them adds nothing. Shapes are public (row
  counts, which rows get logits: as today).
* **Single use.** Each trip key's mask cursor only moves forward; no row is ever sent twice. Tested: across prefill,
  decode, lockstep batches and verify passes no payload row repeats (a repeated row would mean a reused mask).
* **Nothing new revealed.** Head fold: the head rows' count, as the masked head trip reveals today. `local_qkv0`: one
  trip and its payload fewer, strictly less. Parallel blocks: the same trip kinds. The client's secrets (`r_x`, `v`,
  check vectors `u`, `uW`) never leave the client / factory.
* **Timing**: trip counts and sizes are functions of public row counts and the public schedule (fixed at start-up),
  as today.

## 8. Integrity: every returned value is checked

For every pair trip both stages are checked on the unmasked integers before any output is used: stage 1 `|H| <= 2^C`
and `u1 . (H - R) == ((u1 * hmul) W1) . aq (mod p)`; stage 2 `|Z| <= 2^C` and `u2 . Z == (u2 W2') . H (mod p)` with the
verified `H`; `p = 2^61 - 1`, one secret vector per linear: at least 60.8 bits per check (`fused2.soundness_bits`).
New items: the folded head's `Z` (152k logits) is a stage-2 output like any other; the rows without logits take a
single trip of the last down with multipliers, checked with the same `(u * hmul) W1` vectors (the provider returns
`hmul * W1 x_hat`, the correction is `hmul * W1 r_x`). Because the checks run on unmasked values, a wrong factory
correction (`W_head' v`) is caught too. Tests (`tests/engine/test_fused2_headfold.py`, `tests/engine/test_fused_parallel.py`):
masked == plain bit for bit; exact trip counts (2L + 1, 2L, L + 1); `checks == 2 x trips` in a decode pass where every
trip is a pair; tampering with the head's logits, the last `H`, `H` with a consistently recomputed `Z`, `Z` computed
without the residual, the single rows of the last down, dropping a layer's whole residual delta (parallel), and a
wrong factory head correction all raise `IntegrityError` before use.

## 9. The trips x RTT model, and what 20 tok/s takes

`python benchmarks/round_trip_model.py --consumer [--tau ... --rows ...] [--gw target] [--rows-free] [--tp 2]` and
`--twenty`. Per pass: `trips x (RTT + 0.05 ms transport + client work per trip(rows)) + device(rows) + serve +
draft`, tok/s = tau / pass. Inputs: H100 device time measured (7B K32 singles 0.186 ms per layer = 1.25 TB/s, scaled by
weight bytes; fused pair op x1.22 at 1 row; rows curve 1 / 1.62 / 2.83 / 5.37 at 1 / 2 / 4 / 8 rows measured on the 7B
pair items); client work per trip measured at 32B widths on one core (standard 0.105 ms, fused pair 0.58 ms; "target"
0.06 ms per pair with several cores: not measured); serve 2 ms. Bytes on the link are an optional term (the WAN study
owns it); the trips x RTT term is what fusion changes. Other studies plug in tau (tokens per pass) and rows.

**tok/s, one stream, 1 H100 (measured client work, today's kernels):**

| model | schedule | trips | T = 1: RTT 5 / 20 / 50 ms | T = 3 (R = 4): RTT 5 / 20 / 50 ms |
|---|---|---|---|---|
| 0.5B | today (4L + 1) | 97 | 2.0 / 0.51 / 0.21 | 6.0 / 1.53 / 0.62 |
| 0.5B | fused2 folded (2L) | 48 | 4.0 / 1.03 / 0.41 | 11.2 / 3.04 / 1.24 |
| 0.5B | parallel (L + 1) | 25 | 7.5 / 1.97 / 0.79 | 19.9 / 5.71 / 2.35 |
| 3B | today | 145 | 1.35 / 0.34 / 0.14 | 3.9 / 1.02 / 0.41 |
| 3B | fused2 folded | 72 | 2.6 / 0.68 / 0.28 | 6.9 / 1.98 / 0.82 |
| 7B | today | 113 | 1.70 / 0.44 / 0.18 | 4.8 / 1.30 / 0.53 |
| 7B | fused2 folded | 56 | 3.2 / 0.87 / 0.35 | 7.7 / 2.44 / 1.03 |
| 7B | parallel | 29 | 5.6 / 1.63 / 0.67 | 12.0 / 4.38 / 1.93 |
| 32B | today | 257 | 0.74 / 0.19 / 0.08 | 2.0 / 0.56 / 0.23 |
| 32B | fused2 folded | 128 | 1.32 / 0.37 / 0.15 | 3.0 / 1.03 / 0.44 |
| 32B | parallel | 65 | 2.3 / 0.71 / 0.30 | 4.4 / 1.81 / 0.83 |

Reading: at RTT >= 5 ms fusion is a 1.8-2x speed-up at every size; at 0.5 ms (same data centre) it is a loss at 7B+
with today's measured client work per pair trip (0.58 ms at 32B on one core vs 0.105 ms standard) and only a gain once
the pair trip's client work is split over cores (`--gw target --rows-free --tp 2`: 32B 5.7 -> 9.9 tok/s at RTT 0.5 ms,
T = 1; 13.6 -> 22.0 at T = 3).

**What 20 tok/s needs** (`--twenty`: the largest RTT per trip that still meets 50 ms per token; rows = T, target client
work, a memory-bound multi-row kernel, 32B / 72B on 2 H100s: optimistic on purpose, so these are upper bounds):

| model (layers) | schedule | trips | T=1 | T=2 | T=4 | T=8 |
|---|---|---|---|---|---|---|
| 0.5B (24) | today / fused2 folded / parallel | 97 / 48 / 25 | 0.43 / 0.92 / 1.82 | 0.92 / 1.93 / 3.75 | 1.91 / 3.94 / 7.61 | 3.88 / 7.96 / 15.3 |
| 3B (36) | same | 145 / 72 / 37 | 0.24 / 0.54 / 1.11 | 0.55 / 1.20 / 2.39 | 1.18 / 2.50 / 4.94 | 2.44 / 5.11 / 10.0 |
| 7B (28) | same | 113 / 56 / 29 | 0.28 / 0.61 / 1.28 | 0.67 / 1.44 / 2.89 | 1.46 / 3.09 / 6.10 | 3.04 / 6.37 / 12.5 |
| 32B (64) | same | 257 / 128 / 65 | 0.01 / 0.10 / 0.32 | 0.15 / 0.43 / 0.96 | 0.42 / 1.07 / 2.25 | 0.97 / 2.33 / 4.83 |
| 72B (80) | same | 321 / 160 / 81 | - / - / - | 0.01 / 0.13 / 0.43 | 0.18 / 0.58 / 1.37 | 0.53 / 1.45 / 3.25 |

The pure budget per trip, before any device or client work, is `1000 T / (20 x trips)` ms: at 64 layers 0.39 / 1.56 /
3.1 ms (2L, T = 1 / 4 / 8) and 0.77 / 3.1 / 6.2 ms (L + 1). **GPU time per trip at 32B** (memory-bound reads): a
fused o + gate_up trip reads 309 MB (0.30 ms at the measured 1.25 TB/s x 1.22), down + qkv 178 MB (0.17 ms), the
folded head trip adds 0.78 GB (0.6 ms once per pass); per pass about 31 ms on one H100 (15.5 on two; 9.3 ms is the
HBM roofline at 3.35 TB/s), i.e. at T = 1 the device alone uses 60 % of the 50 ms budget, and today's kernels at 4
rows cost 2.8x one row (88 ms per pass): **a memory-bound multi-row kernel and TP 2 are prerequisites for 20 tok/s
at 32B even co-located.** Over the consumer internet (RTT 5-50 ms) no 32B configuration reaches 20 tok/s.

**Client CPU and provider cost.** Client work per pass at 32B: 75 ms on one core with today's fused driver (T = 1;
the native fused2 driver's pair trip is 5.5x the standard trip's work: K56 words, two checks, the residual), about 8
ms with the target split. Mask factory: rows / tau slots per token; at RTT 20 ms the factory needs 0.2-0.9 cores at
7B-72B (K56 digits, x6 a K32 slot; T = 1-3), so the factory is not the limit at internet RTT, but the provider's
maskgen is (7 K32 maskgen products per K56 mask: about 7x the GPU seconds per slot; the H100 estimate of 450 K32
slots/s at 7B becomes ~64 K56 slots/s, about 30-60 concurrent consumer streams per GPU for maskgen alone at 1-2
slots/s each).

## 10. What production needs (not done here)

Provider side: the lane already carries fused2 pair items (`pgw.service.fused2_lane`); it needs the head-fold item
(`down(L-1) + head`, 152k-row second stage: the folded head is 0.5-1.2 GB of int8 at 7B-72B, replacing the K32 masked
head's matrix, not added to it) and the parallel items; the pair op (`pgw.core.ring64`) handles any shapes; the
provider holds no composite. Its memory grows by nothing (the fused tables are the same int8 matrices refolded; the
K32 standard tables are only needed while both schedules are served). Client side (the SDK): K56 masks from the
pks-v3 digit factory (`pgw.factory.fused2_he`, exists for the gateway service; the SDK's factory, store reservations
and stock sizing at 7 bytes per element), the native fused2 driver's head fold and `local_qkv0` (it refuses both
today, fail closed), lane multi calls for the last trip's two items, speculation padding on fused schedules, the
p61 check in the SDK's metrics. Model side: parallel blocks need a parallel-block checkpoint and its architecture in
the engine.

## 11. GPU validation plan (RunPod, proposed; `deploy/runpod/RUNBOOK.md` session 6)

`benchmarks/fused2_pair_bench.py --model {7b,32b,72b} --extra-items --rows 1,2,4,8,16` on one H100: the K56 pair op at
real shapes for the fused2 items, the folded-head item and the parallel item, byte for byte against the prototype
provider, vs the graphed K32 singles. It replaces the scaled 7B numbers in `C_SCHEDULES` / `C_ROWS_DEVICE` with
measured 32B / 72B ones. About 10 min of H100 inside the `perf` session (~$0.5). Then (needs the production pieces) a
7B end to end with the folded head across a real WAN RTT.

## 12. Prototype: what was built and measured

* `pgw.modelpack.fused2_weights`: `Fused2Weights(head=True)` (the folded head), `migration_multipliers(head=True)`
  (multipliers for the last down), `all_linears()`. Defaults unchanged (no head; the last down still refuses
  multipliers without the head).
* `pgw.engine.fused2`: `Fused2Schedule(head_fold=, local_qkv0=)`, `Fused2FoldedHead` (head mode reading the last trip's
  logits; one opt-in hook in `TransformerModel._steps`), single trips of a first linear with multipliers (provider
  and backend), the head's mask key. The native fused2 driver refuses both options (Python path only).
* `pgw.engine.fused_parallel`: `ParallelWeights`, `ParallelFusedSchedule`, `enable_fused_parallel`,
  `parallel_float_reference`.
* `pgw.engine.fused2_remote`, `benchmarks/fused2_e2e.py --head-fold --local-qkv0` (provider in its own process over a
  0600 Unix socket); `benchmarks/fused_trips_e2e.py` (in-process, both schedules); `benchmarks/fused2_quality.py`
  (`stdmh`, `f2mighead<b>`, top-1 vs the first config); `benchmarks/fused2_pair_bench.py --model --extra-items`;
  `benchmarks/round_trip_model.py --consumer / --twenty`.
* Tests: `tests/engine/test_fused2_headfold.py` (34), `tests/engine/test_fused_parallel.py` (16); the existing fused2,
  lane, native, store, masked-head and model-hook suites pass unchanged.

End to end on this box (CPU provider; ms are not a speed measurement). `benchmarks/fused_trips_e2e.py`, Qwen2.5-0.5B-
Instruct, a chat prompt + 6 greedy tokens, in-process provider and plain factory, then the plain reference replays
every call (`benchmarks/results/fused2/trips_e2e_qwen05b_2026-11-27.json`):

| schedule | trips per token (measured) | today | checks per token | bytes up / down per token (K56) | logits vs plain reference |
|---|---|---|---|---|---|
| fused2 + folded head + local qkv0 | **48** (= 2L) | 97 (4L + 1) | 96 (2 per trip) | 1.27 / 3.18 MB | bit-identical, 7 / 7 positions |
| parallel blocks (0.5B read as parallel) | **25** (= L + 1) | - | 49 | 1.12 / 3.04 MB | bit-identical, 7 / 7 positions |

(Today's K32 path at 0.5B moves 0.73 / 1.82 MB per token; the fused K56 bytes are 1.75x per element.)

With the provider as its own process (`benchmarks/fused2_e2e.py --head-fold --local-qkv0 --provider-device cpu`,
0600 Unix socket, the pair op on masked ring elements only, the provider-only weights file; 4 greedy tokens;
`benchmarks/results/fused2/e2e_headfold_qwen05b_cpu_2026-11-27.json`): **48 trips per token** (today 97), 4.46 MB
per token on the socket, 481 checks passed, the plain replay bit-identical at 5 / 5 positions (max logit diff 0.0),
answer "The sky is blue". Trip p50 117 ms, of which 115 ms is the CPU provider's K56 products (a GPU's job).
