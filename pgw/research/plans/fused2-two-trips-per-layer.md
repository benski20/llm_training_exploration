# WP-B4 "fused2": two provider round trips per layer (research + prototype)

Date: 2026-11-22. Scope: cut the sequential provider round trips of exact private decoding from 4 per layer
(qkv, o, gate_up, down) to the 2-per-layer floor of `fewer-round-trips.md` section 0, without weakening
privacy (the provider sees only masked ring elements of public shapes) and with an integrity check on every
provider output that is used. Local CPU only (Qwen2.5-0.5B). Default off. Code: `src/pgw/engine/fused2.py`,
`TransformerModel._fused2_attention` and `PassState.extra` (`model.py`), tests `tests/engine/test_fused2.py`,
quality script `benchmarks/fused2_quality.py`.

## 0. Summary

**Result.** An exact fixed-point reformulation with 2 trips per layer works and the prototype is exact
(masked == plain bit for bit, every provider output checked). Trips per token go from 4L (+1 with a provider
head) to **2L + 1**: 70B 321 -> 161, 7B 113 -> 57, 0.5B 97 -> 49. Elements on the wire and in the mask
store per token are unchanged; the residual needs a **bigger ring: K40 (1.25x bytes) or K48 (1.5x)** instead
of K32. The mask factory does **no extra products** (a reparametrization of the masks removes the chain).
On Qwen2.5-0.5B the new reference is about as good as today's int8 reference by perplexity (23.63 vs 23.51,
float 23.67, 1020 tokens) but **agrees less often with the float model on the top-1 token: 93.0% vs 96.4%**
(wikitext), 94.9% vs 96.9% on fixed prompts. It is a different model, not a free lunch; see section 6.

| | standard (today) | fused2 |
|---|---|---|
| trips per token, gateway LM head | 4L | **2L + 1** (layer-0 qkv, then 2 per layer) |
| trips per token, provider LM head | 4L + 1 | **2L + 1** (head folded into the last trip B; not prototyped) |
| 0.5B / 7B / 70B | 96-97 / 112-113 / 320-321 | **49 / 57 / 161** |
| ring elements on the wire per token | `L (3d + I)` up, `L (2d + 2I + q)` down | **identical** (same counts, regrouped) |
| bytes per element | K32: 4 | K40: 5 (**1.25x**) or K48: 6 (1.5x) |
| mask rows per slot (`r` seeds + `W r` rows) | per linear | **identical counts**; ring K grows as above |
| factory products per slot | `W r` for 4 linears | **the same 4 single-stage products** (no chain, see 2.3) |
| HE factory | q = 2^56 (K32) | K40: q = 2^64 (exists); K48 needs q > 2^64 (not supported) |
| check | ring 7 vectors (21 bits) or p31 (31 bits) | **F_(2^61-1), 60.8 bits per vector**, both stages |
| reference model | pinned int8 (K32A12C27) | **a new pinned fixed-point reference** (quality in section 4) |

## 1. Why the obvious fusion fails, and what fixes it

The standard layer between o and gate_up does, on the gateway: dequantize `o` (float), `h1 = h + o` (float),
`x = g * h1 / rms(h1)` (float), requantize `x` (rounding, per-token absmax). None of that can run on masked
data: the residual add is float, `1/rms` is a nonlinear data-dependent scalar, and requantization rounds.
fused2 removes every one of those from the provider's path:

1. **Residual in fixed point on the output grid of the trip's first linear.** At the start of a trip the
   gateway (which holds `h` in the clear) sets `R[j] = rint(h[j] / (sw1[j] s))`, where `s` is the token's
   activation scale and `sw1` the int8 row scales of the first linear `W1` (o or down). Then `H = R + W1_8 aq`
   is an exact integer and `h1 = H * sw1 * s` channel-wise: the residual add happens on the provider in the
   ring, with no rounding. All rounding (`aq`, `R`) is the gateway's, before masking.
2. **RMSNorm: linear except for one scalar per row.** `W2 (g * h1 / rms(h1)) = (s / rms(h1)) * (W2 diag(g * sw1)) H`.
   So the provider multiplies the *un-normalized* integer `H` by the public matrix
   `W2' = int8(W2 diag(g * sw1))` (the norm gain and `sw1` folded in before int8 rounding, at pack time),
   and the gateway multiplies the exact integer result by `s * sw2' / rms(h1)` afterwards. It computes
   `rms(h1)` from `H`, which the provider returns in the same trip (`H` is needed anyway: it is the new
   residual). No quantization of the normed vector exists any more: the second linear sees the exact
   fixed-point residual.
3. **Exactness and decoding.** The gateway picks, per row, the most activation bits `A_t <= A` for which the
   a-priori bounds `max|R| + wn1 ||aq|| <= 2^C` and `wn2 (||R|| + sigma1 ||aq||) <= 2^C` hold (`wn` = largest
   row norm of the int8 weight, `sigma1` = spectral norm of `W1_8` with a 1e-6 relative margin over the
   float64 SVD). Then every genuine `H`, `Z` is in `[-2^C, 2^C]` with `2^C < 2^(K-1)`: centered decoding is
   unambiguous and the range check uses `2^C`. Lowering `A_t` coarsens both `aq` and the residual grid.

Trips: trip A `(R, aq(attn_out)) -> (H1, Z = W_gu' H1)`; trip B `(R', aq(silu(gate)*up)) -> (H2, Z' =
W_qkv'(next) H2)`. Layer 0's qkv is one extra single trip on the per-row quantized embedding (`W_qkv0 diag(g)`).
The last layer's trip B is the down product alone (the gateway adds `R`, the same integers).

## 2. Masks, privacy, factory

### 2.1 Masking a two-stage chain
Provider computes `H_hat = R_hat + W1 x_hat`, `Z_hat = W2' H_hat` (mod 2^K). With `x_hat = aq + r_x`,
`R_hat = R + r_R`: `H_hat = H + (r_R + W1 r_x)`, `Z_hat = Z + W2'(r_R + W1 r_x)`. The naive correction is a
product chain (`W2' W1 r_x`), i.e. depth 2 for an encrypted factory.

### 2.2 The reparametrization that removes the chain
Draw `r_x` and `v` uniform and independent; set `r_R = v - W1 r_x`. Then `r_R` is uniform and independent of
`r_x` (`v` is), so `(R_hat, x_hat)` is uniform for any data: **the provider's view is exactly today's** (uniform
ring vectors of public shapes; it computes `H_hat` itself, so seeing it adds nothing). The corrections are
`H = H_hat - v` and `Z = Z_hat - W2' v`. The factory computes `W1 r_x` and `W2' v`: two independent
single-stage products, the same work and storage as today's o and gate_up rows. The gateway derives
`r_R = v - W1 r_x` itself (one vector subtraction). **Mask-factory cost is unchanged** in products,
elements and HE depth; only the ring size (section 3) changes it.

### 2.3 Factory cost (chain), stated plainly
- Plain/GPU factory: same GEMMs; digit GEMMs scale with K/8 (K40: 1.25x, K48: 1.5x of K32).
- Encrypted (MLWE) factory: unchanged structure (no ciphertext-times-ciphertext, no depth 2). The modulus
  must hold K-bit messages: K40 uses the existing q = 2^64 path; **K48 would need q > 2^64**, which the
  64-bit kernels do not support (a real blocker for K48 with the HE factory).
- Storage per slot: same element count; K40 = 1.25x the K32 bytes, K48 = 1.5x.

### 2.4 Integrity
Every provider output that is used is checked on the unmasked integers, with Freivalds over
`F_p, p = 2^61 - 1` (one secret vector per linear by default, `u W8 mod p` precomputed by the factory):
- stage 1: range `|H| <= 2^C` and `u1 . (H - R) == (u1 W1) . aq (mod p)`;
- stage 2: range `|Z| <= 2^C` and `u2 . Z == (u2 W2') . H (mod p)`, with `H` already verified.
Soundness per stage per vector: `e = y - y_true`, `|e| <= 2^(C+1) < p`, so `e != 0 mod p` and a uniform `u`
misses it with probability at most `ceil(2^64/p)/2^64 = 1.125/p`: **60.8 bits**. Both stages must pass, so a
wrong `H` (even with `Z` recomputed consistently from it) fails stage 1, and a correct `H` with a wrong `Z`
fails stage 2. As with p31, the check also covers a wrong factory correction (`W1 r_x`, `W2' v`), because it
runs on the unmasked values. The p31 field cannot be used: `2^(C+1)` is far above `2^31`. The gateway-side
check arithmetic in the prototype is numpy limb arithmetic (16-bit limbs, int64 sums), not a native kernel.

## 3. Ring sizes

The residual `R = h / (sw1 s)` is the big term: it is the residual stream on the grid of the o / down output,
so its size is `max|h| / (sw1 s)` and it carries Qwen's massive-activation channels. Measured on 0.5B
(largest `log2 |value|` over every trip of 1020 wikitext tokens and the prompts, A = 12 everywhere at K48):

| quantity | bits |
|---|---|
| layer-0 qkv output (quantized embedding x int8) | 22.4 |
| `H` (residual + o / down output) | 34.2 |
| `Z` (gate_up / next qkv of the un-normalized residual) | 41.3 |

So the second product needs about 41-42 bits plus the a-priori bound's slack (Cauchy-Schwarz with the
spectral norm overestimates by about 2-4 bits):

| ring | cap C | rows below 12 activation bits | effect | bytes/elt | HE factory |
|---|---|---|---|---|---|
| K32 | 29 | most rows (bound ~45 bits) | not usable at quality | 4 | yes |
| **K40** | 37 | **6.9%** (mostly 11 bits; tail down to 4) | ppl 23.64, same as K48 within noise | 5 | **yes (q = 2^64)** |
| K48 | 45 | 0% | the reference at full precision | 6 | no (needs q > 2^64) |

Freivalds: `2^(C+1) < p` with p = 2^61 - 1 holds for all of them (C <= 59). The p31 field (`2^28` cap) and the
ring check (`h = K - log2(4 * 2^C)` = 1 bit per vector at K40C37) cannot be used; hence the p61 check.
Wider models (7B: `n` = 3584/18944, larger residuals) will need a few more bits: K40 there must be measured
(more rows below 12 bits), K48 holds with margin if 0.5B is representative (7B not measured).

## 4. Quality (Qwen2.5-0.5B-Instruct, real weights)

`benchmarks/fused2_quality.py` runs the engine itself with exact integer backends (the pinned references, not a
fake quantization); results in `benchmarks/results/fused2/quality_qwen05b.json`. wikitext-2 test, first 1024
tokens in 4 windows of 256 (1020 predictions); 6 fixed prompts (98 positions). The box was shared and heavily
loaded (load 30-50 on 4 vCPUs) and memory-limited, so the sample is small: differences of a few hundredths in
perplexity are noise; the top-1 gaps are not (about 35 tokens of 1020).

| model | ppl | top-1 = float (wikitext) | top-1 = float (prompts) | max abs dlogprob of next token vs float (prompts) | L-inf dlogprob over vocab vs float |
|---|---|---|---|---|---|
| float (HF fp32) | 23.665 | - | - | - | - |
| int8 reference, K32A12C27 (today) | 23.510 | **96.4%** | 96.9% | 2.45 | 6.55 |
| fused2 K48A12C45 | 23.629 | 92.9% | 94.9% | 2.30 | 6.75 |
| fused2 K40A12C37 | 23.635 | 93.0% | 94.9% | 2.26 | 6.58 |

fused2 vs the int8 reference on the prompts: top-1 equal on 96.9% of positions, max abs dlogprob of the next
token 1.40 (K48) / 1.45 (K40).

Reading. Perplexity is unchanged within noise, and the worst-case next-token logprob error is no worse than
today's. But fused2 flips the float model's top-1 token about twice as often (7% vs 3.6%). The likely cause
is the folded weight quantization: `W diag(g * sw1)` puts Qwen's large RMSNorm gain outliers into the weights
before per-row int8 rounding, so most columns get a coarser grid (the reason SmoothQuant uses a migration
strength < 1). Today's reference quantizes `g * h / rms` per token instead, which absorbs those outliers in
the activation scale. Not tried for lack of time: folding only part of the gain (or a per-channel public
pre-scale `c` with `R` on a `c`-scaled grid), int8 with per-row-group scales for the folded linears, and
keeping layer 0 qkv on the standard path; the fold from the pack's int8 weights (`f2i8`, no float checkpoint)
was implemented but not measured (it rounds twice, so it cannot be better).

## 5. Prototype status

- `pgw.engine.fused2`: `Fused2Weights` (pack-time folding from the float checkpoint, or from the pack's
  int8 weights), `Fused2Schedule` (a `Schedule` hook: 2 trips per layer + layer-0 qkv; `enable_fused2`),
  `Fused2PlainBackend` (the reference: exact integer products), `Fused2MaskedBackend` (mask, call, unmask,
  range + F_p Freivalds on both stages), `Fused2MaskSource` (in-process plain factory: seed-expanded `r_x`,
  `v`, single-use cursors, `W1 r_x`, `W2' v`, check vectors), `Fused2Provider` (the service op on masked
  ring elements only).
- Tests (`tests/engine/test_fused2.py`, tiny Qwen2 and Llama, K48C45 and K40C37): masked == plain bit for
  bit over lockstep prefill + decode of two sequences; exactly `2L + 1` trips per pass and 2 checks per pair
  trip; the provider only ever receives canonical ring elements with high bits set; a tight cap lowers
  activation bits and stays exact (a cap the residual cannot meet is refused, never wrapped); tampering
  (single trips, `H`, `H` with a consistently recomputed `Z`, `Z`, `Z` computed without the residual, deltas
  of 1, 2^30, -2^20) and a wrong factory `W2' v` row all raise `IntegrityError`; masks are single use and the
  corrections are the two independent single-stage products; exact F_p arithmetic; defaults off (the gateway
  still refuses `schedule = "fused2"` via `PENDING_SPEED_SETTINGS`).
- Not done (follow-ups): wire message for a pair op and the service kernel (two ring GEMMs plus an add, on the
  GPU), segment format for pair masks (`W1 r_x`, `W2' v`, `p61` check vectors) and the store, a native p61
  check kernel, the async split-hop path, speculation padding (refused for any schedule hook), the provider
  LM head folded into the last trip, a 7B quality check (the gating condition in `GatewayModelConfig`).

## 6. Verdict: is this "no-compromise"?

**No, not as measured.** Privacy and integrity are uncompromised (the provider's view is uniform ring
elements of public shapes, exactly as today; every output is checked, both stages, at 60.8 bits; the factory
does the same products). Exactness is kept, but against a **new reference model**, and that model is
measurably different from the float model: perplexity equal within noise on 0.5B, top-1 agreement with float
93% instead of 96%. It also costs 1.25x (K40) or 1.5x (K48) bytes per element on the wire, in the mask store
and in factory digit-GEMMs, plus a 61-bit check field on the gateway; K48 is incompatible with today's HE
factory modulus.

What it buys is large where latency is trips: per-token round trips halve (70B: 321 -> 161), per-trip
payload grows about 2.5-3x (two linears' worth at 1.25-1.5x width; 70B average 137 KB -> about 340-410 KB),
so on any link where `RTT >> bytes / bandwidth` the per-token latency nearly halves.

Recommendation: keep it opt-in and off; do not promote to a preset until (1) a weight-folding variant closes
the top-1 gap (section 4 lists candidates; target: top-1 vs float within 1 point of the int8 reference),
(2) the 7B quality check that `GatewayModelConfig.schedule` is gated on passes, and (3) K40 holds at 7B with
few rows below 12 bits (else K48 and an HE modulus above 64 bits). Then the wire op, segment format, native p61
kernel and the async path are routine engineering on the prototype here.
