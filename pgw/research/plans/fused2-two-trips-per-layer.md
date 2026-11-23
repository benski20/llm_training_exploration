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

**Update 2026-11-23 (sections 7-8).** The gap was the int8 rounding of the folded weights; migrating the norm
gain into the residual's fixed-point grid (public integer multipliers, provider computes `R + hmul * W1 x`)
closes it: 0.5B top-1 96.8% vs 97.1% for the int8 reference on 4088 tokens (equal on the prompts; ppl 19.86 vs
19.83), 1.5B 98.0% vs 97.7%, at ring **K56C53** (7 bytes per element) with the same trips, elements, masks and
factory products. An exact socket harness for one 7B e2e run is ready (section 8).

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

**Update 2026-11-23:** condition (1) below is met by the migration of section 7 (0.5B within 0.3 points of the int8
reference on 4088 tokens, 1.5B above it), at the price of K56 instead of K40/K48 (1.75x the K32 bytes; no HE
factory modulus for it). Conditions (2) (7B quality) and the production path (section 8) remain.

Recommendation: keep it opt-in and off; do not promote to a preset until (1) a weight-folding variant closes
the top-1 gap (section 4 lists candidates; target: top-1 vs float within 1 point of the int8 reference),
(2) the 7B quality check that `GatewayModelConfig.schedule` is gated on passes, and (3) K40 holds at 7B with
few rows below 12 bits (else K48 and an HE modulus above 64 bits). Then the wire op, segment format, native p61
kernel and the async path are routine engineering on the prototype here.

## 7. Update 2026-11-23: the top-1 gap diagnosed and closed (residual-grid migration)

### 7.1 Diagnosis
Measured with a float64 simulator of the exact fused2 arithmetic (the same quantizers, folds and residual
grids as the engine; it reproduces the engine's numbers: std 96.4%, fused2 92.6-93.0% on ~1k tokens) by
replacing one rounding at a time:

| variant (Qwen2.5-0.5B, wikitext-2 test) | top-1 = float, 1k tok | 4k tok |
|---|---|---|
| int8 reference (std, K32A12C27) | 96.4% | 96.9% |
| fused2 as of 2026-11-22 (int8 folded weights) | 92.6% | 93.9% |
| fused2 with the folded weights kept in float (all other roundings as fused2) | 96.2% | **97.2%** |
| ... and without residual rounding | 96.7% | - |
| float qkv folds only (gate_up folds int8) | 93.6% | - |
| float gate_up folds only (qkv folds int8) | 95.3% | - |
| int8 fold of one single linear (all others float; layers 0-4 measured) | 96.2-97.1% | - |
| top-16 columns of each fold kept off the int8 matrix (gateway, 16-bit) | 93.4-94.4% | - |

So **the whole gap is the int8 rounding of the folded weights** `W diag(g * sw1)`, spread over all layers
(gate_up folds more than qkv folds; no single layer or step dominates), not the residual or activation
rounding. Mechanism: the folded column scales `c_j = |g_j| * sw1_j` spread by 7x (median over folds, up to
32x) between the median and the largest column, so the per-row int8 grid (set by the largest `|W_ij| c_j`)
is 2.2x coarser (median; up to 8x) for a median column than today's per-row int8 weight. Splitting off a
few outlier columns does not help: the spread is broad, not a handful of outliers.

### 7.2 Fix: migrate the gain into the residual's fixed-point grid (pack time, public, exact)
Per trip's first linear (`o`, `down`) choose public integers `hmul_j >= 1` and let the provider return
`H = R + hmul * (W1_8 aq)` (one elementwise integer multiply mod `2^K`), with the gateway's residual on the
finer grid `R = rint(h / ((sw1 / hmul) s))`; the next fold becomes `int8(W2 diag(g * sw1 / hmul))`.
With `hmul_j = max(1, floor(c_j / kappa))`, `kappa = max_j c_j / 2^b`
(`pgw.engine.fused2.migration_multipliers`, b = 8), every column above `kappa` sees a scale within 2x of the
others: the folded int8 rows are as fine as today's. It is SmoothQuant's migration, done exactly in the ring
(the multiplier is on the integer side, the gateway's residual grid absorbs it), and it also makes the
residual grid finer. Masks: `r_R = v - hmul * (W1 r_x)` (still uniform, `v` is); stage-1 check
`u . (H - R) == ((u * hmul) W1) . aq (mod p)`, same soundness; same trips, elements, factory products.
Cost: about `b` more bits in `H` and `Z` (0.5B, b = 8: `|H|` 41.5, `|Z|` 48.6 bits, a-priori bound 52) ->
**ring K56C53** (7 bytes per element, 1.75x K32; K48 would drop activation bits on most rows).

Also implemented and measured (opt-in): the calibrated split fold (`calibrate_fused2`: b-bit migration plus the
top-k columns and the channels whose migrated `|H|` would exceed the calibration maximum are multiplied by the
gateway itself from the verified `H`, `gateway_bits`-bit weights): it keeps K48, but needs calibration text,
holds 4-18 columns per fold on the gateway and still exceeds the K48C45 bound by 1-2 bits on some rows.

### 7.3 Quality (engine = the exact pinned reference; sim = the float64 simulator)
Qwen2.5-0.5B-Instruct, wikitext-2 test, first 4097 tokens in 8 windows of 512 (4088 predictions), plus the 6 fixed
prompts (98 positions); `benchmarks/results/fused2/quality_qwen05b_4k.json`.

| model | ring | ppl | top-1 = float (wikitext) | top-1 = float (prompts) | max abs dlogprob next token (prompts) | rows below 12 act. bits |
|---|---|---|---|---|---|---|
| float (HF fp32) | - | 19.931 | - | - | - | - |
| int8 reference (today), engine | K32A12C27 | 19.834 | **97.1%** | 96.9% | 2.45 | - |
| fused2 2026-11-22 (int8 fold), engine | K48A12C45 | 20.083 | 94.1% | 94.9% | 2.30 | 0 |
| **fused2 + migration (8 bits), engine** | **K56A12C53** | **19.855** | **96.8%** | **96.9%** | 2.21 | **0** |
| fused2 + calibrated split fold (4 bits, 4 columns), engine, 1022 tokens only | K48A12C45 | - | 96.2% (int8 ref. 96.4% on ~1k) | 98.0% | 1.81 | 0.01% |
| simulator: int8 reference / old fused2 / migration 8 bits / float folds (ceiling) | - | 19.83 / 20.08 / 19.86 / 19.84 | 96.9 / 93.9 / 96.9 / 97.2% | - | - | - |

Qwen2.5-1.5B-Instruct (simulator only: the engine at 1.5B is too slow on this box), wikitext-2, 2049 tokens:

| model | ppl | top-1 = float | max bits H / Z / Z bound |
|---|---|---|---|
| float | 12.744 | - | - |
| int8 reference | 12.821 | 97.65% | - |
| fused2 2026-11-22 | 12.853 | 96.48% | 33.8 / 41.6 / 45.3 |
| **fused2 + migration (8 bits)** | **12.811** | **97.99%** | 40.5 / 49.1 / 52.1 (fits K56C53) |

Reading: with migration fused2 is on par with today's int8 reference (0.5B engine: -0.3 points top-1 on 4088
tokens, about 13 tokens, within the sampling noise of the paired difference, and equal on the prompts; 1.5B:
+0.3 points; ppl equal within noise and better than float on both), against 93.9% / 96.5% before. The bit
budget is the same at 0.5B and 1.5B (Z bound 52.0-52.1 bits), so K56C53 should hold at 7B; the e2e driver's
`--auto-bits` probes public text first and falls back to 6 or 4 migration bits if more than 1% of rows would
lose activation bits.

## 8. Update 2026-11-23: an exact e2e harness (opt-in) and what the production path still lacks

`pgw.engine.fused2_remote`: the provider's pair op as its own process (`Fused2Service`: CUDA ring products via
`PreparedWeight`, any K <= 56, public `w8` + `hmul` only, from `Fused2Weights.save(provider_only=True)`) on a
0600 Unix socket or loopback TCP, and the gateway's client `RemoteFused2Provider` (the `Fused2Provider`
interface, so `Fused2MaskedBackend` masks, unmasks, range-checks and Freivalds-checks both stages unchanged).
`Fused2MaskSource.prefill(rows)` batches the masks ahead of use (the same rows as the lazy path; single use).
`benchmarks/fused2_e2e.py` runs one greedy generation through it and replays it on the plain reference;
`deploy/modal/fused2_e2e.py` is the one-container Modal app for the 7B run (README: "fused2, 2 provider trips
per layer").

Mask demand: one slot per token row, as today (`r_x`, `W1 r_x`, `v`, `W2' v` per trip key: `5d + 3I + q`
elements per layer, the same as the four linears of a standard slot), the same factory MACs per slot
(0.5B: 357.8 M, measured), at K56 7 bytes per element instead of 4.

Local check (Qwen2.5-0.5B, both processes on one loaded 4-vCPU box, provider on CPU, K56A12C53, b = 8):
49 trips per token (2L + 1), all 9 forward calls bit-identical to the plain replay, every row at 12
activation bits, 3.40 MB per token on the socket (K56), gateway work about 2 ms per trip in Python
(+ the LM head), provider CPU products 113 ms per trip (a GPU's job).

Still missing for the production stack (none needed by the harness): a pair item in the wire protocol and the
binary trip lane, a service handler and batcher entry for it (two ring GEMMs, the multiply and the add, the same
kernels), pair-mask segments in the store (`W1 r_x`, `W2' v`, p61 check vectors) and an encrypted factory for
K56 (an HE modulus above 2^64: not supported) or a plain factory with weights on the gateway, a native p61
check and a native per-trip path (the TripEngine / native decode driver have no pair op), speculation padding,
the provider LM head folded into the last trip, and the 7B quality check.

## 9. Update 2026-11-23: fast provider pair op, pair items on the lane, pair-mask segments, native p61 (opt-in)

What changed relative to section 8 (all default off; masked == plain bit for bit; the provider still sees only
uniformly masked ring elements of public shapes; every output still checked, both stages, 60.8 bits):

- **Provider pair op** (`pgw.core.ring64`, kernels in `_ring64_kernel.py`): Triton kernels over `Z/2^K`, any
  `K <= 64` that is a multiple of 8: unpack of the packed `K/8`-byte body, split-K GEMV partials (`sum`: wrapping
  uint64 multiply-adds on CUDA cores, the memory-bound one/few-row kernel; `dot`: `K/8` balanced base-256 digit
  planes of the input, one int8 tensor-core `tl.dot` per plane and tile, int32-exact per tile, recombined in
  wrapping uint64), and an epilogue that sums the partials, multiplies by `hmul`, adds `R`, reduces mod `2^K`,
  keeps `H` on the device for stage 2 and writes the packed reply. `PairOp.capture` makes one CUDA graph of
  `[H2D pinned, unpack, gemv, epilogue, gemv, epilogue, D2H pinned]` per rows. Exactness: every ring operation is
  a wrapping 64-bit unsigned operation (`Z_{2^K}` is a quotient of `Z_{2^64}`), so tiles, splits and digit order
  cannot change the bits; `self_check` compares every variant with the Python-int reference.
- **Lane** (`pgw.service.fused2_lane`, `LaneManager`): a fused2 weight set is its own lane model whose table has
  one item per trip in pass order (`0.qkv/f2` single, `l.o+l.gate_up/f2` and `l.down+(l+1).qkv/f2` pairs, the
  last `down` single). A pair item's table widths are `n = n1 + m1`, `m = m1 + m2`; body `pack(x_hat) ||
  pack(R_hat)`, reply `pack(H_hat) || pack(Z_hat)`: the existing headers, tickets, admission, deadlines and UDS /
  shared-memory / TLS transports carry it unchanged (no Rust change in `pgw-lane`). The lane fingerprint
  (`fused2_fingerprint`) binds base pack, ring and every item's public `w8` / `hmul`. Entries come from a
  provider-only weights file (`PGW_SERVICE_FUSED2_WEIGHTS`) or are derived from the provider's own pack
  (`PGW_SERVICE_FUSED2_FROM_PACK=<bits>`: int8 fold + pack-time migration). On CUDA the lane captures one graph per
  (item, rows) and checks each through the native serving path against the eager op; the eager op runs under the
  same device lock as the graphs. `python -m pgw.service.fused2_lane serve` runs this without the HTTP front end
  for the co-located harness (writes one 0600 ticket file).
- **Gateway** (`pgw.engine.fused2_lane.LaneFused2Provider`): the provider interface over a native `TripChannel`
  (UDS or shm); calls above the lane's 8 rows are split (rows are independent). The masked backend is unchanged.
  **Native p61 check** (`pgw_fast.p61_check`, `rust/pgw-fast/src/p61.rs`): range + Freivalds for both stages, the
  numpy path's verdicts, default on when the extension has it (`PGW_FUSED2_NATIVE_CHECK=0` turns it off). The
  numpy path itself moved to 21-bit limbs and a Mersenne fold.
- **Masks without weights on the gateway** (`pgw.engine.fused2_masks`): a customer-side plain factory
  (`Fused2MaskFactory`, its own process: `python -m pgw.engine.fused2_masks factory`) writes pair-mask segments
  (per trip key the ChaCha20 stream keys of `r_x` and `v`, the W1 correction `W1 r_x` and the W2' correction
  `W2' v`, packed K-bit, SHA-256 trailer) and the secret check vectors once; the gateway (`Fused2SegmentMasks`)
  derives `r_x`, `v` from the keys and `r_R = v - hmul * W1 r_x` itself, and holds no int8 matrix
  (`gateway_tables`: stride-0 views). Same rows as the in-process source for the same seed; single use. Not yet
  the encrypted maskstore format (AES-GCM segments, counters) nor an HE factory for K56.

Measured here (4-vCPU box, CPU only; GPU numbers need `benchmarks/fused2_pair_bench.py` on an H100): gateway check
per 7B layer (4 checks, 1 row): numpy 2.8 ms -> 0.9 ms (21-bit limbs) -> **0.17 ms native**; masking + packing
0.2 ms, unpacking + unmasking 0.45 ms per layer (numpy, 7-byte elements).

Still missing: the native decode driver (`rust/pgw-fast/src/decode.rs`) has no fused2 pass (the harness gateway is
the Python `TransformerModel` with the torch float ops, about 1-2 ms of Python per trip at 7B), the maskstore / HE
factory for pair segments, the provider LM head folded into the last trip, speculation padding.

## 10. Update 2026-11-23: digit-plane tensor-core pair op, native fused2 decode driver, per-token budgets (opt-in)

Measured before (H100, `benchmarks/results/fused2/pair_op_h100_2026-11-23.json`): the graphed K56 pair op at 1 row
costs 0.217 ms (`o+gate_up`) and 0.154 ms (`down+qkv`), about 2.0x the two graphed K32 single trips it replaces
(0.105 / 0.080 ms); 10.4 ms vs 5.2 ms of provider compute per 7B token. The int8 weight bytes are the same, so
the extra cost is the 64-bit CUDA-core arithmetic of the `sum` kernel (and the old `dot` kernel was worse: 7 thin
dots per tile, 8-byte input reloads per M tile, digit peeling in the loop: 1.57 ms at 4 rows).

**Provider: `planes` variant** (`pgw.core.ring64`, `PGW_SERVICE_FUSED2_VARIANT=planes`; opt-in, `auto` unchanged
until the H100 numbers are in):
- the unpack kernel writes each input element directly as 8 balanced int8 digit planes (`d[(row * 8 + i), k]`,
  `x = sum_i d_i 256^i mod 2^64`; the residual `R` as int64), and stage 1's epilogue writes `H` the same way;
- the GEMV is ONE int8 tensor-core `tl.dot` per tile, `W (BLOCK_M, BLOCK_K) @ D^T (BLOCK_K, 8 * rows_pow2)`
  (16 columns for 1-2 rows, 64 for 8), accumulated in **int32 over the whole split** (exact while
  `128 * 128 * n_per_split < 2^31`, enforced by the host: `PLANES_MAX_SPLIT_N = 131071`; 7B `down` is 18944,
  72B 29568), recombined once per program (`sum_i acc_i << 8i`, wrapping uint64). Weight bytes are read once
  (the memory-bound part, as K32); the digit planes are `8 * rows_pow2` bytes per input element and stay in L2;
- unsplit stages (e.g. 7B `gate_up`, 592 tiles of 64 rows) run the epilogue in the GEMV itself (`* hmul`, `+ R`,
  `mod 2^K`, packed reply bytes, next stage's planes): 3 kernels per pair trip, 5 when both stages are split;
  CUDA graphs unchanged (`PairOp.capture`), the lane's graph check unchanged;
- split-K heuristic: 64-row tiles (128 above 4 rows), 256-byte K steps, about two programs per SM, splits of at
  least 4 K steps. Exactness: Python-int reference, interpreter tests for 1-16 rows, K 32/48/56/64, forced splits,
  extreme operands, buffer reuse across row counts (`tests/core/test_ring64.py`); `self_check` covers `planes`.

H100 micro-benchmark (one command, <= ~2 min of GPU, ~$0.13; NOT run here: no Modal spend in this session):

    modal run deploy/modal/gpu_bench.py --target h100_c4 --timeout-min 2 \
        --cmd "python benchmarks/fused2_pair_bench.py --rows 1,2,4,8 --reps 30 --variants planes,sum \
               --budget-s 50 --json /tmp/f2planes.json"

It prints per item and rows the heuristic of each variant, then sweeps tiles / splits while the budget lasts,
checks every configuration byte for byte against the prototype provider, and summarizes pair / (two K32 singles)
at 1 and 8 rows per variant. Expected (not measured): close to the K32 singles (the same weight bytes, one MMA
per tile); if 1-row `o` (56 tiles) is launch- or split-bound, the sweep's `x2` split entry shows it.

**Gateway: native fused2 decode driver** (`pgw_fast.Fused2Driver`, `rust/pgw-fast/src/decode_f2.rs`,
`pgw.engine.fused2_native`; `enable_fused2_native(model)` or `PGW_FUSED2_DECODE_NATIVE=1` via `maybe_enable`):
one GIL-free native call runs a whole fused2 pass of one sequence (<= 8 rows): the `2L + 1` trips on the native
`TripChannel` (UDS or shm; the lane items of section 9), per trip the quantization with the exact bounds (numpy's
float64 roundings and pairwise sums reproduced bit for bit), masking with the pass's mask rows (taken by Python
from `Fused2MaskSource` / the factory's pair-mask segments: one slot per token, weight-free gateway), 7-byte
pack / unpack / unmask / centering, native p61 range + Freivalds checks of both stages before any output is
used, then the float work on the native float ops (RoPE + attention + KV write, SiLU) and the fixed-point residual.
Sampled noise trips hand back `(x_hat, xq)` and end the driver call, so their statistics run before the next
payload is built; duplicate-row fingerprints are recorded before a payload leaves when the backend has a monitor.
fused2 now also runs on `float_ops = "native"` in Python (the reference the driver matches). The head runs after
the layers as before (masked head not folded into the last trip).
Tests (`tests/engine/test_fused2_native.py`, real `LaneManager` + Triton op on CPU): native == Python fused2 ==
plain, bit for bit, over UDS and shm, prompt and decode; byte-identical provider-visible bodies (same mask seed);
one mask slot per token with segment masks; tamper (H, consistent H, Z, Z without residual, skipped multipliers)
caught natively; a sampled trip ends the call; the `planes` service op behind the lane gives the same logits.
Not measured: the native gateway's time at 7B shapes (expected well under 1 ms per layer: the p61 checks were
0.17 ms per 7B layer, masking and unpacking are a few passes over ~80k elements per layer).

**Per-token budgets (1 row decode, provider waits only; the gateway's own work, ~10.8 ms/token at 7B on the 32-core
gateway today, comes on top for both).** Device times scale the measured 7B K32 singles (0.186 ms per layer, 233 MB
of int8 weights) by weight bytes (32B: 488 MB/layer, 72B: 878 MB/layer: conservative, larger matrices stream
better); fused2 "now" = the measured 2.0x, "planes" = the target 1.0-1.3x (projected, needs the benchmark above).
Co-located: the shm lane's ~0.09 ms fixed cost per trip (measured; it reproduces the measured 15.7 ms wait of the
7B 28 tok/s run). Separate host: RTT per trip plus bytes at 50 Gbit/s (K32 4 B, K56 7 B per element).

| model | path | trips | device ms | co-located shm ms | RTT 0.1 ms | RTT 0.2 ms | MB/token |
|---|---|---|---|---|---|---|---|
| 7B | standard K32 | 112 | 5.2 | 15.3 | 17.8 | 29.0 | 8.9 |
| 7B | fused2 now (2.0x) | 57 | 10.4 | 15.5 | 18.6 | 24.3 | 15.6 |
| 7B | fused2 planes 1.0-1.3x | 57 | 5.2-6.8 | **10.3-11.9** | 13.4-14.9 | 19.1-20.6 | 15.6 |
| 32B | standard K32 | 256 | 24.9 | 47.9 | 55.2 | 80.8 | 29.6 |
| 32B | fused2 now | 129 | 49.7 | 61.3 | 70.9 | 83.8 | 51.8 |
| 32B | fused2 planes 1.0-1.3x | 129 | 24.9-32.3 | **36.5-43.9** | 46.1-53.5 | 59.0-66.4 | 51.8 |
| 72B | standard K32 | 320 | 56.0 | 84.8 | 95.1 | 127.1 | 44.8 |
| 72B | fused2 now | 161 | 111.9 | 126.4 | 140.5 | 156.6 | 78.3 |
| 72B | fused2 planes 1.0-1.3x | 161 | 56.0-72.7 | **70.4-87.2** | 84.6-101.4 | 100.7-117.5 | 78.3 |

Reading: at today's 2.0x the pair op only breaks even co-located (7B) and loses at 32B/72B; at 1.0-1.3x fused2
saves 3.4-5.0 ms per 7B token co-located (15.3 -> 10.3-11.9 ms of waiting; on the measured 28 tok/s run, 35.7 ms
per token, that is about 31-32 tok/s if the gateway work stays the same), up to 14 ms at 72B co-located (-2 at 1.3x)
and 9-26 ms at 72B with a 0.2 ms RTT. Its 1.75x bytes
per element make it lose on a slow link (RTT 0.1 ms at 50 Gbit/s: roughly even at 72B). The benchmark decides
which side of 1.3x the op lands on.

## 11. Update 2026-11-23: production mask supply for fused2 (encrypted store, factory, gateway; opt-in)

Goal: a weight-free CPU-only gateway gets its fused2 pair masks exactly as it gets today's masks: encrypted store
segments filled by the customer-side factory, durable single use, (incremental) reservations, crash safety, burn
on close. All default off; existing formats and paths unchanged; masked == plain bit for bit.

**Format** (`pgw.maskstore.format`, SPEC 7.2): a fused2 pair-mask segment is today's segment file with
`schedule = "fused2"` and `check_field = "p61"` (mandatory together; both omitted from standard headers, so their
bytes are unchanged; readers that do not know `schedule` or `p61` refuse the header, `extra = "forbid"`, tested
with a reader model without the field). Entries: the fused2 linears in pass order (`0.qkv/f2, 0.o, 0.gate_up/f2,
0.down, 1.qkv/f2, ...`), each one single-stage product, because the no-chain correction needs nothing else
(section 2): `r`, `W r` = `r_x`, `W1 r_x` for a trip's first linear and `v`, `W2' v` for a pair's second; the
gateway forms `r_R = v - hmul * W1 r_x` itself from the public `hmul`. `r` is derived exactly as today
(`ring_uniform_rows(n, K, slot, 1, "r", key)`), so the store's reservation, extent, chunk-cache, prefetch and burn
machinery is reused unchanged. Check words: `uW = (u * hmul) W mod (2^61 - 1)` per linear, stored as 8-byte words
(values up to 2^61 do not fit K56 elements; `SegmentLayout.check_bpe`), `u = u64le(stream("u-p61", key)) mod p`
(`pgw.maskstore.fused2.p61_check_vectors`). The header fingerprint is the fused2 lane fingerprint (binds base pack,
ring and every `w8` / `hmul`), so fused2 and standard segments never share an inventory key; in addition a
reservation records its segments' schedule and `reserve(..., schedule=)` refuses a mismatch (fail closed: the slots
stay consumed). Header rules: fused2 ids only with the schedule, p61 only with the schedule, a capped ring with
`2^(C+1) < p`, `checks >= 1`; the import soundness floor uses `checks * (log2 p - log2(1 + 2^-3))` (60.8 bits per
vector and stage). Not supported for fused2: the in-memory mask ring (`ringbuf`) and the HE factory (below).

**Factory** (`pgw.factory.fused2_gen.Fused2MaskGenerator`; `python -m pgw.factory.fused2_gen generate|bench`): holds
the public fused2 weights (a provider-only weights file suffices; never anything of the provider's), writes
fresh-seeded segments through `MaskStore.writer(..., schedule="fused2", check_field="p61")` (or an export sink).
Products `W r mod 2^K` by `PreparedWeight` (exact: `cpu_f64` digit GEMMs on the CPU, `cuda_int8` tensor-core digit
GEMMs on a customer GPU, i.e. the GPU mask service's arithmetic; for K a multiple of 8 up to 56 every step is
wrapping 64-bit integer arithmetic); `uW` by three 21-bit limbs of `u * hmul mod p` times the int8 weight in float64
BLAS (every partial sum an integer below 2^53: exact in any order), recombined mod p. Guards: `startup_check()`
(production device vs numpy uint64 wrapping products for fresh rows of sampled linears, `uW` columns vs Python
integers; bit for bit or `IntegrityError`), a per-chunk self-check before every write with private ring vectors
(`u_s . (W r) == (u_s W) . r mod 2^K`, numpy uint64, independent of the GEMM backend; sampled rows, `-1` all), abort
on any error (never committed), device-memory scrub after every segment on CUDA, secrets (`r`, stream keys, `u`,
`uW`) zeroed. Not wired into `FactoryScheduler` / `pgw factory mask-service` yet (its own process entry point).

Throughput, CPU (this 4-vCPU Xeon @ 2.3 GHz, `cpu_f64`, Qwen2.5-7B shapes with random public weights, one layer
measured and scaled by 28, a 256-slot segment encrypted into a real store; `bench --model qwen2.5-7b --layers 1
--slots 256 --store-root DIR`): **5.3 slots/s for the full 7B model**, steady state (6.8 ms per slot and layer:
GEMMs 6.0 ms at 39 GMAC/s, ChaCha20 0.2 ms, self-check 0.02 ms, AES-GCM + write 0.55 ms), plus 0.58 s per layer and
segment for `uW` (16 s per 7B segment: ~2% at 4096 slots) and a one-time 1.4 s per layer for the private self-check
vectors. A fused2 slot has the same MACs as a standard slot (6.5 G for 7B); K56 costs two f64 digit GEMMs per
product (without the store write: 6.1 slots/s at K56 vs 7.2 at K32 with the same generator; the f64 GEMMs are
partly bound by streaming the float64 weights, so the second digit costs ~20%, not 2x). Demand at the measured 28 tok/s with incremental reservations (~1.2-1.3 slots per token) is ~35 slots/s:
~26 such cores on the CPU path, so 7B+ wants the GPU mask service. GPU estimate (not measured, no GPU here): the
`cuda_int8` product at K56 runs 7 int8 digit GEMMs where K32 runs 4, so the measured T4 mask-service rate for 7B
(164-174 slots/s at K32, STATUS 2026-11-22) scales to **~95-100 slots/s on a T4** (L4/A10G 1.5-1.9x); `r` is
expanded on the host (native ChaCha20) and crosses PCIe (6.6 MB of `r` up, 11 MB of `W r` down per 7B slot: a
~600 slots/s ceiling at 12 GB/s), so a larger GPU is host-bound before it is GEMM-bound (moving the K56 PRNG to the
device, as for K32, lifts that). One command to confirm (not run): `python -m pgw.factory.fused2_gen bench --model
qwen2.5-7b --layers 2 --slots 1024 --device cuda`.

**Gateway** (`pgw.engine.fused2_store`): `Fused2MaskedBackend(ring, provider, weights=gateway_tables)` no longer
needs an in-process mask source: a call whose session is a store reservation (`MaskReservation`, or a
`WindowedReservation`, incremental included: `open_fused2_session(store, lane_fp, ring, tables, prompt_rows, cap=,
chunk=)`) takes its pair rows from it (`Fused2StoreMasks`): per trip `take_checked` of the first linear, then the same
number of rows of the second (rows and the check vectors of the extent they came from, atomically; misaligned rows
fail closed). A take never crosses a segment extent, so a trip may run in row pieces (rows are independent; each
piece is masked, checked and wiped like a whole trip). Check vectors are copied once per (segment, linear) into
arrays the source owns (a concurrent close that wipes the store's shared copies cannot turn a check into `u = 0`;
an all-zero `u` is refused), replaced and zeroed when the segment changes, and zeroed when the session is released
(`backend.release(session)`, or swept once closed). The native driver (`Fused2NativeDecode`) takes a whole pass from
the session when every linear's rows fit the current extent (else the Python path runs that pass in pieces),
installs the driver's check vectors only when the segment (token) changes, and `wipe_checks()` zeroes the driver's
copies (also after a pass of a closed session; a driver that served a store session refuses a static source
afterwards). One slot per token: every linear's cursor moves by the pass's rows. `benchmarks/fused2_e2e.py --masks
store` runs the chain: the factory process fills an encrypted store, the gateway (weight-free tables) decodes from
an incremental reservation, the plain replay checks exactness. The gateway service itself still refuses
`schedule = "fused2"` (`PENDING_SPEED_SETTINGS`); wiring its request path to `open_fused2_session` is the next step.
Measured end to end (Qwen2.5-0.5B-Instruct, this box, CPU provider over loopback TCP, `--masks store --tokens 3`):
the factory process wrote one 50-slot encrypted segment (start-up check passed, 4.4 slots/s while the other
processes ran), the gateway decoded from an incremental reservation on weight-free tables: 49 trips per token,
1.0 slot per token, 384 checks passed, logits bit-identical to the plain replay at 4/4 positions (max diff 0.0).

Tests (`tests/engine/test_fused2_store.py`, CPU; a real store, the real native lane for the driver): header round
trip; readers without the field refuse; inconsistent headers and writer arguments refused; a standard reservation
refuses fused2 segments (fail closed); stored rows, products and p61 words vs Python-integer references; start-up
check and per-chunk self-check abort on one flipped bit (nothing committed); masked == plain bit for bit via the
store (Python path with a prompt crossing segment extents, an incremental windowed session, the native driver over
the lane with every decode pass native), one slot per token; tampering caught and the failed pass's rows never
reused; a gateway process killed after taking rows: its slots stay consumed and the next reservation gets other
rows; burn on close. The existing maskstore, factory and fused2 suites pass unchanged.

**HE note: what an encrypted (HE) factory for K56 needs (not implemented).** The encrypted factories compute
`W8 r mod 2^K` blind: v1/v2 (`pgw.hemask.lwe`) encrypt `r` scaled by `Delta = 2^(q-K)` at `q <= 2^64` and need
`q - K >= 24` bits of room for the noise `W8 e`, so `K <= 40`; pks-v3 (`pgw.hemask.pks`) uploads at `2^56` with
`Delta = 2^24` and returns `W8 r mod 2^32` through a `q1 ~ 2^48` packed reply (K32 only). K56 masks therefore need
either
(a) **a direct scheme with `q >= 2^80`** (56 message bits + 24 noise bits): ciphertext arithmetic above 64 bits
(two-limb, or RNS with at least two ~40-bit primes plus the key-switching special modulus), the provider's GEMM
`W8 B mod q` per RNS limb (int8 digit GEMMs per limb: about 2x today's provider work), v2 hints `H = W8 A` or pks
compact hints and a reply modulus of about `2^(56 + 16)` for the 32-row packing, ring dimension and error re-chosen
for 128-bit security at `log2 q ~ 84` (an MLWE dimension of 4096 is still within the HE-standard bound for ternary secrets), a
new noise analysis and an external review; or
(b) **digit decomposition on the existing schemes (no new cryptography)**: write each uniform `r = sum_i 2^(b i) d_i`
with independent uniform digits `d_i < 2^b`, let the existing factory compute each `W8 d_i` blind, recover each
`W8 d_i` as an exact integer (`|W8 d_i| < 2^(b + 7 + log2 n)`, so its value mod `2^K_he` is exact after centering
when `b + 7 + log2 n < K_he - 1`) and recombine `W8 r = sum_i 2^(b i) (W8 d_i) mod 2^56` on the customer side. v2
at K40: `b = 14` (`14 + 7 + 14.9 < 39` up to 72B's `n = 29568`), 4 products per mask; pks-v3 at K32: `b = 8`
(`8 + 7 + 14.9 < 31`), 7 products per mask. HE work, upload and reply traffic per slot grow 4x (v2 at K40 vs K32)
or 7x (pks-v3); privacy is unchanged (each digit plane is a uniform message under a fresh MLWE secret); the factory's
private per-row self-check runs on the recombined `W r mod 2^56`; the p61 check words `(u * hmul) W mod p` need no
HE (public `W`, customer `u`: 0.58 s per 7B layer and segment on this 4-vCPU box). Option (b) is the practical path (reviewed
schemes, 4-7x the HE cost: pks-v3 measured ~1.8 customer cores for ~35 slots/s at K32, so ~13 cores at K56), unless
the GPU mask service is available, which needs no HE at all.

## 12. Update 2026-11-23: fused2 in the gateway service, the factory scheduler, HE by digits, the launcher (opt-in)

**Gateway service.** `schedule = "fused2"` is accepted (removed from `PENDING_SPEED_SETTINGS`). Start-up: the pack is
verified as today; the public fused2 weights are derived from it (`Fused2Weights(pack, multipliers=
migration_multipliers(pack, fused2_migrate_bits))`, the same derivation as the service's
`PGW_SERVICE_FUSED2_FROM_PACK` and the factory's) or loaded (`fused2_weights_path`, base fingerprint checked); the
gateway computes the fused2 lane fingerprint itself and keeps only the weight-free tables. Connect: the base pack
must be served (`/models`), then `POST /pgw/v1/lane/open` for the lane fingerprint; the ticket must name the same
fingerprint, the gateway's item table in pass order, the K56 ring and a socket in `lane_uds_dir` / of a loopback
service (UDS) or a port on the egress allowlist (TLS, verified against `service_ca_file`). A service without that
entry keeps the model `connecting` (retried); pair items have no fallback channel. Requests: the reservation is a
fused2 reservation of the lane fingerprint (`schedule="fused2"`: a standard segment is refused, fail closed),
window or incremental exactly as today; `Fused2MaskedBackend.masks_for(session)` takes pair rows from it;
single-sequence passes of <= 8 rows run in the native driver on the plaintext thread (`sync_trip_ok`), others in
Python; every payload passes the noise monitor; a failed range / Freivalds check trips the circuit breaker; the
check-vector copies are wiped at session release and stop. The masked head is not combined with fused2 (it would
need a standard K32 head reservation next to the fused2 one per request; the head stays the gateway's float head).

**Factory.** `FactorySettings.schedule = "fused2"` (`--schedule fused2`): `build_generators` returns one
`Fused2MaskGenerator` per (pack, ring) keyed by (lane fingerprint, ring); the scheduler drives it unchanged (its
`generate_into_store` takes `chunk_slots`, `throttle` (stop and rate / duty cycle accounted per chunk in
slot-equivalents), `store_chunk_bytes`); a start-up reference check before the first segment (the mask service's
`--check-only` reports it). The setup wizard presets (`schedule`, `ring`, `fused2_migrate_bits`, ...) produce the
fused2 factory environment (`PGW_SETUP_FUSED2_FACTORY=plain` drops HE for the plain customer-side CPU factory).

**HE by digit decomposition (section 11 option (b), implemented: `pgw.factory.fused2_he`).** For a uniform row
`r < 2^56` and `b`-bit digits `d_i = (r >> b i) & (2^b - 1)`: `W r = sum_i 2^(b i) W d_i` over Z; `|W d_i| < 128 *
2^b * n <= 2^(K_he - 1)` (checked per linear before any request) so the centered `W d_i mod 2^K_he` returned by the
encrypted factory is the integer `W d_i`; the recombination runs in wrapping uint64 arithmetic, exact mod 2^64 hence
mod 2^56. Profiles: mlwe v2 K40 / b = 14 / 4 products (n <= 2^18), pks-v3 K32 / b = 8 / 7 products (n <= 2^16).
Provider's view: each digit plane is encrypted like any K40 / K32 mask row (a valid ring element under a fresh
secret and noise; the scheme hides any message), as an ordinary maskgen item (`<layer>.qkv-f2` etc. under the
fused2 lane fingerprint, `PGW_SERVICE_FUSED2_MASKGEN=1` on the provider: `Fused2MaskgenModel` = the lane entry's
public int8 matrices as ordinary `ComputeKernel`s); the request count is `digits x` that of a K40 / K32 segment,
public. Decryption correctness is the scheme's own (same Delta, noise check per linear at start-up); the fused2
factory's per-chunk private self-check and start-up check run on the recombined products. Tests: the arithmetic
against numpy uint64 references (extreme rows and digits), and the real `HEMaskBackend` (rank 16, q = 2^64) and
`PKSMaskBackend` (the service's v3 item code in-process) over the tiny model's fused2 linears: segments equal to the
plain factory's bit for bit (rows, products, p61 words), each request of public shape.

Customer CPU per slot, Qwen2.5-7B shapes, this 4-vCPU box, provider time excluded
(`benchmarks/fused2_he_split_bench.py`; random public int8 weights; each linear keeps its full `n` and 2048 output
rows (plain: 4096), scaled linearly to the full `m`: an upper bound, small `m` amortizes fixed costs worse;
`benchmarks/results/fused2/he_split_7b_cpu_2026-11-23.json`, `plain_factory_7b_cpu_2026-11-23.json`):

| factory | ms per slot and layer | s per full 7B slot (28 layers) | x plain |
|---|---|---|---|
| plain CPU fused2 (cpu_f64 products) | 4.7 | 0.13 | 1.0 |
| mlwe v2 K40, 4 x 14-bit digits (ntt_pre encryption) | 10.2 | 0.29 | 2.2 |
| pks-v3 K32, 7 x 8-bit digits | 5.9 | 0.17 | 1.3 |

(Encryption depends on `n` only and dominates `down`; decryption scales with `m`.) So at the 28 tok/s decode rate
with incremental reservations (~35 slots/s) the encrypted fused2 factory needs on the order of 6-10 such 4-vCPU
boxes' worth of CPU (pks-v3 / mlwe), against ~5 for the plain CPU factory: the GPU mask service (no HE) remains the
7B+ path.

**Launcher.** `launch.py --placement colocated --schedule fused2 [--fused2-factory he|plain] [--fused2-migrate-bits
8]`: the provider serves the fused2 lane entry from its pack (+ maskgen for the encrypted factory), the wizard gets
the fused2 presets, the factory makes fused2 segments (its own fused2 hint on first start, cached by
`--gateway-cache`), the e2e's reference is the fused2 plaintext reference; dry-run tested. 7B command: see
`deploy/modal/README.md`, "fused2 in the co-located demo". Not run here.
