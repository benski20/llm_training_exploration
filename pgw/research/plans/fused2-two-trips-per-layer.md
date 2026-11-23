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
