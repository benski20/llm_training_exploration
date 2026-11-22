# Compact-hint encrypted masks at 7B-72B: survey, `pks-v3` prototype, recommendation

Date: 2026-11-22. Local only (no Modal, no Kaggle): one shared 4-vCPU Xeon @ 2.3 GHz (AVX-512, IFMA, AMX), load
average 20-33 during the runs, so every number below is **single-thread process CPU time** (wall time on this host
includes waiting for a core). Code: `src/pgw/hemask/pks.py` (numpy reference, exact), `rust/pgw-pks` (customer
kernels + `pks-bench`), `tests/hemask/test_pks.py`, `benchmarks/hemask_pks_model.py`, presets for 32B/72B added to
`benchmarks/hemask_decrypt_bench.py` (which now also reports process CPU). Nothing is wired: factory, service,
wire and defaults are unchanged.

## 1. The problem in numbers (today, SPEC 7.9)

Masks are `W8 r mod 2^32` per linear and slot. The customer encrypts `r` (MLWE, rank 16 x degree 256, q = 2^56,
256 slots per group), the provider returns `W8 b`, the customer subtracts `H s` with the hint `H = W8 A`
(4096 coefficients per output row, 24.6 KB per row stored as 6 byte planes, 49 KB in NTT form).

Per slot (sum over all linears, masked LM head excluded): 7B sum m = 1.39M, sum n = 0.83M; 32B 4.65M / 2.75M;
72B 6.86M / 4.33M. Demand at 25-30 tok/s with 2-3 slots per token: **50-90 slots/s**. At 72B that is 27.4 MB of
masks per slot, 1.4-2.5 GB/s.

| model | hint (planes / NTT form) | decrypt CPU per slot, measured (ntt / ntt_pre) | encrypt + r per slot |
|---|---|---|---|
| 7B | 34 / 68 GB | 54 / 23 ms | 31 ms |
| 72B | 169 / 337 GB | 268 / 113 ms | 164 ms |

(`hemask_decrypt_bench.py --groups 4 --max-rows 1024`, `PGW_NTT_THREADS=1`: 10.0 us (ntt) and 4.2 us (ntt_pre) CPU
per hint row and group, the same at every shape; encryption `PretransformedEncryptor`, 4 groups: 32 ns CPU per input
and slot incl. CBD noise, plus 5.8 ns for the `r` keystream.) Two problems: the hint is O(rows x 4096) and must be
streamed once per pass (a pass of 512 slots at 90 slots/s reads the whole 169 GB hint every 5.7 s: RAM, not disk),
and the decryption is `rank` NTT products per output and slot.

## 2. Survey: computing `W r` privately with a hint-free or small-hint customer

Criteria: 128-bit security with a standard parameter set; provider GPU work; bytes per slot; customer CPU per slot;
customer storage; exactness of `W r mod 2^K`. "per output-slot" = per (output row, slot).

| option | security | provider work | up / down per slot | customer CPU | customer storage | exact | verdict |
|---|---|---|---|---|---|---|---|
| A. today (MLWE + hint) | MLWE n=4096, log q 56: HE-std 128 (limit 109), core-SVP 239 | 7 int8 planes x sum(mn) | 6.6-7 B x n / 4.4 B x m | 16 NTT MACs x 3 primes per output-slot (16-39 ns) | 24.6 KB per output row (169 GB at 72B) | yes | storage and CPU do not scale |
| B. A with the hint streamed by the provider | same | + read/send H per pass | + 4096 x 7 B per row per pass: ~14x the reply at 512-slot passes | same | none | yes | bandwidth: rejected |
| C. Cheetah/BOLT/Iron coefficient encoding (one vector per RLWE ct, PackLWEs) | RLWE std | ct-pt poly products + automorphism key switching | for T slots per ring of N: (N/T)/n_b up, (N/T)/m_b down with n_b m_b <= N/T, plus a fresh c1 per ct (2x down) | 1 NTT product per coefficient | none | yes (coefficient encoding allows t = 2^32) | built for one fresh vector per ct; for 256-slot batches the slot-in-coefficient layout of A is strictly better on bytes; BFV slot (SIMD) packing needs a prime plaintext modulus, not 2^32 |
| D. RLWE N=4096, 4096 slots per group (plan he-structured-encryption 5.1) | RLWE std | unchanged kernel | unchanged | ~3x fewer ops than A | unchanged (hint per row = dimension) | yes | does not fix storage; 4096-slot groups = 112 GB of masks per group at 72B |
| E. multi-secret (PVW) packed LWE | LWE std | unchanged | unchanged | `dim` MACs per output-slot (dense) | hint | yes | far too slow |
| F. smaller q / smaller dimension | needs `q >= 2^(K + 23)` for `W e`; n=2048 allows log q 54 only | - | -5-10% | -50% | -50% | yes | not worth a parameter change |
| G. partner factory (plaintext `W r` on a non-colluding GPU, SPEC 7.7) | non-collusion, not lattice | provider-independent GPU | masks only (4 B x m) | none | none | yes | exists; trust model differs |
| **H. pks-v3 (this note)** | upload RLWE n=4096 log q 56 (as A); reply key RLWE n=4096 log Q 108 (HE-std 128 classical limit 109; production: n=8192, core-SVP 218) | A + mod switch, 16-row packing, hybrid key switch: ~100 modmul per output-slot (+384 if A~ is re-transformed every group) | 7 B x n + reply key (16 x N x 13.5 B per linear and group: +3.6% at 72B) / 4.5 B x m | 1 NTT product per 16 rows: **8.8 ns** per output-slot measured (op-count floor ~4-5) | **1.5 KB per output row** (10.5 GB at 72B; 0 if streamed at +6 B x m down) | yes (tested bit-exact at 72B widths) | **recommended to take to review** |

Seeded `A` (never uploaded) and the randomised upload switch (bb = 53) are already in today's scheme; both stay
compatible with pks-v3. Moving NTT-heavy work to the provider is exactly what pks-v3 does.

## 3. The pks-v3 scheme (prototype: `pgw.hemask.pks`)

Notation: `R_256 = Z[X]/(X^256+1)`, `R = Z[Y]/(Y^4096+1)`, embedding `iota: X -> Y^16`.

**Upload (customer, per linear and 256-slot group).** Fresh ternary `t in R`, CBD(21) noise. For each block of 16
features `C`: `B_C = a_C t + e_C + 2^24 M_C mod 2^56`, `M_C[16u+k] = r[u, 16C+k]`, `a_C` from a customer-chosen
seed. Sent as today's `(n, 256)` bodies `b[16C+k, u] = B_C[16u+k]`. In the provider's view feature `16C+k` is a
module-LWE sample under the 16 components `t_l` of `t` with `a`-part row `k` of the multiplication matrix of `a_C`
(`a_mlwe_rows`). **Tested:** today's provider kernel and today's hint decryption (`lwe.decrypt_wr` with
`H = W8 a_mlwe_rows(a)`) recover `W8 r` exactly from these uploads, and the bodies minus the pad are exactly the CBD
noise plus `2^24 r` (`test_upload_is_old_format_mlwe`). The customer's pad is one length-4096 product per 16
features instead of 16 degree-256 products per feature (3 primes, exact CRT, `pad_block`).

**Reply key (customer, per linear and group).** Fresh ternary `t' in R`; `beta_l = k_l t' + e_l + P iota(t_l) mod q1 P`
for `l < 16`, with `k_l` public and seeded (sampled in the NTT domain), `q1 = p1 p2 ~ 2^48`, `P = p3 p4 ~ 2^60`
(primes `= 1 mod 8192`). 885 KB per linear and group (2x at N = 8192).

**Provider.** `y = W8 b mod 2^56` (unchanged GEMM). Modulus switch `y` and `H` to `q1`
(`round(x q1 / 2^56)`, exact integer helper); pack 16 rows: `body = sum_j Y^j iota(y'_j)`,
`A~_l = sum_j Y^j iota(h'_{j,l})` (pure re-indexing: coefficient `16u + j`); hybrid key switch
`C0 = P body - sum_l A~_l beta_l`, `C1 = sum_l A~_l k_l` (mod `q1 P`, RNS NTT), ModDown to `c0, c1 mod q1`;
send `c0` rounded to `yb = 36` bits (4.5 B per output-slot). `A~` is per (model, A seed): the provider holds it
(169 GB at 72B in coefficient form: the object the customer stores today moves to the party with the hardware),
or re-derives it, or a global nothing-up-my-sleeve `A` lets all customers share it (opt-in; per-customer seeds stay
the default).

**Compact hint.** `c1` depends only on public data (`W8`, `A`, `k`), so it is fixed per linear: 4096 coefficients
per 16 rows = **256 per output row** (1.5 KB at 48 bits, 2 KB in NTT form). The customer builds it once with
today's hint-build pass (`H` rows are streamed through the key-switch reduction and never stored), or downloads a
provider-built copy whose hash is reproducible (see 5).

**Decrypt (customer).** `v = round(c0r q1 / 2^36) + c1 t' mod q1` (one pointwise product + one inverse NTT per
prime and 16-row block), `W8 r mod 2^32 = round(2^32 v / q1)`, unpack `16u + j -> (slot u, row j)`. Wipes `t'`.

**Noise (`PKSParams.noise_terms`, same conventions as SPEC 7.9: 24-sigma / Hoeffding at 24).** Budget
`q1 / 2^33 = 2^14.99`. Worst case all-127 rows at n = 29568: `W8 e` 2^12.6, mod switch of `H s` 2^9.6, `c1 t'`
rounding 2^9.6, key switch 2^4.1, reply rounding 2^11: total 2^13.3, one bit margin kept (`noise_ok(margin_bits=1)`;
tested). At real row norms (0.37 x max) the total is 2^12.6.

## 4. Measurements

**Exactness.** `tests/hemask/test_pks.py` (11 tests, all pass): NTT vs schoolbook, exact 56-bit products, exact
rounding helpers, noise budget, old-format compatibility of the upload, full round trip (256 and 77 slots), and
with `PGW_PKS_REAL_SHAPES=1` the Qwen2.5-72B input widths n = 8192 and n = 29568 with all-127 weight rows
(worst-case norm), bit-exact `W8 r mod 2^32` (77 s). Rust: 5 tests (`cargo test -p pgw-pks`): NTT round trip,
decryption vs a 128-bit schoolbook reference on valid ciphertexts, batched == single, pad vs schoolbook.

**Customer CPU, same host, same method (single thread, process CPU).**

| kernel | ns per output (input) and slot |
|---|---|
| today decrypt, hint planes (`ntt`) | 39 |
| today decrypt, NTT-form hint (`ntt_pre`, 2x hint RAM) | 16.4 |
| **pks-v3 decrypt** (`pks-bench decrypt16`, 16 blocks interleaved, auto-vectorised) | **8.4-8.9** (unbatched 13.4) |
| today encrypt (`ntt_pre`, incl. CBD noise) | 32 per input-slot |
| pks-v3 upload pad only (unbatched kernel) | 19 per input-slot (+11.4 CBD noise: about equal to today) |
| CBD(21) noise / `r` keystream (ChaCha20, existing) | 11.4 / 5.8 per input-slot |

The pks decryption does ~14 modular multiplications per output-slot (2 primes x (1 + 6 butterflies)) against ~60
for today's; today's kernel is hand-tuned AVX-512, the prototype is auto-vectorised, so a tuned pks kernel should
reach ~4-5 ns. Encryption is now the larger customer cost; it is dominated by the pad and by noise sampling
(CBD(21) draws 64 bits per coefficient where 42 suffice; an AES-CTR or bit-exact 42-bit sampler is ~3 ns).

**Sizing** (`benchmarks/hemask_pks_model.py`; cores = CPU seconds per second at this host's single-thread speed,
excluding the existing every-row check and I/O; "projected" = `--pks-dec-ns 5 --enc-ns 18 --r-ns 3`, applied to both
schemes):

| model | scheme | customer hint | up / down per slot | CPU ms per slot (measured / projected) | cores @50 / @90 slots/s (measured) | cores @90 (projected) |
|---|---|---|---|---|---|---|
| 7B | today, planes | 34.2 GB | 5.5 / 6.1 MB | 86 / 72 | 4.3 / 7.7 | 6.5 |
| 7B | today, NTT-form | 68.4 GB | 5.5 / 6.1 MB | 54 / 40 | 2.7 / 4.9 | 3.6 |
| 7B | **pks-v3** | **2.1 GB** | 6.2 / 6.3 MB | 44 / 25 | 2.2 / 4.0 | **2.3** |
| 32B | today, planes | 114 GB | 18.2 / 20.4 MB | 286 / 239 | 14.3 / 25.7 | 21.5 |
| 32B | today, NTT-form | 229 GB | 18.2 / 20.4 MB | 180 / 134 | 9.0 / 16.2 | 12.1 |
| 32B | **pks-v3** | **7.1 GB** | 20.2 / 20.9 MB | 147 / 83 | 7.3 / 13.2 | **7.4** |
| 72B | today, planes | 169 GB | 28.7 / 30.0 MB | 431 / 359 | 21.6 / 38.8 | 32.3 |
| 72B | today, NTT-form | 337 GB | 28.7 / 30.0 MB | 276 / 204 | 13.8 / 24.9 | 18.3 |
| 72B | **pks-v3** | **10.5 GB** | 31.4 / 30.9 MB | 226 / 127 | 11.3 / 20.4 | **11.5** |

With the hint streamed instead of stored (zero customer storage) the download grows by 6 B per output-slot
(72B: 30.9 -> 72 MB per slot, 6.5 GB/s at 90 slots/s); storing 10.5 GB is the better trade. At 72B the masks
themselves are the remaining storage: 27.4 MB per slot, i.e. 2.5 GB per second of buffered demand at 90 slots/s.

**Provider (GPU).** The GEMM is unchanged: 7 int8 planes x sum(mn) = 46 G (7B), 218 G (32B), 491 G (72B) int8 MACs
per slot; at 90 slots/s and 72B, 44 T int8 MAC/s (a few % of an H100). pks adds per output-slot ~64 RNS pointwise
products + ~24 butterflies + ModDown (~100 modmul; +384 if `A~` is re-transformed every group instead of cached or
amortised over a pass): 0.7-3.4 G modmul per slot at 72B, 60-300 G/s at 90 slots/s, memory-bound, low single-digit %
of an H100. Provider memory: `A~` per (model, A seed), 34 GB (7B) to 169 GB (72B) host RAM or NVMe.

## 5. Security argument and open questions (for the cryptographic review)

* **Provider view** per (linear, group): the upload bodies `b` and the reply key `beta`. Hybrid: (1) `beta` is
  RLWE(n = 4096, Q = q1 P ~ 2^108, ternary `t'`, CBD(21)) under a fresh `t'` used nowhere else -> replace by uniform;
  (2) then `b` is RLWE(n = 4096, q = 2^56, ternary `t`, CBD(21)) -> uniform. No circular security: `beta` encrypts
  the upload secret's components under an independent key. Fixed public `k_l` and `a_C` across groups are the usual
  common-reference-string use (each instance has a fresh secret and fresh noise).
* **Parameters.** Upload: the HE-standard 128-bit set (n = 4096, log q 56 <= 109; core-SVP 239 in our model, same
  as today's MLWE). Reply key: n = 4096, log Q = 107.99 sits at the HE-standard classical 128-bit limit (109) with
  no margin and above its quantum limit (101); core-SVP 96.9 in our model (the standard's (4096, 109) point gives
  95.5). **Production should use the n = 8192 reply ring** (32 rows per ciphertext, same 256 hint coefficients per
  row, Q = 2^108 against a limit of 218, core-SVP 218): ~8% more customer work per coefficient, 2x reply-key bytes
  (or one reply key per group for all linears, see 6). The prototype uses 4096 to keep numpy fast; the algorithm is
  identical. Run the lattice estimator (`LWE.estimate(LWE.Parameters(n=8192, q=2**108, Xs=ND.UniformMod(3),
  Xe=ND.CenteredBinomial(21)))` and n = 4096, q = 2^56 for the upload) before sign-off.
* **Assumption change:** the upload is RLWE over the degree-4096 ring instead of rank-16 MLWE over degree 256.
  Both have the same dimension and modulus; RLWE is the setting the HE standard tabulates. Needs sign-off.
* **Exactness / failure:** the noise bound above (24-sigma, Hoeffding), checked per linear like v2's plan;
  every decrypted row is still verified by the factory's private every-row check, and online Freivalds is unchanged.
* **Hint integrity:** a wrong `c1` gives wrong masks (caught by the every-row check), but a failure that depends on
  `t'` would be a selective-failure channel. So the hint must be fixed before any `t'` exists: the customer builds it
  (default; the same one-time pass that builds today's hint, needs the int8 pack transiently), or verifies a
  provider-built copy against a reproducible hash (possible when `A` and `k` come from a public seed). A cheap
  algebraic check of a provider-built hint is an open item (the rounding of `H` to `q1` is not linear; doing the
  key switch at q = 2^56, which the n = 8192 ring allows, makes `C1 mod 2^56` linear in `W8` and Freivalds-checkable,
  the `mod P` part still needs the lift carries).
* **Not done / uncertain:** no lattice-estimator run here (offline); the 4096 prototype's reply-key margin; tuned
  AVX-512 kernels (numbers above are an auto-vectorised prototype on a loaded host); provider GPU kernels (only op
  counts); the KSK upload could be shared across the linears of a group with per-linear `a` seeds (a further
  assumption-neutral change, not prototyped).

## 6. Recommendation and migration path

1. **Recommend pks-v3 for review**, production variant n = 8192 reply ring. It cuts the customer hint 16x
   (72B: 169 GB -> 10.5 GB, RAM-resident), cuts decryption CPU ~2-4x (measured 1.9x vs today's best kernel, ~4x by
   op count) and keeps bytes and the provider GEMM unchanged. It does **not** remove the customer's encryption CPU,
   which becomes dominant: 72B at 90 slots/s needs ~20 cores today with prototype kernels, ~11-12 with tuned
   kernels (pad batched + cheaper noise sampler); 7B ~2-4 cores; 32B ~7-13 cores. CPU-only, no GPU.
2. Versioned, default off: scheme id `pks-v3`; maskgen wire `v: 3` items carry `ksk` (reply key) next to `b` and
   return packed `c0` (`m/16 x N x yb` bits); negotiated with `vs` exactly like v2, v1/v2 always served. Customer:
   `PGW_FACTORY_HE_SCHEME=pks-v3` plus an acknowledgement string (like `mlwe-r12`) until the review signs off;
   provider: `PGW_SERVICE_MASKGEN_V3=false` by default. Hint stores keyed by scheme and ring degree; the old hint
   path stays for v1/v2.
3. Order of work: (a) review of section 5 + estimator runs; (b) n = 8192 reply ring in `pks.py` (parameter, not a
   new algorithm); (c) provider GPU kernels (mod switch, pack, RNS key switch, `A~` cache) behind the v3 route;
   (d) tuned customer kernels in `pgw-pks` (AVX-512 IFMA or hand-written u32 Montgomery, batched pad, 42-bit CBD
   sampler) with Python bindings; (e) factory/hint-build integration and the every-row check unchanged; (f) A/B on
   one co-located 7B run (exactness bit-identical to v1 masks is automatic: same `W8 r`), then 72B sizing.
