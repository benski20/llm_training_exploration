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
| **H. pks-v3 (this note)** | upload RLWE n=4096 log q 56 (as A); reply key RLWE n=4096 log Q 108 (HE-std 128 classical limit 109; production: n=8192, log Q 108 against an HE-std limit of 218, core-SVP 251) | A + mod switch, 16-row packing (32 rows at the production n = 8192), hybrid key switch: ~100 modmul per output-slot (+384 if A~ is re-transformed every group) | 7 B x n + reply key (16 x N x 13.5 B per linear and group: +3.6% at 72B) / 4.5 B x m | 1 NTT product per 16 rows: **8.8 ns** per output-slot measured (op-count floor ~4-5) | **1.5 KB per output row** (10.5 GB at 72B; 0 if streamed at +6 B x m down) | yes (tested bit-exact at 72B widths) | **recommended to take to review** |

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
(`round(x q1 / 2^56)`, exact integer helper); pack 16 rows (this paragraph describes the n = 4096 prototype; the production n = 8192 reply ring packs 32 rows, `PACK = NR / 256`): `body = sum_j Y^j iota(y'_j)`,
`A~_l = sum_j Y^j iota(h'_{j,l})` (pure re-indexing: coefficient `16u + j`); hybrid key switch
`C0 = P body - sum_l A~_l beta_l`, `C1 = sum_l A~_l k_l` (mod `q1 P`, RNS NTT), ModDown to `c0, c1 mod q1`;
send `c0` rounded to `yb = 36` bits (4.5 B per output-slot). `A~` is per (model, A seed): the provider holds it
(169 GB at 72B in coefficient form: the object the customer stores today moves to the party with the hardware),
or re-derives it, or a global nothing-up-my-sleeve `A` lets all customers share it (opt-in; per-customer seeds stay
the default).

**Compact hint.** `c1` depends only on public data (`W8`, `A`, `k`), so it is fixed per linear: 4096 coefficients
per 16 rows (8192 per 32 rows at n = 8192) = **256 per output row** (1.5 KB at 48 bits, 2 KB in NTT form). The customer builds it once with
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
  row, Q = 2^108 against an HE-standard limit of 218, core-SVP 251 classical / 228 quantum): ~8% more customer work per coefficient, 2x reply-key bytes
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

## 7. Production version (2026-11-23): parameters, noise, hint integrity, integration

Status: implemented opt-in, default off, **pending the external review** (`research/reviews/pks-v3-review-packet.md`).
Customer: `PGW_FACTORY_HE_SCHEME=pks-v3` and `PGW_FACTORY_HE_PKS_ACK=pks-v3-unreviewed-n8192` (`pks.PKS_V3_ACK`;
refused without it, and for non-K32 rings). Provider: `PGW_SERVICE_MASKGEN_V3=true` (default false; v1/v2 served as
before). Changes against sections 3-6: reply ring n = 8192 (32 rows per ciphertext), all reply primes `= 1 mod 16384`,
per-linear upload seeds, `k` sampled from 128-bit words, hint stored in NTT form (same 6 bytes per coefficient).

### 7.1 Parameters

| object | ring | modulus | secret | noise | samples per secret |
|---|---|---|---|---|---|
| upload `B_C = a_C t + e + 2^24 M` | `Z[Y]/(Y^4096+1)` | `2^56` | uniform ternary, fresh per (linear, group) | CBD(21), sd 3.24 | `n/16` ring samples (<= 1848 at n = 29568) |
| reply key `beta_l = k_l t' + e_l + P iota(t_l)` | `Z[Z]/(Z^8192+1)` | `q1 P`, `log2 = 107.98` | uniform ternary, fresh per (linear, group) | CBD(21) | 16 ring samples |

Primes: `q1 = 16760833 * 16580609` (2^47.98), `P = 1073692673 * 1073643521` (2^60.0), all `= 1 mod 16384`.

Lattice estimates (`pgw.hemask.mlwe_ring.core_svp_beta`: primal uSVP, 2016 estimate, Bai-Galbraith embedding;
`0.292 beta` classical / `0.265 beta` quantum core-SVP):

| instance | HE-standard 128-bit max log q (ternary) | ours | beta | core-SVP classical / quantum |
|---|---|---|---|---|
| upload, n = 4096, q = 2^56 | 109 (classical), 101 (quantum) | 56 | 818 | 239 / 217 |
| reply key, n = 8192, q = 2^108 | 218 (classical), 202 (quantum) | 108 | 860 | 251 / 228 |
| (prototype reply key, n = 4096, q = 2^108: rejected) | 109 / 101 | 108 | 332 | 97 / 88 |

Both instances are well inside the HomomorphicEncryption.org table (standard parameters, no custom set).
**External estimator run still needed** (offline here): `LWE.estimate(LWE.Parameters(n=8192, q=2**108,
Xs=ND.UniformMod(3), Xe=ND.CenteredBinomial(21)))` and `n=4096, q=2**56` with the same distributions, including
hybrid/dual attacks; the sample counts above are far below what the estimators assume available.

### 7.2 Noise (`PKSParams.noise_terms`; units of `1 mod q1`; budget `q1 / 2^33 = 2^14.98`)

Worst case of SPEC 7.9 (all-127 rows at the widest 72B input, `||w|| = 127 sqrt(29568) = 21838`), tail 24 sigma for
Gaussian-like terms and Hoeffding at t = 24 (`2 exp(-288) = 2^-414`) for sums of independent roundings:

| term | bound | log2 |
|---|---|---|
| `W8 e` (upload noise, scaled by `q1/2^56`) | `24 * 3.24 * ||w|| * 2^-8.02` | 12.68 |
| mod switch of `H s` (4096 roundings x ternary) | `24 * 0.5 * 64` | 9.58 |
| ModDown of `c1` times `t'` (8192 roundings x ternary) | `24 * 0.5 * 90.5` | 10.08 |
| key switch `sum_l A~_l e_l / P` (16 x 8192 terms, `|A~| <= q1/2`, `|e| <= 21`) | | 4.46 |
| reply rounding to 36 bits + re-rounding | `q1 / 2^37 + 0.5` | 10.98 |
| mod switch of `y`, ModDown of `c0` | 0.5 + 0.5 | 0 |
| **total** | | **13.35** (margin 1.63 bits) |

The factory refuses to start (`PKSParams.check_noise`, margin 1 bit) for any linear whose stored row norm fails;
Qwen2.5-0.5B's worst linear (q/k/v/o) is 2^12.08 (margin 2.9 bits). A decryption failure gives a wrong row, which the
every-row self-check rejects (the segment fails; no wrong mask is stored). Tested exact: all-127 / -127 rows at
n = 3584, 18944, 8192, 29568 (`PGW_PKS_REAL_SHAPES=1`).

### 7.3 Failure modes

* decryption failure (bound exceeded): wrong `W8 r` rows -> factory self-check -> `IntegrityError`, segment dropped;
* provider deviation (wrong `y` or `c0`): same; the provider learns at most "this segment failed";
* malformed wire: reply-key residues `>= p`, sizes and versions are refused (`ProtocolError` / `version_mismatch`);
* service without v3: the customer gets a `ConfigError` (no silent fallback: its ciphertexts are pks-v3).

### 7.4 Hint integrity

The hint `c1` must be fixed before any reply secret `t'` exists, or a malicious hint could make failures depend on
`t'` (selective failure). Procedure (implemented, `PKSHintStore.build`): the **customer** draws `a_seed` and
`ks_seed` itself, computes `H = W8 A mod 2^56` from its own pack (rows streamed in blocks of 1024, never stored),
`hs = round(H q1 / 2^56) mod q1`, and `c1 = ModDown(sum_l A~_l k_l)` with the same kernel the provider uses; no
provider input is involved. Each hint file's SHA-256 is recorded at build and verified on every open (a modified
store is refused). A provider-built hint is not accepted (an algebraic check of one stays open, section 5). The
provider derives its own `hs` from the customer's public `a_seed` with its own GEMM and caches it.

### 7.5 Kernels and measurements (shared 4-vCPU Xeon @ 2.3 GHz, AVX-512, load 6-12; process CPU, single thread)

`rust/pgw-pks` (extension `pgw_pks`, `rust/tools/build_pks.py`): AVX-512 / AVX2 / baseline copies of the same code
(`multiversion!`, run-time dispatch, `PGW_PKS_ISA`), multi-threaded over blocks with disjoint outputs (identical
results for every ISA and thread count, tested); secrets and secret-dependent scratch are wiped; reductions are
branch-free (`min`), the centred CRT uses a mask, ternary sampling rejects bytes >= 252 (the rejection pattern is
independent of the kept values), CBD by popcount. Every kernel equals the numpy reference bit for bit (tested).

| kernel | measured |
|---|---|
| decrypt, per output-slot (36-bit reply read in place) | 9.5-10.7 ns (NTT part 4.8-6 ns, combine ~4.5 ns) |
| encrypt (pad + CBD + message, `a` regenerated once per call of 4 groups), per input-slot | 33-38 ns |
| reply key, per (linear, group) | 5.4-5.8 ms |

The 4-5 ns decrypt target was **not reached** by these (auto-vectorised) kernels: the remaining cost was the 32x16
transpose in the combine and L2-bound inverse-NTT passes (5 of 13 stages). **Superseded by the tuned kernels of 7.7.**

### 7.6 Not done

External estimator run and the review; GPU timing of the provider path (the torch path is exact on CPU tensors and is
self-tested before use on CUDA; no GPU run was made); one compute service per factory (no sharding); the reply key is
sent per (linear, group): sharing it across the linears of a group is analysed in 7.8 and **not** implemented (not
covered by the present security argument).

### 7.7 Tuned customer kernels (2026-11-23, second pass)

Same scheme, parameters, streams and outputs (n = 4096 upload, n = 8192 reply ring, q = 2^56, q1, P, CBD(21),
ChaCha20 streams); only the customer kernels of `rust/pgw-pks` changed. ISA levels (run-time dispatch, `PGW_PKS_ISA`
caps them; `auto` picks the best the CPU has): `avx512ifma` (new, Ice Lake / Sapphire Rapids / Zen 4 and later),
`avx512`, `avx2`, `scalar`.

* **NTTs** (`vntt.rs`, written once over a 16-lane u32 vector type, instantiated for one `zmm` and for two `ymm`):
  pointwise product fused into the first pass, stages `len < 256` per 256-coefficient sub-block in L1, the remaining
  4-5 stages column by column through an L1 buffer (one pass over the array instead of one per stage; the columns are
  16 KiB apart, i.e. the same L1 set, hence the buffer), broadcast-twiddle Montgomery products (no shift of the
  twiddle), Harvey-style lazy butterflies (`[0, 2p)`, one `vpminud` per output). Measured 1.8-2.0 ns per
  (coefficient, lane) for an 8192-point 16-lane inverse NTT (13 stages) against 2.2-2.8 before; this is the ALU bound
  of ~17 vector ops per 16-lane butterfly on two 512-bit ports (AVX2 reaches the same per-lane rate on three ports).
* **IFMA decryption** (`ifma.rs`): both reply primes are `1 mod 16384`, so `psi = CRT(psi_0, psi_1)` is a primitive
  16384-th root of unity modulo the composite `q1 = p0 p1 < 2^48` and an NTT modulo `q1` with the CRT of the
  per-prime twiddles reduces, index by index, to the two per-prime NTTs. One 52-bit Shoup NTT (`vpmadd52{lo,hi}uq`,
  8 lanes, ~10 ops per 8-lane butterfly) replaces two 24-bit NTTs, and the CRT of the combine disappears; `c1 t' mod q1`
  is the same canonical integer, so the output is bit-identical. The hint is converted from its stored per-prime
  residues on load (two Shoup products per value, amortised over the groups of a call); the reply secret's NTT is
  combined into `q1` per call by a branch-free Garner step, and its Shoup companions `floor(T 2^52 / q1)` by an f64
  estimate with four masked corrections (no data-dependent division).
* **IFMA upload pad** (`ifma_pad.rs`): the exact product `a_C t` (range `2^69`) with two ~2^50 NTT primes on 52-bit
  lanes instead of three ~2^30 primes; the pad is an exact integer, so `b` is bit-identical whatever primes are used.
  The NTT of the ternary `t` is a scalar Shoup NTT (branch-free), its Shoup companions as above.
* **Combine / final rounding** (`simd.rs`, `simd2.rs`, `ifma.rs`): CRT (or the `q1` reduction) on 16 lanes, in-register
  16x16 / 8x8 transposes (constant index tables), the 36-bit reply read in place (u32 words + nibble stream), and the
  same IEEE-754 operations in the same order as the portable code (`v * 2^-4 + w * 2^32/q1`, `+ 0.5`, truncation; no
  FMA contraction; on AVX2 the u64 -> f64 conversions are exact or, above 2^53, one correctly rounded addition as in
  `as f64`). Upload output: CRT digits computed as canonical residues (any correct formula gives the same digits), u64
  assembly, 8x8 transposes, masked gathers of `r`, 64-byte row stores; the portable path writes each `b` row
  sequentially.
* **Noise sampler**: CBD(21) is unchanged, bit for bit (same `pks-e` stream: one 64-bit word per value, popcount of
  21 + 21 bits; it equals the reference `lwe._cbd`). What changed is the CSPRNG implementation: a 16-way AVX-512 and an
  8-way AVX2 ChaCha20 (RFC 8439 block function, same keystream as the `chacha20` crate, tested), 1 KiB at a time
  (the old code allocated and page-faulted 512 KiB per call). A 42-bit packing (5.25 instead of 8 keystream bytes per
  value) was not adopted: it would change the reference stream for ~0.3-0.6 ns per input-slot.
* **Hint load / reply key**: AVX-512 masked 48-byte loads + byte shuffle + transpose for the 24-bit hint; reply-key
  output via transpose (no strided reads).
* **Exactness and constant time**: every copy equals the portable code bit for bit (Rust tests: all ISAs x 1/2/3
  threads, partial chunks and groups, Words at 54 bits to exercise `as f64` rounding, Packed36), and the portable code
  equals the numpy reference (Python tests, every ISA). No branch or memory index depends on secret data (Montgomery /
  Shoup products, `vpminu*` reductions, masked selects; gathers are indexed by public slot/feature positions only);
  secret buffers (`t`/`t'` NTTs, pads, NTT outputs, CRT temporaries, keystream scratch) are wiped.

Measured (same shared 4-vCPU host, load 10-17 during the runs, single thread, best of 21 runs; `pks-bench`, m = 4096
outputs / n = 3584 inputs, 4 groups, 36-bit replies read in place):

| kernel (ns per output- / input-slot, ms per key) | before avx512 / avx2 / scalar | after avx512ifma | after avx512 | after avx2 | after scalar |
|---|---|---|---|---|---|
| **decrypt** total | 10.6 / 12.8 / 20.4 | **4.2** | 6.0 | 6.0 | 19.8 |
| - inverse NTTs + hint load (2 primes; IFMA: 1 composite) | 6.5 / 7.9 / - | 2.9 | 4.0 | 4.3 | - |
| - combine + final rounding | 4.2 / 4.9 / - | 1.3 | 2.0 | 1.8 | - |
| decrypt at m = 8192 (72B `o`) | 9.7 | 4.1 | 6.3 | | |
| **encrypt** incl. noise and message | 34.5 / 35.8 / 39.9 | **12.5** | 15.6 | 18.5 | 42.2 |
| encrypt at n = 29568 (72B `down`) | 33.8 | 12.6 | 16.7 | | |
| - CBD(21) noise per value (ChaCha20 + popcount) | 11.4 (7.5 table) | 1.6 | 1.6 | 3.1 | 5.4 |
| **reply key** per (linear, group) | 5.3 / 5.1 / 5.6 ms | 2.1 ms | 2.0 ms | 2.2 ms | 6.0 ms |
| 8192-pt 16-lane inverse NTT incl. pointwise, per element | 2.4 (portable) | | 1.75 | 1.8 | |
| same, forward | 6.9 (portable) | | 1.95 | 1.9 | |

("before" re-measured in the same runs with the previous binary; the scalar copy is the x86-64 baseline build, not a
target. The AVX2 copy reaches the AVX-512 per-lane rate here because 256-bit ops issue on three ports.)

Sizing with these kernels (`benchmarks/hemask_pks_model.py`, cores = customer CPU s/s at this host's single-thread
speed; customer storage = compact hint; `r` keystream unchanged at 5.8 ns per input-slot):

| model | kernels | customer storage | up / down MB per slot | CPU ms/slot | cores @25/s | cores @50/s | up+down MB/s @25 / @50 |
|---|---|---|---|---|---|---|---|
| 7B | before | 2.1 GB | 6.6 / 6.3 | 50.3 | 1.3 | 2.5 | 321 / 643 |
| 7B | **after, avx512ifma** | 2.1 GB | 6.6 / 6.3 | 22.0 | **0.5** | **1.1** | 321 / 643 |
| 7B | after, avx512 / avx2 | 2.1 GB | 6.6 / 6.3 | 27.0 / 29.5 | 0.7 / 0.7 | 1.4 / 1.5 | 321 / 643 |
| 32B | before | 7.1 GB | 21.0 / 20.9 | 164.4 | 4.1 | 8.2 | 1049 / 2099 |
| 32B | **after, avx512ifma** | 7.1 GB | 21.0 / 20.9 | 72.0 | **1.8** | **3.6** | 1049 / 2099 |
| 32B | after, avx512 / avx2 | 7.1 GB | 21.0 / 20.9 | 88.8 / 97.0 | 2.2 / 2.4 | 4.4 / 4.9 | 1049 / 2099 |
| 72B | before | 10.5 GB | 32.5 / 30.9 | 252.3 | 6.3 | 12.6 | 1585 / 3170 |
| 72B | **after, avx512ifma** | 10.5 GB | 32.5 / 30.9 | 110.7 | **2.8** | **5.5** | 1585 / 3170 |
| 72B | after, avx512 / avx2 | 10.5 GB | 32.5 / 30.9 | 136.4 / 149.2 | 3.4 / 3.7 | 6.8 / 7.5 | 1585 / 3170 |

Bytes are unchanged (same scheme); the reply keys are 0.77 / 1.77 / 2.21 MB of the upload per slot (7.8). Of the
72B "after" CPU, 25 ms per slot is the unchanged `r` keystream; the pks-v3 kernels themselves are 86 ms per slot
(decrypt 29, encrypt 54, reply keys 2.6).

The largest remaining customer costs: the upload pad NTTs (2-3 per input-slot and group), the `r` keystream of the
existing `pgw_fast` expander (5.8 ns per input-slot, outside these kernels; the 16-way ChaCha20 here runs at ~0.15 ns
per byte and could replace it), and memory traffic of `r` (int64 in) and `b` (8 B out) per input-slot.

### 7.8 One reply key per group (shared across the linears of a group)? Analysis: not covered, not implemented

Proposal: one upload secret `t_g` and one reply secret `t'_g` per 256-slot group for **all** linears of the pass (the
per-linear `a` seeds already make the upload instances distinct), one reply key `beta_g` sent once and cached by the
provider (bounded cache keyed by customer and group). Saving: the reply keys are 1.77 MB per (linear, group), i.e.
0.77 / 1.77 / 2.21 MB of the upload per slot at 7B / 32B / 72B (12% / 8% / 7% of the upload), and 2.0-2.2 ms of
customer CPU per (linear, group) (2-4% of the tuned customer CPU). Sharing `t'` alone is impossible: `beta` encrypts the
upload secret's components, so one reply key per group means one upload secret per group across linears.

* **Passive provider (the hybrid of the review packet, section 3).** View per group: all linears' bodies
  `b_L = a_{L,C} t_g + e + 2^24 M` and one `beta_g`. H1 (`beta_g` -> uniform) is unchanged: `t'_g` still appears
  nowhere else in the provider's view (decryption is local, the replies are functions of the provider's own inputs).
  H2 (`b` -> uniform) then needs decision-RLWE(`R_u`, 2^56, ternary, CBD(21)) with `sum_L n_L / 16` samples under one
  secret instead of `n / 16 <= 1848`: 52k (7B), 172k (32B), 271k (72B) ring samples. Still a polynomial number of
  samples of a standard instance (the primal/dual estimates use at most a few `n` samples; BKW / Arora-Ge need
  vastly more), but it is a different statement of the assumption and would need the reviewer's sign-off.
* **Active provider (packet section 6) — not covered.** Today every `(t, t')` pair is used for exactly ONE decryption:
  a deviating provider (wrong `c0` or `y`) learns at most one failure bit per secret pair, about a secret that is
  never used again. With sharing, `t'_g` (and `t_g`, through the `H s` rounding term of the noise) decrypts the
  replies of all 112-320 linears of the group, sequentially and pipelined (`he_depth` requests in flight): the
  provider gets an adaptive decryption-failure / key-mismatch oracle against a **live** secret that still protects
  the masks of the other linears of the group. Reaction attacks on RLWE need O(n)-ish queries per key, so this is
  not a key recovery, but no reduction covers it. Even with abort-on-first-failure the transcript leaks the index of
  the first failure (`log2(#linears + 1)` = 6.8-8.3 bits) about a key whose earlier masks are kept; only discarding
  ALL masks of the group on any failure (group-atomic commit of a whole pass, buffering up to 27 MB x 256 slots =
  7 GB of masks per group at 72B until its last linear decrypts) would bring it back to "leakage about dead keys
  only". The generic guessing argument costs up to those 6.8-8.3 bits of security otherwise.
* **New reuse hazard.** With `t_g` shared, any second upload of the same linear in the same group (a retry after a
  transport error, a re-requested segment) under the same `a_{L,C}` gives `b - b' = e - e' + 2^24 (M - M')`, i.e.
  the mask difference in the clear. Today's fresh `t` per call is immune; sharing would need a strict
  one-upload-per-(linear, group) invariant on every error path.

Conclusion: covered for a passive provider only with a restated (larger) sample count; **not covered** by the
present argument for an active provider. Per the rule "only if the hybrid security argument stays intact", it is not
implemented; the reply key stays per (linear, group) with a fresh `t, t'`. If wanted later, it needs (a) the
reviewer's sign-off on the H2 sample count, (b) group-atomic mask commit with discard-on-failure, (c) a
no-re-encryption invariant per (linear, group), (d) a provider key cache keyed by (customer, group) whose eviction
only ever causes a re-send of the same `beta_g`.
