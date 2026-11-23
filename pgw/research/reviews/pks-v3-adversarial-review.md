# pks-v3: adversarial review

Date: 2026-11-23. Reviewer: an independent internal adversarial review. This is not the external
cryptographic review the packet asks for, and it does not replace it.
Scope: `research/reviews/pks-v3-review-packet.md`, `research/plans/he-compact-hint-pks.md` (sections 1-7.8),
SPEC §16, `src/pgw/hemask/{pks,pks_native,pks_backend,wire}.py`, `src/pgw/service/{pks_provider,pks_gpu}.py`,
`rust/pgw-pks/src/*.rs`, `tests/hemask/test_pks.py`, and the integration points: `factory/generator.py`,
`factory/fused2_gen.py`, `factory/fused2_he.py`, `factory/scheduler.py`, `core/freivalds.py`, and the v1/v2 backend
(`hemask/backend.py`) for comparison. Method: I re-derived the hybrid argument, the noise bound and the core-SVP
numbers, read every customer-side path that touches secrets or provider data, and tried to break the scheme
with an active provider on the real factory code (tests added). No network, no Modal, no Kaggle.

## Executive verdict

**Issues must be fixed first.**

The cryptographic core holds up: I found no critical or high issue in the scheme itself. Specifically, I
checked the hybrid argument, the parameters, the noise bound, the hint-integrity rule, randomness and domain
separation, wire parsing and version negotiation.

The scheme's integrity premise does not hold. The packet says "a deviating provider can only cause detected
failures". That is false in the integration as reviewed:

* **F1 (high).** Any provider can shift any decrypted mask entry by an exact amount of its choosing. The factory's
  private every-row self-check uses only two uniform vectors mod 2^32, so the wrong masks get through:
  * Measured: 19 of 80 tampered segments were accepted and would have been committed.
  * Worst-case soundness against a provider that picks its errors is 2 bits.
  * A tampered mask whose error falls inside the gateway's range gives silent output corruption, or a
    data-dependent circuit-breaker trip, which leaks one bit of a private activation.
* **F2 (high).** In the encrypted fused2 digit split, only 8 rows per chunk are self-checked.

Both are fixed in this branch, in opt-in code only (pks-v3 and fused2+HE), with tests (commit `555b062`). F1 is a
property of the shared generator, so the default mlwe v1/v2 path still has it (recommendation R1).

Before pks-v3 becomes the CPU-only default:

1. Merge these fixes and re-measure their cost (+5.4 ns of customer CPU per (slot, m+n), see F1), or adopt R2,
   which is cheaper.
2. Decide the factory's reaction to integrity failures (F3).
3. Run the external lattice estimator (F5).

After that, the remaining findings are medium or lower, and the scheme would be acceptable to enable pending
the external review.

## Findings

| # | sev | finding | status |
|---|---|---|---|
| F1 | **high** | Factory self-check of provider products: 2 ring vectors give 2 bits (worst case) against a provider that picks its errors; replies are exactly additively malleable | **fixed** (40-bit target for every provider product: mlwe v1/v2, bulk, fleet, pks-v3, fused2 digits; follow-up) |
| F2 | **high** | Encrypted fused2 digit split self-checks 8 sampled rows per chunk | **fixed** (every row, 40-bit target) |
| F3 | medium | Integrity failures from the provider are retried forever (backoff ≤ 10 min, reset by any success) | **fixed** (alert, persistent count, halt after 3, `pgw factory clear-halt`) |
| F4 | medium | Default gateway ring-Freivalds check verifies only the masked product: wrong masks are invisible online; the range check's reaction depends on data | **fixed** (`check_field = "auto"`: p31 for encrypted factories; gateway `prime_check` auto; 40-bit factory check kept) |
| F5 | medium | Parameter claims rest on a core-SVP model; no lattice-estimator run | open (my numbers below agree with the packet) |
| F6 | low | Hint-store SHA-256 digests live in unauthenticated `meta.json` with the seeds and the row norms that drive the noise refusal | **fixed** (HMAC under a customer key, `he_hint_key_file`; trust boundary documented without it) |
| F7 | low | Two factory processes on one hint root: a rebuild replaces seeds under a running process → every decryption fails (DoS) | **fixed** (shared/exclusive `flock` on the store) |
| F8 | low | Secret hygiene in Python: decrypted-mask and upload-secret copies left unwiped (fixed); per-group seeds and stream keys are immutable `bytes`; the numpy reference keeps keystreams in `bytes` | partly fixed |
| F9 | info | Native decryption computes `round(c0r 2^-4 + w 2^32/q1)` in f64 without the reference's inner rounding: bit-identical on valid replies, may differ by 1 on adversarial values | documented |
| F10 | info | Doc inconsistencies (reply-key core-SVP 218 vs 251; 16 vs 32 rows per reply ciphertext; prime names) | **fixed** (core-SVP 251/228, 218 is the HE-standard limit; 32 rows at n = 8192) |
| F11 | info | `test_backend_in_real_factory_matches_local` always skips (spec ids lack `kind`); the "real factory" claim rested on a benchmark script | **fixed** (new test runs the real `MaskGenerator`) |
| F12 | info | Power-of-two upload modulus: the only small quotient is evaluation at X = 1 mod 2, which reveals parity(Σe), uniform for CBD(21) | no issue |
| F13-F17 | info | Hint integrity rule, randomness, downgrade, wire parsing, constant-time: checked, no issue (details below) | — |

### F1 (high): the reply is malleable and the self-check is too weak for an adversarial provider

**Mechanism.** Decryption is `v = c0 + c1 t' mod q1`, `mask = round(2^32 v / q1)` (`pks.py:593-606`, `kernel.rs:615-758`).
`c1` is the customer's fixed hint, and `c0` is entirely provider-controlled. Adding `16 δ` to one 36-bit reply value
moves exactly one mask entry by exactly `δ` (mod 2^32), with no noise growth. This is inherent to any additively
homomorphic reply, and v2 has the same property (add `2^24 δ` to `y`). The defence is the factory's self-check:

* The check is `generator.py`, `_self_check` with private vectors drawn at the old `ring_uniform((2, m), k, "v")`:
  two uniform vectors over `Z_{2^32}`.
* A vector `v` misses an error vector of 2-adic valuation `t` with probability `2^-(32-t)`.
* So `δ = 2^31` in one entry passes both vectors with probability **1/4**.

**Demonstration.** Measured with the real `MaskGenerator`, pks-v3 backend, in-process provider, and one tampered
entry in one linear per segment (script in the session scratchpad; the tests are below). Before the fix, **19/80
segments with a wrong mask were accepted**. After it, 0/80.

**Why it matters.** It breaks integrity, and it can also leak private data.

* The gateway computes `yq = centered(y_hat - mask) = Wx - δ` and range-checks it against the cap. At K32A12C27 the
  cap is `B = 2^27` at real widths.
* `δ = 2^31` always fails the range check. That is a data-independent DoS, which a provider can cause anyway.
* `δ = 2^27` instead passes the range check iff `(Wx)_i >= 0`:
  * If it passes, the output is silently wrong. The default ring Freivalds check (F4) verifies only `y_hat` against the
    masked input, so it does not notice.
  * If it fails, the circuit breaker trips (`gateway/runtime.py`: "IntegrityError trips the breaker").
  * The provider therefore learns the sign of one chosen activation coordinate, or corrupts it silently.
* Such an error has valuation 27. It passes two vectors with probability `2^-10` per segment attempt.
* With F3, attempts are unlimited, so this is a practical 1-bit-per-~1000-segments channel plus an integrity break.

**Fix (this branch, opt-in).**

* `freivalds.private_check_vectors_for(ring, n, bits)` returns `ceil(bits / h)` vectors, with
  `h = K - ceil(log2(4 B))` (the existing `soundness_bits` per vector). Errors whose entries all exceed `2B` are
  rejected by the gateway range check for every input, so only errors with valuation `<= log2(2B)` matter.
* `PKSMaskBackend.self_check_bits = 40`, and `MaskGenerator._private_vectors` honours it. That gives 14 vectors at
  capped K32 widths and 7 at the tiny test widths.
* mlwe v1/v2 is unchanged: no `self_check_bits`, so still 2 vectors.
* Tests:
  * `test_provider_can_shift_a_mask_entry_exactly`.
  * `test_factory_self_check_catches_provider_errors[0 | 2^31 | 2^35]`: an honest provider gives segments identical
    to local generation; tampered replies raise `IntegrityError`.

**Cost.** `he_check_rows`, measured at m = 18944, n = 3584, 256 rows, one core:

| private vectors | ns CPU per (slot, m + n) |
|---|---|
| 2 | 3.0 |
| 7 | 4.7 |
| 14 | 8.4 |

Going from 2 to 14 vectors adds about 12 ms of customer CPU per slot at 7B (Σ(m+n) = 2.2M) and about 60 ms at 72B
(11.2M). That is roughly +50% of the tuned pks-v3 customer CPU. R2 is the cheaper alternative.

### F2 (high, opt-in path): encrypted fused2 self-check is sampled

* `runtime.py:141-143` builds `Fused2MaskGenerator(..., self_check_rows=settings.self_check_rows or 2)`. The default
  is 8, and `fused2_gen._self_check` then checks 8 random rows of each 64-row chunk.
* Meanwhile `fused2_he.py`'s docstring says "any wrong product is caught before a segment is written".
* With an encrypted backend, the provider can corrupt one digit product row: `W r` is off by `2^(b i) δ`. That row is
  unchecked with probability about `(63/64)^8 ≈ 0.88`.

**Fix.** With `he` set, the generator forces `self_check_rows = -1` and uses `private_check_vectors_for(ring, n, 40)`
vectors. At K56A12C53 the widths up to 2^15 are uncapped, so `h = 21` and 2 vectors suffice; a capped width would
need more. Test: `test_encrypted_fused2_checks_every_row_with_enough_vectors`.

### F3 (medium): reaction to failures (answer to the packet's question in §6)

**Is a per-segment failure bit acceptable?** For confidentiality of `r`: yes, with one caveat.

* Each segment uses fresh `(t, t', e)` per (linear, group) and a fresh segment seed (`os.urandom`) for `r`. A failed
  segment is aborted and its `r` never used.
* The failure event is a function of the provider's deviation and of noise from dead secrets.
* The only `r`-dependence is a sub-unit term: `Δ = q1/2^32` is not an integer, so `v - Δ M` has a fractional part that
  depends on `M`. Against a noise spread of about 2^10 units, that biases a boundary-tuned failure by about 2^-10
  (per position) towards a pseudo-random function of `W r`.
* Mutual information is at most 1 bit per segment, and the provider only learns it about segments that succeed.
  This is negligible, but it is not literally zero, as the packet suggests.

**Re-encryption.** I checked every error path: transport retries resend identical bytes (`ComputeClient._call`), and
nothing re-encrypts with old secrets.

**The real problem is volume.** `scheduler.run_forever` treats `IntegrityError` like any transient error:
exponential backoff capped at 10 minutes, reset by any success. An active provider can interleave honest segments
and attempt tampering indefinitely. That was what made F1 practical.

Recommendation: count provider-attributable integrity failures (remote product mismatch, reply decode errors)
separately from transport errors. Alert on the first one. Stop the factory (fail closed) after a small number,
for example 3 per day, and do not reset the count on success.

### F4 (medium): the default gateway check does not see wrong masks

* With the default ring format, the gateway verifies `u·y_hat == (uW)·x_hat` on masked values (`core/freivalds.py`
  header), so a wrong mask is caught only by the range check, which is data-dependent (F1).
* The opt-in `p31` format verifies the unmasked `yq` against `xq` (`|e| <= 2B < p`), so it catches any wrong mask with
  probability `1 - 2^-31`. Its reaction is then the same for every input.

**R2.** Require `check_field = "p31"` for pks-v3; it is K32 only, which is what pks-v3 needs anyway. With it, a
factory self-check escape becomes a detected, data-independent failure (DoS only). The factory's 40-bit target
could then be relaxed to the old 2 vectors to save the F1 cost. Pass `self_check_bits` to `PKSMaskBackend`; it is not
yet exposed in settings.

### F5 (medium/open): parameters

I recomputed the primal uSVP 2016 estimate (Bai–Galbraith scaling, `σ_e = sqrt(10.5)`, ternary secret, optimal `m`)
independently of `mlwe_ring.core_svp_beta`, and a plain dual estimate (no secret scaling):

| instance | primal β | core-SVP classical / quantum | dual (plain) |
|---|---|---|---|
| upload n=4096, log q=56 | 818 (m≈3809) | 239 / 217 | ≈251 (β≈860) |
| reply key n=8192, log q=107.98 | 860 (m≈7617) | 251 / 228 | ≈257 (β≈880) |
| HE-standard edge (4096, 109) | 327 | 95 / 87 | ≈99 |
| HE-standard edge (8192, 218) | 325 | 95 / 86 | ≈96 |

The two pks-v3 instances sit at half the HE-standard log q for their degree, classical and quantum (109/101 and
218/202). They are 2.5x above the table's own 128-bit edge in the same model. The sample counts are well above what
the attacks use: n/16 ≤ 1848 ring samples on the upload side, 16 ring samples (131072 LWE samples) on the reply side.
The secret is uniform ternary, not sparse, and the noise is CBD(21), sd 3.24, at least the table's 3.19.

No hybrid/MITM concern at these dimensions. The lattice-estimator run is still needed for sign-off, but I found
nothing that suggests it will be close.

On the hybrid argument:

* **H1.** `β` is RLWE in `t'`. `t'` appears nowhere else in a passive provider's view, because decryption is local.
  The fixed `k_l` act as a CRS: the reduction embeds the challenge's own uniform `k` for every instance.
* **H2.** `b` is RLWE in `t`, and `β` no longer depends on `t` after H1.
* No circularity: `β` encrypts `t` under an independent `t'`.
* Per-(linear, group) freshness is real. Native path: one `new_seed()` per group, and the labels `pks-t`, `pks-e`,
  `pks-t-reply`, `pks-e-ksk` are distinct. The reference path uses one seed for upload and reply key, still with
  distinct labels. Nothing shares `t` across linears; plan §7.8 is not implemented.

The argument holds for a passive provider. For an active one, it holds once F1 to F4 are addressed, since every
secret then decrypts exactly one reply and the abort bit concerns dead secrets.

### Noise (re-derived)

Every term of `PKSParams.noise_terms` matches my own derivation:

* The Hoeffding sums are valid for fixed rounding errors times independent symmetric ternary `t` / `t'`, and for
  `A~ e` with CBD `e`.
* `W8 e` is sub-Gaussian with variance proxy 10.5 per coefficient.
* The key-switch identity is consistent: `C0 + C1 t' = P(body - Σ A~ ι(t_l)) - Σ A~ e` mod `q1 P`, and the centred
  lifts of `hs` differ by multiples of `q1`, which vanish after `/P`.

Worst case, all-127 rows at n = 29568: `2^12.68 + 2^9.58 + 2^10.08 + 2^4.46 + 2^10.98 + 1.5 = 2^13.351`. The budget is
`q1/2^33 = 2^14.982`, a margin of 1.63 bits.

The runtime guard `check_noise(margin_bits=1)` runs per linear in `PKSMaskBackend.__init__` before any request. It
accepts row norms up to 40931 (= 127·sqrt(103873)), so every Qwen2.5-72B row passes. The per-position failure bound
is 2^-414, plus the union over positions.

Decryption failures are not detected cryptographically; the factory self-check detects them. That makes F1 also a
correctness issue.

### F6-F8 (low)

* **F6.** `PKSHintStore.open` verifies each `.npy` against a digest stored in the same directory's `meta.json`.
  * What it protects against: corruption and partial writes.
  * What it does not protect against: anyone who can write the store. They can also lower `row_norms`, which would
    defeat the noise refusal, or swap `a_seed`/`ks_seed`.
  * That is acceptable while the hint root is inside the customer's trust boundary; say so in the docs, or use a
    keyed MAC.
  * Hashing 10.5 GB on every open at 72B is also a start-up cost.
* **F7.** `PKSHintStore.build` writes to `*.building-<pid>` and does `rmtree(final)` + rename. A second process can
  replace the store, with new seeds, while the first holds the old `meta` and mmaps hint files lazily (`hint()`). The
  failure is deterministic (`c1` mismatch → uniform), so it is not a selective-failure risk. Recommend a lock file.
* **F8.** Fixed:
  * `PKSMaskBackend.matmul` now writes masks into one caller-owned array and wipes the per-call copies, also on
    error. Before, it kept unwiped uint32 copies and the concatenation.
  * `Customer.encrypt` no longer leaves `np.stack` inputs of `t` unwiped, nor `ts` on error.

  Open, memory-disclosure only:
  * Per-group seeds, which determine `t`, `t'` and `e`, together with `b` determine `r`. They and the HKDF stream keys
    are immutable `bytes`.
  * Without `pgw_pks`, `lwe._ternary` / `_cbd` hold keystreams in `bytes`.

### F9-F17 (info)

* **F9.** See the table. The native combine (`kernel.rs` `dec_chunk`, `simd*`, `ifma`) differs from the reference by
  at most 2^-17 + 2^-19 units before rounding. Valid replies are within ±0.16 of an integer, so the outputs are
  identical there.
* **F10.**
  * Plan §2/§5 say "core-SVP 218" for the n = 8192 reply ring; §7.1 and the packet say 251, and 251 is right.
  * Prose in places says 16 rows per reply ciphertext; the code uses `PACK = 32`.
* **F11.** Fixed: `_LayerPack` in `test_pks.py` gives real `LinearId`s, and the factory test runs.
* **F12.** `X^4096+1 ≡ (X+1)^4096 mod 2` has no root mod 4, so the only map to a small ring is evaluation at 1 mod 2.
  It reveals `parity(Σ e)`, uniform for CBD(21). Same modulus as v1/v2.
* **F13 (hint integrity).**
  * `c1` is computed only from the customer's pack and its own `a_seed`/`ks_seed` (`os.urandom`), before any `t'`.
  * `ks_seed` never leaves the customer, and the provider never needs `k`.
  * The provider cannot choose `a_seed`, the wire `aseed` is the customer's, and a provider-built hint is never read.
  * A wrong provider `hs` only causes uniform failures.
* **F14 (randomness).**
  * All seeds come from `os.urandom`.
  * HKDF-SHA256 labels are injective (typed encoding).
  * ChaCha20 runs with a zero nonce under distinct keys, so there is no nonce reuse.
  * `ks_a` reduces 128-bit words: SD ≤ 2^-79 total.
  * Ternary sampling rejects bytes ≥ 252 (exactly uniform). CBD(21) is popcount of 21+21 bits.
  * The test hooks `seeds=` / `_seed=` have no production callers.
* **F15 (downgrade).** A pks-v3 customer only sends `v: 3`. A refusal is a `ConfigError` with no fallback, and the
  response version must equal the request's. A provider cannot force v1/v2, and those carry different ciphertexts
  anyway.
* **F16 (constant time, portable code).** Constant time apart from what is noted:
  * Montgomery/Shoup products and `min`-based reductions.
  * `res_small` by mask; ternary rejection reveals only the rejection pattern.
  * Gathers use public indices.
  * f64 rounding of values below 2^34; a saturating `as u64` never takes its large-value path.
  * Not verified at the assembly level. The SIMD copies were only spot-checked.
* **F17 (wire).** Customer side:
  * Reply length must equal `m·cols·36/8` exactly (`decode_maskgen_response`).
  * Rust `decrypt_item` re-checks every length.
  * Reply-key residues are checked `< p` on the provider.

  Provider side (provider resource, not customer security): a client can force `H` rebuilds per `a_seed`.

## Recommended fixes (in order)

1. **(F1, done for pks-v3)** Keep `self_check_bits = 40` for pks-v3, or adopt R2 and relax it. **R1:** give mlwe
   v1/v2 the same treatment, either by setting `self_check_bits` on `HEMaskBackend` or by R2. It changes a default
   path, so it was not done here.
2. **(F4/R2)** Require or default `check_field = "p31"` with pks-v3. Wrong masks are then caught online, and
   the reaction is data-independent.
3. **(F3)** Fail closed on repeated provider integrity failures, with a non-resetting counter and an alert. Keep
   "segment dropped, fresh seeds" as is.
4. **(F5)** Run the lattice estimator for the two instances. Expect a large margin (table above).
5. **(F6/F7)** Make the hint store's trust assumption explicit, or MAC `meta.json`. Add a build lock.
6. **(F8/F10)** Wipe or avoid immutable secret `bytes` where practical, and fix the doc inconsistencies.

## What was changed in this branch

Commit `555b062`, all in opt-in paths. mlwe v1/v2 and the plain factories behave as before.

* `core/freivalds.py`: `HE_SELF_CHECK_BITS`, `private_check_vectors_for`.
* `factory/generator.py`: `_private_vectors` honours `remote.self_check_bits`.
* `factory/fused2_gen.py`: with `he`, every row is checked with enough vectors.
* `hemask/pks_backend.py`: `self_check_bits = 40`, and decrypted masks are wiped.
* `hemask/pks_native.py`: upload-secret copies are wiped.
* Tests:
  * `test_pks.py`: tampering provider, exact shift, the real factory honest and tampered.
  * `test_fused2_factory.py`: every row with HE.

Results: `tests/hemask/test_pks.py` and `tests/factory/test_fused2_factory.py` give 39 passed and 5 skipped. The
skips are the opt-in real shapes and the old always-skipping test. `tests/factory` and `tests/core` give 939 passed.

Not changed: the scheme, its parameters, its streams and its outputs.
