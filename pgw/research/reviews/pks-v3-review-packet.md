# pks-v3 review packet (for an independent cryptographer)

Scope: the compact-hint encrypted mask scheme `pks-v3` of the PGW private-inference system. It is implemented,
opt-in and off by default; the customer must set an acknowledgement string until this review signs off.
Detailed notes: `research/plans/he-compact-hint-pks.md` (sections 3, 5 and 7). Reference code:
`src/pgw/hemask/pks.py` (numpy, exact); kernels `rust/pgw-pks`; integration `src/pgw/hemask/pks_backend.py`,
`src/pgw/service/pks_provider.py`; tests `tests/hemask/test_pks.py`.

## 1. Setting and goal

A customer holds int8 weights `W8` (`m x n` per linear layer, public to the provider too) and needs masks
`W8 r mod 2^32` for secret uniform `r in Z_{2^32}^n`, for many slots. The provider has the GPU. Privacy goal:
the provider learns nothing about `r` (computational, lattice assumptions only: no TEE, no non-collusion).
Integrity is not a goal of the scheme: every decrypted row is checked by the customer (private Freivalds
vectors, every row) and online checks are unchanged; a deviating provider can only cause detected failures.

## 2. Scheme

Rings `R_256 = Z[X]/(X^256+1)`, `R_u = Z[Y]/(Y^4096+1)` (`Y^16 = X`), `R_r = Z[Z]/(Z^8192+1)` (`Z^32 = X`).
Per (linear, group of 256 slots):

1. **Upload.** Fresh uniform ternary `t in R_u`, noise CBD(21). For each block `C` of 16 features:
   `B_C = a_C t + e_C + 2^24 M_C mod 2^56`, `M_C[16u + k] = r[u, 16C + k]`; `a_C` public, derived from a
   customer-chosen seed and the linear id. Sent as `b[16C + k, u] = B_C[16u + k]`: each feature is an MLWE
   sample (rank 16, degree 256) under the components `t_l` of `t` (`t_l[u] = t[16u + l]`).
2. **Reply key.** Fresh uniform ternary `t' in R_r`, noise CBD(21):
   `beta_l = k_l t' + e_l + P iota(t_l) mod q1 P` for `l < 16`, `iota: X -> Z^32`, `k_l` public (seeded,
   uniform in the NTT domain), `q1 = p0 p1 ~ 2^48`, `P = p2 p3 ~ 2^60`.
3. **Provider.** `y = W8 b mod 2^56`; switch `y` and the public `H = W8 A` to `q1`; pack 32 rows into
   `R_r` (row `j`, slot `u` -> coefficient `32u + j`); hybrid key switch
   `C0 = P body - sum_l A~_l beta_l`, ModDown to `c0 mod q1`; return `c0` rounded to 36 bits.
4. **Compact hint** (customer, once per linear): `c1 = ModDown(sum_l A~_l k_l) mod q1` (public data only).
5. **Decrypt.** `v = c0 + c1 t' mod q1`, mask `= round(2^32 v / q1) mod 2^32`.

## 3. Assumptions and hybrid argument

Provider view per (linear, group): `b` (`n/16` RLWE samples over `R_u` mod `2^56` under `t`) and `beta`
(16 RLWE samples over `R_r` mod `q1 P` under `t'`, encrypting `P iota(t_l)`), plus public data (`W8`, seeds).

* H0: real view. H1: replace `beta` by uniform. Indistinguishable under decision-RLWE(`R_r`, `q1 P`,
  ternary secret, CBD(21)) with 16 samples; `t'` is used nowhere else in the provider's view. H2: replace `b`
  by uniform mod `2^56`. Indistinguishable under decision-RLWE(`R_u`, `2^56`, ternary, CBD(21)) with
  `n/16 <= 1848` samples; `t` appears in H1 only through `b` (the reply key no longer depends on it). In H2 the
  view is independent of `r`. No circular security: `beta` encrypts `t`'s components under an independent key.
* Fresh `t`, `t'` and noise per (linear, group) (one fresh 32-byte seed per group, independent HKDF labels).
* RLWE modulo a power of two (`2^56`) for the upload: as in the existing v1/v2 scheme (rank-16 MLWE at the
  same modulus); the change is from MLWE over `R_256` to RLWE over `R_u` of the same dimension. **Needs sign-off.**
* Public `a_C` (per linear) and `k_l` (per hint store) reused across groups: common-reference-string use with
  fresh secrets and noise per instance.
* Customer-side rounding/switching leaks nothing (all decryption is local).

## 4. Parameters (standard sets only)

| instance | n | log2 q | HE-standard 128-bit limit (classical / quantum) | core-SVP (repo model) classical / quantum |
|---|---|---|---|---|
| upload | 4096 | 56 | 109 / 101 | 239 / 217 (beta 818) |
| reply key | 8192 | 107.98 | 218 / 202 | 251 / 228 (beta 860) |

Request: run the lattice estimator for both (ternary secret, CBD(21) noise, sample counts above), including
hybrid/dual attacks and any concern specific to power-of-two moduli or the subring structure (`t_l` are the
residue classes of one ring element).

## 5. Correctness and noise

Budget `q1 / 2^33 = 2^14.98`; worst-case bound 2^13.35 for all-127 rows at n = 29568 (table in the plan,
section 7.2: tail 24 sigma for Gaussian-like terms, Hoeffding at t = 24 for rounding sums). Checked per linear at
start-up with a 1-bit margin (refuses to start otherwise). A failure yields a wrong row that the every-row check
rejects.

## 6. Failure modes and selective failure

Decryption failures depend on `t'` only through the noise terms `ModDown(c1) t'` and `H s` rounding; a wrong
hint could turn failures into a `t'`-dependent oracle. Mitigation: the customer builds the hint itself from its
own weights and seeds before any `t'` exists, never accepts a provider-built hint, and verifies the stored hint's
SHA-256 on open. A deviating provider (wrong `c0`) causes failures whose occurrence depends on its own deviation
and the fresh `t'` of that group; secrets are never reused after a failure (fresh per group, segment dropped).
**Question for the reviewer:** is a per-segment failure bit (observable as a retried/missing segment) acceptable
given fresh secrets per group, or should failures be rate-limited / the service blocked after a failure?

## 7. What is tested (`tests/hemask/test_pks.py`, 25 tests)

Primes and security envelope; exact products and rounding helpers; noise bound at the worst case and refusal;
upload decrypts with the old MLWE hint path (bodies minus pad are exactly the CBD noise); full round trips
(numpy reference and native, partial groups); native equal to the reference bit for bit (encryption, reply keys,
hint, provider reply); identical results across AVX-512 / AVX2 / scalar and thread counts; tampered reply changes
the masks; torch provider path equal to the CPU path; real widths n = 3584, 18944, 8192, 29568 with all-127 rows
(`PGW_PKS_REAL_SHAPES=1`); wire v3 encode/decode, refusal by v1/v2-only servers, reply-key range checks;
`PGW_SERVICE_MASKGEN_V3` default off; factory refuses pks-v3 without the acknowledgement and for non-K32 rings;
backend exactness and hint-store tamper refusal. End to end: Qwen2.5-0.5B (attention linears) through the real
factory with the in-process provider, segments bit-identical to local generation, every row self-checked.

## 8. Not covered / open

External estimator run; side channels beyond branch-free arithmetic (ternary rejection sampling leaks only the
count of rejected bytes); sharing one reply key across the linears of a group (analysed in the plan, section 7.8: covered for a passive
provider only with a larger H2 sample count, not covered for an active one, not implemented). The customer kernels
were retuned (plan 7.7: AVX-512 / IFMA / AVX2 copies) with outputs bit-identical to the reference; parameters,
streams and the CBD(21) sampler are unchanged.

## 9. Provider-built, client-verified hint (added 2026-11-25)

The SDK lets the provider compute the expensive part of the hint, `H = W8 A mod 2^56` (90+ % of the build), for the
customer's freshly drawn `a_seed` (`ks_seed` never leaves the customer). The customer accepts `H` only after a 2-adic
Freivalds check against the fingerprint-pinned `W8` (42 binary challenge columns per side, soundness error
`<= 2^-41` against any error the provider picks), then computes `switch_hint` and the key switch itself; the stored
hint is bit-identical to a local build with the same seeds. The key-switched hint itself is not checkable cheaply
(two exact roundings between `W8` and it). Argument, attack analysis and costs: `pks-v3-provider-hint.md`.

**Q7 for the reviewer:** is the provider's view or the order of events in this mode different from a local build in
a way that matters (the provider learning `a_seed` and computing `A`, `H` before the first maskgen request; adaptive
choice of `H` across linears; the one-bit pass/fail outcome of a provisioning attempt)? Our argument (section 4 of
that note): all of it is public data, no secret exists yet when the check runs, and challenges are fresh per linear
and attempt.
