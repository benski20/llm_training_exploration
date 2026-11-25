# pks-v3: a provider-built, client-verified hint

Date: 2026-11-25. Scope: `src/pgw/provision/` (client), `src/pgw/service/provision.py` (provider),
`src/pgw/provision/hintcheck.py` (the check), `PKSHintStore.write` (`src/pgw/hemask/pks_backend.py`), tests
`tests/provision/`. Status: implemented; the SDK default (`hint="provider"`), see section 6. This note is an input
to the pending external review of pks-v3 (question Q7 in `pks-v3-review-packet.md`).

## 1. Problem

The pks-v3 compact hint (2.1 GB at Qwen2.5-7B) is built by the customer from its own `W8` and two seeds it draws
(`a_seed`, `ks_seed`), before any reply secret exists (review packet section 6, adversarial review F13). On a small
application server that build takes far too long: about 25-45 min of a 4-vCPU box at 7B (90 % of it the GEMM
`H = W8 A`, `6.5e9 x 4096` multiply-adds mod `2^56`), about 20 min on 32 cores. The provider has the GPUs and
the public weights, so it can compute the expensive part, but the hint-integrity rule says the customer must never
use a hint it did not build. The goal: let the provider do the work and accept the result only if it is exactly
what the customer would have built.

## 2. What the hint is, and why its final form cannot be checked cheaply

Per linear (`W8` int8 `m x n`, public `A = a_mlwe_rows(public_ring_rows(a_seed, n, linear))`, `n x 4096`):

1. `H = W8 A mod 2^56` (the GEMM; linear in `W8`),
2. `hs = round(q1 H / 2^56) mod q1` (exact rounding, `switch_hint`),
3. per 32-row block `C = sum_l A~_l k_l mod q1 P` with `A~ = centred(hs)` packed into `R_r` and `k = ks_a(ks_seed)`
   (linear in `A~`), then `c1 = ModDown(C) = round(C / P) mod q1` (exact rounding), stored in NTT form.

`c1` is not a linear (or low-degree) function of `W8`: two roundings sit between them. A random-projection check
compares a projection of the claimed object with the same projection of `W8`; for that the projection must commute
with every step. It commutes with steps 1 and 3 (and an NTT-slot projection of `C` is even rank-1 in (row, column)
of `hs`, so it would cost one pass over `W8`), but not with the roundings. Making the roundings checkable needs a
witness per rounded value:

* step 3's ModDown: the remainder `[C]_P` (60 bits per coefficient of `c1`): small, about the size of `c1`;
* step 2: the remainder `rho = (q1 H + 2^55) mod 2^56` of every entry of `H` (plus a quotient), and a proof that
  each `rho` lies in `[0, 2^56)`: without the range constraint any `c1` is consistent with some "remainders".
  That witness is as large as `H` itself (56 bits per entry, 16x the hint), and range proofs over 5.7e9 committed
  values need a succinct proof system (lookups / bulletproof-style arguments), which is out of scope here.

Information-theoretically the check of step 2 needs about 56 bits per entry of `H` on the client anyway: given
`hs`, the set of `H` it can come from is an interval of `~2^8` values, and deciding whether `W8 A mod 2^56` falls in
it is not a linear question. So: **the final key-switched form is not verified; the pre-switch form `H` is**, and the
client recomputes steps 2-3 itself. They are deterministic and cheap compared with step 1 (measured below).

## 3. The check

The provider streams `H` (7 bytes per entry, row blocks of 256 rows). Per linear the client draws, from the OS
CSPRNG and never sends anywhere, `U in {0,1}^(m x d)` and `X in {0,1}^(4096 x d)` with `d = 42`, streams `W8` once
(checked against the pinned manifest's SHA-256, i.e. the fingerprint) and accepts iff

    U^T H X == (U^T W8)(A X)   (mod 2^56)          (a d x d matrix)

**Soundness.** Let `E = H - W8 A mod 2^56` be nonzero, `2^v` the largest power of two dividing all its entries
(`v <= 55`), `E' = E / 2^v`, so `E' mod 2 != 0`. If the check passes then `U^T E X = 0 mod 2^56`, hence
`U^T E' X = 0 mod 2^(56-v)`, hence `U^T E' X = 0` over GF(2). Over GF(2), `E' X = 0` happens with probability at most
`2^-d` (each column `x` of `X` is uniform and independent; a nonzero `E'` has `Pr[E' x = 0] = 2^-rank <= 1/2`), and if
`Z = E' X != 0`, `U^T Z = 0` happens with probability at most `2^-d` likewise. So

    Pr[accept | H != W8 A mod 2^56] <= 2^-d + 2^-d = 2^(1-d) = 2^-41,

for every error `E` the provider may choose, of any 2-adic structure. It holds per linear with fresh challenges, so a
hint with any wrong linear is accepted with probability at most `2^-41` (corrupting more linears only lowers it).

**The mod-2^k pitfall.** A Freivalds vector over `Z_2^56` has soundness 1/2 against `E = 2^55 E'`: only the low bit
of each challenge entry matters for such errors. The adversarial review's F1 is the same phenomenon (two uniform
vectors mod `2^32`: 2 bits). Hence `d` independent binary columns (one bit each); the bound needs no field extension.
Equivalent alternatives: a Galois ring `GR(2^56, d)` (same cost), or a prime-field Freivalds over the integer
`W8 A` with the provider also sending the quotients (`~10` bytes per entry instead of 7, `d = 1-2`). The binary
2-adic check is chosen because bytes, not client CPU, are the bottleneck.

**Implementation.** Everything is computed mod `2^64` (a ring map onto mod `2^56`): `H X` as two float64 GEMMs on
28-bit slices (partial sums `< 4096 * 2^28 = 2^40`, exact), `U^T W8` as float32 GEMMs on 4096-row blocks (sums
`< 127 * 4096 < 2^24`, exact), `A X` from the `n/16` ring rows without expanding `A` (the `16 x 16` block structure of
`a_mlwe_rows` turns into one GEMM against a rearranged `X`; equality with the expanded product is tested). The check
is fed every row exactly once (tracked) and fails if a row is missing. The products have no data-dependent branches
(dense BLAS over 0/1 floats), and the verdict is only computed after the last chunk of a linear, so the provider learns
nothing about `U`, `X` before it has sent all of that linear's `H`; the challenges are wiped after the verdict.

**Then.** For every block the client computes `keyswitch(switch_hint(H), k)` with the same functions a local build
uses (`PKSHintStore.write`, shared by both paths) into a temporary store directory; only after every linear passed is
`meta.json` written (with the customer's HMAC) and the directory renamed into place. A failed check raises
`IntegrityError` and leaves nothing behind (tested).

## 4. Why an accepted hint is exactly the honest one

* **Seeds.** Both are drawn by the client from `os.urandom` (`new_seed`). Only `a_seed` is sent; the provider sees it
  on every maskgen request anyway (it is the upload's public `a`), so nothing new is revealed. `ks_seed` (`k`, the
  key-switching key's public part) never leaves the client, as before (F13). The provider cannot influence either
  (a provider-chosen `k` could be fatal, e.g. `k = 0` makes `beta = e + P iota(t)` reveal `t`): the job request
  carries the client's `a_seed` and nothing comes back that could change a seed; the store records the client's
  seeds, never the provider's.
* **`W8`.** Checked against the pinned manifest (SHA-256 per tensor, covered by the fingerprint); the masked head's
  `W8` is derived locally from the pinned plain head with the same deterministic quantization every party uses.
  The row norms that drive the noise refusal come from these weights, as in a local build.
* **`H`.** Accepted only if the check passes: except with probability `<= 2^-41`, `H = W8 A mod 2^56` exactly.
* **The rest** is the same deterministic code on the same inputs. Hence the stored hint files are bit-identical to a
  local build with the same seeds: `tests/provision/test_provision_e2e.py` compares every file's SHA-256 and the
  `meta.json` digests and row norms with `PKSHintStore.build(seeds=...)`.
* **Timing.** As in a local build, the hint exists before any reply secret `t'` is drawn; the factory starts only
  after `provision` returned.

So a provider-built, client-verified hint is equivalent to a locally built one, up to a `2^-41` failure probability
per provisioning (a one-time event, unlike per-segment checks).

## 5. What an adversarial hint could do without the check

Decryption is `v = c0 + c1 t' mod q1`, mask `= round(2^32 v / q1)`. With a wrong `c1' = c1 + delta`:

* **Gross error** (`delta` large or `H` for another seed / another linear): every decryption is garbage. The factory's
  40-bit self-check rejects every segment (DoS), and after 3 provider-attributable integrity failures the factory halts
  (F3). Availability only.
* **Selective failure** (the case the hint-integrity rule exists for, packet section 6): a small structured `delta`
  adds `delta * t'` to `v`; with `|delta t'|` tuned near the rounding margin (budget `q1 / 2^33 = 2^14.98` against a
  worst-case honest noise of `2^13.35`), whether a decrypted entry crosses a rounding boundary depends on the fresh
  reply secret `t'` (and the honest noise). The provider observes a failed (retried / missing) segment. Each segment
  then leaks up to about one bit about the `t'` of that group, also when it does not fail, i.e. about secrets of masks
  that are used; `t'` and `beta` determine `t`, and `t` and `b` determine `r`. The halt after 3 failures bounds the
  number of failure events, and one bit about a fresh 8192-trit secret is far from a break, but it is a leak the
  scheme's argument does not cover, and the online checks cannot see it (the wrong masks are rejected before use; the
  information is in the pattern of rejections).
* **Wrong metadata** (seeds, row norms): not possible here; both are the client's own.

With the check, none of this is reachable except with probability `2^-41` per provisioning.

## 6. Decision: default for the SDK

The argument shows equivalence with a local build (same function of the same inputs; `2^-41` soundness error; no
new information to the provider). So `hint="provider"` is the SDK default (`pgw.provision.client.provision`), with
`hint="local"` as the fallback and the explicit choice for customers who want zero provider involvement. A failed check
is never retried silently with the same challenges: `provision` raises, nothing is stored; the caller retries (fresh
seeds and challenges) or switches to `hint="local"`. The provider endpoints are off by default on the service
(`ServiceSettings.provision`). pks-v3 itself remains pending external review; this mode adds no new assumption to
it (one question below).

## 7. Costs (measured on this 4-vCPU dev box, shared with other jobs; see `benchmarks/provision_hint_bench.py`)

Bytes at Qwen2.5-7B (28 layers + masked head, 1.54M hint rows): `H` 44.2 GB streamed (7 bytes x 4096 per row; never
stored), `W8` 6.5 GB streamed once (the head's `W8` is derived locally), stored hint 2.37 GB (2.1 GB without the head).
The client CPU numbers are in section 7.1 below (filled from the benchmark). The bandwidth is the cost: at 1 Gbit/s
the `H` stream takes about 6 min, at 10 Gbit/s (same region) under a minute; below roughly 150-300 Mbit/s a local
build on 4-32 cores is faster, and `hint="local"` remains available.

### 7.1 Measured

See the table appended by the benchmark run (section 9).

## 8. Options not taken (for the reviewer and a later version)

* **Fewer bytes for some CPU:** the client computes the low `j` byte planes of `H` itself (`j/7` of the local GEMM)
  and the provider sends the upper `56 - 8j` bits; the same check applies mod `2^(56-8j)` to `W8 (A >> 8j)`.
  A linear dial between local build and this mode.
* **A hint that is linear mod a prime (scheme change, "pks-v4"):** with an upload modulus equal to `q1` (no switch of
  `H`), `hs = W8 A mod q1` and `C` are linear in `W8`; only ModDown remains, whose witness `[C]_P` is about the size of
  `c1`. The client would then receive about `2.3 x` the hint (~5 GB at 7B) and check it with prime-field Freivalds
  (mod `p0`, `p1` of `q1` and the special primes), an NTT-slot projection being rank-1 in (row, column), i.e. one pass
  over `W8`. This changes the upload ring's modulus (2^56 -> q1) and needs its own review.
* **Streaming the final hint from remote storage** (`pgw.provision.streaming`): see section 10.

## 9. Review question

Q7 (added to the packet): the customer reveals `a_seed` at provisioning instead of at its first maskgen request, and
verifies a provider-computed `H = W8 A mod 2^56` with the 2-adic check above before computing the hint itself. Is
anything in the provider's view, or in the order of events, different from a local build in a way that matters
(e.g. the provider knowing `A` and `H` earlier, adaptive choice of `H` across linears, the one-bit pass/fail outcome
of a provisioning)? We believe not: all of it is public data, the outcome concerns no secret (none exists yet), and
the challenges are fresh per linear and attempt.
