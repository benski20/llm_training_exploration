# PGW security whitepaper

Version 1.0 (protocol v1) · Status: **internal draft for external review**. This paper has not
yet been reviewed by an independent cryptographer. An external audit is on the roadmap (SPEC 16).

This paper is for security architects, CISOs and auditors who need to decide whether PGW meets
their confidentiality and integrity requirements. It covers the scheme, the security arguments and
their assumptions, what is and is not protected, and the residual risks: collusion, supply chain and
operations. The normative definitions are in [SPEC.md](../../SPEC.md). Numbers come from
[measurements.json](../measurements.json) and are labelled *measured*, *analytic* (exact from the
model shape) or *estimate*.

Contents

1. [Summary](#1-summary)
2. [System model](#2-system-model)
3. [The scheme](#3-the-scheme)
4. [Privacy: perfect secrecy per one-time use](#4-privacy-perfect-secrecy-per-one-time-use)
5. [Integrity: the Freivalds bound](#5-integrity-the-freivalds-bound)
6. [Correctness](#6-correctness)
7. [What is and is not protected](#7-what-is-and-is-not-protected)
8. [Collusion risks](#8-collusion-risks)
9. [Supply-chain risks](#9-supply-chain-risks)
10. [Operational assumptions and defences in depth](#10-operational-assumptions-and-defences-in-depth)
11. [Comparison with alternatives](#11-comparison-with-alternatives)
12. [Limitations and open items](#12-limitations-and-open-items)

---

## 1. Summary

PGW splits LLM inference between a **gateway** inside the customer's trust boundary and a
**compute service** run by us. The service does the expensive part: every weight matrix product,
about 99% of the FLOPs. It sees only one-time-padded vectors `x̂ = x + r mod 2^K`. The gateway keeps
everything else in plaintext: embeddings, normalisation, rotary position encoding, attention with the
KV cache, activation functions, the LM head and sampling.

* **Privacy.** For each linear call, the service's view is uniformly distributed and independent of
  the customer's data. That holds against a computationally unbounded, actively malicious service, on
  three conditions: every mask `r` is uniform, it is used exactly once, and it never reaches the
  service (Theorem 2). The masks are expanded from 256-bit seeds with ChaCha20. The deployed
  guarantee is therefore computational, at the strength of that stream cipher (section 4.4).
* **Integrity.** Every response passes a range check and `c` Freivalds checks over `Z_{2^K}` with
  secret vectors. A wrong response is accepted with probability at most `2^(−c·h)`, where `h` depends
  on the ring and layer width (Theorem 3; for example `2^−16` for K40A12 at width 4096 with `c = 2`).
  The first detected failure aborts the request and trips a circuit breaker.
* **Correctness.** The integer arithmetic is exact. Masked inference yields bit-identical logits to
  the same quantized model run locally (*measured*: max logit difference 0.0 on Qwen2.5-0.5B).
* **Not protected.** Model identity (the weights are public by design), traffic metadata (token
  counts, timing, tenant), anything on a compromised gateway host, and collusion between the mask
  factory and the service operator. See sections 7 to 9.

## 2. System model

### 2.1 Parties

| party | operated by | sees | trust assumption |
|---|---|---|---|
| Applications | customer | their own prompts and completions | n/a (customer side) |
| Gateway (`pgw.gateway`, `pgw.engine`) | customer | all plaintext, masks, scales | trusted; inside the customer boundary |
| Mask store (`pgw.maskstore`) | customer | encrypted segments, KEK access | trusted for confidentiality and single use |
| Mask factory (`pgw.factory`) | customer **or** an independent partner | seeds, `r`, `W r`, `u`, model | must **not collude** with the service operator |
| Compute service (`pgw.service`) | us | `x̂`, `ŷ`, shapes, timing, tenant id | **untrusted**: honest-but-curious or fully malicious |
| Network | anyone | TLS ciphertext, traffic volume and timing | untrusted |

### 2.2 Adversary

The main adversary **A** controls the compute service completely: code, hardware, operators, logs,
and any data it chooses to keep. A may deviate arbitrarily from the protocol (*malicious*), may
coordinate across tenants and sessions, and is not computationally bounded unless we say so. A knows
the model weights, the protocol, the PGW source code and the plaintext distribution (for example
that the customer is a hospital). A does not know the customer's mask seeds or Freivalds vectors.

Secondary adversaries are covered in the [threat model](threat-model.md): a network attacker, a
malicious partner factory, malicious insiders on the customer side, and a supply-chain attacker.

### 2.3 Security goals

| goal | statement |
|---|---|
| G1 input privacy | A learns nothing about prompts, completions, activations or logits beyond the explicitly accepted metadata (section 7.2) |
| G2 output integrity | A cannot make the gateway accept a wrong product without detection, except with probability `2^(−c·h)` per call |
| G3 exactness | Without an attack, outputs equal those of the reference quantized model, bit for bit |
| G4 fail-closed | Every detected anomaly (integrity, mask reuse, noise alarm, rollback) stops masked traffic instead of degrading silently |

## 3. The scheme

### 3.1 Notation

| symbol | meaning |
|---|---|
| `K` | ring bits: arithmetic is in `Z_{2^K}` (presets `K ∈ {32, 40, 48}`) |
| `A` | activation bits; `qmax = 2^(A−1) − 1` |
| `W8 ∈ [−127,127]^{m×n}` | public int8 weight matrix of one linear group, with per-row scales `sw` |
| `x ∈ R^n` | one token's activation entering a linear group; `xq ∈ [−qmax, qmax]^n` is its quantization, with scale `sx` |
| `r ∈ Z_{2^K}^n` | one-time mask (one per token position per linear group) |
| `u_1..u_c ∈ Z_{2^K}^m` | secret Freivalds vectors (per linear, per mask segment) |
| `B = qmax·127·n` | bound on the absolute value of any genuine output `yq_j` |
| `centered(v)` | the representative of `v mod 2^K` in `[−2^(K−1), 2^(K−1))` |

Ring presets: **K40A12 (default)**, K48A20 (maximum quality), K32A8 (cheapest; *measured* to lose
quality, perplexity 11.09 vs 10.92 for float activations on Qwen2.5-0.5B, against 10.93 for A12).

### 3.2 Offline phase (mask factory)

For every `(model fingerprint, ring)` the factory produces *segments* of `S` slots (one slot = one
token position, holding one mask row for each of the `4·L` linear groups):

1. Draw a fresh 32-byte segment seed from the operating system CSPRNG (`os.urandom`).
2. For slot `t` and linear `ℓ`, expand `r[t,ℓ] = SeedExpander(seed).ring_uniform_rows(n, K, t, 1, "r", ℓ)`:
   HKDF-SHA-256 derives a per-label key, ChaCha20 produces a keystream, and each ring element is a
   64-bit little-endian word reduced mod `2^K`. Because `2^K` divides `2^64`, the reduction is exactly
   uniform with no modulo bias.
3. Compute `W8·r mod 2^K` exactly (int8 limb GEMMs on GPU, or exact CPU paths).
4. Draw `u_1..u_c` from the same expander (label `"u"`) and compute `u_iᵀW8 mod 2^K`.
5. Write the segment: the seed and the `uᵀW8` rows in chunk 0, then the `W8·r` rows, each chunk
   encrypted with AES-256-GCM under a random per-segment data key (DEK). The DEK is wrapped by the
   customer's key-encryption key (KEK), or HPKE-sealed to the customer's X25519 key when a partner
   produces the segment (SPEC 7.7). `r` itself is never stored; it is regenerated from the seed.

### 3.3 Online phase (one linear group, one batch of rows)

```
gateway:  xq, sx  = quantize(x)                       per-token symmetric, A bits
          reserve slot(s) durably, regenerate r, read W8·r and uᵀW8
          x̂      = (xq + r) mod 2^K                 ──────▶  service
service:  ŷ      = W8·x̂ mod 2^K                     ◀──────  (K/8 int8 GEMMs; stateless)
gateway:  check   u_i·ŷ ≡ (u_iᵀW8)·x̂  (mod 2^K)  for i = 1..c        (Freivalds)
          yq     = centered(ŷ − W8·r)
          check   |yq_j| ≤ B for all j                                  (range)
          y      = yq · sx · sw + b                                      (float, bias added locally)
```

Each layer has four groups, `qkv`, `o`, `gate_up` and `down`, so a forward pass makes `4·L` round
trips. During prefill, all prompt tokens of a group travel in one round trip, and concurrent
sessions share round trips. Scales `sx`, `sw` and biases never leave the gateway. `sw` and the
weights are public anyway.

### 3.4 Single use

A slot is consumed durably (SQLite transaction plus fsync) **before** any mask byte is decrypted for
the caller (SPEC 7.5). Rows are handed out strictly in order and are never returned. Unused slots of
a finished request are burned. A crash can lose masks; it cannot reuse them. Rollback of the store
state is detected by mirroring its generation counter into witnesses outside the database (SPEC 7.6):
an HMAC-chained journal keyed from the KEK and, when configured, an external monotonic counter (TPM,
HSM or key-provider counter, or a counter file on separate storage). The journal alone detects a
restore of `state.sqlite` by itself. It does **not** detect a restore of the database *and* the
journal from the same snapshot: that needs no key, and by default both files sit on the mask volume.
Only a monotonic counter outside that volume catches a whole-volume snapshot revert. The default
`counter` setting is `none`, so production deployments must configure one, or forbid snapshot and
backup restores of the mask volume and accept the residual risk in writing (threat model T-22).
With a key provider that cannot compute MACs, the journal is disabled as well, and a counter is the
only rollback witness.

## 4. Privacy: perfect secrecy per one-time use

### 4.1 One call

**Lemma 1 (one-time pad over a group).** Let `G = Z_{2^K}^n`. If `r` is uniform on `G` and independent of
`xq`, then `x̂ = xq + r` is uniform on `G` and independent of `xq`.

*Proof.* For any fixed `a` and `v ∈ G`, `Pr[x̂ = v | xq = a] = Pr[r = v − a] = |G|^−1`, which does not
depend on `a`. ∎

So for a single call, the service's view has the same distribution whatever the input is. This is
Shannon's perfect secrecy. The only thing the call reveals is its shape (`n` and the number of rows).

### 4.2 A whole session against an adaptive, malicious service

A session is interactive: the activation `x_i` sent in round `i` depends on the prompt and on what the
service returned in rounds `1..i−1`. A malicious service could pick its responses to steer later
activations. Nevertheless:

**Theorem 2 (transcript privacy).** Fix any service strategy `A` (possibly malicious and unbounded)
and any two customer inputs `P, P'` (prompts, sampling seeds, and so on) that produce the same
metadata: the same model, the same number of rows in every round, and the same number of rounds. Let
the masks `r_1, r_2, …` be uniform, mutually independent, independent of the inputs, never revealed
to `A`, and each used once. Then the joint view `(x̂_1, …, x̂_N)` of `A` has the same distribution
under `P` and under `P'`: it is uniform on `G_1 × … × G_N`.

*Proof sketch.* By induction on rounds. Let `H_{i−1}` be everything `A` has seen and done before
round `i`: its view, its random coins and its responses. The gateway's activation `x_i` is a
deterministic function of the input, the gateway's sampling randomness, `H_{i−1}`, and the masks
`r_1..r_{i−1}` (through unmasking). The mask `r_i` is independent of all of these, because it is
fresh and never used before. By Lemma 1, conditioned on any value of `(input, H_{i−1})`, `x̂_i` is
uniform. Hence `x̂_i` is uniform and independent of `(input, H_{i−1})`. Multiplying the conditional
distributions gives the product-uniform distribution for every input. Everything else `A` sees (its
own responses) is a function of the view and its own coins. ∎

Consequences:

* A malicious service gains nothing by returning crafted responses. They change what the gateway
  computes, but every later activation is again freshly masked. The single caveat is the
  accept/abort decision, which is visible to the service (section 5.4).
* Batching across tenants is safe. Every row any party sends is independently uniform.
* Retries are safe. A transport retry resends the identical `x̂`, which reveals nothing new, and never
  remasks the same row with a different mask.

### 4.3 Why one-time matters: the reuse attack

If one mask `r` were used for two rows, then `x̂_1 − x̂_2 = xq_1 − xq_2 (mod 2^K)`. Activations are small
(`|xq| ≤ qmax ≪ 2^K`), so this difference is the exact plaintext difference, which leaks content
directly (for example equal tokens and edit structure). More generally, any linear dependency among
masks leaks the same linear combination of plaintexts. Single use is therefore a hard invariant. It
is protected by the following:

1. durable consumption **before** decryption (SPEC 7.5);
2. rows taken strictly in order, never returned (reservation API);
3. rollback detection on the store state (HMAC journal against a restored database; a monotonic
   counter off the mask volume against whole-volume snapshot reverts, section 3.4, SPEC 7.6);
4. a fresh 256-bit seed per segment, with segment ids unique per store;
5. a duplicate-row detector in the outbound-noise monitor. A repeated mask on a repeated input
   produces a repeated `x̂` row; the monitor alarms and, in fail-closed mode, blocks traffic.

A `mask_reuse` error is never retried. It stops traffic until the store has been investigated (see
the [audit checklist](../compliance/audit-readiness.md)).

### 4.4 Instantiation: from perfect to computational secrecy

Theorem 2 assumes truly uniform masks. PGW derives masks deterministically from a 32-byte seed
(HKDF-SHA-256 then ChaCha20, SPEC 7.3) so that `r` need not be stored. Against a computationally
unbounded adversary, seed expansion is a weakness in principle: an exhaustive search over `2^256`
seeds would identify the one whose `r` makes `x̂ − r` small. The deployed guarantee is therefore:

> Under the standard assumption that ChaCha20 keyed through HKDF-SHA-256 is a pseudorandom generator,
> no efficient adversary can distinguish the real transcript from uniform with advantage better than
> the PRG advantage, about `2^−256` for classical adversaries and `2^−128` under Grover-type quantum
> search, per segment seed.

This is the same assumption on which TLS 1.3 confidentiality rests. If a customer requires
unconditional secrecy, the factory could store `r` explicitly (doubling mask storage and store I/O).
That variant is a format extension that v1 does not implement.

### 4.5 What the service learns from shapes

The service learns `(model fingerprint, ring, linear id, rows)` per item, plus timing and tenant.
From these it can infer prompt length in tokens (prefill rows), the number of generated tokens, the
number of concurrent sessions, inter-token timing and activity patterns. Section 7.2 lists this
metadata and the mitigations.

### 4.6 Empirical sanity check

Security rests on the proof, not on statistics. As a check that the implementation matches the
proof, the prototype tested what the server sees (*measured*, Qwen2.5-0.5B end to end): a top-byte
chi-square of 241.4 with 255 degrees of freedom (expected 255 ± 22.6), and a correlation with the
plaintext activations of -0.0006 over 141,312 values. In production the outbound-noise monitor runs
these tests continuously on sampled payloads (section 10).

## 5. Integrity: the Freivalds bound

### 5.1 The check

For each response the gateway verifies, for every row and each of `c` secret vectors `u_i`:

```
u_i · ŷ ≡ (u_iᵀ W8) · x̂   (mod 2^K)        and        |centered(ŷ − W8·r)_j| ≤ B  for all j
```

`u_iᵀW8` was precomputed by the factory, so the check costs `O(m + n)` per row instead of the `O(mn)`
of recomputing the product. Production configuration requires `freivalds_rate = 1.0` (every response
is checked) and the default `c = 2` (`PGW_FACTORY_CHECKS`, up to 8).

### 5.2 The bound

**Lemma 3.** Let `e ∈ Z^m`, `e ≠ 0`, and let `v` be the smallest 2-adic valuation among its non-zero
entries. For `u` uniform on `Z_{2^K}^m`, `Pr[u·e ≡ 0 (mod 2^K)] ≤ 2^(v−K)`.

*Proof.* Pick `j` with `ν₂(e_j) = v`. Condition on all other coordinates of `u`. The congruence
`u_j·e_j ≡ c (mod 2^K)` has at most `2^v` solutions for `u_j` out of `2^K`. ∎

**Theorem 3 (soundness per call).** Let the service return `ŷ' = ŷ + e (mod 2^K)` with `e ≢ 0` in some
row. If the range check passes, then `centered(e)` restricted to that row satisfies `0 < |e_j| ≤ 2B`
for some `j`. With

```
h = K − ⌈log2(4·qmax·127·n)⌉     (pgw.core.soundness_bits)
```

the false-accept probability of that row is at most `2^(−c·h)` for `c` independent secret vectors.

*Proof sketch.* The genuine `yq` and the decoded `yq' = centered(yq + e)` both satisfy `|·| ≤ B`, and
`2B < 2^(K−1)`, so `centered(e) = yq' − yq` exactly, with `|e_j| ≤ 2B`. Any non-zero integer of absolute
value at most `2B` has 2-adic valuation `v ≤ ⌊log2(2B)⌋ ≤ K − h − 1`. By Lemma 3, each `u_i` accepts with
probability at most `2^(v−K) ≤ 2^(−h−1) ≤ 2^(−h)`. The `u_i` are independent and secret, so the
probabilities multiply. If the range check fails, the response is rejected anyway. ∎

The bound does not rely on the adversary being ignorant of `yq`. It holds even if the adversary knows
the plaintext. Without the range check, an adversary could use `e = 2^(K−1)` (valuation `K−1`) and pass
each check with probability 1/2. The range check is what forces `e` to low valuation.

### 5.3 Concrete values

`c·h` in bits (*analytic*, from `pgw.core.soundness_bits`); the false-accept probability per call is `2^−(c·h)`:

| ring | width `n` | example linear | `h` | `c = 2` | `c = 4` | `c = 8` |
|---|---|---|---|---|---|---|
| K40A12 | 896 | Qwen2.5-0.5B `qkv` | 10 | 20 | 40 | 80 |
| K40A12 | 4096 | Llama-3-8B `qkv`/`o`/`gate_up` | 8 | 16 | 32 | 64 |
| K40A12 | 14336 | Llama-3-8B `down` | 6 | 12 | 24 | 48 |
| K40A12 | 28672 | Llama-3-70B `down` | 5 | 10 | 20 | 40 |
| K48A20 | 4096 | Llama-3-8B | 8 | 16 | 32 | 64 |
| K32A8 | 4096 | Llama-3-8B | 4 | 8 | 16 | 32 |
| K32A8 | 14336 | Llama-3-8B `down` | 2 | 4 | 8 | 16 |

K48A20 has the same `h` as K40A12 because the eight extra ring bits go to activation precision.
**Recommendation:** with K32A8, or when a single undetected wrong product is costly, set
`checks = 4` or more. The check costs `c` dot products of length `m + n` per row, which is negligible
next to the round trip.

### 5.4 Adaptive attackers, probing and selective failure

*Probing.* The service observes whether the gateway accepted a response (rejection aborts the
request). Until the first rejection, all the adversary has observed is "accepted", so its sequence of
errors is effectively fixed in advance. The first cheat is accepted with probability at most
`2^(−c·h)`. On rejection the gateway fails the request with `integrity_check_failed`, increments
`pgw_gateway_integrity_failures_total`, writes an audit event and opens the circuit breaker for that
model. The breaker reopens after `integrity_cooldown_s` (default 900 s), or only after an admin reset
if the cooldown is unset. Each rejection tells the adversary only that `u·e ≢ 0`, which rules out at most
a `2^(−c·h)` fraction of the vectors. After `N` detected probes, the chance that some cheat was accepted
is approximately `(N+1)·2^(−c·h)` for `N ≪ 2^(c·h)`. Freivalds vectors are per segment, so they are
refreshed every `segment_slots` tokens (default 4096).

*Selective failure (privacy side channel).* A malicious service might choose `e` so that the range
check passes or fails depending on the plaintext (for example `e_j = +B/2` passes iff `yq_j ≤ B/2`), and
then watch for the abort. The Freivalds checks pass or fail independently of the plaintext
(`u` is independent of `x`), and they reject any `e ≠ 0` except with probability `2^(−c·h)`. So the
distributions of the abort bit under any two inputs differ by at most `2^(−c·h)` in statistical
distance. Each such probe gives a distinguishing advantage of at most `2^(−c·h)` and is detected with
probability at least `1 − 2^(−c·h)`. The breaker then cuts the adversary off.

*Scope of the check.* Freivalds verifies the **service's** product `W8·x̂`. It does not verify the
factory's `W8·r` rows. A faulty factory shifts the decoded `yq`. Random corruption almost always
fails the range check, but a small deliberate offset would pass. The factory is therefore trusted
for mask integrity (SPEC 3.1), though not for privacy against the service unless it colludes.

*Denial of service.* A malicious service can always refuse, delay or corrupt responses. PGW detects
corruption but cannot prevent denial of service. Availability is protected contractually (SLA) and by
failing over to another endpoint, not cryptographically.

## 6. Correctness

* Every backend computes `W8·x̂ mod 2^K` exactly: GPU int8 limb GEMMs on balanced base-256 digits with
  int32 accumulation, and exact float64 or float32 CPU paths with bounded partial sums (SPEC 5).
  `pgw doctor` runs an exactness self-test on every available backend at install time.
* Decoding is unambiguous because `qmax·127·n < 2^(K−1)`. This is checked when a pack is loaded
  (`RingParams.check_in_features`).
* *Measured:* Qwen2.5-0.5B masked versus quantized plaintext gave identical tokens with max logit
  difference 0.0. The engine test suite asserts bit-identical logits between `MaskedBackend` and
  `PlainBackend` on every ring.
* Quantization (int8 weights, A-bit activations) is the only approximation. It is the same
  approximation as running that quantized model locally and does not depend on the masking.

## 7. What is and is not protected

### 7.1 Protected against the compute service (G1)

| asset | how |
|---|---|
| Prompts, system prompts, retrieved documents (RAG context), tool outputs | tokenised and embedded on the gateway; never sent |
| Every hidden activation entering a linear layer | one-time masked (Theorem 2) |
| Attention scores, KV cache, norms, SiLU outputs | computed on the gateway only |
| Logits, sampled tokens, completions, log-probs | LM head and sampling on the gateway |
| Activation scales `sx` | stay on the gateway |
| Masks, seeds, Freivalds vectors | never in any wire message; the service has no endpoint that accepts them |
| API keys of applications | terminate at the gateway; the service sees only a tenant token or certificate |

### 7.2 Not protected (accepted metadata and out-of-scope threats)

| item | who learns it | mitigation |
|---|---|---|
| Model identity (pack fingerprint), ring | service | none needed: weights are public by design |
| Rows per round: prompt length, generated length, concurrency | service | bucketed row padding and constant-rate dummy rounds are **planned** (SPEC 3.4) and not in the v1.0 build; batching hides per-session counts within a batch |
| Timing: time to first token, inter-token gaps, activity per hour | service, network | dummy traffic (planned); batching |
| Tenant identity, request volume (for billing) | service | contractual (DPA) |
| Traffic volume and timing on the wire | network observers | TLS hides content; volume is visible; private links (PrivateLink/PSC) |
| Anything on a compromised gateway host | attacker on that host | customer host hardening; out of scope (SPEC 1) |
| Misuse by authorised customer users | customer | API keys, scopes, quotas, audit log |
| Factory and service collusion | colluding parties | section 8 |
| Malicious PGW code | supply-chain attacker | section 9 |
| Availability | service | SLA, multiple endpoints; detection only |
| Secrecy of model weights | nobody | not a goal: open-weight models only |

## 8. Collusion risks

Privacy rests on one organisational assumption: **whoever can learn `r` (the factory) must not share it
with whoever sees `x̂` (the service).** Given seeds and `x̂`, `x = x̂ − r` is immediate.

| configuration | who holds seeds | collusion risk | residual assurance |
|---|---|---|---|
| Customer-run factory (recommended for regulated data) | customer only | none from us | supply chain of the factory code (section 9) |
| Independent partner factory (topology C) | partner | partner and service operator collude | contractual independence, audits, separate ownership and staff, HPKE delivery so the partner never holds the customer's KEK |
| Factory run by the service operator | **forbidden** | total | none: the configuration is not offered |

Controls that make collusion harder or detectable:

* **Structural separation.** Seeds are generated inside the factory and leave it only inside
  AES-GCM-encrypted segments sealed to the customer's key. The partner keeps no copy by contract and
  by default configuration (export then delete). Auditors can verify both.
* **No path.** The compute service has no endpoint, field or code path that accepts masks or seeds
  (SPEC 3.3). An import-graph test enforces that `pgw.service` never imports `pgw.maskstore` or
  `pgw.factory`.
* **Dual sourcing (optional, for high assurance).** The customer can additively combine two
  independent mask sources, one partner and one in-house, so that colluding with a single factory is
  not enough. With `r = r_1 + r_2` the store holds `W8 r_1 + W8 r_2`. This needs a format extension
  and is listed as roadmap, not v1.
* **Customer-only seeds.** A customer with any GPU (a single T4 produces about 550k 8B slots/h, measured)
  or enough CPU can remove the partner from the trust base entirely.

## 9. Supply-chain risks

PGW software runs on both sides of the boundary. A malicious or compromised build is the most
important residual risk, because it can bypass the mathematics.

| component | what a malicious build could do | mitigations |
|---|---|---|
| Gateway | send plaintext instead of `x̂`; exfiltrate prompts over another channel; reuse masks | egress allowlist (only configured service origins); customer firewall egress policy; outbound-noise monitor; source-available code and reproducible builds; signed images with SBOM; independent audit; the customer can run its own egress tap that checks payloads look uniform |
| Mask factory | use predictable seeds (a silent backdoor that needs no network) | factory runs without network (air-gapped option); seed generation is a few lines (`os.urandom(32)`) open to review; an independent Rust implementation with cross-language known-answer vectors (`tests/vectors/core_v1.json`) lets customers build their own factory; roadmap: mix customer-supplied entropy (HSM RNG) into the seed |
| Mask store | skip durable consumption (reuse) | rollback and generation checks; duplicate-row detection in the noise monitor at a separate layer |
| Model pack | substitute weights (quality or backdoor attack, not a privacy break) | SHA-256 fingerprint over every tensor and tokenizer file; gateway, factory and service compare fingerprints; customers can rebuild the pack from the public checkpoint with `pgw pack` and compare fingerprints |
| Dependencies (torch, cryptography, transformers, ...) | any of the above | pinned versions, SBOM, vulnerability scanning; gateway image has no compiler and runs read-only as a non-root user |
| Compute service | anything, by assumption | nothing needed: it is untrusted |

**Important.** The noise monitor ships in the same package as the gateway. It protects against
*bugs* and misconfiguration, not against a deliberately malicious build. Against a malicious
build, the controls are independent: the customer's egress policy, review and rebuild from source,
signature verification and the external audit.

## 10. Operational assumptions and defences in depth

| assumption | enforced or monitored by |
|---|---|
| OS CSPRNG is sound | `pgw doctor` CSPRNG self-test; standard OS RNG |
| Masks single use | SPEC 7.5 transaction ordering; `pgw_masks_consumed_slots_total`; duplicate-row detection |
| No state rollback | HMAC journal (detects a restored database); monotonic counter off the mask volume (TPM, HSM, provider or file counter; needed against whole-volume snapshot reverts, default `none`); `verify_integrity()` refuses to serve on mismatch; no snapshot or backup restore of the mask volume (section 3.4) |
| KEK confidential | file provider (mode 0600), env, PKCS#11 HSM or KMS; DEKs wrapped per segment; shredding of consumed segments (wrapped DEK overwritten first; on copy-on-write file systems and SSDs complete erasure relies on full-disk encryption and KEK rotation) |
| Every payload masked | outbound-noise monitor: top- and low-byte chi-square (255 dof), top-bit balance, correlation with `xq`, duplicate rows; false-alarm rate under 1e-9 per window; fail-closed by default |
| Every response checked | `freivalds_rate = 1.0` enforced in production policy; dev modes refused unless `allow_dev_modes` |
| Fingerprints agree | gateway checks the service's `/pgw/v1/models` and each segment header; mismatch gives `fingerprint_mismatch` |
| TLS to the service | `require_service_tls`; mTLS client certificates; bearer tenant tokens stored hashed |
| No content in logs | logging rules (SPEC 11.2); the audit log has a fixed field allowlist without content fields |

## 11. Comparison with alternatives

Figures from our research notes (`experiments/onprem_privacy/CRYPTO_OPTIONS.md`). Figures for other
systems are their published or our measured prototype values, not product benchmarks.

| approach | servers | privacy basis | server cost per 8B token | client burden | notes |
|---|---|---|---|---|---|
| **PGW one-time masks** | 1 (plus non-colluding factory) | one-time pad (computational via PRG) | 0.45 s batch 1 / 0.014 s batch 32 on a T4 (*measured*) | about 8.3 MB of masks per token (K48); less than 1% of FLOPs online | exact; integrity by Freivalds |
| Linear-only CKKS HE | 1 | lattice (computational) | about 120 s on a T4 (*measured* prototype) | about 1 GB, seconds of CPU | about 10,000x plaintext cost |
| Dual-LPN masking (Maverick-style) | 1 | LPN (computational) | close to plaintext | about 120 GB client tables at 8B (our analytic extrapolation) | no non-collusion assumption |
| 2-party secret sharing (Gupta, Katz, Miers) | 2 non-colluding | information-theoretic sharing | close to plaintext per server | small | two operators must not collude |
| TEEs (confidential GPUs) | 1 | hardware vendor trust | close to plaintext | none | trust in vendor attestation; side channels |
| Full FHE | 1 | lattice | about 20 s on 8 GPUs (published) | small | slow |

PGW trades the non-collusion assumption on the mask supply, plus mask storage and bandwidth, for
exactness and near-plaintext server cost.

## 12. Limitations and open items

* The security argument has not had external review. A formal write-up and an independent audit are
  planned (SPEC 16).
* Metadata (lengths, timing) is not hidden in v1.0. Padding and dummy-round mitigations are planned.
* Integrity bounds are weak for K32A8 at default `c = 2` (section 5.3). Use K40A12 or raise `checks`.
* Availability depends on the service and is not cryptographically protected.
* The guarantee is computational at the PRG level (section 4.4), not unconditional.
* Model weights must be public. Fine-tuned private weights would need a different scheme, because the
  factory and gateway need `W`.
* Latency has a floor of `4·L` round trips per token (*analytic*: 128 round trips for 8B). This favours
  same-region deployment (topology B) and batching.
