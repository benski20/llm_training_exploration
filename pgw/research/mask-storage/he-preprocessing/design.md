# HE preprocessing for PGW masks: design note (track he-preprocessing)

Labels on numbers: **[m]** measured (here or in PGW docs), **[c]** cited, **[e]** estimate or analytic.
Files in this folder: `proto_lh.py` (exact end-to-end prototype), `proto_lh_results.txt`, `client_bench.c`
(gateway cost microbenchmark), `cost_model.py` and `cost_model_out.json`.

## 0. Summary

Delphi-style HE preprocessing works for PGW with one provider. Taken to its end, it removes the mask
factory and the per-token mask store.

The scheme exploits three facts about PGW that the HE inference literature (Delphi, Cheetah,
BumbleBee, NEXUS, Bolt, Rhombus) does not have:

1. **The weights W are not secret from the customer.** So the service needs no circuit privacy or
   noise flooding. Those are a large share of the cost in those papers.
2. **Masks do not depend on the data**, and we always have at least 256 token slots to prepare. So we
   can encrypt **along the token axis**: polynomial coefficient j is token j. Then "plaintext W times
   ciphertexts" is a plain integer GEMM mod q on the GPU. There are no NTTs, rotations or key
   switching on the server.
3. **The `a` part of an LWE ciphertext can be a fixed public matrix A.** Then `W·A` is a one-time
   **hint** H (the SimplePIR idea [c: Henzinger et al., USENIX Security 2023]). The gateway removes
   the mask from W's output with a cheap ring product `<H_i, s>` (NTT, about 46 ns per row per token,
   measured).

Combining the three, the preprocessing round folds into the online call:

**PGW-LH (hinted MLWE masking).** The gateway sends `v = Δ·x − P` per linear. The pad
`P = <a, s> + e` is an MLWE sample made on the gateway CPU. The service returns `z = W·v mod q`,
the same GEMM as today but on 56-bit instead of 40-bit operands. The gateway decodes
`y = round((z + <H, s>)/Δ) = W·x` exactly.

| per 7B token (Qwen2.5-7B) | today | PGW-LH |
|---|---|---|
| customer persistent storage | 6.95 MB per token, so 6.95 TB per 1M tokens [m/e] | **28.5 GB one-time hint** plus a 4.6 GB working buffer, flat in tokens [e] |
| customer compute | mask factory: 5.9 T4-hours per 1M tokens [m: 169k tokens/h] | **0.10 core-s per token, scalar C** (1.2 cores sustain 1M tokens/day) [m-derived] |
| our GPU | online GEMM, 5 int8 digit GEMMs | online GEMM with 7 digit GEMMs (×1.4); no factory [e] |
| network | 4.16 MB up + 6.95 MB down | 5.82 MB up + 9.73 MB down (full z), or 6.26 MB down (truncated z) [e] |
| extra round trips | 0 | 0 online; one hint download per model and customer |
| privacy | PRG (ChaCha20) + non-collusion with the factory | **MLWE** (dimension 2560, q ≈ 2^56, σ ≈ 3.2), single provider, no factory party |
| integrity | Freivalds over Z_2^K, about 20 bits with c = 3–5 checks | **Freivalds over F_q: 2^-55 per check, c = 1**; abort events are independent of client secrets |

Break-even: the hint costs as much as 4,096 tokens of today's masks (`dim·8 B / (K/8 B)`). The same
holds for 0.5B (6.2 GB) and for 70B (138 GB) [e].

I checked correctness end to end on real layer widths (n = 3584 and n = 18944, worst-case ±127
weights, worst-case ±2047 activations). Results are exact. The worst observed noise is 2^17.6
against a budget of 2^21 [m].

## 1. Literature: what exists and why it is overkill here

| work | what it does | cited cost | relevance to PGW |
|---|---|---|---|
| Delphi (Mishra et al., USENIX Security 2020) | Offline: client sends Enc(r), server returns Enc(W r − s) and keeps s. Online: shares. | ResNet-32 online 3.8 s, 60 MB. Gazelle-style preprocessing about 158 s, 8 GB [c] | Template for variant A below. It needs circuit privacy because W is secret there; that is not needed here. |
| Gazelle (Juvekar et al., USENIX Security 2018) | SIMD packing with rotations | see Delphi row [c] | Rotations and key switching are unnecessary when encrypting along the token axis |
| Cheetah (Huang et al., USENIX Security 2022) | Coefficient encoding, no rotations, t = 2^ℓ | ResNet50 under 2.5 min WAN, 2.3 GB [c] | Our layout is Cheetah's coefficient encoding at the corner m_w = n_w = 1, B_w = N, which turns pt × ct into a GEMM |
| Iron (Hao et al., NeurIPS 2022) | First BERT 2PC | (not retrieved) | Same family as Cheetah |
| BumbleBee (Lu et al., NDSS 2025) | Batched matmul (HE), 80–90% less communication than prior work | **LLaMA-7B about 8 min per token on CPU** (NDSS version; the ePrint version says about 14 min) [c] | Shows that full 2PC with private weights is far from usable. We only need the linear part with public W. |
| NEXUS (Zhang et al., NDSS 2025) | Non-interactive FHE transformer | BERT-base 37.3 s, 164 MB. BOLT needs 59.61 GB and 10,509 rounds [c] | FHE for nonlinear layers is not needed; the gateway does those in plaintext |
| Bolt (Pang et al., IEEE S&P 2024) | Communication-efficient BERT 2PC | 59.61 GB per BERT-base inference (as cited by NEXUS) [c] | same as above |
| Rhombus (He et al., CCS 2024) | RLWE coefficient-encoded mat-vec, 7.4–8× over HELiKs [c] | | Best general MVM, but it still pays NTT and packing costs that the token-axis layout avoids |
| SimplePIR (Henzinger et al., USENIX Security 2023) | LWE with a fixed public A; the client keeps the hint D·A | about 10 GB/s per core server; 121 MB hint for a 1 GB database [c] | **This is the key trick.** H = W·A is reusable, so only `s` changes per query |
| Slalom (Tramèr & Boneh, ICLR 2019) | TEE outsources linear layers to a GPU with precomputed blinding and unblinding factors (exactly PGW's scheme), plus Freivalds | Notes that the unblinding factors need "a large amount of RAM, or fast access to disk" [c] | PGW today is Slalom with a gateway in place of the TEE, so it has the same storage problem |
| GPU HE libraries: Phantom (TDSC 2024), HEonGPU, 100x (TCHES 2021), TensorFHE, WarpDrive (1218 KOPS NTT on A100 [c]) | NTT-heavy BFV/CKKS on GPU | | Not needed: in PGW-LH the server does no NTT. The int8 digit GEMMs already in `pgw.core.ringmm` suffice. |

Classic HE matrix-vector product with feature-axis encoding (variant C: HE in the online path) costs
L RNS limbs × the plaintext MACs, plus output packing, plus ciphertext expansion of about
log q / K ≈ 3× on each wire direction for N = 4096. It adds HE latency to every one of the 112 round
trips per token. It is dominated by PGW-LH on every axis, so I did not cost it further.

## 2. The PGW-LH protocol (recommended)

### Parameters, per pack and ring preset

- Plaintext space: t ≥ 2·qmax·127·n_max + 1. For A12 and n = 18944, |Wx| < 2^32.2, so **t = 2^34**.
  For A20: t ≈ 2^42.
- **q**: an NTT-friendly prime, q ≡ 1 mod 2N', q < 2^56 (prototype: q = 72057594037913089).
  Δ = ⌊q/t⌋ ≈ 2^22. Decoding needs |W·e| < Δ/2.
- **MLWE**: ring R_q = Z_q[X]/(X^256+1), rank k = 10 (dimension 2560). Secret s is CBD(η = 2) and
  error e is CBD(21) (σ ≈ 3.24).
  - The HE Standard (Albrecht et al. 2018) gives 128-bit classical security for log q ≤ 54 at N = 2048
    and ≤ 109 at N = 4096 [c, from the standard's table; re-verify]. Interpolating gives log q ≈ 67 at
    dimension 2560, which covers our 56 [e]. **Must be confirmed with lattice-estimator before
    shipping.**
  - For A20 (q ≈ 2^64), use dimension 3072 (k = 12).
- **Group** G = N' = 256 token slots per fresh secret. Choosing N' = 1024 with k = 3 trades a 4×
  larger buffer for 4× less hint streaming per token.

### Setup (once per model pack and customer)

1. The gateway picks a seed σ_A and expands it to a public A ∈ R_q^{n_max × k}, sampled directly in
   NTT form (388 MB for 7B; it can be regenerated).
   - A should be per customer (or per pack), like Kyber's per-key seed. This avoids
     "one global matrix" precomputation and backdoor concerns.
   - One A serves all linears, because each (group, layer, linear) gets its own secret.
2. The service computes the hint **H_lin = W_lin · A[:n_lin] mod q** (m × k ring elements per linear).
   This is a GEMM costing as much as 2,560 tokens of online work, seconds on an H100 [e]. It ships to
   the gateway: 28.5 GB for 7B, stored in NTT form.
3. The gateway **verifies the hint** before first use, with Freivalds over F_q:
   uᵀH ≟ (uᵀW)·A for 2 fresh secret u. This costs about 2·Σn·L·dim ≈ 4·10^9 modmul, minutes on CPU
   [e]. The check involves no client secret, so a cheating service learns nothing from a rejection.
   (A tampered H would otherwise let it mount selective-failure attacks on s.)

### Background (gateway CPU, one batch per group)

For each (layer, linear, group g):

- Sample s_g ∈ R^k and e ∈ R^{n}.
- **Pads**: P_c = <a_c, s_g> + e_c. Coefficient j is token j's pad for input feature c.
- **Corrections**: C_i = <H_i, s_g>. Coefficient j is token j's correction for output i.

Each row costs k pointwise MACs plus one 256-point inverse NTT. Per token that is about 46 ns × rows,
where rows = L·(Σm + Σn) = 2.22 M for 7B, so **0.10 s per token on one scalar core** [m-derived].
AVX-512 IFMA should give another 3–8× [e].

The buffer for one group is 256 × 17.8 MB = 4.6 GB (7B). It can live in RAM or on disk and is
consumed once per slot, exactly like today's mask store but tiny. Only s_g seeds need durable
consumption accounting.

### Online (per linear, per call; no extra round trips)

```
gateway : v_j = Δ·x_j − P[:, j]  mod q        # x_j integer activations, |x| < 2^(A−1)
service : z_j = W·v_j mod q                    # int8 W; 7 int8 digit GEMMs + mod-q fold
gateway : check u·z_j ≡ (uW)·v_j (mod q)       # Freivalds over the field F_q, c = 1
          y_j = round_centered((z_j + C[:, j]) / Δ)   # = W·x_j exactly
```

Correctness: z + C = Δ·Wx − W·e (mod q). With |Δ·Wx| + |We| < q/2 and |We| < Δ/2, the decode is
exact. Analysis gives worst-case |We| ≤ 15σ·127·√n ≈ 2^19.7 at n = 18944. Measured worst was
2^17.6 across 4,096 samples with all-±127 weights [m].

There is **no K-bit ring and no wrap**, so the headroom rule becomes a choice of t. PGW's existing
`int8_ring_matmul` digit machinery computes `W·v` exactly. It needs only ⌈56/8⌉ = 7 digits and a
final mod-q fold (an O(m·B) elementwise step, negligible against the O(m·n·B) GEMM).

**Truncated-z mode.** The service may send only the top t + 2 bits of z. That cuts download to
6.26 MB per token (−10% versus today). The cost is weaker integrity: Freivalds then runs on y over a
gateway prime p, and small low-bit tampering could flip a rounding depending on the secret noise, so
an abort leaks at most 1 bit. Default to full z.

### Security argument (single provider)

- **Service view**: A, H, and every v_j = Δx_j − <a,s_g>_j − e_j. By MLWE with public matrix A and a
  fresh secret per (group, layer, linear), (A, P) is indistinguishable from uniform. So v is
  pseudorandom and independent of x.
  - Many secrets under one A is the standard hybrid argument, as in SimplePIR, Frodo and Kyber.
  - Each secret gets Σn ≈ 30k ring samples. Lattice estimators assume unlimited samples, so this is
    covered.
- **No circuit privacy needed**: the gateway already knows W, H and its own pads.
- **No factory party**: the "independent partner" and the non-collusion assumption disappear. The
  assumption changes from "ChaCha20 PRG + factory ≠ service" to **MLWE** (post-quantum, NIST-standard
  family).
- **Malicious service, integrity.** Freivalds over a prime field is sound to 1/q ≈ 2^-55 per check.
  That is much better than today's 2^-h over Z_2^K (h = 8 at n = 4096 per check), so c = 1 suffices
  and gateway check work drops.
- **Malicious service, selective failure.** Any z ≠ W·v fails with probability 1 − 1/q, and the abort
  event depends only on (u, the service's own deviation). It does not depend on s, e or x, so it
  leaks nothing about customer data.
  - A malformed H is caught at setup.
  - Residual risk: the gateway keeps one Freivalds vector u per pack. Repeated cheating with
    deviations orthogonal to u succeeds with probability 2^-55 per attempt, and the first failure
    trips the existing circuit breaker.
- **Mask-reuse rule still applies.** A pad slot (group, coefficient) must never be used twice,
  otherwise Δ(x1 − x2) leaks. The single-use cursor moves to group seeds.

## 3. Variant A: JIT HE preprocessing (Delphi-shaped, online protocol unchanged)

If the online wire format and the K-bit ring must stay frozen:

- The gateway uploads b = <a,s> + e + Δr (MLWE encryption of today's mask r, along the token axis).
- The service returns W·b mod q.
- The gateway decodes W r mod 2^K = round((W·b − <H,s>)/Δ) and keeps it only for a short JIT buffer.
  - This needs t ≥ 2^K·127·n, so q ≈ 2^(40+15+22) ≈ 2^77 and 10 digits [e]. Alternatively reduce
    mod 2^K with a Δ-scaled message and the q ≡ 1 mod 2^K trick, at q ≈ 2^64.

Delphi's "service keeps s" form (service returns W·x̂ + S) is equivalent, and moves the 6.95 MB per
token buffer to our side.

Costs per 7B token:

- Extra network: +5.8 MB up and +9.7 MB down (more for larger q).
- Extra GPU: 1.4–2× of one online GEMM.
- Gateway CPU: same as PGW-LH.
- Customer storage: buffer only.
- Round trips: one per group (256–4,096 tokens), off the critical path.

It is strictly dominated by PGW-LH, which runs that GEMM once instead of twice. Use it only as a
migration step.

## 4. Cost model per 1M tokens (Qwen2.5-7B; see `cost_model_out.json` for 0.5B and 70B)

| item | today | PGW-LH |
|---|---|---|
| customer storage | 6.95 TB (if kept a day) [e] | 28.5 GB hint + 0.39 GB A + 4.6–9.1 GB buffer [e] |
| customer factory | 5.9 T4-hours ≈ $2.1 at $0.35/h [m rate, e price] | 0 |
| gateway CPU | (online unmask) | +1.0·10^5 core-s ≈ 1.2 cores sustained, scalar [m-derived] |
| hint streaming | — | 111 MB per token from RAM (1.3 GB/s at 1M tokens/day). Use N' = 1024 for 28 MB per token [e] |
| our GPU (H100, 10–30% INT8 utilisation, 1979 TOPS dense [c: NVIDIA H100 datasheet]) | 110–330 s | 154–462 s, so +44–132 s ≈ $0.04–0.11 at $3/h [e]. Scaled from PGW's own "2 ms/token" H100 estimate instead: +800 s ≈ $0.7 [e] |
| network | 4.16 TB up, 6.95 TB down | 5.82 TB up, 9.73 TB down (full z) or 6.26 TB down (truncated) [e] |

Other models [e]:

| | Qwen2.5-0.5B | Llama-3-70B |
|---|---|---|
| hint | 6.2 GB | 138 GB |
| buffer | 1.0 GB | 22 GB (use N' = 1024: 4× less streaming, 4× more buffer) |
| CPU per token | 0.02 s | 0.51 s (about 6 scalar cores for 1M tokens/day) |

## 5. Measurements made

- `proto_lh.py` (numpy, exact limb arithmetic, 2 threads) [m]: exact `y = W·x` on all 256 tokens in
  every case. Freivalds over F_q passes when honest and catches a +1 error in z. Mean of v/q is 0.500.

  | n | m | weights | max noise | budget |
  |---|---|---|---|---|
  | 3584 | 32 | worst-case ±127 | 2^16.6 | 2^21 |
  | 18944 | 16 | worst-case ±127 | 2^17.6 | 2^21 |
  | 3584 | 32 | random | 2^15.7 | 2^21 |

- `client_bench.c` (gcc -O3, one scalar thread, this 2.1 GHz Xeon, shared machine) [m]:

  | k | per row per group | per row per token |
  |---|---|---|
  | 10 | 11.85 µs | 46.3 ns |
  | 3 | 9.41 µs | 36.8 ns |
  | 40 | 27.4 µs | 107 ns |

## 6. Open questions and risks

1. Confirm the (dimension, q, σ, η) choice with lattice-estimator (needs Sage). Consider dimension
   3072 for margin, which adds 20% to hint and CPU.
2. GPU kernel: 7-digit `_int_mm` plus mod-q fold, benchmarked against the 5-digit K40 kernel. The
   1.4× figure is analytic.
3. The hint is 4,096 tokens' worth of today's masks. For very low-volume customers (under about 4k
   tokens per model version) today's factory is cheaper.
4. Wire format and spec changes: ring presets become (t, q, dim). `soundness_bits` moves to field
   Freivalds. The mask store becomes a "pad buffer plus seed cursor". The factory is retired, or
   kept as a fallback.
5. AVX-512 IFMA or a small GPU on the gateway would cut the 0.1–0.5 s per token CPU; not measured.
6. Whether to keep one u per pack or rotate it per epoch (cheap: uᵀW is one more GEMV).
