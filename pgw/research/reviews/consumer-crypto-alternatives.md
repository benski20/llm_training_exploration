# Consumer private chat: cryptographic alternatives to per-layer masking (survey, cost model, recommendation)

Date: 2026-11-27. Question: can a **public consumer chat** reach **20+ tokens/s for models up to 32B**, with
privacy uncompromised (no trusted hardware, no non-collusion assumption, the provider learns nothing about content)
and fidelity uncompromised, using an approach other than pgw's per-layer additive masking? Inputs: the three
consumer studies of today (`consumer-round-trips.md`, `mask-cost-consumer.md`, `consumer-bandwidth.md`), the chat
design note (`encrypted-chat-app.md`), `research/plans/fewer-round-trips.md` section 2 (MPC/GC/TFHE options for one
trip per MLP), and the 2023-2026 literature (section 10).

Cost model: `research/reviews/consumer_crypto_numbers.py` (standard library; run from `pgw/`). It contains no
measurements. Every number is a published anchor (constants `A_*`) or an estimate scaled from the anchors by a rule
stated in the code. This work used no GPU and no external compute.

How figures are marked:
* **[pub]**: from a paper's abstract, text or table, read here;
* **[pub, unverified]**: seen only in a search summary or a secondary source, not checked in the paper;
* **[est]**: derived here (rule in the script);
* **[pgw]**: from this repo's own reviews.

## 0. Verdict

1. **No known approach meets 20 tok/s at 32B for a consumer on the public internet under these constraints, and
   none is likely to before about 2030.** Two independent walls stand in the way:
   * **Information-theoretic schemes** (pgw's class: one-time-padded values, the client evaluates the
     non-linearities):
     * they need ≥ 2 round trips per sequential layer, or L + 1 with parallel blocks;
     * they move ≥ ~3.5 B per value that crosses the wire.
     * At 32B that means ≥ 65-128 trips per pass and ≥ ~22 MB per row. The sustained rate needed is ~4.5 Gbit/s,
       and RTT ≤ 0.4 T ms (T = tokens per pass).
   * **Computational schemes** with one round trip per token (full FHE) have the opposite problem:
     * The bytes are tiny: ~0.1 MB per token.
     * The compute is ~10^5-10^6x plaintext serving: ~170 B200-seconds per token at 32B **[est]**. That is
       ~$100-430 per 1000 tokens and ~3400 B200s to serve one user at 20 tok/s.
     * They have a **latency floor** from the chain of sequential bootstraps: ≥ 0.5-1.9 s per token at 32B on the
       fastest published GPU bootstrap, even with unlimited GPUs **[est]**.
   * Every hybrid surveyed sits on the line between these two walls and is dominated by one endpoint or the
     other (sections 3-5).
2. **One way to get 20 tok/s with this privacy standard exists today: running the model on the user's device.**
   * 7B reaches 20+ tok/s at 4-8 bit on most laptops.
   * 32B reaches ~20-40 tok/s at 4 bit on a 48 GB+ Apple M4 Max or an RTX 4090 **[pub, unverified]**. That costs
     fidelity: 4-bit weights are approximate.
   * Every device that could make pgw's bandwidth-optimal masks (it must hold `W8` and run the model's linear MACs
     per slot) can, in most cases, also run the model itself.
3. **Full FHE is real and improving fast, but it is a research product.**
   * Published state of the art, Llama-3-8B, unmodified weights, CKKS on GPUs [pub]:
     * 12 s per decoded token on 4x B200 (Park et al. 2026);
     * 43 s for a 128-token prefill (Cerium, SOSP'26);
     * 7.5 ms per bootstrap (Cerium).
   * Fidelity is approximate: ~12-13-bit precision. Quality is shown close to FP16 only in plaintext noise
     simulation.
   * The client cannot verify the computation.
   * At consumer prices it is two to three orders of magnitude too expensive even for an "async" mode. 25-100x
     cheaper would make an async 7-8B mode plausible.
4. **Recommendation** (section 9):
   * **Keep pgw as the in-datacentre / enterprise product.** It is exact, verifiable and close to plaintext cost
     there.
   * **For consumers, go local-first:**
     * run the model on the device where the hardware allows, using pgw's own pinned int8 pipeline so outputs
       match the hosted path bit for bit;
     * offer pgw-remote "private slow mode" for small models;
     * place GPUs in the metro where 7B over pgw makes sense.
   * **Track FHE with an explicit go/no-go gate** (≤ $1 per 1000 tokens and ≤ 2 s/token at 8B on public numbers).
     Do not build it yet.
   * **Do not pursue** FSS/2PC non-linears, HE-for-linears in place of masks, client-aided FHE refresh, HE-MLP
     hybrids, or TFHE.

## 1. Comparison table

Single user, decode at short context (~128 private tokens) unless noted. "RTT" is the consumer link's round-trip
time (5 ms = same metro, 20 ms typical, 50 ms cross-continent). Bytes per token count both directions.
Prompt = 500 tokens.

**pgw (one-time-padded values; the client evaluates the non-linears)**

| approach | tok/s 7B | tok/s 32B | round trips / token | bytes / token | bytes / 500-token prompt |
|---|---|---|---|---|---|
| **A. pgw today, consumer internet** (4L+1, pks-v3 masks) | 0.44 @20 ms; ≤0.10 on cable 100/10 [pgw] | 0.19 @20 ms [pgw] | 113 / 257 | ~23 / 73 MB per row [pgw] | ~11 / 36 GB [est from pgw] |
| **B. pgw best sound case, consumer** (parallel blocks L+1, spec T≈3, carry16, device masks) | 1.7-4.4 @20 ms (trip bound); 0.5 on cable, 16 on fibre (byte bound) [pgw] | 0.8-1.8 @20 ms; ≤5 on fibre [pgw] | 29 / 65 per pass | 6.5 / 21.8 MB per row [pgw] | ~3.3 / 11 GB [est] |
| **C. pgw in datacentre** (RTT ~0.2 ms; enterprise, topology B) | ~17 modelled for the SDK; 28 measured co-located (gateway service, H100, Modal) [pgw] | ~6-22 modelled, not measured [pgw] | same | same, inside the DC | same, inside the DC |

**Full FHE (one ciphertext each way per token)**

| approach | tok/s 7B | tok/s 32B | round trips / token | bytes / token | bytes / 500-token prompt |
|---|---|---|---|---|---|
| **D. Full FHE, CKKS, unmodified model, GPUs** (2026 state of the art) | 0.09 (0.06-0.13) on 4 B200; 0.01-0.03 at 1-4k context [est from pub] | 0.023 (0.016-0.033); 0.002-0.012 at 1-4k [est] | 1 (0 if sampling runs under FHE) | ~26-51 KB up, 51-102 KB down with the head on the device, ~3.3 MB with logits; plus 2-18 GB of evaluation keys once [est] | ~12-16 MB [est] |
| **E. Full FHE + model co-design** (polynomial activations, power-softmax, shallower) | 0.2-0.4; floor ≤4.8 [est] | 0.05-0.1; floor ≤2.1 [est] | 1 | as D | as D |

**Hybrids and other cryptography**

| approach | tok/s 7B | tok/s 32B | round trips / token | bytes / token | bytes / 500-token prompt |
|---|---|---|---|---|---|
| **F. FHE with client-aided refresh every k layers** (no server bootstrapping) | ≤0.2 (server HE linears) [est] | ≤0.05 [est] | L/k = 14-28 / 32-64 | 13-344 / 44-786 MB [est] | ≥ GBs [est] |
| **G. HE-MLP hybrid** (attention on the client, the MLP + norm under levelled HE) | ~0.1 [est] | ~0.03 [est] | L = 28 / 64 | ≥28-250 MB [est] | ≥ GBs [est] |
| **H. HE linears + 2PC non-linears** (BOLT, BumbleBee, Nimbus) | ≪0.01 | ≪0.01 | tens or more per layer [est] | tens of GB per inference at BERT scale [pub]; LLM decode not published | not published at LLM scale |
| **I. HE linears + plaintext non-linears on the client** (public weights; Cheetah-style) | ≤ pgw A (same trips) | ≤ pgw A | ≥2L | 27 / 86 MB per row [est] | ~13 / 43 GB [est] |
| **J. FSS 2PC** (SIGMA; Orca-style) | 1/23 s per 128-token pass in a LAN; 5.7 min over a 60 ms WAN [pub] | - | many per layer | 0.1 GB online + 2.3 GB of keys / 0.46 + 9.7 GB [est from pub] | ~1.1 / 4.9 TB of keys [est] |
| **K. TFHE** (programmable bootstrapping per activation) | ≤0.09: the SiLU gates alone take 11 s per H100 [est] | ≤0.03: 37 s [est] | 1 | small (LWE / RLWE) | small |
| **GC: garbled circuits, precomputed** (`fewer-round-trips.md`) | ≪0.01 (bandwidth) | ≪0.01 | 1 per layer, or per pass | 34-850 GB at 7B [pgw est] | - |

**Reference: no provider at all**

| approach | tok/s 7B | tok/s 32B | round trips / token | bytes / token | bytes / 500-token prompt |
|---|---|---|---|---|---|
| **L. On-device inference** (reference; not a provider product) | 11-50 on laptops at 4-8 bit [est]; ~50+ on a 4090 | ~22 on an M4 Max at 4 bit, ~42 on an RTX 4090 at 4 bit [pub, unverified]; 12 on an M4 Max at 8 bit [est] | 0 | 0 | 0 |

Cost, assumptions, results and maturity per row:

| row | provider GPU per token (users per GPU) | client cost | security assumption | fidelity | integrity | maturity |
|---|---|---|---|---|---|---|
| A | online GEMMs on masked int32: a few times plaintext [est]; pks-v3 maskgen 1.2 (32B) / 4.1 (7B) users per H100 at 20 tok/s [pgw] | 1.4 / 4.5 cores per user at 20 tok/s (mask factory) [pgw] | information-theoretic masking online; RLWE (pks-v3) for the masks | **exact** (pinned int8 reference) | **exact Freivalds checks on every trip** | built, tested end to end |
| B | as A; no maskgen with device masks; 3.4-13.5 users per H100 with coalescing [pgw] | the device holds `W8` and runs the model's linear MACs per slot, offline | information-theoretic | exact (new pinned reference, ppl +0.2 % at 0.5B) [pgw] | exact | prototypes |
| C | as A | as A | as A | exact | exact | built |
| D | 43 (30-65) / 171 (120-256) B200-s; 0.15-0.33 / 0.04-0.08 users per B200 at 0.1 tok/s [est]; **$25-108 / $100-430 per 1k tokens** [est] | keygen; ms per token to encrypt and decrypt; optional local LM head (0.5-0.8 GB) | RLWE (post-quantum believed); IND-CPA. IND-CPA^D caveat for CKKS (section 2.5) | approximate: ~12-13 bits after bootstrapping; ppl ≈ FP16 only in plaintext noise simulation at 8B [pub]; unmeasured at 32B | **none** (the client cannot check) | research prototypes (2025-26) |
| E | 2-4x less than D [est] | as D | as D | **a different model**: needs pretraining or distillation; parity not shown at ≥7B (1.4B polynomial LLM) [pub, unverified] | none | research |
| F | ~0.5x D (no bootstrapping; HE linears and polynomials remain) [est] | decrypt + re-encrypt per refresh; ≥10 MB fresh ciphertexts | RLWE | approximate | none | none |
| G | ~0.3-0.5 s per layer of HE [est] | as F | RLWE + masking | approximate | partial (the masked part only) | none |
| H | seconds to minutes at BERT-base [pub] | heavy OT/HE work | standard 2PC (protects the weights too, which pgw does not need) | fixed-point approximations | semi-honest mostly | research |
| I | HE matrix-vector product ~10-100x a plaintext GEMM [est] | HE encrypt / decrypt per trip | RLWE | exact possible | possible with checks | research (Cheetah family) |
| J | 23 s per pass at 7B on an A6000 [pub] | dealer: generate 255 GB of keys per pass [pub] | **a trusted dealer that does not collude with the provider**, or the client as dealer uploading the keys | fixed-point; LAMBADA accuracy ≈ PyTorch [pub] | none (semi-honest) | research (open source) |
| K | ≥11 / 37 H100-s for the gates alone [est] | small | LWE | **≤4-8-bit LUTs**: a quantised model | none | production for small integers; no LLM |
| GC | - | garble ≥34 GB per token | standard 2PC | exact possible | possible | no |
| L | **0** | the device's own GPU / memory | **none needed** (no provider) | exact at bf16/int8; 4-bit is lossy | n/a | production (llama.cpp, MLX) |

Confidence: section 8 lists each key number with its confidence.

## 2. Full FHE inference (item 1)

### 2.1 What has been demonstrated (2024-2026)

| work | setting | result |
|---|---|---|
| NEXUS (NDSS'25) | first non-interactive CKKS transformer; BERT-base | 1103 s CPU, 37.3 s GPU per inference, 164 MB [pub, unverified: search summary of the paper] |
| THOR (2024/2025) | BERT-base, 128 tokens, 1 A100 | 10.4 min [pub, unverified]; Cerium cites 602 s [pub] |
| Powerformer (ACL'25) | HE-friendly BERT-base (power attention, batch-norm) | 5.74 min, -45 % vs THOR "without accuracy loss" [pub, unverified] |
| MOAI (2026) | module-level packing | -52.8 % vs THOR; 224.5 s for 8 input tokens of Llama-3-8B on 1 A100 [pub, unverified] |
| **Cerium** (SOSP'26, Jayashankar et al.) | multi-GPU CKKS compiler/runtime; N = 2^16, logPQ < 1782 | **bootstrapping 7.5 ms**; BERT-base 8 s (8 GPUs) / 66 s (1 A100); **Llama-3-8B 128-token prefill 43 s with 5890 bootstraps**; weights encoded for FHE take **112 TB** for Llama-3-8B (streamed from host); unmodified weights; checked with zero-shot MCQs only [pub] |
| **Park, Cheon, Stehlé et al.** (arXiv 2601.18511v2, 2026; HEaaN2) | Llama-3-8B, CKKS, no fine-tuning (sink tokens + rotations fused into weights); SiLU degree 31, polynomial softmax and rsqrt; 13-bit effective bootstrap | **decode 12 s/token on 4x B200**, 18 s on 8x RTX PRO 6000; prefill of 128 tokens 26 s (B200) / 20 s (RP6000); weights 13 GiB -> 52 GiB packed; per layer at decode 0.98 s on 1 GPU, 0.56 s on 8 (only 1.75x from 8 GPUs); 12 bits of precision match FP16 ppl **in simulation** [pub] |
| DGX Spark forum post (2026; not peer-reviewed) | retrained polynomial-MLP model, undisclosed size | non-interactive ~346-389 s/token; bootstrap 0.69-1.05 s at N = 2^17 on GB10; the interactive variant (client does the non-linears) 1.05 s/token in one process, "tens of MB per token" [pub, unverified] |

The pace is real: 2024-25 BERT-base took ~10 min per inference; 2026 does Llama-3-8B decode at 12 s/token.

### 2.2 Extrapolation to Qwen2.5-7B and 32B (`fhe_decode`)

The per-layer time is split component by component, using Park et al.'s Table VI shares at 128 tokens:
* the projections (qkv, o, gate_up, down) take 419 ms, scaled with the weight count;
* attention on ciphertexts takes 358 ms, scaled with the q width x the context;
* the rest takes 203 ms (norms, activations and the unitemised bootstrap remainder), scaled with the activation
  count d + I.

The central estimate gets a x0.7 / x1.5 band.

| | Llama-3-8B (anchor) | Qwen2.5-7B | Qwen2.5-32B |
|---|---|---|---|
| ratio of work per token | 1 | 0.90 | 3.56 |
| s/token on 4x B200, 128 context | 12 | **10.8 (7.6-16)** | **43 (30-64)** |
| B200-seconds per token | 48 | 43 (30-65) | 171 (120-256) |
| s/token at 1k / 4k context (linear attention growth; upper-ish) | - | 34 / 115 | 119 / 382 |

Chat contexts are 1-8k tokens. Ciphertext attention is already 37 % of a layer at 128 tokens, so real chat is
**3-10x slower** than the headline numbers. Packing improves with more cached tokens, so the 4k column is
pessimistic. Error bars: about x/÷2 at 128 context, x/÷3 at 4k.

### 2.3 The latency floor: sequential bootstraps (`fhe_latency_floor`)

A token's computation passes through every layer in sequence. Each layer needs about 45-55 multiplicative levels:
* qkv, QK^T, AV, o, gate_up, the product and down: ~7;
* softmax: 8-15;
* two rsqrt: ~2 x 8;
* SiLU at degree 31: 5.

A bootstrap at these parameters refreshes ~18 usable levels (Cerium), so an unmodified layer has **≥ 3 bootstraps
on its critical path**. Bootstraps of different ciphertexts run in parallel; bootstraps in series do not. Taking
only that chain, with unlimited GPUs:

| sequential bootstraps per layer | t_boot | 7B max tok/s | 32B max tok/s |
|---|---|---|---|
| 3 (unmodified) | 7.5 ms (best GPU, 2026) | 1.6 | **0.69** |
| 3 | 2.0 ms (hypothetical) | 6.0 | 2.6 |
| 3 | 0.5 ms (hypothetical ASIC class) | 24 | 10 |
| 1 (co-designed model) | 7.5 ms | 4.8 | 2.1 |
| 1 | 0.5 ms | 71 | 31 |

**20 tok/s at 32B needs a full-precision N = 2^16 bootstrap of ≤ ~0.26 ms (unmodified) or ≤ 0.8 ms (co-designed)
on the critical path.** That is 10-30x faster than the best published GPU. Published FHE ASIC designs
(CraterLake-class, Cinnamon) are in the low-ms range per bootstrap [pub, unverified]. Cerium reports matching
CraterLake on GPUs.

Even at that latency, throughput stays at about 120-256 B200-seconds per 32B token: the work does not shrink with
latency.

### 2.4 Cost and users per GPU (`fhe_cost`)

Assumed B200 rental: $3-6 per hour (late 2026).

| | 7B | 32B |
|---|---|---|
| $ per 1000 output tokens (128 context) | **25-108** | **100-430** |
| B200s for one user at 20 tok/s (if parallelism existed) | 600-1300 | 2400-5100 |
| users per B200 at an "async" 0.1 tok/s | 0.15-0.33 | 0.04-0.08 |
| overhead vs plaintext batched serving (5-20k / 1.5-6k tok/s per GPU, assumed) | 1.5x10^5 - 1.3x10^6 | 1.8x10^5 - 1.5x10^6 |

**Batching users does not change this by much.**
* The weights are plaintext, so ciphertext x plaintext products can share their weight reads across users.
* Bootstrapping and key switching use each user's own evaluation keys, so they do not amortise.
* With the projections at 43 % of a layer, sharing them perfectly gives at most 1/(1 - 0.43) = **1.75x**
  aggregate (Amdahl).
* Each active user's evaluation keys (below) must also sit in device memory, which caps concurrency per GPU.

### 2.5 Bytes, keys, fidelity, integrity, security (`fhe_bytes`)

* **Per token:**
  * The device does the embedding lookup and uploads one seeded RLWE ciphertext of the d-vector at a transport
    modulus of ~50 bits: 26 KB at 7B, 51 KB at 32B. The provider ring-embeds it and bootstraps it up.
  * The download is the final hidden state, with the LM head on the device (0.55 / 0.78 GB int8): 51 / 102 KB.
    With the logits instead, it is ~3.3 MB.
  * At 0.1 MB per token and 20 tok/s this is ~16 Mbit/s: **bandwidth is solved**, about 100x below pgw's floor.
  * [est: no paper reports client bytes for LLM decode.]
* **Prompt:** ~12-16 MB for 500 tokens (packed embeddings) [est].
* **One-time evaluation keys:**
  * Bootstrapping plus the rotation keys for BSGS matrix products take 50-200 switching keys at N = 2^16,
    logPQ ≈ 1782, dnum ≈ 3: **~2-18 GB** [est].
  * Cerium: "typically 10-50 GB" for ML workloads [pub].
  * This is a one-time upload per device or key epoch (about 20 minutes to 4 hours on a 10-100 Mbit/s uplink).
    It is then held by the provider: GPU-resident while the user is active.
* **Encrypted KV cache:**
  * It lives on the provider naturally: K and V per token per layer at a 2-limb storage level is ~25x bf16.
  * That is 1.4 MB per token at 7B and 6.6 MB at 32B, or 6 / 27 GB per user at 4k context [est].
  * It is sound under RLWE as long as nothing the provider sees depends on decryptions.
* **Fidelity:**
  * The outputs are **approximate**: CKKS gives ~12-13 bits after bootstrapping, plus polynomial approximations of
    softmax, SiLU and rsqrt.
  * Park et al. show that 12-bit noise matches FP16 perplexity **in plaintext simulation**, not in an encrypted run.
  * Cerium validates with zero-shot MCQs only.
  * By the bar set for this survey ("approximate is acceptable only if quality loss is measured and small"),
    **quality is not yet measured at 7B+ in encrypted runs**, and not at all at 32B.
  * Greedy decoding will diverge from plaintext at some positions; the forum post saw 5 of 6 tokens match.
* **Integrity: none.**
  * A malicious provider can evaluate any other function, for example a cheaper model or a backdoored answer, and
    the client cannot tell.
  * Verifiable FHE (SNARKs over FHE evaluation) adds orders of magnitude.
  * This is a real step down from pgw's exact Freivalds checks.
* **Privacy against a malicious provider:**
  * RLWE IND-CPA protects every query.
  * Two caveats:
    1. **CKKS IND-CPA^D:** if anything that depends on decrypted values reaches the provider, the key can be
       recovered (Li-Micciancio, Eurocrypt'21; Guo et al., USENIX Security'24). Examples: decrypted values
       themselves, aborts, or timing that depends on them. Client-aided designs (F, G) are exposed to this.
       Non-interactive decode is safe only if the client's observable behaviour is independent of plaintexts.
    2. Without integrity, a provider can make the answer depend on the content ("if the prompt mentions X, emit
       garbage") and watch the user's reaction: regenerate, abandon, the length of the next prompt. This is a
       covert channel that pgw's integrity checks close.
* **Leaks** (as in pgw): prompt and answer lengths, and timing, unless padded.

### 2.6 TFHE instead of CKKS (`tfhe_gates`)

* TFHE's programmable bootstrapping evaluates any small LUT exactly. The vendor figure is ~47,900 PBS/s on one H100
  at small precision [pub, unverified].
* The SiLU inputs alone are 530k per token at 7B and 1.77M at 32B: **11 / 37 H100-seconds per token for the gates
  alone**. That is before softmax, the norms, and the linear layers on LWE ciphertexts (n_lwe ≈ 700-1000 times the
  int8 MACs).
* The LUTs take ≤ 4-8-bit inputs, so the model must be re-quantised to that activation width, with unmeasured
  quality at 7B+. Wider inputs need the much slower WoP-PBS.
* Not competitive with CKKS for LLMs.

## 3. HE for linears + interactive non-linears (item 2)

* **BOLT** (S&P'24), **BumbleBee** (NDSS'25) and **Nimbus** (NeurIPS'24) solve a different problem: they also hide
  the **model weights from the client**. So every non-linearity (GELU, softmax, LayerNorm) becomes a 2PC protocol
  (OT/HE comparisons, truncations, conversions).
* Published costs are at BERT scale:
  * BOLT ~25-60 GB per BERT-base inference [pub, unverified: quoted by follow-ups];
  * BumbleBee 15.5 GB for BERT-large [pub, unverified], with >80 % of the traffic in truncations and conversions;
  * Nimbus cuts 60 % of the traffic [pub, unverified];
  * all of it is minutes per inference over a WAN.
* These protocols use **many rounds per layer** (comparison trees, exponentiation). None reports LLM decode.
* **With open weights (pgw's case), the 2PC non-linears are pure overhead:** the client evaluates them in the clear,
  exactly, for free. What remains is row I: HE for the linears with plaintext non-linears on the client
  (Cheetah-style coefficient encoding).
* `he_linear_bytes` compares row I with pgw:

  | per decode row | Qwen2.5-7B | Qwen2.5-32B |
  |---|---|---|
  | HE linears: ~13.6 B per input value up (seeded, q ≈ 2^109, N = 4096), ~10 B per output down [est] | 26.8 MB | 85.6 MB |
  | pgw online (4 B per value) [pgw] | 9.5 MB | 30.3 MB |
  | pgw online + pks-v3 masks [pgw] | 22.3 MB | 71.2 MB |

  * It moves the same values per layer, in the same ≥ 2 trips per layer.
  * It costs 1.2x pgw's total and 2.8x pgw's online bytes.
  * The provider does HE matrix-vector work (~10-100x a plaintext GEMM [est]) in place of int GEMMs on masked data.
  * `mask-cost-consumer.md` reached the same verdict for "HE online": the reply keys and the hint do not amortise
    over the rows of one trip.
  * **Dominated by pgw on every axis.**

pgw's masked linear with plaintext non-linears on the client is already the round-optimal two-party protocol with
preprocessing for public weights (`fewer-round-trips.md` section 2). The masks are the preprocessing.

## 4. 2PC with preprocessing: PCG / silent OT / VOLE, and FSS (item 3)

### 4.1 FSS

* **SIGMA** (PoPETs'24), measured, one Llama2-7B forward pass over 128 tokens on A6000 GPUs [pub]:
  * 23.1 s and 12.1 GB online in a LAN;
  * 5.7 min over a 305 Mbit/s, 60 ms WAN;
  * **255 GB of FSS keys** from the dealer (217 s just to transfer them);
  * 13B: 37.6 s, 18.9 GB online, 419 GB of keys.
* Scaled to decode with a KV cache (÷128, by params; optimistic): **0.1 GB online + 2.3 GB of keys per token at 7B,
  0.46 + 9.7 GB at 32B** [est]. A 2026 ePrint quotes "~2 GB of key material per token" for SIGMA at 7B
  [pub, unverified].
* **Security:** SIGMA's model is 2PC (client, model owner) plus a **trusted dealer** that makes the FSS keys and must
  not collude with the provider. That is a non-collusion assumption, excluded here.
* With the client as the dealer, the client uploads the provider's key share: 2-10 GB per token.
* Rounds: several per non-linearity (SIGMA's GeLU is 5 rounds; softmax and max are more), so tens per layer.
* **Dead for consumers on every axis.**
* Orca and Pika are earlier FSS work at CNN scale.

### 4.2 PCG / silent OT / VOLE

* Pseudorandom correlation generators make correlated randomness with sublinear communication.
* In pgw the correlation is `(r, W8 r)` for a public matrix `W8` and a client-secret `r`.
  * The `r` side can come from a short seed (LPN-style expansion), so the **mask upload** of pks-v3 (7 B per input)
    could in principle become sublinear.
  * The `W8 r` side must still reach the client: ≥ ~4 B per output value. This is `mask-cost-consumer.md` section 3's
    floor, and it applies to any provider-made mask.
* The gain is bounded by the upload share of the mask stream (~40 % of pks-v3's bytes). It is **zero for the online
  trips**, which are the 32B bottleneck.
* No PCG for "matrix-vector correlations with a public matrix" exists at these sizes. It would need new
  cryptography and review.
* **Not worth a workstream** for the consumer target. It is a possible long-term refinement of pks-v3 (input to the
  external review packet).
* Silent OT/VOLE for the non-linears themselves is moot: the client evaluates them in the clear.

### 4.3 Garbled circuits

`fewer-round-trips.md` section 2: 1 trip per layer or per pass, but 34-850 GB per token at 7B. Dead.

## 5. Model-side co-design (item 4)

| lever | effect on FHE (D -> E) | effect on pgw | fidelity |
|---|---|---|---|
| low-degree polynomial activations (degree 2-4 instead of SiLU at degree 31) | -5 levels per layer | none: a product of masked values needs a round (Beaver) or the client | needs pretraining or distillation; not shown at 7B+ |
| power-softmax / ReLU attention (Power-Softmax, AISTATS'26: polynomial LLMs to 1.4B, 32 layers [pub, unverified]) | -8 to -15 levels; no max subtraction | none | as above |
| norm-free or fixed-scale norms | -2 x ~8 levels (rsqrt) | none: the norm is already folded for free (fused2) | retraining |
| together: ~1 bootstrap per layer instead of ≥ 3 | **2-4x throughput, latency floor 3x lower** (Powerformer -45 %, MOAI -53 % on BERT [pub, unverified]) | none | a different model; parity unmeasured at ≥7B |
| parallel attention + MLP blocks | ~1.3x (shorter critical path) | **2L -> L + 1 trips** [pgw] | needs a parallel-block checkpoint (none for Qwen) |
| fewer, wider layers (e.g. 32B at 32 layers) | 2x lower latency floor | 2x fewer trips | pretraining; depth-for-width trades usually lose some quality [unverified] |
| MoE | FHE must evaluate every expert (routing is secret): cost ~ total params | routing leaks to the provider unless every expert runs | - |
| narrower MLP crossings (smaller I) | fewer bootstraps | fewer bytes per row | pretraining |

* **With co-design, FHE improves 2-4x in cost and 3x in latency floor.** Row E: 7B ~0.2-0.4 tok/s, 32B ~0.05-0.1;
  floors 4.8 / 2.1 tok/s at today's bootstrap.
* This is not the 10^3-10^5 that 20 tok/s needs.
* It also means training and shipping a **different model** (open-weight frontier checkpoints will not be
  FHE-friendly), with quality parity not shown at 7B+.
* For pgw, only the architectural levers matter: parallel blocks, depth, MLP width. Polynomial activations do not
  help pgw at all.

## 6. Hybrids specific to pgw (item 5)

1. **First and last layers on the device.**
   * Layer 0's qkv is already local in the folded schedule; the LM head is local in the bandwidth review's best case.
   * Moving more of the first and last layers saves ≤ 2-4 of 2L trips (≤ 3 % at 32B).
   * Bytes: the local head saves 2.4 % (32B) to 7.5 % (7B) of a carry16 row [pgw].
   * Useful polish, not a lever.
2. **FHE-evaluate a block of k layers per round trip.**
   * With server bootstrapping, this is row D with k = L.
   * Without it (client-aided refresh, row F), each refresh needs a fresh top-level ciphertext. That ciphertext is
     uniform mod Q, so ≥ N log Q bits (8-12 MB at N = 2^16, logQ = 1000-1500) **from the key holder**.
   * One decode token's residual uses d/(N/2) of the slots, so even with optimistic slot amortisation it costs
     **13-38 MB per token at 7B and 44-131 MB at 32B**, over 14-64 trips (k = 1-2: a layer needs 45-55 levels, and
     ~35-40 are available at N = 2^16 without bootstrap levels). Without amortisation the per-token figure is
     115-786 MB.
   * The provider still does all the HE linear algebra (~0.5x of D).
   * **Worse than pgw's bytes (6.5 / 21.8 MB per row) with similar trip counts and 10^4x the provider compute.**
     Rejected.
   * A variant where the client uploads `Enc_top(s)` masks offline and the online step is masked plaintext has the
     same offline bytes. Bootstrapping is the only way to get top-level ciphertexts without the key holder sending
     N log Q bits.
3. **HE-MLP hybrid** (row G).
   * Attention runs on the client (it holds the KV). The provider runs o -> residual -> norm -> gate_up -> SiLU
     (polynomial) -> down -> residual -> next norm -> qkv under levelled HE (depth ~25, N = 2^16, no bootstrap).
   * That gives L trips for a *sequential* model (28 / 64).
   * Costs:
     * the residual must be refreshed each layer (the same fresh-ciphertext bytes as F: ≥ 28-250 MB/token at 7B);
     * ~0.3-0.5 s of HE per layer, from Park et al.'s projection timings, so ~10 s/token at 7B;
     * approximate SiLU;
     * the IND-CPA^D exposure of section 2.5.
   * It gains less in trips than parallel blocks (L + 1) do for free. Rejected.
4. **Encrypted KV cache on the provider: sound, but what it buys depends on the scheme.**
   * **Masking (pgw).**
     * The client can store its KV on the provider as an AEAD blob (AES-256-GCM). Bind the key to the user, the
       conversation id and a turn counter or hash the client keeps, so the provider cannot roll back or swap a
       conversation.
     * This is sound: IND-CPA plus integrity. It leaks only the blob's size, which the row count already reveals.
     * The provider **cannot compute with it**: attention is bilinear and softmax is not affine. Computing `q K^T`
       on masked `K + S` costs the client a correction as large as the product itself (section 5 of
       `consumer-round-trips.md` generalises).
     * So the client downloads the blob per turn: 57 KB per cached token at 7B, 262 KB at 32B (bf16).
     * That is **~100-400x cheaper than re-prefilling the conversation under masking** (6.5-23 / 22-73 MB per
       token). It is the right design for stateless or multi-device clients.
     * The local prefix cache (`prefix_cache=True`) is better still (0 bytes), and the provider can tell a hit in
       both designs.
     * Recommended as a small, cheap, sound feature; it does not touch the 20 tok/s wall.
   * **FHE:** the KV is a ciphertext on the provider by construction. That is sound, at ~25x bf16: 6-27 GB per user
     at 4k context [est].
5. **Batching many users' FHE work:** ≤ 1.75x (section 2.4). Not a lever.
6. **Local draft + remote verification.**
   * The device proposes T tokens; the provider verifies them in one pass.
   * pgw already does this (strict speculation, T ≈ 1.7-3).
   * Under FHE the T tokens share ciphertext slots, so a pass costs about the same and tok/s scales ~T. This is the
     cheapest FHE multiplier, but the ceiling is the acceptance rate (T ≈ 2-4).
7. **Weights streamed from SSD for batched device masks (a real pgw niche).**
   * Device-made masks need `W8` and the model's linear MACs per slot, but **batched and offline**.
   * A 16 GB laptop cannot run 32B int8 interactively from SSD (~0.1-0.2 tok/s).
   * It *can* stream 32 GB from a 3-7 GB/s SSD every 5-10 s and make a 256-slot batch per pass: ~25-50 slots/s,
     enough mask stock for 20 tok/s [est].
   * It does not fix the online trips (RTT, ≥ 22 MB per row), so it helps only in the metro or data centre.

## 7. What is impossible, and the product shapes that are achievable (item 6)

### 7.1 Impossible (proved or practically binding)

1. **Information-theoretic privacy (pgw's class) at 20 tok/s for 32B over consumer RTT.**
   * The provider can carry a one-time-padded value only through affine maps, so a sequential layer needs ≥ 2
     client steps (`consumer-round-trips.md` section 5).
   * 20 tok/s with T tokens per pass needs RTT ≤ 50 T / (2L) ms minus work: **0.39 T ms at 32B** (0.77 T with
     parallel blocks).
   * At RTT ≥ 5 ms that needs T ≥ 7-13 accepted tokens per pass, which no strict-speculation draft reaches.
   * The bandwidth floor at the best sound case is **~4.5 Gbit/s at 32B and ~1.4 Gbit/s at 7B** for 20 tok/s
     (`consumer-bandwidth.md` Theorems 1-2): above any home uplink.
   * This is a proof within the model (uniform masks, per-coordinate decoding, exact checks), not an engineering gap.
2. **Full FHE at 20 tok/s for 32B before ~2030, on any hardware roadmap seen.**
   * Latency floor: ≥ 64 layers x ≥ 1-3 sequential bootstraps x t_boot. That needs t_boot ≤ 0.26-0.8 ms at full
     precision: 10-30x past the best GPU.
   * Throughput: 120-256 B200-s per token means ~3400 B200s for one user at 20 tok/s. Even a 100x cost improvement
     leaves $1-4 per 1000 tokens at 32B against ~$0.0005 for plaintext.
   * This is a practical bound, not an information-theoretic one.
3. **Every hybrid in between is dominated.** Moving work across the line trades rounds for bytes or compute:
   * **F, G:** a fresh ciphertext per refresh (≥ N log Q bits);
   * **H, J:** 2PC non-linears, GBs per token;
   * **I:** HE ciphertexts carry more bits per value than masks.
4. **Exactness under FHE.** CKKS is approximate by design. Exact BFV/BGV/TFHE pipelines need integer LUTs or
   polynomial approximations that themselves define a different model.
5. **Integrity under FHE** at a cost within 100x of FHE itself: no known construction.

### 7.2 Achievable product shapes

| shape | model sizes | speed | privacy | fidelity / integrity | when |
|---|---|---|---|---|---|
| **Enterprise / in-cloud pgw** (gateway next to the GPUs, topology B; or a metro link to ≤ 1-2 ms) | 7B-72B | 7B 17-28 tok/s; 32B ~6-22 (modelled; RunPod sessions pending) [pgw] | pgw (masking; provider sees nothing) | exact + Freivalds | **now** |
| **Consumer local-first** (pgw's pinned int8 pipeline run on the device; outputs bit-identical to hosted pgw) | 0.5B-7B on laptops; 32B on 48 GB+ / 500 GB/s+ machines | 20-50+ tok/s (7B); ~10-20 (32B at 8 bit on an M4 Max class), ~20-40 at 4 bit (lossy) | **maximal**: no provider | exact at int8 | **now** (engineering only) |
| **Consumer pgw "private slow mode"** (no local GPU or weights) | 0.5B-3B | 0.5-4 tok/s at 20 ms (fused, speculation); 1-8 tok/s at 5 ms [pgw] | pgw | exact + checks | now (fused schedules opt-in; lane format 3 next) |
| **Consumer pgw, metro GPUs** (RTT ≤ 5 ms) + parallel-block model + speculation | 7B | ~5-12 tok/s on fibre; byte-bound on cable [pgw] | pgw | exact + checks | 6-12 months (needs a parallel-block checkpoint, K56 masks in the SDK) |
| **Consumer "async private deep answer" via FHE** | 7-8B (co-designed later) | 0.01-0.1 tok/s; minutes to hours per answer | RLWE, no integrity | approximate (unmeasured) | research; gated on ≥ 25-100x lower cost |
| Consumer large models (32B) interactive and private, no trust | - | - | - | - | **not achievable** with known techniques |

## 8. Confidence of the key numbers

| number | value | confidence |
|---|---|---|
| FHE Llama-3-8B decode, 4x B200 | 12 s/token | **high** [pub; read in the paper's text] |
| Cerium bootstrap / Llama-3-8B prefill | 7.5 ms / 43 s, 5890 bootstraps | **high** [pub; read in the paper's text] |
| FHE Qwen2.5-7B / 32B decode | 10.8 (7.6-16) / 43 (30-64) s/token | **medium**: scaled component by component from one anchor; multi-GPU scaling and memory (32B packed weights ~200 GiB) uncertain; x/÷2 |
| FHE at 1-4k context | 3-10x slower | **low-medium**: linear attention scaling; packing may halve it |
| FHE $ per 1k tokens | 7B $25-108; 32B $100-430 | **medium**: GPU price assumed ($3-6 per B200-hour) |
| FHE latency floor at 32B | ≥ 0.5 (co-designed) - 1.9 s/token at 7.5 ms per bootstrap | **medium**: depth per layer and ~18 levels per bootstrap from published parameters; the count of sequential bootstraps is my estimate |
| FHE bytes per token | ~0.1 MB (0.08-3.4) | **medium**: first-principles; no paper reports it; transport-modulus and ring-embedding choices vary |
| FHE evaluation keys | 2-18 GB once (Cerium: 10-50 GB typical) | **low-medium** |
| FHE quality | ≈ FP16 at 12 bits, in simulation | **low** for 7B/32B in encrypted runs (not measured) |
| batching gain for FHE | ≤ 1.75x | **medium** (Amdahl on Table VI shares) |
| SIGMA Llama2-7B | 23 s, 12 GB online, 255 GB keys per 128-token pass | **high** [pub; read in the paper's table] |
| SIGMA per decode token | 0.1 GB + 2.3 GB of keys (7B) | **low-medium** (÷128 scaling) |
| HE-linear row I bytes | 27 / 86 MB per row | **medium** (bytes per value assumed: 13.6 up / 10 down) |
| client-aided refresh bytes | 13-131 MB per token (amortised) | **medium** (fresh-ciphertext size is a hard floor; slot amortisation optimistic) |
| TFHE SiLU-only bound | 11 / 37 H100-s per token | **medium** (vendor PBS rate) |
| pgw consumer numbers | as in the three reviews | as stated there (models + CPU measurements; H100 runs pending) |
| local 32B tok/s | M4 Max ~22, RTX 4090 ~42 at Q4 | **low-medium** [pub, unverified: benchmark blogs]; the memory-bound estimate agrees (23 / 43) |
| NEXUS / THOR / Powerformer / MOAI / BOLT / BumbleBee / Nimbus / Power-Softmax figures | see sections 2-5 | **low-medium** [pub, unverified: search summaries] |

## 9. Recommendation and staged roadmap

**Stage 0 (now): positioning.**
* State plainly in the product and the docs that private 20 tok/s at 32B for consumers on the internet is
  unreachable with known cryptography (section 7.1). pgw is the private-inference product for **in-cloud and
  enterprise** use, where it is exact, verifiable and near plaintext cost.
* Keep the fused / parallel / narrow-wire work: it lowers costs for everyone, and it is what consumer metro
  deployment needs.

**Stage 1 (0-3 months): consumer local-first.**
* Give PGW Chat a **local engine** that runs the same pinned int8 pipeline on the device's GPU (Metal / CUDA). The
  client already holds the factory pack in some modes, and the exact int8 kernels exist.
  * Device-class detection picks local (7B everywhere it fits; 32B on 48 GB+ unified memory) or remote.
  * A local answer is bit-identical to the hosted pgw answer, so "fidelity" is one claim across both.
* Remote pgw stays as "private slow mode" for devices that cannot run the model (0.5B-3B), with the fused schedules
  and lane format 3.
* Add the **AEAD KV blob** on the provider for stateless or multi-device clients (section 6.4). It is cheap, sound,
  and 100-400x cheaper than re-prefill.

**Stage 2 (3-12 months): metro pgw for consumers.**
* Regional GPUs at ≤ 5 ms RTT.
* A parallel-block 7B checkpoint (the model team), K56 masks in the SDK, speculation on fused schedules, and
  device-made masks for weight-holding clients, including the SSD-streamed batched maskgen niche.
* Target: 7B at ~5-12 tok/s on fibre, exact and verified.
* Measure it on the RunPod sessions already planned (sessions 6-7).

**Stage 3 (research track, 12-36 months; no product commitment): FHE "async private" mode.**
* Track CKKS GPU systems (Cerium is open source; HEaaN). Budget one internal reproduction of the Llama-3-8B decode
  number on rented B200s (≤ $200) before any further investment.
* **Go/no-go gate** for building an async FHE mode, on public or reproduced numbers:
  * ≤ $1 per 1000 tokens and ≤ 2 s/token at 8B, at 2k context;
  * quality measured in encrypted runs (ppl and greedy agreement on ≥ 1000 tokens);
  * a credible answer to the missing integrity (or an explicit product decision to accept its absence for this
    mode).
* Today's position against the gate is 25-108x off on cost and 5-60x off on latency.
* Co-design (an FHE-friendly 7B) is worth a feasibility study only if the gate trend holds: it is a pretraining
  project.

**Do not pursue:**
* FSS or 2PC for the non-linears (row J; non-collusion or GBs per token);
* HE for the linears in place of masks (row I; dominated);
* client-aided FHE refresh (row F);
* the HE-MLP hybrid (row G);
* TFHE for LLMs (row K);
* garbled circuits;
* trusted hardware (excluded by the brief, and rejected in `encrypted-chat-app.md`).

## 10. Citations

Read here [pub]:
* Park, Park, Park, Ahn, Cheon, Hanrot, Kim, Park, Stehlé. *Scaling up FHE-based Privacy-Preserving ML: Higher
  Throughput, Longer Inputs for Llama-3-8B.* arXiv 2601.18511v2, 2026. https://arxiv.org/abs/2601.18511
* Jayashankar, Kim, Sullivan, Zheng, Skarlatos. *Cerium: A Multi-GPU Framework for Terabyte-Scale Encrypted
  Inference.* SOSP'26 (arXiv 2512.11269). https://arxiv.org/abs/2512.11269
* Gupta, Jawalkar, Mukherjee, Chandran, Gupta, Panwar, Sharma. *SIGMA: Secure GPT Inference with Function Secret
  Sharing.* PoPETs 2024. https://eprint.iacr.org/2023/1269 (Tables 5, 7, 9 read in the PoPETs PDF)

Seen only in search summaries [pub, unverified]:
* NEXUS (Zhang et al., NDSS'25), https://eprint.iacr.org/2024/136
* THOR (Moon et al.), https://eprint.iacr.org/2024/1881
* Powerformer (Park et al., ACL'25)
* MOAI (arXiv 2611.13641)
* EncFormer (arXiv 2604.09975)
* Power-Softmax (Zimerman et al., AISTATS'26), https://arxiv.org/abs/2410.09457
* BOLT (Pang et al., S&P'24), https://eprint.iacr.org/2023/1893
* BumbleBee (Lu et al., NDSS'25), https://eprint.iacr.org/2023/1678
* Nimbus (Li et al., NeurIPS'24), https://arxiv.org/abs/2411.15707
* *Breaking the Layer Barrier* (arXiv 2508.19525)
* Cinnamon (ASPLOS'25), https://www.pdl.cmu.edu/PDL-FTP/BigLearning/cinnamon_abs.shtml
* Bifrost (TEE + FHE, arXiv 2606.17421; excluded: trusted hardware)
* Zama TFHE-rs GPU benchmarks, https://docs.zama.ai/tfhe-rs/get-started/summary/gpu_benchmarks, and blog
  (~47.9k PBS/s per H100)
* DGX Spark forum post "LLM inference under CKKS FHE on one DGX Spark" (NVIDIA Developer Forums, 2026; not
  peer-reviewed)
* Local inference benchmarks: https://kunalganglani.com/llm-benchmarks (Qwen2.5-32B Q4_K_M: ~42 tok/s on an RTX
  4090, ~22 on an M4 Max)

From memory, not re-checked here:
* Li and Micciancio, *On the Security of Homomorphic Encryption on Approximate Numbers*, Eurocrypt 2021;
* Guo et al., *Key Recovery Attacks on Approximate HE with Non-Worst-Case Noise Flooding Countermeasures*, USENIX
  Security 2024;
* Huang et al., *Cheetah*, USENIX Security 2022;
* Boyle et al., *FSS for Mixed-Mode and Fixed-Point Secure Computation*, Eurocrypt 2021;
* Boyle et al., PCG / silent OT (CCS'19);
* CraterLake (ISCA'22).

Internal:
* `research/reviews/consumer-round-trips.md`;
* `research/reviews/mask-cost-consumer.md`;
* `research/reviews/consumer-bandwidth.md`;
* `research/reviews/encrypted-chat-app.md`;
* `research/plans/fewer-round-trips.md`.
