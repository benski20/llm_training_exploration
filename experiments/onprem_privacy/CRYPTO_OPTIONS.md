# Cryptographic options for "we host, customer installs no box" (literature check, Nov 2026)

## The structure that works
The vendor's server does only the **linear** operations (matrix-vector products: about 99% of the compute) on
cryptographically masked activations. The client unmasks the result and does the **non-linear** steps itself
(norms, activations, softmax: under 1% of the compute). Each linear step takes one round trip.

The cheap encodings we tested and broke (t6, t7, rotation, split inference) were deterministic and
non-interactive. These protocols use fresh randomness for every query and one interaction per step.

| Protocol | Servers | Privacy basis | Reported performance | Catch |
|---|---|---|---|---|
| **Maverick** (arXiv 2609.10264, ePrint 2026/1955) | **1 untrusted server** | Dual-LPN pseudorandom masking (computational, post-quantum style assumption). Also verifiable. | Qwen3-4B, 8-token prompt: 518-958 ms end to end (network time excluded), against 14.1 s local on 2 client threads (15-27x). 182 rounds and 58 MB of traffic for 8 tokens. | Model weights are public (open-weight). Client-side preprocessing storage is the stated bottleneck. Every linear step needs a round trip, so at 50 ms RTT that adds about 1 s per token. Evaluated with a CPU server only. |
| **Gupta, Katz, Miers** (ePrint 2026/1849) | 2 non-colluding servers (malicious-secure) | Secret sharing across the two servers. Also verifiable. | 11-14x faster than SIGMA. Llama-2-70B needs only 179 MB on the client, against 140 GB to run it locally. No server overhead beyond normal inference. | Two operators must not collude. Open-weight models. |
| PermLLM (arXiv 2405.18744) | 2 parties + 1 semi-honest dealer | Secret sharing, with non-linear steps run on randomly permuted plaintext | ChatGLM-6B in about 3 s/token (WAN, 10 ms RTT, 1 Gbps) | Non-standard security assumption (permuted values leak) |
| Centaur (ACL 2025) | 2-party MPC + permutation | Hybrid | 5-30x faster than SMPC baselines | Privacy shown empirically, not proven |
| PUMA (MPC) | 3 servers | Secret sharing | About 200 s/token, LLaMA-7B | Slow |
| FHE (arXiv 2601.18511) | 1 server | Lattice encryption | 18-22 s/token, Llama-3-8B, 8 GPUs | Slow |

## Implications
- This corrects our earlier claim that "only FHE/MPC, both slow" exist. Client-aided linear delegation (Maverick) gives
  cryptographic input privacy against a **single** server at close to plaintext server cost. The costs are a client
  that does about 1% of the compute plus storage, and a latency floor of rounds x RTT.
- The model must be open-weight (the client derives its unmasking material from the weights). That suits a business
  that sells hosting, tuning and accreditation rather than secret weights.
- The latency floor depends on RTT. With servers in the same region (about 10 ms) it is roughly 0.2 s/token; at 50 ms it is about 1 s/token.
- Open questions to validate: client storage at 8B-70B scale, GPU server throughput, handling of attention/KV cache, and
  maturity (Maverick is a Sept 2026 preprint; no peer review or independent replication).

## Our replication: single-server LPN masking on Qwen2.5-0.5B (Kaggle CPU, `kaggle/onprem_lpn/`)
Every linear layer was delegated with dual-LPN masking: t = 160 nonzeros, code rate 1/2, field 2^61-1, int8 weights, 20-bit activations.
- **Correctness:** tokens identical to the plaintext int8 model, with a maximum logit difference of **0.0** (the arithmetic is exact).
  Per-row int8 weight quantization on its own changed the answer compared with fp32 after 1 token. That is a quantization issue, not
  a protocol issue; any integer-weight quantization scheme can be used.
- **Security sanity check:** an inverter trained on exactly what the server sees scored **3.3% / 2.7%** (layers 1 / 12), against a
  majority-class baseline of 3.4%. The unmasked control scored 78% / 64%. The top 8 bits are uniform (chi-square 256 / 268 with df = 255).
- **Costs per decoded token (0.5B):** 97 rounds; 1.5 MB up and 3.6 MB down; server 0.85 s (numpy, CPU); client 10.3 s (unoptimised numpy).
  The client reads 817 MB of tables per token, against 247 MB to run a 4-bit model locally. Client precompute took 9 min and needs **7.9 GB of storage (16x the int8 model)**.
- **Scaling (analytic, same parameters):**
  - 8B: 120 GB client storage; the client reads 3.0 GB per token, against 3.75 GB for local int4; 129 rounds; 19 MB per token.
  - 70B: 1.1 TB storage; 14 GB read per token, against 35 GB for local int4; 321 rounds; 89 MB per token.
- **Verdict:** the cryptography works and is exact. The single-server variant is a poor deal for the client: storage is
  16 bytes per parameter, and per-token client work is close to running the model locally until about 70B. Prefill costs the
  same per prompt token, with no batching benefit. A smaller field and smaller t help (perhaps 4x), but they don't change the conclusion.
  The **two-server** variant (additive shares, as in Gupta et al.) has trivial client compute and no storage, but it needs two non-colluding operators.

## Linear-only HE measured (DESILO CUDA CKKS, Kaggle T4 and 4-core CPU), 2026-11-16
Protocol: customer encrypts each activation vector, server multiplies by public weights under CKKS (depth 1, no bootstrapping),
customer decrypts and does norms/attention/SiLU and the LM head in plaintext. Results: `results/kaggle/onprem-linear-he-p100` (ran on a T4),
`results/kaggle/onprem-linear-he-cpu`.

| per 4096x4096 weight tile | 4-core CPU | T4 GPU |
|---|---|---|
| multiply, weights pre-encoded | 0.75 s | 0.27 s |
| encrypt / decrypt (customer) | 6 ms / 8 ms | 2 ms / 1.6 ms |
| relative error | 1.5e-8 | 2.0e-8 |

| per generated token | Llama-3-8B | Llama-3-70B |
|---|---|---|
| server, T4, cached weights | 120 s | 1,114 s |
| server, 4-core CPU | 336 s | 3,118 s |
| customer crypto (4-core CPU) | 3.8 s | 18 s |
| upload + download | 88 + 92 MB | 409 + 441 MB |
| round trips | 128 | 320 |
| customer storage | ~1 GB (embeddings + LM head) | ~2 GB |

End to end on Qwen2.5-0.5B (T4): 8 generated tokens identical to plaintext, max logit diff 2.8e-5 (max logit 20.8);
123 s/token decode (tiles re-encoded per call; padded 1024-slot tiles).

Verdict: customer load is solved (~1 GB, seconds of CPU), but the GPU only gave 3-6x over 4 CPU cores on this library, leaving the
server at ~2 minutes and ~180 MB per 8B token: roughly 10,000x plaintext cost. Not commercially viable for generation today.

## One-time-mask masking with public weights, measured (Kaggle T4 + 4-core CPU), 2026-11-16
Offline: r uniform in Z_2^48, store W r (r from a seed). Online: send x+r, server returns W(x+r) via 6 int8 limb GEMMs, customer subtracts W r.
Results: `results/kaggle/onprem-otp-masks`. Explainer page: `experiments/onprem_privacy/masked_inference.html`.

- 0.5B end to end: identical tokens, logit diff 0.0 vs quantized plaintext; same tokens as the float model.
  Server view uniform (top-byte chi2 241, 255 dof; corr with plaintext -0.0006, 141k values). Customer online crypto 0.18 s for 23 tokens.
- 8B per token: server 0.45 s batch-1 / 0.014 s amortized at batch 32 on T4 (plain fp16 batch-1: 0.059 s); 5.1 MB up + 8.3 MB down;
  8.3 MB of masks (121k tokens/TB); 128 rounds; mask generation 14.4 s on 4 CPU cores (torch int8), ~0.014 s on a T4.
- 70B per token: server 3.1 s batch-1 / 0.093 s amortized; 26 + 40 MB; 40 MB of masks (25k tokens/TB); 320 rounds; mask gen ~141 s on 4 cores (est).
Verdict: exact and information-theoretically private, server cost near normal serving. The load moved to mask generation (needs a GPU)
and bandwidth/mask volume. Most practical form: an independent mask provider on its own GPUs (non-collusion assumption).

### Mask-generation cost (Kaggle T4 + 4 vCPU Xeon 2.0 GHz, avx512f, no VNNI), `results/kaggle/onprem-otp-cost`
- T4, 8B (weights resident): ring 2^48 0.0066 s/token (550k tok/h, $0.36 spot / $0.96 on-demand per M tokens); ring 2^32 0.0043 s ($0.24 / $0.62).
- T4, 70B (weights streamed from host RAM per batch, batch >= 1024): ring 2^48 0.047 s/token (77k tok/h, $2.60 / $6.85 per M); ring 2^32 0.032 s ($1.78 / $4.68).
- CPU: fbgemm int8 0.30 TOPS effective (torch._int_mm CPU only 0.005 TOPS, which explains the earlier 14 s/token). => 8B 0.28 s/token, 70B 2.8 s/token (ring 2^48, GEMM only).
  Laptop NPU (40 TOPS at 30-60% util) estimate: 8B 3-7 ms/token, 70B 34-69 ms/token. Untested.
- Quality (Qwen2.5-0.5B, 2048 wikitext tokens, int8 weights): A16/A20 ppl 10.92, top-1 agreement with float 97.3%; A12 10.93 / 97.4%;
  naive per-token A8 11.09 / 92.6%. So the 32-bit ring (needs A8) costs quality unless SmoothQuant-style fixes work; a 40-bit ring with A12 is safe.
