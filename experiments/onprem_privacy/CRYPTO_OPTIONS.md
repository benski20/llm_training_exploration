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
