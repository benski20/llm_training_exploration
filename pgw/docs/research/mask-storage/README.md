# Reducing customer mask storage: research summary (2026-11-18)

Five research tracks, each followed by an adversarial review that tried to refute its findings.
Raw outputs are in the track folders and in `agent_results.json`. Today's baseline for Qwen2.5-7B is
6.95 MB of masks per token (K40), which is 7 TB/day at 1M tokens/day.

## Key structural finding

Any scheme in which the **customer computes W·r itself**, whether a just-in-time CPU factory,
structured or LPN masks, or partial local compute, costs about as much CPU as running the model's
linear layers locally. That removes the reason to outsource. Without a second party, the only way
out is for **the provider to compute on encrypted masks**. That is the homomorphic-preprocessing
track, and it needs no second compute provider.

## Options that survived review

| Option | Customer storage (7B) | Trust model | Cost moved where | Effort |
|---|---|---|---|---|
| Smaller ring with an exact per-call bound (adaptive activation bits) | -30 to -40% (K24-K28) | unchanged | none; wire and GPU also drop by 20-40% | weeks; 7B quality still to measure |
| Low-bit storage of W·r (store only the decode-relevant bits) | -27% (5.07 MB/token, measured on real 7B weights) | unchanged; soundness proof needs a rewrite | none | weeks |
| Prefix / KV-cache reuse | -40 to -85% for chat and RAG (cited traces) | new metadata: the service sees uncached suffix lengths | gateway RAM for the KV cache | weeks |
| Provider-held blinded half ("z-store") | about 0 | same as today (factory and provider do not collude) | factory uploads 7 MB/token to us; we store it | small |
| **Hinted LWE masking** (provider generates masks on encrypted randomness) | **no per-token growth**; one-time hint of about 25-45 GB per 7B model plus a few GB of buffer (about 140 GB for 70B) | **single provider, LWE** (standard, post-quantum); no factory, no collusion assumption | our GPU about 1.4-1.6x of today's factory work; wire +10-15 MB/token; gateway about 0.1 core-s/token | about 2-3 months to production quality |

## Measured

- Exact ring bound (Qwen2.5-0.5B, wikitext-2): perplexity 17.522 at W8A12/K40. With a K24 ring and
  provably exact adaptive bits it is 17.525; with K23 it is 17.571. The actual output magnitudes
  need only 21-23 bits for qkv. See `number-format/` and `../../benchmarks/results/quant/`.
- Hinted / token-batched LWE preprocessing prototype: bit-exact W·r mod 2^40 on real 0.5B and
  7B-shaped slices. The server's work is a plain int8 GEMM mod 2^64, with no NTTs or rotations,
  about 290x faster than the library ciphertext-times-plaintext path. See `he-prototype/`.

## Rejected

- Dual-LPN or structured masks: security unverified, and the gateway cost is at least as high as
  local inference.
- Just-in-time CPU factory: needs more CPU than computing the linears locally.
- Partial local compute: leaves the provider almost nothing to do.
- Active-expert-only masks for mixture-of-experts models: leaks the routing.
- Classic online HE (BumbleBee and similar): about 8 minutes per 7B token.
- The 25-bit upload-compression variant of the HE prototype: leaks noise information.

## Required before shipping hinted LWE masking

1. Run the lattice-estimator on the chosen parameters (N=4096, Q=2^64, or MLWE n=2560, q≈2^56).
2. The gateway alone chooses the seed of the public matrix, and no slot is ever reused,
   including on retries.
3. Fresh Freivalds vectors per batch, with a stated vector count and range-check argument.
4. An external cryptography review.

## Not completed

The number-format track's projections for 7B and 70B were interrupted by container restarts. The
7B statistics run reached layer 19 of 28. Next step: measure the adaptive-ring quality on 7B on a
Kaggle GPU.
