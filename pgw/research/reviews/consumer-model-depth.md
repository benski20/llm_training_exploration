# Consumer chat: model depth, shallower models, and what crypto can and cannot do

Date: 2026-11-27. Scope: the model side of per-token latency for a consumer device (the client holds all plaintext
and evaluates every non-linear op) talking to our hosted GPUs (masked linear layers only). Four questions: which
open-weight chat models give the most quality per layer; how much depth can be removed (pruning, parallel blocks)
and at what quality; whether any cryptographic or architectural trick lets the provider run more than one
non-linear stage per round trip without trusting it; and what 20 tok/s up to 32B-class quality would take.

Tags: **measured** (here or on Kaggle), **cited** (paper / model card, not reproduced), **derived** (arithmetic on
shapes and measured constants), **estimate** (rests on an assumption). Every derived number comes from
`research/reviews/consumer_depth_numbers.py` (stdlib only, shapes from each checkpoint's public `config.json`
fetched 2026-11-27). The experiment is `benchmarks/depth_prune_heal.py` (Kaggle T4 x 2, kernel `pgw-depth-05b`,
`benchmarks/kaggle/build_depth.py`).

## 0. Summary

1. **The cost of a token is its sequential non-linear stops.** With the client evaluating every non-linearity, a pass
   costs one round trip per sequential non-linear stage plus one (`S + 1`). A standard block has two stops
   (attention, SiLU gate): `2L + 1` with fusion, `4L + 1` today. A **parallel attention+MLP block** has one:
   `L + 1`. This floor is structural: no architecture with a non-linearity per block, and no cryptographic
   technique within a stateless single provider, gets below one trip per block (section 5). The only levers are
   **fewer blocks** and **one stop per block**.
2. **Best quality per layer today (commercial licences): Qwen3.** Qwen3-14B (40 layers, Apache-2.0) scores
   Arena-Hard 86.3 / MMLU-Redux 82.0 (cited, non-thinking), above Qwen2.5-32B (64 layers, 74.5 / 83.9) and level
   with Gemma-3-27B (62 layers): **32B-class chat at 40 layers, 81 fused trips instead of 129**. Qwen3-8B (36 layers,
   AH 79.6) beats Qwen2.5-32B on Arena-Hard too. The engine now supports Qwen3 dense (per-head q/k RMSNorm on the
   client before RoPE; no extra trip; exact, tests at a tiny config; section 9). Pins and SDK support follow the
   normal pin process (not done here: no checkpoint download fits this box's disk).
3. **Parallel blocks and depth pruning (measured on Qwen2.5-0.5B-Instruct, 24 layers, short self-distillation heal):**
   at **equal trips (25 per pass) converting every block to parallel beats dropping half the layers**
   (healed: FineWeb-Edu perplexity 20.4 vs 25.8, top-1 agreement with the original on held-out chat answers 0.765 vs
   0.651, KL 0.31 vs 0.68, ARC-Easy 0.505 vs 0.473, HellaSwag 0.397 vs 0.350), and it matches dropping a quarter of
   the layers (37 trips) on chat agreement (0.765 vs 0.774). **But every shallower student is far from the original**
   after a 7-8M-token heal (original: perplexity 13.6, MMLU 0.495, HellaSwag 0.522): MMLU falls to chance (0.22-0.25)
   for all of them, including the 25% drop. The parallel student's distillation loss was still falling when its time
   budget ended. At this scale and budget the quality gap is large; whether it closes at 7B-14B with ~1B+ tokens is
   the open question (the literature says depth tolerance grows with size; section 4).
4. **MoE does not help a private consumer client.** Hiding the routing (every expert runs on every token; real rows
   masked, the others uniform dummies) keeps the trips of the dense form but multiplies the wire by the
   expert ratio on the expert linears: gpt-oss-20b (24 layers, the fewest of any 32B-class model) moves 4.9 M
   output elements per row-token against 1.0 M routed (leaky), Qwen3-30B-A3B 10.0 M against 1.2 M. Masks stay at
   the routed rows. MoE is a datacenter-gateway option, not a consumer one (section 6).
5. **20 tok/s at 32B-class quality from a consumer device over the internet is not reachable with this protocol
   family, for any model and any trip count we can build.** Two walls, both derived:
   * RTT: with 41 trips (32B-class parallel + pruned to 40 layers, the best case we can imagine) 20 tok/s needs a
     round trip of at most 1.8 ms at 1.7 tokens per pass (3.4 ms at 3). Consumer RTTs are 5-50 ms.
   * Wire: masked data is uniform, so it does not compress. 32B-class at 40 layers moves about 59 MB per token with
     fused2's 7-byte ring (34 MB at today's K32): 20 tok/s needs ~6 Gbit/s down and ~3.5 Gbit/s up. A 1 Gbit/s
     fibre line caps it near 3 tok/s; a 300/30 cable line below 0.2 tok/s (the upload).
   Consumer-direct is **0.2-2 tok/s at 14B-32B-class** (RTT 5-20 ms, 1 Gbit/s; table in section 8). 20 tok/s at
   32B-class needs the client side within ~1-3 ms and 10+ Gbit/s of the GPUs: topology B (the user's own cloud
   VM in the same region as ours), where Qwen3-14B with fused2 models at about 20 tok/s (1.7 tokens per pass)
   and a parallel, pruned 32B-class model at 22-30. That moves the plaintext from the user's device to the user's
   cloud account: the privacy claim then rests on that cloud operator not colluding with us. It must be presented
   as a different deployment, not as the consumer product.
6. **Fused2's wider ring loses on consumer uplinks.** At RTT 20 ms on a 300/30 Mbit/s line fused2 (K56, 7 bytes per
   element, half the trips) is slower than today's K32 schedule (Qwen2.5-7B 0.25 vs 0.31 tok/s, derived): the upload
   dominates. For consumer links the ring width matters as much as the trip count.
7. **Crypto verdicts** (section 5): provider-side non-linearities with client preprocessing (Beaver triples, FSS)
   either need the client online at every non-linear layer (no round saved: that is today's protocol) or a second,
   non-colluding server (a weaker trust model, and ~1-2 GB of FSS keys uploaded per 7B token); FHE removes the
   rounds and the wire but costs ~20 s per 7B token on 8 GPUs (cited) and is approximate and unverified;
   polynomial / ReLU-squared activations and softmax-free attention do not let the provider continue on additive
   masks: decoding `f(x + r)` for a non-affine `f` needs a client product as large as the next layer. Early exit is
   a data-dependent trip count (a leak); padding it away removes the gain (section 7).

Ranked recommendation (consumer chat, commercial licences):

| rank | model | why | trips fused | engine work |
|---|---|---|---|---|
| 1 | **Qwen3-14B** (40L, Apache-2.0) | 32B-class chat (AH 86.3) at 40 layers; dense, so wire and masks scale like a 14B | 81 (41 if converted to parallel) | done here (qwen3 arch); pin + SDK sizing |
| 2 | **Qwen3-8B** (36L) | AH 79.6 > Qwen2.5-32B at 8B cost | 73 | as rank 1 |
| 3 | **Qwen3-4B-Instruct-2507** (36L) | strong small model (AH-v2 43.4 = Mistral-Small-3.2) | 73 | as rank 1 |
| 4 | **Qwen2.5-7B** (28L, supported) | fewest layers at 7B-class today | 57 | none |
| 5 | **Llama-3.2-1B** (16L, Llama licence) | the fewest layers of any usable chat model; weak (MMLU 49) | 33 | none (llama arch); pin |
| research | Qwen3-14B / 32B **-> parallel blocks + prune** | the only route to ~31-41 trips at 32B-class | 31-41 | engine: parallel-block layer (~1 week); training: distillation at 14B-32B scale (tens of GPU-days, section 4) |
| no | MoE (gpt-oss, Qwen3-30B-A3B, Mixtral, DeepSeek-V2-Lite) | routing hiding multiplies the wire 4-23x | 49-97 | large |
| no | Gemma 3 | sandwich (post) norms break fused2's residual folding (3 trips per layer), custom licence | 97-125 | large |

## 1. Cost model

Per pass (one verify pass yields `tau` tokens, `R` rows per pass under strict speculation):

`t_pass = trips x (RTT + o) + bytes_up / U + bytes_down / D + t_device + t_serve`, `tok/s = tau / t_pass`

* `trips = S + 1` fused (fused2 for standard blocks, the fusion workstream's `L + 1` for parallel blocks), `4L + 1`
  today; `S` = sequential non-linear stops.
* `o` = 0.25 ms per trip (lane 0.13 + gateway 0.1, measured at 7B, `benchmarks/round_trip_model.py`); `t_serve` 2 ms;
  `t_device` = every int8 weight read once on an H100 (0.186 ms per 233 MB measured), x1.3 for fused2 pair trips.
* bytes: every masked linear's inputs (up) and outputs (down) plus the masked head (`V` outputs), per row, at 4 bytes
  (K32) today or 7 bytes (fused2's K56C53). They are uniform ring elements: incompressible.
* `R = 3` for `tau = 1.7` (the shipped draft head measures 1.64-1.72 at R = 3-4 on 7B open chat), `R = 5` for
  `tau = 3` (an EAGLE-class head, not trained).

Wire per token is `R / tau x L x (5d + 3I + kv terms) x ring bytes`. For a fixed parameter count
`P ~ L (4d^2 + 3dI)` the wire scales like `P / d`: **wide-and-shallow models are better on both trips and wire.**

## 2. Survey: layers vs quality (open weights, chat)

Quality columns are copied, not measured: AH = Arena-Hard v1, MR = MMLU-Redux, both from the Qwen3 technical report
(arXiv 2505.09388, Tables 14-20, non-thinking mode, one harness for every model in them); other numbers from the
model cards as marked. tok/s here is the RTT-bound figure with fused trips (no wire limit; section 8 adds it).

| model | layers | trips 4L+1 | trips fused | tok/s RTT 5 / 20 / 50, tau 1.7 | tok/s RTT 5 / 20 / 50, tau 3 | quality | licence | engine work |
|---|---:|---:|---:|---|---|---|---|---|
| Llama-3.2-1B-Instruct | 16 | 65 | 33 | 9.7 / 2.5 / 1.0 | 17.1 / 4.5 / 1.8 | MMLU 49.3 (card) | Llama 3.2 Community (commercial, 700M MAU cap, AUP) | none (llama); pin |
| Qwen2.5-0.5B-Instruct | 24 | 97 | 49 | 6.6 / 1.7 / 0.7 | 11.6 / 3.0 / 1.2 | weak | Apache-2.0 | supported |
| Qwen2.5-1.5B-Instruct | 28 | 113 | 57 | 5.6 / 1.5 / 0.6 | 10.0 / 2.6 / 1.0 | AH 9.0 / MR 50.7 | Apache-2.0 | supported |
| Qwen3-1.7B | 28 | 113 | 57 | 5.6 / 1.5 / 0.6 | 10.0 / 2.6 / 1.0 | AH 36.9 / MR 64.4 | Apache-2.0 | qwen3 (done) |
| Llama-3.2-3B-Instruct | 28 | 113 | 57 | 5.6 / 1.5 / 0.6 | 10.0 / 2.6 / 1.0 | MMLU 63.4 (card) | Llama 3.2 Community | none; pin |
| Qwen2.5-7B-Instruct | 28 | 113 | 57 | 5.6 / 1.5 / 0.6 | 10.0 / 2.6 / 1.0 | AH 52.0 / MR 75.4 | Apache-2.0 | supported |
| Qwen2.5-3B-Instruct | 36 | 145 | 73 | 4.4 / 1.1 / 0.5 | 7.8 / 2.0 / 0.8 | AH 23.7 / MR 64.4 | Qwen Research (**non-commercial**) | supported |
| Qwen3-4B (-Instruct-2507) | 36 | 145 | 73 | 4.4 / 1.1 / 0.5 | 7.8 / 2.0 / 0.8 | AH 66.2 / MR 77.3; 2507: AH-v2 43.4 | Apache-2.0 | qwen3 (done) |
| Qwen3-8B | 36 | 145 | 73 | 4.4 / 1.1 / 0.5 | 7.8 / 2.0 / 0.8 | AH 79.6 / MR 79.5 | Apache-2.0 | qwen3 (done) |
| Llama-3.1-8B-Instruct | 32 | 129 | 65 | 5.0 / 1.3 / 0.5 | 8.7 / 2.3 / 0.9 | AH 30.1 / MR 61.7 | Llama 3.1 Community | none; pin |
| **Qwen3-14B** | 40 | 161 | 81 | 4.0 / 1.0 / 0.4 | 7.0 / 1.8 / 0.7 | **AH 86.3 / MR 82.0** | Apache-2.0 | qwen3 (done) |
| Phi-4 (14B) | 40 | 161 | 81 | 4.0 / 1.0 / 0.4 | 7.0 / 1.8 / 0.7 | AH 75.4 / MR 85.3 (IFEval 62.1) | MIT | phi3: split fused `qkv_proj` / `gate_up_proj` in the packer |
| Mistral-Small-3.2-24B | 40 | 161 | 81 | 4.0 / 1.0 / 0.4 | 7.0 / 1.8 / 0.7 | MMLU 80.5, MMLU-Pro 69.1, AH-v2 43.1 (card) | Apache-2.0 | mistral = llama + explicit head_dim; 3.1/3.2 ship as a multimodal wrapper (text tower only) |
| Qwen2.5-14B-Instruct | 48 | 193 | 97 | 3.3 / 0.9 / 0.3 | 5.9 / 1.5 / 0.6 | AH 68.3 / MR 80.0 | Apache-2.0 | pinned, preview |
| Gemma-3-12B-it | 48 | 193 | 97 | 3.3 / 0.9 / 0.3 | 5.9 / 1.5 / 0.6 | AH 82.6 / MR 77.8 | Gemma Terms (custom) | large: GeGLU, post-norms (break fused2 folding), windows, soft caps |
| Gemma-3-27B-it | 62 | 249 | 125 | 2.6 / 0.7 / 0.3 | 4.6 / 1.2 / 0.5 | AH 86.8 / MR 82.6 | Gemma Terms (custom) | large |
| Qwen2.5-32B-Instruct | 64 | 257 | 129 | 2.5 / 0.7 / 0.3 | 4.4 / 1.1 / 0.5 | AH 74.5 / MR 83.9 | Apache-2.0 | pinned, preview |
| Qwen3-32B | 64 | 257 | 129 | 2.5 / 0.7 / 0.3 | 4.4 / 1.1 / 0.5 | AH 92.8 / MR 85.7 | Apache-2.0 | qwen3 (done) |
| Qwen2.5-72B-Instruct | 80 | 321 | 161 | 2.0 / 0.5 / 0.2 | 3.5 / 0.9 / 0.4 | AH 81.2 / MR 86.8 | Qwen licence (< 100M MAU) | pinned, preview |
| gpt-oss-20b (MoE 32/4) | 24 | 97 | 49 | 6.6 / 1.7 / 0.7 | 11.6 / 3.0 / 1.2 | MMLU 80.4 low / 85.3 high effort (arXiv 2508.10925); reasoning model | Apache-2.0 | large (MoE, sinks, windows, biases, clamped SwiGLU, MXFP4) |
| DeepSeek-V2-Lite-Chat (MoE 64/6) | 27 | 109 | 55 | 5.8 / 1.5 / 0.6 | 10.3 / 2.7 / 1.1 | MMLU 55.7 (card) | DeepSeek licence | large (MLA, MoE) |
| Mixtral-8x7B (MoE 8/2) | 32 | 129 | 65 | 5.0 / 1.3 / 0.5 | 8.7 / 2.3 / 0.9 | MMLU 70.6 (paper) | Apache-2.0 | MoE |
| Qwen3-30B-A3B (MoE 128/8) | 48 | 193 | 97 | 3.3 / 0.9 / 0.3 | 5.9 / 1.5 / 0.6 | AH 88.0 / MR 84.1; 2507: AH-v2 69.0 | Apache-2.0 | qwen3_moe |
| gpt-oss-120b (MoE 128/4) | 36 | 145 | 73 | 4.4 / 1.1 / 0.5 | 7.8 / 2.0 / 0.8 | MMLU 85.9 low / 90.0 high | Apache-2.0 | as gpt-oss-20b |
| **native parallel blocks** (`S = L`; today 2L+1, fused L+1): | | | | | | | | |
| Phi-2 (2.7B) | 32 | 65 | 33 | 9.7 / 2.5 / 1.0 | 17.1 / 4.5 / 1.8 | MMLU 56.7 (base; no official chat) | MIT | phi: GELU, LayerNorm+bias, partial rotary |
| StableLM-2-12B-chat | 40 | 81 | 41 | 7.8 / 2.0 / 0.8 | 13.8 / 3.6 / 1.5 | MMLU ~62 (card) | Stability Community (commercial < $1M revenue) | LayerNorm, partial rotary, qk norm |
| Falcon-11B | 60 | 121 | 61 | 5.3 / 1.4 / 0.6 | 9.3 / 2.4 / 1.0 | MMLU 58.4 (base) | Falcon 2 TII (Apache-based) | falcon |
| Falcon-40B | 60 | 121 | 61 | 5.3 / 1.4 / 0.6 | 9.3 / 2.4 / 1.0 | MMLU ~55 (base) | Apache-2.0 | falcon |
| GPT-J-6B / Pythia-12B / GPT-NeoX-20B | 28 / 36 / 44 | | 29 / 37 / 45 | | | MMLU ~26-33 (base) | Apache-2.0 | gpt-j / gpt-neox |
| Command R / R7B | | | | | | parallel, strong | **CC-BY-NC (non-commercial)**: excluded | |

Reading:
* **Quality per layer** (AH per layer, same harness): Qwen3-8B 2.21, Qwen3-14B 2.16, Phi-4 1.89, Qwen2.5-7B 1.86,
  Qwen3-4B 1.84, Qwen3-30B-A3B 1.83, Gemma-3-12B 1.72, Qwen3-32B 1.45, Gemma-3-27B 1.40, Qwen2.5-32B 1.16,
  Qwen2.5-72B 1.02, Llama-3.1-8B 0.94. The Qwen3 generation roughly halves the layers needed for a given chat
  quality relative to Qwen2.5 (Qwen3-14B at 40 layers ~ Qwen2.5-72B at 80 on Arena-Hard).
* **Native parallel-block models are all weak or restricted**: none is a competitive commercial chat model (the best,
  Phi-2 and StableLM-2-12B-chat, sit at MMLU ~57-62, below Qwen3-4B; GPT-J / Pythia / GPT-NeoX / Falcon-7B are near
  chance on MMLU and have no modern chat tune). **Verdict: LayerNorm + GELU (+ partial rotary) engine support for
  Pythia / Falcon / GPT-J is not worth building for consumer chat**: it would buy one trip per layer on models that
  are 2-3 generations behind. One trip per layer at useful quality needs a **converted** model (section 3).
* **Mistral-Small-3.2** is the other 40-layer 32B-class option (Apache-2.0) but its width (I = 32768) makes it 1.5x
  Qwen3-14B's wire per token; Phi-4 is 40 layers and MIT but weaker at chat (IFEval 62).

## 3. One stop per block: parallel attention + MLP

A parallel block computes `y = x + attn(ln1 x) + mlp(ln2 x)`. Attention and the MLP read the same input, so `qkv` and
`gate_up` travel in one trip, and `o`, `down`, the residual and the next layer's input linears in the next
(fused2-style folding; implemented by the fusion workstream as `pgw/src/pgw/engine/fused_parallel.py`, merged on the
main branch after this worktree was cut). `L + 1` trips per pass instead of `2L + 1`, every
weight kept, the provider's view unchanged (masked ring elements of public shapes).

Literature (cited, not reproduced): PaLM (Chowdhery et al. 2022) trained with parallel blocks for ~15% faster
training and reports "a small degradation in quality at 8B scale but no degradation at 62B"; GPT-J, GPT-NeoX-20B,
Pythia, Falcon, ViT-22B and Command R were all trained parallel from scratch. Converting a *trained sequential*
model is less studied: "Transformer layers as painters" (Sun et al. 2024) runs groups of middle layers in parallel
on frozen models with graceful degradation; "Leveraging the true depth of LLMs" (2025) runs pairs of adjacent layers
in parallel and recovers most of the loss with short fine-tuning. Neither converts every block's attention/MLP pair
of a chat model and heals it, which is what this design needs, so we measured it.

### Experiment (measured, Kaggle T4 x 2)

Setup: Qwen2.5-0.5B-Instruct (24 layers, pinned revision), students initialized from it; healed by distillation
from the unmodified model (full-vocabulary KL at every position, AdamW, peak lr 5e-5, cosine, 1,000 steps x 8,192
tokens, embeddings frozen, fp16 on one T4 per student) on 4.8M tokens: 1.6M of chat (4,500 UltraChat prompts, MIT,
answered greedily by the original model itself) and 3.2M of FineWeb-Edu (ODC-By 1.0). Dropped layers: the contiguous
block with the smallest angular distance (Gromov et al.), never the last layer: layers 8-13 for 25%, 7-18 for 50%.
Evaluation never trained on: perplexity on 107 held-out FineWeb-Edu documents x 1,024 tokens; 150 held-out UltraChat
`test_sft` prompts answered by the original (teacher-forced: top-1 agreement, KL); 600 items each of MMLU (MIT; 0-shot,
letter logits), ARC-Easy / ARC-Challenge (CC BY-SA 4.0; per-character normalized log-likelihood) and HellaSwag (MIT).
Results: `benchmarks/results/depth_prune/kaggle_05b_r1/`. **Measured**, one seed, n = 600 per benchmark (a 95%
interval is about +-0.04 near 0.5), a single short heal per variant.

| variant | layers | sequential stops | trips per pass (fused) | stage | ppl | chat top-1 | chat KL | MMLU | ARC-E | ARC-C | HellaSwag |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| original | 24 | 48 | 49 | | **13.57** | 0.999 | 0.000 | **0.495** | **0.620** | 0.332 | **0.522** |
| drop 25% | 18 | 36 | 37 | no heal | 23.81 | 0.582 | 0.914 | 0.240 | 0.495 | 0.252 | 0.412 |
| drop 25% | 18 | 36 | 37 | healed 8.2M | 17.19 | 0.774 | 0.268 | 0.225 | 0.525 | 0.265 | 0.403 |
| drop 50% | 12 | 24 | 25 | no heal | 233.66 | 0.200 | 3.742 | 0.247 | 0.332 | 0.228 | 0.275 |
| drop 50% | 12 | 24 | 25 | healed 8.2M | 25.80 | 0.651 | 0.682 | 0.238 | 0.473 | 0.242 | 0.350 |
| **parallel** | 24 | 24 | 25 | no heal | 2.0e6 | 0.008 | 13.63 | 0.248 | 0.263 | 0.255 | 0.283 |
| **parallel** | 24 | 24 | 25 | healed 7.0M (time cap, 860 steps) | **20.44** | **0.765** | **0.307** | 0.243 | 0.505 | 0.262 | 0.397 |
| parallel + drop 25% | 18 | 18 | 19 | no heal | 3.1e5 | 0.029 | 11.31 | 0.245 | 0.273 | 0.217 | 0.277 |
| parallel + drop 25% | 18 | 18 | 19 | healed 8.2M | 25.49 | 0.687 | 0.550 | 0.240 | 0.410 | 0.233 | 0.337 |

Distillation loss (KL per token) first step -> mean of the last 50 steps: drop 25% 0.60 -> 0.16, drop 50% 2.85 -> 0.48,
parallel 11.3 -> 0.26 (still falling at the cap), parallel + drop 25% 9.3 -> 0.43.

Reading:
* A sequential model converted to parallel blocks is **destroyed without training** (perplexity 2 million: the MLP
  no longer sees the attention output it was trained on) and **recovers fast** with distillation: after 7M tokens it is
  the best 25-trip student by every metric, and on chat agreement it equals the 37-trip drop-25% student.
* Neither recovers knowledge at this budget: MMLU at chance for every student; ARC-Easy keeps ~70-75% and HellaSwag
  ~55% of the original's margin above chance (drop 25% and parallel alike), the other students less. Perplexity gaps of 1.27x (drop 25%) to 1.9x (drop
  50%) remain. A 0.5B model has the least redundancy of any size (Gromov et al.), so this is a pessimistic point; it is
  not evidence that a 14B converts cheaply.
* Combining both (19 trips) costs more than either alone at this budget.
* Greedy samples (in `results.json`) stay fluent for all healed students; the drop-50% and parallel+drop students
  drift off-task more often.

What a decision needs: the same conversion at 1.5B-7B with a ~0.5-1B-token heal (Kaggle T4 x 2 manages ~3k tokens/s at
0.5B, so ~0.25B tokens in a 12-hour session at 0.5B and far less at 7B: a funded run, ~250 H100-hours at 14B for 3B
tokens, section 4).

## 4. Fewer blocks: depth pruning and depth-wise distillation

Literature (cited): Gromov et al. 2024 ("The unreasonable ineffectiveness of the deeper layers") drop the contiguous
block with the smallest angular distance between its input and output and heal with a short QLoRA run on C4; MMLU
holds until a model-dependent fraction of the layers is gone (around half for Llama-2-70B, much less for small
models and for Qwen) and then collapses to chance, while perplexity degrades smoothly. ShortGPT (Men et al. 2024)
removes the layers of lowest "block influence" and keeps most multiple-choice accuracy at ~25% removal on
Llama-2-7B/13B without retraining, but generation degrades more. Minitron (Muralidharan et al. 2024; Sreenivas et
al. 2024) is the strongest data point with real distillation: Llama-3.1-8B (32 layers) -> Llama-3.1-Minitron-4B-Depth
(16 layers) distilled on 94B tokens gives MMLU 58.7 against 65.3 for the 8B (cited), and width pruning to the same
size beat depth pruning (60.5). The pattern across these: **25% of the depth is cheap to remove with healing; 50% costs
real quality even with ~100B tokens of distillation; deeper/larger models tolerate more.**

What the 0.5B experiment adds (section 3's table, same heal for every variant): a 25% drop healed on 8M tokens keeps
perplexity within 1.27x and chat agreement at 0.77 but loses MMLU to chance; a 50% drop is clearly worse than the
parallel conversion at the same trip count.

**Estimate for a 32B-class model at 32-40 layers.** Qwen3-32B has 64 layers; 40 layers is a 37.5% cut, 32 a 50% cut.
From the literature pattern and our small-scale deltas, a Qwen3-32B pruned to 40 layers and distilled on a few
billion tokens would plausibly land between Qwen3-14B and Qwen3-32B (AH 86-93) at roughly 21B parameters, i.e. no
better per trip than **Qwen3-14B as released, which is already 40 layers** and needs no training. Pruning to 32
layers would need Minitron-scale distillation (~100B tokens; ~18,000 H100-hours at 32B by the same arithmetic, estimate) and would
likely fall below Qwen3-14B. So for the 40-layer target the recommendation is to **start from Qwen3-14B, not to prune
Qwen3-32B**; the open training question is converting Qwen3-14B (40 layers) to parallel blocks (41 trips) and possibly
pruning it to 30 (31 trips). Distillation cost at 14B for a few billion tokens is roughly 200-500 H100-hours
(estimate; arithmetic below), out of reach on Kaggle but small for a funded run.

Arithmetic: training FLOPs `6 N T` plus the teacher's `2 N T`: `8 x 14.8e9 x 3e9 = 3.6e20` FLOPs for 3B tokens; at an
effective 400 TFLOPS (bf16 H100, ~40% MFU) that is ~250 H100-hours.

## 5. Crypto-friendly architectures: what is possible and what is not

The provider holds `x_hat = x + r mod 2^K` with `r` uniform and fresh, and nothing else. Two facts settle most of
this section:

* **Privacy is never the problem.** Whatever function the provider applies to `x_hat`, its view is a function of
  `x_hat` alone, which is uniform and independent of `x`. Applying a non-linear op does not leak (in this
  model, information-theoretically for ideal masks). The problem is **decoding**: the client must turn the result
  into `f(x)` cheaply.
* **Additive masks are only linearly homomorphic.** For affine `f`, `f(x + r) - (f(r) - f(0)) = f(x)` with the
  correction precomputed offline (that is the mask factory's `W r`). For any non-affine `f` the correction depends on
  `x`. Degree 2: `(x + r)^2 = x^2 + 2 x r + r^2`, and the next linear layer turns the cross term into
  `2 W diag(r) x`, a product as large as that layer with a matrix that changes every token: decoding costs the client
  as much as computing the layer itself (and needs the weights, which the weight-free client does not hold, and `x`,
  which it only has after a round trip). ReLU, SiLU, exp, max and division have no such decomposition at all.

### 5a. Provider-side non-linearities with client-generated preprocessing (MPC)

| technique | rounds per layer with the client | trust | bandwidth (7B token, estimate) | verdict |
|---|---|---|---|---|
| Beaver triples (client = dealer, provider = one party, client = the other) | >= 1 per multiplicative depth of each approximated non-linearity (SiLU polynomial: 3-4; exp/reciprocal: more) | 2-party, same as today | small | **more rounds than today**: the client evaluating the non-linearity in the clear is the round-optimal case of this family (it is already "FSS where the gate runs at the party holding the data") |
| FSS / masked-wire 2PC (SIGMA-style) with the client as dealer and **two** provider servers evaluating | ~0 with the client (1 per token); 10-20 server-server rounds per layer at datacenter RTT | **adds non-collusion between two GPU providers**: weaker than today's 2-party guarantee, so excluded by the rules | DCF-based keys ~0.5-1 KB per comparison: SiLU 530k elements + softmax ~800k scores at 1k context, so ~1-2 GB of keys uploaded per 7B token (~10x that at 32B) | rejected: weaker trust, and the upload alone is minutes per token on a consumer line; SIGMA reports ~27 s per Llama-2-7B token even on LAN GPUs (cited, prior memo) |
| Garbled circuits (client garbles offline, provider evaluates the whole pass) | 0 (one message) | 2-party | linear layers free on arithmetic labels, but every non-linear element needs garbled tables: ~34 GB (integer LUT reference) to ~850 GB (IEEE float) per 7B token (prior memo, `research/plans/fewer-round-trips.md` section 2) | rejected: 10^4-10^5 x the wire, and the client's garbling work exceeds computing the model |

The prior memo's conclusion stands and extends to any client-preprocessing scheme: **with one provider and no FHE, every
sequential non-linear stage costs at least one interaction with the client.** The client's offline work can remove
online *computation*, never online *rounds*.

### 5b. FHE for the non-linear parts (or everything)

Cited (2025-2026): CKKS on GPUs runs Llama-2-7B generation at **18-22 s per output token on 8 RTX PRO 6000 GPUs**
(Sylph, arXiv 2601.18511); NEXUS is cited at an amortized 51.8 s per Llama-3-8B token on 4 A100s; hybrid CKKS/TFHE
work reports minutes per token, with the non-linear operators above 70% of the decode time. Scaled to 32B (about
4.5x the work): ~100 s per token on 8 GPUs, i.e. **~2,000x too slow for 20 tok/s and ~800 GPU-seconds per token
(~25,000x today's provider cost)**. FHE does remove both walls of section 8 (one encrypted token up, encrypted logits
or a token down: MBs, not tens of MBs, and one round trip), so it is the only family that could ever make
consumer-direct private inference fast, but it is 3-4 orders of magnitude away. It is also approximate (CKKS
polynomial approximations: not bit-exact to any pinned reference) and gives no integrity against a provider that
returns garbage (verifiable FHE is not practical at this size). TFHE programmable bootstrapping for SiLU alone is
~530k bootstraps per 7B token (~50 s at 10^4 PBS/s, prior memo). **Verdict: not viable this decade for chat
latency; worth tracking only as a research option.**

### 5c. Private-inference architectures

MPCFormer (quadratic GeLU and softmax substitutes + distillation), THE-X, BOLT, PUMA, SecFormer and the
"encryption-friendly" LLM work (Gaussian-kernel attention, LoRA) all cut the **cost** of non-linearities inside MPC or
FHE (fewer comparisons, lower polynomial degree). None reduces the number of **sequential non-linear stages**, and in
our design the client already evaluates each stage in the clear at negligible cost. Specific cases:

| proposal | could the provider run several layers on masked data with cheap client corrections? |
|---|---|
| polynomial / ReLU-squared / quadratic activations | **No.** Degree-2 decoding needs `W diag(r) x` per layer (section 5 intro): a full-size client product, i.e. the same as computing that layer locally (which `local_linears` already offers, at that cost). |
| softmax-free attention (linear, ReLU, polynomial kernels) | **No.** Attention multiplies two data-dependent quantities (`q` and `k`, or `phi(q)` and the state `sum phi(k) v^T`): bilinear in secrets, impossible on additive masks of both. The client computes attention today without weights; nothing to gain. |
| data-independent linear token mixing (fixed convolutions, linear recurrences without gates) | Yes in principle: linear in the data, so the provider could even keep a masked recurrent state (the client tracks the mask's contribution offline). But such a block still needs a non-linear channel mix, so it is still one stop per block: no better than a parallel attention+MLP block, and with weaker models (gated SSMs such as Mamba are data-dependent again). |
| fewer non-linear stages (parallel blocks, wider blocks) | Yes: this is the lever (section 3). |

**What is impossible**, stated plainly: (i) a single provider applying any non-affine function to additively masked
data in a way the client can undo for less than the cost of computing it, (ii) fewer than one client interaction per
sequential non-linear stage without FHE, garbling or a second non-colluding server, (iii) compressing masked traffic.

## 6. MoE: who decides the experts, and what hiding it costs

The router is a linear layer on the MLP input, so its logits come back in the same trip as the experts' inputs could
go out, but the **choice** (top-k) is made by the client and decides which expert weights run. If only the chosen
experts ran, the provider would learn the routing per token: a data-dependent leak (topic, language, code vs prose).
Options:

| option | trips | provider compute | wire | masks / factory | privacy |
|---|---|---|---|---|---|
| route openly (only chosen experts) | dense form | k/E of the experts | routed rows | routed | **leaks routing**: excluded |
| **oblivious: every expert on every token** (`pgw.moe.oblivious`): one masked row in, all experts' `gate_up` back (router logits in the same trip); `down`: one row per expert, unselected rows masked zeros, provider sums the expert outputs (one `d`-vector back) | dense form (`2L + 1` fused) | E/k x on expert linears | gate_up download x E/k, down upload x E/k | routed rows only, per-expert pools refilled routing-independently (`pgw.moe.refill`) | perfect (uniform view for every routing); pool stalls are observable and must be sized away |
| PIR-style retrieval of the chosen experts' outputs (provider computes all, client fetches k) | **+1 trip per layer** (routing known only after the router trip) | E/k x plus PIR scans | ~k/E of the oblivious download plus PIR overhead (several x per record) | all experts (precomputed for unknown routing) | computational (lattice PIR) |
| hash / position routing (public) | dense | k/E | routed | routed | hash on token id leaks the token; position routing is not MoE |

Derived per row-token (fused, oblivious): gpt-oss-20b 4.9 M down / 2.5 M up elements (routed: 1.0 M down),
Qwen3-30B-A3B 10.0 M / 5.1 M (routed 1.2 M), Mixtral-8x7B 7.8 M / 4.1 M (routed 2.3 M), gpt-oss-120b 27 M (routed 1.4 M).
Compared with dense Qwen3-14B (2.2 M down), the oblivious MoE wire is 2-12x larger. On a consumer link MoE is
strictly worse than a dense model of the same quality; in a datacenter it is viable (the provider's extra FLOPs are
cheap: batched decode reads every expert anyway). gpt-oss-20b is the most interesting MoE (24 layers, 49 fused
trips, 25 if parallel) but it is a reasoning model whose visible answer follows an analysis channel, so its tokens
per answer are higher, and its MXFP4 weights do not fit a per-row int8 pack exactly.

## 7. Early exit and adaptive depth

Early exit (CALM, LayerSkip and similar) saves ~30-50% of layers on easy tokens in the literature. Here the provider
sees the number of sequential trips per pass and their timing. A per-token exit depth is a function of the token's
difficulty, which correlates with content (names, numbers and code are hard; boilerplate is easy): **a data-dependent
leak, excluded by the design rule that nothing data-dependent shows in pass counts, rows or timing.** Mitigations
and what they leave:

* pad every pass to full depth with dummy trips (uniform rows): private, but the dummy trips are sequential in time,
  so no latency is saved (only the client's float work on dummy layers);
* fire the dummy trips early or in parallel: the timing pattern then shows the exit point;
* choose the depth from public information only (per deployment or per user tier): that is static pruning;
* use early layers as a self-speculative drafter: the draft layers are remote trips too, sequential before the
  verify pass, so it costs trips instead of saving them (the shipped client-side draft head is the right drafter).

**Verdict: adaptive depth gives nothing under the privacy rules; static depth (section 4) is its private form.** The
same argument is why MoE routing must be oblivious.

## 8. What it takes for 20 tok/s up to 32B

Derived (fused trips, K56 ring, `o` 0.25 ms, H100 device time, tau 1.7 / 3):

| model | trips fused | consumer cable, RTT 20, 300/30 Mbit/s | consumer fibre, RTT 20, 1 Gbit/s | metro edge GPU, RTT 5, 1 Gbit/s | same region (topology B), RTT 0.3, 25 Gbit/s |
|---|---:|---:|---:|---:|---:|
| Llama-3.2-1B | 33 | 0.8 / 0.9 | 2.1 / 3.4 | 5.7 / 7.8 | 63.8 / 100.2 |
| Qwen2.5-7B | 57 | 0.3 / 0.3 | 1.1 / 1.6 | 2.4 / 3.1 | 29.7 / 44.2 |
| Qwen3-14B | 81 | 0.2 / 0.2 | 0.8 / 1.1 | 1.6 / 2.1 | **19.8 / 29.5** |
| Mistral-Small-24B | 81 | 0.1 / 0.1 | 0.7 / 1.0 | 1.3 / 1.6 | 15.9 / 23.0 |
| Qwen3-32B | 129 | 0.1 / 0.1 | 0.4 / 0.6 | 0.9 / 1.1 | 10.8 / 15.7 |
| gpt-oss-20b (oblivious MoE) | 49 | 0.1 / 0.1 | 0.8 / 1.0 | 1.1 / 1.3 | 17.0 / 22.6 |
| Qwen3-32B pruned to 40L | 81 | 0.1 / 0.1 | 0.7 / 1.0 | 1.4 / 1.7 | 16.9 / 24.6 |
| Qwen3-14B -> parallel (40L) | 41 | 0.2 / 0.2 | 1.2 / 1.6 | 2.0 / 2.4 | **26.7 / 37.6** |
| Qwen3-32B -> parallel + pruned to 40L | 41 | 0.1 / 0.1 | 1.0 / 1.4 | 1.6 / 1.9 | 21.6 / 30.0 |
| Qwen3-14B -> parallel + pruned to 30L | 31 | 0.2 / 0.3 | 1.6 / 2.1 | 2.7 / 3.2 | 34.6 / 49.0 |

And the requirement turned around (RTT budget ignores wire and device time; wire at 20 tok/s, fused, tau 1.7):

| model | trips fused | max RTT for 20 tok/s, tau 1.7 / 3 | wire at 20 tok/s down / up (Mbit/s) | client mask cores at 20 tok/s (estimate) |
|---|---:|---:|---:|---:|
| Llama-3.2-1B | 33 | 2.27 / 4.23 ms | 998 / 457 | 0.8 |
| Qwen2.5-7B | 57 | 1.21 / 2.35 ms | 3,049 / 1,650 | 2.3 |
| Qwen3-14B | 81 | 0.77 / 1.58 ms | 4,429 / 2,601 | 3.3 |
| Qwen3-32B | 129 | 0.39 / 0.90 ms | 9,367 / 5,580 | 7.1 |
| Qwen3-14B -> parallel | 41 | 1.77 / 3.36 ms | 4,429 / 2,601 | 3.3 |
| Qwen3-32B -> parallel + pruned to 40L | 41 | 1.77 / 3.36 ms | 5,967 / 3,491 | 4.5 |
| gpt-oss-20b -> parallel (oblivious MoE) | 25 | 3.07 / 5.67 ms | 9,658 / 4,845 | 1.5 |

(Mask cores: 112 ms per slot per core measured at 32B with K32 (STATUS, 2026-11-26), scaled by masked outputs per
slot and x1.75 for K56, at 3/1.7 slots per token. An estimate: the fused2 factory cost is not measured.)

What it takes, in order:

1. **Topology, not the model, decides whether 20 tok/s is possible.** Consumer-direct (RTT >= 5 ms, <= 1 Gbit/s) tops
   out at ~1-3 tok/s for 14B-32B-class even with every model-side lever. 20 tok/s needs the client side within a few
   ms and ~5-10 Gbit/s of the GPUs. For a consumer that means a **personal gateway VM** in their own cloud account in
   our region (topology B), with the device as a thin front end. Privacy then rests on that cloud operator not
   colluding with us (no trusted hardware, but a third party in the trust base): a product decision to make
   explicitly, not a free upgrade.
2. **Model: Qwen3-14B with fused2** reaches ~20 tok/s at 1.7 tokens per pass in topology B (derived, unmeasured) with
   32B-class chat quality and no training. Engine support is in (section 9); the pin, SDK sizing and an H100 run are
   next.
3. **Parallel conversion of Qwen3-14B** (41 trips) would give ~27-38 tok/s there and is the best consumer-side lever
   too (1.2-1.6 tok/s at RTT 20 on fibre vs 0.8-1.1). Quality cost: section 3 measures it at 0.5B; a 14B run needs a
   funded distillation (~250 H100-hours for 3B tokens, estimate).
4. **Speculation (tau 3)** multiplies everything by ~1.5-1.8 over tau 1.7 where trips dominate, much less where the
   wire dominates (rows per pass grow with tau).
5. **The ring**: on consumer links fused2's 7-byte elements cost more than the trips they save (summary item 6); a
   fused schedule for consumer links needs a narrower ring (K40-K48 with fewer activation bits), which the fusion
   workstream should cost.

### The client at 32B on a laptop

| | Qwen3-32B / Qwen2.5-32B (64L) | Qwen3-14B (40L) | source |
|---|---|---|---|
| cache on disk, `factory_weights="pack"` | ~45 GB (int8 factory pack 31 GB + hint 7.4 GB + plain parts) + mask stock 8-20 GB | ~21 GB + stock | derived from STATUS 2026-11-26 (32B measured shapes) |
| cache with `factory_weights="stream"` | ~16 GB (no int8 weights; 1.9 GB check stock) + mask stock | ~8 GB + stock | same; costs ~1-2% more network (weights re-streamed per stock refill) |
| one-time hint build | 2,300 CPU-s with AMX (19 min on 4 vCPUs); ~7 CPU-hours on AVX2-only laptops (extrapolated) | ~45% of that | STATUS 2026-11-26 |
| mask CPU per slot, 1 core, K32 | 112 ms (all parts) | ~50 ms (derived) | STATUS current state |
| sustained 20 tok/s at 3/1.7 slots per token | ~4 cores K32, ~7 with K56 (estimate) | ~2 / ~3.3 | derived |
| mask slot size | 18.6 MB (K32), ~33 MB (K56) | 8.9 / 15.6 MB | derived (masked outputs x ring bytes) |
| an 8 / 20 GB stock built while idle covers | 430 / 1,075 slots = ~245 / ~610 tokens (K32) | ~900 / 2,250 slots | derived |
| RAM | embeddings 1.6 GB (bf16) + head + KV 0.52 MB per token in float32 (4k context 2.1 GB, 32k 17 GB; bf16 KV halves it) + working set: ~8-16 GB | ~6-10 GB | derived |
| network | **~34 MB (K32) / ~59 MB (K56) per token at 40 layers; ~53 / 93 MB at 64 layers** | ~25 / 44 MB | derived |

The idle-time stock is what makes bursts work: a chat user who reads for a minute between ~300-token answers needs
~9 slots/s on average (~1 core at 32B), and an 8-20 GB stock absorbs an answer of a few hundred tokens at full speed.
The network is the part a laptop cannot fix.

## 9. Engine support added: Qwen3 dense (`model_type = "qwen3"`)

Qwen3 differs from Qwen2/Llama in one place the engine sees: each q and k head is RMS-normalized (weights of size
`head_dim`) before RoPE; no linear biases; `head_dim` explicit (128, not `hidden / heads`, e.g. 4B, 32B). The norm is
client-side float work between the qkv trip and RoPE: **no trip, no provider change, no wire change**.

* `pgw.core.types.ModelArch.QWEN3`; `pgw.modelpack.arch`: `has_qk_norm`, `HF_QK_NORM_TENSORS`,
  `plain_tensor_shape`, `hf_plain_tensor_names(..., arch)` (unchanged without `arch`), qwen3 config checks (no
  sliding window, no attention bias); manifest: `layers.<i>.q_norm` / `k_norm` required and shape-checked for qwen3
  packs only; packer and shard-wise packer write them (byte-identical to each other).
* `TransformerModel.qk_norms` (None for qwen2/llama: their layer code path is unchanged, bit for bit);
  `_qk_heads` applies the norms in the standard layer and in fused2's attention step.
* The paths without the norm step aside instead of computing another model: `set_float_ops("native*")` raises for
  qwen3, and the native decode drivers (`decode_native`, `fused2_native`) report not eligible.
* Tests (`tests/engine/test_qwen3.py`, tiny config with `head_dim != hidden / heads`, tied and untied): float
  reference == HF `Qwen3ForCausalLM` (and != without the norm); masked == plain bit for bit on K32/K40/K48 rings,
  logits and sampled generation, with 4L trips per pass; fused2 masked == plain with 2L + 1 trips; packer round trip
  and manifest checks; native paths refuse. `tests/modelpack/test_shardwise.py::test_qwen3_byte_identical`.
* Not done: pins (`PINNED_MODELS`) and SDK sizing for Qwen3-8B/14B/32B (no checkpoint fits this box's disk; the
  shard-wise pin script needs ~4.5 GB scratch per model and should run on a box with room), a native float path with
  the q/k norm (the SDK's default float path is the torch one; the native drivers are opt-in), Qwen3's chat template
  `enable_thinking=False` default for chat (the tokenizer path renders the checkpoint's template as shipped).

## Sources

* Qwen3 technical report, arXiv 2505.09388 (Tables 14-20).
* gpt-oss model card, arXiv 2508.10925 (Table 3).
* Mistral-Small-3.2-24B-Instruct-2506 model card (Hugging Face).
* Sylph: arXiv 2601.18511; ROSETTA: IACR eprint 2026/2047 (FHE LLM decoding).
* PaLM, arXiv 2204.02311 (parallel layers); Gromov et al., arXiv 2403.17887; ShortGPT, arXiv 2403.03853;
  Minitron, arXiv 2407.14679 and 2408.11796; Transformer Layers as Painters, arXiv 2407.09298; Leveraging the true
  depth of LLMs, arXiv 2502.02790; SIGMA (PETS 2024); MPCFormer (ICLR 2023); BOLT (S&P 2024).
* Prior internal memos: `research/plans/fewer-round-trips.md` (section 2), `research/plans/fused2-two-trips-per-layer.md`,
  `research/plans/scale-north-star.md` (MoE), `research/reviews/pks-v3-client-setup-scaling.md`.
