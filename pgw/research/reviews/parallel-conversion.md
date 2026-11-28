# Partial parallel conversion: one round trip for the layers that tolerate it

Date: 2026-11-28. Angle 1 of the parallel-block study: get part of the parallel-block saving (1 fused round trip per
layer instead of 2) **without retraining**, by converting only the layers whose MLP barely notices the change.
Other angles (healing, surrogates) write their own sections or files; sections 0-5 of this file are this angle's; other
angles append their own sections below.

Tags: **measured** (Kaggle T4 or this box), **derived** (arithmetic), **estimate** (rests on an assumption).

## 0. Summary

1. **Training-free conversion buys little: about a quarter of the layers, ~12 % of the trips, at a visible quality
   cost.** Converting the best quarter of the layers (measured, held-out sets) costs **+3.6-8.4 % FineWeb-Edu perplexity
   and leaves chat top-1 agreement with the original at 0.86-0.92** on its own answers; MMLU is flat at 3B / 4B
   (-0.005) and Qwen3-1.7B (0.000) but drops 0.02-0.06 at 0.5B / 1.5B. Trips: 49 -> 43 (0.5B), 57 -> 50 (1.5B,
   Qwen3-1.7B), 73 -> 64 (3B, Qwen3-4B). Half the layers is already far off (chat agreement 0.74-0.81, MMLU -0.05 to
   -0.15) and converting every layer destroys every model (perplexity 10^3-10^6). For reference, the fused2 fixed-point
   reference itself agrees with the float model on 96-97 % of positions.
2. **The damage is nearly additive across layers**, so the map is cheap to choose: the sum of the single-layer KLs
   predicts the KL of the combined conversion within 10-20 % up to a third of the layers (e.g. Qwen3-4B, 12 layers:
   0.083 predicted, 0.092 measured), mildly super-additive beyond. The greedy search on the actual marginal KL is only
   0-3 % better than ranking by the single-layer KL (the area under the curve, section 3).
3. **A training-free, evaluation-free proxy works for the first layers**: the relative change of the MLP *output*
   (`||mlp(n2 h) - mlp(n2(h + a))|| / ||y||`, one forward pass on 16 calibration sequences) ranks layers with Spearman
   0.61-0.94 against the measured single-layer KL. On Qwen3 it gives the measured ranking's curve up to k ~ L/4
   (Qwen3-4B: identical to k = 9); on Qwen2.5 it diverges after the first 2-4 layers (Qwen2.5-3B, k = 3: KL 0.060 vs
   0.011), so the L single-layer evaluations are worth their cost there. The cosine of the MLP *input* alone is a
   weaker predictor (0.53-0.76). Position heuristics fail: "last layers first" is 1.1-1.3x
   worse than sensitivity order, and "first layers first" is catastrophic (layer 0 alone: KL 0.6-8.4).
4. **Where the tolerant layers are**: never layer 0 or 1 (or the last layer); in Qwen3 a contiguous late block
   (Qwen3-1.7B: 20-25; Qwen3-4B: 25-33) plus 2-4 in 4B; in Qwen2.5 scattered: late-middle at 0.5B / 1.5B (16-25), early
   (2-10) plus 25-30 at 3B. Qwen3 tolerates conversion slightly better than Qwen2.5 at equal size (1.7B vs 1.5B:
   perplexity +3.6 % vs +8.4 %, MMLU 0.000 vs -0.060 at 25 %).
5. **The tolerated share grows slowly with size** (KL <= 0.1 on the selection set: 25 % of layers at 0.5B-1.7B, 33-36 % at
   3B-4B; KL <= 0.05: 14-18 % at 0.5B-1.7B, 19-25 % at 3B-4B). **Estimate for Qwen3-14B (40 layers): 10-16 layers, 81 ->
   65-71 trips (12-20 % fewer), about 1.1-1.2x tok/s at consumer RTTs** (`round_trip_model.py --mixed`: 0.60 -> 0.68-0.74
   tok/s at RTT 20 ms, one stream), at chat agreement around 0.9 (extrapolated from five models of 0.5B-4B; not
   measured; the trend is weak and two-point at the large end).
6. **Engine and protocol are done and exact** (section 5): any per-layer map runs on the fused schedule with
   `1 + 2S + P` trips, pinned in the pack manifest, item table v4 on the lane, masked == plain bit for bit, every value
   Freivalds-checked. At Qwen2.5-0.5B the exact integer pipeline of a mixed map matches the float conversion within
   0.5 % perplexity (section 4), so the conversion, not the fixed point, sets the quality.

**Verdict.** Training-free partial conversion is a small, real lever (one in six to one in eight trips at ~5 % perplexity
and ~10 % chat-token divergence) and the right *starting point* for healing: the near-additive per-layer cost gives the
map, and healing (other angles) decides whether a larger share becomes usable. Shipping it without healing is not
recommended for chat: the agreement drop is larger than any other approximation in the stack.

## 1. The change and what it costs

A sequential pre-norm block is `h1 = h + attn(n1 h); y = h1 + mlp(n2 h1)`: two non-linear stops, two fused trips.
Its parallel form reads the MLP input from the layer input, `y = h + attn(n1 h) + mlp(n2 h)`: one stop, one trip,
the same weights. With a per-layer map (`s` sequential, `p` parallel) a pass costs **`1 + 2S + P` trips** with the head
folded into the last trip (`S + P = L`; the `1` is layer 0's input trip, 0 with `local_qkv0`). All sequential:
`2L + 1` (fused2); all parallel: `L + 1`.

The only thing that changes for a converted layer is the MLP's input: `n2(h)` instead of `n2(h + attn)`. So the
damage is bounded by how much attention moves the residual *in the directions the MLP reads*.

## 2. Method (measured: Kaggle T4, kernel `bensliv/pgw-parconv-r2`, `benchmarks/parallel_conversion.py`)

Models (float16 on one T4; revisions recorded in each result file): Qwen2.5-0.5B / 1.5B / 3B-Instruct, Qwen3-1.7B / 4B
(chat template with thinking off). Per model, with a switch per layer (`post_attention_layernorm` reads the stashed
layer input when on; verified identical to the harness's `make_parallel`):

* **Sensitivity** (no evaluation): on 8 FineWeb-Edu documents (512 tokens) and 8 UltraChat `train_sft` prompts with the
  model's own answers, per layer: cosine and relative norm of `n2(h)` vs `n2(h + attn)`, and the relative change of
  the MLP output (`rel_mlp`, and `rel_out` relative to the layer's output residual).
* **Selection set** (for choosing maps): 16 held-out FineWeb-Edu documents x 512 tokens and 40 UltraChat `test_sft`
  prompts (not the report prompts) with the original's greedy answers (160 tokens). KL to the original over the
  original's top-64 tokens plus one "rest" bucket (a lower bound of the full KL), top-1 agreement, perplexity.
* **Orderings**, k = 0..L: by `rel_out`, by single-layer KL, last-first, first-last, and greedy marginal KL (shortlist
  of 3 by single-layer KL, half the selection set; for Qwen3-4B the time budget stopped the greedy search at k = 4 and
  the rest follows the single-layer order).
* **Report points** (the harness's metrics, `benchmarks/depth_prune_heal.py`; disjoint from the selection set): FineWeb-
  Edu perplexity (50 documents x 1024 tokens), chat top-1 agreement and full KL on 150 UltraChat `test_sft` prompts
  answered by the original (256 tokens), MMLU / ARC-Easy / ARC-Challenge / HellaSwag (200 each), at 25 / 50 / 75 / 100 %
  of the layers in the best ordering. 3B and 4B stopped after the 50 % / 25 % points (time budget).
* Data: FineWeb-Edu (ODC-By 1.0), UltraChat 200k prompts (MIT; answers by the model itself), MMLU (MIT), ARC (CC BY-SA
  4.0), HellaSwag (MIT); evaluation only, nothing trained. Comparability: same harness functions and data sources as
  the depth / healing runs, but float16 weights and 50 perplexity documents (the 0.5B original reads 14.34 here vs
  13.57 in `kaggle_05b_r1`, float32 weights with autocast and 107 documents): compare deltas, not absolute values.

Results: `benchmarks/results/parallel_conversion/kaggle_t4_r2/` (one JSON per model with every curve, the sensitivity
and single-layer tables and the report points; `log.txt`). Tables below: `python
research/reviews/parallel_conversion_numbers.py`.

## 3. Results

### Sensitivity vs measured single-layer KL (Spearman over layers)

| model | L | rel_out | rel_mlp | rel_in | 1 - cos_in | most tolerant layers (single-layer KL) | least tolerant |
|---|---|---|---|---|---|---|---|
| qwen2.5-0.5b | 24 | 0.77 | 0.89 | 0.76 | 0.68 | 22 (0.004), 16 (0.007), 18 (0.007), 19 (0.011), 7 (0.020), 6 (0.020) | 0 (3.27), 1 (1.28), 23 (0.62) |
| qwen2.5-1.5b | 28 | 0.67 | 0.70 | 0.59 | 0.54 | 22 (0.008), 24 (0.009), 25 (0.011), 14 (0.013), 6 (0.014), 23 (0.015) | 0 (8.39), 1 (0.93), 27 (0.36) |
| qwen3-1.7b | 28 | 0.94 | 0.81 | 0.53 | 0.54 | 24 (0.004), 23 (0.005), 25 (0.007), 22 (0.008), 21 (0.010), 20 (0.026) | 0 (3.07), 2 (0.11), 6 (0.10) |
| qwen2.5-3b | 36 | 0.61 | 0.52 | 0.63 | 0.53 | 5 (0.002), 3 (0.003), 30 (0.005), 29 (0.005), 2 (0.005), 7 (0.006) | 1 (7.04), 0 (1.09), 35 (0.20) |
| qwen3-4b | 36 | 0.78 | 0.74 | 0.76 | 0.76 | 32 (0.004), 27 (0.004), 31 (0.004), 29 (0.005), 26 (0.005), 28 (0.005) | 0 (0.62), 35 (0.07), 12 (0.05) |

### Orderings: mean selection KL over k = 0, 2, 4, ..., L (lower is better)

| model | sensitivity | single_kl | greedy | late_first | early_first |
|---|---|---|---|---|---|
| qwen2.5-0.5b | 1.868 | 1.517 | 1.505 | 2.372 | 8.550 |
| qwen2.5-1.5b | 1.090 | 1.040 | 1.033 | 1.274 | 8.369 |
| qwen3-1.7b | 0.743 | 0.735 | 0.724 | 0.806 | 5.040 |
| qwen2.5-3b | 0.868 | 0.709 | 0.688 | 1.098 | 7.339 |
| qwen3-4b | 0.788 | 0.751 | 0.751 | 0.994 | 5.016 |

### Training-free budget: most layers converted (greedy order) within a selection-KL budget

| model | L | trips 2L+1 | KL <= 0.02: k / trips / chat top-1 / ppl x | KL <= 0.05: k / trips / chat top-1 / ppl x | KL <= 0.1: k / trips / chat top-1 / ppl x | KL <= 0.2: k / trips / chat top-1 / ppl x |
|---|---|---|---|---|---|---|
| qwen2.5-0.5b | 24 | 49 | 3 / 46 / 0.936 / 1.022 | 4 / 45 / 0.914 / 1.033 | 6 / 43 / 0.854 / 1.071 | 9 / 40 / 0.803 / 1.195 |
| qwen2.5-1.5b | 28 | 57 | 2 / 55 / 0.947 / 1.016 | 4 / 53 / 0.909 / 1.043 | 7 / 50 / 0.859 / 1.083 | 11 / 46 / 0.805 / 1.173 |
| qwen3-1.7b | 28 | 57 | 3 / 54 / 0.948 / 1.030 | 5 / 52 / 0.928 / 1.047 | 7 / 50 / 0.880 / 1.043 | 10 / 47 / 0.834 / 1.048 |
| qwen2.5-3b | 36 | 73 | 4 / 69 / 0.941 / 1.013 | 7 / 66 / 0.905 / 1.025 | 13 / 60 / 0.859 / 1.062 | 20 / 53 / 0.790 / 1.155 |
| qwen3-4b | 36 | 73 | 4 / 69 / 0.955 / 1.010 | 9 / 64 / 0.918 / 1.050 | 12 / 61 / 0.899 / 1.081 | 17 / 56 / 0.859 / 1.098 |

### Quality vs trips (greedy order, selection set): k / trips / KL / chat top-1 / web ppl

* **qwen2.5-0.5b** (L = 24): 0 / 49 / 0.000 / 1.00 / 15.9; 2 / 47 / 0.011 / 0.95 / 16.1; 4 / 45 / 0.030 / 0.91 / 16.4; 6 / 43 / 0.076 / 0.85 / 17.0; 8 / 41 / 0.135 / 0.81 / 18.2; 10 / 39 / 0.231 / 0.78 / 20.0; 12 / 37 / 0.329 / 0.73 / 22.1; 14 / 35 / 0.455 / 0.68 / 25.1; 16 / 33 / 0.604 / 0.63 / 29.1; 18 / 31 / 0.845 / 0.58 / 36.0; 20 / 29 / 1.418 / 0.46 / 62.9; 22 / 27 / 3.176 / 0.17 / 314.6; 24 / 25 / 12.247 / 0.01 / 2556916.9
* **qwen2.5-1.5b** (L = 28): 0 / 57 / 0.000 / 1.00 / 11.5; 2 / 55 / 0.017 / 0.95 / 11.7; 4 / 53 / 0.043 / 0.91 / 12.0; 6 / 51 / 0.073 / 0.87 / 12.2; 8 / 49 / 0.108 / 0.85 / 12.6; 10 / 47 / 0.155 / 0.82 / 13.2; 12 / 45 / 0.209 / 0.79 / 13.8; 14 / 43 / 0.264 / 0.76 / 14.5; 16 / 41 / 0.345 / 0.74 / 15.7; 18 / 39 / 0.418 / 0.72 / 17.0; 20 / 37 / 0.524 / 0.68 / 18.6; 22 / 35 / 0.654 / 0.64 / 20.9; 24 / 33 / 0.920 / 0.59 / 28.0; 26 / 31 / 1.461 / 0.46 / 47.1; 28 / 29 / 10.302 / 0.03 / 1060550.1
* **qwen3-1.7b** (L = 28): 0 / 57 / 0.000 / 1.00 / 18.2; 2 / 55 / 0.009 / 0.96 / 18.4; 4 / 53 / 0.026 / 0.94 / 18.9; 6 / 51 / 0.064 / 0.90 / 18.7; 8 / 49 / 0.119 / 0.86 / 19.3; 10 / 47 / 0.174 / 0.83 / 19.0; 12 / 45 / 0.235 / 0.81 / 19.4; 14 / 43 / 0.305 / 0.78 / 19.8; 16 / 41 / 0.386 / 0.76 / 20.1; 18 / 39 / 0.489 / 0.74 / 21.5; 20 / 37 / 0.620 / 0.70 / 23.6; 22 / 35 / 0.770 / 0.66 / 29.0; 24 / 33 / 0.981 / 0.64 / 35.8; 26 / 31 / 1.364 / 0.57 / 49.5; 28 / 29 / 5.311 / 0.18 / 2078.8
* **qwen2.5-3b** (L = 36): 0 / 73 / 0.000 / 1.00 / 10.3; 3 / 70 / 0.011 / 0.95 / 10.4; 6 / 67 / 0.033 / 0.92 / 10.5; 9 / 64 / 0.061 / 0.88 / 10.7; 12 / 61 / 0.089 / 0.87 / 10.9; 15 / 58 / 0.113 / 0.85 / 11.2; 18 / 55 / 0.163 / 0.82 / 11.5; 21 / 52 / 0.221 / 0.78 / 12.1; 24 / 49 / 0.308 / 0.75 / 13.1; 27 / 46 / 0.415 / 0.71 / 14.3; 30 / 43 / 0.570 / 0.66 / 16.3; 33 / 40 / 0.925 / 0.59 / 22.7; 36 / 37 / 8.331 / 0.00 / 26515.2
* **qwen3-4b** (L = 36): 0 / 73 / 0.000 / 1.00 / 15.2; 3 / 70 / 0.012 / 0.96 / 15.4; 6 / 67 / 0.027 / 0.94 / 15.6; 9 / 64 / 0.050 / 0.92 / 16.0; 12 / 61 / 0.092 / 0.90 / 16.5; 15 / 58 / 0.146 / 0.87 / 16.5; 18 / 55 / 0.215 / 0.85 / 16.2; 21 / 52 / 0.294 / 0.82 / 16.8; 24 / 49 / 0.431 / 0.78 / 17.9; 27 / 46 / 0.571 / 0.74 / 19.9; 30 / 43 / 0.847 / 0.69 / 25.2; 33 / 40 / 1.387 / 0.62 / 47.0; 36 / 37 / 7.503 / 0.12 / 8375.1

### Report points (held-out harness sets, full KL; best ordering)

| model | k (share) | trips | FineWeb ppl | chat top-1 | chat KL | MMLU | ARC-E | ARC-C | HellaSwag |
|---|---|---|---|---|---|---|---|---|---|
| qwen2.5-0.5b | 0 (0 %) | 49 | 14.34 | 1.000 | -0.000 | 0.475 | 0.630 | 0.345 | 0.585 |
| qwen2.5-0.5b | 6 (25 %) | 43 | 15.34 | 0.866 | 0.089 | 0.455 | 0.600 | 0.310 | 0.565 |
| qwen2.5-0.5b | 12 (50 %) | 37 | 20.03 | 0.738 | 0.358 | 0.325 | 0.555 | 0.270 | 0.510 |
| qwen2.5-0.5b | 18 (75 %) | 31 | 32.57 | 0.591 | 0.946 | 0.260 | 0.455 | 0.280 | 0.425 |
| qwen2.5-0.5b | 24 (100 %) | 25 | 1730317.98 | 0.008 | 13.637 | 0.215 | 0.285 | 0.290 | 0.215 |
| qwen2.5-1.5b | 0 (0 %) | 57 | 10.41 | 1.000 | -0.000 | 0.575 | 0.750 | 0.455 | 0.735 |
| qwen2.5-1.5b | 7 (25 %) | 50 | 11.29 | 0.864 | 0.099 | 0.515 | 0.760 | 0.480 | 0.730 |
| qwen2.5-1.5b | 14 (50 %) | 43 | 13.31 | 0.774 | 0.301 | 0.460 | 0.680 | 0.420 | 0.690 |
| qwen2.5-1.5b | 21 (75 %) | 36 | 18.19 | 0.666 | 0.662 | 0.360 | 0.545 | 0.390 | 0.600 |
| qwen2.5-1.5b | 28 (100 %) | 29 | 404073.12 | 0.021 | 10.085 | 0.220 | 0.295 | 0.280 | 0.255 |
| qwen3-1.7b | 0 (0 %) | 57 | 15.68 | 1.000 | -0.000 | 0.535 | 0.695 | 0.425 | 0.635 |
| qwen3-1.7b | 7 (25 %) | 50 | 16.24 | 0.885 | 0.127 | 0.535 | 0.665 | 0.420 | 0.655 |
| qwen3-1.7b | 14 (50 %) | 43 | 17.24 | 0.783 | 0.395 | 0.485 | 0.530 | 0.345 | 0.595 |
| qwen3-1.7b | 21 (75 %) | 36 | 22.50 | 0.697 | 0.774 | 0.350 | 0.460 | 0.315 | 0.485 |
| qwen3-1.7b | 28 (100 %) | 29 | 1738.72 | 0.217 | 5.276 | 0.225 | 0.315 | 0.180 | 0.330 |
| qwen2.5-3b | 0 (0 %) | 73 | 9.26 | 1.000 | 0.000 | 0.625 | 0.745 | 0.465 | 0.810 |
| qwen2.5-3b | 9 (25 %) | 64 | 9.69 | 0.884 | 0.075 | 0.620 | 0.775 | 0.500 | 0.785 |
| qwen2.5-3b | 18 (50 %) | 55 | 10.53 | 0.814 | 0.197 | 0.565 | 0.665 | 0.365 | 0.755 |
| qwen3-4b | 0 (0 %) | 73 | 13.27 | 1.000 | -0.000 | 0.660 | 0.805 | 0.555 | 0.685 |
| qwen3-4b | 9 (25 %) | 64 | 13.89 | 0.922 | 0.069 | 0.655 | 0.780 | 0.505 | 0.690 |

### Best training-free map per model (greedy order, selection KL <= 0.1)

* qwen2.5-0.5b: k = 6, 43 trips (from 49), parallel layers [7, 13, 16, 18, 19, 22], map `ssssssspssssspsspsppssps`
* qwen2.5-1.5b: k = 7, 50 trips (from 57), parallel layers [5, 6, 14, 19, 22, 24, 25], map `sssssppssssssspsssspsspsppss`
* qwen3-1.7b: k = 7, 50 trips (from 57), parallel layers [16, 20, 21, 22, 23, 24, 25], map `sssssssssssssssspsssppppppss`
* qwen2.5-3b: k = 13, 60 trips (from 73), parallel layers [2, 3, 5, 7, 8, 9, 10, 13, 25, 26, 28, 29, 30], map `ssppspsppppsspsssssssssssppspppsssss`
* qwen3-4b: k = 12, 61 trips (from 73), parallel layers [2, 3, 4, 25, 26, 27, 28, 29, 30, 31, 32, 33], map `sspppsssssssssssssssssssspppppppppss`

## 4. The exact integer pipeline on a mixed map (measured, this box, Qwen2.5-0.5B-Instruct)

`benchmarks/fused2_quality.py --configs f2i8mighead8:K56A12C53 mix8=<map>:K56A12C53` (the production derivation: int8 fold
from the pack, 8-bit migration, folded head, local qkv0), 2044 wikitext-2 test tokens, maps = the 4 / 8 / 12 lowest
single-layer-KL layers of a local CPU sweep (wikitext; `benchmarks/results/parallel_conversion/local_cpu/`):

| map (parallel layers) | trips | float32 HF conversion ppl | exact mixed pipeline ppl | top-1 vs float original | max \|H\| / \|Z\| bits |
|---|---|---|---|---|---|
| none (fused2) | 49 | 19.22 | 19.19 | 0.960 | 41.3 / 48.4 |
| 16, 18, 19, 22 | 45 | 20.06 | 20.00 | 0.890 | 40.2 / 47.3 |
| + 3, 6, 7, 13 | 41 | 22.27 | 22.29 | 0.785 | 39.3 / 46.6 |
| + 9, 14, 15, 17 | 37 | 27.88 | 27.77 | 0.666 | 39.3 / 46.6 |

Every trip row kept 12 activation bits on ring K56A12C53. The fixed point adds nothing measurable on top of the
conversion. (The local CPU sweep was cut short by a container restart after its single-layer pass; its sensitivity and
single-layer numbers agree with the Kaggle run on the most tolerant layers: 22, 18, 16, 19.)


## 5. Engine and protocol (implemented, opt-in, default off)

* **`pgw.engine.fused_mixed`**: `MixedWeights(pack, layer_map, bits=8, head=True)` and `MixedFusedSchedule`
  (`enable_fused_mixed(model, weights, ring, head_fold=True, local_qkv0=...)`). Per layer: sequential = fused2's two
  pair trips (`o` + residual -> `gate_up`; `down` + residual -> next input stage); parallel = one pair trip
  (`[o | down]` + residual -> next input stage). The next input stage is `qkv` (norm gain folded) before a sequential
  layer and the stacked `[qkv ; gate_up]` (both gains folded) before a parallel one; the residual-grid migration of
  each trip's first linear is taken over the gains of the stage that follows it. An all-`s` map gives fused2's tables
  and outputs bit for bit, an all-`p` map the parallel prototype's (tested), so nothing existing changes.
* **Pinned map**: `ArchConfig.parallel_layers` (sorted layer indices) in the pack manifest, covered by the pack
  fingerprint; absent and fingerprint-neutral when unset (every existing pack keeps its fingerprint; tested on
  in-memory and on-disk packs). `pack_checkpoint(..., parallel_layers=[...])` writes it. A pack that declares a map is
  refused by every schedule that does not compute exactly that map (standard, fused2, parallel prototype, a mixed
  schedule with another map): it can never run silently as sequential blocks. `MixedWeights(pack)` takes the
  declared map; an explicit map must equal it.
* **Lane**: item table **v4** (`pgw.service.fused2_lane.mixed_items`): layer 0's input single, per sequential layer the
  pairs `(o, gate_up)` and `(down, next input)`, per parallel layer `([o | down], next input)`, the last trip and the
  folded head's pair. The ticket offers `fused2_items = 4`; the gateway checks the version and every item name (so
  the map) and the lane fingerprint binds the item table and every item's public weights; v1-v3 peers are refused
  (fail closed).
* **Privacy / integrity**: unchanged from fused2. The provider gets `(R + v - hmul W1 r_x, x + r_x)` per trip, uniform
  for any data; each mask row is used once; every returned `H`, `Z` and logit row passes the range check and the
  Freivalds check over `F_(2^61-1)` before use. The map is public: the provider already sees the trip count and item
  shapes.
* **Round-trip model**: `benchmarks/round_trip_model.py --consumer --layer-map 40%,sspp...` adds mixed schedules
  (a share of parallel layers or an explicit map); `--mixed` prints trips and tok/s vs the share for 0.5B-72B and
  Qwen3-14B.
* **Not done**: the native fused2 decode driver does not run mixed maps (the Python fused path does; the driver is
  not engaged for this schedule), the client library does not expose the schedule (no pinned mixed pack yet), the
  shard-wise packer does not take `parallel_layers`, and the K56 mask factory's encrypted path was not exercised for
  mixed keys (the in-process factory and the lane were).

**Tests** (this box, one directory at a time, absolute `PYTHONPATH`): `tests/engine/test_fused_mixed.py` (95: on 4-layer
Qwen2 / Llama / Qwen3 packs, all-`s` == fused2 and all-`p` == parallel prototype bit for bit; masked == plain bit for
bit for five mixed maps with and without the folded head; `1 + 2S + P` trips per decode pass; 2 checks per pair trip;
no masked payload row ever repeats; every provider value a uniform ring element; prefilled mask stock and local
qkv0; tamper of `H`, `Z`, consistent `H`, dropped residual delta on a parallel trip, a sequential `down` trip and a
sequential `o` trip all raise `IntegrityError`; close to the float64 reference of the same map; the declared map is
pinned and enforced), `tests/engine/test_fused_mixed_lane.py` (4: item table v4 over a real `LaneManager`, masked ==
plain, exactly `4 x (1 + 2S + P)` lane calls for 4 passes, no row sent twice, tamper caught, another map has another
lane fingerprint), `tests/modelpack/test_parallel_layers.py` (2), `tests/benchmarks/test_round_trip_model.py` (+2).
Suites: `tests/engine` 1389 passed, 15 skipped (CUDA); `tests/modelpack` 131 passed, 1 skipped;
`tests/service -k "lane or multi or fused"` 121 passed.
