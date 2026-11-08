# Two novel signals: two-reader consensus, and unit-score dynamics

**Questions.**
1. Does *disagreement* between two different frozen readers (Qwen3-1.7B and Qwen3-0.6B) about the same reply tell
   us whether the reply is wrong?
2. Do features of the *sequence* of sentence/step scores (jumps, slope, variance, ...) add to the worst unit? And does
   "the first unit where the score jumps" find the first wrong ProcessBench step better than the argmax?

**Answers.**
1. **No.** Every disagreement score sits at chance (0.50-0.53). Adding them to the 1.7B stack *lowers* AUROC. The two
   readers' representations are almost the same: the top canonical correlations are 0.97, 0.96, 0.96... in every
   domain. A concatenated head (both readers) gains +0.005 [+0.002, +0.008] in the zero-label sum. That is below the
   pre-stated +0.010 bar.
2. **Detection: no** (all 21 domains). A LODO logistic on 10 sequence features ties the worst unit (−0.001). Adding it
   to the stack gives +0.004 [+0.000, +0.007]. On **long-form** it gives +0.018 [+0.009, +0.027], which passes the bar
   for that format, but it is one hit among 72 format-level tests.
   **Localisation: yes.** "First unit whose z-score crosses τ" (τ chosen leave-one-PB-domain-out) finds the first
   wrong step 47% of the time against 34% for the argmax: +0.13 [+0.08, +0.18]. Within ±1 step it reaches 73% against
   58%. "First jump" is in between: +0.05 [+0.00, +0.10].

- Script: `experiments/novel_consensus_dynamics.py`, stored features only, no model loaded, CPU on 1 thread, 10 min.
- Results: `results/omni/novel_consensus_dynamics.json`.

## Protocol

**Data.**
- 5,107 replies, 21 domains, 4 formats.
- Features of both readers come from `/home/user/omni/q17` and `/home/user/omni/q06`. Reply ids and units match.

**Heads.** As in `experiments/omni_final.py`:
- Reply head: middle-layer mean and judge states, PCA 256, output signals and judge log-odds. LODO, with
  per-domain standardisation.
- Unit head: per unit, trained on 16k labelled units, LODO. The worst unit is its maximum.
- Judge: −log-odds of "Yes".
- **Nested LODO for every learned combiner.** For held-out domain d, the scores of a training row in domain e come
  from heads that saw neither d nor e. Nothing about d leaks through stacked features.

**Baselines reproduced:**

| baseline | AUROC |
|---|---|
| reply | 0.710 |
| worst unit | 0.711 |
| C0 = z(reply) + z(worst) + z(judge) | 0.727 |
| S0 = nested LODO logistic on the same three | 0.730 |

**CIs.** 1,000 bootstrap resamples of replies within each domain, stratified by label, paired across methods. Every
AUROC is computed within a domain and then averaged over the domains.

**Rules, fixed in the script before the run:**
- **R1, signal:** the mean AUROC (a-priori sign) has a 95% CI lower bound above 0.50.
- **R2, useful:** Δ ≥ +0.010 over the matched baseline, with the CI above 0. This holds overall, and a format-level
  claim needs the same within that format.
- **R3, localisation:** the paired CI of the hit-rate difference against argmax is above 0.

## (1) Multi-reader consensus

| score | all | chat (2) | long-form (5) | RAG (10) | reasoning (4) |
|---|---|---|---|---|---|
| reply 0.6B | 0.685 | 0.532 | 0.708 | 0.662 | 0.790 |
| \|z(reply 0.6B) − z(reply 1.7B)\| | 0.495 [0.479, 0.514] | 0.500 | 0.500 | 0.490 | 0.499 |
| \|Δ worst unit\| | 0.504 [0.488, 0.521] | 0.493 | 0.529 | 0.509 | 0.467 |
| \|Δ judge\| | 0.531 [0.515, 0.549] | 0.521 | 0.526 | 0.519 | 0.574 |
| mean \|Δ unit score\| over units | 0.505 [0.489, 0.522] | 0.468 | 0.506 | 0.512 | 0.504 |
| CCA discrepancy (32 comps, fitted on other domains, no labels) | 0.499 [0.484, 0.516] | 0.479 | 0.527 | **0.443** | **0.617** |
| ridge residual 1.7B → 0.6B | 0.506 [0.491, 0.524] | 0.483 | 0.518 | **0.452** | **0.639** |
| LODO logistic on the 32 CCA discrepancies | 0.510 [0.493, 0.528] | 0.496 | 0.491 | 0.533 | 0.487 |

**The CCA and ridge rows flip sign between formats.** Wrong reasoning replies sit off the readers' usual agreement,
while wrong RAG replies sit *on* it. A LODO logistic cannot learn this sign, because it is not shared across formats.

**Adding to the 1.7B stack** (Δ against the matched baseline, [95% CI]):

| variant | all | chat | long-form | RAG | reasoning |
|---|---|---|---|---|---|
| C0 + z(\|Δ reply\|) | −0.019 [−0.026, −0.012] | −0.003 | −0.021 | −0.018 | −0.028 |
| C0 + z(\|Δ reply\| + \|Δ worst\| + \|Δ judge\|) | −0.014 [−0.021, −0.008] | −0.004 | −0.016 | −0.015 | −0.016 |
| C0 + z(CCA) | −0.012 [−0.018, −0.005] | −0.015 | +0.002 | −0.024 | +0.001 |
| C0 + the 0.6B experts (6-expert sum) | −0.011 [−0.018, −0.005] | −0.034 | −0.009 | −0.010 | −0.006 |
| C0 with the concat-reader reply head | **+0.005 [+0.002, +0.008]** | +0.004 | +0.010 [+0.002, +0.018] | +0.005 | −0.003 |
| concat reply head vs 1.7B reply head | +0.004 [−0.004, +0.011] | +0.000 | +0.011 | +0.008 | −0.013 |
| S0 + 0.6B scores (stacker) | +0.000 [−0.004, +0.004] | −0.012 | +0.010 | +0.001 | −0.007 |
| S0 + 0.6B + disagreement + CCA | −0.001 vs S0+0.6B | | | | |
| S0 + disagreement + CCA | −0.002 [−0.003, −0.001] | | | | |

**Verdict.**
- **R1:** only |Δ judge| passes R1, at 0.531.
- **R2:** nothing passes. Long-form for the concat head reaches exactly +0.010, but nothing else does.
- **Interpretation:** a 0.6B reader from the same family is a near-copy of the 1.7B one, only weaker, so their
  disagreement is mostly the noise of the weaker reader. A real test of consensus needs a *different family*. The
  stack already has one in Qwen3-14B and phi-4 as judges. The circuit study found that escalating judge/internal
  disagreement did not help either.

## (2) Unit-score dynamics (1.7B unit head, LODO)

Standalone, a-priori sign (higher means wrong):

| feature | all | chat | long-form | RAG | reasoning |
|---|---|---|---|---|---|
| worst unit (max) | **0.711** | 0.608 | **0.770** | **0.653** | 0.833 |
| mean | 0.696 | 0.599 | 0.725 | 0.649 | 0.826 |
| fraction above 0 | 0.677 | 0.583 | 0.719 | 0.620 | 0.816 |
| count above 0 | 0.676 | 0.562 | 0.727 | 0.622 | 0.805 |
| variance | 0.570 | 0.510 | 0.594 | 0.532 | 0.664 |
| max − median | 0.552 | 0.495 | 0.594 | 0.519 | 0.610 |
| max jump between consecutive units | 0.541 | 0.491 | 0.552 | 0.515 | 0.619 |
| slope | 0.505 | 0.524 | 0.470 | 0.483 | 0.594 |
| position of max | 0.487 | 0.513 | 0.475 | 0.469 | 0.534 |
| log #units | 0.515 | 0.473 | 0.525 | 0.517 | 0.521 |
| **LODO logistic on all 10** | 0.710 | 0.604 | 0.763 | 0.652 | **0.840** |

**Δ against the matched baseline [95% CI]:**

| variant | all | chat | long-form | RAG | reasoning |
|---|---|---|---|---|---|
| dynamics logistic vs worst unit | −0.001 [−0.005, +0.004] | −0.004 | −0.007 | −0.001 | +0.008 [−0.001, +0.016] |
| max jump vs worst unit | −0.170 | | | | |
| C0 + z(dynamics logistic) | +0.004 [+0.000, +0.007] | +0.005 | **+0.018 [+0.009, +0.027]** | −0.004 | +0.005 [+0.001, +0.010] |
| C0, worst unit replaced by dynamics logistic | +0.001 [−0.001, +0.003] | | | | |
| C0 + z(max jump) | −0.017 [−0.023, −0.012] | | | | |
| S0 + dynamics features (stacker) | −0.000 [−0.004, +0.003] | +0.002 | +0.009 [+0.001, +0.018] | −0.006 | +0.002 |

**Verdict.**
- **For detection, the sequence shape adds nothing beyond its maximum.** Jump, slope, position and variance are weak
  alone, from 0.49 to 0.57, and they dilute the stack.
- **Long-form is the one format where it passes R2:** +0.018 in the zero-label sum. Treat it as a lead, not a result.
  - It is one pass among 72 format-level tests.
  - The stacker version gives only +0.009.

## (2b) Localising the first wrong step (ProcessBench, 480 wrong replies, 1.7B unit head, LODO)

**Setup.**
- Unit scores are z-scored per domain without labels.
- τ grid: 0 to 3 in steps of 0.25, chosen per held-out PB domain on the other three.
- If no unit fires, the rule falls back to the argmax.
- First jump: the first step i with z_i − z_{i−1} > τ, where z_{−1} = 0.
- First crossing: the first step with z_i > τ.

**Hit rates:**

| rule | top-1 hit [95% CI] | within ±1 step | GSM8K / MATH / Olympiad / Omni (top-1) |
|---|---|---|---|
| chance (1/#steps) | 0.17 | | 0.21 / 0.18 / 0.14 / 0.14 |
| argmax | 0.342 [0.300, 0.381] | 0.579 | 0.46 / 0.30 / 0.34 / 0.27 |
| first jump (τ 0.75-1.25) | 0.394 [0.352, 0.440] | 0.644 | 0.43 / 0.41 / 0.38 / 0.35 |
| **first crossing (τ 0.5-0.75)** | **0.471 [0.429, 0.515]** | **0.733** | 0.48 / 0.48 / 0.45 / 0.48 |

**Differences from argmax:**

| comparison | top-1 | within ±1 step |
|---|---|---|
| first jump − argmax | +0.052 [+0.002, +0.100] | +0.065 [+0.021, +0.108] |
| first crossing − argmax | **+0.129 [+0.079, +0.177]** | **+0.154 [+0.108, +0.202]** |

**Verdict.** R3 passes for both rules, so a sequence rule does beat the argmax for localisation.
- **Why it works:** after the first wrong step, later steps also score high and often higher, because errors
  compound. The argmax therefore lands too late. The *first* step that crosses a modest threshold is the better
  pointer.
- **The win is cheap and should be shipped.** Change the API's "suspect unit" for reasoning replies from the argmax
  to the first step above τ ≈ 0.75 z.
- **Only the level matters.** The plain crossing beats the jump rule, so the step-to-step difference adds nothing.

## Caveats

- Only two readers were tested, both from the same family (Qwen3). Consensus with a different-family reader at
  feature level was not measured here.
- The ProcessBench localisation uses only the first-wrong-step labels. Steps after it are unlabelled (−1) and were
  not scored.
- The τ grid and both rules were fixed before the run. The comparison of the two rules is post hoc.
