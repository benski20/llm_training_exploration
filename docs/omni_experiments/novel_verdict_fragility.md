# Verdict fragility: steering and patching the 1.7B reader's judge state (causal internals)

**Question.** The reader (Qwen3-1.7B) appends a judge question after the reply and reads its own verdict
j = log P(Yes) − log P(No) at the last position. If we *intervene* on the residual stream there, does how easily the
verdict moves say whether the reply is wrong, beyond the verdict itself and the reply head?

Script: `experiments/novel_verdict_fragility.py`. Results: `results/omni/novel_verdict_fragility.json`.

## Setup

- **Replies.** 840 label-balanced replies, 20 wrong and 20 correct from each of the 21 corpus domains (seed 0),
  max_tokens 1024. CPU, bf16, 2 threads, ≈ 7 s per reply. The Yes/No readout is in fp32 from the final normed
  state, because bf16 logits are quantised to 0.125-0.25 at these magnitudes. The cached-prefix j(0) matches the
  stored `j_logit` to within ±0.3, except where truncation differs.
- **Directions.** Each is unit norm and points toward "wrong". It is added to the output of block L−1
  (= hidden_states[L]) at the judge position only.
  - `dm20` / `dm15`: the difference of class means of the stored judge state at layer 20 / 15, **leave-one-domain-out**
    (the mean over the 20 other domains of mean(wrong) − mean(correct)). The held-out domain's states and labels
    are never used.
  - `head`: the shipped head's weights mapped back through its PCA and standardisation to the layer-20 judge block.
    **Leaky** (the head saw all 21 domains), so it is a reference only. Its cosine with dm20 is ≈ 0.26.
  - α is in units of the projection sd, from the training domains.
- **Interventions.** The prefix is KV-cached, so each intervention reruns one token:
  - steering: α ∈ {±1, ±2, ±4, ±8} for dm20 and {±2, ±4} for the others;
  - **counterfactual patch**: the same reply is read with the user turn emptied (no documents, no question). Its
    layer-L judge state replaces the real one (`cf_patch20/15`). `cf_full` is the counterfactual run's own verdict.
- **Signals.** Each sign was stated a priori, with larger meaning wrong:
  - `flip`: −α\*, the dm20 push at which the verdict crosses 0;
  - `fragility`: −|α\*|;
  - `sens`: (j(−2) − j(+2))/4;
  - `asym`: [j(0) − j(+a)] − [j(−a) − j(0)];
  - `cf_*`: j_patched − j(0).
- **Pre-stated rules.** The bootstrap uses 1,000 paired, within-domain, label-stratified resamples.
  - R1: the mean per-domain AUROC has a CI lower bound > 0.5.
  - R2 (new information): z(judge0) + z(signal) beats z(judge0) by ≥ +0.010, with the CI above 0.
  - R3 (adopt): C0 = z(LODO reply head) + z(judge0) gains ≥ +0.010 with the CI above 0, both for the zero-label
    sum and for a LODO logistic stacker.

## The intervention works mechanically

Mean j over the 840 replies, by dm20 push α:

| α | −8 | −4 | −2 | −1 | 0 | +1 | +2 | +4 | +8 |
|---|---|---|---|---|---|---|---|---|---|
| mean j | 23.1 | 19.4 | 15.9 | 13.5 | 10.6 | 7.2 | 3.6 | −3.2 | −12.2 |

- The response is monotone in 100% of replies.
- 64% of verdicts flip within 4 sd.
- One sd of the LODO direction (norm ≈ 30) moves the verdict by about 3.4 logits. In the smoke check on 3
  replies, a random direction pushed 2.7× as far (norm 80) moved it by at most 1 logit.
- **The direction is causally potent for the verdict.**

## Results (mean of 21 per-domain AUROCs, 95% CI)

| signal | AUROC [CI] | chat | long-form | RAG | reasoning | R2 Δ vs judge0 [CI] | R3 Δ vs C0, sum / stack |
|---|---|---|---|---|---|---|---|
| judge0 (−j(0), baseline) | 0.648 [0.611, 0.684] | 0.63 | 0.67 | 0.58 | 0.80 | | |
| flip (−α\*) | 0.636 [0.597, 0.670] | 0.63 | 0.67 | 0.56 | 0.80 | −0.004 [−0.013, +0.005] | −0.018 / −0.003 |
| fragility (−\|α\*\|) | 0.638 [0.599, 0.675] | 0.63 | 0.66 | 0.56 | 0.80 | −0.004 [−0.016, +0.006] | −0.018 / −0.000 |
| sens20 | 0.619 [0.582, 0.656] | 0.62 | 0.65 | 0.54 | 0.78 | −0.004 [−0.022, +0.013] | −0.015 / +0.000 |
| asym20 (α 2) | 0.463 [0.422, 0.500] | 0.39 | 0.53 | 0.45 | 0.44 | −0.089 | −0.002 / +0.001 |
| asym20 (α 4) | **0.419** [0.380, 0.454] | 0.41 | 0.48 | 0.44 | 0.29 | −0.107 | +0.008 / +0.002 |
| sens15 / asym15 | 0.457 / 0.515 | | | | | −0.079 / −0.044 | −0.024 / −0.017 |
| cf_patch20 | 0.535 [0.497, 0.571] | 0.48 | 0.51 | **0.62** | 0.39 | −0.054 (RAG +0.024 [−0.005, +0.054]) | −0.035 / −0.003 |
| cf_patch15 | 0.489 [0.450, 0.526] | | | | | −0.070 | −0.036 / −0.002 |
| cf_full | 0.527 [0.491, 0.567] | 0.49 | 0.50 | 0.61 | 0.37 | −0.055 (RAG +0.026 [−0.001, +0.055]) | −0.038 / −0.001 |
| head_sens / head_asym (leaky) | 0.562 / 0.495 | | | | | | |
| reply head (LODO 1.7B) | 0.712 [0.681, 0.745] | 0.56 | 0.70 | 0.67 | 0.91 | | |
| C0 sum / LODO stacker | 0.701 / 0.709 | | | | | | |
| stacker + all 10 causal signals | | | | | | | **−0.021** [−0.037, −0.005] |

## Reading

1. **Fragility is the verdict, re-expressed.**
   - The flip point α\* correlates with j(0) at Spearman 0.91, fragility at 0.90 and sens20 at 0.68.
   - Their AUROCs sit just below judge0's, and they add nothing to it (R2 Δ ≈ −0.004).
   - The slope is nearly uniform across replies, so "how far you must push" ≈ "how far from 0 you started".
   - A reply the model "knows" is right or wrong is not more robust than its distance from the boundary implies.
2. **The asymmetry hypothesis is refuted, and its sign is reversed.**
   - With the a-priori sign, α 4 gives AUROC 0.42: pushes toward "wrong" move *correct* replies' verdicts more.
   - This is a saturation artefact: it correlates with judge0 at Spearman −0.78.
   - With its sign flipped after the fact, it still adds nothing over judge0 (−0.016).
3. **The counterfactual patch is a RAG-only weak signal.**
   - Removing the documents and question pushes the verdict up when the context argued against the reply:
     AUROC 0.62 on RAG.
   - On reasoning it is anti-informative (0.37–0.39): removing the problem statement makes the model doubt
     correct solutions.
   - On RAG it adds +0.024 to judge0 (CI touches 0) and nothing to C0 (−0.007).
4. **Rules.**
   - R1: only flip, fragility and sens20 pass (as copies of judge0).
   - R2: no signal passes.
   - R3: no signal passes, in any format.
   - Stacking all 10 causal signals *lowers* the stacker by 0.021.

**Verdict: no.** The judge-position "wrong" direction is causally real: a ±1 sd push shifts the verdict by ±3.4
logits, and the response is monotone in every reply. But the verdict's fragility under it carries no
information beyond the verdict itself. Not adopted. This agrees with the saturated-combination finding in
`docs/omni_verifier.md`: new accuracy has to come from new evidence (judges, NLI, recompute), not from re-reading
the 1.7B reader's own judgement.
