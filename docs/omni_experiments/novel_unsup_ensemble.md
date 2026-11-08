# Zero-label expert weights from expert agreement (`experiments/novel_unsup_ensemble.py`)

**Question.** On a new domain with no labels, can we tell how accurate each expert is just from how the experts
agree on the domain's unlabelled replies? If so, we could weight the experts per domain instead of using the fixed
hand weights. This is the crowdsourcing / unsupervised-ensemble idea: the Spectral Meta-Learner, Dawid-Skene, and
latent-class models.

A second, commercial question: does the same model estimate the **share of wrong replies** in a domain well enough
to report it without labels?

**Data.** Stored, held-out scores only; no model was run:
- `results/omni/expert_scores_q8b.npz` and `results/omni/judge_probs.npz`;
- 21 domains, 5,107 replies.

**Experts.** Each is z-scored within its domain:
- the Qwen3-8B reader's three: reply, worst unit and its own judge;
- two external judges: Qwen3-14B and phi-4, each scored as −logit P(correct).

NLI is left out. It covers only part of 9 domains, and the baseline does not use it either.

**Baseline.** Hand weights (1, 1, 2, 3, 3) on the z-scores, summed: mean AUROC **0.842** with zero labels. The
reference is **0.859**, with weights fitted on 50 labels.

Results are in `results/omni/novel_unsup_ensemble.json`. The run takes about 5 minutes on one CPU thread.

## How the dependence was handled

The reader's three experts read the same hidden states, so their errors are correlated. The model treats them as
**one group**. That gives three groups that are assumed conditionally independent: the reader, Qwen3-14B and
phi-4.

- **SML, grouped (`sml_g`).** For z-scores and a binary hidden label:
  - a rank-one model, Cov(z_i, z_j) = a_i·a_j, is fitted by least squares to the **7 cross-group covariances
    only**; the within-reader block is masked;
  - the within-class covariance is C − aaᵀ on the reader block and on the diagonal, and 0 across groups;
  - the weights are Σ_w⁻¹a. This is LDA, so it removes the double-counting of the correlated reader experts;
  - the prevalence comes from cross-group third moments, (1−2π)/√(π(1−π)), as in Jaffe et al. 2015 for continuous
    scores.
- **Dawid-Skene, grouped (`ds_g`).** Two-coin EM on votes binarised at the domain median, with 3 voters: the
  reader's mean z, Qwen3-14B and phi-4.
  - Each voter's weight is its log-odds, log(sens·spec / ((1−sens)(1−spec))).
  - The reader's weight is split **equally** over its three experts; there is no hand prior.
- **Latent-class model (`gmm`).** A 2-component Gaussian mixture with one shared full covariance, on all 5 scores.
  It assumes no independence at all.
- **Combo (primary).** The mean of the three weight vectors, each scaled to L1 = 1. Its prevalence is the median of
  the three estimates.
- **Controls.** `sml_naive` and `ds_naive` ignore the dependence and treat the 5 experts as independent. `hybrid` is
  half the hand weights and half the combo weights.

## Rules (stated in the script before the first run)

- **R1. Primary test:** combo against the hand weights, mean AUROC on the natural pools.
  - Method: a hierarchical paired bootstrap, B = 300. Domains are resampled, replies are resampled within each
    domain, and the weights are **re-estimated on every resample**.
  - Verdict: "better" only if the 95% CI is above 0, "worse" only if it is below 0.
- **R2.** Everything else is secondary and descriptive. There is no correction for multiple comparisons.
- **R3.** Pools at 20% prevalence, 10 draws per domain.
- **R4.** Calibration of the accuracy estimates.
- **R5.** The prevalence estimate counts as "useful for reporting" only if its MAE is ≤ 0.05 **and** below two
  zero-label baselines that read the judges' raw outputs.

One bug was fixed after the smoke test: the starting value is now clipped into the solver's bounds. The rules did
not change.

## Results: mean AUROC (paired difference from the hand weights, 95% CI)

| method | all 21 | chat (2) | long-form (5) | RAG (10) | reasoning (4) | wins / 21 |
|---|---|---|---|---|---|---|
| hand (1,1,2,3,3) | 0.842 | 0.785 | 0.870 | 0.807 | 0.923 | n/a |
| **combo (primary)** | 0.843, **+0.001 [−0.005, +0.009]** | +0.002 [−0.013, +0.019] | −0.001 [−0.011, +0.011] | +0.003 [−0.009, +0.018] | −0.004 [−0.012, +0.002] | 10 |
| Dawid-Skene, grouped | **0.847, +0.005 [+0.0003, +0.010]** | +0.004 | +0.003 | +0.008 [−0.001, +0.017] | +0.001 | **17** |
| SML, grouped | 0.844, +0.002 [−0.005, +0.009] | +0.007 | −0.002 | +0.005 | −0.002 | 10 |
| latent class (GMM) | 0.803, **−0.039 [−0.066, −0.015]** | −0.039 | −0.027 | −0.031 | −0.073 | 2 |
| hybrid (½ hand + ½ combo) | 0.843, +0.001 [−0.002, +0.006] | | | | | 11 |
| SML, naive (dependence ignored) | 0.836, −0.006 [−0.013, +0.000] | | | | | 6 |
| Dawid-Skene, naive | 0.837, −0.005 [−0.011, +0.002] | | | | | 6 |
| equal weights | 0.837, −0.005 [−0.011, −0.000] | −0.024 | | | | 7 |
| *50 labels (reference)* | *0.859* | | | | | |

**At 20% prevalence (R3):**
- hand weights: 0.839;
- combo: +0.000 ± 0.003;
- **Dawid-Skene grouped: +0.004 ± 0.002, winning 16 of 21 domains**;
- GMM: −0.038;
- the naive variants: −0.006 to −0.007.

**Handling the dependence matters.** Grouping the reader's experts turns a loss of about −0.005 into a gain of
+0.002 to +0.005, for both SML and Dawid-Skene.

## How well the accuracies are estimated (R4, 105 domain-expert pairs)

| method | target | MAE | bias | Pearson r | ranking within a domain (Spearman) |
|---|---|---|---|---|---|
| SML, grouped | AUROC | 0.163 | **+0.163** | 0.52 | 0.47 |
| Dawid-Skene, grouped | balanced accuracy of the median vote | 0.123 | **+0.123** | 0.67 | 0.53 |
| GMM | AUROC | 0.109 | +0.093 | 0.45 | 0.35 |

**Every method over-estimates accuracy, by a lot.** The reason is that the independence assumption fails even
across groups. Within each class:
- Qwen3-14B and phi-4 correlate at **0.65 among clean replies and 0.70 among wrong ones**;
- the reader and the judges correlate at 0.51 among clean replies and 0.57 among wrong ones.

Shared confusion looks like shared accuracy, so the estimates come out too high. The *relative* ranking holds up
only moderately (Spearman about 0.5). That is enough to adjust the weights slightly, but not to trust the absolute
numbers.

## Estimating the share of wrong replies (R5)

The pools are subsampled to 10, 20 and 30% prevalence, plus each domain's natural rate: 271 pools in all.

| estimator | MAE | bias | r | mean estimate at true 10% / 20% / 30% / 50% |
|---|---|---|---|---|
| combo | 0.160 | +0.149 | 0.24 | 0.34 / 0.36 / 0.39 / 0.45 |
| SML, third moments | 0.147 | +0.133 | 0.23 | 0.33 / 0.34 / 0.37 / 0.43 |
| GMM | 0.157 | +0.109 | 0.24 | 0.28 / 0.32 / 0.36 / 0.44 |
| Dawid-Skene | 0.273 | +0.272 | −0.03 | 0.49 / 0.49 / 0.48 / 0.50 |
| judges, share with P(wrong) > 0.5 | 0.292 | +0.290 | 0.26 | 0.45 / 0.50 / 0.55 / 0.65 |
| judges, mean P(wrong) | 0.279 | +0.277 | 0.26 | 0.44 / 0.48 / 0.53 / 0.63 |

**The prevalence estimate fails rule R5.**
- The estimates squash toward 0.35-0.45 whatever the true rate.
- Dawid-Skene on median votes cannot move away from 0.5 at all.
- The model-based estimates do beat the judges' classify-and-count, which over-flags badly, but an MAE of 0.15 is
  useless for reporting.

## Verdict

- **Primary: no detectable difference.** Combo is +0.001 [−0.005, +0.009] against the fixed hand weights, and is not
  close to the 50-label 0.859.
- **Dawid-Skene, grouped** is the one promising variant: +0.005 [+0.0003, +0.010], winning 17 of 21 domains, with
  the same +0.004 at 20% prevalence.
  - It is a secondary result with no multiplicity correction, so it needs replication before it ships.
  - Part of its gain may come from the equal split within the reader, not from the agreement weighting itself.
- **The latent-class GMM hurts** (−0.039). In practice, the mixture split follows the shape of the scores, not
  whether the replies are right or wrong.
- **Accuracy and prevalence estimates are not usable without labels.** The experts' errors are strongly correlated
  even across different judge models (within-class r ≈ 0.65), which breaks the core assumption of these methods.
  Accuracies come out about 0.12-0.16 too high, and prevalence about 0.15 too high and squashed toward the middle.
- **What to keep:** if anything, ship grouped Dawid-Skene as an optional per-domain re-weighting after it is
  replicated. Do not ship the prevalence estimate. The hand weights are already near what agreement can recover.
