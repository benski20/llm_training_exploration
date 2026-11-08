# Similarity-weighted transfer, label-free anomaly scores, label-free direction re-estimation

`experiments/novel_transfer_anomaly.py` → `results/omni/novel_transfer_anomaly.json`. Stored Qwen3-1.7B features only, no
model loaded (about 4 min on one CPU thread). Protocol: leave-one-domain-out over 21 domains, AUROC within each domain.
The head is the 0.712 baseline: reply mean state and judgement state at layers 15/20/24, PCA 256, per-domain
standardisation from unlabelled replies, L2 logistic (l2 = 0.05). Uniform training reproduces **0.712**.

## Rules stated before running

- **R1.** The primary metric is mean AUROC over the 21 held-out domains.
- **R2.** A method "beats uniform" only if both hold:
  - the paired, domain-stratified bootstrap 95% CI of the difference is above 0 (1,000 resamples of replies);
  - it improves at least 12 of the 21 domains.
- **R3.** Hyperparameters were fixed in advance. Only these primary variants count:
  - (1) `knn_mmd_auto`, `mmd_soft`, `kmm_mix`, `iw_tempered`;
  - (2) head + `maha_probe`;
  - (3) `gmm1d_self` with l2p = 0.5.

  Grid members and choices that use target labels ("oracle") are reported but are not valid results.
- **R4.** Anomaly scores are never sign-flipped.
- **R5.** The rule for choosing k uses no target labels. It runs a nested LODO over the training domains: each training
  domain in turn is a pseudo-target, and the head is trained on its k MMD-nearest other training domains.

## (1) Similarity-weighted transfer

Domain similarity comes only from unlabelled replies. It is the RBF-kernel MMD between domains in globally
standardised PCA coordinates, computed on 150 replies per domain.

| method (zero target labels) | mean [95% CI] | Δ vs uniform, 95% CI | domains improved | reasoning | long-form | RAG | chat |
|---|---|---|---|---|---|---|---|
| uniform (baseline) | 0.712 [0.698, 0.726] | – | – | 0.843 | 0.708 | 0.684 | 0.600 |
| **mmd_soft**: domain weight exp(−MMD²/median) | 0.720 | **+0.005, +0.012** | **18** | 0.847 | 0.723 | 0.691 | 0.606 |
| **kmm_mix**: ½ uniform + ½ KMM simplex domain weights | **0.723** | **+0.005, +0.016** | 14 | 0.850 | 0.737 | 0.691 | 0.593 |
| kmm alone | 0.716 | −0.005, +0.012 | 12 | 0.835 | 0.744 | 0.683 | 0.571 |
| **iw_tempered**: √(domain-classifier odds), clipped at the 99th percentile | 0.721 | **+0.005, +0.012** | 14 | 0.853 | 0.737 | 0.682 | 0.607 |
| iw (untempered) | 0.721 | +0.002, +0.017 | 12 | 0.847 | 0.751 | 0.680 | 0.604 |
| **knn_mmd_auto** (R5 rule picks k) | 0.711 | −0.007, +0.004 | 11 | 0.835 | 0.713 | 0.681 | 0.603 |
| knn_mmd k = 1 / 2 / 4 / 8 / 12 / 16 | 0.609 / 0.648 / 0.665 / 0.710 / 0.715 / 0.717 | | | | | | |
| knn by mean distance, k = 8 / 12 / 16 | 0.700 / 0.707 / 0.707 | | | | | | |
| *oracle k per domain (uses target labels)* | *0.732* | *+0.013, +0.028* | *18* | 0.845 | 0.737 | 0.705 | 0.626 |

- **Three of the four primary variants pass R2: soft MMD weights, KMM mixed with uniform, and tempered importance
  weights.** The gain is small, +0.008 to +0.011.
  - A cruder domain-level bootstrap agrees: the 21 per-domain differences, resampled, give kmm_mix
    +0.011 [+0.003, +0.020], iw_tempered +0.009 [+0.002, +0.018], mmd_soft +0.008 [+0.005, +0.012].
  - The four tests are not multiplicity-corrected.
  - Most of the gain is on long-form (+0.015 to +0.03) and reasoning. Chat and RAG barely move.
- **Hard k-nearest selection loses.** Fewer than 8 domains costs 0.05-0.10: diversity beats similarity, as in the
  diversity curve.
- **No label-free rule finds the right k.** The nested curve is flat and peaks at k = 12-16 for every target, so the
  rule picks 12 or 16 and ties uniform (0.711).
  - Choosing k with target labels would reach 0.732. That 0.02 is selection noise plus real per-domain structure, and
    no unlabelled statistic recovered it.
- **Soft down-weighting is better than cutting.** The best rules keep every domain and tilt toward similar ones. KMM
  alone concentrates on 3-6 domains and gives back the gain; mixing it with uniform restores it.
- The importance weights are heavy for reasoning targets: the effective sample size is 11-36% of the pool there,
  against 80-95% for RAG.

## (2) Label-free anomaly scores ("wrong = atypical for its domain")

Each score is fitted on the held-out domain's own unlabelled replies. The variants:
- Mahalanobis distance on the top 10 / 32 PCs.
- Mahalanobis distance in the *probe subspace*: the top 5 singular directions of the 20 training domains'
  per-domain heads, with shrinkage.
- Distance to the 10 nearest neighbours.
- Isolation forest.

| score | alone | z-summed with the head | Δ (sum − head), 95% CI |
|---|---|---|---|
| Mahalanobis, top 32 PCs | 0.487 | 0.654 | −0.069, −0.047 |
| Mahalanobis, probe subspace (primary) | 0.510 | **0.657** | **−0.066, −0.043** |
| kNN distance | 0.495 | 0.645 | −0.079, −0.056 |
| isolation forest | 0.488 | 0.645 | −0.079, −0.056 |
| directional: pooled wrong−right mean-difference direction, domain-centred | 0.709 | 0.716 | −0.000, +0.009 |
| directional: head on centred-only features (no per-domain scaling) | 0.705 | 0.710 | −0.005, +0.001 |
| 8B reply head (0.789) + probe-subspace Mahalanobis / + kNN | – | 0.719 / 0.701 | −0.080, −0.060 / −0.100, −0.077 |

By format, the unsigned anomaly scores alone score:

| format | AUROC |
|---|---|
| reasoning | 0.56-0.63 |
| long-form | 0.52-0.54 |
| RAG | 0.44-0.48 |
| chat | 0.41-0.49 |

- **Wrong replies are not atypical.** Every unsigned distance or density score is at chance overall (0.49-0.51).
  - On RAG and chat they score *below* 0.5: there, wrong replies are, if anything, more typical than correct ones.
  - Only reasoning shows a weak anomaly signal (a wrong step makes a solution unusual).
- **Adding an unsigned score to the head costs 0.05-0.07.** It costs as much at the 8B tier.
- **The "directional anomaly" is the head itself.** A universal direction minus the domain's mean is what per-domain
  standardisation already computes. A mean-difference direction scores 0.709 against 0.712, and summing the two
  adds +0.004, inside noise.

## (3) Label-free re-estimation of the direction on the new domain

Each method produces pseudo-labels from the domain's unlabelled replies. A head is then refitted on them, pulled
toward the universal head; l2p is the prior strength.

| method | mean | Δ, 95% CI | reasoning | long-form | RAG | chat |
|---|---|---|---|---|---|---|
| uniform | 0.712 | – | 0.843 | 0.708 | 0.684 | 0.600 |
| 2-component GMM on head scores (EM) → self-train, l2p 0.5 (primary) | 0.712 | −0.004, +0.004 | 0.844 | 0.713 | 0.681 | 0.599 |
| same, l2p 0.05 / 5 | 0.708 / 0.713 | | | | | |
| GMM self-training iterated 3 rounds | 0.719 | +0.001, +0.014 (11 domains improved: fails R2) | 0.843 | 0.720 | 0.694 | 0.596 |
| shared-covariance GMM in a 16-dim subspace (head direction + domain PCs): posterior / self-train | 0.659 / 0.682 | −0.042, −0.018 (self-train) | 0.788 | 0.656 | 0.679 | 0.546 |
| TSV-style Sinkhorn optimal-transport pseudo-labels: posterior / self-train | 0.682 / 0.707 | −0.013, +0.004 (self-train) | 0.829 | 0.702 | 0.689 | 0.565 |
| *reference: judge pseudo-labels, then self-train (`judge_distillation_q17.json`)* | *0.710 → 0.765* | | *0.844* | *0.756* | *0.759* | *0.661* |

- **Self-training on the head's own pseudo-labels cannot add information.**
  - The GMM on the head's scores recovers what the head already ranks: ±0.00.
  - The richer clustering methods find the domain's dominant variation, which is not correctness. The 16-dim GMM and
    OT lose 0.005-0.05.
- **The judge's +0.055 comes from an external teacher, not from the self-training step.** On RAG and chat the judge
  gives +0.07; self-training gives 0.00.

## Verdict

- **Similarity weighting: a small, real, free gain.** Soft MMD domain weights, KMM mixed with uniform, and tempered
  covariate-shift weights each pass the pre-stated rule at +0.008 to +0.011, about 0.720. That is far smaller than
  judge distillation (+0.055).
  - mmd_soft improves 18 of 21 domains with the tightest CI.
  - Hard k-nearest selection hurts, and the label-free rule for k picks k ≈ all, which ties uniform.
- **Anomaly scores: no.** Wrong replies are not outliers in their domain. Combined with the head they cost 0.05-0.07.
- **Label-free direction re-estimation: no.** EM, subspace GMM and OT pseudo-labels give 0.68-0.72. External
  pseudo-labels (the judge) remain the only zero-label lever that moves the 1.7B head substantially.
- **If anything is shipped:** `mmd_soft` as a default for fitting the head at onboarding. It is one kernel-mean
  computation per new domain, and it adds to the judge route rather than replacing it. The two have not been tested
  together.
