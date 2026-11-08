# Writing a new domain's expert weights from its unlabelled statistics (`experiments/novel_hyper_weights.py`)

**Question.** Can a small model trained on the other domains write the per-domain weights of the verifier's 5 experts
for a new domain from that domain's *unlabelled* replies alone ("hypernetwork lite")? If it could, a new customer would
get adapted weights with zero labels.

**Answer: no.**
- No variant beats the fixed weights (1,1,2,3,3). The best gain is +0.002, with a confidence interval that includes 0.
- Every variant stays about 0.013 below the 50-label refit.
- The regressor's inner cross-validation always chose the strongest shrinkage, so it ignored the descriptors and
  returned the average of the other domains' weights.
- The descriptors do not predict the weights. When the regressor is forced to use them, AUROC falls.

Results are in `results/omni/novel_hyper_weights.json`.

## Setup

The inputs are stored held-out scores only, and no model is run:
- `results/omni/expert_scores_q8b.npz`: reply head, worst unit and the 8B reader's own judge;
- `results/omni/judge_probs.npz`: Qwen3-14B and phi-4, using −logit as the "wrong" score;
- `/home/user/omni/corpus.jsonl`: reply text, units and documents.

Each expert is z-scored on the domain's own replies, which needs no labels. The method has three parts.

1. **Oracle weights per domain.** A balanced L2 logistic regression (l2 = 0.05) on all of the domain's labels. The
   weights are clipped at 0 and rescaled to sum to 10, the scale of the fixed rule. They serve only as regression
   targets for the other domains, plus one in-sample headroom number.
2. **36 descriptors per domain, computed without labels.**
   - per expert: mean, standard deviation and skew of the raw score;
   - the 10 pairwise Spearman correlations between experts;
   - share of replies with a document, and a one-hot of the format;
   - text statistics: log reply length, units per reply, digit density, share of replies with math, log document
     length, share of short replies, and log of the number of replies.
3. **Leave one domain out (LODO), over 21 domains.** Each regressor is fitted on 20 domains and predicts the held-out
   domain's weights. That domain is then scored by AUROC on all of its replies, none of whose labels were used.

## Rules stated before any result

These are written in the script's docstring.
- **Primary method:** `hyper_ridge`. Ridge regression from the standardised descriptors to the 5 weights. Alpha is
  chosen by an inner LODO over the 20 training domains, minimising the error on the weights rather than AUROC, from
  the grid {0.1, 1, 10, 100, 1000}.
- **Primary comparison:** mean AUROC over the 21 domains against the fixed weights. A paired bootstrap over domains
  (10,000 resamples) gives the confidence interval, and "better" requires it to exclude 0.
- **Weight clipping:** every method's weights are clipped at 0 or above.
- **Secondary variants:**

  | variant | how the weights are set |
  |---|---|
  | `mean_others` | mean of the other domains' oracle weights |
  | `format_mean` | mean over the other domains of the same format |
  | `knn3` | mean of the 3 nearest domains in descriptor space |
  | `gp` | kernel ridge / GP mean: RBF kernel, median-heuristic length scale, noise 1 |
  | `pc1` | the domain's first principal component |
  | `one_factor` | one-factor model, λ/ψ, with λ from each expert's correlation with the others' consensus |
  | `consensus_corr` | each expert's leave-one-out correlation with the consensus |
  | `shrink_hyper` | half `hyper_ridge`, half the fixed weights |

- **Post hoc:** `hyper_ridge_a10` forces alpha = 10. It was added after seeing that the inner LODO always picked 1000.

## Results: mean AUROC, zero labels unless stated

| method | all 21 | chat (2) | long-form (5) | RAG (10) | reasoning (4) | Δ vs fixed [95% CI] |
|---|---|---|---|---|---|---|
| **fixed (1,1,2,3,3)** | **0.842** | 0.785 | 0.870 | 0.807 | 0.923 | — |
| equal weights | 0.837 | 0.762 | 0.866 | 0.803 | 0.921 | −0.005 [−0.010, −0.001] |
| **hyper_ridge (primary)** | 0.843 | 0.768 | 0.873 | 0.811 | 0.924 | +0.001 [−0.003, +0.005] |
| hyper_ridge, alpha = 10 (post hoc) | 0.837 | 0.764 | 0.868 | 0.802 | 0.923 | −0.005 [−0.010, −0.000] |
| shrink_hyper | 0.842 | 0.777 | 0.871 | 0.809 | 0.923 | +0.001 [−0.001, +0.002] |
| mean_others | 0.843 | 0.767 | 0.873 | 0.811 | 0.924 | +0.001 [−0.003, +0.005] |
| format_mean (a-priori rule) | 0.843 | 0.775 | 0.871 | 0.811 | 0.923 | +0.002 [−0.002, +0.005] |
| knn3 | 0.837 | 0.758 | 0.871 | 0.802 | 0.923 | −0.005 [−0.011, +0.001] |
| gp | 0.842 | 0.767 | 0.872 | 0.809 | 0.923 | −0.000 [−0.004, +0.003] |
| pc1 (unsupervised) | 0.837 | 0.768 | 0.865 | 0.803 | 0.921 | −0.005 [−0.009, −0.001] |
| one_factor (unsupervised) | 0.836 | 0.773 | 0.861 | 0.804 | 0.918 | −0.006 [−0.010, −0.002] |
| consensus_corr (unsupervised) | 0.837 | 0.769 | 0.865 | 0.803 | 0.921 | −0.005 [−0.009, −0.001] |
| 50 labels, refit (`adapt_50`, scored on the rest) | 0.856 | 0.793 | 0.867 | 0.837 | 0.922 | +0.014 [+0.002, +0.035] |
| oracle, in sample (uses all labels; ceiling) | 0.862 | 0.809 | 0.884 | 0.836 | 0.925 | +0.020 [+0.012, +0.029] |

Against the 50-label refit, every zero-label method falls short:
- `hyper_ridge`: −0.013 [−0.033, −0.000];
- `format_mean`: −0.013 [−0.032, −0.000].

The 50-label refit here scores 0.856. The headline's 0.859 includes NLI and the checker flags. It is also scored on
the replies left after the 50 labelled ones are removed, so the comparison with it is indicative only.

**Squared error of the predicted weights in LODO** (sum over the 5 experts; weights sum to 10):

| predictor | error |
|---|---|
| mean of the others | 9.6 |
| hyper_ridge | 9.85 |
| fixed | 10.8 |
| gp | 11.2 |
| format mean | 11.7 |
| knn3 | 13.7 |
| one_factor | 13.7 |
| ridge, alpha = 10 | 17.6 |

## Why it fails

1. **Little headroom at zero labels.** Even the in-sample oracle, which is non-negative and fitted on all labels,
   adds only +0.020. The fixed weights already sit close to every domain's optimum, because the external judges
   dominate everywhere.
2. **The domain-to-domain variation in weights looks like noise to these descriptors.** For example:
   - `halubench_drop` wants nearly all weight on phi-4, and `halubench_finance` none on phi-4;
   - `ragtruth_qa` and `ragtruth_summary` want the worst unit, and `halueval_general` its own judge.

   None of this is predictable from score moments, correlations or text statistics with 20 training points. The
   inner CV correctly shrank the regressor to the mean, and forcing it to use the descriptors hurts. kNN overfits.
3. **Unsupervised "trust the consensus" weights are wrong.** The experts that agree most with each other are the
   reader-internal ones (reply head, worst unit, own judge), which are correlated with each other rather than more
   accurate. Up-weighting them costs 0.005.
4. **Clipping at 0 blocks some domains.** `halueval_summary` reaches 0.866 with 50 labels against 0.744 for the
   clipped oracle: there a negative weight, an anti-correlated expert, helps. No zero-label rule here can find that.
   Unclipped targets were not tried, as this rule was fixed in advance.

**Verdict.** With 21 domains, writing the weights from unlabelled statistics does no better than the fixed rule;
the average of the other domains' weights is as good, and as small. The 0.014-0.017 gap to 50 labels is
domain-specific signal that only labels reveal. Keep the fixed weights for zero labels and the 50-label refit for
onboarding.
