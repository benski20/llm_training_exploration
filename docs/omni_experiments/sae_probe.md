# Writing the probe from mechanistic understanding: SAE latents chosen without labels

**Question.** Does a probe built from a few interpretable SAE latents, chosen *without correctness labels* and
given hand-set weights, transfer across domains better than a fitted probe?

**Answer: no.** The written probe scores 0.55 AUROC on held-out domains, against 0.69 for the plain fitted
probe on raw activations. With 25-50 labels, refitting only its weights reaches 0.63-0.66, still 0.10 below the
raw probe refitted on the same labels. A sparse fitted probe on all SAE latents roughly ties the raw probe
(0.676 against 0.686, n.s.). It does not beat it.

- Script: `kaggle/omni/sae_probe.py`, run as Kaggle kernel `omni-sae-probe` (2× T4, fp16, sdpa, 87 min).
- Results: `results/omni/sae_probe.json`. The raw kernel outputs, including per-reply scores, are in
  `results/kaggle/omni-sae-probe/`.

## Setup

**Reader, SAE and layer.**
- Reader: Qwen3-1.7B (instruct).
- SAE: Qwen-Scope `Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50` (TopK, k = 50, 32,768 latents, `resid_post`).
- Layer: resid_post of block 14, which is `hidden_states[15]`. This is the "layer 15 of 28" that is the repo's
  best single layer.

**Does the SAE fit the instruct reader?** The SAEs were trained on Qwen3-1.7B-Base. The architecture and layers
match the instruct model. A rule was fixed before the check: use the instruct reader if the SAE's fraction of
variance unexplained (FVU) on its reply-token residuals is ≤ 0.5 and ≤ 1.5× the Base model's FVU. Otherwise use
Base.

| reader | FVU | mean cosine |
|---|---|---|
| Base | 0.289 | 0.92 |
| instruct | 0.340 | 0.91 |

Measured on 35,860 reply tokens from 200 random replies. The instruct reader passes, so it was used. No
non-finite activations appeared in fp16.

**Data.**
- Corpus: built in the kernel with `experiments/omni_corpus.py build` (defaults): 5,107 replies, 21 domains,
  4 formats.
- Tokenisation: `vt/omni.encode(tok, row, max_tokens=1024)`, without the judge suffix. The reader is causal, so
  reply states do not depend on it.

**Features, one pass per reply.**
- The mean over reply tokens of the raw residual (2,048 dimensions).
- The mean over reply tokens of the SAE latents (32,768 dimensions).

**Protocol.**
- Leave-one-domain-out (LODO) over all 21 domains.
- AUROC for "wrong" within each held-out domain, then the mean over domains, overall and per format.
- Two variants:
  - global: z-scores from the training domains;
  - dstd: each domain z-scored by its own unlabelled replies.
- Latent z-scores are clipped to ±10.
- CIs: 1,000 bootstrap resamples of replies within each domain, stratified by label, on the difference from
  (a). For (d), test sets differ between draws, so the bootstrap resamples domains instead.

## The four probes

**(a) Raw-residual probe.** L2 logistic regression (l2 = 0.05, balanced) on the mean residual. This is the
baseline.

**(b) SAE-latent probe.**
- L1 logistic regression (FISTA on the GPU) on the mean-pooled latents.
- Inputs: the 16,971 latents active in at least 1% of replies, a label-free filter.
- λ is chosen from {1e-3, 3e-3, 1e-2, 3e-2} by inner 5-fold leave-domains-out CV on the 20 training domains.
- The fitted probes use a median of 376 latents (global) and 36 latents (dstd).
- The same latents with an L2 penalty are reported as `b_sae_l2`.

**(c) Written probe: no correctness labels anywhere.** The selection rule was fixed in the script before any run:
- **ENT, "reader uncertainty":** the 10 latents whose activation at a token has the highest Pearson correlation
  with the reader's next-token entropy at that token. Computed over all reply tokens of the corpus, unlabelled.
  Only latents that fire on 0.5-50% of tokens qualify.
- **UNK, "unknown entity":** the 10 latents with the largest standardised difference (invented minus real) at
  the last token of the entity. The contrast set is the real/invented-entity prompts of
  `experiments/recall_neurons.py`: 280 prompts with no answers. Only latents that fire on at least 20% of the
  invented prompts qualify.
- **NOV, "not copied from context":** the 10 latents whose activation correlates most with "this reply token is
  not in a trigram that appears in the prompt". Computed over the replies that have a document. Only latents that
  fire on 0.5-50% of tokens qualify.
- **Score:** the mean of the three group means of z-scored latents. Every latent has weight +1/30, and nothing is
  fitted.
- **`c_fit` (diagnostic):** the same 30 latents with weights fitted LODO.

**(d) The written probe plus k = 25 or 50 in-domain labels.**
- Only its 30 weights and the bias are refitted, with an L2 pull toward the hand weights.
- The pull strength is chosen by repeated half-splits of the k labels. The hand weights are kept unless a refit
  wins by more than 0.005.
- Scored on the rest of the domain, averaged over 20 balanced draws.
- Comparator `(a)+k`: the LODO raw probe refitted in the same way on the same draws. dstd variant.

## Results: zero labels, mean AUROC over the 21 held-out domains

Differences from (a) are shown with their 95% CI.

**Global standardisation**

| probe | all | Δ vs (a) [95% CI] | reasoning (4) | long-form (5) | RAG (10) | chat (2) |
|---|---|---|---|---|---|---|
| (a) raw residual, L2 | **0.686** | — | **0.814** | 0.697 | **0.646** | **0.598** |
| (b) SAE latents, L1 | 0.676 | −0.010 [−0.027, +0.005] | 0.807 | **0.703** | 0.631 | 0.569 |
| (b′) SAE latents, L2 | 0.617 | −0.068 [−0.083, −0.053] | 0.764 | 0.630 | 0.567 | 0.546 |
| (c) written, no labels | 0.552 | −0.134 [−0.155, −0.113] | 0.616 | 0.518 | 0.541 | 0.563 |
| `c_fit`: the same 30 latents, fitted LODO | 0.565 | −0.121 [−0.141, −0.100] | 0.704 | 0.627 | 0.491 | 0.504 |

Per-format CIs for (b) − (a):

| format | Δ [95% CI] |
|---|---|
| reasoning | [−0.027, +0.013] |
| long-form | [−0.032, +0.048] |
| RAG | [−0.040, +0.008] |
| chat | [−0.070, +0.019] |

Per-format CIs for (c) − (a):

| format | Δ [95% CI] |
|---|---|
| reasoning | −0.198 [−0.238, −0.157] |
| long-form | −0.179 [−0.228, −0.127] |
| RAG | −0.105 [−0.138, −0.075] |
| chat | −0.035 [−0.091, +0.024] |

**Per-domain standardisation (dstd)**

| probe | all | Δ vs (a) [95% CI] | reasoning | long-form | RAG | chat |
|---|---|---|---|---|---|---|
| (a) raw residual | **0.673** | — | **0.809** | 0.670 | **0.642** | 0.559 |
| (b) SAE latents, L1 | 0.663 | −0.010 [−0.027, +0.008] | 0.775 | **0.672** | 0.637 | 0.547 |
| (b′) SAE latents, L2 | 0.631 | −0.041 [−0.057, −0.026] | 0.767 | 0.638 | 0.592 | 0.539 |
| (c) written, no labels | 0.553 | −0.119 [−0.142, −0.098] | 0.574 | 0.552 | 0.543 | **0.566** (+0.007 [−0.050, +0.066]) |
| `c_fit` | 0.597 | −0.076 [−0.096, −0.057] | 0.678 | 0.641 | 0.556 | 0.529 |

**The written groups separately (dstd, all domains)**

| group | AUROC |
|---|---|
| ENT | 0.561 |
| UNK | 0.529 |
| NOV | 0.521 |

- ENT reaches 0.89 on HaluEval QA and 0.78 on HaluEval summary, but 0.32 on DROP and 0.42 on TruthfulQA.
- UNK is near 0.5 everywhere except FELM writing (0.78).
- NOV is at or below chance on long-form and reasoning.

## Results: few labels (dstd, mean over 20 draws per domain)

| | k | all | reasoning | long-form | RAG | chat | Δ vs (a)+k (domain bootstrap) | Δ vs (a) at zero labels |
|---|---|---|---|---|---|---|---|---|
| (c) written, same test replies | 0 | 0.555 | 0.576 | 0.556 | 0.544 | 0.567 | — | — |
| **(d) written, refitted** | 25 | 0.631 | 0.635 | 0.639 | 0.635 | 0.586 | −0.104 [−0.131, −0.078] | −0.041 [−0.095, +0.026] |
| (a)+k, raw probe refitted | 25 | **0.735** | 0.796 | 0.710 | 0.743 | 0.638 | — | +0.062 [+0.021, +0.116] |
| **(d) written, refitted** | 50 | 0.659 | 0.668 | 0.669 | 0.662 | 0.604 | −0.101 [−0.125, −0.077] | −0.014 [−0.064, +0.045] |
| (a)+k, raw probe refitted | 50 | **0.760** | 0.804 | 0.733 | 0.774 | 0.674 | — | +0.088 [+0.042, +0.147] |

The (a)+k numbers reproduce the repo's prior-pulled refit: 0.754 / 0.770 at 25 / 50 labels, with the full
feature set.

## Verdict

The written, mechanistic probe does **not** transfer better than a fitted one. It is clearly worse:
- **Zero labels:** −0.12 to −0.13 AUROC overall, significant on reasoning, long-form and RAG.
- **The only tie is closed-book chat** (+0.007 [−0.05, +0.07] with dstd). There the fitted probe is itself near
  chance, so the tie means little.
- **Refitting the written weights does not rescue it.** With 25-50 labels it stays 0.10 below the raw probe
  refitted on the same labels. The few chosen latents lack the signal. How they are weighted is not the problem:
  fitting their weights on 20 domains (`c_fit`) only reaches 0.57-0.60.

The SAE basis is not bad, just not better:
- A sparse fitted probe over all latents ties the raw-residual probe: −0.010, CI [−0.027, +0.005]. It is slightly
  ahead on long-form and behind on chat and reasoning (dstd: −0.034 [−0.064, −0.006]).
- This matches the literature in docs/research/mech_interp.md: SAE probes reach parity with dense probes and do
  not beat them out of distribution.

## Why the label-free concepts fail here

- **"Unknown entity" latents barely fire on the replies.** They fire on 0.02-0.2% of reply tokens. They detect
  invented names in questions. Replies from other models rarely contain invented names. When they do, the reader
  is the reader and not the writer: it measures *its own* unfamiliarity, the misfire noted in mech_interp §1.1.
- **The reader's next-token uncertainty is not the writer's error.** Token log-probability is 0.49 in the repo.
  The entropy-correlated latents inherit this: they fire strongly on paraphrase-heavy wrong answers (HaluEval)
  and in reverse on DROP and TruthfulQA.
- **"Novel, not copied" is format-, not error-specific.** Correct summaries and reasoning steps are also novel
  text, so NOV is below chance on long-form.
- **The sign of a concept flips between domains.** Example: ENT scores 0.89 on one domain and 0.32 on another.
  A hand-set weight cannot follow that. A probe fitted on many domains averages over it.

## Caveats

- **Only one SAE layer (14) and one SAE (W32K, L0 = 50).** Other layers, the L0 = 100 SAE, or transcoders
  (`mwhanna/qwen3-1.7b-transcoders-lowl0`) were not tried.
- **The latents were selected by contrasts, not by explanations.** Qwen-Scope publishes no latent explanations,
  and a different label-free rule could pick better ones. Still, the selection rule was fixed in advance and
  three distinct mechanisms all came out near chance.
- **The SAE was trained on the Base model.** On the instruct reader it explains 66% of variance, against 71% on
  Base. The fit is acceptable, but some instruct-specific structure is lost.
- **max_tokens is 1024, not the repo's 2048, and only one layer is used,** so (a) is 0.686 rather than the
  repo's 0.698 (three layers). All probes here share the same replies, folds and truncation.
- **dstd slightly hurt (a) and (b) at this single layer** (−0.013). The repo saw +0.01 to +0.03 with three
  layers plus the judgement state.
- **The (d) CIs resample domains, so they are wider** than the reply-level CIs of the zero-label table.
