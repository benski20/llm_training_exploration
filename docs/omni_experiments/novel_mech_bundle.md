# Mechanistic-interpretability verification signals (ideas 1-6 of the 2026 survey): all null at 1.7B

The test covers the six ranked ideas in [mech_interp_verification_2026.md](../research/mech_interp_verification_2026.md).
- Ideas 1, 2, 3, 5 (attention part) and 6 come from one feature-extraction pass per reply.
- Idea 4 (a different-family peer reader) is a separate check.

Files:
- script: `experiments/novel_mech_bundle.py`;
- numbers: `results/omni/novel_mech_bundle.json`.

**Verdict:**
- No idea passes the gate. Every zero-label sum and every stacked addition *lowers* the mean held-out AUROC of the 1.7B reply head (0.722).
- No mechanistic localiser beats the shipped first-crossing rule for the first wrong ProcessBench step.
- The peer reader (granite-3.3-2B) passed the similarity screen (within-domain debiased CKA 0.89). It adds nothing either: −0.004 [−0.025, +0.016].

## Setup

**Reader and pass**
- Reader: Qwen3-1.7B, bf16, CPU, max 1,024 tokens.
- Each reply is read once as [documents +] question + reply + the shipped judge question, with hidden states and a KV cache.
- The cache is then cropped to the end of the reply, and 5 more judge framings are read on it (a few dozen tokens each).
- Attention probabilities are recomputed exactly on layers 12/15/18/21. This uses each layer's own q/k projections, q/k norms and rotary embeddings, in a pre-hook. Against an eager-attention model, the judge row correlates at 0.9997 per layer; my fp32 softmax is the more precise of the two.
- Cost: 2.6 s per reply with 4 threads, 1,212 replies in 52 min.

**Sample**
- 24 wrong and 24 correct replies per domain × 21 domains = 1,008 (seed 0).
- Plus 204 extra ProcessBench wrong replies, used for localisation only. That gives 72-75 cases per ProcessBench domain.

**Baseline.** The shipped 1.7B reply head:
- stored `/home/user/omni/q17` features, LODO with per-domain standardisation on all 5,107 replies, restricted to the same 1,008;
- nested heads feed the stacker.

**Additions**
- **sum**: z(reply) + z(signal), z within domain, zero labels, a-priori sign.
- **stack**: LODO logistic on [z(nested reply head), z(signal block)].

**Gate, stated in the script before the full run**
- Useful only if sum or stack gains ≥ +0.005 mean AUROC over 21 domains, with the paired bootstrap 95% CI above 0. The bootstrap uses 1,000 resamples within domain, stratified by label.
- Localisation: GO only if exact first-wrong-step accuracy gains ≥ +3 points over the shipped rule, with the CI above 0.
- Idea 6 is killed if its LODO alone is < 0.60 or its addition is < +0.005.
- Idea 4's LODO runs only if debiased CKA and cross-validated linear correlation (within-domain) are both < 0.95.

## Results (mean within-domain AUROC for "wrong", 21 held-out domains, [95% CI])

| signal | alone | reply + signal, sum | reply + signal, stack | best format for the addition |
|---|---|---|---|---|
| **reply head (baseline)** | **0.722** [0.692, 0.752] | | | chat 0.636, long-form 0.697, RAG 0.695, reasoning 0.862 |
| reader's judge, final log-odds (this pass) | 0.631 | | | |
| **1** logit lens, −mean κ over layers 12-22 (primary, label-free) | 0.626 | 0.699 (−0.023 [−0.046, −0.001]) | | reasoning +0.002 |
| 1 −min κ 12-22 / commit depth / sign flips / entropy 12-22 | 0.631 / 0.561 / 0.574 / 0.507 | | | |
| 1 LODO on 168 lens scalars (κ and entropy; residual, attention, MLP paths × 28 layers) | 0.613 (judge-state probe 0.639) | 0.693 (−0.028) | 0.658 (−0.064) | long-form +0.011 (CI spans 0) |
| **2** question-only state, assistant header / user end | 0.562 / 0.556 | −0.033 / −0.046 | −0.045 / −0.064 | chat (header) +0.040 [−0.046, +0.125] |
| 2 shift: judge state − question state | 0.641 | 0.717 (−0.005 [−0.028, +0.018]) | 0.699 (−0.023) | **chat +0.073 [−0.010, +0.151]**, +0.090 stacked [−0.020, +0.192] |
| 2 shift: reply mean − question state | 0.620 | −0.019 | −0.057 | |
| **3** mean of 6 framings / fragility (variance) / polarity inconsistency | 0.609 / 0.531 / 0.411 | −0.035 / −0.034 / −0.053 | | chat fragility +0.022 (CI spans 0) |
| 3 judge-state probe variance across framings | 0.502 | −0.071 | | |
| 3 LODO on all framing features | 0.605 | −0.034 | 0.708 (−0.014 [−0.025, −0.003]) | |
| **5** judge-attention concentration / detection block | 0.459 / 0.572 | −0.078 / −0.040 | 0.711 (−0.011 [−0.020, −0.001]) | reasoning +0.007 (CI spans 0) |
| **6** LapEigvals (4 layers × 16 heads × top-25, PCA 128) | **0.588** (< 0.60: killed) | −0.038 | −0.082 | |
| 6 lookback ratio (64 heads, reply rows and judge row) | 0.605 (RAG 0.555) | −0.031 | −0.058 | |

**No row passes the gate**, overall or in any format. The only positive point estimates of any size are on closed-book chat:
- the judge-state minus question-state shift: +0.07 to +0.09;
- the header prior: +0.04.

Chat is 2 domains × 48 replies, and every one of these CIs crosses 0.

### Idea 1: verdict logit lens

- **The mid-band κ is not a better judge.** It scores 0.626 against 0.631 for the final log-odds (−0.006 [−0.024, +0.012]), and beats it on only 10/21 domains. The "go to 8B" rule needed ≥ 12.
- The commitment-depth and flip summaries carry weak signal (0.56-0.57), mostly on reasoning (0.71-0.72).
- "Late layers overturn the mid band" is *reversed* (0.375): in wrong replies the final κ falls below the mid-band κ. That is the final verdict again, not a new signal.
- A fitted probe on the 168 lens scalars (0.613) is below the plain judge-state probe on the same replies (0.639). Adding the lens to the state does not help either (−0.008). The unembedding readout loses information that a linear probe already has.

### Idea 2: question-only prior and shift

**Duplicates were counted first** (`duplicates` in the JSON).
- HaluEval QA / dialogue / summary and TruthfulQA are 100% paired in the corpus. HaluBench Covid is 13% paired and OlympiadBench 27%.
- 15 domains are unpaired (< 10%). On them:

| feature on the 15 unpaired domains | AUROC |
|---|---|
| reply head | 0.690 |
| prior, assistant header / user end | 0.582 / 0.594 |
| shift: judge state − question state | 0.649 |
| reply + shift, sum | 0.702 (+0.012, point estimate only) |

- Caveat: RAGTruth's user prompt contains the answer itself (the dataset's labelling prompt). On those 3 domains the "question" state has already read the reply.
- The prior alone is weak (0.56) and hurts every format except chat.

### Idea 3: judge-framing ensemble and fragility

**Polarity-flipped framings are anti-informative at 1.7B.**

| framing | AUROC (higher = flags wrong replies) |
|---|---|
| "Does the reply contain any error?", read as P(No) | 0.460 |
| "Is the reply above wrong?", read as P(No) | 0.433 |

- The reader answers "Yes" to these questions more often for *correct* replies. It acts as a yes-sayer that tracks quality, not the question's polarity: the signed correctness from the two framings correlates at −0.21 within domain.
- So:
  - the 6-framing mean (0.609) is below the shipped framing alone (0.631): −0.022 [−0.040, −0.004];
  - polarity inconsistency is reversed (0.411);
  - fragility is at chance (0.531).
- Same-polarity paraphrases correlate at 0.68-0.87 with the shipped framing, and none beats it (0.548-0.628).
- On the 20% of replies where the judge is least confident (pooled after within-domain z), fragility reaches 0.553 and the shipped framing 0.570. The reply head is 0.665 there.

### Idea 5: judge attention as a localiser (ProcessBench first wrong step, 297 cases)

| rule | exact [95% CI] | within one step |
|---|---|---|
| chance | 0.176 | 0.474 |
| **shipped: first unit with z > 0.75 on the unit head** | **0.501** [0.448, 0.561] | **0.737** |
| unit head, highest unit | 0.364 | 0.599 |
| judge attention, mean over 64 heads | 0.047 | 0.188 |
| judge attention, 8 heads with the most within-reply variance (label-free) | 0.050 | 0.195 |
| judge attention, best single head chosen on the other 3 domains (head 38 every time) | 0.101 | 0.295 |
| shipped rule on z(unit head) + z(attention), τ chosen on the other domains | 0.394 (−0.108 [−0.161, −0.057]) | 0.642 |

- Judge-row attention lands *below chance*: it concentrates on the last steps (recency), while first errors are early.
- No GO. Gradient attribution (5b) was not run.

### Idea 6: attention-graph spectral and lookback features

- LapEigvals reaches 0.588 alone: below the pre-stated 0.60, so it is killed. Every addition is negative.
- The lookback ratio is 0.605 overall but only 0.555 on RAG. It ties the earlier pooled-attention failures (copy heads, source flow).

### Idea 4: a different-family peer reader (granite-3.3-2b-instruct)

**Similarity screen** on 105 replies (5 per domain), reply-mean states at the middle layers. Granite layers are 22/28/34 of 40; Qwen layers are 15/20/24 of 28.

| pair | CKA | debiased CKA | CKA on 16 PCs | cross-validated linear corr. |
|---|---|---|---|---|
| Qwen3-1.7B vs granite, global | 0.954 | 0.917 | 0.893 | 0.84 |
| Qwen3-1.7B vs granite, within domain (pre-stated) | 0.964 | **0.892** | 0.779 | **0.90** |
| reference: Qwen3-1.7B vs Qwen3-0.6B, within domain | 0.971 | 0.917 | 0.828 | 0.92 |

- Granite is below 0.95, so the LODO test ran.
- It is only slightly more different from Qwen3-1.7B than Qwen3-0.6B is: 0.89 against 0.92 debiased CKA. Small decoder readers of different families end up close in reply-state geometry.

**LODO on the 1,008 replies.**
- The granite head uses mean + judge state at 3 middle layers (PCA 128), output scalars and its own judge.
- Granite's judge is read in the same pass, with its own chat template.

| | all 21 | chat | long-form | RAG | reasoning |
|---|---|---|---|---|---|
| Qwen reply head | 0.722 | 0.636 | 0.697 | 0.695 | 0.862 |
| granite head | 0.650 | 0.572 | 0.610 | 0.636 | 0.775 |
| Qwen judge / granite judge | 0.637 / 0.615 | **0.480 / 0.640** | 0.638 / 0.561 | 0.590 / 0.573 | 0.833 / 0.775 |
| Qwen reply + granite head, sum | 0.718 (−0.004 [−0.025, +0.016]) | −0.010 | −0.006 | +0.004 | −0.018 |
| Qwen reply + granite block, stack | 0.678 (−0.044 [−0.075, −0.015]) | | | | |
| Qwen reply + judge, plus granite head + judge (sum) | 0.697 against 0.701 (−0.004 [−0.025, +0.015]) | **+0.050** [−0.016, +0.126] | −0.017 | −0.005 | −0.012 |

- **No gain** overall or in any format.
- The one clear difference is granite's own judge on closed-book chat: 0.640 against Qwen's 0.480, +0.159 [+0.053, +0.261]. It is not enough to lift the summed stack (+0.05, CI crosses 0).
- Caveat: the granite head is trained on 960 replies per fold, while the Qwen head is trained on about 4,860 (all stored features). Part of the 0.07 gap between the heads is training-set size, not reader quality.
- Verdict: a second 2B reader of another family is not worth its +80% compute at this tier. Its judge is a candidate chat-only expert, to be checked on more chat domains.

## What this means

**The 1.7B reader's verdict internals do not add held-out accuracy.** This covers the per-layer readout, alternative framings, attention and the question prior.
- Each signal carries some information alone (0.56-0.64).
- That information is a subset of what the reply head already has, which reads the judge state and the reply states.
- Equal-weight sums dilute it, and stacking a 64-168-d block overfits the 20 training domains.

This repeats the repo's pattern: the combination layer is saturated, and new accuracy needs new information (a bigger reader, external judges), not new views of the same pass.

**Two leads are not worth chasing at this tier:**
- the shift on closed-book chat (+0.07, CI crosses 0, 2 domains);
- the survey's suggestion to re-test idea 1 at 8B. The pre-stated condition, the mid band beating the final log-odds on ≥ 12/21 domains, failed (10/21).

**The polarity finding is the most useful by-product.** Never read a reversed judge question from a small reader as P(No). It answers the quality question regardless of the wording.
