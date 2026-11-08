# Which replies should a customer label first? (`experiments/novel_active_labels.py`)

Stored held-out scores only, no model run:
- `results/omni/expert_scores_q8b.npz`: reply head, worst unit and the reader's own judge;
- `results/omni/judge_probs.npz`: the Qwen3-14B and phi-4 judges;
- `/home/user/omni/nli.npz`: NLI scores, on grounded replies.

Results are in `results/omni/novel_active_labels.json`. A scheme sees a label only for a reply it has chosen.

## (a) Adaptation: refitting the expert weights on k labels

**Setup.**
- The features are the z-scored experts: reply, worst unit, the reader's own judge, Qwen3-14B, phi-4, and NLI where
  it exists.
- The weights are refitted as in `omni_stack.py` `adapt_k`: balanced weights, pulled toward the zero-label weights.
- Per domain and draw, the pool is split at random: 60% are candidates for labelling and 40% are a test set that
  selection never sees. Scoring on a fixed test set avoids the bias of scoring "the rest", which would have the
  boundary cases removed.
- 100 draws per domain, 21 domains. The 100-label column covers only 9 domains at 20% prevalence and 19 at the
  natural rate.

**Mean AUROC on the test set, all 21 domains, pool at 20% prevalence:**

| strategy | 10 | 25 | 50 | 100 |
|---|---|---|---|---|
| zero labels | 0.839 | 0.839 | 0.839 | 0.825 |
| uniform (today) | 0.835 | 0.843 | 0.852 | 0.866 |
| uncertainty, static (80th percentile of the zero-label score) | 0.843 | 0.845 | 0.852 | 0.865 |
| uncertainty, sequential (10 seeds, then batches of 5 nearest the boundary) | 0.835 | 0.846 | 0.855 | 0.867 |
| uncertainty + uniform, half each | 0.838 | 0.846 | 0.853 | 0.867 |
| disagreement (external judges vs reader internals) | 0.842 | 0.848 | 0.854 | 0.866 |
| **diversity** (k-center in expert-score space) | **0.843** | **0.850** | **0.855** | 0.866 |

**Paired difference from uniform** (mean over domains ± standard error, and the number of domains won):

| strategy | 10 labels | 25 labels | 50 labels |
|---|---|---|---|
| diversity | +0.008 ± 0.002 (17/21) | +0.007 ± 0.002 (16/21) | +0.003 ± 0.001 (15/21) |
| disagreement | +0.007 ± 0.003 (15/21) | +0.005 ± 0.003 (15/21) | +0.001 ± 0.001 |
| uncertainty, sequential | +0.000 | +0.003 ± 0.002 | +0.003 ± 0.001 (15/21) |

At the natural 50% rate the gains are the same size: diversity is +0.004 at 25 labels and sequential uncertainty
+0.003 at 25-50. Static uncertainty sampling **loses** 0.02, because it collects near-duplicates of one class.

**Verdict.**
- Smart selection for adaptation is worth **+0.003 to +0.008 AUROC**.
- That is about the gain of 10-20 more uniform labels: diversity at 25 labels (0.850) is close to uniform at about
  45 (0.852 at 50).
- It also avoids the dip that 10 uniform labels cause (0.835, below the zero-label 0.839).
- **Use k-center diversity**: it needs no labels to choose and no refits. The gains are real but small, because
  refitting only the expert weights has little room to move (see `omni_verifier.md`).

## (b) The guarantee: at most 10% false alarms with 95% confidence

**Protocol.** As in `judge_calibration_sim.py`:
- per domain, a pool of all clean replies plus wrong ones at 20% prevalence;
- the same 14 domains as the shipped baseline;
- 200 draws per domain;
- B labels, then a threshold set on them;
- the realised false-alarm rate (FAR) and recall are measured on the whole pool, which stands for the traffic.

There are two frames, and they agree:
- **bootstrap (primary):** labels are drawn with replacement, so the traffic is the pool's distribution. This is a
  large population, as for a customer with thousands of unlabelled replies.
- **finite:** labels are drawn without replacement.

**Replication.** In the shipped frame (100 uniform labels, measured on the unlabelled rest) the result is 4.8% false
alarms, 5% of draws over 12% and 45% recall. The documented baseline is 4.9% / 4.5% / 45%.

In the population frames, uniform labelling has only about 1% of draws over 12%: the noise of a small test set is
gone. **The validity criterion here is the share of draws with FAR over 10%, which the promise caps at 5%.**

### The valid scheme: score-stratified labelling with a conservative stratified Clopper-Pearson bound

1. Score every unlabelled reply. This is free.
2. Split them into two strata: H, the top q of scores (q = 0.15), and L, the rest.
3. Label f·B replies uniformly within H and the rest uniformly within L (f = 0.5).
4. For thresholds t inside H, bound the false-alarm rate:

   FAR(t) ≤ P(H) · CPup(#clean labelled in H with s ≥ t ; m_H, 0.03) / [P(L) · CPlo(#clean in L ; m_L, 0.01) + P(H) · CPlo(#clean in H ; m_H, 0.01)]

   - P(H) and P(L) are the stratum shares of the unlabelled traffic.
   - The three error terms add up to 0.05 by Bonferroni.
5. Scan thresholds from the top (fixed-sequence). Every term is an exact finite-sample bound, so the promise holds
   whatever the score's quality. Labels are spent where the threshold lives, and the lightly labelled L stratum only
   needs to show it is mostly clean.

**All 14 domains, 8B stack score (the shipped baseline score), bootstrap frame.** Each cell is mean FAR / share of
draws with FAR over 10% / recall.

| labels | uniform + Clopper-Pearson | **stratified (q 0.15, f 0.5)** | stratified (q 0.2, f 0.67) | lowest score first (biased) | judges' most-clean first (biased) |
|---|---|---|---|---|---|
| 30 | infeasible (fewer than 29 clean) / 0% recall | **1.7% / 0% / 33%** | 1.2% / 0% / 27% | 52.6% / 67% / 67% | 18.1% / 68% / 63% |
| 40 | 2.4% / 2.2% / 32% | **2.5% / 0% / 38%** | 2.1% / 0% / 36% | 71.8% / 98% / 94% | 20.7% / 86% / 71% |
| 50 | 2.1% / 1.0% / 33% | **3.0% / 0% / 41%** | 2.8% / 0% / 40% | 67.0% / 100% / 94% | 24.5% / 93% / 74% |
| 60 | 3.3% / 2.4% / 39% | **3.4% / 0% / 43%** | 3.4% / 0% / 42% | 60.4% / 100% | 22.7% / 93% |
| 80 | 4.0% / 1.9% / 43% | 4.0% / 0% / 45% | 4.2% / 0% / 46% | 48.5% / 100% | 19.6% / 100% |
| 100 | 4.5% / 2.5% / 45% | 4.3% / 0% / 46% | 4.7% / 0% / **48%** | 36.6% / 100% | 16.1% / 100% |

**With both external judges added to the score (stack + judges):**
- uniform reaches 0 / 38 / 38 / 44 / 49 / 51% recall at 30 / 40 / 50 / 60 / 80 / 100 labels;
- stratified (q 0.15, f 0.5) reaches **38 / 42 / 45 / 47 / 49 / 50%**, with 0% of draws over 10%;
- stratified (q 0.2, f 0.67) reaches 53% at 100 labels.

**Per format, stratified (q 0.15, f 0.5), 8B stack score.** Recall; under every format the share of draws with FAR
over 10% is ≤ 0.1%.

| format (domains) | uniform 50 | uniform 100 | stratified 30 | stratified 50 | stratified 60 |
|---|---|---|---|---|---|
| reasoning (4) | 45% | 59% | 46% | 54% | 55% |
| long-form (2) | 36% | 51% | 37% | 47% | 49% |
| RAG (6) | 30% | 41% | 28% | 37% | 39% |
| chat (2) | 16% | 26% | 15% | 22% | 24% |

**Validity.**
- **Every stratified configuration is valid.** The grid covers q in {0.1, 0.15, 0.2, 0.3}, f in {0.5, 0.67, 0.8,
  1}, joint and conditional forms, both scores, both frames and every format. Draws with FAR over 10% reach at most
  0.5%, against up to 2.5% for uniform.
- **Labelling the replies most likely to be clean first is INVALID.** Ordering by the stack score gives 37-72% mean
  false alarms, with 67-100% of draws over 12%. Ordering by the judges gives 16-40% and 62-100%. Clean replies
  chosen for looking clean are the least alarming clean replies, so the threshold falls into the clean mass.
- **Inverse-probability weighting with a Kish effective-size Clopper-Pearson bound** stays valid but is very
  conservative: 23% recall at 60 labels, against 43% stratified. It is not recommended.

## Verdict

- **Guarantee: stratify, do not chase clean replies.**
  - The stratified bound keeps the promise in every draw tested, overall and per format.
  - It makes the guarantee **feasible at 30 labels**. Uniform labelling needs about 40 labels before it has the 29
    clean replies the bound requires.
  - **It reaches the same recall with about 20-25 fewer labels in the 30-60 range** (8B stack):

    | stratified labels (recall) | uniform labels for the same recall |
    |---|---|
    | 30 (33%) | about 50 |
    | 40 (38%) | about 58 |
    | 50 (41%) | about 70 |
    | 60 (43%) | 80 |

    That is 30-40% fewer labels.
  - At 100 labels the gain shrinks to +1 to +3 points of recall.
  - The bound is conservative: realised false alarms are about 3%, not 4.5%. Tighter allocation of the confidence
    budget is the obvious next step.
- **Adaptation:** k-center diversity sampling gives +0.003 to +0.008 AUROC over uniform, worth 10-20 labels at
  25-50 labels. It is small but consistent, and it costs nothing.
- **Caveats.**
  - Pools are small (150-250 replies), and the bootstrap frame assumes stratum shares are known from abundant
    unlabelled traffic.
  - The configuration (q, f) was chosen on these domains. Validity does not depend on that choice; the recall
    figures could be slightly optimistic.
