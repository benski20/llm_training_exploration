# White-box judges: probing the judges' internals instead of reading their Yes / No

**Question.** The two external judges (Qwen3-14B and phi-4, 4-bit, one forward pass each) are read today by
their output: log P(Yes) − log P(No) at the answer position. Does a linear probe on their hidden state at that
position beat the output on held-out domains? Does it add to it?

**Setup.**
- Script: `kaggle/omni/judge_internals.py`, run unchanged on Modal (2× H100, 9 minutes of extraction) through
  `deploy/modal_judge_internals.py`.
- Data: all 5,107 corpus replies, 21 domains; the corpus md5 matches the repository's.
- Probe: global z-score → PCA 256 (label-free) → per-domain standardisation from unlabelled replies →
  leave-one-domain-out L2 logistic.
- Outputs: full table in `results/omni/judge_internals.json` (raw kernel outputs, not committed, in `results/kaggle/omni-ji-modal/`).

## Results (mean of per-domain AUROC on held-out domains)

| | chat | long-form | RAG | reasoning | **all 21** |
|---|---|---|---|---|---|
| Qwen3-14B output | 0.767 | 0.859 | 0.813 | 0.903 | 0.837 |
| Qwen3-14B probe (3 middle layers) | 0.691 | 0.841 | 0.813 | 0.893 | 0.823 |
| phi-4 output | 0.792 | 0.835 | 0.776 | 0.899 | 0.815 |
| phi-4 probe (layer 24) | 0.682 | 0.827 | 0.803 | 0.893 | 0.814 |
| both judges, outputs | 0.794 | 0.864 | 0.811 | 0.918 | 0.842 |
| both judges, probes | 0.709 | 0.863 | **0.835** | 0.915 | 0.845 |
| **both judges, outputs + probes** | 0.775 | **0.876** | **0.838** | **0.922** | **0.857** |

- **A probe alone ties the output.** Per judge it is −0.014 / −0.001; with both judges, 0.845 against 0.842.
  It does not reliably beat it.
- **The probe adds to the output.** Outputs + probes beats outputs by +0.015 on held-out domains: 16 of 21 domains
  improve, domain-bootstrap 95% CI [+0.004, +0.028].
- **The gain is not uniform.**
  - The probes win on RAG and summarisation: HaluEval summary 0.73 → 0.92, DROP 0.64 → 0.71, finance 0.63 → 0.66.
  - They lose on closed-book chat: TruthfulQA 0.89 → 0.75.
  - The judges' internals carry grounding information that their Yes / No does not express. Their factual knowledge
    is better expressed by the answer itself.
- **In-domain, with labels, the probe is stronger.** With 5-fold probes trained inside each domain (the onboarding
  setting), Qwen3-14B scores 0.853 against 0.837 for its output, and phi-4 0.837 against 0.815. Some in-domain
  numbers (HaluEval QA / summary ≈ 0.99) likely include dataset artifacts.

**Verdict.** This is the first internals signal in this project that adds on top of the 14B judges with a CI above 0.
It is a modest gain (+0.015 overall, +0.03 on RAG), not a replacement for the judges. To deploy it:
- keep the judges' hidden state at the answer position (it costs nothing extra: same forward pass);
- add the probe to the score, weighted by format (more for RAG, none for closed-book chat).
