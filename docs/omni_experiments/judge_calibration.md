# Can a judge's labels replace human labels for the false-alarm guarantee? (`kaggle/omni/judge_calibration.py`, `experiments/judge_calibration_sim.py`)

**Judges.** Qwen3-14B (4-bit, thinking off) and microsoft/phi-4 (4-bit) read every corpus reply once (5,107; documents
included for grounded replies) and give P(correct) from the next-token probability of Yes vs No. No generation.
Kaggle 2× T4, one judge per GPU: 1.6 / 1.8 s per reply. Neither judge is the reader (Qwen3-8B).

**Judge accuracy** (AUROC for "wrong", mean over the 21 domains, no training): Qwen3-14B 0.837, phi-4 0.815, their
mean 0.844. The 8B stack scores 0.802. Combined with it: 0.842 with no labels and 0.859 with 50 labels
(`results/omni/stack_plus_judges.json`). Chat goes 0.73 → 0.79.

**Calibration** (`results/omni/judge_calibration.json`). Per domain: 100 replies at 20% prevalence; the threshold on
the shipped score is set by each method; the true false-alarm rate and recall are measured on the rest; 200 draws.
All 21 domains:

| method | mean false alarms | draws over 10% / 12% | recall |
|---|---|---|---|
| human labels, Clopper-Pearson (baseline) | 4.9% | 9.1% / 4.5% | 45% |
| judge (ensemble) labels as truth | 16.6% | 64% / 54% | 64% |
| prediction-powered, 0 human labels | 19.1% | 80% / 73% | 69% |
| prediction-powered, 10 human labels | 9.7% | 40% / 33% | 53% |
| prediction-powered, 20 human labels | 6.2% | 22% / 16% | 45% |
| prediction-powered, 40 human labels | 5.9% | 18% / 11% | 47% |
| finite-sample bound, 10-40 human labels | n/a (never feasible) | n/a | 0% |

**Verdict.** A judge's labels cannot stand in for human labels:
- **Judge labels alone** miss the target in most draws: the judge calls some wrong replies clean, and those set the
  threshold too low.
- **Correcting with 20-40 human labels** gets the mean right but not the tails.
- **A bound valid whatever the judge's errors** never becomes feasible at 40 human labels.

The guarantee needs about 30-80 human-labelled clean replies, as the binomial floor says. The judges are still worth
running, as scoring experts, not as labellers.
