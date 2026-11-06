# Claim contrast (idea 1): each claim against its own alternatives (`experiments/claim_contrast.py`)

**Idea.** Remove the score's scale, which drifted by domain in every earlier test, by comparing each claim with
versions of the same reply that swap in other spans of the same type from the same evidence:
- a number for another number of the same kind (money, percentage, year, plain);
- a name for another name;
- yes for no or maybe.

Each variant is scored by the evidence's support for it: log p with the evidence minus log p without.
- **Fixed rule, no fitted threshold:** flag an answer when a written claim is not the best-supported version of
  itself.
- **Setup:** Qwen3-4B on the same HaluBench sample (seed 1). The run was stopped at 163 of 300 answers because the
  result was already clear.

## Result (results/claim_contrast_qwen3_4b.json; 142 answers had a checkable claim)

| source | AUROC (rank) | rule: false alarms | rule: caught |
|---|---|---|---|
| DROP | 0.40 | 100% | 92% |
| FinanceBench | 0.55 | 100% | 100% |
| PubMedQA | 0.38 | 100% | 100% |
| CovidQA | 0.68 | 50% | 86% |
| HaluEval | **0.86** | 56% | 92% |
| all | 0.65 | **77%** | 95% |

**The rule does not hold a false-alarm rate. It flags most correct answers.** The reason shows in the correct
answers' own claims, and how often each is ranked first:

| claim | ranked first among its alternatives |
|---|---|
| copied from the passage: RAGTruth names and numbers, CovidQA numbers, HaluEval numbers | 81-100% |
| computed: DROP numbers, FinanceBench figures | 0-25% |
| judgements: PubMedQA yes/no | 0% |

- **The alternatives come from the passage, so the evidence literally contains them.** A computed or judged answer,
  correct or not, is less supported than spans the reader can copy. The null is not neutral; it favours copying.
- **The test works only where answers are copied spans.** There, HaluEval reaches 0.86, the best single
  training-free number on that source. But that is where hallucination detection was already easiest.
- **It fails for what remained hard:** computed numbers, figures from filings, and judgements.

## What the experiments since the review settle

Four training-free routes have now been tested for a verifier that holds a stated false-alarm rate without
domain labels:
- copying heads and source flow with equal weights (docs/source_flow.md);
- the generator reading its own answers (docs/self_grounded.md);
- a pooled calibration across domains (docs/source_flow.md);
- claims against their own alternatives (this).

None holds one: every score's meaning moves with the domain and the form of the answer.

What has worked is labels from the target domain:
- the supervised probe on RAGTruth (F1 70.4 from 800 answers, still rising);
- few-shot adaptation of the trained detector (VISION.md: DROP 0.63 → 0.85 with 25 labelled replies and 0.92 with
  200; PubMedQA 0.62 → 0.82);
- per-domain recalibration (calibration error 0.105 → 0.032 with 200 labels).

FinanceBench stayed near chance in every setting.
