# Writing the readout without labels, and how many labels the guarantee needs (`experiments/written_readout.py`)

Prompted by "Sparks of Weight-Space Programming" (MIT CSAIL, 2026): could the readout for a new domain be written
instead of fitted from labels? The domain is RAGTruth QA, with the shipped check over the six signals of
docs/source_flow.md.

**Where weight-writing applies.**
- The readout over those six signals: a handful of weights, written from reasoning and from 300 *unlabeled*
  replies of the domain.
- Writing a neuron-level probe for a 4B reader by hand is not realistic, and was not attempted.
- The exact parts (arithmetic, copying, parsing) are better written as code, as they already are.
- The weights were frozen and committed (`results/written_readout/written_weights.json`, commit 68d2070) before
  any test label was read.

**What the unlabeled replies showed** (results/written_readout/explore.json; 2,716 sentences):
- **One axis, "not drawn from the evidence":** the probe score, low evidence shift and uncopied words correlate
  (0.25-0.59).
- **Copying-head conflict runs against that axis** (−0.58 with uncopied words). In this domain it marks engagement
  with the evidence rather than contradiction, so it got a negative weight.
- **A second axis, contradiction:** disagreement, which goes with copy conflict.

## Result on 300 labelled test replies (results/written_readout/evaluate.json)

| readout, no labels from the domain | AUROC | label-free threshold (two-component mixture): false alarms / caught |
|---|---|---|
| shipped check | **0.765** | n/a |
| written weights | 0.755 | 25% / 58% |
| "not drawn from the evidence" axis | 0.753 | 26% / 59% |
| first principal component | 0.758 | 46% / 83% |
| shipped signals, worst sentence, domain scales | 0.693 | 11% / 27% |

- **Written weights tie the shipped check on ranking.** The structure is real, but it carries no information the
  existing signals lack.
- **A label-free threshold does not hold.** The mixture's estimate of the clean share ranged from 39% to 74%
  across readouts, so false alarms came out at 11-46% against a 10% target.

## How many labels the guarantee needs (results/written_readout/label_budget.json)

Method:
- Draw k labelled calibration replies (balanced RAGTruth QA training replies), 300 draws per k.
- Set the threshold.
- Score the 300 test replies.

| labelled replies | 95%-confidence threshold: false alarms (mean; worst 5%) | share of draws over 12% | caught | plain quantile: false alarms (mean) | share over 12% |
|---|---|---|---|---|---|
| 50 | 4.5%; 16% | 7% | 22% | 15.9% | 69% |
| 100 | 4.4%; 11% | 3% | 22% | 15.9% | 78% |
| 200 | 7.2%; 11% | 2% | 32% | 15.8% | 92% |
| 300 | 10.7% | 0% | **42%** | 16.0% | 100% |

- **The bounded threshold holds the promise from 100 labelled replies on.** More labels buy recall: 22% → 32% → 42%
  caught at 100 / 200 / 300 labels.
- **The plain quantile overshoots at every size** (16% against 10%). Its error is a shift between RAGTruth's
  training and test replies, which more labels cannot fix and only the bound absorbs.

## Conclusion

- **Labels remain necessary, but only for the threshold.** Ranking needs none: the shipped check's 0.765 and the
  written weights' 0.755 are both label-free.
- **The commercial unit is "N of your labelled replies buys a guaranteed false-alarm rate at a stated recall".**
  On RAGTruth QA, 100 labels give a held 10%, and 300 labels give it with 42% of hallucinations caught.
