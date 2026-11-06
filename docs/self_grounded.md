# The model's own answers from documents, read with training-free signals (`experiments/self_grounded.py`)

Idea 2 from the review: verify where the text is written, reading the generating model's own internals.

Its closed-book form was already done, and is not repeated here (VISION.md): trained probes on Qwen2.5 0.5B-7B and
R1-7B reached 0.78-0.86, against 0.65-0.76 for token confidence. They lost calibration on a new dataset and gave
only a modest gain inside generation.

Untested until now: the model answering *from documents*, read with signals that need *no training*, and whether
one fixed setting holds a false-alarm rate across domains.

## Setup

- **Model and answers:** Qwen3-4B answers 80 questions per domain from the passage (greedy, at most 60 tokens).
  Each answer is matched to the gold answer:
  - by number for DROP and FinanceBench (FinanceBench allows any unit scale);
  - by yes/no/maybe for PubMedQA;
  - by words for HaluEval QA.
- **Reading:** each answer is read once with the passage and once without.
- **Results:** results/self_grounded_qwen3_4b.json, per answer in the .jsonl beside it.

## Results

| domain | accuracy | own confidence: low min log p | high mean entropy | evidence shift (low) | copying-head conflict | uncopied words | reply − sink | combination |
|---|---|---|---|---|---|---|---|---|
| DROP (numbers from a passage) | 43% | **0.66** | 0.66 | 0.47 | 0.40 | 0.57 | 0.51 | 0.58 |
| FinanceBench (filings) | 35% | 0.83 | **0.89** | 0.74 | 0.67 | 0.79 | 0.59 | 0.83 |
| HaluEval QA (Wikipedia) | 73% | 0.64 | **0.66** | 0.43 | 0.55 | 0.57 | 0.46 | 0.60 |
| PubMedQA (judgements) | 66% | 0.60 | 0.64 | 0.33 | 0.59 | 0.72 | **0.74** | 0.52 |

AUROC for "wrong". The combination is the sum of low min log p, max entropy, low margin, low evidence shift and
copying-head conflict, each on its domain's own unlabelled scale.

**On its own answers, its own confidence works.** It scores 0.60-0.89, against about 0.5 when the same reader
judges other models' text. This matches the earlier closed-book work.

**The mechanistic signals add nothing without training.**
- The evidence shift points the wrong way in three domains: the model's own wrong answers are not less supported
  by the passage in its own eyes.
- Copying heads and source flow help only here and there (uncopied words on PubMedQA 0.72, reply − sink on
  PubMedQA 0.74).
- The fixed combination is no better than confidence alone.

**No fixed setting holds a false-alarm rate across domains.** A threshold set for 10% false alarms on one domain's
right answers gives these false-alarm rates elsewhere (min log p):

| threshold set on | DROP | FinanceBench | HaluEval QA | PubMedQA |
|---|---|---|---|---|
| DROP | 12% | 11% | 5% | 23% |
| FinanceBench | 18% | 11% | 9% | 45% |
| HaluEval QA | 21% | 18% | 10% | **64%** |
| PubMedQA | 9% | 4% | **0%** | 11% |

- Entropy and the combination behave the same way, up to 93%.
- The scale moves with the answer's form: PubMedQA answers average 40 tokens, DROP answers 4. A longer answer has
  a lower worst-token probability whether or not it is right.

## Caveats

- 80 answers per domain. The false-alarm rates rest on 26-58 right answers each.
- **FinanceBench is inflated:** some of its "wrong" answers are calculations cut off at the 60-token limit, which
  read as low-confidence.
- The labels are automatic. A few are arguable ("founded in 1886" against "the 1880s"; "opera" against "music").

## What this settles

- **Reading the generator's own internals does not escape the problem the reader approach hit.** Its confidence
  ranks its answers moderately well, but every score's scale moves with the domain and the answer's form.
- **This is the third independent test with the same outcome:**
  - the detector's leave-one-source-out runs;
  - HaluBench per source;
  - this.
- **The consequences:**
  - A verifier that holds a stated false-alarm rate has to be calibrated on labelled replies from the deployment
    domain.
  - Or its score must be compared with a null built from the same reply, so that its scale does not depend on the
    domain (idea 1: the claim against its own swapped alternatives).
