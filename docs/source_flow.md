# Source flow: one signal for answers, reasoning and long-form text (`vt/sourceflow.py`)

Copying heads (docs/copy_heads.md) only cover text lifted from evidence. Source flow is the general version: for
every written token, how much of what the model read came from each source.
- **Sink:** the opening and template tokens. Heads parked there move nothing, so the token comes from memory alone.
- **Question:** the user's request.
- **Evidence:** documents in context, if any.
- **Reply:** the output's own earlier text.

Each head's contribution from a position is its attention weight times the size of what it moves from there,
‖v W_O‖. The shares are read at every 4th layer, inside the same single forward pass: the attention is recomputed
from each layer's own projections in a hook, so the model's attention kernel is unchanged.

## Test (`experiments/source_flow.py`, Qwen3-4B, CPU, about 2-7 s an output)

| kind of output | data | unit | units (errors) |
|---|---|---|---|
| answers grounded in a passage | HaluBench, the seed-1 sample of 300 | answer | 300 (150) |
| reasoning, no documents | Math-Shepherd GSM8K/MATH solutions, labels from its rollouts | step | 707 (247) |
| long form, no documents | WikiBio GPT-3 biographies, all 238 | sentence | 1,908 (1,392) |

AUROC for "wrong" (results/source_flow_qwen3_4b.json; intervals: bootstrap over items, 95%):

| score | answers | reasoning steps | biography sentences |
|---|---|---|---|
| reader's own confidence (mean log p of the written tokens; baseline) | 0.45-0.55 | 0.49-0.51 | 0.41-0.43 (inverted) |
| **one label-free score: reply share minus sink share** | **0.585** [0.52, 0.64] | **0.640** [0.58, 0.69] | **0.554** [0.53, 0.59] |
| source flow, fitted within each kind (5-fold, split by item) | 0.633 | 0.668 | 0.667 |
| plus confidence and copying heads, fitted within each kind | 0.739 | 0.718 | 0.676 |
| source flow, fitted on the other two kinds only | 0.442 | 0.522 | 0.505 |

**One general signal.** The same label-free score works on all three kinds of output, with no training and the
same sign everywhere, and every interval lies above chance.
- Wrong text leans more on the output's own earlier text and parks less attention on the sink. A wrong reasoning
  step builds on what was just written rather than on what the question gives (question share AUROC 0.44 for
  steps and 0.38 for sentences, so more question means more likely right).
- The reader's own confidence carries nothing on these outputs, written by other models.

**The fine pattern is per domain.**
- A readout fitted within one kind of output reaches 0.63-0.67 from source flow alone, and 0.68-0.74 with
  confidence and copying heads.
- Fitted on two kinds and scored on the third, it falls to chance (0.44-0.52): which layers carry the signal
  differs by kind of output.
- So the product shape is the one the pipeline already has: one general label-free signal to start with, plus a
  short calibration per domain on a customer's labelled replies.

## Caveats

- Math-Shepherd's step labels come from rollouts and are noisy.
- WikiBio is 73% inaccurate sentences, and items come from one model, GPT-3.
- The label-free score is weak alone (0.55-0.64). It is a general starting point, not a verdict.
- HaluBench's per-source AUROCs from 50 answers each are noisy.

## Next

- Add `reply − sink` to the served check as a fourth label-free signal, next to the probe, the evidence shift and
  the disagreement, so closed-book answers and reasoning get a reading without evidence.
- Let `calibrate` fit the per-layer source-flow readout per domain (it already fits a threshold per domain).

## In the pipeline (`augment`, then `calibrate --fit-weights`; results/pipeline_demo/)

Setup:
- `augment` added the 16 copying heads and the source-flow layers to the shipped probe, with natural scales from
  150 unlabelled RAGTruth answers.
- `calibrate --fit-weights` then fitted the domain's weighting of all signals on 300 labelled HaluBench replies
  (calibration300). Every row below is scored on 300 *other* replies (test300).

| calibration | test AUROC | caught | false alarms (target 10%) | precision |
|---|---|---|---|---|
| threshold only, original signals (earlier demo) | 0.620 | 14.0% | 13.3% | 0.51 |
| weighted, original signals | 0.592 | 19.3% | 18.7% | 0.51 |
| **weighted, with copying heads and source flow** | **0.638** | **24.7%** | 18.7% | **0.57** |

- **The new signals help through the weighting.** Against the same weighting without them: +4.6 AUROC points,
  more caught, higher precision. Against the threshold-only demo: +1.8 points.
- **Copying-head conflict gets the largest weight** (0.76, against 0.27 for the probe). Per source, it lifts
  FinanceBench (0.38 to 0.55) and CovidQA (0.74 to 0.89).
- **Two problems remain:**
  - The weighted versions miss the false-alarm target on new replies (18.7% against 10%).
  - The fit set reads 0.735 and the test set 0.638, so 300 mixed replies over-fit seven weights.
- **Equal weights without labels don't help here.** On the seed-1 sample, equal weights left the
  worst-sentence AUROC level (0.718 against 0.723) and lowered the reply total (0.653 against 0.724).
- So the shipped default stays the original probe. The new signals are for customers who calibrate on their own
  labelled replies, preferably more than 300 from one domain.
