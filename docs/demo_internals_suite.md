# Demo: internals-only verification of document-grounded answers

Live at https://sandboxtesting101--verichat-serve.us-east.modal.run. Use **Held-out tests** in the sidebar.

## What runs

For each of six document-grounded datasets, the server fits at startup a fresh logistic probe on the frozen
Qwen3-8B reader's hidden states. The training data per dataset is:
- **50 human-labelled answers** (25 hallucinated, 25 faithful);
- 60 unlabelled answers, used only for standardisation (labels unused).

What the probe does not use:
- no LLM judge;
- no NLI model;
- no rules;
- none of the shipped head's weights (they were trained on the corpus).

Its decision threshold comes from cross-fitted scores on the same 50 labels.

Each check also runs the two 14B judges (Qwen3-14B, phi-4, one forward pass each, no labels) as a "second opinion".
The page scores both against the truth on **40 held-out answers per dataset (240 total)**. Those answers are used
nowhere in training. Data: `data/rag_suite.jsonl` (RAGTruth, HaluEval; MIT).

## Results (live run, 12 Nov 2026; `results/omni/rag_suite_live.json`)

| dataset (40 held-out answers each) | internals AUROC | judges AUROC | internals accuracy | judges accuracy |
|---|---|---|---|---|
| HaluEval: knowledge-grounded QA | **0.99** | 0.94 | **98%** | 85% |
| HaluEval: document summaries | **0.99** | 0.93 | **83%** | 50% |
| HaluEval: knowledge-grounded dialogue | **0.83** | 0.69 | **73%** | 53% |
| RAGTruth: business records to text | 0.74 | **0.88** | **70%** | 68% |
| RAGTruth: questions over web passages | 0.72 | **0.77** | 60% | **70%** |
| RAGTruth: news summaries | 0.71 | **0.84** | **65%** | 60% |
| **mean of the six** | 0.83 | **0.84** | **75%** | 64% |
| all 240 pooled | **0.84** | 0.79 | **75%** | 64% |

- AUROC is the ranking quality of each method's score; accuracy uses each method's own cut-off.
- The probe's cut-off was learned from its 50 labels.
- The judges' cut-off is their natural Yes/No. They have no labels to set it from, and that is part of the comparison:
  a judge cannot use a customer's labels, a probe can.
- n = 40 per dataset: a per-dataset AUROC carries roughly ±0.08.

**Reading.**
- With 50 labels, internals from a single 8B reader tie two 14B judges on ranking (0.83 vs 0.84, mean of the six).
- They beat the judges on decisions: 75% vs 64% accuracy.
- They win clearly on HaluEval, and lose on ranking on RAGTruth (0.72 vs 0.83, mean of three).
- **Caveat on HaluEval.** Its hallucinated answers were written by ChatGPT on instruction, so a probe can partly learn
  their style. Near-perfect HaluEval scores should be treated as an upper bound.
- RAGTruth, natural answers from real models with human span labels, is the realistic test. At 50 labels the judges
  still rank better there.
- Earlier work in this repository suggests the gap closes with more labels. The 4B/8B reader probe trained on the full
  RAGTruth training set reached AUROC 0.92 on its official test set: `docs/ragtruth_sota.md`.

## A demo script (about 10 minutes)

1. **Held-out tests → All six → Run this set** (about 4 minutes for 240 answers).
   - The scoreboard fills in live: internals vs judges, AUROC and accuracy.
   - Point out that the internals verdict uses one 8B model, 50 labels per dataset, and no judge.
2. **HaluEval: document summaries.**
   - The judges call about half the faithful summaries wrong (accuracy 50%).
   - The probe, calibrated on 50 labels, gets 83%.
   - This shows the commercial point: a judge's threshold cannot learn your data; the probe's can.
3. **Where internals and judges disagree** (after a run): 109 answers.
   - The probe is right in 67 of them, the judges in 42.
   - Open a few of each, e.g. RAGTruth news summaries that the judges flag but are faithful: ids 402, 537, 459, 560.
4. **RAGTruth: questions over web passages.**
   - Show the honest loss. With 50 labels the judges rank these better (0.77 vs 0.72).
   - Example misses by the probe: ids 202 (joists), 115 (time zones). The answer looks fluent; the error is a number
     or a detail.
5. **Verify an answer → Check as: Internals only, ….** Paste any document and answer for a live check. The judges'
   view shows below as a second opinion.

## What it shows, in one line

A probe on one small open model's internals, trained on 50 of a customer's labels, ties two larger LLM judges at
ranking. It makes better decisions, because it learns the customer's threshold, and it wins on the customer's
"style" of error. Judges still rank better on realistic RAG answers at this label count.

## Five examples to walk through (Held-out tests → pick the dataset → Run this set → open the answer)

| # | dataset, answer starts with | truth | internals (no judge) | two LLM judges | the point |
|---|---|---|---|---|---|
| 1 | RAGTruth questions, "Based on the given passages, here is a brief answer on how to set achievable goals" (id 208) | hallucinated: adds "SMART goals" and ties the 80/20 rule to a step the passages never link it to | **flagged**; suspect sentence is the SMART one | missed (lean right) | catches content that sounds right but is not in the source |
| 2 | RAGTruth questions, "Based on the provided passages, here are some tips for growing healthy tomatoes" (id 29) | hallucinated: ends by saying the passages give no advice on staking, but they do (put the supports in first) | **flagged** | missed, confidently (−11.7) | a subtle false claim about the source, in a long mostly-faithful answer |
| 3 | RAGTruth news summaries, "King Charles has been advised to remain dignified" (id 402) | faithful | **cleared** | false alarm (+6.8) | judges' false alarms are what make reviewers stop trusting a checker |
| 4 | HaluEval dialogue, "Yes, it was" (to "Was Find Me Guilty written by Sidney Lumet?"; id 1218) | faithful: the knowledge says Lumet wrote it | **cleared** | false alarm (+13.9) | judges bring outside beliefs; the probe checks agreement with the given source |
| 5 | RAGTruth questions, "The time difference between Chicago (CST) and Bangkok (ICT) is currently 13 hours" (id 115) | hallucinated: the last pair, 10:00 pm ICT = 11:00 am CST, is not in the passages (and should be 9:00 am) | missed | **caught** (+31.5) | honest limit: one wrong number at the end of an answer copied from a table |
