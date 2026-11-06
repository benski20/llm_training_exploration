# Grounding verifier: Qwen/Qwen3-4B

Probe `probe_weighted.calibrated.npz` (sha256 `a242d250706293ed…`), reader `Qwen/Qwen3-4B`.

## What it does

Reads each sentence of a reply against the evidence it should rest on (the documents, the conversation, exact re-computation of stated arithmetic, parsed code) from the reader model's own neurons and probabilities, in two forward passes, and flags the sentences the evidence contradicts or does not hold. It checks grounding, not truth: a claim no evidence addresses is not marked wrong.

## Training

- Constructions: 600 from general (Dolly reference paragraphs), worded like answers, labels by design (no human labels)
- Held-out construction AUROC: 0.933

## Calibration

- On 300 labeled replies, threshold 0.5085 for a target of 10% false alarms

## Measured

On 300 labeled replies (`results/pipeline_demo/test300.jsonl`): AUROC 0.592; at the threshold 19% of 150 hallucinated replies caught, 19% of 150 clean ones flagged, precision 51%; 4.09 s a reply.

| source | AUROC | caught | false alarms |
|---|---|---|---|
| DROP | 0.451 | 32% | 44% |
| FinanceBench | 0.389 | 28% | 48% |
| RAGTruth | 0.698 | 24% | 12% |
| covidQA | 0.717 | 12% | 4% |
| halueval | 0.866 | 12% | 0% |
| pubmedQA | 0.638 | 8% | 4% |

## Intended use and limits

- For replies written from evidence: RAG answers, summaries, document Q&A, support bots on a knowledge base.
- Closed-book questions depend on the lookup finding the deciding fact; without it errors pass.
- Calibrate on your own replies; the false-alarm rate holds for data like the calibration set.
