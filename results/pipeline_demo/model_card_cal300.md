# Grounding verifier: Qwen/Qwen3-4B

Probe `probe300.calibrated.npz` (sha256 `8594f521f0c25262…`), reader `Qwen/Qwen3-4B`.

## What it does

Reads each sentence of a reply against the evidence it should rest on (the documents, the conversation, exact re-computation of stated arithmetic, parsed code) from the reader model's own neurons and probabilities, in two forward passes, and flags the sentences the evidence contradicts or does not hold. It checks grounding, not truth: a claim no evidence addresses is not marked wrong.

## Training

- Constructions: 600 from general (Dolly reference paragraphs), worded like answers, labels by design (no human labels)
- Held-out construction AUROC: 0.933

## Calibration

- On 300 labeled replies, threshold 2.326 for a target of 10% false alarms

## Measured

On 300 labeled replies (`results/pipeline_demo/test300.jsonl`): AUROC 0.62; at the threshold 14% of 150 hallucinated replies caught, 13% of 150 clean ones flagged, precision 51%; 4.57 s a reply.

| source | AUROC | caught | false alarms |
|---|---|---|---|
| DROP | 0.465 | 24% | 24% |
| FinanceBench | 0.384 | 0% | 12% |
| RAGTruth | 0.653 | 36% | 24% |
| covidQA | 0.736 | 0% | 8% |
| halueval | 0.833 | 16% | 0% |
| pubmedQA | 0.704 | 8% | 12% |

## Intended use and limits

- For replies written from evidence: RAG answers, summaries, document Q&A, support bots on a knowledge base.
- Closed-book questions depend on the lookup finding the deciding fact; without it errors pass.
- Calibrate on your own replies; the false-alarm rate holds for data like the calibration set.
