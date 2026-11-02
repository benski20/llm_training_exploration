# Grounding verifier: Qwen/Qwen3-4B

Probe `probe.calibrated.npz` (sha256 `1ea83ecd3588b82e…`), reader `Qwen/Qwen3-4B`.

## What it does

Reads each sentence of a reply against the evidence it should rest on (the documents, the conversation, exact re-computation of stated arithmetic, parsed code) from the reader model's own neurons and probabilities, in two forward passes, and flags the sentences the evidence contradicts or does not hold. It checks grounding, not truth: a claim no evidence addresses is not marked wrong.

## Training

- Constructions: 600 from general (Dolly reference paragraphs), worded like answers, labels by design (no human labels)
- Held-out construction AUROC: 0.933

## Calibration

- On 84 labeled replies, threshold 2.141 for a target of 10% false alarms

## Measured

On 120 labeled replies (`results/pipeline_demo/test.jsonl`): AUROC 0.66; at the threshold 33% of 60 hallucinated replies caught, 13% of 60 clean ones flagged, precision 71%; 4.77 s a reply.

| source | AUROC | caught | false alarms |
|---|---|---|---|
| DROP | 0.36 | 30% | 50% |
| FinanceBench | 0.48 | 10% | 20% |
| RAGTruth | 0.85 | 60% | 10% |
| covidQA | 0.89 | 30% | 0% |
| halueval | 0.81 | 10% | 0% |
| pubmedQA | 0.88 | 60% | 0% |

## Intended use and limits

- For replies written from evidence: RAG answers, summaries, document Q&A, support bots on a knowledge base.
- Closed-book questions depend on the lookup finding the deciding fact; without it errors pass.
- Calibrate on your own replies; the false-alarm rate holds for data like the calibration set.
