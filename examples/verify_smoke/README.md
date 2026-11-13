Smoke-test replies for the verification engine (`vt/verify_engine.py`, `deploy/verify_smoke.py`), from the
RAGTruth test split (ParticleMedia/RAGTruth, MIT licence; via wandb/RAGTruth-processed): question-answering
replies written by LLMs from retrieved passages, with RAGTruth's human labels. Documents under 6,000 characters.

* `good_calibrate.jsonl`: 80 replies with no labelled hallucination (the calibration set: one score needs 59+)
* `good_audit.jsonl`: 40 other such replies (the audit should come back "consistent")
* `unsupported.jsonl`: 20 replies with at least one labelled hallucinated span (`labelled_spans`)
