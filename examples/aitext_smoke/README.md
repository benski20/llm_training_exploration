Smoke-test texts for the AI-text container (`Dockerfile.aitext`, `deploy/aitext_smoke.py`), from HAP-E
(browndw/human-ai-parallel-corpus, MIT licence): news texts, cut to at most 1,500 characters.

* `human_calibrate.jsonl`: 150 human-written texts (the calibration set)
* `human_audit.jsonl`: 50 other human-written texts (the audit should come back "consistent")
* `ai.jsonl`: 20 continuations by Llama-3-8B-Instruct of other news texts (most should be flagged)
