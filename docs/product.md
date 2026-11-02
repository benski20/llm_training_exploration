# Grounding verifier: product

A verification API that sits behind any chatbot, RAG system or document assistant. It takes a reply and the
evidence the reply should rest on, and says, sentence by sentence, which statements the evidence contradicts or
does not hold. It reads a small open "reader" model's internals in two forward passes; nothing is generated and no
LLM is asked for an opinion. It works with replies from any model, open or closed.

## Who buys it

Teams shipping answers written from documents, where an unsupported sentence has a cost:

- support bots on a knowledge base, internal "ask the handbook" assistants;
- legal, compliance, finance and healthcare document Q&A and summaries (the strongest need for an audit trail,
  and for running on their own hardware);
- RAG platforms and LLM observability tools that want a cheap, fast grounding signal to embed.

What they get: a per-sentence verdict with the evidence it was checked against, a calibrated false-alarm rate
on their own traffic, and a reply-level probability of a hallucination, cheap enough to run on every reply.

## The product

| piece | what it is | where |
|---|---|---|
| Verification API | `POST /v1/verify` (and `/batch`): reply, question, documents in; verdict, per-sentence verdicts, probability, usage out. API keys, per-key rate limits, tokens verified per key (billing), Prometheus metrics, size limits | `vt/verify_service.py` |
| Client SDK | standard-library Python client with retries | `vt/verify_client.py` |
| Probe pipeline | `train` (any reader, no human labels) → `calibrate` (customer's labeled replies, chosen false-alarm rate, measured on a held-out part) → `evaluate` (error rates, model card) | `vt/verifier_pipeline.py` |
| Deployment | container image (CPU or CUDA), hosted endpoint on Modal (one GPU, keys from a secret, idle shutdown) | `Dockerfile.verifier`, `deploy/modal_verifier.py` |
| In-chatbot integration | the same check inside the VeriChat server, in place of an LLM judge | `vt/verichat.py`, `--grounding` |

```bash
pip install ".[verifier]"
vt-verifier-pipeline halubench --n 120 --out labeled.jsonl                    # or your own labeled replies
vt-verifier-pipeline calibrate --reader Qwen/Qwen3-4B --probe heads/grounding_probe_qwen3_4b.npz \
    --labeled labeled.jsonl --out probe.calibrated.npz --false-alarms 0.05
vt-verifier-pipeline evaluate --reader Qwen/Qwen3-4B --probe probe.calibrated.npz --labeled test.jsonl --card card.md
VERIFY_API_KEYS="acme:$KEY" vt-verify --reader Qwen/Qwen3-4B --probe probe.calibrated.npz --host 0.0.0.0
```

```python
from vt.verify_client import VerifyClient
r = VerifyClient("https://verify.example.com", api_key=KEY).verify(reply=answer, question=q, documents=docs)
r["verdict"], r.get("p_hallucinated"), [s for s in r["sentences"] if s["verdict"] != "consistent"]
```

## How it works, and why it can be cheap

1. The reader reads the reply once with the evidence in its context (documents; for long ones, their passages
   most relevant to the reply; exact re-computation of stated arithmetic; code parse results), and once with the
   question alone.
2. Per sentence, three readings: a linear probe on its MLP neurons (trained without human labels, on sentences
   whose grounding is known by construction), how much the evidence raises the words written, and whether the
   evidence makes it prefer another word (the mark of a contradiction).
3. On their natural scales (set on the customer's own replies, labels not needed) they are summed; the reply's
   total, against a threshold set for a chosen false-alarm rate, gives the verdict and a probability.

Two forward passes of a 4B model per reply, against an LLM-judge call per claim. No generation.

## What has been measured (docs/grounding_trace.md, CPU, Qwen3-4B reader)

| test | result |
|---|---|
| RAGTruth, human-labeled hallucinations in news, Q&A and reviews (never trained on) | 0.82 [0.76, 0.88] AUROC per response, 0.77 per sentence; a detector trained on RAGTruth itself: 0.86 |
| HaluBench, 300 answers in 6 sources, head to head with MiniCheck (RoBERTa-Large) | overall 0.59 against 0.63; RAGTruth-style long answers 0.88 against 0.66; one-word answers (DROP, FinanceBench) 0.40 against 0.65 and 0.51; long papers (CovidQA) 0.50 against 0.83 |
| End to end, 13 whole replies with Wikipedia evidence | the wrong reply above the right one in all 6 pairs (two by thin margins) |
| Closed-book trivia through the full server | 1-2 of 13-14 wrong answers flagged: the lookup rarely finds the deciding fact |

Where it wins is where it is built for: answers written in sentences from documents. Where it loses is known:
one-word extractive answers (there is no sentence to read) and documents longer than its evidence budget (fixed
since: long documents are reduced to their most relevant passages; re-measurement below).

## Pricing and unit economics (to be confirmed by measurement on GPU)

- Usage: per 1,000 tokens verified (reply plus evidence), metered per key by the API.
- Self-hosted licence for regulated customers: annual, per deployment, the container and pipeline included.
- Calibration service: per domain, on the customer's labeled replies; the model card is the deliverable.

Cost per verification is two forward passes of the reader over evidence plus reply. On CPU it is 13.5 s for a
HaluBench answer with a 4B reader; on a GPU the same work is a fraction of a second, which has to be measured
(the hosted endpoint is written, not yet deployed). An LLM-judge call per claim is the comparison to beat.

## What has to happen before selling

1. Close the gaps the benchmark found (short answers, long documents) and re-measure on HaluBench; add LLM-AggreFact.
2. A larger reader: the Kaggle kernel that trains and scores the probe on Qwen3-32B is ready (`kaggle/grounding`).
3. GPU latency and cost per 1,000 tokens on the hosted endpoint.
4. A design partner's labeled replies for a real calibration and model card.
