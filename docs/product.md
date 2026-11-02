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
| HaluBench, 300 answers in 6 sources, head to head (`results/halubench_compare_qwen3_4b.json`) | see below |
| End to end, 13 whole replies with Wikipedia evidence | the wrong reply above the right one in all 6 pairs (two by thin margins) |
| Closed-book trivia through the full server | 1-2 of 13-14 wrong answers flagged: the lookup rarely finds the deciding fact |

HaluBench head to head, the same 300 answers (50 per source, half hallucinated), AUROC for "hallucinated":

| source | this verifier (4B reader) | MiniCheck (RoBERTa-Large) | Vectara HHEM-2.1-open |
|---|---|---|---|
| overall | 0.59 | **0.63** | 0.58 |
| caught at 10% false alarms | **19%** | 16% | 11% |
| RAGTruth (long-form RAG answers) | 0.88 | 0.66 | **0.89** |
| HaluEval | 0.67 | **0.74** | 0.72 |
| PubMedQA | 0.59 | **0.67** | 0.45 |
| CovidQA (long papers) | 0.50 | **0.83** | 0.48 |
| DROP (one-word answers) | 0.40 | **0.65** | 0.35 |
| FinanceBench (one-word answers) | 0.40 | 0.51 | **0.53** |
| CPU seconds per answer | 13.5 | 2.4 | 11.9 |

That was the first version. Two fixes followed from where it lost: long documents are reduced to their passages
most relevant to the reply instead of being cut at the end, and a fragment reply ("Rams", "$1.2 billion") is read
as one statement with its question. Because they were chosen after seeing those sources, they were confirmed on a
fresh HaluBench sample (`results/halubench_compare_qwen3_4b_seed1.json`, 300 other answers, seed 1):

| fresh sample | this verifier (4B reader) | MiniCheck (RoBERTa-Large) |
|---|---|---|
| **overall** | **0.72** | 0.64 |
| difference (95% CI) | **+0.09 [+0.005, +0.18]** | |
| **caught at 10% false alarms** | **35%** | 11% |
| RAGTruth | **0.92** | 0.65 |
| HaluEval | **0.90** | 0.73 |
| DROP | **0.67** | 0.45 |
| FinanceBench | 0.56 | 0.52 |
| CovidQA | 0.75 | **0.89** |
| PubMedQA | 0.59 | **0.81** |
| CPU seconds per answer | 5.9 (two passes) | 2.4 |

Ahead of MiniCheck overall by a margin outside chance on this sample, and three times as many hallucinations
caught at a 10% false-alarm rate; behind it on medical papers (CovidQA, PubMedQA).

A third sample (300 more answers, seed 4, `results/pipeline_demo/model_card_300.json` and
`minicheck_test300.json`, scored through the calibrated pipeline): 0.62 against MiniCheck's 0.61. Across the three
samples, then: 0.64 / 0.72 / 0.62 against 0.63 / 0.64 / 0.61. Level with or ahead of MiniCheck every time,
clearly ahead once; the honest claim is "at least as good, at a fraction of an LLM judge's cost", not "better".

Where it wins: answers written from documents, long or short. Where it still loses: medical papers (CovidQA,
PubMedQA) and numeric answers over financial filings (FinanceBench, near chance for all three systems).

## The pipeline, end to end (`results/pipeline_demo/`)

What a customer engagement produces, run on HaluBench as a stand-in for their data: `calibrate` on 120 labeled
replies (84 to set the scales and a threshold for a 10% false-alarm target, 36 held out), then `evaluate` on 120
other replies, which writes the model card (`results/pipeline_demo/model_card.md`). On the 120 test replies:
AUROC 0.66; at the calibrated threshold 33% of hallucinated replies caught and 13% of clean ones flagged (target
10%), precision 71%; 4.8 s a reply on CPU. Per source (20 replies each, so wide): PubMedQA 0.88, CovidQA 0.89,
RAGTruth 0.85, HaluEval 0.81, DROP 0.36, FinanceBench 0.48. The false-alarm rate carried from calibration to new
data within a few points; the 36 held-out calibration replies were too few to measure it (24% on 17 clean).
With ~100 replies a customer gets a threshold that roughly holds; a firm error rate needs a few hundred. On 300
further replies the same threshold flagged 19% of clean ones (target 10%), caught 23% of hallucinated ones,
precision 55%, AUROC 0.62: a threshold set on 84 replies mixed across six sources does not hold its rate on new
data. A calibration set should come from one customer's own traffic and number a few hundred replies.

Calibrating on 300 replies and testing on 300 others (`results/pipeline_demo/threshold_only_cal300.json`): with the
probe's own signal scales kept and only the threshold set, **8.7% of clean test replies flagged against the 10%
target**, 19% of hallucinated ones caught, precision 68%, AUROC 0.65; re-deriving the scales from the 300 mixed
calibration replies instead gave 13% false alarms, 14% caught, AUROC 0.62. The pipeline now keeps the probe's
scales by default. What this says about the product on HaluBench's mix: at a false-alarm rate a customer can
live with, it catches about one hallucination in five overall, far more on some sources (HaluEval, CovidQA) than
others (DROP, FinanceBench near chance). Sold as a guardrail for answers written from documents it has to be
calibrated and measured on the customer's own replies, and its value rests on catching a share of errors cheaply
on every reply, not on catching them all.

Through the API itself (`vt-verify` with the calibrated probe, the SDK over HTTP with a key): a correct summary
of a short article reads consistent (total 1.05 under the threshold 2.14), one with the wrong month and cost reads
contradicted (2.35, p = 0.59); a bad key gets 401; tokens are metered per key. 1.4 s a check on CPU once warm
(the first request loads the reader: 60 s).

## Pricing and unit economics

Pricing: per 1,000 tokens verified (reply plus evidence; the API meters it per key), an annual self-hosted
licence for regulated customers (container plus pipeline), and calibration per domain (the model card is the
deliverable).

Cost per check, estimated from measured token counts (an estimate until the hosted endpoint runs on a GPU):

| quantity | value | source |
|---|---|---|
| tokens read per check | 650 mean, 400 median, 1,900 at the 95th percentile, plus ~60 for the second pass | measured on HaluBench (fresh sample) with passage selection |
| compute per check (4B reader) | 2 x 4e9 parameters x ~720 tokens ≈ 6e12 FLOPs | arithmetic |
| one L4 GPU, prefill at ~30-40% of its ~120 bf16 TFLOP/s | ≈ 0.15 s a check, ≈ 20,000 checks an hour | assumption: typical prefill utilisation |
| L4 rental | ≈ $0.80 an hour on Modal; $0.39-0.43 at the cheapest providers | third-party price trackers, Aug-Oct 2026 ([amcompute](https://www.amcompute.com/gpu-cloud-prices/l4), [computeprices](https://computeprices.com/providers/modal/gpus/l4)) |
| **compute cost** | **≈ $0.00004 a check, ≈ $0.04 per 1,000 checks** | |
| CPU only (4 cores, measured) | 1.4 s a check on a short document, 5.9 s on HaluBench's | measured |

For comparison, an LLM judge reads the same evidence once per claim, with several claims per reply: at
GPT-4o-mini's $0.15 per million input tokens about $0.0005 a reply, at GPT-4o's $2.50 about $0.01 (third-party
price trackers, 2026, e.g. [pricepertoken](https://pricepertoken.com/pricing-page/model/openai-gpt-4o-mini); GPT-4o
is listed as legacy, newer flagships may cost differently; batch APIs halve these). So the compute cost is roughly
10 to 250 times lower than an LLM judge, before margin, and a 4-core CPU box can serve low volumes (about 2,500
short checks an hour) with no GPU at all. Selling at $0.50-$1.00 per 1,000 checks would leave a large gross margin
while staying well under LLM-judge prices; the per-token meter in the API supports either.

## What has to happen before selling

1. Close the remaining gaps (medical papers, numbers in filings); add LLM-AggreFact to the benchmark.
2. A larger reader: the Kaggle kernel that trains and scores the probe on Qwen3-32B is ready (`kaggle/grounding`).
3. GPU latency and cost per 1,000 tokens on the hosted endpoint.
4. A design partner's labeled replies for a real calibration and model card.
