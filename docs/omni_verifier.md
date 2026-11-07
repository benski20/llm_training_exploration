# The omni verifier: one cheap pass, any format, any domain, with a stated false-alarm rate

**What it is.** A verifier that scores how likely an LLM reply is wrong:
- **any format:** a RAG answer, a closed-book chat answer, a long-form text, a step-by-step solution;
- **any generator**, open or closed: it only reads the text;
- **one forward pass** of a small frozen open reader (Qwen3-1.7B). There is no generation, no LLM judge and no resampling of the customer's model.

It adapts to a new domain in three steps, each cheaper than the last fails:
1. **Zero labels:** a few hundred of the domain's *unlabelled* replies re-centre the features (`profile`).
2. **25-100 labels:** the head is refitted, pulled toward the universal one (`adapt`).
3. **The guarantee:** a threshold that holds a false-alarm rate with stated confidence, from the domain's clean labelled replies (`calibrate`, Clopper-Pearson, cross-fitted).

The design comes from six literature surveys: [docs/research/README.md](research/README.md).

```bash
vt-omni train   --feats <corpus features> --corpus corpus.jsonl --parts mean judge --out heads/omni_qwen3_1p7b.npz
vt-omni onboard --head heads/omni_qwen3_1p7b.npz --name acme --unlabelled traffic.jsonl --labelled labelled.jsonl \
                --fpr 0.10 --confidence 0.95 --out profiles/acme.json
vt-omni evaluate --head heads/omni_qwen3_1p7b.npz --profile profiles/acme.json --labelled test.jsonl --card card.json
VERIFY_API_KEYS="acme:$KEY" vt-omni serve --head heads/omni_qwen3_1p7b.npz --profiles profiles/ --host 0.0.0.0
curl -H "Authorization: Bearer $KEY" -d '{"profile": "acme", "question": "...", "reply": "...", "documents": ["..."]}' \
     localhost:8000/v1/verify
# -> {"score", "flagged", "guarantee": {"fpr_target", "confidence", "clean_replies"}, "units": [{"text", "score"}], ...}
```

## How it works (`vt/omni.py`)

The reader reads `[documents +] question + reply`, then a fixed judge question appended *after* the reply: "Is the
reply above fully correct, accurate and supported?". The reader is causal, so the reply's own states are those of a
plain read, and the same pass also yields a judgement state and the reader's own log-odds of "Yes" (PRISM-style
framing, free).

Each pass produces:

| expert | what it reads | training |
|---|---|---|
| reply head | mean residual state over the reply at three middle layers, the judgement state at the same layers (PCA 256), output signals (token log-prob, entropy, margin), judge log-odds | logistic, 20 domains, each standardised by its own statistics |
| unit head → worst unit | the same per sentence / reasoning step / segment | logistic on 16,119 labelled units (RAGTruth spans, FELM segments, ProcessBench first wrong step) |
| exact arithmetic | every "a op b = c" in the reply, recomputed (`vt/mathcheck.py`) | none |

**Score.** The sum of the reply head's and the worst unit's z-scores, each standardised by the domain profile's own
statistics. The API also returns each unit's score, which says *which* sentence or step is suspect.

## The test: 21 domains, 4 formats, every number on a domain the head never saw

The corpus (`experiments/omni_corpus.py`) has 5,107 replies, label-balanced within each domain. AUROC is computed
within a domain, so a task-type shortcut cannot inflate it: knowing a reply is data-to-text tells you nothing here.

| format | domains |
|---|---|
| RAG | RAGTruth QA / summary / data-to-text (6 LLMs, human span labels), HaluEval QA / dialogue / summary, HaluBench DROP / PubMedQA / FinanceBench / CovidQA |
| closed-book chat | HaluEval general (real ChatGPT replies), TruthfulQA |
| long-form | FELM math / reasoning / science / world knowledge / writing (ChatGPT, segment labels) |
| reasoning | ProcessBench GSM8K / MATH / OlympiadBench / Omni-MATH (12 models' step-by-step solutions, first wrong step) |

Every result below is leave-one-domain-out (LODO): the heads are trained on the other 20 domains and scored on the
held-out one (`experiments/omni_eval.py`, `experiments/omni_final.py`; `results/omni/`).

### Accuracy with zero labels from the new domain (Qwen3-1.7B reader)

| format | reply head | worst unit | reader's own judge (no training) | **combined** |
|---|---|---|---|---|
| reasoning (4 domains) | 0.84 | 0.83 | 0.74 | **0.85** |
| long-form (5) | 0.71 | 0.77 | 0.64 | **0.75** |
| RAG (10) | 0.68 | 0.65 | 0.62 | **0.69** |
| closed-book chat (2) | 0.59 | 0.61 | 0.54 | **0.61** |
| **all 21** | 0.71 | 0.71 | 0.64 | **0.73** |

AUROC for "wrong". Token log-probability, the usual cheap signal, scores 0.49 on average: it is useless for judging
another model's text.

**Where it works:**
- Step-by-step math: 0.81-0.89, with the wrong *step* localised at 0.82-0.88 (unit AUROC, zero labels).
- FELM reasoning, math and writing: 0.79-0.85.
- Knowledge-grounded QA and dialogue: 0.75-0.86.

**Where it does not:**
- **Closed-book world knowledge.** FELM world knowledge 0.53, TruthfulQA 0.59, HaluEval general 0.64. A 1.7B reader does not know the facts it is asked to judge.
- **FinanceBench** (0.60) and **DROP** (0.51): numbers in long tables and passages.

### What makes zero-label transfer work

| zero labels, mean over 21 held-out domains | AUROC |
|---|---|
| best single layer, mean state (layer 15 of 28) | 0.679 |
| three middle layers, mean state | 0.685 → 0.698 with per-domain standardisation |
| + judgement state | 0.692 → **0.712** with per-domain standardisation |
| + worst unit | **0.731** |
| leave a whole *format* out instead (a new kind of output) | 0.620 → 0.645 with standardisation |

- **Middle layers transfer best.** Layer 15-20 of 28 beat the last layer (0.66) and early layers (0.58).
- **Per-domain standardisation from unlabelled replies helps** (+0.01 to +0.03). It costs nothing but a few hundred of the domain's unlabelled replies.
- **Linear is enough.** A one-hidden-layer MLP on the same features transfers worse (0.688 against 0.713). The
  strength of the L2 penalty barely matters (0.706-0.713 over a 100× range). This matches the literature: the signal
  is close to one linear direction (docs/research/internal_state_probes.md).
- **No invariance trick beats per-domain standardisation on frozen features.** Same features (PCA 128), leave-one-domain-out:

  | method | AUROC |
  |---|---|
  | per-domain standardisation (the shipped choice) | **0.693** |
  | group DRO (weight the worst training domain up) | 0.692 |
  | CORAL (whiten each domain's covariance) | 0.641 |
  | project out the 5 / 10 / 19 directions that separate domains | 0.683 / 0.685 / 0.676 |

  A more invariant probe needs a different representation, not a different fit. That is the case for editing the
  reader (docs/omni_experiments/).
- **Diversity is the main lever.** Training on 1 / 2 / 4 / 8 / 12 / 20 other domains gives 0.56 / 0.56 / 0.61 / 0.67 / 0.69 / **0.71** on the held-out domain, still rising at 20.
  - Every new domain a customer brings, once labelled, raises zero-label accuracy for the next one.

### Adapting with labels (`adapt`): prior-pulled refit

| labelled replies from the new domain (balanced) | universal head as is | fresh probe on the k labels | **refit pulled toward the universal head** |
|---|---|---|---|
| 10 | 0.713 | 0.655 | **0.720** |
| 25 | 0.712 | 0.704 | **0.754** |
| 50 | 0.714 | 0.728 | **0.770** |
| 100 | 0.710 | 0.754 | **0.783** |

- **At 100 labels the refit beats a head trained only on the domain's own data:** 0.783 against 0.773 five-fold in-domain, at roughly 200-300 labels per domain.
- **The prior is worth about +0.03-0.05 over a fresh probe at every budget.**
- **Refitting only the five expert weights gains less** (0.74 at 100 labels): the domain-specific signal sits in the hidden-state direction, not in how the experts are mixed.

### The guarantee: false alarms held from a few dozen clean labels

**Floor.** A false-alarm rate of at most 10% with 95% confidence needs at least **29 clean** labelled replies,
whatever the method: (1 − 0.1)^29 < 0.05 (docs/research/weight_space_adaptation.md). Below that the threshold
falls back to "above every clean reply seen" and makes no promise; the profile's card says so
(`guarantee_possible: false`).

**Realistic onboarding.** Traffic with 20% wrong replies; k replies drawn *uniformly* and labelled; the threshold is
set on their clean ones; the rest of the domain is scored.

| uniform labels (≈ clean) | false alarms, target ≤ 10% | draws over 12% | wrong replies caught |
|---|---|---|---|
| 50 (≈ 40) | 2.5% | 1.9% | 14% |
| 100 (≈ 80) | 5.2% | 5.7% | 28% |

Reasoning domains, 100 labels: 5.3% false alarms, **45% caught**. Balanced labels with adaptation (cross-fitted):
3.3-3.6% false alarms at 50-100 labels. With only 10 balanced labels (5 clean, under the floor) it averaged 13%: the
floor is real.

**The honest reading.**
- **The promise holds.** At ≥ 40 clean labels, false alarms stayed under the target in ≥ 94% of draws, on every format.
- **Its price is recall.** At a held 10%, the verifier catches about a quarter to a half of wrong replies on reasoning, less on RAG and chat.
- **It is a cheap filter on every reply, not a replacement for review.** Recall rises with labels and with the adapted head (0.78 AUROC at 100 labels).

### Reader size

The same corpus, heads and protocol, with only the reader changed (`results/omni/final_q06.json`, `final_q17.json`):

| reader | zero labels, all | reasoning | long-form | RAG | closed-book chat | 50 labels | CPU s / reply |
|---|---|---|---|---|---|---|---|
| Qwen3-0.6B | 0.677 | 0.81 | 0.71 | 0.64 | 0.52 | 0.716 | 0.83 |
| **Qwen3-1.7B** | **0.727** | **0.85** | **0.75** | **0.69** | **0.61** | **0.742** | 0.98 |

- **0.6B is not worth it.** It saves 15% of the time and loses 0.05 AUROC.
- **The reader's own judge collapses at 0.6B** (0.53 against 0.64).
- **Closed-book chat falls to chance** (0.52).

**Qwen3-4B, the accuracy tier.** 4B features were read for 1,800 of the replies (a random 35%, about 85 per domain;
the run is slower on this CPU, 5-8 s a reply). Both readers are compared on exactly those replies, under the same
protocol (`results/omni/final_q4b_subset1800.json`, `final_q17_subset1800.json`):

| same 1,800 replies, held-out domains | zero labels, all | reasoning | long-form | RAG | closed-book chat | 50 labels |
|---|---|---|---|---|---|---|
| Qwen3-1.7B | 0.700 | 0.84 | 0.73 | 0.65 | 0.60 | 0.69 |
| **Qwen3-4B** | **0.779** | **0.88** | **0.76** | **0.77** | **0.69** | **0.80** |

- **The bigger reader gains most where the small one lacked knowledge:** RAG +0.12, closed-book chat +0.10.
- **Its own judge becomes useful on chat:** 0.73 against 0.52.
- **The 4B head ships too:** `heads/omni_qwen3_4b.npz`, trained on these 1,800 replies.
  - For 1.7B, going from 1,800 to all 5,107 replies added 0.03 zero-label. A 4B head on the full corpus should reach about 0.80; this is an estimate, not a measurement.
- **Cost:** about 2.4× the 1.7B compute, ≈ $0.035 per 1,000 checks on a GPU by the same estimate.
- **Two tiers:**
  - 1.7B for cheap screening of every reply;
  - 4B where accuracy matters more than the last cent.

### A real onboarding, through the CLI (`results/omni/demo_covid/`)

The pipeline a customer would run, on a domain the head never saw:
- **Head:** trained on the other 20 domains, leaving out CovidQA (biomedical papers, long passages).
- **Data:** fresh HaluBench CovidQA replies, none of them in the corpus.
  - `onboard` read 200 unlabelled replies, then 100 labelled ones (51 clean).
  - `evaluate` scored 300 further replies (50% wrong).

| profile | AUROC | false alarms (target ≤ 10%, 95%) | wrong caught | precision |
|---|---|---|---|---|
| universal head | 0.720 | n/a | n/a | n/a |
| profiled on 200 unlabelled replies | 0.720 | n/a | n/a | n/a |
| **+ 100 labels (adapted, calibrated)** | **0.752** | **3.3%** | 6.7% | 67% |

- **The guarantee held:** 3.3% false alarms against the 10% target.
- **On this domain it held at very low recall.** At 0.75 AUROC a 10% false-alarm threshold set from 51 clean replies sits high.
- **This is the trade-off the product has to state.** On hard domains, use the 4B tier, or accept more false alarms for more recall.
- **Speed:** 7.3 s a reply on 2 CPU threads with CovidQA's long passages (up to 2,048 tokens), against about 1 s on the corpus's median reply.

## Speed and cost

One pass of Qwen3-1.7B over documents + question + reply + the judge suffix (`vt/omni.py`, logits computed only for
the reply). Measured on 4 CPU cores (bf16):

| measure | value |
|---|---|
| per reply on this corpus (median ≈ 350 tokens, up to 2,048) | **0.98 s** (1.55 s before trimming the logits) |
| live API, short RAG check | 0.7-1.1 s |
| GPU (estimate, same arithmetic as docs/product.md: 2 × 1.7e9 × ~700 tokens ≈ 2.4e12 FLOPs; L4 at ~35% utilisation) | ≈ 0.06 s, ≈ **$0.015 per 1,000 checks** at $0.80/hour |
| storage | the head is 9 MB; the reader 3.4 GB |

That is about the price of the cheapest commercial grounding APIs ($0.03 per 1,000; docs/research/commercial_landscape.md),
and 30-600× below an LLM judge per claim.

## Limits

- **Closed-book facts need retrieval.** A 1.7B reader judging a frontier model's long-tail facts is near chance (0.53-0.64). The route is the document-grounded path: retrieve evidence, then check the reply against it.
  - Wikipedia lookups are wired in (`vt/retrieval.py`) but rate-limited from this cloud address (HTTP 429), so not measured here.
  - The 4B reader run tests how far a bigger reader closes the gap.
- **Not state of the art on plain RAG grounding.** Fine-tuned detectors reach ~79 RAGTruth F1 and decision models ~79 on LLM-AggreFact.
  - Here: 0.69 zero-label AUROC on RAG domains, 0.73-0.78 adapted; RAGTruth with 800 in-domain labels was 0.86 (docs/ragtruth_sota.md).
  - The case for this verifier is any format at one cheap pass, step localisation, and the per-domain guarantee, not peak RAG accuracy.
- **The exact arithmetic check is narrow.** It fires on 3% of wrong replies (≈6% in math domains), at about 66% precision on ProcessBench / FELM math.
  - Remaining false fires are counterfactual prose ("we cannot make 5 + 3 = 9") and notation it cannot parse.
  - It adds +0.001 AUROC overall. It is a pointer to a specific wrong step, not a verdict.
- **Evidence shift does not help at this reader size.** Each grounded reply read with and without its documents
  (`experiments/omni_evidence.py`, 800 replies across the 10 RAG domains). Six shift signals scored 0.43-0.79
  leave-one-domain-out, mean ≈ 0.57, and their sign flips between domains. This signal reached 0.82 with the 4B reader
  in docs/grounding_trace.md, but with 1.7B it is not worth a second pass per check, so it is left out.
- **Written checkers fit invariants, not per-question computation.** A hand-written "every number in the reply is
  in the passage, or a sum, difference, ratio or scale change of two of its numbers" checker was tried on the two
  number-heavy domains:
  - FinanceBench: AUROC 0.48, firing on 28% of wrong and 31% of correct answers. Its answers are multi-step metrics
    (margins, retention ratios, EBITDA − capex) whose wrong versions are plausible near-misses built from real filing
    numbers.
  - DROP: 0.54.

  Checking those needs the question's formula executed on the right rows (a program per question, an escalation
  tier). Checkers written once per domain fit fixed invariants: citation formats, IDs, units, dates, the arithmetic
  already checked here. That is the setting of "weight-space programming" results such as a hand-written
  fake-citation network.
- **Sizes.** 125-400 replies per domain. Per-domain AUROCs carry ±0.05-0.07 confidence intervals; the 21-domain means are much tighter.
