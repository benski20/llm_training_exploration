# Verification from first principles: the grounding trace

## What a model can know about its own claim

A language model has two sources for anything it writes: its context and its weights. So the only thing that
can be read from its internals is whether a claim was *grounded* when it was written: drawn from the context,
or recalled from something it actually knows. A belief that is wrong in the weights, held with confidence,
looks exactly like knowledge from the inside. Only evidence from outside can catch it.

The way out is to bring outside evidence *into the context*. Then every kind of error is the same kind:

| error | with the evidence in context it is |
|---|---|
| confabulation (the model did not know) | a claim not grounded in anything |
| misconception (the weights are wrong) | a claim the evidence contradicts |
| unfaithful summary or answer from documents | a claim the context does not ground, or contradicts |
| wrong arithmetic | a claim its exact re-computation, put in the context, contradicts |

Grounding in context is a mechanism, not a topic: the attention paths and copying machinery that carry a fact
from a document into the reply are the same in a poem, code, a sum or a trivia answer. That is why it should
transfer where topic-trained probes did not (the 32B risk head scored 0.90 in-distribution and 0.35-0.80 on
sources it had not seen, `results/kaggle/risk-32b2/risk/results.json`): a probe trained on hallucination labels
learns whatever separates those labels in that dataset; a mechanism is the same everywhere.

## Why clean neurons matter

Goodfire's "Neural Networks Are Already Clean" (Cammarata, Oct 2026) reports that in a preliminary audit of
Qwen3-32B (the model this repo serves) about 90% of neuron poles have one main job. If that holds, the
mechanism can be *read* rather than *learned*: the neurons whose job is "this entity is familiar", "copy the
next token from earlier in the context", or "I am guessing" can be picked by what they do, from a contrast that
has nothing to do with any dataset's labels, and their reading should mean the same thing in every domain. Two
cautions from the replies to the essay: the labels come from top activations, which can overstate cleanliness
when information is carried by weak firing (Nanda), and they have not been shown to predict behavior.
Both are testable here: a neuron-level detector either transfers across domains or it does not.

## The method

One teacher-forced pass over [evidence; conversation; reply] (no generation, no judge, no question asked of the
model), and for every content token of the reply, where its prediction came from:

1. **the evidence**: how much the evidence raises the token's probability (log p with the evidence minus
   without it), and which evidence positions its attention paths draw on;
2. **recall**: whether the model's "known fact" machinery was engaged at that point (clean neurons on the
   subject, selected by a real-vs-invented-entity contrast);
3. **neither**: the token came from the model's prior: fine in a poem, a red flag in a stated fact.

Verdicts per statement: grounded in the evidence, contradicted by it (the evidence makes the written token much
less likely), recalled, or invented. Only "contradicted" is an alarm by itself; "invented" is the honest
description of a poem line and the warning sign in a factual answer, and the trace shows which tokens it covers.
Evidence: the conversation, the user's documents, exact re-computations of stated arithmetic and parses of code,
and a search for the reply's statements that nothing else grounds.

## How it is being tested (CPU, Qwen3-4B, no spend)

- the evidence path: `experiments/evidence_shift.py` on RAGTruth's human-labeled hallucinations;
- the recall path, and whether clean neurons generalize: neurons picked by a real-vs-invented-entity contrast,
  scored on wrong vs right closed-book answers in domains they were not picked on.

## Results so far

**The evidence path** (`results/evidence_shift_qwen3_4b.json`, 200 RAGTruth test answers, half with
hallucinations, Qwen3-4B, no training): AUROC for "hallucinated"

| level | evidence shift | log-probability with the evidence | reference |
|---|---|---|---|
| token (27,763) | 0.62 | 0.58 | provenance (document share): 0.76 |
| sentence (1,470) | 0.69 (mean shift), 0.68 on contradictions only | 0.60 | the 32B's trained head on a source it had not seen: 0.58 |
| response (200) | 0.69 | 0.55 | a detector trained on RAGTruth itself: 0.86 |

A real signal with no training, better than the trained head's transfer and than the model's plain confidence,
but well short of provenance and of a detector trained on the same data. It is one ingredient, not the method.

**The recall path and clean neurons** (`experiments/recall_neurons.py`, Qwen3-4B, 200 TriviaQA and 200 NQ
questions answered closed-book by the same model, 39% and 19% right): AUROC for "wrong"

| signal | labels used | TriviaQA | NQ |
|---|---|---|---|
| 10 neuron poles picked by the real-vs-invented-entity contrast (layers 19-23) | none about correctness | 0.72 | 0.69 |
| a dense probe on all neurons, fit to the same contrast | none about correctness | 0.68 | 0.65 |
| a dense probe fit to the other dataset's correctness labels | the other dataset's | 0.66 | 0.70 |
| the model's own probability of its answer | none | 0.78 | 0.77 |

The clean-neurons prediction partly holds: ten poles found without one correctness label transfer as well as a
probe trained on real labels does (and better than a dense probe on the same contrast), and the contrast itself
is separated perfectly by single poles. But they are not the strongest signal: the model's plain confidence
in its own answer beats every internal reading here. On RAGTruth, the evidence shift combined with confidence
does not beat the shift alone (0.67-0.68 against 0.69).

Together, confidence and the ten poles: TriviaQA 0.78, NQ 0.75 (confidence alone 0.78, 0.77). On this model
the familiarity neurons carry the same information as the model's confidence, not more.

**Where the evidence prefers another word** (`experiments/evidence_shift.py`, the disagreement signal: the
evidence makes the model prefer a different token than the one written, and raised that token itself):
RAGTruth sentences 0.64 overall, **0.72 on contradictions** (the best of these signals there), 0.53 on
unsupported additions. No training-free signal tested detects unsupported additions (0.53-0.56): they leave no
mark in the model's probabilities when the evidence is simply silent.

**Where this leaves it.** Untrained internal readings plateau near 0.7 on a 4B model (0.69 for faithfulness to
evidence, 0.78 for closed-book answers from the model's own confidence); detectors trained on labels reach
0.86-0.90 on the data they learned from and 0.35-0.80 on new sources. They learn the dataset, not the
mechanism. Next: train on the mechanism itself, with labels made by construction in any domain (a sentence
taken from a document is grounded, the same sentence with an entity or number swapped is contradicted, a
sentence from an unrelated document is unsupported), and test synthetic to natural on RAGTruth's human labels.

**Trained on the mechanism, tested on nature** (`experiments/grounding_probe.py`, `results/grounding_probe_qwen3_4b_v1.json`):
a probe on the 4B's MLP neurons, trained only on constructed examples from Dolly's Wikipedia-style paragraphs
(a sentence from the paragraph, the same with a number or name swapped, a sentence from an unrelated one),
scored on RAGTruth's human-labeled hallucinations in news, passages and reviews it never saw:

| RAGTruth | dense probe | 10 neuron poles | best training-free | 32B head trained on labels, unseen source |
|---|---|---|---|---|
| sentences | **0.71** | 0.68 | 0.69 | 0.58 |
| responses | **0.74** | 0.72 | 0.69 | |
| unsupported additions | **0.73** | 0.73 | 0.56 | |
| contradictions | 0.68 | 0.60 | 0.72 (disagreement) | |

The first signal here that catches unsupported additions, and it transfers better than the head trained on
human labels. Sparse poles do not beat the dense probe on this model. The synthetic task was too easy (0.99
held out): its unsupported sentences came from unrelated paragraphs, which a topic change gives away.

**With on-topic unsupported examples, and combined** (`results/grounding_probe_qwen3_4b.json`: unsupported
examples are sentences taken out of the paragraph shown; the evidence shift and disagreement computed on the
same RAGTruth sentences; nothing from RAGTruth used in training):

| RAGTruth sentences | all | contradictions | unsupported additions | responses |
|---|---|---|---|---|
| dense probe on constructions | 0.73 | 0.65 | **0.78** | 0.72 |
| **one neuron pole** picked on the constructions | 0.72 | 0.69 | 0.74 | 0.70 |
| evidence shift (untrained) | 0.68 | 0.64 | 0.72 | 0.66 |
| disagreement (untrained) | 0.61 | 0.72 | 0.53 | 0.63 |
| **probe + shift + disagreement** (z-scores summed, untuned) | **0.76** | **0.77** | 0.75 | 0.68 |

A single neuron pole does nearly all the dense probe does on natural data from other domains: grounding is
mostly one readable unit on this model, which is what the clean-neurons view predicts. The probe is strongest on
what the context does not say, the disagreement on what it contradicts, and together they reach 0.76 per
sentence with no RAGTruth data; a detector trained on RAGTruth itself reaches 0.86. Responses (scored by their
worst sentence) do worse combined (0.68): the worst-sentence rule amplifies noise.

Responses scored by their total ungroundedness (the sum of each sentence's positive z-score) instead of their
worst sentence: 0.76 for the dense probe, the single pole and every combination (0.68-0.72 by the worst
sentence). The rule was picked among seven on this test set, so read it as slightly optimistic.

**Constructed replies worded like answers** (`--phrasing`, `results/grounding_probe_qwen3_4b_phrasing.json`: a
lead-in such as "According to the text," and parentheses dropped, alike for every kind, so a verbatim copy
cannot give grounding away):

| RAGTruth | before | worded like answers |
|---|---|---|
| dense probe, sentences | 0.73 | **0.75** |
| dense probe, responses (total ungroundedness) | 0.76 | **0.82** |
| dense probe, contradictions | 0.65 | **0.72** |
| probe + evidence shift, sentences | 0.74 | **0.77** |
| probe + evidence shift, responses | 0.77 | **0.82** |

0.82 per response with nothing from RAGTruth, against 0.86 for a detector trained on RAGTruth itself: the probe
had partly learned "copied verbatim means grounded"; without that shortcut it learns grounding.

95% bootstrap intervals over responses (`--phrasing`, 600 constructions): the probe 0.82 [0.76, 0.88] per
response and 0.75 [0.70, 0.79] per sentence; with the evidence shift 0.82 [0.76, 0.88] and 0.77 [0.72, 0.81];
the untrained shift alone 0.70 [0.62, 0.78] and 0.68 [0.64, 0.73]. The trained-on-RAGTruth detector's 0.86 lies
inside the probe's interval. More of the same constructions does not help: 1,500 instead of 600 gave 0.77-0.79
per response while the synthetic held-out score rose to 0.99, the probe learning the constructions' artefacts.
More varied constructions, not more of them, is the lever.

## End to end (`vt/grounding.py`, `experiments/grounding_demo.py`, `results/grounding_demo_qwen3_4b.json`)

The serving check on 13 whole replies, Qwen3-4B reading each once with its evidence (the pasted article; or two
Wikipedia lookups, by the question and by the question with the reply's own words; exact arithmetic; code
parsed), probe, evidence shift and disagreement summed on their natural-sentence scales:

| pair (same question) | right reply | wrong reply |
|---|---|---|
| summary of a pasted article | 1.13 | **2.80** |
| poem: the sea / "Paris, capital of Italy" | 0.54 | **2.15** |
| 17 x 23 = 391 / 401 | 0.00 | **0.39** |
| Pride and Prejudice: Austen / Brontë | 0.38 | **0.58** |
| Eiffel Tower: 1889 / 1925 | 0.82 | **0.84** (thin) |
| Python function: parses / does not | 0.00 | **0.03** (thin) |

All six pairs in the right order; across all 13 replies, wrong ones against the rest at AUROC 0.71 (reassurance
0.50, the sea poem 0.54). Three things the earlier versions of this demo showed: the probe alone missed
single-name and single-date contradictions (the disagreement signal is what catches them); a lookup by the
question alone found Eiffel Tower passages without its date or designers (the reply's own words steer it to
the pages that settle the claim); a code-parse failure needs to be in the evidence to count. Two margins are
thin, and this is a 4B reading; the served model is the 32B the clean-neurons audit is about, and its probe
has to be trained on it (a GPU run of `experiments/grounding_probe.py --export`).

## In the server

`--grounding heads/grounding_probe_qwen3_4b.npz --grounding-model Qwen/Qwen3-4B` (the deploy's default): every
final reply is read by the 4B with its evidence, and each sentence reads consistent (nothing in the evidence
against it, nothing missing: a poem line can be consistent), not grounded, or contradicted; the reply reads
likely wrong (a sentence contradicted), not grounded, or nothing flagged. A local run through HTTP (0.5B
writing, 4B reading, live Wikipedia): a summary of a pasted article (sentences at z -0.1 to -2.7), "Jane
Austen" (-1.2), a haiku and "17 times 23 is 391" all read nothing flagged. The reader is a second model, read
from its neurons and asked nothing; once a probe is trained on the 32B (`experiments/grounding_probe.py --model
unsloth/Qwen3-32B-bnb-4bit --export ...`, a GPU run), the served model reads its own replies.

## 120 everyday messages through the server (`results/mixed_eval_grounding_local.json`)

Local, CPU: the 0.5B writing (so many answers are wrong), the 4B reading, live Wikipedia (no rerank), the 120
mixed messages one at a time; no errors, median 19 s, 95th percentile 42 s.

| kind (n) | the 0.5B right | wrong answers flagged | right answers flagged | wrong vs right, total |
|---|---|---|---|---|
| TriviaQA (20) | 35% | **1 of 13** | 1 of 7 | 0.66 |
| Natural Questions (20) | 5% | 8 of 19 | 0 of 1 | 0.87 (one right answer) |
| GSM8K (20) | 20% | 5 of 16 | 1 of 4 | 0.52 |
| how-to (15) | | 10 of 15 flagged (mostly not grounded) | | |
| code (15) | | 4 of 15 flagged | | |
| creative (15) | | 7 of 15 not grounded | | |
| advice and chat (15) | | 5 of 15 flagged | | |

Closed-book errors mostly pass: the probe was strong on RAGTruth (0.82) with the right documents in context,
but a quick Wikipedia lookup for a trivia question often misses the deciding fact, and a wrong answer that
nothing in the evidence addresses is not marked. Reasoning errors in word problems are barely separated. For
how-to, creative and advice replies "not grounded" is often the plain truth: Wikipedia does not hold a pep talk.
Next: better evidence (the cross-encoder rerank, off in this run), and the reader's own belief in each sentence
without the evidence (already computed for the shift; on its own it marked the 4B's closed-book errors at 0.78).
