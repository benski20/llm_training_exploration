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
