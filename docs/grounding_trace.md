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
