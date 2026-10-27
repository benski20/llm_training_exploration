# Vision: generative models you can audit exactly, and a general verifier

## Thesis

Today's generative models are trained first and checked afterwards, by
sampling. Sampling cannot rule out rare failures. In this repository a model
scored 99.6% on a test set and was still wrong on 647 of 65,536 inputs. An
unverified guard leaked an invalid output in 20,000 samples.

vt takes the other route: **verifiability is an architectural constraint,
chosen before training.** Models are built so that their full behaviour, over
every input or every output, can be decided exactly. Training then optimises
inside that constraint. The bet is that this produces a different kind of
generative system: one whose properties come with proofs and whose failure
probabilities are computed rather than estimated.

## The core technical idea

A model is exactly auditable when its computation, at every position, factors
through a **small summary state**. With this repo's architecture:

* causal mixing is *windowed-uniform* (a few local taps plus one shared
  weight for the whole past),
* so a position's output depends only on (the last w tokens, the counts of
  older tokens),
* which takes polynomially many values in the length L.

A spec given as a deterministic automaton, with finite control plus
counters, has a polynomial reachable state space too. The **product** of the
two is a polynomial graph that covers all V^L inputs or outputs. Dynamic
programming over it decides, exactly, in integer arithmetic:

| question | object | result in this repo |
|---|---|---|
| worst-case margin: is the classifier correct on all inputs? | classifiers | Dyck proved at every even length 2–256 (L=256 is 2²⁵⁶ inputs); aⁿbⁿcⁿ (not context-free) proved |
| support: can the generator emit an invalid string? can it emit every valid one? | generators | Dyck generators proved sound + complete + live; the exact P(violation) is a rational |
| is it stuck? | generators | liveness: no reachable prefix has all-zero next-token scores |

Two design choices make exactness possible:

1. **No softmax at deployment.** Next-token probabilities are
   `relu(z) / Σ relu(z)`. That gives exact rationals and **exact zeros**. A
   softmax generator puts positive mass on every string, so it can *never*
   be sound.
2. **Training on the whole state space.** The product graph is enumerable,
   so the support loss covers every live (state, token) pair rather than a
   sample. Certification then usually succeeds the first time.

## The general verifier: a kernel, not a monolith

"General" means one interface, `verify(artifact, spec, claim) → certificate
| counterexample`, backed by a growing set of complete decision procedures,
each sound for its class:

| model class | spec class | procedure | status |
|---|---|---|---|
| windowed-uniform classifiers | DFA / counter automata | exact DP over the product state | ✅ exact |
| count-state generators | DFA / counter automata | exact support and mass DP | ✅ exact |
| general piecewise-linear networks | built-in specs | MILP with symbolic bounds and case splits | ✅ floating-point |
| GPT MLP blocks | local robustness | exact per-block MILP | ✅ per block |
| any generator | any spec | proved guard (filter with a certified classifier) | ✅ |
| stack-state models | bounded-depth pushdown (JSON, Dyck-k) | DP over bounded stacks | ⬜ next |
| any of the above | Boolean combinations of specs | product automata | ⬜ next |

The kernel's trust base is small and inspectable: integer arithmetic plus a
spec's step function. Certificates bind each verdict to a hash of the exact
weights. Deployment re-proves them; it never trusts JSON.

## Roadmap, with go/no-go tests

1. **Bounded-stack state (the JSON milestone).** Count-state models can't
   match two bracket types (Dyck-2). That needs a stack. Add a model block
   with an explicit bounded stack of symbols (depth D) and verify over stack
   contents: k^D states, fine for D ≤ 8–10.
   *Go if:* a learned validator **and** generator are proved for JSON-like
   bounded-nesting formats at realistic lengths (L ≥ 256).
2. **Spec algebra.** AND/OR/NOT of specs via product automata, so
   certificates compose: proved format ∧ proved content constraint.
3. **Richer exact state.** Several counter heads, multiple blocks whose
   cross-block summaries stay finite (quantised states), SSM-style
   recurrences with integer state.
   *Go if:* expressivity grows (mod-k counting, multi-type nesting) while
   verification stays polynomial.
4. **Scale the exactly-verifiable family as language models.** Train
   count/stack-state generators on real text and measure the perplexity gap
   to transformers at equal compute.
   *Go if:* the gap is small enough that guarantees are worth it. If it isn't,
   use the hybrid: a large unverified generator wrapped in proved guards or
   constrained by a proved generator.
5. **Learned verifiers verified against formal ones.** Fast neural checkers
   (reward or judge models) for domains with a formal reference, carrying
   global certificates. That is the path from "verified formats" to
   "verified judgements".

## The semantic layer: interpretability by design

Truthfulness, reasoning correctness, harmlessness and tone have no formal
definition to prove against. The approach here is to **build the model so its
internals carry the verification signals**, rather than reverse-engineering
them afterwards with an atlas.

* **Named epistemic channels** (`vt/epistemic.py`): reserved residual-stream
  slices such as `grounded` or `checked`. Each is trained against a signal
  computable at training time: the fact is present in context or memory, the
  step passes a solver, the span was copied from a source.
* **Causal role by intervention training**: values are written into the
  channel during training, and the model must act on them (abstain when
  `grounded` is forced low). The readout then reflects what the model uses,
  not a correlate it can route around. At inference the channel is a monitor,
  a steering knob, and the input to a conformally calibrated release gate.
* **Provenance by architecture (next)**: route factual content through an
  explicit memory, so a factual token produced without a memory read is
  structurally visible.
* **Formal checks where they exist**: arithmetic, logic, code and formats go
  through the exact kernel (`vt/pcr`, `vt/exact.py`).

First result on grounded answering (`experiments/grounding_by_design.py`, a
0.8M-parameter byte-level GPT, CPU; `results/grounding_by_design_v1_bytenames.json`):

| in-distribution | post-hoc probe (baseline) | built-in channel |
|---|---|---|
| accuracy | 16% | 37% |
| hallucinated when the fact was absent | 76% | 30% |
| detects "fact absent" (AUROC) | 0.54 | 0.76 |
| forcing "absent" causes abstention | — | 98% |

Failures: both models collapse when contexts are longer than in training (the
channel drops to chance), and the by-design model over-abstains (53% false
abstentions). The conformal gate's bound held in-distribution but not under
shift (13% of hallucinations passed for the baseline, against a 5% target),
as the exchangeability assumption predicts. Base capability is low at this
scale, so these numbers are suggestive, not conclusive.

### Replication and provenance by architecture

With single-byte names (`results/grounding_by_design_v2_single.json`), the
pattern held, and the channel became causal in both directions:

| in-distribution | post-hoc probe | built-in channel |
|---|---|---|
| accuracy | 29% | 43% |
| hallucinated | 47% | 29% |
| detects "fact absent" / flags own errors (AUROC) | 0.53 / 0.15 | 0.66 / 0.71 |
| conformal gate (α=0.05): hallucinations passed, correct answers kept | 0.5%, 0% | 1.8%, 44% |
| force "absent" → abstain; force "present" → answer | — | 100%, 79% |

**Provenance by architecture** (`experiments/provenance_by_design.py`): a
pointer head, with fact-bearing tokens (digits) removed from the vocabulary
path, so a year can only be copied from the context. Every copy's source is
checked exactly against the asked entity.

| | in-distribution | under shift |
|---|---|---|
| raw accuracy / invented / wrong-person copy | 42% / 32% / 15% | 0% / 100% / 100% |
| **false answers released** | **0** | **0** |
| correct answers released | 42% (all) | 0% |

The two mechanisms complement each other. The channel is a cheap,
general-purpose statistical monitor and control knob, whose guarantee depends
on calibration matching deployment. Provenance gives a
distribution-independent guarantee for the classes of content routed through
it: under shift the model fails completely, yet nothing false is released.
(The first provenance run released 2 false answers because of a gap in the
check, now fixed and covered by a test.) The direction this points to is
**route as much content as possible through checkable paths (copy from
sources, solvers, exact checkers), and monitor the rest with designed-in
channels.**

### What did not generalise

* **Channel v3 (ALiBi, `results/grounding_by_design_v3_alibi.json`).** ALiBi
  removed the collapse under longer contexts, but neither model learned the
  lookup: almost no correct years were produced in either condition.
* **Typed generation (`vt/typed_lm.py`).** On WikiText-2 the vocabulary path
  cannot emit names or numbers, so they can only be whole-word copies, checked
  exactly. The guarantee holds by construction, but it rests on a hand-written
  rule: a capitalised word or a digit counts as a fact. That misses lowercase
  facts and relations, so the line was stopped (at step 1750 of 4000).

## Glass-box models: built to be read (`vt/glassbox.py`)

The goal here is to build a transformer whose internals can be read accurately,
instead of training a standard model and reverse-engineering it:

* an **evidence stream**: convex residual mixing, so a distribution over
  sources (input positions and memory reads) is an exact account of how each
  position's content was mixed;
* a **starved core**: fixed embeddings and no MLPs, so a product-key
  **memory** is the only fact store, with every read logged;
* a **self-knowledge variable k**: trained on the model's own correctness, and
  the only input to the answer/abstain decision (or, for open text, to the
  sharpness of the next-token distribution);
* **counterfactual memory patching** during training.

Synthetic knowledge world (`experiments/glassbox_world.py`): 1000 entities x 8
relations, against a GPT with matched parameter counts and post-hoc linear
probes on every layer (best layer chosen on in-distribution data).

| | GPT | glass-box v1 | v4 (zero-init memory) |
|---|---|---|---|
| unasked memory facts answered, in-dist / long context | 25% / 23% | 78% / 89% | **82% / 94%** |
| context facts answered, in-dist / long context | 43% / 27% | 47% / 45% | **59% / 58%** |
| hallucination on never-seen facts | 66% | 73% | 87% |
| "my answer is wrong" AUROC in-dist: built-in k / post-hoc probe | — / 0.68 | 0.69 / 0.69 | 0.66 / 0.62 |
| same, long contexts | — / 0.63 | 0.72 / 0.70 | 0.69 / 0.60 |

What this shows:

* **The explicit memory is a clear capability win.** It answers questions it
  was never asked about stated facts 3-4x better than the GPT, and holds up on
  longer contexts.
* **The built-in readout k matches or beats post-hoc probes, more so under
  shift.** At 0.66-0.72 AUROC, though, this is not yet accurate
  interpretability.
* **The explicit interface made diagnosis cheap.** Patching memory reads at the
  answer position never changed the answer. One sweep showed the causal read
  is at the *entity* token (59% of answers follow a patched read there), so the
  model taught itself an entity lookup.
* **Placing the only memory read after the last attention block makes the read
  causal by construction** (v5). Patching it makes the answer follow the
  patched fact 64% of the time (v1: 0%), and k reaches 0.75 AUROC (probe 0.73).
* **A mechanistic account of this model's hallucinations** (v5). An untrained
  readout of slot usage turned out *inverted* (AUROC 0.34): wrong answers read
  the most heavily used slots. A query for a fact that was never stored has
  nowhere to land, so the top-k read falls onto hub slots holding other facts,
  and the model reports one of them with confidence.
* **Negative results:**
  * patch training at all question positions collapsed learning (v2);
  * a late memory read for (entity, relation) lost capability (v3);
  * the label-free evidence and memory-match readouts do not detect errors;
  * a learned "nothing stored" null slot in the read stays unused (v6, AUROC
    0.48), because nothing in training rewards reading it;
  * no version learned to abstain on facts it never saw (hallucination 73-89%).
    Absence is taught only on specific (entity, relation) pairs, which the
    model memorises one by one.

| version | k / probe AUROC, in-dist | k / probe, long context | hallucination |
|---|---|---|---|
| v1 | 0.69 / 0.69 | 0.72 / 0.70 | 73% |
| v4 zero-init memory | 0.66 / 0.62 | 0.69 / 0.60 | 87% |
| v5 memory after last block | 0.75 / 0.73 | 0.72 / 0.73 | 79% |
| v6 + null slot | 0.71 / 0.68 | 0.70 / 0.65 | 77% |

v7, an ensemble memory with solo-head training, collapsed (8% accuracy).

**As a general chat model** (`experiments/glassbox_chat.py`; 6.1M vs 6.0M
parameters; Alpaca, WikiText-2 and the fact world in natural language;
`results/glassbox_chat/run.json`):

| held-out Alpaca / WikiText tokens | GPT | glass-box |
|---|---|---|
| bits per token | **5.09 / 5.44** | 5.90 / 5.97 |
| top-1 accuracy | **34% / 32%** | 26% / 28% |
| "top prediction wrong" AUROC: the model's own softmax confidence | **0.82 / 0.84** | 0.82 / 0.83 |
| post-hoc probe (best layer) | 0.71 / 0.69 | **0.75 / 0.79** |
| built-in k | — | 0.74 / 0.77 |
| label-free memory / evidence readouts | — | 0.46-0.52 |

Verdict at this scale:
* The glass-box model is **more linearly readable**: post-hoc probes gain 4-10
  AUROC points.
* That gain does not translate into better hallucination detection. In both
  models the plain softmax confidence beats every internal readout.
* The design costs about 0.5-0.8 bits per token of capability.
* Both models are too weak to answer the chat fact questions (at most 5%
  correct), so those numbers mostly measure abstention.

Chat v2 (chatbot-style fact questions, made-up names, multi-turn;
`results/glassbox_chat_v2/`), matched sizes:

| | glass-box | GPT |
|---|---|---|
| bits per token, Alpaca | 5.60 | **4.99** |
| fact given in the message, answered (made-up names) | **28% (32%)** | 18% (10%) |
| hallucination on unstated facts | **63%** | 70% |
| flags its own wrong chat answers: softmax / post-hoc probe | 0.65 / 0.72 (k_min 0.72) | **0.75 / 0.85** |
| post-hoc probe on WikiText tokens | **0.79** | 0.63 |

The glass-box model uses given facts better, hallucinates slightly less and is
more probe-able. But the GPT's own probe detects its errors better, and the
GPT writes better. Chat v3, a hard memory gate with an L1 penalty of 0.05,
closed every gate by step 750: the model gave up memory entirely, so its
certificates were trivial. It was stopped.

**Glassbox-2 CPU pilot** (`vt/glassbox2.py`, `results/glassbox2_chat/`): exact
sparse features, exact attribution, a logged hard-gated memory. Same chat data,
8.0M parameters. Go/no-go tests fixed before the results:

* G1, language within 15% of the GPT: **pass**, 5.47 vs 4.99 bits/token (+9.6%).
* G2, verification value: **fail**. k flags wrong chat answers at 0.81 against
  the GPT probe's 0.85, and 0% of chat replies are certified memory-free.
* G3, soundness: pass, but on chat it is vacuous (no certified replies).
* G4, inspectability: **pass**. On the trained model, attribution matches the
  logits to within 3e-6, and the features are exactly sparse.

It also answered given facts worse (0% vs 28%) and over-refused 44% of the time.
Diagnosis: memory gates open on 2.4% of tokens, but softmax attention gives
every earlier token a nonzero weight, so one open gate taints everything after
it. Next: exactly sparse attention, so each token's dependency set is exact
and small.

The thesis "design for interpretability, get more accurate hallucination
detection" is **not supported** by these experiments. The memory-as-knowledge
store and the mechanistic diagnoses are the parts worth carrying forward.

**Glassbox-2.1** (memory after the last layer, 32 active features;
`results/glassbox2_1_chat/`, compared by `experiments/chat_compare.py`):

| | Glassbox-2.1 | GPT |
|---|---|---|
| bits per token, Alpaca | 5.24 (+5%) | **4.99** |
| flags its own wrong chat answers (best signal) | **0.913** (built-in k_min) | 0.849 (post-hoc probe) |
| same, held-out relations | **0.913** | 0.906 (softmax) |
| hallucination: unstated facts / made-up names | 80% / 23% | **70% / 5%** |
| fact given in the message, answered | 2% | **18%** |
| over-refusal | 18% | 17% |

For the first time a signal built into the model detects its errors better
than a probe fitted to the GPT after training. The model does not act on that
knowledge, though: it hallucinates more, not less. Its memory-free
certificates (100% of tokens) are trivial, because every memory gate was
closed and frozen. This exposed a design bug: `clamp(v, 0, 1)` has zero
gradient once saturated, so the gates freeze at 0 or 1, depending on the
initialization. The fix is a straight-through gradient: the forward value is
unchanged, so exact zeros and the certificates stay exact, and the gradient is
1 for every gate. Glassbox-3 is the attempt to make the model act on what it
knows.

**Glassbox-3: epistemic uncertainty that shapes what the model says**
(`vt/glassbox3.py`). The memory and provenance checks depend on what the model
happened to store, and that is brittle for a general chatbot. Glassbox-3 adds
a mechanism that needs no store and no labels about which facts are known:

* **Epinet** (Osband et al.). A small network reads the final features, with
  the gradient stopped, plus a random index z. It adds a fixed random prior
  network that is linear in z. Trained to predict the next token, its learnt
  part cancels the prior where the data constrains the prediction. Where the
  data does not, the prior survives and predictions differ across z. The
  epistemic uncertainty u is the mutual information between the next token
  and z, computed over a fixed set of indices, so u is deterministic. The base
  model, and its exact attribution, are untouched.
* **Uncertainty in the output, by construction.** u raises the logit of a
  dedicated `<unsure>` token by an explicit term `softplus(c)·u`. It also lowers
  the commitment k that sets the sharpness of every next-token distribution.
  A sharp, committed token therefore requires low epistemic uncertainty. That
  is a property of the architecture, not of a detector fitted afterwards.
* **Self-consistency replay.** During training the model samples several
  answers to factual prompts. Where its own answers disagree, it is trained to
  answer `<unsure>` plus a hedge. The supervision comes from the model's
  disagreement, not from labels.
* **Feature novelty.** Hashed counts of which exactly-sparse features fire
  together in training. Rare co-activation at inference is a label-free
  "unfamiliar territory" signal, and it names the features involved.

Pre-registered tests for the CPU pilot (same chat-v2 data, against the GPT's
70% / 5% hallucination and 16.7% over-refusal):

* hallucination below half the GPT's, on both unstated facts and made-up names;
* over-refusal within about 10 points of the GPT's;
* u (or `<unsure>`) flags wrong answers better than the GPT's post-hoc probe
  (0.85);
* language within about 10% of the GPT (at most 5.48 bits per token).

**Result: no-go** (`results/glassbox3_chat/`). The model says `<unsure>` to
nearly every chat question: over-refusal is 100%, so its 0% hallucination
means nothing. Language is 5.81 bits per token (+16.5%), and u flags wrong
answers at chance (AUROC about 0.5). Diagnosis:

* The `<unsure>` term is not the cause: `c_unsure` never moved, and `<unsure>`
  takes 0.2% of the probability on ordinary text. The base model itself puts
  about 97% on `<unsure>` as the first reply token for every kind of question,
  including ones whose answer is in the message. At this size no glass-box
  model answers the questions (context answers about 0%), so it learns the base
  rate (60% of training answers abstain). Under argmax decoding, one shared
  `<unsure>` token then beats answers spread over many tokens. A competing
  abstention token is the wrong interface. The abstain decision should
  threshold a signal, as in selective prediction.
* The epinet gives no usable signal at this scale.
* The straight-through gate had a second bug, which may explain part of the
  language regression. At a closed gate it sent gradient back into the
  residual stream, a phantom training signal on every token. It is fixed:
  saturated gates now update only their own weights.

**Glassbox-2.2** (2.1 with the fixed straight-through gate, the Kaggle
student's configuration; `results/glassbox2_2_chat/`). Language is unchanged
at 5.24 bits per token, so the gate fix costs nothing. The memory gates still
close everywhere: at this scale the L1 cost outweighs what stored knowledge
buys.

The 300-question chat evaluations were too noisy to compare architectures.
Re-evaluated on 1,500 new questions (300 per kind; 95% intervals;
`experiments/chat_eval_large.py`):

| | GPT | Glassbox-2.1 | Glassbox-2.2 |
|---|---|---|---|
| hallucination, unstated facts | 75% [70-80] | 76% [71-80] | 67% [62-72] |
| hallucination, made-up names | **4% [2-6]** | 35% [30-41] | 10% [7-14] |
| over-refusal | 13% | 13% | 16% |
| fact given in the message, answered | **16%** | 4% | 3% |

At CPU scale the glass-box models have no hallucination advantage. Glassbox-2.2
is slightly lower on unstated facts, but the intervals nearly touch. Both
glass-box models are clearly worse on made-up names and at using facts given
in the message. The 2.1/2.2 gap on made-up names (35% vs 10%) is real: the
gate's gradient changes behaviour, not just language.

**Glassbox-2.3: additive residual stream** (`results/glassbox2_3_chat/`). Both
convex pilots used facts given in the message far worse than the GPT (2-4% vs
16-18%). Convex mixing lets a layer only replace part of the residual stream,
which hampers copying. Exact attribution only needs each layer to be linear
given its decisions, and standard residual sums are linear too (tested, with
certificate soundness). Changing only this:

| | GPT | Glassbox-2.2 (convex) | Glassbox-2.3 (additive) |
|---|---|---|---|
| bits per token, Alpaca | **4.985** | 5.242 (+5.2%) | 5.073 (+1.8%) |
| fact given in the message, answered | 18% | 2% | **28%** |
| over-refusal | 17% | 16% | 16% |
| best built-in error detector (AUROC) | 0.849 (probe) | **0.920** | 0.805 |

On 1,500 new questions (95% intervals), Glassbox-2.3 against the GPT:
hallucination 73% [68-78] vs 75% on unstated facts and 10% [7-14] vs 4% on
made-up names; over-refusal 13% vs 13%; given facts answered 24% [20-29] vs
16% [12-21].

The exactly attributable model is now within 2% of the GPT on language. It
uses facts given in the message better than the GPT does. Error detection
fell below the GPT's probe, so part of the earlier detection gain came from
the convex models' inability to copy. The Kaggle student trainer defaults to
additive mixing from here on. The run already in flight uses convex mixing.

**Exact attribution as a hallucination check: negative**
(`experiments/attribution_grounding.py`). At each reply's riskiest token,
the exact decomposition gives the share of the logit that comes from the
conversation. A grounded answer should draw on a source there, and a
hallucination should not. On Glassbox-2.3 (1,053 answered replies):

* conversation share 0.63 and single-best-source share 0.65 AUROC, against
  0.70 for the reply's NLL;
* on the subset that tests grounding directly (answers to questions whose fact
  is in the message, against hallucinations), 0.55-0.57 against 0.66;
* added to NLL, the attribution scores do not help on held-out replies
  (0.700 vs 0.696).

A likely reason: these questions often include sentences about other people.
A hallucination can copy their values, so "drawn from the conversation" is
not "drawn from the right part of it". An exact attribution is exact about
the computation, not about whether the source is the right one. Telling those
apart needs the relation between the source and the question, which this test
does not use. The 0.84 from a first 20 replies was noise.

**Selective answering** (`experiments/selective_answering.py`). A reply is
withheld when the model's own signal falls below a threshold. The threshold
is calibrated on separate questions with over-refusal capped at 27%, and the
GPT gets the same rule with each of its own signals. The rule barely helps
either model:

| | hallucination (unstated facts) | made-up names | over-refusal (test) |
|---|---|---|---|
| GPT, no rule / post-hoc probe | 70% / 60% | 5% / 5% | 17% / 28% |
| Glassbox-2.1, no rule / built-in k_min | 80% / 70% | 23% / 13% | 18% / 36% |

At 8M parameters the glass-box models answer about 1% of the answerable
questions (the GPT 11%), so a signal has little real knowledge to separate
from guessing. A high AUROC does not change that. This question needs models
that know things. That is the Kaggle run: Glassbox-2 (with the fixed gate)
against a GPT student, on 28,672 replies from Qwen2.5-1.5B-Instruct plus
6,000 made-up-name items, 4 GPU-hours each. It is evaluated on TriviaQA
(accuracy, hallucination, error-bounded release of answers by each signal)
and on made-up names.

## Product wedge

Structured-output reliability for AI applications: formats (JSON-like,
protocols, DSLs, identifiers) where one invalid output breaks a pipeline.
vt sells:

* **certified validators and guards**, which reject invalid output with no
  exceptions, plus a re-provable certificate;
* **certified generators** that cannot emit an invalid string;
* **exact audits**: the computed probability that a generator violates a
  policy.

Each step on the roadmap widens the set of formats and policies covered.

## Known limits, stated plainly

* Count-state models can't represent stacks with mixed types (no Dyck-2 yet)
  or learn periodic counting (mod 3 collapses to a constant at width 32).
* Exact verification needs the architecture constraint; arbitrary networks
  fall back to floating-point MILP or per-block checks.
* No result here says large language models can be proved correct end to
  end. The claim is narrower and testable: exactly auditable generative
  systems can be built, trained and deployed, and the frontier is how much
  expressivity they can carry.
