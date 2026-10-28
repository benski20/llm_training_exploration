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

**Kaggle GPU runs.** Run 1 (convex Glassbox-2 vs GPT, 4 h each) failed both
ways:

* The GPT student died while loading data, most likely out of RAM: two
  processes were tokenizing WikiText-103 into Python lists at once.
* The 146M-parameter Glassbox-2 student stayed at about 7.3 nats (unigram
  level) for 14,000 steps.

Small CPU reproductions learned normally in every variant, so the stall needs
the full configuration. Fixes: tokenize once, in chunks, into int32 arrays; use
additive residuals; train the k temperature as calibration of detached raw
logits, with the LM loss on raw logits as in a GPT. An 18-minute full-size
debug run (`results/kaggle/glassbox-distill-debug/`) then trained both students
normally:

| step | Glassbox-2.3 (146M) | GPT (100M) |
|---|---|---|
| 200 / 600 / 1,000 | 5.55 / 4.09 / 3.52 | 5.29 / 4.03 / 3.45 |
| seconds per step | 0.88 | 0.38 |
| bits per token after 18 minutes | 5.36 | 4.49 |

Per step the glass-box student keeps pace. It is 2.3 times slower per step,
which is the real cost at this size.

Run 2 (additive mixing, coupled temperature, 4 h; `results/kaggle/glassbox-distill-add/`)
trained throughout: training loss went from 5.4 to 1.1-2.6 nats. So run 1's
stall came from convex mixing at full depth, not from the temperature
coupling. But held-out teacher replies scored 9.9 bits per token at 44% top-1
agreement. The 2.4M-token teacher corpus had been seen about 30 times, and
the model was confidently wrong off it: overfitting. The trainer now keeps
the weights with the best loss on 200 held-out replies, disjoint from the
test replies. Attribution is exact to 2e-6 and the
features are exactly sparse.

**Are the built-in features interpretable?** (`experiments/feature_interpretability.py`;
100k tokens.) The comparison is between Glassbox-2.3's exactly sparse features
(32 of 1,024 active), the matched GPT's MLP neurons, and a top-k sparse
autoencoder fitted to the GPT's residual stream afterwards (1,024 latents,
k = 32), the standard post-hoc tool. Token purity is the share of a unit's
20 strongest activations that fall on its most common token, an automatic and
crude proxy for monosemanticity:

| | median purity (layer 5 / 2) | units with purity >= 0.8 | variance missed |
|---|---|---|---|
| GPT MLP neurons | 0.30 / 0.35 | 12% / 10% | n/a |
| Glassbox-2.3 built-in features | 0.45 / 0.55 | 20% / 27% | **0** |
| post-hoc SAE on the GPT | **0.70 / 0.70** | **46% / 46%** | 7% / 10% |

The built-in features are clearly more specific than raw neurons. They are
less so than SAE latents, which do not explain 7-10% of what the GPT computes.
The SAE's objective, sparse reconstruction, pushes latents towards specific
directions; the built-in features are shaped only by the language-model loss.
Purity also only rewards token detectors, so it says little about abstract
features.

**Glassbox-2.4: the SAE objective trained into the model**
(`results/glassbox2_4_chat/`). Each feature layer gets a readout that is used
only in training and is not part of the forward pass. Through it, the active
features must also reconstruct the layer's normalized input. The computation
and the exact attribution are unchanged (tested); only the features are shaped
toward a sparse code of what they read.

| | median purity (layer 5 / 2) | purity >= 0.8 | variance missed | bits per token |
|---|---|---|---|---|
| post-hoc SAE on the GPT | 0.70 / 0.70 | 46% / 46% | 7% / 10% | 4.985 (the GPT) |
| Glassbox-2.3 | 0.45 / 0.55 | 20% / 27% | 0 | 5.073 (+1.8%) |
| **Glassbox-2.4** | **0.65 / 0.70** | **40% / 46%** | **0** | 5.206 (+4.4%) |

On 1,500 new questions (95% intervals), Glassbox-2.4 against the GPT:
hallucination on unstated facts 64% [58-69] vs 75% [70-80]; on made-up names
6% [4-10] vs 4% [2-6]; over-refusal 13% vs 13%; facts given in the message
answered 37% [31-42] vs 16% [12-21]. On held-out relations: 54% vs 65% and
17% vs 25%. A second seed of each model did not reproduce the hallucination gap:

| | GPT seed 0 / 1 | Glassbox-2.4 seed 0 / 1 |
|---|---|---|
| hallucination, unstated facts | 75% / 74% | 64% / 71% |
| hallucination, made-up names | 4% / 5% | 6% / 9% |
| over-refusal | 13% / 10% | 13% / 12% |
| fact given in the message, answered | 16% / 22% | **37% / 30%** |
| bits per token | 4.985 / 4.98 | 5.206 / 5.214 |

The hallucination advantage on unstated facts is at most about 7 points on
average. That is the same size as the seed-to-seed spread, so it is not
robust. Better use of facts given in the message does replicate. Glassbox-2.4
is slightly worse on made-up names. The language cost is a steady +4.5%.

The built-in features now match post-hoc SAE latents on this measure. They are
also the model's exact computation, where the SAE misses 7-10% of the GPT's.
The price is 2.6 more points of language quality.

**Final Kaggle comparison at 146M** (`results/kaggle/glassbox-distill-final2/`;
Glassbox-2.3 vs GPT, 2.5 h each, best checkpoint on held-out replies). Both
peak at step 3,000 and then overfit the 2.4M-token teacher data, so this is
data-limited, not compute-limited: the glass-box student's 2.3x slower steps
cost nothing here.

| | GPT (100M) | Glassbox-2.3 (146M) |
|---|---|---|
| held-out loss / test bits per token | **3.396 / 4.57** | 3.512 / 4.80 (+4.9%) |
| top-1 agreement with the teacher | 43.1% | 43.2% |
| TriviaQA right / abstained / hallucinated | 1.8% / 68% / **30%** | 1.6% / 48% / 51% |
| made-up names, hallucinated | 92% | 90% |
| fact given in the message, answered | **68%** | 55% |
| reply-level error AUROC (best signal) | 0.58 | 0.58 |

Attribution is exact to 2.6e-6, the features are exactly sparse, and there
were no certificate soundness violations. Neither model knows trivia at this
size and data, and neither has a signal that can release answers under an
error bound. On this data the GPT abstains more, hallucinates less and uses
given facts better: the CPU context-use advantage of 2.3 did not carry over.
Its built-in features are also less token-specific than at CPU scale. Median
purity is 0.30 / 0.25 (middle / last layer, 4,096 features) against 0.45-0.55
on CPU, with 7-9% of units at purity >= 0.8
(`experiments/package_features.py`, 60k local tokens).

**Glassbox-2.4 at 146M** (`glassbox-distill-g24`: the same data, budget and
evaluation, plus the in-model SAE objective). The best checkpoint on held-out
replies was step 3,000 of 8,968 (validation 3.546 nats).

| | GPT | Glassbox-2.3 | Glassbox-2.4 |
|---|---|---|---|
| test bits/token | **4.57** | 4.80 | 4.80 |
| TriviaQA right / abstain / hallucinate | 1.8 / 68 / **30%** | 1.6 / 48 / 51% | 1.2 / 55 / 44% |
| made-up names, hallucinate | 92% | 90% | **84%** |
| given facts used correctly | **68%** | 55% | 64% |
| median feature purity (layer 6 / 11) | n/a | 0.30 / 0.25 | **0.35 / 0.30** |
| features with purity >= 0.8 | n/a | 9% / 7% | **19% / 15%** |

The SAE objective still pays at scale. It doubles the share of
single-meaning features, and narrows the gap to the GPT on hallucination and
context use, at no cost in loss against 2.3. The gain is smaller than on CPU
(where median purity was 0.65-0.70), and the GPT still hallucinates less on
TriviaQA. Attribution stays exact (5e-6) and features exactly sparse. None of
the three students knows enough trivia (at most 1.8% right) to test
correctness signals; that is the job of the converted Qwen below.

**Can the internals tell when an answer is correct?**
(`experiments/correctness_from_internals.py`; 1,500 chat questions per model.)
Among replies that answer, each detector was fitted on one half and scored on
the other (AUROC for right versus wrong):

| | GPT | Glassbox-2.4 | Glassbox-2.3 |
|---|---|---|---|
| answered / right | 1,030 / 82 | 990 / 175 | 1,049 / 132 |
| softmax of the answer token | **0.659** | 0.736 | 0.687 |
| post-hoc hidden-state probe | 0.646 | 0.652 | 0.602 |
| built-in k | n/a | 0.543 | 0.592 |
| exact attribution statistics | n/a | **0.742** | 0.681 |
| probe on which sparse features fired | n/a | 0.530 | 0.663 |
| all combined | n/a | 0.625 | 0.683 |

Within each model, no internal signal clearly beats the model's own output
probability. For Glassbox-2.4, attribution (0.742) and softmax (0.736) tie.
The feature probe and the combination were overfit: about 6,000-12,000 inputs
from about 500 fitting replies, with a fixed weak penalty. Probes now choose
their penalty on an inner split. Comparisons across models are confounded,
because each model answers a different set of questions with a different
right rate. The question is being re-asked on a converted Qwen2.5-0.5B, which
knows answers (`kaggle/qwen_glassbox/`).

**Converting a pretrained Qwen2.5-0.5B-Instruct into a glass box**
(`vt/qwen_glassbox.py`, `kaggle/qwen_glassbox/run.py`, Kaggle T4, 66 minutes).
All 24 MLPs were replaced by exactly sparse feature layers (8,192 features, 64
active). Each was distilled from its MLP for 2,500 steps, then the whole model
was fine-tuned for 2,000 steps on KL to the original, with the in-model SAE
objective. Attention was untouched. Results on 1,000 TriviaQA questions
(short-answer prompt, detectors fitted on 500 and scored on 500):

| | original Qwen | converted |
|---|---|---|
| TriviaQA right | **21.6%** | 6.6% |
| top-1 agreement with the original | n/a | 64% |
| bits/token (local text) | **3.26** | 4.27 |
| softmax AUROC (right vs wrong) | 0.68 | 0.57 |
| post-hoc hidden-state probe | **0.759** | 0.56 |
| built-in k / sparse-feature probe | n/a | 0.64 / 0.62 |
| exact attribution error | n/a | 2.3e-5 |
| features with purity >= 0.8 (layer 12 / 23) | n/a | 20% / 22% |

The conversion keeps exactness, and its features are about as readable as
Glassbox-2.4's at 146M. But it loses two thirds of the model's knowledge: the
converted model mostly invents answers ("the next Prime Minister after Arthur
Balfour was George Washington"). Replacing every MLP at once with 1% of its
features active compounds the errors, and 4,500 steps on one T4 cannot recover
them. On the converted model, no signal separates right from wrong well,
because there is little right to find. This is a no-go for a full sparse
conversion at this budget.

The bar it failed to reach is itself the useful finding. On the original
model, a post-hoc linear probe on one hidden layer (19 of 24) beats the
model's own probability, 0.759 against 0.68. Reading the internals does
predict correctness better than the output does. A dense SwiGLU MLP is also
linear in its input once its SiLU gate values are fixed, the same kind of
decision as an attention pattern or RMS scale. So exact attribution needs no
conversion at all, and can be tested directly on the original model.

**Exact attribution on the unmodified Qwen** (`experiments/qwen_correctness.py`,
`qwen_correctness_stats.py`; CPU, the same 1,000 questions and halves). With the
SiLU gates, attention patterns and RMS scales fixed, the first answer token's
logit splits exactly (error 1.4e-5) into a part per question token, one for the
chat template, and one per layer's biases. It also splits into what each
attention and MLP layer wrote. The accuracy (21.6%) and probe (0.759) reproduce
the Kaggle run exactly.

| signal (AUROC, right vs wrong) | fitted? | AUROC | vs softmax (95% CI) |
|---|---|---|---|
| softmax of the first answer token, all 1,000 | no | 0.690 | n/a |
| **how much the question's own tokens push the answer** (one exact number) | no | **0.717** | +0.026 (-0.015, +0.066) |
| share of the score from the question / from biases | no | 0.685 / 0.343 | n/a |
| softmax + NLL + that one number, test half | 3 weights | 0.725 | **+0.043 (+0.020, +0.066)** |
| per-layer contributions (49 numbers), test half | yes | 0.659 | n/a |
| post-hoc hidden-state probe (layer 19), test half | 896 weights | **0.759** | n/a |
| probe + question attribution, test half | yes | 0.764 | +0.005 over the probe (+0.001, +0.009) |

One readable, exact number helps: how strongly the words of the question drive
the answer. Answers the model knows are pushed by the question's subject (" Pacers"
and " sport" give "Basketball"). Guesses lean on the template and on function
words (" the" and " The" give "Jim Henson" for the Chipmunks). On its own the
number is not clearly better than softmax, but added to it, it gives a
significant gain. A trained probe on the hidden state still predicts best, and
attribution adds almost nothing to it. What attribution offers over the probe
is the reason ("the answer is not tied to the question's subject"), not more
accuracy.

The signals are read at the first answer token, which is often a function word
("The Chipmunks were created by ..."). Reading them instead at the answer's
entity, its first new name or number (`experiments/qwen_correctness_entity.py`,
40% of answers), does not lift the ceiling. The best output baseline becomes the
answer's least likely token (0.707), and the grounding number ties it (0.707).
A probe at the entity scores 0.744, and everything combined 0.760. Every
detector built from 500 fitting answers and one sample per question lands
between 0.70 and 0.76.

**Next: a correctness estimator at scale** (`vt/correctness.py`,
`kaggle/correctness/run.py`). The ceiling above is set by the data and the
signal set, not by where we read. The next run trains the entity probe on
6,000 TriviaQA training answers rather than 500, with a curve over training-set
size. It adds sampling consistency (6 sampled answers against the greedy one),
and a probe trained to predict consistency so test time needs one pass. A
stacker combines probe, output probabilities, consistency and exact grounding
into one calibrated probability. It runs on Qwen2.5-1.5B, 3B and 0.5B, and is
tested on TriviaQA validation and, without refitting, on NQ-open (a different
dataset). The library returns, for a question, the reply, P(correct) and its
evidence.

**Result** (`results/kaggle/qwen-correctness/correctness/results.json`; Kaggle
T4, 3.2 h, float32, since float16 Qwen2.5 overflows into NaN replies). AUROC for
right versus wrong, and the error rate among the 25% most confident answers.
Everything is fitted on TriviaQA train and nothing on NQ-open:

| | Qwen2.5-1.5B TriviaQA | 1.5B NQ-open | Qwen2.5-0.5B TriviaQA | 0.5B NQ-open |
|---|---|---|---|---|
| accuracy (answer everything) | 40.1% | 20.2% | 21.6% | 12.2% |
| first-token softmax | 0.719 / 36% | 0.637 / 70% | 0.685 / 62% | 0.603 / 82% |
| best output probability (entity / min / mean) | 0.735 / 35% | 0.703 / 60% | 0.707 / 60% | 0.707 / 77% |
| sampling consistency (6 samples) | 0.792 / 27% | 0.652 / 68% | 0.687 / 64% | 0.563 / 84% |
| exact grounding (one number, unfitted) | 0.712 / 36% | 0.654 / 64% | 0.707 / 58% | 0.689 / 77% |
| entity probe, 500 / 2,000 / 6,000 answers | 0.781 / 0.824 / 0.830 | 0.698 / 0.733 / 0.740 | 0.749 / 0.761 / 0.782 | 0.686 / 0.709 / 0.754 |
| stack: probe + output | 0.846 / 21% | 0.772 / 58% | 0.793 / 51% | 0.768 / 71% |
| stack: + grounding | 0.846 / 21% | 0.781 / 55% | 0.801 / 48% | **0.786** / 70% |
| stack: + consistency | 0.858 / 18% | 0.776 / 57% | 0.800 / 49% | 0.763 / 72% |
| **stack: all** | **0.859 / 18%** | **0.783 / 55%** | **0.807 / 46%** | 0.781 / 71% |
| calibration error (ECE) of the full stack | 0.035 | 0.192 | 0.028 | 0.081 |

What this establishes:

- **The ceiling was data, not the method.** The same entity probe goes from
  0.781 to 0.830 (1.5B) and from 0.749 to 0.782 (0.5B) with 6,000 fitting
  answers instead of 500. Within each model the full stack beats the model's
  own probability by 0.10-0.14 AUROC. For the 1.5B model, it cuts the error among
  the most confident quarter of answers from 35% to 18%: answering only those, it
  is right 82% of the time against 40% when answering everything.
- **It survives a dataset shift in ranking, not in calibration.** On NQ-open,
  never seen in fitting, the stack still ranks answers clearly better than any
  output signal (0.78 against 0.70). Its probabilities, though, are too
  confident there (ECE 0.19 for 1.5B), because NQ's base rate (20%) is half
  TriviaQA's. A new domain needs a small labelled sample to recalibrate the
  threshold, not new training.
- **Exact grounding pays where the shift is.** Unfitted, it matches the output
  probability. In the stack it adds most out of domain (0.5B on NQ: 0.768 to
  0.786; 1.5B on NQ: 0.772 to 0.781) and nothing in domain for 1.5B. Attribution
  stays exact at both sizes (error 6e-5).
- **Sampling consistency is strong only when the model knows enough.** For 1.5B
  on TriviaQA it is the best single unfitted signal (0.792); for 0.5B and on NQ it
  is weak. A probe trained to imitate it is worse than the plain probe.
  Concatenating all ten layers overfits (0.75 against 0.83 for the best single
  layer, 23 of 28).
- **Bigger models are easier to read.** The 1.5B model's internals separate
  right from wrong better than the 0.5B model's (0.859 against 0.807).

The trained heads (`*_heads.pt`, 50-85 KB) are served by
`vt.chat_serve.CorrectnessResponder`. Each reply carries P(correct) and its
evidence, and the answer is withheld below a threshold:
`vt chat-export --hf-dir <model dir> --heads <model>_heads.pt --variant all`,
then `vt chat` / `vt chat-serve`.

**What a reply looks like** (`experiments/correctness_demo.py`,
`results/correctness_demo.json`). Qwen2.5-1.5B-Instruct, packaged with its
`all` head through the production path, answer threshold 0.5, about 10 s per
reply on CPU:

| question | reply | P(correct) | why (evidence) |
|---|---|---|---|
| What is the capital of Australia? | Canberra. | 0.84 | 6/6 samples agree; probe +1.28 |
| Who wrote the novel Pride and Prejudice? | Jane Austen wrote ... | 0.95 | 6/6 agree; token p 0.95; probe +2.14 |
| Who was the man behind The Chipmunks? | **withheld** (guess: Bob Mosher; truth: Ross Bagdasarian) | 0.24 | 2/6 agree; token p 0.40; probe -0.57 |
| Which Lloyd Webber musical premiered in the US on 10 Dec 1993? | **withheld** (guess: The Phantom of the Opera; truth: Sunset Boulevard) | 0.16 | token p 0.64, yet probe -2.70 |
| Who was the first president of the Republic of Zanthoria? (made up) | **withheld** (guess: "did not exist") | 0.13 | 1/6 agree; probe -1.90 |
| Who wrote "He Ain't Heavy, He's My Brother"? (NQ) | **withheld** (guess names the wrong writers) | 0.12 | token p 0.09; probe -2.08 |

All four wrong or unanswerable questions are withheld, and both right answers
are released. The Lloyd Webber case shows the internals catching an error
the output probability would pass: the model is 64% sure of its first token,
but the hidden-state probe reads it as a guess. One rough edge, now fixed: the made-up
country got a correct refusal ("did not exist") that was still withheld,
because the head was trained to judge answers, not refusals. The responder
now passes the model's own refusals through as abstentions, with no
P(correct).

**Selective answering** (`experiments/selective_answering.py`). A reply is
withheld when the model's own signal falls below a threshold. The threshold
is calibrated on separate questions with over-refusal capped at 27%, and the
GPT gets the same rule with each of its own signals. The rule barely helps
either model:

| | hallucination (unstated facts) | made-up names | over-refusal (test) |
|---|---|---|---|
| GPT, no rule / post-hoc probe | 70% / 60% | 5% / 5% | 17% / 28% |
| Glassbox-2.1, no rule / built-in k_min | 80% / 70% | 23% / 13% | 18% / 36% |

With Glassbox-2.3, which knows more, on 1,000 questions per split and with
each model's best signal:

| | hallucination (unstated) | made-up names | over-refusal | answerable, right | all right |
|---|---|---|---|---|---|
| GPT + post-hoc probe | **53%** | 5.5% | 31% | 8.5% | 33.4% |
| Glassbox-2.3 + reply NLL | 61.5% | 5.5% | 28.5% | **13.3%** | 34.6% |

Overall correctness is a tie. The GPT's probe cuts hallucination further,
and the glass-box model answers more answerable questions. A first run on
300 questions had shown Glassbox-2.3 ahead on every column, which was noise.

At 8M parameters Glassbox-2.1 answered about 1% of the answerable
questions (the GPT 11%), so a signal has little real knowledge to separate
from guessing. A high AUROC does not change that. This question needs models
that know things. That is the Kaggle run: Glassbox-2 (with the fixed gate)
against a GPT student, on 28,672 replies from Qwen2.5-1.5B-Instruct plus
6,000 made-up-name items, 4 GPU-hours each. It is evaluated on TriviaQA
(accuracy, hallucination, error-bounded release of answers by each signal)
and on made-up names.

**Any reply, any format: claim-level correctness** (`vt/claims.py`,
`kaggle/claims/run.py`, `vt.chat_serve.ClaimResponder`). The estimator above
judges one short answer per question. The product has to judge any reply: a
paragraph, a list, a reasoning chain, the fifth turn of a conversation. The
unit is therefore the *claim*: each name, number or date a reply asserts that
the conversation did not already contain. The probe reads the hidden state at
each claim's first token, so one forward pass over the reply scores all its
claims. There is no sampling, and the serving overhead is one pass.

Training data comes in five formats with automatic labels:
- *short*: one question;
- *multi*: three questions answered on numbered lines, each line labelled;
- *multiturn*: the question asked after one to three earlier turns;
- *reasoning*: step by step, then a final answer;
- *math*: GSM8K, where the final number is checked.

Final answers are found however the model writes them: `\boxed{}`, "The final
answer is", or the closing line. Models ignore an instructed "Answer:" line,
and a strict parser had labelled no math replies at all.

The evaluation reports, per format:
- TriviaQA / GSM8K test and NQ-open with 95% bootstrap intervals;
- leave-one-format-out (trained without a format, tested on it);
- a training-size curve;
- the risk-coverage area, calibration, and reply-level risk for multi-claim
  replies.

**Float16 without NaNs.** In float16 the earlier Kaggle run produced garbage
(0% accuracy, then a CUDA assert). The cause is not Qwen2.5's activations, which
peak below 7,000. It is its raw attention logits: q.k reaches about 2.6e5 in
the 1.5B model, beyond float16's 65,504. Hugging Face's eager attention
multiplies before scaling, so it overflows. sdpa scales first, so it does not.
Everything now runs float16 with sdpa: about 4-8x cheaper on a T4, and
the 3B model fits (it ran out of memory in float32). Exact attribution now
recomputes each attention pattern in float32 from the module's inputs. It
matches eager exactly (difference 0.0), and in float16 the parts sum to the
logit within 0.005 of 18.

**Doubt-aware generation** (`vt/guided.py`, `experiments/doubt_aware_eval.py`). The probe reads
the hidden state the model has already computed at each step, so checking a claim before it is
written costs nothing. In stop mode a doubted claim is never stated. For example, at 0.5B the
probe doubts "Jim Henson" for the Chipmunks (P = 0.16), and the reply stops instead.

At Qwen2.5-0.5B on 300 new TriviaQA questions (25% right), it gives no gain over stopping on low
token probability, the FLARE-style trigger. Hallucination at 90 / 70 / 50% answered is
65 / 47 / 31% for the probe and 65 / 46 / 32% for token probability. Two reasons:
- 23% of replies contain no capitalised or numeric claim ("Lupine relates to a type of grass"),
  so the heuristic claim detector never fires on them;
- the 0.5B probe is the weakest one; at 1.5B it beat token probability 0.83 to 0.72.

Both point to the general detector below: it scores every token, with no claim heuristic, and
uses bigger readers.

**Exact provenance on RAGTruth: first sanity check** (`experiments/ragtruth_provenance.py`, 0.5B
reader, 40 test responses, 4,845 tokens, 267 inside human-labelled hallucinations, nothing trained).
Each response token's logit was split exactly into document / instruction / response-so-far / bias
(`vt.qwen_glassbox.explain_all`, one pass). Hallucinated tokens draw 2.0% of their logit from the
document, against 6.7% for supported tokens.

| signal | AUROC |
|---|---|
| low document share, per token | **0.76** |
| the reader's own logit for the token | 0.63 |
| low mean document share, per response (only 6 hallucinated responses) | 0.88 |

This is a training-free signal from the smallest reader, and it is promising. The sample is too small
to trust the response-level number. The real test is the 7B detector run (2,700 RAGTruth test
responses, against the Lookback Lens attention ratio on the same tokens).

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
