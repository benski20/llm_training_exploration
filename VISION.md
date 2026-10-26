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
