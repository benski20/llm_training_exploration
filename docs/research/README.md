# Research synthesis: what six fields say about a cheap, fast, accurate, adaptable verifier

Six research agents each surveyed one field, 2023 to November 2026, and wrote a report in this folder.
- Each report tags its numbers by source: read in the paper, abstract only, a secondary summary, or memory.
- Several 2026 items are preprints or launch-week aggregator pages; check them before quoting externally.

| report | field |
|---|---|
| [internal_state_probes.md](internal_state_probes.md) | hallucination and correctness probes on hidden states (SAPLMA → 2026) |
| [mech_interp.md](mech_interp.md) | SAEs, transcoders, circuits, entity-knowledge features, retrieval heads |
| [weight_space_adaptation.md](weight_space_adaptation.md) | weight-space programming, hypernetworks, merging, domain adaptation, conformal methods under shift |
| [uncertainty_blackbox.md](uncertainty_blackbox.md) | black-box and gray-box UQ, long-form claim checking, benchmarks |
| [reasoning_verifiers.md](reasoning_verifiers.md) | ORMs and PRMs, generative verifiers, probes on chains of thought, exact tools |
| [commercial_landscape.md](commercial_landscape.md) | products, prices, latency, the accuracy bar, positioning |

## Where the fields agree

1. **The signal is in the hidden states and is close to linear.**
   - A mid-layer linear probe reaches 0.80-0.95 AUROC on its training distribution.
   - A 2026 study finds one mean-shift direction carries almost all of it.
   - SAE-feature probes do *not* beat raw-activation probes, in any regime: few labels, label noise or out of distribution (four independent sources).
   - SAEs help explain verdicts and prune spurious features, not score better.
   - **Decision:** linear heads on raw mid-layer states. SAEs and transcoders only for audits.
2. **A separate reader works.** A small open model reading another model's text separates hallucinations well, and
   probes transfer across models with little loss.
   - Output probability is useless for judging someone else's text: at or below chance in this repo's detector runs.
   - **Decision:** one frozen small reader judges any model's output, closed models included.
3. **Transfer to a new domain is the core weakness.**
   - Representation-based detectors fall to near chance out of distribution in independent audits.
   - Pooled RAGTruth numbers are inflated by a task-type shortcut: "data-to-text" alone scores 0.71 AUC.
   - Probes hold only under small shifts.
   - What transfers best:
     - middle layers;
     - mean-pooling over the reply;
     - **diverse** training data rather than more of one kind;
     - token- or claim-level labels on long answers (Obeso et al.: long-form → short QA and math, not the reverse).
   - **Decision:** train one universal head on many domains and formats. Evaluate it leave-one-domain-out, label-balanced inside each domain, so a task-type shortcut cannot inflate it.
4. **Ranking transfers; calibration does not.**
   - Calibration error triples on new datasets while AUROC holds.
   - No paper shows a label-free threshold that holds a false-alarm rate across domains, and this repo found the same four times.
   - **Decision:**
     - re-centre features on each domain's *unlabelled* replies (free);
     - adapt the head with a few labels, pulled toward the universal weights (TSV-style: 32-64 labels approach full supervision);
     - set the threshold by a Clopper-Pearson bound on the domain's clean replies, cross-fitted when the same labels adapted the head.
5. **The guarantee's label cost is set by *clean* replies.**
   - At most 10% false alarms with 95% confidence needs at least 29 clean replies; below that no method, hypernetwork or otherwise, can promise it.
   - More clean replies buy recall.
   - Label a *uniform* sample of real traffic, not a balanced one.
6. **Weight-space programming is not the lever.**
   - "Sparks of Weight-Space Programming" (MIT CSAIL, Nov 2026, preprint) edits weights to fix generation behaviour; it says nothing about calibration or adapting a classifier.
   - Hypernetworks that generate adapters (Text-to-LoRA, Drag-and-Drop) need hundreds of training tasks and give no guarantee.
   - What carries over is the cheap version: writing or orienting a readout from unlabelled structure, then a prior-pulled refit. This repo already showed a written readout ties a fitted one (docs/written_readout.md).
7. **Reasoning needs step-level reading plus exact tools.**
   - Probes on the generator's own states reach 0.80-0.95 on math.
   - Published single-pass small PRMs are weak (Skywork-PRM-1.5B: 36.4 ProcessBench F1). Generative judges are strong but cost thousands of generated tokens.
   - Math transfers to science partly, to law or logic poorly.
   - **Decision:**
     - per-step units read in the same single pass;
     - the reply's worst step as a signal;
     - exact checks (arithmetic, code) where they apply;
     - a generative judge only as an escalation tier.
8. **Closed-book chat is the hardest case for a small reader.**
   - The reader only knows what it knows: a 4B reader judging a frontier model's long-tail facts is weak (an extrapolated AUROC of 0.55-0.70).
   - Resampling the generator (semantic entropy, SelfCheckGPT) is strong but costs 5-20× generation.
   - **Decision:**
     - closed-book replies get a reader score plus the reply's own log-probs when an API exposes them;
     - for high stakes, retrieval feeds the grounding check, turning the weak case into the strong one.
9. **A judge prompt is a free knob.**
   - PRISM: wrapping the text in "is this true?" before reading improves probes.
   - With a causal reader the question can be appended *after* the reply, so the reply's own states are unchanged. One pass gives both the reply states and a judgement state, plus the reader's P(yes).

## The market (commercial_landscape.md)

- **Plain document-grounded checks are commoditised.** "Decision models" (Sept-Oct 2026) lead LLM-AggreFact at 78.5-79.5 balanced accuracy, at about $0.03 per 1,000 checks and 0.1-0.5 s.
- **Credible bars:**

  | benchmark | bar |
  |---|---|
  | LLM-AggreFact | 77.4 (Bespoke-MiniCheck-7B) |
  | RAGTruth | ~79 reply F1 (fine-tuned) |
  | HaluBench | ~87% accuracy (Lynx-70B) |

- **No one sells "at most X% false alarms on *your* domain, with 95% confidence, from N of your labels".** The same goes for one verifier across formats (RAG, chat, long-form, reasoning), or step localisation in reasoning, from one cheap pass.
- **Position:** sell the guarantee and the any-format, on-prem, white-box check.
  - Where a decision model is stronger on plain RAG, its score can be one more input to the same calibrated head.

## What was built from this

- `vt/omni.py` and `vt/omni_cli.py` (`vt-omni`): the omni verifier.
- `experiments/omni_corpus.py` and `experiments/omni_eval.py`: the corpus and the transfer and adaptation study.
- Results are in [docs/omni_verifier.md](../omni_verifier.md).
