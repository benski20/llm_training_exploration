# Later ideas: what neuron-level access enables commercially

Ranked by how directly each follows from what exists (`vt/qwen_glassbox.py` exact attribution,
`vt/correctness.py` / `vt/detector.py` probes, `vt/guided.py` doubt-aware generation).

1. **Taint tracking for agents (prompt-injection defense).** Exact attribution of each tool-call
   argument to its source: user, system, retrieved document, web page, tool output. Block or confirm
   actions mostly driven by untrusted content. Policies such as "payment amounts must come from the
   user". This is information-flow control for LLMs, exact rather than attention heuristics.
   First test: prompt-injection benchmarks with the instruction inside a retrieved document; compare
   with attention-based detectors.
2. **Cascade routing (cost).** A small self-hosted model answers when the correctness probe is
   confident; otherwise retrieve or escalate to a large model or API. Measure savings at a fixed error
   rate. The probe ranks answers far better than output probability (0.86 vs 0.72 AUROC at 1.5B).
   Spin-off: a *knowledge map*, the probe run over a customer's entities to see what the model
   reliably knows and what needs retrieval or fine-tuning.
3. **Verifiable audit receipts (regulated industries, EU AI Act).** Per answer, a signed record:
   model hash, inputs, each claim with source sentence, document share and confidence. Re-running the
   pinned model reproduces the exact decomposition.
4. **Reasons for decisions.** For LLM-based scoring (credit, insurance, hiring screens), an exact split
   of the decision over input fields ("adverse action reasons"). Bias audits: how much came from
   names or gendered words.
5. **Data protection.** RAG leak control: block answer tokens attributable to documents the user may
   not see. Memorization flag: long confident spans with almost no context contribution (copyright /
   PII risk). The latter is weaker, because memory has no clean bucket and needs validation.
6. **Many monitors from one pass.** Linear probes for hallucination, sycophancy, refusal,
   toxicity/intent, topic drift, possibly deception, on the same hidden states; a real-time
   model-health API.
7. **Steering.** Shift hidden states along a probe direction to reduce hallucination or enforce a style
   without fine-tuning (cf. ITI; effects are often modest).
8. **Model regression testing.** Detect internally when a fine-tune or new version loses knowledge or
   changes behaviour, before it shows in outputs.

Constraints for all of these: they need open weights (the detector can still judge closed models'
text); explanations are exact for one pass with its decisions fixed, not counterfactual; per-input-token
attribution costs more than group attribution, so run it on demand.
