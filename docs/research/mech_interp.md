# Mechanistic interpretability as a source of verification signals

Research note, November 2026. Written for the grounding verifier (docs/product.md, docs/grounding_trace.md):
a frozen Qwen3-4B reader, a per-sentence probe on MLP activations, evidence shift, and copying-head and
source-flow features. Label-free AUROC is 0.62-0.77, and 0.86 with in-domain labels. Copying heads, source
flow and claim contrast could not hold a false-alarm rate across domains.

**Confidence tags.** **[read]** means I read the paper's own tables or text. **[abstract]** means the number
comes from an abstract or conference listing only. **[secondary]** means a summary written by someone else.
**[verified]** means I checked a Hugging Face repo through the HF API on 2026-11-07.

---

## TL;DR

1. **Causal does not mean calibrated.** Several mechanisms are causally tied to hallucination:
   - known/unknown-entity latents that gate a default "can't answer" circuit;
   - retrieval and copying heads;
   - entropy neurons;
   - a low-dimensional truth subspace.

   Their **direction** carries across domains. Their **scale** does not, and no paper shows one giving a
   calibrated score across domains. Two warnings:
   - A 2026 ICML replication finds that intervening on the entity features is a "hallucination-induction
     circuit, not a calibration knob".
   - A 2025 audit finds every representation-based detector, ReDeEP and SAE features included, near chance
     out of distribution once a task-type confound is removed.
2. **SAE-feature probes do not beat raw-activation probes**, in distribution, out of distribution, or with few
   labels. Four independent sources agree (Kantamneni et al.; the GDM team; Dubanowska et al.; Vicentino). SAEs
   are useful for:
   - choosing features by a label-free contrast;
   - auditing and pruning spurious features such as format, task type or length;
   - explaining a verdict.

   They are not a better classifier.
3. **Off-the-shelf dictionaries exist for our exact reader:**
   - `mwhanna/qwen3-4b-transcoders`: per-layer transcoders for `Qwen/Qwen3-4B`, all 36 layers, 163,840
     features, MIT, works with circuit-tracer;
   - `gzxiong/sae-qwen3-4b`: residual TopK SAEs for all 36 layers, from the RAGLens authors.

   The official Qwen-Scope (Apr 2026) covers Qwen3-1.7B/8B and Qwen3.5, not Qwen3-4B. Gemma Scope / Gemma Scope 2
   and the Llama-3.2-1B transcoders are complete. All of them can be encoded on CPU (about 0.2-0.4 GFLOP per token
   per layer). Attribution graphs cannot be built on CPU at serving scale.
4. **Long chain-of-thought.** The best evidence is for step-aggregated hidden-state probes:
   - ReProbe: a probe under 10M parameters beats PRMs out of domain;
   - Zhang et al.: AUROC above 0.7 and ECE below 0.1, transferring between GSM8K and MATH;
   - entity probes trained on long-form text transfer to MATH at 0.85-0.89 AUROC (Obeso et al.).

   Circuit-based verification (CRV) reaches 0.70 on GSM8K but transfers poorly and is too expensive.
   Retrieval/receiver heads are a good mechanistic prior for *which* steps to weight.
5. **The structural caveat for our product.** Our reader reads *other models'* text, which is off-policy.
   Mechanisms like "do I know this entity", entropy neurons and retrieval-head failure describe the *reader's*
   own processing, not the writer's. The repo has already seen this: reader confidence is at chance on other
   models' outputs. Signals that survive are those about the **text vs evidence** relation, or the reader's
   knowledge used as a proxy for "a 4B model would know this". Off-policy probes also generalize worse
   ([Kirch et al., ACL Findings 2026](https://arxiv.org/abs/2511.17408)).

---

## 1. The literature by mechanism

### 1.1 Entity knowledge: "Do I know this entity?" and the default "can't answer" circuit

- **Ferrando, Obeso, Rajamanoharan, Nanda, ICLR 2025**
  ([arXiv 2411.14257](https://arxiv.org/abs/2411.14257), code `javiferran/sae_entities`) **[read]**
  - **Setup.** Gemma Scope SAEs on Gemma-2-2B. Wikidata entities of four types (players, movies, cities, songs;
    7.5k-10.9k each). "Known" means at least two attributes recalled correctly.
  - **Latents.** Known-entity and unknown-entity latents peak around **layer 9 of 26**. They were selected by
    their worst-case separation over the four entity types, so they are general across types by construction.
    They were validated on 283 songs released after the knowledge cutoff.
  - **Causal role.** Steering with the unknown latent pushes refusal on unknown entities to **about 100%**. The
    mechanism: the latents change how much attribute-extraction heads (L18H5, L20H3) attend to the entity. The
    directions come from the base model and carry over to the chat model.
  - **Uncertainty latent** (Gemma-2B-IT, layer 13): **AUROC 0.732** for correct vs incorrect answers. The paper
    compares it with no baseline.
- **Anthropic, "On the Biology of a Large Language Model"**
  ([transformer-circuits.pub 2025](https://transformer-circuits.pub/2025/attribution-graphs/biology.html))
  **[read]**
  - **Default circuit.** In Claude 3.5 Haiku, "can't answer" features are on by default. "Known entity" and
    "known answer" features inhibit them.
  - **Misfire.** Hallucination happens when known-answer features fire on a familiar name whose specifics the
    model lacks. Example: a paper attributed to Andrej Karpathy that he did not write. Activating known-answer
    features on "Michael Batkin" or "Josh Batson" induces confabulation.
  - **Caveats from the authors.** The interventions needed were large relative to natural activations. Turning on
    unknown-name features did *not* induce refusal for Michael Jordan. The features are "only a part of the
    story". Graphs gave satisfying insight for roughly a quarter of prompts.
- **Vicentino, "Hallucination-Induction, Not Calibration", ICML 2026**
  ([listing](https://icml.cc/virtual/2026/79571)) **[abstract]**
  - Replicates the entity latent on Qwen3.6-27B. The best single SAE latent (layer 31) reaches **AUROC 0.814
    [0.73, 0.90]** on known vs unknown.
  - This is statistically indistinguishable from an **L2 logistic probe (0.887)** and a **difference-of-means
    probe (0.859)**.
  - Single-feature steering has no effect on calibration.
  - Ablating the top 200 features is significant against a random-K null, but a Claude-judge audit shows it
    *induces confabulation*: known-entity wrong answers go from 62% to 77%, correct answers from 8% to 0%.
- **A Gemma-2-2B steering replication**
  ([EA forum](https://forum.effectivealtruism.org/posts/9vkEpghkuEv5QtfKG/entity-recognition-feature-steering-in-gemma-2-2b))
  **[secondary]**: the entity circuit tracks *confidence, not correctness*.
- **In this repo.**
  - Neurons picked by a real-vs-invented-entity contrast give **0.72 on TriviaQA and 0.69 on NQ** with no
    correctness labels, beating a dense probe fit to the same contrast (docs/grounding_trace.md). This matches the
    literature at small scale.
  - The SAE version would mostly add a cleaner basis and named features, not AUROC (see §2).

**Verdict.** This is the best-documented causal hallucination mechanism. It is domain-general across entity
*types* by design. But it measures "the model believes it knows", which is the misfire case. When the reader is
not the writer, it measures the *reader's* familiarity. Use it as one feature: "reader does not know the entity
and the evidence does not mention it". It is not a calibrated score.

### 1.2 Context use vs parametric knowledge: retrieval, copying and lookback

- **Retrieval heads** (Wu et al., ICLR 2025 oral, [arXiv 2404.15574](https://arxiv.org/abs/2404.15574))
  **[abstract]**
  - Fewer than 5% of heads. They exist in every model tested, including short-context pretrained ones.
  - Pruning them causes retrieval failure and hallucination. Pruning random heads does not.
  - They matter strongly for CoT, which refers back to the question and earlier text, and less for
    knowledge-only tasks.
  - This is the causal basis for our copying-head work (docs/copy_heads.md).
- **ReDeEP** (Sun et al., ICLR 2025, [arXiv 2410.11414](https://arxiv.org/abs/2410.11414)) **[read via
  RAGLens and Dubanowska tables]**
  - Mechanism: an External Context Score from copying heads and a Parametric Knowledge Score from late "knowledge
    FFNs". Hallucination = FFNs over-writing while copying heads under-read.
  - RAGTruth, Llama-2-7B: **AUC 0.746**.
- **Dubanowska et al., "Representation-based Broad Hallucination Detectors Fail to Generalize OOD"**
  ([arXiv 2509.19372](https://arxiv.org/abs/2509.19372), Sep 2025) **[read]**
  - **Confound.** On RAGTruth, a classifier that only predicts "hallucinated if the task is data-to-text" gets
    **AUC 0.712**, against ReDeEP-token's 0.732.
  - **In distribution** (Llama-2-7B): plain **logistic regression on residuals 0.795** > SAPLMA 0.749 > ReDeEP
    0.746 > **SAE classifier 0.711**.
  - **Train RAGTruth, test SQuAD:** everything is near 0.5. ReDeEP 0.516, LR 0.514, SAE 0.485, SAPLMA 0.552.
  - **ReDeEP tuned on RAGTruth, tested on Dolly:** 0.50.
  - Our RAGTruth numbers should be checked for this task-type confound (experiment E5).
- **Lookback Lens** (Chuang et al., EMNLP 2024, [arXiv 2407.07071](https://arxiv.org/abs/2407.07071))
  **[secondary + repo README]**
  - Per-head ratio of attention on the context to attention on generated tokens, with a linear classifier.
  - Transfers **QA ↔ summarization at about 0.82-0.85 AUROC** when spans are given (sentence-level), but only
    about 0.66 with sliding windows.
  - Transfers from a 7B to a 13B model without retraining.
  - This is the closest published analogue of our source-flow "reply share minus sink share". Their
    cross-task transfer is better than our fitted-on-other-kinds result (0.44-0.52). The likely reasons:
    *per-head* ratios, not pooled shares, and on-policy reading.
- **Context vs parametric conflict via SAEs** (SpARE, NAACL 2025,
  [arXiv 2410.15999](https://arxiv.org/abs/2410.15999)) **[abstract]**: knowledge conflict shows up in
  *mid-layer* residuals. Steering fewer than 0.05% of SAE activations changes which source the model uses,
  with up to 15% accuracy gains.
- **INTRYGUE** (2026, [arXiv 2603.21607](https://arxiv.org/pdf/2603.21607v1)) **[abstract]**: gates predictive
  entropy by induction-head activity for RAG uncertainty.

**Verdict.**
- Causal: yes. Removing these heads causes hallucination.
- Domain-invariant scale: no. ReDeEP needs per-dataset hyperparameters and collapses out of distribution, and our
  own copying-head false-alarm rate drifted the same way.
- The usable part is that these features **combine well with a probe**. Our copying-head features added 3-5
  AUROC points to the probe. Use them as features inside a calibrated model, not as a threshold.

### 1.3 Confidence-regulation neurons

- **Stolfo, Wu, Gurnee, Belinkov, Song, Sachan, Nanda, NeurIPS 2024**
  ([arXiv 2406.16254](https://arxiv.org/abs/2406.16254)) **[abstract]**
  - **Entropy neurons:** high-norm neurons that write into the unembedding null space and scale logits through
    the final LayerNorm.
  - **Token-frequency neurons:** push the output toward or away from the unigram distribution.
  - Induction heads use entropy neurons to *hedge* on repeated text.
- **No 2025-2026 paper uses them as a hallucination detector** (searched). Two related methods read
  logit-lens or hidden-state trajectories without naming these neurons:
  - TriLens ([arXiv 2606.01033](https://arxiv.org/abs/2606.01033)) **[abstract]**: per-layer logit-lens entropy
    over the attention, MLP and residual paths, fed to a probe. The three paths are complementary. A
    disagreement term adds little.
  - Chain-of-Embedding ([ICLR 2025, arXiv 2410.13640](https://arxiv.org/abs/2410.13640)) **[secondary]**:
    label-free, training-free features of the hidden-state trajectory across layers (magnitude and angle
    changes). Reported **about +8 AUROC over the best output-based baseline** across 7 LLMs, on the model's
    *own* outputs.

**Verdict.**
- These neurons measure the reader's next-token confidence. Our reader's confidence is at chance on other models'
  text (docs/source_flow.md), so expect little.
- One exception is worth a cheap test: **token-frequency-neuron activity** ("falling back to the unigram
  prior") at *reply* tokens may flag tokens the reader finds unpredictable from both context and memory. This is
  close to the existing "neither" category in grounding_trace.md.

### 1.4 Truth directions

- **Bürger, Hamprecht, Nadler, "Truth is Universal", NeurIPS 2024**
  ([arXiv 2407.12831](https://arxiv.org/abs/2407.12831)) **[abstract]**
  - Truth lives in a **2-D subspace**: a general direction t_G plus a polarity direction. Ignoring polarity
    explains why earlier probes failed on negations.
  - The TTPD detector: 95% on simple lies, 82% on real-world lies.
- **Vicidomini, "The Anatomy of a Truth Direction"** (Jul 2026,
  [arXiv 2607.16741](https://arxiv.org/abs/2607.16741)) **[abstract]**: the subspace's dimension depends on what
  the model knows. A label-free decomposition recovers the polarity direction at cosine 0.959. Small models.
- **Liu et al., "Universal Truthfulness Hyperplane", EMNLP 2024**
  ([arXiv 2407.08582](https://arxiv.org/abs/2407.08582)) **[secondary]**
  - A probe trained on 40+ datasets reaches **about 70% cross-task**, against about 50% for single-dataset probes.
  - **Diversity beats volume**: as few as 10 samples per dataset suffice.
  - **This is the most actionable result for fast adaptation.**
- **Orgad et al., "LLMs Know More Than They Show", ICLR 2025**
  ([arXiv 2410.02707](https://arxiv.org/abs/2410.02707)) **[abstract]**
  - Signal concentrates at the *exact answer tokens*.
  - Probes transfer only between tasks that need the same skill. Apparent transfer is often explained by logit
    features.
- **Semantic-entropy probes** (Kossen et al. 2024, [arXiv 2406.15927](https://arxiv.org/abs/2406.15927))
  **[abstract]**: probes trained on *semantic entropy* (label-free, from samples) generalize out of distribution
  **7.7-10.5 AUROC points better** than probes trained on accuracy labels, but are worse in distribution.

**Verdict.** The mechanistic story ("the model keeps a consistent belief state") predicts some cross-domain
validity, and diverse training realizes it partly (about 70%, not 90%). The robust recipe:
- *label-free or proxy targets* (semantic entropy, the entity contrast, grounding-by-construction);
- *many small sources*;
- *read at the claim's key tokens*.

### 1.5 Steering vectors

Steering is evidence that a direction is causal, not a detector:
- ITI ([Li et al. 2023](https://arxiv.org/abs/2306.03341)) roughly doubles TruthfulQA truthfulness for Alpaca.
- Ferrando's latents push refusal to about 100%.

The 2026 replication (§1.1) shows that steering "known" up causes confabulation, while steering "unknown" up
causes refusal. That is useful for **mitigation** in VeriChat (`vt/guided.py`), not for scoring.

### 1.6 Logit lens, tuned lens and layer disagreement

- Correct and incorrect answers show different *commitment trajectories*:
  - [arXiv 2603.13259](https://arxiv.org/html/2603.13259v1): the commitment ratio collapses for wrong answers;
  - [arXiv 2507.06722](https://arxiv.org/abs/2507.06722): tuned-lens trajectories over 11 datasets and 5 models;
  - [arXiv 2602.09486](https://arxiv.org/pdf/2602.09486): interlayer disagreement used for mitigation.
- All of these are correlational and measured on-policy. GhostTrack (ICML 2026,
  [listing](https://icml.cc/virtual/2026/79481)) **[abstract]** tracks competing SAE features across layers.
  HaluEval QA AUROC: **0.903 on Phi-2, 0.722 on GPT-2-M, 0.650 on Qwen2.5-1.5B**. The spread across models is
  large and no baselines are listed.
- SAFER (NeurIPS 2026, [listing](https://neurips.cc/virtual/2026/162629)) **[abstract]**: SAE-feature entropy
  trajectories, Gemma-2-2B-IT. **AUROC 0.726 on PopQA, 0.633 on TriviaQA.** Below the 0.72 our neuron contrast
  already gets on TriviaQA.

### 1.7 SAE-based hallucination detectors, 2025-2026

| work | model / dictionary | data | reported | compared with a raw-activation probe? |
|---|---|---|---|---|
| RAGLens, ICLR 2026 ([arXiv 2512.08892](https://arxiv.org/abs/2512.08892)) **[read]** | Llama-2-7B/13B; SAEs at layer 15; top-1000 features by mutual information, then a GAM | RAGTruth / Dolly | **0.841 / 0.876** AUC (7B), **0.896 / 0.857** (13B); ReDeEP 0.746 / 0.795; SAPLMA 0.711 | **No** plain logistic baseline. Dubanowska's LR on the same task gets 0.795 |
| RAGLens transfer **[read]** | Llama-2-7B | train RAGTruth → AggreFact 0.80, TofuEval 0.76; train AggreFact → RAGTruth **0.53** | asymmetric | |
| RAGLens, Llama-3.2-1B **[read]** | EleutherAI SAE (MLP, layer 6) and transcoder | RAGTruth | accuracy 0.78, F1 0.79; SAE ≈ transcoder; pre-activation > post-activation | |
| Dubanowska et al. **[read]** | Llama-2-7B SAEs, layers 12-15 | RAGTruth → SQuAD | SAE 0.711 in distribution, **0.485 out of distribution** | **Yes**: LR 0.795 / 0.514 |
| Vicentino, ICML 2026 **[abstract]** | Qwen3.6-27B TopK SAE | known vs unknown entity | 0.814 | **Yes**: LR 0.887 |
| HalluSAE ([arXiv 2604.16430](https://arxiv.org/abs/2604.16430)) **[abstract]** | Gemma-2-9B | — | claims state of the art; no numbers in abstract | unknown |
| GhostTrack, SAFER | see §1.6 | | | no |
| Transcoders on VLM hallucination ([arXiv 2605.22902](https://arxiv.org/html/2605.22902v1)) **[secondary]** | transcoder attribution maps | VLM grounding | AUC 0.68 (prior baseline 0.55) | no |

RAGLens's Qwen3-4B SAEs (layer 22) were used only for a CoT-vs-SAE comparison; no AUC was reported for them.

---

## 2. Do SAE-feature probes beat raw-activation probes?

**No, on average, in every regime tested so far.**

- **Kantamneni, Engels, Rajamanoharan, Tegmark, Nanda, ICML 2025**
  ([arXiv 2502.16681](https://arxiv.org/abs/2502.16681)) **[read]**
  - 113 datasets on Gemma-2-9B, replicated on Llama-3.1-8B, in four regimes: few data (2 to 1024 examples),
    class imbalance, label noise, covariate shift.
  - SAE probes give **no meaningful average gain in any regime** and are **worse under covariate shift**. Adding
    SAEs to an ensemble of baselines slightly *hurts*.
  - Bricken et al.'s claimed advantage held on 2.2% of datasets.
  - Probe quality depends mainly on SAE **L0, not width**.
  - Where SAEs do help:
    - one latent: in-distribution AUC 0.99, out-of-distribution 0.64, showing how sparse features can be
      spuriously brittle;
    - **pruning spurious latents improved out-of-distribution AUC by 0.02-0.08**;
    - finding mislabeled data (CoLA);
    - max-pooled multi-token probes (19.6% win rate against last-token baselines, but only 8.7% against attention
      pooling).
- **GDM mech-interp team, Mar 2025**
  ([post](https://deepmindsafetyresearch.medium.com/negative-results-for-saes-on-downstream-tasks-and-deprioritising-sae-research-gdm-mech-interp-team-progress-update-2-d00c3cf4e9a9),
  [mirror](https://www.greaterwrong.com/posts/4uXCAJNuPKtKBsi28)) **[secondary]**
  - Harmful-intent detection under new jailbreaks: k-sparse SAE probes fit and generalize in distribution but are
    **distinctly worse out of distribution** than dense linear probes.
  - Fine-tuning the SAE on chat data closes about half the gap.
  - The team deprioritized SAE research.
- **Dubanowska 2025 and Vicentino 2026:** same conclusion for hallucination specifically (§1.7).
- **Counter-evidence:** RAGLens reports large wins over SAPLMA, ITI and ReDeEP, but no same-layer
  logistic-regression baseline, and its out-of-distribution transfer is asymmetric (0.53 in one direction).
  Treat it as "SAE features + MI selection + GAM ≈ a good probe, with explanations".

**Where SAEs or transcoders *can* help us:**
1. **Label-free feature selection by contrast.** Pick the latents that separate real from invented entities,
   evidence-supported from evidence-swapped sentences, and so on. A dictionary gives *candidate units* with
   cleaner single meanings than raw neurons. But our raw-neuron contrast (10 neuron poles) already beats a dense
   probe on the same contrast. If Goodfire's "neurons are already clean" holds for Qwen3, the SAE adds little.
   Test, do not assume.
2. **Auditing the probe for confounds.** Project the probe's weight vector onto transcoder decoder directions and
   read the top features. If "answer is one word", "JSON/table format", "news register" or "sentence position"
   features dominate, that explains the scale shift with domain and length. Pruning such directions is the one
   documented out-of-distribution gain from SAEs (+0.02 to +0.08).
3. **Explanations for customers.** Name the features behind a verdict (RAGLens's GAM rationales; the audit
   receipts idea in IDEAS.md).

---

## 3. Off-the-shelf dictionaries for small open models

Checked through the HF API on 2026-11-07 **[verified]** unless noted. Sizes are per layer.

### For our reader (Qwen3-4B) and its family

| repo | what | model | coverage | size/layer | license |
|---|---|---|---|---|---|
| **`mwhanna/qwen3-4b-transcoders`** | per-layer **transcoders** (MLP in → MLP out), ReLU, d_feature **163,840**, d_model 2560 | `Qwen/Qwen3-4B` (config.yaml) | all 36 layers, plus `features/` activation dashboards | 1.68 GB (safetensors) | MIT |
| `mntss/qwen3-4b-transcoders-temp` | earlier or temporary version of the above | Qwen3-4B | ? | ? | ? |
| **`gzxiong/sae-qwen3-4b`** | residual **TopK SAE** (EleutherAI `sparsify`), k=16, 32× (81,920 latents) | `Qwen/Qwen3-4B`; trained on SmolLM2 10B, 8-bit model | all 36 layers (`layers.N/sae.safetensors`) | 1.68 GB | MIT |
| `hippotoday/sae-qwen3-4b-base` | sparsify SAE | Qwen3-4B-Base | layer 16 only | 1.68 GB | — |
| `chungimungi/SAE-Qwen_Qwen3-4B-Instruct-2507-layer_{18,24}` | small community SAEs | Qwen3-4B-Instruct-2507 | 2 layers | — | unverified quality |
| **Qwen-Scope** (official, Apr 2026; [report](https://qianwen-res.oss-accelerate.aliyuncs.com/qwen-scope/Qwen_Scope.pdf), arXiv 2605.11887): `Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_{50,100}`, `Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_{50,100}`, `Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_{50,100}`, `Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_{50,100}`, Qwen3.5-27B, Qwen3-30B-A3B, Qwen3.5-35B-A3B | residual TopK SAEs, `layer{n}.sae.pt` with W_enc/W_dec/b_enc/b_dec | **no Qwen3-4B** | every layer | 537 MB (1.7B, 32K) | Qwen license ("other") |
| `OpenMOSS-Team/Llama-Scope-2-Qwen3-1.7B` | **transcoders + LoRSA** (sparse attention replacement) at 8× and 32×, k = 64/128/256 | Qwen3-1.7B | all layers | about 2.2 GB (32× LoRSA) | — |
| `mwhanna/qwen3-1.7b-transcoders-lowl0`, `mwhanna/qwen3-0.6b-transcoders-lowl0`, `mwhanna/qwen3-8b-transcoders`, `mwhanna/qwen3-14b-transcoders-lowl0` | transcoders | Qwen3 family | all layers | — | — |

Neuronpedia hosts dashboards for the Qwen3-4B transcoders, for example
[qwen3-4b/10-transcoder-hp](https://www.neuronpedia.org/qwen3-4b/10-transcoder-hp), and builds attribution graphs
for Qwen3-1.7B.

**Fit with our probe.** Our probe reads MLP activations, and the mwhanna transcoders decompose exactly the MLP
input→output map of our reader. That makes them the natural dictionary. The Qwen3-4B checkpoint is the hybrid
thinking/instruct model, so it matches our reader without base→chat transfer, if the reader is `Qwen/Qwen3-4B`.

### Gemma

- **Gemma Scope** (Gemma-2): `google/gemma-scope-2b-pt-res`, `-mlp`, `-att`, `google/gemma-scope-2b-pt-transcoders`,
  `google/gemma-scope-9b-pt-res`, all verified. JumpReLU, widths 16k-1M, every layer. Ferrando's entity latents
  live here: Gemma-2-2B, layers about 9-13.
- **Cross-layer transcoders for Gemma-2-2B:** `mntss/clt-gemma-2-2b-426k` (verified), `mntss/clt-gemma-2-2b-2.5M`.
  The per-layer set used by circuit-tracer is `mntss/gemma-scope-transcoders`.
- **Gemma Scope 2** (Gemma 3, Dec 2025, [announcement](https://www.alignmentforum.org/posts/YQro5LyYjDzZrBCdb/announcing-gemma-scope-2)):
  - repos `google/gemma-scope-2-{270m,1b,4b,12b,27b}-{pt,it}` (270m-pt, 270m-it, 1b-pt, 1b-it, 4b-it verified);
  - each repo has `resid_post(_all)`, `mlp_out`, `attn_out`, `transcoder(_all)` (with affine skip) and
    `crosscoder`; 270m and 1b also have **`clt`** cross-layer transcoders;
  - the **-it** variants avoid the base→chat mismatch.

### Llama

- `EleutherAI/sae-Llama-3.2-1B-131k` and `EleutherAI/skip-transcoder-Llama-3.2-1B-131k` (both verified; RAGLens used
  the MLP layer-6 one).
- `mntss/transcoder-Llama-3.2-1B` and `mntss/clt-llama-3.2-1b-524k` (verified; circuit-tracer).
- **Llama Scope** for Llama-3.1-8B: `OpenMOSS-Team/Llama-Scope` (the old `fnlp/` names redirect),
  `OpenMOSS-Team/Llama3_1-8B-Base-LXR-8x`, and an R1-distill set `OpenMOSS-Team/Llama-Scope-R1-Distill`.
- `EleutherAI/sae-llama-3-8b-32x`.

### CPU cost

- **Encoding one layer:**
  - Qwen3-4B transcoder: 2560 × 163,840 ≈ 0.42 GFLOP per token;
  - gzxiong SAE: about 0.21 GFLOP per token;
  - a 4B forward pass: about 8 GFLOP per token.

  So encoding 2-4 layers adds about 10-20% to a pass. Memory is about 1.7 GB per layer in bf16. Load the encoder
  half only (W_enc), or slice it to the selected latents. A served detector that uses 50 selected latents needs
  only 50 encoder rows, which costs nothing.
- **Attribution graphs** (circuit-tracer, CRV) need backward passes per active feature. They take minutes per
  short prompt on GPU-class hardware at 1-4B, and are impractical on our 4-core CPU except for a handful of case
  studies on Qwen3-0.6B/1.7B or Gemma-3-270m/1B.

---

## 4. Mechanistic signals for long chain-of-thought verification

| work | signal | numbers | transfer |
|---|---|---|---|
| CRV, Meta FAIR, ICLR 2026 oral ([arXiv 2510.09312](https://arxiv.org/abs/2510.09312)) **[read]** | structure of per-step attribution graphs (own transcoders, Llama-3.1-8B-Instruct) → classifier | AUROC: Boolean 0.759 (baseline 0.588), Arithmetic **0.925** (0.765), **GSM8K 0.702** (Energy 0.626) | **Poor**: Arithmetic→GSM8K 0.570 (below baseline), GSM8K→Boolean 0.458. "Too computationally intensive" for a drop-in verifier. Error signatures are domain-specific |
| Zhang, He et al. 2025 ([arXiv 2504.05419](https://arxiv.org/abs/2504.05419)) **[abstract]** | 2-layer MLP probe on hidden states at intermediate-answer positions in a reasoning model's CoT | AUROC above 0.7, ECE below 0.1 | GSM8K ↔ MATH good; AIME weaker. Correctness is predictable *before* the answer is written. Weaker in non-reasoning models |
| ReProbe, ACL 2026 ([site](https://reprobe.github.io/)) **[read]** | probe under 10M parameters (small transformer) over **step-aggregated** hidden states of a frozen Qwen3-1.7B/8B/32B or Phi-4; labels from an LLM judge or self-annotation | step PR-AUC: math in domain about equal to 7-8B PRMs; **out of domain it beats PRMs** (planning .80 vs .73, QA .47 vs .38 at Qwen3-8B) | step aggregation > token-level > linear |
| Obeso et al. 2025 ([arXiv 2509.03531](https://arxiv.org/abs/2509.03531)) **[read]** | token-level linear or LoRA probe for hallucinated **entities**, layer about 0.95·L, max over span | LongFact 0.85-0.87 (linear), 0.89-0.91 (LoRA); semantic entropy about 0.71; **HealthBench held out 0.86-0.91**; entity-trained probe on **MATH 0.85-0.89** | Long-form → short-form works; the reverse does not. **Cross-model within 0.02-0.04 AUC.** Data on HF: `obalcells/longfact-annotations`, `obalcells/healthbench-annotations`, `obalcells/hallucination-heads-math`, `obalcells/hallucination-heads-longfact-augmented-{medical,legal,biography,citations}` |
| Thought Anchors ([arXiv 2506.19143](https://arxiv.org/abs/2506.19143)) **[abstract]** | **receiver heads** that focus on a few "broadcasting" sentences (planning and uncertainty-management steps) | — | a mechanistic way to find *which* steps the rest of the chain depends on, so an error there matters most |
| Retrieval heads (§1.2) | heads that refer back to the question and earlier steps | masking them hurts CoT strongly | — |
| Motivated reasoning / hint probes ([arXiv 2603.17199](https://arxiv.org/abs/2603.17199); BonaFide [2605.25052](https://arxiv.org/html/2605.25052)) **[abstract]** | residual probes detect hint-driven rationalization even when the CoT hides it | about 0.85-0.95 in hint setups; faithfulness metrics in the wild about 0.59-0.70 and do not transfer | narrow |
| Anthropic biology, CoT section **[read]** | graphs separate "really computed", "bullshitting" (no computation) and "worked backward from the hint" | proof of concept, not a classifier | authors: not shown to generalize |
| This repo (docs/source_flow.md) | reply share minus sink share | **0.640** on Math-Shepherd steps, label-free | the same sign on answers, steps and biography sentences |

**What helps for long CoT:**
1. **Step-aggregated probe readout.** Pool each step's token states (attention pooling), not the last token. ReProbe
   and Kantamneni's pooling results agree.
2. **Entity/claim-token probes reused on math.** Obeso's 0.85-0.89 on MATH from an entity probe suggests a shared
   "I'm making this up" representation.
3. **Source flow plus retrieval/receiver heads**, to tell a step that rests on the question from one that rests on
   its own recent text (our 0.64 signal), and to weight steps by how much later steps depend on them.
4. Attribution graphs: research only.

**Caveat.** Most CoT probe results read the *generating* model's own states. Our reader reading another model's
CoT is off-policy. Obeso's cross-model result (within 0.02-0.04 AUC) is the best evidence that this still works
for entity-type errors.

---

## 5. Answers to the five questions

### (1) Which mechanistic signals are causally tied to hallucination, and are they domain-invariant?

| signal | causal evidence | domain invariance | calibrated? |
|---|---|---|---|
| known/unknown-entity latents gating "can't answer" | strong (steering about 100% refusal; Anthropic misfire; 27B replication) | across entity types, by construction | **No.** Tracks confidence, not correctness; steering induces confabulation; reader ≠ writer |
| retrieval / copying heads (context use) | strong (pruning → hallucination) | weak in practice (ReDeEP OOD 0.50; our false-alarm drift) | No; needs per-domain fitting |
| knowledge FFNs over-writing (ReDeEP PKS) | moderate | weak | No |
| truth subspace (t_G + polarity) | moderate (ITI steering) | partial: about 70% cross-task with diverse training | No, but diversity helps |
| entropy / token-frequency neurons | strong for confidence | n/a for off-policy reading | — |
| layer trajectories (logit lens, CoE, GhostTrack) | correlational | varies a lot by model (0.65-0.90) | No |

No mechanistic signal gives a calibrated, domain-invariant score off the shelf. The mechanism fixes the *sign*.
Base rates, answer length and register move the *scale*. The way to calibrate with no labels is:
- per-domain normalization of each mechanistic feature on the customer's unlabeled traffic (quantiles or
  z-scores, which grounding_trace.md already does for natural scales);
- plus a few labels on top (the universal-hyperplane result: about 10 per source is enough once training is
  diverse).

### (2) Do SAE-feature probes beat raw-activation probes?

No: not in distribution, not out of distribution, not with few labels, not under label noise. Sources:
Kantamneni (113 datasets), GDM, Dubanowska, Vicentino. The out-of-distribution regime is where SAEs are *worse*.
SAEs pay off for label-free feature selection, confound auditing and pruning (+0.02-0.08 out of distribution),
and explanations.

### (3) Off-the-shelf SAEs or transcoders usable on CPU

Yes (§3). For our reader:
- `mwhanna/qwen3-4b-transcoders` (MLP transcoders, matching our MLP probe);
- `gzxiong/sae-qwen3-4b` (residual SAEs).

Smaller alternatives:
- `Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50`, `OpenMOSS-Team/Llama-Scope-2-Qwen3-1.7B`;
- Gemma: `google/gemma-scope-2b-pt-*`, `google/gemma-scope-2-{270m,1b}-{pt,it}` (with CLTs);
- Llama: `EleutherAI/sae-Llama-3.2-1B-131k`, `mntss/clt-llama-3.2-1b-524k`.

### (4) Mechanistic signals for long CoT

Step-pooled hidden-state probes (ReProbe; Zhang et al.), entity/claim probes that transfer to math (Obeso),
source flow, and retrieval/receiver-head step weighting. Circuit-based verification (CRV) gives
interpretability, but its transfer is poor and it is too costly.

### (5) Experiments: see §6

---

## 6. CPU-feasible experiments, ranked by expected value

Costs assume the Qwen3-4B reader on 4 CPU cores. The existing benchmarks are HaluBench (300, 6 sources),
RAGTruth, Math-Shepherd and WikiBio.

**E1. Leave-one-source-out, diverse-few-shot probe recipe.** *(Highest expected value; directly targets
adaptation and calibration.)*
- **Method.** Train the existing MLP probe on many sources with 10-50 labeled sentences each. Use HaluBench's 6
  sources, RAGTruth's 3 task types, the obalcells long-form domains (medical, legal, biography, citations), and
  Math-Shepherd. Test on the held-out source.
- **Compare** with single-source training and the current label-free probe.
- **Add label-free targets** next to human labels: grounding-by-construction, as now, plus the semantic-entropy-
  probe idea (sample the reader 5× on short answers).
- **Report** AUROC and the **false-alarm rate at a threshold fixed from the other sources**.
- **Mechanistic justification:** the universal-hyperplane (diversity > volume) and SEP results.
- **Cost:** features are mostly cached already, so hours.

**E2. Confound audit of the current probe with the matching transcoder.** *(High; explains the scale shift.)*
- Download 2-3 layers of `mwhanna/qwen3-4b-transcoders` around the probe layer.
- Express the probe weight (or its correlation with the activations) in the transcoder's feature basis, and read
  the top features on Neuronpedia.
- Also regress the probe score on answer length, sentence position, source and task type (the RAGTruth
  data-to-text confound).
- Remove the top confound directions and re-test the cross-source false-alarm rate.
- **Expected:** a cut in length- and format-driven drift. Kantamneni reports +0.02-0.08 out of distribution from
  pruning.
- **Cost:** about 5 GB download, a few hours.

**E3. Entity-familiarity feature in the SAE/transcoder basis, label-free.** *(Medium-high.)*
- Repeat the real-vs-invented-entity contrast from `experiments/recall_neurons.py`. Select latents from
  `gzxiong/sae-qwen3-4b` (residual layers about 14-22) and from the transcoders, using the MaxMin rule across
  entity types (Ferrando).
- Score:
  - (a) closed-book TriviaQA/NQ: does it beat the 0.72/0.69 of the 10 raw neuron poles?
  - (b) as a *combined* RAG feature, "reply entity is unknown to the reader AND has low evidence shift", on
    HaluBench and RAGTruth.
- **Expected:** parity with the raw neurons on (a), per Vicentino. Possibly cleaner transfer on (b).
- **Cost:** about 1 day.

**E4. Obeso-style span probe, read off-policy.**
- Train a linear probe on Qwen3-4B reading `obalcells/longfact-annotations` (texts written by Llama), max-pooled
  over entity spans.
- Test on held-out `obalcells/healthbench-annotations`, `obalcells/hallucination-heads-math`, WikiBio and
  RAGTruth.
- This tests whether the strongest published long-form probe survives our *reader-not-writer* setting, and gives
  a closed-book long-form signal the product lacks.
- **Cost:** about 1-2 days of CPU forward passes. Subsample to about 2k spans.

**E5. Lookback-ratio features per head with predefined spans.**
- Compute Lookback Lens per-head ratios per sentence from the attention already hooked in `vt/sourceflow.py`.
- Train on RAGTruth QA and test on Summary, Data2Txt, CovidQA, etc.
- Compare with pooled source-flow shares (0.44-0.52 cross-kind).
- The paper's 0.82-0.85 cross-task transfer is the target to beat. Also run the task-type-only baseline
  (Dubanowska).
- **Cost:** about 1 day.

**E6. Step-pooled probe for CoT (ReProbe-lite).**
- On Math-Shepherd steps: attention-pooled MLP activations per step, plus source flow, plus the E4 entity probe
  score.
- Weight steps by receiver-head attention (Thought Anchors) when building the solution-level score.
- Train on GSM8K steps, test on MATH steps.
- **Cost:** about 1 day; activations are partly cached.

**E7. Label-free layer-trajectory features** (TriLens / Chain-of-Embedding style).
- Per reply token: logit-lens entropy at every 4th layer, and hidden-state angle and norm changes. Also
  token-frequency-neuron activity (projection onto log-unigram logits).
- Add them as features to the probe. Cheap, but low prior: these are on-policy signals and reader confidence is
  already at chance.
- **Cost:** hours.

**E8. Attribution-graph case studies** (no expected AUROC gain).
- circuit-tracer on Qwen3-1.7B (`mwhanna/qwen3-1.7b-transcoders-lowl0`) or Gemma-3-1B CLTs, for 10-20 RAG
  examples where the probe fails.
- Purpose: to understand the failure modes, for example why DROP and FinanceBench one-word answers are
  *inverted* (0.40). Not a product feature.
- **Cost:** CPU minutes to hours per prompt. Strictly a handful.

**Not recommended:**
- training our own SAEs (no GPU, and the dictionaries above exist);
- SAE-probe-vs-raw-probe headline comparisons beyond E3 (the answer is known);
- steering-based detection.

---

## 7. Open uncertainties

- The 2026 papers (Vicentino, GhostTrack, SAFER, HalluSAE, TriLens, the Anatomy paper) were read at abstract
  level only. Their numbers are as the authors claim and unreplicated.
- Qwen-Scope details are from its model card and a secondary summary. It does not include Qwen3-4B.
- The quality of `gzxiong/sae-qwen3-4b` is unknown: k=16 is a low L0, and Kantamneni finds L0 drives probe
  quality. The mwhanna transcoders' L0 and fidelity are undocumented (README is 54 bytes), so check their
  reconstruction loss before relying on them.
- No paper tests mechanistic signals in our exact setting: a fixed reader scoring *other models'* outputs with
  evidence. Obeso's cross-model 0.02-0.04 AUC gap and Lookback Lens's 7B→13B transfer are the closest evidence
  that this works. Kirch et al. warn that off-policy data hurts most for intent-like behaviours, less for
  text-content ones; hallucination is closer to text content.
