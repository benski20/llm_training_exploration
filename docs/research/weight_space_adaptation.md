# Weight-space programming and fast domain adaptation for the verifier

Research note, 2026-11-07. Scope: what "weight-space programming" and related ways of writing or generating weights
offer a verifier built as a frozen reader (Qwen3-4B) plus a small readout. The note also asks how to adapt that
verifier to a new customer domain in minutes, with 0-100 labels, while keeping a guaranteed false-alarm rate (FAR).
Compute assumed: 4-core CPU, no GPU, models of 4B parameters or fewer.

Starting point in this repo:
- docs/written_readout.md: a hand-written 6-weight readout ties the shipped check, 0.755 against 0.765 AUROC on
  RAGTruth QA, label-free. Label-free thresholds miss the 10% target, giving 11-46% false alarms. A Clopper-Pearson
  threshold holds from 100 labelled replies and catches 22/32/42% at 100/200/300 labels.
- docs/product.md: a supervised probe trained on 800 labels reaches 0.86.

Evidence labels used below:
- **[read]**: the primary source, or its full HTML, was fetched for this note.
- **[abstract/secondary]**: only the abstract or a third-party summary was seen.
- **[memory]**: well known, but not re-fetched this session. Check the numbers before quoting them externally.

---

## 0. Answers in brief

1. **Usable from weight-space programming.** Very little of it applies literally. What carries over is an idea:
   - Small, inspectable parameter sets can be written from reasoning plus unlabeled statistics. We have already done
     this, and it tied the default.
   - A "direction" in activation space found from unlabeled contrasts is a cheap, reusable adaptation unit. This
     covers steering vectors, TSV and HaloScope.
   - Two families are not worth building here. Hypernetworks that emit LoRAs (Text-to-LoRA, Drag-and-Drop) target
     *generation quality*, need hundreds of training tasks and GPUs, and give no guarantee. Weight editing
     (ROME/MEMIT, Sparks' Qwen edit) changes what a model *knows or does*, not how a readout is calibrated.
2. **Best recipe for minutes and 0-100 labels.** Split the problem in two:
   - **Ranking: label-free.** Use the shipped readout on domain-standardised signals. Optionally adapt it with a
     few labels plus pseudo-labelled unlabeled data, shrinking toward the written weights.
   - **Threshold: needs labels, but only *clean* ones.** FAR = P(score > t | clean) depends only on the clean-class
     score distribution. A distribution-free guarantee at FAR ≤ 10% with 95% confidence needs **at least 29 clean
     replies**, and recall grows with the clean count (table in section 4).
   - Sample replies at random from traffic, label them, and keep the clean ones for an order-statistic or
     Clopper-Pearson threshold.
   - Use unlabeled traffic for three things: covariate-shift reweighting (weighted conformal), prevalence
     estimation (for precision), and prediction-powered variance reduction.
   - No method can beat the binomial floor for a distribution-free guarantee. Methods differ only in how much
     *recall* they keep at that floor.
3. **Can a hypernetwork map a description or unlabeled samples to probe weights or calibration?**
   - It is plausible in principle. Drag-and-Drop maps 128 *unlabeled* prompts to LoRA weights, and Text-to-LoRA
     maps a description to LoRA weights.
   - The evidence is thin for our case:
     - Both need hundreds of training tasks (T2L: 479; DnD degrades sharply with few training datasets).
     - No published work generates *probe or detector* weights or calibration this way. I searched and found none.
     - Cross-domain transfer of hallucination probes is poor (Orgad et al.; Dubanowska et al., "close to random"
       out of distribution).
   - With our ~10-30 available domains, a "hypernetwork" reduces to meta-regression over 6 readout weights plus a
     threshold prior. That prior can give a 0-label *advisory* threshold, never a guaranteed one.
4. **Experiments, ranked by expected value:** section 5. Top three:
   - (E1) re-express the label budget as *clean* replies and a random-sample protocol;
   - (E2) weighted conformal against the train→test shift that broke the plain quantile;
   - (E3) few-label readout refit shrunk toward the written weights, combined with TSV-style pseudo-labelling of
     unlabeled replies.

---

## 1. "Sparks of Weight-Space Programming": what it actually claims

**Source.** Rikiya Takehi, Jayden Teoh, Alex Zhang, Omar Khattab (MIT CSAIL), blog post plus preprint, 2 Nov 2026.
- It has not been peer reviewed.
- Primary page: <https://rikiyat.github.io/blog/weight-space-programming/> **[read]**.
- Logs: <https://github.com/RikiyaT/lm_understand-logs>.
- Critique: A. Maier, <https://akmaier.substack.com/p/weight-space-programming-what-is> **[read]**.
- Explainer: Turing Post, <https://www.turingpost.com/p/what-is-weight-space-programming-and-why-it-suddenly-blew-up-the-internet> **[read]**.

**Claim.** A frontier LLM, GPT-6 Astra used off the shelf, can *read* neural-network weights and *write* them toward
a natural-language goal "without an optimizer or gradient descent". The authors call these "very early results".
- No training loops or search were allowed.
- Some experiments allowed Python analysis, or a budget of 0-50 forward passes.

| # | Task | Result | Strength |
|---|---|---|---|
| 1 | Name the game from a 17,253-weight policy net | 4/4 correct (Breakout, Pong, Pac-Man, Freeway) | anecdotal, n=4 |
| 2 | Name the shape a 23.8k-weight diffusion model makes | circle/triangle/square right; drew outlines, not filled shapes | anecdotal |
| 3 | Training language of char-LMs from middle layers (7.16M weights); find copying heads | 3/3 languages (6/6 attempts, confidence 0.48-0.62); all copying heads, no false positives | heads part checked against ground truth |
| 4 | Repair a failing Kinetix car policy | 0/256 → 1024/1024 after editing 3.5% of weights in one try | measured |
| 5 | Write MinAtar nets from scratch, no rollouts | Breakout 517.4 vs PPO 25.3; Seaquest 53.1 vs 88.9; Asterix 53.2 vs 55.2 | measured, weak baselines (authors' own PPO) |
| 6 | Write a 9×9 Go net (1.48M weights) | 0 passes: >50% vs small PPO nets; 50 passes: beats same-size PPO trained on ~20M games (70-79%) | measured, weak baselines |
| 7 | Find the sycophantic Gemma-3-270M from weights | 0/4 from weights alone; 3/3 with an attention-pattern tool | anecdotal; 3/3 is chance at 1/8 |
| 8 | Fix Qwen3.5-0.8B looping (edit 131,072 weights = 0.016%, two MLP layers) | MATH-500 7.9→48.1%; GSM8K 13.8→67.1%; IFEval 21.0→62.3%; length-limit hits 93→33% (50 passes) | measured, the headline |
| ctrl | Same procedure on Gemma-3-270M | 6.8 → 7.0-7.7% (no gain) | negative control |

The authors' own limits:
- Editing can "remove an obstacle that hid an ability", not "add an ability the model lacked".
- Context length and harness are bottlenecks.

The critique adds:
- One model was used, and most conditions were run once.
- Results are non-monotone in budget. 10 passes gave 12.3% on MATH-500, and a prompt ablation at 25 passes gave
  14.1%.
- There was no comparison to standard direction ablation or decoding fixes for looping.
- A Claude Code-expanded prompt contained a repetition hint.

**Reading for us.**
- Experiment 8 is in substance *find a direction from contrasting activations, then project it out*. That is
  representation engineering done by an LLM agent.
- Experiment 3 found copying heads, which we already find mechanically (docs/copy_heads.md).
- Nothing in the paper concerns calibration, thresholds, or domain transfer of a classifier.
- The useful lesson is the one we already acted on. For a *small, semantically meaningful* parameter set (our six
  signals), weights written from reasoning plus unlabeled statistics are competitive: 0.755 against 0.765.
- We cannot run GPT-6 Astra with weight access locally.

---

## 2. The landscape, and what each line offers a verifier

### 2.1 Hypernetworks that emit adapters

| Work | Input → output | Key numbers | Relevance |
|---|---|---|---|
| **Text-to-LoRA** (Charakorn, Cetin, Tang, Lange; ICML 2025; <https://arxiv.org/abs/2506.06105>) **[read]** | task description (gte-large embedding) → rank-8 LoRA on q/v of Mistral-7B; hypernet 4.9-55M params | reconstruction of 9 LoRAs: 73.4 vs 73.3 task-specific; **zero-shot on 10 unseen benchmarks: 67.7 vs multi-task LoRA 66.3, base 55.8** (oracle task LoRAs ~75.8 on 8 tasks); trained on 479 SNI tasks × 128 descriptions; reconstruction-trained T2L generalises poorly (61.8); unaligned descriptions drop to 51.4 | the zero-shot gain over a single multi-task adapter is +1.4 points. For us, the analogue of "one multi-task adapter" is the shipped probe. Expect small gains; needs hundreds of training domains |
| **Doc-to-LoRA** (Sakana, ICML 2026; <https://arxiv.org/abs/2602.15902>, <https://sakana.ai/doc-to-lora/>) **[abstract/secondary]** | document → LoRA (meta-learned context distillation), Gemma-2-2b | near-perfect needle-in-a-haystack at 5× the context window | internalises *facts*, which is what a verifier must not rely on; evidence should stay in context |
| **Drag-and-Drop LLMs** (Liang et al., 2025; <https://arxiv.org/abs/2506.16406>) **[read]** | batch of ~128 **unlabeled** prompts (Sentence-BERT) → full LoRA set (Qwen2.5 0.5-7B) | claims "up to 30%" over the training LoRAs on unseen sets (+21.0 common sense, +15.1 HumanEval, +23.4 GSM8K); 0.1-0.7 s per model on A100; prompt+answer conditioning *worse* than prompts alone on common sense (27.0 vs 51.6); **barely improves with only 2 training datasets** | the closest analogue to "unlabeled domain samples → weights". The baseline (LoRAs trained on *other* datasets) is weak, so treat the gains with caution. It shows domain *inputs* alone carry adaptation signal |
| **HyperTuning / HyperT5** (Phang et al. 2022; <https://arxiv.org/abs/2211.12485>), HyperLlama via gisting (2024; <https://arxiv.org/abs/2402.16817>) **[abstract/secondary]** | few-shot examples → prefix/LoRA | "generally underperforms" in-context learning with the same examples | an early negative-ish result: amortising few-shot adaptation into weights loses to just using the examples |
| Generative Adapter (Chen et al. 2024, arXiv 2411.05877) **[memory]** | context stream → adapter | not checked | same family |

**Verdict.** These generate *generation* adapters for a model. Our adaptable object is a 6-weight readout plus a
threshold, or at most a d≈10k-dimensional linear probe.
- The hypernetwork idea transfers only as meta-regression.
- All of these papers need many (task, weights) pairs. T2L's scaling shows the larger hypernet *falls* at fixed
  compute with more tasks. DnD collapses with 2 training datasets.

### 2.2 Task arithmetic and merging

Covered: task vectors (Ilharco et al. 2023, arXiv 2212.04089), TIES (Yadav et al. 2023, arXiv 2306.01708), DARE
(Yu et al. 2024, arXiv 2311.03099). All **[memory]**.
- These add or subtract fine-tuning deltas. TIES and DARE resolve sign conflicts and sparsify the deltas, so several
  adapters can coexist.
- For linear probes, merging is just averaging or ensembling readouts trained on different domains. That is trivial
  and already covered by "a probe trained on diverse data" (2.4).
- **Not useful** beyond a "probe soup" baseline, which is cheap and could be tried inside E7.

### 2.3 Editing and compiling weights

- **ROME/MEMIT** (Meng et al. 2022, arXiv 2202.05262 and 2210.07229) **[memory]**: rank-one or multi-layer MLP edits
  that insert facts. They are a form of "programming", but they change the reader's *knowledge*. A grounding
  verifier should judge against the evidence in context. Not useful, and edits have known ripple and side effects.
- **Tracr / RASP** (Lindner et al. 2023, arXiv 2301.05062), **Learning Transformer Programs** (Friedman et al.
  2023, arXiv 2306.01128) **[memory]**, and **ALTA** (Shaw et al. 2024; TMLR 2025; <https://arxiv.org/abs/2410.18077>)
  **[abstract/secondary]**: these compile symbolic programs (including loops in ALTA: parity, addition, SCAN) into
  transformer weights.
  - Relevance: exact sub-checks (copy match, number comparison, arithmetic) *could* be compiled into heads.
  - We already do them better as code (docs/written_readout.md: "exact parts are better written as code").
  - Not useful for adaptation.

### 2.4 Probes, steering vectors, unlabeled-data detectors

| Work | What | Numbers | Relevance |
|---|---|---|---|
| **HaloScope** (Du, Xiao, Li 2024; <https://arxiv.org/abs/2409.17504>) **[abstract/secondary]** | unlabeled LLM generations → SVD "hallucination subspace" → pseudo-labels → classifier | TruthfulQA AUROC 78.6 (LLaMA-2-7B) vs 81.0 supervised upper bound | label-free from unlabeled in-domain data; the same family as our "not drawn from evidence" axis |
| **TSV, Truthfulness Separator Vector** (Park, Du, Yeh, Wang, Li; ICML 2025; <https://arxiv.org/abs/2503.01917>) **[read]** | ~4k-param steering vector added to the residual stream; trained on **32-64 labelled exemplars**, then Sinkhorn optimal-transport pseudo-labels on unlabeled generations (top K=128 confident) and retrain | LLaMA-3.1-8B AUROC: TruthfulQA 84.2 (SAPLMA supervised 78.2, HaloScope 70.6; fully supervised TSV 85.5); TriviaQA 84.0; SciQ 85.8; NQ 76.1. N=32-64 "almost comparable to oracle", N=8 suboptimal. Pseudo-label accuracy falls 100→82% as K goes 32→512. Transfer TruthfulQA→TriviaQA 79.8 vs 84.0 in-domain. ~0.1 GPU-h at 8B | **the strongest evidence** that a few dozen labels plus a few hundred unlabeled samples nearly match full supervision for internals-based detection. Note: closed-book QA, not grounding. The prior over classes comes from the exemplars (MCAR assumption). Training needs backprop through the LLM (CPU-heavy at 4B); the *pseudo-labelling recipe* ports to our readout cheaply |
| **CCS** (Burns et al. 2022, arXiv 2212.03827) **[memory]** and **Farquhar et al. 2023** (<https://arxiv.org/abs/2312.10029>) **[abstract/secondary]** | unsupervised consistency probe | Farquhar et al. prove arbitrary features satisfy the CCS loss; the probe tracks the most *salient* feature | warns against trusting unlabeled-only directions. Our 2-axis structure (evidence vs contradiction) must be checked against labels, as it was |
| **RepE / LAT** (Zou et al. 2023, arXiv 2310.01405), **ITI** (Li et al. 2023, arXiv 2306.03341), ActAdd (Turner et al. 2023) **[memory]** | directions from contrast pairs | ITI used a few hundred labels | cheap "adaptation units". Mostly for steering generation, not scoring |
| **Universal truthfulness hyperplane** (Liu et al., EMNLP 2024; <https://arxiv.org/abs/2407.08582>) **[abstract/secondary]** | probe trained on 40+ datasets | ~70% cross-task accuracy vs ~50-56% for single-dataset probes; diversity matters more than volume (even ~10 samples per dataset) | supports "train the base probe on many diverse domains" over per-domain hypernets |
| **LLMs Know More Than They Show** (Orgad et al., ICLR 2025; <https://arxiv.org/abs/2410.02707>) **[abstract/secondary]** | probes at exact answer tokens | probes "do not generalize across tasks", only within similar skills | expect per-domain recalibration, always |
| **Broad detectors fail OOD** (Dubanowska et al., EMNLP Findings 2025; <https://arxiv.org/abs/2509.19372>) **[read]** | audit of ReDeEP, probes, SAPLMA on RAGTruth | RAGTruth has a spurious *task-type* correlation: "is it data-to-text" alone gives AUC 0.71 (LLaMA-2-7B). Per task, logistic regression gives QA 0.69, D2T 0.66, Summary 0.64. OOD RAGTruth↔SQuAD 0.51-0.55, "close to random" | **relevant to our numbers**: the pooled RAGTruth 0.82 in product.md may be inflated by the task mix. Our single-domain QA numbers (0.765) are the honest ones |
| **Semantic Entropy Probes** (Kossen et al. 2024; <https://arxiv.org/abs/2406.15927>) **[abstract/secondary]** | probe trained on a semantic-entropy target (no correctness labels) | generalises better OOD than accuracy probes (reported ΔAUROC 7.7-10.5, from a secondary source) | label-free *training targets* transfer better. Our construction-labelled probe is in this spirit |
| **PLLS-CP** (Tsvelenyev, Ryndin; NeurIPS 2026 poster; <https://openreview.net/forum?id=aKIJQBf4JP>) **[abstract only]** | pseudo-label layer selection, unsupervised detector; under shift **only the Mondrian conformal threshold is recalibrated on a small target set** | 18 domains; coverage "close to nominal"; <2.5% selective-accuracy loss; calibration size not stated | the same architecture we converged on: a fixed detector plus a per-domain conformal threshold. Independent confirmation of the design |

### 2.5 Test-time adaptation, and in-context-learned verifiers

**Test-time training.** Covered: TTT (Sun et al. 2020), TENT entropy minimisation (Wang et al. 2021, arXiv
2006.10726), TTT-on-nearest-neighbours for LLMs (Hardt & Sun 2023, arXiv 2305.18466). All **[memory]**.
- These need gradient steps through the model. On a 4-core CPU at 4B that costs minutes per sample. Not viable per
  request.
- The cheap analogues are feature re-standardisation on the domain (already shipped as "natural scales") and
  **CORAL** (Sun, Feng, Saenko 2016, arXiv 1511.05547), which matches second-order feature statistics before
  applying a linear probe.
- CORAL is the obvious untested upgrade for the neuron-level probe.

**In-context-learned classifier: TabPFN-2.5** (Prior Labs, Nov 2025; <https://arxiv.org/abs/2511.08667>)
**[abstract/secondary]**.
- A tabular foundation model doing in-context classification on ≤50k rows and ≤2k features, with a claimed 100%
  win rate vs default XGBoost on ≤10k-row sets.
- It is a genuine "learned learner" for a 20-300-label readout over our signals (plus, say, PCA-64 of activations).
- CPU speed for n≤300 should be fine (unverified).
- It does not touch the guarantee: the threshold must still come from held-out clean scores.

**LLM-as-in-context verifier** (few-shot judge prompts): too slow and costly per reply against our budget. Useful
only *offline*, to impute labels for PPI (2.6).

### 2.6 Statistics for a guarantee under shift, with few labels

| Tool | What it buys | Source |
|---|---|---|
| **Neyman-Pearson umbrella / order-statistic threshold** | distribution-free FAR control from clean-class scores only; minimum clean sample n₀ ≥ log δ / log(1−α) (= **29** for α=0.10, δ=0.05; 59 for α=0.05) | Tong, Feng, Li, Sci. Adv. 2018; Tong et al. JMLR 2020, <https://arxiv.org/abs/1802.02557> **[abstract/secondary]** |
| **Weighted conformal under covariate shift** | reweight calibration points by p_T(x)/p_S(x), estimated by a domain classifier on *unlabeled* target data; valid if p(y\|x) is unchanged | Tibshirani, Barber, Candès, Ramdas 2019, arXiv 1904.06019 **[memory]** |
| **PAC prediction sets under covariate shift** | the PAC (δ-confidence) version, with confidence intervals on estimated weights, via rejection sampling | Park, Dobriban, Lee, Bastani, ICLR 2022, <https://arxiv.org/abs/2106.09848> **[abstract/secondary]** |
| **Label-shift conformal; BBSE; EM prior correction** | prevalence from unlabeled target data via a confusion matrix (BBSE) or EM (Saerens et al. 2002); conformal sets reweighted by class ratio | Podkopaev & Ramdas, UAI 2021, <https://arxiv.org/abs/2103.03323> **[abstract/secondary]**; Lipton et al. 2018, arXiv 1802.03916 **[memory]** |
| **Prediction-powered inference, PPI++** | many unlabeled points with imputed labels plus few true labels → valid, tighter CIs. Labelled-sample need drops roughly by the R² of the imputer (secondary source); only helps if correlation > (n−2)^{−1/2} | Angelopoulos et al., Science 2023 (arXiv 2301.09633) **[memory]**; PPI++ <https://arxiv.org/abs/2311.01453> **[abstract/secondary]** |
| **Semi-supervised risk control via PPI** | tunes a risk-controlling threshold with imputed labels on unlabeled data, corrected by the few true labels; *valid even if the imputations are wrong*, lower variance when they are right | Einbinder, Ringel, Romano 2024, <https://arxiv.org/abs/2412.11174> **[abstract/secondary]** |
| **Conformal Alignment** | conformal selection plus Benjamini-Hochberg: FDR control over "trusted" outputs; ~500 reference samples suffice (secondary source) | Gui, Jin, Ren, NeurIPS 2024, <https://arxiv.org/abs/2405.10301> **[abstract/secondary]** |
| Learn-then-Test; conformal factuality | multi-threshold risk control; claim-level factuality guarantees | Angelopoulos et al. 2021 (arXiv 2110.01052); Mohri & Hashimoto 2024 (arXiv 2402.10978) **[memory]** |

**Three facts that organise the recipe.**

1. **FAR is class-conditional.** FAR(t) = P(s > t | clean). It is invariant to *label shift*: a change in
   hallucination prevalence with p(x | y) fixed. Precision is not invariant; it moves with prevalence. Only covariate
   or concept shift breaks a clean-calibrated threshold.
2. **Under covariate shift, the clean-class weight is estimable without labels.** If p(y | x) is fixed, then
   p_T(x | clean) / p_S(x | clean) ∝ p_T(x) / p_S(x). The marginal density ratio comes from a domain classifier on
   unlabeled replies. So weighted conformal *restricted to clean calibration points* is valid with unlabeled target
   data. This is a derivation, not a cited result; the effective sample size (Σw)²/Σw² shrinks.
3. **The binomial floor.** With n clean calibration replies, the best distribution-free 95%-confidence threshold for
   FAR ≤ 10% is an order statistic. I computed it exactly with Beta(k+1, n−k) tails:

| clean replies n | scores allowed above threshold k | expected FAR actually used | comment |
|---|---|---|---|
| <29 | none | none | **impossible** |
| 29-49 | 0 | ~3.3% | threshold = max clean score: low recall |
| 50 | 1 | 3.9% | |
| 100 | 4 | 5.0% | |
| 150 | 8 | 6.0% | ≈ our "300 balanced labels" case |
| 200 | 12 | 6.5% | |
| 300 | 21 | 7.3% | |
| 500 | 38 | 7.8% | |
| 1000 | 84 | 8.5% | |

The repo's label-budget table (100 balanced labels → 4.4% FAR, 22% caught) is exactly this floor: about 50 clean
replies at the k≈1 order statistic.
- **Recall is bought by clean labels, not total labels.**
- At a production prevalence of ~10-20% hallucinated, a random sample of ~35-60 replies gives ≥29 clean.
- A balanced 50/50 sample wastes half the labelling budget for the guarantee. Hallucinated labels still help, for
  measuring recall and refitting the readout.

---

## 3. What is genuinely usable (question 1)

1. **Writing the small readout from reasoning plus unlabeled statistics.** Already done; ties the default. Keep it
   as the *prior* for few-label refits (E3) and as a sanity anchor. Its ceiling is the information in the six
   signals.
2. **Directions from unlabeled contrasts**, from HaloScope, TSV and the Sparks Exp. 8 mechanism.
   - Use the unlabeled domain sample to find the domain's dominant "not drawn from evidence" direction.
   - Then use a handful of labels to *orient and validate* it.
   - The repo already found this axis; TSV's twist (OT pseudo-labels with a class prior, confident top-K, retrain)
     is the cheap next step.
3. **Diverse-domain training of the base probe** rather than per-domain generated probes. The universal-hyperplane
   result says diversity beats volume. PLLS-CP and our own results say: fix the detector, recalibrate the threshold.
4. **Not usable here:** LoRA hypernetworks (T2L, DnD, Doc-to-LoRA), ROME/MEMIT, Tracr/ALTA compilation, model
   merging beyond probe averaging, and Sparks-style agentic weight editing of the reader. Reasons: no frontier model
   with weight-level tool access locally; the target is generation quality, not calibration; and no FAR guarantee
   can come out of any of them.

---

## 4. Recipe: a new domain in minutes, 0-100 labels, guaranteed FAR (question 2)

Time is dominated by reader passes. On 4 CPU cores, measured at 1.4-6 s per reply, 300 unlabeled replies take
roughly 7-30 minutes. Everything after that is seconds.

**Step 0: no labels (minutes).**
- Read N≈200-300 unlabeled replies from the customer's real traffic. Cache the six signals and the raw probe
  activations.
- Standardise the signals on this sample (shipped). Optionally CORAL-align the neuron-level probe features (E8).
- Rank with the shipped or written readout. Expected AUROC is near the label-free level, ~0.75 on RAGTruth QA.
- Estimate hallucination prevalence by EM or BBSE on the readout score. This sets expected alert volume and
  precision, not FAR. Report it with a wide interval: our mixture prevalence ranged 39-74%, so treat it as weak.
- Optionally run an **advisory threshold** from a cross-domain prior (E7). Label it explicitly as *not guaranteed*.

**Step 1: label a random sample (the only step with a guarantee).**
- Draw replies *uniformly at random* from traffic, not by score, so the clean subset stays exchangeable with future
  clean replies.
- Label until there are ≥29 clean (FAR 10%, 95% confidence) or ≥59 clean (FAR 5%). Roughly 35-70 labels at typical
  prevalence.
- Optional shortcut, flagged: many customers hold *already-approved* historical replies, which are one-class "clean"
  labels.
  - They are only valid if approval was not selected on features that correlate with the score.
  - Otherwise they are not exchangeable and the guarantee is void.
  - Use them for the advisory threshold, not the guaranteed one.
- Threshold: order statistic or one-sided Clopper-Pearson on clean scores (`calibrate --confidence 0.95`, shipped).
- **Optionally, semi-supervised risk control** (Einbinder et al.): an offline judge (MiniCheck, or a bigger LLM run
  in batch) imputes labels on the unlabeled sample; the true labels correct the imputations.
  - This keeps validity and narrows variance, which raises recall at the same guarantee.
  - The gain scales with the imputer's agreement. MiniCheck's AUROC on RAGTruth was only ~0.65, so expect small
    savings (E5 measures it).

**Step 2: use hallucinated labels for ranking (sample-split).**
- Split the labelled set: fit on A, threshold on B. Alternatively, cross-fit, then set the final threshold on clean
  replies never used for fitting.
- Refit only the 6-weight readout: logistic regression with an L2 penalty *toward the written weights*, not toward
  zero. Augment with TSV-style OT pseudo-labels on the unlabeled sample (E3/E4).
- At 0-100 labels, do not refit the 10k-dimensional neuron probe. RAGTruth shows 66.5 F1 at 100 labels against
  72.8 at 800; per-domain gains need hundreds.

**Step 3: maintenance (0 new labels).**
- Monitor drift with a domain classifier between the calibration replies and recent traffic.
- On covariate drift, re-weight the clean calibration scores (weighted conformal, or PAC with weight intervals) and
  move the threshold.
- When the effective sample size falls below 29 clean, request a fresh random-sample top-up.
- Prevalence changes do not move FAR; re-estimate prevalence for precision.

**The sales unit becomes:** "≥29 (better ~150) of your replies that you confirm are clean, drawn at random, buys a
guaranteed FAR; more clean labels buy recall." This is sharper than "100-300 labels", and cheaper to collect.

---

## 5. Can a hypernetwork map a description or unlabeled samples to probe weights or calibration? (question 3)

**Evidence for.**
- DnD shows unlabeled prompts (~128) carry enough signal to generate useful LoRAs. Prompts alone beat
  prompts+answers on common sense.
- T2L shows text descriptions do the same, with +1.4 points over a multi-task adapter zero-shot.
- HaloScope and TSV show unlabeled in-domain generations contain a hallucination direction recoverable by SVD or
  OT. TSV: 32 labels + 128 pseudo-labelled ≈ oracle on TruthfulQA.
- Our written readout shows unlabeled statistics plus reasoning pin down a good 6-weight readout.

**Evidence against, or missing.**
1. **No published hypernetwork generates detector or probe weights or thresholds.** My searches for "probe
   hypernetwork" and "meta-learned hallucination probe" found nothing. The nearest is PLLS-CP, which deliberately
   keeps the detector fixed and recalibrates only the threshold.
2. Hypernetworks need many training tasks: T2L 479; universal hyperplane 40+ datasets; DnD fails with 2.
   - We can assemble perhaps 10-30 labelled domains: HaluBench's 6 sources, RAGTruth's 3 tasks × generators,
     LLM-AggreFact's ~11 datasets, plus synthetic grounding sets.
   - That is enough for leave-one-domain-out *meta-regression*, not a neural hypernet.
3. Cross-domain transfer of internals-based detectors is weak: Orgad et al.; Dubanowska et al. at 0.51-0.55 OOD.
   Our own threshold transfer failed at 4-32% per source. A mapping learned across domains inherits that variance.
4. **For calibration it cannot give a guarantee.** A predicted threshold is a point estimate whose error depends on
   how unlike the training domains the new one is. Our label-free mixture missed by 1-36 points. Any guarantee
   still needs ≥29 exchangeable clean labels.
   - A predicted threshold *can* serve as a prior, for example a Bayesian or hierarchical shrinkage of the clean
     quantile. That improves the *expected* FAR and recall at small n, without a distribution-free guarantee.
5. Domain *descriptions* (T2L-style) are a weaker signal than unlabeled samples. T2L degrades badly with misaligned
   descriptions, and our domains differ in score distributions that text descriptions don't capture.

**Verdict.** Only a "lite" version is worth trying (E7), and the expected gain is modest. The candidate maps the
unlabeled domain's signal statistics (means, covariances, score quantiles, estimated prevalence) to (a) readout
weights and (b) the clean-score 90th percentile, by ridge regression across domains with leave-one-domain-out
evaluation. If it beats "shipped weights + mixture threshold", it becomes the Step-0 advisory threshold and the prior
for Step 2. Do not build a neural hypernetwork until there are ≥50 labelled domains.

---

## 6. CPU-feasible experiments, ranked by expected value (question 4)

Cached signals exist for 300 unlabeled and 300 labelled test RAGTruth QA replies (`results/written_readout/*.jsonl`).
Per-reply signals for the 300 calibration replies (`results/pipeline_demo/ragtruth_qa_cal300.jsonl`) do not appear to
be cached. The `label_budget` code is not in `experiments/written_readout.py` as committed. Re-reading 300 replies is
~10-30 min on CPU.

| Rank | Experiment | Cost (CPU) | Why / expected outcome | Decision it informs |
|---|---|---|---|---|
| **E1** | **Clean-count label budget and random-sample protocol.** Re-run the label-budget simulation by *number of clean replies*, sampled at a realistic prevalence (10%, 20%, 50%). Check the empirical FAR and recall against the binomial-floor table above. Also test "approved-clean only" (clean replies selected by a proxy) to show how a non-random selection breaks the guarantee | minutes (cached; re-read cal300 once) | high certainty of a clean, sellable result: "29 clean → guaranteed 10% FAR; 150 clean → ~6% effective FAR, recall X%". Likely halves the quoted label cost at low prevalence | pricing and the onboarding protocol |
| **E2** | **Weighted conformal against the RAGTruth train→test shift.** Fit a domain classifier (logistic on reply-level signal aggregates, or on the frozen probe's PCA features) between cal300 and *unlabeled* test replies. Weight the clean calibration scores. Compare plain quantile (16% FAR), weighted quantile, CP bound, and weighted CP/PAC (Park et al.) | ~30 min (re-read) + seconds | the plain quantile's overshoot was attributed to train/test shift. If weighting brings it to ~10% *without labels*, the drift-maintenance step is validated. Risk: shift may be concept shift (p(y\|x) changes), where weighting cannot help; that result is informative too | whether Step 3 can be label-free |
| **E3** | **Few-label readout refit with shrinkage toward the written weights.** k ∈ {10, 20, 50, 100} labels, 300 draws each; L2-toward-prior logistic on 6 signals vs from-scratch logistic vs fixed written vs fixed shipped. Threshold on held-out clean replies; report AUROC and recall at guaranteed FAR | seconds (cached) | expect a small AUROC gain (+0.01-0.04) at 50-100 labels and no loss at 10. Our 6 signals cap the gain | whether Step 2 is worth shipping |
| **E4** | **TSV/HaloScope pseudo-labelling on the readout.** Seed with k labels (or the written readout for k=0), Sinkhorn OT with the class prior (from labels, or EM), take confident top-K of the 300 unlabeled, refit, iterate. Compare with E3 | seconds | TSV reports near-oracle from 32-64 labels, but with 4k-param steering of an 8B model on closed-book QA. On 6 signals the headroom is smaller. Moderate chance of +0.02 | label-free or few-label ranking upgrade |
| **E5** | **Semi-supervised (PPI) threshold** with MiniCheck imputations (`results/pipeline_demo/minicheck_test300.json` exists for test; run MiniCheck on unlabeled, ~2.4 s/reply) and/or Qwen3-4B as an offline judge. Measure the variance reduction of the FAR estimate and the recall gain at a held FAR for k=30-100 | ~15-30 min | gain ∝ imputer quality. MiniCheck is weak on RAGTruth (0.65), so expect ≤20% label savings, but the method is valid regardless | whether to offer "judge-assisted calibration" |
| **E6** | **TabPFN-2.5 as an in-context readout** on 6 signals (+ PCA-32/64 of probe activations if cached) at k=20-300 labels vs logistic | minutes (pip install; CPU inference at n≤300) | with 6 features the likely result is no gain; with PCA activations at 200-300 labels it might reach the supervised probe's regime faster. Licence check needed for commercial use | the few-hundred-label tier |
| **E7** | **"Hypernetwork-lite" meta-regression, leave-one-domain-out.** Cache signals for ~300 replies in each of ~10-15 domains (HaluBench 6 sources, RAGTruth QA/Summary/D2T, 2-3 LLM-AggreFact sets). Map unlabeled stats → (readout weights, clean q90) by ridge. Score the held-out domain on AUROC, \|FAR − 10%\| of the advisory threshold, and as a prior for E3 | ~4-8 h CPU of reader passes (background), then seconds | the only direct test of question 3. Expect advisory FAR errors to shrink from the mixture's 1-36 points to perhaps 3-10 points (a guess). Ranking gains are unlikely. Also gives a "probe soup" baseline | whether to invest in a domain-to-calibration model |
| **E8** | **CORAL / whitening transfer of the neuron-level probe:** re-colour source activations to target covariance, or whiten both, before applying the probe weights | needs raw activations cached for 300+300 replies: ~30-60 min + seconds | cheap unsupervised domain adaptation for the 10k-dim probe. Plausible +0.01-0.03; may also fix scale drift | Step 0 upgrade |
| E9 (deprioritised) | TSV proper: train a residual-stream vector through Qwen3-4B on CPU | hours per domain | too slow for "minutes" on CPU; revisit on GPU | none now |
| not recommended | T2L/DnD LoRA generation; ROME/MEMIT; Tracr/ALTA; task-vector merging; Sparks-style agentic weight editing | n/a | wrong target (generation or knowledge), needs GPUs or frontier models, no guarantee | none |

**Suggested order:** E1 and E3 (an afternoon, cached data), then E2 (needs the cal300 re-read, which also enables
E5), then E4/E5, then E7 in the background.

---

## 7. Uncertainty and caveats

- **Sparks** is a non-peer-reviewed blog post. It uses one model (GPT-6 Astra), mostly single runs, weak baselines
  (the authors' own PPO), and no direction-ablation baseline for the headline Qwen result. Secondary coverage
  differs on details (for example the 25-pass prompt-ablation number). The table above follows the primary page.
- **DnD's "up to 30%"** is relative to LoRAs trained on *other* datasets, a weak reference. The paper is also
  internally inconsistent on prompt batch sizes (section 3.1 vs table 7).
- **TSV** numbers are for closed-book QA with 7-8B models and backprop through the LLM. Transfer to grounding
  verification on a 4B reader with only six signals is unproven.
- **HaloScope, SEP, universal-hyperplane, PPI++, Conformal-Alignment** numbers came from abstracts or secondary
  summaries. The SEP ΔAUROC 7.7-10.5 and Conformal-Alignment "~500 samples" come from a third-party review.
- **The PLLS-CP** abstract gives no calibration-set size or FAR numbers.
- **The "clean-only weights from marginal density ratio" claim** (section 2.6, fact 2) is my derivation under strict
  covariate shift. It fails under concept shift, and the estimated weights add their own error. That is why the PAC
  variant with weight intervals exists.
- **The binomial-floor table** is exact for exchangeable clean samples and the 95%/10% setting. The "expected FAR
  actually used" column is the mean of the Beta distribution, not a realised rate.
- **The pooled RAGTruth AUROC** (0.82 in product.md) may be inflated by the task-type confound Dubanowska et al.
  identify. Single-task numbers are safer to quote.
- **[memory]** references (arXiv IDs for task arithmetic, TIES, DARE, ROME, MEMIT, Tracr, LTP, RepE, ITI, CCS,
  TENT, CORAL, weighted conformal, BBSE, PPI, LTT, conformal factuality, Generative Adapter, TTT-NN) were not
  re-fetched. Verify them before external use.

## 8. Sources (fetched or searched this session)

- Sparks of Weight-Space Programming, primary blog: <https://rikiyat.github.io/blog/weight-space-programming/>; logs <https://github.com/RikiyaT/lm_understand-logs>
- Maier critique: <https://akmaier.substack.com/p/weight-space-programming-what-is>
- Turing Post explainer: <https://www.turingpost.com/p/what-is-weight-space-programming-and-why-it-suddenly-blew-up-the-internet>
- Text-to-LoRA: <https://arxiv.org/abs/2506.06105> (HTML v2 read); ICML: <https://proceedings.mlr.press/v267/charakorn25a.html>
- Doc-to-LoRA: <https://arxiv.org/abs/2602.15902>, <https://sakana.ai/doc-to-lora/>, <https://proceedings.mlr.press/v306/charakorn26a.html>
- Drag-and-Drop LLMs: <https://arxiv.org/abs/2506.16406> (HTML read)
- HyperTuning: <https://arxiv.org/abs/2211.12485>; HyperLlama/gisting: <https://arxiv.org/abs/2402.16817>
- ALTA: <https://arxiv.org/abs/2410.18077>
- HaloScope: <https://arxiv.org/abs/2409.17504>
- TSV: <https://arxiv.org/abs/2503.01917> (HTML v2 read); <https://icml.cc/virtual/2025/poster/45122>
- Farquhar et al., unsupervised knowledge discovery: <https://arxiv.org/abs/2312.10029>
- Universal truthfulness hyperplane: <https://arxiv.org/abs/2407.08582>
- LLMs Know More Than They Show: <https://arxiv.org/abs/2410.02707>
- Broad detectors fail OOD: <https://arxiv.org/abs/2509.19372> (HTML read)
- Semantic Entropy Probes: <https://arxiv.org/abs/2406.15927>
- PLLS-CP: <https://neurips.cc/virtual/2026/poster/151476>, <https://openreview.net/forum?id=aKIJQBf4JP>
- TabPFN-2.5: <https://arxiv.org/abs/2511.08667>, <https://priorlabs.ai/technical-reports/tabpfn-2-5-model-report>
- NP classification sample size: <https://arxiv.org/abs/1802.02557>, <https://jmlr.org/papers/v21/18-577.html>
- PAC prediction sets under covariate shift: <https://arxiv.org/abs/2106.09848>
- Label-shift conformal: <https://arxiv.org/abs/2103.03323>
- PPI++: <https://arxiv.org/abs/2311.01453>; PPI non-asymptotic analysis: <https://arxiv.org/abs/2505.20178>; glide-py ESS tutorial: <https://glide-py.readthedocs.io/en/latest/tutorials/ppi/>
- Semi-supervised risk control via PPI: <https://arxiv.org/abs/2412.11174>
- Conformal Alignment: <https://arxiv.org/abs/2405.10301>
