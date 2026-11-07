# Hallucination / correctness detection from model internal states (2023 to 2026): a research review for our verifier

Scope: white-box and gray-box detectors that read hidden states, attention, or logits. For each family: the signal,
the models, the benchmarks, the numbers, the inference cost, whether it needs the generator's weights, the labels
it needs, and above all the evidence on generalization across domains. The review ends with what this means for our
frozen-reader verifier (Qwen3-4B; label-free AUROC 0.62-0.77, 0.86 with 800 RAGTruth labels) and a ranked list
of CPU experiments.

**How the numbers were checked.** Numbers marked ✔ were read from the paper's own text or tables (arXiv HTML/PDF)
during this review. Numbers marked ~ are from secondary summaries, from my memory of the paper, or from tables I
could not fully read. Treat ~ numbers as approximate. AUROC is written as 0.xx throughout; a paper that reports
percent is converted.

---

## 1. Executive summary

1. **The signal is real and mostly linear.** Every serious study finds that a linear probe on mid-layer hidden
   states separates right from wrong outputs well *in distribution*, with AUROC 0.80-0.95. "The Hallucination Signal
   Is a Mean Shift" (EMNLP 2026) finds the signal is dominated by one mean-shift direction: removing it drops
   detection to chance, and an L2 logistic regression matches 12 fancier architectures (0.952 AUROC, paired
   setting). Fancy probe architectures do not buy robustness (ProbeDrift, 2,000+ probes).
2. **Out-of-distribution transfer is the central problem, and it is worse than most papers report.**
   - Dubanowska et al. (EMNLP Findings 2025): RAGTruth→SQuAD AUROC 0.48-0.55 for every method (ReDeEP, SAPLMA,
     logistic regression, SAE features). Cross-task within RAGTruth is near chance. RAGTruth itself carries a
     task-type confound: "is it data-to-text?" alone gets AUROC 0.71, level with ReDeEP. ✔
   - Orgad et al. (ICLR 2025): once the logit baseline is subtracted, probe transfer "rarely exceeds" what logits
     alone give.
   - ProbeDrift (Stacey, Orgad et al., EMNLP 2026): probes hold up under near-OOD shift and degrade beyond it.
3. **The design choices that do generalize better,** all of which have little visible effect in distribution:
   - **(a) Middle-layer features, or a band of layers, not the last layer** (ProbeDrift; SALA's "layer-index shift";
     LayerMix; CLAP).
   - **(b) Mean-pooling over the response tokens, not the last token** (ProbeDrift). For streaming, a max over
     token scores (Obeso et al.).
   - **(c) Training on a model-internal target rather than on correctness labels.** Semantic-entropy probes beat
     accuracy probes out of distribution by +7.7 to +10.5 AUROC points on short-form answers ✔.
   - **(d) Diverse training data, which matters more than volume** (Universal Truthfulness Hyperplane; RAGLens:
     trained on RAGTruth it transfers to AggreFact at 0.80 and TofuEval at 0.76, while single-task training does
     not ✔).
   - **(e) Span-level supervision on long-form text.** It transfers down to short-form QA and math; short-form
     training does not transfer up to long-form (Obeso et al.).
4. **Reading another model's text with a separate "reader" works, which validates our architecture.**
   - Obeso et al.: probes trained on one model's annotations work on other models' outputs, off-diagonal AUROC
     within 0.02-0.04 ✔.
   - Lookback Lens: transfers 7B→13B, losing about 6-10 points ✔.
   - Proxy analyzers (arXiv 2605.07209): a 0.5B open model reading other models' RAGTruth answers reaches AUROC
     0.825 against ReDeEP's 0.73. That probe is supervised on the RAGTruth training set ✔.
   - "External observers may see more clearly" (Oct 2026): a smaller observer can match a generator's
     self-detection.
5. **Calibration transfers worse than ranking.**
   - Reasoning-model probes: ECE below about 0.1 in distribution, but 0.25-0.28 across datasets, while AUROC stays
     at 0.69-0.83 ✔.
   - This matches our own finding. No paper shows a label-free threshold that holds a false-alarm rate across
     domains. **Per-domain conformal calibration remains necessary.**
6. **The most label-efficient recipes:**
   - TSV (ICML 2025): 32-64 labels plus optimal-transport pseudo-labels on unlabeled generations reach 0.80-0.87
     AUROC, beating fully supervised SAPLMA ✔.
   - Simple difference-of-means or shrinkage-LDA directions, whose covariance can come from unlabeled data.
   - The hybrid back-off (HBO) recipe of ProbeDrift: fall back to a label-free score when the input is far from
     the training data (Mahalanobis distance).
7. **What most plausibly moves us above 0.86 supervised and 0.77 label-free.** In rough order:
   - Lightly fine-tune the reader (LoRA probes add +0.04 AUROC over linear probes in Obeso et al. ✔).
   - Use token- or span-level supervision with max aggregation instead of sentence means.
   - Pool a band of mid layers.
   - Use much more and more diverse public training data (+~0.02 AUROC per doubling on our learning curve).
   - Fit per-domain unsupervised centering and whitening before the probe.
   - Stack with a cheap NLI or HHEM feature (as the proxy-analyzer paper does).

---

## 2. Methods, one by one

### 2.1 Early probes and truth directions (2023-2024)

| method | signal | models | benchmarks | headline numbers | generator weights? | labels | OOD evidence |
|---|---|---|---|---|---|---|---|
| **SAPLMA** (Azaria & Mitchell, Findings EMNLP 2023, [arXiv 2304.13734](https://arxiv.org/abs/2304.13734)) | MLP on one layer's hidden state at the last token of a statement | OPT-6.7B, Llama-2-7B | True-False dataset (6 topics) | ~0.70-0.85 accuracy on held-out topics, well above P(true) ~ | reads the statement; works with any reader on given text | full labels | held-out *topic* only. Later studies find it the **most robust supervised baseline** (ProbeDrift; OpenHalDet), but near chance RAGTruth→SQuAD (0.55 ✔) |
| **ITI** (Li et al., NeurIPS 2023, [2306.03341](https://arxiv.org/abs/2306.03341)) | linear probes per attention head; steer along truthful directions | LLaMA-7B, Alpaca, Vicuna | TruthfulQA | best heads ~0.83 probe accuracy; Alpaca TruthfulQA truthfulness ~32.5%→65.1% ~ | yes for steering; probes can be applied to any text | labels | as a RAGTruth detector, 0.65-0.81 AUROC in distribution and 0.50-0.55 on Dolly ✔ (ReDeEP table) |
| **CCS** (Burns et al., ICLR 2023, [2212.03827](https://arxiv.org/abs/2212.03827)) | unsupervised direction from contrast pairs (consistency loss) | T5, DeBERTa, GPT-J etc. | 10 classification QA sets | ~+4% over zero-shot accuracy ~ | any reader | none | weak in practice: 0.51-0.77 on QA (TSV table ✔); "Challenges with unsupervised LLM knowledge discovery" (Farquhar et al. 2023) shows it latches onto arbitrary salient features |
| **Geometry of Truth** (Marks & Tegmark, COLM 2024, [2310.06824](https://arxiv.org/abs/2310.06824)) | difference-of-means ("mass-mean") direction on curated true/false statements | LLaMA-2 7B/13B/70B | curated factual statements | mass-mean probes generalize as well as logistic regression and are more causal; transfer is good between simple statement sets ~ | any reader on statements | labels, few needed | transfer shown on clean statement datasets, not on free generations |
| **Truth is Universal / TTPD** (Bürger et al., NeurIPS 2024, [2407.12831](https://arxiv.org/abs/2407.12831)) | 2-D truth subspace: a general direction plus a polarity direction (earlier probes failed on negations) | Llama-3-8B, Llama-2-13B, Gemma-7B, … | statements; real-world lies | ~94% on simple statements and real-world lie scenarios ~ | any reader | labels | strong on statement-style inputs; **only 0.58-0.68 step-level AUROC** on CoT reasoning steps (Lu et al. 2026 ✔) |
| **Universal Truthfulness Hyperplane** (Liu et al., EMNLP 2024, [2407.08582](https://arxiv.org/abs/2407.08582)) | logistic or mass-mean probe trained on **40+ datasets / 17 task types** | Llama-2-7B-chat, Mistral-7B, Llama-2-13B | held-out tasks | cross-task ~70% accuracy (Mistral ~77%, Llama-2-13B ~74%) against near-chance for single-dataset probes ~ | any reader | labels from many datasets | **diversity of training tasks matters more than examples per dataset** ~ |
| **INSIDE / EigenScore** (Chen et al., ICLR 2024, [2402.03744](https://arxiv.org/abs/2402.03744)) | log-determinant of the covariance of K sampled answers' mid-layer embeddings | LLaMA-7B/13B, OPT-6.7B | CoQA, SQuAD, NQ, TriviaQA | CoQA ~0.77, SQuAD ~0.81 (one model row) ~ | yes (needs sampling from the generator) | none | weak on independent benchmarks: 0.45-0.69 (TSV ✔), 0.50-0.72 (OpenHalDet ✔), 0.60-0.66 on RAGTruth ✔. **Costs K extra generations** |
| **Semantic Entropy Probes** (Kossen et al., 2024, [2406.15927](https://arxiv.org/abs/2406.15927)) | linear probe on hidden states predicting *binarized semantic entropy* (computed offline from 10 samples), not correctness | Llama-2 7B/70B, Llama-3-70B, Mistral-7B, Phi-3-3.8B | TriviaQA, SQuAD, BioASQ, NQ-Open; short and long form | in distribution, SEP − accuracy probe = −2.0 to +2.8 AUROC points; **out of distribution (leave-one-dataset-out), +7.7 to +10.5 short-form, +2.2 to +6.2 long-form** ✔. Below sampling-based SE (which costs 10×) | generator states (the target is the generator's own SE) | **no correctness labels**; needs about 10 samples per training question | best documented label-free-target result; the probe's prediction also follows SE when context is added (it was never trained with context) ✔ |
| **Lookback Lens** (Chuang et al., EMNLP 2024, [2407.07071](https://arxiv.org/abs/2407.07071)) | per head, attention to context ÷ (context + generated); logistic regression; 1,024 features for 7B | Llama-2-7B/13B-chat | CNN/DM, NQ (contextual hallucination) | in-domain test 0.89-0.91; **QA→Summ 0.853, Summ→QA 0.820** against 0.79-0.85 for hidden states; sliding-window spans 0.66 ✔ | the reader can be any model; transfer 7B→13B: 0.735 / 0.782 against 0.840 / 0.843 same-model ✔ | 1-2.5k labeled spans | its cross-task transfer is good, but in ProbeDrift it is the weakest stand-alone feature (PRR 0.04 OOD ✔) |
| **CH-Wang et al.** "Do Androids Know…" (Findings ACL 2024, [2312.17249](https://arxiv.org/abs/2312.17249)) | span-level probes on states during grounded generation | Llama-2 family | summarization, knowledge-grounded dialogue, data-to-text | probes match or beat an expert annotator on 2 of 3 tasks ~ | generator | 15k annotated utterances | **poor cross-task transfer; probes trained on synthetic (edited) hallucinations underperform organic ones** ✔ (abstract) |
| **HaloScope** (Du et al., NeurIPS 2024, [2409.17504](https://arxiv.org/abs/2409.17504)) | SVD "hallucination subspace" of *unlabeled* generations → pseudo-labels → classifier | Llama-2-7B/13B, OPT | TruthfulQA, TriviaQA, CoQA, TydiQA | ~0.79 on TruthfulQA (own paper) ~; independently 0.63-0.81 (TSV ✔), 0.51-0.70 (OpenHalDet ✔) | generator | none (unlabeled only) | independent reruns put it well below supervised probes |

### 2.2 Answer-token and correctness probes on the generator (2024-2025)

| method | signal | models | benchmarks | numbers | labels | OOD evidence |
|---|---|---|---|---|---|---|
| **"LLMs Know More Than They Show"** (Orgad et al., ICLR 2025, [2410.02707](https://arxiv.org/abs/2410.02707)) | logistic probe at the **exact answer tokens** (e.g. "Hartford"), not the last token | Mistral-7B(-Instruct), Llama-3-8B(-Instruct) | 10 datasets: TriviaQA, HotpotQA ± context, NQ, Winobias, Winogrande, MNLI, Math, IMDB, Movies | Mistral-Instr: TriviaQA 0.85 against 0.75 logits-min; Winobias 0.92; Math 0.92. Llama-3-Instr: 0.83 / 0.93 / 0.95. Last-token probe 0.71-0.82 ✔ | full labels | **transfer only between tasks needing the same skill** (factual retrieval ↔ factual retrieval); after subtracting logit-based detection, the gain from transfer "rarely exceeds" logits ✔. Models often *encode* the right answer yet *generate* a wrong one |
| **Reasoning models know when they're right** (Zhang et al., 2025, [2504.05419](https://arxiv.org/abs/2504.05419)) | probe (mostly linear) on the hidden state at the end of each intermediate answer in a CoT | R1-Distill Qwen 1.5B/7B/32B, Llama 8B/70B, QwQ-32B | GSM8K, MATH, AIME, KnowLogic | in distribution AUROC above 0.7 everywhere, above 0.9 for 32B on AIME; ECE mostly below 0.1. Cross-dataset (Llama-8B): GSM8K→MATH 0.80, MATH→GSM8K 0.83, MATH→AIME 0.76 (**ECE 0.28**), logic→math 0.74-0.77 ✔. Probe-based early exit saves 24% of tokens | ≤1,000 per dataset | ranking transfers, **calibration does not**; non-reasoning Llama-3.1-8B probes are clearly worse |
| **SAPLMA/SEP and others in large re-evaluations** | | | | see §3 | | |

### 2.3 RAG and contextual faithfulness from internals

| method | signal | models | numbers (RAGTruth unless stated) | labels | OOD evidence |
|---|---|---|---|---|---|
| **ReDeEP** (Sun et al., ICLR 2025, [2410.11414](https://arxiv.org/abs/2410.11414)) | external-context score (copying heads' attention to the context) + parametric-knowledge score (knowledge FFNs' contribution, via the logit lens) as regression covariates | Llama-2-7B/13B, Llama-3-8B (generator's own states) | AUROC 0.733-0.825 (token/chunk) against SAPLMA 0.70-0.80, SEP 0.70-0.81; Dolly 0.67-0.84 ✔ | hyperparameters and regression weights fit on labeled validation data | Dubanowska et al.: Dolly-tuned→RAGTruth 0.72, but **RAGTruth-tuned→Dolly 0.50** (Llama-3-8B); RAGTruth→SQuAD 0.52 ✔ |
| **RAGLens** (Xiong et al., ICLR 2026, [2512.08892](https://arxiv.org/abs/2512.08892)) | SAE features of mid-layer activations, pooled by max over tokens, selected by mutual information, fed to a GAM | Llama-2-7B/13B; also Llama-3.2-1B, Llama-3.1-8B, **Qwen3-0.6B/4B** | Llama-2-7B 0.841 (ReDeEP 0.746); Llama-2-13B 0.896; Dolly 0.857-0.876 ✔ | RAGTruth training labels | **trained on RAGTruth → AggreFact 0.80, TofuEval 0.76**; AggreFact-trained → RAGTruth 0.53. Summary-trained → QA 0.83, → Data2txt 0.64 ✔. Cross-model use works; **Qwen3-0.6B's detector lags the larger ones** |
| **Proxy analyzers** ([2605.07209](https://arxiv.org/abs/2605.07209), 2026) | 18 signals from a small open model *reading* finished text: per-head source attention (most important), attention entropy, residual and MLP norms, logit-lens slopes, grounding statistics, plus an HHEM-2.1 score; stacking ensemble | Qwen2.5-0.5B … Gemma-2-9B as readers | RAGTruth AUROC 0.814-0.837 across 7 readers; **Qwen2.5-0.5B 0.825** ✔ | 72k mixed labels (HaluEval, RAGTruth, MedHallu, MiniCheck-synthetic, ANLI) | LLM-AggreFact balanced accuracy 0.67-0.70, below MiniCheck-FT5's 0.75 ✔. Closest published analogue of our design |
| **Lookback Lens** | see §2.1 | | | | |
| **Dubanowska et al. "Broad hallucination detectors fail OOD"** (Findings EMNLP 2025, [2509.19372](https://arxiv.org/abs/2509.19372)) | evaluation study | Llama-2-7B/13B, Llama-3-8B, Phi-3.5, Mistral-7B | Llama-2-7B in distribution: **logistic regression 0.795**, SAPLMA 0.749, ReDeEP 0.732, SAE 0.711. The "is it D2T" heuristic alone gets 0.712; task type is decodable with AUROC 1.0 ✔ | | RAGTruth→SQuAD: all methods 0.48-0.55. Cross-task (QA→D2T) 0.56 ✔ |
| Others not read in detail: CORTEX ([2606.31033](https://arxiv.org/abs/2606.31033), contrasts internal representations with and without context at token level, which is close to our two-pass evidence shift), TPA ([2512.07515](https://arxiv.org/abs/2512.07515), next-token probability attribution), RAGognizer ([2604.15945](https://arxiv.org/abs/2604.15945), fine-tunes a detection head into the generator) | | | numbers not verified | | |

### 2.4 Attention- and dynamics-based detectors (mostly label-free)

| method | signal | numbers | cost | notes |
|---|---|---|---|---|
| **RAUQ** (Vazhentsev et al., ICML 2026, [2505.20045](https://arxiv.org/abs/2505.20045)) | per layer, picks the head with the most attention to the *previous token* (no labels); token confidence = blend of probability and that attention, recurrent; max over middle layers | 4 LLMs (Llama-3.1-8B, Qwen-2.5-7B, Gemma-2-9B, Falcon-3-10B), 12 tasks (QA, summarization, MT): mean PRR 0.414 against MSP 0.307 and SE 0.251; **mean ROC-AUC 0.692 against MSP 0.655** ✔ | +0.3% latency ✔ | the best *unsupervised* single-pass gray/white-box score found. Needs the generator (or the reader's attention on the text) |
| **Uncertainty heads** (Shelmanov et al., EMNLP 2025, [2505.08200](https://arxiv.org/abs/2505.08200)) | small transformer over attention maps + top-k probabilities, supervised on claim labels from an LLM judge | claim-level SOTA in and out of domain, cross-lingual transfer (EN→DE/ZH) ~; pre-trained heads released for Mistral/Llama/Gemma | single pass | ProbeDrift counts it among methods that are robust only to near-OOD shifts |
| **HalluShift** (Dasgupta et al., IJCNN 2025, [2504.09482](https://arxiv.org/abs/2504.09482)) | Wasserstein and cosine shifts between consecutive layers' hidden states + token-probability features → small classifier | TruthfulQA ~0.90, TyDiQA ~0.88 (aggregator; model unclear) ~ | single pass | CrossHallu (2026, [2607.04029](https://arxiv.org/abs/2607.04029)) reuses these features: in-domain 0.57-0.83, cross-lingual 0.40-0.79, cross-domain 0.45-0.87 ✔. Cosine shift is the useful part, Wasserstein the weak part |
| **ICR Probe** (Zhang et al., ACL 2025, [2507.16488](https://arxiv.org/abs/2507.16488)) | how much attention vs FFN update the residual stream at each layer → small classifier | strongest non-SpikeScore baseline in SpikeScore's cross-domain table ~ | single pass | |
| **LLM-Check** (Sriramanan et al., NeurIPS 2024) | eigenvalues of attention and hidden covariance within one response | 0.57-0.66 prefix-level on CoT (Lu et al. ✔) | single pass | |
| **CLAP** (2025, [2509.09700](https://arxiv.org/abs/2509.09700)) | attention *across layers* (each layer's activation is a token) | ~+6.5 AUROC over a last-layer probe OOD on Llama-3-8B ~ | single pass | supports "use many layers" |
| **SpikeScore** (Deng et al., ICLR 2026, [2601.19245](https://arxiv.org/abs/2601.19245)) | generate follow-up dialogue turns after the answer; score = maximum second difference of a probe's score along the turns | beats PRISM and the ICR Probe in leave-one-domain-out over 6 datasets × 4 LLMs (Llama-3.2-3B … Qwen3-14B); separability over the ICR Probe +0.015 to +0.082 ✔ | **several extra generated turns** (expensive) | "the instability of the probe score along the continuation" is a domain-invariant signal; it could be emulated cheaply by a reader |
| **PRISM** (Zhang et al., ACL 2025, [2411.04847](https://arxiv.org/abs/2411.04847)) | wrap the text in a prompt ("Is this statement true?") before reading states → probe | OpenHalDet: QA 0.82, RAG 0.80 (Llama-3.1-8B) ✔ | single pass | **directly applicable to a reader: the probing prompt is a free knob** |
| **SALA / layer-index shift** (Li et al., 2026, [OPUS](https://opus.lib.uts.edu.au/handle/10453/197183)) | the best layer moves between domains; weight layers by how stable they are across training domains | best average on unseen domains (4 QA sets, Llama/Qwen 7B-14B) ~ | single pass | |
| **FEPoID layer selection** (Wang et al., ICML 2026, [2605.26366](https://arxiv.org/abs/2605.26366)) | pick the layer at the first peak of intrinsic dimension (TwoNN), label-free | ~ | negligible | plus a token-truncation trick that "substantially" helps ~ |

### 2.5 Long-form, streaming, and reasoning traces

| method | signal | models | numbers | labels | generalization |
|---|---|---|---|---|---|
| **Real-time detection of hallucinated entities** (Obeso, Arditi, Ferrando, …, Nanda, 2025, [2509.03531](https://arxiv.org/abs/2509.03531)) | **token-level** linear probe (or LoRA + head) at layer ~0.95·L; span-weighted loss; response score = max over tokens | Llama-3.1-8B, Llama-3.3-70B; also Qwen-2.5-7B, Gemma-2-9B, Mistral-24B | LongFact (Llama-8B / 70B): linear **0.854 / 0.867**, LoRA **0.894 / 0.905**, against perplexity 0.76 / 0.71 and semantic entropy 0.72 / 0.71. HealthBench (held out): linear 0.856 / 0.873, LoRA 0.896 / 0.906. TriviaQA: linear 0.918 / 0.948. MATH (held out, never trained on math): linear 0.845 / 0.864. Qwen-2.5-7B LongFact: linear 0.838, LoRA 0.895 ✔ | ~25k web-search-verified annotations (Claude + search); labels agree with humans 84% | **long→short transfer good; short→long fails.** Entity probes flag wrong math answers. **Annotations from one model train probes for another**, off-diagonal within 0.02-0.04 ✔. Training a LoRA probe takes under 2 h on one H100 |
| **Streaming hallucination detection in long CoT** (Lu et al., 2026, [2601.02170](https://arxiv.org/abs/2601.02170)) | per-step probe on exponentially time-weighted step states + prefix-level accumulator | Llama-3.1-8B, Qwen2.5-7B, R1-Distill-Llama-8B | step-level AUROC 0.878 / 0.867 / 0.933 against SAPLMA 0.828 / 0.820 / 0.909 and TTPD 0.58-0.68. Prefix-level 0.73 / 0.81 / 0.92 ✔ | 200k steps labeled by Claude 4.5 | cross-dataset not visible in the part read |
| **Reasoning models know when they're right** | see §2.2 | | | | |
| ReProbe ([2511.06209](https://arxiv.org/abs/2511.06209)); transport-geometry step detection ([2605.13772](https://arxiv.org/abs/2605.13772)); IRIS ([2509.10004](https://arxiv.org/abs/2509.10004)) | step probes; label-free teacher-student; probe trained on the model's own verbalized-confidence pseudo-labels | | IRIS (accuracy, Llama-3.1-8B): True-False 90.4 / HaluEval2 70.6 / HELM 68.4, against **supervised SAPLMA 89.7 / 73.3 / 73.6**; cross-topic loss about 3 points ✔ | IRIS: none | IRIS supports "a self-supervised target gets about 95% of supervised performance" |

### 2.6 Label-efficient and unsupervised probes

| method | idea | numbers | labels |
|---|---|---|---|
| **TSV** (Park, Du, …, Li, ICML 2025, [2503.01917](https://arxiv.org/abs/2503.01917)) | learn a steering vector added to the residual stream so that truthful and hallucinated generations separate (prototype loss); pseudo-label unlabeled generations by optimal transport + confidence filtering; retrain | Llama-3.1-8B TruthfulQA / TriviaQA / SciQ / NQ: **0.842 / 0.840 / 0.858 / 0.761** against SAPLMA (fully supervised) 0.782 / 0.837 / 0.773 / 0.628, HaloScope 0.706 / 0.762 / 0.761 / 0.627. Qwen-2.5-7B: 0.873 / 0.798 / 0.820 / 0.738. TruthfulQA→TriviaQA transfer 0.798 (against 0.840 in domain) ✔ | **32-64 labels** + 128 unlabeled |
| **Mean shift** (Lee et al., EMNLP 2026, [2608.28930](https://arxiv.org/abs/2608.28930)) | the signal is one mean-shift direction; shrinkage LDA closes 73% of the gap between 1-D and full-dimensional classifiers; LayerMix aggregates a contiguous band of layers to match the oracle layer | 0.952 (L2 logistic regression, paired paradigm) ✔ | implies few labels are enough to fix the *direction*; the covariance can be estimated without labels |
| **HBO** (ProbeDrift, [2604.11662](https://arxiv.org/abs/2604.11662)) | rank-normalize a supervised probe (SAPLMA, mid layer) and an unsupervised score (MSP); weight them by the Mahalanobis-distance rank of the test input among the training inputs: equal weights near the training data, all unsupervised far from it | beats the hybrid baselines in almost every OOD setting for short-form; **does not work for long-form** ✔ | the probe's training labels |
| **MIND** (Su et al., 2024, [2403.06448](https://arxiv.org/abs/2403.06448)) | automatic labels from Wikipedia continuations → MLP on states | OpenHalDet: 0.62-0.92 across scenarios, often the best white-box method ✔ | none (self-generated) |
| HaloScope, CCS, SEP, IRIS | see above | | |

---

## 3. Large independent re-evaluations (where the real generalization picture is)

**ProbeDrift** (Stacey, Orgad, Inui, Heinzerling, Moosavi; EMNLP 2026, [2604.11662](https://arxiv.org/abs/2604.11662)).
- Over 2,000 probes. Llama-3.1-8B and Gemma-2-9B, with more models in the appendix.
- Metric: PRR (prediction-rejection ratio; higher is better). ID = in distribution; LOO = leave-one-dataset-out;
  DiffTask = trained on a different task.

Feature ablation (average of Llama and Gemma) ✔:

| features (SAPLMA head) | ID | LOO | DiffTask | OOD avg |
|---|---|---|---|---|
| hidden state, **middle layer** | 0.58 | 0.29 | **0.25** | **0.27** |
| hidden state, last layer | 0.52 | 0.17 | −0.05 | 0.06 |
| attention (last token) | 0.54 | 0.28 | 0.13 | 0.21 |
| Lookback-Lens ratio | 0.19 | 0.11 | −0.04 | 0.04 |
| token probabilities | 0.43 | 0.27 | −0.30 | −0.01 |

Token aggregation ✔:

| aggregation | ID | LOO | DiffTask |
|---|---|---|---|
| **mean of response tokens** | 0.58 | 0.29 | **0.25** |
| last response token | 0.57 | 0.30 | 0.09 |
| random response token | 0.42 | 0.24 | 0.20 |
| mean of all tokens (with context) | 0.33 | 0.09 | 0.06 |
| last context token (pre-answer) | 0.46 | 0.25 | 0.07 |

Other findings:
- **Architecture barely matters for robustness.** Linear probes can be less robust depending on how they are
  trained.
- **The choice of correctness function used for training matters a lot on long-form answers.** A probe trained with
  LLM-judge labels is near chance when evaluated with AlignScore labels.
- **The geometry changes, not the information.** Per-distribution probes still work, but the hidden-state↔correctness
  relation changes across distributions. *That is exactly the "scale moves across domains" problem we see.*
- **Prior OOD claims mostly come from near-OOD settings.** Of the papers surveyed, 54% test only one kind of shift
  and 61% omit strong supervised baselines.

**OpenHalDet** ([2606.06959](https://arxiv.org/abs/2606.06959), June 2026). 17 datasets, 16 detectors, a unified
protocol. White-box AUROC on Llama-3.1-8B ✔:

| method | QA | RAG | Summ. | Math | Science | Code | Agent | Multilingual |
|---|---|---|---|---|---|---|---|---|
| SAPLMA | 0.81 | 0.78 | 0.67 | 0.83 | 0.70 | 0.71 | 0.93 | 0.68 |
| SEP | 0.81 | 0.74 | 0.65 | 0.83 | 0.76 | 0.68 | 0.88 | 0.70 |
| PRISM | 0.82 | 0.80 | 0.68 | 0.81 | 0.81 | 0.68 | 0.87 | 0.61 |
| MIND | 0.78 | 0.65 | 0.62 | 0.84 | 0.79 | 0.64 | 0.92 | 0.73 |
| EigenScore | 0.62 | 0.56 | 0.51 | 0.72 | 0.51 | 0.72 | 0.62 | 0.53 |
| HaloScope | 0.61 | 0.61 | 0.52 | 0.64 | 0.59 | 0.65 | 0.62 | 0.67 |

These are in-domain trained probes. **Summarization is the hardest format for every family** (consistent with our
summaries F1 46.6). No family dominates. Gray-box (likelihood) methods are competitive with white-box ones.

**Dubanowska et al. 2025.** See §2.3. The key warning for us: RAGTruth's pooled AUROC is inflated by task type.

**CrossHallu** (2026). Cross-lingual and cross-domain transfer of the HalluShift-style features. 3-layer MLP; GPT-4o
labels. The numbers are very model-dependent: 0.40-0.87 ✔.

---

## 4. Answers to the four questions

### (1) Which internal-state signals generalize best across domains and formats?

Ranked by the weight of evidence:

1. **Mid-layer residual stream, mean-pooled over the response (or pooled over a band of layers).**
   - It is the most robust supervised feature in ProbeDrift.
   - The peak band sits at about 40-60% of depth for Llama/Mistral and about 70-90% for Qwen2.5 (mean-shift and
     quantized-probe papers ✔).
   - Our reader probe already uses "every 4th layer, MLP inputs, mean per sentence", which is in this family. The
     literature says **drop the last ~15% of layers from the feature set** or weight the middle band, and
     **average over tokens** (or max for spans).
2. **Probes trained on a model-internal target** (semantic entropy, self-consistency, verbalized confidence) rather
   than on task-specific correctness labels.
   - +8-10 AUROC points OOD for SEP; IRIS reaches about 95% of supervised SAPLMA with no labels.
   - The reason: correctness labels let the probe learn "this domain is hard" shortcuts. The D2T confound in
     RAGTruth is the extreme case.
3. **Span- or token-level supervision on long-form text from many domains.** Obeso's entity probe trained on
   LongFact transfers to HealthBench at 0.86-0.87 and to MATH at 0.85-0.86 ✔. This is the most impressive transfer
   in the literature. Note that it is all *factual-entity* hallucination in *closed-book* generation.
4. **Attention-to-context ratios** (Lookback Lens, copying heads, ReDeEP).
   - They transfer reasonably between QA and summarization when used *for context faithfulness* (0.82-0.85 ✔), and
     across model sizes with some loss.
   - In broad suites they are weak stand-alone signals and need data-specific tuning (ReDeEP RAGTruth→Dolly 0.50).
   - The previous-token attention heads of RAUQ are the best *label-free* attention signal.
5. **Logit or confidence features.**
   - Domain-portable *ranking* in short-form answers. Orgad shows probe transfer barely beats them.
   - Their *scale* is length-dependent. This is our own measured failure, and ProbeDrift's DiffTask PRR of −0.30
     for token-probability probes.
6. **What does not generalize:**
   - single-dataset accuracy probes, last-layer or last-token features, EigenScore, HaloScope, CCS;
   - probes trained on synthetic or edited hallucinations and applied to organic ones (CH-Wang);
   - SAE features, which are inconsistent (RAGLens shows gains, Dubanowska none).

**Calibration never transfers.** Even when AUROC transfers (0.76-0.83 across math datasets), ECE triples (to
0.25-0.28). Every paper that measured it agrees with our conclusion: a held false-alarm rate needs in-domain labels
or a same-reply null.

### (2) What is the most label-efficient recipe?

Combining TSV, the mean-shift result, SEP and IRIS, and HBO, the evidence supports a four-stage recipe:

1. **Pre-train a general probe on diverse public data** (zero customer labels). Data:
   - RAGTruth (full 15k);
   - LongFact entity annotations (Obeso, released);
   - HaluEval, MedHallu, ANLI, MiniCheck-synthetic (the proxy-analyzer mix);
   - DROP, PubMedQA, FinanceBench answers.

   Use mid-layer-band mean features and a class-balanced L2 logistic regression or shrinkage LDA. Diversity over
   volume.
2. **Adapt to the customer domain without labels:**
   - (a) Re-center and whiten the activations with the domain's own unlabeled replies (subtract the domain mean;
     shrinkage covariance). This targets the "geometry moves" failure directly.
   - (b) Optionally, self-train on label-free targets in the domain: SEP-style consistency or the reader's
     verbalized "is this supported?" probability, as in IRIS and PRISM.
3. **Use 32-64 labels to adjust the direction and bias,** not to learn a new 87k-dimensional probe:
   - fit only a scalar mix of the general direction and a domain difference-of-means direction, plus a bias;
   - or run TSV-style optimal-transport pseudo-labeling of the unlabeled replies seeded by the 32-64 labels.

   TSV's 32 labels beat fully supervised SAPLMA by 3-13 points.
4. **Add 100-300 labels for the conformal threshold** (already in our product).
5. **At inference, use the HBO-style back-off.** When a reply is far, by Mahalanobis distance, from anything the
   probe was trained or adapted on, shift weight to the label-free score. Warning: HBO did not help on long-form.

### (3) What would most likely raise accuracy above 0.77 label-free and 0.86 supervised?

Expected gains are from the cited papers, transferred with uncertainty.

| lever | evidence | expected gain for us | CPU cost |
|---|---|---|---|
| **Fix the evaluation first:** report RAGTruth AUROC *within task* (QA, Summ, D2T separately) and a task-ID baseline | Dubanowska: D2T confound gives 0.71 for free | may *lower* the 0.86 headline; essential before trusting any gain | minutes |
| **More, more diverse training data** (full RAGTruth 15k + LongFact + HaluEval + MedHallu) | our learning curve +0.02 per doubling; UTH and RAGLens diversity effects; the proxy analyzer at 72k reaches 0.825 with a *0.5B* reader | +0.03-0.06 supervised; +0.05-0.10 on unseen domains | hours to days of feature extraction (the Kaggle T4 path) |
| **Token- or span-level training with max aggregation** (instead of sentence means → logistic regression) | Obeso: token probes + max; RAGLens: max-pooled SAE features | +0.02-0.04, especially on summaries, where hallucinations are a few tokens in a long sentence | low (same features, different pooling) |
| **Layer band selection + cross-layer pooling** | ProbeDrift (last layer collapses OOD), LayerMix, CLAP (+~6.5 OOD), SALA | +0.01-0.03 in domain, +0.03-0.08 OOD | low |
| **Lightly train the reader** (LoRA probe; or prompt the reader with a PRISM-style verification prompt before reading) | Obeso LoRA +0.04-0.05 over linear on every dataset; PRISM 0.80 RAG | +0.03-0.05 | LoRA on 4B on CPU is impractical; on Qwen3-0.6B/1.7B it is feasible. The prompt wrapper is free |
| **Unsupervised domain centering and whitening** | ProbeDrift's "geometry changes"; mean-shift paper | aimed at the label-free 0.62-0.77 across domains, possibly +0.03-0.08 on the worst domains (speculative) | minutes |
| **Self-supervised target probes** (SEP/IRIS style: train on the reader's own verdict or consistency) | +8-10 OOD points over accuracy probes (SEP) | label-free path: +0.03-0.07 (speculative) | sampling on CPU is the cost; use the reader's single-pass verbalized probability instead |
| **Stack with a cheap NLI or HHEM score** | the proxy-analyzer uses HHEM-2.1 as a feature; NLI transfers better than probes in some directions (Lookback table: Summ→QA NLI 0.71) | +0.01-0.03 | HHEM-2.1 (T5-base) on CPU is cheap |
| SAE features | RAGLens +0.10 over ReDeEP with Llama-2; Dubanowska: no gain | uncertain; skip unless Qwen3-4B SAEs are at hand | medium |
| Sampling-based signals (SE, EigenScore, SpikeScore) | expensive, mixed generalization | low value for our cost target | high |

**Realistic target.**
- Supervised RAGTruth: response AUROC about 0.88-0.90 and F1 about 74-77 with full data, token-level max pooling
  and a layer band. LoRA on a smaller reader might reach the fine-tuned SOTA (79).
- Label-free cross-domain: the literature has *no* method that reliably exceeds about 0.75-0.80 across arbitrary
  domains. The realistic lever is the 32-64-label adaptation, which plausibly takes a new domain from about 0.70 to
  about 0.80.

### (4) Concrete CPU experiments (Qwen3 0.6B-4B, a few hours each), ranked by expected value

1. **RAGTruth confound audit and within-task AUROC** (≤30 min; reuses `results/ragtruth_sota_qwen3_4b.json`
   features).
   - Compute response and sentence AUROC within QA, Summ and D2T; a task-ID-only baseline; and a probe trained with
     task-mean-centered features.
   - *Decides whether 0.86 is real.*
2. **Probe-design ablation on cached reader features with a leave-one-source-out protocol** (1-2 h, mostly
   re-fitting).
   - Run on HaluBench's 6 sources plus the 3 RAGTruth tasks.
   - Variables: layer band (early, mid, late, all); pooling (sentence mean, response mean, token-max); classifier
     (L2 logistic regression, shrinkage LDA, difference-of-means).
   - Metric: OOD AUROC. This replicates ProbeDrift's findings on our reader and data.
3. **Label-budget curve for adaptation** (1-2 h).
   - For each held-out source, at k ∈ {0, 16, 32, 64, 128} labels, compare:
     - (a) the general probe, unchanged;
     - (b) the general probe after unsupervised domain centering and whitening (domain-mean subtraction, shrinkage
       covariance from unlabeled replies);
     - (c) (b) plus a k-label difference-of-means direction mixed with the general direction;
     - (d) a TSV-lite: k labels seed optimal-transport or confidence pseudo-labels on the unlabeled replies, then
       refit;
     - (e) a probe from scratch on k labels.
   - This is the experiment that defines the product's "adapt with N labels" story.
4. **PRISM-style verification prompt for the reader** (2-3 h for re-extraction at 0.6B/1.7B; longer at 4B).
   - Wrap (evidence, reply) in "Is every claim in the reply supported by the document? Answer:" and read the
     states and the yes/no logit at the answer position.
   - Compare label-free (yes/no probability) and probed AUROC with the current features. It costs nothing at
     inference.
5. **Token-level, max-aggregated probe** on RAGTruth span labels (2-3 h).
   - Train on token states (span-weighted loss, as Obeso does); score a response by the max or a top-k mean of
     token scores.
   - Expected to help most on summaries.
6. **HBO back-off** (1 h).
   - Mahalanobis distance of each reply's pooled features to the probe's training set; rank-blend the supervised
     probe with the best label-free score.
   - Evaluate on leave-one-source-out HaluBench.
7. **Obeso transfer test** (3-4 h at 1.7B; overnight at 4B).
   - Use the released LongFact entity annotations. HF dataset and code are under `obalcells/hallucination_probes`
     ~ (verify the name).
   - Read the Llama completions with Qwen3 (cross-model reading), train a token probe, then test on HaluBench /
     RAGTruth (with evidence absent) and on DROP / FinanceBench.
   - This tests whether closed-book entity knowledge in the reader adds signal beyond grounding.
8. **SEP-style self-supervised target on Qwen3-0.6B** (3-4 h; sampling is the cost).
   - For about 500 closed-book questions per domain, take 5 samples and binarize semantic agreement; probe on
     greedy-answer states.
   - Compare leave-one-domain-out AUROC against an accuracy probe. Lower priority: it concerns generator-side use.
9. **LoRA probe on Qwen3-0.6B as the reader** (overnight on CPU; may need a T4).
   - Only if 1-5 show that the linear ceiling is reached.

---

## 5. Uncertainty and gaps

- I did not read the SAPLMA, ITI, CCS, Geometry of Truth, TTPD, UTH, INSIDE or HaloScope numbers from the primary
  tables in this review. They are marked ~ and come from memory or summaries.
- I read only the abstracts of SpikeScore's per-dataset tables, the SALA results, the CLAP numbers and the
  shrinkage-LDA sample-efficiency curves of the mean-shift paper.
- The details of "External Observers May See More Clearly" (Oct 2026) were not available beyond the abstract. It
  is the most directly relevant new paper for a reader architecture and worth reading in full.
- The proxy-analyzer paper (2605.07209) has internal number inconsistencies, which the authors acknowledge, and its
  per-generator splits are synthetic.
- I found no paper evaluating a label-free threshold that holds a false-alarm rate across domains. Calibration
  under shift is, as far as I can tell, unsolved in this literature. Conformal calibration with in-domain labels is
  the standard answer.

## 6. Key links

- Obeso et al. 2025, https://arxiv.org/abs/2509.03531
- Kossen et al. (SEP) 2024, https://arxiv.org/abs/2406.15927
- Orgad et al. 2024/25, https://arxiv.org/abs/2410.02707
- Zhang et al. (reasoning models know) 2025, https://arxiv.org/abs/2504.05419
- ProbeDrift 2026, https://arxiv.org/abs/2604.11662 (code: github.com/joestacey/ProbeDrift)
- Dubanowska et al. 2025, https://arxiv.org/abs/2509.19372
- ReDeEP 2025, https://arxiv.org/abs/2410.11414
- Lookback Lens 2024, https://arxiv.org/abs/2407.07071
- RAGLens 2026, https://arxiv.org/abs/2512.08892
- Proxy analyzers 2026, https://arxiv.org/abs/2605.07209
- TSV 2025, https://arxiv.org/abs/2503.01917
- Mean shift 2026, https://arxiv.org/abs/2608.28930
- RAUQ 2026, https://arxiv.org/abs/2505.20045
- OpenHalDet 2026, https://arxiv.org/abs/2606.06959
- SpikeScore 2026, https://arxiv.org/abs/2601.19245
- CrossHallu 2026, https://arxiv.org/abs/2607.04029
- Streaming CoT 2026, https://arxiv.org/abs/2601.02170
- IRIS 2025, https://arxiv.org/abs/2509.10004
- PRISM 2025, https://arxiv.org/abs/2411.04847
- External observers 2026, https://arxiv.org/abs/2610.02066
- UTH 2024, https://arxiv.org/abs/2407.08582
- Geometry of Truth, https://arxiv.org/abs/2310.06824
- TTPD, https://arxiv.org/abs/2407.12831
- INSIDE, https://arxiv.org/abs/2402.03744
- HaloScope, https://arxiv.org/abs/2409.17504
- CH-Wang et al., https://arxiv.org/abs/2312.17249
- HalluShift, https://arxiv.org/abs/2504.09482
- CLAP, https://arxiv.org/abs/2509.09700
- FEPoID, https://arxiv.org/abs/2605.26366
