# Uncertainty quantification and black/gray-box hallucination detection: what to use, what to beat

Research note, 2026-11-07. Scope: scoring how likely a reply from a closed model is wrong when we see only
its text, possibly with no evidence. It covers short answers, long-form text and reasoning chains. The note
covers the methods, their measured accuracy and cost, how stable their score scales are, the benchmarks, and
CPU experiments for this repo (Qwen3-4B reader, 4-core CPU).

**Confidence markers.**
- **[V]**: I checked the number in the paper or its HTML/PDF during this research.
- **[M]**: from memory of the paper or a secondary summary. Check it before quoting it to customers.
- **[repo]**: measured in this repository (file named).

---

## 0. Answers in one page

1. **Closed-book, text only, cheapest strong method.** No method is cheap, strong and text-only all at once.
   The evidence ranks the options as follows:
   - **If the API exposes top-k log-probs** (OpenAI and Gemini do; Anthropic does not), use the generator's own
     token probabilities: MSP, CCP, or a small learned head such as HALT. The cost is near zero, and the
     claim-level AUROC is about 0.65-0.75.
   - **Text only, with a small proxy model's internals.** This works when the proxy knows the facts: the repo
     measured 0.85 on TruthfulQA and 0.99 on HaluEval QA. It is weak on a stronger model's long-tail errors:
     **0.62 / 0.54 held out** for a 3B reader on a 32B model's TriviaQA / NQ replies [repo,
     `results/judge_32b_replies.json`]. Output log-prob of a different reader is at chance (0.42-0.56) [repo].
     The literature agrees. A cross-model predictor is only as good as its **own world knowledge**, plus
     question-difficulty cues. A 7B predicting an 8B's correctness is about as good as the 8B predicting its own
     (GCM, AUROC 0.89 on MMLU) [V]. Nobody shows a ≤4B reader catching a frontier model's long-tail errors.
   - **Strongest text-only option: resample the closed model** (SelfCheckGPT / semantic entropy / LUQ, with 5-20
     samples) and score agreement with a small NLI model. Short-form AUROC is about 0.75-0.85 [M]. It costs 5-20×
     the generation, and it is **blind to self-consistent errors**, where AUROC falls to ≤0.5 ("Too Consistent to
     Detect", EMNLP 2025) [V].
   - **Best accuracy per dollar is probably "make it grounded".** Extract claims, retrieve (Wikipedia or web),
     and run the existing grounding reader. SAFE / VeriScore do this with an LLM judge per claim. We would use
     the 4B reader instead, so the per-claim cost is a search call plus about 1 CPU-second.
2. **Long-form (claim level).**
   - **White/gray-box:** CCP is the best training-free method in LM-Polygraph (claim ROC-AUC 0.74 vs 0.65 MSP vs
     0.62 P(True), Mistral-7B biographies) [V]. Supervised "UQ heads" beat it (PR-AUC 0.66 vs 0.50 in domain) [V].
   - **Black-box:** LUQ and SelfCheckGPT-style sentence consistency (r ≈ −0.85 with FActScore on Gemini Pro) [M].
   - **Retrieval verification (SAFE / VeriScore):** the most accurate and the most expensive. SAFE uses about 50k
     LLM tokens per response, VeriScore about 23k, and FaStFact about 5.6k [V].
3. **Scale stability.** No raw score is stable across domains and lengths. What helps, in order:
   - claim-level scoring instead of response-level (it removes the length effect);
   - claim-conditioned scores (CCP loses less with length than MSP or perplexity) [V];
   - **monotone calibration on a few hundred in-domain labels**: isotonic "PCC" [V], post-hoc calibration that
     cut a cross-dataset ECE from 0.105 to 0.031 [V], and the repo's Clopper-Pearson threshold;
   - **conformal methods with group/conditional coverage** (Mondrian / Cherian-Gibbs-Candès) for guarantees per
     domain.

   Label-free normalization did not hold false-alarm rates in any repo test.
4. **Benchmarks that define "extremely accurate".**
   - **Grounded:** LLM-AggreFact (SOTA about 77 BAcc), RAGTruth (about 79-84 response F1), HaluBench (about 87
     accuracy).
   - **Closed-book claim level:** FActScore bios and LongFact (SAFE agreement 72% with crowd labels). Detection is
     scored with claim ROC-AUC, where 0.74 (CCP) to about 0.8 (supervised heads) is the frontier for 7B
     generators.
   - **Reasoning:** ProcessBench (o1-mini 87.9 F1, Qwen2.5-Math-PRM-72B 78.3, PRM-7B 73.5) and PRMBench.
   - **Parametric knowledge (for selective answering):** SimpleQA / SimpleQA Verified.

   HaluEval and TruthfulQA are saturated or leaky. Report them, but do not sell on them.
5. **CPU experiments, best first** (details in §7):
   1. SelfCheckGPT's WikiBio set, which ships the 20 GPT-3 samples, so no generation is needed;
   2. LLM-AggreFact placement of the current product;
   3. retrieval-grounded closed-book check on FActScore bios;
   4. reader-side CCP;
   5. a GCM-style correctness probe with entity-popularity features;
   6. conformal claim filtering per domain;
   7. a ProcessBench subset.

---

## 1. Method landscape (what each needs, costs, scores)

Access levels:
- **B**: black-box, text of one or more samples.
- **G**: gray-box, top-k token log-probs from the API.
- **W**: white-box, the generator's internals.
- **R**: reader / proxy, a different model's internals.

| method | access | extra cost per reply | headline numbers | notes |
|---|---|---|---|---|
| Semantic entropy (SE) ([Farquhar et al., Nature 2024](https://www.nature.com/articles/s41586-024-07421-0); [Kuhn et al., ICLR 2023](https://arxiv.org/abs/2302.09664)) | G (or B, "discrete SE") | 5-10 samples plus O(N²) NLI clustering | about 0.79 mean AUROC over 30 model×task pairs, vs about 0.69-0.70 for naive entropy, P(True) and embedding regression [M] | Strongest for short factoid QA. Long-form via claim-level "discrete SE" on biographies |
| Semantic entropy probes (SEP) ([Kossen et al. 2024](https://arxiv.org/abs/2406.15927)) | W | ~0 (linear probe on hidden state) | Below SE. Generalizes OOD better than accuracy probes ("AUROC gaps of 7.7-10.5" quoted) [M] | Needs the generator's internals, so it does not fit our setting except as a reader-side analogue |
| Kernel language entropy ([Nikitin et al., NeurIPS 2024](https://arxiv.org/abs/2405.20003)) | B/G | same as SE | Beats SE in about 75% of model-dataset cells [M, secondary] | Soft semantic similarity instead of hard clusters |
| SelfCheckGPT ([Manakul et al., EMNLP 2023](https://arxiv.org/abs/2303.08896)) | B | N=20 samples in the paper, plus NLI or LLM prompt per sentence×sample | WikiBio sentence "non-factual" AUC-PR: Prompt 93.4, NLI 92.5, BERTScore 82.0, random 73.0. Passage Pearson: Prompt 78.3, NLI 74.1 [M] | Its dataset ships the samples, so it can be reproduced on CPU without generating (§7) |
| Graph/degree consistency (Deg, EigV, Ecc; [Lin et al., TMLR 2024](https://arxiv.org/abs/2305.19187)) | B | N samples plus NLI | DegMat is the best black-box method in LM-Polygraph for CoQA/TriviaQA and long outputs [V] | |
| SAR / TokenSAR ([Duan et al., ACL 2024](https://arxiv.org/abs/2307.01379)) | G | 1 (TokenSAR) or N (SAR) plus a relevance model | "Most robust across output lengths" in LM-Polygraph [V] | Reweights tokens by how much they change meaning |
| CCP ([Fadeeva et al., ACL Findings 2024](https://arxiv.org/abs/2403.04696)) | G/W (top-k alternatives per token) | 1 pass plus NLI on alternatives | Claim ROC-AUC 0.74 (Mistral-7B, En), 0.68 (Jais En), 0.73 (Ar), 0.67 (Ru), 0.61 (Zh) [V] | Best training-free claim-level method in LM-Polygraph |
| CoCoA ([Vashurin et al., NeurIPS 2025](https://arxiv.org/abs/2502.04964)) | G + B | N samples. "CoCoA-Light" learns the consistency term and needs none | Reported gains over SOTA UQ on QA, summarization and MT [M, numbers not checked] | Confidence × consistency, from minimum Bayes risk |
| P(True) ([Kadavath et al. 2022](https://arxiv.org/abs/2207.05221)) | G (one extra prompt) | 1 forward | Not better than random for 7-12B in LM-Polygraph [V]. Llama-3-70B P(True) AUROC 0.584 on MMLU (GCM paper) [V]. Works at ~50B+ with samples shown [M] | Unreliable as a product signal |
| Verbalized confidence ([Tian et al. 2023](https://arxiv.org/abs/2305.14975); [Xiong et al., ICLR 2024](https://arxiv.org/abs/2306.13063)) | B | ~0 | Better ECE than raw probs for RLHF models (about 50% relative ECE cut) [M]. AUROC typically 0.55-0.65, heavily clustered at 80-100% [M] | Nearly free with any closed model. A weak ranker; useful as one feature |
| APRICOT ([Ulmer et al., ACL 2024](https://arxiv.org/abs/2403.05973)) | B (question plus answer text into DeBERTa) | 1 small-encoder pass | Best misprediction AUROC vs baselines on TriviaQA/CoQA for GPT-3.5 and Vicuna [M] | The text-only proxy-calibrator idea |
| Generalized Correctness Models ([Xiao et al., ICML 2026](https://arxiv.org/abs/2509.24988)) | R (a different LLM, fine-tuned) | 1 forward of a 7-8B model | MMLU, target Llama3.1-8B: GCM AUROC 0.890 / ECE 0.023, self-model SCM 0.857, P(True) 0.807. TriviaQA: GCM 0.905. MMLU-trained → TriviaQA: AUROC 0.896, ECE 0.105 → 0.031 after post-hoc calibration [V] | **Key finding: a model is no better at judging its own correctness than another model of similar strength. The answer's content adds about 7 accuracy points via the predictor's world knowledge** [V] |
| HALT ([Shapiro et al. 2026](https://arxiv.org/abs/2602.02888)) | G (top-20 log-probs) | tiny GRU | Beats LettuceDetect on its HUB benchmark at 30× smaller [V, numbers not extracted] | Fits OpenAI-style APIs that return top-20 logprobs |
| UQ heads ([Shelmanov et al., EMNLP 2025](https://arxiv.org/abs/2505.08200)) | W (attention features of the generator) | ~0 | Claim PR-AUC on Mistral-7B: bios (in-domain) 0.66 vs CCP 0.50, MCP 0.41, Factoscope 0.61, SAPLMA 0.54, Lookback 0.56. OOD domains 0.34-0.56 vs CCP 0.17-0.38 [V] | Labels from a GPT-4o pipeline at about $100 per LLM. **Not tested on another model's text** [V] |
| Proxy analyzers ([2605.07209](https://arxiv.org/abs/2605.07209)) | R (0.5-9B reader, with evidence) | 1 forward plus HHEM | RAGTruth AUC 0.814-0.836 for every size from 0.5B to 9B. LLM-AggreFact BAcc 0.671-0.699 vs MiniCheck-FT5 0.750 [V] | Grounded setting only. Reader size barely matters for grounding |
| Cross-model probe for self-consistent errors ([Tan et al., EMNLP 2025](https://arxiv.org/abs/2505.17656)) | W + R | 1 verifier forward | Consistency methods ≤ 0.5 AUROC on self-consistent errors. Adding a verifier model's hidden-state probe "significantly" improves them [V, abstract] | Supports a second reader, but only when it **knows different things** |
| InterpDetect ([2510.21538](https://arxiv.org/abs/2510.21538)) | R (Qwen3-0.6B) | 1 forward | Probe trained on Qwen3-0.6B signals transfers to GPT-4.1-mini replies [V, abstract; numbers not extracted] | Grounded RAG setting |

**LM-Polygraph's overall verdict** ([TACL 2025](https://aclanthology.org/2025.tacl-1.11/), [V]):
- **Short outputs** (under 7 characters): information-based methods (MSP, perplexity, CCP, TokenSAR) are best and
  cheapest.
- **Long outputs:** sample-diversity methods, NLI-based black-box ones (DegMat, SE, lexical similarity), are best.
- **SAR** is the most robust across lengths.
- **Reflexive methods** (P(True), verbalized) are poor except for big models such as GPT-4o-mini.
- **Claim-level fact-checking:** CCP is best.
- **MSP** is a "very strong and robust baseline".

---

## 2. Q1: closed-book, text-only verification of a closed model

### 2.1 What the evidence says about a small reader judging a big model

| evidence | number | source |
|---|---|---|
| Output log-prob of a *different* model as a detector | 0.42-0.56 AUROC, at or below chance | [repo] VISION.md, 3B reader, 6 sources |
| Same frozen 3B reader, probe on its internals, closed-book-ish sources | TruthfulQA 0.848, HaluEval QA 0.99 (artefacts), HaluEval general (real ChatGPT replies) **0.65** | [repo] `results/kaggle/detector-3b` |
| 3B detector on a 32B model's own factual replies | 0.62 trace, **0.62 / 0.54 held out** (TriviaQA / NQ); combined with the 32B's own claim score: +0.005 (noise) | [repo] `results/judge_32b_replies.json` |
| Generator reading its own answers (Qwen3-4B) | 0.60-0.89 per domain, but scale moves by domain (10% FA target → 0-64%) | [repo] docs/self_grounded.md |
| Another 7B LLM predicting an 8B's correctness | as good as self-prediction or better. GCM AUROC 0.89 (MMLU), 0.905 (TriviaQA) | [V] GCM |
| Consistency methods on self-consistent errors | ≤ 0.5 | [V] Tan et al. 2025 |

**Reading.** A reader's verdict on a closed-book claim draws on three sources:
- (a) **its own knowledge of the fact.** This helps only if the reader knows it. A 4B reader does not know a
  frontier model's long-tail facts.
- (b) **question/entity difficulty.** Obscure entities get more errors (PopQA, Mallen et al. 2023 [M]). The reader
  can sense this without knowing the answer, but it predicts the *rate*, not which answer is wrong.
- (c) **style of the answer**: hedging, specificity, invented-looking names. Weak, and it can be gamed.

GCM's decomposition supports this [V]:
- world knowledge (answer content) is the component that transfers across models;
- answer phrasing is substantial in domain but brittle.

**Expected performance** of a 4B reader on a frontier model's closed-book factoid errors: **AUROC 0.55-0.70**. That
is a triage signal, not a verifier. This matches the repo's 32B test. It is an inference, not a published number
for a ≤4B reader against GPT-4-class text; nobody has published that. Experiment 5 measures it.

### 2.2 Ranked options for a closed model, no evidence

| option | extra cost | expected quality | when to use |
|---|---|---|---|
| A. Generator log-probs, if the API returns top-k (MSP / CCP-from-top-k / HALT-style head) | ~0 | claim AUROC about 0.65-0.75 | OpenAI/Gemini customers. **Not Anthropic** (no logprobs) |
| B. Verbalized confidence from the closed model (ask for it in the same call) | ~0 | about 0.55-0.65 AUROC, poorly calibrated | One cheap feature to stack |
| C. Reader probe on (question, answer), trained GCM-style on historical correctness labels of *that* generator, plus entity-familiarity features | 1 forward of a 4B reader (about 1-5 CPU-s) | 0.55-0.70 on hard long-tail. Higher on misconception-type errors (TruthfulQA 0.85) | Always-on triage |
| D. Resample the closed model 5-10× and score agreement with a small NLI model (SelfCheckGPT-NLI / discrete SE / LUQ) | 5-10× generation cost plus NLI (DeBERTa-large on CPU about 0.1 s per pair) | about 0.75-0.85 short-form. Fails on self-consistent errors | When the customer pays for generation anyway |
| E. Retrieve evidence (search/Wikipedia), then run the **existing grounding reader** per claim | search call(s) plus about 1-6 CPU-s per reply | Grounded performance (0.82-0.92 AUROC on RAGTruth-like data) **when retrieval finds the deciding fact** | Factoid / entity-centric content. This converts the weak spot into the product's strong spot |
| F. SAFE / VeriScore-style LLM judge with search | about 23-50k LLM tokens per response [V] | 72% agreement with crowd labels, wins 76% of disagreements (SAFE) [M] | Offline evaluation and label generation, not serving |

**Recommendation:** C as the always-on score, then E on the claims C flags. D is an optional "premium" tier.

Caveat from the repo: closed-book trivia through the full server caught only 1-2 of 13-14 wrong answers, because
"the lookup rarely finds the deciding fact". So **retrieval recall is the bottleneck for E**, not verification.
Claim-targeted queries (entity plus relation from the claim) are a known fix in SAFE / VeriScore.

---

## 3. Q2: long-form, claim-level

| method | access | cost per long reply (~20-30 claims) | claim-level quality |
|---|---|---|---|
| FActScore ([Min et al., EMNLP 2023](https://arxiv.org/abs/2305.14251)) | evaluator: LLM plus Wikipedia retrieval | one LLM call per atomic fact | Estimator within about 2% of human FActScore [M]. Human scores: InstructGPT 42.5, ChatGPT 58.3, PerplexityAI 71.5 [M] |
| SAFE / LongFact ([Wei et al., NeurIPS 2024](https://arxiv.org/abs/2403.18802)) | LLM plus Google search | about 49.6k tokens per response [V, FaStFact table]. ">20× cheaper than humans" (about $0.19 vs $4 per response) [M] | 72% agreement with crowd annotators. Wins 76% of 100 disagreements [M] |
| VeriScore ([Song et al., EMNLP Findings 2024](https://arxiv.org/abs/2406.19276)) | LLM (or fine-tuned Mistral-7B extractor plus Llama3-8B verifier) plus search | about 22.8k tokens [V]. Example: 14 extraction calls, 23 searches, 23 verification calls per response [V] | Humans preferred its claim extraction to SAFE's in 93% of cases [V] |
| FaStFact ([2510.12839](https://arxiv.org/abs/2510.12839)) / VeriFastScore ([2505.16973](https://arxiv.org/abs/2505.16973)) | LLM plus search, batched | about 5.6k tokens [V] | Comparable or better human agreement [M] |
| CCP (white/gray) | top-k logprobs plus NLI | ~1 pass plus NLI on a handful of alternatives per claim | ROC-AUC 0.74 / PR-AUC 0.46 (20.5% unsupported) [V]. 0.775 ROC-AUC on the multilingual MUCH benchmark [V] |
| UQ heads (white, supervised) | generator attention | ~0 | PR-AUC 0.66 in domain, 0.34-0.56 OOD [V] |
| LUQ ([Zhang et al., EMNLP 2024](https://arxiv.org/abs/2403.20279)) | B, N samples | N long generations plus sentence×sample NLI | Pearson about −0.85 with FActScore (Gemini Pro) [M]. AGSC (2026) claims higher [M] |
| Graph-based claim centrality ([Jiang et al., NeurIPS 2024](https://arxiv.org/abs/2410.20783)) | B | N samples plus claim-sample entailment graph | Reported gains over self-consistency baselines in claim AUPRC [M] |
| Conformal factuality ([Mohri & Hashimoto, ICML 2024](https://arxiv.org/abs/2402.10978)) | any claim score | calibration set (~hundreds of labelled replies) | FActScore factuality 30% → 80% while keeping about half the claims. NQ 78 → 93%, MATH 75 → 95% [V, HTML]. A review says 25%; treat the baseline as uncertain |
| Enhanced / conditional conformal ([Cherian, Gibbs, Candès, NeurIPS 2024](https://arxiv.org/abs/2406.09714)) | any claim score | as above | Conditional (per-group) guarantees and adaptive levels. Retains more claims than Mohri & Hashimoto [M] |
| Coherent factuality ([Rubin-Toles et al. 2025](https://arxiv.org/abs/2505.17126)) | claim graph | as above | Conformal filtering that respects dependencies between reasoning steps [M] |
| Conformal language modeling ([Quach et al., ICLR 2024](https://arxiv.org/abs/2306.10193)) | B, sampling | samples until the stopping rule fires | Set-valued guarantee that at least one acceptable answer is in the set |

**Takeaways for us.**
- The reader-only product can do claim-level work at about 1 forward pass per reply: split into claims and score
  each claim's span. LM-Polygraph's numbers (0.65-0.74) are the bar for a training-free single-pass method on 7B
  biographies. Supervised heads (0.66 PR-AUC) are the bar with labels.
- Retrieval pipelines (SAFE / VeriScore) are 10-100× more expensive, because the LLM judge per claim dominates.
  Replacing that judge with our grounding reader keeps the search cost and removes most of the LLM cost.
- That combination, **search plus a 4B reader as per-claim verifier**, has no direct published comparison. It is
  our niche.

---

## 4. Q3: score scales that hold across domains and lengths

| technique | what it fixes | evidence |
|---|---|---|
| Claim-level instead of response-level scoring, with "weakest claim" aggregation | length: a longer reply has a lower worst-token prob | LM-Polygraph PRR by output length, from 1-2 characters to 211-1k characters: MSP 0.71 → 0.26, perplexity 0.71 → 0.26, TokenSAR 0.70 → 0.26, **CCP 0.59 → 0.36** (best long-tail retention) [V]. [repo] weakest claim beat mean of claims, 0.78 vs 0.72 |
| Claim-conditioned / alternative-normalized scores (CCP, the repo's claim contrast) | lexical/format variance: only meaning-changing alternatives count | CCP above [V]. The repo's contrast null favoured copied spans and failed on computed/judged answers (docs/claim_contrast.md) [repo] |
| Monotone in-domain calibration: isotonic / binned "PCC" | maps raw score to expected quality in [0,1] | LM-Polygraph: isotonic PCC best on calibration MSE, no loss of ranking [V] |
| Post-hoc calibration of a cross-domain predictor | ECE drift under domain shift | GCM, MMLU → TriviaQA: AUROC unchanged (0.896), ECE 0.105 → 0.031 with post-hoc calibration [V] |
| Split conformal with a confidence bound (the repo's Clopper-Pearson threshold) | holds a stated false-alarm rate | [repo] 10.7% FA vs 10% target from 300 labels; 16% with a plain quantile |
| Mondrian / conditional conformal (per domain, per answer type, per length bucket) | one threshold over a mix fails (4-32% FA per HaluBench source [repo]) | Cherian-Gibbs-Candès 2024 [M]: per-group coverage at the cost of more labels per group |
| Label-free per-domain standardization (z-score / quantile on unlabelled traffic) | shifts the location only | [repo] every label-free threshold missed (11-46% FA). **Do not sell this** |
| Length/format features in the calibrator (answer length, answer type: number / name / yes-no / free text) | the scale moves with answer form | [repo] PubMedQA (40 tokens) vs DROP (4 tokens) explained much of the FA drift |

**Bottom line.** Published methods do not get a domain-invariant scale for free either. They all calibrate in
domain (PCC, conformal, post-hoc). The defensible promise remains the repo's: "≤X% false alarms on your domain,
95% confidence, from N labels". Two additions:
- **Per-answer-type Mondrian groups** fix most of the format-driven drift with few labels per group.
- **Claim-level CCP-style normalization** keeps the ranking stable across lengths.

---

## 5. Q4: benchmarks, and the numbers that mean "extremely accurate"

| benchmark | what it tests | metric | SOTA / reference numbers | use for us |
|---|---|---|---|---|
| [LLM-AggreFact](https://llm-aggrefact.github.io/) (11 datasets) | grounded claim verification (doc, claim) | avg balanced accuracy | Bespoke-MiniCheck-7B **77.4**; next about 77.2 (I believe Claude-3.5 Sonnet [M]); Granite Guardian 3.3 76.5; MiniCheck-FT5 75.0; HHEM-2.1 71.8 [V via snapshot, May 2026]. Proxy-analyzer readers 67-70 [V] | **The headline grounded benchmark.** "Extremely accurate" means ≥77 BAcc at small-model cost. Note: BAcc ties hide large instance-level disagreement ([Godbole & Jia 2025](https://arxiv.org/abs/2501.14883)) |
| [RAGTruth](https://aclanthology.org/2024.acl-long.585/) | span/response hallucination in RAG (QA, summary, data-to-text) | response F1, span F1 | LettuceDetect-large 79.2; fine-tuned Llama-2-13B 78.7; GPT-4-turbo prompted 63.4 [V, repo doc]. RAG-HAT (fine-tuned Llama-3-8B) reported slightly above LettuceDetect, about 80-84 [M, unchecked] | Already in use (repo 70.4 from 800 labels) |
| [HaluBench](https://arxiv.org/abs/2407.08488) (Lynx) | 6 sources, grounded QA | accuracy | Lynx-70B about 87.4%, GPT-4o about 86.5% [M] | Already in use. Report per source |
| [HaluEval](https://arxiv.org/abs/2305.11747) | QA/dialogue/summary, ChatGPT-written hallucinations | accuracy / AUROC | Easy for probes: 0.99 [repo] because of style artefacts | Report, but discount. "HaluEval general" (real replies, 0.65 [repo]) is the honest subset |
| [TruthfulQA](https://arxiv.org/abs/2109.07958) | imitative falsehoods / misconceptions | MC1/MC2, judge | Frontier models are saturated or contaminated [M] | Closed-book sanity check (reader 0.85 [repo]), not a sales metric |
| [FELM](https://arxiv.org/abs/2310.00741) | segment-level factuality of ChatGPT answers across world knowledge, math, reasoning, science, writing | segment F1 / balanced acc | GPT-4 detectors: segment F1 roughly in the 40s-50s, balanced acc about 65-70 [M, uncertain] | Good, hard **mixed-format** closed-book set. Small, so CPU-feasible |
| [FActScore bios](https://github.com/shmsw25/FActScore) | atomic-fact precision of biographies | human-labelled facts for InstructGPT / ChatGPT / PerplexityAI | Claim-level detection: CCP 0.74 ROC-AUC (Mistral bios) [V]; UQ heads PR-AUC 0.66 [V] | **Primary closed-book long-form set.** Human labels exist for about 500 bios × 3 systems |
| [LongFact](https://github.com/google-deepmind/long-form-factuality) / SAFE | long-form, 38 topics | F1@K, SAFE agreement | SAFE vs humans: 72% agreement [M] | Prompts only, labels via SAFE. Use as a stress test of cost |
| [ProcessBench](https://arxiv.org/abs/2412.06559) | first erroneous step in math CoT (GSM8K, MATH, OlympiadBench, Omni-MATH) | F1 of error-step identification (harmonic mean of acc on erroneous/correct) | o1-mini **87.9**; Qwen2.5-Math-PRM-72B 78.3, PRM-7B 73.5 ([Zhang et al. 2025](https://arxiv.org/abs/2501.07301)) [V]; QwQ-32B-Preview 71.5; GPT-4o 61.9; Math-Shepherd-7B 31.5 [V]. ThinkPRM up to +8 F1 over discriminative PRMs [V] | Reasoning weak spot. A 4B-reader step probe should be compared against PRM-7B's 73.5 |
| [PRMBench](https://arxiv.org/abs/2501.03124) | fine-grained step errors (redundancy, soundness, sensitivity) | PRMScore (50 = random) | Qwen2.5-Math-PRM-72B 78.6 (per secondary snapshot) [M]; o1-mini / Gemini-thinking about 68-69 in the original paper [M] | Secondary |
| [SimpleQA](https://openai.com/index/introducing-simpleqa/) / [SimpleQA Verified](https://arxiv.org/abs/2509.07968) | short parametric facts. "Not attempted" is allowed | correct / F1 | SimpleQA Verified F1: Gemini 2.5 Pro 55.6, GPT-5 52.3, o3 51.9 [V]. Original SimpleQA: o1-preview about 42.7%, GPT-4o about 38% [M] | Ideal for **closed-book detection on frontier-model text**: half the answers are wrong, so a detector's AUROC on GPT-class answers is a clean test of Q1 |
| [HalluLens](https://arxiv.org/abs/2504.17550), [MUCH](https://arxiv.org/abs/2511.17081) | extrinsic/intrinsic hallucination; multilingual claims | various | MUCH: CCP 0.775, SAR 0.746 ROC-AUC [V] | Optional |

**What "extremely accurate" should mean commercially** (proposed bar):
- **Grounded:** ≥77 BAcc on LLM-AggreFact and ≥79 response F1 on RAGTruth, at ≤ MiniCheck-7B cost.
- **Closed-book claims:** ≥0.80 claim ROC-AUC on FActScore bios and on SimpleQA answers from a frontier model.
  No text-only method has published that; with retrieval it is plausible.
- **Reasoning:** ≥73.5 ProcessBench F1 (PRM-7B parity) from a ≤4B reader.
- **Always:** a false-alarm rate held per domain with a stated confidence.

---

## 6. 2025-2026 developments worth tracking

- **Learned single-pass UQ replaces sampling.** Examples: SEPs, UQ heads, CoCoA-Light, HALT, GCM. The pattern is
  to distil expensive sampling-based or LLM-judge signals into a probe or head. Each only reaches the generator's
  internals, except GCM (a different LLM) and HALT (API log-probs).
- **"Self-knowledge" is not privileged.** GCM (ICML 2026) shows cross-model predictors match self-prediction at
  similar scale. This is good for a reader product, but the cap is the reader's knowledge.
- **Self-consistent errors** (EMNLP 2025) grow as models scale. They defeat every sampling method, so resampling
  alone will not hold up for frontier generators.
- **Proxy/reader detectors for RAG** are now a recognised line of work: proxy analyzers (2605.07209), InterpDetect
  (2510.21538). Our product is in this line. Their numbers (RAGTruth AUC about 0.82-0.84 from 0.5B-9B readers,
  AggreFact about 0.67-0.70 BAcc) are directly comparable to ours (RAGTruth 0.82 untrained, 0.86 trained).
- **Cheaper long-form checking:** VeriFastScore and FaStFact (about 4-9× fewer tokens than SAFE).
- **Reasoning:** generative PRMs / ThinkPRM, and REDE (2607.22098) denoising reasoning traces before detection.
  The REDE abstract does not report a black-box proxy number; I could not verify the claim that it "recovers most
  of the white-box gain".

---

## 7. Q5: CPU-feasible experiments, ranked by expected value

All of these run with Qwen3-4B or smaller on 4 CPU cores. Costs assume about 1-6 CPU-s per reader pass, as
measured in the repo.

| # | experiment | why (value) | data / cost | success criterion |
|---|---|---|---|---|
| 1 | **SelfCheckGPT WikiBio head to head, no generation needed** | Direct, public comparison of our reader against the canonical black-box method on GPT-3 closed-book long-form text. Its 20 samples per passage ship with the dataset. Tests three things: (a) reader probe per sentence with no evidence; (b) the reader as a *cheap NLI scorer* of the shipped samples, i.e. SelfCheck with our reader; (c) the reader grounded on each sample as pseudo-evidence | [potsawee/wiki_bio_gpt3_hallucination](https://huggingface.co/datasets/potsawee/wiki_bio_gpt3_hallucination): 238 passages, about 1.9k sentences. About 2-4 CPU-h | Sentence AUC-PR vs SelfCheck-NLI 92.5 / Prompt 93.4 (random 73.0). Closing half the gap with 1 pass would be notable. (c) is the "samples as evidence" bridge that reuses the existing grounding product |
| 2 | **LLM-AggreFact placement of the current product** | The commercial grounded leaderboard. Cheap, and a direct sales number | Public test sets, subsample about 200 per dataset × 11. About 4-8 CPU-h | ≥75 BAcc (MiniCheck-FT5); 77.4 is SOTA. Report CI |
| 3 | **Retrieval-grounded closed-book check on FActScore bios** | Converts the weakest spot (closed-book) into the product's strength. Measures whether retrieval recall or verification is the bottleneck | FActScore human-labelled atomic facts (ChatGPT / InstructGPT / PerplexityAI bios). Wikipedia pages for each entity ship with FActScore, so retrieval is offline. About 3-6 CPU-h for 100 bios | Claim ROC-AUC ≥0.80. Compare with CCP 0.74 (different generator; note it) and with the reader without evidence |
| 4 | **Reader-side CCP** | CCP is the best training-free claim method but needs the generator's logits. Test whether the *reader's* top-k alternatives at claim tokens, filtered by a small NLI model (DeBERTa-v3-base), give a usable, length-stable claim score on *other models'* text. Cheap, and new | Same FActScore data plus SimpleQA answers. About 2-4 CPU-h | Claim AUROC above reader min-log-p (expected), and a stable FA rate across short vs long claims |
| 5 | **GCM-style correctness probe with entity-familiarity features (closed-book triage)** | Quantifies Q1 directly: can a 4B reader rank a stronger model's closed-book answers? Feature set: reader probe on (question, answer), the reader's own answer agreement, entity frequency/popularity (Wikipedia pageviews or reader's entity log-prob), verbalized confidence if available | Frontier answers on SimpleQA Verified (1k) or PopQA. Use public model outputs where released; otherwise a small API budget (about $5). Train on one generator, test on another | AUROC ≥0.70 on held-out generator; ≥0.65 holding a 10% FA rate per answer type |
| 6 | **Conformal claim filtering with Mondrian groups** | Turns scores into a guarantee customers can buy. Measures claims retained at 80/90% factuality, overall vs per domain | FActScore bios plus FELM segments. Minutes once scores exist | Retain ≥50% claims at 80% factuality (Mohri & Hashimoto parity) with per-group validity |
| 7 | **ProcessBench subset with a reader step probe** | Reasoning weak spot. Gives a first number against PRM-7B 73.5 | 400 items, long CoTs, about 6-10 CPU-h | ≥60 F1 would justify more work; parity is unlikely with ≤4B and no training on PRM data |
| 8 | **FELM mixed-format run** | Tests scale stability across math / reasoning / world knowledge / writing in one dataset | 847 responses, small | Per-domain FA spread under one calibrated threshold vs Mondrian |

**Do not prioritise:**
- Full semantic entropy or LUQ with our own resampling. It needs closed-model samples (cost) and is already well
  characterised.
- P(True) or verbalized confidence from ≤4B models. LM-Polygraph shows they are near random.

---

## 8. Uncertainty and gaps in this note

- [M] numbers were not re-checked against tables. The main ones are Farquhar's means, SelfCheckGPT AUC-PR, SAFE
  agreement and cost, HaluBench (Lynx), SimpleQA original, FELM, PRMBench and LUQ. Before external use, check
  them in the linked papers.
- The second-place LLM-AggreFact entry and current leaderboard state come from a third-party snapshot (May 2026).
  The official page was not opened.
- RAGTruth SOTA beyond LettuceDetect (RAG-HAT, newer 2025-26 detectors) is not confirmed.
- No published result directly measures a ≤4B reader's internals judging frontier-model closed-book claims. The
  0.55-0.70 expectation is an extrapolation from the repo's 3B → 32B test and the GCM decomposition.

Sources not linked inline:
- [LM-Polygraph code](https://github.com/IINemo/lm-polygraph)
- [Bespoke-MiniCheck](https://bespokelabs.ai/bespoke-minicheck)
- [FaStFact](https://arxiv.org/html/2510.12839v2)
- [LLM-AggreFact snapshot](https://benchmarklist.com/benchmarks/llm_aggrefact/)
- [LettuceDetect](https://arxiv.org/abs/2502.17125)
