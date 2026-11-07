# Verifiers for long-form reasoning (CoT, math, code, analysis, agent traces)

Research note for the reasoning arm of the verifier. Literature from 2024 to Nov 2026, with numbers read from
the papers (arXiv HTML or abstract pages) unless flagged. **[unverified]** marks a number taken from a secondary
summary or a figure I could not read, or a claim I could not check.

Our constraints: a frozen reader (Qwen3-4B today) plus a linear probe, two forward passes, about 6 s per RAG reply on
a 4-core CPU (`docs/product.md`). Exact checks already exist: `vt/mathcheck.py` (arithmetic re-computation) and
`vt/pcr/check.py` (exact-rational plan checking). Experiments run on CPU with models of 4B or less.

---

## 0. Short answers

1. **Cheapest reliable flag for a wrong final answer.**
   - *Text only (closed model, one output):* run the deterministic checks first. Recomputing the arithmetic,
     checking SymPy equivalence of stated equations, and executing code against tests are close to free and
     close to 100% precise, but they catch only the error classes they cover. After them, the best single-pass
     signal is a discriminative verifier: a reader with a head, scoring the whole trace in one forward pass.
     Published small discriminative PRMs are weak (Skywork-PRM-1.5B: 36.4 ProcessBench F1). The 7B Qwen2.5-Math-PRM
     is good (73.5) but outside our size budget, and all of them are math-only. A frozen reader plus a probe is
     untested on another model's reasoning; that is experiment E1. If the customer can afford *k* samples,
     self-consistency combined with verbalized confidence reaches 84 to 88 AUROC on math with K=2 to 8, and
     77 to 82 on STEM/humanities. That is the strongest *text-only* signal, but it costs K generations.
   - *With internals (the customer runs an open model):* a linear probe on the generator's own hidden states
     reaches 0.80 to 0.95 AUROC in-domain on math with as few as 20 to 100 labeled problems. Token-confidence
     statistics (DeepConf "tail" and "lowest group" confidence) need only logprobs and cost nothing. Both are
     much cheaper than any PRM. Caveat: a large part of the probe signal is *problem difficulty*, not the trace.
     Correctness can be predicted at 0.84 AUROC from the first 4 tokens, and within-problem effect sizes are
     modest (Cohen's d 0.13 to 0.69).
2. **Transfer across domains.** Within math (GSM8K ↔ MATH), transfer is good. Math to science is partial. Math to
   logic, law, general or ARC-style tasks is poor: probes fall to 0.54 to 0.76 AUROC, and math PRMs are "no better
   than baseline" on law, philosophy and biology. Generative verifiers transfer better than discriminative ones
   (ThinkPRM is +8% on GPQA-physics and +4.5% on LiveCodeBench over a discriminative PRM). Plan on 100 to 500
   labels per new domain for a probe and threshold; the existing calibration pipeline already supports this.
3. **Finding the first wrong step cheaply.** The best accuracy comes from generative step judges: TIPS-Qwen3-4B
   reaches 85.2 ProcessBench F1 and GenPRM-7B 75 to 80. They cost 2K to 5K generated tokens per trace, which is
   minutes on our CPU, so they are only an escalation tier. The cheap route has three layers:
   - (a) exact checks, which localize precisely the steps they cover;
   - (b) per-step scores from one forward pass: the minimum or first step under a threshold from a step-probe or
     PRM head. Uncertainty-head results suggest a probe of under 10M parameters can match 7B PRMs on step PR-AUC,
     but plain linear step-probes have done badly in some papers (≈0.56 AUROC on GSM8K steps);
   - (c) reader-surprisal dips, a DeepConf-style windowed confidence over the trace.

   Agent traces are much harder: the best step attribution is 11 to 14% on TRAIL and Who&When, and 43 to 49% on
   AgentProcessBench with a 4B judge.
4. **Design and experiments.** A tiered cascade: exact tools → one-pass reader features (answer probe, step
   probe, surprisal) → optional generator-side signals (logprobs, k samples) → generative judge escalation.
   Experiments are ranked in §6. Top three: E1, the frozen-reader probe on ProcessBench at answer and step level;
   E2, the exact-tool coverage audit on ProcessBench; E3, cross-domain transfer on MR-Ben and VersaPRM's
   MMLU-Pro-CoT.

---

## 1. Outcome vs process reward models

### 1.1 Foundations

- **PRM800K / "Let's Verify Step by Step"** (Lightman et al., ICLR 2024,
  [arXiv 2305.20050](https://arxiv.org/abs/2305.20050)). 800K human step labels on 75K solutions to 12K MATH
  problems. Their PRM solved **78.2%** of a 500-problem MATH subset with best-of-N, beating the ORM and majority
  voting. The exact ORM and majority-vote numbers (≈72% and ≈70%) are **[unverified]**: I could not confirm them in
  this pass. The data is on HF as [`tasksource/PRM800K`](https://huggingface.co/datasets/tasksource/PRM800K).
- **Math-Shepherd** (Wang et al. 2024). Step labels come automatically from Monte-Carlo (MC) rollouts: a step is
  "good" if continuations from it reach the right answer. 440K samples,
  [`peiyi9979/Math-Shepherd`](https://huggingface.co/datasets/peiyi9979/Math-Shepherd).

### 1.2 What the 2025 "Lessons" paper found

Zhang et al., [arXiv 2501.07301](https://arxiv.org/abs/2501.07301), Qwen team:

- **MC labels are noisy.** On ProcessBench, the PRM trained on MC labels (Math-Shepherd, 440K) scores
  **28.9 F1**. MC labels on their own 860K samples score 40.1, LLM-as-judge labels 46.5, and human PRM800K labels
  (264K) **56.5**. Best-of-8 accuracy hardly separates these (64.3 to 65.9). **Best-of-N is a poor benchmark for
  PRMs.**
- **Consensus filtering** keeps only samples where the MC label and an LLM judge agree on the error step. It keeps
  about 40% of the data and matches the LLM-judge labels. Hard labels beat soft labels after filtering.
- **PRMs drift toward being ORMs.** Several open PRMs put more than 40% of their minimum step scores on the
  *final* step, so they are really judging the outcome. On "correct answer, flawed process" cases, the detection
  rates are:

  | Model | Detection rate |
  |---|---|
  | Math-Shepherd-PRM-7B | 13.9% |
  | Skywork-PRM-7B | 27.8% |
  | Qwen2.5-Math-7B-PRM800K | 38.2% |
  | Qwen2.5-Math-PRM-7B | 53.9% |
  | Qwen2.5-Math-PRM-72B | 58.1% |

  This matters for faithfulness (§7).

### 1.3 ProcessBench

Zheng et al., [arXiv 2412.06559](https://arxiv.org/abs/2412.06559); data at
[`Qwen/ProcessBench`](https://huggingface.co/datasets/Qwen/ProcessBench). About 3.4K expert-labelled solutions in
four splits: GSM8K (400), MATH, OlympiadBench and Omni-MATH (1,000 each). Each item has the earliest error step
(`label`, −1 if none) and `final_answer_correct`. The metric is the harmonic mean of accuracy on erroneous items
(exact first-error step) and on correct items.

Avg F1 (Table 7 of the Lessons paper, plus later papers):

| Verifier | Size | Type | GSM8K | MATH | Olymp. | Omni | **Avg** |
|---|---|---|---|---|---|---|---|
| Math-Shepherd-PRM | 7B | disc. PRM | 47.9 | 29.5 | 24.8 | 23.8 | 31.5 |
| RLHFlow-PRM-Deepseek | 8B | disc. PRM | 38.8 | 33.8 | 16.9 | 16.9 | 26.6 |
| **Skywork-PRM-1.5B** | 1.5B | disc. PRM | 59.0 | 48.0 | 19.3 | 19.2 | **36.4** |
| Skywork-PRM-7B | 7B | disc. PRM | 70.8 | 53.6 | 22.9 | 21.0 | 42.1 |
| Qwen2.5-Math-7B-PRM800K | 7B | disc. PRM (human labels) | 68.2 | 62.6 | 50.7 | 44.3 | 56.5 |
| **Qwen2.5-Math-PRM-7B** | 7B | disc. PRM | 82.4 | 77.6 | 67.5 | 66.3 | **73.5** |
| Qwen2.5-Math-PRM-72B | 72B | disc. PRM | 87.3 | 80.6 | 74.3 | 71.1 | 78.3 |
| Qwen2.5-Math-RM-72B | 72B | **ORM** | 43.5 | 47.2 | 37.6 | 27.4 | 38.9 |
| GPT-4o-0806 (prompted) | – | critic | 79.2 | 63.6 | 51.4 | 53.5 | 61.9 |
| QwQ-32B-Preview (prompted) | 32B | critic | 88.0 | 78.7 | 57.8 | 61.3 | 71.5 |
| o1-mini (prompted) | – | critic | 93.2 | 88.9 | 87.2 | 82.4 | 87.9 |
| GenPRM-1.5B pass@1 / maj@8 ([2504.00891](https://arxiv.org/abs/2504.00891)) | 1.5B | gen. PRM + code exec | 52.8 / 51.3 | 66.6 / 74.4 | 55.1 / 65.3 | 54.5 / 62.5 | **57.3 / 63.4** |
| GenPRM-7B pass@1 / maj@8 | 7B | gen. PRM | 78.7 / 81.0 | 80.3 / 85.7 | 72.2 / 78.4 | 69.8 / 76.8 | 75.2 / 80.5 |
| Qwen3-4B-Thinking-2507 (prompted) ([2609.36641](https://arxiv.org/abs/2609.36641)) | 4B | critic | 52.1 | 70.1 | 59.5 | 59.4 | 60.3 |
| **TIPS-Qwen3-4B-Thinking** (RL on 3.2K *outcome* labels) | 4B | gen. PRM | 89.5 | 87.7 | 83.5 | 80.0 | **85.2** |

ThinkPRM ([2504.16828](https://arxiv.org/abs/2504.16828)) reports only the OlympiadBench and Omni subsets.
ThinkPRM-1.5B scores **76.3 / 75.7** (prompted R1-Qwen-1.5B: 5.0 / 5.4, mostly invalid output) and ThinkPRM-14B
87.3 / 85.7, trained on about 1K synthetic chains (8K step labels).

Takeaways:

- Among *small* discriminative PRMs (≤1.5B) the published ones are poor: Skywork-1.5B is under 20 F1 on the hard
  splits.
- The jump to 70+ F1 needs either 7B models trained on good labels or *generative* judges.
- TIPS shows that a 4B reasoning model trained with RL on only **3.2K outcome-labelled** trajectories becomes an
  85 F1 step judge, close to o1-mini. That fits our "adapt quickly" goal, but it needs a GPU to train and run.

### 1.4 Harder benchmarks: scores collapse

- **PRMBench** ([2501.03124](https://arxiv.org/abs/2501.03124), [HF](https://huggingface.co/datasets/hitsmy/PRMBench_Preview)):
  6,216 problems, 83,456 step labels, fine-grained error types (redundancy, soundness, sensitivity).
  - The best model, Gemini-2-Thinking, scores 68.8 PRMScore against a human score of 83.8.
  - Skywork-PRM-7B scores 65.1 and Skywork-PRM-1.5B 61.1. On this metric, 50 is about chance.
- **Hard2Verify** ([2510.13744](https://arxiv.org/abs/2510.13744),
  [HF](https://huggingface.co/datasets/Salesforce/Hard2Verify)): 1,860 steps from 200 frontier-math proofs.
  - Qwen2.5-Math-PRM-72B drops from 78.3 on ProcessBench to **37.3**.
  - Models with 60+ on ProcessBench score under 20.
  - GPT-5-class critics lead.
- **Implication:** ProcessBench-level competence covers textbook to olympiad *answer* problems. Proof-style or
  research-level reasoning is beyond small verifiers.

---

## 2. Generative verifiers (LLM judges, critics, thinking PRMs)

- **GenRM** (Zhang et al., ICLR 2025, [2408.15240](https://arxiv.org/abs/2408.15240)): the verifier is trained
  with next-token prediction to emit "Yes/No", optionally after a CoT (GenRM-CoT). A Gemma-9B GenRM-CoT verifier
  raised best-of-N on GSM8K for Gemini 1.0 Pro outputs from ≈73% to 92.8%. It beat discriminative RMs at 2B, 7B
  and 9B.
- **ThinkPRM** (Khalifa et al., TMLR 2026, [2504.16828](https://arxiv.org/abs/2504.16828)):
  - Fine-tunes R1-distilled models on about 1K synthetic verification chains (≈1% of PRM800K labels).
  - Beats a discriminative PRM trained on ~712K labels by up to 8 F1.
  - Out of domain it beats the discriminative PRM by **8% on GPQA-physics** and **4.5% on LiveCodeBench**, while
    Qwen2.5-Math-PRM "struggles when applied out-of-domain".
  - Costs: verification chains of 1K to 5K tokens. Scores cluster near 0 or 1 (overconfident). An early wrong
    step label biases later labels.
  - Prompted judges loop or emit invalid labels; 51% invalid for the prompted 1.5B model.
- **GenPRM** ([2504.00891](https://arxiv.org/abs/2504.00891)) writes CoT plus Python per step and executes it.
  Trained on 23K MATH problems. It generates about **2.8K to 4.9K tokens per response verified** (MATH to Minerva).
  Code execution adds ≈1 F1 over CoT alone (80.5 vs 79.3 at 7B; table row match uncertain).
- **TIPS** ([2609.36641](https://arxiv.org/abs/2609.36641), Sept 2026): outcome-only RL. The judge is rewarded only
  when its predicted *outcome* label is right, and step-level verification emerges.
  - 85.2 ProcessBench F1 at 4B.
  - On AgentProcessBench, first-error accuracy is 43.7 for base Qwen3-4B-Thinking and 49.1 with TIPS (§8).
- **DeepCritic** ([2505.00662](https://arxiv.org/abs/2505.00662)) and others: same family, 7B and larger.

**Cost.** Generative judges cost 1K to 5K output tokens per verified trace, times *k* if majority-voted (GenPRM
maj@8). On our 4-core CPU, a 1.5B to 4B model decodes at roughly 5 to 20 tokens/s **[estimate, not measured]**,
so one verification takes **minutes**. They are not viable as the default per-request path on CPU. On a GPU with
vLLM they take seconds. They are the right **escalation tier** and the right **offline label generator** for
training our cheap probes (as UHeads and Lessons do).

---

## 3. Self-verification from hidden states (white-box)

| Paper | Setting | Result |
|---|---|---|
| **Reasoning models know when they're right** (Zhang et al., [2504.05419](https://arxiv.org/abs/2504.05419)) | Probe (mostly *linear*) on the last-layer hidden state at the end of each reasoning chunk (intermediate answer). R1-Distill 1.5B to 70B, QwQ-32B. ≤1,000 training problems | In-distribution AUROC >0.7 everywhere; ECE 0.01 to 0.14. **Transfer (R1-Qwen-1.5B):** GSM8K→MATH 0.90, MATH→AIME 0.84, MATH→KnowLogic 0.63, KnowLogic→GSM8K 0.67. Llama-8B: GSM8K→KnowLogic 0.56. **Early exit:** 19 to 24% fewer tokens at equal accuracy (MATH, 88.6 → 88.2%). Probes on a *non-reasoning* instruct model are "much worse" |
| **Hidden error awareness** ([2605.09502](https://arxiv.org/abs/2605.09502), 2026) | Linear probe on trace correctness, Qwen2.5 1.5B to 72B, Llama-8B, R1-7B, MATH | Qwen2.5-3B: **0.953** (held-out 200 problems) against self-consistency@5 0.823, P(True) 0.721, verbalized 0.674, sequence log-prob 0.676. Already **0.917 with 10 training problems**. From the first step: 0.787, against **0.59 for a TF-IDF text classifier** on the same step. But the effect within a problem is modest (Cohen's d 0.55 to 0.69, 14B: 0.13). **MATH→ARC transfer: 0.54 to 0.55.** Steering or patching with the probe direction does nothing: the signal is "diagnostic, not causal" |
| **Temporal predictors of outcome** ([2511.14773](https://arxiv.org/abs/2511.14773)) | Qwen3-8B, MATH | AUROC **0.84 at t=4 tokens** into the trace. The entropy-plus-length baseline is 0.59. **This is mostly difficulty prediction**: 0.77 on easy items vs 0.72 on hard ones at t=4 |
| **LiLaVe** ([2504.16760](https://arxiv.org/abs/2504.16760)) | XGBoost on the generator's hidden states, math | Beats LLM verifiers in AUC; "over 90% less verification compute" **[unverified]**. Proposes *conditional* majority voting: sample more only when the score is low |
| **UHeads / ReProbe** ([2511.06209](https://arxiv.org/abs/2511.06209), ACL 2026) | Head of under 10M parameters on attention to previous tokens plus top-K logits, frozen Qwen3-8B. 32K PRM800K chains labelled by DeepSeek-R1 (>$2K) or self-labelled | Step PR-AUC overall 0.559, the same as Qwen2.5-Math-7B-PRM800K (0.559) and close to Qwen2.5-Math-PRM-7B (0.565). Better than PRMs out of domain (planning, about 0.74 to 0.80). Self-labels work as well as R1 labels. Adding the head to a PRM gives +0.03 to 0.06. A head trained on Qwen3-8B data works on **Phi-4** traces too (0.448 vs 0.474) |
| **CRV** (Meta, [2510.09312](https://arxiv.org/abs/2510.09312)) | Attribution-graph features via transcoders, Llama-3.1-8B, step-level | GSM8K step AUROC **70.2**, against 55.9 for a **linear probe**, 62.6 for energy and 56.7 for entropy. Cross-domain transfer often falls *below* training-free baselines (Arithmetic→GSM8K 57.0) |
| **Chain-of-Embedding** ([2410.13640](https://arxiv.org/abs/2410.13640)) | Label-free trajectory of hidden states across layers | Training-free baseline; weak in CRV's tables (≈51 to 69 AUROC) |

Reading:

1. **Answer-level** probes on reasoning models are strong (0.8 to 0.95) and need very few labels.
2. **Step-level** linear probes are much weaker (≈0.55 to 0.70 AUROC on GSM8K steps in CRV). Small nonlinear heads
   with attention and logit features (UHeads) do better.
3. Much of the signal is *problem difficulty*. That is still useful for flagging risk, but less so for
   distinguishing two answers to the same question.
4. All of these probe the **generator's own** states. Whether a separate **reader** reading another model's trace
   carries the same signal is *untested in this literature*. Our RAG results suggest a reader works for grounding.
   Discriminative PRMs are, in effect, *fine-tuned* readers, so a frozen reader plus a probe should land between
   a text classifier and a PRM. This is the central open question, and experiment E1 tests it.

---

## 4. Confidence- and sampling-based verifiers

- **DeepConf** (Fu et al., [2508.15260](https://arxiv.org/abs/2508.15260)). Token confidence is the negative mean
  log-probability of the top-k tokens. Trace scores:
  - *bottom-10% group confidence* over sliding windows of 1–2K tokens;
  - *lowest group confidence* (2K window);
  - *tail confidence* (last 2K tokens).

  Results:
  - **Offline, K=512:** keeping the top 10% most confident traces before voting gave on average 84.4 to 84.6% vs
    83.0% for majority voting, over 23 model–dataset pairs. Best case: GPT-OSS-120B on AIME25, 97.0 → 99.9%.
  - **Online:** early-stopping low-confidence traces saves 43 to 85% of tokens (DeepConf-low).
  - **Regressions:** in 8 of 20 settings with aggressive filtering (−0.3 to −4.7 pp), all from "confidently
    wrong" traces. Confidence over the *first* 10% of tokens is weak: 81.9 vs 83.0.
  - **No single-trace AUROC is reported**; the separation is only shown in a histogram. As a per-trace
    verifier, DeepConf is unquantified. It is a *voting filter*.
- **Self-consistency and verbalized confidence**
  ([2603.19118](https://arxiv.org/abs/2603.19118), 2026; gpt-oss-20b, Qwen3-30B-A3B, R1-0528-8B; 17 tasks),
  macro AUROC:

  | Domain | Verbalized, K=1 | Self-consistency, K=2 / 8 | **Hybrid, K=2 / 8** |
  |---|---|---|---|
  | Math | 71.3 | 70.6 / 79.4 | **84.2 / 88.4** |
  | STEM | 73.8 | 66.6 / 75.5 | 80.2 / 82.0 |
  | Humanities | 68.5 | 63.3 / 71.3 | 74.9 / 77.0 |

  Two samples plus a verbalized confidence capture most of the gain. Agreement-only signals are weak in some
  settings: [2607.08065](https://arxiv.org/abs/2607.08065) finds agreement "weakly correlated" with correctness,
  and reports vote-agreement AUROC at or below chance in some LLM-judge cells.
- **Verbalized confidence** of reasoning models is better calibrated than that of non-reasoning models in 33 of 36
  settings ([2505.14489](https://arxiv.org/abs/2505.14489)), but still overconfident on hard tasks. Asking the
  closed model for an extra confidence line, or reading one it already gives, is nearly free.
- **Length.** Within a question, the shortest of the sampled chains is up to 34.5% more accurate than the longest
  ([2505.17813](https://arxiv.org/abs/2505.17813)). Long traces with many "wait" or backtracking markers are a free
  risk feature. It is confounded by difficulty, but can be compared against a per-domain baseline.
- **Self-verification by the same model** is unreliable when errors are correlated. Same-model SymPy self-checks
  falsely accept 13.8% of wrong MATH-500 answers, rising to 30.8% at Level 5
  ([ICML 2026 workshop](https://icml.cc/virtual/2026/78011)). Earlier, BIG-Bench Mistake
  ([2311.08516](https://arxiv.org/abs/2311.08516)) found LLMs cannot reliably *find* errors but can fix them once
  told where; a small trained classifier beat prompting a large model for locating them.

---

## 5. Step-level uncertainty and where errors live

- **Forking tokens.** About 20% of CoT tokens have high entropy and act as "forks"; the rest are near-deterministic
  ([2506.01939](https://arxiv.org/abs/2506.01939)). For a verifier, per-token signals should be read at forks
  (step starts, connectives, numbers), not averaged over everything. That is DeepConf's finding too: windowed
  minima beat means.
- **Thought anchors** ([2506.19143](https://arxiv.org/abs/2506.19143)). Planning and uncertainty-management
  ("wait", backtracking) sentences carry the most counterfactual weight on the final answer, more than computation
  or fact-retrieval sentences. Measuring that needs counterfactual resampling (≈100 rollouts per sentence), which
  is far too expensive for verification. The attention-based variant ("receiver heads") is a cheap proxy that could
  be read from our reader **[untested for verification]**.
- **Implication for localisation.** Errors split into two groups:
  - *local* errors (arithmetic, algebra, a wrong fact): checkable exactly, or visible as a surprisal or probe dip
    at the step;
  - *strategic* errors (a wrong plan or a wrong case split): show up at anchor sentences, and only a judge that
    understands the problem catches them.

  Cheap methods cover the first group well and the second poorly.

---

## 6. Formal tools as verifiers

| Tool | Coverage | Precision | Cost |
|---|---|---|---|
| Arithmetic re-computation (`vt/mathcheck.py`) | Stated `a op b = c` equations only | ≈100% | μs |
| SymPy equivalence of stated equations / final answer against a reference | Algebraic steps, closed-form answers | High; parsing is the failure mode | ms |
| Code execution (parse, compile, run customer or generated tests) | Code answers | High for failures; generated tests can be wrong | s, sandboxed |
| LLM-written SymPy `verify(answer)` from the problem only, by a *different model family* (X-SGRV, [ICML'26 WS](https://icml.cc/virtual/2026/78011)) | 34.3% of MATH-500; **3 to 12%** of post-cutoff competition problems | 171/171 accepted correct (CI ≥0.979) | One LLM call plus sandbox |
| Autoformalization to Isabelle / Lean (DTV, [2403.18120](https://arxiv.org/abs/2403.18120)) | Problems that formalize | Voting over formally verified solutions beats majority voting by >12% on GSM8K | Heavy (large LLM, prover) |
| CodeRM-8B unit-test generation ([HF](https://huggingface.co/KAKA22/CodeRM-8B)) | Code | Best-of-N on HumanEval+ with Llama3-8B rises from 53.6 to 72.0 | 8B generation |

Formal checks give **certificates** (the `proved` level of `vt/pcr`), but low *recall* outside arithmetic and code.
How often ProcessBench first errors are arithmetic slips is **unknown**; that is experiment E2.

---

## 7. Faithfulness: verifying the answer vs verifying the reasoning

- Reasoning models often do not say what drove their answer. Claude 3.7 Sonnet mentions a used hint in 25% of
  cases and DeepSeek-R1 in 39%; unfaithful CoTs are *longer*
  ([2505.05410](https://arxiv.org/abs/2505.05410),
  [Anthropic post](https://www.anthropic.com/research/reasoning-models-dont-say-think)).
- PRMs reward correct answers reached through flawed steps. Even Qwen2.5-Math-PRM-72B catches only 58% of
  "right answer, wrong process" cases (§1.2).
- Closed reasoning APIs often hide the raw thinking or return only a summary. For those models the verifier sees
  the final answer and its *presented* justification, not the real trace.

**Product consequence.** Report two separate things:

- **answer risk**: P(final answer wrong), the thing customers pay for;
- **justification issues**: located step problems in the *visible* text.

Never claim that a clean visible trace means the model "reasoned correctly". For agent traces the visible actions
and tool I/O are the ground truth, so faithfulness is less of an issue there: verify actions against observations.

---

## 8. Agent trajectories and tool calls

| Benchmark | Data | Best reported |
|---|---|---|
| **Who&When** ([2505.00212](https://arxiv.org/abs/2505.00212), ICML'25; [HF](https://huggingface.co/datasets/Kevin355/Who_and_When)) | Failure logs from 127 multi-agent systems | Responsible agent 53.5%; **decisive step 14.2%**; some methods below random |
| **TRAIL** ([2505.08638](https://arxiv.org/abs/2505.08638); [HF](https://huggingface.co/datasets/PatronusAI/TRAIL), evaluation-only licence) | 148 GAIA and SWE-bench traces, 841 errors | Gemini 2.5 Pro **11%** joint location-and-type accuracy |
| **AgentProcessBench** (via TIPS; [HF](https://huggingface.co/datasets/LulaCola/AgentProcessBench)) | HotpotQA, GAIA, BFCL, τ²-bench | First-error accuracy: Qwen3-4B-Thinking 43.7, **TIPS-4B 49.1** |
| **ToolPRMBench** ([2601.12294](https://arxiv.org/abs/2601.12294)) and ToolPRM ([2510.14703](https://arxiv.org/abs/2510.14703)) | Tool-use step rewards | Larger, generally capable models do better; ToolPRM beats ORMs on function calls **[numbers not read]** |

Cheap verification for agents should lean on what is *checkable*:

- tool-call schema and argument validation against the tool spec;
- comparing claims in the final answer with the tool outputs actually returned in the trace. This is exactly the
  grounding verifier we already have, with tool outputs as the evidence;
- detecting loops and repeated failed calls;
- exit codes and test results for code agents.

Learned step attribution in long agent traces is unsolved even for frontier models. Ship it as "suspect steps",
not as a verdict.

---

## 9. Cost and latency summary (per verified trace of about 1 to 4K tokens)

| Method | Needs | Extra compute | CPU (4-core) feasibility | Answer-level quality | Step localisation |
|---|---|---|---|---|---|
| Exact tools (arithmetic, SymPy, execution) | Text | ~0 | Yes | Precise, low recall | Exact for covered steps |
| Length, backtracking markers, verbalized confidence | Text | 0 | Yes | Weak (~0.6 to 0.7 AUROC) **[estimate]** | No |
| Generator logprob stats (DeepConf tail / lowest group) | Generator logprobs | 0 | Yes | Unquantified per trace; helps voting | Coarse (window) |
| Generator hidden-state probe | Generator internals | ~0 (one dot product) | Yes | **0.8 to 0.95** in-domain math; poor cross-domain | Weak (linear); better with UHead-style heads |
| **Frozen reader + probe (ours)** | Text | One forward pass of a 0.6 to 4B reader | Yes: ≈3 to 15 s **[estimate from our 6 s per 2-pass RAG reply; long traces cost more]** | **Unknown (E1)** | Unknown (E1) |
| Discriminative PRM ≤1.5B (Skywork-1.5B) | Text | One forward pass | Yes | ProcessBench 36 F1 (weak) | Weak |
| Discriminative PRM 7B (Qwen2.5-Math-PRM-7B) | Text | One forward pass of 7B | Slow but feasible (int8) | 73.5 F1, math only | Good |
| Self-consistency plus verbalized confidence, K=2 to 8 | K generations | K× generation | Only if the customer samples | **84 to 88 math, 77 to 82 other** | No |
| Generative PRM / judge (ThinkPRM-1.5B, GenPRM, TIPS-4B) | Text | 1 to 5K generated tokens (×8 if voted) | No (minutes); GPU tier | Best (TIPS-4B 85 F1) | Best |
| Frontier API judge (o1-mini class) | Text, $ | API call | Remote | 88 F1 | Good; collapses on Hard2Verify |

---

## 10. Design: the reasoning arm

Input: question, reasoning text (CoT, solution, or agent trace), final answer. Optional: generator logprobs,
generator hidden states, *k* extra samples, reference answer, tests.

```
Segment into steps (blank lines / "Step n" / sentence + equation boundaries; agent: one action+observation = one step)
Tier 0  exact checks (free, certificate-grade)
        - mathcheck arithmetic, SymPy equality/simplify of stated equations, unit sanity
        - final-answer extraction; SymPy match vs reference if customer supplies one
        - code: parse/compile, run customer tests in sandbox; agent: tool-arg schema, tool-output vs claim grounding
        -> any failure: verdict=wrong (proved), first_error=that step
Tier 1  one forward pass of the frozen reader over [question + trace]   (CPU, seconds)
        - answer probe on hidden state at the final-answer token  -> P(wrong)
        - step probe at each step-end token (linear first; UHead-style small MLP on hidden+top-k logits if linear is weak)
        - reader surprisal: lowest windowed mean log-prob, tail confidence, top-k entropy at "fork" positions
        - free text features: length vs domain baseline, # backtracks ("wait", "let me recheck"), verbalized confidence
        -> combine (logistic, per-domain scales) -> P(answer wrong); first_suspect_step = first step with step score
           below threshold, else argmin
Tier 1b if the customer provides them (zero extra cost for us)
        - generator logprobs -> DeepConf tail/lowest-group confidence
        - k samples -> answer agreement + verbalized confidence hybrid
        - generator hidden states (self-hosted open model) -> customer-side probe, ~free
Tier 2  escalation for the uncertain band only (GPU or API): generative step judge
        (TIPS-style Qwen3-4B, or ThinkPRM-1.5B / GenPRM-1.5B) -> first error step + explanation
Calibrate  per domain with the existing pipeline: threshold at a chosen false-alarm rate on 100–500 customer
           labels; conformal guarantee reported as "statistical", Tier-0 as "proved".
Report  answer_risk (calibrated), first_suspect_step (+ evidence: which check/feature), justification_issues;
        never "reasoning verified" from a clean trace.
```

Labels for training the Tier-1 probes, without human annotation:

- outcome labels from answer matching on GSM8K, MATH and OpenR1-Math traces (math_verify);
- PRM800K human step labels;
- consensus-filtered labels (MC rollouts agreeing with a judge), per the Lessons paper;
- a GPU-run generative judge (TIPS or ThinkPRM) labelling a few thousand traces offline, as UHeads did with R1;
  self-labels worked as well as R1 labels.

---

## 11. CPU experiments, ranked by expected value

Assumptions: a 4-core CPU; readers Qwen3-4B (current), Qwen3-1.7B and Qwen2.5-Math-1.5B; one forward pass per item
caching several layers; about 3 to 10 s per ~1K-token item **[estimate]**. Datasets were checked on the HF API.

1. **E1. Frozen-reader probe on ProcessBench: answer and step level.** *Highest EV; answers whether our
   architecture works for reasoning at all.*
   - Data: [`Qwen/ProcessBench`](https://huggingface.co/datasets/Qwen/ProcessBench) (about 3.4K items with step
     labels and `final_answer_correct`), plus a training source with no overlap:
     [`tasksource/PRM800K`](https://huggingface.co/datasets/tasksource/PRM800K) phase2, about 2K solutions.
   - Train linear (and small MLP) probes on step-end and final-token states from 3 to 4 layers.
   - Metrics: answer AUROC; step AUROC; ProcessBench F1 with the rule "first step below τ".
   - Baselines: reader surprisal features, a TF-IDF text classifier (the 0.59 reference), and Skywork-PRM-1.5B run
     on the same subsample ([HF](https://huggingface.co/Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B), CPU-able).
     Quote Qwen2.5-Math-PRM-7B's 73.5 from the paper rather than running it.
   - Cost: 1,000 items × ~6 s ≈ 2 h, plus 2K training items ≈ 3 to 4 h.
   - Success: answer AUROC ≥0.80 on Olympiad/Omni and step F1 >36 (beat Skywork-1.5B). Stretch: >56 (the
     PRM800K-7B level).
   - Also report AUROC on the "correct answer, flawed process" subset.
2. **E2. Exact-tool coverage audit.** *Near-zero cost; sets the ceiling of the "proved" tier.*
   - Run `mathcheck` plus a SymPy equation checker over every ProcessBench step, GSM8K/MATH-500 model solutions,
     and MR-Ben math.
   - Measure: share of erroneous solutions with at least one exact failure, precision of first-error localisation,
     and false alarms on correct solutions.
   - Cost: minutes. Decides how much Tier 0 contributes and which parser gaps to fix.
3. **E3. Cross-domain transfer.**
   - Train on math (E1); test on:
     - [`Randolphzeng/Mr-Ben`](https://huggingface.co/datasets/Randolphzeng/Mr-Ben): about 6K items over math,
       physics, chemistry, biology, medicine, logic and **coding**, with solutions from Mistral, **Claude** and
       GPT-3.5 and first-error labels. This is closed-model outputs, which is our use case.
     - [`UW-Madison-Lee-Lab/MMLU-Pro-CoT-Eval`](https://huggingface.co/datasets/UW-Madison-Lee-Lab/MMLU-Pro-CoT-Eval)
       (14 domains), and BIG-Bench Mistake (logical deduction, Dyck, word sorting, shuffled objects; GitHub
       `WHGTyen/BIG-Bench-Mistake`).
   - Then retrain with 50 / 100 / 300 in-domain labels to get the label-efficiency curve per domain.
   - Cost: ~3 to 5 h. Answers Q2 for *our* reader and sets the label count we promise customers.
4. **E4. Reader surprisal for localisation, DeepConf-style but read by the reader.**
   - From E1's cached forward passes, compute windowed minimum confidence, entropy at fork tokens and attention
     "receiver" scores per step.
   - Compare their first-error localisation, alone and combined with the step probe.
   - Cost: ~0 extra (same passes). Tells us whether the label-free signals add on top of the probes.
5. **E5. Long reasoning-model traces.**
   - Sample [`open-r1/OpenR1-Math-220k`](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k) items that
     have both correct and incorrect R1 generations (the data includes multiple generations with correctness
     flags). Use 300 problems with mixed outcomes, so the comparison is *within-problem*, controlling for
     difficulty.
   - Truncate to the last 2 to 4K tokens (tail confidence was the strongest DeepConf variant).
   - Measure answer AUROC overall *and* within-problem pairwise accuracy, to separate trace signal from
     difficulty.
   - Cost: ~3 to 6 h (long sequences). Tests the product's hardest realistic input.
6. **E6. Free text-only features for closed models.**
   - Features: length relative to a per-dataset median, backtrack-marker counts, verbalized-confidence parsing,
     and the final-answer format.
   - Run them as a logistic baseline on E1, E3 and E5. Run cross-model agreement with a cheap second solver only
     on GSM8K-level items: Qwen3-1.7B non-thinking, ~200 tokens, ~30 s per item on CPU **[estimate]**.
   - Gives the "no internals, no resampling" floor.
7. **E7. Agent traces.**
   - Use the probe and grounding verifier on
     [`Kevin355/Who_and_When`](https://huggingface.co/datasets/Kevin355/Who_and_When) (184 traces) and
     [`LulaCola/AgentProcessBench`](https://huggingface.co/datasets/LulaCola/AgentProcessBench).
   - Treat tool observations as evidence for the existing grounding verifier, and score the steps.
   - Metrics: first-error accuracy against 14.2% (Who&When) and 43.7 to 49.1 (AgentProcessBench, 4B judges).
   - Cost: ~2 to 4 h. Low expected accuracy, but it shows whether the grounding arm reuses directly.
8. **E8. Generative judge as an escalation tier and label source** (GPU, e.g. one Kaggle or Modal session; not
   CPU).
   - Run ThinkPRM-1.5B / GenPRM-1.5B / prompted Qwen3-4B-Thinking on the E1 subsample.
   - Measure cost per trace and the gain on the uncertain band only.
   - Use its labels to train E1 probes on new domains.
   - If budget allows, reproduce TIPS (RL on 3.2K outcome labels) on Qwen3-4B. That is the highest-quality small
     judge known, but RL is GPU work.

Not recommended on our compute:

- training a 7B PRM;
- Lean autoformalization (heavy, low coverage on word problems);
- thought-anchor resampling (≈100 rollouts per sentence);
- CRV transcoders (needs per-model transcoder training, and transfers poorly).

---

## 12. Main uncertainties

- **Reader vs generator.** All strong probe results probe the *generator's own* hidden states. A frozen reader
  reading another model's reasoning is unmeasured in this literature, and E1 decides it. The UHead Phi-4 transfer
  (a head trained on Qwen3-8B data still works on Phi-4 traces) and our RAG results are weak positive evidence.
- **Difficulty confound.** Probe AUROCs of 0.8 to 0.95 partly measure "is this problem hard". A verifier that only
  knows difficulty flags hard questions, not wrong answers. Always report within-problem metrics (E5).
- **Long traces on CPU.** 8 to 30K-token reasoning traces make even one forward pass of a 4B model slow (tens of
  seconds or more, **[estimate]**). Use tail truncation and a smaller reader (Qwen3-1.7B or 0.6B), and measure it.
- **Benchmark validity.** ProcessBench scores overstate ability on proofs and frontier problems (Hard2Verify:
  78 → 37). Best-of-N overstates PRM process quality (Lessons).
- **Unread numbers.** The ORM and majority-vote baselines in Lightman et al., LiLaVe's "90% compute", CRV's
  secondary claims, ToolPRM's numbers and DeepConf per-trace separation were not read from primary tables. The
  TIPS paper (Sept 2026) is very recent and not yet replicated.

## Sources

- [Lightman et al. 2023, Let's Verify Step by Step](https://arxiv.org/abs/2305.20050)
- [Zhang et al. 2025, The Lessons of Developing PRMs](https://arxiv.org/abs/2501.07301)
- [ProcessBench](https://arxiv.org/abs/2412.06559)
- [PRMBench](https://arxiv.org/abs/2501.03124)
- [Hard2Verify](https://arxiv.org/abs/2510.13744)
- [GenRM](https://arxiv.org/abs/2408.15240)
- [ThinkPRM](https://arxiv.org/abs/2504.16828)
- [GenPRM](https://arxiv.org/abs/2504.00891)
- [TIPS](https://arxiv.org/abs/2609.36641)
- [VersaPRM](https://arxiv.org/abs/2502.06737): math PRMs no better than baseline on law, philosophy and biology;
  +7.9% on law vs Qwen2.5-Math-PRM's +1.3%
- [ReasonFlux-PRM](https://arxiv.org/abs/2506.18896)
- [Skywork-PRM-1.5B](https://huggingface.co/Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B)
- [Reasoning Models Know When They're Right](https://arxiv.org/abs/2504.05419)
- [Hidden Error Awareness](https://arxiv.org/abs/2605.09502)
- [Temporal predictors](https://arxiv.org/abs/2511.14773)
- [LiLaVe](https://arxiv.org/abs/2504.16760)
- [UHeads](https://arxiv.org/abs/2511.06209)
- [CRV](https://arxiv.org/abs/2510.09312)
- [Chain-of-Embedding](https://arxiv.org/abs/2410.13640)
- [DeepConf](https://arxiv.org/abs/2508.15260)
- [SC + verbalized confidence scaling](https://arxiv.org/abs/2603.19118)
- [Agreement audit](https://arxiv.org/abs/2607.08065)
- [Reasoning models express confidence](https://arxiv.org/abs/2505.14489)
- [Don't Overthink It](https://arxiv.org/abs/2505.17813)
- [High-entropy forking tokens](https://arxiv.org/abs/2506.01939)
- [Thought Anchors](https://arxiv.org/abs/2506.19143)
- [CoT faithfulness](https://arxiv.org/abs/2505.05410)
- [BIG-Bench Mistake](https://arxiv.org/abs/2311.08516)
- [MR-Ben](https://arxiv.org/abs/2406.13975)
- [Don't Trust: Verify](https://arxiv.org/abs/2403.18120)
- [X-SGRV](https://icml.cc/virtual/2026/78011)
- [SymCode](https://arxiv.org/abs/2510.25975)
- [CodeRM-8B](https://huggingface.co/KAKA22/CodeRM-8B)
- [Who&When](https://arxiv.org/abs/2505.00212)
- [TRAIL](https://arxiv.org/abs/2505.08638)
- [ToolPRMBench](https://arxiv.org/abs/2601.12294)
