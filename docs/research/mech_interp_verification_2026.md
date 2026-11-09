# Mechanistic-interpretability signals for verification we have not tried (2024-2026 survey, ranked)

Research note, 9 November 2026. It follows [mech_interp.md](mech_interp.md) and
[internal_state_probes.md](internal_state_probes.md) and does not repeat them. It looks only for signals that the omni verifier
(docs/omni_verifier.md) has **not** tested. The list of what was tested comes from the "What moved accuracy, and what did not"
table and docs/omni_experiments/.

**Confidence tags.**
- **[html]**: I read the paper's own text or tables through the arXiv HTML or ACL PDF.
- **[abstract]**: the number comes from the abstract or a search snippet only.
- **[secondary]**: the number comes from someone else's summary.

Numbers are as the authors report them and none is replicated, unless stated otherwise. Most 2026 items are single preprints.

**The structural constraint (unchanged).** Our reader reads *other models'* text. Signals about the reader's next-token confidence
are at chance (token log-prob 0.49). The 2026 evidence below that matters most says this is less of a handicap than feared:
- On random samples, a peer model's representation predicts a target model's correctness as well as the target's own does
  ([Masked by Consensus, ACL 2026](https://arxiv.org/abs/2604.12373)).
- In math the peer is never worse.

So signals that read *what the reader thinks of the text* are fine. Signals that read *how the reader would have generated
it* are not.

---

## 0. Ranked shortlist

What the reader exposes today (`vt/omni.py: Reader.features`):
- the mean residual state over the reply at six layers;
- per-unit means;
- the judge-position residual at the same layers;
- the final Yes/No log-odds;
- final-layer token log-prob, entropy and margin.

**Not exposed today:**
- any state at the end of the question, before the reply;
- any per-layer readout through the unembedding;
- any attention;
- any gradient;
- any second framing of the judge question.

Every idea ranked 1-3 below needs no extra forward pass, or only a few cached suffix tokens.

| # | idea | extra cost (1.7B, CPU) | where it should help | expected Δ AUROC, all 21, zero-label (my prior) | P(useful) |
|---|---|---|---|---|---|
| 1 | **Verdict logit-lens trajectory** at the judge position (per layer, per path: Yes−No log-odds, entropy, commit depth) | ≈ +0.2 s per reply (84 unembeddings of one position) | all formats, especially the judge expert; label-free readout | +0.005 to +0.02 | 0.45 |
| 2 | **Question-only prior state + posterior shift** (state at the token before the reply; judge state − prior state) | 0 (a slice of the existing pass) | closed-book chat, long-form world knowledge | +0.01 to +0.03 on chat/long-form, ≈ 0 on paired domains | 0.4 |
| 3 | **Judge-framing ensemble + fragility** (5 appended verdict questions on a cached prefix, including polarity-flipped ones; mean, variance, polarity inconsistency) | ≈ +15% (5 × ~30 suffix tokens with KV cache) | chat, RAG | +0.01 to +0.02 | 0.4 |
| 4 | **Second-family peer reader** (OLMo-2-1B / SmolLM3-3B / granite-3.3-2B states + its own judge as one more expert) | +60-150% | chat, RAG | +0.01 to +0.03 | 0.5 |
| 5 | **Verdict attribution to units** (attention-to-unit from the judge row; attribution patching ∂judge/∂residual × residual per unit; mask-ablation of the top-3 units) | attention: ≈ 0; AtP: +1 backward (≈ 2-3× forward) | **localisation** (ProcessBench first wrong step, RAG span); detection slightly | detection +0.00 to +0.01; exact first-error localisation +3 to +8 points | 0.35 |
| 6 | **Attention-graph spectral features** (LapEigvals: per-head degree-normalised attention Laplacian diagonal, top-k, on reply and judge rows) | needs eager attention on a few layers; +30-60% | RAG in-domain; transfer doubtful | +0.00 to +0.01 out of domain | 0.2 |

Below the line: TSV, semantic-entropy probes, soft-prompt probes (PEP), Latent-Critic LoRA, attribution-graph CRV,
EPGS/CausalGaze gradient sensitivity as a detector, local intrinsic dimension, entropy and "known entity" neurons, verification
heads. §3 explains why each was not ranked or is already covered.

---

## 1. The ranked ideas in detail

### 1. Verdict logit-lens trajectory ("when does the reader commit to No?")

**Signal.** We already read the judge state at six layers and fit a probe on it: 0.688 against 0.639 for the final log-odds at
1.7B. The probe beating the output says intermediate layers carry verdict information that the final layer partly loses or
overwrites. The logit lens reads that information **through the model's own unembedding, with no fitted direction**. It is
therefore a label-free readout, and label-free readouts are the kind that have transferred in this repo (the judge log-odds,
per-domain z-scores).

**Recipe** (TriLens + commitment ratio, adapted to one position):
- Take the judge position (last token of the appended question).
- At every layer ℓ (28 for 1.7B), take three vectors: the residual x_ℓ, the attention output a_ℓ and the MLP output m_ℓ.
  Hooks on `layer.self_attn` and `layer.mlp`.
- Apply the final RMSNorm (`model.model.norm`) and then `lm_head`.
- Read:
  - (i) κ_ℓ = logsumexp(Yes ids) − logsumexp(No ids), the same token sets as `Reader.yes/no`;
  - (ii) the Shannon entropy H_ℓ of the full softmax.

  That is 6 × 28 = 168 scalars.
- Label-free summaries:
  - mean κ over the middle band (layers 12-22);
  - min κ over the band;
  - **commit depth**: the first layer after which sign(κ) never changes;
  - the number of sign flips;
  - the final κ minus the mid-band κ (did the late layers overturn a mid-layer "No"?).
- Optional: the same κ at each **unit's last token**, using the appended judge question only for the reply. This is free, but
  without the judge suffix it is a per-token next-token readout, so it is low prior.

**Evidence.**
- **TriLens** ([arXiv 2606.01033](https://arxiv.org/abs/2606.01033), May 2026) **[html]**: per-layer logit-lens entropy of the
  attention, MLP and residual paths, 3L features, then an MLP or logistic probe.
  - Qwen2.5-7B, Llama-3-8B and Gemma-2-9B on HaluEval, SQuAD2, HotpotQA and TriviaQA. MLP 0.81-0.94; **linear 0.78-0.93**.
  - It beats the ICR Probe in 12/12 cells (+12.1 on average) and SAPLMA in 11/12.
  - **Cross-dataset: off-diagonal mean 0.848, against 0.738 for the ICR Probe and 0.678 for SAPLMA.** One probe on all four
    datasets loses only 1.4 points.
  - Caveat: these are the model's own short answers (≤ 32 tokens), entropy at the scored tokens, and no raw-hidden-state linear
    probe baseline. The transfer is between four short-QA datasets, much closer to each other than our 21 domains.
- **Commitment ratio** ([arXiv 2603.13259](https://arxiv.org/abs/2603.13259), Feb 2026) **[html]**: a normalised logit lens on
  forced correct vs incorrect single tokens.
  - Mid-layer probes peak at depth 0.2-0.5 (accuracy 0.79-0.85, with the final layer 0.72-0.76).
  - Cross-domain AUROC 0.80-0.87 at mid depth.
  - Small (300 queries); **Qwen2-1.5B was at chance (0.50)**, a warning for small Qwen readers.
- **Judges.** [Will the Judge Flip?](https://arxiv.org/abs/2610.07115) (Oct 2026) **[html]**:
  - A probe on the pre-verdict residual of Qwen3-1.7B/4B/8B judges predicts verdict flips at 0.64/0.76/0.85 AUROC.
  - That is +0.06 to +0.11 over the logit margin.
  - Frozen transfer to MT-Bench: 0.69/0.74/0.85.
  - So the pre-verdict state carries reliability information that the output margin lacks, and it grows with judge size.
- [Calibrating LLM Judges](https://arxiv.org/abs/2512.22245) (FAIR, Dec 2025) **[abstract]**: linear probes on judge states give
  better calibration than baselines at about 10× less compute, and "generalise to unseen domains"; the probes are conservative.

**OOD evidence for the logit-lens part specifically:**
- TriLens off-diagonal 0.848 (short QA only).
- Commitment-ratio cross-domain 0.80-0.87 (tiny).

**Cost.** 84 unembedding projections of one 2,048-d vector: 84 × 2048 × 151,936 ≈ 26 GFLOP, about 0.1-0.3 s on CPU per reply.
No storage issue.

**Concrete test** (Qwen3-1.7B, CPU, about 25 min):
1. Draw 1,008 replies from the corpus: 48 per domain × 21, label-balanced, fixed seed.
2. Extend `Reader.features` (in a copy under experiments/, without touching `vt/`) to also return `kappa[3, L]` and `H[3, L]` at
   the judge position.
3. Score, within each domain:
   - (a) each label-free summary alone, zero labels, sign fixed a priori (low κ = wrong);
   - (b) the 168-d vector into the LODO logistic with per-domain standardisation (`experiments/omni_eval.py` recipe, l2 0.05),
     against the existing judge-state probe on the same 1,008 replies;
   - (c) the best label-free summary z-summed with the stored expert scores (`results/omni/expert_scores_q17.npz`).
4. Paired bootstrap over replies, within domain.
5. **Kill** if (c) gains < +0.005 with the CI crossing 0.
6. **Go to 8B** (on the Kaggle GPU, same code) if (a) beats the final log-odds on ≥ 12/21 domains. The judge gains most from
   scale (0.64 → 0.78), so this signal should too.

**Expected value.** Moderate and cheap.
- The most likely win is a mid-band κ that beats the final log-odds as a zero-label judge expert. That would be a "better own
  judge" at zero cost, worth about +0.01 on the combined score.
- Risk: the commitment-ratio paper's Qwen2-1.5B null, and TriLens being on-policy.

### 2. Question-only prior and the prior → posterior shift

**Signal.** Before reading the reply, the reader's state at the end of the user turn encodes "how hard or risky is this
question". After the reply and the judge question, the judge state encodes "what I think of this answer". There are two signals:
- (a) **prior**: a probe on the question-end state predicting a wrong reply (a difficulty or hallucination-risk prior that does
  not depend on the reply's wording, so it is immune to fluent-but-wrong text);
- (b) **shift**: (judge state − prior state) and (reply-mean state − prior state), projected and probed. This measures how far
  the reply moved the reader.

This is the "model's prior vs posterior" contrast. Our evidence-shift test (documents vs no documents) was a different contrast
and failed at 1.7B.

**Recipe.**
- Take `o.hidden_states[L][0, r0 - 1]`, the last token before the reply. In Qwen3's template this is inside the assistant header
  (`<|im_start|>assistant\n`, plus an empty think block if present). Also take the last token of the user content, at the same
  six layers.
- It is free: it comes from the same forward pass.
- Fit LODO probes on:
  - (a) the prior state;
  - (b) the shifts;
  - (c) both added to the current reply head.

**Evidence.**
- **Masked by Consensus** (Ashuach, Gretz, Katz, Belinkov, Ein-Dor; [ACL 2026](https://aclanthology.org/2026.acl-long.483/),
  [arXiv 2604.12373](https://arxiv.org/abs/2604.12373)) **[html]**:
  - Setup: probes on the **question's final-token state** (every 5th layer) predict whether a target model's answer will be
    correct. Llama-3.1-8B, Qwen2.5-7B, Gemma-2-9B and Qwen3-32B; Mintaka, TriviaQA, HotpotQA, MATH, GSM1K.
  - **Peer-model probes match self-probes** on random samples: premium gap −0.025 to +0.021, none significant.
  - Self-probes gain only on factual disagreement subsets (+0.015 to +0.034 layer-averaged).
  - In math, peers are as good or better (GSM1K disagreement: −0.09 to −0.17).
  - The paper reports absolute AUCs in figures only, which I could not read.
  - **For us:** a reader's question state is a legitimate predictor of *another* model's correctness.
- **LLM Internal States Reveal Hallucination Risk Faced With a Query** (Ji et al., BlackboxNLP 2024,
  [arXiv 2407.03282](https://arxiv.org/abs/2407.03282)) **[abstract]**: a query-only probe gives 84.3% average hallucination-risk
  accuracy over 15 NLG tasks (own model).
- **Hallucination as Trajectory Commitment** ([arXiv 2604.15400](https://arxiv.org/abs/2604.15400)) **[html]**:
  - A prompt-final-token probe predicts "mostly hallucinating" prompts at AUROC 0.91-0.94 (Qwen2.5-1.5B).
  - Only 61 prompts, and within-category correlation is near zero (confabulation r = −0.05). The signal is mostly *between*
    categories, which is exactly what per-domain standardisation removes. Treat it as weak.
- [Temporal Predictors of Outcome in Reasoning LMs](https://arxiv.org/abs/2511.14773) **[abstract]** and
  [Hidden Error Awareness](https://arxiv.org/abs/2605.09502) **[abstract]**:
  - Correctness is predictable from the first few reasoning tokens (0.79 AUROC from step 1; 0.95 for the full trace, 9 models,
    1.5B-72B; R1 0.852).
  - Both are on-policy. They support "the prior is informative" for reasoning.

**Important caveat for our corpus.** Where the same question appears with both a correct and a wrong reply, the prior is
uninformative *by construction*. That covers HaluEval QA/dialogue pairs, nuance pairs, and possibly ProcessBench problems with
several solutions. Within-domain AUROC there will be about 0.5. Count question duplicates per domain before running. The prior
can only help where questions differ.

**Cost.** Zero. Six more 2,048-d vectors per reply in the feature file.

**Concrete test** (same 1,008 replies, about 17 min; shares the pass with test 1):
1. Store `q_state[6, d]`.
2. LODO with per-domain standardisation:
   - prior alone;
   - shift alone;
   - reply head + prior;
   - reply head + shift.
3. Report per format, and separately on domains with < 10% duplicated questions.
4. **Kill** if chat + long-form gain < +0.01.
5. If it works, add a fourth expert, "question risk", to `_combine` at weight 1, so a profile can learn to down-weight it on
   paired domains.

**Expected value.** The best free shot at the weakest slice, closed-book chat (0.61 at 1.7B). Masked by Consensus is strong
evidence that a peer's question representation predicts another model's errors. The open question is how much survives
per-domain standardisation and our label balancing.

### 3. Judge-framing ensemble and verdict fragility (a single-pass SpikeScore)

**Signal.** A reply that is truly correct should get a stable verdict across semantically equivalent and polarity-flipped
framings of the judge question. Fragile verdicts flag unreliable ones.

**Recipe.**
- Run the prefix (documents, question, reply) once with `use_cache=True`.
- Then run K = 5 suffixes on copies of the cache:
  1. "Is the reply above fully correct, accurate and supported?" (current)
  2. "Does the reply above contain any error?" (**polarity flipped**: read No − Yes)
  3. "Would an expert accept this reply as correct?"
  4. "Is every claim in the reply supported by the question and documents?"
  5. "Is the final answer in the reply right?"
- Features:
  - the mean signed log-odds (an ensemble);
  - its variance;
  - **polarity inconsistency**: P(correct) from framing 1 and P(error) from framing 2 both high;
  - the variance of the judge-state probe score across the K states.
- Polarity matters: Bürger et al. ("Truth is Universal", NeurIPS 2024) show truth has a general direction plus a polarity
  direction, and ignoring polarity breaks probes.

**Evidence.**
- **SpikeScore** (Deng et al., ICLR 2026, [arXiv 2601.19245](https://arxiv.org/abs/2601.19245)) **[secondary for the headline]**:
  - Generates K = 20 follow-up dialogue turns and takes the maximum second difference of a SAPLMA probe score along them.
  - Leave-one-domain-out mean AUROC ≈ 0.786 (Llama-3.1-8B) and 0.787 (Qwen3-14B), against about 0.74 for cross-domain-specialised
    methods and about 0.70 for training-free ones.
  - Instability is the most domain-invariant signal in its table.
  - It needs 20 generated turns. Ours replaces generation with appended framings, at about 1% of the cost.
- **Will the Judge Flip?** ([arXiv 2610.07115](https://arxiv.org/abs/2610.07115)) **[html]**: order-swap flips occur in 17-50% of
  Qwen3 judge verdicts and are predictable from pre-verdict states. Verdict instability is common and readable at our reader
  sizes.
- **All Verdicts are Not Equal** ([AACL 2026](https://aclanthology.org/2026.aacl-long.207)) **[abstract]**: a "trustworthy verdict
  rate" combining reproducibility, order invariance and accuracy.

**Repo caveat.** Judge errors are correlated (≈ 0.65), and prompt-ensembling one judge is weaker than adding a second model. The
*variance* is the new part, not the mean.

**Cost.** About 5 × 30 suffix tokens on a cached prefix: about +10-20% on CPU.

**Concrete test** (1,008 replies, about 25 min):
1. Score, zero-label, within each domain:
   - (a) the mean of the K framings against framing 1 alone;
   - (b) fragility alone;
   - (c) z-sum into the stored stack.
2. Also report AUROC on the 20% of replies where the reader's judge is least confident (|κ| smallest).
3. **Kill** if (c) gains < +0.005.

**Expected value.** Cheap. The mean is likely +0.005-0.01 (ensemble). Fragility is a genuine unknown with a credible ICLR
precedent.

### 4. A second-family "peer" reader (cross-model representation diversity)

**Signal.** The repo's consensus test failed because Qwen3-0.6B and 1.7B are near copies (canonical correlation 0.97). External
judges from *other families or sizes* (Qwen3-14B, phi-4) were the largest lever after reader size. A small reader from a
different pretraining lineage gives decorrelated hidden states at the reader tier's price.

**Recipe.**
- Run the same `Reader.features` on a non-Qwen open small model:
  - `allenai/OLMo-2-0425-1B-Instruct` (Apache-2.0);
  - `HuggingFaceTB/SmolLM3-3B` (Apache-2.0);
  - `ibm-granite/granite-3.3-2b-instruct` (Apache-2.0).

  Gemma and Llama are gated licences; check access first.
- Train its own reply head and read its own judge.
- Add both as experts with per-domain z-scores.

**Evidence.**
- [Masked by Consensus](https://arxiv.org/abs/2604.12373) **[html]**: peer probes ≈ self probes. Correctness patterns are shared
  across models, but not identically: the factual disagreement subsets exist.
- [The Truth Stays in the Family](https://arxiv.org/abs/2606.15821) **[abstract]**: raw cross-family truth-signal correlation is
  low (Pearson 0.10). Decorrelated errors are what an ensemble needs.
- [Too Consistent to Detect](https://arxiv.org/abs/2505.17656) (EMNLP 2025) **[abstract]**: self-consistent errors are
  model-specific. A **cross-model probe fusing an external verifier's hidden states** "significantly" improves their detection.
- [Prompt Embedding Probes](https://arxiv.org/abs/2608.08024) **[html]**: with detector ≠ generator, the largest detector is
  usually best, and every detector does best on its own model's generations.
- In-repo: external judges took the stack from 0.80 to 0.84.

**Cost.** OLMo-2-1B about 0.6 s per reply on CPU; SmolLM3-3B about 1.8 s.

**Concrete test.**
1. First, about 10 min: on the 1,008 replies, compute canonical correlation between the OLMo-2-1B and Qwen3-1.7B reply-mean
   states (PCA 128 each). If > 0.95, stop.
2. Otherwise run LODO on the 1,008 replies:
   - the Qwen head;
   - the OLMo head;
   - both concatenated;
   - the z-sum of the two heads plus both judges.
3. Use the ≈ 1-h budget for OLMo-2-1B. SmolLM3-3B needs about 2 h or the GPU.

**Expected value.** Highest probability of a real gain among the six, but at 1.6-2.5× the 1.7B tier's compute, and not "mechanistic"
in the strict sense. If it works, the budget tier becomes "two different 1-2B readers".

### 5. Verdict attribution to units: localisation first, detection second

**Signal.** Which sentence or step *causes* the reader's "No"? Today `suspect_unit` uses the first unit crossing 0.75 sd on the
unit head. 47% of ProcessBench first-wrong steps are found exactly.

A causal attribution of the judge's log-odds to units is a different, decorrelated localiser. Its shape gives two candidate
detection features:
- **concentration**: one unit carries the "No";
- **agreement** between the attribution's top unit and the unit head's suspect.

**Recipe** (three levels of cost):
- (a) **Judge attention** (free with eager attention at 3-4 middle layers): for each head, the attention mass from the judge
  position to each unit's tokens, normalised by unit length. Pick heads by label-free variance across units, or with the
  ProcessBench training labels.
- (b) **Attribution patching (AtP)** ([Kramár et al. 2024](https://arxiv.org/abs/2403.00745)):
  - one backward of κ (judge log-odds) to the residual at layer ℓ ∈ {10, 14, 18};
  - the per-token score is (∂κ/∂h_t) · h_t, or ·(h_t − μ_domain) to ablate toward the domain mean;
  - sum over each unit's tokens.
- (c) **Mask ablation** of the top-3 candidate units: a fresh pass with that unit's attention masked from all later positions;
  Δκ is the causal effect. About +3 forward passes.

**Evidence.**
- **CausalGaze** ([arXiv 2604.11087](https://arxiv.org/abs/2604.11087), Apr 2026) **[html]**:
  - Gradient-weighted attention edges (|A ⊙ ∇_A L|) into a GNN reach 0.84-0.93 AUROC on TruthfulQA, TriviaQA, SciQ and HaluEval
    (Llama-2, Qwen2, Mistral 7B).
  - The gains over the non-causal HaluGNN are small (0.000-0.05).
  - **Removing the gradient term drops HaluEval F1 from 0.80 to 0.63**: the causal-sensitivity part matters.
  - Cross-dataset "+3.6 F1 average", in a figure only.
  - The abstract's "+5.2%" does not match its own table.
- **EPGS, "From Flat Facts to Sharp Hallucinations"** ([arXiv 2605.00939](https://arxiv.org/abs/2605.00939)) **[html]**:
  - Gradient norm × (1 − cosine between clean and embedding-noised gradients), at the last block.
  - Beats entropy and consistency baselines in 12/12 cells (Llama-3 TriviaQA 0.81 against 0.72).
  - It is sensitive to the noise scale (10 points on Llama-2 SQuAD, against the claimed "< 2%") and has no OOD test.
  - It is on-policy and loses to entropy under SQuAD-F1 labels.
- **CRV** ([Zhao et al., ICLR 2026 oral, arXiv 2510.09312](https://arxiv.org/abs/2510.09312)) **[abstract]**: attribution-graph
  features predict step errors at 92.5 AUROC on synthetic arithmetic, against 76.5 for the best baseline. The prior survey records
  about 0.70 on GSM8K, with poor transfer.
- **Self-verification geometry** ([Lee et al., arXiv 2504.14379](https://arxiv.org/abs/2504.14379)) **[abstract]**:
  - In an R1-style Qwen2.5-3B, a few **previous-token heads** drive verification, and late-layer GLU vectors respond to "success"
    vs "incorrect".
  - Similar components appear in R1-Distill-Qwen-14B.
  - This motivates looking at heads whose judge-row attention lands on the erroneous step.
- **Latent Critic** ([arXiv 2608.10430](https://arxiv.org/abs/2608.10430)) **[html]**: patching the [POS] verdict state flips 96% of
  verdicts, with the effect concentrated in later layers. The verdict *is* readable causally at one position.

**Cost.**
- (a) About 0 (eager attention on 4 layers only, via hooks).
- (b) One backward. On CPU about 2-3× a forward, so about 600 replies per hour at ≤ 1,024 tokens.
- (c) +3 forwards.

**Concrete test** (ProcessBench units, 600 solutions, about 1 h with (b)):
1. Exact and within-1 first-wrong-step localisation for:
   - the shipped first-crossing rule (47% / 73%);
   - (a);
   - (b);
   - the rank-average of the shipped rule and (b).
2. Detection: add the concentration and agreement features to the reply head under LODO.
3. **Go** if exact localisation gains ≥ +3 points with a CI clear of 0. Localisation is a product feature (`suspect_unit`) whose
   value does not depend on the saturated combination layer.

**Expected value.** The best mechanistic idea for **localisation**; weak for detection.

### 6. Attention-graph spectral features (LapEigvals / TOHA family)

**Signal.**
- Each head's attention over the reply is a graph. The diagonal of its degree-normalised Laplacian (the eigenvalues, since the
  matrix is lower-triangular) summarises how diffusely each token attends.
- Top-k per head, PCA, then logistic.
- A training-free cousin, **TOHA**, scores the topological divergence between prompt and response attention graphs on about 10
  heads chosen with about 50 labels.

**Evidence.**
- **LapEigvals** ([EMNLP 2025, arXiv 2502.17598](https://arxiv.org/abs/2502.17598)) **[html]**:
  - Llama-3.1-8B, own answers: 0.79-0.89 AUROC on 7 QA sets (TriviaQA 0.889, GSM8K 0.872), against 0.49-0.72 for the unsupervised
    AttentionScore.
  - It "outperforms hidden-state probes in most settings" (appendix, not read).
  - Cross-dataset drops are "comparable to baselines". **All methods transfer poorly to TruthfulQA and GSM8K.**
- **TOHA** ([ACL 2026, arXiv 2504.10063](https://arxiv.org/abs/2504.10063)) **[abstract]**: RAG; heads chosen on one dataset keep
  working on others; about 10× faster than comparable methods.
- **Hallucination Is Linearly Decodable** ([arXiv 2606.02628](https://arxiv.org/abs/2606.02628)) **[secondary via alphaXiv]**:
  attention entropy reaches 0.87-0.94 on HaluEval-QA but ≤ 0.60 on TruthfulQA and ≤ 0.69 on FEVER. It helps only where knowledge
  is grounded.

**Repo prior.** Every pooled attention feature we tried (copy heads, source flow, reader consensus) failed to transfer. Lookback
Lens's per-head ratios were never tested in the omni LODO, and LapEigvals is the strongest per-head attention representation
published.

**Cost.** Eager attention on 4-6 layers via hooks; +30-60% CPU.

**Concrete test** (1,008 replies, about 40 min):
1. Features at 4 middle layers × 16 heads, top-k = 20 Laplacian diagonal values over the reply rows, plus the judge row, then
   PCA 128.
2. LODO alone and added to the reply head.
3. Include a per-head lookback ratio (attention to documents vs to the reply) on the RAG domains in the same run. It needs the
   same attention, and settles E5 from the prior survey at no extra cost.
4. **Kill** if LODO alone < 0.60 or the addition < +0.005.

**Expected value.** Low. Run it only because it shares the attention pass with idea 5(a).

---

## 2. Shared protocol for the CPU tests

- **One feature-extraction script** in experiments/ (a copy of `Reader.features` with flags) computes ideas 1, 2, 3, 5(a) and 6
  in **one pass** per reply. That keeps the 1.7B run near 45-60 min for 1,008 replies.
  - Use eager attention only on the 4 layers needed, through hooks.
  - Idea 5(b) and idea 4 are separate runs.
- **Sample.** 48 label-balanced replies per domain × 21 domains, seed 0, the same rows for every idea.
- **Baselines on the same rows:**
  - the shipped 1.7B combined score (from `expert_scores_q17.npz` if the rows are in it; otherwise recompute);
  - the reader judge log-odds alone.
- **Metric.** Within-domain AUROC, averaged over domains and per format, with a paired bootstrap within domain (1,000 resamples).
- **Additions** are tested two ways:
  - zero-label z-sum with an a-priori sign;
  - an extra column in the LODO logistic with per-domain standardisation.
- **The usual gate.** Stop on a gain < +0.005 with a CI crossing 0. Promote to the 8B GPU run only on a CI-clear gain.

---

## 3. Considered and not ranked (with reasons)

| idea | best evidence | why not ranked / status |
|---|---|---|
| **TSV** (truthfulness separator vector + OT pseudo-labels; [ICML 2025, arXiv 2503.01917](https://arxiv.org/abs/2503.01917)) | 0.84-0.86 with 32-64 labels, own outputs | Its two ingredients are already covered: the prior-pulled refit at 25-100 labels (+0.04-0.07), and OT pseudo-label re-estimation, already tried (±0.0). No 2026 follow-up found. |
| **Semantic-entropy probes** ([arXiv 2406.15927](https://arxiv.org/abs/2406.15927)) | +7.7-10.5 AUROC OOD over accuracy probes, own outputs | SE is a property of the *generator's* samples. For us the analogue is the reader's own SE on the question, a "does the reader know" gate. It needs reader sampling at training time and is a weaker version of idea 2. Latent Critic finds SEP at 0.50-0.55 on tool calls. |
| **Soft-prompt probes (PEP)** ([arXiv 2608.08024](https://arxiv.org/abs/2608.08024)) **[html]** | Qwen3-8B: +0.03-0.11 over a linear probe in distribution | **OOD (leave one dataset out): 0.53-0.63, worse than linear on GSM8K.** A learned judge prompt is LoRA editing by another name (+0.02 here). |
| **Latent Critic** (LoRA + verdict token; [arXiv 2608.10430](https://arxiv.org/abs/2608.10430)) **[html]** | Qwen3-4B tool calls: 0.966 / 0.925 OOD against a linear probe's 0.782 / 0.616 | Equivalent to the LoRA verifier already run (+0.02 on held-out domains). Its OOD is a new tool set within one format. The Llama result collapses (0.649 OOD). |
| **Attribution-graph verification (CRV)** | 92.5 synthetic, about 0.70 GSM8K | Transcoder replacement plus graph building per step is CPU-infeasible at serving scale. Our math detection is already 0.85-0.93. Use it for case studies only. |
| **EPGS / CausalGaze gradient sensitivity as a detector** | see idea 5 | On-policy, short answers, and noise-scale fragile. Kept only as AtP for localisation. |
| **Local intrinsic dimension** ([ICML 2024, arXiv 2402.18048](https://arxiv.org/abs/2402.18048)) | +0.03-0.08 over uncertainty baselines | It is a kNN-neighbourhood statistic. Our kNN, Mahalanobis and isolation-forest anomaly scores were ≈ 0.50 ("wrong replies are not outliers"). |
| **FEPoID layer choice by intrinsic dimension** ([arXiv 2605.26366](https://arxiv.org/abs/2605.26366)) | label-free layer choice | Our layer choice is already settled (the middle band, with a flat optimum). |
| **Entropy / token-frequency neurons** ([Stolfo et al. 2024](https://arxiv.org/abs/2406.16254)) | no 2025-2026 detector paper found (searched again) | They measure the reader's next-token confidence, which is at chance on others' text. TriLens's per-layer entropy (idea 1) is the readable version. |
| **Known-entity / unknown-entity latents** (Ferrando et al. ICLR 2025; Vicentino ICML 2026) | 0.73 uncertainty latent; "induction, not calibration" | Covered by mech_interp.md and `experiments/recall_neurons.py`. Idea 2 is the representation-level, label-trained version. |
| **Verification heads / GLU "incorrect" vectors** ([arXiv 2504.14379](https://arxiv.org/abs/2504.14379)) | causal on CountDown | Narrow task. A hand-written readout of named components failed twice here (SAE-written −0.13, written direction ±0). Folded into idea 5(a) as head selection. |
| **HalluTracer** (depth-averaged probe logit; [arXiv 2608.16353](https://arxiv.org/abs/2608.16353)) | numbers not extracted | Depth-averaging probe logits ≈ our three-layer mean (LayerMix in Lee et al. 2026 says the same). Small expected gain. |
| **Trajectory commitment / attractor basins** ([arXiv 2604.15400](https://arxiv.org/abs/2604.15400)) | 87.5% corruption vs 33% correction by patching | 61 prompts, one model, mechanistic not a detector. It informs idea 2. |
| **Probes on reasoning-model thought tokens** ([2504.05419](https://arxiv.org/abs/2504.05419), [2605.09502](https://arxiv.org/abs/2605.09502), [2511.14773](https://arxiv.org/abs/2511.14773)) | 0.85-0.95 on own traces | On-policy. For a reader, the equivalent is our per-step unit head (already 0.85-0.93). Masked by Consensus says a peer loses nothing on math. Making the reader *think* would be generation (cost 100×). |
| **Observer-model residual probe** ([arXiv 2507.23221](https://arxiv.org/abs/2507.23221)) | F1 0.97-0.99 on news fabrications; 0.75-0.84 on contradictions | This is exactly our reply-head design (a frozen observer, the last sentence's full stop). It confirms the design rather than adding a signal. |

---

## 4. Uncertainties

- **Off-policy gap.** Every number above, except Masked by Consensus, PEP's cross-generator figures and the observer paper, is a
  model judging its *own* outputs. Ideas 1 and 3 read the reader's verdict about someone else's text, which is the setting where
  our judge works (0.64 → 0.78 with scale). Ideas 5(b) and 6 inherit the on-policy risk.
- **Single-preprint numbers.** TriLens, CausalGaze, EPGS, Will the Judge Flip?, PEP and the Latent Critic are each one 2026
  preprint.
  - CausalGaze's abstract contradicts its own table.
  - EPGS's robustness claim contradicts its own ablation.
  - SpikeScore's headline numbers are from secondary summaries; its five arXiv versions may differ.
- **Small-Qwen null.** The commitment-ratio probe was at chance on Qwen2-1.5B. If idea 1 is null at 1.7B, re-run it at 8B before
  discarding: the judge expert is the most scale-sensitive part of our stack.
- **Corpus pairing.** Idea 2 is structurally blind on domains with paired replies to the same question. Report it on unpaired
  domains separately, or its average will understate it.
- **Not found** despite targeted searches: a 2025-2026 paper using steering-robustness or "verdict fragility under activation
  steering" as a detector; an entropy-neuron hallucination detector; any 2026 follow-up to TSV or SEP with new numbers.
