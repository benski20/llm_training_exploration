# Editing the reader: LoRA verifier and domain-adversarial LoRA (Qwen3-1.7B)

**Question.** If we fine-tune the reader's weights (LoRA) so that a probe on its internal state predicts "wrong",
does the probe get more accurate on held-out domains and formats? Does a gradient-reversal domain classifier
on the same state make it more domain-invariant?

**Answer.** Only a little.
- LoRA gains about +0.02 to +0.025 AUROC on held-out domains and formats, and +0.035 in-distribution.
- The adversarial term adds nothing. It did not even remove domain information from the state.

Script: `kaggle/omni/lora_adv.py`. Kaggle kernel `omni-lora-adv`, 2× T4, 5.3 h wall time. Results: `results/omni/lora_adv.json`;
the raw outputs (per-reply scores, log) are in `results/kaggle/omni-lora-adv/`.

## Setup

- **Corpus.** Built inside the kernel with `experiments/omni_corpus.py build`: 5,107 replies, 21 domains, 4 formats.
- **Tokenisation.** `vt.omni.encode`, with the judge suffix and **max_tokens 1024** (the product uses 2,048; 1,024 was chosen to save GPU time).
  - The suffix is cut before the forward pass. The reader is causal, so the suffix cannot change the reply's states.
- **Feature.** Mean residual state over the reply tokens at `hidden_states[15]` ("layer 15 of 28").
  - Only decoder layers 0-14 are run, since nothing above them can change this feature.
  - A check in the kernel matches the full model's `output_hidden_states` to within a relative difference of 1e-8.
- **Frozen baseline.** Same kernel, same folds. The probe is PCA 256 + L2 logistic (0.05) with balanced weights (the `vt/omni.train_artifact` recipe). It is fitted with global standardisation, and also with per-domain standardisation ("dstd").
- **LoRA.**
  - Adapters: rank 16, alpha 32, dropout 0.05, on q/k/v/o/gate/up/down of layers 7-14. These are the layers just below the probe; layers above 15 cannot affect it.
  - Head: a linear head on the globally standardised pooled state, **initialised at the frozen probe**.
  - Training: balanced BCE, 2 epochs, AdamW (LoRA lr 1e-4, head lr 5e-4), cosine schedule, fp16 + GradScaler, about 4k tokens per batch.
  - Each variant is scored two ways:
    - (a) its own trained head ("e2e");
    - (b) the frozen-baseline probe recipe refitted on the edited reader's features ("refit").
- **Adversarial.** The same, plus an MLP domain classifier (2048-256-n_domains) behind gradient reversal. λ ramps from 0 to 0.3 (DANN schedule).
- **Folds.**
  - LOFO: 4 folds, every domain of the held-out format is scored.
  - LODO: 6 held-out domains, each fold trained on the other 20.
  - In-distribution (ID): a stratified 80/20 split.
- **Metric.** AUROC within each held-out domain, averaged.
- **CIs.** Paired bootstrap (1,000 resamples, replies resampled within each domain) on the mean difference vs the frozen probe.
- **Seeds.** All seeds are fixed, with one training seed per fold, so training variance is **not** in the CIs.
- **Hyperparameters.** Fixed in advance, not tuned on any test fold.

## Results

| mean AUROC | frozen probe | LoRA e2e | LoRA refit | LoRA+adv e2e | LoRA+adv refit |
|---|---|---|---|---|---|
| **LOFO** (21 domains) | 0.624 | 0.634 | **0.648** | 0.632 | 0.647 |
| rag (10) | 0.528 | 0.544 | 0.557 | 0.540 | 0.555 |
| chat (2) | 0.639 | 0.639 | 0.668 | 0.638 | 0.669 |
| longform (5) | 0.703 | 0.726 | 0.725 | 0.728 | 0.725 |
| reasoning (4) | 0.762 | 0.741 | 0.768 | 0.741 | 0.765 |
| **LODO** (6 domains) | 0.662 | 0.673 | **0.688** | 0.673 | 0.685 |
| halubench_finance | 0.553 | 0.566 | 0.600 | 0.569 | 0.600 |
| ragtruth_summary | 0.620 | 0.600 | 0.598 | 0.596 | 0.591 |
| truthfulqa | 0.623 | 0.624 | 0.665 | 0.622 | 0.664 |
| felm_wk | 0.549 | 0.553 | 0.571 | 0.554 | 0.569 |
| pb_math | 0.848 | 0.877 | 0.881 | 0.877 | 0.879 |
| halueval_dialogue | 0.782 | 0.816 | 0.811 | 0.817 | 0.807 |
| in-distribution (80/20) | 0.724 | 0.759 | 0.758 | 0.760 | 0.758 |

**Difference vs the frozen probe** [95% bootstrap CI]:

| | LoRA e2e | LoRA refit | adv e2e | adv refit |
|---|---|---|---|---|
| LOFO | +0.010 [+0.003, +0.016] | +0.023 [+0.017, +0.030] | +0.008 [+0.002, +0.014] | +0.022 [+0.016, +0.029] |
| LODO | +0.010 [−0.002, +0.023] | +0.025 [+0.012, +0.040] | +0.010 [−0.003, +0.023] | +0.023 [+0.007, +0.038] |
| ID | +0.035 [+0.023, +0.050] | +0.034 [+0.020, +0.050] | +0.035 [+0.023, +0.050] | +0.034 [+0.019, +0.050] |

**Adversarial minus plain LoRA:**
- LOFO: −0.002 [−0.003, −0.000] (e2e) and −0.001 [−0.003, +0.001] (refit).
- LODO: −0.000 [−0.005, +0.005] (e2e) and −0.003 [−0.009, +0.004] (refit).

**Per-domain standardisation** of the probe features made no difference here, for frozen or edited features:
- frozen: −0.002 LOFO, −0.000 LODO;
- edited (refit dstd vs frozen dstd): +0.027 LOFO, +0.025 LODO.

**Domain information in the pooled state.** A multinomial logistic on PCA-64 predicts the domain with mean accuracy
0.874 (frozen), 0.874 (LoRA) and 0.867 (adversarial); chance is 0.078.
- The adversary's own loss stayed at 0.06-0.24 nats: it could still tell the domains apart.
- At λ = 0.3 the reversal barely touched the domain signal. A much larger λ would be needed to remove it, and that would risk the label signal too.

**The frozen baseline reproduces the repo.** LOFO 0.624 here against 0.620 in `docs/omni_verifier.md`.
- On LODO the repo has no number for this exact 6-domain subset (its single-layer mean over all 21 domains is 0.679).
- Per-domain numbers match the repo's pattern: pb_math and halueval_dialogue are high; felm_wk, halubench_finance and TruthfulQA are near 0.55-0.62.

## Cost

**Training.** About 20-25 min per fold per variant on one T4 (2 epochs, 1,000-1,250 steps). The adversarial variant is about 10% slower.

**Inference.** On a T4, fp16, batch 1, median reply about 480 tokens:

| forward pass | s per reply |
|---|---|
| full 28-layer pass | 0.157 |
| layers 0-14 only | 0.081 |
| layers 0-14 with unmerged LoRA | 0.089 (+10%) |
| LoRA merged into the weights | no extra cost |

**What this means for the product.**
- Merged LoRA is free per check *only if the whole product moves to the edited reader*.
- An edited reader changes every downstream state. The judge log-odds, the unit head and the three-layer features would all need retraining, and their own transfer re-measured.
- Keeping the frozen experts as well needs a second pass over layers 0-14: about +50% per check.

## Conclusion

- **Editing the reader helps a little, and reliably.** About +0.02-0.025 AUROC on unseen domains and formats when the probe is refitted on the edited features, with CIs that exclude zero. The gain in-distribution is larger (+0.035).
- **The trained end-to-end head transfers less well** (+0.01; the LODO CI includes 0). Most of the gain shows up only when a fresh probe reads the edited features. The fine-tuned head itself overfits the training domains.
- **The gain is uneven:**
  - It is largest on felm_reasoning, felm_writing, TruthfulQA, FinanceBench and pb_math.
  - It hurts some domains: ProcessBench olympiad/omnimath under LOFO (−0.02 to −0.04 for the e2e head) and ragtruth_summary.
  - LOFO RAG stays near chance (0.56), and halueval_qa / ragtruth_data2text stay below 0.5.
- **Domain-adversarial training is not worth it.** It neither changed accuracy nor removed domain information at λ = 0.3.
- **Verdict: low priority.**
  - +0.02 is about the size of the per-domain-standardisation or worst-unit levers. It is far below a bigger reader (+0.08 for 4B on the same protocol) or 50-100 labels from the new domain (+0.06-0.07).
  - The price is maintaining a custom reader and retraining every other expert on it.
  - If tried again, test it on the full combined score (0.73 baseline), where the gain may shrink.

## Caveats

- **One training seed per fold**, so the CIs cover reply sampling only, not training variance.
- **Hyperparameters were not tuned:** rank, layers, epochs and λ were fixed once. A larger λ or more epochs were not explored.
- **max_tokens 1024**, not the product's 2048. This truncates the long passages of some RAG domains, which matters for both frozen and edited results.
- **Single-feature setting.** Only the layer-15 mean-pooled state was tested, not the product's full feature stack (three layers + judge state + scalars + worst unit).
- **Smoke test in the same session.** GPU slots were full for about 6 h, so the smoke test (2 folds, 24 replies per domain, 0.5 epoch) ran as the first stage of the same kernel session, gated so that a failure would abort the session. It passed.
