# Planted channel: write the probe, then edit the reader to make it true

**Question.** Instead of fitting a probe to a frozen reader, fix the probe first and train the reader so that the
fixed probe carries P(reply is wrong) with a fixed calibration. Does that channel (a) transfer to held-out
domains/formats better than a fitted probe, and (b) hold a false-alarm rate across domains at a *fixed* threshold,
with no per-domain calibration?

**Answer: no on both counts.** The planted channel ties the fitted probe on held-out AUROC (+0.016 leave-one-format-out,
CI [-0.001, 0.032]; +0.002 leave-one-domain-out) and its fixed threshold does not hold false alarms on new domains:
16 of 21 held-out domains exceed 15% false alarms at the threshold set for 10% on the training domains, mean 47%
under format shift. Fixing the readout does not fix where a new domain's clean replies land on it.

Script `kaggle/omni/planted_channel.py` (Kaggle 2x T4, kernel `omni-planted`, about 3.6 h); results
`results/omni/planted_channel.json` (raw kernel outputs incl. per-reply scores in `results/kaggle/omni-planted/`).

## Setup

- **Corpus:** `experiments/omni_corpus.py build` inside the kernel: 5,107 replies, 21 domains, 4 formats, label 1 = wrong,
  balanced within domain.
- **Tokenisation:** `vt/omni.py encode(tok, row, 1024)`. **max_tokens = 1024**: median 385 tokens, 20% of replies hit the cap
  (head trimmed first, then the reply's tail). No judge suffix.
- **The written probe (fixed before training, never fitted):**
  - feature: `h_mean`, the mean residual state over the reply's tokens at the output of block 15 of 28 of Qwen3-1.7B
    (the best single layer in docs/omni_verifier.md);
  - direction: `w`, a random unit vector, `numpy.random.default_rng(12345)`, d = 2048;
  - readout: `score = <h_mean, w>`, `p_wrong = sigmoid(score)`, flag if `score > 0`. No bias, no scale, no threshold fitting.
- **Planting:** LoRA r = 16 (alpha 32, dropout 0.05) on all attention and MLP projections of blocks 1-15 (blocks 16-28 are
  irrelevant to the channel and were dropped). Loss = BCE(sigmoid(score), label). 2 epochs, AdamW lr 2e-4, cosine,
  batch 1 × 8 accumulation, fp16 base + fp32 adapters, GradScaler, sdpa. One LoRA per fold, trained from scratch.
  Before training the written score is 7.7 ± 1.5 on every reply (every reply "wrong"); training loss falls to 0.36-0.42
  in epoch 2.
- **Baselines (same kernel, same folds, same feature):**
  - `probe_global`: frozen reader, L2 logistic probe (`vt.omni.LogReg`, l2 0.05) on `h_mean` standardised with training statistics;
  - `probe_pds`: the same with each domain standardised by its own statistics (label-free; for a held-out domain, from its
    unlabelled replies). This is the repo's usual practice, so it is a second reference, not the requested baseline;
  - `written_frozen`: the written probe on the frozen reader (sanity check: chance).
- **Protocol:** leave-one-format-out (4 folds, 21 held-out domains scored) and leave-one-domain-out on 6 domains
  (halubench_finance, ragtruth_summary, truthfulqa, felm_wk, pb_math, halueval_dialogue). In each fold 15% of the training
  domains' replies (stratified by domain) are never trained on; they set the "10% false alarms on training domains"
  threshold (90th percentile of their clean scores) for every method. Thresholds: fixed 0 (planted score; probe logit),
  and that transferred 10% threshold.
- **CIs:** paired bootstrap (1,000), resampling replies within each held-out domain (stratified by label) and the
  validation clean replies that set the threshold; difference of the mean over domains.

## Results

### Held-out AUROC (mean over held-out domains)

| | planted | probe_global | probe_pds | written_frozen | planted − probe_global [95% CI] |
|---|---|---|---|---|---|
| leave-one-format-out (21 domains) | **0.631** | 0.615 | 0.591 | 0.485 | +0.016 [-0.001, +0.032] |
| leave-one-domain-out (6 domains) | **0.656** | 0.653 | 0.649 | 0.503 | +0.002 [-0.022, +0.028] |
| in-distribution validation (mean of 10 folds) | **0.803** | 0.773 | 0.765 | | |

Planting wins in distribution (+0.03 on average on the training domains' validation split, +0.004 to +0.053 per fold) and the gain
mostly does not survive a new domain or format.

### False alarms at the transferred thresholds

Target 10%. FA = false-alarm rate on the held-out domain's clean replies; rec = wrong replies caught.

| leave-one-format-out (21 domains) | planted | probe_global | probe_pds |
|---|---|---|---|
| fixed threshold 0: FA / rec | 68% / 82% | 65% / 80% | 38% / 52% |
| 10%-on-train threshold: mean FA | 47% | 47% | **18%** |
| mean abs(FA − 10%) | 0.38 | 0.39 | **0.08** |
| max FA | 99% | 97% | **26%** |
| domains with FA > 15% | 16 / 21 | 17 / 21 | 16 / 21 |
| recall at that threshold | 61% | 59% | 27% |

| leave-one-domain-out (6 domains) | planted | probe_global | probe_pds |
|---|---|---|---|
| fixed threshold 0: FA / rec | 43% / 60% | 42% / 62% | 35% / 57% |
| 10%-on-train threshold: mean FA | 19% | 22% | **14%** |
| mean abs(FA − 10%) | 0.13 | 0.17 | **0.07** |
| max FA | 45% | 49% | **23%** |
| domains with FA > 15% | 3 / 6 | 4 / 6 | 4 / 6 |
| recall at that threshold | 35% | 38% | 30% |

Paired differences, planted − probe_global [95% CI]:
- format-out: FA at threshold 0 +3.4 pts [+1.7, +4.8]; FA at the 10% threshold −0.3 [−4.9, +4.1]; abs deviation −0.2 [−4.7, +4.2].
- domain-out: FA at threshold 0 +0.8 [−2.5, +4.0]; FA at the 10% threshold −3.8 [−7.3, +0.4]; abs deviation −4.3 [−7.6, −0.3] (the
  only difference in the planted channel's favour that excludes zero, on 6 domains).

Planted − probe_pds: worse on every false-alarm measure (format-out FA at the 10% threshold +29 pts [+23, +34]; domain-out
+5 pts [+1, +9]), AUROC +0.04 [+0.02, +0.06] format-out, +0.01 [−0.02, +0.03] domain-out.

### Per held-out domain (format-out): AUROC / FA at the 10% threshold

| domain | planted | probe_global | probe_pds |
|---|---|---|---|
| halubench_covid | 0.63 / 67% | 0.55 / 73% | 0.51 / 25% |
| halubench_drop | 0.66 / 84% | 0.50 / 94% | 0.48 / 23% |
| halubench_finance | 0.51 / 89% | 0.47 / 87% | 0.50 / 18% |
| halubench_pubmed | 0.57 / 78% | 0.66 / 58% | 0.64 / 13% |
| halueval_dialogue | 0.65 / 35% | 0.65 / 66% | 0.68 / 12% |
| halueval_qa | 0.48 / 67% | 0.40 / 81% | 0.39 / 23% |
| halueval_summary | 0.60 / 44% | 0.59 / 91% | 0.53 / 16% |
| ragtruth_data2text | 0.49 / 99% | 0.54 / 97% | 0.56 / 19% |
| ragtruth_qa | 0.51 / 48% | 0.63 / 49% | 0.59 / 16% |
| ragtruth_summary | 0.55 / 80% | 0.50 / 58% | 0.48 / 16% |
| halueval_general | 0.58 / 8% | 0.63 / 5% | 0.59 / 20% |
| truthfulqa | 0.58 / 7% | 0.60 / 21% | 0.57 / 21% |
| felm_math | 0.71 / 9% | 0.72 / 10% | 0.72 / 12% |
| felm_reasoning | 0.73 / 32% | 0.69 / 22% | 0.68 / 17% |
| felm_science | 0.71 / 20% | 0.66 / 22% | 0.64 / 16% |
| felm_wk | 0.59 / 32% | 0.51 / 36% | 0.50 / 19% |
| felm_writing | 0.78 / 12% | 0.72 / 16% | 0.64 / 17% |
| pb_gsm8k | 0.73 / 0% | 0.72 / 0% | 0.67 / 13% |
| pb_math | 0.77 / 33% | 0.79 / 15% | 0.77 / 12% |
| pb_olympiad | 0.69 / 74% | 0.70 / 42% | 0.64 / 26% |
| pb_omnimath | 0.72 / 66% | 0.69 / 48% | 0.64 / 23% |

The failure mode is an offset, not a ranking failure: when RAG is the held-out format, the planted channel puts most
clean RAG replies above the training threshold (FA 35-99%), while on GSM8K it puts nearly nothing above it (0% FA, 9%
recall). Calibration is equally off: ECE of sigmoid(score) on held-out domains 0.30 format-out, 0.23 domain-out (probe:
0.29 / 0.21; per-domain standardised probe 0.23 / 0.16).

## Verdict

- **(a) Transfer: a tie.** The planted channel is a fine-tuned reader with a fixed head. It fits the training domains
  better than a linear probe on frozen states (+0.03 validation AUROC on average) and transfers about as well (+0.016 format-out,
  CI touching zero; +0.002 domain-out). The extra capacity is spent on the training distribution.
- **(b) Fixed threshold: does not hold.** Writing the readout in advance fixes the *readout*, not the location of a new
  domain's clean replies on it. Under format shift the transferred 10% threshold gave 47% false alarms on average, up to 99%,
  exactly as bad as the fitted probe. The repo's earlier finding stands: no label-free threshold transfers.
- **The best label-free option remains per-domain standardisation from unlabelled replies** (probe_pds): it ranks slightly
  worse but its transferred threshold lands nearest the target (18% / 14% mean, max 26%), still not a guarantee (16 of 21
  domains over 15%). Holding a stated false-alarm rate still needs the domain's own clean labels (`calibrate`, ≥ 29 clean).
- Not worth pursuing further as a commercial route at this scale.

## Caveats

- One seed, one written direction, one layer, one LoRA recipe (r 16, 2 epochs, lr 2e-4); no hyper-parameter search
  (which would need validation on held-out domains to be fair). A longer or regularised schedule could change the
  in-distribution/transfer balance but there is no sign the fixed-threshold problem is a capacity issue.
- The intervention variant (training the model's behaviour to depend on the channel, as in VISION.md) was not run.
- Planting could also have used a per-domain standardisation of the channel, but then it is no longer a fixed readout.
- 20% of replies truncated at 1,024 tokens (long RAG passages and solutions), the same for every method.
- Per-domain sizes are 125-400 replies; per-domain AUROCs carry about ±0.06. The 10% threshold is set on the pooled
  training-domain validation split (376-736 replies, about half clean).
- Bootstrap CIs resample replies and threshold-setting clean replies, not the training of the LoRA or probe.
