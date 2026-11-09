# Layer-wise commitment (logit lens) as a signal that a reply is wrong

**Question.** Apply the reader's final norm and unembedding to the residual stream after every layer (logit lens).
Is the token a reply actually wrote "committed" earlier and more stably when the content is correct? And is it
committed later or more erratically when the content is fabricated or wrong? If so, does that add to the 1.7B reply
head?

**Answer.**
- **There is a weak signal, and it lives almost entirely in reasoning.** Wrong content is committed slightly later.
  Eight of the nine timing features pass R1 overall, and none is reversed. Over all 21 domains the best single feature
  is 0.57 (mean settle depth). On reasoning, settle depth reaches 0.77 and the LODO head 0.79. Elsewhere the features
  sit at 0.46-0.62.
- **It is not new information.** The reader's 10 output scalars (log-prob, entropy, margin) are already in the reply
  head, and alone they score 0.604. The commitment head scores 0.574. Scalars plus commitment score 0.599
  (−0.005 [−0.025, +0.015]), so R2 fails.
- **It does not add to the reply head; it hurts.** The stack gives −0.016 [−0.032, −0.001] and the z-sum gives
  −0.040 [−0.063, −0.017]. No format passes R3. Reasoning is +0.006 [−0.023, +0.036] for the stack.
- **Why.** On Qwen3-1.7B the lens says almost nothing before about 75% depth. At layer 14 the written token is top-1
  for 3% of tokens; it reaches 46% at layer 22 and 70% at layer 28. The per-layer curves of correct and wrong replies
  overlap to two decimals at every layer (table below). In practice "commitment depth" is a soft re-reading of the
  final log-prob: settle depth correlates 0.53 with −mean log-prob within a domain, and the fraction of tokens never
  settled correlates 0.62.

**Verdict: do not adopt.**

- Script: `experiments/novel_logit_lens.py`.
- Results: `results/omni/novel_logit_lens.json`, which holds per-domain AUROCs, CIs, rules, depth trajectories and ids.
- Compute: CPU, bf16, 2 threads, 4.4 h for 1,050 replies, with the full 151,936-token vocabulary at every layer.

## Protocol

**Data.**
- The subset is 25 wrong and 25 correct replies per domain (seed 0), giving 1,050 replies in 21 domains.
- Each reply gets one forward pass with `output_hidden_states`, max_tokens 1024.
- The lens covers reply positions only: layers 1-27 go through `model.norm` and `lm_head`, and layer 28 is the real
  output. It is exact and computed layer by layer over chunks of 256 positions.

**Token measures** (depth = layer / 28; "never" = 29/28):
- settle: the first layer from which the written token stays top-1;
- first top-1;
- first layer with p > 0.5;
- top-1 flips over layers 7-28;
- KL(final ‖ lens) at layers 14-21;
- mean log(1 + rank) at layers 14-27;
- top-1 at layer 21 ≠ final top-1;
- −mean log p at layers 14-28.

**Reply features (20).**
- Means of the 8 token measures.
- Top-decile means of 6 of them.
- Worst sentence (the maximum unit mean) of 4 of them.
- The fraction of tokens never settled, and the fraction settled late (after layer 21).
- Every feature is signed a priori so that larger means wrong.

**Reply head.** It follows the `omni_final.py` recipe, LODO on all 5,107 stored q17 replies, and its scores are taken
for these ids. The stack uses nested heads: a training row's reply score comes from a head that saw neither the
held-out domain nor the row's own domain.

**CIs.** 1,000 bootstrap resamples within each domain, stratified by label, paired across methods. Every AUROC is
computed within a domain and then averaged.

**Rules, fixed in the script before the full run:**
- **R1, signal:** the CI lower bound is above 0.50.
- **R2, beyond log-prob:** (scalars + commitment) − scalars ≥ +0.010, with the CI above 0.
- **R3, adds:** stack or sum − reply ≥ +0.010, with the CI above 0. A format-level claim needs the same within that
  format.

**Smoke test.** 20 replies ran first: the lens was sane (KL at layer 27 ≈ 0.13), and the pipeline ran end to end.

## Results (mean AUROC over domains [95% CI])

| score | all 21 | chat (2) | long-form (5) | RAG (10) | reasoning (4) |
|---|---|---|---|---|---|
| mean settle depth | 0.573 [0.538, 0.607] | 0.533 | 0.557 | 0.512 | **0.765** |
| mean p>0.5 depth | 0.571 [0.536, 0.606] | 0.569 | 0.604 | 0.477 | 0.765 |
| mean first top-1 depth | 0.556 [0.518, 0.589] | 0.582 | 0.584 | 0.460 | 0.746 |
| mean top-1 flips | 0.539 [0.504, 0.576] | 0.563 | 0.609 | 0.493 | 0.554 |
| mid (L21) ≠ final top-1 | 0.571 [0.534, 0.607] | 0.525 | 0.605 | 0.526 | 0.661 |
| mean KL mid | 0.526 [0.492, 0.560] | 0.560 | 0.573 | 0.466 | 0.602 |
| worst sentence settle | 0.572 [0.538, 0.607] | 0.576 | 0.517 | 0.522 | 0.763 |
| **commitment LODO (20 features)** | **0.574** [0.540, 0.607] | 0.499 | 0.529 | 0.526 | **0.789** |
| output scalars LODO (control) | 0.604 [0.569, 0.635] | 0.450 | 0.571 | 0.579 | 0.784 |
| scalars + commitment LODO | 0.599 [0.566, 0.632] | 0.449 | 0.565 | 0.567 | 0.796 |
| **reply head (1.7B)** | **0.709** [0.679, 0.739] | 0.620 | 0.694 | 0.663 | 0.886 |
| reply + commitment, z-sum | 0.669 [0.639, 0.701] | 0.582 | 0.627 | 0.620 | 0.887 |
| reply + commitment, stack | 0.693 [0.663, 0.722] | 0.605 | 0.663 | 0.646 | 0.892 |

Paired differences:

| comparison | all | chat | long-form | RAG | reasoning |
|---|---|---|---|---|---|
| stack − reply | −0.016 [−0.032, −0.001] | −0.016 [−0.070, +0.034] | −0.031 [−0.066, +0.002] | −0.017 [−0.041, +0.009] | +0.006 [−0.023, +0.036] |
| z-sum − reply | −0.040 [−0.063, −0.017] | −0.038 | −0.067 | −0.043 | +0.001 |
| (scalars + commit) − scalars | −0.005 [−0.025, +0.015] | −0.001 | −0.006 | −0.012 | +0.011 [−0.028, +0.056] |

**Depth trajectories** (all replies; layers 14, 16, ..., 28; correct / wrong):
- written token is top-1: 0.03 0.04 0.09 0.21 0.46 0.55 0.66 0.70 / 0.03 0.04 0.08 0.21 0.45 0.53 0.64 0.69;
- KL to the final distribution: 9.53 9.23 7.92 6.01 2.95 1.97 0.46 0 / 9.52 9.25 7.86 6.00 2.92 2.02 0.52 0.

## Notes

- **RAG goes the "wrong" way on rank-type features.** Mean log-rank scores 0.40 and output −log-prob 0.42. Wrong RAG
  replies are, if anything, more fluent for the reader. The likely reason is that they copy or paraphrase plausible
  text. Commitment timing cannot separate them.
- **Reasoning is where low commitment means wrong.** A wrong step makes the reader surprised by what follows. The reply
  head already captures this (0.886).
- **Not tried:** a tuned lens (learned per-layer translators). It would make middle layers readable, but it is a
  different and costlier idea. Given how little of the trajectory differs here, it is unlikely to change the verdict.
