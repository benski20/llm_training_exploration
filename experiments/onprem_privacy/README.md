# On-prem / private-inference viability tests

These tests check claims from a strategy discussion about private LLM inference: FHE, selective FHE,
pseudonymization, and token or weight obfuscation. Everything runs on a 4-core CPU with no GPU.
The raw numbers are in `results/*.json`.

| Test | Claim tested | Result |
|---|---|---|
| Source check | arXiv 2601.18511 numbers | Real paper: *Scaling up FHE-based PPML ... Llama-3-8B* (CKKS, 8x RTX PRO 6000). 128 encrypted tokens: ~20 s prefill, **18 s/token** generation. 4096-token input with only the **last 128 encrypted**: 64 s prefill, **22 s/token**. The "Llama-2-7B" title in the earlier source list was a mis-citation of the same ID. |
| `t1_rsa.py` | Textbook-RSA homomorphism demo (n=33, e=3) | The math is right (8 x 26 = 10 mod 33). But 10 is a **fixed point**: enc(10) = 10, and 9 of the 33 values encrypt to themselves. A dictionary attack recovers 100% of values. It shows the homomorphism, not security. |
| `t2_taint.py` | "Encrypt only the sensitive span; the rest runs at full speed" | Qwen2.5-0.5B, with only the sensitive span swapped. The **prefix is bit-identical** (diff 0.0). **All 14 tokens after the span change, from layer 1**, by 3-8% relative. Generated text restates the secret ("diabetes education classes"). So everything after the first encrypted token, and every output token, must be computed under encryption. Only a plaintext *prefix* is cheap, and that is exactly how the paper sets things up (last 128 tokens encrypted). |
| `t4_local_speed.py` | "If FHE is feasible for 8B, plaintext local is far cheaper" | Plaintext decode on 4 CPU cores, using an unoptimised HF stack: 0.5B = **0.040 s/token**, 1.5B = **0.094 s/token**. Extrapolating linearly gives about 0.5 s/token for 8B. That is about **45x faster than FHE's 22 s/token on 8 workstation GPUs**, on roughly 1/50 of the hardware. |
| `t3_fhe_bench.py` (a) | FHE vector search is practical | CKKS, 384-dim, 2,000 docs: **1.8 s total (0.9 ms/doc)**, top-1 match, max error 6e-8, query ciphertext 330 KB. That is 1,600x slower than plaintext but fine for shortlists (about 10^4-10^5 docs). Single-core naive scaling puts 1M docs at about 15 min, so it needs packing or parallelism. |
| `t3` (b) | Encrypted transformer linear layers | Cost scales ~d^1.04. Naive CKKS on one core, 1024x1024 matvec: 2.6 s (18,000x plaintext). Extrapolated linear layers alone for Llama-3-8B: **~74 min/token**. Specialised GPU CKKS (the paper) is ~200x faster and still 22 s/token. |
| `t3` (c) | Non-linearities are the hard part | Polynomial fit on [-8, 8]. GELU: deg 7 max error 0.22, deg 15 0.036, deg 31 2e-4. exp for softmax needs deg 15 to reach 1.5e-6. Each degree step adds multiplicative depth, and each level of depth eventually costs a bootstrap. |
| `t3` (d) | A small encrypted classifier head is practical | 384 -> 5 logistic head: **0.9 s**, argmax correct. |
| `t5_pseudonymize.py` (a) | Placeholders don't hurt the task | Qwen2.5-1.5B, 5 counselor notes. Placeholders were kept in the output **5/5**. Word overlap with the real-name output was 0.28, against **0.30 for just changing the names** (the baseline noise level). n=5 and a small model, so this is indicative only. |
| `t5` (b) | Sensitivity is contextual: names are not the only identifier | Synthetic district of 6,000 students, with a *perfect* name/ID detector. If a note keeps school + grade + gender + program + the sensitive condition, **57% of students are unique (k=1) and 96% have k <= 3**. Without the condition: 2.6% unique, 23% k <= 3. Without the program: 0% unique, median k = 36. |
| `t6_permutation_attack.py` | Token-permutation obfuscation is broken | Secret vocab permutation, attacked with frequency analysis plus iterative bigram context. On a same-domain corpus (wikitext, 561k tokens): **71% of token occurrences decoded, 84% of the top 500 types**. On a small cross-domain corpus (63k tokens of technical markdown with a Wikipedia reference): **fails, about 2%**. The attack needs reference text from the same domain. |
| `t7_weight_matching.py` | If the obfuscated model comes from public weights, the key leaks from the weights | E' = P·E·Q (vocab permutation plus a secret orthogonal rotation). With no noise, **99.99% of the permutation is recovered from row norms alone**, without seeing any user traffic. With 1% Gaussian noise, our naive attack fails (0.7-6%). Published attacks (ICML 2025 "Hidden No More") report beating noise defences; we did not reproduce that. |

## What this means

1. **Selective FHE does not give "full speed on the public part" for generation.** Only a plaintext prefix is
   cheap. Generation stays at full FHE cost (22 s/token on 8 GPUs, even when 97% of the prompt is plaintext).
2. **Where FHE is feasible (≤8B), running the same model locally in plaintext is ~45x faster on far less
   hardware.** FHE LLM inference only pays off for models too large to run locally, and that is where it is
   infeasible.
3. **FHE is practical today for linear-heavy steps**: encrypted embedding search and small classifier heads, at
   ~1 ms/doc and ~1 s/classification on one core.
4. **Pseudonymization costs no measurable task quality** (small n), but **it is not anonymization**. The sensitive
   attribute plus a few quasi-identifiers re-identifies most students. The cloud model must not receive the
   attribute together with school, grade and program. Either generalise those fields (k-anonymity) or keep that
   step local.
5. **Cheap obfuscation fails in the realistic threat model.** The server operator knows the base model and the
   domain. Noise slows naive attacks, but noise also degrades the model.

Run: `pip install tenseal torch transformers scipy pandas pyarrow`, then `python t<N>_*.py` from this directory.
