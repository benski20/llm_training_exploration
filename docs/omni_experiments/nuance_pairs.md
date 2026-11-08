# Minimal pairs: one subtle error in an otherwise correct reply

**Status: the test set and the kernel are built; the models were never run.** Every Kaggle push was refused with
`Maximum weekly GPU quota of 30.00 hours reached` (2026-11-08 15:01 UTC). Before that, two pushes were turned away
because the account's 2 concurrent GPU sessions were taken. Loading models locally is ruled out here. So there are **no
AUROC or pairwise-accuracy numbers** for the reader, the judges or NLI yet. The only scored expert is exact arithmetic,
which is pure Python (below). To run it once quota frees:

```bash
python kaggle/kgrun.py --slug omni-nuance-smoke --script kaggle/omni/nuance_pairs.py --bundle vt/omni.py vt/mathcheck.py \
    vt/correctness.py vt/checkers/__init__.py vt/checkers/business_record.py experiments/omni_corpus.py \
    kaggle/omni/nuance_pairs.py --args "--smoke 3" --out results/kaggle/omni-nuance-smoke --wait
python kaggle/kgrun.py --slug omni-nuance ... (same, no --args) --out results/kaggle/omni-nuance --wait   # ≈ 3-4 h
python kaggle/omni/nuance_pairs.py --analyse results/kaggle/omni-nuance --report results/omni/nuance_pairs.json
```

The bundle differs from the requested command in two ways:
- **The script bundles itself.** Its GPU stages run as subprocesses of it.
- **The head is not bundled.** With `heads/omni_qwen3_8b.npz` in the bundle (17.9 MB), the push fails with HTTP 400.
  The kernel downloads the head instead, from the public repo at commit b284b37 (the commit that shipped it), and
  checks its md5 (`2a654983…`) against the local file.

## Setup (`kaggle/omni/nuance_pairs.py`, deterministic, seeded)

**Originals.** Every original is a reply that the corpus labels correct (`experiments/omni_corpus.py build`, md5 b598848a…).
- **long-form:** FELM, all five domains.
- **RAG:** RAGTruth QA and summary with no hallucinated span, plus HaluEval dialogue right responses.
- **reasoning:** ProcessBench solutions with label −1.

Long replies are preferred: sampling is weighted by length. The median original is 763 characters, and the length
terciles split at 550 and 1,198 characters.

**Twins.** Each twin differs from its original in exactly one claim. The check: the sentence-level diff is one sentence.
The question and documents are identical within a pair. RAGTruth's prompt embeds the reply itself, so that embedded
copy is swapped for the twin's text; the task and passages stay unchanged.

**Edit position.** The edit site is drawn from an early, middle or late third of the reply, uniformly among the thirds
that have a candidate. The position is recorded as a fraction of the reply.

| type | method | pairs built (target) |
|---|---|---|
| number | regex: year ±1-3, small integer ±1, larger ×0.75-1.5 at the same precision. With documents, the new figure is absent from them. Figures restated from the task and "N words" are skipped. | 84 (90): longform 45, rag 39 |
| qualifier | fixed swap list: always→usually, all X→some X, increase↔decrease, before↔after, more than↔less than, higher↔lower, … Titles and idioms are skipped. | 90 |
| negation | add "not" after a copula or modal, or remove one (doesn't→does, cannot→can). "here is", "not only" and "there is no" are skipped. | 90 |
| entity swap | Qwen3-8B, greedy, in the kernel. Kept only if the word diff is one span of ≤4 words whose new capitalised word is absent from the task, documents and reply. | 0 of 90 (needs GPU; 144 slots prepared) |
| unsupported addition | Qwen3-8B writes one sentence with a specific detail not in the source (closed-book: plausible but false). It is inserted as its own unit. | 0 of 90 (needs GPU; 144 slots prepared) |
| reasoning, intermediate | ProcessBench: the last "= N" of one non-final step changed. Nothing downstream is changed. | 80 |
| reasoning, final | ProcessBench: a number inside `\boxed{}` changed | 33 (60: few boxed answers are plain numbers) |

The rule-built pairs come to 377 over 12 domains. Their positions:

| type | early | middle | late |
|---|---|---|---|
| number | 27 | 35 | 22 |
| qualifier | 33 | 29 | 28 |
| negation | 39 | 25 | 26 |
| reasoning, intermediate | 12 | 28 | 40 |
| reasoning, final | 0 | 0 | 33 |

**Examples** (more in `results/omni/nuance_pairs.json`):
- number, HaluEval dialogue: "Tortilla Flat was released in 1935" → "1937".
- qualifier, RAGTruth summary: "many analysts predict that Ramadi may soon fall" → "few analysts predict…".
- negation, RAGTruth QA: "Stalin was trying to force the USA out of Berlin" → "was not trying".
- reasoning, intermediate, GSM8K: "12 gifts × 1 bow per gift = 12 bows" → "= 13 bows". Later steps still use 12.
- reasoning, final, MATH: "f(−10) = −25 … \boxed{−25}" → "\boxed{−19}".

**Planned scoring.** The kernel scores every original and every twin, plus 100 seeded corpus replies per source
domain, which form the profile.
- **Reader stack:** `vt.omni.Reader("Qwen/Qwen3-8B")` with the shipped head. It gives the reply head, the worst unit,
  the reader's own judge, arithmetic flags and per-unit scores.
- **External judges:** Qwen3-14B and phi-4, both 4-bit. Each uses `JudgeScorer`'s prompt and a single pass, and is
  loaded as in `judge_calibration.py`.
- **NLI:** `NLIScorer` on the replies that have documents. MiniCheck is also run on all sentences as a diagnostic,
  because the shipped scorer reads only the first 4.
- **Profile:** a per-domain zero-label profile built from the corpus domain's replies with their labels unused, not
  the universal profile. The universal profile carries no judge statistics.
- **Combination:** z-sum as `OmniVerifier._combine`: r + u + 2j + 2·flags + NLI + 3 + 3 (one weight of 3 per judge).

**Planned reports.** The analysis is implemented and was tested on synthetic scores. It reports:
- pairwise accuracy with a bootstrap 95% CI, and AUROC;
- breakdowns by type, format, domain, edit third, length tercile, type×format and type×third;
- top-1 and top-3 hits of the edited unit among the reader's unit scores, against chance.

**Caveat.** The shipped 8B head was trained on all 5,107 corpus replies, so the originals were in its training set as
correct examples. The twins were not.

## The one measured expert: exact arithmetic (`vt.mathcheck`, explicit equations)

| type | twins flagged | originals flagged | pairwise accuracy (ties = 0.5) |
|---|---|---|---|
| reasoning, intermediate (80) | 27 (34%) | 2 | 0.66 |
| number (84) | 6 | 0 | 0.54 |
| reasoning, final (33) | 2 | 1 | 0.52 |
| qualifier, negation (180) | 0 | 0 | 0.50 |

## Verdict (honest)

**Nothing is known yet about the reader, judges or NLI on nuanced errors.** The pairs are ready.

The arithmetic checker catches a third of the slips in the middle of a solution, because those are written as
"a op b = c". It is blind to everything else.

Expectations to test, not results:
- **The final-answer-only change and the late-position edits should be the hard cells for the reader probes.** The
  arithmetic checker misses them.
- **NLI's MiniCheck half cannot see an edit after the 4th sentence.** The diagnostic `mc_all` measures that directly.
