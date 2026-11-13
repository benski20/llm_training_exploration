# Calibrated AI-text detection

**The product.** A detector that says whether a text was written by an AI model. Its false-accusation rate is
a promise the customer sets and can audit: send 150 of your own human-written texts, choose the rate (say 5%), and
the detector picks its thresholds so that at most that share of human texts like yours is flagged, with 95%
confidence. No AI samples are needed from you. It runs on a CPU: two small open models (Qwen3-0.6B base and
Qwen3-0.6B, Apache-2.0) read the first 256 tokens of each text.

Why this angle: false accusations, of students above all, are the main complaint against AI detectors. Existing
tools publish an accuracy figure measured on their own data; none lets a customer set the false-accusation rate
for their own kind of writing and get a bound on it.

## How it decides

Two scores per text:

| score | what it reads | trained? |
|---|---|---|
| **probe** | the observer's mean hidden state at layers 7/14/21/28 plus seven statistics (log-perplexity, cross-perplexity, entropy, ...) | logistic regression on 5,280 RAID texts (8 domains, 11 generators, plain and paraphrased): `heads/aitext_probe_qwen3_0.6b.npz`, 51 KB |
| **Binoculars** (Hans et al., 2024) | performer log-perplexity / observer-performer cross-entropy | no (zero-shot) |

A text is flagged when **either** score passes its threshold. Each threshold is the one-sided Clopper-Pearson
threshold over the customer's human texts at half the false-alarm budget (2.5%) and half the miss confidence
(97.5%), so by the union bound the pair keeps "at most 5% flagged, with 95% confidence". With fewer than 146
human texts no threshold can make that promise, and calibration refuses rather than guess.

## Measured (RAID, results/ai_text/)

Every number is on a writing domain the probe never saw in training (leave one domain out, 8 domains). The promise
is checked over 50 random calibrations per domain: 150 human texts calibrate, the domain's other human texts and
its AI texts test.

| detector, at the 5% / 95% promise | measured false alarms | promise broken in | AI caught | paraphrased AI caught |
|---|---|---|---|---|
| Binoculars alone | 2.0% | 6.2% of calibrations | 51% | 72% |
| probe alone | 1.9% | 2.8% | **82%** | 35% |
| **probe + Binoculars (shipped)** | **1.3%** | **1.3%** | 77% | **74%** |

* Per domain, the shipped pair catches 86-93% of AI text in abstracts, books, news and recipes; 76-80% in
  Reddit and Wikipedia; about 50% in poetry and product reviews. A promise is kept everywhere: no domain broke it
  in more than 4% of calibrations, against the 5% allowed.
* **Unseen generators.** With both the test domain and the test generator's whole family held out of training
  (OpenAI, Meta, Mistral, MosaicML, Cohere), the probe's AUROC is 0.962 on average (0.92 Cohere to 0.996 Meta)
  and it catches 79% at the promise; Binoculars 0.785 and 51% (`raid_generalize.json`, `unseen_family`).
  This was measured for the probe trained without paraphrases; the shipped pair's unseen-family figure is not
  yet measured.
* **A threshold does not carry between kinds of writing.** Set at 1% false alarms on seven domains, the probe
  flagged 29% of human recipes (`raid_qwen3_0.6b.json`, `transfer_1pct`). This is why calibration on the
  customer's own writing is the product, not an option.
* **Cost.** 2.0 s a text (256 tokens, both models, batch of 2) on 4 vCPUs of an Intel Xeon at 2.1 GHz; about
  1,800 texts an hour on one 4-vCPU machine. Calibration on 150 texts: about 5 minutes. On a GPU the two
  0.6B forward passes take milliseconds.

## Limits, stated

* **RAID's human texts are mostly from before ChatGPT**, and the AI texts come from 2023-era models. Human writing
  today is partly AI-edited, and newer models (GPT-4o, Claude, Gemini, Llama 3) are not in the test.
* **Only the paraphrase attack was tested.** RAID's other attacks (homoglyphs, misspellings, synonym swaps,
  whitespace) and humanizer tools were not.
* **Poetry and reviews**: at 5% the pair catches only about half of AI text.
* **Short texts**: the measurements are on texts of 40+ words, read up to 256 tokens. Texts under 64 tokens get a
  note with their verdict.
* **The promise covers human texts like the calibration set.** If a customer calibrates on essays and checks
  emails, the bound does not apply. It is a statement about the share of human texts flagged, not about any one
  text.

## Use

```bash
pip install ".[verifier]"
vt-aitext calibrate --human my_human_texts.jsonl --out calibration.json      # 150+ texts; {"text": ...} per line
vt-aitext check --calibration calibration.json --file to_check.txt
vt-aitext serve --calibration-dir calibrations/ --port 8000                  # API keys: VERIFY_API_KEYS="acme:key1"
```

API (same server hardening, API keys, per-key rate limits and `/metrics` as the grounding verifier):

    POST /v1/aitext/calibrate  {"human_texts": [...150+...], "fpr"?: 0.05, "confidence"?: 0.95}
                               -> the calibration, stored for this API key
    POST /v1/aitext            {"text": "..."}
                               -> {"verdict": "ai" | "not flagged", "fired": ["probe" | "binoculars"],
                                   "probe", "binoculars", "tokens", "promise", "ms"}
    POST /v1/aitext/batch      {"items": [{"text": ...}, ...up to 16]}

Reproduce: `experiments/ai_text_detect.py` (sample RAID, read, evaluate) and `experiments/ai_text_generalize.py`
(unseen generator families, paraphrase, the union).
