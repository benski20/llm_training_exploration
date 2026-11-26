# Shipped draft heads: provenance and data licences

The `*.pgwhead` files here are the trained draft heads for `speculation="draft_head"` (format
`pgw-feature-head-v2`, int8). Each is pinned by SHA-256 and size in `pgw.sdk.draft_heads.PINNED_DRAFT_HEADS`. A
head only chooses which draft tokens a verify pass carries; every answer is the target model's own, bit for bit.

| file | target checkpoint (the SDK's model pin) | d | V | size (bytes) |
|---|---|---|---|---|
| `qwen2.5-0.5b-instruct-d256-v16384-r2.pgwhead` | `Qwen/Qwen2.5-0.5B-Instruct` @ `7ae557604adf67be50417f59c2c2f167def9a775` (Apache-2.0) | 256 | 16384 | 5289977 |
| `qwen2.5-7b-instruct-d384-v16384-r2.pgwhead` | `Qwen/Qwen2.5-7B-Instruct` @ `a09a35458c702b33eeacc393d103063234e8bc28` (Apache-2.0) | 384 | 16384 | 11191929 |

## How they were made

`benchmarks/feature_head_distill.py`, run on Kaggle T4 x 2 (`benchmarks/kaggle/build_head.py`, jobs `pgw-head-05b`
and `pgw-head-7b`), 2026-11-26. The target model answers the training prompts itself (greedy, 25% sampled at
T = 0.7). The head learns from the target's own final-norm features and greedy tokens. The datasets provide only
the prompts. No dataset answer is used, and no text of any dataset is stored in a head file. Each file also records
the list below under its `training_data` key.

## Datasets (Hugging Face Hub, at pinned revisions)

| dataset | revision | licence | used for |
|---|---|---|---|
| `HuggingFaceH4/ultrachat_200k` | `8049631c405ae6576f93f445c6b8166f76f5505a` | MIT | training prompts: first user turn of `train_sft` (75%); held-out chat evaluation: `test_sft` |
| `OpenAssistant/oasst2` | `179dd21fc55192153d94adb0e0ce8f69e222bf75` | Apache-2.0 | training prompts: root prompter messages of `train`, all languages (15%) |
| `bigcode/self-oss-instruct-sc2-exec-filter-50k` | `356bb069eee815daa6e23e9a282eeefe1490ad44` | ODC-By 1.0 | training prompts: the `instruction` field, code (10%) |
| `google-research-datasets/mbpp` | `4bb6404fdc6cacfda99d4ac4205087b89d32030c` | CC BY 4.0 | held-out code evaluation only (`full/test`, 120 problems); never trained on |

All four allow commercial use. None is share-alike or non-commercial. Attribution: UltraChat (Ding et al., 2023,
OpenBMB / Hugging Face H4), OpenAssistant Conversations (LAION / OpenAssistant contributors), StarCoder2
Self-Instruct (BigCode), MBPP (Austin et al., 2021, Google Research).

Notes:

- UltraChat's turns were written by gpt-3.5-turbo. The dataset is published under MIT. We use only its user
  prompts as inputs to the target model. Self-OSS-Instruct's instructions were written by StarCoder2-15B from
  permissively licensed seed code.
- **Replaced (no longer used):** the heads shipped earlier on 2026-11-26 were also trained on Stanford Alpaca
  (`tatsu-lab/alpaca`, CC BY-NC 4.0, non-commercial) and Dolly 15k (`databricks/databricks-dolly-15k`, CC BY-SA
  3.0). Their code evaluation also used Alpaca. Those files are removed from the package. Dolly was dropped as
  well, so that the shipped weights carry no share-alike obligation; OASST2 (Apache-2.0) replaces it.

## Held-out quality (exported head, float kernel; same held-out answers for the old and new heads)

Alpha is the depth-1 top-1 agreement with the target's greedy token. Tokens per pass come from an exact replay of
the target's greedy answers through the strict chain rule, head + prompt lookup (the SDK's `draft_head` mode), at
`spec_rows` R = 2 / 3 / 4. Chat: 200 UltraChat `test_sft` prompts. Code: 120 MBPP test problems.

| model | head | alpha chat / code | tokens per pass, chat, R = 2 / 3 / 4 | code, R = 2 / 3 / 4 |
|---|---|---|---|---|
| 0.5B | earlier (Alpaca + Dolly in training), removed | 0.477 / 0.464 | 1.52 / 1.78 / 1.91 | 1.49 / 1.75 / 1.85 |
| 0.5B | **shipped (`-r2`)** | 0.472 / 0.559 | 1.51 / 1.77 / 1.89 | 1.56 / 1.87 / 2.03 |
| 7B | earlier (Alpaca + Dolly in training), removed | 0.474 / 0.506 | 1.46 / 1.66 / 1.74 | 1.51 / 1.73 / 1.82 |
| 7B | **shipped (`-r2`)** | 0.467 / 0.635 | 1.45 / 1.64 / 1.72 | 1.61 / 1.94 / 2.12 |

Open chat is unchanged within 0.01 alpha. Code is higher, because code instructions are now in training and the
old head saw only Alpaca's code prompts. Raw results: `benchmarks/results/feature_head/kaggle_05b_r2/` and
`kaggle_7b_r2/`.
