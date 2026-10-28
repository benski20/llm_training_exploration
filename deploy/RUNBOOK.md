# VeriChat demo runbook (Modal)

The demo serves Qwen3-32B (4-bit) with its risk head and measured serving settings, the 3B detector
for `/judge`, and a toxicity filter. It runs in one L40S container that scales to zero.

## Before deploying

```bash
pip install "modal>=1.0"
export MODAL_TOKEN_ID=ak-...  MODAL_TOKEN_SECRET=as-...      # or: modal token new (stores ~/.modal.toml)
python -m pytest -q tests/                                   # 182 pass, 2 skip
bash deploy/smoke_test.sh http://127.0.0.1:8095              # against a local 0.5B server, see below
```

Local server for the smoke test (small model, CPU):

```bash
python -m vt.verichat_server --model Qwen/Qwen2.5-0.5B-Instruct --head <0.5B risk head> \
    --detector <0.5B detector head> --moderation --rate-per-minute 0 --port 8095
```

## Deploy, check, stop

```bash
modal deploy deploy/modal_app.py                 # prints https://<workspace>--verichat-serve.modal.run
bash deploy/smoke_test.sh https://<workspace>--verichat-serve.modal.run
VERICHAT_RATE=0 modal deploy deploy/modal_app.py   # for a load test only (no per-client rate limit)
python experiments/load_test.py --url https://<...>.modal.run --clients 1,8,16 --per-client 4
modal deploy deploy/modal_app.py                 # back to the public rate limit
modal app logs verichat                          # live logs
modal app stop verichat                          # stop serving (the URL goes dark)
```

The first request after idle triggers a cold start. The first deploy downloads about 19 GB into the
`verichat-hf-cache` volume, which takes several minutes. Later cold starts only load the model, about
a minute. Until it is ready, `/health` fails and the smoke test keeps waiting (up to 15 minutes).

## Settings (environment variables at deploy time)

| variable | default | meaning |
|---|---|---|
| `VERICHAT_MODEL` | `unsloth/Qwen3-32B-bnb-4bit` | the chat model; the head was trained on this checkpoint |
| `VERICHAT_HEAD` | `results/kaggle/risk-32b2/risk/Qwen3-32B-bnb-4bit_risk_head.pt` | the risk head; its `.serving.json` ships with it |
| `VERICHAT_GPU` | `L40S` | 48 GB fits the 4-bit 32B model, the 3B detector and the KV cache |
| `VERICHAT_SCALEDOWN` | `300` | idle seconds before the container stops |
| `VERICHAT_RATE` | `10` | POSTs per client per minute (0 only for load tests) |
| `VERICHAT_RISK_THRESHOLD` | the head's | only for the sentence-risk mode; the served mode judges sentences by their claims |

Server flags set in `modal_app.py`:
- `--moderation`;
- `--rate-per-minute` from `VERICHAT_RATE` (10 per client);
- `--trust-proxy` (client address from `X-Forwarded-For`);
- `--deadline 300` (thinking replies are slow).

Decoding uses Qwen3's advice: temperature 0.7, top-p 0.8, top-k 20, presence penalty 1.5. Thinking is
off by default, with a per-request switch in the page and the API.

## Costs

An L40S is about $1.95 per hour while a container is up. With scale-to-zero you pay for traffic plus
5 idle minutes after the last request. Set a spending limit in the Modal dashboard before sharing the
URL.

## What the demo can and cannot claim

These are the held-out numbers (`VISION.md`, `results/kaggle/chateval-32h/`):
- 7% of fact-heavy everyday replies get an underline (under 1% of sentences).
- On new trivia questions: AUROC 0.78. It flags 37% of wrong answers and 12% of right ones.
- On Natural Questions it is near chance (0.57).

It is a warning light, not a fact checker: errors the model is confident about pass unflagged. With
documents, provenance and citations are exact for the forward pass and the risk head is strongest
(RAGTruth sentence AUROC 0.90).

## Watching it

- `GET /health`: model layer, thresholds, whether `/judge` and thinking are available.
- `GET /metrics` (Prometheus): requests by path and status, latency, batch sizes. Watch the 503s (over
  capacity or deadline) and the p95 latency.
- To roll back, redeploy an earlier commit: `git checkout <sha> && modal deploy deploy/modal_app.py`.
