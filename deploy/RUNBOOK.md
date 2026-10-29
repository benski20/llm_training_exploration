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
modal run deploy/modal_app.py::download          # once: fill the model cache on a CPU container (no GPU billed)
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

From GitHub instead (the Modal client speaks gRPC, which some sandboxes cannot reach): add the
repository secrets `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET`, then run the `deploy-demo` workflow from
the Actions tab, or commit `deploy/request.json` with `{"action": "deploy"}` or `{"action": "stop"}`.
The workflow fills the cache on CPU and deploys; the job log prints the URL.

## Settings (environment variables at deploy time)

| variable | default | meaning |
|---|---|---|
| `VERICHAT_MODEL` | `unsloth/Qwen3-32B-bnb-4bit` | the chat model; the head was trained on this checkpoint |
| `VERICHAT_HEAD` | `results/kaggle/risk-32b2/risk/Qwen3-32B-bnb-4bit_risk_head.pt` | the risk head; its `.serving.json` ships with it |
| `VERICHAT_GPU` | `L40S` | 48 GB fits the 4-bit 32B model, the 3B detector and the KV cache |
| `VERICHAT_SCALEDOWN` | `120` | idle seconds before the GPU container stops (each idle minute is billed) |
| `VERICHAT_REVISION` | latest | the model's Hugging Face commit; pin it for a stable public demo (supply chain) |
| `VERICHAT_WIKIPEDIA` | `1` | look doubted answers up on Wikipedia and answer again from it, with citations (0: off) |
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

An L40S is about $1.95 per hour (about 3.3 cents a minute), billed only while a GPU container is up:
- Nobody using it costs nothing. The container scales to zero 2 minutes after the last request.
- A cold start loads about 25 GB from the cache volume onto the GPU, about 1-2 minutes, roughly 5 cents.
- A typical tester (a few messages over a few minutes) costs roughly 20-40 cents.
- `max_containers=1` caps it at one GPU, so continuous traffic tops out near $47 a day.
- Weights download on a CPU container (cents), not the GPU. The cache volume holds about 25 GB.
- Set a workspace spending limit in the Modal dashboard (Settings, usage and billing) before sharing
  the URL. That limit is the real guard against runaway cost.

## Measured (Modal, one L40S)

- Cold start: about 48 s to load the model; the first request after idle waits that long.
- One user, short question: 1.1 s per reply (p95 1.45 s).
- Four users at once: all succeed, 13 s median (p95 14 s); a batch ends with its longest reply.
- My testing so far: about 25 GPU minutes over three deploys, roughly 80 cents, plus cents of CPU.
- Lost requests: about 1 in 40-100 got no reply. On 2026-10-29 one of 7 load-test requests reached the
  server (logged on arrival) and was never answered, while the others were. The client gave up about
  4 minutes later. The server now logs `recv #n` and `sent <code> #n <s>` for each POST. A request body
  that stops arriving gets a 408 and an `http error` line instead of a silent close. The page retries a
  dropped connection once and says so when a stream ends early. So the next loss shows where it stopped.
- The earlier losses ended at about 258-309 s ("RemoteDisconnected") or at the client's own 400 s
  timeout, so requests sat on the server about 300 s. The connection cut-off equalled the model
  deadline (both 300 s), so the socket was shut just before the 503 could be written. The cut-off now
  comes 30 s later. Streams send a `: ping` every 10 s. The page aborts after 45 s of silence and
  retries once if no text has arrived. A browser check against a fake server
  (`experiments/page_check/`) confirms it: a request stalled at the socket recovers in 47 s, and a slow
  25 s reply arrives intact.

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
