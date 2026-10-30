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
| `VERICHAT_RERANK` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | reorders lookup passages and searches 5 pages ("" for BM25 over 3 pages) |
| `VERICHAT_VERIFY` | `1` | check each reply's claims against Wikipedia, judged by the chat model (0: off) |
| `VERICHAT_ROUTE` | `1` | sort each message (information / math / code / chat): information answered from one lookup and judged by its sources, math by re-computed arithmetic, code and chat get no fact verdict (0: off) |
| `VERICHAT_SEARCH` | "" | also search the web: `brave` or `tavily` (needs a key, below) |
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

## Live: 40 TriviaQA questions with claim checks on (2026-10-29, run 40919135629)

- Accuracy rose from 0.711 closed-book to 0.789 served. Of 9 lookups, 3 turned wrong answers right and
  none broke a right one (`results/correction_rate_32b_v5.json`).
- The claim checks on the live replies: AUROC 0.80 on the 30 judged first claims.
  - Right answers' first claims: 21 supported, 2 unclear, 2 unsupported, 3 no evidence, 2 no claims.
  - Wrong answers: 3 supported, 2 unsupported, 3 no evidence.
- 2 of 40 requests were still lost (RemoteDisconnected after about 270 s). Median reply time 35 s at 4
  concurrent clients.
- The per-request log (`recv`/`body`/`sent`) found the cause:
  - The server sent a complete 200 for all 40 requests.
  - For one of the two lost ones, Modal's proxy then failed while reading the reply back from the local
    socket (`ClientPayloadError: Response payload is not completed`).
  - That client got nothing and was dropped about 270 s later.
- Fix: the demo now serves in-process as an ASGI app (`vt/asgi.py`, `VERICHAT_ASGI=1`). Modal calls the
  same handler directly, with no socket in between. Locally behind uvicorn, the stress test lost 0 of 640
  (`results/server_stress_asgi.json`).
- First in-process deploy (run 40923362259):
  - The smoke test passed. 9 of 10 load-test requests were answered (1 client about 3 s, 4 clients 3-20 s).
  - One request never reached the container: no `recv` line.
  - The container scaled down 2 minutes after the last reply while that client was still waiting. Modal
    had not counted the request as pending work, so it was lost in Modal's routing, before our code.
- So far both loss modes sit in Modal's layer:
  - web_server: its proxy failed reading a finished reply.
  - asgi_app: it never delivered the request.
- Our server answered every request that reached it: all logged live requests, and 640 + 640 in the
  local stress tests.
- The page copes with a lost request: no bytes or heartbeat for 45 s, then it aborts and retries once
  (`experiments/page_check`). API clients should stream, or set a timeout and retry.
- `VERICHAT_ASGI=0` restores the web_server route.
- Deploy with the paced lookups and the overall verdict (2026-10-29):
  - First attempt (run 40969988659): chat and stream passed, then the long-chat check failed 0.3 s after
    sending. The server log has no `recv` line for it: Modal's edge rejected it before it reached the
    app.
  - The smoke test now prints status codes and retries a gateway-type failure once, counting it.
  - Second attempt (run 40970971211): every check passed with no retries, including thinking mode,
    moderation, /judge, the page and /metrics.
  - Load test: 10 of 10 (1 client 4.3 s, 4 clients 19.7 s median, p95 23.5 s).

## Streaming through Modal (2026-10-30)

200 live questions through the retrying client (run 40983398601, `results/correction_rate_32b_v6_200.json`):
- Accuracy rose from 0.744 closed-book to 0.805 served. Of 33 lookups, 13 fixed a wrong answer and 1 broke
  a right one.
- Live source checks: AUROC 0.813 on the 172 judged first claims.
- 13 requests went silent for 45 s and were answered on the retry; 5 failed.

The server log has a `recv` but no `sent` for each silent request: they arrived and were being answered,
but the client heard nothing. A CPU-only probe (`deploy/stream_probe.py`, `streamprobe` action) served the
same handler around a 50 s fake bot both ways:
- web_server: headers after 4 s, a ping every 10 s, the reply at 54 s.
- asgi_app: not even a status line within 90 s.

Modal does not pass the in-process route's stream through as written, so every reply longer than the
client's 45 s watchdog looked dead and was sent again, adding load. The default is back to web_server
(`VERICHAT_ASGI=0`). Its rare proxy-read failure remains, and the page and `vt.client` retry it.

## Lookups (Wikipedia)

- On 200 TriviaQA questions, the share whose answer appears in the 3 passages the model gets
  (`experiments/retrieval_recall.py`, GitHub runner, no model):
  - lead sections: 0.42
  - best passages of the top pages' full text: 0.56 (now the default)
  - at most one passage per page: 0.53
  - a later run, among the 140 questions Wikipedia did not rate-limit: full text 0.59, 5 pages 0.60,
    reranked 0.66, 5 pages reranked 0.71 (now the default; `results/retrieval_recall_v4.json`)
- The full text costs about 0.4 s more per lookup and about 90 more prompt words.
- Wikimedia throttles clients without a contactable User-Agent. With the old one, lookups from a
  GitHub runner found nothing for 196 of 200 questions. The User-Agent now names the project URL.
  `/health` shows `retrieval_errors`, so throttling on Modal is visible.
- A search with no hits is retried with the question's content words, then its names and numbers.
- Raising the lookup threshold is not worth it. At 0.80 it would look up 80% of wrong trivia answers
  (48% at 0.65), but also 29% of everyday replies (7% at 0.65), each about 10 s slower.

## Claim verification (`--verify-claims`; on for Modal)

Each reply's check-worthy claims (up to 3) are looked up and judged by a chat model reading the evidence:
P(Yes) that the evidence supports the claim. On the 32B's labeled replies (`experiments/evidence_verify.py`,
CPU judge Qwen2.5-1.5B, 185 factual and 56 everyday; `results/evidence_verify_v3_judge1.5b.json`):

| AUROC, wrong vs right | TriviaQA | NQ | all | everyday replies marked |
|---|---|---|---|---|
| internal probe | 0.778 | 0.573 | 0.705 | 7% |
| small NLI model (DeBERTa) | 0.71 | 0.61 | 0.675 | 16% |
| 1.5B judge | 0.77 | 0.626 | 0.712 | 7.1% |
| judge where evidence exists, averaged with the probe; else the probe | 0.791 | 0.616 | 0.732 | |

Where evidence was found (81% of factual replies), the 1.5B judge beats the probe: 0.741 against 0.706
(NQ 0.648 against 0.552).

The 32B as the judge, through the serving code on the same saved evidence (`judge32` action, about 3 GPU
minutes; `results/evidence_verify_v4_judge32b.json`):

| AUROC, wrong vs right | TriviaQA | NQ | all | everyday replies marked |
|---|---|---|---|---|
| internal probe | 0.778 | 0.573 | 0.705 | 7% |
| 32B judge, first claim | 0.793 | 0.64 | 0.74 | 3.6% |
| 32B judge + probe | 0.81 | 0.621 | 0.741 | |

It judged 329 claims in 88 s (batches of 8), about 0.3 s each.

These differences are not yet significant. A paired bootstrap over the 149 replies with evidence (1.5B
judge minus probe):

| | difference | 95% interval | P(judge better) |
|---|---|---|---|
| all | +0.035 | -0.06 to +0.12 | 0.77 |
| NQ | +0.096 | -0.06 to +0.26 | 0.88 |
| TriviaQA | +0.013 | -0.09 to +0.12 | 0.61 |

Settling it needs several hundred more labeled 32B replies. `judge_eval` now prints per-reply scores
(`ROWS ...`) for the same paired test.

## Judge vs probe at scale (2026-10-29, `study` action, `results/study_32b_v1.json`)

The served 32B answered 300 TriviaQA and 300 NQ questions closed-book. Accuracy was 70.7% and 37.0%.

| AUROC, wrong vs right | TriviaQA | NQ | all |
|---|---|---|---|
| probe, all replies | 0.736 | 0.624 | 0.681 |
| probe, replies with evidence | 0.756 | 0.643 | 0.699 |
| 32B judge, replies with evidence | 0.845 | 0.676 | 0.767 |
| judge minus probe (95% CI) | +0.089 (-0.007, +0.192) | +0.034 (-0.079, +0.151) | +0.069 (-0.006, +0.144) |
| P(judge better) | 0.965 | 0.72 | 0.962 |
| combined, fitted on half, tested on half | 0.820 (probe 0.751) | 0.657 (probe 0.615) | 0.738 (probe 0.675) |

- The best combination weights the judge 0.8 to 0.9 where evidence exists and uses the probe elsewhere.
- Only 46% of replies got judged evidence:
  - 49% had claims, but Wikipedia rate-limited the run (about 550 search and 480 page HTTP 429s, plus
    timeouts).
  - 5.5% had no claim the filter recognised: lowercase answers like "fresco" or "blue".
- Fixes since:
  - Wikipedia requests are paced, at most 4 in flight and 50 ms apart.
  - Page texts are cached.
  - A sentence with bold text (the model bolds its answer) counts as check-worthy.
  - A rerank crash on a single passage is fixed.
- The `rejudge` action redoes the lookups for the saved answers on CPU (no GPU billed), then only the
  judging on GPU, a few minutes.

Rejudged with those fixes (run 40965200357, `results/study_32b_v2_rejudge.json`; about 3.5 GPU minutes of
judging):
- Evidence was found for 90% of answers (46% before). HTTP 429s fell from about 1000 to 16.

| AUROC, wrong vs right | TriviaQA | NQ | all |
|---|---|---|---|
| probe, replies with evidence | 0.747 | 0.624 | 0.686 |
| 32B judge, replies with evidence | 0.871 | 0.727 | 0.802 |
| judge minus probe (95% CI) | +0.124 (0.049, 0.194) | +0.103 (0.027, 0.179) | +0.117 (0.066, 0.168) |
| best combination, held-out half | 0.864 (probe 0.751) | 0.738 (probe 0.615) | 0.813 (probe 0.675) |

- The judge's lead is significant on both datasets.
- The verdict weights the judge 0.9 (0.9 and 1.0 tie on both halves). Doubt above 0.6 marks a reply likely
  wrong: about 46% of wrong answers and 12% of right ones.

## Web search (`--web-search brave|tavily`, `vt/websearch.py`)

Web search adds a second source next to Wikipedia. Lookups and claim checks query both in parallel, pool
the passages, and rank them together (BM25, then the reranker). Each passage is labeled with its page
title and site.

- **Brave Search API:** returns results, and the server fetches each page.
- **Tavily:** returns the page text with each result.

Page fetching is narrow (protection against server-side request forgery):
- http(s) only.
- Every host, redirects included, must resolve to public addresses: no loopback, private, link-local
  (e.g. the cloud metadata address) or reserved ones.
- 2 MB cap, 5 s timeout, HTML or plain text only.
- Page text only reaches the model inside the judging prompt, as quoted evidence.
- Failures are counted in `/health` `retrieval_errors`.

The key never goes in the repo or the image. Either:
- create a Modal secret named `verichat-search` holding `VERICHAT_SEARCH_KEY` (dashboard, Secrets), and
  deploy with the `search` input set; or
- pass `search_key` as a one-off workflow input. Like the token inputs, it stays in that run's stored
  event data, so rotate it afterwards.

Both providers have free tiers with monthly limits; check their current pricing. Each lookup costs one
search per claim, up to 3 per reply.

### Three-way judging and fairer labels (run 40975943782, `results/study_32b_v3_threeway.json`)

The saved 600 answers were looked up again (Wikipedia rate-limited the burst: about 600 HTTP 429s;
evidence for 47%) and judged both ways on the same evidence. Each answer was also graded against its
reference answers by the 32B; this agrees with string matching on 87% of answers. NQ accuracy is 48% by
grading and 37% by string match.

| AUROC (graded labels) | TriviaQA | NQ | all |
|---|---|---|---|
| probe, all replies | 0.759 | 0.745 | 0.751 |
| judge vs probe, replies with evidence | 0.905 vs 0.792 | 0.754 vs 0.742 | 0.835 vs 0.769 |
| judge minus probe (95% CI) | +0.113 (0.034, 0.204) | +0.013 (-0.100, +0.123) | +0.066 (-0.003, +0.135) |
| combination, held-out half | 0.833 | 0.742 | 0.802 |

- Three-way judging is no better than yes/no (difference -0.009 to +0.001, CIs span 0). It answers "does
  not say" for two thirds of claims. Yes/no stays the default; `three_way=True` remains an option.
- String labels made the probe look weak on NQ (0.62); graded, it is 0.75. The judge's clear advantage is
  on TriviaQA-style facts (+0.11, significant). Overall it is about +0.07, borderline.
- Bulk lookups from Modal trip Wikipedia's limits even when paced. Demo traffic is far lighter, but a
  search API with a key (`--web-search`) is the robust source for production.

## Clients

`vt.client.chat(url, messages)` streams the reply. If 45 s pass without a byte (the server sends a
heartbeat every 10 s), it abandons the request and sends it again once, as the page does. A request lost
in the platform's routing then costs about 45 s instead of a timeout. `experiments/correction_rate.py
--retry` uses it and counts `retried_and_answered`.

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
