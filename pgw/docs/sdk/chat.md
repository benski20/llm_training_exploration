# PGW Chat: private chat on your own computer

`pgw chat` is a chat app built on the client library. It runs on your computer and talks directly to our GPU
service: your prompts and answers exist in plaintext only on your computer; our GPUs compute on masked numbers and
every result they return is checked on your computer before it is used. No server of ours receives your words.

Design and the reasoning behind it: [research/reviews/encrypted-chat-app.md](../../research/reviews/encrypted-chat-app.md).

## Start it

```bash
pipx install "pgw-sdk"              # or pip install in a virtual environment (Python 3.11+)
export PGW_API_KEY=pgw_live_...     # your key; pgw chat asks for it (hidden input) if it is not set
pgw chat                            # opens the chat in your browser
```

The first start sets the model up once on this computer, in the background, while the page shows the progress:

1. **Connecting your key** (seconds): enrollment with our service; your client key is generated here.
2. **Downloading the public model files**: checked against the fingerprint pinned in the library; an interrupted
   download resumes where it stopped.
3. **First-time setup on this machine**: the one-time mask setup, built on your CPU (at 7B 10-16 CPU-minutes on CPUs
   with AMX or AVX-512 VNNI, 1.5-2 CPU-hours with AVX2 only; smaller models proportionally less), then the first batch
   of masks.

You can close the tab during setup; it continues while `pgw chat` runs and resumes after a restart. Later starts take
seconds. `Ctrl-C` in the terminal stops the app (its background mask processes stop with it).

| option | default | meaning |
|---|---|---|
| `--model` | recommended for this computer | the model selected first; the page's model menu switches |
| `--port` | a free one | the port on 127.0.0.1 |
| `--no-browser` | off | print the link (with its one-time token) instead of opening the browser |
| `--no-history` | off | keep conversations in memory only |
| `--factory-weights stream` | `PGW_FACTORY_WEIGHTS`, else `pack` | keep no int8 weights on disk (about 6 GB less at 7B; slightly more network). Recommended on laptops |
| `--prefix-cache on` | `auto` (off with our masked output layer) | reuse earlier turns: faster long chats, but our service can tell that a request continued an earlier one ([below](#what-our-service-can-see)) |
| `--speculation draft_head` | `PGW_SPECULATION`, else `off` | fewer round trips per token (0.5B and 7B; identical answers); your next message waits for the previous answer's remaining rounds |
| `--schedule standard` / `fused` / `auto` | `PGW_SCHEDULE`, else **`auto`** (`standard` with `--prefix-cache on` or stream weights, which the fused schedule does not combine with yet) | `auto` measures your link once at start-up and uses the fused schedule only where it is faster: half the round trips per token (`2 x layers` instead of `4 x layers + 1`) but 1.74x the bytes, so it helps on fast links a few ms or more from the GPU and is skipped on slow uplinks or in the same data centre. Fused answers are slightly different from the standard schedule's (its own fixed, exactly computed reference; at 7B perplexity 9.45 vs 9.35, same top token 96.8 % of the time), so the choice is fixed for the life of the app and shown in its status; pass `--schedule standard` for answers identical to the library's default. The provider must offer the fused schedule, else `auto` uses standard ([Configuration](configuration.md#schedule-standard-or-fused)) |
| `--no-usage-reports` | reports on | stop sending the content-free per-request record (counts, timing, checks passed) |
| `--max-tokens` | 1024 | longest answer |
| `--cache-dir`, `--state-dir` | `PGW_CACHE_DIR`, `PGW_STATE_DIR` | where model data, masks and conversations live |
| `--demo` | off | canned answers, no model or GPU (for trying the interface; the page says "Demo") |

## Choosing a model

The model menu shows each model's memory, disk and first download, and marks the one recommended for this
computer (the largest of 0.5B / 1.5B / 3B that your cores, memory and free disk fit; 7B is your choice):

| model | memory | disk (stream mode / with weights) | first download | round trips per token |
|---|---|---|---|---|
| Qwen2.5 0.5B | 8 GB | 9 / 10 GB | 1 GB | 97 |
| Qwen2.5 1.5B | 8 GB | 11 / 13 GB | 3 GB | 113 |
| Qwen2.5 3B | 16 GB | 14 / 17 GB | 6 GB | 145 |
| Qwen2.5 7B | 32 GB | 20 / 26 GB | 11 GB | 113 |

Planning figures from [Supported models](models.md); the disk includes the mask stock (8 GB per model by default).

## How fast it is

Every token needs one masked round trip to our GPU per layer (97 to 145 trips, above). On a home connection the
round-trip time dominates: at 20 ms, a 7B or 1.5B token takes at least 2.3 s (about 1.3 s with
`--speculation draft_head`); at 5 ms, 0.6 s (0.3 s). So today PGW Chat is a private, careful chat rather than an
instant one; each answer shows its speed in tokens per second. We are working on both levers: fewer round trips per
token, and GPUs placed close to you. While an answer is slow to start, the page says it is waiting for the GPU; you
can stop an answer at any time, and finished answers are saved.

## What is private

The **Private** indicator at the top opens "What is true right now":

* **Your words stay on this computer.** Prompts, answers, the model's working memory and its sampling exist only in
  the `pgw chat` process and the browser tab that `pgw chat` opened.
* **The GPU sees only masked numbers.** Every layer's input gets a fresh one-time mask before it leaves.
* **Every GPU result is checked** here before use. "Computed privately" under an answer means the whole answer was
  produced with every check passing; a failed check discards the answer and pauses the model (treat it as a security
  event and tell us).
* **History** is saved as files readable only by your user account in `~/.local/state/pgw/chat/conversations/`
  (or `--state-dir`), protected at rest by your disk encryption. Delete a conversation from the sidebar (the trash
  icon), or everything with the trash icon at the bottom. `--no-history` saves nothing.

### What our service can see

Which model you use, how many tokens each request has, when you send requests and how much traffic they make, your
key (that is, your account) and your IP address. With `--prefix-cache on` it can also tell that a request continued
an earlier one and roughly how long the conversation is. It never sees text. We do not describe this as
"end-to-end encrypted" or "anonymous": our GPUs compute on your (masked) data, and your key identifies your account.

## Security of the local app

* The server listens on `127.0.0.1` only. Each launch draws a random token; the page receives it in the link's
  fragment (never sent over the network) and every API call must carry it, so other users of the computer and other
  web pages cannot use your chat. Requests naming another host (DNS rebinding) or coming from another origin are
  refused.
* The page loads nothing from the internet: no fonts, scripts or images from elsewhere (strict Content-Security-
  Policy). Answers are shown as text with simple formatting; images in answers are never loaded, links open only when
  you click them and without a referrer.
* The app writes no logs containing your words, disables core dumps of its process, and never puts the token on a
  command line. Use full-disk encryption: the operating system may write memory to swap or hibernation files.

## Troubleshooting

| what the page says | what to do |
|---|---|
| "Open PGW Chat from your terminal" | the tab has no token (a bookmark, a new tab): run `pgw chat` again or use the link it printed |
| "Provider unreachable" / "Provider is not making progress" | our service cannot be reached or its GPU is stuck; your chats are safe, the app retries by itself ([Reliability](reliability.md)) |
| "Your API key was refused" | the key is wrong, revoked or expired: set a new `PGW_API_KEY` and restart |
| "Integrity check failed: answer discarded" | a GPU result failed its check; nothing unverified was shown. Report it with the time |
| setup is slow | the one-time setup is CPU-bound; on AVX2-only CPUs it takes hours at 7B; pick a smaller model or prepare it ahead with `pgw sdk prepare --model ...` |

`pgw sdk status` shows the same state from the terminal.

## For developers

The web app lives in `pgw/chat/` (React + TypeScript + Vite) and is built into `src/pgw/chat/static/`
(`scripts/build-chat.sh`; `npm run dev:mock` runs it with canned data and no server, `?state=setup|stalled|slow|empty`
selects a state). The server is `src/pgw/chat/server.py` (standard library only); its API is documented in the
module. Tests: `tests/chat/` (server security, status mapping, the build, a Chromium smoke test, and an end-to-end
test against the CPU test provider that checks the streamed answer is bit-identical to the library's with the same
schedule).
