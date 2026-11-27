# A private chat app on the PGW client library (2026-11-27)

Status: design note plus a working prototype of the recommended option (`pgw chat`: a localhost server and a web
UI in `pgw/chat/`, served from `src/pgw/chat/`). User guide: [docs/sdk/chat.md](../../docs/sdk/chat.md).

## The principle this design keeps

The library's guarantee is cryptographic, not organisational: **plaintext exists only in the client process**; the
GPU provider computes on one-time-masked data and every result it returns is checked (Freivalds, range and p31
checks) before use. A consumer chat app is worth building only if it keeps exactly that guarantee: **nobody but
the user can read the chats**, including us. Onboarding may differ from the SDK's; the trust model may not.

So the client half of the library runs **on the user's own device** and talks directly to our hosted GPU provider.
No server of ours ever receives a prompt or an answer.

## Recommendation

Build **option A**: a local app. Ship it first as `pgw chat` (a command that starts a loopback-only server and opens
the chat in the user's browser; built and tested here), then as a desktop app that wraps the same server and UI
(Tauri preferred, design below). Present it as **private chat that is honest about speed**: at today's round-trip
count it is a slow, careful chat over a home connection (well under one token per second at 7B and 20 ms RTT), not
an interactive one. Interactive speed comes from the two workstreams that cut that arithmetic: fewer trips per
token and GPUs placed close to users.

Rejected: a hosted chat backend (option B) in any form, including one inside a confidential VM. Deferred: an
in-browser client (option C). Reasons below.

## The speed arithmetic (read this first)

Every generated token runs the whole model, and every linear layer of it is one masked round trip to the GPU:
`4L + 1` trips per token with the masked LM head. Trips are sequential (each needs the previous one's checked
result), so the network round-trip time (RTT) multiplies directly:

| model | trips per token | 5 ms RTT | 20 ms RTT | 50 ms RTT |
|---|---|---|---|---|
| 0.5B (24 layers) | 97 | 0.49 s/token (2.1 tok/s) | 1.9 s/token (0.52 tok/s) | 4.9 s/token (0.21 tok/s) |
| 7B (28 layers) | 113 | 0.57 s/token (1.8 tok/s) | 2.3 s/token (0.44 tok/s) | 5.7 s/token (0.18 tok/s) |
| 0.5B, draft head (~1.7 tokens per pass) | 57 per token | 0.29 s (3.5 tok/s) | 1.1 s (0.88 tok/s) | 2.9 s (0.35 tok/s) |
| 7B, draft head (~1.7 tokens per pass) | 66 per token | 0.33 s (3.0 tok/s) | 1.3 s (0.75 tok/s) | 3.3 s (0.30 tok/s) |

These are lower bounds: RTT x trips only, before the provider's compute and the client's CPU per trip (about 1-2 ms
at 7B on a laptop). The 1.5B model has the same 113 trips as 7B (28 layers); 3B has 145. A 300-token answer at 7B
and 20 ms RTT takes about 11 minutes (about 7 with the draft head); at 5 ms, about 3 minutes (1.7). In a datacenter,
next to the GPU (0.2 ms RTT), the same 7B model is modelled at about 17 tok/s ([scaling](../../docs/sdk/scaling.md)).

Consequences:

* **A laptop or desktop app is practical for private, slow or batch use, not for interactive chat at today's trip
  count.** Users must be told before they start (the app shows tokens/s and explains slow streaming), and the app
  must stay pleasant while slow: it streams token by token, shows "Waiting for the GPU" with the reason, keeps the
  stop button live, and lets the user leave and come back (answers are saved when done).
* The prompt costs one pass of the same trips (all prompt rows in each trip), so time to first token is about one
  token's time plus the prompt's masks; long conversations cost mostly masks and upload, not extra round trips.
* **Two levers change the picture**, and both are separate workstreams:
  1. *Fewer trips per token*: research is fusing linears across RMSNorm boundaries (fewer sequential trips per
     layer) and strengthening speculation (more tokens per pass than the shipped draft head's ~1.7).
  2. *Lower RTT*: regional GPU placement. 5 ms is a same-metro provider; 20-50 ms is typical across a country or a
     continent. Every halving of RTT halves the time per token.
* Speculation is the strongest lever available today, and at home RTTs it is clearly a win (masks are cheap there:
  at 7B, 0.75 tok/s at ~1.7 tokens per pass with `spec_rows=2` is under one mask slot per second, a few
  percent of one factory core at 35 ms per slot). Its cost is the
  strict-mode tail: after an answer, the rounds the answer did not need still run on random data (so the provider
  cannot tell where the answer ended), and the user's next request waits for them. At 20 ms RTT that tail is up to
  ~40 % of the answer's time. The app therefore offers speculation as an opt-in (`pgw chat --speculation
  draft_head`, shipped heads for 0.5B and 7B) until the tail is shortened; a product decision, listed below.

## Option A: local app (recommended)

### How it works

```
browser tab (UI)  --http://127.0.0.1:<port>, token-->  pgw chat (Python, this machine)
                                                          |  pgw.OpenAI: tokenizer, KV cache, norms, sampling,
                                                          |  masking, checks, mask factory (background, low prio)
                                                          v
                                           our GPU provider (TLS, pinned CA): masked rows only
```

* `pgw chat` (`src/pgw/chat/`) creates one `pgw.OpenAI` client, starts the model's first-run setup in the background,
  serves the built UI on `127.0.0.1` and opens the browser. The UI talks only to that server; the server talks only
  to the SDK; the SDK talks only to the provider named by the user's key (pinned CA).
* Conversations are 0600 JSON files in a 0700 directory under the SDK state directory
  (`~/.local/state/pgw/chat/conversations/`), or memory only with `--no-history`. Delete per conversation or all.
* Every answer carries a "Computed privately" mark (the request completed with every GPU result checked), tokens/s
  and time; the top bar has a quiet "Private" indicator that opens a "What is true right now" screen.

### Privacy analysis

| where | what is there | who can read it |
|---|---|---|
| the `pgw chat` process | prompts, answers, KV cache, sampling, masks, keys | the user (and anyone controlling their machine) |
| the browser tab | the conversation being shown | the user; other web pages cannot reach the API (token, Origin, CSP) |
| conversation files | the saved conversations, plaintext JSON, 0600 | the user's OS account; protected at rest by disk encryption only |
| the GPU provider | masked rows, encrypted mask material, metadata | the provider sees model, token counts, timing, traffic volume, the key (= the account), IP |
| our sign-up / account service | email or anonymous account id, key id, usage counts | us; never content |

What the app must therefore not claim: "end-to-end encrypted" (the provider computes on the data rather than
relaying it, and the phrase suggests a property about relays that is not the one we offer), "anonymous" (the key
identifies the account to the provider), "no metadata" (counts and timing are visible). What it may claim, and does
(`chat/src/lib/privacy.ts`, unit-tested to never say "end-to-end"):

* "Your words stay on this computer. Our GPUs only see masked numbers."
* "Every GPU result is checked on this computer before it is used. N answers verified since pgw chat started."
* "What the GPU service can see: which model you use, how many tokens each request has, when you send them and how
  much traffic they make" (plus the content-free usage record when `usage_reports` is on).
* The prefix-cache and speculation lines below, when those are on; "Demo mode" when it is a demo.

Process hygiene: `pgw chat` sets `RLIMIT_CORE=0` and `PR_SET_DUMPABLE=0` (no core dumps, no same-user ptrace), logs
no content (the HTTP request log is off; errors are logged by type), and never puts the token on a command line (the
browser is opened through a 0600 redirect file, as Jupyter does). Swap and hibernation files are the OS's: the user
guide says to use full-disk encryption.

### The localhost server's security

A server on 127.0.0.1 is reachable by every local user and, through DNS rebinding and cross-site requests, by every
web page the user visits. The prototype closes each path (tests: `tests/chat/test_server_security.py`,
`tests/chat/test_chat_browser.py`):

* bound to `127.0.0.1` only (any other address is refused at construction);
* a 256-bit token per launch, required on every `/api/` call as `Authorization: Bearer` (constant-time compare). It
  reaches the tab in the URL fragment (never sent to a server), moves to `sessionStorage`, and is stripped from the
  address bar;
* `Host` must be `127.0.0.1:<port>` or `localhost:<port>` on every request (DNS rebinding); `Origin`, when sent, must
  be this origin and is required on state-changing calls; `Sec-Fetch-Site` must be `same-origin`/`none`; bodies must
  be JSON; CORS preflights are refused;
* a strict CSP on every response (`default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self';
  frame-ancestors 'none'; ...`), no inline script or style, no external resource, self-hosted fonts with their
  OFL licences; `no-referrer`, `nosniff`, `X-Frame-Options: DENY`, COOP/CORP same-origin;
* answers are rendered by our own Markdown renderer into React elements (never HTML injection); links only for
  http(s)/mailto, opened without referrer; **images are never loaded** (a model answer must not be able to make the
  browser fetch an attacker's URL with data in it, the classic Markdown-image exfiltration); `connect-src 'self'`
  enforces the same for scripts;
* conversation ids are validated before any path is built; symlinks and oversized files are ignored.

### Onboarding for consumers

1. **Install.** Today: `pipx install "pgw-sdk"` then `pgw chat`. Product: a signed desktop app (below).
2. **Account and key.** A sign-up page on our account service (email + password, or passkey; optionally an
   anonymous account with a proof-of-work or a payment token) creates a consumer account and issues **one SDK key per
   user** through the existing admin API (`POST /pgw/v1/admin/orgs/{org}/keys` under a consumer organisation, with
   per-key quotas: `daily_token_quota`, `rows_per_s`, `max_installations`). The desktop app gets the key with a
   device-code flow (the app shows a code and opens the sign-in page; the account service hands the key to the
   app's polling request once the user approves), so the user never copies a secret. The key is stored in the OS
   keychain (Keychain, DPAPI/Credential Manager, libsecret), not in a file. Enrollment, CA pinning and installation
   limits already exist on the provider (`POST /pgw/v1/sdk/enroll`). `pgw chat` today reads `PGW_API_KEY` or asks
   for it with hidden input.
3. **Model choice by device.** The app recommends the largest model the machine meets the planning figures for,
   capped at 3B (`pgw.chat.backend.recommended_model`: 0.5B under 4 cores / 8 GB, 1.5B at 4 cores / 8 GB, 3B at
   8 cores / 16 GB and enough free disk); 7B (16 cores / 32 GB) is opt-in. Speed is about the same for 1.5B and 7B
   (same trip count); the difference is disk, memory, setup time and mask cost.
4. **First-run setup, in the background, with progress.** Enrollment, then the public model files (checked against
   the pinned fingerprint; downloads resume chunk by chunk after an interruption), then the one-time mask setup on
   the user's CPU (at 7B 10-16 CPU-minutes with AMX/AVX-512 VNNI, 1.5-2 CPU-hours with AVX2 only; smaller models
   proportionally less), then the first mask segment. The UI shows each stage with percent and ETA from
   `client.status()` (`pgw.chat.status.summarize`), the stalled-provider state, and failures with a retry. The user
   can close the tab; setup continues while the app runs and resumes after a restart.
5. **Disk and CPU disclosure**, shown in the model picker and the setup card: first download, disk, memory per
   model. The weight-free mode (`factory_weights="stream"`) keeps no int8 weights (7B: about 20 GB instead of 26) at
   about 1-2 % more network, and should be the default for the consumer app. The mask factory uses a quarter of the
   cores at low priority. Mask stock writes are large at 7B (6.2 MB per token: a 300-token answer writes ~2 GB), worth
   stating for laptops with small SSDs.

### Abuse, rate limiting and moderation without reading content

* We cannot see content in option A, by design, so abuse control works on accounts and volume: per-key quotas and
  rates (existing: `daily_token_quota`, `rows_per_s`, `burst_rows`, `max_installations`, enrollments per hour),
  sign-up friction for free accounts (proof-of-work, phone or payment verification), revocation, and abuse reports
  that users submit themselves (with the content they choose to share).
* Content moderation can only run on the user's device (a small on-device classifier) and is advisory: the user
  controls the machine. Whether to ship one, and what it does (warn, refuse locally), is a policy decision for you;
  nothing in the prototype moderates.
* Accounts vs anonymous: anonymous use is compatible with the design (the key need not carry an identity), but
  free anonymous keys need friction and small quotas; paid tiers can raise them.

### Conversation storage

Local only. 0600 files are protected by the OS account and the disk's encryption. An extra encryption layer with a
key in a file next to the data adds nothing; with a key held by the OS keychain (the desktop wrapper can do this) it
protects backups and other-account reads, at the cost of keychain prompts. Recommended for the desktop app, not for
`pgw chat`. No telemetry; no conversation ever leaves the device.

### Prefix cache (decided: off by default, opt-in with an explanation)

With the masked LM head (what our provider serves) the library's prefix cache is opt-in because the provider can
tell a cache hit: a request that continues an earlier one sends only its new tokens' rows, which links a
conversation's turns and shows its length ([prefix-cache review](prefix-cache.md)). For a consumer app the marginal
leak is smaller than in a multi-user gateway (the provider already knows that one key = one user), and the benefit
is large for long chats (no masks and no upload for the earlier turns). The app keeps the library default (off with
the masked head) and offers `--prefix-cache on`; the privacy screen says which is in effect and what it reveals.
Revisit when request padding ships.

### Desktop wrapper (design)

* **Tauri** (Rust, small, the native code is Rust already) as the shell; the existing UI loads in its webview.
* The Python SDK ships as a sidecar: an embedded CPython with the wheels (torch CPU is the largest part, about
  200 MB compressed), started as `pgw chat --no-browser --port 0` with the token handed over a pipe, not argv.
* First hardening step: replace the loopback HTTP server with Tauri IPC (custom protocol), removing the local socket
  and with it the token, Host and Origin checks' whole attack surface. Until then the HTTP checks stay as they are.
* Code signing and notarisation; updates signed with our release key; the key in the OS keychain; the conversation
  store encrypted under a keychain-held key (above).
* Electron works too but bundles a second browser engine (~150 MB) for no gain here.

## Rejected and deferred alternatives

**B. Hosted chat backend (we run the SDK server-side; the browser is a thin client).** Rejected: our server would
see every prompt and answer in plaintext, which breaks the device-only plaintext principle. The GPU provider would
still see only masked data, but the user would have to trust the chat operator, which is exactly what the library
exists to avoid. It would also be the only way to interactive speed today (the client next to the GPU).

**B'. Hosted backend inside a confidential VM / TEE with remote attestation** (the browser verifies the enclave's
measurement and encrypts to a key inside it). Rejected: it replaces the library's cryptographic guarantee with trust
in the TEE vendor's hardware and keys, side-channel resistance and our build pipeline. That is a different (weaker)
promise than "nobody but you can read your chats", and marketing both under one brand would blur the guarantee we
actually prove. (An earlier draft of this work explored it; the code was removed.)

**C. In-browser client (WASM / WebGPU).** Deferred as a research item. It keeps the principle, but:
* the RTT arithmetic above applies unchanged (the client is on the user's network);
* the mask factory's CPU (11-35 ms per slot per core at 0.5B-7B, native AVX-512 code) runs several times slower in
  WASM, and the one-time setup (minutes to hours natively) would be impractical at 7B in a tab;
* the mask stock (8 GB per model by default; 1.8 MB per slot at 0.5B, 6.2 MB at 7B) and the model data (0.6 GB at
  0.5B, 4.4 GB at 7B) must live in OPFS/IndexedDB under browser quotas and eviction;
* the fast lane is raw TCP+TLS, which browsers cannot open: it would need a WebSocket or WebTransport front on the
  provider (the WebSocket stream channel exists but is slower per trip);
* the CSP must allow `connect-src` to the provider, and the page code itself would be delivered by us on every
  visit, so a compromised web server could ship code that leaks; a local app's code is installed and signed once.
At 0.5B with the weight-free stock it may be feasible as a "try it" tier; 7B is not.

## Open questions for you

1. Accounts: email/passkey accounts, anonymous keys with proof-of-work, or both? Free-tier quotas?
2. Moderation: none (privacy-first, abuse handled by quotas and revocation), an advisory on-device classifier, or
   refusal of some categories on-device?
3. Default model and modes for the consumer app: 1.5B with the weight-free stock by default? Speculation on by
   default despite the strict-mode tail? Prefix cache opt-in as now?
4. Desktop wrapper: Tauri with a bundled Python sidecar (recommended), and who signs releases?
5. Regional GPU placement targets (5 ms metro vs 20 ms national) and their cost, which set the usable speed.
6. Should `usage_reports` default off in the consumer app (the provider sees the same counts anyway, but "we send
   nothing extra" is a simpler promise)?
