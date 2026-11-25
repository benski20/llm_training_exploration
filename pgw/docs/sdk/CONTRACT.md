# PGW client library: build contract (internal, for the parallel build)

Goal: a client installs one Python package and changes one import:

```python
from pgw import OpenAI          # drop-in for openai.OpenAI (also AsyncOpenAI)
client = OpenAI(api_key="pgw_...")        # nothing else required
client.chat.completions.create(model="qwen2.5-7b-instruct", messages=[...], stream=True)
```

The gateway runs **in the client's process** (CPU only, no GPU, no model weights for the linears, no
large storage to manage). The provider (our GPU service) only ever sees masked data. Models: Qwen2.5
Instruct 0.5B / 1.5B / 3B / 7B for this release. Nothing here may weaken privacy or integrity; outputs stay
bit-identical to the pinned reference; every new mode defaults off unless proven exact and safe.

## Pieces and owners

| Piece | Location | Owner |
|---|---|---|
| Public API, in-process engine, background mask workers, cache dir | `src/pgw/sdk/` | SDK agent |
| Provisioning: non-secret model artifacts + pks-v3 hint (provider-built, client-verified) | `src/pgw/provision/` (client) and `src/pgw/service/provision.py` (provider) | Provisioning agent |
| Provider accounts: SDK keys, revocation, usage metering, rate limits; container image; RunPod tooling | `src/pgw/service/accounts.py`, `deploy/docker/`, `deploy/runpod/` | Provider agent |
| Developer console (keys, usage, request log), restyle of the existing console | `src/pgw/platform/`, `src/pgw/console/static/` | Console agent |
| Wheels (native extensions prebuilt), docs, examples, client hand-off kit | `pyproject.toml`, `.github/`, `docs/sdk/`, `examples/` | Packaging agent |

## Interfaces (keep these names; payload details are the owner's, documented in this file's appendix)

1. **SDK key.** `pgw_<env>_<base32>`: encodes the provider URL, the provider CA SHA-256 (pinned, as in an
   enrollment code) and a reusable, revocable credential. Parsed by `pgw.sdk.keys.parse_key` (SDK agent);
   issued and checked by the provider (`pgw.service.accounts`, provider agent). `PGW_API_KEY` env var works.
2. **Session bootstrap.** On first use the SDK reuses the enrollment machinery (`pgw.enroll`): CA pinned
   from the key, client key generated locally, `POST /pgw/v1/sdk/enroll` authenticated by the SDK key,
   returns what `pgw.enroll` writes today (service token, client cert, models.json). Cached under the
   cache dir (0600). Provider agent implements the endpoint; SDK agent consumes it.
3. **Provisioning.** `pgw.provision.client.provision(model, *, session, cache_dir, progress) ->
   ProvisionedModel` returns everything the in-process gateway needs that is not secret (tokenizer, the
   gateway's plain tensors, pack metadata + fingerprint) and a ready pks-v3 hint store. Every artifact is
   checked against the pack fingerprint; the hint is built by the provider from **client-chosen public
   seeds** and accepted only after a client-side verification with >= 40-bit soundness, or built locally
   (fallback / `hint="local"`). Provider endpoints under `/pgw/v1/provision/...`.
4. **Masks.** In-process background factory (pks-v3 by default for the SDK, with the library's explicit
   acknowledgement, documented as pending external review; `mask_scheme="mlwe"` and `"plain"` selectable),
   incremental reservations, conversation prefix reuse, p31 checks, native decode driver, masked head.

## Rules for every agent
- Never print or commit secrets. No Modal. No RunPod calls (no key yet). Kaggle is allowed for GPU
  validation only (the `kaggle` CLI is preconfigured).
- Tests: `pytest` for what you touch must pass; add tests for everything new.
- Commit messages end with the session's attribution lines; no model identifiers in commits.

## Appendix: payloads (owners fill in)

### Console agent: the developer platform and what it reads

The developer platform (`src/pgw/platform/`, served at `/platform/`) reads everything through
`pgw.platform.client.AccountsClient`, a synchronous `Protocol` run on a worker thread: `authenticate(token)`,
`principal_for_sso(sub=, email=, role=)`, `org`, `list_keys`, `create_key`, `revoke_key`, `usage(since=, until=)`,
`requests(limit=, before=, key_id=, model=, status=)`, `models`, and optionally `audit(limit=)`. Every call is
scoped to the organisation of the signed-in session; no organisation id is taken from a request.
`pgw.platform.fixtures.FixtureAccounts` implements it with sample data (tests, screenshots,
`python -m pgw.platform --demo`).

`HttpAccountsClient` speaks the provider's **admin API (A4)**:

* Sign-in with an organisation admin token (`pgw service admin-token create --org ORG`): `GET /orgs` with that
  token must list exactly one, enabled organisation. Every later call uses the platform's own global admin token
  (`PGW_PLATFORM_ACCOUNTS_URL`, `PGW_PLATFORM_ACCOUNTS_TOKEN_FILE`). SSO users (`PGW_PLATFORM_OIDC_*`, redirect
  `.../platform/sso/callback`) belong to `PGW_PLATFORM_OIDC_ORG`.
* keys: `GET/POST /orgs/{org}/keys` (`{"key": "pgw_...", "record": key}`, the key shown once),
  `POST /orgs/{org}/keys/{key_id}/revoke`; usage: `GET /orgs/{org}/usage?since&until` (tokens and trips per key,
  model and UTC day); audit: `GET /orgs/{org}/audit`.
* Not in A4 yet; the platform degrades gracefully (a dash, "Request log not available"):
  1. `GET /orgs/{org}/models` -> `{"models": [{"id", "name", "status": "available|degraded|unavailable|preview",
     "params", "context_length", "description", "ttft_ms", "tokens_per_s"}]}` (404 -> empty Models page).
  2. `GET /orgs/{org}/requests?limit&before&key_id&model&status` -> `{"requests": [{"id", "ts", "key_id", "model",
     "status": "ok|error", "verified": bool, "private": bool, "latency_ms", "ttft_ms", "prompt_tokens",
     "completion_tokens", "stream", "error_code"}], "next_before"}`, newest first: **metadata only, never content**
     (the platform parses by allow-list and drops any other field).
  3. Usage rows may add `requests`, `prompt_tokens`, `completion_tokens`, `verified`, `errors` (the client library
     verifies results, so it would report these, e.g. with its usage flush). Without them the platform shows
     tokens and compute round trips and says requests and verification are counted in the client.

Mounting on the provider service (provider agent's call, one line in `pgw.service.app.create_app`):

```python
from pgw.platform import PlatformSettings, mount_platform
if PlatformSettings().enabled:                     # PGW_PLATFORM_ENABLED=1
    mount_platform(app, accounts_client)           # an AccountsClient (in process, or HttpAccountsClient)
```
