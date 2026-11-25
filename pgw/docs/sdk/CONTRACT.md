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

### Packaging (packaging agent)
- Distribution `pgw-sdk` (`pgw` on PyPI is an unrelated project), import name `pgw`, version from
  `pgw.__version__` only. Python >= 3.11 (abi3 wheels: one per platform). Base install = the client: add any
  new runtime dependency of `pgw.sdk` / `pgw.provision` to `[project].dependencies`; server-only packages go
  to extras (`gateway`, `service`, `factory`, ...). `pgw-sdk[client]` is an alias of the base install.
- Native extensions stay top-level modules `pgw_fast`, `pgw_ntt`, `pgw_pks` (setup.py, setuptools-rust);
  `PGW_REQUIRE_NATIVE=1` makes their build mandatory (CI, releases).
- CLI: `pgw doctor` reports the native extensions (check `native`). An `sdk` subcommand group of the CLI is the SDK agent's
  call; docs use `pgw doctor` / `pgw version` until it exists (docs tests resolve every documented command).
- Release smoke test `scripts/smoke_installed.py --require-sdk` imports `from pgw import OpenAI, AsyncOpenAI`
  from the installed wheel; `examples/` and `tests/packaging/test_examples.py` assume the OpenAI surface
  (`chat.completions.create`, attribute access, sync/async streaming, `stream_options`, `usage`).
- Client options documented in `docs/sdk/configuration.md`: `api_key` / `PGW_API_KEY`, `cache_dir`,
  `mask_scheme` (`"pks-v3"` default, `"mlwe"`, `"plain"`), `hint` (`"local"`); model ids
  `qwen2.5-{0.5b,1.5b,3b,7b}-instruct`. Owners: correct the page if the implementation differs.
