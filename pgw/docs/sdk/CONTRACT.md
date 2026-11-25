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

### A3. Provisioning (owner: provisioning agent; `src/pgw/provision/`, `src/pgw/service/provision.py`)

**Client API** (stable names; extra keyword arguments may be added with defaults):

```python
from pgw.provision.client import provision, ProvisionedModel
from pgw.provision.session import ProvisionSession

session = ProvisionSession(base_url="https://provider:8443", token="<service token>",   # from the SDK bootstrap
                           ca_file=..., client_cert_file=..., client_key_file=...)      # all optional but base_url
pm = provision("qwen2.5-7b-instruct",            # model name (resolved via GET .../models) or a 64-hex fingerprint
               session=session, cache_dir=Path("~/.cache/pgw"),
               progress=None,                    # callable(dict) -> None, see "progress events"
               hint="provider",                  # "provider" (provider-built, client-verified) | "local" | "none"
               expected_fingerprint=None,        # pin; default: the session's catalog entry (see trust note)
               masked_head=True,                 # hint/store keyed by head_fingerprint(fp), head linear included
               source="provider")                # artifacts from the provider | "hf" (download HF + pack locally)
```

`ProvisionedModel` (frozen dataclass):

| field | type | meaning |
|---|---|---|
| `name` | `str` | served model name (manifest) |
| `fingerprint` | `str` | base pack fingerprint (manifest; what `models.json` pins) |
| `factory_fingerprint` | `str` | `head_fingerprint(fingerprint)` with `masked_head`, else `fingerprint`: the key of the hint store and of mask segments |
| `pack_dir` | `Path` | weight-free gateway pack: `manifest.json`, the plain safetensors, `linear-meta.safetensors` (every linear's scale/bias) and `tokenizer/`; open with `pm.open_pack()` |
| `tokenizer_dir` | `Path` | `pack_dir / "tokenizer"` |
| `manifest` | `pgw.modelpack.Manifest` | verified against `fingerprint` |
| `hint_root` | `Path \| None` | `PKSHintStore` root (`PKSHintStore.open(hint_root, factory_fingerprint, mac_key=...)`) |
| `hint_store` | `PKSHintStore \| None` | opened store (holds the shared store lock; `close()` releases it) |
| `hint_key_file` | `Path \| None` | customer key that MACs the store's `meta.json` (`<cache_dir>/keys/pks-hint.key`, 0600) |
| `hint_source` | `"provider-verified" \| "local" \| None` | how the hint was obtained |
| `masked_head` | `bool` | |
| `sizes` | `dict[str, int]` | bytes: `artifacts`, `hint`, `downloaded`, `weights_streamed`, `h_streamed` |
| `timings` | `dict[str, float]` | seconds per stage (`artifacts`, `hint`) |

`pm.open_pack()` returns a `pgw.provision.pack.GatewayPack`: a `ModelPack` (same `fingerprint`, `arch`,
`linear_specs()`, `load_plain`, `load_linear_meta`, `load_tensor`, `tokenizer_dir`, `manifest`) whose `load_linear`
raises `ConfigError` (the gateway holds no int8 weights). Wrap it in `MaskedHeadPack(pack, float_head=False)` for the
masked head. The in-process factory needs `W8` only for its private check vectors (`v W8`, once per process):
`pgw.provision.client.stream_weights(pm, session)` yields fingerprint-checked `LinearWeights` one at a time.

Cache layout (under `cache_dir`, created 0700): `models/<fp[:16]>/pack/`, `models/<fp[:16]>/hints/`,
`models/<fp[:16]>/.lock` (`flock`: concurrent `provision` calls for one model are serialised), partial downloads in
`models/<fp[:16]>/partial/` (resumed chunk by chunk). Every directory is built under a temporary name and renamed.

Progress events: `{"stage": "catalog"|"artifacts"|"weights"|"hint"|"done", "done": int, "total": int,
"unit": "bytes"|"linears", "item": str}`.

**Provider endpoints** (mounted by `pgw.service.app.create_app`; same authentication as `/pgw/v1/compute`:
bearer service token and/or verified client certificate, `PGW-Protocol: 1`; errors are the wire JSON error body).
`<fp>` is the base pack fingerprint.

| method, path | request | response |
|---|---|---|
| `GET /pgw/v1/provision/models` | - | `{"models": [{"name", "fingerprint", "masked_head": bool, "hint_build": bool, "device": "cuda"\|"cpu"}]}` |
| `GET /pgw/v1/provision/<fp>/manifest` | - | provisioning manifest (below) |
| `GET /pgw/v1/provision/<fp>/object?id=<id>&chunk=<i>` | - | raw bytes of chunk `i` of object `id` (`application/octet-stream`, header `PGW-Chunk-Sha256`) |
| `POST /pgw/v1/provision/<fp>/hint` | `{"a_seed": <64 hex>, "masked_head": bool, "linears": [keys] \| null}` | `{"job": str, "scheme": "pks-v3", "form": "h56", "rows_per_chunk": int, "device": str, "linears": [{"key", "m", "n", "chunks"}]}` |
| `GET /pgw/v1/provision/hint/<job>/h?linear=<key>&chunk=<i>` | - | rows `[i R, min(m, (i+1) R))` of `H = W8 A(a_seed) mod 2^56`, 7 bytes little endian per entry, row major (`rows x 4096 x 7`) |
| `DELETE /pgw/v1/provision/hint/<job>` | - | `{"ok": true}` |

Provisioning manifest: `{"format": "pgw-provision", "version": 1, "fingerprint", "name", "chunk_bytes",
"objects": [{"id", "kind": "manifest"|"tensor"|"tokenizer"|"weight", "name", "dtype", "shape", "size", "sha256",
"chunks": [sha256 hex per chunk]}]}`. `manifest.json` is checked against the pinned fingerprint first; every other
object's `sha256` must equal the value the fingerprint-bound pack manifest gives (tensor bytes / tokenizer file), so
the provider's own hashes are only used to reject a bad chunk early and to resume. `weight/<linear>` objects (int8
`W8`) are streamed for hint verification / local build / factory check vectors and never cached.

Hint: the client draws `a_seed` and `ks_seed` (CSPRNG). Only `a_seed` is sent (the provider sees it on every
maskgen request anyway); `ks_seed` never leaves the client. The provider computes the expensive `H = W8 A` (GPU when
available); the client verifies every linear's `H` against the fingerprint-checked `W8` with a 2-adic Freivalds
check (soundness error `<= 2^-41`: 42 binary challenge columns on each side), then computes the cheap deterministic
rest locally (`switch_hint`, key switch) and writes the same `PKSHintStore` a local build writes (bit-identical
hint files for the same seeds). The final key-switched hint itself is not checkable cheaply (two exact roundings
between `W8` and it); see `research/reviews/pks-v3-provider-hint.md`.

Trust note: `expected_fingerprint` is the integrity anchor. Without it the catalog's fingerprint (provider-asserted,
like the enrolled `models.json`) is used; `source="hf"` re-derives the pack from the public checkpoint and checks that
it yields the same fingerprint.
