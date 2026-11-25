# PGW: Private Gateway

PGW runs private LLM inference on **public open-weight models** using **one-time masks**. Your
gateway keeps prompts, activations and outputs in plaintext on your side. Our GPU compute service
does the heavy matrix multiplications on data that, to it, is uniformly random noise. Results are
bit-exact, and every result is integrity-checked (Freivalds).

```
apps ── OpenAI API ──▶ gateway (your side) ──x+r──▶ compute service (ours) ──W(x+r)──▶ gateway
                         ▲ masks (r, W r)                                  subtract W r, continue
                       mask store ◀── mask factory (you, or an independent partner)
```

See **[SPEC.md](SPEC.md)** for the architecture, threat model, wire protocol, formats and targets,
**[STATUS.md](STATUS.md)** for the release status, test and GPU results and pilot blockers,
and **[docs/README.md](docs/README.md)** for the whitepaper, threat model, operator guides and sizing.

## Install

Client library (the gateway runs in the application's process; see [docs/sdk/quickstart.md](docs/sdk/quickstart.md)):

```bash
pip install pgw-sdk --extra-index-url https://download.pytorch.org/whl/cpu   # Linux: CPU build of torch
```
```python
from pgw import OpenAI                  # drop-in for openai.OpenAI (also AsyncOpenAI)
```

The distribution is `pgw-sdk` (`pgw` on PyPI is an unrelated project); the import name is `pgw`. Wheels carry
the native extensions (`pgw_fast`, `pgw_ntt`, `pgw_pks`; abi3, CPython 3.11-3.13) for Linux x86_64/aarch64
(manylinux_2_28) and macOS arm64; elsewhere a source install falls back to pure Python (same results, slower).

From a checkout (servers and development):

```bash
pip install -e ".[dev]"          # everything + test tools
pip install -e ".[gateway]"      # customer gateway as a standalone server
pip install -e ".[service]"      # compute service
pip install -e ".[factory]"      # mask factory
make release-check               # build sdist + wheel, install in a fresh venv, smoke test
```
Python 3.11+ and torch 2.4+ are required. On GPU hosts, install the CUDA build of torch. An editable install
builds the native extensions too when a Rust toolchain (1.89+) is present (`rust/tools/build_*.py` rebuild
them in place).

## Quick check

```bash
pgw doctor          # verifies exact ring arithmetic on every available backend
pytest              # unit tests (CPU only, a few minutes)
```

## End-to-end demo

```bash
scripts/e2e_demo.sh                      # pack -> factory -> service -> gateway on localhost, then chat
PGW_E2E_MODEL=tiny scripts/e2e_demo.sh   # same with a tiny random model (no download)
pytest tests/e2e                         # the same pipeline as an automated test (about 4 min)
```

`deploy/compose` runs the same pipeline with Docker (`docker compose up --build`).

## Workflow (CLI)

```bash
pgw pack Qwen/Qwen2.5-0.5B-Instruct packs/qwen0.5b                 # int8 pack + fingerprint
pgw maskstore init --root /var/lib/pgw/masks --key-file /etc/pgw/kek
pgw factory generate --pack packs/qwen0.5b --store /var/lib/pgw/masks --key-file /etc/pgw/kek --slots 4096
PGW_SERVICE_PACKS='["packs/qwen0.5b"]' pgw service serve           # our side
PGW_GATEWAY_MODELS_FILE=models.json pgw gateway serve              # your side
curl localhost:8080/v1/chat/completions -H "Authorization: Bearer $KEY" -d '{...}'
```

## Layout

| path | contents |
|---|---|
| `src/pgw/core` | ring arithmetic, exact ring matmul (CPU/GPU, the one shared int8 digit GEMM), masks, Freivalds, CSPRNG, errors |
| `src/pgw/modelpack` | HF checkpoint → int8 model pack + manifest/fingerprint |
| `src/pgw/engine` | plaintext transformer runtime with pluggable linear backends |
| `src/pgw/factory` | offline mask generation |
| `src/pgw/maskstore` | encrypted single-use mask store |
| `src/pgw/wire` | HTTP/2 + msgpack protocol |
| `src/pgw/service` | stateless masked compute service |
| `src/pgw/gateway` | OpenAI-compatible gateway |
| `src/pgw/observability` | metrics, outbound-noise monitor, dashboard |
| `src/pgw/cli` | `pgw` command |
| `rust/` | `pgw-core-rs`: Rust port of the ring/PRNG/Freivalds core, cross-checked against the Python test vectors (`cargo test`) |
| `tests/` | per-subpackage tests; shared fixtures in `tests/conftest.py` |
| `INTERFACE_NOTES.md` | log of requested interface changes between teams |

## Status

v0.1: every subpackage (core, modelpack, engine, factory, maskstore, wire, service, gateway,
observability, cli) is implemented and covered by CPU tests, including an end-to-end test that
runs pack -> factory -> service -> gateway as real processes (`tests/e2e`). GPU paths
(`cuda_int8`) are validated on Kaggle T4 (`benchmarks/`) and skipped in CPU-only test runs.
Not in v0.1: tools/function calling, structured output, `n > 1`, multimodal inputs (rejected
with 400, see `docs/api/gateway-api.md`).

Encrypted mask factory: masks from the provider are self-checked with 40 bits against errors the provider
chooses, written in the p31 check format by default, and the factory halts after 3 provider integrity failures
(`pgw factory clear-halt` resumes it). The compact-hint scheme **pks-v3** is the default preset of the CPU-only
co-located demo and of the customer-cloud template's CPU-only presets; it is **pending external review
(lattice-estimator run and cryptographer sign-off)**, and the library still refuses it without its explicit
acknowledgement (`PGW_FACTORY_HE_PKS_ACK`). See SPEC 7.9 and 16 and `research/reviews/pks-v3-adversarial-review.md`.

## Measured basis (Kaggle T4 prototype)

| | value |
|---|---|
| Qwen2.5-0.5B masked vs quantized plaintext | identical tokens, max logit diff 0.0 |
| Llama-3-8B server time, batch 1 / amortized batch 32 | 0.45 s / 0.014 s per token |
| mask generation, 8B, T4, K=48 | 0.0066 s per token |
| mask storage, 8B | 8.3 MB/token (K48), 6.9 MB/token (K40) |
