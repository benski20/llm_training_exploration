# Interface notes

Log of interface requests and changes between component teams. **Append only**, newest at the
bottom. The lead architect folds accepted entries into `SPEC.md` and marks them `[done]`.

Use this when you need something from a package outside your scope (a new function, a field, a
behaviour change) instead of editing that package. Do not edit other teams' files.

Entry format:

```
## YYYY-MM-DD <your team> -> <owning team>: <short title>
- What: exact signature / field / behaviour requested
- Why: the use case
- Impact: who else is affected; backwards compatible? (yes/no)
- Status: open | accepted | rejected | done
```

---

<!-- entries below -->

## 2026-11-17 modelpack -> all (engine, factory, service, gateway, cli): modelpack implemented; additive API
- What (all backwards compatible with the stub signatures):
  - `ModelPack.open(path, verify=..., expected_fingerprint=..., components=None|("linears","plain","tokenizer"), workers=None)`.
    `components` = parts that must be present and (with `verify="full"`) get hashed. Service/factory should pass
    `components=("linears",)`; a gateway without the plain/float dev backends can pass `("plain","tokenizer")`.
    `verify="fingerprint"` now also runs `Manifest.check_consistency()` (canonical linear order, shapes implied by arch, parts layout).
  - Streaming/partial loaders: `iter_linears(device, layers=None, kinds=None)`, `load_layer(i)`, `iter_layers(device, layers=None)`
    -> `(layer, {LinearKind: LinearWeights})`, `load_plain_tensors(dtype, device)` (lm_head aliases embed_tokens when tied),
    `load_tensor(name, device, check_hash=False)`, `load_linear(lid, device, check_hash=False)`, `linear_entry(lid)` (parts/rows).
  - Rings: `pack.check_ring(RingParams)` (raises RingOverflowError), `pack.supported_rings()`, `pack.max_in_features()`.
  - Distribution: `pack.export(out_dir, ["linears"])` writes a subset pack (same manifest => same fingerprint), hard-linked when possible.
  - `LinearWeights` gained read-only helpers: `out_features`, `in_features`, `spec`, `dequantize()`.
  - `pack_checkpoint(..., license=None, progress=None)`; recognizes HF-cache snapshot dirs and records `org/name` + commit.
  - `arch_config_from_hf(config, generation_config)` and `expected_linear_shape(arch, lid)` in `pgw.modelpack.arch`.
    Handles transformers-5 `rope_parameters` and classic `rope_theta`/`rope_scaling`; `ArchConfig.rope_scaling` is normalized
    to a dict with `rope_type` (None for default rope). eos ids = union of config and generation_config.
- Storage notes: `embed_tokens`/`lm_head` use `quant.plain_dtype`; norms are always float32 in `plain.safetensors`.
  `tokenizer/` also contains `generation_config.json` when the checkpoint has one.
- Why: service/factory need linears-only packs and layer streaming; gateway needs plain-only loading.
- Impact: additive only. Status: done (implemented in src/pgw/modelpack, tests in tests/modelpack).

## 2026-11-17 engine -> gateway, maskstore, wire, lead architect: engine API as implemented
- What (gateway): `Generator(model, tokenizer, *, max_batch=32, session_factory=None, batching=True)`.
  `session_factory(slots) -> MaskReservation` is called with `len(prompt) + max_tokens` and the generator
  closes (burns) it at the end; or pass `session=` per call and manage it yourself. New: `agenerate()`,
  `generate_many()`, `aclose()`. `TokenEvent` gained `top_logprobs`; `GenerationResult` gained `logprobs`.
  `SamplingParams` gained `ignore_eos: bool = False` and `include_stop_str_in_output: bool = False`
  (backwards compatible defaults); `stop` also accepts a single string. Prompts longer than the context
  raise `pgw.engine.ContextOverflowError` (code `bad_shape`).
- What (maskstore): `MaskedBackend` uses `LinearCall.session` as the reservation and calls only
  `remaining(lid)`, `rows_until_boundary(lid)`, `checks(lid)` (before `take`, for the extent the next rows
  come from), `take(lid, rows)`; optional attribute `ring` (RingParams or name) is cross-checked.
- What (wire): `MaskedBackend` uses `client.compute_sync(req)` for sync calls and `await client.compute(req)`
  for async ones (either may be absent). Transport exceptions that are not `PGWError` become
  `ServiceUnavailableError`. Responses are validated (id, item count/order, linear, rows, byte length).
- What (all): `PlainBackend` is also exported as `QuantPlainBackend`. `MaskedBackend(..., on_integrity_failure=fn)`
  hook for the gateway circuit breaker; `max_rows_per_item` (default 8192) splits items.
  `pgw.engine.LocalComputeClient` / `LocalMaskSource` / `InMemoryPack` are in-process stand-ins for dev/tests.
- Impact: additive, backwards compatible. Status: open

## 2026-11-17 factory -> cli: mount the factory command group
- What: in `src/pgw/cli/main.py` replace the stub `factory` group (`generate`, `run`) with
  `from pgw.factory.cli import factory_group; cli.add_command(factory_group, "factory")`.
  The group provides `pgw factory generate|run|status|benchmark` (options documented in `--help`),
  uses the same exit codes (0 ok, 1 PGWError/config, 3 not implemented, 130 interrupted) and installs
  SIGINT/SIGTERM handlers for `run`. Also runnable today as `python -m pgw.factory ...`.
- Why: brief requires `pgw factory run|status|benchmark`; cli is outside the factory scope.
- Impact: cli only; backwards compatible (`generate` keeps `--pack/--store/--slots/--ring/--device/--segments`).
- Status: open

## 2026-11-17 factory -> maskstore: SegmentWriter usage contract the factory relies on
- What (as implemented by `pgw.factory.MaskGenerator.generate_into_store`, tested with an in-memory fake):
  - `store.writer(fp, ring: RingParams, specs, slots=, seed=, checks=, chunk_slots=, factory_id=)`; `specs` in canonical
    (header) order from `ModelPack.linear_specs()`.
  - `write_linear(lid, wr, uw)` is called linear-major, chunks in slot order, each `wr` int64 `(rows<=chunk_slots, m)`;
    `uw` (int64 `(checks, n)`) is passed on EVERY call for that linear and is identical across calls.
    The factory zeroizes `wr` right after `write_linear` returns, so the writer must copy/encrypt synchronously.
  - Exactly one of `commit()` / `abort()` follows; `abort()` on any exception or stop request.
  - Labels: `r` = `ring_uniform_rows(n, K, slot, rows, "r", lid.key)`, `u` = `ring_uniform((checks, m), K, "u", lid.key)`.
  - Uses `store.inventory(fp, ring)` (`slots_available`), `store.add_low_water_listener(fn)` (optional), and
    `store.settings.low_water_slots/target_slots/root` (optional; defaults otherwise).
- Requests (optional, used when present via `hasattr`):
  - `MaskStore.export_segment(segment_id, dest_dir) -> Path`: move a committed segment file out of the store (and drop
    it from state) for partner export. Without it `ExportSink` moves `<staging>/**/<id>.pgwm` itself.
  - `MaskStore.cleanup_partials()`: delete stale temp files after a crash (called once on scheduler start).
  - `KeyProvider` is used as a wrap-only provider by `pgw.factory.RecipientWrapProvider` (`kek_id = "hpke:<16 hex>"`,
    `wrap()` -> `hpke_wrap_for_recipient`, `unwrap()` raises) so a partner's staging store writes HPKE-sealed segments.
- Impact: none if the store follows SPEC 7.2/7.5. Status: open

## 2026-11-17 factory -> core: exact CPU int8 ring GEMM backend (`cpu_int8`)
- What: `pgw.factory.kernels` implements `cpu_int8`: the `cuda_int8` limb algorithm on CPU via `torch._int_mm`
  (available on CPU in torch >= 2.4/oneDNN), gated by an exactness probe with extreme values
  (`cpu_int8_available()`). Measured on this 4-core box: ~1.0 TOPS int8 vs 0.11 TFLOPS f64 -> ~4x faster ring matmuls
  than `cpu_f64` at K40 (5 limb GEMMs vs 2). Suggest adopting it in `pgw.core.ringmm` (`select_backend("cpu")` -> `cpu_int8`
  when the probe passes); the service's CPU path and `freivalds_precompute` would benefit.
- Impact: additive. Status: open

## 2026-11-17 maskstore -> cli, factory, gateway, engine: maskstore implemented; CLI wiring request
- What (cli): replace the `pgw maskstore init|status|import|gc` stubs with the helpers in `pgw.maskstore.admin`:
  `init_store(root, key_file, **overrides) -> dict`, `store_status(settings|None, **overrides) -> dict`,
  `import_files(files, settings|None, delete_source=False, **overrides) -> list[dict]` (one result per file,
  `ok`/`code`/`error`), `gc_store(settings|None, **overrides) -> int` (bytes freed). New:
  `rotate_store_kek(new_key_file, settings|None, reencrypt=True) -> dict` for a `pgw maskstore rotate-key NEW_KEY_FILE`
  command (offline: gateways/factories stopped; afterwards set `PGW_MASKSTORE_KEY_FILE` to the new file).
  Settings come from `MaskStoreSettings` (`PGW_MASKSTORE_*`). Errors are `PGWError` subclasses (exit code 1).
- What (factory): the requests in "factory -> maskstore: SegmentWriter usage contract" are implemented:
  `MaskStore.export_segment(segment_id, dest_dir) -> Path` and `MaskStore.cleanup_partials(max_age_s=0.0) -> list[str]`
  (call only when no writer of that store is running). Writer contract as described there.
- What (engine/gateway): `MaskStore.reserve(fp, ring, slots, *, session_id="") -> MaskReservation` with
  `remaining/rows_until_boundary/checks/take/taken/close`, `ring`, `slots`, `extents`. `MaskStore.metrics()` returns
  values for `pgw_masks_available_slots|consumed_slots_total|burned_slots_total|low_water` keyed by `(fp, ring)`
  (maskstore does not import observability). `MaskStore.status()` is the `/admin/masks` payload (no secrets).
  `add_low_water_listener(fn(Inventory))` fires once per downward crossing of `low_water_slots`.
- New: `MaskStore.rotate_kek(new_provider, reencrypt=True)` (re-MACs state + journal; resumable; crash-safe via a
  staged `generation.journal.next`).
- Impact: additive. Status: open (cli wiring)

## 2026-11-17 service -> cli, lead architect: compute service implemented; CLI hook and additions
- What (cli): replace the stub body of `pgw service serve` with
  `from pgw.service import serve; serve()` (settings from `PGW_SERVICE_*`, logging from `PGW_LOG_LEVEL`/`PGW_LOG_JSON`;
  installs SIGTERM/SIGINT graceful drain; raises `PGWError`/pydantic `ValidationError` on bad config -> exit 1).
  Also runnable today as `python -m pgw.service`. Load generator: `python -m pgw.service.loadtest --help`.
- What (additive API, `pgw.service`): `create_app(settings, *, registry=None, authenticator=None, metrics=None)`
  (`app.state.pgw` = `ServiceState`), `serve()/serve_async(settings, shutdown_event=None)`, `create_metrics_app(state)`,
  `ServiceHandler` (the `pgw.wire.ComputeHandler`), `BatchScheduler.submit_timed(...) -> (y, BatchTiming)`,
  `ModelRegistry(device, backend, *, devices, placement, rings, verify, cache_float, warmup)` + `linear_dims(fp, ring)`,
  `place_layers(layer_bytes, devices)`, `TenantAuthenticator(tokens_file, *, allow_unauthenticated, mtls_tenants_file,
  token_hashes, mtls_subjects)`, `hash_token(token)`.
- New `ServiceSettings` fields (all optional, env `PGW_SERVICE_*`): `devices`, `placement` (single|by_layer),
  `verify_packs` (full|fingerprint), `backend` gains `cpu_int8` (exact `torch._int_mm` CPU path behind an exactness probe,
  ~4x faster than cpu_f64 at batch>=8 on this box), `metrics_port`/`metrics_host`/`metrics_on_main_port`,
  `max_items`, `max_deadline_s`, `max_queued_rows`, `cpu_workers`, `gpu_workers_per_device`, `codec_offload_bytes`,
  `tenant_max_inflight`, `tenant_rows_per_s`, `tenant_burst_rows`, `retry_after_s`, `mtls_tenants_file`,
  `trusted_client_cert_header`, `drain_timeout_s`, `cache_float_weights`, `warmup`.
- Errors: unknown fingerprint -> `fingerprint_mismatch` (409); ring not served -> `config` (per pgw.wire contract);
  draining / over `max_inflight_requests` / queue full -> `unavailable` + `Retry-After`; tenant quotas -> `rate_limited`.
- mTLS tenant identity: hypercorn 0.18 does not expose the peer certificate to the app, so with direct hypercorn mTLS
  the certificate gates the connection and the bearer token names the tenant. Certificate->tenant mapping works when
  the ASGI server provides the TLS extension or a TLS-terminating proxy sets `trusted_client_cert_header`.
- pyproject: `service` extra gained `h2` and `httpx[http2]` (pgw.wire imports httpx at package import).
- Impact: additive. Status: open (cli wiring)

## 2026-11-17 gateway -> cli, lead architect, observability, engine, packaging: gateway implemented
- What (cli): replace the stub `gateway` group in `src/pgw/cli/main.py` with
  `from pgw.gateway.cli import gateway_group; cli.add_command(gateway_group, "gateway")`. It provides
  `pgw gateway serve [--host --port]` (uvicorn, TLS/mTLS from `PGW_GATEWAY_TLS__*`) and
  `pgw gateway keys create|list|revoke` (`--keys-file` or `PGW_GATEWAY_API_KEYS_FILE`); same exit codes (0/1/3).
  Also runnable as `python -m pgw.gateway`.
- What (packaging): please add `pyyaml>=6` to the `gateway` extra (YAML `models_file`; JSON works without it).
- What (SPEC 9.1, additive): extra endpoints `GET /v1/models/{name}`, `GET /admin/keys` (records + tokens used today,
  no hashes), `POST /admin/models/{name}/reset` (body `{"breaker": bool, "noise": bool}`; clears the integrity circuit
  breaker and/or the noise-monitor alarm). `POST /admin/keys` body: `name, scopes, models, requests_per_minute,
  tokens_per_minute, daily_token_quota, max_concurrent_requests` -> 201 `{"key", "record"}` (key shown once).
  Admin auth = `PGW_GATEWAY_ADMIN_TOKEN` or an API key with scope `admin`.
- What (settings, additive, `PGW_GATEWAY_*`): `allow_dev_modes` (default false; required for `plain`/`float` models,
  `freivalds_rate < 1` and `allow_anonymous`), `require_service_tls` (http only to loopback), `egress_allowlist`
  (service URLs are always allowed; every outbound request goes through `EgressGuardTransport`), `integrity_cooldown_s`
  (None = admin reset only), `audit_log_file`, `daily_token_quota`, `quota_state_file`, `default_max_tokens`,
  `max_tokens_limit`, `max_body_bytes`, `noise_monitor_required`, `noise_sample_rate`, `drain_timeout_s`,
  `metrics_require_admin`, `max_prompts_per_request`. Per model: `fingerprint` (pin), `max_queue`, `queue_timeout_s`,
  `service_timeout_s`, `service_max_retries`, `service_deadline_ms`, `device`.
- What (observability): the gateway constructs `OutboundNoiseMonitor(ring, sample_rate=, fail_closed=)` per masked
  model and relies on `guard(x_hat, xq)` raising `pgw.core.IntegrityError` when alarmed (mapped to HTTP 502), and on
  `report()` exposing `alarm, samples, windows, duplicate_rows, reasons`, plus `reset()`. While the stub raises
  NotImplementedError the gateway logs a warning and runs without it (or refuses to start with
  `noise_monitor_required=true`). Gateway metrics are emitted on a private `CollectorRegistry` built from
  `METRIC_NAMES` (`pgw_gateway_*`, `pgw_masks_*` only; `*_created` series disabled), so `PGWMetrics` is not needed by
  the gateway. `/dashboard` uses `dashboard_html("PGW gateway")` when implemented, else a minimal built-in page.
- What (engine, request): the gateway reserves masks itself (fail fast) and hands the reservation to the engine via
  `Generator(session_factory=...)` + a ContextVar so that the engine closes it at a safe point (never during an
  in-flight batched step). A public `astream(..., session=res, own_session=True)` would make this explicit.
- Impact: additive. Status: open (cli wiring, pyyaml, SPEC fold-in)

## observability: PGWMetrics, OutboundNoiseMonitor and dashboard implemented (observability team)
- What (metrics): `PGWMetrics(registry=None, *, enabled=True, prefixes=None)`; `registry=None` uses a process-wide
  PGW registry (`pgw.observability.default_registry()`, separate from `prometheus_client.REGISTRY`). Collectors are
  created lazily and cached per registry, so several `PGWMetrics` (service, factory, gateway) on one registry share
  them; a collector already registered under a contract name by someone else is reused. Extras: `value(name, **labels)`,
  `preregister()`, `add_collect_hook(fn)`, `render() -> (bytes, content_type)`, `asgi_app()`, `start_http_server(port)`,
  `safe_label(v)`. `*_seconds` histograms use `LATENCY_BUCKETS`, `*_rows` use `ROW_BUCKETS`.
- What (noise monitor): the stub signature is kept; new keyword-only options: `alpha` (1e-9 per window),
  `always_sample_first` (16), `max_values_per_call`, `duplicate_check` (on every call, not only sampled ones),
  `duplicate_history_rows`, `duplicate_columns`, `feature_window_rows`, `metrics` + `model` (sets
  `pgw_gateway_noise_alarm`), `seed`. Also `add_alarm_listener(fn(NoiseReport))`, `flush()`, `alarmed`,
  `NoiseMonitorConfig` + `from_config`. `NoiseReport` gained additive fields (`calls, sampled_calls, rows_checked,
  last_p_top, last_p_low, last_top_bit, last_runs, alarm_since, fail_closed, sample_rate`) and `to_dict()`.
  `guard()` raises `IntegrityError` (fail-closed) and `ShapeError` for non-integer / >2-D input or an `xq` shape mismatch.
  `reset()` keeps the duplicate-row history (a reused mask is still caught after a reset).
- Suggestion (gateway, optional): pass `metrics=`/`model=` and register an alarm listener that writes an audit event
  so alarms page immediately rather than on the next `/readyz` or `/metrics` poll.
- What (dashboard): `dashboard_html(title, *, metrics_url="metrics", status_url="admin/status", refresh_seconds=5)`;
  `mount_dashboard(app, path, ...)` also sets CSP `DASHBOARD_CSP`, `no-store`, `X-Frame-Options: DENY`. HTML/JS
  live in `src/pgw/observability/static/` (already covered by `package-data`). The admin token is entered on the
  page, kept in `sessionStorage` and sent as `Authorization: Bearer` to `/admin/status` (and `/metrics` when
  `metrics_require_admin`).
- Impact: additive. Status: done.

## rust-core: `pgw/rust/pgw-core-rs` and cross-language vectors (rust-core)
- What: new cargo workspace `pgw/rust/` with crate `pgw-core-rs` (ring K32/40/48 incl. presets, balanced digits,
  int8-limb/int32 ringmm + reference, quantization, packing, mask/unmask + range check, Freivalds, HKDF+ChaCha20
  `SeedExpander` incl. seekable `ring_uniform_rows`, single-use `Ledger`/`ReservationCursor` logic). Optional pyo3
  bindings behind feature `python` (module `pgw_core_rs`), not built by tests.
- What (vectors): `tests/vectors/core_v1.json` is generated from `pgw.core` by `python rust/tools/gen_vectors.py`
  and checked by `cargo test` and `pytest tests/vectors` (the pytest fails if `pgw.core` output drifts from the file:
  that is a protocol break for every independent implementation). `PGW_TEST_RUST=1` also runs `cargo test`.
- Suggestion (lead architect): mention `tests/vectors/core_v1.json` next to the KATs in SPEC 7.3; optionally add
  `maturin` to a `rust` extra in pyproject if the bindings should be packaged. CI: `cd pgw/rust && cargo test`.
- Note (core, no change requested): Rust refuses streams that would run past the 2^32-block ChaCha20 counter
  (RFC 8439); Python only checks the start block, so a request ending past 256 GiB of one stream differs. Not
  reachable with real shapes.
- Impact: additive. Status: done.

## 2026-11-17 docs -> lead architect, cli, maskstore, factory: documentation set + findings
- What (done): `pgw/docs/` holds the whitepaper, STRIDE threat model, operator guides (on-prem, customer cloud,
  air-gapped/partner factory), sizing guide plus a stdlib-only calculator `docs/tools/pgw_sizing.py`, gateway API
  reference, compliance FAQ (draft, marked for legal review), audit checklist and glossary. Every quoted number is in
  `docs/measurements.json` with provenance. `tests/docs/` (111 tests, about 8 s) checks the docs against the code:
  CLI commands and options, `PGW_*` env names, ErrorCode/HTTP table, gateway routes, METRIC_NAMES, schema fields,
  limits and security defaults, the Freivalds table against `soundness_bits`, and runnable snippets. **If you rename
  a CLI option, setting, route, error code or metric, `pytest tests/docs` will fail until the docs are updated.**
- Findings for the lead architect (suggest folding into SPEC; docs already describe the behaviour as it is today):
  1. SPEC 1/3.2 say "information-theoretic" privacy, but masks come from HKDF+ChaCha20 seeds (SPEC 7.3). The
     whitepaper (4.4) states it precisely: perfect secrecy for ideal masks, computational at the PRG level as deployed.
  2. Freivalds soundness at K32A8 with the default `checks=2` is weak for wide layers: `c*h` = 4 bits for n = 14336
     (Llama-3-8B down) and 2 bits for n = 18944 (Qwen2.5-7B). Suggest a per-ring default (`checks >= 4` for K32A8) or
     a startup warning when `soundness_bits` < 16.
  3. Freivalds verifies the service product only; a wrong `W r` from the factory is caught by the range check only
     for gross errors (threat model T-29). Optional: a spot-check of `W r` by a party holding the weights.
  4. SPEC 3.4 names `pad_rows_to` and dummy rounds as mitigations; neither is implemented. Docs mark them "planned".
- Requests (optional): `pgw maskstore recipient-key create OUT` CLI (docs use a Python one-liner around
  `RecipientKeyPair.generate`); link `docs/README.md` from the top-level README.
- Impact: none on code. Status: open (findings 1-4 for SPEC).

## 2026-11-17 cli-deploy -> all: CLI wiring done (factory, gateway, maskstore, service requests)
- What (done, in `src/pgw/cli`): `pgw factory ...` and `pgw gateway ...` are the component groups
  (`pgw.factory.cli.factory_group`, `pgw.gateway.cli.gateway_group`), mounted **lazily** so an image
  without an extra still runs the rest of the CLI (a missing extra prints `pip install 'pgw[<extra>]'`, exit 1).
  `pgw maskstore init|status|import|gc|rotate-key` call `pgw.maskstore.admin`; `pgw service serve [--host --port --pack]`
  calls `pgw.service.serve(ServiceSettings(...))`. The "factory -> cli", "maskstore -> cli", "service -> cli" and
  "gateway -> cli" wiring requests above are done.
- New commands (additive; please fold into SPEC 12): `pgw inspect PACK` (fingerprint, rings, mask/wire bytes per token),
  `pgw doctor [--role gateway|service|factory|pack] [--pack] [--ring] [--maskstore --key-file] [--service-url
  --service-token-file --service-ca-file] [--settings] [--strict] [--json]`, `pgw service token create|hash`
  (tokens file lines `tenant:sha256`, token printed once or written 0600 with `--out`), `pgw bench ring|factory|service`,
  `pgw config show ROLE` (effective settings, secrets masked) and `pgw config render TEMPLATE OUT` (`${env:X}` /
  `${file:P}` substitution, 0600 atomic; used by the container entrypoint `PGW_RENDER=src:dst`). Exit code 130 on SIGINT.
- Impact: additive. Status: done

## 2026-11-17 cli-deploy -> gateway: `service_token_file` per model (optional)
- What: `GatewayModelConfig.service_token_file: Path | None` (read at start-up, mutually exclusive with `service_token`).
- Why: Kubernetes/systemd deliver the tenant token as a file; today deployments render the models file with
  `pgw config render` to inline it (works, but the token then sits in a second file on tmpfs).
- Impact: additive; the charts would drop the render step. Status: open

## 2026-11-17 cli-deploy -> service: health endpoints on the metrics listener
- What: serve `GET /healthz` and `/readyz` on `metrics_port` too (plain HTTP, internal), same semantics as the main port.
- Why: with `tls.require_client_cert` kubelet probes cannot present client certificates, so the `pgw-service` chart
  falls back to TCP probes and readiness cannot wait for pack loading/warm-up.
- Impact: additive. Status: open

## 2026-11-17 cli-deploy -> lead architect (packaging): pyproject and repo-level CI
- What: (1) add `pyyaml>=6` to the `gateway` extra (also requested by gateway) and `types-PyYAML` to `dev`; the deploy tests
  and the gateway image use YAML (the images currently pip-install `pyyaml>=6` explicitly). (2) GitHub only reads
  workflows from the repository root: copy `pgw/.github/workflows/ci.yml` to `.github/workflows/pgw-ci.yml` (it already
  uses `PGW_DIR=pgw`). (3) SPEC 12/13: reference `deploy/` (docker, compose, helm, systemd, vmware, marketplace).
- Impact: none at runtime. Status: open

## 2026-11-17 cli-deploy -> gateway (marketplace, later): optional AWS Marketplace metering hook
- What: an optional `PGW_GATEWAY_MARKETPLACE__*` integration calling AWS `RegisterUsage` at start-up (container listing).
- Why: AWS container listings with hourly pricing require it (see deploy/marketplace/aws/README.md).
- Impact: additive, off by default. Status: open

## 2026-11-17 cli-deploy -> docs: `pgw maskstore recipient-key create|show` implemented
- What: `pgw maskstore recipient-key create PRIVATE_KEY_FILE [--public-out PUB.hex]` (private key 0600, refuses to
  overwrite; prints `key_id`, `kek_id`, `public_key_hex`) and `pgw maskstore recipient-key show PRIVATE_KEY_FILE`.
  The hex file is accepted by `pgw factory ... --recipient-key`. Answers the docs request above; docs may replace
  the Python one-liner in docs/operations/air-gapped-factory.md section 4.1.
- Status: done

## 2026-11-17 modelpack (hardening review) -> all: stricter pack validation
- What (behaviour, no signature changes):
  - `ArchConfig` now rejects non-positive dims, `num_heads % num_kv_heads != 0`, non-finite/non-positive
    `rms_norm_eps`/`rope_theta`, negative bos/eos ids. `TensorEntry.shape` must be non-negative; `LinearPart.rows > 0`.
  - `Manifest.check_consistency()` also enforces canonical tensor names (`linears.<id>.weight|scale|bias`), biases
    exactly as SPEC 8.2 (Qwen2 `qkv` only, Llama none) and no unreferenced tensors.
  - `ModelPack.load_linear` always checks scales finite and > 0 and biases finite (ConfigError). With `check_hash=True`
    and in `verify="full"`, int8 weights containing -128 are rejected (SPEC 5: W8 in [-127,127]; the output range
    check and ring headroom rely on it).
  - Packer rejects NaN/Inf or non-float checkpoint tensors, plain tensors whose shape disagrees with config.json, and
    embeddings that overflow `plain_dtype=float16`. Sharded index entries must be plain `*.safetensors` file names.
  - `ModelPack.export` is atomic (temp sibling + rename) and refuses targets that overlap the source pack.
  - `KNOWN_ROPE_TYPES` includes `yarn` (the engine implements it); other types still pack with a warning.
- Impact: valid packs produced by `pack_checkpoint` are unaffected (same fingerprint). Status: done.

## 2026-11-17 engine (hardening review) -> gateway, lead architect: additive engine changes
- What (gateway, answers the "engine, request" item above): `Generator.generate/stream/agenerate/astream` take
  `own_session: bool = False`. With `session=res, own_session=True` the engine closes `res` (burning unused slots)
  when the request ends, at a safe point (never during an in-flight batched step), and also when request validation
  fails before generation starts. The ContextVar/`session_factory` handover keeps working. Note: `astream` is an async
  generator, so if it is never iterated its body never runs; close the reservation yourself in that case.
- What (behaviour): `MaskedBackend` now rejects a reservation whose `fingerprint` differs from the backend's
  (`fingerprint_mismatch`, SPEC 8.4) before taking rows. It validates shapes and quantizes every call before taking any
  mask row (a NaN activation in one session no longer consumes rows of others), wipes `W r` buffers on every failure
  path including cancellation, and sends no request when every call has zero rows (outputs are `(0, m)`).
- What (behaviour): the async batching loop now isolates any exception (not only `PGWError`) raised by one session
  while it isolates a mask-exhaustion failure, so other streams cannot hang.
- Impact: additive. Status: open (SPEC fold-in)

## 2026-11-17 maskstore (hardening review) -> factory, cli, docs: partner segment signatures (SPEC 7.7)
- Why: HPKE base mode does not authenticate the sender. Anyone who knows the customer's recipient *public* key
  (including the compute-service operator) could craft a segment with a seed they know, slip it into the delivery
  channel, and `import_segment` accepted it; Freivalds cannot detect this (the masks are valid), so all plaintext would
  leak. SPEC 7.7 requires the partner to be independent; the store now lets the customer enforce *which* partner.
- What (maskstore, done): `pgw.maskstore.partner` with `PartnerSigningKey` (Ed25519, key file 0600,
  `generate/public_bytes/key_id`), `sign_segment(path, key)` -> writes sidecar `<segment>.pgwm.sig`, `TrustedPartners`.
  Settings `PGW_MASKSTORE_TRUSTED_PARTNER_KEYS` (hex/base64 Ed25519 public keys) and
  `PGW_MASKSTORE_REQUIRE_PARTNER_SIGNATURE`. With trusted keys configured, HPKE segments without a valid signature
  from a trusted key are refused. Without them, import still works but logs a warning (backward compatible).
  `admin.import_files(delete_source=True)` also deletes the sidecar.
- Request (factory): `ExportSink` / `FactorySettings` should take an optional partner signing key file and call
  `pgw.maskstore.sign_segment(out_path, key)` after each exported segment (sign the final file; ship the `.sig` with it).
- Request (cli): `pgw maskstore partner-key create|show` (mirrors `recipient-key`), and `pgw maskstore import` should
  keep `<file>.sig` next to each file (no change needed if files are passed in place).
- Request (docs): air-gapped / partner guide: exchange the partner's Ed25519 public key over an authenticated channel
  and set `PGW_MASKSTORE_TRUSTED_PARTNER_KEYS`; production deployments with a partner factory should always set it.
- Also (behaviour): reservation ownership now records `hostname|pid-namespace`, and live reservations are tracked
  process-wide, so a second `MaskStore` object in the same process, or a sidecar container sharing the hostname,
  can no longer close/shred a segment another one is still reading.
- Status: maskstore part done; factory/cli/docs parts open.

## 2026-11-17 factory (hardening review) -> lead architect, maskstore, cli: factory additions
- What (additive, factory-internal; for SPEC 7.4):
  - Generator self-check: `MaskGenerator(..., self_check_rows=8)` / `PGW_FACTORY_SELF_CHECK_ROWS` (-1 = all rows,
    0 = off). Before a chunk is written, sampled rows are verified with `u . (W8 r) == (u W8) . r (mod 2^K)` using
    the segment's check vectors (two private vectors when `checks=0`); a mismatch raises `IntegrityError` and the
    segment is aborted. Reason: a wrong `W8 r` is invisible to the gateway's Freivalds check on the service and
    would silently corrupt outputs. Cost ~1% (8 rows/chunk), ~10% for all rows on CPU.
  - `cpu_int8_available()` probe now covers 1/2/5/33-row shapes and both operand layouts (the `u W8` path too).
  - Recipient keys for partner export are validated (`validate_recipient_public_key`): small-order X25519 points
    (e.g. all-zero, which yields an all-zero shared secret) are rejected with `ConfigError`.
  - `ExportSink.recover()` (called by the scheduler at every start, under the factory lock) cleans the private
    staging store's partial files and publishes segments committed in staging but not yet moved to the outbox.
  - `Throttle.max_slots_per_hour` no longer accumulates credit while idle (no burst after a full-inventory pause).
- maskstore: the `export_segment`/`cleanup_partials` requests above are implemented and now covered by
  `tests/factory/test_export_e2e.py` (real MaskStore + real HPKE + customer `import_segment`). Status: done.
- Impact: additive, backwards compatible. Status: open (fold into SPEC 7.4)

## gateway hardening review (2026-11-17)
- Resolved: the gateway now hands its mask reservation to the engine with
  `Generator.agenerate/astream(..., session=res, own_session=True)` instead of `session_factory` +
  a ContextVar. The ContextVar version raised `ValueError` ("Token was created in a different Context")
  when a stream was closed from a task other than its consumer (shutdown, reload). The engine's
  `own_session` contract is now load-bearing for the gateway: owned sessions must stay closed on
  validation errors and burned at a safe point.
- New settings constraint: `PGW_GATEWAY_ADMIN_TOKEN` must have at least 16 characters (fails at load).
- `require_service_tls=false` remains allowed in production (service-mesh mTLS), but now logs a warning for
  plain-http non-loopback service URLs. SPEC 6.7/9.5 may want to say this explicitly.
- The integrity breaker is now also checked before every compute round trip, so sessions already
  generating stop sending once it trips (SPEC 9.5 "fail closed"), not only new requests.

### rust-core hardening review (pgw/rust)
- Rust is now stricter than `pgw.core` in three places (see `pgw/rust/README.md`, "Deliberate differences"):
  `mask_activations` rejects `|xq| > qmax`; `quantize_weight_rows` rejects non-finite weights;
  `freivalds_check/verify` refuse `c == 0` (Python accepts every row when there are no check vectors).
  Suggest the core owner adopt the same checks in `pgw.core.masking` / `pgw.core.ring` / `pgw.core.freivalds`
  (esp. `c == 0`, which would silently disable integrity checking if a segment header ever had `checks: 0`;
  the mask store should also reject `checks < 1` at import).

## docs hardening review -> maskstore / cli owners: rollback witness defaults

- `MaskStoreSettings.counter` defaults to `"none"`, and with no counter configured the store logs at
  INFO ("HMAC journal only"). The journal (by default `<root>/generation.journal`, on the mask volume)
  does not detect a revert of a whole-volume snapshot, because the database and the journal come back
  together. That revert means silent mask reuse. Suggestions: (1) log a WARNING, and have
  `pgw doctor` / `pgw maskstore status` report a production warning, when `counter == "none"`;
  (2) with a provider that cannot MAC (e.g. `kms` without `kms_mac_key`) and no counter, there is no
  rollback witness at all; refuse to open unless an explicit `allow_no_rollback_witness` dev flag is set;
  (3) consider defaulting `journal_file` / `counter_file` to a path the operator must set off the
  mask volume. Docs (whitepaper 3.4, threat model T-22, both install guides, audit M-06) now state this.
