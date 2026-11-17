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
