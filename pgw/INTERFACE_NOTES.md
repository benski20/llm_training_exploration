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
