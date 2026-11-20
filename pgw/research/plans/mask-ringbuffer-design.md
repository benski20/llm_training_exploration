# In-memory mask ring: design (WP8, plan item L5)

Status: **implemented** (v1 and v2), 2026-11-20, with every design-review fix applied; see §15 for the
review resolution and what changed against the text below (§15 wins where they differ). Code:
`src/pgw/maskstore/ringbuf.py`, `tiered.py`, `ringshm.py`, `src/pgw/factory/{sinks,scheduler,embedded}.py`,
`src/pgw/gateway/{config,runtime,masks}.py`. Scope: hand masks from a co-located factory to the gateway through
locked RAM. This skips the disk, the at-rest AES-GCM layer and the SQLite cursor transaction for
those masks. The disk store stays as a fallback and overflow tier.

Parent plan: `mask-speed-plan.md` §2a L5, §2c, §4 "L5", §5 WP8. Labels: **[m]** measured,
**[d]** derived, **[e]** estimate.

## 0. Summary

* **What.** A `MaskRing` per `(fingerprint, ring)` inside the gateway process. It is one
  `mmap`ed anonymous arena, `mlock`ed and marked `MADV_DONTDUMP` and `MADV_DONTFORK`. The arena is
  split into fixed-size **slabs**, and each slab holds one RAM segment: seed, `uW` and `W r` rows,
  laid out like the plaintext of a `.pgwm` file. The factory fills a slab and publishes it. The
  gateway reserves slot ranges from published slabs, under one lock and with cursors that only move
  forward. A slab is wiped and reused only after every slot in it has been reserved and every
  reservation on it is closed.
* **Reuse, not reimplementation.** The gateway's take path does not change. A reservation from
  the ring is an ordinary `pgw.maskstore.MaskReservation`. Its extents point at a `RamSegment`, a
  duck-typed `SegmentFile` whose `decrypt_into(chunk)` copies plaintext out of the slab instead of
  reading and decrypting a file. In-order hand-out, extent splitting, prefetch, Freivalds vectors,
  zeroization on close and burn accounting all come from existing, tested code. This works today:
  a scratch probe built a `MaskReservation` over a RAM segment. It handed out rows and check
  vectors, split at extent ends and reported burned rows, with no change to `store.py` [m].
* **Process model.** v1, which this work package implements in 1-2 days, is **same process**. The
  factory scheduler and generator run as threads inside the gateway (`embed_factory`), and the ring
  is private memory. v2, a follow-up of 2-3 days, uses **shared memory** with the gateway's
  supervised factory child process: a `memfd` passed over a Unix socket, with the same slab state
  machine and slab ownership handed over by messages. LAN transport is out of scope.
* **Tiers.** `TieredMaskSource` wraps the `MaskStore` and the rings and has the store's interface:
  `reserve`, `inventory`, `metrics`, `status`, and so on. `reserve()` uses the ring first and falls
  back to the store. The factory scheduler fills the ring first and sends overflow to the store.
  The store can be absent (`ring_only`). In that case no external monotonic counter is needed,
  because nothing is persisted.
* **Crash.** Every ring byte lives in one process's anonymous memory. A crash or kill destroys it,
  and the restarted gateway starts with an empty ring. Masks are lost, never reused. There is no
  state to restore, so rollback is impossible by construction.
* **Gain.** Factory write stage goes from about 20 ms/slot to a memcpy of about 0.5-1 ms/slot [e].
  The gateway skips AES-GCM and the disk read (about 3.7 ms of CPU per 7B slot [d] from the 1.52
  GB/s AES figure in the plan) and the per-reservation `BEGIN IMMEDIATE` plus fsync. The delivered
  rate becomes the GPU rate instead of the disk rate, so 1 GB/s of disk no longer caps supply at
  650k slots/h.

## 1. What exists today (read from the code)

* **Disk store** (`src/pgw/maskstore/store.py`):
  * `reserve(fp, ring, n)` advances per-segment `next_slot` cursors and the global generation in
    one `BEGIN IMMEDIATE` transaction with fsync, under an in-process lock plus `flock`. Only after
    that does it open segment files. It then builds `MaskReservation(store, rid, session, fp, ring,
    [(SlotExtent, SegmentFile)])`.
  * `MaskReservation` hands out rows per linear strictly in order. It derives `r` from the seed
    (HKDF stream key cached per extent and linear, then `pgw_fast.mask_rows_with_key`) and reads
    `W r` by decrypting one chunk per linear into a cache (`_decrypt_buf` → `seg.decrypt_into`).
    It zeroes its caches and burns unused rows on `close()`, which calls
    `store._on_reservation_closed(res, burned)`.
  * Rollback protection (SPEC 7.6) uses an HMAC journal plus an external counter. Without a
    counter, `_require_external_counter` makes `reserve()` refuse to run.
* **Factory**:
  * `MaskGenerator.generate_into_store(sink, slots, ...)` draws a fresh seed with `new_seed()`
    (`os.urandom`). It calls `sink.writer(fp, ring, specs, slots=, seed=, checks=, chunk_slots=,
    factory_id=)`, then `write_linear(lid, wr, uw)` (linear-major, row chunks of any size), then
    `commit()`, or `abort()` on any error.
  * `FactoryScheduler` plans segments from `sink.available_slots` with low and target hysteresis
    and a ramp. It is woken by `add_low_water_listener` and polls every `poll_interval_s`.
    `_fit_to_disk` caps plans to the free disk space.
* **Gateway**:
  * `InferenceRuntime` opens one `MaskStore` and shares it with every `ModelRuntime`.
  * `_reserve_for` reserves the prompt plus `mask_window_tokens` (64), then wraps the result in
    `WindowedReservation`, whose `extend` calls `mask_store.reserve` again.
  * The engine's `_take` calls `take_checked`, and `_maybe_prefetch` calls `prefetch`.
  * `_available_slots` (the fail-fast check) uses `mask_store.inventory`.
  * The gateway can already run the factory as a supervised child (`factory_env_file` →
    `FactorySupervisor`). The gateway itself is a single process with no uvicorn workers.

The ring therefore has to provide two things:

1. a `writer()` with the `SegmentWriter` interface, for the generator and the scheduler;
2. a `reserve()` returning a `MaskReservation`, for the gateway.

It needs no change to the generator, `store.py`, `segment_io.py` or the engine.

## 2. Process model

| | v1: same process (this WP) | v2: shared memory (WP8b) |
|---|---|---|
| where the factory runs | threads in the gateway process (`EmbeddedFactory`) | the gateway's supervised child, `pgw factory run --ring-socket` |
| arena | private anonymous `mmap`, locked in the gateway | `memfd_create`, created and locked by the gateway, `mmap(MAP_SHARED)` by both |
| slab ownership | `threading.Lock` plus state field | messages over a Unix socket (`GRANT` / `PUBLISH` / `ABORT`); each slab has one owner at a time |
| crash of the factory | takes the gateway down (CUDA fault, segfault) | the gateway sees EOF, then zeroes and frees the slabs it had granted; published slabs keep serving |
| crash of the gateway | everything is gone | the factory sees EOF, then zeroes its view, unmaps and reconnects to the next incarnation (new memfd) |
| CPU isolation | factory threads share the gateway's cores and GIL | separate cgroup or CPU limit (SPEC 13: the factory has a hard CPU limit) |
| code | about 900 lines including tests | about +500 lines (protocol, fd passing, reconnect) |

**Decision.** Build v1 now. It proves the state machine, the invariants and the gain with the
least moving parts. Design v2 as the same state machine with a different transport (§9), so
nothing in v1 has to be thrown away.

v1 is acceptable in production when two conditions hold:

* the gateway host has the factory GPU;
* the gateway gets about one extra core.

After WP1 (native `r`) and WP2 (GPU ChaCha), the factory's Python work per slot is small:

* with a ring sink, the writer stage is one numpy copy that releases the GIL;
* the remaining CPU work is the HKDF and the self-check.

Acceptance check: gateway TPOT with the embedded factory running stays within 5% of TPOT without
it (§11). If it does not, v2 becomes a prerequisite for enabling the ring in production.

Why not shared memory first:

* Python has no portable atomics on shared memory, so v2 has to pass ownership by messages
  (§9). That is more code and more failure modes.
* v1 also makes the crash argument trivial: one process, one address space.

## 3. Memory layout

### 3.1 Arena (`LockedArena`, `src/pgw/maskstore/ringbuf.py`)

```
mm = mmap.mmap(-1, size, flags=MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE)
addr = ctypes.addressof(ctypes.c_char.from_buffer(mm))      # export released right after
libc.mlock(addr, size)              # != 0 → ConfigError (fail closed) unless allow_unlocked
mm.madvise(MADV_DONTDUMP)           # not in core dumps
mm.madvise(MADV_DONTFORK)           # children (FactorySupervisor, subprocess) never map it
```

* Verified on this host (Python 3.13, Linux 6.18): `/proc/self/smaps` reports `Locked == Size` and
  `VmFlags: ... lo dc ... dd` for the arena. `explicit_bzero` can be called through `ctypes`, which
  releases the GIL [m].
* `RLIMIT_MEMLOCK` is checked at start-up. If the soft limit is below `size`, raise it to the hard
  limit. If mlock still fails (`ENOMEM`/`EPERM`), raise a `ConfigError` naming the fix:
  * Helm: `securityContext.capabilities.add: [IPC_LOCK]`;
  * systemd: `LimitMEMLOCK=`;
  * docker: `--ulimit memlock=-1` or `--cap-add IPC_LOCK`.

  `allow_unlocked=true` (dev only) logs a warning and sets `pgw_masks_ring_locked_bytes = 0`. This
  host's default is 8 MiB, which is enough for the unit tests.
* Locked memory counts against the cgroup memory limit. `MAP_POPULATE` plus `mlock` commit every
  page at start-up, so an undersized pod fails at start, not mid-request. The memory limit must be
  at least base gateway memory plus `capacity_bytes`.
* Wiping uses `explicit_bzero` (fallback `ctypes.memset`) through `ctypes`. It runs in chunks of
  64 MiB on a `pgw-maskring-wiper` thread, so no wipe holds the GIL or sits on the reserve or take
  path.

### 3.2 Slabs and RAM segments

* One `MaskRing` per masked model `(fingerprint, ring)`. Its geometry comes from the model's linear
  specs (`ModelPack` linears, the same `specs` the generator uses):

  ```
  bpe          = ring.bytes_per_element                 # 4 at K32
  slot_bytes   = Σ m · bpe                              # 5.56 MB for Qwen2.5-7B at K32
  secrets(c)   = 32 + c · Σ n · bpe                     # seed + uW; c = checks (≤ 8)
  slab_bytes   = page_round(secrets(8) + segment_slots · slot_bytes)
  n_slabs      = capacity_bytes // slab_bytes           # must be ≥ 4, else ConfigError
  ```

* `segment_slots` defaults to `clamp((capacity_bytes / 8 - secrets(8)) / slot_bytes, 16, 256)`,
  which gives about 8 slabs. Example (7B, K32, 8 GiB): `secrets(8)` = 26.6 MB, 188 slots and
  1.07 GB per slab, 8 slabs, 1,504 slots in total, about 15 s at 100 slots/s [d].
* Smaller segments make supply more just-in-time. Larger segments make the GEMMs more efficient,
  because the generator is linear-major per segment and batches up to `chunk_slots` rows per GEMM.
  On a T4 with resident weights, 128-256 slots per segment is a reasonable trade-off [e]. Weight
  streaming (70B-class models) needs segments of 1024 or more, which the ring cannot hold. For those
  models the ring is not useful and the configuration says so (§7).
* The layout inside a slab matches the segment plaintext order (SPEC 7.2) without GCM tags:

  ```
  [seed 32][uW lin0: c·n0·bpe][uW lin1]...[W r lin0: slots·m0·bpe][W r lin1: slots·m1·bpe]...
  ```

* `chunk_slots` is whatever the generator passes (`store_chunk_slots(1 MiB)`). It only defines the
  `ChunkRef` granularity in which `MaskReservation` copies rows out, which is the same cache bound
  per session as the disk store (`chunk_cache_bytes`).
* `W r` bytes use `pack_ring`'s encoding, so the existing native unpackers
  (`pgw_fast.unpack_rows_u32`, `unpack_rows`) read them unchanged. At K ≤ 32 with bpe 2 or 4 the
  writer copies with a wrapping `np.copyto(..., casting="unsafe")` into a `<u4` or `<u2` view of
  the slab. That is a single pass, and numpy releases the GIL. Other widths (K24, K40, K48) go
  through `pack_ring` plus a memcpy.

### 3.3 Slab state machine

```
            grant (ring lock)             commit: validate, header, seed-hash check
   FREE ───────────────────────▶ FILLING ─────────────────────────────────────▶ READY
    ▲                               │ abort / error / close                        │ reserve() advances
    │                               ▼                                              │ next_slot (forward only)
    │   wiper: explicit_bzero    WIPING ◀──────────── refs == 0 and next_slot == slots ─┘
    └──────────────────────────────┘          (DRAINING while next_slot == slots and refs > 0)
```

Per slab, the ring tracks this under the ring lock:

* `state`;
* `gen` (incremented on every grant);
* `header` (`SegmentHeader`; `kek_id="ram"`, `wrapped_dek=""`, `nonce_prefix="00000000"`; never
  serialized);
* `seed_hash`;
* `next_slot`;
* `refs`, the number of open `RamSegment` handles.

The rules:

* **R1.** Only `FREE` slabs are granted, and only `WIPING` turns into `FREE`, after the wipe call
  has returned.
* **R2.** `reserve()` only reads `READY` slabs. A slab stays `READY` while `next_slot < slots`.
  Once fully reserved it is `DRAINING`, and it never returns to `READY`.
* **R3.** A slab with `refs > 0` is never wiped or granted. Every `RamSegment` handle holds a
  reference from `reserve()` until `MaskReservation` closes it (`_Extent.wipe` → `seg.close()`).
* **R4 (primary guard).** `RamSegment.decrypt_into` checks `ring.closed`, `slab.gen == handle.gen` and
  `state in {READY, DRAINING}` and copies, all under a **per-slab lock**. Every transition out of
  `READY`/`DRAINING`/`FILLING` (drain, abort, close, reload) takes the same lock, and `close()` moves
  every slab to `WIPING` before anything is wiped, so a late reader always raises (§15, B1).

## 4. Reservations (gateway side)

### 4.1 `MaskRing.reserve(fp, ring, slots, *, session_id)`

1. Under `ring._lock`:
   * if the ring is closed, raise `MaskStoreError`;
   * sum `slots - next_slot` over `READY` slabs in publish (FIFO) order;
   * if that is less than `slots`, raise `RingShort(available)`, a subclass of
     `MaskExhaustedError`.
2. Otherwise pick extents oldest first:
   * `SlotExtent(segment_id, next_slot, take)` per slab;
   * `next_slot += take`; `refs += 1`;
   * a slab that is now full becomes `DRAINING`;
   * `reserved_total += slots`.

   **The cursor moves before anything is handed out.** This is the in-memory equivalent of the
   store's durable-cursor-first rule.
3. Outside the lock, build `MaskReservation(owner, rid, session_id, fp, rp, [(ext, RamSegment(slab,
   gen)), ...])`:
   * `owner` is a small adapter with `_on_reservation_closed(res, burned)`, which counts burned rows
     and drops the reservation from `_open`.
   * If the constructor raises (header mismatch), close every handle and count the slots as
     `lost`. They are never handed out again, as in the store's `reservation_lost_slots` path.
4. Record the reservation in `ring._open` (a weak value map, used for the forced close at
   shutdown) and return it.

Reserving takes the lock for microseconds, never does I/O and never waits on a wipe.

`ring_only` mode adds a short wait. With no disk tier, a `reserve()` that finds the ring short
waits up to `ring_only_wait_s` (default 2 s) on a condition variable signalled by every publish,
then raises `MaskExhaustedError` with `details={"ring_available": ..., "filling": ...}`. This
covers start-up, when the ring is empty until the first segment lands.

### 4.2 Take path (unchanged code, new byte source)

```
engine _take → WindowedReservation.take_checked → MaskReservation.take_checked
  → _take_locked → _derive_locked
      r:   _mask_rows_locked: seed = RamSegment.decrypt_into(secrets_chunk)[:32] (once per extent),
           HKDF stream key cached per (extent, linear), pgw_fast.mask_rows_with_key (bit-identical)
      W r: _read_wr → _decrypt_buf(seg, wr_chunk(li, j)) → RamSegment.decrypt_into(ref)
           = out = bytearray(n); ctypes.memmove(out, arena + off, n)  # no immutable bytes (§15, B4)
           → pgw_fast.unpack_rows_u32 / unpack_rows
```

* `RamSegment.decrypt_into(ref)` maps `ref.index` to a slab offset: index 0 is the secrets region,
  and `1 + li·chunks_per_linear + j` is chunk `j` of linear `li`, at offset
  `wr_base[li] + j·chunk_slots·m_li·bpe`. The offset table is computed once per segment. The
  `SegmentLayout(header, 0)` object is used only for `chunk_rows`, `wr_chunk` and `secrets_chunk`
  index arithmetic; its file offsets are ignored.
* Copy-out is required. Handing out views into the slab would let a caller hold rows of a slab that
  is later wiped and refilled, so the "rows" it holds would silently change into another segment's
  masks. Copies are owned by the `MaskReservation`, which zeroes them on close (existing behaviour).
* `MaskReservation`'s chunk cache holds the whole chunk (up to about 1 MiB per linear), including
  rows of other reservations' slots. This is exactly the disk store's behaviour (see the comment in
  `_derive_locked`). Those rows are only ever handed out through their own reservation's cursor,
  and the copy is zeroed when this reservation moves past the chunk or closes.
* `prefetch`, `checks_compact`, `rows_until_boundary`, `remaining` and `taken` work unchanged.
  `WindowedReservation` chains ring parts and store parts freely, because each part is an
  independent `MaskReservation`.
* Later optimization, not in v1 because it needs `store.py`: a `view_rows` fast path in `_read_wr`
  that unpacks straight from the slab into the output, skipping the chunk copy.

### 4.3 Closing and recycling

`MaskReservation.close()` runs in this order:

1. it zeroes its caches;
2. `_Extent.wipe()` calls `RamSegment.close()`, which only queues the handle on a lock-free
   `queue.SimpleQueue` (it may run from a GC finalizer); the ring applies `refs -= 1` when it next
   takes its lock (§15, B2);
3. if the slab is `DRAINING` and `refs == 0`, it becomes `WIPING` and goes to the wiper queue;
4. `owner._on_reservation_closed(res, burned)` adds `burned` to `burned_total`.

The wiper zeroes `slab_bytes` (the whole slab, including burned and never-taken rows), sets the
slab to `FREE` and notifies free-slab listeners, which wake the factory scheduler.

`RamSegment.close()` is idempotent per handle: a second close does not decrement again.

## 5. Factory side

### 5.1 `RingSegmentWriter` (in `ringbuf.py`) and `RingSink` (in `src/pgw/factory/sinks.py`)

`RingSink(rings).writer(fp, ring, specs, *, slots, seed, checks, chunk_slots, factory_id,
destination=None)` does the following:

* **Refuses** (`ConfigError`):
  * a `destination`, because ring segments are never exported;
  * a fingerprint or ring that has no `MaskRing`;
  * `specs` that differ from the ring's linears (ids, `m`, `n`, order);
  * `slots > segment_slots`;
  * `checks > 8`.
* **Refuses** a seed whose `seed_hash` is in the ring's `seen_seeds` set (`MaskReuseError`). The set
  holds every seed hash published or granted in this process, 32 B per segment.
* Grants a `FREE` slab without blocking. If there is none, it raises `RingFull`, which the
  scheduler treats as "re-plan", not as an error.
* Returns a writer that holds the slab (`FILLING`) and a private copy of the seed in a bytearray,
  zeroed on commit and abort.

`write_linear(lid, wr, uw)` uses the same contract as `SegmentWriter`:

* it validates the shape and dtype (int64, or compact int32/uint32 at K ≤ 32 once WP1/WP2 deliver
  compact output);
* it runs the range check `[0, 2^K)`, unless the generator marks the tensor as kernel-reduced (the
  same flag WP1 introduces for the store writer);
* it requires `uW` to be identical across chunks of one linear;
* it writes rows `[done, done + rows)` straight into the slab region of that linear.

The generator then zeroes `item.wr` as it does today.

`commit()`:

1. checks that every linear has `slots` rows;
2. writes the seed and `uW` into the secrets region;
3. applies the soundness floor: `soundness_bits(ring, n_worst, checks) ≥ min_soundness_bits`, the
   same rule as `MaskStore._check_soundness`, or the segment is refused;
4. builds the `SegmentHeader` (`segment_id = uuid4().hex`);
5. takes the ring lock, checks that `seed_hash` is still unseen, records it, sets the slab to
   `READY`, appends it to the FIFO, and notifies reservers;
6. returns the `segment_id`.

`abort()`, an exception, or `MaskRing.close()` while filling: the slab goes to `WIPING`, then
`FREE`. Rows that were written count as `lost`, not `published`.

There is **no path** from a ring slab to a file, the store or an export. `RingSink` has no
`export_segment`, `recover` or `state_dir`, and a ring segment is never demoted to disk. Overflow
means the *next* segment, with a fresh seed, goes to the store.

### 5.2 Scheduler (`src/pgw/factory/scheduler.py`)

* `FactoryScheduler(settings, store, generators, *, ring_sink=None)`. `RefillPlan` gains
  `tier: Literal["store", "ring"] = "store"` (frozen dataclass; the default keeps old callers
  working).
* Priority in `plan()`, most urgent first, per `(fp, ring)` that has a ring:
  1. **Ring below `ring_low_fraction`** of capacity (default 0.5): one `tier="ring"` plan with
     `segments = free_slabs` and `slots_per_segment = ring.segment_slots`.
  2. **Store below low water**: today's plan, including the ramp.
  3. **Ring not full**: a ring plan.
  4. **Store below target** (hysteresis): today's plan.
* `_produce(plan)` calls `generate_into_store(ring_sink if plan.tier == "ring" else self.sink,
  ...)`.
  * Ring plans skip `factory-state.json`. A crash mid-ring-segment leaves nothing on disk to
    recover, and an fsync per small segment would cost about a millisecond each.
  * `RingFull` drops the rest of that plan without an error.
* `_fit_to_disk` ignores ring plans. `run_forever` registers `ring_sink.add_free_listener(
  self._wake.set)`, so a recycled slab wakes the loop immediately instead of after
  `poll_interval_s` (30 s).
* `status()` and `health()` gain the ring fields (`capacity_slots`, `available_slots`,
  `free_slabs`).

### 5.3 `EmbeddedFactory` (new module, `src/pgw/factory/embedded.py`)

* Builds `FactorySettings()` from `PGW_FACTORY_*` (plus `factory_env_file`, if set, applied as
  defaults), then `build_generators(settings)` and
  `FactoryScheduler(settings, StoreSink(store) or a null sink, gens, ring_sink=RingSink(rings))`.
* Runs `run_forever` on a `pgw-embedded-factory` thread, and acquires the `FactoryLock` on the store
  state directory.
* If another factory holds the lock, it logs an error, sets `pgw_factory_errors_total`, and the
  gateway serves from the store only. This is an availability issue, not a safety issue.
* `stop()` sets the stop event. An in-flight ring writer aborts, which wipes its slab.
* Only generators whose `(fp, ring)` has a `MaskRing` and matching specs are accepted. A mismatch
  is a `ConfigError` at start-up.

## 6. Tiered source (gateway hook)

`TieredMaskSource(store: MaskStore | None, rings: dict[(fp, ring), MaskRing], *, prefer="ring")` is
a new class in `src/pgw/maskstore/tiered.py`:

| method | behaviour |
|---|---|
| `reserve(fp, ring, n, session_id=)` | ring first: on `RingShort`, count `pgw_masks_ring_fallback_reservations_total` and call `store.reserve`. When neither tier alone holds `n`, return a `ChainedReservation` (ring part + store part, both closed if either fails; §15, B7). If both tiers together are short, raise `MaskExhaustedError` with `available = ring + store`. `prefer="store"` reverses the order. |
| `inventory(fp, ring)` | store inventory with `slots_available += ring.available`; `bytes_on_disk` stays store-only |
| `available_slots`, `reservation_cache_bytes` | sum and max of both tiers |
| `metrics()` | store metrics with `pgw_masks_available_slots` and `pgw_masks_low_water` computed on the total supply, plus the `pgw_masks_ring_*` series (§8) |
| `status()` | store status plus `"ring": {fp/ring: ring.status()}` (ids, counts and sizes only) |
| `gc`, `claim_owner`, `add_low_water_listener`, `settings`, `_require_external_counter` | delegated to the store (no-ops when `store is None`) |
| `close()` | close the rings (store closing stays with its owner) |

Hooks (the gateway **does** change: `runtime.py` and `masks.py`, §15 B6). In
`InferenceRuntime.__init__`, right after the store is resolved:

```python
if settings.mask_ring.enabled:
    from pgw.maskstore.tiered import build_tiered_source
    mask_store = build_tiered_source(mask_store, settings.mask_ring, settings.all_models())
```

* Rings need the linear specs. `build_tiered_source` reads them from each masked model's pack
  manifest; the pack is opened anyway.
* `InferenceRuntime.start()` and `stop()` start and stop the `EmbeddedFactory` when `embed_factory`
  is set. `stop()` closes the rings after the model runtimes have drained their sessions.
* In ring-only mode the runtime skips `open_store` and `_require_external_counter`.
* `ModelRuntime.reload` with a new fingerprint: the old ring is closed and wiped, and a new one is
  built.

## 7. Configuration

Nested `GatewaySettings.mask_ring: MaskRingSettings` (env `PGW_GATEWAY_MASK_RING__*`):

| field | default | meaning |
|---|---|---|
| `enabled` | `false` | build rings and the tiered source |
| `capacity_bytes` | required when enabled | total locked RAM, split evenly across masked models unless `per_model_bytes` is set. Sizing: `burst_s × slots/s × slot_bytes`; for example 10 s × 100 slots/s × 5.56 MB = 5.6 GB (7B) |
| `per_model_bytes` | `{}` | `{model_name: bytes}` override |
| `segment_slots` | auto (§3.2) | slots per RAM segment; must give ≥ 4 slabs |
| `prefer` | `ring` | which tier `reserve()` tries first |
| `ring_only` | `false` | no disk store at all (no counter needed); `reserve()` waits up to `ring_only_wait_s` |
| `ring_only_wait_s` | `2.0` | see §4.1 |
| `ring_low_fraction` | `0.5` | scheduler priority threshold (§5.2) |
| `embed_factory` | `false` | run the factory in-process (v1). Mutually exclusive with `factory_env_file` (`ConfigError`), so one GPU never has two factories |
| `allow_unlocked` | `false` | DEV ONLY: run when `mlock` fails |

Validation at start-up:

* `n_slabs ≥ 4`;
* `mlock` succeeds;
* `capacity_bytes ≤ 0.8 × MemAvailable`;
* `segment_slots ≤ max_stream_rows` for every linear;
* a warning when `segment_slots < 64` on CUDA (GEMM efficiency) or when the factory needs weight
  streaming.

`ring_only` with `embed_factory=false` is refused until v2 exists, because nothing would fill the
ring.

## 8. Metrics and logging

New names in `pgw.observability.metrics.METRIC_NAMES`. All carry the labels
`(fingerprint, ring)` unless noted, so `GatewayMetrics.update_masks` applies them unchanged:

| name | type | meaning |
|---|---|---|
| `pgw_masks_ring_capacity_slots` | gauge | `n_slabs × segment_slots` |
| `pgw_masks_ring_available_slots` | gauge | unreserved slots in `READY` slabs |
| `pgw_masks_ring_free_slabs` | gauge | slabs ready to be granted |
| `pgw_masks_ring_draining_slabs` | gauge | fully reserved slabs waiting for reservations to close or for the wipe. If this stays high, reservations are leaking |
| `pgw_masks_ring_published_slots_total` | counter | slots committed into the ring |
| `pgw_masks_ring_reserved_slots_total` | counter | slots reserved from the ring |
| `pgw_masks_ring_burned_slots_total` | counter | reserved but never taken |
| `pgw_masks_ring_lost_slots_total` | counter | written but never reserved and then wiped (abort, shutdown, reload) |
| `pgw_masks_ring_fallback_reservations_total` | counter | reservations served by the store because the ring was short |
| `pgw_masks_ring_locked_bytes` | gauge, no labels | locked arena bytes (0 means unlocked, dev) |

* `pgw_masks_available_slots` becomes the total supply (store plus ring), so the existing runway
  and low-water alerts keep their meaning in `ring_only` mode.
* New alert: `pgw_masks_ring_draining_slabs == n_slabs for 5m` (all slabs pinned by open
  reservations).
* Logs: segment ids, slot counts, slab indices, sizes and timings only. Never seeds, `uW`, rows or
  addresses (SPEC 11.2).
* Ring reservations do not appear in the store's `reservations` audit table. They are logged at
  debug level with their ids.

## 9. v2: shared memory with the supervised factory (WP8b, design only)

* **Transport.** The gateway listens on `<run_dir>/maskring.sock` (mode 0600). It checks the peer
  with `SO_PEERCRED`: the uid must equal the gateway's, and the pid must equal the supervised
  child's when the gateway started it. Messages are length-prefixed JSON and carry no secrets.
  stdlib only: `os.memfd_create`, `fcntl.F_ADD_SEALS`, `socket.send_fds`/`recv_fds`.
* **Arena.**
  1. The gateway creates `memfd_create("pgw-maskring", MFD_CLOEXEC | MFD_ALLOW_SEALING)`,
     `ftruncate`s it, seals it with `F_SEAL_SHRINK | F_SEAL_GROW | F_SEAL_SEAL`, maps it
     `MAP_SHARED`, `mlock`s it and marks it `MADV_DONTDUMP`.
  2. It sends the fd in `HELLO {incarnation: 128-bit random, slab_bytes, n_slabs, models: [{fp,
     ring, linears, segment_slots}]}`.
  3. The factory maps the fd with `MAP_SHARED`, marks it `MADV_DONTDUMP` and `mlock`s it too. The
     shared pages are already locked through the gateway's mapping; the second lock only protects
     against the gateway unmapping first.
* **Ownership by messages** (no shared atomics):
  * `G→F GRANT {slab, gen}`: the gateway grants FREE, already zeroed slabs, up to
    `max_factory_slabs`.
  * `F→G PUBLISH {slab, gen, header}` hands the slab back. The gateway checks `gen`, the geometry
    and `(fp, ring, linears)`. It then **copies the seed and `uW` into private memory** and checks
    `seed_hash` against `seen_seeds`. Only then does the slab become `READY`.
  * `F→G ABORT {slab, gen}`: the slab is wiped.
  * After `PUBLISH` the factory must not touch the slab. That cannot be enforced on a shared
    mapping, but it does not matter for privacy: `r` comes only from the seed copied into private
    memory, so a later write by the factory can only corrupt `W r`. Corrupting `W r` is an
    integrity fault that the factory could already cause today (it signs and encrypts what it
    computes), and its self-check guards against it.
* **Crash handling.**
  * Factory EOF: the gateway wipes the slabs it had granted. `READY` slabs keep serving, because
    they are complete and the seed is in private memory.
  * Gateway EOF: the factory zeroes the slabs it holds, unmaps, closes the fd and reconnects. A new
    gateway incarnation always creates a **new** memfd, so old contents are never re-attached.
* **Kubernetes.** The gateway and factory containers run in one pod, sharing an `emptyDir` for the
  socket. Both need `IPC_LOCK`, and the factory container keeps its own CPU limit.

## 10. Invariants and how tests prove them

Notation: a RAM segment `S` has seed `s`, slots `0..S.slots-1`, and per-linear rows `r_ℓ[j]` and
`(W r)_ℓ[j]`.

* **I1. Exclusive slots.** Every slot `j` of `S` is in at most one extent of at most one
  reservation, ever. Proof: `next_slot` only increases, and only under the ring lock. Extents are
  `[old, new)`, and a slab never returns to `READY` (R2). It is wiped before reuse (R1), and reuse
  comes with a new seed (I3).
* **I2. Single use within a reservation.** Each `(linear, slot)` row is handed out at most once.
  This is `MaskReservation`'s per-linear forward cursor, unchanged and already tested in
  `tests/maskstore`.
* **I3. Unique seeds.** No two segments published by one gateway process share a seed: fresh
  `os.urandom` per segment, plus `seen_seeds` enforced at grant and at commit. A ring seed is never
  written to the store, an export, a file or a socket.
* **I4. No persistence, no restore.** Ring bytes live only in the locked, `DONTDUMP`, `DONTFORK`
  anonymous arena and in `MaskReservation` copies. A new process starts with an empty ring. No
  generation, journal or counter is involved, so SPEC 7.6 rollback cannot apply.
* **I5. Wipe.** Zeroing happens at these points:
  * a slab is zeroed in full before every grant (FREE implies zeroed);
  * a slab is zeroed after abort;
  * a slab is zeroed when it drains;
  * all slabs are zeroed on `close()`;
  * a reservation's copies are zeroed on its `close()`.
* **I6. No recycling under a reader.** A slab with `refs > 0` is never wiped or granted (R3).
* **I7. Same masks as disk.** For the same seed and `W r`, ring takes are bit-identical to store
  takes. Privacy (uniform `r` from ChaCha20) and integrity (Freivalds on `uW`, soundness floor) are
  unchanged.
* **I8. Fail closed on locking.** If `mlock` fails, the ring does not start, unless `allow_unlocked`
  is set.
* **I9. Accounting.** `published = reserved + available + lost`, and per reservation
  `slots = taken + burned`, where `taken = max(cursor)`.

Test plan (all CPU, tiny packs from the test fixtures, arenas ≤ 4 MiB so the default 8 MiB
`RLIMIT_MEMLOCK` is enough). `tests/maskstore/test_ringbuf.py`:

| # | test | proves |
|---|---|---|
| 1 | `test_concurrent_reserve_disjoint`: 8 threads reserve random sizes (1-40) until exhausted while a writer thread publishes, from a seeded `random.Random`. Collect `(segment_id, slot)` from every extent. Assert pairwise disjoint, union ⊆ published, `Σ = reserved_total` | I1, I9 |
| 2 | `test_rows_single_use`: two reservations on one slab take every row of every linear. Assert each slot index appears once overall, and each returned `r` row equals `SeedExpander(seed).ring_uniform_rows(..., slot, 1, "r", lid)` for its slot only | I1, I2 |
| 3 | `test_ring_matches_store`: one generator, one fixed test seed, written once into a fresh ring and once into a fresh tmp store. Reserve all and `take_checked` every linear. Assert `r`, `W r` and `checks_compact` are bit-identical | I7 |
| 4 | `test_close_burns_never_rewinds`: reserve 10 of 64, close untouched, reserve again → starts at slot 10, `burned_total == 10` | I1, I9 |
| 5 | `test_abort_wipes`: write half the linears, raise, then check the slab bytes are all zero, the slab is `FREE`, `available` is unchanged and `lost` counts the rows | I5 |
| 6 | `test_drain_wipes_and_recycles`: reserve all, take some, close. The wiper runs; the slab is all zero and `FREE`; the free listener fired once | I5 |
| 7 | `test_no_recycle_while_open`: with a reservation open on a fully reserved slab, the slab stays `DRAINING`, no grant returns it, and `ring.close()` force-closes the reservation (a later `take` raises `MaskReuseError`) before the wipe | I6 |
| 8 | `test_duplicate_seed_refused` (at `writer()` and at `commit()`, using a seed injected through the test hook) | I3 |
| 9 | `test_soundness_floor`, `test_geometry_refused` (fingerprint, ring, linears, `destination`, too many slots) | I7, §5.1 |
| 10 | `test_arena_locked_dontdump_dontfork`: parse `/proc/self/smaps` for the arena's VMA and assert `Locked == Size` and that `VmFlags` contains `lo`, `dd` and `dc` | I4, I8 |
| 11 | `test_mlock_failure_fails_closed`: monkeypatch `mlock` to return `ENOMEM` → `ConfigError`; with `allow_unlocked`, a warning and `locked_bytes == 0` | I8 |
| 12 | `test_new_ring_is_empty` and `test_close_wipes_all` (the whole arena is zero; `reserve` raises) | I4, I5 |
| 13 | `test_crash_leaves_disk_untouched`: a subprocess (spawn, not fork) runs the ring, a tmp store and a fake generator, publishes slabs and holds a reservation, then gets SIGKILL. Assert the store directory listing and `state.sqlite` contents are identical to before, and that no file under the tmp dir contains any ring segment id | I3, I4 |
| 14 | `test_accounting_property`: random interleaving of publish, reserve, take, close and abort; I9 holds after every step | I9 |

Other tests:

* `tests/factory/test_ring_sink.py`:
  * scheduler ring-first priority;
  * `RingFull` is not an error;
  * a free event wakes `run_forever` well before `poll_interval_s`;
  * `_fit_to_disk` ignores ring plans;
  * no `factory-state.json` writes for ring plans;
  * overflow goes to the store with fresh seeds (store `seed_hash` set ∩ ring `seen_seeds` = ∅).
* `tests/gateway/test_ring_tiered.py`:
  * ring-short falls back to the store and the fallback metric advances;
  * `ring_only` waits, then raises `mask_exhausted`;
  * `inventory` and metrics show the total supply;
  * a masked end-to-end generation through `TieredMaskSource` gives logits **bit-identical to
    `PlainBackend`** (SPEC 9.3), with `WindowedReservation` extensions crossing ring and store
    parts, and the noise monitor reports `duplicate_rows == 0` (SPEC 3.3, runtime backstop for I1).

## 11. Performance expectations and acceptance

| quantity | disk store today | ring v1 |
|---|---|---|
| factory write stage, 7B slot | 20.1 ms [m] | 0.5-1 ms (numpy copy, GIL released) [e] |
| delivered rate on a 1 GB/s disk | ≤ 650k/h [d] | GPU rate (L1-L4: 1.4-2.4M/h raw) [e] |
| gateway read, 7B slot | pread 5.56 MB + AES-GCM ≈ 3.7 ms [d] | memcpy 5.56 MB ≈ 0.5 ms [e] |
| reserve | `BEGIN IMMEDIATE` + fsync + witness, ≈ 1-10 ms [e] | lock + list walk, < 50 µs [e] |
| wipe cost | shred files | `explicit_bzero` about 1 GB per drained slab, about 0.1-0.2 s on the wiper thread [e] |

Acceptance (WP9 runs it):

* Local CPU, tiny model: the ring take path costs no more than the store path per slot.
* Kaggle T4, 7B: an embedded factory into the ring delivers at least 1M slots/h once WP1-WP3 have
  landed.
* Gateway TPOT with the embedded factory running is within 5% of TPOT without it. Otherwise v2
  becomes the default (§2).

## 12. Failure modes

| failure | effect | handling |
|---|---|---|
| factory exception mid-segment | slab partially written | `abort()` → wipe → `FREE`; rows counted as `lost`; scheduler backs off as today |
| factory thread hangs or stops | ring drains | `reserve()` falls back to the store; existing `pgw_factory_last_success_timestamp` stall alert; `ring_only` → `mask_exhausted` after `ring_only_wait_s` |
| gateway crash / SIGKILL / OOM kill | all ring contents gone | nothing persisted; restart with an empty ring; the store serves; no masks reused (I4) |
| `mlock` fails / memlock limit | ring cannot start | `ConfigError` at start-up with the fix (§3.1); dev override only |
| cgroup memory too small | populate or lock fails at start-up | start-up error, never mid-request (`MAP_POPULATE` + `mlock` commit every page) |
| reservation leak (session never closed) | slab pinned in `DRAINING` | `MaskReservation.__del__` closes it on a best-effort basis; `draining_slabs` alert; `ring.close()` force-closes |
| wiper backlog | factory waits for slabs | `free_slabs` metric; one wiper is enough at ≤ 10 GB/s memset [e] |
| duplicate seed (bug) | commit refused | `MaskReuseError`, slab wiped, error logged (ids only) |
| wrong specs, ring or fingerprint | writer refused | `ConfigError` at `writer()`; `EmbeddedFactory` checks at start-up |
| checks below the soundness floor | segment refused | same rule as `MaskStore._check_soundness` |
| two factories on one GPU or store | contention | `embed_factory` excludes `factory_env_file`; `FactoryLock` on the state dir |
| `fork()` of the gateway (subprocess) | child could inherit the arena | `MADV_DONTFORK`: the arena is not mapped in children |
| core dump | masks in the dump | `MADV_DONTDUMP` for the arena. Chunk caches outside the arena are treated as today (optionally `prctl(PR_SET_DUMPABLE, 0)` gateway-wide) |
| host hibernation / VM snapshot | RAM written to disk | out of scope (servers); the docs advise against VM memory snapshots of a gateway, as for any secret in gateway RAM |
| model reload with a new fingerprint | old masks are useless | close and wipe the old ring (`lost`), then build a new one |

The trust boundary is unchanged. The gateway already sees prompts, activations and `x` in
plaintext (SPEC 3.1), and anyone who can read its memory already has everything the masks protect.
At-rest encryption only ever protected against disk theft, snapshots and backups. The ring has no
at-rest state to protect. **Scope of the swap/dump protection:** only the arena (v1) and the slab
mappings (v2) are `mlock`ed and `DONTDUMP`. Reservation chunk caches, prefetch stages, returned
tensors, the writer's seed copy and the generator's buffers live on the ordinary heap. That is why
the ring makes `PR_SET_DUMPABLE=0` mandatory (no core dumps, no same-uid ptrace or `/proc/<pid>/mem`)
and refuses to start with unencrypted disk swap (`swap_policy=refuse`; zram and dm-crypt swap pass).

## 13. SPEC and docs changes (with the implementation)

* **SPEC 7.8 (new) "In-memory mask ring".** Slab state machine, rules R1-R4, the "never persisted,
  never exported, never demoted" rule, mlock and dontdump requirements, and that the external
  counter applies only to persisted segments.
* **SPEC 7.5.** Add a line: ring reservations advance an in-memory cursor under the ring lock before
  any bytes are copied out. They are not durable because the masks are not durable.
* **SPEC 11.1.** The new metric names.
* `docs/operations/sizing.md`: ring capacity, memlock and the pod memory limit.
* Helm gateway chart: `maskRing.*` values, `IPC_LOCK`, a GPU request when `embedFactory`, and a
  memory limit that includes the capacity.

## 14. Implementation plan (one agent, 1-2 days for v1)

| step | files | size |
|---|---|---|
| 1. `LockedArena`, `MaskRing`, `RamSegment`, `RingSegmentWriter`, wiper | new `src/pgw/maskstore/ringbuf.py` | ~450 lines |
| 2. tests 1-14 | new `tests/maskstore/test_ringbuf.py` | ~450 lines |
| 3. `TieredMaskSource`, `build_tiered_source` | new `src/pgw/maskstore/tiered.py` | ~180 lines |
| 4. `RingSink` | `src/pgw/factory/sinks.py` | ~40 lines |
| 5. tier-aware plans, free-slab wake | `src/pgw/factory/scheduler.py` | ~80 lines |
| 6. `EmbeddedFactory` | new `src/pgw/factory/embedded.py` | ~100 lines |
| 7. settings, hook, supervisor exclusivity, metric names | `src/pgw/gateway/config.py`, `src/pgw/gateway/runtime.py` (about 15 lines), `src/pgw/gateway/cli.py` (about 3 lines), `src/pgw/observability/metrics.py` | ~120 lines |
| 8. factory and gateway tests | new `tests/factory/test_ring_sink.py`, `tests/gateway/test_ring_tiered.py` | ~350 lines |
| 9. SPEC 7.8, sizing, Helm values | `SPEC.md`, `docs/operations/sizing.md`, `deploy/helm/pgw-gateway` | docs |

The implementation needs no edits to `store.py`, `segment_io.py`, `generator.py`,
`factory/kernels.py`, `hemask/*`, `service/*` or `rust/pgw-fast`. It depends on these existing
contracts staying stable:

* `MaskReservation(owner, rid, session_id, fp, ring, [(SlotExtent, seg)])`, where `seg` provides
  `header`, `layout`, `decrypt_into(ref) -> bytearray` and `close()`, and `owner` provides
  `_on_reservation_closed`;
* `SegmentWriter`'s `write_linear`, `commit` and `abort` contract, as called by
  `generate_into_store`.

A contract test pins the first one (a `RamSegment` round-trip through `take_checked`, `prefetch`,
`checks_compact` and `close`), so a later `store.py` refactor that breaks it fails loudly.

The v2 shared-memory transport (§9) is a separate follow-up of 2-3 days.

## 15. Design-review resolution and implementation notes (2026-11-20)

The review (verdict `merge_with_fixes`, privacy and exactness blocked) raised seven blocking issues.
All are fixed in code and covered by tests (`tests/maskstore/test_ringbuf.py`, `test_ringshm.py`,
`tests/factory/test_ring_sink.py`, `tests/gateway/test_ring_tiered.py`).

| # | issue | resolution | test |
|---|---|---|---|
| B1 | reserve/close race could serve a zero or stale seed | R4 is the primary guard (per-slab lock around check + copy, every transition takes it); `close()`/reload set `closed` and mark every slab `WIPING` under its lock before any wipe; the reservation (or a placeholder with its handles) is registered in `_open` inside the step-2 critical section; every secrets read verifies `seed_hash` against the hash recorded at publish and refuses an all-zero seed | `test_reserve_interleaved_with_close_never_serves_zero_rows`, `test_concurrent_takes_during_close_raise_or_are_exact`, `test_seed_check_on_secrets_read` |
| B2 | finalizer could deadlock the ring lock | `RamSegment.close()` takes no lock: it queues the handle on a `queue.SimpleQueue`; the ring reconciles under its lock (any public call, `_on_reservation_closed`, or the wiper every 250 ms). Free listeners run on the wiper thread with no lock held. Handles closed by `MaskReservation.__del__` (`RingReservation` marks them) count as `leaked` | `test_finalizer_close_while_ring_lock_held` |
| B3 | wiper could write into unmapped memory | the ctypes export is held for the arena lifetime; `close()`: set closed + WIPING, force-close reservations, join the wiper, wipe the whole arena synchronously, `munlock`, `munmap` (stop the factory first: the runtime does) | `test_close_joins_wiper_before_unmap`, `test_new_ring_is_empty_and_close_wipes_all` |
| B4 | copy-out left an immutable plaintext copy | `ctypes.memmove` from the arena into a fresh `bytearray`; the writer packs with `np.copyto` into the arena; uW is packed into a zeroable `bytearray` (no `pack_ring` bytes) | `test_copy_out_makes_no_immutable_copy` |
| B5 | v2 handed the whole arena to any same-uid process | v2 (`ringshm.py`) uses a `SOCK_SEQPACKET` socketpair created before spawning the child (`pass_fds`, no listening socket); one fresh memfd per grant (never reused, never the arena); both processes `PR_SET_DUMPABLE=0` (the child reports it, a dumpable child is killed); the child `mlock`s (fail closed) and sets `DONTDUMP`/`DONTFORK`; on EOF the gateway kills the child through its pidfd and reaps it before dropping its slabs; `PR_SET_PDEATHSIG` kills the child with the gateway | `test_child_process_fills_ring_and_crash_is_reaped`, `test_runtime_ring_only_embedded_factory[factory_process]` |
| B6 | long sessions pinned every slab; `ring_only` waits stalled the batch | `WindowedReservation` closes fully used parts at the start of the next token step (the first linear's `ensure()`), never mid-step; extensions never wait (`reserve_extension`), and the next window is reserved ahead in a background thread once fewer than `window/2` rows are left | `test_long_session_does_not_pin_the_ring`, `test_retire_only_at_step_start`, `test_windows_reserved_ahead_in_background` |
| B7 | requests that fit only across tiers were refused | `TieredMaskSource.reserve` returns a `ChainedReservation` (ring part, then store part; both closed if either fails); `ring_only` start-up refuses models with `max_context + mask_window_tokens > max_session_fraction (0.5) x capacity_slots` | `test_spanning_reservation_larger_than_either_tier`, `test_spanning_failure_closes_ring_part`, `test_ring_only_waits_then_exhausts_and_extensions_fail_fast` |

Minor issues:

* **Swap/dump scope**: §12 rewritten; `PR_SET_DUMPABLE=0` is mandatory with the ring and swap is
  checked at start-up (`swap_policy`, default `refuse`).
* **Seed uniqueness across tiers**: one `SeedRegistry` (process-wide) is shared by every ring and by
  the embedded factory's `StoreSink` (which claims each store seed); `RingSink` also checks the store's
  seed table. Seeds are claimed at grant (v1) or publish (v2).
* **Freivalds hardening**: the engine treats all-zero check vectors as `MaskReuseError` in the
  soundness check, in `finish_group` and in `finish_chunk` (also when the kernel reported success).
  The ring's soundness floor is the store's `min_soundness_bits`, or the gateway's in `ring_only`.
* **Integrity of W r**: v2 publishes by `F_SEAL_WRITE` (the kernel refuses while the child maps the
  slab writable), so a post-publish write is impossible. v1 is one address space; no checksum was
  added (RAM faults are an ECC matter; the factory self-check and the gateway's range check stay).
* **K < 32 wrap**: compact 32-bit input is range-checked for K < 32; int64 input is always
  range-checked before the wrapping copy.
* **mmap flags**: `MAP_POPULATE`, `MADV_DONTDUMP`, `MADV_DONTFORK` exist on Python 3.13 / Linux 6.x;
  `madvise` failure is fail-closed like `mlock`; `allow_unlocked` needs `allow_dev_modes`.
* **Two embedded factories on one GPU**: besides `factory.lock` in the state directory (`run_dir` in
  `ring_only`), the embedded factory takes a host-wide per-device lock (`/run/lock/pgw/pgw-factory-<device>.lock`).
* **Audit**: `pgw_masks_ring_reservations_total` (ids-only counter) per model; reservations are
  logged at debug level with their ids.

Changes against the text above:

* Metric names are `pgw_masks_ring_*` (the metric contract allows `pgw_masks_*`), plus
  `pgw_masks_ring_leaked_slots_total` and `pgw_masks_ring_reservations_total`.
* Accounting (I9): `published = reserved + available + lost` and
  `reserved = taken + burned + leaked + open`; aborted fills are counted separately (`aborted`).
* v2 slabs are dropped, not zeroed: the write seal that makes them immutable also forbids the wipe.
  The kernel zeroes freed pages before handing them to any process (`init_on_free=1` zeroes at free;
  the gateway checks for it at start-up, see below). Unsealed slabs are zeroed before the drop.
  The gateway maps each slab read-only through a read-only reopen of the memfd, so its locked
  mapping does not block the seal; pages stay locked from grant to drop.
* Config: `mask_ring.factory_process` (v2) is mutually exclusive with `embed_factory` and
  `factory_env_file`; `ring_only` needs one of the two factories. The Kubernetes sidecar layout is
  not supported (it would need a listening socket).
* Implementation review round 2 (all fixed): ring segment ids are never written to
  `factory-state.json` (counts only); the `SeedRegistry` is bounded (live slab seeds pinned until
  their wipe, plus a 65,536-entry recent window; the store's seed table remembers store seeds
  forever); v2 drops sealed slabs unzeroed, which SPEC 7.8 now states, the gateway warns (or with
  `init_on_free_policy=refuse` refuses) at start-up unless `init_on_free=1` is detected, and
  unsealed (aborted) slabs are zeroed through the memfd before they are dropped; every ring `mlock`
  (v1 arena, v2 slabs in both processes) raises `RLIMIT_MEMLOCK` soft-to-hard first
  (`raise_memlock_limit`); `store_knows_seed` prefers a public `store.has_seed()` and fails closed
  when neither it nor the expected internals exist; the memfd unmap and the publish copy both run
  under the slab lock, and `RingBroker.stop()` joins its reader before reaping (a grant that races
  the reap is aborted at once); the v2 supervisor takes the same host-wide per-device lock as the
  embedded factory; `WindowedReservation.ensure` counts in-flight background slots against the cap;
  `pgw_masks_ring_fallback_reservations_total` is per model; the v2 child gets only
  `PGW_FACTORY_*` and runtime environment variables (`ringshm.child_env`).
* Not done: the `view_rows` fast path in `_read_wr` (needs `store.py`, owned by another package),
  Helm values and `docs/operations/sizing.md` (owned by WP9).

Local measurements (4 vCPU, one Qwen2.5-7B-shaped layer at K32, 64 slots; a 7B slot is 28 layers)
[m]: factory write 0.11 ms/layer-slot into the ring vs 4.12 ms into the store (Python AES-GCM,
fsync off), i.e. about 3 ms vs 115 ms per 7B slot; gateway take (decode, one row per linear per step,
`r` regenerated with ChaCha20) 0.23 vs 0.51 ms/layer-slot, i.e. about 6.4 vs 14.3 ms per 7B slot.
