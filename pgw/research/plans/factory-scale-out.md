# X1: horizontal scale-out of mask generation

Date: 2026-11-21. Scope: several factory worker processes feed one gateway's mask store at once, and one
factory spreads its encrypted maskgen calls over several provider GPUs or services, sharded by segment and
never by data. The store must stay safe with concurrent writers and readers: no slot is ever handed out
twice, and the store is crash-safe. Code: `pgw/src/pgw/factory/fleet.py`, `maskstore/segment_io.py`
(partial-file locks). Tests: `tests/factory/test_fleet.py`. Bench: `benchmarks/factory_scale_bench.py`,
Kaggle job `pgw-factory-scale-t4x2`. This work does not overlap M1 (overlapping stages inside one
`HEMaskBackend`), M2 (the provider's GPU maskgen path) or M3 (gateway decrypt, skipping local linears). Those
packages make each worker faster. This one runs more workers. The two gains multiply.

## 1. What runs

```
PGW_FACTORY_WORKERS=4
PGW_FACTORY_HE_SERVICE_URLS='["https://svc-0:8443","https://svc-1:8443"]'
pgw factory run ...                      # or --workers 4; Helm: scaleOut.workers, encrypted.serviceUrls
```

* `FactoryFleet` (supervisor) spawns N fresh interpreters, `w0 .. w{N-1}`. Spawning means no inherited CUDA
  context, file locks or keys. A dead worker is restarted with exponential back-off. SIGTERM makes every
  worker abort its current segment, and nothing partial is committed.
* Each worker is an ordinary factory with its own generators, `factory-state.json` and `factory.lock` under
  `<state or store root>/fleet/w<k>`, and `factory_id = <id>-w<k>`. Worker k uses `devices[k % len]` for
  local generation. Its HE backend starts on service `urls[k % len]`. Its torch threads default to
  `cores / N`, and its rate limit to `max_slots_per_hour / N`.
* `FleetScheduler` produces **one segment per claim**. It takes the fleet's plan lock (an `flock`, held for
  milliseconds), plans from the store inventory **plus the segments its live peers are making right now**,
  writes its claim to the `ProductionLedger` and releases the lock. Then it produces the segment and clears
  the claim after the commit or abort. The disk budget also subtracts the peers' in-flight bytes.
* Liveness: each worker holds an exclusive `flock` on `inflight/<id>.lock` for its whole life. A claim whose
  lock is free belongs to a dead worker and is ignored, so a crash never blocks production.
* `SegmentShardedRemote`: before each segment the scheduler calls `begin_segment()`, and every request of
  that segment then goes to one service. A worker stays on its service (pinned). If a segment fails, the
  segment is aborted and its seed is never used again, and the worker's next segment moves to the next
  service (failover). With fewer workers than services, a worker rotates services per segment. The hook
  lives in `FactoryScheduler._produce`, so a single-process factory (`workers=1`) with several
  `he_service_urls` rotates and fails over too (review fix; before it stayed on the first service).
* One HE hint per fleet: workers share `he_hint_dir` (default `<state_dir or .>/hemask`, as the classic
  factory) and build it under an exclusive lock, one worker at a time (review fix: per-worker state dirs
  meant N hint builds, and a shared dir without the lock let concurrent builds replace each other's
  `W8.A`; the Freivalds check would have rejected those segments, so this was a liveness, not a safety, bug).

## 2. Store safety under concurrent writers and readers

The single-use guarantee does **not** depend on the fleet code. The fleet only decides *how much* to make.
Which slots a gateway gets is decided by the store alone, so a wrong claim can cost throughput or disk
space but can never cause reuse. What the store guarantees (checked in code and by tests):

| property | mechanism | test |
|---|---|---|
| a segment is visible only when complete | temp `.part` file, fsync, atomic rename, then registration in one `BEGIN IMMEDIATE` txn | killed writer: nothing registered (`test_partial_cleanup_...`) |
| two writers never collide | uuid4 segment id and a fresh 32-byte seed, both UNIQUE in SQLite; `_check_new` repeated inside the registration txn | 3 writer processes x 4 segments: 12 distinct ids |
| a slot is handed out at most once | `reserve` advances cursors under the inter-process store `flock` plus `BEGIN IMMEDIATE`, durably before decrypting anything | 3 writers + 3 reader processes at once: every `(segment, slot)` counted, max count 1 (`test_concurrent_writer_and_reader_processes_never_share_slots`) |
| a crash never reuses | cursor committed before use; reservations of dead processes are closed, never re-opened | existing `tests/maskstore/test_concurrency_fuzz.py` |
| crash clean-up spares live writers | **new**: each writer `flock`s its `.part` file from creation until after the rename (the fd now closes after `os.replace`). `cleanup_partials`, `gc` and the orphan scan skip locked files (`partial_in_use`) | live writer's partial file survives `cleanup_partials()` + `gc()` and then commits; a SIGKILLed writer's file is unlocked and removed |

The bug the last row fixes: `FactoryScheduler._start` calls `cleanup_partials()` (max age 0) after a crash
of *this* factory. With several writers, that deleted the temp file of every concurrent segment (on Linux the
writer keeps writing to the unlinked inode, and its `os.replace` then fails, so the segment is lost, not
corrupted). This was not a reuse risk, but under a restart loop it could starve the store. The gateway's `gc`
used a 1-hour age rule, and a slow encrypted 70B segment can stay untouched for longer than that. That rule
now also respects the lock.

Residual window (does not affect safety): between `os.open(O_EXCL)` and `flock` (microseconds), a clean-up
could delete a fresh partial file. The result is one lost segment and nothing else.

Overshoot: every claim is made under the plan lock against `inventory + live claims < target`, so the fleet
stops at `ceil(target / segment_slots)` segments. Tests check this with 2 and 3 workers (7 segments for 100
slots at 16 per segment, exactly) and with two real spawned processes (4 segments). A claim is cleared only
after the commit. For an instant a peer can see the segment twice, in the inventory and in the claim. That
peer then produces less for one plan, never more.

## 3. Privacy and exactness

* **No new data leaves the gateway.** Workers run the unchanged generator and HE backend. Each request is the
  same MLWE ciphertext format and size, with fresh encryption randomness and the same rank-16 parameters.
  Seeds and `W r` exist only in the worker that made them and in the store.
* **What the providers see depends only on public configuration.** Each provider sees whole segments' worth
  of requests (`groups_per_call` groups each). How many segments go to each provider is a function of the
  worker count, the service list and failures. The provider already knows about failures, because it caused
  them or saw them. Colluding providers together see exactly what one provider sees without sharding. The
  refill *timing* reveals the consumption rate, as it already does with one factory. That is unchanged.
* **Exactness:** unchanged code path per segment. `test_he_segments_sharded_over_two_providers_bit_identical`
  generates two segments through two provider stubs, one segment each, and compares them with the plain local
  factory for the same seeds: `torch.equal` on every chunk. Every encrypted row is still Freivalds-checked
  before writing (`self_check_rows = -1` in HE mode). Gateways still run the Freivalds and range checks on
  every trip.
* Defaults: `workers = 1` keeps the classic single-process factory, and `he_service_urls` empty falls back
  to `he_service_url`. The only change in default behaviour is the partial-file lock, which is strictly
  safer.

## 4. Measurements

### 4.1 Local (this box: 4 vCPU, no GPU) - measured

`benchmarks/factory_scale_bench.py --preset qwen2.5-0.5b --layers 1 --workers 1 2 4 --provider-devices cpu
--segment-slots 512 --segments-per-worker 3 --he-encrypt ntt_pre`. Setup: one real `pgw service serve` on
the CPU (the provider stub), plain HTTP on loopback, synthetic checkpoint with 0.5B shapes and one decoder
layer. Steady rate = slots committed after every worker's first segment, divided by the time from the W-th
commit to the last (segment `added_at` in the store). The provider, the encrypt/decrypt and the writers all
share the same 4 cores.

| workers | steady slots/s (1 layer) | speedup | per-segment time per worker | overshoot |
|---|---|---|---|---|
| 1 | 490 | 1.00 | 1.09 s | 0 |
| 2 | 899 | 1.83 | 1.07-1.16 s | 0 |
| 4 | 1,258 | 2.56 | 2.10-2.38 s | 0 |

Whole-model equivalent (x 1/24 layers, **estimate**): 20, 37 and 52 slots/s. Inside one worker, a segment
spends about 0.5 s with the provider, 0.3 s decrypting and 0.1 s encrypting, one stage after another (that is
M1's target). Two workers overlap these stages almost perfectly. At four workers the box runs out of cores,
because the CPU provider competes with the workers.

The same run with **two** CPU provider services and 40 ms of added round-trip time per service (loopback delay
proxy, `--provider-devices cpu cpu --delay-ms 40`), measured: 578, 977 and 1,398 slots/s at 1, 2 and 4
workers (1.69x and 2.42x; whole model about 24, 41 and 58 slots/s, estimate). Every run committed exactly
`workers x 3` segments (overshoot 0).

### 4.2 Kaggle T4x2 (two provider GPUs) - KAGGLE_RESULTS_PENDING

## 5. What is left

1. Throughput per worker is M1/M2/M3's job. The fleet multiplies whatever one worker does until the shared
   resource saturates: the gateway's CPU cores (HE encrypt/decrypt), the provider GPUs or the link. On a CPU
   gateway the cores bind first, so size `workers` to about `cores / (core-seconds per segment / wall seconds
   per segment)`.
2. In-memory ring tier: the gateway's ring child (`maskstore/ringshm.py`) is still a single writer. A fleet
   feeding rings needs one granted socket per worker. That is a gateway-side change and was not done here.
3. Fleet observability: each worker serves `/metrics` on `metrics_port + k`. The Helm chart probes and
   scrapes worker 0 only. A supervisor-level `/healthz` that aggregates the workers would be cleaner.
4. Workers on several hosts would need a shared store. That is deliberately not supported, because the store
   refuses network filesystems. Scale across hosts with one gateway and store per host (users sharded), or
   with partner export and import.
