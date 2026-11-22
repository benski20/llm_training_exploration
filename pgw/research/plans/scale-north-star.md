# Scale plan: hard-to-serve models at high throughput (north star)

Goal: serve models that are hard to run on-prem (Llama-3-70B and 3.1-405B dense, Mixtral-8x22B and
Qwen3-235B-A22B MoE) through PGW at 1-5k tokens/s aggregate. Outputs stay bit-identical to the plaintext
quantized reference, and the provider learns nothing data-dependent.

Every number below comes from `benchmarks/scale_sizing.py` (calculator, stdlib only). Measured inputs are
cited where they appear. Tags: **analytic** = exact arithmetic on shapes; **measured**; **derived** =
arithmetic on a measured number; **estimate** = rests on an assumption. Ring: K32A12C27 (the default).
"Tokens" are prompt plus generated tokens; each uses one mask slot per row.

```
python benchmarks/scale_sizing.py --all --tokens-per-s 1000 --gpu h100 --markdown
python benchmarks/scale_sizing.py --model llama-3-70b --tokens-per-s 5000 --gpu h100 --encrypted
python benchmarks/scale_sizing.py --model mixtral-8x22b --tokens-per-s 1000 --gpu-tops <measured> --encrypted
```

## 1. Summary

1. **Per-token bytes are the wall, not FLOPs.** One 70B token is 43.9 MB of online wire and 26.9 MB of
   masks (analytic). At 1k tokens/s that is 351 Gbit/s of wire and 26.9 GB/s of masks made and consumed.
   With the encrypted supply, another 57.6 MB per token (461 Gbit/s) crosses the link. No protocol knob
   in scope shrinks these bytes: masked data is uniform, so it does not compress. The online upload and
   download are both needed, and the gateway must hold `W r` at K bits, because the Freivalds and range
   checks read the high bits. So the topology has to bring the bandwidth:
   - gateways co-located with the provider (same datacenter, 100-400 Gbit/s NICs);
   - many gateways sharded by user, each NIC-bound at about 200 tokens/s of 70B on 100 Gbit/s;
   - masks kept in RAM or on the GPU, made just in time, never round-tripped through disk.
2. **GPU cost is about 2x plain inference with a trusted factory, about 3x with the encrypted factory.**
   Online provider compute is one exact ring matmul per token (4 int8 limbs at K32). The plain factory
   does the same work again (`W r`). The encrypted factory's provider side uses 7 byte planes, about 1.75x
   the online work.
3. **In encrypted mode the gateway's HE work is the largest unit count.** At 1k tokens/s of 70B the
   encrypt+decrypt needs about 120 four-vCPU units even on the NTT path (7B measured 24 ms/slot on this
   box, scaled by rows), against about 11 H100s for the provider's maskgen. The HE hint `W8 A` is 165 GB
   for 70B, 2.2 TB for Qwen3-235B (all experts), and twice that in NTT form. This is the next engineering
   target after topology (section 6).
4. **MoE can be served privately, but routing hiding costs wire, not masks.** `pgw.moe.oblivious` (new,
   off by default, exact) gives every expert the same public row count. Rows not routed to an expert are
   uniform dummies that need no mask slot. So masks, mask storage and the factory scale with the
   **active** parameters, while wire and provider compute scale with the **total** parameters. For
   Mixtral-8x22B the wire is 3.5x a (leaky) routed-only dispatch; for Qwen3-235B it is 13x (626 MB per
   token). Fine-grained MoE is therefore wire-bound, and section 7.4 gives the levers.
5. **Batching across users is mandatory and sufficient for the provider.** A 70B decode step is 320
   sequential trips. At 0.2 ms RTT plus 0.5 ms service per trip, a step takes 0.22 s (estimate), so 1k
   tokens/s needs about 224 concurrent streams. The trips stay the same; only the rows per trip grow.

## 2. Per-token costs (analytic, K32A12C27)

| model | int8 weights | online wire/token | masks/token | trips/step | HE supply wire/token | HE hint |
|---|---|---|---|---|---|---|
| Qwen2.5-7B (calibration) | 6.5 GB | 8.89 MB | 5.56 MB | 112 | 11.6 MB | 34 GB |
| Llama-3-70B | 68.5 GB | 43.9 MB | 26.9 MB | 320 | 57.6 MB | 165 GB |
| Llama-3.1-405B | 402 GB | 131 MB | 79.5 MB | 504 | 172 MB | 488 GB |
| Mixtral-8x22B (E=8, k=2), oblivious | 140 GB | 116 MB | 20.6 MB | 224 | 43.9 MB | 448 GB |
| Qwen3-235B-A22B (E=128, k=8), oblivious | 234 GB | 626 MB | 26.6 MB | 376 | 64.8 MB | 2,150 GB |

The 7B row matches `docs/operations/sizing.md` (5.56 MB per slot, 8.89 MB per token). For MoE the HE
supply only covers routed rows, but the hint covers every expert.

## 3. At 1,000 tokens/s (H100 provider; estimate unless marked)

| | 70B plain | 70B HE | 405B plain | Mixtral plain | Mixtral HE | Qwen3-235B plain |
|---|---|---|---|---|---|---|
| online wire (analytic) | 351 Gbit/s | 351 Gbit/s | 1.05 Tbit/s | 929 Gbit/s | 929 Gbit/s | 5.0 Tbit/s |
| HE supply wire (analytic) | - | 461 Gbit/s | - | - | 351 Gbit/s | - |
| mask rate (analytic) | 26.9 GB/s | 26.9 GB/s | 79.5 GB/s | 20.6 GB/s | 20.6 GB/s | 26.6 GB/s |
| gateways, 100 Gbit/s NIC at 70% | 5.0 | 11.6 | 15 | 13.3 | 18.3 | 71.5 |
| concurrent streams needed | 224 | 224 | 353 | 157 | 157 | 263 |
| provider H100s online | 6.1 | 6.1 | 35.7 | 12.5 | 12.5 | 20.8 |
| H100s to hold one replica | 1.0 | 1.0 | 6.0 | 2.1 | 2.1 | 3.5 |
| factory H100s (plain, trusted) | 6.1 | - | 35.7 | 3.5 | - | 1.9 |
| provider H100s maskgen (HE) | - | 10.7 | - | - | 6.0 | - |
| gateway HE 4-vCPU units (NTT) | - | 120 | - | - | 91 | - |
| mask store, 10 min buffer | 16 TB | 16 TB | 48 TB | 12 TB | 12 TB | 16 TB |
| mask store bandwidth (write + read) | 54 GB/s | 54 GB/s | 159 GB/s | 41 GB/s | 41 GB/s | 53 GB/s |

The provider GPU counts use the T4's effective ring-matmul throughput, scaled to an H100 by peak int8
TOPS (**estimate**; the scaling is the weakest number in this table). The T4 figure itself is measured;
see section 9. Everything scales linearly in tokens/s: at **5k tokens/s of 70B with HE** the tool gives
about 600 gateway HE units, 58 gateways, 53 maskgen H100s, 30 online H100s and 269 GB/s of mask-store
bandwidth.

## 4. Where the bottleneck moves

| regime | binding constraint | lever |
|---|---|---|
| 7B, one stream | latency: 112 sequential trips x RTT | co-location (RTT 0.1-0.3 ms), fewer trips per token (other agents: per-trip overhead, speculation) |
| 7B, 1k tokens/s | mask bytes: 5.6 GB/s; HE gateway work (24 units at NTT speed) | RAM ring buffer (`maskstore/ringbuf.py`), gateway GPU for HE |
| 70B, 1k tokens/s | wire (351 Gbit/s) -> gateway count; a trusted plain factory needs as many GPUs as the provider | shard users over about 5 gateways at 100 Gbit/s; factory GPUs on the customer's side |
| 70B, 1k tokens/s, HE | gateway HE (120 units) >> provider maskgen (11 H100s) > online (6 H100s) | HE gateway on GPU with NTT (section 6); larger groups per call to amortise the hint stream |
| 70B, 5k tokens/s | mask-store bandwidth (269 GB/s) and wire (1.75 Tbit/s + supply) | no disk in the mask path: factory -> RAM/GPU -> gateway, per gateway |
| Mixtral-8x22B | wire x3.5 from oblivious dispatch | section 7.4 |
| Qwen3-235B-A22B (128 experts) | wire x13 (626 MB/token); 2.2 TB hint | section 7.4; plain factory or per-expert hint streaming |

## 5. Decode batching across users

- **Trips per step** are fixed by the model: `4 L` (qkv, o, gate_up, down). MoE experts go in the same
  two FFN trips, every expert in one multi-linear call. A masked router would add one trip per layer;
  see 7.3.
- **Rows per trip** = concurrent streams on the gateway (`max_concurrent_sessions`, default 32; set it to
  `tokens/s x step latency`: about 224 for 70B at 1k tokens/s). The service already coalesces rows across
  gateways while a batch waits for its device (`batch_window_ms`).
- **Step latency** is about `4 L x (RTT + service per trip)` plus gateway time. The service time per trip
  at B rows comes from the provider bench (section 9). Up to a few hundred rows, a ring GEMM at B rows
  costs little more than at 1 row (weight streaming dominates), so throughput grows almost linearly with
  B until the GPU is compute-bound.
- **Masks per trip**: B rows need B slots per linear. A slot is per token, so batching does not reduce
  mask bytes (one-time pads cannot be shared across rows). It only amortises the per-trip fixed costs.

## 6. Mask supply

**Plain factory (trusted, holds W8).** Cost = one exact ring matmul per slot, the same as online compute.
On a T4 it measured 0.032 s/slot for 70B at K32 with weights streamed from host RAM. With weights resident
and slots batched (1024 slots per GEMM) it is compute-bound; the bench measures it (section 9).

**Encrypted factory (provider multiplies MLWE ciphertexts).** The provider side is 7 byte planes per
`W b`, about 1.75x the online work (estimate, consistent with the measured 7B T4 0.019 s/slot x 9 for
70B). The customer side dominates:

| 7B, per slot, customer encrypt + decrypt | ms | basis |
|---|---|---|
| dense, 4 vCPU without AMX (this box) | 8,527 | measured 2026-11-21, `hemask_ntt_bench.py` |
| NTT drop-in (`pgw_ntt`) | 252 | measured, same run |
| NTT, public matrix in NTT form, 4 groups per call | 84 | measured, same run |
| NTT, matrix and hint in NTT form | **24.2** | measured, same run |
| dense, AMX 4 vCPU or gateway T4 | 116 / 110 | measured earlier (`encrypted-masks.md`) |

Scaling the 24.2 ms by rows gives 0.12 s/slot for 70B, so about 120 four-vCPU units at 1k tokens/s
(estimate). The NTT path's work per row is `rank` pointwise products and one inverse NTT, against
`4096 x planes` MACs on the dense path. A GPU NTT implementation should cut this by another 10-50x; that is
the highest-leverage item for the encrypted supply, but it sits in `hemask/fastmm.py`/CUDA engine
territory, which another group owns. The hint is the other constraint. Holding it in NTT form doubles it
(12 bytes per coefficient), and it must be streamed once per batch of groups: `hint bytes / (256 x
groups_per_call)` per slot. For 70B with 16 groups per call that is about 40 MB of hint read per slot, so
batch more groups per call when RAM bandwidth binds.

**GPU NTT on the gateway (X5, `pgw/src/pgw/hemask/gpuntt.py`, opt-in).** `A` stays on the device in NTT form,
one fused Triton kernel does the rank-16 pointwise MAC, inverse NTT and CRT to mod 2^64, and a second
kernel transforms hint byte planes per decryption pass. The output is bit-identical to the dense and
`ntt_pre` paths (checked on the T4 and by a start-up self-test). Measured on a Kaggle T4 with 4 vCPU, in
us per row and 256-slot group (`hemask_gpuntt_bench.py`, run pgw-x5-gpuntt v2):

| | GPU kernels only | end to end | 0.5B slots/s, end to end |
|---|---|---|---|
| encrypt, gpuntt (Triton) | 0.94 | 5.0 | 281 |
| encrypt, devmm dense int8 GEMM | - | 5.0 | 283 |
| encrypt, ntt_pre (CPU) | - | 6.9-7.5 | 188-204 |
| decrypt, gpuntt, hint cached / streamed | 0.96 | 2.1 / 2.5 | 396 / 338 |
| decrypt, devmm, hint cached / streamed | - | 5.2 / 5.7 | 162 / 147 |

The GPU NTT work is now 5-6x below the end-to-end cost. Encryption is bound by the CPU: the CBD noise
sampler (1.31 us per row and group, which every engine pays) plus about 2.7 us of host-side assembly and
transfers. The kernels alone would allow about 1.5k encrypt slots/s and 0.9k decrypt slots/s at 0.5B,
or about 41 ms per 70B slot on one T4 (16 ms encrypt + 25 ms decrypt; estimate, kernel time scaled by the
sum of n and the sum of m), against 0.12 s per slot for the CPU NTT on 4 vCPU above. The next levers
are on the CPU: a faster noise sampler (for example AES-CTR instead of SHAKE, or a cryptographic
generator on the device, which needs its own review) and pinned, overlapped transfers.

**Privacy notes.** Unchanged MLWE parameters (rank 16, q = 2^56, ternary secret, eta = 21). Nothing in this
plan changes the cryptography. For MoE, refills must not depend on routing (section 7.2).

## 7. MoE

### 7.1 Routing-oblivious dispatch (implemented: `pgw/src/pgw/moe/oblivious.py`, off by default)

For a trip with B token rows and E experts, every expert receives exactly `C >= B` rows. C is public: the
default is `C = B`, and an optional bucket also hides B within the bucket. A token picks k **distinct**
experts, so no expert has more than B real rows. **Overflow is impossible**, so there is no overflow path
and no data-dependent shape. Real rows are `x̂ = xq + r mod 2^K` with a fresh slot of that expert's linear,
exactly as in the dense protocol. Dummy rows are uniform in `Z_{2^K}^n` (ChaCha20 under a fresh OS-random
key) and consume no slot. Real rows sit at random positions.

**Security argument.** The service's view of a trip is (i) shapes `(E, C, n)`, a function of the public B
(B is already visible in the dense protocol), and (ii) `E·C` rows. Each real row is a one-time-pad
ciphertext with a fresh uniform r, so it is uniform and independent of all else. Each dummy is uniform by
construction. The joint distribution of all rows is therefore uniform for **every** routing, and the view
can be simulated without the routing (perfect, information-theoretic, as for dense rows). **Selective
failure:** a provider could corrupt one row and watch for an abort. `unmask_expert_output` runs the same
Freivalds check on all C rows, dummy or real, and aborts the whole trip on any failure. Detection
therefore never depends on whether a row is real. The range check, which runs after unmasking on real rows
only, can add an abort only for an error that has already passed every Freivalds vector. That happens with
probability at most `2^-soundness` per call, the dense protocol's own bound, so the leak is negligible.
**Exactness:** real rows use the unchanged mask arithmetic, so outputs are exact integers. The test
`tests/moe/test_oblivious.py` runs a SwiGLU MoE block masked and in plaintext quantized form and gets a
**bit-identical** float output (`torch.equal`). Gateway cost of dummies: `(E - k)` dummy rows per token and expert
linear, 7.6 M elements per Mixtral token (analytic). ChaCha20 runs at 50-93 M elements/s per core on this
box (measured), so that is 80-150 ms of one core per token, or about 100 cores at 1k tokens/s. At scale
the dummies must come from the gateway GPU PRNG.

### 7.2 Mask-pool refills (implemented: `pgw/src/pgw/moe/refill.py`)

Per-expert consumption follows the routing. With the **encrypted** factory the provider performs every
refill and sees which expert it is for, so "top up the low pool" would leak the routing histogram over
time. `LockstepRefill` refills **all** experts by the same amount, as a function only of the public
row counter. While a routed pool is empty, the gateway sends a constant-shape all-dummy trip
(`plan_dispatch(np.empty((0, k), int), E, rows=B)`). This hides the row values but **not the stall**:
the provider knows which weights each trip uses, so a repeated expert-layer trip and the slower token are
visible, and whether a pool runs dry depends on the routing. Every stall leaks routing information, so rho
and the reserve must be sized for a stall share of zero on the measured routing; the non-zero cells in the
table below are leaky configurations, not just slow ones. Steady state: with per-expert
routing probabilities `p_e` (summing to k), the overprovision must satisfy `rho >= E max_e p_e / k`, and
the waste is `1 - 1/rho` of the expert masks (analytic). Simulated with `simulate_refill` (64 rows per
trip, 300 trips, lag 4 trips; Zipf-skewed routing with exponent s):

| E, k | s = 0 (balanced) | s = 0.5 | s = 1.0 |
|---|---|---|---|
| 8, 2: stall share at rho 1.25 / 2 / 4 | 0 / 0 / 0 % | 4.8 / 0 / 0 % | 41 / 6.8 / 0 % |
| 128, 8: stall share at rho 1.25 / 2 / 4 | 0 / 0 / 0 % | 64 / 43 / 0 % | 89 / 83 / 66 % |

Trained MoEs carry a load-balancing loss, so real routing is closer to s = 0 than s = 1, but this needs a
measured trace. Fine-grained MoE with an encrypted factory needs rho of 2-4, which multiplies maskgen and
HE work for the expert linears. The plain factory's refills are invisible, but a pool that runs dry before a
local refill lands stalls just as visibly, so it must also stay ahead of consumption.

### 7.3 Router

The router `(E x hidden)` is tiny (Mixtral: 2.75 M weights in total). Two options: (a) the provider gives
the gateway the router weights in plaintext, like embeddings and the LM head today, which adds no trip;
(b) the router runs as a masked linear, one more trip per layer and one more mask slot group per token,
fused into neither neighbour because its input is the post-attention hidden state. (a) is the default
assumption in the calculator.

### 7.4 Reducing the oblivious wire overhead (not implemented; ranked)

1. **Shared expert input.** All experts' `gate_up` read the same hidden state. Upload one `x̂` per token
   and let the provider apply every expert to it. The upload drops from E·n to n per layer, but `W_e r`
   is then needed for every expert that could be routed: the factory makes E `W_e r` for one r, which is
   E/k times the masks for gate_up. That is a trade between wire and masks, worthwhile when wire binds and
   the plain factory is cheap.
2. **Provider-side expert sum for `down`.** Return `Σ_e W_e ĥ_e` (one row instead of E). This needs the
   gated activations `g_e h_e` quantized with one scale across experts. That **changes the quantized
   reference**, so it is a new opt-in reference definition, not a free optimisation. It also needs masks
   for the non-routed experts' rows, because their `W_e u_e` must cancel.
3. **Expert-parallel gateways.** No traffic saving. It only spreads the 5 Tbit/s of Qwen3-235B over more
   NICs.

What is **not** allowed: routed-rows-only dispatch, a capacity factor below B with an overflow path, or
refills keyed on pool levels with the encrypted factory. Each leaks routing.

## 8. Mask storage and bandwidth

- Rate = tokens/s x masks/token: 26.9 GB/s for 70B at 1k tokens/s, written once and read once.
- A 10-minute buffer is 16 TB, which is NVMe territory, and NVMe at 3 GB/s per drive needs 18 drives for
  54 GB/s. So **at scale the buffer must be seconds, in RAM or HBM**. The factory (or the HE decrypt) feeds
  each gateway's in-memory ring (`maskstore/ringbuf.py`, already present, off by default), and disk is
  only for an outage reserve consumed at a reduced rate. A 30 s RAM buffer at 70B/1k tokens/s is 0.8 TB
  across about 5 gateways, about 160 GB each.
- Narrower storage (store `W r mod 2^C` instead of `mod 2^K`) would save 16% at C27, but the Freivalds and
  range checks use the high bits of `ŷ - W r`, so it would weaken integrity. **Rejected.**
- `r` is already stored as a seed (only `W r` hits the store), so the remaining compression is zero:
  `W r` is uniform.

## 9. Measurements

- 4-vCPU dev box, `benchmarks/hemask_ntt_bench.py --preset qwen2.5-7b --rows 512 --dense-rows 128 --reps 2`:
  dense 8,527 ms/slot, NTT drop-in 252, NTT with transformed public matrix 97 (84 with 4 groups), NTT with
  matrix and hint transformed **24.2 ms/slot** (148k slots/h).
- 4-vCPU dev box: dummy-row ChaCha20 50-93 M elements/s per core (`uniform_rows`, two runs).
- Kaggle T4, `benchmarks/scale_provider_bench.py` (job `pgw-scale-provider`): KAGGLE_RESULTS_PENDING

## 10. Roadmap (ranked by leverage, with ownership)

1. **Topology for bandwidth** (deploy; owned by the Modal/topology group). Co-located gateways on
   100-400 Gbit/s NICs, sharded by user, about 200 tokens/s of 70B per 100 Gbit/s.
2. **Just-in-time masks in RAM/HBM, no disk in the hot path.** Factory or HE-decrypt output goes straight
   into each gateway's ring buffer. That means turning `ringbuf` on for scale deployments with a small
   outage reserve on NVMe (maskstore/factory; new module recommended).
3. **GPU NTT for the HE customer side**, with the hint in NTT form on GPU (HE CUDA engine owner): the
   largest unit count in encrypted mode.
4. **Batched provider at hundreds of rows per trip.** Service batching exists; provider GPU count follows
   section 9.
5. **MoE in the engine with oblivious dispatch** (this package's `pgw.moe`; engine wiring next: arch
   loader for Mixtral/Qwen-MoE, per-expert mask pools in the store, router option 7.3a).
6. **Wire levers for fine-grained MoE** (7.4), after a measured routing trace sets rho.

## 11. Privacy and exactness review checklist

- The oblivious dispatch's security argument (7.1) and the refill argument (7.2) are information-theoretic
  and rest on the one-time pad. No lattice parameter changes. They still need a human review before any
  deployment turns them on.
- A mask-pool stall (7.2) is visible to the provider and routing-dependent: size rho and the reserve so
  that it does not happen, and report any stall as a leak.
- Freivalds must run on dummy rows too. Any engine integration that skips dummy rows reopens the
  selective-failure channel.
- Dummy rows must come from a CSPRNG under a key the provider never sees, never from a seed it derives.
- The routing itself stays on the gateway: the router runs in plaintext there or masked (7.3).
- The batch size B is public, as in the dense protocol. Bucketing (`capacity_rows(..., bucket=)`) can
  coarsen it.
