#!/usr/bin/env python3
"""Scaling model: private, exact, streamed decoding of 7B-72B dense models through a CPU-only gateway.

Spreadsheet-like arithmetic for ``research/plans/scaling-32b-70b.md``. Standard library only; every input is a
named constant or a ``Scenario`` field with a provenance tag in its comment:

* ``measured``: a number measured in this repository (the source is named);
* ``derived``: arithmetic on a measured number (e.g. ns per vector element from a per-token kernel total);
* ``estimate``: an assumption (hardware datasheet figure, efficiency, a target that has not been built).

What it computes, per model shape (real configs of Qwen2.5-7B/14B/32B/72B and Llama-3-70B) and scenario:

* trips per decode pass (4 per layer + the masked LM head today; 2 per layer if fused linear chains land);
* wire bytes per trip (K/8 bytes per input element up, per output element down, per row);
* gateway CPU per trip: measured native-kernel ns per input / output element at 7B shapes, scaled by the vector
  sizes, divided over the cores a trip is split across;
* provider GPU time per trip: a memory-bandwidth model (int8 weights read once per trip whatever the rows, the
  K/8 limb GEMMs compute-bound only at many rows), tensor parallel by output rows (no all-reduce: each trip is
  one linear and its output goes back to the gateway, so a gather of output slices), per-trip sync latency;
* trip latency = network RTT + service overhead + GPU + wire serialization + gateway CPU;
* per-pass gateway work outside trips: float ops and glue per layer, attention over the KV cache (bytes and
  FLOPs: GQA attention on a CPU is compute-bound at a few rows), the LM head's sampling;
* speculation: chain (sequential or pipelined with m = 0, the reviewed L3 drafter) and tree (greedy optimal tree
  of R nodes under a rank-acceptance model), draft step cost on the CPU (memory-bandwidth bound);
* tok/s = tokens per pass / pass time, and the component that bounds it;
* masks: slot bytes, slots per token, slots/s and bytes/s at the achieved rate; the mask factory's work for the
  encrypted (MLWE + hint, today), plain-on-a-customer-cloud-GPU and hint-free (RLWE, placeholder) schemes;
* storage at the gateway and the factory; $/hour per stream and tenants per GPU set;
* the smallest sets of improvements ("levers") that reach a tok/s goal.

Usage::

    python benchmarks/scaling_model.py                 # the plan's tables (markdown)
    python benchmarks/scaling_model.py --json out.json # everything as JSON
    python benchmarks/scaling_model.py --model qwen2.5-72b --levers all --ctx 4096
"""

from __future__ import annotations

import argparse
import heapq
import itertools
import json
import math
from dataclasses import asdict, dataclass, field, replace
from typing import Any

MB, GB = 1e6, 1e9


# ----------------------------------------------------------------------------------------------------------------
# Model shapes (real configs; config.json of each checkpoint)
# ----------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Shape:
    name: str
    layers: int
    hidden: int
    inter: int
    heads: int
    kv_heads: int
    head_dim: int
    vocab: int
    tied: bool = False

    def linears(self) -> list[tuple[str, int, int]]:
        """(kind, n_in, m_out) of the four masked linears of one layer (SPEC 2)."""
        q, kv = self.heads * self.head_dim, self.kv_heads * self.head_dim
        return [("qkv", self.hidden, q + 2 * kv), ("o", q, self.hidden),
                ("gate_up", self.hidden, 2 * self.inter), ("down", self.inter, self.hidden)]

    def head(self) -> tuple[str, int, int]:
        return ("lm_head", self.hidden, self.vocab)

    def sum_n(self, with_head: bool = True) -> int:
        """Input elements per row-token over every masked linear (the gateway's prepare side)."""
        s = self.layers * sum(n for _, n, _ in self.linears())
        return s + (self.hidden if with_head else 0)

    def sum_m(self, with_head: bool = True) -> int:
        """Output elements per row-token (the gateway's finish side; one mask word each)."""
        s = self.layers * sum(m for _, _, m in self.linears())
        return s + (self.vocab if with_head else 0)

    def sum_mn(self, with_head: bool = True) -> int:
        """Int8 weights the provider holds and reads per trip set (one byte each)."""
        s = self.layers * sum(n * m for _, n, m in self.linears())
        return s + (self.hidden * self.vocab if with_head else 0)

    def params(self) -> float:
        emb = self.vocab * self.hidden
        return self.sum_mn(with_head=False) + emb * (1 if self.tied else 2)

    def kv_bytes_per_ctx_token(self, kv_bytes: int) -> int:
        """KV cache bytes per context position over all layers (K and V)."""
        return 2 * self.layers * self.kv_heads * self.head_dim * kv_bytes

    def attn_flops_per_row(self, ctx: int) -> float:
        """Attention FLOPs of one query row at context ``ctx`` (QK^T and PV, every head)."""
        return 4.0 * self.layers * self.heads * self.head_dim * ctx


SHAPES: dict[str, Shape] = {s.name: s for s in (
    Shape("qwen2.5-7b", 28, 3584, 18944, 28, 4, 128, 152064),
    Shape("qwen2.5-14b", 48, 5120, 13824, 40, 8, 128, 152064),
    Shape("qwen2.5-32b", 64, 5120, 27648, 40, 8, 128, 152064),
    Shape("qwen2.5-72b", 80, 8192, 29568, 64, 8, 128, 152064),
    Shape("llama-3-70b", 80, 8192, 28672, 64, 8, 128, 128256),
)}

#: Draft models (int8 weights streamed per draft step; the step is memory-bandwidth bound on a CPU).
#: Qwen2.5-0.5B: measured 0.49 GB of int8 per step (strict-speculation plan, section 2). Qwen2.5-1.5B and
#: Llama-3.2-1B (same tokenizer as Llama-3-70B): int8 size from the parameter count (estimate).
DRAFTS: dict[str, float] = {"qwen2.5-0.5b": 0.49 * GB, "qwen2.5-1.5b": 1.55 * GB, "llama-3.2-1b": 1.24 * GB}
#: "eagle": a one-layer feature-level drafter on the target's hidden state with a truncated 32k-token head
#: (EAGLE-style; would need training per target; estimate), its bytes computed per shape in ``draft_bytes``.
DRAFT_FOR = {"qwen2.5-7b": "qwen2.5-0.5b", "qwen2.5-14b": "qwen2.5-0.5b", "qwen2.5-32b": "qwen2.5-0.5b",
             "qwen2.5-72b": "qwen2.5-0.5b", "llama-3-70b": "llama-3.2-1b"}


# ----------------------------------------------------------------------------------------------------------------
# Hardware
# ----------------------------------------------------------------------------------------------------------------

#: Provider GPUs (datasheet, SXM; estimate for what a kernel achieves: ``bw_eff`` of the HBM bandwidth on GEMVs
#: of >= ~8 MB, ``tops_eff`` of dense int8 peak). ``usd_h``: assumed on-demand rate per GPU-hour (estimate,
#: market range late 2026; override with --gpu-usd-h).
GPUS: dict[str, dict[str, float]] = {
    "h100": {"hbm_tb_s": 3.35, "mem_gb": 80.0, "int8_tops": 1979.0, "usd_h": 3.0},
    "h200": {"hbm_tb_s": 4.8, "mem_gb": 141.0, "int8_tops": 1979.0, "usd_h": 3.8},
    "b200": {"hbm_tb_s": 8.0, "mem_gb": 180.0, "int8_tops": 4500.0, "usd_h": 6.0},
}
GPU_BW_EFF = 0.75  # estimate: achieved fraction of HBM bandwidth on a large int8 GEMV
GPU_TOPS_EFF = 0.5  # estimate: achieved fraction of int8 peak on the K/8-limb GEMMs
GPU_MEM_USABLE = 0.85  # estimate: fraction of HBM for weights (rest: buffers, graphs, factory batches)

#: Gateway kernels, measured on the 4-vCPU dev box (Xeon @ 2.3 GHz, shared) at Qwen2.5-7B shapes with the p31
#: check, ``native_trip_floor_bench.py --preset 7b --wnorm-frac 0.3`` (STATUS 2026-11-22, p31 table): prepare +
#: input half 6.1-6.2 ms, prime output check 2.2 ms, finish 2.5-2.6 ms, W r read 1.2-1.3 ms, r keystream 3.8 ms per
#: token of 113 trips. Derived: ns per element over sum_n = 835,072 and sum_m = 1,542,656 (7B incl. the head).
_N7, _M7 = SHAPES["qwen2.5-7b"].sum_n(), SHAPES["qwen2.5-7b"].sum_m()
NS_IN_MEASURED = 6.15e6 / _N7  # derived: 7.36 ns per input element (quantize, mask, pack, Freivalds input half)
NS_OUT_MEASURED = (2.2 + 2.55 + 1.25) * 1e6 / _M7  # derived: 3.89 ns per output element (W r, unmask, check)
NS_R_MEASURED = 3.8e6 / _N7  # derived: 4.55 ns per input element (ChaCha20 keystream of r, 8 B per element)
#: Orchestration per trip around the native kernels (lease, request object, lane call, statistics): measured
#: process CPU of the trips-only native path at 7B, p31: 29-37 ms per token, minus the 15.9 ms kernel floor, over
#: 113 trips (``trip_native_bench.py --preset 7b --masked-head``) -> derived 0.15 ms.
ORCH_MS_MEASURED = 0.15
#: Float ops and model glue per layer (norms, RoPE, SiLU, residuals, Python): measured about 0.5 ms per 7B layer
#: on the Python path at short context (weight-free plan, section 2); the native path (`float_ops = native`) is
#: assumed at 0.3 ms per layer today (estimate) of which the element work is 3 ns per element of
#: (6 hidden + 3 inter) per layer (estimate) and the rest glue.
FLOAT_GLUE_MS_TODAY = 0.25
FLOAT_NS_PER_ELEM = 3.0
#: Draft step: measured 75 ms per Qwen2.5-0.5B step on one core of this box, 46-54 ms on 4 vCPUs (9-10 GB/s),
#: 2 ms of it float ops and dispatch (estimate).
DRAFT_FIXED_MS = 2.0
DRAFT_ROW_MS = 0.3  # estimate: extra cost of one more row in a memory-bound draft step on a server
#: Mask factory measurements.
HE_7B_CORE_MS_PER_SLOT = 24.2 * 4  # measured 24.2 ms wall per 7B slot on 4 vCPUs (NTT, A and hint in NTT form)
HE_RANK, HE_DEGREE = 16, 256
HE_HINT_BYTES_PER_ROW = HE_RANK * HE_DEGREE * 6  # 24,576 B (q = 2^56, low byte dropped): measured layout
HE_PLANES = 7  # provider: 56-bit b in 7 int8 digit planes per W8 b (SPEC 7.9)
HE_V2_UP_B, HE_V2_DOWN_B = 6.625, 4.375  # bytes per element of the v2 (53/35) maskgen wire, SPEC 7.9
#: GPU NTT on a T4 (X5): kernels only, 0.94 us encrypt + 0.96 us decrypt per row and 256-slot group, end to end
#: 5.0 + 2.1 us (measured, Kaggle T4); per slot = rows x us / 256.
GPUNTT_T4_US_PER_ROW_GROUP_E2E = {"encrypt": 5.0, "decrypt": 2.1}


# ----------------------------------------------------------------------------------------------------------------
# Scenario: every knob of the per-pass model
# ----------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Scenario:
    name: str = "today"
    # trips
    trips_per_layer: int = 4  # 4 today; 2 with fused linear chains (research, other package)
    masked_head: bool = True  # the head as one more trip (weight-free gateway)
    ring_k: int = 32  # K32A12C27
    # network and service
    rtt_ms: float = 0.10  # estimate: co-located VM-to-VM TCP RTT 30-100 us; 0.10 = the slow end
    service_ms: float = 0.14  # measured: 1-row service time on Modal 0.10-0.16 ms (lane + graph replay)
    nic_gbit: float = 25.0  # estimate: gateway VM NIC
    # provider
    gpu: str = "h100"
    tp_min: int = 1  # tensor parallel degree, at least the smallest power of two the weights fit in
    tp_sync_ms: float = 0.020  # estimate: per-trip gather of output slices across GPUs (NCCL-class small message)
    gpu_kernel_ms: float = 0.008  # estimate: per-trip kernel floor beyond bytes (graph replay; ~3 us persistent)
    # gateway CPU
    core_speed: float = 1.0  # 1.0 = the dev box's core (measured); a current server core ~1.5 (estimate)
    ns_in: float = NS_IN_MEASURED
    ns_out: float = NS_OUT_MEASURED
    ns_r: float = NS_R_MEASURED
    r_offcore: bool = False  # r keystream derived ahead on helper cores (activation-independent)
    trip_cores: int = 1  # cores one trip's prepare/finish is split over
    split_sync_ms: float = 0.004  # estimate: fork/join of a split trip (spinning pool)
    orch_ms: float = ORCH_MS_MEASURED
    float_glue_ms: float = FLOAT_GLUE_MS_TODAY
    # attention on the gateway
    ctx: int = 2048
    kv_bytes: int = 4  # float32 KV (today's pinned reference); 2 = bfloat16 KV (a new pinned reference)
    attn_gb_s: float = 10.0  # estimate: one core streaming the KV cache
    attn_gflops: float = 10.0  # estimate: one core, fixed-order non-FMA float32
    # speculation
    spec: str = "chain"  # none | chain | tree
    max_rows: int = 8
    alpha: float = 0.70  # estimate: per-token acceptance of the target by the draft (no 32B/72B measurement)
    gamma: float = 0.35  # estimate: P(target = draft's next choice | not the previous ones), tree only
    draft: str | None = None  # default: DRAFT_FOR[shape]
    draft_gb_s: float = 8.4  # measured: 0.5B step 60 ms on the Modal H100 run's gVisor host (9-10 GB/s here)
    draft_fixed_ms: float = DRAFT_FIXED_MS
    draft_scale: float = 1.0  # draft bytes multiplier (0.5 = int4 draft weights; a draft only proposes)
    pipe_hyp: bool = False  # hypothesis tree in the pipelined drafter's background block (not built)
    # gateway host memory bandwidth for anything streamed per pass (local LM head); measured under gVisor on
    # Modal ~10-20 GB/s for one stream's threads (lead, 2026-11-22 7B H100 run)
    host_gb_s: float = 15.0
    # gateway VM size and price
    gateway_vcpus: int = 16
    usd_per_vcpu_h: float = 0.045  # estimate: on-demand general-purpose cloud vCPU-hour
    # masks
    narrow_wr: bool = False  # W r stored mod 2^(C+2) = 29 bits (-9.4%; not implemented)
    # prompt tokens and burned reservations per generated token (each takes a slot): measured 1.7 at 7B in the
    # H100 run (2.7 slots per generated token plain); 0 = decode only
    prompt_slots_per_token: float = 0.0

    def with_(self, **kw: Any) -> Scenario:
        return replace(self, **kw)


def tp_degree(shape: Shape, sc: Scenario) -> int:
    """Smallest power of two of GPUs that holds the int8 weights, at least ``tp_min``."""
    g = GPUS[sc.gpu]
    need = shape.sum_mn(sc.masked_head) / (g["mem_gb"] * GB * GPU_MEM_USABLE)
    tp = 1
    while tp < need:
        tp *= 2
    return max(tp, sc.tp_min)


# ----------------------------------------------------------------------------------------------------------------
# Per-trip and per-pass time
# ----------------------------------------------------------------------------------------------------------------


def trip_groups(shape: Shape, sc: Scenario) -> list[tuple[str, int, int, int]]:
    """(kind, n, m, count per pass). With fused chains (2 trips per layer) the elements are unchanged and the
    layer's four linears are served in two trips (assumption until the fused-chain design lands): a trip then
    carries the inputs and outputs of two linears and the provider reads both weights."""
    lin = shape.linears()
    if sc.trips_per_layer == 4:
        groups = [(k, n, m, shape.layers) for k, n, m in lin]
    elif sc.trips_per_layer == 2:
        (_, n0, m0), (_, n1, m1), (_, n2, m2), (_, n3, m3) = lin
        groups = [("attn_pair", n0 + n1, m0 + m1, shape.layers), ("mlp_pair", n2 + n3, m2 + m3, shape.layers)]
        groups = [(k, n, m, c) for k, n, m, c in groups]
    else:
        raise ValueError("trips_per_layer must be 4 or 2")
    if sc.masked_head:
        k, n, m = shape.head()
        groups.append((k, n, m, 1))
    return groups


def _weights_of_group(shape: Shape, kind: str) -> int:
    lin = {k: n * m for k, n, m in shape.linears()}
    if kind == "attn_pair":
        return lin["qkv"] + lin["o"]
    if kind == "mlp_pair":
        return lin["gate_up"] + lin["down"]
    if kind == "lm_head":
        return shape.hidden * shape.vocab
    return lin[kind]


def gpu_trip_ms(weight_bytes: float, rows: int, sc: Scenario, tp: int) -> float:
    """Provider GPU time of one trip: weights read once (memory-bound) or the K/8 limb GEMMs (compute-bound),
    split over ``tp`` GPUs by output rows, plus the kernel floor and the gather sync."""
    g = GPUS[sc.gpu]
    t_mem = weight_bytes / tp / (g["hbm_tb_s"] * 1e12 * GPU_BW_EFF)
    t_cmp = 2.0 * weight_bytes * (sc.ring_k // 8) * rows / tp / (g["int8_tops"] * 1e12 * GPU_TOPS_EFF)
    return 1e3 * max(t_mem, t_cmp) + sc.gpu_kernel_ms + (sc.tp_sync_ms if tp > 1 else 0.0)


def gateway_trip_ms(n: int, m: int, rows: int, sc: Scenario) -> float:
    """Gateway CPU on the critical path of one trip (prepare before, finish after the lane call)."""
    ns_in = sc.ns_in + (0.0 if sc.r_offcore else sc.ns_r)
    work_ms = rows * (n * ns_in + m * sc.ns_out) * 1e-6 / sc.core_speed
    par = max(1, sc.trip_cores)
    return sc.orch_ms + work_ms / par + (sc.split_sync_ms if par > 1 else 0.0)


def wire_trip_ms(n: int, m: int, rows: int, sc: Scenario) -> float:
    return rows * (n + m) * (sc.ring_k / 8) * 8 / (sc.nic_gbit * 1e9) * 1e3


def pass_breakdown(shape: Shape, sc: Scenario, rows: int) -> dict[str, float]:
    """ms per decode pass of ``rows`` rows, by component (excluding draft work)."""
    tp = tp_degree(shape, sc)
    out = {"network": 0.0, "service": 0.0, "gpu": 0.0, "wire": 0.0, "gateway_trip": 0.0}
    trips = 0
    for kind, n, m, count in trip_groups(shape, sc):
        trips += count
        out["network"] += count * sc.rtt_ms
        out["service"] += count * sc.service_ms
        out["gpu"] += count * gpu_trip_ms(_weights_of_group(shape, kind), rows, sc, tp)
        out["wire"] += count * wire_trip_ms(n, m, rows, sc)
        out["gateway_trip"] += count * gateway_trip_ms(n, m, rows, sc)
    elems = shape.layers * (6 * shape.hidden + 3 * shape.inter) * rows
    out["gateway_float"] = shape.layers * sc.float_glue_ms + elems * FLOAT_NS_PER_ELEM * 1e-6 / sc.core_speed \
        / max(1, sc.trip_cores)
    # local bfloat16 LM head on the gateway CPU (masked_head off): the table is streamed once per pass
    out["head_local"] = 0.0 if sc.masked_head else 1e3 * shape.vocab * shape.hidden * 2 / (sc.host_gb_s * GB)
    kv = shape.kv_bytes_per_ctx_token(sc.kv_bytes) * sc.ctx
    flops = shape.attn_flops_per_row(sc.ctx) * rows
    out["attention"] = 1e3 * max(kv / (sc.attn_gb_s * GB), flops / (sc.attn_gflops * 1e9))
    out["sampling"] = 0.15 * rows / sc.core_speed  # estimate: final norm + argmax/sampling over the vocab
    out["_trips"] = trips
    out["_tp"] = tp
    return out


def pass_ms(b: dict[str, float]) -> float:
    return sum(v for k, v in b.items() if not k.startswith("_"))


# ----------------------------------------------------------------------------------------------------------------
# Speculation
# ----------------------------------------------------------------------------------------------------------------


def tau_chain(alpha: float, rows: int) -> float:
    """Tokens per pass of sequential chain drafting with ``rows`` rows (1 real + rows - 1 drafts)."""
    if alpha >= 1.0:
        return float(rows)
    return (1 - alpha ** rows) / (1 - alpha)


def tau_pipe(alpha: float, rows: int, m: int = 0) -> float:
    """Pipelined drafter (L3 reviewed, PipelinedDrafter): stationary tokens per pass of the two-state chain."""
    k = rows - 1
    e_full, h_full = tau_chain(alpha, k + 1), alpha ** (k + 1)
    e_miss, h_miss = tau_chain(alpha, m + 1), alpha ** (m + 1)
    if m >= k:
        return e_full
    p_full = h_miss / ((1 - h_full) + h_miss)
    return p_full * e_full + (1 - p_full) * e_miss


def rank_probs(alpha: float, gamma: float, width: int = 4) -> list[float]:
    """P(target's token = draft's i-th choice): alpha, then (1 - alpha) gamma (1 - gamma)^(i - 2)."""
    return [alpha] + [(1 - alpha) * gamma * (1 - gamma) ** (i - 2) for i in range(2, width + 1)]


def tau_tree(alpha: float, gamma: float, rows: int) -> tuple[float, int]:
    """Greedy optimal draft tree of ``rows - 1`` nodes (the root row is the last accepted token) under
    independent per-depth rank acceptance; returns (expected tokens per pass, depth). Greedy is optimal here
    because a child's path probability never exceeds its parent's."""
    probs = rank_probs(alpha, gamma)
    heap: list[tuple[float, int, int]] = [(-probs[0], 1, 0)]  # (-path prob, depth, rank index)
    total, depth, nodes = 1.0, 0, 0
    pushed_sibling: set[tuple[float, int, int]] = set()
    while heap and nodes < rows - 1:
        negp, d, r = heapq.heappop(heap)
        p = -negp
        total += p
        depth = max(depth, d)
        nodes += 1
        # the node's first child and its next sibling become candidates
        heapq.heappush(heap, (-(p * probs[0]), d + 1, 0))
        if r + 1 < len(probs):
            parent = p / probs[r]
            key = (parent, d, r + 1)
            if key not in pushed_sibling:
                pushed_sibling.add(key)
                heapq.heappush(heap, (-(parent * probs[r + 1]), d, r + 1))
    return total, depth


def draft_bytes(shape: Shape, sc: Scenario) -> float:
    name = sc.draft or DRAFT_FOR[shape.name]
    if name == "eagle":
        return sum(n * m for _, n, m in shape.linears()) + 32768 * shape.hidden
    return DRAFTS[name]


def draft_step_ms(shape: Shape, sc: Scenario, rows: int = 1) -> float:
    return draft_bytes(shape, sc) * sc.draft_scale / (sc.draft_gb_s * GB) * 1e3 + sc.draft_fixed_ms \
        + DRAFT_ROW_MS * (rows - 1)


@dataclass
class Result:
    shape: str
    scenario: str
    mode: str
    rows: int
    tau: float
    pass_ms: float
    tok_s: float
    bound: str
    breakdown: dict[str, float] = field(default_factory=dict)
    trips: int = 0
    tp: int = 1
    trip_ms_1row: float = 0.0


def tau_pipe_hyp(alpha: float, gamma: float, rows: int) -> float:
    """Pipelined drafter with a hypothesis tree in the background block (not built; the strict-speculation plan's
    follow-up): while a pass with k = rows - 1 drafts is in flight, the block drafts one chain per acceptance
    point j = 0..k, each starting from the guessed correction token (the draft's top choice after full acceptance,
    its second choice after a rejection). The next pass carries k drafts if the realised outcome was guessed
    (P = alpha^(k+1) + (1 - alpha^k) gamma after a full pass; alpha + (1 - alpha) gamma after a pass without
    drafts, whose block drafts two chains), else none. The foreground work between passes is the same either way
    (m = 0), as in PipelinedDrafter, so the gap does not depend on the acceptance."""
    k = rows - 1
    e_full = tau_chain(alpha, k + 1)
    h_full = alpha ** (k + 1) + (1 - alpha ** k) * gamma
    h_miss = alpha + (1 - alpha) * gamma
    p_full = h_miss / ((1 - h_full) + h_miss)
    return p_full * e_full + (1 - p_full) * 1.0


def _groups(b: dict[str, float]) -> dict[str, float]:
    return {"network+service": b["network"] + b["service"], "provider GPU": b["gpu"], "wire": b["wire"],
            "gateway trip CPU": b["gateway_trip"],
            "gateway float+attention": b["gateway_float"] + b["attention"] + b["sampling"],
            "gateway local head": b["head_local"], "draft": b.get("draft", 0.0)}


def evaluate_mode(shape: Shape, sc: Scenario, mode: str, rows: int,
                  _cache: dict[int, dict[str, float]] | None = None) -> Result:
    cache = _cache if _cache is not None else {}
    if rows not in cache:
        cache[rows] = pass_breakdown(shape, sc, rows)
    if 1 not in cache:
        cache[1] = pass_breakdown(shape, sc, 1)
    b = dict(cache[rows])
    base = pass_ms(b)
    step = draft_step_ms(shape, sc)
    draft = 0.0
    if mode == "plain":
        tau = 1.0
    elif mode == "chain_seq":
        tau = tau_chain(sc.alpha, rows)
        draft = (rows - 1) * step
    elif mode == "chain_pipe":
        tau = tau_pipe(sc.alpha, rows, 0)
        draft = max(0.0, rows * step - base)  # the background block (k + 1 steps) beyond the pass
    elif mode == "chain_pipe_hyp":
        tau = tau_pipe_hyp(sc.alpha, sc.gamma, rows)
        draft = max(0.0, rows * draft_step_ms(shape, sc, rows) - base)  # k + 1 chains as rows of one step
    elif mode == "tree_seq":
        tau, depth = tau_tree(sc.alpha, sc.gamma, rows)
        width = max(1, math.ceil((rows - 1) / max(depth, 1)))
        draft = depth * draft_step_ms(shape, sc, width)
    else:
        raise ValueError(mode)
    b["draft"] = draft
    total = base + draft
    g = _groups(b)
    bound = max(g, key=g.get)
    one = cache[1]
    trip1 = (one["network"] + one["service"] + one["gpu"] + one["wire"] + one["gateway_trip"]) / one["_trips"]
    return Result(shape.name, sc.name, mode, rows, tau, total, 1e3 * tau / total, bound,
                  {k: round(v, 3) for k, v in b.items()}, int(b["_trips"]), int(b["_tp"]), trip1)


def evaluate(shape: Shape, sc: Scenario) -> Result:
    """Best tok/s over the speculation modes the scenario allows and rows 1..max_rows."""
    cache: dict[int, dict[str, float]] = {}
    cands = [evaluate_mode(shape, sc, "plain", 1, cache)]
    if sc.spec in ("chain", "tree"):
        for r in range(2, sc.max_rows + 1):
            cands.append(evaluate_mode(shape, sc, "chain_seq", r, cache))
            cands.append(evaluate_mode(shape, sc, "chain_pipe", r, cache))
            if sc.pipe_hyp:
                cands.append(evaluate_mode(shape, sc, "chain_pipe_hyp", r, cache))
    if sc.spec == "tree":
        for r in range(3, sc.max_rows + 1):
            cands.append(evaluate_mode(shape, sc, "tree_seq", r, cache))
    return max(cands, key=lambda c: c.tok_s)


# ----------------------------------------------------------------------------------------------------------------
# Masks, factory, storage, cost
# ----------------------------------------------------------------------------------------------------------------


def slot_bytes(shape: Shape, sc: Scenario) -> float:
    """W r bytes per slot (one row of every masked linear; r itself is derived from a seed)."""
    bits = 29 if sc.narrow_wr else sc.ring_k
    return shape.sum_m(sc.masked_head) * bits / 8


def mask_demand(shape: Shape, sc: Scenario, res: Result) -> dict[str, float]:
    # masked fillers: every row of a pass takes a slot; plus prompt rows and burned reservations
    slots_per_token = res.rows / res.tau + sc.prompt_slots_per_token
    slots_s = res.tok_s * slots_per_token
    sb = slot_bytes(shape, sc)
    return {"slot_mb": sb / MB, "slots_per_token": slots_per_token, "slots_per_s": slots_s,
            "mask_gb_s": slots_s * sb / GB, "ring_10s_gb": 10 * slots_s * sb / GB,
            "check_vectors_mb_per_row_token": (shape.sum_n(sc.masked_head) + shape.sum_m(sc.masked_head)) * 4 / MB}


def factory(shape: Shape, sc: Scenario, slots_s: float) -> dict[str, dict[str, float]]:
    """Work of the three factory schemes at ``slots_s`` (estimates scaled from the 7B measurements)."""
    sn, sm, smn = shape.sum_n(sc.masked_head), shape.sum_m(sc.masked_head), shape.sum_mn(sc.masked_head)
    h100 = GPUS["h100"]
    # (1) encrypted MLWE + hint (today): provider multiplies ciphertexts, customer encrypts/decrypts with NTT
    he_core_ms = HE_7B_CORE_MS_PER_SLOT * (sn + sm) / (_N7 + _M7)
    he_gpu_s = 2.0 * smn * HE_PLANES / (h100["int8_tops"] * 1e12 * GPU_TOPS_EFF)
    he_wire = sn * HE_V2_UP_B + sm * HE_V2_DOWN_B
    gpuntt_t4_ms = (sn * GPUNTT_T4_US_PER_ROW_GROUP_E2E["encrypt"] + sm * GPUNTT_T4_US_PER_ROW_GROUP_E2E["decrypt"]) \
        / HE_DEGREE / 1e3
    mlwe = {"customer_core_ms_per_slot": he_core_ms, "customer_cores": he_core_ms * slots_s / 1e3,
            "customer_gpuntt_t4_ms_per_slot": gpuntt_t4_ms, "customer_t4s": gpuntt_t4_ms * slots_s / 1e3,
            "provider_h100_ms_per_slot": he_gpu_s * 1e3, "provider_h100_share": he_gpu_s * slots_s,
            "wire_mb_per_slot": he_wire / MB, "wire_gbit_s": he_wire * slots_s * 8 / 1e9,
            "hint_gb": sm * HE_HINT_BYTES_PER_ROW / GB, "hint_ntt_form_gb": 2 * sm * HE_HINT_BYTES_PER_ROW / GB}
    # (2) plain factory on a customer-cloud GPU: W8 r itself (public weights, K/8 limbs, batched: compute-bound)
    plain_s = 2.0 * smn * (sc.ring_k // 8) / (h100["int8_tops"] * 1e12 * GPU_TOPS_EFF)
    l4_s = 2.0 * smn * (sc.ring_k // 8) / (485e12 * GPU_TOPS_EFF)
    plain = {"h100_ms_per_slot": plain_s * 1e3, "h100_share": plain_s * slots_s,
             "l4_ms_per_slot": l4_s * 1e3, "l4_share": l4_s * slots_s, "weights_gb": smn / GB,
             "hint_gb": 0.0, "r_keystream_core_ms_per_slot": sn * NS_R_MEASURED * 1e-6}
    # (3) hint-free RLWE linear HE (placeholder until the RLWE package reports): slots packed across the
    # ring's coefficient slots (no rotations), provider does m n scalar x ciphertext products in a ~60-bit RNS
    # basis per slot (2 limbs), the customer encrypts n and decrypts m ciphertext coefficients per slot with one
    # NTT each (~log2 N + 2 modmults per coefficient), and the reply is 2 coefficients of ~44 bits after modulus
    # switching. All estimates.
    rlwe_gpu_s = 2.0 * smn * 2 / (h100["int8_tops"] * 1e12 * GPU_TOPS_EFF / 16)  # 64-bit modmul ~16x an int8 MAC
    rlwe_core_ms = (sn + sm) * (math.log2(8192) + 2) * 1.5e-6  # 1.5 ns per 64-bit modmul (estimate)
    rlwe_wire = sn * 8 + sm * 2 * 44 / 8
    rlwe = {"customer_core_ms_per_slot": rlwe_core_ms, "customer_cores": rlwe_core_ms * slots_s / 1e3,
            "provider_h100_ms_per_slot": rlwe_gpu_s * 1e3, "provider_h100_share": rlwe_gpu_s * slots_s,
            "wire_mb_per_slot": rlwe_wire / MB, "wire_gbit_s": rlwe_wire * slots_s * 8 / 1e9, "hint_gb": 0.0}
    return {"mlwe_hint": mlwe, "plain_cloud_gpu": plain, "rlwe_hint_free": rlwe}


def storage(shape: Shape, sc: Scenario, res: Result, md: dict[str, float]) -> dict[str, float]:
    return {"embeddings_bf16_gb": shape.vocab * shape.hidden * 2 / GB,
            "scales_norms_mb": (shape.sum_m(sc.masked_head) * 4 + shape.layers * 2 * shape.hidden * 4) / MB,
            "kv_per_seq_gb": shape.kv_bytes_per_ctx_token(sc.kv_bytes) * sc.ctx / GB,
            "draft_gb": (draft_bytes(shape, sc) if sc.spec != "none" else 0.0) / GB,
            "mask_ring_10s_gb": md["ring_10s_gb"],
            "check_vectors_gb": (shape.sum_n(sc.masked_head) + shape.sum_m(sc.masked_head)) * 4 * 2 / GB}


def provider_util(shape: Shape, sc: Scenario, res: Result) -> dict[str, float]:
    """GPU-set busy fraction for one stream and how many streams share it (time multiplexing, M/D/1)."""
    busy = res.breakdown["gpu"] / res.pass_ms
    rho_max = 0.6
    streams = max(1, int(rho_max / max(busy, 1e-9)))
    rho = min(0.95, streams * busy)
    gpu_per_trip = res.breakdown["gpu"] / max(res.trips, 1)
    wait = rho / (2 * (1 - rho)) * gpu_per_trip  # M/D/1 mean wait per trip
    return {"busy_one_stream": busy, "streams_at_rho_0.6": streams, "added_wait_ms_per_trip": wait,
            "added_ms_per_pass": wait * res.trips}


def cost_per_hour(shape: Shape, sc: Scenario, res: Result, streams: int, factory_usd_h: float) -> dict[str, float]:
    gpus = res.tp * GPUS[sc.gpu]["usd_h"]
    gw = sc.gateway_vcpus * sc.usd_per_vcpu_h
    return {"gpu_set_usd_h": gpus, "gpu_per_stream_usd_h": gpus / streams, "gateway_usd_h": gw,
            "factory_usd_h": factory_usd_h, "per_stream_usd_h": gpus / streams + gw + factory_usd_h,
            "usd_per_mtok": (gpus / streams + gw + factory_usd_h) / (res.tok_s * 3600) * 1e6}


# ----------------------------------------------------------------------------------------------------------------
# Scenarios and levers
# ----------------------------------------------------------------------------------------------------------------

#: Improvements, each with its numeric target. Applied on top of TODAY.
LEVERS: dict[str, dict[str, Any]] = {
    "net": {"rtt_ms": 0.04, "nic_gbit": 100.0},  # same placement group / host, busy-poll or kernel bypass
    "persistent": {"service_ms": 0.025, "gpu_kernel_ms": 0.003},  # persistent kernel, pinned/GPUDirect rings
    "tp_fast": {"tp_min": 4, "tp_sync_ms": 0.005},  # 8-way output split, one-shot NVLink gather
    "gw_native": {"orch_ms": 0.02, "float_glue_ms": 0.03, "r_offcore": True},  # whole trip + layer glue native
    "gw_parallel": {"trip_cores": 16, "core_speed": 1.5, "gateway_vcpus": 48},  # split each trip over 16 cores
    "gw_kernels": {"ns_in": 2.0, "ns_out": 1.5},  # bandwidth-floor kernels (~16 B touched per element)
    "attn_parallel": {"attn_gb_s": 100.0, "attn_gflops": 600.0},  # KV heads x ctx blocks over 16-32 cores
    "kv_bf16": {"kv_bytes": 2},  # a new pinned reference with bfloat16 KV
    "tree": {"spec": "tree", "max_rows": 16},  # tree speculation, R up to 16
    "draft_bw": {"draft_gb_s": 40.0},  # draft on its own cores/memory channels (non-gVisor server VM)
    "fused2": {"trips_per_layer": 2},  # fused linear chains (research)
    "draft_fast": {"draft_scale": 0.5, "draft_gb_s": 80.0, "draft_fixed_ms": 1.0},  # int4 draft, own cores
    "pipe_hyp": {"pipe_hyp": True},  # hypothesis tree in the pipelined drafter's background block
    "net_floor": {"rtt_ms": 0.03, "nic_gbit": 200.0},  # stretch: same rack, kernel-bypass/RDMA lane
    "service_floor": {"service_ms": 0.015, "gpu_kernel_ms": 0.002},  # stretch: GPUDirect RDMA into a polling kernel
    "b200": {"gpu": "b200"},
    "attn_host": {"attn_gb_s": 300.0, "attn_gflops": 1500.0, "gateway_vcpus": 96},  # whole 2-socket host
}

TODAY = Scenario()


def apply(base: Scenario, levers: list[str] | tuple[str, ...], name: str | None = None) -> Scenario:
    sc = base
    for lv in levers:
        sc = sc.with_(**LEVERS[lv])
    return sc.with_(name=name or ("+".join(levers) if levers else base.name))


_B = ["net", "persistent", "tp_fast", "gw_native", "gw_parallel", "gw_kernels", "attn_parallel", "draft_bw"]
_C = [*_B, "net_floor", "service_floor", "draft_fast", "pipe_hyp"]
SCENARIOS: dict[str, list[str]] = {
    "A today": [],
    "B engineered": _B,
    "C B + floors + fast draft": _C,
    "D C + fused 2/layer": [*_C, "fused2"],
    "E D + bf16 KV + tree + B200": [*_C, "fused2", "kv_bf16", "tree", "b200"],
    "F E + whole-host gateway": [*_C, "fused2", "kv_bf16", "tree", "b200", "attn_host"],
}

#: Measured co-located 7B run (lead, 2026-11-22, Modal): H100 provider, 32-core CPU-only gateway under gVisor,
#: every linear remote, native trip path, p31, local bfloat16 LM head on the gateway CPU, short context.
MEASURED_7B_H100 = {"tok_s_plain": 11.0, "ms_per_token": 91.0, "trips": 112, "trip_ms": 0.26, "service_ms": 0.066,
                    "wire_ms": 0.19, "draft_step_ms": 60.0, "tau_r2": 1.19, "tok_s_r2": 7.0,
                    "slots_per_token_plain": 2.7, "slots_per_token_r2": 4.2, "factory_slots_s": (20, 25)}


def alpha_from_tau_pipe(tau: float, rows: int = 2) -> float:
    """Invert ``tau_pipe(alpha, rows, 0)`` by bisection (measured tokens per pass -> per-token acceptance)."""
    lo, hi = 0.0, 0.999
    for _ in range(60):
        mid = (lo + hi) / 2
        if tau_pipe(mid, rows, 0) < tau:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def validate_7b() -> dict[str, float]:
    """The model with the measured run's settings, against the measured run."""
    meas = MEASURED_7B_H100
    lane = meas["wire_ms"]  # measured wire share of the lane call = RTT + serialization under gVisor
    sc = TODAY.with_(name="modal-7b-measured", masked_head=False, rtt_ms=lane, nic_gbit=1e6,
                     service_ms=meas["service_ms"], gpu_kernel_ms=0.0, ctx=128, core_speed=1.5, host_gb_s=20.0,
                     spec="none")
    shape = SHAPES["qwen2.5-7b"]
    r = evaluate(shape, sc)
    trips_ms = r.breakdown["network"] + r.breakdown["service"]
    return {"model_ms_per_token": r.pass_ms, "measured_ms_per_token": meas["ms_per_token"],
            "model_trips": r.trips - 0, "model_lane_ms": trips_ms, "measured_lane_ms": meas["trips"] * meas["trip_ms"],
            "model_gateway_ms": r.pass_ms - trips_ms,
            "measured_gateway_ms": meas["ms_per_token"] - meas["trips"] * meas["trip_ms"],
            "model_head_local_ms": r.breakdown["head_local"], "alpha_from_tau": alpha_from_tau_pipe(meas["tau_r2"]),
            "draft_gb_s_from_step": DRAFTS["qwen2.5-0.5b"] / ((meas["draft_step_ms"] - DRAFT_FIXED_MS) / 1e3) / GB}


def minimal_lever_sets(shape: Shape, goal: float, base: Scenario = TODAY, candidates: list[str] | None = None,
                       max_report: int = 5) -> list[tuple[tuple[str, ...], float]]:
    """Smallest lever sets whose best mode reaches ``goal`` tok/s (then by tok/s, descending).

    Levers are (nearly) monotone, so: the levers without which even all the others miss the goal are
    necessary; the search then adds subsets of the remaining ones to them in increasing size."""
    cands = list(candidates or LEVERS)
    if evaluate(shape, apply(base, cands)).tok_s < goal:
        return []
    need = [lv for lv in cands if evaluate(shape, apply(base, [c for c in cands if c != lv])).tok_s < goal]
    rest = [lv for lv in cands if lv not in need]
    for k in range(0, len(rest) + 1):
        hits = []
        for combo in itertools.combinations(rest, k):
            chosen = tuple(lv for lv in cands if lv in need or lv in combo)
            r = evaluate(shape, apply(base, list(chosen)))
            if r.tok_s >= goal:
                hits.append((chosen, r.tok_s))
        if hits:
            return sorted(hits, key=lambda h: -h[1])[:max_report]
    return []


# ----------------------------------------------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------------------------------------------


def report(models: list[str], ctx: int, alpha: float) -> dict[str, Any]:
    out: dict[str, Any] = {"ctx": ctx, "alpha": alpha, "results": []}
    for sname, levers in SCENARIOS.items():
        sc = apply(TODAY.with_(ctx=ctx, alpha=alpha), levers, sname)
        for mname in models:
            shape = SHAPES[mname]
            res = evaluate(shape, sc)
            md = mask_demand(shape, sc, res)
            fac = factory(shape, sc, md["slots_per_s"])
            util = provider_util(shape, sc, res)
            cost = cost_per_hour(shape, sc, res, util["streams_at_rho_0.6"], 0.0)
            out["results"].append({"result": asdict(res), "mask": md, "factory": fac, "util": util, "cost": cost,
                                   "storage": storage(shape, sc, res, md)})
    return out


def _fmt_table(rep: dict[str, Any]) -> str:
    lines = ["| scenario | model | tok/s | mode, R, tau | pass ms | trips x 1-row trip ms | TP | bound |",
             "|---|---|---|---|---|---|---|---|"]
    for e in rep["results"]:
        r = e["result"]
        lines.append(f"| {r['scenario']} | {r['shape']} | **{r['tok_s']:.1f}** | {r['mode']}, {r['rows']}, "
                     f"{r['tau']:.2f} | {r['pass_ms']:.1f} | {r['trips']} x {r['trip_ms_1row']:.3f} | {r['tp']} | "
                     f"{r['bound']} |")
    return "\n".join(lines)


def _fmt_breakdown(rep: dict[str, Any]) -> str:
    keys = ["network", "service", "gpu", "wire", "gateway_trip", "gateway_float", "attention", "sampling", "draft"]
    lines = ["| scenario | model | " + " | ".join(keys) + " |", "|---|---|" + "---|" * len(keys)]
    for e in rep["results"]:
        r = e["result"]
        lines.append(f"| {r['scenario']} | {r['shape']} | " + " | ".join(f"{r['breakdown'][k]:.1f}" for k in keys)
                     + " |")
    return "\n".join(lines)


def _fmt_masks(rep: dict[str, Any], scenario: str) -> str:
    lines = ["| model | slot MB | slots/token | slots/s | masks GB/s | 10 s ring GB | MLWE: customer cores / hint GB "
             "(NTT form) / factory wire Gbit/s / provider H100 share | plain cloud GPU: H100 share / L4 share / "
             "weights GB | RLWE hint-free (placeholder): cores / wire Gbit/s / H100 share |",
             "|---|---|---|---|---|---|---|---|---|"]
    for e in rep["results"]:
        if e["result"]["scenario"] != scenario:
            continue
        m, f = e["mask"], e["factory"]
        ml, pl, rl = f["mlwe_hint"], f["plain_cloud_gpu"], f["rlwe_hint_free"]
        lines.append(f"| {e['result']['shape']} | {m['slot_mb']:.1f} | {m['slots_per_token']:.2f} | "
                     f"{m['slots_per_s']:.0f} | {m['mask_gb_s']:.2f} | {m['ring_10s_gb']:.0f} | "
                     f"{ml['customer_cores']:.0f} / {ml['hint_gb']:.0f} ({ml['hint_ntt_form_gb']:.0f}) / "
                     f"{ml['wire_gbit_s']:.0f} / {ml['provider_h100_share']:.2f} | {pl['h100_share']:.2f} / "
                     f"{pl['l4_share']:.2f} / {pl['weights_gb']:.0f} | {rl['customer_cores']:.1f} / "
                     f"{rl['wire_gbit_s']:.0f} / {rl['provider_h100_share']:.2f} |")
    return "\n".join(lines)


def _fmt_cost(rep: dict[str, Any], scenario: str) -> str:
    lines = ["| model | GPUs | GPU busy, 1 stream | streams per GPU set (rho 0.6) | added ms/pass | GPU set $/h | "
             "$/h per stream (GPU share + gateway) | $ per M tokens |", "|---|---|---|---|---|---|---|---|"]
    for e in rep["results"]:
        if e["result"]["scenario"] != scenario:
            continue
        u, c, r = e["util"], e["cost"], e["result"]
        lines.append(f"| {r['shape']} | {r['tp']} | {u['busy_one_stream']:.2f} | {u['streams_at_rho_0.6']} | "
                     f"{u['added_ms_per_pass']:.1f} | {c['gpu_set_usd_h']:.1f} | {c['per_stream_usd_h']:.2f} | "
                     f"{c['usd_per_mtok']:.1f} |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", action="append", choices=sorted(SHAPES))
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--alpha", type=float, default=0.70)
    ap.add_argument("--levers", default=None, help="comma list of LEVERS, or 'all', for one custom scenario")
    ap.add_argument("--goal", type=float, nargs="*", default=None, help="find minimal lever sets for these tok/s")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    models = a.model or list(SHAPES)
    if a.levers:
        names = list(LEVERS) if a.levers == "all" else a.levers.split(",")
        sc = apply(TODAY.with_(ctx=a.ctx, alpha=a.alpha), names)
        for mname in models:
            r = evaluate(SHAPES[mname], sc)
            print(json.dumps(asdict(r), indent=1))
        return
    rep = report(models, a.ctx, a.alpha)
    print(f"## tok/s per model and scenario (ctx {a.ctx}, alpha {a.alpha})\n")
    print(_fmt_table(rep))
    print("\n## ms per pass by component\n")
    print(_fmt_breakdown(rep))
    last = "D C + fused 2/layer"
    print(f"\n## masks and factory at the rate of '{last}'\n")
    print(_fmt_masks(rep, last))
    print(f"\n## provider utilisation and cost, '{last}'\n")
    print(_fmt_cost(rep, last))
    print("\n## acceptance sweep (tok/s, best mode)\n")
    alphas = (0.49, 0.6, 0.7, 0.8)
    print("| scenario | model | " + " | ".join(f"alpha {x}" for x in alphas) + " |")
    print("|---|---|" + "---|" * len(alphas))
    rep["alpha_sweep"] = []
    for sname, levers in SCENARIOS.items():
        for mname in models:
            row = []
            for al in alphas:
                r = evaluate(SHAPES[mname], apply(TODAY.with_(ctx=a.ctx, alpha=al), levers, sname))
                row.append(f"{r.tok_s:.1f} ({r.mode} R{r.rows})")
                rep["alpha_sweep"].append({"scenario": sname, "model": mname, "alpha": al, "tok_s": r.tok_s,
                                           "mode": r.mode, "rows": r.rows})
            print(f"| {sname} | {mname} | " + " | ".join(row) + " |")
    print("\n## context sweep (tok/s at alpha 0.7)\n")
    ctxs = (512, 2048, 8192, 32768)
    print("| scenario | model | " + " | ".join(f"ctx {c}" for c in ctxs) + " |")
    print("|---|---|" + "---|" * len(ctxs))
    for sname, levers in list(SCENARIOS.items())[2:]:
        for mname in models:
            vals = [evaluate(SHAPES[mname], apply(TODAY.with_(ctx=c, alpha=0.7), levers, sname)).tok_s for c in ctxs]
            print(f"| {sname} | {mname} | " + " | ".join(f"{v:.1f}" for v in vals) + " |")
    v = validate_7b()
    rep["validate_7b"] = v
    print("\n## validation against the measured 7B H100 run\n")
    for k, val in v.items():
        print(f"- {k}: {val:.2f}")
    goals = a.goal if a.goal is not None else [25.0, 30.0]
    rep["minimal"] = {}
    print("\n## smallest lever sets (from today)\n")
    for mname in models:
        for g in goals:
            sets = minimal_lever_sets(SHAPES[mname], g, TODAY.with_(ctx=a.ctx, alpha=a.alpha))
            rep["minimal"][f"{mname}@{g}"] = [[list(s), t] for s, t in sets]
            txt = "; ".join(f"{'+'.join(s)} ({t:.1f})" for s, t in sets[:3]) or "not reachable with these levers"
            print(f"- {mname} >= {g:.0f} tok/s: {txt}")
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(rep, fh, indent=1)


if __name__ == "__main__":
    main()
