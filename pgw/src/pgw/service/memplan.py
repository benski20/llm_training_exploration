"""Provider GPU memory plan: peak device memory per model and GPU, computed from tensor shapes.

The provider's device memory is a function of public shapes and settings only (weights are int8 ``(m, n)``
matrices padded to multiples of 32; every buffer and temporary has a size fixed by ``m``, ``n``, the ring, the row
buckets and the maskgen block budget). This module adds them up the way the code allocates them, so an
out-of-memory start-up or request is caught before a GPU is rented: ``python -m pgw.service.memplan --table``.

Components per device (bytes):

* **steady** -- held from start-up on: padded weights (one copy each), row sums, maskgen per-row constants, the
  executor's static graph arenas, the CUDA-graph memory pool (the largest graph body's temporaries; graphs share
  one pool), the trip lane's graph buffers, the persistent worker's buffers (opt-in), the CUDA context and cuBLAS
  workspaces (a fixed reserve: :attr:`MemConfig.context_bytes`, measured ~0.1-0.5 GiB).
* **load** -- transients while packs load and warm up (weight upload, row sums, the Triton timing gate on GPUs
  without the int8 tensor-core variant), on top of what is already resident.
* **serving** -- transients while serving: one eager (prefill-sized) call per device (the device lock) plus
  ``maskgen_concurrency`` encrypted-maskgen items at once on their own streams (they do not take the device lock).

The model of every transient is the code's own (:func:`pgw.core.ringmm.offset_byte_work_bytes`,
:func:`pgw.service.gpu_exec.eager_work_bytes`); those are checked against allocations measured on CPU, where the
same torch code allocates the same tensors (``tests/core/test_ringmm_memory.py``). ``rev`` reproduces earlier
revisions: ``"run1"`` is the code of the Kaggle 7B run that ran out of memory (two weight copies, an int64 copy of
the LM head for its row sums), ``"bfee514"`` the first fix (one resident copy, but unbounded eager calls, an
upload transient and int64 copies in maskgen), ``"current"`` this revision. ``tests/service/test_memplan.py``
checks the ``run1`` plan against the numbers in that run's log.

GPU capacities are what ``torch.cuda.get_device_properties(i).total_memory`` reports (T4: measured on Kaggle;
others from NVIDIA's datasheets in MiB as the driver reports them).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Literal

from pgw.core.ringmm import (
    EAGER_WORK_BYTES,
    chunk_rows_for,
    offset_byte_rows,
    offset_byte_work_bytes,
    padded_shape,
)

__all__ = [
    "GPUS",
    "MODELS",
    "check_fit",
    "GpuSpec",
    "MemConfig",
    "MemPlan",
    "ModelShape",
    "main",
    "plan",
    "plan_model",
    "plan_table",
]

GIB = 1 << 30
MIB = 1 << 20
MASKGEN_PART_BYTES_CUDA = 512 * MIB  # pgw.service.kernels.MASKGEN_PART_BYTES_CUDA (mulmod64 column blocks)

Rev = Literal["current", "bfee514", "run1"]


# ---- shapes -----------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelShape:
    """Public architecture of a dense decoder (the published config)."""

    name: str
    hidden: int
    intermediate: int
    heads: int
    kv_heads: int
    head_dim: int
    layers: int
    vocab: int

    def linears(self, masked_head: bool = True) -> list[tuple[str, int, int]]:
        """``(key, m, n)`` of every linear the provider serves (``qkv, o, gate_up, down`` per layer, then the
        masked LM head when ``masked_head``)."""
        q, kv = self.heads * self.head_dim, self.kv_heads * self.head_dim
        out: list[tuple[str, int, int]] = []
        for i in range(self.layers):
            out += [(f"{i}.qkv", q + 2 * kv, self.hidden), (f"{i}.o", self.hidden, q),
                    (f"{i}.gate_up", 2 * self.intermediate, self.hidden), (f"{i}.down", self.hidden, self.intermediate)]
        if masked_head:
            out.append((f"{self.layers}.head", self.vocab, self.hidden))
        return out


#: Qwen2.5 instruct models (config.json of each checkpoint).
MODELS: dict[str, ModelShape] = {
    "qwen2.5-0.5b": ModelShape("qwen2.5-0.5b", 896, 4864, 14, 2, 64, 24, 151936),
    "qwen2.5-1.5b": ModelShape("qwen2.5-1.5b", 1536, 8960, 12, 2, 128, 28, 151936),
    "qwen2.5-3b": ModelShape("qwen2.5-3b", 2048, 11008, 16, 2, 128, 36, 151936),
    "qwen2.5-7b": ModelShape("qwen2.5-7b", 3584, 18944, 28, 4, 128, 28, 152064),
    "qwen2.5-14b": ModelShape("qwen2.5-14b", 5120, 13824, 40, 8, 128, 48, 152064),
    "qwen2.5-32b": ModelShape("qwen2.5-32b", 5120, 27648, 40, 8, 128, 64, 152064),
    "qwen2.5-72b": ModelShape("qwen2.5-72b", 8192, 29568, 64, 8, 128, 80, 152064),
}


@dataclass(frozen=True)
class GpuSpec:
    """A GPU as torch sees it."""

    name: str
    total_bytes: int  #: ``torch.cuda.get_device_properties().total_memory``
    int8_tensor_cores_triton: bool  #: Triton's int8 dot (compute capability >= 8.0): the fused GEMV's fast variant
    gemv_max_rows: int  #: rows the fused ring GEMV serves (16; 1 on a T4, measured on Kaggle)
    source: str = "datasheet"


GPUS: dict[str, GpuSpec] = {
    "t4": GpuSpec("Tesla T4 16GB", int(14.56 * GIB), False, 1, "measured (Kaggle, torch total capacity)"),
    "l4": GpuSpec("L4 24GB", 22_563 * MIB, True, 16),
    "a10g": GpuSpec("A10G 24GB", 22_563 * MIB, True, 16),
    "rtx4090": GpuSpec("RTX 4090 24GB", 24_082 * MIB, True, 16),
    "l40s": GpuSpec("L40S 48GB", 45_589 * MIB, True, 16),
    "a100-40": GpuSpec("A100 40GB", 40_442 * MIB, True, 16),
    "a100-80": GpuSpec("A100 80GB", 81_051 * MIB, True, 16),
    "h100-80": GpuSpec("H100 80GB HBM3", 81_559 * MIB, True, 16),
    "h200": GpuSpec("H200 141GB", 143_771 * MIB, True, 16),
}


# ---- settings ---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class MemConfig:
    """Provider settings that size device memory (defaults: the service's defaults with the RunPod tuning:
    masked head served, trip lane on)."""

    masked_head: bool = True
    tp: int = 1  #: output-row tensor parallel degree (``tp_devices``)
    pipeline: int = 1  #: by-layer placement over this many GPUs (``devices``, ``placement=by_layer``)
    rings_k: tuple[int, ...] = (32, 40, 48, 32)  #: K of the served rings (``rings``; the first one is warmed up)
    graph_buckets: tuple[int, ...] = (32, 64, 128)
    cuda_graphs: bool = True
    warmup: bool = True
    lane: bool = True
    lane_max_rows: int = 8
    persistent: bool = False
    max_batch_rows: int = 4096
    eager_work_bytes: int = EAGER_WORK_BYTES
    maskgen: bool = True
    maskgen_concurrency: int = 2
    #: ``v3``: the SDK's pks-v3 (GEMM + key switch, hint built per customer seed and linear; the RunPod tuning
    #: serves it); ``v2``: the bit-packed v1/v2 wire (:mod:`pgw.service.maskgen_gpu`)
    maskgen_wire: Literal["v2", "v3"] = "v3"
    maskgen_cols: int = 1024  #: columns per maskgen item (256 per group; the SDK's 1024-slot segments: 4 groups)
    maskgen_b_bits: int = 53  #: wire v2 upload bits (the default)
    maskgen_y_bits: int = 35
    maskgen_part_bytes: int = 512 * MIB
    context_bytes: int = 512 * MIB  #: CUDA context, cuBLAS/cuBLASLt handles, Triton modules
    cublas_workspace_bytes: int = 32 * MIB  #: per stream that runs GEMMs
    fragmentation: float = 0.03  #: caching-allocator slack on non-weight memory
    rev: Rev = "current"


@dataclass
class DevicePlan:
    """One device's memory plan (bytes)."""

    device: int
    steady: dict[str, int] = field(default_factory=dict)
    load: dict[str, int] = field(default_factory=dict)
    serving: dict[str, int] = field(default_factory=dict)

    @property
    def steady_bytes(self) -> int:
        return sum(self.steady.values())

    @property
    def peak_load(self) -> int:
        return self.steady_bytes + max(self.load.values(), default=0)

    @property
    def peak_serving(self) -> int:
        return self.steady_bytes + sum(self.serving.values())

    @property
    def peak(self) -> int:
        return max(self.peak_load, self.peak_serving)


@dataclass
class MemPlan:
    """The plan of one model on a set of identical GPUs."""

    model: str
    gpu: GpuSpec
    config: MemConfig
    devices: list[DevicePlan]

    @property
    def peak(self) -> int:
        return max(d.peak for d in self.devices)

    @property
    def headroom(self) -> int:
        return self.gpu.total_bytes - self.peak

    @property
    def fits(self) -> bool:
        return self.headroom >= 0

    def summary(self) -> dict[str, object]:
        d = max(self.devices, key=lambda x: x.peak)
        return {
            "model": self.model, "gpu": self.gpu.name, "gpus": len(self.devices), "rev": self.config.rev,
            "tp": self.config.tp, "pipeline": self.config.pipeline, "masked_head": self.config.masked_head,
            "weights_gib": round(d.steady.get("weights", 0) / GIB, 2),
            "steady_gib": round(d.steady_bytes / GIB, 2), "peak_load_gib": round(d.peak_load / GIB, 2),
            "peak_serving_gib": round(d.peak_serving / GIB, 2), "peak_gib": round(d.peak / GIB, 2),
            "capacity_gib": round(self.gpu.total_bytes / GIB, 2), "headroom_gib": round(self.headroom / GIB, 2),
            "fits": self.fits,
            "steady": {k: round(v / GIB, 3) for k, v in d.steady.items()},
            "load": {k: round(v / GIB, 3) for k, v in d.load.items()},
            "serving": {k: round(v / GIB, 3) for k, v in d.serving.items()},
        }


# ---- transient models -------------------------------------------------------------------------------------
def _legacy_core_bytes(k: int, b: int, n: int, np_: int, mp: int) -> int:
    """Peak of the pre-2026-11-26 offset-byte combine (three ``(K/8, b, mp)`` int64 temporaries at once)."""
    nb = k // 8
    rows, p = nb * b, offset_byte_rows(k, b)
    # the planes and the digit operand stayed referenced to the end; the int64 planes ``y`` too
    return b * n * nb + p * np_ + max(12 * p * mp, 24 * rows * mp, 8 * rows * mp + 24 * b * mp)


def _compact_bytes(k: int, b: int, mp: int, m: int) -> int:
    if k == 32:
        return 8 * b * mp + 16 * b * m
    if k < 32:
        return 8 * b * mp + 4 * b * m
    return 8 * b * mp + (8 * b * m if m != mp else 0)


def _eager_bytes(k: int, b: int, n: int, np_: int, mp: int, m: int, legacy: bool) -> int:
    core = _legacy_core_bytes(k, b, n, np_, mp) if legacy else offset_byte_work_bytes(k, b, n, np_, mp)
    return 8 * b * n + max(core, _compact_bytes(k, b, mp, m))


def _graph_body_bytes(k: int, p: int, n: int, np_: int, mp: int, m: int, legacy: bool) -> int:
    b = p // (k // 8)
    core = _legacy_core_bytes(k, b, n, np_, mp) if legacy else offset_byte_work_bytes(k, b, n, np_, mp, p)
    return max(core, _compact_bytes(k, b, mp, m))


def _mulmod64_bytes(m: int, n: int, cols: int, nb: int, part_bytes: int, legacy: bool) -> int:
    """Device bytes of :meth:`ComputeKernel.mulmod64` (the v3 GEMM and hint build): the uploaded byte planes, the
    ``(m, cols, 8)`` result (held whole only up to ``MASKGEN_OUT_DEVICE_BYTES`` since 2026-11-26), one column
    block's planes, int32 partials and int64 combine."""
    from pgw.service.kernels import MASKGEN_OUT_DEVICE_BYTES, _maskgen_block_cols

    mp, np_ = padded_shape(m, n)
    step = _maskgen_block_cols(mp, nb, cols, part_bytes, 32)
    cb = min(step, cols)
    cbp = -(-cb // 32) * 32
    block = nb * cbp * np_ + 4 * mp * nb * cbp + 2 * 8 * m * cb
    whole = 8 * m * cols
    held = whole if legacy or whole <= MASKGEN_OUT_DEVICE_BYTES else 8 * m * cb
    return n * cols * nb + held + block


def _maskgen_v3_bytes(m: int, n: int, cfg: MemConfig, legacy: bool) -> int:
    """One pks-v3 item (:class:`pgw.service.pks_provider.PKSProvider`): the hint build for a new customer seed
    (a 4096-column GEMM, once per seed and linear) or the item's GEMM, then the device key switch (row blocks of
    :data:`pgw.service.pks_gpu.WORK_BYTES`; unblocked before 2026-11-26)."""
    from pgw.hemask.pks import NU, PACK, D
    from pgw.service.pks_gpu import WORK_BYTES, _block_bytes

    groups = max(1, cfg.maskgen_cols // 256)
    gemm = max(_mulmod64_bytes(m, n, NU, 7, MASKGEN_PART_BYTES_CUDA, legacy),
               _mulmod64_bytes(m, n, cfg.maskgen_cols, 7, MASKGEN_PART_BYTES_CUDA, legacy))
    blocks = m // PACK
    ks = blocks * _block_bytes(groups) if legacy else min(blocks, max(1, WORK_BYTES // _block_bytes(groups))) \
        * _block_bytes(groups)
    return max(gemm, ks + 8 * D * groups * PACK * min(blocks, max(1, WORK_BYTES // _block_bytes(groups))))


def _maskgen_item_bytes(m: int, n: int, cfg: MemConfig, legacy: bool) -> int:
    """Device bytes of one maskgen item of ``cfg.maskgen_cols`` columns (v2: :mod:`pgw.service.maskgen_gpu`)."""
    if cfg.maskgen_wire == "v3":
        return _maskgen_v3_bytes(m, n, cfg, legacy)
    from pgw.service.kernels import _maskgen_block_cols

    cols, bb, yb = cfg.maskgen_cols, cfg.maskgen_b_bits, cfg.maskgen_y_bits
    mp, np_ = padded_shape(m, n)
    nb = -(-bb // 8)
    count = n * cols
    data_in = count * bb // 8
    data_out = m * cols * yb // 8
    t = bb % 8
    vals = count if t else 0
    if not t:
        tail = 0
    elif legacy:
        tail = count * (t + 17)
    else:
        groups = min(count // 8, 1 << 20)
        tail = groups * (8 * t + 8 + 64 + 64 + 8)
    step = _maskgen_block_cols(mp, nb, cols, cfg.maskgen_part_bytes, 32)
    cb = min(step, cols)
    cbp = -(-cb // 32) * 32
    planes, part, y = nb * cbp * np_, 4 * mp * nb * cbp, 8 * m * cb
    block = max(planes + part, part + y, 3 * y)
    if legacy:  # the previous block's planes, partials and result stayed referenced while the next allocated
        block = planes + part + y + max(planes + part, part + y, 3 * y)
    return data_in + data_out + vals + max(tail, block)


# ---- the plan ---------------------------------------------------------------------------------------------
def _device_lists(linears: Sequence[tuple[str, int, int]], cfg: MemConfig) -> list[list[tuple[str, int, int]]]:
    """Per device, the ``(key, m_shard, n)`` it holds."""
    if cfg.tp > 1 and cfg.pipeline > 1:
        raise ValueError("tensor parallel and by-layer placement do not combine")
    if cfg.tp > 1:
        from pgw.service.tensor_parallel import shard_bounds

        devs: list[list[tuple[str, int, int]]] = [[] for _ in range(cfg.tp)]
        for key, m, n in linears:
            if key.endswith(".head"):  # the masked head is one ComputeKernel on the group's first device
                devs[0].append((key, m, n))
                continue
            for j, (m0, m1) in enumerate(shard_bounds(m, cfg.tp)):
                devs[j].append((key, m1 - m0, n))
        return devs
    if cfg.pipeline > 1:
        from pgw.service.registry import place_layers

        layer_of = [int(k.split(".")[0]) for k, _, _ in linears]
        nlayers = max(layer_of) + 1
        lb = [0] * nlayers
        for (_, m, n), li in zip(linears, layer_of, strict=True):
            lb[li] += m * n
        placement = place_layers(lb, [f"cuda:{i}" for i in range(cfg.pipeline)])
        devs = [[] for _ in range(cfg.pipeline)]
        for lin, li in zip(linears, layer_of, strict=True):
            devs[int(str(placement[li]).split(":")[1])].append(lin)
        return devs
    return [list(linears)]


def plan(model: str, linears: Sequence[tuple[str, int, int]], gpu: GpuSpec, cfg: MemConfig | None = None) -> MemPlan:
    """Memory plan of serving ``linears`` (``(key, m, n)``) on ``gpu`` (``cfg.tp`` / ``cfg.pipeline`` of them)."""
    cfg = cfg or MemConfig()
    legacy = cfg.rev != "current"
    plans = []
    ks = sorted(set(cfg.rings_k))
    warm_k = cfg.rings_k[0]
    gemv_rows = gpu.gemv_max_rows
    for di, lins in enumerate(_device_lists(linears, cfg)):
        dp = DevicePlan(di)
        if not lins:
            plans.append(dp)
            continue
        pads = [(m, n, *padded_shape(m, n)) for _, m, n in lins]
        weights = sum(mp * np_ for _, _, mp, np_ in pads)
        n_max = max(n for _, n, _, _ in pads)
        m_max = max(m for m, _, _, _ in pads)
        st = dp.steady
        st["weights"] = weights * (2 if cfg.rev == "run1" else 1)  # run1 kept the unpadded copy too
        st["rowsums"] = sum(8 * mp for _, _, mp, _ in pads)
        if cfg.maskgen:
            st["maskgen_consts"] = sum(2 * 8 * m for m, _, _, _ in pads)
        streams = 2  # the default stream and the executor's graph stream
        if cfg.cuda_graphs and cfg.graph_buckets:
            bcap = max(cfg.graph_buckets) // 4
            st["graph_arenas"] = bcap * n_max * 8 + bcap * m_max * 8
            # graphs are captured lazily for any served K whose rows the fused GEMV does not take (warm-up
            # captures the first ring's); they share one pool sized by the largest body
            pool = 0
            for k in ks if not legacy else [warm_k, *ks]:
                for p in cfg.graph_buckets:
                    if p // (k // 8) <= (gemv_rows if k <= 32 else 0):
                        continue
                    for m, n, mp, np_ in pads:
                        pool = max(pool, _graph_body_bytes(k, p, n, np_, mp, m, legacy))
            st["graph_pool"] = pool
        if cfg.lane:
            rmax = min(cfg.lane_max_rows, gemv_rows)
            st["lane_buffers"] = rmax * n_max * 4 + rmax * m_max * 4
            streams += 1
        if cfg.persistent:
            st["persistent"] = 4 * n_max * 4 + 4 * m_max * 4 + 256
        if cfg.maskgen:
            streams += cfg.maskgen_concurrency
        st["context_workspaces"] = cfg.context_bytes + streams * cfg.cublas_workspace_bytes
        non_weight = sum(v for kk, v in st.items() if kk != "weights")
        st["allocator_slack"] = int(cfg.fragmentation * non_weight)

        # ---- load / warm-up transients
        big_m, big_n, big_mp, big_np = max(pads, key=lambda t: t[2] * t[3])
        if cfg.rev == "run1":
            dp.load["rowsum_int64_weight_copy"] = 8 * big_mp * big_np
        elif cfg.rev == "bfee514":
            dp.load["upload_unpadded_copy"] = big_m * big_n
            dp.load["rowsum_chunk"] = max(8 * min(4096, mp) * np_ for _, _, mp, np_ in pads)
        else:
            dp.load["rowsum_chunk"] = 32 * MIB
        if not gpu.int8_tensor_cores_triton and di == 0:  # the stream variant's timing gate (T4 class)
            gate = 2 * 64 * MIB + (2 * 8 * 64 * MIB if legacy else 2 * 32 * MIB)
            dp.load["triton_timing_gate"] = gate

        # ---- serving transients (one eager call per device; maskgen items concurrently)
        eager = 0
        for k in ks:
            for m, n, mp, np_ in pads:
                b = cfg.max_batch_rows
                if not legacy:
                    b = chunk_rows_for(cfg.eager_work_bytes,
                                       lambda r, k=k, n=n, np_=np_, mp=mp, m=m: _eager_bytes(k, r, n, np_, mp, m, False),
                                       b)
                eager = max(eager, _eager_bytes(k, b, n, np_, mp, m, legacy))
        dp.serving["eager_call"] = eager
        if cfg.maskgen:
            item = max(_maskgen_item_bytes(m, n, cfg, legacy) for m, n, _, _ in pads)
            dp.serving["maskgen_items"] = cfg.maskgen_concurrency * item
            if cfg.rev == "bfee514" and not cfg.warmup:  # row sums for the maskgen constant via an int64 copy
                dp.serving["maskgen_const_int64_copy"] = 8 * big_mp * big_np
        plans.append(dp)
    return MemPlan(model, gpu, cfg, plans)


def plan_model(model: str, gpu: str, cfg: MemConfig | None = None) -> MemPlan:
    """:func:`plan` for a preset model (:data:`MODELS`) and GPU (:data:`GPUS`)."""
    cfg = cfg or MemConfig()
    return plan(model, MODELS[model].linears(cfg.masked_head), GPUS[gpu], cfg)


def plan_table(models: Sequence[str], gpus: Sequence[str], cfg: MemConfig | None = None,
               tps: Sequence[int] = (1,)) -> str:
    """Markdown table: peak GiB per device (load / serving), headroom, fits."""
    cfg = cfg or MemConfig()
    rows = ["| model | GPU | TP | weights/dev | steady | peak load | peak serving | capacity | headroom | fits |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for model in models:
        for g in gpus:
            for tp in tps:
                s = plan_model(model, g, replace(cfg, tp=tp)).summary()
                rows.append(f"| {model} | {g} | {tp} | {s['weights_gib']} | {s['steady_gib']} | {s['peak_load_gib']} "
                            f"| {s['peak_serving_gib']} | {s['capacity_gib']} | {s['headroom_gib']} "
                            f"| {'yes' if s['fits'] else '**no**'} |")
    return "\n".join(rows)


def gpu_spec_of(device: object) -> GpuSpec:  # pragma: no cover - GPU only
    """The :class:`GpuSpec` of a live CUDA device (its reported capacity; Triton's int8 dot needs cc >= 8.0)."""
    import torch

    props = torch.cuda.get_device_properties(device)  # type: ignore[arg-type]
    tc = int(props.major) >= 8
    return GpuSpec(str(props.name), int(props.total_memory), tc, 16 if tc else 1, "live device")


def check_fit(model: str, linears: Sequence[tuple[str, int, int]], gpu: GpuSpec, cfg: MemConfig,
              resident: Sequence[int] = ()) -> str | None:
    """``None`` if ``linears`` fit on the device(s) next to ``resident`` bytes already allocated per device
    (index-aligned with the plan's devices), else a one-line explanation with the planned breakdown."""
    p = plan(model, linears, gpu, cfg)
    worst = None
    for d in p.devices:
        extra = resident[d.device] if d.device < len(resident) else 0
        over = d.peak + extra - gpu.total_bytes
        if over > 0 and (worst is None or over > worst[0]):
            worst = (over, d, extra)
    if worst is None:
        return None
    over, d, extra = worst
    parts = ", ".join(f"{k} {v / GIB:.2f}" for k, v in {**d.steady, **d.load, **d.serving}.items() if v >= 0.05 * GIB)
    return (f"model {model} is planned to need {(d.peak + extra) / GIB:.2f} GiB on device {d.device} "
            f"({gpu.name}, {gpu.total_bytes / GIB:.2f} GiB; already allocated {extra / GIB:.2f} GiB): "
            f"{over / GIB:.2f} GiB short [GiB: {parts}]; see python -m pgw.service.memplan")


def max_fitting(model: str, gpu: str, cfg: MemConfig, field_name: str, candidates: Sequence[int]) -> int | None:
    """The largest of ``candidates`` for ``cfg.<field_name>`` with which the plan fits (``None``: none)."""
    best = None
    for c in sorted(candidates):
        if plan_model(model, gpu, replace(cfg, **{field_name: c})).fits:
            best = c
    return best


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m pgw.service.memplan", description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", action="append", help=f"one of {sorted(MODELS)} (repeat; default: all)")
    ap.add_argument("--gpu", action="append", help=f"one of {sorted(GPUS)} (repeat; default: all)")
    ap.add_argument("--tp", default="1", help="tensor-parallel degrees, comma separated")
    ap.add_argument("--pipeline", type=int, default=1, help="by-layer placement over this many GPUs")
    ap.add_argument("--rev", default="current", choices=["current", "bfee514", "run1"])
    ap.add_argument("--no-masked-head", action="store_true")
    ap.add_argument("--no-lane", action="store_true")
    ap.add_argument("--persistent", action="store_true")
    ap.add_argument("--max-batch-rows", type=int, default=4096)
    ap.add_argument("--eager-work-mb", type=int, default=EAGER_WORK_BYTES >> 20)
    ap.add_argument("--maskgen-concurrency", type=int, default=2)
    ap.add_argument("--maskgen-cols", type=int, default=1024)
    ap.add_argument("--maskgen-wire", default="v3", choices=["v2", "v3"])
    ap.add_argument("--json", action="store_true", help="one JSON object per (model, GPU, TP) with the breakdown")
    a = ap.parse_args(argv)
    cfg = MemConfig(masked_head=not a.no_masked_head, lane=not a.no_lane, persistent=a.persistent,
                    max_batch_rows=a.max_batch_rows, eager_work_bytes=a.eager_work_mb << 20,
                    maskgen_concurrency=a.maskgen_concurrency, maskgen_cols=a.maskgen_cols, rev=a.rev,
                    maskgen_wire=a.maskgen_wire,
                    pipeline=a.pipeline)
    models = a.model or list(MODELS)
    gpus = a.gpu or list(GPUS)
    tps = [int(t) for t in a.tp.split(",")]
    for m in models + gpus:
        if m not in MODELS and m not in GPUS:
            ap.error(f"unknown model or GPU {m!r}")
    if a.json:
        for model in models:
            for g in gpus:
                for tp in tps:
                    s = plan_model(model, g, replace(cfg, tp=tp)).summary()
                    print(json.dumps(s))
        return 0
    print(plan_table(models, gpus, cfg, tps))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
