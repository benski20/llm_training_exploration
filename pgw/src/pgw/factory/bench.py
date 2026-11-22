"""Factory throughput benchmark and cost estimate (``pgw factory benchmark``).

Two modes:

* **pack**: time real segment generation (seed expansion + exact ring GEMMs + Freivalds
  precomputation; optionally encryption into a temporary store) for an actual model pack.
* **preset**: time one decoder layer of a well-known architecture with random int8 weights and
  scale by the number of layers. This estimates 8B/70B throughput on a box that does not hold the
  checkpoint (the work per layer is identical, so the extrapolation is accurate to a few percent).

Outputs slots (= tokens) per second and per hour, storage per token and dollars per million
tokens at a given hourly price. Per-segment work (the Freivalds ``u W`` precomputation) is
amortised over ``segment_slots`` (the production segment size), not over the timed slots.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass
from typing import Any

import torch

from pgw.core.errors import ConfigError
from pgw.core.prng import new_seed
from pgw.core.ring import RingParams
from pgw.core.settings import ring_from_setting
from pgw.core.types import LinearId, LinearKind, LinearSpec

__all__ = [
    "MODEL_PRESETS",
    "BenchmarkResult",
    "ModelShape",
    "benchmark_pack",
    "benchmark_preset",
    "default_price_per_hour",
    "format_report",
]

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelShape:
    """Dimensions needed to derive the four masked linear groups of a decoder layer."""

    name: str
    hidden: int
    intermediate: int
    heads: int
    kv_heads: int
    head_dim: int
    layers: int

    def layer_specs(self, layer: int = 0) -> list[LinearSpec]:
        """``qkv, o, gate_up, down`` specs of one layer."""
        q = self.heads * self.head_dim
        kv = self.kv_heads * self.head_dim
        return [
            LinearSpec(LinearId(layer, LinearKind.QKV), q + 2 * kv, self.hidden),
            LinearSpec(LinearId(layer, LinearKind.O), self.hidden, q),
            LinearSpec(LinearId(layer, LinearKind.GATE_UP), 2 * self.intermediate, self.hidden),
            LinearSpec(LinearId(layer, LinearKind.DOWN), self.hidden, self.intermediate),
        ]

    @property
    def params_per_layer(self) -> int:
        """Masked-linear parameters per layer."""
        return sum(s.out_features * s.in_features for s in self.layer_specs())


#: Public architecture shapes (from the models' published configs).
MODEL_PRESETS: dict[str, ModelShape] = {
    "qwen2.5-0.5b": ModelShape("qwen2.5-0.5b", 896, 4864, 14, 2, 64, 24),
    "qwen2.5-1.5b": ModelShape("qwen2.5-1.5b", 1536, 8960, 12, 2, 128, 28),
    "qwen2.5-7b": ModelShape("qwen2.5-7b", 3584, 18944, 28, 4, 128, 28),
    "qwen2.5-14b": ModelShape("qwen2.5-14b", 5120, 13824, 40, 8, 128, 48),
    "qwen2.5-32b": ModelShape("qwen2.5-32b", 5120, 27648, 40, 8, 128, 64),
    "qwen2.5-72b": ModelShape("qwen2.5-72b", 8192, 29568, 64, 8, 128, 80),
    "llama3-8b": ModelShape("llama3-8b", 4096, 14336, 32, 8, 128, 32),
    "llama3-70b": ModelShape("llama3-70b", 8192, 28672, 64, 8, 128, 80),
}


def default_price_per_hour(device: torch.device) -> float:
    """Rough on-demand/spot price assumption in USD per hour for the benchmarking host.

    CUDA: 0.20 (T4-class spot instance incl. host). CPU: 0.04 per torch thread (vCPU-hour, on demand).
    Always pass ``--price-per-hour`` for real decisions.
    """
    if device.type == "cuda":
        return 0.20
    return 0.04 * torch.get_num_threads()


@dataclass
class BenchmarkResult:
    """Benchmark numbers (all per slot = per token position, all layers)."""

    model: str
    mode: str
    ring: str
    device: str
    backend: str
    slots_measured: int
    seconds: float
    slots_per_second: float
    tokens_per_hour: float
    seconds_per_token: float
    gemm_fraction: float
    bytes_per_token: int
    price_per_hour: float
    usd_per_million_tokens: float
    storage_tb_per_million_tokens: float
    layers_measured: int
    layers_total: int
    gpu_name: str | None = None
    #: ``r`` expanded on the GPU (:mod:`pgw.factory.gpu_prng`; ``PGW_FACTORY_GPU_PRNG``)
    gpu_prng: bool = False
    #: ``r`` expanded on the CPU by the native ChaCha20 (``pgw_fast``) rather than the Python expander
    native_rng: bool = False
    #: GEMM batch used (``chunk_slots``; auto-sized when not given)
    chunk_slots: int = 0
    #: Per-stage seconds of the timed run (see :func:`pgw.factory.generator.format_stats`); rng and write
    #: overlap the GEMMs when pipelined
    rng_seconds: float = 0.0
    gemm_seconds: float = 0.0
    check_seconds: float = 0.0
    write_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable form."""
        return asdict(self)


def _finish(model: str, mode: str, ring: RingParams, device: torch.device, backend: str, slots: int, seconds: float,
            gemm: float, check: float, segment_slots: int, bytes_per_token: int, price: float | None,
            layers_measured: int, layers_total: int) -> BenchmarkResult:
    scale = layers_total / layers_measured
    # per-segment work (Freivalds precomputation) is amortised over a production-size segment
    per_slot = max(seconds - check, 0.0) / slots + check / max(segment_slots, slots)
    sec_per_tok = per_slot * scale
    sps = 1.0 / sec_per_tok if sec_per_tok > 0 else 0.0
    tph = sps * 3600
    p = default_price_per_hour(device) if price is None else float(price)
    usd = p / (tph / 1e6) if tph > 0 else float("inf")
    gpu = torch.cuda.get_device_name(device) if device.type == "cuda" else None
    return BenchmarkResult(
        model=model, mode=mode, ring=ring.name, device=str(device), backend=backend, slots_measured=slots,
        seconds=seconds, slots_per_second=sps, tokens_per_hour=tph, seconds_per_token=sec_per_tok,
        gemm_fraction=gemm / seconds if seconds > 0 else 0.0, bytes_per_token=bytes_per_token, price_per_hour=p,
        usd_per_million_tokens=usd, storage_tb_per_million_tokens=bytes_per_token * 1e6 / 1e12,
        layers_measured=layers_measured, layers_total=layers_total, gpu_name=gpu,
    )


def _stage_stats(res: BenchmarkResult, gen: Any) -> BenchmarkResult:
    st = gen.last_stats
    res.gpu_prng = bool(st.extra.get("gpu_prng", False))
    res.native_rng = bool(st.extra.get("native_rng", False))
    res.chunk_slots = int(st.extra.get("chunk_slots", 0))
    res.rng_seconds, res.gemm_seconds = st.rng_seconds, st.gemm_seconds
    res.check_seconds, res.write_seconds = st.check_seconds, st.write_seconds
    return res


class _SyntheticPack:
    """Minimal pack-like object with random int8 weights for one layer (preset benchmarks)."""

    def __init__(self, shape: ModelShape, layers: int, seed: int = 0):
        self.name = f"{shape.name}-synthetic"
        self.fingerprint = "0" * 64
        self._specs = [s for i in range(layers) for s in shape.layer_specs(i)]
        gen = torch.Generator().manual_seed(seed)
        self._w: dict[tuple[int, int], torch.Tensor] = {}
        for s in shape.layer_specs(0):  # weights are shared across layers (same shapes, same work)
            key = (s.out_features, s.in_features)
            if key not in self._w:
                self._w[key] = torch.randint(-127, 128, key, dtype=torch.int8, generator=gen)

    def linear_specs(self) -> list[LinearSpec]:
        return list(self._specs)

    def check_ring(self, params: RingParams) -> None:
        for s in self._specs:
            params.check_in_features(s.in_features)

    def load_linear(self, lid: LinearId | str, device: str | torch.device = "cpu") -> Any:
        from types import SimpleNamespace

        key = str(lid)
        spec = next(s for s in self._specs if s.id.key == key)
        return SimpleNamespace(w8=self._w[(spec.out_features, spec.in_features)].to(device))


def _time_generation(gen: Any, slots: int, chunk_slots: int | None, checks: int, store: Any | None,
                     warmup: bool) -> tuple[float, float, float]:
    """Return ``(seconds, gemm_seconds, check_seconds)`` for one segment."""
    def sync() -> None:
        if gen.device.type == "cuda":  # pragma: no cover - GPU only
            torch.cuda.synchronize(gen.device)

    chunk_slots = gen.chunk_slots_for(chunk_slots)
    if warmup:
        for _ in gen.generate(new_seed(), min(slots, chunk_slots), checks=checks, chunk_slots=chunk_slots):
            pass
    sync()
    t0 = time.perf_counter()
    if store is not None:
        gen.generate_into_store(store, slots, checks=checks, chunk_slots=chunk_slots, factory_id="benchmark")
    else:
        for item in gen.generate(new_seed(), slots, checks=checks, chunk_slots=chunk_slots):
            del item
    sync()
    return time.perf_counter() - t0, gen.last_stats.gemm_seconds, gen.last_stats.check_seconds


def benchmark_preset(
    preset: str,
    *,
    ring: str | RingParams = "K32A12C27",
    device: str = "auto",
    backend: str = "auto",
    slots: int = 256,
    chunk_slots: int | None = 256,
    checks: int = 2,
    layers: int = 1,
    segment_slots: int = 4096,
    price_per_hour: float | None = None,
    warmup: bool = True,
    store: Any | None = None,
) -> BenchmarkResult:
    """Benchmark ``layers`` synthetic layers of a preset architecture and extrapolate to the full model.

    ``chunk_slots=None`` uses the factory's auto size; ``store`` also times encryption and writing into it.
    """
    from pgw.factory.generator import MaskGenerator, slot_bytes

    shape = MODEL_PRESETS.get(preset.lower())
    if shape is None:
        raise ConfigError(f"unknown model preset {preset!r}; choose one of {sorted(MODEL_PRESETS)}")
    if not 1 <= layers <= shape.layers:
        raise ConfigError(f"layers must be in [1, {shape.layers}]")
    rp = ring_from_setting(ring)
    pack = _SyntheticPack(shape, layers)
    gen = MaskGenerator(pack, rp, device=device, backend=backend)
    seconds, gemm, check = _time_generation(gen, slots, chunk_slots, checks, store, warmup)
    full_specs = [s for i in range(shape.layers) for s in shape.layer_specs(i)]
    res = _finish(shape.name, "preset+store" if store is not None else "preset", rp, gen.device, gen.backend,
                  slots, seconds, gemm, check, segment_slots, slot_bytes(full_specs, rp), price_per_hour, layers,
                  shape.layers)
    return _stage_stats(res, gen)


def benchmark_pack(
    pack: Any,
    *,
    ring: str | RingParams = "K32A12C27",
    device: str = "auto",
    backend: str = "auto",
    slots: int = 256,
    chunk_slots: int | None = 256,
    checks: int = 2,
    store: Any | None = None,
    segment_slots: int = 4096,
    price_per_hour: float | None = None,
    warmup: bool = True,
) -> BenchmarkResult:
    """Benchmark full-segment generation for an opened pack (optionally writing into ``store``)."""
    from pgw.factory.generator import MaskGenerator

    rp = ring_from_setting(ring)
    gen = MaskGenerator(pack, rp, device=device, backend=backend)
    seconds, gemm, check = _time_generation(gen, slots, chunk_slots, checks, store, warmup)
    res = _finish(getattr(pack, "name", "pack"), "pack+store" if store is not None else "pack", rp, gen.device,
                  gen.backend, slots, seconds, gemm, check, segment_slots, gen.bytes_per_slot, price_per_hour, 1, 1)
    return _stage_stats(res, gen)


def format_report(r: BenchmarkResult) -> str:
    """Human-readable benchmark summary."""
    extrap = ""
    if r.layers_measured != r.layers_total:
        extrap = f" (measured {r.layers_measured}/{r.layers_total} layers, extrapolated)"
    dev = f"{r.device} [{r.gpu_name}]" if r.gpu_name else r.device
    rng_src = "gpu" if r.gpu_prng else ("native" if r.native_rng else "python")
    lines = [
        f"model            {r.model}{extrap}",
        f"ring / device    {r.ring} on {dev} via {r.backend}"
        f"{' (r expanded on the GPU)' if r.gpu_prng else ''}",
        f"measured         {r.slots_measured} slots in {r.seconds:.3f}s (GEMM {100 * r.gemm_fraction:.0f}%)",
        f"stages           rng {r.rng_seconds:.3f}s [{rng_src}], gemm {r.gemm_seconds:.3f}s, "
        f"check {r.check_seconds:.3f}s, write {r.write_seconds:.3f}s (chunk {r.chunk_slots} slots)",
        f"throughput       {r.slots_per_second:,.1f} tokens/s = {r.tokens_per_hour:,.0f} tokens/hour",
        f"per token        {1e3 * r.seconds_per_token:.3f} ms, {r.bytes_per_token / 1e6:.2f} MB of masks",
        f"cost             ${r.usd_per_million_tokens:,.3f} per million tokens at ${r.price_per_hour:.3f}/h",
        f"storage          {r.storage_tb_per_million_tokens:.2f} TB per million tokens",
    ]
    return "\n".join(lines)
