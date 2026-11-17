"""``pgw bench ring``: exact ring-matmul micro benchmark per backend.

Times ``y_hat = W8 . x_hat mod 2^K`` (the compute service's whole job) for a given shape, ring and
backend, checks the result against an exact reference on a few rows, and reports latency and
throughput. ``pgw bench factory`` and ``pgw bench service`` delegate to the factory benchmark and
the service load generator.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass
from typing import Any

import torch

from pgw.core.errors import ConfigError

__all__ = ["SHAPE_PRESETS", "RingBenchResult", "available_backends", "bench_ring", "format_ring_results"]

log = logging.getLogger(__name__)

#: ``(m, n)`` of representative linear groups (largest first is the decode bottleneck).
SHAPE_PRESETS: dict[str, tuple[int, int]] = {
    "tiny": (128, 64),
    "qwen2.5-0.5b.gate_up": (9728, 896),
    "qwen2.5-0.5b.down": (896, 4864),
    "llama3-8b.qkv": (6144, 4096),
    "llama3-8b.gate_up": (28672, 4096),
    "llama3-8b.down": (4096, 14336),
}


@dataclass
class RingBenchResult:
    """One (ring, backend, shape, rows) measurement."""

    ring: str
    backend: str
    device: str
    m: int
    n: int
    rows: int
    repeat: int
    exact: bool
    ms_per_call: float
    rows_per_s: float
    ring_macs_per_s: float

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable form."""
        return asdict(self)


def available_backends() -> list[str]:
    """Exact ring-matmul backends usable on this host."""
    from pgw.core.ringmm import cuda_int8_available

    out = ["cpu_f64", "cpu_f32"]
    try:
        from pgw.service.kernels import cpu_int8_available
    except ImportError:  # service extra not installed
        cpu_int8_available = None  # type: ignore[assignment]
    if cpu_int8_available is not None and cpu_int8_available():
        out.insert(0, "cpu_int8")
    if cuda_int8_available():
        out.insert(0, "cuda_int8")
    return out


def _kernel(w8: torch.Tensor, backend: str) -> tuple[Any, str]:
    """A callable ``f(x_hat, ring) -> y_hat`` for ``backend`` and the device it runs on."""
    device = "cuda" if backend == "cuda_int8" else "cpu"
    if backend == "cpu_int8":
        from pgw.core.types import LinearId, LinearSpec
        from pgw.service.kernels import ComputeKernel

        spec = LinearSpec(LinearId(0, "o"), int(w8.shape[0]), int(w8.shape[1]))  # type: ignore[arg-type]
        k = ComputeKernel(spec, w8, device="cpu", backend="cpu_int8", cache_float=False)
        return k.run, device
    from pgw.core.ringmm import PreparedWeight

    pw = PreparedWeight(w8, device=device, backend=backend)  # type: ignore[arg-type]
    return pw.matmul, device


def bench_ring(
    *,
    ring: str = "K40A12",
    backend: str = "cpu_f64",
    m: int = 1024,
    n: int = 1024,
    rows: int = 1,
    repeat: int = 5,
    warmup: int = 1,
    verify_rows: int = 2,
    seed: int = 0,
) -> RingBenchResult:
    """Benchmark one backend; ``exact`` compares ``verify_rows`` rows (first 64 outputs) against Python-integer math."""
    from pgw.core.ring import RingParams
    from pgw.core.ringmm import ringmm_reference

    if min(m, n, rows, repeat) <= 0 or warmup < 0:
        raise ConfigError("m, n, rows and repeat must be positive")
    params = RingParams.from_name(ring)
    params.check_in_features(n)
    if backend not in available_backends():
        raise ConfigError(f"backend {backend!r} is not available here (available: {', '.join(available_backends())})")
    g = torch.Generator().manual_seed(seed)
    w8 = torch.randint(-127, 128, (m, n), generator=g, dtype=torch.int8)
    x_hat = torch.randint(0, 1 << 62, (rows, n), generator=g, dtype=torch.int64) & params.mask
    fn, device = _kernel(w8, backend)
    xd = x_hat.to(device)

    def sync() -> None:
        if device == "cuda":
            torch.cuda.synchronize()

    y = None
    for _ in range(warmup):
        y = fn(xd, params)
    sync()
    t0 = time.perf_counter()
    for _ in range(repeat):
        y = fn(xd, params)
    sync()
    dt = (time.perf_counter() - t0) / repeat
    assert y is not None
    vr = min(verify_rows, rows)
    vc = min(m, 64)  # Python-integer reference is slow: check a corner of the output
    ref = ringmm_reference(x_hat[:vr].numpy(), w8[:vc].numpy(), params.k)
    got = y[:vr, :vc].cpu().numpy()
    exact = bool((ref.astype("int64") == got.astype("int64")).all())
    if not exact:
        log.error("ring bench: %s/%s result is NOT exact", ring, backend)
    return RingBenchResult(
        ring=params.name, backend=backend, device=device, m=m, n=n, rows=rows, repeat=repeat, exact=exact,
        ms_per_call=round(dt * 1e3, 3), rows_per_s=round(rows / dt, 1),
        ring_macs_per_s=float(f"{rows * m * n / dt:.4g}"),
    )


def format_ring_results(results: list[RingBenchResult]) -> str:
    """Fixed-width table."""
    head = f"{'ring':8s} {'backend':10s} {'m x n':>13s} {'rows':>6s} {'ms/call':>10s} {'rows/s':>11s} " \
           f"{'GMAC/s':>9s} exact"
    lines = [head]
    for r in results:
        lines.append(f"{r.ring:8s} {r.backend:10s} {f'{r.m}x{r.n}':>13s} {r.rows:6d} {r.ms_per_call:10.3f} "
                     f"{r.rows_per_s:11.1f} {r.ring_macs_per_s / 1e9:9.2f} {'yes' if r.exact else 'NO'}")
    return "\n".join(lines)
