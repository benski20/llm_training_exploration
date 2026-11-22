"""Model registry: which packs are served, their prepared weights, and device placement.

The registry is keyed by **fingerprint**; the wire protocol never trusts model names. Packs are
opened with ``components=("linears",)`` (the service never needs plain tensors or tokenizers) and,
by default, ``verify="full"``, so every linear shard is hashed against the manifest at startup.

Multi-GPU placement (``placement="by_layer"``): layers are split into contiguous ranges whose
weight bytes are balanced across the configured devices. All four linears of one layer live on
the same device, so the per-device batch workers of :class:`pgw.service.BatchScheduler` naturally
pipeline different layers of different sessions in parallel.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import torch

from pgw.core.errors import (
    ConfigError,
    FingerprintMismatchError,
    ModelNotFoundError,
    PGWError,
    ProtocolError,
)
from pgw.core.fingerprint import short_fingerprint
from pgw.core.ring import RingParams
from pgw.core.types import LinearId, LinearKind, LinearSpec
from pgw.service.gpu_exec import GRAPH_BUCKETS, validate_buckets
from pgw.service.kernels import ComputeKernel, resolve_backend, resolve_device
from pgw.service.tensor_parallel import ShardedKernel, TPGroup

__all__ = ["ModelRegistry", "RegisteredModel", "place_layers"]

log = logging.getLogger(__name__)


def _device_check_operands() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(w8, x, expected)`` for the readiness product: a 64x64 weight (shapes like real linears;
    cuBLASLt int8 GEMMs on some GPUs reject tiny ones), extreme digits, exact Python-int reference."""
    i = torch.arange(64).view(-1, 1)
    j = torch.arange(64).view(1, -1)
    w = (((i * 31 + j * 17) % 255) - 127).to(torch.int8)
    w[0, :8] = torch.tensor([1, -2, 3, -128, 127, 0, 5, 7], dtype=torch.int8)
    x = ((i[:2] * 1_000_003 + j * 7_919_000_017) % (1 << 40)).to(torch.int64)
    x[0, :8] = torch.tensor([(1 << 39) + 3, 2, 1, 1, 1, 9, 0, 1])
    x[0, 8:] = 0
    xa = [[int(v) for v in row] for row in x.tolist()]
    wa = [[int(v) for v in row] for row in w.tolist()]
    exp = [[sum(a * b for a, b in zip(xr, wr, strict=True)) % (1 << 40) for wr in wa] for xr in xa]
    return w, x, torch.tensor(exp, dtype=torch.int64)


@dataclass
class RegisteredModel:
    """A loaded pack: name, fingerprint, linear specs and a :class:`ComputeKernel` per linear.

    Attributes:
        name: Served model name (from the manifest; informational only).
        fingerprint: Pack fingerprint (64 hex chars); the lookup key.
        specs: Linear specs in canonical order.
        kernels: ``{linear_id: ComputeKernel}``.
        rings: Ring names served for this model (configured rings with enough headroom).
        pack_path: Pack directory.
        placement: ``{layer: device string}``.
        loaded_at: Unix time of loading.
    """

    name: str
    fingerprint: str
    specs: Sequence[LinearSpec]
    kernels: dict[str, ComputeKernel | ShardedKernel]
    rings: tuple[str, ...] = ()
    pack_path: Path | None = None
    placement: dict[int, str] = field(default_factory=dict)
    loaded_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        self._dims: dict[str, tuple[int, int]] = {s.id.key: (s.out_features, s.in_features) for s in self.specs}

    @property
    def short_fingerprint(self) -> str:
        """Display form (12 hex chars)."""
        return short_fingerprint(self.fingerprint)

    @property
    def linear_dims(self) -> dict[str, tuple[int, int]]:
        """``{linear_id: (m, n)}``."""
        return self._dims

    @property
    def device_bytes(self) -> int:
        """Device memory held by all kernels."""
        return sum(k.device_bytes for k in self.kernels.values())

    def kernel(self, linear: str) -> ComputeKernel:
        """Kernel for ``linear``; raises :class:`ProtocolError` for an unknown linear id.

        With tensor parallelism this is a :class:`~pgw.service.tensor_parallel.ShardedKernel`, which implements
        the interface the service uses (duck typed; typed as :class:`ComputeKernel` for the callers).
        """
        try:
            return cast(ComputeKernel, self.kernels[linear])
        except KeyError as exc:
            raise ProtocolError(f"unknown linear {linear!r}", details={"linear": linear}) from exc

    def serves_ring(self, ring: str | RingParams) -> bool:
        """True if ``ring`` is served for this model."""
        name = ring.name if isinstance(ring, RingParams) else RingParams.from_name(ring).name
        return name in self.rings

    def devices(self) -> list[str]:
        """Distinct devices used by this model (every shard's device for tensor-parallel kernels)."""
        return sorted({str(d) for k in self.kernels.values() for d in getattr(k, "devices", [k.device])})


def place_layers(layer_bytes: Sequence[int], devices: Sequence[str]) -> dict[int, str]:
    """Assign contiguous layer ranges to devices, balancing bytes greedily.

    Args:
        layer_bytes: Weight bytes per layer (index = layer).
        devices: Target devices (at least one).

    Returns:
        ``{layer: device}``; layers keep their order, every device gets a contiguous range, and
        no device is left empty while another holds more than one layer.
    """
    if not devices:
        raise ConfigError("placement needs at least one device")
    n_layers = len(layer_bytes)
    n_dev = min(len(devices), max(1, n_layers))
    total = float(sum(layer_bytes)) or 1.0
    out: dict[int, str] = {}
    dev_i = 0
    acc = 0.0
    for layer, b in enumerate(layer_bytes):
        remaining_layers = n_layers - layer
        remaining_devs = n_dev - dev_i
        # Move on when this device has its share, or when the remaining devices need one layer each.
        if dev_i < n_dev - 1 and acc > 0 and (
            acc + b / 2 > total * (dev_i + 1) / n_dev or remaining_layers <= remaining_devs - 1
        ):
            dev_i += 1
        out[layer] = devices[dev_i]
        acc += b
    return out


class ModelRegistry:
    """Loads model packs (linears only, ``verify="full"`` at startup) onto the configured devices.

    Lookup is by fingerprint (the wire protocol never trusts model names). Thread-safe.

    Args:
        device: Single device (``auto``, ``cpu``, ``cuda``, ``cuda:i``); used when ``devices`` is empty.
        backend: Kernel backend (see :mod:`pgw.service.kernels`).
        devices: Several devices for multi-GPU placement.
        placement: ``"by_layer"`` (contiguous layer ranges balanced by bytes) or ``"single"``
            (everything on the first device).
        rings: Ring names this service is willing to serve (each pack serves the subset with
            enough headroom).
        verify: Pack verification mode at load time (``"full"`` or ``"fingerprint"``).
        cache_float: Keep float weight copies for CPU float backends.
        warmup: Run one dummy product per linear after loading (on CUDA this also captures
            the CUDA graphs of every row bucket for the first served ring).
        cuda_graphs: CUDA kernels replay captured graphs for small calls (see :class:`ComputeKernel`).
        cuda_graph_buckets: Padded GEMM row counts with a graph.
        triton_gemv: CUDA kernels serve small ``K <= 32`` calls with the fused Triton ring GEMV once
            its device self-check passed (see :class:`ComputeKernel`).
        tp_devices: Output-row tensor parallel (opt-in; :mod:`pgw.service.tensor_parallel`): every linear
            is split by output rows over these devices (in this order) and served by a
            :class:`~pgw.service.tensor_parallel.ShardedKernel`; ``devices`` and ``placement`` are then
            ignored. CPU may repeat (fake multi-device for tests). Empty (default): off.
    """

    def __init__(
        self,
        device: str = "auto",
        backend: str = "auto",
        *,
        devices: Sequence[str] | None = None,
        placement: str = "by_layer",
        rings: Sequence[str] = ("K32A12C27", "K40A12", "K48A20", "K32A8"),
        verify: str = "full",
        cache_float: bool = True,
        warmup: bool = False,
        cuda_graphs: bool = True,
        cuda_graph_buckets: Sequence[int] = GRAPH_BUCKETS,
        triton_gemv: bool = True,
        tp_devices: Sequence[str] | None = None,
    ):
        if placement not in ("single", "by_layer"):
            raise ConfigError(f"unknown placement {placement!r}; use 'single' or 'by_layer'")
        if verify not in ("full", "fingerprint"):
            raise ConfigError(f"unknown verify mode {verify!r}; use 'full' or 'fingerprint'")
        self.tp_group: TPGroup | None = TPGroup(tp_devices) if tp_devices else None
        devs = [str(d) for d in self.tp_group.devices] if self.tp_group is not None else (
            list(devices) if devices else [device])
        resolved = [resolve_device(d) for d in devs]
        uniq: list[torch.device] = []
        for d in resolved:
            if d not in uniq:
                uniq.append(d)
        for d in uniq:  # fail fast on an impossible backend/device combination
            resolve_backend(d, backend)
        self.devices: list[torch.device] = uniq
        self.backend = backend
        self.placement = placement
        self.rings = tuple(RingParams.from_name(r).name for r in rings)
        if not self.rings:
            raise ConfigError("at least one ring must be served")
        self.verify = verify
        self.cache_float = cache_float
        self.warmup = warmup
        self.cuda_graphs = cuda_graphs
        self.cuda_graph_buckets = validate_buckets(cuda_graph_buckets)
        self.triton_gemv = triton_gemv
        self._models: dict[str, RegisteredModel] = {}
        self._lock = threading.RLock()

    # ---- loading -------------------------------------------------------------------------
    def load(self, pack_path: str | Path, *, expected_fingerprint: str | None = None) -> RegisteredModel:
        """Load and register a pack; returns the entry (the existing one if already loaded).

        Raises:
            ConfigError: missing/invalid pack or no configured ring has enough headroom.
            FingerprintMismatchError: content does not match the manifest / ``expected_fingerprint``.
        """
        from pgw.modelpack import ModelPack

        t0 = time.perf_counter()
        pack = ModelPack.open(
            pack_path, verify=self.verify,  # type: ignore[arg-type]
            expected_fingerprint=expected_fingerprint, components=("linears",),
        )
        try:
            fp = pack.fingerprint
            with self._lock:
                if fp in self._models:
                    log.warning("model pack %s (fp=%s) is already loaded; skipping", pack_path, short_fingerprint(fp))
                    return self._models[fp]
            ok = pack.supported_rings([RingParams.from_name(x) for x in self.rings])
            rings = tuple(r for r in self.rings if r in ok)
            if not rings:
                raise ConfigError(
                    f"model pack {pack.name}: none of the configured rings {list(self.rings)} has enough headroom "
                    f"for inner dimension {pack.max_in_features()}",
                    details={"rings": list(self.rings), "max_in_features": pack.max_in_features()},
                )
            specs = pack.linear_specs()
            placement = self._placement(specs, pack.num_layers)
            kernels: dict[str, ComputeKernel | ShardedKernel] = {}
            for spec in specs:
                lw = pack.load_linear(spec.id)
                if self.tp_group is not None:
                    kernels[spec.id.key] = ShardedKernel(
                        spec, lw.w8, self.tp_group, backend=self.backend, cache_float=self.cache_float,
                        cuda_graphs=self.cuda_graphs, graph_buckets=self.cuda_graph_buckets,
                        triton_gemv=self.triton_gemv,
                    )
                else:
                    kernels[spec.id.key] = ComputeKernel(
                        spec, lw.w8, device=placement[spec.id.layer], backend=self.backend,
                        cache_float=self.cache_float, cuda_graphs=self.cuda_graphs,
                        graph_buckets=self.cuda_graph_buckets, triton_gemv=self.triton_gemv,
                    )
                del lw
            entry = RegisteredModel(
                name=pack.name, fingerprint=fp, specs=tuple(specs), kernels=kernels, rings=rings,
                pack_path=Path(pack_path), placement=placement,
            )
        finally:
            pack.close()
        if self.warmup:
            ring = RingParams.from_name(entry.rings[0])
            for k in entry.kernels.values():
                k.warmup(ring, all_buckets=True)
        with self._lock:
            existing = self._models.get(fp)
            if existing is not None:  # lost a race with a concurrent load
                return existing
            self._models[fp] = entry
        log.info(
            "loaded model %s fp=%s linears=%d rings=%s devices=%s backend=%s bytes=%d in %.2fs",
            entry.name, entry.short_fingerprint, len(entry.kernels), ",".join(entry.rings),
            ",".join(entry.devices()), next(iter(entry.kernels.values())).backend if entry.kernels else "-",
            entry.device_bytes, time.perf_counter() - t0,
        )
        return entry

    def register(self, entry: RegisteredModel) -> RegisteredModel:
        """Register an already-built entry (tests, programmatic use). Replaces nothing.

        Raises:
            ConfigError: an entry with the same fingerprint exists.
        """
        with self._lock:
            if entry.fingerprint in self._models:
                raise ConfigError(f"model fingerprint {entry.short_fingerprint} is already registered")
            self._models[entry.fingerprint] = entry
        return entry

    def _placement(self, specs: Sequence[LinearSpec], num_layers: int) -> dict[int, str]:
        devs = [str(d) for d in self.devices]
        if self.tp_group is not None:  # every layer on the whole group (the first device is its scheduling key)
            return {i: str(self.tp_group.devices[0]) for i in range(num_layers)}
        if self.placement == "single" or len(devs) == 1:
            return {i: devs[0] for i in range(num_layers)}
        layer_bytes = [0] * num_layers
        for s in specs:
            layer_bytes[s.id.layer] += s.out_features * s.in_features
        return place_layers(layer_bytes, devs)

    def unload(self, fingerprint: str) -> None:
        """Remove a model (in-flight batches holding kernel references finish normally)."""
        with self._lock:
            if self._models.pop(fingerprint, None) is None:
                raise ModelNotFoundError(f"model {short_fingerprint(fingerprint)} is not loaded")
        log.info("unloaded model fp=%s", short_fingerprint(fingerprint))

    # ---- lookup --------------------------------------------------------------------------
    def get(self, fingerprint: str) -> RegisteredModel:
        """Entry by fingerprint; raises :class:`pgw.core.ModelNotFoundError`."""
        with self._lock:
            m = self._models.get(fingerprint)
        if m is None:
            raise ModelNotFoundError(
                f"model fingerprint {short_fingerprint(fingerprint) if fingerprint else '<empty>'} is not served",
                details={"fingerprint": fingerprint},
            )
        return m

    def models(self) -> list[RegisteredModel]:
        """All registered models (load order)."""
        with self._lock:
            return list(self._models.values())

    def __len__(self) -> int:
        with self._lock:
            return len(self._models)

    def __contains__(self, fingerprint: object) -> bool:
        with self._lock:
            return fingerprint in self._models

    def linear_dims(self, fingerprint: str, ring: str) -> Mapping[str, tuple[int, int]]:
        """Dims of a served model for a served ring.

        Raises:
            FingerprintMismatchError: fingerprint not served.
            ProtocolError: malformed ring name.
            ConfigError: ring not served for this model.
        """
        try:
            m = self.get(fingerprint)
        except ModelNotFoundError as exc:
            served = [x.fingerprint for x in self.models()]
            raise FingerprintMismatchError(exc.message, details={"served": served}) from exc
        try:
            rname = RingParams.from_name(ring).name
        except PGWError as exc:
            raise ProtocolError(f"invalid ring {ring!r}", details={"ring": ring}) from exc
        if rname not in m.rings:
            raise ConfigError(
                f"ring {rname} is not served for model {m.short_fingerprint}",
                details={"ring": rname, "rings": list(m.rings)},
            )
        return m.linear_dims

    # ---- health --------------------------------------------------------------------------
    def check_devices(self) -> dict[str, bool]:
        """Run a tiny exact product on every device; ``{device: ok}`` (used by ``/readyz``)."""
        out: dict[str, bool] = {}
        w, x, expected = _device_check_operands()
        ring = RingParams(40, 12)
        spec = LinearSpec(LinearId(0, LinearKind.O), int(w.shape[0]), int(w.shape[1]))
        for d in self.devices:
            try:
                # no graphs: a throwaway kernel per check (eager path, same executor and staging)
                k = ComputeKernel(spec, w, device=d, backend=self.backend, cache_float=False, cuda_graphs=False)
                out[str(d)] = bool(torch.equal(k.run(x, ring), expected))
                if not out[str(d)]:
                    log.error("device check on %s: wrong ring product", d)
            except Exception as exc:  # report, never raise from a health check
                log.warning("device check failed on %s: %s: %s", d, type(exc).__name__, str(exc)[:300])
                out[str(d)] = False
        return out

    def device_summary(self) -> list[dict[str, object]]:
        """Per-device bytes and linear counts (``/readyz`` details, logs)."""
        acc: dict[str, dict[str, object]] = {str(d): {"device": str(d), "bytes": 0, "linears": 0} for d in self.devices}
        for m in self.models():
            for k in m.kernels.values():
                for part in getattr(k, "shards", [k]):
                    e = acc.setdefault(str(part.device), {"device": str(part.device), "bytes": 0, "linears": 0})
                    e["bytes"] = int(e["bytes"]) + part.device_bytes  # type: ignore[call-overload]
                    e["linears"] = int(e["linears"]) + 1  # type: ignore[call-overload]
        return list(acc.values())
