"""Weight streaming for models larger than device memory (e.g. 70B on a 16 GB T4).

The factory processes one linear group at a time (segments are linear-major, SPEC 7.2), so it
never needs more than one or two weights on the device. :class:`WeightStreamer` keeps as many
linears permanently resident as ``resident_bytes`` allows (greedily, in canonical order) and
streams the rest from the memory-mapped pack on every pass:

* CUDA: host -> pinned staging buffer -> device copy on a dedicated copy stream, double-buffered,
  so the transfer of linear ``i+1`` overlaps the GEMMs of linear ``i``.
* CPU: weights are the pack's memory-mapped tensors (zero copy). The OS page cache does the
  streaming, so models larger than RAM work; resident linears are copied into anonymous memory.
"""

from __future__ import annotations

import itertools
import logging
from collections.abc import Iterator
from typing import Any

import torch

from pgw.core.errors import ConfigError
from pgw.core.types import LinearId, LinearSpec

__all__ = ["WeightStreamer", "auto_resident_bytes"]

log = logging.getLogger(__name__)


def auto_resident_bytes(device: torch.device, fraction: float, reserve_bytes: int = 0) -> int:
    """Budget for resident weights on ``device``.

    CUDA: ``fraction`` of total device memory minus ``reserve_bytes`` (GEMM workspace for the chunk
    buffers). CPU: 0 (memory-mapped pack tensors are already zero-copy).
    """
    if not 0 < fraction <= 1:
        raise ConfigError(f"memory fraction must be in (0, 1], got {fraction}")
    if device.type != "cuda":
        return 0
    total = int(torch.cuda.get_device_properties(device).total_memory)  # pragma: no cover - GPU only
    return max(0, int(total * fraction) - reserve_bytes)  # pragma: no cover - GPU only


class WeightStreamer:
    """Double-buffered host->device streaming of int8 linear weights.

    Keeps pinned host copies and overlaps the copy of linear ``i+1`` with the GEMMs of linear ``i``
    (measured: 70B at batch >= 1024 is GEMM-bound on a T4). Linears that fit within
    ``resident_bytes`` stay on the device permanently.

    Args:
        pack: Open :class:`pgw.modelpack.ModelPack` (only ``linears`` are needed).
        device: Target device.
        resident_bytes: Device bytes available for permanently resident weights. ``-1`` keeps
            everything resident (small models / tests).
    """

    def __init__(self, pack: Any, device: str | torch.device, resident_bytes: int):
        self.pack = pack
        self.device = torch.device(device)
        self.specs: list[LinearSpec] = list(pack.linear_specs())
        if not self.specs:
            raise ConfigError("model pack has no linear groups")
        total = sum(s.out_features * s.in_features for s in self.specs)
        self.total_bytes = total
        budget = total if resident_bytes < 0 else int(resident_bytes)
        self.resident_ids: set[str] = set()
        used = 0
        for s in self.specs:  # greedy in canonical order (deterministic)
            size = s.out_features * s.in_features
            if used + size <= budget:
                self.resident_ids.add(s.id.key)
                used += size
        self.resident_bytes_used = used
        self._resident: dict[str, torch.Tensor] = {}
        self.bytes_streamed = 0
        self._copy_stream: Any = None
        self._pinned: list[torch.Tensor] = []
        self._slot_events: dict[int, Any] = {}
        if self.device.type == "cuda":  # pragma: no cover - GPU only
            self._copy_stream = torch.cuda.Stream(device=self.device)  # type: ignore[no-untyped-call]
        log.info(
            "weight streamer: %d/%d linears resident (%.1f MiB of %.1f MiB) on %s",
            len(self.resident_ids), len(self.specs), used / 2**20, total / 2**20, self.device,
        )

    @property
    def streaming(self) -> bool:
        """True when at least one linear is re-read from the pack on every pass."""
        return len(self.resident_ids) < len(self.specs)

    def _host_weight(self, lid: LinearId) -> torch.Tensor:
        w8: torch.Tensor = self.pack.load_linear(lid, "cpu").w8
        return w8

    def _resident_weight(self, lid: LinearId) -> torch.Tensor:
        key = lid.key
        w = self._resident.get(key)
        if w is None:
            host = self._host_weight(lid)
            w = host.to(self.device) if self.device.type == "cuda" else host.clone()
            self._resident[key] = w
        return w

    # ---- CUDA double buffering -------------------------------------------------------------------
    def _pinned_buffer(self, slot: int, nbytes: int) -> torch.Tensor:  # pragma: no cover - GPU only
        while len(self._pinned) <= slot:
            self._pinned.append(torch.empty(0, dtype=torch.int8))
        buf = self._pinned[slot]
        if buf.numel() < nbytes:
            buf = torch.empty(nbytes, dtype=torch.int8, pin_memory=True)
            self._pinned[slot] = buf
        return buf

    def _start_copy(self, lid: LinearId, slot: int) -> tuple[torch.Tensor, Any]:  # pragma: no cover - GPU only
        host = self._host_weight(lid)
        nbytes = host.numel()
        prev = self._slot_events.get(slot)
        if prev is not None:
            prev.synchronize()  # the DMA that last read this staging buffer must be finished
        staging = self._pinned_buffer(slot, nbytes)[:nbytes].view(host.shape)
        staging.copy_(host)
        with torch.cuda.stream(self._copy_stream):
            dev = staging.to(self.device, non_blocking=True)
            ev = torch.cuda.Event()  # type: ignore[no-untyped-call]
            ev.record(self._copy_stream)
        self._slot_events[slot] = ev
        self.bytes_streamed += nbytes
        return dev, ev

    def iter_device_weights(self) -> Iterator[tuple[LinearId, torch.Tensor]]:
        """Yield ``(id, w8_on_device)`` in canonical order; each tensor valid until the next yield."""
        if self.device.type != "cuda":
            for s in self.specs:
                if s.id.key in self.resident_ids:
                    yield s.id, self._resident_weight(s.id)
                else:
                    w = self._host_weight(s.id)
                    self.bytes_streamed += w.numel()
                    yield s.id, w
            return
        yield from self._iter_cuda()  # pragma: no cover - GPU only

    def _iter_cuda(self) -> Iterator[tuple[LinearId, torch.Tensor]]:  # pragma: no cover - GPU only
        pending: dict[int, tuple[torch.Tensor, Any]] = {}
        streamed = [i for i, s in enumerate(self.specs) if s.id.key not in self.resident_ids]
        nxt = dict(itertools.pairwise(streamed))
        slot_of = {idx: j % 2 for j, idx in enumerate(streamed)}
        if streamed:
            pending[streamed[0]] = self._start_copy(self.specs[streamed[0]].id, slot_of[streamed[0]])
        cur = torch.cuda.current_stream(self.device)
        for i, s in enumerate(self.specs):
            if s.id.key in self.resident_ids:
                yield s.id, self._resident_weight(s.id)
                continue
            w, ev = pending.pop(i)
            cur.wait_event(ev)
            w.record_stream(cur)
            j = nxt.get(i)
            if j is not None:
                # start the next transfer now so it overlaps with the GEMMs on linear i
                pending[j] = self._start_copy(self.specs[j].id, slot_of[j])
            yield s.id, w
            del w

    def close(self) -> None:
        """Drop resident weights and pinned buffers."""
        self._resident.clear()
        self._pinned.clear()
