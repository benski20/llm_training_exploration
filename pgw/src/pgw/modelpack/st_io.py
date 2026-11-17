"""Low-level safetensors I/O for model packs: a streaming writer, a header reader and byte-range hashing.

The safetensors format is ``u64le(header_len) || header JSON || data``. The header maps tensor
names to ``{"dtype", "shape", "data_offsets": [begin, end]}`` relative to the start of the data
section, plus an optional ``"__metadata__"`` string map. Because all shapes are known before any
tensor is computed, :class:`SafetensorsStreamWriter` writes the header first and then streams tensor
bytes one at a time, so packing needs memory for one tensor, not one shard.

Hashing works on the raw byte ranges of the file (they are exactly the tensor's little-endian
bytes), so verification never materializes tensors in torch.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import struct
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import BinaryIO, Literal

import numpy as np
import torch

from pgw.core.errors import ConfigError, ShapeError

__all__ = [
    "DTYPE_BYTES",
    "PackDType",
    "SafetensorsStreamWriter",
    "StHeaderEntry",
    "TensorPlan",
    "hash_tensor_ranges",
    "read_safetensors_header",
    "tensor_bytes",
    "tensor_sha256",
    "torch_dtype",
]

log = logging.getLogger(__name__)

PackDType = Literal["int8", "float32", "float16", "bfloat16"]

#: Pack dtype name -> (safetensors dtype code, torch dtype, bytes per element).
_DTYPES: dict[str, tuple[str, torch.dtype, int]] = {
    "int8": ("I8", torch.int8, 1),
    "float32": ("F32", torch.float32, 4),
    "float16": ("F16", torch.float16, 2),
    "bfloat16": ("BF16", torch.bfloat16, 2),
}
_ST_TO_PACK: dict[str, str] = {v[0]: k for k, v in _DTYPES.items()}
_TORCH_TO_PACK: dict[torch.dtype, str] = {v[1]: k for k, v in _DTYPES.items()}
DTYPE_BYTES: dict[str, int] = {k: v[2] for k, v in _DTYPES.items()}

#: Refuse absurd headers (corrupt files) before allocating.
_MAX_HEADER_BYTES = 256 << 20
_HASH_BLOCK = 8 << 20


def torch_dtype(name: str) -> torch.dtype:
    """Torch dtype of a pack dtype name."""
    try:
        return _DTYPES[name][1]
    except KeyError as exc:
        raise ConfigError(f"unsupported pack dtype {name!r}") from exc


def pack_dtype_of(t: torch.Tensor) -> str:
    """Pack dtype name of a tensor (``int8``/``float32``/``float16``/``bfloat16``)."""
    try:
        return _TORCH_TO_PACK[t.dtype]
    except KeyError as exc:
        raise ShapeError(f"unsupported tensor dtype {t.dtype} for a model pack") from exc


def tensor_bytes(t: torch.Tensor) -> memoryview:
    """Raw little-endian bytes of a CPU tensor (contiguous copy if needed)."""
    t = t.detach().to("cpu").contiguous()
    if t.numel() == 0:
        return memoryview(b"")
    # View as uint8 so bfloat16 (no numpy dtype) works too. Host byte order is little-endian on
    # every supported platform (x86-64, aarch64); guard anyway.
    if not np.little_endian:  # pragma: no cover - big-endian hosts are unsupported
        raise ConfigError("big-endian hosts are not supported")
    return memoryview(t.view(torch.uint8).reshape(-1).numpy())


def tensor_sha256(t: torch.Tensor) -> str:
    """Hex SHA-256 of a tensor's raw little-endian bytes (as stored in the pack)."""
    return hashlib.sha256(tensor_bytes(t)).hexdigest()


@dataclass(frozen=True)
class TensorPlan:
    """A tensor to be written: name, pack dtype and shape (its bytes come later)."""

    name: str
    dtype: str
    shape: tuple[int, ...]

    @property
    def nbytes(self) -> int:
        """Size in bytes."""
        n = 1
        for d in self.shape:
            n *= int(d)
        return n * DTYPE_BYTES[self.dtype]


@dataclass(frozen=True)
class StHeaderEntry:
    """One tensor as described by a safetensors header (offsets are absolute file offsets)."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    begin: int
    end: int

    @property
    def nbytes(self) -> int:
        """Size in bytes."""
        return self.end - self.begin


class SafetensorsStreamWriter:
    """Write a safetensors file whose layout is fixed up front, streaming tensor data in order.

    Usage::

        with SafetensorsStreamWriter(path, plans, metadata={"format": "pt"}) as w:
            for plan in plans:
                digest = w.write(plan.name, tensor)   # must follow the plan order

    The file is fsynced on successful close. If the context exits with an exception, the partial
    file is removed.
    """

    def __init__(self, path: str | Path, plans: Iterable[TensorPlan], metadata: Mapping[str, str] | None = None):
        self.path = Path(path)
        self.plans = list(plans)
        names = [p.name for p in self.plans]
        if len(set(names)) != len(names):
            raise ConfigError(f"duplicate tensor names in shard {self.path.name}")
        header: dict[str, object] = {}
        if metadata:
            header["__metadata__"] = {str(k): str(v) for k, v in metadata.items()}
        off = 0
        for p in self.plans:
            if p.dtype not in _DTYPES:
                raise ConfigError(f"unsupported dtype {p.dtype!r} for tensor {p.name}")
            header[p.name] = {
                "dtype": _DTYPES[p.dtype][0], "shape": list(p.shape), "data_offsets": [off, off + p.nbytes]
            }
            off += p.nbytes
        hb = json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
        hb += b" " * ((-len(hb)) % 8)  # 8-byte alignment of the data section
        self._header = hb
        self.total_bytes = 8 + len(hb) + off
        self._idx = 0
        self._f: BinaryIO | None = None

    def __enter__(self) -> SafetensorsStreamWriter:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "wb")
        self._f.write(struct.pack("<Q", len(self._header)))
        self._f.write(self._header)
        return self

    def write(self, name: str, t: torch.Tensor) -> str:
        """Write the next planned tensor; returns the hex SHA-256 of its bytes."""
        if self._f is None:
            raise ConfigError("SafetensorsStreamWriter used outside its context")
        if self._idx >= len(self.plans):
            raise ConfigError(f"unexpected extra tensor {name!r} for {self.path.name}")
        p = self.plans[self._idx]
        if name != p.name:
            raise ConfigError(f"tensor order mismatch in {self.path.name}: expected {p.name!r}, got {name!r}")
        if pack_dtype_of(t) != p.dtype or tuple(t.shape) != p.shape:
            raise ShapeError(
                f"{name}: planned {p.dtype}{list(p.shape)}, got {pack_dtype_of(t)}{list(t.shape)}",
                details={"tensor": name},
            )
        data = tensor_bytes(t)
        self._f.write(data)
        self._idx += 1
        return hashlib.sha256(data).hexdigest()

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        f, self._f = self._f, None
        if f is None:
            return
        try:
            if exc_type is None:
                if self._idx != len(self.plans):
                    missing = [p.name for p in self.plans[self._idx:]]
                    raise ConfigError(f"{self.path.name}: {len(missing)} planned tensors not written: {missing[:3]}")
                f.flush()
                os.fsync(f.fileno())
        finally:
            f.close()
            if exc_type is not None or self._idx != len(self.plans):
                self.path.unlink(missing_ok=True)


def read_safetensors_header(path: str | Path) -> tuple[dict[str, StHeaderEntry], dict[str, str]]:
    """Parse a safetensors header.

    Returns:
        ``(entries, metadata)`` with absolute byte offsets.

    Raises:
        ConfigError: the file is missing, truncated or malformed.
    """
    p = Path(path)
    try:
        size = p.stat().st_size
        with open(p, "rb") as f:
            raw = f.read(8)
            if len(raw) != 8:
                raise ConfigError(f"{p.name}: truncated safetensors file")
            (hlen,) = struct.unpack("<Q", raw)
            if hlen > _MAX_HEADER_BYTES or 8 + hlen > size:
                raise ConfigError(f"{p.name}: invalid safetensors header length {hlen}")
            hdr = json.loads(f.read(hlen).decode("utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"missing pack file {p.name}", details={"file": p.name}) from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ConfigError(f"{p.name}: malformed safetensors header: {exc}") from exc
    if not isinstance(hdr, dict):
        raise ConfigError(f"{p.name}: malformed safetensors header")
    base = 8 + hlen
    meta_raw = hdr.pop("__metadata__", None) or {}
    entries: dict[str, StHeaderEntry] = {}
    for name, e in hdr.items():
        try:
            dt = _ST_TO_PACK[e["dtype"]]
            b, en = (int(x) for x in e["data_offsets"])
            shape = tuple(int(s) for s in e["shape"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigError(f"{p.name}: bad header entry for {name!r}: {e!r}") from exc
        ent = StHeaderEntry(name, dt, shape, base + b, base + en)
        if ent.end > size or ent.begin > ent.end or ent.nbytes != TensorPlan(name, dt, shape).nbytes:
            raise ConfigError(f"{p.name}: tensor {name!r} has inconsistent offsets (truncated file?)")
        entries[name] = ent
    return entries, {str(k): str(v) for k, v in dict(meta_raw).items()}


def hash_tensor_ranges(path: str | Path, entries: Iterable[StHeaderEntry]) -> dict[str, str]:
    """Hex SHA-256 of each tensor's byte range in ``path`` (sequential reads, bounded memory)."""
    out: dict[str, str] = {}
    with open(path, "rb") as f:
        for e in sorted(entries, key=lambda x: x.begin):
            h = hashlib.sha256()
            f.seek(e.begin)
            remaining = e.nbytes
            while remaining > 0:
                chunk = f.read(min(_HASH_BLOCK, remaining))
                if not chunk:
                    raise ConfigError(f"{Path(path).name}: unexpected end of file in {e.name}")
                h.update(chunk)
                remaining -= len(chunk)
            out[e.name] = h.hexdigest()
    return out
