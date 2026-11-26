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
    "SafetensorsRandomWriter",
    "SafetensorsStreamWriter",
    "StHeaderEntry",
    "TensorPlan",
    "hash_tensor_ranges",
    "planned_layout",
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


def planned_layout(
    file_name: str, plans: list[TensorPlan], metadata: Mapping[str, str] | None
) -> tuple[bytes, dict[str, int], int]:
    """The fixed layout of a planned safetensors file: ``(header bytes, absolute offset per tensor, file size)``.

    The header is canonical JSON (sorted keys, no spaces) padded to 8 bytes, tensors in plan order; the same plans
    and metadata always give the same bytes (both writers below use it)."""
    names = [p.name for p in plans]
    if len(set(names)) != len(names):
        raise ConfigError(f"duplicate tensor names in shard {file_name}")
    header: dict[str, object] = {}
    if metadata:
        header["__metadata__"] = {str(k): str(v) for k, v in metadata.items()}
    off = 0
    rel: dict[str, int] = {}
    for p in plans:
        if p.dtype not in _DTYPES:
            raise ConfigError(f"unsupported dtype {p.dtype!r} for tensor {p.name}")
        header[p.name] = {"dtype": _DTYPES[p.dtype][0], "shape": list(p.shape), "data_offsets": [off, off + p.nbytes]}
        rel[p.name] = off
        off += p.nbytes
    hb = json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
    hb += b" " * ((-len(hb)) % 8)  # 8-byte alignment of the data section
    base = 8 + len(hb)
    return hb, {k: base + v for k, v in rel.items()}, base + off


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
        hb, _, total = planned_layout(self.path.name, self.plans, metadata)
        self._header = hb
        self.total_bytes = total
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


class SafetensorsRandomWriter:
    """A planned safetensors file (same layout and bytes as :class:`SafetensorsStreamWriter`) whose tensors are
    written in any order, possibly across processes: the shard-wise packer fills each tensor when its checkpoint
    shard is present and resumes an interrupted run in the same file.

    ``open()`` creates the file (header + sparse data section of the planned size), or with ``reuse=True`` keeps an
    existing file whose header and size are exactly the planned ones (tensors already written stay; the caller
    knows which, from its own fsynced state). :meth:`write` returns the SHA-256 of the bytes written; :meth:`sync`
    fsyncs. Nothing here decides completeness: the caller records what was written.
    """

    def __init__(self, path: str | Path, plans: Iterable[TensorPlan], metadata: Mapping[str, str] | None = None):
        self.path = Path(path)
        self.plans = {p.name: p for p in plans}
        self._header, self._offsets, self.total_bytes = planned_layout(self.path.name, list(self.plans.values()),
                                                                       metadata)
        self._fd: int | None = None

    def open(self, *, reuse: bool = False) -> SafetensorsRandomWriter:
        """Open (create, or with ``reuse`` keep a matching existing file)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if reuse and self.path.is_file() and self.path.stat().st_size == self.total_bytes:
            fd = os.open(self.path, os.O_RDWR)
            want = struct.pack("<Q", len(self._header)) + self._header
            if os.pread(fd, len(want), 0) == want:
                self._fd = fd
                return self
            os.close(fd)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
        try:
            os.pwrite(fd, struct.pack("<Q", len(self._header)) + self._header, 0)
            os.ftruncate(fd, self.total_bytes)
        except BaseException:
            os.close(fd)
            raise
        self._fd = fd
        return self

    def write(self, name: str, t: torch.Tensor) -> str:
        """Write planned tensor ``name`` at its offset; returns the hex SHA-256 of its bytes."""
        if self._fd is None:
            raise ConfigError("SafetensorsRandomWriter used before open()")
        p = self.plans.get(name)
        if p is None:
            raise ConfigError(f"unexpected tensor {name!r} for {self.path.name}")
        if pack_dtype_of(t) != p.dtype or tuple(t.shape) != p.shape:
            raise ShapeError(
                f"{name}: planned {p.dtype}{list(p.shape)}, got {pack_dtype_of(t)}{list(t.shape)}",
                details={"tensor": name},
            )
        data = tensor_bytes(t)
        off, view, done = self._offsets[name], data, 0
        while done < len(view):
            n = os.pwrite(self._fd, view[done : done + (256 << 20)], off + done)
            if n <= 0:  # pragma: no cover - pwrite raises on errors
                raise OSError(f"short write to {self.path.name}")
            done += n
        return hashlib.sha256(data).hexdigest()

    def sync(self) -> None:
        """fsync the file."""
        if self._fd is not None:
            os.fsync(self._fd)

    def close(self) -> None:
        """fsync and close."""
        fd, self._fd = self._fd, None
        if fd is not None:
            try:
                os.fsync(fd)
            finally:
                os.close(fd)


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
    if not isinstance(meta_raw, dict):
        raise ConfigError(f"{p.name}: malformed safetensors __metadata__")
    entries: dict[str, StHeaderEntry] = {}
    for name, e in hdr.items():
        try:
            dt = _ST_TO_PACK[e["dtype"]]
            offs = [int(x) for x in e["data_offsets"]]
            shape = tuple(int(s) for s in e["shape"])
            b, en = offs
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigError(f"{p.name}: bad header entry for {name!r}: {e!r}") from exc
        if b < 0 or any(d < 0 for d in shape):
            raise ConfigError(f"{p.name}: tensor {name!r} has negative offsets or dimensions")
        ent = StHeaderEntry(name, dt, shape, base + b, base + en)
        if ent.end > size or ent.begin > ent.end or ent.nbytes != TensorPlan(name, dt, shape).nbytes:
            raise ConfigError(f"{p.name}: tensor {name!r} has inconsistent offsets (truncated file?)")
        entries[name] = ent
    return entries, {str(k): str(v) for k, v in meta_raw.items()}


def hash_tensor_ranges(
    path: str | Path, entries: Iterable[StHeaderEntry], *, forbid_int8_min: Iterable[str] = ()
) -> dict[str, str]:
    """Hex SHA-256 of each tensor's byte range in ``path`` (sequential reads, bounded memory).

    Args:
        forbid_int8_min: Names of int8 tensors that must not contain ``-128`` (byte ``0x80``).
            SPEC section 5 fixes ``W8`` to ``[-127, 127]``; the output range check and ring
            headroom rely on it. Scanning happens on the same buffers as hashing (memchr speed).

    Raises:
        ConfigError: short read, or a forbidden ``-128`` value was found.
    """
    out: dict[str, str] = {}
    forbid = set(forbid_int8_min)
    with open(path, "rb") as f:
        for e in sorted(entries, key=lambda x: x.begin):
            h = hashlib.sha256()
            scan = e.name in forbid and e.dtype == "int8"
            f.seek(e.begin)
            remaining = e.nbytes
            while remaining > 0:
                chunk = f.read(min(_HASH_BLOCK, remaining))
                if not chunk:
                    raise ConfigError(f"{Path(path).name}: unexpected end of file in {e.name}")
                if scan and b"\x80" in chunk:
                    raise ConfigError(
                        f"{Path(path).name}: int8 weight {e.name} contains -128 (packs use [-127, 127])",
                        details={"tensor": e.name},
                    )
                h.update(chunk)
                remaining -= len(chunk)
            out[e.name] = h.hexdigest()
    return out
