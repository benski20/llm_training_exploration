"""Optional: read the pks-v3 hint from remote or chunked storage with a bounded local cache (opt-in).

The decryption path reads the hint one linear at a time: :meth:`pgw.hemask.pks_backend.PKSMaskBackend.matmul` asks
``store.hint(linear)`` for the whole ``(m/32, 2, 8192, 3)`` array of that linear and the native decryption reads all
of it for every group. So the natural streaming unit is one linear (58 MB for a 7B ``gate_up``, 233 MB for the 7B
masked head), and :class:`StreamingHintStore` keeps an LRU cache of whole linears under a byte budget.

Integrity does not depend on the storage: the store's ``meta.json`` stays local (HMAC-checked with the customer's
hint key) and every fetched linear must hash to its digest there before it is used. The storage holds only the
customer's own hint files (public data); anything else is rejected.

Cost, stated plainly: the factory walks every linear once per segment, so with a cache smaller than the whole hint
each segment re-reads the whole hint (2.1-2.4 GB at 7B). At 7B and about 25 slots/s with 1024-slot segments that is
about 55 MB/s sustained from the storage. Use it when local disk is the constraint and the storage is close (same
region); otherwise keep the hint on local disk (the default). The hint cannot be streamed from the provider: the
client verifies the provider's ``H``, not the final hint, so the final hint can only come from storage the client
wrote it to (:func:`offload`).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import shutil
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from pgw.core.errors import ConfigError, IntegrityError

__all__ = ["ChunkSource", "DirectorySource", "HttpRangeSource", "StreamingHintStore", "offload"]

_READ = 8 << 20


class ChunkSource(Protocol):
    """Byte ranges of a store's files (``<linear>.npy``)."""

    def size(self, name: str) -> int: ...

    def read(self, name: str, offset: int, length: int) -> bytes: ...


class DirectorySource:
    """A directory (local, or a network / object-storage mount) holding the store's ``.npy`` files."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def size(self, name: str) -> int:
        return (self.root / name).stat().st_size

    def read(self, name: str, offset: int, length: int) -> bytes:
        with open(self.root / name, "rb") as f:
            f.seek(offset)
            return f.read(length)


class HttpRangeSource:
    """``<base_url>/<name>`` with HTTP range requests (e.g. the customer's object storage behind its own auth)."""

    def __init__(self, base_url: str, *, headers: dict[str, str] | None = None, timeout_s: float = 300.0):
        import httpx

        self.base = base_url.rstrip("/")
        self.client = httpx.Client(headers=headers or {}, timeout=timeout_s, follow_redirects=False)

    def size(self, name: str) -> int:
        r = self.client.head(f"{self.base}/{name}")
        r.raise_for_status()
        return int(r.headers["content-length"])

    def read(self, name: str, offset: int, length: int) -> bytes:
        r = self.client.get(f"{self.base}/{name}", headers={"Range": f"bytes={offset}-{offset + length - 1}"})
        r.raise_for_status()
        return bytes(r.content)


class StreamingHintStore:
    """The interface :class:`pgw.hemask.pks_backend.PKSMaskBackend` uses (``meta``, ``a_seed``, ``ks_seed``,
    ``hint(linear)``, ``nbytes``, ``close()``), backed by a :class:`ChunkSource` and an LRU of whole linears."""

    def __init__(self, meta: dict[str, Any], source: ChunkSource, *, cache_bytes: int = 512 << 20):
        self.meta = meta
        self.path = None
        self.a_seed = bytes.fromhex(meta["a_seed"])
        self.ks_seed = bytes.fromhex(meta["ks_seed"])
        self.source = source
        self.cache_bytes = int(cache_bytes)
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._used = 0
        self._lock = threading.Lock()
        self.fetched_bytes = 0
        self.fetches = 0

    @classmethod
    def open(cls, hint_root: str | Path, fingerprint: str, source: ChunkSource, *, mac_key: bytes,
             cache_bytes: int = 512 << 20) -> StreamingHintStore:
        """``meta.json`` from the local store directory (HMAC required: the digests in it are the trust anchor)."""
        from pgw.hemask.pks import NR, SCHEME_ID
        from pgw.hemask.pks_backend import _STORE_VERSION, PKSHintStore, _meta_mac

        d = PKSHintStore.dir_for(Path(hint_root), fingerprint)
        meta = json.loads((d / "meta.json").read_text())
        want = meta.get("mac")
        if not isinstance(want, str) or not hmac.compare_digest(want, _meta_mac(mac_key, meta)):
            raise IntegrityError(f"pks-v3 hint store {d}: meta.json is not authenticated by the hint key")
        if meta.get("version") != _STORE_VERSION or meta.get("scheme") != SCHEME_ID or meta.get("nr") != NR \
                or meta.get("fingerprint") != fingerprint:
            raise ConfigError(f"hint store {d} is of another format or model")
        return cls(meta, source, cache_bytes=cache_bytes)

    def _fetch(self, linear: str) -> np.ndarray:
        name = f"{linear}.npy"
        want = self.meta["digests"].get(linear)
        if want is None:
            raise ConfigError(f"no hint for {linear}")
        size = self.source.size(name)
        buf = bytearray()
        h = hashlib.sha256()
        while len(buf) < size:
            part = self.source.read(name, len(buf), min(_READ, size - len(buf)))
            if not part:
                raise IntegrityError(f"{name}: short read from hint storage")
            h.update(part)
            buf += part
        self.fetched_bytes += len(buf)
        self.fetches += 1
        if not hmac.compare_digest(h.hexdigest(), want):
            raise IntegrityError(f"{name} from hint storage does not match its digest in the authenticated meta.json")
        import io

        arr: np.ndarray = np.load(io.BytesIO(bytes(buf)), allow_pickle=False)
        return arr

    def hint(self, linear: str) -> np.ndarray:
        with self._lock:
            a = self._cache.get(linear)
            if a is not None:
                self._cache.move_to_end(linear)
                return a
        a = self._fetch(linear)
        with self._lock:
            if linear not in self._cache:
                self._cache[linear] = a
                self._used += a.nbytes
                while self._used > self.cache_bytes and len(self._cache) > 1:
                    _k, old = self._cache.popitem(last=False)
                    self._used -= old.nbytes
            return self._cache[linear]

    @property
    def cached_bytes(self) -> int:
        return self._used

    @property
    def nbytes(self) -> int:
        return sum(self.source.size(f"{k}.npy") for k in self.meta["digests"])

    def close(self) -> None:
        with self._lock:
            self._cache.clear()
            self._used = 0


def offload(store: Any, dest: str | Path, *, delete_local: bool = True) -> Path:
    """Copy a built store's hint files to ``dest`` (a mounted storage directory) and, by default, delete the local
    copies (``meta.json`` stays: it is the trust anchor). Hold no factory on the store while doing this."""
    dst = Path(dest)
    dst.mkdir(parents=True, exist_ok=True)
    for p in sorted(Path(store.path).glob("*.npy")):
        shutil.copyfile(p, dst / p.name)
        if delete_local:
            p.unlink()
    return dst
