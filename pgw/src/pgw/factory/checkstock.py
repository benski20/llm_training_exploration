"""Pre-drawn check material for a factory that keeps no int8 weights at rest (``factory_weights = "stream"``).

Option C of ``research/reviews/weight-free-client-checks.md``; security argument:
``research/reviews/weight-free-stock-security.md``. Opt-in; the default (``"pack"``) keeps ``W8`` on disk.

The encrypted factory (pks-v3) never reads ``W8`` for the masks themselves (the provider computes ``W8 r`` under
encryption); it needs ``W8`` only for two kinds of check material:

* **segment sets**: a fresh 32-byte segment seed (``os.urandom``) and, per linear, ``uW = u W8`` for the segment's
  Freivalds vectors ``u`` derived from that seed exactly as SPEC 7.3 says (p31: ``prime_check_vectors``; ring:
  ``ring_uniform(..., "u", linear)``). One set per segment;
* **private sets**: ``c`` uniform ring vectors ``v`` per linear (from a fresh seed, ``ring_uniform(..., "v")``) and
  ``v W8`` for the every-row 40-bit self-check of provider products. One set per worker start and per failed or
  aborted segment (``MaskGenerator._wipe_private``).

:class:`CheckStock` draws both ahead of time during a **pass**: ``W8`` is streamed once from an untrusted source
(the provider's provisioning endpoint, the public checkpoint, or a directory), every tensor is checked against the
manifest the pinned fingerprint covers, and the material for the next ``target_segments`` segments and
``target_private`` private sets is computed by the same kernels as on demand (several seeds' vectors stacked into
one exact product). Nothing of a pass is kept unless every linear passed its check. The weights are then dropped.

At rest (``<root>/``, 0700): one file per item, AES-256-GCM under a fresh data key wrapped by the mask store's
KEK provider (the envelope of the store's segments), the item header as AAD; ``index.json`` lists the items with
their header digests and is HMAC'd under the KEK-derived MAC key with its own domain label; a generation witness
outside the directory detects an index rolled back on its own. A take records the consumption durably (index
rewritten and synced) before the item is read, then deletes the item file: after a crash nothing is handed out
twice. Tampered material fails authentication and is burned. Below the low-water mark a background pass refills
the stock (one pass at a time across processes); with an empty stock a take waits for a pass and fails closed
with :class:`CheckStockEmpty` after ``wait_s``.
"""

from __future__ import annotations

import base64
import contextlib
import fcntl
import hashlib
import hmac
import json
import logging
import os
import struct
import sys
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch

from pgw.core.errors import ConfigError, IntegrityError, MaskStoreError, ServiceUnavailableError
from pgw.core.fingerprint import canonical_json
from pgw.core.prng import SEED_BYTES, SeedExpander, new_seed
from pgw.core.ring import RingParams

__all__ = [
    "CheckStock", "CheckStockEmpty", "CheckStockError", "PrivateItem", "SegmentItem", "StockLinear", "WeightSource",
    "stock_status",
]

log = logging.getLogger(__name__)

ITEM_MAGIC = b"PGWCHK01"
INDEX_FILE = "index.json"
STATUS_FILE = "status.json"
_INDEX_DOMAIN = b"pgw/v1/check-stock-index\0"
_WITNESS_DOMAIN = b"pgw/v1/check-stock-witness\0"
_ITEM_AAD = b"pgw/v1/check-stock-item\0"
_KINDS = ("segment", "private")
_PREFIX = {"segment": "seg", "private": "priv"}
_SUFFIX = ".pgwc"
_MAX_HEADER = 4 << 20

if sys.byteorder != "little":  # pragma: no cover - the stock's records are little-endian uint32 words
    raise ImportError("pgw.factory.checkstock needs a little-endian host")


class CheckStockError(MaskStoreError):
    """Check material at rest failed authentication or is inconsistent (tampered, rolled back, foreign)."""


class CheckStockEmpty(ServiceUnavailableError):  # noqa: N818 - a state, like MaskExhaustedError
    """No check material in stock and no pass delivered any within the wait (fail closed: no segment, no masks)."""


class WeightSource(Protocol):
    """``W8`` per linear key, each checked against the pinned manifest before it is returned (``(m, n)`` int8)."""

    bytes: int

    def get(self, key: str) -> np.ndarray: ...  # pragma: no cover - protocol

    def close(self) -> None: ...  # pragma: no cover - protocol


@dataclass(frozen=True)
class StockLinear:
    """A linear the stock holds material for: key, ``m`` (out), ``n`` (in), ``c`` private vectors."""

    key: str
    out_features: int
    in_features: int
    private_vectors: int

    def as_list(self) -> list[Any]:
        return [self.key, int(self.out_features), int(self.in_features), int(self.private_vectors)]


@dataclass
class SegmentItem:
    """A segment's seed and its check vectors ``uW`` (per linear ``(checks, n)`` uint32). Single use."""

    seq: int
    seed: bytearray
    uw: dict[str, np.ndarray]

    def uw_tensor(self, key: str) -> torch.Tensor:
        """``uW`` of ``key`` as the int64 ``(checks, n)`` tensor the segment writer takes."""
        return torch.from_numpy(self.uw[key].astype(np.int64))

    def wipe(self) -> None:
        self.seed[:] = bytes(len(self.seed))
        for a in self.uw.values():
            a.fill(0)
        self.uw.clear()


@dataclass
class PrivateItem:
    """A private self-check set: its ``v`` seed and ``v W8`` per linear (``(c, n)`` uint32). Single use."""

    seq: int
    v_seed: bytearray
    vw: dict[str, np.ndarray]
    _cache: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)

    def check_vectors(self, lin: StockLinear, ring: RingParams) -> tuple[np.ndarray, np.ndarray]:
        """``(cu, cw)`` exactly as :meth:`pgw.factory.MaskGenerator._private_check` returns them: ``v`` regenerated
        from its seed ``(m, c)`` and ``v W8`` transposed ``(n, c)``, C-contiguous, uint32 (K <= 32) or uint64."""
        hit = self._cache.get(lin.key)
        if hit is not None:
            return hit
        dt = np.uint32 if ring.k <= 32 else np.uint64
        v = SeedExpander(bytes(self.v_seed)).ring_uniform((lin.private_vectors, lin.out_features), ring.k, "v")
        try:
            cu = np.ascontiguousarray(v.t().numpy().astype(dt))
        finally:
            v.zero_()
        cw = np.ascontiguousarray(self.vw[lin.key].T.astype(dt))
        self._cache[lin.key] = (cu, cw)
        return cu, cw

    def wipe(self) -> None:
        self.v_seed[:] = bytes(len(self.v_seed))
        for a in self.vw.values():
            a.fill(0)
        for cu, cw in self._cache.values():
            cu.fill(0)
            cw.fill(0)
        self.vw.clear()
        self._cache.clear()


def _fsync_dir(path: Path) -> None:
    with contextlib.suppress(OSError):
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def _atomic_write(path: Path, data: bytes) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp, path)
    _fsync_dir(path.parent)


def _b64(b: bytes) -> str:
    return base64.b64encode(bytes(b)).decode("ascii")


def _unb64(s: Any) -> bytes:
    try:
        return base64.b64decode(str(s), validate=True)
    except (ValueError, TypeError) as exc:
        raise CheckStockError("check stock item header is malformed") from exc


class _ItemWriter:
    """One item file being written during a pass: header (AAD of every record), then the records in order."""

    def __init__(self, path: Path, header: dict[str, Any], dek: bytes, nonce_prefix: bytes):
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        self.path = path
        self.header_bytes = canonical_json(header)
        self.digest = hashlib.sha256(self.header_bytes).digest()
        self.lengths = list(header["records"])
        self._aead = AESGCM(bytes(dek))
        self._prefix = bytes(nonce_prefix)
        self._i = 0
        self._fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.write(self._fd, ITEM_MAGIC + struct.pack("<I", len(self.header_bytes)) + self.header_bytes)

    def record(self, data: bytes | bytearray | memoryview) -> None:
        if self._i >= len(self.lengths) or len(data) != self.lengths[self._i]:
            raise ConfigError("check stock record of the wrong size")  # a bug, never a stock entry
        ct = self._aead.encrypt(self._prefix + struct.pack(">Q", self._i), data,
                                _ITEM_AAD + self.digest + struct.pack(">I", self._i))
        os.write(self._fd, ct)
        self._i += 1

    def finish(self) -> None:
        if self._i != len(self.lengths):
            raise ConfigError("check stock item is incomplete")
        os.fsync(self._fd)
        os.close(self._fd)
        self._fd = -1

    def discard(self) -> None:
        if self._fd >= 0:
            with contextlib.suppress(OSError):
                os.close(self._fd)
            self._fd = -1
        with contextlib.suppress(OSError):
            self.path.unlink()


def _read_item(path: Path, key_provider: Any, expect: dict[str, Any], digest_hex: str) -> list[bytearray]:
    """Authenticate and decrypt one item file; ``expect`` are header fields that must match the index."""
    from cryptography.exceptions import InvalidTag
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    from pgw.maskstore.keys import WrappedKey

    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise CheckStockError(f"check stock item {path.name} is missing or unreadable") from exc
    if len(raw) < len(ITEM_MAGIC) + 4 or raw[:len(ITEM_MAGIC)] != ITEM_MAGIC:
        raise CheckStockError(f"check stock item {path.name} has a bad magic")
    (hlen,) = struct.unpack("<I", raw[len(ITEM_MAGIC):len(ITEM_MAGIC) + 4])
    if hlen > _MAX_HEADER or len(raw) < len(ITEM_MAGIC) + 4 + hlen:
        raise CheckStockError(f"check stock item {path.name} is truncated")
    hb = raw[len(ITEM_MAGIC) + 4:len(ITEM_MAGIC) + 4 + hlen]
    digest = hashlib.sha256(hb).digest()
    if not hmac.compare_digest(digest.hex(), str(digest_hex)):
        raise CheckStockError(f"check stock item {path.name} does not match its index entry (swapped or tampered)")
    try:
        header = json.loads(hb)
    except ValueError as exc:
        raise CheckStockError(f"check stock item {path.name} header is not JSON") from exc
    for k, v in expect.items():
        if header.get(k) != v:
            raise CheckStockError(f"check stock item {path.name}: {k} does not match the stock")
    dek = bytearray(key_provider.unwrap(WrappedKey(str(header["kek_id"]), _unb64(header["wrapped_dek"]))))
    try:
        aead = AESGCM(bytes(dek))
        prefix = _unb64(header["nonce_prefix"])
        off = len(ITEM_MAGIC) + 4 + hlen
        out: list[bytearray] = []
        for i, n in enumerate(header["records"]):
            ct = raw[off:off + int(n) + 16]
            if len(ct) != int(n) + 16:
                raise CheckStockError(f"check stock item {path.name} is truncated")
            try:
                pt = aead.decrypt(prefix + struct.pack(">Q", i), ct, _ITEM_AAD + digest + struct.pack(">I", i))
            except InvalidTag as exc:
                for b in out:
                    b[:] = bytes(len(b))
                raise CheckStockError(f"check stock item {path.name}: record {i} failed authentication "
                                      "(corrupt or tampered)") from exc
            out.append(bytearray(pt))
            off += int(n) + 16
        if off != len(raw):
            for b in out:
                b[:] = bytes(len(b))
            raise CheckStockError(f"check stock item {path.name} has trailing bytes")
        return out
    finally:
        dek[:] = bytes(len(dek))


class CheckStock:
    """The pre-drawn check material of one (pack, ring) for a weight-free factory (see the module docstring).

    Args:
        root: Stock directory (shared by every worker process of the factory; created 0700).
        key_provider: The mask store's KEK provider (``wrap`` / ``unwrap`` / ``mac``); a provider without ``mac`` is
            refused.
        fingerprint, ring, check_field, checks: What the segment sets are for (``check_field`` ``None`` = the ring
            check, ``"p31"``); a stock made for anything else is discarded.
        linears: Every linear the generator produces, in its order, with its private vector count.
        source: ``source() -> WeightSource`` opened for each pass (``None``: no refill; tests).
        target_segments / low_water_segments: Segment sets a pass fills up to / a pass starts below.
        target_private / low_water_private: The same for private sets.
        wait_s: How long a take waits for a pass when the stock is empty before failing closed.
        witness: Generation witness file (default ``<root>.witness``, next to the directory).
        background: Start refill passes in a daemon thread (``False``: :meth:`refill` only).
    """

    def __init__(self, root: str | Path, key_provider: Any, *, fingerprint: str, ring: RingParams | str,
                 check_field: str | None, checks: int, linears: list[StockLinear],
                 source: Callable[[], WeightSource] | None = None, target_segments: int = 64,
                 low_water_segments: int = 16, target_private: int = 8, low_water_private: int = 3,
                 wait_s: float = 600.0, witness: str | Path | None = None, background: bool = True,
                 retry_s: float = 5.0):
        from pgw.core.settings import ring_from_setting

        self.root = Path(root)
        self.kp = key_provider
        if key_provider.mac(b"probe") is None:
            raise ConfigError("factory_weights='stream' needs a KEK provider that can MAC (file, env, kms)")
        self.fingerprint = str(fingerprint)
        self.ring = ring_from_setting(ring)
        if self.ring.k > 32:
            raise ConfigError("the check stock stores 32-bit words (K <= 32 rings; pks-v3 is K32)")
        self.check_field = None if check_field in (None, "ring") else str(check_field)
        if self.check_field not in (None, "p31"):
            raise ConfigError(f"check stock: unsupported check field {check_field!r}")
        self.checks = int(checks)
        self.linears = list(linears)
        self._lin = {lin.key: lin for lin in self.linears}
        if not self.linears or len(self._lin) != len(self.linears):
            raise ConfigError("check stock needs distinct linears")
        self.source = source
        if not (0 <= low_water_segments < target_segments) or not (0 <= low_water_private < target_private):
            raise ConfigError("check stock: low water must be below the target (segments and private sets)")
        self.target = {"segment": int(target_segments), "private": int(target_private)}
        self.low_water = {"segment": int(low_water_segments), "private": int(low_water_private)}
        self.wait_s = float(wait_s)
        self.witness = Path(witness) if witness is not None else self.root.with_name(self.root.name + ".witness")
        self.background = bool(background)
        self.retry_s = float(retry_s)
        self._mu = threading.Lock()
        self._thread: threading.Thread | None = None
        self._next_try = 0.0
        self._failures = 0
        self._closed = threading.Event()
        self._pass_done = threading.Condition()
        self.params = {"fingerprint": self.fingerprint, "ring": self.ring.name,
                       "check_field": self.check_field or "ring",
                       "checks": self.checks, "linears": [lin.as_list() for lin in self.linears]}
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.root, 0o700)
        with self._locked():
            self._open()

    # ---- locking / index -------------------------------------------------------------------------
    @contextlib.contextmanager
    def _locked(self) -> Iterator[None]:
        with self._mu:
            fd = os.open(self.root / "stock.lock", os.O_RDWR | os.O_CREAT, 0o600)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX)
                yield
            finally:
                os.close(fd)

    def _mac(self, domain: bytes, obj: Any) -> str:
        m = self.kp.mac(domain + canonical_json(obj))
        if m is None:  # pragma: no cover - checked at construction
            raise ConfigError("the KEK provider cannot MAC")
        return bytes(m).hex()

    def _new_index(self) -> dict[str, Any]:
        return {"v": 1, "stock_id": os.urandom(8).hex(), **self.params, "generation": 0, "next_seq": 0,
                "segment": [], "private": [], "passes": 0, "taken": {"segment": 0, "private": 0}}

    def _read_witness(self) -> dict[str, Any] | None:
        try:
            w = json.loads(self.witness.read_text())
        except FileNotFoundError:
            return None
        except (OSError, ValueError) as exc:
            raise CheckStockError("check stock witness is unreadable") from exc
        body = {k: w.get(k) for k in ("stock_id", "generation")}
        if not isinstance(w, dict) or not hmac.compare_digest(str(w.get("mac", "")), self._mac(_WITNESS_DOMAIN, body)):
            raise CheckStockError("check stock witness failed authentication")
        return body

    def _load(self) -> dict[str, Any]:
        """The index, authenticated and not behind its witness (raises :class:`CheckStockError` otherwise)."""
        p = self.root / INDEX_FILE
        try:
            idx = json.loads(p.read_text())
        except FileNotFoundError as exc:
            raise CheckStockError("check stock index is missing") from exc
        except (OSError, ValueError) as exc:
            raise CheckStockError("check stock index is unreadable") from exc
        if not isinstance(idx, dict):
            raise CheckStockError("check stock index is malformed")
        mac = str(idx.pop("mac", ""))
        if not hmac.compare_digest(mac, self._mac(_INDEX_DOMAIN, idx)):
            raise CheckStockError("check stock index failed authentication (tampered, or another KEK)")
        w = self._read_witness()
        if w is None:
            raise CheckStockError("check stock witness is missing (rolled back or tampered)")
        if w["stock_id"] != idx.get("stock_id") or int(idx["generation"]) < int(w["generation"]):
            raise CheckStockError("check stock index is behind its witness or from another stock (rolled back)")
        return idx

    def _store(self, idx: dict[str, Any]) -> None:
        idx["generation"] = int(idx["generation"]) + 1
        body = dict(idx)
        _atomic_write(self.root / INDEX_FILE, json.dumps({**body, "mac": self._mac(_INDEX_DOMAIN, body)}).encode())
        wb = {"stock_id": idx["stock_id"], "generation": idx["generation"]}
        _atomic_write(self.witness, json.dumps({**wb, "mac": self._mac(_WITNESS_DOMAIN, wb)}).encode())
        self._write_status(idx)

    def _item_path(self, kind: str, seq: int) -> Path:
        return self.root / f"{_PREFIX[kind]}-{int(seq):012d}{_SUFFIX}"

    def _open(self) -> None:
        """Load (or create) the index; discard a stock that fails authentication or was made for other parameters;
        delete files the index does not list (a crash after a take, or a pass that never committed)."""
        p = self.root / INDEX_FILE
        idx: dict[str, Any] | None = None
        if p.exists():
            try:
                idx = self._load()
            except CheckStockError as exc:
                log.error("check stock %s: %s; discarding it (never used), a new pass refills it", self.root,
                          exc.message)
                idx = None
            if idx is not None and any(idx.get(k) != v for k, v in self.params.items()):
                log.warning("check stock %s was made for other parameters; discarding it", self.root)
                idx = None
        if idx is None:
            idx = self._new_index()
            self._store(idx)
        listed = {self._item_path(kind, seq).name for kind in _KINDS for seq, _ in idx[kind]}
        running = self._pass_running()
        for f in self.root.iterdir():
            n = f.name
            if (n.endswith(_SUFFIX) and n not in listed) or (n.startswith(".pass-") and not running):
                with contextlib.suppress(OSError):
                    f.unlink()
        _fsync_dir(self.root)
        self._write_status(idx)

    def _pass_running(self) -> bool:
        fd = os.open(self.root / "refill.lock", os.O_RDWR | os.O_CREAT, 0o600)
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                return True
            return False
        finally:
            os.close(fd)

    # ---- status ----------------------------------------------------------------------------------
    def _write_status(self, idx: dict[str, Any] | None = None, **extra: Any) -> None:
        sp = self.root / STATUS_FILE
        try:
            st = json.loads(sp.read_text())
        except (OSError, ValueError):
            st = {}
        if idx is not None:
            st.update({"segments": len(idx["segment"]), "private_sets": len(idx["private"]),
                       "passes": int(idx["passes"]), "taken_segments": int(idx["taken"]["segment"]),
                       "taken_private_sets": int(idx["taken"]["private"])})
        st.update({"target_segments": self.target["segment"], "low_water_segments": self.low_water["segment"],
                   "target_private_sets": self.target["private"], "low_water_private_sets": self.low_water["private"]})
        st.update(extra)
        with contextlib.suppress(OSError):
            tmp = sp.with_name(f".{sp.name}.{os.getpid()}.{threading.get_ident()}.tmp")
            tmp.write_text(json.dumps(st))
            os.chmod(tmp, 0o600)
            os.replace(tmp, sp)

    def status(self) -> dict[str, Any]:
        """Public counts (no secrets): items in stock, passes, last pass, last error."""
        return stock_status(self.root)

    def counts(self) -> dict[str, int]:
        with self._locked():
            idx = self._load()
        return {k: len(idx[k]) for k in _KINDS}

    # ---- takes -----------------------------------------------------------------------------------
    def _take(self, kind: str) -> tuple[int, list[bytearray]] | None:
        with self._locked():
            try:
                idx = self._load()
            except CheckStockError:
                self._discard_locked("the index failed its check")
                raise
            if not idx[kind]:
                return None
            seq, digest = idx[kind].pop(0)
            idx["taken"][kind] = int(idx["taken"][kind]) + 1
            self._store(idx)  # consumed, durably, before the item is read: never handed out twice
            path = self._item_path(kind, int(seq))
            try:
                return int(seq), _read_item(path, self.kp, {"kind": kind, "seq": int(seq), "stock_id": idx["stock_id"],
                                                            **self.params}, digest)
            finally:
                with contextlib.suppress(OSError):
                    path.unlink()
                _fsync_dir(self.root)

    def _wait_take(self, kind: str, timeout_s: float | None,
                   stop: threading.Event | None) -> tuple[int, list[bytearray]]:
        deadline = time.monotonic() + (self.wait_s if timeout_s is None else float(timeout_s))
        while True:
            got = self._take(kind)
            self.maybe_refill()
            if got is not None:
                return got
            if stop is not None and stop.is_set():
                from pgw.factory.governor import FactoryStopped

                raise FactoryStopped("factory stop requested while waiting for check material")
            left = deadline - time.monotonic()
            if left <= 0:
                st = self.status()
                why = st.get("last_pass_error")
                never = not st.get("passes")
                raise CheckStockEmpty(
                    f"no pre-drawn check material ({kind} sets) for the weight-free factory: the stock is empty and "
                    + ("no fingerprint-verified W8 pass has completed yet" if never else "the refill pass has not "
                       "finished") + (f" (last pass failed: {why})" if why else "")
                    + "; no segment is made without it (fail closed)",
                    details={"kind": kind, "passes": int(st.get("passes") or 0)})
            with self._pass_done:
                self._pass_done.wait(min(0.5, left))

    def take_segment(self, *, timeout_s: float | None = None, stop: threading.Event | None = None) -> SegmentItem:
        """The next segment set (seed and ``uW`` of every linear), consumed durably. Waits for a refill pass when the
        stock is empty, up to ``timeout_s`` (default ``wait_s``), then raises :class:`CheckStockEmpty`.

        Raises:
            CheckStockError: the item failed authentication (it is burned; the next take uses the next one).
        """
        seq, recs = self._wait_take("segment", timeout_s, stop)
        try:
            if len(recs) != 1 + len(self.linears) or len(recs[0]) != SEED_BYTES:
                raise CheckStockError("check stock segment item has the wrong layout")
            uw = {lin.key: np.frombuffer(r, dtype=np.uint32).reshape(self.checks, lin.in_features).copy()
                  for lin, r in zip(self.linears, recs[1:], strict=True)}
            return SegmentItem(seq, bytearray(recs[0]), uw)
        finally:
            for r in recs:
                r[:] = bytes(len(r))

    def take_private(self, *, timeout_s: float | None = None, stop: threading.Event | None = None) -> PrivateItem:
        """The next private self-check set, consumed durably (same waiting and errors as :meth:`take_segment`)."""
        seq, recs = self._wait_take("private", timeout_s, stop)
        try:
            if len(recs) != 1 + len(self.linears) or len(recs[0]) != SEED_BYTES:
                raise CheckStockError("check stock private item has the wrong layout")
            vw = {lin.key: np.frombuffer(r, dtype=np.uint32).reshape(lin.private_vectors, lin.in_features).copy()
                  for lin, r in zip(self.linears, recs[1:], strict=True)}
            return PrivateItem(seq, bytearray(recs[0]), vw)
        finally:
            for r in recs:
                r[:] = bytes(len(r))

    def linear(self, key: str) -> StockLinear:
        return self._lin[key]

    # ---- discard ---------------------------------------------------------------------------------
    def _discard_locked(self, why: str) -> None:
        log.error("check stock %s: %s; discarding every item (a new pass refills it)", self.root, why)
        for f in self.root.iterdir():
            if f.name.endswith(_SUFFIX):
                with contextlib.suppress(OSError):
                    f.unlink()
        idx = self._new_index()
        self._store(idx)

    def discard(self) -> None:
        """Delete every item (e.g. after a suspected compromise, with a KEK rotation); a new pass refills."""
        with self._locked():
            self._discard_locked("discard requested")

    # ---- refill ----------------------------------------------------------------------------------
    def maybe_refill(self) -> bool:
        """Start a background pass when a kind is below its low-water mark (one at a time; back-off after a failed
        pass). Returns whether a pass was started."""
        if not self.background or self.source is None or self._closed.is_set():
            return False
        with self._mu:
            if self._thread is not None and self._thread.is_alive():
                return False
            if time.monotonic() < self._next_try:
                return False
        try:
            c = self.counts()
        except CheckStockError:
            return False
        if all(c[k] >= self.low_water[k] and c[k] > 0 for k in _KINDS):
            return False
        with self._mu:
            if self._thread is not None and self._thread.is_alive():
                return False
            self._thread = threading.Thread(target=self._refill_bg, name="pgw-check-stock-pass", daemon=True)
            self._thread.start()
        return True

    def _refill_bg(self) -> None:
        try:
            self.refill(block=False)
            self._failures = 0
            self._write_status(pass_failures=0)
        except Exception as exc:  # recorded in the status; retried with back-off
            self._failures += 1
            delay = min(300.0, self.retry_s * 2 ** min(self._failures - 1, 6))
            self._next_try = time.monotonic() + delay
            self._write_status(pass_failures=self._failures)
            log.error("check stock pass failed (%s: %s); retrying in %.0fs", type(exc).__name__,
                      getattr(exc, "message", str(exc)), delay)
        finally:
            with self._pass_done:
                self._pass_done.notify_all()

    def refill(self, *, block: bool = True, seeds: list[bytes] | None = None) -> dict[str, Any]:
        """Run one pass now: stream ``W8`` (checked against the pin), fill both kinds to their targets, commit.

        ``block=False`` returns at once when another process's pass holds the lock. ``seeds`` (tests only): the
        segment seeds to use instead of fresh ``os.urandom`` ones. Returns the pass summary (counts, bytes, seconds).
        """
        if self.source is None:
            raise ConfigError("this check stock has no W8 source")
        fd = os.open(self.root / "refill.lock", os.O_RDWR | os.O_CREAT, 0o600)
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | (0 if block else fcntl.LOCK_NB))
            except OSError:
                return {"skipped": "another pass is running"}
            return self._pass(seeds)
        finally:
            os.close(fd)

    def _pass(self, test_seeds: list[bytes] | None) -> dict[str, Any]:
        t0 = time.perf_counter()
        with self._locked():
            idx = self._load()
            need = {k: max(0, self.target[k] - len(idx[k])) for k in _KINDS}
            if test_seeds is not None:
                need["segment"] = len(test_seeds)
            if not any(need.values()):
                return {"segments": 0, "private_sets": 0}
            seq0 = int(idx["next_seq"])
            idx["next_seq"] = seq0 + need["segment"] + need["private"]
            stock_id = idx["stock_id"]
            self._store(idx)
        self._write_status(refilling=True, pass_started_at=time.time())
        pass_id = os.urandom(6).hex()
        seeds = [bytearray(s) for s in test_seeds] if test_seeds is not None else \
            [bytearray(new_seed()) for _ in range(need["segment"])]
        vseeds = [bytearray(new_seed()) for _ in range(need["private"])]
        items: list[tuple[str, int, _ItemWriter]] = []
        src: WeightSource | None = None
        try:
            for kind, n in (("segment", need["segment"]), ("private", need["private"])):
                for _ in range(n):
                    seq = seq0 + len(items)
                    items.append((kind, seq, self._writer(kind, seq, stock_id, pass_id)))
            segs = [w for k, _, w in items if k == "segment"]
            privs = [w for k, _, w in items if k == "private"]
            for s, w in zip(seeds, segs, strict=True):
                w.record(s)
            for s, w in zip(vseeds, privs, strict=True):
                w.record(s)
            assert self.source is not None
            src = self.source()
            for lin in self.linears:
                w8 = src.get(lin.key)  # checked against the pinned manifest before it is returned
                if tuple(w8.shape) != (lin.out_features, lin.in_features):
                    raise IntegrityError(f"{lin.key}: streamed weight shape {tuple(w8.shape)} does not match the pack")
                wt = torch.from_numpy(np.require(w8, requirements=["C", "W"]))  # a read-only buffer is copied
                if segs:
                    self._segment_records(lin, wt, seeds, segs)
                if privs:
                    self._private_records(lin, wt, vseeds, privs)
                del wt, w8
            for _, _, w in items:
                w.finish()
            streamed = int(getattr(src, "bytes", 0))
            with self._locked():
                idx = self._load()
                if idx["stock_id"] != stock_id:
                    raise CheckStockError("the check stock was discarded during the pass")
                for kind, seq, w in items:
                    os.replace(w.path, self._item_path(kind, seq))
                    idx[kind].append([seq, w.digest.hex()])
                _fsync_dir(self.root)
                idx["passes"] = int(idx["passes"]) + 1
                self._store(idx)
            dt = time.perf_counter() - t0
            out = {"segments": need["segment"], "private_sets": need["private"], "weights_bytes": streamed,
                   "seconds": round(dt, 3)}
            self._write_status(refilling=False, last_pass_at=time.time(), last_pass_seconds=round(dt, 3),
                               last_pass_weights_bytes=streamed, last_pass_error=None,
                               weights_bytes_total=int(self.status().get("weights_bytes_total") or 0) + streamed)
            log.info("check stock pass: %d segment sets and %d private sets from %.2f GB of fingerprint-checked W8 "
                     "in %.1f s", need["segment"], need["private"], streamed / 1e9, dt)
            return out
        except BaseException as exc:
            for _, _, w in items:
                w.discard()
            self._write_status(refilling=False, last_pass_error=f"{type(exc).__name__}: "
                               f"{getattr(exc, 'message', None) or str(exc)}"[:400], last_pass_failed_at=time.time())
            raise
        finally:
            for s in seeds + vseeds:
                s[:] = bytes(len(s))
            if src is not None:
                with contextlib.suppress(Exception):
                    src.close()

    def _writer(self, kind: str, seq: int, stock_id: str, pass_id: str) -> _ItemWriter:
        dek = bytearray(os.urandom(32))
        try:
            wk = self.kp.wrap(bytes(dek))
            if kind == "segment":
                lens = [SEED_BYTES] + [4 * self.checks * lin.in_features for lin in self.linears]
            else:
                lens = [SEED_BYTES] + [4 * lin.private_vectors * lin.in_features for lin in self.linears]
            prefix = os.urandom(4)
            header = {"v": 1, "kind": kind, "seq": int(seq), "stock_id": stock_id, **self.params,
                      "kek_id": wk.kek_id, "wrapped_dek": _b64(wk.wrapped), "nonce_prefix": _b64(prefix),
                      "records": lens}
            path = self.root / f".pass-{pass_id}-{_PREFIX[kind]}-{int(seq):012d}.tmp"
            return _ItemWriter(path, header, bytes(dek), prefix)
        finally:
            dek[:] = bytes(len(dek))

    def _segment_records(self, lin: StockLinear, w: torch.Tensor, seeds: list[bytearray],
                         writers: list[_ItemWriter]) -> None:
        """``uW`` of every seed for one linear: the segment's ``u`` (SPEC 7.3) stacked into one exact product."""
        from pgw.factory.kernels import prime_vecmat, ring_vecmat

        c, m = self.checks, lin.out_features
        if self.check_field == "p31":
            from pgw.core.freivalds import prime_check_vectors

            us = [prime_check_vectors(bytes(s), c, m, lin.key) for s in seeds]
        else:
            us = [SeedExpander(bytes(s)).ring_uniform((c, m), self.ring.k, "u", lin.key) for s in seeds]
        u = torch.cat(us, 0)  # a copy: the parts are wiped now, the stack after the product
        for t in us:
            t.zero_()
        try:
            uw = prime_vecmat(u, w) if self.check_field == "p31" else ring_vecmat(u, w, self.ring)
        finally:
            u.zero_()
        try:
            a = uw.numpy()
            if a.size and (int(a.min()) < 0 or int(a.max()) > 0xFFFFFFFF):
                raise ConfigError("check vector out of the 32-bit range")  # pragma: no cover - kernels' contract
            a32 = np.ascontiguousarray(a.astype(np.uint32))  # little-endian hosts only (checked at import)
            for i, wr in enumerate(writers):
                wr.record(memoryview(a32[i * c:(i + 1) * c]).cast("B"))
            a32.fill(0)
        finally:
            uw.zero_()

    def _private_records(self, lin: StockLinear, w: torch.Tensor, vseeds: list[bytearray],
                         writers: list[_ItemWriter]) -> None:
        """``v W8`` of every private set for one linear (``v`` as :meth:`MaskGenerator._private_check` draws it)."""
        from pgw.factory.kernels import ring_vecmat

        c, m = lin.private_vectors, lin.out_features
        vs = [SeedExpander(bytes(s)).ring_uniform((c, m), self.ring.k, "v") for s in vseeds]
        v = torch.cat(vs, 0)  # a copy: the parts are wiped now, the stack after the product
        for t in vs:
            t.zero_()
        try:
            vw = ring_vecmat(v, w, self.ring)
        finally:
            v.zero_()
        try:
            a32 = np.ascontiguousarray(vw.numpy().astype(np.uint32))
            for i, wr in enumerate(writers):
                wr.record(memoryview(a32[i * c:(i + 1) * c]).cast("B"))
            a32.fill(0)
        finally:
            vw.zero_()

    def close(self) -> None:
        """Stop starting passes and wait for a running one (it commits or discards its own files)."""
        self._closed.set()
        t = self._thread
        if t is not None and t is not threading.current_thread():
            t.join(timeout=600)


def stock_status(root: str | Path) -> dict[str, Any]:
    """Public status of a stock directory (``status.json``; counts and pass times, no secrets), or ``{}``."""
    try:
        st = json.loads((Path(root) / STATUS_FILE).read_text())
    except (OSError, ValueError):
        return {}
    return st if isinstance(st, dict) else {}
