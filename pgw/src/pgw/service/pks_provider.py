"""Provider side of maskgen wire v3 (scheme ``pks-v3``, :mod:`pgw.hemask.pks`): GEMM, modulus switch,
packing and hybrid key switch of one item, behind ``PGW_SERVICE_MASKGEN_V3`` (default off).

Per (model, customer upload seed, linear) the provider derives the public hint ``H = W8 A mod 2^56`` once
with its own GEMM (``A`` from the customer's public seed, :func:`pgw.hemask.pks.a_mlwe_rows`) and keeps
``round(H q1 / 2^56) mod q1`` in a bounded in-memory cache (``PGW_SERVICE_PKS_CACHE_GB``, default 8;
least recently used first out). Per item: ``y = W8 b mod 2^56`` (the v1/v2 GEMM), then the key switch on
the CPU (``pgw_pks``, numpy reference without it) or, when the kernel runs on CUDA and
``PGW_SERVICE_PKS_GPU`` is not ``0``, on the device (:mod:`pgw.service.pks_gpu`, self-tested bit for bit
against the CPU path before first use; any device error falls back to the CPU for good). The provider sees
only ring-LWE ciphertexts (``b``, ``ksk``) and public data; its outputs are exact functions of them.

The cached hint is ``m x 4096`` uint64 per linear: one seed's hints for the whole model (50.5 / 157 / 230 GB at
7B / 32B / 72B) do not fit the cache, so a decode cycle evicted every entry and every item rebuilt its hint
(GEMM, device -> host, host switch, host -> device). Since 2026-11-26 the default (``PGW_SERVICE_PKS_DEVICE_HINT``
``auto``) recomputes the hint per item on the CUDA device instead whenever the model's hints do not fit the cache
(:func:`pgw.service.pks_gpu.item`, self-tested bit for bit against the host path before first use); see
:class:`PKSProvider`.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from collections import OrderedDict
from collections.abc import Mapping
from typing import Any

import numpy as np

from pgw.core.errors import ProtocolError
from pgw.hemask import pks
from pgw.hemask.pks import NR, PACK, RANK, D, PKSParams
from pgw.hemask.wire import V3_KSK_BITS, V3_KSK_BYTES, V3_Y_BITS, pack_bits, unpack_bits

__all__ = ["HINT_MODES", "PKSProvider", "hint_bytes", "hint_mode", "switch_hint_host", "unpack_ksk"]

log = logging.getLogger(__name__)
_PAR = PKSParams()
_M56 = np.uint64((1 << 56) - 1)


def unpack_ksk(data: bytes | bytearray, groups: int) -> np.ndarray:
    """``(groups, 4, 16, 8192)`` uint32 reply keys from the wire (refuses residues ``>= p``)."""
    if len(data) != groups * V3_KSK_BYTES:
        raise ProtocolError("maskgen v3: reply key size mismatch")
    out = np.empty((groups, 4, RANK, NR), dtype=np.uint32)
    view = memoryview(data)
    pos = 0
    for g in range(groups):
        for i, bits in enumerate(V3_KSK_BITS):
            nb = RANK * NR * bits // 8
            v = unpack_bits(view[pos:pos + nb], RANK * NR, bits)
            pos += nb
            if int(v.max()) >= pks.QP[i]:
                raise ProtocolError("maskgen v3: reply key residue out of range")
            out[g, i] = v.reshape(RANK, NR)
    return out


def pack_ksk(beta: np.ndarray) -> bytearray:
    """Inverse of :func:`unpack_ksk` (customer side; natively when ``pgw_pks`` has ``pack_ksk``, same bytes)."""
    from pgw.hemask.pks_native import native

    nat = native()
    if nat is not None and hasattr(nat, "pack_ksk") and beta.ndim == 4 and beta.shape[1] == len(V3_KSK_BITS):
        return bytearray(nat.pack_ksk(np.ascontiguousarray(beta, dtype=np.uint32), list(V3_KSK_BITS)))
    out = bytearray()
    for g in range(beta.shape[0]):
        for i, bits in enumerate(V3_KSK_BITS):
            out += pack_bits(np.ascontiguousarray(beta[g, i], dtype=np.uint64).reshape(-1), bits)
    return out


def _kernel_operands(kern: Any) -> tuple[Any, Any] | None:
    """``(int8 weight on one ComputeKernel's device, its 7-plane maskgen constant)``, or ``None`` when the kernel
    has no exact int8 GEMM there (CPU float backends, test doubles)."""
    pw = getattr(kern, "_pw", None)
    dev = getattr(kern, "device", None)
    w = None
    if getattr(dev, "type", "cpu") == "cuda" and pw is not None and getattr(pw, "_w_dev", None) is not None:
        w = pw._w_dev  # (mp, np_) zero padded
    elif getattr(kern, "_w", None) is not None and getattr(kern, "backend", "") == "cpu_int8":
        w = kern._w  # (m, n), exact CPU int8 GEMMs (resolve_backend checked)
    if w is None or not hasattr(kern, "_maskgen_const"):
        return None
    return w, kern._maskgen_const(7, 0)


def _device_operands(kern: Any) -> list[tuple[Any, Any, int, int, Any]] | None:
    """The parts of :func:`pgw.service.pks_gpu.item` for ``kern``: ``(weight, constant, row0, row1, device)`` per
    device (one for a ComputeKernel; one per shard of a tensor-parallel ShardedKernel whose shard bounds are whole
    reply ciphertexts, i.e. multiples of 32 rows), or ``None`` (the cache path serves the kernel)."""
    if sys.byteorder != "little":  # pragma: no cover - the plane bytes assume little-endian words
        return None
    shards, bounds = getattr(kern, "shards", None), getattr(kern, "bounds", None)
    pairs = list(zip(shards, bounds, strict=True)) if shards is not None and bounds is not None \
        else [(kern, (0, int(kern.out_features)))]
    parts = []
    for k, (r0, r1) in pairs:
        if r0 % PACK or r1 % PACK:
            return None
        ops = _kernel_operands(k)
        if ops is None:
            return None
        parts.append((ops[0], ops[1], int(r0), int(r1), k.device))
    return parts


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def switch_hint_host(h: np.ndarray) -> np.ndarray:
    """:func:`pgw.hemask.pks.switch_hint` of ``H`` (entries taken mod ``2^56``): natively and multi-threaded when
    ``pgw_pks`` is present (the client's kernel, bit-identical: ``tests/hemask/test_pks.py``), else numpy."""
    from pgw.hemask.pks_native import native

    nat = native()
    h = np.ascontiguousarray(h, dtype=np.uint64)
    if nat is not None and hasattr(nat, "switch_hint"):
        from pgw.hemask.hint_gemm import threads

        out: np.ndarray = nat.switch_hint(h & _M56, threads(), "auto")
        return out
    return pks.switch_hint(h & _M56, _PAR)


#: ``PGW_SERVICE_PKS_DEVICE_HINT``: where the provider-side hint comes from (see :class:`PKSProvider`).
HINT_MODES = ("auto", "device", "cache")


def hint_mode() -> str:
    raw = os.environ.get("PGW_SERVICE_PKS_DEVICE_HINT", "").strip().lower()
    if raw in ("", "auto"):
        return "auto"
    if raw in ("1", "true", "yes", "on", "device"):
        return "device"
    if raw in ("0", "false", "no", "off", "cache", "host"):
        return "cache"
    raise ValueError(f"PGW_SERVICE_PKS_DEVICE_HINT={raw!r}: use auto, device (1) or cache (0)")


def hint_bytes(m: int) -> int:
    """Host bytes of one linear's switched hint (``m x 4096`` uint64)."""
    return int(m) * pks.NU * 8


def _result(r: Any) -> bytes | bytearray:
    if isinstance(r, BaseException):
        raise r
    return r  # type: ignore[no-any-return]


class _Batch:
    __slots__ = ("closed", "done", "entries", "error", "groups", "results")

    def __init__(self) -> None:
        self.entries: list[Any] = []
        self.groups = 0
        self.closed = False
        self.done = threading.Event()
        self.results: list[Any] | None = None
        self.error: BaseException | None = None


class _Coalescer:
    """Group commit of pks-v3 device items (``PGW_SERVICE_PKS_COALESCE_MS``, default ``0``: off).

    The first item of a (model, kernel, linear, public seed) opens a batch and waits ``window_s`` (or until the
    batch holds ``max_groups`` groups, ``PGW_SERVICE_PKS_COALESCE_GROUPS``, default 16); items arriving meanwhile
    join it, then the opener runs them all as one device item (:meth:`PKSProvider.run_items`) and hands each its
    reply. Only public data is shared between the items (the seed's hint); each reply is the item's own, bit for
    bit. The joiners' worker threads wait for the opener (their maskgen slots stay held: queueing only)."""

    def __init__(self, window_s: float, max_groups: int):
        self.window_s = float(window_s)
        self.max_groups = max(1, int(max_groups))
        self._lock = threading.Lock()
        self._open: dict[tuple[Any, ...], _Batch] = {}

    @classmethod
    def from_env(cls) -> _Coalescer | None:
        raw = os.environ.get("PGW_SERVICE_PKS_COALESCE_MS", "").strip()
        ms = float(raw) if raw else 0.0
        if ms <= 0:
            return None
        return cls(ms / 1e3, int(os.environ.get("PGW_SERVICE_PKS_COALESCE_GROUPS", "16") or 16))

    def submit(self, pv: PKSProvider, fp: str, kern: Any, it: Any, a_seed: bytes,
               dims: Mapping[str, Any] | None) -> bytes | bytearray:
        key = (fp, id(kern), it.linear, bytes(a_seed))
        g = int(it.cols) // D
        with self._lock:
            batch = self._open.get(key)
            if batch is not None and not batch.closed and batch.groups + g <= self.max_groups:
                idx = len(batch.entries)
                batch.entries.append(it)
                batch.groups += g
                if batch.groups >= self.max_groups:
                    batch.closed = True
                    self._open.pop(key, None)
                leader = False
            else:
                batch = _Batch()
                batch.entries.append(it)
                batch.groups = g
                idx, leader = 0, True
                if g >= self.max_groups:
                    batch.closed = True
                else:
                    self._open[key] = batch
        if not leader:
            batch.done.wait()
            if batch.error is not None:
                raise batch.error
            assert batch.results is not None
            return _result(batch.results[idx])
        deadline = time.monotonic() + self.window_s
        while not batch.closed:
            left = deadline - time.monotonic()
            if left <= 0:
                break
            time.sleep(min(left, 0.002))
        with self._lock:
            batch.closed = True
            if self._open.get(key) is batch:
                self._open.pop(key, None)
            entries = list(batch.entries)
        try:
            batch.results = pv.run_items(fp, kern, entries, a_seed, dims, return_exceptions=True)
        except BaseException as exc:
            batch.error = exc
            raise
        finally:
            batch.done.set()
        return _result(batch.results[0])


class PKSProvider:
    """Switched-hint cache and the v3 item computation (thread-safe).

    Where the provider-side hint ``round(W8 A q1 / 2^56) mod q1`` comes from (``PGW_SERVICE_PKS_DEVICE_HINT``):

    * ``cache``: built on the kernel's device or CPU (GEMM), switched on the host and kept in the LRU host cache
      (``PGW_SERVICE_PKS_CACHE_GB``); the item's ``y`` and the cached hint go to the key switch.
    * ``device``: nothing is cached; each item recomputes its hint rows on the kernel's device next to its own GEMM
      and key switch, row block by row block (:func:`pgw.service.pks_gpu.item`). Needs an int8 weight on the
      device and that device's self-test (bit for bit against the host path) to pass.
    * ``auto`` (default): ``device`` on CUDA kernels when one customer seed's hints for the whole model do not fit
      the cache (a decode cycle over all linears would evict every entry, so every item would rebuild its hint
      on the host: 50.5 / 157 / 230 GB per seed at 7B / 32B / 72B against 8 GB); ``cache`` otherwise.

    The model's hint size comes from ``dims`` (``{linear: (m, n)}``) when the caller passes it to
    :meth:`run_item`, else from the linears seen so far for that model. Any device error falls back to the cache
    path for good. All paths give the same bytes.
    """

    def __init__(self, cache_bytes: int | None = None, mode: str | None = None):
        if cache_bytes is None:
            cache_bytes = int(float(os.environ.get("PGW_SERVICE_PKS_CACHE_GB", "8") or 8) * 1e9)
        self.cache_bytes = max(0, int(cache_bytes))
        self.mode = hint_mode() if mode is None else mode
        if self.mode not in HINT_MODES:
            raise ValueError(f"unknown hint mode {self.mode!r}")
        self._cache: OrderedDict[tuple[str, bytes, str], np.ndarray] = OrderedDict()
        self._used = 0
        self._lock = threading.Lock()
        self._build_locks: dict[tuple[str, bytes, str], threading.Lock] = {}
        self._gpu: Any = None
        self._gpu_ok: bool | None = None
        self._item_ok: dict[str, bool] = {}  # device -> device-item self-test verdict
        self._selftest_lock = threading.Lock()
        self._model_rows: dict[str, dict[str, int]] = {}  # fp -> {linear: m}
        self.builds = 0
        self.items = 0
        self.device_items = 0
        self.coalesced_items = 0
        self.coalesced_runs = 0
        self._coalescer: _Coalescer | None = _Coalescer.from_env()

    def switched_hint(self, fp: str, kern: Any, linear: str, a_seed: bytes) -> np.ndarray:
        key = (fp, bytes(a_seed), linear)
        with self._lock:
            hs = self._cache.get(key)
            if hs is not None:
                self._cache.move_to_end(key)
                return hs
            lock = self._build_locks.setdefault(key, threading.Lock())
        with lock:
            with self._lock:
                hs = self._cache.get(key)
                if hs is not None:
                    return hs
            n = int(kern.in_features)
            a = pks.a_mlwe_rows(pks.public_ring_rows(a_seed, n, linear))  # (n, 4096) public
            h = np.asarray(kern.mulmod64(np.ascontiguousarray(a), 7), dtype=np.uint64)
            hs = switch_hint_host(h)
            del h, a
            self.builds += 1
            with self._lock:
                self._cache[key] = hs
                self._used += hs.nbytes
                while self._used > self.cache_bytes and len(self._cache) > 1:
                    _k, old = self._cache.popitem(last=False)
                    self._used -= old.nbytes
                self._build_locks.pop(key, None)
            return hs

    def _device_path(self, kern: Any) -> Any:
        dev = getattr(kern, "device", None)
        if getattr(dev, "type", "cpu") != "cuda" or not _env_bool("PGW_SERVICE_PKS_GPU", True) or self._gpu_ok is False:
            return None
        if self._gpu_ok is None:
            from pgw.service import pks_gpu

            try:
                pks_gpu.self_test(dev)
                self._gpu = pks_gpu
                self._gpu_ok = True
                log.info("maskgen v3: key switch on %s (self-test passed)", dev)
            except Exception as exc:  # pragma: no cover - GPU only
                log.warning("maskgen v3: device key switch disabled (%s); using the CPU", exc)
                self._gpu_ok = False
                return None
        return dev

    # ---- policy ---------------------------------------------------------------------------------------------
    def model_hint_bytes(self, fp: str, linear: str, m: int, dims: Mapping[str, Any] | None = None) -> int:
        """One customer seed's switched hints for the whole model ``fp`` (host bytes): from ``dims`` when given,
        else the linears of ``fp`` seen so far (a lower bound that grows to the true size)."""
        with self._lock:
            rows = self._model_rows.setdefault(fp, {})
            if dims is not None and len(rows) < len(dims):
                rows.update({str(k): int(v[0]) for k, v in dims.items()})
            rows[linear] = int(m)
            return sum(hint_bytes(r) for r in rows.values())

    def use_device(self, fp: str, kern: Any, linear: str, dims: Mapping[str, Any] | None = None) -> bool:
        """True when this item's hint is recomputed on the kernel's device (:class:`PKSProvider`'s policy)."""
        if self.mode == "cache":
            return False
        dev = getattr(kern, "device", None)
        if self.mode == "auto":
            if getattr(dev, "type", "cpu") != "cuda":
                return False
            if self.model_hint_bytes(fp, linear, int(kern.out_features), dims) <= self.cache_bytes:
                return False
        parts = _device_operands(kern)
        return parts is not None and all(self._item_device_ok(p[4]) for p in parts)

    def _item_device_ok(self, dev: Any) -> bool:
        key = str(dev)
        ok = self._item_ok.get(key)
        if ok is not None:
            return ok
        with self._selftest_lock:
            ok = self._item_ok.get(key)
            if ok is None:
                from pgw.service import pks_gpu

                try:
                    pks_gpu.item_self_test(dev)
                    ok = True
                    log.info("maskgen v3: provider hint recomputed per item on %s (self-test passed)", dev)
                except Exception as exc:
                    log.warning("maskgen v3: device hint path disabled on %s (%s); using the hint cache", dev, exc)
                    ok = False
                self._item_ok[key] = ok
        return ok

    def _run_device(self, kern: Any, it: Any, a_seed: bytes, b: Any, ops: Any) -> bytes | bytearray | None:
        out = self._run_device_out(kern, it, a_seed, b, ops)
        if out is None:
            return None
        from pgw.service import pks_gpu

        return out.buf if isinstance(out, pks_gpu.PackedReply) else pack_bits(out.reshape(-1), V3_Y_BITS)

    def _run_device_out(self, kern: Any, it: Any, a_seed: bytes, b: Any, ops: Any) -> Any:
        """The device item's reply object (a ``PackedReply`` or a ``(G, m/32, 8192)`` array), or ``None`` after a
        device failure (the caller falls back to the cache path)."""
        from pgw.service import pks_gpu

        parts = _device_operands(kern)
        assert parts is not None
        n, m = int(kern.in_features), int(kern.out_features)
        # the reply's wire bytes, filled by every shard's row blocks (packed on the device where it may)
        out: Any = pks_gpu.PackedReply(ops.shape[0], m // PACK) if pks_gpu.wire_layout_ok() else \
            np.empty((ops.shape[0], m // PACK, pks.NR), dtype=np.uint64)

        def one(part: tuple[Any, Any, int, int, Any]) -> None:
            w, const, r0, r1, dev = part
            if getattr(dev, "type", "cpu") != "cuda":
                ring = pks_gpu.ring_rows_async(a_seed, n, it.linear, dev)
                pks_gpu.item(w, const, r1 - r0, n, ring, b, ops, out=out, out_b0=r0 // PACK)
                return
            import torch  # pragma: no cover - GPU only: each concurrent item on its own stream (as mulmod64)

            from pgw.service.kernels import _borrow_stream, _return_stream

            stream = _borrow_stream(dev)
            try:
                with torch.cuda.device(dev), torch.cuda.stream(stream):
                    ring = pks_gpu.ring_rows_async(a_seed, n, it.linear, dev)
                    pks_gpu.item(w, const, r1 - r0, n, ring, b, ops, out=out, out_b0=r0 // PACK)
                    stream.synchronize()
            finally:
                _return_stream(dev, stream)

        try:
            if len(parts) == 1:
                one(parts[0])
            else:  # tensor-parallel shards: each device computes its own rows (whole reply ciphertexts)
                from concurrent.futures import ThreadPoolExecutor

                with ThreadPoolExecutor(len(parts)) as pool:
                    for f in [pool.submit(one, p) for p in parts]:
                        f.result()
        except ProtocolError:  # the customer's data refused (reply key out of range): not a device failure
            raise
        except Exception as exc:  # pragma: no cover - device errors only
            log.warning("maskgen v3: device hint path failed on %s (%s); using the hint cache from now on",
                        [str(p[4]) for p in parts], exc)
            for p in parts:
                self._item_ok[str(p[4])] = False
            return None
        self.device_items += 1
        return out

    def run_item(self, fp: str, kern: Any, it: Any, a_seed: bytes, dims: Mapping[str, Any] | None = None
                 ) -> bytes | bytearray:
        """The packed v3 reply of one item (``dims``: the model's ``{linear: (m, n)}``, for the hint policy).

        With coalescing on (``PGW_SERVICE_PKS_COALESCE_MS`` > 0, default off) and the device path chosen, the item
        waits up to that long for other items of the same model, linear, kernel and public seed (other requests:
        the same customer's or, with the opt-in common seed :func:`pgw.hemask.pks.crs_a_seed`, other customers') and
        they run as one device item (:meth:`run_items`): one hint computation for all of their groups. The bytes
        are the same as alone."""
        co = self._coalescer
        if co is not None and co.window_s > 0 and self.use_device(fp, kern, it.linear, dims):
            return co.submit(self, fp, kern, it, a_seed, dims)
        return self._run_item(fp, kern, it, a_seed, dims)

    def run_items(self, fp: str, kern: Any, items: list[Any], a_seed: bytes,
                  dims: Mapping[str, Any] | None = None, *, return_exceptions: bool = False) -> list[Any]:
        """The packed v3 replies of several items of ONE linear under ONE public seed, as :meth:`run_item` gives
        them one by one (bit for bit), computed as a single device item whose hint rows (the ``W8 A`` GEMM, the
        switch and the key switch's forward NTTs) are shared by all of their groups. Each group's reply depends only
        on its own upload, its own reply keys and the public hint, so nothing of one item enters another's reply.

        Falls back to :meth:`run_item` per item when the device path is not chosen, when there is one item, and when
        the coalesced item fails (e.g. one item's reply keys out of range: then only that item is refused; with
        ``return_exceptions`` its entry is the exception and the others still get their replies)."""

        def each() -> list[Any]:
            res: list[Any] = []
            for it in items:
                try:
                    res.append(self._run_item(fp, kern, it, a_seed, dims))
                except Exception as exc:
                    if not return_exceptions:
                        raise
                    res.append(exc)
            return res

        if len(items) <= 1 or not self.use_device(fp, kern, items[0].linear, dims):
            return each()
        from pgw.service import pks_gpu

        n, m = int(kern.in_features), int(kern.out_features)
        if n % RANK or m % PACK:
            raise ProtocolError(f"maskgen v3: {items[0].linear} ({m} x {n}) is not a pks-v3 shape")
        if len({it.linear for it in items}) != 1:
            raise ValueError("run_items: items of different linears")
        groups = [int(it.cols) // D for it in items]
        wire = pks_gpu.wire_layout_ok() and all(int(it.b_bits) == 56 for it in items)
        try:
            for it, g in zip(items, groups, strict=True):
                if int(it.cols) % D or len(it.ksk) != g * V3_KSK_BYTES:
                    raise ProtocolError("maskgen v3: reply key size mismatch")
            parts = [pks_gpu.PackedUpload(it.b, n, int(it.cols)) if wire
                     else unpack_bits(it.b, n * int(it.cols), it.b_bits).reshape(n, int(it.cols)) for it in items]
            if pks_gpu.keys_layout_ok():
                keys: Any = pks_gpu.PackedKeys(b"".join(bytes(it.ksk) for it in items), sum(groups))
            else:
                keys = np.concatenate([unpack_ksk(it.ksk, g) for it, g in zip(items, groups, strict=True)])
            out = self._run_device_out(kern, items[0], a_seed, pks_gpu.MultiUpload(parts), keys)
        except ProtocolError:  # one item's data refused: run each alone so only that one fails
            out = None
        if out is None:
            return each()
        self.items += len(items)
        self.coalesced_items += len(items)
        self.coalesced_runs += 1
        return list(pks_gpu.split_reply(out, groups))

    def _run_item(self, fp: str, kern: Any, it: Any, a_seed: bytes, dims: Mapping[str, Any] | None = None
                  ) -> bytes | bytearray:
        n, m, cols = int(kern.in_features), int(kern.out_features), int(it.cols)
        if n % RANK or m % PACK:
            raise ProtocolError(f"maskgen v3: {it.linear} ({m} x {n}) is not a pks-v3 shape")
        groups = cols // D
        if len(it.ksk) != groups * V3_KSK_BYTES:
            raise ProtocolError("maskgen v3: reply key size mismatch")
        keys: Any = None
        if self.use_device(fp, kern, it.linear, dims):
            from pgw.service import pks_gpu

            # the upload stays in its wire form: the device unpacks it where it may (``PGW_SERVICE_PKS_GPU_WIRE``)
            bw: Any = pks_gpu.PackedUpload(it.b, n, cols) if int(it.b_bits) == 56 and pks_gpu.wire_layout_ok() \
                else unpack_bits(it.b, n * cols, it.b_bits).reshape(n, cols)
            # so do the reply keys (unpacked and range-checked on the device where it may, ``PGW_SERVICE_PKS_GPU_KEYS``;
            # a residue out of range is refused before any key switch either way)
            keys = pks_gpu.PackedKeys(it.ksk, groups) if pks_gpu.keys_layout_ok() else unpack_ksk(it.ksk, groups)
            res = self._run_device(kern, it, a_seed, bw, keys)
            if res is not None:
                self.items += 1
                return res
        if keys is None:
            ops = unpack_ksk(it.ksk, groups)
        else:  # the device path fell back (its unpack is reused)
            ops = keys if isinstance(keys, np.ndarray) else keys.host()
        b = unpack_bits(it.b, n * cols, it.b_bits).reshape(n, cols)
        if int(b.max(initial=0)) >= 1 << 56:  # pragma: no cover - unpack_bits gives < 2^56
            raise ProtocolError("maskgen v3: upload value out of range")
        y = np.asarray(kern.mulmod64(b, 7), dtype=np.uint64) & _M56
        del b
        hs = self.switched_hint(fp, kern, it.linear, a_seed)
        dev = self._device_path(kern)
        out: np.ndarray | None = None
        if dev is not None:
            try:
                out = self._gpu.keyswitch(hs, ops, y, dev)
            except Exception as exc:  # pragma: no cover - GPU only
                log.warning("maskgen v3: device key switch failed (%s); using the CPU from now on", exc)
                self._gpu_ok = False
        if out is None:
            from pgw.hemask.pks_native import keyswitch

            out = keyswitch(hs, ops, y)
        self.items += 1
        return pack_bits(out.reshape(-1), V3_Y_BITS)
