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

    def _run_device(self, kern: Any, it: Any, a_seed: bytes, b: Any, ops: np.ndarray) -> bytes | bytearray | None:
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
        except Exception as exc:  # pragma: no cover - device errors only
            log.warning("maskgen v3: device hint path failed on %s (%s); using the hint cache from now on",
                        [str(p[4]) for p in parts], exc)
            for p in parts:
                self._item_ok[str(p[4])] = False
            return None
        self.device_items += 1
        return out.buf if isinstance(out, pks_gpu.PackedReply) else pack_bits(out.reshape(-1), V3_Y_BITS)

    def run_item(self, fp: str, kern: Any, it: Any, a_seed: bytes, dims: Mapping[str, Any] | None = None
                 ) -> bytes | bytearray:
        """The packed v3 reply of one item (``dims``: the model's ``{linear: (m, n)}``, for the hint policy)."""
        n, m, cols = int(kern.in_features), int(kern.out_features), int(it.cols)
        if n % RANK or m % PACK:
            raise ProtocolError(f"maskgen v3: {it.linear} ({m} x {n}) is not a pks-v3 shape")
        groups = cols // D
        ops = unpack_ksk(it.ksk, groups)
        if self.use_device(fp, kern, it.linear, dims):
            from pgw.service import pks_gpu

            # the upload stays in its wire form: the device unpacks it where it may (``PGW_SERVICE_PKS_GPU_WIRE``)
            bw: Any = pks_gpu.PackedUpload(it.b, n, cols) if int(it.b_bits) == 56 and pks_gpu.wire_layout_ok() \
                else unpack_bits(it.b, n * cols, it.b_bits).reshape(n, cols)
            res = self._run_device(kern, it, a_seed, bw, ops)
            if res is not None:
                self.items += 1
                return res
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
