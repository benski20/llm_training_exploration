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
"""

from __future__ import annotations

import logging
import os
import threading
from collections import OrderedDict
from typing import Any

import numpy as np

from pgw.core.errors import ProtocolError
from pgw.hemask import pks
from pgw.hemask.pks import NR, PACK, RANK, D, PKSParams
from pgw.hemask.wire import V3_KSK_BITS, V3_KSK_BYTES, V3_Y_BITS, pack_bits, unpack_bits

__all__ = ["PKSProvider", "unpack_ksk"]

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
    """Inverse of :func:`unpack_ksk` (customer side)."""
    out = bytearray()
    for g in range(beta.shape[0]):
        for i, bits in enumerate(V3_KSK_BITS):
            out += pack_bits(np.ascontiguousarray(beta[g, i], dtype=np.uint64).reshape(-1), bits)
    return out


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


class PKSProvider:
    """Switched-hint cache and the v3 item computation (thread-safe)."""

    def __init__(self, cache_bytes: int | None = None):
        if cache_bytes is None:
            cache_bytes = int(float(os.environ.get("PGW_SERVICE_PKS_CACHE_GB", "8") or 8) * 1e9)
        self.cache_bytes = max(0, int(cache_bytes))
        self._cache: OrderedDict[tuple[str, bytes, str], np.ndarray] = OrderedDict()
        self._used = 0
        self._lock = threading.Lock()
        self._build_locks: dict[tuple[str, bytes, str], threading.Lock] = {}
        self._gpu: Any = None
        self._gpu_ok: bool | None = None
        self.builds = 0
        self.items = 0

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
            h = np.asarray(kern.mulmod64(np.ascontiguousarray(a), 7), dtype=np.uint64) & _M56
            hs = pks.switch_hint(h, _PAR)
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

    def run_item(self, fp: str, kern: Any, it: Any, a_seed: bytes) -> bytes | bytearray:
        """The packed v3 reply of one item."""
        n, m, cols = int(kern.in_features), int(kern.out_features), int(it.cols)
        if n % RANK or m % PACK:
            raise ProtocolError(f"maskgen v3: {it.linear} ({m} x {n}) is not a pks-v3 shape")
        groups = cols // D
        ops = unpack_ksk(it.ksk, groups)
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
