"""Factory backend of the compact-hint scheme ``pks-v3`` (opt-in: ``PGW_FACTORY_HE_SCHEME=pks-v3`` plus
``PGW_FACTORY_HE_PKS_ACK`` = :data:`pgw.hemask.pks.PKS_V3_ACK` until the external review signs off).

:class:`PKSMaskBackend` has the interface :class:`pgw.factory.MaskGenerator` uses for remote products
(``matmul(linear, r) -> W8 r mod 2^32``), so the factory's segments, its every-row self-check and the online
Freivalds / p31 check vectors are exactly as with the v1/v2 scheme; only the encryption and the wire change.

* **Hint** (:class:`PKSHintStore`, keyed by scheme and reply-ring degree): built once by the customer from its
  own ``W8`` and two seeds it draws itself (the upload seed ``a_seed`` and the key-switching seed ``ks_seed``),
  before any reply secret exists (hint integrity, plan section 7.4). The provider never contributes to it.
  Every linear's file is SHA-256 hashed at build and verified when the store is opened.
* **Per request** (one linear, up to ``groups_per_call`` 256-slot groups): fresh secrets per group
  (:class:`pgw.hemask.pks_native.Customer`), one wire v3 request (``b`` 56-bit, reply keys), decryption with
  the compact hint; secrets are wiped after decryption.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pgw.core.errors import ConfigError, ErrorCode, IntegrityError, PGWError
from pgw.core.prng import new_seed
from pgw.hemask import pks
from pgw.hemask.pks import NR, PACK, RANK, SCHEME_ID, D, PKSParams
from pgw.hemask.pks_native import Customer, keyswitch
from pgw.hemask.wire import (
    MASKGEN_PATH,
    V3_B_BITS,
    V3_Y_BITS,
    MaskgenItem,
    MaskgenRequest,
    decode_maskgen_response,
    encode_maskgen_request,
    pack_bits,
)

__all__ = ["InProcessPKSProvider", "PKSHintStore", "PKSMaskBackend", "RemotePKS"]

log = logging.getLogger(__name__)

_META = "meta.json"
_STORE_VERSION = 1
_M56 = np.uint64((1 << 56) - 1)
_ROW_BLOCK = 1024  # hint rows per H block during the build (bounds transient memory: 1024 x 4096 x 8 B)


def _check_shape(key: str, m: int, n: int) -> None:
    if m % PACK or n % RANK:
        raise ConfigError(f"pks-v3 needs out_features % {PACK} == 0 and in_features % {RANK} == 0; {key} is {m} x {n}")


class PKSHintStore:
    """Compact hints ``c1`` of a pack: ``<root>/<fingerprint[:16]>-pks-v3-n8192/{meta.json, <linear>.npy}``.

    Each ``.npy`` is ``(m/32, 2, 8192, 3)`` uint8 (NTT-domain residues mod the two q1 primes, 24 bits each:
    1.5 KB per output row). ``meta.json``: scheme, ring degree, fingerprint, the customer's seeds (public),
    each linear's largest weight-row norm (noise check) and file digest.
    """

    def __init__(self, path: Path, meta: dict[str, Any]):
        self.path = path
        self.meta = meta
        self.a_seed = bytes.fromhex(meta["a_seed"])
        self.ks_seed = bytes.fromhex(meta["ks_seed"])
        self._maps: dict[str, np.ndarray] = {}

    @staticmethod
    def dir_for(root: Path, fingerprint: str) -> Path:
        return Path(root) / f"{fingerprint[:16]}-{SCHEME_ID}-n{NR}"

    @classmethod
    def open(cls, root: Path, fingerprint: str, *, verify: bool = True) -> PKSHintStore | None:
        path = cls.dir_for(root, fingerprint)
        try:
            meta = json.loads((path / _META).read_text())
        except FileNotFoundError:
            return None
        if meta.get("version") != _STORE_VERSION or meta.get("scheme") != SCHEME_ID or meta.get("nr") != NR:
            log.warning("pks-v3 hint store %s has another format: rebuilding", path)
            return None
        if meta.get("fingerprint") != fingerprint:
            raise ConfigError(f"hint store {path} does not match the pack fingerprint; delete it to rebuild")
        hs = cls(path, meta)
        if verify:
            for key, want in meta["digests"].items():
                if _digest(path / f"{key}.npy") != want:
                    raise IntegrityError(f"pks-v3 hint {path / key}.npy does not match its build digest; "
                                         "delete the store to rebuild")
        return hs

    @classmethod
    def build(cls, root: Path, pack: Any, *, only: set[str] | None = None, log_every_s: float = 30.0) -> PKSHintStore:
        """Compute every linear's compact hint (``only``: these linear keys) from the pack (the customer's own
        weights) and fresh seeds."""
        from pgw.hemask.lwe import matmul_mod64

        par = PKSParams()
        final = cls.dir_for(root, pack.fingerprint)
        tmp = final.with_name(final.name + f".building-{os.getpid()}")
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        a_seed, ks_seed = new_seed(), new_seed()  # customer-chosen; both public
        k_hat = pks.ks_a(ks_seed)[None].astype(np.uint32)
        norms: dict[str, float] = {}
        digests: dict[str, str] = {}
        t0 = last = time.monotonic()
        try:
            for lw in pack.iter_linears():
                key = lw.id.key
                if only is not None and key not in only:
                    continue
                w = lw.w8.numpy() if isinstance(lw.w8, torch.Tensor) else np.asarray(lw.w8)
                m, n = w.shape
                _check_shape(key, m, n)
                norms[key] = float(np.sqrt((w.astype(np.float64) ** 2).sum(1)).max())
                a = pks.a_mlwe_rows(pks.public_ring_rows(a_seed, n, key))
                out = np.lib.format.open_memmap(tmp / f"{key}.npy", mode="w+", dtype=np.uint8,
                                                shape=(m // PACK, 2, NR, 3))
                for r0 in range(0, m, _ROW_BLOCK):
                    h = matmul_mod64(np.ascontiguousarray(w[r0:r0 + _ROW_BLOCK]), a) & _M56
                    out[r0 // PACK:(r0 + h.shape[0]) // PACK] = keyswitch(pks.switch_hint(h, par), k_hat)
                    del h
                out.flush()
                del out, a
                digests[key] = _digest(tmp / f"{key}.npy")
                if time.monotonic() - last > log_every_s:
                    last = time.monotonic()
                    log.info("building pks-v3 hint: %s (%.0f s)", key, last - t0)
            meta = {"version": _STORE_VERSION, "scheme": SCHEME_ID, "nr": NR, "fingerprint": pack.fingerprint,
                    "a_seed": a_seed.hex(), "ks_seed": ks_seed.hex(), "row_norms": norms, "digests": digests,
                    "created": time.time()}
            (tmp / _META).write_text(json.dumps(meta, indent=1))
            if final.exists():
                shutil.rmtree(final)
            tmp.rename(final)
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        st = cls(final, meta)
        log.info("pks-v3 hint for %s built in %.0f s: %.2f GB in %s", pack.fingerprint[:12], time.monotonic() - t0,
                 st.nbytes / 1e9, final)
        return st

    @classmethod
    def open_or_build(cls, root: Path, pack: Any, *, only: set[str] | None = None) -> PKSHintStore:
        st = cls.open(root, pack.fingerprint)
        if st is not None and (only is None or only <= set(st.meta["row_norms"])):
            return st
        return cls.build(root, pack, only=only)

    def hint(self, linear: str) -> np.ndarray:
        h = self._maps.get(linear)
        if h is None:
            h = np.load(self.path / f"{linear}.npy", mmap_mode="r")
            self._maps[linear] = h
        return h

    @property
    def nbytes(self) -> int:
        return sum(p.stat().st_size for p in self.path.glob("*.npy"))


def _kind(spec: Any) -> str:
    k = getattr(spec.id, "kind", None)
    return str(getattr(k, "value", k or ""))


def linear_keys(pack: Any, skip_kinds: frozenset[str] | set[str] = frozenset()) -> set[str]:
    """The linear keys the factory generates masks for (``skip_kinds`` left out)."""
    return {s.id.key for s in pack.linear_specs() if _kind(s) not in skip_kinds}


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 22):
            h.update(chunk)
    return h.hexdigest()


class RemotePKS:
    """Maskgen wire v3 requests for one pack (``post(body, request_id, decode)`` sends them: the HTTP
    :class:`pgw.wire.ComputeClient` or an in-process provider). A service that does not serve v3 is a
    configuration error (the customer's ciphertexts are pks-v3: there is no v1/v2 fallback for them)."""

    VERSIONS = (1, 2, 3)

    def __init__(self, post: Callable[[bytes, str, Callable[[bytes], Any]], Any], fingerprint: str,
                 out_features: dict[str, int], a_seed: bytes):
        self.post = post
        self.fingerprint = fingerprint
        self.out_features = dict(out_features)
        self.a_seed = bytes(a_seed)
        self.bytes_up = self.bytes_down = self.calls = 0

    @classmethod
    def http(cls, client: Any, fingerprint: str, out_features: dict[str, int], a_seed: bytes, *,
             timeout_s: float = 600.0) -> RemotePKS:
        def post(body: bytes, rid: str, decode: Callable[[bytes], Any]) -> Any:
            return client.post_raw_sync(MASKGEN_PATH, body, rid, decode, timeout_s=timeout_s)

        return cls(post, fingerprint, out_features, a_seed)

    def __call__(self, linear: str, b: np.ndarray, ksk: bytes | bytearray) -> bytes:
        cols = int(b.shape[1])
        item = MaskgenItem(linear, cols, pack_bits(b, V3_B_BITS), 0, 64, V3_B_BITS, V3_Y_BITS, ksk)
        req = MaskgenRequest(uuid.uuid4().hex, self.fingerprint, (item,), 3, self.VERSIONS, self.a_seed)
        body = encode_maskgen_request(req)
        self.calls += 1
        self.bytes_up += len(body)

        def decode(data: bytes) -> bytes:
            resp = decode_maskgen_response(data, req, self.out_features)
            self.bytes_down += len(data)
            return bytes(resp.items[0].y)

        try:
            out: bytes = self.post(body, req.request_id, decode)
        except PGWError as exc:
            if exc.code in (ErrorCode.VERSION_MISMATCH, ErrorCode.PROTOCOL):
                raise ConfigError(f"the compute service refused maskgen v3 ({exc.code}: {exc.message}); pks-v3 needs "
                                  "PGW_SERVICE_MASKGEN_V3=true on the provider") from exc
            raise
        return out


class InProcessPKSProvider:
    """The service's v3 item computation (:class:`pgw.service.pks_provider.PKSProvider`) called in this
    process: wire bytes in and out, everything the service does but HTTP (tests, local benchmarks)."""

    def __init__(self, pack: Any, kernels: dict[str, Any] | None = None, device: str = "cpu"):
        from pgw.service.kernels import ComputeKernel
        from pgw.service.pks_provider import PKSProvider

        self.fp = pack.fingerprint
        self.kern = kernels or {}
        if not self.kern:
            for lw in pack.iter_linears():
                spec = next(s for s in pack.linear_specs() if s.id.key == lw.id.key)
                w = lw.w8 if isinstance(lw.w8, torch.Tensor) else torch.from_numpy(np.asarray(lw.w8))
                self.kern[lw.id.key] = ComputeKernel(spec, w, device=device)
        self.provider = PKSProvider()

    def __call__(self, body: bytes, rid: str, decode: Callable[[bytes], Any]) -> Any:
        from pgw.hemask.wire import MaskgenResponse, MaskgenResult, decode_maskgen_request, encode_maskgen_response

        req = decode_maskgen_request(body, (1, 2, 3))
        out = [MaskgenResult(it.linear, it.cols, self.provider.run_item(self.fp, self.kern[it.linear], it, req.a_seed))
               for it in req.items]
        return decode(encode_maskgen_response(MaskgenResponse(req.request_id, tuple(out), req.version, (1, 2, 3))))


class PKSMaskBackend:
    """``W8 r mod 2^32`` via the provider with scheme pks-v3, for one pack and a K32 ring."""

    decrypt_mode = "pks-v3"

    def __init__(self, pack: Any, ring: Any, store: PKSHintStore, remote: Callable[..., bytes], *,
                 groups_per_call: int = 4, threads: int | None = None, margin_bits: float = 1.0,
                 skip_kinds: frozenset[str] | set[str] = frozenset()):
        if int(ring.k) != 32:
            raise ConfigError(f"pks-v3 covers K32 mask rings only, not {ring.name}")
        self.params = PKSParams()
        self.ring = ring
        self.store = store
        self.remote = remote
        self.groups_per_call = max(1, min(16, int(groups_per_call)))
        self.specs = {s.id.key: s for s in pack.linear_specs() if _kind(s) not in skip_kinds}
        norms = store.meta["row_norms"]
        for key, s in self.specs.items():
            _check_shape(key, s.out_features, s.in_features)
            if key not in norms:
                raise ConfigError(f"pks-v3 hint store has no hint for {key}; delete it to rebuild")
            self.params.check_noise(float(norms[key]), key, margin_bits)  # refuse before any request
        self.customer = Customer(pks.ks_a(store.ks_seed), threads=threads)
        self.engine = f"pks-v3:{self.customer.engine}"
        self.stats = {"groups": 0, "encrypt_s": 0.0, "remote_s": 0.0, "decrypt_s": 0.0, "wall_s": 0.0, "passes": 0,
                      "max_inflight": 1, "jobs": 0, "yield_s": 0.0}
        worst = max(self.params.noise_bound(float(norms[k])) for k in self.specs)
        log.info("encrypted mask factory: scheme pks-v3 (reply ring %d, hint %.2f GB, %s; worst noise bound 2^%.2f "
                 "< 2^%.2f)", NR, store.nbytes / 1e9, self.customer.engine, np.log2(worst),
                 np.log2(self.params.budget(margin_bits)))

    def preferred_chunk_slots(self) -> int:
        return D * self.groups_per_call

    def matmul(self, linear: str, r: torch.Tensor) -> torch.Tensor:
        from pgw.service.pks_provider import pack_ksk

        spec = self.specs[linear]
        rn = r.detach().cpu().numpy().astype(np.int64, copy=False)
        if rn.ndim != 2 or rn.shape[1] != spec.in_features:
            raise ConfigError(f"{linear}: r must be (rows, {spec.in_features})")
        t_wall = time.perf_counter()
        hint = self.store.hint(linear)
        parts = []
        step = D * self.groups_per_call
        for r0 in range(0, rn.shape[0], step):
            rc = np.ascontiguousarray(rn[r0:r0 + step])
            t0 = time.perf_counter()
            b, beta, secrets = self.customer.encrypt(self.store.a_seed, linear, rc)
            ksk = pack_ksk(beta)
            del beta
            t1 = time.perf_counter()
            try:
                y = self.remote(linear, b, ksk)
                t2 = time.perf_counter()
                wr = self.customer.decrypt(hint, y, secrets)
            finally:
                for s in secrets:
                    s.wipe()
            t3 = time.perf_counter()
            parts.append(wr)
            self._add(encrypt_s=t1 - t0, remote_s=t2 - t1, decrypt_s=t3 - t2, groups=len(secrets), passes=1, jobs=1)
        self._add(wall_s=time.perf_counter() - t_wall)
        out = np.concatenate(parts, axis=0) if len(parts) > 1 else parts[0]
        return torch.from_numpy(out.astype(np.int64))

    def _add(self, **kw: float) -> None:
        for k, v in kw.items():
            self.stats[k] = self.stats.get(k, 0.0) + v
