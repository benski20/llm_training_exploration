"""Factory backend of the compact-hint scheme ``pks-v3`` (opt-in: ``PGW_FACTORY_HE_SCHEME=pks-v3`` plus
``PGW_FACTORY_HE_PKS_ACK`` = :data:`pgw.hemask.pks.PKS_V3_ACK` until the external review signs off).

:class:`PKSMaskBackend` has the interface :class:`pgw.factory.MaskGenerator` uses for remote products
(``matmul(linear, r) -> W8 r mod 2^32``), so the factory's segments, its every-row self-check and the online
Freivalds / p31 check vectors are exactly as with the v1/v2 scheme; only the encryption and the wire change.

* **Hint** (:class:`PKSHintStore`, keyed by scheme and reply-ring degree): built once by the customer from its
  own ``W8`` and two seeds it draws itself (the upload seed ``a_seed`` and the key-switching seed ``ks_seed``),
  before any reply secret exists (hint integrity, plan section 7.4). The provider never contributes to it.
  Every linear's file is SHA-256 hashed at build and verified when the store is opened. Trust boundary: the
  digests live in the store's own ``meta.json``, next to the public seeds and the row norms that drive the noise
  refusal, so on their own they detect corruption and partial writes, not a writer of the hint directory.
  With a customer key (``mac_key``; the factory's ``he_hint_key_file``, kept outside the hint directory)
  ``meta.json`` is authenticated with HMAC-SHA256 and a store whose metadata (seeds, row norms, digests) was
  changed without the key is refused. Without it, the hint directory must be inside the customer's trust
  boundary (pks-v3 adversarial review, F6).
* **Concurrency**: a process that opened a store holds a shared ``flock`` on ``<store>.lock`` for its lifetime;
  a rebuild needs the exclusive lock, so it never replaces seeds under a running factory (F7), and concurrent
  builders of one store are serialised.
* **Per request** (one linear, up to ``groups_per_call`` 256-slot groups): fresh secrets per group
  (:class:`pgw.hemask.pks_native.Customer`), one wire v3 request (``b`` 56-bit, reply keys), decryption with
  the compact hint; secrets are wiped after decryption.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import hmac
import json
import logging
import os
import shutil
import time
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pgw.core.errors import ConfigError, ErrorCode, IntegrityError, PGWError
from pgw.core.prng import new_seed
from pgw.hemask import pks
from pgw.hemask.pks import NR, PACK, RANK, SCHEME_ID, D, PKSParams
from pgw.hemask.pks_native import Customer
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

__all__ = ["HintBuild", "HintLinear", "InProcessPKSProvider", "PKSHintStore", "PKSMaskBackend", "RemotePKS", "row_norm",
           "switch_hint_fast"]

log = logging.getLogger(__name__)

_META = "meta.json"
_STORE_VERSION = 1

_ROW_BLOCK = 1024  # hint rows per H block during the build (bounds transient memory: 1024 x 4096 x 8 B)


def _check_shape(key: str, m: int, n: int) -> None:
    if m % PACK or n % RANK:
        raise ConfigError(f"pks-v3 needs out_features % {PACK} == 0 and in_features % {RANK} == 0; {key} is {m} x {n}")


@dataclass
class HintLinear:
    """One linear for :meth:`PKSHintStore.write`: shape, its largest weight-row norm (a float, or a callable read
    after ``blocks`` is exhausted) and the ``(r0, H rows)`` blocks of ``H = W8 A mod 2^56``."""

    key: str
    m: int
    n: int
    norm: float | Callable[[], float]
    blocks: Iterable[tuple[int, np.ndarray]]


def row_norm(w: np.ndarray) -> float:
    """The store's ``row_norms`` entry: ``max_j ||W8_j||_2`` in float64 (exactly as the original build)."""
    return float(np.sqrt((np.asarray(w).astype(np.float64) ** 2).sum(1)).max())


def switch_hint_fast(h: np.ndarray, threads: int | None = None) -> np.ndarray:
    """:func:`pgw.hemask.pks.switch_hint` on row blocks in threads (numpy releases the GIL in its ufuncs): the
    same function on every row, so bit-identical (``tests/provision/test_hintcheck.py``)."""
    from concurrent.futures import ThreadPoolExecutor

    h = np.asarray(h, dtype=np.uint64)
    par = PKSParams()
    nt = max(1, int(threads or os.cpu_count() or 1))
    from pgw.hemask.pks_native import native

    nat = native()
    if nat is not None and hasattr(nat, "switch_hint") and h.ndim == 2:  # same values (tests/hemask/test_hint_build)
        out_n: np.ndarray = nat.switch_hint(np.ascontiguousarray(h), nt)
        return out_n
    if nt == 1 or h.shape[0] < 2 * nt:
        return pks.switch_hint(h, par)
    out = np.empty(h.shape, dtype=np.uint64)
    step = -(-h.shape[0] // nt)

    def run(r0: int) -> None:
        out[r0:r0 + step] = pks.switch_hint(h[r0:r0 + step], par)

    with ThreadPoolExecutor(max_workers=nt) as ex:
        list(ex.map(run, range(0, h.shape[0], step)))
    return out


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
        #: The shared lock on ``<store>.lock`` (held while this object lives; :meth:`open_or_build`).
        self._lock_fd: int | None = None

    def close(self) -> None:
        """Release the shared store lock (mapped hints stay readable until dropped)."""
        fd, self._lock_fd = self._lock_fd, None
        if fd is not None:
            with contextlib.suppress(OSError):
                os.close(fd)

    def __del__(self) -> None:
        self.close()

    @staticmethod
    def lock_path(root: Path, fingerprint: str) -> Path:
        d = PKSHintStore.dir_for(root, fingerprint)
        return d.with_name(d.name + ".lock")

    @staticmethod
    def dir_for(root: Path, fingerprint: str) -> Path:
        return Path(root) / f"{fingerprint[:16]}-{SCHEME_ID}-n{NR}"

    @classmethod
    def open(cls, root: Path, fingerprint: str, *, verify: bool = True,
             mac_key: bytes | None = None) -> PKSHintStore | None:
        """Open the store of ``fingerprint`` (``None`` if absent or of another format). With ``mac_key`` the
        ``meta.json`` HMAC must verify (else :class:`IntegrityError`)."""
        path = cls.dir_for(root, fingerprint)
        try:
            meta = json.loads((path / _META).read_text())
        except FileNotFoundError:
            return None
        if mac_key is not None:
            want = meta.get("mac")
            if want is None:  # built before a key was configured: rebuild with fresh seeds (never trust it)
                log.warning("pks-v3 hint store %s has no meta.json MAC but a hint key is configured: rebuilding", path)
                return None
            if not isinstance(want, str) or not hmac.compare_digest(want, _meta_mac(mac_key, meta)):
                raise IntegrityError(f"pks-v3 hint store {path}: meta.json is not authenticated by the configured "
                                     "hint key (changed, or built without the key); delete the store to rebuild")
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
    def build(cls, root: Path, pack: Any, *, only: set[str] | None = None, log_every_s: float = 30.0,
              mac_key: bytes | None = None, seeds: tuple[bytes, bytes] | None = None) -> PKSHintStore:
        """Compute every linear's compact hint (``only``: these linear keys) from the pack (the customer's own
        weights) and fresh seeds (``seeds = (a_seed, ks_seed)``: given ones, e.g. a provisioning comparison)."""
        from pgw.hemask.hint_gemm import HintGemm

        a_seed, ks_seed = seeds if seeds is not None else (new_seed(), new_seed())  # customer-chosen; both public

        def items() -> Any:
            for lw in pack.iter_linears():
                key = lw.id.key
                if only is not None and key not in only:
                    continue
                w = lw.w8.numpy() if isinstance(lw.w8, torch.Tensor) else np.asarray(lw.w8)
                m, n = w.shape
                _check_shape(key, m, n)

                def blocks(w: np.ndarray = w, key: str = key, n: int = n) -> Any:
                    gemm = HintGemm(a_seed, n, key)  # A's byte planes once per linear
                    try:
                        for r0 in range(0, w.shape[0], _ROW_BLOCK):
                            yield r0, gemm(w[r0:r0 + _ROW_BLOCK])
                    finally:
                        gemm.close()

                yield HintLinear(key, m, n, row_norm(w), blocks())

        return cls.write(root, pack.fingerprint, a_seed, ks_seed, items(), mac_key=mac_key, log_every_s=log_every_s)

    @classmethod
    def write(cls, root: Path, fingerprint: str, a_seed: bytes, ks_seed: bytes, linears: Any, *,
              mac_key: bytes | None = None, extra_meta: dict[str, Any] | None = None,
              log_every_s: float = 30.0) -> PKSHintStore:
        """Write a store from ``H = W8 A mod 2^56`` row blocks (the shared tail of every build).

        ``linears`` yields :class:`HintLinear` items; each item's ``blocks`` yields ``(r0, H rows)`` with row counts
        that are multiples of 32 and may raise (e.g. a failed provider check) after its last block: then nothing
        is renamed into place. The hint of a block is ``keyswitch(switch_hint(H), k)``, as in the original build."""
        hb = HintBuild(root, fingerprint, mac_key=mac_key, seeds=(a_seed, ks_seed), resumable=False)
        try:
            for item in linears:
                hb.add(item, log_every_s=log_every_s)
            return hb.finish(extra_meta=extra_meta)
        except BaseException:
            hb.abort()
            raise

    @classmethod
    def open_or_build(cls, root: Path, pack: Any, *, only: set[str] | None = None,
                      mac_key: bytes | None = None) -> PKSHintStore:
        """Open the pack's store, or build it; the returned store holds a shared lock on ``<store>.lock``.

        A (re)build takes the exclusive lock: it waits for a concurrent builder and refuses
        (:class:`ConfigError`) while another process has the store open, instead of replacing its seeds."""
        Path(root).mkdir(parents=True, exist_ok=True)
        fd = os.open(cls.lock_path(root, pack.fingerprint), os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_SH)  # waits for a running build
            st = cls.open(root, pack.fingerprint, mac_key=mac_key)
            if st is None or not (only is None or only <= set(st.meta["row_norms"])):
                fcntl.flock(fd, fcntl.LOCK_UN)
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    # a peer is building it (then a shared lock waits for that build) or has it open
                    fcntl.flock(fd, fcntl.LOCK_SH)
                    st = cls.open(root, pack.fingerprint, mac_key=mac_key)
                    if st is not None and (only is None or only <= set(st.meta["row_norms"])):
                        st._lock_fd = fd
                        return st
                    raise ConfigError(f"pks-v3 hint store {cls.dir_for(root, pack.fingerprint)} needs a rebuild but "
                                      "another process holds it open; stop it first") from None
                st = cls.open(root, pack.fingerprint, mac_key=mac_key)  # built by a peer meanwhile?
                if st is None or not (only is None or only <= set(st.meta["row_norms"])):
                    st = cls.build(root, pack, only=only, mac_key=mac_key)
                fcntl.flock(fd, fcntl.LOCK_SH)
            st._lock_fd = fd
            return st
        except BaseException:
            os.close(fd)
            raise

    def hint(self, linear: str) -> np.ndarray:
        h = self._maps.get(linear)
        if h is None:
            h = np.load(self.path / f"{linear}.npy", mmap_mode="r")
            self._maps[linear] = h
        return h

    @property
    def nbytes(self) -> int:
        return sum(p.stat().st_size for p in self.path.glob("*.npy"))


_PARTIAL = "partial.json"


class HintBuild:
    """Writes a hint store one linear at a time, then ``meta.json`` and the rename into place
    (:meth:`PKSHintStore.write` is this with ``resumable=False``; same files and metadata).

    * ``resumable=False``: a private temporary directory, removed by :meth:`abort`.
    * ``resumable=True`` (needs ``mac_key``; the one-time ``pgw sdk prepare`` / provisioning build of a large model):
      the directory ``<store>.partial`` survives an interruption. ``partial.json`` holds the seeds and, per finished
      linear, its file's SHA-256 and row norm, authenticated with HMAC-SHA256 under the customer's hint key (its own
      domain label, so it can never pass as a ``meta.json``). The next build with the same key and fingerprint
      continues from it: every finished linear's file is re-hashed and must equal its authenticated digest (else it
      is rebuilt), and the authenticated seeds are reused. A state that does not verify (another key, edited,
      another fingerprint or format) is discarded and the build starts afresh with new seeds. A linear's file is
      written under a temporary name and renamed only after it is complete and flushed, so a crash never leaves a
      half-written file listed as done. The store becomes visible (``meta.json``, rename) only after every linear,
      exactly as a one-shot build: the hint exists before any reply secret, as the hint-integrity rule requires.

    The caller holds the store's exclusive lock (:func:`pgw.provision.client._build_store`)."""

    def __init__(self, root: Path, fingerprint: str, *, mac_key: bytes | None = None,
                 seeds: tuple[bytes, bytes] | None = None, resumable: bool = False):
        self.final = PKSHintStore.dir_for(root, fingerprint)
        self.fingerprint = fingerprint
        self.mac_key = None if mac_key is None else bytes(mac_key)
        self.resumable = bool(resumable and mac_key is not None)
        self.norms: dict[str, float] = {}
        self.digests: dict[str, str] = {}
        state: dict[str, Any] | None = None
        if self.resumable:
            self.dir = self.final.with_name(self.final.name + ".partial")
            state = self._load_state() if seeds is None else None
            if state is None:
                shutil.rmtree(self.dir, ignore_errors=True)
        else:
            self.dir = self.final.with_name(self.final.name + f".building-{os.getpid()}-{uuid.uuid4().hex[:8]}")
            shutil.rmtree(self.dir, ignore_errors=True)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.resumed = 0
        if state is not None:
            self.a_seed, self.ks_seed = bytes.fromhex(state["a_seed"]), bytes.fromhex(state["ks_seed"])
            for key, d in state["done"].items():
                p = self.dir / f"{key}.npy"
                if p.is_file() and hmac.compare_digest(_digest(p), str(d["digest"])):
                    self.norms[key], self.digests[key] = float(d["norm"]), str(d["digest"])
                else:
                    log.warning("pks-v3 hint build: %s in %s does not match its recorded digest; rebuilding it",
                                key, self.dir)
                    p.unlink(missing_ok=True)
            self.resumed = len(self.digests)
            if self.resumed:
                log.info("pks-v3 hint build: resuming %s with %d finished linear(s)", self.dir, self.resumed)
        else:
            self.a_seed, self.ks_seed = seeds if seeds is not None else (new_seed(), new_seed())
        self._save_state()
        self.k_hat = pks.ks_a(self.ks_seed)[None].astype(np.uint32)
        self.t0 = self._last = time.monotonic()

    # ---- resumable state ------------------------------------------------------------------------
    def _state(self) -> dict[str, Any]:
        return {"version": _STORE_VERSION, "scheme": SCHEME_ID, "nr": NR, "fingerprint": self.fingerprint,
                "a_seed": self.a_seed.hex(), "ks_seed": self.ks_seed.hex(),
                "done": {k: {"digest": self.digests[k], "norm": self.norms[k]} for k in self.digests}}

    def _load_state(self) -> dict[str, Any] | None:
        try:
            st = json.loads((self.dir / _PARTIAL).read_text())
        except (OSError, ValueError):
            return None
        assert self.mac_key is not None
        want = st.get("mac") if isinstance(st, dict) else None
        if not isinstance(want, str) or not hmac.compare_digest(want, _meta_mac(self.mac_key, st, _PARTIAL_DOMAIN)):
            log.warning("pks-v3 hint build state %s is not authenticated by the hint key: starting afresh", self.dir)
            return None
        if (st.get("version"), st.get("scheme"), st.get("nr"), st.get("fingerprint")) != (
                _STORE_VERSION, SCHEME_ID, NR, self.fingerprint) or not isinstance(st.get("done"), dict):
            return None
        out: dict[str, Any] = st
        return out

    def _save_state(self) -> None:
        if not self.resumable:
            return
        assert self.mac_key is not None
        st = self._state()
        st["mac"] = _meta_mac(self.mac_key, st, _PARTIAL_DOMAIN)
        tmp = self.dir / f".{_PARTIAL}.tmp"
        with open(tmp, "w") as f:
            f.write(json.dumps(st, indent=1))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.dir / _PARTIAL)

    # ---- building -------------------------------------------------------------------------------
    def done(self, key: str) -> bool:
        return key in self.digests

    def add(self, item: HintLinear, *, log_every_s: float = 30.0) -> None:
        """Compute and store one linear's hint from its ``H`` blocks (skipped when already done)."""
        from pgw.hemask.hint_gemm import hint_rows

        key, m = item.key, item.m
        if key in self.digests:
            close = getattr(item.blocks, "close", None)
            if close is not None:
                close()
            return
        _check_shape(key, m, item.n)
        part = self.dir / f"{key}.npy.tmp"
        out = np.lib.format.open_memmap(part, mode="w+", dtype=np.uint8, shape=(m // PACK, 2, NR, 3))
        done = 0
        for r0, h in item.blocks:
            rows = int(h.shape[0])
            if r0 % PACK or rows % PACK or r0 + rows > m:
                raise ConfigError(f"{key}: hint block [{r0}, {r0 + rows}) is not 32-row aligned")
            out[r0 // PACK:(r0 + rows) // PACK] = hint_rows(h, self.k_hat)  # keyswitch(switch_hint(H), k)
            done += rows
            del h
        if done != m:
            raise ConfigError(f"{key}: hint blocks cover {done} of {m} rows")
        out.flush()
        del out
        if self.resumable:
            with open(part, "rb") as f:
                os.fsync(f.fileno())
        os.replace(part, self.dir / f"{key}.npy")
        self.norms[key] = float(item.norm if not callable(item.norm) else item.norm())
        self.digests[key] = _digest(self.dir / f"{key}.npy")
        self._save_state()
        if time.monotonic() - self._last > log_every_s:
            self._last = time.monotonic()
            log.info("building pks-v3 hint: %s (%.0f s)", key, self._last - self.t0)

    def finish(self, *, keys: Iterable[str] | None = None, extra_meta: dict[str, Any] | None = None
               ) -> PKSHintStore:
        """``meta.json`` (HMAC'd with ``mac_key``) and the rename into place; ``keys``: every one must be done."""
        missing = sorted(set(keys or ()) - set(self.digests))
        if missing:
            raise ConfigError(f"pks-v3 hint build is missing {missing[:3]}")
        meta = {"version": _STORE_VERSION, "scheme": SCHEME_ID, "nr": NR, "fingerprint": self.fingerprint,
                "a_seed": self.a_seed.hex(), "ks_seed": self.ks_seed.hex(), "row_norms": dict(self.norms),
                "digests": dict(self.digests), "created": time.time(), **(extra_meta or {})}
        if self.mac_key is not None:
            meta["mac"] = _meta_mac(self.mac_key, meta)
        (self.dir / _PARTIAL).unlink(missing_ok=True)
        (self.dir / _META).write_text(json.dumps(meta, indent=1))
        if self.final.exists():
            shutil.rmtree(self.final)
        self.dir.rename(self.final)
        st = PKSHintStore(self.final, meta)
        log.info("pks-v3 hint for %s built in %.0f s: %.2f GB in %s", self.fingerprint[:12],
                 time.monotonic() - self.t0, st.nbytes / 1e9, self.final)
        return st

    def abort(self, *, discard: bool | None = None) -> None:
        """Remove the build directory (default: unless resumable; a resumable build is kept for the next run)."""
        for p in self.dir.glob("*.npy.tmp"):
            p.unlink(missing_ok=True)
        if discard if discard is not None else not self.resumable:
            shutil.rmtree(self.dir, ignore_errors=True)


def _kind(spec: Any) -> str:
    k = getattr(spec.id, "kind", None)
    return str(getattr(k, "value", k or ""))


def linear_keys(pack: Any, skip_kinds: frozenset[str] | set[str] = frozenset()) -> set[str]:
    """The linear keys the factory generates masks for (``skip_kinds`` left out)."""
    return {s.id.key for s in pack.linear_specs() if _kind(s) not in skip_kinds}


_META_DOMAIN = b"pgw-pks-hint-meta\0"
_PARTIAL_DOMAIN = b"pgw-pks-hint-partial\0"


def _meta_mac(key: bytes, meta: dict[str, Any], domain: bytes = _META_DOMAIN) -> str:
    """HMAC-SHA256 of ``meta`` (without ``mac``) in canonical JSON under the customer's hint key (``domain``:
    ``meta.json``, or a resumable build's ``partial.json``)."""
    body = json.dumps({k: v for k, v in meta.items() if k != "mac"}, sort_keys=True, separators=(",", ":"))
    return hmac.new(bytes(key), domain + body.encode(), hashlib.sha256).hexdigest()


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
            # a malformed reply (details["reply"]) is the provider's integrity failure, not a refusal of v3
            if exc.code in (ErrorCode.VERSION_MISMATCH, ErrorCode.PROTOCOL) and not exc.details.get("reply"):
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
        self.dims = {k: (int(kk.out_features), int(kk.in_features)) for k, kk in self.kern.items()}

    def __call__(self, body: bytes, rid: str, decode: Callable[[bytes], Any]) -> Any:
        from pgw.hemask.wire import MaskgenResponse, MaskgenResult, decode_maskgen_request, encode_maskgen_response

        req = decode_maskgen_request(body, (1, 2, 3))
        out = [MaskgenResult(it.linear, it.cols, self.provider.run_item(self.fp, self.kern[it.linear], it, req.a_seed,
                                                                        self.dims))
               for it in req.items]
        return decode(encode_maskgen_response(MaskgenResponse(req.request_id, tuple(out), req.version, (1, 2, 3))))


class PKSMaskBackend:
    """``W8 r mod 2^32`` via the provider with scheme pks-v3, for one pack and a K32 ring."""

    decrypt_mode = "pks-v3"

    def __init__(self, pack: Any, ring: Any, store: PKSHintStore, remote: Callable[..., bytes], *,
                 groups_per_call: int = 4, threads: int | None = None, margin_bits: float = 1.0,
                 skip_kinds: frozenset[str] | set[str] = frozenset(), self_check_bits: float | None = None):
        if int(ring.k) != 32:
            raise ConfigError(f"pks-v3 covers K32 mask rings only, not {ring.name}")
        from pgw.core.freivalds import HE_SELF_CHECK_BITS

        #: Soundness target of the factory's private every-row self-check against a provider that picks its
        #: errors (read by :meth:`pgw.factory.MaskGenerator._private_vectors` and the encrypted fused2 factory):
        #: two uniform ring vectors miss a ``2^31`` error in one entry with probability 1/4.
        self.self_check_bits = float(HE_SELF_CHECK_BITS if self_check_bits is None else self_check_bits)
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
        out = np.empty((rn.shape[0], spec.out_features), dtype=np.int64)  # the caller owns (and wipes) it
        step = D * self.groups_per_call
        try:
            for r0 in range(0, rn.shape[0], step):
                rc = rn[r0:r0 + step]
                if not rc.flags.c_contiguous:
                    rc = np.ascontiguousarray(rc)
                t0 = time.perf_counter()
                b, beta, secrets = self.customer.encrypt(self.store.a_seed, linear, rc)
                if rc.base is None:  # a contiguous copy of masks: wipe it
                    rc.fill(0)
                ksk = pack_ksk(beta)
                del beta
                t1 = time.perf_counter()
                wr: np.ndarray | None = None
                try:
                    y = self.remote(linear, b, ksk)
                    t2 = time.perf_counter()
                    wr = self.customer.decrypt(hint, y, secrets)
                    out[r0:r0 + wr.shape[0]] = wr  # W8 r mod 2^32 (secret): no other copy is kept
                finally:
                    for s in secrets:
                        s.wipe()
                    if wr is not None:
                        wr.fill(0)
                t3 = time.perf_counter()
                self._add(encrypt_s=t1 - t0, remote_s=t2 - t1, decrypt_s=t3 - t2, groups=len(secrets), passes=1,
                          jobs=1)
        except BaseException:
            out.fill(0)
            raise
        self._add(wall_s=time.perf_counter() - t_wall)
        return torch.from_numpy(out)

    def _add(self, **kw: float) -> None:
        for k, v in kw.items():
            self.stats[k] = self.stats.get(k, 0.0) + v
