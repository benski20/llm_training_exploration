"""Factory backend that obtains ``W8 r`` from the compute service on MLWE-encrypted masks.

:class:`HEMaskBackend` plugs into :class:`pgw.factory.MaskGenerator` in place of the local GEMM. The
factory still derives ``r`` from the segment seed, computes the check vectors ``u W8`` locally (it
holds the pack), and writes segments to the mask store exactly as before; only the expensive
``W8 r`` product moves to the provider, which sees MLWE ciphertexts only.

The one-time hint ``H = W8 A mod 2^64`` (``m x dim`` per linear) is computed locally from the pack and a
gateway-chosen public seed, so no provider input is trusted at setup. It is stored as one ``.npy`` per
linear and memory-mapped (about ``4096 x`` today's per-token mask size in total at rank 16).
"""

from __future__ import annotations

import json
import logging
import os
import queue
import shutil
import threading
import time
import warnings
from collections import deque
from collections.abc import Callable
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pgw.core.errors import ConfigError, IntegrityError
from pgw.core.prng import new_seed
from pgw.core.ring import RingParams
from pgw.hemask.fastmm import engine, to_planes
from pgw.hemask.lwe import (
    DEGREE,
    HEParams,
    MaskCiphertext,
    a_plane_count,
    encrypt_masks,
    matmul_mod64,
    public_matrix,
)
from pgw.hemask.ntt import big_planes_mm_small  # exact NTT/CRT product; falls back to fastmm

__all__ = ["DECRYPT_MODES", "MAX_GROUPS_PER_PASS", "MAX_INFLIGHT", "MAX_PIPELINE_DEPTH", "HEJob", "HEMaskBackend",
           "HintStore", "verify_rows"]

log = logging.getLogger(__name__)

_META = "meta.json"
_HINT_VERSION = 2  # v2: int8 byte planes 1..(q/8 - 1) of H mod 2^q (byte 0 is not needed for decryption)
_BUILD_COLS = 512  # hint columns per GEMM block (bounds temporary memory)
_DEC_BLOCK_BYTES = 128 << 20  # int32 GEMM output per hint-row block of the native decryption

# torch.from_numpy warns on the read-only memory-mapped hints; torch only reads them. Filtered once
# here (only for this module's calls) instead of per call: warnings.catch_warnings() swaps the
# process-wide filters, which is not thread-safe while the request threads run.
warnings.filterwarnings("ignore", message="The given NumPy array is not writable", category=UserWarning,
                        module=__name__)


class HintStore:
    """One-time hints ``W8 A`` for a model pack, keyed by ``(fingerprint, rank)``.

    Layout: ``<root>/<fingerprint[:16]>-r<rank>/{meta.json, <linear>.npy}``. ``meta.json`` holds the
    public seed of ``A`` (hex), the fingerprint, the rank and each linear's largest weight-row norm
    (for the decryption noise check). Building writes into a temporary directory and renames it.
    """

    def __init__(self, path: Path, meta: dict[str, Any]):
        self.path = path
        self.meta = meta
        self.a_seed = bytes.fromhex(meta["a_seed"])
        self.rank = int(meta["rank"])
        self._maps: dict[str, np.ndarray] = {}

    @staticmethod
    def dir_for(root: Path, fingerprint: str, rank: int, q_bits: int = 64) -> Path:
        return Path(root) / (f"{fingerprint[:16]}-r{rank}" + ("" if q_bits == 64 else f"-q{q_bits}"))

    @classmethod
    def open(cls, root: Path, fingerprint: str, rank: int, q_bits: int = 64) -> HintStore | None:
        path = cls.dir_for(root, fingerprint, rank, q_bits)
        try:
            meta = json.loads((path / _META).read_text())
        except FileNotFoundError:
            return None
        if meta.get("version") != _HINT_VERSION:
            log.warning("HE hint store %s has format v%s (want v%d): rebuilding", path, meta.get("version"),
                        _HINT_VERSION)
            return None
        if meta.get("fingerprint") != fingerprint or meta.get("rank") != rank:
            raise ConfigError(f"hint store {path} does not match fingerprint/rank; delete it to rebuild")
        return cls(path, meta)

    @classmethod
    def build(cls, root: Path, pack: Any, rank: int, *, q_bits: int = 64, log_every_s: float = 30.0) -> HintStore:
        """Compute ``W8 A`` for every linear of ``pack`` (minutes for 7B on CPU; done once per pack)."""
        params = HEParams(k=16, rank=rank, q_bits=q_bits)  # the hint depends on q and rank, not on K
        planes = q_bits // 8 - 1  # bytes 1 .. q/8 - 1 of H mod 2^q
        final = cls.dir_for(root, pack.fingerprint, rank, q_bits)
        tmp = final.with_name(final.name + f".building-{os.getpid()}")
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        seed = new_seed()  # chosen by the customer: a provider-chosen A could expose the masks
        specs = list(pack.linear_specs())
        n_max = max(s.in_features for s in specs)
        a = public_matrix(seed, n_max, params)
        norms: dict[str, float] = {}
        t0 = last = time.monotonic()
        total = sum(s.out_features for s in specs)
        done = 0
        try:
            for lw in pack.iter_linears():
                key = lw.id.key
                w = lw.w8.numpy() if isinstance(lw.w8, torch.Tensor) else np.asarray(lw.w8)
                m, n = w.shape
                norms[key] = float(np.sqrt((w.astype(np.float64) ** 2).sum(1)).max())
                out = np.lib.format.open_memmap(tmp / f"{key}.npy", mode="w+", dtype=np.int8,
                                                shape=(m, planes, params.dim))
                for c0 in range(0, params.dim, _BUILD_COLS):
                    out[:, :, c0:c0 + _BUILD_COLS] = to_planes(
                        matmul_mod64(w, a[:n, c0:c0 + _BUILD_COLS]))[:, 1:1 + planes]
                out.flush()
                del out
                done += m
                if time.monotonic() - last > log_every_s:
                    last = time.monotonic()
                    log.info("building HE hint: %.0f%% (%.0f s)", 100 * done / total, last - t0)
            meta = {"version": _HINT_VERSION, "fingerprint": pack.fingerprint, "rank": rank, "q_bits": q_bits,
                    "a_seed": seed.hex(),
                    "dim": params.dim, "row_norms": norms, "created": time.time()}
            (tmp / _META).write_text(json.dumps(meta, indent=1))
            if final.exists():
                shutil.rmtree(final)
            tmp.rename(final)
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        nbytes = sum(p.stat().st_size for p in final.glob("*.npy"))
        log.info("HE hint for %s built in %.0f s: %.2f GB in %s", pack.fingerprint[:12], time.monotonic() - t0,
                 nbytes / 1e9, final)
        return cls(final, meta)

    @classmethod
    def open_or_build(cls, root: Path, pack: Any, rank: int, q_bits: int = 64) -> HintStore:
        hs = cls.open(root, pack.fingerprint, rank, q_bits)
        return hs if hs is not None else cls.build(root, pack, rank, q_bits=q_bits)

    def hint(self, linear: str) -> np.ndarray:
        h = self._maps.get(linear)
        if h is None:
            h = np.load(self.path / f"{linear}.npy", mmap_mode="r")
            self._maps[linear] = h
        return h

    @property
    def nbytes(self) -> int:
        return sum(p.stat().st_size for p in self.path.glob("*.npy"))


#: Most 256-slot groups decrypted in one pass over a linear's hint (``E2``: fewer hint reads).
MAX_GROUPS_PER_PASS = 16
#: CPU decryption modes of :class:`HEMaskBackend` (``decrypt=``; ``PGW_FACTORY_HE_DECRYPT``).
DECRYPT_MODES = ("auto", "dense", "ntt", "ntt_pre")
#: Hint rows of the ``auto`` calibration (public random data; the cost per row does not depend on m).
_CAL_ROWS = 256
#: Most maskgen requests in flight per backend (``E6``: fills the link on high-latency paths).
MAX_INFLIGHT = 4
#: Most :meth:`HEMaskBackend.submit` jobs the factory keeps outstanding (``M1``: stage overlap across linears).
MAX_PIPELINE_DEPTH = 8


def verify_rows(r: np.ndarray, wr: np.ndarray, cu: np.ndarray, cw: np.ndarray, mask: int) -> np.ndarray:
    """Rows ``t`` with ``(wr[t] @ cu) & mask != (r[t] @ cw) & mask`` (uint64 arithmetic mod 2^64).

    The every-row Freivalds self-check of encrypted mask generation
    (:meth:`pgw.factory.MaskGenerator._self_check`): ``r`` ``(rows, n)`` and ``wr`` ``(rows, m)`` ring
    elements (int64), ``cu`` ``(m, c)`` and ``cw`` ``(n, c)`` uint64. Native (threaded, GIL released)
    when ``pgw_fast`` provides it, else numpy; both give the same flags. Returns a bool ``(rows,)``.
    """
    from pgw.core.native import fast, native_available

    if native_available("he_check_rows"):
        bad: np.ndarray = fast.he_check_rows(np.ascontiguousarray(r, dtype=np.int64),
                                             np.ascontiguousarray(wr, dtype=np.int64),
                                             np.ascontiguousarray(cu, dtype=np.uint64),
                                             np.ascontiguousarray(cw, dtype=np.uint64), int(mask))
        return bad.astype(bool)
    m64 = np.uint64(mask)
    lhs = (np.asarray(wr).astype(np.uint64) @ np.asarray(cu, dtype=np.uint64)) & m64
    rhs = (np.asarray(r).astype(np.uint64) @ np.asarray(cw, dtype=np.uint64)) & m64
    out: np.ndarray = (lhs != rhs).any(axis=1)
    return out


class HEMaskBackend:
    """``W8 r mod 2^K`` via the provider, for one pack and ring.

    Args:
        pack: Open model pack (for linear specs; the hint store was built from it).
        ring: Mask ring.
        hints: :class:`HintStore` of this pack.
        remote: ``remote(linear, b) -> y`` computing ``W8 b mod 2^64`` on the provider (e.g.
            :class:`pgw.hemask.client.RemoteMulmod64`). Called from up to ``max_inflight`` threads at
            once, so it must be thread-safe.
        groups_per_call: 256-slot groups sent per request (fewer round trips, larger messages).
        max_inflight: Requests with the provider at once (1-4). Encryption of the next request always
            overlaps the ones in flight: with 1, request ``k`` is remote while ``k + 1`` is encrypted
            (the pre-E6 behaviour); with more, several requests keep a high-latency link busy.
        groups_per_pass: Most groups (of one ``matmul`` call, possibly from several replies) decrypted
            in one pass over the linear's hint (1-16, a hard cap: a reply is split across passes when
            needed). The hint (``m x dim`` byte planes, usually memory-mapped) is streamed once per
            pass, so larger passes read it less often.
        pass_bytes: Memory budget of one decryption pass's replies (caps ``groups_per_pass`` for very
            wide linears; hard, except that a pass holds at least one group).
        encrypt: ``"dense"`` (default: byte-plane product ``A[:n] @ S`` on the int8 engine) or
            ``"ntt_pre"`` (:class:`pgw.hemask.mlwe_ring.PretransformedEncryptor`: ``A`` kept in NTT form,
            bit-identical ciphertexts, far fewer operations on a CPU gateway; needs ``pgw_ntt``).
        device: Where encryption and decryption run as fused device passes (:mod:`pgw.hemask.devmm`):
            ``"auto"`` = the gateway GPU when the CUDA engine is chosen and its self-test passed, else the
            CPU engines; ``None`` = the CPU engines; a device name (tests: ``"cpu"`` runs the device code
            on CPU tensors). Same ciphertext distribution and the same decrypted bits either way.
        pipeline_depth: ``0`` (default): the factory calls :meth:`matmul` one linear at a time, so a
            segment's stages run back to back (with a few groups per linear, one request per call:
            encrypt, provider, decrypt in series). ``>= 1``: the factory uses :meth:`submit` and keeps up
            to this many jobs (linears or chunks) outstanding, so encryption of job ``k + 1``, the provider
            work on ``k`` and the decryption of ``k - 1`` overlap (one encryption thread, ``max_inflight``
            request threads, one decryption thread; native kernels release the GIL). Same requests, same
            ciphertexts, same decrypted bits; only the start times move (see :meth:`submit`).
        pipeline_bytes: Soft cap on the factory's outstanding job memory with ``pipeline_depth``
            (``r``, output and ciphertexts; at least one job is always outstanding).
        decrypt: CPU decryption when no device pass runs it (:data:`DECRYPT_MODES`). ``"dense"``: the
            byte-plane GEMM ``H @ S`` (int8 engine plus the native combine). ``"ntt"``: the fused native
            NTT decryption (:func:`pgw.hemask.ntt.decrypt_into`; the correction ``H S`` never leaves native
            scratch memory) on the stored hint planes, no extra memory. ``"ntt_pre"``: as ``"ntt"``, with
            the hints of as many linears as fit ``hint_ntt_bytes`` kept in NTT form in RAM (12 bytes per
            coefficient, twice the stored planes; about 2-3x less work per row than ``"ntt"``), the others
            as ``"ntt"``. ``"auto"`` (default): pre-transformed hints within ``hint_ntt_bytes`` (0: none),
            else, per pass size, whichever of ``"dense"`` and ``"ntt"`` a start-up calibration on public
            random data found faster **and bit-identical** (else ``"dense"``). All modes give the same bits
            (exact products; every decrypted row is also checked by the factory's Freivalds self-check);
            an error in a native NTT call falls back to ``"dense"`` for the rest of the backend's life.
        hint_ntt_bytes: RAM budget for hints held in NTT form (``"ntt_pre"`` / ``"auto"``); 0 = none.
    """

    #: Fused device passes (:mod:`pgw.hemask.devmm`), or ``None`` (the CPU / per-call fastmm engines).
    _dev: Any = None
    #: Class defaults of the decryption state (instances set their own in ``__init__``; tests that build a
    #: bare instance with ``object.__new__`` get the dense path).
    decrypt_mode: str = "dense"
    hint_ntt_bytes: int = 0
    _ntt_broken: bool = False

    def __init__(self, pack: Any, ring: RingParams, hints: HintStore,
                 remote: Callable[[str, np.ndarray], np.ndarray], *, groups_per_call: int = 4,
                 max_inflight: int = 2, groups_per_pass: int = MAX_GROUPS_PER_PASS, pass_bytes: int = 1 << 30,
                 encrypt: str = "dense", device: str | None = "auto", pipeline_depth: int = 0,
                 pipeline_bytes: int = 1 << 30, decrypt: str = "auto", hint_ntt_bytes: int = 0):
        if encrypt not in ("dense", "ntt_pre"):
            raise ConfigError(f"encrypt must be 'dense' or 'ntt_pre', got {encrypt!r}")
        if decrypt not in DECRYPT_MODES:
            raise ConfigError(f"decrypt must be one of {DECRYPT_MODES}, got {decrypt!r}")
        if int(hint_ntt_bytes) < 0:
            raise ConfigError("hint_ntt_bytes must be >= 0")
        self.decrypt_mode = decrypt
        self.hint_ntt_bytes = int(hint_ntt_bytes)
        self._pre: dict[str, Any] = {}  # linear -> ntt.TransformedOperand of its hint
        self._pre_skip: set[str] = set()  # linears whose hint does not fit the budget
        self._pre_used = 0
        self._dec_lock = threading.Lock()
        #: Serialises hint transforms: concurrent passes of one linear wait for one transform instead of each
        #: reserving budget and transforming (double memory, budget over-counted).
        self._pre_build = threading.Lock()
        self._dec_choice: dict[int, str] = {}  # auto: pass groups -> "dense" | "ntt"
        self._ntt_broken = False
        #: Decryption path per pass taken so far (``dense``, ``ntt``, ``ntt_pre``, ``device``) and the
        #: calibration results of ``auto`` (diagnostics).
        self.decrypt_paths: dict[str, int] = {}
        self.decrypt_calibration: dict[int, dict[str, float | str]] = {}
        self.params = HEParams(k=ring.k, rank=hints.rank)
        if int(hints.meta.get("q_bits", 64)) != self.params.q:
            raise ConfigError(f"hint store is for q=2^{hints.meta.get('q_bits', 64)}, ring {ring.name} needs "
                              f"q=2^{self.params.q}")
        self.ring = ring
        self.hints = hints
        self.remote = remote
        self.groups_per_call = max(1, min(16, int(groups_per_call)))
        self.max_inflight = max(1, min(MAX_INFLIGHT, int(max_inflight)))
        self.groups_per_pass = max(1, min(MAX_GROUPS_PER_PASS, int(groups_per_pass)))
        self.pass_bytes = max(1, int(pass_bytes))
        self.pipeline_depth = max(0, min(MAX_PIPELINE_DEPTH, int(pipeline_depth)))
        self.pipeline_bytes = max(1, int(pipeline_bytes))
        self.specs = {s.id.key: s for s in pack.linear_specs()}
        for key, norm in hints.meta["row_norms"].items():
            self.params.check_noise(norm, key)
        a = public_matrix(hints.a_seed, max(s.in_features for s in self.specs.values()), self.params)
        self._fast: Any = None
        dev: torch.device | None = None
        if encrypt == "ntt_pre":
            from pgw.hemask.mlwe_ring import PretransformedEncryptor

            # A in NTT form only (same ciphertexts as the dense path, bit for bit). Encryption stays on the
            # CPU NTT (faster than the fused device pass on a T4); decryption uses the CPU engines.
            self._fast = PretransformedEncryptor(a, self.params)
            self._planes = False
            self._a: np.ndarray = np.empty((0, self.params.dim), dtype=np.uint64)
        else:
            # Public matrix as int8 byte planes for the int8 engines (gateway GPU, CPU AMX/VNNI); as uint64
            # for the float64 engine, which would otherwise rebuild the words on every call. Only the
            # ceil(q/8) low planes are kept (7 at q = 2^56): the top byte vanishes mod 2^q.
            from pgw.hemask.devmm import device_for

            if device == "auto":
                dev = device_for(self.params)
            else:
                dev = None if device is None else torch.device(device)
            self._planes = dev is not None or engine() != "cpu_f64"
            self._a = self._a_planes(a) if self._planes else a
        self.encrypt_mode = encrypt
        del a
        self._dev = None
        if dev is not None:
            from pgw.hemask.devmm import DeviceHE

            self._dev = DeviceHE(self._a, self.params, dev)
            self._a = self._a[:0]  # the device holds A; the host copy is not needed
        #: Where the products run: ``device:<dev>`` (fused device passes) or the fastmm/NTT CPU engine name.
        self.engine = f"device:{dev}" if dev is not None else engine()
        self.stats = {"groups": 0, "encrypt_s": 0.0, "remote_s": 0.0, "decrypt_s": 0.0, "wall_s": 0.0,
                      "passes": 0, "max_inflight": 0, "jobs": 0}
        self._stats_lock = threading.Lock()
        self._dev_lock = threading.Lock()  # the device passes are not used from two threads at once
        self._busy = 0  # submitted jobs not finished yet (wall_s counts the time with at least one)
        self._busy_t0 = 0.0
        self._remote_now = 0  # provider requests running now (submit path)
        self._stage_pools: tuple[ThreadPoolExecutor, ThreadPoolExecutor] | None = None
        self._stage_pools_lock = threading.Lock()
        log.info("encrypted mask factory: encryption/decryption on %s%s", self.engine,
                 f" (hint cache {self._dev.cache.budget / 2**30:.1f} GiB)" if self._dev is not None else "")
        self._pool = ThreadPoolExecutor(max_workers=self.max_inflight, thread_name_prefix="pgw-hemask-remote")

    def _a_planes(self, a: np.ndarray, block: int = 1024) -> np.ndarray:
        npl = a_plane_count(self.params)
        out = np.empty((a.shape[0], npl, a.shape[1]), dtype=np.int8)
        for r0 in range(0, a.shape[0], block):
            out[r0:r0 + block] = to_planes(a[r0:r0 + block])[:, :npl]
        return out

    def preferred_chunk_slots(self, max_bytes: int = 512 << 20) -> int:
        """Factory GEMM batch for encrypted generation: one full decryption pass (``groups_per_pass``
        groups), so that requests pipeline (``max_inflight``, ``groups_per_call``) and the hint is read
        once per pass; capped so that one chunk of the widest linear's output stays within ``max_bytes``."""
        m_max = max(s.out_features for s in self.specs.values())
        groups = max(1, min(self.groups_per_pass, max_bytes // (8 * DEGREE * m_max)))
        return DEGREE * groups

    def _encrypt(self, rn: np.ndarray, n: int) -> list[MaskCiphertext]:
        """The 256-slot groups of ``rn``, encrypted (on the device: one GEMM for all of them)."""
        if self._fast is not None:
            out: list[MaskCiphertext] = self._fast.encrypt(rn)
            return out
        if self._dev is not None:
            with self._dev_lock:
                cts, b = self._dev.encrypt(rn, n)
            for gi, ct in enumerate(cts):
                ct.b = np.ascontiguousarray(b[:, gi * DEGREE:(gi + 1) * DEGREE])
            return cts
        a = self._a[:n]
        if self._planes:
            return [encrypt_masks(rn[g0:g0 + DEGREE], None, self.params, a_planes=a)
                    for g0 in range(0, len(rn), DEGREE)]
        return [encrypt_masks(rn[g0:g0 + DEGREE], a, self.params) for g0 in range(0, len(rn), DEGREE)]

    def _decrypt(self, linear: str, y: np.ndarray, cts: list[MaskCiphertext], m: int) -> np.ndarray:
        """``W8 r`` (``(sum rows, m)`` int64) of the groups of one provider reply."""
        self._check_reply(linear, y, cts, m)
        out = np.empty((sum(ct.rows for ct in cts), m), dtype=np.int64)
        self._decrypt_into(linear, [(y, cts, 0)], out, m)
        return out

    @staticmethod
    def _check_reply(linear: str, y: np.ndarray, cts: list[MaskCiphertext], m: int) -> None:
        if y.shape != (m, DEGREE * len(cts)):
            raise IntegrityError(f"maskgen {linear}: provider returned shape {y.shape}, expected "
                                 f"{(m, DEGREE * len(cts))}")

    def _decrypt_into(self, linear: str, items: list[tuple[np.ndarray, list[MaskCiphertext], int]],
                      out: np.ndarray, m: int) -> None:
        """Decrypt several replies with one pass over the hint: ``hint @ [S_1 | S_2 | ...]``.

        ``items`` are ``(y, cts, s0)``: reply ``y`` (``m x 256 len(cts)``) of groups whose slots start
        at output row ``s0``. Wipes nothing but its own copies (the caller wipes the ciphertexts).
        """
        hint = self.hints.hint(linear)
        if self._dev is not None and hint.ndim == 3:
            with self._dev_lock:
                self._dev.decrypt_into(linear, hint, items, out, self.params.delta_bits, self.ring.k)
            self._count_path("device")
            return
        groups = sum(len(cts) for _, cts, _ in items)
        try:  # the hint transform (ntt_pre) is native too: its faults fall back like the decryption's
            path, big = self._decrypt_path(linear, hint, groups)
        except Exception as exc:
            with self._dec_lock:
                self._ntt_broken = True
            log.error("encrypted mask factory: NTT hint transform failed (%s: %s); using the dense path",
                      type(exc).__name__, str(exc)[:200])
            path, big = "dense", None
        if path != "dense":
            try:
                self._decrypt_ntt(big, items, out)
                self._count_path(path)
                return
            except Exception as exc:  # a native fault: dense from now on (same bits, slower)
                with self._dec_lock:
                    self._ntt_broken = True
                log.error("encrypted mask factory: NTT decryption failed (%s: %s); using the dense path",
                          type(exc).__name__, str(exc)[:200])
        self._decrypt_dense(hint, items, out)
        self._count_path("dense")

    def _count_path(self, path: str) -> None:
        if "_dec_lock" not in self.__dict__:  # a bare test instance
            return
        with self._dec_lock:
            self.decrypt_paths[path] = self.decrypt_paths.get(path, 0) + 1

    # ---- NTT decryption -----------------------------------------------------------------------------
    def _decrypt_path(self, linear: str, hint: np.ndarray, groups: int) -> tuple[str, Any]:
        """``(path, operand)`` of one pass: ``("ntt_pre", TransformedOperand)``, ``("ntt", hint planes)`` or
        ``("dense", None)`` (see the ``decrypt`` argument)."""
        from pgw.hemask import ntt

        mode = self.decrypt_mode
        if mode == "dense" or hint.ndim != 3 or self._ntt_broken or not ntt.decrypt_available():
            return "dense", None
        if mode in ("ntt_pre", "auto") and self.hint_ntt_bytes > 0:
            pre = self._pretransformed(linear, hint)
            if pre is not None:
                return "ntt_pre", pre
        if mode in ("ntt", "ntt_pre"):
            return "ntt", hint
        return (("ntt", hint) if self._calibrated(hint.shape[1], groups) == "ntt" else ("dense", None))

    def _pretransformed(self, linear: str, hint: np.ndarray) -> Any:
        """The NTT form of ``linear``'s hint (made on first use when it fits ``hint_ntt_bytes``), or ``None``."""
        with self._dec_lock:
            pre = self._pre.get(linear)
            if pre is not None or linear in self._pre_skip:
                return pre
        with self._pre_build:  # one transform at a time; a pass that waited finds the result below
            return self._build_pretransformed(linear, hint)

    def _build_pretransformed(self, linear: str, hint: np.ndarray) -> Any:
        from pgw.hemask import ntt

        with self._dec_lock:
            pre = self._pre.get(linear)
            if pre is not None or linear in self._pre_skip:
                return pre
            m, _npl, dim = hint.shape
            need = 12 * (-(-m // 16) * 16) * dim  # 3 primes x 4 bytes, rows padded to 16
            if self._pre_used + need > self.hint_ntt_bytes:
                self._pre_skip.add(linear)
                return None
            self._pre_used += need  # reserved before the (slow) transform, released if it fails
        try:
            t0 = time.perf_counter()
            pre = ntt.TransformedOperand.from_planes(hint, first_byte=1)
            log.info("HE hint of %s in NTT form: %.0f MB in %.1f s (%.1f of %.1f GB budget used)", linear,
                     pre.nbytes / 1e6, time.perf_counter() - t0, self._pre_used / 1e9, self.hint_ntt_bytes / 1e9)
        except Exception:
            with self._dec_lock:
                self._pre_used -= need
                self._pre_skip.add(linear)
            raise
        with self._dec_lock:
            self._pre[linear] = pre
        return pre

    def _decrypt_ntt(self, big: Any, items: list[tuple[np.ndarray, list[MaskCiphertext], int]],
                     out: np.ndarray) -> None:
        """Fused native decryption of one pass (:func:`pgw.hemask.ntt.decrypt_into`)."""
        from pgw.hemask import ntt

        secs = np.empty((sum(len(cts) for _, cts, _ in items), self.params.rank, DEGREE), dtype=np.int8)
        try:
            gi = 0
            ys: list[np.ndarray] = []
            slots: list[np.ndarray] = []
            for y, cts, s0 in items:
                for g, ct in enumerate(cts):
                    sec = ntt.secrets_of(ct.secret)
                    if sec is None:
                        raise ConfigError("secret matrix is not block-negacyclic")
                    secs[gi] = sec[0]
                    sec.fill(0)
                    gi += 1
                    col = np.full(DEGREE, -1, dtype=np.int64)
                    row = s0 + sum(c.rows for c in cts[:g])
                    col[:ct.rows] = np.arange(row, row + ct.rows)
                    slots.append(col)
                ys.append(y)
            ntt.decrypt_into(big, secs, ys, np.concatenate(slots), self.params.delta_bits, self.ring.k, out)
        finally:
            secs.fill(0)

    def _calibrated(self, npl: int, groups: int) -> str:
        """``auto``: ``"ntt"`` or ``"dense"`` for passes of ``groups`` groups, decided once per size on public
        random data (:data:`_CAL_ROWS` hint rows, random secrets and replies): the faster of the two, and
        ``"ntt"`` only when both gave the same bits."""
        with self._dec_lock:
            hit = self._dec_choice.get(groups)
        if hit is not None:
            return hit
        from pgw.hemask.lwe import secret_matrix

        rng = np.random.default_rng()  # public test data: nothing secret is involved
        dim = self.params.dim
        hint = rng.integers(-128, 128, size=(_CAL_ROWS, npl, dim), dtype=np.int8)
        cts = [MaskCiphertext(b=_NO_B, secret=secret_matrix(rng.integers(-1, 2, size=(self.params.rank, DEGREE))),
                              rows=DEGREE) for _ in range(groups)]
        y = rng.integers(0, 1 << 63, size=(_CAL_ROWS, DEGREE * groups), dtype=np.uint64)
        items = [(y, cts, 0)]
        outs = {p: np.zeros((DEGREE * groups, _CAL_ROWS), dtype=np.int64) for p in ("dense", "ntt")}
        times: dict[str, float] = {}
        choice = "dense"
        try:
            for path in ("dense", "ntt", "dense", "ntt"):  # each twice (the first run warms up), best of the second
                t0 = time.perf_counter()
                if path == "dense":
                    self._decrypt_dense(hint, items, outs[path])
                else:
                    self._decrypt_ntt(hint, items, outs[path])
                times[path] = time.perf_counter() - t0
            same = bool(np.array_equal(outs["dense"], outs["ntt"]))
            if not same:
                log.error("encrypted mask factory: NTT and dense decryption differ on the calibration data; "
                          "using the dense path")
            elif times["ntt"] < times["dense"]:
                choice = "ntt"
            cal: dict[str, float | str] = {"dense_s": times["dense"], "ntt_s": times["ntt"], "rows": _CAL_ROWS,
                                           "identical": str(same), "choice": choice}
        except Exception as exc:
            log.error("encrypted mask factory: NTT decryption calibration failed (%s: %s); using the dense path",
                      type(exc).__name__, str(exc)[:200])
            cal = {"error": type(exc).__name__, "choice": choice}
        log.info("encrypted mask factory: decryption of %d-group passes on the %s path (calibration %s)", groups,
                 choice, cal)
        with self._dec_lock:
            self._dec_choice.setdefault(groups, choice)
            self.decrypt_calibration[groups] = cal
            return self._dec_choice[groups]

    def _decrypt_dense(self, hint: np.ndarray, items: list[tuple[np.ndarray, list[MaskCiphertext], int]],
                       out: np.ndarray) -> None:
        """The byte-plane GEMM path: ``H @ [S_1 | ...]`` on the int8 engine (or the NTT product), then the combine."""
        smat = np.concatenate([ct.secret for it in items for ct in it[1]], axis=1)
        try:
            nat = None
            # Fused native path first: on an AMX host it measured faster than the NTT product (115 vs 153 ms for a
            # 0.5B gate_up group of 1024 slots, bit-identical); the NTT product serves hosts without it.
            if hint.ndim == 3 and engine() == "cpu_int8":
                from pgw.core.native import fast, native_available

                nat = fast if native_available("he_decrypt_combine") else None
            if nat is not None:
                self._decrypt_native(nat, hint, smat, items, out)
                return
            corr = big_planes_mm_small(hint, smat, first_byte=1)
        finally:
            smat.fill(0)
        try:
            half = np.uint64(1 << (self.params.delta_bits - 1))
            c0 = 0
            for y, cts, s0 in items:
                v = y - corr[:, c0:c0 + y.shape[1]]
                wr = ((v + half) >> np.uint64(self.params.delta_bits)) & np.uint64(self.ring.mask)
                v.fill(0)
                rows = sum(ct.rows for ct in cts)
                out[s0:s0 + rows] = np.concatenate([wr[:, gi * DEGREE:gi * DEGREE + ct.rows]
                                                    for gi, ct in enumerate(cts)], axis=1).T
                wr.fill(0)
                c0 += y.shape[1]
        finally:
            corr.fill(0)  # H S: with y it gives W8 r

    def _decrypt_native(self, nat: Any, hint: np.ndarray, smat: np.ndarray,
                        items: list[tuple[np.ndarray, list[MaskCiphertext], int]], out: np.ndarray) -> None:
        """Hint-row blocks: int8 GEMM in ``torch._int_mm``, then one fused native pass per reply
        (plane combine, subtract, round, transpose into ``out``)."""
        m, npl, dim = hint.shape
        cols = smat.shape[1]
        rhs = torch.from_numpy(smat)  # (dim, cols) row-major: no transposed copy (the caller wipes smat)
        offset = (128 * smat.sum(0, dtype=np.int64)).astype(np.int32)
        maps = []
        c0 = 0
        for y, cts, s0 in items:
            slots = np.full(y.shape[1], -1, dtype=np.int64)
            row = s0
            for gi, ct in enumerate(cts):
                slots[gi * DEGREE:gi * DEGREE + ct.rows] = np.arange(row, row + ct.rows)
                row += ct.rows
            maps.append((np.ascontiguousarray(y, dtype=np.uint64), c0, slots))
            c0 += y.shape[1]
        block = max(16, min(2048, _DEC_BLOCK_BYTES // (npl * cols * 4)))
        try:
            for r0 in range(0, m, block):
                blk = np.ascontiguousarray(hint[r0:r0 + block])
                nb = blk.shape[0]
                lhs = torch.from_numpy(blk.reshape(nb * npl, dim))  # read-only hint: warning filtered above
                part = torch._int_mm(lhs, rhs).numpy().reshape(nb, npl, cols)
                try:
                    for y, pc0, slots in maps:
                        nat.he_decrypt_combine(part, offset, 1, pc0, y, r0, slots, self.params.delta_bits,
                                               self.ring.k, out)
                finally:
                    part.fill(0)  # partial H S (the decryption correction)
        finally:
            offset.fill(0)  # 128 * column sums of S

    def _add_stats(self, **kw: float) -> None:
        with self._stats_lock:
            for key, v in kw.items():
                self.stats[key] += v

    def _flush_pass(self, linear: str, ready: deque[tuple[np.ndarray, list[MaskCiphertext], int]],
                    pass_groups: int, out: np.ndarray, m: int) -> None:
        """Decrypt the oldest ``pass_groups`` ready groups (fewer at the end) into ``out``, splitting a
        reply across passes when it does not fit; wipes the secrets of the groups it took."""
        t1 = time.perf_counter()
        take: list[tuple[np.ndarray, list[MaskCiphertext], int]] = []
        groups = 0
        while ready and groups < pass_groups:
            y, cts, s0 = ready[0]
            k = min(len(cts), pass_groups - groups)
            if k == len(cts):
                take.append(ready.popleft())
            else:
                head_rows = sum(ct.rows for ct in cts[:k])
                take.append((y[:, :k * DEGREE], cts[:k], s0))
                ready[0] = (y[:, k * DEGREE:], cts[k:], s0 + head_rows)
            groups += k
        try:
            self._decrypt_into(linear, take, out, m)
        finally:
            for _, cts, _ in take:
                for ct in cts:
                    ct.wipe()
        self._add_stats(decrypt_s=time.perf_counter() - t1, groups=groups, passes=1)

    # ---- asynchronous jobs (pipeline_depth >= 1) ------------------------------------------------
    def _pools(self) -> tuple[ThreadPoolExecutor, ThreadPoolExecutor]:
        with self._stage_pools_lock:
            if self._stage_pools is None:
                self._stage_pools = (ThreadPoolExecutor(max_workers=1, thread_name_prefix="pgw-hemask-encrypt"),
                                     ThreadPoolExecutor(max_workers=1, thread_name_prefix="pgw-hemask-decrypt"))
            return self._stage_pools

    def job_bytes(self, linear: str, rows: int) -> int:
        """Gateway memory of one outstanding job of ``rows`` masks (``r``, output, ciphertexts and
        replies; an estimate used against :attr:`pipeline_bytes`)."""
        spec = self.specs[linear]
        return 8 * rows * (2 * spec.in_features + 2 * spec.out_features)

    def _job_started(self) -> None:
        with self._stats_lock:
            if self._busy == 0:
                self._busy_t0 = time.perf_counter()
            self._busy += 1
            self.stats["jobs"] += 1

    def _job_ended(self) -> None:
        with self._stats_lock:
            self._busy -= 1
            if self._busy == 0:
                self.stats["wall_s"] += time.perf_counter() - self._busy_t0

    def _timed_remote(self, linear: str, b: np.ndarray) -> np.ndarray:
        with self._stats_lock:  # requests with the provider at once (the serial path counts submitted ones)
            self._remote_now += 1
            self.stats["max_inflight"] = max(self.stats["max_inflight"], self._remote_now)
        t0 = time.perf_counter()
        try:
            return self.remote(linear, b)
        finally:
            with self._stats_lock:
                self._remote_now -= 1
            self._add_stats(remote_s=time.perf_counter() - t0)

    def submit(self, linear: str, r: torch.Tensor) -> HEJob:
        """Start ``W8 r mod 2^K`` for ``(rows, n)`` masks ``r`` and return at once (:class:`HEJob`).

        The same work as :meth:`matmul` (same batches of ``groups_per_call`` groups, same requests, same
        decryption passes and the same output bits), split into stages that run in their own threads:
        encryption (one thread, jobs in submission order), the provider requests (``max_inflight``
        threads, queued in submission order) and decryption (one thread, jobs in order). While the caller
        keeps several jobs outstanding, job ``k + 1`` is encrypted while the provider works on ``k`` and
        ``k - 1`` is decrypted. ``r`` must stay unchanged until the job is done (:meth:`HEJob.result` or
        :meth:`HEJob.cancel` returned). The provider sees the same ciphertexts, sizes and counts as with
        :meth:`matmul`; only the start times of the requests move, and they depend on stage times that
        are functions of the (public) shapes, not of the masks.
        """
        spec = self.specs[linear]
        n, m = spec.in_features, spec.out_features
        if r.ndim != 2 or r.shape[1] != n:
            raise ConfigError(f"{linear}: masks must be (rows, {n}), got {tuple(r.shape)}")
        rn = r.numpy() if isinstance(r, torch.Tensor) else np.asarray(r)
        job = HEJob(linear, rn, n, m)
        enc_pool, dec_pool = self._pools()
        self._job_started()
        try:
            enc_pool.submit(self._stage_encrypt, job)
        except BaseException:  # the pool is shut down: the job never runs
            self._job_ended()
            raise
        dec_pool.submit(self._stage_decrypt, job)
        return job

    def _stage_encrypt(self, job: HEJob) -> None:
        """Encrypt the job's batches and send each one; hands ``(s0, cts, reply future)`` to the decryption
        stage through ``job.items`` and always ends the queue with ``None`` (or the exception)."""
        rows = job.rows
        step = DEGREE * self.groups_per_call
        cts: list[MaskCiphertext] = []
        end: BaseException | None = None
        try:
            for s0 in range(0, rows, step):
                if job.aborted.is_set():
                    break
                t0 = time.perf_counter()
                cts = self._encrypt(job.rn[s0:min(s0 + step, rows)], job.n)
                b = np.concatenate([ct.b for ct in cts], axis=1)
                for ct in cts:
                    ct.b = _NO_B  # the request owns the ciphertext now
                self._add_stats(encrypt_s=time.perf_counter() - t0)
                fut = self._pool.submit(self._timed_remote, job.linear, b)
                del b
                job.items.put((s0, cts, fut))  # the decryption stage owns (and wipes) cts from here
                cts = []
        except BaseException as exc:  # handed to the decryption stage, which reports it
            end = exc
            for ct in cts:
                ct.wipe()
        finally:
            job.items.put(end)

    def _stage_decrypt(self, job: HEJob) -> None:
        """Decrypt the job and complete ``job.future`` (after the job stopped counting as busy)."""
        res: torch.Tensor | None = None
        err: BaseException | None = None
        try:
            res = self._decrypt_job(job)
        except BaseException as exc:  # delivered through the job's future
            err = exc
        finally:
            self._job_ended()
        if err is not None:
            job.future.set_exception(err)
        else:
            job.future.set_result(res)

    def _decrypt_job(self, job: HEJob) -> torch.Tensor:
        """Collect the job's replies in order and decrypt them in passes; wipes every secret it got."""
        out = np.empty((job.rows, job.m), dtype=np.int64)
        pass_groups = self._pass_groups(job.m)
        ready: deque[tuple[np.ndarray, list[MaskCiphertext], int]] = deque()
        pending: deque[tuple[int, list[MaskCiphertext], Future[np.ndarray]]] = deque()
        ended = False
        try:
            while True:
                item = job.items.get()
                if item is None or isinstance(item, BaseException):
                    ended = True
                    if item is not None:
                        raise item
                    break
                pending.append(item)
                # decrypt what has arrived (in order) while later batches are encrypted or remote
                while pending and pending[0][2].done():
                    self._collect(job, pending, ready, pass_groups, out)
            while pending:
                self._collect(job, pending, ready, pass_groups, out)
            while ready:
                if job.aborted.is_set():
                    raise CancelledError()
                self._flush_pass(job.linear, ready, pass_groups, out, job.m)
            if job.aborted.is_set():
                raise CancelledError()
            return torch.from_numpy(out)
        except BaseException:
            out.fill(0)
            raise
        finally:
            while not ended:  # the encryption stage always ends the queue
                item = job.items.get()
                if item is None or isinstance(item, BaseException):
                    ended = True
                else:
                    pending.append(item)
            for _, cts, fut in pending:
                fut.cancel()
                for ct in cts:
                    ct.wipe()
            for _, cts, _ in ready:
                for ct in cts:
                    ct.wipe()

    def _collect(self, job: HEJob, pending: deque[tuple[int, list[MaskCiphertext], Future[np.ndarray]]],
                 ready: deque[tuple[np.ndarray, list[MaskCiphertext], int]], pass_groups: int,
                 out: np.ndarray) -> None:
        """Move the oldest reply to ``ready`` (waiting for it) and decrypt full passes."""
        if job.aborted.is_set():
            raise CancelledError()
        s0, cts, fut = pending[0]
        y = fut.result()
        self._check_reply(job.linear, y, cts, job.m)  # on failure the stage's finally wipes cts
        pending.popleft()
        ready.append((y, cts, s0))
        while sum(len(c) for _, c, _ in ready) >= pass_groups:
            self._flush_pass(job.linear, ready, pass_groups, out, job.m)

    def _pass_groups(self, m: int) -> int:
        """Groups per decryption pass: ``groups_per_pass`` within ``pass_bytes`` of replies (>= 1)."""
        return max(1, min(self.groups_per_pass, self.pass_bytes // (m * DEGREE * 8)))

    def matmul(self, linear: str, r: torch.Tensor) -> torch.Tensor:
        """``(rows, n)`` masks -> ``(rows, m)`` ``W8 r mod 2^K`` (int64 CPU tensor).

        Pipelined: up to ``max_inflight`` requests are with the provider while the next group batch is
        encrypted; replies are decrypted in passes of at most ``groups_per_pass`` groups (one hint read
        per pass). Int8 GEMMs and native kernels release the GIL.
        """
        spec = self.specs[linear]
        n, m = spec.in_features, spec.out_features
        if r.ndim != 2 or r.shape[1] != n:
            raise ConfigError(f"{linear}: masks must be (rows, {n}), got {tuple(r.shape)}")
        rows = int(r.shape[0])
        rn = r.numpy() if isinstance(r, torch.Tensor) else np.asarray(r)
        out = np.empty((rows, m), dtype=np.int64)
        step = DEGREE * self.groups_per_call
        batches = [(s0, min(s0 + step, rows)) for s0 in range(0, rows, step)]
        pass_groups = self._pass_groups(m)
        t_wall = time.perf_counter()
        inflight: deque[tuple[int, list[MaskCiphertext], Future[np.ndarray], float]] = deque()
        ready: deque[tuple[np.ndarray, list[MaskCiphertext], int]] = deque()

        def collect() -> None:
            s0, cts, fut, t_sent = inflight[0]
            y = fut.result()
            self._check_reply(linear, y, cts, m)  # on failure the finally below still wipes cts
            inflight.popleft()
            ready.append((y, cts, s0))  # wiped by flush() or the finally below
            self.stats["remote_s"] += time.perf_counter() - t_sent

        def ready_groups() -> int:
            return sum(len(cts) for _, cts, _ in ready)

        def flush() -> None:
            self._flush_pass(linear, ready, pass_groups, out, m)

        try:
            for s0, s1 in batches:
                t0 = time.perf_counter()
                cts = self._encrypt(rn[s0:s1], n)  # overlaps the requests in flight
                b = np.concatenate([ct.b for ct in cts], axis=1)
                for ct in cts:
                    ct.b = _NO_B  # the request owns the ciphertext now
                self.stats["encrypt_s"] += time.perf_counter() - t0
                if len(inflight) >= self.max_inflight:  # wait for the oldest before sending more
                    try:
                        collect()
                    except BaseException:
                        for ct in cts:
                            ct.wipe()
                        raise
                inflight.append((s0, cts, self._pool.submit(self.remote, linear, b), time.perf_counter()))
                del b
                self.stats["max_inflight"] = max(self.stats["max_inflight"], len(inflight))
                while ready_groups() >= pass_groups:  # decrypt full passes while the provider works
                    flush()
            while inflight:
                collect()
                while ready_groups() >= pass_groups:
                    flush()
            while ready:
                flush()
        finally:
            for _, cts, fut, _ in inflight:
                fut.cancel()
                for ct in cts:
                    ct.wipe()
            for _, cts, _ in ready:
                for ct in cts:
                    ct.wipe()
        self.stats["wall_s"] += time.perf_counter() - t_wall
        return torch.from_numpy(out)


_NO_B = np.empty((0, DEGREE), dtype=np.uint64)


class HEJob:
    """One :meth:`HEMaskBackend.submit` job: ``W8 r`` of one chunk, computed in the stage threads."""

    def __init__(self, linear: str, rn: np.ndarray, n: int, m: int):
        self.linear = linear
        self.rn = rn
        self.n = n
        self.m = m
        self.rows = int(rn.shape[0])
        self.aborted = threading.Event()
        self.future: Future[torch.Tensor] = Future()
        #: ``(s0, cts, reply future)`` from the encryption stage, ended by ``None`` or an exception.
        self.items: queue.SimpleQueue[Any] = queue.SimpleQueue()

    def done(self) -> bool:
        return self.future.done()

    def result(self, timeout: float | None = None) -> torch.Tensor:
        """``(rows, m)`` int64 ``W8 r mod 2^K`` (the caller owns and wipes it); raises the stage's error."""
        return self.future.result(timeout)

    def cancel(self) -> None:
        """Stop the job, wait until no stage reads ``r`` any more and wipe the output if it finished."""
        self.aborted.set()
        try:
            out = self.future.result()
        except BaseException:  # cancelled or failed: nothing to wipe
            return
        out.zero_()
