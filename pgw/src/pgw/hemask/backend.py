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
import shutil
import time
import warnings
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
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

__all__ = ["MAX_GROUPS_PER_PASS", "MAX_INFLIGHT", "HEMaskBackend", "HintStore", "verify_rows"]

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
#: Most maskgen requests in flight per backend (``E6``: fills the link on high-latency paths).
MAX_INFLIGHT = 4


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
    """

    def __init__(self, pack: Any, ring: RingParams, hints: HintStore,
                 remote: Callable[[str, np.ndarray], np.ndarray], *, groups_per_call: int = 4,
                 max_inflight: int = 2, groups_per_pass: int = MAX_GROUPS_PER_PASS, pass_bytes: int = 1 << 30):
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
        self.specs = {s.id.key: s for s in pack.linear_specs()}
        for key, norm in hints.meta["row_norms"].items():
            self.params.check_noise(norm, key)
        a = public_matrix(hints.a_seed, max(s.in_features for s in self.specs.values()), self.params)
        # Public matrix as int8 byte planes for the int8 engines (gateway GPU, CPU AMX/VNNI); as uint64
        # for the float64 engine, which would otherwise rebuild the words on every call. Only the
        # ceil(q/8) low planes are kept (7 at q = 2^56): the top byte vanishes mod 2^q.
        self._planes = engine() != "cpu_f64"
        self._a: np.ndarray = self._a_planes(a) if self._planes else a
        del a
        self.stats = {"groups": 0, "encrypt_s": 0.0, "remote_s": 0.0, "decrypt_s": 0.0, "wall_s": 0.0,
                      "passes": 0, "max_inflight": 0}
        self._pool = ThreadPoolExecutor(max_workers=self.max_inflight, thread_name_prefix="pgw-hemask-remote")

    def _a_planes(self, a: np.ndarray, block: int = 1024) -> np.ndarray:
        npl = a_plane_count(self.params)
        out = np.empty((a.shape[0], npl, a.shape[1]), dtype=np.int8)
        for r0 in range(0, a.shape[0], block):
            out[r0:r0 + block] = to_planes(a[r0:r0 + block])[:, :npl]
        return out

    def _encrypt(self, rn: np.ndarray, n: int) -> list[MaskCiphertext]:
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
            """Decrypt the oldest ``pass_groups`` ready groups (fewer at the end), splitting a reply
            across passes when it does not fit."""
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
            self.stats["decrypt_s"] += time.perf_counter() - t1
            self.stats["groups"] += groups
            self.stats["passes"] += 1

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
