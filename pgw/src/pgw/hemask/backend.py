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
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pgw.core.errors import ConfigError, IntegrityError
from pgw.core.prng import new_seed
from pgw.core.ring import RingParams
from pgw.hemask.lwe import DEGREE, HEParams, MaskCiphertext, encrypt_masks, matmul_mod64, public_matrix

__all__ = ["HEMaskBackend", "HintStore"]

log = logging.getLogger(__name__)

_META = "meta.json"
_HINT_VERSION = 1
_BUILD_COLS = 512  # hint columns per GEMM block (bounds temporary memory)


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
    def dir_for(root: Path, fingerprint: str, rank: int) -> Path:
        return Path(root) / f"{fingerprint[:16]}-r{rank}"

    @classmethod
    def open(cls, root: Path, fingerprint: str, rank: int) -> HintStore | None:
        path = cls.dir_for(root, fingerprint, rank)
        try:
            meta = json.loads((path / _META).read_text())
        except FileNotFoundError:
            return None
        if meta.get("version") != _HINT_VERSION or meta.get("fingerprint") != fingerprint or meta.get("rank") != rank:
            raise ConfigError(f"hint store {path} does not match fingerprint/rank; delete it to rebuild")
        return cls(path, meta)

    @classmethod
    def build(cls, root: Path, pack: Any, rank: int, *, log_every_s: float = 30.0) -> HintStore:
        """Compute ``W8 A`` for every linear of ``pack`` (minutes for 7B on CPU; done once per pack)."""
        params = HEParams(k=40, rank=rank)  # the hint does not depend on K
        final = cls.dir_for(root, pack.fingerprint, rank)
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
                out = np.lib.format.open_memmap(tmp / f"{key}.npy", mode="w+", dtype=np.uint64,
                                                shape=(m, params.dim))
                for c0 in range(0, params.dim, _BUILD_COLS):
                    out[:, c0:c0 + _BUILD_COLS] = matmul_mod64(w, a[:n, c0:c0 + _BUILD_COLS])
                out.flush()
                del out
                done += m
                if time.monotonic() - last > log_every_s:
                    last = time.monotonic()
                    log.info("building HE hint: %.0f%% (%.0f s)", 100 * done / total, last - t0)
            meta = {"version": _HINT_VERSION, "fingerprint": pack.fingerprint, "rank": rank, "a_seed": seed.hex(),
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
    def open_or_build(cls, root: Path, pack: Any, rank: int) -> HintStore:
        hs = cls.open(root, pack.fingerprint, rank)
        return hs if hs is not None else cls.build(root, pack, rank)

    def hint(self, linear: str) -> np.ndarray:
        h = self._maps.get(linear)
        if h is None:
            h = np.load(self.path / f"{linear}.npy", mmap_mode="r")
            self._maps[linear] = h
        return h

    @property
    def nbytes(self) -> int:
        return sum(p.stat().st_size for p in self.path.glob("*.npy"))


class HEMaskBackend:
    """``W8 r mod 2^K`` via the provider, for one pack and ring.

    Args:
        pack: Open model pack (for linear specs; the hint store was built from it).
        ring: Mask ring.
        hints: :class:`HintStore` of this pack.
        remote: ``remote(linear, b) -> y`` computing ``W8 b mod 2^64`` on the provider (e.g.
            :class:`pgw.hemask.client.RemoteMulmod64`).
        groups_per_call: 256-slot groups sent per request (fewer round trips, larger messages).
    """

    def __init__(self, pack: Any, ring: RingParams, hints: HintStore,
                 remote: Callable[[str, np.ndarray], np.ndarray], *, groups_per_call: int = 4):
        self.params = HEParams(k=ring.k, rank=hints.rank)
        self.ring = ring
        self.hints = hints
        self.remote = remote
        self.groups_per_call = max(1, min(16, int(groups_per_call)))
        self.specs = {s.id.key: s for s in pack.linear_specs()}
        for key, norm in hints.meta["row_norms"].items():
            self.params.check_noise(norm, key)
        self._a = public_matrix(hints.a_seed, max(s.in_features for s in self.specs.values()), self.params)
        self.stats = {"groups": 0, "encrypt_s": 0.0, "remote_s": 0.0, "decrypt_s": 0.0}

    def matmul(self, linear: str, r: torch.Tensor) -> torch.Tensor:
        """``(rows, n)`` masks -> ``(rows, m)`` ``W8 r mod 2^K`` (int64 CPU tensor)."""
        spec = self.specs[linear]
        n, m = spec.in_features, spec.out_features
        if r.ndim != 2 or r.shape[1] != n:
            raise ConfigError(f"{linear}: masks must be (rows, {n}), got {tuple(r.shape)}")
        rows = int(r.shape[0])
        rn = r.numpy() if isinstance(r, torch.Tensor) else np.asarray(r)
        a = self._a[:n]
        hint = self.hints.hint(linear)
        out = np.empty((rows, m), dtype=np.int64)
        step = DEGREE * self.groups_per_call
        for s0 in range(0, rows, step):
            t0 = time.perf_counter()
            cts: list[MaskCiphertext] = [encrypt_masks(rn[g0:min(g0 + DEGREE, rows)], a, self.params)
                                         for g0 in range(s0, min(s0 + step, rows), DEGREE)]
            b = np.concatenate([ct.b for ct in cts], axis=1)
            t1 = time.perf_counter()
            try:
                y = self.remote(linear, b)
                if y.shape != (m, b.shape[1]):
                    raise IntegrityError(f"maskgen {linear}: provider returned shape {y.shape}, expected "
                                         f"{(m, b.shape[1])}")
                t2 = time.perf_counter()
                # One GEMM for all groups: hint @ [S_1 | S_2 | ...]
                smat = np.concatenate([ct.secret for ct in cts], axis=1)
                v = y - matmul_mod64(smat, np.asarray(hint), small_left=False)
                smat.fill(0)
            finally:
                for ct in cts:
                    ct.wipe()
            half = np.uint64(1 << (self.params.delta_bits - 1))
            wr = ((v + half) >> np.uint64(self.params.delta_bits)) & np.uint64(self.ring.mask)
            for gi, ct in enumerate(cts):
                g0 = s0 + gi * DEGREE
                out[g0:g0 + ct.rows] = wr[:, gi * DEGREE:gi * DEGREE + ct.rows].T.astype(np.int64)
            t3 = time.perf_counter()
            self.stats["groups"] += len(cts)
            self.stats["encrypt_s"] += t1 - t0
            self.stats["remote_s"] += t2 - t1
            self.stats["decrypt_s"] += t3 - t2
        return torch.from_numpy(out)
