"""``pks-v3`` kernels: the native extension ``pgw_pks`` (``rust/pgw-pks``) when present, else the numpy
reference :mod:`pgw.hemask.pks`. Both give the same bits for the same seeds (``tests/hemask/test_pks.py``).

Per (linear, 256-slot group) the customer draws ONE fresh 32-byte seed; the streams of the upload secret
``t`` (``("pks-t", 0)``), its noise (``"pks-e"``), the reply secret ``t'`` (``("pks-t-reply", 0)``) and the
reply key's noise (``"pks-e-ksk"``) are distinct HKDF labels of it (independent). Secrets live in
:class:`GroupSecret` and are wiped after decryption.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np

from pgw.core.errors import ConfigError
from pgw.core.prng import SeedExpander, new_seed
from pgw.hemask import pks
from pgw.hemask.pks import NR, NU, PACK, RANK, D, PKSParams

__all__ = ["Customer", "GroupSecret", "available", "keyswitch", "native"]

_PAR = PKSParams()


def native() -> Any:
    """The ``pgw_pks`` module, or ``None`` (``PGW_NATIVE=0`` / ``PGW_PKS_NATIVE=0`` force the reference)."""
    for var in ("PGW_NATIVE", "PGW_PKS_NATIVE"):
        if os.environ.get(var, "1").strip().lower() in ("0", "false", "no", "off"):
            return None
    try:
        import pgw_pks  # type: ignore[import-not-found,unused-ignore]
    except ImportError:
        return None
    return pgw_pks


def available() -> bool:
    return native() is not None


def _threads(threads: int | None) -> int:
    if threads is not None:
        return max(1, int(threads))
    raw = os.environ.get("PGW_PKS_THREADS", "").strip()
    return max(1, int(raw)) if raw.isdigit() else max(1, os.cpu_count() or 1)


@dataclass
class GroupSecret:
    """One group's secrets: upload ``t`` (4096), reply ``t'`` (8192) and its NTT mod the two q1 primes."""

    t: np.ndarray
    t_reply: np.ndarray
    trhat: np.ndarray
    rows: int

    def wipe(self) -> None:
        for a in (self.t, self.t_reply, self.trhat):
            a.fill(0)


def _ternary(nat: Any, sx: SeedExpander, n: int, label: str) -> np.ndarray:
    if nat is not None:
        v = nat.ternary(sx.stream_key(label, 0), n)
        if v is not None:
            return np.asarray(v, dtype=np.int8)
    from pgw.hemask.lwe import _ternary as ref

    return ref(sx, (n,), label).astype(np.int8)


class Customer:
    """Customer-side ``pks-v3`` kernels for one hint store (the public ``k`` is fixed per store)."""

    def __init__(self, k_hat: np.ndarray, *, threads: int | None = None, force_reference: bool = False):
        if k_hat.shape != (4, RANK, NR):
            raise ConfigError("k_hat must be (4, 16, 8192)")
        self.k_hat = np.ascontiguousarray(k_hat, dtype=np.uint64)
        self.nat = None if force_reference else native()
        self.threads = _threads(threads)
        self._kh = self.nat.KHat(self.k_hat.astype(np.uint32)) if self.nat is not None else None

    @property
    def engine(self) -> str:
        return f"pgw_pks:{self.nat.info()['isa']}" if self.nat is not None else "numpy-reference"

    def encrypt(self, a_seed: bytes, linear: str, r: np.ndarray, *, seeds: list[bytes] | None = None
                ) -> tuple[np.ndarray, np.ndarray, list[GroupSecret]]:
        """Upload of ``r`` (``rows x n`` int64 ring elements) in groups of 256 slots.

        Returns ``(b, beta, secrets)``: ``b`` ``(n, 256 G)`` uint64 mod 2^56 (today's layout), ``beta``
        ``(G, 4, 16, 8192)`` uint32 reply keys, one :class:`GroupSecret` per group. ``seeds`` (tests): the
        per-group seeds; fresh by default.
        """
        rows, n = r.shape
        if n % RANK or rows == 0:
            raise ConfigError(f"pks-v3: in_features {n} must be a multiple of {RANK}")
        groups = -(-rows // D)
        seeds = [new_seed() for _ in range(groups)] if seeds is None else seeds
        sxs = [SeedExpander(s) for s in seeds]
        secrets: list[GroupSecret] = []
        beta = np.empty((groups, 4, RANK, NR), dtype=np.uint32)
        r = np.ascontiguousarray(r, dtype=np.int64)
        ts: np.ndarray | None = None
        try:
            if self.nat is not None:
                ts = np.empty((groups, NU), dtype=np.int8)
                for g, sx in enumerate(sxs):
                    tg = _ternary(self.nat, sx, NU, "pks-t")
                    ts[g] = tg
                    tg.fill(0)
                b = self.nat.encrypt(SeedExpander(a_seed).stream_key(pks.A_LABEL, linear), r, ts,
                                     [sx.stream_key("pks-e") for sx in sxs], self.threads)
                for g, sx in enumerate(sxs):
                    tr = _ternary(self.nat, sx, NR, "pks-t-reply")
                    bg, th = self.nat.reply_key(self._kh, ts[g], tr, sx.stream_key("pks-e-ksk"))
                    beta[g] = bg
                    secrets.append(GroupSecret(ts[g].copy(), tr, np.asarray(th), min(D, rows - g * D)))
                ts.fill(0)
                return b, beta, secrets
            a_ring = pks.public_ring_rows(a_seed, n, linear)
            parts = []
            for g, s in enumerate(seeds):
                rg = r[g * D:(g + 1) * D]
                ct = pks.encrypt_rlwe(rg, a_ring, _PAR, _seed=s)
                key = pks.make_reply_key(ct, self.k_hat, _PAR, _seed=s)
                parts.append(ct.b)
                beta[g] = key.beta_hat.astype(np.uint32)
                th = np.stack([pks.Ntt.of(p).fwd(pks._residues(key.t_reply, p)) for p in pks.Q1]).astype(np.uint32)
                secrets.append(GroupSecret(ct.t.astype(np.int8), key.t_reply.astype(np.int8), th, rg.shape[0]))
                ct.t.fill(0)
                key.wipe()
            return np.concatenate(parts, axis=1), beta, secrets
        except BaseException:
            for s in secrets:
                s.wipe()
            if ts is not None:
                ts.fill(0)
            raise

    def decrypt(self, hint: np.ndarray, y: bytes, secrets: list[GroupSecret]) -> np.ndarray:
        """``W8 r mod 2^32`` (``(sum rows, m)`` uint32, groups stacked) from the packed reply ``y``
        (``(G, m/32, 8192)`` values at 36 bits) and the stored compact hint ``(m/32, 2, 8192, 3)``. Wipes
        the secrets."""
        try:
            nb = hint.shape[0]
            rows = [s.rows for s in secrets]
            if self.nat is not None:
                th = np.ascontiguousarray(np.stack([s.trhat for s in secrets]), dtype=np.uint32)
                try:
                    out: np.ndarray = self.nat.decrypt(hint, bytes(y), _PAR.yb, th, rows, self.threads)
                finally:
                    th.fill(0)
                return out
            from pgw.hemask.wire import unpack_bits

            c0 = unpack_bits(y, len(secrets) * nb * NR, _PAR.yb).reshape(len(secrets), nb, NR)
            parts = [pks.decrypt_pks(c0[g], hint, s.t_reply.astype(np.int64), s.rows, _PAR)
                     for g, s in enumerate(secrets)]
            return np.concatenate(parts, axis=0).astype(np.uint32)
        finally:
            for s in secrets:
                s.wipe()


def keyswitch(hs: np.ndarray, ops: np.ndarray, y: np.ndarray | None = None, *, threads: int | None = None,
              force_reference: bool = False) -> np.ndarray:
    """The key switch of one linear: provider reply ``(G, m/32, 8192)`` (``y`` = ``W8 b mod 2^56``,
    ``m x 256 G``) or, with ``y=None`` and one operand (the public ``k``), the compact hint
    ``(m/32, 2, 8192, 3)`` uint8. ``hs`` = :func:`pgw.hemask.pks.switch_hint` of ``H``."""
    nat = None if force_reference else native()
    m = hs.shape[0]
    if m % PACK or hs.shape[1] != NU:
        raise ConfigError(f"pks-v3: out_features {m} must be a multiple of {PACK}")
    if nat is not None:
        hs_c = np.ascontiguousarray(hs, dtype=np.uint64)
        ops_c = np.ascontiguousarray(ops, dtype=np.uint32)
        if y is None:
            v = nat.keyswitch(hs_c, ops_c, None, _PAR.yb, _threads(threads)).astype("<u4")
            return np.ascontiguousarray(v.view(np.uint8).reshape(*v.shape, 4)[..., :3])
        out: np.ndarray = nat.keyswitch(hs_c, ops_c, np.ascontiguousarray(y, dtype=np.uint64), _PAR.yb,
                                        _threads(threads))
        return out
    if y is None:
        return pks.compact_hint(hs, ops[0].astype(np.uint64), _PAR)
    g = ops.shape[0]
    return np.stack([pks.provider_reply(y[:, i * D:(i + 1) * D], hs, ops[i].astype(np.uint64), _PAR)
                     for i in range(g)])
