"""fused2 K56 masks from the existing encrypted factories by digit decomposition (opt-in, default off).

``research/plans/fused2-two-trips-per-layer.md`` section 11, option (b): no new cryptography. The encrypted
factories compute ``W8 x mod 2^K_he`` blind for any ``x`` in ``Z_{2^K_he}`` (mlwe v2 at ``K_he = 40``; pks-v3 at
``K_he = 32``), but fused2 needs ``W r mod 2^56``. Write the uniform mask row ``r`` in base ``2^b``::

    r = sum_{i < D} 2^(b i) d_i,      d_i = (r >> b i) & (2^b - 1),     D = ceil(56 / b)

**Exactness.** Over the integers ``W r = sum_i 2^(b i) (W d_i)`` (linearity). Every entry of ``W d_i`` is a sum of
``n`` products ``w * d`` with ``|w| <= 128`` and ``0 <= d < 2^b``, so ``|W d_i| < 128 * 2^b * n``. When that is at
most ``2^(K_he - 1)`` (checked per linear: :func:`digit_bound_ok`), the factory's exact result ``W d_i mod
2^K_he`` determines ``W d_i`` itself: center it into ``[-2^(K_he-1), 2^(K_he-1))``. The recombination
``sum_i 2^(b i) (W d_i)`` is then computed in wrapping 64-bit integer arithmetic, which is exact mod ``2^64`` and
therefore mod ``2^56`` (``2^56`` divides ``2^64``): the result is ``W r mod 2^56`` bit for bit. Profiles
(:data:`PROFILES`): mlwe v2 at K40 with 14-bit digits, 4 products per mask (``128 * 2^14 * n <= 2^39`` for
``n <= 2^18``: every Qwen2.5 linear up to 72B's ``n = 29568``); pks-v3 at K32 with 8-bit digits, 7 products
(``128 * 2^8 * n <= 2^31`` for ``n <= 2^16``).

**Privacy (the provider's view is unchanged).** Each digit plane ``d_i`` is handed to the existing encrypted
backend exactly like a K40 / K32 mask row: it is a valid element of ``Z_{2^K_he}``, encrypted under a fresh MLWE
secret with fresh noise per request (the scheme is semantically secure for any message, so the message's
distribution does not matter; the digits are moreover uniform on their range and independent of each other, as
``r`` is uniform), and each request is an ordinary v2 / pks-v3 maskgen request of public shape (linear name,
256-slot groups, the same wire). The provider sees ``D`` times the requests of a K40 / K32 segment of the same
slots, which is public (a function of the slot count). Decryption correctness is the scheme's own (the same
``Delta``, the same noise bound ``W8 e``, the same per-linear norm check at start-up); any wrong product is caught
before a segment is written by the fused2 factory's private per-chunk self-check on the recombined ``W r mod 2^56``
(:meth:`pgw.factory.fused2_gen.Fused2MaskGenerator._self_check`) and the start-up check. The p61 check words
``(u * hmul) W mod p`` need no HE (public ``W``, customer ``u``).

The public fused2 linears (folded ``qkv/f2``, ``gate_up/f2``) are not pack linears, so the encrypted backend sees
them through :class:`Fused2HEPack` (the fused2 lane fingerprint, the fused2 weights, file-safe linear names
``"<layer>.qkv-f2"``: :func:`he_key`); the provider serves the same names for that fingerprint from its fused2
lane entry (``PGW_SERVICE_FUSED2_MASKGEN=1``, :mod:`pgw.service.fused2_lane`).
"""

from __future__ import annotations

import logging
import time
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from pgw.core.errors import ConfigError

__all__ = ["PROFILES", "DigitSplitProducts", "Fused2HEPack", "TopDigitProducts", "build_fused2_he", "digit_bound_ok",
           "digit_plan", "he_key", "recombine_digits", "recombine_plan", "split_digits", "split_plan"]

log = logging.getLogger(__name__)

#: HE scheme -> (HE ring bits ``K_he``, digit bits ``b``).
PROFILES: dict[str, tuple[int, int]] = {"mlwe": (40, 14), "pks-v3": (32, 8)}
#: The HE ring each scheme runs at (``build_he_backend``'s ``ring`` argument).
HE_RINGS: dict[str, str] = {"mlwe": "K40A12", "pks-v3": "K32A12C27"}


def he_key(key: str) -> str:
    """File- and wire-safe name of a fused2 linear (``"3.gate_up/f2"`` -> ``"3.gate_up-f2"``)."""
    return key.replace("/", "-")


def digit_bound_ok(n: int, digit_bits: int, k_he: int) -> bool:
    """``|W d| < 128 * 2^b * n <= 2^(K_he - 1)``: a centered ``W d mod 2^K_he`` is the exact integer."""
    return 128 * (1 << int(digit_bits)) * int(n) <= 1 << (int(k_he) - 1)


def split_digits(r: torch.Tensor, k: int, digit_bits: int) -> list[torch.Tensor]:
    """``r`` (int64 in ``[0, 2^k)``) -> ``ceil(k / b)`` int64 digit planes in ``[0, 2^b)`` (least significant first)."""
    b = int(digit_bits)
    mask = (1 << b) - 1
    return [(r >> (b * i)) & mask for i in range(-(-int(k) // b))]


def recombine_digits(ys: list[torch.Tensor], k: int, k_he: int, digit_bits: int) -> torch.Tensor:
    """``sum_i 2^(b i) center(y_i) mod 2^k`` in wrapping int64 arithmetic (exact mod ``2^64``, so mod ``2^k``)."""
    half = 1 << (int(k_he) - 1)
    full = 1 << int(k_he)
    acc: np.ndarray | None = None
    for i, y in enumerate(ys):
        yc = torch.where(y >= half, y - full, y)  # the exact W d_i (|W d_i| < 2^(K_he - 1)), int64
        u = np.ascontiguousarray(yc.numpy()).view(np.uint64)  # two's complement = the value mod 2^64
        term = u << np.uint64(int(digit_bits) * i)  # unsigned: wraps mod 2^64 (well defined)
        if acc is None:
            acc = term
        else:
            acc += term
            term.fill(0)
        yc.zero_()
    assert acc is not None
    acc &= np.uint64((1 << int(k)) - 1)
    return torch.from_numpy(acc.view(np.int64))


# ---- the top-digit plan (``PGW_FUSED2_HE_DIGITS=top``, default off): fewer products, same exactness ----------
#
# Every digit at shift ``s`` contributes ``2^s (W d) mod 2^k``, so only ``W d mod 2^(k - s)`` is needed. For the
# one digit at ``s = k - K_he`` (the top ``K_he`` bits of ``r``) that is ``W d mod 2^K_he``: exactly what the
# encrypted factory returns, for ANY ``d`` in ``Z_{2^K_he}`` and with no exactness bound (24 + 32 = 56 at pks-v3).
# Only the low ``k - K_he`` bits need exact products, and BALANCED digits (``|d| <= 2^(b-1)``) give one more bit:
# ``|W d| <= 128 * 2^(b-1) * n < 2^(K_he - 1)``. At pks-v3 (K32) and Qwen2.5 widths that is ``b = 12`` for
# ``n <= 4096 * 2`` (d = 3584 / 5120: 2 low digits) and ``b = 10`` for ``n <= 2^15`` (I = 18944 / 27648: 3 low
# digits): 3 or 4 products per mask instead of 7 (8-bit unsigned digits), with the same scheme, the same wire and
# the same provider view (each digit plane is an ordinary K32 mask row: the scheme hides any message).


def digit_plan(n: int, k: int, k_he: int) -> list[tuple[int, int, bool]]:
    """``[(shift, bits, exact), ...]``: balanced exact low digits of the largest width with
    ``128 * 2^(b-1) * n < 2^(K_he - 1)``, then one top digit of ``K_he`` bits taken mod ``2^K_he``."""
    low = int(k) - int(k_he)
    if low < 0:
        raise ConfigError("digit_plan: K_he above k")
    b = 1
    while 128 * (1 << b) * int(n) < (1 << (int(k_he) - 1)):  # b + 1 still exact: 128 * 2^((b+1)-1) * n < 2^(K-1)
        b += 1
    if b < 2:
        raise ConfigError(f"digit_plan: no exact digit at K{k_he} for n = {n}")
    plan, s = [], 0
    while s < low:
        w = min(b, low - s)
        plan.append((s, w, True))
        s += w
    plan.append((low, int(k_he), False))
    return plan


def split_plan(r: torch.Tensor, plan: list[tuple[int, int, bool]], k_he: int) -> list[torch.Tensor]:
    """The digit planes of ``r`` (int64 in ``[0, 2^k)``) under ``plan``, each as a K_he ring element in
    ``[0, 2^K_he)`` (a negative balanced digit as its residue mod ``2^K_he``). ``r = sum 2^s d`` mod ``2^k``."""
    rem = r.clone()
    full = (1 << int(k_he)) - 1
    out = []
    for s, w, exact in plan:
        if exact:
            d = (rem >> s) & ((1 << w) - 1)
            d = torch.where(d >= (1 << (w - 1)), d - (1 << w), d) if w > 1 else d  # balanced
            rem = rem - (d << s)  # exact int64: |rem| stays below 2^57
            out.append(d & full)
            d.zero_()
        else:
            out.append((rem >> s) & full)  # the top bits (a carry past 2^k vanishes mod 2^k)
    rem.zero_()
    return out


def recombine_plan(ys: list[torch.Tensor], plan: list[tuple[int, int, bool]], k: int, k_he: int) -> torch.Tensor:
    """``sum 2^s y`` mod ``2^k``: exact digits' products centred (``|W d| < 2^(K_he - 1)``), the top digit's taken
    as its residue mod ``2^K_he`` (only that is needed: ``2^s 2^K_he = 2^k``). Wrapping uint64 arithmetic."""
    half, fullv = 1 << (int(k_he) - 1), 1 << int(k_he)
    acc: np.ndarray | None = None
    for (s, _w, exact), y in zip(plan, ys, strict=True):
        yc = torch.where(y >= half, y - fullv, y) if exact else y & (fullv - 1)
        term = np.ascontiguousarray(yc.numpy()).view(np.uint64) << np.uint64(s)
        acc = term if acc is None else acc + term
        yc.zero_()
    assert acc is not None
    acc &= np.uint64((1 << int(k)) - 1)
    return torch.from_numpy(acc.view(np.int64))


class TopDigitProducts:
    """``W r mod 2^k`` with the top-digit plan (:func:`digit_plan`, per linear): 3-4 products per K56 mask at
    pks-v3 instead of 7. Same interface as :class:`DigitSplitProducts`.

    ``stack`` (default): the digit planes of a chunk go to the backend as ONE call of ``len(plan) x t`` mask rows
    (each row an ordinary K_he mask row, encrypted in 256-slot groups under fresh secrets, exactly like more slots).
    The provider then serves all planes of a linear in one item, so its public hint (``W8 A``, 85 % of a one-group
    item) is computed once per chunk instead of once per plane, and a short chunk's planes share groups instead of
    each padding its own."""

    def __init__(self, backend: Any, *, k: int, k_he: int, shapes: dict[str, tuple[int, int]], scheme: str = "",
                 stack: bool = True):
        self.backend, self.k, self.k_he, self.scheme, self.stack = backend, int(k), int(k_he), scheme, bool(stack)
        self.plans = {key: digit_plan(n, k, k_he) for key, (_m, n) in shapes.items()}
        self.shapes = dict(shapes)
        self.stats = {"digit_products": 0, "rows": 0, "he_s": 0.0, "split_s": 0.0, "recombine_s": 0.0}

    @property
    def resolved_backend(self) -> str:
        counts = sorted({len(p) for p in self.plans.values()})
        return f"he-topdigits:{self.scheme or 'he'}:K{self.k_he}x{'/'.join(map(str, counts))}"

    def matmul(self, key: str, r: torch.Tensor) -> torch.Tensor:
        plan = self.plans[key]
        t0 = time.perf_counter()
        ds = split_plan(r, plan, self.k_he)
        t1 = time.perf_counter()
        ys: list[torch.Tensor] = []
        try:
            if self.stack:
                cat = torch.cat(ds)
                y = self.backend.matmul(he_key(key), cat)
                cat.zero_()
                y = y if y.dtype == torch.int64 else y.to(torch.int64)
                ys = [c.clone() for c in torch.split(y, int(r.shape[0]))]
                y.zero_()
            else:
                for d in ds:
                    y = self.backend.matmul(he_key(key), d)
                    ys.append(y if y.dtype == torch.int64 else y.to(torch.int64))
                    d.zero_()
            t2 = time.perf_counter()
            out = recombine_plan(ys, plan, self.k, self.k_he)
        finally:
            for d in ds:
                d.zero_()
            for y in ys:
                y.zero_()
        st = self.stats
        st["digit_products"] += len(ds)
        st["rows"] += int(r.shape[0])
        st["split_s"] += t1 - t0
        st["he_s"] += t2 - t1
        st["recombine_s"] += time.perf_counter() - t2
        return out


class Fused2HEPack:
    """The public fused2 weights as the encrypted backends' ``pack`` (specs, ``iter_linears``, ``load_linear``)."""

    def __init__(self, weights: Any, fingerprint: str, name: str = "fused2"):
        self.fingerprint = str(fingerprint)
        self.short_fingerprint = self.fingerprint[:12]
        self.name = name
        self._lins = {he_key(lin.key): lin for lin in weights.linears()}

    @staticmethod
    def _id(key: str) -> Any:
        return SimpleNamespace(key=key, kind=SimpleNamespace(value="f2"))

    def linear_specs(self) -> list[Any]:
        return [SimpleNamespace(id=self._id(k), out_features=int(lin.m), in_features=int(lin.n))
                for k, lin in self._lins.items()]

    def load_linear(self, lid: Any, device: str = "cpu") -> Any:
        key = getattr(lid, "key", lid)
        return SimpleNamespace(id=self._id(key), w8=self._lins[key].w8)

    def iter_linears(self) -> Any:
        for k, lin in self._lins.items():
            yield SimpleNamespace(id=self._id(k), w8=lin.w8)


class DigitSplitProducts:
    """``W r mod 2^k`` from an encrypted backend's ``matmul(he_key, x) -> W x mod 2^K_he`` (see the module doc).

    Args:
        backend: :class:`pgw.hemask.backend.HEMaskBackend` (mlwe) or :class:`pgw.hemask.pks_backend.PKSMaskBackend`
            over :class:`Fused2HEPack` (or anything with that ``matmul``).
        k: The fused2 ring bits (56).
        k_he: The backend's ring bits (40 / 32).
        digit_bits: ``b`` (14 / 8).
        shapes: ``{fused2 key: (m, n)}``: every linear's bound is checked here (refused before any request).
    """

    def __init__(self, backend: Any, *, k: int, k_he: int, digit_bits: int, shapes: dict[str, tuple[int, int]],
                 scheme: str = ""):
        if not 1 <= int(digit_bits) < int(k_he):
            raise ConfigError("digit_bits must be in [1, K_he)")
        bad = [key for key, (_m, n) in shapes.items() if not digit_bound_ok(n, digit_bits, k_he)]
        if bad:
            raise ConfigError(f"fused2 digit split: {digit_bits}-bit digits at K{k_he} are not exact for {bad[:3]} "
                              f"(needs 128 * 2^b * n <= 2^(K_he - 1))")
        self.backend = backend
        self.k, self.k_he, self.digit_bits = int(k), int(k_he), int(digit_bits)
        self.digits = -(-self.k // self.digit_bits)
        self.scheme = scheme
        self.shapes = dict(shapes)
        self.stats = {"digit_products": 0, "rows": 0, "he_s": 0.0, "split_s": 0.0, "recombine_s": 0.0}

    @property
    def resolved_backend(self) -> str:
        return f"he-digits:{self.scheme or 'he'}:K{self.k_he}x{self.digits}"

    def matmul(self, key: str, r: torch.Tensor) -> torch.Tensor:
        """``W r mod 2^k`` (int64 ``(t, m)``), ``r`` int64 ``(t, n)`` in ``[0, 2^k)``."""
        t0 = time.perf_counter()
        ds = split_digits(r, self.k, self.digit_bits)
        t1 = time.perf_counter()
        ys: list[torch.Tensor] = []
        try:
            for d in ds:
                y = self.backend.matmul(he_key(key), d)
                ys.append(y if y.dtype == torch.int64 else y.to(torch.int64))
                d.zero_()
            t2 = time.perf_counter()
            out = recombine_digits(ys, self.k, self.k_he, self.digit_bits)
        finally:
            for d in ds:
                d.zero_()
            for y in ys:
                y.zero_()
        t3 = time.perf_counter()
        st = self.stats
        st["digit_products"] += len(ds)
        st["rows"] += int(r.shape[0])
        st["split_s"] += t1 - t0
        st["he_s"] += t2 - t1
        st["recombine_s"] += t3 - t2
        return out


def build_fused2_he(settings: Any, weights: Any, fingerprint: str, k: int, *, remote_backend: Any = None,
                    worker_index: int = 0, workers: int = 1) -> DigitSplitProducts:
    """The digit-split HE product of the factory's ``he_scheme`` for ``weights`` under the fused2 lane
    ``fingerprint`` (``remote_backend``: a prebuilt backend over :class:`Fused2HEPack`, tests)."""
    scheme = str(getattr(settings, "he_scheme", "mlwe"))
    if scheme not in PROFILES:
        raise ConfigError(f"fused2 digit split: unknown he_scheme {scheme!r}")
    k_he, b = PROFILES[scheme]
    adapter = Fused2HEPack(weights, fingerprint)
    if remote_backend is None:
        from pgw.factory.runtime import build_he_backend

        remote_backend = build_he_backend(settings, adapter, HE_RINGS[scheme], worker_index=worker_index,
                                          workers=workers)
    shapes = {lin.key: (int(lin.m), int(lin.n)) for lin in weights.linears()}
    import os

    if os.environ.get("PGW_FUSED2_HE_DIGITS", "").strip().lower() == "top":  # opt-in: 3-4 products instead of 7
        top = TopDigitProducts(remote_backend, k=k, k_he=k_he, shapes=shapes, scheme=scheme)
        log.info("fused2 factory: K%d masks by the top-digit plan on the encrypted %s factory (%s)", k, scheme,
                 top.resolved_backend)
        return top  # type: ignore[return-value]
    dsp = DigitSplitProducts(remote_backend, k=k, k_he=k_he, digit_bits=b, shapes=shapes, scheme=scheme)
    log.info("fused2 factory: K%d masks as %d x %d-bit digits on the encrypted %s factory (K%d)", k, dsp.digits, b,
             scheme, k_he)
    return dsp


def he_customer_seconds(backend: Any) -> float:
    """Customer-side seconds of an encrypted backend's stats (encryption + decryption; not the provider)."""
    st = getattr(backend, "stats", {}) or {}
    return float(st.get("encrypt_s", 0.0)) + float(st.get("decrypt_s", 0.0))


def as_u64(t: torch.Tensor) -> np.ndarray:  # pragma: no cover - debugging helper
    return np.ascontiguousarray(t.numpy()).view(np.uint64)
