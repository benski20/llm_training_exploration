"""Gateway/factory side of ``POST /pgw/v1/maskgen`` on top of :class:`pgw.wire.ComputeClient`.

Wire v2 (:mod:`pgw.hemask.wire`) sends fewer bits each way. The customer picks the bits per linear
and checks the decryption noise budget for each choice:

* **Reply: ``K + guard`` bits.** Decryption reads bits ``delta .. q-1`` of ``y - H s`` and rounds
  at ``delta = q - K``, so the provider can round ``y`` to ``K + guard`` bits. The rounding error,
  ``2^(q - K - guard - 1)``, is part of the budget. With the default guard of 3 bits, the K32 ring
  gets 35-bit replies instead of 40-bit ones.
* **Upload: ``q - 3`` bits (53 at q = 2^56).** The customer switches its own ciphertext from modulus
  ``2^q`` to ``2^(q-3)`` with randomised rounding, ``b' = floor((b + d) / 8)``, where ``d`` is fresh
  and uniform in ``[0, 8)`` for each coefficient. The provider computes ``W8 b' mod 2^(q-3)``, and
  ``8 W8 b' = W8 b - W8 eps`` with ``eps = b - 8 b'``. Given ``b``, each ``eps`` takes the two values
  ``c - 8`` and ``c``, where ``c = b mod 8``, with mean exactly 0. They are independent across
  coefficients. By Hoeffding, ``|W8 eps| < tail_sigmas · 4 · ||w||_2`` except with probability
  ``2 exp(-tail_sigmas^2 / 2)``, the same tail as the ``W8 e`` term. The scaling factor stays
  ``2^delta``, the hint is unchanged, and ``b'`` is a randomised function of ``b`` alone. The
  provider therefore learns nothing it could not compute from ``b``. Lattice security is that of
  the ``2^q`` ciphertext. This is not the rejected 25-bit upload variant, which redefined the mask
  from a public value and leaked noise.

The noise budget for a format ``(bb, yb)`` is::

    tail·sigma·||w|| + [bb < q] tail·2^(q-bb-1)·||w|| + [yb < bb] 2^(q-yb-1) + 255·dim
         <  2^(delta - 1 - margin_bits)

The first term is ``W8 e``, the second the upload switch, the third the reply rounding, and the
last the hint's missing low byte. Each term is a high-probability or worst-case bound. The client
checks this for every linear at startup and again before every v2 request, and it uses v1 for any
linear where no format fits. Decrypted masks are exact either way, so they are identical to v1,
and the factory's every-row check and the online Freivalds checks are unchanged.
"""

from __future__ import annotations

import logging
import math
import os
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from pgw.core.errors import ConfigError, ErrorCode, PGWError
from pgw.hemask.lwe import HEParams
from pgw.hemask.wire import (
    MASKGEN_PATH,
    V2_MIN_BITS,
    MaskgenItem,
    MaskgenRequest,
    decode_maskgen_response,
    encode_maskgen_request,
    pack_b,
    pack_bits,
    unpack_bits,
    unpack_y,
)

__all__ = [
    "DEFAULT_GUARD_BITS",
    "DEFAULT_MARGIN_BITS",
    "UPLOAD_SWITCH_BITS",
    "RemoteMulmod64",
    "V2Format",
    "check_v2",
    "switch_down",
    "v2_formats",
    "v2_noise_bound",
]

log = logging.getLogger(__name__)

#: Bits the customer's randomised modulus switch removes from each uploaded coefficient (53 at q = 2^56).
UPLOAD_SWITCH_BITS = 3
#: Reply bits above the ring: ``y`` comes back at ``K + DEFAULT_GUARD_BITS`` bits (35 for K32).
DEFAULT_GUARD_BITS = 3
#: The v2 noise bound must stay below ``2^(delta - 1 - margin)``: half the scaling factor, halved again.
DEFAULT_MARGIN_BITS = 1
#: After a v2 request is refused, v1 is used and advertisements are ignored for this long.
V2_RETRY_S = 300.0


@dataclass(frozen=True)
class V2Format:
    """One wire v2 format: ``b_bits`` per uploaded coefficient, ``y_bits`` per returned element."""

    b_bits: int
    y_bits: int
    #: High-probability bound on the decryption error with this format (units of ``1 mod 2^q``).
    bound: float

    def bytes_per_slot(self, n: int, m: int) -> float:
        """Wire bytes of one slot of a linear with ``n`` inputs and ``m`` outputs (up + down)."""
        return (n * self.b_bits + m * self.y_bits) / 8


def v2_noise_bound(params: HEParams, max_row_l2: float, b_bits: int, y_bits: int) -> float:
    """Decryption error bound of wire v2 format ``(b_bits, y_bits)`` for weight rows of norm ``<= max_row_l2``.

    ``W8 e`` (``tail·sigma·||w||``) + the upload switch from ``2^q`` to ``2^b_bits``
    (``tail·2^(q-b_bits-1)·||w||``, Hoeffding) + the reply rounding (``2^(q-y_bits-1)``) + the hint's
    dropped low byte (``255·dim``).
    """
    q = params.q
    if not V2_MIN_BITS <= y_bits <= b_bits <= q:
        raise ValueError(f"v2 format ({b_bits}, {y_bits}) does not fit q = 2^{q}")
    switch = q - b_bits
    bound = params.tail_sigmas * params.sigma * max_row_l2
    if switch:
        bound += params.tail_sigmas * 2.0 ** (switch - 1) * max_row_l2
    if y_bits < b_bits:
        bound += 2.0 ** (q - y_bits - 1)
    return bound + params.HINT_DROP_ERROR * params.dim


def _budget(params: HEParams, margin_bits: int) -> float:
    return 2.0 ** (params.delta_bits - 1 - margin_bits)


def check_v2(params: HEParams, max_row_l2: float, fmt: V2Format, *, margin_bits: int = DEFAULT_MARGIN_BITS,
             label: str = "") -> None:
    """Raise :class:`ConfigError` unless ``fmt``'s noise bound is below ``2^(delta - 1 - margin_bits)``."""
    bound = v2_noise_bound(params, max_row_l2, fmt.b_bits, fmt.y_bits)
    if not bound < _budget(params, margin_bits):
        raise ConfigError(f"maskgen wire v2 ({fmt.b_bits}/{fmt.y_bits} bits) noise budget exceeded"
                          f"{(' for ' + label) if label else ''}: bound 2^{math.log2(bound):.2f} >= "
                          f"2^{params.delta_bits - 1 - margin_bits}")


def v2_formats(params: HEParams, max_row_l2: float, *, upload_switch: bool = True,
               guard_bits: int = DEFAULT_GUARD_BITS, margin_bits: int = DEFAULT_MARGIN_BITS) -> tuple[V2Format, ...]:
    """The v2 formats that keep the noise budget, one per upload width (empty: refuse v2).

    For each upload width (``q``, and ``q - UPLOAD_SWITCH_BITS`` with ``upload_switch``), the result
    has the narrowest reply of at least ``K + guard_bits`` bits whose bound passes :func:`check_v2`.
    """
    if guard_bits < 1 or margin_bits < 0:
        raise ConfigError("guard_bits must be >= 1 and margin_bits >= 0")
    if not max_row_l2 >= 0 or not math.isfinite(max_row_l2):
        raise ConfigError(f"invalid weight row norm {max_row_l2!r}")
    q, k = params.q, params.k
    budget = _budget(params, margin_bits)
    out = []
    for bb in ((q, q - UPLOAD_SWITCH_BITS) if upload_switch else (q,)):
        for yb in range(max(V2_MIN_BITS, k + guard_bits), bb + 1):
            bound = v2_noise_bound(params, max_row_l2, bb, yb)
            if bound < budget:
                out.append(V2Format(bb, yb, bound))
                break
    return tuple(out)


def switch_down(b: np.ndarray, q_bits: int, b_bits: int, rng: np.random.Generator) -> np.ndarray:
    """Randomised modulus switch ``2^q -> 2^b_bits``: ``floor((b + d) / 2^s) mod 2^b_bits``.

    ``s = q_bits - b_bits`` and ``d`` is uniform in ``[0, 2^s)`` per coefficient. ``b`` holds uint64
    coefficients below ``2^q``. ``d`` comes from ``rng`` (fresh per call, never stored). It only
    needs to be independent of ``b`` and the weights, because it bounds the error, not the secrecy.
    ``2^s · result = b - eps`` with ``E[eps | b] = 0`` and ``eps in {c - 2^s, c}``, where
    ``c = b mod 2^s``. The result is returned flat (``b.size`` values).
    """
    s = q_bits - b_bits
    flat = np.ascontiguousarray(b, dtype=np.uint64).reshape(-1)
    if s == 0:
        return flat
    if not 0 < s < 8:
        raise ValueError(f"switch of {s} bits not supported")
    d = rng.integers(0, 1 << s, size=flat.size, dtype=np.uint8)
    sh = np.uint64(s)
    if q_bits < 64:  # b + d < 2^64: one add and one shift
        out = flat + d
        out >>= sh
    else:  # avoid the wrap at 2^64
        out = flat >> sh
        out += ((flat & np.uint64((1 << s) - 1)) + d) >> sh
    if b_bits < 64:
        out &= np.uint64((1 << b_bits) - 1)  # floor((b + d) / 2^s) can reach 2^b_bits: that is 0 mod 2^b_bits
    return out


class RemoteMulmod64:
    """``W8 b mod 2^q`` computed by the compute service for one model pack.

    Speaks wire v1 always and wire v2 when both sides support it. The client advertises v2 on
    its requests when it has v2 formats. It switches to v2 after a reply advertises v2, and it falls
    back to v1 (for :data:`V2_RETRY_S`) when a v2 request is refused as a protocol or version error.

    Args:
        client: A connected :class:`pgw.wire.ComputeClient` (TLS, token and retries come from it).
        fingerprint: Pack fingerprint the service serves.
        out_features: ``{linear: m}``, used to validate response sizes.
        timeout_s: Per-call budget (large GEMMs on CPU services can take seconds).
        drop_bits: v1: low bits of ``y`` the provider rounds away (0 or 16).
        q_bits: Ciphertext modulus bits of ``b`` (``HEParams.q``).
        params: The encryption parameters (ring ``K``, rank, noise). With ``row_norms``, enables v2.
        row_norms: ``{linear: largest weight-row L2 norm}`` (the hint store's ``meta["row_norms"]``).
            A linear without a norm uses v1.
        wire_v2: Offer v2 at all. False keeps v1 even when the server offers v2.
        upload_switch: Allow ``q - 3``-bit uploads (randomised modulus switch). The narrower of the
            two upload widths is chosen per linear by total bytes.
        guard_bits: Reply bits above ``K`` (at least).
        margin_bits: Noise margin below half the scaling factor.
    """

    def __init__(self, client: Any, fingerprint: str, out_features: dict[str, int], *, timeout_s: float = 600.0,
                 drop_bits: int = 16, q_bits: int = 64, params: HEParams | None = None,
                 row_norms: Mapping[str, float] | None = None, wire_v2: bool = True, upload_switch: bool = True,
                 guard_bits: int = DEFAULT_GUARD_BITS, margin_bits: int = DEFAULT_MARGIN_BITS):
        self.client = client
        self.drop_bits = drop_bits
        self.q_bits = q_bits
        self.fingerprint = fingerprint
        self.out_features = dict(out_features)
        self.timeout_s = timeout_s
        self.params = params
        self.margin_bits = margin_bits
        self.bytes_up = 0
        self.bytes_down = 0
        self.calls = 0
        self.calls_v2 = 0
        self._norms: dict[str, float] = {}
        self._v2: dict[str, tuple[V2Format, ...]] = {}
        self._chosen: dict[str, V2Format] = {}
        self._server_v2: bool | None = None  # None: not known yet (first replies tell)
        self._v2_refused_at = -math.inf
        if wire_v2 and params is not None and row_norms is not None:
            self._plan_v2(params, row_norms, upload_switch, guard_bits)
        self.versions: tuple[int, ...] = (1, 2) if self._v2 else (1,)

    # ---- v2 planning (startup) ---------------------------------------------------------------
    def _plan_v2(self, params: HEParams, row_norms: Mapping[str, float], upload_switch: bool,
                 guard_bits: int) -> None:
        if params.q != self.q_bits:
            raise ConfigError(f"HE params use q = 2^{params.q}, the client q = 2^{self.q_bits}")
        refused = []
        for linear in self.out_features:
            norm = row_norms.get(linear)
            fmts = () if norm is None else v2_formats(params, float(norm), upload_switch=upload_switch,
                                                       guard_bits=guard_bits, margin_bits=self.margin_bits)
            if not fmts:
                refused.append(linear)
                continue
            for f in fmts:  # startup assertion (v2_formats only returns passing formats)
                check_v2(params, float(norm), f, margin_bits=self.margin_bits, label=linear)  # type: ignore[arg-type]
            self._norms[linear] = float(norm)  # type: ignore[arg-type]
            self._v2[linear] = fmts
        if self._v2:
            worst = max(f.bound for fs in self._v2.values() for f in fs)
            widths = sorted({(f.b_bits, f.y_bits) for fs in self._v2.values() for f in fs})
            log.info("maskgen wire v2 available for %d/%d linears (formats up/down bits %s; worst noise bound "
                     "2^%.2f < 2^%d)", len(self._v2), len(self.out_features), widths, math.log2(worst),
                     params.delta_bits - 1 - self.margin_bits)
        if refused:
            log.warning("maskgen wire v2 refused for %d linears (noise budget or no norm), v1 for: %s",
                        len(refused), ", ".join(refused[:8]) + (" ..." if len(refused) > 8 else ""))

    def v2_format(self, linear: str, n: int) -> V2Format | None:
        """The v2 format for ``linear`` (``n`` inputs), checked against the noise budget, or ``None`` (v1)."""
        fmts = self._v2.get(linear)
        if not fmts or self.params is None:
            return None
        fmt = self._chosen.get(linear)
        if fmt is None:
            m = self.out_features[linear]
            fmt = min(fmts, key=lambda f: (f.bytes_per_slot(n, m), f.b_bits))
            self._chosen[linear] = fmt
        try:  # per-linear assertion before every v2 request
            check_v2(self.params, self._norms[linear], fmt, margin_bits=self.margin_bits, label=linear)
        except ConfigError as exc:  # pragma: no cover - formats are checked at startup
            log.error("%s; using wire v1 for %s", exc, linear)
            self._v2.pop(linear, None)
            return None
        return fmt

    @property
    def server_v2(self) -> bool | None:
        """Whether the service advertised wire v2 (``None`` until the first reply)."""
        return self._server_v2

    def _use(self, linear: str, n: int) -> V2Format | None:
        if self._server_v2 is not True:
            return None
        return self.v2_format(linear, n)

    def _note_versions(self, versions: tuple[int, ...]) -> None:
        if 2 in versions:
            if self._v2 and self._server_v2 is not True and time.monotonic() - self._v2_refused_at >= V2_RETRY_S:
                log.info("maskgen: the service speaks wire v2; using it for %d linears", len(self._v2))
                self._server_v2 = True
        elif self._server_v2 is not False:
            if self._server_v2:
                log.warning("maskgen: the service no longer advertises wire v2; using v1")
            self._server_v2 = False

    def _refuse_v2(self, exc: PGWError) -> None:
        log.warning("maskgen wire v2 request refused (%s: %s); falling back to v1 for %.0f s", exc.code, exc.message,
                    V2_RETRY_S)
        self._server_v2 = False
        self._v2_refused_at = time.monotonic()

    # ---- requests -----------------------------------------------------------------------------
    def _request(self, linear: str, b: np.ndarray, fmt: V2Format | None) -> tuple[MaskgenRequest, bytes]:
        if b.dtype != np.uint64 or b.ndim != 2:
            raise ValueError("ciphertext must be a 2-D uint64 array")
        cols = int(b.shape[1])
        if fmt is None:
            item = MaskgenItem(linear, cols, pack_b(b, self.q_bits), self.drop_bits, self.q_bits)
            version = 1
        else:
            rng = np.random.default_rng(int.from_bytes(os.urandom(16), "little"))  # fresh dither, never stored
            bq = switch_down(b, self.q_bits, fmt.b_bits, rng)
            item = MaskgenItem(linear, cols, pack_bits(bq, fmt.b_bits), 0, 64, fmt.b_bits, fmt.y_bits)
            version = 2
        req = MaskgenRequest(uuid.uuid4().hex, self.fingerprint, (item,), version, self.versions)
        return req, encode_maskgen_request(req)

    def _decode(self, req: MaskgenRequest, linear: str, cols: int, fmt: V2Format | None) -> Any:
        def decode(data: bytes) -> np.ndarray:
            resp = decode_maskgen_response(data, req, self.out_features)
            self.bytes_down += len(data)
            self._note_versions(resp.versions)
            m = self.out_features[linear]
            if fmt is None:
                return unpack_y(resp.items[0].y, (m, cols), self.drop_bits, self.q_bits)
            v = unpack_bits(resp.items[0].y, m * cols, fmt.y_bits)
            v <<= np.uint64(self.q_bits - fmt.y_bits)  # y mod 2^q up to the rounding and switch errors
            return v.reshape(m, cols)
        return decode

    def _post_sync(self, linear: str, b: np.ndarray, fmt: V2Format | None) -> np.ndarray:
        req, body = self._request(linear, b, fmt)
        self.calls += 1
        self.calls_v2 += fmt is not None
        self.bytes_up += len(body)
        out: np.ndarray = self.client.post_raw_sync(MASKGEN_PATH, body, req.request_id,
                                                     self._decode(req, linear, int(b.shape[1]), fmt),
                                                     timeout_s=self.timeout_s)
        return out

    async def _post(self, linear: str, b: np.ndarray, fmt: V2Format | None) -> np.ndarray:
        req, body = self._request(linear, b, fmt)
        self.calls += 1
        self.calls_v2 += fmt is not None
        self.bytes_up += len(body)
        out: np.ndarray = await self.client.post_raw(MASKGEN_PATH, body, req.request_id,
                                                      self._decode(req, linear, int(b.shape[1]), fmt),
                                                      timeout_s=self.timeout_s)
        return out

    @staticmethod
    def _version_refusal(exc: PGWError) -> bool:
        return exc.code in (ErrorCode.PROTOCOL, ErrorCode.VERSION_MISMATCH)

    def __call__(self, linear: str, b: np.ndarray) -> np.ndarray:
        """Blocking call (from a worker thread). Returns ``y`` (``m x cols`` uint64, ``mod 2^q``)."""
        fmt = self._use(linear, int(b.shape[0]))
        if fmt is not None:
            try:
                return self._post_sync(linear, b, fmt)
            except PGWError as exc:
                if not self._version_refusal(exc):
                    raise
                self._refuse_v2(exc)  # resend the same ciphertext as v1
        return self._post_sync(linear, b, None)

    async def acall(self, linear: str, b: np.ndarray) -> np.ndarray:
        fmt = self._use(linear, int(b.shape[0]))
        if fmt is not None:
            try:
                return await self._post(linear, b, fmt)
            except PGWError as exc:
                if not self._version_refusal(exc):
                    raise
                self._refuse_v2(exc)
        return await self._post(linear, b, None)
