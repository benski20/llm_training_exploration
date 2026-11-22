"""Outbound-noise monitor: statistically confirms that what leaves the gateway looks uniform.

A defence-in-depth guard against bugs that would send unmasked or reused-mask activations
(SPEC 3.3, 11.3). Correctly masked payloads ``x_hat = (xq + r) mod 2^K`` with a fresh uniform
``r`` are i.i.d. uniform over ``Z_{2^K}`` and independent of ``xq``. The monitor checks exactly
that, on a random sample of outbound calls:

* top-byte and low-byte histograms (Pearson chi-square, 255 dof, p-value from the regularized
  upper incomplete gamma function) per window,
* top-bit balance over all values and per feature (column), with Hoeffding bounds,
* a runs test on the top bit along each row (the number of bit changes between neighbours of i.i.d.
  fair bits is ``Binomial(n-1, 1/2)``; Hoeffding bound),
* correlation with the plaintext quantized activations it was given: ``S = sum(u_i * xq_i)`` where
  ``u_i`` is ``x_hat`` mapped to a zero-mean value in ``(-1/2, 1/2)``. Given ``xq`` the terms are
  independent, zero-mean and bounded, so ``P(|S| >= t) <= 2 exp(-2 t^2 / sum(xq^2))`` (Hoeffding),
  which is distribution-free in ``xq``. Two mappings are tested (unsigned and centered),
* a ring range check (all values in ``[0, 2^K)``) on **every** call: an unmasked send of signed
  ``xq`` shows up as negative values immediately, not only on sampled calls,
* duplicate-row detection on **every** call (not only sampled ones): a fingerprint of the top bits
  of a few columns of each row goes into a bounded history set. A reused mask gives rows whose top
  bits agree even when ``xq`` differs (``|xq1 - xq2| << 2^K``), so mask reuse is caught on the first
  repeated slot.

Lean path: a caller that has the payload only in its packed wire form calls :meth:`OutboundNoiseMonitor.plan_call`
first (the sampling draw) and then either :meth:`~OutboundNoiseMonitor.guard` (``planned=True``, with the
unpacked values, for the sampled ~5% of calls) or :meth:`~OutboundNoiseMonitor.guard_packed` (the every-call
checks on the packed bytes, with identical duplicate-row fingerprints), so non-sampled calls never build the
int64 ``x_hat`` or ``xq``.

Every statistical test runs at ``alpha / (number of tests)`` so the false-alarm rate is below
``alpha`` (default 1e-9) per window (Bonferroni). The chi-square p-value is the asymptotic one; the
other bounds are exact (non-asymptotic). Windows close after ``window_values`` inspected values.

In fail-closed mode an alarm makes :meth:`OutboundNoiseMonitor.guard` raise
:class:`pgw.core.IntegrityError` before the payload is sent, and keeps raising until an admin calls
:meth:`OutboundNoiseMonitor.reset`. The monitor never stores payloads or activations, only
aggregate counts, and never logs values.
"""

from __future__ import annotations

import logging
import math
import os
import random
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field

from pgw.core.errors import ConfigError, IntegrityError, ShapeError
from pgw.core.ring import RingParams

__all__ = [
    "NoiseMonitorConfig",
    "NoiseReport",
    "OutboundNoiseMonitor",
    "chi2_sf",
    "hoeffding_deviation",
]

log = logging.getLogger(__name__)

#: Statistical tests per window (Bonferroni split of ``alpha``): chi2 top, chi2 low, top-bit
#: balance, runs, correlation (unsigned map), correlation (centered map), and the per-feature
#: top-bit test (itself split over the columns of its width).
_WINDOW_TESTS = 7
_MAX_DUP_WIDTHS = 64
_MIN_WINDOW_VALUES = 4096
_MAX_REASONS = 32
_MAX_FEATURE_WIDTHS = 32
_DUP_GROUPS = 2  # independent column groups per row fingerprint
_M64 = (1 << 64) - 1
_LOG_EVERY_S = 10.0


def chi2_sf(x: float, dof: int) -> float:
    """Survival function ``P(X >= x)`` of a chi-square distribution with ``dof`` degrees of freedom."""
    if x <= 0:
        return 1.0
    a = torch.tensor(dof / 2.0, dtype=torch.float64)
    return float(torch.special.gammaincc(a, torch.tensor(x / 2.0, dtype=torch.float64)))


def hoeffding_deviation(n: int, alpha: float) -> float:
    """Two-sided Hoeffding radius for a mean of ``n`` i.i.d. values in ``[0, 1]`` at level ``alpha``.

    ``P(|mean - mu| >= t) <= 2 exp(-2 n t^2) = alpha``  =>  ``t = sqrt(ln(2/alpha) / (2n))``.
    """
    if n <= 0:
        return math.inf
    return math.sqrt(math.log(2.0 / alpha) / (2.0 * n))


class NoiseMonitorConfig(BaseModel):
    """Validated monitor parameters (see :class:`OutboundNoiseMonitor`)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sample_rate: float = Field(0.05, gt=0.0, le=1.0, description="Fraction of calls inspected statistically")
    window_values: int = Field(1 << 16, ge=_MIN_WINDOW_VALUES, description="Values per statistical window")
    fail_closed: bool = Field(True, description="guard() raises once alarmed")
    alpha: float = Field(1e-9, gt=0.0, lt=0.01, description="False-alarm budget per window")
    always_sample_first: int = Field(16, ge=0, description="Inspect the first N calls unconditionally")
    max_values_per_call: int | None = Field(None, ge=256, description="Cap on values inspected per call")
    duplicate_check: bool = Field(True, description="Duplicate-row (mask reuse) detection on every call")
    duplicate_history_rows: int = Field(1 << 16, ge=1, description="Rows remembered for duplicate detection")
    duplicate_columns: int = Field(4, ge=1, le=64, description="Columns per duplicate fingerprint group")
    feature_window_rows: int = Field(2048, ge=64, description="Rows per per-feature top-bit window")


@dataclass
class NoiseReport:
    """Monitor state for dashboards and ``/admin/status``.

    Attributes:
        samples: Values inspected statistically (cumulative).
        windows: Statistical windows evaluated (cumulative).
        alarm: Whether the monitor currently alarms.
        last_chi2_top: Chi-square statistic of the top byte in the last window.
        last_chi2_low: Chi-square statistic of the low byte in the last window.
        last_corr: Pearson correlation of ``x_hat`` (unsigned map) with ``xq`` in the last window.
        duplicate_rows: Repeated outbound rows seen (cumulative; any is a mask-reuse alarm).
        reasons: Why the monitor alarmed (most recent last).
        calls: ``guard`` calls (cumulative).
        sampled_calls: Calls inspected statistically (cumulative).
        rows_checked: Rows passed through duplicate detection (cumulative).
        last_p_top: p-value of ``last_chi2_top``.
        last_p_low: p-value of ``last_chi2_low``.
        last_top_bit: Fraction of values with the top bit set in the last window (expect 0.5).
        last_runs: Fraction of neighbouring top bits that differ in the last window (expect 0.5).
        alarm_since: Unix time the current alarm was raised.
        fail_closed: Whether alarms block traffic.
        sample_rate: Configured sampling rate.
    """

    samples: int
    windows: int
    alarm: bool
    last_chi2_top: float | None = None
    last_chi2_low: float | None = None
    last_corr: float | None = None
    duplicate_rows: int = 0
    reasons: list[str] = field(default_factory=list)
    calls: int = 0
    sampled_calls: int = 0
    rows_checked: int = 0
    last_p_top: float | None = None
    last_p_low: float | None = None
    last_top_bit: float | None = None
    last_runs: float | None = None
    alarm_since: float | None = None
    fail_closed: bool = True
    sample_rate: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly copy."""
        out = asdict(self)
        out["status"] = "alarm" if self.alarm else ("ok" if self.windows else "warming_up")
        return out


class _Window:
    """Aggregates of one statistical window (counts only; never payload values)."""

    __slots__ = (
        "changes",
        "corr_n",
        "corr_s_c",
        "corr_s_u",
        "corr_su",
        "corr_suu",
        "corr_sx",
        "corr_sxx",
        "low_hist",
        "pairs",
        "top_hist",
        "top_ones",
        "values",
    )

    def __init__(self) -> None:
        self.top_hist = torch.zeros(256, dtype=torch.int64)
        self.low_hist = torch.zeros(256, dtype=torch.int64)
        self.values = 0
        self.top_ones = 0
        self.changes = 0
        self.pairs = 0
        self.corr_n = 0
        self.corr_s_u = 0.0  # sum u_i * xq_i (unsigned map)
        self.corr_s_c = 0.0  # sum c_i * xq_i (centered map)
        self.corr_sx = 0.0
        self.corr_sxx = 0.0
        self.corr_su = 0.0
        self.corr_suu = 0.0


class OutboundNoiseMonitor:
    """See module docstring.

    Args:
        ring: Ring parameters of the payloads.
        sample_rate: Fraction of outbound calls inspected statistically.
        window_values: Values per statistical window (at least 4096).
        fail_closed: If true, :meth:`guard` raises :class:`pgw.core.IntegrityError` once alarmed.
        alpha: False-alarm budget per window (split over the tests).
        always_sample_first: The first N calls are always inspected (fast detection at startup).
        max_values_per_call: At most this many values of one call are inspected (a random block of
            whole rows). Default: ``window_values``.
        duplicate_check: Run duplicate-row detection on every call.
        duplicate_history_rows: How many recent rows duplicate detection remembers.
        duplicate_columns: Columns per row fingerprint group.
        feature_window_rows: Rows per per-feature (per input width) top-bit window.
        metrics: Optional :class:`pgw.observability.PGWMetrics`; ``pgw_gateway_noise_alarm`` is set.
        model: ``model`` label for the alarm gauge.
        seed: Seed for the sampling RNG (tests). Default: OS randomness.
    """

    def __init__(
        self,
        ring: RingParams,
        *,
        sample_rate: float = 0.05,
        window_values: int = 1 << 16,
        fail_closed: bool = True,
        alpha: float = 1e-9,
        always_sample_first: int = 16,
        max_values_per_call: int | None = None,
        duplicate_check: bool = True,
        duplicate_history_rows: int = 1 << 16,
        duplicate_columns: int = 4,
        feature_window_rows: int = 2048,
        metrics: Any | None = None,
        model: str = "",
        seed: int | None = None,
    ):
        if not isinstance(ring, RingParams):
            raise ConfigError(f"ring must be RingParams, got {type(ring).__name__}")
        try:
            cfg = NoiseMonitorConfig(
                sample_rate=sample_rate,
                window_values=window_values,
                fail_closed=fail_closed,
                alpha=alpha,
                always_sample_first=always_sample_first,
                max_values_per_call=max_values_per_call,
                duplicate_check=duplicate_check,
                duplicate_history_rows=duplicate_history_rows,
                duplicate_columns=duplicate_columns,
                feature_window_rows=feature_window_rows,
            )
        except ValueError as exc:
            raise ConfigError(f"invalid noise monitor configuration: {exc}") from exc
        self.ring = ring
        self.config = cfg
        self.fail_closed = cfg.fail_closed
        self._max_values = cfg.max_values_per_call or cfg.window_values
        self._alpha_test = cfg.alpha / _WINDOW_TESTS
        # Sampling decisions only (not key material); seeded from the OS CSPRNG.
        self._rng = random.Random(seed if seed is not None else int.from_bytes(os.urandom(16), "little"))  # noqa: S311
        self._lock = threading.Lock()
        self._listeners: list[Callable[[NoiseReport], None]] = []
        self._metrics = metrics
        self._model = model
        # Duplicate detection: keep the top `h` bits of each sampled column. A reused mask with
        # |xq1 - xq2| < 2^A changes those bits with probability <= 2^A / 2^(K-h) = 1/16 per column.
        self._dup_bits = max(4, min(32, ring.k - ring.a - 4))
        self._dup_cols_per_group = max(cfg.duplicate_columns, math.ceil(80 / self._dup_bits))
        dup_seed = self._rng.getrandbits(64)
        self._dup_mult = np.array(
            [(dup_seed * (2 * i + 1) * 0x9E3779B97F4A7C15 | 1) & _M64 for i in range(2 * 64)], dtype=np.uint64
        )
        self._dup_seen: set[int] = set()
        self._dup_order: deque[int] = deque()
        self._dup_cols_cache: dict[int, np.ndarray] = {}
        #: :meth:`guard_packed`'s fingerprints in one native call (``PGW_NOISE_NATIVE=0``: numpy).
        self._native_keys = _native_dup_keys()
        self._clear_state()
        self._calls = 0
        self._sampled_calls = 0
        self._samples = 0
        self._windows = 0
        self._rows_checked = 0
        self._duplicates = 0

    @classmethod
    def from_config(cls, ring: RingParams, config: NoiseMonitorConfig, **kwargs: Any) -> OutboundNoiseMonitor:
        """Build from a :class:`NoiseMonitorConfig` (``kwargs``: ``metrics``, ``model``, ``seed``)."""
        return cls(ring, **config.model_dump(), **kwargs)

    def _clear_state(self) -> None:
        self._win = _Window()
        self._features: dict[int, tuple[torch.Tensor, list[int]]] = {}
        self._alarm = False
        self._alarm_since: float | None = None
        self._reasons: list[str] = []
        self._last_log = 0.0
        self._last: dict[str, float | None] = dict.fromkeys(
            ("chi2_top", "chi2_low", "p_top", "p_low", "corr", "top_bit", "runs")
        )

    # ---- public API ----------------------------------------------------------------------------
    def add_alarm_listener(self, fn: Callable[[NoiseReport], None]) -> None:
        """Call ``fn(report)`` whenever the monitor raises an alarm (paging, audit log ...).

        Listeners run on the thread that called :meth:`guard`, outside the monitor lock; an
        exception in a listener is logged and swallowed (it never breaks the data path).
        """
        if not callable(fn):
            raise ConfigError("alarm listener must be callable")
        with self._lock:
            self._listeners = [*self._listeners, fn]

    @property
    def alarmed(self) -> bool:
        """Whether the monitor currently alarms."""
        return self._alarm

    def plan_call(self) -> bool:
        """Count one outbound call and decide whether it is inspected statistically.

        The same random draw (and ``always_sample_first`` rule) :meth:`guard` makes for a call, in
        the same order, so splitting a call into :meth:`plan_call` + :meth:`guard` (``planned``) or
        :meth:`guard_packed` samples exactly as one :meth:`guard` would. The caller only needs the
        unpacked ``x_hat`` and ``xq`` when this returns ``True``.

        Raises:
            IntegrityError: The monitor alarms and ``fail_closed`` is set.
        """
        if self.fail_closed and self._alarm:
            self._raise_blocked()
        with self._lock:
            self._calls += 1
            return self._draw_sample_locked()

    def _draw_sample_locked(self) -> bool:
        return self._calls <= self.config.always_sample_first or self._rng.random() < self.config.sample_rate

    def guard(self, x_hat: torch.Tensor, xq: torch.Tensor | None = None, *, planned: bool | None = None) -> None:
        """Inspect (maybe) and raise if alarmed in fail-closed mode. Cheap on the fast path.

        Args:
            x_hat: Outbound masked values ``(T, n)`` (or ``(n,)``), integers in ``[0, 2^K)``.
            xq: The plaintext quantized activations the payload was built from (same shape), used
                only for the correlation test and never retained.
            planned: The result of :meth:`plan_call` for this call (the call was already counted and
                its sampling decided); ``None`` (default) counts and decides here.

        Raises:
            IntegrityError: The monitor alarms (now or earlier) and ``fail_closed`` is set.
            ShapeError: ``x_hat`` is not an integer tensor.
        """
        if self.fail_closed and self._alarm:
            self._raise_blocked()
        x = self._as_2d(x_hat, "x_hat")
        if xq is not None:
            if not isinstance(xq, torch.Tensor):
                raise ShapeError(f"xq must be a torch.Tensor, got {type(xq).__name__}")
            xq_shape = tuple(xq.shape) if xq.dim() != 1 else (1, int(xq.shape[0]))
            if xq_shape != tuple(x.shape):
                raise ShapeError(f"xq shape {tuple(xq.shape)} != x_hat shape {tuple(x_hat.shape)}")
        new_reasons: list[str] = []
        with self._lock:
            if planned is None:
                self._calls += 1
            in_ring = True
            if x.numel():
                # Every call: values outside Z_2^K (e.g. signed xq sent unmasked) are never valid.
                lo, hi = torch.aminmax(x)
                if int(lo) < 0 or int(hi) > self.ring.mask:
                    in_ring = False
                    new_reasons.append(f"outbound values outside the ring Z_2^{self.ring.k}")
            if self.config.duplicate_check and x.numel():
                self._check_duplicates(x, new_reasons)
            sampled = self._draw_sample_locked() if planned is None else bool(planned)
            if sampled and x.numel() and in_ring:
                self._sampled_calls += 1
                xq2 = self._as_2d(xq, "xq") if xq is not None else None
                self._inspect(x, xq2, new_reasons)
            if new_reasons:
                self._set_alarm(new_reasons)
        if new_reasons:
            self._notify()
        if self.fail_closed and self._alarm:
            self._raise_blocked()

    def guard_packed(self, payload: bytes | bytearray | memoryview, rows: int, n: int) -> None:
        """The every-call checks of :meth:`guard` on a packed wire payload, for a call that
        :meth:`plan_call` did not sample (no statistics, so no unpacked ``x_hat`` or ``xq`` needed).

        ``payload`` holds ``rows * n`` little-endian ``ceil(K/8)``-byte ring elements (the
        :func:`pgw.core.packing.pack_ring` layout the service receives). Runs the duplicate-row
        (mask-reuse) fingerprint on the same columns, top bits and keyed multipliers as
        :meth:`guard`, so both produce identical fingerprints for the same values. The ring range
        check only applies when ``K`` is not a multiple of 8: a byte-aligned ``K``-bit encoding cannot
        hold a value outside ``Z_2^K``.

        Raises:
            IntegrityError: The monitor alarms (now or earlier) and ``fail_closed`` is set.
            ShapeError: the payload size does not match ``rows * n`` elements.
        """
        if self.fail_closed and self._alarm:
            self._raise_blocked()
        k = self.ring.k
        nb = (k + 7) // 8
        rows, n = int(rows), int(n)
        if rows < 0 or n < 0 or len(payload) != rows * n * nb:
            raise ShapeError(f"packed payload has {len(payload)} bytes, expected {rows}x{n}x{nb}")
        new_reasons: list[str] = []
        with self._lock:
            if rows and n:
                if k % 8 and self._packed_out_of_ring(payload, rows, n, nb):
                    new_reasons.append(f"outbound values outside the ring Z_2^{k}")
                if self.config.duplicate_check:
                    cols = self._dup_columns(n)
                    if cols.size * self._dup_bits >= 64:
                        fast = self._native_keys
                        if fast is not None:  # same keys as _dup_fold's, in one native call
                            c = int(cols.size)
                            groups = _DUP_GROUPS if c >= 2 * self._dup_cols_per_group else 1
                            per = c // groups
                            # bytes and uint8 arrays are read in place; any other buffer (bytearray,
                            # memoryview) goes as a zero-copy uint8 view (pyo3 would copy it per element)
                            buf = payload if type(payload) is bytes else np.frombuffer(payload, dtype=np.uint8)
                            keys = fast(buf, rows, n, nb, cols, k - self._dup_bits, self._dup_mult, groups, per)
                            self._dup_record(keys, rows, groups, new_reasons)
                        else:
                            vals = _packed_columns(payload, rows, n, nb, cols)
                            self._dup_fold(vals >> np.uint64(k - self._dup_bits), rows, n, new_reasons)
            if new_reasons:
                self._set_alarm(new_reasons)
        if new_reasons:
            self._notify()
        if self.fail_closed and self._alarm:
            self._raise_blocked()

    def _packed_out_of_ring(self, payload: bytes | bytearray | memoryview, rows: int, n: int, nb: int) -> bool:
        raw = np.frombuffer(payload, dtype=np.uint8).reshape(rows * n, nb)
        top = raw[:, nb - 1] >> np.uint8(self.ring.k - 8 * (nb - 1))
        return bool(top.any())

    def flush(self) -> None:
        """Evaluate the current partial window now (if it holds at least 4096 values)."""
        reasons: list[str] = []
        with self._lock:
            if self._win.values >= _MIN_WINDOW_VALUES:
                self._evaluate_window(reasons)
            if reasons:
                self._set_alarm(reasons)
        if reasons:
            self._notify()

    def report(self) -> NoiseReport:
        """Current statistics."""
        with self._lock:
            return NoiseReport(
                samples=self._samples,
                windows=self._windows,
                alarm=self._alarm,
                last_chi2_top=self._last["chi2_top"],
                last_chi2_low=self._last["chi2_low"],
                last_corr=self._last["corr"],
                duplicate_rows=self._duplicates,
                reasons=list(self._reasons),
                calls=self._calls,
                sampled_calls=self._sampled_calls,
                rows_checked=self._rows_checked,
                last_p_top=self._last["p_top"],
                last_p_low=self._last["p_low"],
                last_top_bit=self._last["top_bit"],
                last_runs=self._last["runs"],
                alarm_since=self._alarm_since,
                fail_closed=self.fail_closed,
                sample_rate=self.config.sample_rate,
            )

    def reset(self) -> None:
        """Clear the alarm (admin action, after investigation).

        Also discards the partial statistical windows. Cumulative counters and the duplicate-row
        history are kept, so a mask that was reused before the reset is still caught if it recurs.
        """
        with self._lock:
            was = self._alarm
            self._clear_state()
            self._set_gauge(0.0)
        if was:
            log.warning("outbound-noise monitor alarm reset", extra={"model": self._model})

    # ---- internals -----------------------------------------------------------------------------
    def _as_2d(self, t: torch.Tensor, what: str) -> torch.Tensor:
        if not isinstance(t, torch.Tensor):
            raise ShapeError(f"{what} must be a torch.Tensor, got {type(t).__name__}")
        if t.dtype.is_floating_point or t.dtype.is_complex or t.dtype == torch.bool:
            raise ShapeError(f"{what} must be an integer tensor, got {t.dtype}")
        if t.dim() == 1:
            t = t.unsqueeze(0)
        elif t.dim() != 2:
            raise ShapeError(f"{what} must be 1-D or 2-D, got shape {tuple(t.shape)}")
        if t.device.type != "cpu":
            t = t.cpu()
        return t.to(torch.int64)

    def _raise_blocked(self) -> None:
        with self._lock:
            reasons = list(self._reasons[-4:])
        raise IntegrityError(
            "outbound-noise monitor alarm: masked traffic blocked until an admin reset",
            details={"reasons": reasons},
        )

    def _set_alarm(self, reasons: list[str]) -> None:
        now = time.time()
        first = not self._alarm
        if first:
            self._alarm_since = now
        self._alarm = True
        self._set_gauge(1.0)  # under the lock: ordered with reset(), never left stale
        self._reasons.extend(reasons)
        del self._reasons[:-_MAX_REASONS]
        # Log the transition always; repeated findings while alarmed at most every _LOG_EVERY_S.
        if first or now - self._last_log >= _LOG_EVERY_S:
            self._last_log = now
            log.error(
                "outbound-noise monitor ALARM: outbound payloads do not look uniformly random",
                extra={"model": self._model, "reasons": "; ".join(reasons), "fail_closed": self.fail_closed},
            )

    def _notify(self) -> None:
        listeners = self._listeners
        if not listeners:
            return
        rep = self.report()
        for fn in listeners:
            try:
                fn(rep)
            except Exception:
                log.exception("noise alarm listener failed")

    def _set_gauge(self, v: float) -> None:
        if self._metrics is None:
            return
        try:
            self._metrics.get("pgw_gateway_noise_alarm").labels(model=self._model).set(v)
        except Exception:  # pragma: no cover - metrics never break the data path
            log.debug("could not set noise alarm gauge", exc_info=True)

    # duplicate rows ----------------------------------------------------------------------------
    def _dup_columns(self, n: int) -> np.ndarray:
        cols = self._dup_cols_cache.get(n)
        if cols is None:
            g = self._dup_cols_per_group * _DUP_GROUPS
            if n >= g:
                cols = np.unique(np.linspace(0, n - 1, g).round().astype(np.int64))
                if cols.size < g:  # pragma: no cover - linspace over n >= g is injective
                    cols = np.arange(g, dtype=np.int64)
            else:
                cols = np.arange(n, dtype=np.int64)
            if len(self._dup_cols_cache) >= _MAX_DUP_WIDTHS:
                self._dup_cols_cache.clear()
            self._dup_cols_cache[n] = cols
        return cols

    def _check_duplicates(self, x: torch.Tensor, reasons: list[str]) -> None:
        n = x.shape[1]
        cols = self._dup_columns(n)
        if cols.size * self._dup_bits < 64:
            # Too few bits per row for a negligible false-alarm rate (very narrow rows): skip.
            return
        shift = np.uint64(self.ring.k - self._dup_bits)
        top = x[:, torch.from_numpy(cols)].numpy().astype(np.uint64) >> shift  # (T, c)
        self._dup_fold(top, int(x.shape[0]), n, reasons)

    def _dup_fold(self, top: np.ndarray, t: int, n: int, reasons: list[str]) -> None:
        """Fingerprint rows from the top bits ``top`` (uint64 ``(t, c)``) of their sampled columns."""
        c = top.shape[1]
        groups = _DUP_GROUPS if c >= 2 * self._dup_cols_per_group else 1
        per = c // groups
        top = top[:, : groups * per]
        with np.errstate(over="ignore"):
            # Position-dependent multiply + bijective mix per value, XOR-fold per group, final mix
            # salted with the row width and group index: a 64-bit fingerprint of >= 80 key bits.
            v = _mix64(top * self._dup_mult[: groups * per] + np.uint64(n))
            folded = np.bitwise_xor.reduce(v.reshape(top.shape[0], groups, per), axis=2)
            salts = np.array([(n * 0x9E3779B1 + (gi + 1) * 0x85EBCA77) & _M64 for gi in range(groups)], dtype=np.uint64)
            keys = _mix64(folded ^ salts)
        self._dup_record(keys, t, groups, reasons)

    def _dup_record(self, keys: np.ndarray, t: int, groups: int, reasons: list[str]) -> None:
        """Count rows whose fingerprint (any group's key, uint64 ``(t, groups)``) was seen before."""
        keys_per_group = keys.T.tolist()
        seen, order = self._dup_seen, self._dup_order
        limit = self.config.duplicate_history_rows * groups
        dups = 0
        for row_keys in zip(*keys_per_group, strict=True):
            if any(k in seen for k in row_keys):
                dups += 1
                continue
            for k in row_keys:
                seen.add(k)
                order.append(k)
        while len(order) > limit:
            seen.discard(order.popleft())
        self._rows_checked += t
        if dups:
            self._duplicates += dups
            reasons.append(
                f"duplicate outbound rows detected ({dups} in a call of {t} rows): possible mask reuse"
            )

    # statistics --------------------------------------------------------------------------------
    def _inspect(self, x: torch.Tensor, xq: torch.Tensor | None, reasons: list[str]) -> None:
        t, n = x.shape
        rows = max(1, min(t, self._max_values // max(n, 1)))
        if rows < t:
            start = self._rng.randrange(0, t - rows + 1)
            x = x[start : start + rows]
            xq = xq[start : start + rows] if xq is not None else None
        k = self.ring.k  # values are known to be in the ring (checked by guard on every call)
        w = self._win
        top = x >> (k - 8)
        w.top_hist += torch.bincount(top.reshape(-1), minlength=256)
        w.low_hist += torch.bincount((x & 0xFF).reshape(-1), minlength=256)
        bit = top >> 7  # top bit, (rows, n) in {0, 1}
        w.top_ones += int(bit.sum())
        if n > 1:
            w.changes += int((bit[:, 1:] != bit[:, :-1]).sum())
            w.pairs += bit.shape[0] * (n - 1)
        w.values += x.numel()
        self._samples += x.numel()
        if xq is not None:
            modulus = float(self.ring.modulus)
            xf = x.to(torch.float64)
            u = (xf + 0.5) / modulus - 0.5  # uniform x_hat -> exactly zero mean, in (-1/2, 1/2)
            c = torch.where(x >= (1 << (k - 1)), xf - modulus, xf)
            c = (c + 0.5) / modulus  # centered map, also zero mean
            q = xq.to(torch.float64)
            w.corr_n += x.numel()
            w.corr_s_u += float((u * q).sum())
            w.corr_s_c += float((c * q).sum())
            w.corr_sx += float(q.sum())
            w.corr_sxx += float((q * q).sum())
            w.corr_su += float(u.sum())
            w.corr_suu += float((u * u).sum())
        self._update_features(bit, n, reasons)
        if w.values >= self.config.window_values:
            self._evaluate_window(reasons)

    def _update_features(self, bit: torch.Tensor, n: int, reasons: list[str]) -> None:
        if n < 2:
            return
        ent = self._features.get(n)
        if ent is None:
            if len(self._features) >= _MAX_FEATURE_WIDTHS:
                return
            ent = (torch.zeros(n, dtype=torch.int64), [0])
            self._features[n] = ent
        ones, cnt = ent
        ones += bit.sum(dim=0)
        cnt[0] += bit.shape[0]
        r = cnt[0]
        if r < self.config.feature_window_rows:
            return
        frac = ones.to(torch.float64) / r
        dev = float((frac - 0.5).abs().max())
        thr = hoeffding_deviation(r, self._alpha_test / n)
        if dev > thr:
            col = int((frac - 0.5).abs().argmax())
            reasons.append(
                f"per-feature top-bit imbalance (width {n}, column {col}: "
                f"{float(frac[col]):.3f} over {r} rows, bound {0.5 + thr:.3f})"
            )
        ones.zero_()
        cnt[0] = 0

    def _evaluate_window(self, reasons: list[str]) -> None:
        w = self._win
        a = self._alpha_test
        n = w.values
        exp = n / 256.0
        chi_top = float(((w.top_hist.to(torch.float64) - exp) ** 2).sum() / exp)
        chi_low = float(((w.low_hist.to(torch.float64) - exp) ** 2).sum() / exp)
        p_top, p_low = chi2_sf(chi_top, 255), chi2_sf(chi_low, 255)
        top_frac = w.top_ones / n
        runs_frac = w.changes / w.pairs if w.pairs else None
        last: dict[str, float | None] = {
            "chi2_top": chi_top,
            "chi2_low": chi_low,
            "p_top": p_top,
            "p_low": p_low,
            "top_bit": top_frac,
            "runs": runs_frac,
            "corr": None,
        }
        if p_top < a:
            reasons.append(f"top-byte histogram not uniform (chi2={chi_top:.1f}, p={p_top:.2e}, n={n})")
        if p_low < a:
            reasons.append(f"low-byte histogram not uniform (chi2={chi_low:.1f}, p={p_low:.2e}, n={n})")
        thr = hoeffding_deviation(n, a)
        if abs(top_frac - 0.5) > thr:
            reasons.append(f"top-bit imbalance ({top_frac:.4f} over {n} values, bound 0.5±{thr:.4f})")
        if runs_frac is not None:
            thr_r = hoeffding_deviation(w.pairs, a)
            if abs(runs_frac - 0.5) > thr_r:
                reasons.append(f"runs test failed (change rate {runs_frac:.4f}, bound 0.5±{thr_r:.4f})")
        if w.corr_n and w.corr_sxx > 0:
            # Hoeffding given xq: P(|S| >= t) <= 2 exp(-2 t^2 / sum xq^2).
            t = math.sqrt(w.corr_sxx * math.log(2.0 / a) / 2.0)
            m = w.corr_n
            var_x = w.corr_sxx / m - (w.corr_sx / m) ** 2
            var_u = w.corr_suu / m - (w.corr_su / m) ** 2
            if var_x > 0 and var_u > 0:
                cov = w.corr_s_u / m - (w.corr_su / m) * (w.corr_sx / m)
                last["corr"] = cov / math.sqrt(var_x * var_u)
            if abs(w.corr_s_u) > t:
                reasons.append(
                    f"outbound values correlate with plaintext activations "
                    f"(unsigned map, |S|/bound={abs(w.corr_s_u) / t:.2f})"
                )
            if abs(w.corr_s_c) > t:
                reasons.append(
                    f"outbound values correlate with plaintext activations "
                    f"(centered map, |S|/bound={abs(w.corr_s_c) / t:.2f})"
                )
        self._last = last
        self._windows += 1
        log.debug(
            "noise window evaluated", extra={"values": n, "chi2_top": round(chi_top, 1), "chi2_low": round(chi_low, 1)}
        )
        self._win = _Window()


_S30, _S27, _S31 = np.uint64(30), np.uint64(27), np.uint64(31)
_C1, _C2 = np.uint64(0xBF58476D1CE4E5B9), np.uint64(0x94D049BB133111EB)


def _mix64(h: np.ndarray) -> np.ndarray:
    """splitmix64 finalizer (bijective 64-bit mixing), vectorized over uint64 (caller sets errstate)."""
    h = (h ^ (h >> _S30)) * _C1
    h = (h ^ (h >> _S27)) * _C2
    out: np.ndarray = h ^ (h >> _S31)
    return out


def _native_dup_keys() -> Any:
    """``pgw_fast.noise_dup_keys`` (bit-identical to ``_packed_columns`` + ``_dup_fold``'s keys), or None."""
    if os.environ.get("PGW_NOISE_NATIVE", "1").strip().lower() in ("0", "false", "no", "off"):
        return None
    from pgw.core import native

    return getattr(native.fast, "noise_dup_keys", None) if native.fast is not None else None


def _packed_columns(
    payload: bytes | bytearray | memoryview, rows: int, n: int, nb: int, cols: np.ndarray
) -> np.ndarray:
    """Columns ``cols`` of ``rows x n`` packed little-endian ``nb``-byte elements, as uint64 ``(rows, len(cols))``."""
    raw = np.frombuffer(payload, dtype=np.uint8)
    if nb in (1, 2, 4, 8):
        words = raw.view({1: "<u1", 2: "<u2", 4: "<u4", 8: "<u8"}[nb]).reshape(rows, n)
        return words[:, cols].astype(np.uint64)
    picked = raw.reshape(rows, n, nb)[:, cols, :]
    full = np.zeros((rows, cols.size, 8), dtype=np.uint8)
    full[:, :, :nb] = picked
    return full.view("<u8").reshape(rows, cols.size)
