"""Demand anticipation: a pod takes minutes to start (docs/operations/placement.md, "Cold start"), so the controller
sizes for the demand expected one cold start ahead, not for the demand now.

Per (metro, model): an hour-of-week profile (exponentially weighted mean of the observed peak concurrent streams in
each of the 168 hours) times a trend factor (recent observations over the profile, smoothed, clipped to
[``trend_min``, ``trend_max``]). :meth:`Forecaster.anticipate` is the largest of the latest observation and the
prediction at every hour boundary within ``lead_s``: scale-up starts early; scale-down is left to the controller's
idle timeout.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field

__all__ = ["Forecaster"]

WEEK_H = 168


@dataclass
class _Series:
    profile: dict[int, float] = field(default_factory=dict)
    trend: float = 1.0
    last: float = 0.0
    last_t: float = -1.0


class Forecaster:
    def __init__(self, *, alpha: float = 0.3, trend_alpha: float = 0.2, trend_min: float = 0.5,
                 trend_max: float = 2.0):
        self.alpha, self.trend_alpha = alpha, trend_alpha
        self.trend_min, self.trend_max = trend_min, trend_max
        self.series: dict[tuple[str, str], _Series] = defaultdict(_Series)

    @staticmethod
    def bucket(t: float) -> int:
        return int(t // 3600) % WEEK_H

    def observe(self, t: float, demand: Mapping[tuple[str, str], float]) -> None:
        """One observation (peak concurrent streams per (metro, model)) at time ``t``; absent keys count as 0."""
        for k in set(demand) | set(self.series):
            v = float(demand.get(k, 0.0))
            s = self.series[k]
            b = self.bucket(t)
            prev = s.profile.get(b)
            if prev is not None and prev > 0.5:
                ratio = min(self.trend_max, max(self.trend_min, v / prev))
                s.trend += self.trend_alpha * (ratio - s.trend)
            s.profile[b] = v if prev is None else prev + self.alpha * (v - prev)
            s.last, s.last_t = v, t

    def predict(self, key: tuple[str, str], t: float) -> float:
        s = self.series.get(key)
        if s is None:
            return 0.0
        p = s.profile.get(self.bucket(t))
        return s.last if p is None else p * s.trend

    def anticipate(self, key: tuple[str, str], now: float, lead_s: float) -> float:
        """max(latest observation, predictions over the next ``lead_s``)."""
        s = self.series.get(key)
        if s is None:
            return 0.0
        out = s.last
        t = now
        while t <= now + lead_s:
            out = max(out, self.predict(key, t))
            t += 3600.0
        return max(out, self.predict(key, now + lead_s))
