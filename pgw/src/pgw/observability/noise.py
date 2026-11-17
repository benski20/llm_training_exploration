"""Outbound-noise monitor: statistically confirms that what leaves the gateway looks uniform.

A defence-in-depth guard against bugs that would send unmasked or reused-mask activations. It
samples outbound ``x_hat`` tensors and tests:

* top-byte and low-byte histograms (chi-square, 255 dof) per window,
* per-feature mean of the top bit (should be 0.5),
* correlation with the plaintext quantized activations it was given (should be ~0),
* duplicate-row detection (identical x_hat rows => mask reuse) via a rolling hash set.

Thresholds are set for a false-alarm rate below 1e-9 per window. In fail-closed mode an alarm
makes :meth:`OutboundNoiseMonitor.guard` raise before the payload is sent.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from pgw.core.ring import RingParams

__all__ = ["NoiseReport", "OutboundNoiseMonitor"]


@dataclass
class NoiseReport:
    """Monitor state for dashboards and ``/admin/status``."""

    samples: int
    windows: int
    alarm: bool
    last_chi2_top: float | None = None
    last_chi2_low: float | None = None
    last_corr: float | None = None
    duplicate_rows: int = 0
    reasons: list[str] = field(default_factory=list)


class OutboundNoiseMonitor:
    """See module docstring.

    Args:
        ring: Ring parameters of the payloads.
        sample_rate: Fraction of outbound calls inspected.
        window_values: Values per statistical window.
        fail_closed: If true, :meth:`guard` raises :class:`pgw.core.IntegrityError` once alarmed.
    """

    def __init__(self, ring: RingParams, *, sample_rate: float = 0.05, window_values: int = 1 << 16,
                 fail_closed: bool = True):
        raise NotImplementedError("observability team: OutboundNoiseMonitor")

    def guard(self, x_hat: torch.Tensor, xq: torch.Tensor | None = None) -> None:
        """Inspect (maybe) and raise if alarmed in fail-closed mode. Cheap on the fast path."""
        raise NotImplementedError

    def report(self) -> NoiseReport:
        """Current statistics."""
        raise NotImplementedError

    def reset(self) -> None:
        """Clear the alarm (admin action, after investigation)."""
        raise NotImplementedError
