"""Opt-in 32B-width gateway knobs leave every bit unchanged (timing only, or exact splits).

The knobs are read once per process (``OnceLock`` in ``pgw_fast``), so each combination re-runs existing bit-for-bit
suites in a fresh interpreter with the knobs set:

* standard decode driver: ``PGW_LANE_WAIT_PREFETCH=1`` (prefetch of the trip's ``W r`` / check vectors while the lane
  waits), ``PGW_POOL_ACTIVE_SPIN_US`` (pool workers polling while a driver call is active), ``PGW_POOL_WORKERS``
  above the default, ``PGW_POOL_CPUS`` (pinning) and ``PGW_TRIP_SPLIT`` with a low threshold, so the split kernels
  run inside the driver: ``tests/engine/test_decode_native.py`` (driver == plain, byte-identical payloads, tamper);
* native fused2 driver: ``PGW_FUSED2_SPLIT`` with ``PGW_FUSED2_SPLIT_MIN_ELEMS`` low enough for the tiny model's
  pair stages (unmask of ``Z``, stage-2 check split mod p, normed gate/up + SiLU in element ranges) plus the
  prefetch: ``tests/engine/test_fused2_native.py`` (native == Python fused2 == plain bit for bit, tamper caught).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

fast = pytest.importorskip("pgw_fast")
ROOT = Path(__file__).resolve().parents[2]


def _run(env_extra: dict[str, str], target: str, select: str | None = None) -> None:
    env = {**os.environ, **env_extra}
    cmd = [sys.executable, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider", target]
    if select:
        cmd += ["-k", select]
    r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=1800, check=False)  # noqa: S603
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]


def test_decode_driver_with_prefetch_warm_pool_and_split() -> None:
    cpus = max(1, (os.cpu_count() or 2) - 1)
    _run({"PGW_LANE_WAIT_PREFETCH": "1", "PGW_POOL_ACTIVE_SPIN_US": "2000", "PGW_POOL_WORKERS": "3",
          "PGW_POOL_CPUS": f"0-{cpus}", "PGW_TRIP_SPLIT": "3", "PGW_TRIP_SPLIT_MIN_ELEMS": "16"},
         "tests/engine/test_decode_native.py",
         "driver_equals_plain or native_lane or p31_driver or tamper_on_the_native_lane")


@pytest.mark.skipif(not hasattr(fast, "Fused2Driver"), reason="pgw_fast without Fused2Driver")
def test_fused2_driver_with_split_and_prefetch() -> None:
    _run({"PGW_FUSED2_SPLIT": "3", "PGW_FUSED2_SPLIT_MIN_ELEMS": "64", "PGW_LANE_WAIT_PREFETCH": "1",
          "PGW_POOL_ACTIVE_SPIN_US": "2000"},
         "tests/engine/test_fused2_native.py", "bit_for_bit or tamper or one_slot")
