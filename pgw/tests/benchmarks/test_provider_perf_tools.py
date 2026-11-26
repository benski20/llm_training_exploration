"""Smoke runs of the RunPod provider measurement tools on CPU: the two-sided lane NIC bench over loopback TLS (the
Python executor stands in for the GPU) and the session plan's dry run."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

PGW = Path(__file__).resolve().parents[2]


def _env() -> dict[str, str]:
    return dict(os.environ, PYTHONPATH=str(PGW / "src") + os.pathsep + str(PGW))


def test_lane_nic_bench_serve_and_client(tmp_path: Path) -> None:
    from pgw.core.native import fast

    if fast is None or not hasattr(fast, "LaneService") or not hasattr(fast, "TripChannel"):
        pytest.skip("pgw_fast with the lane is not installed")
    info = tmp_path / "lane.json"
    cmd = [sys.executable, "benchmarks/lane_nic_bench.py", "serve", "--cpu", "--preset", "tiny", "--host",
           "127.0.0.1", "--port", "0", "--stub-port", "0", "--tickets", "8", "--out", str(info), "--seconds", "120"]
    srv = subprocess.Popen(cmd, cwd=PGW, env=_env(), stdout=subprocess.PIPE,  # noqa: S603
                           stderr=subprocess.PIPE, text=True)
    try:
        t0 = time.time()
        while not info.exists():
            assert srv.poll() is None, srv.stderr.read() if srv.stderr else ""
            assert time.time() - t0 < 90, "serve did not start"
            time.sleep(0.2)
        assert oct(info.stat().st_mode & 0o777) == "0o600"  # tickets inside
        out = tmp_path / "client.json"
        cmd = [sys.executable, "benchmarks/lane_nic_bench.py", "client", "--info", str(info), "--conc", "1,2",
               "--iters", "5", "--json", str(out)]
        r = subprocess.run(cmd, cwd=PGW, env=_env(), capture_output=True, text=True, timeout=120)  # noqa: S603
        assert r.returncode == 0, r.stderr[-2000:]
        res = json.loads(out.read_text())
        assert {(x["target"], x["conc"]) for x in res["runs"]} == {("stub", 1), ("stub", 2), ("gpu", 1), ("gpu", 2)}
        assert all(not x["errors"] and x["trips"] == 4 * 5 * x["conc"] for x in res["runs"])
        assert set(res["derived_1conn"]) == {"rtt_us", "transport_us", "device_us", "lane_fixed_us"}
    finally:
        srv.terminate()
        srv.wait(30)


def test_perf_session_dry_run(tmp_path: Path) -> None:
    cmd = [sys.executable, "deploy/runpod/provider_perf_session.py", "--dry-run", "--cpu", "--host", "203.0.113.7",
           "--out", str(tmp_path)]
    r = subprocess.run(cmd, cwd=PGW, env=_env(), capture_output=True, text=True, timeout=120)  # noqa: S603
    assert r.returncode == 0, r.stderr[-2000:]
    log = (tmp_path / "session.log").read_text()
    for needle in ("provider_scale_bench.py", "gpu_kernel_bench.py", "lane_gpu_trip_bench.py",
                   "maskgen_provider_bench.py", "lane_nic_bench.py serve", "client --info"):
        assert needle in log, needle
    assert json.loads((tmp_path / "summary.json").read_text())["phases"]
