"""The mask factory supervisor's watch without a provider (``pgw.sdk.factory.FactoryProcess.health``): first-time
setup progress, provider reachability, stalls and the restart once the provider is back; and how requests waiting
for masks use it (``Engine._open``). End to end with a real provider: ``tests/sdk/test_factory_stall.py``."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from pgw.core.errors import MaskExhaustedError
from pgw.sdk import errors as E
from pgw.sdk.keys import SdkKey


def _plan(tmp_path: Path, **kw: Any) -> Any:
    from pgw.sdk.factory import FactoryPlan

    base: dict[str, Any] = dict(
        model="m", pack_path=tmp_path / "pack", ring="K32A12C27", state_dir=tmp_path / "f",
        store_root=tmp_path / "masks", kek_file=tmp_path / "kek", counter_file=tmp_path / "ctr", scheme="pks-v3",
        check_field="p31", masked_head=False, workers=1, threads=1, nice=10, target_slots=1000,
        low_water_slots=500, segment_slots=256, first_segment_slots=64, service_url="https://127.0.0.1:9",
        hint_dir=tmp_path / "hint", hint_work=(("l.a", 100), ("l.b", 300)))
    base.update(kw)
    return FactoryPlan(**base)


def _proc(tmp_path: Path, probe: Any, **kw: Any) -> Any:
    from pgw.sdk.factory import FactoryProcess

    fp = FactoryProcess(_plan(tmp_path), poll_s=0.05, stall_s=kw.pop("stall_s", 0.5), probe_s=0.05)
    fp._probe_once = probe  # type: ignore[method-assign]
    assert fp._lock.try_acquire()
    fp._spawn_wall = time.time() - 1.0
    return fp


def _look(fp: Any) -> tuple[dict[str, Any], bool]:
    """One observation with the probe finished (the probe runs on its own thread)."""
    nudge = fp._observe(running=True)
    t = fp._probe_thread
    if t is not None:
        t.join(5)
    return fp.health(), nudge


def _state(fp: Any, **st: Any) -> None:
    sd = fp.plan.state_dir
    sd.mkdir(parents=True, exist_ok=True)
    (sd / "factory-state.json").write_text(json.dumps({"totals": {}, "throughput": {}, **st}))


def test_hint_build_progress_is_shown_and_waited_for(tmp_path: Path) -> None:
    """A pks-v3 hint build under way (no factory state yet, a long time without a segment): first-time setup with
    the share of the work done and an ETA; never a stall while the provider answers."""
    from pgw.sdk.factory import describe_setup

    fp = _proc(tmp_path, lambda: None, stall_s=0.3)
    b = tmp_path / "hint" / f"0123456789abcdef-pks-v3-n4096.building-{os.getpid()}-abcd1234"
    b.mkdir(parents=True)
    (b / "l.a.npy").write_bytes(b"x")
    (b / "l.b.npy.tmp").write_bytes(b"x")
    fp._hint_t0 = (b.name, time.monotonic() - 10.0, 0.0)  # seen 10 s ago at 0 %
    for _ in range(8):
        h, _ = _look(fp)
        time.sleep(0.1)
    assert h["phase"] == "hint_build" and not h["stalled"], h
    assert h["setup"]["percent"] == 25.0 and 25 <= h["setup"]["eta_s"] <= 40, h["setup"]
    line = describe_setup(h)
    assert line is not None and line.startswith("first-time setup in progress") and "25% done" in line, line
    assert "min left" in line or "s left" in line
    # supervisor.json: what pgw sdk status and other processes read
    from pgw.sdk.factory import read_factory_state

    assert read_factory_state(fp.plan.state_dir)["watch"]["setup"]["percent"] == 25.0
    # a build directory of a dead process is not a build under way
    b.rename(b.with_name("0123456789abcdef-pks-v3-n4096.building-999999999-dead"))
    h, _ = _look(fp)
    assert (h["setup"] or {}).get("stage") != "hint"


def test_a_dead_provider_stalls_within_the_bound_then_the_factory_restarts_when_it_is_back(tmp_path: Path) -> None:
    """The factory's runs fail with connection errors and the provider's /readyz does not answer: a stall after
    stall_s (not before), reported as the provider's; once the provider answers again, the child that waits out its
    back-off is restarted (a nudge), and the stall is gone."""
    answer: dict[str, str | None] = {"err": "connection failed (ConnectError)"}
    fp = _proc(tmp_path, lambda: answer["err"], stall_s=0.5)
    _state(fp, last_error="ServiceUnavailableError: compute service transport error (ConnectError)")
    h, nudge = _look(fp)
    assert not h["stalled"] and h["phase"] == "retrying" and not nudge
    t0 = time.monotonic()
    while not h["stalled"] and time.monotonic() - t0 < 5:
        h, nudge = _look(fp)
        time.sleep(0.05)
    assert 0.5 <= time.monotonic() - t0 < 2.0
    assert "has not answered" in h["stalled"] and "ConnectError" in h["stalled"], h
    assert h["phase"] == "waiting_for_provider" and h["provider"]["reachable"] is False
    answer["err"] = None  # the provider is back
    time.sleep(1.1)
    h = fp.recheck()  # what a request does before failing on a stall: one more probe, now
    assert not h["stalled"] and h["provider"]["reachable"] is not False, h
    t1 = time.monotonic()
    while not nudge and time.monotonic() - t1 < 5:
        h, nudge = _look(fp)
        time.sleep(0.05)
    assert nudge and h["provider"]["reachable"] is True and not h["stalled"]


def test_an_error_from_before_this_child_is_not_counted(tmp_path: Path) -> None:
    """The state file still holds an earlier child's (or process's) run error: not a failure of this child."""
    fp = _proc(tmp_path, lambda: None, stall_s=0.2)
    _state(fp, last_error="ServiceUnavailableError: compute service transport error (ConnectError)")
    old = time.time() - 60
    os.utime(fp.plan.state_dir / "factory-state.json", (old, old))
    for _ in range(6):
        h, _ = _look(fp)
        time.sleep(0.1)
    assert not h["stalled"] and h["last_error"] is None, h


def test_a_retry_running_against_a_reachable_provider_is_not_a_stall(tmp_path: Path) -> None:
    """After an error the factory retries: while that attempt runs and the provider answers, no stall (a long first
    segment after a blip); when the attempts keep failing with nothing running, a stall naming the error."""
    fp = _proc(tmp_path, lambda: None, stall_s=0.3)
    err = "ServiceUnavailableError: compute request failed (HTTP 503)"
    _state(fp, last_error=err, in_progress={"slots": 64, "started_at": time.time(), "pid": 1})
    for _ in range(6):
        h, _ = _look(fp)
        time.sleep(0.1)
    assert not h["stalled"] and h["setup"]["stage"] == "first_segment", h
    _state(fp, last_error=err)
    for _ in range(3):
        h, _ = _look(fp)
        time.sleep(0.15)
    assert h["stalled"] and "cannot serve mask generation" in h["stalled"] and "HTTP 503" in h["stalled"], h


def test_segment_progress_overrides_a_failing_probe(tmp_path: Path) -> None:
    """Segments being written mean the factory reaches the provider: a failing health probe alone is no stall."""
    fp = _proc(tmp_path, lambda: "connection failed (ConnectError)", stall_s=0.3)
    _state(fp, totals={"x/K": {"segments": 1, "slots": 64, "seconds": 1.0}}, last_run_at=time.time(),
           in_progress={"slots": 256, "started_at": time.time(), "pid": 1}, throughput={"x/K": 64.0})
    for _ in range(6):
        h, _ = _look(fp)
        time.sleep(0.1)
    assert not h["stalled"] and h["phase"] == "producing" and h["segment"]["slots"] == 256, h
    assert h["segment_s"] == 4.0 and 0 < h["segment"]["eta_s"] <= 4.0


def test_the_probe_reports_without_secrets(tmp_path: Path) -> None:
    """The real probe against a closed port: a short reason, no exception."""
    from pgw.sdk.factory import FactoryProcess

    fp = FactoryProcess(_plan(tmp_path, service_url="http://127.0.0.1:9"), stall_s=5, probe_s=2)
    err = fp._probe_once()
    assert err is not None and err.startswith(("connection failed", "no answer")), err


def test_factory_stall_s_option() -> None:
    from pgw.sdk.options import SDKOptions

    assert SDKOptions().factory_stall_s == 120.0
    with pytest.raises(ValueError, match="factory_stall_s"):
        SDKOptions.from_env(factory_stall_s=0)


# ---- requests waiting for masks ---------------------------------------------------------------------------------
def _engine(tmp_path: Path, health: dict[str, Any], **opts: Any) -> Any:
    from pgw.sdk.engine import Engine, ModelHandle
    from pgw.sdk.options import SDKOptions

    key = SdkKey.new("live", "https://gpu.example.com", bytes(32 * [5]))
    eng = Engine(key, SDKOptions(cache_dir=tmp_path / "c", state_dir=tmp_path / "s", usage_reports=False, **opts))
    h = ModelHandle("qwen2.5-0.5b-instruct", state="ready")
    h.masks_seen = True
    h.factory = SimpleNamespace(owner=True, failures=0, last_error=None, health=lambda: dict(health),
                                state=lambda: {"halted": False, "health": dict(health)},
                                plan=SimpleNamespace(target_slots=1000, segment_slots=64, demand_file=None))
    return eng, h


class _RT:
    state = "ready"

    def __init__(self, ready_at: float = float("inf")):
        self.ready_at = ready_at

    async def open_sessions(self, prompts: Any, params: Any, **kw: Any) -> Any:
        if time.monotonic() < self.ready_at:
            raise MaskExhaustedError("no masks", details={"entitled": 40, "available": 10})
        return [SimpleNamespace()]


@pytest.mark.parametrize("first_fill", [False, True])
def test_a_stalled_factory_fails_waiting_requests_at_once(tmp_path: Path, first_fill: bool) -> None:
    stalled = {"phase": "waiting_for_provider", "stalled": "the provider (https://x) has not answered for 130s "
               "(connection failed (ConnectError))", "last_error": "ServiceUnavailableError: transport error",
               "provider": {"reachable": False, "error": "connection failed (ConnectError)"},
               "setup": {"stage": "hint", "percent": 40.0, "eta_s": 600}}
    eng, h = _engine(tmp_path, stalled, mask_wait_s=600.0, factory_stall_s=120.0)
    h.masks_seen = not first_fill
    t0 = time.monotonic()
    try:
        with pytest.raises(E.APIConnectionError) as ei:
            eng.call(eng._open(h, _RT(), [0], None, "r", None, time.monotonic() + 3600,
                               time.monotonic() + 7200 if first_fill else None), timeout_s=30)
    finally:
        eng.close()
    assert time.monotonic() - t0 < 5
    m = ei.value.message
    assert "has not answered" in m and "keeps retrying" in m and "first-time setup in progress" in m, m
    assert ei.value.retryable and ei.value.details["provider_error"]


def test_a_request_waits_for_the_segment_being_made(tmp_path: Path) -> None:
    """mask_wait_s is shorter than the segment under way, which is due well within the request's timeout: the
    request waits for it (bounded by timeout_s) instead of failing with mask_exhausted."""
    producing = {"phase": "producing", "stalled": None, "segment": {"slots": 64, "running_s": 1.0, "eta_s": 1.5},
                 "segment_s": 2.5, "progress_at": time.time()}
    eng, h = _engine(tmp_path, producing, mask_wait_s=0.5, timeout_s=30.0)
    t0 = time.monotonic()
    try:
        eng.call(eng._open(h, _RT(ready_at=t0 + 1.5), [0], None, "r", None, t0 + 30), timeout_s=30)
    finally:
        eng.close()
    assert 1.4 < time.monotonic() - t0 < 5


def test_the_wait_extension_is_bounded(tmp_path: Path) -> None:
    """A segment due after the request's timeout_s does not extend the wait: mask_exhausted after mask_wait_s,
    with when the factory expects the masks."""
    slow = {"phase": "producing", "stalled": None, "segment": {"slots": 64, "running_s": 1.0, "eta_s": 100.0},
            "segment_s": 120.0, "progress_at": time.time()}
    eng, h = _engine(tmp_path, slow, mask_wait_s=0.5, timeout_s=5.0)
    t0 = time.monotonic()
    try:
        with pytest.raises(E.NotReadyError) as ei:
            eng.call(eng._open(h, _RT(), [0], None, "r", None, t0 + 5), timeout_s=30)
    finally:
        eng.close()
    assert time.monotonic() - t0 < 3
    assert ei.value.code == "mask_exhausted" and "due in about 100s" in ei.value.message, ei.value.message
