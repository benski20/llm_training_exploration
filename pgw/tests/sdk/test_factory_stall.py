"""The mask factory against a provider that dies while it generates masks (Kaggle run 2, attempt A, 7B on a T4: the
provider died during the first fill and the first request waited out its 3-hour budget).

A real ``pgw service`` on loopback behind fault-injection proxies (:mod:`tests.sdk.chaos`), the default encrypted
scheme (pks-v3: the factory needs the provider for every segment), a short ``factory_stall_s``:

* the provider dies while the factory makes the first fill: the waiting first request fails within the stall
  bound with ``APIConnectionError`` naming the provider and the factory's error; after the provider restarts the
  same client serves (the factory resumes by itself);
* the same in steady state (masks were served before; a request needs more than the stock);
* a slow but healthy first fill (every provider call takes long but completes) is waited for, well beyond
  ``factory_stall_s``.

Every answer that succeeds is bit-identical to the plain-scheme answer.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from tests.sdk.chaos import ChaosProxy
from tests.sdk.conftest import MODEL, Provider, free_port, restart_provider, start_provider, stop_provider

pytestmark = pytest.mark.timeout(900)

PROMPT = [{"role": "system", "content": "You are terse."}, {"role": "user", "content": "Hello there, who are you?"}]
GREEDY = {"temperature": 0, "max_tokens": 24, "logprobs": True, "extra_body": {"ignore_eos": True}}
STALL_S = 8.0


class Rig:
    def __init__(self, prov: Provider, http: ChaosProxy, lane: ChaosProxy):
        self.prov, self.http, self.lane = prov, http, lane

    def provider_down(self) -> None:
        """The provider process dies (SIGKILL) and nothing answers on its address."""
        self.http.mode = self.lane.mode = "refuse"
        self.prov.proc.kill()
        self.prov.proc.wait(timeout=60)
        self.http.drop()
        self.lane.drop()

    def provider_up(self) -> None:
        restart_provider(self.prov)
        self.http.mode = self.lane.mode = "pass"


@pytest.fixture(scope="module")
def rig(sdk_pack_dir: Path, pki: dict[str, Any], tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    port, lane = free_port(), free_port()
    http, lp = ChaosProxy(port), ChaosProxy(lane)
    prov = start_provider(tmp_path_factory.mktemp("stall-provider"), sdk_pack_dir, pki, port=port, lane_port=lane,
                          public_port=http.port, public_lane_port=lp.port)
    r = Rig(prov, http, lp)
    try:
        yield r
    finally:
        http.mode = lp.mode = "pass"
        http.delay_s = 0.0
        stop_provider(prov)
        http.close()
        lp.close()


def _client(rig: Rig, key: str, root: Path, **kw: Any) -> Any:
    from pgw import OpenAI
    from pgw.sdk import LocalPackProvisioner

    # masked_head off: the tiny model's 260-token head is not a pks-v3 shape (real Qwen2.5 vocabularies are)
    opts: dict[str, Any] = {"cache_dir": root / "cache", "state_dir": root / "state", "mask_scheme": "pks-v3",
                            "masked_head": "off", "mask_workers": 1, "mask_target_slots": 300,
                            "mask_low_water_slots": 150, "mask_segment_slots": 64, "mask_wait_s": 600,
                            "factory_stall_s": STALL_S, "ready_timeout_s": 600}
    opts.update(kw)
    return OpenAI(api_key=key, provisioner=LocalPackProvisioner({MODEL: rig.prov.pack_dir}), **opts)


@pytest.fixture(scope="module")
def want(rig: Rig, tmp_path_factory: pytest.TempPathFactory) -> str:
    """The answer with plain masks (masks never change an answer)."""
    c = _client(rig, rig.prov.new_key("want"), tmp_path_factory.mktemp("want"), mask_scheme="plain")
    try:
        return _text(c)
    finally:
        c.close()


def _text(client: Any, **kw: Any) -> str:
    return str(client.chat.completions.create(model=MODEL, messages=PROMPT, **{**GREEDY, **kw}).choices[0]
               .message.content)


def _wait(pred: Any, timeout_s: float, poll_s: float = 0.1) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(poll_s)
    return bool(pred())


def _factory(client: Any) -> dict[str, Any]:
    return dict(client.status()["models"][MODEL].get("factory") or {})


def _status_text(root: Path) -> str:
    from click.testing import CliRunner

    from pgw.sdk.cli import sdk_group

    r = CliRunner().invoke(sdk_group, ["status", "--cache-dir", str(root / "cache"), "--state-dir",
                                       str(root / "state"), "--no-masks"])
    assert r.exit_code == 0, r.output
    return r.output


def _recover(rig: Rig, c: Any, want: str, **kw: Any) -> float:
    """Restart the provider; the same client serves again (bit-identical). Returns seconds from the provider's
    readiness to the answer."""
    rig.provider_up()
    t0 = time.monotonic()
    deadline = t0 + 240
    while True:  # the first tries may still see the outage (retryable errors), never a wrong answer
        from pgw.sdk import PGWAPIError

        try:
            assert _text(c, **kw) == want
            return time.monotonic() - t0
        except PGWAPIError as exc:
            assert exc.retryable, exc
            if time.monotonic() > deadline:
                raise
            time.sleep(1.0)


def test_provider_dies_during_the_first_fill(rig: Rig, want: str, tmp_path: Path) -> None:
    """Attempt A of run 2: the provider dies while the factory makes the first fill. Before: the factory's
    connection errors were not counted, and the waiting first request waited out ready_timeout_s (hours). Now it
    fails within the stall bound with APIConnectionError naming the provider and the factory's error; after the
    restart the same client serves."""
    from pgw.sdk import APIConnectionError

    c = _client(rig, rig.prov.new_key("first-fill"), tmp_path)
    try:
        c.warmup(MODEL)  # enrolled, connected; the factory starts its first fill now
        rig.provider_down()
        t0 = time.monotonic()
        with pytest.raises(APIConnectionError) as ei:
            _text(c)
        took = time.monotonic() - t0
        msg = ei.value.message
        print(f"\nfirst fill, provider down: the request failed after {took:.1f}s: {msg}")
        assert took < STALL_S + 40, took
        assert ei.value.retryable and ei.value.code == "backend_unavailable"
        assert "provider" in msg and "has not answered" in msg and "retrying" in msg, msg
        f = _factory(c)
        assert f["health"]["stalled"] and f["health"]["phase"] == "waiting_for_provider", f["health"]
        assert f["running"] and not f["halted"]  # still retrying, never halted
        out = _status_text(tmp_path)
        assert "STALLED" in out and "provider not answering" in out, out
        took_up = _recover(rig, c, want)
        f = _factory(c)
        print(f"first fill: answer {took_up:.1f}s after the provider was back (factory starts {f['starts']}, "
              f"nudges {f['nudges']})")
        assert took_up < 120, took_up
        assert not f["health"].get("stalled")
    finally:
        c.close()


def test_provider_dies_in_steady_state(rig: Rig, want: str, tmp_path: Path) -> None:
    """Masks were served before; the provider dies and a request needs more masks than are in stock. It fails
    within the stall bound (not mask_wait_s = 600 s), and after the restart the same request is served."""
    from pgw.sdk import APIConnectionError

    c = _client(rig, rig.prov.new_key("steady"), tmp_path)
    try:
        assert _text(c) == want
        fac = c._engine.models[MODEL].factory
        def avail() -> int:
            return int(c.status()["models"][MODEL]["masks_available"])

        # the stock settles (the factory idle), then is used down to between the low-water mark (150, where the
        # factory would refill on its own) and less than a 280-token request needs (target 300 + one segment of 64
        # is the most a request may need)
        assert _wait(lambda: fac.health().get("phase") == "idle" and not fac.health().get("segment"), 120)
        while avail() > 250:
            _text(c)
        assert _wait(lambda: fac.health().get("phase") == "idle" and not fac.health().get("segment"), 120)
        big = {"max_tokens": 280}
        assert 150 <= avail() <= 250, avail()
        rig.provider_down()
        t0 = time.monotonic()
        with pytest.raises(APIConnectionError) as ei:
            _text(c, **big)
        took = time.monotonic() - t0
        print(f"\nsteady state, provider down: the request failed after {took:.1f}s: {ei.value.message}")
        assert took < STALL_S + 40, took
        assert "provider" in ei.value.message
        took_up = _recover(rig, c, want)
        f = _factory(c)
        print(f"steady state: answer {took_up:.1f}s after the provider was back (factory starts {f['starts']}, "
              f"nudges {f['nudges']})")
        assert took_up < 120
        t1 = time.monotonic()
        _text(c, **big)  # the factory makes what the big request needs
        print(f"steady state: the big request served {time.monotonic() - t1:.1f}s after the first answer")
    finally:
        c.close()


def test_a_slow_healthy_first_fill_is_waited_for(rig: Rig, want: str, tmp_path: Path) -> None:
    """Every provider call of the factory takes long but completes (a slow path, a busy provider): the first fill
    takes several times factory_stall_s, and the first request waits for it instead of failing."""
    c = _client(rig, rig.prov.new_key("slow"), tmp_path, factory_stall_s=3.0)
    try:
        c.warmup(MODEL)
        rig.http.delay_s = 0.1  # per forwarded read: the factory's maskgen calls take seconds each
        t0 = time.monotonic()
        try:
            assert _text(c) == want
        finally:
            rig.http.delay_s = 0.0
        took = time.monotonic() - t0
        f = _factory(c)
        print(f"\nslow first fill: answered after {took:.1f}s (stall bound 3 s), factory starts {f['starts']}")
        assert took > 3 * 3.0, f"the first fill was not slow enough to test anything ({took:.1f}s)"
        assert f["starts"] == 1 and not f["health"].get("stalled")
    finally:
        c.close()


def test_status_shows_first_time_setup(rig: Rig, tmp_path: Path) -> None:
    """While the first fill runs, the client's status and ``pgw sdk status`` say "first-time setup in progress"."""
    c = _client(rig, rig.prov.new_key("setup-status"), tmp_path)
    seen: list[str] = []
    try:
        rig.http.delay_s = 0.1
        c.warmup(MODEL)

        def look() -> bool:
            h = _factory(c).get("health") or {}
            if h.get("setup"):
                seen.append(json.dumps(h["setup"]))
                return "first-time setup in progress" in _status_text(tmp_path)
            return False

        assert _wait(look, 120, poll_s=0.2), seen
    finally:
        rig.http.delay_s = 0.0
        c.close()
