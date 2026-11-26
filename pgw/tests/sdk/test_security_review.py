"""Regression tests of the 2026-11 security review (``research/reviews/security-review-2026-11.md``), client side:
the CA pin (S2, S7) and the content-free request report (S3, S4, S9)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest

from pgw.core.errors import ConfigError
from pgw.enroll.service import pinned_ca
from pgw.sdk import errors as E
from pgw.sdk.cache import CacheLayout
from pgw.sdk.keys import SdkKey
from pgw.sdk.reports import Reporter, build_record
from pgw.sdk.session import ensure_session, load_session
from tests.sdk.test_session import URL, FakeProvider


def _pems(pki: dict[str, Any]) -> tuple[str, str, bytes]:
    return Path(pki["ca.pem"]).read_text(), Path(pki["client-ca.pem"]).read_text(), pki["ca_sha256"]


# ---- S2: a CA reply with an appended certificate -----------------------------------------------------------------
def test_pinned_ca_accepts_exactly_one_certificate(pki: dict[str, Any]) -> None:
    pem, other, fp = _pems(pki)
    got_fp, got_pem = pinned_ca(pem)
    assert got_fp == fp and got_pem.count("BEGIN CERTIFICATE") == 1
    for bad in (pem + other, other + pem, "", "not a pem"):
        with pytest.raises(ConfigError):
            pinned_ca(bad)


def test_appended_attacker_ca_is_refused_before_the_key_is_sent(tmp_path: Path, pki: dict[str, Any]) -> None:
    """PoC of S2: an on-path attacker serves the genuine (pinned) CA first and its own CA after it. Before the fix
    the first certificate matched the pin, the whole text became the trust store, and the SDK key went to whoever
    holds the second CA's key; the CA file kept trusting it for every later connection."""
    pem, attacker, fp = _pems(pki)
    key = SdkKey.new("live", URL, fp)
    prov = FakeProvider(pem + attacker)
    with pytest.raises(E.IntegrityError):
        ensure_session(key, CacheLayout(tmp_path / "c", tmp_path / "s"), transport=prov.transport,
                       client_name="host-1")
    assert prov.enrolls == [] and prov.auth == []  # nothing secret was sent


# ---- S7: the cached CA file is checked against the pin on every load ------------------------------------------
def test_cached_session_with_a_replaced_or_missing_ca_is_not_used(tmp_path: Path, pki: dict[str, Any]) -> None:
    pem, other, fp = _pems(pki)
    key = SdkKey.new("live", URL, fp)
    lay = CacheLayout(tmp_path / "c", tmp_path / "s")
    s = ensure_session(key, lay, transport=FakeProvider(pem, cert=False).transport, client_name="host-1")
    assert load_session(key, lay) is not None and s.ca_file is not None
    s.ca_file.write_text(pem + other)  # an extra trust anchor
    assert load_session(key, lay) is None
    s.ca_file.write_text(other)  # another CA
    assert load_session(key, lay) is None
    s.ca_file.unlink()  # before: the session fell back to the system trust store
    assert load_session(key, lay) is None
    again = ensure_session(key, lay, transport=FakeProvider(pem, cert=False).transport, client_name="host-1")
    assert again.ca_file is not None and pinned_ca(again.ca_file.read_text())[0] == fp


# ---- S3 / S4 / S9: the report carries only what the provider observes -----------------------------------------
def _rec(**kw: Any) -> dict[str, Any]:
    base: dict[str, Any] = dict(ts=1.0, model="m", status="ok", error_code=None, prompt_tokens=120, cached_tokens=0,
                                completion_tokens=7, latency_ms=812.5, ttft_ms=40.0, stream=True, speculation=False)
    base.update(kw)
    return build_record(**base)


def test_report_counts_only_prompt_rows_sent_to_the_provider() -> None:
    """S3: with conversation prefix reuse, 100 of 120 prompt tokens come from the cache and never reach the provider;
    reporting 120 told it the exact length of the reused prefix (how much of the conversation matched)."""
    assert _rec()["prompt_tokens"] == 120
    assert _rec(cached_tokens=100)["prompt_tokens"] == 20
    assert _rec(cached_tokens=500)["prompt_tokens"] == 0


def test_report_has_no_latency_with_strict_speculation() -> None:
    """S4: the answer returns before the strict-mode dummy passes; its latency counts the real passes (the drafts'
    acceptance, a function of the text) that the fixed pass count hides."""
    assert _rec()["latency_ms"] == 812.5
    r = _rec(speculation=True)
    assert r["latency_ms"] is None and r["ttft_ms"] == 40.0 and r["completion_tokens"] == 7


def test_report_error_codes_are_public_codes_only() -> None:
    assert _rec(status="error", error_code="integrity_check_failed")["error_code"] == "integrity_check_failed"
    assert _rec(status="error", error_code="deadline_exceeded")["error_code"] == "deadline_exceeded"
    r = _rec(status="error", error_code="ValueError: secret prompt text")
    assert r["error_code"] == "internal" and r["checks_ok"] is False


def test_reporter_keeps_records_on_429() -> None:
    calls: list[int] = []

    class _S:
        def http_client(self, **kw: Any) -> httpx.Client:
            def h(req: httpx.Request) -> httpx.Response:
                calls.append(1)
                return httpx.Response(429 if len(calls) == 1 else 200, json={})

            return httpx.Client(transport=httpx.MockTransport(h), base_url="http://x")

    rep = Reporter(lambda: _S())
    rep._q.append(_rec())
    rep.flush()
    assert len(rep._q) == 1 and rep.sent == 0  # kept for the next round
    rep.flush()
    assert len(rep._q) == 0 and rep.sent == 1


# ---- S12: requests that never reached the provider are not reported --------------------------------------------
def test_requests_that_never_reached_the_provider_are_not_reported(tmp_path: Path) -> None:
    """A request refused locally (context length, no masks, model blocked) opened no session: the provider never
    saw it, so the report must not tell it that a (long) request happened. It is counted in the local status."""
    from pgw import OpenAI
    from pgw.core.errors import ErrorCode, PGWError

    key = SdkKey.new("live", "https://gpu.example.com", bytes(32 * [5])).encode()
    c = OpenAI(api_key=key, cache_dir=tmp_path / "c", state_dir=tmp_path / "s", timeout=60)
    eng = c._engine
    sent: list[dict[str, Any]] = []
    eng.reporter.add = sent.append  # nothing to send it to

    async def local_refusal(req: Any, *, conversation: Any = None, meta: Any = None, timeout_s: Any = None) -> Any:
        meta["model"] = "qwen2.5-0.5b-instruct"
        meta["prompt_tokens"] = 40000
        raise PGWError("no masks in stock", code=ErrorCode.MASK_EXHAUSTED)
        yield None, None  # pragma: no cover

    async def reached(req: Any, *, conversation: Any = None, meta: Any = None, timeout_s: Any = None) -> Any:
        meta["model"] = "qwen2.5-0.5b-instruct"
        meta["opened"] = True
        meta["prompt_tokens"] = 12
        raise PGWError("provider went away", code=ErrorCode.UNAVAILABLE)
        yield None, None  # pragma: no cover

    msgs = [{"role": "user", "content": "x"}]
    try:
        eng._chat_events = local_refusal
        with pytest.raises(E.PGWAPIError):
            c.chat.completions.create(model="qwen2.5-0.5b-instruct", messages=msgs)
        assert sent == []
        assert eng.status()["local_errors"] == {"mask_exhausted": 1}
        eng._chat_events = reached
        with pytest.raises(E.PGWAPIError):
            c.chat.completions.create(model="qwen2.5-0.5b-instruct", messages=msgs)
        (rec,) = sent
        assert rec["status"] == "error" and rec["error_code"] == "unavailable" and rec["prompt_tokens"] == 12
    finally:
        c.close()


# ---- S11: the integrity halt survives a restart -------------------------------------------------------------
def _halt_store(tmp_path: Path) -> Any:
    from pgw.sdk.halt import HaltStore

    return HaltStore(tmp_path / "state" / "sessions" / "0123456789abcdef", "0123456789abcdef")


def test_integrity_halt_is_persisted_private_and_mac_protected(tmp_path: Path) -> None:
    import json
    import stat

    from pgw.sdk.halt import DOMAIN, HaltStore

    m = "qwen2.5-0.5b-instruct"
    st = _halt_store(tmp_path)
    assert not st.load(m).halted and not st.path(m).exists()  # no record: never failed
    st.record_failure(m, "freivalds")
    st.record_failure(m, "range")
    again = HaltStore(st.state_dir, st.key_id).load(m)  # a new process
    assert again.halted and again.failures == 2 and not again.tampered
    for p in (st.path(m), st.key_file):
        assert stat.S_IMODE(p.stat().st_mode) == 0o600
    assert DOMAIN == b"pgw-sdk-integrity-halt-v1\x00"

    good = st.path(m).read_text()
    data = json.loads(good)
    for edit in ({"halted": False, "failures": 0}, {"mac": "00" * 32}, {"mac": None}, {"extra": 1}):
        st.path(m).write_text(json.dumps({**data, **edit}))
        r = HaltStore(st.state_dir, st.key_id).load(m)
        assert r.halted and r.tampered and "fail closed" in (r.reason or ""), edit
    st.path(m).write_text(json.dumps({k: v for k, v in data.items() if k != "mac"}))
    assert HaltStore(st.state_dir, st.key_id).load(m).tampered == "no MAC"
    st.path(m).write_text("{not json")
    assert HaltStore(st.state_dir, st.key_id).load(m).halted
    st.path(m).write_text(good)
    assert HaltStore(st.state_dir, st.key_id).load(m).failures == 2
    # the record of another key, or of another model, does not verify here
    other = HaltStore(tmp_path / "state" / "sessions" / "fedcba9876543210", "fedcba9876543210")
    other.dir.mkdir(parents=True)
    other.path(m).write_text(good)
    (other.state_dir / "integrity.key").write_bytes(st.key_file.read_bytes())
    assert other.load(m).tampered
    st.path("qwen2.5-7b-instruct").write_text(good)
    assert st.load("qwen2.5-7b-instruct").tampered
    st.path("qwen2.5-7b-instruct").unlink()
    st.key_file.unlink()  # the MAC key is gone: unverifiable, halted
    r = HaltStore(st.state_dir, st.key_id).load(m)
    assert r.halted and "missing" in (r.tampered or "")


def test_engine_trips_the_breaker_from_the_persisted_halt(tmp_path: Path) -> None:
    """A restart must not re-open a halted model: the new runtime's breaker is tripped before any request, a failure
    seen through the breaker's listener is persisted, and only an explicit reset clears both."""
    from types import SimpleNamespace

    from pgw.gateway.safety import BreakerOpenError, CircuitBreaker
    from pgw.sdk.engine import Engine, ModelHandle
    from pgw.sdk.options import SDKOptions

    m = "qwen2.5-0.5b-instruct"
    key = SdkKey.new("live", "https://gpu.example.com", bytes(32 * [6]))

    def engine() -> tuple[Engine, ModelHandle, Any]:
        eng = Engine(key, SDKOptions(cache_dir=tmp_path / "c", state_dir=tmp_path / "s", usage_reports=False))
        h = eng.models[m] = ModelHandle(m)
        rt = SimpleNamespace(breaker=CircuitBreaker(m, None), safety_listeners=[])
        rt.reset_safety = lambda **kw: rt.breaker.reset()
        eng._bind_halt(h, rt)
        h.runtime = rt
        return eng, h, rt

    eng, h, rt = engine()
    assert not rt.breaker.is_open
    for fn in rt.safety_listeners:  # what ModelRuntime._on_integrity_failure does
        fn("safety.integrity_failure", m, "freivalds")
    assert h.halt is not None and h.halt.halted and h.halt.failures == 1
    eng.close()

    eng2, h2, rt2 = engine()  # "restart"
    assert rt2.breaker.is_open and h2.integrity_failures == 1
    with pytest.raises(BreakerOpenError):
        rt2.breaker.check()
    st = eng2.status()["models"][m]["integrity_halt"]
    assert st["halted"] and st["failures"] == 1
    eng2.call(eng2.reset_breaker(m), timeout_s=30)
    assert not rt2.breaker.is_open and not eng2.halts.load(m).halted
    eng2.close()
    eng3, _, rt3 = engine()
    assert not rt3.breaker.is_open
    eng3.close()


def test_sdk_clear_halt_cli(tmp_path: Path) -> None:
    from click.testing import CliRunner

    from pgw.cli.main import cli

    m = "qwen2.5-0.5b-instruct"
    st = _halt_store(tmp_path)
    st.record_failure(m)
    st.path(m).write_text(st.path(m).read_text().replace('"halted": true', '"halted": false'))
    assert st.load(m).tampered  # an edited record stays halted ...
    args = ["sdk", "clear-halt", "--model", m, "--state-dir", str(tmp_path / "state")]
    runner = CliRunner()
    r = runner.invoke(cli, args)
    assert r.exit_code != 0 and "WARNING" in r.output and "--yes" in r.output
    assert st.load(m).halted
    r = runner.invoke(cli, [*args, "--yes", "--note", "provider redeployed"])
    assert r.exit_code == 0, r.output
    assert "cleared qwen2.5-0.5b-instruct for key 0123456789abcdef" in r.output and "unverifiable" in r.output
    rec = st.load(m)
    assert not rec.halted and not rec.tampered and rec.failures == 0 and rec.events[-1]["event"] == "cleared"
    r = runner.invoke(cli, [*args, "--yes"])
    assert r.exit_code == 0 and "no integrity halt" in r.output
    r = runner.invoke(cli, ["sdk", "clear-halt", "--model", "gpt-4", "--yes", "--state-dir", str(tmp_path)])
    assert r.exit_code != 0


# ---- S20: strict-mode pass timing --------------------------------------------------------------------------------
def test_strict_passes_start_no_earlier_than_the_gap_floor(tmp_path: Path) -> None:
    """With strict speculation every forward pass starts at least ``spec_pass_gap_ms`` after the previous one ended,
    so the client's work between passes (more of it at the end of the answer) is not visible below the floor."""
    import asyncio
    import itertools
    import time
    from types import SimpleNamespace

    from pgw.sdk.engine import Engine
    from pgw.sdk.options import SDKOptions

    key = SdkKey.new("live", "https://gpu.example.com", bytes(32 * [7]))
    spans: list[tuple[float, float]] = []

    async def forward(seqs: Any) -> Any:
        t0 = time.monotonic()
        await asyncio.sleep(0.002)
        spans.append((t0, time.monotonic()))
        return seqs

    def rt() -> Any:
        return SimpleNamespace(generator=SimpleNamespace(model=SimpleNamespace(aforward_batch=forward)))

    async def three(model: Any) -> None:
        for _ in range(3):
            await model.aforward_batch([1])

    for spec, gap, paced in (("lookup", 30.0, True), ("lookup", 0.0, False), ("off", 30.0, False)):
        eng = Engine(key, SDKOptions(cache_dir=tmp_path / "c", state_dir=tmp_path / "s", usage_reports=False,
                                     speculation=spec, spec_pass_gap_ms=gap))
        r = rt()
        eng._pace_strict_passes(r)
        eng._pace_strict_passes(r)  # idempotent
        spans.clear()
        asyncio.run(three(r.generator.model))
        gaps = [b[0] - a[1] for a, b in itertools.pairwise(spans)]
        assert (min(gaps) >= 0.029) == paced, (spec, gap, gaps)
        assert len(spans) == 3
    with pytest.raises(ValueError):
        SDKOptions(speculation="lookup", spec_pass_gap_ms=-1).validate()
