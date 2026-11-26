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
    s = ensure_session(key, lay, transport=FakeProvider(pem).transport, client_name="host-1")
    assert load_session(key, lay) is not None and s.ca_file is not None
    s.ca_file.write_text(pem + other)  # an extra trust anchor
    assert load_session(key, lay) is None
    s.ca_file.write_text(other)  # another CA
    assert load_session(key, lay) is None
    s.ca_file.unlink()  # before: the session fell back to the system trust store
    assert load_session(key, lay) is None
    again = ensure_session(key, lay, transport=FakeProvider(pem).transport, client_name="host-1")
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
