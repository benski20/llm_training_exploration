"""Readiness and error views of ``pgw chat`` (content-free mapping of ``client.status()`` and the SDK's errors)."""

from __future__ import annotations

from pgw.chat.backend import recommended_model
from pgw.chat.status import error_view, summarize
from pgw.sdk import errors as E

M = "qwen2.5-0.5b-instruct"


def _st(**h: object) -> dict[str, object]:
    return {"mask_scheme": "pks-v3", "models": {M: h}}


def test_phases() -> None:
    assert summarize({"models": {}}, M)["phase"] == "idle"
    assert summarize({"models": {}}, M, warming=True)["phase"] == "starting"
    assert summarize(_st(state="enrolling"), M)["phase"] == "enrolling"
    p = summarize(_st(state="provisioning", progress="artifacts 12/100 MB"), M)
    assert p["phase"] == "provisioning" and p["detail"] == "artifacts 12/100 MB"
    assert summarize(_st(state="connecting"), M)["phase"] == "connecting"
    s = summarize(_st(state="ready", masks_available=0, factory={"health": {
        "phase": "hint_build", "setup": {"stage": "hint", "percent": 45.0, "eta_s": 540}}}), M)
    assert s["phase"] == "setup" and s["percent"] == 45.0 and s["eta_s"] == 540 and "45%" in s["detail"]
    assert summarize(_st(state="ready", masks_available=0, factory={"health": {"phase": "producing"}}), M)[
        "phase"] == "preparing"
    r = summarize(_st(state="ready", masks_available=900, verified_requests=3, requests=3, lane="open"), M)
    assert r["phase"] == "ready" and r["verified_requests"] == 3 and r["masks_available"] == 900


def test_stalled_provider_and_integrity_block() -> None:
    s = summarize(_st(state="ready", factory={"health": {"phase": "maskgen_stalled", "stalled": "no progress 31s"}}),
                  M)
    assert s["phase"] == "stalled" and s["error"]["kind"] == "provider_stalled"
    s = summarize(_st(state="ready", factory={"health": {"phase": "waiting_for_provider",
                                                         "stalled": "not answered for 130s"}}), M)
    assert s["phase"] == "stalled" and s["error"]["kind"] == "connection"
    assert summarize(_st(state="ready", integrity_halt={"halted": True}), M)["phase"] == "blocked"
    assert summarize(_st(state="ready", breaker={"open": True}), M)["phase"] == "blocked"


def test_failed_start_shows_the_typed_error() -> None:
    werr = error_view(E.AuthenticationError("invalid, revoked or expired SDK key"))
    s = summarize(_st(state="failed", error="x"), M, warm_error=werr)
    assert s["phase"] == "failed" and s["error"]["kind"] == "auth" and s["title"] == werr["title"]


def test_error_views() -> None:
    assert error_view(E.ProviderStalledError("x"))["kind"] == "provider_stalled"
    assert error_view(E.APITimeoutError("x"))["kind"] == "timeout"
    assert error_view(E.RateLimitError("x"))["retryable"] is True
    v = error_view(E.IntegrityError("freivalds"))
    assert v["kind"] == "integrity" and v["retryable"] is False and "security event" in v["message"]
    v = error_view(RuntimeError("secret prompt text"))
    assert v["kind"] == "internal" and "secret prompt text" not in v["message"]


def test_recommended_model_by_device() -> None:
    assert recommended_model({"cpus": 2, "memory_gb": 4}) == "qwen2.5-0.5b-instruct"
    assert recommended_model({"cpus": 8, "memory_gb": 16}) == "qwen2.5-3b-instruct"
    assert recommended_model({"cpus": 4, "memory_gb": 8}) == "qwen2.5-1.5b-instruct"
    assert recommended_model({"cpus": 64, "memory_gb": 256}) == "qwen2.5-3b-instruct"  # 7B is opt-in
    assert recommended_model({"cpus": 8, "memory_gb": 16}, free_disk_gb=10) == "qwen2.5-0.5b-instruct"
