"""Settings policy, models file, circuit breaker, admission queue, egress guard and audit log."""

from __future__ import annotations

import asyncio
import json
import os
import stat
from pathlib import Path

import httpx
import pytest

from pgw.core import ConfigError, IntegrityError, ServiceUnavailableError
from pgw.gateway import (
    AdmissionController,
    AuditLog,
    BreakerOpenError,
    CircuitBreaker,
    EgressBlockedError,
    EgressGuardTransport,
    GatewayModelConfig,
    GatewaySettings,
)
from pgw.gateway.audit import AUDIT_FIELDS
from pgw.gateway.config import load_models_file, origin_of

PACK = Path("/nonexistent/pack")


def masked(name: str = "m", url: str = "https://compute.example.net:8443", **kw: object) -> GatewayModelConfig:
    kw.setdefault("fingerprint", "ab" * 32)
    return GatewayModelConfig(name=name, pack_path=PACK, service_url=url, **kw)  # type: ignore[arg-type]


# ---- config / policy ------------------------------------------------------------------------------
def test_masked_requires_service_url() -> None:
    with pytest.raises(ValueError, match="service_url"):
        GatewayModelConfig(name="m", pack_path=PACK)


def test_model_config_validation() -> None:
    m = masked(ring="k48a20", fingerprint="AB" * 32)
    assert m.ring == "K48A20" and m.fingerprint == "ab" * 32
    with pytest.raises(ValueError):
        masked(ring="K12A3")
    with pytest.raises(ValueError):
        masked(fingerprint="xyz")
    with pytest.raises(ValueError):
        masked(url="ftp://x")
    with pytest.raises(ValueError):
        GatewayModelConfig(name="m", pack_path=PACK, mode="plain", bogus=1)  # type: ignore[call-arg]


def test_production_policy_refuses_dev_modes() -> None:
    s = GatewaySettings(models=[GatewayModelConfig(name="p", pack_path=PACK, mode="plain")])
    with pytest.raises(ConfigError, match="development"):
        s.validate_policy()
    GatewaySettings(models=[GatewayModelConfig(name="p", pack_path=PACK, mode="plain")],
                    allow_dev_modes=True).validate_policy()


def test_production_policy_freivalds_and_anonymous_and_tls() -> None:
    with pytest.raises(ConfigError, match="freivalds"):
        GatewaySettings(models=[masked(freivalds_rate=0.5)]).validate_policy()
    with pytest.raises(ConfigError, match="anonymous"):
        GatewaySettings(models=[masked()], allow_anonymous=True).validate_policy()
    with pytest.raises(ConfigError, match="https"):
        GatewaySettings(models=[masked(url="http://compute.example.net")]).validate_policy()
    GatewaySettings(models=[masked(url="http://127.0.0.1:9000")]).validate_policy()  # loopback sidecar ok
    GatewaySettings(models=[masked(url="http://compute.internal")], require_service_tls=False).validate_policy()


def test_security_review_production_defaults() -> None:
    """Fail-closed defaults from the security review (fingerprint pin, full pack hash, breaker, monitor)."""
    s = GatewaySettings()
    assert s.integrity_cooldown_s is None  # admin reset only (SPEC 3.2)
    assert s.pack_verify == "full" and s.min_soundness_bits >= 20
    assert s.noise_monitor_is_required
    assert GatewaySettings(allow_dev_modes=True).noise_monitor_is_required is False
    with pytest.raises(ConfigError, match="fingerprint"):
        GatewaySettings(models=[masked(fingerprint=None)]).validate_policy()
    GatewaySettings(models=[masked(fingerprint=None)], allow_dev_modes=True).validate_policy()
    with pytest.raises(ConfigError, match="pack_verify"):
        GatewaySettings(models=[masked()], pack_verify="fingerprint").validate_policy()
    with pytest.raises(ConfigError, match="noise_monitor_required"):
        GatewaySettings(models=[masked()], noise_monitor_required=False).validate_policy()


def test_egress_origins() -> None:
    s = GatewaySettings(models=[masked(url="https://Compute.Example.net")],
                        egress_allowlist=["https://backup.example.net:9443"])
    assert s.egress_origins() == {("https", "compute.example.net", 443), ("https", "backup.example.net", 9443)}
    with pytest.raises(ConfigError):
        origin_of("compute.example.net")
    with pytest.raises(ValueError):
        GatewaySettings(egress_allowlist=["not a url"])


def test_models_file_json_yaml_relative_paths(tmp_path: Path) -> None:
    (tmp_path / "packs").mkdir()
    entries = [{"name": "a", "pack_path": "packs/a", "service_url": "https://c.example.net"},
               {"name": "b", "pack_path": "/abs/b", "mode": "plain"}]
    jf = tmp_path / "models.json"
    jf.write_text(json.dumps({"models": entries}))
    got = load_models_file(jf)
    assert got[0].pack_path == (tmp_path / "packs" / "a").resolve() and got[1].pack_path == Path("/abs/b")
    yaml = pytest.importorskip("yaml")
    yf = tmp_path / "models.yaml"
    yf.write_text(yaml.safe_dump(entries))
    assert [m.name for m in load_models_file(yf)] == ["a", "b"]
    s = GatewaySettings(models=[masked(name="a")], models_file=jf)
    with pytest.raises(ConfigError, match="duplicate"):
        s.all_models()
    bad = tmp_path / "bad.json"
    bad.write_text('[{"name": "x"}]')
    with pytest.raises(ConfigError, match="invalid model entry"):
        load_models_file(bad)
    with pytest.raises(ConfigError):
        load_models_file(tmp_path / "missing.json")


def test_settings_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PGW_GATEWAY_PORT", "9999")
    monkeypatch.setenv("PGW_GATEWAY_ADMIN_TOKEN", "s3cret-admin-token-0123")
    monkeypatch.setenv("PGW_GATEWAY_MODELS", json.dumps(
        [{"name": "m", "pack_path": "/p", "service_url": "https://c.example.net"}]))
    s = GatewaySettings()
    assert s.port == 9999 and s.models[0].name == "m"
    assert "s3cret" not in repr(s)
    assert s.admin_token is not None and s.admin_token.get_secret_value() == "s3cret-admin-token-0123"


# ---- circuit breaker ------------------------------------------------------------------------------
def test_breaker_trip_cooldown_and_reset() -> None:
    now = {"t": 0.0}
    changes: list[bool] = []
    b = CircuitBreaker("m", 60.0, clock=lambda: now["t"], on_change=changes.append)
    b.check()
    b.trip(IntegrityError("bad"))
    assert b.is_open
    with pytest.raises(BreakerOpenError) as ei:
        b.check()
    assert ei.value.details["retry_after_s"] == 60.0
    assert ei.value.http_status == 503
    now["t"] = 59.0
    assert b.is_open
    now["t"] = 60.0
    assert not b.is_open
    b.trip()
    assert b.reset() and not b.is_open and not b.reset()
    assert changes == [True, False, True, False]
    assert b.status()["trips"] == 2


def test_breaker_admin_reset_only() -> None:
    now = {"t": 0.0}
    b = CircuitBreaker("m", None, clock=lambda: now["t"])
    b.trip()
    now["t"] = 1e9
    assert b.is_open
    with pytest.raises(BreakerOpenError) as ei:
        b.check()
    assert "retry_after_s" not in ei.value.details


# ---- admission ---------------------------------------------------------------------------------------
async def test_admission_queue_fifo_full_and_timeout() -> None:
    adm = AdmissionController(max_active=1, max_queue=1, queue_timeout_s=0.2)
    await adm.acquire()
    waiter = asyncio.create_task(adm.acquire())
    await asyncio.sleep(0.01)
    assert adm.waiting == 1
    with pytest.raises(ServiceUnavailableError, match="queue is full"):
        await adm.acquire()
    await adm.release()
    waited = await waiter
    assert waited > 0 and adm.active == 1
    with pytest.raises(ServiceUnavailableError, match="timed out"):
        await adm.acquire()
    assert adm.rejected == 2 and adm.waiting == 0
    await adm.release()
    async with adm.slot():
        assert adm.active == 1
    assert adm.active == 0


async def test_admission_drain() -> None:
    adm = AdmissionController(max_active=1, max_queue=4, queue_timeout_s=5)
    await adm.acquire()
    waiter = asyncio.create_task(adm.acquire())
    await asyncio.sleep(0.01)
    drained = asyncio.create_task(adm.drain(1.0))
    with pytest.raises(ServiceUnavailableError, match="draining"):
        await waiter
    await adm.release()
    assert await drained
    with pytest.raises(ServiceUnavailableError, match="draining"):
        await adm.acquire()
    adm.reopen()
    await adm.acquire()


# ---- egress ------------------------------------------------------------------------------------------
async def test_egress_guard_blocks_other_hosts() -> None:
    seen: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(str(req.url))
        return httpx.Response(200, text="ok")

    guard = EgressGuardTransport(httpx.MockTransport(handler), {("https", "compute.example.net", 443)})
    async with httpx.AsyncClient(transport=guard) as c:
        assert (await c.get("https://compute.example.net/pgw/v1/models")).status_code == 200
        for url in ("https://evil.example.net/", "http://compute.example.net/", "https://compute.example.net:444/",
                    "https://telemetry.huggingface.co/"):
            with pytest.raises(EgressBlockedError):
                await c.get(url)
    assert seen == ["https://compute.example.net/pgw/v1/models"] and guard.blocked == 4


def test_disable_library_telemetry(monkeypatch: pytest.MonkeyPatch) -> None:
    from pgw.gateway.egress import disable_library_telemetry

    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    disable_library_telemetry()
    assert os.environ["HF_HUB_OFFLINE"] == "1" and os.environ["HF_HUB_DISABLE_TELEMETRY"] == "1"


# ---- audit ------------------------------------------------------------------------------------------
def test_audit_allowlist_and_file_mode(tmp_path: Path) -> None:
    path = tmp_path / "audit" / "audit.jsonl"
    audit = AuditLog(path)
    rec = audit.event("inference", request_id="r1", key_id="k", prompt="SECRET PROMPT", completion="SECRET",
                      api_key="pgw-x", prompt_tokens=3, status=200)
    audit.close()
    assert "prompt" not in rec and "completion" not in rec and "api_key" not in rec
    assert audit.dropped_fields == 3
    text = path.read_text()
    assert "SECRET" not in text and "pgw-x" not in text
    line = json.loads(text.strip())
    assert line["event"] == "inference" and line["prompt_tokens"] == 3 and line["request_id"] == "r1"
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    assert "prompt" not in AUDIT_FIELDS and "text" not in AUDIT_FIELDS


def test_gateway_refuses_store_without_external_counter_at_startup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without a monotonic counter the gateway fails at start-up instead of 500ing every request."""
    from pgw.gateway.runtime import InferenceRuntime
    from pgw.maskstore.admin import init_store

    root, kek = tmp_path / "masks", tmp_path / "kek"
    init_store(root, kek)
    monkeypatch.setenv("PGW_MASKSTORE_KEY_FILE", str(kek))
    monkeypatch.setenv("PGW_MASKSTORE_COUNTER", "none")
    monkeypatch.setenv("PGW_MASKSTORE_ALLOW_NO_EXTERNAL_COUNTER", "false")
    settings = GatewaySettings(models=[masked(url="http://127.0.0.1:9000")], maskstore_root=root)
    with pytest.raises(ConfigError, match="monotonic counter"):
        InferenceRuntime(settings)
