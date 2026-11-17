"""API key store, rate limiter and quota tracker."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from pgw.core import AuthError, ConfigError, RateLimitedError
from pgw.gateway.auth import ApiKey, ApiKeyStore, KeyNotFoundError, QuotaTracker, RateLimiter, parse_api_key


class FakeClock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


# ---- key store -----------------------------------------------------------------------------------
def test_create_and_verify_in_memory() -> None:
    store = ApiKeyStore(None)
    rec, key = store.create("billing-app", models=("tiny",))
    assert key.startswith(f"pgw-{rec.key_id}-")
    kid, secret = parse_api_key(key)
    assert kid == rec.key_id and len(secret) == 43
    assert secret not in rec.hashed and len(rec.hashed) == 64
    got = store.verify(key)
    assert got.key_id == rec.key_id and got.models == ("tiny",) and got.scopes == ("inference",)
    assert got.allows_model("tiny") and not got.allows_model("other")
    assert "hashed" not in got.public()


@pytest.mark.parametrize("bad", ["", "pgw-", "sk-123", "pgw-zzzzzzzzzzzz-" + "a" * 43, "pgw-0123456789ab-short"])
def test_malformed_keys_rejected(bad: str) -> None:
    with pytest.raises(AuthError):
        ApiKeyStore(None).verify(bad)


def test_wrong_secret_and_unknown_id_rejected() -> None:
    store = ApiKeyStore(None)
    _rec, key = store.create("a")
    kid, secret = parse_api_key(key)
    flipped = secret[:-1] + ("A" if secret[-1] != "A" else "B")
    with pytest.raises(AuthError):
        store.verify(f"pgw-{kid}-{flipped}")
    with pytest.raises(AuthError):
        store.verify(f"pgw-{'0' * 12}-{secret}")


def test_revoke() -> None:
    store = ApiKeyStore(None)
    rec, key = store.create("a")
    store.revoke(rec.key_id)
    with pytest.raises(AuthError, match="revoked"):
        store.verify(key)
    assert store.list()[0].disabled
    with pytest.raises(KeyNotFoundError):
        store.revoke("0123456789ab")


def test_create_validation() -> None:
    store = ApiKeyStore(None)
    with pytest.raises(ConfigError):
        store.create("")
    with pytest.raises(ConfigError):
        store.create("x", scopes=("root",))
    with pytest.raises(ConfigError):
        store.create("x", requests_per_minute=0)


def test_file_store_persists_hashes_only_with_0600(tmp_path: Path) -> None:
    path = tmp_path / "keys.json"
    store = ApiKeyStore(path)
    rec, key = store.create("a", scopes=("inference", "admin"), daily_token_quota=1000)
    mode = stat.S_IMODE(os.stat(path).st_mode)
    assert mode == 0o600
    text = path.read_text()
    assert parse_api_key(key)[1] not in text
    data = json.loads(text)
    assert data["version"] == 1 and data["keys"][0]["key_id"] == rec.key_id
    again = ApiKeyStore(path)
    got = again.verify(key)
    assert got.has_scope("admin") and got.daily_token_quota == 1000


def test_file_store_reloads_changes_from_other_process(tmp_path: Path) -> None:
    path = tmp_path / "keys.json"
    running = ApiKeyStore(path)  # the gateway
    cli = ApiKeyStore(path)  # 'pgw gateway keys create' elsewhere
    rec, key = cli.create("late")
    assert running.verify(key).key_id == rec.key_id
    cli.revoke(rec.key_id)
    with pytest.raises(AuthError):
        running.verify(key)


def test_corrupt_key_file_is_config_error(tmp_path: Path) -> None:
    path = tmp_path / "keys.json"
    path.write_text('{"version": 99, "keys": []}')
    with pytest.raises(ConfigError):
        ApiKeyStore(path)


# ---- rate limiter ------------------------------------------------------------------------------
def _key(**kw: object) -> ApiKey:
    return ApiKey(key_id=str(kw.pop("key_id", "0123456789ab")), hashed="", name="k", **kw)  # type: ignore[arg-type]


def test_request_rate_limit_and_refill() -> None:
    clock = FakeClock()
    rl = RateLimiter(requests_per_minute=3, tokens_per_minute=10_000, clock=clock)
    k = _key()
    for _ in range(3):
        rl.check_request(k, 1)
    with pytest.raises(RateLimitedError) as ei:
        rl.check_request(k, 1)
    assert ei.value.details["limit"] == "requests"
    assert 0 < ei.value.details["retry_after_s"] <= 20.0
    clock.t += 20.0  # 3/min -> one request every 20 s
    rl.check_request(k, 1)


def test_token_rate_limit_reserve_and_settle() -> None:
    clock = FakeClock()
    rl = RateLimiter(requests_per_minute=1000, tokens_per_minute=600, clock=clock)  # 10 tokens/s
    k = _key()
    rl.check_request(k, 500)
    with pytest.raises(RateLimitedError) as ei:
        rl.check_request(k, 200)
    assert ei.value.details["limit"] == "tokens"
    assert ei.value.details["retry_after_s"] == pytest.approx(10.0, abs=0.01)
    rl.record_tokens(k, 100, reserved=500)  # actual usage was smaller: refund 400
    rl.check_request(k, 200)


def test_oversized_request_admitted_when_bucket_full_then_debt() -> None:
    clock = FakeClock()
    rl = RateLimiter(requests_per_minute=1000, tokens_per_minute=60, clock=clock)
    k = _key()
    rl.check_request(k, 1000)  # larger than the bucket: allowed once from a full bucket
    with pytest.raises(RateLimitedError):
        rl.check_request(k, 1)


def test_per_key_overrides_and_isolation() -> None:
    clock = FakeClock()
    rl = RateLimiter(requests_per_minute=100, tokens_per_minute=10_000, clock=clock)
    slow = _key(key_id="aaaaaaaaaaaa", requests_per_minute=1)
    other = _key(key_id="bbbbbbbbbbbb")
    rl.check_request(slow, 1)
    with pytest.raises(RateLimitedError):
        rl.check_request(slow, 1)
    rl.check_request(other, 1)
    rl.forget(slow.key_id)
    rl.check_request(slow, 1)


# ---- quotas -----------------------------------------------------------------------------------------
def test_daily_quota_reserve_settle_and_rollover(tmp_path: Path) -> None:
    day = {"d": "2026-11-17"}
    q = QuotaTracker(100, tmp_path / "quota.json", today=lambda: day["d"])
    k = _key()
    q.reserve(k, 60)
    with pytest.raises(RateLimitedError) as ei:
        q.reserve(k, 50)
    assert ei.value.details["limit"] == "daily_tokens"
    q.settle(k, 60, 30)
    q.reserve(k, 50)
    q.settle(k, 50, 50)
    assert q.usage([k.key_id]) == {k.key_id: 80}
    # persisted and reloaded for the same day
    q2 = QuotaTracker(100, tmp_path / "quota.json", today=lambda: day["d"])
    assert q2.usage()[k.key_id] == 80
    with pytest.raises(RateLimitedError):
        q2.reserve(k, 30)
    day["d"] = "2026-11-18"
    q2.reserve(k, 90)


def test_per_key_quota_overrides_default() -> None:
    q = QuotaTracker(None)
    unlimited = _key(key_id="aaaaaaaaaaaa")
    q.reserve(unlimited, 10**9)
    small = _key(key_id="bbbbbbbbbbbb", daily_token_quota=5)
    with pytest.raises(RateLimitedError):
        q.reserve(small, 6)
