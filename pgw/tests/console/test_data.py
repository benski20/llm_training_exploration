"""Operator data endpoints (overview, usage, audit) and the audit-log reader."""

from __future__ import annotations

import itertools
import json
import time
from pathlib import Path
from typing import Any

import pytest

from pgw.console.usage import AuditReader, format_ts, parse_ts_ms
from pgw.gateway import ApiKeyStore
from tests.console.conftest import client, make_console_app
from tests.gateway.conftest import ADMIN, MODEL


def _rec(ts_ms: int, **kw: Any) -> str:
    rec = {"ts": format_ts(ts_ms), "event": "inference", **kw}
    return json.dumps(rec, sort_keys=True)


def _write_log(path: Path, now_ms: int, key_id: str) -> None:
    lines = [
        # outside every window but 7d
        _rec(now_ms - 2 * 86_400_000, key_id=key_id, model=MODEL, status=200, prompt_tokens=5, completion_tokens=5,
             duration_ms=10.0),
        # within 24h, outside 1h
        _rec(now_ms - 5 * 3_600_000, key_id=key_id, model=MODEL, status=200, prompt_tokens=7, completion_tokens=3,
             duration_ms=20.0, ttft_ms=4.0, masks_consumed=10),
        json.dumps({"ts": format_ts(now_ms - 4 * 3_600_000), "event": "admin.key_created", "key_id": "admin-token",
                    "action": "key_create", "target": key_id}, sort_keys=True),
        "not json at all",
        # within 1h
        _rec(now_ms - 30 * 60_000, key_id=key_id, model=MODEL, status=200, prompt_tokens=10, completion_tokens=20,
             duration_ms=100.0, ttft_ms=10.0, masks_consumed=30, stream=True),
        _rec(now_ms - 20 * 60_000, key_id="console-playground", model=MODEL, status=200, prompt_tokens=1,
             completion_tokens=2, duration_ms=300.0, ttft_ms=30.0, masks_consumed=3),
        _rec(now_ms - 10 * 60_000, key_id=key_id, model="other", status=503, code="unavailable", duration_ms=5.0,
             prompt="SHOULD NEVER BE RETURNED"),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture
def keys() -> ApiKeyStore:
    return ApiKeyStore(None)


async def test_usage_shapes_and_windows(tiny_pack_dir: Path, tmp_path: Path, keys: ApiKeyStore) -> None:
    rec, _ = keys.create("billing")
    now_ms = int(time.time() * 1000)
    _write_log(tmp_path / "audit.jsonl", now_ms, rec.key_id)
    async with client(make_console_app(tiny_pack_dir, tmp_path, keys=keys)) as c:
        r = await c.get("/admin/usage?window=1h", headers=ADMIN)
        assert r.status_code == 200, r.text
        u = r.json()
        assert u["window"] == "1h" and u["bucket_s"] == 60 and u["source"] == "audit_log" and not u["truncated"]
        assert 60 <= len(u["series"]) <= 62
        ts = [b["t"] for b in u["series"]]
        assert ts == sorted(ts) and all(b - a == 60 for a, b in itertools.pairwise(ts))
        keys_ = {"t", "requests", "prompt_tokens", "completion_tokens", "errors", "p50_ms", "p95_ms", "ttft_p50_ms",
                 "masks_consumed"}
        assert all(set(b) == keys_ for b in u["series"])
        assert sum(b["requests"] for b in u["series"]) == 3
        assert sum(b["errors"] for b in u["series"]) == 1
        assert sum(b["masks_consumed"] for b in u["series"]) == 33
        empty = [b for b in u["series"] if b["requests"] == 0]
        assert empty and empty[0]["p50_ms"] is None
        assert u["totals"]["requests"] == 3 and u["totals"]["p50_ms"] == 100.0 and u["totals"]["p95_ms"] == 300.0
        assert u["totals"]["ttft_p50_ms"] == 10.0
        assert set(u["by_model"]) == {MODEL, "other"} and u["by_model"]["other"]["errors"] == 1
        assert u["by_key"][rec.key_id]["name"] == "billing" and u["by_key"][rec.key_id]["requests"] == 2
        assert u["by_key"]["console-playground"]["name"] == "console playground"

        d = (await c.get("/admin/usage", headers=ADMIN)).json()
        assert d["window"] == "24h" and d["bucket_s"] == 900 and d["totals"]["requests"] == 4
        w = (await c.get("/admin/usage?window=7d", headers=ADMIN)).json()
        assert w["bucket_s"] == 3600 and w["totals"]["requests"] == 5
        assert (await c.get("/admin/usage?window=2d", headers=ADMIN)).status_code == 400

        o = (await c.get("/admin/overview", headers=ADMIN)).json()
        assert {"version", "uptime_s", "models", "requests_24h", "tokens_24h", "errors_24h",
                "integrity_failures_total", "freivalds_checked_total", "stream_requests_total",
                "privacy"} <= set(o)
        assert o["requests_24h"] == 4 and o["errors_24h"] == 1 and o["tokens_24h"] == 43
        m = o["models"][0]
        assert {"name", "state", "ring", "ready", "reason", "masks_available", "masks_runway_s"} <= set(m)
        assert m["name"] == MODEL and m["ready"] is True and m["masks_available"] is None  # plain dev mode
        assert o["privacy"]["masked"] is False and set(o["privacy"]) >= {"masked", "ring", "freivalds_rate",
                                                                          "noise_monitor"}


async def test_audit_listing_filters_and_paging(tiny_pack_dir: Path, tmp_path: Path, keys: ApiKeyStore) -> None:
    rec, _ = keys.create("billing")
    now_ms = int(time.time() * 1000)
    _write_log(tmp_path / "audit.jsonl", now_ms, rec.key_id)
    async with client(make_console_app(tiny_pack_dir, tmp_path, keys=keys)) as c:
        a = (await c.get("/admin/audit", headers=ADMIN)).json()
        evs = a["events"]
        assert a["next_before"] is None
        assert evs[0]["event"] == "gateway.start"  # newest first (written by the app at start-up)
        stamps = [parse_ts_ms(e["ts"]) for e in evs]
        assert stamps == sorted(stamps, reverse=True)
        assert len(evs) == 7  # 6 synthetic records + gateway.start; the non-JSON line is skipped
        assert all("prompt" not in e for e in evs)
        assert "SHOULD NEVER" not in json.dumps(a)

        f = (await c.get(f"/admin/audit?key_id={rec.key_id}&status=5xx", headers=ADMIN)).json()["events"]
        assert len(f) == 1 and f[0]["model"] == "other" and f[0]["status"] == 503
        assert len((await c.get(f"/admin/audit?model={MODEL}", headers=ADMIN)).json()["events"]) == 4
        assert len((await c.get("/admin/audit?status=200", headers=ADMIN)).json()["events"]) == 4
        assert len((await c.get("/admin/audit?status=error", headers=ADMIN)).json()["events"]) == 1
        ev = (await c.get("/admin/audit?event=admin.key_created", headers=ADMIN)).json()["events"]
        assert len(ev) == 1 and ev[0]["target"] == rec.key_id

        # paging
        seen: list[dict] = []
        before = None
        for _ in range(10):
            q = "/admin/audit?limit=2" + (f"&before={before}" if before else "")
            page = (await c.get(q, headers=ADMIN)).json()
            seen += page["events"]
            before = page["next_before"]
            if before is None:
                break
        assert seen == evs
        # before also takes Unix seconds
        older = (await c.get(f"/admin/audit?before={(now_ms - 3_600_000) / 1000}", headers=ADMIN)).json()
        assert len(older["events"]) == 3
        assert (await c.get("/admin/audit?limit=0", headers=ADMIN)).status_code == 400
        assert (await c.get("/admin/audit?limit=5000", headers=ADMIN)).status_code == 400
        assert (await c.get("/admin/audit?before=yesterday", headers=ADMIN)).status_code == 400
        assert (await c.get("/admin/audit")).status_code == 401


async def test_no_audit_file(tiny_pack_dir: Path, tmp_path: Path) -> None:
    async with client(make_console_app(tiny_pack_dir, tmp_path, audit_log_file=None)) as c:
        u = (await c.get("/admin/usage?window=1h", headers=ADMIN)).json()
        assert u["source"] is None and u["totals"]["requests"] == 0 and len(u["series"]) >= 60
        assert (await c.get("/admin/audit", headers=ADMIN)).json() == {"events": [], "next_before": None,
                                                                         "truncated": False}


def test_reader_paging_large_file_with_ties(tmp_path: Path) -> None:
    """Binary search + backwards reads over a file far larger than one block; ties never split pages."""
    path = tmp_path / "a.jsonl"
    base = 1_700_000_000_000
    lines = []
    for i in range(6000):
        lines.append(_rec(base + (i // 3) * 7, key_id=f"k{i % 5}", model="m", status=200 if i % 4 else 500,
                          request_id=f"r{i}", duration_ms=float(i)))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert path.stat().st_size > 8 * 64 * 1024
    reader = AuditReader(path)
    got: list[str] = []
    before = None
    pages = 0
    while True:
        page = reader.events(limit=50, before_ms=parse_ts_ms(before) if before else None)
        got += [e["request_id"] for e in page["events"]]
        pages += 1
        before = page["next_before"]
        if before is None:
            break
    assert got == [f"r{i}" for i in reversed(range(6000))]
    assert pages >= 100
    # a filtered listing
    errs = reader.events(limit=1000, status="500")["events"]
    assert [e["request_id"] for e in errs] == [f"r{i}" for i in reversed(range(6000)) if i % 4 == 0][:1000]
    # bounded scans report truncation and still page onwards without skipping
    small = AuditReader(path, max_scan_bytes=100_000)
    got2: list[str] = []
    before_ms = None
    for _ in range(1000):
        page = small.events(limit=10_000 // 10, before_ms=before_ms, key_id="k1")
        got2 += [e["request_id"] for e in page["events"]]
        if page["next_before"] is None:
            break
        before_ms = parse_ts_ms(page["next_before"])
    want = [f"r{i}" for i in reversed(range(6000)) if i % 5 == 1]
    assert list(dict.fromkeys(got2)) == want


def test_reader_usage_cache_and_truncation(tmp_path: Path) -> None:
    path = tmp_path / "a.jsonl"
    now = 1_700_000_000.0
    path.write_text(_rec(int(now * 1000) - 1000, model="m", status=200, duration_ms=1.0) + "\n", encoding="utf-8")
    reader = AuditReader(path, clock=lambda: now)
    assert reader.usage("1h")["totals"]["requests"] == 1
    with path.open("a") as fh:
        fh.write(_rec(int(now * 1000) - 500, model="m", status=200, duration_ms=1.0) + "\n")
    assert reader.usage("1h")["totals"]["requests"] == 2  # file grew: cache invalidated
    tiny = AuditReader(path, max_scan_bytes=50, clock=lambda: now)
    assert tiny.usage("1h")["truncated"] is True


def test_ts_roundtrip() -> None:
    ms = 1_763_553_600_123
    assert parse_ts_ms(format_ts(ms)) == ms
    assert parse_ts_ms("garbage") is None and parse_ts_ms("2026-11-19T12:00:00Z") == 1_795_089_600_000
