"""Reliability review regressions for the factory: disk budget, PRNG range, chunk sizing."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from pgw.core import RING_K40_A12, ConfigError
from pgw.factory import FactoryScheduler, FactorySettings, MaskGenerator
from pgw.factory import scheduler as sched_mod
from pgw.maskstore.format import DEFAULT_CHUNK_BYTES

from .conftest import FakeMaskStore, FakePack


def _sched(tmp_path: Path, store: Any, pack: FakePack, **kw: Any) -> tuple[FactoryScheduler, MaskGenerator]:
    base: dict[str, Any] = {"pack_path": tmp_path / "pack", "segment_slots": 50, "chunk_slots": 16,
                            "low_water_slots": 100, "target_slots": 300, "state_dir": tmp_path / "state"}
    base.update(kw)
    g = MaskGenerator(pack, RING_K40_A12)
    return FactoryScheduler(FactorySettings(**base), store, {g.key: g}, metrics=False), g


def test_plan_is_capped_by_free_space(tmp_path: Path, fake_store: FakeMaskStore, fake_pack: FakePack,
                                      monkeypatch: pytest.MonkeyPatch) -> None:
    sch, gen = _sched(tmp_path, fake_store, fake_pack, min_free_bytes=1000)
    seg = gen.segment_bytes(50)
    free = {"v": 1000 + 2 * seg + seg // 2}
    monkeypatch.setattr(sched_mod.shutil, "disk_usage", lambda p: SimpleNamespace(free=free["v"]))
    plans = sch.plan()
    assert plans[0].segments == 2 and sch.disk_limited  # 6 wanted, 2 fit
    assert len(sch.run_once()) == 2
    free["v"] = 500  # below min_free_bytes: nothing is produced, nothing raises
    assert sch.plan() == [] and sch.disk_limited
    assert sch.run_once() == []
    assert sch.status()["disk_limited"] is True
    sch.close()


def test_plan_is_capped_by_max_store_bytes(tmp_path: Path, fake_store: FakeMaskStore, fake_pack: FakePack,
                                           monkeypatch: pytest.MonkeyPatch) -> None:
    gen = MaskGenerator(fake_pack, RING_K40_A12)
    seg = gen.segment_bytes(50)
    sch, _ = _sched(tmp_path, fake_store, fake_pack, min_free_bytes=0, max_store_bytes=3 * seg)
    monkeypatch.setattr(sched_mod.shutil, "disk_usage", lambda p: SimpleNamespace(free=1 << 50))
    monkeypatch.setattr(type(sch.sink), "stored_bytes", lambda self: seg)
    assert sch.plan()[0].segments == 2
    sch.close()


def test_segment_slots_validated_against_prng_range(tmp_path: Path, fake_store: FakeMaskStore,
                                                    fake_pack: FakePack, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(MaskGenerator, "max_segment_slots", lambda self: 40)
    with pytest.raises(ConfigError, match="PRNG"):
        _sched(tmp_path, fake_store, fake_pack)
    gen = MaskGenerator(fake_pack, RING_K40_A12)
    with pytest.raises(ConfigError, match="PRNG"):
        gen.generate_into_store(fake_store, 50)
    assert not fake_store.writers  # refused before any GPU work or file


def test_store_chunk_is_byte_sized_and_independent_of_gemm_batch(fake_store: FakeMaskStore,
                                                                 fake_pack: FakePack) -> None:
    gen = MaskGenerator(fake_pack, RING_K40_A12)
    seg_id = gen.generate_into_store(fake_store, 10, chunk_slots=4, store_chunk_bytes=2 * 5 * 64)
    widest = max(s.out_features for s in gen.specs)
    assert fake_store.segments[seg_id].chunk_slots == max(1, (2 * 5 * 64) // (widest * 5))
    assert gen.store_chunk_slots() == max(1, DEFAULT_CHUNK_BYTES // (widest * 5))
    s = FactorySettings(pack_path=Path("."))
    assert s.segment_chunk_slots is None and s.segment_chunk_bytes == DEFAULT_CHUNK_BYTES


# ---- liveness / metrics endpoint -------------------------------------------------------------------
def test_health_server_reports_progress_and_failures(tmp_path: Path, fake_store: FakeMaskStore,
                                                     fake_pack: FakePack) -> None:
    import json
    import urllib.error
    import urllib.request

    from pgw.factory.health import FactoryHealthServer
    from pgw.observability.metrics import PGWMetrics

    sch, _ = _sched(tmp_path, fake_store, fake_pack, health_max_stale_s=60, health_max_consecutive_errors=2,
                    poll_interval_s=1)
    sch.metrics = type(sch.metrics)(PGWMetrics(registry=__import__("prometheus_client").CollectorRegistry()))
    sch.run_once()
    with FactoryHealthServer(sch, "127.0.0.1", 0) as srv:
        base = f"http://127.0.0.1:{srv.port}"
        with urllib.request.urlopen(base + "/healthz") as r:  # noqa: S310 - local test server
            assert r.status == 200 and json.loads(r.read())["healthy"]
        with urllib.request.urlopen(base + "/metrics") as r:  # noqa: S310
            text = r.read().decode()
            assert "pgw_factory_last_success_timestamp" in text and "pgw_factory_slots_generated_total" in text
        sch._record_error(RuntimeError("gpu hung"))
        sch._record_error(RuntimeError("gpu hung"))
        with pytest.raises(urllib.error.HTTPError) as ei:
            urllib.request.urlopen(base + "/healthz")  # noqa: S310
        assert ei.value.code == 503
        sch.consecutive_errors = 0
        sch.last_progress_at -= 3600
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(base + "/healthz")  # noqa: S310
    sch.close()
