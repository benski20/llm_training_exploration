"""``pgw doctor``'s mask-store read-rate check (``maskstore.read_rate``)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from pgw.cli import doctor
from pgw.cli.doctor import _check_maskstore_read_rate, _measure_read_rate, run_doctor


def test_probe_measures_and_cleans_up(tmp_path: Path) -> None:
    m = _measure_read_rate(tmp_path, 3 << 20, block=1 << 20)
    assert m["bytes"] == 3 << 20 and m["block_bytes"] == 1 << 20
    assert m["seconds"] > 0 and m["mb_per_s"] > 0
    assert list(tmp_path.iterdir()) == []  # the probe file is gone


def test_probe_rounds_up_to_whole_blocks(tmp_path: Path) -> None:
    m = _measure_read_rate(tmp_path, 1, block=4096)
    assert m["bytes"] == 4096
    assert list(tmp_path.iterdir()) == []


def test_probe_file_removed_on_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*a: Any, **k: Any) -> None:
        raise OSError("disk on fire")

    monkeypatch.setattr(doctor.os, "fsync", boom)
    with pytest.raises(OSError):
        _measure_read_rate(tmp_path, 1 << 20)
    assert list(tmp_path.iterdir()) == []


def _fake_rate(monkeypatch: pytest.MonkeyPatch, mb_per_s: float, cache_dropped: bool = True) -> None:
    def fake(directory: Path, nbytes: int, block: int = 1 << 20) -> dict[str, Any]:
        return {"bytes": nbytes, "seconds": nbytes / (mb_per_s * 1e6), "block_bytes": block,
                "cache_dropped": cache_dropped, "mb_per_s": mb_per_s}

    monkeypatch.setattr(doctor, "_measure_read_rate", fake)
    monkeypatch.delenv(doctor.EXPECTED_TPS_ENV, raising=False)


SLOT_7B = 5_562_000  # about 5.56 MB per slot (Qwen2.5-7B at K32A12C27)


def test_thresholds_with_expected_load(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_rate(monkeypatch, 125.0)  # gp3 baseline
    st, detail, data = _check_maskstore_read_rate(tmp_path, SLOT_7B, 45.0, 1 << 20)
    assert st == "fail" and "NVMe" in detail
    assert data["needed_mb_per_s"] == pytest.approx(250.3, abs=0.1)
    assert data["supported_tokens_per_second"] == pytest.approx(22.5, abs=0.1)

    _fake_rate(monkeypatch, 280.0)  # within 20% of the need
    st, detail, _ = _check_maskstore_read_rate(tmp_path, SLOT_7B, 45.0, 1 << 20)
    assert st == "warn" and "within 20%" in detail

    _fake_rate(monkeypatch, 2000.0)  # NVMe
    st, detail, data = _check_maskstore_read_rate(tmp_path, SLOT_7B, 45.0, 1 << 20)
    assert st == "ok", detail
    assert data["supported_tokens_per_second"] == pytest.approx(359.6, abs=0.1)


def test_expected_load_from_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_rate(monkeypatch, 125.0)
    monkeypatch.setenv(doctor.EXPECTED_TPS_ENV, "10")
    st, _, data = _check_maskstore_read_rate(tmp_path, SLOT_7B, None, 1 << 20)
    assert st == "ok" and data["expected_tokens_per_second"] == 10.0  # 10 x 5.56 MB = 56 MB/s < 125


def test_without_expected_load_uses_the_chat_speed_floor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_rate(monkeypatch, 125.0)
    st, detail, _ = _check_maskstore_read_rate(tmp_path, None, None, 1 << 20)
    assert st == "warn" and "250 MB/s" in detail
    _fake_rate(monkeypatch, 900.0)
    st, _, _ = _check_maskstore_read_rate(tmp_path, None, None, 1 << 20)
    assert st == "ok"


def test_expected_load_without_pack_is_reported_not_judged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_rate(monkeypatch, 125.0)
    st, detail, _ = _check_maskstore_read_rate(tmp_path, None, 45.0, 1 << 20)
    assert st == "ok" and "--pack" in detail


def test_cache_not_dropped_is_noted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_rate(monkeypatch, 900.0, cache_dropped=False)
    st, detail, _ = _check_maskstore_read_rate(tmp_path, None, None, 1 << 20)
    assert st == "ok" and "optimistic" in detail


def test_unwritable_store_warns(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def denied(*a: Any, **k: Any) -> dict[str, Any]:
        raise PermissionError("read-only")

    monkeypatch.setattr(doctor, "_measure_read_rate", denied)
    st, detail, _ = _check_maskstore_read_rate(tmp_path, None, None, 1 << 20)
    assert st == "warn" and "PermissionError" in detail


def test_missing_store_root_skips(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def never(*a: Any, **k: Any) -> dict[str, Any]:
        raise AssertionError("no probe without a store root")

    monkeypatch.setattr(doctor, "_measure_read_rate", never)
    st, detail, _ = _check_maskstore_read_rate(tmp_path / "absent", None, None, 1 << 20)
    assert st == "skip" and "store root missing" in detail


def test_run_doctor_runs_it_with_the_store_and_can_skip_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_rate(monkeypatch, 2000.0)
    monkeypatch.setattr(doctor, "_check_maskstore", lambda root, key: ("ok", "stub", {}))
    rep = run_doctor(maskstore_root=tmp_path, include_gpu=False, maskstore_read_bytes=1 << 20)
    c = rep.get("maskstore.read_rate")
    assert c is not None and c.status == "ok", c
    rep = run_doctor(maskstore_root=tmp_path, include_gpu=False, maskstore_read_bytes=0)
    assert rep.get("maskstore.read_rate") is None
    assert run_doctor(include_gpu=False).get("maskstore.read_rate") is None


def test_run_doctor_takes_slot_bytes_from_the_pack(tiny_pack_dir: Path, tmp_path: Path,
                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_rate(monkeypatch, 1.0)
    monkeypatch.setattr(doctor, "_check_maskstore", lambda root, key: ("ok", "stub", {}))
    rep = run_doctor(pack=tiny_pack_dir, maskstore_root=tmp_path, include_gpu=False,
                     expected_tokens_per_second=1e9, maskstore_read_bytes=1 << 20)
    pack = rep.get("pack")
    c = rep.get("maskstore.read_rate")
    assert pack is not None and c is not None
    per_ring = pack.data["mask_bytes_per_slot"]
    assert c.data["bytes_per_slot"] in per_ring.values()
    assert c.status == "fail"  # 1 MB/s cannot feed 1e9 tokens/s
