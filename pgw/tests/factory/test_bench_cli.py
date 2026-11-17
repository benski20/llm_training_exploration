"""Benchmark helpers and the ``pgw factory`` CLI (store and pack I/O faked where needed)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from pgw.core import ConfigError
from pgw.factory import cli as fcli
from pgw.factory.bench import MODEL_PRESETS, benchmark_pack, benchmark_preset, format_report

from .conftest import FakeMaskStore, FakePack


def test_presets_shapes() -> None:
    l8 = MODEL_PRESETS["llama3-8b"]
    specs = l8.layer_specs(3)
    assert [s.id.key for s in specs] == ["3.qkv", "3.o", "3.gate_up", "3.down"]
    assert [(s.out_features, s.in_features) for s in specs] == [(6144, 4096), (4096, 4096), (28672, 4096), (4096, 14336)]
    # SPEC 7.2: 6.9 MB/slot for Llama-3-8B at K40, 1.5 MB for Qwen2.5-0.5B
    per_slot_8b = sum(s.out_features for s in specs) * 32 * 5
    assert per_slot_8b / 1e6 == pytest.approx(6.9, abs=0.05)
    q = MODEL_PRESETS["qwen2.5-0.5b"]
    assert sum(s.out_features for s in q.layer_specs()) * q.layers * 5 / 1e6 == pytest.approx(1.5, abs=0.05)


def test_benchmark_preset_small() -> None:
    r = benchmark_preset("qwen2.5-0.5b", slots=16, chunk_slots=16, price_per_hour=1.0, warmup=False)
    assert r.layers_measured == 1 and r.layers_total == 24
    assert r.slots_per_second > 0 and r.tokens_per_hour == pytest.approx(3600 * r.slots_per_second)
    assert r.usd_per_million_tokens == pytest.approx(1.0 / (r.tokens_per_hour / 1e6))
    assert r.bytes_per_token == 1_520_640  # sum(m) * 24 layers * 5 bytes
    rep = format_report(r)
    assert "tokens/hour" in rep and "per million tokens" in rep and "extrapolated" in rep
    assert json.loads(json.dumps(r.to_dict()))["model"] == "qwen2.5-0.5b"


def test_benchmark_errors() -> None:
    with pytest.raises(ConfigError):
        benchmark_preset("gpt-17")
    with pytest.raises(ConfigError):
        benchmark_preset("qwen2.5-0.5b", layers=0)


def test_benchmark_pack_with_store(fake_store: FakeMaskStore) -> None:
    r = benchmark_pack(FakePack(), slots=8, chunk_slots=4, store=fake_store, price_per_hour=0.5)
    assert r.mode == "pack+store" and r.layers_total == 1 and len(fake_store.segments) == 1
    r2 = benchmark_pack(FakePack(), slots=8, chunk_slots=4)
    assert r2.mode == "pack" and r2.price_per_hour > 0


# ---- CLI -------------------------------------------------------------------------------------------
def test_cli_benchmark_json() -> None:
    res = CliRunner().invoke(fcli.factory_group, ["benchmark", "--model", "qwen2.5-0.5b", "--slots", "8",
                                                  "--chunk-slots", "8", "--price-per-hour", "2", "--json"])
    assert res.exit_code == 0, res.output
    data = json.loads(res.output)
    assert data["price_per_hour"] == 2 and data["tokens_per_hour"] > 0


def test_cli_benchmark_text_and_usage_errors() -> None:
    r = CliRunner().invoke(fcli.factory_group, ["benchmark", "--model", "qwen2.5-0.5b", "--slots", "8"])
    assert r.exit_code == 0 and "tokens/hour" in r.output
    r = CliRunner().invoke(fcli.factory_group, ["benchmark"])
    assert r.exit_code == 1 and "exactly one" in r.output
    r = CliRunner().invoke(fcli.factory_group, ["benchmark", "--model", "nope"])
    assert r.exit_code == 1 and "unknown model preset" in r.output


def test_cli_benchmark_pack(tiny_pack_dir: Path) -> None:
    r = CliRunner().invoke(fcli.factory_group, ["benchmark", "--pack", str(tiny_pack_dir), "--slots", "4", "--json"])
    assert r.exit_code == 0, r.output
    assert json.loads(r.output)["model"] == "tiny-qwen2"


@pytest.fixture
def patched_store(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> FakeMaskStore:
    store = FakeMaskStore(root=tmp_path / "store", low=10, target=20)
    import pgw.factory.runtime as rt

    def fake_open_store(store_settings: Any = None, **overrides: Any) -> FakeMaskStore:
        return store

    monkeypatch.setattr(rt, "open_store", fake_open_store)
    return store


def test_cli_generate_and_status(tiny_pack_dir: Path, patched_store: FakeMaskStore, tmp_path: Path) -> None:
    runner = CliRunner()
    r = runner.invoke(fcli.factory_group, ["generate", "--pack", str(tiny_pack_dir), "--slots", "6", "--segments", "2",
                                           "--chunk-slots", "4", "--device", "cpu"])
    assert r.exit_code == 0, r.output
    assert r.output.count("segment ") == 2 and len(patched_store.segments) == 2
    r = runner.invoke(fcli.factory_group, ["status", "--pack", str(tiny_pack_dir), "--json"])
    assert r.exit_code == 0, r.output
    data = json.loads(r.output)
    assert data["models"][0]["available_slots"] == 12
    assert data["models"][0]["below_low_water"] is False
    r = runner.invoke(fcli.factory_group, ["status", "--pack", str(tiny_pack_dir), "--low-water", "50", "--target", "60"])
    assert r.exit_code == 0 and "plan:" in r.output and "LOW" in r.output


def test_cli_run_once(tiny_pack_dir: Path, patched_store: FakeMaskStore, tmp_path: Path) -> None:
    r = CliRunner().invoke(fcli.factory_group, [
        "run", "--once", "--pack", str(tiny_pack_dir), "--segment-slots", "8", "--low-water", "10", "--target", "20",
        "--ring", "K40A12", "--ring", "K48A20", "--chunk-slots", "4"], env={"PGW_FACTORY_STATE_DIR": str(tmp_path / "st")})
    assert r.exit_code == 0, r.output
    assert "produced 6 segment(s)" in r.output  # 3 x 8 slots per ring to reach 20
    rings = sorted({s.ring.name for s in patched_store.segments.values()})
    assert rings == ["K40A12", "K48A20"]
    assert (tmp_path / "st" / "factory-state.json").exists()


def test_cli_status_without_pack(patched_store: FakeMaskStore, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PGW_FACTORY_PACK_PATH", raising=False)
    from pgw.factory import MaskGenerator

    MaskGenerator(FakePack(), "K40A12").generate_into_store(patched_store, 3)
    r = CliRunner().invoke(fcli.factory_group, ["status"])
    assert r.exit_code == 0, r.output
    assert "3 slots available" in r.output


def test_cli_config_errors(tmp_path: Path) -> None:
    r = CliRunner().invoke(fcli.factory_group, ["run", "--once"], env={"PGW_FACTORY_PACK_PATH": None})
    assert r.exit_code == 1 and "pack_path" in r.output
    r = CliRunner().invoke(fcli.factory_group, ["run", "--once", "--pack", str(tmp_path)],
                           env={"PGW_MASKSTORE_KEY_FILE": None, "PGW_MASKSTORE_KEY_PROVIDER": "file"})
    assert r.exit_code in (1, 3) and "error" in r.output
    r = CliRunner().invoke(fcli.factory_group, ["generate", "--pack", str(tmp_path), "--segments", "0"])
    assert r.exit_code == 1 and "segments" in r.output


def test_cli_export_status(tmp_path: Path) -> None:
    key = tmp_path / "pub"
    key.write_bytes(bytes(32))
    r = CliRunner().invoke(fcli.factory_group, ["status", "--export-dir", str(tmp_path / "exp"), "--recipient-key",
                                                str(key), "--json"])
    # the staging MaskStore may still be unimplemented: either a clean report or exit code 3
    assert r.exit_code in (0, 3), r.output
    if r.exit_code == 0:
        assert json.loads(r.output)["sink"] == "export"
