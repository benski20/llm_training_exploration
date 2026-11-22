"""``PGW_SETUP_REUSE_PACK`` (opt-in): the wizard's pack job uses a pack already in its directory only when its full
verification gives the enrolled fingerprint; otherwise it deletes it and builds from the checkpoint."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest

from pgw.gateway.setup_jobs import PackBuildJob
from pgw.modelpack import ModelPack


def _model(fp: str) -> dict[str, Any]:
    return {"name": "tiny-qwen2", "fingerprint": fp, "hf_repo": "x/tiny", "revision": None}


def test_reuse_needs_the_switch_and_the_fingerprint(tiny_pack_dir: Path, tiny_hf_dir: Path, tmp_path: Path,
                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    with ModelPack.open(tiny_pack_dir, verify="full") as p:
        fp = p.fingerprint
    out = tmp_path / "packs" / "tiny-qwen2"
    shutil.copytree(tiny_pack_dir, out)
    resolved: list[str] = []

    def resolver(repo: str, rev: str | None) -> Path:
        resolved.append(repo)
        return tiny_hf_dir

    # switched on, matching fingerprint: no download, done
    monkeypatch.setenv("PGW_SETUP_REUSE_PACK", "1")
    job = PackBuildJob(_model(fp), out, resolver)
    job.run()
    assert job.status()["phase"] == "done" and resolved == []
    # off (library default): always builds from the checkpoint
    monkeypatch.delenv("PGW_SETUP_REUSE_PACK")
    job = PackBuildJob(_model(fp), out, resolver)
    job.run()
    assert job.status()["phase"] == "done" and resolved == ["x/tiny"]
    # on, but the enrolled fingerprint differs: the prepared pack is deleted, then the normal build runs (and fails
    # its own fingerprint check here, deleting what it built)
    monkeypatch.setenv("PGW_SETUP_REUSE_PACK", "1")
    job = PackBuildJob(_model("00" * 32), out, resolver)
    job.run()
    assert job.status()["phase"] == "failed" and resolved == ["x/tiny", "x/tiny"] and not out.exists()
    # on, tampered tensor file: the full verification fails, the pack is rebuilt from the checkpoint
    shutil.copytree(tiny_pack_dir, out)
    victim = max((p for p in out.rglob("*") if p.is_file() and p.name != "manifest.json"),
                 key=lambda p: p.stat().st_size)
    b = bytearray(victim.read_bytes())
    b[len(b) // 2] ^= 1
    victim.write_bytes(bytes(b))
    job = PackBuildJob(_model(fp), out, resolver)
    job.run()
    assert job.status()["phase"] == "done" and resolved == ["x/tiny"] * 3
    with ModelPack.open(out, verify="full") as p:
        assert p.fingerprint == fp
