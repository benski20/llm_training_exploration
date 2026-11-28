"""``arch.parallel_layers`` in the pack manifest (a partial parallel conversion, opt-in): written by
``pack_checkpoint(parallel_layers=...)``, covered by the fingerprint, absent (and the fingerprint unchanged) otherwise."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pgw.modelpack import ModelPack, pack_checkpoint


def test_parallel_layers_written_pinned_and_default_unchanged(tiny_hf_dir: Path, tmp_path: Path) -> None:
    plain = pack_checkpoint(tiny_hf_dir, tmp_path / "seq")
    par = pack_checkpoint(tiny_hf_dir, tmp_path / "par", parallel_layers=[1])
    raw = json.loads((tmp_path / "seq" / "manifest.json").read_text())
    assert "parallel_layers" not in raw["arch"]
    assert json.loads((tmp_path / "par" / "manifest.json").read_text())["arch"]["parallel_layers"] == [1]
    assert plain.fingerprint != par.fingerprint
    p = ModelPack.open(tmp_path / "par", verify="full")
    assert p.arch.parallel_layers == (1,) and p.fingerprint == par.fingerprint
    assert ModelPack.open(tmp_path / "seq").arch.parallel_layers is None
    # the field is covered: editing it breaks the stored fingerprint
    m = json.loads((tmp_path / "par" / "manifest.json").read_text())
    m["arch"]["parallel_layers"] = [0]
    (tmp_path / "par" / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(Exception, match=r"(?i)fingerprint"):
        ModelPack.open(tmp_path / "par")


def test_parallel_layers_validated(tiny_hf_dir: Path, tmp_path: Path) -> None:
    with pytest.raises(Exception, match="parallel_layers"):
        pack_checkpoint(tiny_hf_dir, tmp_path / "bad", parallel_layers=[2])
