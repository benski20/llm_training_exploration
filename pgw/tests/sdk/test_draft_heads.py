"""Shipped draft heads (``pgw.sdk.draft_heads``): pinned by SHA-256, found, fetched and refused as they should be."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from pgw.core.errors import ConfigError, IntegrityError
from pgw.sdk.draft_heads import PACKAGE_DIR, PINNED_DRAFT_HEADS, DraftHeadPin, resolve_draft_head, verify_draft_head

#: Hidden size of each supported model (the head's input width).
HIDDEN = {"qwen2.5-0.5b-instruct": 896, "qwen2.5-1.5b-instruct": 1536, "qwen2.5-3b-instruct": 2048,
          "qwen2.5-7b-instruct": 3584}


def _pin_for(path: Path, name: str = "m.pgwhead") -> DraftHeadPin:
    data = path.read_bytes()
    return DraftHeadPin(name, hashlib.sha256(data).hexdigest(), len(data), "org/model", "0" * 40)


@pytest.fixture()
def head(tmp_path: Path) -> Path:
    p = tmp_path / "src" / "m.pgwhead"
    p.parent.mkdir()
    p.write_bytes(b"pgw-head-bytes" * 1000)
    return p


def test_unknown_model_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="no trained draft head"):
        resolve_draft_head("no-such-model", cache_dir=tmp_path, pins={})


def test_shipped_file_is_verified(head: Path, tmp_path: Path) -> None:
    pins = {"m": _pin_for(head)}
    assert resolve_draft_head("m", pins=pins, package_dir=head.parent, cache_dir=tmp_path) == head
    head.write_bytes(head.read_bytes()[:-1] + b"X")  # same size, other bytes
    with pytest.raises(IntegrityError, match="SHA-256"):
        resolve_draft_head("m", pins=pins, package_dir=head.parent, cache_dir=tmp_path)


def test_fetched_from_a_mirror_into_the_cache(head: Path, tmp_path: Path) -> None:
    pins = {"m": _pin_for(head)}
    empty = tmp_path / "pkg"
    empty.mkdir()
    cache = tmp_path / "cache"
    with pytest.raises(ConfigError, match="PGW_DRAFT_HEAD_URL"):
        resolve_draft_head("m", pins=pins, package_dir=empty, cache_dir=cache, base_url=None)
    url = head.parent.as_uri()
    got = resolve_draft_head("m", pins=pins, package_dir=empty, cache_dir=cache, base_url=url)
    assert got == cache / "draft-heads" / "m.pgwhead" and got.read_bytes() == head.read_bytes()
    # a corrupted cache copy is fetched again
    got.write_bytes(b"junk")
    assert resolve_draft_head("m", pins=pins, package_dir=empty, cache_dir=cache, base_url=url).read_bytes() == \
        head.read_bytes()


def test_a_mirror_with_other_bytes_is_refused(head: Path, tmp_path: Path) -> None:
    pins = {"m": DraftHeadPin("m.pgwhead", "0" * 64, head.stat().st_size, "org/model", "0" * 40)}
    empty = tmp_path / "pkg"
    empty.mkdir()
    cache = tmp_path / "cache"
    with pytest.raises(IntegrityError, match="SHA-256"):
        resolve_draft_head("m", pins=pins, package_dir=empty, cache_dir=cache, base_url=head.parent.as_uri())
    assert not list((cache / "draft-heads").iterdir())  # nothing unverified is kept


@pytest.mark.parametrize("model", sorted(PINNED_DRAFT_HEADS))
def test_every_pinned_head_ships_and_matches(model: str) -> None:
    """The release's heads: the shipped file matches its pin, is a v2 head for the pinned checkpoint's hidden size,
    and the pin names the SDK's own checkpoint commit for the model."""
    from pgw.engine.generate import FeatureHeadDrafter
    from pgw.sdk.provisioning import PINNED_MODELS

    pin = PINNED_DRAFT_HEADS[model]
    path = PACKAGE_DIR / pin.file
    assert verify_draft_head(path, pin)
    assert (pin.hf_repo, pin.revision) == (PINNED_MODELS[model].hf_repo, PINNED_MODELS[model].revision)
    blob = torch.load(path, map_location="cpu", weights_only=True)
    assert blob["format"] == FeatureHeadDrafter.FILE_FORMAT and int(blob["hidden"]) == HIDDEN[model]
    assert blob["target"] == pin.hf_repo and "float" not in blob and "in_scale" in blob
    emb = torch.zeros(152064 if model != "qwen2.5-0.5b-instruct" else 151936, HIDDEN[model])
    d = FeatureHeadDrafter.from_file(path, emb, kernel="float")
    assert d.nbytes_per_step() < 20_000_000


def test_engine_uses_the_pinned_head_unless_a_path_is_given(tmp_path: Path) -> None:
    from pgw.sdk.engine import Engine
    from pgw.sdk.options import SDKOptions

    if not PINNED_DRAFT_HEADS:
        pytest.skip("no shipped heads")
    model = sorted(PINNED_DRAFT_HEADS)[0]
    eng = SimpleNamespace(options=SDKOptions(speculation="draft_head"), layout=SimpleNamespace(models_dir=tmp_path))
    kw = Engine._spec_config(eng, model)  # type: ignore[arg-type]
    assert kw["draft_backend"] == "feature_head" and kw["draft_head_path"] == PACKAGE_DIR / PINNED_DRAFT_HEADS[model].file
    own = SimpleNamespace(options=SDKOptions(speculation="draft_head", draft_head_path=Path("/own.pt")))
    assert Engine._spec_config(own, model)["draft_head_path"] == Path("/own.pt")  # type: ignore[arg-type]
    with pytest.raises(ConfigError, match="no trained draft head"):
        Engine._spec_config(eng, "qwen2.5-72b-instruct")  # type: ignore[arg-type]
