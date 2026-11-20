"""Weight-row norms from the manifest (``row_norm_sq_max``): no int8 weight on the gateway's request path."""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any

import pytest
import torch

from pgw.core import RING_K32_A12_C27, FingerprintMismatchError, LinearId
from pgw.core.fingerprint import compute_fingerprint
from pgw.engine import LinearCall, LocalComputeClient, LocalMaskReservation, MaskedBackend, PlainBackend
from pgw.engine.backends import _row_norm_max
from pgw.modelpack import MANIFEST_FILE, Manifest, ModelPack
from pgw.modelpack import loader as loader_mod
from pgw.modelpack.loader import row_norm_sq_max

RING = RING_K32_A12_C27
QKV0 = LinearId.parse("0.qkv")


def _x(rows: int, n: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(rows, n, generator=g) * 3


class GatewayView:
    """The gateway's view of a pack: loading a full int8 weight is a test failure (unless allowed)."""

    def __init__(self, pack: Any, *, allow_weights: bool = False, root: Path | None = None) -> None:
        self._pack = pack
        self.allow_weights = allow_weights
        self.weight_loads = 0
        if root is not None:
            self.root = root

    def __getattr__(self, name: str) -> Any:
        return getattr(self._pack, name)

    def load_linear(self, *a: Any, **k: Any) -> Any:
        self.weight_loads += 1
        if not self.allow_weights:
            raise AssertionError("gateway loaded a full int8 weight")
        return self._pack.load_linear(*a, **k)


def _strip_norms(src: Path, dst: Path) -> Path:
    """Copy a pack and rewrite its manifest as written before ``row_norm_sq_max`` existed."""
    shutil.copytree(src, dst)
    m = Manifest.from_file(dst)
    old = m.model_copy(update={
        "linears": tuple(e.model_copy(update={"row_norm_sq_max": None}) for e in m.linears),
        "fingerprint": None,
    }).with_fingerprint()
    (dst / MANIFEST_FILE).write_text(old.to_json() + "\n", encoding="utf-8")
    return dst


@pytest.fixture
def old_pack(tiny_pack_dir: Path, tmp_path: Path) -> ModelPack:
    return ModelPack.open(_strip_norms(tiny_pack_dir, tmp_path / "old"), verify="full")


# ---- exactness of the integer form ------------------------------------------------------------------
@pytest.mark.parametrize("shape", [(1, 1), (7, 64), (33, 4864), (5, 18944)])
def test_sqrt_of_int_norm_equals_float_norm(shape: tuple[int, int]) -> None:
    g = torch.Generator().manual_seed(shape[1])
    for w8 in (
        torch.randint(-127, 128, shape, generator=g, dtype=torch.int8),
        torch.full(shape, 127, dtype=torch.int8),
        torch.full(shape, -127, dtype=torch.int8),
        torch.zeros(shape, dtype=torch.int8),
    ):
        sq = row_norm_sq_max(w8)
        assert sq == int((w8.to(torch.int64) ** 2).sum(dim=1).max())
        assert sq < 1 << 53  # the float64 row sums are exact, so sqrt is bit-identical
        assert math.sqrt(float(sq)) == _row_norm_max(w8)


def test_blocked_computation_matches(monkeypatch: pytest.MonkeyPatch) -> None:
    g = torch.Generator().manual_seed(1)
    w8 = torch.randint(-127, 128, (37, 300), generator=g, dtype=torch.int8)
    w8[29] = 127  # the maximum sits in a later block
    full = row_norm_sq_max(w8)
    monkeypatch.setattr(loader_mod, "_NORM_BLOCK_ELEMS", 1000)  # 3 rows per block
    assert row_norm_sq_max(w8) == full == 127 * 127 * 300


# ---- manifest / fingerprint -----------------------------------------------------------------------
def test_packer_writes_norms(tiny_pack: ModelPack) -> None:
    for e in tiny_pack.manifest.linears:
        assert e.row_norm_sq_max == row_norm_sq_max(tiny_pack.load_linear(e.id).w8)
        assert tiny_pack.load_linear_meta(e.id).row_norm_sq_max == e.row_norm_sq_max
    assert tiny_pack.verify_row_norms() == len(tiny_pack.manifest.linears)


def test_fingerprint_covers_norm_and_old_packs_keep_theirs(tiny_pack: ModelPack, old_pack: ModelPack) -> None:
    m = tiny_pack.manifest
    # covered: changing one value changes the fingerprint
    e0 = m.linears[0]
    bumped = m.model_copy(update={"linears": (e0.model_copy(update={"row_norm_sq_max": e0.row_norm_sq_max - 1}),
                                              *m.linears[1:])})
    assert bumped.compute_fingerprint() != m.compute_fingerprint()
    # an old manifest fingerprints exactly as before the field existed (no key in the semantic dict)
    om = old_pack.manifest
    legacy = om.model_dump(mode="json")
    for e in legacy["linears"]:
        assert e.pop("row_norm_sq_max") is None
    assert compute_fingerprint(legacy) == om.compute_fingerprint() == old_pack.fingerprint
    assert old_pack.fingerprint != tiny_pack.fingerprint
    assert "row_norm_sq_max" not in om.to_json() and "row_norm_sq_max" in m.to_json()
    assert old_pack.verify_row_norms() == 0


def test_manifest_rejects_impossible_norm(tiny_pack: ModelPack) -> None:
    e = tiny_pack.manifest.linears[0]
    with pytest.raises(ValueError):
        type(e).model_validate({**e.model_dump(), "row_norm_sq_max": 127 * 127 * e.in_features + 1})
    with pytest.raises(ValueError):
        type(e).model_validate({**e.model_dump(), "row_norm_sq_max": -1})


def test_verify_row_norms_detects_mismatch(tiny_pack_dir: Path, tmp_path: Path) -> None:
    dst = tmp_path / "bad"
    shutil.copytree(tiny_pack_dir, dst)
    m = Manifest.from_file(dst)
    e0 = m.linears[0]
    bad = m.model_copy(update={
        "linears": (e0.model_copy(update={"row_norm_sq_max": e0.row_norm_sq_max - 1}), *m.linears[1:]),
        "fingerprint": None,
    }).with_fingerprint()
    (dst / MANIFEST_FILE).write_text(bad.to_json(), encoding="utf-8")
    with ModelPack.open(dst) as p, pytest.raises(FingerprintMismatchError):
        p.verify_row_norms()


# ---- gateway backend -----------------------------------------------------------------------------
def test_masked_capped_uses_manifest_norm(tiny_pack: ModelPack) -> None:
    view = GatewayView(tiny_pack)
    be = MaskedBackend(view, None, LocalComputeClient(tiny_pack), RING, tiny_pack.fingerprint)
    plain = PlainBackend(tiny_pack, RING)
    for spec in tiny_pack.linear_specs():
        x = _x(3, spec.in_features, spec.id.ordinal)
        res = LocalMaskReservation(tiny_pack, RING, 4)
        assert torch.equal(be.forward(LinearCall(spec.id, x, session=res)), plain.forward(LinearCall(spec.id, x)))
    assert view.weight_loads == 0 and be.hot_weight_loads == 0
    assert be.warm_weight_norms() == 0 and view.weight_loads == 0


def test_old_pack_warm_writes_sidecar_then_reads_no_weights(old_pack: ModelPack) -> None:
    root = old_pack.root
    view = GatewayView(old_pack, allow_weights=True)
    be = MaskedBackend(view, None, LocalComputeClient(old_pack), RING, old_pack.fingerprint)
    assert be.warm_weight_norms() == len(old_pack.manifest.linears)
    sidecar = root / f"wnorm.{old_pack.fingerprint}.json"
    doc = json.loads(sidecar.read_text())
    assert doc["fingerprint"] == old_pack.fingerprint
    assert doc["row_norm_sq_max"] == {e.id: row_norm_sq_max(old_pack.load_linear(e.id).w8)
                                      for e in old_pack.manifest.linears}
    loads = view.weight_loads
    view.allow_weights = False  # requests after warm-up never read a weight
    plain = PlainBackend(old_pack, RING)
    x = _x(2, old_pack.arch.hidden_size, 9)
    y = be.forward(LinearCall(QKV0, x, session=LocalMaskReservation(old_pack, RING, 2)))
    assert torch.equal(y, plain.forward(LinearCall(QKV0, x)))
    assert view.weight_loads == loads and be.hot_weight_loads == 0
    # the next start reads the sidecar: no weight at all
    view2 = GatewayView(old_pack)
    be2 = MaskedBackend(view2, None, LocalComputeClient(old_pack), RING, old_pack.fingerprint)
    assert be2.warm_weight_norms() == 0 and view2.weight_loads == 0
    y2 = be2.forward(LinearCall(QKV0, x, session=LocalMaskReservation(old_pack, RING, 2)))
    assert torch.equal(y2, y)


def test_old_pack_read_only_dir_uses_cache_dir(old_pack: ModelPack, tmp_path: Path) -> None:
    view = GatewayView(old_pack, allow_weights=True, root=Path("/proc/pgw-not-writable"))
    be = MaskedBackend(view, None, LocalComputeClient(old_pack), RING, old_pack.fingerprint)
    cache = tmp_path / "cache"
    assert be.warm_weight_norms(cache_dir=cache) > 0
    assert (cache / f"wnorm.{old_pack.fingerprint}.json").is_file()
    view2 = GatewayView(old_pack, root=Path("/proc/pgw-not-writable"))
    be2 = MaskedBackend(view2, None, LocalComputeClient(old_pack), RING, old_pack.fingerprint)
    assert be2.warm_weight_norms(cache_dir=cache) == 0 and view2.weight_loads == 0


def test_invalid_sidecar_is_ignored(old_pack: ModelPack) -> None:
    sidecar = old_pack.root / f"wnorm.{old_pack.fingerprint}.json"
    view = GatewayView(old_pack, allow_weights=True)
    MaskedBackend(view, None, LocalComputeClient(old_pack), RING, old_pack.fingerprint).warm_weight_norms()
    doc = json.loads(sidecar.read_text())
    key = next(iter(doc["row_norm_sq_max"]))
    for tamper in ({"fingerprint": "00" * 32}, {"row_norm_sq_max": {**doc["row_norm_sq_max"], key: 10**12}},
                   {"row_norm_sq_max": {**doc["row_norm_sq_max"], key: "1"}}):
        sidecar.write_text(json.dumps({**doc, **tamper}))
        v = GatewayView(old_pack, allow_weights=True)
        be = MaskedBackend(v, None, LocalComputeClient(old_pack), RING, old_pack.fingerprint)
        assert be.warm_weight_norms() == len(old_pack.manifest.linears)  # recomputed
        assert json.loads(sidecar.read_text()) == doc  # and rewritten correctly


def test_old_pack_without_warm_falls_back_once_per_linear(old_pack: ModelPack) -> None:
    view = GatewayView(old_pack, allow_weights=True)
    be = MaskedBackend(view, None, LocalComputeClient(old_pack), RING, old_pack.fingerprint)
    x = _x(1, old_pack.arch.hidden_size, 3)
    for _ in range(3):
        be.forward(LinearCall(QKV0, x, session=LocalMaskReservation(old_pack, RING, 1)))
    assert be.hot_weight_loads == 1 and view.weight_loads == 1


def test_decode_stats(tiny_pack: ModelPack) -> None:
    be = MaskedBackend(tiny_pack, None, LocalComputeClient(tiny_pack), RING, tiny_pack.fingerprint)
    n = tiny_pack.arch.hidden_size
    be.forward(LinearCall(QKV0, _x(1, n), session=LocalMaskReservation(tiny_pack, RING, 1)))
    be.forward(LinearCall(QKV0, _x(12, n), session=LocalMaskReservation(tiny_pack, RING, 12)))
    st = be.stats.snapshot()
    assert st["round_trips"] == 2 and st["decode_calls"] == 1
    assert 0 < st["decode_local_seconds"] <= st["local_seconds"]
    assert 0 <= st["decode_remote_seconds"] <= st["remote_seconds"]
