"""Tests for :mod:`pgw.service.registry` (tiny pack on CPU)."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest
import torch

from pgw.core import (
    RING_K40_A12,
    ConfigError,
    FingerprintMismatchError,
    ModelNotFoundError,
    ProtocolError,
    RingParams,
)
from pgw.service import ModelRegistry, place_layers
from tests.service.conftest import expected, rand_x


def test_load_and_lookup(shared_registry: ModelRegistry, tiny_pack: Any, pack_weights: dict[str, torch.Tensor]) -> None:
    fp = tiny_pack.fingerprint
    m = shared_registry.get(fp)
    assert m.name == tiny_pack.name and m.fingerprint == fp
    assert fp in shared_registry and len(shared_registry) == 1
    assert [s.id.key for s in m.specs] == [s.id.key for s in tiny_pack.linear_specs()]
    assert set(m.kernels) == set(pack_weights)
    assert m.rings == ("K40A12", "K48A20", "K32A8")
    assert m.devices() == ["cpu"] and set(m.placement.values()) == {"cpu"}
    assert m.device_bytes > 0
    for lid, w in pack_weights.items():
        assert m.linear_dims[lid] == tuple(w.shape)
        x = rand_x(4, w.shape[1], RING_K40_A12, seed=hash(lid) % 1000)
        assert torch.equal(m.kernel(lid).run(x, RING_K40_A12), expected(x, w, RING_K40_A12))
    assert m.serves_ring("k40a12") and m.serves_ring(RingParams(32, 8))
    with pytest.raises(ProtocolError):
        m.kernel("9.qkv")


def test_linear_dims_errors(shared_registry: ModelRegistry, pack_fp: str) -> None:
    dims = shared_registry.linear_dims(pack_fp, "K48A20")
    assert "0.qkv" in dims
    with pytest.raises(FingerprintMismatchError):
        shared_registry.linear_dims("ab" * 32, "K40A12")
    with pytest.raises(ProtocolError):
        shared_registry.linear_dims(pack_fp, "bogus")
    with pytest.raises(ModelNotFoundError):
        shared_registry.get("cd" * 32)


def test_ring_subset_and_not_served(tiny_pack_dir: Path, pack_fp: str) -> None:
    reg = ModelRegistry("cpu", rings=["K48A20"], verify="fingerprint")
    m = reg.load(tiny_pack_dir)
    assert m.rings == ("K48A20",)
    with pytest.raises(ConfigError) as ei:
        reg.linear_dims(pack_fp, "K40A12")
    assert ei.value.details["rings"] == ["K48A20"]


def test_duplicate_load_returns_existing_and_unload(tiny_pack_dir: Path, pack_fp: str) -> None:
    reg = ModelRegistry("cpu", verify="fingerprint")
    a = reg.load(tiny_pack_dir)
    b = reg.load(tiny_pack_dir)
    assert a is b and len(reg) == 1
    with pytest.raises(ConfigError):
        reg.register(a)
    reg.unload(pack_fp)
    assert len(reg) == 0
    with pytest.raises(ModelNotFoundError):
        reg.unload(pack_fp)


def test_expected_fingerprint_mismatch(tiny_pack_dir: Path) -> None:
    reg = ModelRegistry("cpu", verify="fingerprint")
    with pytest.raises(FingerprintMismatchError):
        reg.load(tiny_pack_dir, expected_fingerprint="ab" * 32)


def test_full_verify_detects_tampered_shard(tiny_pack_dir: Path, tmp_path: Path) -> None:
    bad = tmp_path / "pack"
    shutil.copytree(tiny_pack_dir, bad)
    shard = sorted(bad.glob("linears-*.safetensors"))[0]
    data = bytearray(shard.read_bytes())
    data[-1] ^= 0x01  # flip one weight bit
    shard.write_bytes(bytes(data))
    with pytest.raises(FingerprintMismatchError):
        ModelRegistry("cpu", verify="full").load(bad)
    ModelRegistry("cpu", verify="fingerprint").load(bad)  # cheap mode does not hash data


def test_linears_only_subset_pack(tiny_pack: Any, tmp_path: Path) -> None:
    sub = tmp_path / "service-pack"
    tiny_pack.export(sub, ["linears"])
    assert not (sub / "plain.safetensors").exists()
    m = ModelRegistry("cpu").load(sub)
    assert m.fingerprint == tiny_pack.fingerprint


def test_missing_pack_dir(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        ModelRegistry("cpu").load(tmp_path / "nope")


def test_constructor_validation() -> None:
    with pytest.raises(ConfigError):
        ModelRegistry("cpu", placement="random")
    with pytest.raises(ConfigError):
        ModelRegistry("cpu", verify="none")
    with pytest.raises(ConfigError):
        ModelRegistry("cpu", rings=[])
    with pytest.raises(ConfigError):
        ModelRegistry("cpu", rings=["X1"])
    reg = ModelRegistry(devices=["cpu", "cpu"])  # duplicates collapse
    assert reg.devices == [torch.device("cpu")]


def test_ring_without_headroom_rejected(tiny_pack_dir: Path) -> None:
    # K16A8: max inner dim = (2^15-1) // (127*127) = 2 < 64
    reg = ModelRegistry("cpu", rings=["K16A8"], verify="fingerprint")
    with pytest.raises(ConfigError, match="headroom"):
        reg.load(tiny_pack_dir)


def test_check_devices_and_summary(shared_registry: ModelRegistry) -> None:
    assert shared_registry.check_devices() == {"cpu": True}
    summary = shared_registry.device_summary()
    assert summary[0]["device"] == "cpu" and summary[0]["linears"] == 8


@pytest.mark.parametrize(
    ("layer_bytes", "n_dev", "expect"),
    [
        ([10] * 8, 2, [0, 0, 0, 0, 1, 1, 1, 1]),
        ([10] * 6, 3, [0, 0, 1, 1, 2, 2]),
        ([10] * 2, 4, [0, 1]),
        ([10] * 5, 1, [0] * 5),
        ([100, 1, 1, 1], 2, [0, 1, 1, 1]),
        ([1, 1, 1, 100], 2, [0, 0, 0, 1]),
        ([10] * 3, 3, [0, 1, 2]),
    ],
)
def test_place_layers(layer_bytes: list[int], n_dev: int, expect: list[int]) -> None:
    devs = [f"cuda:{i}" for i in range(n_dev)]
    out = place_layers(layer_bytes, devs)
    assert [devs.index(out[i]) for i in range(len(layer_bytes))] == expect


def test_place_layers_properties() -> None:
    import random

    rnd = random.Random(0)
    for _ in range(200):
        n_layers = rnd.randint(1, 40)
        n_dev = rnd.randint(1, 8)
        lb = [rnd.randint(1, 100) for _ in range(n_layers)]
        devs = [f"d{i}" for i in range(n_dev)]
        out = place_layers(lb, devs)
        idx = [devs.index(out[i]) for i in range(n_layers)]
        assert idx == sorted(idx), "ranges must be contiguous and ordered"
        assert idx[0] == 0
        assert len(set(idx)) == min(n_dev, n_layers), "every device used when possible"
    with pytest.raises(ConfigError):
        place_layers([1, 2], [])


def test_by_layer_placement_records_devices(tiny_pack_dir: Path) -> None:
    reg = ModelRegistry(devices=["cpu"], placement="by_layer", verify="fingerprint")
    m = reg.load(tiny_pack_dir)
    assert m.placement == {0: "cpu", 1: "cpu"}


@pytest.mark.gpu
def test_multi_gpu_placement(tiny_pack_dir: Path) -> None:  # pragma: no cover - GPU only
    if torch.cuda.device_count() < 2:
        pytest.skip("needs 2 GPUs")
    reg = ModelRegistry(devices=["cuda:0", "cuda:1"], placement="by_layer")
    m = reg.load(tiny_pack_dir)
    assert m.placement == {0: "cuda:0", 1: "cuda:1"}
    assert str(m.kernel("1.down").device) == "cuda:1"
