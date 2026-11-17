"""Streaming safetensors writer, header reader and byte-range hashing."""

from __future__ import annotations

import hashlib
import struct

import pytest
import torch
from safetensors import safe_open

from pgw.core.errors import ConfigError, ShapeError
from pgw.modelpack.st_io import (
    SafetensorsStreamWriter,
    TensorPlan,
    hash_tensor_ranges,
    read_safetensors_header,
    tensor_sha256,
)


def _tensors() -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(0)
    return {
        "a.weight": torch.randint(-127, 128, (7, 5), dtype=torch.int8, generator=g),
        "a.scale": torch.rand(7, generator=g),
        "b": torch.randn(3, 4, generator=g).to(torch.bfloat16),
        "c": torch.randn(9, generator=g).to(torch.float16),
        "empty": torch.zeros(0, 4),
    }


def _plans(ts: dict[str, torch.Tensor]) -> list[TensorPlan]:
    names = {torch.int8: "int8", torch.float32: "float32", torch.bfloat16: "bfloat16", torch.float16: "float16"}
    return [TensorPlan(k, names[v.dtype], tuple(v.shape)) for k, v in ts.items()]


def test_roundtrip_with_safetensors_reader(tmp_path):
    ts = _tensors()
    path = tmp_path / "x.safetensors"
    with SafetensorsStreamWriter(path, _plans(ts), metadata={"format": "pt"}) as w:
        digests = {k: w.write(k, v) for k, v in ts.items()}
    assert path.stat().st_size == w.total_bytes
    with safe_open(str(path), framework="pt") as f:
        assert set(f.keys()) == set(ts)
        assert f.metadata() == {"format": "pt"}
        for k, v in ts.items():
            got = f.get_tensor(k)
            assert got.dtype == v.dtype and torch.equal(got, v)
    entries, meta = read_safetensors_header(path)
    assert meta == {"format": "pt"}
    assert (8 + struct.unpack("<Q", path.read_bytes()[:8])[0]) % 8 == 0  # aligned data section
    hashed = hash_tensor_ranges(path, entries.values())
    for k, v in ts.items():
        assert hashed[k] == digests[k] == tensor_sha256(v)
    assert digests["empty"] == hashlib.sha256(b"").hexdigest()


def test_tensor_sha256_is_byte_level():
    t = torch.arange(4, dtype=torch.float32)
    assert tensor_sha256(t) == hashlib.sha256(t.numpy().tobytes()).hexdigest()
    assert tensor_sha256(t.reshape(2, 2).t()) == tensor_sha256(t.reshape(2, 2).t().contiguous())


def test_order_and_shape_enforced(tmp_path):
    ts = _tensors()
    path = tmp_path / "x.safetensors"
    with pytest.raises(ConfigError, match="order"), SafetensorsStreamWriter(path, _plans(ts)) as w:
        w.write("a.scale", ts["a.scale"])
    assert not path.exists()  # partial file removed
    with pytest.raises(ShapeError), SafetensorsStreamWriter(path, _plans(ts)) as w:
        w.write("a.weight", ts["a.weight"].float())
    assert not path.exists()


def test_incomplete_write_fails_and_removes_file(tmp_path):
    ts = _tensors()
    path = tmp_path / "x.safetensors"
    with pytest.raises(ConfigError, match="not written"), SafetensorsStreamWriter(path, _plans(ts)) as w:
        w.write("a.weight", ts["a.weight"])
    assert not path.exists()


def test_duplicate_names_and_bad_dtype(tmp_path):
    with pytest.raises(ConfigError, match="duplicate"):
        SafetensorsStreamWriter(tmp_path / "x", [TensorPlan("a", "int8", (1,)), TensorPlan("a", "int8", (1,))])
    with pytest.raises(ConfigError, match="dtype"):
        SafetensorsStreamWriter(tmp_path / "x", [TensorPlan("a", "int4", (1,))])


def test_header_reader_rejects_corrupt_files(tmp_path):
    ts = _tensors()
    path = tmp_path / "x.safetensors"
    with SafetensorsStreamWriter(path, _plans(ts)) as w:
        for k, v in ts.items():
            w.write(k, v)
    data = path.read_bytes()
    (tmp_path / "trunc.safetensors").write_bytes(data[:-3])
    with pytest.raises(ConfigError, match="inconsistent offsets"):
        read_safetensors_header(tmp_path / "trunc.safetensors")
    (tmp_path / "tiny.safetensors").write_bytes(b"\x01\x02")
    with pytest.raises(ConfigError, match="truncated"):
        read_safetensors_header(tmp_path / "tiny.safetensors")
    (tmp_path / "huge.safetensors").write_bytes(struct.pack("<Q", 1 << 40) + b"{}")
    with pytest.raises(ConfigError, match="header length"):
        read_safetensors_header(tmp_path / "huge.safetensors")
    (tmp_path / "junk.safetensors").write_bytes(struct.pack("<Q", 4) + b"{{{{")
    with pytest.raises(ConfigError, match="malformed"):
        read_safetensors_header(tmp_path / "junk.safetensors")
    with pytest.raises(ConfigError, match="missing"):
        read_safetensors_header(tmp_path / "nope.safetensors")
