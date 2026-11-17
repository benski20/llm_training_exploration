"""Tests for ring tensor byte packing."""

import pytest
import torch

from pgw.core import ProtocolError, SeedExpander, ShapeError, pack_ring, packed_size, unpack_ring


@pytest.mark.parametrize("k", [16, 32, 40, 48, 56, 64])
def test_roundtrip(k):
    v = SeedExpander(bytes(32)).ring_uniform((7, 13), min(k, 63), "p")
    if k < 63:
        v[0, 0] = (1 << k) - 1
    data = pack_ring(v, k)
    assert len(data) == packed_size((7, 13), k) == 7 * 13 * k // 8
    assert torch.equal(unpack_ring(data, (7, 13), k), v & ((1 << k) - 1) if k < 64 else v)


def test_reduces_and_validates():
    v = torch.tensor([-1, 1 << 41], dtype=torch.int64)
    assert unpack_ring(pack_ring(v, 40), (2,), 40).tolist() == [(1 << 40) - 1, 0]
    assert pack_ring(torch.tensor([1, 256]), 16) == b"\x01\x00\x00\x01"
    with pytest.raises(ProtocolError):
        unpack_ring(b"\x00" * 9, (2,), 40)
    with pytest.raises(ShapeError):
        pack_ring(torch.zeros(2, dtype=torch.int32), 40)
    with pytest.raises(ShapeError):
        pack_ring(torch.zeros(2, dtype=torch.int64), 12)
