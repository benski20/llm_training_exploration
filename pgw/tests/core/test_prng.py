"""Tests for the CSPRNG seed expander, including frozen known-answer vectors.

The known-answer vectors pin the construction so independent implementations (e.g. a Rust or GPU
mask factory) can be checked against Python. Changing them is a protocol break.
"""

import numpy as np
import pytest
import torch

from pgw.core import SEED_BYTES, ConfigError, SeedExpander, new_seed
from pgw.core.prng import encode_labels

KAT_SEED = bytes(range(32))


def test_known_answers():
    s = SeedExpander(KAT_SEED)
    assert s.random_bytes(16, "mask", "0.qkv", 0).hex() == "a2b5b70fe6fc44d9296ef4cb4aa4a77b"
    assert s.ring_uniform(4, 40, "mask", "0.qkv", 0).tolist() == [
        988106175906, 321249373737, 254846079836, 563174071857,
    ]
    assert s.derive_seed("segment", 1).hex() == "00bc5de3b96a347f37eb7113018cc757222c86488560b4a81e1d4df28c199af7"


def test_deterministic_and_label_separated():
    s = SeedExpander(KAT_SEED)
    a = s.ring_uniform((3, 5), 48, "x", 1)
    assert torch.equal(a, SeedExpander(KAT_SEED).ring_uniform((3, 5), 48, "x", 1))
    assert not torch.equal(a, s.ring_uniform((3, 5), 48, "x", 2))
    assert not torch.equal(a, s.ring_uniform((3, 5), 48, "x", "1"))
    assert s.random_bytes(32, "a", "bc") != s.random_bytes(32, "ab", "c")
    assert encode_labels(["a", "bc"]) != encode_labels(["ab", "c"])
    assert s.random_bytes(10, "q") == s.random_bytes(20, "q")[:10]  # prefix property of the stream
    assert s.child("c").random_bytes(8, "z") == SeedExpander(s.derive_seed("c")).random_bytes(8, "z")


@pytest.mark.parametrize("k", [32, 40, 48])
def test_ring_uniform_range_and_balance(k):
    v = SeedExpander(new_seed()).ring_uniform(200_000, k, "u")
    assert int(v.min()) >= 0 and int(v.max()) < (1 << k)
    top = (v >> (k - 4)).numpy()
    cnt = np.bincount(top, minlength=16)
    assert cnt.min() > 0.9 * len(top) / 16
    assert abs(float((v & 1).double().mean()) - 0.5) < 0.01


def test_uniform_int():
    v = SeedExpander(KAT_SEED).uniform_int((1000,), -127, 128, "w")
    assert int(v.min()) >= -127 and int(v.max()) <= 127
    with pytest.raises(ConfigError):
        SeedExpander(KAT_SEED).uniform_int(3, 5, 5)


def test_seed_validation_and_redaction():
    assert len(new_seed()) == SEED_BYTES
    with pytest.raises(ConfigError):
        SeedExpander(b"short")
    with pytest.raises(ConfigError):
        SeedExpander(KAT_SEED).random_bytes(4, True)  # type: ignore[arg-type]
    with pytest.raises(ConfigError):
        SeedExpander(KAT_SEED).random_bytes(4, 1.5)  # type: ignore[arg-type]
    assert KAT_SEED.hex() not in repr(SeedExpander(KAT_SEED))


@pytest.mark.parametrize("cols,start,count", [(1, 0, 5), (3, 7, 4), (13, 5, 0), (896, 2, 3), (5, 0, 40)])
def test_ring_uniform_rows_matches_full(cols, start, count):
    s = SeedExpander(KAT_SEED)
    full = s.ring_uniform((48, cols), 40, "r", "0.qkv")
    part = s.ring_uniform_rows(cols, 40, start, count, "r", "0.qkv")
    assert torch.equal(part, full[start:start + count])
    with pytest.raises(ConfigError):
        s.ring_uniform_rows(0, 40, 0, 1)
