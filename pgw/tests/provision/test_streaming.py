"""Optional hint streaming (pgw.provision.streaming): bounded cache, digest checks, the real backend on top."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from pgw.core.errors import IntegrityError
from pgw.core.ring import RingParams
from pgw.hemask.pks_backend import PKSHintStore, PKSMaskBackend, RemotePKS
from pgw.provision.streaming import DirectorySource, StreamingHintStore, offload
from tests.hemask.test_pks import _LocalProvider, _Pack

KEY = bytes(range(32))


@pytest.fixture
def built(tmp_path: Path) -> tuple[_Pack, PKSHintStore, Path]:
    pack = _Pack({"blk.0.q": (64, 48), "blk.0.down": (32, 96), "blk.1.q": (96, 48)})
    root = tmp_path / "hints"
    store = PKSHintStore.build(root, pack, mac_key=KEY)
    want = {k: np.array(store.hint(k)) for k in pack.w}
    remote = offload(store, tmp_path / "remote")
    assert not list(Path(store.path).glob("*.npy"))  # only meta.json stays local
    store._maps.clear()
    store.close()
    return pack, store, remote, want  # type: ignore[return-value]


def test_streaming_store_bounded_and_exact(built: tuple) -> None:
    pack, store, remote, want = built
    one = max(a.nbytes for a in want.values())
    st = StreamingHintStore.open(store.path.parent, pack.fingerprint, DirectorySource(remote), mac_key=KEY,
                                 cache_bytes=one)
    for _ in range(2):
        for k, a in want.items():
            assert np.array_equal(st.hint(k), a)
            assert st.cached_bytes <= max(one, a.nbytes)
    assert st.fetches == 2 * len(want)  # cyclic access over a cache of one linear: every pass re-reads
    big = StreamingHintStore.open(store.path.parent, pack.fingerprint, DirectorySource(remote), mac_key=KEY,
                                  cache_bytes=1 << 30)
    for _ in range(3):
        for k in want:
            big.hint(k)
    assert big.fetches == len(want)


def test_streaming_store_drives_the_real_backend(built: tuple) -> None:
    pack, store, remote, _ = built
    st = StreamingHintStore.open(store.path.parent, pack.fingerprint, DirectorySource(remote), mac_key=KEY,
                                 cache_bytes=1)
    out = {k: w.shape[0] for k, w in pack.w.items()}
    be = PKSMaskBackend(pack, RingParams.from_name("K32A12C27"), st, RemotePKS(_LocalProvider(pack), pack.fingerprint,
                                                                               out, st.a_seed))
    rng = np.random.default_rng(1)
    for key, w in pack.w.items():
        r = rng.integers(0, 1 << 32, (300, w.shape[1]), dtype=np.int64)
        got = be.matmul(key, torch.from_numpy(r)).numpy()
        assert np.array_equal(got, (r @ w.astype(np.int64).T) & 0xFFFFFFFF)


def test_streaming_store_rejects_tampered_storage_and_meta(built: tuple) -> None:
    pack, store, remote, _ = built
    p = remote / "blk.0.q.npy"
    raw = bytearray(p.read_bytes())
    raw[-5] ^= 1
    p.write_bytes(bytes(raw))
    st = StreamingHintStore.open(store.path.parent, pack.fingerprint, DirectorySource(remote), mac_key=KEY)
    with pytest.raises(IntegrityError):
        st.hint("blk.0.q")
    with pytest.raises(IntegrityError):
        StreamingHintStore.open(store.path.parent, pack.fingerprint, DirectorySource(remote), mac_key=bytes(32))
