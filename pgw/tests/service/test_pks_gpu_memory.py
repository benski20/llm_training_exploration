"""The pks-v3 device key switch in bounded row blocks (``pgw.service.pks_gpu``): bit-identical to the whole call
and to the CPU path, with any block size (ragged last block included). Runs the torch path on CPU."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pgw.hemask import pks
from pgw.hemask.pks import NU, PACK, RANK, D, NR, PKSParams
from pgw.service import pks_gpu


@pytest.mark.parametrize("groups", [2])
def test_keyswitch_blocks_are_bit_identical(groups: int) -> None:
    from pgw.hemask.pks_native import keyswitch as cpu

    par = PKSParams()
    rng = np.random.default_rng(11 + groups)
    m = 3 * PACK
    hs = rng.integers(0, par.q1, (m, NU), dtype=np.uint64)
    ops = np.stack([np.stack([rng.integers(0, p, (RANK, NR), dtype=np.uint64) for p in pks.QP])
                    for _ in range(groups)]).astype(np.uint32)
    y = rng.integers(0, 1 << 56, (m, D * groups), dtype=np.uint64)
    dev = torch.device("cpu")
    whole = pks_gpu.keyswitch(hs, ops, y, dev, work_bytes=1 << 40)
    assert whole.shape == (groups, m // PACK, NR)
    per_block = pks_gpu._block_bytes(groups)
    for blocks in (1, 2):
        got = pks_gpu.keyswitch(hs, ops, y, dev, work_bytes=blocks * per_block)
        assert np.array_equal(got, whole), blocks
    assert np.array_equal(whole, cpu(hs, ops, y))


def test_work_bytes_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PGW_SERVICE_PKS_GPU_WORK_MB", raising=False)
    assert pks_gpu._work_bytes() == pks_gpu.WORK_BYTES
    monkeypatch.setenv("PGW_SERVICE_PKS_GPU_WORK_MB", "64")
    assert pks_gpu._work_bytes() == 64 << 20
    # one 4-group block of 32 rows is ~16 MiB: the default budget takes ~32 blocks (1024 output rows) at once
    assert 8 << 20 < pks_gpu._block_bytes(4) < 32 << 20
