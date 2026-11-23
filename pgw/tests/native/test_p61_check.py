"""``pgw_fast.p61_check`` (fused2's range + Freivalds check over F_(2^61-1)) gives the numpy path's verdicts."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pgw.engine.fused2 import P61, dot_mod_p, matmul_mod_p

fast = pytest.importorskip("pgw_fast")
pytestmark = pytest.mark.skipif(not hasattr(fast, "p61_check"), reason="pgw_fast without p61_check")


@pytest.mark.parametrize("m,n,t,c", [(1, 1, 1, 1), (37, 53, 1, 1), (300, 129, 3, 2), (4097, 700, 2, 1)])
def test_native_matches_numpy(m: int, n: int, t: int, c: int) -> None:
    rng = np.random.default_rng(m + n)
    w8 = torch.from_numpy(rng.integers(-128, 128, (m, n), dtype=np.int64).astype(np.int8))
    x = rng.integers(-(1 << 12), 1 << 12, (t, n))
    off = rng.integers(-(1 << 45), 1 << 45, (t, m))
    y = off + x @ w8.numpy().astype(np.int64).T
    u = rng.integers(0, P61, (c, m))
    u[:, 0] = P61 - 1
    uw = matmul_mod_p(u, w8)
    bound = 1 << 53
    assert fast.p61_check(y, off, u, x, uw, bound) == 0
    assert fast.p61_check(y - off, None, u, x, uw, bound) == 0
    for idx, delta in ((0, 1), (m - 1, -(1 << 30)), (m // 2, 12345)):
        bad = y.copy()
        bad[-1, idx] += delta
        want = int(not np.array_equal(dot_mod_p(bad - off, u), dot_mod_p(x, uw)))
        assert fast.p61_check(bad, off, u, x, uw, bound) == want == 1
    big = y.copy()
    big[0, 0] = bound + 1
    assert fast.p61_check(big, off, u, x, uw, bound) == 2
    alias = y.copy()
    alias[0, 0] += P61  # invisible mod p: the range check is what refuses it (as in the numpy path)
    assert fast.p61_check(alias, off, u, x, uw, bound) == 2
    with pytest.raises(ValueError):
        fast.p61_check(y, off, u + P61, x, uw, bound)  # check vectors outside [0, p)
    with pytest.raises(ValueError):
        fast.p61_check(y, off[:, :-1] if m > 1 else off[:0], u, x, uw, bound)
