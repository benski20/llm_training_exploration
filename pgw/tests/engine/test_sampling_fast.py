"""Sampler fast paths (numpy argmax, native strict top-k) are bit-identical to the torch path.

``fast_argmax`` replaces ``int(torch.argmax(x))`` and ``fast_topk`` replaces ``torch.topk`` in
:class:`pgw.engine.sampling.Sampler`. Both must give exactly the old answers (token ids, logprob
values, top-logprob lists and their order, RNG consumption) on adversarial inputs: ties, signed
zeros, infinities, NaN, non-contiguous views, other dtypes.
"""

from __future__ import annotations

import struct

import pytest
import torch

from pgw.core import native
from pgw.engine import sampling
from pgw.engine.sampling import Sampler, SamplingParams, fast_argmax, fast_topk


def _torch_topk(x: torch.Tensor, k: int) -> list[tuple[int, float]]:
    top = torch.topk(x, k)
    return [(int(i), float(v)) for v, i in zip(top.values, top.indices, strict=True)]


def _cases(seed: int = 0) -> list[torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    out: list[torch.Tensor] = []
    for n in (1, 2, 3, 7, 64, 1000, 152064):
        base = torch.randn(n, generator=g) * 4.0
        out.append(base.clone())
        out.append((base * 2).round() / 2)  # many ties
        t = base.clone()
        t[torch.randint(0, n, (max(1, n // 50),), generator=g)] = float("-inf")
        out.append(t)
        out.append(torch.zeros(n))
        z = torch.zeros(n)
        z[::2] = -0.0
        out.append(z)
        m = base.clone()
        m[n // 2] = float("inf")
        out.append(m)
        if n > 1:
            m2 = base.clone()
            m2[-1] = m2.max()  # tie for the maximum at the end
            m2[0] = m2.max()
            out.append(m2)
            q = base.clone()
            q[n // 3] = float("nan")
            out.append(q)
            out.append(torch.log_softmax(base.round(), dim=-1))  # ties created by rounding
    return out


def test_fast_argmax_equals_torch() -> None:
    for x in _cases():
        assert fast_argmax(x) == int(torch.argmax(x))
        nc = torch.stack([x, x]).t()[:, 1]  # non-contiguous view
        assert fast_argmax(nc) == int(torch.argmax(nc))
        assert fast_argmax(x.double()) == int(torch.argmax(x.double()))


def test_fast_argmax_signed_zero_first_wins() -> None:
    assert fast_argmax(torch.tensor([-0.0, 0.0, -1.0])) == int(torch.argmax(torch.tensor([-0.0, 0.0, -1.0]))) == 0
    assert fast_argmax(torch.tensor([-1.0, 0.0, -0.0])) == 1
    nan = torch.tensor([1.0, float("nan"), 2.0, float("nan")])
    assert fast_argmax(nan) == int(torch.argmax(nan))


@pytest.mark.parametrize("k", [1, 2, 3, 5, 20])
def test_fast_topk_equals_torch(k: int) -> None:
    for x in _cases(seed=k):
        if k > x.numel():
            continue
        want = _torch_topk(x, k)
        got = fast_topk(x, k)
        assert [i for i, _ in got] == [i for i, _ in want]
        assert [struct.pack("<d", v) for _, v in got] == [struct.pack("<d", v) for _, v in want]  # same bits


@pytest.mark.skipif(not native.native_available("top_k_strict"), reason="pgw_fast.top_k_strict not built")
def test_native_strict_topk_declines_ties_and_nan() -> None:
    import numpy as np

    assert native.fast.top_k_strict(np.array([1.0, 3.0, 2.0], np.float32), 2) == ([1, 2], [3.0, 2.0])
    assert native.fast.top_k_strict(np.array([3.0, 3.0, 2.0], np.float32), 1) is None  # tie at the top
    assert native.fast.top_k_strict(np.array([3.0, 2.0, 2.0], np.float32), 1) == ([0], [3.0])
    assert native.fast.top_k_strict(np.array([3.0, 2.0, 2.0], np.float32), 2) is None  # tie across the boundary
    assert native.fast.top_k_strict(np.array([0.0, -0.0], np.float32), 1) is None
    assert native.fast.top_k_strict(np.array([1.0, np.nan], np.float32), 1) is None
    assert native.fast.top_k_strict(np.array([1.0], np.float32), 2) is None
    strided = np.arange(10, dtype=np.float32)[::2]
    assert native.fast.top_k_strict(strided, 2) == ([4, 3], [8.0, 6.0])


PARAMS = [
    SamplingParams(temperature=0.0, logprobs=2),
    SamplingParams(temperature=0.0, logprobs=0),
    SamplingParams(temperature=0.0, logprobs=20, repetition_penalty=1.3, presence_penalty=0.5),
    SamplingParams(temperature=0.0),
    SamplingParams(temperature=0.8, seed=3, logprobs=5),
    SamplingParams(temperature=1.0, top_k=1, logprobs=1),
    SamplingParams(temperature=0.7, top_k=40, top_p=0.9, seed=11, logprobs=3, frequency_penalty=0.4),
    SamplingParams(temperature=1.3, min_p=0.05, seed=5, logprobs=2),
]


@pytest.mark.parametrize("params", PARAMS)
def test_sampler_matches_the_torch_path(params: SamplingParams, monkeypatch: pytest.MonkeyPatch) -> None:
    """The Sampler with the fast helpers gives the old path's tokens, logprobs, top lists and RNG state."""
    g = torch.Generator().manual_seed(42)
    steps = []
    for i in range(12):
        lg = torch.randn(5000, generator=g) * 3.0
        if i % 3 == 1:
            lg = lg.round()  # ties
        if i % 4 == 2:
            lg[7] = lg.max()  # a tied maximum
        steps.append(lg)

    def run() -> tuple[list[tuple[int, float | None, list[tuple[int, float]] | None]], torch.Tensor]:
        s = Sampler(params, prompt_ids=[1, 2, 3])
        gen: list[int] = []
        out = []
        for lg in steps:
            tok, lp = s.sample(lg, gen)
            out.append((tok, lp, s.last_top_logprobs))
            gen.append(tok)
            out.append((s.propose(lg, gen), None, None))
        return out, s._gen.get_state()

    fast, fast_state = run()
    monkeypatch.setattr(sampling, "fast_argmax", lambda x: int(torch.argmax(x)))
    monkeypatch.setattr(sampling, "fast_topk", _torch_topk)
    ref, ref_state = run()
    assert fast == ref
    if params.seed is not None:  # unseeded samplers draw a fresh seed each
        assert torch.equal(fast_state, ref_state)
