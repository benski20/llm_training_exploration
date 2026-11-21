"""W1: the decode-sized local-linear fast path (native quantize + one two-digit int8 GEMM) is bit-identical
to the reference local product, refuses the same inputs, and falls back when its self-check fails."""

from __future__ import annotations

import pytest
import torch

from pgw.core import RING_K40_A12, IntegrityError
from pgw.core.errors import RingOverflowError
from pgw.core.ring import DEFAULT_RING
from pgw.core.types import LinearId, LinearKind
from pgw.engine import InMemoryPack, LinearCall, LocalComputeClient, MaskedBackend
from pgw.engine import backends as B

pytestmark = pytest.mark.skipif(not B.native_available("trip_prepare_into") or not B.cpu_int8_fast(),
                                reason="needs pgw_fast and exact, fast CPU int8 GEMMs")


def _backend(pack: InMemoryPack, ring=DEFAULT_RING) -> MaskedBackend:  # type: ignore[no-untyped-def]
    return MaskedBackend(pack, None, LocalComputeClient(pack), ring, pack.fingerprint, local_linears="attention")


@pytest.mark.parametrize("rows", [1, 2, 5, 8])
@pytest.mark.parametrize("lid", ["0.qkv", "1.o"])
def test_local_fast_equals_reference(qwen_pack: InMemoryPack, rows: int, lid: str) -> None:
    be = _backend(qwen_pack)
    lin = LinearId.parse(lid)
    n = qwen_pack.load_linear(lin).w8.shape[1]
    gen = torch.Generator().manual_seed(rows)
    for scale in (1.0, 1e-3, 40.0, 1e-30):
        x = torch.randn(rows, n, generator=gen) * scale
        x[0, 3] = 25.0 * scale  # a spike: fewer activation bits for that row on the capped ring
        if rows > 2:
            x[1] = 0.0
        meta, w = be._meta.get(lin), be._local_w.get(lin)
        ref = be._local_reference(x, meta, w, lin)
        got = be._forward_local_one(LinearCall(lin, x))
        assert be._local_fast_checked and all(be._local_fast_checked.values()), "the fast path was not taken"
        assert got.dtype == ref.dtype and got.shape == ref.shape and torch.equal(got, ref)
    be.close()


def test_local_fast_only_for_decode_rows_and_k32(qwen_pack: InMemoryPack) -> None:
    lin = LinearId(0, LinearKind.QKV)
    be = _backend(qwen_pack, RING_K40_A12)
    be.forward_local([LinearCall(lin, torch.randn(1, qwen_pack.arch.hidden_size))])
    assert be._local_fast_on is False and not be._local_fast_checked  # K40: reference path
    be.close()
    be = _backend(qwen_pack)
    be.forward_local([LinearCall(lin, torch.randn(be.local_fast_max_rows + 1, qwen_pack.arch.hidden_size))])
    assert not be._local_fast_checked  # prefill-sized: reference path
    be.close()


def test_local_fast_env_switch(qwen_pack: InMemoryPack, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PGW_LOCAL_FAST", "0")
    be = _backend(qwen_pack)
    be.forward_local([LinearCall(LinearId(0, LinearKind.QKV), torch.randn(1, qwen_pack.arch.hidden_size))])
    assert be._local_fast_on is False
    be.close()


def test_local_fast_self_check_failure_keeps_reference(qwen_pack: InMemoryPack,
                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    be = _backend(qwen_pack)
    real = be._local_fast

    def off(*a: object) -> torch.Tensor:
        y = real(*a)  # type: ignore[arg-type]
        y[0, 0] = torch.nextafter(y[0, 0], torch.tensor(float("inf")))
        return y

    monkeypatch.setattr(be, "_local_fast", off)
    lin = LinearId(0, LinearKind.QKV)
    x = torch.randn(1, qwen_pack.arch.hidden_size)
    got = be._forward_local_one(LinearCall(lin, x))
    assert list(be._local_fast_checked.values()) == [False]
    assert torch.equal(got, be._local_reference(x, be._meta.get(lin), be._local_w.get(lin), lin))
    be.close()


def test_local_fast_refuses_like_the_reference(qwen_pack: InMemoryPack, monkeypatch: pytest.MonkeyPatch) -> None:
    be = _backend(qwen_pack)
    lid = LinearId(0, LinearKind.QKV)
    x = torch.randn(2, qwen_pack.arch.hidden_size)
    be.forward_local([LinearCall(lid, x)])  # the self-check passes first
    assert list(be._local_fast_checked.values()) == [True]
    bound = DEFAULT_RING.output_bound(qwen_pack.arch.hidden_size)
    real = B.int8_digit_gemm

    def too_big(d8: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        y = real(d8, w)
        y[1, 0] = bound + 1  # low digit of row 1
        y[3, 0] = 0  # its high digit
        return y

    monkeypatch.setattr(B, "int8_digit_gemm", too_big)
    with pytest.raises(IntegrityError) as ei:
        be.forward_local([LinearCall(lid, x)])
    assert ei.value.details["bound"] == bound
    monkeypatch.setattr(B, "int8_digit_gemm", real)
    with pytest.raises(RingOverflowError):
        be.forward_local([LinearCall(lid, torch.full((1, qwen_pack.arch.hidden_size), float("nan")))])
    be.close()
