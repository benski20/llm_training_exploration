"""P0 thread scoping: decode float ops on fewer torch threads, only where the bits stay the same."""

from __future__ import annotations

import threading
from typing import Any

import pytest
import torch

from pgw.core import DEFAULT_RING
from pgw.engine import InMemoryPack, PlainBackend
from pgw.engine import lm_head as lm_head_mod
from pgw.engine import model as model_mod
from pgw.engine.model import FloatOps, SequenceInput, TransformerModel, float_threads_self_check
from tests.engine.test_model_hooks import scenarios

needs_omp = pytest.mark.skipif(model_mod._omp_set() is None, reason="no thread-local OpenMP setter here")


def _base() -> int:
    return torch.get_num_threads()


def _scope_target() -> int:
    return 1 if _base() != 1 else 2


def _model(pack: InMemoryPack) -> TransformerModel:
    return TransformerModel(pack, PlainBackend(pack, DEFAULT_RING))


@needs_omp
def test_omp_setter_is_thread_local() -> None:
    """The setter moves only the calling thread's count; the main thread and threads created while it
    is set keep the process default (unlike torch.set_num_threads)."""
    setter = model_mod._omp_set()
    assert setter is not None
    base = _base()
    seen: dict[str, int] = {}

    def worker() -> None:
        torch.get_num_threads()
        setter(_scope_target())
        seen["in"] = torch.get_num_threads()
        child = threading.Thread(target=lambda: seen.__setitem__("child", torch.get_num_threads()))
        child.start()
        child.join()
        setter(base)
        seen["after"] = torch.get_num_threads()

    th = threading.Thread(target=worker)
    th.start()
    th.join()
    assert seen == {"in": _scope_target(), "child": base, "after": base}
    assert _base() == base


@needs_omp
def test_scoped_decode_is_bitwise_equal_and_scoped(qwen_pack: InMemoryPack, llama_pack: InMemoryPack,
                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    """Every scenario (plain/masked, sync/async, decode, verify, lockstep, mixed) gives the same bits
    with the scope on; decode passes ran at the scoped count, prefill at the base count, and the
    count is restored after every pass."""
    ref = scenarios({"qwen": qwen_pack, "llama": llama_pack})
    seen: list[tuple[int, int]] = []
    loading = [False]
    real = model_mod._attend

    def spy(q: torch.Tensor, *a: Any, **k: Any) -> torch.Tensor:
        if not loading[0]:  # not the self-checks at load
            seen.append((int(q.shape[0]), torch.get_num_threads()))
        return real(q, *a, **k)

    real_build = TransformerModel.__init__

    def init(self: TransformerModel, *a: Any, **k: Any) -> None:
        loading[0] = True
        try:
            real_build(self, *a, **k)
            assert self.enable_float_threads(_scope_target())
        finally:
            loading[0] = False
        assert self.float_threads == _scope_target() and self.float_threads_rows == 8  # tiny shapes: all pass

    monkeypatch.setattr(TransformerModel, "__init__", init)
    monkeypatch.setattr(model_mod, "_attend", spy)
    base = _base()
    got = scenarios({"qwen": qwen_pack, "llama": llama_pack})
    for case in ref:
        for k in ref[case]:
            assert torch.equal(ref[case][k], got[case][k]), f"{case}/{k}"
    assert _base() == base
    decode = {n for rows, n in seen if rows == 1}
    assert _scope_target() in decode
    assert {n for rows, n in seen if rows >= 13} == {base}  # prefill blocks (> 8 rows): never scoped


@needs_omp
def test_rows_above_the_checked_count_are_not_scoped(qwen_pack: InMemoryPack,
                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    real_check = model_mod.float_threads_self_check
    monkeypatch.setattr(model_mod, "float_threads_self_check",
                        lambda m, n, rows=1, **k: rows <= 2 and real_check(m, n, rows, **k))
    m = _model(qwen_pack)
    assert m.enable_float_threads(_scope_target())
    assert m.float_threads_rows == 2
    seen: list[tuple[int, int]] = []
    real = model_mod._attend

    def spy(q: torch.Tensor, *a: Any, **k: Any) -> torch.Tensor:
        seen.append((int(q.shape[0]), torch.get_num_threads()))
        return real(q, *a, **k)

    monkeypatch.setattr(model_mod, "_attend", spy)
    cache = m.new_cache(64)
    m.forward_batch([SequenceInput([1, 2, 3, 4, 5], cache)])
    m.forward_batch([SequenceInput([6, 7], cache)])  # 2 rows: scoped
    m.forward_batch([SequenceInput([8, 9, 10], cache, last_logits=3, serial_tail=3)])  # 3 rows: not scoped
    assert [n for _, n in seen[:2]] == [_base()] * 2  # 5-row prefill, 2 layers
    assert [n for _, n in seen[2:4]] == [_scope_target()] * 2
    assert {n for _, n in seen[4:]} == {_base()}


def test_failed_self_check_keeps_todays_threads(qwen_pack: InMemoryPack, monkeypatch: pytest.MonkeyPatch,
                                                caplog: pytest.LogCaptureFixture) -> None:
    monkeypatch.setattr(model_mod, "float_threads_self_check", lambda *a, **k: False)
    m = _model(qwen_pack)
    with caplog.at_level("WARNING", logger="pgw.engine.model"):
        assert not m.enable_float_threads(_scope_target())
    assert m.float_threads is None and m.float_threads_rows == 0
    assert any("keeping" in r.getMessage() for r in caplog.records)
    seen: list[int] = []
    real = model_mod._attend

    def spy(*a: Any, **k: Any) -> torch.Tensor:
        seen.append(torch.get_num_threads())
        return real(*a, **k)

    monkeypatch.setattr(model_mod, "_attend", spy)
    cache = m.new_cache(32)
    m.forward_batch([SequenceInput([1, 2, 3], cache)])
    m.forward_batch([SequenceInput([4], cache)])
    assert set(seen) == {_base()}


@needs_omp
def test_self_check_catches_a_thread_dependent_op(qwen_pack: InMemoryPack) -> None:
    """A float op whose bits depend on the thread count fails the check (and enabling refuses)."""

    class Drifting(FloatOps):
        def silu_mul(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
            return super().silu_mul(gate, up) + torch.get_num_threads() * 1e-3

    m = _model(qwen_pack)
    assert float_threads_self_check(m, _scope_target())
    m.float_ops = Drifting()
    assert not float_threads_self_check(m, _scope_target())
    assert not m.enable_float_threads(_scope_target())
    assert m.float_threads is None


@needs_omp
def test_env_setting_enables_at_load(qwen_pack: InMemoryPack, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PGW_FLOAT_THREADS", str(_scope_target()))
    assert _model(qwen_pack).float_threads == _scope_target()
    monkeypatch.setenv("PGW_FLOAT_THREADS", "0")
    assert _model(qwen_pack).float_threads is None
    monkeypatch.delenv("PGW_FLOAT_THREADS")
    assert _model(qwen_pack).float_threads is None  # default: off (today's threads)


def test_same_count_or_off_is_a_no_op(qwen_pack: InMemoryPack) -> None:
    m = _model(qwen_pack)
    assert m.enable_float_threads(_base()) and m.float_threads is None
    assert m.enable_float_threads(None) and m.float_threads is None


@needs_omp
def test_torch_head_runs_at_the_base_count(qwen_pack: InMemoryPack, monkeypatch: pytest.MonkeyPatch) -> None:
    """Inside a scoped decode pass a torch-computed head (no native kernel) gets the base count back."""
    m = _model(qwen_pack)
    assert m.enable_float_threads(_scope_target())
    monkeypatch.setattr(model_mod, "native_lm_head", lambda: False)
    seen: list[int] = []
    real = lm_head_mod.lm_head_logits

    def spy(h: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        seen.append(torch.get_num_threads())
        return real(h, w)

    monkeypatch.setattr(model_mod, "lm_head_logits", spy)
    cache = m.new_cache(32)
    m.forward_batch([SequenceInput([1, 2, 3], cache)])
    m.forward_batch([SequenceInput([4], cache)])
    assert seen == [_base(), _base()]


@needs_omp
def test_bf16_head_without_native_bf16_kernel_runs_at_the_base_count(qwen_pack: InMemoryPack,
                                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    """A bfloat16 head on a build without ``lm_head_bf16`` falls back to the torch reference inside
    :func:`lm_head_logits`; the scope must give that head the base count back too."""
    m = _model(qwen_pack)
    assert m.enable_float_threads(_scope_target())
    m._head = m._head.to(torch.bfloat16)
    monkeypatch.setattr(model_mod, "native_lm_head", lambda: True)
    real_avail = model_mod.native.native_available
    monkeypatch.setattr(model_mod.native, "native_available",
                        lambda *names: "lm_head_bf16" not in names and real_avail(*names))
    assert not m._head_native()
    seen: list[int] = []
    real_ref = lm_head_mod.lm_head_reference

    def spy(h: torch.Tensor, w: torch.Tensor, *a: Any, **k: Any) -> torch.Tensor:
        seen.append(torch.get_num_threads())
        return real_ref(h, w, *a, **k)

    monkeypatch.setattr(lm_head_mod, "lm_head_reference", spy)
    cache = m.new_cache(32)
    m.forward_batch([SequenceInput([1, 2, 3], cache)])
    m.forward_batch([SequenceInput([4], cache)])
    assert seen == [_base(), _base()]
    m._head = m._head.to(torch.float32)
    assert m._head_native()  # a float32 head always has the native kernel when native_lm_head() is true


@needs_omp
def test_thread_at_another_count_is_not_scoped(qwen_pack: InMemoryPack, monkeypatch: pytest.MonkeyPatch) -> None:
    """A thread running at a count other than the one the self-check compared against keeps its own
    count for the whole pass (the scoped bits were never compared with that count's)."""
    m = _model(qwen_pack)
    assert m.enable_float_threads(_scope_target())
    other = next(c for c in (2, 3, 5) if c not in (_base(), _scope_target()))
    setter = model_mod._omp_set()
    assert setter is not None
    seen: list[int] = []
    real = model_mod._attend

    def spy(*a: Any, **k: Any) -> torch.Tensor:
        seen.append(torch.get_num_threads())
        return real(*a, **k)

    monkeypatch.setattr(model_mod, "_attend", spy)
    errors: list[BaseException] = []

    def work() -> None:
        try:
            torch.get_num_threads()
            setter(other)
            cache = m.new_cache(32)
            m.forward_batch([SequenceInput([1, 2, 3], cache)])
            m.forward_batch([SequenceInput([4], cache)])
            assert getattr(model_mod._scope_tls, "scope", None) is None
        except BaseException as exc:  # re-raised on the test thread
            errors.append(exc)

    th = threading.Thread(target=work)
    th.start()
    th.join()
    assert not errors, errors
    assert seen and set(seen) == {other}
    assert _base() == torch.get_num_threads()


@needs_omp
def test_self_check_sees_the_layers_memory_layout(qwen_pack: InMemoryPack) -> None:
    """The check feeds RoPE the strided column slices the layer does (not fresh contiguous tensors):
    an op whose bits depend on the thread count only for strided inputs is caught from 2 rows on
    (a 1-row slice is contiguous)."""

    class StridedDrift(FloatOps):
        def rope(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
            out = super().rope(x, cos, sin)
            return out if x.is_contiguous() else out + torch.get_num_threads() * 1e-3

    m = _model(qwen_pack)
    m.float_ops = StridedDrift()
    assert float_threads_self_check(m, _scope_target(), 1)
    assert not float_threads_self_check(m, _scope_target(), 2)
    assert m.enable_float_threads(_scope_target())
    assert m.float_threads_rows == 1


@needs_omp
def test_bf16_model_scopes_bitwise_equal(qwen_pack: InMemoryPack) -> None:
    """A bfloat16 model (the casts are real copies there): scoped decode == unscoped, bit for bit."""

    def run(scope: bool) -> list[torch.Tensor]:
        m = TransformerModel(qwen_pack, PlainBackend(qwen_pack, DEFAULT_RING), dtype=torch.bfloat16)
        if scope:
            assert m.enable_float_threads(_scope_target()) and m.float_threads_rows >= 1
        cache = m.new_cache(64)
        out = m.forward_batch([SequenceInput([1, 2, 3, 4, 5], cache)])
        for tok in (6, 7, 8):
            out += m.forward_batch([SequenceInput([tok], cache)])
        out += m.forward_batch([SequenceInput([9, 10], cache, all_logits=True)])
        return out

    ref, got = run(False), run(True)
    assert all(torch.equal(a, b) for a, b in zip(ref, got, strict=True))
