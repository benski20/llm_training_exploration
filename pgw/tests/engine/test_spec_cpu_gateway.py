"""Strict speculation on a CPU-only, weight-light gateway (``local_linears = off``, ``float_ops = native``).

What is checked here (research/plans/strict-speculation-cpu-gateway.md):

* verify rows of a pass equal single-row decoding bit for bit with the native float path and every linear
  remote (masked), with and without padding rows, for R in {2, 4, 8};
* padded strict speculation (``spec_mask_rows``) through the masked backend with every linear remote and the
  native float path is bit-identical to plain decoding (tokens, text, logprobs; greedy and seeded), for a
  draft model on the native int8 draft backend, the pipelined drafter and prompt lookup; the provider sees
  R rows per linear on every decode pass and exactly L passes;
* the native int8 draft backend gives the same bits for any thread count;
* the pipelined drafter's work per pass does not depend on whether its hypothesis held (same number of
  foreground and background draft steps, same rows fed), and it does hit.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import pytest
import torch

from pgw.core import DEFAULT_RING
from pgw.engine import (
    ChatTokenizer,
    Generator,
    InMemoryPack,
    LocalComputeClient,
    LocalMaskReservation,
    MaskedBackend,
    PlainBackend,
    SamplingParams,
    SequenceInput,
    TransformerModel,
)
from pgw.engine.backends import NativeInt8DraftBackend, native_int8_lm_head
from pgw.engine.generate import PipelinedDrafter
from pgw.engine.model import native_float_available

pytestmark = pytest.mark.skipif(not (native_float_available() and NativeInt8DraftBackend.available()),
                                reason="pgw_fast without the native float path or local_linear_into")

LONG = [256, 72, 105, 33, 10, 72, 105, 32, 116, 104, 101, 114, 101]

PARAMS = [
    SamplingParams(max_tokens=20, temperature=0.0, ignore_eos=True, logprobs=2),
    SamplingParams(max_tokens=20, temperature=0.9, top_p=0.95, seed=7, ignore_eos=True, logprobs=0),
]


@pytest.fixture(scope="module")
def other_draft_pack(tmp_path_factory: pytest.TempPathFactory) -> InMemoryPack:
    from tests.conftest import _save_tiny

    d = _save_tiny(tmp_path_factory.mktemp("tiny_draft_cpu_gw"), "qwen2", seed=5)
    return InMemoryPack.from_hf(Path(d), name="tiny-draft")


def _native(model: TransformerModel) -> TransformerModel:
    model.set_float_ops("native")
    return model


def _plain(pack: Any) -> TransformerModel:
    return _native(TransformerModel(pack, PlainBackend(pack, DEFAULT_RING)))


def _cpu_gateway(pack: Any) -> tuple[TransformerModel, LocalComputeClient]:
    """Every linear remote (no local linears: the gateway holds no int8 weights), native float ops."""
    client = LocalComputeClient(pack, record=True)
    be = MaskedBackend(pack, None, client, DEFAULT_RING, pack.fingerprint)
    assert not any(be.local_linear(lid) for lid in [s.id for s in pack.linear_specs()])
    return _native(TransformerModel(pack, be)), client


def _draft(pack: Any, threads: int = 0) -> TransformerModel:
    be = NativeInt8DraftBackend(pack, DEFAULT_RING, threads=threads)
    model = TransformerModel(pack, be)
    model.lm_head_fn = native_int8_lm_head(model.lm_head, DEFAULT_RING, threads=threads)
    return _native(model)


@pytest.mark.parametrize("rows", [2, 4, 8])
@pytest.mark.parametrize("pad", [0, 1])
def test_native_verify_rows_equal_single_row_decode(qwen_pack: InMemoryPack, rows: int, pad: int) -> None:
    """A verify pass of R rows (serial tail, padding rows at the end) gives each real row the logits that
    decoding the same tokens one row at a time gives, bit for bit: masked, all linears remote, native ops."""
    if pad >= rows:
        pytest.skip("needs a real row")
    model, client = _cpu_gateway(qwen_pack)
    prompt = LONG
    ids = [5, 9, 11, 13, 17, 19, 23, 29][:rows]
    real = rows - pad
    res = LocalMaskReservation(qwen_pack, DEFAULT_RING, len(prompt) + 2 * rows + 4)
    c1 = model.new_cache(64)
    model.forward_batch([SequenceInput(prompt, c1, res)])
    lg = model.forward_batch([SequenceInput(ids, c1, res, last_logits=rows, serial_tail=rows - 1,
                                            pad_rows=pad, fixed_item=True)])[0]
    ref, _ = _cpu_gateway(qwen_pack)
    res2 = LocalMaskReservation(qwen_pack, DEFAULT_RING, len(prompt) + rows + 4)
    c2 = ref.new_cache(64)
    ref.forward_batch([SequenceInput(prompt, c2, res2)])
    for j in range(real):
        one = ref.forward_batch([SequenceInput([ids[j]], c2, res2)])[0]
        assert torch.equal(lg[j], one.reshape(-1)), f"row {j} of {rows} differs from single-row decoding"
    # the same against the plain (unmasked) reference with the native path
    plain = _plain(qwen_pack)
    c3 = plain.new_cache(64)
    plain.forward_batch([SequenceInput(prompt, c3)])
    for j in range(real):
        one = plain.forward_batch([SequenceInput([ids[j]], c3)])[0]
        assert torch.equal(lg[j], one.reshape(-1))
    assert [n for k, x in client.received if k == "0.qkv" for n in [int(x.shape[0])]] == [len(prompt), rows]


def _source(name: str, target: Any, other: Any) -> dict[str, Any]:
    if name == "native-int8-same":
        return {"draft": _draft(target)}
    if name == "native-int8-other":
        return {"draft": _draft(other)}
    if name == "pipelined-m0":
        return {"drafter": PipelinedDrafter(_draft(target), post_steps=0, target_vocab=target.arch.vocab_size)}
    if name == "pipelined-m1-other":
        return {"drafter": PipelinedDrafter(_draft(other), post_steps=1, target_vocab=target.arch.vocab_size)}
    return {"lookup": 3}


@pytest.mark.parametrize(("rows", "c"), [(4, 4), (4, 2), (2, 1.5)])
@pytest.mark.parametrize("source", ["native-int8-same", "native-int8-other", "pipelined-m0",
                                    "pipelined-m1-other", "lookup"])
@pytest.mark.parametrize("params", PARAMS, ids=["greedy", "seeded"])
def test_cpu_gateway_strict_speculation_is_exact(qwen_pack: InMemoryPack, other_draft_pack: InMemoryPack,
                                                 tiny_tokenizer: ChatTokenizer, params: SamplingParams,
                                                 source: str, rows: int, c: float) -> None:
    prompt = LONG * 2
    base = Generator(_plain(qwen_pack), tiny_tokenizer).generate(prompt, params)
    model, client = _cpu_gateway(qwen_pack)
    gen = Generator(model, tiny_tokenizer, spec_rows=rows, spec_strict=True, spec_mask_rows=c,
                    **_source(source, qwen_pack, other_draft_pack))
    res = LocalMaskReservation(qwen_pack, DEFAULT_RING, gen.mask_rows_needed(len(prompt), params.max_tokens))
    out = gen.generate(prompt, params, session=res)
    assert out.token_ids == base.token_ids and out.text == base.text
    assert out.finish_reason == base.finish_reason and out.logprobs == base.logprobs
    for lid in ("0.qkv", "1.down"):
        passes = [int(x.shape[0]) for k, x in client.received if k == lid]
        assert passes[0] == len(prompt) and all(n == rows for n in passes[1:])
        assert len(passes) == out.completion_tokens  # strict: L passes, whatever was accepted
    if source.startswith("pipelined"):  # the block ran k + 1 steps on every call
        gen._drafter._wait()
        st = gen._drafter.stats
        assert st["bg_steps"] == rows * st["calls"]


def test_native_int8_draft_same_bits_any_thread_count(qwen_pack: InMemoryPack) -> None:
    outs = []
    for threads in (1, 0):
        m = _draft(qwen_pack, threads=threads)
        cache = m.new_cache(64)
        outs.append(m.forward_batch([SequenceInput(LONG * 2, cache, all_logits=True)])[0])
        outs.append(m.forward_batch([SequenceInput([5, 9, 11], cache, all_logits=True)])[0])
    assert torch.equal(outs[0], outs[2]) and torch.equal(outs[1], outs[3])


class _StepLog:
    """Wraps a draft model's forward_batch: (thread kind, rows fed per sequence) per step."""

    def __init__(self, model: TransformerModel):
        self.model = model
        self.steps: list[tuple[str, tuple[int, ...]]] = []
        self._orig = model.forward_batch
        model.forward_batch = self  # type: ignore[method-assign]

    def __call__(self, seqs: Any, *a: Any, **kw: Any) -> Any:
        bg = threading.current_thread().name == "pgw-draft-block"
        self.steps.append(("bg" if bg else "fg", tuple(len(s.input_ids) for s in seqs)))
        return self._orig(seqs, *a, **kw)


@pytest.mark.parametrize("m", [0, 1, 2])
def test_pipelined_drafter_work_does_not_depend_on_hits(qwen_pack: InMemoryPack, other_draft_pack: InMemoryPack,
                                                        tiny_tokenizer: ChatTokenizer, m: int) -> None:
    """Same draft steps per pass (m foreground, k + 1 background) whether the hypothesis held or not."""
    rows = 4
    k = rows - 1
    params = PARAMS[0]
    prompt = LONG * 2
    base = Generator(_plain(qwen_pack), tiny_tokenizer).generate(prompt, params)
    logs = {}
    stats = {}
    for name, dpack in (("same", qwen_pack), ("other", other_draft_pack)):
        draft = _draft(dpack)
        log = _StepLog(draft)
        drafter = PipelinedDrafter(draft, post_steps=m, target_vocab=qwen_pack.arch.vocab_size, background=False)
        model, _client = _cpu_gateway(qwen_pack)
        gen = Generator(model, tiny_tokenizer, spec_rows=rows, spec_strict=True, spec_mask_rows=rows,
                        drafter=drafter)
        res = LocalMaskReservation(qwen_pack, DEFAULT_RING, gen.mask_rows_needed(len(prompt), params.max_tokens))
        out = gen.generate(prompt, params, session=res)
        assert out.token_ids == base.token_ids and out.logprobs == base.logprobs
        logs[name] = log.steps
        stats[name] = dict(drafter.stats)
    # the self-draft hits (its int8 guesses match the target often); hits and misses both occur somewhere
    assert stats["same"]["hits"] > 0
    assert stats["other"]["misses"] > 0
    for name, steps in logs.items():
        st = stats[name]
        calls = int(st["calls"])
        assert calls == int(st["hits"] + st["misses"])
        # foreground: exactly m steps per propose call; background: exactly k + 1 per call
        assert int(st["steps"]) == m * calls and int(st["bg_steps"]) == (k + 1) * calls
        assert len(steps) == 1 + (m + k + 1) * calls  # + the prefill
        # every decode-time step feeds one sequence with at most 2 rows (the catch-up residual)
        assert all(r[0] <= 2 for kind, r in steps[1:])
    assert int(stats["same"]["calls"]) == int(stats["other"]["calls"])


def test_pipelined_drafter_background_thread_exact(qwen_pack: InMemoryPack, tiny_tokenizer: ChatTokenizer) -> None:
    """The real background thread (the block runs while the pass is in flight): same outputs."""
    params = PARAMS[0]
    prompt = LONG * 2
    base = Generator(_plain(qwen_pack), tiny_tokenizer).generate(prompt, params)
    drafter = PipelinedDrafter(_draft(qwen_pack), post_steps=0, target_vocab=qwen_pack.arch.vocab_size)
    model, _ = _cpu_gateway(qwen_pack)
    gen = Generator(model, tiny_tokenizer, spec_rows=4, spec_strict=True, spec_mask_rows=4, spec_mask_credit=40,
                    drafter=drafter)
    res = LocalMaskReservation(qwen_pack, DEFAULT_RING, gen.mask_rows_needed(len(prompt), params.max_tokens))
    out = gen.generate(prompt, params, session=res)
    assert out.token_ids == base.token_ids and out.logprobs == base.logprobs
    drafter._wait()
    assert drafter.stats["hits"] > 0 and drafter.stats["bg_steps"] == 4 * drafter.stats["calls"]
