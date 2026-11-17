"""Sampler semantics and tokenizer wrapper (chat template, incremental UTF-8 decoding)."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest
import torch
from pydantic import ValidationError

from pgw.core import ConfigError
from pgw.engine import ChatTokenizer, Sampler, SamplingParams


def test_params_validation() -> None:
    p = SamplingParams(stop="END")
    assert p.stop == ("END",)
    assert SamplingParams(temperature=0).greedy and SamplingParams(top_k=1).greedy
    for bad in ({"max_tokens": 0}, {"temperature": 3.0}, {"top_p": 0.0}, {"stop": [""]}, {"unknown": 1},
                {"stop": [str(i) for i in range(17)]}):
        with pytest.raises(ValidationError):
            SamplingParams(**bad)  # type: ignore[arg-type]


def test_greedy_and_logprobs() -> None:
    logits = torch.tensor([0.1, 2.0, -1.0, 1.5])
    s = Sampler(SamplingParams(temperature=0, logprobs=2))
    tok, lp = s.sample(logits, [])
    assert tok == 1
    assert lp == pytest.approx(float(torch.log_softmax(logits, -1)[1]))
    assert s.last_top_logprobs is not None and [t for t, _ in s.last_top_logprobs] == [1, 3]
    tok, lp = Sampler(SamplingParams(temperature=0)).sample(logits, [])
    assert lp is None


def test_seeded_sampling_is_reproducible() -> None:
    logits = torch.randn(50, generator=torch.Generator().manual_seed(0))
    p = SamplingParams(temperature=1.0, seed=123)
    a = [Sampler(p).sample(logits, [])[0] for _ in range(3)]
    s1, s2 = Sampler(p), Sampler(p)
    seq1 = [s1.sample(logits, [])[0] for _ in range(20)]
    seq2 = [s2.sample(logits, [])[0] for _ in range(20)]
    assert seq1 == seq2 and len(set(a)) == 1
    assert len(set(seq1)) > 1  # actually random


def test_top_k_top_p_min_p_restrict_support() -> None:
    logits = torch.tensor([5.0, 4.0, 3.0, 0.0, -1.0, -2.0])
    for params, allowed in (
        (SamplingParams(temperature=1.0, top_k=2, seed=1), {0, 1}),
        (SamplingParams(temperature=1.0, top_p=0.5, seed=1), {0}),
        (SamplingParams(temperature=1.0, top_p=0.9, seed=1), {0, 1}),
        (SamplingParams(temperature=1.0, min_p=0.2, seed=1), {0, 1}),
    ):
        s = Sampler(params)
        seen = {s.sample(logits, [])[0] for _ in range(300)}
        assert seen <= allowed, (params, seen)
        assert 0 in seen


def test_temperature_distribution() -> None:
    logits = torch.log(torch.tensor([0.7, 0.2, 0.1]))
    s = Sampler(SamplingParams(temperature=1.0, seed=7))
    c = Counter(s.sample(logits, [])[0] for _ in range(3000))
    assert abs(c[0] / 3000 - 0.7) < 0.04 and abs(c[2] / 3000 - 0.1) < 0.03


def test_penalties() -> None:
    logits = torch.tensor([2.0, 1.9, -1.0])
    tok, _ = Sampler(SamplingParams(temperature=0, repetition_penalty=2.0), prompt_ids=[0]).sample(logits, [])
    assert tok == 1  # 2.0 / 2 < 1.9
    tok, _ = Sampler(SamplingParams(temperature=0, presence_penalty=0.5)).sample(logits, [0])
    assert tok == 1
    tok, _ = Sampler(SamplingParams(temperature=0, frequency_penalty=0.05)).sample(logits, [0, 0, 0])
    assert tok == 1
    tok, _ = Sampler(SamplingParams(temperature=0, frequency_penalty=0.01)).sample(logits, [0])
    assert tok == 0
    # negative logits get more negative under repetition penalty
    tok, _ = Sampler(SamplingParams(temperature=0, repetition_penalty=10.0)).sample(
        torch.tensor([-1.0, -1.05]), [0])
    assert tok == 1
    with pytest.raises(ValueError):
        Sampler(SamplingParams()).sample(torch.zeros(2, 2), [])


def test_tokenizer_roundtrip_and_chat(tiny_tokenizer: ChatTokenizer) -> None:
    ids = tiny_tokenizer.encode("héllo", add_special_tokens=False)
    assert ids == list("héllo".encode())
    assert tiny_tokenizer.decode(ids) == "héllo"
    assert 258 in tiny_tokenizer.eos_token_ids
    chat = tiny_tokenizer.apply_chat_template([{"role": "user", "content": "hi"}])
    text = tiny_tokenizer.decode(chat, skip_special_tokens=False)
    assert text == "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
    assert tiny_tokenizer.special_ids >= {256, 258, 259}
    with pytest.raises(ConfigError):
        tiny_tokenizer.apply_chat_template([{"content": "no role"}])


def test_incremental_decoder_utf8(tiny_tokenizer: ChatTokenizer) -> None:
    text = "a€b😀c"
    ids = tiny_tokenizer.encode(text, add_special_tokens=False)
    dec = tiny_tokenizer.incremental_decoder()
    pieces = [dec.push(i) for i in ids]
    assert "".join(pieces) + dec.flush() == text
    assert all("�" not in p for p in pieces)
    assert pieces[1] == "" and pieces[3] == "€"  # euro sign held back until its last byte
    # special tokens are skipped; truncated UTF-8 is flushed as a replacement char
    dec = tiny_tokenizer.incremental_decoder()
    out = dec.push(258) + dec.push(0xE2) + dec.push(0x82)
    assert out == ""
    assert dec.flush() == "�"


def test_tokenizer_errors(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        ChatTokenizer(tmp_path / "missing")
    with pytest.raises(ConfigError):
        ChatTokenizer(tmp_path)  # empty dir: cannot load
    with pytest.raises(ConfigError):
        ChatTokenizer(None)
