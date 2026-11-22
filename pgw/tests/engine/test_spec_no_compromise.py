"""L3: strict speculation with padding rows and a public mask charge (``spec_mask_rows``).

``research/plans/speculation-no-compromise.md``. What is checked here:

* exactness: tokens, text, finish reason and logprobs equal plain decoding (greedy, seeded sampling,
  penalties, stop strings) through the masked backend, for R in {2, 4} and C in {2, R};
* the provider's view does not depend on the text: per pass the same items with the same row counts
  (prompt, then R rows per linear, exactly L passes), whatever the drafts proposed or the target accepted;
  with mask extents ending mid-pass a padded call is still one item of R rows;
* mask demand is public: the charge targets (``charge_to``) and the store reservations of a growing
  (windowed) reservation are the same for two texts of the same length, and masks per token are C, not R;
* padding rows and dummy passes are Freivalds-checked like real rows (a corrupted dummy fails);
* pipelined blocks split padded calls by wire rows; the windowed reservation's public mode.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from pgw.core import DEFAULT_RING, IntegrityError
from pgw.core.errors import ConfigError, MaskExhaustedError
from pgw.core.types import LinearId, LinearKind
from pgw.engine import (
    ChatTokenizer,
    Generator,
    InMemoryPack,
    LinearCall,
    LocalComputeClient,
    LocalMaskReservation,
    MaskedBackend,
    PlainBackend,
    SamplingParams,
    TransformerModel,
)
from pgw.engine.backends import DUMMY, PAD_ROWS
from pgw.gateway.masks import WindowedReservation

PROMPT = [256, 72, 105, 33, 10, 72, 105]
LONG = [256, 72, 105, 33, 10, 72, 105, 32, 116, 104, 101, 114, 101]

PARAMS = [
    SamplingParams(max_tokens=20, temperature=0.0, ignore_eos=True, logprobs=2),
    SamplingParams(max_tokens=20, temperature=0.9, top_p=0.95, seed=7, ignore_eos=True, logprobs=0),
    SamplingParams(max_tokens=16, temperature=0.0, repetition_penalty=1.3, frequency_penalty=0.4, ignore_eos=True),
    SamplingParams(max_tokens=24, temperature=0.0, stop=("e",)),
]


@pytest.fixture(scope="module")
def other_draft_pack(tmp_path_factory: pytest.TempPathFactory) -> InMemoryPack:
    from tests.conftest import _save_tiny

    d = _save_tiny(tmp_path_factory.mktemp("tiny_draft_l3"), "qwen2", seed=5)
    return InMemoryPack.from_hf(Path(d), name="tiny-draft")


def _plain(pack: Any) -> TransformerModel:
    return TransformerModel(pack, PlainBackend(pack, DEFAULT_RING))


def _masked(pack: Any, **kw: Any) -> tuple[TransformerModel, LocalComputeClient]:
    client = LocalComputeClient(pack, record=True, **kw)
    return TransformerModel(pack, MaskedBackend(pack, None, client, DEFAULT_RING, pack.fingerprint)), client


class _NoiseDrafter:
    """Random proposals (almost never accepted), at a fixed cost per pass."""

    needs_prefill = False

    def __init__(self, vocab: int):
        self.vocab = vocab
        self.g = torch.Generator().manual_seed(3)

    def open(self, prompt: Any, max_len: int) -> Any:
        return {}

    def prefill(self, states: Any, prompts: Any) -> None:  # pragma: no cover - needs_prefill is False
        raise AssertionError("not called")

    def propose(self, jobs: Any) -> None:
        for job in jobs:
            job.out = torch.randint(0, self.vocab, (job.k,), generator=self.g).tolist()

    def close(self, state: Any) -> None:
        pass


def _source(name: str, qwen_pack: Any, other: Any) -> dict[str, Any]:
    if name == "noise":
        return {"drafter": _NoiseDrafter(int(qwen_pack.arch.vocab_size))}
    return {"same": {"draft": _plain(qwen_pack)}, "other": {"draft": _plain(other)}, "lookup": {"lookup": 3},
            "lookup+other": {"draft": _plain(other), "lookup": 3}}[name]


def _wire(client: LocalComputeClient) -> list[tuple[str, int]]:
    """The provider's view as (linear, rows) per item, in order (payload bytes are uniform elsewhere)."""
    return [(lid, int(x.shape[0])) for lid, x in client.received]


class _Charges:
    """A fixed reservation that records the charge targets (what a growing one would ask the store for)."""

    def __init__(self, inner: LocalMaskReservation):
        self.inner = inner
        self.targets: list[int] = []

    def charge_to(self, target: int) -> None:
        self.targets.append(int(target))

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


@pytest.mark.parametrize(("rows", "c"), [(2, 2), (4, 2), (4, 4)])
@pytest.mark.parametrize("source", ["same", "other", "lookup", "lookup+other"])
@pytest.mark.parametrize("params", PARAMS, ids=["greedy", "seeded", "penalties", "stop"])
def test_padded_strict_is_exact_and_fixed(qwen_pack: InMemoryPack, other_draft_pack: InMemoryPack,
                                          tiny_tokenizer: ChatTokenizer, params: SamplingParams, source: str,
                                          rows: int, c: int) -> None:
    prompt = LONG * 2
    base = Generator(_plain(qwen_pack), tiny_tokenizer).generate(prompt, params)
    model, client = _masked(qwen_pack)
    gen = Generator(model, tiny_tokenizer, spec_rows=rows, spec_strict=True, spec_mask_rows=c,
                    **_source(source, qwen_pack, other_draft_pack))
    need = gen.mask_rows_needed(len(prompt), params.max_tokens)
    assert need == len(prompt) + c * (params.max_tokens - 1)
    res = _Charges(LocalMaskReservation(qwen_pack, DEFAULT_RING, need))
    out = gen.generate(prompt, params, session=res)
    assert out.token_ids == base.token_ids and out.text == base.text
    assert out.finish_reason == base.finish_reason and out.logprobs == base.logprobs
    passes = [n for lid, n in _wire(client) if lid == "0.qkv"]
    assert passes[0] == len(prompt) and all(n == rows for n in passes[1:])
    assert len(passes) == out.completion_tokens  # strict: L passes
    # every decode pass charged C on the public schedule; real rows never exceeded the charge
    assert res.targets == [len(prompt) + c * j for j in range(1, out.completion_tokens)]
    assert out.masks_consumed <= len(prompt) + c * max(0, out.completion_tokens - 1)
    assert res.inner.remaining(LinearId(0, LinearKind.QKV)) == need - out.masks_consumed


@pytest.mark.parametrize("params", PARAMS[:2], ids=["greedy", "seeded"])
def test_provider_view_and_charges_do_not_depend_on_drafts(qwen_pack: InMemoryPack, other_draft_pack: InMemoryPack,
                                                           tiny_tokenizer: ChatTokenizer,
                                                           params: SamplingParams) -> None:
    """Same prompt and length, drafters with very different acceptance: identical wire shapes and charges,
    different mask use (the private part)."""
    views, charges, masks, accepted = [], [], [], []
    for source in ("same", "noise", "lookup"):
        model, client = _masked(qwen_pack)
        gen = Generator(model, tiny_tokenizer, spec_rows=4, spec_strict=True, spec_mask_rows=2,
                        **_source(source, qwen_pack, other_draft_pack))
        res = _Charges(LocalMaskReservation(qwen_pack, DEFAULT_RING, gen.mask_rows_needed(len(PROMPT), 20)))
        out = gen.generate(PROMPT, params, session=res)
        views.append(_wire(client))
        charges.append(res.targets)
        masks.append(out.masks_consumed)
        accepted.append(gen.spec_stats["spec_accepted"])
    assert views[0] == views[1] == views[2]
    assert charges[0] == charges[1] == charges[2]
    assert accepted[0] > accepted[1]  # what differs (drafts accepted, so real passes) stays private
    assert masks[0] <= len(PROMPT) + 2 * 19


def test_masks_per_token_is_c_not_r(qwen_pack: InMemoryPack, tiny_tokenizer: ChatTokenizer) -> None:
    params = SamplingParams(max_tokens=24, temperature=0.0, ignore_eos=True)
    used = {}
    for c in (None, 2):
        model, _client = _masked(qwen_pack)
        gen = Generator(model, tiny_tokenizer, draft=_plain(qwen_pack), spec_rows=4, spec_strict=True,
                        spec_mask_rows=c)
        need = gen.mask_rows_needed(len(PROMPT), params.max_tokens)
        res = LocalMaskReservation(qwen_pack, DEFAULT_RING, need)
        out = gen.generate(PROMPT, params, session=res)
        used[c] = (need, out.masks_consumed, gen.spec_stats["spec_accepted"])
    assert used[None][0] == len(PROMPT) + 1 + 23 * 4  # masked filler rows: R per pass
    assert used[None][1] == used[None][0] - 1  # every row of every pass is masked
    assert used[2][0] == len(PROMPT) + 23 * 2  # C per pass
    assert used[2][1] <= used[2][0]
    assert used[2][2] > 0  # the credit still pays for accepted drafts


def test_padded_call_across_extents_is_one_item(qwen_pack: InMemoryPack, tiny_tokenizer: ChatTokenizer) -> None:
    """Mask extents that end mid-pass (where depends on the masks used so far) never split an item."""
    from tests.engine.conftest import TwoExtentReservation

    params = SamplingParams(max_tokens=16, temperature=0.0, ignore_eos=True, logprobs=1)
    base = Generator(_plain(qwen_pack), tiny_tokenizer).generate(PROMPT, params)
    model, client = _masked(qwen_pack)
    gen = Generator(model, tiny_tokenizer, draft=_plain(qwen_pack), spec_rows=4, spec_strict=True, spec_mask_rows=3)
    need = gen.mask_rows_needed(len(PROMPT), params.max_tokens)
    res = TwoExtentReservation(qwen_pack, DEFAULT_RING, len(PROMPT) + 5, need - len(PROMPT) - 5)
    out = gen.generate(PROMPT, params, session=res)
    assert out.token_ids == base.token_ids and out.logprobs == base.logprobs
    assert any(0 < r < 4 for r in res.boundaries_seen[len(model._lids) * 4:])  # a pass did straddle the boundary
    per_pass = [n for lid, n in _wire(client) if lid == "0.qkv"]
    assert per_pass[0] == len(PROMPT) and all(n == 4 for n in per_pass[1:])
    assert len(per_pass) == params.max_tokens
    assert all(n in (len(PROMPT), 4) for _, n in _wire(client))


def test_corrupted_dummy_pass_fails_like_a_real_one(qwen_pack: InMemoryPack, tiny_tokenizer: ChatTokenizer) -> None:
    """A provider that corrupts a pass learns nothing from whether the gateway notices: dummy passes are
    checked like real ones."""
    params = SamplingParams(max_tokens=12, temperature=0.0, ignore_eos=True)
    state = {"items": 0, "hit": -1}

    def tamper(lid: str, y: torch.Tensor) -> torch.Tensor:
        state["items"] += 1
        if state["items"] == state["hit"]:
            y = y.clone()
            y[-1, 0] += 1
        return y

    # count the items of a clean run, then corrupt one item of the last (dummy) pass
    model, _client = _masked(qwen_pack, tamper=tamper)
    gen = Generator(model, tiny_tokenizer, draft=_plain(qwen_pack), spec_rows=4, spec_strict=True, spec_mask_rows=2)
    need = gen.mask_rows_needed(len(PROMPT), params.max_tokens)
    gen.generate(PROMPT, params, session=LocalMaskReservation(qwen_pack, DEFAULT_RING, need))
    assert gen.spec_stats["dummy_passes"] > 0
    total = state["items"]
    state.update(items=0, hit=total - 3)
    model2, _ = _masked(qwen_pack, tamper=tamper)
    gen2 = Generator(model2, tiny_tokenizer, draft=_plain(qwen_pack), spec_rows=4, spec_strict=True,
                     spec_mask_rows=2)
    with pytest.raises(IntegrityError):
        gen2.generate(PROMPT, params, session=LocalMaskReservation(qwen_pack, DEFAULT_RING, need))
        model2.backend.join_checks()  # type: ignore[attr-defined]


def test_dummy_call_is_checked(qwen_pack: InMemoryPack) -> None:
    def tamper(lid: str, y: torch.Tensor) -> torch.Tensor:
        y = y.clone()
        y[0, 0] += 1
        return y

    lid = LinearId(0, LinearKind.QKV)
    n = int(qwen_pack.load_linear(lid).w8.shape[1])
    be = MaskedBackend(qwen_pack, None, LocalComputeClient(qwen_pack, tamper=tamper), DEFAULT_RING,
                       qwen_pack.fingerprint)
    res = LocalMaskReservation(qwen_pack, DEFAULT_RING, 4)
    with pytest.raises(IntegrityError):
        be.forward(LinearCall(lid, torch.randn(3, n), res, {DUMMY: True}))
    assert res.remaining(lid) == 4  # still mask-free


def test_split_blocks_pads_by_wire_rows() -> None:
    lid = LinearId(0, LinearKind.QKV)
    x = torch.randn(3, 4)
    plain_parts, _ = TransformerModel._split_blocks([LinearCall(lid, torch.randn(8, 4))], 2)
    parts, where = TransformerModel._split_blocks([LinearCall(lid, x, None, {PAD_ROWS: 5})], 2)
    assert [int(b[0].x.shape[0]) + b[0].meta.get(PAD_ROWS, 0) for b in parts] == \
        [int(b[0].x.shape[0]) for b in plain_parts] == [4, 4]
    assert [int(b[0].x.shape[0]) for b in parts] == [3, 0] and [b[0].meta[PAD_ROWS] for b in parts] == [1, 4]
    assert torch.equal(parts[0][0].x, x) and where == [[0], [0]]


def test_model_padding_rows_keep_real_rows_exact(qwen_pack: InMemoryPack) -> None:
    """Model level: padded rows take no masks and the real rows' logits equal the unpadded pass's."""
    from pgw.engine.model import SequenceInput

    model, client = _masked(qwen_pack)
    assert model.supports_pad_rows
    res = LocalMaskReservation(qwen_pack, DEFAULT_RING, 32)
    ref = _plain(qwen_pack)
    c1, c2 = model.new_cache(32), ref.new_cache(32)
    model.forward_batch([SequenceInput(PROMPT, c1, res)])
    ref.forward_batch([SequenceInput(PROMPT, c2)])
    ids = [5, 9, 11, 13]
    lg = model.forward_batch([SequenceInput(ids, c1, res, last_logits=4, serial_tail=3, pad_rows=2)])[0]
    lr = ref.forward_batch([SequenceInput(ids, c2, None, last_logits=4, serial_tail=3)])[0]
    assert torch.equal(lg[0], lr[0]) and torch.equal(lg[1], lr[1])
    lid = LinearId(0, LinearKind.QKV)
    assert res.remaining(lid) == 32 - len(PROMPT) - 2
    assert [n for k, n in _wire(client) if k == "0.qkv"] == [len(PROMPT), 4]
    with pytest.raises(Exception, match="pad_rows"):
        model.forward_batch([SequenceInput([1], c1, res, pad_rows=2)])


def test_config_validation(qwen_pack: InMemoryPack, tiny_tokenizer: ChatTokenizer) -> None:
    target = _plain(qwen_pack)
    with pytest.raises(ConfigError, match="needs spec_rows"):
        Generator(target, tiny_tokenizer, lookup=3, spec_mask_rows=2)
    with pytest.raises(ConfigError, match=r"\[1, spec_rows\]"):
        Generator(target, tiny_tokenizer, lookup=3, spec_rows=3, spec_strict=True, spec_mask_rows=4)
    target.schedule = object()  # type: ignore[assignment]
    with pytest.raises(ConfigError, match="padding rows"):
        Generator(target, tiny_tokenizer, lookup=3, spec_rows=3, spec_strict=True, spec_mask_rows=2)
    target.schedule = None
    gen = Generator(target, tiny_tokenizer, lookup=3, spec_rows=3, spec_strict=True)
    assert not gen._pad_ok and gen.spec_mask_rows is None  # default: masked filler rows, as before


# ---- windowed reservation, public schedule --------------------------------------------------------
class _Store:
    def __init__(self, pack: Any):
        self.pack = pack
        self.calls: list[int] = []

    def reserve(self, n: int) -> LocalMaskReservation:
        self.calls.append(n)
        return LocalMaskReservation(self.pack, DEFAULT_RING, n, seed=bytes([len(self.calls)]) * 32)


def test_windowed_public_mode(qwen_pack: InMemoryPack) -> None:
    lid = LinearId(0, LinearKind.QKV)
    store = _Store(qwen_pack)
    w = WindowedReservation(store.reserve(10), store.reserve, cap=40, window=8, public=True)
    w.take(lid, 10)
    with pytest.raises(MaskExhaustedError):
        w.ensure(lid, 1)  # never on demand
    assert store.calls == [10]
    w.charge_to(11)
    assert store.calls == [10, 8] and w.slots == 18
    w.charge_to(12)  # already held
    assert store.calls == [10, 8]
    w.charge_to(30)  # missing 12 > window
    assert store.calls == [10, 8, 12] and w.slots == 30
    w.charge_to(99)  # capped
    assert w.slots == 40
    w.ensure(lid, 5)
    w.take(lid, 5)
    w.ensure(lid, 1)  # step start: no retire in public mode
    assert w.retired == 0
    w.close()


@pytest.mark.parametrize("params", PARAMS[:2], ids=["greedy", "seeded"])
def test_windowed_store_demand_is_public(qwen_pack: InMemoryPack, other_draft_pack: InMemoryPack,
                                         tiny_tokenizer: ChatTokenizer, params: SamplingParams) -> None:
    """Through a growing reservation: the store is asked for the same slot counts at the same passes
    whatever the drafts did."""
    logs = []
    for source in ("same", "noise"):
        model, client = _masked(qwen_pack)
        gen = Generator(model, tiny_tokenizer, spec_rows=4, spec_strict=True, spec_mask_rows=2,
                        **_source(source, qwen_pack, other_draft_pack))
        cap = gen.mask_rows_needed(len(PROMPT), params.max_tokens)
        store = _Store(qwen_pack)
        log: list[tuple[int, int]] = []

        def reserve(n: int, store: _Store = store, log: list[tuple[int, int]] = log,
                    client: LocalComputeClient = client) -> Any:
            log.append((len(client.received), n))  # (wire items so far, slots): when and how much
            return store.reserve(n)

        w = WindowedReservation(store.reserve(len(PROMPT) + 4), reserve, cap=cap, window=4, public=True)
        out = gen.generate(PROMPT, params, session=w)
        assert out.completion_tokens == params.max_tokens
        logs.append((log, _wire(client)))
        w.close()
    assert logs[0] == logs[1]
    assert len(logs[0][0]) > 1


@pytest.mark.parametrize(("c", "credit"), [(1.5, None), (4, 5), (3, 0)])
def test_fractional_charge_and_credit_cap(qwen_pack: InMemoryPack, tiny_tokenizer: ChatTokenizer, c: float,
                                          credit: int | None) -> None:
    """Public schedule ``prompt + j + min(B, floor((C - 1) * j))``: exact outputs, real rows within it."""
    import math

    params = SamplingParams(max_tokens=18, temperature=0.8, seed=11, ignore_eos=True, logprobs=1)
    base = Generator(_plain(qwen_pack), tiny_tokenizer).generate(PROMPT, params)
    model, client = _masked(qwen_pack)
    gen = Generator(model, tiny_tokenizer, draft=_plain(qwen_pack), spec_rows=4, spec_strict=True,
                    spec_mask_rows=c, spec_mask_credit=credit)

    def sched(j: int) -> int:
        extra = math.floor((c - 1) * j)
        return len(PROMPT) + j + (extra if credit is None else min(credit, extra))

    need = gen.mask_rows_needed(len(PROMPT), params.max_tokens)
    assert need == sched(params.max_tokens - 1)
    res = _Charges(LocalMaskReservation(qwen_pack, DEFAULT_RING, need))
    out = gen.generate(PROMPT, params, session=res)
    assert out.token_ids == base.token_ids and out.logprobs == base.logprobs
    assert res.targets == [sched(j) for j in range(1, params.max_tokens)]
    assert out.masks_consumed <= need
    assert all(n in (len(PROMPT), 4) for _, n in _wire(client))
    if credit == 0:
        assert gen.spec_stats["spec_rows"] == 0  # no credit: no drafts, still R rows per pass


def test_spec_mask_credit_validation(qwen_pack: InMemoryPack, tiny_tokenizer: ChatTokenizer) -> None:
    with pytest.raises(ConfigError, match="needs spec_mask_rows"):
        Generator(_plain(qwen_pack), tiny_tokenizer, lookup=3, spec_rows=3, spec_strict=True, spec_mask_credit=4)
