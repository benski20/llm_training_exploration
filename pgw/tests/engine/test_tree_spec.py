"""Token-tree speculation under strict mode (research/plans/tree-speculation.md).

* tree attention: every node row of a tree pass equals one-row decoding of its root-to-node path, bit for bit
  (torch and native float paths, plain and masked with every linear remote, padding rows), also for a tree
  continued over several passes (``tree_base``), and ``KVCache.keep_path`` leaves the cache as plain decoding;
* strict tree speculation through the generator is bit-identical to plain decoding (greedy and seeded), for
  several public topologies, masked filler rows and padded (``spec_mask_rows``) rows, dummy passes;
* the provider sees exactly R rows per linear on every decode pass and exactly L passes; the drafter's work
  (steps, rows fed per step) is the same every pass.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from pgw.core import DEFAULT_RING
from pgw.core.errors import ConfigError
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
from pgw.engine.generate import BuiltinDrafter, tree_topology
from pgw.engine.model import native_float_available

NATIVE = native_float_available()
LONG = [256, 72, 105, 33, 10, 72, 105, 32, 116, 104, 101, 114, 101]

# (parent, rank) per draft node 1..R-1; node 0 is the pending token
TOPOS = {
    "chain4": ((0, 0), (1, 0), (2, 0)),
    "fork4": ((0, 0), (1, 0), (0, 1)),
    "wide4": ((0, 0), (0, 1), (0, 2)),
    "tree6": ((0, 0), (1, 0), (0, 1), (2, 0), (3, 0)),
    "tree8": ((0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (4, 0), (3, 0)),
}


def _ops(model: TransformerModel, native: bool) -> TransformerModel:
    if native:
        model.set_float_ops("native")
    return model


def _plain(pack: Any, native: bool = False) -> TransformerModel:
    return _ops(TransformerModel(pack, PlainBackend(pack, DEFAULT_RING)), native)


def _masked(pack: Any, native: bool) -> tuple[TransformerModel, LocalComputeClient]:
    client = LocalComputeClient(pack, record=True)
    be = MaskedBackend(pack, None, client, DEFAULT_RING, pack.fingerprint)
    return _ops(TransformerModel(pack, be), native), client


def _paths(parents: list[int], toks: list[int]) -> list[list[int]]:
    out = []
    for i in range(len(parents)):
        p, n = [], i
        while n >= 0:
            p.append(toks[n])
            n = parents[n]
        out.append(p[::-1])
    return out


FLOATS = [False, True] if NATIVE else [False]


@pytest.mark.parametrize("native", FLOATS, ids=lambda x: "native" if x else "torch")
@pytest.mark.parametrize("topo", ["fork4", "wide4", "tree8"])
@pytest.mark.parametrize("masked", [False, True], ids=["plain", "masked"])
def test_tree_rows_equal_path_decoding(qwen_pack: InMemoryPack, native: bool, topo: str, masked: bool) -> None:
    nodes = TOPOS[topo]
    parents = [-1] + [p for p, _ in nodes]
    toks = [7, 11, 13, 17, 19, 23, 29, 31][: len(parents)]
    if masked:
        model, client = _masked(qwen_pack, native)
        res: Any = LocalMaskReservation(qwen_pack, DEFAULT_RING, 64)
    else:
        model, res = _plain(qwen_pack, native), None
    cache = model.new_cache(64)
    model.forward_batch([SequenceInput(LONG, cache, res)])
    n = len(toks)
    lg = model.forward_batch([SequenceInput(toks, cache, res, last_logits=n, tree_parents=parents)])[0]
    ref = _plain(qwen_pack, native)
    for i, path in enumerate(_paths(parents, toks)):
        c = ref.new_cache(64)
        ref.forward_batch([SequenceInput(LONG, c)])
        for t in path:
            one = ref.forward_batch([SequenceInput([t], c)])[0]
        assert torch.equal(lg[i], one.reshape(-1)), f"node {i} differs from decoding its path"
    # keep a path, then decoding continues exactly as plain decoding of that path
    leaf = n - 1
    nodes_path = []
    x = leaf
    while x >= 0:
        nodes_path.append(x)
        x = parents[x]
    nodes_path = nodes_path[::-1]
    base = len(LONG)
    cache.keep_path(base, nodes_path)
    cache.set_length(base + len(nodes_path))
    nxt = model.forward_batch([SequenceInput([3], cache, res)])[0]
    c = ref.new_cache(64)
    ref.forward_batch([SequenceInput(LONG + [toks[j] for j in nodes_path], c)])
    assert torch.equal(nxt.reshape(-1), ref.forward_batch([SequenceInput([3], c)])[0].reshape(-1))
    if masked:
        assert [int(x.shape[0]) for k, x in client.received if k == "0.qkv"] == [len(LONG), n, 1]


@pytest.mark.parametrize("native", FLOATS, ids=lambda x: "native" if x else "torch")
def test_tree_continued_over_passes(qwen_pack: InMemoryPack, native: bool) -> None:
    """A draft model grows a tree level by level (``tree_base``): rows of later passes attend to the right
    nodes of earlier passes."""
    model = _plain(qwen_pack, native)
    cache = model.new_cache(64)
    model.forward_batch([SequenceInput(LONG, cache)])
    base = cache.length
    parents = [-1, -1, 0, 1, 0]
    toks = [5, 9, 11, 13, 17]
    out1 = model.forward_batch([SequenceInput(toks[:2], cache, all_logits=True, tree_parents=parents[:2])])[0]
    out2 = model.forward_batch([SequenceInput(toks[2:], cache, all_logits=True, tree_parents=parents,
                                              tree_base=base)])[0]
    got = list(out1) + list(out2)
    for i, path in enumerate(_paths(parents, toks)):
        c = model.new_cache(64)
        model.forward_batch([SequenceInput(LONG, c)])
        for t in path:
            one = model.forward_batch([SequenceInput([t], c)])[0]
        assert torch.equal(got[i], one.reshape(-1)), i


PARAMS = [
    SamplingParams(max_tokens=18, temperature=0.0, ignore_eos=True, logprobs=2),
    SamplingParams(max_tokens=18, temperature=0.9, top_p=0.95, seed=7, ignore_eos=True, logprobs=0),
]


def _draft(pack: Any, native: bool) -> TransformerModel:
    if native and NativeInt8DraftBackend.available():
        be = NativeInt8DraftBackend(pack, DEFAULT_RING)
        m = TransformerModel(pack, be)
        m.lm_head_fn = native_int8_lm_head(m.lm_head, DEFAULT_RING)
        return _ops(m, True)
    return _plain(pack, native)


class _StepLog:
    def __init__(self, model: TransformerModel):
        self.steps: list[tuple[int, ...]] = []
        self._orig = model.forward_batch
        model.forward_batch = self  # type: ignore[method-assign]

    def __call__(self, seqs: Any, *a: Any, **kw: Any) -> Any:
        self.steps.append(tuple(len(s.input_ids) if not isinstance(s.input_ids, torch.Tensor)
                                else int(s.input_ids.numel()) for s in seqs))
        return self._orig(seqs, *a, **kw)


@pytest.mark.parametrize("native", FLOATS, ids=lambda x: "native" if x else "torch")
@pytest.mark.parametrize("topo", ["fork4", "wide4", "tree6", "tree8"])
@pytest.mark.parametrize("mode", ["filler", "pad", "pad-c2"])
@pytest.mark.parametrize("params", PARAMS, ids=["greedy", "seeded"])
@pytest.mark.parametrize("draft", ["same", "other"])
def test_strict_tree_speculation_is_exact(qwen_pack: InMemoryPack, llama_pack: InMemoryPack,
                                          tiny_tokenizer: ChatTokenizer, native: bool, topo: str, mode: str,
                                          params: SamplingParams, draft: str) -> None:
    nodes = TOPOS[topo]
    rows = len(nodes) + 1
    prompt = LONG * 2
    base = Generator(_plain(qwen_pack, native), tiny_tokenizer).generate(prompt, params)
    model, client = _masked(qwen_pack, native)
    dmodel = _draft(qwen_pack if draft == "same" else llama_pack, native)
    log = _StepLog(dmodel)
    kw: dict[str, Any] = {}
    if mode != "filler":
        kw["spec_mask_rows"] = rows if mode == "pad" else 2
    gen = Generator(model, tiny_tokenizer, spec_rows=rows, spec_strict=True, draft=dmodel, spec_tree=nodes, **kw)
    res = LocalMaskReservation(qwen_pack, DEFAULT_RING, gen.mask_rows_needed(len(prompt), params.max_tokens))
    out = gen.generate(prompt, params, session=res)
    assert out.token_ids == base.token_ids and out.text == base.text
    assert out.finish_reason == base.finish_reason and out.logprobs == base.logprobs
    for lid in ("0.qkv", "1.down"):
        passes = [int(x.shape[0]) for k, x in client.received if k == lid]
        assert passes[0] == len(prompt) and all(n == rows for n in passes[1:])
        assert len(passes) == out.completion_tokens
    # drafter work per pass: one catch-up step, then the topology's level steps with fixed row counts
    levels = log.steps[1:]  # [0] is the prompt prefill
    per = len(tree_topology(nodes).levels) + 1
    assert len(levels) == per * (out.completion_tokens - 1)
    for p in range(out.completion_tokens - 1):
        blk = levels[p * per : (p + 1) * per]
        assert blk[0][0] == rows  # the catch-up step is padded to R rows
        assert [b[0] for b in blk[1:]] == [len(lv) for lv in tree_topology(nodes).levels]
    if draft == "same" and params.greedy and not native:
        assert gen.spec_stats["spec_accepted"] > 0


def test_tree_config_validation(qwen_pack: InMemoryPack, tiny_tokenizer: ChatTokenizer) -> None:
    model = _plain(qwen_pack)
    d = _plain(qwen_pack)
    with pytest.raises(ConfigError):  # R - 1 nodes needed
        Generator(model, tiny_tokenizer, spec_rows=4, spec_strict=True, draft=d, spec_tree=((0, 0), (1, 0)))
    with pytest.raises(ConfigError):  # parent after the node
        Generator(model, tiny_tokenizer, spec_rows=3, spec_strict=True, draft=d, spec_tree=((2, 0), (0, 0)))
    with pytest.raises(ConfigError):  # duplicate (parent, rank)
        Generator(model, tiny_tokenizer, spec_rows=3, spec_strict=True, draft=d, spec_tree=((0, 0), (0, 0)))
    with pytest.raises(ConfigError):  # needs a draft model
        Generator(model, tiny_tokenizer, spec_rows=3, spec_strict=True, lookup=3, spec_tree=((0, 0), (0, 1)))
    with pytest.raises(ConfigError):  # tree needs fixed rows
        Generator(model, tiny_tokenizer, draft=d, spec_tree=((0, 0),))
    assert isinstance(Generator(model, tiny_tokenizer, spec_rows=3, spec_strict=True, draft=d,
                                spec_tree=((0, 0), (0, 1)))._drafter, BuiltinDrafter)


def _head(model: TransformerModel, d: int = 16, seed: int = 0) -> Any:
    from pgw.engine.generate import FeatureHeadDrafter

    g = torch.Generator().manual_seed(seed)
    h = int(model.arch.hidden_size)

    def r(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=g) * 0.1

    w = {"w_in": r(d, 2 * h), "b_in": r(d), "w1": r(2 * d, d), "b1": r(2 * d), "w2": r(d, 2 * d), "b2": r(d),
         "w_out": r(h, d)}
    emb = model.embed_tokens.to(torch.float32)
    return FeatureHeadDrafter(w, emb, list(range(0, int(emb.shape[0]), 2)), target_vocab=int(model.arch.vocab_size))


@pytest.mark.parametrize("topo", ["chain", "tree6", "wide4"])
@pytest.mark.parametrize("params", PARAMS, ids=["greedy", "seeded"])
@pytest.mark.parametrize("native", FLOATS, ids=lambda x: "native" if x else "torch")
def test_feature_head_drafter_strict_exact(qwen_pack: InMemoryPack, tiny_tokenizer: ChatTokenizer, topo: str,
                                           params: SamplingParams, native: bool) -> None:
    """A target-feature head drafter (EAGLE-style, reads the gateway's plaintext final hidden state): exact,
    R rows and L passes for the provider, 1 + levels head steps per pass, a target feature on every job."""
    rows = 4 if topo in ("chain", "wide4") else 6
    nodes = None if topo == "chain" else TOPOS[topo]
    prompt = LONG * 2
    base = Generator(_plain(qwen_pack, native), tiny_tokenizer).generate(prompt, params)
    model, client = _masked(qwen_pack, native)
    head = _head(model)
    seen: list[bool] = []
    orig = head.propose

    def spy(jobs: Any) -> None:
        seen.extend(j.feature is not None for j in jobs)
        orig(jobs)

    head.propose = spy  # type: ignore[method-assign]
    gen = Generator(model, tiny_tokenizer, spec_rows=rows, spec_strict=True, spec_mask_rows=rows, drafter=head,
                    spec_tree=nodes)
    res = LocalMaskReservation(qwen_pack, DEFAULT_RING, gen.mask_rows_needed(len(prompt), params.max_tokens))
    out = gen.generate(prompt, params, session=res)
    assert out.token_ids == base.token_ids and out.text == base.text and out.logprobs == base.logprobs
    passes = [int(x.shape[0]) for k, x in client.received if k == "0.qkv"]
    assert passes[0] == len(prompt) and all(n == rows for n in passes[1:])
    assert len(passes) == out.completion_tokens
    levels = 1 + (len(tree_topology(nodes).levels) if nodes else rows - 2)
    assert head.stats["steps"] == levels * (out.completion_tokens - 1)
    assert seen and all(seen)
    assert head.nbytes_per_step() > 0
