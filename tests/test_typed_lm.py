"""Typed generation: fact words only via whole-word copies, verified exactly."""

import torch

from vt import typed_lm as TL
from vt.lm import GPTConfig

SRC = TL.normalize("Ada Lovelace : she wrote notes on the Analytical Engine in 1843 .", {"she"})


def test_normalize_and_render():
    b = TL.normalize("The engine was built . It ran in 1843 .", {"the", "it"})
    assert b == b"the engine was built . it ran in 1843 ."
    assert TL.render(b) == "The engine was built . It ran in 1843 ."
    assert TL.normalize("Ada met <unk> .", set()) == b"Ada met \x02 ."


def test_holeify_keeps_only_supported_facts():
    t = TL.holeify(b"in 1843 Ada met Babbage", SRC)
    assert t == b"in 1843 Ada met \x01"


def test_check_is_exact():
    out = b"notes from 1843 by Ada"
    s = SRC.index(b"1843")
    good = [(out.index(b"1843"), s, 4), (out.index(b"Ada"), SRC.index(b"Ada"), 3)]
    assert TL.check(SRC, out, good)["ok"]
    assert not TL.check(SRC, out, good[:1])["ok"]                            # Ada not covered
    assert not TL.check(SRC, b"notes from 1842 by Ada", good)["ok"]           # altered digit
    assert not TL.check(SRC, b"notes from 184 by Ada", [(11, s, 3), good[1]])["ok"]  # partial word
    assert TL.check(SRC, b"notes from \x01 by \x01", [])["holes"] == 2


def _model():
    torch.manual_seed(0)
    return TL.TypedLM(GPTConfig(block_size=160, n_layer=1, n_head=2, n_embd=32, pos_encoding="alibi"))


def test_vocabulary_path_cannot_emit_fact_bytes():
    m = _model()
    x, y, meta = TL.encode([(SRC, b"in 1843 Ada wrote")], 160)
    lgen, _, _, _ = m.heads(x, meta["cand_id"] >= 0)
    assert float(lgen[..., TL.NO_GEN].exp().max().detach()) < 1e-30


def test_untrained_generations_always_pass_the_check():
    m = _model()
    g = torch.Generator().manual_seed(0)
    for _ in range(40):
        out, tr = m.generate(SRC, max_bytes=60, temperature=1.5, generator=g)
        c = TL.check(SRC, out, tr)
        assert c["ok"], c


def test_training_loss_is_a_proper_nll():
    m = _model()
    x, y, meta = TL.encode([(SRC, TL.holeify(b"in 1843 Ada met Babbage", SRC))], 160)
    tok = m.token_logp(x, y, meta)
    assert torch.isfinite(tok).all() and (tok <= 1e-6).all()
    loss = m.loss(x, y, meta)
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())


def test_citations_and_cosourcing():
    src = b"Ada : she met Babbage in 1833 . she wrote notes in 1843 ."
    out = b"Babbage met Ada in 1843 ."
    tr = [(0, src.index(b"Babbage"), 7), (12, 0, 3), (19, src.index(b"1843"), 4)]
    c = TL.check(src, out, tr)
    assert c["ok"] and c["citations"] == [[1, 2]] and not c["cosourced"]     # mixes two source sentences
    out2 = b"Ada wrote in 1843 . Babbage met her in 1833 ."
    tr2 = [(0, 0, 3), (13, src.index(b"1843"), 4), (20, src.index(b"Babbage"), 7), (39, src.index(b"1833"), 4)]
    c2 = TL.check(src, out2, tr2)
    assert c2["ok"] and c2["citations"] == [[2], [1]] and c2["cosourced"]


def test_cosource_decoding_never_mixes_sentences():
    m = _model()
    src = TL.normalize("Ada : she met Babbage in 1833 . she wrote notes in 1843 on the Engine .", set())
    g = torch.Generator().manual_seed(1)
    for _ in range(40):
        out, tr = m.generate(src, max_bytes=80, temperature=2.0, generator=g, cosource=True)
        c = TL.check(src, out, tr)
        assert c["ok"] and c["cosourced"], (out, c)
