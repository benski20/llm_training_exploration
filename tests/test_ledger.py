"""vt/ledger.py: a verdict on every claim of every reply."""
import numpy as np

from vt.ledger import Ledger, overall, units


class Typer:                                       # facts: sentences naming a year or a capital
    def p_fact(self, texts):
        return np.array([0.9 if any(w in t for w in ("1889", "Paris", "Rome", "capital")) else 0.1 for t in texts])


class Judge:                                       # supports what the passages say, contradicts Rome
    supported = 0.6

    def __init__(self):
        self.asked = []

    def judge3(self, items):
        self.asked += items
        out = []
        for _, claim, passages in items:
            ev = " ".join(passages)
            out.append((0.05, 0.9, 0.05) if "Rome" in claim else
                       (0.9, 0.05, 0.05) if claim.split()[-1].strip(".") in ev else (0.1, 0.1, 0.8))
        return out


def test_units_sort_code_arithmetic_and_sentences():
    us = units("So 12 * 4 = 50.\n```python\ndef f(x)\n    return x\n```\n```js\nlet a = 1\n```\nHope this helps!")
    kinds = [(u["type"], u.get("verdict")) for u in us]
    assert ("code", "does not parse") in kinds and ("code", "not checked") in kinds
    assert ("arithmetic", "wrong") in kinds and (None, None) in kinds        # the last sentence: still to sort


def test_every_claim_gets_a_verdict_whatever_the_reply():
    looked = []
    led = Ledger(Typer(), Judge(), retriever=lambda q: looked.append(q) or ["The tower opened in 1889 in Paris."])
    poem, answer, docs = led.build_batch([
        {"text": "Waves sing softly. The tower rose in Rome in 1889.", "question": "A poem about Paris?"},
        {"text": "It opened in 1889. I hope this helps!", "question": "When did it open?"},
        {"text": "The capital is Paris.", "question": "Summarize.", "documents": ["Paris is the capital."],
         "source_label": "your text"}])
    assert [u["verdict"] for u in poem] == ["not a factual claim", "contradicted"]  # a wrong fact inside a poem
    assert [u["verdict"] for u in answer] == ["supported", "not a factual claim"]
    assert docs[0]["verdict"] == "supported" and not docs[0]["sources"]       # the user's text: no lookup
    assert len(looked) == 2                                                     # one lookup per unsourced fact
    assert overall(poem)["verdict"] == "likely wrong" and overall(answer)["verdict"] == "likely right"


def test_overall_verdicts_cover_every_kind_of_reply():
    assert overall([{"type": "not a claim", "verdict": "not a factual claim", "claim": "Hi"}])["verdict"] == \
        "no factual claims"
    assert overall([{"type": "arithmetic", "verdict": "correct", "claim": "2+2=4"}])["verdict"] == "likely right"
    assert overall([{"type": "code", "verdict": "does not parse", "claim": "def", "basis": "x"}])["verdict"] == \
        "likely wrong"
    unsure = [{"type": "fact", "verdict": "supported", "claim": "a"}, {"type": "fact", "verdict": "no evidence",
                                                                       "claim": "b"}]
    assert overall(unsure)["verdict"] == "uncertain"
    assert overall(unsure[1:])["verdict"] == "unverified"


def test_lookups_and_facts_are_capped():
    led = Ledger(Typer(), Judge(), retriever=lambda q: [], max_lookups=1, max_facts=2)
    out = led.build_batch([{"text": "Paris one. Paris two. Paris three.", "question": "q"}])[0]
    assert [u["verdict"] for u in out].count("not checked") == 2               # one over the facts, one over lookups
    assert "no evidence" in [u["verdict"] for u in out]


def test_the_trained_typer_tells_facts_from_chat():
    from vt.ledger import ClaimTyper
    try:
        t = ClaimTyper()
    except Exception:                                                          # embedding model not available offline
        import pytest
        pytest.skip("embedding model not available")
    p = t.p_fact(["The Berlin Wall fell in 1989.", "I hope this helps!", "Good luck tomorrow!"])
    assert p[0] > 0.5 > max(p[1], p[2])


def test_a_calculation_answer_is_recomputed():
    assert units("17 times 23 is 391.", explicit=False)[0]["verdict"] == "correct"
    assert units("17 times 23 is 401.", explicit=False)[0]["verdict"] == "wrong"
