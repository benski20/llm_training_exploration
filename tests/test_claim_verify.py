"""Claim verification against evidence (vt/claim_verify.py) with the small local chat model."""
import pytest

from vt.claim_verify import ClaimVerifier, check_worthy, sentences


def test_sentences_and_check_worthiness():
    s = sentences("*The Castle of Otranto* was written in 1764 by **Horace Walpole**. It is great.\n- Hope this helps!")
    assert s[0].startswith("The Castle of Otranto") and [check_worthy(x) for x in s] == [True, False, False]


@pytest.fixture(scope="module")
def small():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    name = "Qwen/Qwen2.5-0.5B-Instruct"
    try:
        tok = AutoTokenizer.from_pretrained(name)
        return AutoModelForCausalLM.from_pretrained(name).eval(), tok
    except OSError:
        pytest.skip("small model not available offline")


def test_verifier_tells_supported_from_unsupported_and_no_evidence(small):
    model, tok = small
    ev = ["Eiffel Tower (Wikipedia): The tower is named after the engineer Gustave Eiffel, whose company "
          "designed and built it between 1887 and 1889."]

    def retriever(q):
        return [] if "Atlantis" in q else ev
    v = ClaimVerifier(model, tok, retriever)
    got = v.verify("Who designed the Eiffel Tower?", "The Eiffel Tower was designed by Gustave Eiffel's company. "
                   "Le Corbusier designed the Eiffel Tower. The capital of Atlantis is Poseidonia.")
    assert [g["verdict"] for g in got][0] == "supported" and got[0]["sources"] == ["Eiffel Tower"]
    assert got[1]["p_supported"] < got[0]["p_supported"] and got[1]["verdict"] != "supported"
    assert got[2]["verdict"] == "no evidence"
    b = v.judge([("q", "Gustave Eiffel's company built it.", ev), ("q", "Le Corbusier built it.", ev)])
    assert len(b) == 2 and b[0] > b[1]                       # batched (left-padded) agrees
