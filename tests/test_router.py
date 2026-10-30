"""vt/router.py: one forward pass sorts a message into information / math / code / chat."""
import pytest

from vt.router import KINDS, Router


@pytest.fixture(scope="module")
def small():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    name = "Qwen/Qwen2.5-0.5B-Instruct"
    try:
        tok = AutoTokenizer.from_pretrained(name)
        return AutoModelForCausalLM.from_pretrained(name).eval(), tok
    except OSError:
        pytest.skip("small model not available offline")


def test_routes_are_probabilities_over_the_four_kinds(small):
    r = Router(*small)
    kind, p = r.route([{"role": "user", "content": "Write a short poem about autumn leaves."}])
    assert kind in KINDS and set(p) == set(KINDS) and abs(sum(p.values()) - 1) < 1e-3
    batch = r.route_batch([[{"role": "user", "content": q}] for q in
                           ("What is 17 times 23?", "Write a Python function that reverses a list.")])
    assert len(batch) == 2


def test_unambiguous_signals_decide_without_the_model():
    from vt.router import rule
    assert rule("Fix this: for i in range(10) print(i)") == "code"
    assert rule("Write a haiku about rain.") == "chat"
    assert rule("Should I learn guitar or piano?") == "chat"
    assert rule("What year did the Berlin Wall fall?") is None
    assert rule("Explain how vaccines work.") is None
