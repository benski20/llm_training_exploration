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
    assert kind in KINDS and set(p) == set(KINDS) | {"text"} and abs(sum(p.values()) - 1) < 1e-3
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
    article = "Summarize the article below.\n\n" + "The council voted on Tuesday to extend the tram line. " * 6
    assert rule(article) == "text"                                  # a task on pasted text
    assert rule("Summarize the plot of Hamlet.") is None            # short: a question, nothing pasted
    assert rule("word " * 320) == "text"                            # very long: text to work on
    assert rule("Who did Prince William marry?\n\n" + "The royal family rebounded in popularity. " * 12) == "text"
    assert rule("Who did Prince William marry?") is None


def test_text_router_sorts_with_a_trained_layer(tmp_path):
    import numpy as np

    from vt.router import TextRouter

    class Embed:                                                    # two directions: "poem" and "number"
        def encode(self, texts, **kw):
            return np.array([[float("poem" in t), float("number" in t)] for t in texts], dtype=np.float32)
    path = tmp_path / "head.npz"
    W = np.array([[0, 0, 0, 5], [0, 5, 0, 0]], dtype=np.float32)     # poem -> chat, number -> math
    np.savez(path, W=W, b=np.zeros(4, np.float32), kinds=np.array(KINDS), embed=np.array("fake"))
    r = TextRouter(str(path), embedder=Embed())
    (k1, p1), (k2, _), (k3, _) = r.route_batch([[{"role": "user", "content": q}] for q in
                                                ("a poem please", "add this number", "def f(x): return x")])
    assert (k1, k2, k3) == ("chat", "math", "code")                 # the rules still decide first
    assert abs(sum(p1.values()) - 1) < 1e-3 and set(p1) == set(KINDS) | {"text"}
