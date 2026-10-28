"""Reasoning models in doubt-aware generation: a <think> block is kept apart from the reply (not scored,
not counted in max_new_tokens, closed by its own budget), streamed apart, and returned as "reasoning".
A tiny random Qwen3 with the real Qwen3 tokenizer; needs transformers >= 4.51."""
import numpy as np
import pytest
import torch

transformers = pytest.importorskip("transformers")
if not hasattr(transformers, "Qwen3Config"):
    pytest.skip("transformers without Qwen3", allow_module_level=True)

from vt.claims import prompt_of  # noqa: E402
from vt.guided import DoubtAwareGenerator  # noqa: E402

R1_TEMPLATE = ("{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
               "<|im_start|>assistant\n<think>\n")


class Zero:
    def logit(self, x):
        return np.zeros(len(x))


@pytest.fixture(scope="module")
def tok():
    try:
        return transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    except Exception as e:                                     # offline
        pytest.skip(f"no Qwen3 tokenizer: {e}")


@pytest.fixture(scope="module")
def model(tok):
    torch.manual_seed(0)
    cfg = transformers.Qwen3Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=3,
                                   num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=512)
    return transformers.AutoModelForCausalLM.from_config(cfg).eval()


def gen(model, tok, **kw):
    return DoubtAwareGenerator(model, tok, Zero(), 1, threshold=0.0, risk_probe=Zero(), risk_layer=1,
                               risk_threshold=0.9, **kw)


def check_split(tok, g, r, budget, max_new):
    ids = r["ids"]
    assert g.think_open not in ids and g.think_close not in ids          # the reply holds no thinking
    close = [g.think_close] + tok("\n\n", add_special_tokens=False).input_ids
    assert r["prefix_ids"][-len(close):] == close
    assert len(r["prefix_ids"]) - len(close) <= budget + 1 and len(ids) <= max_new + 2
    assert r["reasoning"] == tok.decode([t for t in r["prefix_ids"][:-len(close)] if t != g.think_open],
                                        skip_special_tokens=True).strip()


def test_prompt_switch(tok):
    m = [{"role": "user", "content": "hi"}]
    assert prompt_of(tok, m, False).endswith("<think>\n\n</think>\n\n")
    assert not prompt_of(tok, m, True).endswith("</think>\n\n")


def test_r1_style_prompt_thinks_then_answers(model, tok):
    tok2 = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    tok2.chat_template = R1_TEMPLATE                                     # the prompt opens the block itself
    g = gen(model, tok2, think_budget=6)
    texts, reasons = {}, {}
    r = g.generate_batch([{"messages": [{"role": "user", "content": "Who wrote Hamlet?"}], "max_new_tokens": 5,
                           "mode": "flag"}],
                         on_text=lambda i, d: texts.__setitem__(i, texts.get(i, "") + d),
                         on_reasoning=lambda i, d: reasons.__setitem__(i, reasons.get(i, "") + d))[0]
    check_split(tok2, g, r, 6, 5)
    assert r["reasoning"] and reasons[0].strip() == r["reasoning"]
    assert texts.get(0, "").strip() == r["text"]
    assert all(st["text"] in r["text"] for st in r["sentences"])          # sentences are the reply's only


def test_model_opened_think_block(model, tok):
    g = gen(model, tok, think_budget=4)
    calls = {"n": 0}

    def push_think(m, a, o):                                              # the prompt step: open <think>
        calls["n"] += 1
        if calls["n"] == 1:
            o = o.clone()
            o[..., g.think_open] = 1e4
            return o
    hk = model.lm_head.register_forward_hook(push_think)
    try:
        rs = g.generate_batch([{"messages": [{"role": "user", "content": "2+2?"}], "max_new_tokens": 4, "mode": "flag",
                                "thinking": True},
                               {"messages": [{"role": "user", "content": "Hello"}], "max_new_tokens": 4, "mode": "flag",
                                "thinking": False}])
    finally:
        hk.remove()
    assert rs[0]["prefix_ids"][0] == g.think_open
    check_split(tok, g, rs[0], 4, 4)
    assert rs[1]["prefix_ids"][:1] == [g.think_open]                     # the hook pushed both rows
    check_split(tok, g, rs[1], 4, 4)


def test_no_thinking_by_default(model, tok):
    g = gen(model, tok)
    r = g.generate_batch([{"messages": [{"role": "user", "content": "Hello"}], "max_new_tokens": 6, "mode": "flag"}])[0]
    if g.think_open not in r["prefix_ids"][:1]:                          # a random model rarely opens one
        assert r["reasoning"] is None and r["prefix_ids"] == []


def test_server_streams_reasoning_apart(model):
    import json
    import threading
    import urllib.error
    import urllib.request
    from vt.verichat import VeriChat
    from vt.verichat_server import make_server
    tok2 = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    tok2.chat_template = R1_TEMPLATE
    bot = VeriChat(model, tok2, Zero(), 1, think_budget=6)
    srv = make_server(bot, port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_port}"
    try:
        h = json.loads(urllib.request.urlopen(url + "/health").read())
        assert h["thinking_available"] and h["thinking"] is False

        def post(body):
            return urllib.request.urlopen(urllib.request.Request(url + "/chat", json.dumps(body).encode(),
                                                                 {"Content-Type": "application/json"}))
        msgs = [{"role": "user", "content": "Who wrote Hamlet?"}]
        with pytest.raises(urllib.error.HTTPError) as e:
            post({"messages": msgs, "thinking": "yes"})
        assert e.value.code == 400
        resp = post({"messages": msgs, "mode": "flag", "stream": True, "max_new_tokens": 6})
        events = [json.loads(x[6:]) for x in resp.read().decode().split("\n\n") if x.startswith("data: ")]
        reply = events[-1]["reply"]
        assert reply["reasoning"] and "".join(x.get("reasoning", "") for x in events).strip() == reply["reasoning"]
        assert reply["reasoning"] not in reply["text"] or not reply["text"]
    finally:
        srv.shutdown()
