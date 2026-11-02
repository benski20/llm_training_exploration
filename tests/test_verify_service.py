"""The verification API (vt/verify_service.py), its client (vt/verify_client.py) and the pipeline's maths."""
import threading

import numpy as np
import pytest

from vt.verifier_pipeline import model_card, rates, recall_at_fpr
from vt.verify_client import VerifyClient, VerifyError
from vt.verify_service import Verifier, keys_from_env, make_server, validate


class Tok:
    def __call__(self, text, add_special_tokens=False):
        class E:
            input_ids = text.split()
        return E()


class FakeGrounding:
    def __init__(self):
        self.tok, self.retriever, self.calls = Tok(), "WIKI", []

    def check(self, messages, reply, documents=()):
        self.calls.append((messages[-1]["content"], reply, list(documents), self.retriever))
        v = "contradicted" if "1925" in reply else "consistent"
        return {"sentences": [{"text": reply, "verdict": v, "z": 2.0 if v != "consistent" else 0.0}],
                "total": 2.0 if v != "consistent" else 0.0, "verdict": v}


@pytest.fixture
def api():
    g = FakeGrounding()
    srv = make_server(Verifier(g, {"reader": "fake", "probe": "p.npz"}), port=0,
                      keys={"secret-1": "acme", "secret-2": "globex"}, rate_per_minute=5)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", g, srv
    srv.shutdown()


def test_verify_one_and_batch_with_usage(api):
    url, g, srv = api
    c = VerifyClient(url, api_key="secret-1", retries=0)
    r = c.verify("It reopened in 1925.", question="When?", documents=["It reopened in 1889."])
    assert r["verdict"] == "contradicted" and r["usage"]["tokens"] == 4 + 4 and "ms" in r
    rs = c.verify_batch([{"reply": "Fine.", "documents": ["Fine."]}, {"reply": "In 1925.", "lookup": True}])
    assert [x["verdict"] for x in rs] == ["consistent", "contradicted"]
    assert g.calls[-1][3] == "WIKI" and g.calls[-2][3] is None          # a lookup only when asked and undocumented
    assert 'verify_tokens_total{key="acme"}' in srv.usage.text()


def test_keys_rate_limits_and_validation(api):
    url, _, _ = api
    with pytest.raises(VerifyError) as e:
        VerifyClient(url, api_key="wrong", retries=0).verify("x")
    assert e.value.status == 401
    c = VerifyClient(url, api_key="secret-2", retries=0)
    with pytest.raises(VerifyError) as e:
        c.verify("")
    assert e.value.status == 400
    for _ in range(4):
        c.verify("ok.")
    with pytest.raises(VerifyError) as e:                                 # 5 a minute for this key
        c.verify("ok.")
    assert e.value.status == 429
    VerifyClient(url, api_key="secret-1", retries=0).verify("ok.")       # other keys unaffected


def test_validate_limits():
    assert validate({"reply": "a" * 9000})[1].startswith("reply")
    assert validate({"reply": "x", "documents": ["d"] * 21})[1].startswith("documents")
    assert validate({"reply": "x", "documents": "not a list"})[1].startswith("documents")
    assert validate({"reply": "x", "question": "q"})[0] == {"question": "q", "reply": "x", "documents": [], "lookup": False}


def test_keys_from_env(monkeypatch, tmp_path):
    f = tmp_path / "keys"
    f.write_text("# comment\nteam:k3\n")
    monkeypatch.setenv("VERIFY_API_KEYS", "a:k1, b:k2")
    monkeypatch.setenv("VERIFY_API_KEYS_FILE", str(f))
    assert keys_from_env() == {"k1": "a", "k2": "b", "k3": "team"}


def test_thresholds_and_rates():
    y = np.array([0] * 20 + [1] * 10)
    s = np.r_[np.linspace(0, 1, 20), np.linspace(0.5, 2, 10)]
    t, rec = recall_at_fpr(s, y, 0.05)
    assert (s[y == 0] > t).mean() <= 0.05 and rec > 0.5
    r = rates(s, y, t)
    assert r["false_alarms"] <= 0.05 and r["hallucinated"] == 10 and r["clean"] == 20


def test_model_card_states_numbers(tmp_path):
    p = tmp_path / "probe.calibrated.npz"
    np.savez(p, cal_n=np.int32(100), cal_false_alarms=np.float32(0.05))
    res = {"reader": "R", "sha256": "ab" * 32, "n": 50, "auroc": 0.81, "labeled": "x.jsonl", "threshold": 1.2,
           "seconds_per_reply": 0.4, "at_threshold": {"caught": 0.6, "false_alarms": 0.04, "precision": 0.9,
                                                      "hallucinated": 25, "clean": 25},
           "per_source": {"news": {"auroc": 0.8, "caught": 0.5, "false_alarms": 0.0}}}
    card = model_card(res, str(p))
    assert "AUROC 0.81" in card and "60% of 25 hallucinated" in card and "| news | 0.8 |" in card
    assert "checks grounding, not truth" in card
