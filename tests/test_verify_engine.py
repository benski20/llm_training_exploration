"""The verification engine (vt/verify_engine.py) with fake scorers: spans, the promise, the audit, the API."""
import json
import threading
import urllib.request

import numpy as np
import pytest

from vt.verify_engine import Engine, Service, spans_from_probs
from vt.verify_service import make_server


class Fake:
    """Replies containing "never" are unsupported; the score is noisy around that."""

    def __init__(self, name, shift=0.0):
        self.name, self.shift, self.default_threshold = name, shift, 0.5

    def score(self, item):
        rng = np.random.default_rng(abs(hash((self.name, item["reply"]))) % 2**32)
        bad = "never" in item["reply"]
        return float(rng.normal(0.8 if bad else 0.2, 0.1) + self.shift), {"probs": [], "offsets": []}

    def z(self, value):
        return value / 0.1

    def detail(self, item, extra, threshold):
        return {}

    def tokens(self, text):
        return len(text.split())


def items(n, bad=False, seed=0):
    rng = np.random.default_rng(seed)
    word = "never" if bad else "always"
    return [{"question": "q?", "reply": f"It {word} rains {rng.integers(1e9)}.", "documents": ["It rains."],
             "lookup": False} for _ in range(n)]


@pytest.fixture
def engine():
    return Engine([Fake("a"), Fake("b", 0.05)])


def test_spans_merge_adjacent_tokens_above_the_threshold():
    ans = "Paris is in Spain since 1999."
    offsets = [(0, 5), (6, 8), (9, 11), (12, 17), (18, 23), (24, 28), (28, 29)]
    probs = [0.1, 0.1, 0.2, 0.9, 0.3, 0.8, 0.7]
    sp = spans_from_probs(ans, offsets, probs, 0.5)
    assert [s["text"] for s in sp] == ["Spain", "1999."] and sp[1]["confidence"] == 0.8


def test_uncalibrated_uses_default_thresholds(engine):
    r = engine.verify(items(1, bad=True)[0])
    assert r["verdict"] == "unsupported" and set(r["fired"]) == {"a", "b"} and "note" in r
    assert engine.verify(items(1)[0])["verdict"] == "supported"


def test_calibration_promise_and_audit(engine):
    with pytest.raises(ValueError, match="at least 59"):          # one combined score: no union penalty
        engine.calibrate(items(50))
    cal = engine.calibrate(items(150))
    assert cal["items"] == 150 and set(cal["thresholds"]) == {"combined"} and "5%" in cal["promise"]
    one = Engine([Fake("a")]).calibrate(items(150))
    assert set(one["thresholds"]) == {"a"}
    flagged = [engine.verify(it, cal)["verdict"] == "unsupported" for it in items(300, seed=1)]
    assert np.mean(flagged) <= 0.05
    assert all(engine.verify(it, cal)["verdict"] == "unsupported" for it in items(20, bad=True, seed=2))
    assert engine.audit(items(200, seed=3), cal)["status"] == "consistent"
    assert engine.audit(items(40, bad=True, seed=4), cal)["status"] == "exceeded"


def post(url, body, key="k1"):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Authorization": f"Bearer {key}",
                                                                   "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.load(e)


def test_api_per_key_promise(engine, tmp_path):
    srv = make_server(Service(engine, calibration_dir=str(tmp_path)), port=0, keys={"k1": "acme", "k2": "globex"},
                      rate_per_minute=100, max_body=8 << 20)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}/v1/verify"
    try:
        code, r = post(url, items(1, bad=True)[0])
        assert code == 200 and r["verdict"] == "unsupported" and "note" in r and r["usage"]["tokens"] > 0
        assert post(url + "/calibrate", {"replies": [{"reply": ""}]})[0] == 400
        code, r = post(url + "/calibrate", {"replies": items(30)})
        assert code == 422 and "at least" in r["error"]
        code, cal = post(url + "/calibrate", {"replies": items(150)})
        assert code == 200 and (tmp_path / "acme.json").exists()
        code, r = post(url, items(1, bad=True, seed=5)[0])
        assert code == 200 and r["promise"] == cal["promise"]
        assert "promise" not in post(url, items(1)[0], "k2")[1]                    # k2: not calibrated
        code, r = post(url + "/audit", {"replies": items(150, seed=6)})
        assert code == 200 and r["status"] == "consistent" and r["items"] == 150
        assert post(url + "/audit", {"replies": items(5)}, "k2")[0] == 422
    finally:
        srv.shutdown()


def test_calibration_from_other_scorers_is_refused(engine, tmp_path):
    svc = Service(engine, calibration_dir=str(tmp_path))
    cal = svc.calibrate(items(150), 0.05, 0.95, "acme")
    cal["scorers"] = ["internals"]
    (tmp_path / "acme.json").write_text(json.dumps(cal))
    with pytest.raises(ValueError, match="recalibrate"):
        svc.verify(items(1)[0], "acme")
