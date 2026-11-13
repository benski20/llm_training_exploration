"""The AI-text detector (vt/aitext.py): calibration maths, verdicts, and the HTTP API, with a fake reader."""
import json
import threading
import urllib.request

import numpy as np
import pytest

from vt.aitext import (
    Detector,
    Service,
    audit_from_flags,
    calibration_from_scores,
    min_texts,
    train_head,
    validate,
)
from vt.verify_service import make_server

D = 8                                                   # hidden width per layer in the fake


class FakeReader:
    """Texts containing 'delve' look like AI: low perplexity, a hidden direction set."""
    names = ("obs", "perf")

    def features(self, texts, batch=2, max_tokens=256):
        S, H = [], []
        for t in texts:
            rng = np.random.default_rng(abs(hash(t)) % 2**32)
            ai = "delve" in t
            nll = rng.normal(2.0 if ai else 3.5, 0.3)
            S.append([nll, nll * 0.9, nll * (1.3 if ai else 1.0), 2.5, 1.0, 0.4, min(256, len(t.split()))])
            h = rng.normal(0, 1, 4 * D)
            h[0] += 3 if ai else 0
            H.append(h)
        return np.array(S, np.float32), np.array(H, np.float16)


def texts(n, ai, seed=0):
    rng = np.random.default_rng(seed)
    word = "delve" if ai else "walk"
    return [f"{word} " + " ".join(f"w{rng.integers(1e6)}" for _ in range(80)) for _ in range(n)]


@pytest.fixture(scope="module")
def detector(tmp_path_factory):
    r = FakeReader()
    S, H = r.features(texts(200, False, 1) + texts(200, True, 2))
    head = str(tmp_path_factory.mktemp("h") / "head.npz")
    train_head(S, H, np.r_[np.zeros(200), np.ones(200)], head)
    return Detector(head, reader=r)


def test_min_texts_matches_the_bound():
    assert min_texts(0.05, 0.95) == 146                 # union of two scores at 2.5% / 97.5%
    assert min_texts(0.05, 0.95, k=1) == 59


def test_calibrate_refuses_too_few_and_promises_otherwise(detector):
    with pytest.raises(ValueError, match="at least 146"):
        detector.calibrate(texts(100, False, 3))
    cal = detector.calibrate(texts(150, False, 3))
    assert cal["human_texts"] == 150 and set(cal["thresholds"]) == {"probe", "binoculars"}
    assert cal["head_sha256"] == detector.head_sha and "5%" in cal["promise"]
    res = detector.check(texts(300, False, 4) + texts(100, True, 5), cal)
    flagged = np.array([r["verdict"] == "ai" for r in res])
    assert flagged[:300].mean() <= 0.05 and flagged[300:].mean() > 0.9


def test_calibration_from_scores_uses_half_budgets():
    s = {"probe": np.arange(300.0), "binoculars": np.arange(300.0)}
    cal = calibration_from_scores(s, 0.05, 0.95)
    for t in cal["thresholds"].values():
        assert (s["probe"] > t).mean() <= 0.025


def test_audit_reports_whether_the_promise_held(detector):
    ok = audit_from_flags([True] * 4 + [False] * 196, 0.05)
    assert ok["status"] == "consistent" and ok["flagged"] == 4 and ok["ci95"][0] < 0.02 < ok["ci95"][1]
    bad = audit_from_flags([True] * 25 + [False] * 175, 0.05)
    assert bad["status"] == "exceeded" and bad["p_value"] < 0.05 and bad["flagged_indices"][:2] == [0, 1]
    cal = detector.calibrate(texts(150, False, 3))
    assert detector.audit(texts(200, False, 11), cal)["status"] == "consistent"
    assert detector.audit(texts(50, True, 12), cal)["status"] == "exceeded"     # AI passed off as human


def test_check_rejects_a_calibration_from_another_head(detector):
    cal = detector.calibrate(texts(150, False, 3))
    cal["head_sha256"] = "0" * 16
    with pytest.raises(ValueError, match="different probe head"):
        detector.check(texts(1, True), cal)


def test_validate():
    assert validate({"text": "short"})[1].startswith("text")
    assert validate({"text": "x" * 30000})[1].startswith("text")
    assert validate({"text": "a " * 150})[0] == {"text": "a " * 150}


def post(url, body, key):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Authorization": f"Bearer {key}",
                                                                   "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.load(e)


def test_api_per_key_calibration(detector, tmp_path):
    srv = make_server(Service(detector, calibration_dir=str(tmp_path)), port=0, keys={"k1": "acme", "k2": "globex"},
                      rate_per_minute=100, max_body=8 << 20, path="/v1/aitext", validator=validate)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}/v1/aitext"
    try:
        code, r = post(url, {"text": texts(1, True)[0]}, "k1")
        assert code == 200 and r["verdict"] is None and "not calibrated" in r["note"]
        code, r = post(url + "/calibrate", {"human_texts": texts(20, False)}, "k1")
        assert code == 422 and "at least 146" in r["error"]
        code, cal = post(url + "/calibrate", {"human_texts": texts(150, False, 3)}, "k1")
        assert code == 200 and (tmp_path / "acme.json").exists()
        code, r = post(url, {"text": texts(1, True, 9)[0]}, "k1")
        assert code == 200 and r["verdict"] == "ai" and r["promise"] == cal["promise"]
        code, r = post(url, {"text": texts(1, True, 9)[0]}, "k2")              # another key: its own calibration
        assert code == 200 and r["verdict"] is None
        code, r = post(url + "/audit", {"human_texts": texts(160, False, 13)}, "k1")
        assert code == 200 and r["status"] == "consistent" and r["human_texts"] == 160
        assert post(url + "/audit", {"human_texts": texts(5, False)}, "k2")[0] == 422      # k2 has no calibration
        code, r = post(url + "/batch", {"items": [{"text": t} for t in texts(2, False, 7)]}, "k1")
        assert code == 200 and len(r["results"]) == 2
        assert post(url, {"text": texts(1, True)[0]}, "bad")[0] == 401
    finally:
        srv.shutdown()
