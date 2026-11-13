"""End-to-end check of a running verification engine (python -m vt.verify_service --engine ..., or its container).

Standard library only, so it runs inside the container. Steps: health; an uncalibrated check answers with default
thresholds and a note; calibration refuses too few replies; calibrate on 80 replies known to be supported (RAGTruth
QA, examples/verify_smoke/); audit 40 others (must be "consistent"); check 20 replies with labelled hallucinations
in batches (enough flagged, and the flagged words overlap the labelled ones); a bad key is refused. With
--after-restart it only checks that the stored calibration survived a restart.

    python deploy/verify_smoke.py --url http://127.0.0.1:8000 --key "$KEY" --data examples/verify_smoke
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request


def call(url, body=None, key=None, timeout=1500):
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data, headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.load(e)


def items(path):
    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    keep = ("question", "reply", "documents")
    return [{k: r[k] for k in keep} for r in rows], [r.get("labelled_spans", []) for r in rows]


def check(ok, what):
    print(("PASS " if ok else "FAIL ") + what, flush=True)
    if not ok:
        sys.exit(1)


def overlaps(spans, labelled):
    """Does any flagged span share a word with a labelled hallucinated span?"""
    words = {w.strip(".,;:!?\"'()").lower() for s in labelled for w in s.split()} - {""}
    return any(w.strip(".,;:!?\"'()").lower() in words for sp in spans for w in sp["text"].split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--key", required=True)
    ap.add_argument("--data", default="examples/verify_smoke")
    ap.add_argument("--wait", type=int, default=300, help="seconds to wait for /health")
    ap.add_argument("--min-caught", type=int, default=5, help="of the 20 replies with labelled hallucinations")
    ap.add_argument("--after-restart", action="store_true")
    a = ap.parse_args()
    base, api = a.url.rstrip("/"), a.url.rstrip("/") + "/v1/verify"
    for _ in range(a.wait):
        try:
            code, h = call(base + "/health", timeout=5)
            if code == 200:
                break
        except OSError:
            pass
        time.sleep(1)
    else:
        check(False, "health")
    check(h.get("status") == "ok", f"health {json.dumps(h)[:200]}")
    good_cal, _ = items(f"{a.data}/good_calibrate.jsonl")
    good_new, _ = items(f"{a.data}/good_audit.jsonl")
    bad, labelled = items(f"{a.data}/unsupported.jsonl")

    if a.after_restart:
        code, r = call(api, bad[0], a.key)
        check(code == 200 and "promise" in r, "calibration survived the restart")
        return

    code, r = call(api, bad[0], a.key)
    check(code == 200 and r.get("verdict") in ("supported", "unsupported") and "note" in r,
          f"before calibration: default thresholds ({r.get('verdict')}, {r.get('ms')} ms)")
    code, r = call(api + "/calibrate", {"replies": good_cal[:20]}, a.key)
    check(code == 422, "calibration refuses 20 replies")
    t0 = time.time()
    code, cal = call(api + "/calibrate", {"replies": good_cal}, a.key)
    check(code == 200 and cal.get("items") == len(good_cal),
          f"calibrated on {len(good_cal)} supported replies in {time.time() - t0:.0f}s: {cal.get('promise')}")
    t0 = time.time()
    code, aud = call(api + "/audit", {"replies": good_new}, a.key)
    check(code == 200 and aud.get("status") == "consistent",
          f"audit of {len(good_new)} new supported replies: {aud.get('flagged')} flagged, {aud.get('status')} "
          f"({time.time() - t0:.0f}s)")
    caught = located = 0
    for i in range(0, len(bad), 10):
        code, r = call(api + "/batch", {"items": bad[i:i + 10]}, a.key)
        check(code == 200, f"batch {i // 10 + 1}")
        for res, lab in zip(r["results"], labelled[i:i + 10]):
            if res.get("verdict") == "unsupported":
                caught += 1
                located += overlaps(res.get("spans", []), lab)
    check(caught >= a.min_caught, f"{caught} of {len(bad)} labelled hallucinations flagged (need {a.min_caught}); "
          f"in {located} the flagged words overlap the labelled ones")
    check(call(api, bad[0], "not-a-key")[0] == 401, "a bad key is refused")
    print(json.dumps({"caught": caught, "located": located, "of": len(bad),
                      "audit": {k: aud[k] for k in ("flagged", "status")}}))


if __name__ == "__main__":
    main()
