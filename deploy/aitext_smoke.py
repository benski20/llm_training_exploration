"""End-to-end check of a running AI-text API (vt-aitext serve, or the Dockerfile.aitext container).

Standard library only, so it runs inside the container: `docker exec <c> python /smoke/aitext_smoke.py ...`.
Steps: health; a check before calibration gives no verdict; calibration refuses too few texts; calibrate on 150
human texts; audit 50 other human texts (must be "consistent"); check 20 AI texts in batches (most flagged); a
bad key is refused. With --after-restart it only checks that the stored calibration survived a restart.

    python deploy/aitext_smoke.py --url http://127.0.0.1:8000 --key "$KEY" --data examples/aitext_smoke
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


def texts(path):
    with open(path) as f:
        return [json.loads(line)["text"] for line in f if line.strip()]


def check(ok, what):
    print(("PASS " if ok else "FAIL ") + what, flush=True)
    if not ok:
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--key", required=True)
    ap.add_argument("--data", default="examples/aitext_smoke")
    ap.add_argument("--wait", type=int, default=300, help="seconds to wait for /health")
    ap.add_argument("--min-caught", type=int, default=14, help="of the 20 AI texts")
    ap.add_argument("--after-restart", action="store_true")
    a = ap.parse_args()
    base, api = a.url.rstrip("/"), a.url.rstrip("/") + "/v1/aitext"
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
    ai, human_cal, human_new = (texts(f"{a.data}/{n}.jsonl") for n in ("ai", "human_calibrate", "human_audit"))

    if a.after_restart:
        code, r = call(api, {"text": ai[0]}, a.key)
        check(code == 200 and r.get("verdict") in ("ai", "not flagged"), "calibration survived the restart")
        return

    code, r = call(api, {"text": ai[0]}, a.key)
    check(code == 200 and r.get("verdict") is None, "before calibration: scores but no verdict")
    code, r = call(api + "/calibrate", {"human_texts": human_cal[:20]}, a.key)
    check(code == 422, "calibration refuses 20 texts (needs 146)")
    t0 = time.time()
    code, cal = call(api + "/calibrate", {"human_texts": human_cal}, a.key)
    check(code == 200 and cal.get("human_texts") == len(human_cal),
          f"calibrated on {len(human_cal)} human texts in {time.time() - t0:.0f}s: {cal.get('promise')}")
    t0 = time.time()
    code, aud = call(api + "/audit", {"human_texts": human_new}, a.key)
    check(code == 200 and aud.get("status") == "consistent",
          f"audit of {len(human_new)} new human texts: {aud.get('flagged')} flagged, {aud.get('status')} "
          f"({time.time() - t0:.0f}s)")
    caught = 0
    for i in range(0, len(ai), 10):
        code, r = call(api + "/batch", {"items": [{"text": t} for t in ai[i:i + 10]]}, a.key)
        check(code == 200, f"batch {i // 10 + 1}")
        caught += sum(x.get("verdict") == "ai" for x in r["results"])
    check(caught >= a.min_caught, f"{caught} of {len(ai)} AI texts flagged (need {a.min_caught})")
    check(call(api, {"text": ai[0]}, "not-a-key")[0] == 401, "a bad key is refused")
    print(json.dumps({"caught": caught, "of": len(ai), "audit": {k: aud[k] for k in ("flagged", "status")}}))


if __name__ == "__main__":
    main()
