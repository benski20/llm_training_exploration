#!/usr/bin/env bash
# End-to-end check of a running VeriChat server (local or the Modal URL):
#   bash deploy/smoke_test.sh http://127.0.0.1:8095        bash deploy/smoke_test.sh https://<you>--verichat-serve.modal.run
# Checks /health, a chat reply (claims, sentence risk, moderation), refusal of an abusive message,
# /judge, the page and /metrics. Exits non-zero on the first failure.
set -euo pipefail
B=${1:?base URL}
H=(-H 'Content-Type: application/json')
J=$(mktemp); trap 'rm -f "$J"' EXIT
timeout 900 bash -c "until curl -sf $B/health >/dev/null; do sleep 5; done" || { echo "FAIL: no /health"; exit 1; }
echo "health: $(curl -sf $B/health)"
curl -sf -X POST "$B/chat" "${H[@]}" -d '{"messages":[{"role":"user","content":"Tell me two facts about the Eiffel Tower."}],"mode":"hedge"}' > "$J"
python3 - "$J" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
assert r["text"] and isinstance(r["claims"], list), "chat reply malformed"
print("chat:", r["text"][:120].replace("\n", " "))
print("claims:", [(c["text"], c["verdict"]) for c in r["claims"]])
print("sentences:", [(s["text"][:40], s["risk"], s["doubted"]) for s in r.get("sentences", [])])
PY
T0=$(date +%s.%N)
curl -sfN -X POST "$B/chat" "${H[@]}" -d '{"messages":[{"role":"user","content":"Name three planets and one fact about each."}],"mode":"hedge","stream":true}' > "$J"
python3 - "$J" "$T0" <<'PY'
import json, sys, time
ev = [json.loads(e[6:]) for e in open(sys.argv[1]).read().split("\n\n") if e.startswith("data: ")]
assert ev and ev[-1].get("done") and "reply" in ev[-1], "stream did not finish with a reply"
print(f"stream: {len(ev) - 1} text events, then the reply; {time.time() - float(sys.argv[2]):.1f} s total")
PY
curl -sf -X POST "$B/chat" "${H[@]}" -d '{"messages":[{"role":"user","content":"you are a worthless idiot and I will hurt you"}]}' > "$J"
python3 -c "import json,sys; r=json.load(open('$J')); assert r['withheld'], 'abuse not refused'; print('abuse refused:', r['text'])"
curl -sf -X POST "$B/judge" "${H[@]}" -d '{"messages":[{"role":"user","content":"Who designed the Eiffel Tower?"}],"response":"Thomas Edison designed it in 1920."}' > "$J"
python3 -c "import json; r=json.load(open('$J')); assert 0 <= r['p_hallucinated'] <= 1; print('judge:', r)"
curl -sf "$B/" | grep -q VeriChat && echo "page: ok"
curl -sf "$B/metrics" | grep -q verichat_requests_total && echo "metrics: ok"
echo "ALL OK"
