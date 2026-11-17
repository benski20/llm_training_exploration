"""Record the running client demo (``run.py`` on :8080) for the standalone replay page.

    python scripts/client_demo/record.py recording.json

Plays five scenarios through the demo UI's API: three questions, one with the tamper switch armed,
and the same question again after an admin reset. Saves every streamed token and a snapshot of the
service-side view (wire tap and gateway counters) every 0.3 s, with timestamps.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from typing import Any

import httpx

BASE = "http://127.0.0.1:8080"
CLIENT = httpx.Client(timeout=300, trust_env=False)

SCENARIOS: list[tuple[str, bool]] = [
    ("Summarize in two sentences: Q3 revenue fell 12% to $41M; we plan to cut 60 roles in the Austin office "
     "before the board meeting on Oct 3.", False),
    ("Draft a polite two-sentence reply declining a supplier's 8% price increase.", False),
    ("List three risks of sending patient records to a third-party AI service.", False),
    ("What is the capital of France?", True),
    ("What is the capital of France?", False),
]


def snap() -> dict[str, Any]:
    s = CLIENT.get(BASE + "/api/state").json()
    t, g = s["tap"], s["gateway"]
    return {
        "calls": t["calls"], "up": t["bytes_up"], "down": t["bytes_down"],
        "hist": [round(x, 5) for x in t["hist"]], "sample": t["sample"], "lin": t["sample_linear"],
        "values": t["values_seen"],
        "recent": [{"linear": r["linear"], "rows": r["rows"], "bytes": r["bytes"]} for r in t["recent"][:8]],
        "rt": g.get("round_trips"), "fv": g.get("freivalds_checked"), "if": g.get("integrity_failures"),
        "masks": g.get("masks_available"),
    }


def _sse(line: str) -> dict[str, Any] | None:
    if not line.startswith("data: ") or line.strip() == "data: [DONE]":
        return None
    parsed: dict[str, Any] = json.loads(line[6:])
    return parsed


def run(prompt: str, tamper: bool) -> dict[str, Any]:
    if tamper:
        CLIENT.post(BASE + "/api/tamper")
    events: list[dict[str, Any]] = []
    done = threading.Event()
    t0 = time.time()

    def stamp() -> float:
        return round(time.time() - t0, 3)

    def poll() -> None:
        while not done.is_set():
            events.append({"t": stamp(), "k": "s", "s": snap()})
            time.sleep(0.3)

    poller = threading.Thread(target=poll)
    poller.start()
    err = None
    body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": 90}
    with CLIENT.stream("POST", BASE + "/api/chat", json=body) as r:
        buf = ""
        for chunk in r.iter_text():
            buf += chunk
            while "\n\n" in buf:
                event, buf = buf.split("\n\n", 1)
                for line in event.split("\n"):
                    j = _sse(line)
                    if j is None:
                        continue
                    if "demo_error" in j or "error" in j:
                        err = j.get("demo_error") or j.get("error")
                        continue
                    delta = (j.get("choices") or [{}])[0].get("delta", {}).get("content")
                    if delta:
                        events.append({"t": stamp(), "k": "tok", "x": delta})
    time.sleep(0.4)
    done.set()
    poller.join()
    events.append({"t": stamp(), "k": "s", "s": snap()})
    events.append({"t": stamp(), "k": "end", "err": err})
    if tamper:
        CLIENT.post(BASE + "/api/reset")
    text = "".join(e["x"] for e in events if e["k"] == "tok")
    print(f"{prompt[:40]} -> {'ERROR' if err else text[:120]} ({stamp()} s)", file=sys.stderr)
    return {"prompt": prompt, "tamper": tamper, "events": events}


def main() -> None:
    out = {"recorded_at": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), "start": snap(),
           "scenarios": [run(p, t) for p, t in SCENARIOS]}
    with open(sys.argv[1], "w") as fh:
        json.dump(out, fh, separators=(",", ":"))


if __name__ == "__main__":
    main()
