import json, threading, time, sys, httpx
B = "http://127.0.0.1:8080"
c = httpx.Client(timeout=300, trust_env=False)

def snap():
    s = c.get(B + "/api/state").json()
    t, g = s["tap"], s["gateway"]
    return {"calls": t["calls"], "up": t["bytes_up"], "down": t["bytes_down"],
            "hist": [round(x, 5) for x in t["hist"]], "sample": t["sample"], "lin": t["sample_linear"],
            "values": t["values_seen"],
            "recent": [{"linear": r["linear"], "rows": r["rows"], "bytes": r["bytes"]} for r in t["recent"][:8]],
            "rt": g.get("round_trips"), "fv": g.get("freivalds_checked"), "if": g.get("integrity_failures"),
            "masks": g.get("masks_available")}

def run(prompt, tamper=False):
    if tamper:
        c.post(B + "/api/tamper")
    ev, done = [], threading.Event()
    t0 = time.time()
    def poll():
        while not done.is_set():
            ev.append({"t": round(time.time() - t0, 3), "k": "s", "s": snap()}); time.sleep(0.3)
    th = threading.Thread(target=poll); th.start()
    err = None
    with c.stream("POST", B + "/api/chat", json={"messages": [{"role": "user", "content": prompt}], "max_tokens": 90}) as r:
        buf = ""
        for chunk in r.iter_text():
            buf += chunk
            while "\n\n" in buf:
                e, buf = buf.split("\n\n", 1)
                for line in e.split("\n"):
                    if not line.startswith("data: ") or line.strip() == "data: [DONE]": continue
                    j = json.loads(line[6:])
                    if "demo_error" in j or "error" in j: err = j.get("demo_error") or j.get("error"); continue
                    d = (j.get("choices") or [{}])[0].get("delta", {}).get("content")
                    if d: ev.append({"t": round(time.time() - t0, 3), "k": "tok", "x": d})
    time.sleep(0.4); done.set(); th.join()
    ev.append({"t": round(time.time() - t0, 3), "k": "s", "s": snap()})
    ev.append({"t": round(time.time() - t0, 3), "k": "end", "err": err})
    if tamper:
        c.post(B + "/api/reset")
    print(prompt[:40], "->", "ERR" if err else "".join(e["x"] for e in ev if e["k"] == "tok")[:200], round(time.time()-t0,1), file=sys.stderr)
    return {"prompt": prompt, "tamper": tamper, "events": ev}

scen = [
  ("Summarize in two sentences: Q3 revenue fell 12% to $41M; we plan to cut 60 roles in the Austin office before the board meeting on Oct 3.", False),
  ("Draft a polite two-sentence reply declining a supplier's 8% price increase.", False),
  ("List three risks of sending patient records to a third-party AI service.", False),
  ("What is the capital of France?", True),
]
out = {"recorded_at": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), "start": snap(), "scenarios": [run(p, t) for p, t in scen]}
json.dump(out, open(sys.argv[1], "w"), separators=(",", ":"))
