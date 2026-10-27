"""Keep the experiments dashboard in step with the container.

    python experiments/dashboard_sync.py                  # export, then list files that changed
    python experiments/dashboard_sync.py --record NAME BLOB VERSION  # after uploading NAME and updating its
                                                                     # dataset document to VERSION

results/dashboard/state.json maps each data file to the sha256 and blob of
its last upload, so a refresh uploads only what changed and knows which old
blob to delete.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..")
DIR = os.path.join(ROOT, "results", "dashboard")
STATE = os.path.join(DIR, "state.json")
FILES = ["runs", "curves", "behaviour", "detection", "chat"]


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", nargs=3, metavar=("NAME", "BLOB", "VERSION"))
    a = ap.parse_args()
    state = json.load(open(STATE)) if os.path.exists(STATE) else {}
    if a.record:
        name, blob, version = a.record
        state[name] = {"sha": sha(os.path.join(DIR, f"{name}.json")), "blob": blob, "version": int(version)}
        json.dump(state, open(STATE, "w"), indent=1)
        print(json.dumps({"recorded": name, "blob": blob}))
        return
    out = subprocess.run([sys.executable, os.path.join(ROOT, "experiments", "dashboard_export.py")],
                         capture_output=True, text=True, check=True).stdout.strip()
    exported = json.loads(out.splitlines()[-1])
    changed = [n for n in FILES if state.get(n, {}).get("sha") != sha(os.path.join(DIR, f"{n}.json"))]
    runs = json.load(open(os.path.join(DIR, "runs.json")))
    print(json.dumps({"exported_at": exported["exported_at"], "changed": changed,
                      "old": {n: {k: state.get(n, {}).get(k) for k in ("blob", "version")} for n in changed},
                      "running": [r["run"] for r in runs if r["status"] == "running"]}))


if __name__ == "__main__":
    main()
