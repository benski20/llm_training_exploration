"""Summarise results/cegis/*/ into a markdown table (stdout and summary.md)."""
import glob
import json
import os
import re
import time

ROOT = os.path.join(os.path.dirname(__file__), "..", "results", "cegis")


def load(run):
    hist = os.path.join(run, "history.json")
    if os.path.exists(hist):
        with open(hist) as f:
            h = json.load(f)
        return h["status"], h["history"], h["config"], h["elapsed"]
    prog = os.path.join(run, "progress.jsonl")
    if os.path.exists(prog):
        with open(prog) as f:
            rows = [json.loads(line) for line in f if line.strip()]
        stale = time.time() - os.path.getmtime(prog) > 1800
        return ("stopped" if stale else "running"), rows, None, rows[-1]["elapsed"] if rows else 0.0
    return None


def main():
    lines = ["| run | spec | L | inputs | mixing | weights | status | rounds | verifier calls | last verify (s) | total (s) | certificate |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for run in sorted(glob.glob(os.path.join(ROOT, "*"))):
        r = load(run)
        if r is None:
            continue
        status, rows, cfg, elapsed = r
        name = os.path.basename(run)
        mt = re.search(r"L(\d+)(?:-(\d+))?", name)
        L = cfg["L"] if cfg else int(mt.group(2) or mt.group(1))
        spec = "sorted" if name.startswith("sorted") else "dyck"
        V = 4 if spec == "sorted" else 2
        if cfg and cfg.get("lengths"):
            L = f"{min(cfg['lengths'])}..{cfg['L']}"
        calls = [e for e in rows if "verify_status" in e or "verify" in e]

        def vtime(e):
            if "verify_time" in e:
                return e["verify_time"]
            return sum(v.get("time", 0.0) for v in e["verify"].values())
        last = f"{vtime(calls[-1]):.0f}" if calls else "-"
        cert = os.path.join(run, "certificate.json")
        sha = ""
        if os.path.exists(cert):
            with open(cert) as f:
                sha = "`" + json.load(f)["artifact_sha256"][:12] + "`"
        if (cfg and cfg.get("shared")) or "shared" in name:
            mixing = "tied (Toeplitz)"
        elif (cfg and cfg.get("causal")) or "causal" in name:
            mixing = "causal"
        else:
            mixing = "dense"
        if cfg and cfg.get("bits"):
            weights = f"{cfg['bits']}-bit"
        elif cfg and cfg.get("ternary"):
            weights = "ternary"
        else:
            weights = next((w for w in ("4bit", "6bit", "ternary") if w in name), "float").replace("bit", "-bit")
        lines.append(f"| {name} | {spec} | {L} | {float(V) ** (cfg['L'] if cfg else int(L)):.2e} | {mixing} | {weights} | {status} | {len(rows)} | {len(calls)} | "
                     f"{last} | {elapsed:.0f} | {sha} |")
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(ROOT, "summary.md"), "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
