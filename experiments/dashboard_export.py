"""Export experiment status, training curves and results as JSON rows for the
experiments dashboard.

    python experiments/dashboard_export.py --out results/dashboard

Writes runs.json, curves.json, behaviour.json, detection.json and chat.json,
each a list of row objects, built from the logs and result files of the
glass-box experiments.
"""
import argparse
import json
import os
import subprocess
import time

ROOT = os.path.join(os.path.dirname(__file__), "..")
G = "results/glassbox"

# id, title, log, result json, model, total steps, note
RUNS = [
    ("v1", "v1 glass-box", f"{G}/run.log", f"{G}/run.json", "glassbox", 6000, "evidence stream, starved core, memory, k"),
    ("v1", "v1 GPT baseline", f"{G}/run.log", f"{G}/run.json", "baseline", 6000, "matched standard GPT"),
    ("v2", "v2 all-position patching", f"{G}/v2.log", f"{G}/v2_allpos_iit_collapsed.json", "glassbox", 6000,
     "training collapsed"),
    ("v3", "v3 late memory", f"{G}/v3_late_memory.log", f"{G}/v3_late_memory.json", "glassbox", 6000,
     "second memory read after the last block"),
    ("v4", "v4 zero-init memory", f"{G}/v4_zero_memory.log", f"{G}/v4_zero_memory.json", "glassbox", 6000,
     "unwritten memory reads as empty"),
    ("v5", "v5 memory after last block", f"{G}/v5_late_only_familiarity.log", f"{G}/v5_late_only_familiarity.json",
     "glassbox", 6000, "causal read at the answer, usage familiarity"),
    ("v6", "v6 null slot", f"{G}/v6_null_slot.log", f"{G}/v6_null_slot.json", "glassbox", 6000,
     "a 'nothing stored' option in every read"),
    ("v7", "v7 memory heads", f"{G}/v7_memory_heads.log", f"{G}/v7_memory_heads.json", "glassbox", 6000,
     "training collapsed"),
    ("v8", "v8 fresh entities, glass-box", f"{G}/v8_fresh_glassbox.log", f"{G}/v8_fresh_glassbox.json", "glassbox",
     6000, "abstention taught on never-seen entities"),
    ("v8", "v8 fresh entities, GPT", f"{G}/v8_fresh_baseline.log", f"{G}/v8_fresh_baseline.json", "baseline", 6000,
     "same data for the matched GPT"),
    ("chat", "Chat glass-box", "results/glassbox_chat/run.log", "results/glassbox_chat/run.json", "glassbox", 3000,
     "Alpaca + WikiText-2 + fact world"),
    ("chat", "Chat GPT baseline", "results/glassbox_chat/run.log", "results/glassbox_chat/run.json", "baseline", 3000,
     "matched standard GPT"),
]
SPLITS = {"in_dist": "In distribution", "held_relations": "Held-out relations", "long_context": "Long context"}
READOUTS = {"k_builtin": "Built-in k", "probe_best_layer": "Post-hoc probe", "evidence_support": "Evidence support",
            "memory_match": "Memory match", "late_memory_match": "Late memory match",
            "memory_familiarity": "Memory familiarity", "memory_not_null": "Memory not null",
            "memory_head_agreement": "Memory head agreement"}


def steps_of(log, model):
    """Training log rows for one model: [(step, loss)]."""
    out = []
    if not os.path.exists(os.path.join(ROOT, log)):
        return out
    for line in open(os.path.join(ROOT, log)):
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if d.get("model") == model and "step" in d:                       # synthetic world logs
            out.append((d["step"], d["loss"], d.get("qa")))
        elif isinstance(d.get(model), dict) and "step" in d[model]:        # chat logs
            out.append((d[model]["step"], d[model]["lm"], None))
    return out


def running(log):
    try:
        ps = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True, check=False).stdout
    except OSError:
        return False
    stem = os.path.basename(log).replace(".log", "")
    return any("python3" in line and (stem in line or (stem == "run" and os.path.dirname(log) in line))
               for line in ps.splitlines())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/dashboard")
    a = ap.parse_args()
    out = os.path.join(ROOT, a.out)
    os.makedirs(out, exist_ok=True)
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    runs, curves, behaviour, detection, chat = [], [], [], [], []
    for exp, title, log, res, model, total, note in RUNS:
        st = steps_of(log, model)
        path = os.path.join(ROOT, res)
        result = json.load(open(path)).get(model) if os.path.exists(path) else None
        if result is not None:
            status = "collapsed" if "collapsed" in note else "done"
        elif st and running(log):
            status = "running"
        elif st:
            status = "stopped"
        else:
            continue
        run_id = f"{exp}-{model}"
        last = st[-1] if st else (0, None, None)
        runs.append({"run": run_id, "experiment": exp, "title": title, "model": model, "status": status,
                     "step": last[0] + (1 if last[0] == total - 1 else 0), "total_steps": total,
                     "progress": min(1.0, (last[0] + 1) / total) if st else 1.0,
                     "last_loss": last[1], "note": note})
        for s, loss, qa in st:
            curves.append({"run": run_id, "title": title, "step": s, "loss": loss, "qa_loss": qa})
        if result is None:
            continue
        if exp == "chat":
            for split, name in (("tokens_alpaca", "Alpaca"), ("tokens_wikitext", "WikiText-2")):
                r = result[split]
                chat.append({"model": model, "name": "Glass-box" if model == "glassbox" else "GPT", "split": name,
                             "bits_per_token": r["bits_per_token"],
                             "top1_acc": r["top1_acc"], "auroc_msp": r["auroc_msp"], "auroc_probe": r["auroc_probe"],
                             "auroc_k": r.get("auroc_k")})
            continue
        for split, name in SPLITS.items():
            r = result.get(split)
            if not r:
                continue
            behaviour.append({"run": run_id, "experiment": exp, "model": model, "split": name,
                              "accuracy": r.get("accuracy"), "hallucination": r.get("hallucination_rate"),
                              "hallucination_fresh": r.get("hallucination_rate_fresh_entities"),
                              "ctx_acc": r.get("ctx_acc"), "mem_acc": r.get("mem_acc")})
            for key, label in READOUTS.items():
                if key in r.get("auroc", {}):
                    detection.append({"run": run_id, "experiment": exp, "model": model, "split": name,
                                      "readout": label, "auroc": r["auroc"][key]})
    for name, rows in (("runs", runs), ("curves", curves), ("behaviour", behaviour), ("detection", detection),
                       ("chat", chat)):
        json.dump(rows, open(os.path.join(out, f"{name}.json"), "w"), indent=0)
    json.dump({"exported_at": now}, open(os.path.join(out, "exported_at.json"), "w"))
    print(json.dumps({"exported_at": now, "runs": len(runs), "curves": len(curves), "behaviour": len(behaviour),
                      "detection": len(detection), "chat": len(chat)}))


if __name__ == "__main__":
    main()
