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
    ("v9-0.02", "v9 hard memory gate, L1 0.02", f"{G}/v9_hard_gate_0.02.log", f"{G}/v9_hard_gate_0.02.json", "glassbox",
     6000, "gate exactly 0 when memory is unused: exact memory-free certificates"),
    ("v9-0.1", "v9 hard memory gate, L1 0.1", f"{G}/v9_hard_gate_0.1.log", f"{G}/v9_hard_gate_0.1.json", "glassbox",
     6000, "stronger push to close the memory gate"),
    ("chat", "Chat glass-box", "results/glassbox_chat/run.log", "results/glassbox_chat/run.json", "glassbox", 3000,
     "Alpaca + WikiText-2 + fact world"),
    ("chat", "Chat GPT baseline", "results/glassbox_chat/run.log", "results/glassbox_chat/run.json", "baseline", 3000,
     "matched standard GPT"),
    ("chat2", "Chat v2 glass-box", "results/glassbox_chat_v2/glassbox.log", "results/glassbox_chat_v2/glassbox.json",
     "glassbox", 3000, "chatbot-style dialogues, made-up names, zero-init memory"),
    ("chat3", "Chat v3 glass-box, hard memory gate", "results/glassbox_chat_v3/glassbox.log",
     "results/glassbox_chat_v3/glassbox.json", "glassbox", 3000,
     "general chat; tokens with a closed gate are provably made without stored knowledge"),
    ("g2", "Glassbox-2 pilot (CPU)", "results/glassbox2_chat/glassbox2.log", "results/glassbox2_chat/glassbox2.json",
     "glassbox2", 3000, "exact sparse features, exact attribution, logged hard-gated memory"),
    ("g2.1", "Glassbox-2.1 pilot: memory after the last layer", "results/glassbox2_1_chat/glassbox2.log",
     "results/glassbox2_1_chat/glassbox2.json", "glassbox2", 3000,
     "a token is provably memory-free when its own gate is 0; 32 active features"),
    ("g3", "Glassbox-3 pilot: epistemic uncertainty and <unsure>", "results/glassbox3_chat/glassbox3.log",
     "results/glassbox3_chat/glassbox3.json", "glassbox3", 3000,
     "epinet uncertainty raises an explicit <unsure> token; self-consistency replay"),
    ("g2.2", "Glassbox-2.2: straight-through memory gate", "results/glassbox2_2_chat/glassbox2.log",
     "results/glassbox2_2_chat/glassbox2.json", "glassbox2", 3000,
     "2.1 with gates that keep learning when saturated (the Kaggle student's gate)"),
    ("g2.3", "Glassbox-2.3: additive residual stream", "results/glassbox2_3_chat/glassbox2.log",
     "results/glassbox2_3_chat/glassbox2.json", "glassbox2", 3000,
     "2.2 with standard residual sums instead of convex mixing; attribution still exact"),
    ("g2.4", "Glassbox-2.4: in-model feature reconstruction (SAE) objective", "results/glassbox2_4_chat/glassbox2.log",
     "results/glassbox2_4_chat/glassbox2.json", "glassbox2", 3000,
     "2.3 plus a train-only readout: features must reconstruct their own input"),
    ("g2.4s1", "Glassbox-2.4, seed 1 (replicate)", "results/glassbox2_4_s1_chat/glassbox2.log",
     "results/glassbox2_4_s1_chat/glassbox2.json", "glassbox2", 3000, "is the hallucination gain real?"),
    ("gpt-s1", "GPT baseline, seed 1 (replicate)", "results/baseline_s1_chat/baseline.log",
     "results/baseline_s1_chat/baseline.json", "baseline", 3000, "second seed of the matched GPT"),
    ("chat2", "Chat v2 GPT baseline", "results/glassbox_chat_v2/baseline.log", "results/glassbox_chat_v2/baseline.json",
     "baseline", 3000, "same chat data for the matched GPT"),
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


def running(res):
    """A training process is running for this run if its command line names the run's result file."""
    try:
        ps = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True, check=False).stdout
    except OSError:
        return False
    return any(line.startswith("python3") and res in line for line in ps.splitlines())


KAGGLE_LOG = "results/kaggle/launch.log"
KAGGLE_RESULTS = "results/kaggle/verifiable-chat-retrofit/results.json"


def kaggle_rows():
    """The Kaggle GPU retrofit: its status from the launcher's log, and its results once downloaded."""
    log = os.path.join(ROOT, KAGGLE_LOG)
    res = os.path.join(ROOT, KAGGLE_RESULTS)
    if not os.path.exists(log):
        return None, []
    text = open(log).read()
    lines = [ln for ln in text.splitlines() if "status" in ln]
    last = lines[-1] if lines else ""
    if os.path.exists(res):
        status = "done"
    elif "did not complete" in text or "ERROR" in last or "CANCEL" in last:
        status = "stopped"
    else:
        status = "running"
    run = {"run": "kaggle-retrofit", "experiment": "kaggle", "title": "Kaggle retrofit: Qwen2.5-1.5B",
           "model": "qwen-retrofit", "status": status, "step": None, "total_steps": None,
           "progress": 1.0 if status == "done" else None, "last_loss": None,
           "note": "LoRA + self-knowledge on a Kaggle T4 GPU" + (f" ({last.split()[0]} UTC check)" if last else "")}
    rows = []
    if os.path.exists(res):
        r = json.load(open(res))
        for model in ("base", "tuned"):
            for sname, label in (("trivia", "TriviaQA"), ("fictitious_absent", "Made-up names"),
                                 ("fictitious_in_context", "Made-up names, fact given")):
                d = r.get(model, {}).get(sname)
                if not d:
                    continue
                rows.append({"key": f"{model}/{sname}", "model": "Base" if model == "base" else "Tuned",
                             "set": label, "accuracy": d.get("accuracy"), "hallucination": d.get("hallucination_rate"),
                             "abstention": d.get("abstention_rate"), "auroc_softmax": d.get("auroc_softmax"),
                             "auroc_k": d.get("auroc_k_mean"), "auroc_probe": d.get("auroc_probe_last_layer")})
    return run, rows


# id, title, launcher log, result file (present once downloaded), note
DISTILL = [("kaggle-distill-data", "Kaggle kernel 1: teacher data (Qwen2.5-1.5B)", "results/kaggle/distill_data.log",
            "results/kaggle/glassbox-distill/data/distill/stats.json", "chat replies, TriviaQA self-check, made-up names; tokenizer"),
           ("kaggle-distill-train", "Kaggle kernel 2: Glassbox-2 vs GPT students", "results/kaggle/distill_train.log",
            "results/kaggle/glassbox-distill/student/baseline.json", "run 1 failed: Glassbox-2 student stalled at 7.3 nats, GPT student died loading data"),
           ("kaggle-distill-final2", "Kaggle final: Glassbox-2.3 (additive, calibrated k) vs GPT, best checkpoint",
            "results/kaggle/distill_final2.log", "results/kaggle/glassbox-distill-final2/student/baseline.json",
            "2.5 h each; weights with the best held-out loss are kept (4 h runs overfit the teacher data)"),
           ("kaggle-distill-g24", "Kaggle: Glassbox-2.4 student (in-model SAE objective), best checkpoint",
            "results/kaggle/distill_g24.log", "results/kaggle/glassbox-distill-g24/student/glassbox.json",
            "146M Glassbox-2.4; compared with the final run's GPT student"),
           ("kaggle-qwen-glassbox", "Kaggle: Qwen2.5-0.5B converted to a glass-box model",
            "results/kaggle/qwen-glassbox3.log", "results/kaggle/qwen-glassbox3/qwen_glassbox/results.json",
            "MLPs -> exactly sparse feature layers; kept exactness, lost two thirds of its TriviaQA accuracy"),
           ("kaggle-qwen-correctness", "Kaggle: correctness estimator at scale (Qwen2.5 0.5B / 1.5B / 3B)",
            "results/kaggle/qwen-correctness.log", "results/kaggle/qwen-correctness/correctness/results.json",
            "stack AUROC 0.859 (1.5B) / 0.807 (0.5B) vs softmax 0.72 / 0.69; NQ-open 0.78 unseen"),
           ("kaggle-qwen-correctness-3b", "Kaggle: correctness estimator for Qwen2.5-3B",
            "results/kaggle/qwen-correctness-3b.log", "results/kaggle/qwen-correctness-3b/correctness/results.json",
            "failed after 12 min: out of GPU memory in float32 (12.4 GB of 14.6); fixed by float16 + sdpa"),
           ("kaggle-claims-3b", "Kaggle (benski22): claim-level probe, any format, Qwen2.5-3B",
            "results/kaggle/qwen-claims-3b.log", "results/kaggle/qwen-claims-3b/claims/results.json",
            "all formats: stack 0.837 vs token p 0.740 (CIs disjoint); NQ unseen 0.762; held-out formats 0.74-0.82"),
           ("kaggle-claims-7b", "Kaggle (bensliv): claim-level probe, any format, Qwen2.5-7B on 2 GPUs",
            "results/kaggle/claims-7b.log", "results/kaggle/claims-7b/claims/results.json",
            "all formats: stack 0.856 vs token p 0.760; top-25% error 7% vs 18%; NQ 0.736"),
           ("kaggle-claims-r1-7b", "Kaggle (bensliv): claim-level probe on a reasoning model (DeepSeek-R1-Distill-Qwen-7B)",
            "results/kaggle/claims-r1-7b.log", "results/kaggle/claims-r1-7b/claims/results.json",
            "thinks in <think>; claims read from the final answer after </think>; smaller sets for the 12 h limit"),
           ("kaggle-detector-3b", "Kaggle (benski22): general hallucination detector, Qwen2.5-3B reader (scaled down)",
            "results/kaggle/detector-3b.log", "results/kaggle/detector-3b/detector/results.json",
            "RAGTruth + HaluEval x4 + TruthfulQA; token + response probes; leave-one-source-out; provenance vs Lookback Lens"),
           ("kaggle-detector-7b", "Kaggle (bensliv): general hallucination detector, Qwen2.5-7B reader",
            "results/kaggle/detector-7b.log", "results/kaggle/detector-7b/detector/results.json",
            "FAILED after the 3 h train pass: CUDA OOM fitting the token probe on the model's GPU (fixed)"),
           ("kaggle-detector-7b2", "Kaggle (bensliv): general detector, Qwen2.5-7B reader, rerun with HaluBench domains",
            "results/kaggle/detector-7b2.log", "results/kaggle/detector-7b2/detector/results.json",
            "trained and saved its head, then OOM in the test pass (fixed); evaluated from the head (detector-7b-eval)"),
           ("kaggle-detector-7b-eval", "Kaggle (bensliv): 7B detector evaluated from its saved head (test pass only)",
            "results/kaggle/detector-7b-eval.log", "results/kaggle/detector-7b-eval/detector/results.json",
            "10 sources, 4,127 test responses; provenance vs Lookback Lens on 300 document items; no LODO in this mode"),
           ("kaggle-detector-14b", "Kaggle (bensliv): general detector, Qwen2.5-14B reader (4-bit), 10 sources",
            "results/kaggle/detector-14b.log", "results/kaggle/detector-14b/detector/results.json",
            "reduced caps for the 12 h limit; scale point after 3B / 7B; no exact provenance (4-bit weights)"),
           ("kaggle-detector-adapt-3b", "Kaggle (bensliv): few-shot adaptation of the 3B detector + GPU throughput",
            "results/kaggle/detector-adapt-3b.log", "results/kaggle/detector-adapt-3b/adapt/results.json",
            "k = 0/25/50/100/200 labels per unseen HaluBench domain; judge and generation batching on a T4"),
           ("kaggle-risk-32b", "Kaggle (bensliv): Qwen3-32B (4-bit) learned risk head + claim probe",
            "results/kaggle/risk-32b.log", "results/kaggle/risk-32b/risk/results.json",
            "heuristic-free token/sentence risk from own replies + labelled spans + Math-Shepherd steps; launches when a slot frees"),
           ("kaggle-distill-train-add", "Kaggle kernel 2b: Glassbox-2.3 student (additive residuals)",
            "results/kaggle/distill_train_add.log", "results/kaggle/glassbox-distill-add/student/glassbox.json",
            "the pilot-2.3 architecture at 146M parameters; compared with kernel 2's GPT student")]


def distill_rows():
    rows = []
    for rid, title, log, res, note in DISTILL:
        lp = os.path.join(ROOT, log)
        if not os.path.exists(lp):
            continue
        text = open(lp).read()
        st = [ln for ln in text.splitlines() if "has status" in ln]
        last = st[-1] if st else ""
        if os.path.exists(os.path.join(ROOT, res)) or "COMPLETE" in last:
            status = "done"
        elif "ERROR" in last or "CANCEL" in last or "did not complete" in text:
            status = "stopped"
        else:
            status = "running"
        rows.append({"run": rid, "experiment": "kaggle", "title": title, "model": "kaggle", "status": status,
                     "step": None, "total_steps": None, "progress": 1.0 if status == "done" else None,
                     "last_loss": None, "note": note + (f" ({last.split()[0]} UTC check)" if last else "")})
    return rows


# id, title, log (JSON lines with "done"), result file, total, script, note
LOCAL = [("qwen-correctness", "CPU: exact attribution on unmodified Qwen2.5-0.5B vs answer correctness",
          "results/qwen_correctness.log", "results/qwen_correctness.json", 1000, "experiments/qwen_correctness.py",
          "SiLU gates fixed, no conversion; question-grounding number adds to softmax (+0.043), not to a probe"),
         ("qwen-correctness-entity", "CPU: the same signals read at the answer's entity token",
          "results/qwen_correctness_entity.log", "results/qwen_correctness_entity.json", 1000,
          "experiments/qwen_correctness_entity.py", "no lift: every detector 0.70-0.76 with 500 fitting answers"),
         ("recalibrate-shift", "CPU: recalibrating the correctness head on NQ-open from 200 labels",
          "results/recalibrate_shift.log", "results/recalibrate_shift.json", 600, "experiments/recalibrate_shift.py",
          "0.5B head trained on TriviaQA; ECE and released-answer accuracy before/after"),
         ("doubt-aware", "CPU: doubt-aware generation (stop before a doubted claim), 0.5B",
          "results/doubt_aware_eval.log", "results/doubt_aware_eval.json", 300, "experiments/doubt_aware_eval.py",
          "probe on the hidden state already computed vs token probability, at matched abstention"),
         ("doubt-aware-1p5b", "CPU: doubt-aware generation with the 1.5B probe",
          "results/doubt_aware_eval_1p5b.log", "results/doubt_aware_eval_1p5b.json", 300, "experiments/doubt_aware_eval.py",
          "the 0.5B probe tied token probability; does the stronger 1.5B probe cut hallucinations more?"),
         ("ragtruth-provenance", "CPU: exact per-token provenance on RAGTruth (sanity, 0.5B)",
          "results/ragtruth_provenance.log", "results/ragtruth_provenance.json", 40, "experiments/ragtruth_provenance.py",
          "do hallucinated spans draw less of their logit from the document?"),
         ("triggered-retrieval", "CPU: retrieve only when the probe doubts (1.5B, TriviaQA + Wikipedia)",
          "results/triggered_retrieval.log", "results/triggered_retrieval.json", 200, "experiments/triggered_retrieval.py",
          "accuracy per retrieval budget: probe trigger vs token probability (FLARE-style) vs random")]


def local_rows():
    rows = []
    for rid, title, log, res, total, script, note in LOCAL:
        lp = os.path.join(ROOT, log)
        if not os.path.exists(lp):
            continue
        done = [d["done"] for d in (json.loads(ln) for ln in open(lp) if ln.startswith("{\"") and ln.rstrip().endswith("}")) if "done" in d]
        if os.path.exists(os.path.join(ROOT, res)):
            status = "done"
        elif running(script):
            status = "running"
        else:
            status = "stopped"
        step = total if status == "done" else (done[-1] if done else 0)
        rows.append({"run": rid, "experiment": "qwen", "title": title, "model": "qwen", "status": status,
                     "step": step, "total_steps": total, "progress": step / total, "last_loss": None, "note": note})
    return rows


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
        elif st and running(res):
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
        if exp in ("chat", "chat2", "chat3", "g2", "g2.1", "g2.2", "g2.3", "g2.4", "g2.4s1", "gpt-s1", "g3"):
            for split, name in (("tokens_alpaca", "Alpaca"), ("tokens_wikitext", "WikiText-2")):
                r = result[split]
                chat.append({"model": model if exp == "chat" else f"{model}-{exp}",
                             "name": {"glassbox": "Glass-box", "glassbox2": "Glassbox-2", "glassbox3": "Glassbox-3"}.get(model, "GPT")
                             + {"chat2": " v2", "chat3": " v3", "g2.1": ".1", "g2.2": ".2", "g2.3": ".3", "g2.4": ".4",
                                "g2.4s1": ".4 seed 1", "gpt-s1": " seed 1"}.get(exp, ""),
                             "split": name,
                             "bits_per_token": r["bits_per_token"],
                             "top1_acc": r["top1_acc"], "auroc_msp": r["auroc_msp"], "auroc_probe": r["auroc_probe"],
                             "auroc_k": r.get("auroc_k")})
            for split, name in (("chat_facts", "Chat facts"), ("chat_facts_held_relations", "Chat, held-out relations")):
                r = result.get(split)
                if not r:
                    continue
                kinds = [k[:-6] for k in r if k.endswith("_right") and isinstance(r[k], float)]
                behaviour.append({"run": run_id, "experiment": exp, "model": model, "split": name,
                                  "accuracy": sum(r[f"{k}_right"] for k in kinds) / len(kinds),
                                  "hallucination": r.get("hallucination_rate"),
                                  "hallucination_fresh": r.get("hallucination_rate_fresh_names"),
                                  "ctx_acc": r.get("ctx_right"), "mem_acc": r.get("mem_right")})
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
    krun, kaggle = kaggle_rows()
    if krun:
        runs.append(krun)
    runs += distill_rows() + local_rows()
    for name, rows in (("runs", runs), ("curves", curves), ("behaviour", behaviour), ("detection", detection),
                       ("chat", chat), ("kaggle", kaggle)):
        json.dump(rows, open(os.path.join(out, f"{name}.json"), "w"), indent=0)
    json.dump({"exported_at": now}, open(os.path.join(out, "exported_at.json"), "w"))
    print(json.dumps({"exported_at": now, "runs": len(runs), "curves": len(curves), "behaviour": len(behaviour),
                      "detection": len(detection), "chat": len(chat), "kaggle": len(kaggle)}))


if __name__ == "__main__":
    main()
