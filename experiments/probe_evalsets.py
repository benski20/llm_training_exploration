"""Evaluation sets for the large-scale grounding probe, in the same schema as experiments/probe_data.py (never used for
training; probe_data.py excludes them by response hash):

    ragtruth_test   the official RAGTruth test split (2,700 answers, span labels) - the published benchmark
    halubench       PatronusAI/HaluBench (CC-BY-NC: evaluation only), all sources, answer labels
    business_test   data/business_test.jsonl: realistic fictional business documents with subtle single errors

    python experiments/probe_evalsets.py --out data/probe_eval
"""
import argparse
import json
import os


def ragtruth_test():
    from datasets import load_dataset
    out = []
    for x in load_dataset("wandb/RAGTruth-processed", split="test"):  # nosec B615: benchmark data
        sp = [[int(h["start"]), int(h["end"])] for h in json.loads(x["hallucination_labels"] or "[]")]
        q = x["query"] or ""
        if x["task_type"] == "Summary" and not q:
            q = "Summarize the following news."
        if x["task_type"] == "Data2txt" and not q:
            q = "Write an objective overview of this business based on the structured data."
        out.append({"id": f"rtt_{x['id']}", "source": "ragtruth_test", "task": x["task_type"].lower(),
                    "question": q, "documents": [x["context"]], "response": x["output"],
                    "label": int(bool(sp)), "spans": sp, "has_spans": True, "label_origin": "human",
                    "license": "MIT"})
    return out


def halubench():
    from datasets import load_dataset
    out = []
    for k, x in enumerate(load_dataset("PatronusAI/HaluBench", split="test")):  # nosec B615
        out.append({"id": f"hb_{k}", "source": f"halubench_{x['source_ds']}", "task": "qa", "question": x["question"],
                    "documents": [x["passage"]], "response": x["answer"], "label": int(x["label"] == "FAIL"),
                    "spans": [], "has_spans": False, "label_origin": "human/synthetic",
                    "license": "CC-BY-NC-4.0 (evaluation only)"})
    import random
    by = {}
    for r in out:
        by.setdefault(r["source"], []).append(r)
    rng = random.Random(0)                                     # 350 per source keeps every source, at a CPU budget
    return [r for src in sorted(by) for r in rng.sample(by[src], min(350, len(by[src])))]


def business_test(path="data/business_test.jsonl"):
    out = []
    if not os.path.exists(path):
        return out
    for l in open(path):
        x = json.loads(l)
        out.append({"id": x["id"], "source": "business_test", "task": x.get("domain", "business"),
                    "question": x["question"], "documents": x["documents"], "response": x["response"],
                    "label": int(x["label"]), "spans": x.get("spans") or [], "has_spans": True,
                    "error_type": x.get("error_type"), "label_origin": "authored", "license": "repository"})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/probe_eval")
    ap.add_argument("--sets", default="ragtruth_test,halubench,business_test")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fns = {"ragtruth_test": ragtruth_test, "halubench": halubench, "business_test": business_test}
    for name in a.sets.split(","):
        recs = sorted(fns[name](), key=lambda r: r["id"])
        recs = [r for r in recs if sum(len(d) for d in r["documents"]) <= 12000 and len(r["response"]) <= 4000]
        with open(os.path.join(a.out, f"{name}.jsonl"), "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        print(name, len(recs), "hallucinated", sum(r["label"] for r in recs), flush=True)


if __name__ == "__main__":
    main()
