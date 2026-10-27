"""Side-by-side chat results of the glass-box pilots and the matched GPT (same chat-v2 data).

    python experiments/chat_compare.py
"""
import json
import os

ROOT = os.path.join(os.path.dirname(__file__), "..")
RUNS = [("GPT", "results/glassbox_chat_v2/baseline.json", "baseline"),
        ("Glass-box v2", "results/glassbox_chat_v2/glassbox.json", "glassbox"),
        ("Glassbox-2", "results/glassbox2_chat/glassbox2.json", "glassbox2"),
        ("Glassbox-2.1", "results/glassbox2_1_chat/glassbox2.json", "glassbox2"),
        ("Glassbox-2.2", "results/glassbox2_2_chat/glassbox2.json", "glassbox2"),
        ("Glassbox-3", "results/glassbox3_chat/glassbox3.json", "glassbox3")]
DETECTORS = ("auroc_msp", "auroc_neg_nll", "auroc_probe", "auroc_k_min", "auroc_neg_u_first", "auroc_neg_p_unsure_first")


def row(name, r, split):
    c = r[split]
    det = {k: c[k] for k in DETECTORS if c.get(k) is not None}
    best = max(det, key=det.get) if det else None
    return {"model": name, "split": split, "bits_alpaca": round(r["tokens_alpaca"]["bits_per_token"], 3),
            "halluc_unstated": round(c["hallucination_rate"], 3),
            "halluc_made_up": round(c["hallucination_rate_fresh_names"], 3),
            "over_refusal": round(c["over_refusal_rate"], 3), "ctx_right": round(c["ctx_right"], 3),
            "best_detector": best, "best_auroc": round(det[best], 3) if best else None,
            "unsure_emitted": c.get("unsure_emitted")}


def main():
    for split in ("chat_facts", "chat_facts_held_relations"):
        for name, path, key in RUNS:
            p = os.path.join(ROOT, path)
            if os.path.exists(p) and key in json.load(open(p)):
                print(json.dumps(row(name, json.load(open(p))[key], split)))


if __name__ == "__main__":
    main()
