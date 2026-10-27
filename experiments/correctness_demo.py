"""What a served correctness estimator replies: package a Hugging Face model with its trained head
(kaggle/correctness) through the production path, then ask it questions.

    python experiments/correctness_demo.py --model Qwen/Qwen2.5-1.5B-Instruct \\
        --heads results/kaggle/qwen-correctness/correctness/Qwen2.5-1.5B-Instruct_heads.pt --dir <snapshot dir>
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vt.chat_serve import export_correctness_package, load_responder  # noqa: E402

QUESTIONS = [
    "What is the capital of Australia?",
    "Who wrote the novel Pride and Prejudice?",
    "Who was the man behind The Chipmunks?",                       # TriviaQA: Ross Bagdasarian
    "Which Lloyd Webber musical premiered in the US on 10th December 1993?",   # Sunset Boulevard
    "Who was the first president of the Republic of Zanthoria?",   # made up
    "Who wrote he ain't heavy he's my brother lyrics?",            # NQ: Bobby Scott and Bob Russell
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--heads", required=True)
    ap.add_argument("--dir", required=True, help="where to snapshot the model (outside the repo)")
    ap.add_argument("--variant", default="all")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out", default="results/correctness_demo.json")
    a = ap.parse_args()
    from huggingface_hub import snapshot_download
    snapshot_download(a.model, local_dir=a.dir, allow_patterns=["*.json", "*.safetensors", "*.txt", "merges.txt"])
    man = export_correctness_package(a.dir, a.heads, a.variant, a.threshold)
    r = load_responder(a.dir)
    out = {"model": a.model, "variant": a.variant, "threshold": a.threshold,
           "package_sha256": man["package_sha256"], "replies": []}
    for q in QUESTIONS:
        t0 = time.time()
        rep = r.reply(q)
        rep = {"question": q, **rep, "seconds": round(time.time() - t0, 1)}
        out["replies"].append(rep)
        print(json.dumps({k: rep[k] for k in ("question", "text", "p_correct", "evidence", "seconds")}), flush=True)
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
