"""From a Kaggle CPU session (this sandbox cannot reach Modal): clone the branch, build the omni corpus, and run the
judge-internals experiment on Modal (deploy/modal_judge_internals.py). Outputs land in /kaggle/working/ji_out.
The Modal token is passed as arguments at push time (never committed; masked in the log).

    python kaggle/kgrun.py --cpu --slug omni-ji-modal --script deploy/kaggle_modal_run.py \
        --args "--branch claude/keen-euler-0kccyk --id <token id> --secret <token secret> [--ji-args '--per-domain 6']"
"""
import argparse
import os
import shutil
import subprocess
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--branch", required=True)
ap.add_argument("--id", required=True)
ap.add_argument("--secret", required=True)
ap.add_argument("--ji-args", default="")
a = ap.parse_args()
work = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
repo = os.path.join(work, "repo")
subprocess.run(["git", "clone", "-q", "--depth", "1", "-b", a.branch,
                "https://github.com/benski20/llm_training_exploration", repo], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "modal==1.6.3"], check=True)
os.makedirs("/tmp/omni", exist_ok=True)
r = subprocess.run([sys.executable, "experiments/omni_corpus.py", "build", "--out", "/tmp/omni/corpus.jsonl"], cwd=repo,
                   capture_output=True, text=True)
print("corpus build", r.returncode, r.stdout[-1500:], r.stderr[-1500:], flush=True)
env = dict(os.environ, MODAL_TOKEN_ID=a.id, MODAL_TOKEN_SECRET=a.secret)
cmd = ["modal", "run", "deploy/modal_judge_internals.py"] + (["--args", a.ji_args] if a.ji_args else [])
p = subprocess.run(cmd, cwd=repo, env=env, capture_output=True, text=True)
log = (p.stdout + p.stderr).replace(a.secret, "***").replace(a.id, "***")
print(log[-15000:], flush=True)
open(os.path.join(work, "modal_run.log"), "w").write(log)
if os.path.isdir(os.path.join(repo, "ji_out")):
    shutil.copytree(os.path.join(repo, "ji_out"), os.path.join(work, "ji_out"), dirs_exist_ok=True)
shutil.rmtree(repo, ignore_errors=True)
print("modal run exit code", p.returncode)
