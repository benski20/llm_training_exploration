"""Deploy the VeriChat demo to Modal from a Kaggle CPU session (no GPU quota), for when GitHub Actions is unavailable.
The public repository is cloned at a branch and `modal deploy` runs there. The Modal token is passed as arguments at
push time (never committed); the kernel is private, and the token should be rotated afterwards.

    python kaggle/kgrun.py --cpu --slug verichat-deploy --script deploy/kaggle_deploy.py \
        --args "--branch claude/keen-euler-0kccyk --omni heads/omni_qwen3_8b.npz --id <token id> --secret <token secret>" --wait
"""
import argparse
import os
import subprocess
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--branch", required=True)
ap.add_argument("--omni", default="")
ap.add_argument("--set", default="", help="extra VERICHAT_* settings, comma-separated K=V")
ap.add_argument("--id", required=True)
ap.add_argument("--secret", required=True)
a = ap.parse_args()
repo = "/kaggle/working/repo" if os.path.isdir("/kaggle/working") else "repo"
subprocess.run(["git", "clone", "-q", "--depth", "1", "-b", a.branch,
                "https://github.com/benski20/llm_training_exploration", repo], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "modal==1.6.3"], check=True)
env = dict(os.environ, MODAL_TOKEN_ID=a.id, MODAL_TOKEN_SECRET=a.secret, VERICHAT_OMNI=a.omni,
           **dict(kv.split("=", 1) for kv in a.set.split(",") if kv.startswith("VERICHAT_")))
r = subprocess.run(["modal", "deploy", "deploy/modal_app.py"], cwd=repo, env=env, capture_output=True, text=True)
out = (r.stdout + r.stderr).replace(a.secret, "***").replace(a.id, "***")
print(out[-6000:])
open(os.path.join(os.path.dirname(repo), "deploy.log"), "w").write(out)
print("modal deploy exit code", r.returncode)
sys.exit(r.returncode)
