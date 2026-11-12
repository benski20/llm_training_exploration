"""Relay: drive a SECOND Kaggle account from inside a Kaggle session (where no proxy rewrites the API login).

Run as a private CPU kernel of the main account (kaggle/kgrun.py --cpu); the second account's API token is passed as an
argument at push time, never committed. Actions:
    push    create / update a kernel on the second account that clones this repository at --branch and runs
            --entry with --entry-args (GPU, internet on)
    status  print the status of one or more of the second account's kernels
    output  download a kernel's outputs into /kaggle/working/out/<slug> (so the main account can fetch them)

    python kaggle/kgrun.py --cpu --slug relay-push-0 --script kaggle/relay.py \
        --args "--token <tok> --action push --slug probe14-0 --entry kaggle/probe/extract.py --entry-args '...'"
"""
import argparse
import json
import os
import shlex
import shutil
import tempfile

BOOT = '''import subprocess, sys
subprocess.run(["git", "clone", "-q", "--depth", "1", "-b", {branch!r},
                "https://github.com/benski20/llm_training_exploration", "/kaggle/working/repo"], check=True)
subprocess.run([sys.executable, "/kaggle/working/repo/" + {entry!r}] + {args!r}, check=True)
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--token", required=True)
    ap.add_argument("--action", choices=["push", "status", "output"], required=True)
    ap.add_argument("--slug", nargs="+", required=True)
    ap.add_argument("--branch", default="claude/keen-euler-0kccyk")
    ap.add_argument("--entry", default="kaggle/probe/extract.py")
    ap.add_argument("--entry-args", default="")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--expect-user", default="")
    a = ap.parse_args()
    for k in [k for k in os.environ if k.startswith("KAGGLE_API") or k in ("KAGGLE_USERNAME", "KAGGLE_KEY")]:
        os.environ.pop(k)                                   # the session's own token would win over ours
    os.environ["KAGGLE_API_TOKEN"] = a.token
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    user = api.config_values.get("username")
    print("second account:", user, flush=True)
    if a.expect_user and user != a.expect_user:
        raise SystemExit(f"authenticated as {user}, expected {a.expect_user}")
    for slug in a.slug:
        ref = f"{user}/{slug}"
        if a.action == "push":
            d = tempfile.mkdtemp()
            open(os.path.join(d, "run.py"), "w").write(
                BOOT.format(branch=a.branch, entry=a.entry, args=shlex.split(a.entry_args)))
            meta = {"id": ref, "title": slug, "code_file": "run.py", "language": "python", "kernel_type": "script",
                    "is_private": True, "enable_gpu": not a.cpu, "enable_internet": True, "dataset_sources": [],
                    "competition_sources": [], "kernel_sources": [], "model_sources": []}
            if not a.cpu:
                meta["machine_shape"] = "NvidiaTeslaT4"
            json.dump(meta, open(os.path.join(d, "kernel-metadata.json"), "w"))
            r = api.kernels_push(d)
            print("pushed", ref, getattr(r, "url", "") or "", getattr(r, "error", "") or "", flush=True)
            shutil.rmtree(d)
        elif a.action == "status":
            s = api.kernels_status(ref)
            print("status", ref, getattr(s, "status", s), getattr(s, "failure_message", "") or "", flush=True)
        else:
            out = os.path.join("/kaggle/working/out", slug)
            os.makedirs(out, exist_ok=True)
            api.kernels_output(ref, out)
            n = sum(len(f) for _, _, f in os.walk(out))
            print("downloaded", ref, n, "files", flush=True)


if __name__ == "__main__":
    main()
