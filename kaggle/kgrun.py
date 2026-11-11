"""Run a repo script as a private Kaggle GPU kernel (free T4 x2 / P100), with repo files bundled in, and fetch its
outputs. The API token never touches this machine's disk or environment: in this cloud environment the network
proxy injects it into requests to api.kaggle.com, so the client's own token check (which sends the token in a request
body) is skipped and every call is authenticated by the proxy.

    python kaggle/kgrun.py --slug omni-lora --script kaggle/omni/lora.py \
        --bundle vt/omni.py experiments/omni_corpus.py --out results/kaggle/omni-lora [--args "--epochs 2"] [--wait]
    python kaggle/kgrun.py --slug omni-lora --status          # status only
    python kaggle/kgrun.py --slug omni-lora --fetch --out results/kaggle/omni-lora

Inside the kernel the bundled files sit under /kaggle/working/src (on sys.path), outputs go to /kaggle/working.
Kaggle allows only a few GPU sessions at once per account; a push beyond that queues or fails (see --status).
"""
import argparse
import base64
import json
import os
import shutil
import sys
import tempfile
import time

USER = os.environ.get("KAGGLE_USERNAME", "bensliv")
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def api():
    os.environ.setdefault("KAGGLE_API_TOKEN", "proxy-injected")      # the proxy replaces the header
    from kaggle.api.kaggle_api_extended import KaggleApi
    KaggleApi._introspect_token = lambda self, t: USER                 # body-token check cannot see the proxy
    a = KaggleApi()
    a.authenticate()
    return a


PRELUDE = '''# --- bundled by kaggle/kgrun.py ---
import base64 as _b64, os as _os, sys as _sys
_SRC = "/kaggle/working/src"
for _p, _c in {files!r}.items():
    _os.makedirs(_os.path.dirname(_os.path.join(_SRC, _p)), exist_ok=True)
    open(_os.path.join(_SRC, _p), "wb").write(_b64.b64decode(_c))
_sys.path[:0] = [_SRC, _os.path.join(_SRC, "experiments"), _os.path.join(_SRC, "kaggle")]
_sys.argv = [_sys.argv[0]] + {args!r}
# --- end bundle ---
'''


def push(a):
    files = {}
    for p in a.bundle or []:
        files[p] = base64.b64encode(open(os.path.join(ROOT, p), "rb").read()).decode()
    body = open(os.path.join(ROOT, a.script)).read()
    script = PRELUDE.format(files=files, args=a.args.split() if a.args else []) + body
    stage = tempfile.mkdtemp()
    try:
        open(os.path.join(stage, "run.py"), "w").write(script)
        meta = {"id": f"{USER}/{a.slug}", "title": a.slug, "code_file": "run.py", "language": "python",
                "kernel_type": "script", "is_private": True, "enable_gpu": not a.cpu, "enable_internet": True,
                **({} if a.cpu else {"machine_shape": a.machine}), "dataset_sources": [], "competition_sources": [], "kernel_sources": [],
                "model_sources": []}
        json.dump(meta, open(os.path.join(stage, "kernel-metadata.json"), "w"))
        r = api().kernels_push(stage)
        print("pushed", f"{USER}/{a.slug}", getattr(r, "url", "") or "", getattr(r, "error", "") or "", flush=True)
    finally:
        shutil.rmtree(stage)


def status(a):
    s = api().kernels_status(f"{USER}/{a.slug}")
    st = str(getattr(s, "status", s))
    print(st, getattr(s, "failure_message", "") or "", flush=True)
    return st


def fetch(a):
    os.makedirs(a.out, exist_ok=True)
    api().kernels_output(f"{USER}/{a.slug}", a.out)
    print("outputs in", a.out, os.listdir(a.out), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", required=True)
    ap.add_argument("--script")
    ap.add_argument("--bundle", nargs="*")
    ap.add_argument("--args", default="")
    ap.add_argument("--out")
    ap.add_argument("--machine", default="NvidiaTeslaT4", help="NvidiaTeslaT4 (2x T4) or NvidiaTeslaP100")
    ap.add_argument("--cpu", action="store_true", help="a CPU-only session (no GPU quota used)")
    ap.add_argument("--wait", action="store_true", help="poll until done, then fetch outputs")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--fetch", action="store_true")
    a = ap.parse_args()
    if a.status:
        return status(a)
    if a.fetch:
        return fetch(a)
    push(a)
    if a.wait:
        while True:
            time.sleep(60)
            st = status(a).lower()
            if any(k in st for k in ("complete", "error", "cancel")):
                break
        if a.out:
            fetch(a)


if __name__ == "__main__":
    sys.exit(main())
