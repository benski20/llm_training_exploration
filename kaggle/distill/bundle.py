"""Make a single-file Kaggle kernel: embed vt modules and an entry script, run the entry script.

    python kaggle/distill/bundle.py OUT ENTRY [args baked into the entry...]
"""
import base64
import os
import sys

MODULES = ["vt/model.py", "vt/lm.py", "vt/glassbox.py", "vt/glassbox2.py", "vt/glassbox3.py", "vt/qwen_glassbox.py"]


def main():
    out, entry, baked = sys.argv[1], sys.argv[2], sys.argv[3:]
    root = os.path.join(os.path.dirname(__file__), "..", "..")
    files = {m: base64.b64encode(open(os.path.join(root, m), "rb").read()).decode() for m in MODULES}
    files["vt/__init__.py"] = ""
    src = base64.b64encode(open(entry, "rb").read()).decode()
    with open(out, "w") as f:
        f.write(f'''"""Bundled by kaggle/distill/bundle.py: {os.path.basename(entry)} plus vt modules."""
import base64, os, runpy, sys
HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
WORK = "/kaggle/working/bundle" if os.path.isdir("/kaggle/working") else os.path.join(HERE, "bundle")
FILES = {files!r}
for rel, b64 in FILES.items():
    p = os.path.join(WORK, "vtpkg", rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "wb").write(base64.b64decode(b64))
entry = os.path.join(WORK, {os.path.basename(entry)!r})
open(entry, "wb").write(base64.b64decode({src!r}))
sys.argv = [entry] + (sys.argv[1:] or {baked!r})
runpy.run_path(entry, run_name="__main__")
''')


if __name__ == "__main__":
    main()
