#!/usr/bin/env python3
"""Derive the pack fingerprints the client library pins (``pgw.sdk.provisioning.PINNED_MODELS``).

For each model, packs the public Hugging Face checkpoint at the pinned commit exactly as the provider does
(``pgw pack <repo> <dir> --name <model> --revision <commit>``, default packing parameters), prints the fingerprint
and removes the pack. Compare with (or paste into) ``PINNED_MODELS``.

    PYTHONPATH=src python scripts/pin_model_fingerprints.py                       # every release model
    PYTHONPATH=src python scripts/pin_model_fingerprints.py qwen2.5-7b-instruct   # one model
    HF_HUB_OFFLINE=1 PYTHONPATH=src python scripts/pin_model_fingerprints.py      # checkpoints already cached

Needs the checkpoint (downloaded into the Hugging Face cache unless it is there) and disk for one pack at a time
(about 1 GB at 0.5B, 11 GB at 7B), in a temporary directory (``--work``).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path


def main() -> int:
    from pgw.modelpack import ModelPack, pack_checkpoint
    from pgw.sdk.provisioning import PINNED_MODELS

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="*", default=sorted(PINNED_MODELS), help="model names (default: all)")
    ap.add_argument("--work", type=Path, default=None, help="scratch directory (default: a temporary one)")
    args = ap.parse_args()
    out: dict[str, dict[str, str | bool | None]] = {}
    for name in args.models:
        pin = PINNED_MODELS[name]
        work = Path(tempfile.mkdtemp(prefix="pgw-pin-", dir=args.work))
        try:
            pack_checkpoint(pin.hf_repo, work / "pack", name=name, revision=pin.revision)
            with ModelPack.open(work / "pack", verify="fingerprint") as p:
                fp = p.fingerprint
        finally:
            shutil.rmtree(work, ignore_errors=True)
        out[name] = {"hf_repo": pin.hf_repo, "revision": pin.revision, "fingerprint": fp,
                     "baked": pin.fingerprint, "matches": pin.fingerprint == fp if pin.fingerprint else None}
        print(json.dumps({name: out[name]}), flush=True)
    return 1 if any(v["matches"] is False for v in out.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
