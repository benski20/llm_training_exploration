#!/usr/bin/env python3
"""Derive the pack fingerprints the client library pins (``pgw.sdk.provisioning.PINNED_MODELS``).

For each model, packs the public Hugging Face checkpoint at the pinned commit exactly as the provider does
(``pgw pack <repo> <dir> --name <model> --revision <commit>``, default packing parameters: pack format 2), prints
the fingerprint and removes the pack. Compare with (or paste into) ``PINNED_MODELS``.

    PYTHONPATH=src python scripts/pin_model_fingerprints.py                       # every release model
    PYTHONPATH=src python scripts/pin_model_fingerprints.py qwen2.5-7b-instruct   # one model
    HF_HUB_OFFLINE=1 PYTHONPATH=src python scripts/pin_model_fingerprints.py      # checkpoints already cached
    PYTHONPATH=src python scripts/pin_model_fingerprints.py --compare-format1     # + the exactness evidence

``--compare-format1`` also packs the checkpoint in format 1 (``plain_dtype="float32"``, the packs pinned before
2026-11-26), checks that it reproduces the format-1 pin (:data:`FORMAT1_PINS`: same bytes as the old packer), and
compares the two packs: the dtype of each plain tensor in the checkpoint and in the format-2 pack, every plain
tensor as loaded (float32) bit for bit, and every linear tensor entry (dtype, shape, SHA-256). Needs disk for two
packs at a time (about 2 GB at 0.5B, 22 GB at 7B).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

#: The pins of pack format 1 (plain tensors in float32), released before pack format 2. Used only to show that
#: ``plain_dtype="float32"`` still reproduces them; the client library pins format 2.
FORMAT1_PINS: dict[str, str] = {
    "qwen2.5-0.5b-instruct": "915e55c19364542c4fd8ab5c8f7154806e2511048755e0d10426374b5c8fe9c7",
    "qwen2.5-1.5b-instruct": "81bb7adcea1524171297c0004c0efb1edad9f71dd0535462c9b32ab03ac99348",
    "qwen2.5-3b-instruct": "a8e7bac934d82f6435fe9385bd1eb3cfdf1a68f4ff415829e49a3dd4f23bcd23",
    "qwen2.5-7b-instruct": "d3db0ca858981a66dd262143d7f3d5b0120069068e0a98a806e1eea3f12653fa",
}


def _compare(p1: Any, p2: Any, ckpt: Path) -> dict[str, Any]:
    """Exactness evidence: format-1 pack ``p1`` vs format-2 pack ``p2`` of the checkpoint ``ckpt``."""
    import torch

    from pgw.modelpack.arch import hf_plain_tensor_names
    from pgw.modelpack.packer import CheckpointReader

    m1, m2 = p1.manifest, p2.manifest
    hf = hf_plain_tensor_names(m2.arch.num_layers, m2.arch.tie_word_embeddings)
    per: dict[str, Any] = {}
    all_equal = True
    with CheckpointReader(ckpt) as rd:
        src_dtypes = {n: str(rd.dtype(h)).replace("torch.", "") for n, h in hf.items()}
    for n in m2.plain_tensor_names():
        a, b = p1.load_plain(n), p2.load_plain(n)
        eq = a.dtype == b.dtype == torch.float32 and a.shape == b.shape and bool(
            a.view(torch.int32).equal(b.view(torch.int32)))
        all_equal &= eq
        if n in ("embed_tokens", "lm_head") or not eq:
            per[n] = {"source": src_dtypes[n], "format1": m1.tensors[n].dtype, "format2": m2.tensors[n].dtype,
                      "elements": int(a.numel()), "bit_identical_f32": eq}
        del a, b
    norms = [n for n in m2.plain_tensor_names() if n not in ("embed_tokens", "lm_head")]
    lin = [t for e in m1.linears for t in (e.weight, e.scale, e.bias) if t is not None]
    return {
        "source_dtypes": sorted(set(src_dtypes.values())),
        "plain": per,
        "norms": {"count": len(norms), "source": sorted({src_dtypes[n] for n in norms}),
                  "stored": sorted({m2.tensors[n].dtype for n in norms})},
        "all_plain_bit_identical_f32": all_equal,
        "linear_entries_identical": all(m1.tensors[t] == m2.tensors[t] for t in lin),
        "plain_bytes": {"format1": p1.size_bytes("plain"), "format2": p2.size_bytes("plain")},
    }


def main() -> int:
    from pgw.modelpack import ModelPack, pack_checkpoint
    from pgw.modelpack.packer import _resolve_source
    from pgw.sdk.provisioning import PINNED_MODELS

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="*", default=sorted(PINNED_MODELS), help="model names (default: all)")
    ap.add_argument("--work", type=Path, default=None, help="scratch directory (default: a temporary one)")
    ap.add_argument("--compare-format1", action="store_true",
                    help="also pack format 1, check it against FORMAT1_PINS and compare the packs bit for bit")
    args = ap.parse_args()
    out: dict[str, dict[str, Any]] = {}
    for name in args.models:
        pin = PINNED_MODELS[name]
        work = Path(tempfile.mkdtemp(prefix="pgw-pin-", dir=args.work))
        rec: dict[str, Any] = {"hf_repo": pin.hf_repo, "revision": pin.revision}
        try:
            pack_checkpoint(pin.hf_repo, work / "pack", name=name, revision=pin.revision)
            with ModelPack.open(work / "pack", verify="full") as p:
                fp = p.fingerprint
                rec["format"] = p.manifest.version
            if args.compare_format1:
                pack_checkpoint(pin.hf_repo, work / "pack1", name=name, revision=pin.revision, plain_dtype="float32")
                ckpt, _ = _resolve_source(pin.hf_repo, pin.revision)
                with ModelPack.open(work / "pack1", verify="full") as p1, \
                        ModelPack.open(work / "pack", verify="fingerprint") as p2:
                    rec["format1_fingerprint"] = p1.fingerprint
                    rec["format1_matches_old_pin"] = (p1.fingerprint == FORMAT1_PINS[name]
                                                      if name in FORMAT1_PINS else None)
                    rec["exactness"] = _compare(p1, p2, ckpt)
        finally:
            shutil.rmtree(work, ignore_errors=True)
        rec.update(fingerprint=fp, baked=pin.fingerprint, matches=pin.fingerprint == fp if pin.fingerprint else None)
        out[name] = rec
        print(json.dumps({name: rec}), flush=True)
    bad = any(v["matches"] is False or v.get("format1_matches_old_pin") is False
              or (v.get("exactness") and not (v["exactness"]["all_plain_bit_identical_f32"]
                                              and v["exactness"]["linear_entries_identical"]))
              for v in out.values())
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
