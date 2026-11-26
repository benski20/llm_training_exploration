#!/usr/bin/env python3
"""Derive the pack fingerprints the client library pins (``pgw.sdk.provisioning.PINNED_MODELS``).

For each model, packs the public Hugging Face checkpoint at the pinned commit exactly as the provider does
(``pgw pack <repo> <dir> --name <model> --revision <commit>``, default packing parameters: pack format 2), prints
the fingerprint and removes the pack. Compare with (or paste into) ``PINNED_MODELS``.

The default packer is the shard-wise one (:func:`pgw.modelpack.shardwise.stream_pack_checkpoint`): one checkpoint
shard on disk at a time (download, pack, delete), in hash-only mode (the manifest and fingerprint are computed from
the bytes the pack would hold; nothing is written), so 72B needs about 4.5 GB of scratch disk instead of 145 GB for
the checkpoint plus 75 GB for the pack. It is byte-identical to ``pack_checkpoint`` (tests/modelpack/test_shardwise.py).

    PYTHONPATH=src python scripts/pin_model_fingerprints.py                       # every release model
    PYTHONPATH=src python scripts/pin_model_fingerprints.py qwen2.5-72b-instruct  # one model
    HF_HUB_OFFLINE=1 PYTHONPATH=src python scripts/pin_model_fingerprints.py      # checkpoints already cached
    PYTHONPATH=src python scripts/pin_model_fingerprints.py --plan qwen2.5-72b-instruct   # sizes from the headers
    PYTHONPATH=src python scripts/pin_model_fingerprints.py --write-pack ...      # write the pack, verify in full
    PYTHONPATH=src python scripts/pin_model_fingerprints.py --compare-packer ...  # + pack_checkpoint, same bytes
    PYTHONPATH=src python scripts/pin_model_fingerprints.py --compare-format1     # + the format-2 exactness evidence

``--write-pack`` writes the pack shard-wise and opens it with ``verify="full"`` (every byte re-hashed). Needs disk
for one shard plus the pack. ``--compare-packer`` also packs the whole checkpoint with ``pack_checkpoint`` and
requires every file byte-identical and the same manifest (needs the whole checkpoint plus two packs).

``--compare-format1`` (whole-checkpoint packer) also packs the checkpoint in format 1 (``plain_dtype="float32"``,
the packs pinned before 2026-11-26), checks that it reproduces the format-1 pin (:data:`FORMAT1_PINS`: same bytes
as the old packer), and compares the two packs: the dtype of each plain tensor in the checkpoint and in the format-2
pack, every plain tensor as loaded (float32) bit for bit, and every linear tensor entry (dtype, shape, SHA-256).
Needs disk for two packs at a time (about 2 GB at 0.5B, 22 GB at 7B).
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


def _digests(root: Path) -> dict[str, str]:
    import hashlib

    out = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.name != "manifest.json":
            h = hashlib.sha256()
            with open(p, "rb") as f:
                while b := f.read(8 << 20):
                    h.update(b)
            out[str(p.relative_to(root))] = h.hexdigest()
    return out


def _gb(x: float) -> float:
    return round(float(x) / 1e9, 3)


def main() -> int:
    from pgw.modelpack import Manifest, ModelPack, pack_checkpoint
    from pgw.modelpack.packer import _resolve_source
    from pgw.modelpack.shardwise import plan_summary, stream_pack_checkpoint
    from pgw.sdk.provisioning import PINNED_MODELS

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="*", default=sorted(PINNED_MODELS), help="model names (default: all)")
    ap.add_argument("--work", type=Path, default=None, help="scratch directory (default: a temporary one)")
    ap.add_argument("--packer", choices=("shardwise", "whole"), default="shardwise",
                    help="shard-wise streaming packer (default) or pack_checkpoint on the whole checkpoint")
    ap.add_argument("--write-pack", action="store_true",
                    help="shard-wise: write the pack and verify it in full (default: hash only, nothing written)")
    ap.add_argument("--compare-packer", action="store_true",
                    help="also pack with pack_checkpoint (whole checkpoint) and require byte-identical files")
    ap.add_argument("--prefetch", type=int, default=0, help="shard-wise: shards downloaded ahead (disk: +1 shard)")
    ap.add_argument("--plan", action="store_true", help="only print the header-only plan (sizes; no download)")
    ap.add_argument("--compare-format1", action="store_true",
                    help="also pack format 1, check it against FORMAT1_PINS and compare the packs bit for bit")
    args = ap.parse_args()
    out: dict[str, dict[str, Any]] = {}
    for name in args.models:
        pin = PINNED_MODELS[name]
        work = Path(tempfile.mkdtemp(prefix="pgw-pin-", dir=args.work))
        rec: dict[str, Any] = {"hf_repo": pin.hf_repo, "revision": pin.revision}
        try:
            if args.plan:
                s = plan_summary(pin.hf_repo, work_dir=work / "plan", revision=pin.revision, prefetch=args.prefetch)
                print(json.dumps({name: {k: (_gb(v) if k.endswith("bytes") else v) for k, v in s.items()
                                         if k != "shards"} | {"shards": len(s["shards"])}}), flush=True)
                continue
            if args.packer == "whole" or args.compare_format1:
                pack_checkpoint(pin.hf_repo, work / "pack", name=name, revision=pin.revision)
                with ModelPack.open(work / "pack", verify="full") as p:
                    fp = p.fingerprint
                    rec["format"] = p.manifest.version
                rec["packer"] = "whole"
            else:
                res = stream_pack_checkpoint(pin.hf_repo, work / "pack" if args.write_pack else None,
                                             work_dir=work / "stream", name=name, revision=pin.revision,
                                             prefetch=args.prefetch)
                fp = res.fingerprint
                rec["format"] = res.manifest.version
                rec["packer"] = "shardwise" + ("" if args.write_pack else " (hash only)")
                st = res.stats
                rec["stats"] = {"shards": st["shards"], "fetched_gb": _gb(st["fetched_bytes"]),
                                "peak_scratch_gb": _gb(st["peak_work_bytes"]),
                                "spill_peak_gb": _gb(st["spill_peak_bytes"]),
                                "peak_rss_gb": _gb(st["peak_rss_bytes"]),
                                "peak_rss_anon_gb": _gb(st["peak_rss_anon_bytes"]),
                                "fetch_s": round(st["fetch_s"], 1), "compute_s": round(st["compute_s"], 1),
                                "seconds": round(st["seconds"], 1)}
                if args.write_pack:
                    with ModelPack.open(work / "pack", verify="full", expected_fingerprint=fp) as p:
                        rec["verified_full"] = p.fingerprint == fp
                if args.compare_packer:
                    stream_dig = _digests(work / "pack") if args.write_pack else None
                    stream_man = json.loads(res.manifest.to_json())
                    shutil.rmtree(work / "pack", ignore_errors=True)
                    pack_checkpoint(pin.hf_repo, work / "whole", name=name, revision=pin.revision)
                    whole = json.loads(Manifest.from_file(work / "whole" / "manifest.json").to_json())
                    stream_man.pop("created_at", None), whole.pop("created_at", None)
                    rec["compare_packer"] = {
                        "manifest_identical": stream_man == whole,
                        "files_identical": (stream_dig == _digests(work / "whole")) if stream_dig is not None
                        else None}
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
              or v.get("verified_full") is False
              or (v.get("compare_packer") and False in v["compare_packer"].values())
              or (v.get("exactness") and not (v["exactness"]["all_plain_bit_identical_f32"]
                                              and v["exactness"]["linear_entries_identical"]))
              for v in out.values())
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
