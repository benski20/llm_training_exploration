"""Client library (``pgw.OpenAI``) on CPU against a real ``pgw service`` process on loopback.

Starts the provider of the SDK tests (TLS, TCP+TLS trip lane, masked LM head, test SDK enrollment; CPU kernels),
then measures streamed chats through the library with its default stack and, for comparison, with the Python trip
path (``native=False``) and the WebSocket stream lane (``lane="stream"``), each variant in its own process (the
native switches are process-wide). Every variant must give the same tokens and logprobs (checked). Reports the
first-request time (enroll + provision + connect + first masks), TTFT and decode tokens/s (warm chats; medians).

    PYTHONPATH=src:. python benchmarks/sdk_cpu_bench.py --model qwen0.5b --tokens 48 --runs 3

The provider computes on the same CPU, so the numbers measure the whole loop on one box (provider GEMMs
included), not a GPU provider: compare variants, not GPU runs.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
MODEL = "qwen2.5-0.5b-instruct"
PROMPT = [{"role": "user", "content": "Write one sentence about the sea."}]
VARIANTS: dict[str, dict[str, Any]] = {"default": {}, "python_trips": {"native": False},
                                       "stream_lane": {"lane": "stream"}}


def _pack(kind: str, out: Path) -> Path:
    from pgw.modelpack import pack_checkpoint

    if kind == "tiny":
        from tests.conftest import _save_tiny

        (out.parent / "hf").mkdir(parents=True, exist_ok=True)
        src: Any = _save_tiny(out.parent / "hf", "qwen2", seed=0)
    else:
        from huggingface_hub import snapshot_download

        src = snapshot_download("Qwen/Qwen2.5-0.5B-Instruct", allow_patterns=["*.json", "*.safetensors", "*.txt"])
    pack_checkpoint(src, out, name=MODEL)
    return out


def _variant(args: argparse.Namespace) -> None:
    """One variant in this process: prints a JSON line."""
    from pgw import OpenAI
    from pgw.sdk import LocalPackProvisioner

    root = Path(args.root) / args.variant
    c = OpenAI(api_key=os.environ["PGW_API_KEY"], cache_dir=root / "c", state_dir=root / "s",
               mask_scheme=args.scheme, mask_workers=1, mask_target_slots=4 * (args.tokens + 64),
               mask_segment_slots=128, provisioner=LocalPackProvisioner({MODEL: Path(args.pack)}),
               **VARIANTS[args.variant])
    t0 = time.perf_counter()
    c.warmup(MODEL)
    c.chat.completions.create(model=MODEL, messages=PROMPT, max_tokens=1, temperature=0)
    first = time.perf_counter() - t0
    ttft, tps, ref = [], [], None
    for _ in range(args.runs + 1):  # the first chat warms up
        t0 = time.perf_counter()
        times, lp = [], []
        for ch in c.chat.completions.create(model=MODEL, messages=PROMPT, stream=True, temperature=0,
                                            max_tokens=args.tokens, logprobs=True, extra_body={"ignore_eos": True}):
            if ch.choices and ch.choices[0].logprobs:
                times.append(time.perf_counter())
                lp += [(x.token, x.logprob) for x in ch.choices[0].logprobs.content]
        ttft.append(times[0] - t0)
        tps.append((len(times) - 1) / (times[-1] - times[0]))
        ref = lp if ref is None else ref
        assert lp == ref, "the same request gave different tokens"
    st = c.status()["models"][MODEL]
    c.close()
    print(json.dumps({"first_request_s": round(first, 2), "ttft_ms": round(1e3 * statistics.median(ttft[1:]), 1),
                      "decode_tok_s": round(statistics.median(tps[1:]), 2), "lane": st.get("lane"),
                      "native": st.get("native"), "masked_head": st.get("masked_head"), "logprobs": ref}))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", choices=["tiny", "qwen0.5b"], default="tiny")
    ap.add_argument("--tokens", type=int, default=32)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--scheme", default="plain", choices=["plain", "pks-v3", "mlwe"])
    ap.add_argument("--variant", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--root", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--pack", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.variant:
        _variant(args)
        return
    from tests.sdk.conftest import make_pki, start_provider, stop_provider

    with tempfile.TemporaryDirectory(prefix="pgw-sdk-bench-") as d:
        root = Path(d)
        for sub in ("pki", "provider"):
            (root / sub).mkdir()
        pack = _pack(args.model, root / "pack" / "pack")
        prov = start_provider(root / "provider", pack, make_pki(root / "pki"))
        out: dict[str, Any] = {"model": args.model, "tokens": args.tokens, "cpus": os.cpu_count(),
                               "scheme": args.scheme, "variants": {}}
        ref = None
        try:
            env = {k: v for k, v in os.environ.items() if not k.startswith("PGW_")}
            env["PGW_API_KEY"] = prov.new_key()
            env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src"), str(ROOT)])
            for name in VARIANTS:
                r = subprocess.run(  # noqa: S603 - this script, fixed arguments
                    [sys.executable, __file__, "--variant", name, "--root", str(root), "--pack", str(pack),
                     "--tokens", str(args.tokens), "--runs", str(args.runs), "--scheme", args.scheme],
                    env=env, capture_output=True, text=True, timeout=3600)
                if r.returncode != 0:
                    raise SystemExit(f"variant {name} failed:\n{r.stderr[-4000:]}")
                res = json.loads(r.stdout.strip().splitlines()[-1])
                lps = res.pop("logprobs")
                ref = lps if ref is None else ref
                res["bit_identical_to_default"] = lps == ref
                out["variants"][name] = res
                print(name, json.dumps(res), file=sys.stderr, flush=True)
        finally:
            stop_provider(prov)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
