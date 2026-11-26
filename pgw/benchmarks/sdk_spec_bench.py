#!/usr/bin/env python3
"""Strict speculation in the client library, end to end on CPU: tokens per pass, decode tok/s, exactness.

Starts the SDK tests' provider (real ``pgw service``: TLS, TCP+TLS trip lane, masked LM head, accounts; the real
Qwen2.5-0.5B-Instruct pack) and, per configuration (``off``, ``head:R``, ``lookup:R``), one ``pgw.OpenAI`` client
in this process that answers the same held-out prompts (the distillation's evaluation prompts, never trained on:
open chat from UltraChat ``test_sft``, code requests from Alpaca, plus the two prompts of ``sdk_trip_profile.py``)
greedily, streamed. Per prompt and configuration: generated tokens, decode tok/s (first to last token), real
passes and accepted drafts (the generator's ``spec_stats``), and the tokens and logprobs, which must equal the
``off`` configuration's bit for bit.

    PYTHONPATH=src:. python benchmarks/sdk_spec_bench.py --head path/to/head.pgwhead \\
        --configs off,head:2,head:3,head:4,lookup:3 --chat 6 --code 4 --tokens 96 --json out.json

The provider computes on the same 4-vCPU box, so a pass costs what a CPU provider costs; compare configurations.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(ROOT / "benchmarks")]
MODEL = "qwen2.5-0.5b-instruct"


def prompts(n_chat: int, n_code: int) -> list[tuple[str, list[dict[str, str]]]]:
    import feature_head_distill as fd
    import sdk_trip_profile as tp

    chat, code = fd.eval_prompts(n_chat, n_code)
    out = [("chat", m) for m in [tp.PROMPTS["chat"]]] + [("code", tp.PROMPTS["edit"])]
    out += [("chat", [{"role": "user", "content": t}]) for t in chat[:n_chat]]
    out += [("code", [{"role": "user", "content": t}]) for t in code[:n_code]]
    return out


def _stock(c: Any) -> int:
    try:
        return int(c.status()["models"][MODEL].get("masks_available") or 0)
    except Exception:  # noqa: BLE001
        return 0


def _wait_stock(c: Any, need: int, max_s: float) -> int:
    """Wait until the mask factory holds ``need`` slots (or stops growing): timing then measures decoding, not
    the factory (on a 4-vCPU box the factory and the gateway share the cores)."""
    t0, last, t_last = time.monotonic(), -1, time.monotonic()
    while True:
        have = _stock(c)
        if have >= need or time.monotonic() - t0 > max_s:
            return have
        if have > last:
            last, t_last = have, time.monotonic()
        elif time.monotonic() - t_last > 300:
            return have
        time.sleep(5)


def run_config(cfg: str, key: str, pack: Path, root: Path, head: Path | None, items: list, tokens: int,
               stock_s: float = 0.0, extra: dict[str, Any] | None = None) -> list[dict]:
    from pgw import OpenAI
    from pgw.sdk import LocalPackProvisioner

    opts: dict[str, Any] = {}
    if cfg != "off":
        kind, r = cfg.split(":")
        opts["spec_rows"] = int(r)
        if kind == "head":
            opts["speculation"] = "draft_head"
            if head is not None:
                opts["draft_head_path"] = head
        else:
            opts["speculation"] = "lookup"
    kw: dict[str, Any] = {"mask_scheme": "plain", "mask_workers": 1, "mask_segment_slots": 256,
                          "mask_store_gb": float(os.environ.get("PGW_BENCH_MASK_GB", "3")), **opts, **(extra or {})}
    c = OpenAI(api_key=key, cache_dir=root / "c", state_dir=root / "s",
               provisioner=LocalPackProvisioner({MODEL: pack}), **kw)
    try:
        c.warmup(MODEL)
        if stock_s > 0:
            r = int(opts.get("spec_rows", 1))
            need = sum(r * tokens + 64 + 4 * len(str(m)) // 3 for _, m in items)
            print(json.dumps({"config": cfg, "stock_need": need, "stock": _wait_stock(c, need, stock_s)}),
                  flush=True)
        c.chat.completions.create(model=MODEL, messages=items[0][1], max_tokens=4, temperature=0)
        eng = c._engine
        rt = eng.models[MODEL].runtime
        gen = rt.generator
        eng.call(gen.wait_drained(), timeout_s=900)
        res = []
        for kind, msgs in items:
            s0 = dict(getattr(gen, "spec_stats", {}) or {})
            times, lp = [], []
            t0 = time.perf_counter()
            for ch in c.chat.completions.create(model=MODEL, messages=msgs, stream=True, temperature=0,
                                                max_tokens=tokens, logprobs=True):
                if ch.choices and ch.choices[0].logprobs:
                    for x in ch.choices[0].logprobs.content:
                        times.append(time.perf_counter())
                        lp.append((x.token, x.logprob))
            s1 = dict(getattr(gen, "spec_stats", {}) or {})
            eng.call(gen.wait_drained(), timeout_s=900)  # the dummy passes still due (not timed)
            n = len(lp)
            passes = s1.get("spec_passes", 0) - s0.get("spec_passes", 0)
            acc = s1.get("spec_accepted", 0) - s0.get("spec_accepted", 0)
            # with streaming, the tokens of one pass arrive together: decode time from the first token's arrival
            dec_s = times[-1] - times[0] if n > 1 else float("nan")
            res.append({"kind": kind, "tokens": n, "ttft_s": times[0] - t0 if times else None,
                        "decode_tok_s": (n - 1) / dec_s if n > 1 else None,
                        "spec_passes": passes, "spec_accepted": acc,
                        "tokens_per_pass": (passes + acc) / passes if passes else 1.0, "logprobs": lp})
            print(json.dumps({"config": cfg, **{k: v for k, v in res[-1].items() if k != "logprobs"}}), flush=True)
        return res
    finally:
        c.close()


def main() -> None:
    global MODEL
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--configs", default="off,head:2,head:3,head:4,lookup:3")
    ap.add_argument("--head", type=Path, default=None, help="draft head file (default: the shipped, pinned head)")
    ap.add_argument("--chat", type=int, default=6)
    ap.add_argument("--code", type=int, default=4)
    ap.add_argument("--tokens", type=int, default=96)
    ap.add_argument("--pack", default=None, help="an existing pack directory (default: pack into a temp dir)")
    ap.add_argument("--model", default=MODEL, help="SDK model name (a pinned model)")
    ap.add_argument("--provider-device", default="cpu", help="the provider's device (cuda on a GPU host)")
    ap.add_argument("--stock-s", type=float, default=0.0,
                    help="before each configuration, wait up to this long for the mask stock it needs")
    ap.add_argument("--opt", action="append", default=[], help="extra SDKOptions key=value for every client")
    ap.add_argument("--work", default=None, help="work directory (default: a temp dir)")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    MODEL = args.model
    import sdk_trip_profile

    from tests.sdk.conftest import make_pki, start_provider, stop_provider

    sdk_trip_profile.MODEL = MODEL
    extra: dict[str, Any] = {}
    for kv in args.opt:
        k, v = kv.split("=", 1)
        extra[k] = json.loads(v) if v[:1] in "[{0123456789-" or v in ("true", "false", "null") else v
    items = prompts(args.chat, args.code)
    out: dict[str, Any] = {"args": {k: str(v) for k, v in vars(args).items()}, "configs": {}}
    with tempfile.TemporaryDirectory(prefix="pgw-spec-bench-", dir=args.work) as d:
        root = Path(d)
        for sub in ("pki", "provider"):
            (root / sub).mkdir()
        pack = Path(args.pack) if args.pack else sdk_trip_profile._pack(root / "pack" / "pack")
        env = {} if args.provider_device == "cpu" else {"PGW_SERVICE_DEVICE": args.provider_device,
                                                         "PGW_SERVICE_WARMUP": "true", "OMP_NUM_THREADS": "2"}
        prov = start_provider(root / "provider", pack, make_pki(root / "pki"), extra_env=env,
                              ready_timeout_s=1800)
        try:
            for i, cfg in enumerate(args.configs.split(",")):
                out["configs"][cfg] = run_config(cfg, prov.new_key(), pack, root / f"client{i}", args.head, items,
                                                 args.tokens, args.stock_s, extra)
                shutil.rmtree(root / f"client{i}", ignore_errors=True)  # its mask store (disk)
        finally:
            stop_provider(prov)
    ref = out["configs"].get("off")
    summary: dict[str, Any] = {}
    for res in out["configs"].values():
        if ref is not None:
            for a, b in zip(res, ref, strict=True):
                a["identical_to_off"] = a["logprobs"] == b["logprobs"]
    for cfg, res in out["configs"].items():
        row: dict[str, Any] = {}
        for kind in ("chat", "code"):
            rs = [r for r in res if r["kind"] == kind]
            if not rs:
                continue
            toks = sum(r["tokens"] - 1 for r in rs)
            passes = sum(r["spec_passes"] for r in rs)
            acc = sum(r["spec_accepted"] for r in rs)
            row[kind] = {"prompts": len(rs), "tokens": toks + len(rs),
                         "tokens_per_pass": round((passes + acc) / passes, 3) if passes else 1.0,
                         "decode_tok_s_median": round(statistics.median(r["decode_tok_s"] for r in rs), 3),
                         "decode_tok_s_pooled": round(toks / sum((r["tokens"] - 1) / r["decode_tok_s"] for r in rs), 3),
                         "all_identical_to_off": all(r.get("identical_to_off", True) for r in rs)}
        summary[cfg] = row
    for res in out["configs"].values():
        for r in res:
            r.pop("logprobs")
    out["summary"] = summary
    print(json.dumps(summary, indent=1))
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
