"""Client KV prefix cache (``prefix_cache=True``): TTFT, masks and provider rows of a multi-turn chat, cached vs not.

Starts the provider of the SDK tests (TLS, TCP+TLS trip lane, masked LM head; CPU kernels) on loopback and runs the
same conversations through two clients of the library, one with ``prefix_cache=False`` and one with
``prefix_cache=True`` (both with the masked LM head): conversation A (a system prompt and ``--turns`` user turns,
each turn resending the whole history), then conversation B (the same system prompt, a new user message: a
shared-prefix hit) and the next turn of B. Every answer (tokens and logprobs) must be bit-identical between the
two clients. Per turn: TTFT (streamed, time to the first token), prompt tokens, cached tokens, mask slots the
request consumed and rows of its prompt pass (what the provider sees per trip). The mask stock is filled before
each turn, so TTFT does not include waiting for masks.

    PYTHONPATH=src:. python benchmarks/sdk_prefix_cache_bench.py --model tiny
    PYTHONPATH=src:. python benchmarks/sdk_prefix_cache_bench.py --model 0.5b [--pack <packed 0.5B dir>]

The provider computes on the same CPU here: compare the two columns, not absolute GPU numbers.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from benchmarks.sdk_concurrency_bench import MODEL  # noqa: E402

SYSTEM = ("You are a careful assistant for a hardware store. Answer briefly and accurately. When a customer asks "
          "about tools, name the tool, what it is for and one safety tip. When a customer asks about paint, ask "
          "which surface it is for. Never invent prices; say that prices are on the shelf labels. Keep answers "
          "under three sentences unless the customer asks for more detail. Be friendly but do not use emojis.")
USERS = ["Which drill should I use for brick?", "And what bit size for a 6 mm wall plug?",
         "Do I need eye protection for that?", "What about drilling into tiles instead?",
         "Can the same drill drive screws as well?", "Thanks. Which screws go with those plugs?"]
OTHER = ["I want to paint a fence. What do I need?", "It is untreated pine."]
#: the tiny test model has a 256-token context and a byte-level vocabulary: a shorter conversation
SHORT = ("Be terse.", ["Hi?", "Why?", "And?", "Ok."], ["Paint?", "Pine."])


def _pack(kind: str, out: Path) -> Path:
    from benchmarks.sdk_concurrency_bench import _pack as pack

    return pack("tiny" if kind == "tiny" else "qwen0.5b", out)


class _Probe:
    """Records each request's session (masks consumed, reused tokens, prompt-pass rows) by wrapping the runtime."""

    def __init__(self, rt: Any):
        self.sessions: list[Any] = []
        inner = rt.open_sessions

        async def open_sessions(*a: Any, **kw: Any) -> Any:
            out = await inner(*a, **kw)
            self.sessions.extend(out)
            return out

        rt.open_sessions = open_sessions


def _turns(c: Any, probe: _Probe, conv: str, system: str, users: list[str], tokens: int,
           wait_stock: Any) -> list[dict[str, Any]]:
    msgs: list[dict[str, Any]] = [{"role": "system", "content": system}]
    out = []
    for u in users:
        msgs.append({"role": "user", "content": u})
        wait_stock()
        t0 = time.perf_counter()
        first = None
        text, lps = [], []
        usage = None
        for ch in c.chat.completions.create(model=MODEL, messages=msgs, stream=True, temperature=0, max_tokens=tokens,
                                            logprobs=True, top_logprobs=2, user="bench",
                                            stream_options={"include_usage": True},
                                            extra_body={"ignore_eos": True, "conversation_id": conv}):
            if ch.choices and ch.choices[0].logprobs:
                if first is None:
                    first = time.perf_counter()
                lps += [[x.token, x.logprob, [[t.token, t.logprob] for t in x.top_logprobs]]
                        for x in ch.choices[0].logprobs.content]
            if ch.choices and ch.choices[0].delta.content:
                text.append(ch.choices[0].delta.content)
            if ch.usage is not None:
                usage = ch.usage
        t1 = time.perf_counter()
        sess = probe.sessions[-1]
        assert usage is not None and first is not None
        cached = int(usage.prompt_tokens_details.cached_tokens) if usage.prompt_tokens_details else 0
        out.append({"conv": conv, "prompt_tokens": int(usage.prompt_tokens), "cached_tokens": cached,
                    "prompt_pass_rows": len(sess.prompt_ids) - int(sess.prefix_reused),
                    "masks": int(sess.masks_consumed), "ttft_s": round(first - t0, 3), "total_s": round(t1 - t0, 3),
                    "lp": lps})
        msgs.append({"role": "assistant", "content": "".join(text)})
    return out


def run_client(args: argparse.Namespace, key: str, pack: Path, root: Path, cached: bool) -> list[dict[str, Any]]:
    from pgw import OpenAI
    from pgw.sdk import LocalPackProvisioner

    c = OpenAI(api_key=key, cache_dir=root / "c", state_dir=root / "s", mask_scheme="plain", mask_workers=1,
               mask_segment_slots=128, mask_target_slots=args.store_slots,
               mask_low_water_slots=args.store_slots - 1, mask_wait_s=600.0, timeout_s=3600.0,
               prefix_cache=cached, provisioner=LocalPackProvisioner({MODEL: pack}))
    try:
        c.warmup(MODEL)
        st = c.status()["models"][MODEL]
        assert st["masked_head"] is True, "the provider must serve the masked LM head"
        probe = _Probe(c._engine.models[MODEL].runtime)

        def wait_stock() -> None:
            t = time.monotonic()
            while (int(c.status()["models"][MODEL].get("masks_available") or 0) < args.store_slots - 128
                   and time.monotonic() - t < 900):
                time.sleep(0.2)

        # one warm-up request (first-use costs: thread pools, lane connection), not measured
        wait_stock()
        c.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": "Hi"}], max_tokens=2,
                                  temperature=0, user="warmup")
        system, users, other = SHORT if args.model == "tiny" else (SYSTEM, USERS, OTHER)
        rows = _turns(c, probe, "A", system, users[: args.turns], args.tokens, wait_stock)
        rows += _turns(c, probe, "B", system, other, args.tokens, wait_stock)
        rows[-1]["prefix_cache_status"] = c.status()["models"][MODEL].get("prefix_cache")
        return rows
    finally:
        c.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", choices=["tiny", "0.5b"], default="tiny")
    ap.add_argument("--pack", default=None, help="an existing pack directory (else packed into a temp dir)")
    ap.add_argument("--turns", type=int, default=6)
    ap.add_argument("--tokens", type=int, default=16)
    ap.add_argument("--store-slots", type=int, default=1024)
    args = ap.parse_args()
    from tests.sdk.conftest import make_pki, start_provider, stop_provider

    with tempfile.TemporaryDirectory(prefix="pgw-prefix-bench-") as d:
        root = Path(d)
        for sub in ("pki", "provider", "off", "on"):
            (root / sub).mkdir()
        pack = Path(args.pack) if args.pack else _pack(args.model, root / "pack" / "pack")
        prov = start_provider(root / "provider", pack, make_pki(root / "pki"))
        try:
            key = prov.new_key()
            off = run_client(args, key, pack, root / "off", cached=False)
            on = run_client(args, key, pack, root / "on", cached=True)
        finally:
            stop_provider(prov)
    turns = []
    for a, b in zip(off, on, strict=True):
        assert a["lp"] == b["lp"], "cached answer differs from the uncached one"
        turns.append({"conv": a["conv"], "prompt_tokens": a["prompt_tokens"], "cached_tokens": b["cached_tokens"],
                      "rows_off": a["prompt_pass_rows"], "rows_on": b["prompt_pass_rows"],
                      "masks_off": a["masks"], "masks_on": b["masks"], "ttft_off_s": a["ttft_s"],
                      "ttft_on_s": b["ttft_s"]})
    later = [t for t in turns[1:]]
    summary = {
        "model": args.model, "cpus": os.cpu_count(), "turns": turns, "bit_identical": True,
        "masks_off": sum(t["masks_off"] for t in turns), "masks_on": sum(t["masks_on"] for t in turns),
        "prompt_rows_off": sum(t["rows_off"] for t in turns), "prompt_rows_on": sum(t["rows_on"] for t in turns),
        "ttft_off_median_later_s": statistics.median(t["ttft_off_s"] for t in later),
        "ttft_on_median_later_s": statistics.median(t["ttft_on_s"] for t in later),
        "prefix_cache_status": on[-1].get("prefix_cache_status"),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
