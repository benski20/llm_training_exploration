#!/usr/bin/env python3
"""The client library against a REMOTE provider (a RunPod GPU pod), measured from a client machine in the same data
center: first-use setup, TTFT, decode tokens/s, round trips per token, client CPU, with strict speculation off /
lookup / draft_head, then N concurrent streams.

    export PGW_API_KEY=...            # the test SDK key (never on the command line)
    python deploy/runpod/sdk_remote_bench.py --model qwen2.5-7b-instruct --spec off,lookup,draft_head \\
        --conc 1,4,16 --stages setup,spec,conc,prefix --out results/sdk-7b

What it does, in one process (the native switches are process-wide), one cache directory for every stage:

1. ``setup``: ``OpenAI(...).warmup(model)`` (enroll, provision: gateway pack + factory pack, fingerprint-checked) and
   the first answer (waits for the local pks-v3 hint build and the first masks): seconds and the client's CPU.
2. ``spec_<mode>`` for each ``--spec`` mode, a fresh client with ``speculation=<mode>``: ``--runs`` warm, streamed,
   greedy chats per prompt kind (``chat``: open chat; ``edit``: a copy-heavy code edit, what lookup is for), first
   run discarded: TTFT, decode tokens/s, trips per decoded token (the runtime's counters; prompt trips subtracted),
   tokens per pass, the provider's ``compute_us`` per trip, lane fixed cost seen from here, client CPU per token
   (this process, the mask factory children, and the whole box). Every mode must give the same tokens and
   log-probabilities as ``off`` (strict speculation is exact): checked and recorded. ``draft_head`` uses the head
   the library ships pinned for the model (``pgw.sdk.draft_heads``) unless ``--draft-head`` names another file; a
   model without a shipped head skips the mode (noted in the results).
3. ``conc_<N>`` for each ``--conc``: N threads, one streamed greedy request each (four prompts), against a stock
   filled first: aggregate tokens/s, per-stream tokens/s, TTFT p50/p95, errors, bit-identity with each prompt alone.
4. ``prefix`` (``--stages ...,prefix``): the client KV prefix cache, ``prefix_cache=False`` then ``True`` on the same
   multi-turn conversations (a system prompt and ``--prefix-turns`` user turns resending the history, then a second
   conversation with the same system prompt): TTFT per turn, cached tokens, prompt-pass rows and mask slots per
   request, every answer bit-identical between the two clients.

Results: ``<out>/results.json`` (updated after every stage; a stopped run keeps what it measured) and
``<out>/status.jsonl`` (the client's status every 30 s; samples whose factory reports ``maskgen_stalled`` are counted
in ``maskgen_stalled_samples``). The key is read from ``PGW_API_KEY`` and never printed.

``--preview`` runs a model the library pins but does not support yet (e.g. 32B) through the measurement-only opt-in
``PGW_SDK_PREVIEW_MODELS`` (``pgw.sdk.options.preview_models``): the provider's pack is still verified against the
baked pin. Without it such a model is refused, as for a customer.

CPU smoke test (a local CPU provider with the tiny test model, plain masks; numbers meaningless; with ``--preview
--model qwen2.5-32b-instruct`` the tiny pack is served under that name, so the preview path runs; ``draft_head``
uses a random head file of the tiny model's shape)::

    PYTHONPATH=src:. python deploy/runpod/sdk_remote_bench.py --smoke --spec off,lookup,draft_head --conc 1,2 \\
        --tokens 8 --runs 1 --stages setup,spec,conc,prefix --out /tmp/sdkr
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import itertools
import json
import os
import statistics
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(ROOT / "benchmarks")]

CODE = (Path(ROOT / "benchmarks" / "sdk_trip_profile.py").read_text().split('CODE = """', 1)[1].split('"""', 1)[0]
        if (ROOT / "benchmarks" / "sdk_trip_profile.py").exists() else "def f(x):\n    return x\n")
PROMPTS = {
    "chat": [{"role": "user", "content": "Write a short paragraph about why the sky is blue."}],
    "edit": [{"role": "user", "content": "Rename the variable `total` to `paid_total` in this function. Reply with "
              "the full function only, no explanation.\n\n" + CODE}],
}
CONC_PROMPTS = ["Write one sentence about the sea.", "Name three colours.", "What is two plus two?",
                "Describe a cat in a few words."]
R: dict[str, Any] = {"stages": {}, "errors": []}
OUT = Path(".")


def log(*a: Any) -> None:
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def save() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / "results.json.tmp"
    tmp.write_text(json.dumps(R, indent=1, default=str))
    os.replace(tmp, OUT / "results.json")


def _box_cpu() -> tuple[float, float]:
    """(busy, total) jiffies of the whole machine (``/proc/stat``)."""
    try:
        f = [int(x) for x in Path("/proc/stat").read_text().splitlines()[0].split()[1:]]
        return float(sum(f) - f[3] - f[4]), float(sum(f))
    except (OSError, ValueError, IndexError):
        return 0.0, 1.0


def _children_cpu() -> float:
    from sdk_concurrency_bench import _tree_cpu

    return _tree_cpu(os.getpid()) - _self_cpu()


def _self_cpu() -> float:
    from sdk_concurrency_bench import _proc_cpu

    return _proc_cpu(os.getpid())


def _cpu_snapshot() -> dict[str, float]:
    busy, tot = _box_cpu()
    return {"self": time.process_time(), "children": _children_cpu(), "box_busy": busy, "box_total": tot,
            "wall": time.perf_counter()}


def _cpu_delta(a: dict[str, float], b: dict[str, float], tokens: int) -> dict[str, float]:
    wall = max(1e-9, b["wall"] - a["wall"])
    ncpu = os.cpu_count() or 1
    box = (b["box_busy"] - a["box_busy"]) / max(1.0, b["box_total"] - a["box_total"]) * ncpu
    return {"client_cpu_ms_per_token": round(1e3 * (b["self"] - a["self"]) / max(1, tokens), 2),
            "factory_cpu_ms_per_token": round(1e3 * (b["children"] - a["children"]) / max(1, tokens), 2),
            "box_cores_busy": round(box, 2), "wall_s": round(wall, 2)}


def make_client(args: argparse.Namespace, **over: Any) -> Any:
    from pgw import OpenAI

    kw: dict[str, Any] = {"cache_dir": Path(args.cache) / "c", "state_dir": Path(args.cache) / "s",
                          "mask_store_gb": args.mask_store_gb, "timeout_s": 3600.0, "mask_wait_s": 1800.0,
                          "max_concurrent": max(8, max(args.conc_list)), "mask_admission": args.admission}
    if args.mask_workers:
        kw["mask_workers"] = args.mask_workers
    if args.mask_scheme:  # measurement only (the wan session): the same trips, masks made locally
        kw["mask_scheme"] = args.mask_scheme
    if args.smoke:
        from pgw.sdk import LocalPackProvisioner

        kw.update(mask_scheme="plain", mask_workers=1, mask_segment_slots=128,
                  provisioner=LocalPackProvisioner({args.model: Path(args.pack)}))
    kw.update(over)
    return OpenAI(api_key=os.environ["PGW_API_KEY"], **kw)


def _stream(c: Any, model: str, messages: Any, tokens: int, **extra: Any) -> tuple[list[float], list[Any]]:
    times, lp = [], []
    t0 = time.perf_counter()
    for ch in c.chat.completions.create(model=model, messages=messages, stream=True, temperature=0,
                                        max_tokens=tokens, logprobs=True, extra_body={"ignore_eos": True}, **extra):
        if ch.choices and ch.choices[0].logprobs:
            times.append(time.perf_counter())
            lp += [(x.token, x.logprob) for x in ch.choices[0].logprobs.content]
    return [t0, *times], lp


def _stock(c: Any, model: str) -> int:
    with contextlib.suppress(Exception):
        return int(c.status()["models"][model].get("masks_available") or 0)
    return 0


def wait_stock(c: Any, model: str, slots: int, timeout_s: float) -> float:
    t0 = time.monotonic()
    while _stock(c, model) < slots and time.monotonic() - t0 < timeout_s:
        time.sleep(1.0)
    return round(time.monotonic() - t0, 1)


def stage_setup(args: argparse.Namespace) -> None:
    a = _cpu_snapshot()
    c = make_client(args)
    t0 = time.perf_counter()
    c.warmup(args.model)
    t_warm = time.perf_counter() - t0
    times, _ = _stream(c, args.model, PROMPTS["chat"], 8)
    b = _cpu_snapshot()
    st = c.status()["models"].get(args.model, {})
    R["stages"]["setup"] = {"warmup_s": round(t_warm, 1), "first_answer_s": round(times[-1] - t0, 1),
                            "first_ttft_s": round(times[1] - times[0], 2) if len(times) > 1 else None,
                            "cpu": _cpu_delta(a, b, 1), "lane": st.get("lane"), "native": st.get("native"),
                            "masked_head": st.get("masked_head"), "scheme": st.get("mask_scheme")}
    c.close()
    save()


def stage_spec(args: argparse.Namespace, mode: str, refs: dict[str, Any]) -> None:
    from sdk_trip_profile import _counters, _delta

    over: dict[str, Any] = {}
    if mode != "off":
        over.update(speculation=mode, spec_rows=args.spec_rows)
    if mode == "draft_head" and args.draft_head:  # else the head the library ships pinned for the model
        over["draft_head_path"] = Path(args.draft_head)
    c = make_client(args, **over)
    c.warmup(args.model)
    rows = 1 if mode == "off" else args.spec_rows
    fill = wait_stock(c, args.model, min(int(1.2 * rows * (args.tokens + 300) * 2), args.max_fill_slots),
                      args.fill_timeout)
    rt = c._engine.models[args.model].runtime
    out: dict[str, Any] = {"stock_wait_s": fill, "spec_rows": rows}
    if mode == "draft_head":
        from pgw.sdk.draft_heads import PINNED_DRAFT_HEADS

        pin = PINNED_DRAFT_HEADS.get(args.model)
        out["draft_head"] = ({"file": Path(args.draft_head).name, "source": "--draft-head"} if args.draft_head else
                             {"file": pin.file if pin else None, "sha256": pin.sha256 if pin else None,
                              "source": "pinned (shipped)"})
    for kind, msgs in PROMPTS.items():
        _stream(c, args.model, msgs, 1)  # prompt-only trips of this prompt
        c._engine.call(rt.generator.wait_drained(), timeout_s=600)
        a0 = _counters(rt)
        _stream(c, args.model, msgs, 1)
        c._engine.call(rt.generator.wait_drained(), timeout_s=600)
        prompt_d = _delta(a0, _counters(rt))
        runs = []
        for i in range(args.runs + 1):
            a = _counters(rt)
            ca = _cpu_snapshot()
            times, lp = _stream(c, args.model, msgs, args.tokens)
            cb = _cpu_snapshot()
            c._engine.call(rt.generator.wait_drained(), timeout_s=600)  # strict speculation's due dummy passes
            d = _delta(a, _counters(rt))
            ref = refs.setdefault(kind, lp)
            if i == 0:
                continue
            n_dec = max(1, len(times) - 2)
            dec = {k: d[k] - prompt_d.get(k, 0.0) for k in d}
            gaps = [t2 - t1 for t1, t2 in itertools.pairwise(times[1:])]
            ch_trips = max(1.0, dec.get("ch_trips", 0.0))
            runs.append({
                "ttft_ms": round(1e3 * (times[1] - times[0]), 1),
                "decode_tok_s": round(n_dec / max(1e-9, times[-1] - times[1]), 2),
                "decode_ms_per_token_p50": round(1e3 * statistics.median(gaps), 2) if gaps else None,
                "trips_per_token": round(dec.get("backend_round_trips", 0.0) / n_dec, 2),
                "rows_per_trip": round(dec.get("rows", 0.0) / max(1.0, dec.get("backend_round_trips", 0.0)), 2),
                "provider_compute_us": round(dec.get("ch_compute_us", 0.0) / ch_trips, 1),
                "lane_trip_us": round(dec.get("ch_trip_us", 0.0) / ch_trips, 1),
                "lane_wait_beyond_compute_us": round((dec.get("ch_wait_us", 0.0) - dec.get("ch_compute_us", 0.0)
                                                      - dec.get("ch_queue_us", 0.0)) / ch_trips, 1),
                "bit_identical_to_off": lp == ref, **_cpu_delta(ca, cb, n_dec + 1)})
        out[kind] = {"runs": runs, "summary": {k: statistics.median(r[k] for r in runs) for k in runs[0]
                                               if isinstance(runs[0][k], int | float) and not isinstance(
                                                   runs[0][k], bool)} if runs else {},
                     "bit_identical_to_off": all(r["bit_identical_to_off"] for r in runs)}
        log(f"spec {mode} {kind}: {out[kind]['summary']}")
    with contextlib.suppress(Exception):
        out["spec_stats"] = rt.status().get("speed", {}).get("spec_stats")
    c.close()
    R["stages"][f"spec_{mode}"] = out
    save()


def stage_conc(args: argparse.Namespace, n: int, refs: dict[int, Any]) -> None:
    c = make_client(args)
    c.warmup(args.model)
    for j in range(len(CONC_PROMPTS)):
        if j not in refs:
            _, refs[j] = _stream(c, args.model, [{"role": "user", "content": CONC_PROMPTS[j]}], args.conc_tokens)
    fill = wait_stock(c, args.model, min(int(1.2 * n * (args.conc_tokens + 64)), args.max_fill_slots),
                      args.fill_timeout)
    be = c._engine.models[args.model].runtime.backend
    s0 = be.stats.snapshot()
    res: list[dict[str, Any]] = []
    errs: list[str] = []

    def one(i: int) -> None:
        j = i % len(CONC_PROMPTS)
        try:
            times, lp = _stream(c, args.model, [{"role": "user", "content": CONC_PROMPTS[j]}], args.conc_tokens,
                                user=f"user-{i}")
            res.append({"ttft": times[1] - times[0], "tps": (len(times) - 2) / max(1e-9, times[-1] - times[1]),
                        "n": len(lp), "same": lp == refs[j]})
        except Exception as exc:  # recorded
            errs.append(f"{type(exc).__name__}: {str(exc)[:200]}")

    ca = _cpu_snapshot()
    ts = [threading.Thread(target=one, args=(i,)) for i in range(n)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    cb = _cpu_snapshot()
    s1 = be.stats.snapshot()
    toks = sum(r["n"] for r in res)
    ttft = sorted(r["ttft"] for r in res)
    rt_ = s1["round_trips"] - s0["round_trips"]
    R["stages"][f"conc_{n}"] = {
        "streams": n, "stock_wait_s": fill, "errors": len(errs), "error_samples": errs[:3],
        "bit_identical": all(r["same"] for r in res) and not errs, "tokens": toks,
        "agg_tok_s": round(toks / max(1e-9, cb["wall"] - ca["wall"]), 2),
        "stream_tok_s_p50": round(statistics.median(r["tps"] for r in res), 2) if res else None,
        "ttft_p50_s": round(ttft[len(ttft) // 2], 3) if ttft else None,
        "ttft_p95_s": round(ttft[min(len(ttft) - 1, int(0.95 * len(ttft)))], 3) if ttft else None,
        "rows_per_trip": round((s1["rows"] - s0["rows"]) / max(1, rt_), 2), "round_trips": rt_,
        # the answers alone, hashed: compare across processes (e.g. the native switches on and off)
        "refs_sha256": hashlib.sha256(json.dumps([refs[j] for j in sorted(refs)]).encode()).hexdigest()[:16],
        **_cpu_delta(ca, cb, max(1, toks))}
    log(f"conc {n}: {R['stages'][f'conc_{n}']}")
    c.close()
    save()


def _prefix_turns(c: Any, model: str, probe: Any, conv: str, system: str, users: list[str], tokens: int,
                  stock: int) -> list[dict[str, Any]]:
    """One conversation, each turn resending the whole history (``benchmarks/sdk_prefix_cache_bench.py``)."""
    msgs: list[dict[str, Any]] = [{"role": "system", "content": system}]
    out = []
    for u in users:
        msgs.append({"role": "user", "content": u})
        wait_stock(c, model, stock, 600.0)
        t0 = time.perf_counter()
        first = None
        text, lps, usage = [], [], None
        for ch in c.chat.completions.create(model=model, messages=msgs, stream=True, temperature=0, max_tokens=tokens,
                                            logprobs=True, user="bench", stream_options={"include_usage": True},
                                            extra_body={"ignore_eos": True, "conversation_id": conv}):
            if ch.choices and ch.choices[0].logprobs:
                first = first or time.perf_counter()
                lps += [(x.token, x.logprob) for x in ch.choices[0].logprobs.content]
            if ch.choices and ch.choices[0].delta.content:
                text.append(ch.choices[0].delta.content)
            if ch.usage is not None:
                usage = ch.usage
        sess = probe.sessions[-1]
        cached = int(usage.prompt_tokens_details.cached_tokens) if usage and usage.prompt_tokens_details else 0
        out.append({"conv": conv, "prompt_tokens": int(usage.prompt_tokens) if usage else None, "cached_tokens": cached,
                    "prompt_pass_rows": len(sess.prompt_ids) - int(sess.prefix_reused),
                    "masks": int(sess.masks_consumed),
                    "ttft_s": round((first or time.perf_counter()) - t0, 3),
                    "total_s": round(time.perf_counter() - t0, 3), "lp": lps})
        msgs.append({"role": "assistant", "content": "".join(text)})
    return out


def stage_prefix(args: argparse.Namespace) -> None:
    """``prefix_cache=False`` then ``True`` on the same conversations: TTFT, rows and masks per turn, bit-identity."""
    from sdk_prefix_cache_bench import OTHER, SHORT, SYSTEM, USERS, _Probe

    system, users, other = SHORT if args.smoke else (SYSTEM, USERS, OTHER)
    runs: dict[bool, list[dict[str, Any]]] = {}
    for cached in (False, True):
        c = make_client(args, prefix_cache=cached)
        try:
            c.warmup(args.model)
            probe = _Probe(c._engine.models[args.model].runtime)
            stock = min(args.max_fill_slots, 1200 if not args.smoke else 400)
            wait_stock(c, args.model, stock, args.fill_timeout)
            _stream(c, args.model, [{"role": "user", "content": "Hi"}], 2)  # first-use costs, not measured
            ca = _cpu_snapshot()
            rows = _prefix_turns(c, args.model, probe, "A", system, users[: args.prefix_turns], args.prefix_tokens,
                                 stock)
            rows += _prefix_turns(c, args.model, probe, "B", system, other, args.prefix_tokens, stock)
            cb = _cpu_snapshot()
            rows[-1]["cpu"] = _cpu_delta(ca, cb, sum(len(r["lp"]) for r in rows))
            with contextlib.suppress(Exception):
                rows[-1]["prefix_cache_status"] = c.status()["models"][args.model].get("prefix_cache")
            runs[cached] = rows
        finally:
            c.close()
    turns = []
    for a, b in zip(runs[False], runs[True], strict=True):
        turns.append({"conv": a["conv"], "prompt_tokens": a["prompt_tokens"], "cached_tokens": b["cached_tokens"],
                      "rows_off": a["prompt_pass_rows"], "rows_on": b["prompt_pass_rows"], "masks_off": a["masks"],
                      "masks_on": b["masks"], "ttft_off_s": a["ttft_s"], "ttft_on_s": b["ttft_s"],
                      "bit_identical": a["lp"] == b["lp"]})
    later = turns[1:] or turns
    R["stages"]["prefix"] = {
        "turns": turns, "bit_identical": all(t["bit_identical"] for t in turns),
        "masks_off": sum(t["masks_off"] for t in turns), "masks_on": sum(t["masks_on"] for t in turns),
        "prompt_rows_off": sum(t["rows_off"] for t in turns), "prompt_rows_on": sum(t["rows_on"] for t in turns),
        "ttft_off_median_later_s": statistics.median(t["ttft_off_s"] for t in later),
        "ttft_on_median_later_s": statistics.median(t["ttft_on_s"] for t in later),
        "cpu_off": runs[False][-1].get("cpu"), "cpu_on": runs[True][-1].get("cpu"),
        "prefix_cache_status": runs[True][-1].get("prefix_cache_status")}
    log(f"prefix: {({k: v for k, v in R['stages']['prefix'].items() if k != 'turns'})}")
    save()


def _smoke_pack(out: Path, name: str) -> Path:
    """The tiny test model packed under ``name`` (a preview name runs the preview path end to end)."""
    from tests.conftest import _save_tiny

    from pgw.modelpack import pack_checkpoint

    (out.parent / "hf").mkdir(parents=True, exist_ok=True)
    pack_checkpoint(_save_tiny(out.parent / "hf", "qwen2", seed=0), out, name=name)
    return out


def _smoke_head(pack: Path, path: Path) -> Path:
    """A random draft head of the tiny model's shape in the trained-file format (tests/engine/test_tree_spec.py)."""
    from types import SimpleNamespace

    from tests.engine.test_tree_spec import _head_file

    from pgw.modelpack import ModelPack

    with ModelPack.open(pack) as p:
        model = SimpleNamespace(arch=p.arch, embed_tokens=p.load_plain("embed_tokens"))
    _head_file(model, path)  # type: ignore[arg-type]
    return path


def status_sampler(args: argparse.Namespace, stop: threading.Event) -> None:
    from pgw.sdk.cli import status_report

    while not stop.wait(30.0):
        with contextlib.suppress(Exception), (OUT / "status.jsonl").open("a") as f:
            rep = status_report(Path(args.cache) / "c", Path(args.cache) / "s")
            line = json.dumps({"t": time.time(), "status": rep}, default=str)
            if "maskgen_stalled" in line:  # the client's maskgen watchdog fired (a finding unless the provider hung)
                R["maskgen_stalled_samples"] = int(R.get("maskgen_stalled_samples", 0)) + 1
            f.write(line + "\n")


def main(argv: list[str] | None = None) -> int:
    global OUT
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default="qwen2.5-7b-instruct")
    ap.add_argument("--spec", default="off,lookup", help="comma list of off, lookup, draft_head")
    ap.add_argument("--spec-rows", type=int, default=3)
    ap.add_argument("--draft-head", default=None,
                    help="draft head file overriding the head shipped for the model (pgw.sdk.draft_heads)")
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--conc", default="1,4,16")
    ap.add_argument("--conc-tokens", type=int, default=64)
    ap.add_argument("--mask-store-gb", type=float, default=24.0)
    ap.add_argument("--max-fill-slots", type=int, default=3000, help="cap of the stock waited for before a stage")
    ap.add_argument("--fill-timeout", type=float, default=1800.0)
    ap.add_argument("--mask-workers", type=int, default=0, help="0: the library's default")
    ap.add_argument("--admission", default="entitled", choices=["entitled", "overcommit"])
    ap.add_argument("--mask-scheme", default=None, choices=["plain", "pks-v3"],
                    help="measurement only: override the library's mask scheme (plain: masks from the local factory "
                         "pack, no maskgen traffic; the trips and what the provider sees per trip are the same)")
    ap.add_argument("--preview", action="store_true",
                    help="measurement-only: enable --model through PGW_SDK_PREVIEW_MODELS (pinned, not supported yet; "
                         "still verified against its baked pin)")
    ap.add_argument("--prefix-turns", type=int, default=4, help="prefix stage: user turns of the first conversation")
    ap.add_argument("--prefix-tokens", type=int, default=16, help="prefix stage: tokens per answer")
    ap.add_argument("--cache", default=None, help="client cache root (default <out>/cache)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stages", default="setup,spec,conc", help="comma list of setup, spec, conc, prefix")
    ap.add_argument("--smoke", action="store_true", help="local CPU provider, tiny model, plain masks")
    ap.add_argument("--pack", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    OUT = Path(args.out)
    args.cache = args.cache or str(OUT / "cache")
    args.conc_list = [int(x) for x in args.conc.split(",") if x]
    modes = [m for m in args.spec.split(",") if m]
    bad = [m for m in modes if m not in ("off", "lookup", "draft_head")]
    stages = set(args.stages.split(","))
    if bad or stages - {"setup", "spec", "conc", "prefix"}:
        raise SystemExit(f"unknown --spec mode(s) {bad} or --stages {sorted(stages)}")
    if args.draft_head and not Path(args.draft_head).is_file():  # an explicit override must exist: fail early
        raise SystemExit(f"--draft-head {args.draft_head}: no such file")
    if args.smoke and not args.preview:
        args.model = "qwen2.5-0.5b-instruct"
    if args.preview:
        from pgw.sdk.options import PREVIEW_ENV, SUPPORTED_MODELS, preview_models

        if args.model not in SUPPORTED_MODELS:
            os.environ[PREVIEW_ENV] = ",".join(dict.fromkeys(
                [*(x for x in os.environ.get(PREVIEW_ENV, "").split(",") if x.strip()), args.model]))
            preview_models()  # refuses a model without a baked pin before anything else runs
            R.setdefault("notes", []).append(f"{args.model}: measurement-only preview ({PREVIEW_ENV}); the "
                                             "provider's pack is verified against the baked pin")
    if "draft_head" in modes and not args.draft_head and not args.smoke:
        from pgw.sdk.draft_heads import PINNED_DRAFT_HEADS

        if args.model not in PINNED_DRAFT_HEADS:
            log(f"draft_head skipped: no draft head ships for {args.model}")
            R.setdefault("notes", []).append(f"draft_head skipped: no draft head ships for {args.model} "
                                             "(--draft-head to give one)")
            modes.remove("draft_head")
    R.update(model=args.model, host_cpus=os.cpu_count(), args={k: v for k, v in vars(args).items()
                                                                 if k not in ("conc_list",)}, started=time.time())
    prov = None
    tmp = None
    if args.smoke:
        from tests.sdk.conftest import make_pki, start_provider

        tmp = tempfile.TemporaryDirectory(prefix="pgw-sdkr-")
        root = Path(tmp.name)
        for sub in ("pki", "provider"):
            (root / sub).mkdir()
        PROMPTS["edit"] = [{"role": "user", "content": "Rename x to y:\n" + "\n".join(CODE.splitlines()[:3])}]
        args.pack = str(_smoke_pack(root / "pack" / "pack", args.model))
        prov = start_provider(root / "provider", Path(args.pack), make_pki(root / "pki"))
        os.environ["PGW_API_KEY"] = prov.new_key()
        if "draft_head" in modes and not args.draft_head:  # the shipped heads are for the real models' shapes
            args.draft_head = str(_smoke_head(Path(args.pack), root / "head.pt"))
    if not os.environ.get("PGW_API_KEY"):
        raise SystemExit("PGW_API_KEY is not set")
    stop = threading.Event()
    threading.Thread(target=status_sampler, args=(args, stop), daemon=True).start()
    try:
        for name, fn in (("setup", lambda: stage_setup(args)),):
            if name in stages:
                log("stage setup")
                fn()
        refs: dict[str, Any] = {}
        if "spec" in stages:
            for m in (["off"] + [x for x in modes if x != "off"]):
                log(f"stage spec {m}")
                try:
                    stage_spec(args, m, refs)
                except Exception as exc:  # recorded; the next stage still runs
                    R["errors"].append(f"spec {m}: {type(exc).__name__}: {str(exc)[:300]}")
                    save()
        crefs: dict[int, Any] = {}
        if "conc" in stages:
            for n in args.conc_list:
                log(f"stage conc {n}")
                try:
                    stage_conc(args, n, crefs)
                except Exception as exc:
                    R["errors"].append(f"conc {n}: {type(exc).__name__}: {str(exc)[:300]}")
                    save()
        if "prefix" in stages:
            log("stage prefix")
            try:
                stage_prefix(args)
            except Exception as exc:
                R["errors"].append(f"prefix: {type(exc).__name__}: {str(exc)[:300]}")
                save()
    finally:
        stop.set()
        R["finished"] = time.time()
        save()
        if prov is not None:
            from tests.sdk.conftest import stop_provider

            stop_provider(prov)
        if tmp is not None:
            tmp.cleanup()
    log(f"results in {OUT / 'results.json'}")
    return 0 if not R["errors"] else 1


if __name__ == "__main__":
    sys.exit(main())
