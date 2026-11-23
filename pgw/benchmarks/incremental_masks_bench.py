"""Mask slots per generated token of multi-turn chats through the gateway: window vs incremental reservations,
with and without conversation KV-prefix reuse (``mask_reservation``, ``prefix_cache``; SPEC 9.4c).

Real masked path end to end in one process: a real encrypted ``MaskStore`` (fsync and a file counter, as in
production) filled by the real factory, the gateway app (``create_app``: admission, reservations, engine, masked
backend, every linear remote), and an exact in-process provider (``pgw.wire`` router over ``LocalComputeClient``)
that records only (linear, rows) of every call. Each configuration replays the same conversation (greedy; the
assistant turns are the model's own answers) and must give the same answers, logprobs and provider calls as the
others with the same ``prefix_cache``. Per configuration: slots consumed (store inventory before/after, so prompt
rows, decode rows and burned slots), generated tokens, slots per generated token, TTFT and decode time per token
per turn (wall, streaming). The up-front reservation's consumption (``mask_window_tokens = 0``: prompt rows +
max_tokens per turn) is computed from the same transcripts.

    python benchmarks/incremental_masks_bench.py --pack ~/.pgw-client-demo/packs/qwen2.5-0.5b-instruct \\
        --turns 4 --max-tokens 256 --work /tmp/imb --json /tmp/imb.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

SYSTEM = ("You are a careful assistant for an engineering team. Answer briefly and precisely, state assumptions, "
          "prefer concrete numbers, and say when you are unsure. Keep answers under five sentences.")
USER = [
    "We run a gateway that masks activations before sending them to a GPU provider. What limits its latency?",
    "Suppose each token needs 96 round trips at 0.3 ms each. What is the floor in tokens per second?",
    "How would speculative decoding with two drafted tokens change that estimate?",
    "And if the gateway reuses the KV cache of earlier turns, which part of a chat gets cheaper?",
    "Summarize the three numbers we discussed in one line.",
    "One more: what should we measure first next week?",
]
MODEL = "bench"
ADMIN = "bench-admin-token-0123456789abcdef"


class _Handler:
    """Exact provider that keeps only (linear, rows) of every call (no activations: memory stays flat)."""

    def __init__(self, pack: Any, ring: str):
        from pgw.engine import LocalComputeClient

        self.pack = pack
        self.local = LocalComputeClient(pack, rings=(ring,))
        self.ring = ring
        self.calls: list[tuple[str, int]] = []

    def linear_dims(self, fingerprint: str, ring: str) -> Any:
        return {s.id.key: (s.out_features, s.in_features) for s in self.pack.linear_specs()}

    async def compute(self, request: Any, ctx: Any) -> Any:
        self.calls.extend((it.linear, int(it.rows)) for it in request.items)
        return await self.local.compute(request)

    async def models(self) -> Any:
        return await self.local.models()


def fill(store: Any, pack: Any, ring: Any, slots: int, seg: int) -> float:
    from pgw.factory import MaskGenerator

    gen = MaskGenerator(pack, ring)
    t0 = time.perf_counter()
    left = slots
    while left > 0:
        n = min(left, seg)
        gen.generate_into_store(store, n, factory_id="bench")
        left -= n
    return slots / (time.perf_counter() - t0)


async def run_config(name: str, pack_dir: Path, pack: Any, ring: Any, work: Path, slots: int, turns: int,
                     max_tokens: int, settings_kw: dict[str, Any], model_kw: dict[str, Any]) -> dict[str, Any]:
    import httpx
    from fastapi import FastAPI

    from pgw.gateway import ApiKeyStore, GatewayModelConfig, GatewaySettings, create_app
    from pgw.maskstore import FileKeyProvider, MaskStore
    from pgw.maskstore.counters import FileMonotonicCounter
    from pgw.wire.server import create_router

    d = work / name
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    store = MaskStore.open(d / "store", FileKeyProvider(d / "kek", create=True),
                           counter=FileMonotonicCounter(d / "ctr"), low_water_slots=0)
    rate = fill(store, pack, ring, slots, 512)
    print(f"[{name}] filled {slots} slots at {rate:.1f} slots/s", flush=True)
    handler = _Handler(pack, ring.name)
    svc = FastAPI()
    svc.include_router(create_router(handler))
    keys = ApiKeyStore(None)
    key = keys.create("bench")[1]
    model = GatewayModelConfig(name=MODEL, pack_path=pack_dir, mode="masked", fingerprint=pack.fingerprint,
                               service_url="http://127.0.0.1:18443", ring=ring.name, **model_kw)
    settings = GatewaySettings(models=[model], admin_token=ADMIN, maskstore_root=d / "unused",
                               audit_log_file=d / "audit.jsonl", default_max_tokens=max_tokens, **settings_kw)
    app = create_app(settings, mask_store=store, transports={MODEL: httpx.ASGITransport(app=svc)}, key_store=keys)

    def avail() -> int:
        return int(sum(i.slots_available for i in store.inventory(pack.fingerprint, ring)))

    per_turn: list[dict[str, Any]] = []
    answers: list[Any] = []
    async with app.router.lifespan_context(app):
        rt = app.state.pgw.runtime.get(MODEL)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw", timeout=3600) as c:
            msgs: list[dict[str, str]] = [{"role": "system", "content": SYSTEM}]
            # warm-up request (own key/conversation; not counted)
            hdr = {"Authorization": f"Bearer {key}", "x-pgw-conversation-id": "warm"}
            w = await c.post("/v1/chat/completions", headers=hdr,
                             json={"model": MODEL, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 2,
                                   "temperature": 0})
            assert w.status_code == 200, w.text
            await rt.generator.wait_drained()
            await rt.generator.wait_closed()
            calls0 = len(handler.calls)
            s_start = avail()
            for t in range(turns):
                msgs.append({"role": "user", "content": USER[t % len(USER)]})
                before = avail()
                body = {"model": MODEL, "messages": msgs, "max_tokens": max_tokens, "temperature": 0,
                        "logprobs": True, "top_logprobs": 1}
                r = await c.post("/v1/chat/completions", json=body,
                                 headers={"Authorization": f"Bearer {key}", "x-pgw-conversation-id": "c1"})
                assert r.status_code == 200, r.text
                await rt.generator.wait_drained()
                await rt.generator.wait_closed()
                used = before - avail()
                res = r.json()
                answer = res["choices"][0]["message"]["content"]
                answers.append((answer, res["choices"][0]["logprobs"]))
                msgs.append({"role": "assistant", "content": answer})
                usage = res["usage"]
                gen = int(usage["completion_tokens"])
                cached = int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
                aud = [json.loads(x) for x in (d / "audit.jsonl").read_text().splitlines()]
                aud = [a for a in aud if a.get("event") == "inference"][-1]
                ttft = float(aud["ttft_ms"]) / 1e3
                wall = float(aud["duration_ms"]) / 1e3
                per_turn.append({"prompt": usage["prompt_tokens"], "cached": cached, "generated": gen,
                                 "slots": used, "reserved": aud["masks_reserved"], "burned": aud["masks_burned"],
                                 "finish": aud["finish_reason"], "ttft_s": round(ttft, 3),
                                 "decode_ms_per_token": round(1e3 * (wall - ttft) / max(1, gen - 1), 1)})
                print(f"[{name}] turn {t}: {per_turn[-1]}", flush=True)
            consumed = s_start - avail()
            # runs of calls to one linear as sorted row counts: a prompt pass's pipelined blocks are concurrent
            # requests whose arrival order varies between runs of any configuration
            calls: list[tuple[str, list[int]]] = []
            for lid, rows in handler.calls[calls0:]:
                if calls and calls[-1][0] == lid:
                    calls[-1][1].append(rows)
                else:
                    calls.append((lid, [rows]))
            calls = [(lid, sorted(rs)) for lid, rs in calls]
            ledger_refused = rt.mask_ledger.refused
    store.close()
    shutil.rmtree(d, ignore_errors=True)
    gen = sum(p["generated"] for p in per_turn)
    upfront = sum(p["prompt"] - p["cached"] + max_tokens for p in per_turn)
    return {"name": name, "settings": settings_kw, "model": model_kw, "turns": per_turn, "slots": consumed,
            "generated": gen, "slots_per_token": round(consumed / gen, 3),
            "upfront_slots": upfront, "upfront_slots_per_token": round(upfront / gen, 3),
            "prompt_rows": sum(p["prompt"] - p["cached"] for p in per_turn),
            "ttft_s_mean": round(sum(p["ttft_s"] for p in per_turn) / len(per_turn), 3),
            "decode_ms_per_token_mean": round(sum(p["decode_ms_per_token"] for p in per_turn) / len(per_turn), 1),
            "factory_slots_s": round(rate, 1), "ledger_refused": ledger_refused,
            "calls": calls, "answers": answers}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default=str(Path.home() / ".pgw-client-demo/packs/qwen2.5-0.5b-instruct"))
    ap.add_argument("--ring", default=None)
    ap.add_argument("--turns", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--work", required=True, help="scratch directory (deleted at the end)")
    ap.add_argument("--configs", default="window,incremental,window+reuse,incremental+reuse")
    ap.add_argument("--slots-noreuse", type=int, default=2600)
    ap.add_argument("--slots-reuse", type=int, default=1100)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    from pgw.core import DEFAULT_RING, RingParams
    from pgw.modelpack import ModelPack

    pack_dir = Path(args.pack).expanduser()
    pack = ModelPack.open(pack_dir)
    ring = RingParams.from_name(args.ring) if args.ring else DEFAULT_RING
    work = Path(args.work)
    base_model = {"float_ops": "native", "local_linears": "off"}
    out = []
    for cfg in args.configs.split(","):
        reuse = cfg.endswith("+reuse")
        mode = cfg.split("+")[0]
        skw: dict[str, Any] = {"mask_window_tokens": args.window}
        if mode == "incremental":
            skw.update(mask_reservation="incremental", mask_chunk_slots=args.chunk)
        mkw = {**base_model, **({"prefix_cache": "conversation"} if reuse else {})}
        slots = args.slots_reuse if reuse else args.slots_noreuse
        out.append(asyncio.run(run_config(cfg.replace("+", "_"), pack_dir, pack, ring, work, slots, args.turns,
                                          args.max_tokens, skw, mkw)))
    # same answers / provider calls within each prefix_cache setting
    by_reuse: dict[bool, list[dict[str, Any]]] = {}
    for r in out:
        by_reuse.setdefault(bool(r["model"].get("prefix_cache")), []).append(r)
    checks = {}
    for reuse, rs in by_reuse.items():
        checks[f"reuse={reuse}: answers+logprobs identical"] = all(r["answers"] == rs[0]["answers"] for r in rs)
        checks[f"reuse={reuse}: provider calls identical"] = all(r["calls"] == rs[0]["calls"] for r in rs)
    if len(by_reuse) == 2:
        same = by_reuse[True][0]["answers"] == by_reuse[False][0]["answers"]
        checks["answers identical with and without reuse"] = same
    summary = [{k: v for k, v in r.items() if k not in ("calls", "answers")} for r in out]
    for r in summary:
        print(f"{r['name']:>18}: {r['slots']} slots / {r['generated']} tokens = {r['slots_per_token']} slots/token "
              f"(up-front {r['upfront_slots_per_token']}); prompt rows {r['prompt_rows']}; TTFT {r['ttft_s_mean']} s; "
              f"decode {r['decode_ms_per_token_mean']} ms/token")
    for k, v in checks.items():
        print(f"{k}: {v}")
    if args.json:
        Path(args.json).write_text(json.dumps({"configs": summary, "checks": checks, "args": vars(args)}, indent=1))
    shutil.rmtree(work, ignore_errors=True)
    if not all(checks.values()):
        raise SystemExit("mismatch")


if __name__ == "__main__":
    main()
