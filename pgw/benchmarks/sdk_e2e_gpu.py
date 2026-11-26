#!/usr/bin/env python3
"""The client library end to end on a GPU host: provider service on CUDA + ``from pgw import OpenAI`` on loopback.

What ``tests/sdk/test_e2e_product.py`` does on CPU, for a release model on a real GPU and with the library's
defaults (pks-v3 masks, ``hint="local"``, masked LM head when the provider serves it):

1. provider side: pack the public checkpoint at the pinned commit (what the provider image does on first boot),
   check the fingerprint against the library's baked pin, start ``pgw service`` on CUDA with accounts, provisioning,
   the console and the TCP+TLS lane; create an SDK key through the admin API;
2. client side, empty cache: ``warmup`` (enroll + provision: gateway pack and factory pack from the provider,
   fingerprint-checked), then the first request (waits for the local pks-v3 hint build and the first masks),
   non-streaming and streaming, a longer greedy stream for time to first token and steady tokens/s, a short
   multi-turn conversation, then key revocation (the connected client and a fresh one are refused);
3. afterwards (provider and client stopped, to free memory): the existing gateway path (``pgw gateway`` app,
   plaintext reference mode, same model settings) answers every recorded request; each must be bit-identical
   (text and log-probabilities, top alternatives included).

    python benchmarks/sdk_e2e_gpu.py --model qwen2.5-7b-instruct --work /tmp/sdke2e --out /kaggle/working/sdk_e2e

Everything is written to ``--out``: ``results.json`` (timings, checks), ``status.jsonl`` (client status, memory and
GPU samples every 15 s), ``client.log`` (the library's JSON logs), ``provider.log``. On a T4 the timings are a
correctness check, not representative speed.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

SYSTEM = {"role": "system", "content": "You are a concise assistant."}
PROMPT = [SYSTEM, {"role": "user", "content": "Name a colour of the sky. Secret-marker-7f3a."}]
GREEDY = {"temperature": 0, "max_tokens": 8, "logprobs": True, "top_logprobs": 2, "extra_body": {"ignore_eos": True}}
SAMPLED = {"temperature": 0.8, "top_p": 0.95, "seed": 77, "max_tokens": 8, "logprobs": True, "top_logprobs": 2,
           "extra_body": {"ignore_eos": True}}
LONG_PROMPT = [SYSTEM, {"role": "user", "content": "Write a short paragraph about why the sky is blue."}]
LONG = {"temperature": 0, "max_tokens": 64, "logprobs": True, "top_logprobs": 1, "extra_body": {"ignore_eos": True}}
TURNS = ["My name is Ada and I like tea. Reply in one sentence.",
         "What is my name, and what do I like? Answer briefly.",
         "Suggest one snack that goes with it, in five words or fewer."]
TURN = {"temperature": 0, "max_tokens": 32, "logprobs": True, "top_logprobs": 1}

R: dict[str, Any] = {"checks": {}, "timings": {}, "errors": []}
OUT = Path(".")
T0 = time.monotonic()


def log(*a: Any) -> None:
    print(f"[{(time.monotonic() - T0) / 60:6.1f}m]", *a, flush=True)


def save() -> None:
    (OUT / "results.json").write_text(json.dumps(R, indent=1, default=str))


def check(name: str, ok: bool, detail: Any = None) -> bool:
    R["checks"][name] = {"pass": bool(ok), **({"detail": detail} if detail is not None else {})}
    log("CHECK", name, "PASS" if ok else "FAIL", "" if detail is None else str(detail)[:300])
    save()
    return bool(ok)


def mem_sample() -> dict[str, Any]:
    out: dict[str, Any] = {"t": round(time.monotonic() - T0, 1)}
    with contextlib.suppress(Exception):
        mi = dict(ln.split(":", 1) for ln in Path("/proc/meminfo").read_text().splitlines() if ":" in ln)
        out["mem_avail_gb"] = round(int(mi["MemAvailable"].split()[0]) / 2**20, 2)
    with contextlib.suppress(Exception):
        q = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=20).stdout.strip()
        out["gpu"] = q
    return out


class Sampler(threading.Thread):
    """Client status + host memory + GPU every ``every`` seconds into status.jsonl; tracks the memory low."""

    def __init__(self, every: float = 15.0):
        super().__init__(daemon=True)
        self.every, self.client, self.stop_ev = every, None, threading.Event()
        self.min_avail = 1e9

    def run(self) -> None:
        with (OUT / "status.jsonl").open("a") as f:
            while not self.stop_ev.is_set():
                s = mem_sample()
                self.min_avail = min(self.min_avail, s.get("mem_avail_gb", 1e9))
                R["min_mem_avail_gb"] = self.min_avail
                if self.client is not None:
                    with contextlib.suppress(Exception):
                        s["status"] = self.client.status()
                f.write(json.dumps(s, default=str) + "\n")
                f.flush()
                self.stop_ev.wait(self.every)


# ------------------------------------------------------------------------------------------- provider side
def provider_pack(model: str, work: Path, drop_checkpoint: bool) -> tuple[Path, str]:
    from huggingface_hub import snapshot_download

    from pgw.modelpack import ModelPack, pack_checkpoint
    from pgw.sdk.provisioning import PINNED_MODELS

    pin = PINNED_MODELS[model]
    dst = work / "provider-pack"
    if not (dst / "manifest.json").exists():
        t = time.monotonic()
        snapshot_download(pin.hf_repo, revision=pin.revision,
                          allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.tiktoken"])
        R["timings"]["provider_checkpoint_download_s"] = round(time.monotonic() - t, 1)
        t = time.monotonic()
        os.environ["HF_HUB_OFFLINE"] = "1"
        try:  # exactly what the provider image does on first boot (deploy/runpod/pod_boot.ensure_pack)
            pack_checkpoint(pin.hf_repo, dst, name=model, revision=pin.revision)
        finally:
            os.environ.pop("HF_HUB_OFFLINE", None)
        R["timings"]["provider_pack_s"] = round(time.monotonic() - t, 1)
        hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
        if drop_checkpoint:  # the provider image drops it once packed (disk)
            shutil.rmtree(hub / ("models--" + pin.hf_repo.replace("/", "--")), ignore_errors=True)
    with ModelPack.open(dst) as p:
        fp = p.fingerprint
    R["fingerprint"] = fp
    check("provider_pack_equals_baked_pin", pin.fingerprint is not None and fp == pin.fingerprint,
          {"provider": fp, "baked": pin.fingerprint})
    return dst, fp


def start_gpu_provider(work: Path, pack: Path, pki: dict[str, Any], device: str, ready_s: float) -> Any:
    from tests.sdk import conftest

    env = {"PGW_SERVICE_DEVICE": device, "PGW_SERVICE_WARMUP": "true", "PGW_SERVICE_TRIP_STREAM_PRIORITY": "high",
           "OMP_NUM_THREADS": "2"}
    (work / "provider").mkdir(exist_ok=True)
    t = time.monotonic()
    prov = conftest.start_provider(work / "provider", pack, pki, extra_env=env, ready_timeout_s=ready_s)
    R["timings"]["provider_start_s"] = round(time.monotonic() - t, 1)
    return prov


# --------------------------------------------------------------------------------------------- client side
def run_client(prov: Any, model: str, work: Path, sampler: Sampler, overrides: dict[str, Any]
               ) -> list[dict[str, Any]]:
    """The customer's side with the library's defaults; returns the recorded requests for the reference."""
    from pgw import OpenAI
    from pgw.sdk import AuthenticationError, PGWAPIError
    from pgw.sdk.options import SDKOptions

    d = SDKOptions()
    R["sdk_defaults"] = {"mask_scheme": d.mask_scheme, "hint": d.hint, "masked_head": d.masked_head,
                         "workers": d.workers}
    key = prov.new_key("kaggle-e2e", models=[model])
    root = work / "client"
    shutil.rmtree(root, ignore_errors=True)
    handler = logging.FileHandler(OUT / "client.log")
    handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
    logging.getLogger("pgw").addHandler(handler)
    logging.getLogger("pgw").setLevel(logging.INFO)
    # only paths and generous first-use budgets differ from the defaults (the 7B hint build runs on this host)
    R["client_overrides"] = overrides  # {} on the validation runs (local smoke tests only)
    c = OpenAI(api_key=key, cache_dir=root / "cache", state_dir=root / "state", timeout=7200, ready_timeout_s=7200,
               **overrides)
    sampler.client = c
    rec: list[dict[str, Any]] = []
    try:
        t = time.monotonic()
        c.warmup(model)
        R["timings"]["client_enroll_provision_s"] = round(time.monotonic() - t, 1)
        st = c.status()["models"][model]
        R["after_warmup"] = st
        log("warmup done", json.dumps(st, default=str)[:1500])
        save()

        t = time.monotonic()
        r = c.chat.completions.create(model=model, messages=PROMPT, **GREEDY)
        R["timings"]["first_request_after_warmup_s"] = round(time.monotonic() - t, 1)
        rec.append({"name": "greedy", "messages": PROMPT, "body": GREEDY, "stream": False, "resp": r.to_dict()})
        log("first answer", repr(r.choices[0].message.content))
        save()

        chunks = list(c.chat.completions.create(model=model, messages=PROMPT, stream=True,
                                                stream_options={"include_usage": True}, **SAMPLED))
        rec.append({"name": "sampled_stream", "messages": PROMPT, "body": SAMPLED, "stream": True,
                    "chunks": [ch.to_dict() for ch in chunks]})

        for i in range(2):  # time to first token and steady decode, streaming
            t = time.monotonic()
            times, chunks = [], []
            for ch in c.chat.completions.create(model=model, messages=LONG_PROMPT, stream=True,
                                                stream_options={"include_usage": True}, **LONG):
                if ch.choices and ch.choices[0].delta.content:
                    times.append(time.monotonic() - t)
                chunks.append(ch)
            n = len(times)
            R["timings"][f"long_stream_{i}"] = {
                "ttft_s": round(times[0], 3), "total_s": round(time.monotonic() - t, 2), "content_chunks": n,
                "completion_tokens": chunks[-1].usage.completion_tokens if chunks[-1].usage else None,
                "steady_tok_s": round((n - 1) / (times[-1] - times[0]), 2) if n > 1 else None}
            rec.append({"name": f"long_stream_{i}", "messages": LONG_PROMPT, "body": LONG, "stream": True,
                        "chunks": [ch.to_dict() for ch in chunks]})
            log("long stream", R["timings"][f"long_stream_{i}"])
            save()

        msgs: list[dict[str, Any]] = [SYSTEM]
        for i, u in enumerate(TURNS):  # a short conversation: each turn carries the history so far
            msgs = [*msgs, {"role": "user", "content": u}]
            t = time.monotonic()
            r = c.chat.completions.create(model=model, messages=msgs, **TURN)
            R["timings"][f"turn_{i}_s"] = round(time.monotonic() - t, 2)
            rec.append({"name": f"turn_{i}", "messages": msgs, "body": TURN, "stream": False, "resp": r.to_dict()})
            msgs = [*msgs, {"role": "assistant", "content": r.choices[0].message.content}]
            log("turn", i, repr(r.choices[0].message.content))
        R["conversation"] = msgs
        save()

        st = c.status()["models"][model]
        R["status_before_revoke"] = st
        check("no_integrity_failures", st.get("integrity_failures") == 0, st.get("integrity_failures"))
        f = st.get("factory") or {}
        check("factory_scheme_not_halted", f.get("scheme") == c.options.mask_scheme and not f.get("halted")
              and not f.get("provider_integrity_failures"), {k: f.get(k) for k in ("scheme", "halted")})
        if c.options.mask_scheme == "pks-v3":
            check("hint_built_locally", bool(list((root / "cache" / "models").glob("*/hint/*"))))
        R["masked_head"] = bool(st.get("masked_head"))

        with contextlib.suppress(Exception):
            R["provider_usage"] = prov.admin("GET", f"/orgs/{prov.org}/usage")["usage"]
            log_ = prov.admin("GET", f"/orgs/{prov.org}/requests")
            R["provider_request_log"] = log_["requests"]
            check("no_content_in_provider_records",
                  "Secret-marker-7f3a" not in json.dumps(log_) + json.dumps(R["provider_usage"]))

        prov.revoke(key)
        time.sleep(1.5)
        try:
            c.chat.completions.create(model=model, messages=PROMPT, **GREEDY)
            check("revoked_key_refused_connected_client", False, "the call succeeded")
        except PGWAPIError as e:
            check("revoked_key_refused_connected_client", e.code in ("unauthenticated", "permission_denied"),
                  f"{type(e).__name__}: {e.code}")
    finally:
        sampler.client = None
        c.close()
    fresh = OpenAI(api_key=key, cache_dir=root / "fresh" / "cache", state_dir=root / "fresh" / "state")
    try:
        fresh.chat.completions.create(model=model, messages=PROMPT, **GREEDY)
        check("revoked_key_refused_fresh_client", False, "the call succeeded")
    except AuthenticationError as e:
        check("revoked_key_refused_fresh_client", True, f"AuthenticationError: {e.code}")
    except Exception as e:
        check("revoked_key_refused_fresh_client", False, f"{type(e).__name__}: {e}")
    finally:
        fresh.close()
    return rec


# ---------------------------------------------------------------------------------------------- reference
def reference(pack: Path, model: str, recs: list[dict[str, Any]], head_mode: str) -> list[dict[str, Any]]:
    """The existing gateway (``pgw gateway`` app, plaintext reference mode) with the client's model settings."""
    import httpx

    from pgw.gateway import ApiKeyStore, GatewayModelConfig, GatewaySettings, create_app

    keys = ApiKeyStore(None)
    key = keys.create("ref")[1]
    cfg = GatewayModelConfig(name=model, pack_path=pack, mode="plain", head_mode=head_mode, float_ops="native")
    app = create_app(GatewaySettings(models=[cfg], admin_token="ref-admin-token-0123456789", allow_dev_modes=True,
                                     default_max_tokens=8), key_store=keys)

    async def run() -> list[dict[str, Any]]:
        out = []
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://ref.test",
                                         timeout=3600) as h:
                done: dict[str, Any] = {}  # the same request twice (the repeated greedy stream) is answered once
                for rec in recs:
                    req = {k: v for k, v in rec["body"].items() if k != "extra_body"}
                    req.update(rec["body"].get("extra_body") or {})
                    k = json.dumps([rec["messages"], req], sort_keys=True)
                    if k in done:
                        out.append(done[k])
                        continue
                    t = time.monotonic()
                    r = await h.post("/v1/chat/completions", json={"model": model, "messages": rec["messages"], **req},
                                     headers={"Authorization": f"Bearer {key}"})
                    assert r.status_code == 200, r.text
                    done[k] = r.json()
                    out.append(done[k])
                    log("reference", rec["name"], f"{time.monotonic() - t:.1f}s")
        return out

    return asyncio.run(run())


def compare(recs: list[dict[str, Any]], refs: list[dict[str, Any]]) -> None:
    allok = True
    for rec, ref in zip(recs, refs, strict=True):
        rc = ref["choices"][0]
        if rec["stream"]:
            chunks = rec["chunks"]
            text = "".join((ch["choices"][0]["delta"].get("content") or "") for ch in chunks if ch.get("choices"))
            lps = [t for ch in chunks if ch.get("choices") and ch["choices"][0].get("logprobs")
                   for t in ch["choices"][0]["logprobs"]["content"]]
        else:
            text = rec["resp"]["choices"][0]["message"]["content"]
            lps = rec["resp"]["choices"][0]["logprobs"]["content"]
        ok = text == rc["message"]["content"] and lps == rc["logprobs"]["content"]
        allok &= check(f"bit_identical_{rec['name']}", ok,
                       None if ok else {"sdk": text, "reference": rc["message"]["content"]})
    check("bit_identical_all", allok)


def main() -> int:
    global OUT
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="qwen2.5-7b-instruct")
    ap.add_argument("--work", type=Path, default=Path("/tmp/sdke2e"))
    ap.add_argument("--out", type=Path, default=Path("sdk_e2e_out"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--provider-ready-s", type=float, default=2400)
    ap.add_argument("--drop-checkpoint", action="store_true", help="delete the HF checkpoint once packed")
    ap.add_argument("--client-opts", type=json.loads, default={},
                    help="JSON option overrides (local smoke tests only; the validation uses the defaults)")
    a = ap.parse_args()
    OUT = a.out
    OUT.mkdir(parents=True, exist_ok=True)
    a.work.mkdir(parents=True, exist_ok=True)
    R["model"], R["device"] = a.model, a.device
    import torch

    gpus = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    R["env"] = {"torch": torch.__version__, "gpu": gpus, "cpus": os.cpu_count()}
    from pgw.core import native

    with contextlib.suppress(Exception):
        from pgw.hemask import ntt, pks_native

        R["env"]["native"] = {"pgw_fast": native.fast is not None, "pgw_ntt": ntt.available(),
                              "pgw_pks": bool(getattr(pks_native, "available", lambda: None)())}
    log("env", R["env"])
    sampler = Sampler()
    sampler.start()
    prov = None
    recs: list[dict[str, Any]] = []
    try:
        pack, _fp = provider_pack(a.model, a.work, a.drop_checkpoint)
        save()
        from tests.sdk.conftest import make_pki

        pki_dir = a.work / "pki"
        pki_dir.mkdir(exist_ok=True)
        prov = start_gpu_provider(a.work, pack, make_pki(pki_dir), a.device, a.provider_ready_s)
        log("provider ready", R["timings"]["provider_start_s"], "s")
        recs = run_client(prov, a.model, a.work, sampler, a.client_opts)
    except BaseException as e:
        R["errors"].append(traceback.format_exc())
        log("FAILED", repr(e))
    finally:
        if prov is not None:
            from tests.sdk.conftest import stop_provider

            stop_provider(prov)
            shutil.copy(prov.log, OUT / "provider.log")
        save()
    (OUT / "sdk_requests.json").write_text(json.dumps(recs, indent=1, default=str))
    if recs:
        try:
            head = "provider_int8" if R.get("masked_head") else "gateway"
            R["reference_head_mode"] = head
            t = time.monotonic()
            refs = reference(a.work / "provider-pack", a.model, recs, head)
            R["timings"]["reference_s"] = round(time.monotonic() - t, 1)
            (OUT / "reference.json").write_text(json.dumps(refs, indent=1))
            compare(recs, refs)
        except BaseException:
            R["errors"].append(traceback.format_exc())
    sampler.stop_ev.set()
    R["total_min"] = round((time.monotonic() - T0) / 60, 1)
    save()
    fails = [k for k, v in R["checks"].items() if not v["pass"]]
    log("DONE", "failed checks:", fails, "errors:", len(R["errors"]))
    return 1 if fails or R["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
