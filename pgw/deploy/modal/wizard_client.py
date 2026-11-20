"""Drive the gateway's setup wizard over HTTP exactly like the console does, then test inference.

Calls, in order (same endpoints, cookie and CSRF header as the browser console):

    POST /console/api/setup/login      {"token": <setup token>}        -> session cookie + csrf
    POST /console/api/setup/enroll     {"code": <PGW1-...>}
    POST /console/api/setup/model      {"name": <model>}               (download + pack in background)
    GET  /console/api/setup/state      (poll until the pack is ready)
    POST /console/api/setup/maskstore  {}
    POST /console/api/setup/admin      {}                              -> admin token (shown once)
    POST /console/api/setup/verify     {}                              (repeat until masks are available)
    POST /console/api/setup/finish     {}

Then ``POST /admin/keys`` (admin token) and a greedy ``/v1/chat/completions`` with logprobs.

:func:`plaintext_reference` recomputes the same chat on the same pack with the plaintext reference
backend (no masks, no service), for a bit-for-bit comparison of tokens and logprobs.
"""

from __future__ import annotations

import time
from typing import Any

import httpx

CSRF_HEADER = "X-PGW-CSRF"


def _say(msg: str) -> None:
    print(f"[wizard] {msg}", flush=True)


class WizardError(RuntimeError):
    pass


def _check(r: httpx.Response, what: str) -> Any:
    if r.status_code >= 400:
        raise WizardError(f"{what}: HTTP {r.status_code}: {r.text[:800]}")
    return r.json() if r.content else {}


def run_wizard(base_url: str, setup_token: str, code: str, model: str, *, timeout_s: float = 2400,
               min_masks: int = 64) -> dict[str, Any]:
    """Complete the six wizard steps. Returns ``{"admin_token", "verify", "state", "timings"}``."""
    t_start = time.time()
    timings: dict[str, float] = {}
    with httpx.Client(base_url=base_url, timeout=120.0, trust_env=True, follow_redirects=False) as c:
        r = c.get("/console/setup")
        if r.status_code != 200 or "<html" not in r.text.lower():
            raise WizardError("the console page is not HTML")
        login = _check(c.post("/console/api/setup/login", json={"token": setup_token}), "login")
        h = {CSRF_HEADER: login["csrf"]}
        _say("signed in with the setup token")

        t = time.time()
        en = _check(c.post("/console/api/setup/enroll", json={"code": code}, headers=h), "enroll")
        timings["enroll_s"] = time.time() - t
        _say(f"enrolled as tenant {en['tenant']} with {en['service_url']} (client cert: {en['client_cert']}, "
             f"CA pinned: {bool(en.get('ca_fingerprint'))}); models: {[m['name'] for m in en['models']]}")

        t = time.time()
        _check(c.post("/console/api/setup/model", json={"name": model}, headers=h), "model")
        last = ""
        while True:
            st = _check(c.get("/console/api/setup/state"), "state")
            m = st.get("model") or {}
            job = (st.get("jobs") or {}).get("pack") or {}
            line = f"pack: {job.get('phase')} {job.get('done')}/{job.get('total')} {job.get('message', '')}"
            if line != last:
                _say(line)
                last = line
            if m.get("pack_status") == "ready":
                break
            if m.get("pack_status") in ("failed", "interrupted"):
                raise WizardError(f"model preparation failed: {m.get('error') or job.get('error')}")
            if time.time() - t_start > timeout_s:
                raise WizardError("timed out preparing the model")
            time.sleep(3)
        timings["model_s"] = time.time() - t
        _say(f"model ready (fingerprint {m['fingerprint'][:16]}..., matches the service's)")

        ms = _check(c.post("/console/api/setup/maskstore", json={}, headers=h), "maskstore")
        _say(f"mask store at {ms['root']}")
        adm = _check(c.post("/console/api/setup/admin", json={}, headers=h), "admin")
        admin_token = adm["admin_token"]
        _say("admin token created (shown once)")

        t = time.time()
        while True:
            v = _check(c.post("/console/api/setup/verify", json={}, headers=h), "verify")
            st = _check(c.get("/console/api/setup/state"), "state")
            masks = st.get("masks") or 0
            hint = (st.get("jobs") or {}).get("hint") or {}
            summary = ", ".join(f"{x['id']}={x['status']}" for x in v["checks"])
            _say(f"verify ok={v['ok']} masks={masks} hint={hint.get('phase')} {hint.get('done')}/"
                 f"{hint.get('total')} :: {summary}")
            fails = [x for x in v["checks"] if x["status"] == "fail"]
            if fails:
                raise WizardError(f"verify failed: {fails}")
            if masks >= min_masks:
                break
            if time.time() - t_start > timeout_s:
                raise WizardError("timed out waiting for masks")
            time.sleep(15)
        timings["verify_until_masks_s"] = time.time() - t

        fin = _check(c.post("/console/api/setup/finish", json={}, headers=h), "finish")
        _say(f"finish: mode={fin['mode']}: {fin['message'][:200]}")
        if fin["mode"] != "normal":
            raise WizardError(f"finish did not switch to normal mode: {fin}")
    timings["total_s"] = time.time() - t_start
    return {"admin_token": admin_token, "verify": v, "timings": timings, "enroll": en}


def create_api_key(base_url: str, admin_token: str) -> str:
    with httpx.Client(base_url=base_url, timeout=60.0, trust_env=True) as c:
        r = c.post("/admin/keys", json={"name": "modal-e2e"}, headers={"Authorization": f"Bearer {admin_token}"})
        return str(_check(r, "create key")["key"])


def wait_gateway_ready(base_url: str, timeout_s: float = 600) -> None:
    t0 = time.time()
    with httpx.Client(base_url=base_url, timeout=30.0, trust_env=True) as c:
        while time.time() - t0 < timeout_s:
            r = c.get("/readyz", params={"models": "all"})
            if r.status_code == 200:
                return
            time.sleep(3)
    raise WizardError("gateway did not become ready")


def chat(base_url: str, api_key: str, model: str, messages: list[dict[str, str]], max_tokens: int) -> dict[str, Any]:
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0, "logprobs": True,
            "top_logprobs": 0}
    with httpx.Client(base_url=base_url, timeout=900.0, trust_env=True) as c:
        t = time.time()
        r = c.post("/v1/chat/completions", json=body, headers={"Authorization": f"Bearer {api_key}"})
        out: dict[str, Any] = _check(r, "chat")
        out["_wall_s"] = time.time() - t
        return out


def admin_get(base_url: str, admin_token: str, path: str) -> Any:
    with httpx.Client(base_url=base_url, timeout=60.0, trust_env=True) as c:
        return _check(c.get(path, headers={"Authorization": f"Bearer {admin_token}"}), path)


def plaintext_reference(pack_path: str, ring: str, messages: list[dict[str, str]], max_tokens: int) -> dict[str, Any]:
    """Greedy generation on the same pack with the plaintext reference backend (in this process)."""
    import asyncio

    from pgw.core.ring import RingParams
    from pgw.engine import ChatTokenizer, Generator, PlainBackend, SamplingParams, TransformerModel
    from pgw.modelpack import ModelPack

    async def _run() -> dict[str, Any]:
        with ModelPack.open(pack_path, verify="fingerprint") as pack:
            tok = ChatTokenizer.from_pack(pack)
            ids = tok.apply_chat_template(messages, add_generation_prompt=True)
            model = TransformerModel(pack, PlainBackend(pack, RingParams.from_name(ring)))
            gen = Generator(model, tok, max_batch=1)
            params = SamplingParams(max_tokens=max_tokens, temperature=0.0, logprobs=0)
            res = await gen.agenerate(ids, params, session=None)
            await gen.aclose()
            model.close()
            return {"token_ids": list(res.token_ids), "logprobs": list(res.logprobs or []), "text": res.text,
                    "prompt_tokens": len(ids), "finish_reason": str(res.finish_reason),
                    "eos": sorted(getattr(tok, "eos_token_ids", ()) or ())}

    return asyncio.run(_run())


def compare(resp: dict[str, Any], ref: dict[str, Any]) -> dict[str, Any]:
    """Masked gateway answer vs plaintext reference: same text, same number of tokens, identical logprobs."""
    choice = resp["choices"][0]
    content = choice["logprobs"]["content"]
    lps = [x["logprob"] for x in content]
    n = len(lps)
    ref_lps = ref["logprobs"][:n]
    text = choice["message"]["content"]
    return {
        "tokens_masked": n,
        "tokens_reference": len(ref["token_ids"]),
        "text_equal": text == ref["text"],
        "logprobs_bit_identical": lps == ref_lps and len(ref_lps) == n,
        "first_mismatch": next((i for i, (a, b) in enumerate(zip(lps, ref_lps, strict=False)) if a != b), None),
        "max_abs_logprob_diff": max((abs(a - b) for a, b in zip(lps, ref_lps, strict=False)), default=0.0),
    }
