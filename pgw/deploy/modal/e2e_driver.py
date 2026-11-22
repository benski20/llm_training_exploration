"""The scripted end-to-end test of the Modal demo (``launch.py --e2e``); no Modal imports.

Runs in the customer's container (``customer.py``) or in the co-located container (``colocated.py``):
waits for the provider's enrollment code in the ``pgw-live`` Dict (passed in as ``live``, read with
``.get`` only), drives the setup wizard through the gateway's URL, sends two greedy chats, recomputes
the answer with the plaintext reference backend on the same pack and returns the comparison plus the
gateway's client-reported round-trip latency (``trip_gate.py``) and speed settings. Both latency
reports name the network path the gateway used to reach the provider (``network_path``: loopback,
private (i6pn) or public (tunnel)), taken from the service URL the gateway enrolled with.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pgw_modal_boot as boot
import trip_gate
import wizard_client as wiz

PROMPT = [{"role": "user", "content": "Explain in two sentences why one-time pads are secure."}]


def run(live: Any, run_id: str, url: str, token: str, deadline: float, local_url: str,
        config_dir: Path = Path("/srv/pgw/config"), spec_rows: int = 0, sustain_s: float = 0.0,
        sampler_path: str | None = None, float_threads: int | None = None,
        float_ops: str | None = None) -> dict[str, Any]:
    """Wizard + two chats + exactness check; see the module docstring.

    ``spec_rows`` (launch.py --spec-rows): the gateway runs strict speculation with that many rows per pass, so
    the wizard waits for enough mask slots for both chats and the warm window starts after the first chat's
    dummy passes have drained."""
    while True:
        prov = live.get("provider") or {}
        if prov.get("run_id") == run_id and prov.get("code", "").startswith("PGW1-"):
            break
        if time.time() > deadline:
            raise RuntimeError("provider did not publish an enrollment code")
        time.sleep(5)
    model_name = prov.get("model") or boot.MODEL_NAME  # the target the provider serves (launch.py --model)
    # strict speculation reserves spec_rows slots per generated token up front: wait for both chats' slots
    res = wiz.run_wizard(url, token, prov["code"], model_name, timeout_s=max(60, deadline - time.time()),
                         min_masks=boot.e2e_min_masks(spec_rows))
    wiz.wait_gateway_ready(url)
    key = wiz.create_api_key(url, res["admin_token"])
    adm = res["admin_token"]
    path = boot.network_path((res.get("enroll") or {}).get("service_url") or prov.get("service_url"))

    snap_errors: list[str] = []

    def snap() -> tuple[str, dict]:
        # The gateway's own counters, read on loopback (not through Modal's edge). A failed read leaves
        # the latency unmeasured (the gate then reports "no round trips measured"), never the exactness check.
        try:
            return (wiz.metrics_text(local_url, adm),
                    wiz.client_timing(wiz.admin_get(local_url, adm, "/admin/status"), model_name))
        except Exception as exc:
            snap_errors.append(f"{type(exc).__name__}: {str(exc)[:200]}")
            return "", {}

    s0 = snap()
    out = wiz.chat(url, key, model_name, PROMPT, 32)
    # Strict speculation: the dummy passes of the first chat run after its answer, and the next request of the
    # same conversation (this API key) waits for them; wait here so the warm window holds one chat's passes.
    drain_s = wiz.wait_idle(local_url, adm, model_name) if spec_rows else 0.0
    s1 = snap()
    tw0 = time.time()
    out2 = wiz.chat(url, key, model_name, PROMPT, 32)  # second run: determinism + warm timing
    tw1 = time.time()
    s2 = snap()
    # The warm chat (the second one) decides launch.py's --max-trip-ms gate; both chats are reported too.
    trips: dict[str, Any] = {"warm": trip_gate.trip_latency(s1[0], s2[0], s1[1], s2[1]),
                             "both_chats": trip_gate.trip_latency(s0[0], s2[0], s0[1], s2[1]),
                             "path": path}
    if snap_errors:
        trips["errors"] = snap_errors
    # The same warm chat sent to the gateway on loopback (no Modal HTTPS edge on the way): what the
    # gateway itself delivers; reported next to the edge figure, never instead of it. Same answer required.
    out3 = wiz.chat(local_url, key, model_name, PROMPT, 32) if not spec_rows else None
    st = json.loads((config_dir / "setup" / "state.json").read_text())
    t = time.time()
    ref = wiz.plaintext_reference(st["model"]["pack_path"], st["model"].get("ring") or boot.RING, PROMPT, 32,
                                  float_threads=float_threads, float_ops=float_ops)
    ref_s = time.time() - t
    cmp = wiz.compare(out, ref)
    sustain: dict[str, Any] | None = None
    if sustain_s > 0:  # chats back to back (launch.py --sustain-s): tokens/s and factory rate under decode load
        if sampler_path:  # PGW_STACK_SAMPLER in the gateway: count the sustain window alone
            Path(sampler_path + ".reset").touch()
        t0, toks, n, wall, err = time.time(), 0, 0, 0.0, None
        while time.time() - t0 < sustain_s:
            try:
                o = wiz.chat(local_url, key, model_name, PROMPT, 64)
            except Exception as exc:  # e.g. masks ran out: report, never fail the exactness result
                err = f"{type(exc).__name__}: {str(exc)[:300]}"
                break
            toks += int(((o.get("usage") or {}).get("completion_tokens")) or 0)
            wall += float(o["_wall_s"])
            n += 1
        t1 = time.time()
        sustain = {"t0": t0, "t1": t1, "chats": n, "tokens": toks, "seconds": round(t1 - t0, 2),
                   "tokens_per_s": round(toks / max(1e-9, t1 - t0), 2),
                   "tokens_per_s_in_chats": round(toks / max(1e-9, wall), 2), "error": err}
    status = wiz.admin_get(url, res["admin_token"], "/admin/status")
    masks = wiz.admin_get(url, res["admin_token"], "/admin/masks")
    model_status = next((m for m in status.get("models", []) if m.get("name") == model_name), {})
    speed = dict(model_status.get("speed") or {})
    speed.setdefault("rtt_path", path)
    warm_tokens = int(((out2.get("usage") or {}).get("completion_tokens")) or 0)
    if warm_tokens and trips["warm"].get("trips"):
        speed["warm_trips_per_token"] = round(trips["warm"]["trips"] / warm_tokens, 2)
    if warm_tokens and out2.get("_wall_s"):
        speed["warm_tokens_per_s"] = round(warm_tokens / float(out2["_wall_s"]), 2)
    if out3 is not None and out3.get("_wall_s"):
        t3 = int(((out3.get("usage") or {}).get("completion_tokens")) or 0)
        speed["warm_loopback_tokens_per_s"] = round(t3 / float(out3["_wall_s"]), 2)
    if spec_rows:
        speed["spec_drain_wait_s"] = round(drain_s, 1)
        if warm_tokens:
            speed["warm_answer_tokens_per_s"] = round(warm_tokens / max(1e-9, out2["_wall_s"]), 2)
    return {
        "model": model_name, "spec_rows": spec_rows,
        "timings": res["timings"], "verify": res["verify"], "enroll": {k: res["enroll"].get(k) for k in (
            "tenant", "service_url", "client_cert", "ca_fingerprint")},
        "chat_text": out["choices"][0]["message"]["content"], "usage": out.get("usage"),
        "chat_wall_s": out["_wall_s"], "chat2_wall_s": out2["_wall_s"],
        "repeat_identical": out["choices"][0]["message"]["content"] == out2["choices"][0]["message"]["content"]
        and [x["logprob"] for x in out["choices"][0]["logprobs"]["content"]]
        == [x["logprob"] for x in out2["choices"][0]["logprobs"]["content"]],
        "loopback_repeat_identical": None if out3 is None else (
            out3["choices"][0]["message"]["content"] == out["choices"][0]["message"]["content"]
            and [x["logprob"] for x in out3["choices"][0]["logprobs"]["content"]]
            == [x["logprob"] for x in out["choices"][0]["logprobs"]["content"]]),
        "reference_text": ref["text"], "reference_s": ref_s, "compare": cmp,
        "admin_status": status, "admin_masks": masks, "trip_latency": trips, "speed": speed,
        "network_path": path, "sustain": sustain, "warm_window": [tw0, tw1],
    }
