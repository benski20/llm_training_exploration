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
        config_dir: Path = Path("/srv/pgw/config")) -> dict[str, Any]:
    """Wizard + two chats + exactness check; see the module docstring."""
    while True:
        prov = live.get("provider") or {}
        if prov.get("run_id") == run_id and prov.get("code", "").startswith("PGW1-"):
            break
        if time.time() > deadline:
            raise RuntimeError("provider did not publish an enrollment code")
        time.sleep(5)
    res = wiz.run_wizard(url, token, prov["code"], boot.MODEL_NAME, timeout_s=max(60, deadline - time.time()))
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
                    wiz.client_timing(wiz.admin_get(local_url, adm, "/admin/status"), boot.MODEL_NAME))
        except Exception as exc:
            snap_errors.append(f"{type(exc).__name__}: {str(exc)[:200]}")
            return "", {}

    s0 = snap()
    out = wiz.chat(url, key, boot.MODEL_NAME, PROMPT, 32)
    s1 = snap()
    out2 = wiz.chat(url, key, boot.MODEL_NAME, PROMPT, 32)  # second run: determinism + warm timing
    s2 = snap()
    # The warm chat (the second one) decides launch.py's --max-trip-ms gate; both chats are reported too.
    trips: dict[str, Any] = {"warm": trip_gate.trip_latency(s1[0], s2[0], s1[1], s2[1]),
                             "both_chats": trip_gate.trip_latency(s0[0], s2[0], s0[1], s2[1]),
                             "path": path}
    if snap_errors:
        trips["errors"] = snap_errors
    st = json.loads((config_dir / "setup" / "state.json").read_text())
    t = time.time()
    ref = wiz.plaintext_reference(st["model"]["pack_path"], st["model"].get("ring") or boot.RING, PROMPT, 32)
    ref_s = time.time() - t
    cmp = wiz.compare(out, ref)
    status = wiz.admin_get(url, res["admin_token"], "/admin/status")
    masks = wiz.admin_get(url, res["admin_token"], "/admin/masks")
    model_status = next((m for m in status.get("models", []) if m.get("name") == boot.MODEL_NAME), {})
    speed = dict(model_status.get("speed") or {})
    speed.setdefault("rtt_path", path)
    warm_tokens = int(((out2.get("usage") or {}).get("completion_tokens")) or 0)
    if warm_tokens and trips["warm"].get("trips"):
        speed["warm_trips_per_token"] = round(trips["warm"]["trips"] / warm_tokens, 2)
    if warm_tokens and out2.get("_wall_s"):
        speed["warm_tokens_per_s"] = round(warm_tokens / float(out2["_wall_s"]), 2)
    return {
        "timings": res["timings"], "verify": res["verify"], "enroll": {k: res["enroll"].get(k) for k in (
            "tenant", "service_url", "client_cert", "ca_fingerprint")},
        "chat_text": out["choices"][0]["message"]["content"], "usage": out.get("usage"),
        "chat_wall_s": out["_wall_s"], "chat2_wall_s": out2["_wall_s"],
        "repeat_identical": out["choices"][0]["message"]["content"] == out2["choices"][0]["message"]["content"]
        and [x["logprob"] for x in out["choices"][0]["logprobs"]["content"]]
        == [x["logprob"] for x in out2["choices"][0]["logprobs"]["content"]],
        "reference_text": ref["text"], "reference_s": ref_s, "compare": cmp,
        "admin_status": status, "admin_masks": masks, "trip_latency": trips, "speed": speed,
        "network_path": path,
    }
