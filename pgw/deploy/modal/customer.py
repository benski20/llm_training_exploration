"""Party 2 on Modal: the customer's gateway (``pgw-customer``, CPU only).

The image installs pgw (CPU torch) from this repository; it contains no model. :func:`gateway` starts
``pgw gateway serve`` with an empty configuration, so it opens the first-run setup wizard, exposes the
console on a public HTTPS URL (``modal.forward``: TLS ends at Modal's edge) and publishes
``{console_url, setup_token}`` in the ``pgw-live`` Dict. During the wizard the gateway downloads the
model from Hugging Face itself, packs it and checks the fingerprint against the provider's, builds the
encrypted-mask hint and runs the encrypted mask factory, all inside this container.

With ``e2e=True`` the function also plays the administrator: it drives the wizard through the public
URL with the provider's enrollment code (``wizard_client.py``), sends a greedy chat completion, recomputes
it with the plaintext reference backend on the same pack and stores the comparison in the Dict.

Config, secrets, packs and the mask store live on the container's local disk, not on a ``modal.Volume``:
the mask store must never sit on network or snapshotted storage (a restored copy could hand out masks
twice), and this demo is single-use anyway. Use ``launch.py`` to start it.
"""

from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
# Inside the container this file is /root/<name>.py and the source tree is not there (nor needed).
PGW = HERE.parents[1] if modal.is_local() else Path("/opt/pgw-src")
MAX_LIFETIME_S = 88 * 60  # hard cap (function timeout): nothing outlives 90 minutes
DICT_NAME = "pgw-live"

image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install("torch==2.15.0", index_url="https://download.pytorch.org/whl/cpu")
    .add_local_dir(PGW / "src", "/opt/pgw-src/src", copy=True, ignore=["**/__pycache__", "**/*.so"])
    .add_local_file(PGW / "pyproject.toml", "/opt/pgw-src/pyproject.toml", copy=True)
    .add_local_file(PGW / "README.md", "/opt/pgw-src/README.md", copy=True)
    .add_local_file(PGW / "deploy/hf-space/constraints.txt", "/opt/pgw-src/constraints.txt", copy=True)
    .run_commands("pip install --no-cache-dir -c /opt/pgw-src/constraints.txt '/opt/pgw-src[gateway,factory,pack]'")
    .env({"PYTHONUNBUFFERED": "1", "HF_HUB_DISABLE_PROGRESS_BARS": "1", "OMP_WAIT_POLICY": "PASSIVE"})
    .add_local_file(HERE / "pgw_modal_boot.py", "/opt/pgw-modal/pgw_modal_boot.py")
    .add_local_file(HERE / "wizard_client.py", "/opt/pgw-modal/wizard_client.py")
)

app = modal.App("pgw-customer", image=image)

PROMPT = [{"role": "user", "content": "Explain in two sentences why one-time pads are secure."}]


def _e2e(live: modal.Dict, run_id: str, url: str, token: str, deadline: float) -> dict:
    import pgw_modal_boot as boot
    import wizard_client as wiz

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
    out = wiz.chat(url, key, boot.MODEL_NAME, PROMPT, 32)
    out2 = wiz.chat(url, key, boot.MODEL_NAME, PROMPT, 32)  # second run: determinism + warm timing
    st = json.loads(Path("/srv/pgw/config/setup/state.json").read_text())
    t = time.time()
    ref = wiz.plaintext_reference(st["model"]["pack_path"], st["model"].get("ring") or boot.RING, PROMPT, 32)
    ref_s = time.time() - t
    cmp = wiz.compare(out, ref)
    status = wiz.admin_get(url, res["admin_token"], "/admin/status")
    masks = wiz.admin_get(url, res["admin_token"], "/admin/masks")
    return {
        "timings": res["timings"], "verify": res["verify"], "enroll": {k: res["enroll"].get(k) for k in (
            "tenant", "service_url", "client_cert", "ca_fingerprint")},
        "chat_text": out["choices"][0]["message"]["content"], "usage": out.get("usage"),
        "chat_wall_s": out["_wall_s"], "chat2_wall_s": out2["_wall_s"],
        "repeat_identical": out["choices"][0]["message"]["content"] == out2["choices"][0]["message"]["content"]
        and [x["logprob"] for x in out["choices"][0]["logprobs"]["content"]]
        == [x["logprob"] for x in out2["choices"][0]["logprobs"]["content"]],
        "reference_text": ref["text"], "reference_s": ref_s, "compare": cmp,
        "admin_status": status, "admin_masks": masks,
    }


@app.function(cpu=4.0, memory=8192, timeout=MAX_LIFETIME_S, max_containers=1, min_containers=0, retries=0)
def gateway(run_id: str, ttl_s: int = 80 * 60, e2e: bool = False) -> dict:
    import sys

    sys.path.insert(0, "/opt/pgw-modal")
    import pgw_modal_boot as boot

    deadline = time.time() + min(ttl_s, MAX_LIFETIME_S - 120)
    live = modal.Dict.from_name(DICT_NAME, create_if_missing=True)
    root = Path("/srv/pgw")
    port = 8080
    result: dict = {"run_id": run_id}
    with modal.forward(port) as tunnel:
        url = tunnel.url
        env = boot.gateway_env(root / "config", root / "data", port)
        gw, token = boot.start_gateway(env, root / "gateway.log")
        boot.wait_ready(f"http://127.0.0.1:{port}/healthz", gw)
        info = {"run_id": run_id, "console_url": url.rstrip("/") + "/console/", "base_url": url,
                "setup_token": token, "started_at": time.time(), "expires_at": deadline}
        live["customer"] = info
        boot.log(f"gateway in setup mode at {info['console_url']} (setup token published to Dict {DICT_NAME!r})")
        reason = "ttl"
        try:
            if e2e:
                result["e2e"] = _e2e(live, run_id, url, token, deadline)
                result["ok"] = (result["e2e"]["compare"]["logprobs_bit_identical"]
                                and result["e2e"]["compare"]["text_equal"])
                reason = "e2e finished"
            else:
                while time.time() < deadline:
                    if gw.poll() is not None:
                        reason = f"gateway exited ({gw.returncode})"
                        break
                    if live.get("stop") in (run_id, "all"):
                        reason = "stop requested"
                        break
                    time.sleep(10)
        except Exception as exc:
            result["ok"] = False
            result["error"] = f"{type(exc).__name__}: {exc}"
            result["traceback"] = traceback.format_exc()[-3000:]
            reason = "error"
        boot.log(f"shutting down: {reason}")
        gw.terminate()
        try:
            gw.wait(40)
        except Exception:
            gw.kill()
        log = (root / "gateway.log").read_text(errors="replace")
        result["log_tail"] = log.replace(token, "<setup-token>")[-6000:]
    result["reason"] = reason
    if e2e:
        live["e2e"] = {k: v for k, v in result.items() if k != "log_tail"}
    if (live.get("customer") or {}).get("run_id") == run_id:
        live["customer"] = {**info, "setup_token": "(expired)", "stopped_at": time.time(), "stop_reason": reason}
    return result
