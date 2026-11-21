"""Party 2 on Modal: the customer's gateway (``pgw-customer``, CPU only).

The image installs pgw (CPU torch) from this repository; it contains no model. :func:`gateway` starts
``pgw gateway serve`` with an empty configuration, so it opens the first-run setup wizard, exposes the
console on a public HTTPS URL (``modal.forward``: TLS ends at Modal's edge) and publishes
``{console_url, setup_token}`` in the ``pgw-live`` Dict. During the wizard the gateway downloads the
model from Hugging Face itself, packs it and checks the fingerprint against the provider's, builds the
encrypted-mask hint and runs the encrypted mask factory, all inside this container.

With ``e2e=True`` the function also plays the administrator: it drives the wizard through the public
URL with the provider's enrollment code (``wizard_client.py``), sends a greedy chat completion, recomputes
it with the plaintext reference backend on the same pack and stores the comparison in the Dict. It also
reports the gateway's client-reported round-trip latency over the chats (``trip_latency``, read from the
gateway's own ``/metrics`` and ``/admin/status`` on loopback; see ``trip_gate.py``), which ``launch.py
--max-trip-ms`` gates on.

Two functions run the same code: ``gateway`` reaches the provider through its public TCP tunnel and
``gateway_i6pn`` joins Modal's private IPv6 network (i6pn) so that it can reach a provider started with
``serve_i6pn`` directly. ``launch.py`` picks the pair (tunnel by default). The function's first action
is to write ``customer_boot`` to the Dict, so the launcher can tell a container that Modal has not
scheduled yet from one that is still starting up.

Config, secrets, packs and the mask store live on the container's local disk, not on a ``modal.Volume``:
the mask store must never sit on network or snapshotted storage (a restored copy could hand out masks
twice), and this demo is single-use anyway. Use ``launch.py`` to start it.
"""

from __future__ import annotations

import json
import os
import time
import traceback
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
# Inside the container this file is /root/<name>.py and the source tree is not there (nor needed).
PGW = HERE.parents[1] if modal.is_local() else Path("/opt/pgw-src")
MAX_LIFETIME_S = 88 * 60  # hard cap (function timeout): nothing outlives 90 minutes
DICT_NAME = "pgw-live"
# Both parties in one region: every token makes ~100 sequential round trips between them, so a
# cross-region placement (about 85 ms RTT seen) makes decoding 10x slower. PGW_MODAL_REGION="" lifts it.
REGION = os.environ.get("PGW_MODAL_REGION", "us-east") or None
# Optional GPU for the customer gateway (PGW_MODAL_CUSTOMER_GPU=T4 at deploy time). It runs the encrypted mask
# factory's encrypt/decrypt GEMMs and random-number generation on the customer's own GPU (customer side: the
# provider still sees only ciphertexts). Default none (CPU torch image, CPU products).
CUSTOMER_GPU = os.environ.get("PGW_MODAL_CUSTOMER_GPU") or None
# Strict speculation with a draft model (launch.py --spec-draft; PGW_MODAL_DRAFT=1 at deploy time): the image then
# holds a pack of Qwen2.5-0.5B-Instruct (public weights, same tokenizer as the Qwen2.5 targets) that the gateway
# runs in plaintext to propose tokens; it never leaves this container. Default: no draft (prompt lookup only).
DRAFT = os.environ.get("PGW_MODAL_DRAFT", "0") == "1"
DRAFT_REPO, DRAFT_REVISION = "Qwen/Qwen2.5-0.5B-Instruct", "7ae557604adf67be50417f59c2c2f167def9a775"
DRAFT_PACK = "/opt/packs/draft"  # = pgw_modal_boot.DRAFT_PACK

image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install("torch==2.15.0", **({} if CUSTOMER_GPU else {"index_url": "https://download.pytorch.org/whl/cpu"}))
    .add_local_dir(PGW / "src", "/opt/pgw-src/src", copy=True, ignore=["**/__pycache__", "**/*.so"])
    .add_local_file(PGW / "pyproject.toml", "/opt/pgw-src/pyproject.toml", copy=True)
    .add_local_file(PGW / "README.md", "/opt/pgw-src/README.md", copy=True)
    .add_local_file(PGW / "deploy/hf-space/constraints.txt", "/opt/pgw-src/constraints.txt", copy=True)
    .run_commands("pip install --no-cache-dir -c /opt/pgw-src/constraints.txt '/opt/pgw-src[gateway,factory,pack]'")
    # native gateway kernels (pgw_fast); without them pgw runs its much slower pure-Python path.
    # pgw_ntt: exact NTT products for the encrypted mask factory (pgw.hemask.ntt); without it the
    # HE customer silently stays on the dense products.
    .apt_install("curl", "build-essential")
    .add_local_dir(PGW / "rust", "/opt/pgw-src/rust", copy=True, ignore=["target/**", "**/target/**"])
    .run_commands(
        "curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable",
        "cd /opt/pgw-src && . $HOME/.cargo/env && python rust/tools/build_fast.py --site && "
        "python rust/tools/build_ntt.py --site && rm -rf rust/target $HOME/.cargo/registry",
        "cd / && python -c 'from pgw.core import native; assert native.fast is not None; print(native.fast.version())'",
        "cd / && python -c 'from pgw.hemask import ntt; assert ntt.available(); print(\"pgw_ntt\", ntt.info())'",
    )
    .env({"PYTHONUNBUFFERED": "1", "HF_HUB_DISABLE_PROGRESS_BARS": "1", "OMP_WAIT_POLICY": "PASSIVE"})
)
if DRAFT:
    image = image.run_commands(
        f"python -m pgw.cli pack {DRAFT_REPO} {DRAFT_PACK} --name draft --revision {DRAFT_REVISION} -q"
        " && rm -rf /root/.cache/huggingface")
image = (
    image
    .add_local_file(HERE / "pgw_modal_boot.py", "/opt/pgw-modal/pgw_modal_boot.py")
    .add_local_file(HERE / "wizard_client.py", "/opt/pgw-modal/wizard_client.py")
    .add_local_file(HERE / "trip_gate.py", "/opt/pgw-modal/trip_gate.py")
)

app = modal.App("pgw-customer", image=image)

PROMPT = [{"role": "user", "content": "Explain in two sentences why one-time pads are secure."}]


def _e2e(live: modal.Dict, run_id: str, url: str, token: str, deadline: float, local_url: str,
         spec_rows: int = 0) -> dict:
    import pgw_modal_boot as boot
    import trip_gate
    import wizard_client as wiz

    while True:
        prov = live.get("provider") or {}
        if prov.get("run_id") == run_id and prov.get("code", "").startswith("PGW1-"):
            break
        if time.time() > deadline:
            raise RuntimeError("provider did not publish an enrollment code")
        time.sleep(5)
    model_name = prov.get("model") or boot.MODEL_NAME  # the target the provider serves (launch.py --model)
    res = wiz.run_wizard(url, token, prov["code"], model_name, timeout_s=max(60, deadline - time.time()))
    wiz.wait_gateway_ready(url)
    key = wiz.create_api_key(url, res["admin_token"])
    adm = res["admin_token"]

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
    out2 = wiz.chat(url, key, model_name, PROMPT, 32)  # second run: determinism + warm timing
    s2 = snap()
    # The warm chat (the second one) decides launch.py's --max-trip-ms gate; both chats are reported too.
    trips = {"warm": trip_gate.trip_latency(s1[0], s2[0], s1[1], s2[1]),
             "both_chats": trip_gate.trip_latency(s0[0], s2[0], s0[1], s2[1])}
    if snap_errors:
        trips["errors"] = snap_errors
    st = json.loads(Path("/srv/pgw/config/setup/state.json").read_text())
    t = time.time()
    ref = wiz.plaintext_reference(st["model"]["pack_path"], st["model"].get("ring") or boot.RING, PROMPT, 32)
    ref_s = time.time() - t
    cmp = wiz.compare(out, ref)
    status = wiz.admin_get(url, res["admin_token"], "/admin/status")
    masks = wiz.admin_get(url, res["admin_token"], "/admin/masks")
    model_status = next((m for m in status.get("models", []) if m.get("name") == model_name), {})
    speed = dict(model_status.get("speed") or {})
    warm_tokens = int(((out2.get("usage") or {}).get("completion_tokens")) or 0)
    if warm_tokens and trips["warm"].get("trips"):
        speed["warm_trips_per_token"] = round(trips["warm"]["trips"] / warm_tokens, 2)
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
        "reference_text": ref["text"], "reference_s": ref_s, "compare": cmp,
        "admin_status": status, "admin_masks": masks, "trip_latency": trips, "speed": speed,
    }


_FN: dict = dict(gpu=CUSTOMER_GPU, cpu=4.0, memory=8192, timeout=MAX_LIFETIME_S, max_containers=1,
                 min_containers=0, retries=0, region=REGION)


@app.function(**_FN, i6pn=False)
def gateway(run_id: str, ttl_s: int = 80 * 60, e2e: bool = False, local_linears: str = "auto",
            spec_rows: int = 0) -> dict:
    """The customer gateway; reaches the provider through its public TCP tunnel."""
    return _gateway(run_id, ttl_s, e2e, i6pn=False, local_linears=local_linears, spec_rows=spec_rows)


@app.function(**_FN, i6pn=True)
def gateway_i6pn(run_id: str, ttl_s: int = 80 * 60, e2e: bool = False, local_linears: str = "auto",
                 spec_rows: int = 0) -> dict:
    """The same gateway on Modal's private IPv6 network (pairs with the provider's ``serve_i6pn``)."""
    return _gateway(run_id, ttl_s, e2e, i6pn=True, local_linears=local_linears, spec_rows=spec_rows)


def _gateway(run_id: str, ttl_s: int, e2e: bool, *, i6pn: bool, local_linears: str = "auto",
             spec_rows: int = 0) -> dict:
    import sys

    t_boot = time.time()
    live = modal.Dict.from_name(DICT_NAME, create_if_missing=True)
    # First action once Modal has scheduled this container (launch.py's i6pn scheduling fallback watches it).
    live["customer_boot"] = {"run_id": run_id, "at": t_boot, "i6pn": i6pn}
    sys.path.insert(0, "/opt/pgw-modal")
    import pgw_modal_boot as boot

    deadline = t_boot + min(ttl_s, MAX_LIFETIME_S - 120)
    root = Path("/srv/pgw")
    port = 8080
    result: dict = {"run_id": run_id}
    with modal.forward(port) as tunnel:
        url = tunnel.url
        # speed settings for the wizard's models.json (boot.GATEWAY_MODEL_OPTIONS; launch.py --local-linears)
        # launch.py --spec-rows: strict speculation (prompt lookup; plus the image's draft with --spec-draft)
        spec = boot.spec_options(spec_rows, DRAFT_PACK if DRAFT else None, "cuda" if CUSTOMER_GPU else None)
        env = boot.gateway_env(root / "config", root / "data", port,
                               {**boot.GATEWAY_MODEL_OPTIONS, "local_linears": local_linears, **spec})
        gw, token = boot.start_gateway(env, root / "gateway.log")
        boot.wait_ready(f"http://127.0.0.1:{port}/healthz", gw)
        info = {"run_id": run_id, "console_url": url.rstrip("/") + "/console/", "base_url": url,
                "setup_token": token, "started_at": time.time(), "expires_at": deadline,
                "network": "i6pn" if i6pn else "tunnel"}
        live["customer"] = info
        boot.log(f"gateway in setup mode at {info['console_url']} (setup token published to Dict {DICT_NAME!r})")
        reason = "ttl"
        try:
            if e2e:
                result["e2e"] = _e2e(live, run_id, url, token, deadline, f"http://127.0.0.1:{port}", spec_rows)
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
