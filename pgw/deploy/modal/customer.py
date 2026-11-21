"""Party 2 on Modal: the customer's gateway (``pgw-customer``; CPU only unless ``PGW_MODAL_CUSTOMER_GPU`` is set).

The image installs pgw (CPU torch; the CUDA build with a customer GPU) from this repository; it contains no
model. :func:`gateway` starts ``pgw gateway serve`` with an empty configuration, so it opens the first-run
setup wizard, exposes the console on a public HTTPS URL (``modal.forward``: TLS ends at Modal's edge) and publishes
``{console_url, setup_token}`` in the ``pgw-live`` Dict. During the wizard the gateway downloads the
model from Hugging Face itself, packs it and checks the fingerprint against the provider's, builds the
encrypted-mask hint and runs the encrypted mask factory, all inside this container.

With ``e2e=True`` the function also plays the administrator (``e2e_driver.py``): it drives the wizard
through the public URL with the provider's enrollment code (``wizard_client.py``), sends a greedy chat
completion, recomputes it with the plaintext reference backend on the same pack and stores the comparison
in the Dict. It also
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
        # with --customer-gpu the image must hold the CUDA build of torch (and Triton for the GPU mask PRNG);
        # without it the gateway would silently stay on the CPU engines
        *(["cd / && python -c 'import torch, triton; assert torch.version.cuda, \"CPU-only torch in a GPU image\"; "
           "print(\"torch\", torch.__version__, \"cuda\", torch.version.cuda, \"triton\", triton.__version__)'"]
          if CUSTOMER_GPU else []),
    )
    .env({"PYTHONUNBUFFERED": "1", "HF_HUB_DISABLE_PROGRESS_BARS": "1", "OMP_WAIT_POLICY": "PASSIVE"})
    .add_local_file(HERE / "pgw_modal_boot.py", "/opt/pgw-modal/pgw_modal_boot.py")
    .add_local_file(HERE / "wizard_client.py", "/opt/pgw-modal/wizard_client.py")
    .add_local_file(HERE / "trip_gate.py", "/opt/pgw-modal/trip_gate.py")
    .add_local_file(HERE / "e2e_driver.py", "/opt/pgw-modal/e2e_driver.py")
)

app = modal.App("pgw-customer", image=image)

_FN: dict = dict(gpu=CUSTOMER_GPU, cpu=4.0, memory=8192, timeout=MAX_LIFETIME_S, max_containers=1,
                 min_containers=0, retries=0, region=REGION)


@app.function(**_FN, i6pn=False)
def gateway(run_id: str, ttl_s: int = 80 * 60, e2e: bool = False, local_linears: str = "auto") -> dict:
    """The customer gateway; reaches the provider through its public TCP tunnel."""
    return _gateway(run_id, ttl_s, e2e, i6pn=False, local_linears=local_linears)


@app.function(**_FN, i6pn=True)
def gateway_i6pn(run_id: str, ttl_s: int = 80 * 60, e2e: bool = False, local_linears: str = "auto") -> dict:
    """The same gateway on Modal's private IPv6 network (pairs with the provider's ``serve_i6pn``)."""
    return _gateway(run_id, ttl_s, e2e, i6pn=True, local_linears=local_linears)


def _gateway(run_id: str, ttl_s: int, e2e: bool, *, i6pn: bool, local_linears: str = "auto") -> dict:
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
        # with a customer GPU the local qkv/o integer products may run on it ('auto': only when a start-up
        # check finds them identical to the CPU's and faster; the outputs are the same bits either way)
        opts = {**boot.GATEWAY_MODEL_OPTIONS, "local_linears": local_linears,
                **({"local_linears_device": "auto"} if CUSTOMER_GPU else {})}
        env = boot.gateway_env(root / "config", root / "data", port, opts)
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
                import e2e_driver

                result["e2e"] = e2e_driver.run(live, run_id, url, token, deadline, f"http://127.0.0.1:{port}",
                                               root / "config")
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
