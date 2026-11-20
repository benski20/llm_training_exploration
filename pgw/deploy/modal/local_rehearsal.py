"""Local rehearsal of the Modal deployment: both parties on this machine, same bootstrap code.

    python pgw/deploy/modal/local_rehearsal.py --work /tmp/pgw-rehearsal [--through verify]

The provider runs the compute service on CPU with TLS on loopback; the gateway runs in setup mode and
the wizard is driven over HTTP. ``--through`` stops early (the encrypted-mask hint of Qwen2.5-0.5B
needs about 7.5 GB of disk).
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import pgw_modal_boot as boot  # noqa: E402
import wizard_client as wiz  # noqa: E402


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = int(s.getsockname()[1])
    s.close()
    return port


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--through", choices=["admin", "all"], default="all")
    args = ap.parse_args()
    work = Path(args.work)
    prov, cust = work / "provider", work / "customer"
    pack = prov / "packs" / boot.MODEL_NAME
    if not (pack / "manifest.json").exists():
        import subprocess

        subprocess.run([sys.executable, "-m", "pgw.cli", "pack", boot.MODEL_REPO, str(pack), "--name",  # noqa: S603
                        boot.MODEL_NAME, "-q"], check=True)
    port = free_port()
    pki = boot.make_pki(prov / "pki", ["localhost", "127.0.0.1"])
    url = f"https://localhost:{port}"
    env = boot.service_env(prov, pack, pki, url, port, device="cpu")
    procs = []
    try:
        svc = boot.start_service(env, prov / "service.log")
        procs.append(svc)
        boot.wait_ready(f"https://127.0.0.1:{port}/readyz", svc, verify=str(pki["service_ca"]))
        code = boot.create_enrollment_code(env, "customer-demo", "2h", "local rehearsal")
        boot.log(f"enrollment code created (id {code['id']})")
        gport = free_port()
        genv = boot.gateway_env(cust / "config", cust / "data", gport)
        gw, token = boot.start_gateway(genv, cust / "gateway.log")
        procs.append(gw)
        base = f"http://127.0.0.1:{gport}"
        boot.wait_ready(base + "/healthz", gw)
        if args.through == "admin":
            import httpx

            with httpx.Client(base_url=base, timeout=600, trust_env=False) as c:
                h = {wiz.CSRF_HEADER: c.post("/console/api/setup/login", json={"token": token}).json()["csrf"]}
                print(wiz._check(c.post("/console/api/setup/enroll", json={"code": code["code"]}, headers=h), "e"))
                wiz._check(c.post("/console/api/setup/model", json={"name": boot.MODEL_NAME}, headers=h), "m")
                import time

                while True:
                    st = c.get("/console/api/setup/state").json()
                    if st["model"]["pack_status"] != "building":
                        print(json.dumps(st["model"]), json.dumps(st["jobs"]["pack"]))
                        break
                    time.sleep(3)
                print(wiz._check(c.post("/console/api/setup/maskstore", json={}, headers=h), "ms"))
                print(sorted(wiz._check(c.post("/console/api/setup/admin", json={}, headers=h), "a")))
            return
        res = wiz.run_wizard(base, token, code["code"], boot.MODEL_NAME)
        print(json.dumps(res["timings"]))
        wiz.wait_gateway_ready(base)
        key = wiz.create_api_key(base, res["admin_token"])
        msgs = [{"role": "user", "content": "In one sentence, what is the capital of France?"}]
        out = wiz.chat(base, key, boot.MODEL_NAME, msgs, 32)
        print(out["choices"][0]["message"]["content"])
        st = json.loads((cust / "config" / "setup" / "state.json").read_text())
        ref = wiz.plaintext_reference(st["model"]["pack_path"], st["model"]["ring"] or boot.RING, msgs, 32)
        print(json.dumps(wiz.compare(out, ref)))
    finally:
        for p in reversed(procs):
            p.terminate()
            try:
                p.wait(30)
            except Exception:
                p.kill()


if __name__ == "__main__":
    main()
