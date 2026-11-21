"""Local rehearsal of the Modal deployment: both parties on this machine, same bootstrap code.

    python pgw/deploy/modal/local_rehearsal.py --work /tmp/pgw-rehearsal [--through admin]
    python pgw/deploy/modal/local_rehearsal.py --work /tmp/r --device auto --users --json out.json  # = colocated.py
    python pgw/deploy/modal/local_rehearsal.py --work /tmp/r --delay-ms 17 --json out.json         # emulated tunnel

The provider runs the compute service with TLS on loopback (``--device``: ``cpu`` by default, ``auto`` uses
a GPU when there is one); the gateway runs in setup mode and the wizard is driven over HTTP. This is the
co-located placement (``colocated.py``) outside Modal, through the same ``pgw_modal_boot`` functions:

* ``--users`` (root only) runs each party as its own Unix user (``pgwprov`` / ``pgwcust``, created if
  missing) with its own 0700 directory, as ``colocated.py`` does;
* ``--delay-ms D`` puts a loopback proxy that adds ``D`` ms per round trip between the gateway and the
  service (``benchmarks/delay_proxy.py``; the enrollment code points at the proxy), to emulate a slower
  link (17 ms: the Modal tunnel measured on 2026-11-21);
* ``--json PATH`` writes the end-to-end result (``e2e_driver.run``: exactness against the plaintext
  reference, client-reported trip latency, speed settings, tokens per second of the warm chat).

``--through admin`` stops after the admin step (the encrypted-mask hint of Qwen2.5-0.5B needs about
7.5 GB of disk).
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
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
    ap.add_argument("--device", default="cpu", help="compute service device: cpu (default), auto, cuda")
    ap.add_argument("--users", action="store_true", help="run each party as its own Unix user (root only)")
    ap.add_argument("--delay-ms", type=float, default=None, help="add this many ms per round trip (loopback proxy)")
    ap.add_argument("--json", help="write the end-to-end result here")
    ap.add_argument("--local-linears", choices=["off", "attention", "auto"], default="auto")
    args = ap.parse_args()
    work = Path(args.work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    work.chmod(0o711)  # traversable by the party users, not listable
    prov, cust, logs = work / "provider", work / "customer", work / "logs"
    pack = work / "pack" / boot.MODEL_NAME  # outside both parties' directories; read-only for both
    if not (pack / "manifest.json").exists():
        import subprocess

        subprocess.run([sys.executable, "-m", "pgw.cli", "pack", boot.MODEL_REPO, str(pack), "--name",  # noqa: S603
                        boot.MODEL_NAME, "-q"], check=True)
    for d in (work / "pack", *(p for p in pack.rglob("*") if p.is_dir()), pack):
        d.chmod(0o755)
    for f in pack.rglob("*"):
        if f.is_file():
            f.chmod(0o644)
    puser = cuser = None
    if args.users:
        puser, cuser = boot.PROVIDER_USER, boot.CUSTOMER_USER
        boot.ensure_user(puser)
        boot.ensure_user(cuser)
    port = free_port()
    proxy = None
    public_url = None
    if args.delay_ms is not None:
        sys.path.insert(0, str(HERE.parents[1] / "benchmarks"))
        from delay_proxy import DelayProxy

        proxy = DelayProxy(("127.0.0.1", port), one_way_s=args.delay_ms / 2e3)
        public_url = f"https://127.0.0.1:{proxy.start()}"
    procs = []
    result: dict = {"device": args.device, "users": args.users, "delay_ms": args.delay_ms}
    try:
        p = boot.start_provider_party(prov, pack, port, device=args.device, user=puser, log_dir=logs,
                                      code_ttl="2h", note="local rehearsal", public_url=public_url)
        procs.append(p["proc"])
        code = p["code"]
        boot.log(f"enrollment code created (id {code['id']}) for {p['url']}")
        gport = free_port()
        gw, token = boot.start_gateway_party(cust, gport, user=cuser, log_dir=logs,
                                             model_options={**boot.GATEWAY_MODEL_OPTIONS,
                                                            "local_linears": args.local_linears})
        procs.append(gw)
        base = f"http://127.0.0.1:{gport}"
        if args.through == "admin":
            import httpx

            with httpx.Client(base_url=base, timeout=600, trust_env=False) as c:
                h = {wiz.CSRF_HEADER: c.post("/console/api/setup/login", json={"token": token}).json()["csrf"]}
                print(wiz._check(c.post("/console/api/setup/enroll", json={"code": code["code"]}, headers=h), "e"))
                wiz._check(c.post("/console/api/setup/model", json={"name": boot.MODEL_NAME}, headers=h), "m")
                while True:
                    st = c.get("/console/api/setup/state").json()
                    if st["model"]["pack_status"] != "building":
                        print(json.dumps(st["model"]), json.dumps(st["jobs"]["pack"]))
                        break
                    time.sleep(3)
                print(wiz._check(c.post("/console/api/setup/maskstore", json={}, headers=h), "ms"))
                print(sorted(wiz._check(c.post("/console/api/setup/admin", json={}, headers=h), "a")))
            return
        import e2e_driver

        live = {"provider": {"run_id": "local", "code": code["code"], "service_url": p["url"]}}
        t0 = time.time()
        result["e2e"] = e2e_driver.run(live, "local", base, token, time.time() + 3600, base, cust / "config")
        result["e2e_s"] = round(time.time() - t0, 1)
        cmp = result["e2e"]["compare"]
        result["ok"] = bool(cmp["logprobs_bit_identical"] and cmp["text_equal"])
        e = result["e2e"]
        print(json.dumps({"ok": result["ok"], "path": e["network_path"], "speed": e["speed"],
                          "trip_latency_warm": e["trip_latency"]["warm"], "chat2_wall_s": e["chat2_wall_s"],
                          "usage": e["usage"]}, indent=1, default=str))
    except Exception as exc:
        result["ok"] = False
        result["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        for proc in reversed(procs):
            proc.terminate()
            try:
                proc.wait(30)
            except Exception:
                proc.kill()
        if proxy is not None:
            proxy.stop()
        if args.json:
            slim = {k: v for k, v in result.items() if k != "e2e"}
            if "e2e" in result:
                slim["e2e"] = {k: v for k, v in result["e2e"].items() if k not in ("admin_status", "admin_masks")}
            Path(args.json).write_text(json.dumps(slim, indent=1, default=str))


if __name__ == "__main__":
    main()
