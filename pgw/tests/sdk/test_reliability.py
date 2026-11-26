"""Failure handling of the client library end to end (``docs/sdk/reliability.md``): a real ``pgw service`` on
loopback behind fault-injection proxies (:mod:`tests.sdk.chaos`), killed, restarted and cut off mid-stream; keys
revoked mid-stream; the client closed with requests in flight; streams abandoned; the mask factory killed; client
certificates expiring (the provider issues 45-second certificates here); cache files damaged.

Every answer that succeeds is bit-identical to the first one; every failure is a typed, documented error.
"""

from __future__ import annotations

import json
import os
import signal
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from tests.sdk.chaos import ChaosProxy
from tests.sdk.conftest import MODEL, Provider, free_port, restart_provider, start_provider, stop_provider

pytestmark = pytest.mark.timeout(900)

PROMPT = [{"role": "system", "content": "You are terse."}, {"role": "user", "content": "Hello there, who are you?"}]
GREEDY = {"temperature": 0, "max_tokens": 24, "logprobs": True, "extra_body": {"ignore_eos": True}}
CERT_S = 45.0


class Chaos:
    def __init__(self, prov: Provider, http: ChaosProxy, lane: ChaosProxy):
        self.prov, self.http, self.lane = prov, http, lane

    def cut(self) -> None:
        self.http.drop()
        self.lane.drop()

    def mode(self, m: str) -> None:
        self.http.mode = self.lane.mode = m  # type: ignore[assignment]


@pytest.fixture(scope="module")
def chaos(sdk_pack_dir: Path, pki: dict[str, Any], tmp_path_factory: pytest.TempPathFactory) -> Iterator[Chaos]:
    port, lane = free_port(), free_port()
    http, lp = ChaosProxy(port), ChaosProxy(lane)
    prov = start_provider(tmp_path_factory.mktemp("rel-provider"), sdk_pack_dir, pki, port=port, lane_port=lane,
                          public_port=http.port, public_lane_port=lp.port,
                          extra_env={"PGW_SERVICE_CLIENT_CERT_VALIDITY_DAYS": str(CERT_S / 86400)})
    c = Chaos(prov, http, lp)
    try:
        yield c
    finally:
        c.mode("pass")
        stop_provider(prov)
        http.close()
        lp.close()


def _client(chaos: Chaos, key: str, root: Path, **kw: Any) -> Any:
    from pgw import OpenAI
    from pgw.sdk import LocalPackProvisioner

    opts: dict[str, Any] = {"cache_dir": root / "cache", "state_dir": root / "state", "mask_scheme": "plain",
                            "mask_workers": 1, "mask_target_slots": 1200, "mask_low_water_slots": 600,
                            "mask_segment_slots": 256, "timeout": 120, "ready_timeout_s": 300}
    opts.update(kw)
    prov = opts.pop("provisioner", LocalPackProvisioner({MODEL: chaos.prov.pack_dir}))
    return OpenAI(api_key=key, provisioner=prov, **opts)


@pytest.fixture(scope="module")
def shared(chaos: Chaos, tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, Any]]:
    root = tmp_path_factory.mktemp("rel-client")
    key = chaos.prov.new_key("rel")
    c = _client(chaos, key, root)
    want = c.chat.completions.create(model=MODEL, messages=PROMPT, **GREEDY).choices[0].message.content
    st = {"client": c, "key": key, "root": root, "want": want}
    try:
        yield st
    finally:
        st["client"].close()


def _text(client: Any, **kw: Any) -> str:
    return str(client.chat.completions.create(model=MODEL, messages=PROMPT, **{**GREEDY, **kw}).choices[0]
               .message.content)


def _stream_with_fault(client: Any, fault: Any, at: int = 4, **kw: Any) -> tuple[str, BaseException | None]:
    got: list[str] = []
    try:
        for i, ch in enumerate(client.chat.completions.create(model=MODEL, messages=PROMPT, stream=True,
                                                              **{**GREEDY, **kw})):
            if ch.choices and ch.choices[0].delta.content:
                got.append(ch.choices[0].delta.content)
            if i == at:
                fault()
    except Exception as exc:
        return "".join(got), exc
    return "".join(got), None


def _active(client: Any) -> int:
    rt = client._engine.models[MODEL].runtime
    return int(rt.admission.active)


def _wait(pred: Any, timeout_s: float, poll_s: float = 0.1) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(poll_s)
    return bool(pred())


# ---- network and provider ------------------------------------------------------------------------------------
def test_connections_cut_mid_stream(chaos: Chaos, shared: dict[str, Any]) -> None:
    """Every connection reset mid-answer: the trip is sent again (the same masked bytes; nothing new is revealed)
    on a new connection and the answer completes bit-identical, or the request fails with a retryable error; the
    next request is served either way."""
    from pgw.sdk import PGWAPIError

    c = shared["client"]
    text, exc = _stream_with_fault(c, chaos.cut)
    if exc is None:
        assert text == shared["want"]
    else:
        assert isinstance(exc, PGWAPIError) and exc.retryable, exc
    assert _text(c) == shared["want"]


def test_provider_crash_mid_stream_then_restart(chaos: Chaos, shared: dict[str, Any]) -> None:
    """The provider is SIGKILLed mid-answer: the stream fails with a retryable error at once (never hangs, never a
    partial answer passed off as complete); after the provider restarts the same client serves bit-identical
    answers again (new lane ticket, new connections)."""
    from pgw.sdk import PGWAPIError

    c = shared["client"]
    t0 = time.monotonic()
    text, exc = _stream_with_fault(c, chaos.prov.proc.kill)
    assert isinstance(exc, PGWAPIError) and exc.retryable, exc
    assert time.monotonic() - t0 < 60 and text != shared["want"]
    restart_provider(chaos.prov)
    assert _text(c) == shared["want"]
    assert c.status()["models"][MODEL]["lane"] == "open"


def test_a_hung_network_path_is_bounded_by_the_request_timeout(chaos: Chaos, shared: dict[str, Any]) -> None:
    from pgw.sdk import APITimeoutError

    c = shared["client"]
    chaos.mode("blackhole")
    try:
        t0 = time.monotonic()
        with pytest.raises(APITimeoutError):
            _text(c, timeout=5)
        assert time.monotonic() - t0 < 20
    finally:
        chaos.mode("pass")
        chaos.cut()
    assert _text(c) == shared["want"]


def test_close_with_a_request_stuck_in_flight(chaos: Chaos, tmp_path: Path) -> None:
    """``close()`` while another thread's request (no timeout) waits on a dead network path: the caller gets
    APIConnectionError promptly instead of waiting forever, and close returns."""
    from pgw.sdk import APIConnectionError

    c = _client(chaos, chaos.prov.new_key("close"), tmp_path)
    _text(c)
    out: dict[str, Any] = {}

    def caller() -> None:
        try:
            _text(c)
            out["r"] = "ok"
        except Exception as exc:
            out["r"] = exc

    chaos.mode("blackhole")
    try:
        th = threading.Thread(target=caller)
        th.start()
        time.sleep(2.0)
        t0 = time.monotonic()
        c.close()
        assert time.monotonic() - t0 < 45
        th.join(30)
        assert not th.is_alive(), "the caller still waits after close()"
        assert isinstance(out["r"], APIConnectionError), out
    finally:
        chaos.mode("pass")
        chaos.cut()


def test_dropped_and_cancelled_streams_stop_generating(chaos: Chaos, shared: dict[str, Any]) -> None:
    """A stream the application drops, and an async stream whose consumer task is cancelled (an SSE client that
    disconnected), stop at once: no masks or provider work are spent on the rest of the answer."""
    import asyncio
    import contextlib
    import gc

    from pgw import AsyncOpenAI

    c = shared["client"]
    s = c.chat.completions.create(model=MODEL, messages=PROMPT, stream=True, max_tokens=200, temperature=0)
    next(iter(s))
    del s
    gc.collect()
    assert _wait(lambda: _active(c) == 0, 10), "a dropped stream kept generating"

    async def run() -> None:
        async with AsyncOpenAI(api_key=shared["key"], cache_dir=shared["root"] / "cache",
                               state_dir=shared["root"] / "state") as ac:
            st = await ac.chat.completions.create(model=MODEL, messages=PROMPT, stream=True, max_tokens=200,
                                                  temperature=0)

            async def consume() -> None:
                async for _ in st:
                    pass

            t = asyncio.create_task(consume())
            await asyncio.sleep(0.3)
            t.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await t
            assert await asyncio.to_thread(_wait, lambda: _active(c) == 0, 10), "a cancelled stream kept generating"

    asyncio.run(run())
    assert _text(c) == shared["want"]


# ---- credentials -----------------------------------------------------------------------------------------------
def test_key_revoked_mid_stream(chaos: Chaos, tmp_path: Path) -> None:
    """Revoking the key closes its lane connections: the stream fails with AuthenticationError (not a retryable
    503), and so does the next request."""
    from pgw.sdk import AuthenticationError

    key = chaos.prov.new_key("revoke-mid")
    c = _client(chaos, key, tmp_path)
    try:
        _text(c)
        _, exc = _stream_with_fault(c, lambda: chaos.prov.revoke(key), at=2, max_tokens=200)
        assert isinstance(exc, AuthenticationError) and not exc.retryable, exc
        with pytest.raises(AuthenticationError):
            _text(c)
    finally:
        c.close()


def test_provider_lost_its_tokens_then_the_client_enrolls_again(chaos: Chaos, tmp_path: Path) -> None:
    """The provider restarts without its tokens (a lost volume): the installation's service token is refused. The
    request that meets the refusal fails (AuthenticationError, or PermissionDeniedError for the certificate whose
    mapping went too); the next one reconnects, enrolls again with the (still valid) key and is served
    bit-identical."""
    from pgw.sdk import AuthenticationError, PermissionDeniedError

    key = chaos.prov.new_key("lost-tokens")
    c = _client(chaos, key, tmp_path)
    try:
        want = _text(c)
        secrets = chaos.prov.root / "secrets"
        chaos.prov.proc.kill()
        chaos.prov.proc.wait(30)
        saved = {n: (secrets / n).read_text() for n in ("tokens", "mtls_tenants")}
        for n, txt in saved.items():  # keep everyone else's lines: only this key's installation loses its token
            (secrets / n).write_text("".join(ln + "\n" for ln in txt.splitlines()
                                             if not ln.startswith(Provider.tenant(key))))
        restart_provider(chaos.prov)
        ok = 0
        for _ in range(3):
            try:
                assert _text(c) == want
                ok += 1
            except (AuthenticationError, PermissionDeniedError):
                pass
        assert ok >= 2
        meta = json.loads(next((tmp_path / "cache" / "sessions").iterdir()).joinpath("enrollment.json").read_text())
        assert meta["enrolled_at"] > time.time() - 120  # enrolled again just now
    finally:
        c.close()


def test_the_client_certificate_is_renewed_while_running(chaos: Chaos, shared: dict[str, Any]) -> None:
    """45-second certificates: renewed at 2/3 of their lifetime and loaded into the live TLS contexts; after the
    first certificate's expiry, reconnecting (every connection cut) still works. Before: once the certificate had
    expired every new handshake failed, permanently."""
    from cryptography import x509

    c = shared["client"]
    sess = next((shared["root"] / "cache" / "sessions").iterdir())

    def serial() -> int:
        return int(x509.load_pem_x509_certificate((sess / "client.crt").read_bytes()).serial_number)

    first = serial()
    assert _wait(lambda: serial() != first, CERT_S + 15, 0.5), "the certificate was not renewed"
    time.sleep(max(0.0, CERT_S * 0.5))  # the first certificate has expired by now
    chaos.cut()
    assert _text(c) == shared["want"]


def test_an_expired_cached_certificate_enrolls_again(chaos: Chaos, tmp_path: Path) -> None:
    """A cache left idle past its certificate's lifetime: the next start enrolls again instead of failing every
    handshake."""
    key = chaos.prov.new_key("idle")
    c = _client(chaos, key, tmp_path)
    want = _text(c)
    c.close()
    meta_p = next((tmp_path / "cache" / "sessions").iterdir()) / "enrollment.json"
    before = json.loads(meta_p.read_text())["enrolled_at"]
    time.sleep(CERT_S + 2)
    c = _client(chaos, key, tmp_path)
    try:
        assert _text(c) == want
        assert json.loads(meta_p.read_text())["enrolled_at"] > before
    finally:
        c.close()


# ---- masks and the factory ---------------------------------------------------------------------------------------
def test_a_request_bigger_than_the_stock_refills_now(chaos: Chaos, tmp_path: Path) -> None:
    """The factory refills only below its low-water mark: a request that needs more masks than are in stock (but no
    more than the target) used to wait out mask_wait_s and fail; now it asks the factory for them."""
    c = _client(chaos, chaos.prov.new_key("starve"), tmp_path, mask_target_slots=400, mask_low_water_slots=100,
                mask_segment_slots=64, mask_wait_s=120)
    try:
        _text(c)

        def stock() -> int:
            return int(c.status()["models"][MODEL]["masks_available"])

        assert _wait(lambda: stock() > 260, 60)  # the first fill (up to the target) has landed
        while stock() > 230:  # down to above the low-water mark, below what the big request needs
            _text(c, max_tokens=24)
        time.sleep(3)
        assert 100 < stock() <= 230, stock()
        t0 = time.monotonic()
        _text(c, max_tokens=200)  # needs ~240 slots
        assert time.monotonic() - t0 < 60
    finally:
        c.close()


def test_a_request_bigger_than_the_whole_stock_fails_fast(chaos: Chaos, tmp_path: Path) -> None:
    """More than the stock ever holds (its target): waiting cannot help, the request fails at once (400,
    payload_too_large) with what to change, instead of waiting mask_wait_s for a NotReadyError."""
    from pgw.sdk import BadRequestError

    c = _client(chaos, chaos.prov.new_key("too-big"), tmp_path, mask_target_slots=200, mask_low_water_slots=100,
                mask_segment_slots=64, mask_wait_s=120)
    try:
        _text(c)
        t0 = time.monotonic()
        with pytest.raises(BadRequestError) as ei:
            _text(c, max_tokens=230)
        assert ei.value.code == "payload_too_large" and "mask_store_gb" in ei.value.message
        assert time.monotonic() - t0 < 10
        assert _text(c) == _text(c)  # smaller requests are still served
    finally:
        c.close()


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        with open(f"/proc/{pid}/stat") as fh:
            return fh.read().split()[2] != "Z"
    except (OSError, IndexError):
        return False


def _children(pid: int) -> list[int]:
    try:
        return [int(p) for p in Path(f"/proc/{pid}/task/{pid}/children").read_text().split()]
    except OSError:
        return []


def test_an_oom_killed_factory_leaves_no_orphans_and_restarts(chaos: Chaos, tmp_path: Path) -> None:
    """SIGKILL of the factory process (what the out-of-memory killer does): its fleet workers exit with it, the
    supervisor restarts it and reports the reason. Before: the workers kept running as orphans."""
    c = _client(chaos, chaos.prov.new_key("oom"), tmp_path, mask_workers=2)
    try:
        _text(c)
        fac = c._engine.models[MODEL].factory
        assert _wait(lambda: len(_children(fac._proc.pid)) >= 2, 60)
        child = fac._proc.pid
        workers = [p for p in _children(child) if "spawn_main" in Path(f"/proc/{p}/cmdline").read_text(errors="replace")]
        assert workers
        os.kill(child, signal.SIGKILL)
        assert _wait(lambda: not any(_alive(w) for w in workers), 30), "fleet workers outlived the factory"
        assert _wait(lambda: fac.starts >= 2 and fac.running, 60)
        assert fac.last_error is not None and fac.last_error.startswith("killed by SIGKILL")
        _text(c)
    finally:
        c.close()


# ---- damaged cache ---------------------------------------------------------------------------------------------
def test_a_damaged_cache_repairs_itself(chaos: Chaos, tmp_path: Path) -> None:
    """With the default provisioner (artifacts from the provider into the cache): a damaged cached gateway pack is
    downloaded again (checked against the pinned fingerprint, as the first time), damaged session files enroll
    again; answers stay bit-identical. A damaged mask store is never repaired automatically (its rollback
    counter is what stops reuse of masks): the error says what to delete."""
    from pgw.modelpack import ModelPack
    from pgw.sdk import PGWAPIError

    with ModelPack.open(chaos.prov.pack_dir) as p:
        fp = p.fingerprint
    key = chaos.prov.new_key("damage")
    from pgw.sdk.options import SDKOptions
    from pgw.sdk.provisioning import default_provisioner

    def mk() -> Any:
        return _client(chaos, key, tmp_path, model_fingerprints={MODEL: fp},
                       provisioner=default_provisioner(SDKOptions(model_fingerprints={MODEL: fp})))

    c = mk()
    want = _text(c)
    c.close()
    cache = tmp_path / "cache"
    plain = next(cache.glob("models/*/pack/plain.safetensors"))
    b = bytearray(plain.read_bytes())
    b[len(b) // 2] ^= 0x40
    plain.write_bytes(bytes(b))
    sess = next((cache / "sessions").iterdir())
    tok = sess / "secrets" / "service_token"
    tok.write_text(tok.read_text()[:10])  # a half-written token: the provider refuses it, the client enrolls again
    c = mk()
    try:
        assert _text(c) == want
    finally:
        c.close()
    crt = sess / "client.crt"
    crt.write_text(crt.read_text()[:100])  # a half-written certificate: detected when loaded, enrolled again
    c = mk()
    try:
        assert _text(c) == want
    finally:
        c.close()
    counter = next((tmp_path / "state").glob("sessions/*/masks.counter"))
    counter.write_bytes(b"\x00garbage")
    c = mk()
    try:
        with pytest.raises(PGWAPIError) as ei:
            _text(c)
        assert "masks.counter" in ei.value.message and str(cache) in ei.value.message
    finally:
        c.close()


# ---- shutdown ----------------------------------------------------------------------------------------------------
_EXIT_CODE = """
import json, sys
sys.path[:0] = [sys.argv[4], sys.argv[5]]
from pgw import OpenAI
from pgw.sdk import LocalPackProvisioner
c = OpenAI(api_key=sys.argv[1], cache_dir=sys.argv[2] + "/cache", state_dir=sys.argv[2] + "/state",
           mask_scheme="plain", mask_workers=2, mask_target_slots=400, mask_low_water_slots=200,
           provisioner=LocalPackProvisioner({"qwen2.5-0.5b-instruct": sys.argv[3]}))
c.chat.completions.create(model="qwen2.5-0.5b-instruct", messages=[{"role": "user", "content": "Hi"}],
                          temperature=0, max_tokens=4)
fac = c._engine.models["qwen2.5-0.5b-instruct"].factory
print(json.dumps({"factory": fac._proc.pid}), flush=True)
# no close(): interpreter exit (atexit) must stop the engine, the factory and its workers
"""


def _procs_under(path: Path) -> list[int]:
    """Processes whose working directory is under ``path`` (the factory and its workers run in its state dir)."""
    out = []
    for d in Path("/proc").iterdir():
        if d.name.isdigit():
            try:
                if str(Path(os.readlink(d / "cwd"))).startswith(str(path)) and _alive(int(d.name)):
                    out.append(int(d.name))
            except OSError:
                continue
    return out


def test_interpreter_exit_without_close_stops_everything(chaos: Chaos, tmp_path: Path) -> None:
    import subprocess
    import sys

    from tests.sdk.conftest import ROOT

    key = chaos.prov.new_key("atexit")
    p = subprocess.run([sys.executable, "-c", _EXIT_CODE, key, str(tmp_path), str(chaos.prov.pack_dir),  # noqa: S603
                        str(ROOT / "src"), str(ROOT)], capture_output=True, text=True, timeout=300,
                       env={k: v for k, v in os.environ.items() if not k.startswith("PGW_")})
    assert p.returncode == 0, p.stderr[-3000:]
    assert p.stderr == "", p.stderr[-3000:]  # silent library, also at exit
    fac = json.loads(p.stdout.strip().splitlines()[-1])["factory"]
    assert _wait(lambda: not _alive(fac), 15), "the factory outlived the interpreter"
    assert _wait(lambda: _procs_under(tmp_path) == [], 15), _procs_under(tmp_path)
