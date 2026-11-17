"""End-to-end stack on localhost: real CLI, real processes, real encrypted mask store.

``e2e_stack`` (module scope) builds, through the ``pgw`` command line exactly as an operator would:

1. ``pgw pack``             - Qwen/Qwen2.5-0.5B-Instruct (or the tiny random Qwen2 when the hub is
                              unreachable, ``PGW_OFFLINE=1`` or ``PGW_E2E_MODEL=tiny``);
2. ``pgw maskstore init``   - encrypted single-use mask store + KEK;
3. ``pgw factory generate`` - masks into the store;
4. ``pgw service token create`` / ``pgw gateway keys create``;
5. ``pgw service serve``    - compute service process on 127.0.0.1;
6. a recording proxy (:mod:`tests.e2e.proxy`) between gateway and service;
7. ``pgw gateway serve``    - gateway process on 127.0.0.1 with the production policy
                              (masked mode, Freivalds on every response, noise monitor fail-closed).

The QuantPlain reference (:class:`pgw.engine.PlainBackend`) runs in the test process on the same pack.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import pytest
import torch

from pgw.core import RingParams
from tests.cli.conftest import free_port
from tests.e2e.proxy import RecordingProxy

RING = "K40A12"
MODEL = "e2e"
ADMIN_TOKEN = "e2e-admin-token-0123456789abcdef"  # noqa: S105 - test-only value
QWEN = "Qwen/Qwen2.5-0.5B-Instruct"


def _e2e_model_source(tiny_hf_dir: Path) -> tuple[Path, str]:
    """``(checkpoint dir, label)``: the real Qwen2.5-0.5B when reachable, else the tiny model."""
    want = os.environ.get("PGW_E2E_MODEL", "qwen").lower()
    if want == "tiny" or os.environ.get("PGW_OFFLINE") == "1":
        return tiny_hf_dir, "tiny"
    try:
        from huggingface_hub import snapshot_download

        p = Path(snapshot_download(QWEN, allow_patterns=["*.json", "*.safetensors", "*.txt"]))
        return p, "qwen0.5b"
    except Exception as exc:  # pragma: no cover - network dependent
        print(f"e2e: cannot download {QWEN} ({exc!r}); using the tiny random model", file=sys.stderr)
        return tiny_hf_dir, "tiny"


def _base_env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("PGW_")}
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    env["PGW_LOG_LEVEL"] = "INFO"
    env.setdefault("OMP_NUM_THREADS", "2")
    return env


def run_cli(*args: str, env: dict[str, str] | None = None, timeout: float = 1800) -> subprocess.CompletedProcess[str]:
    """``pgw <args>`` in a subprocess; raises with the output on failure."""
    r = subprocess.run([sys.executable, "-m", "pgw.cli.main", *args], env=env or _base_env(),  # noqa: S603
                       capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise AssertionError(f"pgw {' '.join(args)} exited {r.returncode}:\n{r.stdout[-3000:]}\n{r.stderr[-6000:]}")
    return r


@dataclass
class Proc:
    """A long-running ``pgw`` process with its log file."""

    name: str
    popen: subprocess.Popen[bytes]
    log: Path

    def tail(self, n: int = 6000) -> str:
        try:
            return self.log.read_text(errors="replace")[-n:]
        except OSError:  # pragma: no cover
            return ""

    def stop(self, sig: int = signal.SIGTERM, timeout: float = 60) -> None:
        if self.popen.poll() is None:
            self.popen.send_signal(sig)
            try:
                self.popen.wait(timeout=timeout)
            except subprocess.TimeoutExpired:  # pragma: no cover
                self.popen.kill()
                self.popen.wait(timeout=30)


def start_proc(name: str, args: list[str], env: dict[str, str], logdir: Path) -> Proc:
    log = logdir / f"{name}.log"
    fh = log.open("ab")
    p = subprocess.Popen([sys.executable, "-m", "pgw.cli.main", *args], env=env,  # noqa: S603
                         stdout=fh, stderr=subprocess.STDOUT)
    fh.close()
    return Proc(name, p, log)


def wait_ready(url: str, proc: Proc, timeout: float = 600) -> None:
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if proc.popen.poll() is not None:
            raise AssertionError(f"{proc.name} exited {proc.popen.returncode}:\n{proc.tail()}")
        try:
            if httpx.get(url, timeout=5.0, trust_env=False).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.5)
    raise AssertionError(f"{proc.name} not ready at {url}:\n{proc.tail()}")


@dataclass
class E2EStack:
    """Handles to the running stack."""

    root: Path
    label: str
    hf_dir: Path
    pack_dir: Path
    fingerprint: str
    ring: RingParams
    store_root: Path
    key_file: Path
    slots_generated: int
    api_key: str
    service: Proc
    gateway: Proc
    proxy: RecordingProxy
    gw_url: str
    gw_env: dict[str, str]
    gw_port: int
    logdir: Path
    in_features: dict[str, int]
    all_items: list[Any] = field(default_factory=list)  # every compute item the service ever received

    # ---- gateway API helpers -----------------------------------------------------------------------
    def chat(self, body: dict[str, Any], timeout: float = 600) -> httpx.Response:
        return httpx.post(f"{self.gw_url}/v1/chat/completions", json=body, timeout=timeout, trust_env=False,
                          headers={"Authorization": f"Bearer {self.api_key}"})

    def chat_stream(self, body: dict[str, Any], timeout: float = 600) -> tuple[int, list[dict[str, Any]]]:
        events: list[dict[str, Any]] = []
        with httpx.stream("POST", f"{self.gw_url}/v1/chat/completions", json={**body, "stream": True},
                          timeout=timeout, trust_env=False,
                          headers={"Authorization": f"Bearer {self.api_key}"}) as r:
            if r.status_code != 200:
                r.read()
                return r.status_code, [{"error": r.text}]
            for line in r.iter_lines():
                if line.startswith("data: "):
                    data = line[len("data: "):]
                    if data.strip() == "[DONE]":
                        break
                    events.append(json.loads(data))
        return 200, events

    def admin(self, method: str, path: str, **kw: Any) -> httpx.Response:
        return httpx.request(method, f"{self.gw_url}{path}", timeout=60, trust_env=False,
                             headers={"Authorization": f"Bearer {ADMIN_TOKEN}"}, **kw)

    def available_slots(self) -> int:
        r = self.admin("GET", "/admin/masks")
        assert r.status_code == 200, r.text
        return int(r.json()["models"][MODEL]["available_slots"])

    def capture(self) -> list[Any]:
        items = self.proxy.take()
        self.all_items.extend(items)
        return items

    def restart_gateway(self, sig: int = signal.SIGKILL) -> None:
        """Stop the gateway (SIGKILL simulates a crash) and start a fresh process."""
        self.gateway.stop(sig)
        self.gateway = start_proc("gateway", ["gateway", "serve"], self.gw_env, self.logdir)
        wait_ready(f"{self.gw_url}/readyz", self.gateway)


def _slots_for(label: str) -> tuple[int, int]:
    """``(slots per segment, segments)``."""
    return (192, 2) if label != "tiny" else (256, 2)


@pytest.fixture(scope="module")
def e2e_stack(tiny_hf_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> Iterator[E2EStack]:
    from pgw.modelpack import ModelPack

    hf_dir, label = _e2e_model_source(tiny_hf_dir)
    root = tmp_path_factory.mktemp("e2e")
    logdir = root / "logs"
    logdir.mkdir()
    env = _base_env()
    ring = RingParams.from_name(RING)

    # 1. pack
    pack_dir = root / "packs" / MODEL
    run_cli("pack", str(hf_dir), str(pack_dir), "--name", MODEL, "-q", env=env)
    info = json.loads(run_cli("inspect", str(pack_dir), "--json", env=env).stdout)
    fp = info["fingerprint"]
    pack = ModelPack.open(pack_dir)
    in_features = {s.id.key: s.in_features for s in pack.linear_specs()}
    pack.close()

    # 2./3. mask store + factory
    store_root, key_file = root / "masks", root / "kek" / "kek"
    key_file.parent.mkdir(mode=0o700)
    run_cli("maskstore", "init", "--root", str(store_root), "--key-file", str(key_file), env=env)
    slots, segments = _slots_for(label)
    run_cli("factory", "generate", "--pack", str(pack_dir), "--store", str(store_root), "--key-file",
            str(key_file), "--ring", RING, "--slots", str(slots), "--segments", str(segments), env=env)

    # 4. credentials
    secrets = root / "secrets"
    secrets.mkdir(mode=0o700)
    run_cli("service", "token", "create", "--tenant", "e2e", "--tokens-file", str(secrets / "tokens"),
            "--out", str(secrets / "service_token"), env=env)
    keys_file = root / "state" / "api_keys.json"
    keys_file.parent.mkdir()
    gw_key = json.loads(run_cli("gateway", "keys", "create", "--name", "e2e", "--keys-file", str(keys_file),
                                env=env).stdout)["key"]

    # 5. service
    svc_port = free_port()
    svc_env = {**env, "PGW_SERVICE_HOST": "127.0.0.1", "PGW_SERVICE_PORT": str(svc_port),
               "PGW_SERVICE_PACKS": json.dumps([str(pack_dir)]), "PGW_SERVICE_RINGS": json.dumps([RING]),
               "PGW_SERVICE_TOKENS_FILE": str(secrets / "tokens"), "PGW_SERVICE_DEVICE": "cpu",
               "PGW_SERVICE_VERIFY_PACKS": "full", "PGW_SERVICE_CPU_WORKERS": "1"}
    service = start_proc("service", ["service", "serve"], svc_env, logdir)
    proxy: RecordingProxy | None = None
    gateway: Proc | None = None
    try:
        wait_ready(f"http://127.0.0.1:{svc_port}/readyz", service)
        # 6. proxy (the network path to the service)
        proxy = RecordingProxy(f"http://127.0.0.1:{svc_port}", in_features, ring)
        # 7. gateway
        models_file = secrets / "models.json"
        models_file.write_text(json.dumps([{
            "name": MODEL, "pack_path": str(pack_dir), "fingerprint": fp, "ring": RING,
            "service_url": proxy.url, "service_token_file": str(secrets / "service_token"),
            "max_concurrent_sessions": 4,
        }]))
        models_file.chmod(0o600)
        gw_port = free_port()
        gw_env = {**env, "PGW_GATEWAY_HOST": "127.0.0.1", "PGW_GATEWAY_PORT": str(gw_port),
                  "PGW_GATEWAY_MODELS_FILE": str(models_file), "PGW_GATEWAY_API_KEYS_FILE": str(keys_file),
                  "PGW_GATEWAY_ADMIN_TOKEN": ADMIN_TOKEN, "PGW_GATEWAY_MASKSTORE_ROOT": str(store_root),
                  "PGW_GATEWAY_AUDIT_LOG_FILE": str(root / "state" / "audit.jsonl"),
                  "PGW_GATEWAY_NOISE_SAMPLE_RATE": "1.0", "PGW_GATEWAY_NOISE_MONITOR_REQUIRED": "true",
                  "PGW_MASKSTORE_ROOT": str(store_root), "PGW_MASKSTORE_KEY_FILE": str(key_file),
                  "PGW_MASKSTORE_LOW_WATER_SLOTS": "16"}
        gateway = start_proc("gateway", ["gateway", "serve"], gw_env, logdir)
        gw_url = f"http://127.0.0.1:{gw_port}"
        wait_ready(f"{gw_url}/readyz", gateway)
        stack = E2EStack(root, label, hf_dir, pack_dir, fp, ring, store_root, key_file, slots * segments,
                         gw_key, service, gateway, proxy, gw_url, gw_env, gw_port, logdir, in_features)
        yield stack
        gateway = stack.gateway
    finally:
        for p in (gateway, service):
            if p is not None:
                p.stop()
        if proxy is not None:
            proxy.close()


# ---- QuantPlain reference ---------------------------------------------------------------------------
class RecordingPlainBackend:
    """Wraps :class:`pgw.engine.PlainBackend`; records the plaintext quantized activations of every call."""

    def __init__(self, pack: Any, ring: RingParams):
        from pgw.engine import PlainBackend

        self.inner = PlainBackend(pack, ring)
        self.ring = ring
        self.calls: list[tuple[str, torch.Tensor]] = []

    def _record(self, calls: Any) -> None:
        from pgw.core import quantize_activations

        for c in calls:
            xq, _ = quantize_activations(c.x, self.ring)
            self.calls.append((c.id.key, xq))

    def forward(self, call: Any) -> torch.Tensor:
        self._record([call])
        return self.inner.forward(call)

    def forward_many(self, calls: Any) -> list[torch.Tensor]:
        self._record(calls)
        return self.inner.forward_many(calls)

    async def aforward(self, call: Any) -> torch.Tensor:
        return self.forward(call)

    async def aforward_many(self, calls: Any) -> list[torch.Tensor]:
        return self.forward_many(calls)

    def close(self) -> None:
        self.inner.close()


@dataclass
class Reference:
    """In-process QuantPlain engine on the same pack."""

    pack: Any
    tokenizer: Any
    backend: RecordingPlainBackend
    generator: Any


@pytest.fixture(scope="module")
def reference(e2e_stack: E2EStack) -> Iterator[Reference]:
    from pgw.engine import ChatTokenizer, Generator, TransformerModel
    from pgw.modelpack import ModelPack

    pack = ModelPack.open(e2e_stack.pack_dir, expected_fingerprint=e2e_stack.fingerprint)
    backend = RecordingPlainBackend(pack, e2e_stack.ring)
    model = TransformerModel(pack, backend)  # type: ignore[arg-type]
    tok = ChatTokenizer.from_pack(pack)
    yield Reference(pack, tok, backend, Generator(model, tok, batching=False))
    pack.close()
