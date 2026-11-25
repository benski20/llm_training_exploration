"""End to end on CPU: a real compute service on loopback serving /pgw/v1/provision for the tiny pack, and
``pgw.provision.client.provision`` against it (artifacts, provider-built and client-verified hint, fallbacks)."""

from __future__ import annotations

import asyncio
import hashlib
import json
import socket
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest
import torch

from pgw.core.errors import FingerprintMismatchError, IntegrityError, ModelNotFoundError, PGWError
from pgw.hemask.pks_backend import PKSHintStore
from pgw.modelpack import ModelPack
from pgw.provision.client import ProvisionedModel, provision, stream_weights
from pgw.provision.pack import LINEAR_META_FILE, GatewayPack
from pgw.provision.session import ProvisionSession
from pgw.service.auth import hash_token
from pgw.service.config import ServiceSettings
from pgw.service.server import serve_async

TOKEN = "provision-test-token-0123456789abcdef"  # noqa: S105 - test credential


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class _Server:
    def __init__(self, settings: ServiceSettings):
        self.settings = settings
        self.loop = asyncio.new_event_loop()
        self.stop_event: asyncio.Event | None = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)

        async def main() -> None:
            self.stop_event = asyncio.Event()
            await serve_async(self.settings, shutdown_event=self.stop_event)

        self.loop.run_until_complete(main())

    def start(self, url: str) -> None:
        self.thread.start()
        t0 = time.monotonic()
        while time.monotonic() - t0 < 60:
            try:
                if httpx.get(f"{url}/readyz", timeout=2).status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(0.1)
        raise TimeoutError("service not ready")

    def stop(self) -> None:
        if self.stop_event is not None:
            self.loop.call_soon_threadsafe(self.stop_event.set)
        self.thread.join(timeout=30)


@pytest.fixture(scope="module")
def service(tiny_pack_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, Any]]:
    tok = tmp_path_factory.mktemp("tok") / "tenants.txt"
    tok.write_text(f"tenant-a:{hash_token(TOKEN)}\n")
    port = _free_port()
    s = ServiceSettings(host="127.0.0.1", port=port, metrics_port=None, packs=[tiny_pack_dir], device="cpu",
                        tokens_file=tok, batch_window_ms=1.0, rings=["K32A12C27"], drain_timeout_s=2.0,
                        provision=True, provision_chunk_mb=1)
    srv = _Server(s)
    url = f"http://127.0.0.1:{port}"
    srv.start(url)
    try:
        yield {"url": url, "pack_dir": tiny_pack_dir, "fp": ModelPack.open(tiny_pack_dir).fingerprint}
    finally:
        srv.stop()


def _session(service: dict[str, Any], **kw: Any) -> ProvisionSession:
    return ProvisionSession(service["url"], token=TOKEN, retries=0, **kw)


class _Tamper:
    """Wraps a session: ``fn(path, params, data) -> data`` rewrites raw byte replies."""

    def __init__(self, inner: ProvisionSession, fn: Any):
        self.inner, self.fn = inner, fn

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    def get_bytes(self, path: str, **params: Any) -> bytes:
        return self.fn(path, params, self.inner.get_bytes(path, **params))


def _hint_files(store: Any) -> dict[str, str]:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(store.path).glob("*.npy"))}


def _local_reference(pack_dir: Path, pm: ProvisionedModel, tmp: Path) -> Any:
    """A plain local build (PKSHintStore.build on the full pack) with the provisioned store's seeds."""
    st = pm.hint_store
    with ModelPack.open(pack_dir, components=("linears",)) as full:
        return PKSHintStore.build(tmp / "ref", full, seeds=(st.a_seed, st.ks_seed))


def test_catalog_and_auth(service: dict[str, Any]) -> None:
    cat = _session(service).get_json("/models")
    assert [m["fingerprint"] for m in cat["models"]] == [service["fp"]]
    r = httpx.get(service["url"] + "/pgw/v1/provision/models")
    assert r.status_code == 401
    r = httpx.get(service["url"] + "/pgw/v1/provision/models", headers={"Authorization": "Bearer wrong"})
    assert r.status_code == 401


def test_provider_hint_accepted_and_bit_identical_to_local(service: dict[str, Any], tmp_path: Path) -> None:
    events: list[dict[str, Any]] = []
    pm = provision("tiny-qwen2", session=_session(service), cache_dir=tmp_path / "cache", progress=events.append,
                   hint="provider", masked_head=False)
    try:
        assert pm.fingerprint == service["fp"] == pm.factory_fingerprint
        assert pm.hint_source == "provider-verified"
        assert pm.hint_store.meta["check"]["soundness_bits"] >= 40
        ref = _local_reference(service["pack_dir"], pm, tmp_path)
        assert _hint_files(pm.hint_store) == _hint_files(ref)
        assert pm.hint_store.meta["digests"] == ref.meta["digests"]
        assert pm.hint_store.meta["row_norms"] == ref.meta["row_norms"]
        assert {e["stage"] for e in events} >= {"artifacts", "hint", "done"}
        assert pm.sizes["h_streamed"] > 0 and pm.sizes["hint"] == ref.nbytes
        # the store reopens with the customer's MAC key
        from pgw.maskstore.keys import _read_key_file

        pm.close()
        again = PKSHintStore.open(pm.hint_root, pm.factory_fingerprint, mac_key=_read_key_file(pm.hint_key_file))
        assert again is not None and again.a_seed == pm.hint_store.a_seed
        # a second provision call reuses everything (no new seeds)
        pm2 = provision(service["fp"], session=_session(service), cache_dir=tmp_path / "cache", masked_head=False)
        assert pm2.hint_store.a_seed == pm.hint_store.a_seed and pm2.sizes["h_streamed"] == 0
        pm2.close()
    finally:
        pm.close()


def test_gateway_pack_is_weight_free_and_equal(service: dict[str, Any], tmp_path: Path) -> None:
    pm = provision(service["fp"], session=_session(service), cache_dir=tmp_path / "c", hint="none")
    assert pm.hint_store is None
    files = {p.relative_to(pm.pack_dir).as_posix() for p in pm.pack_dir.rglob("*") if p.is_file()}
    assert LINEAR_META_FILE in files and not any(f.startswith("linears-") for f in files)
    gp = pm.open_pack()
    full = ModelPack.open(service["pack_dir"])
    assert gp.fingerprint == full.fingerprint
    for e in full.manifest.linears:
        a, b = gp.load_linear_meta(e.id), full.load_linear_meta(e.id)
        assert torch.equal(a.scale, b.scale) and a.row_norm_sq_max == b.row_norm_sq_max
        assert (a.bias is None) == (b.bias is None) and (a.bias is None or torch.equal(a.bias, b.bias))
    for n in full.manifest.plain_tensor_names():
        assert torch.equal(gp.load_plain(n), full.load_plain(n))
    with pytest.raises(PGWError):
        gp.load_linear(full.manifest.linears[0].id)
    from pgw.modelpack.masked_head import MaskedHeadPack

    hg, hf = MaskedHeadPack(gp, float_head=False), MaskedHeadPack(full)
    mg, mf = hg.load_linear_meta(hg.head_id), hf.load_linear_meta(hf.head_id)
    assert torch.equal(mg.scale, mf.scale) and mg.row_norm_sq_max == mf.row_norm_sq_max
    from pgw.engine.tokenizer import ChatTokenizer

    assert ChatTokenizer.from_pack(gp).encode("hi") == ChatTokenizer.from_pack(full).encode("hi")
    # stream_weights: the factory's transient W8
    got = {lw.id.key: lw for lw in stream_weights(pm, _session(service))}
    for e in full.manifest.linears:
        assert torch.equal(got[e.id].w8, full.load_linear(e.id).w8)


def test_wrong_fingerprint_rejected(service: dict[str, Any], tmp_path: Path) -> None:
    with pytest.raises(FingerprintMismatchError):
        provision("tiny-qwen2", session=_session(service), cache_dir=tmp_path, expected_fingerprint="ab" * 32)
    with pytest.raises(ModelNotFoundError):
        provision("ab" * 32, session=_session(service), cache_dir=tmp_path)
    # a manifest whose content does not hash to the pinned fingerprint
    def swap_manifest(path: str, params: dict[str, Any], data: bytes) -> bytes:
        if params.get("id") == "manifest.json":
            doc = json.loads(data)
            doc["name"] = "evil"
            return json.dumps(doc).encode()
        return data

    s = _session(service)
    with pytest.raises(PGWError):
        provision(service["fp"], session=_Tamper(s, swap_manifest), cache_dir=tmp_path / "m", hint="none")


def test_tampered_chunk_rejected_then_resumed(service: dict[str, Any], tmp_path: Path) -> None:
    hit = {"n": 0}

    def flip(path: str, params: dict[str, Any], data: bytes) -> bytes:
        if params.get("id") == "tensor/linears.1.down.scale" and hit["n"] == 0:
            hit["n"] += 1
            return bytes([data[0] ^ 1]) + data[1:]
        return data

    cache = tmp_path / "cache"
    with pytest.raises(IntegrityError):
        provision(service["fp"], session=_Tamper(_session(service), flip), cache_dir=cache, hint="none")
    assert not (cache / "models" / service["fp"][:16] / "pack").exists()
    # resume: the chunks already written and checked are not fetched again
    fetched: list[str] = []

    def count(path: str, params: dict[str, Any], data: bytes) -> bytes:
        fetched.append(str(params.get("id")))
        return data

    pm = provision(service["fp"], session=_Tamper(_session(service), count), cache_dir=cache, hint="none")
    assert pm.open_pack().fingerprint == service["fp"]
    assert "tensor/linears.1.down.scale" in fetched
    assert "tensor/embed_tokens" not in fetched and "tensor/final_norm" not in fetched  # resumed, not refetched


def test_consistent_but_wrong_object_rejected(service: dict[str, Any], tmp_path: Path) -> None:
    """A provider that serves other bytes for an object (with matching per-chunk hashes in its listing) is caught by
    the whole-object hash from the pinned manifest."""
    real = _session(service)

    class Lying(_Tamper):
        def get_json(self, path: str, **params: Any) -> Any:
            doc = self.inner.get_json(path, **params)
            if path.endswith("/manifest"):
                for o in doc["objects"]:
                    if o["id"] == "tensor/final_norm":
                        o["chunks"] = [hashlib.sha256(b"\x00" * o["size"]).hexdigest()]
            return doc

        def get_bytes(self, path: str, **params: Any) -> bytes:
            data = self.inner.get_bytes(path, **params)
            return b"\x00" * len(data) if params.get("id") == "tensor/final_norm" else data

    with pytest.raises((IntegrityError, FingerprintMismatchError)):
        provision(service["fp"], session=Lying(real, None), cache_dir=tmp_path, hint="none")
    assert not (tmp_path / "models" / service["fp"][:16] / "pack.partial").exists()  # no resume from bad bytes


def _tamper_h(kind: str) -> Any:
    rng = np.random.default_rng(5)
    state = {"done": False}

    def fn(path: str, params: dict[str, Any], data: bytes) -> bytes:
        if "/hint/" not in path or state["done"]:
            return data
        from pgw.provision.hintcheck import decode_h56, encode_h56

        rows = len(data) // (4096 * 7)
        h = decode_h56(data, rows)
        if kind == "single":
            h[int(rng.integers(0, rows)), int(rng.integers(0, 4096))] ^= np.uint64(1 << 55)
        elif kind == "low_bit":
            h[0, 0] ^= np.uint64(1)
        elif kind == "structured":
            h ^= np.uint64(1 << 55)
        state["done"] = True
        return encode_h56(h)

    return fn


@pytest.mark.parametrize("kind", ["single", "low_bit", "structured"])
def test_adversarial_provider_hint_rejected(service: dict[str, Any], tmp_path: Path, kind: str) -> None:
    with pytest.raises(IntegrityError, match="failed the 41-bit check"):
        provision(service["fp"], session=_Tamper(_session(service), _tamper_h(kind)), cache_dir=tmp_path,
                  hint="provider", masked_head=False)
    hints = tmp_path / "models" / service["fp"][:16] / "hints"
    assert not any(p.suffix != ".lock" for p in hints.iterdir())  # nothing stored, not even partially


def test_wrong_seed_provider_hint_rejected(service: dict[str, Any], tmp_path: Path, monkeypatch: Any) -> None:
    """A provider that ignores the client's a_seed (uses its own) is rejected."""
    from pgw.core.prng import new_seed
    from pgw.service import provision as svc_mod

    orig = svc_mod.ProvisionService.create_job
    evil = new_seed()

    def create_job(self: Any, tenant: str, fp: str, a_seed: bytes, **kw: Any) -> Any:
        return orig(self, tenant, fp, evil, **kw)

    monkeypatch.setattr(svc_mod.ProvisionService, "create_job", create_job)
    with pytest.raises(IntegrityError):
        provision(service["fp"], session=_session(service), cache_dir=tmp_path, hint="provider", masked_head=False)


def test_local_hint_and_public_source_fallback(service: dict[str, Any], tiny_hf_dir: Path, tmp_path: Path) -> None:
    """hint='local' (no provider hint) equals the local build; source='hf' re-derives the pack from the public
    checkpoint (here a local HF directory) using the provider's manifest parameters and lands on the same pin."""
    pm = provision(service["fp"], session=_session(service), cache_dir=tmp_path / "a", hint="local",
                   masked_head=False)
    assert pm.hint_source == "local"
    ref = _local_reference(service["pack_dir"], pm, tmp_path)
    assert _hint_files(pm.hint_store) == _hint_files(ref)
    pm.close()
    pm2 = provision(service["fp"], session=_session(service), cache_dir=tmp_path / "b", hint="provider",
                    masked_head=False, source="hf", hf_repo=str(tiny_hf_dir))
    assert pm2.fingerprint == service["fp"] and pm2.hint_source == "provider-verified"
    assert not (tmp_path / "b" / "models" / service["fp"][:16] / "full").exists()  # no large storage kept
    gp = pm2.open_pack()
    assert gp.fingerprint == service["fp"]
    pm2.close()
    # no provider at all: source='hf' + hint='local' with the pin and the packing parameters
    man = ModelPack.open(service["pack_dir"]).manifest
    pm3 = provision(service["fp"], session=None, cache_dir=tmp_path / "c", hint="local", masked_head=False,
                    source="hf", hf_repo=str(tiny_hf_dir), hf_name=man.name, hf_plain_dtype=man.quant.plain_dtype)
    assert pm3.fingerprint == service["fp"]
    pm3.close()
    with pytest.raises(FingerprintMismatchError):
        provision(service["fp"], session=None, cache_dir=tmp_path / "d", hint="none", masked_head=False,
                  source="hf", hf_repo=str(tiny_hf_dir), hf_name="other-name")


def test_provision_disabled_by_default(tiny_pack_dir: Path) -> None:
    assert ServiceSettings().provision is False


def test_gateway_pack_tampered_meta_rejected(service: dict[str, Any], tmp_path: Path) -> None:
    pm = provision(service["fp"], session=_session(service), cache_dir=tmp_path, hint="none")
    p = pm.pack_dir / LINEAR_META_FILE
    raw = bytearray(p.read_bytes())
    raw[-1] ^= 1
    p.write_bytes(bytes(raw))
    with pytest.raises(PGWError):
        GatewayPack.open_gateway(pm.pack_dir, expected_fingerprint=pm.fingerprint, verify="full")
