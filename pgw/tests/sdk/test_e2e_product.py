"""The product end to end on CPU: provider service, admin CLI and API, client library, console.

One real ``pgw service`` process on loopback with TLS, the TCP+TLS trip lane, the masked LM head, accounts (SDK
keys, the admin API, request reports), provisioning and the developer console. Then, as a customer would:

1. the operator creates an organisation and a console sign-in token with the CLI (``pgw service orgs create``,
   ``pgw service admin-token create --org``) and an SDK key with the admin API;
2. with an empty cache directory, ``from pgw import OpenAI`` -> enroll (``POST /pgw/v1/sdk/enroll``) -> provision
   (``/pgw/v1/provision/...``: the weight-free gateway pack, checked against the library's baked fingerprint pin,
   and the factory pack with the fingerprint-checked int8 weights) -> chat, non-streaming and streaming, each
   bit-identical to the existing gateway path (``pgw gateway``'s plaintext reference with the same settings);
3. usage (provider-metered trips and tokens, client-reported requests and the input/output split) and the
   content-free request log are visible through the admin API and the console, without any content;
4. the key is revoked and the next call is refused (an already connected client and a new one).

Model: ``Qwen/Qwen2.5-0.5B-Instruct`` at the release's pinned commit when it is in the local Hugging Face cache (the
provider packs it here and its fingerprint must equal the library's baked pin), else the tiny random model with an
explicit ``model_fingerprints`` pin.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from tests.sdk.conftest import MODEL, Provider, start_provider, stop_provider

pytestmark = pytest.mark.timeout(3600)

PROMPT = [{"role": "system", "content": "You are terse."},
          {"role": "user", "content": "Name a colour of the sky. Secret-marker-7f3a."}]
GREEDY = {"temperature": 0, "max_tokens": 8, "logprobs": True, "top_logprobs": 2, "extra_body": {"ignore_eos": True}}
SAMPLED = {"temperature": 0.8, "top_p": 0.95, "seed": 77, "max_tokens": 8, "logprobs": True, "top_logprobs": 2,
           "extra_body": {"ignore_eos": True}}


def _hf_snapshot(repo: str, revision: str) -> Path | None:
    home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    snap = home / "hub" / ("models--" + repo.replace("/", "--")) / "snapshots" / revision
    return snap if (snap / "model.safetensors").is_file() and (snap / "tokenizer.json").is_file() else None


@pytest.fixture(scope="module")
def product_pack(tiny_hf_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    from pgw.modelpack import ModelPack, pack_checkpoint
    from pgw.sdk.provisioning import PINNED_MODELS

    pin = PINNED_MODELS[MODEL]
    out = tmp_path_factory.mktemp("product-pack") / "pack"
    real = _hf_snapshot(pin.hf_repo, pin.revision) is not None and os.environ.get("PGW_SDK_E2E_MODEL") != "tiny"
    if real:  # exactly what the provider image does on first boot (deploy/runpod/pod_boot.ensure_pack), offline
        old = os.environ.get("HF_HUB_OFFLINE")
        os.environ["HF_HUB_OFFLINE"] = "1"
        try:
            pack_checkpoint(pin.hf_repo, out, name=MODEL, revision=pin.revision)
        finally:
            if old is None:
                os.environ.pop("HF_HUB_OFFLINE", None)
            else:
                os.environ["HF_HUB_OFFLINE"] = old
    else:
        pack_checkpoint(tiny_hf_dir, out, name=MODEL)
    with ModelPack.open(out) as p:
        fp = p.fingerprint
    return {"dir": out, "real": real, "fp": fp, "pin": pin}


@pytest.fixture(scope="module")
def prov(product_pack: dict[str, Any], pki: dict[str, Any], tmp_path_factory: pytest.TempPathFactory
         ) -> Iterator[Provider]:
    p = start_provider(tmp_path_factory.mktemp("product-provider"), product_pack["dir"], pki)
    try:
        yield p
    finally:
        stop_provider(p)


@pytest.fixture(scope="module")
def st(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    return {"root": tmp_path_factory.mktemp("product-client")}


def _client(prov: Provider, st: dict[str, Any], pack: dict[str, Any], **kw: Any) -> Any:
    from pgw import OpenAI

    opts: dict[str, Any] = {"cache_dir": st["root"] / "cache", "state_dir": st["root"] / "state",
                            "mask_scheme": "plain", "mask_workers": 1, "mask_target_slots": 256,
                            "mask_low_water_slots": 128, "mask_segment_slots": 64, "timeout": 900,
                            "ready_timeout_s": 1800}
    if not pack["real"]:  # the tiny model is not a release model: pin it explicitly
        opts["model_fingerprints"] = {MODEL: pack["fp"]}
    opts.update(kw)
    return OpenAI(api_key=st["key"], **opts)  # the default provisioner: the provider's real endpoints


def _reference(pack_dir: Path, body: dict[str, Any]) -> dict[str, Any]:
    from tests.sdk.test_e2e_cpu import _reference as ref

    return ref(pack_dir, body, head_mode="provider_int8", messages=PROMPT)


def _cli(*args: str, db: Path) -> Any:
    from click.testing import CliRunner

    from pgw.cli.main import cli

    r = CliRunner().invoke(cli, ["service", *args, "--accounts-db", str(db)], catch_exceptions=False)
    assert r.exit_code == 0, r.output
    return json.loads(r.stdout)


def test_pin_matches_the_release(product_pack: dict[str, Any]) -> None:
    """The provider's pack of the public checkpoint has the fingerprint the library bakes in (derived independently
    here with the pack tool), so the client accepts it without trusting the provider's catalog."""
    if not product_pack["real"]:
        pytest.skip("tiny model in use (Qwen2.5-0.5B-Instruct not in the local Hugging Face cache, or PGW_SDK_E2E_MODEL=tiny)")
    assert product_pack["fp"] == product_pack["pin"].fingerprint


def test_operator_creates_org_console_token_and_key(prov: Provider, st: dict[str, Any]) -> None:
    db = prov.root / "secrets" / "accounts.db"
    org = _cli("orgs", "create", "product", "--name", "Product Inc", db=db)
    assert org["org_id"] == "product"
    tok = _cli("admin-token", "create", "--name", "console", "--org", "product", db=db)
    st["console_token"] = tok["token"]
    prov.org = "product"
    st["key"] = prov.new_key("backend", models=[MODEL])
    assert st["key"].startswith("pgw_test_")
    keys = prov.admin("GET", "/orgs/product/keys")["keys"]
    assert [k["models"] for k in keys] == [[MODEL]] and st["key"] not in json.dumps(keys)  # shown once


def test_first_use_enroll_provision_chat_bit_identical(prov: Provider, st: dict[str, Any],
                                                       product_pack: dict[str, Any]) -> None:
    root = st["root"]
    assert not (root / "cache").exists()  # a fresh, empty cache
    c = _client(prov, st, product_pack)
    st["calls"] = []
    try:
        t0 = time.monotonic()
        r = c.chat.completions.create(model=MODEL, messages=PROMPT, **GREEDY)
        st["first_s"] = time.monotonic() - t0
        st["calls"].append(r.usage)
        ref = _reference(product_pack["dir"], GREEDY)
        assert r.choices[0].message.content == ref["choices"][0]["message"]["content"]
        assert r.to_dict()["choices"][0]["logprobs"]["content"] == ref["choices"][0]["logprobs"]["content"]
        chunks = list(c.chat.completions.create(model=MODEL, messages=PROMPT, stream=True,
                                                stream_options={"include_usage": True}, **SAMPLED))
        sref = _reference(product_pack["dir"], SAMPLED)
        text = "".join(ch.choices[0].delta.content or "" for ch in chunks if ch.choices)
        assert text == sref["choices"][0]["message"]["content"]
        lps = [t for ch in chunks if ch.choices and ch.choices[0].logprobs
               for t in ch.to_dict()["choices"][0]["logprobs"]["content"]]
        assert lps == sref["choices"][0]["logprobs"]["content"]  # bit for bit
        st["calls"].append(chunks[-1].usage)
        m = c.status()["models"][MODEL]
        assert m["state"] == "ready" and m["masked_head"] is True and m["lane"] == "open", m
        assert m["integrity_failures"] == 0 and m["verified_requests"] == 2
        p = m["provisioned"]
        assert p["fingerprint"] == product_pack["fp"] and p["source"] == "provider"
        pack = Path(p["pack_path"])
        assert (pack / "linear-meta.safetensors").is_file() and not list(pack.glob("linears-*"))  # weight-free
        fpk = pack.parent / "factory-pack"
        assert list(fpk.glob("linears-*")), "the factory pack holds the checked int8 weights"
    finally:
        c.close()
    print(f"\nfirst request on an empty cache (enroll + provision + connect + first masks) {st['first_s']:.1f}s"
          f" ({'Qwen2.5-0.5B-Instruct' if product_pack['real'] else 'tiny model'})")


def test_usage_and_request_log_without_content(prov: Provider, st: dict[str, Any]) -> None:
    prompt_tokens = sum(u.prompt_tokens for u in st["calls"])
    completion_tokens = sum(u.completion_tokens for u in st["calls"])
    deadline = time.monotonic() + 60
    rows: list[dict[str, Any]] = []
    while time.monotonic() < deadline:
        rows = prov.admin("GET", "/orgs/product/usage")["usage"]
        if rows and rows[0].get("requests", 0) >= 2:
            break
        time.sleep(0.5)
    (row,) = rows
    assert row["model"] == MODEL and row["trips"] > 0 and row["tokens"] > 0  # metered by the provider
    assert row["requests"] == 2 and row["errors"] == 0 and row["verified"] == 2  # reported by the client
    assert (row["prompt_tokens"], row["completion_tokens"]) == (prompt_tokens, completion_tokens)
    log = prov.admin("GET", "/orgs/product/requests")
    recs = log["requests"]
    assert len(recs) == 2 and {r["model"] for r in recs} == {MODEL}
    assert sorted(r["stream"] for r in recs) == [False, True]
    allowed = {"id", "ts", "key_id", "model", "status", "error_code", "prompt_tokens", "completion_tokens",
               "latency_ms", "ttft_ms", "stream", "private", "verified", "verification"}
    for r in recs:
        assert set(r) <= allowed and r["status"] == "ok" and r["verified"] is True
        assert r["verification"] == "client-reported" and r["latency_ms"] > 0
    text = json.dumps(log) + json.dumps(rows)
    assert "Secret-marker-7f3a" not in text and "sky" not in text  # never content

    # the console (mounted on the provider) shows the same, signed in with the organisation's token
    with httpx.Client(base_url=prov.url, verify=str(prov.ca_file), trust_env=False, timeout=30) as h:
        r = h.post("/platform/login", data={"token": st["console_token"]}, follow_redirects=False)
        assert r.status_code == 303, r.text[:500]
        page = h.get("/platform/requests").text
        assert MODEL in page and "checked client-side" in page and "Secret-marker-7f3a" not in page
        usage = h.get("/platform/usage").text
        assert MODEL in usage and "Secret-marker-7f3a" not in usage
        assert h.get("/platform/keys").status_code == 200


def test_default_scheme_pks_v3_with_local_hint(prov: Provider, st: dict[str, Any],
                                               product_pack: dict[str, Any]) -> None:
    """The library's defaults: pks-v3 masks from the provider's encrypted maskgen, with the one-time hint built on
    the client from the factory pack (``hint="local"``); bit-identical to the gateway path. Real model only (the
    tiny model's 260-token LM head is not a pks-v3 shape; tests/sdk/test_e2e_cpu.py covers it without the head)."""
    if not product_pack["real"]:
        pytest.skip("needs Qwen2.5-0.5B-Instruct in the local Hugging Face cache")
    from pgw.sdk.options import SDKOptions

    assert SDKOptions().mask_scheme == "pks-v3" and SDKOptions().hint == "local"
    root = st["root"] / "defaults"
    # masked_head off: encrypted maskgen for the 151936 x 896 head on this provider's CPU needed ~10 GB and was
    # OOM-killed on this 15 GB test box (the GPU provider does it routinely); the layers' pks-v3 path is tested here
    c = _client(prov, {**st, "root": root}, product_pack, mask_scheme="pks-v3", mask_target_slots=48,
                mask_low_water_slots=24, mask_segment_slots=16, mask_workers=1, masked_head="off")
    try:
        t0 = time.monotonic()
        r = c.chat.completions.create(model=MODEL, messages=PROMPT, **GREEDY)
        dt_s = time.monotonic() - t0
        from tests.sdk.test_e2e_cpu import _reference as ref_gateway

        ref = ref_gateway(product_pack["dir"], GREEDY, head_mode="gateway", messages=PROMPT)
        assert r.to_dict()["choices"][0]["logprobs"]["content"] == ref["choices"][0]["logprobs"]["content"]
        f = c.status()["models"][MODEL]["factory"]
        assert f["scheme"] == "pks-v3" and not f["halted"] and not f.get("provider_integrity_failures")
        assert list((root / "cache" / "models").glob("*/hint/*")), "the pks-v3 hint was built on the client"
    finally:
        c.close()
    print(f"\npks-v3 with the hint built locally: first answer after {dt_s:.0f}s (Qwen2.5-0.5B-Instruct, empty cache)")


def test_revoked_key_is_refused(prov: Provider, st: dict[str, Any], product_pack: dict[str, Any]) -> None:
    from pgw.sdk import AuthenticationError, PGWAPIError

    c = _client(prov, st, product_pack)
    try:
        r = c.chat.completions.create(model=MODEL, messages=PROMPT, **GREEDY)  # cached session and packs
        assert r.choices[0].finish_reason == "length"
        prov.revoke(st["key"])
        time.sleep(1.5)  # > accounts_refresh_s (the revoking process refuses at once; this covers the others)
        with pytest.raises(PGWAPIError) as ei:
            c.chat.completions.create(model=MODEL, messages=PROMPT, **GREEDY)
        assert ei.value.code in ("unauthenticated", "permission_denied"), ei.value
    finally:
        c.close()
    fresh = _client(prov, {**st, "root": st["root"] / "fresh"}, product_pack)
    try:
        with pytest.raises(AuthenticationError):
            fresh.chat.completions.create(model=MODEL, messages=PROMPT, **GREEDY)
    finally:
        fresh.close()
    events = [e["event"] for e in prov.admin("GET", "/orgs/product/audit")["events"]]
    assert "key.revoked" in events and "sdk.enroll.failed" in events
