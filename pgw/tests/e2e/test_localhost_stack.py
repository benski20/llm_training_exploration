"""PGW end to end on localhost: pack -> factory -> encrypted store -> service process -> gateway process.

The tests in this module share one running stack (``e2e_stack``) and run in file order:

1. non-streaming chat completion: answer tokens and logprobs identical to the QuantPlain reference;
2. streaming chat completion: same tokens;
3. every compute response was Freivalds-checked, none failed;
4. the service never received plaintext: each captured ``x_hat`` is compared with the reference's
   plaintext quantized activations for the same request (same linear/row structure, no agreement
   beyond chance, uniform differences), and the prompt bytes never appear on the wire;
5. all captured outbound traffic passes the outbound-noise monitor (and plaintext would not);
6. a tampered service response is detected (502), trips the circuit breaker (503), burns the
   request's masks, and the stack recovers after an admin reset;
7. masks cannot be reused: no outbound row repeats across the whole run, including across a
   gateway crash (SIGKILL) and restart; store accounting is exact; an exhausted store refuses; a
   rolled-back store database is refused.

Model: Qwen/Qwen2.5-0.5B-Instruct (downloaded once from the HF hub), or the tiny random Qwen2 with
``PGW_E2E_MODEL=tiny`` / ``PGW_OFFLINE=1`` / no network.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from typing import Any

import pytest
import torch

from pgw.core import IntegrityError, MaskExhaustedError, PGWError
from pgw.engine import SamplingParams
from tests.e2e.conftest import MODEL, E2EStack, Reference

pytestmark = [pytest.mark.slow, pytest.mark.e2e]

MESSAGES = [{"role": "user", "content": "What is the capital of France? Answer in one word."}]
MAX_TOKENS = 8
CHAT = {"model": MODEL, "messages": MESSAGES, "temperature": 0, "max_tokens": MAX_TOKENS, "ignore_eos": True,
        "logprobs": True, "top_logprobs": 2}

#: Per-module scratch shared between the ordered tests.
_STATE: dict[str, Any] = {}


def _reference(ref: Reference) -> dict[str, Any]:
    """Run the QuantPlain reference once for CHAT; record tokens, logprobs and plaintext xq per call."""
    if "ref" not in _STATE:
        ids = ref.tokenizer.apply_chat_template(MESSAGES, add_generation_prompt=True)
        ref.backend.calls.clear()
        params = SamplingParams(max_tokens=MAX_TOKENS, temperature=0.0, logprobs=2, ignore_eos=True)
        events = list(ref.generator.stream(ids, params))
        _STATE["ref"] = {
            "prompt_ids": ids,
            "token_ids": [e.token_id for e in events],
            "logprobs": [e.logprob for e in events],
            "top": [[(i, lp) for i, lp in (e.top_logprobs or [])] for e in events],
            "text": "".join(e.text or "" for e in events),
            "calls": list(ref.backend.calls),
        }
        ref.backend.calls.clear()
    return _STATE["ref"]


def _tok(ref: Reference, tid: int) -> str:
    return str(ref.tokenizer.decode([tid], skip_special_tokens=False))


# ---- 1. exactness --------------------------------------------------------------------------------------
def test_nonstreaming_matches_quantplain_reference(e2e_stack: E2EStack, reference: Reference) -> None:
    exp = _reference(reference)
    e2e_stack.proxy.take()
    e2e_stack.proxy.take_raw()
    before = e2e_stack.available_slots()
    r = e2e_stack.chat(CHAT)
    assert r.status_code == 200, r.text
    _STATE["chat_items"] = e2e_stack.capture()
    _STATE["chat_raw"] = e2e_stack.proxy.take_raw()
    body = r.json()
    choice = body["choices"][0]
    assert body["usage"]["prompt_tokens"] == len(exp["prompt_ids"])
    assert body["usage"]["completion_tokens"] == MAX_TOKENS == len(exp["token_ids"])
    assert body["system_fingerprint"] == e2e_stack.fingerprint[:12]
    content = choice["logprobs"]["content"]
    # identical answer tokens (as token strings and bytes), identical log-probabilities (bit for bit)
    assert [c["token"] for c in content] == [_tok(reference, t) for t in exp["token_ids"]]
    assert [c["logprob"] for c in content] == exp["logprobs"]
    assert [[(t["token"], t["logprob"]) for t in c["top_logprobs"]] for c in content] == [
        [(_tok(reference, i), lp) for i, lp in top] for top in exp["top"]
    ]
    assert choice["message"]["content"] == exp["text"]
    # single use: exactly prompt + max_tokens slots consumed for this request
    assert before - e2e_stack.available_slots() == len(exp["prompt_ids"]) + MAX_TOKENS


def test_streaming_matches_quantplain_reference(e2e_stack: E2EStack, reference: Reference) -> None:
    exp = _reference(reference)
    status, events = e2e_stack.chat_stream(CHAT)
    assert status == 200, events
    e2e_stack.capture()
    deltas = [ev["choices"][0]["delta"].get("content") or "" for ev in events if ev.get("choices")]
    assert "".join(deltas) == exp["text"]
    lps = [c["logprob"] for ev in events if ev.get("choices") and ev["choices"][0].get("logprobs")
           for c in ev["choices"][0]["logprobs"]["content"]]
    assert lps == exp["logprobs"]
    finals = [ev["choices"][0].get("finish_reason") for ev in events if ev.get("choices")]
    assert finals[-1] == "length"


# ---- 2. integrity ------------------------------------------------------------------------------------
def test_every_response_freivalds_checked(e2e_stack: E2EStack) -> None:
    st = e2e_stack.admin("GET", "/admin/status")
    assert st.status_code == 200, st.text
    model = next(m for m in st.json()["models"] if m["name"] == MODEL)
    be = model["backend"]
    assert be["round_trips"] > 0
    assert be["freivalds_checked"] == be["round_trips"]
    assert be["freivalds_skipped"] == 0 and be["integrity_failures"] == 0
    assert model["noise_monitor"]["state"] == "active" and not model["noise_monitor"]["alarm"]


# ---- 3. privacy: what the service saw -------------------------------------------------------------------
def test_service_never_receives_plaintext(e2e_stack: E2EStack, reference: Reference) -> None:
    exp = _reference(reference)
    items = _STATE["chat_items"]
    ring = e2e_stack.ring
    calls = exp["calls"]
    # same linear/row structure as the reference run of the same request
    assert [(i.linear, i.rows) for i in items] == [(k, int(x.shape[0])) for k, x in calls]
    total = equal = small = 0
    bound = ring.qmax
    for it, (_, xq) in zip(items, calls, strict=True):
        xh = it.x_hat
        assert xh.dtype == torch.int64 and xh.min() >= 0 and xh.max() < ring.modulus
        total += xh.numel()
        # plaintext sent as-is (or as signed values mod 2^K) would make these equal everywhere
        equal += int((xh == (xq % ring.modulus)).sum())
        centered = torch.where(xh >= ring.modulus // 2, xh - ring.modulus, xh)
        small += int((centered.abs() <= bound).sum())  # quantized activations satisfy |xq| <= qmax
        # the difference x_hat - xq is the mask: uniform top bits
        r = (xh - xq) % ring.modulus
        top = (r >> (ring.k - 4)).flatten()
        if top.numel() >= 4096:
            hist = torch.bincount(top, minlength=16).double()
            exp_n = top.numel() / 16
            assert float(((hist - exp_n) ** 2 / exp_n).sum()) < 80  # 15 dof, p ~ 1e-10
    # by chance each would happen with probability ~2^-K per value
    assert total > 10_000
    assert equal <= 2 and small <= 2, (equal, small, total)
    # no prompt text on the wire (requests the service received, incl. headers)
    needle = MESSAGES[0]["content"].encode()
    prompt_tokens_bytes = json.dumps(exp["prompt_ids"]).encode()
    for raw in _STATE["chat_raw"]:
        blob = raw.body + json.dumps(raw.headers).encode()
        assert needle not in blob and b"capital of France" not in blob and prompt_tokens_bytes not in blob
    # nor in the service's own logs
    assert "capital of France" not in e2e_stack.service.tail(1 << 30)


def test_outbound_traffic_passes_uniformity_monitor(e2e_stack: E2EStack, reference: Reference) -> None:
    from pgw.observability.noise import OutboundNoiseMonitor

    exp = _reference(reference)
    ring = e2e_stack.ring
    mon = OutboundNoiseMonitor(ring, sample_rate=1.0, fail_closed=True, window_values=1 << 14, seed=1)
    calls = exp["calls"]
    for it, (_, xq) in zip(_STATE["chat_items"], calls, strict=True):
        mon.guard(it.x_hat, xq)  # with the plaintext for the correlation test
    seen = {id(it) for it in _STATE["chat_items"]}
    for it in e2e_stack.all_items:  # everything else sent so far (the streaming request)
        if id(it) not in seen:
            mon.guard(it.x_hat)
    mon.flush()
    rep = mon.report()
    assert not rep.alarm, rep.reasons
    assert rep.windows >= 1 and rep.duplicate_rows == 0
    # positive control: the plaintext itself (what a broken masking path would send) alarms at once
    bad = OutboundNoiseMonitor(ring, sample_rate=1.0, fail_closed=True, window_values=1 << 14, seed=1)
    with pytest.raises(IntegrityError):
        for _, xq in calls:
            bad.guard(xq % ring.modulus, xq)
        bad.flush()
        bad.guard(calls[0][1] % ring.modulus)


# ---- 4. a malicious service ------------------------------------------------------------------------------
def test_tampered_response_is_detected_and_fails_closed(e2e_stack: E2EStack, reference: Reference) -> None:
    exp = _reference(reference)
    proxy = e2e_stack.proxy
    before = e2e_stack.available_slots()
    proxy.tamper_next = 1
    r = e2e_stack.chat({**CHAT, "logprobs": False, "top_logprobs": None})
    e2e_stack.capture()
    assert proxy.tampered == 1
    assert r.status_code == 502, r.text
    assert r.json()["error"]["code"] == "integrity_check_failed"
    # the failed request's masks are burned, never handed out again
    assert e2e_stack.available_slots() <= before - len(exp["prompt_ids"])
    # circuit breaker open: refused before reserving masks or sending anything
    n_items, avail = len(proxy.items), e2e_stack.available_slots()
    r = e2e_stack.chat(CHAT)
    assert r.status_code == 503 and r.json()["error"]["code"] == "unavailable", r.text
    assert len(proxy.items) == n_items and e2e_stack.available_slots() == avail
    ready = e2e_stack.admin("GET", "/readyz")
    assert ready.status_code == 503
    st = e2e_stack.admin("GET", "/admin/status").json()
    be = next(m for m in st["models"] if m["name"] == MODEL)["backend"]
    assert be["integrity_failures"] >= 1
    audit = [json.loads(line) for line in (e2e_stack.root / "state" / "audit.jsonl").read_text().splitlines()]
    assert any(a.get("event") == "inference" and a.get("status") == 502 for a in audit) or any(
        "integrity" in json.dumps(a) for a in audit)
    # admin reset after investigation; the stack serves exact answers again
    rr = e2e_stack.admin("POST", f"/admin/models/{MODEL}/reset", json={"breaker": True, "noise": False})
    assert rr.status_code == 200, rr.text
    r = e2e_stack.chat(CHAT)
    assert r.status_code == 200, r.text
    e2e_stack.capture()
    assert [c["logprob"] for c in r.json()["choices"][0]["logprobs"]["content"]] == exp["logprobs"]


# ---- 5. single use ------------------------------------------------------------------------------------------
def _row_digests(items: list[Any]) -> list[bytes]:
    out = []
    for it in items:
        x = it.x_hat.contiguous()
        for row in x:
            out.append(hashlib.blake2b(it.linear.encode() + row.numpy().tobytes(), digest_size=16).digest())
    return out


def test_masks_are_never_reused(e2e_stack: E2EStack, reference: Reference) -> None:
    from pgw.maskstore.admin import open_store
    from pgw.maskstore.config import MaskStoreSettings

    exp = _reference(reference)
    # a crash (SIGKILL) and restart of the gateway must not hand out consumed slots again
    e2e_stack.restart_gateway()
    r = e2e_stack.chat(CHAT)
    assert r.status_code == 200, r.text
    e2e_stack.capture()
    assert [c["logprob"] for c in r.json()["choices"][0]["logprobs"]["content"]] == exp["logprobs"]
    # identical prompts were sent 5 times; with one-time masks no outbound row ever repeats
    digests = _row_digests(e2e_stack.all_items)
    assert len(digests) > 0 and len(set(digests)) == len(digests)

    e2e_stack.gateway.stop()
    settings = MaskStoreSettings(root=e2e_stack.store_root, key_file=e2e_stack.key_file)
    snapshot = e2e_stack.root / "state-snapshot"
    with open_store(settings) as store:
        fp, ring = e2e_stack.fingerprint, e2e_stack.ring
        st = store.status()
        avail = store.available_slots(fp, ring)
        # exact accounting: every generated slot is either available or consumed (incl. burned)
        assert st["consumed_total"] + avail == e2e_stack.slots_generated
        assert st["open_reservations"] == 0
        assert st["rollback_protection"]["journal"] is True
        # keep a copy of the store database as it is now (for the rollback attempt below)
        snapshot.mkdir()
        dbs = [p for p in e2e_stack.store_root.iterdir() if p.suffix in (".sqlite", ".db") or ".sqlite" in p.name]
        assert dbs, sorted(p.name for p in e2e_stack.store_root.iterdir())
        for p in dbs:
            shutil.copy2(p, snapshot / p.name)
        # drain the store: afterwards it refuses, it never wraps around
        res = store.reserve(fp, ring, avail)
        res.close()
        with pytest.raises(MaskExhaustedError):
            store.reserve(fp, ring, 1)
        assert store.status()["consumed_total"] == e2e_stack.slots_generated
    # an attacker (or a backup restore) rolls the database back to make those slots available again
    for p in snapshot.iterdir():
        shutil.copy2(p, e2e_stack.store_root / p.name)
    for suffix in ("-wal", "-shm"):
        for p in e2e_stack.store_root.glob(f"*{suffix}"):
            p.unlink()
    with pytest.raises(PGWError) as ei:
        with open_store(settings) as store:
            store.reserve(e2e_stack.fingerprint, e2e_stack.ring, 1)
    assert ei.value.code.value in ("mask_reuse", "integrity_check_failed"), ei.value
