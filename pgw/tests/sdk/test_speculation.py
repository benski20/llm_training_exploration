"""Strict speculative decoding in the client library (``speculation``, opt-in): exact and draft-independent.

Against the real ``pgw service`` process of the SDK tests (TLS, TCP+TLS trip lane, masked LM head, accounts), the
in-process gateway with ``speculation = "lookup"`` (prompt lookup, no weights) or ``"draft_head"`` (a target-feature
draft head file):

* answers (greedy, seeded sampling, streaming) are bit-identical to the plaintext reference of the gateway path, and
  stay so whether every draft is right (an oracle drafter) or none is proposed;
* the provider's view does not depend on the drafts: the provider-metered trips and rows of the same request are
  equal with all drafts accepted and with none, every pass carries ``spec_rows`` rows, the provider sees one pass
  per generated token (dummy passes after the answer), and the prime-field (p31) check stays on;
* closing the client right after an answer still runs every dummy pass that is due.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tests.sdk.conftest import MODEL, Provider
from tests.sdk.test_e2e_cpu import GREEDY, PROMPT, SAMPLED, _reference

pytestmark = pytest.mark.timeout(1200)

ROWS = 3


def _client(provider: Provider, key: str, root: Path, **kw: Any) -> Any:
    from pgw import OpenAI
    from pgw.sdk import LocalPackProvisioner

    opts: dict[str, Any] = {"cache_dir": root / "cache", "state_dir": root / "state", "mask_scheme": "plain",
                            "mask_workers": 1, "mask_target_slots": 600, "mask_low_water_slots": 300,
                            "mask_segment_slots": 128, "timeout": 300, "ready_timeout_s": 600,
                            "speculation": "lookup", "spec_rows": ROWS}
    opts.update(kw)
    return OpenAI(api_key=key, provisioner=LocalPackProvisioner({MODEL: provider.pack_dir}), **opts)


def _runtime(client: Any) -> Any:
    client.warmup(MODEL)
    return client._engine.models[MODEL].runtime


def _drain(client: Any) -> None:
    client._engine.call(_runtime(client).generator.wait_drained(), timeout_s=300)


def _usage(provider: Provider, settle: float = 1.5) -> tuple[int, ...]:
    """Provider-metered (trips, rows, bytes in, bytes out) of the test organisation (all keys), once the lane
    counters have drained."""
    time.sleep(settle)
    last: tuple[int, ...] | None = None
    for _ in range(20):
        rows = provider.admin("GET", f"/orgs/{provider.org}/usage")["usage"]
        cur = tuple(sum(int(r.get(k, 0) or 0) for r in rows) for k in ("trips", "rows", "bytes_in", "bytes_out"))
        if cur == last:
            return cur
        last = cur
        time.sleep(0.6)
    assert last is not None
    return last


@pytest.fixture(scope="module")
def spec(provider: Provider, tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, Any]]:
    root = tmp_path_factory.mktemp("sdk-spec")
    key = provider.new_key("spec")
    client = _client(provider, key, root)
    st: dict[str, Any] = {"root": root, "key": key, "client": client}
    try:
        yield st
    finally:
        st["client"].close()


def test_options() -> None:
    from pgw.sdk.options import SDKOptions

    assert SDKOptions().speculation == "off"  # opt-in
    with pytest.raises(ValueError, match="speculation"):
        SDKOptions(speculation="fast").validate()  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="spec_rows"):
        SDKOptions(speculation="lookup", spec_rows=9).validate()
    SDKOptions(speculation="draft_head").validate()  # the head pinned for the model (pgw.sdk.draft_heads)
    with pytest.raises(ValueError, match="draft_head_path"):
        SDKOptions(draft_head_path=Path("/h.pt")).validate()
    SDKOptions(speculation="lookup", spec_rows=4).validate()


def test_model_config_is_strict_fixed_rows() -> None:
    """The gateway settings the option maps to: strict passes, fixed rows, masked filler rows (no padding, so the
    p31 check stays valid), p31 requested explicitly; nothing when off."""
    from pgw.sdk.engine import Engine
    from pgw.sdk.options import SDKOptions

    off = SimpleNamespace(options=SDKOptions())
    assert Engine._spec_config(off) == {}  # type: ignore[arg-type]
    assert Engine._prime_check(off) is None  # type: ignore[arg-type]
    on = SimpleNamespace(options=SDKOptions(speculation="lookup", spec_rows=4))
    kw = Engine._spec_config(on)  # type: ignore[arg-type]
    assert kw["spec_rows"] == 4 and kw["spec_strict_passes"] is True and kw["spec_tokens"] == 3
    assert kw["spec_lookup"] > 0 and "spec_mask_rows" not in kw
    assert Engine._prime_check(on) is True  # type: ignore[arg-type]
    head = SimpleNamespace(options=SDKOptions(speculation="draft_head", draft_head_path=Path("/h.pt")))
    kw = Engine._spec_config(head)  # type: ignore[arg-type]
    assert kw["draft_backend"] == "feature_head" and kw["draft_head_path"] == Path("/h.pt")
    assert Engine._prime_check(SimpleNamespace(options=SDKOptions(speculation="lookup", native=False))) is None  # type: ignore[arg-type]


def test_lookup_answers_are_bit_identical(spec: dict[str, Any], provider: Provider) -> None:
    c = spec["client"]
    for body in (GREEDY, SAMPLED):
        r = c.chat.completions.create(model=MODEL, messages=PROMPT, **body)
        ref = _reference(provider.pack_dir, body)
        assert r.choices[0].message.content == ref["choices"][0]["message"]["content"]
        assert r.to_dict()["choices"][0]["logprobs"]["content"] == ref["choices"][0]["logprobs"]["content"]
        chunks = list(c.chat.completions.create(model=MODEL, messages=PROMPT, stream=True, **body))
        assert "".join(ch.choices[0].delta.content or "" for ch in chunks if ch.choices) == \
            ref["choices"][0]["message"]["content"]
    st = c.status()["models"][MODEL]
    assert st["native"]["prime_check"] is True and st["native"]["decode_native"] is True
    speed = _runtime(c).status()["speed"]
    assert speed["spec_rows"] == ROWS and speed["spec_strict_passes"] is True


def _set_drafter(client: Any, oracle: list[int] | None, seen: list[list[int]] | None = None) -> None:
    """Replace the prompt-lookup proposals: ``oracle`` (propose the continuation of that sequence), or none."""
    drafter = _runtime(client).generator._drafter

    def lookup(context: list[int], k: int) -> list[int]:
        if seen is not None:
            seen.append(list(context))
        if oracle is None:
            return []
        n = len(context)
        return list(oracle[n:n + k]) if oracle[:n] == list(context) else []

    drafter._lookup = lookup


def test_provider_view_does_not_depend_on_the_drafts(spec: dict[str, Any], provider: Provider) -> None:
    """The same request with every draft right (oracle) and with no draft at all: identical answers, identical
    provider-metered trips and rows, and one provider pass per generated token in both."""
    c = spec["client"]
    rt = _runtime(c)
    gen = rt.generator
    body = {**GREEDY, "max_tokens": 12}
    ref = _reference(provider.pack_dir, body)
    seen: list[list[int]] = []
    _set_drafter(c, None, seen)
    try:
        _drain(c)
        u0, s0 = _usage(provider), dict(gen.spec_stats)
        r_none = c.chat.completions.create(model=MODEL, messages=PROMPT, **body)
        _drain(c)
        u1, s1 = _usage(provider), dict(gen.spec_stats)
        # the real context of the last real pass: prompt + the answer (the dummy passes feed random tokens after it)
        oracle = max((x for x in seen), key=len)
        _set_drafter(c, oracle)
        r_all = c.chat.completions.create(model=MODEL, messages=PROMPT, **body)
        _drain(c)
        u2, s2 = _usage(provider), dict(gen.spec_stats)
    finally:
        _set_drafter(c, None)
    for r in (r_none, r_all):
        assert r.choices[0].message.content == ref["choices"][0]["message"]["content"]
        assert r.to_dict()["choices"][0]["logprobs"]["content"] == ref["choices"][0]["logprobs"]["content"]
    acc_none = s1["spec_accepted"] - s0["spec_accepted"]
    acc_all = s2["spec_accepted"] - s1["spec_accepted"]
    assert acc_none == 0 and acc_all >= 4, (acc_none, acc_all)  # the drafts did differ
    passes_none = s1["spec_passes"] - s0["spec_passes"]
    passes_all = s2["spec_passes"] - s1["spec_passes"]
    assert passes_all < passes_none  # fewer real passes on the critical path ...
    dummy = (s1["dummy_passes"] - s0["dummy_passes"], s2["dummy_passes"] - s1["dummy_passes"])
    assert passes_none + dummy[0] == passes_all + dummy[1] == body["max_tokens"] - 1  # ... one pass per token
    view_none = tuple(b - a for a, b in zip(u0, u1, strict=True))
    view_all = tuple(b - a for a, b in zip(u1, u2, strict=True))
    assert view_none == view_all and view_none[0] > 0, (view_none, view_all)  # trips, rows, bytes in and out


def test_close_runs_the_dummy_passes(spec: dict[str, Any], provider: Provider) -> None:
    """An application that closes the client right after an answer: the dummy passes still due run before the
    models stop, so the provider sees the same trips and rows as with an explicit drain."""
    c = spec["client"]
    body = {**GREEDY, "max_tokens": 12}
    _set_drafter(c, None)
    c.chat.completions.create(model=MODEL, messages=PROMPT, **body)
    _drain(c)
    u0 = _usage(provider)
    c.chat.completions.create(model=MODEL, messages=PROMPT, **body)
    _drain(c)
    u1 = _usage(provider)
    seen: list[list[int]] = []
    _set_drafter(c, None, seen)
    c.chat.completions.create(model=MODEL, messages=PROMPT, **body)
    _set_drafter(c, max(seen, key=len))
    gen = _runtime(c).generator
    s0 = dict(gen.spec_stats)
    c.chat.completions.create(model=MODEL, messages=PROMPT, **body)  # accepted drafts: dummy passes still due
    s1 = dict(gen.spec_stats)
    due = (body["max_tokens"] - 1) - (s1["spec_passes"] - s0["spec_passes"]) - (s1["dummy_passes"] - s0["dummy_passes"])
    c.close()  # no drain: close() must run them
    assert s1["spec_accepted"] > s0["spec_accepted"] and due > 0, due  # the case is exercised
    u2 = _usage(provider)
    spec["client"] = _client(provider, spec["key"], spec["root"])
    one = tuple(b - a for a, b in zip(u0, u1, strict=True))
    assert tuple(b - a for a, b in zip(u1, u2, strict=True)) == tuple(2 * x for x in one) and one[0] > 0


def test_draft_head(spec: dict[str, Any], provider: Provider, tmp_path: Path) -> None:
    """``speculation = "draft_head"`` with a head file in the trained format: bit-identical answers."""
    from tests.engine.test_tree_spec import _head_file

    head = tmp_path / "head.pt"
    _head_file(_runtime(spec["client"]).model, head)
    c = _client(provider, provider.new_key("spec-head"), tmp_path / "client", speculation="draft_head",
                draft_head_path=head)
    try:
        for body in (GREEDY, SAMPLED):
            r = c.chat.completions.create(model=MODEL, messages=PROMPT, **body)
            ref = _reference(provider.pack_dir, body)
            assert r.choices[0].message.content == ref["choices"][0]["message"]["content"]
            assert r.to_dict()["choices"][0]["logprobs"]["content"] == ref["choices"][0]["logprobs"]["content"]
        speed = _runtime(c).status()["speed"]
        assert speed["spec_rows"] == ROWS and speed["drafter_stats"]["steps"] > 0
    finally:
        c.close()
