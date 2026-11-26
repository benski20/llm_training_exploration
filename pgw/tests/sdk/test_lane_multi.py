"""Concurrent answers on the native trip lane (lane format 2, ``multi`` calls; on in the client library).

Against the real ``pgw service`` process of the SDK tests (TLS, TCP+TLS trip lane, masked LM head, accounts), tiny
model, plain masks:

* several answers decoding at once share each batched trip in one multi lane call: every answer is bit-identical to
  the same request run alone, under different interleavings (start order, staggered starts, streamed and not), and
  identical to the answers with multi calls off (every several-item trip on the stream channel, as before);
* strict speculation: answers identical with multi calls on and off, alone and concurrent, and the provider-metered
  trips and rows of a request do not change with the channel;
* a provider without lane format 2 (``PGW_SERVICE_LANE_MULTI=0``, a format-1 ticket): the client never sends a
  multi frame, several-item trips take the stream channel, answers identical;
* a provider that returns a wrong result inside a multi call is caught by the answer's own checks.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from tests.sdk.conftest import MODEL, Provider, start_provider, stop_provider

pytestmark = pytest.mark.timeout(1500)

PROMPTS = ["Name a colour.", "Count to three.", "Say hello politely."]


def _body(i: int, **kw: Any) -> dict[str, Any]:
    b = {"model": MODEL, "messages": [{"role": "user", "content": PROMPTS[i % 3]}], "temperature": 0.8,
         "seed": 100 + i % 3, "max_tokens": 20, "logprobs": True, "user": f"user-{i}",
         "extra_body": {"ignore_eos": True}}
    b.update(kw)
    return b


def _lp(r: Any) -> list[list[Any]]:
    return [[c.token, c.logprob] for c in r.choices[0].logprobs.content]


def _run(c: Any, i: int, stream: bool = False, delay: float = 0.0) -> list[list[Any]]:
    if delay:
        time.sleep(delay)
    if not stream:
        return _lp(c.chat.completions.create(**_body(i)))
    out: list[list[Any]] = []
    for ch in c.chat.completions.create(**_body(i), stream=True):
        if ch.choices and ch.choices[0].logprobs:
            out += [[x.token, x.logprob] for x in ch.choices[0].logprobs.content]
    return out


def _client(provider: Provider, key: str, root: Path, **kw: Any) -> Any:
    from pgw import OpenAI
    from pgw.sdk import LocalPackProvisioner

    opts: dict[str, Any] = {"cache_dir": root / "cache", "state_dir": root / "state", "mask_scheme": "plain",
                            "mask_workers": 1, "mask_target_slots": 400, "mask_low_water_slots": 200,
                            "mask_segment_slots": 64, "timeout": 600, "ready_timeout_s": 600, "max_concurrent": 8}
    opts.update(kw)
    return OpenAI(api_key=key, provisioner=LocalPackProvisioner({MODEL: provider.pack_dir}), **opts)


def _lane(c: Any) -> Any:
    c.warmup(MODEL)
    return c._engine.models[MODEL].runtime.client


def _stats(c: Any) -> dict[str, Any]:
    ls = _lane(c).lane_stats
    return {"trips": ls.trips, "multi_trips": ls.multi_trips, "multi_items": ls.multi_items,
            "reasons": dict(ls.fallback_reasons)}


@pytest.fixture
def multi_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    """Set ``PGW_LANE_MULTI`` for the clients built next (an explicit value wins over the library's default)."""
    def set_(on: bool) -> None:
        monkeypatch.setenv("PGW_LANE_MULTI", "1" if on else "0")

    yield set_


def _concurrent(c: Any, n: int, *, reverse: bool = False, stream: bool = False, stagger: bool = False
                ) -> dict[int, list[list[Any]]]:
    order = list(range(n))[::-1] if reverse else list(range(n))
    with ThreadPoolExecutor(n) as ex:
        futs = {i: ex.submit(_run, c, i, stream, ((i * 7) % 5) * 0.03 if stagger else 0.0) for i in order}
        return {i: f.result() for i, f in futs.items()}


def test_concurrent_answers_share_multi_lane_calls_and_stay_exact(provider: Provider, tmp_path: Path,
                                                                  multi_env: Any) -> None:
    key = provider.new_key("lane-multi")
    multi_env(False)  # first the previous behaviour: several-item trips on the stream channel
    c = _client(provider, key, tmp_path)
    try:
        refs = [_run(c, j) for j in range(3)]  # each request alone
        off = _concurrent(c, 6)
        s_off = _stats(c)
    finally:
        c.close()
    assert all(o == refs[i % 3] for i, o in off.items())
    assert s_off["multi_trips"] == 0 and s_off["reasons"].get("ineligible:items", 0) > 0
    multi_env(True)
    c = _client(provider, key, tmp_path)
    try:
        assert [_run(c, j) for j in range(3)] == refs  # alone, multi on: one answer per trip, same bytes
        s0 = _stats(c)
        runs = [_concurrent(c, 6), _concurrent(c, 6, reverse=True, stagger=True), _concurrent(c, 5, stream=True),
                _concurrent(c, 8, stagger=True, stream=True)]
        s1 = _stats(c)
        st = c.status()["models"][MODEL]
    finally:
        c.close()
    for outs in runs:
        assert all(o == refs[i % 3] for i, o in outs.items()), "an answer changed under concurrency"
    assert s1["multi_trips"] - s0["multi_trips"] > 50
    assert (s1["multi_items"] - s0["multi_items"]) / (s1["multi_trips"] - s0["multi_trips"]) > 1.5  # shared trips
    assert s1["reasons"].get("ineligible:items", 0) == s0["reasons"].get("ineligible:items", 0) == 0
    assert st["integrity_failures"] == 0


def test_strict_speculation_is_unchanged_by_multi_calls(provider: Provider, tmp_path: Path, multi_env: Any) -> None:
    from tests.sdk.test_speculation import _usage

    key = provider.new_key("lane-multi-spec")
    spec = {"speculation": "lookup", "spec_rows": 3}
    res: dict[bool, Any] = {}
    for on in (False, True):
        multi_env(on)
        c = _client(provider, key, tmp_path, **spec)
        try:
            _run(c, 0)  # warm
            c._engine.call(c._engine.models[MODEL].runtime.generator.wait_drained(), timeout_s=300)
            u0 = _usage(provider)
            alone = _run(c, 1)
            c._engine.call(c._engine.models[MODEL].runtime.generator.wait_drained(), timeout_s=300)
            u1 = _usage(provider)
            conc = _concurrent(c, 3)
            res[on] = (alone, conc, tuple(b - a for a, b in zip(u0, u1, strict=True))[:2], _stats(c))
        finally:
            c.close()
    assert res[True][0] == res[False][0]
    assert res[True][1] == res[False][1]
    assert res[True][2] == res[False][2] and res[True][2][0] > 0  # provider-metered trips and rows of one request
    assert res[True][3]["multi_trips"] > 0


def test_format_1_provider_gets_no_multi_frames(sdk_pack_dir: Path, pki: dict[str, Any], tmp_path: Path,
                                                multi_env: Any) -> None:
    (tmp_path / "old-provider").mkdir()
    prov = start_provider(tmp_path / "old-provider", sdk_pack_dir, pki, extra_env={"PGW_SERVICE_LANE_MULTI": "0"})
    try:
        multi_env(True)
        c = _client(prov, prov.new_key("old"), tmp_path / "client")
        try:
            lane = _lane(c)
            refs = [_run(c, j) for j in range(3)]
            outs = _concurrent(c, 4)
            assert lane.lane_ticket is not None and lane.lane_ticket.max_items == 0
            s = _stats(c)
        finally:
            c.close()
    finally:
        stop_provider(prov)
    assert all(o == refs[i % 3] for i, o in outs.items())
    assert s["multi_trips"] == 0 and s["reasons"].get("ineligible:items", 0) > 0 and s["trips"] > 0


def test_wrong_result_inside_a_multi_call_is_caught(provider: Provider, tmp_path: Path, multi_env: Any) -> None:
    from pgw.sdk import IntegrityError

    multi_env(True)
    c = _client(provider, provider.new_key("lane-multi-tamper"), tmp_path)
    try:
        refs = [_run(c, j) for j in range(3)]
        s0 = _stats(c)
        provider.tamper_file.touch()
        try:
            with ThreadPoolExecutor(3) as ex:
                futs = [ex.submit(_run, c, i) for i in range(3)]
                errs = [f.exception() for f in futs]
                done = {i: f.result() for i, f in enumerate(futs) if f.exception() is None}
        finally:
            provider.tamper_file.unlink()
        s1 = _stats(c)
        st = c.status()["models"][MODEL]
    finally:
        c.close()
    # the answer whose rows were changed fails its own checks; any answer that completes is the exact one
    assert any(isinstance(e, IntegrityError) for e in errs), errs
    assert all(e is None or isinstance(e, IntegrityError) for e in errs), errs
    assert all(o == refs[i] for i, o in done.items())
    assert st["integrity_failures"] >= 1
    assert s1["multi_trips"] > s0["multi_trips"]  # the wrong results came back in multi calls
    assert os.environ.get("PGW_LANE_MULTI") == "1"
