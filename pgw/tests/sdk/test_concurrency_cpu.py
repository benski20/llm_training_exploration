"""Client CPU at many concurrent answers (docs/sdk/scaling.md): the library's three switches for it change no byte.

Against the real ``pgw service`` of the SDK tests (tiny model, plain masks), with ``PGW_SYNC_MULTI`` (a decode step of
several answers as one synchronous pass on the plaintext thread), ``PGW_TRIP_NATIVE_MULTI`` (one native lease per
answer in a shared trip, prepared and finished in one native call) and ``PGW_SAMPLE_THREADS`` (the answers of a step
sample on a thread pool) on, as the library sets them, and all off (the previous path):

* every answer, alone and under concurrency (several interleavings, streamed or not), equals the same request alone
  with the switches off: same tokens and logprobs;
* the provider sees the same trips and rows for a request alone, with and without strict speculation;
* the new paths actually ran (synchronous several-answer passes, leased several-answer trips).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tests.sdk.conftest import MODEL, Provider
from tests.sdk.test_lane_multi import _client, _concurrent, _run

pytestmark = pytest.mark.timeout(1500)

SWITCHES = ("PGW_SYNC_MULTI", "PGW_TRIP_NATIVE_MULTI", "PGW_SAMPLE_THREADS")


def _set(monkeypatch: pytest.MonkeyPatch, on: bool) -> None:
    for k in SWITCHES:
        if on:
            monkeypatch.delenv(k, raising=False)  # the library's own setting
        else:
            monkeypatch.setenv(k, "0")


def test_switches_change_no_answer_and_no_trip(provider: Provider, tmp_path: Path,
                                               monkeypatch: pytest.MonkeyPatch) -> None:
    from pgw.engine.backends import MaskedBackend
    from pgw.engine.model import TransformerModel
    from tests.sdk.test_speculation import _usage

    used = {"sync_multi": 0, "leases_many": 0}
    real_fb = TransformerModel.forward_batch
    real_lm = MaskedBackend._prepare_trip_leases

    def fb(self: Any, seqs: Any) -> Any:
        if len(seqs) > 1 and sum(len(s.input_ids) for s in seqs) > 8:
            used["sync_multi"] += 1
        return real_fb(self, seqs)

    def lm(self: Any, calls: Any) -> Any:
        got = real_lm(self, calls)
        used["leases_many"] += got is not None
        return got

    monkeypatch.setattr(TransformerModel, "forward_batch", fb)
    monkeypatch.setattr(MaskedBackend, "_prepare_trip_leases", lm)
    key = provider.new_key("conc-cpu")
    res: dict[bool, Any] = {}
    for on in (False, True):
        _set(monkeypatch, on)
        used.update(sync_multi=0, leases_many=0)
        # a stock for 12 answers at once (each about 20 prompt + 20 answer slots): steps of more than 8 rows
        c = _client(provider, key, tmp_path, max_concurrent=12, mask_target_slots=1200, mask_low_water_slots=900)
        try:
            c.warmup(MODEL)
            be = c._engine.models[MODEL].runtime.backend
            assert be.sync_multi == on and be.trip_native_multi == on
            c._engine.call(c._engine.models[MODEL].runtime.generator.wait_drained(), timeout_s=300)
            u0 = _usage(provider)
            alone = [_run(c, j) for j in range(3)]
            u1 = _usage(provider)
            runs = [_concurrent(c, 12), _concurrent(c, 10, reverse=True, stagger=True, stream=True)]
            st = c.status()["models"][MODEL]
        finally:
            c.close()
        res[on] = (alone, runs, tuple(b - a for a, b in zip(u0, u1, strict=True))[:2], dict(used))
        assert st["integrity_failures"] == 0
    alone_off = res[False][0]
    for on in (False, True):
        alone, runs, _, _ = res[on]
        assert alone == alone_off
        for outs in runs:
            assert all(o == alone_off[i % 3] for i, o in outs.items()), "an answer changed under concurrency"
    assert res[True][2] == res[False][2] and res[True][2][0] > 0  # provider-metered trips and rows, alone
    assert res[False][3] == {"sync_multi": 0, "leases_many": 0}
    assert res[True][3]["sync_multi"] > 0 and res[True][3]["leases_many"] > 0, res[True][3]


def test_strict_speculation_unchanged(provider: Provider, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.sdk.test_speculation import _usage

    key = provider.new_key("conc-cpu-spec")
    spec = {"speculation": "lookup", "spec_rows": 3}
    res: dict[bool, Any] = {}
    for on in (False, True):
        _set(monkeypatch, on)
        c = _client(provider, key, tmp_path, **spec)
        try:
            _run(c, 0)  # warm
            c._engine.call(c._engine.models[MODEL].runtime.generator.wait_drained(), timeout_s=300)
            u0 = _usage(provider)
            alone = _run(c, 1)
            c._engine.call(c._engine.models[MODEL].runtime.generator.wait_drained(), timeout_s=300)
            u1 = _usage(provider)
            conc = _concurrent(c, 4)
            res[on] = (alone, conc, tuple(b - a for a, b in zip(u0, u1, strict=True))[:2])
        finally:
            c.close()
    assert res[True][0] == res[False][0]
    assert res[True][1] == res[False][1]
    assert res[True][2] == res[False][2] and res[True][2][0] > 0
