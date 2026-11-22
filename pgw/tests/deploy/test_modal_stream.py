"""Modal e2e decode-rate metric (``wizard_client.chat_stream``) and the co-located A/B helpers
(``pgw_modal_boot.ab_phases`` / ``thread_deltas``); no Modal, no network."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import httpx
import pytest

MODAL_DIR = Path(__file__).resolve().parents[2] / "deploy" / "modal"


def _load(name: str):  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location(name + "_t", MODAL_DIR / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_chat_stream_splits_ttft_from_decode_rate(monkeypatch: pytest.MonkeyPatch) -> None:
    wiz = _load("wizard_client")
    chunks = [{"choices": [{"delta": {"role": "assistant", "content": ""}, "logprobs": None}]}]
    for i, tok in enumerate(["A", "B", "C", "D"]):
        chunks.append({"choices": [{"delta": {"content": tok},
                                    "logprobs": {"content": [{"token": tok, "logprob": -0.5 * i}]}}]})
    body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    real = httpx.Client
    monkeypatch.setattr(wiz.httpx, "Client", lambda **kw: real(transport=httpx.MockTransport(handler),
                                                                 base_url=kw["base_url"]))
    ticks = iter([0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    monkeypatch.setattr(wiz, "time", type("FakeTime", (), {"time": staticmethod(lambda: next(ticks))}))
    out = wiz.chat_stream("http://gw", "k", "m", [{"role": "user", "content": "x"}], 4)
    assert seen["body"]["stream"] is True and seen["body"]["temperature"] == 0
    assert out["text"] == "ABCD" and out["logprobs"] == [0.0, -0.5, -1.0, -1.5]
    assert out["tokens"] == 4 and out["ttft_s"] == 0.5
    assert out["decode_ms_per_token"] == pytest.approx(100.0) and out["decode_tokens_per_s"] == pytest.approx(10.0)


def test_ab_phase_parsing_and_thread_deltas() -> None:
    boot = _load("pgw_modal_boot")
    assert [p for p, _ in boot.ab_phases("base, pin,bogus,nofactory,factory,unpin")] == [
        "base", "pin", "nofactory", "factory", "unpin"]
    assert boot.ab_phases("") == []
    cpus = [c for g in boot.PIN_LAYOUT.values() for c in (g or set())]
    assert len(cpus) == len(set(cpus))  # disjoint cores per party
    a = {"1/1": ("python", 1.0), "1/2": ("pool", 0.0)}
    b = {"1/1": ("python", 1.5), "1/2": ("pool", 2.0), "1/3": ("new", 0.1)}
    assert boot.thread_deltas(a, b, 2.0, top=2) == [("1/2", "pool", 1.0), ("1/1", "python", 0.25)]
