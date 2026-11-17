"""OpenAI wire compatibility: nulls, unsupported features, token-id prompts, logprob bytes, admission."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from pgw.gateway import ApiKeyStore, create_app
from pgw.gateway.openai_schemas import FINISH_REASONS, ChatCompletionRequest, CompletionRequest
from tests.gateway.conftest import MODEL, auth, gateway, plain_settings

CHAT = {"model": MODEL, "messages": [{"role": "user", "content": "Hello"}], "temperature": 0, "max_tokens": 4}
NULLABLE = ("temperature", "top_p", "n", "presence_penalty", "frequency_penalty", "stream", "logprobs",
            "top_logprobs", "seed", "top_k", "min_p", "max_tokens", "max_completion_tokens", "stop", "user",
            "stream_options", "repetition_penalty", "ignore_eos", "stop_token_ids", "tools", "tool_choice",
            "response_format", "functions", "function_call")


@pytest.fixture
def keys() -> ApiKeyStore:
    return ApiKeyStore(None)


@pytest.fixture
def key(keys: ApiKeyStore) -> str:
    return keys.create("app")[1]


def make_app(tiny_pack_dir: Path, tmp_path: Path, keys: ApiKeyStore, **kw: Any) -> Any:
    return create_app(plain_settings(tiny_pack_dir, tmp_path, **kw), key_store=keys)


# ---- schema level --------------------------------------------------------------------------------------
def test_every_optional_field_accepts_null() -> None:
    req = ChatCompletionRequest.model_validate({**CHAT, **dict.fromkeys(NULLABLE)})
    assert req.temperature == 1.0 and req.top_p == 1.0 and req.n == 1 and req.stream is False
    assert req.logprobs is False and req.presence_penalty == 0.0 and req.top_k == 0
    comp = CompletionRequest.model_validate({"model": MODEL, "prompt": "x", "echo": None, "suffix": None,
                                             **dict.fromkeys(NULLABLE)})
    assert comp.logprobs is None and comp.echo is False


@pytest.mark.parametrize(
    ("extra", "param"),
    [
        ({"tools": [{"type": "function", "function": {"name": "f"}}]}, "tools"),
        ({"functions": [{"name": "f"}]}, "functions"),
        ({"tool_choice": "required"}, "tool_choice"),
        ({"tool_choice": "auto"}, "tool_choice"),
        ({"function_call": "auto"}, "function_call"),
        ({"response_format": {"type": "json_schema", "json_schema": {"name": "x", "schema": {}}}},
         "response_format"),
        ({"response_format": {"type": "json_object"}}, "response_format"),
        ({"messages": [{"role": "user", "content": "hi"},
                       {"role": "assistant", "content": None,
                        "tool_calls": [{"id": "1", "type": "function",
                                        "function": {"name": "f", "arguments": "{}"}}]}]},
         "messages.1.tool_calls"),
    ],
)
async def test_unsupported_features_fail_fast(tiny_pack_dir: Path, tmp_path: Path, keys: ApiKeyStore, key: str,
                                              extra: dict[str, Any], param: str) -> None:
    async with gateway(make_app(tiny_pack_dir, tmp_path, keys)) as c:
        r = await c.post("/v1/chat/completions", json={**CHAT, **extra}, headers=auth(key))
        assert r.status_code == 400, r.text
        err = r.json()["error"]
        assert err["code"] == "protocol" and err["param"] == param


async def test_nulls_and_noop_tool_fields_are_accepted(tiny_pack_dir: Path, tmp_path: Path, keys: ApiKeyStore,
                                                       key: str) -> None:
    body = {**CHAT, **dict.fromkeys(NULLABLE), "max_tokens": 3, "temperature": 0,
            "tool_choice": "none", "tools": [], "response_format": {"type": "text"}}
    async with gateway(make_app(tiny_pack_dir, tmp_path, keys)) as c:
        r = await c.post("/v1/chat/completions", json=body, headers=auth(key))
        assert r.status_code == 200, r.text
        assert r.json()["choices"][0]["finish_reason"] in FINISH_REASONS
        r = await c.post("/v1/completions", json={"model": MODEL, "prompt": "ab", "max_tokens": 2, "echo": None,
                                                  "logprobs": None, "temperature": None}, headers=auth(key))
        assert r.status_code == 200, r.text


def test_finish_reasons_are_openai_values() -> None:
    assert FINISH_REASONS == ("stop", "length")
    doc = (Path(__file__).resolve().parents[2] / "docs/api/gateway-api.md").read_text(encoding="utf-8")
    assert "`cancelled`" not in doc and "`finish_reason` is `stop` or `length`" in doc


# ---- token-id prompts ----------------------------------------------------------------------------------
async def test_token_id_prompts(tiny_pack_dir: Path, tmp_path: Path, keys: ApiKeyStore, key: str) -> None:
    base = {"model": MODEL, "max_tokens": 4, "temperature": 0, "ignore_eos": True}
    async with gateway(make_app(tiny_pack_dir, tmp_path, keys)) as c:
        text = (await c.post("/v1/completions", json={**base, "prompt": "abc"}, headers=auth(key))).json()
        ids = [97, 98, 99]  # byte-level tiny tokenizer: id == byte
        r = await c.post("/v1/completions", json={**base, "prompt": ids, "echo": True}, headers=auth(key))
        assert r.status_code == 200, r.text
        assert r.json()["choices"][0]["text"] == "abc" + text["choices"][0]["text"]
        r = await c.post("/v1/completions", json={**base, "prompt": [ids, [97]]}, headers=auth(key))
        assert r.status_code == 200 and len(r.json()["choices"]) == 2
        assert r.json()["choices"][0]["text"] == text["choices"][0]["text"]
        for bad in ([97, 100_000], [-1], [[]]):
            r = await c.post("/v1/completions", json={**base, "prompt": bad}, headers=auth(key))
            assert r.status_code == 400 and r.json()["error"]["code"] == "protocol", bad


# ---- logprob bytes -------------------------------------------------------------------------------------
async def test_logprob_bytes_are_raw_token_bytes(tiny_pack_dir: Path, tmp_path: Path, keys: ApiKeyStore,
                                                key: str) -> None:
    app = make_app(tiny_pack_dir, tmp_path, keys)
    async with gateway(app) as c:
        rt = app.state.pgw.runtime.get(MODEL)
        assert rt.tokenizer.token_bytes(0xC3) == b"\xc3"  # half of a 2-byte UTF-8 character
        r = await c.post("/v1/chat/completions", json={**CHAT, "logprobs": True, "top_logprobs": 5,
                                                       "max_tokens": 8, "ignore_eos": True}, headers=auth(key))
        assert r.status_code == 200, r.text
        for item in r.json()["choices"][0]["logprobs"]["content"]:
            raw = bytes(item["bytes"])
            assert raw == rt.tokenizer.token_bytes(_id_of(rt, raw))
            assert item["token"] == raw.decode("utf-8", "replace")
            for top in item["top_logprobs"]:
                assert top["token"] == bytes(top["bytes"]).decode("utf-8", "replace")


def _id_of(rt: Any, raw: bytes) -> int:
    if len(raw) == 1:
        return raw[0]
    return rt.tokenizer.hf.convert_tokens_to_ids(raw.decode())


def test_token_bytes_qwen_style_split_characters() -> None:
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    from transformers.convert_slow_tokenizer import bytes_to_unicode

    from pgw.engine.tokenizer import ChatTokenizer

    b2u = bytes_to_unicode()
    vocab = {b2u[b]: b for b in range(256)}
    euro = "".join(b2u[b] for b in "€".encode())  # 3-byte character as one token
    vocab[euro] = 256
    vocab["".join(b2u[b] for b in "€".encode()[:2])] = 257  # a token holding a partial character
    tok = Tokenizer(models.BPE(vocab=vocab, merges=[], ignore_merges=True))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    tok.decoder = decoders.ByteLevel()
    ct = ChatTokenizer(None, tokenizer=PreTrainedTokenizerFast(tokenizer_object=tok))
    assert ct.token_bytes(256) == "€".encode()
    assert ct.token_bytes(257) == "€".encode()[:2]
    assert ct.token_bytes(ord("a")) == b"a"


# ---- admission -----------------------------------------------------------------------------------------
async def test_more_prompts_than_sessions_is_rejected(tiny_pack_dir: Path, tmp_path: Path, keys: ApiKeyStore,
                                                      key: str) -> None:
    app = make_app(tiny_pack_dir, tmp_path, keys, model_kw={"max_concurrent_sessions": 2})
    async with gateway(app) as c:
        body = {"model": MODEL, "prompt": ["a", "b", "c"], "max_tokens": 2}
        r = await c.post("/v1/completions", json=body, headers=auth(key))
        assert r.status_code == 400 and r.json()["error"]["code"] == "protocol"
        rt = app.state.pgw.runtime.get(MODEL)
        assert rt.admission.active == 0
        r = await c.post("/v1/completions", json={**body, "prompt": ["a", "b"]}, headers=auth(key))
        assert r.status_code == 200 and rt.admission.active == 0
