"""A FastAPI service that answers through PGW.

Two routes:

* ``POST /ask``                 ``{"question": "..."}`` -> ``{"answer": "..."}``: your own API, as an app would have.
* ``POST /v1/chat/completions`` OpenAI-compatible (streaming and non-streaming), so tools that speak the
  OpenAI HTTP API (LangChain ``ChatOpenAI``, LlamaIndex ``OpenAILike``, curl, ...) can use PGW by pointing
  their ``base_url`` at this app.

One ``AsyncOpenAI`` client is created at start-up and shared by all requests: the in-process gateway
batches concurrent requests. Run ONE worker process per machine (each worker holds its own gateway).

    pip install fastapi uvicorn          # not dependencies of pgw-sdk
    export PGW_API_KEY=pgw_...
    export APP_TOKEN=change-me           # optional: require "Authorization: Bearer change-me"
    uvicorn examples.fastapi_app:app --host 127.0.0.1 --port 8000 --workers 1

The app listens on localhost by default. It exposes plaintext prompts and answers to whoever can reach
it: put it behind your usual authentication before binding it to another interface.
"""

from __future__ import annotations

import hmac
import json
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from pgw import AsyncOpenAI

MODEL = os.environ.get("PGW_EXAMPLE_MODEL", "qwen2.5-7b-instruct")
APP_TOKEN = os.environ.get("APP_TOKEN", "")


def _plain(obj: Any) -> Any:
    """Response objects (pydantic models in the OpenAI style) or dicts -> JSON-ready data."""
    return obj.model_dump(exclude_none=True) if hasattr(obj, "model_dump") else obj


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.llm = AsyncOpenAI()  # reads PGW_API_KEY; enrolls and prepares the model on first use
    yield
    close = getattr(app.state.llm, "close", None)
    if close is not None:
        res = close()
        if hasattr(res, "__await__"):
            await res


app = FastAPI(title="PGW example app", lifespan=lifespan)


def require_token(authorization: str = Header(default="")) -> None:
    if APP_TOKEN and not hmac.compare_digest(authorization, f"Bearer {APP_TOKEN}"):
        raise HTTPException(status_code=401, detail="missing or wrong bearer token")


class Ask(BaseModel):
    question: str
    max_tokens: int = 256


@app.post("/ask", dependencies=[Depends(require_token)])
async def ask(body: Ask) -> dict[str, str]:
    resp = await app.state.llm.chat.completions.create(
        model=MODEL, messages=[{"role": "user", "content": body.question}], max_tokens=body.max_tokens
    )
    return {"answer": resp.choices[0].message.content or ""}


@app.post("/v1/chat/completions", dependencies=[Depends(require_token)], response_model=None)
async def chat_completions(body: dict[str, Any]) -> Any:
    """Pass an OpenAI chat-completions request through to PGW (same fields, same response shape)."""
    body = dict(body)
    body.setdefault("model", MODEL)
    if not body.get("stream"):
        return _plain(await app.state.llm.chat.completions.create(**body))

    async def sse() -> AsyncIterator[str]:
        stream = await app.state.llm.chat.completions.create(**body)
        async for chunk in stream:
            yield f"data: {json.dumps(_plain(chunk))}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(sse(), media_type="text/event-stream")


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok", "model": MODEL}
