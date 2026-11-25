"""Async client: several conversations at once, plus async streaming.

Concurrent requests share the in-process gateway, which batches their steps: running them together
is faster than running them one after another.

    export PGW_API_KEY=pgw_...
    python examples/async_chat.py
"""

from __future__ import annotations

import asyncio
import os

from pgw import AsyncOpenAI

MODEL = os.environ.get("PGW_EXAMPLE_MODEL", "qwen2.5-7b-instruct")

QUESTIONS = [
    "Name one benefit of keeping prompts on your own machines.",
    "What is a one-time pad, in one sentence?",
    "Give a one-line definition of an open-weight model.",
]


async def ask(client: AsyncOpenAI, question: str) -> str:
    resp = await client.chat.completions.create(
        model=MODEL, messages=[{"role": "user", "content": question}], max_tokens=64, temperature=0.0
    )
    return resp.choices[0].message.content or ""


async def main() -> None:
    client = AsyncOpenAI()
    answers = await asyncio.gather(*(ask(client, q) for q in QUESTIONS))
    for q, a in zip(QUESTIONS, answers, strict=True):
        print(f"Q: {q}\nA: {a}\n")

    stream = await client.chat.completions.create(
        model=MODEL, messages=[{"role": "user", "content": "Count from one to five in words."}], stream=True
    )
    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print()


if __name__ == "__main__":
    asyncio.run(main())
