"""Streaming chat completion: print tokens as they are generated, then the speed.

    export PGW_API_KEY=pgw_...
    python examples/streaming.py "Write a haiku about private inference."

The last line (on stderr) reports time to first token and tokens per second: the numbers to send us
during a pilot.
"""

from __future__ import annotations

import os
import sys
import time

from pgw import OpenAI

MODEL = os.environ.get("PGW_EXAMPLE_MODEL", "qwen2.5-7b-instruct")


def main() -> None:
    prompt = " ".join(sys.argv[1:]) or "Explain one-time masks to a new engineer in three short bullet points."
    client = OpenAI()
    t0 = time.perf_counter()
    first: float | None = None
    pieces = 0
    usage = None
    stream = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=256,
        stream=True,
        stream_options={"include_usage": True},
    )
    for chunk in stream:
        if getattr(chunk, "usage", None) is not None:
            usage = chunk.usage
        if chunk.choices and chunk.choices[0].delta.content:
            if first is None:
                first = time.perf_counter()
            pieces += 1
            print(chunk.choices[0].delta.content, end="", flush=True)
    end = time.perf_counter()
    print()
    tokens = usage.completion_tokens if usage is not None else pieces
    if first is not None:
        rate = (tokens - 1) / (end - first) if tokens > 1 and end > first else 0.0
        print(f"[{MODEL}: first token after {first - t0:.2f} s, {tokens} tokens, {rate:.1f} tokens/s after the first]",
              file=sys.stderr)


if __name__ == "__main__":
    main()
