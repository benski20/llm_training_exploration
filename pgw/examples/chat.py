"""Minimal chat completion through PGW.

    pip install pgw-sdk --extra-index-url https://download.pytorch.org/whl/cpu
    export PGW_API_KEY=pgw_...        # the key you received
    python examples/chat.py

The first run enrolls this machine and prepares the model in the local cache (a few minutes for 7B);
later runs start in seconds. Prompts and answers never leave this process in readable form.
"""

from __future__ import annotations

import os

from pgw import OpenAI

MODEL = os.environ.get("PGW_EXAMPLE_MODEL", "qwen2.5-7b-instruct")


def main() -> None:
    client = OpenAI()  # reads PGW_API_KEY; or OpenAI(api_key="pgw_...")
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": "In two sentences: why do banks care where their data is processed?"},
        ],
        max_tokens=128,
        temperature=0.0,  # greedy: the same prompt gives the same answer
    )
    print(resp.choices[0].message.content)
    if resp.usage is not None:
        print(f"[{resp.usage.prompt_tokens} prompt + {resp.usage.completion_tokens} completion tokens]")


if __name__ == "__main__":
    main()
