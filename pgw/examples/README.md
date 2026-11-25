# PGW client examples

Each example uses only `pgw-sdk` (plus FastAPI for the web example) and the OpenAI-style interface.
Set your key first: `export PGW_API_KEY=pgw_...`. Change the model with `PGW_EXAMPLE_MODEL`
(default `qwen2.5-7b-instruct`). Run them in the same data center as our GPU service. The first run of a model
prepares it once (for 7B: about 6.5 GB of public weights to download and about 20-30 minutes of CPU on 4 cores;
`pgw sdk status` shows the progress); later runs start in seconds.

| file | what it shows | run |
|---|---|---|
| [`chat.py`](chat.py) | one chat completion, token usage | `python examples/chat.py` |
| [`streaming.py`](streaming.py) | streamed tokens, time to first token and tokens per second | `python examples/streaming.py "your prompt"` |
| [`async_chat.py`](async_chat.py) | `AsyncOpenAI`: concurrent requests and async streaming | `python examples/async_chat.py` |
| [`fastapi_app.py`](fastapi_app.py) | a web service using one shared client; an OpenAI-compatible `/v1/chat/completions` route | `pip install fastapi uvicorn` then `uvicorn examples.fastapi_app:app --port 8000 --workers 1` |

## Using OpenAI-compatible tools (LangChain, LlamaIndex, ...)

Frameworks that talk to the OpenAI HTTP API can use PGW through `fastapi_app.py`, which serves
`/v1/chat/completions` in the OpenAI wire format (streaming included). Run it on the machine next to our
GPU and point the framework's base URL at it. No PGW-specific integration package is needed; the
framework packages are your own dependencies.

LangChain (`pip install langchain-openai`):

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(base_url="http://127.0.0.1:8000/v1", api_key="change-me", model="qwen2.5-7b-instruct")
print(llm.invoke("Hello").content)
```

LlamaIndex (`pip install llama-index-llms-openai-like`):

```python
from llama_index.llms.openai_like import OpenAILike

llm = OpenAILike(api_base="http://127.0.0.1:8000/v1", api_key="change-me",
                 model="qwen2.5-7b-instruct", is_chat_model=True)
print(llm.complete("Hello"))
```

`api_key` here is the `APP_TOKEN` of `fastapi_app.py` (any value if you left it unset), not your PGW key:
the PGW key stays in the app's environment. The OpenAI wire format of the app is covered by
`tests/packaging/test_examples.py`; the framework snippets themselves are not run in our tests.

Tool calling and structured output are not supported in this release, so agents that depend on them will
not work yet.
