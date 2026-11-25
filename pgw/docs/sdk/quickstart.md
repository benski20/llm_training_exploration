# Quickstart: the PGW client library

PGW lets your application use an open-weight LLM on our GPUs without our GPUs ever seeing your prompts or
the answers. You install one Python package and change one import. Your code keeps the OpenAI client
interface it already uses.

This page takes you from nothing to a streamed answer in about ten minutes. Most of that time is the
one-time model preparation on first use.

## 1. Requirements

| | |
|---|---|
| Python | 3.11, 3.12 or 3.13 (CPython) |
| Operating system | Linux x86_64 or aarch64 (glibc 2.28 or newer: Ubuntu 20.04+, Debian 11+, RHEL 8+), macOS on Apple silicon |
| Hardware | CPU only. No GPU, no CUDA. See [Supported models](models.md) for the CPU, memory and disk to plan per model |
| Network | outbound HTTPS to the provider address encoded in your key; low latency to our GPU matters (see [Concepts](concepts.md#why-your-application-should-run-near-our-gpu)) |
| Key | an SDK key that starts with `pgw_`, issued to you by us |

Alpine Linux (musl) and Windows are not supported in this release.

## 2. Install

Use a virtual environment. On Linux, install the CPU build of PyTorch: the default PyTorch package on
PyPI for Linux x86_64 also pulls several gigabytes of CUDA libraries that PGW never uses.

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install --upgrade pip
pip install pgw-sdk --extra-index-url https://download.pytorch.org/whl/cpu
```

On macOS, `pip install pgw-sdk` is enough (PyTorch for macOS has no CUDA). With
the `uv` installer: `uv pip install pgw-sdk --torch-backend=cpu`.

The distribution is called `pgw-sdk`; the Python import name is `pgw`. (The name `pgw` on PyPI belongs to
an unrelated project. Do not install it.)

What you get: a wheel of about 4 MB with prebuilt native extensions for your platform, plus its
dependencies. Measured on Linux x86_64 with the CPU build of PyTorch (Python 3.11 and 3.13), the whole
environment takes about 1.3 GB on disk, of which PyTorch is about 0.8 GB and Hugging Face `transformers`
about 0.13 GB. Model data is downloaded later, on first use, into a cache directory
([Configuration](configuration.md#cache-directory)).

Check the installation:

```bash
pgw version
pgw doctor
```

`pgw doctor` must end with no `FAIL` line. The `native` line must say
`pgw_fast, pgw_ntt, pgw_pks loaded`. If it says `pure-Python fallback`, PGW still works and gives the same
results, but much more slowly; see [Troubleshooting](troubleshooting.md#native-extensions-not-loaded).
Lines about the `gateway`, `service` and `factory` extras are warnings for server deployments and do not
concern the client library.

## 3. Set your key

Keep the key out of your code and out of version control. The library reads it from the environment:

```bash
export PGW_API_KEY='pgw_...'
```

You can also pass it explicitly: `OpenAI(api_key=...)`. The key tells the library where our service is
and which certificate to trust (it is pinned), so there is no `base_url` to set.

## 4. First call

```python
from pgw import OpenAI

client = OpenAI()  # reads PGW_API_KEY
resp = client.chat.completions.create(
    model="qwen2.5-7b-instruct",
    messages=[{"role": "user", "content": "Say hello in one short sentence."}],
    max_tokens=64,
)
print(resp.choices[0].message.content)
```

The first call on a machine does three things before it answers:

1. **Enrollment.** A client key pair is generated on your machine; our service issues a certificate for it.
2. **Model preparation.** The library downloads the non-secret model data it needs (tokenizer and the small
   plaintext parts of the model) and checks every file against the model fingerprint.
3. **Mask setup.** The library prepares the material it needs to generate one-time masks with our help, and
   verifies what we contributed before using it.

All three are cached. Expect a few minutes for the 7B model on the first call and a few seconds to start
after that. Mask generation then keeps running in background threads while your process is up.

## 5. Streaming

```python
stream = client.chat.completions.create(
    model="qwen2.5-7b-instruct",
    messages=[{"role": "user", "content": "Write a haiku about privacy."}],
    stream=True,
)
for chunk in stream:
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

## 6. Async

```python
import asyncio
from pgw import AsyncOpenAI

async def main() -> None:
    client = AsyncOpenAI()
    resp = await client.chat.completions.create(
        model="qwen2.5-7b-instruct", messages=[{"role": "user", "content": "Hi"}]
    )
    print(resp.choices[0].message.content)

asyncio.run(main())
```

Concurrent requests from one process share the in-process gateway, which batches them: create one client
per process and reuse it, rather than one per request.

## 7. Next steps

* [Concepts](concepts.md): what stays private, what runs where, and why placement matters.
* [Configuration reference](configuration.md): client options, request parameters, environment variables.
* [Examples](../../examples/README.md): chat, streaming, async, and a FastAPI service.
* [Troubleshooting](troubleshooting.md) and the [security model](security.md).
