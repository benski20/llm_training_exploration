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
| Network | outbound HTTPS to the provider address encoded in your key. Run in the **same data center** as our GPU service (see [Concepts](concepts.md#why-your-application-should-run-near-our-gpu)) |
| Key | an SDK key that starts with `pgw_`, issued to you by us |

Alpine Linux (musl) and Windows are not supported in this release.

## 2. Install

Use a virtual environment. On Linux, install the CPU build of PyTorch: the default PyTorch package on
PyPI for Linux x86_64 also pulls several gigabytes of CUDA libraries that PGW never uses. Release 0.1 is
measured with PyTorch 2.15.0 (`torch==2.15.0` from the same index pins it); other versions from 2.4 on work and
give the same results.

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
2. **Model preparation.** The library downloads the public model data it needs and checks every file against
   the model fingerprint, which the library itself pins for every supported model (a service serving anything
   else is refused): the tokenizer and the small plaintext parts of the model for your process, and the public
   int8 weights for the background mask generation (about 6.5 GB at 7B; your request handling never loads them).
3. **Mask setup.** With the default `hint="local"`, the background mask processes build the one-time setup on
   your machine from those weights: about 20-30 minutes of CPU at 7B on 4 cores (a few minutes at 0.5B).

All three are cached: expect one-time ~6.5 GB of public weight download plus ~20-30 min of CPU for the 7B
model on a 4-core machine, then a few seconds to start. **Before real traffic, run `pgw sdk prepare --model
qwen2.5-7b-instruct`** once on the machine (or on a bigger one and copy the result:
[Configuration](configuration.md#preparing-a-model-ahead-of-time)): it does all three ahead of time, so your first
requests do not wait for them. While a setup runs, `pgw sdk status` shows `first-time setup in progress` with the
share done and an estimate of the time left. Mask generation keeps running in background
processes while your process is up. The [Configuration reference](configuration.md#cache-directory) has the
disk figures and the `hint="provider"` alternative.

**Placement.** Run the process that imports `pgw` in the same data center as our GPU service: every
generated token needs about a hundred round trips to it ([Concepts](concepts.md#why-your-application-should-run-near-our-gpu)).
For the pilot, that is a CPU machine in the data center we name with your key.

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
