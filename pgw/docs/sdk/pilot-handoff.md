# PGW pilot: getting started

<!-- Sender: fill in every <angle-bracket> field before sending. Send the SDK key separately, through a
secret-sharing channel, never in the same message as this page. -->

Welcome to the PGW pilot. PGW lets your application use an open-weight LLM on our GPUs without our GPUs
ever seeing your prompts or the answers. This page is all you need to start.

## What you get

* **An SDK key** (starts with `pgw_`), sent to you separately. It is personal to your organisation and can
  be revoked and reissued at any time.
* **The client library** `pgw-sdk` for Python 3.11 to 3.13. Your code keeps the OpenAI client interface:
  change `from openai import OpenAI` to `from pgw import OpenAI`.
* **Models:** Qwen2.5-Instruct 0.5B, 1.5B, 3B and 7B (`qwen2.5-7b-instruct` and so on).
* **Our GPU service** in RunPod data center **`<DATA CENTER ID>`** for the pilot period (`<START>` to `<END>`).
* **Support:** `<SUPPORT CONTACT>`, answered within `<RESPONSE TIME>` on working days.

## Prerequisites

* A RunPod account of your own, to run a small CPU machine next to our GPU. (Your application can live
  anywhere; only the process that uses the library must be close to our GPU, because every generated token
  needs about a hundred network round trips to it.)
* Basic familiarity with a Linux shell and Python.

## Setup in five minutes

1. **Create a CPU pod** in the RunPod console: Pods, Deploy, CPU. Choose data center **`<DATA CENTER ID>`**,
   at least 8 vCPUs and 16 GB memory (16 vCPUs and 32 GB for the 7B model), an Ubuntu 22.04 or newer image,
   and at least 30 GB of disk. Deploy it and open its web terminal or connect with SSH.
2. **Install the library** (the CPU build of PyTorch keeps the install to about 1.3 GB):

   ```bash
   python3 --version   # needs 3.11 or newer; otherwise use the uv route below
   python3 -m venv ~/pgw && . ~/pgw/bin/activate
   pip install --upgrade pip
   pip install pgw-sdk --extra-index-url https://download.pytorch.org/whl/cpu
   ```

   If the image has an older Python, uv installs a current one for you:

   ```bash
   pip install uv && uv venv --python 3.12 ~/pgw && . ~/pgw/bin/activate
   uv pip install pgw-sdk --torch-backend=cpu
   ```

3. **Check the machine:** `pgw doctor` must report no `FAIL` and `pgw_fast, pgw_ntt, pgw_pks loaded`.
4. **Set your key** (paste it from the separate message; do not store it in files you share):

   ```bash
   export PGW_API_KEY='<paste your key here>'
   ```

5. **First answer.** Save this as `pgw_pilot.py` and run `python pgw_pilot.py`:

   ```python
   import sys, time
   from pgw import OpenAI

   client = OpenAI()
   t0 = time.perf_counter(); first = None; n = 0
   for chunk in client.chat.completions.create(
           model="qwen2.5-7b-instruct", stream=True, max_tokens=200,
           messages=[{"role": "user", "content": "Explain in three bullet points what you can do."}]):
       if chunk.choices and chunk.choices[0].delta.content:
           first = first or time.perf_counter(); n += 1
           print(chunk.choices[0].delta.content, end="", flush=True)
   end = time.perf_counter()
   print(f"\n[first token {first - t0:.2f} s, {n} tokens, {(n - 1) / (end - first):.1f} tokens/s]", file=sys.stderr)
   ```

   The very first run prepares the model on this machine and takes a few minutes; later runs start in
   seconds.

## What to test

Please try the following and note the results:

1. **Your own prompts**, of the length and kind you would use in production, with `temperature=0` and with
   your usual sampling settings.
2. **Exactness:** the same prompt twice with `temperature=0` must give the same answer, word for word.
3. **Speed:** the `[first token ..., tokens/s]` line for the 7B model and for one smaller model, alone and with
   4 to 8 requests at once (use `AsyncOpenAI`, one client shared by all requests).
4. **Restart:** stop and start your process; the second start must be quick.
5. **Your integration:** call the library from your application, or from a small service next to it (see
   the FastAPI example we include), and let us know which framework you use.

Tool calling, JSON-schema structured output, images and `n > 1` are not available in this release.

## How to report issues

Write to `<SUPPORT CONTACT>` with:

* the output of `pgw doctor --json` and `pip freeze`;
* the time (with time zone), the model, the error message and code, and the request id if one is shown;
* what you expected and what happened.

**Never send us your key, prompts or answers.** If we need an example to reproduce a problem, a made-up
prompt that shows the same behaviour is best. If you see `integrity_check_failed` or `mask_reuse`, stop
and tell us at once: these are security signals, not ordinary errors.

## Our privacy promise, in plain words

* Your prompts and the model's answers are processed inside your own process, on your machine. They are
  never sent to us in readable form.
* What our GPUs receive is masked with random numbers that are used once and then thrown away. To us it
  is noise. We cannot read it, and neither can anyone who breaks into our servers.
* Every result we send back is checked by your machine. If we, or anyone between us, returned a wrong
  result, the request fails instead of giving you a wrong answer.
* We do see that you made a request, for which model, how many tokens, and when. We use this for billing
  and capacity only.
* Honest status: no external security audit of PGW has been completed yet, and the default mask
  scheme (`pks-v3`) is awaiting an independent cryptographic review. We will tell you as soon as the
  reviews are done. Details: [Security model summary](security.md).
