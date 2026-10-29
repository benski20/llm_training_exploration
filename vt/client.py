"""A client for the VeriChat API that survives a lost request. The server sends a heartbeat every 10 s while
it works on a streamed reply, so silence means the request was lost on the way (live, Modal's routing
occasionally dropped one before it reached the server, deploy/RUNBOOK.md): after `silence` seconds
without a byte, the request is abandoned and sent again, at most `retries` times.

    from vt.client import chat
    reply = chat("https://<...>.modal.run", [{"role": "user", "content": "Who designed the Eiffel Tower?"}])
    reply["text"], reply.get("verification"), reply["_attempts"]
"""
import json
import socket
import time
import urllib.error
import urllib.request


class Lost(Exception):
    """No reply after every attempt."""


def _once(url, body, headers, silence):
    if not url.startswith(("https://", "http://")):
        raise ValueError("the server URL must be http(s)")
    req = urllib.request.Request(url.rstrip("/") + "/chat", body, headers)
    with urllib.request.urlopen(req, timeout=silence) as r:  # nosec B310: scheme checked; timeout = silence
        buf, final = b"", None
        while True:
            chunk = r.read1(1 << 16)
            if not chunk:
                break
            buf += chunk
            while b"\n\n" in buf:
                ev, buf = buf.split(b"\n\n", 1)
                if ev.startswith(b"data: "):
                    m = json.loads(ev[6:])
                    if m.get("done"):
                        final = m
    if final is None:
        raise ConnectionError("the stream ended before the reply")
    if final.get("error"):
        raise RuntimeError(final["error"])                     # the server's own answer: not retried
    return final["reply"]


def chat(url, messages, mode="hedge", documents=None, silence=45.0, retries=1, api_key=None, **extra):
    body = json.dumps({"messages": messages, "mode": mode, "documents": documents or [], "stream": True,
                       **extra}).encode()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    errors = []
    for attempt in range(1, retries + 2):
        try:
            reply = _once(url, body, headers, silence)
            reply["_attempts"] = attempt
            return reply
        except urllib.error.HTTPError as e:                    # 4xx/5xx are answers: report, do not retry
            raise RuntimeError(f"HTTP {e.code}: {e.read()[:200].decode(errors='replace')}") from None
        except ValueError:
            raise
        except (socket.timeout, TimeoutError, ConnectionError, urllib.error.URLError, OSError) as e:
            errors.append(repr(e)[:120])
            time.sleep(min(2.0, attempt))
    raise Lost(f"no reply after {retries + 1} attempts: {errors}")
