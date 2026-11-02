"""Client for the verification API (vt/verify_service.py): standard library only.

    from vt.verify_client import VerifyClient
    v = VerifyClient("https://verify.example.com", api_key="...")
    r = v.verify(question="When did the bridge reopen?", reply=answer, documents=[article])
    if r["verdict"] != "consistent":
        for s in r["sentences"]:
            if s["verdict"] != "consistent":
                print(s["verdict"], s["text"])

Retries a rate-limited (429) or failed (5xx) request a few times with backoff; other errors raise VerifyError.
"""
import json
import time
import urllib.error
import urllib.request


class VerifyError(Exception):
    def __init__(self, status, message):
        super().__init__(f"{status}: {message}")
        self.status, self.message = status, message


class VerifyClient:
    def __init__(self, url, api_key=None, timeout=120.0, retries=3, backoff=1.0):
        self.url, self.api_key, self.timeout = url.rstrip("/"), api_key, timeout
        self.retries, self.backoff = retries, backoff

    def _post(self, path, body):
        data = json.dumps(body).encode()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        for attempt in range(self.retries + 1):
            req = urllib.request.Request(self.url + path, data, headers)
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:  # nosec B310: the caller's API URL
                    return json.loads(r.read())
            except urllib.error.HTTPError as e:
                msg = e.read().decode(errors="replace")[:300]
                if (e.code == 429 or e.code >= 500) and attempt < self.retries:
                    time.sleep(self.backoff * 2 ** attempt)
                    continue
                try:
                    msg = json.loads(msg).get("error", msg)
                except ValueError:
                    pass
                raise VerifyError(e.code, msg) from None
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                if attempt < self.retries:
                    time.sleep(self.backoff * 2 ** attempt)
                    continue
                raise VerifyError(0, f"unreachable: {e}") from None

    def verify(self, reply, question="", documents=(), lookup=False):
        return self._post("/v1/verify", {"question": question, "reply": reply, "documents": list(documents),
                                         "lookup": lookup})

    def verify_batch(self, items):
        """items: dicts with reply, question, documents, lookup; at most 16 a call."""
        return self._post("/v1/verify/batch", {"items": list(items)})["results"]
