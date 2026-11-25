"""Content-free request reports to the provider (``POST /pgw/v1/sdk/report``) for the developer console.

After each request the library queues one record: start time, model, status and error code, prompt and completion
token counts, latency and time to first token, whether it streamed, and whether every integrity check of the
request passed (``checks_ok``). Nothing else: no prompt, completion, token ids or logprobs. The provider already
sees the same counts and timing in the masked traffic (rows per trip, number of trips, arrival times); the report
only lets it attribute them to requests for the console's request log and the input/output split.

Records are sent in batches from a daemon thread over the session's pinned TLS connection with its service token.
Reporting never delays or fails a request: a full queue drops the oldest records, a failed send is retried a few
times and then dropped, and a provider without the endpoint (404) turns reporting off. ``usage_reports=False``
turns it off entirely.
"""

from __future__ import annotations

import collections
import contextlib
import logging
import threading
from collections.abc import Callable
from typing import Any

__all__ = ["REPORT_PATH", "Reporter"]

log = logging.getLogger(__name__)

REPORT_PATH = "/pgw/v1/sdk/report"
_FIELDS = ("ts", "model", "status", "error_code", "prompt_tokens", "completion_tokens", "latency_ms", "ttft_ms",
           "stream", "checks_ok")


class Reporter:
    """Batches request records and sends them from a daemon thread.

    Args:
        session: Returns the current :class:`pgw.sdk.session.Session` (``None`` until enrolled).
        interval_s: Seconds between sends.
        max_queue: Records kept while the provider is unreachable (oldest dropped first).
    """

    def __init__(self, session: Callable[[], Any], *, interval_s: float = 5.0, max_queue: int = 4096):
        self._session = session
        self.interval_s = interval_s
        self._q: collections.deque[dict[str, Any]] = collections.deque(maxlen=max_queue)
        self._mu = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.enabled = True
        self.sent = 0
        self.failures = 0

    def add(self, record: dict[str, Any]) -> None:
        """Queue one record (only the allow-listed fields are kept)."""
        if not self.enabled:
            return
        with self._mu:
            self._q.append({k: record.get(k) for k in _FIELDS})
            if self._thread is None:
                self._thread = threading.Thread(target=self._loop, name="pgw-sdk-reports", daemon=True)
                self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(self.interval_s)
            self._wake.clear()
            with contextlib.suppress(Exception):
                self.flush()

    def flush(self, timeout_s: float = 10.0) -> None:
        """Send what is queued (one batch per call of up to 256 records; blocking)."""
        while self.enabled:
            with self._mu:
                batch = [self._q.popleft() for _ in range(min(256, len(self._q)))]
            if not batch:
                return
            if not self._send(batch, timeout_s):
                with self._mu:  # keep them for the next round (the deque bound drops the oldest)
                    self._q.extendleft(reversed(batch))
                return

    def _send(self, batch: list[dict[str, Any]], timeout_s: float) -> bool:
        s = self._session()
        if s is None:
            return False
        try:
            with s.http_client(timeout_s=timeout_s) as h:
                r = h.post(REPORT_PATH, json={"requests": batch})
        except Exception as exc:  # unreachable: try again later
            self.failures += 1
            log.debug("request report not sent (%s)", type(exc).__name__)
            return self.failures % 5 == 0  # give up on this batch after a few attempts
        if r.status_code == 404:
            log.info("the provider does not accept request reports; reporting off")
            self.enabled = False
            return True
        if r.status_code >= 400:
            self.failures += 1
            return r.status_code < 500 or self.failures % 5 == 0  # a 4xx will not get better
        self.sent += len(batch)
        return True

    def close(self, timeout_s: float = 5.0) -> None:
        """Send what is left (best effort) and stop the thread."""
        self._stop.set()
        self._wake.set()
        t, self._thread = self._thread, None
        if t is not None:
            t.join(timeout=timeout_s)
        with contextlib.suppress(Exception):
            self.flush(timeout_s=timeout_s)
