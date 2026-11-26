"""Many concurrent streams through the client library, one and several client processes (real ``pgw service`` on
loopback, tiny model, plain masks so the factory keeps up on CPU).

* a burst larger than the mask stock (each request's entitlement x streams > the stock target): every stream
  completes, bit-identical to the same request run alone, with the default admission and with ``overcommit``;
* three client processes x three threads with the same key and cache directory (one mask store, one factory, the
  cross-process entitlement ledger): every stream completes bit-identical, and the store's reservation records show
  no mask slot handed out twice across the processes.
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from tests.sdk.conftest import MODEL, ROOT, Provider

pytestmark = pytest.mark.timeout(1500)

PROMPTS = ["Name a colour.", "Count to three.", "Say hello politely."]


def _body(i: int) -> dict[str, Any]:
    return {"model": MODEL, "messages": [{"role": "user", "content": PROMPTS[i % 3]}], "temperature": 0.8,
            "seed": 100 + i % 3, "max_tokens": 24, "logprobs": True, "user": f"user-{i}",
            "extra_body": {"ignore_eos": True}}


def _lp(r: Any) -> list[list[Any]]:
    return [[c.token, c.logprob] for c in r.choices[0].logprobs.content]


def _client(provider: Provider, key: str, root: Path, **kw: Any) -> Any:
    from pgw import OpenAI
    from pgw.sdk import LocalPackProvisioner

    opts: dict[str, Any] = {"cache_dir": root / "cache", "state_dir": root / "state", "mask_scheme": "plain",
                            "mask_workers": 1, "mask_target_slots": 160, "mask_low_water_slots": 96,
                            "mask_segment_slots": 64, "timeout": 600, "ready_timeout_s": 600, "max_concurrent": 16}
    opts.update(kw)
    return OpenAI(api_key=key, provisioner=LocalPackProvisioner({MODEL: provider.pack_dir}), **opts)


def _no_slot_twice(root: Path) -> int:
    """Every reservation extent recorded in the client's mask store; asserts no slot is in two of them."""
    dbs = list((root / "cache" / "sessions").glob("*/masks/state.sqlite"))
    assert len(dbs) == 1
    con = sqlite3.connect(f"file:{dbs[0]}?mode=ro", uri=True)
    try:
        ext = con.execute("SELECT segment_id, start, count FROM reservation_extents").fetchall()
    finally:
        con.close()
    seen: set[tuple[str, int]] = set()
    for seg, start, count in ext:
        for s in range(int(start), int(start) + int(count)):
            assert (seg, s) not in seen, "a mask slot was reserved twice"
            seen.add((seg, s))
    return len(seen)


@pytest.mark.parametrize("admission", ["entitled", "overcommit"])
def test_burst_larger_than_the_stock(provider: Provider, tmp_path: Path, admission: str) -> None:
    c = _client(provider, provider.new_key(f"burst-{admission}"), tmp_path, mask_admission=admission)
    try:
        refs = [_lp(c.chat.completions.create(**_body(j))) for j in range(3)]  # each request alone
        with ThreadPoolExecutor(12) as ex:
            outs = list(ex.map(lambda i: _lp(c.chat.completions.create(**_body(i))), range(12)))
        assert all(o == refs[i % 3] for i, o in enumerate(outs)), "an answer changed under concurrency"
        st = c.status()["models"][MODEL]
        assert st["integrity_failures"] == 0 and st["requests_waiting_for_masks"] == 0
    finally:
        c.close()
    assert _no_slot_twice(tmp_path) > 12 * 24


_WORKER = """
import json, sys
from concurrent.futures import ThreadPoolExecutor
sys.path[:0] = [sys.argv[4], sys.argv[5]]
from tests.sdk.test_e2e_scale import _body, _lp
from pgw import OpenAI
from pgw.sdk import LocalPackProvisioner
root, wid = sys.argv[2], int(sys.argv[6])
c = OpenAI(api_key=sys.argv[1], cache_dir=root + "/cache", state_dir=root + "/state", mask_scheme="plain",
           mask_workers=1, mask_target_slots=160, mask_low_water_slots=96, mask_segment_slots=64, timeout=600,
           ready_timeout_s=600, provisioner=LocalPackProvisioner({"qwen2.5-0.5b-instruct": sys.argv[3]}))
with ThreadPoolExecutor(3) as ex:
    outs = list(ex.map(lambda i: [i, _lp(c.chat.completions.create(**_body(i)))], range(wid * 3, wid * 3 + 3)))
c.close()
print(json.dumps(outs))
"""


def test_worker_processes_share_the_stock_and_never_reuse_a_mask(provider: Provider, tmp_path: Path) -> None:
    key = provider.new_key("gunicorn")
    c = _client(provider, key, tmp_path)
    try:  # references (and the first enrollment / provisioning, done once for every worker)
        refs = [_lp(c.chat.completions.create(**_body(j))) for j in range(3)]
    finally:
        c.close()
    env = {k: v for k, v in os.environ.items() if not k.startswith("PGW_")}
    procs = [subprocess.Popen([sys.executable, "-c", _WORKER, key, str(tmp_path), str(provider.pack_dir),  # noqa: S603
                               str(ROOT / "src"), str(ROOT), str(w)], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              text=True, env=env) for w in range(3)]
    outs: list[list[Any]] = []
    for p in procs:
        out, err = p.communicate(timeout=1200)
        assert p.returncode == 0, err[-3000:]
        outs += json.loads(out)
    assert sorted(i for i, _ in outs) == list(range(9))
    assert all(lp == refs[i % 3] for i, lp in outs), "an answer changed with several worker processes"
    assert _no_slot_twice(tmp_path) >= 9 * 24
