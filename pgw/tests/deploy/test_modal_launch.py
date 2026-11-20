"""Modal demo launcher (``deploy/modal/launch.py``): ``--max-trip-ms`` relaunch loop, i6pn scheduling
fallback to the tunnel, GPU switch and the E2 placeholder.

Nothing here talks to Modal: the launcher's Modal calls are replaced by fakes."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

MODAL_DIR = Path(__file__).resolve().parents[2] / "deploy" / "modal"


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"pgw_modal_{name}", MODAL_DIR / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


modal = pytest.importorskip("modal")
launch = _load("launch")


class FakeDict(dict):  # type: ignore[type-arg]
    """``modal.Dict`` stand-in (``get``/``pop``/item access)."""


class FakeCall:
    def __init__(self, oid: str, result: Any = None, done: bool = False):
        self.object_id, self.result, self.done, self.cancelled = oid, result, done, False

    def get(self, timeout: float | None = None) -> Any:
        if self.cancelled:
            raise RuntimeError("cancelled")
        if not self.done:
            raise TimeoutError
        return self.result

    def cancel(self, terminate_containers: bool = False) -> None:
        self.cancelled = True


class FakeFunction:
    def __init__(self, app: str, name: str, log: list[tuple[str, str, tuple[Any, ...]]]):
        self.app, self.name, self.log = app, name, log

    def spawn(self, *args: Any) -> FakeCall:
        self.log.append((self.app, self.name, args))
        return FakeCall(f"fc-{len(self.log)}")


def test_args_gate_needs_e2e_and_gpu_choices(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PGW_MODAL_GPU", raising=False)
    monkeypatch.delenv("PGW_MODAL_I6PN", raising=False)
    a = launch.parse_args([])
    assert a.gpu is None and a.i6pn is False and a.max_trip_ms is None and a.tries == 3
    assert a.local_linears == "off"
    with pytest.raises(SystemExit):
        launch.parse_args(["--max-trip-ms", "1"])
    with pytest.raises(SystemExit):
        launch.parse_args(["--gpu", "L4"])  # never L4
    with pytest.raises(SystemExit):
        launch.parse_args(["--local-linears", "attention"])  # placeholder until WP-A4b/A5
    a = launch.parse_args(["--e2e", "--max-trip-ms", "1", "--gpu", "A10G", "--i6pn"])
    assert a.gpu == "A10G" and a.i6pn and a.max_trip_ms == 1.0
    monkeypatch.setenv("PGW_MODAL_GPU", "T4")
    monkeypatch.setenv("PGW_MODAL_I6PN", "1")
    a = launch.parse_args([])
    assert a.gpu == "T4" and a.i6pn is True


def test_cost_per_gpu_tier() -> None:
    cpu = launch.cost_per_h(None)
    assert cpu["pgw-provider"] == pytest.approx(4 * 0.0473 + 6 * 0.008)
    assert launch.cost_per_h("T4")["pgw-provider"] == pytest.approx(0.59 + 2 * 0.0473 + 6 * 0.008)
    assert set(launch.GPU_PRICE_H) == {"T4", "A10G", "L40S", "H100"}
    assert launch.cost_per_h("H100")["pgw-customer"] == cpu["pgw-customer"]


def test_deploy_passes_the_gpu_to_the_provider_deploy(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[dict[str, str]] = []
    monkeypatch.setattr(launch.subprocess, "run", lambda cmd, check, env: seen.append(env))
    monkeypatch.setenv("PGW_MODAL_GPU", "T4")
    launch.deploy("L40S")
    assert [e["PGW_MODAL_GPU"] for e in seen] == ["L40S", "L40S"]
    seen.clear()
    launch.deploy(None)
    assert all("PGW_MODAL_GPU" not in e for e in seen)


def test_spawn_pair_picks_functions_by_network() -> None:
    log: list[tuple[str, str, tuple[Any, ...]]] = []
    live = FakeDict(stop="x", provider={"run_id": "old"}, customer_boot={"run_id": "old"})
    p = launch.spawn_pair(live, False, 600, True, lookup=lambda app, name: FakeFunction(app, name, log))
    assert [(a, n) for a, n, _ in log] == [("pgw-provider", "serve"), ("pgw-customer", "gateway")]
    assert log[1][2] == (p.run_id, 600, True) and p.network == "tunnel"
    assert not any(k in live for k in launch.RUN_KEYS)  # stale per-run keys cleared
    log.clear()
    p = launch.spawn_pair(live, True, 600, False, lookup=lambda app, name: FakeFunction(app, name, log))
    assert [(a, n) for a, n, _ in log] == [("pgw-provider", "serve_i6pn"), ("pgw-customer", "gateway_i6pn")]
    assert p.network == "i6pn"


def test_modal_apps_define_both_network_variants() -> None:
    import ast

    for fname, names in (("provider.py", launch.FUNCTIONS), ("customer.py", launch.FUNCTIONS)):
        tree = ast.parse((MODAL_DIR / fname).read_text())
        defs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
        idx = 0 if fname == "provider.py" else 1
        for i6pn, pair in names.items():
            fn = defs[pair[idx]]
            deco = fn.decorator_list[0]
            assert isinstance(deco, ast.Call)
            kw = {k.arg: k.value for k in deco.keywords if k.arg}
            assert isinstance(kw["i6pn"], ast.Constant) and kw["i6pn"].value is i6pn, (fname, pair[idx])
        # the boot marker is written before anything slow (the scheduling fallback relies on it)
        impl = defs["_serve" if idx == 0 else "_gateway"]
        src = ast.get_source_segment((MODAL_DIR / fname).read_text(), impl) or ""
        marker = "provider_boot" if idx == 0 else "customer_boot"
        assert src.index(marker) < src.index("import pgw_modal_boot")


class _Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.t += s


def _pair(i6pn: bool = True, run_id: str = "r1") -> Any:
    return launch.Pair(run_id, i6pn, FakeCall("p"), FakeCall("c"), 0.0)


def test_wait_scheduled_times_out_on_missing_party() -> None:
    clock, live = _Clock(), FakeDict(provider_boot={"run_id": "r1"})
    missing = launch.wait_scheduled(live, _pair(), 60, poll_s=5, clock=clock, sleep=clock.sleep)
    assert missing == ["customer"] and clock.t >= 60
    live["customer_boot"] = {"run_id": "r1"}
    assert launch.wait_scheduled(live, _pair(), 60, clock=clock, sleep=clock.sleep) == []
    live["customer_boot"] = {"run_id": "old-run"}  # a marker of another run does not count
    assert launch.wait_scheduled(live, _pair(), 0, clock=clock, sleep=clock.sleep) == ["customer"]


def test_wait_scheduled_raises_when_a_call_dies_before_starting() -> None:
    clock, pair = _Clock(), _pair()
    pair.pcall.cancelled = True
    with pytest.raises(RuntimeError, match="provider ended before it started"):
        launch.wait_scheduled(FakeDict(), pair, 60, clock=clock, sleep=clock.sleep)


def test_i6pn_falls_back_to_the_tunnel_when_not_scheduled() -> None:
    spawned: list[bool] = []
    cancelled: list[str] = []

    def spawn(live: Any, i6pn: bool, ttl_s: int, e2e: bool) -> Any:
        spawned.append(i6pn)
        return _pair(i6pn, f"r{len(spawned)}")

    live = FakeDict()
    p = launch.launch_pair(live, i6pn=True, ttl_s=60, e2e=True, schedule_timeout_s=300, spawn=spawn,
                           wait=lambda live, pair, t: ["provider"], cancel=lambda live, pair: cancelled.append(pair.run_id))
    assert spawned == [True, False] and cancelled == ["r1"]
    assert p.network == "tunnel" and "not scheduled within 5 min" in p.note

    spawned.clear()
    p = launch.launch_pair(live, i6pn=True, ttl_s=60, e2e=True, schedule_timeout_s=300, spawn=spawn,
                           wait=lambda live, pair, t: [], cancel=lambda live, pair: cancelled.append("no"))
    assert spawned == [True] and p.network == "i6pn" and p.note == ""

    spawned.clear()

    def never(*a: Any) -> list[str]:
        raise AssertionError("the tunnel needs no scheduling wait")

    p = launch.launch_pair(live, i6pn=False, ttl_s=60, e2e=True, schedule_timeout_s=300, spawn=spawn, wait=never)
    assert spawned == [False]


def test_cancel_pair_signals_and_cancels_without_undeploying() -> None:
    live, pair = FakeDict(), _pair()
    launch.cancel_pair(live, pair)
    assert live["stop"] == "r1" and pair.pcall.cancelled and pair.ccall.cancelled


def _res(ok: bool, p50_bucket: list[float | None] | None, p50: float | None = None, trips: int = 3000) -> dict:
    warm = {"trips": trips, "p50_ms": p50, "p50_bucket_ms": p50_bucket} if p50_bucket else {"trips": 0}
    return {"ok": ok, "e2e": {"trip_latency": {"warm": warm}}}


def test_e2e_verdicts() -> None:
    assert launch.e2e_verdict(_res(True, None), None)[0] == "ok"
    assert launch.e2e_verdict(_res(False, [0.25, 0.5], 0.4), 1.0)[0] == "exactness_failed"
    assert launch.e2e_verdict(_res(True, [0.25, 0.5], 0.4), 1.0)[0] == "ok"
    assert launch.e2e_verdict(_res(True, [5.0, 10.0], 6.8), 1.0)[0] == "gate_failed"
    assert launch.e2e_verdict(_res(True, None), 1.0)[0] == "unmeasured"


def _run_e2e(monkeypatch: pytest.MonkeyPatch, results: list[dict], argv: list[str]) -> tuple[int, list[Any]]:
    pairs: list[Any] = []

    def fake_launch(live: Any, **kw: Any) -> Any:
        p = _pair(kw["i6pn"], f"r{len(pairs) + 1}")
        p.pcall.done, p.pcall.result = True, {"log_tail": ""}
        pairs.append(p)
        return p

    it = iter(results)
    monkeypatch.setattr(launch, "launch_pair", fake_launch)
    monkeypatch.setattr(launch, "wait_e2e", lambda live, pair, ttl: next(it))
    monkeypatch.setattr(launch, "stop", lambda: None)
    code = launch.run_e2e(FakeDict(), launch.parse_args(["--e2e", *argv]), 600)
    return code, pairs


def test_run_e2e_relaunches_until_the_gate_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    slow, fast = _res(True, [5.0, 10.0], 6.8), _res(True, [0.25, 0.5], 0.4)
    code, pairs = _run_e2e(monkeypatch, [slow, fast], ["--max-trip-ms", "1", "--i6pn"])
    assert code == 0 and len(pairs) == 2
    assert pairs[0].pcall.cancelled  # the first pair is cancelled before the relaunch


def test_run_e2e_gives_up_after_tries(monkeypatch: pytest.MonkeyPatch) -> None:
    slow = _res(True, [5.0, 10.0], 6.8)
    code, pairs = _run_e2e(monkeypatch, [slow] * 5, ["--max-trip-ms", "1"])
    assert code == 2 and len(pairs) == 3
    code, pairs = _run_e2e(monkeypatch, [slow] * 5, ["--max-trip-ms", "1", "--tries", "1"])
    assert code == 2 and len(pairs) == 1


def test_run_e2e_never_retries_an_exactness_failure_or_without_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    bad = _res(False, [0.25, 0.5], 0.4)
    code, pairs = _run_e2e(monkeypatch, [bad] * 3, ["--max-trip-ms", "1"])
    assert code == 1 and len(pairs) == 1
    slow = _res(True, [5.0, 10.0], 6.8)
    code, pairs = _run_e2e(monkeypatch, [slow] * 3, [])  # no gate: one pair, success
    assert code == 0 and len(pairs) == 1
    code, pairs = _run_e2e(monkeypatch, [_res(True, None)] * 3, ["--max-trip-ms", "1"])
    assert code == 2 and len(pairs) == 1  # nothing measured: a relaunch would not help


def test_modal_apps_import_and_refuse_l4(monkeypatch: pytest.MonkeyPatch) -> None:
    """The app modules build their Modal definitions locally (no network) with both network variants."""
    monkeypatch.delenv("PGW_MODAL_GPU", raising=False)
    monkeypatch.delenv("PGW_MODAL_PROVIDER_CPUS", raising=False)
    prov, cust = _load("provider"), _load("customer")
    for i6pn, (pfn, cfn) in launch.FUNCTIONS.items():
        assert hasattr(prov, pfn) and hasattr(cust, cfn), i6pn
    assert prov.GPU is None and prov.CPUS == 4.0
    monkeypatch.setenv("PGW_MODAL_GPU", "A10G")
    assert _load("provider").CPUS == 2.0
    monkeypatch.setenv("PGW_MODAL_GPU", "L4")
    with pytest.raises(ValueError, match="L4"):
        _load("provider")
