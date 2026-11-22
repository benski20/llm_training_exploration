"""Modal demo launcher (``deploy/modal/launch.py``): ``--max-trip-ms`` relaunch loop, placements (tunnel,
i6pn, co-located) and their scheduling fallback to the tunnel, GPU switch and the --local-linears setting.

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
    assert a.local_linears == "auto"
    with pytest.raises(SystemExit):
        launch.parse_args(["--max-trip-ms", "1"])
    with pytest.raises(SystemExit):
        launch.parse_args(["--gpu", "L4"])  # never L4
    assert launch.parse_args(["--local-linears", "attention"]).local_linears == "attention"
    with pytest.raises(SystemExit):
        launch.parse_args(["--local-linears", "always"])
    a = launch.parse_args(["--e2e", "--max-trip-ms", "1", "--gpu", "A10G", "--i6pn"])
    assert a.gpu == "A10G" and a.i6pn and a.max_trip_ms == 1.0
    monkeypatch.setenv("PGW_MODAL_GPU", "T4")
    monkeypatch.setenv("PGW_MODAL_I6PN", "1")
    a = launch.parse_args([])
    assert a.gpu == "T4" and a.i6pn is True


@pytest.mark.parametrize("env", ["a10g", "T4:2", "L4", "l4", "B200"])
def test_gpu_env_default_is_normalized_and_validated(monkeypatch: pytest.MonkeyPatch, env: str) -> None:
    """argparse never checks a default against ``choices``: PGW_MODAL_GPU goes through the same check."""
    monkeypatch.setenv("PGW_MODAL_GPU", env)
    if env == "a10g":
        assert launch.parse_args([]).gpu == "A10G"
        return
    with pytest.raises(SystemExit):
        launch.parse_args([])


def test_normalize_gpu_and_device_tier() -> None:
    assert launch.normalize_gpu(None) is None and launch.normalize_gpu(" none ") is None
    assert launch.normalize_gpu("l40s") == "L40S" and launch.normalize_gpu("h100") == "H100"
    for bad in ("L4", "T4:2", "A100"):
        with pytest.raises(ValueError):
            launch.normalize_gpu(bad)
    assert launch.gpu_tier_of_device("none") is None
    assert launch.gpu_tier_of_device("Tesla T4") == "T4"
    assert launch.gpu_tier_of_device("NVIDIA A10G") == "A10G"
    assert launch.gpu_tier_of_device("NVIDIA L40S") == "L40S"
    assert launch.gpu_tier_of_device("NVIDIA H100 80GB HBM3") == "H100"
    with pytest.raises(KeyError):
        launch.gpu_tier_of_device("NVIDIA L4")


def test_i6pn_schedule_timeout_default_is_longer_with_a_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PGW_MODAL_GPU", raising=False)
    assert launch.parse_args([]).i6pn_schedule_timeout_min == 6.0
    assert launch.parse_args(["--gpu", "T4"]).i6pn_schedule_timeout_min == 10.0
    assert launch.parse_args(["--gpu", "T4", "--i6pn-schedule-timeout-min", "3"]).i6pn_schedule_timeout_min == 3.0
    with pytest.raises(SystemExit):
        launch.parse_args(["--i6pn-schedule-timeout-min", "0"])


def test_cost_follows_the_provider_core_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PGW_MODAL_PROVIDER_CPUS", "8")
    assert launch.cost_per_h("T4")["pgw-provider"] == pytest.approx(0.59 + 8 * 0.0473 + 6 * 0.008)
    assert launch.cost_per_h(None)["pgw-provider"] == pytest.approx(8 * 0.0473 + 6 * 0.008)
    # the cores the provider reports (what is deployed) win over the local environment
    assert launch.cost_per_h("A10G", 2.0)["pgw-provider"] == pytest.approx(1.10 + 2 * 0.0473 + 6 * 0.008)


def test_cost_per_gpu_tier(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PGW_MODAL_PROVIDER_CPUS", raising=False)
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
    assert log[1][2] == (p.run_id, 600, True, "auto", 0) and p.network == "tunnel"
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
    spawned: list[str] = []
    cancelled: list[str] = []

    def spawn(live: Any, placement: str, ttl_s: int, e2e: bool) -> Any:
        spawned.append(placement)
        return _pair(placement, f"r{len(spawned)}")

    live = FakeDict()
    p = launch.launch_pair(live, i6pn=True, ttl_s=60, e2e=True, schedule_timeout_s=300, spawn=spawn,
                           wait=lambda live, pair, t: ["provider"], cancel=lambda live, pair: cancelled.append(pair.run_id))
    assert spawned == ["i6pn", "tunnel"] and cancelled == ["r1"]
    assert p.network == "tunnel" and "i6pn: provider not scheduled within 5 min" in p.note

    spawned.clear()
    p = launch.launch_pair(live, placement="i6pn", ttl_s=60, e2e=True, schedule_timeout_s=300, spawn=spawn,
                           wait=lambda live, pair, t: [], cancel=lambda live, pair: cancelled.append("no"))
    assert spawned == ["i6pn"] and p.network == "i6pn" and p.note == ""

    spawned.clear()

    def never(*a: Any) -> list[str]:
        raise AssertionError("the tunnel needs no scheduling wait")

    p = launch.launch_pair(live, i6pn=False, ttl_s=60, e2e=True, schedule_timeout_s=300, spawn=spawn, wait=never)
    assert spawned == ["tunnel"]


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
        p = _pair(kw["placement"], f"r{len(pairs) + 1}")
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
    code, pairs = _run_e2e(monkeypatch, [slow] * 5, ["--max-trip-ms", "1", "--i6pn"])
    assert code == 2 and len(pairs) == 3
    code, pairs = _run_e2e(monkeypatch, [slow] * 5, ["--max-trip-ms", "1", "--i6pn", "--tries", "1"])
    assert code == 2 and len(pairs) == 1
    code, pairs = _run_e2e(monkeypatch, [slow] * 5, ["--max-trip-ms", "20"])  # tunnel, reachable limit
    assert code == 0 and len(pairs) == 1
    code, pairs = _run_e2e(monkeypatch, [slow] * 5, ["--max-trip-ms", "5"])  # tunnel, above the bucket
    assert code == 2 and len(pairs) == 3


def test_run_e2e_does_not_relaunch_tunnel_pairs_below_the_tunnel_floor(monkeypatch: pytest.MonkeyPatch) -> None:
    slow = _res(True, [5.0, 10.0], 6.8)
    code, pairs = _run_e2e(monkeypatch, [slow] * 5, ["--max-trip-ms", "1"])  # tunnel only: no chance
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


# ---------------------------------------------------------------------------------------- placements
def test_placement_args(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PGW_MODAL_I6PN", raising=False)
    monkeypatch.delenv("PGW_MODAL_PLACEMENT", raising=False)
    monkeypatch.delenv("PGW_MODAL_CUSTOMER_GPU", raising=False)
    assert launch.parse_args([]).placement == "tunnel"  # default unchanged
    a = launch.parse_args(["--placement", "colocated"])
    assert a.placement == "colocated" and a.i6pn is False
    a = launch.parse_args(["--i6pn"])
    assert a.placement == "i6pn" and a.i6pn is True
    assert launch.parse_args(["--placement", "i6pn", "--i6pn"]).placement == "i6pn"
    with pytest.raises(SystemExit):
        launch.parse_args(["--placement", "colocated", "--i6pn"])  # contradictory
    with pytest.raises(SystemExit):
        launch.parse_args(["--placement", "cluster"])
    with pytest.raises(SystemExit):  # the co-located gateway has no GPU of its own
        launch.parse_args(["--placement", "colocated", "--customer-gpu", "T4"])
    monkeypatch.setenv("PGW_MODAL_PLACEMENT", "colocated")
    assert launch.parse_args([]).placement == "colocated"
    assert launch.parse_args(["--placement", "tunnel"]).placement == "tunnel"
    monkeypatch.setenv("PGW_MODAL_PLACEMENT", "nowhere")
    with pytest.raises(SystemExit):
        launch.parse_args([])
    monkeypatch.delenv("PGW_MODAL_PLACEMENT")
    # the scheduling timeout covers every placement; the old flag name still works
    assert launch.parse_args(["--schedule-timeout-min", "4"]).i6pn_schedule_timeout_min == 4.0
    assert launch.parse_args(["--i6pn-schedule-timeout-min", "4"]).i6pn_schedule_timeout_min == 4.0
    assert launch.parse_args(["--placement", "colocated", "--gpu", "T4"]).i6pn_schedule_timeout_min == 10.0
    assert launch.parse_args(["--region", "any"]).region == ""
    assert launch.parse_args(["--region", "us-east"]).region == "us-east"
    assert launch.parse_args([]).region is None


def test_normalize_placement() -> None:
    assert launch.normalize_placement(None) == "tunnel"
    assert launch.normalize_placement(False) == "tunnel" and launch.normalize_placement(True) == "i6pn"
    assert launch.normalize_placement(" Colocated ") == "colocated"
    with pytest.raises(ValueError):
        launch.normalize_placement("cluster")


def test_spawn_colocated_is_one_call_for_both_parties() -> None:
    log: list[tuple[str, str, tuple[Any, ...]]] = []
    live = FakeDict(provider={"run_id": "old"})
    p = launch.spawn_pair(live, "colocated", 600, True, lookup=lambda app, name: FakeFunction(app, name, log),
                          local_linears="off")
    assert [(a, n) for a, n, _ in log] == [launch.COLOCATED_FUNCTION]
    assert log[0][2] == (p.run_id, 600, True, "off", 0)
    assert p.network == "colocated" and p.colocated and not p.i6pn and p.pcall is p.ccall
    assert "provider" not in live
    # cancelling the pair cancels the one call once
    launch.cancel_pair(live, p)
    assert p.pcall.cancelled and live["stop"] == p.run_id


def test_colocated_falls_back_to_the_tunnel_when_not_started() -> None:
    spawned: list[str] = []

    def spawn(live: Any, placement: str, ttl_s: int, e2e: bool) -> Any:
        spawned.append(placement)
        return _pair(placement, f"r{len(spawned)}")

    p = launch.launch_pair(FakeDict(), placement="colocated", ttl_s=60, e2e=True, schedule_timeout_s=600,
                           spawn=spawn, wait=lambda live, pair, t: ["provider", "customer"],
                           cancel=lambda live, pair: None)
    assert spawned == ["colocated", "tunnel"] and p.network == "tunnel"
    assert p.note.startswith("colocated: provider, customer not scheduled within 10 min")
    spawned.clear()
    p = launch.launch_pair(FakeDict(), placement="colocated", ttl_s=60, e2e=True, schedule_timeout_s=600,
                           spawn=spawn, wait=lambda live, pair, t: [], cancel=lambda live, pair: None)
    assert spawned == ["colocated"] and p.colocated


def test_wait_scheduled_for_colocated_needs_both_markers_of_the_run() -> None:
    clock = _Clock()
    pair = launch.Pair("r1", "colocated", FakeCall("x"), None, 0.0)
    pair.ccall = pair.pcall
    live = FakeDict(provider_boot={"run_id": "r1"}, customer_boot={"run_id": "r1"})
    assert launch.wait_scheduled(live, pair, 60, clock=clock, sleep=clock.sleep) == []


def test_deploy_adds_the_colocated_app_and_region(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[tuple[str, dict[str, str]]] = []
    monkeypatch.setattr(launch.subprocess, "run", lambda cmd, check, env: seen.append((Path(cmd[-1]).name, env)))
    monkeypatch.setenv("PGW_MODAL_REGION", "us-east")
    launch.deploy("T4", None, "colocated", "")
    assert [f for f, _ in seen] == ["provider.py", "customer.py", "colocated.py"]  # the tunnel stays deployed
    assert all(e["PGW_MODAL_REGION"] == "" and e["PGW_MODAL_GPU"] == "T4" for _, e in seen)
    seen.clear()
    launch.deploy(None, None, "i6pn")
    assert [f for f, _ in seen] == ["provider.py", "customer.py"]
    assert all(e["PGW_MODAL_REGION"] == "us-east" for _, e in seen)


def test_run_e2e_colocated_prints_the_service_log_from_the_one_result(monkeypatch: pytest.MonkeyPatch,
                                                                     capsys: pytest.CaptureFixture[str]) -> None:
    res = _res(True, [0.25, 0.5], 0.4)
    res["service_log_tail"] = "SERVICE-LOG"
    res["e2e"]["trip_latency"]["path"] = "loopback"
    res["e2e"]["speed"] = {"measured_rtt_ms": 0.3, "rtt_path": "loopback", "local_linears": {"mode": "auto"},
                           "warm_tokens_per_s": 9.5}

    def fake_launch(live: Any, **kw: Any) -> Any:
        assert kw["placement"] == "colocated"
        c = FakeCall("one")
        return launch.Pair("r1", "colocated", c, c, 0.0)

    monkeypatch.setattr(launch, "launch_pair", fake_launch)
    monkeypatch.setattr(launch, "wait_e2e", lambda live, pair, ttl: res)
    monkeypatch.setattr(launch, "stop", lambda: None)
    code = launch.run_e2e(FakeDict(), launch.parse_args(["--e2e", "--placement", "colocated", "--max-trip-ms", "1"]),
                          600)
    out = capsys.readouterr().out
    assert code == 0
    assert "SERVICE-LOG" in out and "SERVICE-LOG" not in out.split("---- gateway log tail")[0]
    assert "trip latency (colocated via loopback, warm chat)" in out
    assert "start-up rtt 0.3 ms over loopback" in out and "9.5 tokens/s" in out


def test_path_label() -> None:
    pair = _pair("tunnel")
    assert launch.path_label({}, pair) == "tunnel"
    res = {"e2e": {"trip_latency": {"path": "public r500.modal.host"}}}
    assert launch.path_label(res, pair) == "tunnel via public r500.modal.host"
    assert launch.path_label({"e2e": {"network_path": "loopback"}}, _pair("colocated")) == "colocated via loopback"


def test_colocated_app_definition(monkeypatch: pytest.MonkeyPatch) -> None:
    """``colocated.py`` builds locally (no network), names the function the launcher spawns, writes both boot
    markers before anything slow, binds the service to loopback only and runs each party as its own user."""
    import ast

    monkeypatch.delenv("PGW_MODAL_GPU", raising=False)
    monkeypatch.delenv("PGW_MODAL_PROVIDER_CPUS", raising=False)
    mod = _load("colocated")
    assert hasattr(mod, launch.COLOCATED_FUNCTION[1]) and mod.app.name == launch.COLOCATED_FUNCTION[0]
    assert mod.PROVIDER_CPUS + mod.GATEWAY_CPUS == 16.0 and mod.PROVIDER_USER != mod.CUSTOMER_USER
    monkeypatch.setenv("PGW_MODAL_GPU", "T4")
    assert _load("colocated").PROVIDER_CPUS == 2.0  # with a GPU: 2 + 12 = 14 cores in all
    monkeypatch.setenv("PGW_MODAL_GPU", "L4")
    with pytest.raises(ValueError, match="L4"):
        _load("colocated")
    text = (MODAL_DIR / "colocated.py").read_text()
    fn = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == "pair")
    src = ast.get_source_segment(text, fn) or ""
    assert src.index("provider_boot") < src.index("import pgw_modal_boot")
    assert src.index("customer_boot") < src.index("import pgw_modal_boot")
    assert "start_provider_party(" in src and "user=puser" in src and mod.SEPARATE_USERS
    assert "start_gateway_party(" in src and "user=cuser" in src
    boot_text = (MODAL_DIR / "pgw_modal_boot.py").read_text()
    assert f'PROVIDER_USER, CUSTOMER_USER = "{mod.PROVIDER_USER}", "{mod.CUSTOMER_USER}"' in boot_text


def test_spec_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    """--spec-rows (runtime, default off), --model and --spec-draft (deploy time) and their checks."""
    for var in ("PGW_MODAL_MODEL", "PGW_MODAL_DRAFT", "PGW_MODAL_GPU"):
        monkeypatch.delenv(var, raising=False)
    a = launch.parse_args([])
    assert a.spec_rows == 0 and a.model == launch.DEFAULT_MODEL and a.spec_draft is False
    assert launch.parse_args(["--spec-rows", "3"]).spec_rows == 3
    for bad in (["--spec-rows", "1"], ["--spec-rows", "9"], ["--spec-draft"],  # the draft needs spec rows
                ["--spec-rows", "4", "--spec-draft"],  # the 0.5B drafting for itself
                ["--model", "qwen2.5-7b-instruct"]):
        with pytest.raises(SystemExit):
            launch.parse_args(bad)
    a = launch.parse_args(["--model", "qwen2.5-1.5b-instruct", "--spec-rows", "4", "--spec-draft"])
    assert a.model == "qwen2.5-1.5b-instruct" and a.spec_draft
    monkeypatch.setenv("PGW_MODAL_MODEL", "qwen2.5-3b-instruct")
    assert launch.parse_args([]).model == "qwen2.5-3b-instruct"


def test_deploy_passes_model_and_draft(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[dict[str, str]] = []
    monkeypatch.setattr(launch.subprocess, "run", lambda cmd, check, env: seen.append(env))
    monkeypatch.setenv("PGW_MODAL_DRAFT", "1")
    launch.deploy(None, None, model=launch.DEFAULT_MODEL, draft=False)
    assert all("PGW_MODAL_MODEL" not in e and "PGW_MODAL_DRAFT" not in e for e in seen)
    seen.clear()
    launch.deploy(None, None, model="qwen2.5-1.5b-instruct", draft=True)
    assert [(e["PGW_MODAL_MODEL"], e["PGW_MODAL_DRAFT"]) for e in seen] == [("qwen2.5-1.5b-instruct", "1")] * 2
    seen.clear()  # the co-located app gets the same deploy-time choices
    launch.deploy(None, None, "colocated", None, "qwen2.5-3b-instruct", True)
    assert [(e["PGW_MODAL_MODEL"], e["PGW_MODAL_DRAFT"]) for e in seen] == [("qwen2.5-3b-instruct", "1")] * 3


def test_spawn_pair_passes_spec_rows_and_budget() -> None:
    log: list[tuple[str, str, tuple[Any, ...]]] = []
    p = launch.spawn_pair(FakeDict(), False, 600, True, lookup=lambda app, name: FakeFunction(app, name, log),
                          spec_rows=4)
    assert log[1][2] == (p.run_id, 600, True, "auto", 4)
    assert launch.spec_budget(4, 41, 32) == {"plain": 73, "spec": 73 + 31 * 3}
    assert launch.spec_budget(1, 41, 32)["spec"] == 73
    log.clear()
    p = launch.spawn_pair(FakeDict(), "colocated", 600, True, lookup=lambda app, name: FakeFunction(app, name, log),
                          spec_rows=3)
    assert [(a, n) for a, n, _ in log] == [launch.COLOCATED_FUNCTION]
    assert log[0][2] == (p.run_id, 600, True, "auto", 3)


def test_colocated_serves_the_chosen_model_and_draft(monkeypatch: pytest.MonkeyPatch) -> None:
    """colocated.py packs the --model target and bakes the --spec-draft draft like provider.py / customer.py,
    passes --spec-rows to its gateway (the draft on the gateway's CPU: its process sees no GPU), and keeps the
    deploy-time choices in the image's environment for Modal's re-import inside the container."""
    import ast

    for var in ("PGW_MODAL_MODEL", "PGW_MODAL_DRAFT", "PGW_MODAL_GPU", "PGW_MODAL_PROVIDER_CPUS"):
        monkeypatch.delenv(var, raising=False)
    col = _load("colocated")
    assert (col.MODEL_NAME, col.DRAFT, col.MEMORY_MIB) == ("qwen2.5-0.5b-instruct", False, 6144 + 8192)
    monkeypatch.setenv("PGW_MODAL_MODEL", "qwen2.5-1.5b-instruct")
    monkeypatch.setenv("PGW_MODAL_DRAFT", "1")
    col = _load("colocated")
    assert col.MODEL_NAME == "qwen2.5-1.5b-instruct" and col.PACK.endswith("qwen2.5-1.5b-instruct")
    assert col.DRAFT and col.MEMORY_MIB == 6144 + 16384
    for var in ("PGW_MODAL_MODEL", "PGW_MODAL_DRAFT"):
        monkeypatch.delenv(var)
    for var, val in col.RUNTIME_ENV.items():
        monkeypatch.setenv(var, val)
    col2 = _load("colocated")
    assert (col2.MODEL_NAME, col2.PACK, col2.DRAFT, col2.MEMORY_MIB) == (col.MODEL_NAME, col.PACK, True,
                                                                         col.MEMORY_MIB)
    monkeypatch.setenv("PGW_MODAL_MODEL", "qwen2.5-7b-instruct")
    with pytest.raises(ValueError, match="PGW_MODAL_MODEL"):
        _load("colocated")
    text = (MODAL_DIR / "colocated.py").read_text()
    fn = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == "pair")
    src = ast.get_source_segment(text, fn) or ""
    assert "spec_rows: int = 0" in src
    assert "boot.spec_options(spec_rows, DRAFT_PACK if DRAFT else None, None)" in src and "**spec" in src
    assert "spec_rows=spec_rows" in src


def test_provider_serves_the_chosen_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PGW_MODAL_GPU", raising=False)
    monkeypatch.setenv("PGW_MODAL_MODEL", "qwen2.5-1.5b-instruct")
    prov = _load("provider")
    assert prov.MODEL_NAME == "qwen2.5-1.5b-instruct" and prov.MODEL_REPO == "Qwen/Qwen2.5-1.5B-Instruct"
    assert prov.PACK.endswith("qwen2.5-1.5b-instruct")
    monkeypatch.setenv("PGW_MODAL_MODEL", "qwen2.5-7b-instruct")
    with pytest.raises(ValueError, match="PGW_MODAL_MODEL"):
        _load("provider")
    monkeypatch.delenv("PGW_MODAL_MODEL")
    monkeypatch.setenv("PGW_MODAL_DRAFT", "1")
    assert _load("customer").DRAFT is True


def test_customer_cost_follows_the_target_and_draft(monkeypatch: pytest.MonkeyPatch) -> None:
    """A larger target or the draft gives the customer 16 GiB (customer.py MEMORY_MIB): the estimate follows."""
    for var in ("PGW_MODAL_MODEL", "PGW_MODAL_DRAFT", "PGW_MODAL_CUSTOMER_GPU"):
        monkeypatch.delenv(var, raising=False)
    base = launch.cost_per_h(None)["pgw-customer"]
    assert base == pytest.approx(4 * 0.0473 + 8 * 0.008)
    monkeypatch.setenv("PGW_MODAL_MODEL", "qwen2.5-1.5b-instruct")
    assert launch.customer_gib() == 16
    assert launch.cost_per_h(None)["pgw-customer"] == pytest.approx(base + 8 * 0.008)
    monkeypatch.delenv("PGW_MODAL_MODEL")
    monkeypatch.setenv("PGW_MODAL_DRAFT", "1")
    assert launch.customer_gib() == 16


def test_container_reimport_keeps_deploy_choices(monkeypatch: pytest.MonkeyPatch) -> None:
    """Modal re-imports provider.py / customer.py inside the container without the deploy shell's environment:
    the image's environment (RUNTIME_ENV) must make them evaluate the same there (target pack, published model
    name, draft on/off, draft device)."""
    for var in ("PGW_MODAL_MODEL", "PGW_MODAL_DRAFT", "PGW_MODAL_CUSTOMER_GPU", "PGW_MODAL_GPU"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("PGW_MODAL_MODEL", "qwen2.5-1.5b-instruct")
    monkeypatch.setenv("PGW_MODAL_DRAFT", "1")
    monkeypatch.setenv("PGW_MODAL_CUSTOMER_GPU", "T4")
    prov, cust = _load("provider"), _load("customer")
    for var in ("PGW_MODAL_MODEL", "PGW_MODAL_DRAFT", "PGW_MODAL_CUSTOMER_GPU"):
        monkeypatch.delenv(var)
    for var, val in {**prov.RUNTIME_ENV, **cust.RUNTIME_ENV}.items():  # the container sees only the image env
        monkeypatch.setenv(var, val)
    prov2, cust2 = _load("provider"), _load("customer")
    assert (prov2.MODEL_NAME, prov2.PACK) == (prov.MODEL_NAME, prov.PACK) == (
        "qwen2.5-1.5b-instruct", "/opt/packs/qwen2.5-1.5b-instruct")
    assert (cust2.DRAFT, cust2.CUSTOMER_GPU, cust2.MEMORY_MIB) == (True, "T4", cust.MEMORY_MIB)
    # the default deploy keeps the defaults in the container too
    for var in ("PGW_MODAL_MODEL", "PGW_MODAL_DRAFT", "PGW_MODAL_CUSTOMER_GPU"):
        monkeypatch.delenv(var)
    prov, cust = _load("provider"), _load("customer")
    for var, val in {**prov.RUNTIME_ENV, **cust.RUNTIME_ENV}.items():
        monkeypatch.setenv(var, val)
    prov2, cust2 = _load("provider"), _load("customer")
    assert prov2.MODEL_NAME == "qwen2.5-0.5b-instruct" and cust2.DRAFT is False and cust2.CUSTOMER_GPU is None


def test_lane_tls_is_colocated_only_and_reaches_the_pair(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PGW_MODAL_I6PN", raising=False)
    monkeypatch.delenv("PGW_MODAL_PLACEMENT", raising=False)
    monkeypatch.delenv("PGW_MODAL_CUSTOMER_GPU", raising=False)
    assert launch.parse_args([]).lane == "stream"  # default unchanged off the co-located placement
    assert launch.parse_args([]).local_linears == "auto"
    co = launch.parse_args(["--placement", "colocated"])  # co-located defaults (measured on Modal)
    assert (co.lane, co.local_linears) == ("tls", "attention_down")
    with pytest.raises(SystemExit):  # the provider's lane port is on its loopback only
        launch.parse_args(["--lane", "tls"])
    args = launch.parse_args(["--placement", "colocated", "--lane", "tls", "--local-linears", "auto"])
    assert args.lane == "tls"
    log: list[tuple[str, str, tuple[Any, ...]]] = []
    monkeypatch.setattr(launch.modal.Function, "from_name", lambda app, name: FakeFunction(app, name, log))
    spawn = launch._spawner(args)
    p = spawn(FakeDict(), "colocated", 600, True)
    # co-located default --float-threads 1: the gateway's torch threads pinned (the reference follows, colocated.py)
    assert log[-1][:2] == launch.COLOCATED_FUNCTION and log[-1][2] == (
        p.run_id, 600, True, "auto", 0, "tls", {"OMP_NUM_THREADS": "1", "PGW_FLOAT_OPS": "native"}, {}, 0.0)
    args0 = launch.parse_args(["--placement", "colocated", "--float-threads", "0", "--float-ops", "torch"])
    launch._spawner(args0)(FakeDict(), "colocated", 600, True)
    assert log[-1][2][1:] == (600, True, "attention_down", 0, "tls")  # 0 = each process's default count
    with pytest.raises(SystemExit):
        launch.parse_args(["--float-threads", "1"])  # co-located only
    spawn(FakeDict(), "tunnel", 600, True)  # the tunnel fallback keeps the stream lane
    assert log[-1][2][-2:] == ("auto", 0)
    p = launch.spawn_pair(FakeDict(), "colocated", 600, False, lookup=lambda app, name: FakeFunction(app, name, log))
    assert log[-1][2] == (p.run_id, 600, False, "auto", 0)  # --lane stream: the call is unchanged


def test_colocated_gateway_env_skips_only_fixed_local_kinds() -> None:
    boot = _load("pgw_modal_boot")
    assert boot.colocated_gateway_env("attention_down")["PGW_FACTORY_SKIP_LINEAR_KINDS"] == "qkv,o,down"
    assert boot.colocated_gateway_env("attention")["PGW_FACTORY_SKIP_LINEAR_KINDS"] == "qkv,o"
    for mode in ("auto", "auto_kinds", "off"):  # decided at start-up or nothing local: every linear's masks
        assert "PGW_FACTORY_SKIP_LINEAR_KINDS" not in boot.colocated_gateway_env(mode)
    env = boot.colocated_gateway_env("attention_down")  # prompt-pass settings of the co-located demo
    assert env["PGW_PROMPT_PREFETCH"] == "1" and env["PGW_PIPELINE_BLOCKS"] == "1"
    assert boot.GATEWAY_GPU_FACTORY_ENV["PGW_LOCAL_GPU_PREFILL_ROWS"] == "64"
    with pytest.raises(ValueError):
        boot.checked_env({"LD_PRELOAD": "x"})
    assert boot.checked_env({"PGW_X": 1}) == {"PGW_X": "1"}


def test_env_pairs_parse_and_refuse() -> None:
    assert launch.parse_env_pairs(["PGW_A=1", "OMP_NUM_THREADS=2"]) == {"PGW_A": "1", "OMP_NUM_THREADS": "2"}
    for bad in (["PATH=/x"], ["PGW_A"]):
        with pytest.raises(ValueError):
            launch.parse_env_pairs(bad)


def test_spec_mask_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PGW_MODAL_SPEC_MASK", raising=False)
    assert launch.parse_args([]).spec_mask == ""
    assert launch.parse_args(["--spec-rows", "4", "--spec-mask", "4/160"]).spec_mask == "4/160"
    assert launch.parse_args(["--spec-rows", "2", "--spec-mask", "1.5"]).spec_mask == "1.5"
    for bad in (["--spec-mask", "2"], ["--spec-rows", "4", "--spec-mask", "5"],
                ["--spec-rows", "4", "--spec-mask", "x"], ["--spec-rows", "4", "--spec-mask", "0.5"]):
        with pytest.raises(SystemExit):
            launch.parse_args(bad)


def test_float_ops_native_is_the_colocated_default() -> None:
    assert launch.parse_args(["--placement", "colocated"]).float_ops == "native"
    assert launch.parse_args([]).float_ops == "torch"
    assert launch.parse_args(["--placement", "colocated", "--float-ops", "torch"]).float_ops == "torch"
    with pytest.raises(SystemExit):
        launch.parse_args(["--float-ops", "native"])
