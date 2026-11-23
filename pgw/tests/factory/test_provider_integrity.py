"""Fail closed on provider integrity failures (pks-v3 adversarial review, F3): count, alert, halt, operator clear."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from pgw.core import RING_K40_A12
from pgw.core.errors import IntegrityError, ProtocolError, ServiceUnavailableError
from pgw.factory import FactoryScheduler, FactorySettings, MaskGenerator
from pgw.factory.scheduler import (
    HALT_FILE,
    STATE_FILE,
    ProviderIntegrityHalt,
    clear_halt_at,
    is_provider_integrity_failure,
)

from .conftest import FakeMaskStore, FakePack


def _sched(tmp_path: Path, store: Any, pack: FakePack, *, remote: bool = True, **kw: Any) -> FactoryScheduler:
    s = FactorySettings(**{"pack_path": tmp_path / "pack", "segment_slots": 50, "chunk_slots": 16,
                           "low_water_slots": 50, "target_slots": 50, "poll_interval_s": 0.01,
                           "state_dir": tmp_path / "state", "error_backoff_s": 0.01, **kw})
    g = MaskGenerator(pack, RING_K40_A12)
    if remote:
        g.remote = object()  # stands for an encrypted-mask backend (generation is replaced below)
    return FactoryScheduler(s, store, {g.key: g}, metrics=False)


def _fail_with(sch: FactoryScheduler, exc: BaseException) -> list[int]:
    calls: list[int] = []
    gen = next(iter(sch.generators.values()))

    def gis(*a: Any, **k: Any) -> str:
        calls.append(1)
        raise exc

    gen.generate_into_store = gis  # type: ignore[method-assign]
    return calls


def test_classification() -> None:
    enc, local = type("G", (), {"remote": object()})(), type("G", (), {"remote": None})()
    fused2 = type("G", (), {"he": object()})()
    assert is_provider_integrity_failure(IntegrityError("x"), enc)
    assert is_provider_integrity_failure(IntegrityError("x"), fused2)
    assert not is_provider_integrity_failure(IntegrityError("x"), local)  # own device fault: retried
    assert is_provider_integrity_failure(ProtocolError("bad reply", details={"reply": True}), enc)
    assert not is_provider_integrity_failure(ProtocolError("refused"), enc)
    for exc in (ServiceUnavailableError("down"), TimeoutError(), ConnectionResetError(), OSError("x")):
        assert not is_provider_integrity_failure(exc, enc)  # transport: retried
    assert not is_provider_integrity_failure(ProviderIntegrityHalt("h"), enc)


def test_wire_marks_malformed_replies() -> None:
    from pgw.hemask.wire import MaskgenItem, MaskgenRequest, decode_maskgen_response

    req = MaskgenRequest("a" * 32, "f" * 64, (MaskgenItem("x", 256, b"\0" * 256 * 8, 0, 64),), 1, (1,))
    with pytest.raises(ProtocolError) as ei:
        decode_maskgen_response(b"\x81\xa1x\x01", req)
    assert ei.value.details.get("reply") is True


def test_halts_after_limit_and_success_does_not_reset(tmp_path: Path, fake_pack: FakePack) -> None:
    store = FakeMaskStore()
    sch = _sched(tmp_path, store, fake_pack)
    gen = next(iter(sch.generators.values()))
    orig = MaskGenerator.generate_into_store.__get__(gen)
    calls = _fail_with(sch, IntegrityError("provider products failed the self-check"))
    for i in (1, 2):
        with pytest.raises(IntegrityError) as ei:
            sch.run_once()
        assert not isinstance(ei.value, ProviderIntegrityHalt)
        assert sch.state.provider_integrity_failures == i and sch.health()[0] is True
    def honest(*a: Any, **k: Any) -> str:
        rem, gen.remote = gen.remote, None  # local products stand in for an honest provider's
        try:
            return str(orig(*a, **k))
        finally:
            gen.remote = rem

    gen.generate_into_store = honest  # type: ignore[method-assign]
    assert len(sch.run_once()) == 1  # an honest segment in between: the count is kept
    assert sch.state.provider_integrity_failures == 2
    store.consume(fake_pack.fingerprint, 50)
    calls = _fail_with(sch, IntegrityError("again"))
    with pytest.raises(ProviderIntegrityHalt):
        sch.run_once()
    assert len(calls) == 1 and sch.state.halted is not None and (tmp_path / "state" / HALT_FILE).exists()
    ok, h = sch.health()
    assert not ok and h["halted"] and h["provider_integrity_failures"] == 3
    st = sch.status()
    assert st["halted"] and st["provider_integrity_failures"] == 3 and st["max_provider_integrity_failures"] == 3
    with pytest.raises(ProviderIntegrityHalt):  # refuses to produce, without asking the provider again
        sch.run_once()
    assert len(calls) == 1
    sch.close()
    # a restart (new process) stays halted, even if the state file was lost
    (tmp_path / "state" / STATE_FILE).unlink()
    sch2 = _sched(tmp_path, store, fake_pack)
    with pytest.raises(ProviderIntegrityHalt):
        sch2.run_once()
    sch2.close()
    # operator action
    cleared = clear_halt_at(tmp_path / "state", "provider rotated")
    assert cleared and cleared[0]["halted"]
    sch3 = _sched(tmp_path, store, fake_pack, remote=False)
    assert len(sch3.run_once()) == 1 and sch3.state.provider_integrity_failures == 0
    assert sch3.state.provider_integrity_log[-1]["cleared"] is True
    sch3.close()


def test_run_forever_fails_closed(tmp_path: Path, fake_pack: FakePack) -> None:
    sch = _sched(tmp_path, FakeMaskStore(), fake_pack)
    calls = _fail_with(sch, IntegrityError("tampered"))
    err: list[BaseException] = []

    def run() -> None:
        try:
            sch.run_forever(threading.Event())
        except BaseException as exc:
            err.append(exc)

    th = threading.Thread(target=run, daemon=True)
    th.start()
    th.join(20)
    assert not th.is_alive() and len(err) == 1 and isinstance(err[0], ProviderIntegrityHalt)
    assert len(calls) == 3 and not sch.lock.held


@pytest.mark.parametrize("remote,exc", [(True, ServiceUnavailableError("provider down")),
                                        (True, TimeoutError("read timed out")),
                                        (False, IntegrityError("local GPU fault"))])
def test_transport_and_local_errors_keep_retrying(tmp_path: Path, fake_pack: FakePack, remote: bool,
                                                  exc: BaseException) -> None:
    sch = _sched(tmp_path, FakeMaskStore(), fake_pack, remote=remote)
    calls = _fail_with(sch, exc)
    stop = threading.Event()
    th = threading.Thread(target=sch.run_forever, args=(stop,), daemon=True)
    th.start()
    deadline = 200
    while len(calls) < 6 and deadline:
        stop.wait(0.05)
        deadline -= 1
    stop.set()
    sch._wake.set()
    th.join(10)
    assert not th.is_alive() and len(calls) >= 6
    assert sch.state.provider_integrity_failures == 0 and sch.halted() is None


def test_limit_setting_and_cli(tmp_path: Path, fake_pack: FakePack) -> None:
    from pgw.factory.cli import factory_group

    sch = _sched(tmp_path, FakeMaskStore(), fake_pack, max_provider_integrity_failures=1)
    _fail_with(sch, ProtocolError("malformed reply", details={"reply": True}))
    with pytest.raises(ProviderIntegrityHalt):
        sch.run_once()
    sch.close()
    runner = CliRunner()
    res = runner.invoke(factory_group, ["clear-halt", "--state-dir", str(tmp_path / "state")])
    assert res.exit_code != 0 and "--yes" in res.output
    assert (tmp_path / "state" / HALT_FILE).exists()
    res = runner.invoke(factory_group, ["clear-halt", "--state-dir", str(tmp_path / "state"), "--yes",
                                        "--note", "checked"])
    assert res.exit_code == 0 and "halted=True" in res.output
    assert not (tmp_path / "state" / HALT_FILE).exists()
    res = runner.invoke(factory_group, ["clear-halt", "--state-dir", str(tmp_path / "state"), "--yes"])
    assert res.exit_code == 0 and "no halt" in res.output
    with pytest.raises(ValueError):
        FactorySettings(pack_path=tmp_path, max_provider_integrity_failures=0)


def test_fleet_worker_halt_stops_every_worker(tmp_path: Path) -> None:
    from pgw.factory.fleet import FleetScheduler, ProductionLedger
    from pgw.maskstore import FileKeyProvider, MaskStore

    pack = FakePack()
    root = tmp_path / "state" / "fleet"
    kp = FileKeyProvider(tmp_path / "kek", create=True)
    store = MaskStore.open(tmp_path / "store", kp)
    scheds = []
    for k in range(2):
        s = FactorySettings(pack_path=tmp_path, segment_slots=16, low_water_slots=10, target_slots=20,
                            state_dir=root / f"w{k}", max_provider_integrity_failures=1)
        g = MaskGenerator(pack, RING_K40_A12)
        g.remote = object()
        scheds.append(FleetScheduler(s, store, {g.key: g}, metrics=False, ledger=ProductionLedger(root, f"w{k}")))
    _fail_with(scheds[0], IntegrityError("tampered"))
    calls1 = _fail_with(scheds[1], IntegrityError("tampered"))
    with pytest.raises(ProviderIntegrityHalt):
        scheds[0].run_once()
    assert (root / HALT_FILE).exists()
    with pytest.raises(ProviderIntegrityHalt):  # the peer stops before asking the provider
        scheds[1].run_once()
    assert calls1 == []
    for sc in scheds:
        sc.close()
    cleared = clear_halt_at(tmp_path / "state")
    assert {c["state_dir"] for c in cleared} == {str(root), str(root / "w0")}
    assert not (root / HALT_FILE).exists() and not (root / "w0" / HALT_FILE).exists()
    store.close()
