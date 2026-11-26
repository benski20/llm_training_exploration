"""Weight-free factory (``factory_weights = "stream"``, :mod:`pgw.factory.checkstock`): pre-drawn check material
gives bit-identical segments, wrong provider products are still caught with no ``W8`` on the client, consumption
survives crashes without reuse, tampered or rolled-back material is rejected, low-water refill, fail-closed
exhaustion. Security argument: research/reviews/weight-free-stock-security.md."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from pgw.core.errors import ConfigError, IntegrityError
from pgw.core.ring import RingParams
from pgw.factory import MaskGenerator
from pgw.factory.checkstock import CheckStock, CheckStockEmpty, CheckStockError
from tests.hemask.test_pks import NATIVE, _LayerPack, _LocalProvider, _TamperingProvider

pytestmark = NATIVE
ROOT = Path(__file__).resolve().parents[2]
RING = RingParams.from_name("K32A12C27")
SEED = bytes(range(1, 33))


class _NoWeights(_LayerPack):
    """The client's view in stream mode: shapes and a fingerprint, no int8 weights anywhere."""

    def __init__(self, full: _LayerPack):
        self.w = {k: np.empty(0) for k in full.w}
        self._shapes = {k: w.shape for k, w in full.w.items()}

    def linear_specs(self) -> list[Any]:  # type: ignore[override]
        from pgw.core.types import LinearId, LinearSpec

        return [LinearSpec(LinearId(0, k.split(".")[1]), *s) for k, s in self._shapes.items()]

    def load_linear(self, lid: Any, device: str = "cpu") -> Any:
        raise AssertionError("a weight-free factory read W8")


class _Source:
    """A pass's W8 source over a dict, with the same contract as the real ones (checked before it is returned)."""

    def __init__(self, pack: _LayerPack, *, fail: str | None = None, delay_s: float = 0.0, log: list[str] | None = None):
        self.pack, self.fail, self.delay_s, self.bytes = pack, fail, delay_s, 0
        self.log = log if log is not None else []

    def get(self, key: str) -> np.ndarray:
        if self.delay_s:
            time.sleep(self.delay_s)
        if self.fail == key:
            raise IntegrityError(f"{key}: bytes do not match the pinned manifest")
        self.log.append(key)
        w = self.pack.w[key]
        self.bytes += w.nbytes
        return w

    def close(self) -> None:
        pass


@pytest.fixture
def env(tmp_path: Path) -> dict[str, Any]:
    from pgw.hemask.pks_backend import PKSHintStore, PKSMaskBackend, RemotePKS
    from pgw.maskstore import FileKeyProvider

    pack = _LayerPack()
    hints = PKSHintStore.open_or_build(tmp_path / "hint", pack)
    out = {k: w.shape[0] for k, w in pack.w.items()}

    def backend(provider: Any = None) -> Any:
        remote = RemotePKS(provider or _LocalProvider(pack), pack.fingerprint, out, hints.a_seed)
        return PKSMaskBackend(pack, RING, hints, remote, groups_per_call=1)

    return {"pack": pack, "backend": backend, "kp": FileKeyProvider(tmp_path / "kek", create=True), "tmp": tmp_path}


def _gen(env: dict[str, Any], *, check_field: str | None = "p31", provider: Any = None, pack: Any = None) -> Any:
    return MaskGenerator(pack or _NoWeights(env["pack"]), RING, device="cpu", remote=env["backend"](provider),
                         check_field=check_field, weights="stream")


def _stock(env: dict[str, Any], gen: Any, name: str = "stock", **kw: Any) -> CheckStock:
    src = kw.pop("source", None) or (lambda: _Source(env["pack"]))
    base: dict[str, Any] = {"target_segments": 4, "low_water_segments": 1, "target_private": 3,
                            "low_water_private": 1, "background": False, "wait_s": 2.0}
    base.update(kw)
    return CheckStock(env["tmp"] / name, env["kp"], fingerprint=gen.fingerprint, ring=RING, check_field=gen.check_field,
                      checks=1, linears=gen.stock_linears(), source=src, **base)


def _store(env: dict[str, Any], name: str) -> Any:
    from pgw.maskstore import MaskStore

    return MaskStore.open(env["tmp"] / name, env["kp"])


def _contents(ms: Any, pack: Any) -> dict[str, tuple[torch.Tensor, np.ndarray]]:
    with ms.reserve(pack.fingerprint, RING, 300) as res:
        return {s.id.key: (res.take(s.id, 300)[1].to(torch.int64) & 0xFFFFFFFF,
                           np.array(res.checks_compact(s.id)[2], dtype=np.int64)) for s in pack.linear_specs()}


# ---- bit-identical -------------------------------------------------------------------------------------------

@pytest.mark.parametrize("check_field", [None, "p31"])
def test_segments_bit_identical_to_pack_mode(env: dict[str, Any], check_field: str | None) -> None:
    """For the same seed, a stream-mode segment (seed and uW pre-drawn at a pass, private set from the stock) holds
    exactly the masks and check vectors of a pack-mode segment, and of local generation."""
    pack = env["pack"]
    ref = MaskGenerator(pack, RING, device="cpu", remote=env["backend"](), check_field=check_field)
    local = MaskGenerator(pack, RING, device="cpu", check_field=check_field)
    a, b, c = _store(env, "a"), _store(env, "b"), _store(env, "c")
    try:
        ref.generate_into_store(a, 300, seed=SEED, checks=1)
        local.generate_into_store(c, 300, seed=SEED, checks=1)
        gen = _gen(env, check_field=check_field)
        stock = _stock(env, gen)
        stock.refill(seeds=[SEED, os.urandom(32)])  # several seeds in one stacked product
        gen.attach_check_source(stock)
        gen.generate_into_store(b, 300)
        assert gen.last_stats.self_checked_rows == 300 * 4  # every row, 40 bits, with the pre-drawn private set
        ga, gb, gc = _contents(a, pack), _contents(b, pack), _contents(c, pack)
        for key in ga:
            assert torch.equal(ga[key][0], gb[key][0]) and torch.equal(ga[key][0], gc[key][0]), key
            assert np.array_equal(ga[key][1], gb[key][1]) and np.array_equal(ga[key][1], gc[key][1]), key
            assert ga[key][1].any()
    finally:
        for s in (a, b, c):
            s.close()


def test_private_set_equals_on_demand_draw(env: dict[str, Any]) -> None:
    """A pre-drawn private set is exactly what _private_check draws for the same v seed: v from the seed (stream "v"),
    v W8 from the same kernel, in the same layout and dtype."""
    from pgw.core.prng import SeedExpander
    from pgw.factory.kernels import ring_vecmat

    gen = _gen(env)
    stock = _stock(env, gen)
    stock.refill()
    item = stock.take_private()
    for lin in stock.linears:
        cu, cw = item.check_vectors(lin, RING)
        v = SeedExpander(bytes(item.v_seed)).ring_uniform((lin.private_vectors, lin.out_features), 32, "v")
        vw = ring_vecmat(v, torch.from_numpy(env["pack"].w[lin.key]), RING)
        assert cu.dtype == np.uint32 and cu.flags.c_contiguous and cw.flags.c_contiguous
        assert np.array_equal(cu, v.t().numpy().astype(np.uint32)) and np.array_equal(cw, vw.t().numpy().astype(np.uint32))
        assert lin.private_vectors == gen._private_vectors(next(s for s in gen.specs if s.id.key == lin.key)) >= 7


def test_weight_free_generator_refuses_without_material(env: dict[str, Any]) -> None:
    gen = _gen(env)
    ms = _store(env, "s")
    try:
        with pytest.raises(ConfigError, match="no check stock"):
            gen.generate_into_store(ms, 300)
        stock = _stock(env, gen)
        gen.attach_check_source(stock)
        with pytest.raises(ConfigError, match="seeds from its check stock"):
            gen.generate_into_store(ms, 300, seed=SEED)
        with pytest.raises(ConfigError, match="pre-drawn"):
            list(gen.generate(SEED, 300, checks=1))
    finally:
        ms.close()
    with pytest.raises(ConfigError, match="encrypted"):
        MaskGenerator(_NoWeights(env["pack"]), RING, device="cpu", weights="stream")
    other = _stock(env, _gen(env, check_field=None), name="ring-stock")
    with pytest.raises(ConfigError, match="another pack, ring or check field"):
        _gen(env).attach_check_source(other)


# ---- a wrong provider product is caught with no W8 on the client ------------------------------------------------

@pytest.mark.parametrize("shift", [1 << 31, 1 << 35])
def test_wrong_provider_result_caught_with_the_pack_deleted(env: dict[str, Any], shift: int) -> None:
    """The client holds no W8 (_NoWeights asserts if one is read); the provider shifts one mask entry; the every-row
    self-check with the pre-drawn private set rejects the segment; the set is burned and the next segment takes a
    fresh one; nothing is committed."""
    gen = _gen(env, provider=_TamperingProvider(env["pack"], shift))
    stock = _stock(env, gen)
    stock.refill()
    gen.attach_check_source(stock)
    ms = _store(env, "s")
    try:
        assert stock.counts() == {"segment": 4, "private": 3}
        with pytest.raises(IntegrityError):
            gen.generate_into_store(ms, 300)
        assert stock.counts() == {"segment": 3, "private": 2}  # both burned: never used again
        assert gen._priv_item is None
        with pytest.raises(IntegrityError):
            gen.generate_into_store(ms, 300)
        assert stock.counts() == {"segment": 2, "private": 1}
        assert ms.available_slots(env["pack"].fingerprint, RING) == 0
    finally:
        ms.close()


def test_private_set_kept_across_accepted_segments(env: dict[str, Any]) -> None:
    """As on demand: one private set per worker start, kept while segments are accepted."""
    gen = _gen(env)
    stock = _stock(env, gen)
    stock.refill()
    gen.attach_check_source(stock)
    ms = _store(env, "s")
    try:
        gen.generate_into_store(ms, 300)
        gen.generate_into_store(ms, 300)
        assert stock.counts() == {"segment": 2, "private": 2}
    finally:
        ms.close()


# ---- durable single use -------------------------------------------------------------------------------------------

_CRASH = """
import os, sys
sys.path[:0] = [sys.argv[1], sys.argv[2]]
from pathlib import Path
import pgw.factory.checkstock as cs
from pgw.core.ring import RingParams
from pgw.maskstore import FileKeyProvider
lin = [cs.StockLinear(*x) for x in __import__("json").loads(sys.argv[5])]
st = cs.CheckStock(Path(sys.argv[3]), FileKeyProvider(Path(sys.argv[4])), fingerprint="f" * 64,
                   ring=RingParams.from_name("K32A12C27"), check_field="p31", checks=1, linears=lin,
                   target_segments=4, low_water_segments=1, target_private=3, low_water_private=1, background=False)
mode = sys.argv[6]
if mode == "take":
    real = cs._read_item
    def crash(path, *a):
        out = real(path, *a)
        print(out[0].hex(), flush=True)  # the seed this process got: then it dies before using it
        os._exit(9)
    cs._read_item = crash
    st.take_segment()
else:
    class Src:
        bytes = 0
        n = 0
        def get(self, key):
            Src.n += 1
            if Src.n == 3:
                os._exit(9)  # killed in the middle of a pass
            import numpy as np
            m, n = next((x.out_features, x.in_features) for x in lin if x.key == key)
            return np.zeros((m, n), dtype=np.int8)
        def close(self):
            pass
    st.source = Src
    st.refill()
"""


def _crash(env: dict[str, Any], stock: CheckStock, mode: str) -> subprocess.CompletedProcess[str]:
    lin = json.dumps([x.as_list() for x in stock.linears])
    return subprocess.run([sys.executable, "-c", _CRASH, str(ROOT / "src"), str(ROOT), str(stock.root),  # noqa: S603
                           str(env["tmp"] / "kek"), lin, mode], capture_output=True, text=True, timeout=300)


def test_consumption_survives_a_crash_without_reuse(env: dict[str, Any]) -> None:
    gen = _gen(env)
    stock = _stock(env, gen)
    seeds = [os.urandom(32) for _ in range(4)]
    stock.refill(seeds=seeds)
    p = _crash(env, stock, "take")
    assert p.returncode == 9, p.stderr[-2000:]
    taken = bytes.fromhex(p.stdout.strip())
    assert taken == seeds[0]
    stock2 = _stock(env, gen)  # restart
    assert stock2.counts()["segment"] == 3
    assert not list(stock2.root.glob("seg-000000000000*"))  # the taken item's file is gone
    got = [bytes(stock2.take_segment().seed) for _ in range(3)]
    assert got == seeds[1:] and taken not in got  # never handed out twice
    with pytest.raises(CheckStockEmpty):
        stock2.take_segment(timeout_s=0.1)
    assert stock2.status()["taken_segments"] == 4


def test_a_pass_killed_midway_leaves_nothing(env: dict[str, Any]) -> None:
    gen = _gen(env)
    stock = _stock(env, gen)
    stock.refill()
    stock.take_segment()
    p = _crash(env, stock, "pass")
    assert p.returncode == 9, p.stderr[-2000:]
    assert list(stock.root.glob(".pass-*"))  # temporary item files of the dead pass
    stock2 = _stock(env, gen)
    assert stock2.counts() == {"segment": 3, "private": 3}  # nothing of the dead pass was committed
    assert not list(stock2.root.glob(".pass-*"))  # and its files are deleted
    stock2.refill()
    assert stock2.counts() == {"segment": 4, "private": 3}


# ---- tamper / rollback ----------------------------------------------------------------------------------------

def _flip(path: Path, offset: int) -> None:
    b = bytearray(path.read_bytes())
    b[offset] ^= 0x01
    path.write_bytes(bytes(b))


def test_tampered_item_is_rejected_and_burned(env: dict[str, Any]) -> None:
    gen = _gen(env)
    stock = _stock(env, gen)
    seeds = [os.urandom(32) for _ in range(4)]
    stock.refill(seeds=seeds)
    items = sorted(stock.root.glob("seg-*.pgwc"))
    _flip(items[0], -5)  # ciphertext of the last record
    with pytest.raises(CheckStockError, match="failed authentication"):
        stock.take_segment()
    assert bytes(stock.take_segment().seed) == seeds[1]  # burned; the next one is fine
    # a swapped item (another slot's file under this slot's name) is refused by its index digest
    os.replace(items[3], items[2])
    with pytest.raises(CheckStockError, match="does not match its index entry"):
        stock.take_segment()
    assert stock.counts()["segment"] == 1
    # a header edit is refused the same way (the header is the AAD and its digest is in the index)
    raw = items[3].parent
    priv = sorted(raw.glob("priv-*.pgwc"))[0]
    _flip(priv, 20)
    with pytest.raises(CheckStockError):
        stock.take_private()


def test_tampered_index_discards_the_stock(env: dict[str, Any]) -> None:
    gen = _gen(env)
    stock = _stock(env, gen)
    stock.refill()
    idx = json.loads((stock.root / "index.json").read_text())
    idx["segment"] = idx["segment"] * 2  # replay every item twice
    (stock.root / "index.json").write_text(json.dumps(idx))
    with pytest.raises(CheckStockError, match="index failed authentication"):
        stock.take_segment()
    assert not list(stock.root.glob("*.pgwc"))  # discarded, never used
    again = _stock(env, gen)
    assert again.counts() == {"segment": 0, "private": 0}


def test_rolled_back_index_is_refused(env: dict[str, Any]) -> None:
    """Restoring an older index and its item files (no key needed) is detected by the witness."""
    gen = _gen(env)
    stock = _stock(env, gen)
    stock.refill()
    snap = env["tmp"] / "snap"
    shutil.copytree(stock.root, snap)
    stock.take_segment()
    stock.take_private()
    shutil.rmtree(stock.root)
    shutil.copytree(snap, stock.root)  # the snapshot from before the takes
    again = _stock(env, gen)
    assert again.counts() == {"segment": 0, "private": 0}  # refused and discarded
    # without the witness the index is refused too
    again.refill()
    again.witness.unlink()
    assert _stock(env, gen).counts() == {"segment": 0, "private": 0}


def test_another_kek_cannot_use_the_stock(env: dict[str, Any]) -> None:
    from pgw.maskstore import FileKeyProvider

    gen = _gen(env)
    _stock(env, gen).refill()
    other = CheckStock(env["tmp"] / "stock", FileKeyProvider(env["tmp"] / "kek2", create=True),
                       fingerprint=gen.fingerprint, ring=RING, check_field="p31", checks=1,
                       linears=gen.stock_linears(), background=False)
    assert other.counts() == {"segment": 0, "private": 0}


def test_a_bad_stream_commits_nothing(env: dict[str, Any]) -> None:
    """A tensor that fails its check against the pin fails the whole pass: no item from it is ever listed."""
    gen = _gen(env)
    stock = _stock(env, gen, source=lambda: _Source(env["pack"], fail="0.down"))
    with pytest.raises(IntegrityError):
        stock.refill()
    assert stock.counts() == {"segment": 0, "private": 0}
    assert not list(stock.root.glob(".pass-*")) and not list(stock.root.glob("*.pgwc"))
    assert "0.down" in stock.status()["last_pass_error"]


def test_files_are_private_and_hold_no_plaintext(env: dict[str, Any]) -> None:
    gen = _gen(env)
    stock = _stock(env, gen)
    stock.refill(seeds=[SEED])
    assert (stock.root.stat().st_mode & 0o777) == 0o700
    for f in stock.root.iterdir():
        assert (f.stat().st_mode & 0o077) == 0, f
        assert SEED not in f.read_bytes(), f
    st = json.dumps(stock.status())
    assert SEED.hex() not in st


# ---- refill / exhaustion --------------------------------------------------------------------------------------

def test_low_water_starts_a_background_pass(env: dict[str, Any]) -> None:
    gen = _gen(env)
    log: list[str] = []
    stock = _stock(env, gen, background=True, target_segments=6, low_water_segments=2,
                   source=lambda: _Source(env["pack"], log=log))
    stock.refill()
    n0 = len(log)
    gen.attach_check_source(stock)
    ms = _store(env, "s")
    try:
        for _ in range(4):  # 6 -> 2: still at the low-water mark, no pass
            gen.generate_into_store(ms, 300)
        assert len(log) == n0
        gen.generate_into_store(ms, 300)  # 1 < 2: the next pass starts in the background
        deadline = time.monotonic() + 60
        while stock.counts()["segment"] < 6 and time.monotonic() < deadline:
            time.sleep(0.05)
        assert stock.counts()["segment"] == 6 and len(log) == 2 * n0
        assert stock.status()["passes"] == 2
    finally:
        stock.close()
        ms.close()


def test_exhaustion_without_a_pass_fails_closed(env: dict[str, Any]) -> None:
    gen = _gen(env)
    stock = _stock(env, gen, background=True, wait_s=1.0, retry_s=0.05,
                   source=lambda: _Source(env["pack"], fail="0.qkv"))
    gen.attach_check_source(stock)
    ms = _store(env, "s")
    try:
        t0 = time.monotonic()
        with pytest.raises(CheckStockEmpty, match="no fingerprint-verified W8 pass has completed") as ei:
            gen.generate_into_store(ms, 300)
        assert 0.9 < time.monotonic() - t0 < 30
        assert "last pass failed" in ei.value.message
        assert ms.available_slots(env["pack"].fingerprint, RING) == 0  # no segment, so no masks, so no request
    finally:
        stock.close()
        ms.close()


def test_exhaustion_waits_for_a_running_pass(env: dict[str, Any]) -> None:
    gen = _gen(env)
    stock = _stock(env, gen, background=True, wait_s=60.0, source=lambda: _Source(env["pack"], delay_s=0.3))
    gen.attach_check_source(stock)
    ms = _store(env, "s")
    try:
        t0 = time.monotonic()
        gen.generate_into_store(ms, 300)  # empty: waits for the first pass, then uses its first set
        assert time.monotonic() - t0 >= 1.0
        assert ms.available_slots(env["pack"].fingerprint, RING) == 300
    finally:
        stock.close()
        ms.close()


def test_stop_interrupts_the_wait(env: dict[str, Any]) -> None:
    from pgw.factory.governor import FactoryStopped

    gen = _gen(env)
    stock = _stock(env, gen, wait_s=60.0)
    ev = threading.Event()
    threading.Timer(0.3, ev.set).start()
    with pytest.raises(FactoryStopped):
        stock.take_segment(stop=ev)


def test_two_processes_never_share_an_item(env: dict[str, Any]) -> None:
    """Concurrent takes from two stock objects (as two fleet workers) hand out distinct items."""
    gen = _gen(env)
    a = _stock(env, gen, target_segments=40, low_water_segments=1)
    a.refill()
    b = _stock(env, gen, target_segments=40, low_water_segments=1)
    got: list[bytes] = []
    mu = threading.Lock()

    def run(s: CheckStock) -> None:
        for _ in range(20):
            x = bytes(s.take_segment().seed)
            with mu:
                got.append(x)

    ts = [threading.Thread(target=run, args=(s,)) for s in (a, b)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert len(got) == 40 and len(set(got)) == 40


def test_cli_status_and_discard(env: dict[str, Any]) -> None:
    from click.testing import CliRunner

    from pgw.factory.cli import factory_group

    gen = _gen(env)
    state = env["tmp"] / "state"
    stock = CheckStock(state / "check-stock" / "fp-ring", env["kp"], fingerprint=gen.fingerprint, ring=RING,
                       check_field="p31", checks=1, linears=gen.stock_linears(), source=lambda: _Source(env["pack"]),
                       target_segments=4, low_water_segments=1, target_private=3, low_water_private=1,
                       background=False)
    stock.refill()
    r = CliRunner().invoke(factory_group, ["check-stock", "--state-dir", str(state), "--json"])
    assert r.exit_code == 0, r.output
    assert json.loads(r.output)["fp-ring"]["segments"] == 4
    r = CliRunner().invoke(factory_group, ["check-stock", "--state-dir", str(state), "--discard"])
    assert r.exit_code == 0, r.output
    assert not stock.root.exists() and not stock.witness.exists()
    again = CheckStock(stock.root, env["kp"], fingerprint=gen.fingerprint, ring=RING, check_field="p31", checks=1,
                       linears=gen.stock_linears(), background=False)
    assert again.counts() == {"segment": 0, "private": 0}
