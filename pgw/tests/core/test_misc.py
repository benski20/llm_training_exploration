"""Tests for errors, types, fingerprints, settings, logging and the self test."""

import json
import logging

import pytest

from pgw.core import (
    ConfigError,
    ErrorCode,
    FingerprintMismatchError,
    IntegrityError,
    LinearId,
    LinearKind,
    LinearSpec,
    MaskExhaustedError,
    PGWError,
    canonical_json,
    compute_fingerprint,
    error_from_code,
    linear_ids,
    require_same_fingerprint,
    sha256_file,
    validate_fingerprint,
)
from pgw.core.log import JsonFormatter, configure_logging
from pgw.core.selftest import run_selftest
from pgw.core.settings import CommonSettings, ring_from_setting


def test_error_codes_roundtrip():
    e = MaskExhaustedError("no masks", details={"model": "x"})
    assert e.code == ErrorCode.MASK_EXHAUSTED and e.http_status == 503 and not e.retryable
    d = e.to_dict()
    back = error_from_code(d["code"], d["message"], d["details"])
    assert isinstance(back, MaskExhaustedError) and back.details == {"model": "x"}
    unk = error_from_code("weird", "m")
    assert type(unk) is PGWError and unk.details["unknown_code"] == "weird"
    assert error_from_code("rate_limited", "slow").retryable
    assert isinstance(error_from_code("integrity_check_failed", "x"), IntegrityError)


def test_linear_ids():
    lid = LinearId.parse("12.gate_up")
    assert lid == LinearId(12, LinearKind.GATE_UP) and str(lid) == "12.gate_up"
    assert lid.ordinal == 4 * 12 + 2
    ids = linear_ids(2)
    assert [i.key for i in ids] == ["0.qkv", "0.o", "0.gate_up", "0.down", "1.qkv", "1.o", "1.gate_up", "1.down"]
    assert sorted(reversed(ids), key=lambda i: i.ordinal) == ids
    for bad in ("x.qkv", "1.mlp", "-1.o", "1.qkv "):
        with pytest.raises(ConfigError):
            LinearId.parse(bad)
    with pytest.raises(ConfigError):
        LinearSpec(lid, 0, 4)


def test_fingerprint(tmp_path):
    m = {"format": "pgw-modelpack", "version": 1, "tensors": {"a": "00"}}
    fp = compute_fingerprint(m)
    assert fp == "81390429e88fd4a461a90250ae404409a419aae7ae8784a43caffea25161c658"  # frozen KAT
    assert compute_fingerprint({**m, "created_at": "now", "fingerprint": fp, "notes": "x"}) == fp
    assert compute_fingerprint({**m, "version": 2}) != fp
    assert validate_fingerprint(fp) == fp
    with pytest.raises(ConfigError):
        validate_fingerprint("ABC")
    require_same_fingerprint(fp, fp)
    with pytest.raises(FingerprintMismatchError):
        require_same_fingerprint(fp, "0" * 64, context="test")
    with pytest.raises(ConfigError):
        canonical_json({"x": float("nan")})
    f = tmp_path / "f.bin"
    f.write_bytes(b"abc")
    assert sha256_file(f) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_settings_from_env(monkeypatch):
    monkeypatch.setenv("PGW_RING", "k48a20")
    monkeypatch.setenv("PGW_LOG_JSON", "true")
    s = CommonSettings()
    assert s.ring == "K48A20" and s.ring_params.k == 48 and s.log_json
    monkeypatch.setenv("PGW_RING", "K36A12")
    with pytest.raises(Exception):
        CommonSettings()
    assert ring_from_setting(None).name == "K40A12"


def test_json_logging(capsys):
    configure_logging("INFO", json_lines=True)
    logging.getLogger("pgw.test").info("hello", extra={"rows": 3})
    line = capsys.readouterr().err.strip().splitlines()[-1]
    rec = json.loads(line)
    assert rec["msg"] == "hello" and rec["rows"] == 3
    assert isinstance(JsonFormatter().format(logging.makeLogRecord({"msg": "x"})), str)
    configure_logging("WARNING")


def test_selftest():
    res = run_selftest(m=24, n=40, tokens=3)
    assert res.ok, [k for k, v in res.checks.items() if not v]
