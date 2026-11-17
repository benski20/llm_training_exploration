"""Fuzzing: arbitrary bytes and structurally random messages must only ever raise typed
:class:`PGWError`s (never crash with another exception type, hang or allocate unboundedly)."""

from __future__ import annotations

import random

import msgpack
import pytest

from pgw.core import RING_K40_A12, RING_K48_A20, PGWError
from pgw.wire import (
    ComputeRequest,
    ComputeResponse,
    ErrorResponse,
    ModelsResponse,
    WireLimits,
    decode_any,
    decode_compute_request,
    decode_message,
    encode_message,
)
from tests.wire.conftest import DIMS, FakeHandler, make_request

TYPES = (ComputeRequest, ComputeResponse, ErrorResponse, ModelsResponse)


def _check(data: bytes) -> None:
    for t in TYPES:
        try:
            decode_message(data, t, max_bytes=1 << 20)
        except PGWError:
            pass
    try:
        decode_compute_request(data, WireLimits(max_request_bytes=1 << 20), DIMS)
    except PGWError:
        pass


def _seeds() -> list[bytes]:
    import asyncio

    req, _ = make_request(RING_K40_A12, [("0.qkv", 2), ("0.down", 1)], deadline_ms=10)
    resp = asyncio.run(FakeHandler().compute(req, None))  # type: ignore[arg-type]
    models = asyncio.run(FakeHandler().models())
    return [encode_message(m) for m in (req, resp, models, ErrorResponse("unavailable", "x", True, "i", {"a": [1]}))]


def test_fuzz_byte_mutations():
    rng = random.Random(1234)
    seeds = _seeds()
    for _ in range(3000):
        data = bytearray(rng.choice(seeds))
        op = rng.randrange(5)
        if op == 0:  # flip bytes
            for _ in range(rng.randint(1, 8)):
                data[rng.randrange(len(data))] = rng.randrange(256)
        elif op == 1:  # truncate
            data = data[: rng.randrange(len(data))]
        elif op == 2:  # insert junk
            pos = rng.randrange(len(data) + 1)
            data[pos:pos] = bytes(rng.randrange(256) for _ in range(rng.randint(1, 16)))
        elif op == 3:  # delete a range
            a = rng.randrange(len(data))
            del data[a : a + rng.randint(1, 16)]
        else:  # splice two seeds
            other = rng.choice(seeds)
            cut = rng.randrange(min(len(data), len(other)))
            data = data[:cut] + other[cut:]
        _check(bytes(data))


def test_fuzz_random_bytes():
    rng = random.Random(99)
    for _ in range(2000):
        _check(bytes(rng.randrange(256) for _ in range(rng.randint(0, 64))))


def _rand_obj(rng: random.Random, depth: int = 0):
    kinds = ["int", "neg", "float", "str", "bin", "bool", "none", "list", "map", "big"]
    if depth > 3:
        kinds = kinds[:7]
    k = rng.choice(kinds)
    if k == "int":
        return rng.randint(0, 10_000)
    if k == "neg":
        return -rng.randint(1, 1 << 40)
    if k == "big":
        return rng.choice([2**63 - 1, 2**64 - 1, -(2**63)])
    if k == "float":
        return rng.choice([0.5, float("inf"), float("nan"), -1e300])
    if k == "str":
        return rng.choice(["", "x", "K40A12", "0.qkv", "compute", "é" * 3, "a" * 300])
    if k == "bin":
        return bytes(rng.randrange(256) for _ in range(rng.randint(0, 20)))
    if k == "bool":
        return rng.random() < 0.5
    if k == "none":
        return None
    if k == "list":
        return [_rand_obj(rng, depth + 1) for _ in range(rng.randint(0, 4))]
    return {
        rng.choice(["v", "type", "id", "items", "fp", "x"]): _rand_obj(rng, depth + 1) for _ in range(rng.randint(0, 4))
    }


@pytest.mark.parametrize("mtype", ["compute", "compute_result", "error", "models"])
def test_fuzz_structured_fields(mtype):
    """Valid envelope, random field values: exercises every validator branch."""
    rng = random.Random(hash(mtype) & 0xFFFF)
    keys = {
        "compute": ["id", "fp", "ring", "items", "deadline_ms"],
        "compute_result": ["id", "items", "timing"],
        "error": ["id", "code", "message", "retryable", "details"],
        "models": ["models", "versions", "server"],
    }[mtype]
    for _ in range(1500):
        d = {"v": 1, "type": mtype}
        for k in keys:
            if rng.random() < 0.85:
                d[k] = _rand_obj(rng)
        try:
            data = msgpack.packb(d, use_bin_type=True)
        except (OverflowError, ValueError):
            continue
        try:
            decode_any(data)
        except PGWError:
            pass


def test_fuzz_item_lengths_vs_dims():
    """decode_compute_request must reject every x_hat whose length != rows*n*K/8."""
    rng = random.Random(7)
    lim = WireLimits(max_rows_per_item=64, max_items=8)
    for _ in range(500):
        lid = rng.choice(list(DIMS))
        rows = rng.randint(1, 70)
        n = DIMS[lid][1]
        good = rows * n * 6
        length = good + rng.choice([0, 0, -1, 1, -6, 6, n])
        if length < 0:
            continue
        d = {
            "v": 1,
            "type": "compute",
            "id": "f",
            "fp": "ab" * 32,
            "ring": RING_K48_A20.name,
            "items": [{"linear": lid, "rows": rows, "x_hat": b"\x00" * length}],
        }
        try:
            req = decode_compute_request(msgpack.packb(d, use_bin_type=True), lim, DIMS)
            assert length == good and rows <= 64
            assert len(req.items[0].x_hat) == good
        except PGWError:
            assert length != good or rows > 64
