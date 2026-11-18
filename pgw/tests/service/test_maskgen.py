"""POST /pgw/v1/maskgen: the service multiplies MLWE mask ciphertexts by W8 (end to end, CPU)."""

from __future__ import annotations

from typing import Any

import httpx
import numpy as np
import pytest
import torch

from pgw.core.errors import PGWError
from pgw.hemask import DEGREE, HEParams, decrypt_wr, encrypt_masks, matmul_mod64, public_matrix
from pgw.hemask.client import RemoteMulmod64
from pgw.hemask.wire import MASKGEN_PATH, MaskgenItem, MaskgenRequest, encode_maskgen_request
from pgw.service import ServiceSettings
from tests.service.conftest import TOKEN_A, asgi_client, running_app


def _settings(pack_dir: Any, tokens_file: Any, **kw: Any) -> ServiceSettings:
    return ServiceSettings(packs=[pack_dir], device="cpu", rings=["K40A12"], tokens_file=tokens_file, **kw)


async def test_maskgen_roundtrip_is_exact(tiny_pack_dir: Any, tokens_file: Any, pack_fp: str,
                                          pack_weights: dict[str, torch.Tensor]) -> None:
    lid, w8 = next(iter(pack_weights.items()))
    w = w8.numpy().astype(np.int64)
    m, n = w.shape
    p = HEParams(k=40, rank=8)
    r = np.random.default_rng(0).integers(0, 1 << 40, size=(DEGREE, n), dtype=np.int64)
    a = public_matrix(bytes(32), n, p)
    hint = matmul_mod64(w, a)
    ct = encrypt_masks(r, a, p)
    async with running_app(_settings(tiny_pack_dir, tokens_file)) as app:
        client = asgi_client(app, TOKEN_A)
        try:
            remote = RemoteMulmod64(client, pack_fp, {k: int(v.shape[0]) for k, v in pack_weights.items()})
            y = await remote.acall(lid, ct.b)
        finally:
            await client.aclose()
    assert y.shape == (m, DEGREE)
    diff = (y - matmul_mod64(w, ct.b)).view(np.int64)  # 16 low bits are rounded away on the wire
    assert int(np.abs(diff).max()) <= 1 << 15
    got = decrypt_wr(y, hint, ct, p)
    ref = np.array((r.astype(object) @ w.T.astype(object)) % (1 << 40), dtype=np.int64)
    assert np.array_equal(got, ref)
    assert remote.bytes_up > 8 * n * DEGREE and 6 * m * DEGREE <= remote.bytes_down < 7 * m * DEGREE


async def test_maskgen_requires_auth_and_valid_sizes(tiny_pack_dir: Any, tokens_file: Any, pack_fp: str,
                                                     pack_weights: dict[str, torch.Tensor]) -> None:
    lid, w8 = next(iter(pack_weights.items()))
    n = int(w8.shape[1])
    good = MaskgenRequest("r1", pack_fp, (MaskgenItem(lid, DEGREE, bytes(8 * n * DEGREE)),))
    short = MaskgenRequest("r2", pack_fp, (MaskgenItem(lid, DEGREE, bytes(8 * (n - 1) * DEGREE)),))
    unknown = MaskgenRequest("r3", pack_fp, (MaskgenItem("99.qkv", DEGREE, bytes(8 * n * DEGREE)),))
    hdr = {"content-type": "application/vnd.pgw+msgpack"}
    async with running_app(_settings(tiny_pack_dir, tokens_file)) as app:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://svc") as h:
            r = await h.post(MASKGEN_PATH, content=encode_maskgen_request(good), headers=hdr)
            assert r.status_code == 401
            auth = {**hdr, "authorization": f"Bearer {TOKEN_A}"}
            assert (await h.post(MASKGEN_PATH, content=encode_maskgen_request(good), headers=auth)).status_code == 200
            assert (await h.post(MASKGEN_PATH, content=encode_maskgen_request(short), headers=auth)).status_code == 400
            assert (await h.post(MASKGEN_PATH, content=encode_maskgen_request(unknown), headers=auth)).status_code == 400
            assert (await h.post(MASKGEN_PATH, content=b"\x00garbage", headers=auth)).status_code == 400


async def test_maskgen_unknown_model_is_typed_error(tiny_pack_dir: Any, tokens_file: Any,
                                                    pack_weights: dict[str, torch.Tensor]) -> None:
    lid, w8 = next(iter(pack_weights.items()))
    async with running_app(_settings(tiny_pack_dir, tokens_file)) as app:
        client = asgi_client(app, TOKEN_A)
        try:
            remote = RemoteMulmod64(client, "f" * 64, {lid: int(w8.shape[0])})
            with pytest.raises(PGWError):
                await remote.acall(lid, np.zeros((int(w8.shape[1]), DEGREE), dtype=np.uint64))
        finally:
            await client.aclose()
