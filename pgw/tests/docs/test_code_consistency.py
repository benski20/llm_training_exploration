"""The documents must agree with the code: CLI, settings, error codes, endpoints, metrics, schemas."""

from __future__ import annotations

import inspect
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import click
import pytest
from pydantic import BaseModel

from tests.docs.conftest import code_blocks, section, strip_code, table_rows, unticks

# ---------------------------------------------------------------------------------------------- CLI


def _documented_commands(docs: dict[str, str]) -> list[tuple[str, list[str]]]:
    """``(doc, argv)`` for every ``pgw ...`` invocation in bash blocks and inline code."""
    out: list[tuple[str, list[str]]] = []
    for rel, text in docs.items():
        for block in code_blocks(text, ("bash", "sh", "")):
            joined = re.sub(r"\\\n\s*", " ", block)
            for line in joined.splitlines():
                line = line.split(" #", 1)[0].strip()
                m = re.search(r"(?:^|[;&|]\s*|^\S+=\S+\s+)(pgw\s.*)$", line)
                if m and not line.startswith("#"):
                    out.append((rel, shlex.split(m.group(1).replace("$", "X"))))
        for frag in re.findall(r"`(pgw [^`]+)`", strip_code(text)):
            out.append((rel, shlex.split(frag)))
    return out


def _resolve(argv: list[str]) -> tuple[click.Command, list[str], list[str]]:
    """Walk the click tree; returns (leaf command, remaining args, command path)."""
    from pgw.cli.main import cli

    cmd: click.Command = cli
    ctx = click.Context(cli, info_name="pgw")
    path = ["pgw"]
    rest = argv[1:]
    while isinstance(cmd, click.Group) and rest and not rest[0].startswith("-"):
        sub = cmd.get_command(ctx, rest[0])
        assert sub is not None, f"unknown command {' '.join([*path, rest[0]])}"
        path.append(rest[0])
        cmd = sub
        ctx = click.Context(cmd, info_name=rest[0], parent=ctx)
        rest = rest[1:]
    return cmd, rest, path


def test_documented_cli_commands_exist_with_their_options(docs: dict[str, str]) -> None:
    cmds = _documented_commands(docs)
    assert len(cmds) >= 30
    seen_paths = set()
    for rel, argv in cmds:
        cmd, rest, path = _resolve(argv)
        seen_paths.add(" ".join(path))
        if isinstance(cmd, click.Group) and not rest:
            continue  # e.g. "pgw factory" mentioned as a group
        assert not isinstance(cmd, click.Group), f"{rel}: incomplete command {' '.join(argv)}"
        opts = {o for p in cmd.params for o in (*p.opts, *p.secondary_opts)}
        for tok in rest:
            if tok.startswith("--"):
                name = tok.split("=", 1)[0]
                assert name in opts, f"{rel}: '{' '.join(path)}' has no option {name}"
    for required in ("pgw pack", "pgw inspect", "pgw doctor", "pgw maskstore init", "pgw maskstore import",
                     "pgw factory run", "pgw factory generate", "pgw factory benchmark", "pgw gateway serve",
                     "pgw gateway keys create", "pgw maskstore rotate-key"):
        assert required in seen_paths, f"operator docs never show {required}"


def test_documented_help_runs() -> None:
    """Smoke: the commands the guides rely on print help without error."""
    for argv in (["factory", "run", "--help"], ["gateway", "keys", "create", "--help"], ["maskstore", "import", "--help"]):
        r = subprocess.run([sys.executable, "-c", "from pgw.cli.main import main; main()", *argv],  # noqa: S603
                           capture_output=True, text=True, timeout=120)
        assert r.returncode == 0, r.stderr


# ------------------------------------------------------------------------------------------ settings


def _settings_classes() -> dict[str, type[BaseModel]]:
    from pgw.core.settings import CommonSettings
    from pgw.factory.config import FactorySettings
    from pgw.gateway.config import GatewaySettings
    from pgw.maskstore.config import MaskStoreSettings
    from pgw.service.config import ServiceSettings

    return {"GATEWAY": GatewaySettings, "SERVICE": ServiceSettings, "FACTORY": FactorySettings,
            "MASKSTORE": MaskStoreSettings, "": CommonSettings}


#: Env names that are documented but are not settings fields.
_NON_FIELD_ENV = {
    "PGW_MASKSTORE_KEK",  # default value of MaskStoreSettings.key_env_var
    "PGW_DOCTOR_EXPECTED_TOKENS_PER_S",  # input of `pgw doctor`'s mask-supply check (pgw.cli.doctor), not a setting
    "PGW_FLOAT_THREADS",  # engine switch read by pgw.engine.model (per model: GatewayModelConfig.float_threads)
    "PGW_SERVICE_MASKGEN_CONCURRENCY",  # env override read by pgw.service.handler (maskgen slots per device)
    "PGW_SERVICE_MASKGEN_FUSED",  # env switch read by pgw.service.maskgen_gpu (fused CUDA maskgen path)
    "PGW_SERVICE_MASKGEN_YIELD_MS",  # env knob read by pgw.service.maskgen_gpu (decode yield between blocks)
    "PGW_HEMASK_HINT_PREFETCH",  # env switch read by pgw.hemask.devmm (double-buffered hint stream)
    "PGW_HEMASK_GPU_HINT_STALE_S",  # env knob read by pgw.hemask.devmm (hint cache keeps hints in use)
    "PGW_API_KEY",  # SDK key read by the client library (pgw.sdk, docs/sdk)
    "PGW_CACHE_DIR",  # client library options (pgw.sdk.options.SDKOptions / pgw.sdk.cache), docs/sdk
    "PGW_STATE_DIR",
    "PGW_HINT",
    "PGW_MASK_SCHEME",
    "PGW_MASK_WORKERS",
    "PGW_MASK_STORE_GB",
    "PGW_SDK_LOG",
    "PGW_FACTORY_WEIGHTS",  # SDKOptions.factory_weights (the factory child gets PGW_FACTORY_FACTORY_WEIGHTS)
    "PGW_FACTORY_WEIGHTS_SOURCE",  # SDKOptions.factory_weights_source (= FactorySettings.weights_source)
    "PGW_CHECK_STOCK_SEGMENTS",  # SDKOptions.check_stock_segments
    "PGW_PLATFORM_ENABLED",  # developer console on the provider (pgw.platform.PlatformSettings)
    "PGW_PLATFORM_ACCOUNTS_URL",
    "PGW_PLATFORM_ACCOUNTS_TOKEN_FILE",
    "PGW_PLATFORM_OIDC",
    "PGW_PLATFORM_OIDC_ORG",
    "PGW_TRIP_FAST",  # native trip / decode stack switches the SDK sets (pgw.sdk.engine.NATIVE_ENV)
    "PGW_TRIP_NATIVE",
    "PGW_TRIP_DRIVER",
    "PGW_DECODE_NATIVE",
    "PGW_TRIP_DECRYPT_AHEAD",
    "PGW_LANE_SPIN_AUTO_US",
    "PGW_LANE_WAIT_PREFETCH",
    "PGW_LANE_MULTI",  # lane format 2 multi calls (pgw.wire.service_lane.build_lane_client; NATIVE_ENV sets it)
    "PGW_SERVICE_LANE_MULTI",  # lane format 2 offer of the service lane (pgw.service.lane._multi_items)
    "PGW_NATIVE",  # native-extension switch read by pgw.core.native, pgw.hemask.ntt, pgw.hemask.pks_native
    "PGW_NTT",  # native NTT switch read by pgw.hemask.ntt
    "PGW_PKS_NATIVE",  # native pks-v3 switch read by pgw.hemask.pks_native
    "PGW_REQUIRE_NATIVE",  # build-time switch read by setup.py (fail when the extensions cannot be compiled)
}


def _field_path_exists(model: type[BaseModel], parts: list[str]) -> bool:
    fields = model.model_fields
    head = parts[0]
    if head not in fields:
        return False
    if len(parts) == 1:
        return True
    ann = fields[head].annotation
    sub = next((a for a in (ann, *getattr(ann, "__args__", ())) if inspect.isclass(a) and issubclass(a, BaseModel)),
               None)
    return sub is not None and _field_path_exists(sub, parts[1:])


def test_documented_env_vars_are_real_settings(docs: dict[str, str]) -> None:
    classes = _settings_classes()
    found = set()
    for rel, text in docs.items():
        for name in set(re.findall(r"\bPGW_[A-Z0-9_]*[A-Z0-9]\b", text)):
            found.add(name)
            if name in _NON_FIELD_ENV:
                continue
            m = re.match(r"PGW_(GATEWAY|SERVICE|FACTORY|MASKSTORE)_(.+)", name)
            prefix, rest = (m.group(1), m.group(2)) if m else ("", name[len("PGW_"):])
            parts = rest.lower().split("__")
            assert _field_path_exists(classes[prefix], parts), f"{rel}: {name} is not a setting"
    assert len(found) >= 25


def test_gateway_limits_table_matches_defaults(docs: dict[str, str]) -> None:
    from pgw.gateway.config import GatewayModelConfig, GatewaySettings

    fields = GatewaySettings.model_fields
    sec = section(docs["api/gateway-api.md"], "7. Limits")
    rows = table_rows(sec)
    assert len(rows) >= 9
    checked = 0
    for _limit, setting, default in rows:
        names = [n for n in unticks(setting) if n.isupper()]
        for n in names:
            f = fields[n.lower()]
            num = re.sub(r"[ ,]", "", re.sub(r"\(.*?\)", "", default.split("/")[0]))
            if f.default is None:
                assert "unlimited" in default
            elif "MiB" in default:
                assert f.default == int(num.replace("MiB", "")) << 20
            else:
                assert f.default == int(num), f"{n}: documented {default}, code {f.default}"
            checked += 1
    assert checked >= 7
    mf = GatewayModelConfig.model_fields
    assert mf["max_concurrent_sessions"].default == 32 and mf["max_queue"].default == 256
    assert mf["queue_timeout_s"].default == 30.0


def test_security_relevant_defaults_match_the_docs(docs: dict[str, str]) -> None:
    from pgw.core.ring import DEFAULT_RING
    from pgw.factory.config import FactorySettings
    from pgw.gateway.config import GatewayModelConfig, GatewaySettings
    from pgw.maskstore.config import MaskStoreSettings
    from pgw.service.config import ServiceSettings

    g = GatewaySettings.model_fields
    assert GatewayModelConfig.model_fields["freivalds_rate"].default == 1.0
    assert GatewayModelConfig.model_fields["mode"].default == "masked"
    assert GatewayModelConfig.model_fields["ring"].default == DEFAULT_RING.name == "K32A12C27"
    assert g["allow_dev_modes"].default is False and g["allow_anonymous"].default is False
    assert g["require_service_tls"].default is True
    assert g["noise_monitor"].default is True and g["noise_monitor_fail_closed"].default is True
    assert g["integrity_cooldown_s"].default is None  # admin reset only (security review)
    assert g["min_soundness_bits"].default == 20.0 and g["pack_verify"].default == "full"
    assert g["default_max_tokens"].default == 512
    f = FactorySettings.model_fields
    assert f["checks"].default is None and f["segment_slots"].default == 4096  # None = auto (>= 20 bits)
    assert f["checks"].metadata and any(getattr(m, "le", None) == 8 for m in f["checks"].metadata)
    m = MaskStoreSettings.model_fields
    assert m["shred_consumed"].default is True and m["fsync"].default is True
    assert m["min_soundness_bits"].default == 20.0 and m["allow_unsigned_partner_segments"].default is False
    s = ServiceSettings.model_fields
    assert s["batch_window_ms"].default == 0.0  # no idle wait; rows coalesce while a batch waits for its worker
    wp = docs["security/whitepaper.md"]
    assert "only an admin reset" in wp and "default 4096" in wp and "at least 20 bits" in wp and "up to 8" in wp


# ------------------------------------------------------------------------------------- error codes


def test_error_table_matches_error_codes(docs: dict[str, str]) -> None:
    from pgw.core.errors import HTTP_STATUS, ErrorCode
    from pgw.gateway.app import _OPENAI_TYPE

    sec = section(docs["api/gateway-api.md"], "6. Errors")
    rows = [r for r in table_rows(sec) if r[0].startswith("`")]
    documented = {unticks(r[0])[0]: r for r in rows}
    assert set(documented) == {c.value for c in ErrorCode}, "error table must list every ErrorCode exactly"
    for code, row in documented.items():
        ec = ErrorCode(code)
        assert int(row[1]) == HTTP_STATUS[ec], f"{code}: HTTP {row[1]} vs {HTTP_STATUS[ec]}"
        expected_type = _OPENAI_TYPE.get(HTTP_STATUS[ec], "server_error" if HTTP_STATUS[ec] >= 500
                                         else "invalid_request_error")
        assert unticks(row[2])[0] == expected_type, f"{code}: type"


def test_api_key_format_matches_auth() -> None:
    from pgw.gateway import auth

    assert auth._KEY_RE.pattern.startswith("^pgw-([0-9a-f]{12})-")


# --------------------------------------------------------------------------------------- endpoints


def test_endpoint_table_matches_gateway_routes(docs: dict[str, str]) -> None:
    import pgw.console.routes as console_mod
    import pgw.gateway.app as app_mod

    src = "".join(Path(m.__file__).read_text(encoding="utf-8") for m in (app_mod, console_mod))
    routes = {(m.upper(), re.sub(r":path}", "}", p))
              for m, p in re.findall(r'@app\.(get|post|delete|put|patch)\("([^"]+)"', src)}
    sec = section(docs["api/gateway-api.md"], "2. Endpoints")
    documented = {(r[0], unticks(r[1])[0]) for r in table_rows(sec)}
    assert documented == routes


def test_chat_request_fields_exist() -> None:
    from pgw.gateway.openai_schemas import ChatCompletionRequest, CompletionRequest

    text = (Path(__file__).resolve().parents[2] / "docs/api/gateway-api.md").read_text(encoding="utf-8")
    for sec_name, model in (("3. `POST /v1/chat/completions`", ChatCompletionRequest),
                            ("4. `POST /v1/completions`", CompletionRequest)):
        sec = section(text, sec_name)
        names: list[str] = []
        for row in table_rows(sec):
            for frag in unticks(row[0]):
                names.append(frag.split(".")[0])
        assert names
        for n in names:
            assert n in model.model_fields, f"{model.__name__} has no field {n}"
    f = ChatCompletionRequest.model_fields
    assert f["temperature"].default == 1.0 and f["top_p"].default == 1.0 and f["n"].default == 1


def test_streaming_terminator_and_fingerprint_length(docs: dict[str, str]) -> None:
    import pgw.gateway.app as app_mod

    src = Path(app_mod.__file__).read_text(encoding="utf-8")
    assert "[DONE]" in src and "fingerprint[:12]" in src
    api = docs["api/gateway-api.md"]
    assert "data: [DONE]" in api and "first 12 hex characters" in api


# ------------------------------------------------------------------------------------------ metrics


def test_metric_names_exist(docs: dict[str, str]) -> None:
    from pgw.observability.metrics import METRIC_NAMES

    names = set(METRIC_NAMES) if not isinstance(METRIC_NAMES, dict) else set(METRIC_NAMES.keys())
    mentioned: set[str] = set()
    for rel, text in docs.items():
        for n in re.findall(r"\bpgw_(?:gateway|masks|service|factory)_[a-z_]+[a-z]\b", text):
            mentioned.add(n)
            assert n in names, f"{rel}: unknown metric {n}"
    table = section(docs["api/gateway-api.md"], "9. Metrics")
    in_table = {unticks(r[0])[0] for r in table_rows(table)}
    expected = {n for n in names if n.startswith(("pgw_gateway_", "pgw_masks_"))}
    assert in_table == expected, f"metrics table differs: {in_table ^ expected}"


# ---------------------------------------------------------------------------------- audit and logs


def test_audit_log_has_no_content_fields() -> None:
    from pgw.gateway.audit import AUDIT_FIELDS

    for forbidden in ("prompt", "completion", "messages", "text", "content", "x_hat", "mask", "seed", "api_key"):
        assert forbidden not in AUDIT_FIELDS


def test_service_never_imports_masks_package() -> None:
    """Whitepaper section 8 claims the import-graph separation; check it on the source."""
    import pgw.service

    root = Path(pgw.service.__file__).parent
    for p in root.rglob("*.py"):
        src = p.read_text(encoding="utf-8")
        assert not re.search(r"^\s*(from|import)\s+pgw\.(maskstore|factory)\b", src, flags=re.M), p.name


# --------------------------------------------------------------------------------- crypto claims


def test_freivalds_table_matches_soundness_bits(docs: dict[str, str]) -> None:
    from pgw.core.freivalds import soundness_bits
    from pgw.core.ring import RingParams

    sec = section(docs["security/whitepaper.md"], "5. Integrity")
    rows = [r for r in table_rows(sec) if re.fullmatch(r"K\d\dA\d+", r[0])]
    assert len(rows) >= 6
    for ring, n, _example, h, c2, c4, c8 in rows:
        p = RingParams.from_name(ring)
        n_i = int(n)
        assert soundness_bits(p, n_i, 1) == int(h), f"{ring} n={n}: h"
        for c, v in ((2, c2), (4, c4), (8, c8)):
            assert soundness_bits(p, n_i, c) == int(v), f"{ring} n={n} c={c}"
    # the headline claim: auto default c = 3 at width 4096 gives 24 bits (floor 20)
    k40 = RingParams.from_name("K40A12")
    assert soundness_bits(k40, 4096, 2) < 20 <= soundness_bits(k40, 4096, 3) == 24
    minus24 = "2^\u221224"  # typographic minus as used in the docs
    assert minus24 in docs["security/whitepaper.md"] and minus24 in docs["README.md"]


def test_one_time_pad_lemma_numerically() -> None:
    """Lemma 1 sanity: x + r mod 2^K with uniform r has the same distribution for any x (small ring)."""
    import numpy as np

    k, n_samples = 8, 200_000
    rng = np.random.default_rng(0)
    r = rng.integers(0, 2**k, n_samples)
    h0 = np.bincount((0 + r) % 2**k, minlength=2**k)
    h1 = np.bincount((77 + r) % 2**k, minlength=2**k)
    # the shifted histogram is a permutation of the original: identical multiset of counts
    assert sorted(h0) == sorted(h1)
    assert np.array_equal(np.roll(h0, 77), h1)


def test_mask_reuse_leak_claim() -> None:
    """Whitepaper 4.3: reuse reveals the exact plaintext difference."""
    import numpy as np

    from pgw.core.ring import RingParams

    ring = RingParams.from_name("K40A12")
    mod = 1 << ring.k
    rng = np.random.default_rng(1)
    x1 = rng.integers(-ring.qmax, ring.qmax + 1, 64)
    x2 = rng.integers(-ring.qmax, ring.qmax + 1, 64)
    r = rng.integers(0, mod, 64, dtype=np.int64)
    d = ((x1 + r) % mod - (x2 + r) % mod) % mod
    d = np.where(d >= mod // 2, d - mod, d)
    assert np.array_equal(d, x1 - x2)


def test_freivalds_bound_lemma_small_ring() -> None:
    """Lemma 3: Pr[u.e = 0 mod 2^K] = 2^(v-K) exactly for a single coordinate (exhaustive, K=8)."""
    k = 8
    for e in (1, 2, 4, 6, 12, 64, 128):
        v = (e & -e).bit_length() - 1
        hits = sum(1 for u in range(2**k) if (u * e) % 2**k == 0)
        assert hits / 2**k == 2 ** (v - k)


def test_mask_derivation_claims() -> None:
    """SPEC 7.3 / whitepaper 3.2: masks come from the seed expander, are uniform-range and reproducible."""
    import numpy as np

    from pgw.core import SeedExpander

    seed = bytes(range(32))
    a = SeedExpander(seed).ring_uniform_rows(16, 40, 5, 3, "r", "0.qkv")
    b = SeedExpander(seed).ring_uniform_rows(16, 40, 5, 3, "r", "0.qkv")
    c = SeedExpander(seed).ring_uniform_rows(16, 40, 6, 1, "r", "0.qkv")
    a, b, c = (np.asarray(t) for t in (a, b, c))
    assert np.array_equal(a, b)
    assert a.min() >= 0 and a.max() < 2**40
    assert np.array_equal(a[1], c[0]), "rows are seekable by slot"


# ------------------------------------------------------------------------------- runnable snippets


def test_recipient_key_snippet_runs(docs: dict[str, str], tmp_path: Path) -> None:
    """The guide's key commands (recipient + partner keys) run as documented."""
    text = docs["operations/air-gapped-factory.md"]
    blocks = code_blocks(text, ("bash",))
    rec = next(b for b in blocks if "recipient-key create" in b)
    part = next(b for b in blocks if "partner-key create" in b)
    assert "--partner-key" in part and "PGW_MASKSTORE_TRUSTED_PARTNER_KEYS" in text
    rec_line = next(line for line in rec.splitlines() if line.startswith("pgw maskstore recipient-key create"))
    args = rec_line.split()[1:]
    args = [a.replace("/etc/pgw/secrets/recipient.key", str(tmp_path / "recipient.key"))
            .replace("/etc/pgw/recipient.pub.hex", str(tmp_path / "recipient.pub.hex")) for a in args]
    r = subprocess.run([sys.executable, "-m", "pgw.cli", *args], capture_output=True, text=True, timeout=60)  # noqa: S603
    assert r.returncode == 0, r.stderr
    assert '"kek_id": "hpke:' in r.stdout
    assert (tmp_path / "recipient.key").stat().st_mode & 0o077 == 0
    from pgw.factory.sinks import load_recipient_public_key

    assert len(load_recipient_public_key(tmp_path / "recipient.pub.hex")) == 32
    part_line = next(line for line in part.splitlines() if line.startswith("pgw maskstore partner-key create"))
    args = [a.replace("/keys/", str(tmp_path) + "/") for a in part_line.split("#")[0].split()[1:]]
    r = subprocess.run([sys.executable, "-m", "pgw.cli", *args], capture_output=True, text=True, timeout=60)  # noqa: S603
    assert r.returncode == 0, r.stderr
    assert len(bytes.fromhex((tmp_path / "partner.pub.hex").read_text().strip())) == 32


def test_models_json_example_validates(docs: dict[str, str], tmp_path: Path) -> None:
    import json

    from pgw.gateway.config import GatewaySettings, load_models_file

    block = next(b for b in code_blocks(docs["operations/install-on-prem.md"], ("json",)) if "pack_path" in b)
    data: list[dict[str, Any]] = json.loads(block.replace("<64 hex characters from pgw inspect>", "ab" * 32))
    p = tmp_path / "models.json"
    p.write_text(json.dumps(data))
    models = load_models_file(p)
    assert models[0].mode == "masked" and models[0].ring == "K40A12"
    GatewaySettings(models_file=None).validate_policy(models)  # production policy accepts the example


def test_admin_key_example_fields(docs: dict[str, str]) -> None:
    import json

    sec = section(docs["api/gateway-api.md"], "8. Admin API")
    body = json.loads(code_blocks(sec, ("json",))[0])
    import pgw.gateway.app as app_mod

    src = Path(app_mod.__file__).read_text(encoding="utf-8")
    for k in body:
        assert f'"{k}"' in src or f"_opt_int(\"{k}\")" in src, f"admin key field {k} not accepted"


@pytest.mark.parametrize("rel", ["operations/install-on-prem.md", "operations/install-customer-cloud.md"])
def test_operator_guides_never_suggest_restoring_masks(rel: str, docs: dict[str, str]) -> None:
    text = docs[rel].lower()
    assert "restore" in text
    for para in re.split(r"\n\s*\n", text):
        if "restore" in para and ("mask" in para or "snapshot" in para or "store" in para):
            assert re.search(r"never|refuse|do not|disable", para), para[:200]


# ------------------------------------------------------------------------------ rollback protection


def test_rollback_docs_match_store_defaults(docs: dict[str, str]) -> None:
    """The journal alone cannot detect a whole-volume snapshot revert, and the default counter is
    ``none``. The docs must not claim otherwise (regression: a guide claimed every restored snapshot
    is refused)."""
    from pgw.maskstore import counters
    from pgw.maskstore.config import MaskStoreSettings

    m = MaskStoreSettings.model_fields
    assert m["counter"].default == "none" and m["journal"].default is True
    assert "only detectable with a monotonic counter" in (counters.__doc__ or "")
    joined = "\n".join(docs.values())
    assert "A restored\n  snapshot is refused by the rollback check" not in joined
    assert "snapshot is refused by the rollback check, so it causes downtime, never reuse" not in joined
    for rel in ("security/whitepaper.md", "security/threat-model.md", "operations/install-on-prem.md",
                "operations/install-customer-cloud.md"):
        text = " ".join(docs[rel].split())
        assert "snapshot" in text and "counter" in text, rel
    wp = " ".join(docs["security/whitepaper.md"].split())
    assert "does **not** detect a restore of the database *and* the journal" in wp
    assert "default `counter` setting is `none`" in wp
