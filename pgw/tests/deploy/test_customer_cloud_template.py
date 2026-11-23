"""Static checks of the customer-cloud Terraform template (deploy/customer-cloud): no cloud calls, no credentials.

``terraform fmt``/``validate`` were run by hand (Terraform 1.9.8, hashicorp/aws 6.73, cloudinit 2.4, random 3.9);
these tests keep the contract the README documents: required inputs, a sensitive enrollment code, no public
inbound except the client API, the mask service reachable only from the gateway, the bootstrap scripts parse and
use the CLI commands that exist.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2] / "deploy" / "customer-cloud"
AWS = ROOT / "aws"


def _blocks(text: str, kind: str) -> dict[str, str]:
    """``{name: body}`` of top-level ``kind "name" { ... }`` blocks (brace matching, strings skipped)."""
    out: dict[str, str] = {}
    for m in re.finditer(rf'^{kind}\s+"([^"]+)"\s*(?:"([^"]+)"\s*)?\{{', text, re.M):
        i, depth, in_str = m.end(), 1, False
        while depth:
            c = text[i]
            if c == '"' and text[i - 1] != "\\":
                in_str = not in_str
            elif not in_str:
                depth += {"{": 1, "}": -1}.get(c, 0)
            i += 1
        key = m.group(1) if m.group(2) is None else f"{m.group(1)}.{m.group(2)}"
        out[key] = text[m.end():i - 1]
    return out


def _all_tf() -> str:
    return "\n".join(p.read_text() for p in sorted(AWS.glob("*.tf")))


def test_required_variables_and_sensitive_code() -> None:
    variables = _blocks((AWS / "variables.tf").read_text(), "variable")
    required = {k for k, body in variables.items() if not re.search(r"^\s*default\s*=", body, re.M)}
    assert required == {"region", "enrollment_code", "client_api_cidrs", "image_registry"}
    assert re.search(r"sensitive\s*=\s*true", variables["enrollment_code"])
    for k, body in variables.items():
        assert re.search(r'description\s*=\s*"', body), f"variable {k} has no description"
    for k in ("model", "mask_supply", "check_field", "name", "enrollment_code"):
        assert "validation" in variables[k]
    # parameters the task asks for
    for k in ("region", "create_vpc", "vpc_id", "subnet_id", "gateway_instance_type", "mask_service_instance_type",
              "model", "enrollment_code"):
        assert k in variables
    # opt-in defaults: the client API is private, GPU only by the model's preset
    assert re.search(r'default\s*=\s*false', variables["expose_client_api_publicly"])
    assert re.search(r'default\s*=\s*"auto"', variables["mask_supply"])


def test_examples_set_only_known_variables_and_no_code() -> None:
    known = set(_blocks((AWS / "variables.tf").read_text(), "variable"))
    for ex in sorted((AWS / "examples").glob("*.tfvars")):
        keys = set(re.findall(r"^([a-z_]+)\s*=", ex.read_text(), re.M))
        assert keys <= known, f"{ex.name}: unknown {keys - known}"
        assert "enrollment_code" not in keys and "PGW1-" not in ex.read_text()


def test_no_public_inbound_except_client_api() -> None:
    tf = _all_tf()
    ingress = _blocks(tf, "resource")
    rules = {k: v for k, v in ingress.items() if k.startswith("aws_vpc_security_group_ingress_rule.")}
    assert rules
    for name, body in rules.items():
        assert "0.0.0.0/0" not in body, name
        port = re.search(r"from_port\s*=\s*(\S+)", body)
        assert port is not None
        if "mask_service" in name:
            assert port.group(1) == "8470" and "aws_security_group.gateway.id" in body
        elif "endpoint" in name:
            assert "aws_security_group.gateway.id" in body
        else:  # gateway / NLB: the client API only
            assert port.group(1) == "8443", name
    # no inline ingress blocks and no SSH rule anywhere
    assert not re.search(r"^\s*ingress\s*\{", tf, re.M)
    assert not re.search(r"port\s*=\s*22\b", tf)
    # instances: IMDSv2, encrypted volumes, private subnet only
    for name in ("aws_instance.gateway", "aws_instance.mask_service"):
        body = ingress[name]
        assert 'http_tokens                 = "required"' in body
        assert body.count("encrypted") >= 2 and "encrypted   = false" not in body
        assert "associate_public_ip_address" not in body


def test_mask_service_cannot_read_enrollment_code() -> None:
    body = _blocks((AWS / "instances.tf").read_text(), "data")["aws_iam_policy_document.mask_service"]
    assert "enrollment-code" not in body
    assert "/gateway/*" in body and "/mask-service/*" in body


@pytest.mark.parametrize("script", sorted((ROOT / "scripts").glob("*.sh")), ids=lambda p: p.name)
def test_bootstrap_scripts_parse_and_use_real_commands(script: Path) -> None:
    subprocess.run(["bash", "-n", str(script)], check=True)  # noqa: S603, S607
    text = script.read_text()
    assert text.startswith("#!/bin/bash") and "set -euo pipefail" in text
    from click.testing import CliRunner

    from pgw.cli.main import cli

    # every `pgw <group> <command>` the script runs must exist in this CLI
    cmds = set(re.findall(r"\bpgw ((?:gateway|maskstore|factory) [a-z-]+(?: [a-z-]+)?)", text))
    assert cmds
    runner = CliRunner()
    for c in sorted(cmds):
        parts = c.split()
        res = runner.invoke(cli, [*parts, "--help"])
        if res.exit_code != 0 and len(parts) == 3:  # third word is an argument, not a subcommand
            res = runner.invoke(cli, [*parts[:2], "--help"])
        assert res.exit_code == 0, f"{script.name}: 'pgw {c}' is not a command: {res.output[-300:]}"
    if shutil.which("shellcheck"):
        subprocess.run(["shellcheck", "-S", "warning", str(script)], check=True)  # noqa: S603, S607


def test_cpu_only_preset_defaults_to_pks_v3_with_p31_and_explicit_ack() -> None:
    """CPU-only presets (cpu_encrypted): pks-v3 by default (pending external review), the acknowledgement passed
    explicitly, a hint key outside the hint directory, and check_field auto (p31 for provider-computed masks)."""
    from pgw.hemask.pks import PKS_V3_ACK

    variables = _blocks((AWS / "variables.tf").read_text(), "variable")
    assert re.search(r'default\s*=\s*"auto"', variables["he_scheme"]) and "validation" in variables["he_scheme"]
    assert re.search(r'default\s*=\s*"auto"', variables["check_field"])
    assert "PENDING EXTERNAL REVIEW" in variables["he_scheme"]
    main = (AWS / "main.tf").read_text()
    assert re.search(r'local\.supply == "cpu_encrypted" \? "pks-v3" : "mlwe"', main)
    assert "PGW_HE_SCHEME       = local.he_scheme" in (AWS / "instances.tf").read_text()
    boot = (ROOT / "scripts" / "pgw-gateway-bootstrap.sh").read_text()
    assert f"PGW_FACTORY_HE_PKS_ACK={PKS_V3_ACK}" in boot and "PGW_FACTORY_HE_SCHEME=pks-v3" in boot
    assert "PGW_FACTORY_HE_HINT_KEY_FILE=/etc/pgw/secrets/pks-hint.key" in boot
    assert "PGW_HE_SCHEME:-mlwe" in boot  # without the variable: the library default
    assert "PGW_GATEWAY_PRIME_CHECK=true" in boot  # the gateway runs the p31 check whenever p31 is in use
    assert "PENDING EXTERNAL REVIEW" in (ROOT / "README.md").read_text()


def test_readme_has_onboarding_steps_and_sizing() -> None:
    text = (ROOT / "README.md").read_text()
    for needle in ("Onboarding in about an hour", "terraform destroy", "quota", "same region", "Sizing",
                   "qwen2.5-7b", "qwen2.5-32b", "qwen2.5-72b", "L4", "A10G", "per hour"):
        assert needle.lower() in text.lower(), needle
