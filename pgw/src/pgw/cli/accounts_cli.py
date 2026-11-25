"""``pgw service orgs|keys|usage|audit|admin-token``: provider accounts (SDK keys) on the accounts database.

Every command works on ``--accounts-db`` (default ``PGW_SERVICE_ACCOUNTS_DB``) directly, so it also works while
the service runs (SQLite WAL); a running service picks changes up within ``accounts_refresh_s``. Keys and admin
tokens are shown once at creation and stored as hashes only.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
from pathlib import Path
from typing import Any

import click

from pgw.cli._common import echo_json, handle_errors
from pgw.core.errors import ConfigError

__all__ = ["register"]

_db_opt = click.option("--accounts-db", type=click.Path(path_type=Path, dir_okay=False), default=None,
                       help="Accounts database (default PGW_SERVICE_ACCOUNTS_DB)")


def _store(accounts_db: Path | None) -> Any:
    from pgw.service.accounts import AccountsStore

    path = accounts_db or (Path(os.environ["PGW_SERVICE_ACCOUNTS_DB"]) if os.environ.get("PGW_SERVICE_ACCOUNTS_DB")
                           else None)
    if path is None:
        raise ConfigError("no accounts database: pass --accounts-db or set PGW_SERVICE_ACCOUNTS_DB")
    return AccountsStore(path)


def _actor() -> str:
    import getpass

    try:
        return "cli:" + getpass.getuser()
    except Exception:  # no user name in some containers
        return "cli"


def parse_duration(text: str) -> float:
    """``90d`` / ``12h`` / ``30m`` / ``3600`` -> seconds (positive)."""
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([smhdy]?)\s*", text or "")
    if not m:
        raise ConfigError(f"invalid duration {text!r}: use e.g. 30m, 12h, 90d, 1y")
    s = float(m.group(1)) * {"": 1, "s": 1, "m": 60, "h": 3600, "d": 86400, "y": 365 * 86400}[m.group(2)]
    if s <= 0:
        raise ConfigError("duration must be positive")
    return s


def _ts(v: float | None) -> str:
    return dt.datetime.fromtimestamp(v, dt.UTC).strftime("%Y-%m-%d %H:%M") if v else "-"


# ---- organisations ------------------------------------------------------------------------------------
@click.group("orgs")
def orgs() -> None:
    """Organisations (customers) that own SDK keys."""


@orgs.command("create")
@click.argument("org_id")
@click.option("--name", default=None, help="Display name (default: the id)")
@click.option("--daily-token-quota", type=int, default=None, help="Tokens per UTC day over all the org's keys")
@_db_opt
@handle_errors
def orgs_create(org_id: str, name: str | None, daily_token_quota: int | None, accounts_db: Path | None) -> None:
    """Create an organisation (id: [a-z0-9][a-z0-9_-]{0,47})."""
    echo_json(_store(accounts_db).create_org(org_id, name, daily_token_quota=daily_token_quota, actor=_actor()))


@orgs.command("list")
@_db_opt
@handle_errors
def orgs_list(accounts_db: Path | None) -> None:
    """List organisations."""
    echo_json(_store(accounts_db).list_orgs())


@orgs.command("disable")
@click.argument("org_id")
@click.option("--enable", is_flag=True, help="Re-enable instead")
@_db_opt
@handle_errors
def orgs_disable(org_id: str, enable: bool, accounts_db: Path | None) -> None:
    """Disable an organisation: every key of it stops working (reversible with --enable)."""
    echo_json(_store(accounts_db).set_org_disabled(org_id, not enable, actor=_actor()))


# ---- SDK keys -----------------------------------------------------------------------------------------
@click.group("keys")
def keys() -> None:
    """SDK keys (pgw_<env>_...): reusable, revocable, scoped to models; hashed at rest."""


@keys.command("create")
@click.option("--org", "org_id", required=True, help="Organisation id")
@click.option("--create-org", is_flag=True, help="Create the organisation if it does not exist")
@click.option("--name", required=True, help="Key name ([A-Za-z0-9._-], max 64), e.g. 'prod' or 'ci'")
@click.option("--models", default="", help="Comma-separated model names the key may use (default: all served)")
@click.option("--env", "env", default=None, help="Key environment label (default PGW_SERVICE_SDK_KEY_ENV or 'live')")
@click.option("--expires", default=None, help="Lifetime, e.g. 90d, 1y (default: until revoked)")
@click.option("--daily-token-quota", type=int, default=None, help="Tokens per UTC day (default: none)")
@click.option("--rows-per-s", type=float, default=None, help="Masked rows per second over all installations")
@click.option("--burst-rows", type=int, default=None, help="Row bucket size (default 2 s of --rows-per-s)")
@click.option("--max-installations", type=int, default=None, help="Active installations at most (default: none)")
@click.option("--service-url", default=None, help="Provider URL in the key (default PGW_SERVICE_PUBLIC_URL)")
@click.option("--ca-file", type=click.Path(path_type=Path, exists=True, dir_okay=False), default=None,
              help="Provider CA to pin (default PGW_SERVICE_ENROLLMENT_CA_FILE)")
@click.option("--out", "out_file", type=click.Path(path_type=Path, dir_okay=False), default=None,
              help="Write the key to this file (0600) instead of stdout")
@_db_opt
@click.option("--json", "as_json", is_flag=True, help="Machine-readable output")
@handle_errors
def keys_create(org_id: str, create_org: bool, name: str, models: str, env: str | None, expires: str | None,
                daily_token_quota: int | None, rows_per_s: float | None, burst_rows: int | None,
                max_installations: int | None, service_url: str | None, ca_file: Path | None, out_file: Path | None,
                accounts_db: Path | None, as_json: bool) -> None:
    """Create an SDK key. It is shown once (or written to --out); only its hash is stored."""
    from pgw.core.fileio import write_private
    from pgw.enroll.code import validate_service_url
    from pgw.enroll.service import ca_fingerprint, load_ca_pem

    url = service_url or os.environ.get("PGW_SERVICE_PUBLIC_URL")
    if not url:
        raise ConfigError("no provider URL: pass --service-url or set PGW_SERVICE_PUBLIC_URL")
    url = validate_service_url(url)
    ca = ca_file or (Path(os.environ["PGW_SERVICE_ENROLLMENT_CA_FILE"])
                     if os.environ.get("PGW_SERVICE_ENROLLMENT_CA_FILE") else None)
    fp = ca_fingerprint(load_ca_pem(ca)) if ca is not None else None
    if fp is None and url.startswith("https://"):
        raise ConfigError("an https provider URL needs the CA to pin: pass --ca-file or set "
                          "PGW_SERVICE_ENROLLMENT_CA_FILE")
    st = _store(accounts_db)
    if create_org and not any(o["org_id"] == org_id for o in st.list_orgs()):
        st.create_org(org_id, actor=_actor())
    key, rec = st.create_key(
        org_id, name, url, fp, env=env or os.environ.get("PGW_SERVICE_SDK_KEY_ENV") or "live",
        models=[m for m in models.split(",") if m.strip()], expires_in_s=parse_duration(expires) if expires else None,
        daily_token_quota=daily_token_quota, rows_per_s=rows_per_s, burst_rows=burst_rows,
        max_installations=max_installations, actor=_actor())
    secret_text = key.encode()
    out: dict[str, Any] = rec.public()
    if out_file is not None:
        write_private(out_file, secret_text + "\n", append=False)
        out["key_file"] = str(out_file)
    if as_json:
        if out_file is None:
            out["key"] = secret_text
        echo_json(out)
        return
    if out_file is None:
        click.echo(secret_text)
    click.echo(f"\n{rec.key_id} for {org_id} ({name}), env {rec.env}, "
               + (f"models {', '.join(rec.models)}" if rec.models else "all served models")
               + (f", expires {_ts(rec.expires_at)} UTC" if rec.expires_at else ""), err=True)
    click.echo("The key is shown once. Give it to the customer over a trusted channel; revoke it with "
               f"'pgw service keys revoke {rec.key_id}' if it leaks.", err=True)


@keys.command("list")
@click.option("--org", "org_id", default=None, help="Only this organisation")
@_db_opt
@click.option("--json", "as_json", is_flag=True, help="Machine-readable output")
@handle_errors
def keys_list(org_id: str | None, accounts_db: Path | None, as_json: bool) -> None:
    """List SDK keys (ids, scope, state, limits; never the key or its hash)."""
    recs = [k.public() for k in _store(accounts_db).list_keys(org_id)]
    if as_json:
        echo_json(recs)
        return
    if not recs:
        click.echo("(no SDK keys)")
    for r in recs:
        lim = ", ".join(f"{k}={v}" for k, v in r["limits"].items() if v is not None)
        click.echo(f"{r['key_id']}  {r['org_id']:<16} {r['name']:<16} {r['env']:<5} {r['state']:<8} "
                   f"inst={r['installations']:<4} models={','.join(r['models']) or '*'}  "
                   f"last_used={_ts(r['last_used_at'])}" + (f"  {lim}" if lim else ""))


@keys.command("revoke")
@click.argument("key_id")
@click.option("--tokens-file", type=click.Path(path_type=Path), default=None,
              help="Remove the installations' service tokens from here (default PGW_SERVICE_TOKENS_FILE)")
@_db_opt
@handle_errors
def keys_revoke(key_id: str, tokens_file: Path | None, accounts_db: Path | None) -> None:
    """Revoke an SDK key and every installation enrolled with it (their service tokens are removed). A running
    service refuses the key's tenant within its refresh interval and closes its open channels."""
    from pgw.service.accounts import revoke_key_everywhere

    tf = tokens_file or (Path(os.environ["PGW_SERVICE_TOKENS_FILE"]) if os.environ.get("PGW_SERVICE_TOKENS_FILE")
                         else None)
    if tf is None:
        click.echo("warning: no tokens file given; the installations' service tokens stay in it until the key's "
                   "tenant is refused by the running service (it is, within its refresh interval)", err=True)
    out = revoke_key_everywhere(_store(accounts_db), key_id, tf, actor=_actor())
    click.echo(json.dumps({"revoked": out["key_id"], "org_id": out["org_id"], "state": out["state"],
                           "tokens_removed": out["tokens_removed"]}))


@keys.command("installations")
@click.option("--org", "org_id", default=None)
@click.option("--key", "key_id", default=None)
@_db_opt
@handle_errors
def keys_installations(org_id: str | None, key_id: str | None, accounts_db: Path | None) -> None:
    """List the installations enrolled with SDK keys."""
    echo_json(_store(accounts_db).list_installations(org_id, key_id))


# ---- usage and audit ----------------------------------------------------------------------------------
@click.command("usage")
@click.option("--org", "org_id", default=None, help="Only this organisation")
@click.option("--key", "key_id", default=None, help="Only this key (sk_...)")
@click.option("--since", default=None, help="First day, YYYY-MM-DD (UTC; default: 30 days ago)")
@click.option("--until", default=None, help="Last day, YYYY-MM-DD (UTC)")
@click.option("--by", type=click.Choice(["row", "day", "key", "model"]), default="row", show_default=True,
              help="Aggregate the rows")
@_db_opt
@click.option("--json", "as_json", is_flag=True, help="Machine-readable output")
@handle_errors
def usage(org_id: str | None, key_id: str | None, since: str | None, until: str | None, by: str,
          accounts_db: Path | None, as_json: bool) -> None:
    """Metered usage per key, model and UTC day (trips, rows, tokens, maskgen columns, bytes)."""
    import time

    from pgw.service.accounts import day_of

    since = since or day_of(time.time() - 29 * 86400)
    rows = _store(accounts_db).usage(org_id=org_id, key_id=key_id, since=since, until=until)
    fields = ("trips", "rows", "tokens", "maskgen_cols", "maskgen_requests", "bytes_in", "bytes_out")
    if by != "row":
        col = {"day": "day", "key": "key_id", "model": "model"}[by]
        agg: dict[str, dict[str, Any]] = {}
        for r in rows:
            a = agg.setdefault(r[col], {col: r[col], **dict.fromkeys(fields, 0)})
            for f in fields:
                a[f] += r[f]
        rows = sorted(agg.values(), key=lambda a: a[col], reverse=by == "day")
    if as_json:
        echo_json(rows)
        return
    if not rows:
        click.echo("(no usage)")
    for r in rows:
        head = "  ".join(str(r[c]) for c in ("day", "key_id", "model") if c in r)
        click.echo(f"{head}  trips={r['trips']} rows={r['rows']} tokens={r['tokens']:.0f} "
                   f"maskgen_cols={r['maskgen_cols']} in={r['bytes_in']} out={r['bytes_out']}")


@click.command("audit")
@click.option("--org", "org_id", default=None)
@click.option("--limit", type=int, default=50, show_default=True)
@_db_opt
@handle_errors
def audit(org_id: str | None, limit: int, accounts_db: Path | None) -> None:
    """Accounts audit log, newest first (no secrets in it)."""
    echo_json(_store(accounts_db).audit_events(org_id, limit=limit))


# ---- admin tokens -------------------------------------------------------------------------------------
@click.group("admin-token")
def admin_token() -> None:
    """Bearer tokens for the accounts admin API (/pgw/v1/admin; PGW_SERVICE_ACCOUNTS_ADMIN_API=true)."""


@admin_token.command("create")
@click.option("--name", required=True)
@click.option("--org", "org_id", default=None, help="Limit the token to one organisation (default: global)")
@click.option("--out", "out_file", type=click.Path(path_type=Path, dir_okay=False), default=None,
              help="Write the token to this file (0600) instead of stdout")
@_db_opt
@handle_errors
def admin_token_create(name: str, org_id: str | None, out_file: Path | None, accounts_db: Path | None) -> None:
    """Create an admin token (shown once)."""
    from pgw.core.fileio import write_private

    token, rec = _store(accounts_db).create_admin_token(name, org_id, actor=_actor())
    if out_file is not None:
        write_private(out_file, token + "\n", append=False)
        echo_json({**rec, "token_file": str(out_file)})
    else:
        echo_json({**rec, "token": token})


@admin_token.command("list")
@_db_opt
@handle_errors
def admin_token_list(accounts_db: Path | None) -> None:
    """List admin tokens (never the token or its hash)."""
    echo_json(_store(accounts_db).list_admin_tokens())


@admin_token.command("revoke")
@click.argument("token_id")
@_db_opt
@handle_errors
def admin_token_revoke(token_id: str, accounts_db: Path | None) -> None:
    """Revoke an admin token."""
    echo_json(_store(accounts_db).revoke_admin_token(token_id, actor=_actor()))


def register(service: click.Group) -> None:
    """Add the accounts commands to ``pgw service``."""
    for cmd in (orgs, keys, usage, audit, admin_token):
        service.add_command(cmd)
