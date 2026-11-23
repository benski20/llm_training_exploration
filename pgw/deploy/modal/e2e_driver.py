"""The scripted end-to-end test of the Modal demo (``launch.py --e2e``); no Modal imports.

Runs in the customer's container (``customer.py``) or in the co-located container (``colocated.py``):
waits for the provider's enrollment code in the ``pgw-live`` Dict (passed in as ``live``, read with
``.get`` only), drives the setup wizard through the gateway's URL, sends two greedy chats, recomputes
the answer with the plaintext reference backend on the same pack and returns the comparison plus the
gateway's client-reported round-trip latency (``trip_gate.py``) and speed settings. Both latency
reports name the network path the gateway used to reach the provider (``network_path``: loopback,
private (i6pn) or public (tunnel)), taken from the service URL the gateway enrolled with.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pgw_modal_boot as boot
import trip_gate
import wizard_client as wiz

PROMPT = [{"role": "user", "content": "Explain in two sentences why one-time pads are secure."}]


def run(live: Any, run_id: str, url: str, token: str, deadline: float, local_url: str,
        config_dir: Path = Path("/srv/pgw/config"), spec_rows: int = 0, sustain_s: float = 0.0,
        sampler_path: str | None = None, float_threads: int | None = None,
        float_ops: str | None = None, ab: list[tuple[str, Any]] | None = None, masked_head: bool = False,
        ab_cleanup: Any = None, ab_chats: int = 3, sustain_min_stock: int = 0,
        private: dict[str, Any] | None = None, ref_int_threads: int | None = None,
        fused2_bits: int | None = None, fused2_ring: str = "K56A12C53",
        fused2_tables: str | None = None) -> dict[str, Any]:
    """Wizard + two chats + exactness check; see the module docstring.

    ``ref_int_threads`` (``--gateway-env PGW_MODAL_REF_INT_THREADS=N``): the plaintext reference's exact integer
    products on N threads (``wizard_client.plaintext_reference(int_threads=)``; same bits), for large targets.

    ``spec_rows`` (launch.py --spec-rows): the gateway runs strict speculation with that many rows per pass, so
    the wizard waits for enough mask slots for both chats and the warm window starts after the first chat's
    dummy passes have drained."""
    while True:
        prov = live.get("provider") or {}
        if prov.get("run_id") == run_id and prov.get("code", "").startswith("PGW1-"):
            break
        if time.time() > deadline:
            raise RuntimeError("provider did not publish an enrollment code")
        time.sleep(5)
    model_name = prov.get("model") or boot.MODEL_NAME  # the target the provider serves (launch.py --model)
    # strict speculation reserves spec_rows slots per generated token up front: wait for both chats' slots
    res = wiz.run_wizard(url, token, prov["code"], model_name, timeout_s=max(60, deadline - time.time()),
                         min_masks=boot.e2e_min_masks(spec_rows))
    wiz.wait_gateway_ready(url)
    key = wiz.create_api_key(url, res["admin_token"])
    adm = res["admin_token"]
    path = boot.network_path((res.get("enroll") or {}).get("service_url") or prov.get("service_url"))

    snap_errors: list[str] = []

    def snap() -> tuple[str, dict]:
        # The gateway's own counters, read on loopback (not through Modal's edge). A failed read leaves
        # the latency unmeasured (the gate then reports "no round trips measured"), never the exactness check.
        try:
            return (wiz.metrics_text(local_url, adm),
                    wiz.client_timing(wiz.admin_get(local_url, adm, "/admin/status"), model_name))
        except Exception as exc:
            snap_errors.append(f"{type(exc).__name__}: {str(exc)[:200]}")
            return "", {}

    s0 = snap()
    out = wiz.chat(url, key, model_name, PROMPT, 32)
    if private is not None:  # an in-run plain phase (colocated.py --ab-plain) reuses these; never in the result
        private.update(api_key=key, admin=adm, ref_out=out)
    # Strict speculation: the dummy passes of the first chat run after its answer, and the next request of the
    # same conversation (this API key) waits for them; wait here so the warm window holds one chat's passes.
    drain_s = wiz.wait_idle(local_url, adm, model_name) if spec_rows else 0.0
    s1 = snap()
    tw0 = time.time()
    out2 = wiz.chat(url, key, model_name, PROMPT, 32)  # second run: determinism + warm timing
    tw1 = time.time()
    s2 = snap()
    # The warm chat (the second one) decides launch.py's --max-trip-ms gate; both chats are reported too.
    trips: dict[str, Any] = {"warm": trip_gate.trip_latency(s1[0], s2[0], s1[1], s2[1]),
                             "both_chats": trip_gate.trip_latency(s0[0], s2[0], s0[1], s2[1]),
                             "path": path}
    if snap_errors:
        trips["errors"] = snap_errors
    # The same warm chat sent to the gateway on loopback (no Modal HTTPS edge on the way): what the
    # gateway itself delivers; reported next to the edge figure, never instead of it. Same answer required.
    out3 = wiz.chat(local_url, key, model_name, PROMPT, 32) if not spec_rows else None
    # Streamed on loopback (twice; the second is the warm figure): time to first token and the decode rate
    # (n-1)/(t_last - t_first), next to the wall-time figures above. Same answer required (stream_identical).
    # Strict speculation: each stream starts once the previous answer's dummy passes have drained (they would
    # otherwise count as time to first token: a conversation's next request waits for them).
    idle = _idle_fn(local_url, adm, model_name, spec_rows)
    streams: list[dict[str, Any]] = []
    spec0 = _spec_stats(local_url, adm, model_name) if spec_rows else None
    for _ in range(2):
        try:
            drain = idle()
            streams.append(wiz.chat_stream(local_url, key, model_name, PROMPT, 32))
            if spec_rows:
                streams[-1]["drain_before_s"] = round(drain, 2)
        except Exception as exc:  # report, never fail the exactness result
            streams.append({"error": f"{type(exc).__name__}: {str(exc)[:300]}"})
    stream_spec = _spec_delta(spec0, _spec_stats(local_url, adm, model_name)) if spec_rows else None
    def ab_snap() -> tuple[str, dict, dict]:  # trip counters and the local-linear timing (gateway, loopback)
        m, ct = snap()
        try:
            st0 = wiz.admin_get(local_url, adm, "/admin/status")
            sp0 = next((x for x in st0.get("models", []) if x.get("name") == model_name), {}).get("speed") or {}
        except Exception:
            sp0 = {}
        return m, ct, dict(sp0.get("local_linear_timing") or {})

    ab_out = (_ab_streams(local_url, key, model_name, out, ab, ab_cleanup, ab_chats, ab_snap, idle=idle)
              if ab else None)
    st = json.loads((config_dir / "setup" / "state.json").read_text())
    t = time.time()
    ref_ring = fused2_ring if fused2_bits is not None else (st["model"].get("ring") or boot.RING)
    ref = wiz.plaintext_reference(st["model"]["pack_path"], ref_ring, PROMPT, 32,
                                  float_threads=float_threads, float_ops=float_ops, int_threads=ref_int_threads,
                                  masked_head=masked_head, fused2_bits=fused2_bits, fused2_tables=fused2_tables)
    ref_s = time.time() - t
    cmp = wiz.compare(out, ref)
    sustain: dict[str, Any] | None = None
    if sustain_s > 0:  # chats back to back (launch.py --sustain-s): tokens/s and factory keep-up under decode load
        sustain = _sustain(local_url, key, adm, model_name, out, sustain_s, sampler_path, sustain_min_stock,
                           probe=snap, spec_rows=spec_rows)
    status = wiz.admin_get(url, res["admin_token"], "/admin/status")
    masks = wiz.admin_get(url, res["admin_token"], "/admin/masks")
    model_status = next((m for m in status.get("models", []) if m.get("name") == model_name), {})
    speed = dict(model_status.get("speed") or {})
    speed.setdefault("rtt_path", path)
    warm_tokens = int(((out2.get("usage") or {}).get("completion_tokens")) or 0)
    if warm_tokens and trips["warm"].get("trips"):
        speed["warm_trips_per_token"] = round(trips["warm"]["trips"] / warm_tokens, 2)
    if warm_tokens and out2.get("_wall_s"):
        speed["warm_tokens_per_s"] = round(warm_tokens / float(out2["_wall_s"]), 2)
    if out3 is not None and out3.get("_wall_s"):
        t3 = int(((out3.get("usage") or {}).get("completion_tokens")) or 0)
        speed["warm_loopback_tokens_per_s"] = round(t3 / float(out3["_wall_s"]), 2)
    if streams and "error" not in streams[-1]:
        ws = streams[-1]
        speed["warm_stream_ttft_s"] = ws["ttft_s"]
        speed["warm_stream_decode_tokens_per_s"] = ws["decode_tokens_per_s"]
        speed["warm_stream_decode_ms_per_token"] = ws["decode_ms_per_token"]
        speed["warm_stream_wall_s"] = round(ws["_wall_s"], 3)
    if spec_rows:
        speed["spec_drain_wait_s"] = round(drain_s, 1)
        speed["stream_spec"] = stream_spec
        if warm_tokens:
            speed["warm_answer_tokens_per_s"] = round(warm_tokens / max(1e-9, out2["_wall_s"]), 2)
    return {
        "model": model_name, "spec_rows": spec_rows,
        "timings": res["timings"], "verify": res["verify"], "enroll": {k: res["enroll"].get(k) for k in (
            "tenant", "service_url", "client_cert", "ca_fingerprint")},
        "chat_text": out["choices"][0]["message"]["content"], "usage": out.get("usage"),
        "chat_wall_s": out["_wall_s"], "chat2_wall_s": out2["_wall_s"],
        "repeat_identical": out["choices"][0]["message"]["content"] == out2["choices"][0]["message"]["content"]
        and [x["logprob"] for x in out["choices"][0]["logprobs"]["content"]]
        == [x["logprob"] for x in out2["choices"][0]["logprobs"]["content"]],
        "loopback_repeat_identical": None if out3 is None else (
            out3["choices"][0]["message"]["content"] == out["choices"][0]["message"]["content"]
            and [x["logprob"] for x in out3["choices"][0]["logprobs"]["content"]]
            == [x["logprob"] for x in out["choices"][0]["logprobs"]["content"]]),
        "stream_identical": None if not streams else all(
            "error" not in x and x["text"] == out["choices"][0]["message"]["content"]
            and x["logprobs"] == [y["logprob"] for y in out["choices"][0]["logprobs"]["content"]] for x in streams),
        "streams": [{k: v for k, v in x.items() if k not in ("text", "logprobs")} for x in streams],
        "reference_text": ref["text"], "reference_s": ref_s, "compare": cmp,
        "admin_status": status, "admin_masks": masks, "trip_latency": trips, "speed": speed,
        "ab": ab_out,
        "network_path": path, "sustain": sustain, "warm_window": [tw0, tw1],
    }


def _idle_fn(local_url: str, adm: str, model_name: str, spec_rows: int, quiet_s: float = 1.5) -> Any:
    """``() -> seconds``: with strict speculation, wait until the previous answer's dummy passes are done (the
    model's round-trip counters still for ``quiet_s``); without it, nothing to wait for (0)."""
    if not spec_rows:
        return lambda: 0.0
    return lambda: wiz.wait_idle(local_url, adm, model_name, quiet_s=quiet_s, poll_s=0.25)


def _spec_stats(local_url: str, adm: str, model_name: str) -> dict[str, Any] | None:
    """The gateway's speculation counters (``spec_stats``, ``drafter_stats`` of ``/admin/status``; local)."""
    try:
        st = wiz.admin_get(local_url, adm, "/admin/status")
        sp = next((x for x in st.get("models", []) if x.get("name") == model_name), {}).get("speed") or {}
        return {"spec": dict(sp.get("spec_stats") or {}), "drafter": dict(sp.get("drafter_stats") or {})}
    except Exception:
        return None


def _spec_delta(a: dict[str, Any] | None, b: dict[str, Any] | None) -> dict[str, Any] | None:
    """Speculation counters between two :func:`_spec_stats` snapshots: verify passes with a real answer row
    (``spec_passes``), accepted drafts, tokens per verify pass ((accepted + passes) / passes: each pass yields its
    accepted drafts plus the target's own token), dummy passes, draft-model ms per planning call and the
    pipelined drafter's hit rate, foreground/background steps and wait."""
    if not a or not b:
        return None
    d = {k: round(float(b["spec"].get(k, 0.0)) - float(a["spec"].get(k, 0.0)), 3) for k in b["spec"]}
    out: dict[str, Any] = {"counters": d}
    if d.get("spec_passes"):
        out["tokens_per_pass"] = round((d.get("spec_accepted", 0.0) + d["spec_passes"]) / d["spec_passes"], 3)
        out["draft_rows_per_pass"] = round(d.get("spec_rows", 0.0) / d["spec_passes"], 3)
    if d.get("draft_calls"):
        out["draft_ms_per_call"] = round(d.get("draft_ms", 0.0) / d["draft_calls"], 2)
    if b.get("drafter"):
        dd = {k: round(float(b["drafter"].get(k, 0.0)) - float((a.get("drafter") or {}).get(k, 0.0)), 3)
              for k in b["drafter"]}
        out["drafter"] = dd
        n = dd.get("hits", 0.0) + dd.get("misses", 0.0)
        if n:
            out["drafter_hit_rate"] = round(dd.get("hits", 0.0) / n, 3)
        if dd.get("bg_steps"):
            out["draft_ms_per_bg_step"] = round(dd.get("bg_ms", 0.0) / dd["bg_steps"], 2)
        if dd.get("calls"):
            out["drafter_wait_ms_per_call"] = round(dd.get("wait_ms", 0.0) / dd["calls"], 2)
            out["drafter_fg_ms_per_call"] = round(dd.get("ms", 0.0) / dd["calls"], 2)
    return out


def plain_phase(local_url: str, key: str, adm: str, model_name: str, ref_out: dict[str, Any],
                restart: Any, chats: int = 3, sustain_s: float = 0.0, sustain_min_stock: int = 0,
                sampler_path: str | None = None) -> dict[str, Any]:
    """In-run A/B (``launch.py --ab-plain``): ``restart()`` restarts the gateway without speculation (same pack,
    mask store, keys, service and speed settings; returns a report), then ``chats`` streamed chats on loopback
    (each must give the reference answer: plain and strict speculative decoding are both exact) with their trip
    latency, and optionally a back-to-back window (:func:`_sustain`) of ``sustain_s`` seconds."""
    out: dict[str, Any] = {"phase": "plain"}
    t0 = time.time()
    out["restart"] = restart()
    wiz.wait_gateway_ready(local_url, timeout_s=1200)
    out["restart_s"] = round(time.time() - t0, 1)

    def snap() -> tuple[str, dict]:
        try:
            return (wiz.metrics_text(local_url, adm),
                    wiz.client_timing(wiz.admin_get(local_url, adm, "/admin/status"), model_name))
        except Exception:
            return "", {}

    want_text = ref_out["choices"][0]["message"]["content"]
    want_lp = [y["logprob"] for y in ref_out["choices"][0]["logprobs"]["content"]]
    runs: list[dict[str, Any]] = []
    try:
        runs.append(wiz.chat_stream(local_url, key, model_name, PROMPT, 32))  # warm-up after the restart
        p0 = snap()
        runs += [wiz.chat_stream(local_url, key, model_name, PROMPT, 32) for _ in range(chats)]
        p1 = snap()
        lat = trip_gate.trip_latency(p0[0], p1[0], p0[1], p1[1])
        out["trips"] = {k: lat.get(k) for k in ("trips", "mean_ms", "p50_ms", "rows_1_mean_ms",
                                                "rows_1_service_ms", "rows_1_wire_ms")}
        toks = sum(max(0, int(r["tokens"])) for r in runs[1:])
        if toks and lat.get("trips"):
            out["trips_per_token"] = round(lat["trips"] / toks, 2)
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
    out["identical"] = bool(runs) and all(r["text"] == want_text and r["logprobs"] == want_lp for r in runs)
    out["decode_tokens_per_s"] = [r["decode_tokens_per_s"] for r in runs[1:]]
    out["decode_ms_per_token"] = [r["decode_ms_per_token"] for r in runs[1:]]
    out["ttft_s"] = [r["ttft_s"] for r in runs[1:]]
    if sustain_s > 0 and "error" not in out:
        out["sustain"] = _sustain(local_url, key, adm, model_name, ref_out, sustain_s, sampler_path,
                                  sustain_min_stock, probe=snap, spec_rows=0)
    return out


def _ab_streams(local_url: str, key: str, model_name: str, ref_out: dict[str, Any], phases: list[tuple[str, Any]],
                cleanup: Any, chats: int, probe: Any = None, idle: Any = None) -> list[dict[str, Any]]:
    """Diagnostics (``PGW_MODAL_AB``): per phase, apply its action (re-pin or pause a party process, see
    ``pgw_modal_boot.ab_phases``), then ``chats`` streamed chats on loopback; each must give the reference answer."""
    want_text = ref_out["choices"][0]["message"]["content"]
    want_lp = [y["logprob"] for y in ref_out["choices"][0]["logprobs"]["content"]]
    rows: list[dict[str, Any]] = []
    try:
        for label, act in phases:
            row: dict[str, Any] = {"phase": label}
            try:
                row["action"] = act()
                time.sleep(0.5)
                p0 = probe() if probe is not None else None
                a, t0 = boot.thread_cpu("gateway"), time.time()
                runs = []
                for _ in range(chats):
                    if idle is not None:
                        idle()
                    runs.append(wiz.chat_stream(local_url, key, model_name, PROMPT, 32))
                row["gateway_threads"] = boot.thread_deltas(a, boot.thread_cpu("gateway"), time.time() - t0)
                if p0 is not None:
                    p1 = probe()
                    lat = trip_gate.trip_latency(p0[0], p1[0], p0[1], p1[1])
                    row["trips"] = {k: lat.get(k) for k in ("trips", "mean_ms", "p50_ms", "rows_1_mean_ms",
                                                            "rows_1_service_ms", "rows_1_wire_ms")}
                    dc = int(p1[2].get("decode_calls", 0)) - int(p0[2].get("decode_calls", 0))
                    if dc > 0:
                        row["local_decode_us_per_call"] = round(
                            1e6 * (float(p1[2].get("decode_s", 0.0)) - float(p0[2].get("decode_s", 0.0))) / dc, 1)
                row["identical"] = all(r["text"] == want_text and r["logprobs"] == want_lp for r in runs)
                row["decode_ms_per_token"] = [r["decode_ms_per_token"] for r in runs]
                row["ttft_s"] = [r["ttft_s"] for r in runs]
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
            rows.append(row)
    finally:
        if cleanup is not None:
            try:
                cleanup()
            except Exception:  # noqa: S110 - best effort; the run's shutdown terminates the parties anyway
                pass
    return rows


def _stock(local_url: str, adm: str, model_name: str) -> int | None:
    try:
        m = wiz.admin_get(local_url, adm, "/admin/masks")
        v = ((m.get("models") or {}).get(model_name) or {}).get("available_slots")
        return None if v is None else int(v)
    except Exception:
        return None


def _sustain(local_url: str, key: str, adm: str, model_name: str, ref_out: dict[str, Any], sustain_s: float,
             sampler_path: str | None, min_stock: int, probe: Any = None, spec_rows: int = 0) -> dict[str, Any]:
    """Streamed greedy chats back to back on loopback for ``sustain_s``: per-chat decode rate, exactness (every
    chat must equal the first one, and its first tokens the exactness-checked chat), and the mask stock every 2 s
    (``stock``: ``[seconds since t0, slots]``) so the factory's keep-up is visible (colocated.py adds the
    factory's production from its log). ``min_stock``: first wait (at most 240 s) for that many slots."""
    import threading

    t_wait = time.time()
    s_wait = _stock(local_url, adm, model_name)
    while min_stock and (s_wait is None or s_wait < min_stock) and time.time() - t_wait < 240:
        time.sleep(2)
        s_wait = _stock(local_url, adm, model_name)
    waited = round(time.time() - t_wait, 1)
    if sampler_path:  # PGW_STACK_SAMPLER in the gateway: count the sustain window alone
        Path(sampler_path + ".reset").touch()
    series: list[list[float]] = []
    stop = threading.Event()
    fac0, gw0 = boot.thread_cpu("factory"), boot.thread_cpu("gateway")
    # trip counters and speculation counters over the window (gateway-local, loopback)
    p0 = probe() if probe is not None else None
    spec0 = _spec_stats(local_url, adm, model_name) if spec_rows else None
    idle = _idle_fn(local_url, adm, model_name, spec_rows, quiet_s=1.0)
    drain_total = 0.0
    t0 = time.time()

    def sample() -> None:
        while True:
            v = _stock(local_url, adm, model_name)
            if v is not None:
                series.append([round(time.time() - t0, 1), v])
            if stop.wait(2.0):
                return

    th = threading.Thread(target=sample, daemon=True)
    th.start()
    want_lp = [y["logprob"] for y in ref_out["choices"][0]["logprobs"]["content"]]
    first: dict[str, Any] | None = None
    toks = n = 0
    dec_rates: list[float] = []
    ttfts: list[float] = []
    starts: list[float] = []
    dec_s = dec_n = 0.0
    identical = True
    err = None
    while time.time() - t0 < sustain_s:
        try:
            # strict speculation: the previous answer's dummy passes first (else they count as time to first token)
            drain_total += idle()
            if time.time() - t0 >= sustain_s:
                break
            starts.append(round(time.time() - t0, 1))
            o = wiz.chat_stream(local_url, key, model_name, PROMPT, 64)
        except Exception as exc:  # e.g. masks ran out: report, never fail the exactness result
            err = f"{type(exc).__name__}: {str(exc)[:300]}"
            break
        if first is None:
            first = o
            identical = o["logprobs"][:len(want_lp)] == want_lp  # its first tokens = the exactness-checked chat
        else:
            identical = identical and o["text"] == first["text"] and o["logprobs"] == first["logprobs"]
        toks += int(o["tokens"])
        n += 1
        if o.get("decode_tokens_per_s"):
            dec_rates.append(float(o["decode_tokens_per_s"]))
            dec_s += float(o["decode_s"])
            dec_n += int(o["tokens"]) - 1
        if o.get("ttft_s") is not None:
            ttfts.append(float(o["ttft_s"]))
    t1 = time.time()
    p1 = probe() if probe is not None else None
    trips: dict[str, Any] | None = None
    if p0 is not None and p1 is not None:
        lat = trip_gate.trip_latency(p0[0], p1[0], p0[1], p1[1])
        trips = {k: lat.get(k) for k in ("trips", "mean_ms", "p50_ms", "p90_ms", "rows_1_mean_ms",
                                         "rows_1_service_ms", "rows_1_wire_ms")}
    spec_d = _spec_delta(spec0, _spec_stats(local_url, adm, model_name)) if spec_rows else None
    threads = {"factory": boot.comm_deltas(fac0, boot.thread_cpu("factory"), t1 - t0),
               "gateway_top": boot.thread_deltas(gw0, boot.thread_cpu("gateway"), t1 - t0, top=6)}
    stop.set()
    th.join(5)
    ts = sorted(dec_rates)
    return {"t0": t0, "t1": t1, "chats": n, "tokens": toks, "seconds": round(t1 - t0, 2),
            "tokens_per_s": round(toks / max(1e-9, t1 - t0), 2), "error": err, "streamed": True,
            "identical": bool(identical and n > 0), "min_stock_wait_s": waited, "min_stock": min_stock,
            "decode_tokens_per_s_median": ts[len(ts) // 2] if ts else None,
            "decode_tokens_per_s_mean": round(sum(ts) / len(ts), 2) if ts else None,
            "decode_tokens_per_s_min": ts[0] if ts else None,
            "decode_tokens_per_s_all": dec_rates, "chat_start_s": starts[:len(dec_rates)],
            # all decoded tokens after each chat's first over all decode time (token-weighted)
            "decode_tokens_per_s_overall": round(dec_n / dec_s, 2) if dec_s else None, "ttft_s_median": sorted(ttfts)[len(ttfts) // 2] if ttfts else None,
            "threads": threads, "stock": series, "stock_start": series[0][1] if series else None,
            "stock_end": series[-1][1] if series else None,
            "stock_min": min(x[1] for x in series) if series else None,
            "trips": trips, "spec": spec_d, "spec_drain_s": round(drain_total, 1) if spec_rows else None}
