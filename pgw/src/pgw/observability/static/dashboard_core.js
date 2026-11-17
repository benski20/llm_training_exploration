/*PGW_CORE_BEGIN*/
const PGWCore = (function () {
  "use strict";
  // Parse Prometheus text exposition into {name: [{labels: {...}, value: Number}]}.
  function parseLabels(s) {
    const out = {};
    let i = 0;
    while (i < s.length) {
      while (i < s.length && (s[i] === "," || s[i] === " ")) i++;
      const eq = s.indexOf("=", i);
      if (eq < 0) break;
      const key = s.slice(i, eq).trim();
      i = eq + 1;
      if (s[i] !== '"') break;
      i++;
      let val = "";
      while (i < s.length && s[i] !== '"') {
        if (s[i] === "\\" && i + 1 < s.length) {
          const n = s[i + 1];
          val += n === "n" ? "\n" : n;
          i += 2;
        } else {
          val += s[i++];
        }
      }
      i++;
      out[key] = val;
    }
    return out;
  }
  function parseValue(v) {
    if (v === "+Inf" || v === "Inf") return Infinity;
    if (v === "-Inf") return -Infinity;
    if (v === "NaN") return NaN;
    return Number(v);
  }
  function parse(text) {
    const out = {};
    const lines = String(text || "").split("\n");
    for (const raw of lines) {
      const line = raw.trim();
      if (!line || line[0] === "#") continue;
      let name, labels = {}, rest;
      const brace = line.indexOf("{");
      const sp = line.search(/\s/);
      if (brace >= 0 && (sp < 0 || brace < sp)) {
        name = line.slice(0, brace);
        const close = line.lastIndexOf("}");
        if (close < brace) continue;
        labels = parseLabels(line.slice(brace + 1, close));
        rest = line.slice(close + 1).trim();
      } else {
        if (sp < 0) continue;
        name = line.slice(0, sp);
        rest = line.slice(sp).trim();
      }
      const value = parseValue(rest.split(/\s+/)[0]);
      if (!/^[a-zA-Z_:][a-zA-Z0-9_:]*$/.test(name)) continue;
      (out[name] = out[name] || []).push({ labels: labels, value: value });
    }
    return out;
  }
  function matches(labels, filter) {
    if (!filter) return true;
    for (const k in filter) if (labels[k] !== filter[k]) return false;
    return true;
  }
  // Sum of all series of `name` matching `filter`; null if the metric is absent.
  function sum(m, name, filter) {
    const s = m[name];
    if (!s) return null;
    let t = 0, any = false;
    for (const e of s) if (matches(e.labels, filter) && isFinite(e.value)) { t += e.value; any = true; }
    return any ? t : null;
  }
  function max(m, name, filter) {
    const s = m[name];
    if (!s) return null;
    let t = null;
    for (const e of s) if (matches(e.labels, filter) && isFinite(e.value)) t = t === null ? e.value : Math.max(t, e.value);
    return t;
  }
  // {labelValue: sum} grouped by one label.
  function groupBy(m, name, label, filter) {
    const out = {};
    for (const e of m[name] || []) {
      if (!matches(e.labels, filter)) continue;
      const k = e.labels[label] === undefined ? "" : e.labels[label];
      out[k] = (out[k] || 0) + e.value;
    }
    return out;
  }
  // Per-second rate of a counter between two parsed scrapes (null if unknown; resets give null).
  function rate(prev, cur, name, dt, filter) {
    if (!prev || !(dt > 0)) return null;
    const a = sum(prev, name, filter), b = sum(cur, name, filter);
    if (a === null || b === null) return null;
    if (b < a) return null;
    return (b - a) / dt;
  }
  // Cumulative histogram buckets {le: count} summed over series matching `filter`.
  function buckets(m, name, filter) {
    const out = {};
    for (const e of m[name + "_bucket"] || []) {
      if (!matches(e.labels, filter)) continue;
      const le = e.labels.le;
      out[le] = (out[le] || 0) + e.value;
    }
    return out;
  }
  // Quantile from cumulative buckets (linear interpolation inside the bucket, like PromQL).
  function quantile(q, bk) {
    const les = Object.keys(bk).map(function (k) { return [parseValue(k), bk[k]]; })
      .sort(function (x, y) { return x[0] - y[0]; });
    if (!les.length) return null;
    const total = les[les.length - 1][1];
    if (!(total > 0)) return null;
    const rank = q * total;
    let prevLe = 0, prevC = 0;
    for (const [le, c] of les) {
      if (c >= rank) {
        if (le === Infinity) return prevLe;
        if (c === prevC) return le;
        return prevLe + (le - prevLe) * (rank - prevC) / (c - prevC);
      }
      prevLe = le; prevC = c;
    }
    return prevLe;
  }
  // Histogram summary over the interval prev->cur (falls back to cumulative without prev).
  function histSummary(prev, cur, name, filter) {
    const cb = buckets(cur, name, filter);
    if (!Object.keys(cb).length) return null;
    let bk = cb;
    let s = sum(cur, name + "_sum", filter) || 0, c = sum(cur, name + "_count", filter) || 0;
    if (prev) {
      const pb = buckets(prev, name, filter);
      const ps = sum(prev, name + "_sum", filter), pc = sum(prev, name + "_count", filter);
      if (ps !== null && pc !== null && c - pc > 0) {
        const d = {};
        for (const k in cb) d[k] = cb[k] - (pb[k] || 0);
        bk = d; s = s - ps; c = c - pc;
      } else if (pc !== null && c - pc === 0) {
        return { count: 0, mean: null, p50: null, p95: null, interval: true, total: c };
      }
    }
    return { count: c, mean: c > 0 ? s / c : null, p50: quantile(0.5, bk), p95: quantile(0.95, bk),
             interval: bk !== cb, total: sum(cur, name + "_count", filter) };
  }
  function fmtNum(v, digits) {
    if (v === null || v === undefined || !isFinite(v)) return "–";
    const a = Math.abs(v);
    if (a >= 1e12) return (v / 1e12).toFixed(2) + "T";
    if (a >= 1e9) return (v / 1e9).toFixed(2) + "G";
    if (a >= 1e6) return (v / 1e6).toFixed(2) + "M";
    if (a >= 1e4) return (v / 1e3).toFixed(1) + "k";
    return v.toFixed(digits === undefined ? (a >= 100 || Number.isInteger(v) ? 0 : 2) : digits);
  }
  function fmtSeconds(v) {
    if (v === null || v === undefined || !isFinite(v)) return "–";
    if (v < 1e-3) return (v * 1e6).toFixed(0) + " µs";
    if (v < 1) return (v * 1e3).toFixed(v < 0.01 ? 2 : 0) + " ms";
    if (v < 120) return v.toFixed(2) + " s";
    if (v < 7200) return (v / 60).toFixed(1) + " min";
    if (v < 172800) return (v / 3600).toFixed(1) + " h";
    return (v / 86400).toFixed(1) + " d";
  }
  // Everything the page shows, from two scrapes `dt` seconds apart (prev may be null).
  function summarize(prev, cur, dt) {
    const masks = sum(cur, "pgw_masks_available_slots");
    const burn = rate(prev, cur, "pgw_masks_consumed_slots_total", dt);
    const out = {
      masks: {
        available: masks,
        lowWater: (max(cur, "pgw_masks_low_water") || 0) > 0,
        burnRate: burn,
        burnedRate: rate(prev, cur, "pgw_masks_burned_slots_total", dt),
        consumedTotal: sum(cur, "pgw_masks_consumed_slots_total"),
        secondsLeft: masks !== null && burn !== null && burn > 0 ? masks / burn : null,
      },
      gateway: {
        present: !!(cur.pgw_gateway_requests_total || cur.pgw_gateway_tokens_total ||
                    cur.pgw_gateway_noise_alarm || cur.pgw_gateway_ttft_seconds_bucket),
        completionTps: rate(prev, cur, "pgw_gateway_tokens_total", dt, { kind: "completion" }),
        promptTps: rate(prev, cur, "pgw_gateway_tokens_total", dt, { kind: "prompt" }),
        completionTotal: sum(cur, "pgw_gateway_tokens_total", { kind: "completion" }),
        requestRate: rate(prev, cur, "pgw_gateway_requests_total", dt),
        requestsTotal: sum(cur, "pgw_gateway_requests_total"),
        errorsTotal: null,
        activeSessions: sum(cur, "pgw_gateway_active_sessions"),
        ttft: histSummary(prev, cur, "pgw_gateway_ttft_seconds"),
        tpot: histSummary(prev, cur, "pgw_gateway_tpot_seconds"),
        roundTrip: histSummary(prev, cur, "pgw_gateway_round_trip_seconds"),
      },
      integrity: {
        total: sum(cur, "pgw_gateway_integrity_failures_total"),
        recent: null,
        byCheck: groupBy(cur, "pgw_gateway_integrity_failures_total", "check"),
      },
      noise: {
        known: !!cur.pgw_gateway_noise_alarm,
        alarm: (max(cur, "pgw_gateway_noise_alarm") || 0) > 0,
        byModel: groupBy(cur, "pgw_gateway_noise_alarm", "model"),
      },
      service: {
        present: !!(cur.pgw_service_requests_total || cur.pgw_service_inflight),
        requestRate: rate(prev, cur, "pgw_service_requests_total", dt),
        rowRate: rate(prev, cur, "pgw_service_rows_total", dt),
        inflight: sum(cur, "pgw_service_inflight"),
        queueDepth: sum(cur, "pgw_service_queue_depth"),
        latency: histSummary(prev, cur, "pgw_service_request_seconds"),
        bytesInRate: rate(prev, cur, "pgw_service_bytes_in_total", dt),
      },
      factory: {
        present: !!(cur.pgw_factory_slots_generated_total || cur.pgw_factory_slots_per_second),
        slotsPerSecond: sum(cur, "pgw_factory_slots_per_second"),
        generatedTotal: sum(cur, "pgw_factory_slots_generated_total"),
      },
    };
    let errs = 0, anyReq = false;
    for (const e of cur.pgw_gateway_requests_total || []) {
      anyReq = true;
      const st = String(e.labels.status || "");
      const code = parseInt(st, 10);
      if ((isFinite(code) && code >= 400) || st === "error") errs += e.value;
    }
    out.gateway.errorsTotal = anyReq ? errs : null;
    if (prev && out.integrity.total !== null) {
      const p = sum(prev, "pgw_gateway_integrity_failures_total");
      out.integrity.recent = p === null ? out.integrity.total : Math.max(0, out.integrity.total - p);
    }
    out.masks.slotsPerToken = null;
    if (burn !== null && out.gateway.completionTps) out.masks.slotsPerToken = burn / out.gateway.completionTps;
    return out;
  }
  return { parse: parse, parseLabels: parseLabels, sum: sum, max: max, groupBy: groupBy, rate: rate,
           buckets: buckets, quantile: quantile, histSummary: histSummary, summarize: summarize,
           fmtNum: fmtNum, fmtSeconds: fmtSeconds };
})();
if (typeof module !== "undefined") module.exports = PGWCore;
/*PGW_CORE_END*/
