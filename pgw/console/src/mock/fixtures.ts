// Realistic fixture data for `npm run dev:mock` (never part of the production build).

import type { ApiKeyRecord, AuditEvent, Masks, ModelStatus, Overview, Status, Usage, UsageWindow } from "../api/types";

/** Deterministic PRNG so screenshots are stable. */
export function rng(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Pinned "now" for stable screenshots: rounded to the hour of the real clock.
export const NOW = Math.floor(Date.now() / 3_600_000) * 3_600_000 - 7 * 60_000;

// The pinned fingerprints of the supported Qwen2.5 instruct models (pgw.sdk.provisioning.PINNED_MODELS).
const FP = {
  q7b: "f7c6fb4f5f8fbc5e4149972c07afdc749de8e5f9adc604d30edb7eb94b834b67",
  q3b: "0ec0b8b24b2bc59632d9530e1bfd749b082c101f82c5ff819f8e8c5ff4d8d752",
  q05b: "d773a502fb976d45070332cbd4222ee30bc971d14c7f9c56a317805c230c0b44",
};
export const RING = "K32A12C27";
const M7 = "qwen2.5-7b-instruct";
const M3 = "qwen2.5-3b-instruct";
const M05 = "qwen2.5-0.5b-instruct";
const SERVICE = { server: "pgw-service/0.1.0", origin: "https://compute.pgw.example:8443", max_rows_per_item: 8192 };

export const models: ModelStatus[] = [
  {
    name: M7,
    mode: "masked",
    started: true,
    state: "ready",
    last_error: null,
    connect_attempts: 1,
    ready: true,
    reason: "ok",
    fingerprint: FP.q7b,
    fingerprint_short: FP.q7b.slice(0, 12),
    head_mode: "provider_int8",
    schedule: "standard",
    ring: RING,
    max_context: 32768,
    speculation: { mode: "draft_head", tokens: 2, strict: true, rows: 3 },
    prefix_cache: { mode: "conversation", hits: 41_208, misses: 18_377, mismatches: 412, reused_tokens: 52_904_118, stored: 59_581, evicted: 1_904, expired: 56_870, entries: 318, bytes: 702_545_920, max_bytes: 1_073_741_824, ttl_s: 600 },
    sessions: { active: 7, waiting: 1, max_active: 32, max_queue: 256, rejected: 3, draining: false, rejected_by_reason: { queue_full: 0, timeout: 3 } },
    breaker: { open: false, trips: 0, cooldown_s: 300, open_for_s: null, last_code: null },
    freivalds_rate: 1,
    service: SERVICE,
    masks_available: 18_412_560,
    noise_monitor: { state: "ok", alarm: false, samples: 4_812_221, windows: 1_204, duplicate_rows: 0, reasons: [] },
    client: { requests: 912_440, attempts: 912_611, retries: 171, failures: 0, bytes_sent: 41_221_908_112, bytes_received: 38_014_222_710, last_rtt_s: 0.0182, stream_requests: 0 },
    backend: {
      calls: 2_418_220,
      rows: 18_662_104,
      round_trips: 912_440,
      bytes_up: 41_221_908_112,
      bytes_down: 38_014_222_710,
      freivalds_checked: 2_418_220,
      freivalds_skipped: 0,
      integrity_failures: 0,
      remote_seconds: 15_812.4,
      local_seconds: 6_104.9,
    },
  },
  {
    name: M3,
    mode: "masked",
    started: true,
    state: "ready",
    ready: true,
    reason: "ok",
    fingerprint: FP.q3b,
    fingerprint_short: FP.q3b.slice(0, 12),
    head_mode: "provider_int8",
    schedule: "standard",
    ring: RING,
    max_context: 32768,
    speculation: { mode: "lookup", tokens: 2, strict: false, rows: null },
    prefix_cache: { mode: "off" },
    sessions: { active: 2, waiting: 0, max_active: 32, max_queue: 256, rejected: 0, draining: false },
    breaker: { open: false, trips: 1, cooldown_s: 300, open_for_s: null, last_code: null },
    freivalds_rate: 1,
    service: SERVICE,
    masks_available: 412_300,
    noise_monitor: { state: "ok", alarm: false, samples: 1_022_871, windows: 311, duplicate_rows: 0, reasons: [] },
    client: { requests: 208_116, attempts: 208_140, retries: 24, failures: 2, bytes_sent: 9_822_114_002, bytes_received: 9_100_871_224, last_rtt_s: 0.0211 },
    backend: {
      calls: 640_210,
      rows: 4_920_552,
      round_trips: 208_116,
      bytes_up: 9_822_114_002,
      bytes_down: 9_100_871_224,
      freivalds_checked: 640_210,
      freivalds_skipped: 0,
      integrity_failures: 0,
      remote_seconds: 3_902.1,
      local_seconds: 1_644.7,
    },
  },
  {
    name: M05,
    mode: "masked",
    started: true,
    state: "ready",
    last_error: null,
    connect_attempts: 1,
    ready: false,
    reason: "no mask slots",
    fingerprint: FP.q05b,
    fingerprint_short: FP.q05b.slice(0, 12),
    head_mode: "provider_int8",
    schedule: "standard",
    ring: RING,
    max_context: 32768,
    speculation: { mode: "off" },
    prefix_cache: { mode: "off" },
    sessions: { active: 0, waiting: 0, max_active: 16, max_queue: 128, rejected: 0, draining: false },
    breaker: { open: false, trips: 0, cooldown_s: 300, open_for_s: null, last_code: null },
    freivalds_rate: 1,
    service: SERVICE,
    masks_available: 0,
    noise_monitor: { state: "warming", alarm: false, samples: 0, windows: 0, duplicate_rows: 0, reasons: [] },
    client: { requests: 0, attempts: 1, retries: 0, failures: 0, last_rtt_s: 0.0179 },
    backend: {},
  },
];

export const status: Status = {
  version: 1,
  uptime_s: 6 * 86400 + 4 * 3600 + 1260,
  production_policy: true,
  models,
  api_keys: 5,
  egress_allowlist: ["https://compute.pgw.example:8443"],
  audit_log: "/var/log/pgw/audit.jsonl",
  telemetry: "disabled",
};

function bucketFor(w: UsageWindow): { n: number; bucket: number } {
  // the gateway's buckets (docs/api/console-api.md): 60 s, 900 s, 3600 s
  if (w === "1h") return { n: 60, bucket: 60 };
  if (w === "7d") return { n: 168, bucket: 3600 };
  return { n: 96, bucket: 900 };
}

export function usage(w: UsageWindow): Usage {
  const { n, bucket } = bucketFor(w);
  const r = rng(w.length * 7919 + n);
  const end = Math.floor(NOW / 1000 / bucket) * bucket;
  const series = Array.from({ length: n }, (_, i) => {
    const t = end - (n - 1 - i) * bucket;
    const hour = new Date(t * 1000).getHours() + new Date(t * 1000).getMinutes() / 60;
    const diurnal = 0.35 + 0.65 * Math.max(0, Math.sin(((hour - 6) / 24) * Math.PI * 2) * 0.5 + 0.5);
    const base = (bucket / 60) * 9 * diurnal * (0.85 + r() * 0.3);
    const requests = Math.round(base);
    const prompt = Math.round(requests * (410 + r() * 120));
    const completion = Math.round(requests * (180 + r() * 60));
    const errors = Math.round(requests * (0.004 + (r() < 0.08 ? 0.03 : 0) + r() * 0.004));
    const p50 = 1500 + diurnal * 900 + r() * 300;
    return {
      t,
      requests,
      prompt_tokens: prompt,
      completion_tokens: completion,
      errors,
      p50_ms: Math.round(p50),
      p95_ms: Math.round(p50 * (2.1 + r() * 0.6)),
      ttft_p50_ms: Math.round(240 + diurnal * 160 + r() * 60),
      masks_consumed: prompt + completion,
    };
  });
  const sumOf = (k: "requests" | "prompt_tokens" | "completion_tokens" | "errors" | "masks_consumed") => series.reduce((a, p) => a + p[k], 0);
  const share = { [M7]: 0.81, [M3]: 0.19 } as const;
  const by_model = Object.fromEntries(
    Object.entries(share).map(([m, s]) => [
      m,
      { requests: Math.round(sumOf("requests") * s), prompt_tokens: Math.round(sumOf("prompt_tokens") * s), completion_tokens: Math.round(sumOf("completion_tokens") * s), errors: Math.round(sumOf("errors") * s), masks_consumed: Math.round(sumOf("masks_consumed") * s) },
    ]),
  );
  return { window: w, bucket_s: bucket, series, by_model, by_key: {} };
}

export function overview(): Overview {
  const u = usage("24h");
  const tot = (k: "requests" | "errors") => u.series.reduce((a, p) => a + p[k], 0);
  const tokens = u.series.reduce((a, p) => a + p.prompt_tokens + p.completion_tokens, 0);
  const rate = (m: string) => {
    const c = (u.by_model?.[m]?.masks_consumed as number) ?? 0;
    return c / 86400;
  };
  return {
    version: "0.1.0",
    uptime_s: status.uptime_s,
    models: models.map((m) => ({
      name: m.name,
      state: m.state,
      ring: m.ring,
      ready: m.ready,
      reason: m.reason,
      masks_available: m.masks_available,
      masks_runway_s: m.masks_available && rate(m.name) ? m.masks_available / rate(m.name) : null,
    })),
    requests_24h: tot("requests"),
    tokens_24h: tokens,
    errors_24h: tot("errors"),
    integrity_failures_total: 0,
    freivalds_checked_total: 2_578_272,
    stream_requests_total: 611_208,
    privacy: { masked: true, ring: RING, freivalds_rate: 1, noise_monitor: { state: "ok", alarm: false } },
  };
}

export const masks = (): Masks => ({
  store: {
    root: "/var/lib/pgw/masks",
    store_id: "ms_7d2a91c4e08b4f63a1d5",
    generation: 14,
    consumed_total: 482_118_442,
    open_reservations: 9,
    segments_by_status: { active: 212, consumed: 1_880, shredded: 1_840 },
    key_provider: { kind: "kms", key_id: "pgw-kek-2026" },
    rollback_protection: { journal: true, counter: "tpm", hardware_counter: true, row_macs: true },
    low_water_slots: 500_000,
    inventory: [
      { fingerprint: FP.q7b, ring: RING, segments: 120, slots_total: 19_660_800, slots_available: 18_412_560, bytes_on_disk: 41_934_000_000 },
      { fingerprint: FP.q3b, ring: RING, segments: 44, slots_total: 1_441_792, slots_available: 412_300, bytes_on_disk: 1_602_000_000 },
    ],
    tainted: false,
  },
  models: {
    [M7]: { fingerprint: FP.q7b, ring: RING, available_slots: 18_412_560, low_water_slots: 500_000 },
    [M3]: { fingerprint: FP.q3b, ring: RING, available_slots: 412_300, low_water_slots: 500_000 },
    [M05]: { fingerprint: FP.q05b, ring: RING, available_slots: 0, low_water_slots: 500_000 },
  },
});

const iso = (ms: number) => new Date(ms).toISOString().replace(/\.\d{3}Z$/, `.${String(ms % 1000).padStart(3, "0")}Z`);

export const keys: ApiKeyRecord[] = [
  { key_id: "4b1e9c07d2a3", name: "support-assistant", scopes: ["inference"], models: [M7], requests_per_minute: 600, tokens_per_minute: 400_000, daily_token_quota: 60_000_000, max_concurrent_requests: 16, created_at: iso(NOW - 41 * 86400e3), disabled: false, meta: {}, tokens_today: 41_822_310, last_used_at: iso(NOW - 12e3) },
  { key_id: "a90f3d6e18c2", name: "contracts-summarizer", scopes: ["inference"], models: [], requests_per_minute: 120, tokens_per_minute: 200_000, daily_token_quota: 20_000_000, max_concurrent_requests: 4, created_at: iso(NOW - 19 * 86400e3), disabled: false, meta: {}, tokens_today: 18_904_114, last_used_at: iso(NOW - 4 * 60e3) },
  { key_id: "c3d82b5f7e01", name: "eval-harness", scopes: ["inference"], models: [M3], requests_per_minute: null, tokens_per_minute: null, daily_token_quota: null, max_concurrent_requests: null, created_at: iso(NOW - 6 * 86400e3), disabled: false, meta: {}, tokens_today: 2_210_480, last_used_at: iso(NOW - 3 * 3600e3) },
  { key_id: "77e0a4c9b6d5", name: "ops-automation", scopes: ["inference", "admin"], models: [], requests_per_minute: 60, tokens_per_minute: null, daily_token_quota: null, max_concurrent_requests: 2, created_at: iso(NOW - 88 * 86400e3), disabled: false, meta: {}, tokens_today: 0, last_used_at: iso(NOW - 2 * 86400e3) },
  { key_id: "e5b2f1a08c94", name: "console-playground", scopes: ["inference"], models: [], requests_per_minute: 30, tokens_per_minute: 50_000, daily_token_quota: 1_000_000, max_concurrent_requests: 2, created_at: iso(NOW - 6 * 86400e3), disabled: false, meta: { builtin: "true" }, tokens_today: 14_212, last_used_at: iso(NOW - 26 * 60e3) },
];

export function auditEvents(): AuditEvent[] {
  const r = rng(42);
  const out: AuditEvent[] = [];
  let t = NOW - 9_000;
  const inferenceKeys = keys.filter((k) => k.scopes.includes("inference"));
  for (let i = 0; i < 420; i++) {
    t -= Math.round(2_000 + r() * 40_000);
    const roll = r();
    if (roll < 0.03) {
      const action = ["key_create", "model_reload", "safety_reset", "console_login"][Math.floor(r() * 4)];
      out.push({ ts: iso(t), event: action === "console_login" ? "console.login" : `admin.${action === "key_create" ? "key_created" : action}`, action, key_id: "admin-token", target: action === "key_create" ? keys[Math.floor(r() * 3)].key_id : action === "console_login" ? undefined : M7, client: "10.20.4.18", status: action === "console_login" ? 200 : undefined });
      continue;
    }
    const k = inferenceKeys[Math.floor(r() * inferenceKeys.length)];
    const model = k.models[0] ?? (r() < 0.8 ? M7 : M3);
    const err = r();
    const status = err < 0.015 ? 429 : err < 0.022 ? 503 : err < 0.026 ? 400 : 200;
    const code = { 429: "rate_limited", 503: "mask_exhausted", 400: "protocol", 200: undefined }[status];
    const pt = Math.round(120 + r() * 900);
    const ct = status === 200 ? Math.round(20 + r() * 420) : 0;
    const stream = r() < 0.7;
    const ttft = Math.round(180 + r() * 380);
    out.push({
      ts: iso(t),
      event: "inference",
      request_id: `chatcmpl-${Math.floor(r() * 2 ** 32).toString(16).padStart(8, "0")}${Math.floor(r() * 2 ** 32).toString(16).padStart(8, "0")}`,
      key_id: k.key_id,
      key_name: k.name,
      model,
      endpoint: r() < 0.9 ? "chat" : "completions",
      status,
      code,
      stream,
      prompt_tokens: status === 200 ? pt : 0,
      completion_tokens: ct,
      masks_reserved: status === 200 ? pt + 512 : 0,
      masks_consumed: status === 200 ? pt + ct : 0,
      masks_burned: status === 200 ? 512 - ct : 0,
      queue_ms: Math.round(r() * 40 * 10) / 10,
      ttft_ms: status === 200 ? ttft : undefined,
      finish_reason: status === 200 ? (r() < 0.9 ? "stop" : "length") : undefined,
      duration_ms: status === 200 ? Math.round(ttft + ct * (35 + r() * 20)) : Math.round(2 + r() * 10),
      mode: "masked",
      ring: RING,
      retry_after_s: status === 429 ? 2 : undefined,
    });
  }
  return out;
}

export const CANNED = [
  "Masked inference keeps your data on your side. For every linear layer, the gateway adds a fresh one-time pad to the activations before sending them out, so the compute provider only ever sees uniformly random values. ",
  "It returns the matrix product, the gateway checks it with a Freivalds test, subtracts the precomputed mask term, and continues locally. ",
  "The result is bit-exact compared with running the same quantized model in plaintext.",
].join("");
