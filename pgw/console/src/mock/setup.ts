// Demo/mock answers for the first-run setup API (/console/api/setup/*). Active only while the
// "setup" flag is set (the demo banner's "Try the setup wizard" link), so the normal console demo
// is unchanged. Setup token: "setup". Any well-formed PGW1- code is accepted.

import type { JobStatus, SetupCheck, SetupState } from "../api/setup";

export const SETUP_FLAG = "pgw-mock-setup";
export const SAMPLE_CODE =
  "PGW1-MFRG-GZDF-MZTW-Q2LK-NNWG-23TP-OBYX-E43U-OV3H-Q6L2-GAYT-EMZU-GU3T-QOJQ-MFRG-GZDF-MZTW-Q2LK-NNWG-23TP-OBYX-E43U";

const GB = 1024 ** 3;

export function setupActive(): boolean {
  try {
    return sessionStorage.getItem(SETUP_FLAG) === "1";
  } catch {
    return false;
  }
}

function setActive(on: boolean) {
  try {
    if (on) sessionStorage.setItem(SETUP_FLAG, "1");
    else sessionStorage.removeItem(SETUP_FLAG);
  } catch {
    /* ignore */
  }
}

let csrf: string | null = null;
let packStarted = 0;
let hintStarted = 0;
const PACK_MS = 6000;
const HINT_MS = 5000;

const st: SetupState = {
  setup_mode: true,
  finished: false,
  steps: ["enroll", "model", "maskstore", "admin", "verify", "finish"],
  estimate: { basis: "typical_7b", hint_bytes: 34 * GB, buffer_bytes: 6 * GB, total_bytes: 40 * GB, buffer_slots: 1024, disk_free_bytes: 412 * GB },
  jobs: { pack: null, hint: null, factory: null },
  masks: null,
  paths: { config_dir: "/etc/pgw", data_dir: "/var/lib/pgw" },
  kms_options: [
    { id: "file", label: "Local key file (0600)", available: true },
    { id: "aws", label: "AWS KMS", available: false },
    { id: "gcp", label: "Google Cloud KMS", available: false },
    { id: "azure", label: "Azure Key Vault", available: false },
  ],
};

function progress(started: number, ms: number, total: number, phases: JobStatus["phase"][], message: string): JobStatus {
  const f = Math.min(1, (Date.now() - started) / ms);
  const phase = f >= 1 ? "done" : phases[Math.min(phases.length - 1, Math.floor(f * phases.length))];
  return { phase, done: Math.round(f * total), total, message: f >= 1 ? "done" : message, error: null };
}

function snapshot(): SetupState {
  if (packStarted && st.model) {
    st.jobs.pack = progress(packStarted, PACK_MS, 28, ["downloading", "downloading", "packing", "verifying"], "Qwen2.5-7B-Instruct");
    st.model.pack_status = st.jobs.pack.phase === "done" ? "ready" : "building";
  }
  if (hintStarted) {
    st.jobs.hint = progress(hintStarted, HINT_MS, 112, ["building"], "encryption hint");
    if (st.jobs.hint.phase === "done") {
      st.jobs.factory = { running: true, restarts: 0, last_exit: null };
      st.masks = Math.min(4096, Math.round((Date.now() - hintStarted - HINT_MS) / 4));
      if (!st.verify?.ok) st.verify = { checks: checks(true), ok: true, at: Date.now() / 1000 };
    }
  }
  return structuredClone(st);
}

function checks(done: boolean): SetupCheck[] {
  const pending = done ? "ok" : "info";
  return [
    { id: "tls", label: "Encrypted connection to your provider", status: "ok", detail: "TLS 1.3, provider certificate matches the code" },
    { id: "cert", label: "Gateway identity certificate", status: "ok", detail: "client certificate valid for 30 days" },
    { id: "model", label: "Model matches the provider", status: "ok", detail: "fingerprint 3f9a…c21e on both sides" },
    { id: "store", label: "Mask store encrypted at rest", status: "ok", detail: "key file is private (0600), counter kept apart" },
    { id: "policy", label: "Production safety settings", status: "ok", detail: "no development options are on" },
    { id: "masks", label: "Masks being generated", status: pending, detail: done ? "mask factory running" : "waiting for the encryption hint" },
  ];
}

const json = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
const err = (status: number, message: string) => json({ error: { message, type: "invalid_request_error", code: status === 401 ? "unauthenticated" : "invalid_request", param: null } }, status);

export function handleSetup(method: string, p: string, headers: Headers, body: unknown): Response | null {
  if (!p.startsWith("/console/api/setup/")) return null;
  const active = setupActive();
  if (p === "/console/api/setup/status") return json({ setup_mode: active, signed_in: active && csrf !== null });
  if (!active) return err(404, "the gateway is not in setup mode");
  if (p === "/console/api/setup/login" && method === "POST") {
    if ((body as { token?: string })?.token !== "setup") return err(401, "invalid setup token");
    csrf = Math.random().toString(16).slice(2);
    return json({ csrf });
  }
  if (!csrf) return err(401, "not signed in");
  if (method !== "GET" && headers.get("X-PGW-CSRF") !== csrf) return err(403, "missing or invalid CSRF token");
  const b = (body ?? {}) as Record<string, unknown>;
  switch (p) {
    case "/console/api/setup/state":
      return json(snapshot());
    case "/console/api/setup/enroll":
      st.enroll = {
        tenant: "northwind-health",
        service_url: "https://compute.provider.example:8443",
        client_cert: true,
        ca_fingerprint: "sha256:7c41e0d29b5a0e3f",
        models: [
          { name: "qwen2.5-7b-instruct", fingerprint: "3f9a51d0c21e", ring: "K32A12C27", hf_repo: "Qwen/Qwen2.5-7B-Instruct", license: "Apache-2.0" },
          { name: "qwen2.5-0.5b-instruct", fingerprint: "a20c77e19b04", ring: "K32A12C27", hf_repo: "Qwen/Qwen2.5-0.5B-Instruct", license: "Apache-2.0" },
        ],
      };
      return json(st.enroll);
    case "/console/api/setup/model": {
      const m = st.enroll?.models.find((x) => x.name === b.name);
      if (!m) return err(400, "unknown model");
      st.model = { name: m.name, fingerprint: m.fingerprint, ring: m.ring, hf_repo: m.hf_repo, pack_status: "building" };
      st.estimate = { ...st.estimate, basis: "model" };
      packStarted = Date.now();
      return json(snapshot().model);
    }
    case "/console/api/setup/maskstore":
      st.maskstore = { root: (b.root as string) || "/var/lib/pgw/masks", key_file: "/etc/pgw/maskstore.key", counter_file: "/var/lib/pgw/state/maskstore.counter" };
      return json(st.maskstore);
    case "/console/api/setup/admin":
      st.admin = { token_file: "/etc/pgw/admin-token", sso: (b.sso as Record<string, unknown>) ?? null };
      return json({ admin_token: `pgwadm-${crypto.randomUUID().replace(/-/g, "")}`, token_file: st.admin.token_file, sso: Boolean(b.sso) });
    case "/console/api/setup/verify":
      if (!hintStarted) hintStarted = Date.now();
      st.verify = { checks: checks(false), ok: false, at: Date.now() / 1000 };
      return json(snapshot().verify);
    case "/console/api/setup/finish":
      st.finished = true;
      setActive(false);
      return json({ mode: "normal", message: "Setup is complete. The gateway is now serving qwen2.5-7b-instruct." });
  }
  return err(404, `mock: no handler for ${method} ${p}`);
}

export function startSetupDemo() {
  setActive(true);
  window.location.reload();
}
