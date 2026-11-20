// Client for the gateway's first-run setup API (/console/api/setup/*, pgw.gateway.setup).
// Same-origin; the setup session lives in the HttpOnly cookie, the CSRF token in memory.

import { ApiError, errorFromResponse } from "./client";

export type CheckStatus = "ok" | "warn" | "fail" | "info";

export interface SetupCheck {
  id: string;
  label: string;
  status: CheckStatus;
  detail: string;
}

export interface EnrolledModel {
  name: string;
  fingerprint: string;
  ring: string | null;
  hf_repo: string | null;
  revision?: string | null;
  license?: string | null;
}

export interface JobStatus {
  phase: "queued" | "downloading" | "packing" | "verifying" | "building" | "done" | "failed";
  done: number;
  total: number;
  message: string;
  error: string | null;
}

export interface FactoryStatus {
  running: boolean;
  restarts: number;
  last_exit: number | null;
}

export interface SetupState {
  setup_mode: true;
  finished: boolean;
  steps: string[];
  enroll?: { tenant: string; service_url: string; models: EnrolledModel[]; client_cert: boolean; ca_fingerprint: string | null };
  model?: { name: string; fingerprint: string; ring: string | null; hf_repo: string | null; pack_status: "building" | "ready" | "failed" | "interrupted"; error?: string | null };
  maskstore?: { root: string; key_file: string; counter_file: string };
  admin?: { token_file: string; sso: Record<string, unknown> | null };
  verify?: { checks: SetupCheck[]; ok: boolean; at: number };
  estimate: { basis: "model" | "typical_7b"; hint_bytes: number; buffer_bytes: number | null; total_bytes: number | null; buffer_slots: number; disk_free_bytes: number | null };
  jobs: { pack: JobStatus | null; hint: JobStatus | null; factory: FactoryStatus | null };
  masks: number | null;
  paths: { config_dir: string; data_dir: string };
  kms_options: { id: string; label: string; available: boolean }[];
}

export interface SSOInput {
  issuer: string;
  client_id: string;
  client_secret: string;
  redirect_url: string;
  admin_values: string[];
  viewer_values: string[];
  label: string;
}

export interface FinishResult {
  mode: "normal" | "restart_required";
  message: string;
  notes?: string[];
}

type FetchLike = (input: string, init?: RequestInit) => Promise<Response>;

/** Light client-side check of an enrollment code (the gateway verifies the checksum). */
export function codeProblem(code: string): string | null {
  const s = code.replace(/[\s_-]+/g, "").toUpperCase();
  if (!s) return "Paste the enrollment code you received from your provider.";
  if (!s.startsWith("PGW")) return "An enrollment code starts with PGW1-.";
  if (!s.startsWith("PGW1")) return "This code is for a different version of the gateway. Ask your provider for a new code.";
  if (!/^[A-Z2-7018]+$/.test(s.slice(4))) return "The code contains characters that cannot appear in a code. Copy it again.";
  if (s.length < 90) return "The code looks incomplete. Copy the whole code, from PGW1- to the end.";
  return null;
}

export function createSetupApi(fetchImpl: FetchLike = (i, init) => fetch(i, init)) {
  let csrf: string | null = null;

  async function call<T>(method: string, path: string, body?: unknown): Promise<T> {
    const headers: Record<string, string> = { Accept: "application/json" };
    if (body !== undefined) headers["Content-Type"] = "application/json";
    if (method !== "GET" && csrf) headers["X-PGW-CSRF"] = csrf;
    let res: Response;
    try {
      res = await fetchImpl(path, { method, headers, body: body === undefined ? undefined : JSON.stringify(body), credentials: "same-origin", cache: "no-store" });
    } catch {
      throw new ApiError(0, "The gateway could not be reached. Is it still running?", { code: "network" });
    }
    if (!res.ok) throw await errorFromResponse(res);
    const text = await res.text();
    return (text ? JSON.parse(text) : undefined) as T;
  }

  return {
    /** `setup_mode: true` while the gateway runs the wizard (normal gateways answer false). */
    status: () => call<{ setup_mode: boolean; signed_in: boolean }>("GET", "/console/api/setup/status"),
    async login(token: string): Promise<void> {
      const r = await call<{ csrf: string }>("POST", "/console/api/setup/login", { token });
      csrf = r.csrf;
    },
    async resume(): Promise<boolean> {
      try {
        const r = await call<{ csrf: string }>("GET", "/console/api/me");
        csrf = r.csrf;
        return true;
      } catch {
        return false;
      }
    },
    state: () => call<SetupState>("GET", "/console/api/setup/state"),
    enroll: (code: string) => call<NonNullable<SetupState["enroll"]>>("POST", "/console/api/setup/enroll", { code }),
    chooseModel: (name: string) => call<NonNullable<SetupState["model"]>>("POST", "/console/api/setup/model", { name }),
    maskstore: (root: string | null) => call<NonNullable<SetupState["maskstore"]>>("POST", "/console/api/setup/maskstore", root ? { root } : {}),
    admin: (sso: SSOInput | null) => call<{ admin_token: string; token_file: string; sso: boolean }>("POST", "/console/api/setup/admin", sso ? { sso } : {}),
    verify: () => call<NonNullable<SetupState["verify"]>>("POST", "/console/api/setup/verify", {}),
    finish: () => call<FinishResult>("POST", "/console/api/setup/finish", {}),
  };
}

export type SetupApi = ReturnType<typeof createSetupApi>;

export const setupApi = createSetupApi();
