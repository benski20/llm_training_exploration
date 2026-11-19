// Typed client for the gateway's console + admin API (pgw/docs/api/console-api.md).
// Same-origin only; the session lives in the HttpOnly `pgw_console` cookie, the CSRF token in memory.

import { readSSE } from "./sse";
import type {
  ApiErrorBody,
  ApiKeyRecord,
  AuditPage,
  AuthMethods,
  AuditQuery,
  ChatChunk,
  ChatCompletion,
  ChatRequest,
  ChatUsage,
  CreateKeyRequest,
  CreateKeyResponse,
  LoginResponse,
  Masks,
  MeResponse,
  Overview,
  Status,
  Usage,
  UsageWindow,
} from "./types";

export class ApiError extends Error {
  readonly status: number;
  readonly code: string | null;
  readonly type: string | null;
  readonly param: string | null;
  /** Seconds from `Retry-After`, when the server sent one. */
  readonly retryAfter: number | null;
  readonly requestId: string | null;

  constructor(
    status: number,
    message: string,
    opts: { code?: string | null; type?: string | null; param?: string | null; retryAfter?: number | null; requestId?: string | null } = {},
  ) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = opts.code ?? null;
    this.type = opts.type ?? null;
    this.param = opts.param ?? null;
    this.retryAfter = opts.retryAfter ?? null;
    this.requestId = opts.requestId ?? null;
  }

  /** Worth retrying later (rate limit, temporarily unavailable, deadline). */
  get retryable(): boolean {
    return this.status === 429 || this.status === 503 || this.status === 504 || this.status === 0;
  }
}

export type ApiEvent = { kind: "unauthorized" } | { kind: "error"; error: ApiError; method: string; path: string };

type FetchLike = (input: string, init?: RequestInit) => Promise<Response>;

export interface StreamResult {
  content: string;
  finishReason: string | null;
  usage: ChatUsage | null;
  fingerprint: string | null;
  id: string | null;
  requestId: string | null;
}

export interface StreamCallbacks {
  onDelta?: (text: string, chunk: ChatChunk) => void;
  signal?: AbortSignal;
}

const MUTATING = new Set(["POST", "PUT", "PATCH", "DELETE"]);

function parseRetryAfter(v: string | null): number | null {
  if (!v) return null;
  const n = Number(v);
  if (Number.isFinite(n)) return Math.max(0, n);
  const d = Date.parse(v);
  return Number.isNaN(d) ? null : Math.max(0, Math.round((d - Date.now()) / 1000));
}

export async function errorFromResponse(res: Response): Promise<ApiError> {
  let body: Partial<ApiErrorBody> | null = null;
  try {
    body = (await res.json()) as Partial<ApiErrorBody>;
  } catch {
    body = null;
  }
  const e = body?.error;
  const message = (e && typeof e.message === "string" && e.message) || res.statusText || `HTTP ${res.status}`;
  return new ApiError(res.status, message, {
    code: e?.code ?? null,
    type: e?.type ?? null,
    param: e?.param ?? null,
    retryAfter: parseRetryAfter(res.headers.get("retry-after")),
    requestId: res.headers.get("x-request-id"),
  });
}

function qs(params: Record<string, string | number | null | undefined>): string {
  const u = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) {
    if (v !== undefined && v !== null && v !== "") u.set(k, String(v));
  }
  const s = u.toString();
  return s ? `?${s}` : "";
}

export function createApi(fetchImpl: FetchLike = (i, init) => fetch(i, init)) {
  let csrf: string | null = null;
  const listeners = new Set<(e: ApiEvent) => void>();
  const emit = (e: ApiEvent) => listeners.forEach((l) => l(e));

  async function raw(method: string, path: string, body?: unknown, opts: { quiet401?: boolean; signal?: AbortSignal } = {}): Promise<Response> {
    const headers: Record<string, string> = { Accept: "application/json" };
    if (body !== undefined) headers["Content-Type"] = "application/json";
    if (MUTATING.has(method) && csrf) headers["X-PGW-CSRF"] = csrf;
    let res: Response;
    try {
      res = await fetchImpl(path, {
        method,
        headers,
        body: body === undefined ? undefined : JSON.stringify(body),
        credentials: "same-origin",
        cache: "no-store",
        signal: opts.signal,
      });
    } catch (err) {
      if ((err as Error)?.name === "AbortError") throw err;
      const e = new ApiError(0, "Network error: the gateway could not be reached.", { code: "network" });
      emit({ kind: "error", error: e, method, path });
      throw e;
    }
    if (!res.ok) {
      const e = await errorFromResponse(res);
      if (res.status === 401 && !opts.quiet401) {
        csrf = null;
        emit({ kind: "unauthorized" });
      } else if (res.status !== 401) {
        emit({ kind: "error", error: e, method, path });
      }
      throw e;
    }
    return res;
  }

  async function json<T>(method: string, path: string, body?: unknown, opts?: { quiet401?: boolean }): Promise<T> {
    const res = await raw(method, path, body, opts);
    if (res.status === 204) return undefined as T;
    const text = await res.text();
    return (text ? JSON.parse(text) : undefined) as T;
  }

  return {
    /** Subscribe to global API events (401 -> sign-in, errors -> toasts). */
    subscribe(fn: (e: ApiEvent) => void): () => void {
      listeners.add(fn);
      return () => listeners.delete(fn);
    },
    get csrf(): string | null {
      return csrf;
    },
    setCsrf(v: string | null) {
      csrf = v;
    },

    async login(token: string): Promise<LoginResponse> {
      const r = await json<LoginResponse>("POST", "/console/api/login", { token }, { quiet401: true });
      csrf = r.csrf;
      return r;
    },
    async logout(): Promise<void> {
      try {
        await raw("POST", "/console/api/logout", undefined, { quiet401: true });
      } finally {
        csrf = null;
      }
    },
    /** Sign-in methods offered by the gateway (no session needed). */
    authMethods: () => json<AuthMethods>("GET", "/console/api/auth-methods", undefined, { quiet401: true }),
    async me(): Promise<MeResponse> {
      const r = await json<MeResponse>("GET", "/console/api/me", undefined, { quiet401: true });
      csrf = r.csrf;
      return r;
    },

    overview: () => json<Overview>("GET", "/admin/overview"),
    usage: (window: UsageWindow = "24h") => json<Usage>("GET", `/admin/usage${qs({ window })}`),
    status: () => json<Status>("GET", "/admin/status"),
    masks: () => json<Masks>("GET", "/admin/masks"),
    keys: () => json<{ keys: ApiKeyRecord[] }>("GET", "/admin/keys"),
    createKey: (req: CreateKeyRequest) => json<CreateKeyResponse>("POST", "/admin/keys", req),
    revokeKey: (keyId: string) => json<{ revoked: string }>("DELETE", `/admin/keys/${encodeURIComponent(keyId)}`),
    reloadModel: (name: string) =>
      json<{ reloaded: string; fingerprint: string | null; ready: boolean }>("POST", `/admin/models/${encodeURIComponent(name)}/reload`),
    resetModel: (name: string, flags: { breaker?: boolean; noise?: boolean } = {}) =>
      json<Record<string, unknown>>("POST", `/admin/models/${encodeURIComponent(name)}/reset`, flags),
    audit: (q: AuditQuery = {}) =>
      json<AuditPage>(
        "GET",
        `/admin/audit${qs({ limit: q.limit ?? 50, before: q.before ?? undefined, key_id: q.key_id, model: q.model, status: q.status })}`,
      ),

    /** Non-streaming playground call. */
    chat: (req: ChatRequest) => json<ChatCompletion>("POST", "/console/api/playground/chat", { ...req, stream: false }),

    /** Streaming playground call (SSE). Resolves when the stream ends; rejects on an SSE error event. */
    async chatStream(req: ChatRequest, cb: StreamCallbacks = {}): Promise<StreamResult> {
      const res = await raw(
        "POST",
        "/console/api/playground/chat",
        { ...req, stream: true, stream_options: { include_usage: true } },
        { signal: cb.signal },
      );
      const out: StreamResult = {
        content: "",
        finishReason: null,
        usage: null,
        fingerprint: null,
        id: null,
        requestId: res.headers.get("x-request-id"),
      };
      if (!res.body) throw new ApiError(0, "The response has no body to stream.");
      let streamError: ApiError | null = null;
      await readSSE(res.body, (data) => {
        if (data === "[DONE]" || streamError) return;
        let obj: ChatChunk | Partial<ApiErrorBody>;
        try {
          obj = JSON.parse(data);
        } catch {
          return;
        }
        if ("error" in obj && obj.error) {
          const e = obj.error;
          streamError = new ApiError(200, e.message || "stream error", { code: e.code, type: e.type, requestId: out.requestId });
          return;
        }
        const chunk = obj as ChatChunk;
        out.id = chunk.id ?? out.id;
        if (chunk.system_fingerprint) out.fingerprint = chunk.system_fingerprint;
        if (chunk.usage) out.usage = chunk.usage;
        for (const c of chunk.choices ?? []) {
          const d = c.delta?.content;
          if (d) {
            out.content += d;
            cb.onDelta?.(d, chunk);
          }
          if (c.finish_reason) out.finishReason = c.finish_reason;
        }
      });
      if (streamError) throw streamError;
      return out;
    },
  };
}

export type Api = ReturnType<typeof createApi>;

/** Starts single sign-on: a full-page navigation (the gateway redirects to the identity provider). */
export const SSO_LOGIN_PATH = "/console/api/oidc/login";

/** The app-wide client instance. */
export const api = createApi();
