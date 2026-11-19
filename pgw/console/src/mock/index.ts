// Fetch shim for `npm run dev:mock` / `npm run build:mock`: answers the console + admin API from
// fixtures so every page renders without a backend. Tokens: "admin" -> admin, "viewer" -> viewer.

import type { ApiKeyRecord, ChatRequest, CreateKeyRequest } from "../api/types";
import * as fx from "./fixtures";

type Role = "admin" | "viewer";
const SESSION_KEY = "pgw-mock-session";

function loadSession(): { role: Role; csrf: string } | null {
  try {
    const v = sessionStorage.getItem(SESSION_KEY);
    return v ? (JSON.parse(v) as { role: Role; csrf: string }) : null;
  } catch {
    return null;
  }
}

let session = loadSession();
const keys: ApiKeyRecord[] = fx.keys.map((k) => ({ ...k }));
const events = fx.auditEvents();

const json = (body: unknown, status = 200, headers: Record<string, string> = {}) =>
  new Response(status === 204 ? null : JSON.stringify(body), { status, headers: { "Content-Type": "application/json", "x-request-id": `req-${Math.random().toString(16).slice(2, 10)}`, ...headers } });

const err = (status: number, code: string, message: string, headers: Record<string, string> = {}) =>
  json({ error: { message, type: status >= 500 ? "server_error" : "invalid_request_error", code, param: null } }, status, headers);

const delay = (ms: number) => new Promise((r) => setTimeout(r, ms));

function randHex(n: number) {
  const a = new Uint8Array(Math.ceil(n / 2));
  crypto.getRandomValues(a);
  return Array.from(a, (b) => b.toString(16).padStart(2, "0")).join("").slice(0, n);
}

function matchStatus(s: number | undefined, f: string): boolean {
  if (!f) return true;
  if (s === undefined) return false;
  if (/^\dxx$/.test(f)) return Math.floor(s / 100) === Number(f[0]);
  return String(s) === f;
}

function streamChat(req: ChatRequest, signal?: AbortSignal | null): Response {
  const id = `chatcmpl-${randHex(24)}`;
  const words = fx.CANNED.split(/(?<= )/).slice(0, req.max_tokens ?? 512);
  const enc = new TextEncoder();
  const base = { id, object: "chat.completion.chunk", created: Math.floor(Date.now() / 1000), model: req.model, system_fingerprint: fx.models.find((m) => m.name === req.model)?.fingerprint_short ?? null };
  const body = new ReadableStream<Uint8Array>({
    async start(ctl) {
      const send = (o: unknown) => ctl.enqueue(enc.encode(`data: ${JSON.stringify(o)}\n\n`));
      await delay(260);
      send({ ...base, choices: [{ index: 0, delta: { role: "assistant", content: "" }, finish_reason: null }] });
      for (let i = 0; i < words.length; i++) {
        await delay(28 + Math.random() * 18);
        if (signal?.aborted) return ctl.error(new DOMException("Aborted", "AbortError"));
        send({ ...base, choices: [{ index: 0, delta: { content: words[i] }, finish_reason: i === words.length - 1 ? "stop" : null }] });
      }
      send({ ...base, choices: [], usage: { prompt_tokens: 38, completion_tokens: words.length, total_tokens: 38 + words.length } });
      ctl.enqueue(enc.encode("data: [DONE]\n\n"));
      ctl.close();
    },
  });
  return new Response(body, { status: 200, headers: { "Content-Type": "text/event-stream", "x-request-id": id } });
}

async function handle(method: string, url: URL, headers: Headers, body: unknown, signal?: AbortSignal | null): Promise<Response> {
  const p = url.pathname;
  await delay(80 + Math.random() * 120);

  if (p === "/console/api/login" && method === "POST") {
    const token = (body as { token?: string })?.token;
    const role: Role | null = token === "admin" ? "admin" : token === "viewer" ? "viewer" : null;
    if (!role) return err(401, "unauthenticated", "invalid token");
    session = { role, csrf: randHex(32) };
    sessionStorage.setItem(SESSION_KEY, JSON.stringify(session));
    return json(session);
  }
  if (p === "/console/api/auth-methods") return json({ token: true, oidc: { enabled: false, label: "SSO" } });
  if (p === "/console/api/logout" && method === "POST") {
    session = null;
    sessionStorage.removeItem(SESSION_KEY);
    return json(null, 204);
  }
  if (!session) return err(401, "unauthenticated", "not signed in");
  if (p === "/console/api/me") return json({ ...session, expires_at: new Date(Date.now() + 8 * 3600e3).toISOString(), version: "0.1.0", auth: "token", user: null });

  if (method !== "GET") {
    if (headers.get("X-PGW-CSRF") !== session.csrf) return err(403, "permission_denied", "missing or invalid CSRF token");
    if (session.role !== "admin") return err(403, "permission_denied", "admin role required");
  }

  if (p === "/admin/overview") return json(fx.overview());
  if (p === "/admin/usage") return json(fx.usage((url.searchParams.get("window") as "1h" | "24h" | "7d") || "24h"));
  if (p === "/admin/status") return json(fx.status);
  if (p === "/admin/masks") return json(fx.masks());
  if (p === "/admin/keys" && method === "GET") return json({ keys });
  if (p === "/admin/keys" && method === "POST") {
    const b = body as CreateKeyRequest;
    const id = randHex(12);
    const rec: ApiKeyRecord = {
      key_id: id,
      name: b.name,
      scopes: b.scopes ?? ["inference"],
      models: b.models ?? [],
      requests_per_minute: b.requests_per_minute ?? null,
      tokens_per_minute: b.tokens_per_minute ?? null,
      daily_token_quota: b.daily_token_quota ?? null,
      max_concurrent_requests: b.max_concurrent_requests ?? null,
      created_at: new Date().toISOString(),
      disabled: false,
      meta: {},
      tokens_today: 0,
    };
    keys.unshift(rec);
    return json({ key: `pgw-${id}-${randHex(43)}`, record: rec }, 201);
  }
  const revoke = p.match(/^\/admin\/keys\/([^/]+)$/);
  if (revoke && method === "DELETE") {
    const i = keys.findIndex((k) => k.key_id === revoke[1]);
    if (i < 0) return err(404, "model_not_found", "unknown key");
    keys.splice(i, 1);
    return json({ revoked: revoke[1] });
  }
  const action = p.match(/^\/admin\/models\/([^/]+)\/(reload|reset)$/);
  if (action && method === "POST") {
    const name = decodeURIComponent(action[1]);
    const m = fx.models.find((x) => x.name === name);
    if (!m) return err(404, "model_not_found", "unknown model");
    await delay(700);
    return json(action[2] === "reload" ? { reloaded: name, fingerprint: m.fingerprint, ready: m.ready } : { model: name, breaker: false, noise: false });
  }
  if (p === "/admin/audit") {
    const q = url.searchParams;
    const limit = Number(q.get("limit") || 100);
    const before = q.get("before");
    const list = events.filter(
      (e) =>
        (!before || e.ts < before) &&
        (!q.get("key_id") || e.key_id === q.get("key_id")) &&
        (!q.get("model") || e.model === q.get("model") || e.target === q.get("model")) &&
        matchStatus(e.status, q.get("status") || ""),
    );
    const page = list.slice(0, limit);
    return json({ events: page, next_before: list.length > limit ? page[page.length - 1].ts : null });
  }
  if (p === "/console/api/playground/chat" && method === "POST") {
    const req = body as ChatRequest;
    const m = fx.models.find((x) => x.name === req.model);
    if (!m) return err(404, "model_not_found", `model '${req.model}' not found`);
    if (!m.ready) return err(503, "unavailable", `model '${req.model}' is not ready: ${m.reason}`, { "Retry-After": "5" });
    if (req.stream) return streamChat(req, signal);
    await delay(900);
    const content = fx.CANNED;
    return json({
      id: `chatcmpl-${randHex(24)}`,
      object: "chat.completion",
      created: Math.floor(Date.now() / 1000),
      model: req.model,
      system_fingerprint: m.fingerprint_short,
      choices: [{ index: 0, message: { role: "assistant", content }, logprobs: null, finish_reason: "stop" }],
      usage: { prompt_tokens: 38, completion_tokens: 92, total_tokens: 130 },
    });
  }
  return err(404, "not_found", `mock: no handler for ${method} ${p}`);
}

export function installMock(): void {
  const real = window.fetch.bind(window);
  window.fetch = async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = new URL(typeof input === "string" ? input : input instanceof URL ? input.href : input.url, window.location.origin);
    if (url.origin !== window.location.origin || !(url.pathname.startsWith("/admin/") || url.pathname.startsWith("/console/api/"))) return real(input, init);
    const method = (init?.method || "GET").toUpperCase();
    let body: unknown = undefined;
    if (typeof init?.body === "string") {
      try {
        body = JSON.parse(init.body);
      } catch {
        body = undefined;
      }
    }
    if (init?.signal?.aborted) throw new DOMException("Aborted", "AbortError");
    return handle(method, url, new Headers(init?.headers), body, init?.signal);
  };
  console.info("[pgw-console] mock API active: sign in with token 'admin' or 'viewer'");
}
