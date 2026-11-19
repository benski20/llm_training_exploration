import { describe, expect, it, vi } from "vitest";
import { ApiError, createApi, type ApiEvent } from "./client";
import { createSSEParser } from "./sse";

type Call = { url: string; init: RequestInit };

function fakeFetch(handler: (url: string, init: RequestInit) => Response) {
  const calls: Call[] = [];
  const fn = vi.fn(async (url: string, init: RequestInit = {}) => {
    calls.push({ url, init });
    return handler(url, init);
  });
  return { fn, calls };
}

const json = (body: unknown, status = 200, headers: Record<string, string> = {}) =>
  new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json", ...headers } });

const header = (c: Call, name: string) => new Headers(c.init.headers).get(name);

describe("api client", () => {
  it("stores the csrf token from login and sends it only on state-changing requests", async () => {
    const { fn, calls } = fakeFetch((url) => (url === "/console/api/login" ? json({ role: "admin", csrf: "tok123" }) : json({ keys: [] })));
    const api = createApi(fn);
    const r = await api.login("secret");
    expect(r.role).toBe("admin");
    expect(api.csrf).toBe("tok123");
    expect(JSON.parse(String(calls[0].init.body))).toEqual({ token: "secret" });

    await api.keys();
    expect(header(calls[1], "X-PGW-CSRF")).toBeNull();
    await api.revokeKey("abc123def456");
    expect(calls[2].init.method).toBe("DELETE");
    expect(calls[2].url).toBe("/admin/keys/abc123def456");
    expect(header(calls[2], "X-PGW-CSRF")).toBe("tok123");
  });

  it("reads the sign-in methods and the SSO user from /me", async () => {
    const { fn, calls } = fakeFetch((url) =>
      url === "/console/api/auth-methods"
        ? json({ token: false, oidc: { enabled: true, label: "Entra ID" } })
        : json({ role: "viewer", csrf: "c1", auth: "oidc", user: { sub: "u1", email: "a@corp.example.com", name: "A" } }),
    );
    const api = createApi(fn);
    const m = await api.authMethods();
    expect(m.oidc).toEqual({ enabled: true, label: "Entra ID" });
    expect(m.token).toBe(false);
    const me = await api.me();
    expect(me.auth).toBe("oidc");
    expect(me.user?.email).toBe("a@corp.example.com");
    expect(api.csrf).toBe("c1");
    expect(calls.map((c) => c.init.method)).toEqual(["GET", "GET"]);
  });

  it("builds audit query strings without empty filters", async () => {
    const { fn, calls } = fakeFetch(() => json({ events: [], next_before: null }));
    const api = createApi(fn);
    await api.audit({ limit: 25, before: "2026-11-19T10:00:00.000Z", key_id: "", model: "qwen7b", status: "5xx" });
    expect(calls[0].url).toBe("/admin/audit?limit=25&before=2026-11-19T10%3A00%3A00.000Z&model=qwen7b&status=5xx");
  });

  it("parses the gateway error shape and Retry-After into ApiError and emits an error event", async () => {
    const { fn } = fakeFetch(() =>
      json({ error: { message: "mask store has 120 slots", type: "server_error", code: "mask_exhausted", param: null } }, 503, { "Retry-After": "7", "x-request-id": "rid-1" }),
    );
    const api = createApi(fn);
    const events: ApiEvent[] = [];
    api.subscribe((e) => events.push(e));
    const err = await api.overview().catch((e: unknown) => e);
    expect(err).toBeInstanceOf(ApiError);
    const e = err as ApiError;
    expect(e.status).toBe(503);
    expect(e.code).toBe("mask_exhausted");
    expect(e.retryAfter).toBe(7);
    expect(e.requestId).toBe("rid-1");
    expect(e.retryable).toBe(true);
    expect(events).toHaveLength(1);
    expect(events[0].kind).toBe("error");
  });

  it("emits unauthorized on 401 and clears the csrf token, but not for /me", async () => {
    let status = 200;
    const { fn } = fakeFetch((url) => {
      if (url === "/console/api/me") return status === 200 ? json({ role: "viewer", csrf: "c1" }) : json({ error: { message: "no session" } }, 401);
      return json({ error: { message: "expired" } }, 401);
    });
    const api = createApi(fn);
    const events: ApiEvent[] = [];
    api.subscribe((e) => events.push(e));
    await api.me();
    expect(api.csrf).toBe("c1");
    await expect(api.status()).rejects.toBeInstanceOf(ApiError);
    expect(events).toEqual([{ kind: "unauthorized" }]);
    expect(api.csrf).toBeNull();
    status = 401;
    await expect(api.me()).rejects.toBeInstanceOf(ApiError);
    expect(events).toHaveLength(1);
  });

  it("maps network failures to status 0", async () => {
    const api = createApi(async () => {
      throw new TypeError("Failed to fetch");
    });
    const e = (await api.masks().catch((x: unknown) => x)) as ApiError;
    expect(e.status).toBe(0);
    expect(e.retryable).toBe(true);
  });

  it("streams chat chunks and collects content, usage and fingerprint", async () => {
    const sse = [
      `data: {"id":"c1","object":"chat.completion.chunk","system_fingerprint":"3fa9c2d1e07b","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}`,
      `data: {"id":"c1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"Hel"},"finish_reason":null}]}`,
      `data: {"id":"c1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"lo"},"finish_reason":"stop"}]}`,
      `data: {"id":"c1","object":"chat.completion.chunk","choices":[],"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}`,
      `data: [DONE]`,
      ``,
    ].join("\n\n");
    const { fn, calls } = fakeFetch(() => new Response(sse, { status: 200, headers: { "Content-Type": "text/event-stream", "x-request-id": "rid-9" } }));
    const api = createApi(fn);
    api.setCsrf("c");
    const deltas: string[] = [];
    const r = await api.chatStream({ model: "m", messages: [{ role: "user", content: "hi" }] }, { onDelta: (d) => deltas.push(d) });
    expect(deltas).toEqual(["Hel", "lo"]);
    expect(r.content).toBe("Hello");
    expect(r.finishReason).toBe("stop");
    expect(r.usage?.completion_tokens).toBe(2);
    expect(r.fingerprint).toBe("3fa9c2d1e07b");
    expect(r.requestId).toBe("rid-9");
    const body = JSON.parse(String(calls[0].init.body));
    expect(body.stream).toBe(true);
    expect(body.stream_options).toEqual({ include_usage: true });
    expect(header(calls[0], "X-PGW-CSRF")).toBe("c");
  });

  it("rejects when the stream carries an error event", async () => {
    const sse = `data: {"error":{"message":"integrity check failed","type":"server_error","code":"integrity_check_failed"}}\n\ndata: [DONE]\n\n`;
    const api = createApi(async () => new Response(sse, { status: 200 }));
    const e = (await api.chatStream({ model: "m", messages: [] }).catch((x: unknown) => x)) as ApiError;
    expect(e).toBeInstanceOf(ApiError);
    expect(e.code).toBe("integrity_check_failed");
  });
});

describe("SSE parser", () => {
  it("handles events split across chunks, CRLF, comments and multi-line data", () => {
    const out: string[] = [];
    const p = createSSEParser((d) => out.push(d));
    p.push(": keep-alive\r\n\r\ndata: {\"a\"");
    p.push(":1}\r\n\r\ndata: line1\ndata: line2\n\n");
    p.push("data:no-space");
    p.end();
    expect(out).toEqual(['{"a":1}', "line1\nline2", "no-space"]);
  });
});
