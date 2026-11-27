// Mock mode (`npm run dev:mock`, `npm run build:mock`): a fake `pgw chat` server inside the page, with canned
// streamed answers, for UI work, screenshots and tests. `?state=` picks what the status endpoint reports:
// ready (default), setup, provisioning, stalled, blocked, empty (no saved chats), slow (slow streaming).

import type { ChatEvent, Conversation, Info, Message, Status } from "../api/types";

const MODELS = [
  { id: "qwen2.5-0.5b-instruct", label: "Qwen2.5 0.5B", params: "0.5B", trips_per_token: 97, download_gb: 1, disk_gb: 10, stream_disk_gb: 9, memory_gb: 8, cpus: 4, note: "Fastest to set up; short, simple answers" },
  { id: "qwen2.5-1.5b-instruct", label: "Qwen2.5 1.5B", params: "1.5B", trips_per_token: 113, download_gb: 3, disk_gb: 13, stream_disk_gb: 11, memory_gb: 8, cpus: 4, note: "Good default for most laptops" },
  { id: "qwen2.5-3b-instruct", label: "Qwen2.5 3B", params: "3B", trips_per_token: 145, download_gb: 6, disk_gb: 17, stream_disk_gb: 14, memory_gb: 16, cpus: 8, note: "Better answers; Qwen research licence" },
  { id: "qwen2.5-7b-instruct", label: "Qwen2.5 7B", params: "7B", trips_per_token: 113, download_gb: 11, disk_gb: 26, stream_disk_gb: 20, memory_gb: 32, cpus: 16, note: "Best answers; needs a strong machine" },
];

export const ANSWER =
  "Masking keeps your words on this computer. Before each layer runs on the GPU, the app adds a fresh random " +
  "mask to its input, so the GPU only ever sees noise-like numbers:\n\n" +
  "1. **Mask** the input `x` with a one-time value `r`.\n" +
  "2. The GPU computes `W(x + r)` on the masked row.\n" +
  "3. This computer removes `W r` and **checks** the result before using it.\n\n" +
  "```python\ny = unmask(gpu_matmul(x + r)) # verified here\n```\n\n" +
  "The GPU service still sees *metadata*: the model, token counts and timing.";

function now(): number {
  return Date.now() / 1000;
}

function seed(): Map<string, Conversation> {
  const t = now();
  const mk = (id: string, title: string, age: number, msgs: [string, string][]): Conversation => ({
    id,
    title,
    model: "qwen2.5-1.5b-instruct",
    created_at: t - age,
    updated_at: t - age,
    messages: msgs.flatMap(([q, a], i) => [
      { id: `${id}u${i}`, role: "user", content: q, created_at: t - age } as Message,
      {
        id: `${id}a${i}`,
        role: "assistant",
        content: a,
        created_at: t - age,
        status: "done",
        verified: true,
        tokens_per_s: 4.2,
        elapsed_s: 21.4,
        usage: { prompt_tokens: 40, completion_tokens: 88, cached_tokens: 0 },
      } as Message,
    ]),
  });
  const convs = [
    mk("c_000000000000000000000001", "How does masking keep my words private?", 600, [
      ["How does masking keep my words private?", ANSWER],
    ]),
    mk("c_000000000000000000000002", "Draft a polite note to my landlord", 7200, [
      ["Draft a polite note to my landlord about the heating.", "Here is a short draft:\n\n> Dear Ms Rossi, the heating in flat 3 has not worked since Monday..."],
    ]),
    mk("c_000000000000000000000003", "Explain Freivalds' algorithm simply", 3 * 86400, [
      ["Explain Freivalds' algorithm simply", "Pick a random vector `v` and compare `A(Bv)` with `Cv`: two cheap products instead of a full one."],
    ]),
    mk("c_000000000000000000000004", "Weekend trip ideas near the coast", 12 * 86400, [
      ["Weekend trip ideas near the coast", "- A ferry to the islands\n- A cliff walk\n- A seafood market morning"],
    ]),
  ];
  return new Map(convs.map((c) => [c.id, c]));
}

export function installMock(win: Window & typeof globalThis = window): void {
  const params = new URLSearchParams(win.location.search);
  const state = params.get("state") ?? "ready";
  const convs = state === "empty" || state === "setup" ? new Map<string, Conversation>() : seed();
  let verified = 3;
  const delay = state === "slow" ? 450 : 45;
  const stops = new Set<string>();

  const info: Info = {
    app: "PGW Chat",
    version: "0.1.0",
    deployment: "local",
    demo: false,
    default_model: "qwen2.5-1.5b-instruct",
    recommended_model: "qwen2.5-1.5b-instruct",
    models: MODELS,
    device: { cpus: 8, memory_gb: 16, free_disk_gb: 182 },
    settings: { mask_scheme: "pks-v3", masked_head: "auto", prefix_cache: "auto", speculation: "off", factory_weights: "stream", usage_reports: true, service_url: "https://gpu.example.com" },
    history: { persistent: true, location: "~/.local/state/pgw/chat/conversations" },
    limits: { max_message_chars: 32000, max_tokens: 1024 },
  };

  const status = (model: string): Status => {
    const base: Status = {
      model, phase: "ready", title: "Ready", detail: null, percent: null, eta_s: null, masks_available: 4096,
      requests: verified, verified_requests: verified, integrity_failures: 0, last_verified_at: now() - 30, lane: "open",
      masked_head: true, prefix_cache: false, mask_scheme: "pks-v3", error: null,
    };
    if (state === "setup")
      return { ...base, phase: "setup", title: "First-time setup on this machine", detail: "Building the one-time pks-v3 hint, 42% done (about 6 min left), then the first mask segment", percent: 42, eta_s: 372, masks_available: 0, verified_requests: 0, requests: 0 };
    if (state === "provisioning")
      return { ...base, phase: "provisioning", title: "Downloading the public model files", detail: "artifacts 412 / 1,630 MB", percent: 25, masks_available: null, verified_requests: 0 };
    if (state === "stalled")
      return { ...base, phase: "stalled", title: "Provider unreachable", detail: "the provider (https://gpu.example.com) has not answered for 134s (ConnectError)", error: { kind: "connection", title: "Provider unreachable", message: "", retryable: true } };
    if (state === "blocked")
      return { ...base, phase: "blocked", title: "Paused after a failed integrity check", detail: "A result from the provider failed its check and was discarded.", integrity_failures: 1 };
    return base;
  };

  const json = (data: unknown, code = 200) =>
    new Response(JSON.stringify(data), { status: code, headers: { "Content-Type": "application/json" } });

  const stream = (events: (push: (e: ChatEvent) => void, done: () => boolean) => Promise<void>) => {
    const enc = new TextEncoder();
    const body = new ReadableStream<Uint8Array>({
      async start(ctrl) {
        let closed = false;
        await events(
          (e) => {
            if (!closed) ctrl.enqueue(enc.encode(`data: ${JSON.stringify(e)}\n\n`));
          },
          () => closed,
        );
        closed = true;
        ctrl.close();
      },
    });
    return new Response(body, { status: 200, headers: { "Content-Type": "text/event-stream" } });
  };

  const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
  const real = win.fetch.bind(win);

  win.fetch = async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    const url = new URL(typeof input === "string" ? input : input instanceof URL ? input.href : input.url, win.location.href);
    if (!url.pathname.startsWith("/api/")) return real(input, init);
    const method = (init?.method ?? "GET").toUpperCase();
    const body = init?.body ? JSON.parse(String(init.body)) : {};
    const p = url.pathname;
    if (p === "/api/info") return json(info);
    if (p === "/api/status") return json(status(url.searchParams.get("model") ?? info.default_model));
    if (p === "/api/warmup") return json({ ok: true }, 202);
    if (p === "/api/conversations" && method === "GET")
      return json({
        conversations: [...convs.values()]
          .sort((a, b) => b.updated_at - a.updated_at)
          .map((c) => ({ id: c.id, title: c.title, model: c.model, created_at: c.created_at, updated_at: c.updated_at, messages: c.messages.length })),
        active: [],
      });
    if (p === "/api/conversations" && method === "DELETE") {
      const n = convs.size;
      convs.clear();
      return json({ deleted: n });
    }
    const cm = /^\/api\/conversations\/([\w]+)$/.exec(p);
    if (cm) {
      const c = convs.get(cm[1]);
      if (!c) return json({ error: { code: "not_found", message: "no such conversation" } }, 404);
      if (method === "DELETE") {
        convs.delete(cm[1]);
        return json({ deleted: true });
      }
      return json(c);
    }
    const sm = /^\/api\/chat\/(g_\w+)\/stop$/.exec(p);
    if (sm) {
      stops.add(sm[1]);
      return json({ stopped: true });
    }
    if (p === "/api/chat") {
      let conv = body.conversation_id ? convs.get(body.conversation_id) : undefined;
      if (!conv) {
        const id = "c_" + Math.random().toString(16).slice(2).padEnd(24, "0").slice(0, 24);
        conv = { id, title: "", model: body.model, created_at: now(), updated_at: now(), messages: [] };
        convs.set(id, conv);
      }
      const c = conv;
      let userMsg: Message | null = null;
      if (!body.regenerate) {
        userMsg = { id: "m_" + Math.random().toString(16).slice(2, 18), role: "user", content: body.message, created_at: now() };
        c.messages.push(userMsg);
        if (!c.title) c.title = String(body.message).slice(0, 80);
      } else {
        while (c.messages.length && c.messages[c.messages.length - 1].role !== "user") c.messages.pop();
      }
      const gid = "g_" + Math.random().toString(16).slice(2, 18).padEnd(16, "0");
      const text: string = String(body.message ?? "");
      return stream(async (push, closed) => {
        push({ type: "start", conversation_id: c.id, generation_id: gid, model: body.model, title: c.title, user_message: userMsg });
        const t0 = performance.now();
        if (text.startsWith("/error")) {
          await sleep(500);
          const kind = text.split(" ")[1] ?? "connection";
          const err = kind === "integrity"
            ? { kind, title: "Integrity check failed: answer discarded", message: "A result from the provider failed its integrity check, so it was discarded and nothing unverified was shown.", retryable: false }
            : { kind: "connection", title: "Provider unreachable", message: "connection failed (ConnectError)", retryable: true };
          const m: Message = { id: "m_err", role: "assistant", content: "", created_at: now(), status: "error", verified: false, error: err };
          c.messages.push(m);
          push({ type: "error", message: m, error: err });
          return;
        }
        await sleep(state === "slow" ? 7000 : 500);
        const words = ANSWER.split(" ");
        let out = "";
        let stopped = false;
        for (let i = 0; i < words.length; i++) {
          if (stops.has(gid) || closed()) {
            stopped = true;
            break;
          }
          const w = words[i] + (i < words.length - 1 ? " " : "");
          out += w;
          push({ type: "delta", text: w });
          await sleep(delay);
        }
        verified += 1;
        const el = (performance.now() - t0) / 1000;
        const m: Message = {
          id: "m_" + Math.random().toString(16).slice(2, 18), role: "assistant", content: out, created_at: now(),
          status: stopped ? "stopped" : "done", verified: true, finish_reason: stopped ? null : "stop",
          usage: { prompt_tokens: 30, completion_tokens: words.length, cached_tokens: 0 }, elapsed_s: el, ttft_s: 0.5,
          tokens_per_s: state === "slow" ? 2.2 : 18.5,
        };
        c.messages.push(m);
        c.updated_at = now();
        push({ type: "done", message: m, error: null });
      });
    }
    return json({ error: { code: "not_found", message: "no such endpoint" } }, 404);
  };
}
