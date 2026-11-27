// Calls to the `pgw chat` server on this machine. Every call carries the per-launch token; nothing is sent to any
// other origin (the CSP's connect-src 'self' enforces it).

import { readSSE } from "./sse";
import type { ChatEvent, Conversation, ConversationSummary, Info, Status } from "./types";

export class ApiError extends Error {
  constructor(
    public status: number,
    public code: string,
    message: string,
  ) {
    super(message);
  }
}

let token: string | null = null;

export function setToken(t: string | null): void {
  token = t;
}

function headers(json: boolean): Record<string, string> {
  const h: Record<string, string> = { Authorization: `Bearer ${token ?? ""}` };
  if (json) h["Content-Type"] = "application/json";
  return h;
}

async function fail(r: Response): Promise<never> {
  let code = "http_" + r.status;
  let message = r.statusText || "request failed";
  try {
    const b = (await r.json()) as { error?: { code?: string; message?: string } };
    if (b.error?.code) code = b.error.code;
    if (b.error?.message) message = b.error.message;
  } catch {
    // not JSON
  }
  throw new ApiError(r.status, code, message);
}

async function get<T>(path: string): Promise<T> {
  const r = await fetch(path, { headers: headers(false), cache: "no-store", credentials: "omit" });
  if (!r.ok) return fail(r);
  return (await r.json()) as T;
}

async function send<T>(method: string, path: string, body?: unknown): Promise<T> {
  const r = await fetch(path, {
    method,
    headers: headers(true),
    body: JSON.stringify(body ?? {}),
    cache: "no-store",
    credentials: "omit",
  });
  if (!r.ok) return fail(r);
  return (await r.json()) as T;
}

export const api = {
  info: () => get<Info>("/api/info"),
  status: (model: string) => get<Status>(`/api/status?model=${encodeURIComponent(model)}`),
  warmup: (model: string) => send<{ ok: boolean }>("POST", "/api/warmup", { model }),
  conversations: () => get<{ conversations: ConversationSummary[]; active: string[] }>("/api/conversations"),
  conversation: (id: string) => get<Conversation>(`/api/conversations/${encodeURIComponent(id)}`),
  deleteConversation: (id: string) => send<{ deleted: boolean }>("DELETE", `/api/conversations/${encodeURIComponent(id)}`),
  deleteAll: () => send<{ deleted: number }>("DELETE", "/api/conversations"),
  stop: (generation: string) => send<{ stopped: boolean }>("POST", `/api/chat/${encodeURIComponent(generation)}/stop`),
};

export interface ChatRequest {
  conversation_id: string | null;
  model: string;
  message?: string;
  regenerate?: boolean;
  temperature?: number;
}

/** POST /api/chat and hand every event to `onEvent` until the stream ends (or `signal` aborts it). */
export async function chat(req: ChatRequest, onEvent: (e: ChatEvent) => void, signal?: AbortSignal): Promise<void> {
  const r = await fetch("/api/chat", {
    method: "POST",
    headers: headers(true),
    body: JSON.stringify(req),
    cache: "no-store",
    credentials: "omit",
    signal,
  });
  if (!r.ok) return fail(r);
  if (!r.body) throw new ApiError(0, "no_body", "the response had no body");
  await readSSE(r.body, (data) => {
    try {
      onEvent(JSON.parse(data) as ChatEvent);
    } catch {
      // a malformed event is ignored
    }
  });
}
