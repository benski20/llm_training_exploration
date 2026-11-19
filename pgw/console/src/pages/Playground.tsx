import { useEffect, useMemo, useRef, useState, type FormEvent, type KeyboardEvent } from "react";
import { api, ApiError } from "../api/client";
import type { ChatMessage, ChatRequest } from "../api/types";
import { useGateway } from "../app/gateway";
import { useSession } from "../app/session";
import { describeError } from "../lib/errors";
import { fmtInt, fmtMs, shortHash } from "../lib/format";
import { curlSnippet, pythonSnippet } from "../lib/snippets";
import {
  Button,
  Callout,
  Card,
  CodeBlock,
  CopyButton,
  EmptyState,
  Field,
  IconChat,
  IconLock,
  IconSend,
  IconStop,
  Input,
  PageHeader,
  Select,
  Switch,
  Tabs,
  Textarea,
  tabPanelProps,
} from "../ui";

interface Meta {
  ttftMs: number | null;
  tokPerS: number | null;
  completionTokens: number | null;
  promptTokens: number | null;
  totalMs: number;
  fingerprint: string | null;
  finishReason: string | null;
  requestId: string | null;
}

interface Turn {
  role: "user" | "assistant";
  content: string;
  meta?: Meta;
  error?: string;
  pending?: boolean;
}

interface Params {
  model: string;
  system: string;
  temperature: number;
  topP: number;
  maxTokens: number;
  seed: string;
  stream: boolean;
}

function buildRequest(p: Params, turns: Turn[], next?: string): ChatRequest {
  const messages: ChatMessage[] = [];
  if (p.system.trim()) messages.push({ role: "system", content: p.system.trim() });
  for (const t of turns) if (!t.error && !t.pending && t.content) messages.push({ role: t.role, content: t.content });
  if (next !== undefined) messages.push({ role: "user", content: next });
  const seed = p.seed.trim() === "" ? undefined : Number(p.seed);
  return {
    model: p.model,
    messages,
    temperature: p.temperature,
    top_p: p.topP,
    max_tokens: p.maxTokens,
    seed: Number.isFinite(seed) ? seed : undefined,
    stream: p.stream,
  };
}

function MetaLine({ m }: { m: Meta }) {
  const parts: [string, string][] = [];
  if (m.ttftMs !== null) parts.push(["TTFT", fmtMs(m.ttftMs)]);
  if (m.tokPerS !== null) parts.push(["Speed", `${m.tokPerS.toFixed(1)} tok/s`]);
  if (m.completionTokens !== null) parts.push(["Tokens", fmtInt(m.completionTokens)]);
  parts.push(["Total", fmtMs(m.totalMs)]);
  if (m.finishReason) parts.push(["Finish", m.finishReason]);
  return (
    <div className="msg-meta">
      {parts.map(([k, v]) => (
        <span key={k} className="msg-meta-item">
          <span className="msg-meta-k">{k}</span>
          <span className="num">{v}</span>
        </span>
      ))}
      {m.fingerprint ? (
        <span className="msg-meta-item" title="system_fingerprint: the model pack's fingerprint prefix">
          <span className="msg-meta-k">fp</span>
          <span className="mono">{shortHash(m.fingerprint)}</span>
        </span>
      ) : null}
    </div>
  );
}

export function PlaygroundPage() {
  const { isAdmin } = useSession();
  const gw = useGateway();
  const [p, setP] = useState<Params>({ model: "", system: "", temperature: 0.7, topP: 1, maxTokens: 256, seed: "", stream: true });
  const [turns, setTurns] = useState<Turn[]>([]);
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const [side, setSide] = useState<"params" | "code">("params");
  const [lang, setLang] = useState<"curl" | "python">("curl");
  const abortRef = useRef<AbortController | null>(null);
  const logRef = useRef<HTMLDivElement>(null);
  const model = p.model || gw.model || "";

  useEffect(() => {
    logRef.current?.scrollTo({ top: logRef.current.scrollHeight });
  }, [turns]);
  useEffect(() => () => abortRef.current?.abort(), []);

  const set = <K extends keyof Params>(k: K, v: Params[K]) => setP((x) => ({ ...x, [k]: v }));
  const [lastReq, setLastReq] = useState<ChatRequest | null>(null);
  // Snippet: the message being typed (with history), else the last request sent, else an example.
  const preview = useMemo(
    () => (draft.trim() ? buildRequest({ ...p, model }, turns, draft.trim()) : lastReq ?? buildRequest({ ...p, model }, [], "Hello")),
    [p, model, turns, draft, lastReq],
  );
  const origin = typeof window !== "undefined" ? window.location.origin : "https://pgw.internal:8080";

  const send = async (e?: FormEvent) => {
    e?.preventDefault();
    const text = draft.trim();
    if (!text || busy || !model) return;
    const req = buildRequest({ ...p, model }, turns, text);
    setLastReq(req);
    setDraft("");
    setBusy(true);
    setTurns((t) => [...t, { role: "user", content: text }, { role: "assistant", content: "", pending: true }]);
    const update = (fn: (t: Turn) => Turn) => setTurns((ts) => ts.map((t, i) => (i === ts.length - 1 ? fn(t) : t)));
    const ac = new AbortController();
    abortRef.current = ac;
    const t0 = performance.now();
    let tFirst: number | null = null;
    let deltas = 0;
    try {
      if (req.stream) {
        const r = await api.chatStream(req, {
          signal: ac.signal,
          onDelta: (d) => {
            if (tFirst === null) tFirst = performance.now();
            deltas++;
            update((t) => ({ ...t, content: t.content + d }));
          },
        });
        const t1 = performance.now();
        const n = r.usage?.completion_tokens ?? deltas;
        const decodeS = tFirst !== null ? (t1 - tFirst) / 1000 : 0;
        update((t) => ({
          ...t,
          pending: false,
          content: r.content,
          meta: {
            ttftMs: tFirst !== null ? tFirst - t0 : null,
            tokPerS: decodeS > 0 && n > 1 ? (n - 1) / decodeS : null,
            completionTokens: n,
            promptTokens: r.usage?.prompt_tokens ?? null,
            totalMs: t1 - t0,
            fingerprint: r.fingerprint,
            finishReason: r.finishReason,
            requestId: r.requestId,
          },
        }));
      } else {
        const r = await api.chat(req);
        const t1 = performance.now();
        const n = r.usage?.completion_tokens ?? null;
        update((t) => ({
          ...t,
          pending: false,
          content: r.choices[0]?.message.content ?? "",
          meta: {
            ttftMs: null,
            tokPerS: n && t1 > t0 ? n / ((t1 - t0) / 1000) : null,
            completionTokens: n,
            promptTokens: r.usage?.prompt_tokens ?? null,
            totalMs: t1 - t0,
            fingerprint: r.system_fingerprint ?? null,
            finishReason: r.choices[0]?.finish_reason ?? null,
            requestId: r.id,
          },
        }));
      }
    } catch (err) {
      if ((err as Error)?.name === "AbortError") {
        update((t) => ({ ...t, pending: false, error: t.content ? undefined : "Stopped.", meta: t.content ? { ttftMs: tFirst !== null ? tFirst - t0 : null, tokPerS: null, completionTokens: deltas, promptTokens: null, totalMs: performance.now() - t0, fingerprint: null, finishReason: "cancelled", requestId: null } : undefined }));
      } else {
        const d = describeError(err);
        update((t) => ({ ...t, pending: false, error: `${d.title}: ${err instanceof ApiError ? err.message : d.detail}` }));
      }
    } finally {
      abortRef.current = null;
      setBusy(false);
    }
  };

  const onKey = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      void send();
    }
  };

  const code = lang === "curl" ? curlSnippet(preview, origin) : pythonSnippet(preview, origin);

  return (
    <div className="page page-playground">
      <PageHeader
        title="Playground"
        description="Chat with a model through the masked pipeline. Requests run as the built-in key console-playground and count toward its quota."
        actions={
          <Button variant="ghost" onClick={() => setTurns([])} disabled={busy || !turns.length}>
            Clear chat
          </Button>
        }
      />
      {!isAdmin ? (
        <Callout tone="neutral" icon={<IconLock />} title="Read-only session">
          The playground runs inference and is available to admin sessions only.
        </Callout>
      ) : null}
      <div className="playground">
        <Card className="chat-card" flush>
          <div className="chat-log" ref={logRef} aria-live="polite" aria-busy={busy}>
            {turns.length === 0 ? (
              <EmptyState icon={<IconChat size={20} />} title="Start a conversation">
                Prompts, activations and completions stay on this gateway. The compute service only receives masked activations.
              </EmptyState>
            ) : (
              turns.map((t, i) => (
                <div key={i} className={`msg msg-${t.role}`}>
                  <div className="msg-role">{t.role === "user" ? "You" : model}</div>
                  {t.error ? (
                    <div className="msg-error" role="alert">
                      {t.error}
                    </div>
                  ) : (
                    <div className="msg-body">
                      {t.content}
                      {t.pending ? <span className="caret" aria-hidden="true" /> : null}
                    </div>
                  )}
                  {t.meta ? <MetaLine m={t.meta} /> : null}
                  {t.role === "assistant" && t.content && !t.pending ? (
                    <div className="msg-actions">
                      <CopyButton text={t.content} label="Copy response" />
                    </div>
                  ) : null}
                </div>
              ))
            )}
          </div>
          <form className="composer" onSubmit={send}>
            <label htmlFor="pg-input" className="sr-only">
              Message
            </label>
            <Textarea
              id="pg-input"
              rows={2}
              placeholder={isAdmin ? "Send a message (Enter to send, Shift+Enter for a new line)" : "Admin role required"}
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              onKeyDown={onKey}
              disabled={!isAdmin}
            />
            {busy ? (
              <Button variant="secondary" icon={<IconStop />} onClick={() => abortRef.current?.abort()} iconOnly label="Stop generating" />
            ) : (
              <Button type="submit" variant="primary" icon={<IconSend />} iconOnly label="Send message" disabled={!isAdmin || !draft.trim() || !model} />
            )}
          </form>
        </Card>

        <Card className="pg-side" flush>
          <div className="pg-side-tabs">
            <Tabs
              idBase="pg"
              label="Playground panel"
              items={[
                { value: "params", label: "Parameters" },
                { value: "code", label: "Code" },
              ]}
              value={side}
              onChange={setSide}
            />
          </div>
          {side === "params" ? (
            <div className="pg-params" {...tabPanelProps("pg", "params")}>
              <Field label="Model">
                {({ id }) => (
                  <Select id={id} value={model} onChange={(e) => set("model", e.target.value)}>
                    {gw.models.map((m) => (
                      <option key={m.name} value={m.name} disabled={!m.ready}>
                        {m.name}
                        {m.ready ? "" : " (not ready)"}
                      </option>
                    ))}
                  </Select>
                )}
              </Field>
              <Field label="System prompt">
                {({ id }) => <Textarea id={id} rows={3} value={p.system} onChange={(e) => set("system", e.target.value)} placeholder="Optional" />}
              </Field>
              <Field label="Temperature" aside={<span className="num">{p.temperature.toFixed(2)}</span>} hint="0 is greedy decoding.">
                {({ id, describedBy }) => (
                  <input id={id} aria-describedby={describedBy} className="range" type="range" min={0} max={2} step={0.05} value={p.temperature} onChange={(e) => set("temperature", Number(e.target.value))} />
                )}
              </Field>
              <Field label="Top p" aside={<span className="num">{p.topP.toFixed(2)}</span>}>
                {({ id }) => <input id={id} className="range" type="range" min={0.05} max={1} step={0.05} value={p.topP} onChange={(e) => set("topP", Number(e.target.value))} />}
              </Field>
              <div className="field-row">
                <Field label="Max tokens" hint="Reserves mask slots up front.">
                  {({ id, describedBy }) => (
                    <Input id={id} aria-describedby={describedBy} type="number" min={1} max={8192} value={p.maxTokens} onChange={(e) => set("maxTokens", Math.max(1, Math.min(8192, Number(e.target.value) || 1)))} className="num" />
                  )}
                </Field>
                <Field label="Seed">
                  {({ id }) => <Input id={id} type="number" value={p.seed} onChange={(e) => set("seed", e.target.value)} placeholder="random" className="num" />}
                </Field>
              </div>
              <div className="switch-row">
                <label htmlFor="pg-stream" className="field-label">
                  Stream response
                </label>
                <Switch id="pg-stream" label="Stream response" checked={p.stream} onChange={(v) => set("stream", v)} />
              </div>
            </div>
          ) : (
            <div className="pg-code" {...tabPanelProps("pg", "code")}>
              <p className="muted small">The same request against the OpenAI-compatible API, with an application key.</p>
              <Tabs
                label="Snippet language"
                variant="segmented"
                items={[
                  { value: "curl", label: "curl" },
                  { value: "python", label: "Python" },
                ]}
                value={lang}
                onChange={setLang}
              />
              <CodeBlock code={code} label={lang === "curl" ? "curl" : "Python (openai SDK)"} />
            </div>
          )}
        </Card>
      </div>
    </div>
  );
}
