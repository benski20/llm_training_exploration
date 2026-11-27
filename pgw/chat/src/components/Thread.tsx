import { useState } from "react";
import type { ErrorView, Message } from "../api/types";
import { duration, tokensPerSecond } from "../lib/format";
import { Markdown } from "../lib/markdown";
import { slowHint } from "../lib/privacy";
import { IconCheck, IconCopy, IconRetry, IconWarn } from "./Icons";

export interface Live {
  text: string;
  tokens: number;
  startedAt: number;
  firstAt: number | null;
  lastAt: number | null;
  rate: number | null;
}

interface Props {
  messages: Message[];
  live: Live | null;
  now: number;
  tripsPerToken?: number;
  onRetry(): void;
  failure: ErrorView | null;
}

function Copy({ text }: { text: string }) {
  const [done, setDone] = useState(false);
  return (
    <button type="button" className="icon-btn subtle" aria-label={done ? "Copied" : "Copy answer"} title="Copy"
      onClick={() => {
        void navigator.clipboard?.writeText(text).then(() => {
          setDone(true);
          setTimeout(() => setDone(false), 1500);
        });
      }}>
      {done ? <IconCheck /> : <IconCopy />}
    </button>
  );
}

export function ErrorNotice({ error, onRetry }: { error: ErrorView; onRetry?: () => void }) {
  return (
    <div className={`notice notice-${error.kind === "integrity" ? "bad" : "warn"}`} role="alert">
      <IconWarn />
      <div className="notice-body">
        <div className="notice-title">{error.title}</div>
        {error.message ? <div className="notice-text">{error.message}</div> : null}
        {error.retryable && onRetry ? (
          <button type="button" className="link-btn" onClick={onRetry}>
            Try again
          </button>
        ) : null}
      </div>
    </div>
  );
}

function Meta({ m, last, onRetry }: { m: Message; last: boolean; onRetry(): void }) {
  const tps = tokensPerSecond(m.tokens_per_s);
  return (
    <div className="msg-meta">
      {m.verified ? (
        <span className="private-tag" title="Plaintext never left this computer; every GPU result was checked here before use.">
          <IconCheck />
          Computed privately
        </span>
      ) : null}
      {m.status === "stopped" ? <span className="meta-item">Stopped</span> : null}
      {tps ? <span className="meta-item">{tps}</span> : null}
      {m.elapsed_s ? <span className="meta-item">{duration(m.elapsed_s)}</span> : null}
      <span className="meta-actions">
        {m.content ? <Copy text={m.content} /> : null}
        {last ? (
          <button type="button" className="icon-btn subtle" aria-label="Regenerate answer" title="Regenerate" onClick={onRetry}>
            <IconRetry />
          </button>
        ) : null}
      </span>
    </div>
  );
}

export function Thread({ messages, live, now, tripsPerToken, onRetry, failure }: Props) {
  const lastAssistant = [...messages].reverse().find((m) => m.role === "assistant");
  const waitingFor = live ? (now - (live.lastAt ?? live.startedAt)) / 1000 : 0;
  return (
    <div className="thread">
      {messages.map((m) =>
        m.role === "user" ? (
          <div key={m.id} className="msg msg-user">
            <div className="bubble">{m.content}</div>
          </div>
        ) : (
          <div key={m.id} className="msg msg-assistant">
            {m.content ? (
              <div className="prose">
                <Markdown text={m.content} />
              </div>
            ) : null}
            {m.error ? <ErrorNotice error={m.error} onRetry={m === lastAssistant ? onRetry : undefined} /> : null}
            {m.content ? <Meta m={m} last={!live && m === lastAssistant} onRetry={onRetry} /> : null}
          </div>
        ),
      )}
      {live ? (
        <div className="msg msg-assistant is-live" aria-live="polite" aria-busy="true">
          {live.text ? (
            <div className="prose is-streaming">
              <Markdown text={live.text} />
            </div>
          ) : (
            <div className="thinking">
              <span className="dots" aria-hidden="true">
                <i />
                <i />
                <i />
              </span>
              <span>{waitingFor > 3 ? "Waiting for the GPU" : "Computing privately"}</span>
            </div>
          )}
          <div className="msg-meta">
            <span className="private-tag is-live">
              <IconCheck />
              Computing privately
            </span>
            {live.rate ? <span className="meta-item">{tokensPerSecond(live.rate)}</span> : null}
          </div>
          {waitingFor > 4 ? <p className="slow-hint">{slowHint(tripsPerToken)}</p> : null}
        </div>
      ) : null}
      {failure && !live ? <ErrorNotice error={failure} onRetry={failure.retryable ? onRetry : undefined} /> : null}
    </div>
  );
}
