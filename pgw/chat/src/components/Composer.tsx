import { useEffect, useRef, type FormEvent, type KeyboardEvent } from "react";
import { IconArrowUp, IconStop } from "./Icons";

interface Props {
  value: string;
  onChange(v: string): void;
  onSend(): void;
  onStop(): void;
  streaming: boolean;
  disabled: boolean;
  hint: string | null;
  maxChars: number;
  autoFocus?: boolean;
}

export function Composer(p: Props) {
  const ref = useRef<HTMLTextAreaElement>(null);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    // grow with the text up to a few lines (rows attribute: no inline style)
    const lines = Math.min(10, Math.max(1, p.value.split("\n").length + Math.floor(p.value.length / 90)));
    el.rows = lines;
  }, [p.value]);
  useEffect(() => {
    if (p.autoFocus) ref.current?.focus();
  }, [p.autoFocus]);
  const canSend = !p.streaming && !p.disabled && p.value.trim().length > 0 && p.value.length <= p.maxChars;
  const submit = (e?: FormEvent) => {
    e?.preventDefault();
    if (canSend) p.onSend();
  };
  const key = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      submit();
    }
  };
  return (
    <form className="composer" onSubmit={submit}>
      <div className="composer-box">
        <textarea
          ref={ref}
          value={p.value}
          onChange={(e) => p.onChange(e.target.value)}
          onKeyDown={key}
          placeholder="Message PGW Chat"
          aria-label="Message"
          rows={1}
          maxLength={p.maxChars}
        />
        <div className="composer-row">
          <span className="composer-hint">{p.hint}</span>
          {p.streaming ? (
            <button type="button" className="send-btn is-stop" onClick={p.onStop} aria-label="Stop generating">
              <IconStop />
            </button>
          ) : (
            <button type="submit" className="send-btn" disabled={!canSend} aria-label="Send message">
              <IconArrowUp />
            </button>
          )}
        </div>
      </div>
    </form>
  );
}
