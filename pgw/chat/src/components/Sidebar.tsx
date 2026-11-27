import { useState } from "react";
import type { ConversationSummary } from "../api/types";
import { groupOf } from "../lib/format";
import type { ThemePref } from "../lib/theme";
import { IconClose, IconLock, IconMonitor, IconMoon, IconPlus, IconSidebar, IconSun, IconTrash, Mark } from "./Icons";

interface Props {
  open: boolean;
  conversations: ConversationSummary[];
  current: string | null;
  persistent: boolean;
  theme: ThemePref;
  onTheme(t: ThemePref): void;
  onNew(): void;
  onOpen(id: string): void;
  onDelete(id: string): void;
  onDeleteAll(): void;
  onClose(): void;
  onPrivacy(): void;
}

export function Sidebar(p: Props) {
  const [confirm, setConfirm] = useState<string | null>(null);
  const groups: [string, ConversationSummary[]][] = [];
  for (const c of p.conversations) {
    const g = groupOf(c.updated_at);
    const last = groups[groups.length - 1];
    if (last && last[0] === g) last[1].push(c);
    else groups.push([g, [c]]);
  }
  const nextTheme: Record<ThemePref, ThemePref> = { system: "light", light: "dark", dark: "system" };
  return (
    <aside className={`sidebar${p.open ? " is-open" : ""}`} aria-label="Conversations">
      <div className="sidebar-head">
        <div className="brand">
          <Mark />
          <span className="brand-name">PGW Chat</span>
        </div>
        <button type="button" className="icon-btn" onClick={p.onClose} aria-label="Close sidebar">
          <IconSidebar className="only-wide" />
          <IconClose className="only-narrow" />
        </button>
      </div>
      <button type="button" className="new-chat" onClick={p.onNew}>
        <IconPlus />
        New chat
      </button>
      <nav className="conv-list">
        {p.conversations.length === 0 ? (
          <p className="conv-empty">{p.persistent ? "Your conversations will appear here." : "History is off: chats are not saved."}</p>
        ) : null}
        {groups.map(([g, items]) => (
          <div key={g} className="conv-group">
            <div className="conv-group-label">{g}</div>
            {items.map((c) => (
              <div key={c.id} className={`conv-item${c.id === p.current ? " is-current" : ""}`}>
                {confirm === c.id ? (
                  <div className="conv-confirm">
                    <span>Delete this chat?</span>
                    <button type="button" className="link-btn danger" onClick={() => { setConfirm(null); p.onDelete(c.id); }}>
                      Delete
                    </button>
                    <button type="button" className="link-btn" onClick={() => setConfirm(null)}>
                      Keep
                    </button>
                  </div>
                ) : (
                  <>
                    <button type="button" className="conv-title" onClick={() => p.onOpen(c.id)} title={c.title}
                      aria-current={c.id === p.current ? "page" : undefined}>
                      {c.title || "New chat"}
                    </button>
                    <button type="button" className="icon-btn conv-del" aria-label={`Delete ${c.title || "chat"}`}
                      onClick={() => setConfirm(c.id)}>
                      <IconTrash />
                    </button>
                  </>
                )}
              </div>
            ))}
          </div>
        ))}
      </nav>
      <div className="sidebar-foot">
        <button type="button" className="foot-btn" onClick={p.onPrivacy}>
          <IconLock />
          Privacy
        </button>
        <button type="button" className="foot-btn" onClick={() => p.onTheme(nextTheme[p.theme])}
          aria-label={`Theme: ${p.theme}`} title={`Theme: ${p.theme}`}>
          {p.theme === "light" ? <IconSun /> : p.theme === "dark" ? <IconMoon /> : <IconMonitor />}
        </button>
        {p.conversations.length > 0 ? (
          confirm === "*" ? (
            <span className="conv-confirm">
              <button type="button" className="link-btn danger" onClick={() => { setConfirm(null); p.onDeleteAll(); }}>
                Delete all
              </button>
              <button type="button" className="link-btn" onClick={() => setConfirm(null)}>
                Cancel
              </button>
            </span>
          ) : (
            <button type="button" className="foot-btn" onClick={() => setConfirm("*")} title="Delete all chats" aria-label="Delete all chats">
              <IconTrash />
            </button>
          )
        ) : null}
      </div>
    </aside>
  );
}
