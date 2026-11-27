import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError, api, chat, setToken } from "./api/client";
import type { ChatEvent, ConversationSummary, ErrorView, Info, Message, Status } from "./api/types";
import { Composer } from "./components/Composer";
import { IconLock, IconSidebar, Mark } from "./components/Icons";
import { ModelPicker } from "./components/ModelPicker";
import { PrivacyPanel } from "./components/PrivacyPanel";
import { SetupCard, isSettingUp } from "./components/SetupCard";
import { Sidebar } from "./components/Sidebar";
import { type Live, Thread } from "./components/Thread";
import { liveRate } from "./lib/format";
import { indicator } from "./lib/privacy";
import { applyTheme, loadTheme, type ThemePref } from "./lib/theme";
import { takeTokenFromLocation } from "./lib/token";

const MODEL_KEY = "pgw-chat-model";

function conversationFromPath(path: string): string | null {
  const m = /^\/c\/(c_[0-9a-f]{24})$/.exec(path);
  return m ? m[1] : null;
}

function apiErrorView(e: unknown): ErrorView {
  if (e instanceof ApiError) {
    if (e.status === 401)
      return { kind: "auth", title: "This tab is not connected", message: "Open PGW Chat from the link pgw chat printed (or run it again).", retryable: false };
    if (e.code === "busy") return { kind: "busy", title: "Still answering", message: e.message, retryable: true };
    return { kind: e.code, title: "Request not accepted", message: e.message, retryable: e.status >= 500 };
  }
  if (e instanceof DOMException && e.name === "AbortError") return { kind: "aborted", title: "Stopped", message: "", retryable: true };
  return { kind: "connection", title: "PGW Chat is not running", message: "The app on this computer did not answer. Start pgw chat again.", retryable: true };
}

export function NoToken() {
  return (
    <div className="gate">
      <Mark size={36} />
      <h1>Open PGW Chat from your terminal</h1>
      <p>
        This page only works in the tab that <code>pgw chat</code> opened, which holds a private key for this session.
        Run <code>pgw chat</code> again, or open the link it printed.
      </p>
    </div>
  );
}

export function App() {
  const [token] = useState(() => takeTokenFromLocation(window.location, window.history));
  const [info, setInfo] = useState<Info | null>(null);
  const [fatal, setFatal] = useState<ErrorView | null>(null);
  const [status, setStatus] = useState<Status | null>(null);
  const [model, setModel] = useState<string>("");
  const [convs, setConvs] = useState<ConversationSummary[]>([]);
  const [current, setCurrent] = useState<string | null>(() => conversationFromPath(window.location.pathname));
  const [messages, setMessages] = useState<Message[]>([]);
  const [live, setLive] = useState<Live | null>(null);
  const [failure, setFailure] = useState<ErrorView | null>(null);
  const [draft, setDraft] = useState("");
  const [sidebar, setSidebar] = useState(() => window.matchMedia?.("(min-width: 900px)").matches ?? true);
  const [privacy, setPrivacy] = useState(false);
  const [theme, setTheme] = useState<ThemePref>(loadTheme);
  const [now, setNow] = useState(Date.now());
  const gen = useRef<string | null>(null);
  const abort = useRef<AbortController | null>(null);
  const scroller = useRef<HTMLDivElement>(null);
  const stick = useRef(true);

  setToken(token);

  useEffect(() => applyTheme(theme), [theme]);

  const refreshList = useCallback(() => {
    api.conversations().then((r) => setConvs(r.conversations), () => undefined);
  }, []);

  // boot
  useEffect(() => {
    if (!token) return;
    api.info().then(
      (i) => {
        setInfo(i);
        let m = i.default_model;
        try {
          const saved = localStorage.getItem(MODEL_KEY);
          if (saved && i.models.some((x) => x.id === saved)) m = saved;
        } catch {
          // storage blocked
        }
        setModel(m);
        void api.warmup(m).catch(() => undefined);
      },
      (e) => setFatal(apiErrorView(e)),
    );
    refreshList();
  }, [token, refreshList]);

  const idle = live === null;
  // status polling: fast while setting up, slow when ready (and once more after each answer)
  useEffect(() => {
    if (!token || !model) return;
    let stop = false;
    let timer: ReturnType<typeof setTimeout>;
    const tick = () => {
      api.status(model).then(
        (s) => {
          if (stop) return;
          setStatus(s);
          timer = setTimeout(tick, s.phase === "ready" ? 5000 : 1500);
        },
        () => {
          if (!stop) timer = setTimeout(tick, 4000);
        },
      );
    };
    tick();
    return () => {
      stop = true;
      clearTimeout(timer);
    };
  }, [token, model, idle]);

  // open a conversation
  useEffect(() => {
    if (!token) return;
    if (!current) {
      setMessages([]);
      return;
    }
    if (gen.current) return; // the stream fills it in
    api.conversation(current).then(
      (c) => setMessages(c.messages),
      () => {
        setCurrent(null);
        window.history.replaceState(null, "", "/");
      },
    );
  }, [token, current]);

  useEffect(() => {
    const pop = () => setCurrent(conversationFromPath(window.location.pathname));
    window.addEventListener("popstate", pop);
    return () => window.removeEventListener("popstate", pop);
  }, []);

  // a clock for "waiting for the GPU" and the live rate
  useEffect(() => {
    if (idle) return;
    const t = setInterval(() => setNow(Date.now()), 500);
    return () => clearInterval(t);
  }, [idle]);

  // keep the newest text in view unless the reader scrolled up
  useEffect(() => {
    const el = scroller.current;
    if (el && stick.current) el.scrollTop = el.scrollHeight;
  }, [messages, live?.text]);

  const go = (id: string | null) => {
    if (gen.current) return;
    setFailure(null);
    setCurrent(id);
    window.history.pushState(null, "", id ? `/c/${id}` : "/");
    if (!window.matchMedia?.("(min-width: 900px)").matches) setSidebar(false);
  };

  const run = async (req: { message?: string; regenerate?: boolean }) => {
    if (!info || gen.current) return;
    setFailure(null);
    const ctl = new AbortController();
    abort.current = ctl;
    gen.current = "pending";
    stick.current = true;
    const t0 = Date.now();
    setLive({ text: "", tokens: 0, startedAt: t0, firstAt: null, lastAt: null, rate: null });
    if (req.message) {
      setMessages((ms) => [...ms, { id: "pending-user", role: "user", content: req.message!, created_at: t0 / 1000 }]);
    } else {
      setMessages((ms) => {
        const out = [...ms];
        while (out.length && out[out.length - 1].role !== "user") out.pop();
        return out;
      });
    }
    let convId = current;
    const onEvent = (e: ChatEvent) => {
      if (e.type === "start") {
        gen.current = e.generation_id;
        if (!convId) {
          convId = e.conversation_id;
          setCurrent(convId);
          window.history.pushState(null, "", `/c/${convId}`);
        }
        if (e.user_message) {
          const um = e.user_message;
          setMessages((ms) => ms.map((m) => (m.id === "pending-user" ? um : m)));
        }
        refreshList();
      } else if (e.type === "delta") {
        const t = Date.now();
        setLive((l) => {
          if (!l) return l;
          const tokens = l.tokens + 1;
          const firstAt = l.firstAt ?? t;
          return { ...l, text: l.text + e.text, tokens, firstAt, lastAt: t, rate: liveRate(tokens, firstAt, t) };
        });
      } else if (e.type === "done" || e.type === "error") {
        setMessages((ms) => [...ms, e.message]);
        setLive(null);
      }
    };
    try {
      await chat({ conversation_id: convId, model, ...req }, onEvent, ctl.signal);
    } catch (e) {
      const v = apiErrorView(e);
      setLive((l) => {
        if (l?.text) setMessages((ms) => [...ms, { id: "partial", role: "assistant", content: l.text, created_at: Date.now() / 1000, status: "stopped", verified: true }]);
        return null;
      });
      if (v.kind !== "aborted") setFailure(v);
    } finally {
      gen.current = null;
      abort.current = null;
      setLive(null);
      refreshList();
      api.status(model).then(setStatus, () => undefined);
    }
  };

  const send = () => {
    const text = draft.trim();
    if (!text) return;
    setDraft("");
    void run({ message: text });
  };

  const stopGen = () => {
    const g = gen.current;
    if (g && g !== "pending") void api.stop(g).catch(() => abort.current?.abort());
    else abort.current?.abort();
  };

  const selectModel = (m: string) => {
    setModel(m);
    setStatus(null);
    try {
      localStorage.setItem(MODEL_KEY, m);
    } catch {
      // storage blocked
    }
    void api.warmup(m).catch(() => undefined);
  };

  const remove = (id: string) => {
    void api.deleteConversation(id).then(() => {
      if (id === current) go(null);
      refreshList();
    });
  };

  const removeAll = () => {
    void api.deleteAll().then(() => {
      go(null);
      refreshList();
    });
  };

  if (!token) return <NoToken />;
  if (fatal)
    return (
      <div className="gate">
        <Mark size={36} />
        <h1>{fatal.title}</h1>
        <p>{fatal.message}</p>
      </div>
    );
  if (!info) return <div className="gate is-loading" aria-busy="true" />;

  const ind = indicator(info, status);
  const setting = isSettingUp(status);
  const blocked = status?.phase === "blocked";
  const empty = messages.length === 0 && !live;
  const mInfo = info.models.find((x) => x.id === model);
  const hint = setting
    ? "Setup is still running; you can send when it is ready."
    : blocked
      ? "Paused after a failed integrity check."
      : null;

  const composer = (
    <Composer value={draft} onChange={setDraft} onSend={send} onStop={stopGen} streaming={live !== null}
      disabled={setting || blocked} hint={hint} maxChars={info.limits.max_message_chars} autoFocus />
  );

  return (
    <div className={`app${sidebar ? " has-sidebar" : ""}`}>
      {info.demo ? <div className="demo-banner">Demo mode: canned answers, no model and no GPU.</div> : null}
      <Sidebar open={sidebar} conversations={convs} current={current} persistent={info.history.persistent}
        theme={theme} onTheme={setTheme} onNew={() => go(null)} onOpen={go} onDelete={remove} onDeleteAll={removeAll}
        onClose={() => setSidebar(false)} onPrivacy={() => setPrivacy(true)} />
      {sidebar ? <div className="scrim" onClick={() => setSidebar(false)} aria-hidden="true" /> : null}
      <main className="main">
        <header className="topbar">
          {!sidebar ? (
            <button type="button" className="icon-btn" onClick={() => setSidebar(true)} aria-label="Open sidebar">
              <IconSidebar />
            </button>
          ) : null}
          <ModelPicker info={info} model={model} onSelect={selectModel} disabled={live !== null} />
          <span className="spacer" />
          <button type="button" className={`indicator is-${ind.tone}`} onClick={() => setPrivacy(true)}
            title={ind.detail} aria-label={`${ind.label}: ${ind.detail}. Show privacy details`}>
            <IconLock />
            <span className="indicator-label">{ind.label}</span>
            <span className="indicator-detail">{ind.detail}</span>
          </button>
        </header>
        <div className="scroll" ref={scroller}
          onScroll={(e) => {
            const el = e.currentTarget;
            stick.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80;
          }}>
          {empty ? (
            <div className="welcome">
              <h1 className="welcome-title">What&rsquo;s on your mind?</h1>
              <p className="welcome-sub">Your words stay on this computer. Our GPUs only see masked numbers.</p>
              <SetupCard info={info} status={status} onRetry={() => void api.warmup(model)} />
              <div className="welcome-composer">{composer}</div>
            </div>
          ) : (
            <>
              <div className="column">
                <SetupCard info={info} status={status?.phase === "preparing" ? null : status} onRetry={() => void api.warmup(model)} />
                <Thread messages={messages} live={live} now={now} tripsPerToken={mInfo?.trips_per_token}
                  onRetry={() => void run({ regenerate: true })} failure={failure} />
              </div>
            </>
          )}
        </div>
        {!empty ? (
          <div className="dock">
            {composer}
            <p className="dock-note">Answers can be wrong. Plaintext never leaves this computer.</p>
          </div>
        ) : null}
      </main>
      {privacy ? <PrivacyPanel info={info} status={status} onClose={() => setPrivacy(false)} /> : null}
    </div>
  );
}
