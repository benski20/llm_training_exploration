// What the privacy indicator and screen say. Every line must be true right now for this app: plaintext exists only
// on this computer (the client library runs in the `pgw chat` process here); the GPU service computes on masked data
// only; every GPU result is checked here. We never claim more (no "end-to-end encrypted": the provider computes on
// the data, it does not merely relay it) and we name what the provider does see (metadata).

import type { Info, Status } from "../api/types";

export type Tone = "ok" | "info" | "warn" | "bad";

export interface Fact {
  id: string;
  tone: Tone;
  title: string;
  detail: string;
}

export function indicator(info: Info | null, st: Status | null): { tone: Tone; label: string; detail: string } {
  if (info?.demo) return { tone: "warn", label: "Demo", detail: "Canned answers: no model and no GPU are used." };
  if (st?.phase === "blocked") return { tone: "bad", label: "Paused", detail: "An integrity check failed." };
  if (st && st.integrity_failures > 0)
    return { tone: "bad", label: "Check failed", detail: "A GPU result failed its integrity check and was discarded." };
  const n = st?.verified_requests ?? 0;
  return {
    tone: "ok",
    label: "Private",
    detail: n > 0 ? `${n} answer${n === 1 ? "" : "s"} verified this session` : "Plaintext stays on this computer",
  };
}

export function facts(info: Info, st: Status | null): Fact[] {
  const s = info.settings;
  const out: Fact[] = [];
  if (info.demo)
    out.push({
      id: "demo",
      tone: "warn",
      title: "Demo mode",
      detail: "Answers are canned text from this app. No model runs and nothing is sent to a GPU.",
    });
  out.push({
    id: "device",
    tone: "ok",
    title: "Your words stay on this computer",
    detail:
      "Prompts, answers, the model's working memory and its sampling exist only in the pgw chat process on this " +
      "computer and in this browser tab. No server of ours receives them.",
  });
  out.push({
    id: "masked",
    tone: "ok",
    title: "The GPU sees only masked numbers",
    detail:
      `Every layer's input is hidden with a fresh one-time mask before it leaves this computer` +
      (s.mask_scheme ? ` (mask scheme ${s.mask_scheme})` : "") +
      ". The GPU service computes on those masked values and cannot recover your text from them.",
  });
  const n = st?.verified_requests ?? 0;
  const bad = st?.integrity_failures ?? 0;
  out.push({
    id: "verified",
    tone: bad > 0 ? "bad" : "ok",
    title: bad > 0 ? `${bad} result${bad === 1 ? "" : "s"} failed a check` : "Every GPU result is checked",
    detail:
      (bad > 0
        ? "A result that fails its check is discarded and the model pauses; nothing unverified is shown. "
        : "") +
      `Each result is verified on this computer (Freivalds' test and range checks) before it is used. ` +
      `${n} answer${n === 1 ? "" : "s"} verified since pgw chat started.`,
  });
  out.push({
    id: "metadata",
    tone: "info",
    title: "What the GPU service can see",
    detail:
      "Which model you use, how many tokens each request has, when you send them and how much traffic they make. " +
      (s.usage_reports
        ? "The app also sends it a short record per request with these same counts and whether every check passed, never any text."
        : "Usage records are off."),
  });
  const pc = st?.prefix_cache ?? s.prefix_cache === "on";
  out.push({
    id: "prefix",
    tone: pc ? "warn" : "info",
    title: pc ? "Conversation reuse is on" : "Conversation reuse is off",
    detail: pc
      ? "Earlier turns are kept in this computer's memory and not sent again, so answers start sooner. The GPU service " +
        "can then tell that a request continued an earlier one and roughly how long the conversation is. It sees no more text."
      : "Each turn sends the whole conversation again (masked). Turning reuse on (pgw chat --prefix-cache on) is faster " +
        "for long chats, but lets the GPU service tell that two requests belong to one conversation.",
  });
  if (s.speculation && s.speculation !== "off")
    out.push({
      id: "spec",
      tone: "info",
      title: "Speculative decoding is on",
      detail:
        "A small draft model on this computer guesses several tokens; the GPU checks them in one pass. Answers are " +
        "identical, and the GPU service sees the same number of passes whatever the guesses.",
    });
  out.push({
    id: "history",
    tone: "info",
    title: info.history.persistent ? "History is saved on this computer" : "History is not saved",
    detail: info.history.persistent
      ? `Conversations are files readable only by your user account${info.history.location ? ` in ${info.history.location}` : ""}. ` +
        "They are protected by your disk encryption, not by an extra password. Delete any conversation from the sidebar."
      : "Conversations live in memory and disappear when pgw chat stops.",
  });
  return out;
}

/** The plain-language line under an answer while tokens are slow to arrive. */
export function slowHint(tripsPerToken: number | undefined): string {
  const t = tripsPerToken ?? 113;
  return `Each token takes about ${t} masked round trips to the GPU, so answers arrive slowly over long distances.`;
}
