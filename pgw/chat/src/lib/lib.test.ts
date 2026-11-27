import { beforeEach, describe, expect, it, vi } from "vitest";
import type { Info, Status } from "../api/types";
import { createSSEParser } from "../api/sse";
import { duration, groupOf, liveRate, tokensPerSecond } from "./format";
import { facts, indicator } from "./privacy";
import { takeTokenFromLocation } from "./token";

const TOKEN = "AbCdEfGhIjKlMnOpQrStUvWxYz0123456789_-abcd";

describe("token", () => {
  beforeEach(() => sessionStorage.clear());
  it("takes the token from the fragment, stores it for this tab and strips it from the address bar", () => {
    const replaceState = vi.fn();
    const t = takeTokenFromLocation({ hash: `#token=${TOKEN}`, pathname: "/", search: "" }, { replaceState } as unknown as History);
    expect(t).toBe(TOKEN);
    expect(replaceState).toHaveBeenCalledWith(null, "", "/");
    expect(sessionStorage.getItem("pgw-chat-token")).toBe(TOKEN);
    expect(takeTokenFromLocation({ hash: "", pathname: "/", search: "" })).toBe(TOKEN);
  });
  it("ignores malformed tokens", () => {
    expect(takeTokenFromLocation({ hash: "#token=short", pathname: "/", search: "" })).toBeNull();
    expect(takeTokenFromLocation({ hash: "#token=<script>aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", pathname: "/", search: "" })).toBeNull();
  });
});

describe("format", () => {
  it("formats durations, rates and groups", () => {
    expect(duration(0.25)).toBe("250 ms");
    expect(duration(12.4)).toBe("12 s");
    expect(duration(372)).toBe("6 min");
    expect(tokensPerSecond(3.456)).toBe("3.5 tok/s");
    expect(tokensPerSecond(null)).toBe("");
    expect(liveRate(11, 1000, 3000)).toBe(5);
    expect(liveRate(1, 1000, 3000)).toBeNull();
    expect(groupOf(1000, 1000 + 60)).toBe("Today");
    expect(groupOf(1000, 1000 + 3 * 86400)).toBe("Previous 7 days");
  });
});

describe("sse", () => {
  it("parses data events split across chunks and skips keep-alives", () => {
    const got: string[] = [];
    const p = createSSEParser((d) => got.push(d));
    p.push('data: {"type":"del');
    p.push('ta"}\n\n: ping\n\ndata: {"type":"done"}\n');
    p.end();
    expect(got).toEqual(['{"type":"delta"}', '{"type":"done"}']);
  });
});

const INFO = {
  demo: false,
  version: "0.1.0",
  settings: { mask_scheme: "pks-v3", prefix_cache: "auto", speculation: "off", usage_reports: true, factory_weights: "pack", masked_head: "auto" },
  history: { persistent: true, location: "/home/u/.local/state/pgw/chat/conversations" },
} as unknown as Info;
const ST = { verified_requests: 2, integrity_failures: 0, prefix_cache: false, phase: "ready" } as unknown as Status;

describe("privacy copy", () => {
  it("states only what is true, and never claims end-to-end encryption", () => {
    const all = facts(INFO, ST);
    const text = all.map((f) => `${f.title} ${f.detail}`).join(" ");
    expect(text).not.toMatch(/end-to-end|e2e|zero.knowledge|anonymous/i);
    expect(all.find((f) => f.id === "device")?.tone).toBe("ok");
    expect(all.find((f) => f.id === "metadata")?.detail).toMatch(/token/);
    expect(all.find((f) => f.id === "verified")?.detail).toContain("2 answers verified");
    expect(all.find((f) => f.id === "prefix")?.title).toBe("Conversation reuse is off");
    expect(all.find((f) => f.id === "history")?.detail).toContain("not by an extra password");
  });
  it("names the prefix-cache signal when it is on, and flags demo and failures", () => {
    const on = facts(INFO, { ...ST, prefix_cache: true });
    expect(on.find((f) => f.id === "prefix")?.detail).toMatch(/can then tell that a request continued/);
    expect(indicator({ ...INFO, demo: true }, ST).label).toBe("Demo");
    expect(facts({ ...INFO, demo: true }, ST)[0].id).toBe("demo");
    expect(indicator(INFO, { ...ST, integrity_failures: 1 }).tone).toBe("bad");
    expect(indicator(INFO, ST)).toEqual({ tone: "ok", label: "Private", detail: "2 answers verified this session" });
  });
});
