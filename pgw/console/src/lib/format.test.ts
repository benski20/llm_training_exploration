import { describe, expect, it } from "vitest";
import { DASH, fmtBytes, fmtCompact, fmtDuration, fmtInt, fmtMs, fmtPercent, fmtRelative, keyPrefix, shortHash, toEpochMs } from "./format";
import { consumptionRate, runwaySeconds } from "./usage";
import { curlSnippet, pythonSnippet } from "./snippets";

describe("format", () => {
  it("formats integers and compact numbers", () => {
    expect(fmtInt(1234567)).toBe("1,234,567");
    expect(fmtInt(null)).toBe(DASH);
    expect(fmtCompact(950)).toBe("950");
    expect(fmtCompact(1234)).toBe("1.23k");
    expect(fmtCompact(86_420)).toBe("86.4k");
    expect(fmtCompact(1_284_310)).toBe("1.28M");
    expect(fmtCompact(2_000_000_000)).toBe("2B");
    expect(fmtCompact(undefined)).toBe(DASH);
  });

  it("formats bytes, milliseconds and durations", () => {
    expect(fmtBytes(512)).toBe("512 B");
    expect(fmtBytes(1536)).toBe("1.5 KB");
    expect(fmtBytes(41_221_908_112)).toBe("38.4 GB");
    expect(fmtMs(84.4)).toBe("84 ms");
    expect(fmtMs(1240)).toBe("1.24 s");
    expect(fmtMs(0.5)).toBe("0.5 ms");
    expect(fmtDuration(42)).toBe("42s");
    expect(fmtDuration(720)).toBe("12m");
    expect(fmtDuration(3 * 3600 + 20 * 60)).toBe("3h 20m");
    expect(fmtDuration(6 * 86400 + 4 * 3600)).toBe("6d 4h");
    expect(fmtDuration(null)).toBe(DASH);
  });

  it("formats percentages", () => {
    expect(fmtPercent(1)).toBe("100%");
    expect(fmtPercent(0.25)).toBe("25%");
    expect(fmtPercent(0.0625)).toBe("6.3%");
    expect(fmtPercent(0.0084, 2)).toBe("0.84%");
  });

  it("parses ISO and unix timestamps", () => {
    expect(toEpochMs(1_763_337_600)).toBe(1_763_337_600_000);
    expect(toEpochMs("2026-11-19T10:00:00.000Z")).toBe(Date.UTC(2026, 10, 19, 10));
    expect(toEpochMs("nope")).toBeNull();
    const now = Date.UTC(2026, 10, 19, 12);
    expect(fmtRelative(now - 10_000, now)).toBe("just now");
    expect(fmtRelative(now - 5 * 60_000, now)).toBe("5 min ago");
    expect(fmtRelative(now - 3 * 3600_000, now)).toBe("3 h ago");
  });

  it("shortens hashes and key ids", () => {
    expect(shortHash("3fa9c2d1e07b5c8e9a41")).toBe("3fa9c2d1e07b");
    expect(shortHash(null)).toBe(DASH);
    expect(keyPrefix("3fa9c2d1e07b")).toBe("pgw-3fa9c2d1e07b");
  });
});

describe("usage helpers", () => {
  it("estimates consumption rate and runway", () => {
    const series = Array.from({ length: 4 }, (_, i) => ({ t: i * 60, requests: 1, prompt_tokens: 0, completion_tokens: 0, errors: 0, p50_ms: null, p95_ms: null, ttft_p50_ms: null, masks_consumed: 600 }));
    const rate = consumptionRate({ window: "1h", bucket_s: 60, series });
    expect(rate).toBe(10);
    expect(runwaySeconds(3600, rate)).toBe(360);
    expect(runwaySeconds(100, 0)).toBeNull();
    expect(consumptionRate(undefined)).toBeNull();
  });
});

describe("snippets", () => {
  const req = { model: "qwen7b", messages: [{ role: "user" as const, content: "It's fine" }], temperature: 0.7, max_tokens: 64, stream: true };
  it("builds a curl command with shell-safe quoting", () => {
    const s = curlSnippet(req, "https://pgw.internal:8080");
    expect(s).toContain("https://pgw.internal:8080/v1/chat/completions");
    expect(s).toContain("$PGW_API_KEY");
    expect(s).toContain(`It'\\''s fine`);
    expect(s).toContain("-N");
  });
  it("builds an OpenAI SDK snippet", () => {
    const s = pythonSnippet({ ...req, stream: false, seed: 7 }, "https://pgw.internal:8080");
    expect(s).toContain('base_url="https://pgw.internal:8080/v1"');
    expect(s).toContain("seed=7");
    expect(s).toContain('{"role": "user", "content": "It\'s fine"}');
    expect(s).not.toContain("stream=True");
  });
});
