import type { ChatRequest } from "../api/types";

/** Body for /v1/chat/completions mirroring a playground request (unset optionals omitted). */
export function requestBody(req: ChatRequest): Record<string, unknown> {
  const out: Record<string, unknown> = { model: req.model, messages: req.messages };
  if (req.temperature !== undefined) out.temperature = req.temperature;
  if (req.top_p !== undefined) out.top_p = req.top_p;
  if (req.max_tokens !== undefined) out.max_tokens = req.max_tokens;
  if (req.seed !== undefined) out.seed = req.seed;
  if (req.stream) out.stream = true;
  return out;
}

function shellQuote(s: string): string {
  return `'${s.replace(/'/g, `'\\''`)}'`;
}

export function curlSnippet(req: ChatRequest, origin: string): string {
  const body = JSON.stringify(requestBody(req), null, 2);
  return [
    `curl -s ${origin}/v1/chat/completions \\`,
    `  -H "Authorization: Bearer $PGW_API_KEY" \\`,
    `  -H "Content-Type: application/json" \\`,
    req.stream ? `  -N \\` : null,
    `  -d ${shellQuote(body)}`,
  ]
    .filter(Boolean)
    .join("\n");
}

function pyLiteral(v: unknown, indent = 0): string {
  const pad = "    ".repeat(indent);
  if (v === null || v === undefined) return "None";
  if (typeof v === "boolean") return v ? "True" : "False";
  if (typeof v === "number") return String(v);
  if (typeof v === "string") return JSON.stringify(v);
  if (Array.isArray(v)) {
    if (!v.length) return "[]";
    return `[\n${v.map((x) => `${pad}    ${pyLiteral(x, indent + 1)}`).join(",\n")},\n${pad}]`;
  }
  const entries = Object.entries(v as Record<string, unknown>);
  return `{${entries.map(([k, x]) => `${JSON.stringify(k)}: ${pyLiteral(x, indent)}`).join(", ")}}`;
}

export function pythonSnippet(req: ChatRequest, origin: string): string {
  const lines = [
    "import os",
    "from openai import OpenAI",
    "",
    `client = OpenAI(base_url="${origin}/v1", api_key=os.environ["PGW_API_KEY"])`,
    "",
    "resp = client.chat.completions.create(",
    `    model=${JSON.stringify(req.model)},`,
    `    messages=${pyLiteral(req.messages, 1)},`,
  ];
  if (req.temperature !== undefined) lines.push(`    temperature=${req.temperature},`);
  if (req.top_p !== undefined) lines.push(`    top_p=${req.top_p},`);
  if (req.max_tokens !== undefined) lines.push(`    max_tokens=${req.max_tokens},`);
  if (req.seed !== undefined) lines.push(`    seed=${req.seed},`);
  if (req.stream) {
    lines.push("    stream=True,", ")", "for chunk in resp:", "    if chunk.choices and chunk.choices[0].delta.content:", '        print(chunk.choices[0].delta.content, end="", flush=True)');
  } else {
    lines.push(")", "print(resp.choices[0].message.content)");
  }
  return lines.join("\n");
}
