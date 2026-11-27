// Minimal Server-Sent Events parser for `text/event-stream` responses (only `data:` fields matter
// for the OpenAI-style chat stream). Push-based so it can be unit tested without streams.

export type SSEHandler = (data: string) => void;

export function createSSEParser(onData: SSEHandler): { push(text: string): void; end(): void } {
  let buf = "";
  let data: string[] = [];

  const dispatch = () => {
    if (data.length) onData(data.join("\n"));
    data = [];
  };

  const line = (raw: string) => {
    const l = raw.endsWith("\r") ? raw.slice(0, -1) : raw;
    if (l === "") return dispatch();
    if (l.startsWith(":")) return; // comment / keep-alive
    const i = l.indexOf(":");
    const field = i < 0 ? l : l.slice(0, i);
    let value = i < 0 ? "" : l.slice(i + 1);
    if (value.startsWith(" ")) value = value.slice(1);
    if (field === "data") data.push(value);
  };

  return {
    push(text: string) {
      buf += text;
      let nl: number;
      while ((nl = buf.indexOf("\n")) >= 0) {
        line(buf.slice(0, nl));
        buf = buf.slice(nl + 1);
      }
    },
    end() {
      if (buf) line(buf);
      buf = "";
      dispatch();
    },
  };
}

/** Read a fetch body as SSE, calling `onData` for every event's data payload. */
export async function readSSE(body: ReadableStream<Uint8Array>, onData: SSEHandler): Promise<void> {
  const parser = createSSEParser(onData);
  const reader = body.getReader();
  const dec = new TextDecoder();
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      parser.push(dec.decode(value, { stream: true }));
    }
    parser.push(dec.decode());
    parser.end();
  } finally {
    reader.releaseLock();
  }
}
