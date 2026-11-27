// A small, safe Markdown renderer for model answers. It builds React elements only (never raw HTML injection), so any HTML
// in an answer is shown as text. Links are kept only for http(s) and mailto URLs and open in a new tab without a
// referrer; images are NOT loaded (an answer must never make the browser fetch a URL by itself: that would leak
// data to whoever controls the URL), they render as a plain link the user can choose to follow.
//
// Supported: paragraphs with line breaks, # headings, fenced and indented-free code blocks, `inline code`,
// **bold**, *italic*, ~~strike~~, [links](https://...), bare https:// links, > quotes, - / 1. lists (nested by
// indentation), | tables |, --- rules. Unclosed constructs while streaming render as what they are so far.

import { Fragment, type ReactNode } from "react";

export type Block =
  | { t: "p"; text: string }
  | { t: "h"; level: number; text: string }
  | { t: "code"; lang: string; text: string }
  | { t: "quote"; blocks: Block[] }
  | { t: "list"; ordered: boolean; start: number; items: Block[][] }
  | { t: "table"; head: string[]; align: ("left" | "center" | "right" | null)[]; rows: string[][] }
  | { t: "hr" };

const FENCE = /^ {0,3}(`{3,}|~{3,})\s*([\w+#.-]*)[^`]*$/;
const HEADING = /^ {0,3}(#{1,6})\s+(.*?)\s*#*\s*$/;
const HR = /^ {0,3}([-*_])(?:\s*\1){2,}\s*$/;
const QUOTE = /^ {0,3}>\s?(.*)$/;
const LIST = /^(\s*)([-*+]|\d{1,9}[.)])\s+(.*)$/;
const TABLE_SEP = /^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$/;

function splitRow(line: string): string[] {
  let s = line.trim();
  if (s.startsWith("|")) s = s.slice(1);
  if (s.endsWith("|") && !s.endsWith("\\|")) s = s.slice(0, -1);
  return s.split(/(?<!\\)\|/).map((c) => c.trim().replace(/\\\|/g, "|"));
}

function startsBlock(line: string, next: string | undefined): boolean {
  return (
    FENCE.test(line) ||
    HEADING.test(line) ||
    HR.test(line) ||
    QUOTE.test(line) ||
    LIST.test(line) ||
    (line.includes("|") && next !== undefined && TABLE_SEP.test(next))
  );
}

export function parseBlocks(src: string): Block[] {
  const lines = src.replace(/\r\n?/g, "\n").split("\n");
  const out: Block[] = [];
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    if (!line.trim()) {
      i++;
      continue;
    }
    const fence = FENCE.exec(line);
    if (fence) {
      const marker = fence[1];
      const body: string[] = [];
      i++;
      while (i < lines.length && !new RegExp(`^ {0,3}${marker[0]}{${marker.length},}\\s*$`).test(lines[i])) {
        body.push(lines[i]);
        i++;
      }
      i++; // the closing fence (or the end of a still-streaming block)
      out.push({ t: "code", lang: fence[2] || "", text: body.join("\n") });
      continue;
    }
    const h = HEADING.exec(line);
    if (h) {
      out.push({ t: "h", level: h[1].length, text: h[2] });
      i++;
      continue;
    }
    if (HR.test(line)) {
      out.push({ t: "hr" });
      i++;
      continue;
    }
    if (QUOTE.test(line)) {
      const body: string[] = [];
      while (i < lines.length && QUOTE.test(lines[i])) {
        body.push(QUOTE.exec(lines[i])![1]);
        i++;
      }
      out.push({ t: "quote", blocks: parseBlocks(body.join("\n")) });
      continue;
    }
    if (line.includes("|") && i + 1 < lines.length && TABLE_SEP.test(lines[i + 1])) {
      const head = splitRow(line);
      const align = splitRow(lines[i + 1]).map((c) =>
        c.startsWith(":") && c.endsWith(":") ? "center" : c.endsWith(":") ? "right" : c.startsWith(":") ? "left" : null,
      );
      i += 2;
      const rows: string[][] = [];
      while (i < lines.length && lines[i].includes("|") && lines[i].trim()) {
        rows.push(splitRow(lines[i]));
        i++;
      }
      out.push({ t: "table", head, align, rows });
      continue;
    }
    const li = LIST.exec(line);
    if (li) {
      const indent = li[1].length;
      const ordered = /\d/.test(li[2]);
      const start = ordered ? parseInt(li[2], 10) : 1;
      const items: string[][] = [];
      while (i < lines.length) {
        const m = LIST.exec(lines[i]);
        if (m && m[1].length <= indent + 1 && /\d/.test(m[2]) === ordered) {
          items.push([m[3]]);
          i++;
          continue;
        }
        if (!items.length) break;
        const cur = lines[i];
        if (!cur.trim()) {
          // a blank line continues the list only if the next line is indented or another item
          const nxt = lines[i + 1];
          const nm = nxt !== undefined ? LIST.exec(nxt) : null;
          if (nxt !== undefined && (/^\s{2,}\S/.test(nxt) || (nm && nm[1].length <= indent + 1))) {
            items[items.length - 1].push("");
            i++;
            continue;
          }
          break;
        }
        const lead = /^(\s*)/.exec(cur)![1].length;
        if (lead > indent) {
          items[items.length - 1].push(cur.slice(Math.min(lead, indent + 2)));
          i++;
          continue;
        }
        if (startsBlock(cur, lines[i + 1])) break;
        items[items.length - 1].push(cur.trim()); // a lazy continuation line
        i++;
      }
      out.push({ t: "list", ordered, start, items: items.map((it) => parseBlocks(it.join("\n"))) });
      continue;
    }
    const para: string[] = [line];
    i++;
    while (i < lines.length && lines[i].trim() && !startsBlock(lines[i], lines[i + 1])) {
      para.push(lines[i]);
      i++;
    }
    out.push({ t: "p", text: para.join("\n") });
  }
  return out;
}

/** Only these URL schemes become links; anything else (javascript:, data:, file:, relative) stays text. */
export function safeHref(url: string): string | null {
  const u = url.trim();
  if (/^https?:\/\/[^\s<>"]+$/i.test(u) || /^mailto:[^\s<>"]+$/i.test(u)) return u;
  return null;
}

const INLINE =
  /(`+)([\s\S]*?[^`])\1(?!`)|!\[([^\]]*)\]\(([^)\s]+)(?:\s+"[^"]*")?\)|\[([^\]]+)\]\(([^)\s]+)(?:\s+"[^"]*")?\)|\*\*([\s\S]+?)\*\*|__([\s\S]+?)__|~~([\s\S]+?)~~|(?<![\w*])\*(?!\s)([\s\S]+?)(?<!\s)\*(?!\*)|(?<!\w)_(?!\s)([\s\S]+?)(?<!\s)_(?!\w)|(https?:\/\/[^\s<>()]+[^\s<>().,;:!?'"\]])|\n/;

function link(href: string, children: ReactNode, key: string): ReactNode {
  return (
    <a key={key} href={href} target="_blank" rel="noopener noreferrer nofollow" title={href}>
      {children}
    </a>
  );
}

export function renderInline(text: string, prefix = "i"): ReactNode[] {
  const out: ReactNode[] = [];
  let rest = text;
  let n = 0;
  while (rest) {
    const m = INLINE.exec(rest);
    if (!m) {
      out.push(rest);
      break;
    }
    if (m.index > 0) out.push(rest.slice(0, m.index));
    const k = `${prefix}${n++}`;
    if (m[1]) out.push(<code key={k}>{m[2].replace(/^ (.*) $/, "$1")}</code>);
    else if (m[4] !== undefined) {
      const href = safeHref(m[4]);
      const label = `image: ${m[3] || "untitled"}`;
      out.push(href ? link(href, `[${label}]`, k) : `[${label}]`);
    } else if (m[6] !== undefined) {
      const href = safeHref(m[6]);
      out.push(href ? link(href, renderInline(m[5], k), k) : <Fragment key={k}>{renderInline(m[5], k)}</Fragment>);
    } else if (m[7] !== undefined || m[8] !== undefined) out.push(<strong key={k}>{renderInline(m[7] ?? m[8], k)}</strong>);
    else if (m[9] !== undefined) out.push(<del key={k}>{renderInline(m[9], k)}</del>);
    else if (m[10] !== undefined || m[11] !== undefined) out.push(<em key={k}>{renderInline(m[10] ?? m[11], k)}</em>);
    else if (m[12] !== undefined) out.push(link(m[12], m[12], k));
    else out.push(<br key={k} />);
    rest = rest.slice(m.index + m[0].length);
  }
  return out;
}

function renderBlock(b: Block, key: string): ReactNode {
  switch (b.t) {
    case "p":
      return <p key={key}>{renderInline(b.text, key)}</p>;
    case "h": {
      const Tag = (["h3", "h3", "h4", "h5", "h6", "h6"] as const)[b.level - 1];
      return <Tag key={key}>{renderInline(b.text, key)}</Tag>;
    }
    case "code":
      return (
        <pre key={key} className="md-code" data-lang={b.lang || undefined}>
          {b.lang ? <span className="md-code-lang">{b.lang}</span> : null}
          <code>{b.text}</code>
        </pre>
      );
    case "quote":
      return <blockquote key={key}>{b.blocks.map((x, i) => renderBlock(x, `${key}.${i}`))}</blockquote>;
    case "list": {
      const items = b.items.map((it, i) => (
        <li key={i}>
          {it.length === 1 && it[0].t === "p" ? renderInline(it[0].text, `${key}.${i}`) : it.map((x, j) => renderBlock(x, `${key}.${i}.${j}`))}
        </li>
      ));
      return b.ordered ? (
        <ol key={key} start={b.start !== 1 ? b.start : undefined}>
          {items}
        </ol>
      ) : (
        <ul key={key}>{items}</ul>
      );
    }
    case "table":
      return (
        <div key={key} className="md-table">
          <table>
            <thead>
              <tr>
                {b.head.map((c, i) => (
                  <th key={i} className={b.align[i] ? `al-${b.align[i]}` : undefined}>
                    {renderInline(c, `${key}h${i}`)}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {b.rows.map((r, ri) => (
                <tr key={ri}>
                  {b.head.map((_, ci) => (
                    <td key={ci} className={b.align[ci] ? `al-${b.align[ci]}` : undefined}>
                      {renderInline(r[ci] ?? "", `${key}r${ri}c${ci}`)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      );
    case "hr":
      return <hr key={key} />;
  }
}

export function Markdown({ text }: { text: string }) {
  return <>{parseBlocks(text).map((b, i) => renderBlock(b, `b${i}`))}</>;
}
