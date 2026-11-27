import { render } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { Markdown, parseBlocks, safeHref } from "./markdown";

function html(text: string): HTMLElement {
  return render(<Markdown text={text} />).container;
}

describe("markdown", () => {
  it("renders the common constructs", () => {
    const el = html("# Title\n\nSome **bold**, *it*, `code` and ~~gone~~.\n\n- a\n- b\n  - nested\n\n1. one\n2. two\n\n> quote\n\n---\n\n| a | b |\n|---|--:|\n| 1 | 2 |");
    expect(el.querySelector("h3")?.textContent).toBe("Title");
    expect(el.querySelector("strong")?.textContent).toBe("bold");
    expect(el.querySelector("em")?.textContent).toBe("it");
    expect(el.querySelector("p code")?.textContent).toBe("code");
    expect(el.querySelector("del")?.textContent).toBe("gone");
    expect(el.querySelectorAll("ul > li").length).toBeGreaterThanOrEqual(2);
    expect(el.querySelector("ul ul li")?.textContent).toBe("nested");
    expect(el.querySelectorAll("ol > li").length).toBe(2);
    expect(el.querySelector("blockquote")?.textContent).toBe("quote");
    expect(el.querySelector("hr")).not.toBeNull();
    expect(el.querySelector("td.al-right")?.textContent).toBe("2");
  });

  it("renders fenced code verbatim, also while still streaming", () => {
    const el = html("```python\nx = '<b>' * 2\n```\n");
    expect(el.querySelector("pre code")?.textContent).toBe("x = '<b>' * 2");
    expect(el.querySelector(".md-code-lang")?.textContent).toBe("python");
    const open = html("text\n\n```js\nlet a = 1");
    expect(open.querySelector("pre code")?.textContent).toBe("let a = 1");
  });

  it("never renders raw HTML", () => {
    const el = html('<script>alert(1)</script>\n\n<img src=x onerror=alert(1)> <b>bold?</b>\n\n<iframe src="https://evil"></iframe>');
    expect(el.querySelector("script, img, iframe, b")).toBeNull();
    expect(el.textContent).toContain("<script>alert(1)</script>");
    expect(el.innerHTML).not.toMatch(/<(script|img|iframe)/i);
  });

  it("links only http(s) and mailto, without referrer, and never loads images", () => {
    const el = html("[ok](https://example.com/a) [js](javascript:alert(1)) [data](data:text/html,x) [rel](/api/info) ![pic](https://tracker.example/p.png) see https://example.org/x.");
    const as = [...el.querySelectorAll("a")];
    expect(as.map((a) => a.getAttribute("href"))).toEqual(["https://example.com/a", "https://tracker.example/p.png", "https://example.org/x"]);
    for (const a of as) {
      expect(a.getAttribute("rel")).toContain("noreferrer");
      expect(a.getAttribute("target")).toBe("_blank");
    }
    expect(el.querySelector("img")).toBeNull();
    expect(el.textContent).toContain("[image: pic]");
    expect(el.textContent).toContain("js");
    expect(safeHref("javascript:alert(1)")).toBeNull();
    expect(safeHref(" JAVASCRIPT:alert(1)")).toBeNull();
    expect(safeHref("https://a.b/c")).toBe("https://a.b/c");
  });

  it("keeps line breaks inside a paragraph", () => {
    const el = html("line one\nline two");
    expect(el.querySelectorAll("p br").length).toBe(1);
    expect(parseBlocks("a\nb\n\nc").length).toBe(2);
  });
});
