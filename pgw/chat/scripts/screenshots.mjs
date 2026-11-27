// Screenshots of PGW Chat (light and dark, desktop and phone) from the mock build served by `vite preview` under
// the production CSP. Usage:
//   npm run build:mock && npm run preview:mock &   (serves http://localhost:4175/)
//   npm run screenshots                             (PGW_SCREENSHOTS_DIR overrides the output folder)
// Files: chat-<view>-<theme>-<desktop|phone>.png in docs/design/screenshots/. Any console error or CSP violation
// makes the script exit non-zero.

import { createRequire } from "node:module";
import { mkdirSync } from "node:fs";
import { fileURLToPath } from "node:url";

// Playwright is preinstalled globally (NODE_PATH), not a project dependency; CJS resolution honours NODE_PATH.
const { chromium } = createRequire(import.meta.url)("playwright");

const BASE = process.env.PGW_CHAT_URL ?? "http://localhost:4175";
const OUT = process.env.PGW_SCREENSHOTS_DIR
  ? process.env.PGW_SCREENSHOTS_DIR.replace(/\/?$/, "/")
  : fileURLToPath(new URL("../../docs/design/screenshots/", import.meta.url));
mkdirSync(OUT, { recursive: true });

const VIEWPORTS = { desktop: { width: 1440, height: 900, scale: 1 }, phone: { width: 390, height: 844, scale: 2 } };
const problems = [];

async function context(browser, theme, vp, tag) {
  const v = VIEWPORTS[vp];
  const ctx = await browser.newContext({ viewport: { width: v.width, height: v.height }, deviceScaleFactor: v.scale, colorScheme: theme });
  const page = await ctx.newPage();
  page.on("console", (m) => {
    if (m.type() === "error") problems.push(`[${tag}] console: ${m.text()}`);
  });
  page.on("pageerror", (e) => problems.push(`[${tag}] pageerror: ${e.message}`));
  return [ctx, page];
}

async function shot(page, name) {
  await page.waitForTimeout(400);
  await page.screenshot({ path: `${OUT}${name}.png` });
  console.log("saved", name);
}

async function run(browser, theme, vp) {
  const sfx = `${theme}-${vp}`;
  const phone = vp === "phone";
  let [ctx, page] = await context(browser, theme, vp, sfx);

  // welcome (no saved chats)
  await page.goto(`${BASE}/?state=empty`);
  await page.waitForSelector(".welcome-title");
  await shot(page, `chat-welcome-${sfx}`);

  // a conversation
  await page.goto(`${BASE}/c/c_000000000000000000000001`);
  await page.waitForSelector(".msg-assistant .private-tag");
  await shot(page, `chat-conversation-${sfx}`);

  // streaming (slow link): mid-answer, stop button visible
  await page.goto(`${BASE}/?state=slow`);
  await page.waitForSelector(".welcome-title");
  if (phone && (await page.$(".sidebar.is-open"))) await page.click(".scrim");
  await page.fill("textarea", "How does masking keep my words private?");
  await page.keyboard.press("Enter");
  await page.waitForSelector(".thinking");
  await page.waitForSelector(".slow-hint", { timeout: 10000 });
  await shot(page, `chat-waiting-${sfx}`);
  await page.waitForSelector(".is-live .prose", { timeout: 15000 });
  await page.waitForTimeout(5200);
  await shot(page, `chat-streaming-${sfx}`);
  await page.click(".send-btn.is-stop");
  await page.waitForSelector(".msg-meta .meta-item");

  // privacy screen
  await page.goto(`${BASE}/c/c_000000000000000000000001`);
  await page.waitForSelector(".indicator");
  if (phone && (await page.$(".sidebar.is-open"))) await page.click(".scrim");
  await page.click(".indicator");
  await page.waitForSelector(".privacy .fact");
  await shot(page, `chat-privacy-${sfx}`);
  await page.keyboard.press("Escape");

  // first-run setup and the stalled provider
  await page.goto(`${BASE}/?state=setup`);
  await page.waitForSelector(".setup-card .setup-progress");
  await shot(page, `chat-setup-${sfx}`);
  await page.goto(`${BASE}/c/c_000000000000000000000002?state=stalled`);
  await page.waitForSelector(".setup-card.is-warn");
  if (phone && (await page.$(".sidebar.is-open"))) await page.click(".scrim");
  await shot(page, `chat-stalled-${sfx}`);

  if (!phone) {
    await page.goto(`${BASE}/c/c_000000000000000000000001`);
    await page.click(".picker-btn");
    await page.waitForSelector(".picker-menu");
    await shot(page, `chat-models-${sfx}`);
  } else {
    await page.goto(`${BASE}/c/c_000000000000000000000001`);
    await page.waitForSelector(".topbar");
    if (!(await page.$(".sidebar.is-open"))) await page.click('[aria-label="Open sidebar"]');
    await page.waitForSelector(".sidebar.is-open");
    await shot(page, `chat-sidebar-${sfx}`);
  }
  await ctx.close();
}

const browser = await chromium.launch();
try {
  for (const theme of ["light", "dark"]) for (const vp of ["desktop", "phone"]) await run(browser, theme, vp);
} finally {
  await browser.close();
}
if (problems.length) {
  console.error(problems.join("\n"));
  process.exit(1);
}
