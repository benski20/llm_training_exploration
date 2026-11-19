// Screenshots of every page in light and dark theme against the mock build served by
// `vite preview` under the production CSP. Usage:
//   npm run build:mock && npm run preview:mock &   (serves http://localhost:4174/console/)
//   npm run screenshots
// Any console error or CSP violation is printed and makes the script exit non-zero.

import { createRequire } from "node:module";
import { mkdirSync } from "node:fs";
import { fileURLToPath } from "node:url";

// Playwright is preinstalled globally (NODE_PATH), not a project dependency; CJS resolution honours NODE_PATH.
const { chromium } = createRequire(import.meta.url)("playwright");

const BASE = process.env.PGW_CONSOLE_URL ?? "http://localhost:4174/console";
const OUT = fileURLToPath(new URL("../screenshots/", import.meta.url));
mkdirSync(OUT, { recursive: true });

const PAGES = [
  ["overview", "/"],
  ["models", "/models"],
  ["masks", "/masks"],
  ["audit", "/audit"],
  ["keys", "/keys"],
  ["security", "/security"],
];

const problems = [];

async function shot(page, name) {
  await page.waitForTimeout(450);
  await page.screenshot({ path: `${OUT}${name}.png` });
  console.log("saved", name);
}

async function run(theme) {
  const browser = await chromium.launch();
  const ctx = await browser.newContext({ viewport: { width: 1280, height: 800 }, deviceScaleFactor: 1, colorScheme: theme });
  const page = await ctx.newPage();
  page.on("console", (m) => {
    if (m.type() === "error") problems.push(`[${theme}] console: ${m.text()}`);
  });
  page.on("pageerror", (e) => problems.push(`[${theme}] pageerror: ${e.message}`));

  await page.goto(`${BASE}/login`);
  await page.waitForSelector(".login-card");
  await shot(page, `login-${theme}`);
  await page.fill("input[type=password]", "admin");
  await page.click("button[type=submit]");
  await page.waitForSelector(".stats-grid .stat-value:not(:has(.skeleton))");

  for (const [name, path] of PAGES) {
    await page.goto(`${BASE}${path}`);
    await page.waitForSelector(".page");
    await page.waitForLoadState("networkidle");
    await shot(page, `${name}-${theme}`);
  }

  // Detail views.
  await page.goto(`${BASE}/audit`);
  await page.waitForSelector(".row-click");
  await page.click(".row-click >> nth=1");
  await page.waitForSelector("dialog[open]");
  await shot(page, `audit-detail-${theme}`);
  await page.keyboard.press("Escape");

  await page.goto(`${BASE}/keys`);
  await page.getByRole("button", { name: "Create key" }).first().click();
  await page.waitForSelector("dialog[open]");
  await page.fill("dialog[open] input >> nth=0", "analytics-pipeline");
  await shot(page, `keys-create-${theme}`);
  await page.getByRole("button", { name: "Create key" }).last().click();
  await page.waitForSelector(".secret");
  await shot(page, `keys-secret-${theme}`);

  await page.goto(`${BASE}/playground`);
  await page.waitForSelector(".model-picker");
  await page.fill("#pg-input", "How does masked inference keep my prompts private?");
  await page.keyboard.press("Enter");
  await page.waitForSelector(".msg-assistant .msg-meta", { timeout: 15000 });
  await shot(page, `playground-${theme}`);
  await page.getByRole("tab", { name: "Code" }).click();
  await shot(page, `playground-code-${theme}`);

  await browser.close();
}

async function viewer() {
  const browser = await chromium.launch();
  const ctx = await browser.newContext({ viewport: { width: 1024, height: 768 }, deviceScaleFactor: 1, colorScheme: "light" });
  const page = await ctx.newPage();
  page.on("pageerror", (e) => problems.push(`[viewer] pageerror: ${e.message}`));
  await page.goto(`${BASE}/login`);
  await page.fill("input[type=password]", "viewer");
  await page.click("button[type=submit]");
  await page.waitForSelector(".stats-grid");
  await page.goto(`${BASE}/keys`);
  await page.waitForSelector(".table tbody tr:not(.row-skeleton)");
  await shot(page, "keys-viewer-tablet");
  await browser.close();
}

await run("light");
await run("dark");
await viewer();

if (problems.length) {
  console.error(problems.join("\n"));
  process.exit(1);
}
