// Screenshots of every page (light and dark, desktop and phone) and of the first-run setup wizard against the
// mock build served by `vite preview` under the production CSP. Usage:
//   npm run build:mock && npm run preview:mock &   (serves http://localhost:4174/console/)
//   npm run screenshots                             (PGW_SCREENSHOTS_DIR overrides the output folder)
// Files are named console-<page>-<theme>-<desktop|phone>.png and setup-<step>-<theme>-<desktop|phone>.png.
// Any console error or CSP violation is printed and makes the script exit non-zero.

import { createRequire } from "node:module";
import { mkdirSync } from "node:fs";
import { fileURLToPath } from "node:url";

// Playwright is preinstalled globally (NODE_PATH), not a project dependency; CJS resolution honours NODE_PATH.
const { chromium } = createRequire(import.meta.url)("playwright");

const BASE = process.env.PGW_CONSOLE_URL ?? "http://localhost:4174/console";
const OUT = process.env.PGW_SCREENSHOTS_DIR
  ? process.env.PGW_SCREENSHOTS_DIR.replace(/\/?$/, "/")
  : fileURLToPath(new URL("../../docs/design/screenshots/", import.meta.url));
mkdirSync(OUT, { recursive: true });

const PAGES = [
  ["overview", "/"],
  ["models", "/models"],
  ["masks", "/masks"],
  ["audit", "/audit"],
  ["keys", "/keys"],
  ["security", "/security"],
];
const VIEWPORTS = { desktop: { width: 1440, height: 900, scale: 1 }, phone: { width: 390, height: 844, scale: 2 } };

const problems = [];

async function shot(page, name, full = true) {
  await page.waitForTimeout(450);
  await page.screenshot({ path: `${OUT}${name}.png`, fullPage: full });
  console.log("saved", name);
}

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

async function run(browser, theme, vp) {
  const [ctx, page] = await context(browser, theme, vp, `${theme}/${vp}`);
  const sfx = `${theme}-${vp}`;
  await page.goto(`${BASE}/login`);
  await page.waitForSelector(".login-card");
  await shot(page, `console-login-${sfx}`);
  await page.fill("input[type=password]", "admin");
  await page.click("button[type=submit]");
  await page.waitForSelector(".stats-grid .stat-value:not(:has(.skeleton))");

  for (const [name, path] of PAGES) {
    await page.goto(`${BASE}${path}`);
    await page.waitForSelector(".page");
    await page.waitForLoadState("networkidle");
    await shot(page, `console-${name}-${sfx}`);
  }
  if (vp === "desktop") {
    // Detail views.
    await page.goto(`${BASE}/audit`);
    await page.waitForSelector(".row-click");
    await page.click(".row-click >> nth=1");
    await page.waitForSelector("dialog[open]");
    await shot(page, `console-audit-detail-${sfx}`, false);
    await page.keyboard.press("Escape");

    await page.goto(`${BASE}/keys`);
    await page.getByRole("button", { name: "Create key" }).first().click();
    await page.waitForSelector("dialog[open]");
    await page.fill("dialog[open] input >> nth=0", "analytics-pipeline");
    await shot(page, `console-keys-create-${sfx}`, false);
    await page.getByRole("button", { name: "Create key" }).last().click();
    await page.waitForSelector(".secret");
    await shot(page, `console-keys-secret-${sfx}`, false);
  }
  await page.goto(`${BASE}/playground`);
  await page.waitForSelector(".model-picker");
  await page.fill("#pg-input", "How does masked inference keep my prompts private?");
  await page.keyboard.press("Enter");
  await page.waitForSelector(".msg-assistant .msg-meta", { timeout: 15000 });
  await shot(page, `console-playground-${sfx}`);
  await ctx.close();
}

async function setup(browser, theme, vp) {
  const [ctx, page] = await context(browser, theme, vp, `setup ${theme}/${vp}`);
  await page.addInitScript(() => sessionStorage.setItem("pgw-mock-setup", "1"));
  const sfx = `${theme}-${vp}`;
  await page.goto(`${BASE}/`);
  await page.waitForSelector(".login-form input[type=password]");
  await shot(page, `setup-1-token-${sfx}`);
  await page.fill(".login-form input[type=password]", "setup");
  await page.click(".login-form button[type=submit]");
  await page.waitForSelector(".setup-form");
  await shot(page, `setup-2-enroll-${sfx}`);
  await ctx.close();
}

const browser = await chromium.launch();
for (const vp of Object.keys(VIEWPORTS)) {
  for (const theme of ["light", "dark"]) {
    await run(browser, theme, vp);
    await setup(browser, theme, vp);
  }
}
await browser.close();

if (problems.length) {
  console.error(problems.join("\n"));
  process.exit(1);
}
