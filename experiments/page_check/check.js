// Drives the page against experiments/page_check/fake_server.py (port 8098).
const { chromium } = require('playwright')   // NODE_PATH=$(npm root -g) if installed globally;
(async () => {
  const b = await chromium.launch(); const pg = await b.newPage(); const errs = [];
  pg.on('pageerror', e => errs.push(String(e)));
  await pg.goto('http://127.0.0.1:8098/');
  let t = Date.now();
  await pg.fill('#msg', 'first question'); await pg.click('#send');
  await pg.waitForFunction(() => { const c = document.querySelectorAll('.card.bot')[0]; return c && c.textContent.includes('Reply') }, null, {timeout: 120000});
  console.log('stuck one recovered after', (Date.now() - t) / 1000, 's:', (await pg.locator('.card.bot').nth(0).innerText()).slice(0, 60));
  t = Date.now();
  await pg.fill('#msg', 'second question'); await pg.click('#send');
  await pg.waitForFunction(() => { const c = document.querySelectorAll('.card.bot'); return c.length == 2 && c[1].textContent.includes('Reply') }, null, {timeout: 120000});
  console.log('slow one arrived after', (Date.now() - t) / 1000, 's:', (await pg.locator('.card.bot').nth(1).innerText()).slice(0, 60));
  console.log('history:', JSON.stringify(await pg.evaluate(() => msgs.map(m => m.role))), 'page errors:', errs);
  await b.close();
})();
