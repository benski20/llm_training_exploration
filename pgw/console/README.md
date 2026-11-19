# PGW Console (web app source)

React + TypeScript + Vite source of the operator console that the gateway serves at `/console/`.
The build output is committed to `../src/pgw/console/static/` and ships inside the `pgw` Python
package and the gateway image, so operators never need Node.js. Operator documentation:
[docs/operations/console.md](../docs/operations/console.md). API contract:
[docs/api/console-api.md](../docs/api/console-api.md).

## Rebuilding

After changing anything under `console/`, rebuild and commit the static files:

```bash
scripts/build-console.sh            # from pgw/: npm ci, lint, unit tests, build, static checks
scripts/build-console.sh --quick    # npm ci, build, static checks only
```

Needs Node.js 20+ and npm. The script runs `npm ci` and `npm run build` here (`tsc -b && vite
build`, output to `../src/pgw/console/static/`, emptied first), then
`tests/console/test_console_build.py`, which fails when `index.html` is missing, references an asset
that does not exist, contains inline scripts or styles (blocked by the console CSP), or when the
package metadata would not ship the files. Asset names carry a content hash, so an unchanged source
gives an identical build.

## Development

```bash
npm ci
npm run dev:mock          # http://localhost:5173/console/ with mock data (no gateway needed)
npm run lint && npm run typecheck && npm test
```

| script | what it does |
|---|---|
| `dev`, `dev:mock` | Vite dev server (`dev:mock` with built-in mock data); to try the app against a real gateway, build it and open the gateway's `/console/` |
| `build` | type-check and build into `../src/pgw/console/static/` |
| `build:mock` / `preview:mock` | mock-data build served by `vite preview` under the production CSP |
| `screenshots` | screenshots of every page (light and dark) from the mock preview; fails on console errors |
| `lint`, `typecheck`, `test` | ESLint, `tsc -b`, Vitest unit tests |

Rules the gateway relies on:

* Base path `/console/` (`vite.config.ts`); the app calls only `/console/api/*` and `/admin/*` on its
  own origin.
* No inline scripts or styles, no external resources: the gateway sends
  `script-src 'self'; style-src 'self'; connect-src 'self'`.
* Hashed file names under `assets/` (served with a one-year immutable cache).

End-to-end check against a real gateway in a real browser (headless Chromium, Python Playwright):

```bash
cd .. && python -m pytest -q tests/console/test_console_browser.py
```
