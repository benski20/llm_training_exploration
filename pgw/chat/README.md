# PGW Chat (web app source)

React + TypeScript + Vite source of the chat UI that `pgw chat` serves on `127.0.0.1`. The build output is
committed to `../src/pgw/chat/static/` and ships inside the `pgw` package, so users never need Node.js. User guide:
[docs/sdk/chat.md](../docs/sdk/chat.md). Design: [research/reviews/encrypted-chat-app.md](../research/reviews/encrypted-chat-app.md).

```bash
../scripts/build-chat.sh            # npm ci, lint, unit tests, build, static checks
npm ci && npm run dev:mock          # http://localhost:5174/ with canned data (?state=setup|stalled|slow|empty|blocked)
npm run lint && npm run typecheck && npm test
npm run build:mock && npm run preview:mock &  npm run screenshots   # -> ../docs/design/screenshots/chat-*.png
```

Rules the server relies on: the app calls only `/api/*` on its own origin with the per-launch token; no inline
scripts or styles and no external resources (the CSP is `default-src 'none'; script-src 'self'; style-src 'self';
connect-src 'self'; ...`); answers are rendered by `src/lib/markdown.tsx` into elements (never raw HTML; images are
not loaded); hashed asset names under `assets/`. Fonts are self-hosted from `../src/pgw/design/fonts/` (OFL, licences
copied to `public/licenses/`).
