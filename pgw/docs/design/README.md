# PGW design system

One stylesheet, `src/pgw/design/pgw.css`, styles every PGW web surface: the developer platform served by
the provider (`src/pgw/platform/`), the gateway console and its first-run setup wizard (`console/`, built
into `src/pgw/console/static/`). The console imports it at build time (Vite bundles it with the fonts);
the platform serves it as is at `/platform/assets/pgw.css`.

| | |
|---|---|
| Fonts | Inter (UI) and Geist Mono (code), variable woff2 in `src/pgw/design/fonts/` (SIL OFL 1.1, licences alongside). No CDN, no runtime network call. |
| Palette | Neutral greys with one accent (iris `#5b5bd6` light, `#7b7bf0` dark). Green, amber and red carry state only. Every text token meets WCAG AA (4.5:1) on the surfaces it is used on, in both themes. |
| Themes | Light and dark; dark follows `prefers-color-scheme` unless the page sets `<html data-theme>` (theme toggle, Settings). |
| Type | 11 / 12 / 13 (UI) / 14 (reading) / 16 / 20 / 24 / 30 px, tight tracking on headings, tabular numerals for figures. |
| Space, radius | 4 px grid; radius 4 / 6 / 10 / 14 px. |
| Focus | 2 px accent outline on `:focus-visible` everywhere; inputs add a soft ring. Skip link to the main content. |
| Motion | 100-240 ms, decelerating; off under `prefers-reduced-motion`. |
| Layout | Sidebar + top bar shell; an icon rail below 960 px; on phones one header row and a scrolling tab row. |

Rules both apps follow: strict CSP (no inline script or style, no external origin), charts as SVG with
geometry in attributes, status never by colour alone (dot + label), and no prompt or response content
anywhere in a console.

## Screenshots

`screenshots/` holds every page in light and dark at desktop (1440 px) and phone (390 px) width:
`platform-*` (developer platform), `console-*` (gateway console) and `setup-*` (setup wizard).

```bash
python scripts/platform_screenshots.py                    # platform, from the in-process demo
cd console && npm run build:mock && (npm run preview:mock &) && npm run screenshots   # console + wizard
```
