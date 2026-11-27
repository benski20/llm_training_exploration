# PGW design system

One stylesheet, `src/pgw/design/pgw.css`, styles every PGW web surface: the developer platform served by
the provider (`src/pgw/platform/`), the gateway console and its first-run setup wizard (`console/`, built
into `src/pgw/console/static/`). The console imports it at build time (Vite bundles it with the fonts);
the platform serves it as is at `/platform/assets/pgw.css`.

| | |
|---|---|
| Fonts | IBM Plex Sans (UI, variable) and IBM Plex Mono (identifiers, fingerprints, axis ticks, code), woff2 in `src/pgw/design/fonts/` (SIL OFL 1.1, licence files alongside; from the Fontsource 5.3.0 packages). No CDN, no runtime network call. |
| Palette | Graphite neutrals with one accent (cobalt `#1f4bd1` light; `#4f7fff` for lines and `#2d5ae6` for filled buttons dark). Muted green, amber and red carry state only. Every text token meets WCAG AA (4.5:1) on the surfaces it is used on, in both themes; status colours also on their soft backgrounds. |
| Themes | Light and dark; dark follows `prefers-color-scheme` unless the page sets `<html data-theme>` (theme toggle, Settings). |
| Type | 11 / 12 / 13 (UI) / 14 (reading) / 15 / 18 / 22 / 26 px; page titles 18 px; uppercase tracked 11 px labels (stat labels, table headers, nav groups); tabular numerals for figures, numbers right-aligned in tables. |
| Space, radius, elevation | 4 px grid, dense rhythm (30 px controls, 34 px table rows); radius 2 px (4 px on dialogs), square status dots; hairline 1 px borders, no soft shadows (one crisp 1 px drop on overlays). Charts: 1.25 px lines, a flat 6-9 % fill under the primary series (no gradients), hairline grid, compact legends. |
| Focus | 2 px square accent outline on `:focus-visible` everywhere (1 px offset); inputs add a 1 px ring. Skip link to the main content. |
| Motion | 80-200 ms, decelerating; off under `prefers-reduced-motion`. |
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
