/// <reference types="vitest" />
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { fileURLToPath } from "node:url";

const outDir = fileURLToPath(new URL("../src/pgw/console/static", import.meta.url));

// The same policy the gateway sends on /console/*; used by `vite preview` so screenshots and manual
// checks run under the production CSP (the dev server needs inline scripts for HMR, so not there).
const CSP =
  "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; " +
  "frame-ancestors 'none'; base-uri 'none'; form-action 'self'";

export default defineConfig(({ mode }) => ({
  // "demo": a self-contained preview with fixture data (relative paths, hash routing), not served by a gateway.
  base: mode === "demo" ? "./" : "/console/",
  plugins: [react()],
  build: {
    outDir,
    emptyOutDir: true,
    target: "es2020",
    sourcemap: false,
    assetsInlineLimit: 0,
    modulePreload: { polyfill: false },
    reportCompressedSize: true,
  },
  define: {
    __MOCK__: JSON.stringify(mode === "mock" || mode === "demo"),
    __DEMO__: JSON.stringify(mode === "demo"),
  },
  server: { port: 5173 },
  preview: { headers: { "Content-Security-Policy": CSP } },
  test: {
    environment: "jsdom",
    include: ["src/**/*.test.{ts,tsx}"],
    setupFiles: ["src/test-setup.ts"],
  },
}));
