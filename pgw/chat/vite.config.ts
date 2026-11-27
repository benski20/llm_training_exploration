/// <reference types="vitest" />
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { fileURLToPath } from "node:url";

const outDir = fileURLToPath(new URL("../src/pgw/chat/static", import.meta.url));

// The policy `pgw chat` sends (src/pgw/chat/server.py CSP); used by `vite preview` so screenshots and manual
// checks run under the production CSP (the dev server needs inline scripts for HMR, so not there).
const CSP =
  "default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self' data:; font-src 'self'; " +
  "connect-src 'self'; manifest-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'";

export default defineConfig(({ mode }) => ({
  base: "/",
  plugins: [react()],
  build: {
    outDir,
    emptyOutDir: true,
    target: "es2020",
    sourcemap: false,
    assetsInlineLimit: 0,
    modulePreload: { polyfill: false },
    reportCompressedSize: false,
  },
  define: {
    __MOCK__: JSON.stringify(mode === "mock"),
  },
  server: { port: 5174, fs: { allow: [".."] } },
  preview: { headers: { "Content-Security-Policy": CSP } },
  test: {
    environment: "jsdom",
    include: ["src/**/*.test.{ts,tsx}"],
    setupFiles: ["src/test-setup.ts"],
    define: { __MOCK__: "false" },
  },
}));
