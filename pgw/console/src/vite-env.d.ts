/// <reference types="vite/client" />

/** True only in `--mode mock` builds/dev servers (see vite.config.ts). */
declare const __MOCK__: boolean;

/** True only in `--mode demo` builds: standalone preview with fixture data and hash routing. */
declare const __DEMO__: boolean;
