import { defineConfig } from 'vite';
import { existsSync, readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = dirname(fileURLToPath(import.meta.url));

// HTTPS (needed by WebXR on the headset) only when the certificates exist; plain http otherwise.
const keyPath = resolve(root, '../ssl/key.pem');
const certPath = resolve(root, '../ssl/cert.pem');
const https = existsSync(keyPath) && existsSync(certPath)
  ? { key: readFileSync(keyPath), cert: readFileSync(certPath) }
  : undefined;

// The Python backend: the REST API and the headset/spectator websockets go through the dev server so the
// page and the API share one origin (no CORS, no mixed content). Regexes so that /headset.html stays a page.
const backend = process.env.DVSIM_BACKEND_URL || 'http://localhost:8000';
const proxy = {
  '^/api(/|$)': { target: backend, changeOrigin: true, ws: true },
  '^/(spectate|headset)(\\?.*)?$': { target: backend, changeOrigin: true, ws: true },
};

export default defineConfig({
  server: {
    host: '0.0.0.0',
    https,
    port: 3000,
    proxy,
  },
  preview: {
    proxy,
  },
  build: {
    rollupOptions: {
      input: {
        main: resolve(root, 'index.html'),
        headset: resolve(root, 'headset.html'),
        phone: resolve(root, 'phone.html'),
      },
    },
  },
});
