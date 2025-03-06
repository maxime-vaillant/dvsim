import { defineConfig } from 'vite';
import fs from 'fs';
import path from 'path';

export default defineConfig({
  server: {
    host: '0.0.0.0',
    https: {
      key: fs.readFileSync(path.resolve(__dirname, '../ssl/key.pem')),
      cert: fs.readFileSync(path.resolve(__dirname, '../ssl/cert.pem')),
    },
    port: 3000
  }
});