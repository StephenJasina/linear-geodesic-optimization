import { fileURLToPath } from 'node:url';
import { resolve } from 'node:path';
import { defineConfig } from 'vite';

const siteDirectory = fileURLToPath(new URL('.', import.meta.url));

export default defineConfig({
  build: {
    rollupOptions: {
      input: {
        main: resolve(siteDirectory, 'index.html'),
        comparison: resolve(siteDirectory, 'compare.html'),
      },
    },
  },
});
