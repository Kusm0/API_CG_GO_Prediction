import { defineConfig } from 'vite';

export default defineConfig({
  build: {
    outDir: 'static/js',
    rollupOptions: {
      input: './src/app.js',
      output: {
        entryFileNames: 'index.js',
      },
    },
  },
});
