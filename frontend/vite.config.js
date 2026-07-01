import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';

// The viewer talks to the FastAPI backend over a WebSocket. In dev we proxy
// /ws to the backend so the frontend can use a same-origin relative URL.
// BASE_PATH is set for the GitHub Pages build (project subpath).
export default defineConfig({
  base: process.env.BASE_PATH || '/',
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      '/ws': { target: 'ws://127.0.0.1:8000', ws: true },
      '/health': { target: 'http://127.0.0.1:8000' },
    },
  },
});
