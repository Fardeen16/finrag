import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    // Proxying rather than calling http://localhost:8080 directly keeps dev
    // same-origin, which is how the app is served in production once FastAPI
    // mounts the built SPA. Avoids CORS behaviour that only exists in dev.
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8080',
        changeOrigin: true,
      },
    },
  },
})
