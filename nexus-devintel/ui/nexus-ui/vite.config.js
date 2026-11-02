import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  // Vite default config: react() plugin handles JSX, no special transformers needed
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:8098',
        changeOrigin: true,
        secure: false,
      },
    },
  },
})