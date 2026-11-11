import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// Cible du backend FastAPI. Surchargeable via VITE_API_TARGET quand le port
// 8000 est deja pris (ex. un autre projet Docker sur la meme machine).
const apiTarget = process.env.VITE_API_TARGET || 'http://localhost:8000'

const proxy = {
  '/api': {
    target: apiTarget,
    changeOrigin: true,
    secure: false,
  },
}

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  // Vite default config: react() plugin handles JSX, no special transformers needed
  server: {
    proxy,
  },
  preview: {
    proxy,
  },
})