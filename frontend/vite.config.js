import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // Fail loudly if 5173 is already taken, rather than silently switching
    // to 5174/5175/... -- a silent port change is exactly what caused a
    // real CORS mismatch during development of this project (the backend's
    // ALLOWED_ORIGINS didn't match the port Vite quietly switched to). If
    // this throws, either free up port 5173 or update backend/.env's
    // ALLOWED_ORIGINS to match whatever port you configure here instead.
    strictPort: true,
  },
})
