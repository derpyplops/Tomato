import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0', // Allow access from remote machines
    port: 5173,
    allowedHosts: [
      '1af5bc9fca95.ngrok-free.app',
      '.ngrok-free.app', // Allow all ngrok free domains
      '.ngrok.io', // Allow ngrok.io domains
      '.ngrok.app', // Allow ngrok.app domains
    ],
  }
})
