import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// Relative base so the bundle works at any GitHub Pages subpath
// (e.g. https://<org>.github.io/index-futures-stat-arb/) and from file://.
export default defineConfig({
  base: './',
  plugins: [react()],
  build: { sourcemap: false, chunkSizeWarningLimit: 1200 },
})
