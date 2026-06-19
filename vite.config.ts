import { resolve } from 'node:path'
import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'
import { VitePWA } from 'vite-plugin-pwa'

// https://vite.dev/config/
// Base path: defaults to the GitHub Pages project path so `npm run build`
// (used by the central web-deploy/web-ci workflows) produces correct asset URLs.
// Override with VITE_BASE=/ for local/preview builds.
export default defineConfig({
  base: process.env.VITE_BASE ?? '/llmvram/',
  plugins: [
    react(),
    tailwindcss(),
    VitePWA({
      registerType: 'autoUpdate',
      // No SW in dev — avoids stale-cache confusion during development
      devOptions: { enabled: false },
      workbox: {
        // Precache all build assets including bundled JSON data files
        globPatterns: ['**/*.{js,css,html,ico,png,svg,woff2,json}'],
        // Exclude source maps — large, not needed for offline functionality
        globIgnores: ['**/*.map'],
      },
      manifest: {
        name: 'LLM VRAM Calculator',
        short_name: 'VRAM Calc',
        description:
          'Estimate VRAM requirements and performance for running LLMs on various GPU configurations',
        // start_url / scope intentionally omitted — vite-plugin-pwa derives them from
        // the Vite `base` option: '/' locally, '/llmvram/' in GitHub Pages CI
        display: 'standalone',
        background_color: '#0f172a',
        theme_color: '#6366f1',
        orientation: 'any',
        icons: [
          {
            src: 'pwa-64x64.png',
            sizes: '64x64',
            type: 'image/png',
          },
          {
            src: 'pwa-192x192.png',
            sizes: '192x192',
            type: 'image/png',
            purpose: 'any',
          },
          {
            src: 'pwa-512x512.png',
            sizes: '512x512',
            type: 'image/png',
            purpose: 'any',
          },
          {
            src: 'maskable-icon-512x512.png',
            sizes: '512x512',
            type: 'image/png',
            purpose: 'maskable',
          },
        ],
      },
    }),
  ],
  resolve: {
    alias: {
      '@': resolve(__dirname, './src'),
      '@engines': resolve(__dirname, './src/engines'),
      '@components': resolve(__dirname, './src/components'),
      '@store': resolve(__dirname, './src/store'),
      '@types': resolve(__dirname, './src/types'),
      '@utils': resolve(__dirname, './src/utils'),
      '@data': resolve(__dirname, './src/data'),
      '@hooks': resolve(__dirname, './src/hooks'),
    },
  },
  worker: {
    format: 'es',
  },
  build: {
    target: 'esnext',
    sourcemap: true,
  },
})
