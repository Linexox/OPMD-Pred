import { Agent as HttpAgent } from 'node:http'
import { Agent as HttpsAgent } from 'node:https'
import { defineConfig, loadEnv } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig(({ mode }) => {
  const target = loadEnv(mode, process.cwd(), 'VITE_').VITE_DEMO_SERVICE_URL?.trim().replace(/\/$/, '')
  return {
    plugins: [vue()],
    base: './',
    server: {
      proxy: target && /^https?:\/\//.test(target) ? {
        '/api': {
          target,
          changeOrigin: true,
          // Development requests go directly to the model host, independently
          // of the browser/system proxy. APK requests retain their own URL.
          agent: target.startsWith('https:') ? new HttpsAgent() : new HttpAgent(),
        },
      } : undefined,
    },
  }
})
