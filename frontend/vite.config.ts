import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

export default defineConfig(({ mode }) => {
  const envDir = path.resolve(__dirname, '..')
  const env = loadEnv(mode, envDir, '')

  const apiProxy = {
    '/api': {
      target: env.VITE_API_PROXY_TARGET || 'http://localhost:8000',
      changeOrigin: true,
    },
  }

  const minioPrefix = (env.VITE_MINIO_DEV_PROXY_PREFIX || '/__minio').replace(/\/$/, '') || '/__minio'
  const minioTarget = (env.VITE_MINIO_PROXY_TARGET || env.VITE_MINIO_BROWSER_ORIGIN || '').trim()

  const proxy: Record<string, object> = { ...apiProxy }
  if (minioTarget) {
    proxy[minioPrefix] = {
      target: minioTarget,
      changeOrigin: true,
      rewrite: (p: string) => p.replace(new RegExp(`^${minioPrefix.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}`), ''),
    }
  }

  return {
    envDir,
    plugins: [react()],
    resolve: {
      alias: {
        '@': path.resolve(__dirname, './src'),
      },
    },
    server: {
      port: parseInt(env.FRONTEND_PORT || '9002', 10),
      host: env.HOST || '0.0.0.0',
      proxy,
    },
  }
})
