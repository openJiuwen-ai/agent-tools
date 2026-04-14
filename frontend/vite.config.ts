import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

function devApiProxyTarget(env: Record<string, string>): string {
  const host = (env.BACKEND_URL || 'localhost').trim() || 'localhost'
  const port = (env.BACKEND_PORT || '8100').trim() || '8100'
  return `http://${host}:${port}`
}

export default defineConfig(({ mode }) => {
  const envDir = path.resolve(__dirname, '..')
  const env = loadEnv(mode, envDir, '')

  const apiProxy = {
    '/api': {
      target: devApiProxyTarget(env),
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
    base: '/hub',
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
