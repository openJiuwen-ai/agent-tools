/**
 * 将接口返回的对象存储图标 URL 转为开发时可用的地址。
 *
 * - 配置 `VITE_MINIO_BROWSER_ORIGIN` 且与 icon_uri 的 origin 一致时，改走 Vite 同源前缀
 *   `VITE_MINIO_DEV_PROXY_PREFIX`（默认 `/__minio`），由 vite 代理到 `VITE_MINIO_PROXY_TARGET`。
 * - **不能**绕过私有桶的 403：桶需匿名读策略，或仍依赖后端预签名/网关。
 */
const browserOrigin = (import.meta.env.VITE_MINIO_BROWSER_ORIGIN as string | undefined)?.trim()
const proxyPrefix = (
  (import.meta.env.VITE_MINIO_DEV_PROXY_PREFIX as string | undefined)?.trim() || '/__minio'
).replace(/\/$/, '')

export function resolvePluginIconUrl(uri: string | null | undefined): string {
  if (!uri?.trim()) return ''
  const raw = uri.trim()
  if (!browserOrigin) return raw
  try {
    const u = new URL(raw)
    const o = new URL(browserOrigin)
    if (u.origin !== o.origin) return raw
    return `${proxyPrefix}${u.pathname}${u.search}`
  } catch {
    return raw
  }
}
