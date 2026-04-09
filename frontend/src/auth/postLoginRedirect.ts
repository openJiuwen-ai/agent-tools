/** OAuth 回调会回到 /login，Router state 丢失，用 sessionStorage 保存登录后目标路径。 */

export const POST_LOGIN_REDIRECT_KEY = 'marketplace_post_login_redirect'

/**
 * 仅允许站内相对路径，防止异常写入 sessionStorage 后的开放重定向（如 //evil.com）。
 * 规则：必须以 `/` 开头、不能以 `//` 开头、不含反斜杠；`/login` 统一回落到首页避免循环。
 */
export function sanitizePostLoginPath(raw: string | null | undefined, fallback = '/'): string {
  if (raw == null || typeof raw !== 'string') return fallback
  let p = raw.trim()
  if (!p) return fallback
  try {
    p = decodeURIComponent(p)
  } catch {
    return fallback
  }
  p = p.trim()
  if (!p.startsWith('/') || p.startsWith('//')) return fallback
  if (p.includes('\\')) return fallback
  const pathOnly = p.split(/[?#]/, 1)[0] ?? p
  if (pathOnly === '/login' || pathOnly.startsWith('/login/')) return fallback
  return p
}

export function setPostLoginRedirect(path: string): void {
  try {
    const safe = sanitizePostLoginPath(path, '/')
    sessionStorage.setItem(POST_LOGIN_REDIRECT_KEY, safe)
  } catch {
    /* ignore */
  }
}
