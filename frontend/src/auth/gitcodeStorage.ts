const TOKEN_KEY = 'marketplace_gitcode_access_token'
const USER_KEY = 'marketplace_gitcode_user'

export type GitCodeUser = {
  id: string
  name: string
  login: string
  avatar_url?: string | null
}

export function getStoredGitCodeToken(): string | null {
  try {
    const t = sessionStorage.getItem(TOKEN_KEY)
    return t && t.trim() ? t.trim() : null
  } catch {
    return null
  }
}

export function getStoredGitCodeUser(): GitCodeUser | null {
  try {
    const raw = sessionStorage.getItem(USER_KEY)
    if (!raw) return null
    const u = JSON.parse(raw) as GitCodeUser
    if (!u || typeof u.id !== 'string') return null
    return u
  } catch {
    return null
  }
}

export function setGitCodeSession(token: string, user: GitCodeUser): void {
  sessionStorage.setItem(TOKEN_KEY, token)
  sessionStorage.setItem(USER_KEY, JSON.stringify(user))
}

export function clearGitCodeSession(): void {
  sessionStorage.removeItem(TOKEN_KEY)
  sessionStorage.removeItem(USER_KEY)
}
