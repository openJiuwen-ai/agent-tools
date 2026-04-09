import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from 'react'
import { fetchGitCodeMe } from '@/api/auth'
import {
  clearGitCodeSession,
  getStoredGitCodeToken,
  getStoredGitCodeUser,
  setGitCodeSession,
  type GitCodeUser,
} from './gitcodeStorage'

type GitCodeAuthState = {
  token: string | null
  user: GitCodeUser | null
  isAuthenticated: boolean
  login: (token: string, user: GitCodeUser) => void
  logout: () => void
}

const GitCodeAuthContext = createContext<GitCodeAuthState | null>(null)

export function GitCodeAuthProvider({ children }: { children: ReactNode }) {
  /**
   * 首帧即读 sessionStorage，避免刷新 /profile 等页时先渲染「未登录」、误跳 /login，
   * 再在登录页被当成已登录却未写入 postLoginRedirect 而落到默认「/」市场首页。
   */
  const [token, setToken] = useState<string | null>(() => getStoredGitCodeToken())
  const [user, setUser] = useState<GitCodeUser | null>(() => getStoredGitCodeUser())

  useEffect(() => {
    const t = getStoredGitCodeToken()
    const u = getStoredGitCodeUser()
    setToken(t)
    setUser(u)
    if (!t) return
    void fetchGitCodeMe(t)
      .then(profile => {
        setUser(profile)
        setGitCodeSession(t, profile)
      })
      .catch(() => {
        clearGitCodeSession()
        setToken(null)
        setUser(null)
      })
  }, [])

  const login = useCallback((t: string, u: GitCodeUser) => {
    setGitCodeSession(t, u)
    setToken(t)
    setUser(u)
  }, [])

  const logout = useCallback(() => {
    clearGitCodeSession()
    setToken(null)
    setUser(null)
  }, [])

  const value = useMemo(
    () => ({
      token,
      user,
      isAuthenticated: Boolean(token),
      login,
      logout,
    }),
    [token, user, login, logout],
  )

  return <GitCodeAuthContext.Provider value={value}>{children}</GitCodeAuthContext.Provider>
}

export function useGitCodeAuth(): GitCodeAuthState {
  const ctx = useContext(GitCodeAuthContext)
  if (!ctx) {
    throw new Error('useGitCodeAuth must be used within GitCodeAuthProvider')
  }
  return ctx
}
