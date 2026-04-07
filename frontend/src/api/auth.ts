import axios from 'axios'
import { API_CONFIG } from './config'
import type { GitCodeUser } from '@/auth/gitcodeStorage'

/** 与 claw-market 一致：避免 StrictMode 下重复兑换 */
export const GITCODE_OAUTH_PENDING_KEY = '__marketplace_gitcode_oauth_pending'

export function getOAuthGitCodeStartUrl(): string {
  const base = (API_CONFIG.BASE_URL || '/api/v1').replace(/\/$/, '')
  return `${base}/auth/oauth/gitcode/start`
}

export type OAuthSessionData = {
  access_token: string
  token_type?: string
  user: GitCodeUser
}

export async function exchangeGitCodeOAuthSession(sessionId: string): Promise<OAuthSessionData> {
  const base = (API_CONFIG.BASE_URL || '/api/v1').replace(/\/$/, '')
  const url = `${base}/auth/oauth/gitcode/session`
  const { data } = await axios.post<{ code: number; message: string; data: OAuthSessionData }>(
    url,
    { session: sessionId },
    {
      timeout: API_CONFIG.TIMEOUT,
      headers: API_CONFIG.HEADERS,
    },
  )
  if (data.code !== 200 || !data.data?.access_token || !data.data.user) {
    throw new Error(data.message || 'OAuth session exchange failed')
  }
  return data.data
}

export async function fetchGitCodeMe(accessToken: string): Promise<GitCodeUser> {
  const base = (API_CONFIG.BASE_URL || '/api/v1').replace(/\/$/, '')
  const url = `${base}/auth/me`
  const { data } = await axios.get<{ code: number; message: string; data: GitCodeUser }>(url, {
    headers: {
      ...API_CONFIG.HEADERS,
      Authorization: `Bearer ${accessToken}`,
    },
    timeout: API_CONFIG.TIMEOUT,
  })
  if (data.code !== 200 || !data.data?.id) {
    throw new Error(data.message || 'auth/me failed')
  }
  return data.data
}
