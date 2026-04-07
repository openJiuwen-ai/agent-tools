import axios from 'axios'
import type { AxiosInstance, InternalAxiosRequestConfig } from 'axios'
import { getStoredGitCodeToken } from '@/auth/gitcodeStorage'
import { API_CONFIG, API_ENDPOINTS } from './config'

let apiClient: AxiosInstance | null = null

/**
 * 与后端公开读接口对齐：列表/版本详情等查询不带 Bearer，减小 token 在日志与中间链路的暴露面。
 * 下载类（/artifacts/...）仅在本地已存 token（登录后）时附带 Authorization。
 */
function shouldAttachGitCodeBearer(config: InternalAxiosRequestConfig): boolean {
  const method = (config.method || 'get').toLowerCase()
  if (method !== 'get') {
    return true
  }
  const raw = config.url || ''
  const path = raw.split('?')[0].replace(/\/$/, '') || '/'
  const listPath = API_ENDPOINTS.PLUGINS.LIST.replace(/\/$/, '') || '/plugins'
  if (path === listPath) {
    return false
  }
  if (/^\/plugins\/[^/]+\/versions\/[^/]+$/.test(path)) {
    return false
  }
  return true
}

export function getApiClient(): AxiosInstance {
  if (!apiClient) {
    apiClient = axios.create({
      baseURL: API_CONFIG.BASE_URL,
      timeout: API_CONFIG.TIMEOUT,
      headers: API_CONFIG.HEADERS,
    })
    apiClient.interceptors.request.use(config => {
      if (!shouldAttachGitCodeBearer(config)) {
        delete config.headers.Authorization
        return config
      }
      const t = getStoredGitCodeToken()
      if (t) {
        config.headers.Authorization = `Bearer ${t}`
      } else {
        delete config.headers.Authorization
      }
      return config
    })
  }
  return apiClient
}
