export const API_CONFIG = {
  BASE_URL:
    (typeof import.meta !== 'undefined' && (import.meta as unknown as { env?: Record<string, string> }).env?.VITE_API_BASE_URL) ||
    '/api/v1',
  TIMEOUT: 300000,
  HEADERS: {
    'Content-Type': 'application/json',
    Accept: 'application/json',
  },
}

export const setApiBaseUrl = (baseUrl: string) => {
  API_CONFIG.BASE_URL = baseUrl
}

export const API_ENDPOINTS = {
  PLUGINS: {
    LIST: '/plugins',
    /** GET /api/v1/plugins/publish-template — 需 Bearer，返回私有桶模板 zip 预签名 URL */
    PUBLISH_TEMPLATE: '/plugins/publish-template',
    versionDetail: (assetId: string, version: string) =>
      `/plugins/${encodeURIComponent(assetId)}/versions/${encodeURIComponent(version)}`,
  },
  ARTIFACTS: {
    /** GET /api/v1/artifacts/{asset_id} */
    download: (assetId: string) => `/artifacts/${encodeURIComponent(assetId)}`,
  },
} as const
