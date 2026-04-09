import axios from 'axios'
import { getStoredGitCodeToken } from '@/auth/gitcodeStorage'
import { getApiClient } from './client'
import { API_CONFIG, API_ENDPOINTS } from './config'

export type MarketplacePluginOrderBy =
  | 'install_count'
  | 'like_count'
  | 'create_time'
  | 'update_time'
  | 'review_count'

export interface MarketplacePluginListRequest {
  page?: number
  page_size?: number
  search_keyword?: string
  /** 与后端 `publisher_id` 一致：筛选指定发布者的插件 */
  publisher_id?: string
  /** 与后端 `asset_id` 一致 */
  asset_id?: string
  /** 与后端 Query 一致：`plugin_type`（如 tools / mcp-stdio / restful-api） */
  plugin_type?: string
  order_by?: MarketplacePluginOrderBy
  desc?: boolean
}

export interface MarketplacePluginItem {
  asset_id: string
  asset_type: string
  name: string
  display_name?: string | null
  /** 部分网关 / 服务可能返回 camelCase */
  displayName?: string | null
  short_desc?: string | null
  shortDesc?: string | null
  detail_desc?: string | null
  detailDesc?: string | null
  icon_uri?: string | null
  publisher_id: string
  publisher_name: string
  tags?: string[] | null
  certification?: string | null
  plugin_type?: string | null
  /** 旧字段名，仅作兼容 */
  run_time?: string | null
  latest_version?: string | null
  view_count: number
  install_count: number
  like_count: number
  review_count: number
  average_rating: number
  create_time?: number | null
  update_time?: number | null
  createTime?: number | null
  updateTime?: number | null
}

export interface MarketplacePluginListData {
  page: number
  page_size: number
  total: number
  items: MarketplacePluginItem[]
}

export interface MarketplacePluginListResponse {
  code: number
  message: string
  data: MarketplacePluginListData
}

/** GET /api/v1/artifacts/{id} 响应 data */
export interface PluginDownloadData {
  download_url: string
  asset_id: string
  name: string
  version: string
  file_size: number
  checksum_sha256: string
}

export interface PluginDownloadResponse {
  code: number
  message: string
  data: PluginDownloadData
}

function downloadErrorMessage(err: unknown, fallback: string): string {
  if (axios.isAxiosError(err)) {
    const payload = err.response?.data as {
      message?: string
      detail?: string | { message?: string }
    }
    if (payload?.message) return String(payload.message)
    const d = payload?.detail
    if (typeof d === 'string') return d
    if (d && typeof d === 'object' && 'message' in d && d.message != null) return String(d.message)
    if (err.message) return err.message
  }
  if (err instanceof Error && err.message) return err.message
  return fallback
}

/** Request download metadata (public URL); server increments install count. */
export async function getPluginArtifactDownload(assetId: string): Promise<PluginDownloadData> {
  const client = getApiClient()
  try {
    const { data } = await client.get<PluginDownloadResponse>(API_ENDPOINTS.ARTIFACTS.download(assetId))
    if (data.code !== 200 || !data.data?.download_url) {
      throw new Error(data.message || 'Download failed')
    }
    return data.data
  } catch (e) {
    throw new Error(downloadErrorMessage(e, 'Download failed'))
  }
}

export class MarketplaceApiError extends Error {
  readonly code?: number

  constructor(message: string, code?: number) {
    super(message)
    this.name = 'MarketplaceApiError'
    this.code = code
  }
}

export async function getPlugins(
  request: MarketplacePluginListRequest = {}
): Promise<MarketplacePluginListResponse> {
  const client = getApiClient()
  const { data } = await client.get<MarketplacePluginListResponse>(API_ENDPOINTS.PLUGINS.LIST, {
    params: {
      page: request.page ?? 1,
      page_size: request.page_size ?? 20,
      search_keyword: request.search_keyword || undefined,
      publisher_id: request.publisher_id || undefined,
      asset_id: request.asset_id || undefined,
      plugin_type: request.plugin_type || undefined,
      order_by: request.order_by ?? 'install_count',
      desc: request.desc ?? true,
    },
  })

  if (data == null || typeof data !== 'object') {
    throw new MarketplaceApiError('插件列表响应无效')
  }
  if (data.code !== 200) {
    throw new MarketplaceApiError(data.message || `插件列表失败（code ${data.code}）`, data.code)
  }
  const body = data.data
  if (body == null || typeof body !== 'object') {
    throw new MarketplaceApiError(data.message || '插件列表 data 为空')
  }
  if (!Array.isArray(body.items)) {
    throw new MarketplaceApiError(data.message || '插件列表缺少 items')
  }

  return data
}

/** GET /api/v1/plugins/{asset_id}/versions/{version} 响应 data */
export interface PluginVersionDetailData {
  asset_id: string
  version: string
  asset_type: string
  plugin_type?: string | null
  name: string
  display_name: string
  short_desc?: string | null
  detail_desc?: string | null
  publisher_id: string
  publisher_name: string
  tags?: string[] | null
  certification?: string | null
  changelog?: string | null
  file_path?: string | null
  icon_uri?: string | null
}

export interface PluginVersionDetailResponse {
  code: number
  message: string
  data: PluginVersionDetailData
}

export async function getPluginVersionDetail(
  assetId: string,
  version: string,
): Promise<PluginVersionDetailData> {
  const client = getApiClient()
  const { data } = await client.get<PluginVersionDetailResponse>(
    API_ENDPOINTS.PLUGINS.versionDetail(assetId, version),
  )
  if (data.code !== 200 || !data.data?.asset_id) {
    throw new MarketplaceApiError(data.message || '插件版本详情失败', data.code)
  }
  return data.data
}

export interface PluginVersionDeleteResult {
  asset_id: string
  version: string
}

export interface PluginVersionDeleteResponse {
  code: number
  message: string
  data: PluginVersionDeleteResult
}

/** GET /api/v1/plugins/publish-template 响应 data */
export interface PluginTemplatePresignData {
  download_url: string
  expires_in: number
  filename: string
}

export interface PluginTemplatePresignResponse {
  code: number
  message: string
  data: PluginTemplatePresignData
}

/**
 * 获取发布页模板 zip 的预签名下载 URL（私有桶对象，需登录 Bearer）。
 */
export async function getPublishTemplatePresigned(): Promise<PluginTemplatePresignData> {
  const token = getStoredGitCodeToken()
  if (!token) {
    throw new Error('请先登录后再下载模板')
  }
  const client = getApiClient()
  try {
    const { data } = await client.get<PluginTemplatePresignResponse>(API_ENDPOINTS.PLUGINS.PUBLISH_TEMPLATE)
    if (data.code !== 200 || !data.data?.download_url) {
      throw new MarketplaceApiError(data.message || '获取模板链接失败', data.code)
    }
    return data.data
  } catch (e) {
    if (e instanceof MarketplaceApiError) throw e
    throw new Error(publishErrorMessage(e, '获取模板链接失败'))
  }
}

/** POST /api/v1/plugins 成功时 data */
export interface PluginPublishResultData {
  plugin_id: string
  name: string
  version: string
  status: string
  published_at: string
  storage_url: string
}

export interface PluginPublishResponse {
  code: number
  message: string
  data: PluginPublishResultData
}

function publishErrorMessage(err: unknown, fallback: string): string {
  if (axios.isAxiosError(err)) {
    const payload = err.response?.data as {
      message?: string
      detail?: string | { message?: string }
    }
    if (payload?.message) return String(payload.message)
    const d = payload?.detail
    if (typeof d === 'string') return d
    if (d && typeof d === 'object' && 'message' in d && d.message != null) return String(d.message)
    if (err.message) return err.message
  }
  if (err instanceof Error && err.message) return err.message
  return fallback
}

/**
 * POST /api/v1/plugins，multipart/form-data。
 * 使用独立 axios 请求，避免带默认 `Content-Type: application/json` 的实例破坏 multipart。
 */
export async function publishPlugin(params: {
  file: File
  checksumSha256Hex: string
  pluginId?: string
  pluginVersion?: string
  versionDesc?: string
  force?: boolean
}): Promise<PluginPublishResultData> {
  const token = getStoredGitCodeToken()
  if (!token) {
    throw new Error('请先登录后再发布插件')
  }
  const base = (API_CONFIG.BASE_URL || '/api/v1').replace(/\/$/, '')
  const form = new FormData()
  form.append('file', params.file)
  if (params.pluginId?.trim()) form.append('plugin_id', params.pluginId.trim())
  if (params.pluginVersion?.trim()) form.append('plugin_version', params.pluginVersion.trim())
  if (params.versionDesc?.trim()) form.append('version_desc', params.versionDesc.trim())
  if (params.force) form.append('force', 'true')

  try {
    const { data } = await axios.post<PluginPublishResponse>(`${base}${API_ENDPOINTS.PLUGINS.LIST}`, form, {
      headers: {
        Authorization: `Bearer ${token}`,
        'X-Checksum-SHA256': params.checksumSha256Hex.toLowerCase(),
      },
      timeout: API_CONFIG.TIMEOUT,
    })
    if (data.code !== 200 || !data.data?.plugin_id) {
      throw new MarketplaceApiError(data.message || '发布失败', data.code)
    }
    return data.data
  } catch (e) {
    if (e instanceof MarketplaceApiError) throw e
    throw new Error(publishErrorMessage(e, '发布失败'))
  }
}

/** DELETE /api/v1/plugins/{asset_id}/versions/all — 需 Bearer，删除资产及全部版本 */
export async function deletePluginAllVersions(assetId: string): Promise<PluginVersionDeleteResult> {
  const client = getApiClient()
  try {
    const { data } = await client.delete<PluginVersionDeleteResponse>(
      API_ENDPOINTS.PLUGINS.versionDetail(assetId, 'all'),
    )
    if (data.code !== 200 || !data.data?.asset_id) {
      throw new MarketplaceApiError(data.message || '删除失败', data.code)
    }
    return data.data
  } catch (e) {
    if (e instanceof MarketplaceApiError) throw e
    throw new Error(downloadErrorMessage(e, '删除失败'))
  }
}
