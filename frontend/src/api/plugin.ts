import axios from 'axios'
import { getApiClient } from './client'
import { API_ENDPOINTS } from './config'

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
  run_time?: string
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
      run_time: request.run_time || undefined,
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
