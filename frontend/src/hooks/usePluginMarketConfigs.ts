import { useMemo } from 'react'
import { usePluginListQuery, type MarketplacePluginItem, type MarketplacePluginListRequest } from '@/api'
import { resolvePluginIconUrl } from '@/utils/resolvePluginIconUrl'

export interface MarketPlugin {
  assetId: string
  assetType: string
  name: string
  displayName: string
  shortDesc: string
  detailDesc: string
  iconUri: string
  publisherId: string
  publisherName: string
  tags: string[]
  certification: string
  runTime: string
  latestVersion: string
  /** 全部版本号（列表接口 all_versions） */
  allVersions: string[]
  viewCount: number
  installCount: number
  likeCount: number
  reviewCount: number
  averageRating: number
  createTime?: number | null
  updateTime?: number | null
}

export type MarketCatalogKind = 'plugin' | 'skill'

export interface UsePluginMarketConfigsParams {
  page: number
  pageSize: number
  searchKeyword?: string
  runTime?: string
  /** 市场大类：插件（排除 skill）或仅 skill */
  catalogKind?: MarketCatalogKind
  orderBy?: MarketplacePluginListRequest['order_by']
  desc?: boolean
}

export interface UsePluginMarketConfigsReturn {
  marketPlugins: MarketPlugin[]
  total: number
  page: number
  pageSize: number
  loading: boolean
  error: string | null
  refreshMarketPlugins: () => Promise<unknown>
}

function firstString(...candidates: Array<string | null | undefined>): string {
  for (const c of candidates) {
    if (c != null && String(c).length > 0) return String(c)
  }
  return ''
}

function mapPlugin(item: MarketplacePluginItem): MarketPlugin {
  return {
    assetId: item.asset_id,
    assetType: item.asset_type,
    name: item.name,
    displayName: firstString(item.display_name, item.displayName) || item.name,
    shortDesc: firstString(item.short_desc, item.shortDesc),
    detailDesc: firstString(item.detail_desc, item.detailDesc),
    iconUri: resolvePluginIconUrl(item.icon_uri || ''),
    publisherId: item.publisher_id,
    publisherName: item.publisher_name,
    tags: item.tags || [],
    certification: item.certification || '',
    runTime: firstString(item.plugin_type, item.run_time),
    latestVersion: item.latest_version || '',
    allVersions: Array.isArray(item.all_versions) ? item.all_versions : [],
    viewCount: item.view_count,
    installCount: item.install_count,
    likeCount: item.like_count,
    reviewCount: item.review_count,
    averageRating: item.average_rating,
    createTime: item.create_time ?? item.createTime ?? null,
    updateTime: item.update_time ?? item.updateTime ?? null,
  }
}

export function usePluginMarketConfigs(params: UsePluginMarketConfigsParams): UsePluginMarketConfigsReturn {
  const catalog = params.catalogKind ?? 'plugin'
  const query = usePluginListQuery({
    page: params.page,
    page_size: params.pageSize,
    search_keyword: params.searchKeyword || undefined,
    plugin_type: catalog === 'skill' ? 'skill' : params.runTime || undefined,
    plugin_type_exclude: catalog === 'skill' ? undefined : 'skill',
    order_by: params.orderBy ?? 'install_count',
    desc: params.desc ?? true,
  })

  const listPayload = query.data?.data

  const marketPlugins = useMemo(() => {
    return (listPayload?.items ?? []).map(mapPlugin)
  }, [listPayload])

  return {
    marketPlugins,
    total: listPayload?.total ?? 0,
    page: listPayload?.page ?? params.page,
    pageSize: listPayload?.page_size ?? params.pageSize,
    // 仅首次/无缓存时视为整页 loading；后台 refetch（如下载后刷新列表）不应替换整页内容，避免闪白
    loading: query.isLoading,
    error: query.error instanceof Error ? query.error.message : null,
    refreshMarketPlugins: query.refetch,
  }
}
