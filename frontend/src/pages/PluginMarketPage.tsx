import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useNavigate } from 'react-router-dom'
import {
  AlignLeft,
  BarChart3,
  BookOpen,
  Bookmark,
  CalendarClock,
  CalendarPlus,
  Cpu,
  Download,
  Eye,
  Heart,
  MessageCircle,
  RefreshCw,
  ScrollText,
  Tag,
  User,
} from 'lucide-react'
import {
  Button,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  CircularProgress,
  FormControl,
  IconButton,
  InputLabel,
  MenuItem,
  Select,
  Tooltip,
  Typography,
} from '@mui/material'
import { pluginCardTooltipProps, pluginDetailHeaderTooltipProps } from '@/components/Common/pluginCardTooltip'
import { PluginMarkdown } from '@/components/Common/PluginMarkdown'
import { CommonPageLayout, LanguageSwitcher, SearchInput } from '@/components/Common/common-page'
import { UserAccountMenu } from '@/components/Common/UserAccountMenu'
import type { ViewType } from '@/components/Common/common-page'
import { ConfigCard, type ConfigCardTag, type EditingState } from '@/components/Common/common-grid'
import { ConfigTable, type TableColumn } from '@/components/Common/common-table'
import { Empty } from '@/components/Common/Empty'
import axios from 'axios'
import { getPluginArtifactDownload, getPluginVersionDetail } from '@/api/plugin'
import { useGitCodeAuth } from '@/auth/GitCodeAuthContext'
import { setPostLoginRedirect } from '@/auth/postLoginRedirect'
import { usePluginMarketConfigs, type MarketPlugin } from '@/hooks/usePluginMarketConfigs'

function isCanceledRequest(err: unknown): boolean {
  if (axios.isCancel(err)) return true
  if (!axios.isAxiosError(err)) return false
  return err.code === 'ERR_CANCELED' || err.name === 'CanceledError'
}

/**
 * 尽量在同一文档内触发下载，避免 `target="_blank"` 先打开空白新标签造成的「闪白」体感。
 * 顺序：CORS 允许时 fetch+Blob；否则隐藏 iframe；最后才新开标签。
 */
async function triggerPluginFileDownload(url: string, filename: string): Promise<void> {
  try {
    const res = await fetch(url, { mode: 'cors', credentials: 'omit' })
    if (!res.ok) throw new Error(`HTTP ${res.status}`)
    const blob = await res.blob()
    const objUrl = URL.createObjectURL(blob)
    try {
      const a = document.createElement('a')
      a.href = objUrl
      a.download = filename
      a.rel = 'noopener'
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
    } finally {
      URL.revokeObjectURL(objUrl)
    }
    return
  } catch {
    /* 存储未配 CORS 等，再走 iframe / 新窗口 */
  }

  try {
    const iframe = document.createElement('iframe')
    iframe.setAttribute('aria-hidden', 'true')
    iframe.style.cssText = 'position:fixed;left:-9999px;top:0;width:1px;height:1px;opacity:0;border:0'
    iframe.src = url
    document.body.appendChild(iframe)
    window.setTimeout(() => {
      try {
        iframe.remove()
      } catch {
        /* ignore */
      }
    }, 120_000)
    return
  } catch {
    /* ignore */
  }

  const a = document.createElement('a')
  a.href = url
  a.target = '_blank'
  a.rel = 'noopener noreferrer'
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
}

const editingState: EditingState = {
  id: null,
  field: null,
  value: '',
  isEditing: false,
}

const PAGE_SIZE_OPTIONS = [20, 60, 100]

/** 列表/卡片简介展示字符上限（Unicode 标量），超出显示 … */
const PLUGIN_INTRO_DISPLAY_MAX = 50

function truncatePluginIntro(
  text: string,
  maxChars: number,
): { display: string; full: string; truncated: boolean } {
  const full = text
  const chars = [...full]
  if (chars.length <= maxChars) {
    return { display: full, full, truncated: false }
  }
  return { display: `${chars.slice(0, maxChars).join('')}...`, full, truncated: true }
}

/** 卡片上：类型 / 版本 / 标签 最多展示条数，超出部分收入 +N */
const PLUGIN_CARD_TAG_MAX = 3

type PluginCardTagHints = {
  runtimeType: string
  version: string
  tag: string
  overflowIntro: string
}

/** 构建插件卡片标签：类型(蓝)、版本(绿)、用户 tag(轮换色)；超过 3 条则截断并追加 +N */
function buildPluginCardTags(plugin: MarketPlugin, runtimeLabel: string, hints: PluginCardTagHints): ConfigCardTag[] {
  const items: ConfigCardTag[] = [
    { label: runtimeLabel, bgColor: '#DBEAFE', color: '#1D4ED8', tooltip: hints.runtimeType },
  ]
  if (plugin.latestVersion) {
    items.push({
      label: `v${plugin.latestVersion}`,
      bgColor: '#D1FAE5',
      color: '#047857',
      tooltip: hints.version,
    })
  }
  const tagBg = ['#FEF3C7', '#EDE9FE', '#FCE7F3'] as const
  const tagFg = ['#B45309', '#5B21B6', '#BE185D'] as const
  const extraTags = plugin.tags ?? []
  extraTags.forEach((label, i) => {
    const idx = i % tagBg.length
    items.push({
      label,
      bgColor: tagBg[idx],
      color: tagFg[idx],
      tooltip: hints.tag,
    })
  })
  if (items.length <= PLUGIN_CARD_TAG_MAX) return items
  const hidden = items.slice(PLUGIN_CARD_TAG_MAX)
  const hiddenLine = hidden.map(t => t.label).join(' · ')
  return [
    ...items.slice(0, PLUGIN_CARD_TAG_MAX),
    {
      label: `+${hidden.length}`,
      bgColor: '#E5E7EB',
      color: '#4B5563',
      tooltip: `${hints.overflowIntro}: ${hiddenLine}`,
    },
  ]
}

function formatPluginDateTime(ts: number | null | undefined, locale: string): string {
  if (ts == null || ts === 0) return '-'
  const ms = ts > 1_000_000_000_000 ? ts : ts * 1000
  return new Date(ms).toLocaleString(locale.startsWith('zh') ? 'zh-CN' : 'en-US', {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  })
}

/**
 * 浅色底 + 同色系较深字色，突出字母且整体柔和。
 * 每项：淡背景渐变 / 纯色底 + 对应 hue 的 700/800 字色
 */
const AVATAR_PALETTE = [
  'border border-violet-200/90 bg-gradient-to-br from-violet-50 to-violet-100 text-violet-500',
  'border border-sky-200/90 bg-gradient-to-br from-sky-50 to-sky-100 text-sky-500',
  'border border-emerald-200/90 bg-gradient-to-br from-emerald-50 to-emerald-100 text-emerald-500',
  'border border-amber-200/90 bg-gradient-to-br from-amber-50 to-amber-100 text-amber-600',
  'border border-rose-200/90 bg-gradient-to-br from-rose-50 to-rose-100 text-rose-500',
  'border border-cyan-200/90 bg-gradient-to-br from-cyan-50 to-cyan-100 text-cyan-500',
  'border border-lime-200/90 bg-gradient-to-br from-lime-50 to-lime-100 text-lime-600',
  'border border-indigo-200/90 bg-gradient-to-br from-indigo-50 to-indigo-100 text-indigo-500',
  'border border-fuchsia-200/90 bg-gradient-to-br from-fuchsia-50 to-fuchsia-100 text-fuchsia-500',
  'border border-blue-200/90 bg-gradient-to-br from-blue-50 to-blue-100 text-blue-500',
]
const AVATAR_FALLBACK_DEFAULT =
  'border border-slate-200/90 bg-gradient-to-br from-slate-100 to-slate-200 text-slate-500'

function getPluginAvatarChar(displayName: string): string {
  const trimmed = displayName.trim()
  if (!trimmed) return ''
  const first = [...trimmed][0]
  if (!first) return ''
  if (/^[a-z]$/i.test(first)) return first.toUpperCase()
  return first
}

function paletteIndexForChar(ch: string): number {
  let h = 0
  for (let i = 0; i < ch.length; i += 1) h = (h * 31 + ch.charCodeAt(i)) | 0
  return Math.abs(h) % AVATAR_PALETTE.length
}

function PluginAvatarFallback({ label, className }: { label: string; className?: string }) {
  const ch = getPluginAvatarChar(label)
  const display = ch || '?'
  const paletteClass =
    ch === '' ? AVATAR_FALLBACK_DEFAULT : AVATAR_PALETTE[paletteIndexForChar(ch)] ?? AVATAR_FALLBACK_DEFAULT
  return (
    <span
      className={`flex shrink-0 items-center justify-center rounded-lg font-bold leading-none select-none ${paletteClass} ${className ?? ''}`}
      aria-hidden
    >
      {display}
    </span>
  )
}

function isIconUrl(icon: string | undefined): boolean {
  if (typeof icon !== 'string' || !icon.trim()) return false
  const t = icon.trim()
  if (t.startsWith('http://') || t.startsWith('https://')) return true
  // 单斜杠站内路径；排除 //host 协议相对 URL
  if (t.startsWith('/') && !t.startsWith('//')) return true
  return t.includes('.')
}

type PluginMarketIconSize = 'card' | 'table' | 'dialog'

const ICON_SIZE_CLASS: Record<PluginMarketIconSize, string> = {
  card: 'h-12 w-12 min-h-12 min-w-12 text-xl',
  table: 'h-10 w-10 min-h-10 min-w-10 text-lg',
  dialog: 'h-12 w-12 min-h-12 min-w-12 text-2xl',
}

function PluginIconImage({
  src,
  label,
  size,
}: {
  src: string
  label: string
  size: PluginMarketIconSize
}) {
  const [failed, setFailed] = useState(false)
  const dim = ICON_SIZE_CLASS[size]
  const imgWrap =
    size === 'dialog'
      ? 'border border-slate-200/90 bg-gradient-to-br from-slate-50 to-slate-100'
      : 'border border-blue-200/75 bg-gradient-to-r from-blue-50/90 to-indigo-50/90'

  if (failed) {
    return <PluginAvatarFallback label={label} className={dim} />
  }
  return (
    <div className={`${dim} overflow-hidden rounded-lg ${imgWrap} flex items-center justify-center`}>
      <img src={src} alt="" className="h-full w-full object-cover" onError={() => setFailed(true)} />
    </div>
  )
}

/** 网格 / 列表 / 详情：统一图标；无图或加载失败时显示名称首字 + 多色底 */
/** 详情弹窗信息区：标签列表，最多 3 个，多出的收入 +N；颜色轮换 */
function DetailPluginTags({ tags }: { tags: string[] }) {
  const list = tags ?? []
  const MAX = 3
  const tagBg = ['#FEF3C7', '#EDE9FE', '#FCE7F3'] as const
  const tagFg = ['#B45309', '#5B21B6', '#BE185D'] as const
  if (list.length === 0) return null
  const visible = list.slice(0, MAX)
  const hidden = list.slice(MAX)
  return (
    <div className="flex flex-wrap items-center gap-1 min-w-0">
      {visible.map((tag, i) => (
        <span
          key={`${tag}-${i}`}
          className="shrink-0 px-2 py-0.5 rounded-md text-[11px] font-medium border border-black/5"
          style={{ backgroundColor: tagBg[i % tagBg.length], color: tagFg[i % tagFg.length] }}
        >
          {tag}
        </span>
      ))}
      {hidden.length > 0 && (
        <Tooltip {...pluginCardTooltipProps} title={hidden.join(' · ')}>
          <span className="shrink-0 px-2 py-0.5 rounded-md text-[11px] font-medium bg-gray-200 text-gray-700 border border-gray-300/80 cursor-default">
            +{hidden.length}
          </span>
        </Tooltip>
      )}
    </div>
  )
}

function PluginMarketUserMenu() {
  const { t } = useTranslation()
  const navigate = useNavigate()
  const { user, logout } = useGitCodeAuth()
  if (!user) return null
  return (
    <UserAccountMenu
      primaryLabel={user.name || user.login}
      title={user.name || user.login}
      items={[
        { id: 'profile', label: t('profile.toolbarLink'), onClick: () => navigate('/profile') },
        { id: 'logout', label: t('auth.toolbar.logout'), onClick: () => logout() },
      ]}
    />
  )
}

function PluginMarketIcon({
  plugin,
  size,
}: {
  plugin: Pick<MarketPlugin, 'iconUri' | 'displayName'>
  size: PluginMarketIconSize
}) {
  const dim = ICON_SIZE_CLASS[size]
  if (!isIconUrl(plugin.iconUri)) {
    return <PluginAvatarFallback label={plugin.displayName} className={dim} />
  }
  return <PluginIconImage src={plugin.iconUri!} label={plugin.displayName} size={size} />
}

export default function PluginMarketPage() {
  const { t, i18n } = useTranslation()
  const navigate = useNavigate()
  const { user, isAuthenticated } = useGitCodeAuth()
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid')
  const viewType: ViewType = viewMode === 'grid' ? 'grid' : 'table'
  const [searchInput, setSearchInput] = useState('')
  const [searchKeyword, setSearchKeyword] = useState('')
  const [runtimeFilter, setRuntimeFilter] = useState<string>('all')
  const [marketCatalogTab, setMarketCatalogTab] = useState<'plugin' | 'skill'>('skill')
  const [currentPage, setCurrentPage] = useState(1)
  const [pageSize, setPageSize] = useState(20)
  const [selectedPlugin, setSelectedPlugin] = useState<MarketPlugin | null>(null)
  const [detailDialogOpen, setDetailDialogOpen] = useState(false)
  const [downloadingAssetId, setDownloadingAssetId] = useState<string | null>(null)
  const downloadLockRef = useRef(false)
  /** 详情弹窗内：指定要下载的版本（列表接口 `all_versions`） */
  const [detailDownloadVersion, setDetailDownloadVersion] = useState('')
  const [detailChangelog, setDetailChangelog] = useState<string | null>(null)
  const [detailChangelogLoading, setDetailChangelogLoading] = useState(false)
  const [detailChangelogError, setDetailChangelogError] = useState<string | null>(null)

  useEffect(() => {
    const timer = window.setTimeout(() => {
      setSearchKeyword(searchInput.trim())
      setCurrentPage(1)
    }, 300)
    return () => window.clearTimeout(timer)
  }, [searchInput])

  useEffect(() => {
    setCurrentPage(1)
  }, [runtimeFilter])

  useEffect(() => {
    setCurrentPage(1)
  }, [marketCatalogTab])

  const { marketPlugins, total, page, pageSize: serverPageSize, loading, error, refreshMarketPlugins } =
    usePluginMarketConfigs({
      page: currentPage,
      pageSize,
      searchKeyword,
      runTime: runtimeFilter === 'all' ? undefined : runtimeFilter,
      catalogKind: marketCatalogTab,
    })

  const runtimeOptions = useMemo(
    () => [
      { value: 'tools', label: t('plugins.runtime.tools') },
      { value: 'mcp-stdio', label: t('plugins.runtime.mcpStdio') },
      { value: 'restful-api', label: t('plugins.runtime.restfulApi') },
    ],
    [t]
  )

  const getRuntimeText = (runTime: string) => {
    switch (runTime) {
      case 'tools':
        return t('plugins.runtime.tools')
      case 'mcp-stdio':
        return t('plugins.runtime.mcpStdio')
      case 'restful-api':
        return t('plugins.runtime.restfulApi')
      case 'skill':
        return t('plugins.runtime.skill')
      default:
        return t('plugins.runtime.unknown', { type: runTime || '-' })
    }
  }

  const defaultDownloadVersion = useCallback((plugin: MarketPlugin) => {
    const versions = plugin.allVersions
    const latest = plugin.latestVersion?.trim()
    if (latest && versions.includes(latest)) return latest
    if (versions.length) return versions[versions.length - 1]
    return latest || ''
  }, [])

  const effectiveDetailVersion = useMemo(() => {
    if (!selectedPlugin) return ''
    return (detailDownloadVersion || defaultDownloadVersion(selectedPlugin)).trim()
  }, [selectedPlugin, detailDownloadVersion, defaultDownloadVersion])

  useEffect(() => {
    if (!detailDialogOpen || !selectedPlugin || !effectiveDetailVersion) {
      setDetailChangelog(null)
      setDetailChangelogLoading(false)
      setDetailChangelogError(null)
      return
    }
    const ac = new AbortController()
    setDetailChangelogLoading(true)
    setDetailChangelogError(null)
    setDetailChangelog(null)
    void getPluginVersionDetail(selectedPlugin.assetId, effectiveDetailVersion, { signal: ac.signal })
      .then(data => {
        const raw = data.changelog?.trim()
        setDetailChangelog(raw && raw.length > 0 ? raw : null)
        setDetailChangelogLoading(false)
      })
      .catch((err: unknown) => {
        if (isCanceledRequest(err)) {
          return
        }
        setDetailChangelogError(err instanceof Error ? err.message : t('plugins.detail.changelogLoadFailed'))
        setDetailChangelogLoading(false)
      })
    return () => ac.abort()
  }, [detailDialogOpen, selectedPlugin, effectiveDetailVersion, t])

  const handleViewPlugin = (plugin: MarketPlugin) => {
    setSelectedPlugin(plugin)
    setDetailDownloadVersion(defaultDownloadVersion(plugin))
    setDetailDialogOpen(true)
  }

  const handleRefresh = async () => {
    await refreshMarketPlugins()
  }

  const handlePublishClick = useCallback(() => {
    const path = marketCatalogTab === 'skill' ? '/profile/publish?kind=skill' : '/profile/publish'
    if (isAuthenticated) {
      navigate(path)
      return
    }
    setPostLoginRedirect(path)
    navigate('/login')
  }, [isAuthenticated, marketCatalogTab, navigate])

  const handleFavoriteComingSoon = () => {
    window.alert(t('plugins.actions.favoritePending'))
  }

  const handleDownloadPlugin = useCallback(
    async (plugin: MarketPlugin, version?: string) => {
      if (downloadLockRef.current) return
      downloadLockRef.current = true
      setDownloadingAssetId(plugin.assetId)
      try {
        const meta = await getPluginArtifactDownload(plugin.assetId, version)
        const baseName = meta.name.trim() || plugin.displayName.trim() || plugin.assetId || 'plugin'
        const safeName = baseName.replace(/\s+/g, '-')
        const filename = `${safeName}_${meta.version}.zip`
        await triggerPluginFileDownload(meta.download_url, filename)
        // 与浏览器下载起始错开一帧，减少与列表重绘叠在一起造成的视觉闪烁
        requestAnimationFrame(() => {
          requestAnimationFrame(() => {
            void refreshMarketPlugins()
          })
        })
      } catch {
        window.alert(t('plugins.actions.downloadFailed'))
      } finally {
        downloadLockRef.current = false
        setDownloadingAssetId(null)
      }
    },
    [refreshMarketPlugins, t],
  )

  const gridView = useMemo(() => {
    if (marketPlugins.length === 0)
      return (
        <Empty
          searchTerm={searchKeyword}
          type="plugins"
          customTitle={marketCatalogTab === 'skill' ? t('plugins.noMatchingSkill') : undefined}
          customDescription={marketCatalogTab === 'skill' ? t('plugins.noMatchingSkillDescription') : undefined}
        />
      )

    return (
      <div className="grid grid-cols-1 md:grid-cols-3 xl:grid-cols-4 2xl:grid-cols-5 gap-3">
        {marketPlugins.map(plugin => {
          const intro = truncatePluginIntro(plugin.shortDesc || t('plugins.noDescription'), PLUGIN_INTRO_DISPLAY_MAX)
          return (
          <div key={plugin.assetId} className="w-full">
            <ConfigCard
              id={plugin.assetId}
              className="min-h-[200px]"
              icon={<PluginMarketIcon plugin={plugin} size="card" />}
              iconBgColor="bg-transparent"
              iconTextColor=""
              title={plugin.displayName}
              titleExtra={
                <div className="ml-auto flex items-center gap-1 max-w-[45%]">
                  <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-gray-100 text-gray-700 text-[11px] min-w-0">
                    <User className="w-3 h-3 shrink-0" />
                    <span className="truncate">{plugin.publisherName}</span>
                  </span>
                </div>
              }
              description={intro.display}
              descriptionTitle={intro.truncated ? intro.full : undefined}
              tags={buildPluginCardTags(plugin, getRuntimeText(plugin.runTime), {
                runtimeType: t('plugins.cardTags.runtimeType'),
                version: t('plugins.cardTags.version'),
                tag: t('plugins.cardTags.tag'),
                overflowIntro: t('plugins.cardTags.overflowIntro'),
              })}
              maxVisibleTags={4}
              editingState={editingState}
              actions={[]}
              onClick={() => handleViewPlugin(plugin)}
              footer={
                <div className="flex items-center justify-between w-full text-xs text-[#4B5563] gap-2">
                  <div className="flex items-center gap-1.5 min-w-0">
                    <Tooltip {...pluginCardTooltipProps} title={t('plugins.detail.viewCount')}>
                      <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded bg-[#f0f9ff] text-[#0369a1] tabular-nums">
                        <Eye className="w-3 h-3" />
                        {plugin.viewCount}
                      </span>
                    </Tooltip>
                    <Tooltip {...pluginCardTooltipProps} title={t('plugins.detail.installCount')}>
                      <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded bg-[#eef2ff] text-[#4338ca] tabular-nums">
                        <Download className="w-3 h-3" />
                        {plugin.installCount}
                      </span>
                    </Tooltip>
                    <Tooltip {...pluginCardTooltipProps} title={t('plugins.detail.likeCount')}>
                      <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded bg-[#fff1f2] text-[#be123c] tabular-nums">
                        <Heart className="w-3 h-3" />
                        {plugin.likeCount}
                      </span>
                    </Tooltip>
                    <Tooltip {...pluginCardTooltipProps} title={t('plugins.detail.reviewCount')}>
                      <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded bg-[#ecfeff] text-[#0e7490] tabular-nums">
                        <MessageCircle className="w-3 h-3" />
                        {plugin.reviewCount}
                      </span>
                    </Tooltip>
                  </div>
                  <div className="flex items-center gap-2">
                    <button
                      type="button"
                      disabled={downloadingAssetId === plugin.assetId}
                      onClick={e => {
                        e.stopPropagation()
                        void handleDownloadPlugin(plugin)
                      }}
                      className="text-xs flex items-center gap-1 text-[#4B5563] hover:text-[#1f2937] transition-colors disabled:opacity-50 disabled:pointer-events-none"
                      title={t('plugins.actions.download')}
                    >
                      <Download className="w-3 h-3" />
                      {t('plugins.actions.download')}
                    </button>
                    <button
                      onClick={e => {
                        e.stopPropagation()
                        handleViewPlugin(plugin)
                      }}
                      className="text-xs flex items-center gap-1 text-[#4B5563] hover:text-[#1f2937] transition-colors"
                      title={t('plugins.actions.view')}
                    >
                      <Eye className="w-3 h-3" />
                      {t('plugins.actions.view')}
                    </button>
                  </div>
                </div>
              }
            />
          </div>
          )
        })}
      </div>
    )
  }, [marketPlugins, searchKeyword, marketCatalogTab, t, handleDownloadPlugin, downloadingAssetId])

  const tableColumns: TableColumn<MarketPlugin>[] = useMemo(
    () => [
      {
        key: 'index',
        title: '序号',
        width: 68,
        align: 'center',
        render: ({ rowIndex }) => <span className="tabular-nums text-gray-600">{rowIndex + 1}</span>,
      },
      {
        key: 'plugin',
        title: t('plugins.tableView.columns.plugin'),
        dataIndex: 'displayName',
        align: 'center',
        width: 280,
        render: ({ row }) => {
          return (
          <div
            className="relative flex w-full items-center justify-center cursor-pointer"
            onClick={() => handleViewPlugin(row)}
          >
            <div className="inline-flex max-w-full items-center justify-center gap-3">
              <PluginMarketIcon plugin={row} size="table" />
              <div className="min-w-0 max-w-[220px] text-center">
                <div className="font-semibold text-gray-900 truncate">{row.displayName}</div>
              </div>
            </div>
          </div>
          )
        },
      },
      {
        key: 'summary',
        title: t('plugins.tableView.columns.summary'),
        dataIndex: 'shortDesc',
        align: 'center',
        width: 240,
        render: ({ row }) => {
          const raw = row.shortDesc || t('plugins.noDescription')
          const intro = truncatePluginIntro(raw, PLUGIN_INTRO_DISPLAY_MAX)
          const cell = (
            <button
              type="button"
              onClick={() => handleViewPlugin(row)}
              className="max-w-[220px] truncate text-center text-sm text-gray-600 hover:text-gray-800"
            >
              {intro.display}
            </button>
          )
          return intro.truncated ? (
            <Tooltip {...pluginCardTooltipProps} title={intro.full}>
              <span className="inline-flex max-w-full justify-center">{cell}</span>
            </Tooltip>
          ) : (
            cell
          )
        },
      },
      {
        key: 'runtime',
        title: t('plugins.tableView.columns.type'),
        dataIndex: 'runTime',
        align: 'center',
        width: 160,
        render: ({ row }) => (
          <button onClick={() => handleViewPlugin(row)} className="px-2 py-1 text-xs font-medium bg-blue-100 text-blue-700 rounded-full">
            {getRuntimeText(row.runTime)}
          </button>
        ),
      },
      {
        key: 'publisher',
        title: t('plugins.tableView.columns.publisher'),
        dataIndex: 'publisherName',
        align: 'center',
        width: 180,
        render: ({ row }) => (
          <button onClick={() => handleViewPlugin(row)} className="truncate text-gray-800 hover:text-gray-900">
            {row.publisherName || '-'}
          </button>
        ),
      },
      {
        key: 'version',
        title: t('plugins.tableView.columns.version'),
        dataIndex: 'latestVersion',
        align: 'center',
        width: 100,
        render: ({ row }) => (
          <button onClick={() => handleViewPlugin(row)} className="tabular-nums text-gray-800 hover:text-gray-900">
            {row.latestVersion || '-'}
          </button>
        ),
      },
      {
        key: 'viewCount',
        title: t('plugins.tableView.columns.viewCount'),
        dataIndex: 'viewCount',
        width: 72,
        align: 'center',
        render: ({ row }) => (
          <button onClick={() => handleViewPlugin(row)} className="tabular-nums hover:text-gray-900">
            {row.viewCount}
          </button>
        ),
      },
      {
        key: 'installCount',
        title: t('plugins.tableView.columns.installCount'),
        dataIndex: 'installCount',
        width: 72,
        align: 'center',
        render: ({ row }) => (
          <button onClick={() => handleViewPlugin(row)} className="tabular-nums hover:text-gray-900">
            {row.installCount}
          </button>
        ),
      },
      {
        key: 'likeCount',
        title: t('plugins.tableView.columns.likeCount'),
        dataIndex: 'likeCount',
        width: 72,
        align: 'center',
        render: ({ row }) => (
          <button onClick={() => handleViewPlugin(row)} className="tabular-nums hover:text-gray-900">
            {row.likeCount}
          </button>
        ),
      },
      {
        key: 'downloadAction',
        title: t('plugins.actions.download'),
        type: 'operate',
        align: 'center',
        width: 80,
        render: ({ row }) => (
          <div className="flex items-center justify-center">
            <Tooltip title={t('plugins.actions.download')}>
              <span>
                <IconButton
                  size="small"
                  disabled={downloadingAssetId === row.assetId}
                  onClick={() => void handleDownloadPlugin(row)}
                  sx={{ color: '#777777' }}
                >
                  <Download className="w-4 h-4" />
                </IconButton>
              </span>
            </Tooltip>
          </div>
        ),
      },
      {
        key: 'actions',
        title: t('plugins.actions.view'),
        type: 'operate',
        align: 'center',
        width: 80,
        render: ({ row }) => (
          <div className="flex items-center justify-center">
            <Tooltip title={t('plugins.actions.view')}>
              <IconButton size="small" onClick={() => handleViewPlugin(row)} sx={{ color: '#777777' }}>
                <Eye className="w-4 h-4" />
              </IconButton>
            </Tooltip>
          </div>
        ),
      },
    ],
    [t, handleDownloadPlugin, downloadingAssetId]
  )

  const tableView = useMemo(
    () => (
      <ConfigTable
        tableData={{ columns: tableColumns, rows: marketPlugins }}
        loading={loading}
        size="small"
        stickyHeader
        emptyState={
          <Empty
            searchTerm={searchKeyword}
            type="plugins"
            customTitle={marketCatalogTab === 'skill' ? t('plugins.noMatchingSkill') : undefined}
            customDescription={marketCatalogTab === 'skill' ? t('plugins.noMatchingSkillDescription') : undefined}
          />
        }
      />
    ),
    [tableColumns, marketPlugins, loading, searchKeyword, marketCatalogTab, t]
  )

  const toolbarLeft = useMemo(
    () => (
      <>
        <SearchInput searchTerm={searchInput} placeholder={t('plugins.searchPlaceholder')} onChange={setSearchInput} />
        {marketCatalogTab === 'plugin' ? (
          <select
            value={runtimeFilter}
            onChange={e => setRuntimeFilter(e.target.value)}
            className="h-10 px-3 bg-white/95 border border-[#d7e2f6] text-[#1f2937] rounded-lg text-sm shadow-[0_1px_3px_rgba(15,23,42,0.06)] focus:outline-none focus:border-[#3b82f6] focus:ring-2 focus:ring-[#bfdbfe] transition-colors"
          >
            <option value="all">{t('plugins.filters.allCategories')}</option>
            {runtimeOptions.map(option => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        ) : null}
      </>
    ),
    [marketCatalogTab, runtimeFilter, runtimeOptions, searchInput, t]
  )

  const marketTabs = useMemo(
    () => [
      { key: 'skill', label: t('plugins.marketTab.skill') },
      { key: 'plugin', label: t('plugins.marketTab.plugin') },
    ],
    [t],
  )

  const toolbarRight = useMemo(
    () => (
      <div className="flex items-center gap-2">
        <button
          type="button"
          onClick={handlePublishClick}
          className="h-10 shrink-0 px-3 rounded-lg text-sm font-medium text-white bg-[#0891b2] hover:bg-[#0e7490] shadow-[0_1px_3px_rgba(15,23,42,0.08)] transition-colors"
        >
          {marketCatalogTab === 'skill' ? t('profile.publishSkill') : t('profile.publishPlugin')}
        </button>
        {isAuthenticated && user ? (
          <PluginMarketUserMenu />
        ) : (
          <button
            type="button"
            onClick={() => navigate('/login')}
            className="h-10 px-3 text-sm font-medium text-[#0369a1] hover:text-[#0c4a6e] underline-offset-2 hover:underline"
          >
            {t('auth.toolbar.login')}
          </button>
        )}
        <LanguageSwitcher />
        <button
          onClick={handleRefresh}
          disabled={loading}
          className="h-10 px-3 bg-white/95 border border-[#d7e2f6] text-[#1f2937] rounded-lg text-sm font-medium shadow-[0_1px_3px_rgba(15,23,42,0.06)] hover:bg-[#f8fbff] hover:border-[#bfdbfe] transition-colors flex items-center space-x-2"
        >
          {loading ? <span className="inline-flex"><RefreshCw className="w-4 h-4 animate-spin" /></span> : <RefreshCw className="w-4 h-4" />}
          <span>{t('plugins.actions.refresh')}</span>
        </button>
      </div>
    ),
    [loading, t, isAuthenticated, user, navigate, handlePublishClick, marketCatalogTab]
  )

  return (
    <div className="relative flex min-h-0 flex-1 flex-col overflow-hidden bg-gradient-to-br from-[#f8fbff] via-[#f6faff] to-[#eef4ff]">
      <div className="pointer-events-none absolute -top-20 -right-20 h-72 w-72 rounded-full bg-blue-100/50 blur-3xl" />
      <div className="pointer-events-none absolute -bottom-28 -left-24 h-72 w-72 rounded-full bg-indigo-100/40 blur-3xl" />
      <CommonPageLayout
        className="min-h-0 flex-1"
        title={t('plugins.marketTitle')}
        tabs={marketTabs}
        tabsAriaLabel={t('plugins.segmentedTabsAria')}
        tabKey={marketCatalogTab}
        onTabChange={key => setMarketCatalogTab(key === 'skill' ? 'skill' : 'plugin')}
        viewType={viewType}
        onViewTypeChange={type => setViewMode(type === 'grid' ? 'grid' : 'list')}
        pager={{
          total,
          currentPage: page,
          pageSize: serverPageSize,
          pageSizeOptions: PAGE_SIZE_OPTIONS,
        }}
        onPagerChange={(nextPage, nextPageSize) => {
          setCurrentPage(nextPage)
          setPageSize(nextPageSize)
        }}
        loading={loading}
        error={error}
        gridView={gridView}
        tableView={tableView}
        toolbarLeft={toolbarLeft}
        toolbarRight={toolbarRight}
      />

      <Dialog
        open={detailDialogOpen}
        onClose={() => setDetailDialogOpen(false)}
        maxWidth="md"
        fullWidth
        slotProps={{ paper: { sx: { borderRadius: 3 } } }}
      >
        {selectedPlugin && (
          <>
            <DialogTitle className="flex items-start justify-between gap-3">
              <div className="flex items-center space-x-3 min-w-0">
                <PluginMarketIcon plugin={selectedPlugin} size="dialog" />
                <div className="min-w-0">
                  <div className="flex flex-wrap items-center gap-x-2 gap-y-1 min-w-0">
                    <Typography variant="h6" className="truncate text-[#111827] font-black min-w-0">
                      {selectedPlugin.displayName}
                    </Typography>
                    {selectedPlugin.latestVersion ? (
                      <span className="shrink-0 px-2 py-0.5 rounded-md text-xs font-semibold bg-indigo-50 text-indigo-700 border border-indigo-100/80">
                        v{selectedPlugin.latestVersion}
                      </span>
                    ) : null}
                  </div>
                  <Typography variant="caption" color="text.secondary" className="block mt-0.5 truncate">
                    {t('plugins.detail.publisher')}: {selectedPlugin.publisherName || '-'}
                  </Typography>
                </div>
              </div>
              <div className="flex items-center gap-2 shrink-0">
                <Tooltip {...pluginDetailHeaderTooltipProps} title={t('plugins.detail.ratingTooltip')}>
                  <span className="inline-flex items-center gap-1 rounded-full bg-amber-50 border border-amber-100 px-2 py-1 text-amber-700 text-xs font-semibold cursor-default">
                    <BarChart3 className="w-3.5 h-3.5 text-amber-500" />
                    {selectedPlugin.averageRating}
                  </span>
                </Tooltip>
                <Tooltip {...pluginDetailHeaderTooltipProps} title={t('plugins.actions.favorite')}>
                  <IconButton
                    size="small"
                    onClick={handleFavoriteComingSoon}
                    sx={{ color: '#64748b' }}
                  >
                    <Bookmark className="w-4 h-4" />
                  </IconButton>
                </Tooltip>
              </div>
            </DialogTitle>
            <DialogContent>
              <div className="space-y-5">
                <div>
                  <div className="flex items-center gap-2 min-w-0">
                    <div className="flex items-center gap-1.5 shrink-0">
                      <AlignLeft className="w-4 h-4 shrink-0 text-slate-600" aria-hidden />
                      <Typography variant="subtitle1" component="span" className="font-bold text-gray-900">
                        {t('plugins.detail.summary')}:
                      </Typography>
                    </div>
                    <div className="flex min-w-0 flex-1 items-center">
                      {selectedPlugin.shortDesc ? (
                        <PluginMarkdown
                          source={selectedPlugin.shortDesc}
                          className="prose prose-sm prose-neutral max-w-none flex-1 min-w-0 text-gray-900 prose-p:my-0 prose-headings:scroll-mt-2 [&_p]:leading-snug [&_p]:text-[0.9375rem]"
                        />
                      ) : (
                        <Typography variant="body2">{t('plugins.noDescription')}</Typography>
                      )}
                    </div>
                  </div>
                </div>
                <div>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                    <div className="rounded-lg border border-[#DCEEFE] bg-[#F3FAFF] px-3 py-3 min-h-[108px]">
                      <div className="flex flex-col items-center text-center">
                        <Eye className="w-4 h-4 text-sky-600 mb-2" />
                        <div className="text-sky-700 tabular-nums font-extrabold text-lg leading-7">
                        {selectedPlugin.viewCount}
                        </div>
                        <div className="text-[11px] text-sky-600 mt-2">{t('plugins.detail.viewCount')}</div>
                      </div>
                    </div>
                    <div className="rounded-lg border border-[#E0E7FF] bg-[#F4F6FF] px-3 py-3 min-h-[108px]">
                      <div className="flex flex-col items-center text-center">
                        <Download className="w-4 h-4 text-indigo-600 mb-2" />
                        <div className="text-indigo-700 tabular-nums font-extrabold text-lg leading-7">
                        {selectedPlugin.installCount}
                        </div>
                        <div className="text-[11px] text-indigo-600 mt-2">{t('plugins.detail.installCount')}</div>
                      </div>
                    </div>
                    <div className="rounded-lg border border-[#FFE2EA] bg-[#FFF4F7] px-3 py-3 min-h-[108px]">
                      <div className="flex flex-col items-center text-center">
                        <Heart className="w-4 h-4 text-rose-600 mb-2" />
                        <div className="text-rose-700 tabular-nums font-extrabold text-lg leading-7">
                        {selectedPlugin.likeCount}
                        </div>
                        <div className="text-[11px] text-rose-600 mt-2">{t('plugins.detail.likeCount')}</div>
                      </div>
                    </div>
                    <div className="rounded-lg border border-[#D8F2F5] bg-[#F2FBFC] px-3 py-3 min-h-[108px]">
                      <div className="flex flex-col items-center text-center">
                        <MessageCircle className="w-4 h-4 text-cyan-600 mb-2" />
                        <div className="text-cyan-700 tabular-nums font-extrabold text-lg leading-7">
                        {selectedPlugin.reviewCount}
                        </div>
                        <div className="text-[11px] text-cyan-600 mt-2">{t('plugins.detail.reviewCount')}</div>
                      </div>
                    </div>
                  </div>
                </div>
                {(selectedPlugin.detailDesc || '').trim().length > 0 && (
                  <div>
                    <div className="flex items-center gap-1.5 mb-2">
                      <BookOpen className="w-4 h-4 shrink-0 text-slate-600" aria-hidden />
                      <Typography variant="subtitle1" className="font-bold text-gray-900">
                        {t('plugins.detail.description')}
                      </Typography>
                    </div>
                    <PluginMarkdown
                      source={selectedPlugin.detailDesc}
                      className="prose prose-sm prose-neutral max-w-none h-64 overflow-y-auto p-4 bg-blue-50 rounded-lg border border-blue-100 shadow-sm prose-headings:scroll-mt-2 prose-pre:bg-gray-900 prose-pre:text-gray-100"
                    />
                  </div>
                )}
                {effectiveDetailVersion ? (
                  <div className="rounded-lg border border-slate-200/90 bg-slate-50/80 p-4">
                    <div className="mb-2 flex items-center gap-1.5">
                      <ScrollText className="h-4 w-4 shrink-0 text-slate-600" aria-hidden />
                      <Typography variant="subtitle1" className="font-bold text-gray-900">
                        {t('plugins.detail.versionChangelog', { version: effectiveDetailVersion })}
                      </Typography>
                    </div>
                    {detailChangelogLoading ? (
                      <div className="flex items-center gap-2 py-2 text-slate-600">
                        <CircularProgress size={18} />
                        <Typography variant="body2">{t('plugins.detail.changelogLoading')}</Typography>
                      </div>
                    ) : detailChangelogError ? (
                      <Typography variant="body2" color="error">
                        {detailChangelogError}
                      </Typography>
                    ) : detailChangelog ? (
                      <PluginMarkdown
                        source={detailChangelog}
                        className="prose prose-sm prose-neutral max-w-none max-h-48 overflow-y-auto text-gray-900 prose-p:my-1 prose-headings:my-2 prose-headings:scroll-mt-2 [&_p]:text-[0.9375rem]"
                      />
                    ) : (
                      <Typography variant="body2" color="text.secondary">
                        {t('plugins.detail.changelogEmpty')}
                      </Typography>
                    )}
                  </div>
                ) : null}
                {selectedPlugin.allVersions.length > 1 ? (
                  <FormControl fullWidth size="small">
                    <InputLabel id="plugin-detail-version-label">{t('plugins.detail.downloadVersion')}</InputLabel>
                    <Select
                      labelId="plugin-detail-version-label"
                      label={t('plugins.detail.downloadVersion')}
                      value={detailDownloadVersion || defaultDownloadVersion(selectedPlugin)}
                      onChange={e => setDetailDownloadVersion(String(e.target.value))}
                    >
                      {[...selectedPlugin.allVersions].reverse().map(v => (
                        <MenuItem key={v} value={v}>
                          v{v}
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                ) : null}
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <div className="flex items-center gap-1.5">
                      <Cpu className="w-4 h-4 shrink-0 text-slate-600" aria-hidden />
                      <Typography variant="subtitle1" className="font-bold text-gray-900">
                        {t('plugins.detail.runtime')}
                      </Typography>
                    </div>
                    <Typography variant="body2" className="mt-0.5">{getRuntimeText(selectedPlugin.runTime)}</Typography>
                  </div>
                  <div>
                    <div className="flex items-center gap-1.5">
                      <Tag className="w-4 h-4 shrink-0 text-slate-600" aria-hidden />
                      <Typography variant="subtitle1" className="font-bold text-gray-900">
                        {t('plugins.detail.tags')}
                      </Typography>
                    </div>
                    <div className="mt-1 min-h-[22px]">
                      {selectedPlugin.tags?.length ? (
                        <DetailPluginTags tags={selectedPlugin.tags} />
                      ) : (
                        <Typography variant="body2" color="text.secondary">
                          -
                        </Typography>
                      )}
                    </div>
                  </div>
                  <div>
                    <div className="flex items-center gap-1.5">
                      <CalendarPlus className="w-4 h-4 shrink-0 text-slate-600" aria-hidden />
                      <Typography variant="subtitle1" className="font-bold text-gray-900">
                        {t('plugins.detail.createTime')}
                      </Typography>
                    </div>
                    <Typography variant="body2" className="mt-0.5">
                      {formatPluginDateTime(selectedPlugin.createTime, i18n.language)}
                    </Typography>
                  </div>
                  <div>
                    <div className="flex items-center gap-1.5">
                      <CalendarClock className="w-4 h-4 shrink-0 text-slate-600" aria-hidden />
                      <Typography variant="subtitle1" className="font-bold text-gray-900">
                        {t('plugins.detail.updateTime')}
                      </Typography>
                    </div>
                    <Typography variant="body2" className="mt-0.5">
                      {formatPluginDateTime(selectedPlugin.updateTime, i18n.language)}
                    </Typography>
                  </div>
                </div>
              </div>
            </DialogContent>
            <DialogActions sx={{ flexWrap: 'wrap', gap: 1 }}>
              <Button onClick={() => setDetailDialogOpen(false)}>{t('common.buttons.close')}</Button>
              <Button
                variant="contained"
                startIcon={<Download className="h-4 w-4" />}
                disabled={downloadingAssetId === selectedPlugin.assetId}
                onClick={() =>
                  void handleDownloadPlugin(
                    selectedPlugin,
                    selectedPlugin.allVersions.length > 1
                      ? detailDownloadVersion || defaultDownloadVersion(selectedPlugin)
                      : undefined,
                  )
                }
                sx={{ textTransform: 'none' }}
              >
                {t('plugins.actions.download')}
              </Button>
            </DialogActions>
          </>
        )}
      </Dialog>
    </div>
  )
}


