import { useEffect, useMemo, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { Link, useNavigate, useSearchParams } from 'react-router-dom'
import { ArrowLeft } from 'lucide-react'
import { Button, Typography } from '@mui/material'
import { SegmentedTabs } from '@/components/Common/common-page'
import { UserAccountMenu } from '@/components/Common/UserAccountMenu'
import { Pagination } from '@/components/Common/common-table'
import { useQuery } from 'react-query'
import { getPlugins, type MarketplacePluginItem } from '@/api/plugin'
import { useGitCodeAuth } from '@/auth/GitCodeAuthContext'
import { setPostLoginRedirect } from '@/auth/postLoginRedirect'

const PROFILE_PAGE_SIZE_OPTIONS = [10, 20, 50] as const

export default function MyProfilePage() {
  const { t } = useTranslation()
  const navigate = useNavigate()
  const [searchParams, setSearchParams] = useSearchParams()
  const { user, isAuthenticated, logout } = useGitCodeAuth()
  /** 默认「插件」；仅 `?tab=skill` 时展示 Skill（发布 Skill 页返回会带上） */
  const publishedTab: 'plugin' | 'skill' = searchParams.get('tab') === 'skill' ? 'skill' : 'plugin'
  const setPublishedTab = (next: 'plugin' | 'skill') => {
    setSearchParams(
      prev => {
        const p = new URLSearchParams(prev)
        if (next === 'skill') p.set('tab', 'skill')
        else p.delete('tab')
        return p
      },
      { replace: true },
    )
  }
  const [page, setPage] = useState(1)
  const [pageSize, setPageSize] = useState(20)

  useEffect(() => {
    if (isAuthenticated) return
    setPostLoginRedirect('/profile')
    navigate('/login', { replace: true })
  }, [isAuthenticated, navigate])

  const publisherId = user?.id

  const { data, isLoading, error, refetch } = useQuery(
    ['my-published-plugins', publisherId, publishedTab, page, pageSize],
    () =>
      getPlugins({
        page,
        page_size: pageSize,
        publisher_id: publisherId,
        order_by: 'update_time',
        desc: true,
        ...(publishedTab === 'skill'
          ? { plugin_type: 'skill' }
          : { plugin_type_exclude: 'skill' }),
      }),
    {
      enabled: Boolean(publisherId),
      keepPreviousData: true,
    },
  )

  useEffect(() => {
    setPage(1)
  }, [publishedTab])

  const items = data?.data.items ?? []
  const total = data?.data.total ?? 0

  useEffect(() => {
    if (total <= 0) return
    const totalPages = Math.max(1, Math.ceil(total / pageSize))
    if (page > totalPages) setPage(totalPages)
  }, [total, pageSize, page])

  const handlePagerChange = (nextPage: number, nextPageSize: number) => {
    setPageSize(nextPageSize)
    setPage(nextPage)
  }

  const errMsg = useMemo(() => {
    if (!error) return ''
    return error instanceof Error ? error.message : String(error)
  }, [error])

  const openDetail = (row: MarketplacePluginItem) => {
    const v = row.latest_version?.trim()
    const versions = Array.isArray(row.all_versions) ? row.all_versions : []
    const fallback = versions.length ? versions[versions.length - 1] : ''
    const hint = v || fallback
    if (!hint) {
      window.alert(t('profile.missingVersion'))
      return
    }
    navigate(`/profile/plugins/${encodeURIComponent(row.asset_id)}`, {
      state: { latestVersion: hint },
    })
  }

  if (!isAuthenticated || !user) {
    return (
      <div className="flex min-h-dvh items-center justify-center bg-gradient-to-br from-[#f8fbff] via-[#f6faff] to-[#eef4ff]">
        <Typography variant="body2" color="text.secondary">
          {t('profile.redirecting')}
        </Typography>
      </div>
    )
  }

  return (
    <div className="relative flex min-h-0 flex-1 flex-col overflow-hidden bg-gradient-to-br from-[#f8fbff] via-[#f6faff] to-[#eef4ff]">
      <div className="pointer-events-none absolute -top-20 -right-20 h-72 w-72 rounded-full bg-blue-100/50 blur-3xl" />
      <header className="relative z-10 shrink-0 border-b border-slate-200/80 bg-white/90 px-4 py-3 shadow-sm shadow-slate-200/40">
        <div className="mx-auto flex max-w-5xl flex-wrap items-center justify-between gap-3">
          <div className="flex min-w-0 items-center gap-3">
            <Link
              to="/"
              className="inline-flex items-center gap-1 text-sm font-medium text-[#0369a1] hover:text-[#0c4a6e]"
            >
              <ArrowLeft className="h-4 w-4" />
              {t('profile.backToMarket')}
            </Link>
            <Typography variant="h6" className="truncate font-bold text-slate-900">
              {t('profile.title')}
            </Typography>
          </div>
          <UserAccountMenu
            primaryLabel={user.name || user.login}
            title={user.name || user.login}
            items={[
              {
                id: 'logout',
                label: t('auth.toolbar.logout'),
                onClick: () => {
                  logout()
                  navigate('/', { replace: true })
                },
              },
            ]}
          />
        </div>
      </header>

      <main className="relative z-10 mx-auto flex min-h-0 w-full max-w-5xl flex-1 flex-col px-4 py-6">
        {errMsg ? (
          <div className="mb-4 shrink-0 rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-800">{errMsg}</div>
        ) : null}

        <div className="mb-4 flex shrink-0 flex-col gap-3">
          <SegmentedTabs
            align="start"
            size="sm"
            value={publishedTab}
            onChange={v => setPublishedTab(v === 'skill' ? 'skill' : 'plugin')}
            options={[
              { value: 'skill', label: t('profile.tabSkills') },
              { value: 'plugin', label: t('profile.tabPlugins') },
            ]}
            aria-label={t('profile.segmentedTabsAria')}
          />
          <div className="flex flex-wrap items-center justify-between gap-2">
            <Typography variant="body2" className="text-slate-600">
              {publishedTab === 'skill' ? t('profile.subtitleSkills') : t('profile.subtitlePlugins')}
            </Typography>
            <div className="flex flex-wrap items-center gap-2">
            <Button
              size="small"
              variant="contained"
              onClick={() =>
                navigate(publishedTab === 'skill' ? '/profile/publish?kind=skill' : '/profile/publish')
              }
              sx={{ textTransform: 'none', bgcolor: '#0891b2', '&:hover': { bgcolor: '#0e7490' } }}
            >
              {publishedTab === 'skill' ? t('profile.publishSkill') : t('profile.publishPlugin')}
            </Button>
            <Button size="small" variant="text" onClick={() => void refetch()} disabled={isLoading} sx={{ textTransform: 'none' }}>
              {t('plugins.actions.refresh')}
            </Button>
            </div>
          </div>
        </div>

        <div className="flex min-h-0 flex-1 flex-col overflow-hidden">
          {isLoading && !data ? (
            <Typography variant="body2" className="text-slate-500">
              {t('plugins.loading')}
            </Typography>
          ) : items.length === 0 ? (
            <Typography variant="body2" className="text-slate-500">
              {publishedTab === 'skill' ? t('profile.emptySkills') : t('profile.emptyPlugins')}
            </Typography>
          ) : (
            <div className="flex min-h-0 flex-1 flex-col overflow-hidden rounded-xl border border-slate-200/80 bg-white/95 shadow-sm">
              <div className="min-h-0 flex-1 overflow-auto">
                <table className="w-full text-left text-sm">
                  <thead className="sticky top-0 z-[1] border-b border-slate-200 bg-slate-50/95 text-xs font-semibold uppercase tracking-wide text-slate-600 backdrop-blur-sm">
                    <tr>
                      <th className="px-4 py-3">{t('profile.table.name')}</th>
                      <th className="px-4 py-3">{t('plugins.detail.version')}</th>
                      <th className="px-4 py-3">{t('plugins.tableView.columns.type')}</th>
                      <th className="w-[120px] px-4 py-3">{t('profile.table.action')}</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100">
                    {items.map((row: MarketplacePluginItem) => (
                      <tr key={row.asset_id} className="hover:bg-slate-50/80">
                        <td className="px-4 py-3 font-medium text-slate-900">
                          <div className="max-w-[280px] truncate">{row.display_name || row.name}</div>
                          <div className="truncate text-xs font-normal text-slate-500">{row.name || '—'}</div>
                        </td>
                        <td className="px-4 py-3 tabular-nums text-slate-700">{row.latest_version ?? '—'}</td>
                        <td className="px-4 py-3 text-slate-600">{row.plugin_type ?? '—'}</td>
                        <td className="px-4 py-3">
                          <button
                            type="button"
                            onClick={() => openDetail(row)}
                            className="text-sm font-medium text-[#0369a1] hover:underline"
                          >
                            {t('profile.viewDetail')}
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>

        {total > 0 && data ? (
          <div className="w-full shrink-0 border-t border-[#e5e7eb] bg-[#F8F9FC] py-4">
            <Pagination
              pager={{
                total,
                currentPage: page,
                pageSize,
                pageSizeOptions: [...PROFILE_PAGE_SIZE_OPTIONS],
              }}
              loading={false}
              onPagerChange={handlePagerChange}
            />
          </div>
        ) : null}
      </main>
    </div>
  )
}
