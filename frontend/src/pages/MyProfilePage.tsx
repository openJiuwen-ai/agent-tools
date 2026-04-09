import { useEffect, useMemo, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { Link, useNavigate } from 'react-router-dom'
import { ArrowLeft, ChevronLeft, ChevronRight } from 'lucide-react'
import { Button, Typography } from '@mui/material'
import { UserAccountMenu } from '@/components/Common/UserAccountMenu'
import { useQuery } from 'react-query'
import { getPlugins, type MarketplacePluginItem } from '@/api/plugin'
import { useGitCodeAuth } from '@/auth/GitCodeAuthContext'
import { setPostLoginRedirect } from '@/auth/postLoginRedirect'

const PAGE_SIZE = 20

export default function MyProfilePage() {
  const { t } = useTranslation()
  const navigate = useNavigate()
  const { user, isAuthenticated, logout } = useGitCodeAuth()
  const [page, setPage] = useState(1)

  useEffect(() => {
    if (isAuthenticated) return
    setPostLoginRedirect('/profile')
    navigate('/login', { replace: true })
  }, [isAuthenticated, navigate])

  const publisherId = user?.id

  const { data, isLoading, error, refetch } = useQuery(
    ['my-published-plugins', publisherId, page],
    () =>
      getPlugins({
        page,
        page_size: PAGE_SIZE,
        publisher_id: publisherId,
        order_by: 'update_time',
        desc: true,
      }),
    {
      enabled: Boolean(publisherId),
      keepPreviousData: true,
    },
  )

  const items = data?.data.items ?? []
  const total = data?.data.total ?? 0
  const totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE))

  const errMsg = useMemo(() => {
    if (!error) return ''
    return error instanceof Error ? error.message : String(error)
  }, [error])

  const openDetail = (row: MarketplacePluginItem) => {
    const v = row.latest_version?.trim()
    if (!v) {
      window.alert(t('profile.missingVersion'))
      return
    }
    navigate(`/profile/plugins/${encodeURIComponent(row.asset_id)}`, {
      state: { latestVersion: v },
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
    <div className="relative flex min-h-dvh flex-col bg-gradient-to-br from-[#f8fbff] via-[#f6faff] to-[#eef4ff]">
      <div className="pointer-events-none absolute -top-20 -right-20 h-72 w-72 rounded-full bg-blue-100/50 blur-3xl" />
      <header className="relative z-10 border-b border-slate-200/80 bg-white/90 px-4 py-3 shadow-sm shadow-slate-200/40">
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

      <main className="relative z-10 mx-auto w-full max-w-5xl flex-1 px-4 py-6">
        {errMsg ? (
          <div className="mb-4 rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-800">{errMsg}</div>
        ) : null}

        <div className="mb-4 flex flex-wrap items-center justify-between gap-2">
          <Typography variant="body2" className="text-slate-600">
            {t('profile.subtitle')}
          </Typography>
          <div className="flex flex-wrap items-center gap-2">
            <Button
              size="small"
              variant="contained"
              onClick={() => navigate('/profile/publish')}
              sx={{ textTransform: 'none', bgcolor: '#0891b2', '&:hover': { bgcolor: '#0e7490' } }}
            >
              {t('profile.publishEntry')}
            </Button>
            <Button size="small" variant="text" onClick={() => void refetch()} disabled={isLoading} sx={{ textTransform: 'none' }}>
              {t('plugins.actions.refresh')}
            </Button>
          </div>
        </div>

        {isLoading && !data ? (
          <Typography variant="body2" className="text-slate-500">
            {t('plugins.loading')}
          </Typography>
        ) : items.length === 0 ? (
          <Typography variant="body2" className="text-slate-500">
            {t('profile.empty')}
          </Typography>
        ) : (
          <div className="overflow-hidden rounded-xl border border-slate-200/80 bg-white/95 shadow-sm">
            <table className="w-full text-left text-sm">
              <thead className="border-b border-slate-200 bg-slate-50/90 text-xs font-semibold uppercase tracking-wide text-slate-600">
                <tr>
                  <th className="px-4 py-3">{t('profile.table.name')}</th>
                  <th className="px-4 py-3">{t('plugins.detail.version')}</th>
                  <th className="px-4 py-3">{t('plugins.tableView.columns.type')}</th>
                  <th className="px-4 py-3 w-[120px]">{t('profile.table.action')}</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {items.map((row: MarketplacePluginItem) => (
                  <tr key={row.asset_id} className="hover:bg-slate-50/80">
                    <td className="px-4 py-3 font-medium text-slate-900">
                      <div className="max-w-[280px] truncate">{row.display_name || row.name}</div>
                      <div className="text-xs font-normal text-slate-500 truncate">{row.asset_id}</div>
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
        )}

        {totalPages > 1 ? (
          <div className="mt-4 flex items-center justify-center gap-2">
            <button
              type="button"
              disabled={page <= 1 || isLoading}
              onClick={() => setPage(p => Math.max(1, p - 1))}
              className="inline-flex h-9 items-center rounded-lg border border-slate-200 bg-white px-3 text-sm disabled:opacity-40"
            >
              <ChevronLeft className="h-4 w-4" />
            </button>
            <span className="text-sm text-slate-600">
              {t('profile.pageIndicator', { page, total: totalPages })}
            </span>
            <button
              type="button"
              disabled={page >= totalPages || isLoading}
              onClick={() => setPage(p => Math.min(totalPages, p + 1))}
              className="inline-flex h-9 items-center rounded-lg border border-slate-200 bg-white px-3 text-sm disabled:opacity-40"
            >
              <ChevronRight className="h-4 w-4" />
            </button>
          </div>
        ) : null}
      </main>
    </div>
  )
}
