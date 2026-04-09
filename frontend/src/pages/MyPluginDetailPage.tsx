import { useEffect, useMemo, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { Link, useLocation, useNavigate, useParams } from 'react-router-dom'
import { ArrowLeft } from 'lucide-react'
import { Button, Dialog, DialogActions, DialogContent, DialogTitle, Typography } from '@mui/material'
import { UserAccountMenu } from '@/components/Common/UserAccountMenu'
import { useQuery } from 'react-query'
import { deletePluginAllVersions, getPluginVersionDetail, getPlugins, MarketplaceApiError } from '@/api/plugin'
import { PluginMarkdown } from '@/components/Common/PluginMarkdown'
import { useGitCodeAuth } from '@/auth/GitCodeAuthContext'
import { setPostLoginRedirect } from '@/auth/postLoginRedirect'

export default function MyPluginDetailPage() {
  const { t } = useTranslation()
  const navigate = useNavigate()
  const location = useLocation()
  const { assetId: assetIdParam } = useParams<{ assetId: string }>()
  const assetId = assetIdParam ? decodeURIComponent(assetIdParam) : ''
  const { user, isAuthenticated, logout } = useGitCodeAuth()
  const stateVersion = (location.state as { latestVersion?: string } | null)?.latestVersion

  const [resolvedVersion, setResolvedVersion] = useState<string | null>(stateVersion?.trim() || null)
  const [resolveFailed, setResolveFailed] = useState(false)
  const [deleteOpen, setDeleteOpen] = useState(false)
  const [deleting, setDeleting] = useState(false)

  useEffect(() => {
    if (isAuthenticated) return
    setPostLoginRedirect(`/profile/plugins/${encodeURIComponent(assetId)}`)
    navigate('/login', { replace: true })
  }, [isAuthenticated, navigate, assetId])

  useEffect(() => {
    if (!user?.id || !assetId || resolvedVersion) return
    let cancelled = false
    void (async () => {
      try {
        const res = await getPlugins({
          publisher_id: user.id,
          asset_id: assetId,
          page: 1,
          page_size: 1,
        })
        const v = res.data.items[0]?.latest_version?.trim()
        if (!cancelled) {
          if (v) setResolvedVersion(v)
          else setResolveFailed(true)
        }
      } catch {
        if (!cancelled) setResolveFailed(true)
      }
    })()
    return () => {
      cancelled = true
    }
  }, [user?.id, assetId, resolvedVersion])

  const { data: detail, isLoading, error } = useQuery(
    ['my-plugin-version', assetId, resolvedVersion],
    () => getPluginVersionDetail(assetId, resolvedVersion!),
    {
      enabled: Boolean(assetId && resolvedVersion && user?.id),
    },
  )

  /**
   * GET 插件版本详情为公开接口，非所有者仍能拿到 detail；此处仅用于隐藏删除等写操作。
   * 若日后对 GET 加鉴权并返回 403，需走错误态文案。
   */
  const forbidden =
    detail && user?.id && detail.publisher_id !== user.id ? true : false

  const errMsg = useMemo(() => {
    if (!error) return ''
    if (error instanceof MarketplaceApiError) return error.message
    return error instanceof Error ? error.message : String(error)
  }, [error])

  const handleDelete = async () => {
    if (!assetId || !user?.id) return
    setDeleting(true)
    try {
      await deletePluginAllVersions(assetId)
      setDeleteOpen(false)
      navigate('/profile', { replace: true })
    } catch (e) {
      const msg = e instanceof Error ? e.message : t('profile.deleteFailed')
      window.alert(msg)
    } finally {
      setDeleting(false)
    }
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
      <header className="relative z-10 border-b border-slate-200/80 bg-white/90 px-4 py-3 shadow-sm shadow-slate-200/40">
        <div className="mx-auto flex max-w-3xl flex-wrap items-center justify-between gap-3">
          <div className="flex min-w-0 items-center gap-3">
            <Link
              to="/profile"
              className="inline-flex items-center gap-1 text-sm font-medium text-[#0369a1] hover:text-[#0c4a6e]"
            >
              <ArrowLeft className="h-4 w-4" />
              {t('profile.backToList')}
            </Link>
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

      <main className="relative z-10 mx-auto w-full max-w-3xl flex-1 px-4 py-6">
        {errMsg ? (
          <div className="mb-4 rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-800">{errMsg}</div>
        ) : null}
        {resolveFailed ? (
          <div className="mb-4 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-900">
            {t('profile.missingVersion')}
          </div>
        ) : null}
        {forbidden ? (
          <div className="mb-4 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-900">
            {t('profile.notOwner')}
          </div>
        ) : null}

        {(isLoading || !resolvedVersion) && !resolveFailed ? (
          <Typography variant="body2" className="text-slate-500">
            {t('plugins.loading')}
          </Typography>
        ) : detail ? (
          <div className="rounded-2xl border border-slate-200/80 bg-white/95 p-6 shadow-sm">
            <Typography variant="h5" className="mb-1 font-bold text-slate-900">
              {detail.display_name || detail.name}
            </Typography>
            <Typography variant="caption" className="mb-4 block text-slate-500">
              {detail.asset_id} · v{detail.version}
            </Typography>

            <div className="mb-4 space-y-1 text-sm text-slate-700">
              <div>
                <span className="font-medium text-slate-900">{t('plugins.detail.publisher')}: </span>
                {detail.publisher_name} ({detail.publisher_id})
              </div>
              {detail.plugin_type ? (
                <div>
                  <span className="font-medium text-slate-900">{t('plugins.detail.runtime')}: </span>
                  {detail.plugin_type}
                </div>
              ) : null}
            </div>

            {detail.short_desc ? (
              <div className="mb-4">
                <Typography variant="subtitle2" className="mb-1 font-bold text-slate-900">
                  {t('plugins.detail.summary')}
                </Typography>
                <PluginMarkdown
                  source={detail.short_desc}
                  className="prose prose-sm prose-neutral max-w-none text-slate-800 prose-p:my-1"
                />
              </div>
            ) : null}

            {detail.detail_desc ? (
              <div className="mb-4">
                <Typography variant="subtitle2" className="mb-1 font-bold text-slate-900">
                  {t('plugins.detail.description')}
                </Typography>
                <PluginMarkdown
                  source={detail.detail_desc}
                  className="prose prose-sm prose-neutral max-w-none text-slate-800 prose-p:my-1"
                />
              </div>
            ) : null}

            {detail.changelog ? (
              <div className="mb-6">
                <Typography variant="subtitle2" className="mb-1 font-bold text-slate-900">
                  {t('profile.changelog')}
                </Typography>
                <pre className="whitespace-pre-wrap rounded-lg bg-slate-50 px-3 py-2 text-sm text-slate-800">
                  {detail.changelog}
                </pre>
              </div>
            ) : null}

            {!forbidden ? (
              <Button
                color="error"
                variant="outlined"
                onClick={() => setDeleteOpen(true)}
                sx={{ textTransform: 'none' }}
              >
                {t('profile.deleteAll')}
              </Button>
            ) : null}
          </div>
        ) : resolveFailed ? null : (
          <Typography variant="body2" className="text-slate-500">
            {t('profile.noDetail')}
          </Typography>
        )}
      </main>

      <Dialog open={deleteOpen} onClose={() => !deleting && setDeleteOpen(false)} maxWidth="xs" fullWidth>
        <DialogTitle>{t('profile.deleteConfirmTitle')}</DialogTitle>
        <DialogContent>
          <Typography variant="body2" className="text-slate-700">
            {t('profile.deleteConfirmBody')}
          </Typography>
        </DialogContent>
        <DialogActions sx={{ px: 3, pb: 2 }}>
          <Button onClick={() => setDeleteOpen(false)} disabled={deleting} sx={{ textTransform: 'none' }}>
            {t('common.buttons.close')}
          </Button>
          <Button color="error" variant="contained" onClick={() => void handleDelete()} disabled={deleting} sx={{ textTransform: 'none' }}>
            {t('profile.deleteConfirmAction')}
          </Button>
        </DialogActions>
      </Dialog>
    </div>
  )
}
