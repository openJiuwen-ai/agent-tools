import { useCallback, useEffect, useMemo, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { Link, useLocation, useNavigate, useParams } from 'react-router-dom'
import { ArrowLeft, Trash2 } from 'lucide-react'
import {
  Button,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  FormControl,
  InputLabel,
  MenuItem,
  Select,
  Typography,
} from '@mui/material'
import { UserAccountMenu } from '@/components/Common/UserAccountMenu'
import { useQuery, useQueryClient } from 'react-query'
import {
  deletePluginAllVersions,
  deletePluginVersion,
  getPluginVersionDetail,
  getPlugins,
  MarketplaceApiError,
} from '@/api/plugin'
import { PluginMarkdown } from '@/components/Common/PluginMarkdown'
import { useGitCodeAuth } from '@/auth/GitCodeAuthContext'
import { setPostLoginRedirect } from '@/auth/postLoginRedirect'

export default function MyPluginDetailPage() {
  const { t } = useTranslation()
  const navigate = useNavigate()
  const location = useLocation()
  const queryClient = useQueryClient()
  const { assetId: assetIdParam } = useParams<{ assetId: string }>()
  const assetId = assetIdParam ? decodeURIComponent(assetIdParam) : ''
  const { user, isAuthenticated, logout } = useGitCodeAuth()
  const stateVersion = (location.state as { latestVersion?: string } | null)?.latestVersion

  const [selectedVersion, setSelectedVersion] = useState<string | null>(null)
  const [deleteAllOpen, setDeleteAllOpen] = useState(false)
  const [deleteOneOpen, setDeleteOneOpen] = useState(false)
  const [versionToDelete, setVersionToDelete] = useState<string | null>(null)
  const [deleting, setDeleting] = useState(false)

  useEffect(() => {
    if (isAuthenticated) return
    setPostLoginRedirect(`/profile/plugins/${encodeURIComponent(assetId)}`)
    navigate('/login', { replace: true })
  }, [isAuthenticated, navigate, assetId])

  const {
    data: summaryRes,
    isLoading: summaryLoading,
    error: summaryError,
    refetch: refetchSummary,
  } = useQuery(
    ['my-plugin-summary', assetId, user?.id],
    () =>
      getPlugins({
        publisher_id: user!.id,
        asset_id: assetId,
        page: 1,
        page_size: 1,
      }),
    {
      enabled: Boolean(assetId && user?.id),
      staleTime: 0,
    },
  )

  const summaryItem = summaryRes?.data?.items?.[0]
  const allVersions = useMemo(() => {
    const raw = summaryItem?.all_versions
    if (Array.isArray(raw) && raw.length > 0) return raw
    const lv = summaryItem?.latest_version?.trim()
    return lv ? [lv] : []
  }, [summaryItem])

  /** 与列表同步：保持选中版本在 all_versions 内；初次用路由 state / latest 兜底 */
  useEffect(() => {
    if (allVersions.length === 0) {
      setSelectedVersion(null)
      return
    }
    setSelectedVersion(prev => {
      if (prev && allVersions.includes(prev)) return prev
      const hint = stateVersion?.trim()
      if (hint && allVersions.includes(hint)) return hint
      const latest = summaryItem?.latest_version?.trim()
      if (latest && allVersions.includes(latest)) return latest
      return allVersions[allVersions.length - 1]
    })
  }, [allVersions, summaryItem?.latest_version, summaryItem?.asset_id, stateVersion])

  const { data: detail, isLoading: detailLoading, error } = useQuery(
    ['my-plugin-version', assetId, selectedVersion],
    () => getPluginVersionDetail(assetId, selectedVersion!),
    {
      enabled: Boolean(assetId && selectedVersion && user?.id),
      staleTime: 0,
    },
  )

  /**
   * GET 插件版本详情为公开接口，非所有者仍能拿到 detail；此处仅用于隐藏删除等写操作。
   * 若日后对 GET 加鉴权并返回 403，需走错误态文案。
   */
  const forbidden =
    detail && user?.id && detail.publisher_id !== user.id ? true : false

  const summaryErrMsg = useMemo(() => {
    if (!summaryError) return ''
    if (summaryError instanceof MarketplaceApiError) return summaryError.message
    return summaryError instanceof Error ? summaryError.message : String(summaryError)
  }, [summaryError])

  const errMsg = useMemo(() => {
    if (!error) return ''
    if (error instanceof MarketplaceApiError) return error.message
    return error instanceof Error ? error.message : String(error)
  }, [error])

  const notFound = !summaryLoading && !summaryItem && !summaryErrMsg

  const handleDeleteAll = async () => {
    if (!assetId || !user?.id) return
    setDeleting(true)
    try {
      await deletePluginAllVersions(assetId)
      setDeleteAllOpen(false)
      await queryClient.invalidateQueries({ queryKey: ['my-plugin-summary'] })
      navigate('/profile', { replace: true })
    } catch (e) {
      const msg = e instanceof Error ? e.message : t('profile.deleteFailed')
      window.alert(msg)
    } finally {
      setDeleting(false)
    }
  }

  const handleDeleteOne = async () => {
    if (!assetId || !user?.id || !versionToDelete) return
    setDeleting(true)
    const deleted = versionToDelete
    try {
      await deletePluginVersion(assetId, deleted)
      setDeleteOneOpen(false)
      setVersionToDelete(null)

      queryClient.removeQueries({ queryKey: ['my-plugin-version', assetId, deleted], exact: true })

      const rest = allVersions.filter(v => v !== deleted)
      if (rest.length === 0) {
        await queryClient.invalidateQueries({ queryKey: ['my-plugin-summary'] })
        await queryClient.invalidateQueries({ queryKey: ['plugins'] })
        navigate('/profile', { replace: true })
        return
      }

      const nextSel =
        selectedVersion && selectedVersion !== deleted && rest.includes(selectedVersion)
          ? selectedVersion
          : rest[rest.length - 1]
      setSelectedVersion(nextSel)

      const [sumResult] = await Promise.all([
        refetchSummary(),
        queryClient.refetchQueries({ queryKey: ['my-plugin-version', assetId, nextSel], exact: true }),
      ])
      await queryClient.invalidateQueries({ queryKey: ['plugins'] })

      if (!sumResult.data?.data?.items?.[0]) {
        navigate('/profile', { replace: true })
      }
    } catch (e) {
      const msg = e instanceof Error ? e.message : t('profile.deleteFailed')
      window.alert(msg)
    } finally {
      setDeleting(false)
    }
  }

  const openDeleteOne = useCallback((v: string) => {
    setVersionToDelete(v)
    setDeleteOneOpen(true)
  }, [])

  const versionsNewestFirst = useMemo(() => [...allVersions].reverse(), [allVersions])

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
        {summaryErrMsg ? (
          <div className="mb-4 rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-800">{summaryErrMsg}</div>
        ) : null}
        {notFound ? (
          <div className="mb-4 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-900">
            {t('profile.pluginNotFound')}
          </div>
        ) : null}
        {summaryLoading ? (
          <Typography variant="body2" className="text-slate-500">
            {t('plugins.loading')}
          </Typography>
        ) : summaryItem && allVersions.length === 0 ? (
          <div className="mb-4 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-900">
            {t('profile.missingVersion')}
          </div>
        ) : null}

        {summaryItem && allVersions.length > 0 ? (
          <div className="rounded-2xl border border-slate-200/80 bg-white/95 p-6 shadow-sm">
            <Typography variant="h5" className="mb-1 font-bold text-slate-900">
              {summaryItem.display_name || summaryItem.displayName || summaryItem.name}
            </Typography>

            <div className="mb-4 mt-4 flex flex-wrap items-center gap-3 border-t border-slate-100 pt-4">
              <FormControl size="small" className="min-w-[220px] flex-1" sx={{ maxWidth: 360 }}>
                <InputLabel id="profile-plugin-version-label">{t('profile.selectVersion')}</InputLabel>
                <Select
                  labelId="profile-plugin-version-label"
                  label={t('profile.selectVersion')}
                  value={selectedVersion ?? ''}
                  onChange={e => setSelectedVersion(String(e.target.value))}
                >
                  {versionsNewestFirst.map(v => (
                    <MenuItem key={v} value={v}>
                      v{v}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
              {!forbidden ? (
                <Button
                  color="error"
                  variant="outlined"
                  size="small"
                  disabled={!selectedVersion}
                  startIcon={<Trash2 className="h-4 w-4" aria-hidden />}
                  onClick={() => selectedVersion && openDeleteOne(selectedVersion)}
                  sx={{ textTransform: 'none', flexShrink: 0 }}
                >
                  {t('profile.deleteVersion')}
                </Button>
              ) : null}
            </div>

            {errMsg ? (
              <div className="mb-4 rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-800">{errMsg}</div>
            ) : null}
            {forbidden ? (
              <div className="mb-4 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-900">
                {t('profile.notOwner')}
              </div>
            ) : null}

            {detailLoading && selectedVersion ? (
              <Typography variant="body2" className="mb-4 text-slate-500">
                {t('plugins.loading')}
              </Typography>
            ) : null}

            {detail ? (
              <>
                <div className="mb-4 space-y-1 text-sm text-slate-700">
                  <div>
                    <span className="font-medium text-slate-900">{t('plugins.detail.publisher')}: </span>
                    {detail.publisher_name}
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

                <div className="mb-4 rounded-xl border border-slate-200/90 bg-slate-50/90 p-4">
                  <Typography variant="subtitle2" className="mb-2 font-bold text-slate-900">
                    {t('profile.changelog')} · v{detail.version}
                  </Typography>
                  {detail.changelog?.trim() ? (
                    <PluginMarkdown
                      source={detail.changelog.trim()}
                      className="prose prose-sm prose-neutral max-w-none max-h-64 overflow-y-auto text-slate-800 prose-p:my-1 prose-headings:my-2 prose-headings:scroll-mt-2 [&_p]:text-[0.9375rem]"
                    />
                  ) : (
                    <Typography variant="body2" color="text.secondary">
                      {t('profile.changelogEmpty')}
                    </Typography>
                  )}
                </div>
              </>
            ) : !detailLoading && selectedVersion && !errMsg ? (
              <Typography variant="body2" className="mb-4 text-slate-500">
                {t('profile.noDetail')}
              </Typography>
            ) : null}

            {!forbidden ? (
              <Button
                color="error"
                variant="outlined"
                onClick={() => setDeleteAllOpen(true)}
                sx={{ textTransform: 'none' }}
              >
                {t('profile.deleteAll')}
              </Button>
            ) : null}
          </div>
        ) : null}
      </main>

      <Dialog open={deleteAllOpen} onClose={() => !deleting && setDeleteAllOpen(false)} maxWidth="xs" fullWidth>
        <DialogTitle>{t('profile.deleteConfirmTitle')}</DialogTitle>
        <DialogContent>
          <Typography variant="body2" className="text-slate-700">
            {t('profile.deleteConfirmBody')}
          </Typography>
        </DialogContent>
        <DialogActions sx={{ px: 3, pb: 2 }}>
          <Button onClick={() => setDeleteAllOpen(false)} disabled={deleting} sx={{ textTransform: 'none' }}>
            {t('common.buttons.close')}
          </Button>
          <Button color="error" variant="contained" onClick={() => void handleDeleteAll()} disabled={deleting} sx={{ textTransform: 'none' }}>
            {t('profile.deleteConfirmAction')}
          </Button>
        </DialogActions>
      </Dialog>

      <Dialog open={deleteOneOpen} onClose={() => !deleting && setDeleteOneOpen(false)} maxWidth="xs" fullWidth>
        <DialogTitle>{t('profile.deleteVersionConfirmTitle')}</DialogTitle>
        <DialogContent>
          <Typography variant="body2" className="text-slate-700">
            {t('profile.deleteVersionConfirmBody', { version: versionToDelete ?? '' })}
          </Typography>
        </DialogContent>
        <DialogActions sx={{ px: 3, pb: 2 }}>
          <Button onClick={() => setDeleteOneOpen(false)} disabled={deleting} sx={{ textTransform: 'none' }}>
            {t('common.buttons.close')}
          </Button>
          <Button color="error" variant="contained" onClick={() => void handleDeleteOne()} disabled={deleting} sx={{ textTransform: 'none' }}>
            {t('profile.deleteVersionConfirmAction')}
          </Button>
        </DialogActions>
      </Dialog>
    </div>
  )
}
