import { type FormEvent, useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { Link, useNavigate } from 'react-router-dom'
import { ArrowLeft, Download } from 'lucide-react'
import { Button, Checkbox, FormControlLabel, TextField, Typography } from '@mui/material'
import { UserAccountMenu } from '@/components/Common/UserAccountMenu'
import { getPublishTemplatePresigned, publishPlugin } from '@/api/plugin'
import { useGitCodeAuth } from '@/auth/GitCodeAuthContext'
import { setPostLoginRedirect } from '@/auth/postLoginRedirect'
import { sha256HexOfFile } from '@/utils/sha256File'

/** 成功提示展示时长（毫秒），再跳转个人中心 */
const SUCCESS_REDIRECT_MS = 2200

export default function PublishPluginPage() {
  const { t } = useTranslation()
  const navigate = useNavigate()
  const { user, isAuthenticated, logout } = useGitCodeAuth()

  const [file, setFile] = useState<File | null>(null)
  const [checksum, setChecksum] = useState('')
  const [hashing, setHashing] = useState(false)
  const [pluginId, setPluginId] = useState('')
  const [pluginVersion, setPluginVersion] = useState('')
  const [versionDesc, setVersionDesc] = useState('')
  const [force, setForce] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState('')
  const [successMsg, setSuccessMsg] = useState('')
  const [fileInputKey, setFileInputKey] = useState(0)
  const [templateBusy, setTemplateBusy] = useState(false)
  const [templateError, setTemplateError] = useState('')

  useEffect(() => {
    if (isAuthenticated) return
    setPostLoginRedirect('/profile/publish')
    navigate('/login', { replace: true })
  }, [isAuthenticated, navigate])

  useEffect(() => {
    if (!file) {
      setChecksum('')
      return
    }
    let cancelled = false
    setHashing(true)
    setError('')
    void sha256HexOfFile(file)
      .then(hex => {
        if (!cancelled) setChecksum(hex)
      })
      .catch(() => {
        if (!cancelled) {
          setChecksum('')
          setError(t('publish.hashFailed'))
        }
      })
      .finally(() => {
        if (!cancelled) setHashing(false)
      })
    return () => {
      cancelled = true
    }
  }, [file, t])

  useEffect(() => {
    if (!successMsg) return
    const id = window.setTimeout(() => {
      navigate('/profile', { replace: true })
    }, SUCCESS_REDIRECT_MS)
    return () => clearTimeout(id)
  }, [successMsg, navigate])

  const onSubmit = async (e: FormEvent) => {
    e.preventDefault()
    if (!file || !checksum || hashing || successMsg) return
    setUploading(true)
    setError('')
    setSuccessMsg('')
    try {
      const data = await publishPlugin({
        file,
        checksumSha256Hex: checksum,
        pluginId: pluginId.trim() || undefined,
        pluginVersion: pluginVersion.trim() || undefined,
        versionDesc: versionDesc.trim() || undefined,
        force,
      })
      setFile(null)
      setChecksum('')
      setPluginId('')
      setPluginVersion('')
      setVersionDesc('')
      setForce(false)
      setFileInputKey(k => k + 1)
      setSuccessMsg(
        t('publish.successDetail', {
          name: data.name,
          version: data.version,
          pluginId: data.plugin_id,
        }),
      )
    } catch (err) {
      setError(err instanceof Error ? err.message : t('publish.uploadFailed'))
    } finally {
      setUploading(false)
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

  const canSubmit = Boolean(file && checksum && !hashing && !uploading && !successMsg)

  const onDownloadTemplate = async () => {
    setTemplateError('')
    setTemplateBusy(true)
    try {
      const { download_url: url } = await getPublishTemplatePresigned()
      window.location.assign(url)
    } catch (err) {
      setTemplateError(err instanceof Error ? err.message : t('publish.templateDownloadFailed'))
    } finally {
      setTemplateBusy(false)
    }
  }

  return (
    <div className="relative flex min-h-dvh flex-col bg-gradient-to-br from-[#f8fbff] via-[#f6faff] to-[#eef4ff]">
      <div className="pointer-events-none absolute -top-20 -right-20 h-72 w-72 rounded-full bg-blue-100/50 blur-3xl" />
      <header className="relative z-10 border-b border-slate-200/80 bg-white/90 px-4 py-3 shadow-sm shadow-slate-200/40">
        <div className="mx-auto flex max-w-3xl flex-wrap items-center justify-between gap-3">
          <div className="flex min-w-0 items-center gap-3">
            <Link
              to="/profile"
              className="inline-flex items-center gap-1 text-sm font-medium text-[#0369a1] hover:text-[#0c4a6e]"
            >
              <ArrowLeft className="h-4 w-4" />
              {t('publish.backToProfile')}
            </Link>
            <Typography variant="h6" className="truncate font-bold text-slate-900">
              {t('publish.title')}
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

      <main className="relative z-10 mx-auto w-full max-w-3xl flex-1 px-4 py-6">
        <Typography variant="body2" className="mb-4 text-slate-600">
          {t('publish.intro')}
        </Typography>

        <div className="mb-6 flex flex-wrap items-center gap-3">
          <Button
            type="button"
            variant="outlined"
            disabled={templateBusy}
            onClick={() => void onDownloadTemplate()}
            startIcon={<Download className="h-4 w-4" />}
            sx={{ textTransform: 'none' }}
          >
            {templateBusy ? t('publish.templateFetching') : t('publish.downloadTemplate')}
          </Button>
          <Typography variant="caption" className="max-w-md text-slate-500">
            {t('publish.templateHint')}
          </Typography>
        </div>
        {templateError ? (
          <div className="mb-4 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-900">
            {templateError}
          </div>
        ) : null}

        {error ? (
          <div className="mb-4 rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-800">{error}</div>
        ) : null}
        {successMsg ? (
          <div className="mb-4 rounded-lg border border-emerald-200 bg-emerald-50 px-3 py-2 text-sm text-emerald-900">
            <div>{successMsg}</div>
            <div className="mt-1 text-xs text-emerald-800/90">{t('publish.redirectHint')}</div>
          </div>
        ) : null}
        <form onSubmit={onSubmit} className="space-y-4 rounded-2xl border border-slate-200/80 bg-white/95 p-6 shadow-sm">
          <div>
            <Typography variant="subtitle2" className="mb-1 font-semibold text-slate-900">
              {t('publish.zipLabel')}
            </Typography>
            <input
              key={fileInputKey}
              type="file"
              accept=".zip,application/zip"
              className="block w-full text-sm text-slate-700 file:mr-3 file:rounded-lg file:border-0 file:bg-sky-50 file:px-3 file:py-2 file:text-sm file:font-medium file:text-sky-800 hover:file:bg-sky-100"
              onChange={e => {
                const f = e.target.files?.[0] ?? null
                setFile(f)
              }}
            />
          </div>

          <div>
            <Typography variant="subtitle2" className="mb-1 font-semibold text-slate-900">
              {t('publish.checksumLabel')}
            </Typography>
            <Typography variant="caption" className="mb-2 block text-slate-500">
              {t('publish.checksumHint')}
            </Typography>
            {hashing ? (
              <Typography variant="body2" className="text-slate-500">
                {t('publish.hashing')}
              </Typography>
            ) : (
              <TextField
                value={checksum}
                fullWidth
                size="small"
                InputProps={{ readOnly: true }}
                placeholder={file ? '' : t('publish.checksumPlaceholder')}
                sx={{ '& .MuiInputBase-input': { fontFamily: 'ui-monospace, monospace', fontSize: 13 } }}
              />
            )}
          </div>

          <TextField
            label={t('publish.fieldPluginId')}
            value={pluginId}
            onChange={e => setPluginId(e.target.value)}
            fullWidth
            size="small"
            helperText={t('publish.fieldPluginIdHelp')}
          />
          <TextField
            label={t('publish.fieldVersion')}
            value={pluginVersion}
            onChange={e => setPluginVersion(e.target.value)}
            fullWidth
            size="small"
            helperText={t('publish.fieldVersionHelp')}
          />
          <TextField
            label={t('publish.fieldVersionDesc')}
            value={versionDesc}
            onChange={e => setVersionDesc(e.target.value)}
            fullWidth
            size="small"
            multiline
            minRows={2}
          />
          <FormControlLabel
            control={<Checkbox checked={force} onChange={e => setForce(e.target.checked)} />}
            label={t('publish.fieldForce')}
          />

          <Button type="submit" variant="contained" disabled={!canSubmit} sx={{ textTransform: 'none', mt: 1 }}>
            {uploading ? t('publish.uploading') : t('publish.submit')}
          </Button>
        </form>
      </main>
    </div>
  )
}
