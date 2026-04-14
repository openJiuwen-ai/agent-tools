import { type FormEvent, useEffect, useId, useMemo, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { Link, useNavigate, useSearchParams } from 'react-router-dom'
import { ArrowLeft, Download } from 'lucide-react'
import {
  Button,
  Checkbox,
  CircularProgress,
  FormControl,
  FormControlLabel,
  FormHelperText,
  InputLabel,
  ListItemText,
  MenuItem,
  Select,
  TextField,
  Typography,
} from '@mui/material'
import { UserAccountMenu } from '@/components/Common/UserAccountMenu'
import { useQuery } from 'react-query'
import { getPlugins, getPublishTemplatePresigned, publishPlugin } from '@/api/plugin'
import { useGitCodeAuth } from '@/auth/GitCodeAuthContext'
import { setPostLoginRedirect } from '@/auth/postLoginRedirect'
import { sha256HexOfFile } from '@/utils/sha256File'
import { buildSkillPublishZip } from '@/utils/buildSkillPublishZip'

/** 成功提示展示时长（毫秒），再跳转个人中心 */
const SUCCESS_REDIRECT_MS = 2200

const SKILL_ZIP_ERROR_KEYS: Record<string, string> = {
  INVALID_NAME: 'publish.skillErrorInvalidName',
  INVALID_VERSION: 'publish.skillErrorInvalidVersion',
  INVALID_DISPLAY_NAME: 'publish.skillErrorInvalidDisplayName',
  INVALID_DESCRIPTION: 'publish.skillErrorInvalidDescription',
  INVALID_SKILL_DESC: 'publish.skillErrorInvalidSkillDesc',
  INVALID_TAG: 'publish.skillErrorInvalidTag',
  INVALID_AUTHOR: 'publish.skillErrorInvalidAuthor',
  NO_SKILL_FILES: 'publish.skillErrorNoFiles',
  MISSING_RELATIVE_PATH: 'publish.skillErrorMissingRelativePath',
  SKILL_MD_NOT_AT_ROOT: 'publish.skillErrorSkillMdNotAtRoot',
  MISSING_SKILL_MD: 'publish.skillErrorMissingSkillMd',
  ICON_NOT_PNG: 'publish.skillErrorIconNotPng',
  ICON_TOO_LARGE: 'publish.skillErrorIconTooLarge',
  TOO_MANY_ZIP_ENTRIES: 'publish.skillErrorTooManyEntries',
}

export default function PublishPluginPage() {
  const { t } = useTranslation()
  const navigate = useNavigate()
  const [searchParams] = useSearchParams()
  const isSkillPublish = searchParams.get('kind') === 'skill'
  const skillFolderInputId = useId()
  const { user, isAuthenticated, logout } = useGitCodeAuth()

  const [file, setFile] = useState<File | null>(null)
  const [checksum, setChecksum] = useState('')
  const [hashing, setHashing] = useState(false)
  /** 空字符串 = 按新插件发布（不传 plugin_id）；否则为已有插件的 asset_id */
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

  const [skillPkgName, setSkillPkgName] = useState('')
  const [skillDisplayName, setSkillDisplayName] = useState('')
  const [skillDescription, setSkillDescription] = useState('')
  const [skillTagsInput, setSkillTagsInput] = useState('')
  const [skillIconFile, setSkillIconFile] = useState<File | null>(null)
  const [skillFolderFiles, setSkillFolderFiles] = useState<File[] | null>(null)
  const [skillFolderInputKey, setSkillFolderInputKey] = useState(0)
  const [skillIconInputKey, setSkillIconInputKey] = useState(0)
  const [packing, setPacking] = useState(false)

  useEffect(() => {
    if (isSkillPublish) {
      setFile(null)
      setChecksum('')
      setFileInputKey(k => k + 1)
    } else {
      setSkillPkgName('')
      setSkillDisplayName('')
      setSkillDescription('')
      setSkillTagsInput('')
      setSkillIconFile(null)
      setSkillFolderFiles(null)
      setSkillFolderInputKey(k => k + 1)
      setSkillIconInputKey(k => k + 1)
      setPacking(false)
    }
  }, [isSkillPublish])

  useEffect(() => {
    if (isAuthenticated) return
    setPostLoginRedirect(isSkillPublish ? '/profile/publish?kind=skill' : '/profile/publish')
    navigate('/login', { replace: true })
  }, [isAuthenticated, isSkillPublish, navigate])

  const { data: myPluginsRes, isLoading: myPluginsLoading } = useQuery(
    ['publish-my-plugins', user?.id, isSkillPublish],
    () =>
      getPlugins({
        publisher_id: user!.id,
        page: 1,
        page_size: 100,
        ...(isSkillPublish ? { plugin_type: 'skill' } : { plugin_type_exclude: 'skill' }),
      }),
    { enabled: Boolean(isAuthenticated && user?.id) },
  )

  const myPlugins = useMemo(() => {
    const items = myPluginsRes?.data?.items ?? []
    return [...items].sort((a, b) => {
      const na = (a.display_name || a.displayName || a.name || '').toLowerCase()
      const nb = (b.display_name || b.displayName || b.name || '').toLowerCase()
      return na.localeCompare(nb, undefined, { sensitivity: 'base' })
    })
  }, [myPluginsRes])

  const skillFolderRootName = useMemo(() => {
    const first = skillFolderFiles?.[0] as (File & { webkitRelativePath?: string }) | undefined
    const p = first?.webkitRelativePath
    if (!p) return ''
    return p.split(/[/\\]/).filter(Boolean)[0] ?? ''
  }, [skillFolderFiles])

  useEffect(() => {
    if (!isSkillPublish) return
    const login = user?.login?.trim()
    const icon = skillIconFile
    const folder = skillFolderFiles
    const ready =
      Boolean(login) &&
      icon &&
      folder &&
      folder.length > 0 &&
      skillPkgName.trim() &&
      pluginVersion.trim() &&
      skillDisplayName.trim() &&
      skillDescription.trim()

    if (!ready) {
      setPacking(false)
      setFile(null)
      return
    }

    let cancelled = false
    setPacking(true)
    const timer = window.setTimeout(() => {
      void (async () => {
        try {
          const tags = skillTagsInput
            .split(/[,，]/)
            .map(s => s.trim())
            .filter(Boolean)
          const zipFile = await buildSkillPublishZip({
            name: skillPkgName.trim(),
            version: pluginVersion.trim(),
            displayName: skillDisplayName.trim(),
            description: skillDescription.trim(),
            tags,
            authorLogin: login!,
            iconFile: icon!,
            skillDirectoryFiles: folder!,
          })
          if (cancelled) return
          setFile(zipFile)
          setError('')
        } catch (e) {
          if (cancelled) return
          setFile(null)
          const code = e instanceof Error ? e.message : ''
          const i18nKey = SKILL_ZIP_ERROR_KEYS[code]
          setError(i18nKey ? t(i18nKey) : t('publish.skillPackFailed'))
        } finally {
          if (!cancelled) setPacking(false)
        }
      })()
    }, 700)

    return () => {
      cancelled = true
      window.clearTimeout(timer)
    }
  }, [
    isSkillPublish,
    user?.login,
    skillIconFile,
    skillFolderFiles,
    skillPkgName,
    pluginVersion,
    skillDisplayName,
    skillDescription,
    skillTagsInput,
    t,
  ])

  useEffect(() => {
    if (!file) {
      setChecksum('')
      return
    }
    let cancelled = false
    setHashing(true)
    if (!isSkillPublish) setError('')
    void sha256HexOfFile(file)
      .then(hex => {
        if (!cancelled) {
          setChecksum(hex)
          if (isSkillPublish) setError('')
        }
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
  }, [file, t, isSkillPublish])

  useEffect(() => {
    if (!successMsg) return
    const id = window.setTimeout(() => {
      navigate('/profile', { replace: true })
    }, SUCCESS_REDIRECT_MS)
    return () => clearTimeout(id)
  }, [successMsg, navigate])

  const onSubmit = async (e: FormEvent) => {
    e.preventDefault()
    if (!file || !checksum || hashing || packing || successMsg) return
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
      if (isSkillPublish) {
        setSkillPkgName('')
        setSkillDisplayName('')
        setSkillDescription('')
        setSkillTagsInput('')
        setSkillIconFile(null)
        setSkillFolderFiles(null)
        setSkillFolderInputKey(k => k + 1)
        setSkillIconInputKey(k => k + 1)
      }
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

  const canSubmit = Boolean(
    file && checksum && !hashing && !packing && !uploading && !successMsg,
  )

  const onDownloadTemplate = async () => {
    setTemplateError('')
    setTemplateBusy(true)
    try {
      const { download_url: url } = await getPublishTemplatePresigned({ kind: 'plugin' })
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
              {isSkillPublish ? t('publish.titleSkill') : t('publish.title')}
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
          {isSkillPublish ? t('publish.introSkill') : t('publish.intro')}
        </Typography>

        {!isSkillPublish ? (
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
        ) : null}
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
          {isSkillPublish ? (
            <>
              <TextField
                label={t('publish.fieldSkillName')}
                value={skillPkgName}
                onChange={e => setSkillPkgName(e.target.value)}
                fullWidth
                size="small"
                required
                helperText={t('publish.fieldSkillNameHelp')}
              />
              <TextField
                label={t('publish.fieldVersionSkill')}
                value={pluginVersion}
                onChange={e => setPluginVersion(e.target.value)}
                fullWidth
                size="small"
                required
                helperText={t('publish.fieldVersionSkillHelp')}
              />
              <TextField
                label={t('publish.fieldSkillDisplayName')}
                value={skillDisplayName}
                onChange={e => setSkillDisplayName(e.target.value)}
                fullWidth
                size="small"
                required
                helperText={t('publish.fieldSkillDisplayNameHelp')}
              />
              <TextField
                label={t('publish.fieldSkillDescription')}
                value={skillDescription}
                onChange={e => setSkillDescription(e.target.value)}
                fullWidth
                size="small"
                required
                multiline
                minRows={3}
              />
              <TextField
                label={t('publish.fieldSkillTags')}
                value={skillTagsInput}
                onChange={e => setSkillTagsInput(e.target.value)}
                fullWidth
                size="small"
                helperText={t('publish.fieldSkillTagsHelp')}
              />
              <div>
                <Typography variant="subtitle2" className="mb-1 font-semibold text-slate-900">
                  {t('publish.fieldSkillIcon')}
                </Typography>
                <Typography variant="caption" className="mb-2 block text-slate-500">
                  {t('publish.fieldSkillIconHelp')}
                </Typography>
                <input
                  key={skillIconInputKey}
                  type="file"
                  accept="image/png,.png"
                  className="block w-full text-sm text-slate-700 file:mr-3 file:rounded-lg file:border-0 file:bg-sky-50 file:px-3 file:py-2 file:text-sm file:font-medium file:text-sky-800 hover:file:bg-sky-100"
                  onChange={e => {
                    const f = e.target.files?.[0] ?? null
                    setSkillIconFile(f)
                  }}
                />
              </div>
              <div>
                <Typography variant="subtitle2" className="mb-1 font-semibold text-slate-900">
                  {t('publish.fieldSkillFolder')}
                </Typography>
                <Typography variant="caption" className="mb-2 block text-slate-500">
                  {t('publish.fieldSkillFolderHelp')}
                </Typography>
                <input
                  key={skillFolderInputKey}
                  id={skillFolderInputId}
                  type="file"
                  className="sr-only"
                  // @ts-expect-error webkitdirectory 非标准属性，用于选择文件夹
                  webkitdirectory=""
                  multiple
                  onChange={e => {
                    const list = e.target.files
                    setSkillFolderFiles(list && list.length ? Array.from(list) : null)
                  }}
                />
                <div className="flex flex-wrap items-center gap-3">
                  <label
                    htmlFor={skillFolderInputId}
                    className="inline-flex cursor-pointer rounded-lg border border-sky-200/80 bg-sky-50 px-3 py-2 text-sm font-medium text-sky-800 shadow-sm hover:bg-sky-100"
                  >
                    {t('publish.skillFolderChoose')}
                  </label>
                  <Typography variant="body2" className="text-slate-600">
                    {skillFolderFiles?.length
                      ? t('publish.skillFolderSelected', {
                          name: skillFolderRootName || '—',
                        })
                      : t('publish.skillFolderNone')}
                  </Typography>
                </div>
              </div>
            </>
          ) : (
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
          )}

          <div>
            <Typography variant="subtitle2" className="mb-1 font-semibold text-slate-900">
              {t('publish.checksumLabel')}
            </Typography>
            <Typography variant="caption" className="mb-2 block text-slate-500">
              {isSkillPublish ? t('publish.checksumHintSkill') : t('publish.checksumHint')}
            </Typography>
            {packing || hashing ? (
              <Typography variant="body2" className="text-slate-500">
                {packing ? t('publish.skillPacking') : t('publish.hashing')}
              </Typography>
            ) : (
              <TextField
                value={checksum}
                fullWidth
                size="small"
                InputProps={{ readOnly: true }}
                placeholder={
                  file ? '' : isSkillPublish ? t('publish.checksumPlaceholderSkill') : t('publish.checksumPlaceholder')
                }
                sx={{ '& .MuiInputBase-input': { fontFamily: 'ui-monospace, monospace', fontSize: 13 } }}
              />
            )}
          </div>

          <FormControl fullWidth size="small" variant="outlined">
            <InputLabel id="publish-plugin-id-label" shrink>
              {isSkillPublish ? t('publish.fieldPluginIdSkill') : t('publish.fieldPluginId')}
            </InputLabel>
            <Select
              labelId="publish-plugin-id-label"
              label={isSkillPublish ? t('publish.fieldPluginIdSkill') : t('publish.fieldPluginId')}
              value={pluginId}
              onChange={e => setPluginId(String(e.target.value))}
              displayEmpty
              notched
              inputProps={{
                'aria-label': isSkillPublish ? t('publish.fieldPluginIdSkill') : t('publish.fieldPluginId'),
              }}
              renderValue={selected => {
                if (!selected) {
                  return isSkillPublish ? t('publish.pluginIdNewOptionSkill') : t('publish.pluginIdNewOption')
                }
                const row = myPlugins.find(p => p.asset_id === selected)
                if (!row) return selected
                const title = row.display_name || row.displayName || row.name
                const pkgName = row.name?.trim() || ''
                if (!pkgName || pkgName === title) return title
                return `${title} (${pkgName})`
              }}
            >
              <MenuItem value="">
                {isSkillPublish ? t('publish.pluginIdNewOptionSkill') : t('publish.pluginIdNewOption')}
              </MenuItem>
              {myPlugins.map(p => {
                const title = p.display_name || p.displayName || p.name
                const pkgName = p.name?.trim() || '—'
                return (
                  <MenuItem key={p.asset_id} value={p.asset_id}>
                    <ListItemText primary={title} secondary={pkgName} />
                  </MenuItem>
                )
              })}
            </Select>
            {myPluginsLoading ? (
              <FormHelperText sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <CircularProgress size={14} />
                {t('publish.pluginListLoading')}
              </FormHelperText>
            ) : (
              <FormHelperText>
                {isSkillPublish ? t('publish.fieldPluginIdHelpSkill') : t('publish.fieldPluginIdHelp')}
              </FormHelperText>
            )}
          </FormControl>

          {!isSkillPublish ? (
            <TextField
              label={t('publish.fieldVersion')}
              value={pluginVersion}
              onChange={e => setPluginVersion(e.target.value)}
              fullWidth
              size="small"
              helperText={t('publish.fieldVersionHelp')}
            />
          ) : null}

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
