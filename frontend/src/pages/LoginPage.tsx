import { useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { LogIn } from 'lucide-react'
import { Button, Typography } from '@mui/material'
import {
  exchangeGitCodeOAuthSession,
  getOAuthGitCodeStartUrl,
  GITCODE_OAUTH_PENDING_KEY,
} from '@/api/auth'
import { useGitCodeAuth } from '@/auth/GitCodeAuthContext'

export default function LoginPage() {
  const { t } = useTranslation()
  const navigate = useNavigate()
  const [searchParams] = useSearchParams()
  const { login, isAuthenticated } = useGitCodeAuth()
  const [commonError, setCommonError] = useState('')
  const [exchanging, setExchanging] = useState(false)

  useEffect(() => {
    if (isAuthenticated) {
      navigate('/', { replace: true })
    }
  }, [isAuthenticated, navigate])

  useEffect(() => {
    const oauthErr = searchParams.get('oauth_error')
    if (!oauthErr) return
    try {
      setCommonError(decodeURIComponent(oauthErr.replace(/\+/g, ' ')))
    } catch {
      setCommonError(oauthErr)
    }
    navigate('/login', { replace: true })
  }, [searchParams, navigate])

  useEffect(() => {
    const fromUrl = searchParams.get('oauth_session')
    if (fromUrl) {
      sessionStorage.setItem(GITCODE_OAUTH_PENDING_KEY, fromUrl)
      navigate('/login', { replace: true })
      return
    }
    const sid = sessionStorage.getItem(GITCODE_OAUTH_PENDING_KEY)
    if (!sid) return
    sessionStorage.removeItem(GITCODE_OAUTH_PENDING_KEY)
    setExchanging(true)
    setCommonError('')
    exchangeGitCodeOAuthSession(sid)
      .then(data => {
        login(data.access_token, data.user)
        navigate('/', { replace: true })
      })
      .catch(e => {
        const msg = e instanceof Error ? e.message : t('auth.oauth.genericError')
        setCommonError(msg)
      })
      .finally(() => setExchanging(false))
  }, [searchParams, navigate, login, t])

  const startGitCode = () => {
    setCommonError('')
    window.location.href = getOAuthGitCodeStartUrl()
  }

  return (
    <div className="flex min-h-dvh flex-col items-center justify-center bg-gradient-to-br from-[#f8fbff] via-[#f6faff] to-[#eef4ff] px-4">
      <div className="w-full max-w-md rounded-2xl border border-slate-200/80 bg-white/95 p-8 shadow-lg shadow-slate-200/60">
        <Typography variant="h5" className="mb-2 font-bold text-slate-900">
          {t('auth.login.title')}
        </Typography>
        <Typography variant="body2" className="mb-6 text-slate-600">
          {t('auth.login.subtitle')}
        </Typography>
        {commonError ? (
          <div className="mb-4 rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-800">{commonError}</div>
        ) : null}
        {exchanging ? (
          <Typography variant="body2" className="mb-4 text-slate-600">
            {t('auth.oauth.exchanging')}
          </Typography>
        ) : null}
        <Button
          variant="contained"
          fullWidth
          size="large"
          disabled={exchanging}
          onClick={startGitCode}
          sx={{
            py: 1.5,
            textTransform: 'none',
            fontWeight: 600,
            bgcolor: '#0891b2',
            '&:hover': { bgcolor: '#0e7490' },
          }}
          startIcon={<LogIn className="h-5 w-5" />}
        >
          {t('auth.login.gitcodeButton')}
        </Button>
        <Typography variant="caption" className="mt-4 block text-center text-slate-500">
          {t('auth.login.hintGitcodeSession')}
        </Typography>
      </div>
    </div>
  )
}
