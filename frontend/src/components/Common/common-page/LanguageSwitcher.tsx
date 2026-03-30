import { useTranslation } from 'react-i18next'
import type { SupportedLocale } from '@/i18n'

const LOCALE_CONFIG: { code: SupportedLocale; labelKey: 'common.lang.zh' | 'common.lang.en' }[] = [
  { code: 'zh-CN', labelKey: 'common.lang.zh' },
  { code: 'en-US', labelKey: 'common.lang.en' },
]

function resolveActive(code: SupportedLocale, resolved: string): boolean {
  const r = resolved || 'zh-CN'
  if (code === 'zh-CN') {
    return r === 'zh-CN' || r.startsWith('zh')
  }
  return r === 'en-US' || (r.startsWith('en') && !r.startsWith('zh'))
}

export const LanguageSwitcher = () => {
  const { i18n, t } = useTranslation()
  const resolved = i18n.resolvedLanguage ?? i18n.language

  return (
    <div
      className="flex h-8 rounded-[4px] p-0.5"
      style={{ backgroundColor: '#ECECF0' }}
      role="group"
      aria-label={t('common.lang.switchAria')}
    >
      {LOCALE_CONFIG.map(({ code, labelKey }) => {
        const active = resolveActive(code, resolved)
        return (
          <button
            key={code}
            type="button"
            onClick={() => void i18n.changeLanguage(code)}
            className={`h-7 min-w-[44px] rounded-[3px] px-2 text-xs font-medium transition-colors ${
              active ? 'text-[#295BFB]' : 'text-[#777777] hover:bg-white/60'
            }`}
            style={{ backgroundColor: active ? '#FFFFFF' : 'transparent' }}
            aria-pressed={active}
            aria-label={t(labelKey)}
          >
            {t(labelKey)}
          </button>
        )
      })}
    </div>
  )
}

export default LanguageSwitcher
