import i18n from 'i18next'
import LanguageDetector from 'i18next-browser-languagedetector'
import { initReactI18next } from 'react-i18next'
import zhCN from './locales/zh-CN.json'
import enUS from './locales/en-US.json'

export const SUPPORTED_LOCALES = ['zh-CN', 'en-US'] as const
export type SupportedLocale = (typeof SUPPORTED_LOCALES)[number]

function applyHtmlLang(lng: string) {
  document.documentElement.lang = lng.startsWith('zh') ? 'zh-CN' : 'en'
}

/** 与 LanguageDetector 可能给出的 zh / en 对齐，避免找不到资源而回退成显示 key */
const resources = {
  'zh-CN': { translation: zhCN },
  zh: { translation: zhCN },
  'en-US': { translation: enUS },
  en: { translation: enUS },
} as const

i18n
  .use(LanguageDetector)
  .use(initReactI18next)
  .init({
    resources,
    fallbackLng: 'zh-CN',
    supportedLngs: ['zh-CN', 'en-US', 'zh', 'en'],
    nonExplicitSupportedLngs: true,
    keySeparator: '.',
    nsSeparator: ':',
    interpolation: { escapeValue: false },
    react: {
      useSuspense: false,
    },
    detection: {
      order: ['localStorage', 'navigator', 'htmlTag'],
      caches: ['localStorage'],
      lookupLocalStorage: 'i18nextLng',
    },
  })

applyHtmlLang(i18n.language || 'zh-CN')
i18n.on('languageChanged', lng => {
  applyHtmlLang(lng)
})

export default i18n
