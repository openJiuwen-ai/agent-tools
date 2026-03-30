import React from 'react'
import { Package } from 'lucide-react'
import { useTranslation } from 'react-i18next'

interface EmptyProps {
  searchTerm?: string
  type: 'plugins'
  customTitle?: string
  customDescription?: string
}

export const Empty: React.FC<EmptyProps> = ({
  searchTerm = '',
  customTitle,
  customDescription,
}) => {
  const { t } = useTranslation()
  const hasSearch = searchTerm.trim().length > 0
  const title =
    customTitle ??
    (hasSearch ? t('plugins.noMatching') : t('plugins.noMatching'))
  const description =
    customDescription ??
    (hasSearch ? t('plugins.noMatchingDescription') : t('plugins.noMatchingDescription'))

  return (
    <div className="flex flex-col items-center justify-center py-16">
      <div className="w-24 h-24 rounded-full bg-[#f3f4f6] flex items-center justify-center mb-6">
        <Package className="w-12 h-12 text-[#6b7280]" />
      </div>
      <div className="text-lg font-semibold text-[#1f2937] mb-2">{title}</div>
      <p className="text-[#6b7280] text-sm mb-6">{description}</p>
    </div>
  )
}

export default Empty
