import { ToggleButton, ToggleButtonGroup } from '@mui/material'

export interface SegmentedTabOption {
  value: string
  label: string
}

export interface SegmentedTabsProps {
  value: string
  options: SegmentedTabOption[]
  onChange: (value: string) => void
  /** 市场首页等与标题居中对齐 */
  align?: 'center' | 'start'
  size?: 'sm' | 'md'
  'aria-label'?: string
  className?: string
}

/**
 * 分段页签：浅灰轨道 + 选中项浮起白底，常见于设置页/资源切换（Segmented Control）。
 */
export default function SegmentedTabs({
  value,
  options,
  onChange,
  align = 'center',
  size = 'md',
  'aria-label': ariaLabel = '切换分类',
  className,
}: SegmentedTabsProps) {
  const padY = size === 'sm' ? '6px' : '9px'
  const padX = size === 'sm' ? '18px' : '22px'
  const fontSize = size === 'sm' ? '0.8125rem' : '0.875rem'

  const group = (
    <ToggleButtonGroup
      value={value}
      exclusive
      onChange={(_e, v) => {
        if (v != null) onChange(String(v))
      }}
      aria-label={ariaLabel}
      sx={{
        borderRadius: '14px',
        p: '5px',
        gap: '5px',
        backgroundColor: 'rgba(241, 245, 249, 0.85)',
        border: '1px solid rgba(203, 213, 225, 0.9)',
        boxShadow: 'inset 0 1px 0 rgba(255, 255, 255, 0.7)',
        '& .MuiToggleButtonGroup-grouped': {
          margin: 0,
          border: 'none',
          borderRadius: '10px !important',
          py: padY,
          px: padX,
          minWidth: size === 'sm' ? 72 : 88,
          textTransform: 'none',
          fontWeight: 600,
          letterSpacing: '0.01em',
          fontSize,
          color: '#64748b',
          backgroundColor: 'transparent',
          transition: 'color 0.18s ease, background-color 0.18s ease, box-shadow 0.18s ease',
          '&:hover': {
            backgroundColor: 'rgba(255, 255, 255, 0.65)',
            color: '#475569',
          },
          '&.Mui-focusVisible': {
            outline: '2px solid #93c5fd',
            outlineOffset: 2,
          },
          '&.Mui-selected': {
            backgroundColor: '#ffffff',
            color: '#1d4ed8',
            boxShadow:
              '0 1px 2px rgba(15, 23, 42, 0.06), 0 2px 10px rgba(37, 99, 235, 0.12), 0 0 0 1px rgba(191, 219, 254, 0.6)',
            '&:hover': {
              backgroundColor: '#ffffff',
              color: '#1e40af',
            },
          },
        },
      }}
    >
      {options.map(opt => (
        <ToggleButton key={opt.value} value={opt.value}>
          {opt.label}
        </ToggleButton>
      ))}
    </ToggleButtonGroup>
  )

  if (align === 'center') {
    return <div className={`flex w-full justify-center ${className ?? ''}`}>{group}</div>
  }
  return <div className={className}>{group}</div>
}
