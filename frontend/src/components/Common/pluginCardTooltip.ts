import type { TooltipProps } from '@mui/material/Tooltip'

/**
 * 插件列表卡片：标签区与底部浏览/下载/赞/评等 Tooltip 统一外观（与 MUI 默认灰底提示条对齐并略作圆角与阴影）。
 */
const tooltipSx = {
  bgcolor: '#374151',
  color: '#fff',
  fontSize: '0.75rem',
  fontWeight: 500,
  lineHeight: 1.45,
  px: 1.25,
  py: 0.75,
  borderRadius: '8px',
  boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.08)',
  maxWidth: 320,
} as const

export const pluginCardTooltipProps: Partial<TooltipProps> = {
  placement: 'top',
  slotProps: {
    tooltip: {
      sx: tooltipSx,
    },
  },
}

/** 详情页标题区（评分、收藏等）：提示在下方，与顶栏不重叠，样式与 pluginCardTooltipProps 一致 */
export const pluginDetailHeaderTooltipProps: Partial<TooltipProps> = {
  placement: 'bottom',
  slotProps: {
    tooltip: {
      sx: tooltipSx,
    },
  },
}
