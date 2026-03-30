import React, { useState, useCallback, useEffect } from 'react'
import { AlertCircle } from 'lucide-react'
import { CircularProgress, Tab, Tabs } from '@mui/material'
import { Pagination } from '../common-table'
import { ViewToggle } from './ViewToggle'
import type { PagerState, PagerChangeHandler } from '../common-table/Pagination'

export type ViewType = 'grid' | 'table'

export interface TabConfig {
  key: string
  label: string
}

const LoadingState: React.FC = () => (
  <div className="flex items-center justify-center py-12">
    <CircularProgress />
  </div>
)

interface PageHeaderProps {
  title: string
  tabs?: TabConfig[]
  activeTab?: string
  onTabChange?: (key: string) => void
  viewType?: ViewType
  onViewTypeChange?: (type: ViewType) => void
  showViewToggle?: boolean
  toolbarLeft?: React.ReactNode
  toolbarRight?: React.ReactNode
  toolbarSlogan?: string
  className?: string
}

const PageHeader: React.FC<PageHeaderProps> = ({
  title,
  tabs,
  activeTab,
  onTabChange,
  viewType = 'grid',
  onViewTypeChange,
  showViewToggle = false,
  toolbarLeft,
  toolbarRight,
  toolbarSlogan,
  className = '',
}) => (
  <div className={`mb-0 w-full rounded-2xl border border-[#e6edf9] bg-gradient-to-b from-[#f7faff] to-[#f2f7ff] px-6 py-5 shadow-[0_8px_24px_rgba(15,23,42,0.04)] ${className}`}>
    <div className="mb-4 text-center">
      <span className="inline-block bg-gradient-to-r from-[#0f172a] via-[#1d4ed8] to-[#4338ca] bg-clip-text text-transparent text-[30px] font-extrabold tracking-[0.2px]">
        {title}
      </span>
    </div>
    {tabs != null && tabs.length > 0 && (
      <div className="mb-4">
        <Tabs
          value={activeTab}
          onChange={(_e, newValue) => onTabChange?.(newValue)}
          sx={{
            height: '32px !important',
            minHeight: '32px !important',
            '& .MuiTab-root': {
              px: 0,
              height: '28px !important',
              minHeight: '28px !important',
              pb: 2,
              mr: 2,
              minWidth: 'auto',
              fontSize: '0.875rem',
            },
          }}
        >
          {tabs.map(tab => (
            <Tab key={tab.key} value={tab.key} label={tab.label} disableRipple />
          ))}
        </Tabs>
      </div>
    )}
    <div className="mt-2">
      <div className="relative flex items-center justify-end min-h-12 px-1">
        {toolbarSlogan ? (
          <div className="pointer-events-none absolute left-2 top-1/2 -translate-y-1/2 select-none">
            <span className="inline-block bg-gradient-to-r from-[#93c5fd] via-[#60a5fa] to-[#a5b4fc] bg-clip-text text-transparent text-[22px] font-extrabold tracking-[1.2px] opacity-30 whitespace-nowrap">
              {toolbarSlogan}
            </span>
          </div>
        ) : null}
        <div className="absolute left-1/2 -translate-x-1/2 flex items-center space-x-2">{toolbarLeft}</div>
        <div className="flex items-center space-x-2">
          {showViewToggle && onViewTypeChange && <ViewToggle viewType={viewType} onChange={onViewTypeChange} />}
          {toolbarRight}
        </div>
      </div>
    </div>
  </div>
)

export interface CommonPageLayoutProps {
  title: string
  tabs?: TabConfig[]
  defaultTabKey?: string
  onTabChange?: (key: string) => void
  defaultViewType?: ViewType
  viewType?: ViewType
  onViewTypeChange?: (type: ViewType) => void
  showViewToggle?: boolean
  gridView?: React.ReactNode
  tableView?: React.ReactNode
  toolbarLeft?: React.ReactNode
  toolbarRight?: React.ReactNode
  toolbarSlogan?: string
  pager: PagerState
  onPagerChange?: PagerChangeHandler
  showPagination?: boolean
  loading?: boolean
  error?: string | null
  renderContentAbove?: () => React.ReactNode
  renderContentBelow?: () => React.ReactNode
  renderPagination?: () => React.ReactNode
  className?: string
}

function CommonPageLayoutInner(props: CommonPageLayoutProps) {
  const {
    title,
    tabs,
    defaultTabKey,
    onTabChange,
    defaultViewType = 'grid',
    viewType: controlledViewType,
    onViewTypeChange,
    showViewToggle = true,
    gridView,
    tableView,
    toolbarLeft,
    toolbarRight,
    toolbarSlogan,
    pager,
    onPagerChange,
    showPagination = true,
    loading: externalLoading,
    error,
    renderContentAbove,
    renderContentBelow,
    renderPagination,
    className,
  } = props

  const [internalViewType, setInternalViewType] = useState<ViewType>(defaultViewType)
  const [activeTab, setActiveTab] = useState<string>(defaultTabKey ?? tabs?.[0]?.key ?? '')
  const [isViewSwitching, setIsViewSwitching] = useState(false)

  const viewType = controlledViewType !== undefined ? controlledViewType : internalViewType
  const effectiveLoading = externalLoading || isViewSwitching

  useEffect(() => {
    if (!externalLoading && isViewSwitching) setIsViewSwitching(false)
  }, [externalLoading, isViewSwitching])

  const handleTabChange = useCallback(
    (key: string) => {
      setActiveTab(key)
      onTabChange?.(key)
    },
    [onTabChange]
  )

  const handleViewTypeChange = useCallback(
    (type: ViewType) => {
      if (type !== viewType) setIsViewSwitching(true)
      if (onViewTypeChange) onViewTypeChange(type)
      else setInternalViewType(type)
    },
    [onViewTypeChange, viewType]
  )

  const handlePageChange = useCallback(
    (page: number, pageSize: number) => {
      onPagerChange?.(page, pageSize)
    },
    [onPagerChange]
  )

  return (
    <div className={`h-full flex flex-col bg-[#F8F9FC] ${className ?? ''}`}>
      <PageHeader
        title={title}
        tabs={tabs}
        activeTab={activeTab}
        onTabChange={handleTabChange}
        viewType={viewType}
        onViewTypeChange={handleViewTypeChange}
        showViewToggle={showViewToggle}
        toolbarLeft={toolbarLeft}
        toolbarRight={toolbarRight}
        toolbarSlogan={toolbarSlogan}
      />
      {error && (
        <div className="mx-0 mt-4 bg-red-50 border border-red-200 rounded-[4px] p-3">
          <div className="flex items-center">
            <AlertCircle className="w-4 h-4 text-red-500 mr-2" />
            <span className="text-red-800 text-sm">{error}</span>
          </div>
        </div>
      )}
      {renderContentAbove?.()}
      <div className="mx-1 mb-2 h-px bg-gradient-to-r from-transparent via-[#dbe7fb] to-transparent" />
      <div className="flex-1 overflow-hidden flex flex-col" key={`tab-${activeTab}-view-${viewType}`}>
        {effectiveLoading ? (
          <LoadingState />
        ) : viewType === 'grid' && gridView ? (
          <div className="flex-1 min-h-0 overflow-auto pt-2">{gridView}</div>
        ) : viewType === 'table' && tableView ? (
          <div className="h-full">{tableView}</div>
        ) : null}
      </div>
      {renderContentBelow?.()}
      {showPagination && pager.total > 0 && (
        <div className="w-full shrink-0 border-t border-[#e5e7eb] bg-[#F8F9FC] py-4">
          {renderPagination ? (
            renderPagination()
          ) : (
            <Pagination pager={pager} loading={effectiveLoading} error={error} onPagerChange={handlePageChange} />
          )}
        </div>
      )}
    </div>
  )
}

export const CommonPageLayout = CommonPageLayoutInner as (
  props: CommonPageLayoutProps & { key?: string | number }
) => React.ReactElement | null

export default CommonPageLayout
