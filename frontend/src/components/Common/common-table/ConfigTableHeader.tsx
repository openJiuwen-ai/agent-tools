import React from 'react'
import { Box, IconButton, TableCell } from '@mui/material'
import FilterAltIcon from '@mui/icons-material/FilterAlt'
import ArrowDropUpIcon from '@mui/icons-material/ArrowDropUp'
import ArrowDropDownIcon from '@mui/icons-material/ArrowDropDown'
import { TableColumn, SortState, ColumnFilterState, TableStyles } from './types'
import { hasActiveFilter } from './utils'

export interface ConfigTableHeaderProps<T> {
  columns: TableColumn<T>[]
  sortState: SortState
  filters: ColumnFilterState
  columnWidths: Record<string, number>
  tableStyles?: TableStyles
  /** 与 MUI Table `stickyHeader` 一致：开启后表头行在纵向滚动时固定 */
  stickyHeader?: boolean
  onSortChange: (columnKey: string) => void
  onOpenFilterMenu: (event: React.MouseEvent<HTMLElement>, columnKey: string) => void
  onColumnResizeMouseDown: (event: React.MouseEvent<HTMLDivElement>, columnKey: string) => void
}

/**
 * ConfigTableHeader component - renders table header cells (returns array of TableCell)
 */
export function ConfigTableHeader<T extends object>({
  columns,
  sortState,
  filters,
  columnWidths,
  tableStyles,
  stickyHeader = false,
  onSortChange,
  onOpenFilterMenu,
  onColumnResizeMouseDown,
}: ConfigTableHeaderProps<T>) {
  const headerIconButtonSx = {
    borderRadius: 0.75,
    p: 0.25,
    width: 16,
    height: 24,
  }

  return (
    <>
      {columns.map(column => {
        const width = columnWidths[column.key] ?? column.width
        const sortFieldValue = column.sortField || column.key
        const isSorted = sortState.field === sortFieldValue && !!sortState.order
        const cellAlign = column.align ?? 'center'
        const justifyContent =
          cellAlign === 'left' ? 'flex-start' : cellAlign === 'right' ? 'flex-end' : 'center'
        return (
          <TableCell
            key={column.key}
            align={cellAlign}
            sx={{
              width: width,
              minWidth: column.minWidth ?? 80,
              maxWidth: column.maxWidth,
              whiteSpace: 'nowrap',
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              fontWeight: 700,
              height: 44,
              py: 0,
              lineHeight: 1.2,
              verticalAlign: 'middle',
              ...tableStyles?.tableCell?.head,
              ...(stickyHeader
                ? {
                    position: 'sticky',
                    top: 0,
                    zIndex: 3,
                    backgroundColor: 'var(--table-bg-cell)',
                  }
                : { position: 'relative' }),
            }}
          >
            <Box
              display="flex"
              alignItems="center"
              justifyContent={justifyContent}
              position="relative"
              height="100%"
              minHeight={0}
              pr={column.filterable || column.sortable ? 4 : 0}
            >
              <Box
                component="span"
                sx={{
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  fontWeight: 700,
                }}
              >
                {column.title}
              </Box>
              {(column.filterable || column.sortable) && (
                <Box display="flex" alignItems="center" ml={0.5} gap={0.5} position="absolute" right={0}>
                  {column.filterable && (
                    <IconButton
                      size="small"
                      aria-label="Filter"
                      onClick={event => onOpenFilterMenu(event, column.key)}
                      sx={headerIconButtonSx}
                    >
                      <FilterAltIcon fontSize="small" color={hasActiveFilter(filters[column.key]) ? 'primary' : 'disabled'} />
                    </IconButton>
                  )}
                  {column.sortable && (
                    <IconButton size="small" aria-label="Sort" onClick={() => onSortChange(column.key)} sx={headerIconButtonSx}>
                      <Box display="flex" flexDirection="column" lineHeight={1} alignItems="center">
                        <ArrowDropUpIcon fontSize="small" color={isSorted && sortState.order === 'asc' ? 'primary' : 'disabled'} sx={{ mb: '-6px' }} />
                        <ArrowDropDownIcon
                          fontSize="small"
                          color={isSorted && sortState.order === 'desc' ? 'primary' : 'disabled'}
                          sx={{ mt: '-6px' }}
                        />
                      </Box>
                    </IconButton>
                  )}
                </Box>
              )}
            </Box>
            <Box
              onMouseDown={event => onColumnResizeMouseDown(event, column.key)}
              sx={{
                position: 'absolute',
                right: 0,
                top: 0,
                bottom: 0,
                width: 4,
                cursor: 'col-resize',
                zIndex: 1,
              }}
            />
          </TableCell>
        )
      })}
    </>
  )
}

export default ConfigTableHeader
