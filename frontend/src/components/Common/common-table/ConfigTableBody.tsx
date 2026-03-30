import React from 'react'
import { Box, Checkbox, TableCell, TableRow } from '@mui/material'
import { TableColumn, TableStyles } from './types'
import { getDefaultRowId } from './utils'
import { ActionButtons } from './ConfigTableOperations'

/** 行悬停：纯色高亮（无渐变），与行底色协调 */
const ROW_HOVER_BG = 'rgba(59, 130, 246, 0.09)'
const ROW_BASE_BACKGROUNDS = ['#f8fbff']
const ROW_ACCENT_BARS = ['#0ea5e9', '#6366f1', '#10b981', '#f43f5e', '#f59e0b']

export interface ConfigTableBodyProps<T> {
  rows: T[]
  columns: TableColumn<T>[]
  columnWidths: Record<string, number>
  tableStyles?: TableStyles
  slotMap: Record<string, React.ReactElement>
  enableSelection: boolean
  selectionKey?: keyof T & string
  getRowId?: (row: T) => string | number
  effectiveSelectedIds: Array<string | number>
  onToggleRow: (row: T, index: number) => void
  onOpenOperationMenu: (event: React.MouseEvent<HTMLElement>, rowId: string | number, columnKey: string) => void
  operationMenuState: {
    anchorEl: HTMLElement | null
    rowId: string | number | null
    columnKey: string | null
  }
}

/**
 * ConfigTableBody component - renders the table body with rows
 */
export function ConfigTableBody<T extends object>({
  rows,
  columns,
  columnWidths,
  tableStyles,
  slotMap,
  enableSelection,
  selectionKey,
  getRowId,
  effectiveSelectedIds,
  onToggleRow,
  onOpenOperationMenu,
  operationMenuState,
}: ConfigTableBodyProps<T>) {
  const getRowIdInternal = (row: T, index: number) => getDefaultRowId(row, index, selectionKey, getRowId)

  return (
    <>
      {rows.map((row, rowIndex) => {
        const rowId = getRowIdInternal(row, rowIndex)
        const isSelected = effectiveSelectedIds.includes(rowId)
        const baseBackground = ROW_BASE_BACKGROUNDS[rowIndex % ROW_BASE_BACKGROUNDS.length]
        const accentBar = ROW_ACCENT_BARS[rowIndex % ROW_ACCENT_BARS.length]
        return (
          <TableRow
            hover
            key={rowId}
            selected={isSelected}
            sx={{
              '--row-base-bg': baseBackground,
              '--row-accent-bar': accentBar,
              transition: 'box-shadow 0.2s ease',
              ...tableStyles?.tableRow?.root,
              '& .MuiTableCell-root': {
                transition: 'background-color 0.2s ease, box-shadow 0.2s ease',
                borderBottom: 0,
                borderTop: '1px solid #dbe7ff',
                borderBottomColor: 'transparent',
                backgroundClip: 'padding-box',
              },
              '& .MuiTableCell-body:first-of-type': {
                borderTopLeftRadius: '12px',
                borderBottomLeftRadius: '12px',
                borderLeft: '1px solid #dbe7ff',
                position: 'relative',
                '&::before': {
                  content: '""',
                  position: 'absolute',
                  left: 4,
                  top: '50%',
                  transform: 'translateY(-50%)',
                  width: 3,
                  height: 26,
                  borderRadius: 999,
                  backgroundColor: 'var(--row-accent-bar)',
                },
              },
              '& .MuiTableCell-body:last-of-type': {
                borderTopRightRadius: '12px',
                borderBottomRightRadius: '12px',
                borderRight: '1px solid #dbe7ff',
              },
              '& .MuiTableCell-body': {
                backgroundColor: 'var(--row-base-bg)',
                borderBottom: '1px solid #dbe7ff',
              },
              '&:hover': {
                boxShadow: '0 4px 14px rgba(15, 23, 42, 0.07)',
                ...tableStyles?.tableRow?.hover,
                '& .MuiTableCell-body': {
                  backgroundColor: ROW_HOVER_BG,
                  backgroundImage: 'none',
                },
              },
            }}
          >
            {enableSelection && (
              <TableCell
                padding="checkbox"
                align="center"
                sx={{
                  width: 50,
                  minWidth: 50,
                  maxWidth: 50,
                  position: 'sticky',
                  left: 0,
                  zIndex: 1,
                  ...tableStyles?.tableCell?.body,
                }}
              >
                <Checkbox checked={isSelected} onChange={() => onToggleRow(row, rowIndex)} />
              </TableCell>
            )}
            {columns.map(column => {
              const dataKey = (column.dataIndex ?? (column.key as keyof T & string)) as keyof T
              const value = (row as any)[dataKey]
              const slot = slotMap[column.key]
              let content: React.ReactNode
              if (slot) {
                content = React.cloneElement(
                  slot as React.ReactElement<any>,
                  {
                    row,
                    value,
                    rowIndex,
                    column,
                  } as any,
                )
              } else if (column.render) {
                content = column.render({
                  row,
                  value,
                  rowIndex,
                  column,
                })
              } else if (column.type === 'date') {
                if (column.dateFormatter) {
                  content = column.dateFormatter(value, row, rowIndex, column)
                } else {
                  content = (
                    <Box component="span" sx={{ color: 'error.main', fontSize: '0.75rem' }}>
                      dateFormatter required
                    </Box>
                  )
                }
              } else if (column.type === 'operate') {
                const operations = column.operations ?? []
                if (operations.length === 0) {
                  content = null
                } else {
                  const isMenuOpen = operationMenuState.rowId === rowId && operationMenuState.columnKey === column.key
                  content = (
                    <ActionButtons
                      operations={operations}
                      row={row}
                      rowIndex={rowIndex}
                      rowId={rowId}
                      columnKey={column.key}
                      onOpenMenu={onOpenOperationMenu}
                      isMenuOpen={isMenuOpen}
                    />
                  )
                }
              } else {
                content = value === undefined || value === null ? '' : String(value)
              }
              const width = columnWidths[column.key] ?? column.width
              return (
                <TableCell
                  key={column.key}
                  align={column.align}
                  sx={{
                    width: width,
                    minWidth: column.minWidth ?? 80,
                    maxWidth: column.maxWidth,
                    whiteSpace: 'nowrap',
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    ...tableStyles?.tableCell?.body,
                  }}
                >
                  {content}
                </TableCell>
              )
            })}
          </TableRow>
        )
      })}
    </>
  )
}

export default ConfigTableBody
