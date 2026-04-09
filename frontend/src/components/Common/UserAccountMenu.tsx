import { useCallback, useEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { ChevronDown } from 'lucide-react'

export type UserAccountMenuItem = {
  id: string
  label: string
  onClick: () => void
}

type UserAccountMenuProps = {
  primaryLabel: string
  title?: string
  items: UserAccountMenuItem[]
}

export function UserAccountMenu({ primaryLabel, title, items }: UserAccountMenuProps) {
  const [panelVisible, setPanelVisible] = useState(false)
  const rootRef = useRef<HTMLDivElement>(null)
  const menuRef = useRef<HTMLUListElement>(null)
  const [menuCoords, setMenuCoords] = useState<{ top: number; right: number } | null>(null)

  const recalcMenuPosition = useCallback(() => {
    const el = rootRef.current
    if (!el) return
    const r = el.getBoundingClientRect()
    setMenuCoords({ top: r.bottom + 4, right: window.innerWidth - r.right })
  }, [])

  useEffect(() => {
    if (!panelVisible) {
      setMenuCoords(null)
      return
    }
    recalcMenuPosition()
    const onScrollOrResize = () => recalcMenuPosition()
    window.addEventListener('scroll', onScrollOrResize, true)
    window.addEventListener('resize', onScrollOrResize)
    return () => {
      window.removeEventListener('scroll', onScrollOrResize, true)
      window.removeEventListener('resize', onScrollOrResize)
    }
  }, [panelVisible, recalcMenuPosition])

  useEffect(() => {
    if (!panelVisible) return
    const closeIfOutside = (event: MouseEvent) => {
      const t = event.target as Node
      if (rootRef.current?.contains(t)) return
      if (menuRef.current?.contains(t)) return
      setMenuCoords(null)
      setPanelVisible(false)
    }
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        setMenuCoords(null)
        setPanelVisible(false)
      }
    }
    document.addEventListener('mousedown', closeIfOutside)
    document.addEventListener('keydown', onKeyDown)
    return () => {
      document.removeEventListener('mousedown', closeIfOutside)
      document.removeEventListener('keydown', onKeyDown)
    }
  }, [panelVisible])

  const menu =
    panelVisible && menuCoords ? (
      <ul
        ref={menuRef}
        role="menu"
        style={{
          position: 'fixed',
          top: menuCoords.top,
          right: menuCoords.right,
          zIndex: 1400,
        }}
        className="min-w-[168px] rounded-lg border border-slate-200/90 bg-white py-1 shadow-lg shadow-slate-200/50"
      >
        {items.map(item => (
          <li key={item.id} role="none">
            <button
              type="button"
              role="menuitem"
              className="w-full px-3 py-2.5 text-left text-sm text-slate-800 hover:bg-slate-50 active:bg-slate-100"
              onClick={() => {
                setMenuCoords(null)
                setPanelVisible(false)
                item.onClick()
              }}
            >
              {item.label}
            </button>
          </li>
        ))}
      </ul>
    ) : null

  return (
    <div className="relative shrink-0" ref={rootRef}>
      <button
        type="button"
        aria-haspopup="menu"
        aria-expanded={panelVisible}
        onClick={() => {
          setPanelVisible(prev => {
            if (prev) {
              setMenuCoords(null)
              return false
            }
            const el = rootRef.current
            if (el) {
              const r = el.getBoundingClientRect()
              setMenuCoords({ top: r.bottom + 4, right: window.innerWidth - r.right })
            }
            return true
          })
        }}
        title={title ?? primaryLabel}
        className="inline-flex h-10 max-w-[220px] items-center gap-1 rounded-lg border border-[#d7e2f6] bg-white/95 px-3 text-sm font-medium text-[#1f2937] shadow-[0_1px_3px_rgba(15,23,42,0.06)] transition-colors hover:border-[#bfdbfe] hover:bg-[#f8fbff]"
      >
        <span className="truncate">{primaryLabel}</span>
        <ChevronDown className={`h-4 w-4 shrink-0 opacity-70 transition-transform ${panelVisible ? 'rotate-180' : ''}`} aria-hidden />
      </button>
      {typeof document !== 'undefined' && menu ? createPortal(menu, document.body) : null}
    </div>
  )
}
