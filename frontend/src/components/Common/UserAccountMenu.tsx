import { useEffect, useRef, useState } from 'react'
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

  useEffect(() => {
    if (!panelVisible) return
    const closeIfOutside = (event: MouseEvent) => {
      const el = rootRef.current
      if (el && !el.contains(event.target as Node)) {
        setPanelVisible(false)
      }
    }
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setPanelVisible(false)
    }
    document.addEventListener('mousedown', closeIfOutside)
    document.addEventListener('keydown', onKeyDown)
    return () => {
      document.removeEventListener('mousedown', closeIfOutside)
      document.removeEventListener('keydown', onKeyDown)
    }
  }, [panelVisible])

  return (
    <div className="relative shrink-0" ref={rootRef}>
      <button
        type="button"
        aria-haspopup="menu"
        aria-expanded={panelVisible}
        onClick={() => setPanelVisible(v => !v)}
        title={title ?? primaryLabel}
        className="inline-flex h-10 max-w-[220px] items-center gap-1 rounded-lg border border-[#d7e2f6] bg-white/95 px-3 text-sm font-medium text-[#1f2937] shadow-[0_1px_3px_rgba(15,23,42,0.06)] hover:bg-[#f8fbff] hover:border-[#bfdbfe] transition-colors"
      >
        <span className="truncate">{primaryLabel}</span>
        <ChevronDown className={`h-4 w-4 shrink-0 opacity-70 transition-transform ${panelVisible ? 'rotate-180' : ''}`} aria-hidden />
      </button>
      {panelVisible ? (
        <ul
          role="menu"
          className="absolute right-0 z-[200] mt-1 min-w-[168px] rounded-lg border border-slate-200/90 bg-white py-1 shadow-lg shadow-slate-200/50"
        >
          {items.map(item => (
            <li key={item.id} role="none">
              <button
                type="button"
                role="menuitem"
                className="w-full px-3 py-2.5 text-left text-sm text-slate-800 hover:bg-slate-50 active:bg-slate-100"
                onClick={() => {
                  setPanelVisible(false)
                  item.onClick()
                }}
              >
                {item.label}
              </button>
            </li>
          ))}
        </ul>
      ) : null}
    </div>
  )
}
