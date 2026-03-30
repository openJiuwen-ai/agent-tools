import React from 'react'
import { Search, X } from 'lucide-react'

export interface SearchInputProps {
  searchTerm: string
  placeholder: string
  onChange: (value: string) => void
}

export const SearchInput: React.FC<SearchInputProps> = ({ searchTerm, placeholder, onChange }) => {
  return (
    <div className="relative w-[min(46vw,460px)] min-w-[320px]">
      <Search className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4.5 h-4.5 text-[#94A3B8]" />
      <input
        type="text"
        placeholder={placeholder}
        value={searchTerm}
        onChange={e => onChange(e.target.value)}
        className="w-full h-10 pl-10 pr-9 rounded-lg border border-[#d7e2f6] bg-white/95 text-sm text-[#1F2937] placeholder-[#94A3B8] shadow-[inset_0_1px_0_rgba(255,255,255,0.8),0_1px_3px_rgba(15,23,42,0.06)] transition-all hover:border-[#bfd0ef] focus:outline-none focus:border-[#3B82F6] focus:ring-2 focus:ring-[#bfdbfe]"
      />
      {searchTerm && (
        <button
          onClick={() => onChange('')}
          className="absolute right-3 top-1/2 -translate-y-1/2 text-[#94A3B8] hover:text-[#64748b] transition-colors"
          type="button"
        >
          <X className="w-3.5 h-3.5" />
        </button>
      )}
    </div>
  )
}

export default SearchInput
