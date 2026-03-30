import type { AnchorHTMLAttributes } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkBreaks from 'remark-breaks'
import remarkGfm from 'remark-gfm'

type PluginMarkdownProps = {
  /** Markdown 源码；非字符串会转为字符串 */
  source: string | null | undefined
  className?: string
}

type MarkdownLinkProps = AnchorHTMLAttributes<HTMLAnchorElement> & { node?: unknown }

/** 仅允许 http(s)、mailto、站内根路径 /，屏蔽 javascript:/data:/vbscript: 与 // 协议相对 URL */
function sanitizeMarkdownHref(href: string | undefined): string | undefined {
  if (href == null || typeof href !== 'string') return undefined
  const t = href.trim()
  if (!t) return undefined
  const lower = t.toLowerCase()
  if (lower.startsWith('javascript:') || lower.startsWith('data:') || lower.startsWith('vbscript:')) {
    return undefined
  }
  if (lower.startsWith('http://') || lower.startsWith('https://')) return t
  if (lower.startsWith('mailto:')) return t
  if (t.startsWith('/') && !t.startsWith('//')) return t
  return undefined
}

function MarkdownAnchor({ href, children, ...rest }: MarkdownLinkProps) {
  const safe = sanitizeMarkdownHref(href)
  if (!safe) {
    return <span className="text-gray-800">{children}</span>
  }
  const external = /^https?:\/\//i.test(safe)
  return (
    <a
      {...rest}
      href={safe}
      className="text-blue-600 underline underline-offset-2"
      {...(external ? { target: '_blank', rel: 'noopener noreferrer' } : {})}
    >
      {children}
    </a>
  )
}

function decodeEscapedMarkdown(input: string): string {
  // 后端偶发返回多层 JSON 字符串字面量（包含 \"、\\r\\n 等），做有限次解码
  let text = input
  for (let i = 0; i < 3; i += 1) {
    const trimmed = text.trim()
    if (!(trimmed.startsWith('"') && trimmed.endsWith('"'))) break
    try {
      const parsed = JSON.parse(trimmed)
      if (typeof parsed !== 'string') break
      text = parsed
    } catch {
      break
    }
  }

  // 回退：将常见的转义换行符还原为真实换行
  return text.replace(/\\r\\n/g, '\n').replace(/\\n/g, '\n').replace(/\\r/g, '\n')
}

/**
 * 插件市场详情等场景使用的 Markdown 渲染；使用显式 `source` 避免 JSX 子节点空白折叠问题。
 */
export function PluginMarkdown({ source, className }: PluginMarkdownProps) {
  const raw = source == null ? '' : typeof source === 'string' ? source : String(source)
  const text = decodeEscapedMarkdown(raw)
  return (
    <div className={className}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkBreaks]}
        components={{
          h1: props => <h1 className="text-xl font-bold leading-7 text-gray-900 my-3 first:mt-0" {...props} />,
          h2: props => <h2 className="text-lg font-semibold leading-7 text-gray-900 my-3 first:mt-0" {...props} />,
          h3: props => <h3 className="text-base font-semibold leading-6 text-gray-900 my-2.5 first:mt-0" {...props} />,
          p: props => <p className="text-sm leading-6 text-gray-800 my-2 first:mt-0 last:mb-0" {...props} />,
          ul: props => <ul className="list-disc pl-5 my-2 space-y-1" {...props} />,
          ol: props => <ol className="list-decimal pl-5 my-2 space-y-1" {...props} />,
          li: props => <li className="text-sm leading-6 text-gray-800" {...props} />,
          code: props => <code className="rounded bg-slate-100 px-1.5 py-0.5 text-[0.85em] text-slate-900" {...props} />,
          pre: props => <pre className="overflow-x-auto rounded-md bg-slate-900 p-3 text-slate-100 my-3" {...props} />,
          a: props => <MarkdownAnchor {...props} />,
        }}
      >
        {text}
      </ReactMarkdown>
    </div>
  )
}
