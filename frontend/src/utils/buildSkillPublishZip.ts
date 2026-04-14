import JSZip from 'jszip'
import { dump as yamlDump } from 'js-yaml'

/** 与 marketplace `plugins_market.validation.constants` 对齐 */
const SKILL_NAME_PATTERN = /^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$/
const SEMVER_PATTERN = /^\d+\.\d+\.\d+$/
const DISPLAY_NAME_MAX_LEN = 128
const PLUGIN_YAML_DESCRIPTION_MAX_LEN = 1024
const SKILL_DESC_MAX_LEN = 1024
const MAX_ZIP_ENTRIES = 1000
const PNG_MAGIC = new Uint8Array([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a])

const PACK_IGNORE_DIR_NAMES = new Set([
  '.git',
  '__pycache__',
  '.venv',
  'venv',
  '.eggs',
  'dist',
  'out',
  '__MACOSX',
])

export type BuildSkillPublishZipInput = {
  /** plugin.yaml `name`，须与 skill 子目录名一致 */
  name: string
  /** plugin.yaml `version`，x.y.z */
  version: string
  displayName: string
  /** plugin.yaml `description`；同时写入 SKILL.md frontmatter */
  description: string
  tags: string[]
  /** GitCode 登录名 → metadata.author */
  authorLogin: string
  iconFile: File
  /** 来自 `<input webkitdirectory>` 的 File 列表 */
  skillDirectoryFiles: File[]
}

function normalizeSemver(raw: string): string {
  const s = raw.trim().replace(/^v+/i, '')
  if (!SEMVER_PATTERN.test(s)) {
    throw new Error('INVALID_VERSION')
  }
  return s
}

function normalizeSkillSlug(raw: string): string {
  const s = raw.trim().toLowerCase()
  if (!s || s.length > 64 || !SKILL_NAME_PATTERN.test(s)) {
    throw new Error('INVALID_NAME')
  }
  return s
}

function parseTags(raw: string): string[] {
  const parts = raw
    .split(/[,，]/)
    .map(s => s.trim())
    .filter(Boolean)
  return parts
}

async function assertPng(file: File): Promise<void> {
  if (file.size > 5 * 1024 * 1024) {
    throw new Error('ICON_TOO_LARGE')
  }
  const buf = new Uint8Array(await file.slice(0, 8).arrayBuffer())
  if (buf.length < PNG_MAGIC.length) throw new Error('ICON_NOT_PNG')
  for (let i = 0; i < PNG_MAGIC.length; i++) {
    if (buf[i] !== PNG_MAGIC[i]) throw new Error('ICON_NOT_PNG')
  }
}

function shouldIgnorePath(relPosix: string): boolean {
  const parts = relPosix.split('/').filter(Boolean)
  for (const p of parts) {
    if (p === '.' || p === '..') return true
    if (p.startsWith('.')) return true
    if (PACK_IGNORE_DIR_NAMES.has(p) || p.endsWith('.egg-info')) return true
    if (p === '.DS_Store') return true
  }
  return false
}

function stripRootFolder(webkitRelativePath: string): string {
  const posix = webkitRelativePath.replace(/\\/g, '/').replace(/^\//, '')
  const parts = posix.split('/').filter(Boolean)
  if (parts.length < 2) return ''
  return parts.slice(1).join('/')
}

function buildSkillMd(name: string, description: string): string {
  const fm = yamlDump(
    { name, description: description.trim() },
    { lineWidth: -1, noRefs: true, quotingType: '"' },
  ).trimEnd()
  return `---\n${fm}\n---\n\n`
}

/**
 * 按市场 skill 包结构打包：`{name}/plugin.yaml`、`icon.png`、`{name}/SKILL.md` 及用户目录内其余文件。
 */
export async function buildSkillPublishZip(input: BuildSkillPublishZipInput): Promise<File> {
  const name = normalizeSkillSlug(input.name)
  const version = normalizeSemver(input.version)
  const displayName = input.displayName.trim()
  if (!displayName || displayName.length > DISPLAY_NAME_MAX_LEN) {
    throw new Error('INVALID_DISPLAY_NAME')
  }
  const description = input.description.trim()
  if (!description || description.length > PLUGIN_YAML_DESCRIPTION_MAX_LEN) {
    throw new Error('INVALID_DESCRIPTION')
  }
  if (description.length > SKILL_DESC_MAX_LEN) {
    throw new Error('INVALID_SKILL_DESC')
  }
  const author = input.authorLogin.trim()
  if (!author) throw new Error('INVALID_AUTHOR')

  await assertPng(input.iconFile)

  const tags = input.tags.length ? input.tags : []
  for (const t of tags) {
    if (!t.trim()) throw new Error('INVALID_TAG')
  }

  const files = input.skillDirectoryFiles
  if (!files.length) throw new Error('NO_SKILL_FILES')

  let hasSkillMd = false
  const entries: { relInSkill: string; file: File }[] = []

  for (const f of files) {
    const wrp = (f as File & { webkitRelativePath?: string }).webkitRelativePath
    if (!wrp || typeof wrp !== 'string') {
      throw new Error('MISSING_RELATIVE_PATH')
    }
    const rel = stripRootFolder(wrp)
    if (!rel) continue
    if (shouldIgnorePath(rel)) continue
    const lower = rel.toLowerCase()
    if (lower.endsWith('.pyc') || lower.endsWith('.pyo')) continue

    if (rel === 'SKILL.md' || rel.endsWith('/SKILL.md')) {
      if (rel !== 'SKILL.md') {
        throw new Error('SKILL_MD_NOT_AT_ROOT')
      }
      hasSkillMd = true
    }
    entries.push({ relInSkill: rel, file: f })
  }

  if (!hasSkillMd) throw new Error('MISSING_SKILL_MD')

  const zipRoot = name
  const inner = `${name}/${name}`

  const yamlDoc = {
    name,
    version,
    display_name: displayName,
    description,
    runtime: { type: 'skill' },
    metadata: {
      author,
      tags,
    },
  }
  const pluginYamlText = yamlDump(yamlDoc, {
    lineWidth: 120,
    noRefs: true,
    quotingType: '"',
    sortKeys: false,
  })

  const skillMdText = buildSkillMd(name, description)

  const zip = new JSZip()
  zip.file(`${zipRoot}/plugin.yaml`, pluginYamlText)
  zip.file(`${zipRoot}/icon.png`, await input.iconFile.arrayBuffer())
  zip.file(`${inner}/SKILL.md`, skillMdText)

  let entryCount = 3
  const seen = new Set<string>([`${inner}/SKILL.md`.toLowerCase()])

  for (const { relInSkill, file } of entries) {
    if (relInSkill === 'SKILL.md') {
      const userBody = await file.text()
      const merged = mergeUserSkillMdBody(skillMdText, userBody)
      zip.file(`${inner}/SKILL.md`, merged)
      continue
    }
    const arc = `${inner}/${relInSkill}`.replace(/\/+/g, '/')
    const key = arc.toLowerCase()
    if (seen.has(key)) continue
    seen.add(key)
    zip.file(arc, await file.arrayBuffer())
    entryCount++
    if (entryCount > MAX_ZIP_ENTRIES) {
      throw new Error('TOO_MANY_ZIP_ENTRIES')
    }
  }

  const blob = await zip.generateAsync({
    type: 'blob',
    compression: 'DEFLATE',
    compressionOptions: { level: 6 },
  })
  return new File([blob], `${name}-${version}.zip`, { type: 'application/zip' })
}

/** 用表单生成的 frontmatter 覆盖用户 SKILL.md，保留其正文（第二个 --- 之后）。 */
function mergeUserSkillMdBody(generatedWithFm: string, userRaw: string): string {
  const text = userRaw.replace(/^\uFEFF/, '')
  let body = ''
  if (text.startsWith('---')) {
    const lines = text.split(/\r?\n/)
    let end = -1
    for (let i = 1; i < lines.length; i++) {
      if (lines[i]?.trim() === '---') {
        end = i
        break
      }
    }
    body = end >= 0 ? lines.slice(end + 1).join('\n') : text
  } else {
    body = text
  }
  const b = body.replace(/^\n+/, '')
  if (!b.trim()) return generatedWithFm
  return generatedWithFm + (b.endsWith('\n') ? b : `${b}\n`)
}

