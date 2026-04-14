import { sha256 } from '@noble/hashes/sha2'

/** 对文件内容做 SHA-256，返回 64 位小写十六进制（与后端 X-Checksum-SHA256 一致）。 */
function bytesToHex(bytes: Uint8Array): string {
  let hex = ''
  for (let i = 0; i < bytes.length; i++) {
    hex += bytes[i]!.toString(16).padStart(2, '0')
  }
  return hex
}

/**
 * `crypto.subtle` 仅在安全上下文可用（HTTPS，或 localhost / 127.0.0.1 等例外）。
 * 云上若用「内网 IP + HTTP」打开前端，浏览器不提供 subtle，原实现会整段失败；
 * 此处用纯 JS 实现回退，结果与 Web Crypto 一致。
 */
export async function sha256HexOfFile(file: File): Promise<string> {
  const buf = await file.arrayBuffer()
  const bytes = new Uint8Array(buf)
  if (globalThis.isSecureContext && globalThis.crypto?.subtle) {
    try {
      const digest = await crypto.subtle.digest('SHA-256', buf)
      return bytesToHex(new Uint8Array(digest))
    } catch {
      /* 极少数环境（如部分隐私模式策略）subtle 存在但 digest 失败 */
    }
  }
  return bytesToHex(sha256(bytes))
}
