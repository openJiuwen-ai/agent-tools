/** 对文件内容做 SHA-256，返回 64 位小写十六进制（与后端 X-Checksum-SHA256 一致）。 */
export async function sha256HexOfFile(file: File): Promise<string> {
  const buf = await file.arrayBuffer()
  const digest = await crypto.subtle.digest('SHA-256', buf)
  const bytes = new Uint8Array(digest)
  let hex = ''
  for (let i = 0; i < bytes.length; i++) {
    hex += bytes[i]!.toString(16).padStart(2, '0')
  }
  return hex
}
