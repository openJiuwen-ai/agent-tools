# Troubleshooting

> Auto-generated from evolutions.json. Do not edit directly.

### [ev_6909ec7f] ## 非UTF-8编码日志文件的读取处理
- 采集包中的 MANIFEST.txt 和 desktop-launcher.log 在中文 Windows 系统上常为 GBK/GB2312 编码，read_file 工具默认 UTF-8 会报解码错误（如 `0xc8 invalid continuation byte`）
- 遇到 UTF-8 解码失败时，改用 bash 工具读取：PowerShell 用 `Get-Content -Encoding Default`，Linux/macOS 用 `iconv -f GBK -t UTF-8` 转换后再读取
- 优先对 MANIFEST.txt 采用此策略，因为它是最先读取的文件且编码问题最常见

*Source: conversation_review | 2026-05-12T09:09:21.013803+00:00*

---
