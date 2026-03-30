# Firecrawl Plugin（OpenJiuwen）

基于官方 Python SDK [**firecrawl-py**](https://pypi.org/project/firecrawl-py/)：统一客户端 `Firecrawl` 的 **V2 API**（`scrape` / `map` / `crawl`）。爬取仅支持同步 **`crawl`**（等待完成后返回），不提供 `start_crawl` 与按 job 查询/取消工具。

## 功能

| 工具 | 说明 |
|------|------|
| **firecrawl_scrape** | V2 `scrape`，默认 `formats=markdown`，完整响应在 `raw` |
| **firecrawl_map** | V2 `map`（`sitemap=skip` 对应 `ignoreSitemap=true`） |
| **firecrawl_crawl** | V2 `crawl`，轮询直至任务完成后再返回 |

## 配置

### 1. 获取 API Key

在 [Firecrawl API Keys](https://www.firecrawl.dev/app/api-keys) 创建密钥。若使用**自托管**，密钥可填任意非空字符串。

### 2. 环境变量

```bash
export firecrawl_api_key="你的 API Key"
```

可选：自定义 API 根地址（默认 `https://api.firecrawl.dev`）：

```bash
export firecrawl_base_url="https://your-self-hosted.example.com"
```

### 3. 安装

会安装依赖 `firecrawl-py`（与 `pip install firecrawl-py` 相同包名）。

```bash
cd plugins/firecrawl_plugin
pip install -e .
```

## 返回值约定

- 成功：`{"report": <str>, "raw": <dict>}` — `firecrawl_scrape` 的 `report` 为页面 Markdown（若 API 未返回则为空字符串）；其余工具的 `report` 为格式化的 JSON 文本摘要。
- 失败：`{"error": <str>}`

## 参数说明（概要）

### firecrawl_scrape

- **url**（必填）
- **formats**：逗号分隔，如 `markdown,links`
- **onlyMainContent**：默认 `true`（与参考插件 Python 实现一致）
- **includeTags** / **excludeTags**：逗号分隔
- **headers** / **schema**：JSON 字符串
- **waitFor** / **timeout**：毫秒
- **systemPrompt** / **prompt**：结构化提取用

### firecrawl_map

- **url**（必填）
- **search**：可选关键词
- **ignoreSitemap**（默认 true）、**includeSubdomains**（默认 false）、**limit**（默认 5000）

### firecrawl_crawl

- **url**（必填）
- **poll_interval**：轮询任务状态的间隔（秒），默认 5
- **excludePaths** / **includePaths**：逗号分隔模式
- **maxDepth**：映射为 V2 **`max_discovery_depth`**（发现链接深度）
- **limit**、**ignoreSitemap**、**allowBackwardLinks**（→ `crawl_entire_domain`）、**allowExternalLinks**、**webhook**
- **formats**、**headers**、**includeTags**、**excludeTags**、**onlyMainContent**、**waitFor**：子页面 **`ScrapeOptions`**

## 参考

- [Firecrawl 文档](https://docs.firecrawl.dev/)
- [firecrawl-py（PyPI）](https://pypi.org/project/firecrawl-py/)
