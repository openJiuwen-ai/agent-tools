# Jina Plugin

基于 [Jina AI](https://jina.ai/) 的 Reader / Search 服务，为 openJiuwen 提供**网页读取（含 PDF）**与**联网搜索**能力。

## 功能

| 工具 | 说明 |
|------|------|
| **jina_reader** | 读取单个 URL（网页或 PDF），返回适合大模型处理的文本/Markdown |
| **jina_search** | 联网搜索，返回适合大模型处理的结果文本 |

## 配置

### 1. 获取（可选）API Key

在 [Jina AI](https://jina.ai/) 获取 API Key（可选；不填也可能可用，但通常速率/额度更低）。

### 2. 配置环境变量

在运行 openJiuwen 的环境中设置：

```bash
export jina_api_key="你的 API Key"
```

### 3. 安装插件

在插件目录下安装依赖：

```bash
cd plugins/jina_plugin
pip install -e .
```

## 参数说明

### jina_reader

- **url**（必填）：目标网页或 PDF 的 URL
- **request_params**：可选，请求 querystring 参数（JSON 字符串），例如 `{"key":"value"}`
- **target_selector**：可选，CSS 选择器，仅抓取匹配元素
- **wait_for_selector**：可选，等待某元素出现后再抓取
- **remove_images**：可选，移除图片（默认 false）
- **image_caption**：可选，为图片生成说明（默认 false）
- **gather_all_links_at_the_end**：可选，在末尾汇总链接（默认 false）
- **gather_all_images_at_the_end**：可选，在末尾汇总图片（默认 false）
- **proxy_server**：可选，代理 URL
- **no_cache**：可选，绕过缓存（默认 false）
- **no_cache_track**：可选，不缓存/不追踪（默认 false）
- **content_format**：可选，`default`/`markdown`/`html`/`text`/`screenshot`/`pageshot`（默认 `default`）

### jina_search

- **query**（必填）：搜索查询（问题/关键词）
- **image_caption**：可选，为图片生成说明（默认 false）
- **gather_all_links_at_the_end**：可选，在末尾汇总链接（默认 false）
- **gather_all_images_at_the_end**：可选，在末尾汇总图片（默认 false）
- **proxy_server**：可选，代理 URL
- **no_cache**：可选，绕过缓存（默认 false）

## 返回值约定

两个工具都会返回一个 dict：

- 成功时：`{"report": <string>, "status_code": <int>}`
- 失败时：`{"error": <string>}`

## 常用场景

- **网页内容抓取与总结**：先用 `jina_reader` 拉取页面正文（可用 `content_format=markdown`），再交给模型总结/抽取要点
- **事实核查/信息补全**：用 `jina_search` 找到相关来源，再对关键链接用 `jina_reader` 拉取全文进行交叉验证

## 参考

- Jina Reader: `https://r.jina.ai/`
- Jina Search: `https://s.jina.ai/`
