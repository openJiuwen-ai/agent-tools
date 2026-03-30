# Tavily Plugin（openJiuwen）

基于 [Tavily API](https://docs.tavily.com/) 的搜索与内容提取插件，为 openJiuwen 提供 AI 友好的网页搜索、URL 内容提取、站点结构映射与深度研究能力。Tavily 专为 LLM 设计，返回结构化、可直接用于推理的结果。

## 功能

| 工具 | 说明 |
|------|------|
| **tavily_search** | 网页搜索：支持 basic/advanced 深度、主题（general/news/finance）、时间范围与域名过滤，可返回摘要答案与原始内容 |
| **tavily_extract** | 从指定 URL 提取网页正文，支持 basic/advanced 深度与 markdown/text 格式 |
| **tavily_map** | 从根 URL 发现并列出站点内链接，可配置深度、广度、路径/域名过滤与超时 |
| **tavily_research** | 对给定问题执行多步深度研究，返回带引用与来源的综合报告（异步轮询完成） |

## 配置

### 1. 获取 API Key

在 [Tavily 控制台](https://app.tavily.com) 注册并创建 API Key。免费档约 1,000 次调用/月。

### 2. 配置环境变量

在运行 openJiuwen 的环境中设置：

```bash
export tavily_api_key="你的 API Key"
```

### 3. 安装插件

在插件目录下安装依赖：

```bash
cd plugins/tavily
pip install -e .
```

## 参数说明

### tavily_search

- **query**（必填）：搜索关键词
- **search_depth**：`basic`（默认）、`advanced`、`fast`、`ultra-fast`
- **topic**：`general`（默认）、`news`、`finance`
- **time_range**：`not_specified`、`day`、`week`、`month`、`year`
- **include_answer**：是否返回简短答案，可选 `false`、`true`/`basic`、`advanced`
- **include_raw_content**：是否返回每条结果的原始内容，可选 `false`、`true`/`markdown`、`text`
- **include_domains** / **exclude_domains**：逗号或空格分隔的域名列表
- **max_results**：结果数量（1–20），默认 5
- **country**：优先某国家/地区结果（如 `china`、`us`），仅 topic 为 general 时有效

### tavily_extract

- **urls**（必填）：要提取的 URL，多个可用逗号分隔
- **query**：可选，用于对提取片段按相关性重排
- **extract_depth**：`basic`（默认）、`advanced`
- **format**：`markdown`（默认）、`text`
- **chunks_per_source**：每站返回片段数（1–5），默认 3
- **timeout**：请求超时秒数（1–60）

### tavily_map

- **url**（必填）：起始根 URL
- **max_depth**：探索深度（1–5），默认 1
- **max_breadth**：每层跟踪链接数（1–500），默认 20
- **limit**：总链接数上限，默认 50
- **select_paths** / **exclude_paths**：按路径过滤的逗号分隔正则
- **select_domains** / **exclude_domains**：按域名过滤的逗号分隔正则
- **allow_external**：是否包含站外链接，默认 true
- **timeout**：最大等待秒数（10–150），默认 150

### tavily_research

- **input**（必填）：研究问题或任务描述，越具体效果越好
- **model**：`auto`（默认）、`mini`（快速）、`pro`（全面）
- **citation_format**：引用格式 `numbered`、`mla`、`apa`、`chicago`，默认 `numbered`

## 常用场景

- **事实核查**：用户提问 → `tavily_search`（`include_answer: true`）→ LLM 校验并回答
- **内容聚合**：主题 → `tavily_search` → 对前几条 URL 调用 `tavily_extract` → LLM 汇总
- **竞品/站点分析**：竞品根 URL → `tavily_map` 发现链接 → 对关键页 `tavily_extract` → LLM 分析
- **深度调研**：复杂问题 → `tavily_research` 获取带引用的报告 → LLM 提炼或二次加工

## 参考

- [Tavily API 文档](https://docs.tavily.com/documentation/api-reference/introduction)
- [Tavily 控制台](https://app.tavily.com)
- [Tavily Discord](https://discord.gg/TPu2gkaWp2)
