# WebScraperTool（单工具版）

## 1. Tool 应用场景

### 1.1 核心场景定义

WebScraperTool 是一个**通用网页结构化抽取 Tool**。输入一个网页 URL，并提供抽取规则（CSS 选择器 / 正则），工具会在服务端内部完成网页抓取并输出结构化结果。

该 Tool 设计目标是：

- 在 Agent Studio 中调用稳定
- 避免把整段 HTML 作为参数传入（容易导致工具调用参数解析失败）
- 通过可配置规则适配多站点抽取

### 1.2 具体应用场景

| 场景类别 | 典型用户需求 | 工具价值 |
| :-- | :-- | :-- |
| **信息采集** | 抓取榜单/列表页标题、链接、评分等 | 自动化获取结构化数据 |
| **内容整理** | 把网页内容提炼为字段（标题/摘要/作者/时间） | 便于后续总结/检索 |
| **Agent 外挂能力** | Agent 需要从网页中取“可信依据”并引用 | 结果可追溯、可复用 |

## 2. Agent Tool 技术方案

### 2.1 工具定义（标准化接口）

说明：以下是用于“外部说明/方案文档”的标准化定义；在 Agent Studio 中创建 URL 插件时，仍以“方法 + 路径 + 入参/出参”方式配置。

```
{
  "tool_name": "web_extract",
  "description": "输入网页URL与抽取规则（CSS/正则），服务端抓取页面并返回结构化字段（避免传大段HTML导致解析失败）",
  "version": "1.0.0",
  "parameters": {
    "type": "object",
    "properties": {
      "url": {
        "type": "string",
        "description": "目标网页URL（必填）"
      },
      "css": {
        "type": "string",
        "description": "CSS抽取规则JSON字符串（数组）。每条规则包含 name/selector/attr/multiple/text_join"
      },
      "regex": {
        "type": "string",
        "description": "正则抽取规则JSON字符串（数组）。每条规则包含 name/pattern/group/multiple"
      }
    },
    "required": ["url"]
  },
  "returns": {
    "type": "object",
    "properties": {
      "success": {"type": "boolean", "description": "是否成功"},
      "data": {"type": "object", "description": "抽取结果对象（key=规则name）"},
      "error": {"type": "string", "description": "失败原因（success=false时）"}
    }
  }
}
```

### 2.2 服务形态与接口

- 服务：FastAPI
- 健康检查：`GET /system/health`
- 工具接口：`GET /scrape/extract`（推荐）
  - 同时支持 `POST /scrape/extract`

## 3. 部署与启动

### 3.1 安装

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

### 3.2 启动

```bash
python run.py
```

默认：

- Host：`0.0.0.0`
- Port：`18086`

文档：

- `http://127.0.0.1:18086/docs`
- `http://127.0.0.1:18086/openapi.json`

## 4. Agent Studio（URL 插件）接入配置（最终版：只保留一个 Tool）

### 4.1 创建 URL 插件

- **插件类型**：API / URL 插件（CLOUD_API）
- **服务地址**：`http://127.0.0.1:18086`
- **Headers**：先保持为空（不建议提前加自定义头）

### 4.2 Tool 列表（只创建 1 个）

#### Tool 基本信息

- **Tool Name**：`web_extract`
- **Method**：`GET`
- **Path**：`/scrape/extract`

#### Tool 描述（直接复制）

输入网页URL与抽取规则（CSS/正则），服务端抓取页面并返回结构化字段。为避免工具参数解析失败，不需要传入整段HTML。

#### 入参配置（Input Parameters）

- `url`
  - 类型：string
  - 必填：是
  - 描述（复制）：目标网页URL（必填）。
- `css`
  - 类型：string
  - 必填：否
  - 描述（复制）：CSS抽取规则JSON字符串数组。每条规则包含 name/selector/attr/multiple/text_join。
- `regex`
  - 类型：string
  - 必填：否
  - 描述（复制）：正则抽取规则JSON字符串数组。每条规则包含 name/pattern/group/multiple。

#### 出参配置（Output Parameters）

- `success`
  - 类型：boolean
  - 描述（复制）：是否成功。
- `data`
  - 类型：object
  - 描述（复制）：抽取结果对象（key=规则name）。
- `error`
  - 类型：string
  - 描述（复制）：失败原因（success=false时）。

## 5. 使用示例

### 5.1 CSS 规则格式

CSS 规则是一个数组，每个元素：

```json
{
  "name": "field_name",
  "selector": "CSS selector",
  "attr": "href",
  "multiple": true,
  "text_join": " "
}
```

### 5.2 示例：豆瓣电影 Top250

- `url`：`https://movie.douban.com/top250`

`css`（作为 JSON 字符串粘贴到入参中）：

```json
[
  {"name":"titles","selector":".grid_view .item .info .hd a span:nth-child(1)","multiple":true},
  {"name":"detail_urls","selector":".grid_view .item .info .hd a","attr":"href","multiple":true},
  {"name":"ratings","selector":".grid_view .item .info .bd .star .rating_num","multiple":true}
]
```

返回示例：

```json
{
  "success": true,
  "data": {
    "titles": ["肖申克的救赎", "霸王别姬"],
    "detail_urls": ["https://movie.douban.com/subject/..."],
    "ratings": ["9.7", "9.6"]
  },
  "error": null
}
```

