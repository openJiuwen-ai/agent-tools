# Google Plugin（OpenJiuwen）

基于 [SerpAPI](https://serpapi.com/) 的 Google 搜索插件，为 openJiuwen 提供网页搜索与图片搜索能力。

## 功能

| 工具 | 说明 |
|------|------|
| **google_search** | Google 网页搜索，返回标题、链接与摘要片段 |
| **google_image_search** | Google 图片搜索，返回图片 URL 与缩略图 |

## 配置

### 1. 获取 SerpAPI API Key

在 [SerpAPI Manage API Key](https://serpapi.com/manage-api-key) 注册并获取 API Key。

### 2. 配置环境变量

在运行 openJiuwen 的环境中设置：

```bash
export serpapi_api_key="你的 API Key"
```

### 3. 安装插件

在插件目录下安装依赖：

```bash
cd plugins/google
pip install -e .
```

## 参数说明

### google_search

- **query**（必填）：搜索关键词
- **hl**：界面语言代码，默认 `en`，见 [Google 语言列表](https://serpapi.com/google-languages)
- **gl**：国家/地区代码，默认 `us`，见 [Google 国家列表](https://serpapi.com/google-countries)
- **location**：可选，搜索发起位置（如城市名）

### google_image_search

- **query**（必填）：图片搜索关键词
- **hl** / **gl** / **location**：同上
- **max_results**：返回图片数量上限（1–20），默认 3

## 参考

- [SerpAPI 文档](https://serpapi.com/search-api)
