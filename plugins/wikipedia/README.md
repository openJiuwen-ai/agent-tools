# Wikipedia Plugin（OpenJiuwen）

基于 [wikipedia](https://pypi.org/project/wikipedia/) 的维基百科搜索插件，为 openJiuwen 提供百科知识查询能力。无需 API Key。

## 功能

| 工具 | 说明 |
|------|------|
| **wikipedia_search** | 在维基百科中搜索并返回页面摘要，支持多语言 |

## 安装

```bash
cd plugins/wikipedia
pip install -e .
```

## 参数说明

- **query**（必填）：搜索关键词
- **language**：维基百科语言代码，默认 `en`。常用：`en` 英语、`zh` 中文、`ja` 日语、`de` 德语、`fr` 法语、`ko` 韩语等（见 [Wikipedia 语言列表](https://en.wikipedia.org/wiki/List_of_Wikipedias)）
- **top_k_results**：返回条目数量上限（1–10），默认 3
- **doc_content_chars_max**：返回内容最大字符数（500–8000），默认 4000
