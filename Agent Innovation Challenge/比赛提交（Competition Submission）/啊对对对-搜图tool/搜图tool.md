# OpenJiuWen 反向图片搜索工具

## 📌 工具简介
本工具为 OpenJiuWen 定制开发，基于 SerpAPI 实现**Google 反向图片搜索**功能，可输入图片 URL 检索相似图片，并适配 OpenJiuWen 的消息格式（文本/图片结构化返回），支持多轮重试、URL 可达性检查、结果去重等特性。

## ✨ 核心功能
1. **反向相似图片搜索**：输入图片 URL，获取相似图片列表（含图片 URL、标题、来源等信息）；
2. **消息格式适配**：按 `ContentItem` 结构返回文本/图片，兼容 OpenJiuWen 消息交互逻辑；
3. **鲁棒性保障**：支持请求重试、URL 可达性校验、结果去重，强制 IPv4 访问避免网络问题；
4. **简化调用**：提供 `message_search` 封装函数，降低参数配置和调用门槛。

## 🛠 环境准备
### 1. 依赖安装
```bash
pip install requests pydantic
```
2. API Key 配置
申请 SerpAPI Key：从 serper.dev 申请 API Key；
配置方式（二选一）：
环境变量配置（推荐）：
```bash

# Linux/Mac
export SERPAPI_IMAGE_SEARCH_KEY="你的API Key"
# Windows
set SERPAPI_IMAGE_SEARCH_KEY="你的API Key"
```
直接修改代码：将代码中 SERPAPI_IMAGE_SEARCH_KEY = "8888888" 替换为实际 Key。
### 🚀 快速使用
基础调用示例
```python

# 引入工具函数（替换为实际文件名）
from image_search_tool import message_search

# 测试图片URL（需替换为公开可访问的图片URL）
test_image_url = "https://img10.360buyimg.com/n4/s330x330_jfs/t1/359786/35/3306/92151/6909cd6eFe97c7548/23b08acbaadc3464.jpg"

# 调用反向图片搜索
try:
    result = message_search(
        img_url=test_image_url,
        search_text="电子产品 商品图片 相似款",  # 辅助搜索文本（可选）
        hl="zh-CN",  # 搜索语言
        gl="cn",     # 搜索地区
        no_cache=False,  # 是否禁用缓存
        num=20       # 返回结果数量
    )

    # 打印结果
    print("\n" + "=" * 80)
    print("【反向图片搜索结果】")
    print("-" * 80)
    for idx, item in enumerate(result):
        if item.text:
            print(f"[{idx+1}] 文本信息：{item.text}")
        if item.image:
            print(f"[{idx+1}] 相似图片URL：{item.image}")
        print("-" * 80)
except Exception as e:
    print(f"搜索失败：{str(e)}")
```

工具类调用示例
```python
from image_search_tool import ReverseImageSearch, Message, ContentItem

# 构造消息体
messages = [
    Message(
        role="user",
        content=[
            ContentItem(image="https://xxx.com/test.jpg"),  # 图片URL
            ContentItem(text="辅助搜索文本")  # 可选
        ]
    )
]

# 初始化工具并调用
tool = ReverseImageSearch()
params = {"img_idx": 0}  # 指定消息中第1张图片（索引从0开始）
result = tool.call(params=params, messages=messages)

# 输出结果
for item in result:
    if item.text:
        print(f"文本：{item.text}")
    if item.image:
        print(f"图片URL：{item.image}")
```
### 核心类/函数说明
| 类 / 函数 | 功能说明 |
|-----------|----------|
| ContentItem | 消息内容模型，封装文本（text）和图片 URL（image）字段 |
| Message | 消息体模型，包含角色（role）和内容列表（content） |
| ImageResult | 图片搜索结果模型，封装图片 URL、标题、尺寸、来源等信息 |
| ReverseImageSearch | 核心工具类，适配 OpenJiuWen 工具规范，提供call方法处理搜索逻辑 |
| serper_reverse_image_search | 底层反向搜索函数，处理请求发送、结果解析、重试逻辑 |
| message_search | 简化封装函数，适配普通调用场景，内置参数校验 |
| check_image_url_accessibility | 检查图片 URL 是否可访问，返回可达性状态 |
| extract_images_from_messages | 从消息列表中提取所有图片 URL |

### ⚙️ 配置参数说明
| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| img_url | str | - | 待搜索的图片 URL（必填，需公开可访问） |
| search_text | str | "" | 辅助搜索的文本内容 |
| hl | str | "zh-CN" | 搜索语言（如 "en-US" 表示英文） |
| gl | str | "cn" | 搜索地区（如 "us" 表示美国） |
| no_cache | bool | True | 是否禁用 SerpAPI 缓存 |
| num | int | 10 | 返回相似图片的数量 |
| max_retry | int | 3 | 请求最大重试次数（可通过环境变量QWEN_IMAGE_SEARCH_MAX_RETRY_TIMES调整） |