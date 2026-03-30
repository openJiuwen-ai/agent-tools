# Azure DALL-E Plugin（openJiuwen）

基于 [Azure OpenAI DALL-E 3](https://learn.microsoft.com/azure/ai-services/openai/concepts/models) 的文生图插件。

## 功能

| 工具 | 说明 |
|------|------|
| **azure_dalle3** | 根据文本提示生成图像；支持方图/竖图/横图、standard/hd、vivid/natural，可选 `seed_id`；返回 Markdown（内嵌 data URL）及 `raw` 中的 base64 |

## 配置

### 1. Azure 资源

在 Azure 创建 OpenAI 资源，部署 **DALL-E 3** 模型，记录：

- API Key  
- Endpoint（如 `https://xxx.openai.azure.com`）  
- API 版本（如 `2024-02-01` 或门户推荐的预览版本）  
- **部署名称**（Deployment name）

### 2. 环境变量

```bash
export azure_openai_api_key="你的 API Key"
export azure_openai_base_url="https://你的资源名.openai.azure.com"
export azure_openai_api_version="2024-02-01"
export azure_openai_api_model_name="你的-DALL-E-部署名称"
```

### 3. 安装插件

```bash
cd plugins/azuredalle_plugin
pip install -e .
```

## 参数说明（azure_dalle3）

- **prompt**（必填）：图像描述提示词  
- **seed_id**（可选）：种子字符串；不传则自动生成 8 位字母数字  
- **size**：`square`（默认）、`vertical`、`horizontal`  
- **quality**：`standard`（默认）、`hd`  
- **style**：`vivid`（默认）、`natural`  

## 返回值

- **report**：Markdown，含可选 `revised_prompt` 说明与 `![generated](data:...;base64,...)`  
- **raw**：`mime_type`、`base64`、`seed_id`、`size`、`quality`、`style`，以及可能存在的 `revised_prompt`  
- 失败时：**error** 字段说明原因  

## 参考

- [Azure DALL-E 快速入门](https://learn.microsoft.com/azure/ai-services/openai/dall-e-quickstart)
