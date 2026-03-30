"""Azure OpenAI DALL-E 3 文生图工具（逻辑参考 langgenius-azuredalle）。"""

import base64
import random
from typing import Any

from openai import AzureOpenAI

from openjiuwen.core.foundation.tool import tool

from azuredalle_plugin._config import get_azure_dalle_settings

SIZE_MAPPING = {"square": "1024x1024", "vertical": "1024x1792", "horizontal": "1792x1024"}


def _generate_random_id(length: int = 8) -> str:
    characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    return "".join(random.choices(characters, k=length))


def _decode_image(base64_image: str) -> tuple[str, bytes]:
    """
    解码 base64 图片。若无 data:image 前缀则按 PNG 处理。
    """
    if not base64_image.startswith("data:image"):
        return ("image/png", base64.b64decode(base64_image))
    mime_type = base64_image.split(";")[0].split(":")[1]
    image_data_base64 = base64_image.split(",", 1)[1]
    return (mime_type, base64.b64decode(image_data_base64))


@tool(
    name="azure_dalle3",
    description=(
        "使用 Azure OpenAI 上部署的 DALL-E 3 根据文本提示生成图像。"
        "DALL-E 3 text-to-image via Azure OpenAI deployment; returns image as base64 in report/raw."
    ),
    input_params={
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": (
                    "图像提示词，尽量具体描述画面内容、风格与构图。"
                    "Image prompt; describe subject, style, and composition in detail."
                ),
            },
            "seed_id": {
                "type": "string",
                "description": (
                    "可选。8 位左右字母数字种子，用于系列图一致性。"
                    "Optional seed string for reproducibility across a series."
                ),
            },
            "size": {
                "type": "string",
                "description": "画幅：square 1024x1024、vertical 1024x1792、horizontal 1792x1024。",
                "enum": ["square", "vertical", "horizontal"],
                "default": "square",
            },
            "quality": {
                "type": "string",
                "description": "图像质量：standard 或 hd。",
                "enum": ["standard", "hd"],
                "default": "standard",
            },
            "style": {
                "type": "string",
                "description": "风格：vivid 更饱和，natural 更写实。",
                "enum": ["vivid", "natural"],
                "default": "vivid",
            },
        },
        "required": ["prompt"],
    },
)
def azure_dalle3(params: dict[str, Any] | None = None, **kwargs) -> dict:
    """
    调用 Azure DALL-E 3 生成图像，返回 Markdown 报告（内嵌 data URL）与 raw 结构。
    """
    params = params or kwargs
    prompt = (params.get("prompt") or "").strip()
    if not prompt:
        return {"error": "请输入提示词 prompt。"}

    creds, cfg_err = get_azure_dalle_settings()
    if cfg_err or not creds:
        return {"error": cfg_err or "Azure OpenAI 配置无效。"}

    size_key = params.get("size") or "square"
    if size_key not in SIZE_MAPPING:
        return {"error": f"无效的 size：{size_key}，应为 square / vertical / horizontal。"}
    size = SIZE_MAPPING[size_key]

    quality = params.get("quality") or "standard"
    if quality not in ("standard", "hd"):
        return {"error": "quality 必须为 standard 或 hd。"}

    style = params.get("style") or "vivid"
    if style not in ("natural", "vivid"):
        return {"error": "style 必须为 natural 或 vivid。"}

    seed_id = (params.get("seed_id") or "").strip() or _generate_random_id(8)
    extra_body: dict[str, Any] = {"seed": seed_id}

    client = AzureOpenAI(
        api_version=creds["azure_openai_api_version"],
        azure_endpoint=creds["azure_openai_base_url"],
        api_key=creds["azure_openai_api_key"],
    )
    model = creds["azure_openai_api_model_name"]

    try:
        response = client.images.generate(
            prompt=prompt,
            model=model,
            size=size,  # type: ignore[arg-type]
            n=1,
            extra_body=extra_body,
            style=style,
            quality=quality,
            response_format="b64_json",
        )
    except Exception as e:
        return {"error": f"调用 Azure DALL-E 失败：{e!s}"}

    if not response.data:
        return {"error": "API 未返回图像数据。", "raw": {}}

    first = response.data[0]
    b64 = getattr(first, "b64_json", None)
    if not b64:
        return {"error": "响应中缺少 b64_json。", "raw": {}}

    mime_type, image_bytes = _decode_image(b64)
    b64_out = base64.b64encode(image_bytes).decode("ascii")
    revised = getattr(first, "revised_prompt", None)

    report_lines = [
        "**Azure DALL-E 3 生成成功**",
        f"- **seed_id**: `{seed_id}`",
    ]
    if revised:
        report_lines.append(f"- **revised_prompt**: {revised}")
    report_lines.append("")
    report_lines.append(f"![generated](data:{mime_type};base64,{b64_out})")

    raw: dict[str, Any] = {
        "mime_type": mime_type,
        "base64": b64_out,
        "seed_id": seed_id,
        "size": size,
        "quality": quality,
        "style": style,
    }
    if revised:
        raw["revised_prompt"] = revised

    return {"report": "\n".join(report_lines), "raw": raw}
