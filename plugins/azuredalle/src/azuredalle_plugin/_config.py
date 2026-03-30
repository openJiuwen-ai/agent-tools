"""从环境变量读取 Azure OpenAI（DALL-E）配置。"""

import os
from typing import Any


def _first_nonempty(*keys: str) -> str | None:
    for k in keys:
        v = os.getenv(k)
        if v is not None and str(v).strip():
            return str(v).strip()
    return None


def get_azure_dalle_settings() -> tuple[dict[str, str] | None, str | None]:
    """
    返回 (credentials 字典, 错误信息)。
    credentials 键与 Dify langgenius-azuredalle 凭据命名一致，便于迁移。
    """
    api_key = _first_nonempty("azure_openai_api_key")
    base_url = _first_nonempty("azure_openai_base_url")
    api_version = _first_nonempty("azure_openai_api_version")
    deployment = _first_nonempty("azure_openai_api_model_name")

    missing: list[str] = []
    if not api_key:
        missing.append("azure_openai_api_key")
    if not base_url:
        missing.append("azure_openai_base_url")
    if not api_version:
        missing.append("azure_openai_api_version")
    if not deployment:
        missing.append("azure_openai_api_model_name（DALL-E 部署名称）")

    if missing:
        return None, "缺少 Azure OpenAI 配置：" + "、".join(missing) + "。"

    # 与 Azure SDK 一致：endpoint 不要尾随斜杠
    endpoint = base_url.rstrip("/")

    creds: dict[str, str] = {
        "azure_openai_api_key": api_key,
        "azure_openai_base_url": endpoint,
        "azure_openai_api_version": api_version,
        "azure_openai_api_model_name": deployment,
    }
    return creds, None
