from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Store 服务配置。"""

    app_name: str = "Store Service"
    app_version: str = "0.1.0"
    debug: bool = True

    host: str = "0.0.0.0"
    port: int = 8100

    db_url: str = ""

    # 鉴权：系统管理员 token，与请求头 X-System-Token 比对（环境变量 SYSTEM_ADMIN_TOKEN）
    system_admin_token: str = Field(default="", validation_alias="SYSTEM_ADMIN_TOKEN")
    # 系统管理员用户标识（管理员请求时作为 publisher_id / user_id 写入）；默认 system_admin
    system_admin_user: str = Field(default="system_admin", validation_alias="SYSTEM_ADMIN_USER")
    # 用户信息接口（默认 GitCode https://gitcode.com/api/v5/user，使用 query access_token）
    auth_user_api_url: str = Field(default="https://gitcode.com/api/v5/user", validation_alias="AUTH_USER_API_URL")

    @field_validator("system_admin_user", mode="before")
    @classmethod
    def _normalize_system_admin_user(cls, v: object) -> str:
        s = "" if v is None else str(v).strip()
        return s or "system_admin"

    class Config:
        env_prefix = "MARKET_"
        env_file = "../../../.env"
        case_sensitive = False
        extra = "ignore"


settings = Settings()

