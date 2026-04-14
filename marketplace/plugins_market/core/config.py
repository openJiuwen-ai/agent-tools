from pydantic import AliasChoices, Field, field_validator
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

    # OAuth 回调后浏览器重定向到此前缀下的 /login?oauth_session=...（须与前端挂载路径一致，默认含 /hub）
    oauth_frontend_origin: str = Field(
        default="http://localhost:9002/hub",
        validation_alias=AliasChoices("MARKET_OAUTH_FRONTEND_ORIGIN", "OAUTH_FRONTEND_ORIGIN"),
    )

    # 可选：Redis（多 worker / 多实例时必须配置，否则 OAuth pending 仅存内存）；REDIS_HOST 为空则仅用进程内存
    redis_host: str = Field(default="", validation_alias=AliasChoices("MARKET_REDIS_HOST", "REDIS_HOST"))
    redis_port: int = Field(default=6379, validation_alias=AliasChoices("MARKET_REDIS_PORT", "REDIS_PORT"))
    redis_db: int = Field(default=0, validation_alias=AliasChoices("MARKET_REDIS_DB", "REDIS_DB"))
    redis_password: str = Field(default="", validation_alias=AliasChoices("MARKET_REDIS_PASSWORD", "REDIS_PASSWORD"))

    # GitCode OAuth2（应用回调 URL 须与 gitcode_oauth_redirect_uri 完全一致）
    gitcode_oauth_enabled: bool = Field(
        default=False,
        validation_alias=AliasChoices("MARKET_GITCODE_OAUTH_ENABLED", "GITCODE_OAUTH_ENABLED"),
    )
    gitcode_oauth_client_id: str = Field(
        default="",
        validation_alias=AliasChoices("MARKET_GITCODE_OAUTH_CLIENT_ID", "GITCODE_OAUTH_CLIENT_ID"),
    )
    gitcode_oauth_client_secret: str = Field(
        default="",
        validation_alias=AliasChoices("MARKET_GITCODE_OAUTH_CLIENT_SECRET", "GITCODE_OAUTH_CLIENT_SECRET"),
    )
    gitcode_oauth_redirect_uri: str = Field(
        default="",
        validation_alias=AliasChoices("MARKET_GITCODE_OAUTH_REDIRECT_URI", "GITCODE_OAUTH_REDIRECT_URI"),
    )
    gitcode_oauth_scope: str = Field(
        default="user_info",
        validation_alias=AliasChoices("MARKET_GITCODE_OAUTH_SCOPE", "GITCODE_OAUTH_SCOPE"),
    )
    gitcode_oauth_authorize_url: str = Field(
        default="https://gitcode.com/oauth/authorize",
        validation_alias=AliasChoices("MARKET_GITCODE_OAUTH_AUTHORIZE_URL", "GITCODE_OAUTH_AUTHORIZE_URL"),
    )
    gitcode_oauth_token_url: str = Field(
        default="https://gitcode.com/oauth/token",
        validation_alias=AliasChoices("MARKET_GITCODE_OAUTH_TOKEN_URL", "GITCODE_OAUTH_TOKEN_URL"),
    )

    # 发布页「下载模板」zip：桶内对象 Key（私有桶）；为空则 GET /plugins/publish-template 返回 503
    # 仅读取环境变量 MARKET_PLUGIN_TEMPLATE_OBJECT_KEY（与类上 env_prefix 拼接字段名）
    plugin_template_object_key: str = Field(default="")

    # 发布页「下载 Skill 模板」zip：桶内对象 Key；为空则 kind=skill 时 GET /plugins/publish-template 返回 503
    # 仅读取 MARKET_SKILL_TEMPLATE_OBJECT_KEY
    skill_template_object_key: str = Field(default="")

    # skill-import：单进程滑动窗口限流（每分钟请求数，0 关闭）；多 worker 时各进程独立计数
    skill_import_rate_limit_per_minute: int = Field(
        default=30,
        ge=0,
        validation_alias=AliasChoices(
            "MARKET_SKILL_IMPORT_RATE_LIMIT_PER_MINUTE",
            "SKILL_IMPORT_RATE_LIMIT_PER_MINUTE",
        ),
    )

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

