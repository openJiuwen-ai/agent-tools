from pydantic import Field, computed_field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Store 服务配置。"""

    app_name: str = "Store Service"
    app_version: str = "0.1.0"
    debug: bool = True

    host: str = "0.0.0.0"
    port: int = 8100

    db_url: str = "sqlite:///./data/store.db"

    # 鉴权：鉴权服务地址（环境变量 AUTH_SERVICE_HOST / AUTH_SERVICE_PORT）
    auth_service_host: str = Field(default="localhost", validation_alias="AUTH_SERVICE_HOST")
    auth_service_port: int = Field(default=8000, validation_alias="AUTH_SERVICE_PORT")
    # 鉴权：系统管理员 token，与请求头 X-System-Token 比对（环境变量 SYSTEM_ADMIN_TOKEN）
    system_admin_token: str = Field(default="", validation_alias="SYSTEM_ADMIN_TOKEN")

    @computed_field
    @property
    def auth_service_base_url(self) -> str:
        return f"http://{self.auth_service_host}:{self.auth_service_port}"

    class Config:
        env_prefix = "MARKET_"
        env_file = "../../../.env"
        case_sensitive = False
        extra = "ignore"


settings = Settings()

