from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Store 服务配置。"""

    app_name: str = "Store Service"
    app_version: str = "0.1.0"
    debug: bool = True

    host: str = "0.0.0.0"
    port: int = 8100

    db_url: str = "sqlite:///./data/store.db"

    class Config:
        env_prefix = "STORE_"
        env_file = "../../../.env"
        case_sensitive = False
        extra = "ignore"


settings = Settings()

