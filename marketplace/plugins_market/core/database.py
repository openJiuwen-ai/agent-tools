import os
from pathlib import Path
from typing import Generator

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from common.security.security_utils import SecurityUtils
from .config import settings


PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env")


def _normalize_db_url(raw_url: str) -> str:
    if raw_url.startswith("sqlite:///"):
        path_str = raw_url.replace("sqlite:///", "", 1)
        db_path = Path(path_str)
        if db_path.parent:
            db_path.parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{db_path.resolve()}"
    return raw_url


def _build_database_url_from_shared_env() -> str | None:
    if os.getenv("STORE_DB_URL"):
        return None

    db_type = os.getenv("DB_TYPE")
    if not db_type:
        return None

    db_type = db_type.lower()

    if db_type == "mysql":
        host = os.getenv("DB_HOST", "localhost")
        port = os.getenv("DB_PORT", "3306")
        user = os.getenv("DB_USER", "root")
        password = SecurityUtils.get_decrypt_secret("DB_PASSWORD", default="") or ""
        
        store_db_name = os.getenv("STORE_DB_NAME", "openjiuwen_market")

        return (
            f"mysql+pymysql://{user}:{password}@"
            f"{host}:{port}/{store_db_name}?charset=utf8mb4"
        )

    return None


def _get_effective_database_url() -> str:
    """返回实际使用的数据库 URL。"""
    # 优先尝试根据共享 DB_* 配置 + STORE_DB_NAME 生成
    shared_url = _build_database_url_from_shared_env()
    if shared_url:
        return _normalize_db_url(shared_url)

    # 否则退回到 settings.db_url（支持 STORE_DB_URL 或默认 SQLite）
    return _normalize_db_url(settings.db_url)

DATABASE_URL = _get_effective_database_url()

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite:///") else {},
    pool_pre_ping=True,
    pool_recycle=3600
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Generator:
    """用于 FastAPI 依赖注入的 DB Session。"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

