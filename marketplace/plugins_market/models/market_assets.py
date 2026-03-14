from sqlalchemy import (
    BigInteger,
    Column,
    Integer,
    String,
    Text,
    JSON,
    Numeric,
    Index,
    UniqueConstraint,
    ForeignKey,
)

from .base import Base


class MarketAssetDB(Base):
    __tablename__ = "market_assets"

    asset_id = Column(String(64), primary_key=True, nullable=False)
    asset_type = Column(String(32), nullable=False)
    name = Column(String(128), nullable=False)
    short_desc = Column(String(512), nullable=True)
    detail_desc = Column(Text, nullable=True)
    icon_uri = Column(String(255), nullable=True)
    publisher_id = Column(String(64), nullable=False)
    publisher_name = Column(String(128), nullable=False)
    category_id = Column(String(64), nullable=False)
    tags = Column(JSON, nullable=True)
    status = Column(String(32), nullable=True, default="PUBLISHED")
    certification = Column(String(32), nullable=True)
    latest_version = Column(String(32), nullable=True)
    view_count = Column(Integer, nullable=False, default=0)
    install_count = Column(Integer, nullable=False, default=0)
    like_count = Column(Integer, nullable=False, default=0)
    create_time = Column(BigInteger, nullable=True)
    update_time = Column(BigInteger, nullable=True)
    review_count = Column(Integer, nullable=False, default=0)
    average_rating = Column(Numeric(3, 2), nullable=False, default=8.00)

    __table_args__ = (
        Index("idx_asset_type", asset_type),
        Index("idx_name", name),
        Index("idx_publisher_id", publisher_id),
        Index("idx_category_id", category_id),
        Index("idx_status", status),
        Index("idx_certification", certification),
        Index("idx_install_count", install_count),
        Index("idx_like_count", like_count),
    )


class MarketAssetVersionDB(Base):
    __tablename__ = "market_asset_versions"

    version_id = Column(String(64), primary_key=True, nullable=False)
    asset_id = Column(
        String(64),
        ForeignKey("market_assets.asset_id"),
        nullable=False,
        index=True,
    )
    version = Column(String(32), nullable=False)
    source_publish_id = Column(String(64), nullable=False)
    changelog = Column(Text, nullable=True)
    status = Column(String(32), nullable=True, default="pending_review")
    create_time = Column(BigInteger, nullable=True)

    __table_args__ = (
        UniqueConstraint("asset_id", "version", name="uk_asset_version"),
    )

