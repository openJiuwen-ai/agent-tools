from typing import List, Optional
import time

from sqlalchemy.orm import Session

from plugins_market.models.market_assets import MarketAssetDB, MarketAssetVersionDB
from plugins_market.schemas.plugin import AssetCreate, AssetVersionCreate
from .base_repository import MarketBaseRepository


class MarketAssetRepository(MarketBaseRepository[MarketAssetDB]):
    """Data access for market_assets."""

    def __init__(self, db: Session):
        super().__init__(db, MarketAssetDB)

    def create_asset(self, params: AssetCreate) -> MarketAssetDB:
        now_ms = int(time.time() * 1000)
        obj_in = params.model_dump()
        obj_in.update({"create_time": now_ms, "update_time": now_ms})
        return self.create(obj_in)

    def update_latest_version(self, asset: MarketAssetDB, version: str) -> MarketAssetDB:
        now_ms = int(time.time() * 1000)
        return self.update(
            asset,
            {
                "latest_version": version,
                "update_time": now_ms,
            },
        )

    def get_by_asset_id(self, asset_id: str) -> Optional[MarketAssetDB]:
        return self.filter_by(asset_id=asset_id).first()

    def list_by_category(
        self,
        category_id: str,
        status: Optional[str] = None,
        limit: int = 50,
    ) -> List[MarketAssetDB]:
        query = self.filter_by(category_id=category_id)
        if status:
            query = query.filter(MarketAssetDB.status == status)
        return (
            query.order_by(MarketAssetDB.install_count.desc())
            .limit(limit)
            .all()
        )

    def list_by_publisher(
        self,
        publisher_id: str,
        limit: int = 50,
    ) -> List[MarketAssetDB]:
        return (
            self.filter_by(publisher_id=publisher_id)
            .order_by(MarketAssetDB.create_time.desc())
            .limit(limit)
            .all()
        )

    def search_by_name(
        self,
        keyword: str,
        limit: int = 50,
    ) -> List[MarketAssetDB]:
        return (
            self.query()
            .filter(MarketAssetDB.name.ilike(f"%{keyword}%"))
            .order_by(MarketAssetDB.view_count.desc())
            .limit(limit)
            .all()
        )


class MarketAssetVersionRepository(MarketBaseRepository[MarketAssetVersionDB]):
    """Data access for market_asset_versions."""

    def __init__(self, db: Session):
        super().__init__(db, MarketAssetVersionDB)

    def list_versions(self, asset_id: str) -> List[MarketAssetVersionDB]:
        return (
            self.filter_by(asset_id=asset_id)
            .order_by(MarketAssetVersionDB.create_time.desc())
            .all()
        )

    def get_version(
        self,
        asset_id: str,
        version: str,
    ) -> Optional[MarketAssetVersionDB]:
        return self.filter_by(asset_id=asset_id, version=version).first()

    def create_version(self, params: AssetVersionCreate) -> MarketAssetVersionDB:
        now_ms = int(time.time() * 1000)
        obj_in = params.model_dump()
        obj_in["create_time"] = now_ms
        return self.create(obj_in)
