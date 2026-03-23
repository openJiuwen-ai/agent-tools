from typing import List, Optional, Tuple
import time

from sqlalchemy import and_, asc, desc, or_
from sqlalchemy.orm import Session

from plugins_market.models.market_assets import MarketAssetDB, MarketAssetVersionDB
from plugins_market.schemas.plugin import AssetCreate, AssetVersionCreate, PluginListQuery
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

    def list_by_publisher_name_and_type(
        self,
        publisher_id: str,
        name: str,
        asset_type: str = "plugin",
    ) -> List[MarketAssetDB]:
        """All assets for same publisher + exact name + type (0/1/many for publish resolution)."""
        return self.filter_by(
            publisher_id=publisher_id,
            name=name,
            asset_type=asset_type,
        ).all()

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

    def list_plugins(
        self,
        params: PluginListQuery,
    ) -> Tuple[List[Tuple[MarketAssetDB, Optional[str]]], int]:
        """
        分页查询插件列表，默认排除 status=OFFLINE 的资源。
        支持按 asset_id、asset_type、publisher_id、publisher_name（模糊）、
        search_keyword（对 name/short_desc/detail_desc 做 OR 模糊）过滤，
        按 order_by 排序。
        """
        q_assets = self.query().filter(MarketAssetDB.status != "OFFLINE")

        if params.asset_id:
            q_assets = q_assets.filter(MarketAssetDB.asset_id == params.asset_id)
        if params.asset_type:
            q_assets = q_assets.filter(MarketAssetDB.asset_type == params.asset_type)
        if params.publisher_id:
            q_assets = q_assets.filter(MarketAssetDB.publisher_id == params.publisher_id)
        if params.publisher_name and params.publisher_name.strip():
            q_assets = q_assets.filter(
                MarketAssetDB.publisher_name.ilike(f"%{params.publisher_name.strip()}%")
            )
        if params.run_time and params.run_time.strip():
            # run_time 来自 plugin.yaml 的 runtime.type；这里做精确匹配。
            q_assets = q_assets.filter(MarketAssetDB.run_time == params.run_time.strip())
        if params.search_keyword and params.search_keyword.strip():
            kw = f"%{params.search_keyword.strip()}%"
            q_assets = q_assets.filter(
                or_(
                    MarketAssetDB.name.ilike(kw),
                    MarketAssetDB.short_desc.ilike(kw),
                    MarketAssetDB.detail_desc.ilike(kw),
                )
            )

        total = q_assets.count()

        order_col = getattr(
            MarketAssetDB,
            params.order_by if hasattr(MarketAssetDB, params.order_by) else "install_count",
        )
        q_assets = q_assets.order_by(desc(order_col) if params.desc else asc(order_col))

        page = max(1, params.page)
        page_size = max(1, min(params.page_size, 100))
        offset = (page - 1) * page_size
        q = (
            q_assets.outerjoin(
                MarketAssetVersionDB,
                and_(
                    MarketAssetVersionDB.asset_id == MarketAssetDB.asset_id,
                    MarketAssetVersionDB.version == MarketAssetDB.latest_version,
                ),
            )
            .add_columns(MarketAssetVersionDB.icon_uri)
        )
        rows: List[Tuple[MarketAssetDB, Optional[str]]] = q.offset(offset).limit(page_size).all()

        return rows, total

    def delete_asset(self, asset_id: str) -> int:
        """Delete asset by asset_id. Returns number of rows deleted (0 or 1)."""
        n = self.query().filter(MarketAssetDB.asset_id == asset_id).delete(synchronize_session=False)
        self.db.commit()
        return n


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

    def list_versions_chronological(self, asset_id: str) -> List[MarketAssetVersionDB]:
        """Oldest first — used to build cumulative changelog.log from all version rows."""
        return (
            self.filter_by(asset_id=asset_id)
            .order_by(
                MarketAssetVersionDB.create_time.asc(),
                MarketAssetVersionDB.version.asc(),
            )
            .all()
        )

    def get_version(
        self,
        asset_id: str,
        version: str,
    ) -> Optional[MarketAssetVersionDB]:
        return self.filter_by(asset_id=asset_id, version=version).first()

    def count_versions(self, asset_id: str) -> int:
        return self.filter_by(asset_id=asset_id).count()

    def delete_version(self, asset_id: str, version: str) -> int:
        """Delete one version by asset_id and version. Returns number of rows deleted (0 or 1)."""
        n = (
            self.query()
            .filter(
                MarketAssetVersionDB.asset_id == asset_id,
                MarketAssetVersionDB.version == version,
            )
            .delete(synchronize_session=False)
        )
        self.db.commit()
        return n

    def delete_all_versions(self, asset_id: str) -> int:
        """Delete all versions of an asset. Returns number of rows deleted."""
        n = self.query().filter(MarketAssetVersionDB.asset_id == asset_id).delete(synchronize_session=False)
        self.db.commit()
        return n

    def create_version(self, params: AssetVersionCreate) -> MarketAssetVersionDB:
        now_ms = int(time.time() * 1000)
        obj_in = params.model_dump()
        obj_in["create_time"] = now_ms
        return self.create(obj_in)
