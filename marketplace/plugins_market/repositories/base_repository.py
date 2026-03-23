import logging
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, ContextManager, Dict, Generic, List, Optional, Type, TypeVar

from fastapi import status
from sqlalchemy import asc, desc, or_
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Query, Session

from plugins_market.core.database import SessionLocal
from plugins_market.schemas.common import ResponseModel

T = TypeVar('T')


@dataclass
class PaginationQuery:
    page: int = 1
    page_size: int = 20
    order_by: Optional[str] = None
    order_desc: bool = True
    search: Optional[str] = None
    search_fields: Optional[List[str]] = None

logger = logging.getLogger(__name__)


@contextmanager
def generate_db():
    """Generate a database session context manager."""
    db_session = SessionLocal()
    try:
        yield db_session
    finally:
        db_session.close()


def get_db_session(db_session: Optional[Session] = None) -> ContextManager[Session]:
    """Get database session context manager.

    Reuses existing session if provided, otherwise creates a new one.

    Args:
        db_session: Optional existing database session

    Returns:
        Context manager for database session
    """
    if db_session is not None:
        return nullcontext(db_session)
    return generate_db()


class BaseRepository(Generic[T]):
    """Generic data access base class.

    Provides standard CRUD operations and query building functionality.
    Supports generics to ensure type safety.
    """

    def __init__(self, db: Session, model_class: Type[T]):
        """Initialize repository.

        Args:
            db: Database session
            model_class: Model class
        """
        self.db = db
        self.model_class = model_class

    def create(self, obj_in: Dict[str, Any] | T) -> T:
        """Create a new record.

        Args:
            obj_in: Creation data dictionary or model instance

        Returns:
            Created model instance
        """
        try:
            db_obj = obj_in if isinstance(obj_in, self.model_class) else self.model_class(**obj_in)
            self.db.add(db_obj)
            self.db.commit()
            self.db.refresh(db_obj)
            return db_obj
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Failed to create database record: {type(e).__name__}")
            raise

    def get_by_id(self, record_id: int) -> Optional[T]:
        """Get record by ID.

        Args:
            record_id: Record ID

        Returns:
            Model instance or None
        """
        try:
            return self.db.query(self.model_class).filter(
                self.model_class.id == record_id
            ).first()
        except SQLAlchemyError as e:
            logger.error(f"Failed to get record by ID: {type(e).__name__}")
            raise

    def get_multi(
        self,
        skip: int = 0,
        limit: int = 100,
        order_by: Optional[str] = None,
        order_desc: bool = False
    ) -> List[T]:
        """Get multiple records.

        Args:
            skip: Number of records to skip
            limit: Maximum number of records to return
            order_by: Field to order by
            order_desc: Whether to order in descending order

        Returns:
            List of model instances
        """
        try:
            query = self.db.query(self.model_class)

            if order_by and hasattr(self.model_class, order_by):
                col = getattr(self.model_class, order_by)
                query = query.order_by(desc(col) if order_desc else asc(col))

            return query.offset(skip).limit(limit).all()
        except SQLAlchemyError as e:
            logger.error(f"Failed to get multiple database records: {type(e).__name__}")
            raise

    def update(self, db_obj: T, obj_in: Dict[str, Any]) -> T:
        """Update a record.

        Args:
            db_obj: Database object to update
            obj_in: Update data dictionary

        Returns:
            Updated model instance
        """
        try:
            for field, value in obj_in.items():
                if hasattr(db_obj, field):
                    setattr(db_obj, field, value)

            self.db.commit()
            self.db.refresh(db_obj)
            return db_obj
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Failed to update database record: {type(e).__name__}")
            raise

    def delete(self, record_id: int) -> bool:
        """Delete a record.

        Args:
            record_id: Record ID

        Returns:
            True if deleted, False if not found
        """
        try:
            db_obj = self.get_by_id(record_id)
            if db_obj:
                self.db.delete(db_obj)
                self.db.commit()
                return True
            return False
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Failed to delete database record: {type(e).__name__}")
            raise

    def delete_by_filter(self, **filters) -> int:
        """Batch delete records by filter conditions.

        Args:
            **filters: Filter conditions

        Returns:
            Number of deleted records
        """
        try:
            query = self.db.query(self.model_class)

            for field, value in filters.items():
                if hasattr(self.model_class, field):
                    query = query.filter(getattr(self.model_class, field) == value)

            deleted_count = query.delete(synchronize_session=False)
            self.db.commit()
            return deleted_count
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Failed to delete database records by filter: {type(e).__name__}")
            raise

    def count(self, **filters) -> int:
        """Count records.

        Args:
            **filters: Filter conditions

        Returns:
            Total number of records
        """
        try:
            query = self.db.query(self.model_class)

            for field, value in filters.items():
                if hasattr(self.model_class, field):
                    query = query.filter(getattr(self.model_class, field) == value)

            return query.count()
        except SQLAlchemyError as e:
            logger.error(f"Failed to count database records: {type(e).__name__}")
            raise

    def exists(self, **filters) -> bool:
        """Check if record exists.

        Args:
            **filters: Filter conditions

        Returns:
            True if record exists, False otherwise
        """
        try:
            query = self.db.query(self.model_class)

            for field, value in filters.items():
                if hasattr(self.model_class, field):
                    query = query.filter(getattr(self.model_class, field) == value)

            return query.first() is not None
        except SQLAlchemyError as e:
            logger.error(f"Failed to check record existence: {type(e).__name__}")
            raise

    def query(self) -> Query:
        """Get query builder.

        Returns:
            SQLAlchemy query object
        """
        return self.db.query(self.model_class)

    def filter_by(self, **kwargs) -> Query:
        """Filter by conditions.

        Args:
            **kwargs: Filter conditions

        Returns:
            Filtered query object
        """
        return self.query().filter_by(**kwargs)

    def get_or_create(self, defaults: Optional[Dict[str, Any]] = None, **kwargs) -> tuple[T, bool]:
        """Get or create a record.

        Args:
            defaults: Default values for creation
            **kwargs: Query conditions

        Returns:
            Tuple of (model instance, whether newly created)
        """
        try:
            instance = self.filter_by(**kwargs).first()
            if instance:
                return instance, False
            else:
                params = dict(kwargs)
                if defaults:
                    params.update(defaults)
                instance = self.create(params)
                return instance, True
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Failed to get or create database record: {type(e).__name__}")
            raise

    def bulk_create(self, objects: List[Dict[str, Any]]) -> List[T]:
        """Bulk create records.

        Args:
            objects: List of creation data dictionaries

        Returns:
            List of created model instances
        """
        try:
            db_objects = [self.model_class(**obj) for obj in objects]
            self.db.add_all(db_objects)
            self.db.commit()

            for obj in db_objects:
                self.db.refresh(obj)

            return db_objects
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Failed to bulk create database records: {type(e).__name__}")
            raise


class MarketBaseRepository(BaseRepository[T]):
    """Extended repository with advanced query capabilities.

    Provides richer query capabilities and unified exception handling.
    """

    def __init__(self, db: Session, model_class: Type[T]):
        super().__init__(db, model_class)

    @staticmethod
    def with_exception_handling(func: Callable) -> Callable:
        """Exception handling decorator."""
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            except Exception as e:
                return ResponseModel(
                    code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    message=f"DB error: {str(e)}"
                )
        return wrapper

    def get_list_with_pagination(
        self,
        params: Optional[PaginationQuery] = None,
    ) -> ResponseModel[Dict[str, Any]]:
        """Get paginated list data with optional search.

        Args:
            params: 分页与搜索参数，默认 None 时使用 PaginationQuery() 默认值

        Returns:
            ResponseModel containing items and total count
        """
        opts = params or PaginationQuery()
        try:
            query = self.db.query(self.model_class)

            if opts.search and opts.search_fields:
                search_conditions = []
                for field in opts.search_fields:
                    if hasattr(self.model_class, field):
                        col = getattr(self.model_class, field)
                        search_conditions.append(col.ilike(f"%{opts.search}%"))
                if search_conditions:
                    query = query.filter(or_(*search_conditions))

            total = query.count()

            if opts.order_by and hasattr(self.model_class, opts.order_by):
                col = getattr(self.model_class, opts.order_by)
                query = query.order_by(desc(col) if opts.order_desc else asc(col))

            skip = (opts.page - 1) * opts.page_size
            items = query.offset(skip).limit(opts.page_size).all()

            return ResponseModel(
                code=status.HTTP_200_OK,
                message="Get list successfully",
                data={
                    "items": items,
                    "total": total,
                    "page": opts.page,
                    "page_size": opts.page_size
                }
            )
        except Exception as e:
            return ResponseModel(
                code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                message=f"Get list failed: {str(e)}"
            )

    def create_with_response(self, obj_in: Dict[str, Any]) -> ResponseModel[T]:
        """Create a record and return ResponseModel."""
        try:
            obj = self.create(obj_in)
            return ResponseModel(
                code=status.HTTP_200_OK,
                message="Create successfully",
                data=obj
            )
        except Exception as e:
            return ResponseModel(
                code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                message=f"Create failed: {str(e)}"
            )

    def get_by_id_with_response(self, record_id: int) -> ResponseModel[T]:
        """Get record by ID and return ResponseModel."""
        try:
            obj = self.get_by_id(record_id)
            if obj:
                return ResponseModel(
                    code=status.HTTP_200_OK,
                    message="Get successfully",
                    data=obj
                )
            return ResponseModel(
                code=status.HTTP_404_NOT_FOUND,
                message="Record not found"
            )
        except Exception as e:
            return ResponseModel(
                code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                message=f"Get failed: {str(e)}"
            )

    def update_with_response(self, record_id: int, obj_in: Dict[str, Any]) -> ResponseModel[T]:
        """Update a record and return ResponseModel."""
        try:
            obj = self.get_by_id(record_id)
            if not obj:
                return ResponseModel(
                    code=status.HTTP_404_NOT_FOUND,
                    message="Record not found"
                )
            updated_obj = self.update(obj, obj_in)
            return ResponseModel(
                code=status.HTTP_200_OK,
                message="Update successfully",
                data=updated_obj
            )
        except Exception as e:
            return ResponseModel(
                code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                message=f"Update failed: {str(e)}"
            )

    def delete_with_response(self, record_id: int) -> ResponseModel[None]:
        """Delete a record and return ResponseModel."""
        try:
            success = self.delete(record_id)
            if success:
                return ResponseModel(
                    code=status.HTTP_200_OK,
                    message="Delete successfully"
                )
            return ResponseModel(
                code=status.HTTP_404_NOT_FOUND,
                message="Record not found"
            )
        except Exception as e:
            return ResponseModel(
                code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                message=f"Delete failed: {str(e)}"
            )
