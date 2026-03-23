from typing import Generic, List, Optional, TypeVar

from pydantic import BaseModel, ConfigDict

T = TypeVar("T")


class ResponseModel(BaseModel, Generic[T]):
    """通用响应模型。"""

    model_config = ConfigDict(arbitrary_types_allowed=True, exclude_none=True)

    code: int = 200
    message: str = "Success"
    data: Optional[T] = None


class PageMeta(BaseModel):
    """分页元数据。"""

    page: int = 1
    page_size: int = 10
    total: int = 0


class PaginationParams(BaseModel):
    page: int = 1
    size: int = 20
    total: Optional[int] = None


class PaginatedResponse(BaseModel, Generic[T]):
    items: List[T]
    pagination: PaginationParams
