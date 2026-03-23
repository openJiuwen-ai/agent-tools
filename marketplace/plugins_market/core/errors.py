from typing import Any, Optional, TypedDict


class ErrorPayload(TypedDict, total=False):
    code: int
    data: Any
    error: str
    message: str


def _payload(
    *,
    code: int,
    error: str,
    message: str,
    data: Optional[Any] = None,
) -> ErrorPayload:
    return {"code": code, "data": data, "error": error, "message": message}


class PublishError(Exception):
    """Business error during publish; router maps to HTTPException."""

    def __init__(
        self,
        *,
        code: int,
        error: str,
        message: str,
        data: Optional[Any] = None,
    ):
        self.detail = _payload(code=code, error=error, message=message, data=data)
        self.status_code = code
        super().__init__(f"PublishError {code}: {error}")

