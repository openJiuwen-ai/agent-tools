"""An invalid API key must not be written to logs verbatim."""

import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from loguru import logger

from openjiuwentools.infer_router.api.auth import verify_api_key
from openjiuwentools.infer_router.config.config import settings


@pytest.mark.asyncio
async def test_invalid_key_is_not_logged_verbatim(monkeypatch) -> None:
    monkeypatch.setattr(settings, "enable_auth", True)
    monkeypatch.setattr(settings, "api_key", "valid-key")
    messages: list[str] = []
    sink_id = logger.add(lambda m: messages.append(str(m)), format="{message}")

    try:
        with pytest.raises(HTTPException):
            await verify_api_key(
                HTTPAuthorizationCredentials(scheme="Bearer", credentials="bad-secret-key")
            )
    finally:
        logger.remove(sink_id)

    assert not any("bad-secret-key" in message for message in messages)
