import logging
import os
import threading
import time
from datetime import datetime, timezone
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


class HuaweiCloudIAM:
    """Huawei Cloud IAM auth client via HTTP API."""

    def __init__(
        self,
        username: str,
        password: str,
        domain_name: Optional[str] = None,
        iam_endpoint: Optional[str] = None,
    ):
        self.username = username
        self.password = password
        self.domain_name = domain_name or "Default"
        self.iam_endpoint = iam_endpoint or os.getenv(
            "HUAWEICLOUD_IAM_ENDPOINT", "https://iam.myhuaweicloud.com"
        )
        self.project_id: Optional[str] = None
        self._client = httpx.Client(timeout=10.0)
        self._token_lock = threading.Lock()
        self._cached_token: Optional[str] = None
        self._token_expire_at_ts: float = 0.0
        self._token_scope: Optional[tuple[str, str]] = None

    @staticmethod
    def _parse_expire_ts(expires_at: Optional[str]) -> float:
        if not expires_at:
            return 0.0
        try:
            return datetime.fromisoformat(expires_at.replace("Z", "+00:00")).timestamp()
        except Exception:
            return 0.0

    def get_token(
        self,
        project_name: Optional[str] = None,
        project_id: Optional[str] = None,
        domain_name: Optional[str] = None,
    ) -> str:
        """Get IAM token for KMS calls."""
        url = f"{self.iam_endpoint}/v3/auth/tokens?nocatalog=true"
        effective_domain = domain_name or self.domain_name
        scope_tuple = ("project_name", project_name) if project_name else (
            ("project_id", project_id) if project_id else ("domain", effective_domain)
        )

        with self._token_lock:
            now = time.time()
            if (
                self._cached_token
                and self._token_scope == scope_tuple
                and now < (self._token_expire_at_ts - 60)
            ):
                return self._cached_token

        auth_identity = {
            "methods": ["password"],
            "password": {
                "user": {
                    "name": self.username,
                    "password": self.password,
                    "domain": {"name": effective_domain},
                }
            },
        }

        auth_scope = {}
        if project_name:
            auth_scope["project"] = {"name": project_name}
        elif project_id:
            auth_scope["project"] = {"id": project_id}
        else:
            auth_scope["domain"] = {"name": effective_domain}

        body = {"auth": {"identity": auth_identity, "scope": auth_scope}}

        try:
            response = self._client.post(url, json=body, headers={"Content-Type": "application/json"})
            response.raise_for_status()

            token = response.headers.get("X-Subject-Token")
            if not token:
                raise ValueError("Failed to get X-Subject-Token from IAM response")

            if not self.project_id:
                if project_id:
                    self.project_id = project_id
                else:
                    try:
                        token_data = response.json().get("token", {})
                        project_info = token_data.get("project", {})
                        if project_info:
                            self.project_id = project_info.get("id")
                    except Exception:
                        logger.debug("Failed to parse IAM token response for project_id", exc_info=True)

            logger.info("Successfully obtained IAM token")
            with self._token_lock:
                token_data = response.json().get("token", {})
                expire_at = token_data.get("expires_at")
                expire_ts = self._parse_expire_ts(expire_at)
                self._cached_token = token
                self._token_expire_at_ts = expire_ts or (time.time() + 300)
                self._token_scope = scope_tuple
            return token
        except httpx.HTTPError as e:
            logger.error("Failed to get IAM token: %s", e)
            if hasattr(e, "response") and e.response is not None:
                try:
                    logger.error("IAM error detail: %s", e.response.json())
                except Exception:
                    logger.error("IAM error response: %s", e.response.text)
            raise RuntimeError(f"IAM authentication failed: {e}") from e
