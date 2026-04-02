import logging
import os
import threading
import time
from datetime import datetime
from typing import Optional, TypedDict

import httpx

logger = logging.getLogger(__name__)


class TemporaryCredentials(TypedDict):
    access: str
    secret: str
    securitytoken: str
    expires_at: str


class HuaweiCloudIAM:
    """Huawei Cloud IAM auth client via HTTP API."""

    @classmethod
    def from_env(cls, *, require_project_scope: bool = True) -> "HuaweiCloudIAM":
        """
        使用 HUAWEICLOUD_* 环境变量构造客户端。
        """
        username = (os.getenv("HUAWEICLOUD_USERNAME") or "").strip()
        password = (os.getenv("HUAWEICLOUD_PASSWORD") or "").strip()
        if not username or not password:
            raise RuntimeError("缺少 HUAWEICLOUD_USERNAME 或 HUAWEICLOUD_PASSWORD")

        domain_raw = os.getenv("HUAWEICLOUD_DOMAIN_NAME")
        if domain_raw is None:
            domain_name = None
        else:
            domain_name = domain_raw.strip() or None

        iam_ep_raw = os.getenv("HUAWEICLOUD_IAM_ENDPOINT")
        iam_endpoint = (iam_ep_raw or "").strip() or None

        if require_project_scope:
            pid = (os.getenv("HUAWEICLOUD_PROJECT_ID") or "").strip() or None
            pname = (os.getenv("HUAWEICLOUD_PROJECT_NAME") or "").strip() or None
            if not pid and not pname:
                raise RuntimeError("缺少 HUAWEICLOUD_PROJECT_ID 或 HUAWEICLOUD_PROJECT_NAME")

        return cls(
            username=username,
            password=password,
            domain_name=domain_name,
            iam_endpoint=iam_endpoint,
        )

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
        self._cached_sts: Optional[TemporaryCredentials] = None
        self._sts_expire_at_ts: float = 0.0
        self._sts_scope: Optional[tuple[str, str]] = None
        self._closed: bool = False

    def close(self) -> None:
        """释放底层 HTTP 连接池资源。"""
        if self._closed:
            return
        self._closed = True
        try:
            self._client.close()
        except Exception:
            # best-effort
            pass

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

        auth_scope: dict = {}
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

            payload = response.json()
            token_data = payload.get("token", {}) or {}
            if not self.project_id:
                if project_id:
                    self.project_id = project_id
                else:
                    project_info = token_data.get("project", {}) or {}
                    if project_info:
                        self.project_id = project_info.get("id")

            logger.info("Successfully obtained IAM token")
            with self._token_lock:
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

    def get_temporary_credentials(
        self,
        *,
        project_id: Optional[str] = None,
        project_name: Optional[str] = None,
        duration_seconds: int = 3600,
    ) -> TemporaryCredentials:
        """
        Exchange IAM token for temporary AK/SK/SecurityToken.
        API: POST /v3.0/OS-CREDENTIAL/securitytokens
        """
        if project_name and not project_id:
            self.get_token(project_name=project_name)
            project_id = self.project_id
        elif not project_id:
            project_id = self.project_id

        if not project_id:
            raise ValueError("project_id is required to get temporary credentials")

        scope = ("project_id", project_id)
        with self._token_lock:
            now = time.time()
            if (
                self._cached_sts
                and self._sts_scope == scope
                and now < (self._sts_expire_at_ts - 120)
            ):
                return self._cached_sts

        token = self.get_token(project_id=project_id)
        url = f"{self.iam_endpoint}/v3.0/OS-CREDENTIAL/securitytokens"
        body = {
            "auth": {
                "identity": {
                    "methods": ["token"],
                    "token": {"id": token},
                },
                "scope": {"project": {"id": project_id}},
            }
        }
        _ = duration_seconds

        try:
            response = self._client.post(
                url,
                json=body,
                headers={"Content-Type": "application/json", "X-Auth-Token": token},
            )
            response.raise_for_status()
            payload = response.json()
            cred = payload.get("credential") or {}
            access = str(cred.get("access") or "").strip()
            secret = str(cred.get("secret") or "").strip()
            securitytoken = str(cred.get("securitytoken") or "").strip()
            expires_at = str(cred.get("expires_at") or "").strip()
            if not (access and secret and securitytoken):
                raise ValueError("Invalid temporary credentials response from IAM")
            out: TemporaryCredentials = {
                "access": access,
                "secret": secret,
                "securitytoken": securitytoken,
                "expires_at": expires_at,
            }
            with self._token_lock:
                self._cached_sts = out
                self._sts_expire_at_ts = self._parse_expire_ts(expires_at) or (time.time() + 300)
                self._sts_scope = scope
            return out
        except httpx.HTTPError as e:
            logger.error("Failed to get temporary credentials: %s", e)
            if hasattr(e, "response") and e.response is not None:
                try:
                    logger.error("IAM STS error detail: %s", e.response.json())
                except Exception:
                    logger.error("IAM STS error response: %s", e.response.text)
            raise RuntimeError(f"IAM temporary credentials failed: {e}") from e

    def invalidate_sts_cache(self) -> None:
        """丢弃已缓存的 securitytokens，下次 get_temporary_credentials 将重新请求 IAM。"""
        with self._token_lock:
            self._cached_sts = None
            self._sts_expire_at_ts = 0.0
            self._sts_scope = None

    def get_temporary_credentials_from_env(
        self,
        *,
        duration_seconds: int = 3600,
    ) -> TemporaryCredentials:
        """读取 HUAWEICLOUD_PROJECT_ID / NAME，换取 OBS 用临时 AK/SK/Token。"""
        project_id = (os.getenv("HUAWEICLOUD_PROJECT_ID") or "").strip() or None
        project_name = (os.getenv("HUAWEICLOUD_PROJECT_NAME") or "").strip() or None
        if not project_id and not project_name:
            raise RuntimeError("缺少 HUAWEICLOUD_PROJECT_ID 或 HUAWEICLOUD_PROJECT_NAME")
        return self.get_temporary_credentials(
            project_id=project_id,
            project_name=project_name,
            duration_seconds=duration_seconds,
        )
