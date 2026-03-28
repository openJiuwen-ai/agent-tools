import base64
import logging
import os
from typing import Optional

import httpx

from common.huaweicloud.huaweicloud_iam import HuaweiCloudIAM

logger = logging.getLogger(__name__)


class HuaweiCloudKMS:
    """Huawei Cloud DEW KMS client via HTTP API."""

    def __init__(
        self,
        iam_client: HuaweiCloudIAM,
        project_id: str,
        region: str = "ap-southeast-1",
        kms_endpoint: Optional[str] = None,
        encryption_algorithm: Optional[str] = None,
    ):
        self.iam_client = iam_client
        self.project_id = project_id
        self.region = region
        self.kms_endpoint = (
            kms_endpoint
            or os.getenv("HUAWEICLOUD_KMS_ENDPOINT")
            or f"https://kms.{region}.myhuaweicloud.com"
        )
        self.encryption_algorithm = (
            encryption_algorithm
            or os.getenv("HUAWEICLOUD_KMS_ENCRYPTION_ALGORITHM")
            or "RSAES_OAEP_SHA_256"
        )
        self._client = httpx.Client(timeout=30.0)

    def _get_headers(self) -> dict[str, str]:
        token = self.iam_client.get_token(project_id=self.project_id)
        return {"Content-Type": "application/json", "X-Auth-Token": token}

    def _request_kms(self, endpoint: str, body: dict, action: str, key_id: str) -> dict:
        url = f"{self.kms_endpoint}/v1.0/{self.project_id}/kms/{endpoint}"
        try:
            response = self._client.post(url, json=body, headers=self._get_headers())
            response.raise_for_status()
            logger.info("Successfully %s data using KMS key: %s", action, key_id)
            return response.json()
        except httpx.HTTPError as e:
            logger.error("KMS %s failed: %s", action, e)
            if hasattr(e, "response") and e.response is not None:
                try:
                    logger.error("KMS error detail: %s", e.response.json())
                except Exception:
                    logger.error("KMS error response: %s", e.response.text)
            raise RuntimeError(f"KMS {action} failed: {e}") from e

    def encrypt(self, key_id: str, plaintext: bytes) -> str:
        """Encrypt plaintext bytes via KMS CMK."""
        plaintext_b64 = base64.b64encode(plaintext).decode("utf-8")
        body = {
            "key_id": key_id,
            "plain_text": plaintext_b64,
            "encryption_algorithm": self.encryption_algorithm,
        }
        result = self._request_kms("encrypt-data", body, "encrypting", key_id)
        ciphertext = result.get("cipher_text")
        if not ciphertext:
            raise ValueError("Failed to get cipher_text from KMS response")
        return str(ciphertext)

    def decrypt(self, key_id: str, ciphertext: str) -> str:
        """Decrypt base64 ciphertext via KMS CMK and return plain_text(base64)."""
        body = {
            "key_id": key_id,
            "cipher_text": ciphertext,
            "encryption_algorithm": self.encryption_algorithm,
        }
        result = self._request_kms("decrypt-data", body, "decrypting", key_id)
        plaintext = result.get("plain_text")
        if not plaintext:
            raise ValueError("Failed to get plain_text from KMS response")
        return str(plaintext)
