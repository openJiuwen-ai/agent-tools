import os
from typing import Dict, Any, Optional, List

import boto3
from botocore.config import Config

STORAGE_TYPES = ("MinIO", "OBS")


class S3StorageConfig:
    """S3-compatible storage config. STORAGE_TYPE must be MinIO or OBS."""

    def __init__(self, storage_type: str = "MinIO"):
        """
        :param storage_type: Must be "MinIO" or "OBS".
        """
        raw = (storage_type or "MinIO").strip()
        if raw.upper() == "MINIO":
            self.storage_type = "MinIO"
        elif raw.upper() == "OBS":
            self.storage_type = "OBS"
        else:
            raise ValueError(
                f"STORAGE_TYPE must be one of {STORAGE_TYPES}, got: {storage_type!r}"
            )

        # Public URL：优先 MARKET_STORAGE_PUBLIC_URL；否则 endpoint+bucket
        self.public_base_url = (os.getenv("MARKET_STORAGE_PUBLIC_URL") or "").rstrip("/")
        self.endpoint_url = (os.getenv("MARKET_S3_ENDPOINT") or "").rstrip("/")
        self.access_key = os.getenv("MARKET_S3_ACCESS_KEY") or ""
        self.secret_key = os.getenv("MARKET_S3_SECRET_KEY") or ""
        self.bucket_name = os.getenv("MARKET_BUCKET_NAME", "openjiuwen-market")
        self.region_name = os.getenv("MARKET_S3_REGION", "us-east-1")

        # use_ssl:
        # - MinIO 默认 http
        # - OBS 默认 https
        # - 同时允许显式覆盖：MARKET_S3_USE_SSL=true|false
        use_ssl_env = os.getenv("MARKET_S3_USE_SSL", "").strip().lower()
        if use_ssl_env in ("true", "1", "on"):
            self.use_ssl = True
        elif use_ssl_env in ("false", "0", "off"):
            self.use_ssl = False
        else:
            if self.storage_type == "MinIO":
                self.use_ssl = bool(self.endpoint_url.startswith("https://"))
            else:
                self.use_ssl = True

        if not self.public_base_url and self.endpoint_url:
            self.public_base_url = f"{self.endpoint_url}/{self.bucket_name}"


class S3StorageClient:
    """S3存储客户端，支持本地MinIO和云端S3"""
    
    def __init__(self, config: S3StorageConfig):
        """
        初始化S3客户端
        :param config: S3存储配置
        """
        self.config = config

        if config.storage_type == "OBS":
            os.environ["AWS_REQUEST_CHECKSUM_CALCULATION"] = "WHEN_REQUIRED"
            os.environ["AWS_RESPONSE_CHECKSUM_VALIDATION"] = "WHEN_REQUIRED"

        addressing_style = "virtual" if config.storage_type == "OBS" else "path"
        s3_config: Dict[str, Any] = {"addressing_style": addressing_style}
        if config.storage_type == "OBS":
            s3_config["payload_signing_enabled"] = False

        self.s3_client = boto3.client(
            "s3",
            endpoint_url=config.endpoint_url,
            aws_access_key_id=config.access_key,
            aws_secret_access_key=config.secret_key,
            region_name=config.region_name,
            use_ssl=config.use_ssl,
            config=Config(
                signature_version="s3v4",
                s3=s3_config,
            ),
        )

        # 确保存储桶存在
        self._ensure_bucket_exists()

    def _ensure_bucket_exists(self):
        """确保存储桶存在，如果不存在则抛出异常"""
        try:
            self.s3_client.head_bucket(Bucket=self.config.bucket_name)
        except self.s3_client.exceptions.ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            if error_code == '404':
                raise Exception(f"存储桶 '{self.config.bucket_name}' 不存在，请联系管理员创建存储桶") from e
            if self.config.storage_type == "OBS" and str(error_code) in {"403", "AccessDenied", "Forbidden"}:
                return
            raise Exception(f"无法访问存储桶 '{self.config.bucket_name}': {str(e)}") from e
    
    def public_url_for_key(self, key: str) -> str:
        """Build public URL for an object key (for icon_uri etc.)."""
        if not self.config.public_base_url:
            return ""
        return f"{self.config.public_base_url.rstrip('/')}/{key}"

    def delete_object(self, key: str) -> Dict[str, Any]:
        """Delete one object by key. Key is path within bucket."""
        try:
            self.s3_client.delete_object(Bucket=self.config.bucket_name, Key=key)
            return {"success": True, "key": key, "storage_type": self.config.storage_type}
        except Exception as e:
            return {"success": False, "error": str(e), "key": key, "storage_type": self.config.storage_type}

    def delete_objects(self, keys: list) -> Dict[str, Any]:
        """删除多个对象：统一逐条 delete_object（MinIO / OBS 同一套逻辑）。"""
        errors: List[Dict[str, Any]] = []
        for k in keys:
            if not k:
                continue
            r = self.delete_object(k)
            if not r.get("success"):
                errors.append({"key": k, "error": r.get("error", "unknown")})
        return {"success": len(errors) == 0, "errors": errors}

    def list_keys(self, prefix: str) -> List[str]:
        """List all object keys under prefix."""
        keys: List[str] = []
        token: Optional[str] = None
        while True:
            kwargs = {"Bucket": self.config.bucket_name, "Prefix": prefix}
            if token:
                kwargs["ContinuationToken"] = token
            resp = self.s3_client.list_objects_v2(**kwargs)
            for obj in resp.get("Contents", []) or []:
                k = obj.get("Key")
                if k:
                    keys.append(k)
            if resp.get("IsTruncated"):
                token = resp.get("NextContinuationToken")
                if not token:
                    break
            else:
                break
        return keys

    def delete_prefix(self, prefix: str) -> Dict[str, Any]:
        """Delete all objects under prefix."""
        keys = self.list_keys(prefix)
        if not keys:
            return {"success": True, "deleted": 0, "errors": []}
        r = self.delete_objects(keys)
        return {"success": r.get("success", False), "deleted": len(keys), "errors": r.get("errors", [])}

    def upload_bytes(self, body: bytes, s3_key: str) -> Dict[str, Any]:
        """Upload bytes to S3/MinIO. Key is path within bucket (e.g. plugins/...)."""
        try:
            self.s3_client.put_object(
                Bucket=self.config.bucket_name,
                Key=s3_key,
                Body=body,
            )
            return {
                "success": True,
                "bucket": self.config.bucket_name,
                "key": s3_key,
                "storage_type": self.config.storage_type,
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "storage_type": self.config.storage_type,
            }


_storage_client: Optional["S3StorageClient"] = None


def get_storage_client() -> "S3StorageClient":
    """Build or return cached S3StorageClient from env (STORAGE_TYPE=MinIO|OBS). Used as FastAPI Depends()."""
    global _storage_client
    if _storage_client is None:
        storage_type = os.getenv("STORAGE_TYPE", "MinIO").strip()
        config = S3StorageConfig(storage_type=storage_type or "MinIO")
        _storage_client = S3StorageClient(config)
    return _storage_client
