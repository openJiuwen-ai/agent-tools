"""Security helpers: AES-GCM encryption/decryption with optional Huawei Cloud KMS."""

import base64
import os
import threading
import time
from typing import Any, Optional

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from common.huaweicloud.huaweicloud_iam import HuaweiCloudIAM
from common.huaweicloud.huaweicloud_kms import HuaweiCloudKMS

_SECURITY_UTILS_INIT_GUARD = threading.local()
_SECRET_CACHE: dict[str, tuple[str, float]] = {}
_SECRET_CACHE_LOCK = threading.Lock()
_SECRET_KEY_LOCKS: dict[str, threading.Lock] = {}
_SECRET_KEY_LOCKS_LOCK = threading.Lock()
_NO_CACHE_SECRET_KEYS = {"SYSTEM_ADMIN_TOKEN"}
_SALT_LEN = 16
_NONCE_LEN = 12
_TAG_LEN = 16
_MASTER_KEY_BYTES = 32


def _env_bool(name: str, default: bool = False) -> bool:
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _get_secret_key_lock(env_key: str) -> threading.Lock:
    with _SECRET_KEY_LOCKS_LOCK:
        key_lock = _SECRET_KEY_LOCKS.get(env_key)
        if key_lock is None:
            key_lock = threading.Lock()
            _SECRET_KEY_LOCKS[env_key] = key_lock
        return key_lock


class SecurityUtils:
    """Sensitive data security helper with optional KMS-backed root key."""

    def __init__(self, use_kms: Optional[bool] = None) -> None:
        self.master_key: Optional[bytes] = None
        self.kms_client: Optional[HuaweiCloudKMS] = None
        self.kms_key_id: Optional[str] = None
        self.encrypted_root_key: Optional[str] = None

        if use_kms is None:
            use_kms = _env_bool("HUAWEICLOUD_KMS_ENABLED", default=False)
        self.use_kms = use_kms

        if self.use_kms:
            # KMS mode: SERVER_AES_MASTER_KEY stores encrypted root key ciphertext.
            self.encrypted_root_key = (os.getenv("SERVER_AES_MASTER_KEY") or "").strip() or None
            self._init_kms()
            self.master_key = self._get_master_key()
        else:
            # Local AES mode: SERVER_AES_MASTER_KEY stores base64-encoded 32-byte key.
            key_base64 = os.getenv("SERVER_AES_MASTER_KEY")
            if key_base64:
                try:
                    self.master_key = base64.b64decode(key_base64)
                except Exception as e:
                    raise ValueError("SERVER_AES_MASTER_KEY must be valid base64") from e

        if self.master_key and len(self.master_key) != _MASTER_KEY_BYTES:
            raise ValueError("master_key length must be 32 bytes")

    def get_initialized_master_key(self) -> Optional[bytes]:
        return self.master_key

    def clear_sensitive_state(self) -> None:
        """
        Best-effort clear of sensitive in-memory state.
        Note: Python cannot guarantee zeroization of immutable bytes in all cases.
        """
        self.master_key = None
        self.encrypted_root_key = None
        self.kms_client = None
        self.kms_key_id = None

    def _init_kms(self) -> None:
        username = os.getenv("HUAWEICLOUD_USERNAME")
        password = os.getenv("HUAWEICLOUD_PASSWORD")
        domain_name = os.getenv("HUAWEICLOUD_DOMAIN_NAME")
        project_name = os.getenv("HUAWEICLOUD_PROJECT_NAME")
        project_id = os.getenv("HUAWEICLOUD_PROJECT_ID")
        region = os.getenv("HUAWEICLOUD_REGION", "cn-north-4")
        iam_endpoint = os.getenv("HUAWEICLOUD_IAM_ENDPOINT")
        kms_endpoint = os.getenv("HUAWEICLOUD_KMS_ENDPOINT")

        if not all([username, password]):
            raise ValueError(
                "Missing KMS configuration. Required environment variables: "
                "HUAWEICLOUD_USERNAME, HUAWEICLOUD_PASSWORD"
            )
        if not project_name and not project_id:
            raise ValueError(
                "Missing project configuration. Required one of: "
                "HUAWEICLOUD_PROJECT_NAME or HUAWEICLOUD_PROJECT_ID"
            )

        iam_client = HuaweiCloudIAM(
            username=username,
            password=password,
            domain_name=domain_name,
            iam_endpoint=iam_endpoint,
        )
        if project_name and not project_id:
            iam_client.get_token(project_name=project_name)
            project_id = iam_client.project_id
            if not project_id:
                raise ValueError(f"Failed to get project_id for project_name: {project_name}")

        self.kms_client = HuaweiCloudKMS(
            iam_client=iam_client,
            project_id=project_id or "",
            region=region,
            kms_endpoint=kms_endpoint,
        )
        self.kms_key_id = os.getenv("HUAWEICLOUD_KMS_KEY_ID")
        if not self.kms_key_id:
            raise ValueError("Missing HUAWEICLOUD_KMS_KEY_ID environment variable")

    def _get_master_key(self) -> bytes:
        if not self.kms_client:
            raise ValueError("KMS client not initialized")
        if not self.encrypted_root_key:
            raise ValueError("SERVER_AES_MASTER_KEY not found in environment (KMS mode requires encrypted root key)")

        root_key_b64 = self.kms_client.decrypt(self.kms_key_id or "", self.encrypted_root_key)
        return base64.b64decode(root_key_b64)

    @classmethod
    def generate_random_key(cls, length: int = 16) -> bytes:
        return os.urandom(length)

    @classmethod
    def hkdf_drive(cls, master_key: bytes, salt: bytes) -> bytes:
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            info=b"sensitive-data-salt",
        )
        return hkdf.derive(master_key)

    def encrypt_secret(self, plaintext: str) -> Optional[str]:
        """Encrypt plaintext string and return ciphertext string (base64)."""
        if not plaintext:
            return None
        if not isinstance(plaintext, str):
            raise ValueError("plaintext must be a string")
        if not self.master_key:
            return plaintext

        salt = self.generate_random_key(_SALT_LEN)
        encryption_key = self.hkdf_drive(self.master_key, salt)
        nonce = self.generate_random_key(_NONCE_LEN)
        aesgcm = AESGCM(encryption_key)
        encrypted = aesgcm.encrypt(nonce, plaintext.encode("utf-8"), None)
        ciphertext = encrypted[:-_TAG_LEN]
        auth_tag = encrypted[-_TAG_LEN:]
        combined = salt + nonce + ciphertext + auth_tag
        return base64.b64encode(combined).decode("utf-8")

    def decrypt_secret(self, ciphertext: str) -> Optional[str]:
        """Decrypt ciphertext string (base64) and return plaintext string."""
        if not ciphertext:
            return None
        if not isinstance(ciphertext, str):
            raise ValueError("ciphertext must be a string")
        if not self.master_key:
            return ciphertext

        try:
            data = base64.b64decode(ciphertext)
        except Exception:
            return ciphertext

        min_len = _SALT_LEN + _NONCE_LEN + _TAG_LEN
        if len(data) < min_len:
            return ciphertext

        salt = data[:_SALT_LEN]
        nonce = data[_SALT_LEN:_SALT_LEN + _NONCE_LEN]
        ciphertext_bytes = data[_SALT_LEN + _NONCE_LEN:-_TAG_LEN]
        auth_tag = data[-_TAG_LEN:]

        encryption_key = self.hkdf_drive(self.master_key, salt)
        aesgcm = AESGCM(encryption_key)
        plaintext = aesgcm.decrypt(nonce, ciphertext_bytes + auth_tag, None)
        return plaintext.decode("utf-8")

    @staticmethod
    def mask_secret(secret: str, visible_chars: int = 4) -> Optional[str]:
        """Mask a secret for display (keep last N chars)."""
        if not secret:
            return None
        if len(secret) <= visible_chars:
            return "*" * len(secret)
        return "*" * (len(secret) - visible_chars) + secret[-visible_chars:]

    @staticmethod
    def get_decrypted_secret(env_key: str, default: Optional[str] = None) -> Optional[str]:
        encrypted_value = os.getenv(env_key, default)
        if not encrypted_value:
            return default

        use_kms = _env_bool("HUAWEICLOUD_KMS_ENABLED", default=False)
        key_configured = bool((os.getenv("SERVER_AES_MASTER_KEY") or "").strip())
        if not key_configured:
            return encrypted_value

        if getattr(_SECURITY_UTILS_INIT_GUARD, "initializing", False):
            raise RuntimeError(
                "Recursive secret decryption detected during SecurityUtils initialization"
            )

        cache_enabled = _env_bool("SECURITY_APPLICATION_SECRET_CACHE_ENABLED", default=True)
        if env_key in _NO_CACHE_SECRET_KEYS:
            cache_enabled = False
        now = time.time()
        if cache_enabled:
            with _SECRET_CACHE_LOCK:
                hit = _SECRET_CACHE.get(env_key)
                if hit and now < hit[1]:
                    return hit[0]

        def _decrypt_with_fresh_security_utils() -> Optional[str]:
            # Do not cache root key: instantiate per miss and clear sensitive state immediately.
            _SECURITY_UTILS_INIT_GUARD.initializing = True
            security_utils: Optional["SecurityUtils"] = None
            try:
                security_utils = SecurityUtils(use_kms=use_kms)
                return security_utils.decrypt_secret(encrypted_value)
            finally:
                _SECURITY_UTILS_INIT_GUARD.initializing = False
                if security_utils is not None:
                    try:
                        security_utils.clear_sensitive_state()
                    except Exception:
                        # Never mask the original init/decrypt exception because of best-effort cleanup.
                        pass
                    finally:
                        security_utils = None

        if cache_enabled:
            ttl_seconds = max(1, _env_int("SECURITY_APPLICATION_SECRET_CACHE_TTL_SECONDS", default=60))
            key_lock = _get_secret_key_lock(env_key)
            with key_lock:
                # Double-check cache under per-key lock to prevent thundering-herd decryptions.
                now = time.time()
                with _SECRET_CACHE_LOCK:
                    hit = _SECRET_CACHE.get(env_key)
                    if hit and now < hit[1]:
                        return hit[0]
                decrypted = _decrypt_with_fresh_security_utils()
                if decrypted == encrypted_value:
                    raise ValueError(
                        f"Secret '{env_key}' appears to be plaintext, but encryption is enabled. "
                        "Please encrypt the value before setting it in environment variables."
                    )
                if decrypted is not None:
                    with _SECRET_CACHE_LOCK:
                        _SECRET_CACHE[env_key] = (decrypted, time.time() + ttl_seconds)
                return decrypted

        decrypted = _decrypt_with_fresh_security_utils()

        if decrypted == encrypted_value:
            raise ValueError(
                f"Secret '{env_key}' appears to be plaintext, but encryption is enabled. "
                "Please encrypt the value before setting it in environment variables."
            )
        return decrypted

    @staticmethod
    def get_decrypt_secret(env_key: str, default: Optional[str] = None) -> Optional[str]:
        """Unified business entrypoint (alias)."""
        return SecurityUtils.get_decrypted_secret(env_key, default=default)
