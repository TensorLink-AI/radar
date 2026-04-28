"""S3-compatible storage for experiment artifacts.

Primary backend is **Hippius** (Substrate-based decentralized object storage,
S3-compatible at ``https://s3.hippius.com``). Cloudflare R2 is supported as a
legacy backend during the migration: if no ``HIPPIUS_*`` env vars are set the
client falls back to the historical ``R2_*`` env vars and the per-account R2
endpoint URL.

The class name ``R2AuditLog`` is preserved as a stable alias so the rest of the
codebase doesn't need a sweeping rename. New code should prefer
``HippiusStorage`` (same class, friendlier name).
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


# Hippius S3 endpoint (Substrate-backed decentralized storage).
# See https://docs.hippius.com/storage/s3/integration for details.
HIPPIUS_DEFAULT_ENDPOINT = "https://s3.hippius.com"
HIPPIUS_DEFAULT_REGION = "decentralized"


def _first_set(*values: str) -> str:
    """Return the first truthy value, or ``""``."""
    for v in values:
        if v:
            return v
    return ""


class HippiusStorage:
    """S3-compatible artifact storage (Hippius primary, R2 legacy fallback)."""

    def __init__(
        self,
        access_key_id: str = "",
        secret_access_key: str = "",
        bucket: str = "",
        endpoint_url: str = "",
        region: str = "",
        # Legacy: R2 used a per-account endpoint derived from account_id.
        # Kept for backwards compatibility with callers that still pass it.
        account_id: str = "",
    ):
        self.bucket = bucket or _first_set(
            os.getenv("HIPPIUS_BUCKET", ""),
            os.getenv("R2_BUCKET", ""),
        ) or "radar-experiments"

        access_key_id = access_key_id or _first_set(
            os.getenv("HIPPIUS_ACCESS_KEY_ID", ""),
            os.getenv("R2_ACCESS_KEY_ID", ""),
        )
        secret_access_key = secret_access_key or _first_set(
            os.getenv("HIPPIUS_SECRET_ACCESS_KEY", ""),
            os.getenv("R2_SECRET_ACCESS_KEY", ""),
        )
        region = region or _first_set(
            os.getenv("HIPPIUS_REGION", ""),
            os.getenv("R2_REGION", ""),
            HIPPIUS_DEFAULT_REGION,
        )

        # Endpoint resolution order:
        #   1. Explicit constructor arg
        #   2. MOCK_S3_ENDPOINT / MOCK_R2_ENDPOINT (test override)
        #   3. HIPPIUS_ENDPOINT_URL (production override / private gateway)
        #   4. Legacy R2 per-account endpoint (only if R2_ACCOUNT_ID is set
        #      and no Hippius endpoint is otherwise configured)
        #   5. HIPPIUS_DEFAULT_ENDPOINT (https://s3.hippius.com)
        mock_endpoint = _first_set(
            os.getenv("MOCK_S3_ENDPOINT", ""),
            os.getenv("MOCK_R2_ENDPOINT", ""),
        )
        legacy_account_id = account_id or os.getenv("R2_ACCOUNT_ID", "")
        if endpoint_url:
            resolved_endpoint = endpoint_url
        elif mock_endpoint:
            resolved_endpoint = mock_endpoint
        elif os.getenv("HIPPIUS_ENDPOINT_URL", ""):
            resolved_endpoint = os.getenv("HIPPIUS_ENDPOINT_URL", "")
        elif legacy_account_id and not (
            os.getenv("HIPPIUS_ACCESS_KEY_ID")
            or os.getenv("HIPPIUS_BUCKET")
        ):
            # Only fall back to R2's per-account endpoint when the operator
            # has actively configured R2 (set R2_ACCOUNT_ID) and not begun
            # the Hippius migration. Removes the "I set HIPPIUS_* but it
            # still hits R2" footgun.
            resolved_endpoint = f"https://{legacy_account_id}.r2.cloudflarestorage.com"
        else:
            resolved_endpoint = HIPPIUS_DEFAULT_ENDPOINT

        import boto3
        from botocore.config import Config as BotoConfig

        # Hippius requires path-style addressing (it does not host buckets at
        # virtual subdomains the way AWS does). R2 also accepts path-style,
        # so this is safe for the legacy backend too.
        self._s3 = boto3.client(
            "s3",
            endpoint_url=resolved_endpoint,
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            region_name=region,
            config=BotoConfig(
                signature_version="s3v4",
                s3={"addressing_style": "path"},
            ),
        )

    def upload_json(self, key: str, data: dict) -> bool:
        """Upload a JSON object."""
        try:
            body = json.dumps(data, indent=2).encode()
            self._s3.put_object(Bucket=self.bucket, Key=key, Body=body)
            return True
        except Exception as e:
            logger.error("Failed to upload JSON %s: %s", key, e)
            return False

    def download_json(self, key: str) -> Optional[dict]:
        """Download and parse a JSON object."""
        try:
            resp = self._s3.get_object(Bucket=self.bucket, Key=key)
            return json.loads(resp["Body"].read())
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                logger.debug("Key not found: %s", key)
            else:
                logger.error("Failed to download JSON %s: %s", key, e)
            return None
        except Exception as e:
            logger.error("Failed to download JSON %s: %s", key, e)
            return None

    def list_experiments(self, prefix: str = "") -> list[str]:
        """List all experiment keys matching prefix."""
        try:
            resp = self._s3.list_objects_v2(Bucket=self.bucket, Prefix=prefix)
            keys = [obj["Key"] for obj in resp.get("Contents", [])]
            experiments = set()
            for key in keys:
                parts = key.split("/")
                if len(parts) >= 2:
                    experiments.add(parts[0])
            return sorted(experiments)
        except Exception as e:
            logger.error("Failed to list experiments with prefix %s: %s", prefix, e)
            return []

    def list_keys(self, prefix: str = "") -> list[str]:
        """List all keys matching prefix."""
        try:
            resp = self._s3.list_objects_v2(Bucket=self.bucket, Prefix=prefix)
            return [obj["Key"] for obj in resp.get("Contents", [])]
        except Exception as e:
            logger.error("Failed to list keys with prefix %s: %s", prefix, e)
            return []

    def upload_file_from_disk(self, local_path: str, key: str) -> bool:
        """Upload a file from local disk."""
        try:
            self._s3.upload_file(local_path, self.bucket, key)
            logger.info("Uploaded file: %s", key)
            return True
        except Exception as e:
            logger.error("Failed to upload file %s: %s", key, e)
            return False

    def download_file_to_disk(self, key: str, local_path: str) -> bool:
        """Download a file to local disk."""
        try:
            resp = self._s3.get_object(Bucket=self.bucket, Key=key)
            with open(local_path, "wb") as f:
                f.write(resp["Body"].read())
            return True
        except Exception as e:
            logger.error("Failed to download file %s: %s", key, e)
            return False

    def upload_text(self, key: str, text: str) -> bool:
        """Upload a text string."""
        try:
            self._s3.put_object(Bucket=self.bucket, Key=key, Body=text.encode())
            return True
        except Exception as e:
            logger.error("Failed to upload text %s: %s", key, e)
            return False

    def download_text(self, key: str) -> Optional[str]:
        """Download a text string."""
        try:
            resp = self._s3.get_object(Bucket=self.bucket, Key=key)
            return resp["Body"].read().decode()
        except Exception as e:
            logger.error("Failed to download text %s: %s", key, e)
            return None

    def generate_presigned_put_url(
        self,
        key: str,
        ttl: int = 3600,
        max_content_length: int = 0,
    ) -> str:
        """Generate a pre-signed PUT URL for uploading to a specific key."""
        try:
            params: dict = {"Bucket": self.bucket, "Key": key}
            if max_content_length > 0:
                params["ContentLength"] = max_content_length
            url = self._s3.generate_presigned_url(
                "put_object",
                Params=params,
                ExpiresIn=ttl,
            )
            return url
        except Exception as e:
            logger.error("Failed to generate presigned PUT URL for %s: %s", key, e)
            return ""

    def generate_presigned_get_url(self, key: str, ttl: int = 900) -> str:
        """Generate a presigned GET URL for downloading."""
        try:
            return self._s3.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket, "Key": key},
                ExpiresIn=ttl,
            )
        except Exception as e:
            logger.error("Failed to generate presigned GET URL for %s: %s", key, e)
            return ""

    def key_exists(self, key: str) -> bool:
        """Check if a key exists in the bucket."""
        try:
            self._s3.head_object(Bucket=self.bucket, Key=key)
            return True
        except Exception:
            return False


# Backwards-compatible alias. Existing callers import ``R2AuditLog``; we keep
# the name so the migration doesn't fan out across every module. New code
# should prefer ``HippiusStorage``.
R2AuditLog = HippiusStorage


__all__ = ["HippiusStorage", "R2AuditLog"]
